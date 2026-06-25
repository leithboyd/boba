"""Span & head SELECTION engines — the feature-agnostic step that follows screening.

Screening (`boba.research.screening`) answers *does this feature pass the gates*. Once it does, two
questions remain, and they are what this module answers — generically, for any feature, from a
`ScreeningContext` plus a built family `{params -> {source -> vector}}` (`screening.build_family`):

  Step 2a  WHICH span(s)?   The in-sample IC breakdown across the fast/slow span family — per source,
           per head — that tells you where the signal lives (`ic_grid`); the count-conditioned price
           targets the price-head grids score against (`fixed_move_targets`); and whether a SECOND
           span adds anything walk-forward over the in-sample pick (`second_span_adds`).
  Step 2b  per-exchange WORTH IT?   The two out-of-sample ICs — all source legs kept per-exchange vs the
           single best source (`per_exchange_vs_single`): does keeping every source's leg add over one?
           (We keep per-source instances or collapse to one source; we never merge sources into an
           averaged value — there is no "pooling".)

Ported from the monolith template's §3 (`fixed_move_target`), §7 (the IC grids, the `show_grids`
plotter — LEFT to the notebook — and the "does a 2nd span add" conditional partial-IC + walk-forward
join), and §9 (the per-exchange-vs-single comparison — made REAL, with no feature pooling).

Every inner statistic is `boba.research.gates` (`ic` / `wf_ic`); the only hand-composed number is a
partial-IC, built from `gates.ic` on a COMMON finite mask exactly as `screening._partial_ic` is — see
that helper's docstring. Nothing price_dislocation-specific: a function takes `(ctx, family, ...)`
where `family` is `{params -> {source -> vector}}` and a head is a plain target vector (or, for the
conditional, the chosen params token), so the same code selects spans for any feature.
"""
from __future__ import annotations

import math

import numpy as np

from boba.features.base import Params
from boba.research.screening import ScreeningContext, _ffill


# --------------------------------------------------------------------------------------------------
# §3 — the count-conditioned price target.
# --------------------------------------------------------------------------------------------------
def fixed_move_targets(
    ctx: ScreeningContext,
    counts: tuple[int, ...] = (1, 3, 6, 9),
) -> dict[int, np.ndarray]:
    """The price-head targets conditioned on a FIXED number of future target moves, per count `n`.

    For each `n`, the target is the SIGNED return from the anchor's current target mid to the target's
    `n`-th future mid-MOVE, divided by `σ_ev` at the anchor (the volatility yardstick) — `nan` once the
    block runs out of future moves. This is the count-conditioned twin of `ctx.price_target` (which is
    the 100 ms forward return / σ_ev): it isolates the per-move direction the price head's `D_k` family
    learns, instead of a wall-clock window. Causal: the `n`-th move strictly AFTER the anchor.

    Ports the monolith §3 `fixed_move_target` exactly: `nth = searchsorted(move_rx, anchor, "right") +
    n - 1`, return = `log(move_mid[nth]) - log(mid_now)`, scaled by `1/σ_ev`. Built from the context's
    target-move stream (`ctx._mv_rx`) and raw target mids (`ctx._mids`) + `ctx.anchor_ts`.
    """
    target_ex = ctx.target.split("_", 1)[0]
    rx, mid = ctx._mids[target_ex]                       # raw target mid stream
    move_rx = ctx._mv_rx                                 # timestamps where the target mid actually moved
    move_lm = np.log(_ffill(rx, mid, move_rx))           # log target mid AT each move (last raw mid <= move_rx)
    mid_now = _ffill(rx, mid, ctx.anchor_ts)             # target mid at-or-before each anchor (causal)
    log_now = np.log(mid_now)
    sigma = ctx.sigma_at_anchor
    n_anchor = len(ctx.anchor_ts)

    out: dict[int, np.ndarray] = {}
    for n in counts:
        nth = np.searchsorted(move_rx, ctx.anchor_ts, "right") + n - 1   # the n-th future move after the anchor
        ok = nth < len(move_lm)
        ret_n = np.full(n_anchor, np.nan)
        ret_n[ok] = move_lm[nth[ok]] - log_now[ok]
        out[n] = ret_n / sigma
    return out


# --------------------------------------------------------------------------------------------------
# §7 — the in-sample IC grid across the span family.
# --------------------------------------------------------------------------------------------------
def _fast_slow_axes(family: dict[Params, dict[str, np.ndarray]]) -> tuple[list, list]:
    """Sorted unique fast and slow spans from the family's `(n_fast, n_slow)` param keys — the grid
    axes. Assumes the opaque params token is a `(fast, slow)` pair (the span-family convention)."""
    fasts = sorted({p[0] for p in family})
    slows = sorted({p[1] for p in family})
    return fasts, slows


def ic_grid(
    ctx: ScreeningContext,
    family: dict[Params, dict[str, np.ndarray]],
    target: np.ndarray,
    *,
    magnitude: bool = False,
) -> dict[str, np.ndarray]:
    """In-sample masked rank-IC of every family member vs `target`, as a `[fast, slow]` grid per source.

    `family` is `{(n_fast, n_slow) -> {source -> vector}}` from `screening.build_family`; the fast and
    slow axes are the sorted unique spans in those keys. Cell `[i, j]` is `gates.ic(member, target)` (or
    `gates.ic(|member|, target)` when `magnitude`) for the `(fasts[i], slows[j])` member of that source;
    cells with `n_fast >= n_slow` (no valid pair) are left `nan`. The IC is the tested masked Spearman
    (drops non-finite rows) — the same diagnostic the monolith §7 grids show. The grid's `nanargmax` is
    the in-sample span pick (used ONLY to pick a time-scale, never as an OOS claim).
    """
    from boba.research import gates as g

    fasts, slows = _fast_slow_axes(family)
    sources = sorted({s for legs in family.values() for s in legs})
    grids = {s: np.full((len(fasts), len(slows)), np.nan) for s in sources}
    fi = {f: i for i, f in enumerate(fasts)}
    sj = {s: j for j, s in enumerate(slows)}
    for (nf, ns), legs in family.items():
        if nf >= ns:                                     # no valid fast<slow pair -> leave nan
            continue
        i, j = fi[nf], sj[ns]
        for src, vec in legs.items():
            score = np.abs(vec) if magnitude else vec
            grids[src][i, j] = g.ic(score, target)
    return grids


# --------------------------------------------------------------------------------------------------
# §7 — partial-IC (controls for the chosen span). Composed from gates.ic on a COMMON mask, exactly
# like screening._partial_ic — the one allowed hand-composition (a partial of the tested masked IC).
# --------------------------------------------------------------------------------------------------
def _partial_ic(f: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
    """Partial rank-IC of `f` with `y` controlling for `c` — `gates.ic` on the common-finite subset,
    combined with the standard partial-correlation formula (identical to `screening._partial_ic`)."""
    from boba.research import gates as g

    v = np.isfinite(f) & np.isfinite(y) & np.isfinite(c)
    if v.sum() <= 100:
        return float("nan")
    rfy, rfc, rcy = g.ic(f[v], y[v]), g.ic(f[v], c[v]), g.ic(c[v], y[v])
    return (rfy - rfc * rcy) / math.sqrt(max((1.0 - rfc ** 2) * (1.0 - rcy ** 2), 1e-12))


def second_span_adds(
    ctx: ScreeningContext,
    family: dict[Params, dict[str, np.ndarray]],
    chosen_params: Params,
    target: np.ndarray,
    *,
    magnitude: bool = False,
) -> dict[str, dict]:
    """Does a SECOND span add over the in-sample pick `chosen_params`, per source? (monolith §7.)

    Two-stage, exactly the monolith `_conditional_member`:
      1. IN-SAMPLE SCREEN — re-score the whole family as the partial-IC of each member against `target`
         CONTROLLING for the chosen member (`_partial_ic`, `gates.ic` on a common mask). The chosen cell
         scores `0`; the most-orthogonal alternative is the `nanargmax` of `|partial-IC|` — the span that
         carries the most signal the pick does NOT already explain.
      2. WALK-FORWARD DECISION — `wf_ic([chosen])` (solo) vs `wf_ic([chosen, best_alt])` (joint), both
         OOS via `gates.wf_ic`; `keep = (joint - solo) >= 0.01` — the OOS joint gain decides, not the
         in-sample partial.

    `magnitude` scores `|member|` / `|target|` (the rate-head / magnitude diagnostics). Per source returns
    `dict(best_alt, cond_ic, oos_solo, oos_joint, keep)` — `best_alt` the chosen alternative params token,
    `cond_ic` its in-sample partial-IC given the pick, `oos_solo`/`oos_joint` the walk-forward ICs.
    """
    from boba.research import gates as g

    fasts, slows = _fast_slow_axes(family)
    sources = sorted({s for legs in family.values() for s in legs})
    ci, cj = chosen_params
    tgt = np.abs(target) if magnitude else target

    def member(src: str, params: Params) -> np.ndarray:
        vec = family[params][src]
        return np.abs(vec) if magnitude else vec

    out: dict[str, dict] = {}
    for src in sources:
        chosen = member(src, chosen_params)
        cond = np.full((len(fasts), len(slows)), np.nan)
        cell_params = {}
        for (nf, ns), legs in family.items():
            if nf >= ns or src not in legs:
                continue
            i, j = fasts.index(nf), slows.index(ns)
            cell_params[(i, j)] = (nf, ns)
            cond[i, j] = 0.0 if (nf, ns) == (ci, cj) else _partial_ic(member(src, (nf, ns)), tgt, chosen)
        bi, bj = np.unravel_index(np.nanargmax(np.abs(cond)), cond.shape)   # the most-orthogonal alternative
        best_alt = cell_params[(bi, bj)]
        alt = member(src, best_alt)
        solo = g.wf_ic([chosen], tgt)
        joint = g.wf_ic([chosen, alt], tgt)
        out[src] = dict(
            best_alt=best_alt,
            cond_ic=float(cond[bi, bj]),
            oos_solo=solo,
            oos_joint=joint,
            keep=bool(np.isfinite(joint) and np.isfinite(solo) and (joint - solo) >= 0.01),
        )
    return out


# --------------------------------------------------------------------------------------------------
# §9 — single exchange vs per-exchange. We keep per-source instances (or collapse to one source); we
# never merge sources into one averaged value, so there is no "pooling".
# --------------------------------------------------------------------------------------------------
def per_exchange_vs_single(
    ctx: ScreeningContext,
    family: dict[Params, dict[str, np.ndarray]],
    params: Params,
    target: np.ndarray,
) -> dict:
    """Does keeping EVERY source's leg (per-exchange) add over the single best source, out-of-sample?

    For a feature that fans out into one instance per source, the choice is **single exchange** (keep one
    leg) vs **per-exchange** (keep them all and let the model weight whichever leads). For one fixed
    `params` (one span), the family's `{source -> leg}` legs are scored two ways against `target`, both
    walk-forward via `gates.wf_ic`:
      per_exchange -- the legs kept SEPARATE, scored JOINTLY (`wf_ic(list(legs), target)`).
      best_single  -- the single best source's leg scored alone (`max` over `wf_ic([leg], target)`), with
                      which source won.
    Returns `dict(per_exchange, best_single=dict(ic, source), adds_over_single)`, where `adds_over_single`
    is whether per-exchange beats the best single leg by a meaningful margin (>= 0.01) — the "the sources
    genuinely differ, keep them all" signal. Runs on any fanned-out family.
    """
    from boba.research import gates as g

    legs = family[params]
    keys = sorted(legs)
    leg_list = [legs[k] for k in keys]

    per_exchange = g.wf_ic(leg_list, target)
    singles = {k: g.wf_ic([legs[k]], target) for k in keys}
    best_key = max(keys, key=lambda k: (singles[k] if np.isfinite(singles[k]) else -np.inf))
    best_single = dict(ic=singles[best_key], source=best_key)
    adds = bool(np.isfinite(per_exchange) and np.isfinite(best_single["ic"])
                and (per_exchange - best_single["ic"]) >= 0.01)
    return dict(per_exchange=per_exchange, best_single=best_single, adds_over_single=adds)
