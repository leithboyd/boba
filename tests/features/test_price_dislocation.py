"""Dedicated, self-contained tests for `boba.features.price_dislocation`.

The validation trio from `AUTHORING.md`, all on a synthetic block this file builds itself, plus a
real-block parity test (skipped without DATA_DIR):

  (a) vectorized-vs-INDEPENDENT-oracle — an explicit per-tick event loop (NO shared code with the
      production build) computing the live-front EMA of the log gap `g = log(mid_src) - log(mid_byb)`
      on the trade clock (committed per trade + a fresh front at the anchor), fast-minus-slow,
      divided by an independent `σ_ev` loop. Modelled on `tests/test_ofi.py::_ofi_leg_oracle`.
  (b) the mirror COMMUTATION invariant `spec.mirror(feature(books)) == feature(mirror_books(books))`
      against the FULL book reflection (reflect every price through byb's mid).
  (c) the per-venue fan-out — one independent leg per source, each equal to that source's solo build.

`span = 1` (α = 1, the most-used fast leg) is swept everywhere and asserted FINITE wherever the
mids exist (`AUTHORING.md`'s span=1 Do-rule), with a consistent NaN where a source has not quoted.
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.price_dislocation  # noqa: F401  (registers the feature)
from boba.research.screening import RawEventStream, ScreeningContext


# --------------------------------------------------------------------------------------------------
# self-contained synthetic context — byb (target) + foreign sources, each a mid path on its own rx
# clock, evaluated on a shared trade clock (`merged_ts`) and an anchor grid.
# --------------------------------------------------------------------------------------------------
def _synthetic_mids(seed=0, n=4000, c=100.0):
    """`{ex -> (rx, mid)}` for byb + two sources, a geometric-random-walk mid each on its own rx clock,
    plus the shared trade clock `merged_ts` and the anchor grid. All quotes start before the first
    trade so every mid is finite (the warm-up NaN path is exercised by `_late_quote_mids`)."""
    rng = np.random.default_rng(seed)
    mids = {}
    for k, ex in enumerate(("byb", "bin", "okx")):
        rx = (np.arange(0, n + 5) * 8 + k).astype(np.int64)               # a quote every ~8 ns, offset per ex
        mid = c * np.exp(np.cumsum(rng.standard_normal(n + 5) * 1e-4))
        mids[ex] = (rx, mid)
    last = min(int(rx[-1]) for rx, _ in mids.values())
    merged_ts = np.unique(np.concatenate([
        (np.arange(1, n + 1) * 8)[::3],                                   # trades coincident-ish with quotes
        (np.arange(1, n + 1) * 8)[1::7] + 2,                             # trade-only timestamps between quotes
    ])).astype(np.int64)
    merged_ts = merged_ts[merged_ts <= last]
    anchor_ts = np.arange(int(merged_ts[60]), last, 37, dtype=np.int64)
    return mids, merged_ts, anchor_ts


def _late_quote_mids(seed=1, n=4000, c=100.0):
    """Same, but `okx` does not quote until well after the warm-up — so its leg is a consistent NaN on
    the early anchors (a source that has not quoted yet) and finite once it does."""
    mids, merged_ts, anchor_ts = _synthetic_mids(seed=seed, n=n, c=c)
    rx, mid = mids["okx"]
    start = anchor_ts[len(anchor_ts) // 2]                                # okx silent until mid-grid
    keep = rx >= start
    mids["okx"] = (rx[keep], mid[keep])
    return mids, merged_ts, anchor_ts


def _ctx(mids, merged_ts, anchor_ts, sigma, target="byb_x", sources=("bin", "okx")):
    ctx = ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.searchsorted(merged_ts, anchor_ts, "right") - 1,
        sigma_at_anchor=sigma, lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0),
        vol_regime=np.empty(0), raw_events=RawEventStream(*([np.empty(0)] * 8), ()),
        _mids=dict(mids),
    )
    ctx.target_logmid_on_clock = np.log(ctx.mid_on_clock(target.split("_", 1)[0]))
    return ctx


# --------------------------------------------------------------------------------------------------
# INDEPENDENT oracle — an explicit one-tick-at-a-time loop, NO shared code with the production build.
# (modelled on tests/test_ofi.py::_ofi_leg_oracle)
# --------------------------------------------------------------------------------------------------
def _ffill_scalar(rx, val, t):
    """The last `val` whose `rx <= t`, or None if none — a dead-simple causal forward-fill at a scalar t."""
    j = int(np.searchsorted(rx, t, "right")) - 1
    return None if j < 0 else float(val[j])


def _dislocation_leg_oracle(src_mids, byb_mids, merged_ts, anchor_ts, sigma, span):
    """The fast/slow live-front leg's NUMERATOR EMA at `span`, by a dead-simple loop.

    Committed EMA on the trade clock of the log gap `g = log(mid_src) - log(mid_byb)` (a missing mid on
    the clock contributes a 0 gap, matching the build's `nan_to_num`), read at each anchor as the live
    front `(1-a)*committed + a*g_fresh` where `g_fresh` is the freshest gap AT the anchor (NaN if either
    side has not quoted there). `span == 1` (a = 1) ⇒ the live front collapses to `g_fresh`."""
    src_rx, src_mid = src_mids
    byb_rx, byb_mid = byb_mids
    a = 2.0 / (span + 1.0)

    # committed EMA over the trade clock (y[-1] = 0), one step per merged_ts tick.
    committed_at_tick = np.empty(len(merged_ts))
    ema = 0.0
    for i, ts in enumerate(merged_ts):
        ms = _ffill_scalar(src_rx, src_mid, ts)
        mb = _ffill_scalar(byb_rx, byb_mid, ts)
        g = 0.0 if (ms is None or mb is None) else (np.log(ms) - np.log(mb))   # nan_to_num(., 0.0)
        ema = (1.0 - a) * ema + a * g
        committed_at_tick[i] = ema

    out = np.full(len(anchor_ts), np.nan)
    for i, t in enumerate(anchor_ts):
        ms = _ffill_scalar(src_rx, src_mid, t)
        mb = _ffill_scalar(byb_rx, byb_mid, t)
        if ms is None or mb is None:
            continue                                                          # no fresh gap -> NaN leg
        g_fresh = np.log(ms) - np.log(mb)
        tick = int(np.searchsorted(merged_ts, t, "right")) - 1
        committed = committed_at_tick[tick] if tick >= 0 else 0.0
        out[i] = (1.0 - a) * committed + a * g_fresh
    return out


def _dislocation_oracle(src_mids, byb_mids, merged_ts, anchor_ts, sigma, n_fast, n_slow):
    """The whole feature for one source: (fast leg - slow leg) / σ_ev, both live-front EMAs of the gap."""
    fast = _dislocation_leg_oracle(src_mids, byb_mids, merged_ts, anchor_ts, sigma, n_fast)
    slow = _dislocation_leg_oracle(src_mids, byb_mids, merged_ts, anchor_ts, sigma, n_slow)
    return (fast - slow) / sigma


SPANS = [(1, 200), (1, 100), (10, 100), (50, 500), (200, 2000)]


def test_vectorized_matches_independent_oracle():
    mids, merged_ts, anchor_ts = _synthetic_mids()
    sigma = np.abs(np.random.default_rng(9).standard_normal(len(anchor_ts))) * 1e-4 + 1e-5
    ctx = _ctx(mids, merged_ts, anchor_ts, sigma)
    spec = base.get("price_dislocation")
    for params in SPANS:
        got = spec.vectorized(ctx, params)
        assert set(got) == {"bin", "okx"}
        for ex in ("bin", "okx"):
            ref = _dislocation_oracle(mids[ex], mids["byb"], merged_ts, anchor_ts, sigma, *params)
            ok = np.isfinite(got[ex]) & np.isfinite(ref)
            assert ok.sum() > 100
            np.testing.assert_allclose(got[ex][ok], ref[ok], rtol=1e-7, atol=1e-9)


def test_span_one_leg_finite_where_mids_exist_and_nan_otherwise():
    """AUTHORING's span=1 Do-rule: at α=1 the live front collapses to the fresh gap, so the value must be
    FINITE wherever both mids exist, and a CONSISTENT NaN exactly where a source has not quoted yet."""
    mids, merged_ts, anchor_ts = _late_quote_mids()
    sigma = np.full(len(anchor_ts), 1e-4)                                     # finite, non-zero everywhere
    ctx = _ctx(mids, merged_ts, anchor_ts, sigma)
    spec = base.get("price_dislocation")

    okx_rx = mids["okx"][0]
    okx_quoted = np.array([np.searchsorted(okx_rx, t, "right") > 0 for t in anchor_ts])
    assert okx_quoted.sum() > 50 and (~okx_quoted).sum() > 50                 # both branches exercised

    for params in [(1, 200), (1, 100)]:                                      # a span=1 fast leg
        out = spec.vectorized(ctx, params)
        # bin quotes from the start -> finite at every anchor (no spurious inf / nan).
        assert np.all(np.isfinite(out["bin"]))
        # okx: finite exactly where it has quoted, NaN exactly where it has not (a consistent NaN).
        assert np.all(np.isfinite(out["okx"][okx_quoted]))
        assert np.all(np.isnan(out["okx"][~okx_quoted]))
        assert not np.any(np.isinf(out["okx"]))


def test_fans_out_one_independent_leg_per_source():
    """One independent leg per foreign source; each equals that source's own solo build (no cross-talk)."""
    mids, merged_ts, anchor_ts = _synthetic_mids(seed=4)
    sigma = np.abs(np.random.default_rng(11).standard_normal(len(anchor_ts))) * 1e-4 + 1e-5
    spec = base.get("price_dislocation")
    params = (10, 100)

    ctx = _ctx(mids, merged_ts, anchor_ts, sigma, sources=("bin", "okx"))
    out = spec.vectorized(ctx, params)
    assert set(out) == {"bin", "okx"}                                        # one leg per foreign source
    for ex in ("bin", "okx"):
        solo_mids = {"byb": mids["byb"], ex: mids[ex]}
        solo_ctx = _ctx(solo_mids, merged_ts, anchor_ts, sigma, sources=(ex,))
        solo = spec.vectorized(solo_ctx, params)[ex]
        np.testing.assert_array_equal(out[ex], solo)                         # leg independent of the other source


# --------------------------------------------------------------------------------------------------
# mirror commutation invariant — reflect EVERY price through byb's mid (a price mirror: prices reflect
# `mid -> c**2/mid`; sizes/clock unchanged). price_dislocation is ODD (log-gaps) -> `mirror` negates.
# --------------------------------------------------------------------------------------------------
def _mirror_mids(mids, c=100.0):
    """Reflect every venue's mid through the fixed price level c (`mid -> c**2/mid`, i.e.
    `log mid -> 2 ln c - log mid`). The exact data-level reflection the feature's `mirror` must commute with."""
    return {ex: (rx, c * c / mid) for ex, (rx, mid) in mids.items()}


def test_mirror_commutes_with_full_book_reflection():
    c = 100.0
    mids, merged_ts, anchor_ts = _synthetic_mids(seed=2, c=c)
    sigma = np.abs(np.random.default_rng(7).standard_normal(len(anchor_ts))) * 1e-4 + 1e-5  # even -> unchanged
    ctx = _ctx(mids, merged_ts, anchor_ts, sigma)
    mctx = _ctx(_mirror_mids(mids, c), merged_ts, anchor_ts, sigma)
    spec = base.get("price_dislocation")
    for params in SPANS:
        feat = spec.vectorized(ctx, params)                                  # feature(books)
        refl = spec.vectorized(mctx, params)                                 # feature(mirror_books(books))
        for ex in ("bin", "okx"):
            lhs = spec.mirror(feat[ex])                                       # mirror(feature(books))
            ok = np.isfinite(lhs) & np.isfinite(refl[ex])
            assert ok.sum() > 100
            np.testing.assert_allclose(lhs[ok], refl[ex][ok], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — the streaming build reproduces the vectorized one.
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_price_dislocation_parity():
    from boba.research.screening import build_context, parity_check

    ctx = build_context(hours=2)
    spec = base.get("price_dislocation")
    rep = parity_check(ctx, spec, [(1, 100), (10, 500)], tol=1e-6)           # a span=1 fast leg in the sweep
    assert rep.passed, str(rep)
