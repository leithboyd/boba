"""Column specification for the raw dataset: template + args → concrete columns.

A dataset is defined by an ordered tuple of :class:`ColumnSpec`. Each spec names
a column template (the exact template strings documented in docs/raw_features.md)
and binds the template's placeholders through ``args`` — a sequence of mappings,
each of which must bind exactly the template's placeholders.

A scalar value binds its placeholder directly; a list/tuple value fans out. As
soon as any value in a mapping is a list, every value is treated as a (possibly
length-1) list and the mapping expands to the cross product, staged in template
placeholder order:

    ColumnSpec(
        "{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms",
        args=({"LISTING1": "bin_eth_usdt", "LISTING2": "byb_eth_usdt", "N": [10, 20]},),
    )
    → bin_eth_usdt_byb_eth_usdt_ema_log_microprice_ratio_10ms
      bin_eth_usdt_byb_eth_usdt_ema_log_microprice_ratio_20ms

Output column order is exactly the expansion order: spec order, then arg-mapping
order within a spec, then cross-product order within a mapping.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Optional, Sequence


# ── Template registry ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Template:
    template: str
    family: str                      # builder dispatch key
    placeholders: tuple[str, ...]    # in template order

    @property
    def cross(self) -> bool:
        return "LISTING1" in self.placeholders

    @property
    def has_n(self) -> bool:
        return "N" in self.placeholders


def _t(template: str, family: str) -> Template:
    ph = tuple(f for _, f, _, _ in string.Formatter().parse(template) if f)
    return Template(template, family, ph)


TEMPLATES: dict[str, Template] = {t.template: t for t in (
    # ── Raw 1ms grid state (forward-filled instantaneous BBO state) ──
    _t("{LISTING}_microprice",                          "microprice"),
    _t("{LISTING}_spread_width",                        "spread_width"),
    _t("{LISTING}_book_depth",                          "book_depth"),
    _t("{LISTING}_book_imbalance",                      "book_imbalance"),
    _t("{LISTING}_spread_wide_flag",                    "spread_wide_flag"),
    _t("{LISTING}_time_since_last_trade_ms",            "time_since_last_trade_ms"),
    _t("{LISTING}_time_since_spread_wide_ms",           "time_since_spread_wide_ms"),
    _t("{LISTING}_feed_latency_excess_ms",              "feed_latency_excess_ms"),
    # per-ms trade-flow sums (0 on quiet ms — flow, not state)
    _t("{LISTING}_buy_trade_value",                     "buy_trade_value"),
    _t("{LISTING}_sell_trade_value",                    "sell_trade_value"),
    # event-count windows
    _t("{LISTING}_dt_{N}b",                             "dt_b"),
    _t("{LISTING}_dt_{N}t",                             "dt_t"),
    # dt over the last N MID moves (book_mid=(bid+ask)/2 changes) — the real-move event clock /
    # subordination ladder (rate = N/dt). Distinct from dt_b (all book events incl. size-only).
    _t("{LISTING}_dt_{N}m",                             "dt_m"),
    # trailing wall-clock trade count over (t - N ms, t] — causal; the physical lead-lag signal
    # (fine N = a venue "just fired"). Per-listing; selecting all listings gives the cross-venue view.
    _t("{LISTING}_trade_count_{N}ms",                  "trade_count_ms"),
    # ── Temporal 1ms grid features ──
    _t("{LISTING}_return_{N}ms",                        "return_ms"),
    _t("{LISTING}_ema_microprice_centered_{N}ms",       "ema_microprice_centered_ms"),
    _t("{LISTING}_ema_microprice_centered_sq_{N}ms",    "ema_microprice_centered_sq_ms"),
    # Return realized-variance EMA (RiskMetrics-style RV rate): EWMA of squared
    # 1ms log-microprice returns. THE correct forward-move vol normalizer —
    # r = sqrt(ema_microprice_return_sq)·1e4. Drift-immune (differences before
    # squaring), dimensionless. Unlike ema_microprice_centered_sq (variance of
    # the price LEVEL, which absorbs trend), this is variance of the RETURN.
    _t("{LISTING}_ema_microprice_return_sq_{N}ms",      "ema_microprice_return_sq_ms"),
    _t("{LISTING}_ema_buy_trade_value_{N}ms",           "ema_buy_trade_value_ms"),
    _t("{LISTING}_ema_sell_trade_value_{N}ms",          "ema_sell_trade_value_ms"),
    # ── Microstructure EMAs (event clock) ──
    _t("{LISTING}_ema_buy_trade_qty_{N}t",              "ema_buy_trade_qty_t"),
    _t("{LISTING}_ema_sell_trade_qty_{N}t",             "ema_sell_trade_qty_t"),
    _t("{LISTING}_ema_buy_trade_value_{N}t",            "ema_buy_trade_value_t"),
    _t("{LISTING}_ema_sell_trade_value_{N}t",           "ema_sell_trade_value_t"),
    _t("{LISTING}_ema_trade_serial_cov_{N}t",           "ema_trade_serial_cov_t"),
    _t("{LISTING}_ema_ofi_{N}b",                        "ema_ofi_b"),
    _t("{LISTING}_ema_ofi_sq_{N}b",                     "ema_ofi_sq_b"),
    _t("{LISTING}_ema_abs_log_ret_{N}b",                "ema_abs_log_ret_b"),
    _t("{LISTING}_ema_book_imbalance_{N}b",             "ema_book_imbalance_b"),
    _t("{LISTING}_ema_book_imbalance_sq_{N}b",          "ema_book_imbalance_sq_b"),
    _t("{LISTING}_ema_book_depth_{N}b",                 "ema_book_depth_b"),
    _t("{LISTING}_ema_book_depth_sq_{N}b",              "ema_book_depth_sq_b"),
    _t("{LISTING}_ema_spread_wide_flag_{N}b",           "ema_spread_wide_flag_b"),
    # ── Cross-listing (one column per ordered pair; ema(log a/b) = −ema(log b/a)) ──
    _t("{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms",    "cross_ratio_ms"),
    _t("{LISTING1}_{LISTING2}_ema_log_microprice_ratio_sq_{N}ms", "cross_ratio_sq_ms"),
)}


# ── Specs ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColumnSpec:
    """One template plus the bindings that instantiate it (see module docstring)."""
    template: str
    args: Sequence[Mapping[str, Any]]


def col(template: str, **kwargs: Any) -> ColumnSpec:
    """Sugar for the common single-mapping spec: col("{LISTING}_ema_ofi_{N}b",
    LISTING=["bin", "byb"], N=[3, 10]) ≡ ColumnSpec(template, ({...},))."""
    return ColumnSpec(template, ({**kwargs},))


@dataclass(frozen=True)
class ColumnUnit:
    """One concrete column: a template fully bound to scalar values."""
    name: str                            # full column name
    family: str                          # Template.family
    listing: Optional[str]               # per-listing families
    local_name: Optional[str]            # name without the "{listing}_" prefix
    pair: Optional[tuple[str, str]]      # cross families: (LISTING1, LISTING2)
    n: Optional[int]


@dataclass(frozen=True)
class ExpandedColumns:
    names: tuple[str, ...]
    units: tuple[ColumnUnit, ...]


# ── Expansion ─────────────────────────────────────────────────────────────────

def _expand_one(t: Template, m: Mapping[str, Any], where: str) -> list[dict[str, Any]]:
    """Expand one arg mapping to scalar bindings (cross product over list values)."""
    given, want = set(m), set(t.placeholders)
    if given != want:
        missing, extra = sorted(want - given), sorted(given - want)
        raise ValueError(
            f"{where}: args must bind exactly {sorted(want)} for template "
            f"{t.template!r}" + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else ""))
    staged: list[list[Any]] = []
    for p in t.placeholders:
        v = m[p]
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        if not vals:
            raise ValueError(f"{where}: empty value list for {p!r}")
        staged.append(vals)
    return [dict(zip(t.placeholders, combo)) for combo in product(*staged)]


def _validate_binding(t: Template, b: dict[str, Any], listings: Optional[Sequence[str]], where: str) -> None:
    for p in t.placeholders:
        v = b[p]
        if p == "N":
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{where}: N must be a positive int, got {v!r}")
        else:
            if not isinstance(v, str) or not v:
                raise ValueError(f"{where}: {p} must be a non-empty listing token, got {v!r}")
            if listings is not None and v not in listings:
                raise ValueError(f"{where}: {p}={v!r} is not one of the listings {sorted(listings)}")
    if t.cross and b["LISTING1"] == b["LISTING2"]:
        raise ValueError(f"{where}: LISTING1 and LISTING2 must differ, both are {b['LISTING1']!r}")


def _unit(t: Template, b: dict[str, Any]) -> ColumnUnit:
    name = t.template.format(**b)
    if t.cross:
        return ColumnUnit(name=name, family=t.family, listing=None, local_name=None,
                          pair=(b["LISTING1"], b["LISTING2"]), n=b.get("N"))
    local = t.template[len("{LISTING}_"):].format(**b)
    return ColumnUnit(name=name, family=t.family, listing=b["LISTING"],
                      local_name=local, pair=None, n=b.get("N"))


def expand_columns(
    specs: Sequence[ColumnSpec],
    listings: Optional[Sequence[str]] = None,
) -> ExpandedColumns:
    """Expand specs to concrete columns, in order. When ``listings`` is given,
    every bound listing token must be a member. Duplicate names raise."""
    if not specs:
        raise ValueError("at least one ColumnSpec is required — `columns` defines the dataset")
    units: list[ColumnUnit] = []
    for si, spec in enumerate(specs):
        t = TEMPLATES.get(spec.template)
        if t is None:
            raise ValueError(
                f"columns[{si}]: unknown template {spec.template!r} — "
                f"see boba.dataset.columns.TEMPLATES / docs/raw_features.md")
        if not spec.args:
            raise ValueError(f"columns[{si}] ({t.template}): args is empty — nothing to expand")
        for ai, m in enumerate(spec.args):
            where = f"columns[{si}].args[{ai}] ({t.template})"
            for b in _expand_one(t, m, where):
                _validate_binding(t, b, listings, where)
                units.append(_unit(t, b))
    names = [u.name for u in units]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate columns after expansion: {dupes[:10]}")
    return ExpandedColumns(names=tuple(names), units=tuple(units))


def listing_spans(units: Sequence[ColumnUnit], listing: str) -> dict[str, list[int]]:
    """Per-family sorted span lists for one listing's N-parameterized units."""
    out: dict[str, set[int]] = {}
    for u in units:
        if u.listing == listing and u.n is not None:
            out.setdefault(u.family, set()).add(u.n)
    return {k: sorted(v) for k, v in out.items()}
