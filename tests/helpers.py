"""Test helpers: legacy-equivalent column catalogues for the spec-driven API.

The old API selected features via per-family span tuples on the config and a
canonical catalogue order. These helpers rebuild that exact catalogue (same
names, same order) as explicit ColumnSpecs, so tests written against the old
full-catalogue behaviour keep their coverage under the columns-required API.
"""
from __future__ import annotations

import re
from itertools import combinations

from boba.dataset.columns import ColumnSpec, col
from boba.dataset.raw import DatasetRawConfig, _COST_FIELDS

# Legacy default spans (the old DatasetRawConfig defaults)
DT_B_SPANS = (10, 100, 1000, 10000)
DT_T_SPANS = (3, 10, 100, 1000)
RETURN_WINDOWS_MS = (1, 5, 20, 100, 1000, 5000)
EMA_MICROPRICE_MS_SPANS = (100, 200, 500, 1000, 10000, 60000)
EMA_TRADE_VALUE_MS_SPANS = (10, 25, 50, 100)
EMA_TRADE_SPANS = (3, 10, 50, 100, 1000)
EMA_TRADE_SERIAL_COV_SPANS = (100, 1000)
EMA_OFI_SPANS = (3, 10, 100, 1000)
EMA_OFI_SQ_SPANS = (1000, 10000)
EMA_ABS_LOG_RET_SPANS = (100, 1000, 10000)
EMA_BOOK_IMBALANCE_SPANS = (10, 100, 1000, 10000)
EMA_BOOK_IMBALANCE_SQ_SPANS = (1000, 10000)
EMA_BOOK_DEPTH_SPANS = (10, 100, 1000, 10000)
EMA_BOOK_DEPTH_SQ_SPANS = (1000, 10000)
EMA_SPREAD_WIDE_FLAG_SPANS = (100, 1000)

# Spans mappings for the compute_* unit-test surface (legacy defaults)
DEFAULT_BBO_SPANS = {
    "ema_ofi_b": list(EMA_OFI_SPANS),
    "ema_ofi_sq_b": list(EMA_OFI_SQ_SPANS),
    "ema_abs_log_ret_b": list(EMA_ABS_LOG_RET_SPANS),
    "ema_book_imbalance_b": list(EMA_BOOK_IMBALANCE_SPANS),
    "ema_book_imbalance_sq_b": list(EMA_BOOK_IMBALANCE_SQ_SPANS),
    "ema_book_depth_b": list(EMA_BOOK_DEPTH_SPANS),
    "ema_book_depth_sq_b": list(EMA_BOOK_DEPTH_SQ_SPANS),
    "ema_spread_wide_flag_b": list(EMA_SPREAD_WIDE_FLAG_SPANS),
    "dt_b": list(DT_B_SPANS),
}
DEFAULT_TRADE_SPANS = {
    "ema_buy_trade_qty_t": list(EMA_TRADE_SPANS),
    "ema_sell_trade_qty_t": list(EMA_TRADE_SPANS),
    "ema_buy_trade_value_t": list(EMA_TRADE_SPANS),
    "ema_sell_trade_value_t": list(EMA_TRADE_SPANS),
    "ema_trade_serial_cov_t": list(EMA_TRADE_SERIAL_COV_SPANS),
    "dt_t": list(DT_T_SPANS),
}
DEFAULT_TEMPORAL_SPANS = {
    "return_ms": list(RETURN_WINDOWS_MS),
    "ema_microprice_centered_ms": list(EMA_MICROPRICE_MS_SPANS),
    "ema_microprice_centered_sq_ms": list(EMA_MICROPRICE_MS_SPANS),
}


def listing_columns(
    listing: str,
    dt_b_spans=DT_B_SPANS,
    dt_t_spans=DT_T_SPANS,
    return_windows_ms=RETURN_WINDOWS_MS,
    ema_microprice_ms_spans=EMA_MICROPRICE_MS_SPANS,
    ema_trade_value_ms_spans=EMA_TRADE_VALUE_MS_SPANS,
    ema_trade_spans=EMA_TRADE_SPANS,
    ema_trade_serial_cov_spans=EMA_TRADE_SERIAL_COV_SPANS,
    ema_ofi_spans=EMA_OFI_SPANS,
    ema_ofi_sq_spans=EMA_OFI_SQ_SPANS,
    ema_abs_log_ret_spans=EMA_ABS_LOG_RET_SPANS,
    ema_book_imbalance_spans=EMA_BOOK_IMBALANCE_SPANS,
    ema_book_imbalance_sq_spans=EMA_BOOK_IMBALANCE_SQ_SPANS,
    ema_book_depth_spans=EMA_BOOK_DEPTH_SPANS,
    ema_book_depth_sq_spans=EMA_BOOK_DEPTH_SQ_SPANS,
    ema_spread_wide_flag_spans=EMA_SPREAD_WIDE_FLAG_SPANS,
) -> list[ColumnSpec]:
    """One listing's full per-listing catalogue, in the legacy canonical order."""
    L = listing
    specs = [
        col("{LISTING}_microprice", LISTING=L),
        col("{LISTING}_spread_width", LISTING=L),
        col("{LISTING}_book_depth", LISTING=L),
        col("{LISTING}_book_imbalance", LISTING=L),
        col("{LISTING}_spread_wide_flag", LISTING=L),
    ]
    if dt_b_spans:
        specs.append(col("{LISTING}_dt_{N}b", LISTING=L, N=list(dt_b_spans)))
    if dt_t_spans:
        specs.append(col("{LISTING}_dt_{N}t", LISTING=L, N=list(dt_t_spans)))
    specs += [
        col("{LISTING}_time_since_last_trade_ms", LISTING=L),
        col("{LISTING}_time_since_spread_wide_ms", LISTING=L),
        col("{LISTING}_feed_latency_excess_ms", LISTING=L),
        col("{LISTING}_buy_trade_value", LISTING=L),
        col("{LISTING}_sell_trade_value", LISTING=L),
    ]
    if return_windows_ms:
        specs.append(col("{LISTING}_return_{N}ms", LISTING=L, N=list(return_windows_ms)))
    for N in ema_microprice_ms_spans:
        specs += [col("{LISTING}_ema_microprice_centered_{N}ms", LISTING=L, N=N),
                  col("{LISTING}_ema_microprice_centered_sq_{N}ms", LISTING=L, N=N)]
    for N in ema_trade_value_ms_spans:
        specs += [col("{LISTING}_ema_buy_trade_value_{N}ms", LISTING=L, N=N),
                  col("{LISTING}_ema_sell_trade_value_{N}ms", LISTING=L, N=N)]
    for N in ema_trade_spans:
        specs += [col("{LISTING}_ema_buy_trade_qty_{N}t", LISTING=L, N=N),
                  col("{LISTING}_ema_sell_trade_qty_{N}t", LISTING=L, N=N),
                  col("{LISTING}_ema_buy_trade_value_{N}t", LISTING=L, N=N),
                  col("{LISTING}_ema_sell_trade_value_{N}t", LISTING=L, N=N)]
    if ema_trade_serial_cov_spans:
        specs.append(col("{LISTING}_ema_trade_serial_cov_{N}t", LISTING=L, N=list(ema_trade_serial_cov_spans)))
    if ema_ofi_spans:
        specs.append(col("{LISTING}_ema_ofi_{N}b", LISTING=L, N=list(ema_ofi_spans)))
    if ema_ofi_sq_spans:
        specs.append(col("{LISTING}_ema_ofi_sq_{N}b", LISTING=L, N=list(ema_ofi_sq_spans)))
    if ema_abs_log_ret_spans:
        specs.append(col("{LISTING}_ema_abs_log_ret_{N}b", LISTING=L, N=list(ema_abs_log_ret_spans)))
    if ema_book_imbalance_spans:
        specs.append(col("{LISTING}_ema_book_imbalance_{N}b", LISTING=L, N=list(ema_book_imbalance_spans)))
    if ema_book_imbalance_sq_spans:
        specs.append(col("{LISTING}_ema_book_imbalance_sq_{N}b", LISTING=L, N=list(ema_book_imbalance_sq_spans)))
    if ema_book_depth_spans:
        specs.append(col("{LISTING}_ema_book_depth_{N}b", LISTING=L, N=list(ema_book_depth_spans)))
    if ema_book_depth_sq_spans:
        specs.append(col("{LISTING}_ema_book_depth_sq_{N}b", LISTING=L, N=list(ema_book_depth_sq_spans)))
    if ema_spread_wide_flag_spans:
        specs.append(col("{LISTING}_ema_spread_wide_flag_{N}b", LISTING=L, N=list(ema_spread_wide_flag_spans)))
    return specs


def legacy_columns(listings, cross: bool = False, **span_overrides) -> tuple[ColumnSpec, ...]:
    """Full legacy catalogue: sorted listings × per-listing block, plus (when
    ``cross``) the alphabetical-pair cross block, in the legacy order."""
    specs: list[ColumnSpec] = []
    for L in sorted(listings):
        specs += listing_columns(L, **span_overrides)
    if cross:
        mp_spans = span_overrides.get("ema_microprice_ms_spans", EMA_MICROPRICE_MS_SPANS)
        for l1, l2 in combinations(sorted(listings), 2):
            for N in mp_spans:
                specs += [
                    col("{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms",
                        LISTING1=l1, LISTING2=l2, N=N),
                    col("{LISTING1}_{LISTING2}_ema_log_microprice_ratio_sq_{N}ms",
                        LISTING1=l1, LISTING2=l2, N=N),
                ]
    return tuple(specs)


_SPAN_KEYS = {
    "dt_b_spans", "dt_t_spans", "return_windows_ms", "ema_microprice_ms_spans",
    "ema_trade_value_ms_spans", "ema_trade_spans", "ema_trade_serial_cov_spans",
    "ema_ofi_spans", "ema_ofi_sq_spans", "ema_abs_log_ret_spans",
    "ema_book_imbalance_spans", "ema_book_imbalance_sq_spans",
    "ema_book_depth_spans", "ema_book_depth_sq_spans", "ema_spread_wide_flag_spans",
}


def make_cfg(listings=("bin", "byb", "okx"), cross: bool = False, columns=None, **kw) -> DatasetRawConfig:
    """Legacy-equivalent config. Span-tuple kwargs (old config field names) shape
    the catalogue; everything else passes through to DatasetRawConfig. Pass
    ``columns`` explicitly to bypass the catalogue.

    cost_fields defaults to the full catalogue (the old all-cost-fields behaviour) and
    microprice_ref to the old per-listing 0.15 (DatasetRawConfig no longer defaults it),
    so legacy tests keep their behaviour; pass either explicitly to override."""
    spans = {k: kw.pop(k) for k in list(kw) if k in _SPAN_KEYS}
    if columns is None:
        columns = legacy_columns(listings, cross=cross, **spans)
    else:
        assert not spans, "span overrides only apply to the generated catalogue"
    kw.setdefault("cost_fields", _COST_FIELDS)
    kw.setdefault("microprice_ref", {lst: 0.15 for lst in listings})
    return DatasetRawConfig(columns=tuple(columns), listings=tuple(sorted(listings)), **kw)


def legacy_local_names(**span_overrides) -> list[str]:
    """The old feature_names_per_listing(cfg): one listing's local column names
    in legacy canonical order."""
    from boba.dataset.columns import expand_columns
    specs = listing_columns("L", **span_overrides)
    return [u.local_name for u in expand_columns(specs).units]


def cols_from_names(names, listings=("bin", "byb", "okx")) -> tuple[ColumnSpec, ...]:
    """Concrete legacy column names → specs (order preserved). Listing tokens are
    matched longest-prefix-first against ``listings``; the remainder is matched
    against the template catalogue."""
    from boba.dataset.columns import TEMPLATES
    by_len = sorted(listings, key=len, reverse=True)
    specs: list[ColumnSpec] = []
    for name in names:
        spec = None
        # cross templates: two listing prefixes — first (longest-prefix) match wins
        for l1 in by_len:
            if spec is not None:
                break
            if not name.startswith(l1 + "_"):
                continue
            rest1 = name[len(l1) + 1:]
            for l2 in by_len:
                if l2 == l1 or not rest1.startswith(l2 + "_"):
                    continue
                tail = rest1[len(l2) + 1:]
                m = re.fullmatch(r"ema_log_microprice_ratio_(\d+)ms", tail)
                if m:
                    spec = col("{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms",
                               LISTING1=l1, LISTING2=l2, N=int(m.group(1)))
                    break
                m = re.fullmatch(r"ema_log_microprice_ratio_sq_(\d+)ms", tail)
                if m:
                    spec = col("{LISTING1}_{LISTING2}_ema_log_microprice_ratio_sq_{N}ms",
                               LISTING1=l1, LISTING2=l2, N=int(m.group(1)))
                    break
        if spec is None:
            for L in by_len:
                if not name.startswith(L + "_"):
                    continue
                local = name[len(L) + 1:]
                for t in TEMPLATES.values():
                    if t.cross:
                        continue
                    pat = re.escape(t.template[len("{LISTING}_"):]).replace(r"\{N\}", r"(\d+)")
                    m = re.fullmatch(pat, local)
                    if m:
                        kw = {"LISTING": L}
                        if t.has_n:
                            kw["N"] = int(m.group(1))
                        spec = col(t.template, **kw)
                        break
                if spec is not None:
                    break
        if spec is None:
            raise ValueError(f"cannot map legacy name {name!r} to a template")
        specs.append(spec)
    return tuple(specs)
