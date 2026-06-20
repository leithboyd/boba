"""Tests for the column-spec API (boba.dataset.columns): template + args expansion."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from boba.dataset.columns import ColumnSpec, TEMPLATES, col, expand_columns, listing_spans
from boba.dataset.raw import DatasetRawConfig, build_features_raw


class TestExpansion:
    def test_scalar_args_one_column(self):
        exp = expand_columns([col("{LISTING}_microprice", LISTING="bin")])
        assert list(exp.names) == ["bin_microprice"]
        u = exp.units[0]
        assert (u.family, u.listing, u.local_name, u.n, u.pair) == (
            "microprice", "bin", "microprice", None, None)

    def test_array_fans_out_in_order(self):
        exp = expand_columns([col("{LISTING}_ema_ofi_{N}b", LISTING="bin", N=[7, 3, 100])])
        # array order is preserved, not sorted
        assert list(exp.names) == ["bin_ema_ofi_7b", "bin_ema_ofi_3b", "bin_ema_ofi_100b"]

    def test_cross_product_staged_in_placeholder_order(self):
        exp = expand_columns([col("{LISTING}_dt_{N}b", LISTING=["bin", "byb"], N=[10, 20])])
        # LISTING appears before N in the template → LISTING is the outer stage
        assert list(exp.names) == ["bin_dt_10b", "bin_dt_20b", "byb_dt_10b", "byb_dt_20b"]

    def test_mixed_scalar_and_array(self):
        exp = expand_columns([col("{LISTING}_return_{N}ms", LISTING="okx", N=[1, 5])])
        assert list(exp.names) == ["okx_return_1ms", "okx_return_5ms"]

    def test_multiple_arg_mappings_concatenate(self):
        spec = ColumnSpec("{LISTING}_ema_ofi_{N}b", (
            {"LISTING": "bin", "N": 100},
            {"LISTING": "byb", "N": [3, 10]},
        ))
        exp = expand_columns([spec])
        assert list(exp.names) == ["bin_ema_ofi_100b", "byb_ema_ofi_3b", "byb_ema_ofi_10b"]

    def test_spec_order_is_output_order(self):
        exp = expand_columns([
            col("{LISTING}_spread_width", LISTING="byb"),
            col("{LISTING}_microprice", LISTING="bin"),
        ])
        assert list(exp.names) == ["byb_spread_width", "bin_microprice"]

    def test_cross_template_unit(self):
        exp = expand_columns([col(
            "{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms",
            LISTING1="bin", LISTING2="byb", N=[10, 20])])
        assert list(exp.names) == [
            "bin_byb_ema_log_microprice_ratio_10ms",
            "bin_byb_ema_log_microprice_ratio_20ms",
        ]
        assert exp.units[0].pair == ("bin", "byb")
        assert exp.units[0].listing is None

    def test_listing_spans_grouping(self):
        exp = expand_columns([
            col("{LISTING}_ema_ofi_{N}b", LISTING=["bin", "byb"], N=[100, 3]),
            col("{LISTING}_dt_{N}t", LISTING="bin", N=10),
        ])
        assert listing_spans(exp.units, "bin") == {"ema_ofi_b": [3, 100], "dt_t": [10]}
        assert listing_spans(exp.units, "byb") == {"ema_ofi_b": [3, 100]}


class TestValidation:
    def test_unknown_template(self):
        with pytest.raises(ValueError, match="unknown template"):
            expand_columns([col("{LISTING}_ema_ofx_{N}b", LISTING="bin", N=3)])

    def test_missing_placeholder(self):
        with pytest.raises(ValueError, match="missing.*N"):
            expand_columns([col("{LISTING}_ema_ofi_{N}b", LISTING="bin")])

    def test_extra_placeholder(self):
        with pytest.raises(ValueError, match="unexpected"):
            expand_columns([col("{LISTING}_microprice", LISTING="bin", N=3)])

    @pytest.mark.parametrize("bad_n", [0, -1, True, "10", 1.5])
    def test_bad_n(self, bad_n):
        with pytest.raises(ValueError, match="N must be a positive int"):
            expand_columns([col("{LISTING}_ema_ofi_{N}b", LISTING="bin", N=bad_n)])

    def test_cross_same_listing_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            expand_columns([col(
                "{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms",
                LISTING1="bin", LISTING2="bin", N=10)])

    def test_listing_vocabulary_enforced(self):
        specs = [col("{LISTING}_microprice", LISTING="okx")]
        expand_columns(specs, listings=["okx"])  # ok
        with pytest.raises(ValueError, match="not one of the listings"):
            expand_columns(specs, listings=["bin", "byb"])

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            expand_columns([
                col("{LISTING}_microprice", LISTING="bin"),
                col("{LISTING}_microprice", LISTING="bin"),
            ])

    def test_empty_specs_rejected(self):
        with pytest.raises(ValueError, match="at least one ColumnSpec"):
            expand_columns([])

    def test_empty_args_rejected(self):
        with pytest.raises(ValueError, match="args is empty"):
            expand_columns([ColumnSpec("{LISTING}_microprice", ())])

    def test_empty_value_list_rejected(self):
        with pytest.raises(ValueError, match="empty value list"):
            expand_columns([col("{LISTING}_ema_ofi_{N}b", LISTING="bin", N=[])])


class TestConfig:
    def _cfg(self, columns):
        return DatasetRawConfig(columns=tuple(columns))

    def test_columns_required(self):
        with pytest.raises(TypeError):
            DatasetRawConfig()

    def test_order_changes_config_str(self):
        a = self._cfg([col("{LISTING}_microprice", LISTING="bin"),
                       col("{LISTING}_spread_width", LISTING="bin")])
        b = self._cfg([col("{LISTING}_spread_width", LISTING="bin"),
                       col("{LISTING}_microprice", LISTING="bin")])
        assert a.config_str() != b.config_str()      # order is part of the output

    def test_knobs_change_config_str(self):
        a = self._cfg([col("{LISTING}_spread_wide_flag", LISTING="bin")])
        b = DatasetRawConfig(columns=a.columns, wide_threshold={"bin": 5e-4})
        assert a.config_str() != b.config_str()

    def test_n_features(self):
        cfg = self._cfg([col("{LISTING}_dt_{N}b", LISTING=["bin", "byb"], N=[10, 100, 1000])])
        assert cfg.n_features() == 6


def _binding_for(template: str, listing: str = "bin") -> dict:
    """One valid binding per template (cross pairs are always bin/byb), N=10."""
    t = TEMPLATES[template]
    b: dict = {}
    if t.cross:
        b["LISTING1"], b["LISTING2"] = "bin", "byb"
    else:
        b["LISTING"] = listing
    if t.has_n:
        b["N"] = 10
    return b


# (template, listing) cases: per-listing templates run on BOTH listings — "bin"
# has history before grid start (no-warmup regime) while "byb" starts ~50ms in
# (warmup regime: forward-filled microprice is 0 on the early grid, exercising
# the return/centered-EMA validity gates). Cross runs once (covers both).
_CASES = [(t, L) for t in sorted(TEMPLATES)
          for L in (("bin",) if TEMPLATES[t].cross else ("bin", "byb"))]


_RICH_LISTINGS = ["bin", "byb"]
# Deliberately NON-default per-listing knobs (and different per listing), so a
# production mutation that hardcodes the fallback (1e-4 / 0.0) or drops the
# config plumbing is caught by the oracle comparison.
_WIDE_THR = {"bin": 2.3e-4, "byb": 1.7e-4}
_MP_REF = {"bin": 0.15, "byb": 0.14}
_RICH_KW = dict(warmup_ms=0, horizon_ms=5.0, only_on_event=False,
                wide_threshold=dict(_WIDE_THR), microprice_ref=dict(_MP_REF))


@pytest.fixture(scope="module")
def rich_session():
    from boba.dataset.session_data import SessionData
    rng = np.random.default_rng(11)
    n, m = 400, 120
    listing_book_t, bids, asks, bqs, aqs, feeds = {}, {}, {}, {}, {}, {}
    tr_t, tr_prc, tr_qty, tr_dir = {}, {}, {}, {}
    for e in _RICH_LISTINGS:
        t = np.sort(rng.integers(0, 400_000_000, n).astype(np.int64))
        t[50:53] = t[50]                      # a same-ns burst
        if e == "byb":
            t = t + 50_000_000                # byb's book starts ~50ms AFTER grid start
        bid = 0.15 + rng.normal(0, 1e-5, n).cumsum()
        ask = bid + np.abs(rng.normal(2e-5, 1e-5, n)) + 1e-6
        listing_book_t[e] = t
        bids[e], asks[e] = bid, ask
        bqs[e] = rng.uniform(100, 5000, n)
        aqs[e] = rng.uniform(100, 5000, n)
        feeds[e] = rng.integers(0, 2_000_000, n).astype(np.int64)
        tt = np.sort(rng.integers(0, 400_000_000, m).astype(np.int64))
        tr_t[e] = tt
        tr_prc[e] = 0.15 + rng.normal(0, 2e-5, m).cumsum()
        tr_qty[e] = rng.uniform(10, 1000, m)
        tr_dir[e] = rng.choice([1.0, -1.0], m)
    return SessionData(
        listing_book_t=listing_book_t, listing_book_bid=bids, listing_book_ask=asks,
        listing_book_bid_qty=bqs, listing_book_ask_qty=aqs,
        listing_feed_latency_excess_ns=feeds,
        trade_ts=tr_t, trade_exchange_ts=tr_t,
        trade_prc=tr_prc, trade_qty=tr_qty, trade_dir=tr_dir,
        target_listing="bin",
        book_t=listing_book_t["bin"], book_bid=bids["bin"], book_ask=asks["bin"],
        book_mid=(bids["bin"] + asks["bin"]) / 2,
        feed_latency_raw_ns=feeds["bin"], feed_latency_excess_ns=feeds["bin"],
        all_rx=np.sort(np.concatenate([*listing_book_t.values(), *tr_t.values()])),
    )


@pytest.fixture(scope="module")
def full_build(rich_session):
    specs = tuple(
        ColumnSpec(t, tuple(_binding_for(t, L)
                            for L in (("bin",) if TEMPLATES[t].cross else ("bin", "byb"))))
        for t in TEMPLATES
    )
    cfg = DatasetRawConfig(columns=specs, **_RICH_KW)
    res = build_features_raw(rich_session, _RICH_LISTINGS, cfg)
    return res, {n: i for i, n in enumerate(res.column_names)}


class TestEverySingleColumnAlone:
    """Every template in the registry works as the ONLY requested column, and
    its values are bit-identical to the same column of a build that requests
    one instance of every template at once."""

    @pytest.mark.parametrize("template,listing", _CASES)
    def test_single_column_build(self, template, listing, rich_session, full_build):
        b = _binding_for(template, listing)
        name = template.format(**b)
        cfg = DatasetRawConfig(columns=(ColumnSpec(template, (b,)),), **_RICH_KW)
        res = build_features_raw(rich_session, _RICH_LISTINGS, cfg)
        assert res.column_names == [name]
        assert res.x.shape == (len(res.timestamp_ms), 1)
        assert np.isfinite(res.x).all()
        full, ci = full_build
        np.testing.assert_array_equal(
            res.x[:, 0], full.x[:, ci[name]],
            err_msg=f"single-column build of {name} differs from the full build")


class TestArbitrarySpanEndToEnd:
    """A span that never existed in the legacy catalogue (N=7) computes correctly."""

    def _make_session(self):
        from boba.dataset.session_data import SessionData
        n = 50
        bbo_t = np.arange(n, dtype=np.int64) * 1_000_000
        rng = np.random.default_rng(7)
        bid = 100.0 + rng.normal(0, 0.01, n).cumsum()
        ask = bid + 0.1
        bq = rng.uniform(5, 15, n)
        aq = rng.uniform(5, 15, n)
        feed = np.zeros(n, dtype=np.int64)
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return SessionData(
            listing_book_t={"bin": bbo_t}, listing_book_bid={"bin": bid},
            listing_book_ask={"bin": ask}, listing_book_bid_qty={"bin": bq},
            listing_book_ask_qty={"bin": aq},
            trade_ts={"bin": empty_i}, trade_exchange_ts={"bin": empty_i},
            trade_prc={"bin": empty_f}, trade_qty={"bin": empty_f},
            trade_dir={"bin": empty_f},
            listing_feed_latency_excess_ns={"bin": feed},
            target_listing="bin",
            book_t=bbo_t, book_bid=bid, book_ask=ask, book_mid=(bid + ask) / 2,
            feed_latency_raw_ns=feed, feed_latency_excess_ns=feed,
            all_rx=bbo_t,
        )

    def test_novel_ofi_span_matches_manual_ema(self):
        data = self._make_session()
        cfg = DatasetRawConfig(
            columns=(col("{LISTING}_ema_ofi_{N}b", LISTING="bin", N=7),),
            warmup_ms=0, horizon_ms=0, only_on_event=False,
        )
        res = build_features_raw(data, ["bin"], cfg)
        assert res.column_names == ["bin_ema_ofi_7b"]

        from boba.dataset.raw import compute_ofi_events, forward_fill_to_ms_grid
        ofi = compute_ofi_events(data.listing_book_bid["bin"], data.listing_book_ask["bin"],
                                 data.listing_book_bid_qty["bin"], data.listing_book_ask_qty["bin"])
        alpha = 2.0 / (7 + 1)
        ema = lfilter([alpha], [1.0, -(1.0 - alpha)], ofi)
        grid_t_ns = (res.timestamp_ms.astype(np.int64)) * 1_000_000
        expected = forward_fill_to_ms_grid(data.listing_book_t["bin"], ema, grid_t_ns, 0.0)
        np.testing.assert_allclose(res.x[:, 0], expected.astype(np.float32), rtol=1e-6)


# ── Independent oracle ────────────────────────────────────────────────────────
# A from-scratch reimplementation of every template from docs/raw_features.md
# using plain Python loops — deliberately sharing NO kernels with boba/dataset/raw.py
# (hand-rolled EMA recursion instead of scipy.lfilter, two-pointer forward-fill
# instead of searchsorted, OFI/microprice/aggregation rules re-derived from the
# spec text). Used to pin down VALUE correctness, not just selection consistency.

def _o_ema(x, N):
    a = 2.0 / (N + 1)
    y, out = 0.0, np.empty(len(x), np.float64)
    for i, v in enumerate(x):
        y = a * float(v) + (1.0 - a) * y
        out[i] = y
    return out


def _o_ffill(ev_t, vals, grid_t, fill=0.0):
    out = np.full(len(grid_t), fill, np.float64)
    j = -1
    for i, t in enumerate(grid_t):
        while j + 1 < len(ev_t) and ev_t[j + 1] <= t:
            j += 1
        if j >= 0:
            out[i] = float(vals[j])
    return out


def _o_agg_bbo(ts, bid, ask, bq, aq):
    """Same-ns aggregation: keep the LAST state of each ns burst."""
    keep = [i for i in range(len(ts)) if i == len(ts) - 1 or ts[i + 1] != ts[i]]
    return ts[keep], bid[keep], ask[keep], bq[keep], aq[keep]


def _o_agg_trades(ts, prc, qty, dr):
    """Consecutive same-(ts, side) runs → one event: qty summed, VWAP price."""
    groups = []
    for i in range(len(ts)):
        if groups and groups[-1]["ts"] == ts[i] and groups[-1]["dir"] == dr[i]:
            groups[-1]["notional"] += prc[i] * qty[i]
            groups[-1]["qty"] += qty[i]
        else:
            groups.append(dict(ts=int(ts[i]), notional=prc[i] * qty[i],
                               qty=float(qty[i]), dir=float(dr[i])))
    t = np.array([g["ts"] for g in groups], np.int64)
    q = np.array([g["qty"] for g in groups], np.float64)
    p = np.array([g["notional"] / g["qty"] for g in groups], np.float64)
    d = np.array([g["dir"] for g in groups], np.float64)
    return t, p, q, d


def _o_bbo_state(data, e):
    bts, bid, ask, bq, aq = _o_agg_bbo(
        data.listing_book_t[e], data.listing_book_bid[e], data.listing_book_ask[e],
        data.listing_book_bid_qty[e], data.listing_book_ask_qty[e])
    mp = (bq * ask + aq * bid) / (bq + aq)
    sw = (ask - bid) / mp
    depth = bq + aq
    imb = (bq - aq) / (bq + aq)
    wide = (sw > _WIDE_THR[e]).astype(np.float64)
    ofi = [0.0]
    for i in range(1, len(bts)):
        if bid[i] > bid[i - 1]:
            eb = bq[i]
        elif bid[i] < bid[i - 1]:
            eb = -bq[i - 1]
        else:
            eb = bq[i] - bq[i - 1]
        if ask[i] < ask[i - 1]:
            ea = -aq[i]
        elif ask[i] > ask[i - 1]:
            ea = aq[i - 1]
        else:
            ea = -(aq[i] - aq[i - 1])
        ofi.append(eb + ea)
    ofi = np.array(ofi, np.float64)
    alr = np.zeros(len(bts), np.float64)
    for i in range(1, len(bts)):
        alr[i] = abs(np.log(mp[i] / mp[i - 1]))
    return bts, mp, sw, depth, imb, wide, ofi, alr


def _o_dt(ts, N):
    out = np.zeros(len(ts), np.float64)
    for i in range(N, len(ts)):
        out[i] = float(ts[i] - ts[i - N]) / 1e6
    return out


def _o_per_ms_trade_value(data, e, grid_t):
    """Per-tick Σ qty·prc over RAW trades in (t − 1ms, t], split by aggressor."""
    buy = np.zeros(len(grid_t), np.float64)
    sell = np.zeros(len(grid_t), np.float64)
    ts, prc, qty, dr = (data.trade_ts[e], data.trade_prc[e],
                        data.trade_qty[e], data.trade_dir[e])
    for k in range(len(ts)):
        for i, t in enumerate(grid_t):
            if t - 1_000_000 < ts[k] <= t:
                (buy if dr[k] > 0 else sell)[i] += prc[k] * qty[k]
                break
            if ts[k] <= t - 1_000_000:
                break
    return buy, sell


def _oracle_column(data, grid_t, template, binding):
    t = TEMPLATES[template]
    fam, N = t.family, binding.get("N")

    if t.cross:
        def mp_grid(listing):
            bts, mp, *_ = _o_bbo_state(data, listing)
            return _o_ffill(bts, mp, grid_t, 0.0)
        mp1, mp2 = mp_grid(binding["LISTING1"]), mp_grid(binding["LISTING2"])
        valid = (mp1 > 0) & (mp2 > 0)
        ratio = np.where(valid,
                         np.log(np.where(valid, mp1, 1.0)) - np.log(np.where(valid, mp2, 1.0)),
                         0.0)
        return _o_ema(ratio if fam == "cross_ratio_ms" else ratio * ratio, N)

    e = binding["LISTING"]
    bts, mp, sw, depth, imb, wide, ofi, alr = _o_bbo_state(data, e)
    tts, tprc, tqty, tdir = _o_agg_trades(
        data.trade_ts[e], data.trade_prc[e], data.trade_qty[e], data.trade_dir[e])

    if fam == "microprice":
        return _o_ffill(bts, mp, grid_t)
    if fam == "spread_width":
        return _o_ffill(bts, sw, grid_t)
    if fam == "book_depth":
        return _o_ffill(bts, depth, grid_t)
    if fam == "book_imbalance":
        return _o_ffill(bts, imb, grid_t)
    if fam == "spread_wide_flag":
        return _o_ffill(bts, wide, grid_t)
    if fam == "dt_b":
        return _o_ffill(bts, _o_dt(bts, N), grid_t)
    if fam == "dt_t":
        return _o_ffill(tts, _o_dt(tts, N), grid_t)
    if fam == "dt_m":                              # dt over last N MID moves ((bid+ask)/2 changes)
        bts2, bid2, ask2, _, _ = _o_agg_bbo(
            data.listing_book_t[e], data.listing_book_bid[e], data.listing_book_ask[e],
            data.listing_book_bid_qty[e], data.listing_book_ask_qty[e])
        mid = (bid2 + ask2) / 2.0
        mvmask = np.zeros(len(bts2), bool)
        if len(bts2) > 1:
            mvmask[1:] = np.diff(mid) != 0.0
        mid_t = bts2[mvmask]
        return _o_ffill(mid_t, _o_dt(mid_t, N), grid_t)
    if fam == "trade_count_ms":
        win_ns = N * 1_000_000
        out = np.zeros(len(grid_t), np.float64)
        for i, g in enumerate(grid_t):          # independent per-tick count over (g-N ms, g]
            out[i] = float(np.sum((tts > g - win_ns) & (tts <= g)))
        return out
    if fam == "time_since_last_trade_ms":
        out = np.zeros(len(grid_t), np.float64)
        for i, g in enumerate(grid_t):
            prior = tts[tts <= g]
            out[i] = float(g - prior[-1]) / 1e6 if len(prior) else 0.0
        return out
    if fam == "time_since_spread_wide_ms":
        wide_t = bts[wide > 0.5]
        out = np.zeros(len(grid_t), np.float64)
        for i, g in enumerate(grid_t):
            prior = wide_t[wide_t <= g]
            out[i] = float(g - prior[-1]) / 1e6 if len(prior) else 0.0
        return out
    if fam == "feed_latency_excess_ms":
        return _o_ffill(data.listing_book_t[e],
                        data.listing_feed_latency_excess_ns[e].astype(np.float64) / 1e6, grid_t)
    if fam in ("buy_trade_value", "sell_trade_value",
               "ema_buy_trade_value_ms", "ema_sell_trade_value_ms"):
        buy, sell = _o_per_ms_trade_value(data, e, grid_t)
        if fam == "buy_trade_value":
            return buy
        if fam == "sell_trade_value":
            return sell
        return _o_ema(buy if fam == "ema_buy_trade_value_ms" else sell, N)
    if fam == "return_ms":
        mpg = _o_ffill(bts, mp, grid_t)
        out = np.zeros(len(grid_t), np.float64)
        for i in range(N, len(grid_t)):
            if mpg[i] > 0 and mpg[i - N] > 0:
                out[i] = np.log(mpg[i]) - np.log(mpg[i - N])
        return out
    if fam in ("ema_microprice_centered_ms", "ema_microprice_centered_sq_ms"):
        mpg = _o_ffill(bts, mp, grid_t)
        c = np.where(mpg > 0, mpg - _MP_REF[e], 0.0)
        return _o_ema(c if fam == "ema_microprice_centered_ms" else c * c, N)
    if fam == "ema_microprice_return_sq_ms":
        mpg = _o_ffill(bts, mp, grid_t)
        r1 = np.zeros(len(grid_t), np.float64)
        for i in range(1, len(grid_t)):
            if mpg[i] > 0 and mpg[i - 1] > 0:
                r1[i] = np.log(mpg[i]) - np.log(mpg[i - 1])
        return _o_ema(r1 * r1, N)
    if fam in ("ema_buy_trade_qty_t", "ema_sell_trade_qty_t",
               "ema_buy_trade_value_t", "ema_sell_trade_value_t"):
        is_buy = tdir > 0
        base = tqty if fam.endswith("qty_t") else tqty * tprc
        x = np.where(is_buy if "buy" in fam else ~is_buy, base, 0.0)
        return _o_ffill(tts, _o_ema(x, N), grid_t)
    if fam == "ema_trade_serial_cov_t":
        cov = np.zeros(len(tprc), np.float64)
        for i in range(2, len(tprc)):
            cov[i] = (tprc[i] - tprc[i - 1]) * (tprc[i - 1] - tprc[i - 2])
        return _o_ffill(tts, _o_ema(cov, N), grid_t)
    bbo_inputs = {
        "ema_ofi_b": ofi, "ema_ofi_sq_b": ofi ** 2, "ema_abs_log_ret_b": alr,
        "ema_book_imbalance_b": imb, "ema_book_imbalance_sq_b": imb ** 2,
        "ema_book_depth_b": depth, "ema_book_depth_sq_b": depth ** 2,
        "ema_spread_wide_flag_b": wide,
    }
    if fam in bbo_inputs:
        return _o_ffill(bts, _o_ema(bbo_inputs[fam], N), grid_t)
    raise AssertionError(f"oracle has no implementation for family {fam!r}")


class TestSingleColumnCorrectness:
    """Production VALUES for every template match the independent pure-python
    oracle above. Combined with TestEverySingleColumnAlone (single-column build
    ≡ full build, bit-identical), this pins both the formulas and the
    selective-computation plumbing for all 32 column types."""

    @pytest.mark.parametrize("template,listing", _CASES)
    def test_matches_independent_oracle(self, template, listing, rich_session, full_build):
        full, ci = full_build
        # the ms-window families assume a contiguous grid (index shift == ms shift)
        assert np.array_equal(np.diff(full.timestamp_ms),
                              np.ones(len(full.timestamp_ms) - 1))
        grid_t = full.timestamp_ms.astype(np.int64) * 1_000_000
        b = _binding_for(template, listing)
        name = template.format(**b)
        expected = _oracle_column(rich_session, grid_t, template, b)
        # a vacuous all-zeros comparison would prove nothing — the session is
        # constructed so every template produces signal
        assert np.any(expected != 0.0), f"oracle column {name} is degenerate"
        np.testing.assert_allclose(
            full.x[:, ci[name]].astype(np.float64), expected, rtol=2e-5, atol=1e-12,
            err_msg=f"{name} deviates from the spec oracle")


class TestLegacyCatalogueCounts:
    """Literal pins of the legacy catalogue size, independent of the expansion
    code path (the old repo's n_per_listing() arithmetic: 18 instantaneous/dt/flow
    + 26 temporal + 45 microstructure = 89 per listing; ×3 listings = 267;
    + C(3,2)·6 spans·2 = 36 cross = 303)."""

    def test_counts(self):
        from tests.helpers import legacy_local_names, make_cfg
        assert len(legacy_local_names()) == 89
        assert make_cfg().n_features() == 267
        assert make_cfg(cross=True).n_features() == 303
