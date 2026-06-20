"""Tests for the cross-listing log-microprice-ratio features in boba/dataset/raw.py.

Cross-listing columns are requested explicitly as ColumnSpecs on the
"{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms" / "..._ratio_sq_{N}ms"
templates — computed iff requested, output order = request order. Any ordered
pair of DISTINCT listings is computable (ema(log a/b) = −ema(log b/a); the _sq
column is order-invariant); LISTING1 == LISTING2 raises. Covers the legacy
catalogue's alphabetical-pair cross block, the sign-flip identity, _sq
order-invariance, variance ≥ 0, end-to-end builds with a known constant ratio
and with an explicitly reversed pair, and explicit column selection
(n_features / order-sensitive cache_key / subset == full-build values).
"""
from __future__ import annotations

import math
import re
from itertools import combinations

import numpy as np
import pytest

from boba.dataset.columns import col
from boba.dataset.raw import _alpha, _ewm_1d, build_features_raw, feature_names
from tests.helpers import (
    EMA_MICROPRICE_MS_SPANS, cols_from_names, legacy_local_names, make_cfg,
)

RATIO_T = "{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms"
RATIO_SQ_T = "{LISTING1}_{LISTING2}_ema_log_microprice_ratio_sq_{N}ms"


class TestCrossListingColumns:
    def test_count_matches_n_features(self):
        cfg = make_cfg(cross=True)
        names = feature_names(cfg, list(cfg.listings))
        assert len(names) == cfg.n_features()
        assert len(set(names)) == len(names)               # all unique

    def test_cross_count(self):
        cfg = make_cfg(cross=True); L = list(cfg.listings)
        names = feature_names(cfg, L)
        cross = [n for n in names if "ema_log_microprice_ratio" in n]
        assert len(cross) == math.comb(len(L), 2) * len(EMA_MICROPRICE_MS_SPANS) * 2

    def test_legacy_catalogue_pairs_alphabetical_one_direction(self):
        # The legacy helper catalogue still emits one alphabetical direction per
        # pair, with the mean column immediately followed by its _sq twin.
        cfg = make_cfg(cross=True); L = list(cfg.listings)
        names = feature_names(cfg, L)
        mean_names = [n for n in names if re.search(r"_ema_log_microprice_ratio_\d+ms$", n)]
        pairs: list[tuple[str, str]] = []
        for n in mean_names:
            l1, l2 = re.match(r"([a-z]+)_([a-z]+)_ema_log_microprice_ratio", n).group(1, 2)
            if (l1, l2) not in pairs:
                pairs.append((l1, l2))
        assert pairs == list(combinations(sorted(L), 2))   # alphabetical, no reverse dup
        for n in mean_names:                               # mean/sq interleaved
            i = names.index(n)
            assert names[i + 1] == n.replace("_ratio_", "_ratio_sq_")

    def test_same_listing_pair_raises(self):
        cfg = make_cfg(columns=(col(RATIO_T, LISTING1="bin", LISTING2="bin", N=100),))
        with pytest.raises(ValueError, match="must differ"):
            cfg.n_features()

    def test_reversed_pair_allowed(self):
        # Pairs are no longer restricted to alphabetical order: (byb, bin) is
        # requestable as-is and names follow the requested direction.
        cfg = make_cfg(columns=(
            col(RATIO_T, LISTING1="byb", LISTING2="bin", N=100),
            col(RATIO_SQ_T, LISTING1="byb", LISTING2="bin", N=100),
        ))
        assert feature_names(cfg, list(cfg.listings)) == [
            "byb_bin_ema_log_microprice_ratio_100ms",
            "byb_bin_ema_log_microprice_ratio_sq_100ms",
        ]

    def test_cross_block_after_per_listing(self):
        cfg = make_cfg(cross=True); L = list(cfg.listings)
        names = feature_names(cfg, L)
        n_per = len(legacy_local_names()) * len(L)
        assert all("ema_log_microprice_ratio" not in n for n in names[:n_per])
        exp_cross = []
        for l1, l2 in combinations(sorted(L), 2):
            for N in EMA_MICROPRICE_MS_SPANS:
                exp_cross += [f"{l1}_{l2}_ema_log_microprice_ratio_{N}ms",
                              f"{l1}_{l2}_ema_log_microprice_ratio_sq_{N}ms"]
        assert names[n_per:] == exp_cross

    def test_no_cross_templates_means_no_cross_columns(self):
        cfg = make_cfg(cross=False)
        names = feature_names(cfg, list(cfg.listings))
        assert all("ema_log_microprice_ratio" not in n for n in names)


class TestCrossListingMath:
    def _two_walks(self, seed):
        rng = np.random.default_rng(seed)
        a = 100 + np.cumsum(rng.standard_normal(3000) * 0.01)
        b = 100 + np.cumsum(rng.standard_normal(3000) * 0.01)
        return np.log(a) - np.log(b)                       # log(a/b)

    def test_ratio_sign_flip_identity(self):
        r = self._two_walks(0); al = _alpha(100)
        np.testing.assert_allclose(_ewm_1d(r, al), -_ewm_1d(-r, al), atol=1e-12)

    def test_sq_order_invariant(self):
        r = self._two_walks(1); al = _alpha(100)
        np.testing.assert_array_equal(_ewm_1d(r * r, al), _ewm_1d((-r) * (-r), al))

    def test_variance_nonnegative(self):
        r = self._two_walks(2); al = _alpha(200)
        var = _ewm_1d(r * r, al) - _ewm_1d(r, al) ** 2     # EWM variance ≥ 0 (Cauchy–Schwarz)
        assert (var >= -1e-12).all()


class TestCrossListingEndToEnd:
    def _two_listing_session(self, mid_a, mid_b, bbo_t):
        from boba.dataset.session_data import SessionData
        la, lb = "aaa_doge_usdt", "bbb_doge_usdt"           # alphabetical: la < lb
        n = len(bbo_t); sp = 0.001; bq = aq = np.full(n, 10.0)   # equal qty → microprice == mid
        bid_a, ask_a = mid_a - sp / 2, mid_a + sp / 2
        bid_b, ask_b = mid_b - sp / 2, mid_b + sp / 2
        tr_t = bbo_t[::10]; tr_prc = mid_a[::10]; tr_qty = np.full(len(tr_t), 5.0); tr_dir = np.ones(len(tr_t))
        feed = np.zeros(n, np.int64)
        data = SessionData(
            listing_book_t={la: bbo_t, lb: bbo_t},
            listing_book_bid={la: bid_a, lb: bid_b}, listing_book_ask={la: ask_a, lb: ask_b},
            listing_book_bid_qty={la: bq, lb: bq}, listing_book_ask_qty={la: aq, lb: aq},
            trade_ts={la: tr_t, lb: tr_t}, trade_exchange_ts={la: tr_t, lb: tr_t},
            trade_prc={la: tr_prc, lb: tr_prc}, trade_qty={la: tr_qty, lb: tr_qty},
            trade_dir={la: tr_dir, lb: tr_dir},
            listing_feed_latency_excess_ns={la: feed, lb: feed},
            target_listing=la, book_t=bbo_t, book_bid=bid_a, book_ask=ask_a, book_mid=mid_a,
            feed_latency_raw_ns=feed, feed_latency_excess_ns=feed, all_rx=bbo_t,
        )
        return data, la, lb

    def test_constant_ratio_converges_and_zero_variance(self):
        bbo_t = np.arange(1500, dtype=np.int64) * 1_000_000           # 1500 events, 1ms apart
        mid_a = np.full(1500, 100.0); k = 1.001; mid_b = mid_a * k    # constant ratio
        data, la, lb = self._two_listing_session(mid_a, mid_b, bbo_t)
        cfg = make_cfg(listings=(la, lb), cross=True, target_listing=la,
                       ema_microprice_ms_spans=(100,), warmup_ms=0,
                       horizon_ms=20.0, only_on_event=False)
        res = build_features_raw(data, [la, lb], cfg)
        cols = res.column_names
        mean_col = f"{la}_{lb}_ema_log_microprice_ratio_100ms"
        sq_col = f"{la}_{lb}_ema_log_microprice_ratio_sq_100ms"
        assert mean_col in cols and sq_col in cols
        mean = res.x[:, cols.index(mean_col)]; sq = res.x[:, cols.index(sq_col)]
        assert np.isfinite(mean).all() and np.isfinite(sq).all()
        # log(mid_a/mid_b) = −log(k), constant → EMA converges to it; variance → 0
        np.testing.assert_allclose(mean[-1], -math.log(k), atol=5e-5)
        assert (sq - mean ** 2 >= -1e-9).all()
        assert sq[-1] - mean[-1] ** 2 == pytest.approx(0.0, abs=1e-6)

    def test_reversed_pair_negates_mean_preserves_sq(self):
        # Explicitly request the (lb, la) direction: the ratio-mean column is
        # the NEGATION of the (la, lb) build's (ema(log b/a) = −ema(log a/b))
        # and the _sq column is identical (squaring kills the sign).
        bbo_t = np.arange(1500, dtype=np.int64) * 1_000_000
        rng = np.random.default_rng(7)
        mid_a = 100.0 + np.cumsum(rng.standard_normal(1500) * 0.01)
        mid_b = mid_a * (1.001 + 1e-4 * np.sin(np.arange(1500) / 50.0))  # varying ratio
        data, la, lb = self._two_listing_session(mid_a, mid_b, bbo_t)
        kw = dict(listings=(la, lb), target_listing=la, warmup_ms=0,
                  horizon_ms=20.0, only_on_event=False)

        def build(l1, l2):
            cols = (col(RATIO_T, LISTING1=l1, LISTING2=l2, N=100),
                    col(RATIO_SQ_T, LISTING1=l1, LISTING2=l2, N=100))
            return build_features_raw(data, [la, lb], make_cfg(columns=cols, **kw))

        fwd = build(la, lb); rev = build(lb, la)
        assert fwd.column_names == [f"{la}_{lb}_ema_log_microprice_ratio_100ms",
                                    f"{la}_{lb}_ema_log_microprice_ratio_sq_100ms"]
        assert rev.column_names == [f"{lb}_{la}_ema_log_microprice_ratio_100ms",
                                    f"{lb}_{la}_ema_log_microprice_ratio_sq_100ms"]
        assert not np.allclose(fwd.x[:, 0], 0.0)           # non-degenerate input
        np.testing.assert_allclose(rev.x[:, 0], -fwd.x[:, 0], atol=1e-12)
        np.testing.assert_array_equal(rev.x[:, 1], fwd.x[:, 1])


class TestColumnSelection:
    """Explicit ColumnSpec selections: cross-listing columns are selectable,
    output order = request order (no canonical reordering), n_features /
    cache_key track the ORDERED selection, and a build with only some cross
    columns matches the same-named columns of the full build."""

    def test_n_features_and_filtered_names(self):
        cfg0 = make_cfg(cross=True)
        full = feature_names(cfg0, list(cfg0.listings))
        pick = (full[-1], full[0], full[200])              # includes a cross col (first)
        cfg = make_cfg(columns=cols_from_names(pick))
        assert cfg.n_features() == 3
        # Output order is the REQUEST order — no re-sorting to catalogue order
        assert feature_names(cfg, list(cfg.listings)) == list(pick)

    def test_cache_key_tracks_columns_order_sensitively(self):
        cfg0 = make_cfg(cross=True)
        full = feature_names(cfg0, list(cfg0.listings))
        cfg = make_cfg(columns=cols_from_names((full[0], full[-1])))
        cfg_rev = make_cfg(columns=cols_from_names((full[-1], full[0])))
        assert cfg.cache_key() != cfg0.cache_key()
        # Column order is part of the output, so permuted selections are
        # different datasets → DIFFERENT cache keys (ordered names are hashed)
        assert cfg_rev.cache_key() != cfg.cache_key()

    def test_cross_subset_end_to_end(self):
        bbo_t = np.arange(1500, dtype=np.int64) * 1_000_000
        mid_a = np.full(1500, 100.0); mid_b = mid_a * 1.001
        data, la, lb = TestCrossListingEndToEnd()._two_listing_session(mid_a, mid_b, bbo_t)
        kw = dict(listings=(la, lb), target_listing=la,
                  warmup_ms=0, horizon_ms=20.0, only_on_event=False)
        full = build_features_raw(
            data, [la, lb], make_cfg(cross=True, ema_microprice_ms_spans=(100,), **kw))
        # sq-only cross selection (the ratio-mean column is NOT selected),
        # requested BEFORE the per-listing column: output follows the request
        # order — the old API would have re-sorted this pick
        pick = (f"{la}_{lb}_ema_log_microprice_ratio_sq_100ms", f"{la}_microprice")
        sub = build_features_raw(
            data, [la, lb],
            make_cfg(columns=cols_from_names(pick, listings=(la, lb)), **kw))
        assert sub.column_names == list(pick)
        fci = {n: i for i, n in enumerate(full.column_names)}
        for j, nm in enumerate(sub.column_names):
            np.testing.assert_array_equal(sub.x[:, j], full.x[:, fci[nm]], err_msg=nm)
