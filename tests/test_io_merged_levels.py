"""Tests for the synthetic ``merged_levels`` price stream (boba.io).

Validation per CLAUDE.md: the vectorized builder is checked against a dead-simple,
independent reference loop on a REAL block (real blocks carry the arbitrary-ns,
multi-source event interleaving synthetic fixtures can't exercise). Plus synthetic
property tests of the merge semantics that need no data.
"""
import numpy as np
import polars as pl
import pytest

import boba.io as io


# ── synthetic property tests (no data) ───────────────────────────────────────

def _merge_side_ref(snap_rx, snap_ex, snap_px, tr_rx, tr_ex, tr_px):
    """Independent reference for io._merge_side: explicit loop, newest-by-exch wins, and on an
    exchange_time TIE the LATEST event in sequence wins (`>=`) — so a same-exchange_time sweep
    resolves to its last/deepest print."""
    ev = sorted(
        [(rx, 0, ex, px) for rx, ex, px in zip(snap_rx, snap_ex, snap_px)]
        + [(rx, 1, ex, px) for rx, ex, px in zip(tr_rx, tr_ex, tr_px)],
        key=lambda e: (e[0], e[1]),                       # snapshot before trade on equal rx
    )
    best_px, best_ex = np.nan, np.iinfo(np.int64).min
    rx_o, px_o, ex_o = [], [], []
    for rx, _kind, ex, px in ev:
        if ex >= best_ex:                                 # >= : ties resolve to the latest-in-sequence event
            best_px, best_ex = px, ex
        rx_o.append(rx); px_o.append(best_px); ex_o.append(best_ex)
    return np.array(rx_o), np.array(px_o), np.array(ex_o)


def test_merge_side_matches_reference():
    rng = np.random.default_rng(0)
    snap_rx = np.sort(rng.integers(0, 10_000, 200))
    snap_ex = snap_rx - rng.integers(0, 50, 200)          # exch slightly before rx
    snap_px = rng.normal(100, 1, 200)
    tr_rx = np.sort(rng.integers(0, 10_000, 120))
    tr_ex = tr_rx - rng.integers(0, 50, 120)
    tr_px = rng.normal(100, 1, 120)
    rx, px, ex = io._merge_side(snap_rx, snap_ex, snap_px, tr_rx, tr_ex, tr_px)
    rrx, rpx, rex = _merge_side_ref(snap_rx, snap_ex, snap_px, tr_rx, tr_ex, tr_px)
    assert np.array_equal(rx, rrx)
    assert np.allclose(px, rpx, equal_nan=True)
    assert np.array_equal(ex, rex)


def test_merge_side_newest_exchange_time_wins_not_rx_order():
    # a trade arriving later (rx) but stamped older (exch) must NOT override a fresher snapshot
    snap_rx = np.array([0, 100]); snap_ex = np.array([0, 90]); snap_px = np.array([10.0, 11.0])
    tr_rx = np.array([101]); tr_ex = np.array([50]); tr_px = np.array([99.0])   # stale exch
    _, px, _ = io._merge_side(snap_rx, snap_ex, snap_px, tr_rx, tr_ex, tr_px)
    assert px[-1] == 11.0                                  # held the fresher snapshot, ignored the stale trade


def test_merge_side_same_exchange_time_sweep_holds_latest_print():
    # one aggressive order printing several trades at ONE exchange_time (a book sweep, arriving in
    # sequence): the merge must hold the LAST/deepest print, not the first.
    snap_rx = np.array([0]); snap_ex = np.array([0]); snap_px = np.array([10.0])
    tr_rx = np.array([10, 11, 12]); tr_ex = np.array([5, 5, 5]); tr_px = np.array([10.1, 10.2, 10.3])
    rx, px, ex = io._merge_side(snap_rx, snap_ex, snap_px, tr_rx, tr_ex, tr_px)
    assert px[-1] == 10.3 and ex[-1] == 5                  # deepest sweep print, NOT 10.1 (the first)
    rrx, rpx, rex = _merge_side_ref(snap_rx, snap_ex, snap_px, tr_rx, tr_ex, tr_px)
    assert np.array_equal(rx, rrx) and np.allclose(px, rpx) and np.array_equal(ex, rex)


def test_uncross_book_trusts_fresher_side_and_pushes_stale_one_tick():
    # the independently-fused sides can cross; un-cross trusts the side with the newer exchange_time and
    # pushes the STALE side exactly one tick past it. Four rows: crossed/ask-fresh, not-crossed,
    # crossed/bid-fresh, locked(ask==bid).
    tick = 0.01
    bid    = np.array([100.05, 100.00, 100.20, 100.00])
    ask    = np.array([100.00, 100.50, 100.10, 100.00])
    bid_ex = np.array([1,      5,      9,      7])
    ask_ex = np.array([2,      4,      3,      7])
    b, a = io._uncross_book(bid, ask, bid_ex, ask_ex, tick)
    assert b[0] == pytest.approx(99.99) and a[0] == pytest.approx(100.00)   # ask fresher -> bid down a tick
    assert b[1] == pytest.approx(100.00) and a[1] == pytest.approx(100.50)  # not crossed -> unchanged
    assert b[2] == pytest.approx(100.20) and a[2] == pytest.approx(100.21)  # bid fresher -> ask up a tick
    assert b[3] == pytest.approx(100.00) and a[3] == pytest.approx(100.00)  # locked (ask==bid) -> unchanged
    assert np.all(a >= b)                                                   # every row is uncrossed


# ── real-block oracle (skipped without data) ─────────────────────────────────

_HAS_DATA = io.DATA_DIR is not None


def _build_ref(fl: pl.DataFrame, td: pl.DataFrame, tick: float) -> dict:
    """Dead-simple reference for the full merged stream: stream every event in rx order, newest-by-exch
    wins per side, keep the final state per distinct rx — then UN-CROSS (where ask < bid, trust the side
    with the newer exchange_time and push the stale one one `tick` past it; ties -> ask fresher)."""
    s_rx = fl["rx_time"].cast(pl.Int64).to_numpy(); s_ex = fl["exchange_time"].cast(pl.Int64).to_numpy()
    bid = fl["bid_prc"].to_numpy(); ask = fl["ask_prc"].to_numpy()
    t_rx = td["rx_time"].cast(pl.Int64).to_numpy(); t_ex = td["exchange_time"].cast(pl.Int64).to_numpy()
    t_px = td["prc"].to_numpy(); buy = td["aggressor"].to_numpy() == "Bid"
    ev = sorted(
        [(rx, 0, ex, b, a, False) for rx, ex, b, a in zip(s_rx, s_ex, bid, ask)]
        + [(rx, 1, ex, px, bu, True) for rx, ex, px, bu in zip(t_rx, t_ex, t_px, buy)],
        key=lambda e: (e[0], e[1]),
    )
    bb = ba = np.nan; bex = aex = np.iinfo(np.int64).min; out = {}
    for rx, _kind, ex, p1, p2, istr in ev:                  # >= : on an exchange_time tie the latest event in sequence wins
        if not istr:
            if ex >= bex: bb, bex = p1, ex
            if ex >= aex: ba, aex = p2, ex
        elif p2:                                            # buy -> ask
            if ex >= aex: ba, aex = p1, ex
        else:                                               # sell -> bid
            if ex >= bex: bb, bex = p1, ex
        b, a = bb, ba
        if a < b:                                           # un-cross this row
            if aex >= bex: b = a - tick                     # ask fresher -> clamp bid down a tick
            else:          a = b + tick                     # bid fresher -> clamp ask up a tick
        out[rx] = (b, a, bex, aex)
    rx = np.array(sorted(out))
    return dict(rx_time=rx,
                bid_prc=np.array([out[r][0] for r in rx]), ask_prc=np.array([out[r][1] for r in rx]),
                bid_exchange_time=np.array([out[r][2] for r in rx]), ask_exchange_time=np.array([out[r][3] for r in rx]))


@pytest.mark.skipif(not _HAS_DATA, reason="DATA_DIR not configured")
def test_merged_levels_matches_oracle_on_real_block():
    listing = "byb_eth_usdt_p"
    blocks = io.list_blocks(listing, "merged_levels")
    if not blocks:
        pytest.skip(f"no merged_levels blocks for {listing}")
    blk = blocks[0]
    fl = (io.load_block(blk, listing, "front_levels")
          .select("rx_time", "exchange_time", "bid_prc", "ask_prc").drop_nulls().sort("rx_time"))
    td = (io.load_block(blk, listing, "trade")
          .select("rx_time", "exchange_time", "aggressor", "prc", "qty")
          .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)).drop_nulls().sort("rx_time"))
    # restrict to the first 30 min so the pure-Python oracle loop stays fast
    cut = fl["rx_time"].cast(pl.Int64).to_numpy()[0] + 30 * 60 * 1_000_000_000
    fl = fl.filter(pl.col("rx_time").cast(pl.Int64) <= cut)
    td = td.filter(pl.col("rx_time").cast(pl.Int64) <= cut)

    # vectorized builder on the same restricted inputs (reuse io internals)
    s_rx = fl["rx_time"].cast(pl.Int64).to_numpy(); s_ex = fl["exchange_time"].cast(pl.Int64).to_numpy()
    bid = fl["bid_prc"].to_numpy(); ask = fl["ask_prc"].to_numpy()
    t_rx = td["rx_time"].cast(pl.Int64).to_numpy(); t_ex = td["exchange_time"].cast(pl.Int64).to_numpy()
    t_px = td["prc"].to_numpy(); buy = td["aggressor"].to_numpy() == "Bid"
    ask_rx, ask_px, ask_ex = io._merge_side(s_rx, s_ex, ask, t_rx[buy], t_ex[buy], t_px[buy])
    bid_rx, bid_px, bid_ex = io._merge_side(s_rx, s_ex, bid, t_rx[~buy], t_ex[~buy], t_px[~buy])
    uniq = np.unique(np.concatenate([s_rx, t_rx]))
    ai = np.searchsorted(ask_rx, uniq, "right") - 1; bi = np.searchsorted(bid_rx, uniq, "right") - 1
    tick = io.tick_size(listing)
    b_unc, a_unc = io._uncross_book(bid_px[bi], ask_px[ai], bid_ex[bi], ask_ex[ai], tick)   # the production un-cross

    ref = _build_ref(fl, td, tick)
    assert np.array_equal(uniq, ref["rx_time"])
    assert np.array_equal(b_unc, ref["bid_prc"])                # un-crossed prices match the oracle
    assert np.array_equal(a_unc, ref["ask_prc"])
    assert np.array_equal(bid_ex[bi], ref["bid_exchange_time"])  # exchange_times unchanged by un-crossing
    assert np.array_equal(ask_ex[ai], ref["ask_exchange_time"])
    assert np.all(a_unc >= b_unc)                              # and the result is genuinely UNCROSSED


@pytest.mark.skipif(not _HAS_DATA, reason="DATA_DIR not configured")
def test_merged_levels_load_and_cache_roundtrip():
    listing = "byb_eth_usdt_p"
    blk = io.list_blocks(listing, "merged_levels")[0]
    df = io.load_block(blk, listing, "merged_levels")
    assert df.columns == ["rx_time", "bid_prc", "ask_prc", "bid_exchange_time", "ask_exchange_time"]
    assert "bid_qty" not in df.columns and "ask_qty" not in df.columns      # price-only by design
    assert df["rx_time"].is_sorted()
    assert df["rx_time"].n_unique() == len(df)                              # one row per distinct rx
    assert (df["bid_prc"] > 0).all() and (df["ask_prc"] > 0).all()
    # NB: bid/ask are fused independently (newest-by-exch per side), so the stream
    # can transiently CROSS (ask < bid) — do NOT assert ask >= bid (it's ~0.1-0.3%).
    if io.SYNTHETIC_DATA_DIR is not None:                                   # cache written, and reads back identical
        assert (io.SYNTHETIC_DATA_DIR / f"{blk}.{listing}.merged_levels.parquet").exists()
        assert io.load_block(blk, listing, "merged_levels").equals(df)


def _next_snapshot_r2(blk: str, listing: str) -> float:
    """nb02's metric, sourced from the PRODUCTION merged_levels file: how much of the
    next-BBO-snapshot mid error the merged mid removes vs holding the last snapshot.
    R2 = 1 - SS_merged/SS_bbo. Uses only public load_block output (no io internals)."""
    f = io.load_block(blk, listing, "merged_levels")
    f_rx = f["rx_time"].cast(pl.Int64).to_numpy()
    f_mid = ((f["bid_prc"] + f["ask_prc"]) / 2.0).to_numpy()
    fl = (io.load_block(blk, listing, "front_levels")
          .select("rx_time", "bid_prc", "ask_prc").drop_nulls().sort("rx_time"))
    s_rx = fl["rx_time"].cast(pl.Int64).to_numpy()
    mid = ((fl["bid_prc"] + fl["ask_prc"]) / 2.0).to_numpy()
    q = s_rx[1:] - 1                                          # just before each next snapshot
    merged_pred = f_mid[np.maximum(np.searchsorted(f_rx, q, "right") - 1, 0)]
    bbo_pred, truth = mid[:-1], mid[1:]
    return 1.0 - np.sum((merged_pred - truth) ** 2) / np.sum((bbo_pred - truth) ** 2)


@pytest.mark.skipif(not _HAS_DATA, reason="DATA_DIR not configured")
@pytest.mark.parametrize("listing, published_r2", [
    # baseline after the sweep fix + the tick-aware un-cross (un-crossing is a rare/shallow correction,
    # so it barely moves R2: byb 0.567 -> 0.566, okx 0.531 -> 0.534).
    ("byb_eth_usdt_p", 0.566),     # slow ~20ms feed -> merge adds a clear lift
    ("okx_eth_usdt_p", 0.534),     # slow ~20ms feed -> bigger lift
])
def test_merged_levels_preserves_predictive_lift(listing, published_r2):
    """Guards the SIGNAL, not just the shape: the production stream must reproduce
    notebook 02's next-snapshot R2 (so a change that still builds a valid file but
    loses the predictive merge is caught) — and the un-crossed book never crosses."""
    blocks = io.list_blocks(listing, "merged_levels")
    if not blocks:
        pytest.skip(f"no merged_levels blocks for {listing}")
    df = io.load_block(blocks[0], listing, "merged_levels")
    assert (df["ask_prc"] >= df["bid_prc"]).all()            # the un-crossed book is never crossed
    r2 = _next_snapshot_r2(blocks[0], listing)
    assert r2 == pytest.approx(published_r2, abs=0.05)        # reproduce the positive lift
    assert r2 > 0.2


def test_merged_levels_blocked_for_bin_perp_only():
    """Only bin PERP is disabled (sub-ms feed, merge hurts). bin SPOT is slower and
    the merge is ~neutral, so it is allowed."""
    assert io._merged_levels_blocked("bin_eth_usdt_p")        # bin perp: blocked
    assert not io._merged_levels_blocked("bin_eth_usdt")      # bin spot: allowed
    assert not io._merged_levels_blocked("byb_eth_usdt_p")
    assert not io._merged_levels_blocked("okx_eth_usdt")
    if io.DATA_DIR is not None:                               # load_block needs DATA_DIR to reach the guard
        assert io.list_blocks("bin_eth_usdt_p", "merged_levels") == []
        assert io.list_blocks("bin_eth_usdt", "merged_levels") != []      # spot allowed
        with pytest.raises(ValueError, match="disabled for"):
            io.load_block("holocron.x.0", "bin_eth_usdt_p", "merged_levels")


def test_venue_market():
    assert io._venue_market("bin_eth_usdt_p") == "bin_perp"
    assert io._venue_market("bin_eth_usdt") == "bin_spot"
    assert io._venue_market("byb_doge_usdt_p") == "byb_perp"
    assert io._venue_market("okx_btc_usdt") == "okx_spot"


def test_trade_lifts_ask_convention():
    agg = np.array(["Bid", "Ask", "Bid", "Ask"])
    for L in ("byb_eth_usdt_p", "okx_eth_usdt", "bin_eth_usdt_p", "bin_btc_usdt_p"):
        assert not io._aggressor_inverted(L)
        assert np.array_equal(io._trade_lifts_ask(L, agg), agg == "Bid")   # standard: 'Bid' lifts ask
    for L in ("bin_eth_usdt", "bin_btc_usdt", "bin_doge_usdt"):           # Binance spot (default setting)
        assert io._aggressor_inverted(L)
        assert np.array_equal(io._trade_lifts_ask(L, agg), agg == "Ask")


def test_aggressor_convention_is_setting_driven(monkeypatch):
    # the convention comes from the setting, not hardcoded logic
    agg = np.array(["Bid", "Ask"])
    monkeypatch.setattr(io, "_AGGRESSOR_INVERTED", {"byb_perp"})
    assert io._aggressor_inverted("byb_eth_usdt_p")
    assert np.array_equal(io._trade_lifts_ask("byb_eth_usdt_p", agg), agg == "Ask")
    assert not io._aggressor_inverted("bin_eth_usdt")                     # no longer inverted under this set


def _bid_aggressor_signal(fl: pl.DataFrame, td: pl.DataFrame) -> float:
    """Empirical: mean (trade-mid)/halfspread for 'Bid'-aggressor trades; >0 standard, <0 inverted."""
    s_rx = fl["rx_time"].cast(pl.Int64).to_numpy()
    bid = fl["bid_prc"].to_numpy(); ask = fl["ask_prc"].to_numpy(); mid = (bid + ask) / 2.0
    t_rx = td["rx_time"].cast(pl.Int64).to_numpy(); tp = td["prc"].to_numpy(); agg = td["aggressor"].to_numpy()
    j = np.searchsorted(s_rx, t_rx, "right") - 1; ok = j >= 0; jj = j[ok]
    hs = np.where((ask[jj] - bid[jj]) > 0, (ask[jj] - bid[jj]) / 2.0, np.nan)
    return float(np.nanmean(((tp[ok] - mid[jj]) / hs)[agg[ok] == "Bid"]))


@pytest.mark.skipif(not _HAS_DATA, reason="DATA_DIR not configured")
@pytest.mark.parametrize("listing, inverted", [
    ("byb_eth_usdt_p", False),     # standard
    ("okx_eth_usdt", False),       # standard spot
    ("bin_eth_usdt", True),        # Binance spot: inverted
])
def test_trade_lifts_ask_agrees_with_data(listing, inverted):
    """The structural aggressor rule must match the empirical signal on a real block —
    so a future capture-convention change is caught, not silently mis-merged."""
    blocks = io.list_blocks(listing, "front_levels")
    if not blocks:
        pytest.skip(f"no blocks for {listing}")
    blk = blocks[0]
    fl = (io.load_block(blk, listing, "front_levels").select("rx_time", "bid_prc", "ask_prc")
          .drop_nulls().sort("rx_time"))
    td = (io.load_block(blk, listing, "trade").select("rx_time", "aggressor", "prc", "qty")
          .filter((pl.col("prc") > 0) & (pl.col("qty") > 0)).drop_nulls().sort("rx_time"))
    assert (_bid_aggressor_signal(fl, td) < 0) == inverted                  # data agrees with the rule
    agg = td["aggressor"].to_numpy()
    assert np.array_equal(io._trade_lifts_ask(listing, agg), agg == ("Ask" if inverted else "Bid"))


@pytest.mark.skipif(not _HAS_DATA, reason="DATA_DIR not configured")
def test_merged_levels_bin_spot_builds():
    """bin spot is allowed (only bin perp is blocked) and must build correctly under the
    inverted aggressor convention."""
    listing = "bin_eth_usdt"
    blocks = io.list_blocks(listing, "merged_levels")
    assert blocks, "bin spot should be buildable (not blocked)"
    df = io.load_block(blocks[0], listing, "merged_levels")
    assert df.columns == ["rx_time", "bid_prc", "ask_prc", "bid_exchange_time", "ask_exchange_time"]
    assert (df["bid_prc"] > 0).all() and (df["ask_prc"] > 0).all()
    assert df["rx_time"].n_unique() == len(df)


# ── tick sizes (tick_sizes.toml) ─────────────────────────────────────────────

_CONFIGURED_LISTINGS = sorted(io._TICK_SIZES)               # every listing in tick_sizes.toml (auto-covers new ones)


def test_tick_size_config_loads_and_raises_on_unknown():
    assert io.tick_size("byb_eth_usdt_p") == 0.01           # ETH: $0.01 everywhere
    assert io.tick_size("okx_btc_usdt_p") == 0.1            # BTC: $0.1 …
    assert io.tick_size("bin_btc_usdt") == 0.01             # … except binance SPOT BTC (finer $0.01)
    with pytest.raises(KeyError):                            # unconfigured listing -> raise (don't guess a tick)
        io.tick_size("byb_sol_usdt_p")


@pytest.mark.skipif(not _HAS_DATA, reason="DATA_DIR not configured")
@pytest.mark.parametrize("listing", _CONFIGURED_LISTINGS)
def test_tick_size_matches_inferred_from_data(listing):
    """Validate EVERY configured tick against the data THREE independent ways, so it's verified not
    guessed (magnitude-robust: tolerances scale with the price / tick, since a $95k BTC price carries
    more float noise than a $3.5k ETH one):
      (1) every price is an integer multiple of it — all quotes live on the tick grid (the strongest);
      (2) it equals the minimum gap between distinct prices — it is the finest resolution seen;
      (3) it equals the minimum positive bid-ask spread — the book reaches exactly one tick wide."""
    blk = io.list_blocks(listing, "front_levels")[0]
    fl = io.load_block(blk, listing, "front_levels").select("bid_prc", "ask_prc").drop_nulls()
    bid, ask = fl["bid_prc"].to_numpy(), fl["ask_prc"].to_numpy()
    tick = io.tick_size(listing)
    px = np.unique(np.concatenate([bid, ask]))

    # (1) every price sits on the tick grid: price / tick is an integer (to magnitude-relative float noise)
    assert np.all(np.abs(px - np.round(px / tick) * tick) < np.abs(px) * 1e-9 + 1e-9), "a price is off the tick grid"
    # (2) the tick is the finest resolution: the min gap between distinct prices == tick
    d = np.diff(np.sort(px)); d = d[d > tick * 0.5]         # drop any float-dup near-zero gaps
    assert abs(float(np.min(d)) - tick) < tick * 1e-4
    # (3) the book reaches exactly one tick wide: the min positive bid-ask spread == tick
    sp = (ask - bid); sp = sp[sp > tick * 0.5]
    assert abs(float(np.min(sp)) - tick) < tick * 1e-4
