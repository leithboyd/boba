"""Tests for boba.features.streaming — the shared streaming primitives composed by book-based features:
`uncross_quote` and `LiveMergedBook` (the online twin of `io._build_merged_levels`). Both expose the
un-crossed `(bid, ask)`; the caller derives mid/spread/…. The end-to-end match with the vectorized io
merge is covered by the per-feature real-block parity tests (test_price_*); here we unit-test the fuse +
un-cross logic on deterministic synthetic events, so it isn't only validated through a feature."""
import math

import numpy as np
import pytest

import boba.io as io
from boba.features.base import BookEvent, FeatureSpec, ParamKind, Series, TradeEvent
from boba.features.shared import _yardsticks
from boba.features.streaming import LiveMergedBook, RateYardstick, VolYardstick, uncross_quote


def test_uncross_quote_scalar():
    assert uncross_quote(100.00, 100.20, 1, 1, 0.01) == pytest.approx((100.00, 100.20))   # not crossed -> unchanged
    assert uncross_quote(100.05, 100.00, 1, 2, 0.01) == pytest.approx((99.99, 100.00))    # crossed, ask fresher -> bid down
    assert uncross_quote(100.20, 100.10, 9, 3, 0.01) == pytest.approx((100.20, 100.21))   # crossed, bid fresher -> ask up
    assert uncross_quote(100.05, 100.00, 1, 1, 0.01) == pytest.approx((99.99, 100.00))    # crossed tie -> ask fresher
    assert uncross_quote(100.05, 100.00, 1, 2, None) == pytest.approx((100.05, 100.00))   # tick None -> raw (no un-cross)


def test_live_merged_book_fuses_newest_by_exch_and_uncrosses():
    L = "byb_x"; tick = 0.01
    bk = LiveMergedBook({L: tick})
    assert bk.quote(L) is None                                  # before any quote
    bk.on_book(BookEvent(L, 1, 1, 100.00, 100.02, 1.0, 1.0))       # snapshot (both sides exch 1)
    assert bk.quote(L) == pytest.approx((100.00, 100.02))
    bk.on_trade(TradeEvent(L, 5, 5, 99.99, 1.0, 1.0))             # a buy lifts the ask at 99.99 (exch 5) -> CROSSED
    assert bk.ask[L] == pytest.approx(99.99) and bk.bid[L] == pytest.approx(100.00)   # raw held state is crossed
    assert bk.quote(L) == pytest.approx((99.98, 99.99))        # quote un-crosses: ask fresher -> bid = ask - tick
    bk.on_book(BookEvent(L, 8, 8, 99.97, 99.99, 1.0, 1.0))        # fresh snapshot (exch 8) -> naturally uncrossed
    assert bk.quote(L) == pytest.approx((99.97, 99.99))


def test_live_merged_book_book_only_listing_is_raw_no_uncross():
    bk = LiveMergedBook({})                                     # no fused listings
    bk.on_book(BookEvent("bin_x", 1, 1, 50.0, 50.2, 1.0, 1.0))
    assert bk.quote("bin_x") == pytest.approx((50.0, 50.2))
    bk.on_trade(TradeEvent("bin_x", 5, 5, 49.0, 1.0, 1.0))        # trades ignored for a book-only listing
    assert bk.quote("bin_x") == pytest.approx((50.0, 50.2))


def test_live_merged_book_same_exch_sweep_holds_latest_print():
    L = "byb_x"; tick = 0.01
    bk = LiveMergedBook({L: tick})
    bk.on_book(BookEvent(L, 1, 1, 99.00, 100.00, 1.0, 1.0))
    for px in (100.01, 100.02, 100.03):                        # a sweep: lift-ask trades at ONE exch_time, in sequence
        bk.on_trade(TradeEvent(L, 5, 5, px, 1.0, 1.0))
    assert bk.ask[L] == pytest.approx(100.03)                  # holds the LATEST at the max exch_time, not the first


# --------------------------------------------------------------------------------------------------
# yardsticks — VolYardstick (σ_ev) vs an independent loop; both VolYardstick + RateYardstick vs the
# vectorized boba.features.shared._yardsticks
# --------------------------------------------------------------------------------------------------
def _sigma_ref(seq, span):
    """σ_ev by an explicit one-event-at-a-time loop: inject (Δlog)^2 on a real move, decay on a trade."""
    a = 2.0 / (span + 1.0)
    E = W = 0.0
    prev = None
    out = []
    for log_mid, traded in seq:
        if log_mid is not None:
            if prev is not None and log_mid != prev:
                E += a * (log_mid - prev) ** 2
                W += a
            prev = log_mid
        if traded:
            E *= (1.0 - a); W *= (1.0 - a)
        out.append((E / W) ** 0.5 if W > 0 else float("nan"))
    return np.array(out)


def test_volyardstick_matches_reference():
    rng = np.random.default_rng(3)
    span = 25
    seq = []                                              # (log_mid|None, traded) per timestamp
    lm = 0.0
    for _ in range(4000):
        if rng.random() < 0.4:
            lm = lm + rng.standard_normal() * 1e-4
        seq.append((lm if rng.random() < 0.9 else None, rng.random() < 0.7))
    y = VolYardstick(span)
    got = []
    for log_mid, traded in seq:
        y.on_target_logmid(log_mid)
        if traded:
            y.tick()
        got.append(y.sigma())
    got, ref = np.array(got), _sigma_ref(seq, span)
    ok = np.isfinite(ref)
    assert ok.sum() > 1000
    assert np.allclose(got[ok], ref[ok], rtol=1e-12, atol=1e-15)


def test_yardsticks_match_vectorized():
    """VolYardstick.sigma() and RateYardstick.lam(), driven event-by-event, reproduce the vectorized
    σ_ev / λ_ev (`_yardsticks`) read at the same points — so both streaming twins match their offline build."""
    rng = np.random.default_rng(0)
    span = 25
    n = 3000
    base = np.arange(1, n + 1) * 100
    rx_mid = (base + 0).astype(np.int64)                  # target mid updates
    clock = (base[:n - 50] + 40).astype(np.int64)         # trade ticks (decay clock); all timestamps distinct
    out_ts = (base[50:n - 60] + 70).astype(np.int64)      # read points
    moved = rng.random(n) < 0.5
    moved[0] = True
    mid = 100.0 * np.exp(np.cumsum(np.where(moved, rng.standard_normal(n) * 1e-4, 0.0)))

    sig_vec, lam_vec = _yardsticks(Series(rx_mid, mid), clock, out_ts, span)   # the vectorized twin

    vol, rate = VolYardstick(span), RateYardstick(span)
    events = sorted([(int(t), 0, i) for i, t in enumerate(rx_mid)]             # 0=mid update
                    + [(int(t), 1, -1) for t in clock]                        # 1=trade tick (decay)
                    + [(int(t), 2, i) for i, t in enumerate(out_ts)])         # 2=read
    sig_str = np.full(len(out_ts), np.nan)
    lam_str = np.full(len(out_ts), np.nan)
    for ts, kind, idx in events:
        if kind == 0:
            lm = math.log(mid[idx]); vol.on_target_logmid(lm); rate.on_target_logmid(lm)
        elif kind == 1:
            vol.tick(); rate.tick(ts)
        else:
            sig_str[idx] = vol.sigma(); lam_str[idx] = rate.lam()

    ok = np.isfinite(sig_str) & np.isfinite(lam_str)      # past warm-up (a move seen; an inter-tick gap seen)
    assert ok.sum() > 1000
    np.testing.assert_allclose(sig_str[ok], sig_vec[ok], rtol=1e-9, atol=1e-12)   # VolYardstick == vectorized σ_ev
    np.testing.assert_allclose(lam_str[ok], lam_vec[ok], rtol=1e-9, atol=1e-12)   # RateYardstick == vectorized λ_ev


# --------------------------------------------------------------------------------------------------
# yardsticks driven THROUGH the generic parity driver — wrap each as a throwaway streaming feature +
# vectorized builder and re-use parity_check (proves both compose as real streaming features; the rate
# yardstick reads the event `rx` for Δt, which the driver now supplies). Not registered (throwaway).
# --------------------------------------------------------------------------------------------------
def _yardstick_spec(which):
    """A single-leg FeatureSpec whose value() is σ_ev ('vol') or λ_ev ('rate') of the target, so the
    generic parity driver can tie the streaming yardstick to the vectorized `_yardsticks`."""

    def vec(raw, shared, config, params):
        sig, lam = _yardsticks(shared.listings[config.target_listing].mid, shared.clock, shared.event_ts, params)
        return {which: sig if which == "vol" else lam}

    class Live:
        def __init__(self, config, params):
            self.keys = (which,)
            fuse_tick = {l: config.tick_size[l] for l in config.all_listings
                         if config.mid_stream.get(l) == "merged_levels"}
            self.book = LiveMergedBook(fuse_tick)             # reconstruct the target's (un-crossed) mid
            self.target = config.target_listing
            self.yard = VolYardstick(params) if which == "vol" else RateYardstick(params)
            self.was_trade = False
            self._rx = 0

        def on_book(self, ev):
            self.book.on_book(ev); self._rx = ev.rx

        def on_trade(self, ev):
            self.book.on_trade(ev); self._rx = ev.rx; self.was_trade = True

        def refresh(self):
            traded, self.was_trade = self.was_trade, False
            q = self.book.quote(self.target)
            self.yard.on_target_logmid(math.log(0.5 * (q[0] + q[1])) if q is not None else None)
            if traded:
                self.yard.tick() if which == "vol" else self.yard.tick(self._rx)   # rate needs the tick's rx

        def value(self):
            return {which: self.yard.sigma() if which == "vol" else self.yard.lam()}

    return FeatureSpec(f"_{which}_yardstick", vec, lambda config, p: Live(config, p),
                       lambda config, p: (which,), param_kind=ParamKind.SINGLE)


@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_yardsticks_via_parity_on_real_block():
    """Drive the wrapped VolYardstick / RateYardstick through the generic parity driver on a real block:
    the streaming yardstick reproduces the vectorized `_yardsticks` — so each composes as a real streaming
    feature (and RateYardstick gets its Δt from the event `rx` the driver now carries)."""
    from boba.research.screening import build_context, parity_check
    ctx = build_context(hours=2)
    for which in ("vol", "rate"):
        rep = parity_check(ctx, _yardstick_spec(which), [ctx.yardstick_span], n_grid=40_000, tol=1e-6)
        assert rep.passed, f"{which}: {rep}"


# --------------------------------------------------------------------------------------------------
# event `rx` — the driver must wire the receive-time, NOT exch_time, into the events (RateYardstick reads
# it for Δt). Synthetic _raw_events elsewhere set exch_time == rx, so they can't catch a swap; here
# exch_time = rx - VARIABLE latency, so the rate-yardstick parity only holds if the driver used rx.
# --------------------------------------------------------------------------------------------------
def _rate_parity_ctx_exch_neq_rx(n=2000):
    from boba.features.base import Config, FrontLevels, ListingRaw, RawData, Trade
    from boba.features.shared import build_shared_data
    from boba.research.screening import RawEventStream, ScreeningContext
    rng = np.random.default_rng(2)
    coin = "x"; tgt = f"byb_{coin}"; span = 20
    base = np.arange(1, n + 1) * 1000
    rx_fl = (base + 0).astype(np.int64)                          # book updates (the target mid)
    rx_tr = (base + 400).astype(np.int64)                        # trades (the decay clock)
    lat_fl = rng.integers(1, 300, n).astype(np.int64)           # VARIABLE feed latency -> exch_time != rx,
    lat_tr = rng.integers(1, 300, n).astype(np.int64)           #   and exch_time GAPS != rx gaps
    moved = rng.random(n) < 0.5; moved[0] = True
    mid = 100.0 * np.exp(np.cumsum(np.where(moved, rng.standard_normal(n) * 1e-4, 0.0)))
    lifts = (rng.random(n) < 0.5).astype(float)
    raw = RawData(listings={tgt: ListingRaw(
        front_levels=FrontLevels(rx_fl, rx_fl - lat_fl, mid, np.ones(n), mid, np.ones(n)),
        trade=Trade(rx_tr, rx_tr - lat_tr, mid, lifts, np.ones(n)),
    )})
    config = Config(tgt, (), coin, {tgt: "front_levels"}, yardstick_span=span)
    shared = build_shared_data(raw, config)                      # vectorized λ_ev built on the rx clock
    rx = np.concatenate([rx_fl, rx_tr])
    kind = np.concatenate([np.zeros(n, np.int8), np.ones(n, np.int8)])
    lid = np.zeros(2 * n, np.int8)
    t = np.concatenate([rx_fl - lat_fl, rx_tr - lat_tr])         # the exch_time column (!= rx, variable gaps)
    a = np.concatenate([mid, mid])                               # book bid / trade px
    b = np.concatenate([mid, lifts])                             # book ask / trade lifts_ask
    c = np.ones(2 * n)                                           # book bid_qty / trade qty
    d = np.concatenate([np.ones(n), np.full(n, np.nan)])         # book ask_qty / trade (unused)
    order = np.lexsort((kind, rx))
    raw_events = RawEventStream(rx[order], kind[order], lid[order], t[order],
                                a[order], b[order], c[order], d[order], (tgt,))
    anchor_ts = rx_fl[100:-100:5]
    return ScreeningContext(
        block="syn", coin=coin, target=tgt, sources=(), horizon_ns=0, yardstick_span=span,
        mid_stream={}, merged_ts=shared.clock, anchor_ts=anchor_ts,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0),
        vol_regime=np.empty(0), raw_events=raw_events, raw_data=raw, shared_data=shared, config=config)


def test_rate_yardstick_parity_uses_event_rx_not_exch_time():
    from boba.research.screening import parity_check
    ctx = _rate_parity_ctx_exch_neq_rx()
    rep = parity_check(ctx, _yardstick_spec("rate"), [20], n_grid=len(ctx.anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)
