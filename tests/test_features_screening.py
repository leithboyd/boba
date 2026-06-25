"""Tests for the generic screening engines (boba.research.screening) + the feature interface.

The generic machinery (registry, the parity driver, the gate orchestration, the shared yardstick) is
exercised on SYNTHETIC data so it runs with no DATA_DIR. A real-block integration test ties the
price_dislocation port together end-to-end when DATA_DIR is set (skipped otherwise)."""
import math

import numpy as np
import pytest

import boba.io as io
from boba.features import base
from boba.features.base import FeatureSpec
import boba.features.price_dislocation  # noqa: F401  (registers the feature)
from boba.research.screening import (
    HeadConfig, LiveYardstick, RawEventStream, ScreeningContext,
    build_family, parity_check, run_gates,
)


# --------------------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------------------
def test_registry_get_and_duplicate():
    spec = base.get("price_dislocation")
    assert spec.name == "price_dislocation"
    assert "price_dislocation" in [s.name for s in base.all_specs()]
    with pytest.raises(KeyError):
        base.get("nope")
    dummy = FeatureSpec("dummy_reg_test", spec.vectorized, spec.make_streaming, spec.keys_for)
    base.register(dummy)
    with pytest.raises(ValueError):           # duplicate name rejected
        base.register(dummy)


# --------------------------------------------------------------------------------------------------
# LiveYardstick vs a dead-simple independent reference (the shared σ_ev streamer)
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
                W += a * 1.0
            prev = log_mid
        if traded:
            E *= (1.0 - a)
            W *= (1.0 - a)
        out.append((E / W) ** 0.5 if W > 0 else float("nan"))
    return out


def test_liveyardstick_matches_reference():
    rng = np.random.default_rng(3)
    span = 25
    # a stream of (log_mid|None, traded) — some timestamps move the mid, some only trade, some neither
    seq = []
    lm = 0.0
    for _ in range(4000):
        moved = rng.random() < 0.4
        if moved:
            lm = lm + rng.standard_normal() * 1e-4
        seq.append((lm if rng.random() < 0.9 else None, rng.random() < 0.7))
    ref = _sigma_ref(seq, span)
    y = LiveYardstick(span)
    got = []
    for log_mid, traded in seq:
        y.on_target_logmid(log_mid)
        if traded:
            y.tick()
        got.append(y.sigma())
    got, ref = np.array(got), np.array(ref)
    ok = np.isfinite(ref)
    assert ok.sum() > 1000
    assert np.allclose(got[ok], ref[ok], rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------------------------------
# the GENERIC parity driver — a trivial live-mid feature on synthetic events (exact, no EMA drift)
# --------------------------------------------------------------------------------------------------
def _live_mid_spec():
    """A throwaway feature: value() = each source's live mid; vectorized = the causal mid at each anchor.
    Exercises the driver, the read-at-anchor timing, fan-out, and key agreement — must match exactly."""

    class LiveMid:
        def __init__(self, ctx, params):
            self.keys = tuple(ctx.sources)
            self.fuse_trades = frozenset()
            self._key = {f"{s}_{ctx.coin}": s for s in ctx.sources}
            self._m: dict = {}

        def on_book(self, listing, exch_time, bid, ask):
            self._m[listing] = 0.5 * (bid + ask)

        def on_trade(self, listing, exch_time, px, lifts_ask):
            pass

        def refresh(self):
            pass

        def value(self):
            return {sh: self._m.get(full, float("nan")) for full, sh in self._key.items()}

    def vec(ctx, params):
        return {s: ctx.mid_at_anchor(s) for s in ctx.sources}

    return FeatureSpec("livemid_test", vec, lambda ctx, p: LiveMid(ctx, p), lambda ctx, p: tuple(ctx.sources))


def _synthetic_event_ctx():
    coin = "x"
    sources = ("aaa", "bbb")
    rx_a = np.array([10, 30, 50, 70, 90], np.int64); mid_a = np.array([100., 101, 102, 103, 104])
    rx_b = np.array([20, 40, 60, 80], np.int64); mid_b = np.array([200., 201, 202, 203])
    mids = {"aaa": (rx_a, mid_a), "bbb": (rx_b, mid_b)}
    rx = np.concatenate([rx_a, rx_b])
    kind = np.zeros(len(rx), np.int8)
    lid = np.concatenate([np.zeros(len(rx_a), np.int8), np.ones(len(rx_b), np.int8)])
    a = np.concatenate([mid_a, mid_b]); b = a.copy(); t = rx.astype(np.int64)
    order = np.lexsort((kind, rx))
    raw = RawEventStream(rx[order], kind[order], lid[order], t[order], a[order], b[order], ("aaa_x", "bbb_x"))
    anchor_ts = np.array([25, 45, 65, 85, 95], np.int64)
    return ScreeningContext(
        block="syn", coin=coin, target="aaa_x", sources=sources, horizon_ns=0, yardstick_span=10,
        mid_stream={"aaa": "front_levels", "bbb": "front_levels"}, merged_ts=np.array([15, 35, 55, 75], np.int64),
        anchor_ts=anchor_ts, tick_at_anchor=np.empty(0), sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0),
        price_target=np.empty(0), rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0),
        vol_regime=np.empty(0), raw_events=raw, _mids=mids)


def test_parity_driver_generic_exact():
    ctx = _synthetic_event_ctx()
    spec = _live_mid_spec()
    rep = parity_check(ctx, spec, [(1, 1)], n_grid=10, tol=1e-12)
    assert rep.passed, str(rep)
    for k in ("aaa", "bbb"):
        assert rep.n_points[((1, 1), k)] == 5
        assert rep.max_diff[((1, 1), k)] == 0.0     # exact: no EMA, integer-clean mids


def test_build_family_keys_and_parallel():
    ctx = _synthetic_event_ctx()
    spec = _live_mid_spec()
    fam = build_family(ctx, spec.vectorized, [(1, 1), (2, 2)], n_jobs=2)
    assert set(fam) == {(1, 1), (2, 2)}
    assert set(fam[(1, 1)]) == {"aaa", "bbb"}
    assert len(fam[(1, 1)]["aaa"]) == len(ctx.anchor_ts)


# --------------------------------------------------------------------------------------------------
# the gate orchestration — signal carries, noise doesn't (synthetic)
# --------------------------------------------------------------------------------------------------
def _gate_ctx(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    pt = rng.standard_normal(n)
    vol_level = rng.standard_normal(n)
    rate_level = rng.standard_normal(n)
    base_controls = [rng.standard_normal(n) * 0.01, rng.standard_normal(n) * 0.01]
    vol_regime = np.digitize(vol_level, np.percentile(vol_level, [33, 67]))
    ctx = ScreeningContext(
        block="syn", coin="x", target="aaa_x", sources=("aaa",), horizon_ns=0, yardstick_span=10,
        mid_stream={}, merged_ts=np.empty(0), anchor_ts=np.empty(0), tick_at_anchor=np.empty(0),
        sigma_at_anchor=np.exp(0.3 * vol_level), lam_at_anchor=np.exp(0.3 * rate_level),
        price_target=pt, rate_target=rng.standard_normal(n), base=base_controls,
        vol_level=vol_level, rate_level=rate_level, vol_regime=vol_regime,
        raw_events=RawEventStream(*([np.empty(0)] * 6), ()))
    return ctx, pt, rng


def test_run_gates_signal_vs_noise():
    ctx, pt, rng = _gate_ctx()
    head = HeadConfig.price(ctx)

    rep_signal = run_gates({"aaa": pt * 0.3 + rng.standard_normal(len(pt))}, ctx, head)
    assert rep_signal.head == "price"
    assert rep_signal.rows[0]["gate"].startswith("B · signal")
    assert rep_signal.rows[0]["value"] > 0.05          # marginal IC clearly positive
    assert any(r["gate"].startswith("A · regime-inv") for r in rep_signal.rows)
    assert rep_signal.passed                            # signal + regime-invariant -> pass

    rep_noise = run_gates({"aaa": rng.standard_normal(len(pt))}, ctx, head)
    assert abs(rep_noise.rows[0]["value"]) < 0.03       # noise carries ~no marginal signal
    assert not rep_noise.passed                          # fails Gate B floor


def test_headconfig_factories():
    ctx, _, _ = _gate_ctx(n=200)
    p, r = HeadConfig.price(ctx), HeadConfig.rate(ctx)
    assert p.name == "price" and not p.score_magnitude
    assert r.name == "rate" and r.score_magnitude
    assert p.coupling_yardstick is ctx.sigma_at_anchor and r.coupling_yardstick is ctx.lam_at_anchor


# --------------------------------------------------------------------------------------------------
# small shared diagnostics — span pick + echo-netting
# --------------------------------------------------------------------------------------------------
def test_best_span_picks_max_ic():
    from boba.research.screening import best_span
    rng = np.random.default_rng(1)
    n = 5000
    target = rng.standard_normal(n)
    ctx, _, _ = _gate_ctx(n=n)
    family = {
        (1, 1): {"aaa": rng.standard_normal(n)},                       # noise
        (2, 2): {"aaa": target * 0.5 + rng.standard_normal(n)},        # strong
        (3, 3): {"aaa": target * 0.1 + rng.standard_normal(n)},        # weak
    }
    assert best_span(ctx, family, target) == (2, 2)


def test_echo_netted_ic_real_vs_echo():
    from boba.research.screening import echo_netted_ic, _ffill
    rng = np.random.default_rng(2)
    n_ticks = 20_000
    rx = (np.arange(1, n_ticks + 1) * 10).astype(np.int64)
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n_ticks) * 1e-4))      # random walk
    H = 1000
    anchors = np.arange(int(rx[H + 10]), int(rx[-1] - H), 50, dtype=np.int64)
    ctx = ScreeningContext(
        block="syn", coin="x", target="byb_x", sources=(), horizon_ns=H, yardstick_span=10,
        mid_stream={}, merged_ts=np.empty(0), anchor_ts=anchors, tick_at_anchor=np.empty(0),
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0),
        vol_regime=np.empty(0), raw_events=RawEventStream(*([np.empty(0)] * 6), ()), _mids={"byb": (rx, mid)})
    fwd = np.log(_ffill(rx, mid, anchors + H) / _ffill(rx, mid, anchors))
    trail = np.log(_ffill(rx, mid, anchors) / _ffill(rx, mid, anchors - H))
    real = echo_netted_ic(ctx, fwd)                 # a leg that IS the forward move: survives netting
    assert real["raw"] > 0.9 and abs(real["backward"]) < 0.1 and real["netted"] > 0.9
    echo = echo_netted_ic(ctx, trail)               # a leg that IS the trailing move: all echo, nets to ~0
    assert echo["backward"] > 0.9 and abs(echo["netted"]) < 0.15


# --------------------------------------------------------------------------------------------------
# real-block integration (skipped without DATA_DIR) — the price_dislocation port end-to-end
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity_and_gates():
    from boba.research.screening import build_context

    ctx = build_context()
    spec = base.get("price_dislocation")
    rep = parity_check(ctx, spec, [(1, 200), (10, 100)], n_grid=50_000, tol=1e-6)
    assert rep.passed, str(rep)                          # streaming reproduces the vectorized build

    legs = build_family(ctx, spec.vectorized, [(10, 100)])[(10, 100)]
    assert set(legs) == set(ctx.sources)
    gr = run_gates(legs, ctx, HeadConfig.price(ctx))
    assert gr.head == "price" and len(gr.rows) > 0
