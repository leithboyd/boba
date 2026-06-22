"""KernelMeanEMA validated against dead-simple references (plain loop / lfilter / closed form)."""
import math

import numpy as np
from scipy.signal import lfilter

from boba.ema import KernelMeanEMA, EventEMA, LiveFrontEMA


def _ewma_ref(x, span):
    a = 2.0 / (span + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], np.asarray(x, float))


def test_mark_mean_matches_lfilter_ratio():
    # FLOW use (volatility): E/W = EWMA(r^2)/EWMA(count) = per-move mean r^2
    rng = np.random.default_rng(1)
    moved = rng.random(4000) < 0.3
    rets = rng.standard_normal(4000) * moved
    span = 50
    E, W = _ewma_ref(rets**2, span), _ewma_ref(moved.astype(float), span)
    ref = E / np.where(W > 0, W, np.nan)
    f = KernelMeanEMA(span)
    out = []
    for r, m in zip(rets, moved):
        f.tick()                       # one clock event (trade): decay
        if m:
            f.add(r * r)               # a move: inject, weight 1
        out.append(f.value())
    ok = np.isfinite(ref)
    assert np.allclose(np.array(out)[ok], ref[ok], rtol=1e-9, atol=1e-12)


def test_w1_every_step_is_debiased_ewma():
    # degenerate w=1, mark every step == bias-corrected EWMA == lfilter / (1-(1-a)^t)
    x = np.random.default_rng(0).standard_normal(2000)
    span = 30
    a = 2.0 / (span + 1.0)
    ref = _ewma_ref(x, span) / (1.0 - (1.0 - a) ** np.arange(1, len(x) + 1))
    f = KernelMeanEMA(span)
    out = []
    for v in x:
        f.tick(); f.add(v)
        out.append(f.value())
    assert np.allclose(out, ref, rtol=1e-9, atol=1e-12)


def test_weighted_mean():
    # E/W is the weight-weighted mean Σ w y / Σ w (no ticks -> no decay; alpha cancels exactly)
    f = KernelMeanEMA(100)
    data = [(2.0, 1.0), (4.0, 3.0), (10.0, 1.0)]
    for y, w in data:
        f.add(y, w)
    assert math.isclose(f.value(), sum(y * w for y, w in data) / sum(w for _, w in data), abs_tol=1e-12)


def test_dwell_weight_equals_repeated_injection():
    # LEVEL use: add(gap, dwell) within one interval == re-injecting the held gap `dwell` times
    f1, f2 = KernelMeanEMA(100), KernelMeanEMA(100)
    f1.add(2.0, 3.0); f1.add(8.0, 1.0)          # gap 2 held 3 updates, gap 8 held 1
    for _ in range(3): f2.add(2.0)              # re-inject the held gap per update
    f2.add(8.0)
    assert math.isclose(f1.value(), f2.value(), abs_tol=1e-12)
    assert math.isclose(f1.value(), (2 * 3 + 8 * 1) / 4, abs_tol=1e-12)


def test_held_flat_between_events():
    # no injection -> value() unchanged across clock ticks (decay cancels in E/W). The property
    # that makes reading between ticks well-posed.
    f = KernelMeanEMA(10)
    f.add(4.0)
    held = f.value()
    for _ in range(100):
        f.tick()
        assert math.isclose(f.value(), held, rel_tol=0.0, abs_tol=1e-15)


def test_nan_before_first_event():
    f = KernelMeanEMA(10)
    assert math.isnan(f.value())
    for _ in range(5):
        f.tick()
    assert math.isnan(f.value())               # decaying nothing is still nothing


def test_event_ema_matches_lfilter():
    # the basic single-scalar EMA == standard lfilter (y[-1]=0 convention)
    x = np.random.default_rng(2).standard_normal(3000)
    span = 40
    e = EventEMA(span)
    out = []
    for v in x:
        e.step(v); out.append(e.value())
    assert np.allclose(out, _ewma_ref(x, span), rtol=1e-12, atol=1e-12)
    assert math.isnan(EventEMA(10).value())    # nan before first step


def test_livefront_value_is_live_front():
    # add() per book update only refreshes the front; tick() commits the value-at-trade into a plain
    # EMA. value() = (1-a)*committed + a*latest, committed weighted by trades (not book-update churn).
    a = 2.0 / 11.0
    lf = LiveFrontEMA(10)
    ref = EventEMA(10)                                       # reference committed engine
    sched = [([1.0, 1.5], 1.5), ([2.0], 2.0), ([3.0, 2.8, 2.9], 2.9)]   # (book updates, value at trade)
    for updates, vtrade in sched:
        for u in updates:
            lf.add(u)                                       # every book update
        lf.tick()                                           # the trade commits the latest
        ref.step(vtrade)
    assert math.isclose(lf.value(), (1 - a) * ref.value() + a * 2.9, abs_tol=1e-12)


def test_livefront_value_fresh_between_ticks():
    a = 2.0 / 11.0
    lf, ref = LiveFrontEMA(10), EventEMA(10)
    for v in (1.0, 2.0, 3.0):
        lf.add(v); lf.tick(); ref.step(v)
    before = lf.value()
    lf.add(9.0)                                             # a book update after the last trade (no tick)
    assert lf.value() != before                            # read moved with fresh data...
    assert math.isclose(lf.value(), (1 - a) * ref.value() + a * 9.0, abs_tol=1e-12)   # ...committed unchanged


def test_livefront_is_drop_in_for_kernelmean():
    # identical interface: tick(), add(value), value() work the same way on both
    for cls in (KernelMeanEMA, LiveFrontEMA):
        c = cls(10)
        assert math.isnan(c.value())                       # nan before first commit
        c.tick(); c.add(5.0); c.tick()
        assert math.isfinite(c.value())
