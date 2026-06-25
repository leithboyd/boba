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


# --- span == 1 (alpha == 1): the extreme "no memory / no smoothing" case the notebooks' n_fast=1 leg uses ---
# span=1 -> alpha = 2/(1+1) = 1, so the geometric kernel (1-alpha)^j is 1 at j=0 and 0 for every j>=1:
# only the current epoch survives. These pin that behavior in the tested library.

def test_kernelmean_span1_only_current_epoch_then_resets():
    # FLOW, alpha=1: value() is the weight-weighted mean of marks added SINCE the last tick; a tick
    # decays E,W by (1-alpha)=0, i.e. a full reset -> nothing survives the clock step.
    f = KernelMeanEMA(1)
    assert f.alpha == 1.0
    f.add(3.0, 1.0); f.add(5.0, 3.0)                       # two weighted marks in this epoch
    assert math.isclose(f.value(), (3 * 1 + 5 * 3) / (1 + 3), abs_tol=1e-12)   # weighted mean of THIS epoch only
    f.tick()                                               # decay by 0 -> E=W=0
    assert math.isnan(f.value())                           # nothing remembered past a tick
    f.add(7.0)
    assert math.isclose(f.value(), 7.0, abs_tol=1e-12)     # only the fresh epoch counts


def test_livefront_span1_reads_latest_no_smoothing():
    # LEVEL, alpha=1: value() == the freshest observation once committed; no memory of prior values,
    # and the live front tracks a book update between trades exactly (no smoothing).
    lf = LiveFrontEMA(1)
    lf.add(2.0); lf.tick()
    assert math.isclose(lf.value(), 2.0, abs_tol=1e-12)
    lf.add(9.0); lf.tick()
    assert math.isclose(lf.value(), 9.0, abs_tol=1e-12)    # the committed 2.0 is fully forgotten
    lf.add(4.0)                                            # a fresh book update between trades (no tick)
    assert math.isclose(lf.value(), 4.0, abs_tol=1e-12)    # live front == latest, not the last committed 9.0


def test_livefront_span1_nan_before_first_tick():
    # warm-up asymmetry: even with alpha=1 and a value added, value() is nan until the first commit
    # ((1-1)*nan + 1*x -> nan). The notebooks' offline path neutralises this with a 0-fill; pin it here.
    lf = LiveFrontEMA(1)
    assert math.isnan(lf.value())                          # nothing added or committed
    lf.add(5.0)
    assert math.isnan(lf.value())                          # committed still nan -> live front nan
    lf.tick()
    assert math.isclose(lf.value(), 5.0, abs_tol=1e-12)    # first commit -> finite, == latest


def test_eventema_span1_is_passthrough():
    # the composed primitive at alpha=1: step(v) -> value() == v (no memory)
    e = EventEMA(1)
    assert e.alpha == 1.0
    for v in (2.0, -3.0, 11.0):
        e.step(v)
        assert math.isclose(e.value(), v, abs_tol=1e-12)
