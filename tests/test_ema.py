"""KernelMeanEMA validated against dead-simple references (plain loop / lfilter / closed form)."""
import math

import numpy as np
from scipy.signal import lfilter

from boba.ema import KernelMeanEMA


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
