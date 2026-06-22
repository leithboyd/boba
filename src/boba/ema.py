"""A self-normalised exponentially-weighted mean on an event clock — one primitive.

`KernelMeanEMA` is the causal **geometric-kernel Nadaraya–Watson estimator**: the kernel-weighted
conditional mean of a marked event stream,

    value = Σᵢ κᵢ wᵢ yᵢ / Σᵢ κᵢ wᵢ ,   κᵢ = (1 − α)^(clock events since i) ,   α = 2/(span + 1)

carried as two accumulators `E = Σ κ w y` and `W = Σ κ w` decayed on the same clock; the read is
`E / W`. The clock is an **event count** (e.g. one step per trade) — operational / transaction time,
*not* seconds. There is deliberately no wall-clock anywhere.

Why this one estimator covers both the cases we used to split into a "flow" and a "level":

  • Sparse FLOW / mark mean (volatility): inject `add(return**2)` on each price move, `tick()` on
    each trade, read `value()` = mean squared-return per move; `sqrt` of that is σ_ev.
  • Forward-filled LEVEL (a cross-venue gap): inject `add(gap)` on each book update, `tick()` on each
    trade, read `value()` = the gap's EMA. Re-injecting the held gap on every update dwell-weights it
    by update count (≈ Eckner's `EMA_last`); pass `add(gap, dwell)` for exact dwell weighting. No held
    state to interpolate, so the read is well-posed *between* trades — no double-count / forget.

Two integrity conditions (from the literature validation):
  1. `E` and `W` decay at the **same** α (they share one clock here) — required for the design-density
     cancellation that makes `E/W` a conditional mean independent of event intensity. Reading the
     intensity-cancellation as "merged-clock ≈ own-clock" holds only at a span matched to the own-event
     rate.
  2. The mean is per-clock-event; between events the common decay cancels in the ratio, so `value()`
     is held flat — correct, and the reason it is safe to read between ticks.

Provenance (all verified against the event-space literature):
  • count = operational/transaction time: Mandelbrot–Taylor 1967; Clark 1973; Ané–Geman 2000 (trade
    count specifically); Easley–López de Prado–O'Hara 2012.
  • the ratio estimator: Nadaraya 1964; Watson 1964; spatstat `Smooth.ppp` (the "Nadaraya–Watson
    smoother" of marks); weighted form = van Lieshout intrinsically-weighted means.
  • the `w = 1` degenerate case = bias-corrected EWMA: Kingma & Ba 2015 (Adam `m/(1−β^t)`); pandas
    `ewm(adjust=True)`.
  • `EMA_last` on a monotone event-count index (the held-level reading we no longer need): Zumbach &
    Müller 2001; Eckner 2017.

Validate any concrete use against a plain one-event-at-a-time loop on a real block, the way
`notebooks/03_ema_clock_validation.ipynb` does.
"""


class KernelMeanEMA:
    """Causal geometric-kernel Nadaraya–Watson estimator (self-normalised EWMA) on an event clock.

    Estimates the kernel-weighted conditional mean `E[mark | recent events]` as `E / W`, with the
    kernel `(1 − α)^j` over the ordinal lag `j` in clock events.

    Methods
    -------
    tick()
        Advance the clock one event (e.g. one trade): decay `E` and `W`. No injection.
    add(value, weight=1.0)
        Register one data event with mark `value` and weight `weight`. The reported mean is
        weight-weighted: `add(r**2)` for a count-weighted move mean (volatility); `add(gap)` per book
        update for a (dwell-by-update-count) level mean; `add(gap, dwell)` for an exact dwell weight.
    value()
        The current `E / W` — the conditional mean (`nan` before the first event). For volatility take
        `sqrt(value())`.

    Notes
    -----
    Starts from 0; `value()` is `nan` until the first `add`. `α` cancels in the ratio, so the absolute
    scale of `add` is irrelevant — only relative weights matter.
    """

    __slots__ = ("alpha", "E", "W")

    def __init__(self, span):
        self.alpha = 2.0 / (span + 1.0)
        self.E = 0.0
        self.W = 0.0

    def tick(self):
        self.E *= (1.0 - self.alpha)
        self.W *= (1.0 - self.alpha)

    def add(self, value, weight=1.0):
        self.E += self.alpha * value * weight
        self.W += self.alpha * weight

    def value(self):
        return self.E / self.W if self.W > 0.0 else float("nan")
