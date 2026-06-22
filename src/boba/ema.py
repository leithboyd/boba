"""Exponentially-weighted means on an event clock — α = 2/(span+1), decayed once per clock event
(e.g. one step per trade — operational/transaction time, never wall-clock).

Three small classes, all sharing the `tick` / `add(value, weight=1.0)` / `value()` interface so any
one drops in for another during feature development:

  • `KernelMeanEMA` — the self-normalising **`E/W`** estimator (causal geometric-kernel
    Nadaraya–Watson conditional mean). Use for a sparse FLOW / mark mean: `add(return**2)` on each
    move → mean squared-return per move (`sqrt` ⇒ σ_ev). `value()` is the committed mean (`EMA_last`).
  • `EventEMA` — a plain single-scalar EMA, one step per event. The basic primitive; no `E/W` ratio.
  • `LiveFrontEMA` — reads a forward-filled LEVEL (a cross-venue gap) with a **live front**: the
    committed mean carried one step toward the freshest value. `add` refreshes that value on every
    book update; the commit (and so the dwell-weighting) stays on the trade clock via `tick`. Because
    it only commits on `tick`, it composes an `EventEMA` internally — `KernelMeanEMA`'s ratio is not
    needed. `value()` is always the live front (`EMA_next`); it is a drop-in for `KernelMeanEMA`, so
    swapping the class is the only change to A/B committed vs live-front.

Provenance (verified against the event-space literature):
  • count = operational/transaction time: Mandelbrot–Taylor 1967; Clark 1973; Ané–Geman 2000;
    Easley–López de Prado–O'Hara 2012.
  • the `E/W` ratio estimator: Nadaraya 1964; Watson 1964; spatstat `Smooth.ppp` mark smoother;
    weighted form = van Lieshout intrinsically-weighted means; `w = 1` ⇒ bias-corrected EWMA
    (Adam `m/(1−β^t)`, pandas `ewm(adjust=True)`).
  • committed vs live-front read = `EMA_last` vs `EMA_next` interpolation (ν dial): Zumbach & Müller
    2001 (NAG g13mec); Eckner 2017. Live front = SES filtered/updated level (Brown 1959/63; Hyndman
    et al.) = the Kalman *filtered* (posterior) estimate vs *predicted* (prior) for the committed read.

Validate any concrete use against a plain one-event-at-a-time loop on a real block, the way
`notebooks/03_ema_clock_validation.ipynb` does.
"""


class KernelMeanEMA:
    """Causal geometric-kernel Nadaraya–Watson estimator (self-normalised `E/W`) on an event clock.

    Estimates `E[mark | recent events] = E / W` as of the last `tick`, kernel `(1 − α)^j` over the
    ordinal lag `j` in clock events.

    tick()                advance the clock one event (e.g. one trade): decay `E` and `W`.
    add(value, weight=1)  inject one data event with mark `value`, weight `weight` (the mean is
                          weight-weighted): `add(r**2)` count-weighted move mean; `add(gap, dwell)`
                          dwell-weighted level mean.
    value()               the committed mean `E / W` (`nan` before the first event); for volatility
                          take `sqrt(value())`.

    `α` cancels in the ratio, so only relative weights matter. Decay lives on whatever clock you `tick`.
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


class EventEMA:
    """A plain single-scalar EMA, one step per clock event (no `E/W` ratio).

    step(value)   fold `value` into the EMA (decay-and-inject in one); the standard recursion
                  `ema = (1 − α)·ema + α·value`, started from 0 (`y[-1] = 0` convention).
    value()       the current EMA (`nan` before the first `step`).
    """

    __slots__ = ("alpha", "ema", "started")

    def __init__(self, span):
        self.alpha = 2.0 / (span + 1.0)
        self.ema = 0.0
        self.started = False

    def step(self, value):
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * value
        self.started = True

    def value(self):
        return self.ema if self.started else float("nan")


class LiveFrontEMA:
    """Forward-filled LEVEL read with a live front — drop-in for `KernelMeanEMA`.

    Same `tick` / `add` / `value` interface, so swapping the class is the only change. It composes a
    plain `EventEMA` (the committed mean) plus the latest observed value, and `value()` reads the
    **live front** — the committed mean carried one step toward the freshest value.

    tick()                a trade: commit the current value into the EMA and advance the clock.
    add(value, weight=1)  a book update: refresh the freshest value. Does NOT inject or decay, so the
                          dwell-weighting stays on the trade clock; `weight` is accepted for interface
                          compatibility and ignored.
    value()               `(1 − α)·committed + α·latest` — the live-front read (`EMA_next`), current
                          between trades, never frozen on the last trade. `nan` until the first commit.
    """

    __slots__ = ("_ema", "_latest")

    def __init__(self, span):
        self._ema = EventEMA(span)
        self._latest = None

    def tick(self):
        if self._latest is not None:
            self._ema.step(self._latest)

    def add(self, value, weight=1.0):
        self._latest = value

    def value(self):
        c = self._ema.value()
        if self._latest is None:
            return c
        a = self._ema.alpha
        return (1.0 - a) * c + a * self._latest      # nan if c is nan (warm-up), else the live front
