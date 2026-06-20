# EMA on the merged trade event clock

How moving-average features are computed across listings on a shared event clock.
This is the agreed foundation for the feature set in `nn_features.md`; read that
first for the clock/window definitions and the `IFeatureBuilder` interface.

The whole scheme is one idea: **pick a single monotonic clock, make every moving
average an exponential kernel over that clock, and realise it with an O(1)
gap-update recursion.** No event buffers — the key practicality constraint.

---

## 1. The clock

Let `c` be the **merged-trade counter**: it increments by 1 on every trade from
any listing in `L`, processed in arrival (`rx_time`) order so research and
production see the same sequence. `c` is the only notion of "time" the EMAs use.

- **Only trades advance `c`.** `front_levels` and `funding` events update *held
  level state* (current mid, spread, funding) but do **not** tick the clock.
- **Consequence — levels are sampled at trade instants.** Between two trades the
  clock is frozen, so a `front_levels` update only matters as the level seen at
  the *next* trade tick. This is the correct and consistent reading of a *trade*
  clock; sub-trade book dynamics are intentionally invisible to these features.

The same machinery works verbatim with `c = rx_time_ns` if a wall-clock sibling
is ever wanted — only the meaning of "one tick" changes.

---

## 2. The EMA kernel (gap-power, O(1))

For a target window `N` (in merged-clock ticks) use the repo convention

```
α = 2 / (N + 1)          λ = 1 − α = (N − 1)/(N + 1)
```

Every EMA is the exponential kernel `E(c) = Σ_i α · λ^(c − c_i) · x_i`, realised
by a recursion that runs **only when an event arrives** and folds in the elapsed
clock gap `Δ = c − c_last` via `λ^Δ`. State per `(feature, N)` is just
`(E, c_last)` — no buffer.

`λ^Δ` is computed exactly as `exp(Δ · ln λ)` (do **not** use `exp(−Δ/τ)`; that
continuous-time shorthand is only valid for small α / large N).

When `L` is a single listing and `c` is that listing's own trades, `Δ ≡ 1` and
this reduces exactly to the existing `α = 2/(span+1)` lfilter EMA — so it is a
strict generalisation of the current convention.

---

## 3. Two flavours: flow vs level

Pick the flavour by the quantity's nature:

**Flow** — additive quantities (volume, trade counts, squared returns). Decay
toward 0 between events:

```
on event with payload x:   E ← λ^Δ · E + α · x
                           W ← λ^Δ · W + α          # accumulated weight
```

`E` estimates intensity (payload per merged event); `W` is the accumulated
weight (see §4). On merged ticks where this series has no event, only the decay
applies — handled lazily by the next `λ^Δ`.

**Level** — stock quantities (mid, spread, microprice, imbalance, funding).
Decay toward the held level `L`:

```
on any tick (or at read time):   E ← λ^Δ · E + (1 − λ^Δ) · L
                                 W ← λ^Δ · W + (1 − λ^Δ)
```

`E` is the time-(clock-)weighted average level; `W → 1` once warm.

---

## 4. The rate / intensity / average decomposition

The flow flavour yields three features from one object. Maintain a **global**
weight `W_full` (a flow EMA with `x = 1` updated on *every* merged tick,
`W_full → 1` warm) and, per `(listing, N)`, the listing's accumulator `E` and
weight `W` (`W → p`, the listing's share of merged flow). Then:

| feature | expression | meaning |
|---|---|---|
| **event rate** | `W / W_full` | listing's share of merged activity |
| **intensity** | `E / W_full` | payload per merged event |
| **average** | `E / W` | payload per *listing* event (clock-rate-independent) |

and they compose exactly: `intensity = rate × average`. The rate feature is
literally the normaliser of the volume feature. Volatility is the same machine
with `x = r²` (squared mid log-returns sampled at trade ticks): `E/W_full` =
variance intensity, `E/W` = per-trade variance, `√` for vol.

Optimisation: all flow features of the same `(listing, N)` share one `W` (same
events, α, clock) — store only a per-feature `E`.

---

## 5. Debiasing & cold start

The ratios in §4 are **self-debiasing**: numerator and denominator share the same
decay and cold-start, so `E/W`, `E/W_full`, `W/W_full` are unbiased from the
first event. Use them rather than raw `E`. For a level feature, the debiased
level is `E / W` (equals `E` once `W ≈ 1`).

Cold-start bias therefore lives only in the *raw* accumulators and is removed by
always reading through a ratio. Warmup length is ~N events, so it dissipates fast
at small N and is the only place large N pays a (one-time) cost.

---

## 6. Small-N behaviour & the minority-venue caveat

The soft-window approximation stays faithful down to small N: center of mass
`(N−1)/2`, ~86% of weight inside the last N, and **effective sample size
`n_eff = N`** all hold at N = 10 just as at N = 10000 (see Appendix). So EMA(N)
is *not* a worse estimator than a literal N-trade window at small N — it has the
same `n_eff` and degrades more gracefully (no empty-window when a venue is
briefly silent).

The real small-N risk is **under-sampling of a minority venue**: a 10-merged-trade
window holds only ~`N · share` of that venue's own trades (~3 byb trades if byb is
⅓ of flow). That is a property of the *question*, not the EMA, and a literal
window suffers it equally (and worse — `0/0`). Two second-order effects:

- The raw intensity `E` is modulated by *which* venue just ticked (spiky). The
  `E/W` average removes that, but `W` (minority rate) can be small at small N, so
  guard the division with **shrinkage** toward the next-coarser scale:
  `avg ≈ (E + κ · avg_coarse) / (W + κ)` for a small pseudo-count `κ`.
- No estimator manufactures samples that are not there — small-N minority
  features are intrinsically noisy.

**Chosen design (option 1).** Keep a **single merged clock everywhere**, read all
flow features through the `E/W` ratios with small-`W` shrinkage, and expose **all
scales** `N ∈ {10, 100, 1000, 10000}` to the model so it can down-weight noisy
small-N per-listing inputs itself (their noise is decorrelated from the large-N
estimate). The alternative — per-listing features on each listing's *own* trade
clock at the smallest scales (option 2) — roots out the under-sampling but breaks
the single-clock simplicity; defer it unless small-N minority features prove
unusable in practice.

---

## 7. Streaming interface (`IFeatureBuilder`)

```
on_trade(listing, e):        c += 1
                             W_full.add(c, 1)
                             update flow EMAs for `listing`  (volume, r², counts …)
                             refresh held mid/spread level for `listing`
on_front_levels(listing,e):  set held level for `listing`        # does NOT tick c
on_funding(listing, e):      set held funding level for `listing`
emit():                      for every EMA: decay-to-current-c, return debiased value
```

`emit()` decays each EMA from its `c_last` to the current `c` (toward 0 for flow,
toward the held level for level) so all features are reported as-of the same clock
position even when a listing has been quiet — without any per-tick loop.

---

## 8. Reference implementation

```python
import math

class ScalarEMA:
    """EMA on an externally supplied monotonic clock `c` (the merged-trade count).

    Decay is keyed to the clock via lam**(gap), so updates are O(1) and need only
    the last value + last clock position -- no event buffer. kind = "flow"
    (decay toward 0; additive payloads) or "level" (decay toward a held level).
    """
    def __init__(self, N, kind="flow"):
        self.alpha = 2.0 / (N + 1.0)
        self.lam   = 1.0 - self.alpha
        self.ln_lam = math.log(self.lam)
        self.kind  = kind
        self.e = 0.0          # un-normalised accumulator
        self.w = 0.0          # accumulated weight (rate / debiasing)
        self.level = 0.0      # held level (kind="level")
        self.c_last = None

    def _decay_to(self, c):
        if self.c_last is None:
            self.c_last = c
            return
        d = c - self.c_last
        if d <= 0:
            return
        f = math.exp(d * self.ln_lam)            # lam**d, exact
        if self.kind == "flow":
            self.e *= f
            self.w *= f
        else:                                    # decay toward held level
            self.e = f * self.e + (1.0 - f) * self.level
            self.w = f * self.w + (1.0 - f)
        self.c_last = c

    def add(self, c, x=1.0):                      # flow: a relevant event arrived
        self._decay_to(c)
        self.e += self.alpha * x
        self.w += self.alpha

    def set_level(self, L):                       # level: book/funding state changed
        self.level = L

    # ---- debiased reads (pass the global full-weight EMA for rate/intensity) ----
    def average(self, c, w_floor=1e-9):           # payload per listing-event / level
        self._decay_to(c)
        return self.e / self.w if self.w > w_floor else 0.0

    def rate(self, c, w_full):                     # share of merged activity
        self._decay_to(c)
        return self.w / w_full if w_full > 0 else 0.0

    def intensity(self, c, w_full):                # payload per merged event
        self._decay_to(c)
        return self.e / w_full if w_full > 0 else 0.0
```

`w_full` is a single shared `ScalarEMA(N, "flow")` fed `add(c, 1.0)` on every
merged trade.

---

## 9. Design decisions (settled)

1. **One merged trade clock**, ticking on trades only, ordered by `rx_time`.
2. **Levels sampled at trade ticks**; `front_levels`/`funding` set held state,
   don't tick the clock.
3. **`α = 2/(N+1)`**, exact `λ^Δ` gap update, O(1) state, no buffers.
4. **Flow vs level** flavours; rate/intensity/average via the `E`,`W`,`W_full`
   ratios; **read through ratios** (self-debiasing).
5. **Small-N minority features**: single clock + `E/W` shrinkage + expose all
   scales (option 1). Option 2 (per-listing own clock at small N) deferred.
6. Scales `N ∈ {10, 100, 1000, 10000}`.

---

## 10. Validation requirement

Per the project rule (`CLAUDE.md`), every feature built on this lands with a blind
oracle. For the EMA core that means: a dead-simple reference that materialises the
full kernel `E(c) = Σ α λ^(c−c_i) x_i` by an explicit sum over events (no
recursion) must match the O(1) `ScalarEMA` to float tolerance **on a real block**
(real blocks have the irregular, multi-venue interleaving that synthetic fixtures
can't exercise). Flow and level flavours, and the `E/W` ratios, each get the trio:
blind oracle + synthetic property tests + a real-block diff.

---

## Appendix — properties used above

For `α = 2/(N+1)`, `λ = 1 − α`:

- **Center of mass** of the weights `α λ^k` is `λ/α = (N−1)/2` — identical to an
  N-length boxcar. (Matched first moment ⇒ "EMA(N) ≈ last N trades".)
- **Weight inside the last N**: `Σ_{k<N} α λ^k = 1 − λ^N ≈ 1 − e⁻² ≈ 86%`
  (already true by N≈10). The remaining ~14% is an exponential tail, with
  ~98% inside 2N and ~99.8% inside 3N.
- **Effective sample size**: `n_eff = (Σw)²/Σw² = (2−α)/α = N` — the same as an
  N-length SMA, at every N.
