"""Feature interface contracts for the screening pipeline.

A *feature* is two implementations of the same maths plus the metadata the generic screening
engines (parity, family sweep, gates) need to drive it with NO per-feature scaffolding:

  - a VECTORIZED builder : (ctx, params) -> {leg_key -> feature vector on the anchor grid}
  - a STREAMING factory  : (ctx, params) -> a StreamingFeature whose value() returns
                           {leg_key -> scalar}, the SAME keys as the vectorized dict.

The two must agree to floating-point round-off on a real block — the parity driver enforces it.
Bundle them in a `FeatureSpec` and `register()` it; the screening notebook and the test suite then
drive every feature through the same engines in `boba.research.screening`.

The EMA-type and inject/decay rules your two implementations MUST follow are in `AUTHORING.md`
(this directory) — the implementation contract; mis-following them fails the parity check.

Design choices locked here (see the screening refactor discussion):
  - `params` is an OPAQUE token (a plain-EMA feature uses `N`; a fast/slow feature uses
    `(n_fast, n_slow)`). The engines use it only as a dict key and never destructure it, so one set
    of generic code serves any param arity.
  - The yardstick vectors a feature divides by live on the `ScreeningContext` (computed once in
    Step 0); they are NOT passed as separate builder args.
  - A feature owns its fan-out: the builder/streaming `keys` are feature-defined (one key per
    foreign source for a cross-source feature; a single key for a single-source feature).

STUBS: signatures + contracts only. Bodies raise `NotImplementedError` until the working code is
extracted from `notebooks/features_v2`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Protocol, runtime_checkable

import numpy as np

# A feature-family member is identified by an OPAQUE params token. Examples: a plain EMA -> `N`
# (int); a fast/slow EMA -> `(n_fast, n_slow)` (tuple). Engines treat it only as a hashable key.
Params = Any


# --------------------------------------------------------------------------------------------------
# Raw events fed to a streaming feature. ONE record per event carries EVERY property the raw stream
# holds, so `on_book` / `on_trade` take a single `ev` and their signature never changes when a new
# feature needs another field — add the field here (and to the raw stream) instead. A feature reads
# only the props it uses (`ev.bid`, `ev.ask_qty`, `ev.px`, …); the rest cost nothing.
# --------------------------------------------------------------------------------------------------
class BookEvent(NamedTuple):
    listing: str
    exch_time: int
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float


class TradeEvent(NamedTuple):
    listing: str
    exch_time: int
    px: float
    lifts_ask: float
    qty: float


@runtime_checkable
class StreamingFeature(Protocol):
    """O(1)-per-event *production* build of one feature-family member.

    The generic parity driver feeds raw events in receive-time order, calls `refresh()` once per
    receive-timestamp, and reads `value()` at each grid anchor. State must be O(1) (scalar EMAs
    only) and built ONLY from `boba.ema` classes (`KernelMeanEMA` / `LiveFrontEMA`) — never a
    hand-rolled EMA, never `EventEMA` directly, never `lfilter`. Compose `boba.research.screening.
    LiveYardstick` rather than recomputing a yardstick, so yardstick parity is established once.

    Contract the driver relies on:
      keys          -- the leg keys; MUST equal the vectorized builder's dict keys for the same
                       `params` (the driver asserts this before driving).
      fuse_trades   -- the listings whose mid folds in trades (merged-levels); the rest are
                       book-only. The driver uses this only to know the feature exists; the class
                       applies the policy itself in on_book/on_trade.
      on_book/on_trade -- mutate internal state for ONE raw event, given the full event record
                       (`BookEvent` / `TradeEvent`). Read only the props you need. Do NOT advance the
                       clock or read.
      refresh()     -- called ONCE per receive-timestamp, after all that timestamp's events are
                       applied: update live fronts / inject flows, then advance the trade clock AT
                       MOST ONCE (iff a trade landed this timestamp). Same-timestamp events are one
                       update / one decay.
      value()       -- {leg_key -> scalar}: the live-front feature at the instant it is read
                       (after every book update up to the anchor, never frozen at the last trade).
    """

    keys: tuple[str, ...]
    fuse_trades: frozenset[str]

    def on_book(self, ev: BookEvent) -> None: ...
    def on_trade(self, ev: TradeEvent) -> None: ...
    def refresh(self) -> None: ...
    def value(self) -> dict[str, float]: ...


class VectorizedBuilder(Protocol):
    """Offline (array) build of one feature-family member — the analysis path.

    `(ctx, params) -> {leg_key -> feature vector}`, each vector length `len(ctx.anchor_ts)`.

    Causal: each row uses only data at-or-before its anchor (NaN where the feature is undefined,
    e.g. before a source has quoted — the gates mask non-finite rows). The yardstick vectors the
    feature divides by are on `ctx` (`ctx.sigma_at_anchor` / `ctx.lam_at_anchor`). Keys MUST match
    the streaming feature's `keys` for the same `params`. May use `lfilter` / any offline trick —
    this is NOT the production path (the streaming class is), and the parity check ties them.
    """

    def __call__(self, ctx: "ScreeningContext", params: Params) -> dict[str, np.ndarray]: ...


# A factory returning a FRESH streaming instance for one (ctx, params). The parity driver builds
# one per params token and drives them all through the event stream in a single pass.
StreamingFactory = Callable[["ScreeningContext", Params], StreamingFeature]


@dataclass(frozen=True)
class FeatureSpec:
    """Everything the generic engines need to screen one feature, with no per-feature glue.

    name           -- registry key / display name.
    vectorized     -- the offline builder.
    make_streaming -- factory returning a fresh `StreamingFeature` for `(ctx, params)`.
    keys_for       -- the leg keys for a given `(ctx, params)`; lets the engines validate that the
                      vectorized and streaming keys agree before driving (cheap fail-fast).
    mirror         -- how a built feature-leg vector reflects under the *mirror augmentation* (the
                      reflection of the tape through byb's mid; see AUTHORING.md → Mirror augmentation).
                      A callable `vec -> reflected vec`: `np.negative` for a feature ODD in byb's mid (a
                      signed gap / imbalance — the common case), or a function returning the vector
                      unchanged for an EVEN feature. The selection/screening engines reflect each feature
                      leg through this and negate the signed target to score a direction-free IC. `None`
                      (the default) means the feature has not declared its reflection and may NOT be
                      mirror-augmented — the engines simply skip it.
    """

    name: str
    vectorized: VectorizedBuilder
    make_streaming: StreamingFactory
    keys_for: Callable[["ScreeningContext", Params], tuple[str, ...]]
    mirror: Callable[[np.ndarray], np.ndarray] | None = None


# --------------------------------------------------------------------------------------------------
# Registry — the notebook drives a feature by name; the test suite parametrizes over all of them.
# --------------------------------------------------------------------------------------------------
_REGISTRY: dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> FeatureSpec:
    """Register `spec` under `spec.name` (raising on a duplicate). Returns it for convenience."""
    if spec.name in _REGISTRY:
        raise ValueError(f"feature {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> FeatureSpec:
    """Look up a registered feature by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no feature registered as {name!r}; have {sorted(_REGISTRY)}") from None


def all_specs() -> tuple[FeatureSpec, ...]:
    """Every registered feature — `tests/` parametrizes parity + gate checks over these."""
    return tuple(_REGISTRY.values())


if False:  # typing-only forward ref; avoids a runtime import cycle with boba.research.screening
    from boba.research.screening import ScreeningContext  # noqa: F401
