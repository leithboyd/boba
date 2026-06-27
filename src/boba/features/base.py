"""Feature interface contracts for the screening pipeline.

A *feature* is two implementations of the same maths plus the metadata the generic screening
engines (parity, family sweep, gates) need to drive it with NO per-feature scaffolding:

  - a VECTORIZED builder : (raw_data, shared_data, config, params) -> {leg_key -> vector per event_ts}
  - a STREAMING factory  : (config, params) -> a StreamingFeature whose value() returns
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
  - The yardstick vectors a feature divides by live on `shared_data` (`vol_yardstick` / `rate_yardstick`,
    computed once by `build_shared_data`, one value per `event_ts`); read them there, not as builder args.
  - A feature owns its fan-out: the builder/streaming `keys` are feature-defined (one key per
    foreign source for a cross-source feature; a single key for a single-source feature).

A feature is STANDALONE: it reads only the data CONTRACTS defined at the bottom of this file
(`RawData` / `SharedData` / `Config`, built by `boba.features.shared.build_shared_data`) and imports
nothing from `boba.research`. The vectorized builder emits one value per `shared_data.event_ts`;
choosing where to read it (an eval grid) and what to predict (a target) happen downstream in research.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Protocol

import numpy as np

# A feature-family member is identified by an OPAQUE params token. Examples: a plain EMA -> `N`
# (int); a fast/slow EMA -> `(n_fast, n_slow)` (tuple). Engines treat it only as a hashable key.
Params = Any


class ParamKind(Enum):
    """The shape of a feature's `params` token — declared on `FeatureSpec` so the finalize notebook can
    pick the right span sweep and visualisation WITHOUT a per-feature copy (it must know the shape before
    it can build the family, so this is declared, not inferred):

      SINGLE     params = N (one EMA span)          -> a 1-D scan over N   (`selection.ic_scan`, a line per leg)
      FAST_SLOW  params = (n_fast, n_slow)          -> a 2-D fast/slow grid (`selection.ic_grid`, a heatmap per leg)
    """

    SINGLE = "single"
    FAST_SLOW = "fast_slow"


# --------------------------------------------------------------------------------------------------
# Raw events fed to a streaming feature. ONE record per event carries EVERY property the raw stream
# holds, so `on_book` / `on_trade` take a single `ev` and their signature never changes when a new
# feature needs another field — add the field here (and to the raw stream) instead. A feature reads
# only the props it uses (`ev.bid`, `ev.ask_qty`, `ev.px`, …); the rest cost nothing.
# --------------------------------------------------------------------------------------------------
class BookEvent(NamedTuple):
    listing: str
    rx: int                 # receive-time ("now" when we got it); rx - exch_time = feed latency
    exch_time: int
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float


class TradeEvent(NamedTuple):
    listing: str
    rx: int                 # receive-time ("now"); the trade clock ticks on rx, so a rate yardstick reads it
    exch_time: int
    px: float
    lifts_ask: float
    qty: float


class StreamingFeature(Protocol):
    """O(1)-per-event *production* build of one feature-family member.

    The generic parity driver feeds raw events in receive-time order, calls `refresh()` once per
    receive-timestamp, and reads `value()` at each grid anchor. State must be O(1) (scalar EMAs
    only) and built ONLY from `boba.ema` classes (`KernelMeanEMA` / `LiveFrontEMA`) — never a
    hand-rolled EMA, never `EventEMA` directly, never `lfilter`. Compose `boba.features.streaming.
    VolYardstick` / `RateYardstick` rather than recomputing a yardstick, so yardstick parity is established once.

    Contract the driver relies on:
      keys          -- the leg keys; MUST equal the vectorized builder's dict keys for the same
                       `params` (the driver asserts this before driving).
      on_book/on_trade -- mutate internal state for ONE raw event, given the full event record
                       (`BookEvent` / `TradeEvent`). Read only the props you need. Do NOT advance the
                       clock or read. (A feature whose mid fuses trades composes `LiveMergedBook`, which
                       holds the per-listing fuse policy internally — there is no driver-level fuse flag.)
      refresh()     -- called ONCE per receive-timestamp, after all that timestamp's events are
                       applied: update live fronts / inject flows, then advance the trade clock AT
                       MOST ONCE (iff a trade landed this timestamp). Same-timestamp events are one
                       update / one decay.
      value()       -- {leg_key -> scalar}: the live-front feature at the instant it is read
                       (after every book update up to the anchor, never frozen at the last trade).
    """

    keys: tuple[str, ...]

    def on_book(self, ev: BookEvent) -> None: ...
    def on_trade(self, ev: TradeEvent) -> None: ...
    def refresh(self) -> None: ...
    def value(self) -> dict[str, float]: ...


class VectorizedBuilder(Protocol):
    """Offline (array) build of one feature-family member — the analysis path. A PURE time-series
    transform of the standalone feature inputs:

    `(raw_data, shared_data, config, params) -> {leg_key -> feature vector}`, each vector length
    `len(shared_data.event_ts)` — ONE value per event timestamp (NOT a research eval grid; sampling
    onto an eval grid is the caller's job, downstream). Causal: each row uses only data at-or-before
    its event timestamp (NaN where the feature is undefined, e.g. before a source has quoted — masked
    downstream). The yardstick vectors the feature divides by are on `shared_data`
    (`vol_yardstick` / `rate_yardstick`, already on `event_ts`). Keys MUST match the streaming
    feature's `keys` for the same `params`. May use `lfilter` / any offline trick — this is NOT the
    production path (the streaming class is), and the parity check ties them.
    """

    def __call__(self, raw: "RawData", shared: "SharedData", config: "Config",
                 params: Params) -> dict[str, np.ndarray]: ...


# A factory returning a FRESH streaming instance for one (config, params). The parity driver builds
# one per params token and drives them all through the event stream in a single pass.
StreamingFactory = Callable[["Config", Params], StreamingFeature]


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
    param_kind     -- the shape of the `params` token (`ParamKind`): `SINGLE` (one span `N`) or `FAST_SLOW`
                      (a `(n_fast, n_slow)` pair). The screening/finalize notebooks read this to build the
                      right span sweep (1-D scan vs 2-D grid) from ONE shared notebook. Defaults to
                      `FAST_SLOW` (the common case); a single-span feature must set `SINGLE`.
    """

    name: str
    vectorized: VectorizedBuilder
    make_streaming: StreamingFactory
    keys_for: Callable[["Config", Params], tuple[str, ...]]
    mirror: Callable[[np.ndarray], np.ndarray] | None = None
    param_kind: ParamKind = ParamKind.FAST_SLOW


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


# ==================================================================================================
# Standalone feature inputs — the data CONTRACTS a vectorized feature runs on, defined HERE so a feature
# is independent of any research / notebook logic. The eval grid, the prediction target, and the
# sampling onto that grid are the caller's job, applied AFTER the feature is built. Three values:
#
#   raw_data    -- exactly what was loaded off disk, UNTRANSFORMED (front_levels / trade / merged_levels
#                  per listing, concatenated across blocks) + per-block extents (BlockMeta).
#   shared_data -- everything PRECOMPUTED ONCE from raw_data and shared by every vectorized feature
#                  (the σ_ev / λ_ev yardsticks, the decay clock, per-listing mids). Its vectors live on
#                  `event_ts` -- the feature OUTPUT grid: one slot per timestamp that carried an event.
#   config      -- what to build over (target + other listings + the per-listing mid policy + span).
#
# A vectorized builder is a PURE time-series transform `(raw_data, shared_data, config, params) ->
# {listing -> value per event_ts}`; choosing where to READ it (an eval grid) and what to predict (a
# target) happen downstream in research. These are the TYPES only; the precompute that fills `shared_data`
# (`build_shared_data`) and the shared `flow_at` primitive live in `boba.features.shared` -- the
# IMPLEMENTATION, separated from these contracts, importing nothing from `boba.research`. (The legacy
# `ScreeningContext`-based builder signature above is migrated onto this triple feature-by-feature.)
# ==================================================================================================


class Series(NamedTuple):
    """A timestamped scalar series: `value[i]` is the quantity at receive-time `rx[i]`. `rx` is int64
    and ascending. A `NamedTuple`, so it still unpacks as `rx, value = series` and indexes like a tuple."""

    rx: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class BlockMeta:
    """Identity + extent of one loaded ~24h block. INTENTIONAL scaffolding (not yet constructed): the
    per-block extents are how a consumer will TRIM warm-up rows at a block boundary — when the stream
    transitions from one block to the next, EMA/yardstick state carried across the seam is stale, so the
    rows just after each `start_ns` must be dropped. `start_ns` / `end_ns` are the first / last event
    receive-times in the block. Do not remove as 'dead' — it is the hook for block-boundary trimming."""

    name: str
    start_ns: int
    end_ns: int


class FrontLevels(NamedTuple):
    """Raw top-of-book snapshot stream (the `front_levels` parquet): best bid/ask + their sizes at each
    receive-time `rx`, plus the venue's `exchange_time` stamp -- so `rx - exchange_time` is the feed
    LATENCY (a feature in its own right; a latency blow-out is a signal). Sizes are the venue's last
    snapshot qty (can be stale between snapshots). A `NamedTuple` -- unpacks / indexes like a tuple while
    exposing named columns."""

    rx: np.ndarray
    exchange_time: np.ndarray
    bid: np.ndarray
    bid_qty: np.ndarray
    ask: np.ndarray
    ask_qty: np.ndarray


class Trade(NamedTuple):
    """Raw trade stream (the `trade` parquet): one executed print per row -- `price` and `qty` (size) of
    the fill at receive-time `rx`, with the venue's `exchange_time` stamp (`rx - exchange_time` = feed
    latency). `lifts_ask` is 1.0 for a buy (the aggressor lifts the ask) and 0.0 for a sell (hits the bid)."""

    rx: np.ndarray
    exchange_time: np.ndarray
    price: np.ndarray
    lifts_ask: np.ndarray
    qty: np.ndarray


class MergedLevels(NamedTuple):
    """Raw trade-fused top-of-book stream (the `merged_levels` parquet, present for some listings only):
    best bid/ask after folding trades into the book -- the freshest mid source. Each side carries its OWN
    exchange stamp (`bid_exchange_time` / `ask_exchange_time`), since the bid and ask can come from
    different source updates (so per-side latency is available too)."""

    rx: np.ndarray
    bid_exchange_time: np.ndarray
    ask_exchange_time: np.ndarray
    bid: np.ndarray
    ask: np.ndarray


@dataclass(frozen=True)
class ListingRaw:
    """One listing's RAW, untransformed streams -- exactly the parquet columns (concatenated across
    blocks), nothing derived. Every listing has `front_levels` + `trade`; only venues whose mid fuses
    trades carry `merged_levels`. Arrays are receive-time (`rx`) ordered."""

    front_levels: FrontLevels
    trade: Trade
    merged_levels: Optional[MergedLevels] = None


@dataclass(frozen=True)
class RawData:
    """All raw data for a run: per-listing raw streams (keyed by full listing id, e.g.
    `"byb_eth_usdt_p"`) + the per-block extents (`blocks`) used to trim warm-up at block boundaries.
    UNTRANSFORMED -- a mid / clock / yardstick are all derived and live in `SharedData`, never here."""

    listings: dict[str, ListingRaw]
    blocks: tuple[BlockMeta, ...] = ()      # per-block extents for boundary warm-up trimming (see BlockMeta)


@dataclass(frozen=True)
class ListingShared:
    """Per-listing quantities PRECOMPUTED from raw_data (derived -> here, not in `RawData`).

        mid : event-time mid `Series` = (bid+ask)/2 from the listing's mid-policy stream
    """

    mid: Series


@dataclass(frozen=True)
class SharedData:
    """Everything precomputed once from `raw_data` and shared by every vectorized feature. The vectors
    are indexed by `event_ts` -- the feature OUTPUT grid (one slot per timestamp that carried an event
    on any listing); a feature emits a value for each. The DECAY clock is separate (a subset of it).

        event_ts       : every timestamp with an event -- the feature output index
        clock          : the DECAY clock -- trade-tick timestamps where EMAs decay
        vol_yardstick  : σ_ev at each event_ts  (RMS target mid-move per move)
        rate_yardstick : λ_ev at each event_ts  (target moves per second)
        listings       : {listing -> ListingShared}
    """

    event_ts: np.ndarray
    clock: np.ndarray
    vol_yardstick: np.ndarray
    rate_yardstick: np.ndarray
    listings: dict[str, ListingShared]


@dataclass(frozen=True)
class Config:
    """What to build a feature over -- independent of any research / eval choice.

        target_listing : the prediction target whose mid-moves define the yardsticks (e.g. byb)
        other_listings : the foreign sources a feature may fan out over
        coin           : the instrument token (e.g. "eth_usdt_p")
        mid_stream     : per-listing mid policy -- "merged_levels" (fuse trades) or "front_levels"
        yardstick_span : the EMA span of σ_ev / λ_ev
        tick_size      : per-listing price tick (from `io.tick_sizes`) -- the streaming merged-book
                         reconstruction needs it to un-cross a fused book; `{}` when no fused listing
                         needs it (e.g. synthetic tests with book-only mids).
    """

    target_listing: str
    other_listings: tuple[str, ...]
    coin: str
    mid_stream: dict[str, str]
    yardstick_span: int = 10_000
    tick_size: dict[str, float] = field(default_factory=dict)

    @property
    def all_listings(self) -> tuple[str, ...]:
        return (self.target_listing,) + tuple(self.other_listings)
