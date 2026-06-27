"""Streaming-side primitives the production (O(1)-per-event) feature builds compose.

Kept in the feature layer (not research) so a streaming feature is standalone -- it imports nothing
from `boba.research`:
  * `VolYardstick` / `RateYardstick` -- the shared σ_ev / λ_ev (vectorized twin: `boba.features.shared._yardsticks`).
  * `LiveMergedBook` -- the shared merged-BOOK reconstruction (vectorized twin: `io._build_merged_levels`):
                        exposes the un-crossed `(bid, ask)` per listing so every book-based feature reads
                        ONE fuse+un-cross implementation, not its own (the caller derives mid/spread/…).
"""
from __future__ import annotations

from typing import Optional

from boba.ema import KernelMeanEMA


def uncross_quote(bid: float, ask: float, bid_t: int, ask_t: int, tick: Optional[float]) -> tuple[float, float]:
    """The un-crossed `(bid, ask)` of a possibly-crossed fused book -- the scalar online twin of
    `io._uncross_book`. If `ask < bid` and a `tick` is given, trust the side with the newer exchange_time
    and push the STALE side one tick past it (ties -> ask fresher). `tick=None` leaves a book-only quote
    as-is. Returns the raw quote; the caller derives whatever it needs (mid, spread, microprice, …)."""
    if tick is not None and ask < bid:
        if ask_t >= bid_t:
            bid = ask - tick          # ask fresher -> clamp bid down one tick
        else:
            ask = bid + tick          # bid fresher -> clamp ask up one tick
    return bid, ask


class LiveMergedBook:
    """Streaming reconstruction of `io.merged_levels` -- the ONLINE twin of `io._build_merged_levels`,
    composed by any streaming feature that reads a fused mid so the fuse+un-cross logic lives in ONE
    place (not copied into each feature). For a FUSED listing (one in `fuse_tick`) it holds, per side,
    the price of the newest-by-exchange_time event among {BBO snapshots, qualifying trades} -- ties go to
    the LATEST in sequence (`>=`, matching the vectorized merge's sweep rule) -- and un-crosses with the
    listing's tick. A non-fused listing just holds its raw BBO. Feed every `on_book`/`on_trade`; read
    `quote(listing)` for the un-crossed `(bid, ask)` -- the caller derives the mid / spread / … it wants.
    """

    __slots__ = ("fuse_tick", "bid", "ask", "bid_t", "ask_t")

    def __init__(self, fuse_tick: dict[str, float]):
        self.fuse_tick = fuse_tick            # {fused listing -> tick}; listings not here are book-only
        self.bid: dict[str, float] = {}
        self.ask: dict[str, float] = {}
        self.bid_t: dict[str, int] = {}
        self.ask_t: dict[str, int] = {}

    def _side(self, listing: str, is_ask: bool, px: float, t: int) -> None:
        held_t = self.ask_t if is_ask else self.bid_t
        if t >= held_t.get(listing, -1):      # >= : a same-exchange_time sweep holds its LATEST/deepest print
            (self.ask if is_ask else self.bid)[listing] = px
            held_t[listing] = t

    def on_book(self, ev) -> None:
        if ev.listing in self.fuse_tick:      # fused: hold newest-by-exch per side
            self._side(ev.listing, False, ev.bid, ev.exch_time)
            self._side(ev.listing, True, ev.ask, ev.exch_time)
        else:                                 # book-only: take the raw snapshot
            self.bid[ev.listing] = ev.bid
            self.ask[ev.listing] = ev.ask

    def on_trade(self, ev) -> None:
        if ev.listing in self.fuse_tick:      # a trade updates the side it lifts (already venue-corrected)
            self._side(ev.listing, ev.lifts_ask, ev.px, ev.exch_time)

    def quote(self, listing: str) -> Optional[tuple[float, float]]:
        """The UN-CROSSED `(bid, ask)` for `listing`, or `None` if either side hasn't quoted. The caller
        derives whatever it needs -- mid `(bid+ask)/2`, spread, microprice -- this exposes the primitive."""
        b, a = self.bid.get(listing), self.ask.get(listing)
        if b is None or a is None:
            return None
        return uncross_quote(b, a, self.bid_t.get(listing, -1), self.ask_t.get(listing, -1),
                             self.fuse_tick.get(listing))


class VolYardstick:
    """Streaming σ_ev (composes `boba.ema.KernelMeanEMA`): the online twin of
    `boba.features.shared._yardsticks`' σ_ev — RMS target mid-move per move. It decays on the shared trade
    clock and injects a squared return on each *real* target mid-move. A feature COMPOSES this rather than
    recomputing σ_ev, so its parity validates only its own maths. (`RateYardstick` is the λ_ev twin.)
    """

    __slots__ = ("vol", "_prev")

    def __init__(self, span: int):
        self.vol = KernelMeanEMA(span)      # E/W mean of squared target moves -> sqrt(E/W) = σ_ev
        self._prev: Optional[float] = None  # last target log-mid, to detect a real move

    def on_target_logmid(self, log_mid: Optional[float]) -> None:
        """Feed the current target log-mid (or None before it has quoted). Injects `(Δlog)**2`
        iff the mid actually moved — a flow on the target-move stream. Does NOT decay."""
        if log_mid is None:
            return
        if self._prev is not None and log_mid != self._prev:
            self.vol.add((log_mid - self._prev) ** 2)
        self._prev = log_mid

    def tick(self) -> None:
        """Advance the shared trade clock one step (decay E and W)."""
        self.vol.tick()

    def sigma(self) -> float:
        """σ_ev = sqrt(E/W) — RMS mid-move per move, live (nan during warm-up)."""
        return self.vol.value() ** 0.5


class RateYardstick:
    """Streaming λ_ev (composes `boba.ema.KernelMeanEMA`): the online twin of
    `boba.features.shared._yardsticks`' λ_ev — target moves per second. A perfectly valid normaliser for a
    streaming feature (anything measured against the target's move RATE), just not used by one yet.

    It holds two flows on the shared trade clock and reads `λ_ev = (move-count flow) / (inter-tick-time
    flow)`. The move-count flow is the `W` of a `KernelMeanEMA` that injects one unit per *real* target
    mid-move (its `value()` is uninteresting — we read `W` = `e_mv`). The time flow is the EMA of the
    seconds between trade ticks, so `tick(now_ns)` needs the tick's timestamp (unlike `VolYardstick.tick`).
    """

    __slots__ = ("count", "alpha", "dt", "_prev", "_last_tick_ns")

    def __init__(self, span: int):
        self.count = KernelMeanEMA(span)    # injects 1 per move; its W = e_mv (the move-count flow)
        self.alpha = 2.0 / (span + 1.0)
        self.dt = 0.0                       # e_dt: EMA of inter-tick seconds (lfilter twin)
        self._prev: Optional[float] = None  # last target log-mid, to detect a real move
        self._last_tick_ns: Optional[int] = None

    def on_target_logmid(self, log_mid: Optional[float]) -> None:
        """Feed the current target log-mid; inject one unit into the move-count flow iff the mid moved."""
        if log_mid is None:
            return
        if self._prev is not None and log_mid != self._prev:
            self.count.add(1.0)
        self._prev = log_mid

    def tick(self, now_ns: int) -> None:
        """Advance the shared trade clock one step at receive-time `now_ns`: fold the gap since the last
        tick into the time flow `e_dt = (1-α)·e_dt + α·Δt`, then decay the move-count flow."""
        dt = 0.0 if self._last_tick_ns is None else (now_ns - self._last_tick_ns) / 1e9
        self.dt = (1.0 - self.alpha) * self.dt + self.alpha * dt
        self._last_tick_ns = now_ns
        self.count.tick()

    def lam(self) -> float:
        """λ_ev = e_mv / e_dt — target moves per second, live (nan before the first inter-tick gap)."""
        return self.count.W / self.dt if self.dt > 0.0 else float("nan")
