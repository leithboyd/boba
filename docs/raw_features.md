# Raw features spec (`boba/dataset/raw.py`)

Features are computed per **listing** — `{exchange}_{pair}[_p]` (e.g. `bin_doge_usdt_p`,
`bin_doge_usdt`) across bin/byb/okx — and built per ~24h block with caching.

## Column selection

The `{LISTING}…` lines below are the **template registry** (`boba.dataset.columns.TEMPLATES`),
and a dataset is defined by explicitly instantiating templates — there is no default
catalogue. `DatasetRawConfig.columns` takes an ordered tuple of `ColumnSpec(template, args)`
where `args` is a sequence of mappings, each binding exactly the template's placeholders:

    ColumnSpec("{LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms",
               args=({"LISTING1": "bin_eth_usdt", "LISTING2": "byb_eth_usdt", "N": 10},
                     {"LISTING1": "bin_eth_usdt", "LISTING2": "byb_eth_usdt", "N": 20}))

A scalar value binds directly; a list/tuple value fans out. As soon as any value in a
mapping is a list, every value is treated as a (possibly length-1) list and the mapping
expands to the **cross product**, staged in template placeholder order:

    ColumnSpec("{LISTING}_ema_ofi_{N}b",
               args=({"LISTING": ["bin_eth_usdt", "byb_eth_usdt"], "N": [3, 10, 50]},))
    # → 6 columns; any positive integer N is valid, not just the spans listed below

`col(template, **kwargs)` is sugar for a single-mapping spec. Output column order is
exactly the expansion order (spec → mapping → cross product), and the ORDERED name list
feeds the cache key, so permuted selections are distinct datasets. Cross-listing pairs
may be requested in either direction (the reverse is the negation — see below) but
LISTING1 == LISTING2 is rejected, so array×array cross-listing bindings must use
disjoint listing sets (or enumerate pairs as separate mappings).

The `N ∈ {…}` sets below are the spans the feature design was tuned around —
recommended defaults, not constraints. Caveat for large N: EMAs converge from a
y[−1]=0 initial condition, so columns with span ≫ warmup_ms are still in their
transient at the start of each block.

## Raw 1MS grid features

Use the latest available value for a 1MS window if none exists

  {LISTING}_microprice
  {LISTING}_spread_width                     (ask - bid) / microprice
  {LISTING}_book_depth                       bid_qty + ask_qty
  {LISTING}_book_imbalance                   (bid_qty − ask_qty) / (bid_qty + ask_qty)
  {LISTING}_spread_wide_flag                 (spread_width > {LISTING_wide_threshold}).astype(np.float32)

  {LISTING}_dt_{N}b                          wallclock time elapsed over the last N book ticks       N ∈ {   10, 100, 1000, 10000}
  {LISTING}_dt_{N}t                          wallclock time elapsed over the last N trades           N ∈ {3, 10, 100, 1000}
  {LISTING}_dt_{N}m                          wallclock time elapsed over the last N mid moves        N ∈ {1, 10, 100, 1000}
  {LISTING}_trade_count_{N}ms                count of trades in the trailing window (t − N ms, t]    N ∈ {1, 5, 25, 100}
  {LISTING}_time_since_last_trade_ms         wallclock ms since the last trade event
  {LISTING}_time_since_spread_wide_ms        wallclock ms since spread_wide_flag last = 1

  dt_{N}m is the real-move event clock / subordination ladder (rate = N / dt): a "mid move" is any
  book event that changes book_mid = (bid + ask) / 2, so — unlike dt_{N}b — size-only updates that
  leave the mid unchanged do not advance it. Forward-filled to the grid like the other dt_* columns.

  trade_count_{N}ms is a trailing wall-clock count over the left-open/right-closed window (t − N ms, t]
  (a trade exactly at t − N ms is excluded, one exactly at t is included). Causal — only trades at or
  before t. Per listing; requesting it for every listing gives the cross-venue lead-lag picture (a fine
  N spikes on the venue that "just fired").

  {LISTING}_feed_latency_excess_ms           excess feed latency above 30k-row rolling min

  Per-ms trade flow — sum within the 1ms window, NOT forward-filled (0 when no trade lands in
  the window; these are flow, not state, so the "latest available value" rule above does not
  apply). Buy/sell split by aggressor (Bid = BUY convention). Value sums are invariant to the
  same-ns trade aggregation (VWAP preserves qty·prc = total notional, impl note 3).

  {LISTING}_buy_trade_value                  Σ qty·prc over buy-aggressor trades in this ms
  {LISTING}_sell_trade_value                 Σ qty·prc over sell-aggressor trades in this ms

## Temporal 1MS grid features

Calculated on top of 1ms grid. Computed in float64 internally, cast to float32 on output.

  {LISTING}_return_{N}ms                      log(microprice_{t} / microprice_{t − Nms})              N ∈ {1, 5, 20, 100, 1000, 5000}

  {LISTING}_ema_microprice_centered_{N}ms     EMA(microprice − {LISTING}_microprice_ref,  span=N ms)      N ∈ {100, 200, 500, 1000, 10000, 60000}
  {LISTING}_ema_microprice_centered_sq_{N}ms  EMA((microprice − {LISTING}_microprice_ref)², span=N ms)    N ∈ {100, 200, 500, 1000, 10000, 60000}
  {LISTING}_ema_microprice_return_sq_{N}ms    EMA((Δ log microprice)², span=N ms)  [1ms log-return RV]    N ∈ {100, 200, 500, 1000, 10000, 60000}

  Centering note: subtracting a per-exchange reference price (config:
  microprice_ref, e.g. 0.15 for DOGE-perp) before the EMA keeps stored values
  small so the float32 variance reconstruction `ema_sq − ema²` stays precise
  even for tiny variances at short spans.

  Return realized-variance note: ema_microprice_return_sq is the RiskMetrics-style RV rate — an EWMA
  of squared 1ms log-microprice returns (Δ log microprice = log mp_t − log mp_{t−1}). Differencing
  before squaring makes it drift-immune and dimensionless; the forward-move vol normalizer is
  r = sqrt(ema_microprice_return_sq)·1e4. This is the variance of the RETURN, distinct from
  ema_microprice_centered_sq, which is the variance of the price LEVEL (and so absorbs trend). No
  microprice_ref centering applies here — the differencing already removes the level.

  Cross-listing price ratios (pairs ordered alphabetically, LISTING1 < LISTING2 → one per pair;
  the reverse is just the negation, since ema(log(a/b)) = −ema(log(b/a))). 15 pairs for the 6
  spot+perp listings:

  {LISTING1}_{LISTING2}_ema_log_microprice_ratio_{N}ms      EMA(log(microprice_{LISTING1} / microprice_{LISTING2}),  span=N ms)   N ∈ {100, 200, 500, 1000, 10000, 60000}
  {LISTING1}_{LISTING2}_ema_log_microprice_ratio_sq_{N}ms   EMA(log(microprice_{LISTING1} / microprice_{LISTING2})², span=N ms)   N ∈ {100, 200, 500, 1000, 10000, 60000}

  Variance of the cross-venue log-ratio at span N = ema_sq − ema² (how volatile the spread is at
  that timescale). The log-ratio is already ~0-centered (similar prices → log-ratio ≈ 0), so no
  ref-subtraction is needed for the float32 reconstruction. Note the _sq term is order-INVARIANT
  (squaring kills the sign: log(a/b)² = log(b/a)²), so it — and hence the variance — is identical
  for either pair ordering, even though the ratio EMA itself negates.

  Identity (why one direction suffices):
      ema(log(a/b)) = −ema(log(b/a))
  EMA is a linear operator and log(a/b) = −log(b/a), so the reverse-ordered ratio is exactly the
  negation — store the alphabetical direction, negate to recover the other. (Note this holds only
  in LOG space: for raw ratios ema(a/b) ≠ 1/ema(b/a), since averaging and reciprocal don't commute.)

  Deviation feature = instantaneous log(mp_{l1}/mp_{l2}) − this EMA — "how far the cross-venue
  spread sits from its recent norm" (mean-reversion / lead-lag). The instantaneous log-ratio is
  derived at normalisation from the two listings' microprices (already stored), so only the EMA
  is a new raw column here.

  Trade-flow rate EMAs — over the per-ms `buy/sell_trade_value` grid columns (zeros between
  trades):

  {LISTING}_ema_buy_trade_value_{N}ms         EMA(buy_trade_value,  span=N ms)                    N ∈ {10, 25, 50, 100}
  {LISTING}_ema_sell_trade_value_{N}ms        EMA(sell_trade_value, span=N ms)                    N ∈ {10, 25, 50, 100}

  Wallclock counterpart of the event-clocked `ema_{buy,sell}_trade_value_{N}t` (Microstructure
  EMAs below): a time-decayed $-flow rate. Because the EMA absorbs a 0 every quiet ms, it decays
  in real time during silence — unlike the `_t` versions, which update only on trade events and
  hold their last value on a dead tape. Same wallclock window on every venue (no event-clock
  timescale mixing), so cross-venue flow comparisons are aligned. A single trade of value V in an
  otherwise-quiet stretch reads V·α at the ms it lands (α = 2/(N+1)), then decays by (1−α) per ms.

## Microstructure EMAs 

These features must be calculated at the tick level before aggregating to the 1ms grid.
When multiple ticks have the same exchange timestamp then aggregate events before updating the EMA (assuming the exchange gives us a timestamp with nano/micro second time resolution)
For OFI and trade based features the aggregation is sum. For microprice we use the last microprice in each time window to calculate microprice_{i} / microprice_{i−1}.


  Trades: EMA updated on every trade event. Forall N ∈ {3, 10, 50, 100, 1000}


  {LISTING}_ema_buy_trade_qty_{N}t            EMA(qty   if buy  else 0, span=N)
  {LISTING}_ema_sell_trade_qty_{N}t           EMA(qty   if sell else 0, span=N)
  {LISTING}_ema_buy_trade_value_{N}t          EMA(qty·p if buy  else 0, span=N)
  {LISTING}_ema_sell_trade_value_{N}t         EMA(qty·p if sell else 0, span=N)

  {LISTING}_ema_trade_serial_cov_{N}t         EMA(Δp_t · Δp_{t−1}, span=N)                             N ∈ {100, 1000}
                                          Δp_t = trade_prc_t − trade_prc_{t−1} (after same-ns aggregation)
                                          Roll's effective spread = 2·sqrt(max(−ema_trade_serial_cov, 0))


  BBO: EMA updated on every BBO update event

  {LISTING}_ema_ofi_{N}b                      EMA(signed OFI event, span=N)                            N ∈ {3, 10, 100, 1000}
  {LISTING}_ema_ofi_sq_{N}b                   EMA(ofi_event², span=N)                                  N ∈ {            1000, 10000}
  {LISTING}_ema_abs_log_ret_{N}b              EMA(|log(microprice_{i} / microprice_{i−1})|, span=N)    N ∈ {       100, 1000, 10000}

  {LISTING}_ema_book_imbalance_{N}b           EMA(book_imbalance,  span=N)                             N ∈ {   10, 100, 1000, 10000}
  {LISTING}_ema_book_imbalance_sq_{N}b        EMA(book_imbalance², span=N)                             N ∈ {            1000, 10000}
  {LISTING}_ema_book_depth_{N}b               EMA(book_depth,      span=N)                             N ∈ {   10, 100, 1000, 10000}
  {LISTING}_ema_book_depth_sq_{N}b            EMA(book_depth²,     span=N)                             N ∈ {            1000, 10000}

  {LISTING}_ema_spread_wide_flag_{N}b         EMA(spread_wide_flag, span=N)                            N ∈ {       100, 1000}


## Shared trade-clock event features

Event-clocked features whose clock can be **decoupled from the data** and pointed at another
listing's trade stream, via an optional `@{CLOCK}` qualifier. The clock token names its event
type and listing: `trades_<listing>`. The bare form clocks off the column's OWN trades
(`{LISTING}_vol_{N}t` ≡ `{LISTING}_vol_{N}t@trades_{LISTING}`). The clock only sets the event
windows / decay steps; the measured quantity stays full-resolution. The point is cross-exchange
alignment: on one shared clock a busy and a quiet venue's event-EMAs advance in lockstep —
span N = the same wall-clock window everywhere — instead of at each venue's own event rate. So
two venues tracing the same line give (near-)identical columns; a down-sampling that drops only
no-move book updates changes nothing. `@trades_{CLOCK}` requires {CLOCK} to be one of the loaded
listings, and the clock is part of the column name (and the cache identity). Everything is
tick-EMA'd over the clock's trade events (span N = N clock trades), forward-filled to the grid,
0 before the first clock trade.

  Realized vol of the MID = (bid + ask)/2. tick-EMA of the per-interval realized variance (the
  sum of squared 1ms mid log-returns in each clock interval) — clock = window ruler, mid path
  full-resolution (so it counts every move, not just those at trade instants):

  {LISTING}_vol_{N}t[@trades_{CLOCK}]            sqrt( EMA( Σ_interval (Δlog mid)², span=N trades ) )

  Book-event quantities on the trade clock — the `_b` families above, re-clocked onto trades
  (own or @trades_{CLOCK}). Two semantics:
    LEVEL — sample the state at each clock trade (forward-fill), then tick-EMA.
    FLOW  — sum the quantity over each clock interval, then tick-EMA.
  `_sq` squares the sampled level / per-event flow before the EMA.

  LEVEL families:
  {LISTING}_ema_book_imbalance_{N}t[@…]          EMA( book_imbalance sampled at each clock trade, span=N )
  {LISTING}_ema_book_imbalance_sq_{N}t[@…]       EMA( book_imbalance² sampled at each clock trade, span=N )
  {LISTING}_ema_book_depth_{N}t[@…]              EMA( book_depth sampled …, span=N )
  {LISTING}_ema_book_depth_sq_{N}t[@…]           EMA( book_depth² sampled …, span=N )
  {LISTING}_ema_spread_wide_flag_{N}t[@…]        EMA( spread_wide_flag sampled …, span=N )
  {LISTING}_ema_microprice_centered_{N}t[@…]     EMA( (microprice − {LISTING}_microprice_ref) sampled …, span=N )
  {LISTING}_ema_microprice_centered_sq_{N}t[@…]  EMA( (microprice − microprice_ref)² sampled …, span=N )

  FLOW families:
  {LISTING}_ema_ofi_{N}t[@…]                     EMA( Σ_interval signed OFI event, span=N )
  {LISTING}_ema_ofi_sq_{N}t[@…]                  EMA( Σ_interval ofi_event², span=N )
  {LISTING}_ema_abs_log_ret_{N}t[@…]             EMA( Σ_interval |log(microprice_i / microprice_{i−1})|, span=N )

  Trade-flow families on the trade clock — the existing `_t` trade EMAs, now with an optional
  @{CLOCK}. FLOW = sum the listing's OWN per-trade quantity over each clock interval; the bare
  own-clock form has one trade per interval, so it is identical to the existing per-trade EMA.
  {LISTING}_ema_buy_trade_qty_{N}t[@…]           EMA( Σ_interval (qty   if buy  else 0), span=N )
  {LISTING}_ema_sell_trade_qty_{N}t[@…]          EMA( Σ_interval (qty   if sell else 0), span=N )
  {LISTING}_ema_buy_trade_value_{N}t[@…]         EMA( Σ_interval (qty·p if buy  else 0), span=N )
  {LISTING}_ema_sell_trade_value_{N}t[@…]        EMA( Σ_interval (qty·p if sell else 0), span=N )
  {LISTING}_ema_trade_serial_cov_{N}t[@…]        EMA( Σ_interval (Δp_t · Δp_{t−1}), span=N )


## Implementation notes — deviations from spec

The current implementation in `boba/dataset/raw.py` differs from the spec above in a
few places. None are correctness bugs for exploration; flagged here so future
work can revisit if needed.

  1. Aggregation timestamp uses `rx_time`, not `exchange_time`.
       The spec says aggregate same-ns events by exchange timestamp. The
       implementation uses rx_time because BBO exchange timestamps are not
       currently exposed in SessionData (only consumed when computing
       `feed_latency_excess_ns`). For trades, the same choice was made for
       consistency even though `trade_exchange_ts` is available.
       Practical impact: rx-time aggregation is at the mercy of feed-handler
       arrival jitter. Distinct ME events arriving in the same rx-ns would
       be incorrectly collapsed; one ME event split across rx-ns would not
       be collapsed. On DOGE-perp at ns rx timestamping this is rare.

  2. OFI aggregation: last-state-then-compute, not sum of intermediate OFIs.
       The spec says "For OFI and trade based features the aggregation is sum".
       The implementation instead collapses BBO state to last-value per ns and
       computes OFI between consecutive aggregated states. The two differ when
       same-ns paths are non-monotonic (e.g. bid 100 → 101 → 100 within one ns).
       The implementation choice treats the transient at 101 as a feed-ordering
       artifact; the spec literal would credit it as real intermediate flow.

  3. Trade aggregation: split by side, VWAP price.
       The spec says "aggregation is sum" for trades but does not specify the
       price aggregation or whether opposite-side same-ns trades should be
       collapsed. The implementation:
         - sums qty within each (ts, side) group,
         - uses VWAP for the aggregated price (so qty × price = total notional),
         - keeps opposite-side same-ns trades as separate events (different
           aggressor direction → different economic event).

  4. Float32 precision floor for `ema_microprice_centered_sq` reconstruction.
       At microprice ≈ 0.15 with `microprice_ref` matched, variance recovery
       (`ema_sq − ema²`) is reliable down to ~1e-10 (verified by test). Below
       that scale (vanishingly quiet markets at short spans), the recovered
       variance is dominated by float32 rounding noise.

  5. EMA convergence transient.
       For the first ~3× span samples of each session, the EMAs have not
       converged from their y[−1] = 0 initial condition, so the recovered
       "variance" includes transient bias rather than just the input
       variance. Downstream code should either skip the transient or expect
       biased values during it.
