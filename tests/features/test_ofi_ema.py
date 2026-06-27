"""Self-contained test suite for `boba.features.ofi_ema` — a single EMA of byb's path-sum L1 OFI.

The AUTHORING.md validation trio plus the streaming coverage, all on synthetic data with NO shared
code with the production build:

  - `test_vectorized_matches_oracle`     vectorized vs an INDEPENDENT dead-simple event-loop oracle.
  - `test_fans_out_over_all_exchanges`   one independent OFI leg per venue (target + each source).
  - `test_mirror_commutes_with_reflection`  the COMMUTATION invariant against the FULL book reflection.
  - `test_span1_finite_where_inputs_exist`  span=1 (α=1) is finite wherever the flow exists, NaN only
                                            during warm-up — the AUTHORING span=1 Do-rule, both builds.
  - `test_streaming_matches_vectorized`  drive `LiveOFIEma` over a synthetic raw-event stream and
                                         assert it matches `spec.vectorized` (incl. a span=1 leg).
  - `test_real_block_parity`             streaming-vs-vectorized parity on a real block (DATA_DIR-gated).

The oracle (`_ofi_leg_oracle`) is written from the feature's definition alone — per consecutive raw
book row form the Cont-Kukanov-Stoikov L1 increment, SUM increments sharing a receive-timestamp into one
flow sample, then walk every event timestamp injecting (book change) / decaying (trade), reading the
self-normalising E/W after the last event at-or-before each anchor.
"""
import numpy as np
import pytest

import boba.io as io
from boba.features import base
import boba.features.ofi_ema  # noqa: F401  (registers the spec)
from boba.research.screening import RawEventStream, ScreeningContext, parity_check


# --------------------------------------------------------------------------------------------------
# synthetic raw-event stream + context (own builders — no shared code with production)
# --------------------------------------------------------------------------------------------------
def _raw_events(books, merged_ts, coin="x"):
    """Pack per-venue raw book rows + one trade per shared trade-timestamp into a `RawEventStream`.
    OFI is book-only, so the trades exist purely to drive the shared decay clock (their payload is dummy).
    A book event is packed `(rx, kind=0, lid, t=rx, bid, ask, bid_qty, ask_qty)` — the layout the parity
    driver unpacks into `BookEvent(listing, t, bid, ask, bid_qty, ask_qty)`."""
    listings = tuple(f"{ex}_{coin}" for ex in books)
    cols: dict[str, list] = {k: [] for k in "rx kind lid t a b c d".split()}

    def add(rx, kind, lid, t, a, b, c, d):
        n = len(rx)
        cols["rx"].append(rx.astype(np.int64))
        cols["kind"].append(np.full(n, kind, np.int8))
        cols["lid"].append(np.full(n, lid, np.int8))
        cols["t"].append(t.astype(np.int64))
        for k, v in (("a", a), ("b", b), ("c", c), ("d", d)):
            cols[k].append(v.astype(float))

    for lid, (ex, book) in enumerate(books.items()):
        rx, bid, bq, ask, aq = book
        add(rx, 0, lid, rx, bid, ask, bq, aq)
    # one trade per shared trade-timestamp (listing 0) — drives the decay clock only
    nt = len(merged_ts)
    add(merged_ts, 1, 0, merged_ts, np.ones(nt), np.ones(nt), np.ones(nt), np.full(nt, np.nan))

    C = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((C["kind"], C["rx"]))      # books (kind 0) before trades (kind 1) within a ts
    return RawEventStream(*(C[k][order] for k in "rx kind lid t a b c d".split()), listings)


def _ctx(books, merged_ts, anchor_ts, target="byb_x", sources=(), raw=False):
    """Synthetic `ScreeningContext`. `books` is {short_ex -> (rx, bid, bid_qty, ask, ask_qty)}; the feature
    fans out over target+sources. `raw=True` also builds the raw-event stream (needed for streaming parity)."""
    return ScreeningContext(
        block="syn", coin="x", target=target, sources=tuple(sources), horizon_ns=0,
        yardstick_span=10, mid_stream={}, merged_ts=merged_ts, anchor_ts=anchor_ts,
        tick_at_anchor=np.searchsorted(merged_ts, anchor_ts, "right") - 1,
        sigma_at_anchor=np.empty(0), lam_at_anchor=np.empty(0), price_target=np.empty(0),
        rate_target=np.empty(0), base=[], vol_level=np.empty(0), rate_level=np.empty(0),
        vol_regime=np.empty(0),
        raw_events=_raw_events(books, merged_ts) if raw else RawEventStream(*([np.empty(0)] * 8), ()),
        _books=dict(books),
    )


def _synthetic_book(seed=0, n=4000):
    """A book row every 10 ns; a few same-ts bursts so the path-sum (intra-ns sum) path is exercised."""
    rng = np.random.default_rng(seed)
    rx = (np.arange(1, n + 1) * 10).astype(np.int64)
    mid = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 1e-4))
    hs = 0.01 + 0.01 * rng.random(n)
    bid, ask = mid - hs, mid + hs
    bq, aq = rng.uniform(1.0, 100.0, n), rng.uniform(1.0, 100.0, n)
    rx[100:105] = rx[100]                                          # same-ts book burst -> summed increment
    rx[100:105] = np.sort(rx[100:105])
    # trade clock: coincident with every 3rd row plus some trade-only timestamps between rows
    merged_ts = np.unique(np.concatenate([rx[::3], rx[1::7] + 3]))
    anchor_ts = rx[200:-200:9]
    return (rx, bid, bq, ask, aq), merged_ts, anchor_ts


# --------------------------------------------------------------------------------------------------
# INDEPENDENT oracle — an explicit one-event-at-a-time loop, no shared code with the production build
# --------------------------------------------------------------------------------------------------
def _ofi_leg_oracle(book, merged_ts, anchor_ts, span):
    """E/W of the path-sum OFI flow at each anchor, dead-simple. Per consecutive raw row form the CKS L1
    increment `(Δbid depth on the bid side) - (Δask depth on the ask side)`, SUM increments sharing a
    receive-ts into one flow sample, then walk all event timestamps in order injecting (book change) /
    decaying (trade), reading E/W after the last event at-or-before each anchor. Inject-then-decay within
    a timestamp, matching `refresh()`. α = 2/(span+1); span=1 => α=1 (no memory)."""
    rx, bid, bq, ask, aq = book
    a = 2.0 / (span + 1.0)
    sums: dict[int, float] = {}
    for i in range(1, len(rx)):
        pbp, pbq, pap, paq = bid[i - 1], bq[i - 1], ask[i - 1], aq[i - 1]   # previous raw row
        cbp, cbq, cap, caq = bid[i], bq[i], ask[i], aq[i]                   # current raw row
        inc = ((cbq if cbp >= pbp else 0.0) - (pbq if cbp <= pbp else 0.0)
               - (caq if cap <= pap else 0.0) + (paq if cap >= pap else 0.0))
        sums[int(rx[i])] = sums.get(int(rx[i]), 0.0) + inc
    trades = set(int(t) for t in merged_ts)
    all_ts = sorted(set(sums) | trades)
    E = W = 0.0
    ts_arr, Es, Ws = [], [], []
    for ts in all_ts:
        if ts in sums:                       # inject the summed OFI as one weight-1 sample
            E += a * sums[ts]; W += a
        if ts in trades:                     # decay once on the trade clock (inject-then-decay)
            E *= (1.0 - a); W *= (1.0 - a)
        ts_arr.append(ts); Es.append(E); Ws.append(W)
    out = np.full(len(anchor_ts), np.nan)
    if not ts_arr:
        return out
    ts_arr, Es, Ws = np.array(ts_arr), np.array(Es), np.array(Ws)
    j = np.searchsorted(ts_arr, anchor_ts, "right") - 1          # last event at-or-before each anchor
    ok = j >= 0
    Ej, Wj = Es[j[ok]], Ws[j[ok]]
    out[ok] = np.where(Wj > 0.0, Ej / np.where(Wj == 0.0, np.nan, Wj), np.nan)
    return out


# --------------------------------------------------------------------------------------------------
# full book reflection — the data-level mirror the declared `mirror` must commute with
# --------------------------------------------------------------------------------------------------
def _mirror_book(book, c=100.0):
    """Reflect the L1 book through price level c: the two sides SWAP and reflect (`p -> c**2/p`), and each
    size follows its original price level (the new bid's size is the old ask's size). Sizes/clock unchanged
    per AUTHORING.md's reflection table — the operation a signed-flow feature's `mirror` must commute with."""
    rx, bid, bq, ask, aq = book
    return (rx, c * c / ask, aq, c * c / bid, bq)


# --------------------------------------------------------------------------------------------------
# (a) vectorized vs the independent oracle, across spans incl. span=1
# --------------------------------------------------------------------------------------------------
def test_vectorized_matches_oracle():
    book, merged_ts, anchor_ts = _synthetic_book()
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    spec = base.get("ofi_ema")
    for n in (1, 10, 100):                                        # includes span=1 (α=1, no smoothing)
        got = spec.vectorized(ctx, n)["byb"]
        ref = _ofi_leg_oracle(book, merged_ts, anchor_ts, n)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.sum() > 50
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9, atol=1e-9)
        # where one build is undefined (NaN) the other must agree it is undefined
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))


# --------------------------------------------------------------------------------------------------
# (b) per-venue fan-out: one independent leg per exchange, each the single-venue own-book build
# --------------------------------------------------------------------------------------------------
def test_fans_out_over_all_exchanges():
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=3)
    b_bin, _, _ = _synthetic_book(seed=4)
    b_okx, _, _ = _synthetic_book(seed=5)
    books = {"byb": b_byb, "bin": b_bin, "okx": b_okx}
    ctx = _ctx(books, merged_ts, anchor_ts, sources=("bin", "okx"))
    spec = base.get("ofi_ema")
    for params in (1, 50):
        out = spec.vectorized(ctx, params)
        assert set(out) == {"byb", "bin", "okx"}                 # one leg per exchange
        assert tuple(spec.keys_for(ctx, params)) == ("byb", "bin", "okx")
        for ex, book in books.items():                           # each leg == that venue's own-book OFI, independent
            solo = spec.vectorized(_ctx({ex: book}, merged_ts, anchor_ts, target=f"{ex}_x"), params)[ex]
            np.testing.assert_array_equal(out[ex], solo)


# --------------------------------------------------------------------------------------------------
# (c) the mirror COMMUTATION invariant against the full book reflection
# --------------------------------------------------------------------------------------------------
def test_mirror_commutes_with_reflection():
    book, merged_ts, anchor_ts = _synthetic_book(seed=2)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    mctx = _ctx({"byb": _mirror_book(book)}, merged_ts, anchor_ts)
    spec = base.get("ofi_ema")
    assert spec.mirror is not None                                # OFI must declare its reflection (odd)
    for params in (1, 50):
        feat = spec.vectorized(ctx, params)["byb"]               # feature(books)
        refl = spec.vectorized(mctx, params)["byb"]              # feature(mirror_books(books))
        ok = np.isfinite(feat) & np.isfinite(refl)
        assert ok.sum() > 50
        # ODD feature: mirror(feature(books)) == feature(mirror_books(books))
        np.testing.assert_allclose(spec.mirror(feat)[ok], refl[ok], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# span = 1 (α = 1): finite wherever the flow exists, consistent NaN only during warm-up
# --------------------------------------------------------------------------------------------------
def test_span1_finite_where_inputs_exist():
    book, merged_ts, anchor_ts = _synthetic_book(seed=6)
    ctx = _ctx({"byb": book}, merged_ts, anchor_ts)
    spec = base.get("ofi_ema")
    got = spec.vectorized(ctx, 1)["byb"]
    ref = _ofi_leg_oracle(book, merged_ts, anchor_ts, 1)
    assert np.isfinite(got).any()                                # the span=1 leg is not all-NaN
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))  # NaN only where the oracle is undefined
    assert not np.isinf(got).any()                               # never a spurious inf at α=1
    # every anchor at-or-after the first OFI flow sample (which is at-or-after the 2nd book row) is finite
    first_flow_ts = int(book[0][1])                              # rx of the 2nd raw row = earliest increment ts
    after_first = anchor_ts >= first_flow_ts
    assert after_first.any()
    assert np.isfinite(got[after_first]).all()


# --------------------------------------------------------------------------------------------------
# streaming (LiveOFIEma) vs vectorized — synthetic parity, including a span=1 leg
# --------------------------------------------------------------------------------------------------
def test_streaming_matches_vectorized():
    b_byb, merged_ts, anchor_ts = _synthetic_book(seed=7)
    b_bin, _, _ = _synthetic_book(seed=8)
    books = {"byb": b_byb, "bin": b_bin}
    ctx = _ctx(books, merged_ts, anchor_ts, sources=("bin",), raw=True)
    spec = base.get("ofi_ema")
    rep = parity_check(ctx, spec, [1, 10, 100], n_grid=len(anchor_ts), tol=1e-9)
    assert rep.passed, str(rep)


# --------------------------------------------------------------------------------------------------
# real-block parity (skipped without DATA_DIR) — streaming reproduces vectorized, incl. span=1
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(getattr(io, "DATA_DIR", None) is None, reason="no DATA_DIR configured")
def test_real_block_parity():
    from boba.research.screening import build_context

    ctx = build_context(hours=2)
    spec = base.get("ofi_ema")
    rep = parity_check(ctx, spec, [1, 100], tol=1e-6)
    assert rep.passed, str(rep)
