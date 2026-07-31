// ── Entry-fill feedback payloads ────────────────────────────────────────
// REPAIR 2026-07-31 (KNOWN BROKEN #12(c) contributor, found while
// live-verifying #12(b)'s open gate): the ETF entry path (bot.ts's "ETF
// execution" branch — 2x-leveraged-ETF trades the instrument selector
// chose over the underlying stock) calls addPositionToMonitor but never
// track_fill. Both other entry paths (regular-hours, morning-queue) DO
// call track_fill on entry, so their later WS exits can find a matching
// open record. The ETF path's positions get monitored and eventually
// exited via the exact same WS mechanism (recordExitFill, same ticker),
// but with no entry record ever written, ml_model_v2._find_entry_record
// always misses -> every ETF exit becomes a permanent, unrecoverable
// orphan_exit. Confirmed live via /api/diag/ml this session: 70/70
// non-seeded feedback records dated 2026-07-10..30 are orphan_exit, zero
// matched win/loss ever recorded — direct evidence against KNOWN BROKEN
// #12(b)'s working assumption that D2's WS exit path would soon produce
// real outcomes on its own. This payload gives the ETF path parity with
// the two already-working entry sites.
//
// Deliberately NOT extended to the options-entry monitor call in the same
// function (options positions are monitored via the underlying ticker's
// price for stop-loss purposes) — an entry/exit pair recorded there would
// price pnl_pct off the UNDERLYING's move, not the option's actual
// premium P&L, which is exactly the "options fill realism" gap
// open_questions.md #12(c) already gates behind a dedicated design
// decision (quote-based premium fills). Writing a stock-priced entry
// record under an "options" instrument tag would mislabel training data
// with the wrong P&L basis — left open, filed as a follow-up.

export interface EntryFillArgs {
  ticker: string;
  side: string;
  qty: number;
  fillPrice: number;
  session: string;
  volume?: number;
  score?: number;
  instrument: string;
  /** Injected clock for tests. */
  nowMs?: number;
  codeVersion: string;
}

export function buildEntryFillPayload(a: EntryFillArgs) {
  const nowMs = a.nowMs ?? Date.now();
  return {
    ticker: a.ticker,
    side: a.side,
    qty: a.qty,
    expected_price: a.fillPrice,
    fill_price: a.fillPrice,
    time_placed: new Date(nowMs).toISOString(),
    session: a.session,
    volume: a.volume ?? 1000000,
    score: a.score,
    instrument: a.instrument,
    code_version: a.codeVersion,
  };
}
