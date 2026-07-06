// ── Exit-fill feedback payloads ─────────────────────────────────────────
// REPAIR 2026-07-06 (R12-D2): ml_model_v2.track_fill has carried exit-fill
// machinery since Bug #13's fix (v1.0.34) — it detects exit_context /
// exit_reason / is_close and closes the matching open entry record with a
// real outcome + pnl_pct. But NO caller in bot.ts ever passed those keys:
// both existing track_fill sites are ENTRY fills, so no feedback record
// could ever be closed and the ML loop had zero live outcome labels.
//
// This module builds the exit payload (pure, unit-tested) so the WS exit
// monitor and the per-position risk kill can record exits. It deliberately
// does NOT submit orders and knows nothing about order transmission —
// recording happens strictly AFTER the order POST, which stays frozen.
//
// SCALE-OUTS ARE EXCLUDED BY DESIGN: a partial take-profit leaves the
// position open; closing the entry record on the first scale would label
// the whole trade with a partial P&L. Only full exits (remaining-qty WS
// exits, position kills) may close a record.

export interface ExitFillArgs {
  ticker: string;
  /** Order side of the exit ("sell" closes long, "buy" covers short). */
  exitSide: string;
  qty: number;
  /** Best-known execution price at submit time (WS current price). */
  fillPrice: number;
  /** The bot's own P&L accounting for the position, in percent. */
  pnlPct: number;
  /** stop_loss | trailing_stop | take_profit | time_stop | position_kill */
  exitReason: string;
  /** Position entry date (ISO) — used to derive days_held. */
  entryDate?: string;
  /** Injected clock for tests. */
  nowMs?: number;
}

export function buildExitFillPayload(a: ExitFillArgs) {
  const nowMs = a.nowMs ?? Date.now();
  let daysHeld = 0;
  if (a.entryDate) {
    const t = new Date(a.entryDate).getTime();
    if (Number.isFinite(t)) daysHeld = Math.max(0, Math.round((nowMs - t) / 86400000));
  }
  return {
    ticker: a.ticker,
    side: a.exitSide,
    // v1.0.153's qty resolution accepts qty_filled — same contract as the
    // regular-hours entry payload.
    qty_filled: a.qty,
    fill_price: a.fillPrice,
    expected_price: a.fillPrice,
    time_filled: new Date(nowMs).toISOString(),
    session: "regular",
    exit_reason: a.exitReason,
    // exit_context is what _is_exit_fill keys on; its pnl_pct wins over the
    // entry-vs-exit price recomputation inside track_fill (matches the
    // bot's own accounting incl. scale-outs that happened along the way).
    exit_context: {
      pnl_pct: a.pnlPct,
      exit_reason: a.exitReason,
      days_held: daysHeld,
    },
  };
}
