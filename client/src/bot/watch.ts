// Bot page slice B — the Watch surface: what the last scan was looking at.
// Pure (tested). Reads only what the scan result carries; it never invents a
// stop, target or admission threshold for a candidate (the engine computes
// those at entry from that day's ATR, and MIN_SCORE is regime-adaptive and
// not part of the scan result), so a row says taken / blocked / not taken and
// shows the score next to the others.
import type { LastScan } from "./useBotData";

export type WatchStatus = "taken" | "blocked" | "not taken";
export interface WatchRow {
  ticker: string; score: number | null; side: string; action: string; price: number | null; changePct: number | null;
  reasons: string[]; status: WatchStatus; blockReason: string | null; rank: number;
}
export interface WatchSummary { scanned: number | null; filtered: number | null; deepAnalyzed: number | null; slots: number | null; timestamp: string | null }
export interface Watch { rows: WatchRow[]; summary: WatchSummary; empty: boolean }

type Loose = Record<string, unknown>;
const n = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const s = (v: unknown): string => (typeof v === "string" ? v : "");

/** Rows from the last scan: its top 10 by score, each marked by what the
 *  trade loop did with it; trades and sizing refusals that fell outside the
 *  top 10 are appended so nothing the bot acted on or refused is hidden. */
export function deriveWatch(scan: LastScan | undefined | null): Watch {
  const sc = (scan ?? {}) as LastScan & Loose;
  const top = Array.isArray(sc.top_10) ? (sc.top_10 as unknown as Loose[]) : [];
  const trades = Array.isArray(sc.new_trades) ? (sc.new_trades as Loose[]) : [];
  const blocks = Array.isArray(sc.sizing_blocks) ? sc.sizing_blocks : [];
  const summary: WatchSummary = {
    scanned: n(sc.scanned), filtered: n(sc.filtered), deepAnalyzed: n(sc.deep_analyzed), slots: n((sc as Loose).slots_available), timestamp: s(sc.timestamp) || null,
  };
  const takenBy = new Map<string, Loose>(); for (const t of trades) { const k = s(t.ticker).toUpperCase(); if (k && !takenBy.has(k)) takenBy.set(k, t); }
  const blockedBy = new Map<string, string>(); for (const b of blocks) { const k = s(b.ticker).toUpperCase(); if (k && !blockedBy.has(k)) blockedBy.set(k, s(b.reason)); }
  const rows: WatchRow[] = []; const seen = new Set<string>();
  const push = (r: Loose, rank: number) => {
    const ticker = s(r.ticker).toUpperCase(); if (!ticker || seen.has(ticker)) return; seen.add(ticker);
    const status: WatchStatus = takenBy.has(ticker) ? "taken" : blockedBy.has(ticker) ? "blocked" : "not taken";
    rows.push({
      ticker, score: n(r.score), side: s(r.side) || "buy", action: s(r.action_label) || (s(r.side) === "sell" ? "SELL" : "BUY"), price: n(r.price), changePct: n(r.change_pct),
      reasons: Array.isArray(r.reasons) ? (r.reasons as unknown[]).filter((x): x is string => typeof x === "string") : [], status, blockReason: blockedBy.get(ticker) ?? null, rank,
    });
  };
  top.forEach((r, i) => push(r, n(r.rank) ?? i + 1));
  for (const t of trades) push(t, rows.length + 1);
  for (const b of blocks) push(b as unknown as Loose, rows.length + 1);
  return { rows, summary, empty: rows.length === 0 };
}

/** The scan's age in ms, or null when it carries no (valid) timestamp. */
export function scanAgeMs(w: Watch, nowMs: number): number | null {
  if (!w.summary.timestamp) return null;
  // the engine stamps naive local-server ISO strings (Railway = UTC)
  const t = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(w.summary.timestamp) ? w.summary.timestamp : `${w.summary.timestamp}Z`);
  return Number.isFinite(t) ? Math.max(0, nowMs - t) : null;
}
