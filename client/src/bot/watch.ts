// Bot page — the Watch model (pure): what the bot looked at in its last scan,
// what it decided, and what it refused and why. Everything comes from
// /api/bot/last-scan (the engine's own result: top_10 with its factor
// breakdown, new_trades, sizing_blocks) and the audit rows Node writes when
// it refuses a scanner trade (SKIP, SIZE-CAP, HEAT-CAP, CORR-CAP, HALT-SKIP,
// DUP-SKIP, OPTIONS-SLOT-FULL, OPTIONS-SKIP, SIZE-BLOCK) or queues one
// (QUEUE). Nothing is inferred beyond those rows. Pinned by watch.test.ts.
import type { AuditRow } from "./pageState";

export interface WatchFactors { momentum: number | null; mean_reversion: number | null; vrp: number | null; squeeze: number | null; volume: number | null }
export const FACTOR_KEYS: ReadonlyArray<keyof WatchFactors> = ["momentum", "mean_reversion", "vrp", "squeeze", "volume"];
export const FACTOR_SHORT: Record<keyof WatchFactors, string> = { momentum: "mom", mean_reversion: "mr", vrp: "vrp", squeeze: "sqz", volume: "vol" };

export interface WatchCandidate {
  ticker: string;
  rank: number | null;
  score: number | null;
  price: number | null;
  changePct: number | null;
  side: string | null;
  action: string;
  reasons: string[];
  sector: string | null;
  sectorClash: boolean;
  factors: WatchFactors | null;
  /** score ≥ the Node admission threshold (null threshold → null) */
  admitted: boolean | null;
  /** the scan handed this ticker to Node (it is in new_trades) */
  decided: boolean;
  /** the newest refusal / queue row for this ticker, when one exists */
  outcome: WatchOutcome | null;
}

export type OutcomeKind = "queued" | "refused" | "blocked";
export interface WatchOutcome { kind: OutcomeKind; type: string; reason: string; time: string }

export interface WatchDecision { ticker: string; action: string; score: number | null; kind: string | null; reasons: string[]; price: number | null; rulesScore: number | null; mlScore: number | null }
export interface WatchRefusal { ticker: string; type: string; reason: string; time: string; score: number | null }

export interface WatchModel {
  scanAt: string | null;
  partial: boolean;
  scanned: number | null;
  filtered: number | null;
  deep: number | null;
  slots: number | null;
  optionsSlots: number | null;
  threshold: number | null;
  candidates: WatchCandidate[];
  decided: WatchDecision[];
  refused: WatchRefusal[];
}

/** Audit types that refuse a scanner trade (ticker-first messages). */
export const REFUSAL_TYPES = new Set(["SKIP", "SIZE-CAP", "HEAT-CAP", "CORR-CAP", "HALT-SKIP", "DUP-SKIP", "OPTIONS-SLOT-FULL", "OPTIONS-SKIP"]);
/** Sizing refusals carry "<ticker> [strategy]: reason". */
export const BLOCK_TYPES = new Set(["SIZE-BLOCK"]);
export const QUEUE_TYPES = new Set(["QUEUE"]);
/** How far back the refusal rows are read: the scan cadence is 5 min in
 *  RTH, so two hours covers a morning's worth of decisions. */
export const WATCH_OUTCOME_WINDOW_MS = 2 * 3600_000;

const num = (v: unknown): number | null => { if (v === null || v === undefined || v === "") return null; const n = typeof v === "number" ? v : Number(v); return Number.isFinite(n) ? n : null; };
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/** "KWEB [stocks]: reason", "KWEB: reason", "KWEB queued for market open (score 75)" → ticker + rest. */
export function parseTickerMessage(message: string): { ticker: string; rest: string } | null {
  const m = /^([A-Z][A-Z0-9.\-]{0,9}(?:\d{6}[CP]\d{8})?)\s*(?:\[[^\]]*\])?\s*(?::\s*|\s+)(.*)$/.exec(message.trim());
  if (!m) return null;
  return { ticker: m[1].toUpperCase(), rest: m[2].trim() };
}

function parseFactors(v: unknown): WatchFactors | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const f: WatchFactors = { momentum: num(o.momentum), mean_reversion: num(o.mean_reversion), vrp: num(o.vrp), squeeze: num(o.squeeze), volume: num(o.volume) };
  return FACTOR_KEYS.some((k) => f[k] !== null) ? f : null;
}

export function outcomeRows(audit: readonly AuditRow[], nowMs: number, windowMs = WATCH_OUTCOME_WINDOW_MS): { byTicker: Map<string, WatchOutcome>; refused: WatchRefusal[] } {
  const byTicker = new Map<string, WatchOutcome>();
  const refused: WatchRefusal[] = [];
  const rows = [...audit].filter((r) => r && typeof r.time === "string" && Number.isFinite(Date.parse(r.time)) && nowMs - Date.parse(r.time) <= windowMs).sort((a, b) => Date.parse(b.time) - Date.parse(a.time));
  for (const r of rows) {
    const kind: OutcomeKind | null = QUEUE_TYPES.has(r.type) ? "queued" : REFUSAL_TYPES.has(r.type) ? "refused" : BLOCK_TYPES.has(r.type) ? "blocked" : null;
    if (!kind) continue;
    const p = parseTickerMessage(r.message);
    if (!p) continue;
    if (!byTicker.has(p.ticker)) byTicker.set(p.ticker, { kind, type: r.type, reason: p.rest, time: r.time });
    if (kind !== "queued" && !refused.some((x) => x.ticker === p.ticker && x.type === r.type)) {
      const sc = /score\s+([\d.]+)/i.exec(p.rest);
      refused.push({ ticker: p.ticker, type: r.type, reason: p.rest, time: r.time, score: sc ? num(sc[1]) : null });
    }
  }
  return { byTicker, refused };
}

const rowTime = (r: { time: string }): number => { const t = r.time ? Date.parse(r.time) : NaN; return Number.isFinite(t) ? t : 0; };

export interface WatchInput { lastScan: unknown; audit: readonly AuditRow[]; threshold: number | null; nowMs: number }

export function watchModel(i: WatchInput): WatchModel {
  const s = (i.lastScan && typeof i.lastScan === "object" ? i.lastScan : {}) as Record<string, unknown>;
  const { byTicker, refused } = outcomeRows(i.audit, i.nowMs);
  const trades = Array.isArray(s.new_trades) ? s.new_trades : [];
  const decided: WatchDecision[] = trades.map((t) => (t && typeof t === "object" ? (t as Record<string, unknown>) : null)).filter((t): t is Record<string, unknown> => !!t && typeof t.ticker === "string" && t.ticker.trim() !== "").map((t) => ({
    ticker: String(t.ticker).toUpperCase(), action: str(t.action_label) ?? str(t.side) ?? "—", score: num(t.score), kind: str(t.trade_type) ?? str(t.instrument_strategy) ?? str(t.options_strategy) ?? null,
    reasons: Array.isArray(t.reasons) ? t.reasons.map(String).slice(0, 3) : [], price: num(t.price), rulesScore: num(t.rules_score), mlScore: num(t.ml_score_raw),
  }));
  const decidedSet = new Set(decided.map((d) => d.ticker));
  const top = Array.isArray(s.top_10) ? s.top_10 : [];
  const candidates: WatchCandidate[] = top.map((c) => (c && typeof c === "object" ? (c as Record<string, unknown>) : null)).filter((c): c is Record<string, unknown> => !!c && typeof c.ticker === "string" && c.ticker.trim() !== "").map((c, k) => {
    const ticker = String(c.ticker).toUpperCase();
    const score = num(c.score);
    return {
      ticker, rank: num(c.rank) ?? k + 1, score, price: num(c.price), changePct: num(c.change_pct), side: str(c.side), action: str(c.action_label) ?? "WATCH",
      reasons: Array.isArray(c.reasons) ? c.reasons.map(String).slice(0, 3) : [], sector: str(c.sector), sectorClash: c.sector_clash === true, factors: parseFactors(c.factors),
      admitted: i.threshold !== null && score !== null ? score >= i.threshold : null, decided: decidedSet.has(ticker), outcome: byTicker.get(ticker) ?? null,
    };
  }).sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99) || (b.score ?? -1) - (a.score ?? -1));
  // sizing blocks the scan itself reported (engine-side, before Node)
  const blocks = Array.isArray(s.sizing_blocks) ? s.sizing_blocks : [];
  for (const b of blocks) {
    if (!b || typeof b !== "object") continue;
    const o = b as Record<string, unknown>;
    const ticker = str(o.ticker)?.toUpperCase();
    if (!ticker || refused.some((x) => x.ticker === ticker && x.type === "SIZE-BLOCK")) continue;
    refused.push({ ticker, type: "SIZE-BLOCK", reason: `${str(o.strategy) ? `[${o.strategy}] ` : ""}${str(o.reason) ?? "refused by sizing"}`, time: str(s.timestamp) ?? "", score: num(o.score) });
  }
  return {
    scanAt: str(s.timestamp), partial: s.partial === true, scanned: num(s.scanned), filtered: num(s.filtered), deep: num(s.deep_analyzed), slots: num(s.slots_available), optionsSlots: num(s.options_slots_available),
    threshold: i.threshold, candidates, decided, refused: refused.sort((a, b) => rowTime(b) - rowTime(a)),
  };
}

/** The chart subject key for a candidate (a candidate is not a position). */
export const candidateKey = (ticker: string): string => `cand:${ticker.toUpperCase()}`;
