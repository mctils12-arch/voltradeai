// Bot page — the page changes when the bot does (slice F of
// research/bot_page_program.md; human directive 2026-10-21: "if something
// changes, it pops up and sort of changes how the view looks — not static").
// Two pure pieces: (1) EVENTS — audit rows that arrived since the page last
// looked become a short feed (what happened, to which ticker, how loud), so a
// trade, an exit, a queue, a refusal or a halt announces itself the tick it
// lands; (2) FLASHES — a value that differs from the tick before flashes for
// FLASH_MS with its direction. Nothing is inferred beyond the row's type and
// its ticker-first message. Pinned by live.test.ts.
import type { AuditRow } from "./pageState";
import { parseTickerMessage } from "./watch";

export type EventKind = "trade" | "exit" | "queue" | "refusal" | "halt" | "resume" | "error" | "warn" | "model";
export interface LiveEvent { key: string; time: string; t: number; kind: EventKind; ticker: string | null; title: string; detail: string; loud: boolean }

/** audit row type → event kind; a loud one also tints its panel for a moment.
 *  Scans (TIER2 every minute), stream signals and diagnostics are NOT events:
 *  the Watch and Stages panels carry them, and a feed of them would bury the
 *  decisions the feed exists for. */
export const EVENT_TYPES: Readonly<Record<string, { kind: EventKind; loud: boolean }>> = {
  TRADE: { kind: "trade", loud: true }, "ETF-TRADE": { kind: "trade", loud: true }, "OPTIONS-TRADE": { kind: "trade", loud: true }, "MORNING-TRADE": { kind: "trade", loud: true }, "OPTIONS-SCANNER-EXEC": { kind: "trade", loud: true }, T1: { kind: "trade", loud: true }, T2: { kind: "trade", loud: true },
  "EXIT-FILL": { kind: "exit", loud: true }, "WS-EXIT": { kind: "exit", loud: true }, "POS-KILL": { kind: "exit", loud: true }, "POS-KILL-OPTION": { kind: "exit", loud: true }, "OPTIONS-EXIT": { kind: "exit", loud: true }, "STOP-LOSS": { kind: "exit", loud: true }, "TAKE-PROFIT": { kind: "exit", loud: true }, "TRAILING-STOP": { kind: "exit", loud: true }, "SCALE-OUT": { kind: "exit", loud: false },
  QUEUE: { kind: "queue", loud: false },
  SKIP: { kind: "refusal", loud: false }, "SIZE-CAP": { kind: "refusal", loud: false }, "HEAT-CAP": { kind: "refusal", loud: false }, "CORR-CAP": { kind: "refusal", loud: false }, "HALT-SKIP": { kind: "refusal", loud: false }, "DUP-SKIP": { kind: "refusal", loud: false }, "OPTIONS-SLOT-FULL": { kind: "refusal", loud: false }, "OPTIONS-SKIP": { kind: "refusal", loud: false }, "SIZE-BLOCK": { kind: "refusal", loud: false },
  "DRAWDOWN-KILL": { kind: "halt", loud: true }, "KILL-SWITCH": { kind: "halt", loud: true }, HALT: { kind: "halt", loud: true }, "TIER-KILL": { kind: "halt", loud: true },
  "KILL-SWITCH-RESUME": { kind: "resume", loud: true }, "AUTO-RESUME": { kind: "resume", loud: true }, RESUME: { kind: "resume", loud: true },
  "TIER2-ERROR": { kind: "error", loud: true }, "TIER2-FAIL": { kind: "error", loud: true }, "STREAM-DISCONNECT": { kind: "error", loud: false },
  "POS-WARN": { kind: "warn", loud: false },
  LEARN: { kind: "model", loud: false }, "ML-RETRAIN": { kind: "model", loud: false }, "ML-FALLBACK": { kind: "model", loud: true },
};
export const KIND_WORD: Readonly<Record<EventKind, string>> = { trade: "trade", exit: "exit", queue: "queued", refusal: "refused", halt: "halted", resume: "resumed", error: "error", warn: "warning", model: "model" };
/** the feed's length and how long a fresh item or a flash stays marked */
export const FEED_MAX = 6;
export const FRESH_MS = 10_000;
export const FLASH_MS = 1400;

export function eventOf(row: AuditRow): LiveEvent | null {
  const spec = EVENT_TYPES[row.type];
  if (!spec) return null;
  const t = Date.parse(row.time);
  if (!Number.isFinite(t)) return null;
  const parsed = parseTickerMessage(row.message);
  const ticker = parsed ? parsed.ticker : null;
  const detail = (parsed ? parsed.rest : row.message).replace(/\s+/g, " ").trim().slice(0, 100);
  return { key: `${row.time}|${row.type}|${row.message.slice(0, 40)}`, time: row.time, t, kind: spec.kind, ticker, title: ticker ? `${KIND_WORD[spec.kind]} · ${ticker}` : KIND_WORD[spec.kind], detail, loud: spec.loud };
}

/** the events among `rows` newer than `sinceMs`, newest first, at most `max` */
export function liveEvents(rows: readonly AuditRow[], sinceMs: number, max = FEED_MAX): LiveEvent[] {
  const out: LiveEvent[] = [];
  for (const r of rows) {
    const e = eventOf(r);
    if (e && e.t > sinceMs) out.push(e);
  }
  out.sort((a, b) => b.t - a.t);
  return out.slice(0, max);
}

export type FlashDir = "up" | "down" | "changed";
export interface Flash { dir: FlashDir; at: number }

/** the flashes after a tick: a key whose value differs from the previous
 *  tick flashes from now (numbers carry a direction, anything else
 *  "changed"; a value appearing from null/undefined counts as changed, one
 *  vanishing does not flash); flashes older than FLASH_MS are dropped. The
 *  first tick (no previous values) flashes nothing. */
export function flashes<K extends string>(prev: Partial<Record<K, unknown>> | null, next: Record<K, unknown>, current: Partial<Record<K, Flash>>, nowMs: number): Partial<Record<K, Flash>> {
  const out: Partial<Record<K, Flash>> = {};
  for (const k of Object.keys(current) as K[]) { const f = current[k]; if (f && nowMs - f.at < FLASH_MS) out[k] = f; }
  if (!prev) return out;
  for (const k of Object.keys(next) as K[]) {
    const a = prev[k], b = next[k];
    if (b === null || b === undefined || a === b) continue;
    if (typeof a === "number" && typeof b === "number") { if (Number.isNaN(a) && Number.isNaN(b)) continue; out[k] = { dir: b > a ? "up" : "down", at: nowMs }; }
    else out[k] = { dir: "changed", at: nowMs };
  }
  return out;
}

/** the data-flash attribute for a value, or undefined once the flash is over */
export function flashAttr(f: Flash | undefined, nowMs: number): FlashDir | undefined {
  return f && nowMs - f.at < FLASH_MS ? f.dir : undefined;
}
