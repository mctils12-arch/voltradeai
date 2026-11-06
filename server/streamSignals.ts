// TIER-0 STREAM SIGNALS, RESTORED AS A MEASURED SIGNAL (2026-11-06; refocus
// spec section 2, "review previously removed features").
//
// What was broken:
// - The WebSocket 1-minute bars flagged "volume >= 2.5x average, bar move
//   >= 0.8%, 5-bar momentum >= 0.3%". Each flag was queued for the next Tier 2
//   scan, whose audit row said "processing first". The scan then cleared the
//   queue and ran without it: run_full_scan takes no tickers. The flag never
//   reached any decision.
// - The 20-bar volume baseline mixed thin pre-market bars with the open.
//   13 of the 41 live flags (2026-10-28 to 11-05) fired at 09:31-09:32 ET at
//   9.5x-849x; the other 28 had a median of 4.1x.
//
// Why it is measured, not traded:
// - Llorente, Michaely, Saar & Wang (RFS 2002): "Stocks of larger firms, or
//   stocks with smaller bid-ask spreads, show almost no continuation and
//   mostly return reversal following high-volume days."
// - The stream watches 43 mega caps.
// - The idea passes no ladder gate yet (gate 2: does it predict anything?).
//
// So the flag is recorded with its price for a gate-2 test
// (scripts/stream_signal_gate2.py: continuation vs reversal at the same-day
// close, +1 and +5 sessions) and changes no trade.
//
// The rule's thresholds are unchanged. The baseline now holds only today's
// regular-session bars (09:30-16:00 ET), reset each ET day, so the first
// flag can come from the fifth bar (09:34).
import fs from "fs";
import path from "path";

/** The original rule, unchanged (bot.ts Tier 0 since 2026-04). */
export const STREAM_SIGNAL = {
  VOL_RATIO: 2.5,        // bar volume >= 2.5x the average of the earlier bars kept
  MIN_BAR_PCT: 0.8,      // |close - open| / open >= 0.8%
  MIN_MOM5_PCT: 0.3,     // |5-bar momentum| >= 0.3%
  HISTORY: 20,           // bars kept per ticker
  MIN_HISTORY: 5,        // bars needed, the current one included
  COOLDOWN_MS: 10 * 60_000,
} as const;
export const STREAM_SIGNAL_LOG_FILE = "voltrade_stream_signals.jsonl";
export const STREAM_SIGNAL_LOG_MAX_BYTES = 5_000_000;   // then rotated to .1 (one older file kept)
const LOG_OFF_VALUES = new Set(["0", "off", "false", "none", "no"]);

export interface StreamHistory { day: string; vols: number[]; closes: number[] }
export interface StreamSignal { volRatio: number; barPct: number; mom5Pct: number; avgVolume: number; direction: "bullish" | "bearish" }

const ET_PARTS = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});

/** The ET trading day and minute of the day of a bar's start time (Alpaca `t`), or null. */
export function etSession(barIso: string): { day: string; minute: number } | null {
  const ms = Date.parse(barIso);
  if (!Number.isFinite(ms)) return null;
  const p: Record<string, string> = {};
  for (const part of ET_PARTS.formatToParts(new Date(ms))) p[part.type] = part.value;
  const minute = Number(p.hour) * 60 + Number(p.minute);
  if (!p.year || !Number.isFinite(minute)) return null;
  return { day: `${p.year}-${p.month}-${p.day}`, minute };
}

/** 09:30 <= bar start < 16:00 ET. The broker calendar (holidays, half days) is not consulted: no bars arrive then. */
export function isRegularSessionMinute(minute: number): boolean {
  return minute >= 9 * 60 + 30 && minute < 16 * 60;
}

/** Add a regular-session bar to a ticker's history; a new ET day starts it empty. */
export function pushRegularBar(h: StreamHistory | undefined, day: string, volume: number, close: number): StreamHistory {
  const cur = h && h.day === day ? h : { day, vols: [], closes: [] };
  cur.vols.push(volume);
  cur.closes.push(close);
  while (cur.vols.length > STREAM_SIGNAL.HISTORY) cur.vols.shift();
  while (cur.closes.length > STREAM_SIGNAL.HISTORY) cur.closes.shift();
  return cur;
}

/** The rule on the history whose last entry is the bar just received; null when it does not fire. */
export function detectStreamSignal(h: StreamHistory, open: number, close: number): StreamSignal | null {
  if (h.vols.length < STREAM_SIGNAL.MIN_HISTORY || h.closes.length < STREAM_SIGNAL.MIN_HISTORY) return null;
  const earlier = h.vols.slice(0, -1);
  const avgVolume = earlier.reduce((a, b) => a + b, 0) / earlier.length;
  const volume = h.vols[h.vols.length - 1];
  const volRatio = avgVolume > 0 ? volume / avgVolume : 1;
  const barPct = open > 0 ? ((close - open) / open) * 100 : 0;
  const c = h.closes;
  const mom5Pct = c[c.length - 5] > 0 ? ((c[c.length - 1] - c[c.length - 5]) / c[c.length - 5]) * 100 : 0;
  if (volRatio < STREAM_SIGNAL.VOL_RATIO || Math.abs(barPct) < STREAM_SIGNAL.MIN_BAR_PCT || Math.abs(mom5Pct) < STREAM_SIGNAL.MIN_MOM5_PCT) return null;
  return { volRatio, barPct, mom5Pct, avgVolume, direction: barPct > 0 ? "bullish" : "bearish" };
}

/** env VOLTRADE_STREAM_SIGNAL_LOG = off/0/false/none/no stops the log (the one off switch). */
export function streamSignalLogEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return !LOG_OFF_VALUES.has(String(env.VOLTRADE_STREAM_SIGNAL_LOG ?? "").trim().toLowerCase());
}

export function streamSignalLogPath(dataDir: string = fs.existsSync("/data/voltrade") ? "/data/voltrade" : "/tmp"): string {
  return path.join(dataDir, STREAM_SIGNAL_LOG_FILE);
}

/** Append one flag as a JSON line; above maxBytes the file is rotated to `.1` first. Returns false (and logs why) on failure. */
export function appendStreamSignal(row: Record<string, unknown>, file: string = streamSignalLogPath(),
                                   maxBytes: number = STREAM_SIGNAL_LOG_MAX_BYTES): boolean {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const st = fs.existsSync(file) ? fs.statSync(file) : null;
    if (st && !st.isFile()) throw new Error(`${file} exists and is not a regular file`);
    if (st && st.size > maxBytes) fs.renameSync(file, `${file}.1`);
    fs.appendFileSync(file, JSON.stringify(row) + "\n");
    return true;
  } catch (e: unknown) {
    console.error("[stream-signal] log write failed:", e instanceof Error ? e.message : e);
    return false;
  }
}
