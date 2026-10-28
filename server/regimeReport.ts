/**
 * regimeReport.ts — read-only shaping of the regime engine's persisted state
 * (docs/regime/FINDINGS.md step 3; the engine itself is regime_engine.py).
 *
 * The Python engine writes `voltrade_regime_state.json` (latest) and appends
 * one line per trading day / transition to `voltrade_regime_history.jsonl`
 * under DATA_DIR. This module reads those files and shapes (a) the daily
 * regime report served on the owner's GET /api/bot/regime and the token-gated
 * /api/diag/regime, and (b) the NON-GATING `checks.regime` block on
 * /api/health. Nothing here computes a regime, trades, or gates serving:
 * server/healthGate.ts's SERVING_CHECKS are untouched.
 *
 * Pure except for the two file readers (node:test safe).
 */
import fs from "fs";
import path from "path";

export const REGIME_STATE_FILE = "voltrade_regime_state.json";
export const REGIME_HISTORY_FILE = "voltrade_regime_history.jsonl";
export const REGIME_STALE_MARKET_DAYS = 2;   // regime_engine DEFAULT_PARAMS.stale_market_days
export const REGIME_FLIP_ALARM = 6;          // regime_engine DEFAULT_PARAMS.flip_alarm_12m
export const REGIME_HISTORY_LIMIT = 400;
export const REGIME_SIZING_FILE = "voltrade_regime_sizing.json";   // step 6: regime_engine.flush_sizing_counts

export type RegimeStateFile = Record<string, unknown> & {
  as_of?: string;
  composite_regime?: string;
  legacy_label?: string;
  trend_state?: string;
  fast_state?: string;
  vol_state?: string;
  breadth_state?: string | null;
  exposure_multiplier?: number;
  core_multiplier?: number;
  confidence?: number;
  flips_12m?: number;
  composite_changes_12m?: number;
  alarm?: boolean;
  stale?: boolean;
  error?: string;
  transition?: { from?: string; to?: string; from_as_of?: string; to_as_of?: string } | null;
};

export function regimeEngineEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = String(env.VOLTRADE_REGIME_ENGINE ?? "0").trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

export function regimeDataDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}

export function readRegimeState(dir: string): RegimeStateFile | null {
  try {
    const raw = fs.readFileSync(path.join(dir, REGIME_STATE_FILE), "utf8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as RegimeStateFile) : null;
  } catch {
    return null;
  }
}

/** The day's sizing-multiplier counters written by regime_engine.flush_sizing_counts (step 6). */
export type RegimeSizingFile = {
  day?: string;
  by_sleeve?: Record<string, Record<string, number>>;
  stale_fallbacks?: number;
  zero_blocks?: number;
  last?: Record<string, unknown> | null;
  written_at?: string;
  error?: string;
};

export function readRegimeSizing(dir: string): RegimeSizingFile | null {
  try {
    const parsed = JSON.parse(fs.readFileSync(path.join(dir, REGIME_SIZING_FILE), "utf8"));
    return parsed && typeof parsed === "object" ? (parsed as RegimeSizingFile) : null;
  } catch {
    return null;
  }
}

export function readRegimeHistory(dir: string, limit = REGIME_HISTORY_LIMIT): RegimeStateFile[] {
  let raw: string;
  try {
    raw = fs.readFileSync(path.join(dir, REGIME_HISTORY_FILE), "utf8");
  } catch {
    return [];
  }
  const lines = raw.split("\n").filter((l) => l.trim().length > 0);
  const out: RegimeStateFile[] = [];
  for (const line of lines) {
    const row = parseHistoryLine(line);
    if (row) out.push(row);
  }
  return out.slice(-limit);   // the last `limit` VALID rows (the file is a few hundred rows a year)
}

/** One history row, or null for a torn line (the writer appends one row at a time; a partial last line is possible mid-write). */
export function parseHistoryLine(line: string): RegimeStateFile | null {
  try {
    const row = JSON.parse(line);
    return row && typeof row === "object" ? (row as RegimeStateFile) : null;
  } catch {
    return null;
  }
}

/** Weekdays strictly after `fromISO` (YYYY-MM-DD) up to and including `now`'s UTC date. */
export function marketDaysBetween(fromISO: string | undefined, now: Date): number | null {
  if (!fromISO || !/^\d{4}-\d{2}-\d{2}/.test(fromISO)) return null;
  const from = new Date(`${fromISO.slice(0, 10)}T00:00:00Z`);
  if (Number.isNaN(from.getTime())) return null;
  const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  let n = 0;
  const d = new Date(from.getTime());
  while (d < end) {
    d.setUTCDate(d.getUTCDate() + 1);
    const wd = d.getUTCDay();
    if (wd !== 0 && wd !== 6) n += 1;
  }
  return n;
}

export interface RegimeHealthBlock {
  status: "ok" | "warning" | "off";
  detail?: string;
  as_of?: string;
  age_market_days?: number | null;
  composite?: string;
  exposure_multiplier?: number;
  alarm?: boolean;
  gates_top_level_status: false;
}

/** The /api/health block. NEVER gates serving (healthGate.ts decides the HTTP code from server+database only). */
export function regimeHealth(state: RegimeStateFile | null, now: Date, enabled: boolean, sizing: RegimeSizingFile | null = null): RegimeHealthBlock {
  if (!enabled && !state) {
    return { status: "off", detail: "VOLTRADE_REGIME_ENGINE off; no persisted state", gates_top_level_status: false };
  }
  if (!state) {
    return { status: "warning", detail: "engine enabled but no state persisted yet (runs on the next scan)", gates_top_level_status: false };
  }
  const age = marketDaysBetween(state.as_of, now);
  const problems: string[] = [];
  if (age !== null && age > REGIME_STALE_MARKET_DAYS) problems.push(`state is ${age} market days old`);
  if (state.error) problems.push(`last update failed: ${String(state.error).slice(0, 100)}`);
  if (state.alarm) problems.push(`flip alarm: ${state.flips_12m} primary flips in 12 months (> ${REGIME_FLIP_ALARM})`);
  // step 6: a sizer that fell back to 1.0 on a stale state is loud here too (today's counters only)
  const today = now.toISOString().slice(0, 10);
  if (sizing && sizing.day === today && (sizing.stale_fallbacks ?? 0) > 0) {
    problems.push(`sizing fell back to 1.0 on a stale state ${sizing.stale_fallbacks}x today`);
  }
  return {
    status: problems.length ? "warning" : enabled ? "ok" : "off",
    ...(problems.length ? { detail: problems.join("; ") } : enabled ? {} : { detail: "engine off; serving the last persisted state" }),
    as_of: state.as_of,
    age_market_days: age,
    composite: state.composite_regime,
    exposure_multiplier: state.exposure_multiplier,
    alarm: Boolean(state.alarm),
    gates_top_level_status: false,
  };
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** The daily regime report: latest state, the recent history, transitions, time at each exposure level. */
export function regimeReportBody(state: RegimeStateFile | null, history: RegimeStateFile[], now: Date, enabled: boolean, sizing: RegimeSizingFile | null = null) {
  const rows = history.map((h) => ({
    as_of: h.as_of,
    composite: h.composite_regime,
    legacy: h.legacy_label,
    trend: h.trend_state,
    fast: h.fast_state,
    vol: h.vol_state,
    breadth: h.breadth_state ?? null,
    exposure_multiplier: num(h.exposure_multiplier),
    core_multiplier: num(h.core_multiplier),
    confidence: num(h.confidence),
    flips_12m: num(h.flips_12m),
  }));
  const transitions: Array<{ as_of?: string; from?: string; to?: string }> = [];
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].composite !== rows[i - 1].composite) transitions.push({ as_of: rows[i].as_of, from: rows[i - 1].composite, to: rows[i].composite });
  }
  const timeAtLevels: Record<string, number> = {};
  for (const r of rows) {
    if (r.exposure_multiplier === null) continue;
    const k = r.exposure_multiplier.toFixed(2);
    timeAtLevels[k] = (timeAtLevels[k] || 0) + 1;
  }
  const health = regimeHealth(state, now, enabled, sizing);
  return {
    enabled_env: enabled,
    state_present: Boolean(state),
    health,
    state,
    history_days: rows.length,
    history: rows.slice(-60),
    transitions: transitions.slice(-40),
    time_at_levels: timeAtLevels,
    sizing: sizing ?? null,   // step 6: today's applied multipliers by sleeve, stale fallbacks, zero blocks
    note: "shadow: no consumer reads this state until its own VOLTRADE_REGIME_* flag is set (docs/regime/FINDINGS.md)",
  };
}

/** One audit line's text for a composite transition reported in a scan result (null when there is none). */
export function regimeTransitionAudit(regime: RegimeStateFile | null | undefined): string | null {
  const t = regime?.transition;
  if (!t || !t.to) return null;
  return `${t.from ?? "?"} -> ${t.to} (as_of ${t.to_as_of ?? regime?.as_of ?? "?"}; exposure ${regime?.exposure_multiplier ?? "?"}, core ${regime?.core_multiplier ?? "?"}, confidence ${regime?.confidence ?? "?"})`;
}
