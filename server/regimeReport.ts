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
  // 2026-11-06 regime risk budget (regime_engine.risk_budget_detail)
  risk_budget?: number | null;
  risk_budget_detail?: Record<string, unknown> | null;
  budget_version?: string;
};

/**
 * Defaults of the six regime flags — a MIRROR of system_config.py BASE_CONFIG
 * (REGIME_ENGINE_ENABLED True -> "1", REGIME_SELECTION_DEFAULT / _OPTIONS_ / _SIZING_
 * "all", REGIME_LABELS_DEFAULT True -> "1"; human directive 2026-10-28, prior: all off;
 * REGIME_BUDGET_DEFAULT True -> "1", 2026-11-06 risk budget, new).
 * Read exactly like regime_engine.flag_raw: the env var when it is PRESENT (any value —
 * "off" / "0" / "" included — wins), the default only when it is ABSENT. The Node process
 * and the daemon share one environment, so both sides agree on what is on.
 * test_regime_flag_defaults.py pins this table against BASE_CONFIG.
 */
export const REGIME_FLAG_DEFAULTS: Readonly<Record<string, string>> = Object.freeze({
  VOLTRADE_REGIME_ENGINE: "1",
  VOLTRADE_REGIME_SELECTION: "all",
  VOLTRADE_REGIME_OPTIONS: "all",
  VOLTRADE_REGIME_SIZING: "all",
  VOLTRADE_REGIME_LABELS: "1",
  VOLTRADE_REGIME_BUDGET: "1",
});

export type RegimeFlag = { value: string; source: "env" | "config" };

export function regimeFlagRaw(name: string, env: NodeJS.ProcessEnv = process.env): RegimeFlag {
  const v = env[name];
  if (v !== undefined && v !== null) return { value: String(v), source: "env" };
  return { value: REGIME_FLAG_DEFAULTS[name] ?? "", source: "config" };
}

/** Every regime flag's effective value and where it came from (served on the report). */
export function regimeFlags(env: NodeJS.ProcessEnv = process.env): Record<string, RegimeFlag> {
  const out: Record<string, RegimeFlag> = {};
  for (const name of Object.keys(REGIME_FLAG_DEFAULTS)) out[name] = regimeFlagRaw(name, env);
  return out;
}

function flagOn(value: string): boolean {
  const v = value.trim().toLowerCase();
  return v.length > 0 && v !== "0" && v !== "off" && v !== "false" && v !== "none";
}

export function regimeEngineEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = regimeFlagRaw("VOLTRADE_REGIME_ENGINE", env).value.trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

/** The consumer flags that act (selection / options / sizing / labels), for the report note. */
export function regimeConsumersOn(flags: Record<string, RegimeFlag>): string[] {
  return ["VOLTRADE_REGIME_SELECTION", "VOLTRADE_REGIME_OPTIONS", "VOLTRADE_REGIME_SIZING", "VOLTRADE_REGIME_LABELS"]
    .filter((n) => flags[n] !== undefined && flagOn(flags[n].value));
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
  risk_budget?: number | null;   // 2026-11-06: the regime risk budget (null before its first compute)
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
    risk_budget: num(state.risk_budget),
    alarm: Boolean(state.alarm),
    gates_top_level_status: false,
  };
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** One driver of the regime risk budget (regime_engine.risk_budget_detail "drivers"). */
export type RiskBudgetDriver = {
  name: string;           // trend | term_structure | vix_level | breadth
  label: string;          // "VIX term structure"
  value: number | null;   // the reading: trend score, VIX/VIX3M, VIX, breadth share
  score: number | null;   // 0..1, 1 = no reason to cut
  weight: number;         // normalized weight
  cut: number;            // weight x (1 - score): the share of the budget this driver removes
  short: string;          // "VIX term structure inverted (1.04)"
  reason: string;         // the full plain-English sentence
};

/** The report's risk-budget block: what the UI shows as "risk budget 0.60: VIX term structure inverted, breadth 38%". */
export interface RiskBudgetBlock {
  budget: number | null;  // [0, 1]; the value consumers apply
  raw: number | null;     // today's reading before the rise glide
  band: string | null;    // full | reduced | defensive | minimal
  summary: string | null; // plain English, names the drivers that cut it
  drivers: RiskBudgetDriver[];
  missing: string[];
  as_of: string | null;
  age_market_days: number | null;
  stale: boolean;
  enabled: boolean;       // engine + VOLTRADE_REGIME_BUDGET on
  applied: boolean;       // enabled, a budget exists and the state is fresh
  consumers: string[];
  history: Array<{ as_of?: string; budget: number; band: string | null }>;
  note: string;
}

export const RISK_BUDGET_CONSUMERS: readonly string[] = ["tier_exposure_cap"];   // regime_engine.BUDGET_CONSUMERS

/** Mirrors regime_engine.budget_enabled + engine_allowed (1/true/on/yes; env when present, else the default). */
export function riskBudgetEnabled(flags: Record<string, RegimeFlag> | null, engineEnabled: boolean): boolean {
  if (!engineEnabled) return false;
  const v = (flags?.VOLTRADE_REGIME_BUDGET?.value ?? REGIME_FLAG_DEFAULTS.VOLTRADE_REGIME_BUDGET).trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

function bandOf(detail: unknown): string | null {
  const b = detail && typeof detail === "object" ? (detail as Record<string, unknown>).band : null;
  return typeof b === "string" ? b : null;
}

/** The risk budget as of the latest state plus its recent history (pure; same staleness rule as the sizers). */
export function regimeRiskBudget(state: RegimeStateFile | null, history: RegimeStateFile[], now: Date, engineEnabled: boolean, flags: Record<string, RegimeFlag> | null = null): RiskBudgetBlock {
  const det = state?.risk_budget_detail && typeof state.risk_budget_detail === "object" ? state.risk_budget_detail : null;
  const budget = num(state?.risk_budget);
  const age = marketDaysBetween(state?.as_of, now);
  const stale = Boolean(state?.error) || Boolean(state?.stale) || (age !== null && age > REGIME_STALE_MARKET_DAYS);
  const enabled = riskBudgetEnabled(flags, engineEnabled);
  const applied = enabled && budget !== null && !stale;
  const hist: RiskBudgetBlock["history"] = [];
  for (const h of history) {
    const b = num(h.risk_budget);
    if (b !== null) hist.push({ as_of: h.as_of, budget: b, band: bandOf(h.risk_budget_detail) });
  }
  const drivers = Array.isArray(det?.drivers) ? (det?.drivers as RiskBudgetDriver[]) : [];
  const missing = Array.isArray(det?.missing) ? (det?.missing as unknown[]).map(String) : [];
  let note: string;
  if (budget === null) note = "no risk budget in the regime state yet (it is computed on the next scan)";
  else if (applied) note = "applied: the tier engine's room above the passive floor (MAX_TOTAL_EXPOSURE) is multiplied by the budget";
  else if (stale) note = "not applied: the regime state is stale, so consumers keep their legacy caps";
  else note = "not applied: VOLTRADE_REGIME_BUDGET is off (computed and logged only)";
  return {
    budget,
    raw: num(det?.raw),
    band: bandOf(det),
    summary: typeof det?.summary === "string" ? det.summary : null,
    drivers,
    missing,
    as_of: state?.as_of ?? null,
    age_market_days: age,
    stale,
    enabled,
    applied,
    consumers: [...RISK_BUDGET_CONSUMERS],
    history: hist.slice(-60),
    note,
  };
}

/** The RISK-BUDGET audit line for a scan's `risk_budget` field (regime_engine.risk_budget_view):
 *  once per (as_of, band, applied) — `{ key, line }`, or null when nothing new. */
export function regimeBudgetAudit(view: unknown, lastKey: string): { key: string; line: string } | null {
  if (!view || typeof view !== "object") return null;
  const v = view as Record<string, unknown>;
  if (typeof v.error === "string" && v.error) {
    const key = `error|${v.error}`;
    return key === lastKey ? null : { key, line: `risk budget view failed: ${v.error.slice(0, 160)}` };
  }
  const budget = num(v.budget);
  if (budget === null) return null;
  const key = `${String(v.as_of ?? "?")}|${String(v.band ?? "?")}|${v.applied === true ? "applied" : "logged"}`;
  if (key === lastKey) return null;
  const summary = typeof v.summary === "string" && v.summary ? v.summary : `Risk budget ${budget.toFixed(2)}.`;
  const tail = v.applied === true ? "applied to the tier engine's satellite headroom"
    : v.stale === true ? "not applied: stale regime state" : "not applied: VOLTRADE_REGIME_BUDGET off";
  return { key, line: `${summary} [${tail}; as_of ${String(v.as_of ?? "?")}]` };
}

/** The daily regime report: latest state, the recent history, transitions, time at each exposure level. */
export function regimeReportBody(state: RegimeStateFile | null, history: RegimeStateFile[], now: Date, enabled: boolean, sizing: RegimeSizingFile | null = null, flags: Record<string, RegimeFlag> | null = null) {
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
    risk_budget: num(h.risk_budget),
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
  const consumersOn = flags ? regimeConsumersOn(flags) : [];
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
    flags: flags ?? null,     // every VOLTRADE_REGIME_* flag's effective value and source (env | config default)
    // 2026-11-06: the regime risk budget, its plain-English drivers and whether a consumer applies it
    risk_budget: regimeRiskBudget(state, history, now, enabled, flags),
    note: consumersOn.length && enabled
      ? `live: ${consumersOn.join(", ")} act on this state (docs/regime/RUNBOOK.md; rollback = set the env var to off)`
      : "shadow: no consumer reads this state until its own VOLTRADE_REGIME_* flag is set (docs/regime/FINDINGS.md)",
  };
}

/** One audit line's text for a composite transition reported in a scan result (null when there is none). */
export function regimeTransitionAudit(regime: RegimeStateFile | null | undefined): string | null {
  const t = regime?.transition;
  if (!t || !t.to) return null;
  return `${t.from ?? "?"} -> ${t.to} (as_of ${t.to_as_of ?? regime?.as_of ?? "?"}; exposure ${regime?.exposure_multiplier ?? "?"}, core ${regime?.core_multiplier ?? "?"}, confidence ${regime?.confidence ?? "?"})`;
}
