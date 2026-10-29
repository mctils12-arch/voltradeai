/**
 * jobHeartbeat.ts — "no silent failures" (human Phase 3, 2026-10-27).
 *
 * Every scheduled job writes a heartbeat with start, finish and status; an
 * independent watchdog OUTSIDE bot.ts and the daemon (a scheduled Claude
 * routine) polls the read-only `heartbeats` diag probe and alerts when any
 * job is missing, late or errored. This module is the server side of that:
 *
 *   - beat(job, "start") / beat(job, "finish", status, detail) record runs,
 *     persisted to ONE small JSON on the volume (same DATA_DIR convention as
 *     crashSafeRefresh.ts / portDwellCapture.ts), written atomically
 *     (tmp + rename) and bounded (last MAX_RUNS_PER_JOB runs per job,
 *     MAX_JOBS jobs, detail capped).
 *   - JOB_REGISTRY names every EXPECTED job with its cadence, grace and
 *     active window — what "late" means for that job.
 *   - evaluate(now, beats, registry, calendar) is PURE: per job
 *     ok | late | missing | errored | not_due, plus last start/finish/status.
 *
 * Why the registry is not inferred from the beats: a job that never runs
 * never beats, so a watchdog that only reads beats cannot see the one
 * failure that matters most (the 2026-10-27 audit, D2: the 4am daily reset,
 * retrain and backup were unreachable since inception and nothing said so).
 *
 * A recording failure must never break the job it observes: beat() never
 * throws.
 */
import fs from "node:fs";
import path from "node:path";
import { tier2Disabled, tier3Disabled } from "./bisectionFlags";

export type BeatPhase = "start" | "finish";
export type BeatStatus = "ok" | "error" | "skipped";
export type JobVerdict = "ok" | "late" | "missing" | "errored" | "not_due";

export const MAX_RUNS_PER_JOB = 12;
export const MAX_JOBS = 64;
export const MAX_DETAIL_CHARS = 200;
export const HEARTBEAT_FILENAME = "voltrade_job_heartbeats.json";

export interface JobRun {
  startedAt: number;
  finishedAt?: number;
  status?: BeatStatus;
  detail?: string;
  /** an error noted mid-run by a step that swallows its own exception */
  notedError?: string;
}

export interface HeartbeatFile {
  version: 1;
  savedAt?: string;
  jobs: Record<string, JobRun[]>;
}

// ── registry ────────────────────────────────────────────────────────────────

export type JobWindow =
  | { kind: "always" }
  /** NYSE regular session on trading days: 9:30 ET to 16:00 (13:00 half days) */
  | { kind: "market_hours" }
  /** an ET wall-clock range every day; endMin < startMin wraps midnight */
  | { kind: "et_range"; startMin: number; endMin: number };

export interface IntervalJobSpec {
  id: string;
  label: string;
  /** where its beat() calls live (pinned by jobHeartbeat.test.ts) */
  source: string;
  kind: "interval";
  window: JobWindow;
  /** the longest gap the scheduler itself leaves between run starts */
  cadenceMs: number;
  /** tolerance on top of the cadence before the job reads "late" */
  graceMs: number;
  /** a run open longer than this reads "late" (stuck) */
  maxRunMs: number;
  enabledWhen?: (env: Record<string, string | undefined>) => boolean;
  disabledReason?: string;
}

export interface DailyJobSpec {
  id: string;
  label: string;
  source: string;
  kind: "daily";
  tz: "UTC" | "America/New_York";
  /** minutes after local midnight the job becomes due */
  dueMin: number;
  /** due + grace = the deadline; no run since `due` by then = late */
  graceMs: number;
  maxRunMs: number;
  tradingDaysOnly: boolean;
  enabledWhen?: (env: Record<string, string | undefined>) => boolean;
  disabledReason?: string;
}

export type JobSpec = IntervalJobSpec | DailyJobSpec;

const SEC = 1000;
const MIN = 60 * SEC;
const HOUR = 60 * MIN;

/** Boot warm-up: right after a restart a job that has not had its first
 *  tick yet is not late. Capped so a long-cadence job (ml_retrain, 24h)
 *  cannot hide a miss behind frequent deploys. */
export const BOOT_WARMUP_CAP_MS = 30 * MIN;

/**
 * EVERY scheduled job the trading process runs. Cadences are read from the
 * code that schedules them (bot.ts / optionsChainArchive.ts) — change the
 * scheduler, change this table; jobHeartbeat.test.ts pins that every id here
 * has start AND finish beats in its source file.
 */
export const JOB_REGISTRY: readonly JobSpec[] = [
  {
    // bot.ts TIER 1 setInterval(45000). Outside market hours / killed /
    // inactive the tick still beats "skipped", so the timer itself is seen.
    id: "tier1_reflex", label: "Tier 1 reflex (stale orders, position sync, morning queue)",
    source: "server/bot.ts", kind: "interval", window: { kind: "market_hours" },
    cadenceMs: 45 * SEC, graceMs: 3 * MIN, maxRunMs: 5 * MIN,
  },
  {
    // bot.ts scheduleTier2: adaptive 1-15 min inside clock-open || 04:00-20:00
    // ET (every day — the gate does not check the weekday); max interval in
    // that window is 15 min (16-20 ET), failure back-off caps at 10 min.
    // Grace covers a 300s daemon scan + trade execution.
    id: "tier2_scan", label: "Tier 2 scan (adaptive 1-15 min)",
    source: "server/bot.ts", kind: "interval", window: { kind: "et_range", startMin: 4 * 60, endMin: 20 * 60 },
    cadenceMs: 15 * MIN, graceMs: 20 * MIN, maxRunMs: 20 * MIN,
    enabledWhen: (env) => !tier2Disabled(env), disabledReason: "VOLTRADE_DISABLE_TIER2=1 (bisection flag)",
  },
  {
    // bot.ts TIER 3 setInterval(3600000) + the startup run 30s after boot.
    id: "tier3_strategic", label: "Tier 3 strategic (hourly: retrain check, manipulation, diagnostics, COT, Form 4, port dwell)",
    source: "server/bot.ts", kind: "interval", window: { kind: "always" },
    cadenceMs: HOUR, graceMs: 15 * MIN, maxRunMs: 45 * MIN,
    enabledWhen: (env) => !tier3Disabled(env), disabledReason: "VOLTRADE_DISABLE_TIER3=1 (bisection flag)",
  },
  {
    // ml_retrain_safe.py, run from Tier 3 when the model file is >24h old
    // (so roughly every 25h) — a retrain start more than 30h apart means the
    // retrain path stopped running.
    id: "ml_retrain", label: "ML retrain (model older than 24h)",
    source: "server/bot.ts", kind: "interval", window: { kind: "always" },
    cadenceMs: 24 * HOUR, graceMs: 6 * HOUR, maxRunMs: 20 * MIN,
    enabledWhen: (env) => !tier3Disabled(env), disabledReason: "VOLTRADE_DISABLE_TIER3=1 (the retrain runs from Tier 3)",
  },
  {
    // bot.ts setInterval(60000) around syncMonitoredPositions().
    id: "position_sync", label: "Position monitor sync (60s)",
    source: "server/bot.ts", kind: "interval", window: { kind: "always" },
    cadenceMs: 60 * SEC, graceMs: 3 * MIN, maxRunMs: 5 * MIN,
  },
  {
    // bot.ts checkShadowBackfill: 5-min poll, works in UTC hour 22 only,
    // retried each poll until it succeeds -> deadline 23:00 UTC.
    id: "shadow_backfill", label: "Shadow outcome backfill (22:00 UTC)",
    source: "server/bot.ts", kind: "daily", tz: "UTC", dueMin: 22 * 60,
    graceMs: 60 * MIN, maxRunMs: 5 * MIN, tradingDaysOnly: false,
  },
  {
    // bot.ts checkShadowBackfill's second slot (same hour, its own flag).
    id: "lookup_ledger_backfill", label: "Look-up ledger outcome backfill (22:00 UTC)",
    source: "server/bot.ts", kind: "daily", tz: "UTC", dueMin: 22 * 60,
    graceMs: 60 * MIN, maxRunMs: 5 * MIN, tradingDaysOnly: false,
  },
  {
    // bot.ts checkShadowBackfill's third slot (same hour, its own flag): the
    // ticker dossier's calibration ledger scorer (dossier_ranges.score_ledger).
    id: "dossier_ledger_score", label: "Ticker dossier range/plan ledger scorer (22:00 UTC)",
    source: "server/bot.ts", kind: "daily", tz: "UTC", dueMin: 22 * 60,
    graceMs: 60 * MIN, maxRunMs: 5 * MIN, tradingDaysOnly: false,
  },
  {
    // bot.ts checkShadowBackfillExitAligned: 5-min poll, UTC hour 22 (after
    // the old ruler succeeds) or 23, max 3 attempts a night.
    id: "shadow_backfill_exit_aligned", label: "Exit-aligned shadow backfill (22-23 UTC)",
    source: "server/bot.ts", kind: "daily", tz: "UTC", dueMin: 22 * 60,
    graceMs: 105 * MIN, maxRunMs: 5 * MIN, tradingDaysOnly: false,
  },
  {
    // optionsChainArchive.ts bootChainArchive: hourly tick, runs once per
    // trading day after 16:15 ET, retries the same day until it claims it.
    id: "options_chain_archive", label: "Options chain archive (trading days, after 16:15 ET)",
    source: "server/optionsChainArchive.ts", kind: "daily", tz: "America/New_York", dueMin: 16 * 60 + 15,
    graceMs: 135 * MIN, maxRunMs: 30 * MIN, tradingDaysOnly: true,
    enabledWhen: (env) => Boolean(env.ALPACA_KEY && env.ALPACA_SECRET),
    disabledReason: "ALPACA_KEY/ALPACA_SECRET unset (bootChainArchive no-ops)",
  },
  {
    // optionsOiArchive.ts bootOiArchive: same post-close window as the chain
    // archive; Alpaca /v2/options/contracts open interest for SPY/QQQ/IWM.
    id: "options_oi_archive", label: "Options open-interest archive (trading days, after 16:30 ET)",
    source: "server/optionsOiArchive.ts", kind: "daily", tz: "America/New_York", dueMin: 16 * 60 + 30,
    graceMs: 135 * MIN, maxRunMs: 30 * MIN, tradingDaysOnly: true,
    enabledWhen: (env) => Boolean(env.ALPACA_KEY && env.ALPACA_SECRET),
    disabledReason: "ALPACA_KEY/ALPACA_SECRET unset (bootOiArchive no-ops)",
  },
  {
    // bot.ts runOvernightResearch: "once per hour" inside 20:00-04:00 ET.
    id: "overnight_research", label: "Overnight research (hourly, 20:00-04:00 ET)",
    source: "server/bot.ts", kind: "interval", window: { kind: "et_range", startMin: 20 * 60, endMin: 4 * 60 },
    cadenceMs: HOUR, graceMs: 30 * MIN, maxRunMs: 30 * MIN,
  },
  {
    // bot.ts runOvernightResearch's 4am ET branch: daily state reset +
    // daily ML retrain + nightly backup.
    id: "daily_reset", label: "Daily reset + 4am retrain + nightly backup (04:00 ET)",
    source: "server/bot.ts", kind: "daily", tz: "America/New_York", dueMin: 4 * 60,
    graceMs: 2 * HOUR, maxRunMs: 30 * MIN, tradingDaysOnly: false,
  },
  {
    // bot.ts daemon liveness probe: pythonRpc("health") every 5 min.
    id: "daemon_liveness", label: "Python daemon liveness (health RPC, 5 min)",
    source: "server/bot.ts", kind: "interval", window: { kind: "always" },
    cadenceMs: 5 * MIN, graceMs: 5 * MIN, maxRunMs: 2 * MIN,
    enabledWhen: (env) => env.VOLTRADE_DAEMON_ENABLED !== "false",
    disabledReason: "VOLTRADE_DAEMON_ENABLED=false",
  },
  {
    // pricePanel.ts bootPricePanel: 15-min tick, works 20:15-04:00 ET only
    // (dueMin = the window start). A session runs until the build completes
    // or the window closes; afterwards one "skipped" beat per night while the
    // build is fresh (< 30 days). Disk-floor / window pauses beat "skipped",
    // API failures "error"; one attempt per night.
    id: "price_panel_build", label: "Price panel build (2016+ daily bars incl. delisted, 20:15-04:00 ET)",
    source: "server/pricePanel.ts", kind: "daily", tz: "America/New_York", dueMin: 20 * 60 + 15,
    graceMs: 45 * MIN, maxRunMs: 8 * HOUR, tradingDaysOnly: false,
    enabledWhen: (env) => {
      const raw = env.VOLTRADE_PRICE_PANEL;
      const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
      return (v === "1" || v === "true" || v === "on" || v === "yes") && Boolean(env.ALPACA_KEY && env.ALPACA_SECRET);
    },
    disabledReason: "VOLTRADE_PRICE_PANEL off or ALPACA_KEY/ALPACA_SECRET unset",
  },
  {
    // walkforward.ts bootWalkforward: 30-min tick, acts Sat/Sun 06:00-16:00 ET
    // once per ISO week — a run (walkforward.py, up to 3 h) or, with no
    // complete panel / a fresh result, one "skipped" beat. Either way a start
    // lands every ~7 days; the 2-day grace covers a Saturday deploy gap.
    id: "walkforward_weekly", label: "Long-term walk-forward of the stock score (weekly, Sat/Sun 06:00-16:00 ET)",
    source: "server/walkforward.ts", kind: "interval", window: { kind: "always" },
    cadenceMs: 7 * 24 * HOUR, graceMs: 2 * 24 * HOUR, maxRunMs: 4 * HOUR,
    enabledWhen: (env) => {
      const raw = env.VOLTRADE_WALKFORWARD;
      const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
      return v === "1" || v === "true" || v === "on" || v === "yes";
    },
    disabledReason: "VOLTRADE_WALKFORWARD off",
  },
];

// ── calendar ────────────────────────────────────────────────────────────────

/** ET dates (YYYY-MM-DD). Built by the caller from market_calendar.py via
 *  optionsChainArchive's marketHolidays()/marketHalfDays() — the frozen
 *  Python file stays the one holiday source. */
export interface MarketCalendar {
  holidays: ReadonlySet<string>;
  halfDays: ReadonlySet<string>;
}

const PARTS_FMT = new Map<string, Intl.DateTimeFormat>();
function fmt(tz: string): Intl.DateTimeFormat {
  let f = PARTS_FMT.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat("en-US", {
      timeZone: tz, hourCycle: "h23", weekday: "short",
      year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
    PARTS_FMT.set(tz, f);
  }
  return f;
}

interface ZonedParts { date: string; weekday: string; minutes: number; y: number; mo: number; d: number; h: number; mi: number; s: number }

export function zonedParts(ms: number, tz: string): ZonedParts {
  const parts = fmt(tz).formatToParts(ms);
  const get = (t: string) => parts.find((p) => p.type === t)?.value || "";
  const y = +get("year"), mo = +get("month"), d = +get("day");
  const h = +get("hour") % 24, mi = +get("minute"), s = +get("second");
  return {
    date: `${get("year")}-${get("month")}-${get("day")}`,
    weekday: get("weekday"), minutes: h * 60 + mi, y, mo, d, h, mi, s,
  };
}

/** The epoch ms of `minutes` after local midnight of `date` (YYYY-MM-DD) in tz. */
export function zonedInstant(date: string, minutes: number, tz: string): number {
  const [y, mo, d] = date.split("-").map(Number);
  const naive = Date.UTC(y, mo - 1, d, 0, 0) + minutes * MIN;
  let guess = naive;
  for (let i = 0; i < 2; i++) {
    const p = zonedParts(guess, tz);
    const asUtc = Date.UTC(p.y, p.mo - 1, p.d, p.h, p.mi, p.s);
    guess = naive - (asUtc - guess);
  }
  return guess;
}

function shiftDate(date: string, days: number): string {
  const [y, mo, d] = date.split("-").map(Number);
  return new Date(Date.UTC(y, mo - 1, d + days)).toISOString().slice(0, 10);
}

export function isTradingDay(etDate: string, cal: MarketCalendar): boolean {
  const [y, mo, d] = etDate.split("-").map(Number);
  const dow = new Date(Date.UTC(y, mo - 1, d)).getUTCDay();
  return dow !== 0 && dow !== 6 && !cal.holidays.has(etDate);
}

const ET = "America/New_York";
const OPEN_MIN = 9 * 60 + 30;
const CLOSE_MIN = 16 * 60;
const HALF_CLOSE_MIN = 13 * 60;

/** Is `window` active at nowMs, and when did the current active stretch open? */
export function windowState(window: JobWindow, nowMs: number, cal: MarketCalendar): { active: boolean; openedAt: number; why: string } {
  if (window.kind === "always") return { active: true, openedAt: -Infinity, why: "always" };
  const p = zonedParts(nowMs, ET);
  if (window.kind === "market_hours") {
    if (!isTradingDay(p.date, cal)) return { active: false, openedAt: NaN, why: `${p.date} is not a trading day` };
    const close = cal.halfDays.has(p.date) ? HALF_CLOSE_MIN : CLOSE_MIN;
    if (p.minutes < OPEN_MIN || p.minutes >= close) {
      return { active: false, openedAt: NaN, why: `outside market hours (09:30-${close === CLOSE_MIN ? "16:00" : "13:00"} ET)` };
    }
    return { active: true, openedAt: zonedInstant(p.date, OPEN_MIN, ET), why: "market hours" };
  }
  const { startMin, endMin } = window;
  const label = `${hhmm(startMin)}-${hhmm(endMin)} ET`;
  if (startMin <= endMin) {
    const active = p.minutes >= startMin && p.minutes < endMin;
    return { active, openedAt: active ? zonedInstant(p.date, startMin, ET) : NaN, why: active ? label : `outside ${label}` };
  }
  // wraps midnight (e.g. 20:00-04:00)
  if (p.minutes >= startMin) return { active: true, openedAt: zonedInstant(p.date, startMin, ET), why: label };
  if (p.minutes < endMin) return { active: true, openedAt: zonedInstant(shiftDate(p.date, -1), startMin, ET), why: label };
  return { active: false, openedAt: NaN, why: `outside ${label}` };
}

function hhmm(min: number): string {
  return `${String(Math.floor(min / 60)).padStart(2, "0")}:${String(min % 60).padStart(2, "0")}`;
}

// ── evaluate (pure) ─────────────────────────────────────────────────────────

export interface JobEvaluation {
  job: string;
  label: string;
  status: JobVerdict;
  reason: string;
  schedule: string;
  lastStartAt: string | null;
  lastFinishAt: string | null;
  lastStatus: BeatStatus | "running" | null;
  lastDetail: string | null;
  /** seconds since the last run started (null = never) */
  ageSec: number | null;
  running: boolean;
  /** error runs among the stored history (last MAX_RUNS_PER_JOB) */
  recentErrors: number;
}

export interface EvaluateContext {
  calendar: MarketCalendar;
  /** process boot time; jobs get a capped warm-up after a restart */
  bootAtMs?: number;
  env?: Record<string, string | undefined>;
}

function iso(ms: number | undefined): string | null {
  return typeof ms === "number" && Number.isFinite(ms) ? new Date(ms).toISOString() : null;
}

function scheduleText(spec: JobSpec): string {
  if (spec.kind === "daily") {
    return `daily${spec.tradingDaysOnly ? " (trading days)" : ""} due ${hhmm(spec.dueMin)} ${spec.tz === "UTC" ? "UTC" : "ET"}, grace ${Math.round(spec.graceMs / MIN)}m`;
  }
  const w = spec.window.kind === "always" ? "always" : spec.window.kind === "market_hours" ? "market hours" : `${hhmm(spec.window.startMin)}-${hhmm(spec.window.endMin)} ET`;
  return `every ${fmtDur(spec.cadenceMs)} (${w}), grace ${fmtDur(spec.graceMs)}`;
}

function fmtDur(ms: number): string {
  if (ms >= HOUR && ms % HOUR === 0) return `${ms / HOUR}h`;
  if (ms >= MIN && ms % MIN === 0) return `${ms / MIN}m`;
  return `${Math.round(ms / SEC)}s`;
}

export function evaluateJob(spec: JobSpec, runs: readonly JobRun[] | undefined, nowMs: number, ctx: EvaluateContext): JobEvaluation {
  const list = Array.isArray(runs) ? runs : [];
  const last = list.length > 0 ? list[list.length - 1] : undefined;
  const lastFinished = [...list].reverse().find((r) => typeof r.finishedAt === "number");
  const running = !!last && last.finishedAt === undefined;
  const base: JobEvaluation = {
    job: spec.id, label: spec.label, status: "ok", reason: "", schedule: scheduleText(spec),
    lastStartAt: iso(last?.startedAt), lastFinishAt: iso(lastFinished?.finishedAt),
    lastStatus: running ? "running" : (last?.status ?? null),
    lastDetail: (running ? last?.notedError : last?.detail) ?? null,
    ageSec: last ? Math.max(0, Math.round((nowMs - last.startedAt) / SEC)) : null,
    running, recentErrors: list.filter((r) => r.status === "error").length,
  };
  const out = (status: JobVerdict, reason: string): JobEvaluation => ({ ...base, status, reason });

  if (spec.enabledWhen && !spec.enabledWhen(ctx.env ?? {})) {
    return out("not_due", `disabled: ${spec.disabledReason ?? "not enabled in this environment"}`);
  }
  if (running && last && nowMs - last.startedAt > spec.maxRunMs) {
    return out("late", `run started ${fmtDur(nowMs - last.startedAt)} ago has not finished (max ${fmtDur(spec.maxRunMs)})`);
  }

  if (spec.kind === "daily") {
    const p = zonedParts(nowMs, spec.tz);
    if (spec.tradingDaysOnly) {
      const etDate = zonedParts(nowMs, ET).date;
      if (!isTradingDay(etDate, ctx.calendar)) return out("not_due", `${etDate} is not a trading day`);
    }
    const dueAt = zonedInstant(p.date, spec.dueMin, spec.tz);
    if (nowMs < dueAt) return out("not_due", `due ${iso(dueAt)}`);
    const sinceDue = list.filter((r) => r.startedAt >= dueAt);
    const latest = sinceDue[sinceDue.length - 1];
    if (latest) {
      if (latest.finishedAt === undefined) return out("ok", "running");
      if (latest.status === "error") return out("errored", `last run since ${iso(dueAt)} errored: ${latest.detail ?? "no detail"}`);
      return out("ok", `ran ${iso(latest.startedAt)} (${latest.status ?? "ok"})`);
    }
    const deadline = dueAt + spec.graceMs;
    if (nowMs <= deadline) return out("not_due", `due ${iso(dueAt)}, deadline ${iso(deadline)}`);
    return last
      ? out("late", `no run since ${iso(dueAt)} (deadline ${iso(deadline)} passed; last run ${iso(last.startedAt)})`)
      : out("missing", `never ran (deadline ${iso(deadline)} passed)`);
  }

  const w = windowState(spec.window, nowMs, ctx.calendar);
  if (!w.active) return out("not_due", w.why);
  const limit = spec.cadenceMs + spec.graceMs;
  const boot = typeof ctx.bootAtMs === "number" ? ctx.bootAtMs : -Infinity;
  const anchor = Math.max(last ? last.startedAt : boot, w.openedAt);
  if (nowMs - anchor > limit) {
    const warmup = Math.min(limit, BOOT_WARMUP_CAP_MS);
    if (nowMs - boot < warmup) return out("not_due", `warming up after boot (${fmtDur(nowMs - boot)} < ${fmtDur(warmup)})`);
    return last
      ? out("late", `last start ${fmtDur(nowMs - last.startedAt)} ago > ${fmtDur(limit)} (cadence + grace)`)
      : out("missing", `never ran; expected every ${fmtDur(spec.cadenceMs)}`);
  }
  if (lastFinished?.status === "error" && lastFinished === last) {
    return out("errored", `last run errored: ${lastFinished.detail ?? "no detail"}`);
  }
  return out("ok", last ? `last start ${fmtDur(nowMs - last.startedAt)} ago` : "within first cadence of its window");
}

export interface HeartbeatReport {
  evaluatedAt: string;
  verdict: "ok" | "degraded";
  failing: string[];
  counts: Record<JobVerdict, number>;
  jobs: JobEvaluation[];
  /** jobs that beat but are not in the registry (a scheduler nobody registered) */
  unregistered: string[];
}

export function evaluate(
  nowMs: number, beats: HeartbeatFile | null | undefined, registry: readonly JobSpec[], ctx: EvaluateContext,
): HeartbeatReport {
  const jobs = registry.map((spec) => evaluateJob(spec, beats?.jobs?.[spec.id], nowMs, ctx));
  const counts: Record<JobVerdict, number> = { ok: 0, late: 0, missing: 0, errored: 0, not_due: 0 };
  for (const j of jobs) counts[j.status] += 1;
  const failing = jobs.filter((j) => j.status === "late" || j.status === "missing" || j.status === "errored").map((j) => j.job);
  const known = new Set(registry.map((s) => s.id));
  return {
    evaluatedAt: new Date(nowMs).toISOString(),
    verdict: failing.length > 0 ? "degraded" : "ok",
    failing, counts, jobs,
    unregistered: Object.keys(beats?.jobs ?? {}).filter((k) => !known.has(k)),
  };
}

/** The compact, unauthenticated-safe block for /api/health: names and
 *  counts only, no free-text detail (that stays behind the diag token). */
export function healthJobsBlock(r: HeartbeatReport): { status: "ok" | "degraded"; failing: string[]; counts: Record<JobVerdict, number>; gates_top_level_status: false } {
  return { status: r.verdict, failing: r.failing, counts: r.counts, gates_top_level_status: false };
}

// ── pure transitions ────────────────────────────────────────────────────────

function clip(s: unknown): string | undefined {
  if (s === undefined || s === null) return undefined;
  const t = String(s).replace(/\s+/g, " ").trim();
  return t ? t.slice(0, MAX_DETAIL_CHARS) : undefined;
}

/** Apply one beat to a job's run list (returns a NEW list, bounded). */
export function applyBeat(runs: readonly JobRun[] | undefined, phase: BeatPhase, nowMs: number, status?: BeatStatus, detail?: unknown): JobRun[] {
  const list = (Array.isArray(runs) ? runs : []).map((r) => ({ ...r }));
  const open = list.length > 0 && list[list.length - 1].finishedAt === undefined ? list[list.length - 1] : undefined;
  if (phase === "start") {
    if (open) {
      open.finishedAt = nowMs;
      open.status = "error";
      open.detail = clip(open.notedError ?? "no finish beat before the next start");
      delete open.notedError;
    }
    list.push({ startedAt: nowMs });
  } else if (open) {
    const noted = open.notedError;
    open.finishedAt = nowMs;
    // a step that swallowed its exception still makes the run an error,
    // even a gated ("skipped") tick whose protective half failed
    open.status = noted ? "error" : (status ?? "ok");
    const d = clip(status === "error" ? (detail ?? noted) : (noted ?? detail));
    if (d) open.detail = d; else delete open.detail;
    delete open.notedError;
  } else {
    // a finish with no open run: a one-shot beat (e.g. a skipped tick)
    const d = clip(detail);
    list.push({ startedAt: nowMs, finishedAt: nowMs, status: status ?? "ok", ...(d ? { detail: d } : {}) });
  }
  return list.slice(-MAX_RUNS_PER_JOB);
}

/** Note an error inside an open run whose step swallows its own exception;
 *  the run's finish then records "error". No open run -> a one-shot error. */
export function applyNoteError(runs: readonly JobRun[] | undefined, nowMs: number, detail: unknown): JobRun[] {
  const list = (Array.isArray(runs) ? runs : []).map((r) => ({ ...r }));
  const open = list.length > 0 && list[list.length - 1].finishedAt === undefined ? list[list.length - 1] : undefined;
  const d = clip(detail) ?? "error";
  if (open) {
    open.notedError = open.notedError ? clip(`${open.notedError}; ${d}`) : d;
    return list;
  }
  list.push({ startedAt: nowMs, finishedAt: nowMs, status: "error", detail: d });
  return list.slice(-MAX_RUNS_PER_JOB);
}

/** A run left open by a previous process (crash, deploy) is closed as an
 *  error on load: the process never wrote its finish. */
export function closeInterruptedRuns(file: HeartbeatFile): HeartbeatFile {
  const jobs: Record<string, JobRun[]> = {};
  for (const [k, runs] of Object.entries(file.jobs)) {
    jobs[k] = runs.map((r) => (r.finishedAt === undefined
      ? { startedAt: r.startedAt, finishedAt: r.startedAt, status: "error" as const, detail: clip(`interrupted: the process exited before this run finished${r.notedError ? ` (${r.notedError})` : ""}`) }
      : r));
  }
  return { ...file, jobs };
}

export function parseHeartbeatFile(raw: unknown): HeartbeatFile {
  const empty: HeartbeatFile = { version: 1, jobs: {} };
  if (!raw || typeof raw !== "object") return empty;
  const jobsRaw = (raw as { jobs?: unknown }).jobs;
  if (!jobsRaw || typeof jobsRaw !== "object") return empty;
  const jobs: Record<string, JobRun[]> = {};
  for (const [k, v] of Object.entries(jobsRaw as Record<string, unknown>).slice(0, MAX_JOBS)) {
    if (!Array.isArray(v)) continue;
    const runs: JobRun[] = [];
    for (const r of v) {
      if (!r || typeof r !== "object") continue;
      const o = r as Record<string, unknown>;
      if (typeof o.startedAt !== "number" || !Number.isFinite(o.startedAt)) continue;
      const run: JobRun = { startedAt: o.startedAt };
      if (typeof o.finishedAt === "number" && Number.isFinite(o.finishedAt)) run.finishedAt = o.finishedAt;
      if (o.status === "ok" || o.status === "error" || o.status === "skipped") run.status = o.status;
      const d = clip(o.detail); if (d) run.detail = d;
      const n = clip(o.notedError); if (n) run.notedError = n;
      runs.push(run);
    }
    jobs[k] = runs.slice(-MAX_RUNS_PER_JOB);
  }
  return { version: 1, jobs };
}

// ── persistence ─────────────────────────────────────────────────────────────

export function heartbeatDir(env: Record<string, string | undefined> = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}

export interface HeartbeatStore {
  beat(job: string, phase: BeatPhase, status?: BeatStatus, detail?: unknown): void;
  noteError(job: string, detail: unknown): void;
  snapshot(): HeartbeatFile;
  readonly file: string;
  /** last persistence error (null when the last write succeeded) */
  lastWriteError(): string | null;
}

/** Atomic write: a reader (or a crash mid-write) sees the old file or the
 *  new one, never a torn one. */
export function writeHeartbeatFileAtomic(file: string, data: HeartbeatFile): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(tmp, JSON.stringify({ ...data, savedAt: new Date().toISOString() }));
    fs.renameSync(tmp, file);
  } catch (e) {
    // the original file is untouched either way; force: a missing tmp is fine
    try { fs.rmSync(tmp, { force: true }); } catch (cleanupErr) {
      console.error("[jobHeartbeat] could not remove tmp file:", cleanupErr instanceof Error ? cleanupErr.message : cleanupErr);
    }
    throw e;
  }
}

export function readHeartbeatFile(file: string): HeartbeatFile {
  try {
    return parseHeartbeatFile(JSON.parse(fs.readFileSync(file, "utf8")));
  } catch {
    return { version: 1, jobs: {} };
  }
}

export function createHeartbeatStore(file: string, now: () => number = Date.now): HeartbeatStore {
  let state: HeartbeatFile | null = null;
  let writeErr: string | null = null;
  const load = (): HeartbeatFile => {
    if (!state) state = closeInterruptedRuns(readHeartbeatFile(file));
    return state;
  };
  const persist = () => {
    try { writeHeartbeatFileAtomic(file, load()); writeErr = null; } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (writeErr === null) console.error("[jobHeartbeat] could not persist heartbeats:", msg);
      writeErr = msg;
    }
  };
  const update = (job: string, fn: (runs: JobRun[] | undefined) => JobRun[]) => {
    try {
      const s = load();
      if (!(job in s.jobs) && Object.keys(s.jobs).length >= MAX_JOBS) return;
      s.jobs[job] = fn(s.jobs[job]);
      persist();
    } catch (e) {
      console.error("[jobHeartbeat] beat failed:", e instanceof Error ? e.message : e);
    }
  };
  return {
    file,
    beat: (job, phase, status, detail) => update(job, (r) => applyBeat(r, phase, now(), status, detail)),
    noteError: (job, detail) => update(job, (r) => applyNoteError(r, now(), detail)),
    snapshot: () => {
      try { return JSON.parse(JSON.stringify(load())) as HeartbeatFile; } catch { return { version: 1, jobs: {} }; }
    },
    lastWriteError: () => writeErr,
  };
}

let defaultStore: HeartbeatStore | null = null;
export function heartbeatStore(): HeartbeatStore {
  if (!defaultStore) defaultStore = createHeartbeatStore(path.join(heartbeatDir(), HEARTBEAT_FILENAME));
  return defaultStore;
}

/** Record a heartbeat. Never throws. */
export function beat(job: string, phase: BeatPhase, status?: BeatStatus, detail?: unknown): void {
  try { heartbeatStore().beat(job, phase, status, detail); } catch (e) {
    console.error("[jobHeartbeat] beat failed:", e instanceof Error ? e.message : e);
  }
}

/** Mark the job's open run as errored (for steps that swallow their own
 *  exception). Never throws. */
export function noteJobError(job: string, detail: unknown): void {
  try { heartbeatStore().noteError(job, detail); } catch (e) {
    console.error("[jobHeartbeat] noteError failed:", e instanceof Error ? e.message : e);
  }
}

/** start beat, run, finish beat ("ok", or "error" + message and rethrow).
 *  Behavior of `fn` is unchanged: its result and its exception pass through. */
export async function withJobBeat<T>(job: string, fn: () => Promise<T>): Promise<T> {
  beat(job, "start");
  try {
    const r = await fn();
    beat(job, "finish", "ok");
    return r;
  } catch (e) {
    beat(job, "finish", "error", e instanceof Error ? e.message : String(e));
    throw e;
  }
}
