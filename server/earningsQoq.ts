/**
 * earningsQoq.ts — the weekly job that runs scripts/earnings_language_qoq_gate2.py
 * (ladder gate 2 of the SEC 8-K earnings-language root: the pre-registered
 * PAIRED-FILING QoQ language-delta test) and the read-only status behind the
 * `earnings_qoq` diag probe. Research plumbing: it schedules and reports;
 * nothing it produces is read by any trading path or surfaced on /data.
 *
 * WHY A JOB AND NOT A SESSION: the public /api/data/earnings-language/history
 * route caps at the newest 500 filings, and in the Q3 season those 500 cover
 * under a week — the gate's input (the full archive since 2026-07-04) lives
 * only on the Railway volume. The registered trigger is not_before 2026-11-15
 * (datacore/signal_ladder.json); the first Sunday run lands exactly then and
 * every later run is informational (the Python carries `first_sufficient_run`
 * forward unchanged — the first sufficient look is the registered verdict).
 *
 * WHEN: Sundays 18:00-19:00 ET start, 75-minute budget — after lazyPrices'
 * 16:00-18:00 window and done before the 20:15 panel-build window, so the
 * Sunday long-job lane stays serialized even though this job never reads the
 * panel (returns come from Alpaca daily bars via backtest_v2.fetch_bars, the
 * same path the level script and live trading use). At most one attempt per
 * ISO week; re-runs for a new code version or a result older than 6 days.
 * HOW: same long-job lane as edgeP2/lazyPrices (detached process group, thread
 * caps, nice 19, peak-RSS budget enforced by the Python). Result files are
 * KB-sized under DATA_DIR/earnings_qoq; derived statistics only — no ticker,
 * CIK, company or price (Alpaca data licence). The bars disk cache stays in
 * the repo dir (container), never the volume.
 * FLAG: VOLTRADE_EARNINGS_QOQ, default ON; present env var wins.
 */
import fs from "fs";
import path from "path";
import { beat, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { WALKFORWARD_ENV_CAPS, isoWeekKey, type RunOutcome, type WalkforwardExec } from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const EARNINGS_QOQ_SUBDIR = "earnings_qoq";
export const EARNINGS_QOQ_TIMEOUT_MS = 75 * 60_000;
export const EARNINGS_QOQ_TICK_MS = 30 * 60_000;
export const EARNINGS_QOQ_MAX_RSS_MB = 900;
/** Weekly re-read of a growing archive (informational after the first sufficient run). */
export const EARNINGS_QOQ_STALE_DAYS = 6;
export const EARNINGS_QOQ_WINDOW_START_MIN = 18 * 60;
export const EARNINGS_QOQ_WINDOW_END_MIN = 19 * 60;
/** Sanity margin only: the job writes KB-sized results; the volume hovers ~1 GB free. */
export const EARNINGS_QOQ_MIN_FREE_BYTES = 64_000_000;
const ET = "America/New_York";
const DAY_MS = 86_400_000;

export function earningsQoqFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_EARNINGS_QOQ;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function earningsQoqRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), EARNINGS_QOQ_SUBDIR);
}

export function earningsQoqCmd(codeVersion: string = pkgVersion): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  return `python3 scripts/earnings_language_qoq_gate2.py --run --nice 19 --max-rss-mb ${EARNINGS_QOQ_MAX_RSS_MB} --code-version ${v}`;
}

/** Sunday, 18:00-19:00 ET. */
export function inEarningsQoqWindow(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.weekday === "Sun" && p.minutes >= EARNINGS_QOQ_WINDOW_START_MIN && p.minutes < EARNINGS_QOQ_WINDOW_END_MIN;
}

export interface EarningsQoqLatest { code_version: string | null; generated_at: string | null }

export type EarningsQoqDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. */
export function decideEarningsQoq(nowMs: number, s: {
  enabled: boolean; latest: EarningsQoqLatest | null; lastAttemptWeek: string; codeVersion: string; freeBytes: number | null;
}): EarningsQoqDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_EARNINGS_QOQ off" };
  if (!inEarningsQoqWindow(nowMs)) return { action: "idle", reason: "outside Sunday 18:00-19:00 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) return { action: "idle", reason: `already attempted ${week}` };
  if (s.freeBytes !== null && s.freeBytes < EARNINGS_QOQ_MIN_FREE_BYTES) {
    return { action: "skip", week, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${EARNINGS_QOQ_MIN_FREE_BYTES / 1e6} MB)` };
  }
  const l = s.latest;
  if (l && l.code_version === s.codeVersion && l.generated_at) {
    const age = nowMs - Date.parse(l.generated_at);
    if (Number.isFinite(age) && age < EARNINGS_QOQ_STALE_DAYS * DAY_MS) {
      return { action: "skip", week, reason: `fresh: code ${l.code_version}, ${Math.floor(age / DAY_MS)}d old` };
    }
  }
  return { action: "run", week, reason: l ? `code ${l.code_version ?? "?"} -> ${s.codeVersion}, last ${l.generated_at ?? "?"}` : "no result yet" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[earnings_qoq] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

export function earningsQoqLatest(root: string): EarningsQoqLatest | null {
  const l = readJson(path.join(root, "latest.json"));
  if (!l) return null;
  return { code_version: str(l.code_version), generated_at: str(l.generated_at) };
}

/** Parse the Python's one-line JSON summary (its last stdout line). */
export function parseEarningsQoqSummary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") return { status: "ok", detail: `verdict ${str(j.verdict)}: ${String(j.pairs)} pairs, n_obs ${JSON.stringify(j.n_obs ?? {})}, ${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB` };
  if (st === "skipped") return { status: "skipped", detail: str(j.reason) ?? st };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

/** Run the gate once (process-group kill on timeout). Never throws. */
export async function runEarningsQoq(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(earningsQoqCmd(pkgVersion), {
      timeout: EARNINGS_QOQ_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseEarningsQoqSummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(EARNINGS_QOQ_TIMEOUT_MS / 60_000)}-minute budget — process group killed` };
    const fromStdout = err.stdout ? parseEarningsQoqSummary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Read-only status for the "earnings_qoq" diag probe: verdict and headline blocks only. */
export function earningsQoqStatus(env: Record<string, string | undefined> = process.env, root = earningsQoqRoot(env)): Record<string, unknown> {
  const l = readJson(path.join(root, "latest.json"));
  const flag = earningsQoqFlag(env);
  return {
    probe: "earnings_qoq",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `Sunday ${EARNINGS_QOQ_WINDOW_START_MIN / 60}:00-${EARNINGS_QOQ_WINDOW_END_MIN / 60}:00 ET start, once per ISO week, budget ${EARNINGS_QOQ_TIMEOUT_MS / 60_000} min, ${EARNINGS_QOQ_MAX_RSS_MB} MB`,
    registered_not_before: "2026-11-15",
    code_version: pkgVersion,
    latest: l ? {
      schema: str(l.schema), generated_at: str(l.generated_at), code_version: str(l.code_version), source: str(l.source),
      archive: obj(l.archive), pairing: obj(l.pairing), unique_tickers: l.unique_tickers ?? null, fetch_error_count: l.fetch_error_count ?? null,
      features: obj(l.features), root_verdict: obj(l.root_verdict), variants: l.variants ?? null,
      first_sufficient_run: l.first_sufficient_run ?? null, verdict_rule: str(l.verdict_rule), caveat: str(l.caveat),
      elapsed_s: l.elapsed_s ?? null, peak_rss_mb: l.peak_rss_mb ?? null,
    } : null,
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

let booted = false;

/** Weekly tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootEarningsQoq(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  const tick = async () => {
    if (running) return;
    const root = earningsQoqRoot(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideEarningsQoq(Date.now(), {
      enabled: earningsQoqFlag(env).on, latest: earningsQoqLatest(root), lastAttemptWeek, codeVersion: pkgVersion, freeBytes: free,
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    if (d.action === "skip") {
      beat("earnings_qoq_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("earnings_qoq_weekly", "start");
    try {
      const out = await runEarningsQoq(env);
      beat("earnings_qoq_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("earnings_qoq_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, EARNINGS_QOQ_TICK_MS).unref?.();
}
