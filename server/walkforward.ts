/**
 * walkforward.ts — the weekly job that runs walkforward.py (the long-term,
 * point-in-time, regime-split test of the stock score's price inputs over the
 * 2016+ price panel) and the read-only status behind the `walkforward` diag
 * probe. [RULE-REVIEW] measurement plumbing: it schedules and reports, it
 * never changes what is measured or anything the bot trades.
 *
 * WHEN: Saturdays and Sundays, 06:00-16:00 ET — after the price-panel build
 * window (20:15-04:00 ET) has closed, so the build directory being read can
 * never be swapped out underneath the run (pricePanel.ts deletes the previous
 * build only when a new one completes, inside its own window), and outside
 * regular market hours. At most one attempt per ISO week. It runs when a
 * COMPLETE panel build exists (current.json) and the last result is not
 * already for that build + this code version + younger than
 * WALKFORWARD_STALE_DAYS; otherwise the week's beat is a "skipped" with the
 * reason, so the heartbeat watchdog sees the scheduler alive either way.
 *
 * HOW (server/pythonExec.ts lanes): NOT the global Python mutex (90 s ceiling)
 * and NOT the daemon — its own single-flight long-job lane, like the ML
 * retrain's: a detached process group (execGroupKill: a timeout kills every
 * descendant), an explicit WALKFORWARD_TIMEOUT_MS budget, thread caps of 1,
 * niceness 19 (set by walkforward.py itself), and a peak-RSS budget the Python
 * enforces (it aborts with status "error" and leaves the last good
 * latest.json untouched). Never awaited by a scheduler path; never started by
 * a boot (the first tick is one WALKFORWARD_TICK_MS after boot).
 *
 * FLAG: VOLTRADE_WALKFORWARD, default ON; when the env var is PRESENT it wins
 * ("0"/"off"/"" disable), like VOLTRADE_PRICE_PANEL.
 *
 * OUTPUT (storage_config.WALKFORWARD_DIR = DATA_DIR/walkforward):
 * latest.json (last success; the Brain page's long-term column — contract in
 * research/walkforward.md) and last_run.json (every run). Derived statistics
 * only: no ticker or price leaves either file (Alpaca data licence).
 */
import fs from "fs";
import path from "path";
import type { ExecOptions } from "child_process";
import { beat, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { pricePanelRoot } from "./pricePanel";
import { version as pkgVersion } from "../package.json";

export const WALKFORWARD_SUBDIR = "walkforward";
/** 3 h: the synthetic-panel profile projects ~15-20 min on a full panel; 9x headroom, still ends long before the 20:15 ET panel window. */
export const WALKFORWARD_TIMEOUT_MS = 3 * 60 * 60_000;
export const WALKFORWARD_TICK_MS = 30 * 60_000;
/** Peak-RSS budget handed to walkforward.py (--max-rss-mb). */
export const WALKFORWARD_MAX_RSS_MB = 900;
/** A result older than this is recomputed even when the build and code are unchanged. */
export const WALKFORWARD_STALE_DAYS = 27;
export const WF_WINDOW_START_MIN = 6 * 60;
export const WF_WINDOW_END_MIN = 16 * 60;
const ET = "America/New_York";
const DAY_MS = 86_400_000;
const BUILD_ID_RE = /^b\d{8}$/;

/** Env var PRESENT wins (any value; "0"/"off"/"" disable); absent = default ON. */
export function walkforwardFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_WALKFORWARD;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function walkforwardRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), WALKFORWARD_SUBDIR);
}

/** The command the job runs (cwd = the app root, where walkforward.py lives). */
export function walkforwardCmd(codeVersion: string = pkgVersion): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  return `python3 walkforward.py --run --nice 19 --max-rss-mb ${WALKFORWARD_MAX_RSS_MB} --code-version ${v}`;
}

/** ISO-8601 week key ("2026-W44") of a YYYY-MM-DD date. */
export function isoWeekKey(ymd: string): string {
  const [y, m, d] = ymd.split("-").map(Number);
  const t = new Date(Date.UTC(y, m - 1, d));
  const dow = t.getUTCDay() || 7;
  t.setUTCDate(t.getUTCDate() + 4 - dow);                 // the week's Thursday decides its year
  const yearStart = Date.UTC(t.getUTCFullYear(), 0, 1);
  const week = Math.ceil(((t.getTime() - yearStart) / DAY_MS + 1) / 7);
  return `${t.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

/** Saturday or Sunday, 06:00-16:00 ET. */
export function inWalkforwardWindow(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return (p.weekday === "Sat" || p.weekday === "Sun") && p.minutes >= WF_WINDOW_START_MIN && p.minutes < WF_WINDOW_END_MIN;
}

export interface LatestSummary { build_id: string | null; code_version: string | null; generated_at: string | null }

export type WalkforwardDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. */
export function decideWalkforward(nowMs: number, s: {
  enabled: boolean; panelBuildId: string | null; latest: LatestSummary | null; lastAttemptWeek: string; codeVersion: string;
}): WalkforwardDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_WALKFORWARD off" };
  if (!inWalkforwardWindow(nowMs)) return { action: "idle", reason: "outside Sat/Sun 06:00-16:00 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) return { action: "idle", reason: `already attempted ${week}` };
  if (!s.panelBuildId) return { action: "skip", week, reason: "no complete price-panel build yet (current.json absent)" };
  const l = s.latest;
  if (l && l.build_id === s.panelBuildId && l.code_version === s.codeVersion && l.generated_at) {
    const age = nowMs - Date.parse(l.generated_at);
    if (Number.isFinite(age) && age < WALKFORWARD_STALE_DAYS * DAY_MS) {
      return { action: "skip", week, reason: `fresh: build ${l.build_id}, code ${l.code_version}, ${Math.floor(age / DAY_MS)}d old` };
    }
  }
  return { action: "run", week, reason: l ? `build ${l.build_id ?? "?"}/code ${l.code_version ?? "?"} -> ${s.panelBuildId}/${s.codeVersion}` : "no result yet" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[walkforward] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}

const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

export function currentPanelBuild(panelRoot: string): string | null {
  const id = str(readJson(path.join(panelRoot, "current.json"))?.build_id);
  return id && BUILD_ID_RE.test(id) ? id : null;
}

export function latestSummary(root: string): LatestSummary | null {
  const l = readJson(path.join(root, "latest.json"));
  if (!l) return null;
  return { build_id: str(obj(l.panel).build_id), code_version: str(l.code_version), generated_at: str(l.generated_at) };
}

export interface RunOutcome { status: "ok" | "skipped" | "insufficient" | "error"; detail: string }

/** Parse walkforward.py's one-line JSON summary (its last stdout line). */
export function parseRunSummary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown> | null = null;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") return { status: "ok", detail: `build ${str(j.build_id)} as_of ${str(j.as_of)}: ${String(j.rows)} rows, ${String(j.symbols)} symbols, ${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB` };
  if (st === "skipped" || st === "insufficient") return { status: st, detail: str(j.reason) ?? st };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

export type WalkforwardExec = (cmd: string, opts: ExecOptions & { encoding?: BufferEncoding }) => Promise<{ stdout: string; stderr: string }>;

export const WALKFORWARD_ENV_CAPS: Readonly<Record<string, string>> = Object.freeze({
  OPENBLAS_NUM_THREADS: "1", MKL_NUM_THREADS: "1", OMP_NUM_THREADS: "1", NUMEXPR_MAX_THREADS: "1", VECLIB_MAXIMUM_THREADS: "1",
});

/** Run walkforward.py once (process-group kill on timeout). Never throws. */
export async function runWalkforward(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(walkforwardCmd(), {
      timeout: WALKFORWARD_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseRunSummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(WALKFORWARD_TIMEOUT_MS / 60_000)}-minute budget — process group killed` };
    const fromStdout = err.stdout ? parseRunSummary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

// ── status (diag probe) ─────────────────────────────────────────────────────
/** Read-only status for the "walkforward" diag probe: the two result files'
 *  headline fields only (verdicts and primary-horizon numbers), no network,
 *  no Python, no query parameters. */
export function walkforwardStatus(env: Record<string, string | undefined> = process.env, root = walkforwardRoot(env), panelRoot = pricePanelRoot(env)): Record<string, unknown> {
  const l = readJson(path.join(root, "latest.json"));
  const inputs = obj(l?.inputs);
  const verdicts: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(inputs)) {
    const r = obj(v);
    const h5 = obj(obj(r.horizons)["5"]);
    const h20 = obj(obj(r.horizons)["20"]);
    verdicts[k] = {
      verdict: str(r.verdict), ic_5d: obj(h5.ic).mean ?? null, p_bonferroni_5d: obj(h5.ic).p_bonferroni ?? null,
      ic_20d: obj(h20.ic).mean ?? null, top_n_excess_vs_random_5d_pct: obj(h5.top_n).excess_vs_random_pct ?? null,
      dsr_5d: obj(h5.top_n).dsr ?? null, live_exit_excess_vs_random_pct: obj(r.live_exit).excess_vs_random_pct ?? null,
    };
  }
  const flag = walkforwardFlag(env);
  return {
    probe: "walkforward",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `Sat/Sun ${WF_WINDOW_START_MIN / 60}:00-${WF_WINDOW_END_MIN / 60}:00 ET, once per ISO week, budget ${WALKFORWARD_TIMEOUT_MS / 60_000} min, ${WALKFORWARD_MAX_RSS_MB} MB`,
    panel_build: currentPanelBuild(panelRoot),
    code_version: pkgVersion,
    latest: l ? {
      schema: str(l.schema), status: str(l.status), generated_at: str(l.generated_at), as_of: str(l.as_of), code_version: str(l.code_version),
      panel: obj(l.panel), window: obj(l.window), universe: obj(l.universe), multiple_testing: obj(l.multiple_testing),
      regime: { quality: obj(l.regime).quality ?? null, vxx_missing_share: obj(l.regime).vxx_missing_share ?? null },
      verdicts, not_measurable: Object.keys(obj(l.not_measurable)), elapsed_s: l.elapsed_s ?? null, peak_rss_mb: l.peak_rss_mb ?? null,
    } : null,
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

// ── scheduler ────────────────────────────────────────────────────────────────
let booted = false;

/** Weekly tick (every WALKFORWARD_TICK_MS, unref'd; first tick one interval
 *  after boot). One run at a time; the run is not awaited by any other path. */
export function bootWalkforward(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  const tick = async () => {
    if (running) return;
    const root = walkforwardRoot(env);
    const d = decideWalkforward(Date.now(), {
      enabled: walkforwardFlag(env).on, panelBuildId: currentPanelBuild(pricePanelRoot(env)),
      latest: latestSummary(root), lastAttemptWeek, codeVersion: pkgVersion,
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    if (d.action === "skip") {
      beat("walkforward_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("walkforward_weekly", "start");
    try {
      const out = await runWalkforward(env);
      beat("walkforward_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("walkforward_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, WALKFORWARD_TICK_MS).unref?.();
}
