/**
 * lazyPricesBackfill.ts — the NIGHTLY job that runs lazy_prices_backfill.py
 * (accumulating the Lazy Prices gate-2 INPUT: point-in-time 10-K similarity
 * rows on the volume, raw bytes in container tmp) and the read-only status the
 * `lazy_prices` diag probe embeds as `backfill`. Research plumbing: it schedules
 * and reports; nothing it produces is read by any trading path or surfaced on
 * /data.
 *
 * WHY: the weekly gate-2 runner (lazyPrices.ts) read a `filings_10kq` archive
 * that never existed on the production volume and cannot fit there (a 10-K is
 * 5-30 MB; ~50k of them since 2015). The Python module's docstring has the
 * byte math; this file only gives it a nightly slot.
 *
 * WHEN: every night, start window 21:00-23:00 ET, once per ET day. Tier-2 scans
 * stop at 20:00 ET and the 04:00 ET daily branch (resets, retrain, backup) is
 * well after the 3.5 h Python budget (4 h hard kill) ends. The 20:15-04:00 ET
 * price-panel build may overlap: it is one Alpaca request at a time, CPU-light,
 * and currently blocked by its own disk floor; this job is CPU-bound HTML
 * extraction at nice 19 against EDGAR, a different host. Weekend nights
 * included — Sat/Sun daytime belongs to walkforward/edgeP2/lazyPrices.
 * HOW: same long-job lane as edgeP2 (detached process group, thread caps,
 * nice 19, peak-RSS budget enforced by the Python, which checkpoints every
 * chunk so a kill loses at most one chunk of fetches, never a row).
 * FLAG: VOLTRADE_LAZY_PRICES_BACKFILL, default ON; present env var wins.
 */
import fs from "fs";
import os from "os";
import path from "path";
import { beat, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { WALKFORWARD_ENV_CAPS, type RunOutcome, type WalkforwardExec } from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const LAZY_PRICES_BACKFILL_SUBDIR = path.join("lazy_prices", "backfill");
export const LAZY_PRICES_BACKFILL_BUDGET_S = 3.5 * 3600;           // the Python stops and checkpoints here
export const LAZY_PRICES_BACKFILL_TIMEOUT_MS = 4 * 60 * 60_000;    // the hard kill, after the budget
export const LAZY_PRICES_BACKFILL_TICK_MS = 30 * 60_000;
export const LAZY_PRICES_BACKFILL_MAX_RSS_MB = 1500;
export const LAZY_PRICES_BACKFILL_WINDOW_START_MIN = 21 * 60;
export const LAZY_PRICES_BACKFILL_WINDOW_END_MIN = 23 * 60;
/** Mirrors the Python's MIN_FREE_BYTES: the job writes ~50 MB of derived rows/state in total. */
export const LAZY_PRICES_BACKFILL_MIN_FREE_BYTES = 256 * 1024 * 1024;
const ET = "America/New_York";
const SAFE_PATH = /^[0-9A-Za-z_./-]{1,200}$/;

export function lazyPricesBackfillFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_LAZY_PRICES_BACKFILL;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function lazyPricesBackfillRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), LAZY_PRICES_BACKFILL_SUBDIR);
}

/** Container-local (non-volume) SEC pairs cache — shared with lazyPrices.ts; safe override only. */
export function lazyPricesBackfillPairsDir(env: Record<string, string | undefined> = process.env): string {
  const o = env.VOLTRADE_LAZY_PRICES_PAIRS_DIR;
  return o && SAFE_PATH.test(o) ? o : path.join(os.tmpdir(), "voltrade_lazy_prices_pairs");
}

/** Container-local (non-volume) raw/sections scratch — never the volume. */
export function lazyPricesBackfillTmpArchive(env: Record<string, string | undefined> = process.env): string {
  const o = env.VOLTRADE_LAZY_PRICES_TMP_ARCHIVE;
  return o && SAFE_PATH.test(o) ? o : path.join(os.tmpdir(), "voltrade_lazy_prices_tmp_archive");
}

export function lazyPricesBackfillCmd(codeVersion: string = pkgVersion, env: Record<string, string | undefined> = process.env): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  const pairs = lazyPricesBackfillPairsDir(env);
  const tmp = lazyPricesBackfillTmpArchive(env);
  return `python3 lazy_prices_backfill.py --run --nice 19 --budget-seconds ${LAZY_PRICES_BACKFILL_BUDGET_S} `
    + `--max-rss-mb ${LAZY_PRICES_BACKFILL_MAX_RSS_MB} --code-version ${v} --pairs-dir ${pairs} --tmp-archive ${tmp}`;
}

/** Any day, 21:00-23:00 ET. */
export function inLazyPricesBackfillWindow(nowMs: number): boolean {
  const m = zonedParts(nowMs, ET).minutes;
  return m >= LAZY_PRICES_BACKFILL_WINDOW_START_MIN && m < LAZY_PRICES_BACKFILL_WINDOW_END_MIN;
}

export type LazyPricesBackfillDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; day: string };

/** Pure: what this tick should do. No freshness skip — the backfill has work until the plan is exhausted,
 *  and an exhausted plan returns in seconds. */
export function decideLazyPricesBackfill(nowMs: number, s: { enabled: boolean; lastAttemptDay: string; freeBytes: number | null }): LazyPricesBackfillDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_LAZY_PRICES_BACKFILL off" };
  if (!inLazyPricesBackfillWindow(nowMs)) return { action: "idle", reason: "outside 21:00-23:00 ET" };
  const day = zonedParts(nowMs, ET).date;
  if (s.lastAttemptDay === day) return { action: "idle", reason: `already attempted ${day}` };
  if (s.freeBytes !== null && s.freeBytes < LAZY_PRICES_BACKFILL_MIN_FREE_BYTES) {
    return { action: "skip", day, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${Math.floor(LAZY_PRICES_BACKFILL_MIN_FREE_BYTES / 1e6)} MB)` };
  }
  return { action: "run", day, reason: "nightly slice" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[lazy_prices_backfill] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

/** Parse lazy_prices_backfill.py's one-line JSON summary (its last stdout line). */
export function parseLazyPricesBackfillSummary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") {
    if (str(j.phase) === "plan") {
      return { status: "ok", detail: `plan phase: ${String(j.plan_ciks_listed)}/${String(j.plan_ciks_total)} CIKs listed, ${String(j.plan_filings)} filings so far, ${String(j.elapsed_s)}s` };
    }
    const c = obj(j.counts);
    return { status: "ok", detail: `scored ${String(j.processed)} filings (${String(c.rows)} rows total) to ${String(j.cursor_day)}${j.cursor_closed === false ? " (open)" : ""}; `
      + `stop: ${String(j.stop_reason)}; ${String(j.remaining_in_plan)} left in plan of ${String(j.plan_filings)}; ${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB` };
  }
  if (st === "skipped") return { status: "skipped", detail: str(j.reason) ?? st };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

/** Run lazy_prices_backfill.py once (process-group kill on timeout). Never throws. */
export async function runLazyPricesBackfill(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(lazyPricesBackfillCmd(pkgVersion, env), {
      timeout: LAZY_PRICES_BACKFILL_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseLazyPricesBackfillSummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(LAZY_PRICES_BACKFILL_TIMEOUT_MS / 60_000)}-minute hard budget — process group killed (checkpoint on disk is the last chunk)` };
    const fromStdout = err.stdout ? parseLazyPricesBackfillSummary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Read-only status: headline counts only (status.json never carries a CIK, accession or price). */
export function lazyPricesBackfillStatus(env: Record<string, string | undefined> = process.env, root = lazyPricesBackfillRoot(env)): Record<string, unknown> {
  const s = readJson(path.join(root, "status.json"));
  const meta = readJson(path.join(root, "plan_meta.json"));
  const flag = lazyPricesBackfillFlag(env);
  let months = 0;
  try { months = fs.readdirSync(path.join(root, "rows")).filter((n) => n.endsWith(".jsonl.gz")).length; } catch { months = 0; }
  return {
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `nightly ${LAZY_PRICES_BACKFILL_WINDOW_START_MIN / 60}:00-${LAZY_PRICES_BACKFILL_WINDOW_END_MIN / 60}:00 ET start, once per ET day, budget ${LAZY_PRICES_BACKFILL_BUDGET_S / 60} min (kill ${LAZY_PRICES_BACKFILL_TIMEOUT_MS / 60_000}), ${LAZY_PRICES_BACKFILL_MAX_RSS_MB} MB`,
    rows_months: months,
    plan: meta ? { complete: meta.complete ?? null, filings: meta.filings ?? null, ciks: Array.isArray(meta.done_ciks) ? meta.done_ciks.length : null,
      completed_at: str(meta.completed_at), list_errors: meta.list_errors ?? null } : null,
    last_run: s ? {
      status: str(s.status), phase: str(s.phase), finished_at: str(s.finished_at), code_version: str(s.code_version), reason: str(s.reason),
      processed: s.processed ?? null, cursor_day: str(s.cursor_day), cursor_closed: s.cursor_closed ?? null, remaining_in_plan: s.remaining_in_plan ?? null,
      plan_filings: s.plan_filings ?? null, stop_reason: str(s.stop_reason), counts: obj(s.counts), io_noise: obj(s.io_noise), http: obj(s.http),
      elapsed_s: s.elapsed_s ?? null, peak_rss_mb: s.peak_rss_mb ?? null,
    } : null,
  };
}

let booted = false;

/** Nightly tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootLazyPricesBackfill(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptDay = "";
  const tick = async () => {
    if (running) return;
    const root = lazyPricesBackfillRoot(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : os.tmpdir()); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideLazyPricesBackfill(Date.now(), { enabled: lazyPricesBackfillFlag(env).on, lastAttemptDay, freeBytes: free });
    if (d.action === "idle") return;
    lastAttemptDay = d.day;
    if (d.action === "skip") {
      beat("lazy_prices_backfill_nightly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("lazy_prices_backfill_nightly", "start");
    try {
      const out = await runLazyPricesBackfill(env);
      beat("lazy_prices_backfill_nightly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("lazy_prices_backfill_nightly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, LAZY_PRICES_BACKFILL_TICK_MS).unref?.();
}
