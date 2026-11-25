/**
 * snapshotRefresh.ts — the nightly job that runs snapshot_refresh_job.py (repo
 * root: the production image ships root *.py, never scripts/) and tells the
 * readers when a newer copy has been written.
 *
 * WHAT IT REPLACES (research/open_questions.md (ah4), 2026-11-25): four
 * whole-file snapshots — EIA weekly stocks, NOAA CPC degree days, STB EP724
 * rail, JODI oil stocks — were refreshed only when a session ran
 * scripts/snapshot_refresh.py and committed the files. Their 21-day staleness
 * trigger fired on 10-12 (at 99 days), 11-02 and 11-24. The Python now runs
 * here every night and writes to <DATA_DIR>/snapshot_refresh/; server/
 * lazyAssets.ts serves the newer of that copy and the bundled one.
 *
 * WHAT THE PYTHON DOES: per source, skips a copy built under 20 h ago (JODI:
 * 6 days), otherwise runs the same refresher a session ran, and accepts the
 * result only if it is newer, holds at least 90% as many series and does not
 * end earlier than the served copy (its header has the rules). One source's
 * outage does not stop the others. The run's status is "error" when any
 * source failed or was refused, so the Pipeline health job row and
 * /api/health's jobs block say which.
 *
 * WHEN: once per ET day, start inside 01:00-03:00 ET (after EIA's Wednesday
 * and Thursday releases have been up for hours; before the 04:00 ET retrain
 * and the first scan). The day's attempt is read from the Python's own
 * last_run.json, so a restarted container does not fetch again. HOW: the
 * long-job lane's exec (detached process group, nice 19). Measured from a
 * session machine 2026-11-25: all four sources, 18 s, 58 MB peak.
 *
 * NOT KNOWN AT BUILD TIME: whether Railway reaches all four hosts (eia.gov,
 * ftp.cpc.ncep.noaa.gov, stb.gov, jodidata.org). The first night's job row
 * answers it per source; an unreachable source keeps its bundled copy.
 *
 * Raw public data, unchanged in content; nothing here is read by a trading
 * path. FLAG: VOLTRADE_SNAPSHOT_REFRESH, default ON; off stops the job AND
 * makes the readers ignore the volume copies (lazyAssets.ts).
 */
import fs from "fs";
import path from "path";
import { beat, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { WALKFORWARD_ENV_CAPS, type RunOutcome, type WalkforwardExec } from "./walkforward";
import { REFRESHABLE_TABLES, snapshotRefreshEnabled, snapshotRefreshRoot } from "./lazyAssets";
import { resetSnapshotStreams } from "./snapshotStreams";
import { version as pkgVersion } from "../package.json";

export const SNAPSHOT_REFRESH_JOB = "snapshot_refresh_nightly";
export const SNAPSHOT_REFRESH_TIMEOUT_MS = 20 * 60_000;
export const SNAPSHOT_REFRESH_TICK_MS = 20 * 60_000;
export const SNAPSHOT_REFRESH_MAX_RSS_MB = 600;
export const SNAPSHOT_REFRESH_WINDOW_START_MIN = 1 * 60;
export const SNAPSHOT_REFRESH_WINDOW_END_MIN = 3 * 60;
/** The four files total about 3.5 MB and each is staged beside the copy it
 *  replaces; below this the job writes nothing. */
export const SNAPSHOT_REFRESH_MIN_FREE_BYTES = 64_000_000;
/** The sources, in the Python's order (snapshot_refresh_job.py SOURCES). */
export const SNAPSHOT_REFRESH_SOURCES = ["eia", "cpc", "rail", "jodi"] as const;
const ET = "America/New_York";

export function snapshotRefreshCmd(codeVersion: string = pkgVersion): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  return `python3 snapshot_refresh_job.py --run --nice 19 --max-rss-mb ${SNAPSHOT_REFRESH_MAX_RSS_MB} --code-version ${v}`;
}

/** Any day, 01:00-03:00 ET. */
export function inSnapshotRefreshWindow(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.minutes >= SNAPSHOT_REFRESH_WINDOW_START_MIN && p.minutes < SNAPSHOT_REFRESH_WINDOW_END_MIN;
}

export type SnapshotRefreshDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; day: string };

/** Pure: what this tick should do. `lastFinishedAt` is last_run.json's
 *  finished_at (any status: one attempt a day, also after an error). */
export function decideSnapshotRefresh(nowMs: number, s: {
  enabled: boolean; lastFinishedAt: string | null; lastAttemptDay: string; freeBytes: number | null;
}): SnapshotRefreshDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_SNAPSHOT_REFRESH off" };
  if (!inSnapshotRefreshWindow(nowMs)) return { action: "idle", reason: "outside 01:00-03:00 ET" };
  const day = zonedParts(nowMs, ET).date;
  if (s.lastAttemptDay === day) return { action: "idle", reason: `already attempted ${day}` };
  const last = s.lastFinishedAt ? Date.parse(s.lastFinishedAt) : NaN;
  if (Number.isFinite(last) && zonedParts(last, ET).date === day) return { action: "idle", reason: `already ran ${day} (before a restart)` };
  if (s.freeBytes !== null && s.freeBytes < SNAPSHOT_REFRESH_MIN_FREE_BYTES) {
    return { action: "skip", day, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${SNAPSHOT_REFRESH_MIN_FREE_BYTES / 1e6} MB)` };
  }
  return { action: "run", day, reason: s.lastFinishedAt ? `last finished ${s.lastFinishedAt}` : "no run yet" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[snapshot_refresh] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

export function snapshotRefreshLastFinishedAt(root: string): string | null {
  return str(readJson(path.join(root, "last_run.json"))?.finished_at);
}

export interface SnapshotRefreshOutcome extends RunOutcome {
  /** how many sources wrote a newer copy this run */
  refreshed: number;
}

/** One source for the job row, short: the row's detail is cut at 200
 *  characters and a failure's reason matters more than a success's date. */
export function describeSource(key: string, r: Record<string, unknown>): string {
  const st = str(r.status) ?? "?";
  if (st === "refreshed") return `${key} refreshed to ${str(r.latest) ?? "?"}`;
  if (st === "fresh") return `${key} fresh`;
  return `${key} ${st}: ${(str(r.reason) ?? "no reason given").slice(0, 60)}`;
}

/** Parse the Python's one-line JSON summary (its last stdout line). */
export function parseSnapshotRefreshSummary(stdout: string): SnapshotRefreshOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", refreshed: 0, detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  const sources = obj(j.sources);
  const keys = Object.keys(sources);
  if ((st !== "ok" && st !== "error") || keys.length === 0) {
    return { status: "error", refreshed: 0, detail: str(j.reason) ?? `status ${String(st)}, no sources reported` };
  }
  // Failures first: the row is cut at 200 characters.
  const bad = keys.filter((k) => { const s = str(obj(sources[k]).status); return s === "error" || s === "rejected"; });
  const good = keys.filter((k) => !bad.includes(k));
  const detail = [...bad, ...good].map((k) => describeSource(k, obj(sources[k]))).join("; ");
  const refreshed = typeof j.refreshed === "number" && Number.isFinite(j.refreshed) ? j.refreshed : 0;
  return { status: st, refreshed, detail: `${detail} (${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB)` };
}

/** Run the refresh once (process-group kill on timeout). Never throws. A run
 *  in which some source failed exits 1 with a full summary on stdout: that
 *  summary is the outcome, including how many OTHER sources were refreshed. */
export async function runSnapshotRefresh(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<SnapshotRefreshOutcome> {
  try {
    const { stdout } = await exec(snapshotRefreshCmd(pkgVersion), {
      timeout: SNAPSHOT_REFRESH_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseSnapshotRefreshSummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", refreshed: 0, detail: `exceeded the ${Math.round(SNAPSHOT_REFRESH_TIMEOUT_MS / 60_000)}-minute budget — process group killed` };
    const fromStdout = err.stdout ? parseSnapshotRefreshSummary(err.stdout) : null;
    if (fromStdout && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", refreshed: 0, detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Make the readers see what the job wrote: the four tables are re-read on
 *  their next use and the Streams inventory re-stamps them. */
export function adoptRefreshedSnapshots(): void {
  for (const t of REFRESHABLE_TABLES) t.reset();
  resetSnapshotStreams();
}

let booted = false;

/** Nightly tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootSnapshotRefresh(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptDay = "";
  const tick = async () => {
    if (running) return;
    const root = snapshotRefreshRoot(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideSnapshotRefresh(Date.now(), {
      enabled: snapshotRefreshEnabled(env), lastFinishedAt: snapshotRefreshLastFinishedAt(root), lastAttemptDay, freeBytes: free,
    });
    if (d.action === "idle") return;
    lastAttemptDay = d.day;
    running = true;
    beat("snapshot_refresh_nightly", "start");
    try {
      if (d.action === "skip") {
        beat("snapshot_refresh_nightly", "finish", "skipped", d.reason);
        return;
      }
      const out = await runSnapshotRefresh(env);
      if (out.refreshed > 0) adoptRefreshedSnapshots();
      beat("snapshot_refresh_nightly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", out.detail);
    } catch (e) {
      beat("snapshot_refresh_nightly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, SNAPSHOT_REFRESH_TICK_MS).unref?.();
}
