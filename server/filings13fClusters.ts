/**
 * filings13fClusters.ts — the nightly job that runs filings13f_clusters.py (repo
 * root: the production image ships root *.py, never scripts/) and the read-only
 * status behind the `filings13f_clusters` diag probe.
 *
 * WHAT THE PYTHON DOES: reads the 13F archive on the volume and lists the
 * securities that at least three focused managers newly reported in the same
 * quarter, each dated by the day the third entrant's filing became public
 * (ladder root sec_edgar_13f_institutional_clustering; the definitions are in
 * the Python's header). It reads no price and computes no return. The gate-2
 * test that will read the list is pre-registered in research/open_questions.md
 * (2026-11-22) and is not in this change.
 *
 * WHY A JOB AND NOT A SESSION: the public 13F history route serves the 25
 * largest rows of each table and 500 filings a page; the full tables are only
 * on the Railway volume. And the list must exist, dated, before the forward
 * returns it will be tested against do.
 *
 * WHEN: once per ET day, start inside 05:00-06:00 ET (after the XBRL read's
 * 02:00-05:00 window, before the 13F completeness pass's 06:00-09:00 one: rows
 * that pass recovers are counted the next night). The day's attempt is read
 * from the Python's own last_run.json, so a restarted container does not run
 * it again. HOW: the long-job lane's exec (detached process group, thread caps,
 * nice 19, peak-RSS budget enforced by the Python); the archive is ~40 MB
 * compressed and a run is seconds.
 *
 * Research plumbing: nothing here is read by a trading path, and the list is
 * not on /data (an interpreted reading waits for ladder gate 2).
 * FLAG: VOLTRADE_13F_CLUSTERS, default ON; a present env var wins.
 */
import fs from "fs";
import path from "path";
import { beat, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { WALKFORWARD_ENV_CAPS, type RunOutcome, type WalkforwardExec } from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const FILINGS13F_CLUSTERS_JOB = "filings13f_clusters_nightly";
export const FILINGS13F_CLUSTERS_SUBDIR = "filings13f_clusters";
export const FILINGS13F_CLUSTERS_TIMEOUT_MS = 20 * 60_000;
export const FILINGS13F_CLUSTERS_TICK_MS = 20 * 60_000;
export const FILINGS13F_CLUSTERS_MAX_RSS_MB = 700;
export const FILINGS13F_CLUSTERS_WINDOW_START_MIN = 5 * 60;
export const FILINGS13F_CLUSTERS_WINDOW_END_MIN = 6 * 60;
/** Sanity margin only: the job writes a few hundred KB. */
export const FILINGS13F_CLUSTERS_MIN_FREE_BYTES = 64_000_000;
const ET = "America/New_York";

export function filings13fClustersFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_13F_CLUSTERS;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function filings13fClustersRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), FILINGS13F_CLUSTERS_SUBDIR);
}

export function filings13fClustersCmd(codeVersion: string = pkgVersion): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  return `python3 filings13f_clusters.py --run --nice 19 --max-rss-mb ${FILINGS13F_CLUSTERS_MAX_RSS_MB} --code-version ${v}`;
}

/** Any day, 05:00-06:00 ET. */
export function inFilings13fClustersWindow(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.minutes >= FILINGS13F_CLUSTERS_WINDOW_START_MIN && p.minutes < FILINGS13F_CLUSTERS_WINDOW_END_MIN;
}

export type Filings13fClustersDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; day: string };

/** Pure: what this tick should do. `lastFinishedAt` is last_run.json's
 *  finished_at (any status: one attempt a day, also after an error). */
export function decideFilings13fClusters(nowMs: number, s: {
  enabled: boolean; lastFinishedAt: string | null; lastAttemptDay: string; freeBytes: number | null;
}): Filings13fClustersDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_13F_CLUSTERS off" };
  if (!inFilings13fClustersWindow(nowMs)) return { action: "idle", reason: "outside 05:00-06:00 ET" };
  const day = zonedParts(nowMs, ET).date;
  if (s.lastAttemptDay === day) return { action: "idle", reason: `already attempted ${day}` };
  const last = s.lastFinishedAt ? Date.parse(s.lastFinishedAt) : NaN;
  if (Number.isFinite(last) && zonedParts(last, ET).date === day) return { action: "idle", reason: `already ran ${day} (before a restart)` };
  if (s.freeBytes !== null && s.freeBytes < FILINGS13F_CLUSTERS_MIN_FREE_BYTES) {
    return { action: "skip", day, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${FILINGS13F_CLUSTERS_MIN_FREE_BYTES / 1e6} MB)` };
  }
  return { action: "run", day, reason: s.lastFinishedAt ? `last finished ${s.lastFinishedAt}` : "no run yet" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[filings13f_clusters] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

export function filings13fClustersLastFinishedAt(root: string): string | null {
  return str(readJson(path.join(root, "last_run.json"))?.finished_at);
}

/** Parse the Python's one-line JSON summary (its last stdout line). */
export function parseFilings13fClustersSummary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") {
    return {
      status: "ok",
      detail: `${String(j.full_filings)} filings with tables, ${String(j.pairs)} quarter pair(s), ${String(j.clusters)} clusters (${String(j.thin_clusters)} thinly held), ${String(j.first_seen_appended)} new to the record, ${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB`,
    };
  }
  if (st === "skipped") return { status: "skipped", detail: str(j.reason) ?? st };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

/** Run the detector once (process-group kill on timeout). Never throws. */
export async function runFilings13fClusters(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(filings13fClustersCmd(pkgVersion), {
      timeout: FILINGS13F_CLUSTERS_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseFilings13fClustersSummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(FILINGS13F_CLUSTERS_TIMEOUT_MS / 60_000)}-minute budget — process group killed` };
    const fromStdout = err.stdout ? parseFilings13fClustersSummary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Read-only status for the "filings13f_clusters" diag probe. Public SEC
 *  filings data only (issuer names and CUSIPs as filed); no price, no return. */
export function filings13fClustersStatus(env: Record<string, string | undefined> = process.env, root = filings13fClustersRoot(env)): Record<string, unknown> {
  const l = readJson(path.join(root, "latest.json"));
  const flag = filings13fClustersFlag(env);
  return {
    probe: "filings13f_clusters",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `once per ET day, start ${FILINGS13F_CLUSTERS_WINDOW_START_MIN / 60}:00-${FILINGS13F_CLUSTERS_WINDOW_END_MIN / 60}:00 ET, budget ${FILINGS13F_CLUSTERS_TIMEOUT_MS / 60_000} min, ${FILINGS13F_CLUSTERS_MAX_RSS_MB} MB`,
    claim: "none: a count of as-filed 13F records. No price is read and no return computed; the gate-2 test is pre-registered (research/open_questions.md 2026-11-22) and not yet run.",
    code_version: pkgVersion,
    latest: l ? {
      schema: str(l.schema), generated_at: str(l.generated_at), code_version: str(l.code_version),
      archive: obj(l.archive), periods_full_filings: obj(l.periods_full_filings), pairs: l.pairs ?? null,
      newest_pair_top_thin: l.newest_pair_top_thin ?? null, first_seen_appended: l.first_seen_appended ?? null,
      definitions: obj(l.definitions), elapsed_s: l.elapsed_s ?? null, peak_rss_mb: l.peak_rss_mb ?? null,
    } : null,
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

let booted = false;

/** Nightly tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootFilings13fClusters(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptDay = "";
  const tick = async () => {
    if (running) return;
    const root = filings13fClustersRoot(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideFilings13fClusters(Date.now(), {
      enabled: filings13fClustersFlag(env).on, lastFinishedAt: filings13fClustersLastFinishedAt(root), lastAttemptDay, freeBytes: free,
    });
    if (d.action === "idle") return;
    lastAttemptDay = d.day;
    running = true;
    beat("filings13f_clusters_nightly", "start");
    try {
      if (d.action === "skip") {
        beat("filings13f_clusters_nightly", "finish", "skipped", d.reason);
        return;
      }
      const out = await runFilings13fClusters(env);
      beat("filings13f_clusters_nightly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("filings13f_clusters_nightly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, FILINGS13F_CLUSTERS_TICK_MS).unref?.();
}
