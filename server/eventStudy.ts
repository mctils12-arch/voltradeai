/**
 * eventStudy.ts — the weekly job that runs event_reaction_panel.py (the
 * PRE-REGISTERED event-reaction study, research/event_reaction_study.md §1,
 * re-run on the 2016+ delisting-inclusive price panel) and the read-only status
 * behind the `event_study` diag probe. Research plumbing: it schedules and
 * reports; nothing it produces is read by any trading path.
 *
 * WHEN: Saturdays 07:00-15:00 ET — after the walk-forward's Saturday 06:00 ET
 * slot, before Edge P2's Sunday window, and outside the panel-build window
 * (20:15-04:00 ET): with the 5 h budget a run that starts by 15:00 ET ends by
 * 20:00 ET, so the build directory being read can never be swapped underneath
 * it. While another panel reader (walkforward_weekly, edge_p2_weekly) or the
 * panel build itself has an open heartbeat run, the tick waits WITHOUT
 * spending the week's attempt. At most one attempt per ISO week; it re-runs
 * only for a new panel build, a change to the study code (the STUDY CODE HASH
 * of event_reaction.py + event_reaction_panel.py, not the package version: one
 * full run is ~40-45k SEC requests, so an unrelated deploy must not repeat it), or
 * a result older than EVENT_STUDY_STALE_DAYS.
 *
 * HOW: walkforward.ts's long-job lane (detached process group, explicit
 * budget, thread caps, niceness 19, a peak-RSS budget enforced by the Python),
 * SEC at EVENT_STUDY_SEC_RATE req/s with the declared User-Agent. Never awaited
 * by a scheduler path; never started synchronously by a boot.
 * OUTPUT (DATA_DIR/event_study, KB-sized): latest.json (last success: aggregate
 * tables only — no ticker, price or per-event row; Alpaca licence) and
 * last_run.json.
 * THE SEC CACHE (~15-25 MB estimated, compact public-data rows) AND DEPLOYS
 * (2026-11-22): the cache used to live in the container tmp dir only. A merge to
 * main replaces the container, so a killed run started again from zero, and one
 * run is about 2.5 h of SEC requests. Saturday 2026-11-21 had seven deploys
 * between 13:49Z and 20:57Z, the longest gap 2 h 21 min: no run could finish, the
 * third Saturday without a result. The Python already checkpoints every stage
 * into the cache, so the cache now lives on the volume (DATA_DIR/event_study/
 * sec_cache) whenever eventStudyCachePlan allows it, and a replaced container
 * carries on from the last checkpoint. It falls back to container tmp when the
 * volume has less than EVENT_STUDY_CACHE_MIN_FREE_BYTES free, and a volume cache
 * above EVENT_STUDY_CACHE_MAX_BYTES is deleted. The first tick is
 * EVENT_STUDY_FIRST_TICK_MS after boot (lazyPricesBackfill.ts's pattern), not one
 * 30-min interval later. A restarted container has no memory of the week's
 * attempt, so inside the window it starts again: that is what resumes the run.
 * FLAG: VOLTRADE_EVENT_STUDY, default ON; present env var wins.
 */
import crypto from "crypto";
import fs from "fs";
import os from "os";
import path from "path";
import { JOB_REGISTRY, beat, heartbeatStore, zonedParts, type HeartbeatFile } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { pricePanelRoot } from "./pricePanel";
import { WALKFORWARD_ENV_CAPS, currentPanelBuild, isoWeekKey, type RunOutcome, type WalkforwardExec } from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const EVENT_STUDY_SUBDIR = "event_study";
/** ~40-45k SEC requests at 5/s is ~2.5 h (measured: ~3 s per insider zip, 52 zips), plus ~2 min of measurement:
 *  2x headroom for a run that starts with an empty cache. */
export const EVENT_STUDY_TIMEOUT_MS = 5 * 60 * 60_000;
export const EVENT_STUDY_TICK_MS = 30 * 60_000;
/** The first tick after boot: after boot settles, inside the container lifetimes of a busy merge day. */
export const EVENT_STUDY_FIRST_TICK_MS = 2 * 60_000;
/** The SEC cache's directory under the event-study root when it is kept on the volume. */
export const EVENT_STUDY_CACHE_SUBDIR = "sec_cache";
/** The cache goes on the volume only while this much stays free: the bot's 1 GiB headroom
 *  (globalScopes.MIN_FREE_BYTES; the test pins the sum) plus a margin, as flightPlanVersions'
 *  FPV_MIN_FREE_BYTES does. Below it the cache is container tmp, as before. */
export const EVENT_STUDY_CACHE_MIN_FREE_BYTES = 1_073_741_824 + 256 * 1024 * 1024;
/** A volume cache larger than this is deleted and the run uses container tmp: 8x the upper estimate, so an
 *  estimate that turns out wrong cannot take the volume. */
export const EVENT_STUDY_CACHE_MAX_BYTES = 200 * 1024 * 1024;
/** Peak-RSS budget handed to event_reaction_panel.py (--max-rss-mb). */
export const EVENT_STUDY_MAX_RSS_MB = 1200;
export const EVENT_STUDY_STALE_DAYS = 27;
export const EVENT_STUDY_WINDOW_START_MIN = 7 * 60;
export const EVENT_STUDY_WINDOW_END_MIN = 15 * 60;
/** SEC requests per second (SEC's fair-access ceiling is 10; this leaves room for the live EDGAR feeds). */
export const EVENT_STUDY_SEC_RATE = 5;
export const EVENT_STUDY_WORKERS = 4;
/** The volume only receives KB-sized result files (the SEC cache is in container tmp). */
export const EVENT_STUDY_MIN_FREE_BYTES = 64_000_000;
/** Hashed in this order by studyCodeHash and by event_reaction_panel.study_code_hash. */
export const EVENT_STUDY_FILES: readonly string[] = ["event_reaction.py", "event_reaction_panel.py"];
/** Jobs that read (or write) the same panel build: this job never starts while one has an open run. */
export const PANEL_READER_JOBS: readonly string[] = ["walkforward_weekly", "edge_p2_weekly", "price_panel_build"];
const ET = "America/New_York";
const DAY_MS = 86_400_000;
const SAFE_PATH = /^[0-9A-Za-z_./-]{1,200}$/;

export function eventStudyFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_EVENT_STUDY;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function eventStudyRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), EVENT_STUDY_SUBDIR);
}

/** Container-local (non-volume) SEC cache dir; an env override must be a plain path. */
export function eventStudyCacheDir(env: Record<string, string | undefined> = process.env): string {
  const o = env.VOLTRADE_EVENT_STUDY_CACHE_DIR;
  return o && SAFE_PATH.test(o) ? o : path.join(os.tmpdir(), "voltrade_event_study");
}

export interface EventStudyCachePlan { dir: string; where: "override" | "volume" | "tmp"; reason: string }

/** Pure: where this run's SEC cache lives. `root` is the event-study root, `freeBytes` the free space of
 *  the file system holding it (null = unreadable), `volumeCacheBytes` the size of the volume cache today. */
export function eventStudyCachePlan(env: Record<string, string | undefined>, s: {
  root: string; freeBytes: number | null; volumeCacheBytes: number; tmpDir?: string;
}): EventStudyCachePlan {
  const o = env.VOLTRADE_EVENT_STUDY_CACHE_DIR;
  if (o && SAFE_PATH.test(o)) return { dir: o, where: "override", reason: "VOLTRADE_EVENT_STUDY_CACHE_DIR" };
  const tmpBase = s.tmpDir ?? os.tmpdir();
  const tmp = (reason: string): EventStudyCachePlan => ({ dir: path.join(tmpBase, "voltrade_event_study"), where: "tmp", reason });
  const vol = path.join(s.root, EVENT_STUDY_CACHE_SUBDIR);
  const rel = path.relative(tmpBase, s.root);
  if (!rel.startsWith("..") && !path.isAbsolute(rel)) return tmp("the event-study root is itself in the tmp dir (no volume here)");
  if (!SAFE_PATH.test(vol)) return tmp("the volume path is not a plain path");
  if (s.volumeCacheBytes > EVENT_STUDY_CACHE_MAX_BYTES) {
    return tmp(`the volume cache is ${Math.round(s.volumeCacheBytes / 1e6)} MB, above the ${Math.round(EVENT_STUDY_CACHE_MAX_BYTES / 1e6)} MB cap`);
  }
  if (s.freeBytes === null) return tmp("the volume's free space could not be read");
  if (s.freeBytes < EVENT_STUDY_CACHE_MIN_FREE_BYTES) {
    return tmp(`${Math.floor(s.freeBytes / 1e6)} MB free on the volume, under the ${Math.round(EVENT_STUDY_CACHE_MIN_FREE_BYTES / 1e6)} MB this cache leaves free`);
  }
  return { dir: vol, where: "volume", reason: "kept across a container replacement" };
}

/** Bytes and files under a directory (0, 0 when absent). Sync: a few dozen files. */
export function dirUsage(dir: string): { bytes: number; files: number } {
  let bytes = 0, files = 0;
  let entries: fs.Dirent[];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[event_study] unreadable dir", dir, e instanceof Error ? e.message : e);
    return { bytes, files };
  }
  for (const ent of entries) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) { const u = dirUsage(p); bytes += u.bytes; files += u.files; }
    else if (ent.isFile()) {
      try { bytes += fs.statSync(p).size; files += 1; } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[event_study] unreadable file", p, e instanceof Error ? e.message : e);
      }
    }
  }
  return { bytes, files };
}

function freeBytesAt(root: string): number | null {
  try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); return Number(s.bavail) * Number(s.bsize); } catch { return null; }
}

/** The plan on this machine, read now (statfs + the volume cache's size). */
export function currentEventStudyCachePlan(env: Record<string, string | undefined> = process.env, root: string = eventStudyRoot(env)): EventStudyCachePlan & { volumeCacheBytes: number; volumeCacheFiles: number } {
  const u = dirUsage(path.join(root, EVENT_STUDY_CACHE_SUBDIR));
  return { ...eventStudyCachePlan(env, { root, freeBytes: freeBytesAt(root), volumeCacheBytes: u.bytes }), volumeCacheBytes: u.bytes, volumeCacheFiles: u.files };
}

export function eventStudyCmd(codeVersion: string = pkgVersion, env: Record<string, string | undefined> = process.env, cacheDir?: string): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  const dir = cacheDir ?? eventStudyCacheDir(env);
  const d = SAFE_PATH.test(dir) ? dir : "/tmp/voltrade_event_study";
  return `python3 event_reaction_panel.py --run --nice 19 --max-rss-mb ${EVENT_STUDY_MAX_RSS_MB} --rate ${EVENT_STUDY_SEC_RATE} --workers ${EVENT_STUDY_WORKERS} --code-version ${v} --cache-dir ${d}`;
}

/** sha256 over (name, NUL, bytes, NUL) of EVENT_STUDY_FILES, first 16 hex — the
 *  Python side (study_code_hash) computes the same digest; both suites pin it. */
export function studyCodeHash(dir: string = process.cwd()): string {
  const h = crypto.createHash("sha256");
  for (const name of EVENT_STUDY_FILES) {
    let data: Buffer;
    try { data = fs.readFileSync(path.join(dir, name)); } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[event_study] unreadable", name, e instanceof Error ? e.message : e);
      return `missing:${name}`;
    }
    h.update(Buffer.from(name, "utf8"));
    h.update(Buffer.from([0]));
    h.update(data);
    h.update(Buffer.from([0]));
  }
  return h.digest("hex").slice(0, 16);
}

/** Saturday, 07:00-15:00 ET. */
export function inEventStudyWindow(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.weekday === "Sat" && p.minutes >= EVENT_STUDY_WINDOW_START_MIN && p.minutes < EVENT_STUDY_WINDOW_END_MIN;
}

/** Pure: the first panel reader with an open heartbeat run younger than its
 *  registered maxRunMs (an older open run is a stuck beat, not a reader), or null. */
export function busyPanelReader(snapshot: HeartbeatFile, nowMs: number): string | null {
  for (const id of PANEL_READER_JOBS) {
    const runs = snapshot.jobs[id];
    const last = Array.isArray(runs) && runs.length ? runs[runs.length - 1] : undefined;
    const maxRunMs = JOB_REGISTRY.find((j) => j.id === id)?.maxRunMs ?? 4 * 60 * 60_000;
    if (last && last.finishedAt === undefined && nowMs - last.startedAt < maxRunMs) return id;
  }
  return null;
}

export interface EventStudyLatest { build_id: string | null; study_code: string | null; generated_at: string | null }

/** An "insufficient" run whose reason starts with this lacked SEC downloads (event_reaction_panel.py
 *  FORM345_GAP_REASON). It does not spend the week: the quarters it did fetch stay in the container-tmp cache,
 *  so a later attempt the same Saturday only asks for the missing ones. */
export const EVENT_STUDY_SEC_GAP_REASON = "SEC insider data sets missing for";
/** Attempts per ISO week when each ends on missing SEC downloads, and the wait between them. */
export const EVENT_STUDY_MAX_ATTEMPTS = 3;
export const EVENT_STUDY_RETRY_WAIT_MS = 60 * 60_000;

/** The week's attempts so far that ended on missing SEC downloads, and when the next may start. */
export interface EventStudyRetry { week: string; attempts: number; notBeforeMs: number }

/** Pure: the retry record after a run. null = the week is spent (any other outcome, or the last attempt). */
export function nextEventStudyRetry(prev: EventStudyRetry | null, week: string, out: RunOutcome, nowMs: number): EventStudyRetry | null {
  if (out.status !== "insufficient" || !out.detail.startsWith(EVENT_STUDY_SEC_GAP_REASON)) return null;
  const attempts = (prev && prev.week === week ? prev.attempts : 0) + 1;
  return attempts >= EVENT_STUDY_MAX_ATTEMPTS ? null : { week, attempts, notBeforeMs: nowMs + EVENT_STUDY_RETRY_WAIT_MS };
}

export type EventStudyDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. */
export function decideEventStudy(nowMs: number, s: {
  enabled: boolean; panelBuildId: string | null; latest: EventStudyLatest | null; lastAttemptWeek: string;
  studyCode: string; freeBytes: number | null; busy: string | null; retry?: EventStudyRetry | null;
}): EventStudyDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_EVENT_STUDY off" };
  if (!inEventStudyWindow(nowMs)) return { action: "idle", reason: "outside Saturday 07:00-15:00 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) {
    const r = s.retry && s.retry.week === week ? s.retry : null;
    if (!r) return { action: "idle", reason: `already attempted ${week}` };
    if (nowMs < r.notBeforeMs) return { action: "idle", reason: `attempt ${r.attempts} of ${EVENT_STUDY_MAX_ATTEMPTS} lacked SEC downloads; the next waits until ${new Date(r.notBeforeMs).toISOString()}` };
  }
  if (s.busy) return { action: "idle", reason: `waiting: ${s.busy} is reading the panel` };
  if (!s.panelBuildId) return { action: "skip", week, reason: "no complete price-panel build yet (current.json absent)" };
  if (s.freeBytes !== null && s.freeBytes < EVENT_STUDY_MIN_FREE_BYTES) {
    return { action: "skip", week, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${EVENT_STUDY_MIN_FREE_BYTES / 1e6} MB): the result files need at least that much` };
  }
  const l = s.latest;
  if (l && l.build_id === s.panelBuildId && l.study_code === s.studyCode && l.generated_at) {
    const age = nowMs - Date.parse(l.generated_at);
    if (Number.isFinite(age) && age < EVENT_STUDY_STALE_DAYS * DAY_MS) {
      return { action: "skip", week, reason: `fresh: build ${l.build_id}, study code ${l.study_code}, ${Math.floor(age / DAY_MS)}d old` };
    }
  }
  return { action: "run", week, reason: l ? `build ${l.build_id ?? "?"}/study ${l.study_code ?? "?"} -> ${s.panelBuildId}/${s.studyCode}` : "no result yet" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[event_study] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function eventStudyLatest(root: string): EventStudyLatest | null {
  const l = readJson(path.join(root, "latest.json"));
  if (!l) return null;
  return { build_id: str(obj(l.panel).build_id), study_code: str(l.study_code), generated_at: str(l.generated_at) };
}

/** Parse event_reaction_panel.py's one-line JSON summary (its last stdout line). */
export function parseEventStudySummary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") {
    const v = Object.entries(obj(j.verdicts)).map(([k, x]) => `${k.split("_")[0]} ${String(x)}`).join("; ");
    return { status: "ok", detail: `build ${str(j.build_id)} as_of ${str(j.as_of)}: ${String(j.events)} events, ${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB — ${v}` };
  }
  if (st === "skipped" || st === "insufficient") return { status: st, detail: str(j.reason) ?? st };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

/** Run event_reaction_panel.py once (process-group kill on timeout), its SEC cache where the plan says.
 *  A volume cache above the cap is deleted first. Never throws. */
export async function runEventStudy(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill,
  plan: EventStudyCachePlan & { volumeCacheBytes?: number } = currentEventStudyCachePlan(env)): Promise<RunOutcome> {
  try {
    if (plan.where === "tmp" && (plan.volumeCacheBytes ?? 0) > EVENT_STUDY_CACHE_MAX_BYTES) {
      console.error(`[event_study] ${plan.reason}: deleting it; this run's cache is container tmp`);
      fs.rmSync(path.join(eventStudyRoot(env), EVENT_STUDY_CACHE_SUBDIR), { recursive: true, force: true });
    }
    const { stdout } = await exec(eventStudyCmd(pkgVersion, env, plan.dir), {
      timeout: EVENT_STUDY_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseEventStudySummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(EVENT_STUDY_TIMEOUT_MS / 60_000)}-minute budget — process group killed (the SEC cache keeps what was fetched)` };
    const fromStdout = err.stdout ? parseEventStudySummary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

// ── status (diag probe) ─────────────────────────────────────────────────────
const TYPES = ["E1_insider_cluster", "E2_item_1_01", "E3_item_4_02", "E4_ceo_cfo_exit"] as const;
const GROUPS = ["pooled", "small", "mid", "large"] as const;

function stat(v: unknown): { n: number | null; mean: number | null; t: number | null } {
  const s = obj(v);
  return { n: num(s.n), mean: num(s.mean), t: num(s.t) };
}

/** Headline cells of one event_reaction.summarize output: per type, per group
 *  (pooled/small/mid/large), the pre-registered primary window, its two halves
 *  and the reaction window (aggregates only). Kept shallow: sanitizeDiag cuts
 *  anything deeper than 8 levels, and survivor_subset.cells sits 3 levels down. */
function cells(summary: unknown): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const t of TYPES) {
    const blk = obj(obj(summary)[t]);
    const tbl = obj(blk.table);
    const row: Record<string, unknown> = { n: num(blk.n), t_bar: num(blk.t_bar), verdict: str(blk.verdict), verdict_core: str(blk.verdict_core) };
    for (const g of GROUPS) {
      const r = obj(tbl[g]);
      const p = obj(r["2_20"]);
      const h = obj(r.halves);
      row[g] = {
        car_0_1: stat(r["0_1"]),
        car_2_20: { ...stat(p), median: num(p.median), hit: num(p.hit) },
        car_2_20_2016_2020: stat(h["2016_2020"]), car_2_20_2021_on: stat(h["2021_on"]),
        car_2_60: stat(r["2_60"]),
        spread_median: num(r.spread_median), late_share: num(r.late_share),
        verdict: str(r.verdict), verdict_core: str(r.verdict_core),
      };
    }
    out[t] = row;
  }
  return out;
}

/** Read-only status for the "event_study" diag probe: verdicts and headline cells
 *  of the panel run and of its survivor subset, the run's counts and last_run.json.
 *  No network, no Python, no query parameters, no ticker or price. */
export function eventStudyStatus(env: Record<string, string | undefined> = process.env, root = eventStudyRoot(env),
  panelRoot = pricePanelRoot(env), codeDir: string = process.cwd()): Record<string, unknown> {
  const l = readJson(path.join(root, "latest.json"));
  const flag = eventStudyFlag(env);
  const surv = obj(l?.survivor_subset);
  const plan = currentEventStudyCachePlan(env, root);
  const used = plan.where === "volume" ? { bytes: plan.volumeCacheBytes, files: plan.volumeCacheFiles } : dirUsage(plan.dir);
  return {
    probe: "event_study",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    sec_cache: {
      where: plan.where, reason: plan.reason, survives_a_deploy: plan.where !== "tmp", bytes: used.bytes, files: used.files,
      volume_min_free_mb: Math.round(EVENT_STUDY_CACHE_MIN_FREE_BYTES / 1e6), max_mb: Math.round(EVENT_STUDY_CACHE_MAX_BYTES / 1e6),
    },
    schedule: `Saturday ${EVENT_STUDY_WINDOW_START_MIN / 60}:00-${EVENT_STUDY_WINDOW_END_MIN / 60}:00 ET, first tick ${EVENT_STUDY_FIRST_TICK_MS / 60_000} min after boot, once per ISO week per process (a restarted container starts again inside the window and resumes from the SEC cache), budget ${EVENT_STUDY_TIMEOUT_MS / 60_000} min, ${EVENT_STUDY_MAX_RSS_MB} MB, SEC ${EVENT_STUDY_SEC_RATE} req/s; waits for ${PANEL_READER_JOBS.join(", ")}`,
    panel_build: currentPanelBuild(panelRoot),
    study_code: studyCodeHash(codeDir),
    code_version: pkgVersion,
    latest: l ? {
      schema: str(l.schema), generated_at: str(l.generated_at), as_of: str(l.as_of), code_version: str(l.code_version),
      study_code: str(l.study_code), panel: obj(l.panel), preregistration: str(l.preregistration), bars: obj(l.bars),
      delisting_assumption: obj(l.delisting_assumption), prior: str(l.prior),
      deviations: Array.isArray(l.deviations) ? l.deviations.filter((x) => typeof x === "string") : [],
      verdicts: obj(l.verdicts), cells: cells(l.summary),
      survivor_subset: { verdicts: obj(surv.verdicts), cells: cells(surv.summary) },
      meta: obj(l.meta), elapsed_s: num(l.elapsed_s), peak_rss_mb: num(l.peak_rss_mb),
    } : null,
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

// ── scheduler ────────────────────────────────────────────────────────────────
let booted = false;

/** Weekly tick (unref'd; first tick EVENT_STUDY_FIRST_TICK_MS after boot, then every
 *  EVENT_STUDY_TICK_MS). One run at a time; the run is not awaited by any other path. */
export function bootEventStudy(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  let retry: EventStudyRetry | null = null;
  const tick = async () => {
    if (running) return;
    const root = eventStudyRoot(env);
    const free = freeBytesAt(root);
    const now = Date.now();
    const d = decideEventStudy(now, {
      enabled: eventStudyFlag(env).on, panelBuildId: currentPanelBuild(pricePanelRoot(env)), latest: eventStudyLatest(root),
      lastAttemptWeek, studyCode: studyCodeHash(), freeBytes: free, busy: busyPanelReader(heartbeatStore().snapshot(), now), retry,
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    const before = retry;
    retry = null;
    if (d.action === "skip") {
      beat("event_study_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("event_study_weekly", "start");
    try {
      const out = await runEventStudy(env);
      retry = nextEventStudyRetry(before, d.week, out, Date.now());
      const again = retry ? ` — attempt ${retry.attempts} of ${EVENT_STUDY_MAX_ATTEMPTS}; the next starts after ${new Date(retry.notBeforeMs).toISOString()} if still inside the window` : "";
      beat("event_study_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}${again}`);
    } catch (e) {
      beat("event_study_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  // The first tick is a short timeout, not the first interval: on a day of merges a container may be
  // replaced before its first interval, and each minute it does live is fetch time the cache keeps.
  setTimeout(() => { void tick(); }, EVENT_STUDY_FIRST_TICK_MS).unref?.();
  setInterval(() => { void tick(); }, EVENT_STUDY_TICK_MS).unref?.();
}
