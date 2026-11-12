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
 * by a scheduler path; never started by a boot (first tick one interval later).
 * OUTPUT (DATA_DIR/event_study, KB-sized): latest.json (last success: aggregate
 * tables only — no ticker, price or per-event row; Alpaca licence) and
 * last_run.json. The SEC cache (~15-25 MB of compact public-data rows) lives in
 * the CONTAINER tmp dir, never the volume.
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
 *  2x headroom, because a deploy wipes the container-tmp SEC cache and a killed run would restart from zero. */
export const EVENT_STUDY_TIMEOUT_MS = 5 * 60 * 60_000;
export const EVENT_STUDY_TICK_MS = 30 * 60_000;
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

export function eventStudyCmd(codeVersion: string = pkgVersion, env: Record<string, string | undefined> = process.env): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  const dir = eventStudyCacheDir(env);
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

export type EventStudyDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. */
export function decideEventStudy(nowMs: number, s: {
  enabled: boolean; panelBuildId: string | null; latest: EventStudyLatest | null; lastAttemptWeek: string;
  studyCode: string; freeBytes: number | null; busy: string | null;
}): EventStudyDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_EVENT_STUDY off" };
  if (!inEventStudyWindow(nowMs)) return { action: "idle", reason: "outside Saturday 07:00-15:00 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) return { action: "idle", reason: `already attempted ${week}` };
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

/** Run event_reaction_panel.py once (process-group kill on timeout). Never throws. */
export async function runEventStudy(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(eventStudyCmd(pkgVersion, env), {
      timeout: EVENT_STUDY_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseEventStudySummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(EVENT_STUDY_TIMEOUT_MS / 60_000)}-minute budget — process group killed (the SEC cache in container tmp resumes next week)` };
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
  return {
    probe: "event_study",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `Saturday ${EVENT_STUDY_WINDOW_START_MIN / 60}:00-${EVENT_STUDY_WINDOW_END_MIN / 60}:00 ET, once per ISO week, budget ${EVENT_STUDY_TIMEOUT_MS / 60_000} min, ${EVENT_STUDY_MAX_RSS_MB} MB, SEC ${EVENT_STUDY_SEC_RATE} req/s; waits for ${PANEL_READER_JOBS.join(", ")}`,
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

/** Weekly tick (every EVENT_STUDY_TICK_MS, unref'd; first tick one interval
 *  after boot). One run at a time; the run is not awaited by any other path. */
export function bootEventStudy(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  const tick = async () => {
    if (running) return;
    const root = eventStudyRoot(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const now = Date.now();
    const d = decideEventStudy(now, {
      enabled: eventStudyFlag(env).on, panelBuildId: currentPanelBuild(pricePanelRoot(env)), latest: eventStudyLatest(root),
      lastAttemptWeek, studyCode: studyCodeHash(), freeBytes: free, busy: busyPanelReader(heartbeatStore().snapshot(), now),
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    if (d.action === "skip") {
      beat("event_study_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("event_study_weekly", "start");
    try {
      const out = await runEventStudy(env);
      beat("event_study_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("event_study_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, EVENT_STUDY_TICK_MS).unref?.();
}
