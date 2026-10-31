/**
 * edgeP2.ts — the weekly job that runs edge_p2_insider.py (ladder gate 2 for
 * pre-registration P2: opportunistic insider purchases over the 2016+ price
 * panel) and the read-only status behind the `edge_p2` diag probe. Research
 * plumbing: it schedules and reports; nothing it produces is read by any
 * trading path.
 *
 * WHEN: Sundays 10:00-16:00 ET — after the panel build window (20:15-04:00 ET)
 * and after the Saturday walk-forward, so two panel readers never overlap and
 * the build directory cannot be swapped underneath the run. At most one attempt
 * per ISO week; re-runs only for a new panel build, a new code version, or a
 * result older than EDGE_P2_STALE_DAYS (new Form 4 quarters land quarterly).
 *
 * HOW: same long-job lane as walkforward.ts (detached process group, explicit
 * budget, thread caps, niceness 19, peak-RSS budget enforced by the Python).
 * OUTPUT (DATA_DIR/edge_p2): latest.json (last success), last_run.json,
 * label_spotcheck.json, form4_cache/ (compact SEC rows, ~tens of MB). Derived
 * statistics only — no ticker or price in latest.json (Alpaca data licence).
 * FLAG: VOLTRADE_EDGE_P2, default ON; present env var wins.
 */
import fs from "fs";
import path from "path";
import { beat, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { pricePanelRoot } from "./pricePanel";
import {
  WALKFORWARD_ENV_CAPS, currentPanelBuild, isoWeekKey, type RunOutcome, type WalkforwardExec,
} from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const EDGE_P2_SUBDIR = "edge_p2";
export const EDGE_P2_TIMEOUT_MS = 3 * 60 * 60_000;
export const EDGE_P2_TICK_MS = 30 * 60_000;
export const EDGE_P2_MAX_RSS_MB = 1500;
export const EDGE_P2_STALE_DAYS = 27;
export const EDGE_P2_WINDOW_START_MIN = 10 * 60;
export const EDGE_P2_WINDOW_END_MIN = 16 * 60;
/** Skip the run when the volume has less than this free: the Form 4 cache must never starve the price-panel build. */
export const EDGE_P2_MIN_FREE_BYTES = 1_500_000_000;
const ET = "America/New_York";
const DAY_MS = 86_400_000;

export function edgeP2Flag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_EDGE_P2;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function edgeP2Root(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), EDGE_P2_SUBDIR);
}

export function edgeP2Cmd(codeVersion: string = pkgVersion): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  return `python3 edge_p2_insider.py --run --nice 19 --max-rss-mb ${EDGE_P2_MAX_RSS_MB} --code-version ${v}`;
}

/** Sunday, 10:00-16:00 ET. */
export function inEdgeP2Window(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.weekday === "Sun" && p.minutes >= EDGE_P2_WINDOW_START_MIN && p.minutes < EDGE_P2_WINDOW_END_MIN;
}

export interface EdgeP2Latest { build_id: string | null; code_version: string | null; generated_at: string | null }

export type EdgeP2Decision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. */
export function decideEdgeP2(nowMs: number, s: {
  enabled: boolean; panelBuildId: string | null; latest: EdgeP2Latest | null; lastAttemptWeek: string; codeVersion: string; freeBytes: number | null;
}): EdgeP2Decision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_EDGE_P2 off" };
  if (!inEdgeP2Window(nowMs)) return { action: "idle", reason: "outside Sunday 10:00-16:00 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) return { action: "idle", reason: `already attempted ${week}` };
  if (!s.panelBuildId) return { action: "skip", week, reason: "no complete price-panel build yet (current.json absent)" };
  if (s.freeBytes !== null && s.freeBytes < EDGE_P2_MIN_FREE_BYTES) {
    return { action: "skip", week, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${EDGE_P2_MIN_FREE_BYTES / 1e6} MB): the Form 4 cache must not starve the panel build` };
  }
  const l = s.latest;
  if (l && l.build_id === s.panelBuildId && l.code_version === s.codeVersion && l.generated_at) {
    const age = nowMs - Date.parse(l.generated_at);
    if (Number.isFinite(age) && age < EDGE_P2_STALE_DAYS * DAY_MS) {
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
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[edge_p2] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

export function edgeP2Latest(root: string): EdgeP2Latest | null {
  const l = readJson(path.join(root, "latest.json"));
  if (!l) return null;
  return { build_id: str(obj(l.panel).build_id), code_version: str(l.code_version), generated_at: str(l.generated_at) };
}

/** Parse edge_p2_insider.py's one-line JSON summary (its last stdout line). */
export function parseEdgeP2Summary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") return { status: "ok", detail: `verdict ${str(j.verdict)} build ${str(j.build_id)}: ${String(j.events)} matched events, ${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB` };
  if (st === "skipped" || st === "insufficient") return { status: st, detail: str(j.reason) ?? st };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

/** Run edge_p2_insider.py once (process-group kill on timeout). Never throws. */
export async function runEdgeP2(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(edgeP2Cmd(), {
      timeout: EDGE_P2_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseEdgeP2Summary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(EDGE_P2_TIMEOUT_MS / 60_000)}-minute budget — process group killed` };
    const fromStdout = err.stdout ? parseEdgeP2Summary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Read-only status for the "edge_p2" diag probe: the verdict and headline cells only. */
export function edgeP2Status(env: Record<string, string | undefined> = process.env, root = edgeP2Root(env), panelRoot = pricePanelRoot(env)): Record<string, unknown> {
  const l = readJson(path.join(root, "latest.json"));
  const flag = edgeP2Flag(env);
  const pit = obj(obj(l?.results).pit);
  const surv = obj(obj(l?.results).survivor_only);
  const cell = (r: Record<string, unknown>, k: string) => {
    const c = obj(r[k]);
    return { months: c.months ?? null, mean_bp: c.mean_bp ?? null, t_nw3: c.t_nw3 ?? null, sd_bp: c.sd_bp ?? null, mde_bp: c.mde_bp ?? null };
  };
  return {
    probe: "edge_p2",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `Sunday ${EDGE_P2_WINDOW_START_MIN / 60}:00-${EDGE_P2_WINDOW_END_MIN / 60}:00 ET, once per ISO week, budget ${EDGE_P2_TIMEOUT_MS / 60_000} min, ${EDGE_P2_MAX_RSS_MB} MB`,
    panel_build: currentPanelBuild(panelRoot),
    code_version: pkgVersion,
    latest: l ? {
      schema: str(l.schema), generated_at: str(l.generated_at), as_of: str(l.as_of), code_version: str(l.code_version), panel: obj(l.panel),
      verdict: str(l.verdict), rule: str(l.rule), why: str(l.why), threshold: obj(l.threshold), window: obj(l.window), deviations: l.deviations ?? null,
      events: obj(l.events), universe: obj(l.universe), form4: obj(l.form4),
      pit: { opportunistic_21d: cell(pit, "opportunistic_21d_matched"), routine_21d: cell(pit, "routine_21d_matched"),
             opportunistic_63d: cell(pit, "opportunistic_63d_matched"), opportunistic_21d_raw: cell(pit, "opportunistic_21d_raw_unmatched") },
      survivor_only: { opportunistic_21d: cell(surv, "opportunistic_21d_matched") },
      elapsed_s: l.elapsed_s ?? null, peak_rss_mb: l.peak_rss_mb ?? null,
    } : null,
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

let booted = false;

/** Weekly tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootEdgeP2(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  const tick = async () => {
    if (running) return;
    const root = edgeP2Root(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideEdgeP2(Date.now(), {
      enabled: edgeP2Flag(env).on, panelBuildId: currentPanelBuild(pricePanelRoot(env)),
      latest: edgeP2Latest(root), lastAttemptWeek, codeVersion: pkgVersion, freeBytes: free,
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    if (d.action === "skip") {
      beat("edge_p2_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("edge_p2_weekly", "start");
    try {
      const out = await runEdgeP2(env);
      beat("edge_p2_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("edge_p2_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, EDGE_P2_TICK_MS).unref?.();
}
