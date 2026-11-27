/**
 * dtccTrsGate2.ts — the weekly job that runs dtcc_trs_gate2.py: the
 * PRE-REGISTERED gate-2 (SIGNAL) test of `dtcc_sbsdr_equity_swaps`
 * (research/open_questions.md 2026-11-25, "DTCC equity swaps: GATE 2
 * PRE-REGISTERED", with the 2026-11-26 addenda; the runner is (as2) step 2,
 * 2026-11-27), and the read-only status behind the `dtcc_trs_gate2` diag
 * probe. Research plumbing: it schedules and reports. Nothing it produces is
 * read by a trading path, and nothing is served to a page before a verdict.
 *
 * WHAT THE PYTHON DOES: joins the event list the nightly panel job keeps
 * (server/dtccTrsPanel.ts) to the price panel by a dated CUSIP-to-symbol
 * record, applies the registered filter to events and to each day's
 * comparison set alike, and computes the registered statistic. It reads no
 * network. Its rules, and the readings taken where the registration is
 * silent, are in the Python's header.
 *
 * ONE LOOK. The registration allows one look, and a second only after an
 * INCONCLUSIVE first and 126 more trading days of vendor files. The Python
 * writes each look to registered.json and never overwrites one; with a
 * verdict of record this job beats "skipped" without starting the Python.
 * A run that ends before a statistic (inputs not ready, under 300 tested
 * events, or "not_armed": the join found too few prices to be trusted)
 * spends no look and is tried again the next week.
 *
 * WHEN: Sundays 18:00-19:15 ET, at most one attempt per ISO week per
 * process. The market is closed, no Tier-2 scan is near its limit, and with
 * the 40-minute kill a run ends before the nightly panel job (20:00 ET)
 * rewrites the event panel and before the price-panel build window (20:15
 * ET) can replace the build being read. While a panel reader, the price
 * build, the Lazy Prices run, the earnings run or the nightly DTCC job has
 * an open heartbeat run, the tick waits WITHOUT spending the week's attempt.
 * HOW: walkforward.ts's long-job lane (detached process group, thread caps,
 * niceness 19, a peak-RSS budget enforced by the Python).
 * OUTPUT (DATA_DIR/dtcc_trs_gate2, KB-sized): latest.json, registered.json,
 * last_run.json. Derived statistics and counts only: no CUSIP, ticker, price
 * or per-event row (the price panel is Alpaca data, internal research only).
 * FLAG: VOLTRADE_DTCC_TRS_GATE2, default ON; present env var wins.
 */
import fs from "fs";
import path from "path";
import { busyPanelReader, PANEL_READER_JOBS } from "./eventStudy";
import { beat, heartbeatStore, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { pricePanelRoot } from "./pricePanel";
import { WALKFORWARD_ENV_CAPS, currentPanelBuild, isoWeekKey, type RunOutcome, type WalkforwardExec } from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const DTCC_TRS_GATE2_JOB = "dtcc_trs_gate2_weekly";
export const DTCC_TRS_GATE2_SUBDIR = "dtcc_trs_gate2";
/** One pass over the price panel (edge_p2's whole run is under 4 minutes on
 *  the same panel) plus the event panel: a 10x margin, and still over before
 *  20:00 ET from the last start the window allows. */
export const DTCC_TRS_GATE2_TIMEOUT_MS = 40 * 60_000;
export const DTCC_TRS_GATE2_TICK_MS = 10 * 60_000;
export const DTCC_TRS_GATE2_MAX_RSS_MB = 900;
export const DTCC_TRS_GATE2_WINDOW_START_MIN = 18 * 60;
export const DTCC_TRS_GATE2_WINDOW_END_MIN = 19 * 60 + 15;
/** The volume only receives KB-sized result files. */
export const DTCC_TRS_GATE2_MIN_FREE_BYTES = 64_000_000;
/** Jobs this one never starts beside: the panel readers and the build, the
 *  two other Sunday-evening Python runs, and the job that writes its events. */
export const DTCC_TRS_GATE2_WAITS_FOR: readonly string[] = [
  ...PANEL_READER_JOBS, "event_study_weekly", "lazy_prices_weekly", "earnings_qoq_weekly", "dtcc_trs_panel_daily",
];
const ET = "America/New_York";

export function dtccTrsGate2Flag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_DTCC_TRS_GATE2;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function dtccTrsGate2Root(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), DTCC_TRS_GATE2_SUBDIR);
}

export function dtccTrsGate2Cmd(codeVersion: string = pkgVersion): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  return `python3 dtcc_trs_gate2.py --run --nice 19 --max-rss-mb ${DTCC_TRS_GATE2_MAX_RSS_MB} --code-version ${v}`;
}

/** Sunday, 18:00-19:15 ET. */
export function inDtccTrsGate2Window(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.weekday === "Sun" && p.minutes >= DTCC_TRS_GATE2_WINDOW_START_MIN && p.minutes < DTCC_TRS_GATE2_WINDOW_END_MIN;
}

export type DtccTrsGate2Decision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. `final` is registered.json's verdict of
 *  record (null while a look is still owed); `busy` an open run this job
 *  waits for. */
export function decideDtccTrsGate2(nowMs: number, s: {
  enabled: boolean; panelBuildId: string | null; final: string | null; lastAttemptWeek: string; freeBytes: number | null; busy: string | null;
}): DtccTrsGate2Decision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_DTCC_TRS_GATE2 off" };
  if (!inDtccTrsGate2Window(nowMs)) return { action: "idle", reason: "outside Sunday 18:00-19:15 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) return { action: "idle", reason: `already attempted ${week}` };
  if (s.final) return { action: "skip", week, reason: `verdict of record: ${s.final}; the registration allows no further look` };
  if (s.busy) return { action: "idle", reason: `waiting for ${s.busy}` };
  if (!s.panelBuildId) return { action: "skip", week, reason: "no complete price-panel build yet (current.json absent)" };
  if (s.freeBytes !== null && s.freeBytes < DTCC_TRS_GATE2_MIN_FREE_BYTES) {
    return { action: "skip", week, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${DTCC_TRS_GATE2_MIN_FREE_BYTES / 1e6} MB)` };
  }
  return { action: "run", week, reason: "a look is owed" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[dtcc_trs_gate2] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

/** registered.json's verdict of record, or null. */
export function dtccTrsGate2Final(root: string): string | null {
  return str(readJson(path.join(root, "registered.json"))?.final);
}

/** Parse the Python's one-line JSON summary (its last stdout line). */
export function parseDtccTrsGate2Summary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") {
    return { status: "ok", detail: `look ${String(j.look)}: ${str(j.verdict)}${str(j.final) ? "" : " (not final)"}, ${String(j.tested)} tested events, statistic ${String(j.mean_pct)}% t ${String(j.t)}, build ${str(j.build_id)} (${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB)` };
  }
  if (st === "skipped") return { status: "skipped", detail: str(j.reason) ?? st };
  // Counts written, no statistic, no look taken: the row says so first.
  if (st === "insufficient" || st === "not_armed") return { status: "insufficient", detail: `${st === "not_armed" ? "NOT ARMED, no look taken: " : ""}${str(j.reason) ?? st}` };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

/** Run the test once (process-group kill on timeout). Never throws. */
export async function runDtccTrsGate2(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(dtccTrsGate2Cmd(pkgVersion), {
      timeout: DTCC_TRS_GATE2_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseDtccTrsGate2Summary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(DTCC_TRS_GATE2_TIMEOUT_MS / 60_000)}-minute budget — process group killed (no look is recorded by an unfinished run)` };
    const fromStdout = err.stdout ? parseDtccTrsGate2Summary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Read-only status for the "dtcc_trs_gate2" diag probe: the looks of record
 *  and the last run's counts and statistics. Derived numbers only: no CUSIP,
 *  ticker, price or per-event row; no network; no Python. */
export function dtccTrsGate2Status(env: Record<string, string | undefined> = process.env, root = dtccTrsGate2Root(env), panelRoot = pricePanelRoot(env)): Record<string, unknown> {
  const flag = dtccTrsGate2Flag(env);
  const reg = readJson(path.join(root, "registered.json"));
  const looks = reg && Array.isArray(reg.looks) ? reg.looks : [];
  const ladder = reg ? str(reg.ladder_verdict) : null;
  return {
    probe: "dtcc_trs_gate2",
    job: DTCC_TRS_GATE2_JOB,
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `Sunday 18:00-19:15 ET, once per ISO week while a look is owed, budget ${DTCC_TRS_GATE2_TIMEOUT_MS / 60_000} min, ${DTCC_TRS_GATE2_MAX_RSS_MB} MB; waits for ${DTCC_TRS_GATE2_WAITS_FOR.join(", ")}`,
    registration: "research/open_questions.md 2026-11-25 'DTCC equity swaps: GATE 2 PRE-REGISTERED' and the 2026-11-26 addenda. One look; a second only after an INCONCLUSIVE first and 126 more trading days of vendor files, then killed.",
    how_to_read: "`registered.looks` are the looks of record and are never overwritten; `registered.ladder_verdict` (the same value as `registered.first_sufficient_run.ladder_verdict_of_record`) is PASS, FAIL or null. `latest` is the last run: with status `not_armed` or `insufficient` it holds counts only and no look was taken. `latest.beside_never_the_verdict` is descriptive (registration item 9).",
    panel_build: currentPanelBuild(panelRoot),
    code_version: pkgVersion,
    registered: {
      looks, final: reg ? str(reg.final) : null, ladder_verdict: ladder,
      // The first run with at least 300 tested events is the registered look.
      // `ladder_verdict_of_record` is what scripts/ladder_verdict_check.py
      // reads: PASS or FAIL once the registration has nothing left to look
      // at (this look, or the one later look an INCONCLUSIVE first allows).
      first_sufficient_run: looks.length ? { ...obj(looks[0]), ladder_verdict_of_record: ladder } : null,
    },
    latest: readJson(path.join(root, "latest.json")),
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

let booted = false;

/** 10-minute tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootDtccTrsGate2(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  const tick = async () => {
    if (running) return;
    const now = Date.now();
    if (!inDtccTrsGate2Window(now)) return;
    const root = dtccTrsGate2Root(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideDtccTrsGate2(now, {
      enabled: dtccTrsGate2Flag(env).on, panelBuildId: currentPanelBuild(pricePanelRoot(env)), final: dtccTrsGate2Final(root),
      lastAttemptWeek, freeBytes: free, busy: busyPanelReader(heartbeatStore().snapshot(), now, DTCC_TRS_GATE2_WAITS_FOR),
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    if (d.action === "skip") {
      beat("dtcc_trs_gate2_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("dtcc_trs_gate2_weekly", "start");
    try {
      const out = await runDtccTrsGate2(env);
      beat("dtcc_trs_gate2_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("dtcc_trs_gate2_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, DTCC_TRS_GATE2_TICK_MS).unref?.();
}
