/**
 * dtccTrsPanel.ts — the job that runs dtcc_trs_panel_job.py (repo root: the
 * production image ships root *.py and not datacore/) so the event panel of
 * the registered DTCC equity-swap gate-2 test exists on the server, beside
 * the price panel the test will read (research/open_questions.md (aq1),
 * 2026-11-26; the registration is the 2026-11-25 entry "GATE 2
 * PRE-REGISTERED").
 *
 * WHAT THE PYTHON DOES: reads the vendor's dated files it does not hold,
 * oldest first (the vendor stops serving a file two years after it is
 * written, one a day), one row per US underlier per vendor day, to
 * <DATA_DIR>/dtcc_trs_panel/. When every day is held it writes the event
 * list and compares it with the registered one. It reads no price. Nothing
 * here is read by a trading path, and nothing here is served: the panel is
 * the input of a test that is not built yet.
 *
 * SECURITY TYPES (open_questions.md (as2) step 1, 2026-11-26). Once the event
 * list exists, the same run asks OpenFIGI what each panel CUSIP is
 * (cusip_figi.py: common stock, fund, preferred, ...), which item 3 of the
 * registered filter needs, with the seconds the day reads left of
 * DTCC_TRS_PANEL_TOTAL_S: 10 CUSIPs a request, 25 requests a minute, each
 * answer kept under <root>/figi/. While CUSIPs are owed and the last run got
 * answers, the catch-up lane keeps running. The `dtcc_trs_panel` diag probe
 * (dtccTrsPanelStatus) returns the last run's counts.
 *
 * THE DATED TICKER (open_questions.md (at2), 2026-11-26). Today's ticker
 * cannot carry a 2025 event to its prices when the company was bought or
 * renamed since. The same run reads the SEC fails-to-deliver half-month files
 * it does not hold (cusip_ftd.py: the CUSIP beside the symbol as printed on
 * each settlement day; at most 200 s a run, before the type step) and counts
 * how many events and underliers that record covers: the join's gate 1, no
 * price. While periods are owed and the last run read some, the catch-up
 * lane keeps running. The counts are in the probe's `last_run.ftd`.
 *
 * WHEN. Only between 20:00 and 04:00 ET, every day: the hours in which
 * the Tier-2 scan does not run (its window is 04:00-20:00 ET), so the job
 * never shares a core with a scan that is near its own time limit (KNOWN
 * BROKEN #60). The vendor publishes day D at about 19:16 ET on D.
 *   - CATCH-UP: while the last run left days owed and read at least one,
 *     every tick in the window starts another run (about 730 days at the
 *     first deploy; 10 minutes of reading per 20-minute tick).
 *   - NIGHTLY: otherwise one run per night. A catch-up that stopped making
 *     progress, or ended in error, falls back to this for the rest of the
 *     night, so a vendor outage is not asked every 20 minutes.
 * HOW: the long-job lane's exec (detached process group, nice 19, one
 * process, a memory budget enforced by the Python). The vendor zip (37 to
 * 155 MB) is held in the container's temporary directory, not the volume.
 * FLAG: VOLTRADE_DTCC_TRS_PANEL, default ON.
 */
import fs from "fs";
import path from "path";
import { beat, zonedInstant, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { WALKFORWARD_ENV_CAPS, type RunOutcome, type WalkforwardExec } from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const DTCC_TRS_PANEL_JOB = "dtcc_trs_panel_daily";
export const DTCC_TRS_PANEL_SUBDIR = "dtcc_trs_panel";
export const DTCC_TRS_PANEL_TICK_MS = 20 * 60_000;
/** One run reads for this long, then finishes the day in hand. */
export const DTCC_TRS_PANEL_BUDGET_S = 600;
/** The whole run: day reads, the event pass, then the security-type step
 *  with what is left. Five minutes under the kill below, for the request in
 *  hand and one rate-limit wait. */
export const DTCC_TRS_PANEL_TOTAL_S = 840;
/** Budget + the slowest single day (a 155 MB file on a slow link) + the
 *  compaction and event pass (measured in the PR that added this file). */
export const DTCC_TRS_PANEL_TIMEOUT_MS = 19 * 60_000;
export const DTCC_TRS_PANEL_MAX_RSS_MB = 700;
/** The finished panel is about 50 MB on the volume (gzipped day files plus
 *  the compacted copy); below this free space the job writes nothing. */
export const DTCC_TRS_PANEL_MIN_FREE_BYTES = 300_000_000;
export const DTCC_TRS_PANEL_WINDOW_START_MIN = 20 * 60;
export const DTCC_TRS_PANEL_WINDOW_END_MIN = 4 * 60;
const ET = "America/New_York";

export function dtccTrsPanelFlag(env: Record<string, string | undefined>): boolean {
  const raw = env.VOLTRADE_DTCC_TRS_PANEL;
  const v = (raw === undefined ? "1" : String(raw)).trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

export function dtccTrsPanelRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), DTCC_TRS_PANEL_SUBDIR);
}

export function dtccTrsPanelCmd(codeVersion: string = pkgVersion): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  return `python3 dtcc_trs_panel_job.py --run --nice 19 --max-rss-mb ${DTCC_TRS_PANEL_MAX_RSS_MB} --budget-s ${DTCC_TRS_PANEL_BUDGET_S} --total-s ${DTCC_TRS_PANEL_TOTAL_S} --code-version ${v}`;
}

/** The night a moment belongs to: the ISO instant of its 20:00 ET start, or
 *  null outside 20:00-04:00 ET. */
export function dtccTrsPanelNight(nowMs: number): string | null {
  const p = zonedParts(nowMs, ET);
  if (p.minutes >= DTCC_TRS_PANEL_WINDOW_START_MIN) return new Date(zonedInstant(p.date, DTCC_TRS_PANEL_WINDOW_START_MIN, ET)).toISOString();
  if (p.minutes < DTCC_TRS_PANEL_WINDOW_END_MIN) return new Date(zonedInstant(p.date, DTCC_TRS_PANEL_WINDOW_START_MIN - 24 * 60, ET)).toISOString();
  return null;
}

/** What the scheduler needs of the Python's last_run.json. */
export interface DtccTrsPanelLastRun {
  finishedAt: string | null; backlog: number | null; read: number | null;
  /** The security-type step: CUSIPs not yet asked, and answers the last run got. */
  figiOwed?: number | null; figiAsked?: number | null;
  /** The dated-ticker step: SEC half-month files not yet held, and files the last run read. */
  ftdOwed?: number | null; ftdRead?: number | null;
}

export type DtccTrsPanelDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; mode: "catch-up" | "nightly"; night: string };

/** Pure: what this tick should do. The two `...Night` fields are this
 *  process's memory: the night whose nightly run it already tried, and the
 *  night on which a catch-up run ended in error. */
export function decideDtccTrsPanel(nowMs: number, s: {
  enabled: boolean; last: DtccTrsPanelLastRun | null; lastNightlyAttemptNight: string; freeBytes: number | null;
  catchUpFailedNight?: string;
}): DtccTrsPanelDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_DTCC_TRS_PANEL off" };
  const night = dtccTrsPanelNight(nowMs);
  if (!night) return { action: "idle", reason: "outside 20:00-04:00 ET" };
  const low = s.freeBytes !== null && s.freeBytes < DTCC_TRS_PANEL_MIN_FREE_BYTES
    ? `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${DTCC_TRS_PANEL_MIN_FREE_BYTES / 1e6} MB)` : null;
  const last = s.last;
  // No run yet, or the last one left old days owed and was still reading
  // them, or left CUSIPs unasked and was still getting answers.
  const daysOwed = !!last && (last.backlog ?? 0) > 0 && (last.read ?? 0) > 0;
  const typesOwed = !!last && (last.figiOwed ?? 0) > 0 && (last.figiAsked ?? 0) > 0;
  const periodsOwed = !!last && (last.ftdOwed ?? 0) > 0 && (last.ftdRead ?? 0) > 0;
  const catchUp = !last || !last.finishedAt || daysOwed || typesOwed || periodsOwed;
  if (catchUp && !low && s.catchUpFailedNight !== night) {
    const why = !last?.finishedAt ? "no run yet" : daysOwed ? `${last.backlog} vendor days owed`
      : typesOwed ? `${last.figiOwed} CUSIPs not yet asked` : `${last.ftdOwed} SEC half-month files not yet read`;
    return { action: "run", mode: "catch-up", night, reason: why };
  }
  if (s.lastNightlyAttemptNight === night) return { action: "idle", reason: "already attempted tonight" };
  const finished = last?.finishedAt ? Date.parse(last.finishedAt) : NaN;
  if (Number.isFinite(finished) && finished >= Date.parse(night)) return { action: "idle", reason: "already ran tonight (before a restart)" };
  if (low) return { action: "skip", mode: "nightly", night, reason: low };
  return { action: "run", mode: "nightly", night, reason: last?.finishedAt ? `last finished ${last.finishedAt}` : "no run yet" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[dtcc_trs_panel] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

export function dtccTrsPanelLastRun(root: string): DtccTrsPanelLastRun | null {
  const j = readJson(path.join(root, "last_run.json"));
  if (!j) return null;
  const figi = obj(j.figi);
  const ftd = obj(j.ftd);
  return {
    finishedAt: str(j.finished_at), backlog: num(j.backlog), read: num(j.read), figiOwed: num(figi.owed), figiAsked: num(figi.asked),
    ftdOwed: num(ftd.owed), ftdRead: num(ftd.read),
  };
}

/** The security-type step, for the job row: "; types <held>/<wanted>". */
export function describeFigi(figi: Record<string, unknown>): string {
  if (num(figi.wanted) === null || num(figi.held) === null) return "";
  return `; types ${num(figi.held)}/${num(figi.wanted)}`;
}

/** The dated-ticker step, for the job row: "; ftd <held>/<wanted>" (SEC half-month files). */
export function describeFtd(ftd: Record<string, unknown>): string {
  if (num(ftd.wanted) === null || num(ftd.held) === null) return "";
  return `; ftd ${num(ftd.held)}/${num(ftd.wanted)}`;
}

/** The event list against the registered one, for the job row. */
export function describeEventsCheck(events: Record<string, unknown>): string {
  const reg = obj(events.registered);
  if (num(events.events) === null) return "";
  if (reg.match === true) return `; events ${num(events.events)}, registered list reproduced (${num(reg.registered)} to ${str(reg.through)})`;
  const months = Array.isArray(reg.months_differing) ? reg.months_differing.length : 0;
  return `; events ${num(events.events)}, REGISTERED LIST NOT REPRODUCED: ${num(reg.here)} here vs ${num(reg.registered)} to ${str(reg.through)}, ${months} month(s) differ`;
}

/** Parse the Python's one-line JSON summary (its last stdout line). The row
 *  is cut at 200 characters, so what failed comes first. */
export function parseDtccTrsPanelSummary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if ((st !== "ok" && st !== "error") || num(j.days_ok) === null) {
    return { status: "error", detail: str(j.reason) ?? `status ${String(st)}, no day counts reported` };
  }
  const bad: string[] = [];
  if (num(j.errors)) bad.push(`${num(j.errors)} day(s) unread (${(str(j.last_error) ?? "no reason").slice(0, 70)})`);
  if (str(j.events_error)) bad.push(`event list not written: ${str(j.events_error)!.slice(0, 70)}`);
  const figi = obj(j.figi);
  if (str(figi.status) === "error") bad.push(`types not read (${(str(figi.last_error) ?? str(figi.stopped) ?? "no reason").slice(0, 60)})`);
  const ftd = obj(j.ftd);
  if (str(ftd.status) === "error") bad.push(`SEC files not read (${(str(ftd.last_error) ?? str(ftd.stopped) ?? "no reason").slice(0, 60)})`);
  const head = bad.length ? `${bad.join("; ")}; ` : "";
  const unavailable = num(j.days_unavailable) ? `, ${num(j.days_unavailable)} refused by the vendor (oldest ${str(j.oldest_unavailable)})` : "";
  const owed = num(j.backlog) ? `, ${num(j.backlog)} owed` : "";
  const mb = num(j.bytes) !== null ? `, ${(num(j.bytes)! / 1e6).toFixed(1)} MB` : "";
  return {
    status: st,
    detail: `${head}read ${num(j.read) ?? 0}; holds ${num(j.days_ok)} days ${str(j.first_day) ?? "?"} to ${str(j.last_day) ?? "?"}${owed}${unavailable}`
      + `${describeEventsCheck(obj(j.events))}${describeFigi(figi)}${describeFtd(ftd)} (${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB${mb})`,
  };
}

/** Run the job once (process-group kill on timeout). Never throws. A run in
 *  which a day could not be read exits 1 with a full summary on stdout: that
 *  summary is the outcome. */
export async function runDtccTrsPanel(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(dtccTrsPanelCmd(pkgVersion), {
      timeout: DTCC_TRS_PANEL_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseDtccTrsPanelSummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(DTCC_TRS_PANEL_TIMEOUT_MS / 60_000)}-minute budget — process group killed (days already written are kept)` };
    const fromStdout = err.stdout ? parseDtccTrsPanelSummary(err.stdout) : null;
    if (fromStdout && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Read-only status for the "dtcc_trs_panel" diag probe: the last run's
 *  summary (day counts, the event list against the registered one, the
 *  security-type step's counts by class) and the kept event check. Counts,
 *  dates and OpenFIGI type names only: no CUSIP, ticker, price or return; no
 *  network; no Python. */
export function dtccTrsPanelStatus(env: Record<string, string | undefined> = process.env, root = dtccTrsPanelRoot(env)): Record<string, unknown> {
  return {
    probe: "dtcc_trs_panel",
    job: DTCC_TRS_PANEL_JOB,
    flag: { value: env.VOLTRADE_DTCC_TRS_PANEL ?? null, on: dtccTrsPanelFlag(env) },
    schedule: `20:00-04:00 ET; one run a night, catch-up runs every ${DTCC_TRS_PANEL_TICK_MS / 60_000} min while vendor days, CUSIPs or SEC files are owed; ${DTCC_TRS_PANEL_TOTAL_S}s a run, ${DTCC_TRS_PANEL_MAX_RSS_MB} MB`,
    claim: "none: the event half of a pre-registered test (research/open_questions.md 2026-11-25), the security type of each underlier, and how many events a dated CUSIP-to-ticker record covers. No price is read and no return computed; the test is not run here.",
    dated_join: "last_run.ftd: SEC fails-to-deliver half-month files held, and `coverage`, the share of events and underliers whose CUSIP the SEC printed beside a symbol within 45 days of the event (cusip_ftd.py, defined 2026-11-26 before any file was read). `joined_back` is a pair known on the event day. An underlier is counted at its first event. Counts only.",
    tradable_class: "kept: market sector Equity and security type exactly Common Stock, ADR or REIT (cusip_figi.py, 2026-11-26, before any price was read). A US listing today is not required: a name bought or renamed since the event would be dropped by it. Types are today's, not the event day's.",
    code_version: pkgVersion,
    last_run: readJson(path.join(root, "last_run.json")),
    events_check: readJson(path.join(root, "panel", "events_check.json")),
  };
}

let booted = false;

/** 20-minute tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootDtccTrsPanel(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastNightlyAttemptNight = "";
  let catchUpFailedNight = "";
  const tick = async () => {
    if (running) return;
    const root = dtccTrsPanelRoot(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideDtccTrsPanel(Date.now(), {
      enabled: dtccTrsPanelFlag(env), last: dtccTrsPanelLastRun(root), lastNightlyAttemptNight, freeBytes: free, catchUpFailedNight,
    });
    if (d.action === "idle") return;
    if (d.mode === "nightly") lastNightlyAttemptNight = d.night;
    running = true;
    beat("dtcc_trs_panel_daily", "start");
    try {
      if (d.action === "skip") {
        beat("dtcc_trs_panel_daily", "finish", "skipped", d.reason);
        return;
      }
      const out = await runDtccTrsPanel(env);
      if (d.mode === "catch-up" && out.status === "error") catchUpFailedNight = d.night;
      beat("dtcc_trs_panel_daily", "finish", out.status === "error" ? "error" : "ok", `${d.mode}: ${out.detail}`);
    } catch (e) {
      beat("dtcc_trs_panel_daily", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, DTCC_TRS_PANEL_TICK_MS).unref?.();
}
