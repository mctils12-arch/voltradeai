/**
 * Fire-detection archive: the daily completeness pass and the refill of
 * missing days (KNOWN BROKEN #56 follow-up, 2026-11-20).
 *
 * WHY. The live poll (nasaFirms.ts) asks NASA FIRMS for "1 day", which NASA
 * defines as the current UTC date only. Two losses follow from that:
 *   1. a detection acquired late in a UTC day and published after midnight is
 *      never asked for (NASA states about 3 hours of near-real-time latency);
 *   2. a day on which the poll archived nothing (2026-11-03 to 2026-11-20,
 *      when the one satellite it asked for returned no rows) stays empty.
 * NASA's area request also takes a date
 * (`/api/area/csv/[KEY]/[SOURCE]/[AREA]/[DAY_RANGE]/[DATE]`, read on
 * firms.modaps.eosdis.nasa.gov/api/area 2026-11-20). This pass uses it.
 *
 * WHAT IT DOES, once per UTC day at or after 06:00 UTC:
 *   - YESTERDAY: asks NASA for yesterday by date and appends the rows the
 *     day file does not hold. Only a plain `.jsonl` is appended to; a day
 *     file that is already gzipped is never rewritten.
 *   - MISSING DAYS: for each of the last FIRES_REFILL_LOOKBACK_DAYS days with
 *     NO day file at all, asks NASA for that date and writes the day file
 *     (gzipped), oldest day first: the oldest is the first NASA stops serving.
 * It never touches a day that has a gzipped file, so it cannot disagree with
 * a copy the permanent off-site tier already holds.
 *
 * WHAT IT DOES NOT KNOW: how far back NASA's near-real-time sources answer a
 * dated request. A day NASA no longer serves comes back with no rows from
 * every source; after FIRES_DAY_MAX_ATTEMPTS it is left as `no_rows` and the
 * ledger says so. A refilled day is NASA's answer on the day it was asked,
 * not what the poll would have recorded live: the ledger keeps `refilled_at`.
 *
 * The ledger (`voltrade_fires_completeness.json` beside the heartbeats) is
 * the measurement: per day, how many rows NASA returned, how many the archive
 * already held and how many were added. `added / upstream_rows` on yesterday
 * is the share the live poll missed.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import type { Express } from "express";
import { archiveBaseDir } from "./datacoreArchive";
import { beat, heartbeatDir } from "./jobHeartbeat";
import { permanentLocalDays } from "./archivePermanentTier";
import { promisify } from "util";
import { FIRMS_SOURCES, fetchFirmsDetections, firmsKey, scrubFirmsKey, type FireDetection } from "./nasaFirms";

const gzipAsync = promisify(zlib.gzip);

export const FIRES_COMPLETENESS_JOB = "fires_completeness_daily";
/** UTC hour from which yesterday is asked for: NASA's ~3 h latency, doubled. */
export const FIRES_COMPLETENESS_START_HOUR_UTC = 6;
/** Days back that a missing day file is refilled. Must stay under the fires
 *  stream's local window on the permanent tier: older than that, a day with
 *  no local file may simply have been moved off-site. */
export const FIRES_REFILL_LOOKBACK_DAYS = 25;
export const FIRES_DAY_MAX_ATTEMPTS = 3;
/** one run asks for at most this many days (each is one ~6 MB answer) */
export const FIRES_MAX_DAYS_PER_RUN = 20;
export const FIRES_COMPLETENESS_DELAY_MS = 1500;
/** a dated world answer is a larger job for NASA than the live one */
export const FIRES_COMPLETENESS_TIMEOUT_MS = 60_000;
export const FIRES_COMPLETENESS_FAIL_STOP = 3;
export const FIRES_COMPLETENESS_STATE_FILENAME = "voltrade_fires_completeness.json";

const DAY_MS = 86400_000;
/** NASA's `satellite` column -> the source that serves it */
const SOURCE_OF_SATELLITE: Record<string, string> = {
  N: "VIIRS_SNPP_NRT", N20: "VIIRS_NOAA20_NRT", N21: "VIIRS_NOAA21_NRT",
};

export type FiresFetch = (url: string, init?: unknown) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export type FiresDayStatus =
  /** yesterday checked: every row NASA returned is now held */
  | "complete"
  /** a day with no file was written from NASA's dated answer */
  | "refilled"
  /** every source answered the date with no rows */
  | "no_rows"
  /** every request failed */
  | "failed";

export interface FiresLedgerDay {
  status: FiresDayStatus;
  source: string | null;
  upstream_rows: number;
  held_before: number;
  added: number;
  attempts: number;
  error: string | null;
  at: string;
  refilled_at?: string;
}

export interface FiresLedger {
  version: 1;
  last_run_day: string | null;
  last_run_at: string | null;
  last_run_detail: string | null;
  days: Record<string, FiresLedgerDay>;
}

export function firesCompletenessEnabled(env: Record<string, string | undefined>): boolean {
  const raw = env.VOLTRADE_FIRES_COMPLETENESS;
  const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
  return (v === "1" || v === "true" || v === "on" || v === "yes") && Boolean(firmsKey(env as NodeJS.ProcessEnv));
}

const dayOf = (ms: number): string => new Date(ms).toISOString().slice(0, 10);

function ledgerPath(env: Record<string, string | undefined> = process.env): string {
  return path.join(heartbeatDir(env), FIRES_COMPLETENESS_STATE_FILENAME);
}

export function readFiresLedger(file = ledgerPath()): FiresLedger {
  const empty: FiresLedger = { version: 1, last_run_day: null, last_run_at: null, last_run_detail: null, days: {} };
  if (!fs.existsSync(file)) return empty;
  try {
    const j = JSON.parse(fs.readFileSync(file, "utf8")) as FiresLedger;
    return j && typeof j === "object" && j.days && typeof j.days === "object" ? { ...empty, ...j } : empty;
  } catch (e) {
    console.error("[datacore] fires completeness ledger unreadable:", e instanceof Error ? e.message : e);
    return empty;
  }
}

export function writeFiresLedger(ledger: FiresLedger, file = ledgerPath()): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(ledger));
  fs.renameSync(tmp, file);
}

/** What the archive holds for one day: which file, and its rows' ids. */
export function heldFireDay(dir: string, day: string): { file: "plain" | "gz" | "none"; ids: Set<string>; satellite: string | null; unreadable_lines: number } {
  const plain = path.join(dir, `${day}.jsonl`);
  const gz = path.join(dir, `${day}.jsonl.gz`);
  const ids = new Set<string>();
  let satellite: string | null = null;
  let unreadable = 0;
  const take = (text: string) => {
    for (const line of text.split("\n")) {
      if (!line) continue;
      try {
        const r = JSON.parse(line) as { id?: string; satellite?: string };
        if (r.id) ids.add(r.id);
        if (!satellite && r.satellite) satellite = r.satellite;
      } catch {
        unreadable += 1; // a torn line (a write cut short): not a held row
      }
    }
  };
  const hasPlain = fs.existsSync(plain), hasGz = fs.existsSync(gz);
  if (hasGz) take(zlib.gunzipSync(fs.readFileSync(gz)).toString("utf8"));
  if (hasPlain) take(fs.readFileSync(plain, "utf8"));
  return { file: hasGz ? "gz" : hasPlain ? "plain" : "none", ids, satellite, unreadable_lines: unreadable };
}

/** The days one run works on: yesterday first (when its file is still plain
 *  or absent), then days with no file at all, oldest first. A day the ledger
 *  has settled, or has tried FIRES_DAY_MAX_ATTEMPTS times, is left alone. */
export function firesCandidateDays(dir: string, ledger: FiresLedger, nowMs: number, lookback = FIRES_REFILL_LOOKBACK_DAYS): string[] {
  const today = dayOf(nowMs);
  const settled = (day: string): boolean => {
    const l = ledger.days[day];
    if (!l) return false;
    if (l.status === "complete" || l.status === "refilled") return true;
    return l.attempts >= FIRES_DAY_MAX_ATTEMPTS;
  };
  const exists = (day: string, ext: string) => fs.existsSync(path.join(dir, `${day}${ext}`));
  const out: string[] = [];
  const yesterday = dayOf(nowMs - DAY_MS);
  if (!settled(yesterday) && !exists(yesterday, ".jsonl.gz")) out.push(yesterday);
  for (let back = lookback; back >= 2; back--) {
    const day = dayOf(nowMs - back * DAY_MS);
    if (day >= today || settled(day)) continue;
    if (exists(day, ".jsonl") || exists(day, ".jsonl.gz")) continue;
    out.push(day);
  }
  return out;
}

export interface FiresCompletenessDeps {
  mapKey: string;
  fetch: FiresFetch;
  dir?: string;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
  ledger: FiresLedger;
  maxDays?: number;
  /** called after each day, so a pass cut short by a restart keeps the days it finished */
  persist?: (ledger: FiresLedger) => void;
}

export interface FiresCompletenessRun {
  at: string;
  days_asked: number;
  rows_added: number;
  refilled_days: number;
  failed_days: number;
  stopped_early: boolean;
  detail: string;
  status: "ok" | "error";
}

const realSleep = (ms: number) => new Promise<void>((res) => { setTimeout(res, ms); });

/** NASA's dated answer for one day: the source the day's held rows came from,
 *  or, for a day with none, the live poll's order, first source with rows. */
async function askDay(deps: FiresCompletenessDeps, day: string, heldSatellite: string | null): Promise<{ rows: FireDetection[]; source: string | null; error: string | null }> {
  const own = heldSatellite ? SOURCE_OF_SATELLITE[heldSatellite] : undefined;
  const sources = own ? [own] : [...FIRMS_SOURCES];
  const errors: string[] = [];
  let answered = false;
  for (const source of sources) {
    try {
      const rows = (await fetchFirmsDetections(deps.mapKey, deps.fetch as never, source, day, FIRES_COMPLETENESS_TIMEOUT_MS))
        // NASA's answer is for the date asked; a row of another date is not this day's
        .filter((r) => r.acq_date === day);
      answered = true;
      if (rows.length > 0) return { rows, source, error: null };
    } catch (e) {
      errors.push(`${source}: ${scrubFirmsKey(e instanceof Error ? e.message : String(e), deps.mapKey)}`);
    }
  }
  return { rows: [], source: null, error: answered ? null : errors.join("; ") };
}

/** One pass. Mutates and returns through `deps.ledger`; the caller persists it. */
export async function runFiresCompleteness(deps: FiresCompletenessDeps): Promise<FiresCompletenessRun> {
  const now = deps.now ?? Date.now;
  const sleep = deps.sleep ?? realSleep;
  const dir = deps.dir ?? path.join(archiveBaseDir(), "fires");
  const ledger = deps.ledger;
  const days = firesCandidateDays(dir, ledger, now()).slice(0, deps.maxDays ?? FIRES_MAX_DAYS_PER_RUN);
  const run: FiresCompletenessRun = { at: new Date(now()).toISOString(), days_asked: 0, rows_added: 0, refilled_days: 0, failed_days: 0, stopped_early: false, detail: "", status: "ok" };
  let failsInRow = 0;
  for (const day of days) {
    if (run.days_asked > 0) await sleep(FIRES_COMPLETENESS_DELAY_MS);
    const held = heldFireDay(dir, day);
    if (held.file === "gz") continue; // gzipped since the list was made: never rewritten
    const prev = ledger.days[day];
    const rec: FiresLedgerDay = {
      status: "failed", source: null, upstream_rows: 0, held_before: held.ids.size, added: 0,
      attempts: (prev?.attempts ?? 0) + 1, error: null, at: new Date(now()).toISOString(),
    };
    run.days_asked++;
    const got = await askDay(deps, day, held.satellite);
    rec.source = got.source;
    rec.upstream_rows = got.rows.length;
    if (got.error) {
      rec.error = got.error.slice(0, 300);
      run.failed_days++;
      failsInRow++;
    } else if (got.rows.length === 0) {
      // every source answered and none has the date: NASA no longer serves it,
      // or (yesterday) nothing was seen. Nothing is written.
      rec.status = held.ids.size > 0 ? "complete" : "no_rows";
      failsInRow = 0;
    } else {
      failsInRow = 0;
      const seen = new Set(held.ids);
      const fresh: FireDetection[] = [];
      for (const r of got.rows) {
        if (!r.id || seen.has(r.id)) continue;
        seen.add(r.id);
        fresh.push(r);
      }
      try {
        if (fresh.length > 0) {
          fs.mkdirSync(dir, { recursive: true });
          const text = fresh.map((r) => JSON.stringify(r)).join("\n") + "\n";
          if (held.file === "plain") {
            fs.appendFileSync(path.join(dir, `${day}.jsonl`), text);
          } else {
            const fp = path.join(dir, `${day}.jsonl.gz`);
            const tmp = `${fp}.${process.pid}.tmp`;
            // a day is about 15 MB of text: compressed off the event loop
            fs.writeFileSync(tmp, await gzipAsync(text));
            fs.renameSync(tmp, fp);
          }
        }
        rec.added = fresh.length;
        run.rows_added += fresh.length;
        if (held.file === "none") {
          rec.status = "refilled";
          rec.refilled_at = rec.at;
          run.refilled_days++;
        } else {
          rec.status = "complete";
        }
      } catch (e) {
        rec.error = `write failed: ${e instanceof Error ? e.message : String(e)}`.slice(0, 300);
        run.failed_days++;
      }
    }
    ledger.days[day] = rec;
    try { deps.persist?.(ledger); } catch (e) {
      console.error("[datacore] fires completeness: ledger not written:", e instanceof Error ? e.message : e);
    }
    if (failsInRow >= FIRES_COMPLETENESS_FAIL_STOP) { run.stopped_early = true; break; }
  }
  // keep the ledger bounded: two lookbacks of history
  const floor = dayOf(now() - 2 * FIRES_REFILL_LOOKBACK_DAYS * DAY_MS);
  for (const d of Object.keys(ledger.days)) if (d < floor) delete ledger.days[d];
  run.detail = `${run.days_asked} day(s) asked, ${run.rows_added} row(s) added, ${run.refilled_days} day(s) refilled, ${run.failed_days} failed${run.stopped_early ? ", stopped after repeated failures" : ""}`;
  run.status = run.failed_days > 0 ? "error" : "ok";
  ledger.last_run_day = dayOf(now());
  ledger.last_run_at = run.at;
  ledger.last_run_detail = run.detail;
  return run;
}

/** Counts per day, newest first: what the status route serves. No key, no URL. */
export function firesCompletenessStatus(env: Record<string, string | undefined> = process.env): Record<string, unknown> {
  const ledger = readFiresLedger(ledgerPath(env));
  const days = Object.keys(ledger.days).sort().reverse().map((day) => ({ day, ...ledger.days[day] }));
  const yesterdays = days.filter((d) => d.status === "complete" && d.upstream_rows > 0);
  const up = yesterdays.reduce((a, d) => a + d.upstream_rows, 0);
  const added = yesterdays.reduce((a, d) => a + d.added, 0);
  return {
    kind: "raw",
    enabled: firesCompletenessEnabled(env),
    job: FIRES_COMPLETENESS_JOB,
    last_run_at: ledger.last_run_at,
    last_run_detail: ledger.last_run_detail,
    refilled_days: days.filter((d) => d.status === "refilled").length,
    no_rows_days: days.filter((d) => d.status === "no_rows").length,
    failed_days: days.filter((d) => d.status === "failed").length,
    /** over the checked days: rows NASA returned by date that the live poll had not archived */
    live_poll_missed: { days: yesterdays.length, upstream_rows: up, added, share: up > 0 ? Math.round((added / up) * 10000) / 10000 : null },
    days,
    note: "A refilled day is NASA's dated answer on the day it was asked (refilled_at), not what the live poll would have recorded. no_rows: every satellite source answered that date with nothing, so NASA no longer serves it or nothing was seen.",
    generated_at: new Date().toISOString(),
  };
}

// ── scheduler ────────────────────────────────────────────────────────────────

let booted = false;
let timers: Array<ReturnType<typeof setTimeout>> = [];
const TICK_MS = 30 * 60_000;
const FIRST_TICK_MS = 2 * 60_000;

/** True when a pass is due: at or after 06:00 UTC and not yet run this UTC day. */
export function firesCompletenessDue(ledger: FiresLedger, nowMs: number): boolean {
  if (new Date(nowMs).getUTCHours() < FIRES_COMPLETENESS_START_HOUR_UTC) return false;
  return ledger.last_run_day !== dayOf(nowMs);
}

/** First tick 2 minutes after boot (a container replaced within the half hour
 *  still gets its pass), then every 30 minutes; one pass per UTC day, kept in
 *  the ledger so a restart does not ask NASA again. */
export function bootFiresCompleteness(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  const tick = async () => {
    if (running || !firesCompletenessEnabled(env)) return;
    const ledger = readFiresLedger(ledgerPath(env));
    if (!firesCompletenessDue(ledger, Date.now())) return;
    running = true;
    beat("fires_completeness_daily", "start");
    try {
      const run = await runFiresCompleteness({
        mapKey: firmsKey(env), fetch: fetch as unknown as FiresFetch, ledger,
        persist: (l) => writeFiresLedger(l, ledgerPath(env)),
      });
      writeFiresLedger(ledger, ledgerPath(env));
      beat("fires_completeness_daily", "finish", run.status, run.detail);
    } catch (e) {
      beat("fires_completeness_daily", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  const first = setTimeout(() => { void tick(); }, FIRST_TICK_MS);
  first.unref?.();
  const every = setInterval(() => { void tick(); }, TICK_MS);
  every.unref?.();
  timers = [first, every];
}

/** GET /api/data/fires/completeness (counts only) + the daily tick. */
export function registerFiresCompletenessRoutes(app: Express, env: NodeJS.ProcessEnv = process.env): void {
  app.get("/api/data/fires/completeness", (_req, res) => {
    try { res.json(firesCompletenessStatus(env)); } catch (e: unknown) {
      res.status(500).json({ error: e instanceof Error ? e.message : "fires completeness status failed" });
    }
  });
  bootFiresCompleteness(env);
}

/** Test-only: stop the ticks and allow a fresh boot. */
export function _resetFiresCompletenessForTests(): void {
  for (const t of timers) { clearTimeout(t); clearInterval(t); }
  timers = [];
  booted = false;
}

// The lookback must sit inside the fires stream's local window on the
// permanent tier (see FIRES_REFILL_LOOKBACK_DAYS).
export const FIRES_LOCAL_WINDOW_DAYS = permanentLocalDays("fires");
