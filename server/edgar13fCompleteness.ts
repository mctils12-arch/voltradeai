// 13F archive COMPLETENESS PASS — the nightly job that makes the 13F archive
// hold every 13F-HR / 13F-HR/A EDGAR received, not only the ones the
// 15-minute poll saw.
//
// WHY ([PIPELINE] 2026-11-19, open_questions.md 2026-11-17 item (k)).
// Measured 2026-11-19 against EDGAR's daily form index
// (scripts/edgar13f_index_check.py, research/edgar13f_index_check_2026-11-19.json):
// on the eight quiet filing days of 2026-11-02..11-17 the archive held every
// filing the index lists; on the three days around the 45-day deadline it
// held 755 of 782 (11-12), 905 of 1,087 (11-13) and 462 of 1,712 (11-16,
// 27%). The poll keeps the 40 newest filings every 15 minutes; a deadline day
// files more than that per quarter-hour. The filers who file on the deadline
// are disproportionately the small managers this archive exists for.
//
// WHAT. Each night the pass walks the weekdays from yesterday back to the
// archive's first day, newest first, skipping days its ledger has settled.
// For each day it reads EDGAR's daily form index, keeps form types "13F-HR"
// and "13F-HR/A" (exact), subtracts the accessions the archive holds, fetches
// each missing filing through the poll's own parser (fetch13FFilingDocs) and
// appends it through the poll's own writer (archive13FFilings) with two extra
// fields: `form` (the index's exact type) and `backfilledAt`. `filedAt` is
// the index's filing date. Nothing is rewritten; the archive stays
// append-only. The focused-manager cap applies unchanged.
//
// A day is SETTLED once every missing filing was tried and none is left
// failed (or FILINGS13F_COMPLETENESS_MAX_ATTEMPTS nights failed on the same
// ones); a settled day is not read again. The per-day ledger
// (voltrade_13f_day_ledger.json) is the record that tells a complete day
// from a sampled one.
//
// RAW archive, ladder gate 1 (DATA): the daily index IS the external truth
// for "which 13F-HRs exist". No signal claim is made here.
//
// Off switch: VOLTRADE_13F_COMPLETENESS=0.

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { promisify } from "util";
import { archiveBaseDir } from "./datacoreArchive";
import { FILINGS13F_ARCHIVE_START, archive13FFilings, fetch13FFilingDocs, type Filing13F } from "./edgar13f";
import { dailyFormIndexUrl, filingDirUrl, type CompletenessFetch } from "./edgarForm4Completeness";
import { beat, heartbeatDir, zonedParts } from "./jobHeartbeat";

const ET = "America/New_York";
const DAY_MS = 86_400_000;
const gunzip = promisify(zlib.gunzip);

export const FILINGS13F_COMPLETENESS_JOB = "filings13f_completeness_nightly";
/** Most filings fetched in one night (three requests each; about 8 minutes
 *  at the measured 0.76 s a filing). Recovered rows sit in the night's own
 *  day file and the history reader returns its newest day whole, so this
 *  also bounds that page: 600 rows at the ~6-8 KB a full row measured is
 *  under 5 MB on the keyed route, which serves the whole holdings table. */
export const FILINGS13F_COMPLETENESS_MAX_PER_RUN = 600;
/** Pause after each filing. With the three EDGAR polls beside it this keeps
 *  the process far under SEC's 10 requests/second. */
export const FILINGS13F_COMPLETENESS_DELAY_MS = 400;
/** This many filings in a row answered 403/429 = EDGAR is refusing us; stop. */
export const FILINGS13F_COMPLETENESS_REFUSAL_STOP = 5;
/** This many weekdays in a row answering "no index" is EDGAR refusing the
 *  index reads, not holidays (no three weekdays in a row are closed). */
export const FILINGS13F_COMPLETENESS_NO_INDEX_STOP = 3;
/** Nights a day's failed filings are retried before the day is settled with
 *  its failures stated. */
export const FILINGS13F_COMPLETENESS_MAX_ATTEMPTS = 3;
/** A weekday with no index is settled as "EDGAR closed" only this many days
 *  later: yesterday's index may simply not be published yet. */
export const FILINGS13F_NO_INDEX_SETTLE_AFTER_DAYS = 3;
/** The pass works 06:00-09:00 ET: after the Form 4 pass's 02:00-06:00
 *  window, so the two never fetch from EDGAR at once. */
export const FILINGS13F_COMPLETENESS_WINDOW_START_MIN = 6 * 60;
export const FILINGS13F_COMPLETENESS_WINDOW_END_MIN = 9 * 60;
export const FILINGS13F_DAY_LEDGER_FILENAME = "voltrade_13f_day_ledger.json";
export const FILINGS13F_COMPLETENESS_STATE_FILENAME = "voltrade_13f_completeness.json";
const APPEND_BATCH = 25;

const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };

// ── the daily form index ─────────────────────────────────────────────────────

export interface Index13F {
  accession: string;
  /** the index's exact form type */
  form: "13F-HR" | "13F-HR/A";
  /** first filer CIK the index lists for the accession (its directory) */
  cik: string;
  /** YYYY-MM-DD */
  filedAt: string;
}

const INDEX_LINE = /^(13F-HR|13F-HR\/A)\s{2,}.*\s(\d{1,10})\s+(\d{8})\s+edgar\/data\/\d+\/(\d{10}-\d{2}-\d{6})\.txt\s*$/;

/** The 13F-HR and 13F-HR/A filings of a daily form index, one per accession.
 *  The form type is matched exactly: "13F-NT" (a notice, no holdings) and
 *  "13F-NT/A" are not holdings reports. */
export function parseDailyFormIndex13F(text: string): Index13F[] {
  const seen = new Set<string>();
  const out: Index13F[] = [];
  for (const line of text.split("\n")) {
    const m = INDEX_LINE.exec(line);
    if (!m) continue;
    const accession = m[4];
    if (seen.has(accession)) continue;
    seen.add(accession);
    const ymd = m[3];
    out.push({
      accession,
      form: m[1] as Index13F["form"],
      cik: m[2],
      filedAt: `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}`,
    });
  }
  return out;
}

// ── what the archive holds ───────────────────────────────────────────────────

function filings13fDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "filings13f");
}

const ACCESSION_IN_ROW = /"accession":"(\d{10}-\d{2}-\d{6})"/;

/** Every accession in every day file of the 13F archive, read once per run.
 *  The accession is taken by pattern, not by parsing 40 KB rows; the files
 *  are read one at a time off the event loop. */
export async function held13FAccessions(dir: string): Promise<Set<string>> {
  const held = new Set<string>();
  let names: string[];
  try { names = await fs.promises.readdir(dir); } catch { return held; }
  for (const name of names) {
    if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(name)) continue;
    let text: string;
    try {
      const buf = await fs.promises.readFile(path.join(dir, name));
      text = name.endsWith(".gz") ? (await gunzip(buf)).toString("utf8") : buf.toString("utf8");
    } catch { continue; /* a file gzipped or removed mid-read is read next night */ }
    for (const line of text.split("\n")) {
      const m = ACCESSION_IN_ROW.exec(line);
      if (m) held.add(m[1]);
    }
  }
  return held;
}

// ── which days ───────────────────────────────────────────────────────────────

function ymd(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

/** The weekdays from yesterday (ET) back to the archive's first day, newest first. */
export function weekdaysBack(nowMs: number, archiveStart = FILINGS13F_ARCHIVE_START): string[] {
  const today = Date.parse(`${zonedParts(nowMs, ET).date}T00:00:00Z`);
  const start = Date.parse(`${archiveStart}T00:00:00Z`);
  const out: string[] = [];
  for (let t = today - DAY_MS; t >= start; t -= DAY_MS) {
    const dow = new Date(t).getUTCDay();
    if (dow === 0 || dow === 6) continue;
    out.push(ymd(t));
  }
  return out;
}

export function in13FCompletenessWindow(nowMs: number): boolean {
  const m = zonedParts(nowMs, ET).minutes;
  return m >= FILINGS13F_COMPLETENESS_WINDOW_START_MIN && m < FILINGS13F_COMPLETENESS_WINDOW_END_MIN;
}

export function filings13fCompletenessEnabled(env: Record<string, string | undefined>): boolean {
  const raw = env.VOLTRADE_13F_COMPLETENESS;
  const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

// ── the ledger ───────────────────────────────────────────────────────────────

/** One filed day's standing against EDGAR's index, kept across nights. */
export interface Ledger13FDay {
  /** "checked": index read and compared; "no_index": EDGAR has no index for the day */
  status: "checked" | "no_index";
  /** distinct 13F-HR + 13F-HR/A accessions in EDGAR's index */
  index_13f: number;
  /** of those, in the archive after the latest night that read the day */
  held: number;
  /** fetched, but the filing's directory had no primary document (not appended) */
  unparsed: number;
  /** fetch failed on the latest night */
  failed: number;
  /** left for a later night by the per-run cap or an early stop */
  deferred: number;
  /** nights this day's index was read */
  attempts: number;
  /** held == index_13f */
  complete: boolean;
  /** not read again */
  settled: boolean;
  checked_at: string;
}

export interface Ledger13F {
  days: Record<string, Ledger13FDay>;
}

function ledgerPath(env: Record<string, string | undefined> = process.env): string {
  return path.join(heartbeatDir(env), FILINGS13F_DAY_LEDGER_FILENAME);
}

function statePath(env: Record<string, string | undefined> = process.env): string {
  return path.join(heartbeatDir(env), FILINGS13F_COMPLETENESS_STATE_FILENAME);
}

function readJson<T>(file: string, what: string): T | null {
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")) as T;
  } catch (e) {
    console.error(`[datacore] 13F completeness ${what} unreadable:`, e instanceof Error ? e.message : e);
    return null;
  }
}

function writeJson(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value));
  fs.renameSync(tmp, file);
}

export function read13FLedger(file = ledgerPath()): Ledger13F {
  const l = readJson<Ledger13F>(file, "ledger");
  return l && typeof l === "object" && l.days && typeof l.days === "object" ? l : { days: {} };
}

export function write13FLedger(ledger: Ledger13F, file = ledgerPath()): void {
  writeJson(file, ledger);
}

// ── the pass ─────────────────────────────────────────────────────────────────

export interface Completeness13FRun {
  started_at: string;
  finished_at: string;
  status: "ok" | "partial" | "error";
  detail: string;
  /** days whose index was read this night */
  days_read: number;
  totals: { index_13f: number; held_before: number; backfilled: number; unparsed: number; failed: number; deferred: number };
  /** the per-run cap stopped the pass before every missing filing was tried */
  capped: boolean;
  /** EDGAR refused filings or index reads and the pass stopped */
  refused: boolean;
}

export interface Completeness13FDeps {
  fetch: CompletenessFetch;
  ledger: Ledger13F;
  now?: () => number;
  baseDir?: string;
  delayMs?: number;
  maxPerRun?: number;
  archiveStart?: string;
  /** the archive writer; the poll's own by default */
  append?: (rows: Filing13F[], baseDir?: string, nowMs?: number) => number;
}

class HttpStatusError extends Error {
  constructor(readonly status: number, url: string) {
    super(`${url} -> ${status}`);
  }
}

async function getText(fetchImpl: CompletenessFetch, url: string, timeoutMs: number): Promise<string> {
  const r = await fetchImpl(url, { headers: SEC_UA, signal: AbortSignal.timeout(timeoutMs) });
  if (!r.ok) throw new HttpStatusError(r.status, url);
  return r.text();
}

const sleep = (ms: number) => new Promise<void>((res) => { setTimeout(res, ms); });

/** One night's pass. Idempotent: a second pass finds the rows the first one
 *  appended and the days the ledger settled. Mutates `deps.ledger`. */
export async function run13FCompleteness(deps: Completeness13FDeps): Promise<Completeness13FRun> {
  const now = deps.now ?? Date.now;
  const delayMs = deps.delayMs ?? FILINGS13F_COMPLETENESS_DELAY_MS;
  const max = deps.maxPerRun ?? FILINGS13F_COMPLETENESS_MAX_PER_RUN;
  const append = deps.append ?? archive13FFilings;
  const ledger = deps.ledger;
  const startedMs = now();
  const today = Date.parse(`${zonedParts(startedMs, ET).date}T00:00:00Z`);
  const totals = { index_13f: 0, held_before: 0, backfilled: 0, unparsed: 0, failed: 0, deferred: 0 };
  let tried = 0, consecutiveRefusals = 0, refused = false, capped = false, indexErrors = 0, daysRead = 0;
  let noIndexRun: string[] = [];
  const settleNoIndex = (day: string) => {
    const old = (today - Date.parse(`${day}T00:00:00Z`)) / DAY_MS >= FILINGS13F_NO_INDEX_SETTLE_AFTER_DAYS;
    ledger.days[day] = {
      status: "no_index", index_13f: 0, held: 0, unparsed: 0, failed: 0, deferred: 0,
      attempts: (ledger.days[day]?.attempts ?? 0) + 1, complete: false, settled: old,
      checked_at: new Date(now()).toISOString(),
    };
  };

  let held: Set<string> | null = null;
  for (const day of weekdaysBack(startedMs, deps.archiveStart)) {
    if (refused || capped) break;
    if (ledger.days[day]?.settled) continue;
    let indexText: string;
    try {
      indexText = await getText(deps.fetch, dailyFormIndexUrl(day), 60_000);
    } catch (e) {
      // EDGAR answers 403 or 404 for a day it has no index for, and 403 when
      // it refuses us: one or two in a row are holidays, three are a refusal.
      const status = e instanceof HttpStatusError ? e.status : 0;
      if (status === 403 || status === 404) {
        noIndexRun.push(day);
        if (noIndexRun.length >= FILINGS13F_COMPLETENESS_NO_INDEX_STOP) refused = true;
      } else {
        indexErrors++;
        noIndexRun = [];
      }
      continue;
    }
    for (const d of noIndexRun) settleNoIndex(d);
    noIndexRun = [];
    daysRead++;
    held ??= await held13FAccessions(filings13fDir(deps.baseDir));
    const heldNow = held;
    const listed = parseDailyFormIndex13F(indexText);
    const missing = listed.filter((f) => !heldNow.has(f.accession));
    const rec = { backfilled: 0, unparsed: 0, failed: 0, deferred: 0 };
    let batch: Filing13F[] = [];
    const flush = () => {
      if (!batch.length) return;
      rec.backfilled += append(batch, deps.baseDir, now());
      for (const f of batch) heldNow.add(f.accession);
      batch = [];
    };
    for (let i = 0; i < missing.length; i++) {
      if (refused || tried >= max) {
        if (!refused) capped = true;
        rec.deferred = missing.length - i;
        break;
      }
      const f = missing[i];
      tried++;
      try {
        const base = filingDirUrl(f);
        const doc = await fetch13FFilingDocs(base, (url) => getText(deps.fetch, url, 20_000));
        consecutiveRefusals = 0;
        if (doc) {
          batch.push({
            ...doc,
            accession: f.accession,
            filedAt: f.filedAt,
            cik: f.cik.padStart(10, "0"), // the live feed's rows carry the 10-digit form
            indexUrl: `${base}${f.accession}-index.htm`,
            form: f.form,
            backfilledAt: new Date(now()).toISOString(),
          });
          if (batch.length >= APPEND_BATCH) flush();
        } else {
          rec.unparsed++;
        }
      } catch (e) {
        rec.failed++;
        const status = e instanceof HttpStatusError ? e.status : 0;
        consecutiveRefusals = status === 403 || status === 429 ? consecutiveRefusals + 1 : 0;
        if (consecutiveRefusals >= FILINGS13F_COMPLETENESS_REFUSAL_STOP) refused = true;
      }
      if (delayMs > 0) await sleep(delayMs);
    }
    flush();
    const heldBefore = listed.length - missing.length;
    const attempts = (ledger.days[day]?.attempts ?? 0) + 1;
    const heldAfter = heldBefore + rec.backfilled;
    ledger.days[day] = {
      status: "checked", index_13f: listed.length, held: heldAfter,
      unparsed: rec.unparsed, failed: rec.failed, deferred: rec.deferred, attempts,
      complete: heldAfter === listed.length,
      settled: rec.deferred === 0 && (rec.failed === 0 || attempts >= FILINGS13F_COMPLETENESS_MAX_ATTEMPTS),
      checked_at: new Date(now()).toISOString(),
    };
    totals.index_13f += listed.length;
    totals.held_before += heldBefore;
    totals.backfilled += rec.backfilled;
    totals.unparsed += rec.unparsed;
    totals.failed += rec.failed;
    totals.deferred += rec.deferred;
  }
  // A trailing run of no-index days that did not reach the refusal count
  // (the walk ended on a holiday) is recorded; a refusal records nothing.
  if (!refused) for (const d of noIndexRun) settleNoIndex(d);

  const status: Completeness13FRun["status"] =
    refused || (daysRead === 0 && indexErrors > 0) ? "error"
      : capped || indexErrors > 0 || totals.failed > 0 ? "partial"
        : "ok";
  const detail =
    `${daysRead} day(s) read; index ${totals.index_13f}, held ${totals.held_before}, backfilled ${totals.backfilled}` +
    `, unparsed ${totals.unparsed}, failed ${totals.failed}, deferred ${totals.deferred}` +
    (refused ? "; stopped: EDGAR refused requests" : capped ? "; stopped at the per-run cap" : "") +
    (indexErrors ? `; ${indexErrors} index read(s) failed` : "");
  return {
    started_at: new Date(startedMs).toISOString(), finished_at: new Date(now()).toISOString(),
    status, detail, days_read: daysRead, totals, capped, refused,
  };
}

// ── status (token-gated diag probe `filings13f_completeness`) ────────────────

/** The ledger in counts: how many weekdays are settled, complete, short. */
export function ledger13FSummary(ledger: Ledger13F, nowMs: number, archiveStart = FILINGS13F_ARCHIVE_START): Record<string, unknown> {
  const weekdays = weekdaysBack(nowMs, archiveStart);
  let settled = 0, complete = 0, closed = 0, short = 0, index = 0, held = 0;
  let completeBackTo: string | null = null;
  let unbroken = true;
  const shortDays: Array<{ day: string; index_13f: number; held: number }> = [];
  for (const day of weekdays) { // newest first
    const d = ledger.days[day];
    const ok = !!d && ((d.status === "checked" && d.complete) || (d.status === "no_index" && d.settled));
    if (unbroken && ok) completeBackTo = day; else unbroken = false;
    if (!d) continue;
    if (d.settled) settled++;
    if (d.status === "no_index") { if (d.settled) closed++; continue; }
    index += d.index_13f;
    held += d.held;
    if (d.complete) complete++;
    else {
      short++;
      if (shortDays.length < 20) shortDays.push({ day, index_13f: d.index_13f, held: d.held });
    }
  }
  return {
    weekdays: weekdays.length,
    settled_days: settled,
    complete_days: complete,
    closed_days: closed,
    short_days: short,
    not_read_days: weekdays.filter((d) => !ledger.days[d]).length,
    /** every weekday from here to yesterday is complete or EDGAR was closed */
    complete_back_to: completeBackTo,
    index_13f_checked: index,
    held_checked: held,
    /** newest first, at most 20: the days still short of the index */
    short: shortDays,
  };
}

/** Counts and days only: no manager, CIK, accession or URL. */
export function filings13fCompletenessStatus(env: Record<string, string | undefined> = process.env): Record<string, unknown> {
  return {
    probe: "filings13f_completeness",
    enabled: filings13fCompletenessEnabled(env),
    rule: `Nightly ${FILINGS13F_COMPLETENESS_WINDOW_START_MIN / 60}:00-${FILINGS13F_COMPLETENESS_WINDOW_END_MIN / 60}:00 ET: for each weekday from yesterday back to ${FILINGS13F_ARCHIVE_START} that the ledger has not settled, newest first, EDGAR's daily form index (types 13F-HR and 13F-HR/A, exact) minus the accessions the archive holds is fetched and appended with form + backfilledAt. At most ${FILINGS13F_COMPLETENESS_MAX_PER_RUN} filings a night. A day is settled when every missing filing was tried and none is left failed, or after ${FILINGS13F_COMPLETENESS_MAX_ATTEMPTS} nights.`,
    truth_source: "https://www.sec.gov/Archives/edgar/daily-index/ form.<yyyymmdd>.idx",
    latest: readJson<Completeness13FRun>(statePath(env), "state"),
    ledger: ledger13FSummary(read13FLedger(ledgerPath(env)), Date.now()),
  };
}

// ── scheduler ────────────────────────────────────────────────────────────────

/** One night's work; persists the ledger and the probe state. */
export async function run13FNight(
  deps: Omit<Completeness13FDeps, "ledger">,
  env: Record<string, string | undefined> = process.env,
): Promise<{ run: Completeness13FRun; failed: boolean }> {
  const ledger = read13FLedger(ledgerPath(env));
  const run = await run13FCompleteness({ ...deps, ledger });
  write13FLedger(ledger, ledgerPath(env));
  writeJson(statePath(env), run);
  return { run, failed: run.status === "error" };
}

let booted = false;
let timer: ReturnType<typeof setInterval> | null = null;
const TICK_MS = 30 * 60_000;

/** 30-minute tick (unref'd; a boot never starts a pass by itself). One pass
 *  per ET morning inside the window; a restart inside the window may run it
 *  again, which is harmless (the pass is idempotent). */
export function boot13FCompleteness(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastNight = "";
  const tick = async () => {
    if (running || !filings13fCompletenessEnabled(env)) return;
    const nowMs = Date.now();
    if (!in13FCompletenessWindow(nowMs)) return;
    const night = zonedParts(nowMs, ET).date;
    if (lastNight === night) return;
    lastNight = night;
    running = true;
    beat("filings13f_completeness_nightly", "start");
    try {
      const { run, failed } = await run13FNight({ fetch: fetch as unknown as CompletenessFetch }, env);
      beat("filings13f_completeness_nightly", "finish", failed ? "error" : "ok", run.detail);
    } catch (e) {
      beat("filings13f_completeness_nightly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  timer = setInterval(() => { void tick(); }, TICK_MS);
  timer.unref?.();
}

/** Test-only: stop the tick and allow a fresh boot. */
export function _reset13FCompletenessForTests(): void {
  if (timer) clearInterval(timer);
  timer = null;
  booted = false;
}
