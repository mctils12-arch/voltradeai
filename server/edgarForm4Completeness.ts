// Form 4 archive COMPLETENESS PASS — the nightly job that makes the Form 4
// archive hold every Form 4 EDGAR received, not only the ones the 15-minute
// poll happened to see.
//
// WHY ([PIPELINE] 2026-11-17, open_questions.md 2026-11-16 "the Form 4 archive
// recorded nothing…" NEXT (1) and (3)). Measured 2026-11-17 against EDGAR's
// own daily form index for Friday 2026-11-13: the index lists 663 distinct
// Form 4 accessions and 12 Form 4/A; the archive held 404 and 7 (61%). The
// live poll keeps the 25 newest entries of a feed whose `type=4` is a PREFIX
// match, so 424B2 / 424B3 / 485BPOS / 497 filings take poll slots (189 of the
// 600 rows filed that day are such filings, archived as empty shells), and a
// busy quarter-hour holds more than 25 Form 4s. Monday 2026-11-16 additionally
// lost about eleven hours to a refused page size.
//
// WHAT. EDGAR publishes `Archives/edgar/daily-index/<year>/QTR<q>/
// form.<yyyymmdd>.idx` for every dissemination day: a permanent, complete,
// exactly-typed list of that day's accessions. Each night this pass reads the
// index of each recent weekday, keeps form types "4" and "4/A", subtracts the
// accessions the archive already holds as parsed rows, fetches each missing
// filing through the poll's own index.json -> ownership XML path, and appends
// it through the poll's own writer (`archiveFilings`) with two extra fields:
// `form` (the index's exact type) and `backfilledAt` (so a reader can always
// tell a row recorded live from one recovered later). `filedAt` is the
// index's filing date. Nothing is rewritten; the archive stays append-only.
//
// RAW archive, ladder gate 1 (DATA): the daily index IS the external truth
// source for "which Form 4s exist". No signal claim is made here.
//
// Off switch: VOLTRADE_FORM4_COMPLETENESS=0.
//
// OLDER DAYS ([PIPELINE] 2026-11-17, open_questions.md 2026-11-17 "STILL OPEN
// (a)"). The pass above reaches 7 days. Every weekday from the archive's first
// day to the edge of that reach was recorded by the poll alone. Measured
// 2026-11-17 against EDGAR's index: the archive holds 576 of 1,227 Form 4s
// filed 2026-07-06 (47%) and 63 of 511 filed 2026-09-15 (12%, the week
// production was down). The daily indexes are permanent, so after the recent
// days each night the same pass walks the older weekdays, newest first, up to
// FORM4_REFILL_MAX_PER_RUN filings a night, and keeps a per-day LEDGER
// (voltrade_form4_day_ledger.json) of what EDGAR listed and what the archive
// holds. A day is SETTLED once every missing filing was tried and none is
// left failed (or three nights failed on the same ones); a settled day is not
// read again. The ledger is the record a reader uses to tell a complete day
// from a sampled one. Off switch for the older days only:
// VOLTRADE_FORM4_REFILL=0.

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { FORM4_ARCHIVE_START, archiveFilings, parseForm4Xml, pickOwnershipXmlName, type Form4Filing } from "./edgarForm4";
import { beat, heartbeatDir, zonedParts } from "./jobHeartbeat";

const ET = "America/New_York";
const DAY_MS = 86_400_000;

export const FORM4_COMPLETENESS_JOB = "form4_completeness_nightly";
/** Calendar days back from yesterday (ET) whose index is checked each night. */
export const FORM4_COMPLETENESS_LOOKBACK_DAYS = 7;
/** Most filings fetched in one night; the rest wait for the next night. */
export const FORM4_COMPLETENESS_MAX_PER_RUN = 2500;
/** Pause after each filing (two requests). With the three EDGAR polls running
 *  beside it this keeps the process far under SEC's 10 requests/second. */
export const FORM4_COMPLETENESS_DELAY_MS = 400;
/** This many filings in a row answered 403/429 = EDGAR is refusing us; stop. */
export const FORM4_COMPLETENESS_REFUSAL_STOP = 5;
/** Rows handed to the writer at a time, so an interrupted run keeps its work. */
const APPEND_BATCH = 50;
/** The pass works 02:00-06:00 ET: the previous day's index is published and
 *  nothing else in the process is busy. */
export const FORM4_COMPLETENESS_WINDOW_START_MIN = 2 * 60;
export const FORM4_COMPLETENESS_WINDOW_END_MIN = 6 * 60;
export const FORM4_COMPLETENESS_STATE_FILENAME = "voltrade_form4_completeness.json";
/** Most OLDER-day filings fetched in one night (about 31 minutes at the
 *  measured 0.62 s a filing). Recovered rows sit in the night's own day file,
 *  and the history reader returns its newest day whole, so this also bounds
 *  how large that page can get. */
export const FORM4_REFILL_MAX_PER_RUN = 3000;
/** Nights a day's failed filings are retried before the day is settled with
 *  its failures stated. */
export const FORM4_REFILL_MAX_ATTEMPTS = 3;
/** This many older days in a row answering "no index" is EDGAR refusing the
 *  index reads, not holidays (no three weekdays in a row are closed). */
export const FORM4_REFILL_NO_INDEX_STOP = 3;
export const FORM4_DAY_LEDGER_FILENAME = "voltrade_form4_day_ledger.json";

const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };

export type CompletenessFetch = (
  url: string,
  init?: { headers?: Record<string, string>; signal?: AbortSignal },
) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

// ── the daily form index ─────────────────────────────────────────────────────

export interface IndexFiling {
  accession: string;
  /** the index's exact form type: "4" or "4/A" */
  form: "4" | "4/A";
  /** first filer CIK the index lists for the accession (its directory) */
  cik: string;
  /** YYYY-MM-DD */
  filedAt: string;
}

/** URL of EDGAR's daily form index for a YYYY-MM-DD day. */
export function dailyFormIndexUrl(day: string): string {
  const [y, m, d] = day.split("-");
  const quarter = Math.floor((Number(m) - 1) / 3) + 1;
  return `https://www.sec.gov/Archives/edgar/daily-index/${y}/QTR${quarter}/form.${y}${m}${d}.idx`;
}

const INDEX_LINE = /^(4|4\/A)\s{2,}.*\s(\d{1,10})\s+(\d{8})\s+edgar\/data\/\d+\/(\d{10}-\d{2}-\d{6})\.txt\s*$/;

/** The Form 4 and Form 4/A filings of a daily form index, one per accession
 *  (the index repeats an accession once per filer: issuer and each owner).
 *  The form type is matched exactly — "424B2", "40-F", "425" are not Form 4. */
export function parseDailyFormIndex(text: string): IndexFiling[] {
  const seen = new Set<string>();
  const out: IndexFiling[] = [];
  for (const line of text.split("\n")) {
    const m = INDEX_LINE.exec(line);
    if (!m) continue;
    const accession = m[4];
    if (seen.has(accession)) continue;
    seen.add(accession);
    const ymd = m[3];
    out.push({
      accession,
      form: m[1] as "4" | "4/A",
      cik: m[2],
      filedAt: `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}`,
    });
  }
  return out;
}

/** The filing's directory, the same one the live feed's index link points into. */
export function filingDirUrl(f: Pick<IndexFiling, "accession" | "cik">): string {
  return `https://www.sec.gov/Archives/edgar/data/${Number(f.cik)}/${f.accession.replace(/-/g, "")}/`;
}

// ── what the archive holds ───────────────────────────────────────────────────

function filingsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "filings");
}

function ymd(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

function readDayText(dir: string, day: string): string {
  let text = "";
  for (const name of [`${day}.jsonl.gz`, `${day}.jsonl`]) {
    const fp = path.join(dir, name);
    if (!fs.existsSync(fp)) continue;
    const buf = fs.readFileSync(fp);
    text += (name.endsWith(".gz") ? zlib.gunzipSync(buf).toString("utf8") : buf.toString("utf8")) + "\n";
  }
  return text;
}

/** Accessions held as PARSED ownership rows in the day files `fromDay`..`toDay`
 *  (inclusive, UTC file days). A row with no issuer is an empty shell (the
 *  poll parsed a non-ownership document) and does not count as held. A filing
 *  filed on day D is never archived under a file day before D. */
export function heldParsedAccessions(dir: string, fromDay: string, toDay: string): Set<string> {
  const held = new Set<string>();
  const end = Date.parse(`${toDay}T00:00:00Z`);
  for (let t = Date.parse(`${fromDay}T00:00:00Z`); t <= end; t += DAY_MS) {
    for (const line of readDayText(dir, ymd(t)).split("\n")) {
      if (!line) continue;
      let row: { accession?: unknown; issuerCik?: unknown };
      try { row = JSON.parse(line); } catch { continue; /* a torn line is not a held filing */ }
      if (typeof row.accession === "string" && row.issuerCik) held.add(row.accession);
    }
  }
  return held;
}

// ── which days to check ──────────────────────────────────────────────────────

/** The weekdays (Mon-Fri) among the `lookback` calendar days before today's
 *  ET date, oldest first. A market holiday is simply a day with no index. */
export function candidateDays(nowMs: number, lookback = FORM4_COMPLETENESS_LOOKBACK_DAYS): string[] {
  const today = Date.parse(`${zonedParts(nowMs, ET).date}T00:00:00Z`);
  const out: string[] = [];
  for (let back = lookback; back >= 1; back--) {
    const t = today - back * DAY_MS;
    const dow = new Date(t).getUTCDay();
    if (dow === 0 || dow === 6) continue;
    out.push(ymd(t));
  }
  return out;
}

export function inCompletenessWindow(nowMs: number): boolean {
  const m = zonedParts(nowMs, ET).minutes;
  return m >= FORM4_COMPLETENESS_WINDOW_START_MIN && m < FORM4_COMPLETENESS_WINDOW_END_MIN;
}

export function form4CompletenessEnabled(env: Record<string, string | undefined>): boolean {
  const raw = env.VOLTRADE_FORM4_COMPLETENESS;
  const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

// ── the pass ─────────────────────────────────────────────────────────────────

export interface DayCompleteness {
  day: string;
  /** "checked": index read and compared; "no_index": EDGAR has no index for
   *  the day (holiday, or not published yet); "error": the index read failed */
  status: "checked" | "no_index" | "error";
  /** distinct Form 4 + 4/A accessions in EDGAR's index */
  index_form4: number;
  /** of those, held as parsed rows before this run */
  held_before: number;
  /** rows this run appended */
  backfilled: number;
  /** fetched, but the document held no issuer (not appended) */
  unparsed: number;
  /** fetch failed (retried the next night) */
  failed: number;
  /** left for the next night by the per-run cap or an early stop */
  deferred: number;
  /** held_before + backfilled == index_form4 */
  complete: boolean;
  detail?: string;
}

export interface CompletenessRun {
  started_at: string;
  finished_at: string;
  status: "ok" | "partial" | "error";
  detail: string;
  days: DayCompleteness[];
  totals: { index_form4: number; held_before: number; backfilled: number; unparsed: number; failed: number; deferred: number };
  /** the per-run cap stopped the pass before every missing filing was tried */
  capped: boolean;
  /** EDGAR answered 403/429 to several filings in a row and the pass stopped */
  refused: boolean;
}

export interface CompletenessDeps {
  fetch: CompletenessFetch;
  now?: () => number;
  baseDir?: string;
  delayMs?: number;
  maxPerRun?: number;
  lookbackDays?: number;
  /** the archive writer; the poll's own by default */
  append?: (rows: Form4Filing[], baseDir?: string, nowMs?: number) => number;
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

/** Shared between the recent pass and the older-day refill: how many filings
 *  this run has tried, its cap, and whether EDGAR is refusing. */
interface FetchBudget {
  tried: number;
  max: number;
  consecutiveRefusals: number;
  refused: boolean;
  capped: boolean;
}

/** Fetches each missing filing of one day through the poll's own index.json
 *  -> ownership XML path and appends the parsed ones in batches. Fills
 *  `rec.backfilled / unparsed / failed / deferred`. */
async function fetchMissingFilings(
  missing: IndexFiling[],
  rec: DayCompleteness,
  budget: FetchBudget,
  deps: Pick<CompletenessDeps, "fetch" | "now" | "baseDir" | "delayMs" | "append">,
): Promise<void> {
  const now = deps.now ?? Date.now;
  const delayMs = deps.delayMs ?? FORM4_COMPLETENESS_DELAY_MS;
  const append = deps.append ?? archiveFilings;
  let batch: Form4Filing[] = [];
  const flush = () => {
    if (batch.length) rec.backfilled += append(batch, deps.baseDir, now());
    batch = [];
  };
  for (let i = 0; i < missing.length; i++) {
    if (budget.refused || budget.tried >= budget.max) {
      if (!budget.refused) budget.capped = true;
      rec.deferred = missing.length - i;
      break;
    }
    const f = missing[i];
    budget.tried++;
    try {
      const base = filingDirUrl(f);
      const xmlName = pickOwnershipXmlName(JSON.parse(await getText(deps.fetch, `${base}index.json`, 12_000)));
      const parsed = xmlName ? parseForm4Xml(await getText(deps.fetch, `${base}${xmlName}`, 12_000)) : null;
      budget.consecutiveRefusals = 0;
      if (parsed && parsed.issuerCik) {
        batch.push({
          ...parsed,
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
      budget.consecutiveRefusals = status === 403 || status === 429 ? budget.consecutiveRefusals + 1 : 0;
      if (budget.consecutiveRefusals >= FORM4_COMPLETENESS_REFUSAL_STOP) budget.refused = true;
    }
    if (delayMs > 0) await sleep(delayMs);
  }
  flush();
}

/** One completeness pass over the recent weekdays, oldest first. Idempotent:
 *  a second pass over the same days finds the rows the first one appended. */
export async function runForm4Completeness(deps: CompletenessDeps): Promise<CompletenessRun> {
  const now = deps.now ?? Date.now;
  const maxPerRun = deps.maxPerRun ?? FORM4_COMPLETENESS_MAX_PER_RUN;
  const dir = filingsDir(deps.baseDir);
  const startedMs = now();
  const days: DayCompleteness[] = [];
  const budget: FetchBudget = { tried: 0, max: maxPerRun, consecutiveRefusals: 0, refused: false, capped: false };

  for (const day of candidateDays(startedMs, deps.lookbackDays)) {
    const rec: DayCompleteness = { day, status: "checked", index_form4: 0, held_before: 0, backfilled: 0, unparsed: 0, failed: 0, deferred: 0, complete: false };
    days.push(rec);
    let indexText: string;
    try {
      indexText = await getText(deps.fetch, dailyFormIndexUrl(day), 60_000);
    } catch (e) {
      // EDGAR answers 403 or 404 for a day it has no index for.
      const status = e instanceof HttpStatusError ? e.status : 0;
      rec.status = status === 403 || status === 404 ? "no_index" : "error";
      rec.detail = e instanceof Error ? e.message.slice(0, 160) : String(e).slice(0, 160);
      continue;
    }
    const listed = parseDailyFormIndex(indexText);
    const held = heldParsedAccessions(dir, day, ymd(now()));
    const missing = listed.filter((f) => !held.has(f.accession));
    rec.index_form4 = listed.length;
    rec.held_before = listed.length - missing.length;

    await fetchMissingFilings(missing, rec, budget, deps);
    rec.complete = rec.held_before + rec.backfilled === rec.index_form4;
  }

  const totals = { index_form4: 0, held_before: 0, backfilled: 0, unparsed: 0, failed: 0, deferred: 0 };
  for (const d of days) {
    totals.index_form4 += d.index_form4;
    totals.held_before += d.held_before;
    totals.backfilled += d.backfilled;
    totals.unparsed += d.unparsed;
    totals.failed += d.failed;
    totals.deferred += d.deferred;
  }
  const { refused, capped } = budget;
  const checked = days.filter((d) => d.status === "checked");
  const indexErrors = days.filter((d) => d.status === "error").length;
  const status: CompletenessRun["status"] =
    refused || (checked.length === 0 && indexErrors > 0) ? "error"
      : capped || indexErrors > 0 || checked.some((d) => !d.complete) ? "partial"
        : "ok";
  const detail =
    `${checked.length} day(s) checked; index ${totals.index_form4}, held ${totals.held_before}, backfilled ${totals.backfilled}` +
    `, unparsed ${totals.unparsed}, failed ${totals.failed}, deferred ${totals.deferred}` +
    (refused ? "; stopped: EDGAR refused requests" : capped ? "; stopped at the per-run cap" : "") +
    (indexErrors ? `; ${indexErrors} index read(s) failed` : "");
  return { started_at: new Date(startedMs).toISOString(), finished_at: new Date(now()).toISOString(), status, detail, days, totals, capped, refused };
}

// ── older days: the refill and its per-day ledger ────────────────────────────

/** One archive day's standing against EDGAR's index, kept across nights. */
export interface LedgerDay {
  day: string;
  status: DayCompleteness["status"];
  /** distinct Form 4 + 4/A accessions in EDGAR's index */
  index_form4: number;
  /** of those, held as parsed rows after the latest check */
  held: number;
  /** fetched, but the document held no issuer (latest check) */
  unparsed: number;
  /** fetch failed on the latest check */
  failed: number;
  /** not yet tried (per-run cap or an early stop) */
  deferred: number;
  /** held == index_form4 */
  complete: boolean;
  /** nothing more will be fetched for this day: every missing filing was
   *  tried and none is left failed, or the retries are used up */
  settled: boolean;
  /** nights this day's filings were fetched (or its index was absent) */
  attempts: number;
  checked_at: string;
}

export interface DayLedger {
  archive_start: string;
  updated_at: string;
  days: Record<string, LedgerDay>;
}

function ledgerPath(env: Record<string, string | undefined> = process.env): string {
  return path.join(heartbeatDir(env), FORM4_DAY_LEDGER_FILENAME);
}

export function readDayLedger(file = ledgerPath()): DayLedger {
  const empty: DayLedger = { archive_start: FORM4_ARCHIVE_START, updated_at: "", days: {} };
  if (!fs.existsSync(file)) return empty;
  try {
    const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as DayLedger;
    return parsed && typeof parsed.days === "object" && parsed.days ? parsed : empty;
  } catch (e) {
    // An unreadable ledger costs one re-read of the indexes, never a wrong "complete".
    console.error("[datacore] form4 day ledger unreadable:", e instanceof Error ? e.message : e);
    return empty;
  }
}

export function writeDayLedger(ledger: DayLedger, file = ledgerPath()): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(ledger));
  fs.renameSync(tmp, file);
}

/** Folds one night's result for a day into the ledger. */
export function noteLedgerDay(ledger: DayLedger, rec: DayCompleteness, nowMs: number): LedgerDay {
  const prior = ledger.days[rec.day];
  const attempts = (prior?.attempts ?? 0) + (rec.status === "error" ? 0 : 1);
  const held = rec.held_before + rec.backfilled;
  const complete = rec.status === "checked" && held === rec.index_form4;
  const retriesSpent = attempts >= FORM4_REFILL_MAX_ATTEMPTS;
  const settled =
    rec.status === "checked" ? complete || (rec.deferred === 0 && (rec.failed === 0 || retriesSpent))
      : rec.status === "no_index" ? retriesSpent
        : false;
  const next: LedgerDay = {
    day: rec.day, status: rec.status, index_form4: rec.index_form4, held,
    unparsed: rec.unparsed, failed: rec.failed, deferred: rec.deferred,
    complete, settled, attempts, checked_at: new Date(nowMs).toISOString(),
  };
  ledger.days[rec.day] = next;
  ledger.updated_at = next.checked_at;
  return next;
}

/** A day still inside the recent pass's 7-day reach: it is retried every night
 *  there, so it is settled only when complete and spends none of the older
 *  refill's retries. */
export function noteRecentLedgerDay(ledger: DayLedger, rec: DayCompleteness, nowMs: number): LedgerDay {
  const held = rec.held_before + rec.backfilled;
  const complete = rec.status === "checked" && held === rec.index_form4;
  const next: LedgerDay = {
    day: rec.day, status: rec.status, index_form4: rec.index_form4, held,
    unparsed: rec.unparsed, failed: rec.failed, deferred: rec.deferred,
    complete, settled: complete, attempts: 0, checked_at: new Date(nowMs).toISOString(),
  };
  ledger.days[rec.day] = next;
  ledger.updated_at = next.checked_at;
  return next;
}

/** The weekdays from the archive's first day to the day before the recent
 *  pass's reach, newest first (the series is then complete back to a date
 *  that moves earlier each night). */
export function olderDays(nowMs: number, archiveStart = FORM4_ARCHIVE_START, lookback = FORM4_COMPLETENESS_LOOKBACK_DAYS): string[] {
  const today = Date.parse(`${zonedParts(nowMs, ET).date}T00:00:00Z`);
  const start = Date.parse(`${archiveStart}T00:00:00Z`);
  const out: string[] = [];
  for (let t = today - (lookback + 1) * DAY_MS; t >= start; t -= DAY_MS) {
    const dow = new Date(t).getUTCDay();
    if (dow === 0 || dow === 6) continue;
    out.push(ymd(t));
  }
  return out;
}

const yieldToLoop = () => new Promise<void>((res) => { setImmediate(res); });

/** Every accession the archive holds as a parsed row, over ALL its day files.
 *  One day file is read per event-loop turn: the archive is months of files
 *  and this process also runs the trading loop. */
export async function heldParsedAccessionsAll(dir: string): Promise<Set<string>> {
  const held = new Set<string>();
  let names: string[] = [];
  try { names = fs.readdirSync(dir); } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") throw e; // no archive yet = nothing held
  }
  const days = Array.from(new Set(names.filter((n) => /^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(n)).map((n) => n.slice(0, 10))));
  for (const day of days) {
    heldParsedAccessions(dir, day, day).forEach((a) => held.add(a));
    await yieldToLoop();
  }
  return held;
}

export function form4RefillEnabled(env: Record<string, string | undefined>): boolean {
  const raw = env.VOLTRADE_FORM4_REFILL;
  const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

export interface RefillRun {
  started_at: string;
  finished_at: string;
  status: "ok" | "partial" | "error" | "finished";
  detail: string;
  /** the days this run read, newest first */
  days: DayCompleteness[];
  totals: { index_form4: number; held_before: number; backfilled: number; unparsed: number; failed: number; deferred: number };
  /** older weekdays still not settled after this run */
  remaining_days: number;
  capped: boolean;
  refused: boolean;
}

export interface RefillDeps extends CompletenessDeps {
  ledger: DayLedger;
  archiveStart?: string;
}

/** One night of the older-day refill. Reads only days the ledger has not
 *  settled, newest first, and stops at the per-run cap. Mutates `deps.ledger`
 *  (the caller persists it). */
export async function runForm4Refill(deps: RefillDeps): Promise<RefillRun> {
  const now = deps.now ?? Date.now;
  const startedMs = now();
  const ledger = deps.ledger;
  const all = olderDays(startedMs, deps.archiveStart ?? ledger.archive_start, deps.lookbackDays);
  const open = all.filter((d) => !ledger.days[d]?.settled);
  const days: DayCompleteness[] = [];
  const budget: FetchBudget = { tried: 0, max: deps.maxPerRun ?? FORM4_REFILL_MAX_PER_RUN, consecutiveRefusals: 0, refused: false, capped: false };
  const totals = { index_form4: 0, held_before: 0, backfilled: 0, unparsed: 0, failed: 0, deferred: 0 };
  let indexErrors = 0;

  if (open.length) {
    const held = await heldParsedAccessionsAll(filingsDir(deps.baseDir));
    let noIndexRun: DayCompleteness[] = [];
    for (const day of open) {
      if (budget.refused || budget.capped) break;
      const rec: DayCompleteness = { day, status: "checked", index_form4: 0, held_before: 0, backfilled: 0, unparsed: 0, failed: 0, deferred: 0, complete: false };
      let indexText: string;
      try {
        indexText = await getText(deps.fetch, dailyFormIndexUrl(day), 60_000);
      } catch (e) {
        const status = e instanceof HttpStatusError ? e.status : 0;
        rec.status = status === 403 || status === 404 ? "no_index" : "error";
        rec.detail = e instanceof Error ? e.message.slice(0, 160) : String(e).slice(0, 160);
        days.push(rec);
        if (rec.status === "error") { indexErrors++; noIndexRun = []; continue; }
        // A lone "no index" is a federal holiday. Several weekdays in a row
        // is EDGAR refusing us: none of them is recorded as a holiday.
        noIndexRun.push(rec);
        if (noIndexRun.length >= FORM4_REFILL_NO_INDEX_STOP) budget.refused = true;
        continue;
      }
      for (const r of noIndexRun) noteLedgerDay(ledger, r, now());
      noIndexRun = [];
      const listed = parseDailyFormIndex(indexText);
      const missing = listed.filter((f) => !held.has(f.accession));
      rec.index_form4 = listed.length;
      rec.held_before = listed.length - missing.length;
      days.push(rec);
      await fetchMissingFilings(missing, rec, budget, deps);
      rec.complete = rec.held_before + rec.backfilled === rec.index_form4;
      noteLedgerDay(ledger, rec, now());
    }
    // Trailing "no index" days (fewer than the stop) with no refusal: holidays.
    if (!budget.refused) for (const r of noIndexRun) noteLedgerDay(ledger, r, now());
  }

  for (const d of days) {
    totals.index_form4 += d.index_form4;
    totals.held_before += d.held_before;
    totals.backfilled += d.backfilled;
    totals.unparsed += d.unparsed;
    totals.failed += d.failed;
    totals.deferred += d.deferred;
  }
  const remaining = all.filter((d) => !ledger.days[d]?.settled).length;
  const status: RefillRun["status"] =
    budget.refused ? "error"
      : remaining === 0 ? "finished"
        : indexErrors > 0 && days.every((d) => d.status === "error") ? "error"
          : budget.capped || indexErrors > 0 ? "partial"
            : "ok";
  const detail =
    `older days: ${days.length} read, backfilled ${totals.backfilled}, unparsed ${totals.unparsed}, failed ${totals.failed}` +
    `, deferred ${totals.deferred}; ${remaining} of ${all.length} weekday(s) not settled` +
    (budget.refused ? "; stopped: EDGAR refused requests" : budget.capped ? "; stopped at the per-run cap" : "") +
    (indexErrors ? `; ${indexErrors} index read(s) failed` : "");
  return {
    started_at: new Date(startedMs).toISOString(), finished_at: new Date(now()).toISOString(),
    status, detail, days, totals, remaining_days: remaining, capped: budget.capped, refused: budget.refused,
  };
}

/** Counts for the probe: how much of the archive's span is complete. */
export function ledgerSummary(ledger: DayLedger, nowMs: number): Record<string, unknown> {
  const weekdays = [...olderDays(nowMs, ledger.archive_start), ...candidateDays(nowMs)];
  let settled = 0, complete = 0, noIndex = 0, index = 0, held = 0, shortDays = 0;
  let completeBackTo: string | null = null;
  // newest first: the unbroken run of complete (or closed) days from yesterday back
  const newestFirst = [...weekdays].sort().reverse();
  let unbroken = true;
  for (const day of newestFirst) {
    const d = ledger.days[day];
    const closed = d?.status === "no_index" && d.settled;
    if (d?.settled) settled++;
    if (d?.complete) complete++;
    if (closed) noIndex++;
    if (d?.status === "checked") { index += d.index_form4; held += d.held; if (d.settled && !d.complete) shortDays++; }
    if (unbroken && (d?.complete || closed)) completeBackTo = day; else unbroken = false;
  }
  return {
    archive_start: ledger.archive_start,
    updated_at: ledger.updated_at || null,
    weekdays: weekdays.length,
    settled_days: settled,
    complete_days: complete,
    closed_days: noIndex,
    settled_short_days: shortDays,
    not_settled_days: weekdays.length - settled,
    /** every weekday from here to yesterday is complete or EDGAR was closed */
    complete_back_to: completeBackTo,
    index_form4_checked: index,
    held_checked: held,
  };
}

// ── status (token-gated diag probe `form4_completeness`) ─────────────────────

function statePath(env: Record<string, string | undefined> = process.env): string {
  return path.join(heartbeatDir(env), FORM4_COMPLETENESS_STATE_FILENAME);
}

export function writeCompletenessState(run: CompletenessRun, file = statePath()): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(run));
  fs.renameSync(tmp, file);
}

export function readCompletenessState(file = statePath()): CompletenessRun | null {
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")) as CompletenessRun;
  } catch (e) {
    console.error("[datacore] form4 completeness state unreadable:", e instanceof Error ? e.message : e);
    return null;
  }
}

function refillStatePath(env: Record<string, string | undefined> = process.env): string {
  return path.join(heartbeatDir(env), "voltrade_form4_refill.json");
}

export function writeRefillState(run: RefillRun, file = refillStatePath()): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(run));
  fs.renameSync(tmp, file);
}

export function readRefillState(file = refillStatePath()): RefillRun | null {
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")) as RefillRun;
  } catch (e) {
    console.error("[datacore] form4 refill state unreadable:", e instanceof Error ? e.message : e);
    return null;
  }
}

/** Counts only: no ticker, name, CIK, accession or URL. */
export function form4CompletenessStatus(env: Record<string, string | undefined> = process.env): Record<string, unknown> {
  return {
    probe: "form4_completeness",
    enabled: form4CompletenessEnabled(env),
    rule: `Nightly ${FORM4_COMPLETENESS_WINDOW_START_MIN / 60}:00-${FORM4_COMPLETENESS_WINDOW_END_MIN / 60}:00 ET: for each weekday of the last ${FORM4_COMPLETENESS_LOOKBACK_DAYS} days, EDGAR's daily form index (types 4 and 4/A, exact) minus the accessions the archive holds as parsed rows is fetched and appended with form + backfilledAt. At most ${FORM4_COMPLETENESS_MAX_PER_RUN} filings a night.`,
    truth_source: "https://www.sec.gov/Archives/edgar/daily-index/ form.<yyyymmdd>.idx",
    latest: readCompletenessState(statePath(env)),
    refill: {
      enabled: form4RefillEnabled(env),
      rule: `After the recent days, each night: the weekdays from ${FORM4_ARCHIVE_START} to the edge of the 7-day reach that the ledger has not settled, newest first, at most ${FORM4_REFILL_MAX_PER_RUN} filings a night. A day is settled when every missing filing was tried and none is left failed, or after ${FORM4_REFILL_MAX_ATTEMPTS} nights.`,
      ledger: ledgerSummary(readDayLedger(ledgerPath(env)), Date.now()),
      latest: readRefillState(refillStatePath(env)),
    },
  };
}

// ── scheduler ────────────────────────────────────────────────────────────────

/** One night's work: the recent days, then (when EDGAR did not refuse and
 *  the refill is on) the older days. Persists the probe state and the ledger. */
export async function runForm4Night(
  deps: CompletenessDeps & { refillMaxPerRun?: number },
  env: Record<string, string | undefined> = process.env,
): Promise<{ run: CompletenessRun; refill: RefillRun | null; detail: string; failed: boolean }> {
  const now = deps.now ?? Date.now;
  const run = await runForm4Completeness(deps);
  writeCompletenessState(run, statePath(env));
  // The ledger keeps every day's standing after it leaves the 7-day reach;
  // a day that left it incomplete is picked up by the older-day refill.
  const ledger = readDayLedger(ledgerPath(env));
  for (const d of run.days) if (d.status !== "error") noteRecentLedgerDay(ledger, d, now());
  writeDayLedger(ledger, ledgerPath(env));
  let refill: RefillRun | null = null;
  if (!run.refused && form4RefillEnabled(env)) {
    refill = await runForm4Refill({ ...deps, maxPerRun: deps.refillMaxPerRun, ledger });
    writeDayLedger(ledger, ledgerPath(env));
    writeRefillState(refill, refillStatePath(env));
  }
  return {
    run, refill,
    detail: refill ? `${run.detail} | ${refill.detail}` : run.detail,
    failed: run.status === "error" || refill?.status === "error",
  };
}

let booted = false;
let timer: ReturnType<typeof setInterval> | null = null;
const TICK_MS = 30 * 60_000;

/** 30-minute tick (unref'd; a boot never starts a pass by itself). One pass
 *  per ET night inside the window; a restart inside the window may run it
 *  again, which is harmless (the pass is idempotent). */
export function bootForm4Completeness(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastNight = "";
  const tick = async () => {
    if (running || !form4CompletenessEnabled(env)) return;
    const nowMs = Date.now();
    if (!inCompletenessWindow(nowMs)) return;
    const night = zonedParts(nowMs, ET).date;
    if (lastNight === night) return;
    lastNight = night;
    running = true;
    beat("form4_completeness_nightly", "start");
    try {
      const { detail, failed } = await runForm4Night({ fetch: fetch as unknown as CompletenessFetch }, env);
      beat("form4_completeness_nightly", "finish", failed ? "error" : "ok", detail);
    } catch (e) {
      beat("form4_completeness_nightly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  timer = setInterval(() => { void tick(); }, TICK_MS);
  timer.unref?.();
}

/** Test-only: stop the tick and allow a fresh boot. */
export function _resetForm4CompletenessForTests(): void {
  if (timer) clearInterval(timer);
  timer = null;
  booted = false;
}
