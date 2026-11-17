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

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { archiveFilings, parseForm4Xml, pickOwnershipXmlName, type Form4Filing } from "./edgarForm4";
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

/** One completeness pass over the recent weekdays, oldest first. Idempotent:
 *  a second pass over the same days finds the rows the first one appended. */
export async function runForm4Completeness(deps: CompletenessDeps): Promise<CompletenessRun> {
  const now = deps.now ?? Date.now;
  const delayMs = deps.delayMs ?? FORM4_COMPLETENESS_DELAY_MS;
  const maxPerRun = deps.maxPerRun ?? FORM4_COMPLETENESS_MAX_PER_RUN;
  const append = deps.append ?? archiveFilings;
  const dir = filingsDir(deps.baseDir);
  const startedMs = now();
  const days: DayCompleteness[] = [];
  let tried = 0;
  let consecutiveRefusals = 0;
  let refused = false;
  let capped = false;

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

    let batch: Form4Filing[] = [];
    const flush = () => {
      if (batch.length) rec.backfilled += append(batch, deps.baseDir, now());
      batch = [];
    };
    for (let i = 0; i < missing.length; i++) {
      if (refused || tried >= maxPerRun) {
        if (!refused) capped = true;
        rec.deferred = missing.length - i;
        break;
      }
      const f = missing[i];
      tried++;
      try {
        const base = filingDirUrl(f);
        const xmlName = pickOwnershipXmlName(JSON.parse(await getText(deps.fetch, `${base}index.json`, 12_000)));
        const parsed = xmlName ? parseForm4Xml(await getText(deps.fetch, `${base}${xmlName}`, 12_000)) : null;
        consecutiveRefusals = 0;
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
        consecutiveRefusals = status === 403 || status === 429 ? consecutiveRefusals + 1 : 0;
        if (consecutiveRefusals >= FORM4_COMPLETENESS_REFUSAL_STOP) refused = true;
      }
      if (delayMs > 0) await sleep(delayMs);
    }
    flush();
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

/** Counts only: no ticker, name, CIK, accession or URL. */
export function form4CompletenessStatus(env: Record<string, string | undefined> = process.env): Record<string, unknown> {
  return {
    probe: "form4_completeness",
    enabled: form4CompletenessEnabled(env),
    rule: `Nightly ${FORM4_COMPLETENESS_WINDOW_START_MIN / 60}:00-${FORM4_COMPLETENESS_WINDOW_END_MIN / 60}:00 ET: for each weekday of the last ${FORM4_COMPLETENESS_LOOKBACK_DAYS} days, EDGAR's daily form index (types 4 and 4/A, exact) minus the accessions the archive holds as parsed rows is fetched and appended with form + backfilledAt. At most ${FORM4_COMPLETENESS_MAX_PER_RUN} filings a night.`,
    truth_source: "https://www.sec.gov/Archives/edgar/daily-index/ form.<yyyymmdd>.idx",
    latest: readCompletenessState(statePath(env)),
  };
}

// ── scheduler ────────────────────────────────────────────────────────────────

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
      const run = await runForm4Completeness({ fetch: fetch as unknown as CompletenessFetch });
      writeCompletenessState(run, statePath(env));
      beat("form4_completeness_nightly", "finish", run.status === "error" ? "error" : "ok", run.detail);
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
