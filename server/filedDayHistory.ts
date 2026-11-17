/**
 * filedDayHistory.ts — a filing archive read by the day a filing was FILED,
 * not the day it was archived.
 *
 * WHY ([PRODUCT] 2026-11-17, open_questions.md 2026-11-17 "(c) FILED-DATE
 * WINDOWS FOR CUSTOMERS"). The archive is append-only day files named by the
 * UTC day a row was WRITTEN. Until 2026-11-17 a Form 4's archive day was its
 * filed day or the day after, so the archive-day window of
 * server/filingHistoryPage.ts was a fair stand-in for "filed in these days".
 * The nightly completeness pass (server/edgarForm4Completeness.ts) ended
 * that: a filing the poll missed is recovered later and sits in the day file
 * of the night it was recovered. Measured on production 2026-11-17 18:20Z:
 * `/api/data/insider/history?from=2026-11-13&to=2026-11-15` returned 411 of
 * the 675 Form 4s EDGAR lists for Friday 2026-11-13 (61%) beside 189 rows
 * that are not Form 4s at all; the other 262 sit under archive day
 * 2026-11-17. Once the pass refills July-November, an archive-day window
 * over September returns the poll's sample and a November archive day holds
 * thousands of older filings.
 *
 * WHAT. A small in-memory index (archive day -> the filed days its rows
 * carry, refreshed by file size and mtime, one day file per event-loop
 * turn) answers "which day files hold rows filed on D". A filed-day window
 * is then read newest filed day first, whole FILED days at a time, under the
 * same page contract as the archive-day reader: `covered`, `truncated`,
 * `next`. Each filed day in the page is listed with its row count, so a
 * consumer can check a day against an outside count.
 *
 * Rows the archive's `isRecord` rejects (for Form 4: no issuer — the poll's
 * prefix match archived 424B2 / 485BPOS documents as empty Form 4 rows) are
 * left out of a filed-day page and COUNTED per day (`omitted`), never
 * silently dropped. The archive-day reader is unchanged and still returns
 * every row as written.
 *
 * Not served here: rows with no `filedAt`, and the live poll cache (a row
 * reaches a filed-day page once it is in a day file, which the poll does in
 * the same refresh).
 *
 * Research plumbing + product surface only: nothing here is read by any
 * trading path.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { readFilingArchiveDay, utcDay, type FilingHistoryArchive } from "./filingHistoryPage";

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const DAY_FILE_RE = /^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/;
const FILED_AT_RE = /"filedAt"\s*:\s*"(\d{4}-\d{2}-\d{2})/g;
const DAY_MS = 86_400_000;
/** Parsed archive days kept while one page is built. A Form 4 refill night
 *  is about 3,000 rows; a page needs two to four day files per filed day. */
const PAGE_FILE_CACHE = 8;

const dayMs = (day: string): number => Date.parse(`${day}T00:00:00Z`);
const isValidDay = (s: unknown): s is string =>
  typeof s === "string" && DAY_RE.test(s) && !Number.isNaN(dayMs(s)) && utcDay(dayMs(s)) === s;
const yieldToLoop = () => new Promise<void>((res) => { setImmediate(res); });

function queryStr(query: Record<string, unknown>, k: string): string | undefined {
  const v = query[k];
  return typeof v === "string" ? v.trim() : Array.isArray(v) && typeof v[0] === "string" ? String(v[0]).trim() : undefined;
}

// ── the query ────────────────────────────────────────────────────────────────

export interface FiledDayWindow {
  /** inclusive filed day, YYYY-MM-DD */
  from: string;
  /** inclusive filed day, YYYY-MM-DD */
  to: string;
  /** page cap (soft by at most one filed day) */
  maxFilings: number;
}

/** A filed-day window from a route's query, or null when the query asks for
 *  none (neither `filed_from` nor `filed_to` is a valid day) — the caller then
 *  serves the archive-day window. `filed_to` alone means the archive's
 *  default span ending there; `filed_from` alone runs to today. `filed_to`
 *  is clamped to today, `filed_from` to `filed_to`, the span to the archive's
 *  maxSpanDays by raising `filed_from`. There is no clamp at the archive
 *  start: a filing can be filed before the day it was first archived. */
export function parseFiledDayQuery<T extends { accession: string }>(
  query: Record<string, unknown>,
  archive: FilingHistoryArchive<T>,
  nowMs = Date.now(),
): FiledDayWindow | null {
  const fromRaw = queryStr(query, "filed_from");
  const toRaw = queryStr(query, "filed_to");
  if (!isValidDay(fromRaw) && !isValidDay(toRaw)) return null;
  const today = utcDay(nowMs);
  let to = isValidDay(toRaw) ? toRaw : today;
  if (dayMs(to) > dayMs(today)) to = today;
  let from = isValidDay(fromRaw) ? fromRaw : utcDay(dayMs(to) - (archive.defaultDays - 1) * DAY_MS);
  if (dayMs(from) > dayMs(to)) from = to;
  const spanDays = Math.round((dayMs(to) - dayMs(from)) / DAY_MS) + 1;
  if (spanDays > archive.maxSpanDays) from = utcDay(dayMs(to) - (archive.maxSpanDays - 1) * DAY_MS);
  const limitParsed = parseInt(queryStr(query, "limit") ?? "", 10);
  const maxFilings = Math.min(archive.maxFilings, Math.max(1, Number.isFinite(limitParsed) ? limitParsed : archive.maxFilings));
  return { from, to, maxFilings };
}

// ── the index: archive day -> filed days its rows carry ──────────────────────

interface IndexedDay { sig: string; filedDays: Set<string> }
interface DirIndex { days: Map<string, IndexedDay>; refreshing: Promise<void> | null; scans: number }
const indexes = new Map<string, DirIndex>();

/** Size and mtime of a day's plain and gzipped files; "" when neither exists. */
function daySignature(dir: string, day: string): string {
  const parts: string[] = [];
  for (const name of [`${day}.jsonl`, `${day}.jsonl.gz`]) {
    try {
      const st = fs.statSync(path.join(dir, name));
      parts.push(`${name}:${st.size}:${st.mtimeMs}`);
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") throw e;
    }
  }
  return parts.join("|");
}

/** The filed days named by one archive day's rows, read from the raw text
 *  (no row is parsed: the index needs only the dates). */
export function filedDaysInArchiveDay(dir: string, day: string): Set<string> {
  const found = new Set<string>();
  for (const name of [`${day}.jsonl`, `${day}.jsonl.gz`]) {
    let text: string;
    try {
      const buf = fs.readFileSync(path.join(dir, name));
      text = name.endsWith(".gz") ? zlib.gunzipSync(buf).toString("utf8") : buf.toString("utf8");
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code === "ENOENT") continue;
      throw e;
    }
    FILED_AT_RE.lastIndex = 0;
    for (let m = FILED_AT_RE.exec(text); m; m = FILED_AT_RE.exec(text)) found.add(m[1]);
  }
  return found;
}

async function refreshInto(dir: string, idx: DirIndex): Promise<void> {
  let names: string[] = [];
  try { names = fs.readdirSync(dir); } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") throw e; // no archive yet = an empty index
  }
  const present = new Set<string>();
  for (const n of names) { const m = DAY_FILE_RE.exec(n); if (m) present.add(m[1]); }
  for (const day of Array.from(idx.days.keys())) if (!present.has(day)) idx.days.delete(day);
  for (const day of Array.from(present)) {
    const sig = daySignature(dir, day);
    if (idx.days.get(day)?.sig === sig) continue;
    idx.days.set(day, { sig, filedDays: filedDaysInArchiveDay(dir, day) });
    idx.scans++;
    await yieldToLoop(); // months of day files; this process also runs the trading loop
  }
}

/** Brings the index for `dir` up to date and returns it. Only day files whose
 *  size or mtime changed are read again; concurrent callers share one refresh. */
export async function refreshFiledDayIndex(dir: string): Promise<Map<string, IndexedDay>> {
  let idx = indexes.get(dir);
  if (!idx) { idx = { days: new Map(), refreshing: null, scans: 0 }; indexes.set(dir, idx); }
  if (!idx.refreshing) {
    const own = idx;
    own.refreshing = refreshInto(dir, own).finally(() => { own.refreshing = null; });
  }
  await idx.refreshing;
  return idx.days;
}

/** How many day files the index has read for `dir` since it was created. */
export function filedDayIndexScans(dir: string): number {
  return indexes.get(dir)?.scans ?? 0;
}

export function _resetFiledDayIndexForTests(): void {
  indexes.clear();
}

// ── the page ─────────────────────────────────────────────────────────────────

export interface FiledDayRow {
  /** the filed day */
  day: string;
  /** distinct record rows the archive holds for it (what this page returns) */
  filings: number;
  /** distinct rows filed that day that are not records (see the module comment) */
  omitted: number;
}

export interface FiledDayOptions<T> {
  /** false = the row is left out of the page and counted as omitted. Default: every row is a record. */
  isRecord?: (row: T) => boolean;
}

export interface FiledDayResponse<T> {
  window_by: "filed_day";
  /** always null on a filed-day page (the legacy `?days=` lookback is an archive-day window) */
  days: null;
  window: { from: string; to: string };
  /** filed days read IN FULL into this page */
  covered: { from: string; to: string };
  /** true when older filed days inside `window` were left for the next page */
  truncated: boolean;
  /** query string continuing a truncated page, or null */
  next: string | null;
  max_filings: number;
  /** filed days with at least one row that were read into this page */
  days_scanned: number;
  archive_start: string;
  /** unparseable lines in the day files this page read (a torn line from a kill mid-append) */
  malformed_lines: number;
  /** day files opened to build this page */
  archive_days_read: number;
  /** rows left out across the page because they are not records */
  omitted: number;
  /** one entry per filed day in the page, newest first */
  filed_days: FiledDayRow[];
  count: number;
  filings: T[];
}

/** Reads the archive over a window of FILED days, newest first, whole filed
 *  days at a time, stopping before the filed day that would push the page
 *  over the cap (the first is always returned whole). One row per accession;
 *  a record row is kept over a non-record row with the same accession. */
export async function buildFiledDayPage<T extends { accession: string; filedAt?: string | null }>(
  win: FiledDayWindow,
  archive: FilingHistoryArchive<T>,
  opts: FiledDayOptions<T> = {},
): Promise<FiledDayResponse<T>> {
  const isRecord = opts.isRecord ?? (() => true);
  const index = await refreshFiledDayIndex(archive.dir);

  // filed day -> the archive days that hold rows filed on it
  const holders = new Map<string, string[]>();
  index.forEach((entry, archiveDay) => {
    entry.filedDays.forEach((filed) => {
      if (filed < win.from || filed > win.to) return;
      const list = holders.get(filed);
      if (list) list.push(archiveDay); else holders.set(filed, [archiveDay]);
    });
  });
  const filedDays = Array.from(holders.keys()).sort().reverse();

  const parsed = new Map<string, { rows: T[]; malformed: number }>();
  const opened = new Set<string>();
  let malformed = 0;
  const load = async (archiveDay: string): Promise<T[]> => {
    const hit = parsed.get(archiveDay);
    if (hit) return hit.rows;
    const read = readFilingArchiveDay<T>(archive.dir, archiveDay);
    if (!opened.has(archiveDay)) { opened.add(archiveDay); malformed += read.malformed; }
    if (parsed.size >= PAGE_FILE_CACHE) parsed.delete(parsed.keys().next().value as string);
    parsed.set(archiveDay, read);
    await yieldToLoop();
    return read.rows;
  };

  const out: T[] = [];
  const perDay: FiledDayRow[] = [];
  let truncated = false;
  let nextTo: string | null = null;
  let coveredFrom = win.from;
  let omittedTotal = 0;
  for (const filed of filedDays) {
    const byAccession = new Map<string, T>();
    for (const archiveDay of (holders.get(filed) as string[]).sort()) {
      for (const row of await load(archiveDay)) {
        if (typeof row.filedAt !== "string" || row.filedAt.slice(0, 10) !== filed) continue;
        const prior = byAccession.get(row.accession);
        if (!prior || (!isRecord(prior) && isRecord(row))) byAccession.set(row.accession, row);
      }
    }
    const rows: T[] = [];
    let omitted = 0;
    byAccession.forEach((row) => { if (isRecord(row)) rows.push(row); else omitted++; });
    if (out.length > 0 && out.length + rows.length > win.maxFilings) {
      truncated = true;
      nextTo = filed;
      break;
    }
    for (const row of rows) out.push(archive.project ? archive.project(row) : row);
    perDay.push({ day: filed, filings: rows.length, omitted });
    omittedTotal += omitted;
    coveredFrom = filed;
  }
  if (!truncated) coveredFrom = win.from;

  let next: string | null = null;
  if (truncated && nextTo) {
    const parts = [`filed_from=${win.from}`, `filed_to=${nextTo}`];
    if (win.maxFilings !== archive.maxFilings) parts.push(`limit=${win.maxFilings}`);
    next = `?${parts.join("&")}`;
  }
  return {
    window_by: "filed_day",
    days: null,
    window: { from: win.from, to: win.to },
    covered: { from: coveredFrom, to: win.to },
    truncated,
    next,
    max_filings: win.maxFilings,
    days_scanned: perDay.length,
    archive_start: archive.archiveStart,
    malformed_lines: malformed,
    archive_days_read: opened.size,
    omitted: omittedTotal,
    filed_days: perDay,
    count: out.length,
    filings: out,
  };
}
