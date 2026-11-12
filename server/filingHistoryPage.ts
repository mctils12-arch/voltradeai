/**
 * filingHistoryPage.ts — one honest windowed, paged reader for every
 * append-only per-UTC-day JSONL filing archive under datacore_archive/
 * (Form 4 `filings/`, 13F `filings13f/`, …).
 *
 * WHY THIS EXISTS (2026-11-12, scheduled-routine [PRODUCT] session): the
 * `*-history` routes and their keyed /api/v1 mirrors were built as
 * "newest N filings within the last `days` days", echoing `days` back
 * unmodified. In an EDGAR-heavy season that is a silent lie: live on
 * production, `/api/data/insider/history?days=90` (and the SOLD
 * `/api/v1/data/insider-history?days=90`) returned 2,000 Form 4 filings
 * spanning 2026-11-04..11-10 — five archive days, the oldest cut partway —
 * labelled `days: 90`, `count: 2000`, with no marker anywhere. 13F did the
 * same at its 500 cap (two days labelled 120). sec8kEarnings.ts fixed its
 * own route the day before (v1.0.1350) with exactly this contract; this
 * module is that contract made generic so the sibling archives stop
 * re-implementing — and re-breaking — it one at a time.
 *
 * THE CONTRACT (identical to the earnings-language page, so a customer sees
 * ONE shape across roots):
 *   · an explicit [from, to] window of UTC ARCHIVE days (a filing's
 *     `filedAt` can be one day earlier than its archive day — filed late in
 *     the US evening, archived on the next UTC day);
 *   · whole days are atomic: a day's file is appended as a unit, so a day is
 *     either fully in the page or not at all — the page stops BEFORE the day
 *     that would push it over the cap, except the newest day, which is
 *     always returned whole (the page must always make progress);
 *   · `covered` names the days actually read in full, `truncated` says
 *     whether older days inside the window were left out, `next` is the
 *     query string that continues the walk with neither gaps nor
 *     duplicates;
 *   · legacy `?days=N` still works and means [today-(N-1), today]; `from`/
 *     `to` take precedence; `limit` sets the page cap (≤ the archive's own
 *     max). Bad input never throws — it falls back to a default and the
 *     effective window is echoed, so nothing is silent.
 *
 * Research plumbing + product surface only: nothing here is read by any
 * trading path.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";

export interface FilingHistoryArchive<T extends { accession: string; filedAt?: string | null }> {
  /** Directory holding `YYYY-MM-DD.jsonl[.gz]` day files. */
  dir: string;
  /** Oldest archive day (the poll loop has appended every filing since). */
  archiveStart: string;
  /** Default and maximum filings per page (the archive's pre-existing cap, now reported instead of hidden). */
  maxFilings: number;
  /** Legacy `?days=` lookback cap. */
  maxDays: number;
  /** Default legacy lookback when nothing is sent. */
  defaultDays: number;
  /** Widest [from, to] span scanned in one request — bounds the day loop on a sparse archive. */
  maxSpanDays: number;
  /** Optional per-day projection applied to a day's parsed rows BEFORE dedup/cap
   *  (e.g. 13F's trimHoldings). Must keep `accession`. */
  project?: (row: T) => T;
}

export interface FilingHistoryWindow {
  /** inclusive UTC archive day, YYYY-MM-DD */
  from: string;
  /** inclusive UTC archive day, YYYY-MM-DD */
  to: string;
  /** page cap (soft by at most one day, see the module comment) */
  maxFilings: number;
  /** the legacy `?days=` value when the window came from it (echoed for compat), else null */
  days: number | null;
}

export interface FilingHistoryPage<T> {
  filings: T[];
  window: { from: string; to: string };
  /** archive days read IN FULL into this page (newest = `to`) */
  covered: { from: string; to: string };
  /** true when older days inside `window` were NOT read because the page cap
   *  would have been exceeded — continue with `?from=<window.from>&to=<next_to>` */
  truncated: boolean;
  next_to: string | null;
  days_scanned: number;
  max_filings: number;
  /** archive lines in the covered days that did not parse as a filing (a
   *  torn line from a kill mid-append) — reported, never silently dropped */
  malformed_lines: number;
}

/** The route-level object both the RAW preview route and its keyed v1 mirror serve. */
export interface FilingHistoryResponse<T> {
  days: number | null;
  window: { from: string; to: string };
  covered: { from: string; to: string };
  truncated: boolean;
  /** query string continuing a truncated page, or null */
  next: string | null;
  max_filings: number;
  days_scanned: number;
  archive_start: string;
  malformed_lines: number;
  /** the TRUE row count of THIS page (after the live merge) */
  count: number;
  filings: T[];
}

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const DAY_MS = 86_400_000;
export const utcDay = (ms: number): string => new Date(ms).toISOString().slice(0, 10);
const dayMs = (day: string): number => Date.parse(`${day}T00:00:00Z`);
const isValidDay = (s: unknown): s is string =>
  typeof s === "string" && DAY_RE.test(s) && !Number.isNaN(dayMs(s)) && utcDay(dayMs(s)) === s;

function queryStr(query: Record<string, unknown>, k: string): string | undefined {
  const v = query[k];
  return typeof v === "string" ? v.trim() : Array.isArray(v) && typeof v[0] === "string" ? String(v[0]).trim() : undefined;
}

/** Turns a route's query object into a validated window for `archive`.
 *  Legacy `?days=N` (1..maxDays, default defaultDays) means [today-(N-1),
 *  today]; `from`/`to` (YYYY-MM-DD UTC archive days) take precedence over
 *  `days`; `to` is clamped to today, `from` to the archive start and to `to`;
 *  the span is bounded to maxSpanDays by raising `from`. `limit` 1..maxFilings
 *  (default maxFilings). Never throws on bad input. */
export function parseFilingHistoryQuery<T extends { accession: string }>(
  query: Record<string, unknown>,
  archive: FilingHistoryArchive<T>,
  nowMs = Date.now(),
): FilingHistoryWindow {
  const today = utcDay(nowMs);
  const toRaw = queryStr(query, "to");
  const fromRaw = queryStr(query, "from");
  let days: number | null = null;
  if (!isValidDay(fromRaw) && !isValidDay(toRaw)) {
    const parsed = parseInt(queryStr(query, "days") ?? "", 10);
    days = Math.min(archive.maxDays, Math.max(1, Number.isFinite(parsed) ? parsed : archive.defaultDays));
  }
  let to = isValidDay(toRaw) ? toRaw : today;
  if (dayMs(to) > dayMs(today)) to = today;
  let from: string;
  if (isValidDay(fromRaw)) from = fromRaw;
  else if (days !== null) from = utcDay(dayMs(to) - (days - 1) * DAY_MS);
  else from = utcDay(dayMs(to) - (archive.defaultDays - 1) * DAY_MS);
  if (dayMs(from) < dayMs(archive.archiveStart)) from = archive.archiveStart;
  if (dayMs(from) > dayMs(to)) from = to;
  const spanDays = Math.round((dayMs(to) - dayMs(from)) / DAY_MS) + 1;
  if (spanDays > archive.maxSpanDays) from = utcDay(dayMs(to) - (archive.maxSpanDays - 1) * DAY_MS);
  const limitParsed = parseInt(queryStr(query, "limit") ?? "", 10);
  const maxFilings = Math.min(archive.maxFilings, Math.max(1, Number.isFinite(limitParsed) ? limitParsed : archive.maxFilings));
  return { from, to, maxFilings, days };
}

/** One archive day's rows (plain or gzipped), with the count of lines that
 *  did not parse as a filing. A missing day is an empty day, not an error. */
export function readFilingArchiveDay<T extends { accession: string }>(dir: string, day: string): { rows: T[]; malformed: number } {
  const rows: T[] = [];
  let malformed = 0;
  for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    for (const line of text.split("\n")) {
      if (!line) continue;
      try {
        const f = JSON.parse(line);
        if (f && typeof f.accession === "string" && f.accession) rows.push(f as T); else malformed++;
      } catch { malformed++; } // a torn line (kill mid-append) is counted, never silently dropped
    }
  }
  return { rows, malformed };
}

/** Reads the archive over an explicit [from, to] window of UTC archive days,
 *  newest day first, whole days at a time, stopping BEFORE the day that would
 *  push the page over `maxFilings` (except the first day, which is always
 *  included whole). Deduped by accession (first-seen = newest day wins),
 *  sorted newest-first by filedAt. */
export function readFilingHistoryWindow<T extends { accession: string; filedAt?: string | null }>(
  win: FilingHistoryWindow,
  archive: FilingHistoryArchive<T>,
): FilingHistoryPage<T> {
  const out: T[] = [];
  const seen = new Set<string>();
  let truncated = false;
  let nextTo: string | null = null;
  let coveredFrom = win.to;
  let daysScanned = 0;
  let malformed = 0;
  const fromMs = dayMs(win.from);
  for (let ms = dayMs(win.to); ms >= fromMs; ms -= DAY_MS) {
    const day = utcDay(ms);
    const read = readFilingArchiveDay<T>(archive.dir, day);
    const projected = archive.project ? read.rows.map(archive.project) : read.rows;
    // one row per accession across the page: a day's own duplicates and rows an
    // earlier (newer) day already served are dropped; the cap counts unique rows
    const rows: T[] = [];
    const dayAccessions = new Set<string>();
    for (const f of projected) {
      if (seen.has(f.accession) || dayAccessions.has(f.accession)) continue;
      dayAccessions.add(f.accession);
      rows.push(f);
    }
    if (out.length > 0 && out.length + rows.length > win.maxFilings) {
      truncated = true;
      nextTo = day;
      break;
    }
    for (const f of rows) { seen.add(f.accession); out.push(f); }
    malformed += read.malformed;
    coveredFrom = day;
    daysScanned++;
  }
  out.sort((a, b) => String(b.filedAt || "").localeCompare(String(a.filedAt || "")));
  return {
    filings: out,
    window: { from: win.from, to: win.to },
    covered: { from: coveredFrom, to: win.to },
    truncated,
    next_to: nextTo,
    days_scanned: daysScanned,
    max_filings: win.maxFilings,
    malformed_lines: malformed,
  };
}

/** The query string a consumer sends to continue a truncated page — or null
 *  when the page already reached `window.from`. */
export function filingHistoryNextQuery<T>(page: FilingHistoryPage<T>, archive: { maxFilings: number }): string | null {
  if (!page.truncated || !page.next_to) return null;
  const parts = [`from=${page.window.from}`, `to=${page.next_to}`];
  if (page.max_filings !== archive.maxFilings) parts.push(`limit=${page.max_filings}`);
  return `?${parts.join("&")}`;
}

/** One page for both the RAW preview route and its keyed v1 mirror. Live
 *  (not-yet-archived) poll rows are merged in only when the page's `to` is
 *  today — a page from the past must not gain today's rows. `days` is echoed
 *  for consumers that sent it (null when the window came from from/to);
 *  `count` is the true row count of THIS page. */
export function buildFilingHistoryPage<T extends { accession: string; filedAt?: string | null }>(
  query: Record<string, unknown>,
  archive: FilingHistoryArchive<T>,
  live: readonly T[] | null | undefined,
  nowMs = Date.now(),
): FilingHistoryResponse<T> {
  const win = parseFilingHistoryQuery(query, archive, nowMs);
  const page = readFilingHistoryWindow(win, archive);
  let filings = page.filings;
  if (win.to === utcDay(nowMs) && live && live.length) {
    const seen = new Set(filings.map((f) => f.accession));
    const fresh = live.filter((f) => f.accession && !seen.has(f.accession)).map((f) => (archive.project ? archive.project(f) : f));
    filings = [...fresh, ...filings];
  }
  return {
    days: win.days,
    window: page.window,
    covered: page.covered,
    truncated: page.truncated,
    next: filingHistoryNextQuery(page, archive),
    max_filings: page.max_filings,
    days_scanned: page.days_scanned,
    archive_start: archive.archiveStart,
    malformed_lines: page.malformed_lines,
    count: filings.length,
    filings,
  };
}
