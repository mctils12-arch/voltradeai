// filingCoverage — what a filing-history page actually holds, in words
// ([PRODUCT] 2026-11-15).
//
// server/filingHistoryPage.ts answers every `*-history` route with an honest
// page: `window` (the UTC archive days asked for), `covered` (the days read
// IN FULL into this page), `truncated` (older days inside the window were
// left out by the page cap) and `next` (the query string that continues the
// walk). Until this module the /data filings pages read only `days` and
// `count`, so in a heavy EDGAR week the Form 4 page said "1,864 filings, last
// 30 days" over six days of filings, and the 13F page "160 filings, last 45
// days" over two (read live 2026-11-15). The pages now say what is loaded and
// can walk to the older days.
//
// Pure functions only: no fetch, no React. Days are UTC ARCHIVE days (a
// filing's own `filedAt` can be one day earlier than its archive day) unless
// the payload says `window_by: "filed_day"`.
//
// FILED-DAY PAGES ([PRODUCT] 2026-11-18). The Form 4 route also pages by the
// day a filing was FILED (`?filed_from=`; server/filedDayHistory.ts), and each
// filed day comes with EDGAR's own count for it and whether the archive holds
// all of it (`filed_days[]`). The Form 4 page reads that window since the
// nightly refill started putting months-old filings into the newest archive
// day. Two things differ from an archive-day page: consecutive pages do not
// abut (the server walks the filed days that hold rows, so a weekend lies
// between Monday's page and Friday's), and each day has a completeness
// statement the page must print.

export interface FilingHistoryMeta {
  days?: number | null;
  window?: { from?: string; to?: string } | null;
  covered?: { from?: string; to?: string } | null;
  truncated?: boolean;
  next?: string | null;
  count?: number;
  archive_start?: string;
  /** "filed_day" when the page was read by filing date; absent or "archive_day" otherwise */
  window_by?: string;
  /** filed-day pages only: one entry per filed day read into the page */
  filed_days?: unknown;
}

export interface FilingCoverage {
  /** what the days are: the day a row was archived, or the day the filing was filed */
  by: "archive_day" | "filed_day";
  /** the UTC days asked for (inclusive) */
  window: { from: string; to: string };
  /** the UTC archive days loaded in full (inclusive); covered.to === window.to */
  covered: { from: string; to: string };
  /** older days inside `window` are not loaded */
  truncated: boolean;
  /** query string that loads the next older page, or null */
  next: string | null;
  windowDays: number;
  coveredDays: number;
}

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const DAY_MS = 86_400_000;
const dayMs = (d: string): number => Date.parse(`${d}T00:00:00Z`);
const isDay = (s: unknown): s is string => typeof s === "string" && DAY_RE.test(s) && !Number.isNaN(dayMs(s));

/** Inclusive day count of [from, to]; 0 when the pair is not two ordered days. */
export function spanDays(from: string, to: string): number {
  if (!isDay(from) || !isDay(to)) return 0;
  const n = Math.round((dayMs(to) - dayMs(from)) / DAY_MS) + 1;
  return n > 0 ? n : 0;
}

/** The coverage a history payload states, or null when it states none (a
 *  server older than the contract, a warming-up body, a malformed window) —
 *  the page then claims no window at all rather than guessing one. */
export function readCoverage(meta: FilingHistoryMeta | null | undefined): FilingCoverage | null {
  const w = meta?.window, c = meta?.covered;
  if (!w || !c || !isDay(w.from) || !isDay(w.to) || !isDay(c.from) || !isDay(c.to)) return null;
  const windowDays = spanDays(w.from, w.to);
  const coveredDays = spanDays(c.from, c.to);
  if (!windowDays || !coveredDays) return null;
  const next = typeof meta?.next === "string" && meta.next.startsWith("?") ? meta.next : null;
  // truncated without a way to continue is still truncated: the words say so, the button does not render
  const truncated = meta?.truncated === true || coveredDays < windowDays && next !== null;
  const by = meta?.window_by === "filed_day" ? "filed_day" : "archive_day";
  return { by, window: { from: w.from, to: w.to }, covered: { from: c.from, to: c.to }, truncated, next, windowDays, coveredDays };
}

/** The filed day a filed-day `next` continues from (its `filed_to`), or null. */
function nextFiledTo(next: string | null): string | null {
  const m = /[?&]filed_to=(\d{4}-\d{2}-\d{2})(?:&|$)/.exec(next ?? "");
  return m && isDay(m[1]) ? m[1] : null;
}

/** Coverage after an older page (fetched with `have.next`) is appended: the
 *  loaded run of days now reaches the older page's `covered.from`. An older
 *  page that does not continue the run (wrong window, no coverage) changes
 *  nothing except that the walk stops — never a gap presented as loaded.
 *
 *  What "continues" means depends on the window. Archive days: the older page
 *  ends on the day before the loaded run starts. Filed days: the older page
 *  ends on exactly the day `have.next` asked for, earlier than the loaded run
 *  (the server skipped the days between because the archive holds no filing
 *  filed on them: read live 2026-11-18, a page covering Mon 11-16 → Wed 11-18
 *  continued at `filed_to=2026-11-13`, the Friday). */
export function extendCoverage(have: FilingCoverage, older: FilingHistoryMeta | null | undefined): FilingCoverage {
  const o = readCoverage(older);
  const stop: FilingCoverage = { ...have, next: null };
  if (!o) return stop;
  if (o.by !== have.by) return stop;
  if (have.by === "filed_day") {
    const asked = nextFiledTo(have.next);
    if (!asked || o.covered.to !== asked || dayMs(asked) >= dayMs(have.covered.from)) return stop;
  } else if (dayMs(o.covered.to) !== dayMs(have.covered.from) - DAY_MS) {
    // contiguous: the older page must end on the day before the loaded run starts
    return stop;
  }
  if (dayMs(o.covered.from) < dayMs(have.window.from)) return stop;
  const covered = { from: o.covered.from, to: have.covered.to };
  const coveredDays = spanDays(covered.from, covered.to);
  const truncated = o.truncated && coveredDays < have.windowDays;
  return { by: have.by, window: have.window, covered, truncated, next: truncated ? o.next : null, windowDays: have.windowDays, coveredDays };
}

/** Rows of an older page appended to the loaded rows, one row per accession
 *  (the loaded, newer copy wins). Order is kept: loaded first, then older. */
export function appendOlderFilings<T extends { accession: string }>(have: readonly T[], older: readonly T[] | null | undefined): T[] {
  const seen = new Set(have.map((f) => f.accession));
  const out = have.slice();
  for (const f of older || []) {
    if (!f || typeof f.accession !== "string" || seen.has(f.accession)) continue;
    seen.add(f.accession);
    out.push(f);
  }
  return out;
}

const dayWord = (n: number): string => (n === 1 ? "day" : "days");

/** The header's words for what is loaded, e.g.
 *   "2026-10-17 → 2026-11-15 · all 30 days"
 *   "2026-11-10 → 2026-11-15 · 6 of the 30 days asked for — older days not loaded"
 *  Empty string when the payload stated no coverage. */
export function coverageWords(c: FilingCoverage | null): string {
  if (!c) return "";
  const range = c.covered.from === c.covered.to ? c.covered.from : `${c.covered.from} → ${c.covered.to}`;
  if (!c.truncated && c.coveredDays >= c.windowDays) {
    return `${range} · ${c.windowDays === 1 ? "1 day" : `all ${c.windowDays} days`}`;
  }
  return `${range} · ${c.coveredDays} of the ${c.windowDays} ${dayWord(c.windowDays)} asked for — older days not loaded`;
}

/** The "load older" control's label, or null when there is nothing older to load. */
export function loadOlderLabel(c: FilingCoverage | null, loading: boolean): string | null {
  if (!c || !c.truncated || !c.next) return null;
  if (loading) return "Loading older days…";
  const left = c.windowDays - c.coveredDays;
  return `Load older days (${left} ${dayWord(left)} before ${c.covered.from} not loaded)`;
}

// ── filed days: each day against EDGAR's own count ───────────────────────────

/** One filed day as a filed-day page states it. `edgarIndex` / `complete` are
 *  null when the nightly completeness pass has not compared the day with
 *  EDGAR's daily form index: the archive then holds the 15-minute poll's
 *  sample of that day, and the page must not present it as the whole day. */
export interface FiledDayStatus {
  day: string;
  /** filings the archive holds for the day (what the page loaded) */
  filings: number;
  /** distinct filings EDGAR's daily index lists for the day; null = not checked */
  edgarIndex: number | null;
  /** the archive holds every one of them; null = not checked */
  complete: boolean | null;
}

const asCount = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? Math.floor(v) : null);

/** The filed days a payload lists, newest first, one entry per day. Entries
 *  that are not a day with a count are dropped; a payload with no list (an
 *  archive-day page, an older server) yields []. A `complete: true` with no
 *  index count is read as not checked: "all N" needs the N. */
export function readFiledDays(meta: FilingHistoryMeta | null | undefined): FiledDayStatus[] {
  const list = Array.isArray(meta?.filed_days) ? (meta?.filed_days as unknown[]) : [];
  const out: FiledDayStatus[] = [];
  const seen = new Set<string>();
  for (const raw of list) {
    const r = raw as { day?: unknown; filings?: unknown; edgar_index?: unknown; complete?: unknown } | null;
    const filings = asCount(r?.filings);
    if (!r || !isDay(r.day) || filings === null || seen.has(r.day)) continue;
    seen.add(r.day);
    const edgarIndex = asCount(r.edgar_index);
    const complete = edgarIndex === null ? null : r.complete === true ? true : r.complete === false ? false : null;
    out.push({ day: r.day, filings, edgarIndex: complete === null ? null : edgarIndex, complete });
  }
  return out.sort((a, b) => (a.day < b.day ? 1 : a.day > b.day ? -1 : 0));
}

/** Filed days of an older page added to the loaded ones (the loaded entry wins), newest first. */
export function appendFiledDays(have: readonly FiledDayStatus[], older: readonly FiledDayStatus[]): FiledDayStatus[] {
  const seen = new Set(have.map((d) => d.day));
  return have.concat(older.filter((d) => !seen.has(d.day))).sort((a, b) => (a.day < b.day ? 1 : a.day > b.day ? -1 : 0));
}

export type FiledDayState = "complete" | "partial" | "sample";
export const filedDayState = (d: FiledDayStatus): FiledDayState =>
  d.complete === true ? "complete" : d.complete === false ? "partial" : "sample";

const fmt = (v: number): string => v.toLocaleString("en-US");

/** One filed day's completeness in words, e.g.
 *   "complete · all 1,047 on EDGAR's index"
 *   "partial · 512 of the 515 on EDGAR's index"
 *   "sample · not yet checked against EDGAR's index" */
export function filedDayWords(d: FiledDayStatus): string {
  if (d.complete === true && d.edgarIndex !== null) return `complete · all ${fmt(d.edgarIndex)} on EDGAR's index`;
  if (d.complete === false && d.edgarIndex !== null) return `partial · ${fmt(d.filings)} of the ${fmt(d.edgarIndex)} on EDGAR's index`;
  return "sample · not yet checked against EDGAR's index";
}

/** The header's one-line count over the loaded filed days, e.g.
 *   "2 of 3 filed days complete against EDGAR's index · 1 not yet checked"
 *  Empty string when no filed day is loaded. */
export function filedDaysSummary(days: readonly FiledDayStatus[]): string {
  if (!days.length) return "";
  const complete = days.filter((d) => d.complete === true).length;
  const partial = days.filter((d) => d.complete === false).length;
  const sample = days.length - complete - partial;
  const head = complete === days.length
    ? `${days.length === 1 ? "the 1 filed day is" : `all ${days.length} filed days are`} complete against EDGAR's index`
    : `${complete} of ${days.length} filed ${dayWord(days.length)} complete against EDGAR's index`;
  const rest = [partial ? `${partial} partial` : "", sample ? `${sample} not yet checked` : ""].filter(Boolean);
  return [head, ...rest].join(" · ");
}

/** The first page's query for a filed-day window of `days` UTC days ending
 *  today: `filed_from` alone, so the server's own clock sets the last day. */
export function filedDayFirstQuery(nowMs: number, days: number): string {
  const span = Math.max(1, Math.floor(days));
  const from = new Date(nowMs - (span - 1) * DAY_MS).toISOString().slice(0, 10);
  return `?filed_from=${from}`;
}
