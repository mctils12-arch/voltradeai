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
// filing's own `filedAt` can be one day earlier than its archive day).

export interface FilingHistoryMeta {
  days?: number | null;
  window?: { from?: string; to?: string } | null;
  covered?: { from?: string; to?: string } | null;
  truncated?: boolean;
  next?: string | null;
  count?: number;
  archive_start?: string;
}

export interface FilingCoverage {
  /** the UTC archive days asked for (inclusive) */
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
  return { window: { from: w.from, to: w.to }, covered: { from: c.from, to: c.to }, truncated, next, windowDays, coveredDays };
}

/** Coverage after an older page (fetched with `have.next`) is appended: the
 *  loaded run of days now reaches the older page's `covered.from`. An older
 *  page that does not continue the run (wrong window, no coverage) changes
 *  nothing except that the walk stops — never a gap presented as loaded. */
export function extendCoverage(have: FilingCoverage, older: FilingHistoryMeta | null | undefined): FilingCoverage {
  const o = readCoverage(older);
  const stop: FilingCoverage = { ...have, next: null };
  if (!o) return stop;
  // contiguous: the older page must end on the day before the loaded run starts
  if (dayMs(o.covered.to) !== dayMs(have.covered.from) - DAY_MS) return stop;
  if (dayMs(o.covered.from) < dayMs(have.window.from)) return stop;
  const covered = { from: o.covered.from, to: have.covered.to };
  const coveredDays = spanDays(covered.from, covered.to);
  const truncated = o.truncated && coveredDays < have.windowDays;
  return { window: have.window, covered, truncated, next: truncated ? o.next : null, windowDays: have.windowDays, coveredDays };
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
