// filingHistory.ts — what a /data filing page actually covers, said honestly
// (2026-11-12 [PRODUCT]). The three EDGAR archive pages (Form 4, 13F-HR,
// 8-K earnings language) read their `*-history` routes with `?days=N` and
// printed "N filings, last 30 days" from the echoed `days` — while the
// server, since v1.0.1350/1366/1367, answers a WINDOWED, PAGED read whose
// page stops at the archive's filing cap: live on 2026-11-12 the Form 4 page
// held 1,800 filings from 7 of the 30 days it asked for, the 13F page 532
// filings from ONE day of 45 (deadline week). Premium presentation of a
// wrong range is the fault PREMIUM EXPERIENCE STANDARD names; this module is
// the one place the pages turn the route's contract (`window`, `covered`,
// `truncated`, `next`) into words and into the next page to load.
//
// Pure functions only — the hook that fetches lives in useFilingHistory.ts.

export interface DaySpan { from: string; to: string }

/** The shape every filing history route answers (server/filingHistoryPage.ts
 *  and sec8kEarnings.ts). Older servers lack `covered`: then `legacy` is true
 *  and the label falls back to the echoed `days`. */
export interface FilingHistoryPayload<T> {
  filings?: T[];
  count?: number;
  days?: number | null;
  window?: DaySpan;
  covered?: DaySpan;
  truncated?: boolean;
  next?: string | null;
  archive_start?: string;
  max_filings?: number;
}

export interface FilingCoverage {
  /** distinct filings loaded across every page read so far */
  count: number;
  /** archive days read in full: the oldest page's `covered.from` to the newest page's `covered.to` */
  covered: DaySpan | null;
  /** the window the first request asked for */
  window: DaySpan | null;
  daysCovered: number | null;
  daysRequested: number | null;
  /** older days inside the window are still not loaded (the newest page's `truncated`) */
  truncated: boolean;
  /** the query string that continues the walk, from the newest page read */
  next: string | null;
  /** the server predates the contract — no `covered` on the first page */
  legacy: boolean;
}

/** Inclusive calendar-day count between two YYYY-MM-DD strings; null when either is malformed. */
export function inclusiveDays(from: string | undefined, to: string | undefined): number | null {
  if (!from || !to) return null;
  const a = Date.parse(`${from}T00:00:00Z`);
  const b = Date.parse(`${to}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b) || b < a) return null;
  return Math.round((b - a) / 86_400_000) + 1;
}

/** Merge the pages read so far (newest first, in read order) into one filing
 *  list and one coverage statement. Filings are deduplicated on `accession`
 *  — the archive can hold the same accession in two day files (filed late
 *  one UTC evening, re-seen the next), and the API tells consumers to dedup. */
export function mergeFilingPages<T extends { accession: string }>(
  pages: FilingHistoryPayload<T>[],
  fallbackDays: number,
): { filings: T[]; coverage: FilingCoverage } {
  const seen = new Set<string>();
  const filings: T[] = [];
  for (const p of pages) {
    for (const f of p.filings ?? []) {
      if (!f || typeof f.accession !== "string") continue;
      if (seen.has(f.accession)) continue;
      seen.add(f.accession);
      filings.push(f);
    }
  }
  const first = pages[0];
  const last = pages[pages.length - 1];
  const legacy = !first || !first.covered || !first.covered.from || !first.covered.to;
  if (legacy) {
    const days = typeof first?.days === "number" && first.days > 0 ? first.days : fallbackDays;
    return {
      filings,
      coverage: { count: filings.length, covered: null, window: null, daysCovered: null, daysRequested: days, truncated: false, next: null, legacy: true },
    };
  }
  const window = first.window?.from && first.window?.to ? { from: first.window.from, to: first.window.to } : null;
  // the oldest page read that still names `covered` bounds the span
  let oldestFrom = first.covered!.from;
  for (const p of pages) if (p.covered?.from && p.covered.from < oldestFrom) oldestFrom = p.covered.from;
  const covered = { from: oldestFrom, to: first.covered!.to };
  const daysRequested = window ? inclusiveDays(window.from, window.to) : (typeof first.days === "number" ? first.days : null);
  return {
    filings,
    coverage: {
      count: filings.length,
      covered,
      window,
      daysCovered: inclusiveDays(covered.from, covered.to),
      daysRequested,
      truncated: Boolean(last?.truncated),
      next: typeof last?.next === "string" && last.next.length > 0 ? last.next : null,
      legacy: false,
    },
  };
}

/** The sub-line words for a coverage statement. `noun` is the page's own
 *  ("filings", "releases"). Examples:
 *    legacy     → "1,800 filings, last 30 days"
 *    complete   → "1,800 filings · 2026-10-14 → 2026-11-12 · all 30 days requested"
 *    truncated  → "1,800 filings · 2026-11-06 → 2026-11-12 · 7 of the 30 days requested — older days not loaded" */
export function coverageLabel(c: FilingCoverage | null, noun: string): string {
  if (!c) return `loading ${noun}…`;
  const n = `${c.count.toLocaleString()} ${noun}`;
  if (c.legacy || !c.covered) return `${n}, last ${c.daysRequested ?? "—"} days`;
  const span = c.covered.from === c.covered.to ? c.covered.to : `${c.covered.from} → ${c.covered.to}`;
  const req = c.daysRequested != null ? `${c.daysRequested} days requested` : "the days requested";
  if (c.truncated) {
    const got = c.daysCovered != null ? `${c.daysCovered} of the ${req}` : `part of ${req}`;
    return `${n} · ${span} · ${got} — older days not loaded`;
  }
  return `${n} · ${span} · all ${req}`;
}

/** The URL for the next (older) page: the first request's path with the
 *  server's own `next` query string; null when there is nothing older to
 *  load. `next` carries every parameter the walk needs (window, limit,
 *  text=0) — nothing is re-derived here. */
export function olderPageUrl(firstUrl: string, next: string | null | undefined): string | null {
  if (!next) return null;
  const q = firstUrl.indexOf("?");
  const base = q >= 0 ? firstUrl.slice(0, q) : firstUrl;
  return `${base}${next.startsWith("?") ? next : `?${next}`}`;
}
