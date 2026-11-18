// Filed-day groups for the Form 4 page ([PRODUCT] 2026-11-18, open_questions
// item (l) "the row cap hides most of a day").
//
// The page used to show the first 500 transaction rows of everything it had
// loaded. Read live 2026-11-18: three filed days, 1,765 filings, 4,397
// transaction rows, so the 500 rows ended partway through the second day and
// the third could only be reached through a filter. Now every loaded filed
// day is a line that opens and closes, and an open day shows its rows a page
// at a time. Nothing is hidden behind a cap: what is not on screen is one
// tap on a line that says how many rows it holds.
//
// This file is the rules (pure, tested); pages/filings.tsx only renders them.
import type { FiledDayStatus } from "./filingCoverage";

/** rows an open day shows before "show more" is asked for */
export const DAY_ROWS_FIRST = 25;
/** rows each "show more" adds */
export const DAY_ROWS_STEP = 200;

export interface FiledDayGroup<R> {
  /** YYYY-MM-DD, or "" for rows whose filing states no filed time */
  day: string;
  /** the day's rows (after the page's filter), in the order given */
  rows: R[];
  /** the day against EDGAR's index, when the server stated it */
  status: FiledDayStatus | null;
}

/** One group per filed day, newest first: every day the server listed AND every
 *  day a row was filed on. A listed day with no row left (a filter matched
 *  nothing, or its filings hold no transaction) keeps its line, so a day never
 *  disappears from the page. Rows with no filed day go last. */
export function groupByFiledDay<R>(
  rows: readonly R[],
  dayOf: (r: R) => string,
  filedDays: readonly FiledDayStatus[],
): FiledDayGroup<R>[] {
  const byDay = new Map<string, FiledDayGroup<R>>();
  for (const d of filedDays) if (!byDay.has(d.day)) byDay.set(d.day, { day: d.day, rows: [], status: d });
  for (const r of rows) {
    const day = dayOf(r) || "";
    let g = byDay.get(day);
    if (!g) { g = { day, rows: [], status: null }; byDay.set(day, g); }
    g.rows.push(r);
  }
  return Array.from(byDay.values()).sort((a, b) =>
    a.day === b.day ? 0 : a.day === "" ? 1 : b.day === "" ? -1 : a.day < b.day ? 1 : -1);
}

/** The day that is open before the visitor has touched anything: the newest
 *  one holding a row. null when no group holds a row. */
export function defaultOpenDay<R>(groups: readonly FiledDayGroup<R>[]): string | null {
  const g = groups.find((x) => x.rows.length > 0);
  return g ? g.day : null;
}

/** Whether a day is open: the visitor's own choice when there is one, else the default day. */
export function isDayOpen(chosen: ReadonlyMap<string, boolean>, day: string, defaultDay: string | null): boolean {
  const c = chosen.get(day);
  return c === undefined ? day === defaultDay : c;
}

/** Rows an open day shows after `steps` uses of "show more". */
export function rowsShown(total: number, steps: number): number {
  const n = DAY_ROWS_FIRST + Math.max(0, Math.floor(steps)) * DAY_ROWS_STEP;
  return Math.max(0, Math.min(Math.floor(total), n));
}

const fmt = (v: number): string => v.toLocaleString("en-US");

/** The "show more" control's words, or null when the whole day is on screen, e.g.
 *   "Show 200 more · 25 of 2,800 rows shown"
 *   "Show the last 12 · 25 of 37 rows shown" */
export function showMoreLabel(shown: number, total: number): string | null {
  const left = total - shown;
  if (left <= 0) return null;
  const head = left <= DAY_ROWS_STEP ? `Show the last ${fmt(left)}` : `Show ${fmt(DAY_ROWS_STEP)} more`;
  return `${head} · ${fmt(shown)} of ${fmt(total)} rows shown`;
}

/** A day line's row count in words. `filtered` = a filter other than "All" is on. */
export function dayRowsWords(rows: number, filtered: boolean): string {
  if (rows <= 0) return filtered ? "no rows match the filter" : "no transaction rows";
  const noun = rows === 1 ? "row" : "rows";
  return filtered ? `${fmt(rows)} matching ${noun}` : `${fmt(rows)} ${noun}`;
}
