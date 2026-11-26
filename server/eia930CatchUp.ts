/**
 * eia930CatchUp.ts — after EIA stops publishing for a while, read back to the
 * last hour the archive already holds.
 *
 * WHY (2026-11-26, [REPAIR]): gridDemand.ts and gridGeneration.ts each ask EIA
 * for one page of the newest rows per respondent (48 hours of demand; 60 to
 * 120 hours of generation, by how many fuel types a respondent reports) and
 * archive what they have not seen. That is whole only while EIA never falls
 * further behind than one page. On 2026-11-26 EIA's hourly grid data had not
 * advanced for about 36 hours (measured by hand against api.eia.gov: the same
 * newest hours production held), and the NW demand series was 52 hours behind.
 * When EIA publishes the missing hours at once, a poller that reads one page
 * keeps the newest page and never asks for the hours beyond it: a hole in the
 * archive that nothing fills (gridGeneration has no backfill at all).
 *
 * THE RULE. A page is sorted newest first. If the OLDEST hour in a page has a
 * row the archive does not hold, the page did not reach the archive, so the
 * next page is read (same query, `offset` one page further), until
 *   - a page's oldest hour is wholly archived   -> "joined": nothing is missing;
 *   - EIA returns fewer rows than asked          -> "exhausted": nothing older exists;
 *   - a page cannot be read                      -> "failed": the caller archives
 *     NOTHING for this respondent this poll, because archiving the newer pages
 *     would make the next poll's first page look joined and hide the hole;
 *   - the page limit is reached                  -> "capped": what was read is
 *     archived and the caller logs the oldest hour read, loudly.
 * A respondent with no archived row yet is never read back: a first run keeps
 * one page, as before (history is the backfill's job, not the poll's).
 *
 * Cost in normal running: none. The first page's oldest hour is two or more
 * days old and already archived, so no second request is made.
 */

/** Extra pages per respondent per poll. Twelve respondents x eleven pages x a
 *  30 s timeout is about 67 minutes at worst, inside the two-hour poll. */
export const CATCH_UP_MAX_EXTRA_PAGES = 10;

export interface HourRow { period: string }

/** One page as the source returned it: the parsed rows, and how many rows the
 *  source sent before parsing (a short page means nothing older exists). */
export interface SourcePage<T> { rows: T[]; rawCount: number }

export type CatchUpOutcome = "first_failed" | "none" | "joined" | "exhausted" | "failed" | "capped";

export interface CatchUpResult<T> {
  /** The newest page: what the live cache is computed from. */
  first: T[];
  /** Rows of the pages read after the first. */
  older: T[];
  extraPages: number;
  outcome: CatchUpOutcome;
  /** Oldest hour read on any page, null when nothing was read. */
  oldestRead: string | null;
}

/** True when the page's oldest hour carries a row the archive does not hold. */
export function oldestHourUnseen<T extends HourRow>(rows: T[], isSeen: (row: T) => boolean): boolean {
  if (!rows.length) return false;
  const oldest = rows.reduce((mn, r) => (r.period < mn ? r.period : mn), rows[0].period);
  return rows.some((r) => r.period === oldest && !isSeen(r));
}

const oldestOf = <T extends HourRow>(rows: T[], prior: string | null): string | null =>
  rows.reduce<string | null>((mn, r) => (mn == null || r.period < mn ? r.period : mn), prior);

export async function readBackToArchive<T extends HourRow>(opts: {
  /** Reads the page starting `offset` rows from the newest; null when the
   *  request failed. */
  fetchPage: (offset: number) => Promise<SourcePage<T> | null>;
  pageLength: number;
  isSeen: (row: T) => boolean;
  /** False for a respondent the archive holds nothing for. */
  hasHistory: boolean;
  maxExtraPages?: number;
  /** Awaited before each extra request (the caller's call spacing). */
  pause?: () => Promise<void>;
}): Promise<CatchUpResult<T>> {
  const max = opts.maxExtraPages ?? CATCH_UP_MAX_EXTRA_PAGES;
  const firstPage = await opts.fetchPage(0);
  if (!firstPage) return { first: [], older: [], extraPages: 0, outcome: "first_failed", oldestRead: null };
  const first = firstPage.rows;
  let oldestRead = oldestOf(first, null);
  const done = (outcome: CatchUpOutcome, older: T[], extraPages: number): CatchUpResult<T> =>
    ({ first, older, extraPages, outcome, oldestRead });

  if (!opts.hasHistory || !oldestHourUnseen(first, opts.isSeen)) return done("none", [], 0);
  if (firstPage.rawCount < opts.pageLength) return done("exhausted", [], 0);

  const older: T[] = [];
  for (let page = 1; page <= max; page++) {
    if (opts.pause) await opts.pause();
    const next = await opts.fetchPage(page * opts.pageLength);
    if (!next) return done("failed", older, page - 1);
    older.push(...next.rows);
    oldestRead = oldestOf(next.rows, oldestRead);
    if (next.rows.length > 0 && !oldestHourUnseen(next.rows, opts.isSeen)) return done("joined", older, page);
    if (next.rawCount < opts.pageLength) return done("exhausted", older, page);
  }
  return done("capped", older, max);
}

/** What one poll did for the respondents it had to read back for; the stream's
 *  route serves it so a session can see a catch-up without the server's log. */
export interface CatchUpNote {
  respondent: string;
  outcome: CatchUpOutcome;
  extra_pages: number;
  oldest_read: string | null;
}

export interface PollState {
  at: number;
  /** Newest hour any respondent returned on this poll. */
  newest_period: string | null;
  /** Rows appended to the archive by this poll. */
  rows_archived: number;
  /** Respondents whose first page could not be read. */
  first_failed: string[];
  /** Only respondents that were read back (outcome other than "none"). */
  catch_up: CatchUpNote[];
}
