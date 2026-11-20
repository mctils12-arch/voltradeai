/**
 * edgar13fFiledHistory.ts — the one reader behind /api/data/filings13f/history
 * and the keyed /api/v1/data/13f-holdings-history: the archive-day window
 * (server/filingHistoryPage.ts, unchanged) or, when the query carries
 * `filed_from` / `filed_to`, the filed-day window (server/filedDayHistory.ts).
 *
 * WHY ([PRODUCT] 2026-11-19, open_questions.md 2026-11-17 item (i)). Since
 * 2026-11-19 the nightly completeness pass (server/edgar13fCompleteness.ts)
 * recovers the 13F filings the 15-minute poll missed and stores each under
 * the night it was recovered. Measured on production 2026-11-20 00:20Z:
 * `/api/data/filings13f/history?days=2` returned 640 filings, 600 of them
 * filed on Monday 2026-11-16 (the quarterly deadline) and sitting under
 * archive day 2026-11-19. An archive-day window therefore no longer means
 * "filed in these days", and the pass will move July-November the same way.
 *
 * A filed-day page lists each filed day with the count EDGAR's daily form
 * index gives for it (13F-HR and 13F-HR/A, exact) and whether the archive
 * holds every one, read from the pass's per-day ledger. A day the pass has
 * not checked says so (`edgar_index: null`, `complete: null`): the archive
 * then holds the poll's sample of that day, measured at 27% on the
 * 2026-11-16 deadline and 100% on eight quiet days.
 *
 * This lives in its own module because the completeness pass imports
 * edgar13f.ts; the history reader needs both.
 */
import { filings13fHistoryArchive, filings13fHistoryPage, latest13FFilings, type Filing13F } from "./edgar13f";
import { read13FLedger, type Ledger13F } from "./edgar13fCompleteness";
import { buildFiledDayPage, parseFiledDayQuery, type FiledDayResponse, type FiledDayRow } from "./filedDayHistory";
import type { FilingHistoryResponse } from "./filingHistoryPage";

export interface Filings13FFiledDay extends FiledDayRow {
  /** distinct 13F-HR + 13F-HR/A accessions in EDGAR's daily form index for the day; null = not checked yet */
  edgar_index: number | null;
  /** the archive holds every one of them (by accession); null = not checked yet */
  complete: boolean | null;
  /** when the completeness pass last compared the day with EDGAR's index */
  checked_at: string | null;
}

export type Filings13FFiledHistory = Omit<FiledDayResponse<Filing13F>, "filed_days"> & {
  filed_days: Filings13FFiledDay[];
  /** filed days in this page the completeness pass has made complete */
  complete_days: number;
};

export type Filings13FArchiveHistory = FilingHistoryResponse<Filing13F> & { window_by: "archive_day" };

export function filings13fFiledDayStatus(day: FiledDayRow, ledger: Ledger13F): Filings13FFiledDay {
  const rec = ledger.days[day.day];
  if (!rec || rec.status !== "checked") return { ...day, edgar_index: null, complete: null, checked_at: null };
  return { ...day, edgar_index: rec.index_13f, complete: rec.complete, checked_at: rec.checked_at };
}

/** One page of the 13F archive: by filed day when `filed_from` or `filed_to`
 *  is a valid day, otherwise by archive day exactly as before. `topHoldings`
 *  is the per-filing holdings trim (25 on the RAW preview route,
 *  FOCUSED_MAX_HOLDINGS on the keyed mirror) and applies to both windows. A
 *  filed day is returned whole even over the page cap when it is the first
 *  in the page: the 2026-11-16 deadline is 1,712 filings against a cap of 500. */
export async function filings13fHistory(
  query: Record<string, unknown>,
  topHoldings: number,
  baseDir?: string,
  nowMs?: number,
  live: readonly Filing13F[] | null = latest13FFilings()?.filings ?? null,
  ledger?: Ledger13F,
): Promise<Filings13FFiledHistory | Filings13FArchiveHistory> {
  const archive = filings13fHistoryArchive(topHoldings, baseDir);
  const win = parseFiledDayQuery(query, archive, nowMs);
  if (!win) return { window_by: "archive_day", ...filings13fHistoryPage(query, topHoldings, baseDir, nowMs, live) };
  const page = await buildFiledDayPage(win, archive);
  const book = ledger ?? read13FLedger();
  const filedDays = page.filed_days.map((d) => filings13fFiledDayStatus(d, book));
  const { filings, ...head } = page;
  return { ...head, filed_days: filedDays, complete_days: filedDays.filter((d) => d.complete === true).length, filings };
}
