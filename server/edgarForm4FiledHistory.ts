/**
 * edgarForm4FiledHistory.ts — the one reader behind /api/data/insider/history
 * and the keyed /api/v1/data/insider-history: the archive-day window
 * (server/filingHistoryPage.ts, unchanged) or, when the query carries
 * `filed_from` / `filed_to`, the filed-day window (server/filedDayHistory.ts).
 *
 * A filed-day page lists each filed day with the count EDGAR's daily form
 * index gives for it and whether the nightly completeness pass has made the
 * day complete, read from that pass's per-day ledger
 * (server/edgarForm4Completeness.ts). A day the pass has not checked says so
 * (`edgar_index: null`, `complete: null`): the archive then holds the 15-minute
 * poll's sample of that day, measured at 12% to 61% on the days checked so far.
 *
 * This lives in its own module because the completeness pass imports
 * edgarForm4.ts; the history reader needs both.
 */
import { form4HistoryArchive, form4HistoryPage, latestForm4Filings, type Form4Filing } from "./edgarForm4";
import { readDayLedger, type DayLedger } from "./edgarForm4Completeness";
import { buildFiledDayPage, parseFiledDayQuery, type FiledDayResponse, type FiledDayRow } from "./filedDayHistory";
import type { FilingHistoryResponse } from "./filingHistoryPage";

export interface Form4FiledDay extends FiledDayRow {
  /** distinct Form 4 + 4/A accessions in EDGAR's daily form index for the day; null = not checked yet */
  edgar_index: number | null;
  /** the archive holds every one of them (by accession); null = not checked yet */
  complete: boolean | null;
  /** when the completeness pass last compared the day with EDGAR's index */
  checked_at: string | null;
}

export type Form4FiledHistory = Omit<FiledDayResponse<Form4Filing>, "filed_days"> & {
  filed_days: Form4FiledDay[];
  /** filed days in this page the completeness pass has made complete */
  complete_days: number;
};

export type Form4ArchiveHistory = FilingHistoryResponse<Form4Filing> & { window_by: "archive_day" };

/** A Form 4 row is a record when the ownership document named an issuer. The
 *  poll's `type=4` prefix match archived other forms as rows with none. */
export const isForm4Record = (row: Form4Filing): boolean => Boolean(row.issuerCik);

export function form4FiledDayStatus(day: FiledDayRow, ledger: DayLedger): Form4FiledDay {
  const rec = ledger.days[day.day];
  if (!rec || rec.status !== "checked") return { ...day, edgar_index: null, complete: null, checked_at: null };
  return { ...day, edgar_index: rec.index_form4, complete: rec.complete, checked_at: rec.checked_at };
}

/** One page of the Form 4 archive: by filed day when `filed_from` or
 *  `filed_to` is a valid day, otherwise by archive day exactly as before. */
export async function form4History(
  query: Record<string, unknown>,
  baseDir?: string,
  nowMs?: number,
  live: readonly Form4Filing[] | null = latestForm4Filings()?.filings ?? null,
  ledger?: DayLedger,
): Promise<Form4FiledHistory | Form4ArchiveHistory> {
  const archive = form4HistoryArchive(baseDir);
  const win = parseFiledDayQuery(query, archive, nowMs);
  if (!win) return { window_by: "archive_day", ...form4HistoryPage(query, baseDir, nowMs, live) };
  const page = await buildFiledDayPage(win, archive, { isRecord: isForm4Record });
  const book = ledger ?? readDayLedger();
  const filedDays = page.filed_days.map((d) => form4FiledDayStatus(d, book));
  const { filings, ...head } = page;
  return { ...head, filed_days: filedDays, complete_days: filedDays.filter((d) => d.complete === true).length, filings };
}
