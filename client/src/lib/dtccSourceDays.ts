// dtccSourceDays.ts — the one line the DTCC swaps page says about which
// vendor files the archive has read (server/dtccSwaps.ts SOURCE-DAY LEDGER,
// 2026-11-25, open_questions.md (an2)). Each dated vendor file holds one
// day's events and no later file repeats them, so "N events archived" is only
// as complete as the list of files read. The line states that list's counts
// as the server holds them: nothing is inferred here.

export interface DtccSourceDaysSummary {
  enabled: boolean;
  floor: string;
  through: string | null;
  expected: number;
  read: number;
  unavailable: number;
  pending: number;
  refilledDays: number;
  refilledRows: number;
  lastRefilled: { date: string; at: string; newRows: number } | null;
  /** Vendor days older than the floor, read before the vendor's two-year
   *  limit took them (server/dtccSwaps.ts THE EDGE). Absent on an older server. */
  edge?: { enabled: boolean; days: number; rows: number; oldest: string | null; newest: string | null };
}

const n = (x: number) => x.toLocaleString("en-US");
const days = (x: number) => `${n(x)} ${x === 1 ? "day" : "days"}`;

/** null when the server sent no summary or has not read a file yet. */
export function dtccSourceDaysLine(s: DtccSourceDaysSummary | null | undefined): string | null {
  if (!s || !s.through || !(s.expected > 0)) return null;
  const parts: string[] = [
    `Vendor files ${s.floor} to ${s.through}: ${n(s.read)} of ${days(s.expected)} read`,
  ];
  if (s.pending > 0) {
    parts.push(s.enabled
      ? `${days(s.pending)} not yet checked (two are read after each poll)`
      : `${days(s.pending)} not checked (the catch-up is switched off)`);
  }
  if (s.unavailable > 0) parts.push(`${days(s.unavailable)} the vendor did not serve`);
  if (s.refilledDays > 0) {
    parts.push(`${days(s.refilledDays)} had been missed and ${s.refilledDays === 1 ? "was" : "were"} read late (${n(s.refilledRows)} events added)`);
  } else if (s.pending === 0 && s.unavailable === 0) {
    parts.push("no day was missed");
  }
  return parts.join(" · ");
}

/** The second line: what was read at the far end of the vendor's window. The
 *  vendor stops serving a file two years after it is written; null when no
 *  such day has been read (or the server sent no edge block). */
export function dtccEdgeLine(s: DtccSourceDaysSummary | null | undefined): string | null {
  const e = s?.edge;
  if (!e || !(e.days > 0) || !e.oldest || !e.newest) return null;
  const span = e.oldest === e.newest ? e.oldest : `${e.oldest} to ${e.newest}`;
  return `The vendor stops serving a file two years after it is written: ${days(e.days)} at that limit read in time (${span}, ${n(e.rows)} events)`
    + (e.enabled ? "" : " · these reads are switched off now");
}
