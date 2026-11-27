// dtccNotional.ts — how the DTCC swaps page prints a notional and what it
// says about the ranking (server/dtccSwaps.ts DtccNotionalClass, 2026-11-27,
// open_questions.md (aq2)). The list ranks dollars only. The source writes a
// value at or above its dissemination cap as "250,000,000+"; the page prints
// it the same way. Everything the list leaves out is counted by the server
// and said here: nothing is inferred on the client.

export interface DtccNotionalMix {
  usd: number;
  usd_capped: number;
  other_currency: number;
  over_cap_unmarked: number;
  blank: number;
}

const n = (x: number) => x.toLocaleString("en-US");

/** One table cell. A capped value keeps the source's "+"; a blank is not a zero. */
export function dtccNotionalText(amount: number | null, currency: string, capped?: boolean): string {
  if (amount == null) return "— (not reported)";
  return `${currency} ${amount.toLocaleString("en-US", { maximumFractionDigits: 0 })}${capped ? "+" : ""}`;
}

/** null when the server sent no counts (an older server). */
export function dtccRankingLine(mix: DtccNotionalMix | null | undefined, capUsd?: number): string | null {
  if (!mix) return null;
  const ranked = mix.usd + mix.usd_capped;
  const parts: string[] = [`Ranked in US dollars: ${n(ranked)} USD ${ranked === 1 ? "event" : "events"} in this file`];
  if (mix.usd_capped > 0) {
    const cap = capUsd && capUsd > 0 ? `${n(capUsd)}+` : "a value ending in +";
    parts.push(`${n(mix.usd_capped)} at or above the source's cap, written ${cap}, listed first`);
  }
  const out: string[] = [];
  if (mix.other_currency > 0) out.push(`${n(mix.other_currency)} in other currencies (no exchange rate is applied)`);
  if (mix.over_cap_unmarked > 0) {
    out.push(`${n(mix.over_cap_unmarked)} unmarked ${mix.over_cap_unmarked === 1 ? "value" : "values"} above the cap (a reporter's placeholder)`);
  }
  if (out.length > 0) parts.push(`not in the list: ${out.join(", ")}`);
  if (mix.blank > 0) parts.push(`${n(mix.blank)} with no notional reported`);
  return parts.join(" · ");
}

/** The cap-mark record against the vendor days read (server/dtccCapMarks.ts,
 *  open_questions.md (bc1)). Absent on an older server. */
export interface DtccCapMarksSummary {
  enabled: boolean;
  days_read: number;
  recorded: number;
  owed: number;
  lost: number;
  lost_days: string[];
  capped_events: number;
}

const files = (x: number) => `${n(x)} ${x === 1 ? "file" : "files"}`;

/** What the archive knows about the cap mark on events it stored before the
 *  page could read it. null when the server sent no counts or has read no
 *  file. Says "all" only when nothing is owed and nothing was lost. */
export function dtccCapMarksLine(s: DtccCapMarksSummary | null | undefined): string | null {
  if (!s || !(s.days_read > 0)) return null;
  const complete = s.owed === 0 && s.lost === 0;
  const parts: string[] = [
    `Events archived before 2026-11-27 hold a blank where the source wrote its cap mark: the mark is on record for ${complete ? `all ${files(s.days_read)}` : `${n(s.recorded)} of ${files(s.days_read)}`} read`
    + ` (${n(s.capped_events)} capped ${s.capped_events === 1 ? "event" : "events"})`,
  ];
  if (s.owed > 0) {
    parts.push(s.enabled
      ? `${files(s.owed)} still to read again (two after each poll, oldest first)`
      : `${files(s.owed)} not read again (the re-read is switched off)`);
  }
  if (s.lost > 0) {
    const named = s.lost_days.length > 0 && s.lost_days.length <= 4 ? ` (${s.lost_days.join(", ")})` : "";
    parts.push(`${files(s.lost)} left the vendor before the mark was read${named}: on ${s.lost === 1 ? "that day" : "those days"} a blank is either no notional or a capped one`);
  }
  return parts.join(" · ");
}
