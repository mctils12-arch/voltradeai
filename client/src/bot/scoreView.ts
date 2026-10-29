// Bot page — the daily bot-vs-SPY scoreboard's view arithmetic (pure). The
// numbers come from GET /api/bot/scoreboard (server/scoreboard.ts); this
// file only picks rows and writes the honesty lines around them.
import type { Scoreboard, ScoreRow, ScoreSummary } from "./useBotData";
import { fmtMoney } from "./format";

/** Rows shown before "show all": about a month of sessions. */
export const SCORE_VISIBLE_DAYS = 30;

export function visibleRows(rows: readonly ScoreRow[], showAll: boolean): readonly ScoreRow[] {
  return showAll ? rows : rows.slice(0, SCORE_VISIBLE_DAYS);
}

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "2026-10-27" → "Oct 27" (the year only when it is not `year`). */
export function fmtDay(date: string | null | undefined, year?: number): string {
  const m = date ? /^(\d{4})-(\d{2})-(\d{2})$/.exec(date) : null;
  if (!m) return "—";
  const mon = MON[Number(m[2]) - 1] ?? m[2];
  return `${mon} ${Number(m[3])}${year !== undefined && Number(m[1]) !== year ? ` ${m[1]}` : ""}`;
}

/** "48 of 120 days · 40%" — strictly beating SPY, over the rows compared. */
export function beatLine(s: ScoreSummary): string {
  const pct = s.daysCompared ? Math.round((s.daysBeat / s.daysCompared) * 100) : 0;
  return `${s.daysBeat} of ${s.daysCompared} day${s.daysCompared === 1 ? "" : "s"} · ${pct}%`;
}

/** The deposits/withdrawals caveat: the daily return is the raw equity
 *  change, so a journal or transfer would read as performance. */
export function cashFlowNote(cf: Scoreboard["cashFlows"]): { tone: "ok" | "warn" | "unknown"; text: string } {
  if (!cf) return { tone: "unknown", text: "deposits / withdrawals not checked yet (the non-trade ledger is still loading) — daily returns are the raw equity change" };
  const since = cf.since ? ` since ${fmtDay(cf.since.slice(0, 10))}` : "";
  const moved = (cf.journals ?? 0) + (cf.transfers ?? 0);
  if (moved === 0) return { tone: "ok", text: `journals / transfers net $0${since} (Alpaca non-trade ledger) — no deposit or withdrawal inflates the daily returns` };
  return { tone: "warn", text: `${fmtMoney(moved)} of journals / transfers${since} sit inside the equity change — daily returns are NOT adjusted for them` };
}
