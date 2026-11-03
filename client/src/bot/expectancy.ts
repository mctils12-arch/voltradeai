// Bot page — expectancy by sleeve, the view model ([PRODUCT] 2026-11-03,
// human-approved item B2). The server computes the split once per sleeve-
// cache refresh (shared/expectancy.ts, folded into /api/bot/performance's
// broker block by server/edgeLedger.ts); this file only decides how each
// number is printed, why a cell is a dash, which sleeve is flagged and what
// the TOP issue says. Pure, tested (expectancy.test.ts); the table itself is
// expectancyTable.tsx.
import type { SleeveExpectancy, SleeveStats, LotGroup, SleeveId } from "../../../shared/expectancy";
import { SLEEVE_IDS } from "../../../shared/expectancy";
import { fmtPct, fmtCount, fmtMonDay, fmtClockET, fmtAge, fmtMoney } from "./format";

export const GROUP_LABEL: Record<LotGroup, string> = { options: "options", stocks: "stocks", active: "options + stocks", floor: "ETF floor", vrp: "VRP leg", all: "all lots" };
/** the small word after a total's label: which rows it adds up */
export const GROUP_NOTE: Partial<Record<LotGroup, string>> = { active: "headline", all: "+ floor + VRP" };
/** the table reads as a reconciliation: the two trading sleeves, their total
 *  (the Expectancy tile's headline), the two passive legs, every lot */
export const TABLE_ORDER: readonly LotGroup[] = ["options", "stocks", "active", "floor", "vrp", "all"];
export const TOTAL_GROUPS: ReadonlySet<LotGroup> = new Set<LotGroup>(["active", "all"]);
/** the human's rule: a sleeve below zero over at least this many closed lots is the page's TOP issue */
export const FLAG_MIN_N = 30;

export type Col = "n" | "winRate" | "avgWin" | "avgLoss" | "payoff" | "breakeven" | "expectancy";
/** expectancy first: the headline metric stays visible at the phone width, the detail columns scroll */
export const COLS: readonly Col[] = ["expectancy", "n", "winRate", "avgWin", "avgLoss", "payoff", "breakeven"];
export const COL_LABEL: Record<Col, string> = { n: "n", winRate: "win", avgWin: "avg win", avgLoss: "avg loss", payoff: "payoff", breakeven: "breakeven", expectancy: "expectancy" };

export type Tone = "g" | "r" | "t2" | "t3";
export interface Cell { text: string; tone: Tone; title: string }

const DASH = "—";
/** why a ratio is a dash: the first reason that applies, in the shape's own terms */
function missing(s: SleeveStats, needsLoss: boolean): string {
  if (s.n === 0) return "no closed lots in the window";
  if (s.wins === 0) return "no winning lot";
  if (s.losses === 0) return "no losing lot";
  if (needsLoss && s.avgLossPct === 0) return "the losses average 0 % — nothing to divide by";
  return "not computable";
}

/** One printed cell: text, colour tone (never the only channel — signed
 *  numbers carry the true minus) and the hover line that says what it is. */
export function cell(s: SleeveStats, col: Col): Cell {
  switch (col) {
    case "n":
      return { text: fmtCount(s.n), tone: s.n ? "t2" : "t3", title: `${fmtCount(s.n)} priced closed lot${s.n === 1 ? "" : "s"}${s.unpriced ? ` · ${fmtCount(s.unpriced)} closed against an unknown basis excluded` : ""}` };
    case "winRate": {
      if (s.winRate === null) return { text: DASH, tone: "t3", title: missing(s, false) };
      const be = s.breakevenWinRate;
      const tone: Tone = be === null ? "t2" : s.winRate < be ? "r" : "g";
      return { text: `${s.winRate.toFixed(1)}%`, tone, title: `${fmtCount(s.wins)} of ${fmtCount(s.n)} lots made money${be !== null ? ` · breakeven ${be.toFixed(1)}% — ${s.winRate < be ? "below it: the sleeve loses on average" : "above it"}` : ""}` };
    }
    case "avgWin":
      return s.avgWinPct === null ? { text: DASH, tone: "t3", title: missing(s, false) } : { text: fmtPct(s.avgWinPct), tone: "g", title: `mean return of the ${fmtCount(s.wins)} winning lot${s.wins === 1 ? "" : "s"}, P&L ÷ cost` };
    case "avgLoss":
      return s.avgLossPct === null ? { text: DASH, tone: "t3", title: missing(s, false) } : { text: fmtPct(s.avgLossPct), tone: s.avgLossPct === 0 ? "t3" : "r", title: `mean return of the ${fmtCount(s.losses)} losing lot${s.losses === 1 ? "" : "s"} (a flat close counts as a loss of 0), P&L ÷ cost` };
    case "payoff":
      return s.payoffRatio === null ? { text: DASH, tone: "t3", title: missing(s, true) } : { text: s.payoffRatio.toFixed(2), tone: "t2", title: "average win ÷ average loss" };
    case "breakeven":
      return s.breakevenWinRate === null ? { text: DASH, tone: "t3", title: missing(s, false) } : { text: `${s.breakevenWinRate.toFixed(1)}%`, tone: "t2", title: "avg loss ÷ (avg win + avg loss): the win rate at which expectancy is exactly zero" };
    case "expectancy":
      return s.expectancyPct === null ? { text: DASH, tone: "t3", title: missing(s, false) } : { text: fmtPct(s.expectancyPct), tone: s.expectancyPct < 0 ? "r" : s.expectancyPct > 0 ? "g" : "t2", title: `mean P&L ÷ cost over every one of the ${fmtCount(s.n)} lots — ${fmtMoney(s.realizedUsd)} realized` };
  }
}

/** the hover line of a row's label: what the sleeve is and its dollars */
export const GROUP_WHAT: Record<LotGroup, string> = {
  options: "every option contract (OCC symbol) the bot closed",
  stocks: "single stocks and ETF instruments outside the passive set",
  active: "options + stocks — the sleeves the bot trades; the Expectancy tile's headline",
  floor: "the passive floor basket and its legs (QQQ, SPY, SMH, KWEB, VXUS, GLD) — rebalancing, not a trade decision",
  vrp: "the third leg's VRP harvest (SVXY)",
  all: "every priced closed lot, the four sleeves together",
};
export function rowTitle(g: LotGroup, s: SleeveStats): string {
  return `${GROUP_WHAT[g]} · ${fmtMoney(s.realizedUsd)} realized${s.lastCloseAt ? ` · last close ${fmtMonDay(s.lastCloseAt)}` : " · no closed lots"}`;
}

/** the human's rule, on the rounded number the row prints */
export function flagged(s: SleeveStats, minN = FLAG_MIN_N): boolean {
  return s.n >= minN && s.expectancyPct !== null && s.expectancyPct < 0;
}

export interface NegativeSleeve { sleeve: SleeveId; n: number; expectancyPct: number; winRate: number | null; breakevenWinRate: number | null }
/** the sleeves that trip the rule, in the table's order; the totals never flag on their own */
export function negativeSleeves(x: SleeveExpectancy | null | undefined, minN = FLAG_MIN_N): NegativeSleeve[] {
  if (!x) return [];
  return SLEEVE_IDS.filter((s) => flagged(x[s], minN)).map((s) => ({ sleeve: s, n: x[s].n, expectancyPct: x[s].expectancyPct as number, winRate: x[s].winRate, breakevenWinRate: x[s].breakevenWinRate }));
}

/** The TOP issue's words (derive.ts builds the Issue around them): one issue
 *  naming every failing sleeve with its numbers, or null. */
export function negativeEdgeIssue(x: SleeveExpectancy | null | undefined, minN = FLAG_MIN_N): { title: string; condition: string; consequence: string; clears: string } | null {
  const neg = negativeSleeves(x, minN);
  if (!neg.length) return null;
  return {
    title: `Negative expectancy · ${neg.map((s) => GROUP_LABEL[s.sleeve]).join(", ")}`,
    condition: `${neg.map((s) => `${GROUP_LABEL[s.sleeve]} ${fmtPct(s.expectancyPct)} over ${fmtCount(s.n)} lots${s.winRate !== null && s.breakevenWinRate !== null ? ` (win ${s.winRate.toFixed(1)}% vs breakeven ${s.breakevenWinRate.toFixed(1)}%)` : ""}`).join(" · ")} · broker fills, n ≥ ${minN}`,
    consequence: "every lot the sleeve closes loses on average — it compounds the wrong way while it keeps trading",
    clears: `the sleeve's expectancy per closed lot turns positive over n ≥ ${minN}, or the sleeve stops entering (shadow mode)`,
  };
}

/** The provenance line under the table: where, since when, how fresh. Every
 *  number on the page carries its freshness (PREMIUM EXPERIENCE STANDARD). */
export interface CaptionPart { text: string; tone?: "warn" }
/** the window opens at a UTC midnight (server/bot.ts refreshSleeves: the equity curve's first day + 1), so its
 *  calendar day is the UTC one — the ET formatter would print the evening before (May 6 for 2026-05-07T00:00Z) */
export function fmtWindowDay(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}
export function expectancyCaption(b: { windowStart: string | null; asOf: string; source: string; cached_s: number; stale: boolean; lastError: string | null; bySleeve?: SleeveExpectancy }): CaptionPart[] {
  const x = b.bySleeve;
  const parts: CaptionPart[] = [];
  parts.push({ text: b.windowStart ? `since ${fmtWindowDay(b.windowStart)}` : "whole fill history" });
  if (x) {
    parts.push({ text: x.all.lastCloseAt ? `last close ${fmtMonDay(x.all.lastCloseAt)}` : "no closed lots yet" });
    parts.push({ text: `${fmtCount(x.all.n)} lots${x.all.unpriced ? ` · ${fmtCount(x.all.unpriced)} unpriced excluded` : ""}${x.unclassified ? ` · ${fmtCount(x.unclassified)} unclassified` : ""}` });
  }
  parts.push({ text: `as of ${fmtClockET(b.asOf)} ET` });
  if (b.stale) parts.push({ text: `cache ${fmtAge(b.cached_s * 1000)} old · a refresh is running or failed`, tone: "warn" });
  parts.push({ text: b.source });
  if (b.lastError) parts.push({ text: `last refresh failed: ${b.lastError}`, tone: "warn" });
  return parts;
}
