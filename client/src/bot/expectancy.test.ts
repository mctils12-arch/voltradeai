import { test } from "node:test";
import assert from "node:assert/strict";
import { cell, flagged, negativeSleeves, negativeEdgeIssue, expectancyCaption, fmtWindowDay, rowTitle, TABLE_ORDER, TOTAL_GROUPS, COLS, GROUP_LABEL, FLAG_MIN_N } from "./expectancy";
import { expectancyBySleeve, type SleeveStats, type SleeveExpectancy } from "../../../shared/expectancy";
import { MINUS } from "./format";

const S = (p: Partial<SleeveStats> = {}): SleeveStats => ({ n: 0, wins: 0, losses: 0, winRate: null, avgWinPct: null, avgLossPct: null, payoffRatio: null, breakevenWinRate: null, expectancyPct: null, realizedUsd: 0, lastCloseAt: null, unpriced: 0, ...p });
// the live 2026-11-03 readings (the lots probe, window since May 7)
const OPTIONS = S({ n: 376, wins: 71, losses: 305, winRate: 18.88, avgWinPct: 31.73, avgLossPct: -28.47, payoffRatio: 1.11, breakevenWinRate: 47.3, expectancyPct: -17.1, realizedUsd: -12658, lastCloseAt: "2026-11-02T19:09:58.229Z" });
const STOCKS = S({ n: 101, wins: 32, losses: 69, winRate: 31.68, avgWinPct: 9.74, avgLossPct: -5.05, payoffRatio: 1.93, breakevenWinRate: 34.12, expectancyPct: -0.36, realizedUsd: -499.39, lastCloseAt: "2026-10-14T19:59:02.000Z" });
const FLOOR = S({ n: 1464, wins: 723, losses: 741, winRate: 49.39, avgWinPct: 0.59, avgLossPct: -0.6, payoffRatio: 1, breakevenWinRate: 50.04, expectancyPct: -0.01, realizedUsd: -447.45, lastCloseAt: "2026-11-02T19:09:58.229Z" });
const LIVE: SleeveExpectancy = { options: OPTIONS, stocks: STOCKS, floor: FLOOR, vrp: S(), active: S({ n: 477, wins: 103, losses: 374, winRate: 21.59, avgWinPct: 24.9, avgLossPct: -24.15, payoffRatio: 1.03, breakevenWinRate: 49.24, expectancyPct: -13.56, realizedUsd: -13157.39, lastCloseAt: "2026-11-02T19:09:58.229Z" }), all: S({ n: 1941, wins: 826, losses: 1115, winRate: 42.56, avgWinPct: 3.62, avgLossPct: -8.5, payoffRatio: 0.43, breakevenWinRate: 70.09, expectancyPct: -3.34, realizedUsd: -13604.84, lastCloseAt: "2026-11-02T19:09:58.229Z" }), unclassified: 0 };

test("the table is a reconciliation: options, stocks, their total (the headline), the passive legs, every lot; seven columns", () => {
  assert.deepEqual(TABLE_ORDER, ["options", "stocks", "active", "floor", "vrp", "all"]);
  assert.deepEqual([...TOTAL_GROUPS], ["active", "all"]);
  assert.deepEqual(COLS, ["expectancy", "n", "winRate", "avgWin", "avgLoss", "payoff", "breakeven"], "the headline metric first: visible at 390 px without a scroll");
  assert.equal(GROUP_LABEL.active, "options + stocks");
  assert.equal(GROUP_LABEL.vrp, "VRP leg");
  assert.match(rowTitle("floor", FLOOR), /rebalancing, not a trade decision · −\$447 realized · last close Nov 2/);
  assert.match(rowTitle("vrp", S()), /SVXY.*\$0 realized · no closed lots$/);
});

test("cells print the live numbers with the true minus and a hover line; the win rate is red below its breakeven, green above", () => {
  assert.deepEqual(cell(OPTIONS, "n"), { text: "376", tone: "t2", title: "376 priced closed lots" });
  const wr = cell(OPTIONS, "winRate");
  assert.equal(wr.text, "18.9%"); assert.equal(wr.tone, "r"); assert.match(wr.title, /71 of 376 lots made money · breakeven 47\.3% — below it/);
  assert.equal(cell(OPTIONS, "avgWin").text, "+31.73%");
  assert.equal(cell(OPTIONS, "avgWin").tone, "g");
  assert.equal(cell(OPTIONS, "avgLoss").text, `${MINUS}28.47%`);
  assert.equal(cell(OPTIONS, "avgLoss").tone, "r");
  assert.deepEqual([cell(OPTIONS, "payoff").text, cell(OPTIONS, "breakeven").text, cell(OPTIONS, "expectancy").text], ["1.11", "47.3%", `${MINUS}17.10%`]);
  assert.equal(cell(OPTIONS, "expectancy").tone, "r");
  assert.match(cell(OPTIONS, "expectancy").title, /−\$12,658 realized/);
  assert.match(cell(OPTIONS, "breakeven").title, /avg loss ÷ \(avg win \+ avg loss\)/);
  const up = cell(S({ n: 10, wins: 6, losses: 4, winRate: 60, avgWinPct: 2, avgLossPct: -1, payoffRatio: 2, breakevenWinRate: 33.33, expectancyPct: 0.8 }), "winRate");
  assert.equal(up.tone, "g");
  assert.equal(cell(S({ n: 3, wins: 1, losses: 2, winRate: 33.33, avgWinPct: 1, avgLossPct: -1, payoffRatio: 1, breakevenWinRate: 50, expectancyPct: 0 }), "expectancy").tone, "t2", "exactly zero is neither red nor green");
  assert.equal(cell(S({ n: 5, unpriced: 3, winRate: 0, losses: 5, avgLossPct: -1, expectancyPct: -1 }), "n").title, "5 priced closed lots · 3 closed against an unknown basis excluded");
});

test("a dash always carries its reason: no lots, no winning lot, no losing lot, losses averaging zero — never NaN or Infinity", () => {
  for (const c of COLS) {
    const k = cell(S(), c);
    assert.equal(k.text, c === "n" ? "0" : "—", `${c} on an empty sleeve`);
    assert.ok(!/NaN|Infinity/.test(k.text));
    if (c !== "n") assert.equal(k.title, "no closed lots in the window");
  }
  const noLoss = S({ n: 2, wins: 2, winRate: 100, avgWinPct: 15, expectancyPct: 15 });
  assert.deepEqual([cell(noLoss, "avgLoss").text, cell(noLoss, "avgLoss").title], ["—", "no losing lot"]);
  assert.deepEqual([cell(noLoss, "payoff").title, cell(noLoss, "breakeven").title], ["no losing lot", "no losing lot"]);
  assert.equal(cell(noLoss, "winRate").tone, "t2", "no breakeven to compare with: neutral");
  const noWin = S({ n: 2, losses: 2, winRate: 0, avgLossPct: -15, expectancyPct: -15 });
  assert.deepEqual([cell(noWin, "avgWin").title, cell(noWin, "payoff").title], ["no winning lot", "no winning lot"]);
  const flat = S({ n: 2, wins: 1, losses: 1, winRate: 50, avgWinPct: 10, avgLossPct: 0, payoffRatio: null, breakevenWinRate: 0, expectancyPct: 5 });
  assert.equal(cell(flat, "payoff").title, "the losses average 0 % — nothing to divide by");
  assert.equal(cell(flat, "avgLoss").tone, "t3");
  assert.equal(cell(flat, "breakeven").text, "0.0%");
  // the whole pipeline on an empty ledger prints only dashes and zeros
  const x = expectancyBySleeve([], new Set(["QQQ"]));
  for (const g of TABLE_ORDER) for (const c of COLS) assert.ok(!/NaN|Infinity/.test(cell(x[g], c).text));
});

test("the human's rule: below zero over n ≥ 30 flags a sleeve on the number it prints; n < 30, a positive sleeve and the totals never flag", () => {
  assert.equal(FLAG_MIN_N, 30);
  assert.equal(flagged(OPTIONS), true);
  assert.equal(flagged(STOCKS), true);
  assert.equal(flagged(FLOOR), true, "−0.01 % over 1,464 lots is below zero: the rule is literal, the magnitude is in the issue's words");
  assert.equal(flagged(S({ ...OPTIONS, n: 29 })), false);
  assert.equal(flagged(S({ ...OPTIONS, expectancyPct: 0 })), false);
  assert.equal(flagged(S({ ...OPTIONS, expectancyPct: 0.01 })), false);
  assert.equal(flagged(S()), false);
  assert.deepEqual(negativeSleeves(LIVE).map((s) => s.sleeve), ["options", "floor", "stocks"], "the table's sleeve order, totals left out");
  assert.deepEqual(negativeSleeves({ ...LIVE, options: S({ ...OPTIONS, n: 29 }), floor: S({ ...FLOOR, expectancyPct: 0.01 }), stocks: S({ ...STOCKS, expectancyPct: 0 }) }), []);
  assert.deepEqual(negativeSleeves(null), []);
  assert.deepEqual(negativeSleeves(undefined), []);
});

test("the TOP issue names every failing sleeve with its expectancy, n and win-vs-breakeven; nothing failing → null", () => {
  const iss = negativeEdgeIssue(LIVE)!;
  assert.equal(iss.title, "Negative expectancy · options, ETF floor, stocks");
  assert.equal(iss.condition, `options ${MINUS}17.10% over 376 lots (win 18.9% vs breakeven 47.3%) · ETF floor ${MINUS}0.01% over 1,464 lots (win 49.4% vs breakeven 50.0%) · stocks ${MINUS}0.36% over 101 lots (win 31.7% vs breakeven 34.1%) · broker fills, n ≥ 30`);
  assert.match(iss.consequence, /loses on average/);
  assert.match(iss.clears, /turns positive over n ≥ 30, or the sleeve stops entering/);
  assert.equal(negativeEdgeIssue({ ...LIVE, options: S({ ...OPTIONS, n: 29 }), floor: S({ ...FLOOR, expectancyPct: 0 }), stocks: S({ ...STOCKS, expectancyPct: 1 }) }), null);
  assert.equal(negativeEdgeIssue(null), null);
  // the totals being negative never makes an issue by themselves
  assert.equal(negativeEdgeIssue({ ...LIVE, options: S({ ...OPTIONS, expectancyPct: 1 }), floor: S({ ...FLOOR, expectancyPct: 1 }), stocks: S({ ...STOCKS, expectancyPct: 1 }) }), null);
});

test("the caption carries the window, the ledger's last close, the lot count, the cache time, the source — and says when the cache is stale or a refresh failed", () => {
  const b = { windowStart: "2026-05-07T00:00:00.000Z", asOf: "2026-11-03T17:00:29.456Z", source: "alpaca fills via sleeve ledger", cached_s: 240, stale: false, lastError: null, bySleeve: LIVE };
  assert.deepEqual(expectancyCaption(b).map((p) => p.text), ["since May 7", "last close Nov 2", "1,941 lots", "as of 12:00:29 ET", "alpaca fills via sleeve ledger"]);
  const stale = expectancyCaption({ ...b, stale: true, cached_s: 3600, lastError: "activities unreadable" });
  assert.deepEqual(stale.filter((p) => p.tone === "warn").map((p) => p.text), ["cache 1h old · a refresh is running or failed", "last refresh failed: activities unreadable"]);
  const empty = expectancyCaption({ ...b, bySleeve: { ...LIVE, all: S({ unpriced: 4 }), unclassified: 1 } });
  assert.deepEqual(empty.slice(1, 3).map((p) => p.text), ["no closed lots yet", "0 lots · 4 unpriced excluded · 1 unclassified"]);
  assert.equal(expectancyCaption({ ...b, windowStart: null, bySleeve: undefined })[0].text, "whole fill history");
  // the window opens at a UTC midnight: its day is the UTC one (the ET formatter would say May 6)
  assert.equal(fmtWindowDay("2026-05-07T00:00:00.000Z"), "May 7");
  assert.equal(fmtWindowDay("not a date"), "—");
});
