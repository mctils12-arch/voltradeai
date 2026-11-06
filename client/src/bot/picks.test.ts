// picks — the overview's plain-English card per pick: the decision journal's row matched to each held position,
// the honest fallback without one, multi-leg units, closed picks and the candidates passed on.
// Run: npx tsx --test client/src/bot/picks.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseDecisions, pickCards, closedPicks, passedOn, decisionMatches, entryDecision, optionWords, plainLine, routeMissing, NO_JOURNAL_WHY, type DecisionRow } from "./picks";
import type { BotPosition } from "./useBotData";

const NOW = Date.parse("2026-11-06T15:00:00Z");
const ago = (h: number) => new Date(NOW - h * 3_600_000).toISOString();
const pos = (ticker: string, o: Partial<BotPosition> = {}): BotPosition => ({ ticker, qty: 10, side: "long", entryPrice: 100, currentPrice: 102, marketValue: 1020, pnl: 20, pnlPct: 2, stopPrice: null, takeProfitPrice: null, phase: 1, rMultiple: 0, highestPnl: 0, daysHeld: 3, asset_class: "us_equity", isOption: false, levels: null, ...o });
const opt = (ticker: string, under: string, strike: number, type: "PUT" | "CALL", qty: number, entry: number, strategy: string | null, o: Partial<BotPosition> = {}): BotPosition =>
  pos(ticker, { qty, entryPrice: entry, currentPrice: entry, marketValue: entry * 100 * qty, pnl: 0, pnlPct: 0, asset_class: "us_option", isOption: true, optionMeta: { underlyingTicker: under, expiry: "2026-11-20", strike, optionType: type, strategy, setup: null, daysToExpiry: 14 }, ...o });

const raw = [
  { ts: ago(70), ticker: "QQQ", action: "buy", instrument: "shares", chosen: "61 QQQ shares", rejected: [{ instrument: "QQQ calls", reason: "IV rank 82 — calls cost more than the move they need" }, "SPY: lower expected return after short-term tax"], stages: { regime: "NEUTRAL, VIX 17 in contango — full risk budget", tax: { allowed: true, afterTaxPct: 1.9, reason: "edge 2.5% → 1.9% after short-term tax beats SPY's 0.6%" }, flows: null, extra_stage: 7 }, sizing: { qty: 61, notional: 45220.5, reason: "a third of the target size (scale-in 1 of 3)" }, summary: "Bought QQQ: tech momentum with the regime's full budget; the edge survives short-term tax." },
  { ts: ago(20), ticker: "QQQ", action: "scale_in", summary: "Added the second third: the first target held." },
  { ts: ago(30), ticker: "CRWV", action: "buy", instrument: "cash-secured put", chosen: "CRWV261120P00055000", rejected: [{ instrument: "shares", reason: "a put collects the premium without the downside of owning the stock" }], summary: "Sold a CRWV put: IV rank 91, premium rich." },
  { ts: ago(500), ticker: "SMH", action: "buy", summary: "an OLD buy from a previous holding" },
  { ts: ago(5), ticker: "TSLA", action: "sell", summary: "Sold TSLA: the stop at $412 was hit." },
  { ts: ago(6), ticker: "NVDA", action: "skip", rejected: [{ instrument: "shares", reason: "tax filter: 0.4% after tax vs SPY 0.6%" }] },
  { ts: ago(7), ticker: "NVDA", action: "skip", summary: "older NVDA skip" },
  { ts: ago(90), ticker: "AMD", action: "skip", summary: "too old to show" },
  { ticker: "BAD", action: "buy" }, // no ts — dropped
  "junk",
];

test("parse: rows newest first; stages in the pipeline's order with plain lines; sizing and rejected alternatives as words", () => {
  const p = parseDecisions(raw);
  assert.equal(p.ok, true);
  assert.equal(p.rows.length, 8);
  assert.deepEqual(p.rows.map((r) => r.ticker), ["TSLA", "NVDA", "NVDA", "QQQ", "CRWV", "QQQ", "AMD", "SMH"]);
  const q = p.rows.find((r) => r.ticker === "QQQ" && r.action === "buy")!;
  assert.deepEqual(q.stages.map((s) => s.label), ["Market regime", "Tax", "extra stage"]);
  assert.equal(q.stages[1].text, "edge 2.5% → 1.9% after short-term tax beats SPY's 0.6%");
  assert.equal(q.stages[2].text, "7");
  assert.equal(q.sizing, "61 for $45,221 — a third of the target size (scale-in 1 of 3)");
  assert.deepEqual(q.rejected, [{ instrument: "QQQ calls", reason: "IV rank 82 — calls cost more than the move they need" }, { instrument: "", reason: "SPY: lower expected return after short-term tax" }]);
});

test("the journal's real row shape: object stages read as plain lines, its sizing block as words, symbol / backfilled / pnl_pct carried", () => {
  const [r] = parseDecisions([{ v: 1, id: "x", ts: ago(2), ticker: "CRWV", action: "buy", instrument: "option", chosen: "sell_cash_secured_put", symbol: "crwv261120p00055000", backfilled: true, pnl_pct: null, rejected: [],
    stages: { regime: { label: "NEUTRAL", gate_regime: "NEUTRAL", vxx_ratio: 1.02, reason: "NEUTRAL market (VXX 1.02x its 30-day average)" }, sector: { name: "Technology", reason: null },
      selection: { score: 75.2, threshold: 65, drivers: [], dominant: { name: "vrp", label: "volatility risk premium", points: 9 }, reasons: ["IV rank 91", "VRP +6.1"] }, calendar: null },
    sizing: { qty: 1, price: 2.47, notional: 247, pct_equity: 0.25, reasoning: "one contract, $5,500 of cash set aside", scalars: {} }, summary: "Sold a CRWV put." }]).rows;
  assert.deepEqual(r.stages.map((x) => [x.key, x.text]), [["regime", "NEUTRAL market (VXX 1.02x its 30-day average)"], ["sector", "Technology"], ["selection", "score 75.2 vs the 65 bar · led by volatility risk premium (+9.0 points) · IV rank 91; VRP +6.1"]]);
  assert.equal(r.sizing, "1 at $2.47 = $247 (0.3% of equity) — one contract, $5,500 of cash set aside");
  assert.equal(r.symbol, "CRWV261120P00055000");
  assert.equal(r.backfilled, true);
  // the ordered symbol names the instrument: this row explains the contract, never CRWV shares
  const leg = opt("CRWV261120P00055000", "CRWV", 55, "PUT", -1, 2.47, "sell_cash_secured_put");
  assert.equal(decisionMatches(r, leg), true);
  assert.equal(decisionMatches(r, pos("CRWV")), false);
  assert.equal(pickCards([leg], [r], NOW)[0].backfilled, true);
  // an exit's result rides on the closed pick; a skip's "none" is no choice
  const rows = parseDecisions([{ ts: ago(5), ticker: "TSLA", action: "sell", pnl_pct: -4.2, summary: "stopped" }, { ts: ago(6), ticker: "NVDA", action: "skip", chosen: "none", summary: "tax" }]).rows;
  assert.equal(closedPicks(rows, [], NOW)[0].pnlPct, -4.2);
  assert.equal(rows.find((x) => x.ticker === "NVDA")?.chosen, null);
});

test("parse: anything but a list is 'not on this server yet'; an error says why; the route-missing test knows 404 and the SPA page", () => {
  assert.deepEqual(parseDecisions({}), { ok: false, rows: [], reason: "the decision journal is not on this server yet" });
  assert.deepEqual(parseDecisions(undefined), { ok: false, rows: [], reason: null });
  assert.equal(parseDecisions({ decisions: raw }).rows.length, 8);
  assert.equal(parseDecisions(undefined, new Error("404: Not Found")).reason, "the decision journal is not on this server yet");
  assert.equal(parseDecisions(undefined, new SyntaxError("Unexpected token '<', \"<!DOCTYPE \"... is not valid JSON")).reason, "the decision journal is not on this server yet");
  assert.match(parseDecisions(undefined, new Error("500: boom")).reason ?? "", /^decision journal unavailable · 500: boom/);
  assert.equal(routeMissing(new Error("404: x")), true);
  assert.equal(routeMissing(new Error("timeout")), false);
  assert.equal(plainLine({ a: 1, b: "x", c: { deep: 1 } }), "a 1 · b x");
  assert.equal(plainLine([{ summary: "one" }, "two", null]), "one; two");
  assert.equal(plainLine(null), null);
});

test("cards: the journal's reason, the alternatives, the stages, the size and what happened since, per held position", () => {
  const rows = parseDecisions(raw).rows;
  const cards = pickCards([pos("QQQ", { qty: 61, entryPrice: 741.32, daysHeld: 3 }), opt("CRWV261120P00055000", "CRWV", 55, "PUT", -1, 2.47, "sell_cash_secured_put", { daysHeld: 1 })], rows, NOW);
  assert.equal(cards.length, 2);
  const [q, c] = cards;
  assert.equal(q.source, "journal");
  assert.equal(q.why, "Bought QQQ: tech momentum with the regime's full budget; the edge survives short-term tax.");
  assert.equal(q.chosen, "61 QQQ shares");
  assert.equal(q.rejected.length, 2);
  assert.equal(q.sizing, "61 for $45,221 — a third of the target size (scale-in 1 of 3)");
  assert.deepEqual(q.later.map((l) => [l.action, l.text]), [["scale_in", "Added the second third: the first target held."]]);
  assert.equal(q.title, "QQQ");
  assert.equal(q.sub, "61 shares · held 3d");
  assert.equal(q.what, "Bought 61 shares of QQQ at $741.32, 3 days ago.");
  assert.equal(q.pnlPct, 2);
  assert.equal(c.source, "journal");
  assert.equal(c.why, "Sold a CRWV put: IV rank 91, premium rich.");
  assert.equal(c.title, "CRWV 55P 20-Nov");
  assert.equal(c.sub, "sold put · 14 days to expiry");
  assert.equal(c.what, "Sold 1 put on CRWV: keeps the $247 premium if CRWV stays above $55 through Nov 20; below about $52.53 it loses money.");
  assert.ok(c.exit.length > 2, "the option's exit plan comes from shared/positionLevels");
  assert.equal(c.exit[0].text, "whole lot at once");
});

test("cards without a journal row say so and keep the facts: no reason is inferred", () => {
  const rows = parseDecisions(raw).rows;
  // SMH's only buy is 500 h old while the position is 3 days old — a previous holding, never this one's reason
  const [s] = pickCards([pos("SMH", { daysHeld: 3 })], rows, NOW);
  assert.equal(s.source, "none");
  assert.equal(s.why, NO_JOURNAL_WHY);
  assert.equal(s.decidedAt, null);
  assert.deepEqual(s.rejected, []);
  assert.equal(s.exit[0].text, "stop state not recorded yet", "the stock exit line is the Book's");
  const none = pickCards([pos("KWEB")], [], NOW);
  assert.equal(none[0].source, "none");
  assert.deepEqual(pickCards(undefined, rows, NOW), []);
});

test("an option row never explains a share lot (and the other way round); an exact OCC symbol always matches", () => {
  const shareRow: DecisionRow = { ts: ago(1), t: NOW - 3_600_000, ticker: "META", action: "buy", instrument: "shares", chosen: null, rejected: [], stages: [], sizing: null, summary: "shares" };
  const optRow: DecisionRow = { ...shareRow, instrument: "bull put credit spread", summary: "spread" };
  const leg = opt("META261120P00600000", "META", 600, "PUT", -1, 4.1, "bull_put_credit_spread");
  assert.equal(decisionMatches(shareRow, leg), false);
  assert.equal(decisionMatches(optRow, leg), true);
  assert.equal(decisionMatches(optRow, pos("META")), false);
  assert.equal(decisionMatches(shareRow, pos("META")), true);
  assert.equal(decisionMatches({ ...shareRow, ticker: "XYZ", chosen: "META261120P00600000" }, leg), true);
  assert.equal(entryDecision(leg, [shareRow, optRow], NOW)?.summary, "spread");
  // a row that chose a DIFFERENT contract on the same underlying explains neither this leg nor the shares
  const other: DecisionRow = { ...shareRow, instrument: null, chosen: "META261219C00700000" };
  assert.equal(decisionMatches(other, leg), false);
  assert.equal(decisionMatches(other, pos("META")), false);
});

test("the legs of one multi-leg strategy are ONE card with their net P&L; single legs and shares keep their own", () => {
  const a = opt("META261120P00600000", "META", 600, "PUT", -1, 4.1, "bull_put_credit_spread", { pnl: 90, pnlPct: 22 });
  const b = opt("META261120P00580000", "META", 580, "PUT", 1, 1.9, "bull_put_credit_spread", { pnl: -60, pnlPct: -31.6 });
  const g = opt("GOOGL261219C00280000", "GOOGL", 280, "CALL", 1, 3.2, null, { pnl: 960, pnlPct: 300 });
  const cards = pickCards([a, g, pos("QQQ"), b], [], NOW);
  assert.deepEqual(cards.map((c) => c.title), ["QQQ", "META bull put credit spread", "GOOGL 280C 19-Dec"]);
  const spread = cards[1];
  assert.equal(spread.legs.length, 2);
  assert.equal(spread.pnlUsd, 30);
  assert.equal(spread.pnlPct, null, "a net percent of a spread reads wrong; each leg carries its own");
  assert.equal(spread.sub, "2 legs · 14 days to expiry");
  assert.equal(spread.exit[0].text, "all 2 legs as one");
  assert.equal(cards[2].what, "Bought 1 call on GOOGL: pays off if GOOGL is above about $283.20 by Nov 20; the most it can lose is the $320 paid.");
});

test("option words: a sold call, a bought put, and a leg with no strike recorded", () => {
  assert.equal(optionWords(opt("X", "KWEB", 30, "CALL", -2, 0.5, "covered_call")), "Sold 2 calls on KWEB: keeps the $100 premium if KWEB stays below $30 through Nov 20; above about $30.50 it loses money.");
  assert.equal(optionWords(opt("X", "SPY", 600, "PUT", 1, 5, null)), "Bought 1 put on SPY: pays off if SPY is below about $595.00 by Nov 20; the most it can lose is the $500 paid.");
  const bare = opt("X", "IWM", 0, "PUT", 1, 1, null);
  bare.optionMeta = null;
  assert.equal(optionWords(bare), "Bought 1 option on X.");
});

test("closed picks: exits of names no longer held, inside 14 days, one per ticker; passed on: skips inside 48 h, one per ticker, the reasons when no summary", () => {
  const rows = parseDecisions(raw).rows;
  const held = [pos("QQQ")];
  assert.deepEqual(closedPicks(rows, held, NOW).map((c) => [c.ticker, c.text]), [["TSLA", "Sold TSLA: the stop at $412 was hit."]]);
  assert.deepEqual(closedPicks(rows, [pos("TSLA")], NOW), [], "a name held again is not a closed pick");
  const passed = passedOn(rows, held, NOW);
  assert.deepEqual(passed.map((p) => [p.ticker, p.text]), [["NVDA", "tax filter: 0.4% after tax vs SPY 0.6%"]]);
  assert.equal(passedOn(rows, held, NOW, 100).length, 2, "AMD shows inside a wider window");
});
