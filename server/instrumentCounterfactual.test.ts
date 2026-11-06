// INSTRUMENT COUNTERFACTUAL, Node half (2026-11-06, server/instrumentCounterfactual.ts;
// human refocus spec §3 step 7 "instrument choice: stock vs options vs both; log
// why the rejected instrument lost"). Pins: the would-be leg an entered,
// selector-decided trade records for the instrument it passed over (and none
// for a decision the selector did not make); the journal row's rejected entry
// carries it (normalize + the row cut keep it); the ledger's decision row (one
// per such entry, its shape and plain-English summary, the off switch, the
// rotation, never throwing); the fold of decision / proxy / outcome rows across
// generations; the journal read path attaching the settled outcomes; the
// byInstrumentChoice aggregate (means, edge, equal-capital dollars, counts,
// reasons, the window) in GET /api/bot/attribution; the bot.ts wiring. The
// Python half (option_alternative, the outcome math, the nightly job, parity):
// test_instrument_counterfactual.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  INSTRUMENT_CF_FILENAME, INSTRUMENT_CF_ROW_MAX_BYTES, appendInstrumentCounterfactuals, attachCounterfactualOutcomes,
  contractLegs, counterfactualRow, foldCounterfactuals, readInstrumentCounterfactuals, wouldBeFor,
} from "./instrumentCounterfactual";
import { appendDecisions, entryDecision, exitDecision, normalizeDecision, readDecisions, skipDecision, tierDecision, type DecisionRow } from "./decisionJournal";
import { attributionReport, instrumentChoiceBuckets, type AttributionInputs } from "./decisionAttribution";
import { JOB_REGISTRY } from "./jobHeartbeat";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const src = fs.readFileSync(path.join(here, "instrumentCounterfactual.ts"), "utf8");
const journalSrc = fs.readFileSync(path.join(here, "decisionJournal.ts"), "utf8");
const T0 = Date.parse("2026-11-06T15:30:00Z");     // 10:30 ET
const ON = {} as Record<string, string | undefined>;
const tmpDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "instcf-"));
const SHORT = "AMD261218P00145000", LONG = "AMD261218P00140000";

const PY_ALT = {
  instrument: "option", strategy: "sell_cash_secured_put", selector_score: 41, scored: true, chosen_by_selector: false,
  selector_reason: "options (sell_cash_secured_put) scored 41 vs 63 for the stock", contract: null,
  contract_reason: "no option chain was fetched for this candidate in the scan",
  proxy: { kind: "atm_call", right: "C", side: "buy", dte_min: 30, dte_max: 45, underlying: "AMD", underlying_price: 150.2 },
  source: "instrument_selector",
};
const TRADE = {
  ticker: "AMD", side: "buy", trade_type: "stock", price: 150.2, shares: 12, score: 71.3, stop_loss: 142.7, take_profit: 165.2,
  position_value: 1802.4, instrument: "stock", instrument_strategy: "buy_stock",
  instrument_reasoning: "CHOSEN: STOCK (buy_stock) | Scores: [stock=63 | options=41] | Stock",
  instrument_scores: { stock: { score: 63, reasoning: "Stock" }, etf: null, options: { score: 41, strategy: "sell_cash_secured_put", reasoning: "Options (sell_cash_secured_put)" } },
  option_alternative: PY_ALT,
};
const stockOutcome = () => ({ instrument: "stock" as const, chosen: "buy_stock", side: "buy", symbol: "AMD", qty: 12, price: 150.2, notional: 1802.4,
  status: "accepted", orderId: "ord-1", mode: "live" as const, source: "bot.ts executeTrades.stock", equity: 100_000 });
const OPT_TRADE = { ...TRADE, instrument: "options", use_options: true, options_strategy: "bull_put_credit_spread",
  instrument_scores: { stock: { score: 63, reasoning: "Stock" }, etf: null, options: { score: 68, strategy: "bull_put_credit_spread", reasoning: "Options" } } };
const optionOutcome = () => ({ instrument: "option" as const, chosen: "bull_put_credit_spread", side: "sell", symbol: SHORT, qty: 2, price: 1.1, notional: 780,
  status: "accepted", orderId: "ord-2", mode: "live" as const, source: "bot.ts executeTrades.options", equity: 100_000,
  contract: { short_leg: SHORT, long_leg: LONG, qty: 2, net_credit: 1.1, limit_price: 1.1, max_loss: 780 } });

// ── the would-be leg ─────────────────────────────────────────────────────────
test("wouldBeFor: a share entry carries the selector's option block; an options entry its stock plan; no selector decision -> null", () => {
  assert.deepEqual(wouldBeFor(TRADE, { instrument: "stock" }), { instrument: "option", would_be: PY_ALT });
  assert.deepEqual(wouldBeFor(TRADE, { instrument: "etf" })?.instrument, "option", "an ETF entry passed over the option too");
  const noPrice = wouldBeFor({ ...TRADE, option_alternative: { ...PY_ALT, proxy: { ...PY_ALT.proxy, underlying_price: null } } }, { instrument: "stock" });
  assert.equal((noPrice?.would_be.proxy as Record<string, unknown>).underlying_price, 150.2, "the scan's price fills a missing proxy price");
  const built = wouldBeFor({ ...TRADE, option_alternative: undefined, instrument_reasoning: "CHOSEN: STOCK | Options ruled out: options exposure 17.0% at max 16%" }, { instrument: "stock" });
  assert.deepEqual(built, { instrument: "option", would_be: {
    instrument: "option", strategy: "sell_cash_secured_put", selector_score: 41, scored: true, chosen_by_selector: false,
    selector_reason: "options were ruled out: options exposure 17.0% at max 16%", contract: null,
    contract_reason: "the selector's option block was not on the trade, so the ATM proxy is resolved and priced afterwards from Alpaca's contract list and daily option bars",
    source: "the scan trade (no option_alternative from the selector)",
    proxy: { kind: "atm_call", right: "C", side: "buy", dte_min: 30, dte_max: 45, underlying: "AMD", underlying_price: 150.2 },
  } }, "built from the trade when the selector's block is missing");
  assert.deepEqual(wouldBeFor(OPT_TRADE, { instrument: "option" }), { instrument: "stock", would_be: {
    instrument: "stock", symbol: "AMD", side: "buy", entry_price: 150.2, qty: 12, notional: 1802.4, stop: 142.7, target: 165.2,
    selector_score: 63, source: "the scan's stock plan (price, size, stop and target on the trade)",
  } });
  assert.equal(wouldBeFor({ ...OPT_TRADE, side: "sell" }, { instrument: "option" })?.would_be.note, "stock shorting is off: a short-stock alternative is hypothetical");
  assert.equal(wouldBeFor({ ...TRADE, regime_at_entry: "OPTIONS_SCANNER" }, { instrument: "option" }), null, "the scanner's setups: no selector decision");
  assert.equal(wouldBeFor({ ...TRADE, instrument_scores: {} }, { instrument: "stock" }), null, "no selector ran");
  assert.equal(wouldBeFor(TRADE, { instrument: "floor" }), null);
  assert.equal(wouldBeFor(null, { instrument: "stock" }), null);
});

test("contractLegs: short_* sold, long_* bought, a single contract takes its side; OCC symbols only, each once", () => {
  assert.deepEqual(contractLegs({ short_leg: SHORT, long_leg: LONG, occ_symbol: SHORT }, "sell"),
    [{ symbol: SHORT, side: "sell" }, { symbol: LONG, side: "buy" }]);
  assert.deepEqual(contractLegs({ occ_symbol: "TLT261219P00085000", side: "sell" }, "buy"), [{ symbol: "TLT261219P00085000", side: "sell" }]);
  assert.deepEqual(contractLegs({ call_leg: "SPY261218C00600000", put_leg: "SPY261218P00600000" }, "buy").map((l) => l.side), ["buy", "buy"]);
  assert.deepEqual(contractLegs({ occ_symbol: "AMD" }, "buy"), []);
  assert.deepEqual(contractLegs(null, "buy"), []);
});

test("entry rows: the passed-over instrument's rejected entry carries would_be; an options entry's legs are on its stage", () => {
  const r = entryDecision(TRADE, stockOutcome(), T0) as DecisionRow;
  assert.deepEqual(r.rejected.map((x) => [x.instrument, Boolean(x.would_be)]), [["etf", false], ["option", true]]);
  assert.equal(r.rejected[1].reason, "scored 41 vs 63 for the stock (Options (sell_cash_secured_put))", "the reason is unchanged");
  assert.deepEqual(r.rejected[1].would_be, PY_ALT);
  const o = entryDecision(OPT_TRADE, optionOutcome(), T0) as DecisionRow;
  assert.deepEqual(o.stages.options?.legs, [{ symbol: SHORT, side: "sell" }, { symbol: LONG, side: "buy" }]);
  const stockAlt = o.rejected.find((x) => x.instrument === "stock");
  assert.equal(stockAlt?.would_be?.entry_price, 150.2);
  assert.equal(o.rejected.filter((x) => x.would_be).length, 1, "one passed-over leg per decision");
  const scanner = entryDecision({ ticker: "TLT", trade_type: "options", regime_at_entry: "OPTIONS_SCANNER", options_strategy: "sell_cash_secured_put", instrument_scores: {} },
    { ...optionOutcome(), symbol: "TLT261219P00085000", contract: { occ_symbol: "TLT261219P00085000", side: "sell" } }, T0) as DecisionRow;
  assert.deepEqual(scanner.rejected, []);
  assert.deepEqual(scanner.stages.options?.legs, [{ symbol: "TLT261219P00085000", side: "sell" }]);
});

test("normalizeDecision and the row cut keep would_be; an empty one is dropped; it survives the journal round trip", () => {
  const n = normalizeDecision({ action: "buy", ticker: "amd", rejected: [
    { instrument: "option", reason: "r", would_be: { instrument: "option", proxy: { underlying_price: Number.NaN } } },
    { instrument: "etf", reason: "no ETF", would_be: {} }] });
  assert.deepEqual(n?.rejected, [{ instrument: "option", reason: "r", would_be: { instrument: "option", proxy: { underlying_price: null } } },
    { instrument: "etf", reason: "no ETF" }]);
  const dir = tmpDir();
  const big = entryDecision({ ...TRADE, reasons: ["x".repeat(2000)], score_breakdown: Object.fromEntries(Array.from({ length: 50 }, (_, k) => [`k${k}`, k])),
    sizing_reasoning: "s".repeat(220) }, stockOutcome(), T0) as DecisionRow;
  big.stages.selection = { ...big.stages.selection, reasons: Array.from({ length: 6 }, () => "y".repeat(1500)) };
  assert.equal(appendDecisions(dir, [big], undefined, ON), 1);
  const back = readDecisions(dir)[0];
  assert.equal(back.truncated, true, "the row was cut to DECISION_ROW_MAX_BYTES");
  assert.deepEqual(back.rejected.find((x) => x.instrument === "option")?.would_be, PY_ALT, "would_be survives the cut");
});

// ── the ledger's decision rows ───────────────────────────────────────────────
test("append: one decision row per entry with a would_be — skips, exits, tier rows and scanner entries write none", () => {
  const dir = tmpDir();
  const stock = entryDecision(TRADE, stockOutcome(), T0);
  const opt = entryDecision(OPT_TRADE, optionOutcome(), T0);
  const rows = [stock, opt, skipDecision(TRADE, "rejected_size_cap", "too big", "bot.ts executeTrades", T0),
    exitDecision({ ticker: "AMD", exitType: "stop_loss", reason: "stop", side: "sell", qty: 12, price: 140, pnlPct: -6, source: "ws" }, T0),
    tierDecision({ ticker: "TLT", action: "SELL_CSP", tier: 1, reason: "harvest" }, { status: "submitted", occ_symbol: "TLT261219P00085000", qty: 1, limit_price: 1.2 }, null, T0),
    entryDecision({ ...TRADE, regime_at_entry: "OPTIONS_SCANNER" }, stockOutcome(), T0), null];
  assert.equal(appendInstrumentCounterfactuals(dir, rows, undefined, ON), 2);
  const lines = fs.readFileSync(path.join(dir, INSTRUMENT_CF_FILENAME), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, 2);
  const [a, b] = lines;
  assert.deepEqual([a.v, a.kind, a.id, a.ts, a.entry_date, a.ticker, a.mode, a.source, a.choice],
    [1, "decision", stock?.id, "2026-11-06T15:30:00.000Z", "2026-11-06", "AMD", "live", "bot.ts executeTrades.stock", "stock_over_option"]);
  assert.deepEqual(a.chosen, { instrument: "stock", symbol: "AMD", side: "buy", strategy: "buy_stock", price: 150.2, qty: 12, notional: 1802.4 });
  assert.deepEqual(a.rejected, { ...PY_ALT, instrument: "option", reason: "scored 41 vs 63 for the stock (Options (sell_cash_secured_put))" });
  assert.equal(a.summary, "Chose stock over options for AMD: bought 12 AMD at ~$150.20. Why not options: scored 41 vs 63 for the stock "
    + "(Options (sell_cash_secured_put)). The option leg is a proxy: a bought ATM call expiring 30-45 days after 2026-11-06, resolved and "
    + "priced by the nightly job. The selector's own options idea was a sell cash secured put (score 41); it is named, not priced.");
  assert.equal(b.choice, "option_over_stock");
  assert.deepEqual(b.chosen, { instrument: "option", symbol: SHORT, side: "sell", strategy: "bull_put_credit_spread", price: 1.1, qty: 2, notional: 780,
    legs: [{ symbol: SHORT, side: "sell" }, { symbol: LONG, side: "buy" }], cash_required: null, max_loss: 780, credit: 1.1 });
  assert.equal(b.rejected.instrument, "stock");
  assert.equal(b.summary, "Chose options over the stock for AMD: opened a bull put credit spread (AMD $145 put expiring 2026-12-18). "
    + "Why not the stock: scored 63 vs 68 for the option (Stock). The stock leg: 12 AMD at ~$150.20, stop $142.70, target $165.20.");
  assert.ok(lines.every((l) => Buffer.byteLength(JSON.stringify(l)) <= INSTRUMENT_CF_ROW_MAX_BYTES));
  assert.equal(counterfactualRow(skipDecision(TRADE, "rejected_size_cap", "x")), null);
});

test("append: the off switch, rotation above the cap, and a write failure never throws", () => {
  const dir = tmpDir();
  const row = entryDecision(TRADE, stockOutcome(), T0);
  assert.equal(appendInstrumentCounterfactuals(dir, [row], undefined, { VOLTRADE_INSTRUMENT_COUNTERFACTUAL: "off" }), 0);
  assert.equal(fs.existsSync(path.join(dir, INSTRUMENT_CF_FILENAME)), false);
  assert.equal(appendInstrumentCounterfactuals(dir, [row], 10, ON), 1);
  assert.equal(appendInstrumentCounterfactuals(dir, [row], 10, ON), 1);
  assert.ok(fs.existsSync(path.join(dir, INSTRUMENT_CF_FILENAME + ".1")), "above maxBytes the file moved to .1");
  const notADir = path.join(dir, "file");
  fs.writeFileSync(notADir, "x");
  const orig = console.error;
  const errs: unknown[] = [];
  console.error = (...a: unknown[]) => { errs.push(a); };
  try { assert.equal(appendInstrumentCounterfactuals(notADir, [row], undefined, ON), 0); } finally { console.error = orig; }
  assert.equal(errs.length, 1, "logged, never thrown");
});

// ── reading: decisions joined to the job's rows ──────────────────────────────
const DEC = (id: string, ticker: string, ts: string, choice: string, notional: number) =>
  ({ v: 1, kind: "decision", id, ts, entry_date: ts.slice(0, 10), ticker, mode: "live", choice, chosen: { instrument: choice === "option_over_stock" ? "option" : "stock", notional }, rejected: {}, summary: `${ticker} decision` });
const OUT = (id: string, h: string, c: number | null, r: number | null, proxy = true) =>
  ({ v: 1, kind: "outcome", cf_id: id, horizon: h, end_date: "2026-11-09", proxy,
    chosen: c === null ? { priced: false, reason: "no bars" } : { priced: true, pct: c, basis: "price" },
    rejected: r === null ? { priced: false, reason: "no listed call" } : { priced: true, pct: r, basis: "premium" },
    diff_pp: c !== null && r !== null ? Math.round((c - r) * 1000) / 1000 : null, line: `${id} ${h} line` });

test("fold + read: both generations, outcomes and the proxy joined, malformed lines skipped, re-read when the file changes", () => {
  const folded = foldCounterfactuals([OUT("d1", "+1d", 1, 10), { kind: "proxy", cf_id: "d1", contract: { symbol: "AMD261211C00145000" }, reason: null },
    DEC("d1", "AMD", "2026-11-02T15:30:00.000Z", "stock_over_option", 1802.4), OUT("d1", "+1d", 99, 99), OUT("zz", "+1d", 1, 1),
    { kind: "mystery" }, OUT("d1", "+7d", 1, 1)]);
  assert.equal(folded.length, 1);
  assert.deepEqual(Object.keys(folded[0].outcomes), ["+1d"]);
  assert.equal(folded[0].outcomes["+1d"]?.chosen.pct, 1, "the first outcome row per horizon wins");
  assert.equal(folded[0].outcomes["+1d"]?.diffPp, -9);
  assert.deepEqual(folded[0].proxy, { symbol: "AMD261211C00145000" });
  const dir = tmpDir();
  const p = path.join(dir, INSTRUMENT_CF_FILENAME);
  fs.writeFileSync(p + ".1", JSON.stringify(DEC("d1", "AMD", "2026-11-02T15:30:00.000Z", "stock_over_option", 1802.4)) + "\n");
  fs.writeFileSync(p, `${JSON.stringify(OUT("d1", "+5d", 2.1, 38))}\n{bad json\n`);
  const orig = console.error;
  const errs: unknown[] = [];
  console.error = (...a: unknown[]) => { errs.push(a); };
  let first;
  try { first = readInstrumentCounterfactuals(dir); } finally { console.error = orig; }
  assert.equal(errs.length, 1, "the malformed line is counted in the log");
  assert.equal(first[0].outcomes["+5d"]?.line, "d1 +5d line", "the decision in .1, its outcome in the current file");
  assert.equal(readInstrumentCounterfactuals(dir), first, "unchanged files: the cached fold");
  fs.appendFileSync(p, JSON.stringify(OUT("d1", "+20d", 3, 1)) + "\n");
  assert.ok(readInstrumentCounterfactuals(dir)[0].outcomes["+20d"], "a changed file is re-read");
  assert.deepEqual(readInstrumentCounterfactuals(tmpDir()), []);
});

test("the journal read path: an entry row's would_be gains its settled outcomes and the resolved proxy", () => {
  const dir = tmpDir();
  const row = entryDecision(TRADE, stockOutcome(), T0) as DecisionRow;
  appendDecisions(dir, [row], undefined, ON);
  appendInstrumentCounterfactuals(dir, [row], undefined, ON);
  fs.appendFileSync(path.join(dir, INSTRUMENT_CF_FILENAME), [
    { v: 1, kind: "proxy", cf_id: row.id, contract: { symbol: "AMD261211C00145000", strike: 145, expiry: "2026-12-11" }, reason: null },
    { ...OUT(row.id, "+5d", 2.1, 38), line: "chose stock: +2.1%; the ATM call proxy it passed over would have made +38.0%", winner: "rejected" },
  ].map((r) => JSON.stringify(r) + "\n").join(""));
  const got = readDecisions(dir)[0];
  const wb = got.rejected.find((x) => x.instrument === "option")?.would_be as Record<string, unknown>;
  assert.deepEqual(wb.outcomes, { "+5d": { line: "chose stock: +2.1%; the ATM call proxy it passed over would have made +38.0%",
    chosen_pct: 2.1, rejected_pct: 38, diff_pp: -35.9, winner: "rejected", end_date: "2026-11-09" } });
  assert.deepEqual(wb.proxy_contract, { symbol: "AMD261211C00145000", strike: 145, expiry: "2026-12-11" });
  assert.equal(wb.strategy, "sell_cash_secured_put", "the would-be entry itself is kept");
  const plain = [normalizeDecision({ action: "skip", ticker: "X" }) as DecisionRow];
  assert.equal(attachCounterfactualOutcomes(plain, dir), plain, "rows without a would_be pass through");
});

// ── the aggregate: byInstrumentChoice ────────────────────────────────────────
const NOW = Date.parse("2026-11-06T21:00:00Z");
function choices() {
  return foldCounterfactuals([
    DEC("d1", "AMD", "2026-11-02T15:30:00.000Z", "stock_over_option", 1802.4), OUT("d1", "+1d", 1, 10), OUT("d1", "+5d", 2.1, 38),
    DEC("d2", "NVDA", "2026-11-03T15:30:00.000Z", "stock_over_option", 1000), OUT("d2", "+1d", -1, -20),
    DEC("d3", "KO", "2026-11-03T16:00:00.000Z", "stock_over_option", 900), OUT("d3", "+1d", 0.5, null),
    DEC("d4", "TLT", "2026-11-02T16:00:00.000Z", "option_over_stock", 8500), OUT("d4", "+1d", 0.25, 0.55, false),
    DEC("d5", "XOM", "2026-09-01T15:30:00.000Z", "stock_over_option", 5000), OUT("d5", "+1d", 50, -50),
  ]);
}

test("byInstrumentChoice: per choice and horizon — means, the edge, equal-capital dollars, counts, plain-English reasons; the window", () => {
  const b = instrumentChoiceBuckets(choices(), NOW - 30 * 86_400_000, NOW);
  assert.deepEqual(b.map((x) => [x.choice, x.horizon, x.decisions, x.n, x.pending, x.unpriced]), [
    ["stock over options", "+1d", 3, 2, 0, 1], ["stock over options", "+5d", 3, 1, 2, 0], ["stock over options", "+20d", 3, 0, 3, 0],
    ["options over stock", "+1d", 1, 1, 0, 0], ["options over stock", "+5d", 1, 0, 1, 0], ["options over stock", "+20d", 1, 0, 1, 0],
  ], "XOM is outside the window; no ETF decision, no ETF row");
  const s1 = b[0];
  assert.deepEqual([s1.chosenPct, s1.rejectedPct, s1.edgePct, s1.edgeDollars, s1.chosenBetter, s1.rejectedBetter], [0, -5, 5, 27.78, 1, 1]);
  assert.equal(s1.reason, "Stock over options at 1 trading day: 2 decisions settled (1 could not be priced). The stock bought averaged +0.0%; "
    + "the option alternatives passed over (ATM option proxies) averaged -5.0% — choosing stock gained 5.0 points on average; "
    + "at equal capital the choice made $27.78; stock did better in 1 of 2.");
  assert.deepEqual(s1.examples.map((e) => [e.ticker, e.diffPp, e.line]), [["NVDA", 19, "d2 +1d line"], ["AMD", -9, "d1 +1d line"]], "largest difference first");
  assert.equal(b[1].reason, "Stock over options at 5 trading days: 1 decision settled (2 still pending). The stock bought averaged +2.1%; "
    + "the option alternatives passed over (ATM option proxies) averaged +38.0% — choosing stock cost 35.9 points on average; "
    + "at equal capital the choice cost $647; stock did better in 0 of 1.");
  assert.equal(b[2].reason, "Stock over options: 3 decisions in the window, none settled at 20 trading days yet (3 still pending).");
  assert.deepEqual([b[2].chosenPct, b[2].edgePct, b[2].edgeDollars, b[2].examples], [null, null, null, []]);
  assert.equal(b[3].reason, "Options over stock at 1 trading day: 1 decision settled. The options opened averaged +0.3%; the stock passed over "
    + "averaged +0.6% — choosing options cost 0.3 points on average; at equal capital the choice cost $25.50; options did better in 0 of 1.");
  assert.equal(b[4].reason, "Options over stock: 1 decision in the window, none settled at 5 trading days yet (1 still pending).");
  assert.deepEqual(instrumentChoiceBuckets([], 0, NOW), []);
});

test("attribution: the report carries byInstrumentChoice and a note when the ledger is given; [] without it", () => {
  const base: AttributionInputs = { nowMs: NOW, days: 30, events: [], fills: [], positions: [], decisions: [], orders: null,
    ledger: { rows: [], malformed: 0 }, floorTickers: new Set<string>(), sectors: {} };
  const r = attributionReport({ ...base, instrumentChoices: choices() });
  assert.deepEqual(r.byInstrumentChoice, instrumentChoiceBuckets(choices(), NOW - 30 * 86_400_000, NOW));
  assert.equal(r.sources.instrumentChoices, 5);
  assert.ok(r.notes.some((n) => n.startsWith("Instrument choice is not P&L:") && n.includes("expiring 30-45 days out — a proxy, labelled so")));
  const none = attributionReport(base);
  assert.deepEqual(none.byInstrumentChoice, []);
  assert.equal(none.sources.instrumentChoices, null);
  assert.ok(!none.notes.some((n) => n.startsWith("Instrument choice")));
});

// ── wiring ───────────────────────────────────────────────────────────────────
test("wiring: journalDecisions writes the ledger with the journal; attribution reads it; the nightly job; the heartbeat registry", () => {
  const jd = bot.slice(bot.indexOf("function journalDecisions("), bot.indexOf("// ─── Python Subprocess Serialization"));
  assert.ok(jd.indexOf("appendInstrumentCounterfactuals(decisionsDir(), rows);") > jd.indexOf("appendDecisions(decisionsDir(), rows);"));
  const attr = bot.slice(bot.indexOf("async function attributionBody(daysRaw: unknown)"), bot.indexOf('app.get("/api/bot/attribution", requireOwner'));
  assert.ok(attr.includes("instrumentChoices: readInstrumentCounterfactuals(dir),"));
  const nightly = bot.slice(bot.indexOf("async function checkShadowBackfill()"), bot.indexOf("// [RULE-REVIEW] 2026-10-14 — the EXIT-ALIGNED shadow ruler's nightly job"));
  const slot = nightly.slice(nightly.indexOf("if (nowHour === 22 && !_instrumentCfFilledToday) {"), nightly.indexOf("if (nowHour === 0) {"));
  for (const needle of ["_instrumentCfFilledToday = true;", 'beat("instrument_cf_backfill", "start");',
    'python3 -c "from instrument_counterfactual import backfill; import json; print(json.dumps(backfill(40)))"',
    "execPythonSerialized(", 'audit("INSTRUMENT-CF-BACKFILL", ', 'beat("instrument_cf_backfill", "finish", "ok");',
    'audit("INSTRUMENT-CF-BACKFILL-ERROR", ', 'beat("instrument_cf_backfill", "finish", "error", ']) {
    assert.ok(slot.includes(needle), needle);
  }
  const reset = nightly.slice(nightly.indexOf("if (nowHour === 0) {"));
  assert.ok(reset.slice(0, reset.indexOf("}")).includes("_instrumentCfFilledToday = false;"), "reset at midnight UTC");
  const spec = JOB_REGISTRY.find((j) => j.id === "instrument_cf_backfill");
  assert.ok(spec && spec.kind === "daily" && spec.source === "server/bot.ts");
  assert.ok(journalSrc.includes("const wb = wouldBeFor(t, o);"), "the entry builder attaches the would-be leg");
  assert.ok(journalSrc.includes("return attachCounterfactualOutcomes("), "the journal reader joins the outcomes");
  assert.ok(!/^import (?!type)[^;]*from "\.\/decisionJournal";/m.test(src), "types only from decisionJournal.ts: no runtime import cycle");
});
