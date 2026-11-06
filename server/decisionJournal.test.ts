// DECISION JOURNAL, Node half (2026-11-06, server/decisionJournal.ts; human
// refocus spec §7 "log every signal, decision, and outcome"). The writer, the
// reader and the builders are pure-ish (a temp DATA_DIR) and tested directly;
// the bot.ts call sites live inside the registerBotRoutes closure and run
// against Alpaca, so their wiring is pinned on the source text (the
// entryShadow.test.ts convention). Python half + parity: test_decision_journal.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DECISIONS_FILENAME, DECISIONS_BACKFILL_FILENAME, DECISION_ROW_MAX_BYTES, STAGE_KEYS,
  appendDecisions, readDecisions, decisionsQuery, normalizeDecision, journalEnabled, describeOcc,
  entryDecision, skipDecision, tierDecision, exitDecision, scoreDrivers, rejectedAlternatives,
  resetDecisionDedupe, skipDedupeKey, etDay, ruleText, type DecisionRow,
} from "./decisionJournal";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const CONTRACT = ["ts", "ticker", "action", "instrument", "chosen", "rejected", "stages", "sizing", "summary"];

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "decisions-"));
}
const ON = {} as Record<string, string | undefined>;
const T0 = Date.parse("2026-11-06T15:30:00Z");

const STOCK_TRADE = {
  ticker: "AMD", side: "buy", trade_type: "stock", price: 150.2, shares: 12, score: 71.3, score_threshold: 63,
  scan_regime: "BULL", regime_at_entry: "BULL", sector: "Technology", reasons: ["Momentum strong", "ADX 31"],
  score_breakdown: { momentum: 80, mean_reversion: 40, vrp: 56, squeeze: 50, volume: 60, macro: -2, news: 3, adx: 3, rules_raw: 64 },
  instrument: "stock", instrument_strategy: "buy_stock",
  instrument_reasoning: "CHOSEN: STOCK (buy_stock) | Scores: [stock=63 | etf=0 | options=0] | Stock: score 71.3 | Options ruled out: score 58 < 65",
  instrument_scores: {
    stock: { score: 63, reasoning: "Stock: score 71.3, hold 3d" },
    etf: { score: 0, reasoning: "200MA gate: AMD below 200d MA (-4.0%) [RULED OUT: Below 200d MA — bull leveraged ETFs avoid below-trend entries]" },
    options: null,
  },
  position_value: 1802.4, sizing_reasoning: "Kelly 0.42 x regime 0.8", sizing_scalars: { kelly: 0.42 },
  rules_score: 64, ml_score_raw: 58,
};

// ── contract / writer / reader ───────────────────────────────────────────────
test("normalizeDecision: every contract key and stage key, invalid rows refused", () => {
  const r = normalizeDecision({ action: "SKIP", ticker: "amd", stages: { flows: { pressure: -0.3 }, extra: { a: 1 } }, summary: "  a   b " });
  assert.ok(r);
  for (const k of CONTRACT) assert.ok(k in r, k);
  for (const k of STAGE_KEYS) assert.ok(k in r.stages, k);
  assert.equal(r.action, "skip");
  assert.equal(r.ticker, "AMD");
  assert.equal(r.chosen, "none");
  assert.equal(r.summary, "a b");
  assert.deepEqual(r.stages.flows, { pressure: -0.3 }, "another module's stage is kept as-is");
  assert.deepEqual(r.stages.extra, { a: 1 });
  for (const bad of [{ action: "hold", ticker: "X" }, { action: "buy" }, null, 7]) assert.equal(normalizeDecision(bad), null);
  const nan = normalizeDecision({ action: "buy", ticker: "X", price: Number.NaN, sizing: { a: Number.POSITIVE_INFINITY } });
  assert.equal(nan?.price, undefined);
  assert.deepEqual(nan?.sizing, { a: null });
});

test("append + read: newest first, ticker matches the underlying or the ordered symbol, limit, malformed lines skipped", () => {
  resetDecisionDedupe();
  const dir = tmpDir();
  const rows = [
    { ts: "2026-11-06T14:00:00.000Z", action: "buy", ticker: "TLT", symbol: "TLT261219P00085000", instrument: "option" },
    { ts: "2026-11-06T14:01:00.000Z", action: "buy", ticker: "AMD" },
    { ts: "2026-11-06T14:02:00.000Z", action: "sell", ticker: "AMD" },
  ];
  assert.equal(appendDecisions(dir, rows, undefined, ON), 3);
  fs.appendFileSync(path.join(dir, DECISIONS_FILENAME), "{not json\n\n");
  assert.equal(appendDecisions(dir, [{ ts: "2026-11-06T14:03:00.000Z", action: "skip", ticker: "NVDA" }], undefined, ON), 1);
  assert.deepEqual(readDecisions(dir, { limit: 10 }).map((r) => r.ts.slice(11, 16)), ["14:03", "14:02", "14:01", "14:00"]);
  assert.deepEqual(readDecisions(dir, { ticker: "amd" }).map((r) => r.action), ["sell", "buy"]);
  assert.deepEqual(readDecisions(dir, { ticker: "TLT261219P00085000" }).map((r) => r.ticker), ["TLT"]);
  assert.equal(readDecisions(dir, { limit: 2 }).length, 2);
  assert.deepEqual(readDecisions(dir, { actions: ["buy"] }).map((r) => r.ticker), ["AMD", "TLT"]);
  assert.deepEqual(readDecisions(tmpDir()), [], "no journal yet = no rows");
});

test("the reader splits on the newline byte: multi-byte text across 64 KB chunk edges survives", () => {
  resetDecisionDedupe();
  const dir = tmpDir();
  const many = Array.from({ length: 400 }, (_, i) => ({
    ts: new Date(T0 + i * 1000).toISOString(), action: "skip", ticker: `T${i}`, summary: `é—€ ${"ü".repeat(150 + (i % 37))} ✓`,
  }));
  assert.equal(appendDecisions(dir, many, undefined, ON), 400);
  assert.ok(fs.statSync(path.join(dir, DECISIONS_FILENAME)).size > 3 * 65_536);
  const got = readDecisions(dir, { limit: 1000 });
  assert.equal(got.length, 400);
  assert.ok(got.every((r) => r.summary.startsWith("é—€ ") && r.summary.endsWith(" ✓")));
  assert.equal(got[0].ticker, "T399");
  assert.equal(readDecisions(dir, { ticker: "T7" })[0].summary, many[7].summary);
});

test("size cap: above maxBytes the file rotates to .1 and both generations are read", () => {
  resetDecisionDedupe();
  const dir = tmpDir();
  const row = (i: number) => ({ ts: new Date(T0 + i * 1000).toISOString(), action: "skip", ticker: `R${i}`, summary: "x".repeat(500) });
  for (let i = 0; i < 30; i++) appendDecisions(dir, [row(i)], 4_000, ON);
  const p = path.join(dir, DECISIONS_FILENAME);
  assert.ok(fs.existsSync(`${p}.1`), "rotated");
  assert.ok(fs.statSync(p).size <= 4_000 + 2_000, "the live file stays near the cap");
  const got = readDecisions(dir, { limit: 1000 });
  assert.equal(got[0].ticker, "R29");
  assert.ok(got.length > fs.readFileSync(p, "utf8").trim().split("\n").length, "the .1 generation is read too");
});

test("an oversized row is cut to DECISION_ROW_MAX_BYTES and keeps its contract keys", () => {
  resetDecisionDedupe();
  const dir = tmpDir();
  appendDecisions(dir, [{
    action: "buy", ticker: "AMD", summary: "s".repeat(5000),
    rejected: Array.from({ length: 6 }, (_, i) => ({ instrument: `i${i}`, reason: "r".repeat(900) })),
    stages: { selection: { reasons: Array(6).fill("x".repeat(300)), blob: "b".repeat(7000) } },
  }], undefined, ON);
  const raw = fs.readFileSync(path.join(dir, DECISIONS_FILENAME), "utf8");
  assert.ok(Buffer.byteLength(raw) <= DECISION_ROW_MAX_BYTES + 1);
  const r = JSON.parse(raw) as DecisionRow;
  assert.equal(r.truncated, true);
  for (const k of CONTRACT) assert.ok(k in r, k);
});

test("off switch: VOLTRADE_DECISION_JOURNAL=off writes nothing; absent = on", () => {
  const dir = tmpDir();
  assert.equal(journalEnabled({}), true);
  for (const v of ["off", "0", "false", "", "none"]) assert.equal(journalEnabled({ VOLTRADE_DECISION_JOURNAL: v }), false, v);
  assert.equal(appendDecisions(dir, [{ action: "buy", ticker: "AMD" }], undefined, { VOLTRADE_DECISION_JOURNAL: "off" }), 0);
  assert.equal(fs.existsSync(path.join(dir, DECISIONS_FILENAME)), false);
});

test("skips are written once per ticker, gate and ET day (the Python key format)", () => {
  resetDecisionDedupe();
  const dir = tmpDir();
  const a = skipDecision(STOCK_TRADE, "rejected_options_slots", "options slots full (6/6)", "bot.ts executeTrades", T0);
  const b = skipDecision(STOCK_TRADE, "rejected_options_slots", "options slots full (6/6)", "bot.ts executeTrades", T0 + 60_000);
  const nextDay = skipDecision(STOCK_TRADE, "rejected_options_slots", "x", "bot.ts executeTrades", T0 + 86_400_000);
  assert.equal(a?.dedupe, "skip|AMD|rejected_options_slots|2026-11-06");
  assert.equal(skipDedupeKey("AMD", "rejected_score", Date.parse("2026-11-07T03:00:00Z")), "skip|AMD|rejected_score|2026-11-06", "ET day, not UTC");
  assert.equal(etDay(Date.parse("2026-11-07T03:00:00Z")), "2026-11-06");
  assert.equal(appendDecisions(dir, [a, b], undefined, ON), 1);
  assert.equal(appendDecisions(dir, [a], undefined, ON), 0);
  assert.equal(appendDecisions(dir, [nextDay], undefined, ON), 1);
  assert.equal(appendDecisions(dir, [entryDecision(STOCK_TRADE, stockOutcome(), T0), entryDecision(STOCK_TRADE, stockOutcome(), T0)], undefined, ON), 2, "entries are real orders: never deduplicated");
});

test("backfill rows merge into the read by time", () => {
  resetDecisionDedupe();
  const dir = tmpDir();
  appendDecisions(dir, [{ ts: "2026-11-06T14:00:00.000Z", action: "buy", ticker: "AMD" }], undefined, ON);
  fs.writeFileSync(path.join(dir, DECISIONS_BACKFILL_FILENAME), [
    JSON.stringify({ ts: "2026-10-20T14:00:00.000Z", action: "buy", ticker: "AMD", backfilled: true }),
    JSON.stringify({ ts: "2026-11-01T14:00:00.000Z", action: "sell", ticker: "AMD", backfilled: true }),
  ].join("\n") + "\n");
  const got = readDecisions(dir, { ticker: "AMD" });
  assert.deepEqual(got.map((r) => [r.ts.slice(0, 10), r.backfilled ?? false]), [["2026-11-06", false], ["2026-11-01", true], ["2026-10-20", true]]);
});

test("decisionsQuery: ticker sanitized, limit 1..500 default 50", () => {
  assert.deepEqual(decisionsQuery({}), { ticker: null, limit: 50 });
  assert.deepEqual(decisionsQuery({ ticker: " amd;drop ", limit: "9999" }), { ticker: "AMDDROP", limit: 500 });
  assert.deepEqual(decisionsQuery({ ticker: "BRK.B", limit: "0" }), { ticker: "BRK.B", limit: 1 });
});

// ── builders ────────────────────────────────────────────────────────────────
function stockOutcome() {
  return { instrument: "stock" as const, chosen: "buy_stock", side: "buy", symbol: "AMD", qty: 12, price: 150.2, notional: 1802.4,
    status: "accepted", orderId: "ord-1", mode: "live" as const, source: "bot.ts executeTrades.stock", equity: 100_000 };
}

test("entry (stock): the stages, the instrument alternatives and their reasons, sizing, a plain-English summary", () => {
  const r = entryDecision(STOCK_TRADE, stockOutcome(), T0);
  assert.ok(r);
  assert.equal(r.action, "buy");
  assert.equal(r.instrument, "stock");
  assert.equal(r.stages.regime?.label, "BULL");
  assert.equal(r.stages.sector?.name, "Technology");
  assert.equal(r.stages.selection?.score, 71.3);
  assert.equal(r.stages.selection?.threshold, 63);
  assert.deepEqual(r.stages.selection?.dominant, { name: "momentum", label: "momentum", points: 7.5 });
  assert.equal(r.stages.flows, null);
  assert.equal(r.stages.tax, null);
  assert.deepEqual(r.rejected, [
    { instrument: "etf", reason: "ruled out: Below 200d MA — bull leveraged ETFs avoid below-trend entries" },
    { instrument: "option", reason: "ruled out: score 58 < 65" },
  ]);
  assert.equal(r.sizing?.pct_equity, 1.8);
  assert.equal(r.signal, "deep_score scan, momentum-led");
  assert.equal(r.order?.id, "ord-1");
  assert.equal(r.summary,
    "Bought 12 AMD at ~$150.20 ($1,802, 1.8% of equity). Why: score 71.3 cleared the 63 needed in a BULL market; "
    + "biggest driver momentum (+7.5 pts); Momentum strong. Instead of: etf (ruled out: Below 200d MA — bull leveraged ETFs "
    + "avoid below-trend entries); option (ruled out: score 58 < 65). Sizing: Kelly 0.42 x regime 0.8. Order accepted.");
});

test("entry: a lower-scoring alternative is explained by its score; ETF and options entries; shadow mode", () => {
  const t = { ...STOCK_TRADE, instrument_scores: { stock: { score: 63, reasoning: "Stock" }, etf: { score: 41, reasoning: "ETF SSO (2x AMD): hold 3d" }, options: null } };
  assert.deepEqual(rejectedAlternatives(t, "stock")[0], { instrument: "etf", reason: "scored 41 vs 63 for the stock (ETF SSO (2x AMD): hold 3d)" });
  const etf = entryDecision({ ...t, instrument: "etf", instrument_ticker: "SSO", instrument_scores: { stock: { score: 50, reasoning: "Stock" }, etf: { score: 72, reasoning: "ETF" } } },
    { ...stockOutcome(), instrument: "etf", chosen: "buy_etf_bull", symbol: "SSO", qty: 20, price: 95.1, notional: 1902 }, T0);
  assert.equal(etf?.instrument, "etf");
  assert.match(etf?.summary ?? "", /^Bought 20 SSO \(a leveraged ETF on AMD\) at ~\$95\.10/);
  assert.deepEqual(etf?.rejected[0], { instrument: "stock", reason: "scored 50 vs 72 for the etf (Stock)" });
  const opt = entryDecision({ ticker: "TLT", trade_type: "options", regime_at_entry: "OPTIONS_SCANNER", setup: "iv_crush", options_strategy: "sell_cash_secured_put", score: 77, instrument_scores: {} },
    { instrument: "option", chosen: "sell_cash_secured_put", side: "sell", symbol: "TLT261219P00085000", qty: 1, price: 1.2, notional: 8500,
      status: "submitted", orderId: "o2", mode: "live", source: "bot.ts executeTrades.options", equity: 100_000,
      contract: { occ_symbol: "TLT261219P00085000", limit_price: 1.2, cash_required: 8500, delta: -0.3 } }, T0);
  assert.equal(opt?.instrument, "option");
  assert.equal(opt?.signal, "options scanner: iv_crush");
  assert.equal(opt?.stages.regime, null, "the scanner marker is not a regime");
  assert.equal(opt?.stages.options?.contract, "TLT $85 put expiring 2026-12-19");
  assert.equal(opt?.stages.options?.cash_required, 8500);
  assert.deepEqual(opt?.rejected, [], "no selector ran: no invented alternatives");
  assert.match(opt?.summary ?? "", /^Opened a sell cash secured put on TLT: sold 1 TLT \$85 put expiring 2026-12-19 at a \$1\.20 limit\./);
  const sh = entryDecision(STOCK_TRADE, { ...stockOutcome(), mode: "shadow", status: "shadowed" }, T0);
  assert.equal(sh?.mode, "shadow");
  assert.match(sh?.summary ?? "", /^Shadow mode, no order placed: Bought 12 AMD/);
  assert.doesNotMatch(sh?.summary ?? "", /Order /);
});

test("entry: the stock bought after the ETF or options path fell through says so", () => {
  const r = entryDecision({ ...STOCK_TRADE, instrument: "etf", instrument_ticker: "SSO", use_options: true, options_strategy: "buy_call",
    instrument_scores: { stock: { score: 50, reasoning: "Stock" }, etf: { score: 72, reasoning: "ETF" }, options: { score: 60, reasoning: "Options" } } }, stockOutcome(), T0);
  assert.equal(r?.rejected[0].instrument, "etf");
  assert.match(r?.rejected[0].reason ?? "", /chose the leveraged ETF SSO, but its order could not be planned or placed/);
  assert.equal(r?.rejected[1].instrument, "option");
  assert.match(r?.rejected[1].reason ?? "", /the buy call path declined at execution or its order failed/);
  assert.equal(r?.rejected.length, 2, "the fallback replaces that instrument's score line");
});

test("finished agents' inputs: trade.flows is the flows stage; the scan's risk_budget summary is the regime reason", () => {
  const flows = { applied: true, multiplier: 1.08, sector: "Technology", etf: "XLK", pressure: 0.4, reason: "Technology: expected net buying — size x1.08", label: "ESTIMATE" };
  const budget = { budget: 0.62, band: "reduced", summary: "Risk budget 0.62 (reduced): VIX 22 above its 1-year median.", drivers: ["VIX 22.1"], applied: true, stale: false };
  const r = entryDecision({ ...STOCK_TRADE, flows, risk_budget: budget }, stockOutcome(), T0);
  assert.deepEqual(r?.stages.flows, flows);
  assert.equal(r?.stages.regime?.reason, "Risk budget 0.62 (reduced): VIX 22 above its 1-year median. (BULL market)");
  assert.equal((r?.stages.regime?.risk_budget as { budget?: number } | undefined)?.budget, 0.62);
  const sk = skipDecision({ ...STOCK_TRADE, flows, risk_budget: budget }, "rejected_size_cap", "x", "bot.ts executeTrades", T0);
  assert.equal(sk?.stages.flows?.etf, "XLK");
  assert.equal((sk?.stages.regime?.risk_budget as { band?: string } | undefined)?.band, "reduced");
  const tier = tierDecision({ tier: 1, action: "SELL_CSP", ticker: "TLT", reason: "T1", metadata: { regime_label: "BULL" }, risk_budget: budget },
    { status: "submitted", occ_symbol: "TLT261219P00085000", qty: 1 }, null, T0);
  assert.equal(tier?.stages.regime?.reason, "Risk budget 0.62 (reduced): VIX 22 above its 1-year median. (BULL market)");
  const noBudget = entryDecision({ ...STOCK_TRADE, risk_budget: { error: "regime engine off" } }, stockOutcome(), T0);
  assert.equal(noBudget?.stages.regime?.reason, "BULL market", "an errored budget is ignored, never shown");
  assert.equal(entryDecision(STOCK_TRADE, stockOutcome(), T0)?.stages.flows, null);
});

test("skip (executeTrades gate): plain-English rule, the proposed trade, its stages", () => {
  const r = skipDecision(STOCK_TRADE, "rejected_exposure_cap", "total exposure would exceed 95% — loop ended", "bot.ts executeTrades", T0);
  assert.equal(r?.action, "skip");
  assert.equal(r?.chosen, "none");
  assert.equal(r?.summary, "Skipped AMD: total exposure would have gone over the cap (total exposure would exceed 95% — loop ended).");
  assert.equal(r?.rejected[0].reason, "a stock buy was refused because total exposure would have gone over the cap: total exposure would exceed 95% — loop ended");
  assert.equal(r?.stages.selection?.score, 71.3);
  assert.equal(r?.rule, "rejected_exposure_cap");
  assert.equal(ruleText("rejected_regime_knife"), "the regime guard 'knife' blocked it");
  assert.equal(ruleText("rejected_brand_new"), "brand new");
});

test("tier dispatcher: SELL_CSP submitted / shadowed / failed, the Tier-3 BUY", () => {
  const action = { tier: 1, action: "SELL_CSP", ticker: "TLT", strategy: "csp_core", size_pct: 0.05,
    reason: "T1 CSP harvest (regime=BULL, vxx=0.95, DD=-3.1%, size=85%)", metadata: { regime_label: "BULL", vxx_ratio: 0.95, target_delta: 0.3, target_dte_min: 30, target_dte_max: 45 } };
  const ok = tierDecision(action, { status: "submitted", order_id: "o9", occ_symbol: "TLT261219P00085000", qty: 1, limit_price: 1.2, side: "sell", cash_required: 8500 }, null, T0);
  assert.equal(ok?.action, "buy");
  assert.equal(ok?.chosen, "sell_cash_secured_put");
  assert.equal(ok?.side, "sell");
  assert.equal(ok?.stages.regime?.label, "BULL");
  assert.equal(ok?.signal, "tier 1 CSP harvest");
  assert.equal(ok?.summary, "Sold 1 TLT $85 put expiring 2026-12-19 at a $1.20 limit (cash-secured: $8,500 set aside). Why: T1 CSP harvest (regime=BULL, vxx=0.95, DD=-3.1%, size=85%). Order submitted.");
  const sh = tierDecision(action, { status: "shadowed", occ_symbol: "TLT261219P00085000", qty: 1 }, null, T0);
  assert.equal(sh?.mode, "shadow");
  const bad = tierDecision(action, { status: "error", reason: "Insufficient budget for 1 contract" }, null, T0);
  assert.equal(bad?.action, "skip");
  assert.equal(bad?.rule, "rejected_order");
  assert.match(bad?.summary ?? "", /could not be placed \(Insufficient budget for 1 contract\)/);
  assert.equal(bad?.dedupe, "skip|TLT|rejected_order:SELL_CSP|2026-11-06");
  const t3 = tierDecision({ tier: 3, action: "BUY", ticker: "SPY", size_pct: 0.1, reason: "T3 trend" }, { status: "accepted", order_id: "o3" }, { qty: 10, price: 580.1 }, T0);
  assert.equal(t3?.instrument, "stock");
  assert.equal(t3?.notional, 5801);
  assert.equal(t3?.summary, "Bought 10 SPY at ~$580.10. Why: T3 trend. Order accepted.");
  assert.equal(tierDecision({}, {}, null, T0), null);
});

test("exits: WS exit, scale-out, POS-KILL on an option", () => {
  const ws = exitDecision({ ticker: "AMD", exitType: "trailing_stop", reason: "WS TRAILING STOP: $147.10 < stop $147.50 (Phase 3)", side: "sell", qty: 12,
    price: 147.1, pnlPct: -2.04, entryDate: "2026-11-03T15:00:00Z", orderId: "x1", status: "accepted", source: "bot.ts WS exit" }, T0);
  assert.equal(ws?.action, "sell");
  assert.equal(ws?.chosen, "trailing_stop");
  assert.equal(ws?.summary, "Sold 12 AMD at ~$147.10: the trailing stop was hit. The position was down 2% at the decision. Held since 2026-11-03. WS TRAILING STOP: $147.10 < stop $147.50 (Phase 3). Order accepted.");
  const so = exitDecision({ ticker: "AMD", exitType: "scale_out", reason: "SCALE 1/3 at +1R", side: "sell", qty: 4, price: 155, pnlPct: 3.2, partial: true, source: "bot.ts WS scale-out" }, T0);
  assert.equal(so?.action, "scale_out");
  const pk = exitDecision({ ticker: "IREN261204P00020000", exitType: "position_kill", reason: "down 52.0% against the -50% backstop", side: "sell", qty: 3,
    price: 1.1, pnlPct: -52, source: "bot.ts POS-KILL", instrument: "option" }, T0);
  assert.equal(pk?.ticker, "IREN");
  assert.equal(pk?.instrument, "option");
  assert.equal(pk?.symbol, "IREN261204P00020000");
  assert.equal(pk?.order?.status, "decided", "the decision row is written before the order's outcome is known");
  assert.match(pk?.summary ?? "", /^Sold IREN \$20 put expiring 2026-12-04 at ~\$1\.10: the position-loss backstop/);
});

test("builders never throw into an order path (a hostile trade dict returns null)", () => {
  const hostile = new Proxy({}, { get() { throw new Error("boom"); } });
  const errors: unknown[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errors.push(a); };
  try {
    assert.equal(entryDecision(hostile, stockOutcome(), T0), null);
    assert.equal(skipDecision(hostile, "rejected_score", "x"), null);
    assert.equal(tierDecision(hostile, hostile, null), null);
  } finally {
    console.error = orig;
  }
  assert.ok(errors.length >= 3, "each failure is logged, not silent");
});

test("drivers: points in the rules score vs a neutral 50, adjustments as added", () => {
  const d = Object.fromEntries(scoreDrivers(STOCK_TRADE.score_breakdown).map((x) => [x.name, x.points]));
  assert.deepEqual(d, { momentum: 7.5, mean_reversion: -2, vrp: 1.5, squeeze: 0, volume: 1.5, macro: -2, news: 3, adx: 3 });
  assert.equal(describeOcc("SPY"), "SPY");
});

// ── wiring (bot.ts) ─────────────────────────────────────────────────────────
function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

test("wiring: the owner route and the diag probe serve the same reader", () => {
  assert.match(bot, /app\.get\("\/api\/bot\/decisions", requireOwner, \(req, res\) => \{\s*res\.json\(readDecisions\(decisionsDir\(\), decisionsQuery\(req\.query as Record<string, unknown>\)\)\);/);
  const probe = slice('case "decisions": {', 'case "option_bars": {');
  assert.ok(probe.includes("readDecisions(decisionsDir()"));
  assert.ok(probe.includes("sanitizeDiag(r)"));
  assert.ok(probe.includes("Math.min(q.limit ?? 50, 200)"));
});

test("wiring: executeTrades journals every entry path and every gate refusal, in one write after the loop", () => {
  const et = slice("async function executeTrades(", "// ── Tier 3: Strategic");
  const note = et.slice(et.indexOf("const noteGate = "), et.indexOf("const noteRemaining = "));
  assert.ok(note.includes('if (decision !== "rejected_held") decisionRows.push(skipDecision(t, decision, reason));'));
  for (const src of ["bot.ts executeTrades.etf", "bot.ts executeTrades.options", "bot.ts executeTrades.stock"]) {
    const n = et.split(`source: "${src}"`).length - 1;
    assert.equal(n, 2, `${src}: a live and a shadow entry row`);
  }
  assert.ok(et.indexOf("journalDecisions(decisionRows);") > et.indexOf("await flushShadowEntries(shadowEntries, \"executeTrades\");"));
});

test("wiring: the scan's risk_budget rides each trade and tier action into executeTrades / the dispatcher", () => {
  const line = 'for (const t of [...(result.new_trades || []), ...(result.tier_actions || [])]) if (t && typeof t === "object" && !("risk_budget" in t)) t.risk_budget = result.risk_budget ?? null;';
  const at = bot.indexOf(line);
  assert.ok(at > 0, "the hand-off line is missing");
  assert.ok(bot.indexOf("await executeTrades(result.new_trades || [], equity);", at) - at < 300, "right before executeTrades");
});

test("wiring: the tier dispatcher journals SELL_CSP, BUY_PUT and Tier-3 BUY outcomes", () => {
  const td = slice("const tierDecisionRows: Array<DecisionRow | null> = [];", "journalDecisions(tierDecisionRows);");
  assert.equal(td.split("tierDecisionRows.push(tierDecision(action, r));").length - 1, 2, "SELL_CSP and BUY_PUT results");
  assert.equal(td.split("tierDecisionRows.push(tierDecision(action, {").length - 1, 4, "T3: shadowed, submitted, failed, skipped");
});

test("wiring: stock exits are journaled; the per-position loop's calls are typeof-guarded (harnesses run it standalone)", () => {
  assert.ok(bot.includes('journalDecisions([exitDecision({ ticker, exitType, reason, side: exitSide, qty: exitQty'));
  assert.ok(bot.includes('journalDecisions([exitDecision({ ticker, exitType: "scale_out"'));
  const loop = slice("async function syncMonitoredPositions()", "// Read stop state for this ticker");
  for (const t of ["position_kill", "time_exit"]) {
    assert.ok(loop.includes(`if (typeof journalDecisions === "function") journalDecisions([exitDecision({ ticker, exitType: "${t}"`), t);
  }
  assert.ok(!/(?<!typeof )journalDecisions\(\[/.test(loop.replace(/if \(typeof journalDecisions === "function"\) journalDecisions\(\[/g, "")), "no unguarded call in the loop");
  const helper = slice("function journalDecisions(rows: ReadonlyArray<DecisionRow | null>): void {", "// ─── Python Subprocess Serialization");
  assert.ok(helper.includes("try {") && helper.includes("console.error"), "the helper never throws");
});

test("wiring: the one-time backfill runs once the sleeve ledger is warm, unref'd, and audits its outcome", () => {
  const bf = slice("async function runDecisionBackfill(attempt = 1): Promise<void> {", 'case "decisions"');
  assert.ok(bf.includes("if (backfillDone(dir)) return;"));
  assert.ok(bf.includes("sleeveCache.taxInputs.events") && bf.includes("sleeveCache.taxInputs.fills"));
  assert.ok(bf.includes('audit("DECISION-BACKFILL", backfillAuditLine(built, writeBackfill(dir, built.rows)));'));
  assert.ok(bf.includes("DECISION_BACKFILL_DELAY_MS).unref();"));
});
