// ONE-TIME DECISION-JOURNAL BACKFILL (2026-11-06, server/decisionBackfill.ts).
// Pins: the option structure each leg set forms; when each position opened
// (closed lots + still-open positions); the audit-message parsers — and that
// bot.ts still WRITES those formats, so a format change fails here instead of
// silently matching nothing; the build (floor left out, the 30-day window,
// open positions whatever their age, legs grouped into one structure row,
// reasons from the audit log or "not recorded", every row labelled backfilled);
// the write-once marker.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  inferOptionStructure, openings, parseTradeAudit, parseInstrumentAudit, parseTierAudit, parseOptionsAudit, parseExitAudit,
  buildBackfill, writeBackfill, backfillDone, backfillAuditLine, BACKFILL_AUDIT_TYPES, BACKFILL_WINDOW_DAYS, type AuditLite,
} from "./decisionBackfill";
import { DECISIONS_BACKFILL_FILENAME, readDecisions } from "./decisionJournal";
import type { RealizedEvent } from "./sleeveLedger";
import type { FillRow, PositionRow } from "./killSwitchAutoResume";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const NOW = Date.parse("2026-11-06T21:00:00Z");
const DAY = 86_400_000;
const FLOOR = new Set(["SPY", "SSO", "QQQ", "GLD", "SVXY"]);

test("structure inference from the legs that opened together", () => {
  const s = (x: Array<[string, boolean]>) => inferOptionStructure(x.map(([symbol, short]) => ({ symbol, short })));
  assert.equal(s([["TLT261219P00085000", true]]), "short put (cash-secured)");
  assert.equal(s([["TLT261219C00095000", true]]), "short call");
  assert.equal(s([["TLT261219P00085000", false]]), "long put");
  assert.equal(s([["IREN261204P00021000", true], ["IREN261204P00020000", false]]), "put credit spread");
  assert.equal(s([["IREN261204P00021000", false], ["IREN261204P00020000", true]]), "put debit spread");
  assert.equal(s([["X261204C00023000", true], ["X261204C00024000", false]]), "call credit spread");
  assert.equal(s([["X261204C00024000", true], ["X261204C00023000", false]]), "call debit spread");
  assert.equal(s([["X261204P00020000", false], ["X261204P00021000", true], ["X261204C00023000", true], ["X261204C00024000", false]]), "iron condor");
  assert.equal(s([["X261204P00020000", true], ["X261204C00020000", true]]), "short straddle");
  assert.equal(s([["X261204P00019000", false], ["X261204C00021000", false]]), "long strangle");
  assert.equal(s([["X261204P00020000", true], ["X270115P00020000", false]]), "calendar spread");
  assert.equal(s([["X261204P00020000", false], ["X261204P00021000", true], ["X261204C00023000", true]]), "3-leg structure");
  assert.equal(s([["AAPL", true]]), "option");
});

const ev = (symbol: string, openedAt: number | null, t: number, qty: number, basis: number, pnl: number, closeDir: 1 | -1): RealizedEvent => ({ symbol, openedAt, t, qty, basis, pnl, closeDir });
const fill = (symbol: string, t: number, dir: 1 | -1, qty: number, price: number): FillRow => ({ symbol, t, dir, qty, price });

test("openings: closed lots keyed by symbol + open time (scale-outs summed), plus the still-open positions' opening fill", () => {
  const o1 = NOW - 3 * DAY;
  const events = [ev("AMD", o1, NOW - 2 * DAY, 4, 600, 20, -1), ev("AMD", o1, NOW - DAY, 8, 1200, -40, -1), ev("TLT261219P00085000", o1, NOW - DAY, 1, 120, 60, 1)];
  const fills = [fill("NVDA", NOW - 40 * DAY, 1, 5, 100), fill("NVDA", NOW - 30 * DAY, -1, 5, 110), fill("NVDA", NOW - 5 * DAY, 1, 3, 120), fill("NVDA", NOW - 4 * DAY, 1, 2, 125)];
  const positions: PositionRow[] = [{ symbol: "NVDA", qty: 5, avgEntry: 122, unrealizedPl: 15, option: false }];
  const o = openings(events, fills, positions);
  const amd = o.find((x) => x.symbol === "AMD")!;
  assert.deepEqual([amd.qty, amd.price, amd.short, amd.open], [12, 150, false, false]);
  const put = o.find((x) => x.symbol === "TLT261219P00085000")!;
  assert.deepEqual([put.price, put.short], [1.2, true], "a credit lot: sold at basis / (qty x 100)");
  const nv = o.find((x) => x.symbol === "NVDA")!;
  assert.deepEqual([nv.t, nv.price, nv.open], [NOW - 5 * DAY, 120, true], "the last flat-to-long crossing, not the first fill ever");
});

// The exact templates bot.ts writes (pinned below against bot.ts itself).
const A = (type: string, t: number, message: string): AuditLite => ({ type, time: new Date(t).toISOString(), message });

test("parsers: the message formats bot.ts writes", () => {
  assert.deepEqual(parseTradeAudit("LIMIT BUY 12 AMD @ ~$150.2 | Score: 71.3 | Rules: 68 ML: 62 Blend: 71.3 | $1802.4 (1.8%) | Sizing: Kelly 0.42"),
    { ticker: "AMD", score: 71.3, rules: "rules 68, ML 62", sizing: "Kelly 0.42" });
  assert.deepEqual(parseTradeAudit("MARKET BUY 3 KO @ ~$61 | Score: 66 | $183 (0.2%)"), { ticker: "KO", score: 66, rules: null, sizing: null });
  assert.equal(parseTradeAudit("BUY 3 KO @ market"), null, "the manual-order TRADE row is not a scan entry");
  assert.deepEqual(parseInstrumentAudit("AMD → STOCK chosen | scores: stock=63 etf=41 options=N/A | CHOSEN: STOCK (buy_stock)"),
    { ticker: "AMD", chosen: "stock", scores: { stock: "63", etf: "41", options: "N/A" }, reasoning: "CHOSEN: STOCK (buy_stock)" });
  assert.deepEqual(parseTierAudit("T1", "SELL_CSP TLT | T1 CSP harvest (regime=BULL, vxx=0.95, DD=-3.1%, size=85%)"),
    { kind: "SELL_CSP", ticker: "TLT", reason: "T1 CSP harvest (regime=BULL, vxx=0.95, DD=-3.1%, size=85%)" });
  assert.deepEqual(parseTierAudit("T4", "BUY_PUT SPY hedge | T4 tail hedge"), { kind: "BUY_PUT", ticker: "SPY", reason: "T4 tail hedge" });
  assert.deepEqual(parseTierAudit("T3", "BUY 10 SPY @ ~$580.10 | T3 trend"), { kind: "BUY", ticker: "SPY", reason: "T3 trend" });
  assert.deepEqual(parseOptionsAudit("OPTIONS-TRADE", "SELL_CASH_SECURED_PUT TLT | TLT261219P00085000 | VRP 6%"),
    { ticker: "TLT", strategy: "sell_cash_secured_put", occ: "TLT261219P00085000", reason: "VRP 6%", ok: true });
  assert.deepEqual(parseOptionsAudit("OPTIONS-SCANNER-EXEC", "IREN | strategy=iron_condor | setup=iv_crush | result=accepted"),
    { ticker: "IREN", strategy: "iron_condor", setup: "iv_crush", reason: "options scanner setup iv_crush", ok: true });
  assert.equal(parseOptionsAudit("OPTIONS-SCANNER-EXEC", "IREN | strategy=x | setup=y | result=no_order")?.ok, false);
  assert.deepEqual(parseExitAudit("WS-EXIT", "WS TRAILING STOP: $147.10 < stop $147.50 (Phase 3) | LIMIT sell 12 AMD @ $147.10"),
    { ticker: "AMD", reason: "WS TRAILING STOP: $147.10 < stop $147.50 (Phase 3)", exitType: "trailing_stop" });
  assert.equal(parseExitAudit("WS-EXIT", "WS TAKE PROFIT: +6.1% hit target +6.0% (Phase 2) | LIMIT sell 4 AMD @ $159.30")?.exitType, "take_profit");
  assert.equal(parseExitAudit("WS-EXIT", "AMD: skipping — 1 open sell orders already exist (12 shares held)"), null);
  assert.deepEqual(parseExitAudit("WS-SCALE-OUT", "SCALE 1/3 at +1R | LIMIT sell 4 of 12 AMD @ $155.00 (8 remaining)"),
    { ticker: "AMD", reason: "SCALE 1/3 at +1R", exitType: "scale_out" });
  assert.deepEqual(parseExitAudit("POS-KILL", "IREN261204P00020000: position down -52.0% — forcing liquidation (threshold: -50%)"),
    { ticker: "IREN261204P00020000", reason: "position down -52.0% — forcing liquidation (threshold: -50%)", exitType: "position_kill" });
  assert.deepEqual(parseExitAudit("TIME-EXIT", "KO: 15d old, pnl 0.4% — closing stale"), { ticker: "KO", reason: "15d old, pnl 0.4% — closing stale", exitType: "time_exit" });
  assert.deepEqual(parseExitAudit("OPTIONS-EXIT", "TLT sell_cash_secured_put: PROFIT TARGET: 52% of max profit reached · order submitted"),
    { ticker: "TLT", reason: "PROFIT TARGET: 52% of max profit reached", exitType: "options_exit" });
  assert.equal(parseExitAudit("OPTIONS-EXIT", "… 3 more options exits/rolls this scan"), null);
  assert.equal(parseExitAudit("OPTIONS-EXIT", "GOOGL GOOGL261127P00300000: 21 DTE EXIT: 21 DTE — deferred: market closed · order deferred (none)"), null, "a deferred close sent nothing");
  assert.deepEqual(parseExitAudit("OPTIONS-EXIT", "IREN IREN261211P00018000: LOSS LIMIT: Cost to close $3.60 is 2.1x the credit received $1.68 — capping loss at 2.0x · order submitted"),
    { ticker: "IREN", reason: "LOSS LIMIT: Cost to close $3.60 is 2.1x the credit received $1.68 — capping loss at 2.0x", exitType: "options_exit" }, "a live row (2026-11-05)");
  assert.deepEqual(parseTierAudit("T1", "SELL_CSP FIS | T1 CSP harvest (regime=NEUTRAL, vxx=1.00, DD=-16.1%, size=38%) [scaled 59% by cap enforcer]")?.ticker, "FIS");
  assert.deepEqual(parseOptionsAudit("OPTIONS-TRADE", "IRON_CONDOR IREN |  | Scanner direct execution"),
    { ticker: "IREN", strategy: "iron_condor", occ: undefined, reason: "Scanner direct execution", ok: true }, "a live row: no OCC for a multi-leg open");
});

test("bot.ts still writes the formats the parsers read", () => {
  for (const tpl of [
    'audit("TRADE", `${orderParams.type.toUpperCase()} ${side.toUpperCase()} ${qty} ${trade.ticker} @ ~$${trade.price} | Score: ${trade.score}${scoreAttrib} | $${trade.position_value} (',
    "const scoreAttrib = trade.rules_score != null ? ` | Rules: ${trade.rules_score} ML: ${trade.ml_score_raw ?? 'N/A'} Blend: ${trade.score}` : \"\";",
    "audit(\"INSTRUMENT\", `${trade.ticker} → ${(trade.instrument || 'stock').toUpperCase()} chosen | scores: stock=${sScore} etf=${eScore} options=${oScore} | ",
    'leveraged ETF for ${trade.ticker}, sizing=',
    "audit(\"OPTIONS-TRADE\", `${(trade.options_strategy || '').toUpperCase()} ${trade.ticker} | ${optExec.contract?.occ_symbol || ''} | ",
    "audit(\"OPTIONS-SCANNER-EXEC\", `${trade.ticker} | strategy=${trade.options_strategy} | setup=${trade.setup || '?'} | result=",
    'audit("T" + action.tier, `SELL_CSP ${action.ticker} | ${action.reason}`);',
    'audit("T4", `BUY_PUT ${action.ticker} hedge | ${action.reason}`);',
    'audit("T3", `BUY ${shares} ${action.ticker} @ ~$${currentPrice.toFixed(2)} | ${action.reason}`);',
    'audit("WS-EXIT", `${reason} | ${orderParams.type.toUpperCase()} ${exitSide} ${exitQty} ${ticker} @ $${currentPrice.toFixed(2)}`);',
    'audit("WS-SCALE-OUT", `${scaleOutLabel} | ${orderParams.type.toUpperCase()} ${exitSide} ${scaleOutQty} of ${pos.originalQty} ${ticker} @ $${currentPrice.toFixed(2)} (',
    'audit("POS-KILL", `${ticker}: position down ${pnlPct.toFixed(1)}% — forcing liquidation',
    'audit("TIME-EXIT", `${ticker}: ${Math.round(ageDays)}d old, pnl ${pnlPct.toFixed(1)}% — closing stale`);',
  ]) assert.ok(bot.includes(tpl), `bot.ts no longer writes: ${tpl.slice(0, 80)}`);
  const mgmt = fs.readFileSync(path.join(here, "optionsMgmtAudit.ts"), "utf8");
  assert.ok(mgmt.includes('const parts = [`${String(a.ticker || "?").toUpperCase()} ${what}: ${String(a.reason || a.type || "no reason given").slice(0, 200)}`];'));
  assert.ok(mgmt.includes('return parts.join(" · ");'));
  for (const t of BACKFILL_AUDIT_TYPES) assert.ok(bot.includes(`"${t}"`) || t.startsWith("T") || mgmt.includes(`"${t}"`), t);
});

test("build: floor out, 30-day window, open positions whatever their age, one row per structure, audit reasons, labelled backfilled", () => {
  const o = NOW - 2 * DAY;           // inside the window
  const old = NOW - 60 * DAY;        // outside it
  const events = [
    ev("AMD", o, NOW - DAY, 12, 1800, -40, -1),
    ev("IREN261204P00020000", o + 400, NOW - DAY, 3, 702, -48, -1),
    ev("IREN261204P00021000", o + 400, NOW - DAY, 3, 825, 9, 1),
    ev("IREN261204C00023000", o + 400, NOW - DAY, 3, 783, -9, 1),
    ev("IREN261204C00024000", o + 400, NOW - DAY, 3, 300, 5, -1),
    ev("GLD", o, NOW - DAY, 10, 3800, 20, -1),          // floor: never backfilled
    ev("KO", old, old + DAY, 5, 300, 4, -1),            // closed long ago: no row
  ];
  const fills = [fill("NVDA", old, 1, 5, 100)];         // still held since 60 days ago
  const positions: PositionRow[] = [{ symbol: "NVDA", qty: 5, avgEntry: 100, unrealizedPl: 75, option: false }];
  const audit: AuditLite[] = [
    A("INSTRUMENT", o - 60_000, "AMD → STOCK chosen | scores: stock=63 etf=41 options=N/A | CHOSEN: STOCK (buy_stock)"),
    A("TRADE", o - 30_000, "LIMIT BUY 12 AMD @ ~$150 | Score: 71.3 | $1800 (1.8%) | Sizing: Kelly 0.42"),
    A("OPTIONS-SCANNER-EXEC", o, "IREN | strategy=iron_condor | setup=iv_crush | result=accepted"),
    A("WS-EXIT", NOW - DAY - 5_000, "WS STOP LOSS: -2.2% | LIMIT sell 12 AMD @ $146.67"),
    A("OPTIONS-EXIT", NOW - DAY - 1_000, "IREN iron_condor: strategy_profit_target: retry #1 — 50% kept · order submitted"),
  ];
  const r = buildBackfill({ events, fills, positions, audit, floorTickers: FLOOR, nowMs: NOW });
  assert.equal(r.entries, 3, "AMD, the IREN condor (one row), NVDA (open)");
  assert.equal(r.matchedEntries, 2);
  assert.ok(r.rows.every((x) => x.backfilled === true && x.summary.startsWith("Backfilled on the journal's first run from the broker's fill history")));
  assert.ok(!r.rows.some((x) => x.ticker === "GLD" || x.ticker === "KO"));
  const amd = r.rows.find((x) => x.ticker === "AMD" && x.action === "buy")!;
  assert.equal(amd.chosen, "buy_stock");
  assert.equal(amd.stages.selection?.score, 71.3);
  assert.deepEqual(amd.rejected, [{ instrument: "etf", reason: "scored 41 vs 63 for the stock" }, { instrument: "option", reason: "not scored" }]);
  assert.match(amd.summary, /Bought 12 AMD at \$150\.00\. Why: stock \(score 71\.3\): sizing: Kelly 0\.42; CHOSEN: STOCK \(buy_stock\)\./);
  const iren = r.rows.filter((x) => x.ticker === "IREN" && x.action === "buy");
  assert.equal(iren.length, 1, "the four legs are one structure row");
  assert.equal(iren[0].stages.options?.strategy, "iron condor");
  assert.equal(iren[0].stages.options?.inferred, true);
  assert.equal(iren[0].chosen, "iron_condor (iv_crush)");
  assert.equal(iren[0].signal, "options scanner: iv_crush", "the label a live scanner row carries");
  const nv = r.rows.find((x) => x.ticker === "NVDA")!;
  assert.match(nv.summary, /the reason was not recorded \(no matching audit row in the 30-day window\)\. Still open\.$/);
  assert.equal(nv.signal, "not recorded (before the decision journal)");
  const exits = r.rows.filter((x) => x.action !== "buy");
  assert.equal(r.exits, exits.length);
  assert.deepEqual(exits.map((x) => [x.ticker, x.chosen]).sort(), [["AMD", "stop_loss"], ["IREN", "options_exit"], ["IREN", "options_exit"], ["IREN", "options_exit"], ["IREN", "options_exit"]]);
  assert.match(exits.find((x) => x.ticker === "AMD")!.summary, /closed 12 AMD at \$146\.67 for -\$40\.00\. Why \(audit WS-EXIT\): WS STOP LOSS: -2\.2%\./);
  assert.deepEqual(r.rows.map((x) => x.ts), [...r.rows.map((x) => x.ts)].sort((a, b) => Date.parse(a) - Date.parse(b)), "oldest first");
  assert.equal(BACKFILL_WINDOW_DAYS, 30);
});

test("backfilled signals use the live rows' labels; the scanner's OPTIONS-TRADE twin is skipped", () => {
  const o = NOW - DAY;
  const lots = [ev("FIS261219P00060000", o, NOW - 3_600_000, 1, 80, -20, 1), ev("KWEB261219P00030000", o + 60_000, NOW - 3_600_000, 1, 50, 5, 1),
    ev("NBIS261219P00090000", o + 120_000, NOW - 3_600_000, 1, 90, 9, 1)];
  const audit = [
    A("T1", o - 1000, "SELL_CSP FIS | T1 CSP harvest (regime=NEUTRAL, vxx=1.00, DD=-16.1%, size=38%) [scaled 59% by cap enforcer]"),
    A("OPTIONS-TRADE", o + 59_000, "SELL_CASH_SECURED_PUT KWEB | KWEB261219P00030000 | VRP 6%"),
    A("OPTIONS-SCANNER-EXEC", o + 119_000, "NBIS | strategy=sell_cash_secured_put | setup=high_iv_premium_sale | result=pending_new"),
    A("OPTIONS-TRADE", o + 119_500, "SELL_CASH_SECURED_PUT NBIS |  | Scanner direct execution"),
  ];
  const r = buildBackfill({ events: lots, fills: [], positions: [], audit, floorTickers: FLOOR, nowMs: NOW });
  const sig = Object.fromEntries(r.rows.filter((x) => x.action === "buy").map((x) => [x.ticker, x.signal]));
  assert.deepEqual(sig, { FIS: "tier 1 CSP harvest", KWEB: "deep_score scan (options)", NBIS: "options scanner: high_iv_premium_sale" });
  const fis = r.rows.find((x) => x.ticker === "FIS")!;
  assert.equal(fis.stages.regime?.label, "NEUTRAL");
  assert.equal(fis.stages.regime?.vxx_ratio, 1);
});

test("write once: the file is the marker; a second write is refused; the reader serves it", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "backfill-"));
  assert.equal(backfillDone(dir), false);
  const r = buildBackfill({ events: [ev("AMD", NOW - DAY, NOW - 3_600_000, 1, 100, 1, -1)], fills: [], positions: [], audit: [], floorTickers: FLOOR, nowMs: NOW });
  assert.equal(writeBackfill(dir, r.rows), 1);
  assert.equal(backfillDone(dir), true);
  assert.equal(writeBackfill(dir, r.rows), null);
  assert.equal(fs.readFileSync(path.join(dir, DECISIONS_BACKFILL_FILENAME), "utf8").trim().split("\n").length, 1);
  assert.equal(readDecisions(dir, { ticker: "AMD" })[0].backfilled, true);
  assert.match(backfillAuditLine(r, 1), /^decision journal backfill: 1 rows \(1 entries, 0 with an audit reason; 0 exits with an audit reason\) — labelled backfilled$/);
  assert.match(backfillAuditLine(r, null), /not written/);
  const empty = fs.mkdtempSync(path.join(os.tmpdir(), "backfill-"));
  assert.equal(writeBackfill(empty, []), 0, "an empty history still writes the marker");
  assert.equal(backfillDone(empty), true);
});
