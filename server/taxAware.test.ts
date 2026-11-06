// [RULE-REVIEW] 2026-11-06 — GET /api/bot/tax (server/taxAware.ts) over the
// tax filter's decision log (tax_filter.py): row shaping, the newest-first
// bounded read, the body's shape (the UI contract), the route + diag probe
// wiring, and a row written by the PYTHON writer read back by this reader.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  TAX_DECISIONS_FILE, TAX_DECISIONS_MAX_LIMIT, TAX_DECISIONS_DEFAULT_LIMIT, TAX_LT_WINDOW_DAYS,
  taxDecisionRow, parseTaxDecisions, readTaxDecisions, taxBody, taxLimit,
  taxHoldOf, TaxDeferAudit, taxDeferLine, fifoOpenLots, lotTerm, ltApproachingRows,
  TAX_REALIZED_FILE, TAX_HARVEST_REVIEW_FILE, taxRealizedSnapshot, writeTaxRealized, readHarvestReview, taxHarvestAuditRows,
} from "./taxAware";
import { DEFAULT_TAX_RATES, taxLedger } from "./taxLedger";
import { DIAG_PROBES } from "./diag";
import type { EdgeLedger } from "./edgeLedger";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");

const row = (ticker: string, allowed: boolean, extra: Record<string, unknown> = {}) => JSON.stringify({
  schema: "tax_decision/v1", ts: `2026-11-06T15:0${ticker.length}:00+00:00`, ticker, side: "buy", kind: "entry",
  instrument: "stock", source: "scan_market", enforced: true, allowed, wouldAllow: allowed, priced: true,
  expectedPct: 0.36, costPct: 0.09, afterTaxPct: 0.21, spyAfterTaxPct: 0.15, edgePp: 0.06, marginPp: 0.035,
  holdingDays: 6.39, bucket: "csp_options", reason: allowed ? "Allowed: ..." : "Skipped: ...",
  spyAssumptionAnnualPct: 10, ...extra,
});

test("a log row maps to the contract fields; torn and foreign lines are rejected", () => {
  const r = taxDecisionRow(JSON.parse(row("AMD", false)));
  assert.ok(r);
  assert.deepEqual(
    { ts: r.ts, ticker: r.ticker, side: r.side, allowed: r.allowed, expectedPct: r.expectedPct, afterTaxPct: r.afterTaxPct, spyAfterTaxPct: r.spyAfterTaxPct, holdingDays: r.holdingDays, reason: r.reason },
    { ts: "2026-11-06T15:03:00+00:00", ticker: "AMD", side: "buy", allowed: false, expectedPct: 0.36, afterTaxPct: 0.21, spyAfterTaxPct: 0.15, holdingDays: 6.39, reason: "Skipped: ..." },
  );
  assert.equal(r.wouldAllow, false);
  assert.equal(r.priced, true);
  assert.equal(taxDecisionRow(null), null);
  assert.equal(taxDecisionRow({ ticker: "X" }), null, "no ts / allowed: not a decision row");
  // an unpriced row keeps nulls, never zeros
  const u = taxDecisionRow({ ts: "2026-11-06T00:00:00Z", ticker: "SPY", allowed: true, priced: false, expectedPct: null, reason: "Allowed (hedge): ..." });
  assert.equal(u?.expectedPct, null);
  assert.equal(u?.priced, false);
});

test("parse returns the newest rows first, at most `limit`, skipping torn lines", () => {
  const text = [row("A", true), "{\"ticker\": \"torn\"", row("BB", false), row("CCC", true), ""].join("\n");
  const got = parseTaxDecisions(text, 2);
  assert.deepEqual(got.rows.map((r) => r.ticker), ["CCC", "BB"]);
  assert.equal(got.spyAssumption, 10);
});

test("the read: a missing log is no error; a present one is read from its tail", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "taxaware-"));
  try {
    assert.deepEqual(readTaxDecisions(dir, 10), { rows: [], spyAssumption: null, error: null });
    fs.writeFileSync(path.join(dir, TAX_DECISIONS_FILE), [row("A", true), row("BB", false)].join("\n") + "\n");
    const got = readTaxDecisions(dir, 10);
    assert.equal(got.error, null);
    assert.deepEqual(got.rows.map((r) => r.ticker), ["BB", "A"]);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("?limit is bounded", () => {
  assert.equal(taxLimit(undefined), TAX_DECISIONS_DEFAULT_LIMIT);
  assert.equal(taxLimit("0"), TAX_DECISIONS_DEFAULT_LIMIT);
  assert.equal(taxLimit("7"), 7);
  assert.equal(taxLimit("99999"), TAX_DECISIONS_MAX_LIMIT);
});

test("the body: contract shape, the ledger's rates, summary counts, live expectancy beside the estimate", () => {
  const rows = parseTaxDecisions([row("A", true), row("BB", false), row("CCC", true, { priced: false, enforced: true })].join("\n"), 10).rows;
  const edge = { options: { n: 316, winRate: 20.6, expectancyPct: -19.21, lastCloseAt: "2026-11-05T20:00:00.000Z" }, stocks: { n: 336, winRate: 43.8, expectancyPct: 1.14, lastCloseAt: null } } as unknown as EdgeLedger;
  const body = taxBody({ rows, spyAssumption: 10, readError: null, edge });
  assert.deepEqual(Object.keys(body).slice(0, 4), ["rates", "decisions", "harvest", "ltApproaching"]);
  assert.deepEqual(body.rates, { shortTerm: DEFAULT_TAX_RATES.shortTerm, longTerm: DEFAULT_TAX_RATES.longTerm });
  assert.deepEqual(body.harvest, []);
  assert.deepEqual(body.ltApproaching, []);
  assert.equal(body.summary.rows, 3);
  assert.equal(body.summary.skipped, 1);
  assert.equal(body.summary.allowed, 2);
  assert.equal(body.summary.notPriced, 1);
  assert.equal(body.summary.enforced, true);
  assert.deepEqual(body.liveExpectancy?.options, { n: 316, winRatePct: 20.6, expectancyPct: -19.21, lastCloseAt: "2026-11-05T20:00:00.000Z" });
  assert.ok(body.notes.some((n) => n.includes("paper account pays no tax")));
  const cold = taxBody({ rows: [], spyAssumption: null, readError: "EACCES", edge: null });
  assert.equal(cold.liveExpectancy, null);
  assert.ok(cold.notes.some((n) => n.includes("decision log unreadable: EACCES")));
});

test("wiring pinned: owner route and diag probe serve the ONE body", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('app.get("/api/bot/tax", requireOwner, async (req, res) => {'), "owner-gated route");
  assert.ok(bot.includes("res.json(await taxRouteBody(req.query.limit));"));
  const probe = bot.slice(bot.indexOf('case "tax": {'), bot.indexOf('case "tax": {') + 400);
  assert.ok(probe.includes('sanitizeDiag({ probe: "tax", ...(await taxRouteBody(req.query.limit)) })'));
  assert.ok((DIAG_PROBES as readonly string[]).includes("tax"));
});

test("cross-language: a row the Python writer logs is read back with its contract fields", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "taxaware-py-"));
  try {
    const file = path.join(dir, TAX_DECISIONS_FILE);
    const program = [
      "import sys, tax_filter as tf",
      "st = {'bucket': 'csp_options', 'winRate': 0.716, 'avgWin': 0.69, 'avgLoss': 0.46, 'n': 0, 'source': 'Kelly prior'}",
      "r = tf.decide(ticker='cmcsa', side='sell', instrument='option', bucket='csp_options', stats=st, cost_pct=0.15, holding_days=6.39, source='test', cfg=tf.config({'ENABLED': True}))",
      "assert tf.log_decision(r, path=sys.argv[1])",
    ].join("\n");
    execFileSync("python3", ["-c", program, file], { cwd: repo, stdio: "pipe" });
    const got = readTaxDecisions(dir, 5);
    assert.equal(got.error, null);
    assert.equal(got.rows.length, 1);
    const r = got.rows[0];
    assert.equal(r.ticker, "CMCSA");
    assert.equal(r.side, "sell");
    assert.equal(r.allowed, false);
    assert.equal(r.holdingDays, 6.39);
    assert.ok(typeof r.expectedPct === "number" && typeof r.afterTaxPct === "number" && typeof r.spyAfterTaxPct === "number");
    assert.ok(r.reason.startsWith("Skipped: "));
    assert.equal(got.spyAssumption, 10);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── 2026-11-06: holding winners past one year ───────────────────────────────

const NOW = new Date("2026-11-06T15:00:00Z");
const ms = (iso: string) => Date.parse(iso);

test("a stop-state hold applies only before its long-term date", () => {
  const ps = { tax_defer_until: "2026-12-02", tax_defer_reason: "Deferred: AAPL is 26 days from long-term" };
  assert.deepEqual(taxHoldOf(ps, NOW), { until: "2026-12-02", reason: ps.tax_defer_reason });
  assert.equal(taxHoldOf(ps, new Date("2026-12-02T00:30:00Z")), null, "long-term from that day: nothing to wait for");
  assert.equal(taxHoldOf({ tax_defer_until: null }, NOW), null);
  assert.equal(taxHoldOf({ tax_defer_until: "soon" }, NOW), null);
  assert.equal(taxHoldOf(undefined, NOW), null);
  assert.match(taxHoldOf({ tax_defer_until: "2026-12-02" }, NOW)!.reason, /tax hold until 2026-12-02/);
});

test("TAX-DEFER audit rows: once per ticker and exit kind per UTC day", () => {
  const a = new TaxDeferAudit();
  assert.equal(a.due("aapl", "scale_out", NOW), true);
  assert.equal(a.due("AAPL", "scale_out", NOW), false);
  assert.equal(a.due("AAPL", "take_profit", NOW), true);
  assert.equal(a.due("AAPL", "scale_out", new Date("2026-11-07T15:00:00Z")), true);
  assert.match(taxDeferLine("aapl", "scale-out", { until: "2026-12-02", reason: "r" }), /^AAPL: scale-out held for tax until 2026-12-02 \(stops still run\) — r$/);
});

test("FIFO lots: sells relieve the oldest lot first", () => {
  const lots = fifoOpenLots([
    { symbol: "X", dir: 1, qty: 10, price: 100, t: ms("2025-11-01T15:00:00Z") },
    { symbol: "X", dir: 1, qty: 10, price: 120, t: ms("2026-03-01T15:00:00Z") },
    { symbol: "X", dir: -1, qty: 15, price: 130, t: ms("2026-06-01T15:00:00Z") },
  ]);
  assert.deepEqual(lots, [{ qty: 5, price: 120, t: ms("2026-03-01T15:00:00Z") }]);
});

test("the long-term date is day 366 after purchase", () => {
  const term = lotTerm(ms("2025-12-01T19:59:00Z"), NOW);
  assert.deepEqual(term, { daysHeld: 340, daysToLongTerm: 26, longTermDate: "2026-12-02" });
});

test("ltApproaching: a winner and a loser inside the window, priced by the live position", () => {
  const fills = [
    { symbol: "AAPL", dir: 1 as const, qty: 30, price: 100, t: ms("2025-12-01T15:00:00Z") },
    { symbol: "XYZ", dir: 1 as const, qty: 10, price: 50, t: ms("2025-11-20T15:00:00Z") },
    { symbol: "NEW", dir: 1 as const, qty: 10, price: 50, t: ms("2026-10-20T15:00:00Z") },
  ];
  const positions = [
    { symbol: "AAPL", qty: 30, avgEntry: 100, unrealizedPl: 1200, option: false },   // now $140
    { symbol: "XYZ", qty: 10, avgEntry: 50, unrealizedPl: -100, option: false },     // now $40
    { symbol: "NEW", qty: 10, avgEntry: 50, unrealizedPl: 50, option: false },       // far from long-term
    { symbol: "AAPL261218P00100000", qty: -1, avgEntry: 1, unrealizedPl: 0, option: true },
  ];
  const holds = (t: string) => (t === "AAPL" ? { until: "2026-12-02", reason: "r" } : null);
  const rows = ltApproachingRows(fills, positions, NOW, holds);
  assert.deepEqual(rows.map((r) => [r.ticker, r.daysToLongTerm, r.gain]), [["XYZ", 15, -100], ["AAPL", 26, 1200]]);
  assert.match(rows[1].reason, /taxed at 24% instead of 15%, about \$108 more tax on the \$1,200 gain; its profit-taking exits are held until 2026-12-02/);
  assert.match(rows[0].reason, /at a \$100 loss: sold before then it stays short-term/);
});

test("parity: the TS window mirrors system_config's LT_DEFER_WINDOW_DAYS", () => {
  const out = execFileSync("python3", ["-c", "import tax_filter as tf; print(tf.config()['LT_DEFER_WINDOW_DAYS'])"], { cwd: repo }).toString().trim();
  assert.equal(Number(out), TAX_LT_WINDOW_DAYS);
});

// The two hold blocks of checkPositionOnTick, run as written (the tick needs Alpaca otherwise).
function tickBlock(from: string, to: string): string {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const fn = bot.slice(bot.indexOf("async function checkPositionOnTick("));
  const start = fn.indexOf(from);
  const end = fn.indexOf(to, start);
  assert.ok(start > 0 && end > start, `markers not found: ${from}`);
  return fn.slice(start, end);
}

test("checkPositionOnTick: a hold stops the scale-out; without one it proceeds", () => {
  const block = tickBlock("const taxHold = taxHoldOf(", "// ── DUPLICATE SELL ORDER GUARD");
  const run = new Function("taxHoldOf", "taxDeferAudit", "taxDeferLine", "audit", "pos", "ticker", "shouldScaleOut",
    `${block}; return { shouldScaleOut, taxHold };`);
  const audits: string[] = [];
  const audit = (_t: string, m: string) => audits.push(m);
  const held = run(taxHoldOf, new TaxDeferAudit(), taxDeferLine, audit, { taxDeferUntil: "2099-01-01", taxDeferReason: "r" }, "AAPL", true);
  assert.equal(held.shouldScaleOut, false);
  assert.equal(audits.length, 1);
  const free = run(taxHoldOf, new TaxDeferAudit(), taxDeferLine, audit, { taxDeferUntil: null }, "AAPL", true);
  assert.equal(free.shouldScaleOut, true);
});

test("checkPositionOnTick: a hold clears take-profit and time stop, never a stop", () => {
  const block = tickBlock("if (!shouldStop && (shouldTP || shouldTimeStop) && taxHold) {", "// ETF max hold");
  const run = new Function("taxDeferAudit", "taxDeferLine", "audit", "ticker", "taxHold", "shouldStop", "shouldTP", "shouldTimeStop",
    `${block}; return { shouldStop, shouldTP, shouldTimeStop };`);
  const hold = { until: "2099-01-01", reason: "r" };
  const audits: string[] = [];
  const audit = (_t: string, m: string) => audits.push(m);
  assert.deepEqual(run(new TaxDeferAudit(), taxDeferLine, audit, "AAPL", hold, false, true, false), { shouldStop: false, shouldTP: false, shouldTimeStop: false });
  assert.deepEqual(run(new TaxDeferAudit(), taxDeferLine, audit, "AAPL", hold, false, false, true), { shouldStop: false, shouldTP: false, shouldTimeStop: false });
  assert.deepEqual(run(new TaxDeferAudit(), taxDeferLine, audit, "AAPL", hold, true, false, false), { shouldStop: true, shouldTP: false, shouldTimeStop: false }, "a stop is untouched");
  assert.deepEqual(run(new TaxDeferAudit(), taxDeferLine, audit, "AAPL", null, false, true, false), { shouldStop: false, shouldTP: true, shouldTimeStop: false }, "no hold: the exit runs");
  assert.equal(audits.length, 2);
});

test("wiring: the sync carries the stop state's hold; TIME-EXIT and the stop checks are untouched", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes("taxDeferUntil: ps.tax_defer_until || null, taxDeferReason: ps.tax_defer_reason || null,"));
  const fn = bot.slice(bot.indexOf("async function checkPositionOnTick("), bot.indexOf("function startStreaming("));
  assert.ok(fn.indexOf("const taxHold = taxHoldOf(") < fn.indexOf("// ── DUPLICATE SELL ORDER GUARD"), "the scale-out hold sits before any order");
  assert.ok(fn.indexOf("if (!shouldStop && (shouldTP || shouldTimeStop) && taxHold) {") < fn.indexOf("// ── FIRE EXIT ORDER"), "the TP / time-stop hold sits before the exit order");
  assert.ok(bot.includes("if (hasMark && ageDays > 14 && pnlPct >= -5.0 && pnlPct <= 2.0) {"), "TIME-EXIT unchanged");
});

// ── 2026-11-06: tax-loss harvesting ─────────────────────────────────────────

const ev = (symbol: string, iso: string, pnl: number) => ({ symbol, t: ms(iso), pnl, openedAt: ms(iso) - 86_400_000, qty: 10, closeDir: -1 as const, basis: 1000 });

test("the realized snapshot: this calendar year only, the ledger's netting, recent loss sales and buys", () => {
  const events = [ev("OLD", "2025-12-15T15:00:00Z", 5000), ev("AMD", "2026-03-02T15:00:00Z", 900), ev("XLK", "2026-10-20T15:00:00Z", -200)];
  const fills = [{ symbol: "AMD", dir: 1 as const, qty: 10, price: 100, t: ms("2026-10-25T15:00:00Z") },
                 { symbol: "JPM", dir: 1 as const, qty: 10, price: 100, t: ms("2026-08-01T15:00:00Z") }];
  const snap = taxRealizedSnapshot(events, fills, NOW);
  const ledger = taxLedger(events, fills, { windowStartMs: Date.UTC(2026, 0, 1) });
  assert.equal(snap.year, 2026);
  assert.equal(snap.shortAfterNetting, ledger.netting.shortAfterNetting);
  assert.equal(snap.shortAfterNetting, 700, "the 2025 gain is not this year's; +900 - 200");
  assert.deepEqual(snap.recentLossSales, { XLK: "2026-10-20T15:00:00.000Z" });
  assert.deepEqual(snap.recentBuys, { AMD: "2026-10-25T15:00:00.000Z" });
});

test("cross-language: the snapshot bot.ts writes is the one tax_filter reads; the review tax_filter writes is the one the route serves", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "taxaware-hv-"));
  try {
    const now = new Date();
    assert.equal(writeTaxRealized(dir, taxRealizedSnapshot([ev("AMD", now.toISOString(), 1234)], [], now)), null);
    assert.ok(fs.existsSync(path.join(dir, TAX_REALIZED_FILE)));
    const program = [
      "import json, sys, tax_filter as tf",
      "tf._data_dir = lambda: sys.argv[1]",
      "snap, why = tf.realized_snapshot()",
      "assert snap is not None, why",
      "rows = tf.harvest_candidates([{'symbol': 'AMD', 'qty': '50', 'current_price': '100', 'unrealized_pl': '-800', 'asset_class': 'us_equity', 'side': 'long'}], snap, lambda s: 'Technology', cfg=tf.config({'HARVEST_ENABLED': True}))",
      "tf._write_review(rows, snap, '', tf.config())",
      "print(json.dumps(snap['shortAfterNetting']))",
    ].join("\n");
    const out = execFileSync("python3", ["-c", program, dir], { cwd: repo }).toString().trim();
    assert.equal(Number(out), 1234);
    const got = readHarvestReview(dir);
    assert.equal(got.error, null);
    assert.equal(got.review?.harvest.length, 1);
    const row = got.review!.harvest[0];
    assert.deepEqual([row.ticker, row.unrealized, row.replacement, row.action], ["AMD", -800, "XLK", "harvest"]);
    assert.ok(row.washSaleUntil && /^\d{4}-\d{2}-\d{2}$/.test(row.washSaleUntil));
    assert.ok(row.reason.startsWith("Harvest: sell 50 AMD"));
    assert.ok(fs.existsSync(path.join(dir, TAX_HARVEST_REVIEW_FILE)));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("a missing review is not an error; a corrupt one is reported", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "taxaware-hv2-"));
  try {
    assert.deepEqual(readHarvestReview(dir), { review: null, error: null });
    fs.writeFileSync(path.join(dir, TAX_HARVEST_REVIEW_FILE), "{not json");
    assert.ok(readHarvestReview(dir).error);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("harvest audit rows: one per swap, the failure kinds, the step's error", () => {
  const rows = taxHarvestAuditRows({
    actions: [
      { ticker: "AMD", replacement: "XLK", sold: { ok: true }, bought: { ok: true } },
      { ticker: "JPM", replacement: "XLF", sold: { ok: true }, bought: { ok: false, error: "rejected" } },
      { ticker: "TSLA", replacement: "XLY", sold: { ok: false, error: "market closed" }, bought: null },
    ],
    error: "RuntimeError: x",
  });
  assert.deepEqual(rows.map((r) => r.type), ["TAX-HARVEST", "TAX-HARVEST", "TAX-HARVEST-FAIL", "TAX-HARVEST-ERROR"]);
  assert.match(rows[0].message, /^AMD: sold for a tax-loss harvest; bought XLK with the proceeds; AMD is blocked from re-entry for 30 days/);
  assert.match(rows[1].message, /XLF NOT bought \(rejected\), proceeds in cash/);
  assert.match(rows[2].message, /harvest sale refused \(market closed\); nothing bought/);
  assert.deepEqual(taxHarvestAuditRows(undefined), []);
});

test("wiring: the ledger refresh writes the snapshot, Tier 3 keeps it fresh, Tier 2 audits the swaps, the route serves the review", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const refresh = bot.slice(bot.indexOf("async function refreshSleeves()"), bot.indexOf("async function sleevesBody("));
  assert.ok(refresh.indexOf("writeTaxRealized(regimeDataDir(process.env), taxRealizedSnapshot(walked.events, fills.rows, new Date()))") > refresh.indexOf("sleeveError = null;"));
  const t3 = bot.slice(bot.indexOf("async function tier3Strategic()"), bot.indexOf("// 1. ML model retrain (if needed)"));
  assert.ok(t3.includes("Date.now() - sleeveCache.at > TAX_REALIZED_REFRESH_MS"));
  assert.ok(bot.indexOf("for (const row of taxHarvestAuditRows(result.tax_harvest)) audit(row.type, row.message);") > bot.indexOf('floorAuditRows(result.spy_floor, "scan")'));
  assert.ok(bot.includes("harvest: hv.review?.harvest ?? []"));
});
