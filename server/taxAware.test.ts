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
  TAX_DECISIONS_FILE, TAX_DECISIONS_MAX_LIMIT, TAX_DECISIONS_DEFAULT_LIMIT,
  taxDecisionRow, parseTaxDecisions, readTaxDecisions, taxBody, taxLimit,
} from "./taxAware";
import { DEFAULT_TAX_RATES } from "./taxLedger";
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
  assert.ok(bot.includes('app.get("/api/bot/tax", requireOwner, (req, res) => {'), "owner-gated route");
  assert.ok(bot.includes("res.json(taxRouteBody(req.query.limit));"));
  const probe = bot.slice(bot.indexOf('case "tax": {'), bot.indexOf('case "tax": {') + 400);
  assert.ok(probe.includes('sanitizeDiag({ probe: "tax", ...taxRouteBody(req.query.limit) })'));
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
