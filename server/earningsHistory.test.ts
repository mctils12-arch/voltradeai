// EARNINGS HISTORY, Node half (2026-11-06; earnings_history.py is the logic,
// server/earningsHistory.ts the Tier-3 command + audit row). Pins the audit
// row for each refresh outcome, the Tier-3 wiring (mutex lane, 60 s kill,
// after the flows step), and the decision journal's selection stage text on
// entry rows (decisionJournal.ts earningsView, decision_journal.py's twin —
// test_earnings_history.py pins the vocabulary parity).
// Run: npx tsx --test server/earningsHistory.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { earningsRefreshAuditMessage, EARNINGS_HISTORY_REFRESH_CMD, EARNINGS_HISTORY_REFRESH_TIMEOUT_MS } from "./earningsHistory";
import { planPythonExec } from "./pythonExec";
import { earningsView, selectionStage, entryDecision, EARNINGS_REASON_TEXT } from "./decisionJournal";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");
const module_ = fs.readFileSync(path.join(here, "..", "earnings_history.py"), "utf8");

const DONE = { status: "done", wanted: 9, due: 6, attempted: 5, ok: 3, empty: 1, failed: 1, left: 1, stopped: null,
  first_error: "ERR1: HTTP 502 from Finnhub endpoint /stock/earnings", status_changed: false };

test("audit row: a run that fetched says what it got, what failed and what is still queued", () => {
  assert.deepEqual(earningsRefreshAuditMessage(DONE), {
    type: "EARNINGS-HISTORY",
    message: "earnings history: fetched 5 names from Finnhub (3 with reported quarters, 1 with none on record, 1 failed (ERR1: HTTP 502 from Finnhub endpoint /stock/earnings)); 1 name still queued for the next hourly run",
  });
  const allFailed = earningsRefreshAuditMessage({ ...DONE, ok: 0, empty: 0, failed: 2, left: 0 });
  assert.equal(allFailed?.type, "EARNINGS-HISTORY-ERROR");
  const limited = earningsRefreshAuditMessage({ ...DONE, ok: 1, empty: 0, failed: 0, left: 4, stopped: "Finnhub answered HTTP 429", first_error: "LIM1: HTTP 429 from Finnhub endpoint /stock/earnings" });
  assert.equal(limited?.type, "EARNINGS-HISTORY");
  assert.equal(limited?.message, "earnings history: fetched 1 name from Finnhub (1 with reported quarters); stopped early: Finnhub answered HTTP 429 (LIM1: HTTP 429 from Finnhub endpoint /stock/earnings); 4 names still queued for the next hourly run");
  assert.equal(earningsRefreshAuditMessage({ ...DONE, ok: 0, empty: 0, failed: 0, stopped: "Finnhub answered HTTP 401" })?.type, "EARNINGS-HISTORY-ERROR");
});

test("audit row: idle, off and an unchanged missing key write nothing; a failure and a newly missing key are loud", () => {
  for (const r of [{ status: "idle" }, { status: "off" }, { status: "no_key", status_changed: false },
    { ...DONE, ok: 0, empty: 0, failed: 0, attempted: 0 }, null, "junk"]) {
    assert.equal(earningsRefreshAuditMessage(r), null, JSON.stringify(r));
  }
  assert.deepEqual(earningsRefreshAuditMessage({ status: "no_key", status_changed: true, wanted: 7 }), {
    type: "EARNINGS-HISTORY-ERROR",
    message: "earnings history: FINNHUB_KEY is not set, so no earnings surprises are fetched; 7 deep-scored names log the reason no_api_key",
  });
  assert.deepEqual(earningsRefreshAuditMessage({ status: "error", first_error: "OSError: disk full" }),
    { type: "EARNINGS-HISTORY-ERROR", message: "earnings history refresh failed: OSError: disk full" });
});

test("wiring: Tier 3 step after the flows refresh, in the mutex lane, the daemon route, the Python entry point", () => {
  assert.ok(bot.includes('import { earningsRefreshAuditMessage, EARNINGS_HISTORY_REFRESH_CMD, EARNINGS_HISTORY_REFRESH_TIMEOUT_MS } from "./earningsHistory";'));
  const t3 = bot.slice(bot.indexOf("async function tier3Strategic() {"), bot.indexOf('audit("TIER3", "Strategic scan complete");'));
  const iFlows = t3.indexOf("execPythonSerialized(FLOWS_REFRESH_CMD, { timeout: FLOWS_REFRESH_TIMEOUT_MS })");
  const iEarn = t3.indexOf("execPythonSerialized(EARNINGS_HISTORY_REFRESH_CMD, { timeout: EARNINGS_HISTORY_REFRESH_TIMEOUT_MS })");
  assert.ok(iFlows > 0 && iEarn > iFlows, "runs inside tier3Strategic, after the flows step");
  assert.ok(t3.includes("earningsRefreshAuditMessage(JSON.parse(earnOut.trim()))"));
  assert.ok(t3.includes('noteJobError("tier3_strategic", `earnings history: ${why.slice(0, 200)}`);'));
  const plan = planPythonExec(EARNINGS_HISTORY_REFRESH_CMD, EARNINGS_HISTORY_REFRESH_TIMEOUT_MS);
  assert.equal(plan.lane, "mutex");
  assert.equal(plan.timeoutMs, 60_000);
  assert.ok(EARNINGS_HISTORY_REFRESH_CMD.includes("from earnings_history import refresh_wanted"));
  assert.ok(module_.includes("def refresh_wanted(") && module_.includes('"run_budget_s": 40.0'), "the job's own budget sits inside the 60 s kill");
  assert.ok(daemon.includes('"earnings_history_refresh": ("earnings_history", "refresh_wanted")'));
});

const BLOCK = { source: "finnhub", beats_last_3: 3, surprise_pct_last_3: [12.35, 4.1, 0.52], quarters_seen: 3,
  latest_period: "2026-09-30", cache_age_h: 2.5, reason: null };

test("selection stage: the earnings history in plain English, logged not scored", () => {
  assert.equal(earningsView(BLOCK)?.text, "beat estimates 3 of the last 3 quarters (EPS surprise +12.35%, +4.10%, +0.52%, newest first); logged, not used in the score");
  assert.equal(earningsView({ ...BLOCK, beats_last_3: null, reason: "not_cached", surprise_pct_last_3: null, quarters_seen: null })?.text,
    "earnings history unavailable: it has not been fetched yet (the hourly Finnhub refresh adds it); logged, not used in the score");
  assert.equal(earningsView({ ...BLOCK, beats_last_3: null, reason: "stale", surprise_pct_last_3: null, quarters_seen: null, cache_age_h: 52.7 })?.text,
    "earnings history unavailable: the cached copy is 52 hours old, too old to use; logged, not used in the score");
  assert.equal(earningsView({ ...BLOCK, beats_last_3: null, reason: "incomplete", surprise_pct_last_3: [null, 0, -10] })?.text,
    "earnings history incomplete: a recent quarter has no estimate or actual on record (EPS surprise n/a, +0.00%, -10.00%, newest first); logged, not used in the score");
  assert.equal(earningsView(null), null);
  assert.equal(Object.keys(EARNINGS_REASON_TEXT).length, 8);
  const trade = { ticker: "AMD", score: 71.3, score_breakdown: { momentum: 80 }, reasons: ["Momentum strong"], earnings_history: BLOCK };
  assert.equal((selectionStage(trade)?.earnings as Record<string, unknown>).beats_last_3, 3);
  assert.ok(!("earnings" in (selectionStage({ ...trade, earnings_history: undefined }) ?? {})), "no block, no key: older rows keep their shape");
});

test("an entry row's selection stage carries it; the card's selection line is unchanged", () => {
  const trade = { ticker: "AMD", side: "buy", trade_type: "stock", price: 150.2, score: 71.3, score_threshold: 63, reasons: ["Momentum strong"],
    score_breakdown: { momentum: 80, mean_reversion: 40 }, earnings_history: BLOCK };
  const row = entryDecision(trade, { instrument: "stock", chosen: "buy_stock", side: "buy", symbol: "AMD", qty: 12, price: 150.2,
    notional: 1802.4, status: "accepted", mode: "live", source: "bot.ts executeTrades.stock", equity: 100_000 });
  const sel = row?.stages.selection as Record<string, unknown>;
  assert.ok(String((sel.earnings as Record<string, unknown>).text).startsWith("beat estimates 3 of the last 3 quarters"));
  assert.ok(!("text" in sel) && !("reason" in sel), "no top-level text/reason: client/src/bot/picks.ts plainLine still renders the score line");
  assert.ok(!row?.summary.includes("beat estimates"), "the summary's 'why' is the score's, not the logged-only history");
});
