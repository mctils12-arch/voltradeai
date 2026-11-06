// Options exits used to be invisible: scan_market returned them as
// `options_management` and bot.ts never read it (REPAIR 2026-10-20).
// Run: npx tsx --test server/optionsMgmtAudit.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { formatOptionsMgmt, OPTIONS_EARNINGS_NOTICE_MAX_CHARS, OPTIONS_EARNINGS_NOTICE_TYPES, OPTIONS_MGMT_AUDIT_CAP } from "./optionsMgmtAudit";

// the live 2026-10-19 TSLA condor close, in options_manager's own action shape
const CONDOR_CLOSE = {
  action: "CLOSE_STRATEGY", ticker: "TSLA", strategy: "iron_condor",
  legs: ["TSLA261030C00470000", "TSLA261030C00485000", "TSLA261030P00420000", "TSLA261030P00405000"],
  reason: "21 DTE MANAGEMENT: 9 DTE — closing iron_condor", type: "strategy_dte_exit",
  combined_pnl: -66.0, order: { status: "submitted", method: "mleg", detail: "mleg close: 4 legs" },
};

test("a strategy close becomes one OPTIONS-EXIT row with reason, P&L at decision and order result", () => {
  const rows = formatOptionsMgmt({ actions: [CONDOR_CLOSE], positions_checked: 4 });
  assert.deepEqual(rows, [{
    type: "OPTIONS-EXIT",
    message: "TSLA iron_condor: 21 DTE MANAGEMENT: 9 DTE — closing iron_condor · unrealized −$66.00 at decision · 4 legs · order submitted (mleg)",
  }]);
});

test("single-leg closes, rolls and failed orders are all visible; advisory warnings are not", () => {
  const rows = formatOptionsMgmt({ actions: [
    { action: "CLOSE", ticker: "bac", occ_symbol: "BAC261120P00048000", reason: "21 DTE EXIT: 21 DTE — roll failed, closing", type: "dte_close", order: { status: "error", detail: "insufficient buying power" } },
    { action: "ROLL", ticker: "XLF", occ_symbol: "XLF261023P00050000", reason: "21 DTE MANAGEMENT: 20 DTE — rolled to later expiry for credit", type: "dte_roll", roll_detail: "-> XLF261120P00049000 @ 0.92" },
    { action: "WARNING", ticker: "KMI", strategy: "buy_straddle", reason: "EDGE WARNING: VRP at +6.0%", type: "sabr_edge_warning" },
  ] });
  assert.deepEqual(rows.map((r) => r.type), ["OPTIONS-EXIT", "OPTIONS-ROLL"]);
  assert.match(rows[0].message, /^BAC BAC261120P00048000: 21 DTE EXIT.* · order FAILED: insufficient buying power$/);
  assert.match(rows[1].message, /rolled to later expiry for credit · -> XLF261120P00049000 @ 0\.92$/);
});

test("the per-leg fallback's rejected legs are counted, not hidden behind 'submitted'", () => {
  const [row] = formatOptionsMgmt({ actions: [{ ...CONDOR_CLOSE, order: { status: "submitted", method: "individual_legs", leg_results: [{ status: "submitted" }, { status: "error" }, { status: "submitted" }, { status: "error" }] } }] });
  assert.match(row.message, / · order submitted \(individual_legs\) · 2\/4 legs FAILED$/);
});

test("exit ladder (2026-11-05): a laddered close names its step; steps and the fill are OPTIONS-LADDER rows, not exits", () => {
  const rows = formatOptionsMgmt({ actions: [
    { action: "CLOSE", ticker: "BLOS", occ_symbol: "BLOS261218C00520000", reason: "LOSS LIMIT: Bought option down 64% from entry", type: "bought_loss_limit",
      order: { status: "submitted", detail: "Close sell 2x BLOS261218C00520000 @ $1.80", ladder: { step: 1, of: 3, price: 1.8 } } },
    { action: "LADDER", ticker: "BLOS", occ_symbol: "BLOS261218C00520000", reason: "exit ladder step 2/3: limit $1.70 (bid 1.6 / ask 2.0)", type: "exit_ladder_step",
      order: { status: "replaced", ladder: { step: 2, of: 3, price: 1.7 } } },
    { action: "LADDER", ticker: "BLOS", occ_symbol: "BLOS261218C00520000", reason: "exit ladder FILLED at step 3/3 (the touch) at $1.60, 2m 1s after step 1", type: "exit_ladder_filled",
      order: { status: "filled", ladder: { step: 3, of: 3, price: 1.6 } } },
  ] });
  assert.deepEqual(rows.map((r) => r.type), ["OPTIONS-EXIT", "OPTIONS-LADDER", "OPTIONS-LADDER"]);
  assert.match(rows[0].message, / · order submitted · exit ladder step 1\/3 @ \$1\.80$/);
  assert.match(rows[1].message, /^BLOS BLOS261218C00520000: exit ladder step 2\/3: limit \$1\.70 .* · order replaced · exit ladder step 2\/3 @ \$1\.70$/);
  assert.match(rows[2].message, /FILLED at step 3\/3 \(the touch\).* · order filled · exit ladder step 3\/3 @ \$1\.60$/);
  // a close without a ladder renders exactly as before
  const [plain] = formatOptionsMgmt({ actions: [{ action: "CLOSE", ticker: "X", occ_symbol: "X261218C00010000", reason: "PROFIT TARGET", order: { status: "submitted" } }] });
  assert.equal(plain.message, "X X261218C00010000: PROFIT TARGET · order submitted");
});

test("a manager error is surfaced; nothing / garbage gives no rows; a busy scan is capped", () => {
  assert.deepEqual(formatOptionsMgmt({ actions: [], error: "Position fetch failed: 500" }),
    [{ type: "OPTIONS-MGMT-ERROR", message: "options manager failed this scan: Position fetch failed: 500" }]);
  for (const v of [undefined, null, "x", 3, { actions: "no" }, { actions: [null, 1, {}] }]) assert.deepEqual(formatOptionsMgmt(v), []);
  const many = Array.from({ length: OPTIONS_MGMT_AUDIT_CAP + 3 }, () => CONDOR_CLOSE);
  const rows = formatOptionsMgmt({ actions: many });
  assert.equal(rows.length, OPTIONS_MGMT_AUDIT_CAP + 1);
  assert.equal(rows.at(-1)?.message, "… 3 more options exits/rolls this scan");
});

// 2026-11-05 EARNINGS CLOSE: options_manager's two notices that do not repeat
// every scan, in its own action shapes (options_manager._earnings_close_context)
const EARNINGS_BACKFILL = {
  action: "INFO", type: "earnings_backfill",
  legs: [{ occ_symbol: "IREN261211P00018000", underlying: "IREN", expiry: "2026-12-11", report_date: "2026-11-05", report_hour: "amc", close_session: "2026-11-05", status: "due" }],
  reason: "EARNINGS BACKFILL (first pass since start): 1 sold option(s) checked — IREN261211P00018000: IREN reports 2026-11-05 after the close, close session 2026-11-05 (due now)",
};
const EARNINGS_LOOKUP_FAILED = {
  action: "WARNING", ticker: "AAL", type: "earnings_lookup_failed", legs: ["AAL261204P00013000"],
  reason: "EARNINGS CLOSE: AAL's next report date is unknown (the Finnhub lookup failed) — 1 sold option(s) not closed on a guess (AAL261204P00013000); retried every tick, warned once per ET day",
};

test("earnings notices (the deploy backfill, a failed report-date lookup) become OPTIONS-EARNINGS rows; other warnings still do not", () => {
  const earningsClose = { action: "CLOSE", ticker: "IREN", occ_symbol: "IREN261211P00018000", type: "earnings_close",
    reason: "EARNINGS CLOSE: IREN reports 2026-11-05 after the close; 2026-11-05 is the last session before it — closing the sold option rather than holding it through the report",
    order: { status: "submitted", detail: "Close buy 1x IREN261211P00018000 @ $2.70" } };
  const rows = formatOptionsMgmt({ actions: [
    EARNINGS_BACKFILL,
    { action: "WARNING", ticker: "KMI", strategy: "buy_straddle", reason: "EDGE WARNING: VRP at +6.0%", type: "sabr_edge_warning" },
    EARNINGS_LOOKUP_FAILED,
    earningsClose,
  ] });
  assert.deepEqual(rows.map((r) => r.type), ["OPTIONS-EARNINGS", "OPTIONS-EARNINGS", "OPTIONS-EXIT"]);
  assert.equal(rows[0].message, EARNINGS_BACKFILL.reason);
  assert.equal(rows[1].message, EARNINGS_LOOKUP_FAILED.reason);
  assert.match(rows[2].message, /^IREN IREN261211P00018000: EARNINGS CLOSE: IREN reports 2026-11-05 after the close.* · order submitted$/);
});

test("a long earnings notice is cut at OPTIONS_EARNINGS_NOTICE_MAX_CHARS and never counts against the exit cap", () => {
  const long = { ...EARNINGS_BACKFILL, reason: "EARNINGS BACKFILL: " + "X261211P00018000: no X report before expiry 2026-12-11; ".repeat(60) };
  const many = Array.from({ length: OPTIONS_MGMT_AUDIT_CAP }, () => CONDOR_CLOSE);
  const rows = formatOptionsMgmt({ actions: [long, ...many] });
  assert.equal(rows[0].type, "OPTIONS-EARNINGS");
  assert.equal(rows[0].message.length, OPTIONS_EARNINGS_NOTICE_MAX_CHARS);
  assert.equal(rows.filter((r) => r.type === "OPTIONS-EXIT").length, OPTIONS_MGMT_AUDIT_CAP, "no '… more' row: the notice is not an exit");
  assert.deepEqual(OPTIONS_EARNINGS_NOTICE_TYPES, ["earnings_backfill", "earnings_lookup_failed"]);
});

test("the notice types are the ones options_manager.py emits", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const om = fs.readFileSync(path.join(here, "..", "options_manager.py"), "utf8");
  assert.ok(om.includes('EARNINGS_BACKFILL_TYPE = "earnings_backfill"'));
  assert.ok(om.includes('EARNINGS_LOOKUP_FAILED_TYPE = "earnings_lookup_failed"'));
});

test("bot.ts audits the scan's options_management rows", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(/for \(const row of formatOptionsMgmt\(result\.options_management\)\) audit\(row\.type, row\.message\);/.test(bot));
});
