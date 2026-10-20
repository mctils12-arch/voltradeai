// Options exits used to be invisible: scan_market returned them as
// `options_management` and bot.ts never read it (REPAIR 2026-10-20).
// Run: npx tsx --test server/optionsMgmtAudit.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { formatOptionsMgmt, OPTIONS_MGMT_AUDIT_CAP } from "./optionsMgmtAudit";

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

test("a manager error is surfaced; nothing / garbage gives no rows; a busy scan is capped", () => {
  assert.deepEqual(formatOptionsMgmt({ actions: [], error: "Position fetch failed: 500" }),
    [{ type: "OPTIONS-MGMT-ERROR", message: "options manager failed this scan: Position fetch failed: 500" }]);
  for (const v of [undefined, null, "x", 3, { actions: "no" }, { actions: [null, 1, {}] }]) assert.deepEqual(formatOptionsMgmt(v), []);
  const many = Array.from({ length: OPTIONS_MGMT_AUDIT_CAP + 3 }, () => CONDOR_CLOSE);
  const rows = formatOptionsMgmt({ actions: many });
  assert.equal(rows.length, OPTIONS_MGMT_AUDIT_CAP + 1);
  assert.equal(rows.at(-1)?.message, "… 3 more options exits/rolls this scan");
});

test("bot.ts audits the scan's options_management rows", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(/for \(const row of formatOptionsMgmt\(result\.options_management\)\) audit\(row\.type, row\.message\);/.test(bot));
});
