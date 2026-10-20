// accountActivities — non-trade cash flows + the ledger-to-equity
// reconciliation ([RULE-REVIEW] 2026-10-20). Run: npx tsx --test server/accountActivities.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseNonTradeActivities, activityLedger, reconcile, groupOf, ACTIVITY_GROUPS, ACTIVITY_GROUP_ORDER } from "./accountActivities";

// one untyped /v2/account/activities page: fills mixed with everything else
const RAW: unknown[] = [
  { activity_type: "FILL", id: "1", transaction_time: "2026-08-03T14:00:00Z", symbol: "QQQ", side: "buy", qty: "10", price: "600" },
  { activity_type: "DIV", id: "2", date: "2026-08-15", net_amount: "312.30", symbol: "QQQ", qty: "100", per_share_amount: "3.123" },
  { activity_type: "DIV", id: "3", date: "2026-09-15", net_amount: "100", symbol: "SMH" },
  { activity_type: "INT", id: "4", date: "2026-08-31", net_amount: 12.5 },
  { activity_type: "FEE", id: "5", date: "2026-08-31", net_amount: "-4.00" },
  { activity_type: "JNLC", id: "6", date: "2026-05-05", net_amount: "4300" },
  { activity_type: "JNLS", id: "7", date: "2026-05-05", symbol: "AAPL", qty: "10" },           // shares, no cash
  { activity_type: "OPEXP", id: "8", date: "2026-10-17", net_amount: "0", symbol: "BAC261017P00048000", qty: "-1" },
  { activity_type: "XYZ", id: "9", date: "2026-09-02", net_amount: "-1.5" },                   // undocumented type
  { activity_type: "FILL", id: "10", transaction_time: "2026-08-20T14:00:00Z", symbol: "QQQ", side: "sell", qty: "10", price: "590" },
  { id: "11", date: "2026-09-01", net_amount: "5" },                                             // no type
  { activity_type: "DIV", id: "12", date: "not a date", net_amount: "5" },                       // bad date
  null,
];

test("parse: FILL rows are the sleeve ledger's (skipped, counted); typeless / undated rows are malformed; the rest keep type, date, cash and shares", () => {
  const p = parseNonTradeActivities(RAW);
  assert.ok(p);
  assert.equal(p.fills, 2);
  assert.equal(p.malformed, 3, "no type, bad date, null");
  assert.equal(p.rows.length, 8);
  assert.deepEqual(p.rows.map((r) => r.type), ["JNLC", "JNLS", "DIV", "INT", "FEE", "XYZ", "DIV", "OPEXP"], "oldest first");
  const jnls = p.rows[1];
  assert.equal(jnls.cash, false);
  assert.equal(jnls.net, 0);
  assert.equal(jnls.qty, 10);
  assert.equal(jnls.symbol, "AAPL");
  assert.equal(p.rows[2].net, 312.3);
  assert.equal(parseNonTradeActivities({ not: "a list" }), null);
  assert.deepEqual(parseNonTradeActivities([]), { rows: [], fills: 0, malformed: 0 });
});

test("ledger: totals by group, by month and by type; share-only rows counted, never valued; undocumented types kept by name under other", () => {
  const L = activityLedger(parseNonTradeActivities(RAW)!.rows);
  assert.equal(L.total, 4719.3);
  assert.equal(L.count, 8);
  assert.equal(L.nonCashRows, 1);
  assert.deepEqual(L.groups, { dividends: 412.3, interest: 12.5, fees: -4, journals: 4300, optionEvents: 0, corporate: 0, transfers: 0, other: -1.5 });
  assert.deepEqual(Object.keys(L.groups), [...ACTIVITY_GROUP_ORDER]);
  assert.deepEqual(L.byMonth, { "2026-05": 4300, "2026-08": 320.8, "2026-09": 98.5, "2026-10": 0 });
  assert.deepEqual(L.types.map((t) => t.type), ["JNLC", "DIV", "INT", "FEE", "XYZ", "JNLS", "OPEXP"], "largest |net| first, then by name");
  const div = L.types.find((t) => t.type === "DIV")!;
  assert.equal(div.count, 2);
  assert.deepEqual(div.byMonth, { "2026-08": 312.3, "2026-09": 100 });
  assert.equal(L.types.find((t) => t.type === "XYZ")!.group, "other");
  assert.equal(L.types.find((t) => t.type === "JNLS")!.net, 0);
  assert.equal(L.first, "2026-05-05");
  assert.equal(L.last, "2026-10-17");
  const empty = activityLedger([]);
  assert.equal(empty.total, 0);
  assert.equal(empty.first, null);
  assert.deepEqual(empty.types, []);
});

test("every documented Alpaca type has a group; unknown types fall to other", () => {
  for (const [type, group] of Object.entries(ACTIVITY_GROUPS)) assert.ok(ACTIVITY_GROUP_ORDER.includes(group), `${type} → ${group}`);
  assert.equal(groupOf("DIV"), "dividends");
  assert.equal(groupOf("OPASN"), "optionEvents");
  assert.equal(groupOf("JNLC"), "journals");
  assert.equal(groupOf("DIVFEE"), "fees", "a fee on a dividend is a fee: dividends stay gross");
  assert.equal(groupOf("NEVERHEARDOFIT"), "other");
});

test("reconcile: equity change minus (realized + open + non-trade) is the residual; null open P&L gives a null residual; no curve gives null", () => {
  // bot.ts's curve is newest first
  const curve = [{ date: "2026-10-20", value: 99373 }, { date: "2026-07-01", value: 103000 }, { date: "2026-05-06", value: 106743 }];
  const r = reconcile(curve, -9453.01, -2581.44, 4719.3);
  assert.ok(r);
  assert.equal(r.equityFrom, "2026-05-06");
  assert.equal(r.equityTo, "2026-10-20");
  assert.equal(r.equityChange, -7370);
  assert.equal(r.explained, -7315.15);
  assert.equal(r.residual, -54.85);
  assert.equal(reconcile(curve, -100, null, 0)!.residual, null);
  assert.equal(reconcile([], -100, 0, 0), null);
  const one = reconcile([{ date: "2026-10-20", value: 100 }], 0, 0, 0)!;
  assert.equal(one.equityChange, 0);
  assert.equal(one.residual, 0);
});
