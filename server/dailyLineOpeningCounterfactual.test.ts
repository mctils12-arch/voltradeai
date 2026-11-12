// OPENING-WINDOW COUNTERFACTUAL in the DAILY LINE (2026-11-12, measurement
// only). The line gains one optional block (what waiting out the 09:30–09:45
// window cost or saved) and nothing it already reported changes: the same
// inputs give the same line, field for field, and the same audit message.
// Run: npx tsx --test server/dailyLineOpeningCounterfactual.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildDailyLine, dailyLineBody, dailyLineMessage, fillsSlippage, loadDailyLines, optionFillsInWindow, parseLedger, saveDailyLine,
  type DailyLineInputs,
} from "./dailyLine";
import { sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (rel: string) => fs.readFileSync(path.join(here, rel), "utf8");
const ms = (iso: string) => Date.parse(iso);

const DAY = "2026-11-12";
const AT = "2026-11-12T09:31:07-05:00";
const SELL = "BLOS261218C00520000";              // held by the window, sold after it
const DROP = "GAMA261204C00185000";              // held by the window, never sent
const RECORD = { key: SELL, at: AT, rule: "bought_loss_limit", kind: "single_leg", bid: 1.6, ask: 2.0, mid: 1.8 };

// The ledger as option_fill_ledger.py and bot.ts store it.
const ENTRY_ROW = { ts: "2026-11-12T15:00:00+00:00", order_id: "entry-1", symbol: "TLT261218C00090000", side: "buy", qty: 1.0, bid: 0.98, ask: 1.02, mid: 1.0,
  source: "options_execution.submit_options_order", limit: 1.02, quote_age_s: 1.2 };
const CLOSE_ROW = { ts: "2026-11-12T14:46:03+00:00", order_id: "close-1", symbol: SELL, side: "sell", qty: 2.0, bid: 1.62, ask: 1.98, mid: 1.8,
  source: "options_manager._submit_close_order", limit: 1.62, quote_age_s: 0.0, first_fired: RECORD };
const NODE_ROW = { ts: "2026-11-12T14:50:00.000Z", order_id: "ts-1", symbol: "GOOGL261127P00300000", side: "sell", qty: 1, bid: null, ask: null, mid: null,
  source: "bot.ts:POS-KILL ladder", limit: null, quote_age_s: null };
const text = (...rows: object[]) => `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`;
const LEDGER_TEXT = text(ENTRY_ROW, CLOSE_ROW, NODE_ROW);
const { first_fired: _record, ...CLOSE_ROW_BEFORE } = CLOSE_ROW;      // the row as it was written before this change

const ORDERS: unknown[] = [
  { id: "entry-1", symbol: "TLT261218C00090000", asset_class: "us_option", side: "buy", status: "filled", filled_qty: "1", filled_avg_price: "1.10", filled_at: "2026-11-12T15:00:05Z" },
  { id: "close-1", symbol: SELL, asset_class: "us_option", side: "sell", status: "filled", filled_qty: "2", filled_avg_price: "1.70", filled_at: "2026-11-12T14:47:00Z" },
  { id: "ts-1", symbol: "GOOGL261127P00300000", asset_class: "us_option", side: "sell", status: "filled", filled_qty: "1", filled_avg_price: "10.65", filled_at: "2026-11-12T14:50:30Z" },
];

const fired = (symbol: string, rule: string, bid: number, ask: number) => ({
  kind: "single_leg", rule, rules: [rule], first_fired_at: AT, last_fired_at: AT, fires: 1, refused: 0, feed: "indicative",
  legs: [{ symbol, side: "sell", qty: symbol === SELL ? 2 : 1, bid, ask, mid: (bid + ask) / 2, mark: bid, quote_age_s: 0.3 }],
});
const FIRST_FIRED = { version: 1, updated_at: "2026-11-12T09:31:09", days: { [DAY]: { [SELL]: fired(SELL, "bought_loss_limit", 1.6, 2.0), [DROP]: fired(DROP, "gamma_risk", 4.3, 4.7) } } };
const DEFERRALS = { version: 1, days: { [DAY]: { single_leg: [`${SELL}|bought_loss_limit`, `${DROP}|gamma_risk`], group: [] } } };
const CLAUSE = "opening-window exits: 2 first fired, 1 sent, 1 never; waiting cost median +5.56% vs mid (1 priced), -6.25% vs touch (1 priced)";

/** The inputs server/bot.ts gives the line; `firstFired` is the one input this change adds. */
function inputs(over: Partial<DailyLineInputs> = {}): DailyLineInputs {
  return {
    day: DAY, nowMs: ms("2026-11-12T21:20:30Z"), session: { date: DAY, closeMin: 16 * 60 }, account: null, positions: null, spyBars: null, curve: [],
    floorCfg: null, vmState: null, prev: null, windowFromMs: ms("2026-11-12T05:00:00Z"), events: null, orders: ORDERS,
    ledger: parseLedger(LEDGER_TEXT), deferrals: DEFERRALS, posKillRows: [], missing: [],
    ...over,
  };
}

test("ledger: the eight fields every reader had are what they were; a row also carries its time, limit and record when stored", () => {
  const got = parseLedger(LEDGER_TEXT);
  assert.equal(got.malformed, 0);
  const core = got.rows.map((r) => ({ order_id: r.order_id, symbol: r.symbol, side: r.side, qty: r.qty, bid: r.bid, ask: r.ask, mid: r.mid, source: r.source }));
  assert.deepEqual(core, [
    { order_id: "entry-1", symbol: "TLT261218C00090000", side: "buy", qty: 1, bid: 0.98, ask: 1.02, mid: 1, source: "options_execution.submit_options_order" },
    { order_id: "close-1", symbol: SELL, side: "sell", qty: 2, bid: 1.62, ask: 1.98, mid: 1.8, source: "options_manager._submit_close_order" },
    { order_id: "ts-1", symbol: "GOOGL261127P00300000", side: "sell", qty: 1, bid: null, ask: null, mid: null, source: "bot.ts:POS-KILL ladder" },
  ]);
  assert.deepEqual(got.rows[0], { ...core[0], ts: "2026-11-12T15:00:00+00:00", limit: 1.02 }, "an entry has no record");
  assert.deepEqual(got.rows[1], { ...core[1], ts: "2026-11-12T14:46:03+00:00", limit: 1.62, first_fired: RECORD });
  assert.deepEqual(got.rows[2], { ...core[2], ts: "2026-11-12T14:50:00.000Z" }, "a null limit adds nothing");
  // a record that cannot be read is dropped; the row stays a row
  const odd = parseLedger(text({ ...CLOSE_ROW, first_fired: { key: "", at: AT } }, { ...CLOSE_ROW, first_fired: "x" }));
  assert.deepEqual(odd.rows.map((r) => [r.order_id, r.mid, "first_fired" in r]), [["close-1", 1.8, false], ["close-1", 1.8, false]]);
  assert.equal(odd.malformed, 0);
});

test("fills: the record on a row changes nothing in the fills/slippage split", () => {
  const fills = optionFillsInWindow(ORDERS, ms("2026-11-12T05:00:00Z"), ms("2026-11-12T21:20:30Z"));
  assert.deepEqual(parseLedger(LEDGER_TEXT).rows.map((r) => !!r.first_fired), [false, true, false], "the close's row carries its record");
  const withRecord = fillsSlippage(fills.fills, parseLedger(LEDGER_TEXT), fills.mleg_no_leg_fills);
  const before = fillsSlippage(fills.fills, parseLedger(text(ENTRY_ROW, CLOSE_ROW_BEFORE, NODE_ROW)), fills.mleg_no_leg_fills);
  assert.deepEqual(withRecord, before);
  // entry: bought 0.10 over its mid = $10; close: sold 0.10 under the mid AT SUBMISSION (1.80) x 2 = $20; the Node row has no mid
  assert.deepEqual([withRecord.fills, withRecord.attributed, withRecord.cost, withRecord.unattributed.no_mid], [3, 2, 30, 1]);
});

test("the line: the same inputs give the line they gave (no new key, the same message); the record file adds one block and one clause", () => {
  // what the line was built from before this change: no first-fired file, and a ledger whose rows carry no record
  const old = { ledger: parseLedger(text(ENTRY_ROW, CLOSE_ROW_BEFORE, NODE_ROW)) };
  const before = buildDailyLine(inputs(old));
  assert.deepEqual(Object.keys(before), ["version", "day", "written_at", "session_close", "account_vs_spy", "floor", "options", "opening_window",
    "close_check", "missing", "message"], "the line has exactly the keys it had");
  assert.match(before.message, /\| opening-window blocks 2 \(manager 2, POS-KILL 0\)$/);
  assert.deepEqual(buildDailyLine(inputs({ ...old, firstFired: null })), before, "an unreadable or missing file is the same as none");

  const after = buildDailyLine(inputs({ firstFired: FIRST_FIRED }));
  const { opening_counterfactual: block, message, ...rest } = after;
  const { message: messageBefore, ...restBefore } = before;
  assert.deepEqual(rest, restBefore, "every field the line already had is unchanged");
  assert.equal(message, `${messageBefore} | ${CLAUSE}`, "the message it was, then one clause");
  assert.equal(dailyLineMessage(after), message);
  // the clause is last, after the close check and the partial marker: the 490-char cut can only shorten it
  for (const [pad, room] of [[200, true], [400, false]] as Array<[number, boolean]>) {
    const missing = ["m".repeat(pad)];
    const was = dailyLineMessage({ ...before, missing });
    const now = dailyLineMessage({ ...after, missing });
    assert.equal(was.length < 490, room);
    assert.equal(now.length, 490, "persistAudit keeps 500 chars");
    assert.ok(now.startsWith(was), "the text the message already carried is never cut for the clause");
    assert.equal(now.includes("opening-window exits:"), room, room ? "some room: the clause is cut short" : "no room: the clause is dropped");
  }
  assert.ok(block);
  assert.deepEqual([block.first_fired, block.sent, block.never_sent, block.refused, block.by_fill, block.by_limit, block.unpriced], [2, 1, 1, 0, 1, 0, 0]);
  assert.deepEqual(block.vs_mid, { priced: 1, mean_pct: 5.56, median_pct: 5.56, usd: 20 });
  assert.deepEqual(block.vs_touch, { priced: 1, mean_pct: -6.25, median_pct: -6.25, usd: -20 });
  const [sold, dropped] = block.exits;
  assert.deepEqual([sold.key, sold.rule, sold.sent, sold.via, sold.priced_by, sold.first_mid, sold.after, sold.delay_min], [SELL, "bought_loss_limit", true, "link", "fill", 1.8, 1.7, 14.9]);
  assert.deepEqual([dropped.key, dropped.rule, dropped.sent, dropped.cost_pct], [DROP, "gamma_risk", false, null]);
  assert.equal(after.opening_window.total, 2, "the block count is the count it was");
});

test("the line: a day nothing fired on has no clause; unreadable orders price the exit at its limit and say so", () => {
  const calm = { ledger: parseLedger(text(ENTRY_ROW, NODE_ROW)) };
  const quiet = buildDailyLine(inputs({ ...calm, firstFired: { version: 1, days: { "2026-11-11": FIRST_FIRED.days[DAY] } } }));
  assert.deepEqual(quiet, buildDailyLine(inputs(calm)), "the file exists but nothing fired today: the line is the line it was");
  assert.ok(!("opening_counterfactual" in quiet) && !quiet.message.includes("opening-window exits:"));
  const blind = buildDailyLine(inputs({ firstFired: FIRST_FIRED, orders: null }));
  const sold = blind.opening_counterfactual?.exits[0];
  assert.deepEqual([sold?.priced_by, sold?.after, sold?.cost_pct], ["limit", 1.62, 10], "the limit sent (1.62), since no fill could be read");
  assert.match(String(blind.opening_counterfactual?.reason), /orders are unreadable: no fill is joined/);
  assert.match(blind.message, /waiting cost median \+10\.00% vs mid \(1 priced\)/);
});

test("stored and served: the block survives the line file and the sanitized daily_line probe body", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "daily-line-cf-"));
  try {
    const line = buildDailyLine(inputs({ firstFired: FIRST_FIRED }));
    assert.equal(line.opening_counterfactual?.exits.length, 2);
    assert.ok(saveDailyLine(dir, line));
    const stored = loadDailyLines(dir);
    assert.deepEqual(stored[0].opening_counterfactual, line.opening_counterfactual);
    const body = sanitizeDiag({ probe: "daily_line", ...dailyLineBody({ lines: stored, error: null }, "10", null, ms("2026-11-12T21:30:00Z"), { holidays: new Set<string>(), halfDays: new Set<string>() }) });
    assert.deepEqual(body.latest?.opening_counterfactual, line.opening_counterfactual, "nothing redacted, nothing cut by depth");
    assert.deepEqual(body.recent[0].opening_counterfactual, line.opening_counterfactual);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("wiring: bot.ts reads the first-fired file for the line; the join is pure; Python and Node name one file", () => {
  const bot = read("bot.ts");
  const a = bot.indexOf("async function computeDailyLine(");
  const compute = bot.slice(a, bot.indexOf("async function checkDailyLine(", a));
  assert.ok(a > 0 && compute.includes("firstFired: readJsonFile(path.join(dir, OPENING_FIRST_FIRED_FILENAME)),"), "one more file read, like the deferral count's");
  assert.ok(bot.includes('import { OPENING_FIRST_FIRED_FILENAME } from "./openingCounterfactual";'));
  const mod = read("openingCounterfactual.ts");
  const imports = mod.split("\n").filter((l) => /^import /.test(l));
  assert.deepEqual(imports, ['import { zonedParts } from "./jobHeartbeat";'], "no file, network or broker access: it is given everything it reads");
  const name = /export const OPENING_FIRST_FIRED_FILENAME = "([^"]+)";/.exec(mod)?.[1];
  assert.equal(name, "voltrade_option_opening_first_fired.json");
  assert.ok(read("../storage_config.py").includes(`OPTION_OPENING_FIRST_FIRED_PATH = os.path.join(DATA_DIR, "${name}")`), "storage_config.py owns the path");
  assert.ok(read("../options_manager.py").includes("return os.path.join(DATA_DIR, os.path.basename(OPTION_OPENING_FIRST_FIRED_PATH))"));
  const line = read("dailyLine.ts");
  assert.ok(line.includes("const counterfactual = openingCounterfactual(i.firstFired, i.day, i.ledger.rows, i.orders);"));
  assert.ok(line.includes("if (counterfactual) line.opening_counterfactual = counterfactual;"), "absent, not null, when there is nothing: the stored line is the line it was");
});
