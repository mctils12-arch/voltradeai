// OPENING-WINDOW COUNTERFACTUAL (2026-11-12, measurement only): the join of
// options_manager's first-fired records with the fill ledger and the day's
// orders. research/open_questions.md, "opening-window exits: the
// counterfactual to log".
// Run: npx tsx --test server/openingCounterfactual.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  OPENING_COUNTERFACTUAL_MAX_EXITS, WAITING_COST_SIGN, ledgerRowExtras, openingCounterfactual, openingCounterfactualText,
  parseFirstFiredLink, type CounterfactualRow, type FirstFiredLink, type OpeningCounterfactual,
} from "./openingCounterfactual";

const DAY = "2026-11-12";
const AT = "2026-11-12T09:31:07-05:00";          // 14:31:07Z
const SELL = "BLOS261218C00520000";              // a bought call, sold to close
const BUY = "IREN261211P00018000";               // a sold put, bought back
const XLF = "XLF261127P00055500";                // the live case: 0 / 5.00 at the open
const DROP = "GAMA261204C00185000";              // the rule no longer held after 09:45
const REFUSED = "ASGN261120P00060000";           // the broker refused its close
const LEGS = ["COND261127C00355000", "COND261127C00365000", "COND261127P00290000", "COND261127P00300000"];
const GROUP = LEGS.join(",");

const leg = (symbol: string, side: string, qty: number, bid: number | null, ask: number | null) =>
  ({ symbol, side, qty, bid, ask, mid: bid !== null && ask !== null && bid > 0 && ask >= bid ? Math.round(((bid + ask) / 2) * 1e4) / 1e4 : null, mark: ask, quote_age_s: 0.4 });
const record = (rule: string, kind: string, legs: unknown[], over: Record<string, unknown> = {}) =>
  ({ kind, rule, rules: [rule], first_fired_at: AT, last_fired_at: "2026-11-12T09:44:02-05:00", fires: 3, refused: 0, feed: "indicative", legs, ...over });

const DOC = { version: 1, updated_at: "2026-11-12T09:44:02", days: {
  [DAY]: {
    [SELL]: record("bought_loss_limit", "single_leg", [leg(SELL, "sell", 2, 1.60, 2.00)]),
    [BUY]: record("profit_target", "single_leg", [leg(BUY, "buy", 1, 0.40, 0.60)]),
    [XLF]: record("loss_limit", "single_leg", [leg(XLF, "buy", 1, 0, 5.00)], { fires: 1 }),
    [DROP]: record("gamma_risk", "single_leg", [leg(DROP, "sell", 1, 4.30, 4.70)], { fires: 1 }),
    [REFUSED]: record("assignment_close", "single_leg", [leg(REFUSED, "buy", 1, 5.90, 6.10)], { refused: 2 }),
    [GROUP]: record("strategy_dte_exit", "group", [
      leg(LEGS[0], "buy", 2, 4.30, 4.50), leg(LEGS[1], "sell", 2, 2.70, 2.90), leg(LEGS[2], "sell", 2, 3.00, 3.20), leg(LEGS[3], "buy", 2, 4.70, 4.90),
    ]),
  },
  "2026-11-11": { OLD261127P00010000: record("loss_limit", "single_leg", [leg("OLD261127P00010000", "buy", 1, 0.5, 0.6)], { first_fired_at: "2026-11-11T09:33:00-05:00" }) },
} };

const link = (key: string, rule: string, kind: string, bid: number | null, ask: number | null, mid: number | null): FirstFiredLink => ({ key, at: AT, rule, kind, bid, ask, mid });
const row = (order_id: string, symbol: string, side: string, qty: number, limit: number | null, ts: string, first_fired?: FirstFiredLink): CounterfactualRow =>
  ({ order_id, symbol, side, qty, ts, limit, ...(first_fired ? { first_fired } : {}) });

const SENT_AT = "2026-11-12T14:46:03+00:00";     // 09:46:03 ET
const ROWS: CounterfactualRow[] = [
  // a mid-first exit ladder: step 1 at the mid, then a replace (a new order id) at 1.70, which filled
  row("lad-1", SELL, "sell", 2, 1.80, SENT_AT, link(SELL, "bought_loss_limit", "single_leg", 1.60, 2.00, 1.80)),
  row("lad-2", SELL, "sell", 2, 1.70, "2026-11-12T14:47:10+00:00", link(SELL, "bought_loss_limit", "single_leg", 1.60, 2.00, 1.80)),
  row("buy-1", BUY, "buy", 1, 0.50, "2026-11-12T14:46:05+00:00", link(BUY, "profit_target", "single_leg", 0.40, 0.60, 0.50)),
  row("xlf-1", XLF, "buy", 1, 0.51, "2026-11-12T14:46:06+00:00", link(XLF, "loss_limit", "single_leg", 0, 5.00, null)),
  // the group's one mleg order: a row per leg under the parent id
  row("mleg-1", LEGS[2], "sell", 2, null, "2026-11-12T14:46:08+00:00", link(GROUP, "strategy_dte_exit", "group", 3.00, 3.20, 3.10)),
  row("mleg-1", LEGS[3], "buy", 2, null, "2026-11-12T14:46:08+00:00", link(GROUP, "strategy_dte_exit", "group", 4.70, 4.90, 4.80)),
  row("mleg-1", LEGS[0], "buy", 2, null, "2026-11-12T14:46:08+00:00", link(GROUP, "strategy_dte_exit", "group", 4.30, 4.50, 4.40)),
  row("mleg-1", LEGS[1], "sell", 2, null, "2026-11-12T14:46:08+00:00", link(GROUP, "strategy_dte_exit", "group", 2.70, 2.90, 2.80)),
  // an entry on another contract, and yesterday's linked close: neither is today's exit
  row("entry-1", "TLT261218C00090000", "buy", 1, 1.10, "2026-11-12T15:00:00+00:00"),
  row("old-1", "OLD261127P00010000", "buy", 1, 0.55, "2026-11-11T14:50:00+00:00", { key: "OLD261127P00010000", at: "2026-11-11T09:33:00-05:00", rule: "loss_limit", kind: "single_leg", bid: 0.5, ask: 0.6, mid: 0.55 }),
];

const ORDERS: unknown[] = [
  { id: "lad-1", symbol: SELL, side: "sell", status: "replaced", replaced_by: "lad-2", filled_qty: "0", filled_avg_price: null },
  { id: "lad-2", symbol: SELL, side: "sell", status: "filled", replaces: "lad-1", filled_qty: "2", filled_avg_price: "1.70", filled_at: "2026-11-12T14:48:00Z" },
  { id: "buy-1", symbol: BUY, side: "buy", status: "filled", filled_qty: "1", filled_avg_price: "0.46", filled_at: "2026-11-12T14:46:30Z" },
  { id: "xlf-1", symbol: XLF, side: "buy", status: "filled", filled_qty: "1", filled_avg_price: "0.51", filled_at: "2026-11-12T14:46:40Z" },
  { id: "mleg-1", order_class: "mleg", symbol: "", side: "", status: "filled", filled_qty: "2", filled_avg_price: "3.36", legs: [
    { id: "m-a", symbol: LEGS[0], side: "buy", filled_qty: "2", filled_avg_price: "4.48" },
    { id: "m-b", symbol: LEGS[1], side: "sell", filled_qty: "2", filled_avg_price: "2.72" },
    { id: "m-c", symbol: LEGS[2], side: "sell", filled_qty: "2", filled_avg_price: "3.02" },
    { id: "m-d", symbol: LEGS[3], side: "buy", filled_qty: "2", filled_avg_price: "4.86" },
  ] },
  { id: "entry-1", symbol: "TLT261218C00090000", side: "buy", status: "filled", filled_qty: "1", filled_avg_price: "1.10" },
];

function block(doc: unknown = DOC, rows: readonly CounterfactualRow[] = ROWS, orders: unknown[] | null = ORDERS): OpeningCounterfactual {
  const b = openingCounterfactual(doc, DAY, rows, orders);
  assert.ok(b, "a day with records has a block");
  return b;
}
const exit = (b: OpeningCounterfactual, key: string) => {
  const x = b.exits.find((e) => e.key === key);
  assert.ok(x, `exit ${key}`);
  return x;
};

test("counts: the day's records are the exits first fired; sent = a later order's row, never sent = none, refused apart", () => {
  const b = block();
  assert.deepEqual([b.first_fired, b.sent, b.never_sent, b.refused], [6, 4, 2, 1]);
  assert.deepEqual(b.exits.map((x) => x.key).sort(), [SELL, BUY, XLF, DROP, REFUSED, GROUP].sort(), "yesterday's record is not today's");
  const dropped = exit(b, DROP);
  assert.deepEqual([dropped.sent, dropped.refused, dropped.note, dropped.cost_pct, dropped.orders], [false, 0, null, null, 0],
    "the rule no longer held after 09:45: counted, and given no price");
  assert.deepEqual([dropped.rule, dropped.first_fired_at, dropped.fires, dropped.side, dropped.first_mid], ["gamma_risk", AT, 1, "sell", 4.5]);
  const refused = exit(b, REFUSED);
  assert.deepEqual([refused.sent, refused.refused], [false, 2]);
  assert.match(String(refused.note), /2 close\(s\) refused by the broker/, "a refused exit is not read as one the rule let go");
  assert.equal(b.sign, WAITING_COST_SIGN);
  assert.match(b.sign, /> 0: the later order paid more \(buy-to-close\) or received less \(sell-to-close\)/);
  assert.equal(b.reason, null);
});

test("sign: a SELL-to-close filled under the first-fired mid COST money; a BUY-to-close filled under it SAVED money", () => {
  const b = block();
  const sell = exit(b, SELL);
  // sold 2 at 1.70 against a first-fired 1.60 / 2.00 (mid 1.80): 0.10 less per contract than the mid, 0.10 more than the bid
  assert.deepEqual([sell.side, sell.first_bid, sell.first_ask, sell.first_mid, sell.after, sell.priced_by], ["sell", 1.6, 2, 1.8, 1.7, "fill"]);
  assert.deepEqual([sell.cost_usd, sell.cost_pct], [20, 5.56], "received $20 less than the mid: waiting cost money");
  assert.deepEqual([sell.touch_cost_usd, sell.touch_cost_pct], [-20, -6.25], "received $20 more than the bid an order then would have hit");
  const buy = exit(b, BUY);
  // bought back 1 at 0.46 against a first-fired 0.40 / 0.60 (mid 0.50)
  assert.deepEqual([buy.side, buy.first_mid, buy.after, buy.priced_by], ["buy", 0.5, 0.46, "fill"]);
  assert.deepEqual([buy.cost_usd, buy.cost_pct], [-4, -8], "paid $4 less than the mid: waiting saved money");
  assert.deepEqual([buy.touch_cost_usd, buy.touch_cost_pct], [-14, -23.33]);
  // the mirror cases, so the sign cannot be an accident of these two fills
  const dearer = block(DOC, ROWS, ORDERS.map((o) => ((o as { id: string }).id === "buy-1" ? { ...(o as object), filled_avg_price: "0.58" } : o)));
  assert.deepEqual([exit(dearer, BUY).cost_usd, exit(dearer, BUY).cost_pct], [8, 16], "a buy-to-close that paid MORE than the mid cost money");
  const richer = block(DOC, ROWS, ORDERS.map((o) => ((o as { id: string }).id === "lad-2" ? { ...(o as object), filled_avg_price: "1.95" } : o)));
  assert.deepEqual([exit(richer, SELL).cost_usd, exit(richer, SELL).cost_pct], [-30, -8.33], "a sell-to-close that received MORE than the mid saved money");
});

test("a laddered exit: the first order and its replace are one exit, and its fill is counted once", () => {
  const sell = exit(block(), SELL);
  assert.deepEqual([sell.orders, sell.via, sell.sent_at, sell.delay_min], [2, "link", "2026-11-12T14:46:03.000Z", 14.9]);
  assert.equal(sell.cost_usd, 20, "2 contracts, not 4: lad-1 leads to lad-2, and lad-2's own row is the same order");
  // a partly filled order that was then replaced keeps its own fill
  const partial = block(DOC, ROWS, ORDERS.map((o) => {
    const id = (o as { id: string }).id;
    if (id === "lad-1") return { ...(o as object), filled_qty: "1", filled_avg_price: "1.80" };
    if (id === "lad-2") return { ...(o as object), filled_qty: "1" };
    return o;
  }));
  assert.deepEqual([exit(partial, SELL).after, exit(partial, SELL).cost_usd], [1.75, 10], "1 at 1.80 and 1 at 1.70");
});

test("a one-sided first-fired quote has no mid: the exit is sent, priced against the touch only, and says why", () => {
  const xlf = exit(block(), XLF);
  assert.deepEqual([xlf.sent, xlf.first_bid, xlf.first_ask, xlf.first_mid, xlf.after, xlf.priced_by], [true, 0, 5, null, 0.51, "fill"]);
  assert.deepEqual([xlf.cost_usd, xlf.cost_pct], [null, null], "no mid is guessed from a 0 / 5.00 quote");
  assert.match(String(xlf.note), /no two-sided first-fired quote/);
  assert.deepEqual([xlf.touch_cost_usd, xlf.touch_cost_pct], [-449, -89.8], "bought back at 0.51 where the 09:31 ask was 5.00");
});

test("a group: each leg against its own first-fired quote, summed; never a partial net", () => {
  const b = block();
  const g = exit(b, GROUP);
  assert.deepEqual([g.kind, g.rule, g.legs, g.sent, g.orders, g.priced_by], ["group", "strategy_dte_exit", 4, true, 1, "fill"]);
  assert.deepEqual([g.side, g.first_mid, g.after], [null, null, null], "the per-contract fields are a single leg's");
  // buys: (4.48 − 4.40) + (4.86 − 4.80) = +0.14; sells: −(2.72 − 2.80) − (3.02 − 3.10) = +0.16; x 2 x 100
  assert.deepEqual([g.cost_usd, g.cost_pct], [60, 1.99], "$60 on $3,020 of first-fired mid value");
  // touches: buys at the ask (4.50, 4.90), sells at the bid (2.70, 3.00): −0.02 − 0.04 − 0.02 − 0.02 = −0.10
  assert.deepEqual([g.touch_cost_usd, g.touch_cost_pct], [-20, -0.66]);
  // one leg with no leg fill: the whole exit is unpriced rather than a net of three legs
  const noLeg = block(DOC, ROWS, ORDERS.map((o) => ((o as { id: string }).id === "mleg-1"
    ? { ...(o as object), legs: ((o as { legs: unknown[] }).legs).slice(0, 3) } : o)));
  const u = exit(noLeg, GROUP);
  assert.deepEqual([u.sent, u.priced_by, u.cost_usd, u.touch_cost_usd], [true, null, null, null]);
  assert.match(String(u.note), /no fill and no limit recorded/);
  assert.equal(noLeg.unpriced, 1);
});

test("the day: mean and median of the priced exits, dollars summed, what each was priced at", () => {
  const b = block();
  assert.deepEqual([b.by_fill, b.by_limit, b.unpriced], [4, 0, 0]);
  // vs mid: SELL +5.56, BUY −8.00, GROUP +1.99 (XLF has no mid)
  assert.deepEqual(b.vs_mid, { priced: 3, mean_pct: -0.15, median_pct: 1.99, usd: 76 });
  // vs touch: SELL −6.25, BUY −23.33, XLF −89.80, GROUP −0.66
  assert.deepEqual(b.vs_touch, { priced: 4, mean_pct: -30.01, median_pct: -14.79, usd: -503 });
  assert.equal(b.median_delay_min, 15, "minutes from the first fire to the first order: 14.9, 15.0, 15.0, 15.0");
  assert.equal(b.exits_omitted, 0);
});

test("no fill: the exit is priced at the last limit sent, counted apart, and the unreadable orders are named", () => {
  const b = block(DOC, ROWS, null);
  assert.match(String(b.reason), /orders are unreadable: no fill is joined/);
  const sell = exit(b, SELL);
  assert.deepEqual([sell.priced_by, sell.after, sell.cost_usd], ["limit", 1.7, 20], "the last limit (the replace at 1.70), not the first");
  assert.deepEqual([b.by_fill, b.by_limit, b.unpriced], [0, 3, 1], "an mleg row has no limit: the group is sent, not priced");
  assert.match(String(exit(b, GROUP).note), /no limit recorded, and the day's orders are unreadable/);
  // orders readable but the order unfilled all day: still the limit
  const resting = block(DOC, ROWS, ORDERS.map((o) => ((o as { id: string }).id === "buy-1" ? { ...(o as object), status: "expired", filled_qty: "0", filled_avg_price: null } : o)));
  assert.deepEqual([exit(resting, BUY).priced_by, exit(resting, BUY).after, exit(resting, BUY).cost_pct], ["limit", 0.5, 0]);
  assert.deepEqual([resting.by_fill, resting.by_limit], [3, 1]);
});

test("another order path's close (no record on its row): the same contract and side after the first fire counts as sent", () => {
  const node = (ts: string, side = "sell", symbol = DROP) => row("pk-1", symbol, side, 1, null, ts);
  const orders = [...ORDERS, { id: "pk-1", symbol: DROP, side: "sell", status: "filled", filled_qty: "1", filled_avg_price: "4.20" }];
  const b = block(DOC, [...ROWS, node("2026-11-12T14:47:00.000Z")], orders);
  const x = exit(b, DROP);
  assert.deepEqual([x.sent, x.via, x.priced_by, x.after, x.cost_usd, x.cost_pct], [true, "match", "fill", 4.2, 30, 6.67]);
  assert.deepEqual([b.sent, b.never_sent], [5, 1]);
  assert.equal(exit(b, SELL).via, "link");
  // where the manager's own orders carry the record, another path's row for the same contract is not added to them
  const extra = block(DOC, [...ROWS, row("pk-2", SELL, "sell", 2, null, "2026-11-12T19:00:00.000Z")],
    [...ORDERS, { id: "pk-2", symbol: SELL, side: "sell", status: "filled", filled_qty: "2", filled_avg_price: "1.00" }]);
  assert.deepEqual([exit(extra, SELL).orders, exit(extra, SELL).after, exit(extra, SELL).cost_usd], [2, 1.7, 20]);
  for (const [why, r] of [
    ["before the first fire", node("2026-11-12T14:20:00.000Z")],
    ["the next ET day", node("2026-11-13T14:47:00.000Z")],
    ["the other side (it adds to the position)", node("2026-11-12T14:47:00.000Z", "buy")],
    ["another contract", node("2026-11-12T14:47:00.000Z", "sell", "GAMA261204C00190000")],
    ["no time on the row", { order_id: "pk-1", symbol: DROP, side: "sell", qty: 1 }],
  ] as Array<[string, CounterfactualRow]>) {
    assert.equal(exit(block(DOC, [...ROWS, r], orders), DROP).sent, false, why);
  }
});

test("the file gone but rows carry the record: the sent exits are rebuilt from the rows, and the gap is named", () => {
  const b = block(null, ROWS, ORDERS);
  assert.deepEqual([b.first_fired, b.sent, b.never_sent], [4, 4, 0]);
  assert.match(String(b.reason), /first-fired file is unreadable: exits never sent are not counted/);
  const sell = exit(b, SELL);
  assert.deepEqual([sell.kind, sell.rule, sell.fires, sell.first_mid, sell.cost_usd], ["single_leg", "bought_loss_limit", null, 1.8, 20]);
  assert.deepEqual([exit(b, GROUP).kind, exit(b, GROUP).legs, exit(b, GROUP).cost_usd], ["group", 4, 60]);
  for (const bad of [null, undefined, 7, "x", [], { days: [] }, { days: "x" }]) {
    assert.equal(openingCounterfactual(bad, DAY, [], ORDERS), null, "no file and no row: nothing was recorded");
  }
  for (const quiet of [{ version: 1, days: {} }, { version: 1, days: { "2026-11-11": DOC.days["2026-11-11"] } }, { days: { [DAY]: {} } }]) {
    assert.equal(openingCounterfactual(quiet, DAY, [], ORDERS), null, "a readable file with no record for the day: nothing fired, no block");
  }
  // junk inside a readable file never throws: what is not a record is skipped, a record with unreadable legs is still an exit
  const junk = openingCounterfactual({ days: { [DAY]: { A: null, B: 5, C: ["x"], D: { rule: "gamma_risk", legs: "x" }, E: { legs: [null, { symbol: "" }] } } } }, DAY, [], null);
  assert.deepEqual([junk?.first_fired, junk?.sent, junk?.exits.map((x) => [x.key, x.legs, x.rule])], [2, 0, [["D", 0, "gamma_risk"], ["E", 0, ""]]]);
});

test("the list on a line is capped and the rest counted", () => {
  const many: Record<string, unknown> = {};
  for (let i = 0; i < OPENING_COUNTERFACTUAL_MAX_EXITS + 7; i++) {
    const occ = `Z${String(i).padStart(2, "0")}261218C00010000`;
    many[occ] = record("gamma_risk", "single_leg", [leg(occ, "sell", 1, 1, 1.2)]);
  }
  const b = block({ days: { [DAY]: many } }, [], ORDERS);
  assert.deepEqual([b.first_fired, b.exits.length, b.exits_omitted], [47, OPENING_COUNTERFACTUAL_MAX_EXITS, 7]);
});

test("the stored row: the link and the three optional fields, each only when the row has it", () => {
  const stored = { key: SELL, at: AT, rule: "bought_loss_limit", kind: "single_leg", bid: 1.6, ask: 2, mid: 1.8 };
  assert.deepEqual(parseFirstFiredLink(stored), stored);
  assert.deepEqual(parseFirstFiredLink({ key: XLF, at: AT, bid: "0.0", ask: 5, mid: null }), { key: XLF, at: AT, rule: "", kind: "", bid: 0, ask: 5, mid: null });
  for (const bad of [null, undefined, "x", [], {}, { key: SELL }, { at: AT }]) assert.equal(parseFirstFiredLink(bad), null);
  assert.deepEqual(ledgerRowExtras({ order_id: "o", symbol: SELL, limit: null, quote_age_s: null }), {}, "a row with none of them gains nothing");
  assert.deepEqual(ledgerRowExtras({ ts: SENT_AT, limit: 1.8, first_fired: stored }), { ts: SENT_AT, limit: 1.8, first_fired: stored });
  assert.deepEqual(ledgerRowExtras({ ts: SENT_AT, limit: "x", first_fired: { key: "" } }), { ts: SENT_AT });
});

test("the audit text: only when an exit first fired; the cost is signed (+ = waiting cost money)", () => {
  assert.equal(openingCounterfactualText(null), "");
  assert.equal(openingCounterfactualText(undefined), "");
  assert.equal(openingCounterfactualText(openingCounterfactual({ days: {} }, DAY, [], ORDERS)), "");
  assert.equal(openingCounterfactualText({ ...block(), first_fired: 0 }), "");
  assert.equal(openingCounterfactualText(block()),
    "opening-window exits: 6 first fired, 4 sent, 2 never (1 refused); waiting cost median +1.99% vs mid (3 priced), -14.79% vs touch (4 priced)");
  // only the touch is priced (the one-sided open): the mid says n/a rather than borrowing the touch's number
  assert.equal(openingCounterfactualText(block({ days: { [DAY]: { [XLF]: DOC.days[DAY][XLF] } } }, ROWS.filter((r) => r.symbol === XLF), ORDERS)),
    "opening-window exits: 1 first fired, 1 sent, 0 never; waiting cost median n/a vs mid (0 priced), -89.80% vs touch (1 priced)");
  assert.equal(openingCounterfactualText(block({ days: { [DAY]: { [DROP]: DOC.days[DAY][DROP] } } }, [], ORDERS)),
    "opening-window exits: 1 first fired, 0 sent, 1 never; waiting cost n/a");
});
