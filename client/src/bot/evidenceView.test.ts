// evidenceView.test — the six evidence reads, parsed from the shapes the server
// answers with (taken from the live probes, 2026-11-20) and from the shapes it
// must survive: a route that is not there, a body of another shape, nulls.
import test from "node:test";
import assert from "node:assert/strict";
import {
  parseLots, lotsCsv, lotsCaption, lotSideWords, heldWords, LOTS_CSV_HEADER,
  parseOrderHistory, orderOutcome, orderAsk, orderStepWords, orderHistorySummary,
  parseFloorTracking, floorSentence, floorTriggerWords,
  parseRuleCounterfactuals, rulesSentence, ruleGroupWords,
  parseRegimeCounterfactual, regimeCfSentence, parseOptionsFeed,
} from "./evidenceView.ts";

const LOTS = {
  count: 2141, truncated: true, since: "2026-05-07T00:00:00.000Z", fetchedAt: "2026-11-20T10:01:37.731Z", pending: false, lastError: null,
  rows: [
    { symbol: "IREN261204C00023000", option: true, side: "short", qty: 3, closeDir: 1, openedAt: "2026-11-04T18:53:48.966Z", closedAt: "2026-11-04T19:57:41.836Z", basis: 783, pnl: -9.000000000000075, returnPct: -1.15, entryPrice: 2.61, exitPrice: 2.64 },
    { symbol: "CMCSA261218P00024000", option: true, side: "short", qty: 1, closeDir: 1, openedAt: "2026-11-04T20:27:56.447Z", closedAt: "2026-11-16T15:14:18.025Z", basis: 79, pnl: 41, returnPct: 51.9, entryPrice: 0.79, exitPrice: 0.38 },
    { symbol: "GLD", option: false, side: "long", qty: 12, closeDir: -1, openedAt: null, closedAt: "2026-11-06T14:47:56.000Z", basis: 4529.64, pnl: -81.48, returnPct: -1.8, entryPrice: 377.47, exitPrice: 370.68 },
  ],
};

test("closed lots: newest first, held time from the two fills, an open that predates the history has none", () => {
  const v = parseLots(LOTS)!;
  assert.equal(v.state, "ready");
  assert.deepEqual(v.rows.map((r) => r.symbol), ["CMCSA261218P00024000", "GLD", "IREN261204C00023000"]);
  assert.equal(v.rows[0].label, "CMCSA 24P 18-Dec");
  assert.equal(heldWords(v.rows[0].heldMs), "12d");
  assert.equal(heldWords(v.rows[2].heldMs), "1h");
  assert.equal(v.rows[1].heldMs, null);
  assert.equal(heldWords(null), "—");
  assert.equal(heldWords(20_000), "1m");
  assert.equal(lotSideWords(v.rows[0]), "sold 1 ct, bought back");
  assert.equal(lotSideWords(v.rows[1]), "bought 12, sold");
  assert.equal(v.count, 2141);
  assert.ok(v.truncated);
  assert.match(lotsCaption(v, 3), /^the newest 3 of 2,141 closed lots since May 7 /);
});

test("closed lots: the designed absences — still computing, nothing closed, not the ledger's shape", () => {
  assert.equal(parseLots({ count: 0, rows: [], pending: true })!.state, "pending");
  assert.equal(parseLots({ count: 0, rows: [], pending: false })!.state, "empty");
  assert.equal(parseLots(null), null);
  assert.equal(parseLots("<!doctype html>"), null);
  assert.equal(parseLots({ error: "boom" }), null);
  // rows that are not lots are dropped, never rendered as a lot of nothing
  const v = parseLots({ rows: [null, { symbol: "SPY" }, { symbol: "SPY", closedAt: "not a date" }, { symbol: "SPY", closedAt: "2026-11-06T14:47:57Z", pnl: "12.5" }] })!;
  assert.equal(v.rows.length, 1);
  assert.equal(v.rows[0].pnl, 12.5);
  assert.equal(v.rows[0].basis, null);
  assert.equal(lotsCaption(v, 1).startsWith("1 closed lot "), true);
});

test("the CSV: one header, one line per lot, the ledger's own figures, quoting that survives a spreadsheet", () => {
  const v = parseLots(LOTS)!;
  const csv = lotsCsv(v.rows);
  const lines = csv.trimEnd().split("\r\n");
  assert.equal(lines[0], LOTS_CSV_HEADER.join(","));
  assert.equal(lines.length, 4);
  assert.equal(lines[1], "2026-11-16T15:14:18.025Z,2026-11-04T20:27:56.447Z,CMCSA261218P00024000,yes,short,1,0.79,0.38,79,41,51.9");
  assert.equal(lines[2], "2026-11-06T14:47:56.000Z,,GLD,no,long,12,377.47,370.68,4529.64,-81.48,-1.8");
  assert.ok(csv.endsWith("\r\n"));
  // a symbol a spreadsheet would run as a formula is written as text; a comma or quote is quoted
  const odd = lotsCsv([{ ...v.rows[0], symbol: "=HYPERLINK(\"x\"),1" }]).split("\r\n")[1];
  assert.ok(odd.includes(`"'=HYPERLINK(""x""),1"`), odd);
  // a negative NUMBER stays a number
  assert.ok(lotsCsv([v.rows[1]]).includes(",-81.48,"));
});

const ORDERS = {
  count: 6,
  orders: [
    { id: "a3", symbol: "AAL261231P00011500", asset_class: "us_option", side: "sell", type: "limit", qty: 1, filled_qty: 0, filled_avg_price: null, limit_price: 0.3, status: "canceled", submitted_at: "2026-11-19T18:50:20.91992Z", filled_at: null, canceled_at: "2026-11-19T19:02:54.805327Z" },
    { id: "a2", symbol: "AAL261231P00011500", asset_class: "us_option", side: "sell", type: "limit", qty: 1, filled_qty: 0, filled_avg_price: null, limit_price: 0.29, status: "canceled", submitted_at: "2026-11-19T15:30:08.87254Z", filled_at: null, canceled_at: "2026-11-19T15:42:10Z" },
    { id: "c1", symbol: "CMCSA261218P00024000", asset_class: "us_option", side: "buy", type: "limit", qty: 1, filled_qty: 1, filled_avg_price: 0.38, limit_price: 0.38, status: "filled", submitted_at: "2026-11-16T15:14:17.995215Z", filled_at: "2026-11-16T15:14:18.025Z", canceled_at: null },
    { id: "s1", symbol: "SPY", asset_class: "us_equity", side: "buy", type: "limit", qty: 82, filled_qty: 82, filled_avg_price: 740.13, limit_price: 740.5, status: "filled", submitted_at: "2026-11-06T14:47:57.706793Z", filled_at: "2026-11-06T14:47:58.976Z", canceled_at: null },
    { id: "v1", symbol: "VXUS", asset_class: "us_equity", side: "sell", type: "market", qty: 30, filled_qty: 30, filled_avg_price: null, limit_price: null, status: "filled", submitted_at: "2026-11-05T17:33:12.562479Z", filled_at: "2026-11-05T17:33:12.9Z", canceled_at: null },
    { id: "w1", symbol: "T261218P00022000", asset_class: "us_option", side: "sell", type: "limit", qty: 2, filled_qty: 1, filled_avg_price: 0.44, limit_price: 0.44, status: "partially_filled", submitted_at: "2026-11-19T19:30:00Z", filled_at: null, canceled_at: null },
  ],
};

test("order history: newest first; two orders for one contract and side on one ET day are steps of one attempt", () => {
  const rows = parseOrderHistory(ORDERS)!;
  assert.deepEqual(rows.map((r) => r.id), ["w1", "a3", "a2", "c1", "s1", "v1"]);
  const [a3, a2] = [rows[1], rows[2]];
  assert.deepEqual([a2.step, a2.steps, a2.moved], [1, 2, null]);
  assert.deepEqual([a3.step, a3.steps, a3.moved], [2, 2, 0.01]);
  assert.equal(orderStepWords(a2), "step 1 of 2");
  assert.equal(orderStepWords(a3), "step 2 of 2 · +$0.01");
  assert.equal(orderStepWords(rows[3]), null);
  assert.equal(orderAsk(a3), "sell 1 ct · limit $0.30");
  assert.equal(orderAsk(rows[4]), "buy 82 · limit $740.50");
  assert.equal(orderAsk(rows[5]), "sell 30 · market");
});

test("order history: how each order ended, in words — and a fill with no price is not given one", () => {
  const rows = parseOrderHistory(ORDERS)!;
  assert.deepEqual(orderOutcome(rows[1]), { text: "cancelled after 13m, unfilled", tone: "none" });
  assert.deepEqual(orderOutcome(rows[3]), { text: "filled at $0.38", tone: "ok" });
  assert.deepEqual(orderOutcome(rows[5]), { text: "filled", tone: "ok" });
  assert.deepEqual(orderOutcome(rows[0]), { text: "working · 1 of 2 filled", tone: "wait" });
  assert.equal(orderOutcome({ ...rows[1], status: "new", endedAt: null }).text, "working");
  assert.equal(orderOutcome({ ...rows[1], status: "rejected" }).text, "rejected by the broker");
  assert.equal(orderOutcome({ ...rows[1], status: "canceled", filledQty: 1, qty: 3 }).text, "cancelled after 13m, 1 of 3 filled");
  assert.equal(orderHistorySummary(rows), "6 orders since Nov 5 · 3 filled · 3 not");
  assert.equal(orderHistorySummary([]), "no orders in the broker's recent history");
});

test("order history: the same contract on another day is another attempt; a bare array and a wrong shape are handled", () => {
  const two = parseOrderHistory([
    { id: "x1", symbol: "JD261218P00025000", side: "sell", type: "limit", qty: 1, limit_price: 0.62, status: "canceled", submitted_at: "2026-11-17T16:10:58Z", canceled_at: "2026-11-17T16:23:00Z" },
    { id: "x2", symbol: "JD261218P00025000", side: "sell", type: "limit", qty: 1, limit_price: 0.60, status: "canceled", submitted_at: "2026-11-18T16:10:58Z", canceled_at: "2026-11-18T16:23:00Z" },
    { id: "x3", symbol: "JD261218P00025000", side: "sell", type: "limit", qty: 1, limit_price: 0.58, status: "new", submitted_at: "2026-11-18T16:40:00Z" },
  ])!;
  assert.deepEqual(two.map((r) => [r.id, r.step, r.steps, r.moved]), [["x3", 2, 2, -0.02], ["x2", 1, 2, null], ["x1", 1, 1, null]]);
  assert.equal(orderStepWords(two[0]), "step 2 of 2 · −$0.02");
  assert.equal(two[0].option, true);
  assert.equal(parseOrderHistory({ error: "x" }), null);
  assert.equal(parseOrderHistory(undefined), null);
  assert.deepEqual(parseOrderHistory({ orders: [null, { symbol: "SPY" }] }), []);
});

const FLOOR = {
  version: 1, generated_at: "2026-11-20T10:04:39.983Z", status: "ok", reason: null, mode: "vol_managed_spy",
  config: { base: "SPY", levered: "SSO", cash_fund: "BOXX", capital_pct: 0.85, band: 0.1 },
  definitions: { tracking_pp: "floor_pct - rule_pct, in percentage points of the floor's capital" },
  entry: { day: "2026-11-06", shortfall_pp: -0.7225, note: "floor_pnl runs from the fills to the close" },
  latest: { day: "2026-11-19", entry: false, spy_pct: -1.984, target_exposure: 0.7039, held_exposure: 0.7935, drift: 0.0896, rule_pct: -1.3965, floor_pct: -1.5743, tracking_pp: -0.1778, rebalance: false, reason: null },
  rolling: { sessions: 9, from: "2026-11-09", to: "2026-11-19", floor_pct: -2.1184, rule_pct: -1.6541, tracking_pp: -0.4644, window_sessions: 20, complete: false },
  rollback_trigger: { text: "rollback trigger: live floor tracking vs its own rule > 3 pp over any 20 sessions", status: "undetermined", summary: "tracking: 9 measured sessions, a window needs 20 in a row; -0.46pp so far", tracking: { status: "not_enough_history", limit_pp: 3 } },
  turnover: { window: { sessions: 9, core: { orders: 0, filled: 0, dollars: 0 }, rebalance_sessions: 0, outside_band_no_trade: 0 } },
  sessions: [
    { day: "2026-11-19", entry: false, spy_pct: -1.984, target_exposure: 0.7039, held_exposure: 0.7935, drift: 0.0896, rule_pct: -1.3965, floor_pct: -1.5743, tracking_pp: -0.1778, rebalance: false, reason: null },
    { day: "2026-11-18", entry: false, spy_pct: -0.284, target_exposure: 0.6992, held_exposure: 0.7941, drift: 0.0949, rule_pct: -0.1986, floor_pct: -0.2255, tracking_pp: -0.027, rebalance: false, reason: null },
  ],
};

test("the base holding: the sentence quotes the report's own fields — the session, the exposure held, the period", () => {
  const f = parseFloorTracking(FLOOR)!;
  assert.equal(f.ok, true);
  assert.equal(f.sessions.length, 2);
  assert.equal(
    floorSentence(f),
    "Nov 19: the SPY holding −1.57% against −1.40% for its rule (0.70× of SPY −1.98%). It went in holding 0.79× against a target of 0.70×: inside the 0.10× band, where the rule does not trade. Over the last 9 sessions: −0.46 pt against the rule.",
  );
  assert.deepEqual(floorTriggerWords(f), { text: "Rollback line not yet testable: tracking: 9 measured sessions, a window needs 20 in a row; -0.46pp so far", tone: "wait" });
  assert.equal(f.turnover?.orders, 0);
  assert.equal(f.entry?.shortfallPp, -0.7225);
});

test("the base holding: outside the band, a met trigger, and no report are each said as they are", () => {
  const out = parseFloorTracking({ ...FLOOR, latest: { ...FLOOR.latest, held_exposure: 0.95, drift: 0.2461 } })!;
  assert.match(floorSentence(out), /outside the 0\.10× band; no rebalance filled that session\./);
  const reb = parseFloorTracking({ ...FLOOR, latest: { ...FLOOR.latest, held_exposure: 0.95, drift: 0.2461, rebalance: true } })!;
  assert.match(floorSentence(reb), /outside the 0\.10× band; it rebalanced that session\./);
  const met = parseFloorTracking({ ...FLOOR, rollback_trigger: { ...FLOOR.rollback_trigger, status: "met", summary: "tracking BEYOND 3pp", tracking: { status: "met", limit_pp: 3 } } })!;
  assert.deepEqual(floorTriggerWords(met), { text: "Rollback trigger MET: tracking BEYOND 3pp", tone: "warn" });
  const inside = parseFloorTracking({ ...FLOOR, rollback_trigger: { ...FLOOR.rollback_trigger, tracking: { status: "not_met", limit_pp: 3 } } })!;
  assert.equal(floorTriggerWords(inside)!.tone, "ok");
  const na = parseFloorTracking({ status: "not_applicable", reason: "FLOOR_MODE is basket", sessions: [] })!;
  assert.equal(floorSentence(na), "No tracking report: FLOOR_MODE is basket.");
  assert.equal(parseFloorTracking({ error: "x" }), null);
  assert.equal(parseFloorTracking([]), null);
  // a band the server could not read leaves the band clause out, it is never assumed
  const noBand = parseFloorTracking({ ...FLOOR, config: { ...FLOOR.config, band: null } })!;
  assert.match(floorSentence(noBand), /against a target of 0\.70×\. Over/);
});

const cell = (labeled: number, mean: number | null, baseMean: number | null, baseN: number, pending: number, usd: { sized: number; prevention: number } = { sized: 0, prevention: 0 }) => ({
  labeled, pending, win_rate_pct: 50, ...(mean === null ? {} : { return_pct: { mean } }),
  base_scanned_same_days: baseMean === null ? { n: 0 } : { n: baseN, mean_return_pct: baseMean },
  dollars: { estimate: true, sized_positions: usd.sized, prevention_pnl_dollars: usd.prevention },
});
const RULES = {
  via: "daemon", generated_at: "2026-11-20T10:04:41+00:00", min_n: 5, log: { first_day: "2026-09-03", last_day: "2026-11-20" },
  dollar_basis: "ESTIMATE. cut dollars x the stock price-path label's return.",
  limits: ["Outcomes are path labels of the UNDERLYING stock, long.", "Records from before 2026-11-12 carry no size."],
  gates: {
    exposure_ceiling: { what: "the regime's MAX_TOTAL_EXPOSURE", rows: 378, name_days: 45, names: 28, groups: [
      { tier: "1", action: "dropped", name_days: 45, first_day: "2026-10-29", last_day: "2026-11-17", "+5d": cell(14, -1, -0.623, 131, 19) },
      { tier: "1", action: "scaled", name_days: 3, first_day: "2026-11-17", last_day: "2026-11-17", "+5d": cell(0, null, null, 0, 3) },
    ] },
    daily_loss: { what: "the tier engine's daily loss limit", rows: 68, name_days: 15, names: 14, groups: [
      { tier: "1", action: "dropped", name_days: 15, first_day: "2026-10-29", last_day: "2026-11-05", "+5d": cell(13, -2.154, -2.026, 41, 0) },
    ] },
    drawdown: { what: "the tier engine's portfolio drawdown kill", rows: 0, name_days: 0, names: 0, groups: [] },
    cash_reserve: { what: "the regime's cash reserve", rows: 57, name_days: 16, names: 15, groups: [
      { tier: "unrecorded", action: "dropped", name_days: 7, first_day: "2026-11-06", last_day: "2026-11-18", "+5d": cell(3, 2, 0.838, 32, 4) },
      { tier: "scanner", action: "dropped", name_days: 4, first_day: "2026-11-09", last_day: "2026-11-13", "+5d": cell(8, 1.5, -1, 20, 0, { sized: 4, prevention: -312.4 }) },
    ] },
  },
};

test("rules: one row per rule, tier and action as the report groups them; a reading needs min_n judged outcomes", () => {
  const v = parseRuleCounterfactuals(RULES)!;
  assert.equal(v.ok, true);
  assert.deepEqual(v.rules.map((r) => [r.name, r.nameDays]), [["Exposure cap", 45], ["Daily loss limit", 15], ["Drawdown kill (tier engine)", 0], ["Cash reserve", 16]]);
  // best measured first
  assert.deepEqual(v.groups.map((g) => [g.gate, g.tier, g.action, g.labeled, g.verdict]), [
    ["exposure_ceiling", "1", "dropped", 14, "helped"],
    ["daily_loss", "1", "dropped", 13, "even"],
    ["cash_reserve", "scanner", "dropped", 8, "cost"],
    ["cash_reserve", "unrecorded", "dropped", 3, "thin"],
    ["exposure_ceiling", "1", "scaled", 0, "thin"],
  ]);
  // dollars only where a blocked name carried a size: an unsized group has no estimate, never $0
  assert.equal(v.groups[0].preventionUsd, null);
  assert.equal(v.groups[2].preventionUsd, -312.4);
  assert.equal(ruleGroupWords(v.groups[3]), "tier puts (tier not recorded), dropped");
  assert.equal(ruleGroupWords(v.groups[4]), "tier 1 puts, cut in size");
  assert.equal(
    rulesSentence(v),
    "3 rules blocked 76 name-days since Sep 3. Best measured: exposure cap (tier 1 puts, dropped): the 14 judged so far returned −1.0% on average over 5 trading days, worse than the −0.6% of everything scanned those days.",
  );
});

test("rules: nothing judged yet, nothing blocked, a failed report and an unknown shape each get their own sentence", () => {
  const thin = parseRuleCounterfactuals({ min_n: 5, gates: { cash_reserve: { name_days: 4, names: 4, groups: [{ tier: "1", action: "scaled", name_days: 4, "+5d": cell(0, null, null, 0, 4) }] } } })!;
  assert.equal(rulesSentence(thin), "1 rule blocked 4 name-days. None has 5 judged outcomes yet, so none can be called.");
  const none = parseRuleCounterfactuals({ gates: { drawdown: { name_days: 0, names: 0, groups: [] } } })!;
  assert.equal(rulesSentence(none), "No rule has blocked a trade in the logged period.");
  const failed = parseRuleCounterfactuals({ via: "subprocess", error: "gate_counterfactuals_probe failed" })!;
  assert.equal(failed.ok, false);
  assert.equal(rulesSentence(failed), "The rule report could not be computed: gate_counterfactuals_probe failed.");
  assert.equal(parseRuleCounterfactuals({ hello: 1 }), null);
  assert.equal(parseRuleCounterfactuals(null), null);
});

const REGIME_CF = {
  days: 14, since: "2026-11-07", rows: 3112, note: "legacy = what the pre-engine system would have done",
  by_day: [
    { date: "2026-11-18", sizing: { tier1: { n: 14, mean_legacy_multiplier: 0.8607, mean_engine_multiplier: 0.5 } }, selection: { n: 32, differs: 6, decisions: { block: 6, take: 26 }, blocks_by_guard: { knife: 6, rs: 1 }, size_cut: 0 }, options: { n: 15, differs: 9, decisions: { take: 15 }, blocks_by_guard: {}, size_cut: 9 }, label: { n: 28, differs: 28 } },
    { date: "2026-11-19", sizing: { tier1: { n: 14, mean_legacy_multiplier: 0.6643, mean_engine_multiplier: 0.5 } }, selection: { n: 18, differs: 9, decisions: { take: 9, block: 9 }, blocks_by_guard: { knife: 2, rs: 9 }, size_cut: 0 }, options: { n: 10, differs: 10, decisions: { take: 10 }, blocks_by_guard: {}, size_cut: 10 }, label: { n: 26, differs: 26 } },
    { date: "2026-11-20", sizing: { satellite_reserve: { n: 16 } }, selection: { n: 0, differs: 0, decisions: {}, blocks_by_guard: {}, size_cut: 0 }, options: { n: 0, differs: 0, decisions: {}, blocks_by_guard: {}, size_cut: 0 }, label: { n: 16, differs: 16 } },
  ],
};

test("regime engine vs the old rule: the daily counts added up, the newest tier-1 sizing, and no P&L claimed", () => {
  const v = parseRegimeCounterfactual(REGIME_CF)!;
  assert.deepEqual([v.decisions, v.differs, v.blocked, v.sizeCuts, v.relabelled], [75, 34, 15, 19, 70]);
  assert.deepEqual(v.blocks, [{ guard: "rs", n: 10 }, { guard: "knife", n: 8 }]);
  assert.deepEqual(v.tier1, { date: "2026-11-19", legacy: 0.6643, engine: 0.5, n: 14 });
  assert.equal(
    regimeCfSentence(v),
    "Against the old rule, the last 14 days: 34 of 75 entry decisions came out differently; 15 entries blocked (weak against the market 10, falling knife 8; one entry can trip several); 19 option entries cut in size; tier 1 sized at 0.50× against 0.66× under the old rule on Nov 19. Counts only: what the blocked names went on to do is not in this report.",
  );
  const empty = parseRegimeCounterfactual({ days: 14, since: "2026-11-07", rows: 0, by_day: [] })!;
  assert.equal(regimeCfSentence(empty), "Against the old rule, the last 14 days: no decision point logged yet.");
  assert.equal(parseRegimeCounterfactual({ rows: 3 }), null);
});

test("option quote feed: indicative is said plainly, an inconclusive check is not turned into a claim", () => {
  const ind = parseOptionsFeed({ feed: "indicative", opra: "not_entitled", indicative: "entitled", checked_at: "2026-11-20T10:04:54.963Z" })!;
  assert.equal(ind.tone, "warn");
  assert.match(ind.text, /^option quotes are INDICATIVE: /);
  assert.equal(parseOptionsFeed({ feed: "opra" })!.tone, "ok");
  assert.equal(parseOptionsFeed({ feed: "none" })!.tone, "warn");
  const unk = parseOptionsFeed({ feed: "unknown", opra: "rate_limited_inconclusive" })!;
  assert.equal(unk.tone, "none");
  assert.match(unk.text, /could not be checked/);
  assert.equal(parseOptionsFeed({ feed: "something-new" })!.feed, "unknown");
  assert.equal(parseOptionsFeed({}), null);
  assert.equal(parseOptionsFeed(null), null);
});
