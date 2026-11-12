// The daily line card's words ([PRODUCT] 2026-11-06): GET /api/bot/daily-line -> rows, states, today's line.
// Run: npx tsx --test client/src/bot/dailyLineView.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { dailyLineView, dailyRow, todayLine, sessionLabel, rowTitle, rowNotes, optWords, optSplitWords, optSplitParts, gapWords, pctWords, acctWords, acctBasisWords, checkWords } from "./dailyLineView";
import { MINUS } from "./format";

// the live line of 2026-11-05 (the daily_line diag probe, read 2026-11-06 06:00 ET), trimmed to the fields the card reads
const NOV5 = {
  version: 1, day: "2026-11-05", written_at: "2026-11-06T01:12:26.339Z", session_close: "16:00",
  account_vs_spy: { basis: "account: Alpaca equity vs last_equity (no dividends in the paper account) · SPY: PRICE-ONLY daily closes (Alpaca bars, adjustment=split, feed iex)",
    day: { account_pct: -4.07, spy_pct: -4.34, gap_pp: 0.27, equity: 89814.42, last_equity: 93623.46, spy_bar_date: "2026-11-05", reason: null } },
  floor: { mode: "vol_managed_spy", exposure_x: 0, basis: "(SPY + 2 x SSO market value) / equity", base_mv: 0, levered_mv: 0, floor_mv: 0, legacy_mv: 26603.7, target_x: null, target_at: null, reason: null },
  options: { window: { from: "2026-11-05T05:00:00.000Z", to: "2026-11-06T01:12:26.339Z", from_line: null, single_session: true }, realized: 948, realized_lots: 3, unrealized_start: null, unrealized_end: -18,
    closed_without_fill: [], total: null, fills: { fills: 3, attributed: 0, cost: 0, pnl: 0, unattributed: { count: 3, no_row: 3, no_mid: 0, mleg_no_leg_fills: 0, notional: 2584 } }, direction: null,
    reason: "no previous line within 7 days to start from (the option total starts with the next line)" },
  opening_window: { total: 0, manager_single_leg: 0, manager_group: 0, pos_kill: 0, pos_kill_tickers: [], keys: [], reason: null },
  missing: [],
};
// a later day with every piece computed: the vol-managed floor with its target, an option total split into fills and direction
// (and, as a line written since the [RULE-REVIEW] 2026-11-06 close reading, its account day at the close)
const NOV6 = {
  ...NOV5, day: "2026-11-06", written_at: "2026-11-06T21:22:10.000Z",
  account_vs_spy: { ...NOV5.account_vs_spy, day: { account_pct: 0.42, spy_pct: 0.61, gap_pp: -0.19, equity: 90100, last_equity: 89724.1, spy_bar_date: "2026-11-06", reason: null,
    account_source: "close", equity_end: 90100, equity_end_at: "2026-11-06T20:59:52.000Z", close_reason: null } },
  floor: { ...NOV5.floor, exposure_x: 0.8512, target_x: 0.9204, target_at: "2026-11-06T20:50:00Z" },
  options: { ...NOV5.options, window: { from: NOV5.written_at, to: "2026-11-06T21:22:10.000Z", from_line: "2026-11-05", single_session: true }, total: -120.4, direction: -80.1,
    fills: { fills: 5, attributed: 2, cost: 40.3, pnl: -40.3, unattributed: { count: 3, no_row: 3, no_mid: 0, mleg_no_leg_fills: 0, notional: 900 } }, reason: null },
  opening_window: { total: 3, manager_single_leg: 1, manager_group: 1, pos_kill: 1, pos_kill_tickers: ["AMD"], keys: [], reason: null },
};
const noLeak = (v: unknown) => assert.ok(!/NaN|undefined|\bnull\b|Infinity/.test(JSON.stringify(v).replace(/"[a-zA-Z]+":null/g, "")), `a NaN / undefined / null leaked into the words: ${JSON.stringify(v)}`);
const body = (recent: unknown[], extra: Record<string, unknown> = {}) => ({ count: recent.length, latest: recent[0] ?? null, recent, lastAttempt: null, schedule: { day: "2026-11-06", session: true, due_at: "2026-11-06T21:20:00.000Z" }, keep: 60, error: null, ...extra });
const Q = { loading: false, error: null };

test("sessionLabel: an ET session date as the card names it", () => {
  assert.equal(sessionLabel("2026-11-05"), "Thu Nov 5");
  assert.equal(sessionLabel("2026-11-27"), "Fri Nov 27");
  for (const bad of [null, undefined, "", "Nov 5", "2026-11-5"]) assert.equal(sessionLabel(bad), "—");
});

test("a row is the line's own fields: account vs SPY, the floor, the options split, the opening-window holds — a missing piece stays a dash with its reason", () => {
  const r = dailyRow(NOV5)!;
  assert.equal(r.label, "Thu Nov 5");
  assert.deepEqual([r.acct, r.spy, r.gap, r.dayReason], [-4.07, -4.34, 0.27, null]);
  assert.equal(r.floor, "0.00×");
  assert.equal(r.floorTarget, null);
  assert.equal(r.optTotal, null, "no previous line: the option total is not computable, never a zero");
  assert.equal(r.optFills, 0);
  assert.equal(r.priced, "0/3 priced");
  assert.match(r.optReason ?? "", /^no previous line within 7 days/);
  assert.equal(r.optSince, null);
  assert.deepEqual([r.blocks, r.blocksDetail], [0, "manager 0 · POS-KILL 0"]);
  assert.deepEqual(r.partial, []);
  assert.equal(optWords(r), "n/a");
  assert.equal(optSplitWords(r), "fills $0 (0/3 priced)");
  assert.equal(pctWords(r.acct), `${MINUS}4.07%`);
  assert.equal(gapWords(r.gap), "+0.27 pp");
  const full = dailyRow(NOV6)!;
  assert.equal(full.floor, "0.85×");
  assert.equal(full.floorTarget, 0.9204);
  assert.equal(optWords(full), `${MINUS}$120`);
  assert.equal(optSplitWords(full), `fills ${MINUS}$40 · direction ${MINUS}$80 (2/5 priced)`);
  assert.deepEqual(optSplitParts(full), { main: `fills ${MINUS}$40 · direction ${MINUS}$80`, priced: "(2/5 priced)" }, "the card keeps the second part unbroken");
  assert.deepEqual(optSplitParts(r), { main: "fills $0", priced: "(0/3 priced)" });
  assert.deepEqual([full.blocks, full.blocksDetail], [3, "manager 2 · POS-KILL 1"]);
  assert.match(rowTitle(full), /^Fri Nov 6: account \+0\.42% vs SPY \+0\.61% \(−0\.19 pp\) · floor 0\.85× \(target 0\.92×\) · options −\$120 = fills −\$40 · direction −\$80 \(2\/5 priced\) · opening-window blocks 3 \(manager 2 · POS-KILL 1\) · written Nov 6 16:22 ET$/);
  noLeak(r); noLeak(full);
});

test("basket mode reads in equity terms, a missed line widens the option window, a partial line names its unreadable inputs", () => {
  const basket = dailyRow({ ...NOV5, floor: { ...NOV5.floor, mode: "basket", exposure_x: 0.4012 } })!;
  assert.equal(basket.floor, "0.40× eq", "not SPY-equivalent: said so");
  const missed = dailyRow({ ...NOV6, options: { ...NOV6.options, window: { ...NOV6.options.window, from_line: "2026-11-02", single_session: false } } })!;
  assert.equal(missed.optSince, "Mon Nov 2");
  assert.match(rowTitle(missed), /since the Mon Nov 2 line/);
  const partial = dailyRow({ ...NOV5, missing: ["spy bars: no 2026-11-05 bar yet (last 2026-11-04)", "orders: timeout"], floor: { ...NOV5.floor, exposure_x: null, reason: "positions unreadable" } })!;
  assert.deepEqual(partial.partial, ["spy bars", "orders"]);
  assert.equal(partial.floor, "—");
  const notes = rowNotes(partial);
  assert.ok(notes.includes("Thu Nov 5: floor — positions unreadable"));
  assert.ok(notes.includes("Thu Nov 5: written partial — spy bars, orders unreadable"));
  assert.ok(notes.some((n) => /options total n\/a — no previous line/.test(n)));
  assert.deepEqual(rowNotes(dailyRow(NOV6)!), [], "a complete line has nothing to explain");
  for (const bad of [null, "x", [], { day: "Nov 5" }, { day: 5 }]) assert.equal(dailyRow(bad), null);
});

test("states: loading, not on this server (404), unavailable, a shape it cannot read, none yet, an unreadable file — each a designed sentence", () => {
  assert.equal(dailyLineView(undefined, { loading: true, error: null }, 0).state, "loading");
  assert.equal(dailyLineView(undefined, { loading: false, error: null }, 0).state, "loading", "not fetched yet (a query not yet enabled) is the skeleton too");
  const missing = dailyLineView(undefined, { loading: false, error: new Error("404: Not Found") }, 0);
  assert.equal(missing.state, "error");
  assert.equal(missing.note, "the daily line is not on this server yet");
  const down = dailyLineView(undefined, { loading: false, error: new Error("500: boom") }, 0);
  assert.equal(down.note, "the daily line unavailable · 500: boom");
  assert.equal(dailyLineView({ lines: [] }, Q, 0).note, "the daily line answered in a shape this page does not read");
  const none = dailyLineView(body([]), Q, Date.parse("2026-11-06T15:00:00Z"));
  assert.equal(none.state, "empty");
  assert.equal(none.note, "No daily line yet: the job writes one after each session's close (+20 min)");
  assert.deepEqual(none.today, { text: "Today's line is written after the close, from 16:20 ET", tone: "ok" });
  const unreadable = dailyLineView(body([], { error: "the daily line file is unreadable (Unexpected token n in JSON at position 1)" }), Q, 0);
  assert.equal(unreadable.state, "error");
  assert.equal(unreadable.note, "the daily line file is unreadable (Unexpected token n in JSON at position 1)");
});

test("a failed refresh keeps the lines fetched before it on screen and says so", () => {
  const v = dailyLineView(body([NOV5]), { loading: false, error: new Error("503: upstream") }, 0);
  assert.equal(v.state, "ready");
  assert.equal(v.rows.length, 1);
  assert.equal(v.note, "the last refresh failed (503: upstream) — these are the lines fetched before it");
  assert.equal(dailyLineView(body([NOV5]), Q, 0).note, null);
});

test("ready: newest first, at most ten, the stored count, the newest line's basis", () => {
  const days = Array.from({ length: 12 }, (_, i) => ({ ...NOV5, day: `2026-10-${String(10 + i).padStart(2, "0")}` }));
  const v = dailyLineView(body([...days].reverse().slice(0, 10), { count: 12 }), Q, Date.parse("2026-11-06T15:00:00Z"));
  assert.equal(v.state, "ready");
  assert.equal(v.rows.length, 10);
  assert.equal(v.rows[0].day, "2026-10-21");
  assert.equal(v.newest?.day, "2026-10-21");
  assert.equal(v.stored, 12);
  assert.equal(v.basis, NOV5.account_vs_spy.basis);
  // the route's order is not trusted: rows are sorted newest first
  const shuffled = dailyLineView(body([NOV5, NOV6]), Q, 0);
  assert.deepEqual(shuffled.rows.map((r) => r.day), ["2026-11-06", "2026-11-05"]);
  noLeak(v.rows);
});

test("today's line: written, due later, waiting for inputs, failed, overdue, no session", () => {
  const sch = { day: "2026-11-06", session: true, due_at: "2026-11-06T21:20:00.000Z" };
  const nov5 = dailyRow(NOV5);
  const at = (iso: string) => Date.parse(iso);
  assert.deepEqual(todayLine({ schedule: sch }, dailyRow(NOV6), at("2026-11-06T22:00:00Z")), { text: "Today's line was written at 16:22 ET", tone: "ok" });
  assert.deepEqual(todayLine({ schedule: sch }, nov5, at("2026-11-06T15:00:00Z")), { text: "Today's line is written after the close, from 16:20 ET", tone: "ok" });
  assert.deepEqual(todayLine({ schedule: sch, lastAttempt: { at: "2026-11-06T21:25:00Z", day: "2026-11-06", status: "waiting", detail: "spy bars: no 2026-11-06 bar yet (last 2026-11-05)" } }, nov5, at("2026-11-06T21:30:00Z")),
    { text: "Today's line is waiting for its inputs (spy bars: no 2026-11-06 bar yet (last 2026-11-05)); it is written partial if one stays unreadable", tone: "wait" });
  assert.deepEqual(todayLine({ schedule: sch, lastAttempt: { at: "2026-11-06T21:25:00Z", day: "2026-11-06", status: "error", detail: "ENOSPC" } }, nov5, at("2026-11-06T21:30:00Z")),
    { text: "Today's line failed: ENOSPC; the job retries", tone: "warn" });
  assert.deepEqual(todayLine({ schedule: sch, lastAttempt: { at: "2026-11-05T21:25:00Z", day: "2026-11-05", status: "written", detail: null } }, nov5, at("2026-11-06T23:00:00Z")),
    { text: "Today's line was due at 16:20 ET and is not written yet", tone: "warn" }, "yesterday's attempt says nothing about today");
  assert.deepEqual(todayLine({ schedule: { day: "2026-11-07", session: false, due_at: null } }, dailyRow(NOV6), at("2026-11-07T15:00:00Z")),
    { text: "No session today: the next line follows the next session's close", tone: "ok" });
  assert.equal(todayLine({}, nov5, 0), null, "no schedule from an older server: nothing said");
  assert.equal(todayLine({ schedule: { ...sch, due_at: "2026-11-27T18:20:00.000Z", day: "2026-11-27" } }, nov5, at("2026-11-27T15:00:00Z"))?.text, "Today's line is written after the close, from 13:20 ET", "a half day");
});

// ── the account day's basis ([RULE-REVIEW] 2026-11-06): at the close; the fallback and the older lines marked "*" ─────

test("basis: a line on the close reading reads plainly; a line from before it (no source) is the equity when written, marked *", () => {
  const old = dailyRow(NOV5)!;
  assert.deepEqual([old.acctBasis, old.acctAt, old.closeReason, old.check], ["write_time", NOV5.written_at, null, null], "the live 11-05 line predates the close reading");
  assert.equal(acctWords(old), `${MINUS}4.07%*`);
  assert.match(rowTitle(old), /^Thu Nov 5: account −4\.07%\* \(equity when the line was written, 20:12 ET: after-hours marks included\) vs SPY −4\.34% \(\+0\.27 pp\) · /);
  assert.deepEqual(rowNotes(old).filter((n) => /account =/.test(n)), [], "no close reason to spell out: the card's footnote says what * is");
  const close = dailyRow(NOV6)!;
  assert.deepEqual([close.acctBasis, close.acctAt, close.closeReason], ["close", "2026-11-06T20:59:52.000Z", null]);
  assert.equal(acctWords(close), "+0.42%");
  assert.equal(dailyRow({ ...NOV5, account_vs_spy: { ...NOV5.account_vs_spy, day: { ...NOV5.account_vs_spy.day, account_pct: null, reason: "account read invalid" } } })!.acctBasis, null, "no account day, no basis");
  noLeak(old); noLeak(close);
});

test("basis: a line that fell back says why on the card; the next line's close check is in the hover", () => {
  const fell = dailyRow({ ...NOV6, day: "2026-11-10", written_at: "2026-11-10T21:22:10.000Z",
    account_vs_spy: { ...NOV6.account_vs_spy, day: { ...NOV6.account_vs_spy.day, account_source: "write_time", equity_end_at: "2026-11-10T21:22:10.000Z", close_reason: "no close reading for 2026-11-10" } } })!;
  assert.equal(acctWords(fell), "+0.42%*");
  assert.deepEqual(rowNotes(fell), ["Tue Nov 10: account = equity when the line was written (16:22 ET) — no close reading for 2026-11-10"]);
  assert.match(rowTitle(fell), /^Tue Nov 10: account \+0\.42%\* \(equity when the line was written, 16:22 ET: after-hours marks included; no close reading for 2026-11-10\) vs SPY/);
  const nov9 = dailyRow({ ...NOV6, day: "2026-11-09", written_at: "2026-11-09T21:22:10.000Z",
    close_check: { day: "2026-11-06", line_pct: 0.42, line_equity: 90100, line_source: "close", alpaca_close: 90082.15, alpaca_pct: 0.4, error_pp: 0.02, error_usd: 17.85, balance_asof: "2026-11-06", reason: null } })!;
  assert.deepEqual(nov9.check, { day: "2026-11-06", label: "Fri Nov 6", linePct: 0.42, alpacaPct: 0.4, errorPp: 0.02 });
  assert.equal(checkWords(nov9.check!), "the Fri Nov 6 line read +0.42%, Alpaca's close +0.40% (+0.02 pp)");
  assert.match(rowTitle(nov9), / · close check: the Fri Nov 6 line read \+0\.42%, Alpaca's close \+0\.40% \(\+0\.02 pp\)$/);
  assert.deepEqual(rowNotes(nov9), [], "a check is not a missing piece");
  const unchecked = dailyRow({ ...NOV6, close_check: { day: "2026-11-05", line_pct: -4.07, alpaca_close: null, alpaca_pct: null, error_pp: null, reason: "last_equity unreadable" } })!;
  assert.equal(unchecked.check, null, "a check that could not be made is not shown as one");
  noLeak(fell); noLeak(nov9);
});

test("basis: the card's sentence names the close, and the * clause only when a shown row carries it", () => {
  assert.equal(acctBasisWords(false), "Account = equity at the close (the last read before the bell) vs the prior close (Alpaca last_equity)");
  assert.equal(acctBasisWords(true), "Account = equity at the close (the last read before the bell) vs the prior close (Alpaca last_equity) · * = equity when the line was written (no close reading; after-hours marks can move it)");
  assert.equal(dailyLineView(body([NOV6, NOV5]), Q, 0).writeTime, true, "the 11-05 line is shown");
  assert.equal(dailyLineView(body([NOV6]), Q, 0).writeTime, false);
  assert.equal(dailyLineView(body([]), Q, 0).writeTime, false);
  assert.equal(dailyLineView(undefined, { loading: true, error: null }, 0).writeTime, false);
});
