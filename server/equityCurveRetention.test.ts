// EQUITY CURVE RETENTION (2026-11-06, [REPAIR]) — server/bot.ts recordDailyEquity.
//
// The curve is newest-first (index 0 = today). The old trim did
// `unshift(today); if (length > 365) shift()`, i.e. it removed the row it had
// just added: from the 366th day on, no new day could ever be recorded, which
// would freeze the SPY benchmark, the scoreboard and the daily line. Now the
// trim drops the OLDEST rows (the end of the array) and the cap is
// EQUITY_CURVE_MAX_ROWS (ten years), at load and at record time.
// The function under test is the REAL one, cut out of bot.ts and run against stubs.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

/** Index just past the `}` closing the brace at `open` (strings and comments skipped:
 * recordDailyEquity's comments contain apostrophes). */
function matchBrace(src: string, open: number): number {
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === "/" && src[i + 1] === "/") { i = src.indexOf("\n", i); continue; }
    if (c === "/" && src[i + 1] === "*") { i = src.indexOf("*/", i) + 1; continue; }
    if (c === '"' || c === "'" || c === "`") {
      for (i++; i < src.length && src[i] !== c; i++) if (src[i] === "\\") i++;
      continue;
    }
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return i + 1;
  }
  throw new Error("unbalanced braces");
}

function functionSource(signature: string): string {
  const start = bot.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in bot.ts`);
  const open = bot.indexOf("{", start + signature.length - 1);
  return bot.slice(start, matchBrace(bot, open));
}

const MAX = Number(/const EQUITY_CURVE_MAX_ROWS = (\d+);/.exec(bot)?.[1]);

type Row = { date: string; value: number; pnl: number };

function makeRecorder(curve: Row[], todayIso: string) {
  const saves: number[] = [];
  class FakeDate extends Date {
    constructor(v?: number | string | Date) { if (v === undefined) super(todayIso); else super(v); }
    static now(): number { return Date.parse(todayIso); }
  }
  const alpaca = async () => ({ portfolio_value: "90000.5", equity: "90000.5", last_equity: "90500.5" });
  const js = transformSync(functionSource("function recordDailyEquity()"), { loader: "ts" }).code;
  const record = new Function("alpaca", "equityCurve", "saveEquityCurve", "Date", "EQUITY_CURVE_MAX_ROWS",
    `${js}\nreturn recordDailyEquity;`,
  )(alpaca, curve, () => saves.push(curve.length), FakeDate, MAX) as () => void;
  return { record: async () => { record(); await new Promise((r) => setImmediate(r)); }, saves };
}

/** n newest-first daily rows ending the day before `lastIso`. */
function history(n: number, lastIso: string): Row[] {
  const end = Date.parse(lastIso);
  return Array.from({ length: n }, (_, i) => ({
    date: new Date(end - (i + 1) * 86_400_000).toISOString().slice(0, 10), value: 100_000 + i, pnl: 0,
  }));
}

test("the cap is ten years of daily rows", () => {
  assert.equal(MAX, 3660);
});

test("a full curve still records today: the OLDEST row is dropped, never the new one", async () => {
  const curve = history(MAX, "2036-11-06T21:00:00Z");
  const oldestBefore = curve[curve.length - 1].date;
  const secondOldest = curve[curve.length - 2].date;
  const { record } = makeRecorder(curve, "2036-11-06T21:00:00Z");
  await record();
  assert.equal(curve.length, MAX);
  assert.deepEqual(curve[0], { date: "2036-11-06", value: 90000.5, pnl: -500 }, "today is the newest row");
  assert.equal(curve[1].date, "2036-11-05", "yesterday is kept");
  assert.equal(curve[curve.length - 1].date, secondOldest, "the oldest row went");
  assert.notEqual(curve[curve.length - 1].date, oldestBefore);
});

test("the old bug's exact state (365 rows): the 366th day is recorded", async () => {
  const curve = history(365, "2027-05-06T21:00:00Z");
  const { record } = makeRecorder(curve, "2027-05-06T21:00:00Z");
  await record();
  assert.equal(curve.length, 366, "under the new cap nothing is dropped");
  assert.equal(curve[0].date, "2027-05-06");
});

test("a second call on the same day updates today's row in place", async () => {
  const curve = history(10, "2026-11-06T15:00:00Z");
  const { record, saves } = makeRecorder(curve, "2026-11-06T15:00:00Z");
  await record();
  curve[0].value = 1;
  await record();
  assert.equal(curve.length, 11);
  assert.equal(curve[0].date, "2026-11-06");
  assert.equal(curve[0].value, 90000.5);
  assert.deepEqual(saves, [11, 11], "persisted after every update");
});

test("load keeps the newest EQUITY_CURVE_MAX_ROWS rows, not 365", () => {
  const load = bot.slice(bot.indexOf("const equityCurve: Array<{ date: string; value: number; pnl: number }> = (() => {"));
  const body = load.slice(0, load.indexOf("})();"));
  assert.ok(body.includes(").slice(0, EQUITY_CURVE_MAX_ROWS);"), "the load slice uses the cap");
  assert.ok(!/\.slice\(0, 365\)/.test(body));
  assert.ok(bot.indexOf("const EQUITY_CURVE_MAX_ROWS = ") < bot.indexOf("const equityCurve: Array<"), "declared before the load runs");
});
