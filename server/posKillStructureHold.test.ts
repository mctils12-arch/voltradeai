// POS-KILL STRUCTURE HOLD (2026-11-02, [RULE-REVIEW]) — the per-position kill
// in syncMonitoredPositions() (server/bot.ts) judged every option leg by its
// own premium P&L. Live 10-28/29 it read the two LONG WINGS of the GOOGL
// Nov-27 iron condor as "down −99.8 %" (122 pre-market retries), bought the
// short 310P back on a "−309 %" reading, and submitted the 300P wing for a
// sell on a zero mark — on a DEFINED-RISK structure whose worst case its own
// wings bound, while options_manager's group rules fired on the same legs.
//
// The rule: a leg whose recorded strategy (voltrade_options_state.json) is in
// OPTIONS_DEFINED_RISK_STRATEGIES is never killed on its own — one
// POS-KILL-STRUCTURE-HOLD line per ticker per UTC day, no POST. Single legs
// and undefined-risk groups (short straddles) keep the backstop unchanged.
//
// Same harness as posKillMarkGuard.test.ts: the REAL per-position loop body
// is cut out of bot.ts, transpiled and run tick by tick against stubs. The
// structure map is what syncMonitoredPositions parks on `state` before the
// loop (`state._posKillStructures`), so the harness sets it directly.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import { OPTIONS_DEFINED_RISK_STRATEGIES, OPTIONS_MULTI_LEG_STRATEGIES } from "../shared/positionLevels";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function matchBrace(src: string, open: number): number {
  assert.equal(src[open], "{");
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === '"' || c === "'" || c === "`") {
      for (i++; i < src.length && src[i] !== c; i++) if (src[i] === "\\") i++;
      continue;
    }
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return i + 1;
  }
  throw new Error("unbalanced braces");
}
function functionSourceIfPresent(signaturePrefix: string): string {
  const start = bot.indexOf(signaturePrefix);
  if (start < 0) return "";
  return bot.slice(start, matchBrace(bot, bot.indexOf("{", start + signaturePrefix.length)));
}
const transpile = (ts: string): string => transformSync(ts, { loader: "ts" }).code;

const syncStart = bot.indexOf("async function syncMonitoredPositions()");
assert.ok(syncStart > 0);
const loopStart = bot.indexOf("for (const pos of positions) {", syncStart);
const loopEnd = bot.indexOf("// Read stop state for this ticker", loopStart);
assert.ok(loopStart > syncStart && loopEnd > loopStart, "the per-position loop was not found");
const loopBody = bot.slice(loopStart, loopEnd) + "\n}";
const HELPERS = [
  "function isOptionSymbol(ticker: string): boolean",
  "function isPosKillExemptOption(symbol: string, assetClass?: string): boolean",
  'function posKillThresholdPct(isOption: boolean, side: "long" | "short", stockPct: number,',
  "function posKillHasValidMark(currentPriceRaw: unknown, plpcRaw: unknown): boolean",
  "function posKillObserveBreach(pending: Map<string, PosKillBreach>, symbol: string,",
].map(functionSourceIfPresent).join("\n");

type Row = { symbol: string; asset_class?: string; side: string; qty: string; avg_entry_price?: string; current_price?: string; unrealized_plpc?: string; created_at?: string };
function makeHarness(structures: Map<string, string>, startIso = "2026-10-29T13:30:09Z") {
  let nowMs = Date.parse(startIso);
  class FakeDate extends Date { constructor(v?: number | string | Date) { if (v === undefined) super(nowMs); else super(v); } static now(): number { return nowMs; } }
  const audits: Array<{ type: string; msg: string }> = [];
  const posts: Array<{ path: string; body: Record<string, unknown> }> = [];
  const openOrders: Array<Record<string, unknown>> = [];
  const env = {
    FLOOR_AND_LEG_TICKERS: new Set(["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"]),
    LEGACY_DEFENSIVE_CANDIDATE_TICKERS: new Set(["VTI", "IWM", "TLT", "IEF", "SCHP"]),
    audit: (type: string, msg: string) => { audits.push({ type, msg }); },
    alpaca: async (p: string, init?: { method?: string; body?: string }) => {
      if (init?.method === "POST") { posts.push({ path: p, body: JSON.parse(init.body ?? "{}") }); return { id: `ord-${posts.length}` }; }
      return [];
    },
    state: {} as Record<string, unknown>,
    structures,
    openOrders,
    getOrderParams: () => ({ type: "market", time_in_force: "day" }),
    notify: () => undefined,
    pkgVersion: "test",
    posKillPendingBreach: new Map<string, { firstSeenMs: number; pnlPct: number }>(),
    posKillNoMarkAuditDay: new Map<string, string>(),
  };
  const js = transpile(`
${HELPERS}
let posKillStructuresCurrent = new Map();
const posKillStructureHoldDay = new Map();
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders,
          getOrderParams, notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay } = env;
  posKillStructuresCurrent = env.structures;
  const activeTickers = new Set();
  ${loopBody}
}
`);
  const runTick = new Function("Date", `${js}\nreturn runTick;`)(FakeDate) as (positions: Row[], e: typeof env) => Promise<void>;
  const tick = async (positions: Row[]) => { await runTick(positions, env); nowMs += 45_000; };
  const audited = (type: string) => audits.filter((a) => a.type === type);
  return { tick, audits, posts, audited, advanceDay: () => { nowMs += 86_400_000; } };
}
function optionAt(symbol: string, side: "long" | "short", entry: number, pnlPct: number): Row {
  const mark = Math.max(0.01, entry * (1 + (side === "long" ? pnlPct : -pnlPct) / 100));
  return { symbol, asset_class: "us_option", side, qty: side === "long" ? "2" : "-2", avg_entry_price: String(entry), current_price: mark.toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: "2026-10-20T15:12:00Z" };
}
const CONDOR = new Map<string, string>([
  ["GOOGL261127P00310000", "iron_condor"], ["GOOGL261127P00300000", "iron_condor"],
  ["GOOGL261127C00345000", "iron_condor"], ["GOOGL261127C00355000", "iron_condor"],
]);

test("the defined-risk set is a subset of the multi-leg set and excludes the undefined-risk short straddle", () => {
  for (const s of OPTIONS_DEFINED_RISK_STRATEGIES) assert.ok(OPTIONS_MULTI_LEG_STRATEGIES.includes(s), `${s} must be a multi-leg strategy`);
  assert.ok(!OPTIONS_DEFINED_RISK_STRATEGIES.includes("short_straddle"));
  for (const s of ["iron_condor", "qqq_iron_condor", "bull_put_credit_spread", "bull_call_spread", "bear_put_spread", "buy_straddle"]) assert.ok(OPTIONS_DEFINED_RISK_STRATEGIES.includes(s), s);
});

test("posKillDefinedRiskLegs: only legs whose recorded strategy is defined-risk; unknown, single and short-straddle legs are not structures", () => {
  const src = functionSourceIfPresent("function posKillDefinedRiskLegs(raw: unknown, definedRisk: readonly string[] = OPTIONS_DEFINED_RISK_STRATEGIES): Map<string, string>");
  assert.ok(src, "posKillDefinedRiskLegs must exist in bot.ts");
  const fn = new Function(`const OPTIONS_DEFINED_RISK_STRATEGIES = ${JSON.stringify(OPTIONS_DEFINED_RISK_STRATEGIES)};\n${transpile(src)}\nreturn posKillDefinedRiskLegs;`)() as (raw: unknown) => Map<string, string>;
  const legs = fn({
    GOOGL261127P00310000: { strategy: "iron_condor" }, GOOGL261127P00300000: { strategy: "iron_condor", close_requested_at: "x" },
    CRWV261120P00055000: { strategy: "sell_cash_secured_put" }, TEM261120P00070000: { strategy: "unknown" }, XLK261219P00100000: {},
    KMI261219C00030000: { strategy: "short_straddle" }, GOOGL: { strategy: "iron_condor", options_strategy: "iron_condor" }, junk: "not an object",
  });
  assert.deepEqual([...legs.entries()].sort(), [["GOOGL", "iron_condor"], ["GOOGL261127P00300000", "iron_condor"], ["GOOGL261127P00310000", "iron_condor"]]);
  assert.equal(fn(null).size, 0); assert.equal(fn("x").size, 0); assert.equal(fn([]).size, 0);
});

test("wiring: the structure legs are read once per sync before the loop; the hold is the first exit inside the UNCHANGED kill-branch opener, before the mark guard", () => {
  const syncAt = bot.indexOf("async function syncMonitoredPositions()");
  const sync = bot.slice(syncAt, bot.indexOf("for (const pos of positions) {", syncAt));
  assert.ok(sync.includes("posKillStructuresCurrent = posKillOptionsStructures();"), "the map is read once per sync, before the loop");
  assert.equal(bot.split("if (pnlPct <= POSITION_KILL_LOSS_PCT) {").length - 1, 1, "the kill-branch opener is untouched (sibling tests slice on it)");
  const kill = bot.slice(bot.indexOf("if (pnlPct <= POSITION_KILL_LOSS_PCT) {"));
  const holdAt = kill.indexOf("if (structureStrategy) {");
  const nomarkAt = kill.indexOf('if (breach === "nomark") {');
  assert.ok(holdAt > 0 && holdAt < nomarkAt, "the structure hold leaves the branch before the mark guard");
  assert.ok(/continue;/.test(kill.slice(holdAt, nomarkAt)), "the hold leaves the branch, never falls through to the POST");
  assert.ok(bot.includes('audit("POS-KILL-STRUCTURE-HOLD"'), "the hold is audited");
  assert.ok(bot.includes('typeof posKillStructuresCurrent !== "undefined"'), "the loop body stays runnable in harnesses that do not define the map");
  assert.ok(bot.includes("OPTIONS_DEFINED_RISK_STRATEGIES } from \"../shared/positionLevels\""), "one source of truth for the set");
  assert.ok(bot.includes("if (isOptionPos && pnlPct <= STOCK_KILL_LOSS_PCT && pnlPct > POSITION_KILL_LOSS_PCT) {"), "the option-skip branch is untouched");
});

test("(a) a condor's long wing at −99 % on a real mark is never killed: no POST, one POS-KILL-STRUCTURE-HOLD line per day", async () => {
  const h = makeHarness(CONDOR);
  const wing = optionAt("GOOGL261127C00355000", "long", 5.15, -98.8);
  for (let i = 0; i < 4; i++) await h.tick([wing]);
  assert.equal(h.posts.length, 0, "a defined-risk structure's wing must never reach the liquidation POST");
  assert.equal(h.audited("POS-KILL").length, 0);
  assert.equal(h.audited("POS-KILL-PENDING").length, 0, "no breach bookkeeping for a structure leg");
  const hold = h.audited("POS-KILL-STRUCTURE-HOLD");
  assert.equal(hold.length, 1, "once per ticker per UTC day");
  assert.ok(hold[0].msg.startsWith("GOOGL261127C00355000: iron_condor leg down -98.8% vs its premium"), hold[0].msg);
  h.advanceDay();
  await h.tick([wing]);
  assert.equal(h.audited("POS-KILL-STRUCTURE-HOLD").length, 2, "a new UTC day logs once more");
  assert.equal(h.posts.length, 0);
});

test("(b) the condor's short leg at −309 % (past the −150 % backstop) is the group's loss: no POST either", async () => {
  const h = makeHarness(CONDOR);
  const shortPut = optionAt("GOOGL261127P00310000", "short", 6.95, -309.4);
  await h.tick([shortPut]); await h.tick([shortPut]); await h.tick([shortPut]);
  assert.equal(h.posts.length, 0);
  assert.equal(h.audited("POS-KILL-STRUCTURE-HOLD").length, 1);
});

test("(c) a standalone sold put at −160 % and a short-straddle leg at −160 % still hit the backstop (two valid-mark ticks, then the POST)", async () => {
  const h = makeHarness(CONDOR);
  const naked = optionAt("XLF261127P00055500", "short", 0.72, -160);
  await h.tick([naked]);
  assert.equal(h.audited("POS-KILL-PENDING").length, 1, "first valid-mark sighting confirms on the next tick");
  await h.tick([naked]);
  assert.equal(h.posts.length, 1, "the standalone short put is still killed");
  assert.equal(h.posts[0].body.symbol, "XLF261127P00055500");
  assert.equal(h.audited("POS-KILL-STRUCTURE-HOLD").length, 0);
  // a short straddle is undefined-risk: posKillDefinedRiskLegs never admits it (pinned above), so the
  // per-sync map a straddle-holding account produces is exactly what the harness passes here — empty
  const h2 = makeHarness(new Map());
  const straddleLeg = optionAt("KMI261219C00030000", "short", 1.0, -160);
  await h2.tick([straddleLeg]); await h2.tick([straddleLeg]);
  assert.equal(h2.posts.length, 1, "an undefined-risk group's leg keeps the per-leg backstop");
});

test("(d) a leg between the stock rule and its backstop keeps the informational POS-KILL-OPTION-HOLD line (unchanged); the structure hold logs only where a kill would have fired", async () => {
  const h = makeHarness(CONDOR);
  await h.tick([optionAt("XLK261219P00100000", "long", 4.0, -40)]);
  assert.equal(h.audited("POS-KILL-OPTION-HOLD").length, 1);
  await h.tick([optionAt("GOOGL261127P00300000", "long", 4.7, -40)]);
  assert.equal(h.audited("POS-KILL-STRUCTURE-HOLD").length, 0, "the hold line is logged only where a kill would have fired");
  assert.equal(h.posts.length, 0);
});
