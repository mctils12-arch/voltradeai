// floorTrackingWiring.test.ts — how server/bot.ts serves the floor tracking
// report ([RULE-REVIEW] measurement, 2026-11-12): a whitelisted, sanitized diag
// probe and one audit row per daily line, both from one body that only reads.
// Source pins, like diag.test.ts: the report's arithmetic is pinned in
// floorTracking.test.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { DIAG_PROBES } from "./diag";
import { isTrivialPython, planPythonExec } from "./pythonExec";
import type { FloorTrackingReport } from "./floorTracking";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(start: string, end: string): string {
  const a = bot.indexOf(start);
  assert.ok(a >= 0, `anchor not found: ${start}`);
  const b = bot.indexOf(end, a + start.length);
  assert.ok(b > a, `end anchor not found after ${start}: ${end}`);
  return bot.slice(a, b);
}

test("floor_tracking is a whitelisted diag probe: one sanitized body, no parameters, and the case itself reads nothing", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("floor_tracking"), "floor_tracking must be whitelisted in diag.ts");
  const probe = slice('case "floor_tracking": {', 'case "account": {');
  assert.ok(probe.includes('return res.json(sanitizeDiag({ probe: "floor_tracking", ...(await floorTrackingBody()) }));'), "the probe serves floorTrackingBody() through the sanitizer");
  assert.ok(!/alpaca\(|fetch\(|execPython|pythonRpc|req\.query/.test(probe), "every read lives in floorTrackingBody; the probe takes no query parameter");
  assert.ok(bot.includes("} from \"./floorTracking\";"), "bot.ts imports the pure module");
});

test("floorTrackingBody only reads: config, the stored lines, positions, the floor tickers' orders, daily closes", () => {
  const body = slice("async function floorTrackingBody(): Promise<FloorTrackingReport> {", "async function auditFloorTracking(): Promise<void> {");
  for (const read of [
    "await execPythonSerialized(FLOOR_TRACKING_CFG_CMD, { timeout: 10000 })", "readDailyLines(dailyLineDir(process.env))",
    'alpaca("/v2/positions")', "floorTrackingOrders(tickers, new Date(fromMs).toISOString())", "await spyBenchBars(", "await fetchDailyBarsRange(ticker, fromDay,",
    "parsePositionsLite(positions.value)", "return floorTracking(inputs);",
  ]) assert.ok(body.includes(read), `floorTrackingBody must contain: ${read}`);
  assert.ok(!/method:|saveDailyLine|writeFileSync|appendFileSync|audit\(|persistAudit/.test(body), "no order, no file write, no audit row from the body");
  assert.ok(!body.includes('fetchDailyBarsRange("SPY"'), "SPY comes from the benchmark's one shared series, never a second fetch");
  assert.equal((body.match(/alpaca\(/g) ?? []).length, 1, "one direct broker read: the positions");
  // the broker is not read at all when the floor is not the volatility-managed one or no line is stored
  const guard = body.indexOf("if (!cfg || cfg.mode !== FLOOR_TRACKING_MODE || stored.error || !stored.lines.length) return floorTracking(inputs);");
  assert.ok(guard > 0 && guard < body.indexOf('alpaca("/v2/positions")') && guard < body.indexOf("floorTrackingOrders(tickers"), "the early return precedes every broker read");
  // a failed read becomes that input's error, never a thrown probe
  for (const name of ["positions", "orders", "base_bars", "levered_bars", "cash_bars", "config", "lines"]) {
    assert.ok(body.includes(`errors.${name} = `), `a failed ${name} read is recorded in errors.${name}`);
  }
  assert.ok(body.includes("await Promise.allSettled(["), "the reads are taken together and each may fail alone");
  // the orders reader: a GET, filtered to the floor's tickers, every status, paged
  const orders = slice("async function floorTrackingOrders(symbols: readonly string[], afterIso: string): Promise<unknown[]> {", "async function floorTrackingBody(");
  assert.ok(orders.includes("await alpaca(`/v2/orders?status=all&symbols=${encodeURIComponent(symbols.join(\",\"))}&after=${encodeURIComponent(after)}&limit=500&direction=asc`)"));
  assert.ok(!/method:/.test(orders), "a read: no method, so never a POST, PATCH or DELETE");
  assert.ok(orders.includes('throw new Error("more than 5000 floor orders in the window")') && orders.includes('throw new Error("orders paging stalled")'), "a truncated read fails loudly instead of rebuilding positions from part of the orders");
});

test("the config read is one short Python command in the no-mutex lane, and it returns the keys the report needs", () => {
  const m = /const FLOOR_TRACKING_CFG_CMD = `(python3 -c "([^`]+)")`;/.exec(bot);
  assert.ok(m, "FLOOR_TRACKING_CFG_CMD not found in bot.ts");
  assert.ok(m[1].length <= 150, `the command is ${m[1].length} chars: over 150 it would wait on the Python mutex`);
  assert.equal(isTrivialPython(m[1], 10000), true);
  assert.deepEqual(planPythonExec(m[1], 10000), { lane: "trivial", timeoutMs: 10000, mutexWaitMs: 0 });
  assert.ok(!/import\s+bot_engine|from\s+bot_engine/.test(m[2]), "system_config only: no engine import for six config values");
  const out = execFileSync("python3", ["-c", m[2]], { cwd: repo, encoding: "utf8" }).trim().split("\n").pop() ?? "";
  const cfg = JSON.parse(out) as Record<string, unknown>;
  for (const key of ["FLOOR_MODE", "FLOOR_VM_TICKER", "FLOOR_VM_LEVERED_TICKER", "FLOOR_VM_CASH_TICKER", "FLOOR_VM_CAPITAL_PCT", "FLOOR_VM_BAND"]) {
    assert.ok(key in cfg, `system_config has no ${key}`);
  }
  assert.equal(typeof cfg.FLOOR_VM_CAPITAL_PCT, "number");
});

test("once per daily line: the FLOOR-TRACKING row follows the line's own row, from the same body, and never blocks or fails the line", () => {
  const check = slice("async function checkDailyLine(): Promise<void> {", "// ── TIER 0: Real-Time Streaming");
  const hook = check.indexOf("void auditFloorTracking();");
  assert.ok(hook > 0, "checkDailyLine does not call auditFloorTracking");
  assert.ok(hook > check.indexOf('audit("DAILY-LINE", line.message)') && hook > check.indexOf("saveDailyLine(dir, line)") && hook > check.indexOf('note("written"'),
    "after the line is saved, logged and noted: the line never waits on the tracking report");
  assert.ok(!check.includes("await auditFloorTracking"), "not awaited: a slow broker read cannot hold the daily line job");
  assert.equal(bot.split("auditFloorTracking(").length - 1, 2, "the declaration and the one call");
  const fn = slice("async function auditFloorTracking(): Promise<void> {", "// ── Token-gated READ-ONLY diagnostics");
  assert.ok(fn.includes("const report = await floorTrackingBody();"), "the same body the probe serves");
  assert.ok(fn.includes('if (report.status !== "not_applicable") audit("FLOOR-TRACKING", report.message);'), "no row while the floor is not the volatility-managed one");
  assert.ok(/catch \(e: unknown\) \{\s*audit\("FLOOR-TRACKING", `floor tracking unavailable: /.test(fn), "a failure is logged, never thrown into the daily line job");
  assert.ok(!/alpaca\(|method:|saveDailyLine/.test(fn));
});

test("the pure module names no order endpoint (the order-site ratchet counts files) and bot.ts keeps its one SPY series", () => {
  const mod = fs.readFileSync(path.join(here, "floorTracking.ts"), "utf8");
  assert.ok(!mod.includes("/v2/orders"), "floorTracking.ts must not reference the orders endpoint: bot.ts makes the read");
  assert.equal(bot.split('fetchDailyBarsRange("SPY"').length - 1, 1, "one SPY fetch in bot.ts, as before");
});

// ── the REAL gatherer, cut out of bot.ts and run against stubs ───────────────
// (the orderGateWiring.test.ts method). The pure module is imported inside each
// test, so the source pins above stand on their own.
import { transformSync } from "esbuild";

interface Stubs {
  config?: unknown; configThrows?: string;
  lines?: unknown[]; linesError?: string | null; linesThrow?: string;
  positions?: unknown; positionsThrow?: string;
  orderPages?: unknown[];
  bars?: Record<string, Array<{ date: string; close: number }> | Error>;
  curve?: Array<{ date: string }>;
}
interface Wired {
  floorTrackingBody: () => Promise<FloorTrackingReport>;
  auditFloorTracking: () => Promise<void>;
  calls: { alpaca: Array<{ path: string; opts: unknown }>; python: Array<{ cmd: string; timeout: unknown }>; spy: string[]; bars: string[][]; audit: string[][] };
}

// the stored records, as floorTracking.test.ts carries them: the bars' own closes (the reconstruct_pnl probe, 2026-11-12)
const SPY_BARS = [["2026-11-04", 766.73], ["2026-11-05", 733.43], ["2026-11-06", 735.4], ["2026-11-09", 752.72], ["2026-11-10", 749.93], ["2026-11-11", 759.8]]
  .map(([date, close]) => ({ date: String(date), close: Number(close) }));
const BOXX_BARS = [["2026-11-04", 118.87], ["2026-11-05", 118.9], ["2026-11-06", 118.95], ["2026-11-09", 118.97], ["2026-11-10", 118.99], ["2026-11-11", 118.99]]
  .map(([date, close]) => ({ date: String(date), close: Number(close) }));
const LINES = [["2026-11-05", null, 93_623.46, 0], ["2026-11-06", 0.6835, 89_888.9, 60_347.08], ["2026-11-09", 0.6888, 89_160.63, 61_721.4],
  ["2026-11-10", 0.6225, 90_582.83, 61_583.64], ["2026-11-11", 0.6225, 90_363.79, 62_330.66]]
  .map(([day, targetX, lastEquity, floorMv]) => ({ day, session_close: "16:00", account_vs_spy: { day: { last_equity: lastEquity } }, floor: { mode: "vol_managed_spy", target_x: targetX, floor_mv: floorMv } }));
const ORDERS = [
  { id: "4bfaf255-21f5-41be-bd81-a26f0b3e233f", symbol: "SPY", side: "buy", filled_qty: "82", filled_avg_price: "740.12", submitted_at: "2026-11-06T14:47:57.1Z", filled_at: "2026-11-06T14:47:58.976943Z" },
  { id: "fe4fcbcf-e5e3-428c-a4bb-07cacf5b1926", symbol: "BOXX", side: "buy", filled_qty: "110", filled_avg_price: "118.93", submitted_at: "2026-11-06T14:50:54.9Z", filled_at: "2026-11-06T14:51:00.342235Z" },
];
const POSITIONS = [{ symbol: "SPY", qty: "82", side: "long", market_value: "62366.74" }, { symbol: "BOXX", qty: "110", side: "long", market_value: "13090.55" }];
const CONFIG = { FLOOR_MODE: "vol_managed_spy", FLOOR_VM_TICKER: "SPY", FLOOR_VM_LEVERED_TICKER: "SSO", FLOOR_VM_CASH_TICKER: "BOXX", FLOOR_VM_CAPITAL_PCT: 0.85, FLOOR_VM_BAND: 0.1 };

async function wire(s: Stubs = {}): Promise<Wired> {
  const ft = await import("./floorTracking");
  const { parsePositionsLite } = await import("./dailyLine");
  const src = slice("const FLOOR_TRACKING_CFG_CMD = `", "// ── Token-gated READ-ONLY diagnostics");
  const js = transformSync(`function wire(env) {
  const { alpaca, execPythonSerialized, readDailyLines, dailyLineDir, process, marketHolidays, marketHalfDays, spyBenchBars, fetchDailyBarsRange,
          equityCurve, parsePositionsLite, floorTracking, parseFloorTrackingConfig, FLOOR_TRACKING_MODE, FLOOR_TRACKING_ORDERS_LOOKBACK_DAYS, audit } = env;
  ${src}
  return { floorTrackingBody, auditFloorTracking };
}`, { loader: "ts" }).code;
  const calls: Wired["calls"] = { alpaca: [], python: [], spy: [], bars: [], audit: [] };
  const pages = [...(s.orderPages ?? [ORDERS])];
  const barsFor = (ticker: string) => {
    const b = (s.bars ?? { SPY: SPY_BARS, SSO: [], BOXX: BOXX_BARS })[ticker];
    if (b instanceof Error) throw b;
    return b ?? [];
  };
  const env = {
    alpaca: async (p: string, opts?: unknown) => {
      calls.alpaca.push({ path: p, opts });
      if (p === "/v2/positions") { if (s.positionsThrow) throw new Error(s.positionsThrow); return s.positions === undefined ? POSITIONS : s.positions; }
      if (p.startsWith("/v2/orders?")) return pages.length ? pages.shift() : [];
      throw new Error(`unexpected broker path ${p}`);
    },
    execPythonSerialized: async (cmd: string, opts: { timeout?: number }) => {
      calls.python.push({ cmd, timeout: opts?.timeout });
      if (s.configThrows) throw new Error(s.configThrows);
      return { stdout: `a warning line\n${JSON.stringify(s.config === undefined ? CONFIG : s.config)}\n` };
    },
    readDailyLines: () => { if (s.linesThrow) throw new Error(s.linesThrow); return { lines: s.lines ?? LINES, error: s.linesError ?? null }; },
    dailyLineDir: () => "/nowhere", process: { env: {} },
    marketHolidays: () => new Set<string>(), marketHalfDays: () => new Set<string>(),
    spyBenchBars: async (oldest: string) => { calls.spy.push(oldest); return { at: Date.parse("2026-11-12T14:00:00Z"), from: oldest, bars: barsFor("SPY") }; },
    fetchDailyBarsRange: async (ticker: string, from: string, to: string) => { calls.bars.push([ticker, from, to.slice(0, 4)]); return barsFor(ticker); },
    equityCurve: s.curve ?? [{ date: "2026-11-12" }, { date: "2026-05-06" }],
    parsePositionsLite, floorTracking: ft.floorTracking, parseFloorTrackingConfig: ft.parseFloorTrackingConfig,
    FLOOR_TRACKING_MODE: ft.FLOOR_TRACKING_MODE, FLOOR_TRACKING_ORDERS_LOOKBACK_DAYS: ft.FLOOR_TRACKING_ORDERS_LOOKBACK_DAYS,
    audit: (type: string, message: string) => { calls.audit.push([type, message]); },
  };
  const fns = new Function(`${js}\nreturn wire;`)()(env) as Pick<Wired, "floorTrackingBody" | "auditFloorTracking">;
  return { ...fns, calls };
}

test("the real floorTrackingBody on the live records: one config read, positions, the floor tickers' orders since 10 days before the first line, three sets of closes", async () => {
  const w = await wire();
  const r = await w.floorTrackingBody();
  assert.equal(r.status, "ok");
  assert.equal(r.latest?.day, "2026-11-11");
  assert.equal(r.entry?.day, "2026-11-06");
  // 82 SPY bought at 740.12 and closed at 735.40, against 0.6835 x 89,888.90 of SPY from 733.43: on 0.85 x 89,888.90
  const shortfall = ((82 * (735.4 - 740.12) - 0.6835 * 89_888.9 * (735.4 / 733.43 - 1)) / (0.85 * 89_888.9)) * 100;
  assert.ok(r.entry?.shortfall_pp != null && Math.abs(r.entry.shortfall_pp - shortfall) < 1e-4 && Math.abs(shortfall + 0.7225) < 1e-3, `entry shortfall ${r.entry?.shortfall_pp}`);
  assert.equal(r.stored.sessions, 3);
  assert.equal(r.rollback_trigger.tracking.status, "not_enough_history");
  assert.deepEqual(r.inputs.errors, {});
  assert.deepEqual(r.config, { base: "SPY", levered: "SSO", cash_fund: "BOXX", capital_pct: 0.85, band: 0.1 });
  // the reads, exactly
  assert.equal(w.calls.python.length, 1);
  assert.equal(w.calls.python[0].timeout, 10000);
  assert.ok(w.calls.python[0].cmd.startsWith('python3 -c "import json,system_config as s;'));
  assert.deepEqual(w.calls.alpaca.map((c) => c.path).sort(), [
    "/v2/orders?status=all&symbols=SPY%2CSSO%2CBOXX&after=2026-10-26T00%3A00%3A00.000Z&limit=500&direction=asc", "/v2/positions",
  ]);
  assert.ok(w.calls.alpaca.every((c) => c.opts === undefined), "every broker call is a plain GET: no method, no body");
  assert.deepEqual(w.calls.spy, ["2026-05-06"], "SPY over the benchmark's own range (the curve's oldest day), so the shared cache is not narrowed");
  assert.deepEqual(w.calls.bars, [["SSO", "2026-10-26", "2026"], ["BOXX", "2026-10-26", "2026"]], "the other two tickers from 10 days before the first line");
  // with no equity curve the SPY read starts where the others do
  const fresh = await wire({ curve: [] });
  await fresh.floorTrackingBody();
  assert.deepEqual(fresh.calls.spy, ["2026-10-26"]);
});

test("the real floorTrackingBody never reads the broker when there is nothing to measure", async () => {
  for (const [stubs, status, why] of [
    [{ config: { ...CONFIG, FLOOR_MODE: "basket" } }, "not_applicable", /FLOOR_MODE is "basket"/],
    [{ configThrows: "Python mutex timeout" }, "unavailable", /the floor's config is unreadable \(Python mutex timeout\)/],
    [{ config: { FLOOR_MODE: "vol_managed_spy" } }, "unavailable", /config is unreadable \(system_config has no FLOOR_VM_TICKER or FLOOR_VM_LEVERED_TICKER\)/],
    [{ lines: [] }, "unavailable", /no daily line is stored yet/],
    [{ lines: [], linesError: "the daily line file is unreadable (Unexpected token)" }, "unavailable", /the daily lines are unreadable/],
  ] as Array<[Stubs, string, RegExp]>) {
    const w = await wire(stubs);
    const r = await w.floorTrackingBody();
    assert.equal(r.status, status);
    assert.match(r.reason ?? "", why);
    assert.deepEqual([w.calls.alpaca.length, w.calls.spy.length, w.calls.bars.length], [0, 0, 0], `no broker or bar read when ${status}: ${r.reason}`);
  }
});

test("the real floorTrackingBody: one failed read becomes that input's reason, and what does not need it is still measured", async () => {
  const noPositions = await (await wire({ positionsThrow: "Alpaca 503: upstream" })).floorTrackingBody();
  assert.equal(noPositions.status, "unavailable");
  assert.equal(noPositions.reason, "positions are unreadable (Alpaca 503: upstream)");
  assert.equal(noPositions.inputs.errors.positions, "Alpaca 503: upstream");
  const badPositions = await (await wire({ positions: { message: "forbidden" } })).floorTrackingBody();
  assert.equal(badPositions.reason, "positions are unreadable (the answer was not a list)");
  const badOrders = await (await wire({ orderPages: [{ message: "forbidden" }] })).floorTrackingBody();
  assert.equal(badOrders.reason, "the floor's orders are unreadable (an orders page was not a list)");
  const noSpy = await (await wire({ bars: { SPY: new Error("alpaca bars SPY -> 500"), SSO: [], BOXX: BOXX_BARS } })).floorTrackingBody();
  assert.equal(noSpy.reason, "SPY daily closes are unreadable (alpaca bars SPY -> 500)");
  // SSO is neither held nor traded: its failed read is recorded and costs nothing
  const noSso = await (await wire({ bars: { SPY: SPY_BARS, SSO: new Error("alpaca bars SSO -> 429"), BOXX: BOXX_BARS } })).floorTrackingBody();
  assert.equal(noSso.status, "ok");
  assert.equal(noSso.inputs.errors.levered_bars, "alpaca bars SSO -> 429");
  assert.equal(noSso.stored.sessions, 3);
  // the cash fund's failed read voids only the cash fund's own number
  const noBoxx = await (await wire({ bars: { SPY: SPY_BARS, SSO: [], BOXX: new Error("alpaca bars BOXX -> 404") } })).floorTrackingBody();
  assert.equal(noBoxx.stored.sessions, 3);
  assert.equal(noBoxx.latest?.cash_fund_pnl, null);
  assert.equal(noBoxx.latest?.reason, "cash fund: BOXX daily closes are unreadable");
});

test("the real orders reader pages by the last submission time and refuses a stalled or endless read", async () => {
  const page = (n: number, at: string) => Array.from({ length: n }, (_, k) => ({ id: `o${k}`, symbol: "QQQ", submitted_at: at }));
  const two = await wire({ orderPages: [page(500, "2026-11-01T15:00:00Z"), ORDERS] });
  const r = await two.floorTrackingBody();
  assert.equal(r.status, "ok");
  assert.equal(r.inputs.fills, 2, "the second page's fills are in");
  const orderCalls = two.calls.alpaca.map((c) => c.path).filter((p) => p.startsWith("/v2/orders?"));
  assert.equal(orderCalls.length, 2);
  assert.ok(orderCalls[1].includes(`&after=${encodeURIComponent("2026-11-01T15:00:00Z")}&`), "the next page starts after the last order of the page before");
  const stalled = await (await wire({ orderPages: [page(500, "2026-10-26T00:00:00.000Z")] })).floorTrackingBody();
  assert.equal(stalled.reason, "the floor's orders are unreadable (orders paging stalled)");
  const endless = await (await wire({ orderPages: Array.from({ length: 11 }, (_, k) => page(500, `2026-11-01T15:${String(k).padStart(2, "0")}:00Z`)) })).floorTrackingBody();
  assert.equal(endless.reason, "the floor's orders are unreadable (more than 5000 floor orders in the window)");
});

test("the real auditFloorTracking writes the report's line as one FLOOR-TRACKING row, nothing in basket mode, and a failure as a row", async () => {
  const w = await wire();
  await w.auditFloorTracking();
  assert.equal(w.calls.audit.length, 1);
  assert.equal(w.calls.audit[0][0], "FLOOR-TRACKING");
  assert.match(w.calls.audit[0][1], /^2026-11-11 floor \+1\.05% vs rule \+0\.96% \(0\.73x of SPY \+1\.32%\): tracking \+0\.09pp \| last 3 of 20 sessions \+0\.03pp \| entry 11-06 -0\.72pp \(in no window\) \| rollback trigger undetermined: /);
  assert.ok(w.calls.audit[0][1].length <= 490, "fits the audit log's 500 characters");
  const basket = await wire({ config: { ...CONFIG, FLOOR_MODE: "basket" } });
  await basket.auditFloorTracking();
  assert.deepEqual(basket.calls.audit, []);
  const down = await wire({ positionsThrow: "Alpaca 503: upstream" });
  await down.auditFloorTracking();
  assert.deepEqual(down.calls.audit, [["FLOOR-TRACKING", "floor tracking unavailable: positions are unreadable (Alpaca 503: upstream)"]]);
  const thrown = await wire({ linesThrow: "EIO: i/o error, read" });
  await thrown.auditFloorTracking();
  assert.deepEqual(thrown.calls.audit, [["FLOOR-TRACKING", "floor tracking unavailable: EIO: i/o error, read"]], "a throw inside the body is a row, never an unhandled rejection in the daily line job");
});
