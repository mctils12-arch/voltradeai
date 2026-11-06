// LOSS ATTRIBUTION (2026-11-06, server/decisionAttribution.ts; refocus spec §8
// "why are we losing money"). The fixture is small enough to check by hand:
// realized in the window -$360 (two losing cash-secured puts closed at the open,
// a winning AMD, a floor GLD lot, a two-leg IREN structure nobody recorded),
// one open NVDA at -$60, one lot closed long before the window. Pins: every
// grouping adds up to the same total; each lot's signal / sector / instrument /
// regime label and where it came from; the plain-English reasons; execution
// through the daily line's own optionFillsInWindow + fillsSlippage; ET time
// windows; the wiring in bot.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  attributionReport, attributionDays, buildLots, readSectorCache, timeWindow, driverPhrase, reasonFor,
  NOT_RECORDED_SIGNAL, ATTRIBUTION_DEFAULT_DAYS, ATTRIBUTION_MAX_DAYS, SECTOR_CACHE_FILENAME, type AttributionInputs, type Bucket, type Lot,
} from "./decisionAttribution";
import { normalizeDecision, type DecisionRow } from "./decisionJournal";
import type { RealizedEvent } from "./sleeveLedger";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const src = fs.readFileSync(path.join(here, "decisionAttribution.ts"), "utf8");
const T = (iso: string) => Date.parse(iso);
const NOW = T("2026-11-06T21:00:00Z");              // 16:00 ET (EST, UTC-5)
const OCC1 = "TLT261219P00085000", OCC2 = "TLT261219P00084000";
const IREN_S = "IREN261204P00021000", IREN_L = "IREN261204P00020000";
const FLOOR = new Set(["SPY", "SSO", "QQQ", "GLD", "SVXY"]);

const ev = (symbol: string, openedAt: string | null, t: string, pnl: number, closeDir: 1 | -1, qty = 1): RealizedEvent =>
  ({ symbol, openedAt: openedAt === null ? null : T(openedAt), t: T(t), pnl, qty, closeDir, basis: 100 });
const row = (r: Record<string, unknown>): DecisionRow => normalizeDecision(r) as DecisionRow;

function inputs(over: Partial<AttributionInputs> = {}): AttributionInputs {
  return {
    nowMs: NOW, days: 30,
    events: [
      ev(OCC1, "2026-11-03T15:00:00Z", "2026-11-04T14:35:00Z", -300, 1),      // opened 10:00 ET, closed 09:35 ET
      ev(OCC2, "2026-11-04T14:45:00Z", "2026-11-05T14:40:00Z", -200, 1),      // opened 09:45 ET, closed 09:40 ET
      ev("AMD", "2026-11-02T15:30:00Z", "2026-11-05T19:00:00Z", 150, -1, 10), // 10:30 -> 14:00 ET
      ev("GLD", "2026-10-20T15:00:00Z", "2026-11-05T15:00:00Z", 40, -1, 5),   // the floor
      ev(IREN_S, "2026-11-02T18:00:00Z", "2026-11-04T20:50:00Z", -80, 1),     // a structure nobody recorded,
      ev(IREN_L, "2026-11-02T18:00:00Z", "2026-11-04T20:50:00Z", 30, -1),     //   closed at 15:50 ET
      ev("KO", "2026-08-01T15:00:00Z", "2026-09-01T15:00:00Z", -999, -1),     // outside the window
    ],
    fills: [{ symbol: "NVDA", t: T("2026-11-05T15:00:00Z"), dir: 1, qty: 5, price: 120 }],
    positions: [{ symbol: "NVDA", qty: 5, avgEntry: 120, unrealizedPl: -60, option: false }],
    decisions: [
      row({ ts: "2026-11-03T14:59:30Z", action: "buy", ticker: "TLT", symbol: OCC1, instrument: "option", chosen: "sell_cash_secured_put", signal: "tier 1 CSP harvest", stages: { regime: { label: "BULL" } } }),
      row({ ts: "2026-11-04T14:44:30Z", action: "buy", ticker: "TLT", symbol: OCC2, instrument: "option", chosen: "sell_cash_secured_put", signal: "tier 1 CSP harvest", stages: { regime: { label: "BULL" } } }),
      row({ ts: "2026-11-02T15:29:00Z", action: "buy", ticker: "AMD", symbol: "AMD", instrument: "stock", chosen: "buy_stock", signal: "deep_score scan, momentum-led", stages: { regime: { label: "BULL" }, sector: { name: "Technology" } } }),
      row({ ts: "2026-11-04T14:34:50Z", action: "sell", ticker: "TLT", symbol: OCC1, instrument: "option", chosen: "loss_limit" }),
      row({ ts: "2026-11-01T14:00:00Z", action: "skip", ticker: "AMD", rule: "rejected_score" }),
    ],
    orders: [
      { id: "o1", symbol: OCC1, asset_class: "us_option", side: "buy", filled_qty: "1", filled_avg_price: "2.10", filled_at: "2026-11-04T14:35:00Z" },
      { id: "o2", symbol: OCC2, asset_class: "us_option", side: "buy", filled_qty: "1", filled_avg_price: "1.50", filled_at: "2026-11-05T14:40:00Z" },
      { id: "o3", symbol: "AMD", asset_class: "us_equity", side: "sell", filled_qty: "10", filled_avg_price: "165", filled_at: "2026-11-05T19:00:00Z" },
    ],
    ledger: { rows: [{ order_id: "o1", symbol: OCC1, side: "buy", qty: 1, bid: 1.8, ask: 2.2, mid: 2.0, source: "options_manager" }], malformed: 0 },
    floorTickers: FLOOR,
    sectors: { NVDA: "Technology" },
    ...over,
  };
}

const sum = (b: readonly Bucket[]) => Math.round(b.reduce((s, x) => s + x.pnl, 0) * 100) / 100;
const by = (b: readonly Bucket[], key: string, v: string) => b.find((x) => x[key] === v);

test("totals: realized in the window + open P&L; every grouping (and each timing dimension) adds up to the total", () => {
  const r = attributionReport(inputs());
  assert.deepEqual([r.realized, r.unrealized, r.total, r.lots, r.openPositions], [-360, -60, -420, 6, 1]);
  for (const k of ["bySignal", "bySector", "byInstrument"] as const) assert.equal(sum(r[k]), -420, k);
  for (const dim of ["exit time", "entry time", "hold"]) assert.equal(sum(r.byTiming.filter((b) => b.dimension === dim)), -420, dim);
  assert.equal(r.window.days, 30);
  assert.equal(r.window.label, "the last 30 days");
  assert.ok(!JSON.stringify(r).includes("KO"), "a lot closed before the window is not in it");
  assert.deepEqual(r.bySignal.map((b) => b.signal), ["tier 1 CSP harvest", NOT_RECORDED_SIGNAL, "floor rebalance", "deep_score scan, momentum-led"], "biggest loss first");
});

test("labels: each lot's signal / instrument / sector / regime and where it came from", () => {
  const lots = buildLots(inputs());
  const get = (s: string): Lot => lots.find((l) => l.symbol === s)!;
  assert.deepEqual([get(OCC1).signal, get(OCC1).instrument, get(OCC1).sector, get(OCC1).regime, get(OCC1).exitRule],
    ["tier 1 CSP harvest", "options: cash-secured puts", "Treasuries", "BULL", "loss_limit"]);
  assert.deepEqual([get("AMD").signal, get("AMD").instrument, get("AMD").sector], ["deep_score scan, momentum-led", "stocks", "Technology"]);
  assert.deepEqual([get("GLD").signal, get("GLD").instrument, get("GLD").sector], ["floor rebalance", "floor", "Gold"]);
  assert.deepEqual([get(IREN_S).signal, get(IREN_S).instrument, get(IREN_S).sector, get(IREN_S).inferred], [NOT_RECORDED_SIGNAL, "options: put credit spreads", "Unknown", true]);
  assert.equal(get(OCC1).inferred, false, "a recorded strategy is not inferred");
  assert.equal(get(IREN_L).instrument, get(IREN_S).instrument, "both legs of the structure carry its inferred name");
  const nv = get("NVDA");
  assert.deepEqual([nv.open, nv.openedAt, nv.signal, nv.sector, nv.hold], [true, T("2026-11-05T15:00:00Z"), NOT_RECORDED_SIGNAL, "Technology", "1–7d"]);
  assert.deepEqual([get(OCC1).exitWindow, get(OCC2).entryWindow, get("AMD").entryWindow, get(IREN_S).exitWindow],
    ["the first 30 minutes (09:30–10:00 ET)", "the first 30 minutes (09:30–10:00 ET)", "the middle of the day (10:00–15:30 ET)", "the last 30 minutes (15:30–16:00 ET)"]);
});

test("reasons: plain English, with the main driver and the fill cost from the daily line's math", () => {
  const r = attributionReport(inputs());
  assert.equal(by(r.byInstrument, "instrument", "options: cash-secured puts")?.reason,
    "Options: cash-secured puts lost $500 across 2 closed trades (0 won, 2 lost); mostly from exits in the first 30 minutes "
    + "(09:30–10:00 ET): -$500 across 2 lots (100% of it); fills versus the mid at submission cost $10.00 of it (1 of 2 option fills priced).");
  assert.equal(by(r.byInstrument, "instrument", "stocks")?.reason,
    "Stocks made $90.00: $150 realized across 1 closed trade (1 won, 0 lost) and -$60.00 unrealized on 1 open position; "
    + "exits in the middle of the day (10:00–15:30 ET) alone came to $150 across 1 lot, more than the net (other lots offset part of it).");
  assert.equal(by(r.bySignal, "signal", NOT_RECORDED_SIGNAL)?.reason,
    "Trades whose entry decision was not recorded (opened before the decision journal) lost $110: -$50.00 realized across 2 closed "
    + "trades (1 won, 1 lost) and -$60.00 unrealized on 1 open position; mostly from positions held 1–7 days: -$110 across 3 lots (100% of it) "
    + "(the strategy of 2 of these 3 lots was inferred from the legs that opened together).");
  assert.equal(by(r.bySignal, "signal", "floor rebalance")?.reason, "The floor's rebalancing trades made $40.00 across 1 closed trade (1 won, 0 lost).");
  assert.equal(by(r.bySector, "sector", "Unknown")?.reason,
    "Unknown positions lost $50.00 across 2 closed trades (1 won, 1 lost); mostly from exits in the last 30 minutes (15:30–16:00 ET): -$50.00 across 2 lots (100% of it) "
    + "(the strategy of 2 of these 2 lots was inferred from the legs that opened together).");
  assert.equal(by(r.byInstrument, "instrument", "floor")?.reason, "Floor made $40.00 across 1 closed trade (1 won, 0 lost).");
  assert.equal(r.byTiming.find((b) => b.timing === "exits in the first 30 minutes (09:30–10:00 ET)")?.reason,
    "Exits in the first 30 minutes (09:30–10:00 ET) lost $500 across 2 closed trades (0 won, 2 lost); mostly from entries in a BULL market: "
    + "-$500 across 2 lots (100% of it); fills versus the mid at submission cost $10.00 of it (1 of 2 option fills priced).");
  assert.equal(r.byTiming.find((b) => b.timing === "still open")?.reason, "Still open is down $60.00 unrealized on 1 open position.");
});

test("execution: the window's option fills through optionFillsInWindow + fillsSlippage, split by time of day", () => {
  const r = attributionReport(inputs());
  assert.equal(r.byExecution.length, 1);
  const e = r.byExecution[0];
  assert.deepEqual([e.execution, e.pnl, e.n, e.fills, e.unattributed, e.cost], ["option fills in the first 30 minutes (09:30–10:00 ET)", -10, 1, 2, 1, 10]);
  assert.equal(e.reason, `1 of 2 option fills in the first 30 minutes (09:30–10:00 ET) were priced against the mid recorded at submission: they cost $10.00 (worst: ${OCC1}, buy 1 at 2.1 vs a 2 mid, $10.00). 1 had no recorded mid; their slippage stays inside the other buckets.`);
  const none = attributionReport(inputs({ orders: null, ordersError: "Alpaca 500" }));
  assert.deepEqual(none.byExecution, []);
  assert.ok(none.notes.some((n) => n.includes("could not be read (Alpaca 500)")));
  assert.doesNotMatch(by(none.byInstrument, "instrument", "options: cash-secured puts")?.reason ?? "", /fills versus the mid/);
  const noMid = attributionReport(inputs({ ledger: { rows: [], malformed: 0 } }));
  assert.match(noMid.byExecution[0].reason, /^2 option fills in the first 30 minutes \(09:30–10:00 ET\), none with a recorded mid/);
  // no second slippage formula here: the daily line's functions do the math
  assert.ok(src.includes('import { fillsSlippage, optionFillsInWindow, type LedgerRow, type OptionFill } from "./dailyLine";'));
  assert.ok(!/f\.price - |\.mid - f\./.test(src), "decisionAttribution.ts must not restate the fills-vs-mid formula");
});

test("worst trades, notes and sources", () => {
  const r = attributionReport(inputs());
  assert.deepEqual(r.worstTrades.map((w) => [w.symbol, w.pnl]), [[OCC1, -300], [OCC2, -200], [IREN_S, -80], ["NVDA", -60]]);
  assert.equal(r.worstTrades[3].closedAt, null);
  assert.equal(r.sources.unmatchedLots, 3, "IREN x2 and NVDA have no entry decision");
  assert.ok(r.notes.some((n) => n.startsWith("P&L: realized is the broker fill history's closed-lot walk")));
  assert.ok(r.notes.some((n) => n.includes("live since 2026-11-02")));
});

test("time windows (ET, DST-aware) and the day window", () => {
  assert.equal(timeWindow(T("2026-11-06T14:29:00Z")), "outside regular hours");                 // 09:29 EST
  assert.equal(timeWindow(T("2026-11-06T14:30:00Z")), "the first 30 minutes (09:30–10:00 ET)");
  assert.equal(timeWindow(T("2026-11-06T15:00:00Z")), "the middle of the day (10:00–15:30 ET)");
  assert.equal(timeWindow(T("2026-11-06T20:30:00Z")), "the last 30 minutes (15:30–16:00 ET)");
  assert.equal(timeWindow(T("2026-11-06T21:00:00Z")), "outside regular hours");
  assert.equal(timeWindow(T("2026-10-20T13:30:00Z")), "the first 30 minutes (09:30–10:00 ET)");  // 09:30 EDT
  assert.equal(timeWindow(T("2026-11-07T15:00:00Z")), "outside regular hours", "Saturday");
  assert.equal(timeWindow(null), "unknown");
  assert.equal(attributionDays(undefined), ATTRIBUTION_DEFAULT_DAYS);
  assert.equal(attributionDays("0"), 1);
  assert.equal(attributionDays("9999"), ATTRIBUTION_MAX_DAYS);
  assert.equal(attributionDays("7"), 7);
});

test("drivers: a sub-group above the net is said as such; one lot has no driver; a shared category is not a driver", () => {
  const lot = (pnl: number, regime: string): Lot => ({ symbol: "X", root: "X", option: false, sleeve: "stocks", openedAt: null, closedAt: 1, qty: 1, pnl, open: false,
    instrument: "stocks", signal: "s", sector: "Tech", regime, exitRule: null, entryWindow: "unknown", exitWindow: "unknown", hold: "unknown", inferred: false });
  const floorLots = [lot(-50, "x"), lot(-30, "x")].map((l, k) => ({ ...l, instrument: "floor", signal: "floor rebalance", regime: "unknown", sector: k ? "Gold" : "Index: Nasdaq-100" }));
  assert.equal(driverPhrase(floorLots, -80, ["instrument"], ["signal"]), "; mostly from Index: Nasdaq-100 names: -$50.00 across 1 lot (63% of it)",
    "floor <-> floor rebalance explains nothing: the driver is the next dimension");
  assert.equal(driverPhrase(floorLots, -80, ["sector"]), "; mostly from the floor: -$80.00 across 2 lots (100% of it)", "but it does explain a sector bucket");
  assert.equal(driverPhrase([lot(-300, "BULL"), lot(200, "BEAR")], -100, ["instrument", "signal", "sector"]),
    "; entries in a BULL market alone came to -$300 across 1 lot, more than the net (other lots offset part of it)");
  assert.equal(driverPhrase([lot(-300, "BULL")], -300, []), "");
  assert.equal(reasonFor("Stocks", [lot(0, "BULL"), lot(0, "BULL")], [], ""), "Stocks broke even on $0.00 across 2 closed trades (0 won, 0 lost).");
});

test("sector cache: missing or malformed = {}, keys upper-cased", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "attr-"));
  assert.deepEqual(readSectorCache(dir), {});
  fs.writeFileSync(path.join(dir, SECTOR_CACHE_FILENAME), "{bad");
  const orig = console.error;
  const errs: unknown[] = [];
  console.error = (...a: unknown[]) => { errs.push(a); };
  try { assert.deepEqual(readSectorCache(dir), {}); } finally { console.error = orig; }
  assert.equal(errs.length, 1, "an unreadable cache is logged, not silent");
  fs.writeFileSync(path.join(dir, SECTOR_CACHE_FILENAME), JSON.stringify({ amd: "Technology", X: 3 }));
  assert.deepEqual(readSectorCache(dir), { AMD: "Technology" });
});

test("wiring: the owner route and the diag probe share one cached body built from the sleeve cache, the journal and the daily line's readers", () => {
  const start = bot.indexOf("async function attributionBody(daysRaw: unknown)");
  assert.ok(start > 0);
  const body = bot.slice(start, bot.indexOf('app.get("/api/bot/attribution", requireOwner', start) + 400);
  for (const needle of ["attributionCache.get(days)", "ATTRIBUTION_CACHE_MS", "await sleevesBody();", "sleeveCache.taxInputs.events", "sleeveCache.taxInputs.fills",
    "await dailyLineOrders(", "readOptionFillLedger(dailyLineDir(process.env))", "readDecisions(dir, { limit: ATTRIBUTION_MAX_DECISIONS })",
    "readSectorCache(dir)", "FLOOR_AND_LEG_TICKERS", 'app.get("/api/bot/attribution", requireOwner']) {
    assert.ok(body.includes(needle), needle);
  }
  const probe = bot.slice(bot.indexOf('case "attribution": {'), bot.indexOf('case "decisions": {'));
  assert.ok(probe.includes('sanitizeDiag({ probe: "attribution", ...(await attributionBody(req.query.days)) })'));
});
