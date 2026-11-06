// server/positionMgmt.ts — the Node half of position management (2026-11-06):
// the cash-reserve budget an entry loop spends, the rotation plan's parse and
// live-book preflight, and the shared JSONL log. Pure module, direct tests.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ReserveBudget, parseReserveSnapshot, fitTradeToReserve, reserveAuditLine,
  parseRotationPlan, rotationPreflight, appendPositionMgmtLog, positionMgmtLogPath,
  POSITION_MGMT_LOG_FILE, POSITION_MGMT_LOG_MAX_BYTES, type PreflightInput, type ReserveSnapshot,
} from "./positionMgmt";

const SNAP = {
  enabled: true, regime: "BULL", reserve_pct: 0.05, reserve_dollars: 5000, headroom_dollars: 10000,
  floor_committed: 85000, satellite_committed: 0, min_entry_pct: 0.01, summary: "BULL market: 5% ...",
};
const snap = (over: Record<string, unknown> = {}): ReserveSnapshot => parseReserveSnapshot({ ...SNAP, ...over })!;

test("parseReserveSnapshot reads position_mgmt's dict and refuses anything else", () => {
  const s = parseReserveSnapshot(SNAP)!;
  assert.equal(s.enabled, true);
  assert.equal(s.regime, "BULL");
  assert.equal(s.headroomDollars, 10000);
  assert.equal(s.reservePct, 0.05);
  assert.equal(s.minEntryPct, 0.01);
  assert.equal(parseReserveSnapshot(null), null);
  assert.equal(parseReserveSnapshot({ headroom_dollars: 5 }), null);          // no enabled flag
  assert.equal(parseReserveSnapshot({ enabled: true, headroom_dollars: "x" }), null);
  assert.equal(parseReserveSnapshot({ ...SNAP, headroom_dollars: -50 })!.headroomDollars, 0);
});

test("fitShares allows, cuts to whole shares, or blocks below the smallest worthwhile size", () => {
  const b = new ReserveBudget(snap(), 100_000);
  assert.deepEqual(b.fitShares(100, 40).action, "allow");
  const cut = b.fitShares(100, 150);                                          // $15k wanted, $10k room
  assert.equal(cut.action, "trim");
  assert.equal(cut.qty, 100);
  assert.equal(cut.value, 10_000);
  assert.match(cut.reason, /cut from \$15,000 to \$10,000/);
  const small = new ReserveBudget(snap({ headroom_dollars: 900 }), 100_000).fitShares(100, 20);
  assert.equal(small.action, "block");                                        // 9 shares = $900 < 1% of equity
  assert.match(small.reason, /smallest worthwhile/);
  assert.equal(new ReserveBudget(snap({ headroom_dollars: 0 }), 100_000).fitShares(100, 1).action, "block");
});

test("spend draws the budget down; an inactive budget gates nothing", () => {
  const b = new ReserveBudget(snap(), 100_000);
  b.spend(7_000);
  assert.equal(b.remaining, 3_000);
  assert.equal(b.fitShares(100, 40).action, "trim");
  b.spend(3_000);
  assert.equal(b.fitOpen().action, "block");
  for (const off of [new ReserveBudget(null, 100_000), new ReserveBudget(snap({ enabled: false }), 100_000)]) {
    assert.equal(off.active, false);
    assert.equal(off.fitShares(100, 10_000).action, "allow");
    assert.equal(off.fitOpen().action, "allow");
    assert.equal(off.capFor(1234), 1234);
    off.spend(1e9);
    assert.equal(off.remaining, Number.POSITIVE_INFINITY);
  }
});

test("capFor caps a sell-to-open's collateral at the room above the reserve", () => {
  const b = new ReserveBudget(snap(), 100_000);
  assert.equal(b.capFor(25_000), 10_000);
  assert.equal(b.capFor(4_000), 4_000);
});

test("fitTradeToReserve: stock trims its shares, an ETF its own shares and the stock fallback's", () => {
  const stock = { ticker: "AAPL", price: 200, shares: 75, position_value: 15_000 };
  const f1 = fitTradeToReserve(new ReserveBudget(snap(), 100_000), stock, false);
  assert.equal(f1.action, "trim");
  assert.equal(stock.shares, 50);
  assert.equal(stock.position_value, 10_000);

  const etf = { ticker: "TSLA", price: 250, shares: 60, position_value: 15_000, instrument: "etf",
    instrument_ticker: "TSLL", etf_shares: 1200, etf_price: 12.5, etf_position_value: 15_000 };
  const b = new ReserveBudget(snap(), 100_000);
  const f2 = fitTradeToReserve(b, etf, false);
  assert.equal(f2.action, "trim");
  assert.equal(etf.etf_shares, 800);
  assert.equal(etf.etf_position_value, 10_000);
  assert.equal(etf.shares, 40);                                                // the stock fallback, same dollars
  assert.equal(b.remaining, 0);

  const opt = { ticker: "AMD", price: 150, shares: 0, position_value: 0 };
  const b2 = new ReserveBudget(snap(), 100_000);
  assert.equal(fitTradeToReserve(b2, opt, true).action, "allow");
  assert.equal(b2.remaining, 10_000, "an option open spends only once its cost is known");

  const b3 = new ReserveBudget(snap({ headroom_dollars: 0 }), 100_000);
  assert.equal(fitTradeToReserve(b3, { ticker: "X", price: 10, shares: 5 }, false).action, "block");
});

test("reserveAuditLine is one plain-English sentence with the regime and the reserve", () => {
  const line = reserveAuditLine("NVDA", { action: "block", qty: 0, value: 0, reason: "the cash reserve leaves no room for a new satellite position" }, snap({ regime: "BEAR", reserve_pct: 0.25 }));
  assert.equal(line, "NVDA: the cash reserve leaves no room for a new satellite position — the BEAR market keeps 25% of equity in cash");
});

test("reserveAuditLine names the regime risk budget when it set the reserve", () => {
  const s = snap({ reserve_reason: "the regime risk budget x0.50 lets satellites use only 50% of the $15,000 room above the floor (VIX 24)", risk_budget: 0.5 });
  assert.equal(s.riskBudget, 0.5);
  assert.equal(snap().riskBudget, null);
  const line = reserveAuditLine("AMD", { action: "trim", qty: 10, value: 1000, reason: "cut from $2,000 to $1,000" }, s);
  assert.equal(line, "AMD: cut from $2,000 to $1,000 — the regime risk budget x0.50 lets satellites use only 50% of the $15,000 room above the floor (VIX 24)");
});

const PLAN = {
  rotations: [{
    sell: { ticker: "aapl", qty: 10, market_value: 1053, pnl_pct: 6, state: "run_its_course" },
    buy: { ticker: "TNA", underlying: "TNA", price: 40, score: 80, instrument: "etf", value: 1050 },
    summary: "Sell AAPL ... buy TNA", gain_pct: 1.4, ev_hold_pct: 0, ev_candidate_pct: 2.2, tax_now: 14.3, cost_dollars: 2.1,
  }, { sell: { ticker: "X" }, buy: { ticker: "X" } }, "junk", { sell: {}, buy: { ticker: "Y" } }],
  rejected: [],
};

test("parseRotationPlan keeps well-formed swaps only", () => {
  const rows = parseRotationPlan(PLAN);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].sell.ticker, "AAPL");
  assert.equal(rows[0].buy.value, 1050);
  assert.equal(rows[0].evCandidatePct, 2.2);
  assert.deepEqual(parseRotationPlan(null), []);
  assert.deepEqual(parseRotationPlan({ rotations: "x" }), []);
  assert.deepEqual(parseRotationPlan({ rotations: [], error: "boom" }), []);
});

const ROW = parseRotationPlan(PLAN)[0];
const base = (over: Partial<PreflightInput> = {}): PreflightInput => ({
  positions: [{ symbol: "AAPL", qty: "10", side: "long", asset_class: "us_equity", market_value: "1053", unrealized_plpc: "0.06" }],
  openOrders: [], floorTickers: new Set(["SPY", "SSO", "GLD"]), monitored: undefined, exiting: false,
  etHour: 11, marketOpen: true, ...over,
});

test("rotationPreflight passes a still-valid swap and sizes the buy from the live sale value", () => {
  const p = rotationPreflight(ROW, base());
  assert.equal(p.ok, true);
  assert.equal(p.sellQty, 10);
  assert.equal(p.sellValue, 1053);
  assert.equal(p.buyValue, 1050);
  assert.equal(p.buyQty, 26);                                                  // floor(1050 / 40)
  assert.ok(Math.abs(p.pnlPct - 6) < 1e-9);
});

test("rotationPreflight refuses every unsafe case with a reason", () => {
  const cases: Array<[Partial<PreflightInput>, RegExp]> = [
    [{ etHour: 9.6 }, /rotation window/],
    [{ marketOpen: false }, /rotation window/],
    [{ positions: [] }, /no longer held/],
    [{ monitored: { pendingExit: true } }, /exit working/],
    [{ exiting: true }, /exit working/],
    [{ openOrders: [{ symbol: "AAPL" }] }, /order working at the broker/],
    [{ positions: [...base().positions, { symbol: "TNA", qty: "5" }] }, /TNA is already held/],
    [{ openOrders: [{ symbol: "TNA" }] }, /TNA already has an order working/],
  ];
  for (const [over, why] of cases) {
    const p = rotationPreflight(ROW, base(over));
    assert.equal(p.ok, false, String(why));
    assert.match(p.reason, why);
  }
  assert.match(rotationPreflight({ ...ROW, sell: { ...ROW.sell, ticker: "GLD" } }, base()).reason, /floor or options/);
  assert.match(rotationPreflight({ ...ROW, buy: { ...ROW.buy, price: 0 } }, base()).reason, /no price/);
  assert.match(rotationPreflight({ ...ROW, buy: { ...ROW.buy, price: 5000 } }, base()).reason, /more than the/);
});

test("appendPositionMgmtLog writes one JSON object per line, deduplicates, and rotates", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pm-log-"));
  const file = path.join(dir, POSITION_MGMT_LOG_FILE);
  assert.equal(appendPositionMgmtLog({ kind: "rotation", action: "rotate", ticker: "aapl", summary: "Sold AAPL." }, { file, nowMs: 1_000 }), true);
  assert.equal(appendPositionMgmtLog({ kind: "cash_reserve", action: "trim", ticker: "nvda", summary: "x" }, { file, dedupKey: "k", dedupMs: 60_000, nowMs: 2_000 }), true);
  assert.equal(appendPositionMgmtLog({ kind: "cash_reserve", action: "trim", ticker: "nvda", summary: "x" }, { file, dedupKey: "k", dedupMs: 60_000, nowMs: 3_000 }), false);
  const rows = fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(rows.length, 2);
  assert.deepEqual([rows[0].ticker, rows[0].kind, rows[0].ts], ["AAPL", "rotation", new Date(1_000).toISOString()]);
  fs.writeFileSync(file, Buffer.alloc(POSITION_MGMT_LOG_MAX_BYTES + 1, 0x20));
  assert.equal(appendPositionMgmtLog({ kind: "rotation", action: "failed", ticker: "x", summary: "y" }, { file }), true);
  assert.ok(fs.existsSync(`${file}.1`));
  assert.equal(fs.readFileSync(file, "utf8").trim().split("\n").length, 1);
  assert.equal(positionMgmtLogPath({ DATA_DIR: "/srv/x" }), path.join("/srv/x", POSITION_MGMT_LOG_FILE));
});

test("the log file name is the one storage_config / position_mgmt.py write", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const storage = fs.readFileSync(path.join(here, "..", "storage_config.py"), "utf8");
  assert.ok(storage.includes(`POSITION_MGMT_LOG_PATH = os.path.join(DATA_DIR, "${POSITION_MGMT_LOG_FILE}")`));
});
