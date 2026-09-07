// unComtrade.ts — pure-function view over the static UN Comtrade archive.
// Mirrors jodiOil.test.ts's synthetic-fixture style (no network, no live
// archive dependency for the logic tests) plus a real-file coherence check.
import { test } from "node:test";
import assert from "node:assert/strict";
import { unComtradeView, UN_COMTRADE_GATE_NOTE } from "./unComtrade";
import unComtradeArchive from "../datacore/un_comtrade/bilateral_trade.json";

function fixture(overrides: Record<string, unknown> = {}) {
  return {
    source: "https://comtradeapi.un.org/public/v1/preview/C/M/HS",
    attribution: "UN Comtrade Database, https://comtradeapi.un.org",
    license: "free with citation",
    built_at: "2026-09-07T00:00:00Z",
    reporter: { code: 842, name: "USA" },
    partners: { "156": "China", "484": "Mexico", "999": "NoData" },
    flows: { M: "imports", X: "exports" },
    cmd_code: "TOTAL",
    latest_period: "202506",
    series_count: 4,
    series: {
      "156|M": { points: [["202505", 21791159433, 20276165298], ["202506", 20175592074, 18946982831]], n: 2, first: "202505", last: "202506" },
      "156|X": { points: [["202506", null, 9758375944]], n: 1, first: "202506", last: "202506" },
      "484|M": { points: [["202506", 26438967616, 25100000000]], n: 1, first: "202506", last: "202506" },
      // "484|X" intentionally absent — a partner can have imports with no matching exports row yet
      "999|M": { points: [], n: 0, first: null, last: null }, // empty series — excluded
      ...overrides,
    },
  } as any;
}

test("a partner with no M-flow points is excluded, never zero-filled", () => {
  const v = unComtradeView(fixture());
  const codes = v.rows.map((r) => r.partnerCode).sort();
  assert.deepEqual(codes, [156, 484]);
});

test("uses each partner's own latest period, not a shared calendar date", () => {
  const v = unComtradeView(fixture());
  const china = v.rows.find((r) => r.partnerCode === 156)!;
  assert.equal(china.period, "202506");
  assert.equal(china.importsCifUsd, 20175592074);
  assert.equal(china.priorPeriod, "202505");
  assert.equal(china.priorImportsCifUsd, 21791159433);
});

test("rows sort by latest imports (CIF) descending", () => {
  const v = unComtradeView(fixture());
  // fixture: Mexico's latest CIF (26.4B) > China's (20.2B)
  assert.equal(v.rows[0].partnerCode, 484);
  assert.equal(v.rows[1].partnerCode, 156);
});

test("delta pct computed against the series own prior point", () => {
  const v = unComtradeView(fixture());
  const china = v.rows.find((r) => r.partnerCode === 156)!;
  const expected = Math.round(((20175592074 - 21791159433) / 21791159433) * 1000) / 10;
  assert.equal(china.importsDeltaPct, expected);
  assert.ok(china.importsDeltaPct! < 0, "imports fell month over month in this fixture");
});

test("single-point series reports null prior/delta, never zero", () => {
  const v = unComtradeView(fixture());
  const mexico = v.rows.find((r) => r.partnerCode === 484)!;
  assert.equal(mexico.priorPeriod, null);
  assert.equal(mexico.priorImportsCifUsd, null);
  assert.equal(mexico.importsDeltaPct, null);
});

test("trade balance is fob(exports) minus cif(imports) only when both sides exist for the same period", () => {
  const v = unComtradeView(fixture());
  const china = v.rows.find((r) => r.partnerCode === 156)!;
  assert.equal(china.tradeBalanceUsd, Math.round(9758375944 - 20175592074));
  const mexico = v.rows.find((r) => r.partnerCode === 484)!;
  assert.equal(mexico.tradeBalanceUsd, null, "no matching export point in this fixture");
});

test("view is marked raw/non-predictive with the gate-1 note carried verbatim", () => {
  const v = unComtradeView(fixture());
  assert.equal(v.kind, "raw");
  assert.equal(v.predictive, false);
  assert.equal(v.note, UN_COMTRADE_GATE_NOTE);
  assert.match(v.note, /GATE 1/);
});

test("committed archive produces a coherent live view", () => {
  const v = unComtradeView(unComtradeArchive as any);
  assert.ok(v.rows.length >= 5, "expects at least 5 of the 6 archived partners to have M-flow data");
  for (const r of v.rows) {
    assert.ok(r.importsCifUsd == null || r.importsCifUsd > 0);
    assert.ok(typeof r.partnerName === "string" && r.partnerName.length > 0);
  }
  // sorted descending by imports
  for (let i = 1; i < v.rows.length; i++) {
    assert.ok((v.rows[i - 1].importsCifUsd ?? 0) >= (v.rows[i].importsCifUsd ?? 0));
  }
});
