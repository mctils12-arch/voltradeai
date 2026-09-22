// reconstruct_pnl diag probe's pure aggregator: option exclusion, exact-date
// close-to-close leg math, missing/short bar handling, and the optional
// reported_pnl side-by-side gap. Synthetic bar fixtures only — this module
// never touches fs/network. See research/open_questions.md KNOWN BROKEN #42
// (why this exists) and #44 (why it is TypeScript, not a scripts/*.py probe).
import { test } from "node:test";
import assert from "node:assert/strict";
import { isOptionPosition, reconstructPortfolioPnl, type Bar, type PositionInput } from "./reconstructPnl";

function bars(rows: Array<[string, number]>): Bar[] {
  return rows.map(([date, close]) => ({ date, close }));
}

test("isOptionPosition: OCC-length symbol or explicit us_option asset_class, never a plain equity ticker", () => {
  assert.equal(isOptionPosition("BAC261016P00057500"), true, "OCC symbols are always >8 chars");
  assert.equal(isOptionPosition("QQQ", "us_option"), true, "explicit asset_class always wins");
  assert.equal(isOptionPosition("QQQ", "us_equity"), false);
  assert.equal(isOptionPosition("VXUS"), false, "4-char equity ticker, no asset_class given");
});

test("reconstructPortfolioPnl: sums qty * (close - prevClose) across a multi-leg equity book", () => {
  const positions: PositionInput[] = [
    { symbol: "QQQ", qty: 51, assetClass: "us_equity", bars: bars([["2026-09-08", 718.36], ["2026-09-09", 716.31]]) },
    { symbol: "SMH", qty: 20, assetClass: "us_equity", bars: bars([["2026-09-08", 573.73], ["2026-09-09", 574.29]]) },
  ];
  const result = reconstructPortfolioPnl(positions, "2026-09-09");
  // QQQ: 51 * (716.31 - 718.36) = -104.55 ; SMH: 20 * (574.29 - 573.73) = 11.2
  assert.equal(result.reconstructed_pnl, -93.35);
  assert.equal(result.legs.length, 2);
  assert.equal(result.legs[0].contribution, -104.55);
  assert.equal(result.legs[1].contribution, 11.2);
  assert.deepEqual(result.excluded_options, []);
  assert.deepEqual(result.excluded_no_data, []);
});

test("reconstructPortfolioPnl: option legs are excluded, never mistreated as equities", () => {
  const positions: PositionInput[] = [
    { symbol: "BAC261016P00057500", qty: -1, assetClass: "us_option", bars: null },
    { symbol: "QQQ", qty: 10, assetClass: "us_equity", bars: bars([["2026-09-08", 100], ["2026-09-09", 101]]) },
  ];
  const result = reconstructPortfolioPnl(positions, "2026-09-09");
  assert.deepEqual(result.excluded_options, ["BAC261016P00057500"]);
  assert.equal(result.legs.length, 1);
  assert.equal(result.reconstructed_pnl, 10);
});

test("reconstructPortfolioPnl: excludes a symbol whose date isn't in the fetched bars (weekend/holiday), not a nearby substitute", () => {
  const positions: PositionInput[] = [
    { symbol: "QQQ", qty: 10, assetClass: "us_equity", bars: bars([["2026-09-04", 100], ["2026-09-08", 102]]) },
  ];
  const result = reconstructPortfolioPnl(positions, "2026-09-06"); // a Sunday, not in the series
  assert.equal(result.legs.length, 0);
  assert.equal(result.excluded_no_data.length, 1);
  assert.equal(result.excluded_no_data[0].reason, "date not in fetched bars");
  assert.equal(result.reconstructed_pnl, 0);
});

test("reconstructPortfolioPnl: excludes a symbol whose date is the FIRST bar in the window (no prior close to diff against)", () => {
  const positions: PositionInput[] = [
    { symbol: "QQQ", qty: 10, assetClass: "us_equity", bars: bars([["2026-09-09", 100]]) },
  ];
  const result = reconstructPortfolioPnl(positions, "2026-09-09");
  assert.equal(result.legs.length, 0);
  assert.equal(result.excluded_no_data[0].reason, "no prior-day bar in lookback window");
});

test("reconstructPortfolioPnl: null bars (fetch failure) excludes with an honest reason, never crashes or silently zeroes", () => {
  const positions: PositionInput[] = [
    { symbol: "FCEL", qty: 70, assetClass: "us_equity", bars: null },
  ];
  const result = reconstructPortfolioPnl(positions, "2026-09-09");
  assert.equal(result.excluded_no_data[0].reason, "bars fetch failed");
});

test("reconstructPortfolioPnl: zero/non-finite qty positions are skipped, never counted as a zero-value leg", () => {
  const positions: PositionInput[] = [
    { symbol: "QQQ", qty: 0, assetClass: "us_equity", bars: bars([["2026-09-08", 100], ["2026-09-09", 101]]) },
    { symbol: "SMH", qty: NaN, assetClass: "us_equity", bars: bars([["2026-09-08", 100], ["2026-09-09", 101]]) },
  ];
  const result = reconstructPortfolioPnl(positions, "2026-09-09");
  assert.equal(result.legs.length, 0);
  assert.equal(result.excluded_no_data.length, 0, "silently skipped, not reported as a data-failure exclusion");
});

test("reconstructPortfolioPnl: optional reported_pnl adds a side-by-side gap reading (the actual incident use case)", () => {
  const positions: PositionInput[] = [
    { symbol: "QQQ", qty: 51, assetClass: "us_equity", bars: bars([["2026-09-08", 718.36], ["2026-09-09", 716.31]]) },
  ];
  const result = reconstructPortfolioPnl(positions, "2026-09-09", -12059.74);
  assert.equal(result.reported_pnl, -12059.74);
  // reconstructed (-104.55) - reported (-12059.74) = 11955.19
  assert.equal(result.gap, 11955.19);
});

test("reconstructPortfolioPnl: omitted reported_pnl leaves reported_pnl/gap absent (not null/0)", () => {
  const result = reconstructPortfolioPnl([], "2026-09-09");
  assert.equal("reported_pnl" in result, false);
  assert.equal("gap" in result, false);
  assert.equal(result.reconstructed_pnl, 0);
});
