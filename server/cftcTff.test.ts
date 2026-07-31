// CFTC TFF battery (BUILD ORDER 6 #1): financial-futures field-name
// quirks (dealer keeps _all, asset_mgr/lev_money drop it), week
// boundary discipline, week-level dedup, restart rebuild from archive.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseTff, fetchLatestTff, archiveTffWeek, gzipOldTffWeeks,
  refreshTff, latestTff, readArchivedTffWeek, TFF_WEEK_ROW_LIMIT,
  tffAccountingIssues,
} from "./cftcTff";

// A REAL row pulled live from publicreporting.cftc.gov/resource/gpe5-46if.json
// 2026-07-31 (report_date 2026-07-21, 3 YEAR ERIS SOFR SWAP, code 344606) —
// used to gate-1-verify the accounting-identity check against genuine CFTC
// arithmetic, not a hand-built fixture that could bake in the same bug it's
// meant to catch.
const REAL_ROW = {
  market_and_exchange_names: "3 YEAR ERIS SOFR SWAP - CHICAGO BOARD OF TRADE",
  report_date_as_yyyy_mm_dd: "2026-07-21T00:00:00.000",
  cftc_contract_market_code: "344606",
  commodity_name: "2-7-YEAR INTEREST RATE SWAPS ERIS & MAC",
  open_interest_all: "78626",
  dealer_positions_long_all: "135",
  dealer_positions_short_all: "26162",
  dealer_positions_spread_all: "6843",
  asset_mgr_positions_long: "40000",
  asset_mgr_positions_short: "4548",
  asset_mgr_positions_spread: "0",
  lev_money_positions_long: "7447",
  lev_money_positions_short: "11985",
  lev_money_positions_spread: "19691",
  other_rept_positions_long: "4500",
  other_rept_positions_short: "9396",
  other_rept_positions_spread: "0",
  tot_rept_positions_long_all: "78616",
  tot_rept_positions_short: "78625",
  nonrept_positions_long_all: "10",
  nonrept_positions_short_all: "1",
};

test("tffAccountingIssues: GATE 1 — a real CFTC record passes its own accounting identities", () => {
  assert.deepEqual(tffAccountingIssues(REAL_ROW), []);
});

test("tffAccountingIssues: rejects a corrupted long-side total (field-mapping-bug shape)", () => {
  // Corrupting tot_rept_positions_long_all breaks BOTH identities that read it
  // (the category-sum check AND the long-side open-interest check) — real
  // cross-field coupling, not a test artifact; assert both fire.
  const bad = { ...REAL_ROW, tot_rept_positions_long_all: "99999" };
  const issues = tffAccountingIssues(bad);
  assert.equal(issues.length, 2);
  assert.ok(issues.every((i) => i.rule === "identity"));
  assert.deepEqual(issues.map((i) => i.field).sort(), ["open_interest_all", "tot_rept_positions_long_all"]);
});

test("tffAccountingIssues: rejects a corrupted open-interest denominator", () => {
  const bad = { ...REAL_ROW, open_interest_all: "1" };
  const issues = tffAccountingIssues(bad);
  assert.ok(issues.some((i) => i.field === "open_interest_all"));
});

test("tffAccountingIssues: zero/missing open interest rejects immediately, no false identity noise", () => {
  assert.deepEqual(tffAccountingIssues({ ...REAL_ROW, open_interest_all: "0" }),
    [{ field: "open_interest_all", rule: "min", detail: "open interest zero/missing" }]);
  assert.deepEqual(tffAccountingIssues({}),
    [{ field: "open_interest_all", rule: "min", detail: "open interest zero/missing" }]);
});

test("fetchLatestTff: GATE 1 quarantines a corrupted row and archives only the valid one", async () => {
  const corrupted = { ...REAL_ROW, cftc_contract_market_code: "999999", tot_rept_positions_short: "1" };
  const ok = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([REAL_ROW, corrupted]) });
  const rows = await fetchLatestTff(ok as any, Date.parse("2026-07-31T00:00:00Z"));
  assert.equal(rows.length, 1, "the accounting-broken row never reaches parseTff/the archive");
  assert.equal(rows[0].code, "344606");
});

// Mirrors the live Socrata shape verified 2026-07-06 — dealer_* keeps
// the _all suffix, asset_mgr_*/lev_money_* drop it.
const ROW = (date: string, code: string, mkt: string, levLong: string, levShort: string) => ({
  market_and_exchange_names: mkt,
  report_date_as_yyyy_mm_dd: `${date}T00:00:00.000`,
  cftc_contract_market_code: code,
  commodity_name: "E-MINI S&P 500",
  open_interest_all: "2143567",
  dealer_positions_long_all: "112034",
  dealer_positions_short_all: "487221",
  dealer_positions_spread_all: "51002",
  asset_mgr_positions_long: "1183440",
  asset_mgr_positions_short: "231819",
  asset_mgr_positions_spread: "70455",
  lev_money_positions_long: levLong,
  lev_money_positions_short: levShort,
  lev_money_positions_spread: "34120",
  other_rept_positions_long: "89211",
  other_rept_positions_short: "44120",
  other_rept_positions_spread: "12005",
  nonrept_positions_long_all: "155230",
  nonrept_positions_short_all: "98122",
});

test("parseTff: quirky suffixes land, strings become numbers, ''/missing stay null", () => {
  const rows = parseTff([ROW("2026-06-23", "13874A", "E-MINI S&P 500 - CME", "301442", "")], "2026-07-06");
  assert.equal(rows.length, 1);
  const r = rows[0];
  assert.equal(r.report_date, "2026-06-23");
  assert.equal(r.dealer_short, 487221, "dealer field WITH _all suffix parsed");
  assert.equal(r.asset_mgr_long, 1183440, "asset_mgr field WITHOUT _all suffix parsed");
  assert.equal(r.lev_money_long, 301442);
  assert.equal(r.lev_money_short, null, "empty string is null, never zero");
  assert.equal(r.nonrept_long, 155230);
  assert.deepEqual(parseTff([], "x"), []);
  assert.deepEqual(parseTff(null, "x"), []);
  assert.deepEqual(parseTff([{ no_fields: 1 }], "x"), []);
});

test("parseTff: a DESC batch straddling two weeks keeps ONLY the newest week", () => {
  const rows = parseTff([
    ROW("2026-06-23", "A1", "NEW WEEK MKT", "1", "2"),
    ROW("2026-06-16", "A1", "OLD WEEK MKT", "9", "9"),
    ROW("2026-06-23", "B2", "NEW WEEK MKT 2", "3", "4"),
  ], "2026-07-06");
  assert.equal(rows.length, 2);
  assert.ok(rows.every((r) => r.report_date === "2026-06-23"), "vintages never mix in one file");
});

test("fetchLatestTff: non-200 returns [] and logs; url carries limit + DESC order", async () => {
  const urls: string[] = [];
  const bad = async (url: string) => { urls.push(url); return { ok: false, status: 500, text: async () => "" }; };
  assert.deepEqual(await fetchLatestTff(bad as any), []);
  assert.ok(urls[0].includes(`$limit=${TFF_WEEK_ROW_LIMIT}`) && urls[0].includes("DESC"));
  assert.ok(urls[0].includes("gpe5-46if"), "TFF dataset id, never the disaggregated one");
});

test("archive: week-level dedup + gz after supersession window + gz readback", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tff-"));
  const rows = parseTff([ROW("2026-06-23", "13874A", "E-MINI S&P 500 - CME", "301442", "15590")], "2026-07-06");
  assert.equal(archiveTffWeek(rows, base), 1);
  assert.equal(archiveTffWeek(rows, base), 0, "same report week never re-archives");
  assert.equal(gzipOldTffWeeks(base, Date.parse("2026-06-30T00:00:00Z")), 0, "within 9d stays plain");
  assert.equal(gzipOldTffWeeks(base, Date.parse("2026-07-03T00:00:00Z")), 1);
  assert.equal(readArchivedTffWeek("2026-06-23", base).length, 1, "gz week still readable");
});

test("refresh: restart with fetch down rebuilds cache from the newest archived week", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tff-"));
  archiveTffWeek(parseTff([ROW("2026-06-30", "Z9", "RESTART MKT", "5", "6")], "2026-07-06"), base);
  const dead = async () => ({ ok: false, status: 503, text: async () => "" });
  await refreshTff(dead as any, Date.parse("2026-07-06T12:00:00Z"), base);
  const hit = latestTff();
  assert.ok(hit, "cache rebuilt from archive despite fetch being down");
  assert.equal(hit!.report_date, "2026-06-30");
  assert.equal(hit!.rows[0].market, "RESTART MKT");
});
