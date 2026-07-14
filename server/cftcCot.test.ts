// CFTC COT battery (BUILD ORDER 5 #2): quirky-field-name parse, week
// boundary discipline, week-level dedup, restart rebuild from archive.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseCot, fetchLatestCot, archiveCotWeek, gzipOldCotWeeks,
  refreshCot, latestCot, readArchivedWeek, WEEK_ROW_LIMIT,
  listArchivedDates, searchMarkets, lookupMarketHistory, readAggregateHistory,
} from "./cftcCot";

// Mirrors the live Socrata shape verified 2026-07-05 — including the
// double-underscore swap fields and the missing _all suffixes.
const ROW = (date: string, code: string, mkt: string, mmLong: string, mmShort: string) => ({
  market_and_exchange_names: mkt,
  report_date_as_yyyy_mm_dd: `${date}T00:00:00.000`,
  cftc_contract_market_code: code,
  commodity_name: "CRUDE OIL",
  open_interest_all: "428305",
  prod_merc_positions_long: "61192",
  prod_merc_positions_short: "83194",
  swap_positions_long_all: "89952",
  swap__positions_short_all: "19306",
  swap__positions_spread_all: "13358",
  m_money_positions_long_all: mmLong,
  m_money_positions_short_all: mmShort,
  m_money_positions_spread: "86545",
  other_rept_positions_long: "49364",
  other_rept_positions_short: "27342",
  other_rept_positions_spread: "25749",
});

test("parseCot: quirky field names land, strings become numbers, ''/missing stay null", () => {
  const rows = parseCot([ROW("2026-06-23", "067651", "CRUDE OIL - NYMEX", "68457", "")], "2026-07-05");
  assert.equal(rows.length, 1);
  const r = rows[0];
  assert.equal(r.report_date, "2026-06-23");
  assert.equal(r.swap_short, 19306, "double-underscore source field parsed");
  assert.equal(r.swap_spread, 13358);
  assert.equal(r.m_money_long, 68457);
  assert.equal(r.m_money_short, null, "empty string is null, never zero");
  assert.deepEqual(parseCot([], "x"), []);
  assert.deepEqual(parseCot(null, "x"), []);
  assert.deepEqual(parseCot([{ no_fields: 1 }], "x"), []);
});

test("parseCot: a DESC batch straddling two weeks keeps ONLY the newest week", () => {
  const rows = parseCot([
    ROW("2026-06-23", "A1", "NEW WEEK MKT", "1", "2"),
    ROW("2026-06-16", "A1", "OLD WEEK MKT", "9", "9"),
    ROW("2026-06-23", "B2", "NEW WEEK MKT 2", "3", "4"),
  ], "2026-07-05");
  assert.equal(rows.length, 2);
  assert.ok(rows.every((r) => r.report_date === "2026-06-23"), "vintages never mix in one file");
});

test("fetchLatestCot: non-200 returns [] and logs; url carries limit + DESC order", async () => {
  const urls: string[] = [];
  const bad = async (url: string) => { urls.push(url); return { ok: false, status: 500, text: async () => "" }; };
  assert.deepEqual(await fetchLatestCot(bad as any), []);
  assert.ok(urls[0].includes(`$limit=${WEEK_ROW_LIMIT}`) && urls[0].includes("DESC"));
});

test("archive: week-level dedup + gz after supersession window + gz readback", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "cot-"));
  const rows = parseCot([ROW("2026-06-23", "067651", "CRUDE OIL - NYMEX", "68457", "138890")], "2026-07-05");
  assert.equal(archiveCotWeek(rows, base), 1);
  assert.equal(archiveCotWeek(rows, base), 0, "same report week never re-archives");
  assert.equal(gzipOldCotWeeks(base, Date.parse("2026-06-30T00:00:00Z")), 0, "within 9d stays plain");
  assert.equal(gzipOldCotWeeks(base, Date.parse("2026-07-03T00:00:00Z")), 1);
  assert.equal(readArchivedWeek("2026-06-23", base).length, 1, "gz week still readable");
});

test("refresh: restart with fetch down rebuilds cache from the newest archived week", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "cot-"));
  archiveCotWeek(parseCot([ROW("2026-06-30", "Z9", "RESTART MKT", "5", "6")], "2026-07-05"), base);
  const dead = async () => ({ ok: false, status: 503, text: async () => "" });
  await refreshCot(dead as any, Date.parse("2026-07-05T12:00:00Z"), base);
  const hit = latestCot();
  assert.ok(hit, "cache rebuilt from archive despite fetch being down");
  assert.equal(hit!.report_date, "2026-06-30");
  assert.equal(hit!.rows[0].market, "RESTART MKT");
});

// #/data/cot full view (BUILD ORDER 5 #2 UI follow-up): listArchivedDates/
// searchMarkets/lookupMarketHistory/readAggregateHistory, mirroring the
// wikiAttention.test.ts ratchet for its sibling history functions. Day-files
// written DIRECTLY to disk rather than via archiveCotWeek, sidestepping
// that function's module-level `archivedWeeks` dedup Set (shared across
// this whole test file regardless of base dir — same caveat
// wikiAttention.test.ts documents for its own archivedKeys set).
function writeWeek(base: string, date: string, rows: ReturnType<typeof parseCot>) {
  const dir = path.join(base, "cftccot");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, `${date}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
}

test("listArchivedDates: newest-first, plain and gz files both listed", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "cot-"));
  writeWeek(base, "2026-05-19", parseCot([ROW("2026-05-19", "A1", "OLDER MKT", "1", "2")], "rt"));
  writeWeek(base, "2026-05-26", parseCot([ROW("2026-05-26", "A1", "NEWER MKT", "3", "4")], "rt"));
  gzipOldCotWeeks(base, Date.parse("2026-07-10T00:00:00Z")); // both weeks now >9d old -> gz
  assert.deepEqual(listArchivedDates(base), ["2026-05-26", "2026-05-19"]);
  assert.deepEqual(listArchivedDates(base, 1), ["2026-05-26"], "limit respected");
});

test("searchMarkets: matches by exact code or case-insensitive name/commodity substring, off the newest week only", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "cot-"));
  writeWeek(base, "2026-05-19", parseCot([ROW("2026-05-19", "OLD1", "OLD WEEK CRUDE", "1", "2")], "rt"));
  writeWeek(base, "2026-05-26", parseCot([
    ROW("2026-05-26", "067651", "CRUDE OIL - NYMEX", "68457", "138890"),
    ROW("2026-05-26", "088691", "GOLD - COMMODITY EXCHANGE", "91200", "40100"),
  ], "rt"));
  assert.deepEqual(searchMarkets("067651", base).map((m) => m.code), ["067651"], "exact code match");
  assert.deepEqual(searchMarkets("nymex", base).map((m) => m.code), ["067651"], "case-insensitive name substring");
  assert.deepEqual(searchMarkets("gold", base).map((m) => m.code), ["088691"], "market-name substring");
  // both fixture rows share the ROW() helper's hardcoded commodity_name
  // ("CRUDE OIL"), so a commodity-field search legitimately matches both.
  assert.deepEqual(searchMarkets("crude", base).map((m) => m.code).sort(), ["067651", "088691"], "commodity substring (both rows share the fixture's commodity field)");
  assert.deepEqual(searchMarkets("old week", base), [], "only searches the newest archived week");
  assert.deepEqual(searchMarkets("", base), [], "empty query returns nothing");
  assert.deepEqual(searchMarkets("nonexistent-market-xyz", base), []);
});

test("lookupMarketHistory: managed-money net + net%OI series, ascending, missing weeks honestly omitted", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "cot-"));
  writeWeek(base, "2026-05-05", parseCot([ROW("2026-05-05", "067651", "CRUDE OIL - NYMEX", "60000", "100000")], "rt"));
  writeWeek(base, "2026-05-12", parseCot([ROW("2026-05-12", "Z9", "OTHER MARKET", "1", "2")], "rt")); // no 067651 this week
  writeWeek(base, "2026-05-19", parseCot([ROW("2026-05-19", "067651", "CRUDE OIL - NYMEX", "68457", "138890")], "rt"));
  const series = lookupMarketHistory("067651", 10, base);
  assert.equal(series.length, 2, "the week without this code is omitted, never zero-filled");
  assert.deepEqual(series.map((p) => p.report_date), ["2026-05-05", "2026-05-19"], "ascending by date");
  assert.equal(series[1].m_money_net, 68457 - 138890);
  assert.equal(series[1].m_money_net_pct_oi, (68457 - 138890) / 428305);
  assert.equal(lookupMarketHistory("nope", 10, base).length, 0);
});

test("readAggregateHistory: total open interest + market count per archived week, ascending", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "cot-"));
  writeWeek(base, "2026-04-21", parseCot([
    ROW("2026-04-21", "A1", "MKT A", "1", "2"),
    ROW("2026-04-21", "B2", "MKT B", "3", "4"),
  ], "rt"));
  writeWeek(base, "2026-04-28", parseCot([ROW("2026-04-28", "A1", "MKT A", "1", "2")], "rt"));
  const trend = readAggregateHistory(10, base);
  assert.deepEqual(trend.map((t) => t.report_date), ["2026-04-21", "2026-04-28"]);
  assert.equal(trend[0].markets, 2);
  assert.equal(trend[0].total_open_interest, 428305 * 2);
  assert.equal(trend[1].markets, 1);
});
