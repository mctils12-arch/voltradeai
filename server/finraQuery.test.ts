// finraQuery tests — settlement-stress datasets (census build #4 part 1).
// Fake fetch shaped exactly like the LIVE-VERIFIED contract (2026-07-07
// workup): POST-only filters, record-total header, 204-as-empty, silent
// 5000 clamp, partitions endpoint newest-first, listed-but-empty
// partitions, unordered pagination.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import zlib from "zlib";
import {
  postQueryPage, fetchPartitions, fetchPartitionTuples, fetchPartitionRows, fetchPartitionRowsMulti,
  archivePartition, isPartitionArchived, readPartition, compositeKey,
  summarizeShortInterest, summarizeThreshold,
  summarizeWeeklyBySymbol, summarizeMonthlyBySymbol, summarizeAtsBlocks,
  refreshFinraQuery, refreshFinraAts, latestFinraSi, latestFinraAts, backfillEnabled,
  _resetFinraQueryForTests, PAGE_LIMIT, SI_ADV_FLOOR, SI_POSITION_FLOOR,
} from "./finraQuery";

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "finraq-"));
}

function resp(status: number, body: any, headers: Record<string, string> = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (n: string) => headers[n.toLowerCase()] ?? null },
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  };
}

/** Fake API: partitions + EQUAL-filtered pagination with record-total.
 *  Keys in `data[dataset]` may be a single value ("2026-06-15") or a
 *  compositeKey()-joined composite ("2026-06-15__T1") — the partitions
 *  listing splits on "__" into tuple form and the data endpoint matches
 *  ALL compareFilters (not just the first), so both single- and
 *  composite-key datasets work through the same fake. */
function fakeApi(data: Record<string, Record<string, any[]>>, opts: { shuffle?: boolean } = {}) {
  const calls: string[] = [];
  const fetchImpl = async (url: string, init?: any) => {
    calls.push(url);
    const pm = url.match(/\/partitions\/group\/otcMarket\/name\/(\w+)$/);
    if (pm) {
      const ds = data[pm[1]];
      if (!ds) return resp(404, { statusCode: 404, message: "Unable to find the specified dataset. " });
      const values = Object.keys(ds).sort().reverse(); // newest first
      return resp(200, { partitionFields: ["d"], availablePartitions: values.map((v) => ({ partitions: v.split("__") })) });
    }
    const dm = url.match(/\/data\/group\/otcMarket\/name\/(\w+)$/);
    if (dm) {
      const ds = data[dm[1]];
      if (!ds) return resp(404, { statusCode: 404, message: "Unable to find the specified dataset. " });
      const body = JSON.parse(init?.body || "{}");
      const filters = body.compareFilters || [];
      const key = filters.map((f: any) => f.fieldValue).join("__");
      const rows = filters.length ? (ds[key] || []) : Object.values(ds).flat();
      if (!rows.length) return resp(204, "");
      const page = rows.slice(body.offset || 0, (body.offset || 0) + Math.min(body.limit || 1000, PAGE_LIMIT));
      const out = opts.shuffle ? [...page].reverse() : page;
      return resp(200, out, { "record-total": String(rows.length) });
    }
    return resp(404, "no route");
  };
  return { fetchImpl: fetchImpl as any, calls };
}

const SI_ROW = (symbol: string, over: Record<string, any> = {}) => ({
  symbolCode: symbol, issueName: `${symbol} Inc.`, settlementDate: "2026-06-15",
  currentShortPositionQuantity: 5_000_000, previousShortPositionQuantity: 4_000_000,
  averageDailyVolumeQuantity: 2_000_000, daysToCoverQuantity: 2.5,
  changePercent: 25.0, marketClassCode: "NYSE", ...over,
});
const TH_ROW = (symbol: string) => ({
  tradeDate: "2026-07-02", issueSymbolIdentifier: symbol, issueName: `${symbol} ADR`,
  marketClassCode: "OTC", marketCategoryDescription: "Other OTC", regShoThresholdFlag: "N", rule4320Flag: "Y",
});

beforeEach(() => _resetFinraQueryForTests());

test("postQueryPage: 204 is empty-not-error; record-total header read; error body logged and null returned", async () => {
  const { fetchImpl } = fakeApi({ thresholdList: { "2026-07-02": [] } });
  const empty = await postQueryPage("thresholdList", { limit: 5 }, fetchImpl);
  assert.deepEqual(empty, { status: 204, recordTotal: 0, rows: [] });
  const bad = await postQueryPage("notARealDataset", { limit: 5 }, fetchImpl);
  assert.equal(bad, null);
});

test("fetchPartitions: newest-first values flattened from tuples", async () => {
  const { fetchImpl } = fakeApi({ thresholdList: { "2026-07-01": [TH_ROW("A")], "2026-07-02": [TH_ROW("B")] } });
  assert.deepEqual(await fetchPartitions("thresholdList", fetchImpl), ["2026-07-02", "2026-07-01"]);
});

test("fetchPartitionRows: paginates to record-total and refuses an incomplete set", async () => {
  const many = Array.from({ length: PAGE_LIMIT + 3 }, (_, i) => SI_ROW(`S${i}`));
  const { fetchImpl, calls } = fakeApi({ consolidatedShortInterest: { "2026-06-15": many } });
  const rows = await fetchPartitionRows("consolidatedShortInterest", "settlementDate", "2026-06-15", fetchImpl);
  assert.equal(rows!.length, PAGE_LIMIT + 3);
  assert.equal(calls.filter((c) => c.includes("/data/")).length, 2, "two pages expected");

  // truncation guard A: transport failure mid-pagination → whole partition refused
  let page = 0;
  const dropsMidway = async (url: string) => {
    if (!url.includes("/data/")) return resp(404, "");
    if (page++ === 0) return resp(200, Array.from({ length: PAGE_LIMIT }, (_, i) => SI_ROW(`P${i}`)), { "record-total": String(PAGE_LIMIT + 100) });
    return resp(500, "Internal Server Error");
  };
  assert.equal(await fetchPartitionRows("consolidatedShortInterest", "settlementDate", "2026-06-15", dropsMidway as any),
    null, "mid-pagination failure must refuse the whole partition, not archive a prefix");

  // truncation guard B: record-total drift between pages → count mismatch refused
  let page2 = 0;
  const drifting = async (url: string) => {
    if (!url.includes("/data/")) return resp(404, "");
    if (page2++ === 0) return resp(200, Array.from({ length: PAGE_LIMIT }, (_, i) => SI_ROW(`Q${i}`)), { "record-total": String(PAGE_LIMIT + 5) });
    return resp(204, ""); // second page suddenly empty: rows 5000 ≠ record-total 0
  };
  assert.equal(await fetchPartitionRows("consolidatedShortInterest", "settlementDate", "2026-06-15", drifting as any),
    null, "record-total drift must be refused");
});

test("archive: SI gz-on-write, threshold plain, partition-level dedup, rt capture stamped", () => {
  const base = tmp();
  const n = archivePartition("finrashortinterest", "2026-06-15", [SI_ROW("AA")], "2026-07-07", base);
  assert.equal(n, 1);
  assert.ok(fs.existsSync(path.join(base, "finrashortinterest", "2026-06-15.jsonl.gz")));
  assert.equal(archivePartition("finrashortinterest", "2026-06-15", [SI_ROW("AA")], "2026-07-07", base), 0, "dedup");
  const back = readPartition("finrashortinterest", "2026-06-15", base);
  assert.equal(back[0].rt, "2026-07-07");

  archivePartition("finrathreshold", "2026-07-02", [TH_ROW("CHLSY")], "2026-07-07", base);
  assert.ok(fs.existsSync(path.join(base, "finrathreshold", "2026-07-02.jsonl")), "threshold days stay plain");
});

test("dedup Set seeds from disk across restarts", () => {
  const base = tmp();
  const dir = path.join(base, "finrashortinterest");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-05-29.jsonl.gz"), zlib.gzipSync('{"symbolCode":"OLD"}\n'));
  _resetFinraQueryForTests();
  assert.equal(isPartitionArchived("finrashortinterest", "2026-05-29", base), true);
});

test("summarizeShortInterest: floors stated and enforced, sorts honest, null on empty", () => {
  assert.equal(summarizeShortInterest([]), null);
  const rows = [
    SI_ROW("BIG", { daysToCoverQuantity: 9.9 }),
    SI_ROW("ILLIQ", { daysToCoverQuantity: 50, averageDailyVolumeQuantity: SI_ADV_FLOOR - 1 }),
    SI_ROW("TINY", { changePercent: 900, currentShortPositionQuantity: SI_POSITION_FLOOR - 1 }),
    SI_ROW("MOVER", { changePercent: 80.5 }),
    // the live 2026-07-07 artifact: astronomical % from a near-zero base
    SI_ROW("ZEROBASE", { changePercent: 259_847_500, previousShortPositionQuantity: 4 }),
  ];
  const s = summarizeShortInterest(rows)!;
  assert.equal(s.settlement_date, "2026-06-15");
  assert.equal(s.records, 5);
  assert.equal(s.top_days_to_cover[0].symbol, "BIG");
  assert.ok(!s.top_days_to_cover.some((r) => r.symbol === "ILLIQ"), "ADV floor keeps illiquid junk out");
  assert.equal(s.top_change_pct[0].symbol, "MOVER");
  assert.ok(!s.top_change_pct.some((r) => r.symbol === "TINY"), "position floor enforced");
  assert.ok(!s.top_change_pct.some((r) => r.symbol === "ZEROBASE"), "near-zero-base % explosion excluded by prev floor");
  assert.equal(s.adv_floor, SI_ADV_FLOOR);
});

test("refresh end-to-end: archives new partitions, caches newest summaries, second run no-ops", async () => {
  const base = tmp();
  const api = fakeApi({
    consolidatedShortInterest: {
      "2026-06-15": [SI_ROW("GME"), SI_ROW("AMC")],
      "2026-05-29": [SI_ROW("OLD", { settlementDate: "2026-05-29" })],
    },
    thresholdList: { "2026-07-02": [TH_ROW("CHLSY")], "2026-07-01": [TH_ROW("AACAY")] },
  });
  await refreshFinraQuery(api.fetchImpl, Date.parse("2026-07-07T02:00:00Z"), base);
  const c = latestFinraSi()!;
  assert.equal(c.si!.settlement_date, "2026-06-15");
  assert.equal(c.si!.records, 2);
  assert.equal(c.threshold!.trade_date, "2026-07-02");
  assert.equal(c.threshold!.count, 1);
  assert.ok(isPartitionArchived("finrashortinterest", "2026-05-29", base), "window covers older partitions too");

  const callsBefore = api.calls.length;
  await refreshFinraQuery(api.fetchImpl, Date.parse("2026-07-07T08:00:00Z"), base);
  const dataCalls = api.calls.slice(callsBefore).filter((u) => u.includes("/data/"));
  assert.equal(dataCalls.length, 0, "everything archived — only partition listings on the second run");
});

test("refresh survives transport failure with honest warming state (no cache fabrication)", async () => {
  const dead = async () => { throw new Error("ECONNRESET"); };
  await refreshFinraQuery(dead as any, Date.parse("2026-07-07T02:00:00Z"), tmp());
  assert.equal(latestFinraSi(), null);
});

test("backfill is env-gated OFF by default (R8 crash-loop lesson)", () => {
  assert.equal(backfillEnabled({} as any), false);
  assert.equal(backfillEnabled({ FINRA_QUERY_BACKFILL: "1" } as any), true);
});

// ── Part 2: ATS venue summaries (composite-key partitions) ──────────────────

const WK_ROW = (over: Record<string, any> = {}) => ({
  weekStartDate: "2026-06-15", tierIdentifier: "T1", summaryTypeCode: "OTC_W_SMBL",
  issueSymbolIdentifier: "AAPL", issueName: "Apple Inc.",
  totalWeeklyShareQuantity: 1_000_000, totalWeeklyTradeCount: 500, totalNotionalSum: 50_000_000,
  ...over,
});
const MO_ROW = (over: Record<string, any> = {}) => ({
  monthStartDate: "2026-05-01", tierIdentifier: "NMS", summaryTypeCode: "OTC_M_SMBL",
  issueSymbolIdentifier: "AA", issueName: "Alcoa Corporation",
  totalMonthlyShareQuantity: 2_000_000, totalMonthlyTradeCount: 900, totalNotionalSum: 60_000_000,
  ...over,
});
const BLK_ROW = (mpid: string, over: Record<string, any> = {}) => ({
  monthStartDate: "2026-05-01", MPID: mpid, marketParticipantName: `${mpid} VENUE`,
  ATSBlockQuantity: 100_000, ATSBlockCount: 50, averageBlockSize: 2000,
  ATSSharePercent: 0.5, ATSBlockShareRank: 1,
  ...over,
});

test("compositeKey joins parts with a double-underscore", () => {
  assert.equal(compositeKey(["2026-06-15", "T1"]), "2026-06-15__T1");
  assert.equal(compositeKey(["2026-05-01"]), "2026-05-01");
});

test("fetchPartitionTuples: composite keys kept as full tuples, newest first", async () => {
  const { fetchImpl } = fakeApi({
    weeklySummary: { "2026-06-15__T1": [WK_ROW()], "2026-06-08__T1": [WK_ROW({ weekStartDate: "2026-06-08" })] },
  });
  assert.deepEqual(await fetchPartitionTuples("weeklySummary", fetchImpl),
    [["2026-06-15", "T1"], ["2026-06-08", "T1"]]);
});

test("fetchPartitionRowsMulti: matches on ALL filters together (not either alone), respects a raised page cap", async () => {
  const manyT1 = Array.from({ length: PAGE_LIMIT * 3 + 1 }, () => WK_ROW());
  const { fetchImpl, calls } = fakeApi({
    weeklySummary: {
      "2026-06-15__T1": manyT1,
      "2026-06-15__T2": [WK_ROW({ tierIdentifier: "T2" })],
      "2026-06-08__T1": [WK_ROW({ weekStartDate: "2026-06-08" })],
    },
  });
  const t1 = await fetchPartitionRowsMulti("weeklySummary",
    [{ field: "weekStartDate", value: "2026-06-15" }, { field: "tierIdentifier", value: "T1" }], fetchImpl, 50);
  assert.equal(t1!.length, PAGE_LIMIT * 3 + 1, "raised maxPages covers a partition beyond part-1's 60k cap");
  assert.equal(calls.filter((c) => c.includes("/data/")).length, 4, "4 pages for 15001 rows at 5000/page");

  // a single-field query for a value that's only ever paired with OTHER
  // second-field values ("T1" only ever appears with 2026-06-15 or
  // 2026-06-08, never bare) must not accidentally match either partition.
  const bare = await fetchPartitionRowsMulti("weeklySummary", [{ field: "tierIdentifier", value: "T1" }], fetchImpl);
  assert.deepEqual(bare, [], "a lone filter value that never appears as a standalone key matches nothing, not everything");
});

test("summarizeWeeklyBySymbol: only *_SMBL rows ranked, FIRM/SMBL_FIRM rows excluded but counted in composition", () => {
  const rows = [
    WK_ROW({ issueSymbolIdentifier: "AAPL", totalWeeklyShareQuantity: 5_000_000, summaryTypeCode: "OTC_W_SMBL" }),
    WK_ROW({ issueSymbolIdentifier: "MSFT", totalWeeklyShareQuantity: 9_000_000, summaryTypeCode: "ATS_W_SMBL" }),
    WK_ROW({ issueSymbolIdentifier: "TSLA", totalWeeklyShareQuantity: 99_000_000, summaryTypeCode: "OTC_W_SMBL_FIRM", marketParticipantName: "SOMEFIRM" }),
    WK_ROW({ issueSymbolIdentifier: null, marketParticipantName: "BIGFIRM", summaryTypeCode: "OTC_W_FIRM" }),
  ];
  assert.equal(summarizeWeeklyBySymbol([], ["T1"]), null);
  const s = summarizeWeeklyBySymbol(rows, ["T1", "T2"])!;
  assert.equal(s.week_start, "2026-06-15");
  assert.deepEqual(s.tiers_covered, ["T1", "T2"]);
  assert.equal(s.records, 4);
  assert.equal(s.composition["OTC_W_SMBL_FIRM"], 1, "FIRM-blended rows counted in composition, not the leaderboard");
  assert.ok(!s.top_otc_by_symbol.some((r) => r.symbol === "TSLA"), "SMBL_FIRM row excluded from the ranked leaderboard");
  assert.equal(s.top_otc_by_symbol[0].symbol, "AAPL");
  assert.equal(s.top_ats_by_symbol![0].symbol, "MSFT");
});

test("summarizeMonthlyBySymbol: OTC_M_SMBL only, null on empty", () => {
  assert.equal(summarizeMonthlyBySymbol([], ["NMS"]), null);
  const rows = [MO_ROW(), MO_ROW({ issueSymbolIdentifier: "MSFT", totalMonthlyShareQuantity: 1, summaryTypeCode: "OTC_M_FIRM", marketParticipantName: "X" })];
  const s = summarizeMonthlyBySymbol(rows, ["NMS"])!;
  assert.equal(s.month_start, "2026-05-01");
  assert.equal(s.records, 2);
  assert.equal(s.top_otc_by_symbol.length, 1, "the FIRM row is excluded from ranking");
  assert.equal(s.top_otc_by_symbol[0].symbol, "AA");
});

test("summarizeAtsBlocks: ranks venues by block quantity, null on empty", () => {
  assert.equal(summarizeAtsBlocks([]), null);
  const rows = [BLK_ROW("SMALL", { ATSBlockQuantity: 1000 }), BLK_ROW("BIG", { ATSBlockQuantity: 9_000_000 })];
  const s = summarizeAtsBlocks(rows)!;
  assert.equal(s.month_start, "2026-05-01");
  assert.equal(s.records, 2);
  assert.equal(s.top_venues_by_block_volume[0].mpid, "BIG");
});

test("refreshFinraAts end-to-end: archives composite partitions across tiers, combines archived tiers into one weekly reading, blocks stay single-key", async () => {
  const base = tmp();
  const api = fakeApi({
    weeklySummary: {
      "2026-06-15__T1": [WK_ROW({ issueSymbolIdentifier: "AAPL" })],
      "2026-06-15__T2": [WK_ROW({ tierIdentifier: "T2", issueSymbolIdentifier: "MSFT", summaryTypeCode: "ATS_W_SMBL" })],
      // OTCE/NA have no rows for this week in the fixture (mirrors the
      // live 2026-07-07 probe, where only T1 was populated) -> the fake
      // 204s those two composite keys; refreshComposite must skip them as
      // honestly empty, not abort the whole refresh — assert below that
      // T1+T2 still combine cleanly.
    },
    monthlySummary: { "2026-05-01__NMS": [MO_ROW()] },
    blocksSummary: { "2026-05-01": [BLK_ROW("ONE")] },
  });
  await refreshFinraAts(api.fetchImpl, Date.parse("2026-07-08T02:00:00Z"), base, 1, 1);
  const c = latestFinraAts()!;
  assert.equal(c.weekly!.week_start, "2026-06-15");
  assert.deepEqual(c.weekly!.tiers_covered.sort(), ["T1", "T2"], "only the archived tiers, honestly");
  assert.equal(c.weekly!.records, 2, "T1 + T2 rows combined");
  assert.equal(c.monthly!.month_start, "2026-05-01");
  assert.equal(c.blocks!.month_start, "2026-05-01");
  assert.equal(c.blocks!.top_venues_by_block_volume[0].mpid, "ONE");
  assert.ok(isPartitionArchived("finraweekly", compositeKey(["2026-06-15", "T1"]), base));
  assert.ok(isPartitionArchived("finraweekly", compositeKey(["2026-06-15", "T2"]), base));
  assert.ok(fs.existsSync(path.join(base, "finrablocks", "2026-05-01.jsonl")), "blocks stay plain, single-key filename");

  // second run: the 3 rows-having partitions (weekly T1/T2, monthly NMS,
  // blocks) are archived and never re-fetched; the 3 EMPTY tier/venue
  // combos (weekly OTCE+NA, monthly OTCE) were deliberately never marked
  // archived (matches the existing SI/threshold "204 is not a done-marker"
  // pattern elsewhere in this file — an empty partition can still fill in
  // later) so they're honestly re-checked every cycle. Assert both halves.
  const callsBefore = api.calls.length;
  await refreshFinraAts(api.fetchImpl, Date.parse("2026-07-08T08:00:00Z"), base, 1, 1);
  const dataCalls = api.calls.slice(callsBefore).filter((u) => u.includes("/data/"));
  assert.equal(dataCalls.length, 3, "only the still-empty tier/venue combos are re-polled");
  assert.equal(dataCalls.filter((u) => u.includes("weeklySummary")).length, 2, "weekly OTCE + NA");
  assert.equal(dataCalls.filter((u) => u.includes("monthlySummary")).length, 1, "monthly OTCE");
  assert.equal(dataCalls.filter((u) => u.includes("blocksSummary")).length, 0, "blocks already archived, not re-checked");
});

test("refreshFinraAts: transport failure leaves the cache honestly null, no fabrication", async () => {
  const dead = async () => { throw new Error("ECONNRESET"); };
  await refreshFinraAts(dead as any, Date.parse("2026-07-08T02:00:00Z"), tmp());
  assert.equal(latestFinraAts(), null);
});
