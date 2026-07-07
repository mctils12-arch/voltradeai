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
  postQueryPage, fetchPartitions, fetchPartitionRows,
  archivePartition, isPartitionArchived, readPartition,
  summarizeShortInterest, summarizeThreshold,
  refreshFinraQuery, latestFinraSi, backfillEnabled,
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

/** Fake API: partitions + EQUAL-filtered pagination with record-total. */
function fakeApi(data: Record<string, Record<string, any[]>>, opts: { shuffle?: boolean } = {}) {
  const calls: string[] = [];
  const fetchImpl = async (url: string, init?: any) => {
    calls.push(url);
    const pm = url.match(/\/partitions\/group\/otcMarket\/name\/(\w+)$/);
    if (pm) {
      const ds = data[pm[1]];
      if (!ds) return resp(404, { statusCode: 404, message: "Unable to find the specified dataset. " });
      const values = Object.keys(ds).sort().reverse(); // newest first
      return resp(200, { partitionFields: ["d"], availablePartitions: values.map((v) => ({ partitions: [v] })) });
    }
    const dm = url.match(/\/data\/group\/otcMarket\/name\/(\w+)$/);
    if (dm) {
      const ds = data[dm[1]];
      if (!ds) return resp(404, { statusCode: 404, message: "Unable to find the specified dataset. " });
      const body = JSON.parse(init?.body || "{}");
      const filter = (body.compareFilters || [])[0];
      const rows = filter ? (ds[filter.fieldValue] || []) : Object.values(ds).flat();
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
