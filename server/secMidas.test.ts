// secMidas tests — SEC MIDAS individual-security market-structure metrics
// (census build #10). Fixtures match the LIVE-VERIFIED contract (this
// session's workup): exact 19-field header, ETF quartile vs Stock decile
// rank scales, blank ranks for thin listings, the quarterly URL pattern,
// and the one-new-period-per-poll memory guard.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import zlib from "zlib";
import {
  unzipSingleCsv, parseMidas, candidateQuarters, fetchMidasPeriod,
  archiveMidasPeriod, isMidasPeriodArchived, readMidasPeriod, summarizeMidas,
  refreshMidas, latestMidas, _resetMidasForTests,
  MIDAS_SMALLCAP_MAX_RANK, MIDAS_MIN_TRADES_FOR_HIDDEN,
} from "./secMidas";

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "secmidas-"));
}

/** Minimal in-test multi-member zip builder (stored=0 or deflate=8) — the
 *  reader is dependency-free so the fixture is too. CRC left zero. */
function buildZip(members: Array<{ name: string; content: string; method: 0 | 8 }>): Buffer {
  const localBlocks: Buffer[] = [];
  const centralBlocks: Buffer[] = [];
  let offset = 0;
  for (const { name, content, method } of members) {
    const data = Buffer.from(content, "utf8");
    const payload = method === 8 ? zlib.deflateRawSync(data) : data;
    const fn = Buffer.from(name, "utf8");
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(method, 8);
    local.writeUInt32LE(payload.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(fn.length, 26);
    const localBlock = Buffer.concat([local, fn, payload]);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(method, 10);
    central.writeUInt32LE(payload.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(fn.length, 28);
    central.writeUInt32LE(offset, 42);
    const centralBlock = Buffer.concat([central, fn]);
    localBlocks.push(localBlock);
    centralBlocks.push(centralBlock);
    offset += localBlock.length;
  }
  const localAll = Buffer.concat(localBlocks);
  const centralAll = Buffer.concat(centralBlocks);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(members.length, 8);
  eocd.writeUInt16LE(members.length, 10);
  eocd.writeUInt32LE(centralAll.length, 12);
  eocd.writeUInt32LE(localAll.length, 16);
  return Buffer.concat([localAll, centralAll, eocd]);
}

const HDR = "Date,Security,Ticker,McapRank,TurnRank,VolatilityRank,PriceRank,LitVol('000),OrderVol('000),Hidden,TradesForHidden,HiddenVol('000),TradeVolForHidden('000),Cancels,LitTrades,OddLots,TradesForOddLots,OddLotVol('000),TradeVolForOddLots('000)";
function row(date: string, kind: "Stock" | "ETF", ticker: string, mcap: number | "", opts: Partial<Record<string, number>> = {}): string {
  const o = { turn: mcap === "" ? "" : mcap, vol: mcap === "" ? "" : mcap, price: mcap === "" ? "" : mcap,
    litVol: 100, orderVol: 1000, hidden: 10, tradesForHidden: 40, hiddenVol: 5, tradeVolForHidden: 50,
    cancels: 200, litTrades: 20, oddLots: 15, tradesForOddLots: 40, oddLotVol: 1, tradeVolForOddLots: 50, ...opts };
  return [date, kind, ticker, mcap, o.turn, o.vol, o.price, o.litVol, o.orderVol, o.hidden, o.tradesForHidden,
    o.hiddenVol, o.tradeVolForHidden, o.cancels, o.litTrades, o.oddLots, o.tradesForOddLots, o.oddLotVol, o.tradeVolForOddLots].join(",");
}
function midasCsv(rows: string[]): string {
  return [HDR, ...rows].join("\n") + "\n";
}

beforeEach(() => _resetMidasForTests());

test("unzipSingleCsv: finds the .csv member and skips README.txt, stored and deflated both extract", () => {
  const csv = midasCsv([row("20251001", "Stock", "AAPL", 10)]);
  const zip = buildZip([
    { name: "README.txt", content: "MAR User Data\n", method: 0 },
    { name: "q4_2025_all.csv", content: csv, method: 8 },
  ]);
  assert.equal(unzipSingleCsv(zip), csv);
  const zipStored = buildZip([{ name: "q4_2025_all.csv", content: csv, method: 0 }]);
  assert.equal(unzipSingleCsv(zipStored), csv);
  assert.throws(() => unzipSingleCsv(Buffer.from("not a zip at all")), /EOCD/);
});

test("parseMidas: ETF quartile vs Stock decile ranks, blank rank -> null, date ISO, rt stamped", () => {
  const rows = parseMidas(midasCsv([
    row("20251001", "Stock", "AAPL", 10),
    row("20251001", "ETF", "SPY", 4),
    row("20251001", "Stock", "NEWCO", ""),
  ]), "2026-07-10");
  assert.equal(rows.length, 3);
  assert.equal(rows[0].date, "2025-10-01");
  assert.equal(rows[0].kind, "Stock");
  assert.equal(rows[0].mcapRank, 10);
  assert.equal(rows[1].kind, "ETF");
  assert.equal(rows[1].mcapRank, 4);
  assert.equal(rows[2].mcapRank, null, "blank rank stored as null, never coerced");
  assert.equal(rows[0].rt, "2026-07-10");
});

test("parseMidas: wrong header refused, malformed-row-rate guard refuses corrupted files, tolerates a few bad lines", () => {
  assert.equal(parseMidas("WRONG,HEADER\nrow", "rt").length, 0, "header mismatch refused");
  // 1 good row + 99 garbage rows (99% malformed) -> refused
  const garbage = Array.from({ length: 99 }, () => "not,a,valid,row");
  assert.equal(parseMidas(midasCsv([row("20251001", "Stock", "AAPL", 10), ...garbage]), "rt").length, 0,
    "malformed-row rate over the guard refuses the whole file");
  // a single bad line among many good ones (well under the 1% guard) is tolerated (skipped, not refusing)
  const good = Array.from({ length: 200 }, (_, i) => row("20251001", "Stock", `T${i}`, 5));
  const rows = parseMidas(midasCsv([...good, "garbage,line"]), "rt");
  assert.equal(rows.length, 200, "isolated bad line (1/201, under the 1% guard) skipped, not fatal");
});

test("candidateQuarters: current quarter first, newest first, year wrap", () => {
  assert.deepEqual(candidateQuarters(Date.parse("2026-07-10T12:00:00Z"), 6),
    ["2026q3", "2026q2", "2026q1", "2025q4", "2025q3", "2025q2"]);
  assert.deepEqual(candidateQuarters(Date.parse("2026-01-15T12:00:00Z"), 3),
    ["2026q1", "2025q4", "2025q3"], "year boundary walks back cleanly");
});

test("fetchMidasPeriod: 200 parses, 404 is an honest not-yet-published empty, transport error is null", async () => {
  const csv = midasCsv([row("20251001", "Stock", "AAPL", 10)]);
  const zip = buildZip([{ name: "q4_2025_all.csv", content: csv, method: 8 }]);
  const okFetch = async (url: string) => {
    assert.ok(url.includes("individual_security_2025_q4.zip"));
    return { ok: true, status: 200, arrayBuffer: async () => zip.buffer.slice(zip.byteOffset, zip.byteOffset + zip.byteLength) };
  };
  const rows = await fetchMidasPeriod("2025q4", okFetch as any);
  assert.equal(rows!.length, 1);

  const notYet = await fetchMidasPeriod("2026q2", (async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) })) as any);
  assert.deepEqual(notYet, [], "404 = not published, honest empty, not an error");

  const dead = await fetchMidasPeriod("2026q1", (async () => { throw new Error("ECONNRESET"); }) as any);
  assert.equal(dead, null, "transport error = null, retry next poll");
});

test("archive: gz-on-write, period dedup, disk-seeded across restarts, roundtrip", () => {
  const base = tmp();
  const rows = parseMidas(midasCsv([row("20251001", "Stock", "AAPL", 10)]), "2026-07-10");
  assert.equal(archiveMidasPeriod("2025q4", rows, base), 1);
  assert.ok(fs.existsSync(path.join(base, "secmidas", "2025q4.jsonl.gz")));
  assert.equal(archiveMidasPeriod("2025q4", rows, base), 0, "dedup");
  _resetMidasForTests();
  assert.equal(isMidasPeriodArchived("2025q4", base), true, "seeded from disk");
  assert.equal(readMidasPeriod("2025q4", base)[0].ticker, "AAPL");
});

test("summarizeMidas: small-cap watch is Stock-only, McapRank<=2, trades-for-hidden floor enforced, sorted by cancel-to-trade desc", () => {
  const rows = parseMidas(midasCsv([
    // small-cap stock, high cancel/trade — should be first
    row("20251001", "Stock", "SMOL", 1, { cancels: 1000, litTrades: 10, tradesForHidden: 40 }),
    // small-cap stock, lower cancel/trade
    row("20251001", "Stock", "TINY", 2, { cancels: 200, litTrades: 20, tradesForHidden: 40 }),
    // large-cap stock, excluded by rank
    row("20251001", "Stock", "BIGCO", 10, { cancels: 5000, litTrades: 5, tradesForHidden: 40 }),
    // ETF, excluded by kind (different rank scale entirely)
    row("20251001", "ETF", "SPY", 1, { cancels: 5000, litTrades: 5, tradesForHidden: 40 }),
    // small-cap stock, excluded by thin trading floor
    row("20251001", "Stock", "GHOST", 1, { tradesForHidden: MIDAS_MIN_TRADES_FOR_HIDDEN - 1 }),
  ]), "rt");
  const s = summarizeMidas("2025q4", rows)!;
  assert.equal(s.kind_counts.stock, 4);
  assert.equal(s.kind_counts.etf, 1);
  assert.equal(s.newest_date, "2025-10-01");
  assert.deepEqual(s.smallcap_watch.map((r) => r.ticker), ["SMOL", "TINY"],
    "BIGCO excluded by rank, SPY excluded by kind, GHOST excluded by activity floor, sorted desc");
  assert.equal(s.smallcap_watch[0].cancelToTrade, 100, "1000/10");
  assert.equal(s.smallcap_max_rank, MIDAS_SMALLCAP_MAX_RANK);
});

test("refresh end-to-end: archives at most ONE new period per call, caches newest, dedup on repeat", async () => {
  const base = tmp();
  const pub: Record<string, Buffer> = {
    "2025q4": buildZip([{ name: "q4_2025_all.csv", content: midasCsv([row("20251001", "Stock", "AAPL", 10)]), method: 8 }]),
    "2025q3": buildZip([{ name: "q3_2025_all.csv", content: midasCsv([row("20250701", "Stock", "AAPL", 10)]), method: 8 }]),
  };
  const calls: string[] = [];
  const fetchImpl = async (url: string) => {
    calls.push(url);
    const m = url.match(/individual_security_(\d{4}_q[1-4])\.zip$/);
    const key = m ? m[1].replace("_q", "q") : null;
    const zip = key && pub[key];
    if (zip) return { ok: true, status: 200, arrayBuffer: async () => zip.buffer.slice(zip.byteOffset, zip.byteOffset + zip.byteLength) };
    return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
  };
  const NOW = Date.parse("2026-07-10T12:00:00Z"); // current quarter 2026q3, both pub periods are in the past
  await refreshMidas(fetchImpl as any, NOW, base);
  assert.equal(latestMidas()!.summary.period, "2025q4", "newest PUBLISHED quarter archived+cached first (2026q3/q2/q1 honestly 404)");
  assert.equal(isMidasPeriodArchived("2025q3", base), false, "only ONE new period archived per call — memory guard");
  assert.equal(calls.filter((u) => u.includes("2025_q3")).length, 0, "older period not even attempted until 2025q4 is archived");

  await refreshMidas(fetchImpl as any, NOW + 60_000, base);
  assert.equal(isMidasPeriodArchived("2025q3", base), true, "second call picks up the next backlog period");

  const before = calls.length;
  await refreshMidas(fetchImpl as any, NOW + 120_000, base);
  const dataFetches = calls.slice(before).filter((u) => /2025_q[34]/.test(u));
  assert.equal(dataFetches.length, 0, "fully-archived periods are never re-fetched");
});
