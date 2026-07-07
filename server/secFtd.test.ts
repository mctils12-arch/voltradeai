// secFtd tests — SEC fails-to-deliver (census build #6). Fixtures match
// the LIVE-VERIFIED contract (2026-07-07 workup): exact header, two-line
// trailer checksum, PRICE ".", CINS cusips, CRLF-era files, the
// non-uniform URL chain (202605a at /files/data/other/ TODAY), and the
// verified 1-14/15-EOM half-month boundary.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import zlib from "zlib";
import {
  unzipSingleText, parseFtd, candidatePeriods, fetchFtdPeriod,
  archiveFtdPeriod, isFtdPeriodArchived, readFtdPeriod, summarizeFtd,
  refreshFtd, latestFtd, _resetFtdForTests, FTD_QTY_FLOOR,
} from "./secFtd";

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "secftd-"));
}

/** Minimal in-test zip builder (stored=0 or deflate=8) — the reader is
 *  dependency-free so the fixture is too. CRC left zero (not validated). */
function buildZip(name: string, content: string, method: 0 | 8): Buffer {
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
  central.writeUInt32LE(0, 42); // local header offset
  const centralBlock = Buffer.concat([central, fn]);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(1, 8);
  eocd.writeUInt16LE(1, 10);
  eocd.writeUInt32LE(centralBlock.length, 12);
  eocd.writeUInt32LE(localBlock.length, 16);
  return Buffer.concat([localBlock, centralBlock, eocd]);
}

const HDR = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE";
function ftdText(rows: Array<[string, string, string, number, string, string]>, opts: { count?: number; qty?: number; crlf?: boolean } = {}): string {
  const body = rows.map((r) => r.join("|"));
  const count = opts.count ?? rows.length;
  const qty = opts.qty ?? rows.reduce((a, r) => a + r[3], 0);
  const lines = [HDR, ...body, `Trailer record count ${count}`, `Trailer total quantity of shares ${qty}`];
  return lines.join(opts.crlf ? "\r\n" : "\n") + (opts.crlf ? "\r\n" : "\n");
}

beforeEach(() => _resetFtdForTests());

test("unzipSingleText: stored and deflated single-member zips both extract", () => {
  const text = ftdText([["20260601", "B5950S113", "MDXH", 9998, "MDXHEALTH SA SHS NEW(BELGIUM) ", "0.76"]]);
  assert.equal(unzipSingleText(buildZip("cnsfails202606a.txt", text, 0)), text);
  assert.equal(unzipSingleText(buildZip("cnsfails202606a.txt", text, 8)), text);
  assert.throws(() => unzipSingleText(Buffer.from("not a zip at all, definitely")), /EOCD/);
});

test("parseFtd: CINS cusips kept, PRICE '.' -> null, DESCRIPTION rstripped, CRLF era ok, date ISO", () => {
  const rows = parseFtd(ftdText([
    ["20260601", "B5950S113", "MDXH", 9998, "MDXHEALTH SA SHS NEW(BELGIUM) ", "0.76"],
    ["20260612", "98985Y108", "ZYME", 12377, "ZYMEWORKS INC COM(DE)", "."],
  ], { crlf: true }), "2026-07-07");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].date, "2026-06-01");
  assert.equal(rows[0].cusip, "B5950S113", "CINS letter prefix preserved");
  assert.equal(rows[0].name, "MDXHEALTH SA SHS NEW(BELGIUM)", "trailing spaces stripped");
  assert.equal(rows[0].price, 0.76);
  assert.equal(rows[1].price, null, "literal '.' is null, never zero");
  assert.equal(rows[1].rt, "2026-07-07");
});

test("parseFtd trailer checksums: count mismatch refused, qty mismatch refused, wrong header refused", () => {
  const good: Array<[string, string, string, number, string, string]> =
    [["20260601", "037833100", "AAPL", 500000, "APPLE INC", "212.44"]];
  assert.equal(parseFtd(ftdText(good, { count: 99 }), "rt").length, 0, "count mismatch");
  assert.equal(parseFtd(ftdText(good, { qty: 1 }), "rt").length, 0, "qty checksum mismatch");
  assert.equal(parseFtd("WRONG|HEADER\nrow", "rt").length, 0, "header change alarms");
  assert.equal(parseFtd(ftdText(good), "rt").length, 1, "clean file parses");
});

test("candidatePeriods: verified 1-14/15-EOM boundary, newest first, year wrap", () => {
  assert.deepEqual(candidatePeriods(Date.parse("2026-07-07T12:00:00Z"), 5),
    ["202607a", "202606b", "202606a", "202605b", "202605a"]);
  assert.deepEqual(candidatePeriods(Date.parse("2026-07-15T12:00:00Z"), 2),
    ["202607b", "202607a"], "the 15th belongs to 'b'");
  assert.deepEqual(candidatePeriods(Date.parse("2026-07-14T12:00:00Z"), 1),
    ["202607a"], "the 14th still belongs to 'a'");
  assert.deepEqual(candidatePeriods(Date.parse("2026-01-05T12:00:00Z"), 3),
    ["202601a", "202512b", "202512a"], "year boundary walks back cleanly");
});

test("fetchFtdPeriod: URL fallback chain finds files the standard path 404s (the live 202605a case)", async () => {
  const text = ftdText([["20260514", "037833100", "AAPL", 500000, "APPLE INC", "212.44"]]);
  const zip = buildZip("cnsfails202605a.txt", text, 8);
  const urls: string[] = [];
  const fetchImpl = async (url: string) => {
    urls.push(url);
    if (url === "https://www.sec.gov/files/data/other/fails-deliver-data/cnsfails202605a.zip") {
      return { ok: true, status: 200, arrayBuffer: async () => zip.buffer.slice(zip.byteOffset, zip.byteOffset + zip.byteLength) };
    }
    return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
  };
  const rows = await fetchFtdPeriod("202605a", fetchImpl as any, Date.parse("2026-07-07T00:00:00Z"));
  assert.equal(rows!.length, 1);
  assert.ok(urls[0].includes("/files/data/fails-deliver-data/"), "standard path tried first");

  // all-404 = honestly not published ([]), not an error (null)
  const notYet = await fetchFtdPeriod("202606b", (async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) })) as any);
  assert.deepEqual(notYet, []);
  // transport failure = null (retry next poll)
  const dead = await fetchFtdPeriod("202606a", (async () => { throw new Error("ECONNRESET"); }) as any);
  assert.equal(dead, null);
});

test("archive: gz-on-write, period dedup, disk-seeded across restarts, roundtrip", () => {
  const base = tmp();
  const rows = parseFtd(ftdText([["20260601", "037833100", "AAPL", 500000, "APPLE INC", "212.44"]]), "2026-07-07");
  assert.equal(archiveFtdPeriod("202606a", rows, base), 1);
  assert.ok(fs.existsSync(path.join(base, "secftd", "202606a.jsonl.gz")));
  assert.equal(archiveFtdPeriod("202606a", rows, base), 0, "dedup");
  _resetFtdForTests();
  assert.equal(isFtdPeriodArchived("202606a", base), true, "seeded from disk");
  assert.equal(readFtdPeriod("202606a", base)[0].symbol, "AAPL");
});

test("summary: newest settlement date only, qty floor stated and enforced", () => {
  const rows = parseFtd(ftdText([
    ["20260601", "037833100", "AAPL", FTD_QTY_FLOOR + 1, "APPLE INC", "212.44"],
    ["20260612", "98985Y108", "ZYME", FTD_QTY_FLOOR + 5, "ZYMEWORKS INC COM(DE)", "22.97"],
    ["20260612", "12345X108", "TINY", FTD_QTY_FLOOR - 1, "TINY CO", "1.00"],
  ]), "rt");
  const s = summarizeFtd("202606a", rows)!;
  assert.equal(s.newest_date, "2026-06-12");
  assert.equal(s.settlement_dates, 2);
  assert.equal(s.rows, 3);
  assert.deepEqual(s.top_fails.map((r) => r.symbol), ["ZYME"], "older-date and sub-floor rows excluded");
  assert.equal(s.qty_floor, FTD_QTY_FLOOR);
});

test("refresh end-to-end: archives published periods, caches newest, second run makes no data fetches", async () => {
  const base = tmp();
  const pub: Record<string, Buffer> = {
    "202606a": buildZip("cnsfails202606a.txt", ftdText([["20260612", "98985Y108", "ZYME", 500000, "ZYMEWORKS INC COM(DE)", "22.97"]]), 8),
    "202605b": buildZip("cnsfails202605b.txt", ftdText([["20260529", "037833100", "AAPL", 700000, "APPLE INC", "210.00"]]), 8),
  };
  const calls: string[] = [];
  const fetchImpl = async (url: string) => {
    calls.push(url);
    const m = url.match(/cnsfails(\d{6}[ab])(?:_0)?\.zip$/);
    const zip = m && pub[m[1]];
    if (zip && url.includes("/files/data/fails-deliver-data/")) {
      return { ok: true, status: 200, arrayBuffer: async () => zip.buffer.slice(zip.byteOffset, zip.byteOffset + zip.byteLength) };
    }
    return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
  };
  const NOW = Date.parse("2026-07-07T12:00:00Z");
  await refreshFtd(fetchImpl as any, NOW, base);
  assert.equal(latestFtd()!.summary.period, "202606a", "newest PUBLISHED period cached (202607a/202606b honestly absent)");
  assert.ok(isFtdPeriodArchived("202605b", base));
  const before = calls.length;
  await refreshFtd(fetchImpl as any, NOW + 3600_000, base);
  const dataFetches = calls.slice(before).filter((u) => /cnsfails(202606a|202605b)/.test(u));
  assert.equal(dataFetches.length, 0, "archived periods are never re-fetched");
});
