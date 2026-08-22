// DTCC SBSDR equity swap stream (DATA CENSUS #11). Fixtures use the VERBATIM
// header + a handful of rows from the 2026-08-22 live probe (SEC_CUMULATIVE_
// EQUITIES_2026_08_21.zip), incl. a China-domiciled ISIN row (must be
// dropped — US-only volume-budget scope) and a basket row (semicolon ISIN
// list — must archive but not be graded by the checksum gate).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { Readable } from "node:stream";
import {
  parseCsvLine, parseHeader, REQUIRED_COLUMNS,
  isinCheckDigitValid, cusipCheckDigitValid, cusipFromUsIsin,
  isUsUnderlier, isBasketField, parseDtccLine,
  tryParseLocalHeader, streamDtccZip, dtccUrl,
  archiveNewRows, _resetDtccForTests,
} from "./dtccSwaps";

const HEADER_COLS = [...REQUIRED_COLUMNS, "Some Future Column"];
const HEADER_LINE = HEADER_COLS.map((c) => `"${c}"`).join(",");

function row(diss: string, id: string, src: string, name: string, notional = "28,000,000", ccy = "USD") {
  const byName: Record<string, string> = {
    "Dissemination Identifier": diss,
    "Original Dissemination Identifier": diss,
    "Action type": "NEWT",
    "Event timestamp": "2026-08-21T12:33:36Z",
    "Effective Date": "2026-08-21",
    "Asset Class": "EQ",
    "Notional amount-Leg 1": notional,
    "Notional currency-Leg 1": ccy,
    "Underlier ID-Leg 1": id,
    "Underlier ID source-Leg 1": src,
    "UPI Underlier Name": name,
    "Some Future Column": "x",
  };
  return HEADER_COLS.map((c) => `"${byName[c]}"`).join(",");
}

test("parseCsvLine: quoted commas (thousands-formatted notional) and doubled-quote escaping", () => {
  assert.deepEqual(parseCsvLine('"4895571859001896201","28,000,000","He said ""hi"""'),
    ["4895571859001896201", "28,000,000", 'He said "hi"']);
});

test("parseHeader: builds index map, throws on missing required column (schema-drift guard)", () => {
  const idx = parseHeader(HEADER_LINE);
  assert.equal(idx["Underlier ID-Leg 1"], HEADER_COLS.indexOf("Underlier ID-Leg 1"));
  assert.throws(() => parseHeader('"Dissemination Identifier","Action type"'), /schema drift/);
});

test("isinCheckDigitValid: AAPL/TSLA real ISINs pass, single mutated digit fails", () => {
  assert.equal(isinCheckDigitValid("US0378331005"), true, "AAPL");
  assert.equal(isinCheckDigitValid("US88160R1014"), true, "TSLA");
  assert.equal(isinCheckDigitValid("US0378331006"), false, "mutated check digit");
  assert.equal(isinCheckDigitValid("not-an-isin"), false);
});

test("cusipCheckDigitValid: AAPL/TSLA real CUSIPs (incl. a letter-containing one) pass; mutation fails", () => {
  assert.equal(cusipCheckDigitValid("037833100"), true, "AAPL, all-digit");
  assert.equal(cusipCheckDigitValid("88160R101"), true, "TSLA, letter in base");
  assert.equal(cusipCheckDigitValid("037833101"), false, "mutated check digit");
});

test("cusipFromUsIsin: extracts chars 3-11, matches the vendor CUSIP for the same security", () => {
  assert.equal(cusipFromUsIsin("US0378331005"), "037833100");
  assert.equal(cusipFromUsIsin("US88160R1014"), "88160R101");
  assert.equal(cusipFromUsIsin("DE0007236101"), null, "non-US ISIN");
});

test("isUsUnderlier: CUSIP source or US-prefixed ISIN only; RIC/non-US ISIN excluded (volume-budget scope)", () => {
  assert.equal(isUsUnderlier("037833100", "CUSIP"), true);
  assert.equal(isUsUnderlier("US0378331005", "ISIN"), true);
  assert.equal(isUsUnderlier("CNE100003662", "ISIN"), false, "China A-share, out of scope");
  assert.equal(isUsUnderlier("AAPL.OQ", "RIC"), false);
});

test("isBasketField: detects the semicolon-delimited multi-ISIN list in EITHER the source or the id field", () => {
  assert.equal(isBasketField("US1;US2;US3", "ISIN;ISIN;ISIN"), true, "multi-value source (the common encoding)");
  assert.equal(isBasketField("US1;US2;US3", "ISIN"), true, "singular source, multi-value id (the rarer encoding found live 2026-08-22)");
  assert.equal(isBasketField("US0378331005", "ISIN"), false);
});

test("parseDtccLine: filters non-US rows, drops basket rows (ambiguous US-ness), parses notional, drops on column-count mismatch", () => {
  const idx = parseHeader(HEADER_LINE);
  const n = HEADER_COLS.length;
  const usRow = parseDtccLine(row("1", "US0378331005", "ISIN", "APPLE INC"), idx, n);
  assert.ok(usRow);
  assert.equal(usRow!.underlierId, "US0378331005");
  assert.equal(usRow!.notionalAmountLeg1, 28_000_000);

  const cnRow = parseDtccLine(row("2", "CNE100003662", "ISIN", "CATL"), idx, n);
  assert.equal(cnRow, null, "non-US underlier is dropped");

  const basketRow = parseDtccLine(row("3", "US1;US2;US3", "ISIN;ISIN;ISIN", "Basket"), idx, n);
  assert.equal(basketRow, null, "basket rows are out of v1 scope, dropped rather than mislabeled");
  assert.equal(isBasketField("US1;US2;US3", "ISIN;ISIN;ISIN"), true, "still detectable for the gate-1 count");

  const oddBasketRow = parseDtccLine(row("3b", "US1;US2;US3", "ISIN", "Basket"), idx, n);
  assert.equal(oddBasketRow, null, "the singular-source/multi-value-id basket encoding is also dropped");

  const truncated = "\"1\",\"2\"";
  assert.equal(parseDtccLine(truncated, idx, n), null, "wrong column count is dropped, not mis-shifted");
});

test("parseDtccLine: blank notional (Dodd-Frank masking) parses as null, never a fabricated 0", () => {
  const idx = parseHeader(HEADER_LINE);
  const r = parseDtccLine(row("4", "US0378331005", "ISIN", "APPLE INC", ""), idx, HEADER_COLS.length);
  assert.equal(r!.notionalAmountLeg1, null);
});

// ── streaming zip reader ────────────────────────────────────────────────

function buildZip(csvText: string, fileName = "SEC_CUMULATIVE_EQUITIES_2026_08_21.csv"): Buffer {
  const nameBuf = Buffer.from(fileName, "utf8");
  const dataBuf = Buffer.from(csvText, "utf8");
  const compressed = zlib.deflateRawSync(dataBuf);
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(20, 4);           // version needed
  header.writeUInt16LE(0, 6);            // flags — no data descriptor
  header.writeUInt16LE(8, 8);            // method = deflate
  header.writeUInt16LE(0, 10);           // mod time
  header.writeUInt16LE(0, 12);           // mod date
  header.writeUInt32LE(0, 16);           // crc32 (unused by our reader)
  header.writeUInt32LE(compressed.length, 18);
  header.writeUInt32LE(dataBuf.length, 22);
  header.writeUInt16LE(nameBuf.length, 26);
  header.writeUInt16LE(0, 28);           // extra field length
  // trailing garbage (stand-in for a central directory) proves the reader
  // stops at compressedSize and never tries to consume it.
  return Buffer.concat([header, nameBuf, compressed, Buffer.from("TRAILING-CENTRAL-DIRECTORY-JUNK")]);
}

function fakeFetch(buf: Buffer, ok = true, status = 200) {
  return async () => ({
    ok, status,
    body: Readable.toWeb ? Readable.toWeb(Readable.from([buf])) : Readable.from([buf]),
  });
}

test("tryParseLocalHeader: parses a real local file header, rejects ZIP64/streamed sentinels", () => {
  const zip = buildZip(HEADER_LINE + "\n" + row("1", "US0378331005", "ISIN", "APPLE INC"));
  const h = tryParseLocalHeader(zip);
  assert.ok(h);
  assert.equal(h!.method, 8);
  assert.equal(tryParseLocalHeader(Buffer.alloc(10)), null, "incomplete buffer returns null, not a throw");

  const zip64 = Buffer.from(zip);
  zip64.writeUInt32LE(0xffffffff, 18);
  assert.throws(() => tryParseLocalHeader(zip64), /ZIP64/);
});

test("streamDtccZip: end-to-end through a real deflate+zip-framed fixture, no full-body buffering required by the caller", async () => {
  const csv = [HEADER_LINE,
    row("100", "US0378331005", "ISIN", "APPLE INC"),
    row("101", "CNE100003662", "ISIN", "CATL"),
    row("102", "US88160R1014", "ISIN", "TESLA INC"),
  ].join("\n") + "\n";
  const zip = buildZip(csv);
  const lines: string[] = [];
  const result = await streamDtccZip(dtccUrl("2026_08_21"), (line) => lines.push(line), fakeFetch(zip));
  assert.equal(result.ok, true);
  assert.equal(result.lines, 4, "header + 3 data rows");
  assert.equal(lines[0], HEADER_LINE);
  assert.match(lines[1], /US0378331005/);
  assert.match(lines[2], /CNE100003662/);
  assert.match(lines[3], /US88160R1014/);
});

test("streamDtccZip: non-ok fetch reported honestly, never thrown as a parse error", async () => {
  const result = await streamDtccZip(dtccUrl("2026_08_21"), () => {}, fakeFetch(Buffer.alloc(0), false, 404));
  assert.equal(result.ok, false);
  assert.equal(result.status, 404);
});

// ── archive / dedup ──────────────────────────────────────────────────────

test("archiveNewRows: dedups by Dissemination Identifier across days, seeded from disk on first call", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-archive-"));
  _resetDtccForTests();
  const idx = parseHeader(HEADER_LINE);
  const n = HEADER_COLS.length;
  const day1 = [parseDtccLine(row("1", "US0378331005", "ISIN", "APPLE INC"), idx, n)!,
                parseDtccLine(row("2", "US88160R1014", "ISIN", "TESLA INC"), idx, n)!];
  const r1 = archiveNewRows(day1, "2026-08-21", dir);
  assert.equal(r1.newRows, 2);
  assert.equal(r1.seenTotal, 2);

  // Day 2: the upstream cumulative file re-serves both prior IDs plus one new one.
  const day2 = [parseDtccLine(row("1", "US0378331005", "ISIN", "APPLE INC"), idx, n)!,
                parseDtccLine(row("2", "US88160R1014", "ISIN", "TESLA INC"), idx, n)!,
                parseDtccLine(row("3", "037833100", "CUSIP", "APPLE INC"), idx, n)!];
  const r2 = archiveNewRows(day2, "2026-08-22", dir);
  assert.equal(r2.newRows, 1, "only the genuinely new dissemination id is written");
  assert.equal(r2.seenTotal, 3);

  // Cold-start reload (simulates a process restart) must re-seed from disk.
  _resetDtccForTests();
  const r3 = archiveNewRows(day2, "2026-08-23", dir);
  assert.equal(r3.newRows, 0, "already-archived ids stay deduped across a fresh in-memory Set");

  fs.rmSync(dir, { recursive: true, force: true });
});
