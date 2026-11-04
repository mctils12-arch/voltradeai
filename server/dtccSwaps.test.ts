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
  archiveNewRows, _resetDtccForTests, loadSeenIds,
  refreshDtccSwaps, latestDtcc, topNotionalRows, readArchivedDtccRows,
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

// ── topNotionalRows (the /data browsing view's row set) ─────────────────

test("topNotionalRows: sorts descending by notional, masked (null) rows sort last rather than as zero", () => {
  const idx = parseHeader(HEADER_LINE);
  const n = HEADER_COLS.length;
  const small = parseDtccLine(row("1", "US0378331005", "ISIN", "APPLE INC", "1,000,000"), idx, n)!;
  const large = parseDtccLine(row("2", "US88160R1014", "ISIN", "TESLA INC", "50,000,000"), idx, n)!;
  const masked = parseDtccLine(row("3", "037833100", "CUSIP", "APPLE INC", ""), idx, n)!;
  const mid = parseDtccLine(row("4", "88160R101", "CUSIP", "TESLA INC", "10,000,000"), idx, n)!;

  const sorted = topNotionalRows([small, masked, large, mid]);
  assert.deepEqual(sorted.map((r) => r.disseminationId), ["2", "4", "1", "3"],
    "largest first; the Dodd-Frank-masked (null) row sorts after every rated row");
});

test("topNotionalRows: caps at the given size instead of returning every row", () => {
  const idx = parseHeader(HEADER_LINE);
  const n = HEADER_COLS.length;
  const rows = Array.from({ length: 5 }, (_, i) =>
    parseDtccLine(row(String(i), "US0378331005", "ISIN", "APPLE INC", String((i + 1) * 1000)), idx, n)!);
  assert.equal(topNotionalRows(rows, 2).length, 2);
  assert.deepEqual(topNotionalRows(rows, 2).map((r) => r.disseminationId), ["4", "3"], "the two largest survive the cap");
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

// ── refreshDtccSwaps: backward date-walk ────────────────────────────────
// Live production finding, 2026-08-23: the archive directory had NEVER
// been created despite the pipeline passing gate 1 two sessions earlier.
// Root cause verified against the real vendor host (kgc0418-tdw-data-0.
// s3.amazonaws.com): GET on the current day AND the immediately-prior day
// both 403 (S3 AccessDenied, its stand-in for "no such key" without
// ListBucket), while 3+ days back 200s cleanly. The old code fetched only
// "today" and relied on the next 6h poll to "self-heal" — which never
// recovers on a day whose own dated file is never going to be published
// (a weekend). These tests pin the fix: walk backward until a real file
// is found, archive rows under the day the pipeline ran, and record which
// vendor-dated file actually supplied them.

function dateBlockedFetch(unavailableUrlFragments: string[], zip: Buffer) {
  return async (url: string) => {
    if (unavailableUrlFragments.some((frag) => url.includes(frag))) {
      return { ok: false, status: 403, body: undefined as any };
    }
    return {
      ok: true,
      status: 200,
      body: Readable.toWeb ? Readable.toWeb(Readable.from([zip])) : Readable.from([zip]),
    };
  };
}

test("refreshDtccSwaps: today's file already published — archives immediately, sourceDate == today", async () => {
  _resetDtccForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-refresh-"));
  const csv = [HEADER_LINE, row("500", "US0378331005", "ISIN", "APPLE INC")].join("\n") + "\n";
  const zip = buildZip(csv);
  const nowMs = Date.parse("2026-08-21T18:00:00Z");
  await refreshDtccSwaps(dateBlockedFetch([], zip), nowMs, dir);
  const hit = latestDtcc();
  assert.ok(hit);
  assert.equal(hit!.sourceDate, "2026-08-21");
  assert.equal(hit!.fileDate, "2026-08-21");
  assert.equal(hit!.usRows, 1);
  assert.equal(hit!.newRows, 1);
  assert.equal(hit!.topRows.length, 1, "the one archived row surfaces in topRows for the /data view");
  assert.equal(hit!.topRows[0].underlierId, "US0378331005");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshDtccSwaps: today + yesterday both 403 (weekend, no file ever published for either) — walks back to the last real file instead of staying warming_up forever", async () => {
  _resetDtccForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-refresh-"));
  const csv = [HEADER_LINE, row("501", "US88160R1014", "ISIN", "TESLA INC")].join("\n") + "\n";
  const zip = buildZip(csv);
  // "now" = Sunday 2026-08-23 — matches the live 403s found on 08-23 and 08-22.
  const nowMs = Date.parse("2026-08-23T12:00:00Z");
  await refreshDtccSwaps(dateBlockedFetch(["2026_08_23", "2026_08_22"], zip), nowMs, dir);
  const hit = latestDtcc();
  assert.ok(hit, "must not stay warming_up forever just because today itself never gets a file");
  assert.equal(hit!.sourceDate, "2026-08-21", "walked back 2 days to the last published file");
  assert.equal(hit!.fileDate, "2026-08-23", "still archived under the day the pipeline actually ran");
  assert.equal(hit!.usRows, 1);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshDtccSwaps: nothing published anywhere in the lookback window — leaves cache unset rather than crashing or fabricating a result", async () => {
  _resetDtccForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-refresh-"));
  const zip = buildZip(HEADER_LINE + "\n");
  const nowMs = Date.parse("2026-08-23T12:00:00Z");
  const allBlocked = Array.from({ length: 8 }, (_, i) =>
    new Date(nowMs - i * 86_400_000).toISOString().slice(0, 10).replace(/-/g, "_"));
  await refreshDtccSwaps(dateBlockedFetch(allBlocked, zip), nowMs, dir);
  assert.equal(latestDtcc(), null);
  fs.rmSync(dir, { recursive: true, force: true });
});

// ── readArchivedDtccRows / cold-cache disk backfill (closes the VULNERABLE-
//    queue's last remaining entry, research/open_questions.md's cold-cache-
//    no-disk-backfill table) ─────────────────────────────────────────────

test("readArchivedDtccRows: returns the newest archived day within the window, skipping empty/missing days", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-backfill-"));
  const r = (diss: string) => ({
    disseminationId: diss, originalDisseminationId: diss, actionType: "NEWT",
    eventTimestamp: "2026-08-10T12:00:00Z", effectiveDate: "2026-08-10",
    notionalAmountLeg1: 50_000_000, notionalCurrencyLeg1: "USD",
    underlierId: "US0378331005", underlierIdSource: "ISIN", underlierName: "APPLE INC",
  });
  fs.mkdirSync(path.join(dir, "dtccswaps"), { recursive: true });
  fs.writeFileSync(path.join(dir, "dtccswaps", "2026-08-08.jsonl.gz"), zlib.gzipSync(JSON.stringify(r("800")) + "\n"));
  fs.writeFileSync(path.join(dir, "dtccswaps", "2026-08-10.jsonl"), JSON.stringify(r("801")) + "\n");
  const nowMs = Date.parse("2026-08-11T12:00:00Z"); // "today" itself has no file
  const result = readArchivedDtccRows(dir, nowMs);
  assert.ok(result);
  assert.equal(result!.fileDate, "2026-08-10", "newest archived day wins, not the oldest in range");
  assert.equal(result!.rows.length, 1);
  assert.equal(result!.rows[0].disseminationId, "801");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("readArchivedDtccRows: no archived day anywhere in the window returns null, not an empty-but-truthy result", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-backfill-empty-"));
  assert.equal(readArchivedDtccRows(dir, Date.parse("2026-08-11T12:00:00Z")), null);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshDtccSwaps: nothing published anywhere in the lookback window, but an earlier day is already archived on disk — backfills instead of staying warming_up forever", async () => {
  _resetDtccForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-refresh-"));
  const archivedRow = {
    disseminationId: "900", originalDisseminationId: "900", actionType: "NEWT",
    eventTimestamp: "2026-08-10T12:00:00Z", effectiveDate: "2026-08-10",
    notionalAmountLeg1: 50_000_000, notionalCurrencyLeg1: "USD",
    underlierId: "US0378331005", underlierIdSource: "ISIN", underlierName: "APPLE INC",
  };
  fs.mkdirSync(path.join(dir, "dtccswaps"), { recursive: true });
  fs.writeFileSync(path.join(dir, "dtccswaps", "2026-08-10.jsonl.gz"), zlib.gzipSync(JSON.stringify(archivedRow) + "\n"));

  const zip = buildZip(HEADER_LINE + "\n");
  const nowMs = Date.parse("2026-08-23T12:00:00Z");
  const allBlocked = Array.from({ length: 8 }, (_, i) =>
    new Date(nowMs - i * 86_400_000).toISOString().slice(0, 10).replace(/-/g, "_"));
  await refreshDtccSwaps(dateBlockedFetch(allBlocked, zip), nowMs, dir);
  const hit = latestDtcc();
  assert.ok(hit, "must not stay warming_up forever — a real archived day sits on disk");
  assert.equal(hit!.fileDate, "2026-08-10", "backfilled from the newest archived day file");
  assert.equal(hit!.sourceDate, "", "vendor source date is honestly unknown on a disk backfill, never fabricated");
  assert.equal(hit!.usRows, 1);
  assert.equal(hit!.newRows, 0, "a backfill archives nothing new this cycle");
  assert.equal(hit!.totalArchived, 1);
  assert.equal(hit!.topRows[0].underlierId, "US0378331005");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshDtccSwaps: an already-warm cache is not clobbered by a disk backfill on a later fully-blocked poll", async () => {
  _resetDtccForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-refresh-"));
  const csv = [HEADER_LINE, row("501", "US88160R1014", "ISIN", "TESLA INC")].join("\n") + "\n";
  const zip = buildZip(csv);
  const firstNowMs = Date.parse("2026-08-21T18:00:00Z");
  await refreshDtccSwaps(dateBlockedFetch([], zip), firstNowMs, dir);
  const warm = latestDtcc();
  assert.ok(warm);

  const secondNowMs = Date.parse("2026-08-25T12:00:00Z");
  const allBlocked = Array.from({ length: 8 }, (_, i) =>
    new Date(secondNowMs - i * 86_400_000).toISOString().slice(0, 10).replace(/-/g, "_"));
  await refreshDtccSwaps(dateBlockedFetch(allBlocked, zip), secondNowMs, dir);
  assert.deepEqual(latestDtcc(), warm, "cache untouched — the backfill guard is `if (!cache)`, not unconditional");
  fs.rmSync(dir, { recursive: true, force: true });
});

// ── seenIds compact index (KNOWN BROKEN #41 leak-audit NEXT(1)) ─────────
// Boot-burst fix: loadSeenIds must prefer a persisted compact index over
// re-scanning every archived day file, self-heal (build + persist the
// index) the first time it finds none, and never let a corrupt/foreign
// file in the same directory poison either path.

test("loadSeenIds: fresh archive dir, no index yet — empty Set, and an index is written so a later legitimately-empty archive doesn't re-scan every boot", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-seenids-"));
  const seen = loadSeenIds(dir);
  assert.equal(seen.size, 0);
  assert.ok(fs.existsSync(path.join(dir, "_seen_ids_index.jsonl.gz")));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("loadSeenIds: migration path — pre-existing day files with no index yet are scanned once, and the index written afterward matches", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-seenids-"));
  const day = [{ disseminationId: "A1" }, { disseminationId: "A2" }]
    .map((r) => JSON.stringify(r)).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-08-21.jsonl.gz"), zlib.gzipSync(day));

  const seen = loadSeenIds(dir);
  assert.deepEqual([...seen].sort(), ["A1", "A2"]);

  const indexRaw = zlib.gunzipSync(fs.readFileSync(path.join(dir, "_seen_ids_index.jsonl.gz"))).toString("utf8");
  assert.deepEqual(indexRaw.split("\n").filter(Boolean).sort(), ["A1", "A2"],
    "index now matches the scanned archive, so the NEXT boot takes the fast path");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("loadSeenIds: fast path — once an index exists it is trusted over the day files, never re-scanned", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-seenids-"));
  // A day file that would produce a DIFFERENT id set if scanned — proves
  // the index, not the day file, is what actually gets returned.
  const day = JSON.stringify({ disseminationId: "STALE-WOULD-BE-WRONG" }) + "\n";
  fs.writeFileSync(path.join(dir, "2026-08-21.jsonl.gz"), zlib.gzipSync(day));
  fs.writeFileSync(path.join(dir, "_seen_ids_index.jsonl.gz"), zlib.gzipSync("B1\nB2\n"));

  const seen = loadSeenIds(dir);
  assert.deepEqual([...seen].sort(), ["B1", "B2"]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("loadSeenIds: a corrupt index falls back to the day-file scan, and the name-anchored glob never misreads the index file itself as a day file", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-seenids-"));
  const day = JSON.stringify({ disseminationId: "C1" }) + "\n";
  fs.writeFileSync(path.join(dir, "2026-08-21.jsonl.gz"), zlib.gzipSync(day));
  fs.writeFileSync(path.join(dir, "_seen_ids_index.jsonl.gz"), Buffer.from("not gzip"));

  const seen = loadSeenIds(dir);
  assert.deepEqual([...seen], ["C1"],
    "recovers from the real day file; picks up nothing from its own unparseable, wrong-shape index file sitting in the same directory");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("archiveNewRows: newly-archived ids are appended to the compact index, not just the day file", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-archive-idx-"));
  _resetDtccForTests();
  const idx = parseHeader(HEADER_LINE);
  const n = HEADER_COLS.length;
  const r = [parseDtccLine(row("X1", "US0378331005", "ISIN", "APPLE INC"), idx, n)!];
  archiveNewRows(r, "2026-08-21", dir);

  const indexPath = path.join(dir, "dtccswaps", "_seen_ids_index.jsonl.gz");
  assert.ok(fs.existsSync(indexPath));
  const ids = zlib.gunzipSync(fs.readFileSync(indexPath)).toString("utf8").split("\n").filter(Boolean);
  assert.deepEqual(ids, ["X1"]);
  fs.rmSync(dir, { recursive: true, force: true });
});

// ── refreshDtccSwaps: crash-loop guard (KNOWN BROKEN #41 leak-audit
//    NEXT(1)) — the same guardedRefresh cooldown convention routes.ts
//    already applies to refreshShadowStats/refreshPortDwell.

test("refreshDtccSwaps: an unresolved crash marker (simulated OOM mid-fetch on a prior boot) triggers the cooldown — skips this poll instead of immediately re-running the expensive fetch", async () => {
  _resetDtccForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-crashguard-"));
  const csv = [HEADER_LINE, row("900", "US0378331005", "ISIN", "APPLE INC")].join("\n") + "\n";
  const zip = buildZip(csv);
  const nowMs = Date.parse("2026-09-19T12:00:00Z");

  // A real process crash never reaches guardedRefresh's `finally`, leaving
  // exactly this shape on disk: started, never completed.
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "voltrade_refresh_attempt_dtccswaps.json"),
    JSON.stringify({ startedAt: nowMs - 60_000 }));

  await refreshDtccSwaps(dateBlockedFetch([], zip), nowMs, dir);
  assert.equal(latestDtcc(), null, "cooldown suppresses the poll even though the fetch would have succeeded");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshDtccSwaps: a normal completed poll resolves the crash marker so the next boot is not stuck in cooldown", async () => {
  _resetDtccForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-crashguard-"));
  const csv = [HEADER_LINE, row("901", "US0378331005", "ISIN", "APPLE INC")].join("\n") + "\n";
  const zip = buildZip(csv);
  const nowMs = Date.parse("2026-09-19T12:00:00Z");

  await refreshDtccSwaps(dateBlockedFetch([], zip), nowMs, dir);
  assert.ok(latestDtcc(), "first poll runs normally");

  const marker = JSON.parse(fs.readFileSync(path.join(dir, "voltrade_refresh_attempt_dtccswaps.json"), "utf8"));
  assert.ok(marker.completedAt, "marker resolved after a normal (non-crashing) run, so it never falsely cools down the next boot");
  fs.rmSync(dir, { recursive: true, force: true });
});

// ── On-disk ID store modes (memory fix A1, 2026-11-03) ───────────────────
// VOLTRADE_DTCC_IDS_STORE = memory | dual (default) | sqlite. The dedup
// DECISION is the only thing computed differently: archived rows, the
// compact index and the API-facing cache must be identical across modes;
// dual logs every disagreement as DTCC-IDS-MISMATCH and serves the memory
// answer; sqlite never builds the Set; memory is always a clean rollback.

import Database from "better-sqlite3";
import { dtccIdsStats, setDtccIdsAuditSink } from "./dtccSwaps";

type IdsMode = "memory" | "dual" | "sqlite";
const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";

async function withMode<T>(mode: IdsMode, fn: () => Promise<T> | T): Promise<T> {
  const prev = process.env[IDS_ENV];
  process.env[IDS_ENV] = mode;
  _resetDtccForTests();
  try {
    return await fn();
  } finally {
    _resetDtccForTests();
    if (prev === undefined) delete process.env[IDS_ENV]; else process.env[IDS_ENV] = prev;
  }
}

function captureConsole() {
  const errors: string[] = [];
  const logs: string[] = [];
  const origErr = console.error;
  const origLog = console.log;
  console.error = (...args: unknown[]) => { errors.push(args.map(String).join(" ")); };
  console.log = (...args: unknown[]) => { logs.push(args.map(String).join(" ")); };
  return { errors, logs, restore: () => { console.error = origErr; console.log = origLog; } };
}

function seedIndex(baseDir: string, ids: readonly string[]): void {
  fs.mkdirSync(path.join(baseDir, "dtccswaps"), { recursive: true });
  fs.writeFileSync(path.join(baseDir, "dtccswaps", "_seen_ids_index.jsonl.gz"),
    zlib.gzipSync(ids.join("\n") + (ids.length ? "\n" : "")));
}

function fourRows() {
  const idx = parseHeader(HEADER_LINE);
  const n = HEADER_COLS.length;
  return [
    parseDtccLine(row("1", "US0378331005", "ISIN", "APPLE INC", "1,000,000"), idx, n)!,
    parseDtccLine(row("2", "US88160R1014", "ISIN", "TESLA INC", "50,000,000"), idx, n)!,
    parseDtccLine(row("3", "037833100", "CUSIP", "APPLE INC", ""), idx, n)!,
    parseDtccLine(row("4", "88160R101", "CUSIP", "TESLA INC", "10,000,000"), idx, n)!,
  ];
}

test("ids store modes: memory / dual / sqlite make the IDENTICAL fresh selection from the same seeded index, and leave byte-identical day files and compact indexes", async () => {
  const outputs: Record<string, { day: Buffer; index: Buffer; r1: { newRows: number; seenTotal: number }; r2: { newRows: number; seenTotal: number } }> = {};
  for (const mode of ["memory", "dual", "sqlite"] as const) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `dtcc-mode-${mode}-`));
    seedIndex(dir, ["1", "2"]); // two already-archived ids, two new ones in the rows
    await withMode(mode, () => {
      const r1 = archiveNewRows(fourRows(), "2026-11-03", dir);
      assert.equal(r1.newRows, 2, `${mode}: only ids 3 and 4 are fresh`);
      assert.equal(r1.seenTotal, 4, `${mode}: total distinct ids`);
      const r2 = archiveNewRows(fourRows(), "2026-11-03", dir);
      assert.equal(r2.newRows, 0, `${mode}: a re-served file archives nothing`);
      assert.equal(r2.seenTotal, 4);
      const stats = dtccIdsStats();
      assert.equal(stats.mode, mode);
      assert.equal(stats.mismatches, 0, `${mode}: no disagreement`);
      if (mode === "memory") {
        assert.equal(stats.memoryCount, 4);
        assert.equal(stats.dbCount, null, "memory mode never opens the db");
        assert.equal(stats.backfill, null);
        assert.ok(!fs.existsSync(path.join(dir, "dtccswaps", "_seen_ids.db")), "memory mode leaves no db file behind");
      } else if (mode === "dual") {
        assert.equal(stats.memoryCount, 4);
        assert.equal(stats.dbCount, 4);
        assert.equal(stats.backfill?.state, "done");
        assert.equal(stats.lastLookupRows, 4);
        assert.ok(typeof stats.lastLookupMs === "number" && stats.lastLookupMs >= 0);
      } else {
        assert.equal(stats.memoryCount, null, "sqlite mode NEVER builds the Set — the heap win");
        assert.equal(stats.dbCount, 4);
        assert.equal(stats.backfill?.dbBefore, 0);
        assert.equal(stats.backfill?.indexCount, 2, "backfilled the two seeded ids before deciding");
        assert.equal(stats.journalMode, "wal");
        assert.equal(stats.dbPath, path.join(dir, "dtccswaps", "_seen_ids.db"));
      }
      outputs[mode] = {
        day: fs.readFileSync(path.join(dir, "dtccswaps", "2026-11-03.jsonl.gz")),
        index: fs.readFileSync(path.join(dir, "dtccswaps", "_seen_ids_index.jsonl.gz")),
        r1, r2,
      };
    });
    fs.rmSync(dir, { recursive: true, force: true });
  }
  for (const mode of ["dual", "sqlite"] as const) {
    assert.ok(outputs[mode].day.equals(outputs.memory.day), `${mode}: day file byte-identical to memory mode`);
    assert.ok(outputs[mode].index.equals(outputs.memory.index), `${mode}: compact index byte-identical to memory mode (still appended, same format)`);
    assert.deepEqual(outputs[mode].r1, outputs.memory.r1);
    assert.deepEqual(outputs[mode].r2, outputs.memory.r2);
  }
  const dayIds = zlib.gunzipSync(outputs.memory.day).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).disseminationId);
  assert.deepEqual(dayIds, ["3", "4"]);
  assert.deepEqual(zlib.gunzipSync(outputs.memory.index).toString("utf8").split("\n").filter(Boolean), ["1", "2", "3", "4"]);
});

test("ids store modes: refreshDtccSwaps produces the identical API-facing cache in all three modes (only `at` differs), and dual/sqlite run the STREAMED backfill before archiving", async () => {
  const csv = [HEADER_LINE,
    row("700", "US0378331005", "ISIN", "APPLE INC", "5,000,000"),
    row("701", "US88160R1014", "ISIN", "TESLA INC", "9,000,000"),
    row("702", "037833100", "CUSIP", "APPLE INC", ""),
  ].join("\n") + "\n";
  const zip = buildZip(csv);
  const nowMs = Date.parse("2026-11-03T18:00:00Z");
  const caches: Record<string, unknown> = {};
  for (const mode of ["memory", "dual", "sqlite"] as const) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `dtcc-mode-refresh-${mode}-`));
    seedIndex(dir, ["700"]); // one of the three is already archived
    await withMode(mode, async () => {
      const cap = captureConsole();
      try {
        await refreshDtccSwaps(dateBlockedFetch([], zip), nowMs, dir);
      } finally { cap.restore(); }
      const hit = latestDtcc();
      assert.ok(hit, `${mode}: poll produced a cache`);
      assert.equal(hit!.newRows, 2, `${mode}: 701 and 702 are new`);
      assert.equal(hit!.totalArchived, 3);
      assert.equal(hit!.usRows, 3);
      const { at: _at, ...rest } = hit!;
      caches[mode] = rest;
      const stats = dtccIdsStats();
      if (mode === "memory") {
        assert.equal(stats.dbCount, null);
        assert.ok(!cap.logs.some((l) => l.includes("DTCC-IDS-BACKFILL")), "memory mode: no backfill at all");
      } else {
        assert.equal(stats.backfill?.state, "done");
        assert.equal(stats.backfill?.source, "streamed", `${mode}: the boot path awaited the async backfill before the poll archived`);
        assert.equal(stats.backfill?.dbBefore, 0);
        assert.equal(stats.backfill?.indexCount, 1);
        assert.equal(stats.dbCount, 3);
        assert.ok(cap.logs.some((l) => /DTCC-IDS-BACKFILL ok: db 0 -> 1, index 1, imported 1/.test(l)), `${mode}: backfill line carries both counts`);
        assert.equal(stats.mismatches, 0);
      }
      if (mode === "sqlite") assert.equal(stats.memoryCount, null, "sqlite: no Set, even through the full refresh path");
    });
    fs.rmSync(dir, { recursive: true, force: true });
  }
  assert.deepEqual(caches.dual, caches.memory, "dual == memory");
  assert.deepEqual(caches.sqlite, caches.memory, "sqlite == memory");
});

test("dual mode: a dedup disagreement is logged as DTCC-IDS-MISMATCH (counts + example ids), counted in dtccIdsStats, delivered to the audit sink, and the MEMORY answer is served", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-dual-mismatch-"));
  seedIndex(dir, ["1", "2"]);
  // A db that already holds "3" (which the index/Set do not) — the two
  // sources now disagree on whether row 3 is fresh.
  const raw = new Database(path.join(dir, "dtccswaps", "_seen_ids.db"));
  raw.exec("CREATE TABLE IF NOT EXISTS seen_ids (id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID");
  raw.prepare("INSERT INTO seen_ids (id) VALUES (?)").run("1");
  raw.prepare("INSERT INTO seen_ids (id) VALUES (?)").run("2");
  raw.prepare("INSERT INTO seen_ids (id) VALUES (?)").run("3");
  raw.close();
  const audited: Array<[string, string]> = [];
  setDtccIdsAuditSink((action, detail) => audited.push([action, detail]));
  try {
    await withMode("dual", () => {
      const cap = captureConsole();
      let r: { newRows: number; seenTotal: number };
      try { r = archiveNewRows(fourRows(), "2026-11-03", dir); } finally { cap.restore(); }
      assert.equal(r.newRows, 2, "MEMORY answer served: rows 3 and 4 archived (the db alone would have archived only 4)");
      assert.equal(r.seenTotal, 4);
      const stats = dtccIdsStats();
      assert.equal(stats.mismatches, 1);
      assert.equal(stats.lastMismatch?.kind, "decision");
      assert.match(stats.lastMismatch!.detail, /memory says 2 fresh, db says 1/);
      assert.match(stats.lastMismatch!.detail, /fresh-only-in-memory 1 \(e\.g\. 3\)/);
      assert.ok(cap.errors.some((l) => l.includes("DTCC-IDS-MISMATCH: decision:")), "console.error carries the mismatch");
      assert.equal(stats.backfill?.state, "mismatch", "the boot-time count check also flagged db 3 != index 2");
      assert.ok(cap.errors.some((l) => /DTCC-IDS-BACKFILL-MISMATCH: db 3 != index 2/.test(l)));
      assert.deepEqual(audited.map((a) => a[0]), ["DTCC-IDS-MISMATCH"], "the audit sink gets the mismatch (backfill lines are the store's own channel)");
      // After the poll both sides hold {1,2,3,4}: the count check agrees.
      assert.equal(stats.dbCount, 4);
      assert.equal(stats.memoryCount, 4);
      const dayIds = zlib.gunzipSync(fs.readFileSync(path.join(dir, "dtccswaps", "2026-11-03.jsonl.gz"))).toString("utf8")
        .split("\n").filter(Boolean).map((l) => JSON.parse(l).disseminationId);
      assert.deepEqual(dayIds, ["3", "4"], "the archive follows the memory verdict");
    });
  } finally {
    setDtccIdsAuditSink(null);
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("dual mode: the db drifting from the Set after archiving is caught as a 'count' mismatch on the next poll (and the decision mismatch it implies)", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-dual-count-"));
  await withMode("dual", () => {
    assert.equal(archiveNewRows(fourRows(), "2026-11-03", dir).newRows, 4);
    assert.equal(dtccIdsStats().mismatches, 0);
    // Simulate drift: a row vanishes from the db behind the module's back.
    const raw = new Database(path.join(dir, "dtccswaps", "_seen_ids.db"));
    raw.prepare("DELETE FROM seen_ids WHERE id = ?").run("2");
    raw.close();
    const cap = captureConsole();
    let r: { newRows: number; seenTotal: number };
    try { r = archiveNewRows(fourRows(), "2026-11-03", dir); } finally { cap.restore(); }
    assert.equal(r.newRows, 0, "memory still says everything is seen — served");
    const stats = dtccIdsStats();
    assert.equal(stats.mismatches, 2, "one decision mismatch (db thinks 2 is fresh) + one count mismatch (4 vs 3)");
    assert.equal(stats.lastMismatch?.kind, "count");
    assert.match(stats.lastMismatch!.detail, /set 4 vs db 3/);
    assert.ok(cap.errors.some((l) => l.includes("DTCC-IDS-MISMATCH: decision:")));
    assert.ok(cap.errors.some((l) => l.includes("DTCC-IDS-MISMATCH: count:")));
  });
  fs.rmSync(dir, { recursive: true, force: true });
});

test("rollback: sqlite mode archives (index still appended) → memory mode dedups from the index alone, nothing re-archived → back to sqlite backfills the gap and continues", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-rollback-"));
  const rows = fourRows();
  await withMode("sqlite", () => {
    const r = archiveNewRows(rows.slice(0, 3), "2026-11-01", dir);
    assert.deepEqual(r, { newRows: 3, seenTotal: 3 });
    assert.equal(dtccIdsStats().memoryCount, null, "sqlite: Set never built");
    assert.equal(dtccIdsStats().dbCount, 3);
    assert.deepEqual(zlib.gunzipSync(fs.readFileSync(path.join(dir, "dtccswaps", "_seen_ids_index.jsonl.gz"))).toString("utf8").split("\n").filter(Boolean),
      ["1", "2", "3"], "the compact index is appended in sqlite mode exactly as before — the rollback path stays intact");
  });
  await withMode("memory", () => {
    const r = archiveNewRows(rows, "2026-11-02", dir);
    assert.deepEqual(r, { newRows: 1, seenTotal: 4 }, "memory mode (rolled back) sees 1-3 via the index, archives only 4");
    assert.equal(dtccIdsStats().dbCount, null, "memory mode did not touch the db");
    assert.equal(dtccIdsStats().memoryCount, 4);
  });
  await withMode("sqlite", () => {
    const idx = parseHeader(HEADER_LINE);
    const five = [...rows, parseDtccLine(row("5", "US0378331005", "ISIN", "APPLE INC", "2,000,000"), idx, HEADER_COLS.length)!];
    const cap = captureConsole();
    let r: { newRows: number; seenTotal: number };
    try { r = archiveNewRows(five, "2026-11-03", dir); } finally { cap.restore(); }
    assert.deepEqual(r, { newRows: 1, seenTotal: 5 }, "db was behind the index by the id archived in memory mode; backfilled, then only 5 is new");
    const stats = dtccIdsStats();
    assert.equal(stats.backfill?.state, "done");
    assert.equal(stats.backfill?.dbBefore, 3);
    assert.equal(stats.backfill?.indexCount, 4);
    assert.equal(stats.backfill?.imported, 1);
    assert.equal(stats.backfill?.source, "sync", "a direct archive call before any boot pass takes the synchronous twin");
    assert.ok(cap.logs.some((l) => /DTCC-IDS-BACKFILL ok: db 3 -> 4, index 4, imported 1/.test(l)));
    assert.equal(stats.dbCount, 5);
    assert.equal(stats.memoryCount, null);
  });
  fs.rmSync(dir, { recursive: true, force: true });
});

test("store failure degrades the call to the memory path LOUDLY (DTCC-IDS-DB-ERROR) instead of stopping the archive — in sqlite mode too", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-store-fail-"));
  // A directory where the db file should be: SQLite cannot open it.
  fs.mkdirSync(path.join(dir, "dtccswaps", "_seen_ids.db"), { recursive: true });
  await withMode("sqlite", () => {
    const cap = captureConsole();
    let r: { newRows: number; seenTotal: number };
    try { r = archiveNewRows(fourRows(), "2026-11-03", dir); } finally { cap.restore(); }
    assert.deepEqual(r, { newRows: 4, seenTotal: 4 }, "archived via the in-memory Set");
    const stats = dtccIdsStats();
    assert.ok(stats.storeError, "the open failure is recorded for the health diagnostic");
    assert.equal(stats.memoryCount, 4, "the Set was built as the degrade path");
    assert.equal(stats.dbCount, null);
    assert.ok(cap.errors.some((l) => l.includes("DTCC-IDS-DB-ERROR: could not open")), "never silent");
    assert.ok(fs.existsSync(path.join(dir, "dtccswaps", "2026-11-03.jsonl.gz")), "the day file was written regardless");
  });
  fs.rmSync(dir, { recursive: true, force: true });
});
