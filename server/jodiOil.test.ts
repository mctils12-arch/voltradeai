// JODI oil/gas battery: CSV filter correctness (CLOSTLV x CONVBBL only,
// '-'/'x' -> null), recent-vs-frozen period archiving + gzip, HEAD-probe
// change detection, and restart-rehydration from disk (no re-download).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { zipSync } from "fflate";
import {
  parseJodiCsv, isRecentPeriod, metaUnchanged, probeJodiMeta, fetchJodiRows,
  archiveJodiRows, readArchivedPeriod, listArchivedPeriods,
  readStoredMeta, writeStoredMeta, refreshJodi, latestJodi, RECENT_MONTHS,
} from "./jodiOil";

const CSV_HEADER = "REF_AREA,TIME_PERIOD,ENERGY_PRODUCT,FLOW_BREAKDOWN,UNIT_MEASURE,OBS_VALUE,ASSESSMENT_CODE\r\n";

test("parseJodiCsv: keeps only CLOSTLV x CONVBBL; other flows/units dropped; no-data tokens -> null", () => {
  const csv = CSV_HEADER +
    "AE,2002-01,CRUDEOIL,CLOSTLV,CONVBBL,7596.0000,3\r\n" +
    "AE,2002-01,CRUDEOIL,CLOSTLV,KBBL,-,3\r\n" +          // wrong unit, dropped
    "AE,2002-01,CRUDEOIL,INDPROD,CONVBBL,1234.0,1\r\n" +  // wrong flow, dropped
    "SA,2002-01,CRUDEOIL,CLOSTLV,CONVBBL,x,4\r\n" +       // confidential -> null
    "SA,2002-01,NGL,CLOSTLV,CONVBBL,-,2\r\n" +            // no data -> null
    "AE,2002-04,NGL,CLOSTLV,CONVBBL,N/A,3\r\n" +          // live-verified token -> null
    "AE,2002-05,NGL,CLOSTLV,CONVBBL,..,3\r\n";             // live-verified token -> null
  const rows = parseJodiCsv(Buffer.from(csv, "utf8"), "2026-07-07");
  assert.equal(rows.length, 5);
  assert.deepEqual(rows[0], {
    ref_area: "AE", time_period: "2002-01", energy_product: "CRUDEOIL",
    stock: 7596, assessment_code: "3", rt: "2026-07-07",
  });
  assert.equal(rows[1].stock, null, "'x' (confidential) becomes null, not zero");
  assert.equal(rows[2].stock, null, "'-' (no data) becomes null, not zero");
  assert.equal(rows[3].stock, null, "'N/A' (live-verified JODI no-data token) becomes null");
  assert.equal(rows[4].stock, null, "'..' (live-verified JODI no-data token) becomes null");
});

test("parseJodiCsv: handles no trailing newline and blank lines", () => {
  const csv = CSV_HEADER + "\nAE,2002-02,CRUDEOIL,CLOSTLV,CONVBBL,100.5,1";
  const rows = parseJodiCsv(Buffer.from(csv, "utf8"), "2026-07-07");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].stock, 100.5);
});

test("isRecentPeriod: within RECENT_MONTHS of now is revisable, older is frozen", () => {
  const now = Date.parse("2026-07-07T00:00:00Z");
  assert.ok(isRecentPeriod("2026-07", now));
  assert.ok(isRecentPeriod("2026-02", now), `${RECENT_MONTHS}-month-old boundary still recent`);
  assert.ok(!isRecentPeriod("2025-01", now), "18 months old is frozen");
  assert.ok(!isRecentPeriod("not-a-period", now));
});

test("metaUnchanged: requires both etag+last_modified to match; empty meta never matches", () => {
  const a = { etag: '"x"', last_modified: "Thu, 25 Jun 2026 03:50:04 GMT" };
  assert.ok(metaUnchanged(a, { ...a }));
  assert.ok(!metaUnchanged(a, { ...a, etag: '"y"' }));
  assert.ok(!metaUnchanged(a, null));
  assert.ok(!metaUnchanged({ etag: null, last_modified: null }, { etag: null, last_modified: null }));
});

test("probeJodiMeta: HEAD request reads etag/last-modified; non-200 -> null", async () => {
  let seenMethod: string | undefined;
  const ok = async (_url: string, init: any) => {
    seenMethod = init?.method;
    return { ok: true, status: 200, headers: { get: (k: string) => (k === "etag" ? '"abc"' : "Thu, 25 Jun 2026 03:50:04 GMT") }, arrayBuffer: async () => new ArrayBuffer(0) };
  };
  const meta = await probeJodiMeta(ok as any);
  assert.equal(seenMethod, "HEAD");
  assert.deepEqual(meta, { etag: '"abc"', last_modified: "Thu, 25 Jun 2026 03:50:04 GMT" });
  const bad = async () => ({ ok: false, status: 500, arrayBuffer: async () => new ArrayBuffer(0) });
  assert.equal(await probeJodiMeta(bad as any), null);
});

test("fetchJodiRows: unzips a real (fflate-built) single-entry zip and filters", async () => {
  const csv = CSV_HEADER + "AE,2002-01,CRUDEOIL,CLOSTLV,CONVBBL,7596.0000,3\r\n";
  const zipped = zipSync({ "NewProcedure_Primary_CSV.csv": new TextEncoder().encode(csv) });
  const fetchImpl = async () => ({
    ok: true, status: 200,
    arrayBuffer: async () => zipped.buffer.slice(zipped.byteOffset, zipped.byteOffset + zipped.byteLength),
  });
  const rows = await fetchJodiRows(fetchImpl as any, Date.parse("2026-07-07T00:00:00Z"));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].ref_area, "AE");
  assert.equal(rows[0].stock, 7596);
});

test("fetchJodiRows: non-200 throws", async () => {
  const bad = async () => ({ ok: false, status: 503, arrayBuffer: async () => new ArrayBuffer(0) });
  await assert.rejects(() => fetchJodiRows(bad as any));
});

test("archive: recent periods always rewritten; older periods archived once then gzipped", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "jodi-"));
  const now = Date.parse("2026-07-07T00:00:00Z");
  const oldRows = [{ ref_area: "AE", time_period: "2010-01", energy_product: "CRUDEOIL", stock: 1, assessment_code: "3", rt: "x" }];
  assert.equal(archiveJodiRows(oldRows as any, base, now), 1);
  const oldDir = path.join(base, "jodioil");
  assert.ok(fs.existsSync(path.join(oldDir, "2010-01.jsonl.gz")), "old period gzipped immediately");
  assert.ok(!fs.existsSync(path.join(oldDir, "2010-01.jsonl")), "no lingering plain file for a frozen period");

  // second run: old period already archived -> skipped (not rewritten)
  const oldRowsRevised = [{ ...oldRows[0], stock: 999 }];
  assert.equal(archiveJodiRows(oldRowsRevised as any, base, now), 0, "frozen period is never rewritten");
  assert.deepEqual(readArchivedPeriod("2010-01", base)[0].stock, 1, "on-disk value unchanged");

  // recent period: written plain, and rewritten again with new data (revision)
  const recentV1 = [{ ref_area: "AE", time_period: "2026-06", energy_product: "CRUDEOIL", stock: 10, assessment_code: "3", rt: "x" }];
  assert.equal(archiveJodiRows(recentV1 as any, base, now), 1);
  assert.ok(fs.existsSync(path.join(oldDir, "2026-06.jsonl")), "recent period stays plain (revisable)");
  const recentV2 = [{ ...recentV1[0], stock: 20 }];
  assert.equal(archiveJodiRows(recentV2 as any, base, now), 1, "recent period IS rewritten");
  assert.equal(readArchivedPeriod("2026-06", base)[0].stock, 20, "revision landed");

  assert.deepEqual(listArchivedPeriods(base), ["2010-01", "2026-06"]);
});

test("readArchivedPeriod: reads both plain and gz, missing period -> []", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "jodi-"));
  assert.deepEqual(readArchivedPeriod("2099-01", base), []);
  const dir = path.join(base, "jodioil");
  fs.mkdirSync(dir, { recursive: true });
  const row = { ref_area: "SA", time_period: "2020-05", energy_product: "NGL", stock: 5, assessment_code: "1", rt: "x" };
  fs.writeFileSync(path.join(dir, "2020-05.jsonl.gz"), zlib.gzipSync(JSON.stringify(row) + "\n"));
  assert.deepEqual(readArchivedPeriod("2020-05", base), [row]);
});

test("stored meta: round-trips through disk", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "jodi-"));
  assert.equal(readStoredMeta(base), null);
  writeStoredMeta({ etag: '"a"', last_modified: "L" }, base);
  assert.deepEqual(readStoredMeta(base), { etag: '"a"', last_modified: "L" });
});

test("refreshJodi: unchanged HEAD probe skips the expensive fetch entirely", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "jodi-"));
  const now = Date.parse("2026-07-07T00:00:00Z");
  writeStoredMeta({ etag: '"same"', last_modified: "L" }, base);
  const dir = path.join(base, "jodioil");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-06.jsonl"),
    JSON.stringify({ ref_area: "AE", time_period: "2026-06", energy_product: "CRUDEOIL", stock: 1, assessment_code: "3", rt: "x" }) + "\n");

  let getCalled = false;
  const fetchImpl = async (_url: string, init: any) => {
    if (init?.method === "HEAD") {
      return { ok: true, status: 200, headers: { get: (k: string) => (k === "etag" ? '"same"' : "L") }, arrayBuffer: async () => new ArrayBuffer(0) };
    }
    getCalled = true;
    return { ok: true, status: 200, arrayBuffer: async () => new ArrayBuffer(0) };
  };
  await refreshJodi(fetchImpl as any, now, base);
  assert.equal(getCalled, false, "no GET when HEAD says unchanged");
  const hit = latestJodi();
  assert.ok(hit);
  assert.equal(hit!.period, "2026-06");
});

test("refreshJodi: changed HEAD probe triggers full fetch + archive + cache update", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "jodi-"));
  const now = Date.parse("2026-07-07T00:00:00Z");
  writeStoredMeta({ etag: '"old"', last_modified: "OLD" }, base);
  const csv = CSV_HEADER +
    "AE,2026-06,CRUDEOIL,CLOSTLV,CONVBBL,111,3\r\n" +
    "SA,2026-04,CRUDEOIL,CLOSTLV,CONVBBL,222,3\r\n";
  const zipped = zipSync({ "NewProcedure_Primary_CSV.csv": new TextEncoder().encode(csv) });
  const fetchImpl = async (_url: string, init: any) => {
    if (init?.method === "HEAD") {
      return { ok: true, status: 200, headers: { get: (k: string) => (k === "etag" ? '"new"' : "NEW") }, arrayBuffer: async () => new ArrayBuffer(0) };
    }
    return { ok: true, status: 200, arrayBuffer: async () => zipped.buffer.slice(zipped.byteOffset, zipped.byteOffset + zipped.byteLength) };
  };
  await refreshJodi(fetchImpl as any, now, base);
  const hit = latestJodi();
  assert.ok(hit);
  assert.equal(hit!.period, "2026-06", "cache holds only the newest period's rows");
  assert.equal(hit!.rows.length, 1);
  assert.deepEqual(readStoredMeta(base), { etag: '"new"', last_modified: "NEW" });
  assert.deepEqual(listArchivedPeriods(base), ["2026-04", "2026-06"]);
});
