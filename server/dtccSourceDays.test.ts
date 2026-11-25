// dtccSourceDays.test.ts — [PIPELINE] 2026-11-25, open_questions.md (an2):
// the DTCC swaps poll read only the newest vendor file, each file holds one
// day's events, and nothing recorded WHICH files had been read. A vendor day
// that was never the newest at the moment of a poll was a silent, permanent
// gap. These tests pin the source-day ledger and the catch-up that reads the
// days it does not hold.
// The edge reads (open_questions.md (ap4), server/dtccEdgeDays.test.ts) are
// switched off in every poll here (VOLTRADE_DTCC_EDGE=0) so each test's list
// of files asked for is the catch-up's alone.
// Run: npx tsx --test server/dtccSourceDays.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { Readable } from "node:stream";
import {
  REQUIRED_COLUMNS, refreshDtccSwaps, latestDtcc, loadSeenIds, setDtccIdsAuditSink, _resetDtccForTests,
  readDtccSourceDays, dtccCatchupCandidates, dtccSourceDaysSummary, dtccCatchupEnabled,
  DTCC_CATCHUP_FLOOR, DTCC_CATCHUP_PER_POLL, DTCC_CATCHUP_RETRY_MS, DTCC_CATCHUP_ENV, type DtccSourceDays,
  DTCC_EDGE_ENV,
} from "./dtccSwaps";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const HEADER_COLS = [...REQUIRED_COLUMNS];
const HEADER_LINE = HEADER_COLS.map((c) => `"${c}"`).join(",");

function csvRow(diss: string): string {
  const byName: Record<string, string> = {
    "Dissemination Identifier": diss, "Original Dissemination Identifier": diss, "Action type": "NEWT",
    "Event timestamp": "2026-08-21T12:33:36Z", "Effective Date": "2026-08-21", "Asset Class": "EQ",
    "Notional amount-Leg 1": "1,000,000", "Notional currency-Leg 1": "USD",
    "Underlier ID-Leg 1": "US0378331005", "Underlier ID source-Leg 1": "ISIN", "UPI Underlier Name": "APPLE INC",
  };
  return HEADER_COLS.map((c) => `"${byName[c] ?? ""}"`).join(",");
}

function buildZip(ids: string[], truncateCompressedTo?: number): Buffer {
  const nameBuf = Buffer.from("SEC_CUMULATIVE_EQUITIES.csv", "utf8");
  const dataBuf = Buffer.from([HEADER_LINE, ...ids.map(csvRow)].join("\n") + "\n", "utf8");
  const compressed = zlib.deflateRawSync(dataBuf);
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(20, 4);
  header.writeUInt16LE(8, 8);
  header.writeUInt32LE(compressed.length, 18);
  header.writeUInt32LE(dataBuf.length, 22);
  header.writeUInt16LE(nameBuf.length, 26);
  // truncated: the header still promises the whole member, the body ends early
  const body = truncateCompressedTo === undefined ? compressed : compressed.subarray(0, truncateCompressedTo);
  return Buffer.concat([header, nameBuf, body]);
}

/** Serves the vendor days in `files` (key YYYY-MM-DD); every other day is 403. */
function vendor(files: Record<string, Buffer>) {
  const asked: string[] = [];
  const fetchImpl = async (url: string) => {
    const m = /EQUITIES_(\d{4})_(\d{2})_(\d{2})\.zip/.exec(url);
    const day = m ? `${m[1]}-${m[2]}-${m[3]}` : "";
    asked.push(day);
    const zip = files[day];
    if (!zip) return { ok: false, status: 403, body: undefined as any };
    return { ok: true, status: 200, body: Readable.toWeb ? Readable.toWeb(Readable.from([zip])) : Readable.from([zip]) };
  };
  return { fetchImpl: fetchImpl as any, asked };
}

async function withEnv<T>(vars: Record<string, string | undefined>, fn: () => Promise<T>): Promise<T> {
  const before: Record<string, string | undefined> = {};
  for (const k of Object.keys(vars)) { before[k] = process.env[k]; if (vars[k] === undefined) delete process.env[k]; else process.env[k] = vars[k]; }
  try { return await fn(); } finally {
    for (const k of Object.keys(vars)) { if (before[k] === undefined) delete process.env[k]; else process.env[k] = before[k]; }
  }
}

function freshDir(): string {
  _resetDtccForTests();
  return fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-sourcedays-"));
}

function dayFileIds(dir: string, day: string): string[] {
  const fp = path.join(dir, "dtccswaps", `${day}.jsonl.gz`);
  if (!fs.existsSync(fp)) return [];
  return zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).disseminationId);
}

const at = (iso: string) => Date.parse(iso);
const read = (lastAt: string): DtccSourceDays[string] => ({ firstAt: lastAt, lastAt, via: "poll", usRows: 1, newRows: 1, reads: 1 });

test("candidates: strictly before the newest file, newest first, capped, never below the floor", () => {
  const now = at("2026-08-26T01:00:00Z");
  assert.equal(DTCC_CATCHUP_FLOOR, "2026-08-21");
  assert.deepEqual(dtccCatchupCandidates({}, "2026-08-25", now, 2), ["2026-08-24", "2026-08-23"]);
  const held: DtccSourceDays = { "2026-08-24": read("2026-08-25T01:00:00Z"), "2026-08-23": read("2026-08-25T01:00:00Z") };
  assert.deepEqual(dtccCatchupCandidates(held, "2026-08-25", now, 2), ["2026-08-22", "2026-08-21"]);
  assert.deepEqual(dtccCatchupCandidates(held, "2026-08-25", now, 10), ["2026-08-22", "2026-08-21"], "2026-08-20 is below the floor");
  held["2026-08-22"] = read("2026-08-25T01:00:00Z"); held["2026-08-21"] = read("2026-08-25T01:00:00Z");
  assert.deepEqual(dtccCatchupCandidates(held, "2026-08-25", now, 2), [], "every day held: nothing is asked again");
  assert.deepEqual(dtccCatchupCandidates({}, "2026-08-21", now, 2), [], "the newest file is the poll's, not the catch-up's");
  assert.deepEqual(dtccCatchupCandidates({}, "not-a-date", now, 2), []);
});

test("candidates: a day that did not answer is asked again only after the retry interval", () => {
  const asked = "2026-08-25T01:00:00.000Z";
  const days: DtccSourceDays = { "2026-08-24": { firstAt: asked, lastAt: asked, via: "catchup", usRows: 0, newRows: 0, reads: 1, unavailable: 403 } };
  assert.deepEqual(dtccCatchupCandidates(days, "2026-08-25", at(asked) + DTCC_CATCHUP_RETRY_MS - 1, 1), ["2026-08-23"]);
  assert.deepEqual(dtccCatchupCandidates(days, "2026-08-25", at(asked) + DTCC_CATCHUP_RETRY_MS, 1), ["2026-08-24"]);
});

test("THE GAP: a vendor day no poll read while it was the newest is read by the next poll, archived, counted and audited", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: undefined, [DTCC_EDGE_ENV]: "0" }, async () => {
    const dir = freshDir();
    const audits: string[] = [];
    setDtccIdsAuditSink((type, message) => audits.push(`${type}: ${message}`));
    try {
      const files = { "2026-08-22": buildZip(["1", "2"]), "2026-08-23": buildZip(["3", "4"]), "2026-08-24": buildZip(["5", "6"]) };
      // Saturday's poll reads the 08-22 file; then no poll runs until Tuesday 01:00Z
      await refreshDtccSwaps(vendor({ "2026-08-22": files["2026-08-22"] }).fetchImpl, at("2026-08-22T18:00:00Z"), dir);
      assert.equal(latestDtcc()!.sourceDate, "2026-08-22");
      assert.equal(audits.length, 0, "2026-08-21 does not answer here; that is not a refill");

      const v = vendor(files);
      await refreshDtccSwaps(v.fetchImpl, at("2026-08-25T01:00:00Z"), dir);
      const hit = latestDtcc()!;
      assert.equal(hit.sourceDate, "2026-08-24", "the poll's own file is still the newest that answers");
      assert.equal(hit.newRows, 2, "the poll's count is the newest file's, not the catch-up's");
      assert.equal(hit.totalArchived, 6, "the total includes what the catch-up added");

      _resetDtccForTests();
      const seen = loadSeenIds(path.join(dir, "dtccswaps"));
      for (const id of ["1", "2", "3", "4", "5", "6"]) assert.ok(seen.has(id), `id ${id} is archived`);
      assert.deepEqual(dayFileIds(dir, "2026-08-25").sort(), ["3", "4", "5", "6"], "refilled rows land in the day file of the day the catch-up ran");

      const refills = audits.filter((a) => a.startsWith("DTCC-SOURCE-DAY-REFILLED"));
      assert.equal(refills.length, 1);
      assert.match(refills[0], /vendor file dated 2026-08-23 held 2 of 2 US-underlier row\(s\) the archive did not/);

      const days = readDtccSourceDays(dir);
      assert.equal(days["2026-08-24"].via, "poll");
      assert.equal(days["2026-08-23"].via, "catchup");
      assert.equal(days["2026-08-23"].newRows, 2);
      assert.equal(days["2026-08-22"].via, "poll");
      assert.equal(days["2026-08-22"].reads, 1, "a day the ledger holds is not fetched again");
      assert.equal(v.asked.filter((d) => d === "2026-08-22").length, 0);
      assert.equal(days["2026-08-21"].unavailable, 403, "the second catch-up read of this poll: not served, recorded as such");

      assert.deepEqual(dtccSourceDaysSummary(dir), {
        enabled: true, floor: "2026-08-21", through: "2026-08-24", expected: 4, read: 3, unavailable: 1, pending: 0,
        refilledDays: 1, refilledRows: 2, lastRefilled: { date: "2026-08-23", at: "2026-08-25T01:00:00.000Z", newRows: 2 },
        edge: { enabled: false, days: 0, rows: 0, oldest: null, newest: null },
      });
    } finally {
      setDtccIdsAuditSink(null);
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

test("a day already held costs one read and adds nothing; each poll reads at most the cap and the sweep moves back", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: undefined, [DTCC_EDGE_ENV]: "0" }, async () => {
    const dir = freshDir();
    const audits: string[] = [];
    setDtccIdsAuditSink((type) => audits.push(type));
    try {
      assert.equal(DTCC_CATCHUP_PER_POLL, 2);
      // five vendor days, each already archived by its own poll BEFORE the ledger existed
      const files: Record<string, Buffer> = {};
      for (const [i, d] of ["2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24", "2026-08-25"].entries()) files[d] = buildZip([`${i}a`, `${i}b`]);
      await withEnv({ [DTCC_CATCHUP_ENV]: "0" }, async () => {
        for (const d of Object.keys(files)) await refreshDtccSwaps(vendor({ [d]: files[d] }).fetchImpl, at(`${d}T23:00:00Z`), dir);
      });
      fs.rmSync(path.join(dir, "dtccswaps", "_source_days.json"));

      const v1 = vendor(files);
      await refreshDtccSwaps(v1.fetchImpl, at("2026-08-26T01:00:00Z"), dir);
      assert.deepEqual(v1.asked, ["2026-08-26", "2026-08-25", "2026-08-24", "2026-08-23"], "the walk to the newest file, then two days of catch-up");
      let s = dtccSourceDaysSummary(dir);
      assert.deepEqual([s.read, s.pending, s.refilledDays, s.refilledRows], [3, 2, 0, 0]);

      const v2 = vendor(files);
      await refreshDtccSwaps(v2.fetchImpl, at("2026-08-26T07:00:00Z"), dir);
      assert.deepEqual(v2.asked, ["2026-08-26", "2026-08-25", "2026-08-22", "2026-08-21"]);
      s = dtccSourceDaysSummary(dir);
      assert.deepEqual([s.expected, s.read, s.pending, s.unavailable, s.refilledDays], [5, 5, 0, 0, 0]);
      assert.equal(s.lastRefilled, null);

      const v3 = vendor(files);
      await refreshDtccSwaps(v3.fetchImpl, at("2026-08-26T13:00:00Z"), dir);
      assert.deepEqual(v3.asked, ["2026-08-26", "2026-08-25"], "the sweep is done: only the poll's own walk");
      assert.equal(latestDtcc()!.totalArchived, 10);
      assert.deepEqual(audits, [], "no refill, no error");
      assert.equal(readDtccSourceDays(dir)["2026-08-25"].reads, 3, "the newest file is read by every poll, as before");
    } finally {
      setDtccIdsAuditSink(null);
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

test("off switch: no catch-up read is made; the poll still records its own file", async () => {
  for (const off of ["0", "off", "false"]) assert.equal(dtccCatchupEnabled({ [DTCC_CATCHUP_ENV]: off }), false);
  assert.equal(dtccCatchupEnabled({}), true);
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: "0", [DTCC_EDGE_ENV]: "0" }, async () => {
    const dir = freshDir();
    try {
      const v = vendor({ "2026-08-23": buildZip(["3"]), "2026-08-24": buildZip(["5"]) });
      await refreshDtccSwaps(v.fetchImpl, at("2026-08-24T23:00:00Z"), dir);
      assert.deepEqual(v.asked, ["2026-08-24"]);
      const s = dtccSourceDaysSummary(dir);
      assert.equal(s.enabled, false);
      assert.deepEqual([s.through, s.expected, s.read, s.pending], ["2026-08-24", 4, 1, 3]);
    } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  });
});

test("a vendor file cut off mid-body is NOT recorded as read: the poll's result stands and the day is asked again", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: undefined, [DTCC_EDGE_ENV]: "0" }, async () => {
    const dir = freshDir();
    try {
      const many = Array.from({ length: 400 }, (_, i) => `t${i}`);
      const whole = buildZip(many);
      const cut = buildZip(many, 200);
      assert.ok(cut.length < whole.length);
      const v = vendor({ "2026-08-24": buildZip(["5"]), "2026-08-23": cut });
      await refreshDtccSwaps(v.fetchImpl, at("2026-08-24T23:00:00Z"), dir);
      assert.equal(latestDtcc()!.sourceDate, "2026-08-24");
      assert.equal(latestDtcc()!.newRows, 1);
      const days = readDtccSourceDays(dir);
      assert.equal(days["2026-08-23"], undefined, "a partial read is no read");
      assert.deepEqual(dayFileIds(dir, "2026-08-24"), ["5"], "none of the partial file's rows were archived");

      const v2 = vendor({ "2026-08-24": buildZip(["5"]), "2026-08-23": whole });
      await refreshDtccSwaps(v2.fetchImpl, at("2026-08-25T05:00:00Z"), dir);
      assert.equal(readDtccSourceDays(dir)["2026-08-23"].newRows, 400);
    } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  });
});

test("a day whose rows could not be written is not recorded as read", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: undefined, [DTCC_EDGE_ENV]: "0" }, async () => {
    const dir = freshDir();
    const audits: string[] = [];
    setDtccIdsAuditSink((type) => audits.push(type));
    try {
      const files = { "2026-08-24": buildZip(["5"]), "2026-08-23": buildZip(["3"]) };
      await withEnv({ [DTCC_CATCHUP_ENV]: "0" }, () => refreshDtccSwaps(vendor(files).fetchImpl, at("2026-08-24T23:00:00Z"), dir));
      // the next day's file cannot be written: its path is a directory. The
      // poll's own file adds nothing (held), so the first write is the catch-up's.
      fs.mkdirSync(path.join(dir, "dtccswaps", "2026-08-25.jsonl.gz"));
      await refreshDtccSwaps(vendor(files).fetchImpl, at("2026-08-25T05:00:00Z"), dir);
      assert.deepEqual(audits, ["DTCC-ARCHIVE-ERROR"]);
      assert.equal(readDtccSourceDays(dir)["2026-08-23"], undefined);
      assert.equal(dtccSourceDaysSummary(dir).refilledDays, 0);
    } finally {
      setDtccIdsAuditSink(null);
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

test("a ledger that is not JSON is treated as empty: days are read again and add nothing", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: undefined, [DTCC_EDGE_ENV]: "0" }, async () => {
    const dir = freshDir();
    try {
      const files = { "2026-08-24": buildZip(["5"]), "2026-08-23": buildZip(["3"]) };
      await refreshDtccSwaps(vendor(files).fetchImpl, at("2026-08-24T23:00:00Z"), dir);
      assert.equal(latestDtcc()!.totalArchived, 2);
      fs.writeFileSync(path.join(dir, "dtccswaps", "_source_days.json"), "{not json");
      assert.deepEqual(readDtccSourceDays(dir), {});
      await refreshDtccSwaps(vendor(files).fetchImpl, at("2026-08-25T05:00:00Z"), dir);
      assert.equal(latestDtcc()!.totalArchived, 2, "no row twice");
      const days = readDtccSourceDays(dir);
      assert.equal(days["2026-08-23"].via, "catchup");
      assert.equal(days["2026-08-23"].newRows, 0);
    } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  });
});
