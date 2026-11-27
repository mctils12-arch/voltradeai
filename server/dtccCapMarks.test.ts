// dtccCapMarks.test.ts — [PIPELINE] 2026-11-27, open_questions.md (bc1) and
// (bc4). Until v1.0.1572 the DTCC swaps parser read the source's cap mark
// ("250,000,000+") as a malformed number and archived null, on every day file
// from 2026-08-22. These tests pin the record that holds the mark per vendor
// day, the re-read of the days read before, and what happens to a day that
// left the vendor's two-year window first. The first test reads REAL vendor
// lines (server/__fixtures__/dtcc_swaps, written by
// scripts/dtcc_cap_marks_seed.ts --sample): for three months every fixture row
// was typed by a session and none carried a "+".
// Run: npx tsx --test server/dtccCapMarks.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import {
  REQUIRED_COLUMNS, parseHeader, parseCsvLine, parseDtccLine, dtccNotionalClass, refreshDtccSwaps, setDtccIdsAuditSink,
  _resetDtccForTests, _setDtccCapMarksSeedForTests, readDtccSourceDays, dtccCapMarksSummary,
  DTCC_NOTIONAL_CAP_USD, DTCC_CATCHUP_ENV, DTCC_EDGE_ENV, DTCC_EDGE_OLDEST_DAYS, DTCC_CATCHUP_FLOOR, type DtccSourceDays,
} from "./dtccSwaps";
import {
  foldDtccCapMarks, readDtccCapMarks, recordDtccCapMarks, recordDtccCapMarksUnread, dtccCapMarkCandidates,
  dtccDayPastVendorWindow, summarizeDtccCapMarks, dtccCappedIdSet, dtccNullNotionalMeaning, applyDtccCapMarksSeed,
  dtccCapMarksEnabled, DTCC_CAP_MARKS_FILE, DTCC_CAP_MARKS_PER_POLL, DTCC_CAP_MARKS_ATTEMPTS_PER_POLL,
  DTCC_CAP_MARKS_RETRY_MS, DTCC_CAP_MARKS_ENV, type DtccCapMarks,
} from "./dtccCapMarks";
import seedJson from "../datacore/dtcc_cap_marks_seed.json";

const here = path.dirname(fileURLToPath(import.meta.url));
const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const HEADER_COLS = [...REQUIRED_COLUMNS];
const HEADER_LINE = HEADER_COLS.map((c) => `"${c}"`).join(",");
const at = (iso: string) => Date.parse(iso);

function csvRow(diss: string, notional = "1,000,000"): string {
  const byName: Record<string, string> = {
    "Dissemination Identifier": diss, "Original Dissemination Identifier": diss, "Action type": "NEWT",
    "Event timestamp": "2026-08-21T12:33:36Z", "Effective Date": "2026-08-21", "Asset Class": "EQ",
    "Notional amount-Leg 1": notional, "Notional currency-Leg 1": "USD",
    "Underlier ID-Leg 1": "US0378331005", "Underlier ID source-Leg 1": "ISIN", "UPI Underlier Name": "APPLE INC",
  };
  return HEADER_COLS.map((c) => `"${byName[c] ?? ""}"`).join(",");
}

/** A vendor file: `plain` rows at 1,000,000 and `capped` rows written "250,000,000+". */
function buildZip(plain: string[], capped: string[] = []): Buffer {
  const nameBuf = Buffer.from("SEC_CUMULATIVE_EQUITIES.csv", "utf8");
  const lines = [HEADER_LINE, ...plain.map((id) => csvRow(id)), ...capped.map((id) => csvRow(id, "250,000,000+"))];
  const dataBuf = Buffer.from(lines.join("\n") + "\n", "utf8");
  const compressed = zlib.deflateRawSync(dataBuf);
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(20, 4);
  header.writeUInt16LE(8, 8);
  header.writeUInt32LE(compressed.length, 18);
  header.writeUInt32LE(dataBuf.length, 22);
  header.writeUInt16LE(nameBuf.length, 26);
  return Buffer.concat([header, nameBuf, compressed]);
}

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
  return fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-capmarks-"));
}

const streamDir = (dir: string) => path.join(dir, "dtccswaps");
const heldRead = (iso: string, via: "poll" | "catchup" | "edge" = "poll"): DtccSourceDays[string] => ({ firstAt: iso, lastAt: iso, via, usRows: 2, newRows: 2, reads: 1 });

/** The state a server of before 2026-11-27 left: a ledger of days read, no cap-mark record. */
function writeLedger(dir: string, days: DtccSourceDays): void {
  fs.mkdirSync(streamDir(dir), { recursive: true });
  fs.writeFileSync(path.join(streamDir(dir), "_source_days.json"), JSON.stringify({ version: 1, days }));
}

function dayFileRowCount(dir: string): number {
  let n = 0;
  for (const f of fs.readdirSync(streamDir(dir))) {
    if (!/^\d{4}-\d{2}-\d{2}\.jsonl\.gz$/.test(f)) continue;
    n += zlib.gunzipSync(fs.readFileSync(path.join(streamDir(dir), f))).toString("utf8").split("\n").filter(Boolean).length;
  }
  return n;
}

test("REAL VENDOR LINES: one line of each notional form, as the source wrote it on 2024-11-26, reads as its class", () => {
  const lines = fs.readFileSync(path.join(here, "__fixtures__", "dtcc_swaps", "vendor_lines_2024-11-26.csv"), "utf8").split("\n").filter(Boolean);
  const idx = parseHeader(lines[0]);
  const cols = parseCsvLine(lines[0]).length;
  const rows = lines.slice(1).map((l) => parseDtccLine(l, idx, cols));
  assert.ok(rows.every(Boolean), "every sample line is a US-underlier row the parser accepts");
  assert.deepEqual(rows.map((r) => dtccNotionalClass(r!)), ["usd", "usd_capped", "other_currency", "over_cap_unmarked", "blank"]);
  const raw = lines.slice(1).map((l) => parseCsvLine(l)[idx["Notional amount-Leg 1"]]);
  assert.equal(raw[1], "250,000,000+", "the mark as the source writes it");
  assert.equal(rows[1]!.notionalAmountLeg1, DTCC_NOTIONAL_CAP_USD);
  assert.equal(rows[1]!.notionalCappedLeg1, true);
  assert.equal(rows[0]!.notionalAmountLeg1, 5_000_000);
  assert.equal(rows[0]!.notionalCappedLeg1, undefined, "an uncapped row carries no flag");
  assert.equal(rows[4]!.notionalAmountLeg1, null, "a blank stays null, never 0");
  assert.equal(raw[4], "");
});

test("the record folds line by line: a reading is final, lost is final unless read, unread is replaced", () => {
  const text = [
    JSON.stringify({ day: "2026-08-22", at: "2026-11-27T00:00:00.000Z", unread: 500 }),
    JSON.stringify({ day: "2026-08-22", at: "2026-11-28T00:00:00.000Z", via: "reread", usRows: 9, ids: ["7", "8"] }),
    JSON.stringify({ day: "2026-08-22", at: "2026-11-29T00:00:00.000Z", lost: 403 }),
    JSON.stringify({ day: "2024-11-24", at: "2026-11-27T00:00:00.000Z", unread: 403 }),
    JSON.stringify({ day: "2024-11-24", at: "2026-11-27T06:00:00.000Z", lost: 403 }),
    JSON.stringify({ day: "2024-11-24", at: "2026-11-27T12:00:00.000Z", unread: 0 }),
    JSON.stringify({ day: "2026-08-23", at: "2026-11-27T00:00:00.000Z", via: "poll", usRows: 4, ids: [] }),
    "{not json",
    JSON.stringify({ day: "nope", at: "x", ids: [] }),
    JSON.stringify({ day: "2026-08-24", at: "2026-11-27T00:00:00.000Z" }),
    "",
  ].join("\n");
  const { days, malformed } = foldDtccCapMarks(text);
  assert.equal(malformed, 3);
  assert.deepEqual(days["2026-08-22"], { at: "2026-11-28T00:00:00.000Z", via: "reread", usRows: 9, ids: ["7", "8"] });
  assert.deepEqual(days["2024-11-24"], { at: "2026-11-27T06:00:00.000Z", lost: 403 });
  assert.deepEqual(days["2026-08-23"].ids, [], "an empty list is a reading: no row carried the mark");
  assert.equal(days["2026-08-24"], undefined);
});

test("recording: IDs are stored once and sorted; the same file read again writes nothing", () => {
  const dir = streamDir(freshDir());
  assert.equal(recordDtccCapMarks(dir, "2026-11-26", "poll", 10, ["9", "3", "9"], at("2026-11-27T01:00:00Z")), true);
  assert.equal(recordDtccCapMarks(dir, "2026-11-26", "poll", 10, ["3", "9"], at("2026-11-27T07:00:00Z")), false, "the poll reads the newest file several times a day");
  assert.equal(fs.readFileSync(path.join(dir, DTCC_CAP_MARKS_FILE), "utf8").split("\n").filter(Boolean).length, 1);
  assert.deepEqual(readDtccCapMarks(dir)["2026-11-26"].ids, ["3", "9"]);
  assert.equal(recordDtccCapMarks(dir, "2026-11-26", "poll", 11, ["3", "9", "12"], at("2026-11-27T13:00:00Z")), true, "a file that grew is recorded again");
  assert.deepEqual(readDtccCapMarks(dir)["2026-11-26"].ids, ["12", "3", "9"]);
  assert.equal(recordDtccCapMarks(dir, "not-a-day", "poll", 1, [], 0), false);
  assert.deepEqual(readDtccCapMarks(streamDir(freshDir())), {}, "no record yet is an empty one");
});

test("candidates: ledger days held as read with no reading, oldest first; a refused edge day is asked by every poll, a later day after the retry interval", () => {
  const t0 = "2026-11-27T00:00:00.000Z";
  const ledger: DtccSourceDays = {
    "2024-11-26": heldRead(t0, "edge"), "2024-11-27": heldRead(t0, "edge"),
    "2026-08-22": heldRead(t0), "2026-08-23": heldRead(t0), "2026-08-24": heldRead(t0),
    "2026-08-25": { ...heldRead(t0), unavailable: 403 },
  };
  assert.deepEqual(dtccCapMarkCandidates(ledger, {}, at(t0), DTCC_CATCHUP_FLOOR), ["2024-11-26", "2024-11-27", "2026-08-22", "2026-08-23", "2026-08-24"], "a day the vendor never served is not owed a reading");
  const marks: DtccCapMarks = {
    "2024-11-26": { at: t0, lost: 403 },
    "2024-11-27": { at: t0, unread: 500 },
    "2026-08-22": { at: t0, via: "reread", usRows: 2, ids: [] },
    "2026-08-23": { at: t0, unread: 500 },
  };
  assert.deepEqual(dtccCapMarkCandidates(ledger, marks, at(t0) + 1, DTCC_CATCHUP_FLOOR), ["2024-11-27", "2026-08-24"]);
  assert.deepEqual(dtccCapMarkCandidates(ledger, marks, at(t0) + DTCC_CAP_MARKS_RETRY_MS, DTCC_CATCHUP_FLOOR), ["2024-11-27", "2026-08-23", "2026-08-24"]);
});

test("the vendor's window: on UTC day T the file dated T-731 is the oldest inside it", () => {
  const now = at("2026-11-27T18:00:00Z");
  assert.equal(DTCC_EDGE_OLDEST_DAYS, 731);
  assert.equal(dtccDayPastVendorWindow("2024-11-26", now, DTCC_EDGE_OLDEST_DAYS), false);
  assert.equal(dtccDayPastVendorWindow("2024-11-25", now, DTCC_EDGE_OLDEST_DAYS), true, "refused from a session machine at 18:07Z on 2026-11-27");
  assert.equal(dtccDayPastVendorWindow("2024-11-26", at("2026-11-28T00:00:01Z"), DTCC_EDGE_OLDEST_DAYS), true);
});

test("summary and the reader's rule: a null is capped, blank or unknown, and only a reading can say blank", () => {
  const t0 = "2026-11-27T00:00:00.000Z";
  const ledger: DtccSourceDays = {
    "2024-11-25": heldRead(t0, "edge"), "2026-08-22": heldRead(t0), "2026-08-23": heldRead(t0),
    "2026-08-24": { ...heldRead(t0), unavailable: 403 },
  };
  const marks: DtccCapMarks = { "2024-11-25": { at: t0, lost: 403 }, "2026-08-22": { at: t0, via: "reread", usRows: 5, ids: ["41", "42"] } };
  assert.deepEqual(summarizeDtccCapMarks(ledger, marks, true), {
    enabled: true, days_read: 3, recorded: 1, owed: 1, lost: 1, lost_days: ["2024-11-25"], capped_events: 2,
  });
  assert.deepEqual([...dtccCappedIdSet(marks)].sort(), ["41", "42"]);
  assert.equal(dtccNullNotionalMeaning("41", "2026-08-22", marks), "capped");
  assert.equal(dtccNullNotionalMeaning("41", null, marks), "capped", "the ID alone is enough for a capped row");
  assert.equal(dtccNullNotionalMeaning("50", "2026-08-22", marks), "blank");
  assert.equal(dtccNullNotionalMeaning("50", "2024-11-25", marks), "unknown", "a lost day cannot say blank");
  assert.equal(dtccNullNotionalMeaning("50", "2026-08-23", marks), "unknown", "neither can a day still owed");
  assert.equal(dtccNullNotionalMeaning("50", null, marks), "unknown");
});

test("the seed: applied once, never over a reading; the bundled seed is well formed", () => {
  const dir = streamDir(freshDir());
  recordDtccCapMarks(dir, "2024-11-27", "reread", 3, ["1"], at("2026-11-27T00:00:00Z"));
  const seed = { days: { "2024-11-26": { usRows: 5, ids: ["b", "a"], readAt: "2026-11-27T18:30:00.000Z" }, "2024-11-27": { usRows: 3, ids: ["zzz"], readAt: "x" } } };
  assert.deepEqual(applyDtccCapMarksSeed(dir, seed, at("2026-11-27T19:00:00Z")), ["2024-11-26"]);
  assert.deepEqual(applyDtccCapMarksSeed(dir, seed, at("2026-11-27T20:00:00Z")), [], "a second boot writes nothing");
  const marks = readDtccCapMarks(dir);
  assert.deepEqual(marks["2024-11-26"], { at: "2026-11-27T19:00:00.000Z", via: "seed", usRows: 5, ids: ["a", "b"] });
  assert.deepEqual(marks["2024-11-27"].ids, ["1"], "the server's own reading stands");
  assert.deepEqual(applyDtccCapMarksSeed(dir, null, 0), []);

  const bundled = (seedJson as any).days as Record<string, { usRows: number; ids: string[]; readAt: string }>;
  const days = Object.keys(bundled);
  assert.ok(days.includes("2024-11-26"), "the day that was hours from leaving the vendor when this shipped");
  for (const d of days) {
    assert.match(d, /^\d{4}-\d{2}-\d{2}$/);
    assert.ok(bundled[d].usRows > 0 && bundled[d].ids.length <= bundled[d].usRows);
    assert.ok(bundled[d].ids.every((x) => /^\d+$/.test(x)), "Dissemination Identifiers");
    assert.deepEqual(bundled[d].ids, [...new Set(bundled[d].ids)].sort());
  }
});

test("off switch", () => {
  assert.equal(dtccCapMarksEnabled({}), true);
  for (const v of ["0", "off", "FALSE"]) assert.equal(dtccCapMarksEnabled({ [DTCC_CAP_MARKS_ENV]: v }), false);
});

test("THE GAP: days read before the parser knew the mark are re-read oldest first; a day past the vendor's window is recorded lost, audited once and never asked again; a re-read archives nothing", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: "0", [DTCC_EDGE_ENV]: "0", [DTCC_CAP_MARKS_ENV]: undefined }, async () => {
    const dir = freshDir();
    _setDtccCapMarksSeedForTests(null);
    const audits: string[] = [];
    setDtccIdsAuditSink((type, message) => audits.push(`${type}: ${message}`));
    try {
      const t0 = "2026-11-26T01:00:00.000Z";
      // what a server of 2026-11-26 held: three days read, no record of marks
      writeLedger(dir, { "2024-11-25": heldRead(t0, "edge"), "2024-11-26": heldRead(t0, "edge"), "2026-11-24": heldRead(t0, "catchup") });
      const files = {
        "2024-11-26": buildZip(["100", "101"], ["102"]),
        "2026-11-24": buildZip(["200"], []),
        "2026-11-26": buildZip(["300", "301"], ["302", "303"]),
      };
      const now = at("2026-11-27T18:00:00Z");
      const v = vendor(files);
      await refreshDtccSwaps(v.fetchImpl, now, dir);

      assert.deepEqual(v.asked, ["2026-11-27", "2026-11-26", "2024-11-25", "2024-11-26", "2026-11-24"], "the poll's walk, then the owed days oldest first");
      const marks = readDtccCapMarks(streamDir(dir));
      assert.deepEqual(marks["2026-11-26"], { at: new Date(now).toISOString(), via: "poll", usRows: 4, ids: ["302", "303"] }, "the poll's own file records its marks as it is read");
      assert.deepEqual(marks["2024-11-26"], { at: new Date(now).toISOString(), via: "reread", usRows: 3, ids: ["102"] });
      assert.deepEqual(marks["2026-11-24"].ids, [], "a day with no capped row has a reading that says so");
      assert.deepEqual(marks["2024-11-25"], { at: new Date(now).toISOString(), lost: 403 });
      const lost = audits.filter((a) => a.startsWith("DTCC-CAP-MARK-LOST"));
      assert.equal(lost.length, 1);
      assert.match(lost[0], /2024-11-25/);
      assert.equal(dayFileRowCount(dir), 4, "only the poll's four rows are in a day file: a re-read keeps IDs, not rows");
      assert.deepEqual(Object.keys(readDtccSourceDays(dir)).sort(), ["2024-11-25", "2024-11-26", "2026-11-24", "2026-11-26"], "a re-read does not touch the source-day ledger");
      assert.deepEqual(dtccCapMarksSummary(dir), { enabled: true, days_read: 4, recorded: 3, owed: 0, lost: 1, lost_days: ["2024-11-25"], capped_events: 3 });

      // six hours later: nothing is owed, the lost day is not asked again
      const v2 = vendor(files);
      await refreshDtccSwaps(v2.fetchImpl, now + 6 * 3_600_000, dir);
      assert.deepEqual(v2.asked, ["2026-11-28", "2026-11-27", "2026-11-26"]);
      assert.equal(audits.filter((a) => a.startsWith("DTCC-CAP-MARK-LOST")).length, 1, "audited once");
    } finally {
      setDtccIdsAuditSink(null);
      _setDtccCapMarksSeedForTests(seedJson as any);
    }
  });
});

test("a refused day inside the window is unread, not lost, and does not use a download's turn; downloads per poll are capped", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: "0", [DTCC_EDGE_ENV]: "0", [DTCC_CAP_MARKS_ENV]: undefined }, async () => {
    const dir = freshDir();
    _setDtccCapMarksSeedForTests(null);
    try {
      const t0 = "2026-11-26T01:00:00.000Z";
      const ledger: DtccSourceDays = {};
      for (const d of ["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22"]) ledger[d] = heldRead(t0);
      writeLedger(dir, ledger);
      const files: Record<string, Buffer> = {
        "2026-11-19": buildZip(["1"], ["2"]), "2026-11-20": buildZip(["3"]), "2026-11-21": buildZip(["4"]), "2026-11-22": buildZip(["5"]),
        "2026-11-26": buildZip(["9"]),
      };
      assert.equal(DTCC_CAP_MARKS_PER_POLL, 2);
      assert.ok(DTCC_CAP_MARKS_ATTEMPTS_PER_POLL > DTCC_CAP_MARKS_PER_POLL);
      const now = at("2026-11-27T18:00:00Z");
      const v = vendor(files);
      await refreshDtccSwaps(v.fetchImpl, now, dir);
      assert.deepEqual(v.asked.slice(2), ["2026-11-18", "2026-11-19", "2026-11-20"], "the refusal of 11-18 did not use one of the two downloads");
      let marks = readDtccCapMarks(streamDir(dir));
      assert.deepEqual(marks["2026-11-18"], { at: new Date(now).toISOString(), unread: 403 });
      assert.deepEqual(marks["2026-11-19"].ids, ["2"]);
      assert.equal(dtccCapMarksSummary(dir).owed, 3, "11-18 (unread), 11-21 and 11-22");

      const v2 = vendor(files);
      await refreshDtccSwaps(v2.fetchImpl, now + 6 * 3_600_000, dir);
      assert.deepEqual(v2.asked.slice(3), ["2026-11-21", "2026-11-22"], "11-18 waits for its retry interval");
      marks = readDtccCapMarks(streamDir(dir));
      assert.equal(dtccCapMarksSummary(dir).owed, 1);
    } finally {
      _setDtccCapMarksSeedForTests(seedJson as any);
    }
  });
});

test("switched off: no day is re-read and no seed is applied; a file the poll reads still records its marks", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: "0", [DTCC_EDGE_ENV]: "0", [DTCC_CAP_MARKS_ENV]: "0" }, async () => {
    const dir = freshDir();
    writeLedger(dir, { "2026-11-24": heldRead("2026-11-26T01:00:00.000Z") });
    const v = vendor({ "2026-11-24": buildZip(["1"], ["2"]), "2026-11-26": buildZip(["3"], ["4"]) });
    await refreshDtccSwaps(v.fetchImpl, at("2026-11-27T18:00:00Z"), dir);
    assert.deepEqual(v.asked, ["2026-11-27", "2026-11-26"]);
    const marks = readDtccCapMarks(streamDir(dir));
    assert.deepEqual(Object.keys(marks), ["2026-11-26"]);
    assert.deepEqual(marks["2026-11-26"].ids, ["4"]);
    assert.equal(dtccCapMarksSummary(dir).enabled, false);
  });
});

test("the catch-up's and the edge's own reads record their marks, so a day read from now on is never owed", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_CATCHUP_ENV]: undefined, [DTCC_EDGE_ENV]: undefined, [DTCC_CAP_MARKS_ENV]: undefined }, async () => {
    const dir = freshDir();
    _setDtccCapMarksSeedForTests(null);
    try {
      const now = at("2026-11-27T18:00:00Z");
      const v = vendor({
        "2026-11-26": buildZip(["1"], ["2"]), "2026-11-25": buildZip(["3"], ["4", "5"]),
        "2024-11-27": buildZip(["6"], ["7"]),
      });
      await refreshDtccSwaps(v.fetchImpl, now, dir);
      const marks = readDtccCapMarks(streamDir(dir));
      assert.equal(marks["2026-11-25"].via, "catchup");
      assert.deepEqual(marks["2026-11-25"].ids, ["4", "5"]);
      assert.equal(marks["2024-11-27"].via, "edge");
      assert.deepEqual(marks["2024-11-27"].ids, ["7"]);
      assert.equal(v.asked.filter((d) => d === "2026-11-25").length, 1, "read once, not once more for its marks");
      assert.equal(dtccCapMarksSummary(dir).owed, 0);
    } finally {
      _setDtccCapMarksSeedForTests(seedJson as any);
    }
  });
});
