// dtccOffsiteDays.test.ts — [PIPELINE] 2026-11-25, open_questions (aa4): the
// DTCC swaps day files move to the permanent R2 tier (archivePermanentTier.ts,
// 14 days kept local). The seen-ID rebuild — the scan that runs only when the
// dedup index is missing or unreadable — reads LOCAL day files, so with days
// off-site it would build a set short of every older ID and write it as the
// index. These tests pin what was built against that:
//   - the rebuild records the off-site day files as OWED before it writes the
//     index, in both loaders;
//   - the next poll reads them back through the tier's cold source, gives the
//     index exactly the IDs it lacks, and archives none of their rows again
//     ((aa4)'s required test: an unreadable index plus evicted days
//     re-archives nothing) — in memory, dual and sqlite modes;
//   - with the off-site tier unreachable the poll still archives (an unarchived
//     day does not refill), says what is owed, and the following poll settles
//     it with no ID in the index twice;
//   - an owed record that cannot be written stops the rebuild before the index.
// The real tier (upload, read-back, eviction, cold read) runs against these
// same functions in archivePermanentTier.test.ts.
// Run: npx tsx --test server/dtccOffsiteDays.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  REQUIRED_COLUMNS, parseHeader, parseDtccLine, type DtccSwapRow,
  archiveNewRows, archiveNewRowsAsync, loadSeenIds, loadSeenIdsAsync, dtccOwedDayFiles,
  setDtccIdsAuditSink, _resetDtccForTests, _setDtccFreeBytesReaderForTests,
} from "./dtccSwaps";
import { setColdDayFileSource, type ColdDayFileSource } from "./datacoreArchive";
import { savePermanentManifest, permanentKey, type PermanentManifest } from "./archivePermanentTier";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const INDEX = "_seen_ids_index.jsonl.gz";
const OWED = "_seen_ids_owed.json";
const STREAM = "dtccswaps";

/** Live-shaped Dissemination Identifiers (19-digit canonical decimals). */
const idOf = (day: number, i: number) => `61${String(day).padStart(3, "0")}${String(i).padStart(12, "0")}01`;

function rowsOf(ids: readonly string[]): DtccSwapRow[] {
  const cols = [...REQUIRED_COLUMNS];
  const idx = parseHeader(cols.map((c) => `"${c}"`).join(","));
  return ids.map((d) => {
    const byName: Record<string, string> = {
      "Dissemination Identifier": d, "Original Dissemination Identifier": d, "Action type": "NEWT",
      "Event timestamp": "2026-11-05T12:33:36Z", "Effective Date": "2026-11-05", "Asset Class": "EQ",
      "Notional amount-Leg 1": "28,000,000", "Notional currency-Leg 1": "USD", "Underlier ID-Leg 1": "US0378331005",
      "Underlier ID source-Leg 1": "ISIN", "UPI Underlier Name": "APPLE INC",
    };
    return parseDtccLine(cols.map((c) => `"${byName[c] ?? ""}"`).join(","), idx, cols.length)!;
  });
}

const quiet = () => {
  const e = console.error, l = console.log;
  console.error = () => {};
  console.log = () => {};
  return () => { console.error = e; console.log = l; };
};

const DAYS = ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"];
const PER_DAY = 40;
const idsOfDay = (d: number) => Array.from({ length: PER_DAY }, (_, i) => idOf(d + 1, i));
const ALL_IDS = DAYS.flatMap((_, d) => idsOfDay(d));

interface Fixture {
  base: string;
  dir: string;
  cold: string;
  /** the fake off-site tier: which of its files it will hand back */
  reachable: { on: boolean };
  resolved: string[];
  audit: Array<[string, string]>;
  done: () => void;
}

/** Five archived days written by the real archiver; the first `offsite` of
 *  them then moved off the volume the way the tier leaves things: the file
 *  gone locally, an evicted entry in the tier's manifest, the bytes behind a
 *  cold source. */
function fixture(mode: "memory" | "dual" | "sqlite", offsite = 3): Fixture {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-offsite-"));
  const dir = path.join(base, STREAM);
  const cold = path.join(base, "..", `${path.basename(base)}-cold`);
  fs.mkdirSync(cold, { recursive: true });
  const prev = process.env[IDS_ENV];
  process.env[IDS_ENV] = mode;
  _resetDtccForTests();
  _setDtccFreeBytesReaderForTests(() => null); // free space unknown: the dedup DB is allowed
  const restore = quiet();
  DAYS.forEach((day, d) => {
    assert.equal(archiveNewRows(rowsOf(idsOfDay(d)), day, base).newRows, PER_DAY);
  });
  const manifest: PermanentManifest = { version: 1, entries: {}, conflicts: {}, missing: [], lastReconcileAt: null };
  for (const day of DAYS.slice(0, offsite)) {
    const file = `${day}.jsonl.gz`;
    const bytes = fs.readFileSync(path.join(dir, file));
    fs.writeFileSync(path.join(cold, file), bytes);
    fs.rmSync(path.join(dir, file));
    manifest.entries[permanentKey(STREAM, file)] = {
      stream: STREAM, file, date: day, bytes: bytes.length, sha256: "x", lines: PER_DAY,
      src: "uploaded", srcMtimeMs: 1, at: 1, evictedAt: 2,
    };
  }
  // another stream's evicted day must never be owed to this index
  manifest.entries[permanentKey("occvolume", "2026-10-01.jsonl.gz")] = {
    stream: "occvolume", file: "2026-10-01.jsonl.gz", date: "2026-10-01", bytes: 1, sha256: "x", lines: 1,
    src: "uploaded", srcMtimeMs: 1, at: 1, evictedAt: 2,
  };
  savePermanentManifest(base, manifest);
  const reachable = { on: true };
  const resolved: string[] = [];
  const source: ColdDayFileSource = {
    async resolve(stream, day) {
      resolved.push(`${stream}/${day}`);
      const fp = path.join(cold, `${day}.jsonl.gz`);
      return reachable.on && stream === STREAM && fs.existsSync(fp) ? [fp] : [];
    },
    heldOnlyInR2: () => [],
  };
  setColdDayFileSource(source);
  const audit: Array<[string, string]> = [];
  setDtccIdsAuditSink((action, detail) => { audit.push([action, detail]); });
  // a fresh process whose index is gone
  _resetDtccForTests();
  fs.rmSync(path.join(dir, INDEX));
  for (const f of ["_seen_ids.db", "_seen_ids.db-wal", "_seen_ids.db-shm"]) fs.rmSync(path.join(dir, f), { force: true });
  return {
    base, dir, cold, reachable, resolved, audit,
    done: () => {
      restore();
      setDtccIdsAuditSink(null);
      setColdDayFileSource(null);
      _setDtccFreeBytesReaderForTests(null);
      _resetDtccForTests();
      if (prev === undefined) delete process.env[IDS_ENV]; else process.env[IDS_ENV] = prev;
      fs.rmSync(base, { recursive: true, force: true });
      fs.rmSync(cold, { recursive: true, force: true });
    },
  };
}

const indexLines = (dir: string) => zlib.gunzipSync(fs.readFileSync(path.join(dir, INDEX))).toString("utf8").split("\n").filter(Boolean);
const dayRows = (dir: string, day: string) => {
  const fp = path.join(dir, `${day}.jsonl.gz`);
  if (!fs.existsSync(fp)) return [] as string[];
  return zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).disseminationId as string);
};
const OFFSITE_FILES = DAYS.slice(0, 3).map((d) => `${d}.jsonl.gz`);

// ── (1) the rebuild says what it could not scan, before it writes the index ──

test("a rebuild with day files off-site records them as owed (and only them) before writing the index — sync and async alike", async () => {
  for (const loader of ["sync", "async"] as const) {
    const f = fixture("memory");
    try {
      // an entry the tier has uploaded but not evicted, its local file gone
      // anyway (a crash between the unlink and the manifest save): owed too
      const m = JSON.parse(fs.readFileSync(path.join(f.base, "archive_permanent_manifest.json"), "utf8"));
      m.entries[permanentKey(STREAM, OFFSITE_FILES[2])].evictedAt = null;
      // an entry whose local copy is still here is NOT owed: the scan reads it
      m.entries[permanentKey(STREAM, `${DAYS[4]}.jsonl.gz`)] = { ...m.entries[permanentKey(STREAM, OFFSITE_FILES[0])], file: `${DAYS[4]}.jsonl.gz`, date: DAYS[4], evictedAt: null };
      fs.writeFileSync(path.join(f.base, "archive_permanent_manifest.json"), JSON.stringify(m));

      const built = [...(loader === "sync" ? loadSeenIds(f.dir) : await loadSeenIdsAsync(f.dir))];
      const local = [...idsOfDay(3), ...idsOfDay(4)];
      assert.deepEqual(built.slice().sort(), local.slice().sort(), `${loader}: the local days' IDs, as before`);
      assert.deepEqual(dtccOwedDayFiles(f.base), OFFSITE_FILES, `${loader}: the three off-site day files, no other stream's, not the local one`);
      assert.deepEqual(indexLines(f.dir).sort(), local.slice().sort(), `${loader}: the index is written, short by the owed days`);
      const owedRows = f.audit.filter(([a]) => a === "DTCC-IDS-OWED");
      assert.equal(owedRows.length, 1, `${loader}: one audit row`);
      assert.match(owedRows[0][1], /3 day file\(s\) \(2026-10-01\.jsonl\.gz \.\. 2026-10-03\.jsonl\.gz\) are held off-site/);
      assert.ok(owedRows[0][1].length <= 500, "fits the audit_log message cap");
      assert.deepEqual(f.resolved, [], `${loader}: the rebuild itself asks the off-site tier for nothing`);
    } finally {
      f.done();
    }
  }
});

test("no manifest, or nothing off-site: the rebuild is what it was — no owed record, and a stale one is removed", () => {
  const f = fixture("memory", 0);
  try {
    fs.writeFileSync(path.join(f.dir, OWED), JSON.stringify({ version: 1, at: 1, files: ["2026-09-01.jsonl.gz"] }));
    const built = [...loadSeenIds(f.dir)];
    assert.deepEqual(built.slice().sort(), ALL_IDS.slice().sort());
    assert.equal(fs.existsSync(path.join(f.dir, OWED)), false, "every day was scanned locally: nothing is owed");
    assert.deepEqual(indexLines(f.dir).sort(), ALL_IDS.slice().sort());
    assert.deepEqual(f.audit, []);
    fs.rmSync(path.join(f.base, "archive_permanent_manifest.json"));
    fs.rmSync(path.join(f.dir, INDEX));
    _resetDtccForTests();
    assert.equal([...loadSeenIds(f.dir)].length, ALL_IDS.length);
    assert.deepEqual(dtccOwedDayFiles(f.base), []);
  } finally {
    f.done();
  }
});

test("an owed record that cannot be written stops the rebuild: no index is written, the poll archives nothing, the next poll rebuilds again", async () => {
  const f = fixture("memory");
  try {
    fs.mkdirSync(path.join(f.dir, OWED)); // a rename onto a directory fails
    assert.throws(() => loadSeenIds(f.dir), /owed record not written/);
    assert.equal(fs.existsSync(path.join(f.dir, INDEX)), false, "sync: a short index is never written with nothing saying so");
    await assert.rejects(loadSeenIdsAsync(f.dir), /owed record not written/);
    assert.equal(fs.existsSync(path.join(f.dir, INDEX)), false, "async: the same");
    await assert.rejects(archiveNewRowsAsync(rowsOf([idOf(1, 0), idOf(99, 0)]), "2026-11-25", f.base), /owed record not written/);
    assert.deepEqual(dayRows(f.dir, "2026-11-25"), [], "nothing archived by the failed poll");
    assert.ok(f.audit.some(([a, d]) => a === "DTCC-ARCHIVE-ERROR" && /the index is NOT written/.test(d)));
    // the volume takes the record again: the same poll now goes through whole
    fs.rmdirSync(path.join(f.dir, OWED));
    const r = await archiveNewRowsAsync(rowsOf([idOf(1, 0), idOf(99, 0)]), "2026-11-25", f.base);
    assert.deepEqual(r, { newRows: 1, seenTotal: ALL_IDS.length + 1 });
  } finally {
    f.done();
  }
});

// ── (2) (aa4)'s required test ────────────────────────────────────────────────

for (const mode of ["memory", "dual", "sqlite"] as const) {
  test(`[${mode}] an unreadable index plus evicted days re-archives nothing: the poll reads the owed days back, restores their IDs, and writes only the rows never archived`, async () => {
    const f = fixture(mode);
    try {
      // the vendor serves rows of an EVICTED day, of a LOCAL day, and two new ones
      const served = [...idsOfDay(0).slice(0, 10), ...idsOfDay(1).slice(0, 5), ...idsOfDay(4).slice(0, 5), idOf(99, 1), idOf(99, 2)];
      const r = await archiveNewRowsAsync(rowsOf(served), "2026-11-25", f.base);
      assert.deepEqual(r, { newRows: 2, seenTotal: ALL_IDS.length + 2 }, "only the two rows never archived; the count is every ID ever archived");
      assert.deepEqual(dayRows(f.dir, "2026-11-25").sort(), [idOf(99, 1), idOf(99, 2)], "the day file holds the new rows and no row of an evicted day");
      assert.equal(fs.existsSync(path.join(f.dir, OWED)), false, "nothing owed any more");
      const lines = indexLines(f.dir);
      assert.equal(lines.length, ALL_IDS.length + 2, "the index holds every ID once");
      assert.deepEqual(lines.slice().sort(), [...ALL_IDS, idOf(99, 1), idOf(99, 2)].sort());
      assert.deepEqual(f.resolved, DAYS.slice(0, 3).map((d) => `${STREAM}/${d}`), "each owed day asked for once, no other");
      const settled = f.audit.filter(([a]) => a === "DTCC-IDS-OWED-SETTLED");
      assert.equal(settled.length, 1);
      assert.match(settled[0][1], new RegExp(`3 off-site day file\\(s\\) read back, ${3 * PER_DAY} id\\(s\\) restored`));
      assert.deepEqual(f.audit.filter(([a]) => a === "DTCC-IDS-MISMATCH"), [], "the Set and the DB never disagree over the restored IDs");

      // a restart reads the healed index on its fast path: still nothing re-archived
      _resetDtccForTests();
      f.resolved.length = 0;
      const again = await archiveNewRowsAsync(rowsOf(served), "2026-11-26", f.base);
      assert.deepEqual(again, { newRows: 0, seenTotal: ALL_IDS.length + 2 });
      assert.deepEqual(f.resolved, [], "a whole index asks the off-site tier for nothing");
      assert.deepEqual(f.audit.filter(([a]) => a === "DTCC-IDS-MISMATCH"), []);
    } finally {
      f.done();
    }
  });
}

// ── (3) the off-site tier cannot be reached ──────────────────────────────────

test("off-site tier unreachable: the poll still archives and says what is owed; the next poll settles it and no ID is in the index twice", async () => {
  const f = fixture("memory");
  try {
    f.reachable.on = false;
    const served = [...idsOfDay(0).slice(0, 4), ...idsOfDay(4).slice(0, 5), idOf(99, 1)];
    const r = await archiveNewRowsAsync(rowsOf(served), "2026-11-25", f.base);
    // the 4 rows of the unread evicted day are written again: a duplicate, never a gap
    assert.deepEqual(r, { newRows: 5, seenTotal: 2 * PER_DAY + 5 });
    assert.deepEqual(dtccOwedDayFiles(f.base), OFFSITE_FILES, "all three still owed");
    assert.equal(f.resolved.length, 3, "three failures in a row end the asking for this poll");
    const open = f.audit.filter(([a, d]) => a === "DTCC-IDS-OWED" && /still owed/.test(d));
    assert.equal(open.length, 1);
    assert.match(open[0][1], /3 off-site day file\(s\) still owed .*0 read back/);
    assert.ok(open[0][1].length <= 500, "fits the audit_log message cap");

    // one file comes back, then the tier fails again: progress is kept file by file
    f.reachable.on = true;
    fs.renameSync(path.join(f.cold, OFFSITE_FILES[1]), path.join(f.cold, "hidden-1"));
    fs.renameSync(path.join(f.cold, OFFSITE_FILES[2]), path.join(f.cold, "hidden-2"));
    await archiveNewRowsAsync(rowsOf([idOf(99, 1)]), "2026-11-25", f.base);
    assert.deepEqual(dtccOwedDayFiles(f.base), OFFSITE_FILES.slice(1), "the file read back is struck off; the two unread stay");

    fs.renameSync(path.join(f.cold, "hidden-1"), path.join(f.cold, OFFSITE_FILES[1]));
    fs.renameSync(path.join(f.cold, "hidden-2"), path.join(f.cold, OFFSITE_FILES[2]));
    const last = await archiveNewRowsAsync(rowsOf([idOf(99, 1), ...idsOfDay(2).slice(0, 6)]), "2026-11-26", f.base);
    assert.deepEqual(last, { newRows: 0, seenTotal: ALL_IDS.length + 1 }, "every ID ever archived, the 4 duplicated rows counted once");
    assert.deepEqual(dtccOwedDayFiles(f.base), []);
    const lines = indexLines(f.dir);
    assert.equal(new Set(lines).size, lines.length, "no ID appended to the index twice");
    assert.deepEqual(lines.slice().sort(), [...ALL_IDS, idOf(99, 1)].sort());
  } finally {
    f.done();
  }
});

test("an owed record that is malformed counts every off-site day as owed", async () => {
  const f = fixture("memory");
  try {
    loadSeenIds(f.dir);
    fs.writeFileSync(path.join(f.dir, OWED), "{ not json");
    assert.deepEqual(dtccOwedDayFiles(f.base), OFFSITE_FILES);
    _resetDtccForTests();
    const r = await archiveNewRowsAsync(rowsOf(idsOfDay(0).slice(0, 3)), "2026-11-25", f.base);
    assert.deepEqual(r, { newRows: 0, seenTotal: ALL_IDS.length });
    assert.equal(fs.existsSync(path.join(f.dir, OWED)), false);
  } finally {
    f.done();
  }
});
