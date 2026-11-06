// dtccSeenRebuild.test.ts — [REPAIR] 2026-11-06: the LEGACY seen-ID rebuild
// (loadSeenIds / loadSeenIdsAsync's scan of the day files, used only when the
// compact index is missing or unreadable) reads each day file on its own.
// Before, one try wrapped the whole loop: the first unreadable day file ended
// the scan, so every LATER day's IDs were missing from the rebuilt set — and
// that partial set was then written as the index, so it stuck — and the next
// poll re-archived those days' rows as duplicates (the upstream file re-serves
// every row daily). Now an unreadable file is skipped, counted and reported
// (DTCC-ARCHIVE-ERROR through the audit sink), the scan goes on, and a file
// adds its IDs only once it has been read to its end — in both loaders alike.
// Run: npx tsx --test server/dtccSeenRebuild.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  REQUIRED_COLUMNS, parseHeader, parseDtccLine, type DtccSwapRow,
  archiveNewRows, loadSeenIds, loadSeenIdsAsync, setDtccIdsAuditSink, _resetDtccForTests,
} from "./dtccSwaps";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const INDEX = "_seen_ids_index.jsonl.gz";
const DAY_RE = /^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/;

/** Live-shaped Dissemination Identifiers (19-digit canonical decimals). */
const idOf = (day: number, i: number) => `61${String(day).padStart(3, "0")}${String(i).padStart(12, "0")}01`;
const dayLine = (id: string) => JSON.stringify({ disseminationId: id, actionType: "NEWT", underlierIdLeg1: "US0378331005" });

function tmp(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}
const quiet = () => {
  const lines: string[] = [];
  const e = console.error, l = console.log;
  console.error = (...a: unknown[]) => { lines.push(a.map(String).join(" ")); };
  console.log = (...a: unknown[]) => { lines.push(a.map(String).join(" ")); };
  return { lines, restore: () => { console.error = e; console.log = l; } };
};

interface Rebuild { ids: string[]; index: string; audit: Array<[string, string]>; log: string[] }

/** Runs one loader on `dir` with no index present; returns what it built, the
 *  index it wrote, the audit rows it sent and its console lines. The index is
 *  removed afterwards so the next loader takes the same fallback over the
 *  same directory entries (same readdir order). */
async function rebuild(dir: string, loader: "sync" | "async"): Promise<Rebuild> {
  assert.equal(fs.existsSync(path.join(dir, INDEX)), false, "no index: the legacy day-file scan runs");
  const audit: Array<[string, string]> = [];
  setDtccIdsAuditSink((action, detail) => { audit.push([action, detail]); });
  const q = quiet();
  let ids: string[];
  try {
    ids = [...(loader === "sync" ? loadSeenIds(dir) : await loadSeenIdsAsync(dir))];
  } finally {
    q.restore();
    setDtccIdsAuditSink(null);
  }
  const index = zlib.gunzipSync(fs.readFileSync(path.join(dir, INDEX))).toString("utf8");
  fs.rmSync(path.join(dir, INDEX));
  return { ids, index, audit, log: q.lines };
}

/** The audit rows as [action, day file named] — the error text itself comes
 *  from zlib/fs and is not part of the contract. */
const named = (audit: Array<[string, string]>, files: string[]) =>
  audit.map(([action, detail]) => [action, files.filter((f) => detail.includes(f))]);

// ── (1) the first-listed day file is unreadable ──────────────────────────────

test("an unreadable day file no longer ends the legacy rebuild: every other day's IDs are rebuilt and indexed, the file is skipped, counted and reported (DTCC-ARCHIVE-ERROR) — sync and async alike", async () => {
  const base = tmp("dtcc-rebuild-first-");
  const dir = path.join(base, "dtccswaps");
  fs.mkdirSync(dir, { recursive: true });
  const byFile = new Map<string, string[]>();
  for (let d = 1; d <= 4; d++) {
    const ids = Array.from({ length: 50 }, (_, i) => idOf(d, i));
    const file = `2026-10-0${d}.jsonl.gz`;
    fs.writeFileSync(path.join(dir, file), zlib.gzipSync(ids.map(dayLine).join("\n") + "\n"));
    byFile.set(file, ids);
  }
  fs.writeFileSync(path.join(dir, "notes.jsonl.gz"), zlib.gzipSync(dayLine("FOREIGN") + "\n"));
  // The day file the scan meets FIRST becomes unreadable (overwritten in place,
  // so its directory entry — and the readdir order — stay as they were). Before
  // the fix, nothing after it was read: the rebuilt set was EMPTY here.
  const order = fs.readdirSync(dir).filter((f) => DAY_RE.test(f));
  assert.equal(order.length, 4);
  const victim = order[0];
  fs.writeFileSync(path.join(dir, victim), Buffer.from("not gzip — a corrupt day file"));
  const survivors = order.slice(1).flatMap((f) => byFile.get(f)!);

  try {
    const sync = await rebuild(dir, "sync");
    const asyncR = await rebuild(dir, "async");
    for (const [name, r] of [["sync", sync], ["async", asyncR]] as const) {
      assert.deepEqual(r.ids.slice().sort(), survivors.slice().sort(),
        `${name}: every readable day's IDs, none of the unreadable one's, never the foreign file`);
      assert.deepEqual(r.index.split("\n").filter(Boolean).sort(), survivors.slice().sort(),
        `${name}: the persisted index holds them all, so the next boot's fast path dedups those days`);
      assert.deepEqual(named(r.audit, order), [["DTCC-ARCHIVE-ERROR", [victim]]],
        `${name}: one audit row, naming the skipped file`);
      assert.match(r.audit[0][1], /skipped/);
      assert.ok(r.audit[0][1].length <= 500, `${name}: fits the audit_log message cap`);
      assert.ok(r.log.some((l) => l.includes("1 of 4 day file(s) unreadable")), `${name}: the skip is counted: ${r.log.join(" | ")}`);
    }
    assert.deepEqual(asyncR.ids, sync.ids, "identical members AND iteration order");
    assert.equal(asyncR.index, sync.index, "identical rebuilt index text");

    // The harm the old scan did: with every later day's IDs dropped from the
    // index, the next poll re-archived those rows as duplicates. Now only the
    // unreadable day's rows (the upstream file still serves them) are archived
    // again — a readable copy of a day nothing could read.
    fs.writeFileSync(path.join(dir, INDEX), zlib.gzipSync(sync.index));
    const prev = process.env[IDS_ENV];
    process.env[IDS_ENV] = "memory";
    _resetDtccForTests();
    const q = quiet();
    try {
      const reserved = [...survivors.slice(0, 5), ...byFile.get(victim)!.slice(0, 3)];
      const r = archiveNewRows(rowsOf(reserved), "2026-11-06", base);
      assert.deepEqual(r, { newRows: 3, seenTotal: survivors.length + 3 },
        "the readable days' rows are recognised as archived; only the unreadable day's rows are written again");
    } finally {
      q.restore();
      _resetDtccForTests();
      if (prev === undefined) delete process.env[IDS_ENV]; else process.env[IDS_ENV] = prev;
    }
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

// ── (2) a file that fails PART-WAY adds nothing, in the streamed loader too ──

test("a day file that fails part-way (truncated gzip) adds none of its IDs and none of its malformed lines in EITHER loader; an unreadable plain file (a directory) is skipped too; the scan goes on past both", async () => {
  const dir = tmp("dtcc-rebuild-partial-");
  const good1 = Array.from({ length: 40 }, (_, i) => idOf(1, i));
  const truncatedIds = Array.from({ length: 20_000 }, (_, i) => idOf(2, i));
  const good4 = Array.from({ length: 30 }, (_, i) => idOf(4, i));
  fs.writeFileSync(path.join(dir, "2026-10-01.jsonl.gz"),
    zlib.gzipSync([...good1.slice(0, 20).map(dayLine), "{not json", ...good1.slice(20).map(dayLine)].join("\n") + "\n"));
  // ~2 MB of JSONL; the gzip cut at 60% still streams thousands of whole lines
  // before the stream reports "unexpected end of file" (a crash mid-append)
  const full = zlib.gzipSync([dayLine(truncatedIds[0]), "{also not json", ...truncatedIds.slice(1).map(dayLine)].join("\n") + "\n");
  fs.writeFileSync(path.join(dir, "2026-10-02.jsonl.gz"), full.subarray(0, Math.floor(full.length * 0.6)));
  fs.mkdirSync(path.join(dir, "2026-10-03.jsonl")); // a "plain day file" that cannot be read (EISDIR)
  fs.writeFileSync(path.join(dir, "2026-10-04.jsonl"), good4.map(dayLine).join("\n") + "\n");

  // The fixture really exercises the hazard: streamed, the truncated file
  // hands over whole lines BEFORE it fails (all-at-once, it fails outright).
  let streamedBytes = 0;
  await assert.rejects(async () => {
    for await (const chunk of fs.createReadStream(path.join(dir, "2026-10-02.jsonl.gz")).pipe(zlib.createGunzip())) {
      streamedBytes += (chunk as Buffer).length;
    }
  }, /unexpected end of file/);
  assert.ok(streamedBytes > 100_000, `the streamed read yields ${streamedBytes} bytes of lines before failing`);
  assert.throws(() => zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-10-02.jsonl.gz"))), /unexpected end of file/);

  const files = ["2026-10-01.jsonl.gz", "2026-10-02.jsonl.gz", "2026-10-03.jsonl", "2026-10-04.jsonl"];
  try {
    const sync = await rebuild(dir, "sync");
    const asyncR = await rebuild(dir, "async");
    for (const [name, r] of [["sync", sync], ["async", asyncR]] as const) {
      assert.deepEqual(r.ids.slice().sort(), [...good1, ...good4].sort(), `${name}: both readable files, past both unreadable ones`);
      assert.equal(r.ids.filter((id) => truncatedIds.includes(id)).length, 0,
        `${name}: not one ID of the file that failed part-way (the streamed loader staged them and dropped them)`);
      assert.deepEqual(named(r.audit, files).sort(), [["DTCC-ARCHIVE-ERROR", ["2026-10-02.jsonl.gz"]], ["DTCC-ARCHIVE-ERROR", ["2026-10-03.jsonl"]]],
        `${name}: one audit row per skipped file`);
      assert.ok(r.log.some((l) => l.includes("2 of 4 day file(s) unreadable")), `${name}: counted: ${r.log.join(" | ")}`);
      assert.deepEqual(r.log.filter((l) => l.includes("malformed archive line")), ["[datacore] dtccswaps loadSeenIds: 1 malformed archive line(s) skipped"],
        `${name}: only the readable file's malformed line is counted, never the skipped file's`);
    }
    assert.deepEqual(asyncR.ids, sync.ids, "identical members AND iteration order");
    assert.equal(asyncR.index, sync.index, "identical rebuilt index text");
    assert.ok(!fs.existsSync(path.join(dir, `${INDEX}.rebuild.tmp`)), "no temp file left behind");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── (3) nothing to skip: unchanged ───────────────────────────────────────────

test("a fully readable archive rebuilds exactly as before: no audit row, no unreadable count", async () => {
  const dir = tmp("dtcc-rebuild-clean-");
  const ids = Array.from({ length: 25 }, (_, i) => idOf(7, i));
  fs.writeFileSync(path.join(dir, "2026-10-07.jsonl.gz"), Buffer.concat([
    zlib.gzipSync(ids.slice(0, 10).map(dayLine).join("\n") + "\n"),
    zlib.gzipSync(ids.slice(10).map(dayLine).join("\n") + "\n"), // two appended members, as archiveNewRows writes
  ]));
  try {
    for (const loader of ["sync", "async"] as const) {
      const r = await rebuild(dir, loader);
      assert.deepEqual(r.ids.slice().sort(), ids.slice().sort(), loader);
      assert.deepEqual(r.audit, [], `${loader}: nothing reported`);
      assert.equal(r.log.some((l) => l.includes("unreadable")), false, `${loader}: no unreadable count`);
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── helpers: CSV rows for archiveNewRows (as dtccArchiveGap.test.ts) ─────────

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
