// dtccArchiveGap.test.ts — [REPAIR] 2026-11-06, archive integrity: a failed
// day-file write never becomes a permanent archive gap. Before: the fresh IDs
// were appended to the seen-ID index (and kept in the in-memory claim) even
// when their rows never reached the day file, so every later poll skipped
// them. Now nothing records them, the claim is dropped (the next poll reloads
// the set from the index and re-archives the rows), and the failure is a
// DTCC-ARCHIVE-ERROR row through the audit sink, which routes.ts now wires.
// Run: npx tsx --test server/dtccArchiveGap.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  REQUIRED_COLUMNS, parseHeader, parseDtccLine, type DtccSwapRow,
  archiveNewRows, archiveNewRowsAsync, setDtccIdsAuditSink,
  _resetDtccForTests, _setDtccFreeBytesReaderForTests,
} from "./dtccSwaps";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const INDEX = "_seen_ids_index.jsonl.gz";
const DAY = "2026-11-06";
const HEADER_COLS = [...REQUIRED_COLUMNS];
const HEADER_LINE = HEADER_COLS.map((c) => `"${c}"`).join(",");

function csvRow(diss: string): string {
  const byName: Record<string, string> = {
    "Dissemination Identifier": diss, "Original Dissemination Identifier": diss, "Action type": "NEWT",
    "Event timestamp": "2026-11-05T12:33:36Z", "Effective Date": "2026-11-05", "Asset Class": "EQ",
    "Notional amount-Leg 1": "28,000,000", "Notional currency-Leg 1": "USD", "Underlier ID-Leg 1": "US0378331005",
    "Underlier ID source-Leg 1": "ISIN", "UPI Underlier Name": "APPLE INC",
  };
  return HEADER_COLS.map((c) => `"${byName[c] ?? ""}"`).join(",");
}
function rowsOf(ids: readonly string[]): DtccSwapRow[] {
  const idx = parseHeader(HEADER_LINE);
  return ids.map((d) => parseDtccLine(csvRow(d), idx, HEADER_COLS.length)!);
}
function gunzipLines(file: string): string[] {
  return zlib.gunzipSync(fs.readFileSync(file)).toString("utf8").split("\n").filter(Boolean);
}
function seed(prefix: string, ids: readonly string[]): { base: string; dir: string } {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  const dir = path.join(base, "dtccswaps");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, INDEX), zlib.gzipSync(ids.join("\n") + (ids.length ? "\n" : "")));
  return { base, dir };
}
async function withSink<T>(fn: (rows: Array<{ type: string; message: string }>) => Promise<T>): Promise<T> {
  const prev = process.env[IDS_ENV];
  process.env[IDS_ENV] = "memory";
  _resetDtccForTests();
  _setDtccFreeBytesReaderForTests(() => 64 * 2 ** 30);
  const rows: Array<{ type: string; message: string }> = [];
  setDtccIdsAuditSink((type, message) => { rows.push({ type, message }); });
  const err = console.error;
  console.error = () => {};
  try {
    return await fn(rows);
  } finally {
    console.error = err;
    setDtccIdsAuditSink(null);
    _setDtccFreeBytesReaderForTests(null);
    _resetDtccForTests();
    if (prev === undefined) delete process.env[IDS_ENV]; else process.env[IDS_ENV] = prev;
  }
}

for (const asyncPath of [true, false]) {
  const label = asyncPath ? "archiveNewRowsAsync" : "archiveNewRows";
  const run = (ids: string[], base: string) =>
    asyncPath ? archiveNewRowsAsync(rowsOf(ids), DAY, base) : Promise.resolve(archiveNewRows(rowsOf(ids), DAY, base));

  test(`${label}: a failed day-file write records nothing as seen, raises DTCC-ARCHIVE-ERROR, and the next poll archives the rows`, async () => {
    const { base, dir } = seed(`dtcc-gap-${asyncPath ? "a" : "s"}-`, ["1", "2"]);
    try {
      await withSink(async (audit) => {
        const dayPath = path.join(dir, `${DAY}.jsonl.gz`);
        fs.mkdirSync(dayPath);                       // the day file cannot be written (EISDIR)
        const r1 = await run(["1", "2", "3", "4"], base);
        assert.deepEqual(r1, { newRows: 0, seenTotal: 2 }, "nothing archived; the total is the set before the claim");
        assert.deepEqual(gunzipLines(path.join(dir, INDEX)), ["1", "2"], "the index never got the unwritten IDs");
        const errs = audit.filter((a) => a.type === "DTCC-ARCHIVE-ERROR");
        assert.equal(errs.length, 1);
        assert.match(errs[0].message, /2 new row\(s\) not written to 2026-11-06\.jsonl\.gz: .* not recorded as seen; the next poll retries them/);

        fs.rmdirSync(dayPath);                       // the volume recovers
        const r2 = await run(["1", "2", "3", "4"], base);
        assert.deepEqual(r2, { newRows: 2, seenTotal: 4 }, "the retried rows are fresh again and archived");
        assert.deepEqual(gunzipLines(dayPath).map((l) => JSON.parse(l).disseminationId), ["3", "4"]);
        assert.deepEqual(gunzipLines(path.join(dir, INDEX)), ["1", "2", "3", "4"]);
        const r3 = await run(["3", "4"], base);
        assert.deepEqual(r3, { newRows: 0, seenTotal: 4 }, "and then they are seen");
      });
    } finally {
      fs.rmSync(base, { recursive: true, force: true });
    }
  });

  test(`${label}: an index append failure keeps the archived rows and is loud (a duplicate after restart, never a gap)`, async () => {
    const { base, dir } = seed(`dtcc-gapidx-${asyncPath ? "a" : "s"}-`, ["1"]);
    try {
      await withSink(async (audit) => {
        assert.deepEqual(await run(["1", "2"], base), { newRows: 1, seenTotal: 2 }, "loads the set, archives 2");
        fs.rmSync(path.join(dir, INDEX));
        fs.mkdirSync(path.join(dir, INDEX));         // the index cannot be appended (EISDIR); the set is already loaded
        assert.deepEqual(await run(["2", "3"], base), { newRows: 1, seenTotal: 3 }, "the row still reaches the day file");
        assert.deepEqual(gunzipLines(path.join(dir, `${DAY}.jsonl.gz`)).map((l) => JSON.parse(l).disseminationId), ["2", "3"]);
        const errs = audit.filter((a) => a.type === "DTCC-ARCHIVE-ERROR");
        assert.equal(errs.length, 1);
        assert.match(errs[0].message, /seen-ID index append of 1 id\(s\) failed: .*\(a duplicate, never a gap\)$/);
        assert.deepEqual(await run(["3"], base), { newRows: 0, seenTotal: 3 }, "this process still knows the row");
      });
    } finally {
      fs.rmSync(base, { recursive: true, force: true });
    }
  });
}

test("routes.ts wires the DTCC audit sink to the audit log before the poll boots", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  const sink = src.indexOf("setDtccIdsAuditSink((type, message) => {");
  const boot = src.indexOf("bootDtccSwapsPoll();");
  assert.ok(sink > 0, "the sink is wired");
  assert.ok(boot > sink, "before the first poll");
  assert.ok(src.slice(sink, boot).includes('db.prepare("INSERT INTO audit_log (time, type, message) VALUES (?, ?, ?)")'));
});
