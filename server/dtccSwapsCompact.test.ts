// dtccSwapsCompact.test.ts — dtccSwaps' dedup set is a CompactIdSet
// (2026-11-06 dtcc-compact REPAIR). Pins, at the module boundary: both index
// loaders hold EXACTLY the strings the original `toString().split("\n")`
// reader produced (invalid UTF-8 included); both size the set once from the
// counted lines and build identical tables; the health gauge reports the set's
// true bytes; a warm poll that would overfill the table grows it before the
// claim and still archives exactly the fresh rows.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  archiveNewRows, archiveNewRowsAsync, loadSeenIds, loadSeenIdsAsync, dtccIdsStats, type DtccSwapRow,
  _resetDtccForTests,
} from "./dtccSwaps";
import { CompactIdSet, idSetCapacityFor } from "./dtccIdSet";
import { DtccIdsStore, DTCC_IDS_INSERT_BATCH } from "./dtccIdsStore";
import { registerBuiltinHeapGauges, DTCC_SEEN_ID_BYTES_EST } from "./heapGauges";
import { readHeapGauges, _resetHeapGaugesForTests } from "./heapDiag";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const INDEX = "_seen_ids_index.jsonl.gz";
const tmp = (prefix: string) => fs.mkdtempSync(path.join(os.tmpdir(), prefix));
const sorted = (it: Iterable<string>) => Array.from(it).sort();
const row = (id: string): DtccSwapRow => ({
  disseminationId: id, originalDisseminationId: id, actionType: "NEWT", eventTimestamp: "2026-11-05T12:33:36Z",
  effectiveDate: "2026-11-05", notionalAmountLeg1: 1, notionalCurrencyLeg1: "USD",
  underlierId: "US0378331005", underlierIdSource: "ISIN", underlierName: "APPLE INC",
});
const numericIds = (n: number, lead = "61") => Array.from({ length: n }, (_, i) => `${lead}${String(i * 3).padStart(15, "0")}01`);

async function inMemoryMode<T>(fn: () => Promise<T> | T): Promise<T> {
  const prev = process.env[IDS_ENV];
  process.env[IDS_ENV] = "memory";
  _resetDtccForTests();
  try {
    return await fn();
  } finally {
    _resetDtccForTests();
    if (prev === undefined) delete process.env[IDS_ENV]; else process.env[IDS_ENV] = prev;
  }
}

test("both loaders hold exactly the strings the original split('\\n') reader produced: canonical ids, leading-zero / 20-digit / alphanumeric ids, '\\r', multi-byte and INVALID UTF-8 lines, across gzip members", async () => {
  const dir = tmp("dtcc-compact-exact-");
  const parts: Buffer[] = [
    Buffer.from(numericIds(5_000).join("\n") + "\n"),
    Buffer.from("0123\n123\n00\n0\n12345678901234567890\n6192117454000002001\nA1\nwith\rcr\n\n ünïcödé-字 \n"),
    Buffer.concat([Buffer.from([0x31, 0x32, 0xc3, 0x0a, 0x80, 0x37, 0x0a, 0xff, 0xfe, 0x0a, 0xe5, 0xad]), Buffer.from("\n6192117454000002001\nlast-no-newline")]),
  ];
  const file = path.join(dir, INDEX);
  fs.writeFileSync(file, Buffer.concat(parts.map((p) => zlib.gzipSync(p))));
  const original = new Set(zlib.gunzipSync(fs.readFileSync(file)).toString("utf8").split("\n").filter(Boolean));
  const sync = loadSeenIds(dir);
  const async = await loadSeenIdsAsync(dir);
  for (const [name, set] of [["sync", sync], ["async", async]] as const) {
    assert.equal(set.size, original.size, `${name}: same member count`);
    assert.deepEqual(sorted(set), sorted(original), `${name}: same members`);
    for (const id of original) assert.equal(set.has(id), true, `${name}: has ${JSON.stringify(id)}`);
    assert.equal(set.has("6192117454000002000"), false);
  }
  assert.equal(sync.numericSize, 5_000 + 3, "the 5,000 seeded ids + 123, 0, 6192117454000002001 are numeric; the rest fall back");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("the index loaders size the set ONCE from the counted lines (no resize while loading) and build identical tables", async () => {
  const dir = tmp("dtcc-compact-sized-");
  const ids = numericIds(120_000);
  fs.writeFileSync(path.join(dir, INDEX), Buffer.concat([zlib.gzipSync(ids.slice(0, 70_000).join("\n") + "\n"), zlib.gzipSync(ids.slice(70_000).join("\n") + "\nX9\n")]));
  const sync = loadSeenIds(dir);
  const async = await loadSeenIdsAsync(dir);
  assert.equal(sync.tableCapacity, idSetCapacityFor(120_001), "sized from the 120,001 counted lines");
  assert.equal(async.tableCapacity, sync.tableCapacity);
  assert.deepEqual([...async], [...sync], "same table, same iteration order");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("dtccIdsStats and the dtcc.seenIds gauge report the set's true bytes: the typed-array table exactly, fallback strings at the per-string estimate", async () => {
  const base = tmp("dtcc-compact-gauge-");
  fs.mkdirSync(path.join(base, "dtccswaps"), { recursive: true });
  const ids = [...numericIds(50_000), "A1", "0123", "x-y"];
  fs.writeFileSync(path.join(base, "dtccswaps", INDEX), zlib.gzipSync(ids.join("\n") + "\n"));
  _resetHeapGaugesForTests();
  registerBuiltinHeapGauges();
  try {
    await inMemoryMode(() => {
      assert.deepEqual(archiveNewRows([row(ids[0]), row("6200000000000000001")], "2026-11-06", base), { newRows: 1, seenTotal: ids.length + 1 });
      const st = dtccIdsStats();
      assert.equal(st.memoryCount, ids.length + 1);
      assert.equal(st.memoryTableBytes, idSetCapacityFor(ids.length) * 8, "exact: capacity x 8 bytes");
      assert.equal(st.memoryFallbackCount, 3);
      const g = readHeapGauges()["dtcc.seenIds"] as { entries: number; bytes: number; tableBytes: number; fallbackCount: number; estimate: string };
      assert.equal(g.entries, ids.length + 1);
      assert.equal(g.bytes, st.memoryTableBytes! + 3 * DTCC_SEEN_ID_BYTES_EST);
      assert.equal(g.tableBytes, st.memoryTableBytes);
      assert.equal(g.fallbackCount, 3);
      assert.ok(g.bytes / g.entries < 13, `~11.4 B/id, not the old 72 (${(g.bytes / g.entries).toFixed(1)})`);
    });
  } finally {
    _resetHeapGaugesForTests();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("a warm poll that would overfill the table grows it BEFORE the claim and archives exactly the fresh rows (sync and async paths agree)", async () => {
  const seeded = numericIds(200_000);
  const fresh = numericIds(60_000, "62"); // 30% growth: past the table's 0.85 fill limit
  const rows = [...seeded.slice(0, 1_000), ...fresh, ...fresh.slice(0, 10)].map(row);
  const out: Array<{ r: { newRows: number; seenTotal: number }; day: Buffer; index: Buffer; capacityGrew: boolean }> = [];
  for (const asyncPath of [false, true]) {
    const base = tmp("dtcc-compact-warm-");
    fs.mkdirSync(path.join(base, "dtccswaps"), { recursive: true });
    fs.writeFileSync(path.join(base, "dtccswaps", INDEX), zlib.gzipSync(seeded.join("\n") + "\n"));
    await inMemoryMode(async () => {
      archiveNewRows([row(seeded[0])], "2026-11-05", base); // load the set (sized for the 200k seeded ids)
      const before = dtccIdsStats().memoryTableBytes!;
      const r = asyncPath ? await archiveNewRowsAsync(rows, "2026-11-06", base) : archiveNewRows(rows, "2026-11-06", base);
      out.push({
        r, capacityGrew: dtccIdsStats().memoryTableBytes! > before,
        day: fs.readFileSync(path.join(base, "dtccswaps", "2026-11-06.jsonl.gz")),
        index: fs.readFileSync(path.join(base, "dtccswaps", INDEX)),
      });
    });
    fs.rmSync(base, { recursive: true, force: true });
  }
  for (const o of out) {
    assert.deepEqual(o.r, { newRows: 60_010, seenTotal: 260_000 }, "the in-poll duplicates of a fresh id are both archived, as with a Set");
    assert.equal(o.capacityGrew, true);
  }
  assert.ok(out[1].day.equals(out[0].day), "day file byte-identical across the two paths");
  assert.ok(out[1].index.equals(out[0].index), "index byte-identical across the two paths");
});

test("the dedup DB's rebuild fallback imports every id of a CompactIdSet, batch by batch off its iterator (never all decoded at once)", async () => {
  const dir = tmp("dtcc-compact-rebuild-");
  const ids = [...numericIds(DTCC_IDS_INSERT_BATCH * 2 + 345), "A1", "0123"];
  const rebuilt = new CompactIdSet();
  for (const id of ids) rebuilt.add(id);
  const store = new DtccIdsStore({ dir, indexPath: path.join(dir, INDEX), rebuildIds: () => rebuilt, log: () => {}, logError: () => {} });
  try {
    const r = await store.ensureReadyAsync(); // no index file -> the rebuild fallback
    assert.equal(r.state, "done");
    assert.equal(r.source, "rebuilt");
    assert.equal(r.imported, ids.length);
    assert.equal(r.dbAfter, ids.length);
    for (const id of ["A1", "0123", ids[0], ids[ids.length - 3]]) assert.equal(store.has(id), true, id);
    assert.equal(store.has("123"), false, "\"0123\" was imported as itself");
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
