// dtccSwapsStall.test.ts — the DTCC refresh path no longer blocks the event
// loop (2026-11-06 dtcc-stall REPAIR, KEEP ALIVE). Live: an EVENTLOOP-LAG of
// 7.7-10.2 s after 26 of 39 boots, the eager poll's synchronous loadSeenIds
// (gunzipSync + split + Set of 6.94M IDs) plus a full read-modify-write of the
// compact index (gunzip + re-gzip of the whole file) on every poll with fresh
// rows. Pins: (1) the index and the day file grow by one appended gzip member
// per poll — the earlier bytes are never rewritten — and EVERY reader of those
// files reads all members; (2) the streamed loader builds exactly the Set the
// synchronous one builds, on every path including each fallback; (3) the async
// archive step makes the same decisions and writes the same bytes as the sync
// one in all three store modes; (4) the loop keeps turning while a large index
// loads, and nobody ever sees, or dedups against, a partial Set.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import readline from "node:readline";
import { promisify } from "node:util";
import { Readable } from "node:stream";
import { performance } from "node:perf_hooks";
import Database from "better-sqlite3";
import {
  REQUIRED_COLUMNS, parseHeader, parseDtccLine, type DtccSwapRow,
  archiveNewRows, archiveNewRowsAsync, loadSeenIds, loadSeenIdsAsync, readArchivedDtccRows,
  refreshDtccSwaps, latestDtcc, dtccIdsStats, setDtccIdsAuditSink,
  _resetDtccForTests, _setDtccFreeBytesReaderForTests, DTCC_ASYNC_SLICE_MS,
} from "./dtccSwaps";
import { DtccIdsStore, countGzipLines, gzipLineBatches } from "./dtccIdsStore";
import { countFileLines } from "./platformStats";
import { readArchivedCusips } from "../scripts/dtcc_resolve_cusips";
import { registerBuiltinHeapGauges } from "./heapGauges";
import { readHeapGauges, _resetHeapGaugesForTests } from "./heapDiag";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const INDEX = "_seen_ids_index.jsonl.gz";
const HEADER_COLS = [...REQUIRED_COLUMNS, "Some Future Column"];
const HEADER_LINE = HEADER_COLS.map((c) => `"${c}"`).join(",");

function csvRow(diss: string, id = "US0378331005", src = "ISIN", name = "APPLE INC", notional = "28,000,000") {
  const byName: Record<string, string> = {
    "Dissemination Identifier": diss, "Original Dissemination Identifier": diss, "Action type": "NEWT",
    "Event timestamp": "2026-11-05T12:33:36Z", "Effective Date": "2026-11-05", "Asset Class": "EQ",
    "Notional amount-Leg 1": notional, "Notional currency-Leg 1": "USD", "Underlier ID-Leg 1": id,
    "Underlier ID source-Leg 1": src, "UPI Underlier Name": name, "Some Future Column": "x",
  };
  return HEADER_COLS.map((c) => `"${byName[c]}"`).join(",");
}
function rowsOf(ids: readonly string[], underlier = "US0378331005"): DtccSwapRow[] {
  const idx = parseHeader(HEADER_LINE);
  return ids.map((d) => parseDtccLine(csvRow(d, underlier), idx, HEADER_COLS.length)!);
}
function buildZip(csvText: string): Buffer {
  const name = Buffer.from("SEC_CUMULATIVE_EQUITIES_2026_11_05.csv");
  const data = Buffer.from(csvText, "utf8");
  const compressed = zlib.deflateRawSync(data);
  const h = Buffer.alloc(30);
  h.writeUInt32LE(0x04034b50, 0); h.writeUInt16LE(20, 4); h.writeUInt16LE(0, 6); h.writeUInt16LE(8, 8);
  h.writeUInt32LE(compressed.length, 18); h.writeUInt32LE(data.length, 22); h.writeUInt16LE(name.length, 26); h.writeUInt16LE(0, 28);
  return Buffer.concat([h, name, compressed, Buffer.from("CENTRAL-DIRECTORY")]);
}
function okFetch(zip: Buffer) {
  return async () => ({ ok: true, status: 200, body: Readable.toWeb(Readable.from([zip])) as unknown as ReadableStream<Uint8Array> });
}
function tmp(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}
function seedIndex(baseDir: string, ids: readonly string[]): string {
  const dir = path.join(baseDir, "dtccswaps");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, INDEX), zlib.gzipSync(ids.join("\n") + (ids.length ? "\n" : "")));
  return dir;
}
const jsonl = (rows: DtccSwapRow[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
const quiet = () => {
  const lines: string[] = [];
  const e = console.error, l = console.log;
  console.error = (...a: unknown[]) => { lines.push(a.map(String).join(" ")); };
  console.log = (...a: unknown[]) => { lines.push(a.map(String).join(" ")); };
  return { lines, restore: () => { console.error = e; console.log = l; } };
};
async function withMode<T>(mode: "memory" | "dual" | "sqlite", fn: () => Promise<T> | T): Promise<T> {
  const prev = process.env[IDS_ENV];
  process.env[IDS_ENV] = mode;
  _resetDtccForTests();
  _setDtccFreeBytesReaderForTests(() => 64 * 2 ** 30); // a roomy volume; dtccIdsDiskGuard.test.ts covers the guard
  try {
    return await fn();
  } finally {
    _setDtccFreeBytesReaderForTests(null);
    _resetDtccForTests();
    if (prev === undefined) delete process.env[IDS_ENV]; else process.env[IDS_ENV] = prev;
  }
}
/** Longest gap between 1 ms timer ticks while `fn` runs, plus how many ticks
 *  landed. One more timer turn is awaited AFTER fn: a long synchronous block
 *  that is the awaited work's last step resolves through microtasks only, and
 *  would otherwise never be observed. */
async function loopGaps<T>(fn: () => Promise<T> | T): Promise<{ out: T; maxGapMs: number; ticks: number; wallMs: number }> {
  let last = performance.now();
  let maxGapMs = 0;
  let ticks = 0;
  const timer = setInterval(() => {
    const now = performance.now();
    maxGapMs = Math.max(maxGapMs, now - last);
    last = now;
    ticks++;
  }, 1);
  const t0 = performance.now();
  try {
    const out = await fn();
    const wallMs = performance.now() - t0;
    await new Promise((r) => setTimeout(r, 25));
    return { out, maxGapMs, ticks, wallMs };
  } finally {
    clearInterval(timer);
  }
}

// ── (1) appended gzip members, and every reader ───────────────────────────────

test("index + day file grow by ONE appended gzip member per poll — the earlier bytes are never re-read or rewritten", async () => {
  const base = tmp("dtcc-stall-members-");
  const dir = seedIndex(base, ["1", "2"]);
  const seed = fs.readFileSync(path.join(dir, INDEX));
  await withMode("memory", () => {
    const first = rowsOf(["1", "2", "3", "4"]);
    assert.deepEqual(archiveNewRows(first, "2026-11-06", base), { newRows: 2, seenTotal: 4 });
    const fresh1 = first.slice(2);
    assert.ok(fs.readFileSync(path.join(dir, INDEX)).equals(Buffer.concat([seed, zlib.gzipSync("3\n4\n")])),
      "the index is the seed bytes untouched + one new member holding only the new ids");
    assert.ok(fs.readFileSync(path.join(dir, "2026-11-06.jsonl.gz")).equals(zlib.gzipSync(jsonl(fresh1))));
    const second = rowsOf(["3", "5"]);
    assert.deepEqual(archiveNewRows(second, "2026-11-06", base), { newRows: 1, seenTotal: 5 });
    assert.ok(fs.readFileSync(path.join(dir, INDEX)).equals(Buffer.concat([seed, zlib.gzipSync("3\n4\n"), zlib.gzipSync("5\n")])));
    assert.ok(fs.readFileSync(path.join(dir, "2026-11-06.jsonl.gz")).equals(Buffer.concat([zlib.gzipSync(jsonl(fresh1)), zlib.gzipSync(jsonl(second.slice(1)))])),
      "the same day's second batch is appended as its own member, not merged into a rewrite");
  });
  fs.rmSync(base, { recursive: true, force: true });
});

test("EVERY reader of the index and the day files reads all appended members (zlib sync/async/stream, readline, dtccIdsStore, loadSeenIds both ways, readArchivedDtccRows, platformStats, dtcc_resolve_cusips)", async () => {
  const base = tmp("dtcc-stall-readers-");
  const dir = seedIndex(base, ["s1", "s2"]);
  const batchA = rowsOf(["a1", "a2"], "US0378331005"); // AAPL
  const batchB = rowsOf(["b1"], "US88160R1014");        // TSLA — a second underlier only the second member holds
  await withMode("memory", () => {
    archiveNewRows(batchA, "2026-11-06", base);
    archiveNewRows(batchB, "2026-11-06", base);
  });
  const indexFile = path.join(dir, INDEX);
  const dayFile = path.join(dir, "2026-11-06.jsonl.gz");
  const allIds = ["s1", "s2", "a1", "a2", "b1"];
  const indexText = allIds.join("\n") + "\n";
  const dayText = jsonl([...batchA, ...batchB]);
  // the files really are multi-member: the first member alone is a strict prefix
  assert.ok(fs.readFileSync(indexFile).length > zlib.gzipSync("s1\ns2\n").length + 20);

  // zlib primitives (streamsInventory's peek uses the promisified async gunzip)
  for (const [file, text] of [[indexFile, indexText], [dayFile, dayText]] as const) {
    assert.equal(zlib.gunzipSync(fs.readFileSync(file)).toString("utf8"), text, `gunzipSync ${path.basename(file)}`);
    assert.equal((await promisify(zlib.gunzip)(fs.readFileSync(file))).toString("utf8"), text, `gunzip ${path.basename(file)}`);
    const chunks: Buffer[] = [];
    for await (const c of fs.createReadStream(file).pipe(zlib.createGunzip())) chunks.push(c as Buffer);
    assert.equal(Buffer.concat(chunks).toString("utf8"), text, `createGunzip ${path.basename(file)}`);
    const lines: string[] = [];
    for await (const l of readline.createInterface({ input: fs.createReadStream(file).pipe(zlib.createGunzip()), crlfDelay: Infinity })) lines.push(l);
    assert.deepEqual(lines, text.split("\n").filter(Boolean), `readline over createGunzip ${path.basename(file)}`);
  }
  // platformStats counts both (it lists every *.jsonl.gz, the index included)
  assert.equal(await countFileLines(indexFile), 5);
  assert.equal(await countFileLines(dayFile), 3);
  // dtccIdsStore: the streamed counter/reader and BOTH backfill twins import every member's ids
  assert.equal(await countGzipLines(indexFile), 5);
  const streamed: string[] = [];
  for await (const batch of gzipLineBatches(indexFile)) streamed.push(...batch);
  assert.deepEqual(streamed, allIds);
  for (const sync of [true, false]) {
    const dbDir = tmp("dtcc-stall-store-");
    fs.copyFileSync(indexFile, path.join(dbDir, INDEX));
    const store = new DtccIdsStore({ dir: dbDir, indexPath: path.join(dbDir, INDEX), log: () => {}, logError: () => {} });
    try {
      const r = sync ? store.ensureReadySync() : await store.ensureReadyAsync();
      assert.equal(r.state, "done", `backfill sync=${sync}`);
      assert.equal(r.indexCount, 5);
      assert.equal(r.imported, 5);
      assert.ok(allIds.every((id) => store.has(id)));
    } finally {
      store.close();
      fs.rmSync(dbDir, { recursive: true, force: true });
    }
  }
  // dtccSwaps' own readers: the index both ways, the day file, and the legacy day-file scan both ways
  assert.deepEqual([...loadSeenIds(dir)], allIds);
  assert.deepEqual([...(await loadSeenIdsAsync(dir))], allIds);
  const archived = readArchivedDtccRows(base, Date.parse("2026-11-06T12:00:00Z"));
  assert.deepEqual(archived?.rows.map((r) => r.disseminationId), ["a1", "a2", "b1"]);
  fs.rmSync(indexFile);
  const q = quiet();
  try {
    assert.deepEqual([...loadSeenIds(dir)].sort(), ["a1", "a2", "b1"], "legacy scan, sync: rows of both day-file members");
    fs.rmSync(indexFile);
    assert.deepEqual([...(await loadSeenIdsAsync(dir))].sort(), ["a1", "a2", "b1"], "legacy scan, async: rows of both day-file members");
  } finally { q.restore(); }
  // scripts/dtcc_resolve_cusips.ts reads every *.jsonl.gz here (day files AND
  // the index, whose bare-id lines it counts as malformed — unchanged behavior)
  const q2 = quiet();
  let cusips: string[];
  try { cusips = readArchivedCusips(dir); } finally { q2.restore(); }
  assert.deepEqual(cusips.sort(), ["037833100", "88160R101"], "the TSLA CUSIP lives only in the second member");
  fs.rmSync(base, { recursive: true, force: true });
});

// ── (2) the streamed loader builds exactly the synchronous loader's Set ──────

test("loadSeenIdsAsync == loadSeenIds: same members in the same order over a multi-member index — unicode, '\\r', empty and unterminated lines, lines straddling stream chunks", async () => {
  const dir = tmp("dtcc-stall-exact-");
  const many = Array.from({ length: 60_000 }, (_, i) => `61${String(i).padStart(15, "0")}01`);
  // incl. a 1 MB line that spans dozens of stream chunks (joined once, not re-copied per chunk)
  const awkward = ["A1", "ünïcödé-字-🙂", "with\rcarriage", "L".repeat(1_000_000) + "-end", " spaced ", "0123", "123", "x"];
  const text1 = [...many.slice(0, 30_000), ...awkward.slice(0, 3)].join("\n") + "\n\n";
  const text2 = [...awkward.slice(3), ...many.slice(30_000)].join("\n") + "\n";
  const text3 = "tail-without-newline";
  fs.writeFileSync(path.join(dir, INDEX), Buffer.concat([zlib.gzipSync(text1), zlib.gzipSync(text2), zlib.gzipSync(text3)]));
  const sync = loadSeenIds(dir);
  const asyncSet = await loadSeenIdsAsync(dir);
  assert.equal(asyncSet.size, sync.size);
  assert.deepEqual([...asyncSet], [...sync], "identical members AND iteration order (both loaders size and fill the set the same way)");
  assert.ok(asyncSet.has("0123") && asyncSet.has("123") && asyncSet.has("with\rcarriage") && asyncSet.has("tail-without-newline"));
  assert.equal(asyncSet.has(""), false, "empty lines are skipped, as split+if(line) does");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("loadSeenIdsAsync == loadSeenIds on every fallback (corrupt / empty / truncated-last-member index): the same day-file scan, the same malformed count, the same rebuilt index text", async () => {
  const day1 = [JSON.stringify({ disseminationId: "D1" }), "{not json", JSON.stringify({ disseminationId: "D2" })].join("\n") + "\n";
  const day2 = JSON.stringify({ disseminationId: "D3" }) + "\n" + JSON.stringify({ disseminationId: "D1" }) + "\n";
  const badIndexes: Record<string, Buffer> = {
    corrupt: Buffer.from("not gzip"),
    empty: Buffer.alloc(0),
    truncatedLastMember: Buffer.concat([zlib.gzipSync("D1\n"), zlib.gzipSync("D2\nD3\n").subarray(0, 12)]),
  };
  for (const [name, bad] of Object.entries(badIndexes)) {
    const results: Array<{ ids: string[]; index: string; malformed: string[] }> = [];
    for (const loader of ["sync", "async"] as const) {
      const dir = tmp(`dtcc-stall-fallback-${name}-`);
      fs.writeFileSync(path.join(dir, "2026-11-01.jsonl.gz"), zlib.gzipSync(day1));
      fs.writeFileSync(path.join(dir, "2026-11-02.jsonl"), day2); // a plain (non-gz) day file
      fs.writeFileSync(path.join(dir, "notes.jsonl.gz"), zlib.gzipSync(JSON.stringify({ disseminationId: "FOREIGN" }) + "\n"));
      fs.writeFileSync(path.join(dir, INDEX), bad);
      const q = quiet();
      let ids: string[];
      try {
        ids = [...(loader === "sync" ? loadSeenIds(dir) : await loadSeenIdsAsync(dir))];
      } finally { q.restore(); }
      results.push({
        ids,
        index: zlib.gunzipSync(fs.readFileSync(path.join(dir, INDEX))).toString("utf8"),
        malformed: q.lines.filter((l) => l.includes("malformed archive line")),
      });
      assert.ok(!fs.existsSync(path.join(dir, `${INDEX}.rebuild.tmp`)), `${name}/${loader}: no temp file left behind`);
      fs.rmSync(dir, { recursive: true, force: true });
    }
    assert.deepEqual(results[1], results[0], `${name}: async fallback == sync fallback`);
    assert.deepEqual(results[0].ids.slice().sort(), ["D1", "D2", "D3"], `${name}: recovered from the day files, never the foreign file`);
    assert.equal(results[0].malformed.length, 1);
  }
});

// ── (3) the async archive step == the sync one ───────────────────────────────

test("archiveNewRowsAsync == archiveNewRows in memory / dual / sqlite: same verdicts and totals, byte-identical day files and indexes, no mismatch", async () => {
  for (const mode of ["memory", "dual", "sqlite"] as const) {
    const outputs: Array<{ r: unknown[]; day: Buffer; index: Buffer; mismatches: number }> = [];
    for (const asyncPath of [false, true]) {
      const base = tmp(`dtcc-stall-twin-${mode}-`);
      const dir = seedIndex(base, ["1", "2"]);
      const r: unknown[] = [];
      await withMode(mode, async () => {
        const q = quiet();
        try {
          for (const ids of [["1", "2", "3", "4"], ["4", "5"], ["1", "2"]]) {
            r.push(asyncPath ? await archiveNewRowsAsync(rowsOf(ids), "2026-11-06", base) : archiveNewRows(rowsOf(ids), "2026-11-06", base));
          }
        } finally { q.restore(); }
        outputs.push({
          r, mismatches: dtccIdsStats().mismatches,
          day: fs.readFileSync(path.join(dir, "2026-11-06.jsonl.gz")), index: fs.readFileSync(path.join(dir, INDEX)),
        });
      });
      fs.rmSync(base, { recursive: true, force: true });
    }
    assert.deepEqual(outputs[0].r, [{ newRows: 2, seenTotal: 4 }, { newRows: 1, seenTotal: 5 }, { newRows: 0, seenTotal: 5 }], mode);
    assert.deepEqual(outputs[1].r, outputs[0].r, `${mode}: async verdicts == sync verdicts`);
    assert.ok(outputs[1].day.equals(outputs[0].day), `${mode}: day file byte-identical`);
    assert.ok(outputs[1].index.equals(outputs[0].index), `${mode}: compact index byte-identical`);
    assert.equal(outputs[1].mismatches, 0);
  }
});

test("async path, dual mode: a dedup disagreement is still logged as DTCC-IDS-MISMATCH with the same detail, and the MEMORY answer is served", async () => {
  const base = tmp("dtcc-stall-dual-mismatch-");
  const dir = seedIndex(base, ["1", "2"]);
  const raw = new Database(path.join(dir, "_seen_ids.db"));
  raw.exec("CREATE TABLE IF NOT EXISTS seen_ids (id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID");
  for (const id of ["1", "2", "3"]) raw.prepare("INSERT INTO seen_ids (id) VALUES (?)").run(id);
  raw.close();
  const audited: string[] = [];
  setDtccIdsAuditSink((action) => audited.push(action));
  try {
    await withMode("dual", async () => {
      const q = quiet();
      let r: { newRows: number; seenTotal: number };
      try { r = await archiveNewRowsAsync(rowsOf(["1", "2", "3", "4"]), "2026-11-06", base); } finally { q.restore(); }
      assert.deepEqual(r, { newRows: 2, seenTotal: 4 }, "memory verdict: 3 and 4 (the db alone would say only 4)");
      const st = dtccIdsStats();
      assert.equal(st.mismatches, 1);
      assert.match(st.lastMismatch!.detail, /memory says 2 fresh, db says 1; fresh-only-in-memory 1 \(e\.g\. 3\)/);
      assert.deepEqual(audited, ["DTCC-IDS-MISMATCH"]);
      assert.equal(st.backfill?.source, "streamed", "the async path backfills through the streamed twin, never the whole-index gunzipSync");
    });
  } finally {
    setDtccIdsAuditSink(null);
    fs.rmSync(base, { recursive: true, force: true });
  }
});

// ── (4) the loop keeps turning; nobody sees or dedups against a partial Set ──

test("refreshDtccSwaps never blocks the loop for the index load: the longest gap is a small fraction of the synchronous load's single block, and the health gauge reports loading (never a partial count) until the full Set is installed", async () => {
  // just under 2^20, so V8's last Set rehash during the load is the 524k one
  // while the synchronous block (measured AFTER, so its garbage cannot be
  // collected inside the measured window) keeps growing with N
  const N = 1_000_000;
  const base = tmp("dtcc-stall-slice-");
  const ids = Array.from({ length: N }, (_, i) => `61${String(i * 7).padStart(15, "0")}01`);
  const dir = seedIndex(base, ids);

  const csv = [HEADER_LINE, csvRow(ids[5]), csvRow(ids[N - 1]), csvRow("6299999999999999901")].join("\n") + "\n";
  _resetHeapGaugesForTests();
  registerBuiltinHeapGauges();
  const gaugeSamples: Array<{ entries: number; memoryCount: number | null; loading: boolean }> = [];
  let sampling = true;
  const sample = () => {
    if (!sampling) return;
    const g = readHeapGauges()["dtcc.seenIds"] as { entries: number; memoryCount: number | null; loading: boolean };
    gaugeSamples.push({ entries: g.entries, memoryCount: g.memoryCount, loading: g.loading });
    setTimeout(sample, 2);
  };
  let m: { maxGapMs: number; ticks: number; wallMs: number };
  try {
    m = await withMode("memory", async () => {
      sample();
      const q = quiet();
      let measured: { maxGapMs: number; ticks: number; wallMs: number };
      try {
        measured = await loopGaps(() => refreshDtccSwaps(okFetch(buildZip(csv)), Date.parse("2026-11-06T15:00:00Z"), base));
      } finally { q.restore(); sampling = false; }
      const hit = latestDtcc();
      assert.ok(hit, "the poll completed");
      assert.equal(hit!.newRows, 1, "only the never-seen id is new");
      assert.equal(hit!.totalArchived, N + 1);
      return measured;
    });
  } finally {
    _resetHeapGaugesForTests();
  }
  // the synchronous loader's one block on the same index, for scale
  const syncT0 = performance.now();
  assert.equal(loadSeenIds(dir).size, N + 1);
  const syncBlockMs = performance.now() - syncT0;
  console.log(`[bench] ${N} ids: sync loadSeenIds block ${syncBlockMs.toFixed(0)} ms; async refresh longest loop gap ${m.maxGapMs.toFixed(1)} ms over ${m.wallMs.toFixed(0)} ms wall, ${m.ticks} ticks (slice target ${DTCC_ASYNC_SLICE_MS} ms)`);
  assert.ok(m.ticks >= 20, `the loop kept turning during the load (${m.ticks} ticks)`);
  assert.ok(m.maxGapMs < syncBlockMs / 2, `longest gap ${m.maxGapMs.toFixed(1)} ms vs the sync block ${syncBlockMs.toFixed(0)} ms`);
  assert.ok(gaugeSamples.some((s) => s.loading && s.memoryCount === null && s.entries === 0), "mid-load: loading, no count, no entries");
  for (const s of gaugeSamples) {
    assert.ok(s.memoryCount === null || s.memoryCount >= N, `never a partial count (saw ${s.memoryCount})`);
    if (s.loading) assert.equal(s.memoryCount, null);
  }
  fs.rmSync(base, { recursive: true, force: true });
});

/** Turns the loop until the Set load is in flight (bounded). */
async function untilLoading(): Promise<void> {
  for (let i = 0; i < 100_000 && !dtccIdsStats().memoryLoading; i++) await new Promise((r) => setImmediate(r));
  assert.equal(dtccIdsStats().memoryLoading, true, "the load is in flight");
}

test("archive calls that arrive while the Set loads all wait for the same complete load, and never double-archive a row", async () => {
  const N = 200_000;
  const base = tmp("dtcc-stall-concurrent-");
  const ids = Array.from({ length: N }, (_, i) => `62${String(i).padStart(15, "0")}01`);
  const dir = seedIndex(base, ids);
  await withMode("memory", async () => {
    const fresh = ["6300000000000000101", "6300000000000000201", "6300000000000000301"];
    const p1 = archiveNewRowsAsync(rowsOf([ids[0], fresh[0], fresh[1]]), "2026-11-06", base);
    const p2 = archiveNewRowsAsync(rowsOf([fresh[1], fresh[2], ids[N - 1]]), "2026-11-06", base);
    await untilLoading();
    assert.equal(dtccIdsStats().memoryCount, null, "nothing partial is exposed while it loads");
    const [r1, r2] = await Promise.all([p1, p2]);
    assert.deepEqual(r1, { newRows: 2, seenTotal: N + 2 });
    assert.deepEqual(r2, { newRows: 1, seenTotal: N + 3 }, "the second call saw the first call's claims: fresh[1] is not archived twice");
    const dayIds = zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-11-06.jsonl.gz"))).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).disseminationId);
    assert.deepEqual(dayIds, fresh, "each fresh row archived exactly once, in call order");
    assert.equal(dtccIdsStats().memoryLoading, false);
    assert.equal(dtccIdsStats().memoryCount, N + 3);
  });
  fs.rmSync(base, { recursive: true, force: true });
});

test("the sliced verdict is never used stale: an ID claimed by another archive call while the slices ran is re-judged at claim time (archived exactly once)", async () => {
  const N = 400_000;
  const base = tmp("dtcc-stall-stale-verdict-");
  const ids = Array.from({ length: N }, (_, i) => `65${String(i).padStart(15, "0")}01`);
  const dir = seedIndex(base, ids);
  const row = (id: string): DtccSwapRow => ({
    disseminationId: id, originalDisseminationId: id, actionType: "NEWT", eventTimestamp: "2026-11-05T12:33:36Z",
    effectiveDate: "2026-11-05", notionalAmountLeg1: 1, notionalCurrencyLeg1: "USD",
    underlierId: "US0378331005", underlierIdSource: "ISIN", underlierName: "APPLE INC",
  });
  await withMode("memory", async () => {
    // the contested id comes FIRST, so the first slice judges it fresh; ~400k lookups follow, several slices
    const big = [row("CONTESTED"), ...ids.map(row), row("ONLY-ASYNC")];
    const pending = archiveNewRowsAsync(big, "2026-11-06", base);
    await untilLoading();
    while (dtccIdsStats().memoryLoading) await new Promise((r) => setImmediate(r));
    // the load is installed and the async verdict is mid-slices: another call claims CONTESTED now
    assert.deepEqual(archiveNewRows([row("CONTESTED")], "2026-11-06", base), { newRows: 1, seenTotal: N + 1 });
    assert.deepEqual(await pending, { newRows: 1, seenTotal: N + 2 }, "only ONLY-ASYNC — CONTESTED was re-judged as seen");
    const dayIds = zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-11-06.jsonl.gz"))).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).disseminationId);
    assert.deepEqual(dayIds, ["CONTESTED", "ONLY-ASYNC"], "each archived exactly once");
  });
  fs.rmSync(base, { recursive: true, force: true });
});

test("a load still in flight at _resetDtccForTests never installs its Set afterwards (epoch guard)", async () => {
  const bigBase = tmp("dtcc-stall-epoch-big-");
  seedIndex(bigBase, Array.from({ length: 150_000 }, (_, i) => `64${String(i).padStart(15, "0")}01`));
  const smallBase = tmp("dtcc-stall-epoch-small-");
  seedIndex(smallBase, ["only-one"]);
  await withMode("memory", async () => {
    const stale = archiveNewRowsAsync(rowsOf(["x"]), "2026-11-06", bigBase);
    await untilLoading();
    _resetDtccForTests();
    assert.equal(dtccIdsStats().memoryLoading, false, "the reset forgot the in-flight load");
    assert.deepEqual(archiveNewRows(rowsOf(["only-one", "y"]), "2026-11-06", smallBase), { newRows: 1, seenTotal: 2 });
    // the stale call finishes against the Set installed AFTER the reset, never the 150k one it was loading
    assert.deepEqual(await stale, { newRows: 1, seenTotal: 3 });
    assert.equal(dtccIdsStats().memoryCount, 3, "the pre-reset 150k-id Set was never installed");
  });
  fs.rmSync(bigBase, { recursive: true, force: true });
  fs.rmSync(smallBase, { recursive: true, force: true });
});
