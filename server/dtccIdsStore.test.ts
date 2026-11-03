// dtccIdsStore.test.ts — the on-disk DTCC swaps dedup ID set (memory fix A1,
// 2026-11-03). Pins the store's contract on its own, with no rows involved:
// schema/pragmas, idempotent inserts, chunked lookups agreeing with per-row
// lookups, the backfill-from-compact-index contract (counts match or an
// ERROR line is written — never silent), the index-line reader's exact
// `split("\n")` semantics, the env-flag parser, and the measured numbers
// behind the PR (1M synthetic ids: backfill, steady-state boot count pass,
// 118k lookups per-row vs batched, 118k routine inserts).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import Database from "better-sqlite3";
import {
  DtccIdsStore, dtccIdsMode, gzipLineBatches, bufferLines, countGzipLines, countBufferLines, countNonEmptyLines,
  defaultDtccIdsDir, DTCC_IDS_DIRNAME,
  DTCC_IDS_DB_FILE, DTCC_IDS_INSERT_BATCH, DTCC_IDS_LOOKUP_CHUNK, DTCC_IDS_STORE_ENV, DTCC_IDS_DEFAULT_MODE,
  _resetDtccIdsModeWarningForTests,
} from "./dtccIdsStore";

const INDEX = "_seen_ids_index.jsonl.gz";

function mkdir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}
function writeIndex(dir: string, ids: readonly string[]): void {
  fs.writeFileSync(path.join(dir, INDEX), zlib.gzipSync(ids.join("\n") + (ids.length ? "\n" : "")));
}
function openStore(dir: string, rebuildIds?: () => Set<string>) {
  const log: string[] = [];
  const err: string[] = [];
  const store = new DtccIdsStore({
    dir, indexPath: path.join(dir, INDEX), rebuildIds,
    log: (l) => log.push(l), logError: (l) => err.push(l),
  });
  return { store, log, err };
}
function ids(n: number, prefix = "id-"): string[] {
  return Array.from({ length: n }, (_, i) => `${prefix}${i}`);
}

test("open: creates _seen_ids.db in the dtccswaps dir with WAL journal + synchronous NORMAL, seen_ids table, empty count", () => {
  const dir = mkdir("dtcc-ids-open-");
  const { store } = openStore(dir);
  try {
    assert.equal(store.path, path.join(dir, DTCC_IDS_DB_FILE));
    assert.ok(fs.existsSync(store.path), "db file created on open");
    assert.equal(store.journalMode, "wal");
    assert.equal(store.synchronous, "1", "SQLite reports NORMAL as 1");
    assert.equal(store.count(), 0);
    assert.equal(store.isReady, false, "not ready until a backfill pass ran");
    // Independent connection sees the same schema (the table really exists
    // on disk, WITHOUT ROWID so the id IS the b-tree key).
    const raw = new Database(store.path, { readonly: true });
    const sql = String(raw.prepare("SELECT sql FROM sqlite_master WHERE name = 'seen_ids'").pluck().get());
    raw.close();
    assert.match(sql, /WITHOUT ROWID/);
    assert.match(sql, /id TEXT PRIMARY KEY/);
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("default dir: with no `dir` the store lives under archiveBaseDir()/dtccswaps — the persistent volume (durability ratchet), honoring DATA_DIR", () => {
  const base = mkdir("dtcc-ids-datadir-");
  const prev = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  let store: DtccIdsStore | null = null;
  try {
    assert.equal(defaultDtccIdsDir(), path.join(base, "datacore_archive", DTCC_IDS_DIRNAME));
    store = new DtccIdsStore({ indexPath: path.join(base, "datacore_archive", DTCC_IDS_DIRNAME, INDEX) });
    assert.equal(store.path, path.join(base, "datacore_archive", "dtccswaps", DTCC_IDS_DB_FILE));
    assert.ok(fs.existsSync(store.path), "created under the archive base, directory made on demand");
    assert.equal(store.count(), 0);
  } finally {
    if (store) store.close();
    if (prev === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prev;
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("insertMany: INSERT OR IGNORE — re-inserts count zero, count() tracks changes, recount() agrees, has() reflects membership", () => {
  const dir = mkdir("dtcc-ids-insert-");
  const { store } = openStore(dir);
  try {
    assert.equal(store.insertMany(["a", "b", "c"]), 3);
    assert.equal(store.insertMany(["b", "c", "d"]), 1, "only d is new");
    assert.equal(store.insertMany([]), 0);
    assert.equal(store.count(), 4);
    assert.equal(store.recount(), 4, "the cached count matches a real COUNT(*)");
    assert.equal(store.has("a"), true);
    assert.equal(store.has("d"), true);
    assert.equal(store.has("zzz"), false);
    // Across the batch boundary (transactions of DTCC_IDS_INSERT_BATCH).
    const many = ids(DTCC_IDS_INSERT_BATCH * 2 + 7, "m-");
    assert.equal(store.insertMany(many), many.length);
    assert.equal(store.count(), 4 + many.length);
    assert.equal(store.recount(), 4 + many.length);
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("missingFrom: chunked IN-list lookups agree with per-row has() across chunk boundaries; duplicate inputs are looked up once and judged like the Set would", () => {
  const dir = mkdir("dtcc-ids-missing-");
  const { store } = openStore(dir);
  try {
    const present = ids(DTCC_IDS_LOOKUP_CHUNK * 2 + 300, "p-");
    store.insertMany(present);
    const probe = [
      ...present.filter((_, i) => i % 3 === 0),
      ...ids(DTCC_IDS_LOOKUP_CHUNK + 11, "absent-"),
      "p-0", "p-0", "absent-0", // duplicates within one poll
    ];
    const missing = store.missingFrom(probe);
    for (const id of probe) {
      assert.equal(missing.has(id), !store.has(id), `batched and per-row verdicts agree for ${id}`);
    }
    assert.equal(missing.size, DTCC_IDS_LOOKUP_CHUNK + 11, "every absent id exactly once, no present id");
    assert.equal(store.missingFrom([]).size, 0);
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("backfill (streamed): db behind the compact index → imports exactly the missing ids, counts match, DTCC-IDS-BACKFILL carries both counts; a second call is a no-op", async () => {
  const dir = mkdir("dtcc-ids-backfill-");
  const all = ids(DTCC_IDS_INSERT_BATCH * 2 + 403, "bf-");
  writeIndex(dir, all);
  const { store, log, err } = openStore(dir);
  try {
    store.insertMany(all.slice(0, 100)); // a db that fell behind
    const r = await store.ensureReadyAsync();
    assert.equal(r.state, "done");
    assert.equal(r.source, "streamed");
    assert.equal(r.dbBefore, 100);
    assert.equal(r.indexCount, all.length);
    assert.equal(r.imported, all.length - 100);
    assert.equal(r.dbAfter, all.length);
    assert.equal(store.count(), all.length);
    assert.equal(store.recount(), all.length);
    assert.equal(store.isReady, true);
    assert.equal(err.length, 0, "no error line on a clean backfill");
    assert.equal(log.length, 1);
    assert.match(log[0], /DTCC-IDS-BACKFILL ok: db 100 -> 10403, index 10403, imported 10303/);
    assert.equal(store.has(all[all.length - 1]), true);

    const again = await store.ensureReadyAsync();
    assert.equal(again, r, "idempotent once ready — same result object, no second pass");
    assert.equal(log.length, 1, "no second log line");
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("backfill (sync twin): same contract as the streamed path, source 'sync'", () => {
  const dir = mkdir("dtcc-ids-backfill-sync-");
  const all = ids(DTCC_IDS_INSERT_BATCH + 9, "s-");
  writeIndex(dir, all);
  const { store, log, err } = openStore(dir);
  try {
    const r = store.ensureReadySync();
    assert.equal(r.state, "done");
    assert.equal(r.source, "sync");
    assert.equal(r.dbBefore, 0);
    assert.equal(r.imported, all.length);
    assert.equal(r.dbAfter, all.length);
    assert.equal(r.indexCount, all.length);
    assert.equal(err.length, 0);
    assert.match(log[0], new RegExp(`db 0 -> ${all.length}, index ${all.length}, imported ${all.length}`));
    assert.equal(store.ensureReadySync(), r, "idempotent");
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("backfill: db already complete → no import, an 'ok' line with equal counts", async () => {
  const dir = mkdir("dtcc-ids-backfill-complete-");
  const all = ids(10, "c-");
  writeIndex(dir, all);
  const { store, log, err } = openStore(dir);
  try {
    store.insertMany(all);
    const r = await store.ensureReadyAsync();
    assert.equal(r.state, "done");
    assert.equal(r.imported, 0);
    assert.equal(r.dbBefore, 10);
    assert.equal(r.dbAfter, 10);
    assert.equal(r.indexCount, 10);
    assert.equal(err.length, 0);
    assert.match(log[0], /db 10 -> 10, index 10, imported 0/);
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("backfill: db AHEAD of the index → DTCC-IDS-BACKFILL-MISMATCH error line with both counts and the rollback hint; store still ready so dedup continues", async () => {
  const dir = mkdir("dtcc-ids-backfill-ahead-");
  writeIndex(dir, ids(5, "a-"));
  const { store, log, err } = openStore(dir);
  try {
    store.insertMany([...ids(5, "a-"), "extra-1", "extra-2"]);
    const r = await store.ensureReadyAsync();
    assert.equal(r.state, "mismatch");
    assert.equal(r.dbAfter, 7);
    assert.equal(r.indexCount, 5);
    assert.equal(r.imported, 0, "nothing to import when the db is ahead");
    assert.equal(store.isReady, true, "a count mismatch is surfaced, not a reason to stop deduping");
    assert.equal(log.length, 0, "the mismatch goes to the ERROR channel, never the info one");
    assert.equal(err.length, 1);
    assert.match(err[0], /DTCC-IDS-BACKFILL-MISMATCH: db 7 != index 5/);
    assert.match(err[0], /rollback to memory mode would re-archive/);
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("backfill: duplicate lines in the index cannot be matched by a distinct-id db → error line names the duplicate-lines cause", () => {
  const dir = mkdir("dtcc-ids-backfill-dups-");
  writeIndex(dir, ["x", "x", "y"]);
  const { store, err } = openStore(dir);
  try {
    const r = store.ensureReadySync();
    assert.equal(r.state, "mismatch");
    assert.equal(r.indexCount, 3);
    assert.equal(r.dbAfter, 2);
    assert.equal(r.imported, 2);
    assert.match(err[0], /db 2 != index 3/);
    assert.match(err[0], /duplicate lines/);
  } finally {
    store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("backfill: no index file → rebuildIds fallback (the memory path's own day-file self-heal), source 'rebuilt'; without a fallback the attempt FAILS loudly and stays retryable", async () => {
  const dir = mkdir("dtcc-ids-backfill-noindex-");
  const withFallback = openStore(dir, () => new Set(["r1", "r2"]));
  try {
    const r = await withFallback.store.ensureReadyAsync();
    assert.equal(r.state, "done");
    assert.equal(r.source, "rebuilt");
    assert.equal(r.indexCount, 2);
    assert.equal(r.dbAfter, 2);
    assert.equal(withFallback.store.has("r2"), true);
  } finally {
    withFallback.store.close();
  }
  const dir2 = mkdir("dtcc-ids-backfill-noindex2-");
  const bare = openStore(dir2);
  try {
    const r = await bare.store.ensureReadyAsync();
    assert.equal(r.state, "failed");
    assert.equal(bare.store.isReady, false, "a failed attempt is retried on the next ensure*");
    assert.match(String(r.error), /no rebuildIds fallback/);
    assert.equal(bare.err.length, 1);
    assert.match(bare.err[0], /DTCC-IDS-BACKFILL-FAILED/);
  } finally {
    bare.store.close();
    fs.rmSync(dir, { recursive: true, force: true });
    fs.rmSync(dir2, { recursive: true, force: true });
  }
});

test("backfill: a corrupt index → 'unreadable' error line, then the rebuild fallback (streamed and sync paths alike)", async () => {
  for (const sync of [false, true]) {
    const dir = mkdir("dtcc-ids-backfill-corrupt-");
    fs.writeFileSync(path.join(dir, INDEX), Buffer.from("definitely not gzip"));
    const { store, err } = openStore(dir, () => new Set(["c1"]));
    try {
      const r = sync ? store.ensureReadySync() : await store.ensureReadyAsync();
      assert.equal(r.state, "done", `sync=${sync}`);
      assert.equal(r.source, "rebuilt");
      assert.equal(r.dbAfter, 1);
      assert.equal(err.length, 1);
      assert.match(err[0], /compact index unreadable/);
    } finally {
      store.close();
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
});

test("gzipLineBatches / bufferLines / count*Lines: split on \\n exactly like split('\\n')+filter(Boolean) — no trimming, empty lines skipped, final unterminated line kept, chunk boundaries and multi-byte text handled", async () => {
  const dir = mkdir("dtcc-ids-lines-");
  try {
    // Long enough (>>16KB decompressed) that the gunzip stream yields many
    // chunks, so lines straddle chunk boundaries; plus a few awkward lines.
    const lines = [...ids(20_000, "4895571859001896"), "", " spaced out ", "multi-byte-ünïcödé-字", "tail-without-newline"];
    const text = lines.join("\n"); // no trailing "\n" → last line unterminated
    const file = path.join(dir, "lines.gz");
    fs.writeFileSync(file, zlib.gzipSync(text));
    const expected = text.split("\n").filter(Boolean);
    const streamed: string[] = [];
    let batches = 0;
    for await (const lines of gzipLineBatches(file)) { streamed.push(...lines); batches++; }
    assert.deepEqual(streamed, expected);
    assert.ok(batches > 5, `the stream really arrived in many chunks (${batches}), so lines straddled chunk boundaries`);
    assert.deepEqual(Array.from(bufferLines(Buffer.from(text, "utf8"))), expected);
    assert.deepEqual(Array.from(bufferLines(Buffer.from("", "utf8"))), []);
    assert.deepEqual(Array.from(bufferLines(Buffer.from("\n\n", "utf8"))), []);
    // The byte-level counters agree with the materialized split, including
    // across chunk boundaries and for the unterminated tail.
    assert.equal(await countGzipLines(file), expected.length);
    assert.equal(countBufferLines(Buffer.from(text, "utf8")), expected.length);
    assert.equal(countBufferLines(Buffer.from("", "utf8")), 0);
    assert.equal(countBufferLines(Buffer.from("\n\n \n", "utf8")), 1, "a whitespace-only line is non-empty, like split+filter says");
    assert.equal(countBufferLines(Buffer.from("a\nb", "utf8")), 2, "unterminated last line counted");
    // A line split across two chunks is counted once, by whichever chunk ends it.
    const state = { lineHasBytes: false };
    const n = countNonEmptyLines(Buffer.from("x\nab", "utf8"), state) + countNonEmptyLines(Buffer.from("cd\n\n", "utf8"), state);
    assert.equal(n + (state.lineHasBytes ? 1 : 0), 2);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("dtccIdsMode: default dual; memory/dual/sqlite parsed case-insensitively; an unknown value falls back to the default and warns ONCE", () => {
  _resetDtccIdsModeWarningForTests();
  assert.equal(DTCC_IDS_DEFAULT_MODE, "dual", "dual is the DEFAULT for this PR (the parent flips to sqlite after a mismatch-free day)");
  assert.equal(dtccIdsMode({}), "dual");
  assert.equal(dtccIdsMode({ [DTCC_IDS_STORE_ENV]: "" }), "dual");
  assert.equal(dtccIdsMode({ [DTCC_IDS_STORE_ENV]: "memory" }), "memory");
  assert.equal(dtccIdsMode({ [DTCC_IDS_STORE_ENV]: " SQLite " }), "sqlite");
  assert.equal(dtccIdsMode({ [DTCC_IDS_STORE_ENV]: "DUAL" }), "dual");
  const errors: string[] = [];
  const orig = console.error;
  console.error = (...args: unknown[]) => { errors.push(args.map(String).join(" ")); };
  try {
    assert.equal(dtccIdsMode({ [DTCC_IDS_STORE_ENV]: "sqlit" }), "dual", "typo → default, never a crash or a silent third behavior");
    assert.equal(dtccIdsMode({ [DTCC_IDS_STORE_ENV]: "sqlit" }), "dual");
    assert.equal(errors.length, 1, "warned once for the same bad value");
    assert.match(errors[0], /DTCC-IDS-MODE: unknown VOLTRADE_DTCC_IDS_STORE="sqlit"/);
  } finally {
    console.error = orig;
    _resetDtccIdsModeWarningForTests();
  }
});

// ── The numbers behind the PR ───────────────────────────────────────────────
// 1,000,000 synthetic 19-char ids (the live shape: "4895571859001896201").
// Ceilings are deliberately loose (CI runners vary 5-10x); the point is the
// printed [bench] lines, which the PR summary quotes. Live scale for
// reference: 6.56M ids archived, ~118k new per day.
test("benchmark — 1M synthetic ids: backfill from the compact index, steady-state boot count pass, 118k lookups (per-row vs batched), 118k routine inserts", async () => {
  const N = 1_000_000;
  const LOOKUPS = 118_000;
  const dir = mkdir("dtcc-ids-bench-");
  const pad = (i: number) => "48955718590" + String(i).padStart(8, "0"); // 19 chars, unique
  const existing = Array.from({ length: N }, (_, i) => pad(i));
  writeIndex(dir, existing);
  const indexBytes = fs.statSync(path.join(dir, INDEX)).size;
  const first = openStore(dir);
  try {
    let t = Date.now();
    const r = await first.store.ensureReadyAsync();
    const backfillMs = Date.now() - t;
    assert.equal(r.state, "done");
    assert.equal(r.imported, N);
    assert.equal(first.store.count(), N);
    assert.equal(first.store.recount(), N);

    // 59k present + 59k absent, interleaved — one day's worth of lookups.
    const probe: string[] = [];
    for (let i = 0; i < LOOKUPS; i++) probe.push(i % 2 === 0 ? existing[(i * 7919) % N] : "99" + String(i).padStart(17, "0"));
    t = Date.now();
    const missing = first.store.missingFrom(probe);
    const batchedMs = Date.now() - t;
    assert.equal(missing.size, LOOKUPS / 2);
    t = Date.now();
    let absent = 0;
    for (const id of probe) if (!first.store.has(id)) absent++;
    const perRowMs = Date.now() - t;
    assert.equal(absent, LOOKUPS / 2);

    const fresh = Array.from({ length: LOOKUPS }, (_, i) => "77" + String(i).padStart(17, "0"));
    t = Date.now();
    assert.equal(first.store.insertMany(fresh), LOOKUPS);
    const insertMs = Date.now() - t;
    assert.equal(first.store.count(), N + LOOKUPS);
    const dbBytes = first.store.dbBytes() ?? 0;
    first.store.close();

    // Steady-state boot: the db is complete, so ensureReadyAsync only COUNTS
    // the index (one streamed pass) and imports nothing.
    writeIndex(dir, [...existing, ...fresh]);
    const second = openStore(dir);
    t = Date.now();
    const r2 = await second.store.ensureReadyAsync();
    const countPassMs = Date.now() - t;
    assert.equal(r2.state, "done");
    assert.equal(r2.imported, 0);
    assert.equal(r2.dbAfter, N + LOOKUPS);
    second.store.close();

    console.log(`[bench] dtccIdsStore 1M ids: index.gz ${(indexBytes / 1e6).toFixed(1)} MB; backfill (count pass + ${N} inserts) ${backfillMs} ms; ` +
      `steady-state boot count pass ${countPassMs} ms; ${LOOKUPS} lookups batched IN ${batchedMs} ms vs per-row has() ${perRowMs} ms; ` +
      `${LOOKUPS} routine inserts ${insertMs} ms; db file ${(dbBytes / 1e6).toFixed(1)} MB (${(dbBytes / (N + LOOKUPS)).toFixed(1)} B/id)`);
    assert.ok(backfillMs < 120_000, `backfill ${backfillMs}ms`);
    assert.ok(countPassMs < 60_000, `count pass ${countPassMs}ms`);
    assert.ok(batchedMs < 30_000, `batched lookups ${batchedMs}ms`);
    assert.ok(perRowMs < 60_000, `per-row lookups ${perRowMs}ms`);
    assert.ok(insertMs < 30_000, `routine inserts ${insertMs}ms`);
  } finally {
    if (first.store.isOpen) first.store.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
