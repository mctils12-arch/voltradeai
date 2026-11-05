// [REPAIR 2026-11-05] The landing hero's counters ("live map layers / data
// streams recording / archived observations") showed only their "—"
// placeholders: /api/data/platform/stats awaited a full recount of every
// archive line on the first request after each boot, and the request
// outlived the edge proxy (HTTP 000 after 123 s, live 2026-11-05 00:31Z).
// These tests pin the fix: the response never waits on a big count, a
// restart serves the persisted total immediately, and files whose size and
// mtime have not changed are never read again.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  countObservations, countFileLines, platformStats, listArchiveDirs,
  _resetPlatformStatsCache, _setLineCounterForTests, PLATFORM_STATS_CACHE_FILE, FIRST_COUNT_WAIT_MS,
} from "./platformStats";

const LAYERS = [{ id: "aircraft", status: "live" }, { id: "vessels", status: "live" }, { id: "tank_fill", status: "planned" }];
const TTL_PLUS = 11 * 60_000;

function makeArchive(): string {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "pstats-inc-"));
  fs.mkdirSync(path.join(base, "aircraft"));
  fs.writeFileSync(path.join(base, "aircraft", "2026-11-04-01.jsonl"), '{"a":1}\n{"a":2}\n{"a":3}\n');
  fs.writeFileSync(path.join(base, "aircraft", "2026-11-04-02.jsonl"), '{"a":4}\n');
  fs.mkdirSync(path.join(base, "vessels"));
  fs.writeFileSync(path.join(base, "vessels", "2026-11-03-23.jsonl.gz"), zlib.gzipSync('{"v":1}\n{"v":2}\n'));
  fs.mkdirSync(path.join(base, "waitlist"));
  fs.writeFileSync(path.join(base, "waitlist", "2026-11-04.jsonl"), '{"email":"x"}\n');
  return base;
}

/** Wraps the real counter and records which files were read. */
function recordingCounter() {
  const reads: string[] = [];
  return { reads, fn: (fp: string) => { reads.push(path.basename(fp)); return countFileLines(fp); } };
}
const hangingCounter = () => new Promise<number>(() => { /* never resolves: a count that outlives any request */ });
const tick = (ms = 30) => new Promise((r) => setTimeout(r, ms));

test("the very first count (nothing persisted) never holds the response past the wait; the counters it can give it gives", async () => {
  const base = makeArchive();
  _resetPlatformStatsCache();
  _setLineCounterForTests(hangingCounter);
  try {
    const t0 = Date.now();
    const s = await platformStats(LAYERS, base, Date.now(), { firstWaitMs: 50 });
    const took = Date.now() - t0;
    assert.ok(took < 1_000, `answered in ${took} ms — the old code awaited the whole count`);
    assert.equal(s.layers_live, 2);
    assert.equal(s.streams_recording, 2);
    assert.equal(s.observations, null, "an unknown count is null, never a fabricated 0");
    assert.equal(s.observations_as_of, null);
    assert.equal(s.observations_pending, true, "the count is running behind the response");
    assert.ok(FIRST_COUNT_WAIT_MS <= 5_000, "the production wait stays far inside the edge proxy's patience");
  } finally {
    _setLineCounterForTests(null);
    _resetPlatformStatsCache();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("a restart serves the persisted total at once, with its own as_of, while the recount runs behind it", async () => {
  const base = makeArchive();
  _resetPlatformStatsCache();
  try {
    const first = await platformStats(LAYERS, base);
    assert.equal(first.observations, 6, "3 + 1 + 2 lines; the waitlist PII stream excluded");
    assert.equal(first.observations_pending, false);
    assert.ok(fs.existsSync(path.join(base, PLATFORM_STATS_CACHE_FILE)), "the count is persisted beside the archive");
    const persistedAsOf = first.observations_as_of;

    // restart: memory gone, the file stays; one archive file grew, and its recount hangs
    _resetPlatformStatsCache();
    fs.appendFileSync(path.join(base, "aircraft", "2026-11-04-02.jsonl"), '{"a":5}\n');
    _setLineCounterForTests(hangingCounter);
    const t0 = Date.now();
    const s = await platformStats(LAYERS, base, Date.now() + TTL_PLUS, { firstWaitMs: 5_000 });
    assert.ok(Date.now() - t0 < 1_000, "a persisted total means no wait at all");
    assert.equal(s.observations, 6, "the last known total, not a spinner and not null");
    assert.equal(s.observations_as_of, persistedAsOf, "served with the time it was counted");
    assert.equal(s.observations_pending, true, "the stale total is refreshing behind the response");
  } finally {
    _setLineCounterForTests(null);
    _resetPlatformStatsCache();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("incremental: unchanged files are never read again; grown and new files are; vanished files leave the total", async () => {
  const base = makeArchive();
  _resetPlatformStatsCache();
  const rec = recordingCounter();
  _setLineCounterForTests(rec.fn);
  try {
    assert.equal(await countObservations(base), 6);
    assert.deepEqual(rec.reads.sort(), ["2026-11-03-23.jsonl.gz", "2026-11-04-01.jsonl", "2026-11-04-02.jsonl"]);
    rec.reads.length = 0;
    assert.equal(await countObservations(base), 6);
    assert.deepEqual(rec.reads, [], "nothing changed, nothing read");
    fs.appendFileSync(path.join(base, "aircraft", "2026-11-04-02.jsonl"), '{"a":5}\n{"a":6}\n');
    fs.writeFileSync(path.join(base, "vessels", "2026-11-04-00.jsonl"), '{"v":3}\n');
    assert.equal(await countObservations(base), 9);
    assert.deepEqual(rec.reads.sort(), ["2026-11-04-00.jsonl", "2026-11-04-02.jsonl"], "only the grown and the new file");
    rec.reads.length = 0;
    fs.rmSync(path.join(base, "aircraft", "2026-11-04-01.jsonl"));
    assert.equal(await countObservations(base), 6, "an offloaded/pruned file's lines leave the total");
    assert.deepEqual(rec.reads, []);
  } finally {
    _setLineCounterForTests(null);
    _resetPlatformStatsCache();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("persisted per-file counts survive a restart: the refresh after a boot reads only what changed", async () => {
  const base = makeArchive();
  _resetPlatformStatsCache();
  try {
    await platformStats(LAYERS, base);                      // full count, persisted
    _resetPlatformStatsCache();                             // restart
    const rec = recordingCounter();
    _setLineCounterForTests(rec.fn);
    fs.writeFileSync(path.join(base, "aircraft", "2026-11-04-03.jsonl"), '{"a":7}\n{"a":8}\n');
    const stale = await platformStats(LAYERS, base, Date.now() + TTL_PLUS);
    assert.equal(stale.observations, 6, "served from the persisted total first");
    for (let i = 0; i < 100 && (await platformStats(LAYERS, base)).observations_pending; i++) await tick();
    const fresh = await platformStats(LAYERS, base);
    assert.equal(fresh.observations, 8, "the refresh added the new file");
    assert.deepEqual(rec.reads, ["2026-11-04-03.jsonl"], "the three unchanged files were trusted from the persisted counts");
    const onDisk = JSON.parse(fs.readFileSync(path.join(base, PLATFORM_STATS_CACHE_FILE), "utf8"));
    assert.equal(onDisk.v, 1);
    assert.equal(onDisk.observations, 8);
    assert.equal(Object.keys(onDisk.files).length, 4, "one entry per archive file, the PII stream never counted");
  } finally {
    _setLineCounterForTests(null);
    _resetPlatformStatsCache();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("a corrupt or foreign cache file is never trusted: the count starts from scratch", async () => {
  const base = makeArchive();
  _resetPlatformStatsCache();
  try {
    fs.writeFileSync(path.join(base, PLATFORM_STATS_CACHE_FILE), "{not json");
    const a = await platformStats(LAYERS, base);
    assert.equal(a.observations, 6);
    _resetPlatformStatsCache();
    fs.writeFileSync(path.join(base, PLATFORM_STATS_CACHE_FILE),
      JSON.stringify({ v: 2, at: Date.now(), observations: 999, files: {} }));
    const b = await platformStats(LAYERS, base);
    assert.equal(b.observations, 6, "an unknown cache version is ignored, its total never served");
  } finally {
    _resetPlatformStatsCache();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("wiring: the route never lets a browser or proxy keep an answer that has no count yet", () => {
  const routes = fs.readFileSync(path.join(path.dirname(new URL(import.meta.url).pathname), "routes.ts"), "utf8");
  const i = routes.indexOf('app.get("/api/data/platform/stats"');
  assert.ok(i > 0, "route registered");
  const handler = routes.slice(i, routes.indexOf("});", i));
  assert.ok(handler.includes('s.observations === null ? "no-store" : "public, max-age=300"'),
    "a pending answer cached for 5 minutes would pin the hero on 'counting…' through every retry");
  assert.ok(handler.indexOf("await platformStats(") < handler.indexOf('res.set("Cache-Control"'),
    "the header is chosen from the answer, after it exists");
});

test("the cache file is a dotfile at the archive root that no directory-enumerating reader picks up", () => {
  const base = makeArchive();
  try {
    fs.writeFileSync(path.join(base, PLATFORM_STATS_CACHE_FILE), "{}");
    assert.ok(PLATFORM_STATS_CACHE_FILE.startsWith("."));
    assert.deepEqual(listArchiveDirs(base).sort(), ["aircraft", "vessels", "waitlist"]);
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});
