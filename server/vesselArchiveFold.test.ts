// vesselArchiveFold.ts — the shared incremental-fold machinery (memory fix
// A2, 2026-11-03). Module-agnostic pieces only; the per-module exactness
// proofs live in shadowFleetIncremental.test.ts / portDwellIncremental.test.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  readFoldMode, DEFAULT_FOLD_MODE, listVesselWindowFiles, streamVesselFile, CLOSED_HOUR_MARGIN_SEC,
  histAdd, histMergeInto, histMedian, StringInterner, PartialCache, PartialCacheRegistry,
  foldDiffSummary, FoldStatsTracker, type WindowFile, type CachedPartial, type SpeedHist,
} from "./vesselArchiveFold";

const NOW = Date.UTC(2026, 6, 4, 12, 30, 0);
const NOW_SEC = Math.floor(NOW / 1000);
const hourFileName = (tSec: number, gz = false) => {
  const d = new Date(tSec * 1000).toISOString();
  return `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl${gz ? ".gz" : ""}`;
};

function writeHour(base: string, tSec: number, lines: string[], gz = false): string {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const name = hourFileName(tSec, gz);
  const body = lines.join("\n") + "\n";
  fs.writeFileSync(path.join(dir, name), gz ? zlib.gzipSync(body) : body);
  return name;
}

test("readFoldMode: unset -> shadow (the default), explicit values honored (case-insensitive), unknown -> shadow", () => {
  assert.equal(DEFAULT_FOLD_MODE, "shadow");
  assert.equal(readFoldMode("X_FOLD", {}), "shadow");
  assert.equal(readFoldMode("X_FOLD", { X_FOLD: "full" }), "full");
  assert.equal(readFoldMode("X_FOLD", { X_FOLD: " Incremental " }), "incremental");
  assert.equal(readFoldMode("X_FOLD", { X_FOLD: "shadow" }), "shadow");
  const warnings: string[] = [];
  const orig = console.warn;
  console.warn = (...a: unknown[]) => { warnings.push(a.map(String).join(" ")); };
  try {
    assert.equal(readFoldMode("X_FOLD_TYPO", { X_FOLD_TYPO: "incrmental" }), "shadow");
    assert.equal(readFoldMode("X_FOLD_TYPO", { X_FOLD_TYPO: "incrmental" }), "shadow");
  } finally { console.warn = orig; }
  assert.equal(warnings.length, 1, "an unrecognized value warns exactly once per variable");
  assert.match(warnings[0], /X_FOLD_TYPO/);
});

test("listVesselWindowFiles: same file set + order as the original fold filter; only closed, fully-inside hours are cacheable", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-fold-list-"));
  const h = (hoursAgo: number) => NOW_SEC - Math.round(hoursAgo * 3600);
  // 12:30 now. Hour files: 09:xx (closed), 10:xx (closed), 11:xx (ended at 12:00,
  // but within the 2-min... no: ended 30 min ago -> closed), 12:xx (open).
  writeHour(base, h(3), ['{"t":1,"i":"a","la":1,"lo":1}']);          // 09
  writeHour(base, h(2), ['{"t":1,"i":"a","la":1,"lo":1}'], true);    // 10 (gz)
  writeHour(base, h(1), ['{"t":1,"i":"a","la":1,"lo":1}']);          // 11
  writeHour(base, h(0), ['{"t":1,"i":"a","la":1,"lo":1}']);          // 12 (open)
  writeHour(base, h(80), ['{"t":1,"i":"a","la":1,"lo":1}']);         // far outside a 72h window
  fs.writeFileSync(path.join(base, "vessels", "_partials.json"), "{}"); // non-hour name: ignored
  const listing = listVesselWindowFiles(72, base, NOW);
  assert.equal(listing.nowSec, NOW_SEC);
  assert.equal(listing.cutoff, NOW_SEC - 72 * 3600);
  assert.deepEqual(listing.files.map((f) => f.name), [
    hourFileName(h(3)), hourFileName(h(2), true), hourFileName(h(1)), hourFileName(h(0)),
  ], "sorted hour files inside [now-(W+1)h, now], nothing else");
  assert.deepEqual(listing.files.map((f) => f.cacheable), [true, true, true, false],
    "closed hours cacheable; the open hour never");
  assert.ok(listing.files.every((f) => f.size > 0 && f.mtimeMs > 0), "stat captured for validation");

  // The hour that ended less than CLOSED_HOUR_MARGIN_SEC ago is still live.
  const justAfter = (h(1) - (h(1) % 3600)) * 1000 + 3600_000 + (CLOSED_HOUR_MARGIN_SEC - 1) * 1000; // 12:01:59
  const l2 = listVesselWindowFiles(72, base, justAfter);
  const hour11 = l2.files.find((f) => f.name === hourFileName(h(1)))!;
  assert.equal(hour11.cacheable, false, "within the margin the just-ended hour is streamed live");

  // A 3h window at 12:30 has cutoff 09:30: the 09 file straddles the cutoff -> live.
  const l3 = listVesselWindowFiles(3, base, NOW);
  assert.deepEqual(l3.files.map((f) => [f.name.slice(11, 13), f.cacheable]), [["09", false], ["10", true], ["11", true], ["12", false]]);

  // Missing archive dir -> empty listing, never a throw.
  assert.deepEqual(listVesselWindowFiles(72, path.join(base, "nope"), NOW).files, []);
});

test("streamVesselFile: identical accept predicate to the original fold (cutoff/now bounds, null la/lo, missing i), counts bad lines, reports ok=false on a truncated gzip", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-fold-stream-"));
  const cutoff = NOW_SEC - 72 * 3600;
  const name = writeHour(base, NOW_SEC - 7200, [
    JSON.stringify({ t: NOW_SEC - 7200, i: "A", la: 1, lo: 2, v: 3.5, c: "ALPHA", st: 80 }),
    JSON.stringify({ t: cutoff - 1, i: "B", la: 1, lo: 2 }),         // before cutoff
    JSON.stringify({ t: NOW_SEC + 1, i: "C", la: 1, lo: 2 }),        // after now
    JSON.stringify({ t: NOW_SEC - 7100, i: "D", la: null, lo: 2 }),  // null la
    JSON.stringify({ t: NOW_SEC - 7100, la: 1, lo: 2 }),             // no i
    "not json at all",
    "42",                                                            // valid JSON, not an object
    JSON.stringify({ t: cutoff, i: "E", la: 0, lo: 0 }),             // exactly at cutoff: kept
    JSON.stringify({ t: NOW_SEC, i: "F", la: 0, lo: 0 }),            // exactly at now: kept
  ]);
  const got: Array<[string, number]> = [];
  const r = await streamVesselFile(path.join(base, "vessels", name), cutoff, NOW_SEC, (m, p) => got.push([m, p.t]));
  assert.equal(r.ok, true);
  assert.equal(r.badLines, 2);
  assert.deepEqual(got, [["A", NOW_SEC - 7200], ["E", cutoff], ["F", NOW_SEC]]);

  const good = zlib.gzipSync(JSON.stringify({ t: NOW_SEC - 100, i: "Z", la: 1, lo: 1 }) + "\n");
  const trunc = path.join(base, "vessels", "trunc.jsonl.gz");
  fs.writeFileSync(trunc, good.subarray(0, good.length - 4));
  const r2 = await streamVesselFile(trunc, cutoff, NOW_SEC, () => {});
  assert.equal(r2.ok, false, "a partial read must be reported, never silently equal to a full one");
  const r3 = await streamVesselFile(path.join(base, "vessels", "missing.jsonl"), cutoff, NOW_SEC, () => {});
  assert.equal(r3.ok, false);
});

test("histMedian: exactly sorted[floor(n/2)] for every multiset, including merges — the detectors' median formula", () => {
  let seed = 7;
  const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (let trial = 0; trial < 200; trial++) {
    const n = 1 + Math.floor(rand() * 40);
    const speeds: number[] = [];
    const hA: SpeedHist = new Map(), hB: SpeedHist = new Map();
    for (let i = 0; i < n; i++) {
      const v = Math.round(rand() * 60) / 10; // 0.1-kt grid like the archive, many ties
      speeds.push(v);
      histAdd(i % 2 ? hA : hB, v);
    }
    const expected = speeds.slice().sort((a, b) => a - b)[Math.floor(n / 2)];
    const merged: SpeedHist = new Map(hA);
    histMergeInto(merged, hB);
    assert.equal(histMedian(merged, n), expected, `trial ${trial}: n=${n}`);
  }
  assert.equal(histMedian(new Map([[2, 1], [9, 1]]), 2), 9, "even n: the upper-middle element, as sorted[1] of [2,9]");
  assert.equal(histMedian(new Map([[0, 3]]), 3), 0);
});

interface FakePartial extends CachedPartial { ids: number[] }
const fake = (ids: number[], bytes = 10): FakePartial => ({
  ids, bytes,
  remapIds(map) { for (let i = 0; i < this.ids.length; i++) this.ids[i] = map(this.ids[i]); },
});
const wf = (name: string, stampSec: number, size = 1, mtimeMs = 1): WindowFile =>
  ({ name, fp: `/x/${name}`, stampSec, size, mtimeMs, cacheable: true });

test("PartialCache: size/mtime-validated get, prune to the wanted set then to the cap, and string-table compaction after evictions", () => {
  const cache = new PartialCache<FakePartial>();
  const a = cache.interner.id("MMSI-A"), b = cache.interner.id("MMSI-B"), c = cache.interner.id("MMSI-C");
  cache.set(wf("h1", 1000), fake([a, b]));
  cache.set(wf("h2", 2000), fake([b, c]));
  cache.set(wf("h3", 3000), fake([c]));
  assert.equal(cache.size, 3);
  assert.ok(cache.get(wf("h1", 1000)), "same size/mtime -> hit");
  assert.equal(cache.get(wf("h1", 1000, 2, 1)), undefined, "size changed -> miss, entry dropped");
  assert.equal(cache.size, 2);

  cache.set(wf("h1", 1000, 2, 1), fake([a]));
  // h1 falls out of the window; h2/h3 stay; cap of 1 then drops the older h2.
  const evicted = cache.prune(new Set(["h2", "h3"]), 1);
  assert.equal(evicted, 2);
  assert.equal(cache.size, 1);
  const h3 = cache.get(wf("h3", 3000))!;
  assert.equal(cache.interner.size, 1, "compaction released ids no surviving partial references");
  assert.equal(cache.interner.str(h3.ids[0]), "MMSI-C", "surviving partial's ids were remapped onto the new table");

  // uncacheable marks are validated by size/mtime too, survive while the file
  // is still listed (even once it is no longer cacheable), and go with the window
  cache.markUncacheable(wf("bad", 4000, 5, 5));
  assert.equal(cache.isUncacheable(wf("bad", 4000, 5, 5)), true);
  assert.equal(cache.isUncacheable(wf("bad", 4000, 6, 5)), false, "a changed file gets a fresh build attempt");
  cache.prune(new Set(["h3"]), 10, new Set(["h3", "bad"]));
  assert.equal(cache.isUncacheable(wf("bad", 4000, 5, 5)), true, "still listed -> mark kept");
  cache.prune(new Set(["h3"]), 10);
  assert.equal(cache.isUncacheable(wf("bad", 4000, 5, 5)), false, "no longer listed -> mark dropped");
  // a partial whose file is listed but no longer cacheable (the cutoff-straddling hour) is dropped
  cache.set(wf("h4", 4000), fake([cache.interner.id("MMSI-D")]));
  cache.prune(new Set(["h3"]), 10, new Set(["h3", "h4"]));
  assert.equal(cache.peek("h4"), undefined);
  assert.ok(cache.bytesEstimate() > 0);
});

test("PartialCacheRegistry: one cache per key, LRU-bounded, totals aggregate", () => {
  const reg = new PartialCacheRegistry<FakePartial>(2);
  const c1 = reg.forKey("dir1");
  c1.set(wf("h1", 1), fake([c1.interner.id("A")], 100));
  reg.forKey("dir2");
  assert.equal(reg.forKey("dir1"), c1, "same key -> same cache");
  reg.forKey("dir3"); // evicts the least-recently-used dir2
  assert.equal(reg.forKey("dir1"), c1, "dir1 was refreshed and survives");
  assert.notEqual(reg.forKey("dir2").size, 1, "dir2 was evicted and comes back empty");
  const t = reg.totals();
  assert.ok(t.partialsCached >= 1 && t.partialBytesEstimate >= 100);
});

test("foldDiffSummary: [] on identical; arrays compared in order; object/Map key order ignored; NaN equals NaN; bounded output", () => {
  assert.deepEqual(foldDiffSummary({ a: 1, b: [1, 2], m: new Map([["k", 1]]) }, { b: [1, 2], a: 1, m: new Map([["k", 1]]) }), []);
  assert.deepEqual(foldDiffSummary({ a: NaN }, { a: NaN }), []);
  assert.deepEqual(foldDiffSummary({ a: [1, 2] }, { a: [2, 1] }), ["a[0]: 1 vs 2", "a[1]: 2 vs 1"]);
  assert.deepEqual(foldDiffSummary({ a: [1] }, { a: [1, 2] }), ["a.length: 1 vs 2"]);
  assert.deepEqual(foldDiffSummary({ n: 1, s: "x" }, { n: 2, s: "x", extra: true }), ["extra: undefined vs true", "n: 1 vs 2"]);
  assert.deepEqual(foldDiffSummary(new Map([["p", [1]]]), new Map([["p", [1]], ["q", []]])), ['<root>["q"]: missing vs present'].map((s) => s.replace("<root>", "")));
  assert.deepEqual(foldDiffSummary({ s: new Set([1, 2]) }, { s: new Set([2, 3]) }), ["s: Set differs (only-a 1, only-b 1)"]);
  const big = foldDiffSummary(Array.from({ length: 50 }, (_, i) => i), Array.from({ length: 50 }, (_, i) => -i), 3);
  assert.equal(big.length, 3, "capped");
  assert.deepEqual(foldDiffSummary({ name: undefined }, {}), [], "an undefined property equals a missing one — JSON.stringify drops both");
});

test("FoldStatsTracker: snapshot is a copy, bump/update mutate, reset returns to defaults", () => {
  const t = new FoldStatsTracker();
  const s0 = t.snapshot();
  assert.equal(s0.mode, "shadow");
  assert.equal(s0.mismatches, 0);
  t.bump("mismatches"); t.update({ lastFullMs: 42 });
  assert.equal(s0.mismatches, 0, "snapshots do not alias the live record");
  assert.equal(t.snapshot().mismatches, 1);
  assert.equal(t.snapshot().lastFullMs, 42);
  t.reset();
  assert.deepEqual(t.snapshot(), s0);
  assert.deepEqual(Object.keys(s0).sort(), [
    "compactDrops", "compareSkipped", "fallbacks", "filesCachedLastRun", "filesStreamedLastRun", "lastFullMs",
    "lastIncrementalMs", "lastMismatchAt", "lastRunAt", "matches", "mismatches", "mode",
    "partialBytesEstimate", "partialsCached", "uncacheableFiles",
  ], "the /api/health diagnostic's field set is pinned");
});
