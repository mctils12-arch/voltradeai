// [A2 2026-11-03] Incremental shadow-fleet fold — exactness proofs and the
// shadow-compare harness (memory fix A2: fold each closed hour once, combine
// per-hour partials each cycle; VOLTRADE_SHADOWFLEET_FOLD=full|shadow|incremental).
//
// Every exactness test compares THREE computations on the same archive:
// the sync materializing reference (computeShadowStats, tests-only), the
// original online fold (mode=full), and the incremental fold (cold cache,
// then warm cache). The fixtures deliberately carry the shapes that make a
// naive merge wrong: a gap whose dark period spans many hour files, a zone
// loiter spanning hours, a vessel seen in exactly one hour, exact duplicate
// lines, a STALE duplicate written into the wrong hour (breaks the hour
// invariant), out-of-order timestamps inside an hour, the cutoff-straddling
// oldest hour and the still-open current hour.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  computeShadowStats, computeShadowStatsAsync, shadowFleetFoldStats, _resetShadowFoldForTests,
  _shadowFoldCacheForTests, SHADOWFLEET_FOLD_ENV, ShadowAggregator, foldVesselArchiveAsync,
  SHADOW_STRIDE, type ShadowZone, type ShadowStats,
} from "./shadowFleet";
import { StringInterner, listVesselWindowFiles } from "./vesselArchiveFold";

const NOW = Date.UTC(2026, 6, 4, 12, 30, 0); // 12:30Z — hours <= 11 closed, hour 12 open
const t = (hoursAgo: number, from = NOW) => Math.floor((from - hoursAgo * 3600_000) / 1000);
const hourFile = (tSec: number) => { const d = new Date(tSec * 1000).toISOString(); return `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl`; };

type Line = Record<string, unknown> & { t: number; _file?: string };

/** Writes points into hour files by their own `t` (the live writer's
 *  layout); `_file` forces a line into another hour's file (an invariant
 *  break the live writer cannot produce — used to prove the fallback). */
function writeArchive(base: string, points: Line[], gzEvery = 0): string[] {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const byFile = new Map<string, string[]>();
  for (const p of points) {
    const { _file, ...rest } = p;
    const f = _file ?? hourFile(p.t);
    if (!byFile.has(f)) byFile.set(f, []);
    byFile.get(f)!.push(JSON.stringify(rest));
  }
  const names: string[] = [];
  let k = 0;
  for (const [f, lines] of byFile) {
    const body = lines.join("\n") + "\n";
    if (gzEvery && ++k % gzEvery === 0) { fs.writeFileSync(path.join(dir, f + ".gz"), zlib.gzipSync(body)); names.push(f + ".gz"); }
    else { fs.writeFileSync(path.join(dir, f), body); names.push(f); }
  }
  return names;
}

const ZONE: ShadowZone = { id: "laconian_gulf", name: "Laconian Gulf", lat: 36.55, lon: 22.75, radius_km: 45 };
const ZONES = [ZONE];

async function withMode<T>(mode: string, fn: () => Promise<T>): Promise<T> {
  const prev = process.env[SHADOWFLEET_FOLD_ENV];
  process.env[SHADOWFLEET_FOLD_ENV] = mode;
  try { return await fn(); } finally {
    if (prev === undefined) delete process.env[SHADOWFLEET_FOLD_ENV]; else process.env[SHADOWFLEET_FOLD_ENV] = prev;
  }
}

/** The adversarial fixture described in the file header. */
function adversarialFixture(): Line[] {
  const pts: Line[] = [
    // GAPPER (tanker): dark 10h across many hour files, reappears ~400km away
    { t: t(20), i: "111000111", c: "GAPPER", la: 36.0, lo: 20.0, v: 12, st: 80 },
    { t: t(10), i: "111000111", c: "GAPPER", la: 36.0, lo: 24.5, v: 12, st: 80 },
    // NEAR-MISS gapper: 10h dark but only ~20km -> no event
    { t: t(20), i: "222000222", c: "STEADY", la: 40.0, lo: 5.0, v: 3 },
    { t: t(10), i: "222000222", c: "STEADY", la: 40.15, lo: 5.1, v: 3 },
    // LOITERER: sustained slow presence in the zone spanning six hour files
    { t: t(9), i: "777000777", c: "IDLER", la: 36.5, lo: 22.7, v: 0.4 },
    { t: t(8.5), i: "777000777", c: "IDLER", la: 36.51, lo: 22.71, v: 0.3 },
    { t: t(7), i: "777000777", c: "IDLER", la: 36.52, lo: 22.72, v: 0.6 },
    { t: t(6.2), i: "777000777", c: "IDLER", la: 36.5, lo: 22.7, v: 0.2 },
    { t: t(5), i: "777000777", c: "IDLER", la: 36.51, lo: 22.71, v: 0.2 },
    { t: t(3), i: "777000777", c: "IDLER", la: 36.5, lo: 22.7, v: 0.5 },
    // TRANSIT through the same zone, fast -> must not count
    { t: t(6), i: "888000888", c: "TRANSIT", la: 36.4, lo: 22.6, v: 14 },
    { t: t(5), i: "888000888", c: "TRANSIT", la: 36.6, lo: 22.9, v: 14 },
    { t: t(4), i: "888000888", c: "TRANSIT", la: 36.8, lo: 23.2, v: 14 },
    // NAME REUSE under two MMSIs (different hours)
    { t: t(30), i: "333000333", c: "TWIN STAR", la: 10, lo: 10, v: 5, st: 81 },
    { t: t(29), i: "444000444", c: "TWIN STAR", la: 30, lo: 30, v: 5 },
    // HULL SWAP: A last seen, B first seen 3h later 5km away (different hours)
    { t: t(20), i: "555000555", c: "OLD ID", la: 10.5, lo: 10.7, v: 0 },
    { t: t(17), i: "666000666", c: "NEW ID", la: 10.53, lo: 10.72, v: 0 },
    // ONE-HOUR-ONLY vessel: two points, same hour file, nothing elsewhere
    { t: t(40.4), i: "999000999", c: "BRIEF", la: -20, lo: 50, v: 9 },
    { t: t(40.2), i: "999000999", c: "BRIEF", la: -20.01, lo: 50.01, v: 9 },
    // OUT-OF-ORDER inside one hour (07:54, 07:06, 07:30 in file order), then a point next hour
    { t: t(4.6), i: "123123123", c: "SHUFFLE", la: 0.5, lo: 0.5, v: 4 },
    { t: t(5.4), i: "123123123", c: "SHUFFLE", la: 0.3, lo: 0.3, v: 4 },
    { t: t(5.0), i: "123123123", c: "SHUFFLE", la: 0.4, lo: 0.4, v: 4 },
    { t: t(3.9), i: "123123123", c: "SHUFFLE", la: 0.6, lo: 0.6, v: 4 },
    // EXACT DUPLICATE lines inside one hour (both count in points_read)
    { t: t(15), i: "246246246", c: "DOUBLE", la: 45, lo: -30, v: 7, st: 70 },
    { t: t(15), i: "246246246", c: "DOUBLE", la: 45, lo: -30, v: 7, st: 70 },
    // STRADDLING oldest hour (72h window, cutoff 12:30 three days ago): one point
    // just before the cutoff (excluded), one just after (included)
    { t: t(72.3), i: "101010101", c: "EDGE", la: 60, lo: 60, v: 2 },
    { t: t(71.9), i: "101010101", c: "EDGE", la: 60.01, lo: 60.01, v: 2 },
    // OPEN current hour (12:xx): streamed live every cycle
    { t: t(0.4), i: "135135135", c: "LIVE", la: 50, lo: 0, v: 11 },
    { t: t(0.2), i: "135135135", c: "LIVE", la: 50.1, lo: 0.1, v: 11 },
  ];
  return pts;
}

/** A STALE duplicate of an hour-21 line written into hour 22's file — a point
 *  whose timestamp lies outside its file's hour. The live writer cannot do
 *  this (t and the hour file come from one `now`); it breaks the invariant
 *  the shadow partial's exact merge relies on, so that file must be streamed
 *  live (never cached) and the result must still be identical. */
function staleDuplicate(): Line {
  return { t: t(15), i: "246246246", c: "DOUBLE", la: 45, lo: -30, v: 7, st: 70, _file: hourFile(t(14)) };
}

async function threeWay(base: string, zones: ShadowZone[], windowHours: number, now: number):
    Promise<{ sync: ShadowStats; full: ShadowStats; cold: ShadowStats; warm: ShadowStats }> {
  const sync = computeShadowStats(zones, windowHours, base, now);
  const full = await withMode("full", () => computeShadowStatsAsync(zones, windowHours, base, now));
  const cold = await withMode("incremental", () => computeShadowStatsAsync(zones, windowHours, base, now));
  const warm = await withMode("incremental", () => computeShadowStatsAsync(zones, windowHours, base, now));
  return { sync, full, cold, warm };
}

test("EXACT: incremental fold (cold and warm cache) == original online fold == sync reference on the adversarial fixture, incl. gz files", async () => {
  _resetShadowFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-inc-"));
  writeArchive(base, adversarialFixture(), 3);
  const { sync, full, cold, warm } = await threeWay(base, ZONES, 72, NOW);
  assert.deepEqual(full, sync, "mode=full must still equal the sync reference (untouched behavior)");
  assert.deepEqual(cold, full, "cold incremental (every closed hour built this cycle, merged via partials) must equal full");
  assert.deepEqual(warm, full, "warm incremental (closed hours from cache) must equal full");
  // The fixture really exercises what it claims to.
  assert.equal(full.gap_events, 1);
  assert.equal(full.gap_examples[0].mmsi, "111000111");
  assert.equal(full.loiter_by_zone.laconian_gulf, 1);
  assert.ok(full.identity_candidates >= 2, "name reuse + hull swap");
  assert.equal(full.points_read, adversarialFixture().length - 1, "the pre-cutoff straddler point is excluded, duplicates both counted");
  const s = shadowFleetFoldStats();
  assert.equal(s.mode, "incremental");
  const listing = listVesselWindowFiles(72, base, NOW);
  const cacheable = listing.files.filter((f) => f.cacheable).length;
  assert.ok(cacheable >= 10, `fixture must span many closed hours (got ${cacheable})`);
  assert.equal(s.filesCachedLastRun, cacheable, "warm cycle served every closed hour from cache");
  assert.equal(s.filesStreamedLastRun, listing.files.length - cacheable, "warm cycle streamed only the open hour and the cutoff-straddling hour");
  assert.equal(s.partialsCached, cacheable);
  assert.equal(s.fallbacks, 0);
  assert.ok(s.partialBytesEstimate > 0 && s.lastIncrementalMs != null);
});

test("EXACT under an hour-invariant break: a stale duplicate written into the wrong hour makes that file uncacheable (streamed live), result still identical, no fallback needed", async () => {
  _resetShadowFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-inc-stale-"));
  writeArchive(base, [...adversarialFixture(), staleDuplicate()]);
  const { sync, full, cold, warm } = await threeWay(base, ZONES, 72, NOW);
  assert.deepEqual(full, sync);
  assert.deepEqual(cold, full);
  assert.deepEqual(warm, full);
  const s = shadowFleetFoldStats();
  assert.equal(s.uncacheableFiles, 1, "exactly the file carrying the foreign-hour timestamp is marked uncacheable");
  assert.equal(s.fallbacks, 0, "an uncacheable file is streamed live, it is not a fallback");
  const cacheable = listVesselWindowFiles(72, base, NOW).files.filter((f) => f.cacheable).length;
  assert.equal(s.partialsCached, cacheable - 1);
});

test("EXACT across a sliding window: as `now` advances hour by hour the cache is reused, evicted beyond the window, and every cycle equals a fresh full fold", async () => {
  _resetShadowFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-inc-slide-"));
  // 90 hours of hourly points for a few vessels drifting around (and one
  // gapping periodically), so every window boundary move changes the answer.
  const pts: Line[] = [];
  for (let h = 90; h >= 0; h--) {
    for (let v = 0; v < 6; v++) {
      if (v === 5 && h % 16 >= 8) continue; // vessel 5 is dark half the time -> gap events as it reappears far away
      pts.push({ t: t(h + 0.25 + (v % 3) * 0.2), i: `20000000${v}`, c: `DRIFT ${v}`,
                 la: 30 + v + (v === 5 ? (h % 16 < 8 ? 0 : 5) : 0) + h * 0.01, lo: 10 + v + (v === 5 && h % 32 < 16 ? 4 : 0), v: v === 2 ? 0.3 : 6 });
    }
    // zone loiterer every other hour
    if (h % 2 === 0) pts.push({ t: t(h + 0.5), i: "300000003", c: "ZONER", la: 36.5 + (h % 3) * 0.01, lo: 22.7, v: 0.4 });
  }
  writeArchive(base, pts, 2);
  const WINDOW = 72;
  let maxCached = 0;
  for (let step = 0; step <= 8; step++) {
    const now = NOW + step * 3600_000 + (step % 2) * 1_800_000; // alternate :30 and :00 anchoring
    const full = await withMode("full", () => computeShadowStatsAsync(ZONES, WINDOW, base, now));
    const inc = await withMode("incremental", () => computeShadowStatsAsync(ZONES, WINDOW, base, now));
    assert.deepEqual(inc, full, `step ${step}: incremental must equal full at now=${new Date(now).toISOString()}`);
    assert.deepEqual(inc, computeShadowStats(ZONES, WINDOW, base, now), `step ${step}: and the sync reference`);
    const s = shadowFleetFoldStats();
    maxCached = Math.max(maxCached, s.partialsCached);
    assert.ok(s.partialsCached <= WINDOW + 2, `cache cap: ${s.partialsCached} partials > ${WINDOW + 2}`);
    const cacheable = listVesselWindowFiles(WINDOW, base, now).files.filter((f) => f.cacheable).length;
    assert.equal(s.partialsCached, cacheable, `step ${step}: exactly the window's closed hours are cached (old ones evicted)`);
    assert.equal(s.fallbacks, 0);
  }
  assert.ok(maxCached >= 60, `the fixture must have exercised a nearly-full cache (max ${maxCached})`);
  assert.ok(shadowFleetFoldStats().mode === "incremental");
});

test("SHADOW MODE (the default): serves the full fold, compares it against the incremental one every cycle, counts matches; a poisoned partial is caught as FOLD-MISMATCH and the full result is still served", async () => {
  _resetShadowFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-inc-shadow-"));
  writeArchive(base, adversarialFixture(), 2);
  const full = await withMode("full", () => computeShadowStatsAsync(ZONES, 72, base, NOW));
  const errors: string[] = [];
  const origErr = console.error;
  console.error = (...a: unknown[]) => { errors.push(a.map(String).join(" ")); };
  try {
    const prev = process.env[SHADOWFLEET_FOLD_ENV];
    delete process.env[SHADOWFLEET_FOLD_ENV]; // unset == shadow, the default for this PR
    try {
      const r1 = await computeShadowStatsAsync(ZONES, 72, base, NOW);
      const r2 = await computeShadowStatsAsync(ZONES, 72, base, NOW);
      assert.deepEqual(r1, full); assert.deepEqual(r2, full);
    } finally { if (prev === undefined) delete process.env[SHADOWFLEET_FOLD_ENV]; else process.env[SHADOWFLEET_FOLD_ENV] = prev; }
    let s = shadowFleetFoldStats();
    assert.equal(s.mode, "shadow");
    assert.equal(s.matches, 2, "both the cold and the warm cycle compared equal");
    assert.equal(s.mismatches, 0);
    assert.equal(s.lastMismatchAt, null);
    assert.equal(errors.length, 0, "no FOLD-MISMATCH on a correct incremental fold");
    assert.ok(s.lastFullMs != null && s.lastIncrementalMs != null && s.lastFullMs >= 0 && s.lastIncrementalMs >= 0,
      "shadow mode records both the pass time and the incremental-cost estimate");

    // Poison one cached partial: pretend that hour held one more point.
    const cache = _shadowFoldCacheForTests(base, ZONES);
    const victim = listVesselWindowFiles(72, base, NOW).files.find((f) => f.cacheable)!;
    const partial = cache.peek(victim.name)!;
    partial.points += 1;
    const r3 = await withMode("shadow", () => computeShadowStatsAsync(ZONES, 72, base, NOW));
    assert.deepEqual(r3, full, "shadow mode SERVES THE FULL FOLD even when the incremental one disagrees");
    s = shadowFleetFoldStats();
    assert.equal(s.mismatches, 1);
    assert.ok(s.lastMismatchAt != null);
    assert.equal(errors.length, 1);
    assert.match(errors[0], /^FOLD-MISMATCH shadowfleet /);
    assert.match(errors[0], /points_read: \d+ vs \d+/, "the diff summary names the field");

    // In incremental mode the same poison would reach the served result —
    // which is exactly why the switch waits for 24h of zero mismatches.
    const r4 = await withMode("incremental", () => computeShadowStatsAsync(ZONES, 72, base, NOW));
    assert.equal(r4.points_read, full.points_read + 1);
  } finally { console.error = origErr; }
});

test("FULL MODE: the original path, untouched — equals the sync reference and never builds a partial", async () => {
  _resetShadowFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-inc-full-"));
  writeArchive(base, adversarialFixture());
  const r = await withMode("full", () => computeShadowStatsAsync(ZONES, 72, base, NOW));
  assert.deepEqual(r, computeShadowStats(ZONES, 72, base, NOW));
  const s = shadowFleetFoldStats();
  assert.equal(s.mode, "full");
  assert.equal(s.partialsCached, 0);
  assert.equal(s.partialBytesEstimate, 0);
  assert.ok(s.lastFullMs != null && s.lastRunAt != null);
});

test("FALLBACK: the same closed hour present as .jsonl AND .jsonl.gz (the compressOldHours window) violates the partial precondition -> full fold runs, result identical, fallbacks counted", async () => {
  _resetShadowFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-inc-dup-"));
  const names = writeArchive(base, adversarialFixture());
  const victim = names.find((n) => n === hourFile(t(20)))!; // the hour holding GAPPER's dark point
  const fp = path.join(base, "vessels", victim);
  fs.writeFileSync(fp + ".gz", zlib.gzipSync(fs.readFileSync(fp))); // both files now exist
  const { sync, full, cold, warm } = await threeWay(base, ZONES, 72, NOW);
  assert.deepEqual(full, sync, "the original fold reads both copies (an existing transient) — the reference does too");
  assert.deepEqual(cold, full);
  assert.deepEqual(warm, full);
  const s = shadowFleetFoldStats();
  assert.equal(s.fallbacks, 2, "both incremental cycles had to fall back (the duplicate hour fails the newer-than precondition)");
  assert.ok(s.lastFullMs != null, "the fallback's full pass is timed as the full fold");

  // In shadow mode the same situation skips the compare instead of
  // reporting a mismatch — the incremental result is discarded, not wrong.
  const r = await withMode("shadow", () => computeShadowStatsAsync(ZONES, 72, base, NOW));
  assert.deepEqual(r, full);
  const s2 = shadowFleetFoldStats();
  assert.equal(s2.mismatches, 0);
  assert.equal(s2.compareSkipped, 1);
  assert.equal(s2.fallbacks, 3);
});

test("absorbPartial unit: exact composition of first/prev/gaps/names/zone runs across two hour partials, gap emission order preserved, precondition enforced", async () => {
  // Hour A (06:xx) and hour B (07:xx): vessel X dark 7h+ between A and an
  // earlier point, vessel Y only in B, vessel Z in both with a zone run that
  // crosses the boundary, two vessels gapping into B in a fixed line order.
  const zones = [ZONE];
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-absorb-"));
  const A = t(6.5), B = t(5.5); // 06:00 and 07:00 files
  writeArchive(base, [
    { t: t(14), i: "X", c: "XRAY", la: 10, lo: 10, v: 5 },
    { t: t(14), i: "W", c: "WHISKY", la: -10, lo: -10, v: 5 },
    { t: A, i: "Z", c: "ZULU", la: 36.5, lo: 22.7, v: 0.3 },
    { t: A + 600, i: "Z", c: "ZULU", la: 36.51, lo: 22.71, v: 0.4 },
    // hour B, in this line order: W reappears far away, then X reappears far away
    { t: B, i: "W", c: "WHISKY", la: -10, lo: -15, v: 5 },
    { t: B + 60, i: "X", c: "XRAY", la: 10, lo: 15, v: 5 },
    { t: B + 120, i: "Y", c: "YANKEE", la: 0, lo: 0, v: 1 },
    { t: B + 180, i: "Z", c: "ZULU", la: 36.52, lo: 22.72, v: 0.2 },
    { t: B + 240, i: "Z", c: "ZULU", la: 36.5, lo: 22.7, v: 0.2 },
  ]);
  const files = listVesselWindowFiles(72, base, NOW).files;
  const interner = new StringInterner();
  // Build the partials for every file through the real builder path.
  const partials = [];
  for (const f of files) {
    const b = new ShadowAggregator(zones, undefined, undefined, { trackOrder: true });
    const pts: Array<[string, Parameters<ShadowAggregator["push"]>[1]]> = [];
    await foldVesselArchiveAsync(72, (m, p) => { if (Math.floor(p.t / 3600) === Math.floor(f.stampSec / 3600)) pts.push([m, p]); }, base, NOW);
    for (const [m, p] of pts) b.push(m, p);
    const partial = b.exportPartial(f.stampSec, interner);
    assert.ok(partial, `${f.name} satisfies the hour invariant`);
    partials.push(partial);
  }
  const merged = new ShadowAggregator(zones);
  for (const p of partials) assert.equal(merged.absorbPartial(p, interner), true);
  const sequential = new ShadowAggregator(zones);
  await foldVesselArchiveAsync(72, (m, p) => sequential.push(m, p), base, NOW);
  const mFin = merged.finish(72), sFin = sequential.finish(72);
  assert.deepEqual(mFin, sFin);
  assert.equal(sFin.gap_events, 2);
  // Both gaps have identical gapHours (8.5h) -> finish()'s stable sort keeps
  // emission order: W (first line of hour B) before X.
  assert.deepEqual(sFin.gap_examples.map((g) => g.mmsi), ["W", "X"]);
  assert.deepEqual(mFin.gap_examples.map((g) => g.mmsi), ["W", "X"], "merged fold emits the cross-hour gaps in the sequential fold's line order");
  assert.equal(sFin.loiter_by_zone.laconian_gulf, 0, "Z's zone run spans only ~1h: below the 4h loiter bar");
  assert.deepEqual(merged.gate1Inputs(), sequential.gate1Inputs());

  // Precondition: absorbing hour A again after hour B (its points are not
  // newer than what the aggregator retains) must be refused, not merged.
  assert.equal(merged.absorbPartial(partials[partials.length - 2], interner), false);

  // Partial shape sanity: one row per vessel, ids resolvable, bytes positive.
  const pB = partials[partials.length - 1];
  assert.equal(pB.n, 4);
  assert.equal(pB.cols.length, 4 * SHADOW_STRIDE);
  assert.equal(interner.str(pB.cols[0]), "W", "vessels are stored in first-appearance order");
  assert.ok(pB.bytes > pB.cols.byteLength);
});

test("exportPartial refuses a file whose points leave its hour (the invariant the merge relies on) — such a file is streamed live, never cached", () => {
  const b = new ShadowAggregator(ZONES, undefined, undefined, { trackOrder: true });
  const stamp = t(5.5) - (t(5.5) % 3600);
  b.push("A", { t: stamp + 10, la: 1, lo: 1 });
  b.push("A", { t: stamp + 3600, la: 1, lo: 1 }); // first second of the NEXT hour
  assert.equal(b.exportPartial(stamp, new StringInterner()), null);
  const ok = new ShadowAggregator(ZONES, undefined, undefined, { trackOrder: true });
  ok.push("A", { t: stamp, la: 1, lo: 1 });
  ok.push("A", { t: stamp + 3599, la: 1, lo: 1 });
  assert.ok(ok.exportPartial(stamp, new StringInterner()));
  assert.throws(() => new ShadowAggregator(ZONES).exportPartial(stamp, new StringInterner()), /trackOrder/);
});

test("MEASURE (7-day synthetic archive, 168h window): incremental warm cycle is a small fraction of the full fold; partial cache is a small fraction of the raw archive; outputs identical", async () => {
  _resetShadowFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-inc-7d-"));
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  // 250 vessels, ~10 archived fixes per vessel-hour (the live archive's
  // order of magnitude), a loiter zone small enough that only ~10% of the
  // fleet ever enters it (live zones are STS anchorages, not the whole coast).
  const HOURS = 169, VESSELS = 250, PTS = 10;
  let rawBytes = 0, total = 0;
  for (let h = HOURS; h >= 0; h--) {
    const lines: string[] = [];
    const hourStart = Math.floor((NOW - h * 3600_000) / 1000);
    const hourStartAligned = hourStart - (hourStart % 3600);
    for (let v = 0; v < VESSELS; v++) {
      if ((v + h) % 7 === 0) continue; // vessels come and go across hours
      for (let k = 0; k < PTS; k++) {
        const ts = hourStartAligned + k * Math.floor(3600 / PTS) + (v % 37);
        if (ts > Math.floor(NOW / 1000)) continue;
        lines.push(JSON.stringify({ i: `V${v}`, t: ts, c: `VESSEL ${v}`,
          la: 33 + (v % 50) * 0.01 + Math.sin(h / 3 + v) * 0.1,
          lo: -118 - (v % 50) * 0.01 + Math.cos(h / 5 + v) * 0.1,
          v: (v % 9 === 0) ? 0.2 : (v + k) % 14, st: v % 4 === 0 ? 80 : 70 }));
        total++;
      }
    }
    const body = lines.join("\n") + "\n";
    rawBytes += body.length;
    const name = hourFile(hourStartAligned);
    if (h % 2 === 0) fs.writeFileSync(path.join(dir, name + ".gz"), zlib.gzipSync(body));
    else fs.writeFileSync(path.join(dir, name), body);
  }
  const zones: ShadowZone[] = [{ id: "z1", name: "Test Zone", lat: 33.45, lon: -118.45, radius_km: 5 }];
  const time = async (fn: () => Promise<ShadowStats>) => { const s = performance.now(); const r = await fn(); return { r, ms: performance.now() - s }; };
  const full = await time(() => withMode("full", () => computeShadowStatsAsync(zones, 168, base, NOW)));
  const heapBefore = process.memoryUsage().heapUsed;
  const cold = await time(() => withMode("incremental", () => computeShadowStatsAsync(zones, 168, base, NOW)));
  const heapAfterCold = process.memoryUsage().heapUsed;
  const warm = await time(() => withMode("incremental", () => computeShadowStatsAsync(zones, 168, base, NOW)));
  const warm2 = await time(() => withMode("incremental", () => computeShadowStatsAsync(zones, 168, base, NOW)));
  assert.deepEqual(cold.r, full.r);
  assert.deepEqual(warm.r, full.r);
  assert.deepEqual(warm2.r, full.r);
  assert.ok(full.r.points_read > 100_000 && full.r.vessels_seen === VESSELS, `fixture scale: ${full.r.points_read} points, ${full.r.vessels_seen} vessels`);
  const s = shadowFleetFoldStats();
  const bestWarm = Math.min(warm.ms, warm2.ms);
  console.log(`[A2 measure shadowfleet 168h] points=${full.r.points_read} rawJsonlBytes=${rawBytes} ` +
    `fullMs=${full.ms.toFixed(0)} coldIncMs=${cold.ms.toFixed(0)} warmIncMs=${bestWarm.toFixed(0)} ` +
    `partials=${s.partialsCached} partialBytesEstimate=${s.partialBytesEstimate} ` +
    `heapDeltaAfterColdMB=${((heapAfterCold - heapBefore) / 1048576).toFixed(1)} filesStreamedWarm=${s.filesStreamedLastRun}`);
  assert.ok(bestWarm < full.ms / 3, `warm incremental ${bestWarm.toFixed(0)}ms should be well under a third of the full fold ${full.ms.toFixed(0)}ms`);
  assert.ok(s.partialBytesEstimate < rawBytes / 4, `partials ${s.partialBytesEstimate}B must be far smaller than the raw window ${rawBytes}B`);
  assert.equal(s.filesStreamedLastRun, 2, "warm cycle streams only the open hour and the cutoff-straddling hour");
  assert.equal(s.fallbacks, 0);
  fs.rmSync(base, { recursive: true, force: true });
});
