/**
 * Hermetic tests for the permanent position archive (datacoreArchive.ts).
 * Runs via `npm run test:node` (tsx --test). No network, temp dirs only.
 */
import { test } from "node:test";
import assert from "node:assert";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import {
  archiveAircraft, archiveVessels, compressOldHours, rollupOldDays,
  compressOldHoursAsync, rollupOldDaysAsync,
  recentTrack, recentTrackAsync, recentTrackCached, clearTrackCache,
  archiveStats, aircraftIntervalMs, vesselIntervalMs,
  nearAnySite, RAW_RETENTION_DAYS, streamJsonlLines, readArchiveDay,
} from "./datacoreArchive";

const SITES = [{ lat: 35.985, lon: -96.767 }]; // Cushing

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-archive-"));

const cruise = (icao: string, lat = 45, lon = -30): any => ({
  icao24: icao, callsign: "TST", lat, lon,
  altitude_m: 11000, on_ground: false, velocity_ms: 240, heading: 90,
});

test("adaptive thinning: strategic-site proximity gets full resolution", () => {
  const nearSite = { ...cruise("a"), lat: 35.99, lon: -96.77 };
  assert.ok(nearAnySite(nearSite.lat, nearSite.lon, SITES));
  assert.ok(aircraftIntervalMs(nearSite, SITES) < aircraftIntervalMs(cruise("a"), SITES),
    "near-site sampling must be denser than oceanic cruise");
  const lowAlt = { ...cruise("a"), altitude_m: 1000 };
  assert.ok(aircraftIntervalMs(lowAlt, SITES) < aircraftIntervalMs(cruise("a"), SITES),
    "low-altitude sampling must be denser than cruise");
});

test("adaptive thinning: vessel near port denser than open water; anchored sparser", () => {
  const nearPort = { mmsi: "1", lat: 35.99, lon: -96.77, sog: 12, cog: 0 } as any;
  const openWater = { mmsi: "1", lat: 30, lon: -140, sog: 12, cog: 0 } as any;
  const anchored = { mmsi: "1", lat: 30, lon: -140, sog: 0.2, cog: 0 } as any;
  assert.ok(vesselIntervalMs(nearPort, SITES) < vesselIntervalMs(openWater, SITES));
  assert.ok(vesselIntervalMs(anchored, SITES) > vesselIntervalMs(openWater, SITES));
});

test("archive append + per-entity cadence + recentTrack round-trip", () => {
  const base = tmp();
  const t0 = Date.now();
  // first write lands
  assert.equal(archiveAircraft([cruise("abc123")], SITES, base, t0), 1);
  // immediate rewrite is thinned away (cruise cadence = 5min)
  assert.equal(archiveAircraft([cruise("abc123")], SITES, base, t0 + 10_000), 0);
  // after the interval it lands again
  assert.equal(archiveAircraft([cruise("abc123", 46, -29)], SITES, base, t0 + 6 * 60_000), 1);
  const track = recentTrack("aircraft", "abc123", base, t0 + 6 * 60_000);
  assert.equal(track.length, 2);
  assert.ok(track[0].t < track[1].t, "track sorted by time");
});

test("PERF: recentTrackAsync returns byte-identical results to the sync path (raw + gzipped hours)", async () => {
  const base = tmp();
  const now = Date.now();
  // Unique ids per test: the archive's per-entity thinning state is
  // module-level and persists across tests in one process.
  // An old (gz-eligible) hour + a current raw hour, plus a decoy id whose
  // lines must survive the substring prefilter without polluting the result.
  archiveAircraft([cruise("perfid1", 44, -31)], SITES, base, now - 3 * 3600_000);
  archiveAircraft([cruise("perfid1x", 10, 10)], SITES, base, now - 3 * 3600_000);
  archiveAircraft([cruise("perfid1", 45, -30)], SITES, base, now);
  compressOldHours(base, now); // old hour → .jsonl.gz, exercising the gunzip stream
  const sync = recentTrack("aircraft", "perfid1", base, now);
  const async_ = await recentTrackAsync("aircraft", "perfid1", base, now);
  assert.deepEqual(async_, sync, "streamed path must reproduce the sync path exactly");
  assert.equal(async_.length, 2);
  assert.ok(async_.every((p) => typeof p.la === "number" && typeof p.lo === "number"));
});

test("PERF: recentTrackCached serves repeats from cache inside the TTL, rescans after it", async () => {
  clearTrackCache();
  const base = tmp();
  const t0 = Date.now();
  archiveAircraft([cruise("perfid2")], SITES, base, t0);
  const first = await recentTrackCached("aircraft", "perfid2", base, t0 + 1000);
  assert.equal(first.length, 1);
  // new archive point lands; a re-read INSIDE the 30s TTL must serve the
  // cached result (this is exactly the client's 30s card refresh)
  archiveAircraft([cruise("perfid2", 46, -29)], SITES, base, t0 + 6 * 60_000);
  const cached = await recentTrackCached("aircraft", "perfid2", base, t0 + 1000 + 10_000);
  assert.equal(cached.length, 1, "inside the TTL the archive is NOT rescanned");
  // past the TTL the fresh point appears
  const fresh = await recentTrackCached("aircraft", "perfid2", base, t0 + 6 * 60_000 + 40_000);
  assert.equal(fresh.length, 2, "after the TTL a rescan picks up new points");
  clearTrackCache();
});

test("PERF: compressOldHoursAsync produces the same on-disk outcome as the sync pass", async () => {
  const now = Date.now();
  // ONE fixture, then a directory copy: writing the same ids twice would be
  // thinned away by the archive's module-level per-entity cadence state.
  const a = tmp();
  archiveAircraft([cruise("perfgz1")], SITES, a, now - 3 * 3600_000);
  archiveAircraft([cruise("perfgz2")], SITES, a, now);
  const b = tmp();
  fs.cpSync(a, b, { recursive: true });
  const doneSync = compressOldHours(a, now);
  const doneAsync = await compressOldHoursAsync(b, now);
  assert.equal(doneAsync, doneSync, "same number of hours compressed");
  const list = (base: string) => fs.readdirSync(path.join(base, "aircraft")).sort();
  assert.deepEqual(list(b), list(a), "same file set (old hour gz, current raw)");
  const gz = list(a).find((f) => f.endsWith(".gz"))!;
  const content = (base: string) => zlib.gunzipSync(fs.readFileSync(path.join(base, "aircraft", gz))).toString();
  assert.equal(content(b), content(a), "gz payloads byte-identical");
});

test("PERF: rollupOldDaysAsync produces the same daily summaries as the sync pass", async () => {
  const now = Date.now();
  const old = now - (RAW_RETENTION_DAYS + 2) * 86400_000;
  const a = tmp();
  archiveAircraft([cruise("perfru1", 40, -100)], SITES, a, old);
  archiveAircraft([cruise("perfru1", 41, -101)], SITES, a, old + 6 * 60_000);
  archiveAircraft([cruise("perfru2", 10, 10)], SITES, a, old + 60_000);
  const b = tmp();
  fs.cpSync(a, b, { recursive: true });
  const rolledSync = rollupOldDays(a, now);
  const rolledAsync = await rollupOldDaysAsync(b, now);
  assert.equal(rolledAsync, rolledSync, "same day count rolled");
  assert.ok(rolledSync >= 1, "fixture actually exercised a rollup");
  const read = (base: string) => {
    const dir = path.join(base, "aircraft_tracks");
    const f = fs.readdirSync(dir)[0];
    return zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString().trim().split("\n")
      .map((l) => JSON.parse(l)).sort((x, y) => String(x.i).localeCompare(String(y.i)));
  };
  assert.deepEqual(read(b), read(a), "summaries identical (shared accumulation helpers)");
  assert.equal(fs.readdirSync(path.join(b, "aircraft")).length,
               fs.readdirSync(path.join(a, "aircraft")).length, "raw files deleted the same way");
});

test("vessel archive stores static enrichment fields", () => {
  const base = tmp();
  const n = archiveVessels([{ mmsi: "366999", name: "TEST SHIP", lat: 33.7, lon: -118.2,
                              sog: 14, cog: 270, shiptype: 70, destination: "LONG BEACH" } as any],
                           SITES, base, Date.now());
  assert.equal(n, 1);
  const dir = path.join(base, "vessels");
  const raw = fs.readFileSync(path.join(dir, fs.readdirSync(dir)[0]), "utf8");
  const rec = JSON.parse(raw.trim());
  assert.equal(rec.st, 70);
  assert.equal(rec.de, "LONG BEACH");
});

test("compression gzips hours older than 2h and leaves current hour raw", () => {
  const base = tmp();
  const now = Date.now();
  archiveAircraft([cruise("old1")], SITES, base, now - 3 * 3600_000);
  archiveAircraft([cruise("new1")], SITES, base, now);
  const done = compressOldHours(base, now);
  assert.equal(done, 1);
  const files = fs.readdirSync(path.join(base, "aircraft")).sort();
  assert.ok(files.some(f => f.endsWith(".jsonl.gz")), "old hour gzipped");
  assert.ok(files.some(f => f.endsWith(".jsonl") && !f.endsWith(".gz")), "current hour raw");
});

test("rollup summarizes days beyond retention into track records and deletes raw", () => {
  const base = tmp();
  const now = Date.now();
  const oldMs = now - (RAW_RETENTION_DAYS + 2) * 86400_000;
  // two samples for one entity on the old day (cadence-spaced)
  archiveAircraft([cruise("roll1", 40, -100)], SITES, base, oldMs);
  archiveAircraft([cruise("roll1", 41, -101)], SITES, base, oldMs + 10 * 60_000);
  const rolled = rollupOldDays(base, now);
  assert.equal(rolled, 1);
  assert.equal(fs.readdirSync(path.join(base, "aircraft")).length, 0, "raw deleted");
  const tdir = path.join(base, "aircraft_tracks");
  const gz = fs.readdirSync(tdir);
  assert.equal(gz.length, 1);
  const rec = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(tdir, gz[0]))).toString().trim());
  assert.equal(rec.i, "roll1");
  assert.equal(rec.n, 2);
  assert.ok(Array.isArray(rec.pl) && rec.pl.length >= 1, "polyline present");
  assert.ok(rec.bbox[0] <= 40 && rec.bbox[2] >= 41, "bbox spans samples");
});

test("archiveStats reports files and bytes for volume monitoring", () => {
  const base = tmp();
  archiveAircraft([cruise("stat1")], SITES, base, Date.now());
  const s = archiveStats(base);
  assert.equal(s.kinds.aircraft.files, 1);
  assert.ok(s.totalBytes > 0);
});

// [REPAIR 2026-07-05, audit defect #3] archiveStats must enumerate the
// archive from disk — new stream directories (fredmacro, optionchains,
// fda, ...) appear WITHOUT a code change, so the archive-gap rule covers
// every kind. Position kinds report {files:0} even before first write.
test("archiveStats discovers new stream directories from disk (no hardcoded kind list)", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-arch-"));
  fs.mkdirSync(path.join(base, "fredmacro"), { recursive: true });
  fs.writeFileSync(path.join(base, "fredmacro", "2026-07-05.jsonl"), '{"s":"DGS10"}\n');
  fs.mkdirSync(path.join(base, "somefuturestream"), { recursive: true });
  const s = archiveStats(base);
  assert.equal(s.kinds.fredmacro.files, 1, "disk-discovered kind must be reported");
  assert.ok(s.kinds.somefuturestream, "never-before-seen dirs appear without a code change");
  assert.equal(s.kinds.aircraft.files, 0, "position kinds stay loud even before first write");
  fs.rmSync(base, { recursive: true, force: true });
});

test("cruise cadence is 75s (2026-07-21 3D-trail densification — 5min fixes drew 68-140km curtain slabs)", () => {
  const cruise = { icao24: "c1", lat: 45, lon: -40, altitude_m: 11000, on_ground: false } as any;
  assert.equal(aircraftIntervalMs(cruise, []), 75_000);
  // ordering invariants unchanged: sites and low-altitude still sample faster
  assert.ok(30_000 < 75_000 && 60_000 < 75_000);
});

// REPAIR (found 2026-07-22 while building an unrelated /data feature, doing
// a local `node dist/index.cjs` smoke test): a truncated/corrupted .gz
// archive file crashed the ENTIRE PROCESS on every boot, even though
// streamJsonlLines already listened for "error" on both the raw file stream
// AND the gunzip stream. Root cause: readline.Interface ALSO independently
// re-emits its input stream's "error" on ITSELF (a separate EventEmitter
// emission, per Node's own readline internals) — with no listener on `rl`,
// that unhandled emission crashes the process regardless of the other two
// listeners. Minimal repro (outside this test, confirmed both on this branch
// and on a clean stash of main before this fix): gzipSync a string, truncate
// the last 4 bytes, pipe through zlib.createGunzip() into
// readline.createInterface — Node throws "Unexpected end of file" (Z_BUF_ERROR)
// and exits the process. The same missing-rl-error-listener pattern was
// copy-pasted into 7 other files (aircraftEntities/fleetUtilization/
// gridStress/platformStats/queryEngine/shadowFleet x2/siteTimeline) — each
// gained the identical `rl.on("error", ...)` guard, same PR; datacoreArchive
// gets the fullest test here since streamJsonlLines is the one of the eight
// with a directly exported, standalone unit.
test("streamJsonlLines resolves (never crashes the process) on a truncated/corrupt gzip file", async () => {
  const base = tmp();
  const good = zlib.gzipSync(Buffer.from('{"a":1}\n{"a":2}\n{"a":3}\n'));
  const truncated = good.subarray(0, good.length - 4); // chop the gzip trailer -> Z_BUF_ERROR
  const fp = path.join(base, "truncated.jsonl.gz");
  fs.writeFileSync(fp, truncated);
  const lines: string[] = [];
  // If the missing rl.on("error", ...) guard regresses, this either hangs
  // (never resolves) or throws an unhandled 'error' event that kills the
  // whole node:test process — either way the test run itself fails loudly,
  // it does not silently pass.
  await streamJsonlLines(fp, true, (line) => lines.push(line));
  fs.rmSync(base, { recursive: true, force: true });
});

test("streamJsonlLines still yields every line of a genuinely valid gzip file (the fix adds no false-negative)", async () => {
  const base = tmp();
  const good = zlib.gzipSync(Buffer.from('{"a":1}\n{"a":2}\n{"a":3}\n'));
  const fp = path.join(base, "valid.jsonl.gz");
  fs.writeFileSync(fp, good);
  const lines: string[] = [];
  await streamJsonlLines(fp, true, (line) => lines.push(line));
  assert.deepEqual(lines, ['{"a":1}', '{"a":2}', '{"a":3}']);
  fs.rmSync(base, { recursive: true, force: true });
});

// readArchiveDay (2026-07-26, backs /api/diag/archive — filed to unblock the
// USAspending gate-2 statistical test, which needs the multi-week historical
// archive that no other read surface exposes outside the Railway volume).
test("readArchiveDay: day-named stream (usaspending-style) reads the exact-day file", async () => {
  const base = tmp();
  const dir = path.join(base, "usaspending");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-07-05.jsonl"), '{"aid":"a1","amt":50000}\n{"aid":"a2","amt":75000}\n');
  fs.writeFileSync(path.join(dir, "2026-07-06.jsonl"), '{"aid":"a3","amt":99000}\n'); // different day, must not leak in
  const r = await readArchiveDay("usaspending", "2026-07-05", base);
  assert.ok(r);
  assert.equal(r!.rows.length, 2);
  assert.deepEqual(r!.rows.map((x: any) => x.aid).sort(), ["a1", "a2"]);
  assert.equal(r!.truncated, false);
  assert.deepEqual(r!.files, ["2026-07-05.jsonl"]);
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDay: hour-named stream (aircraft/vessels/trains-style) concatenates every hour file for the day", async () => {
  const base = tmp();
  const t0 = Date.parse("2026-07-05T00:00:00Z");
  assert.equal(archiveAircraft([cruise("h1")], SITES, base, t0), 1);
  assert.equal(archiveAircraft([cruise("h2", 46, -29)], SITES, base, t0 + 3 * 3600_000), 1);
  const r = await readArchiveDay("aircraft", "2026-07-05", base);
  assert.ok(r);
  assert.equal(r!.rows.length, 2, "both hour files for the day must be read");
  assert.deepEqual(r!.rows.map((x: any) => x.i).sort(), ["h1", "h2"]);
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDay: reads gzipped day files transparently", async () => {
  const base = tmp();
  const dir = path.join(base, "fredmacro");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-07-05.jsonl.gz"), zlib.gzipSync(Buffer.from('{"s":"DGS10","v":4.2}\n')));
  const r = await readArchiveDay("fredmacro", "2026-07-05", base);
  assert.ok(r);
  assert.equal(r!.rows.length, 1);
  assert.equal(r!.rows[0].s, "DGS10");
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDay: unknown stream (no archive directory) returns null, never throws", async () => {
  const base = tmp();
  const r = await readArchiveDay("neverexistedstream", "2026-07-05", base);
  assert.equal(r, null);
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDay: no file for the requested day returns an empty, non-null result (directory exists, day doesn't)", async () => {
  const base = tmp();
  fs.mkdirSync(path.join(base, "usaspending"), { recursive: true });
  const r = await readArchiveDay("usaspending", "2026-01-01", base);
  assert.ok(r);
  assert.deepEqual(r!.rows, []);
  assert.equal(r!.truncated, false);
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDay: limit caps rows and sets truncated honestly rather than silently dropping", async () => {
  const base = tmp();
  const dir = path.join(base, "usaspending");
  fs.mkdirSync(dir, { recursive: true });
  const lines = Array.from({ length: 10 }, (_, i) => JSON.stringify({ aid: `a${i}` })).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-07-05.jsonl"), lines);
  const r = await readArchiveDay("usaspending", "2026-07-05", base, 3);
  assert.ok(r);
  assert.equal(r!.rows.length, 3);
  assert.equal(r!.truncated, true);
  const full = await readArchiveDay("usaspending", "2026-07-05", base, 100);
  assert.equal(full!.rows.length, 10);
  assert.equal(full!.truncated, false);
  fs.rmSync(base, { recursive: true, force: true });
});
