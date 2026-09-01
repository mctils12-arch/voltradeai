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
  readArchiveDayEvenSample, originOfPosType, oldestRawHour,
  archiveFileTimestampRanges, archiveDayFiles,
} from "./datacoreArchive";

const SITES = [{ lat: 35.985, lon: -96.767 }]; // Cushing

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-archive-"));

/** A timestamp safely beyond RAW_RETENTION_DAYS, anchored to 12:00 UTC.
 *
 *  UTC-MIDNIGHT FIX (Q15, 2026-08-14). These fixtures used
 *  `now - (RAW_RETENTION_DAYS + 2) * 86400_000` directly. Because that offset
 *  is a whole number of days, the result inherits `now`'s TIME OF DAY — so when
 *  the suite ran within ~10 minutes of UTC midnight, the second sample each
 *  fixture writes at `+6..10 min` landed on the NEXT UTC day. `archiveAircraft`
 *  names files `YYYY-MM-DD-HH.jsonl` and `rollupOldDays` groups by that day
 *  string, so one intended day became two: `rolled` came back 2 where the test
 *  asserts 1, and the hold-back test found a second day it had not corrupted
 *  and deleted it, so its "nothing deleted" assertion saw 1 instead of 0.
 *
 *  Caught by T1.1's baseline run and confirmed by experiment rather than
 *  argument: both tests failed at 23:55Z and passed at 01:00Z on the same
 *  commit. A ~1h nightly red window is exactly the kind of thing that destroys
 *  trust in a new gate, so this is FIXED rather than quarantined — it was
 *  always a bug in the test, never in the code under test.
 *
 *  Anchoring to midday makes the fixture's date arithmetic independent of when
 *  the suite happens to run: a +/- few-minute sample can no longer cross a date
 *  boundary. Still comfortably beyond retention — the shift is at most 12h
 *  against a 2-day margin (retention 30d, fixture 32d). */
const oldDayMidday = (now: number): number => {
  const d = new Date(now - (RAW_RETENTION_DAYS + 2) * 86400_000);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 12, 0, 0);
};

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

// Q15 REGRESSION (2026-08-14). Pins the UTC-midnight bug deterministically so
// it can never come back on a clock rather than on a diff. The fixtures above
// are written at `oldDayMidday(now)`; this drives them with a `now` fixed
// INSIDE the failure window and asserts both samples land on one UTC day.
//
// The bug was invisible for a reason worth remembering: these tests pass 23
// hours a day. They were found by T1.1's baseline run happening to execute at
// 23:55Z, and confirmed by re-running at 01:00Z on the same commit. A test that
// depends on wall-clock time is a test that will eventually red a merge gate
// for reasons nobody can reproduce the next morning.
test("Q15: rollup fixtures do not straddle a UTC day when the suite runs near midnight", () => {
  // 23:55:00Z — five minutes before the boundary, so the fixtures' `+10 min`
  // second sample would cross it under the old `now - 32d` arithmetic.
  const now = Date.UTC(2026, 7, 13, 23, 55, 0);

  // The naive value this replaced: same time-of-day as `now`, so +10 min rolls
  // the date over. Asserting it here keeps the regression test honest — if this
  // ever stops being true, the test below has stopped proving anything.
  const naive = now - (RAW_RETENTION_DAYS + 2) * 86400_000;
  assert.notEqual(
    new Date(naive).toISOString().slice(0, 10),
    new Date(naive + 10 * 60_000).toISOString().slice(0, 10),
    "fixture premise: the naive offset must straddle midnight at 23:55Z",
  );

  const anchored = oldDayMidday(now);
  assert.equal(
    new Date(anchored).toISOString().slice(0, 10),
    new Date(anchored + 10 * 60_000).toISOString().slice(0, 10),
    "anchored offset must keep both samples on one UTC day",
  );
  assert.ok(anchored < now - RAW_RETENTION_DAYS * 86400_000,
    "anchored fixture must still be beyond RAW_RETENTION_DAYS");

  // End-to-end: the real archive + rollup path, driven at the bad hour.
  const base = tmp();
  archiveAircraft([cruise("q15roll", 40, -100)], SITES, base, anchored);
  archiveAircraft([cruise("q15roll", 41, -101)], SITES, base, anchored + 10 * 60_000);
  const days = new Set(fs.readdirSync(path.join(base, "aircraft")).map((f) => f.slice(0, 10)));
  assert.equal(days.size, 1, `fixture spans ${days.size} UTC days: ${[...days].join(", ")}`);
  assert.equal(rollupOldDays(base, now), 1, "exactly one day rolled");
});

test("PERF: rollupOldDaysAsync produces the same daily summaries as the sync pass", async () => {
  const now = Date.now();
  const old = oldDayMidday(now);
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

// [2026-08-19 GATE-1 finding] oldestRawHour reports the true raw-retention
// boundary so callers (the portdwell_window diag probe) can tell "no raw
// data because it was rolled up" from "no raw data because nothing
// happened" instead of a bare, misleading zero.
test("oldestRawHour: no directory returns null, never throws", () => {
  const base = tmp();
  assert.equal(oldestRawHour("vessels", base), null);
});

test("oldestRawHour: reports the earliest hour-file timestamp among several", () => {
  const base = tmp();
  const now = Date.now();
  archiveVessels([{ mmsi: "1", name: "A", lat: 10, lon: 10, sog: 1, cog: 0 } as any],
                 SITES, base, now - 5 * 3600_000);
  archiveVessels([{ mmsi: "1", name: "A", lat: 10, lon: 10, sog: 1, cog: 0 } as any],
                 SITES, base, now - 2 * 3600_000);
  archiveVessels([{ mmsi: "1", name: "A", lat: 10, lon: 10, sog: 1, cog: 0 } as any],
                 SITES, base, now);
  const dir = path.join(base, "vessels");
  assert.equal(fs.readdirSync(dir).length, 3, "three distinct hour files written");
  const oldest = oldestRawHour("vessels", base);
  assert.ok(oldest != null);
  // the oldest file's hour-stamp, not the exact write time within that hour
  const expectedHourStart = Math.floor((now - 5 * 3600_000) / 3600_000) * 3600_000;
  assert.equal(oldest, expectedHourStart);
});

test("oldestRawHour: sees gzipped hours the same as raw .jsonl ones", () => {
  const base = tmp();
  const now = Date.now();
  // distinct mmsi from the previous test — the write-cadence dedup cache
  // (lastWrite) is module-level/shared across tests in this file, so a
  // reused mmsi with an earlier `now` here would look "too soon since last
  // write" against the previous test's later timestamp and silently write
  // nothing.
  archiveVessels([{ mmsi: "9", name: "A", lat: 10, lon: 10, sog: 1, cog: 0 } as any],
                 SITES, base, now - 10 * 3600_000);
  compressOldHours(base, now); // gzips the >2h-old hour in place
  const dir = path.join(base, "vessels");
  const files = fs.readdirSync(dir);
  assert.ok(files.some((f) => f.endsWith(".gz")), "sanity: the fixture file was actually gzipped");
  assert.ok(oldestRawHour("vessels", base) != null, "gzipped hour files still count as raw retention");
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
  const oldMs = oldDayMidday(now);
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

// [REPAIR, 2026-08-06 full-code-review finding "rollup deleting unreadable
// hour files"] rollupOldDays previously unlinked every file in a day's
// group unconditionally, even ones that had thrown while being read/
// decompressed a moment earlier — a corrupt hour file's data was silently
// dropped from the summary AND the raw evidence deleted in the same pass.
// Archive gaps never refill (GOAL Priority 1), so an unreadable file must
// block deletion of its whole day, not get silently discarded.
test("rollup holds back (does not delete) a day containing an unreadable hour file", () => {
  const base = tmp();
  const now = Date.now();
  const oldMs = oldDayMidday(now);
  archiveAircraft([cruise("good1", 40, -100)], SITES, base, oldMs);
  archiveAircraft([cruise("good1", 41, -101)], SITES, base, oldMs + 10 * 60_000);
  const dir = path.join(base, "aircraft");
  const realFile = fs.readdirSync(dir)[0];
  const day = realFile.slice(0, 10);
  const realHour = realFile.slice(11, 13);
  const corruptHour = realHour === "00" ? "01" : "00"; // guaranteed distinct from the real file's hour
  // a second hour file for the SAME day, corrupted (garbage bytes, not a
  // real gzip stream) so gunzipSync throws on read.
  fs.writeFileSync(path.join(dir, `${day}-${corruptHour}.jsonl.gz`), Buffer.from("not gzip"));
  const before = fs.readdirSync(dir).sort();
  const rolled = rollupOldDays(base, now);
  assert.equal(rolled, 0, "the whole day is held back, not partially rolled");
  assert.deepEqual(fs.readdirSync(dir).sort(), before, "no raw file deleted while one is unreadable");
  assert.ok(!fs.existsSync(path.join(base, "aircraft_tracks")), "no summary written for an unread day");
});

test("rollupOldDaysAsync holds back a day containing an unreadable hour file", async () => {
  const base = tmp();
  const now = Date.now();
  const oldMs = oldDayMidday(now);
  archiveAircraft([cruise("good2", 40, -100)], SITES, base, oldMs);
  const dir = path.join(base, "aircraft");
  const realFile = fs.readdirSync(dir)[0];
  const day = realFile.slice(0, 10);
  const realHour = realFile.slice(11, 13);
  const corruptHour = realHour === "00" ? "01" : "00";
  fs.writeFileSync(path.join(dir, `${day}-${corruptHour}.jsonl.gz`), Buffer.from("not gzip"));
  const before = fs.readdirSync(dir).sort();
  const rolled = await rollupOldDaysAsync(base, now);
  assert.equal(rolled, 0, "the whole day is held back, not partially rolled");
  assert.deepEqual(fs.readdirSync(dir).sort(), before, "no raw file deleted while one is unreadable");
});

test("streamJsonlLines reports false on a corrupt gzip file, without throwing", async () => {
  const base = tmp();
  const fp = path.join(base, "corrupt.jsonl.gz");
  fs.writeFileSync(fp, Buffer.from("not gzip"));
  const lines: string[] = [];
  const ok = await streamJsonlLines(fp, true, (l) => lines.push(l));
  assert.equal(ok, false);
  assert.equal(lines.length, 0);
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

// readArchiveDay rowFilter (2026-08-21): fixes the same defect class
// readArchiveDayEvenSample's rowFilter fixed for hour-file streams, but for
// a single-file-per-day GLOBAL-population stream (e.g. `fires`, fetched
// world-wide) — confirmed live via /api/diag/archive?stream=fires that every
// real day hits `truncated:true` at the probe's 5,000 cap, so a caller
// isolating rows near a handful of facilities was getting an arbitrary
// first-N-in-file-order slice, not a representative one.
test("readArchiveDay: rowFilter applied inline keeps only matching rows and does not count non-matching rows against limit", async () => {
  const base = tmp();
  const dir = path.join(base, "fires");
  fs.mkdirSync(dir, { recursive: true });
  // 8 rows far from the target region, 2 rows inside it, in file order
  // BEFORE the 2 matching rows — the old (unfiltered) reader at limit=2
  // would return only the 2 far rows and report truncated, hiding both
  // real matches entirely.
  const far = Array.from({ length: 8 }, (_, i) => JSON.stringify({ id: `far${i}`, lat: 60 + i, lon: -140 }));
  const near = [JSON.stringify({ id: "near0", lat: 36.0, lon: -96.5 }), JSON.stringify({ id: "near1", lat: 36.1, lon: -96.6 })];
  fs.writeFileSync(path.join(dir, "2026-08-20.jsonl"), [...far, ...near].join("\n") + "\n");
  const inBox = (row: Record<string, unknown>) => {
    const lat = Number(row.lat), lon = Number(row.lon);
    return lat >= 35 && lat <= 37 && lon >= -98 && lon <= -95;
  };

  const filtered = await readArchiveDay("fires", "2026-08-20", base, 5000, inBox);
  assert.ok(filtered);
  assert.deepEqual(filtered!.rows.map((r) => r.id).sort(), ["near0", "near1"],
    "both in-box rows must be returned regardless of their position in the file");
  assert.equal(filtered!.truncated, false, "10 total rows is well under the limit — reading the whole file is not truncation");

  const capped = await readArchiveDay("fires", "2026-08-20", base, 1, inBox);
  assert.ok(capped);
  assert.equal(capped!.rows.length, 1, "limit still caps the number of MATCHING rows returned");
  assert.equal(capped!.truncated, true, "a real match existed beyond the cap");

  const unfiltered = await readArchiveDay("fires", "2026-08-20", base, 2);
  assert.deepEqual(unfiltered!.rows.map((r) => r.id), ["far0", "far1"],
    "omitting rowFilter is byte-identical to the pre-2026-08-21 behavior");
  fs.rmSync(base, { recursive: true, force: true });
});

// archiveFileTimestampRanges (2026-09-01, closes KNOWN BROKEN #37's own
// filed NEXT step (2): report each file's OWN embedded `t` range so a
// filename/content mismatch is visible without Railway volume access).
test("archiveFileTimestampRanges: reports the true min/max t per file, unaffected by any row limit", async () => {
  const base = tmp();
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const rows0 = [100, 200, 150].map((t) => JSON.stringify({ t, i: "m1" }));
  fs.writeFileSync(path.join(dir, "2026-08-05-00.jsonl"), rows0.join("\n") + "\n");
  const rows1 = [9000, 9500].map((t) => JSON.stringify({ t, i: "m2" }));
  fs.writeFileSync(path.join(dir, "2026-08-05-01.jsonl"), rows1.join("\n") + "\n");
  const files = archiveDayFiles(dir, "2026-08-05");
  const ranges = await archiveFileTimestampRanges(files);
  assert.deepEqual(ranges.map((r) => r.file), ["2026-08-05-00.jsonl", "2026-08-05-01.jsonl"]);
  assert.deepEqual(ranges[0], { file: "2026-08-05-00.jsonl", rows: 3, minT: 100, maxT: 200 });
  assert.deepEqual(ranges[1], { file: "2026-08-05-01.jsonl", rows: 2, minT: 9000, maxT: 9500 });
  fs.rmSync(base, { recursive: true, force: true });
});

test("archiveFileTimestampRanges: a file NAMED for one hour but whose rows carry a DIFFERENT hour's t is exactly the mismatch this exists to surface", async () => {
  const base = tmp();
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  // Named 2026-08-06-00 (the day the live anomaly's archiveDayFiles listing
  // found zero files for) but its rows' own t values fall on 2026-08-05.
  const mismatchedRowT = Math.floor(Date.parse("2026-08-05T23:50:00Z") / 1000);
  fs.writeFileSync(path.join(dir, "2026-08-06-00.jsonl"), JSON.stringify({ t: mismatchedRowT, i: "m3" }) + "\n");
  const ranges = await archiveFileTimestampRanges(archiveDayFiles(dir, "2026-08-06"));
  assert.equal(ranges.length, 1);
  const namedHourStartSec = Math.floor(Date.parse("2026-08-06T00:00:00Z") / 1000);
  assert.ok(ranges[0].maxT! < namedHourStartSec,
    "row t predates the hour the filename claims — this is the mismatch item #37 needs made visible");
  fs.rmSync(base, { recursive: true, force: true });
});

test("archiveFileTimestampRanges: empty file reports zero rows and null min/max rather than throwing", async () => {
  const base = tmp();
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-08-05-02.jsonl"), "");
  const ranges = await archiveFileTimestampRanges([path.join(dir, "2026-08-05-02.jsonl")]);
  assert.deepEqual(ranges, [{ file: "2026-08-05-02.jsonl", rows: 0, minT: null, maxT: null }]);
  fs.rmSync(base, { recursive: true, force: true });
});

test("archiveFileTimestampRanges: rows with a missing/non-numeric t are excluded from the count and range, not crashing or coercing to 0", async () => {
  const base = tmp();
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const lines = [
    JSON.stringify({ i: "no-t" }),
    JSON.stringify({ t: "not-a-number", i: "bad-t" }),
    JSON.stringify({ t: 500, i: "good" }),
    "not even json",
  ];
  fs.writeFileSync(path.join(dir, "2026-08-05-03.jsonl"), lines.join("\n") + "\n");
  const ranges = await archiveFileTimestampRanges([path.join(dir, "2026-08-05-03.jsonl")]);
  assert.deepEqual(ranges, [{ file: "2026-08-05-03.jsonl", rows: 1, minT: 500, maxT: 500 }]);
  fs.rmSync(base, { recursive: true, force: true });
});

test("archiveFileTimestampRanges: reads gzipped files transparently, same as readArchiveDay", async () => {
  const base = tmp();
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-08-05-04.jsonl.gz"),
    zlib.gzipSync(Buffer.from(JSON.stringify({ t: 42, i: "gz1" }) + "\n")));
  const ranges = await archiveFileTimestampRanges([path.join(dir, "2026-08-05-04.jsonl.gz")]);
  assert.deepEqual(ranges, [{ file: "2026-08-05-04.jsonl.gz", rows: 1, minT: 42, maxT: 42 }]);
  fs.rmSync(base, { recursive: true, force: true });
});

// readArchiveDayEvenSample (2026-08-12): fixes the live symptom that blocked
// the GNSS-integrity Phase 4 gate-2 read (research/open_questions.md, the
// 2026-08-11 Bilawal-scan finding #1) — a bbox-scoped query over a busy
// hour-file stream must not have its whole row budget consumed by the
// FIRST hour file it opens, or any region whose traffic concentrates in a
// later UTC hour becomes invisible regardless of whether the signal exists.
test("readArchiveDayEvenSample: spreads the row budget evenly across hour files instead of exhausting the first one", async () => {
  const base = tmp();
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  const hourLines = (h: number) =>
    Array.from({ length: 20 }, (_, i) => JSON.stringify({ i: `h${h}-${i}` })).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-07-05-00.jsonl"), hourLines(0));
  fs.writeFileSync(path.join(dir, "2026-07-05-01.jsonl"), hourLines(1));
  fs.writeFileSync(path.join(dir, "2026-07-05-02.jsonl"), hourLines(2));
  const r = await readArchiveDayEvenSample("aircraft", "2026-07-05", base, 9);
  assert.ok(r);
  assert.equal(r!.rows.length, 9, "ceil(9/3 files) = 3 rows per file * 3 files");
  assert.equal(r!.truncated, true, "each file had more rows than its even share");
  const hoursRepresented = new Set(r!.rows.map((x: any) => x.i.split("-")[0]));
  assert.deepEqual([...hoursRepresented].sort(), ["h0", "h1", "h2"],
    "the old file-by-file readArchiveDay would have returned only h0 rows at this limit");
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDayEvenSample: falls back to readArchiveDay's own behavior on a single-file (day-granularity) stream", async () => {
  const base = tmp();
  const dir = path.join(base, "usaspending");
  fs.mkdirSync(dir, { recursive: true });
  const lines = Array.from({ length: 10 }, (_, i) => JSON.stringify({ aid: `a${i}` })).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-07-05.jsonl"), lines);
  const r = await readArchiveDayEvenSample("usaspending", "2026-07-05", base, 3);
  assert.ok(r);
  assert.equal(r!.rows.length, 3);
  assert.equal(r!.truncated, true);
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDayEvenSample: rowFilter is applied BEFORE a row counts against perFileLimit, so the budget is spent on matching rows", async () => {
  const base = tmp();
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  // One hour file: 8 non-matching rows followed by 2 matching rows. With
  // perFileLimit=2 counted BEFORE filtering, the 2 matching rows would
  // never be reached (budget exhausted on the non-matching prefix) — this
  // is exactly the live density problem readGnssIntegrityWindow hit.
  const lines = [
    ...Array.from({ length: 8 }, (_, i) => JSON.stringify({ i: `no${i}`, region: "elsewhere" })),
    ...Array.from({ length: 2 }, (_, i) => JSON.stringify({ i: `yes${i}`, region: "target" })),
  ].join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-07-05-00.jsonl"), lines);
  const r = await readArchiveDayEvenSample("aircraft", "2026-07-05", base, 2, (row: any) => row.region === "target");
  assert.ok(r);
  assert.equal(r!.rows.length, 2, "both matching rows counted, not starved by the non-matching prefix");
  assert.ok(r!.rows.every((row: any) => row.region === "target"));
  fs.rmSync(base, { recursive: true, force: true });
});

test("readArchiveDayEvenSample: unknown stream returns null; no files for the day returns empty non-null", async () => {
  const base = tmp();
  assert.equal(await readArchiveDayEvenSample("neverexistedstream", "2026-07-05", base), null);
  fs.mkdirSync(path.join(base, "usaspending"), { recursive: true });
  const r = await readArchiveDayEvenSample("usaspending", "2026-01-01", base);
  assert.ok(r);
  assert.deepEqual(r!.rows, []);
  assert.equal(r!.truncated, false);
  fs.rmSync(base, { recursive: true, force: true });
});

// ── GNSS integrity passthrough (2026-08-11): the archive must persist the
//    integrity/origin/provenance fields and round-trip them on REAL data,
//    with the null-is-not-zero guarantee. ──
const readAircraftLines = (base: string): any[] => {
  const dir = path.join(base, "aircraft");
  const out: any[] = [];
  for (const f of fs.readdirSync(dir)) {
    for (const ln of fs.readFileSync(path.join(dir, f), "utf8").split("\n").filter(Boolean)) {
      out.push(JSON.parse(ln));
    }
  }
  return out;
};

test("integrity fields round-trip through the archive (nulls stay null)", () => {
  const base = tmp();
  const t0 = Date.now();
  const p: any = {
    ...cruise("intg01"),
    nic: 8, nac_p: 10, nac_v: 2, sil: 3, sil_type: "perhour", rc: 186, gva: 2, sda: 2,
    nic_baro: 1, adsb_version: 2, pos_type: "adsb_icao", mlat_fields: null, tisb_fields: null,
    lkg_lat: 56.9, lkg_lon: 12.5, lkg_before: 3.2, seen_pos: 0.4, provider: "adsblol",
  };
  assert.equal(archiveAircraft([p], SITES, base, t0), 1);
  const [r] = readAircraftLines(base);
  assert.equal(r.ni, 8); assert.equal(r.np, 10); assert.equal(r.nv, 2); assert.equal(r.si, 3);
  assert.equal(r.st, "perhour"); assert.equal(r.rc, 186); assert.equal(r.gv, 2); assert.equal(r.sd, 2);
  assert.equal(r.nb, 1); assert.equal(r.pt, "adsb_icao");
  assert.equal(r.av, 2, "adsb_version (equipage control) is archived under `av`");
  assert.equal(r.kla, 56.9); assert.equal(r.klo, 12.5); assert.equal(r.kb, 3.2); assert.equal(r.sp, 0.4);
  assert.equal(r.pv, "adsblol");
  // null derivation arrays are omitted, not stored as [] or 0
  assert.ok(!("ml" in r) && !("tb" in r), "empty/null mlat/tisb arrays omitted");
  fs.rmSync(base, { recursive: true, force: true });
});

test("NULL IS NOT ZERO: a reported 0 persists as 0; an absent field is omitted, never 0", () => {
  const base = tmp();
  const t0 = Date.now();
  // reported zero-integrity (the signal-carrying case) MUST survive as 0
  const zero: any = { ...cruise("zero01"), nic: 0, nac_p: 0, sil: 0, pos_type: "mlat", provider: "adsblol" };
  // total silence — every integrity field null
  const silent: any = { ...cruise("silent1", 46, -29), nic: null, nac_p: null, sil: null,
    pos_type: null, provider: "adsblol" };
  assert.equal(archiveAircraft([zero], SITES, base, t0), 1);
  assert.equal(archiveAircraft([silent], SITES, base, t0 + 6 * 60_000), 1);
  const rows = readAircraftLines(base);
  const rz = rows.find((r) => r.i === "zero01");
  const rs = rows.find((r) => r.i === "silent1");
  assert.strictEqual(rz.ni, 0, "reported nic=0 must be stored as 0");
  assert.strictEqual(rz.np, 0); assert.strictEqual(rz.si, 0);
  // the crux: a null field must be ABSENT from the JSON, never serialized as 0
  for (const k of ["ni", "np", "si", "pt"]) {
    assert.ok(!(k in rs), `null field ${k} must be omitted, not written`);
  }
  // and prove no stringified line ever turned a null into a literal 0 key
  const dir = path.join(base, "aircraft");
  for (const f of fs.readdirSync(dir)) {
    const raw = fs.readFileSync(path.join(dir, f), "utf8");
    for (const ln of raw.split("\n").filter(Boolean)) {
      const o = JSON.parse(ln);
      // any integrity key present must reflect a real value we set, never a fabricated 0
      if (o.i === "silent1") assert.ok(o.ni === undefined && o.np === undefined && o.si === undefined);
    }
  }
  fs.rmSync(base, { recursive: true, force: true });
});

test("provenance filter: an adsb.lol-only subset is separable by the provider field", () => {
  const base = tmp();
  const t0 = Date.now();
  archiveAircraft([{ ...cruise("lol1"), provider: "adsblol" } as any], SITES, base, t0);
  archiveAircraft([{ ...cruise("live1", 46, -29), provider: "airplaneslive" } as any], SITES, base, t0 + 6 * 60_000);
  archiveAircraft([{ ...cruise("fi1", 47, -28), provider: "adsbfi" } as any], SITES, base, t0 + 12 * 60_000);
  const rows = readAircraftLines(base);
  const lolOnly = rows.filter((r) => r.pv === "adsblol");
  assert.equal(lolOnly.length, 1);
  assert.equal(lolOnly[0].i, "lol1");
  assert.ok(rows.every((r) => r.pv), "every archived row carries a provider tag");
  assert.ok(!lolOnly.some((r) => r.pv === "airplaneslive" || r.pv === "adsbfi"),
    "the adsb.lol subset contains no non-commercial-provider rows");
  fs.rmSync(base, { recursive: true, force: true });
});

test("originOfPosType decodes broadcast vs ground-derived (the one decode table)", () => {
  assert.equal(originOfPosType("adsb_icao"), "broadcast");
  assert.equal(originOfPosType("adsr_icao"), "broadcast");
  assert.equal(originOfPosType("mlat"), "ground");
  assert.equal(originOfPosType("tisb_trackfile"), "ground");
  assert.equal(originOfPosType("mode_s"), "mode_s");
  assert.equal(originOfPosType(null), "unknown");
  assert.equal(originOfPosType("something_new"), "unknown");
});
