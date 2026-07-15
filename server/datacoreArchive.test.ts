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
  recentTrack, recentTrackAsync, recentTrackCached, clearTrackCache,
  archiveStats, aircraftIntervalMs, vesselIntervalMs,
  nearAnySite, RAW_RETENTION_DAYS,
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
