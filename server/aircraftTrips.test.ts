import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { splitTrips, fullTrackAsync, tripsCoverage, classifyTrip, TRIP_GAP_SEC, type ArchivedFix } from "./aircraftTrips";

const fix = (t: number, extra: Partial<ArchivedFix> = {}): ArchivedFix =>
  ({ t, la: 40 + t / 1e6, lo: -95 + t / 1e6, al: 9000, c: "SWA762", ...extra });

// ── splitTrips ──────────────────────────────────────────────────────────────

test("one continuous airborne track is one trip, newest-first ordering fields intact", () => {
  const fixes = [0, 60, 120, 180, 240].map((s) => fix(1000 + s));
  const trips = splitTrips(fixes);
  assert.equal(trips.length, 1);
  assert.equal(trips[0].fixes, 5);
  assert.equal(trips[0].duration_s, 240);
  assert.deepEqual(trips[0].callsigns, ["SWA762"]);
  assert.equal(trips[0].max_alt_m, 9000);
});

test("a gap > 45 min splits into two trips, newest first", () => {
  const a = [0, 60, 120].map((s) => fix(1000 + s));
  const b = [0, 60, 120].map((s) => fix(1000 + 120 + TRIP_GAP_SEC + 60 + s, { c: "SWA100" }));
  const trips = splitTrips([...a, ...b]);
  assert.equal(trips.length, 2);
  assert.ok(trips[0].start_t > trips[1].start_t, "newest trip first");
  assert.deepEqual(trips[0].callsigns, ["SWA100"]);
});

test("a >=15min ground dwell inside an airborne track splits; the dwell's last fix leads the next trip in", () => {
  const leg1 = [0, 60, 120].map((s) => fix(1000 + s));
  const dwell = [180, 600, 1080].map((s) => fix(1000 + s, { al: null, g: true }));
  const leg2 = [1140, 1200, 1260].map((s) => fix(1000 + s));
  const trips = splitTrips([...leg1, ...dwell, ...leg2]);
  assert.equal(trips.length, 2);
  const older = trips[1], newer = trips[0];
  assert.equal(older.end_t, 1000 + 1080, "trip 1 ends at the dwell");
  assert.equal(newer.start_t, 1000 + 1080, "dwell's last fix is the takeoff lead-in");
});

test("a parked transponder alone (never airborne) is one ground track, not many trips", () => {
  const parked = [0, 1200, 2400, 3600].map((s) => fix(1000 + s, { al: null, g: true }));
  const trips = splitTrips(parked);
  assert.equal(trips.length, 1);
  assert.equal(trips[0].max_alt_m, null, "no altitude ever seen -> honest null");
});

test("segments below the minimum fix count are dropped as noise", () => {
  const lone = [fix(1000), fix(1000 + TRIP_GAP_SEC + 100)];
  assert.equal(splitTrips(lone).length, 0);
});

test("empty input -> empty output", () => {
  assert.deepEqual(splitTrips([]), []);
});

// ── fullTrackAsync (real files in a temp archive) ───────────────────────────

test("fullTrackAsync reads ALL retained days (not 48h), keeps c/al/g, respects time bounds", async () => {
  const base = mkdtempSync(path.join(tmpdir(), "trips-"));
  const dir = path.join(base, "aircraft");
  mkdirSync(dir, { recursive: true });
  const mk = (day: string, hour: string, rows: object[]) =>
    writeFileSync(path.join(dir, `${day}-${hour}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const t3d = Math.floor(Date.parse("2026-08-05T10:00:00Z") / 1000);
  const t1d = Math.floor(Date.parse("2026-08-07T10:00:00Z") / 1000);
  mk("2026-08-05", "10", [
    { t: t3d, i: "abe872", c: "SWA762", la: 37.3, lo: -102.7, al: 11000 },
    { t: t3d + 60, i: "ffffff", c: "OTHER1", la: 1, lo: 1, al: 5000 }, // other hex filtered out
  ]);
  mk("2026-08-07", "10", [
    { t: t1d, i: "abe872", c: "SWA100", la: 38.0, lo: -100.0, g: true },
  ]);
  const all = await fullTrackAsync("aircraft", "abe872", base);
  assert.equal(all.length, 2, "3-day-old fix included — recentTrack's 48h cap does not apply");
  assert.equal(all[0].c, "SWA762");
  assert.equal(all[1].g, true);
  assert.equal(all[1].al, null);
  const bounded = await fullTrackAsync("aircraft", "abe872", base, { fromSec: t1d - 3600 });
  assert.equal(bounded.length, 1, "time bound excludes the older day");
  assert.equal(bounded[0].c, "SWA100");
  rmSync(base, { recursive: true, force: true });
});

test("tripsCoverage states the raw-retention bound and the thinning caveat", () => {
  const c = tripsCoverage();
  assert.equal(c.raw_days, 30);
  assert.match(c.note, /retained 30 days/);
  assert.match(c.note, /lower bounds/);
});

test("fullTrackAsync collapses same-second poller+backfill duplicates, keeping the altitude-bearing fix", async () => {
  const base = mkdtempSync(path.join(tmpdir(), "dedupe-"));
  const dir = path.join(base, "aircraft");
  mkdirSync(dir, { recursive: true });
  const t0 = Math.floor(Date.parse("2026-08-08T10:00:00Z") / 1000);
  writeFileSync(path.join(dir, "2026-08-08-10.jsonl"), [
    { t: t0, i: "ab8c8e", la: 41.92, lo: -72.70 },              // poller fix, no altitude
    { t: t0, i: "ab8c8e", la: 41.9221, lo: -72.7071, al: 594 }, // backfill same second, altitude
    { t: t0 + 30, i: "ab8c8e", la: 41.93, lo: -72.69, al: 800 },
  ].map((r) => JSON.stringify(r)).join("\n") + "\n");
  const pts = await fullTrackAsync("aircraft", "ab8c8e", base);
  assert.equal(pts.length, 2, "same-second duplicate collapsed");
  assert.equal(pts[0].al, 594, "the altitude-bearing fix wins");
  rmSync(base, { recursive: true, force: true });
});

// ── QC-1 trip quality (human directive 2026-08-11) ──────────────────────────

test("classifyTrip: a real flight — ground, climb, cruise, descend, ground — is complete", () => {
  const s = [
    fix(0, { al: null, g: true }), fix(60, { al: 300 }), fix(120, { al: 5000 }),
    fix(180, { al: 9000 }), fix(240, { al: 2000 }), fix(300, { al: null, g: true }),
  ];
  assert.equal(classifyTrip(s).quality, "complete");
});

test("classifyTrip: taxi-around with ADS-B on is NOT a flight", () => {
  const s = [fix(0, { al: null, g: true }), fix(60, { al: null, g: true }), fix(120, { al: null, g: true })];
  const q = classifyTrip(s);
  assert.equal(q.quality, "taxi_only");
  // low hops under the airborne threshold also stay taxi_only
  const hop = [fix(0, { al: 200 }), fix(60, { al: 350 }), fix(120, { al: 210 })];
  assert.equal(classifyTrip(hop).quality, "taxi_only");
});

test("classifyTrip: first-seen-at-cruise is partial_start (coverage began mid-flight)", () => {
  const s = [fix(0, { al: 10000 }), fix(60, { al: 9000 }), fix(120, { al: 400 }), fix(180, { al: null, g: true })];
  assert.equal(classifyTrip(s).quality, "partial_start");
});

test("classifyTrip: last-seen-at-altitude is signal_lost_airborne — logged, never asserted as a crash", () => {
  const s = [fix(0, { al: null, g: true }), fix(60, { al: 4000 }), fix(120, { al: 11000 })];
  const q = classifyTrip(s);
  assert.equal(q.quality, "signal_lost_airborne");
  assert.match(q.basis, /never asserted/);
});

test("classifyTrip: airborne at both ends is a pass through coverage (partial_both)", () => {
  const s = [fix(0, { al: 11000 }), fix(60, { al: 11500 }), fix(120, { al: 10800 })];
  assert.equal(classifyTrip(s).quality, "partial_both");
});

test("splitTrips carries quality + basis on every trip", () => {
  const fixes = [
    fix(0, { al: null, g: true }), fix(60, { al: 5000 }), fix(120, { al: 9000 }), fix(180, { al: null, g: true }),
  ];
  const trips = splitTrips(fixes);
  assert.equal(trips.length, 1);
  assert.equal(trips[0].quality, "complete");
  assert.ok(trips[0].quality_basis.length > 0);
});
