// Trains layer: pure mapping tests with real sample payloads (captured
// 2026-07-04 from both live APIs), archive round-trip, and wiring pins.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { mapDigitraffic, mapEntur, TRAIN_SOURCES } from "./trainsFeed";
import { archiveTrains, recentTrack, trainIntervalMs } from "./datacoreArchive";

const here = path.dirname(fileURLToPath(import.meta.url));

test("mapDigitraffic: real payload shape -> unified trains (no bearing in source)", () => {
  const raw = [
    { accuracy: 4, location: { type: "Point", coordinates: [27.312089, 63.763156] }, speed: 139, trainNumber: 62, departureDate: "2026-07-04", timestamp: "2026-07-04T01:09:06.000Z" },
    { accuracy: 2, location: { type: "Point", coordinates: [24.94, 60.17] }, speed: 0, trainNumber: 104, departureDate: "2026-07-04", timestamp: "2026-07-04T01:09:11.000Z" },
    { location: null, trainNumber: 9 },                       // malformed -> dropped
  ];
  const out = mapDigitraffic(raw);
  assert.equal(out.length, 2);
  assert.equal(out[0].id, "FI-62-2026-07-04");
  assert.equal(out[0].country, "FI");
  assert.equal(out[0].lat, 63.763156);
  assert.equal(out[0].lon, 27.312089);
  assert.equal(out[0].speed_kmh, 139);
  assert.equal(out[0].bearing, null, "Digitraffic publishes no heading — must stay null so the icon renders upright");
  assert.ok(out[0].ts! > 1_700_000_000);
});

test("mapEntur: real payload shape -> unified trains (m/s -> km/h, bearing kept)", () => {
  const raw = { data: { vehicles: [
    { lastUpdated: "2026-07-04T01:08:13.156036Z", line: { lineRef: "FLT:Line:FLY1" }, location: { latitude: 59.74097, longitude: 10.2016 }, speed: 30, bearing: 0.78, vehicleId: "71-12" },
    { lastUpdated: null, line: null, location: { latitude: 60.0, longitude: 10.5 }, speed: null, bearing: null, vehicleId: "99-1" },
    { vehicleId: null, location: { latitude: 1, longitude: 1 } },   // malformed -> dropped
  ] } };
  const out = mapEntur(raw);
  assert.equal(out.length, 2);
  assert.equal(out[0].id, "NO-71-12");
  assert.equal(out[0].country, "NO");
  assert.equal(out[0].speed_kmh, 108, "30 m/s = 108 km/h");
  assert.equal(out[0].bearing, 0.78);
  assert.equal(out[0].label, "FLY1", "line ref shortened for display");
  assert.equal(out[1].speed_kmh, null);
});

test("both mappers tolerate garbage", () => {
  assert.deepEqual(mapDigitraffic(null), []);
  assert.deepEqual(mapDigitraffic({ nope: 1 }), []);
  assert.deepEqual(mapEntur(null), []);
  assert.deepEqual(mapEntur({ data: {} }), []);
});

test("trains archive: cadence gate + track round-trip via the shared machinery", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-trains-arch-"));
  const t0 = Date.UTC(2026, 6, 4, 12, 0, 0);
  const p = { id: "FI-62-2026-07-04", country: "FI", lat: 63.7632, lon: 27.3121, speed_kmh: 139, bearing: null, label: "Train 62" };
  assert.equal(archiveTrains([p], base, t0), 1, "first sample writes");
  assert.equal(archiveTrains([p], base, t0 + 30_000), 0, "inside the cadence window -> thinned");
  assert.equal(archiveTrains([{ ...p, lat: 63.8 }], base, t0 + trainIntervalMs() + 1000), 1, "next window writes");
  const track = recentTrack("trains", "FI-62-2026-07-04", base, t0 + trainIntervalMs() + 2000);
  assert.equal(track.length, 2);
  assert.equal(track[0].la, 63.7632);
  assert.equal(track[1].la, 63.8);
});

test("wiring: route, layer registry with honest coverage, panel source labels", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("/api/data/trains"), "route missing");
  assert.ok(routes.includes("rata.digitraffic.fi"), "Digitraffic upstream missing");
  assert.ok(routes.includes("api.entur.io"), "Entur upstream missing");
  assert.ok(routes.includes("ET-Client-Name"), "Entur requires the client-name header");
  assert.ok(routes.includes("archiveTrains"), "trains not feeding the position archive");
  const layers = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  const tr = layers.layers.find((l: any) => l.id === "trains");
  assert.ok(tr, "layers.json entry missing");
  assert.equal(tr.kind, "raw");
  assert.ok(tr.source.includes("CC BY 4.0") && tr.source.includes("NLOD"), "source licenses missing");
  assert.ok(tr.description.includes("proprietary"), "US-freight-proprietary fact must be stated where users read it");
  assert.equal(TRAIN_SOURCES.length, 2);
});
