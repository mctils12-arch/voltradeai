import { test } from "node:test";
import assert from "node:assert/strict";
import { mapDigitrafficAis, freshAisFixes, DIGITRAFFIC_AIS_ATTRIBUTION } from "./aisFeed";

// Shapes below are copied from a LIVE probe of the two endpoints on
// 2026-08-12 (849 locations / 794 vessel records), not invented.
const LOCATIONS = {
  type: "FeatureCollection",
  features: [
    { mmsi: 230998680, type: "Feature",
      geometry: { type: "Point", coordinates: [22.668848, 60.042517] },
      properties: { mmsi: 230998680, sog: 0.0, cog: 360.0, navStat: 5, rot: -128,
                    posAcc: true, raim: true, heading: 511, timestamp: 17,
                    timestampExternal: 1786530737232 } },
    { mmsi: 273359290, type: "Feature",
      geometry: { type: "Point", coordinates: [25.1, 60.15] },
      properties: { mmsi: 273359290, sog: 12.4, cog: 88.2,
                    timestampExternal: 1786530700000 } },
  ],
};
const VESSELS = [
  { mmsi: 273359290, name: "ALLEK     ", destination: "SANKT PETERBURG",
    imo: 9400253, callSign: "UBQJ9 ", shipType: 70, draught: 23 },
];

test("mapDigitrafficAis: joins positions with metadata and normalizes shape", () => {
  const out = mapDigitrafficAis(LOCATIONS, VESSELS, 1786530800);
  assert.equal(out.length, 2);
  const allek = out.find((f) => f.mmsi === "273359290")!;
  assert.equal(allek.lat, 60.15);
  assert.equal(allek.lon, 25.1);
  assert.equal(allek.sog, 12.4);
  assert.equal(allek.cog, 88.2);
  assert.equal(allek.name, "ALLEK", "trailing pad stripped");
  assert.equal(allek.destination, "SANKT PETERBURG");
  assert.equal(allek.shiptype, 70);
  assert.equal(allek.at, 1786530700, "epoch ms -> seconds");
});

test("mapDigitrafficAis: AIS not-available sentinels become null, never fake values", () => {
  const out = mapDigitrafficAis(LOCATIONS, VESSELS, 1786530800);
  const moored = out.find((f) => f.mmsi === "230998680")!;
  assert.equal(moored.cog, null, "COG 360 = not available");
  assert.equal(moored.sog, 0, "a real 0.0 kt (moored) is NOT a sentinel and must survive");
  assert.equal(moored.name, null, "no metadata row — honest null, not a guess");
  // SOG 102.3 is the not-available sentinel
  const na = mapDigitrafficAis(
    { features: [{ mmsi: 1, geometry: { type: "Point", coordinates: [25, 60] }, properties: { sog: 102.3, cog: 12 } }] },
    [], 1000);
  assert.equal(na[0].sog, null);
});

test("mapDigitrafficAis: unusable rows are dropped, never placed at a guessed position", () => {
  const bad = {
    features: [
      { mmsi: 1, geometry: { type: "Point", coordinates: [] }, properties: {} },       // no coords
      { mmsi: 2, properties: {} },                                                      // no geometry
      { geometry: { type: "Point", coordinates: [25, 60] }, properties: {} },           // no mmsi
      { mmsi: 3, geometry: { type: "Point", coordinates: [999, 60] }, properties: {} }, // out of range
      { mmsi: 4, geometry: { type: "Point", coordinates: [25, 60] }, properties: {} },  // the one keeper
    ],
  };
  const out = mapDigitrafficAis(bad, [], 1000);
  assert.deepEqual(out.map((f) => f.mmsi), ["4"]);
});

test("mapDigitrafficAis: empty/garbage payloads yield [] instead of throwing", () => {
  assert.deepEqual(mapDigitrafficAis(null, null, 1), []);
  assert.deepEqual(mapDigitrafficAis({}, undefined, 1), []);
  assert.deepEqual(mapDigitrafficAis({ features: [] }, [], 1), []);
});

test("freshAisFixes: stale positions are dropped — a stale fix shown as live is the same lie as a dead feed claiming live", () => {
  const now = 10_000;
  const fixes = mapDigitrafficAis(
    { features: [
      { mmsi: 1, geometry: { type: "Point", coordinates: [25, 60] }, properties: { timestampExternal: (now - 100) * 1000 } },
      { mmsi: 2, geometry: { type: "Point", coordinates: [25, 60] }, properties: { timestampExternal: (now - 7200) * 1000 } },
    ] }, [], now);
  const fresh = freshAisFixes(fixes, now, 3600);
  assert.deepEqual(fresh.map((f) => f.mmsi), ["1"]);
});

test("attribution string is exactly what the CC 4.0 BY terms require", () => {
  assert.equal(DIGITRAFFIC_AIS_ATTRIBUTION, "Source: Fintraffic / digitraffic.fi, license CC 4.0 BY");
});
