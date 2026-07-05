// NWS alerts battery (BUILD ORDER 2 #3, 2026-07-05): CAP parsing, geometry
// honesty (zone-only counted, never drawn or guessed), ring decimation,
// id dedup, gz lifecycle. Fixture mirrors the live api.weather.gov shape
// probed 2026-07-05.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  parseAlerts, simplifyRings, archiveAlerts, gzipOldAlertDays, type AlertRec,
} from "./nwsAlerts";

const square = (n = 5): any => ({
  type: "Polygon",
  coordinates: [Array.from({ length: n }, (_, i) => {
    const t = (i / (n - 1)) * 2 * Math.PI;
    return [-97 + 0.5 * Math.cos(t), 35 + 0.5 * Math.sin(t)];
  })],
});

const feature = (id: string, over: any = {}, geometry: any = square()) => ({
  geometry,
  properties: {
    id, event: "Tornado Warning", severity: "Extreme", urgency: "Immediate",
    certainty: "Observed", sent: "2026-07-05T12:00:00-05:00", onset: null,
    ends: "2026-07-05T13:00:00-05:00", areaDesc: "Payne, OK; Lincoln, OK", ...over,
  },
});

test("parseAlerts: full record + display split; zone-only counted never drawn", () => {
  const { recs, display, zoneOnly } = parseAlerts({
    features: [
      feature("urn:a1"),
      feature("urn:a2", { event: "Flood Watch", severity: "Moderate" }, null), // zone-coded
      { properties: { event: "no id" } },                                       // dropped
    ],
  }, "2026-07-05");
  assert.equal(recs.length, 2);
  assert.equal(display.length, 1, "only polygon-carrying alerts render");
  assert.equal(zoneOnly, 1);
  const a1 = recs[0];
  assert.equal(a1.id, "urn:a1");
  assert.equal(a1.severity, "Extreme");
  assert.equal(a1.area, "Payne, OK; Lincoln, OK");
  assert.ok(a1.lat! > 34.4 && a1.lat! < 35.6 && a1.lon! > -97.6 && a1.lon! < -96.4, "centroid inside the polygon");
  assert.ok(a1.bbox && a1.bbox[0] < a1.bbox[2] && a1.bbox[1] < a1.bbox[3]);
  const a2 = recs[1];
  assert.equal(a2.lat, null, "zone-coded alert archives with null geo, never guessed");
  assert.equal(a2.bbox, null);
});

test("simplifyRings: decimates long rings, stays closed, handles MultiPolygon", () => {
  const big = { type: "Polygon", coordinates: [Array.from({ length: 500 }, (_, i) => {
    const t = (i / 499) * 2 * Math.PI;
    return [Math.cos(t), Math.sin(t)];
  })] };
  const rings = simplifyRings(big);
  assert.equal(rings.length, 1);
  assert.ok(rings[0].length <= 66, `expected <=66 points, got ${rings[0].length}`);
  const r = rings[0];
  assert.deepEqual(r[0], r[r.length - 1], "ring stays closed after decimation");
  const multi = { type: "MultiPolygon", coordinates: [square().coordinates, square().coordinates] };
  assert.equal(simplifyRings(multi).length, 2);
  assert.deepEqual(simplifyRings(null), []);
  assert.deepEqual(simplifyRings({ type: "Point", coordinates: [0, 0] }), []);
});

test("archive: dedup by id across polls, gz after 2 days", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nws-"));
  const now = Date.parse("2026-07-05T12:00:00Z");
  const { recs } = parseAlerts({ features: [feature("urn:a1"), feature("urn:a2", {}, null)] }, "2026-07-05");
  assert.equal(archiveAlerts(recs, base, now), 2);
  assert.equal(archiveAlerts(recs, base, now), 0, "same ids never re-archive");
  const { recs: recs2 } = parseAlerts({ features: [feature("urn:a3")] }, "2026-07-05");
  assert.equal(archiveAlerts(recs2, base, now), 1);
  const day = path.join(base, "nwsalerts", "2026-07-05.jsonl");
  const lines = fs.readFileSync(day, "utf8").trim().split("\n").map((l) => JSON.parse(l) as AlertRec);
  assert.deepEqual(lines.map((r) => r.id), ["urn:a1", "urn:a2", "urn:a3"]);
  // gz lifecycle
  assert.equal(gzipOldAlertDays(base, now + 3 * 86400_000), 1);
  assert.ok(!fs.existsSync(day) && fs.existsSync(`${day}.gz`));
  const unz = zlib.gunzipSync(fs.readFileSync(`${day}.gz`)).toString("utf8");
  assert.ok(unz.includes("urn:a1"));
});
