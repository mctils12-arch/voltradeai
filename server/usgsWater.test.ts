import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { parseGaugeObs, archiveGaugeObs, fetchGauges, USGS_GAUGES } from "./usgsWater";

const here = path.dirname(fileURLToPath(import.meta.url));

// Real WaterML-JSON envelope shape (verified live 2026-07-05: St. Louis
// 00065 = 15.13 ft; Memphis publishes 00060 discharge only).
const WATERML = {
  value: { timeSeries: [
    {
      sourceInfo: {
        siteName: "Mississippi River at St. Louis, MO",
        siteCode: [{ value: "07010000" }],
        geoLocation: { geogLocation: { latitude: 38.62889, longitude: -90.17972 } },
      },
      variable: { variableCode: [{ value: "00065" }] },
      values: [{ value: [
        { value: "15.20", qualifiers: ["P"], dateTime: "2026-07-05T00:00:00.000-05:00" },
        { value: "15.13", qualifiers: ["P"], dateTime: "2026-07-05T00:15:00.000-05:00" },
      ] }],
    },
    {
      sourceInfo: {
        siteName: "Mississippi River at Memphis, TN",
        siteCode: [{ value: "07032000" }],
        geoLocation: { geogLocation: { latitude: 35.12278, longitude: -90.07750 } },
      },
      variable: { variableCode: [{ value: "00060" }] },
      values: [{ value: [
        { value: "512000", qualifiers: ["P"], dateTime: "2026-07-05T00:15:00.000-06:00" },
      ] }],
    },
    {
      sourceInfo: { siteName: "Sentinel test", siteCode: [{ value: "99999999" }] },
      variable: { variableCode: [{ value: "00065" }] },
      values: [{ value: [{ value: "-999999", qualifiers: ["P"], dateTime: "2026-07-05T00:15:00.000-05:00" }] }],
    },
  ] },
};

test("parseGaugeObs: latest value per series, both params, geo carried, sentinels dropped", () => {
  const obs = parseGaugeObs(WATERML, "2026-07-05");
  assert.equal(obs.length, 2, "-999999 sentinel row dropped");
  assert.equal(obs[0].site, "07010000");
  assert.equal(obs[0].param, "00065");
  assert.equal(obs[0].v, 15.13, "LATEST value taken, not the first");
  assert.equal(obs[0].q, "P");
  assert.equal(obs[0].lat, 38.62889);
  assert.equal(obs[1].param, "00060", "discharge-only gauges stored under 00060");
});

test("gauge table sane: 14 verified sites, unique, dead Metropolis gauge excluded", () => {
  const sites = USGS_GAUGES.map((g) => g.site);
  assert.equal(sites.length, 14);
  assert.equal(new Set(sites).size, 14);
  assert.ok(!sites.includes("03611500"), "dead gauge must stay excluded");
  assert.ok(sites.includes("07032000") && sites.includes("07289000"), "Memphis + Vicksburg (drought headline gauges) present");
});

test("fetchGauges requests both parameter codes for all sites in one call", async () => {
  let seen = "";
  const impl = async (url: string) => { seen = url; return { ok: true, status: 200, text: async () => JSON.stringify(WATERML) }; };
  await fetchGauges(impl as any, Date.parse("2026-07-05T12:00:00Z"));
  assert.ok(seen.includes("parameterCd=00065,00060"), "must request stage AND discharge");
  assert.ok(seen.includes("07010000") && seen.includes("03612600"), "all sites in one request");
});

test("archive: revision (provisional -> approved, or value change) appends as a new vintage row", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtusgs-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const obs = parseGaugeObs(WATERML, "2026-07-05");
  assert.equal(archiveGaugeObs(obs, dir, t0), 2);
  assert.equal(archiveGaugeObs(obs, dir, t0), 0, "identical readings never archive twice");
  const revised = [{ ...obs[0], v: 15.14, q: "A", rt: "2026-07-06" }];
  assert.equal(archiveGaugeObs(revised as any, dir, t0 + 86400_000), 1, "revision appends");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("routes.ts boots the USGS poll and registers /api/data/rivergauges; manifest present", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootUsgsPoll()"), "USGS poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/rivergauges"'), "route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "usgswater.json"), "utf8"));
  assert.equal(manifest.stream, "usgswater");
  assert.deepEqual(manifest.geo_fields, ["lat", "lon"], "gauges are map-plottable");
  assert.ok(String(manifest.confidence_model).includes("gate-2"), "low-water signal stays gated");
});
