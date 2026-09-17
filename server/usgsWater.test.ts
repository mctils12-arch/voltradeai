import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  parseGaugeObs, archiveGaugeObs, fetchGauges, USGS_GAUGES,
  backfillGaugesFromArchive, refreshGaugeCache, latestGauges, _resetGaugeCacheForTests,
} from "./usgsWater";

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

// Cold-cache-no-disk-backfill thread (research/open_questions.md — usgsWater.ts
// was one of the remaining "new reader needed" modules from the 2026-09-15
// module audit). The archive is revision-append (a provisional value later
// revised to approved lands as a NEW row, same identity), so backfill must
// reconstruct "latest known reading per (site, param)" rather than replay
// every historical revision as if it were current.

test("backfillGaugesFromArchive: keeps only the latest reading per (site, param) identity across the lookback window", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vtusgs-backfill-"));
  const dir = path.join(root, "usgswater");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-10T12:00:00Z");
  const day0 = new Date(now).toISOString().slice(0, 10);
  const day1 = new Date(now - 86400_000).toISOString().slice(0, 10);
  const older = { site: "07010000", name: "St. Louis", param: "00065", d: "2026-09-09T00:15:00.000-05:00", v: 15.10, q: "P", lat: 38.6, lon: -90.1, rt: "2026-09-09" };
  const newer = { site: "07010000", name: "St. Louis", param: "00065", d: "2026-09-10T00:15:00.000-05:00", v: 15.30, q: "A", lat: 38.6, lon: -90.1, rt: "2026-09-10" };
  const otherSite = { site: "07032000", name: "Memphis", param: "00060", d: "2026-09-10T00:15:00.000-06:00", v: 512000, q: "P", lat: 35.1, lon: -90.0, rt: "2026-09-10" };
  fs.writeFileSync(path.join(dir, `${day1}.jsonl`), JSON.stringify(older) + "\n");
  fs.writeFileSync(path.join(dir, `${day0}.jsonl`), [newer, otherSite].map((o) => JSON.stringify(o)).join("\n") + "\n");
  const out = backfillGaugesFromArchive(root, now, 3);
  assert.equal(out.length, 2, "two distinct (site, param) identities");
  const stl = out.find((o) => o.site === "07010000")!;
  assert.equal(stl.v, 15.30, "the newer-d reading wins, not the older provisional one");
  assert.equal(stl.q, "A");
  assert.ok(out.find((o) => o.site === "07032000"), "the second identity is preserved");
});

test("backfillGaugesFromArchive: respects the days window and an empty lookback reconstructs nothing", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vtusgs-backfill-window-"));
  const dir = path.join(root, "usgswater");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-10T12:00:00Z");
  const tooOld = new Date(now - 10 * 86400_000).toISOString().slice(0, 10);
  const row = { site: "07010000", name: "St. Louis", param: "00065", d: "2026-08-31T00:15:00.000-05:00", v: 14.0, q: "A", lat: 38.6, lon: -90.1, rt: "2026-08-31" };
  fs.writeFileSync(path.join(dir, `${tooOld}.jsonl`), JSON.stringify(row) + "\n");
  assert.deepEqual(backfillGaugesFromArchive(root, now, 3), [], "a day 10 days back is outside the default 3-day window");
  assert.deepEqual(backfillGaugesFromArchive(root, now, 30), [row], "widening the window picks it up");
});

test("refreshGaugeCache: cold cache backfills from the on-disk archive when the live poll throws", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vtusgs-coldcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetGaugeCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "usgswater");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    const row = { site: "07010000", name: "St. Louis", param: "00065", d: "2026-09-10T00:15:00.000-05:00", v: 15.30, q: "A", lat: 38.6, lon: -90.1, rt: "2026-09-10" };
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(row) + "\n");
    assert.equal(latestGauges(), null, "cache must still be cold going into this cycle");
    const throwingFetch = (async () => { throw new Error("waterservices.usgs.gov unreachable"); }) as any;
    await refreshGaugeCache(throwingFetch);
    const cached = latestGauges();
    assert.ok(cached, "cache must be populated, not left null, despite the live poll throwing");
    assert.equal(cached!.gauges.length, 1);
    assert.equal(cached!.gauges[0].site, "07010000", "backfilled from the archived reading, not fabricated");
  } finally {
    _resetGaugeCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshGaugeCache: a subsequently-warm cache is untouched by a second transport failure", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vtusgs-warmcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetGaugeCacheForTests();
  try {
    const goodFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify(WATERML) });
    await refreshGaugeCache(goodFetch as any, Date.parse("2026-09-10T12:00:00Z"));
    const warm = latestGauges();
    assert.equal(warm!.gauges.length, 2, "warm the cache first");

    const throwingFetch = (async () => { throw new Error("503"); }) as any;
    await refreshGaugeCache(throwingFetch);
    assert.equal(latestGauges(), warm, "a transport failure must not clobber an already-warm cache with a stale disk read");
  } finally {
    _resetGaugeCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshGaugeCache: a genuinely empty but successful poll on a cold cache is still trusted, not overridden by the archive", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vtusgs-coldcache-empty-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetGaugeCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "usgswater");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    const row = { site: "07010000", name: "St. Louis", param: "00065", d: "2026-09-10T00:15:00.000-05:00", v: 15.30, q: "A", lat: 38.6, lon: -90.1, rt: "2026-09-10" };
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(row) + "\n");
    const emptyFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ value: { timeSeries: [] } }) });
    await refreshGaugeCache(emptyFetch as any);
    const cached = latestGauges();
    assert.ok(cached, "cache must be populated on a successful poll, even an empty one");
    assert.equal(cached!.gauges.length, 0, "an empty-but-successful poll is trusted as-is, never silently swapped for a stale archive read");
  } finally {
    _resetGaugeCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});
