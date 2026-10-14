// Coverage gap-fill (2026-10-14): adsb.lol saw 0 aircraft on the CLT surface
// while adsb.fi saw 9, so a plane taxiing out there never appeared. These
// tests pin the merge, the politeness bound, provenance and both wirings.
// Run: npx tsx --test server/coverageGapFill.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  shouldGapFill, mergeMissing, createGapFiller, gapFillEnabled, adsbFiDiscUrl, adsbFiHexUrl,
  GAPFILL_MAX_RADIUS_NM, GAPFILL_MIN_INTERVAL_MS,
} from "./coverageGapFill";
import { createFastLane } from "./aircraftFastLane";
import { GlobalSnapshot } from "./globalSnapshot";
import { subscribeFixes, type FixBatch } from "./aircraftFixBus";
import { UpstreamGovernor } from "./adsbGovernor";

const here = path.dirname(fileURLToPath(import.meta.url));
const CLT = { lat: 35.214, lon: -80.943 };
const disc = (r: number, n = 1) => ({ centers: Array.from({ length: n }, () => ({ ...CLT })), radiusNm: r });
const fiRow = (hex: string, ground = true) => ({
  hex, flight: "AAL1651 ", r: "N123AA", t: "A321", lat: 35.21, lon: -80.94,
  alt_baro: ground ? "ground" : 2500, gs: 12, track: 180, category: "A3",
});
const res = (body: unknown, status = 200) =>
  ({ ok: status >= 200 && status < 300, status, json: async () => body }) as unknown as Response;

test("shouldGapFill: one zoomed-in disc only, and never when adsb.fi already answered", () => {
  assert.equal(shouldGapFill(disc(4), ["adsb.lol (ADS-B, community)"]), true);
  assert.equal(shouldGapFill(disc(GAPFILL_MAX_RADIUS_NM), ["adsb.lol"]), true);
  assert.equal(shouldGapFill(disc(GAPFILL_MAX_RADIUS_NM + 1), ["adsb.lol"]), false, "wide view: no extra request");
  assert.equal(shouldGapFill(disc(30, 2), ["adsb.lol"]), false, "multi-disc tiling: no extra request");
  assert.equal(shouldGapFill(disc(4), ["adsb.fi (ADS-B, community)"]), false, "adsb.fi was the chain's answer already");
});

test("mergeMissing: primary rows always win; only missing hexes are added, once", () => {
  const primary = [{ icao24: "aaa111", src: "lol" }];
  const extra = [{ icao24: "AAA111", src: "fi" }, { icao24: "bbb222", src: "fi" }, { icao24: "bbb222", src: "fi2" }, { icao24: "", src: "nohex" }];
  const { merged, added } = mergeMissing(primary, extra);
  assert.equal(added, 1);
  assert.deepEqual(merged.map((a) => a.src), ["lol", "fi"]);
  assert.equal(mergeMissing(primary, []).merged, primary, "no-op returns the same array");
});

test("fillDisc: CLT surface traffic adsb.lol is blind to is merged with adsbfi provenance and published", async () => {
  const urls: string[] = [];
  let t = 1_000_000;
  const filler = createGapFiller({
    enabled: true, now: () => t,
    fetchImpl: (async (u: string) => { urls.push(u); return res({ now: t, aircraft: [fiRow("a00001"), fiRow("a00002", false), fiRow("adb111")] }); }) as unknown as typeof fetch,
  });
  const batches: FixBatch[] = [];
  const off = subscribeFixes((b) => batches.push(b));
  try {
    const primary = [{ icao24: "adb111", provider: "adsblol" }];
    const out = await filler.fillDisc(disc(4), primary, ["adsb.lol (ADS-B, community)"]);
    assert.deepEqual(urls, [adsbFiDiscUrl(CLT.lat, CLT.lon, 4)]);
    assert.equal(out.added, 2);
    assert.equal(out.aircraft[0], primary[0], "the primary row is kept as-is");
    const added = out.aircraft.slice(1) as Array<{ icao24: string; on_ground: boolean; provider?: string }>;
    assert.deepEqual(added.map((a) => a.icao24), ["a00001", "a00002"]);
    assert.equal(added[0].on_ground, true, "surface traffic arrives as on-ground");
    assert.ok(added.every((a) => a.provider === "adsbfi"), "provenance kept: lawful-only + tripwire still see adsb.fi");
    const gf = batches.filter((b) => b.origin === "gapfill");
    assert.equal(gf.length, 1);
    assert.equal(gf[0].provider, "adsbfi");
    assert.equal(gf[0].disc, undefined, "no sweep coverage credit for a gap-fill");
    assert.equal(filler.stats().rows_added, 2);
  } finally { off(); }
});

test("politeness: at most one adsb.fi request per GAPFILL_MIN_INTERVAL_MS across both paths", async () => {
  let t = 0;
  let calls = 0;
  const filler = createGapFiller({
    enabled: true, now: () => t,
    fetchImpl: (async () => { calls++; return res({ aircraft: [], ac: [] }); }) as unknown as typeof fetch,
  });
  await filler.fillDisc(disc(4), [], ["adsb.lol"]);
  t += GAPFILL_MIN_INTERVAL_MS - 1;
  await filler.fetchHex("ab8c8e");
  await filler.fillDisc(disc(4), [], ["adsb.lol"]);
  assert.equal(calls, 1);
  assert.equal(filler.stats().skipped_rate, 2);
  t += 1;
  await filler.fetchHex("ab8c8e");
  assert.equal(calls, 2);
});

test("disabled (AIRCRAFT_GAPFILL=0) or failing: the primary answer passes through untouched", async () => {
  assert.equal(gapFillEnabled({ AIRCRAFT_GAPFILL: "0" }), false);
  assert.equal(gapFillEnabled({}), true);
  let calls = 0;
  const off = createGapFiller({ enabled: false, fetchImpl: (async () => { calls++; return res({}); }) as unknown as typeof fetch });
  const p = [{ icao24: "x" }];
  assert.equal((await off.fillDisc(disc(4), p, ["adsb.lol"])).aircraft, p);
  assert.deepEqual(await off.fetchHex("ab8c8e"), []);
  assert.equal(calls, 0);

  const logged: unknown[] = [];
  const real = console.error;
  console.error = (...a: unknown[]) => { logged.push(a); };
  try {
    const bad = createGapFiller({ enabled: true, now: () => 5, fetchImpl: (async () => res({}, 503)) as unknown as typeof fetch });
    const out = await bad.fillDisc(disc(4), p, ["adsb.lol"]);
    assert.equal(out.aircraft, p);
    assert.equal(bad.stats().errors, 1);
  } finally { console.error = real; }
  assert.equal(logged.length, 1, "the first failure is logged, never silent");
});

test("fetchHex: only the asked hex, from adsb.fi's hex endpoint", async () => {
  const urls: string[] = [];
  const filler = createGapFiller({
    enabled: true,
    fetchImpl: (async (u: string) => { urls.push(u); return res({ ac: [fiRow("ab8c8e"), fiRow("ffffff")] }); }) as unknown as typeof fetch,
  });
  const rows = await filler.fetchHex("ab8c8e");
  assert.deepEqual(urls, [adsbFiHexUrl("ab8c8e")]);
  assert.deepEqual(rows.map((r: { icao24: string }) => r.icao24), ["ab8c8e"]);
});

test("fast lane: a hex adsb.lol returns NOTHING for is filled from adsb.fi (provider adsbfi)", async () => {
  const clock = 1_790_799_400_000;
  const snap = new GlobalSnapshot();
  const published: FixBatch[] = [];
  const lane = createFastLane({
    snapshotGet: (h) => snap.get(h),
    publish: (b) => { published.push(b); snap.ingest(b, clock); },
    fetchImpl: (async () => ({ ok: true, status: 200, headers: { get: () => null }, json: async () => ({ now: clock, ac: [] }) })) as unknown as typeof fetch,
    governor: new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 }),
    now: () => clock,
    fallbackHex: async (hex) => [{ icao24: hex, callsign: "N843S", lat: 35.21, lon: -80.94, altitude_m: 0, on_ground: true, heading: 180, provider: "adsbfi" }],
  });
  const r = await lane.lookup("ab8c8e");
  assert.equal(published.length, 1);
  assert.equal(published[0].provider, "adsbfi");
  assert.ok(r.row, "the taxiing plane now has a live row");
  assert.equal(lane.stats().fallback_rows, 1);
});

test("wiring: routes.ts gap-fills the viewport answer before archiving; globalAircraft passes the fast-lane fallback", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  const iFetch = routes.indexOf("const fetched = await fetchDiscs(plan");
  const iFill = routes.indexOf("gapFiller().fillDisc(plan, fetched.aircraft, sources)");
  const iArchive = routes.indexOf("archiveAircraft(aircraft, ARCHIVE_SITES)");
  assert.ok(iFetch > 0 && iFill > iFetch && iArchive > iFill, "fetch -> gap-fill -> archive (taxi fixes are recorded)");
  const ga = fs.readFileSync(path.join(here, "globalAircraft.ts"), "utf8");
  assert.match(ga, /fallbackHex: \(hex\) => filler\.fetchHex\(hex\)/);
});
