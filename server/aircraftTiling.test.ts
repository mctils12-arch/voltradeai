// Aircraft viewport tiling (human directive 2026-07-16: "the whole visible
// range on the map", not one 250nm disc). Covers: disc-packing geometry
// (bbox -> centers, minimal count, politeness cap), dedupe across discs,
// backward compat (no bbox = single legacy query), coverage-metadata
// truthfulness, and the orchestration through a real express server with
// the adsb.lol fetch MOCKED — tests never hit the live APIs.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import {
  planDiscs, dedupeByHex, fetchDiscs, tilingEnvelope, mapPointAircraft,
  DISC_RADIUS_MAX_NM, MAX_DISCS_PER_REFRESH, DISC_SPACING_MS,
  type DiscPlan, type DiscProvider,
} from "./aircraftTiling";

const here = path.dirname(fileURLToPath(import.meta.url));

// Same local metric the planner uses (1 deg lat = 60nm, lon scaled by
// cos(bbox-center lat)) — coverage assertions are exact in this metric.
function nmDist(aLat: number, aLon: number, bLat: number, bLon: number, clat: number): number {
  const cos = Math.max(0.1, Math.cos((clat * Math.PI) / 180));
  return Math.hypot((aLat - bLat) * 60, (aLon - bLon) * 60 * cos);
}

function assertBboxCovered(plan: DiscPlan, lamin: number, lamax: number, lomin: number, lomax: number) {
  const clat = (lamin + lamax) / 2;
  const N = 12;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const lat = lamin + (i / N) * (lamax - lamin);
      const lon = lomin + (j / N) * (lomax - lomin);
      const d = Math.min(...plan.centers.map((c) => nmDist(lat, lon, c.lat, c.lon, clat)));
      assert.ok(d <= plan.radiusNm + 1e-6,
        `point (${lat.toFixed(2)},${lon.toFixed(2)}) is ${d.toFixed(1)}nm from nearest disc (radius ${plan.radiusNm})`);
    }
  }
}

// ── disc-packing geometry ────────────────────────────────────────────────────

test("planDiscs: small bbox -> one disc at center, legacy 50nm floor, covered", () => {
  const p = planDiscs(39.5, 40.5, -75.5, -74.5);
  assert.equal(p.centers.length, 1);
  assert.equal(p.needed, 1);
  assert.equal(p.bboxCovered, true);
  assert.equal(p.centers[0].lat, 40);
  assert.equal(p.centers[0].lon, -75);
  assert.equal(p.radiusNm, 50, "tiny viewport keeps the legacy 50nm request floor");
  assertBboxCovered(p, 39.5, 40.5, -75.5, -74.5);
});

test("planDiscs: archive-tick region (Cushing box) still needs exactly one disc", () => {
  // Regions in routes.ts ARCHIVE_TICK_REGIONS were sized for one point-query;
  // tiling must not change what the eager archive tick fetches.
  const p = planDiscs(33, 39, -100, -93);
  assert.equal(p.centers.length, 1);
  assert.equal(p.needed, 1);
  // Legacy radius formula: min(250, max(50, ceil(hypot(latNm, lonNm)/2)))
  const latNm = 6 * 60;
  const lonNm = 7 * 60 * Math.cos((36 * Math.PI) / 180);
  assert.equal(p.radiusNm, Math.max(50, Math.ceil(Math.hypot(latNm, lonNm) / 2)));
  assert.ok(p.radiusNm <= DISC_RADIUS_MAX_NM);
});

test("planDiscs: wide equatorial bbox -> MINIMAL 2-disc cover, exact centers/radius", () => {
  // 3 deg lat x 8 deg lon at the equator = 180 x 480 nm: one 250nm disc
  // cannot cover it (needs ~257nm), two side-by-side discs can.
  const p = planDiscs(-1.5, 1.5, -4, 4);
  assert.equal(p.needed, 2, "grid must be minimal — 2 discs, not more");
  assert.equal(p.centers.length, 2);
  assert.equal(p.bboxCovered, true);
  assert.deepEqual(p.centers.map((c) => c.lat), [0, 0]);
  assert.deepEqual(p.centers.map((c) => c.lon).sort((a, b) => a - b), [-2, 2]);
  assert.equal(p.radiusNm, 150, "radius shrinks to the cell's half-diagonal, not blanket 250");
  assertBboxCovered(p, -1.5, 1.5, -4, 4);
});

test("planDiscs: multi-row cover — every sampled viewport point is inside some disc", () => {
  // ~CONUS-northeast quadrant scale: needs a 2x3-ish grid but stays under the cap.
  const cases: Array<[number, number, number, number]> = [
    [32, 42, -96, -84],    // 600 x ~575 nm US south-central: 2x2 grid
    [45, 55, 0, 15],       // Europe band
    [-38, -28, 142, 152],  // Australia band
  ];
  for (const [a, b, c, d] of cases) {
    const p = planDiscs(a, b, c, d);
    assert.ok(p.needed >= 2 && p.needed <= MAX_DISCS_PER_REFRESH, `case ${a},${b},${c},${d}: needed=${p.needed}`);
    assert.equal(p.centers.length, p.needed);
    assert.equal(p.bboxCovered, true);
    assert.ok(p.radiusNm <= DISC_RADIUS_MAX_NM);
    assertBboxCovered(p, a, b, c, d);
  }
});

test("planDiscs: politeness cap — global viewport gets 8 central discs, honestly uncovered", () => {
  const p = planDiscs(-85, 85, -180, 180);
  assert.equal(p.centers.length, MAX_DISCS_PER_REFRESH, "never more than the cap per refresh");
  assert.ok(p.needed > MAX_DISCS_PER_REFRESH, "full cover would need far more — stated, not hidden");
  assert.equal(p.bboxCovered, false);
  assert.equal(p.cappedAt, MAX_DISCS_PER_REFRESH);
  // Central region: disc centroid sits on the viewport center.
  const mLat = p.centers.reduce((s, c) => s + c.lat, 0) / p.centers.length;
  const mLon = p.centers.reduce((s, c) => s + c.lon, 0) / p.centers.length;
  assert.ok(Math.abs(mLat) < 1e-9 && Math.abs(mLon) < 1e-9, "capped discs center on the viewport");
  for (const c of p.centers) {
    assert.ok(c.lat >= -85 && c.lat <= 85 && c.lon >= -180 && c.lon <= 180);
  }
  // Disc #1 is the most central (provider probe = best single disc).
  const d0 = nmDist(p.centers[0].lat, p.centers[0].lon, 0, 0, 0);
  for (const c of p.centers) assert.ok(d0 <= nmDist(c.lat, c.lon, 0, 0, 0) + 1e-9);
});

test("planDiscs: maxDiscs=1 reproduces the LEGACY single point-query exactly", () => {
  // Wide bbox: legacy = one disc at center, radius clamped to 250.
  const wide = planDiscs(-85, 85, -180, 180, 1);
  assert.equal(wide.centers.length, 1);
  assert.deepEqual(wide.centers[0], { lat: 0, lon: 0 });
  assert.equal(wide.radiusNm, 250);
  assert.equal(wide.bboxCovered, false);
  assert.ok(wide.neededNm > 250, "legacy coverage_note needs the honest neededNm");
  // Small bbox: identical to the tiled plan (one disc either way).
  const small1 = planDiscs(39.5, 40.5, -75.5, -74.5, 1);
  const smallN = planDiscs(39.5, 40.5, -75.5, -74.5);
  assert.deepEqual(small1.centers, smallN.centers);
  assert.equal(small1.radiusNm, smallN.radiusNm);
});

// ── dedupe across discs ──────────────────────────────────────────────────────

test("dedupeByHex: overlapping discs merge to unique aircraft, first (most central) wins", () => {
  const discA = [{ icao24: "a1", lat: 1 }, { icao24: "b2", lat: 2 }];
  const discB = [{ icao24: "b2", lat: 99 }, { icao24: "c3", lat: 3 }];
  const discC = [{ icao24: "a1", lat: 98 }, { icao24: "d4", lat: 4 }, { lat: 5 }, { lat: 6 }];
  const out = dedupeByHex([discA, discB, discC]);
  assert.equal(out.length, 6, "4 unique hexes + 2 id-less records kept");
  assert.deepEqual(out.filter((a) => a.icao24).map((a) => a.icao24), ["a1", "b2", "c3", "d4"]);
  assert.equal(out.find((a) => a.icao24 === "b2").lat, 2, "first occurrence wins");
  assert.equal(out.find((a) => a.icao24 === "a1").lat, 1, "first occurrence wins");
});

// ── mapping (extracted verbatim from routes.ts) ─────────────────────────────

test("mapPointAircraft: readsb record -> unified shape (units, ground, null-pos filter)", () => {
  const raw = { ac: [
    { hex: "abc123", flight: "UAL1  ", r: "N12345", lat: 40, lon: -75, alt_baro: 35000, gs: 450, track: 270, t: "B738", category: "A3" },
    { hex: "def456", lat: 41, lon: -74, alt_baro: "ground", gs: 0 },
    { hex: "nopos1", alt_baro: 1000 },
  ] };
  const out = mapPointAircraft(raw, "ac");
  assert.equal(out.length, 2, "records without lat/lon are dropped");
  assert.equal(out[0].icao24, "abc123");
  assert.equal(out[0].callsign, "UAL1");
  assert.equal(out[0].altitude_m, Math.round(35000 * 0.3048));
  assert.equal(out[0].velocity_ms, Math.round(450 * 0.5144));
  assert.equal(out[0].on_ground, false);
  assert.equal(out[1].on_ground, true);
  assert.equal(out[1].altitude_m, null);
  assert.deepEqual(mapPointAircraft(null, "ac"), []);
  assert.deepEqual(mapPointAircraft({ aircraft: [] }, "aircraft"), []);
});

test("mapPointAircraft: alt_geom fallback — a missing alt_baro no longer nulls the altitude of an airborne row", () => {
  const raw = { ac: [
    // alt_baro dropped mid-flight, alt_geom (GNSS) still broadcast — the
    // 2026-07-31 curtain-gap case: the row is airborne with a real altitude
    { hex: "geo001", lat: 40, lon: -75, alt_geom: 36000, gs: 440 },
    // both broadcast → baro wins (the aviation convention)
    { hex: "both01", lat: 41, lon: -74, alt_baro: 35000, alt_geom: 35400 },
    // neither broadcast → honestly null (never invented)
    { hex: "none01", lat: 42, lon: -73, gs: 250 },
    // on ground: alt_geom may still carry field elevation — altitude stays
    // null and on_ground true, exactly as before
    { hex: "gnd001", lat: 43, lon: -72, alt_baro: "ground", alt_geom: 650 },
  ] };
  const out = mapPointAircraft(raw, "ac");
  assert.equal(out.find((a) => a.icao24 === "geo001").altitude_m, Math.round(36000 * 0.3048));
  assert.equal(out.find((a) => a.icao24 === "geo001").on_ground, false);
  assert.equal(out.find((a) => a.icao24 === "both01").altitude_m, Math.round(35000 * 0.3048));
  assert.equal(out.find((a) => a.icao24 === "none01").altitude_m, null);
  assert.equal(out.find((a) => a.icao24 === "gnd001").altitude_m, null);
  assert.equal(out.find((a) => a.icao24 === "gnd001").on_ground, true);
});

// ── fetchDiscs orchestration (mocked upstream) ──────────────────────────────

const CHAIN: DiscProvider[] = [
  { key: "adsblol", label: "adsb.lol (ADS-B, community)", arr: "ac", url: (la, lo, r) => `https://api.adsb.lol/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${r}` },
  { key: "airplaneslive", label: "airplanes.live (ADS-B, community)", arr: "ac", url: (la, lo, r) => `https://api.airplanes.live/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${r}` },
];
const ok = (body: any) => ({ ok: true, json: async () => body }) as any;
const bad = (status: number) => ({ ok: false, status }) as any;
const ac = (...hexes: string[]) => ({ ac: hexes.map((h) => ({ hex: h, lat: 1, lon: 1 })) });

test("fetchDiscs: single-disc plan = exactly one upstream call, adsb.lol first", async () => {
  const plan = planDiscs(39.5, 40.5, -75.5, -74.5);
  const calls: string[] = [];
  const r = await fetchDiscs(plan, {
    providers: CHAIN,
    fetchImpl: (async (u: any) => { calls.push(String(u)); return ok(ac("x1", "x2")); }) as any,
    sleep: async () => {},
  });
  assert.equal(calls.length, 1, "no bbox tiling -> single legacy query");
  assert.ok(calls[0].startsWith("https://api.adsb.lol/v2/point/"), "ODbL provider leads");
  assert.ok(calls[0].endsWith("/50"), "legacy radius floor in the URL");
  assert.equal(r.discsOk, 1);
  assert.equal(r.aircraft.length, 2);
  assert.deepEqual(r.sources, ["adsb.lol (ADS-B, community)"]);
});

test("fetchDiscs: multi-disc — dedupes across discs, staggers sub-queries, one provider", async () => {
  const plan = planDiscs(-1.5, 1.5, -4, 4); // 2 discs at lon -2 / +2
  const calls: string[] = [];
  const sleeps: number[] = [];
  const r = await fetchDiscs(plan, {
    providers: CHAIN,
    fetchImpl: (async (u: any) => {
      calls.push(String(u));
      // The overlap zone: hex "shared" appears in BOTH discs.
      return ok(String(u).includes("/-2.000/") ? ac("aaa", "shared") : ac("bbb", "shared", "ccc"));
    }) as any,
    sleep: async (ms) => { sleeps.push(ms); },
  });
  assert.equal(calls.length, 2);
  assert.ok(calls.every((u) => u.startsWith("https://api.adsb.lol/")));
  assert.deepEqual(sleeps, [DISC_SPACING_MS], "sub-queries launch spaced apart, never as a burst");
  assert.equal(r.discsOk, 2);
  assert.equal(r.aircraft.length, 4, "shared hex counted once across discs");
  assert.deepEqual(r.aircraft.map((a: any) => a.icao24).sort(), ["aaa", "bbb", "ccc", "shared"]);
});

test("fetchDiscs: a failing disc falls through the chain per-disc; bump/clear hooks fire", async () => {
  const plan = planDiscs(-1.5, 1.5, -4, 4);
  const bumped: string[] = [];
  const cleared: string[] = [];
  const r = await fetchDiscs(plan, {
    providers: CHAIN,
    fetchImpl: (async (u: any) => {
      const s = String(u);
      if (s.includes("adsb.lol") && s.includes("/2.000/")) return bad(500); // disc 2 primary fails
      return ok(ac(s.includes("airplanes.live") ? "via-fallback" : "via-primary"));
    }) as any,
    backoffBump: (k) => bumped.push(k),
    backoffClear: (k) => cleared.push(k),
    sleep: async () => {},
  });
  assert.equal(r.discsOk, 2, "fallback provider rescues the failing disc");
  assert.deepEqual(bumped, ["adsblol"]);
  assert.ok(cleared.includes("airplaneslive"));
  assert.deepEqual([...r.sources].sort(), ["adsb.lol (ADS-B, community)", "airplanes.live (ADS-B, community)"]);
  assert.ok(r.errs.some((e) => e.includes("adsblol: adsblol 500")));
});

test("fetchDiscs: disc #1 chain failure throws BEFORE any fan-out (outage costs one walk)", async () => {
  const plan = planDiscs(-1.5, 1.5, -4, 4);
  let calls = 0;
  await assert.rejects(
    fetchDiscs(plan, {
      providers: CHAIN,
      fetchImpl: (async () => { calls++; return bad(503); }) as any,
      sleep: async () => {},
    }),
    /503/,
  );
  assert.equal(calls, CHAIN.length, "only disc #1 walked the chain — discs 2..N never launched");
});

test("fetchDiscs: a non-first disc failing on ALL providers degrades coverage, not the refresh", async () => {
  const plan = planDiscs(-1.5, 1.5, -4, 4);
  const r = await fetchDiscs(plan, {
    providers: CHAIN,
    fetchImpl: (async (u: any) =>
      String(u).includes("/2.000/") ? bad(500) : ok(ac("only-central"))) as any,
    sleep: async () => {},
  });
  assert.equal(r.discsOk, 1);
  assert.equal(r.aircraft.length, 1);
  const env = tilingEnvelope(plan, r.discsOk, false);
  assert.equal(env.coverage, "partial");
  assert.equal(env.tiling.bboxCovered, false, "a failed disc means the bbox was NOT covered — never claim it was");
  assert.match(env.coverage_note!, /partial refresh: 1 of 2/);
});

// ── coverage metadata truthfulness ──────────────────────────────────────────

test("tilingEnvelope: full cover + all discs answered -> coverage full, no note", () => {
  const plan = planDiscs(-1.5, 1.5, -4, 4);
  const env = tilingEnvelope(plan, 2, false);
  assert.equal(env.coverage, "full");
  assert.equal(env.coverage_note, undefined);
  assert.deepEqual(env.tiling, { discs: 2, needed: 2, cappedAt: MAX_DISCS_PER_REFRESH, bboxCovered: true });
});

test("tilingEnvelope: capped viewport -> honest 'central region' note + structured metadata", () => {
  const plan = planDiscs(-85, 85, -180, 180);
  const env = tilingEnvelope(plan, plan.centers.length, false);
  assert.equal(env.coverage, "partial");
  assert.match(env.coverage_note!, /viewport too large — showing central region/);
  assert.match(env.coverage_note!, new RegExp(`${plan.centers.length} of ${plan.needed} 250nm coverage discs`));
  assert.deepEqual(env.tiling, {
    discs: plan.centers.length, needed: plan.needed,
    cappedAt: MAX_DISCS_PER_REFRESH, bboxCovered: false,
  });
});

test("tilingEnvelope: legacy (no-bbox) mode keeps the old coverage_note wording verbatim", () => {
  const plan = planDiscs(-85, 85, -180, 180, 1);
  const env = tilingEnvelope(plan, 1, true);
  assert.equal(env.coverage, "partial");
  assert.equal(env.coverage_note,
    `feed covers ~250nm around view center (viewport needs ~${plan.neededNm}nm) — zoom in for full coverage`);
  assert.equal(env.tiling.discs, 1);
  assert.equal(env.tiling.bboxCovered, false);
});

// ── end-to-end through a real express server (upstream mocked) ─────────────

test("express e2e: bbox tiles the viewport, no bbox stays a single legacy query", async () => {
  const calls: string[] = [];
  const mockFetch = (async (u: any) => {
    calls.push(String(u));
    return ok(String(u).includes("/-2.000/") ? ac("west1", "shared") : ac("east1", "shared"));
  }) as any;

  // Mini-mount assembled from the SAME building blocks routes.ts uses
  // (planDiscs -> fetchDiscs -> tilingEnvelope; a source-pin test below
  // keeps routes.ts on these blocks so this mount can't silently drift).
  const app = express();
  app.get("/api/data/aircraft", async (req, res) => {
    const q = req.query as Record<string, any>;
    const tiled = ["lamin", "lamax", "lomin", "lomax"].every(
      (k) => q[k] !== undefined && Number.isFinite(parseFloat(String(q[k]))));
    const num = (v: any, d: number) => (Number.isFinite(parseFloat(String(v))) ? parseFloat(String(v)) : d);
    const plan = planDiscs(num(q.lamin, -85), num(q.lamax, 85), num(q.lomin, -180), num(q.lomax, 180),
      tiled ? MAX_DISCS_PER_REFRESH : 1);
    try {
      const r = await fetchDiscs(plan, { providers: CHAIN, fetchImpl: mockFetch, sleep: async () => {} });
      res.json({ source: r.sources.join(" + "), kind: "raw", time: 1,
        ...tilingEnvelope(plan, r.discsOk, !tiled), count: r.aircraft.length, aircraft: r.aircraft });
    } catch (e: any) {
      res.status(502).json({ error: String(e?.message || e), aircraft: [] });
    }
  });
  const srv = app.listen(0);
  try {
    const port = (srv.address() as any).port;

    // Explicit viewport bbox -> 2 discs, deduped merge, truthful metadata.
    const tiledResp = await (await fetch(
      `http://127.0.0.1:${port}/api/data/aircraft?lamin=-1.5&lamax=1.5&lomin=-4&lomax=4`)).json() as any;
    assert.equal(calls.length, 2, "wide bbox = one query per covering disc");
    assert.equal(tiledResp.coverage, "full");
    assert.deepEqual(tiledResp.tiling, { discs: 2, needed: 2, cappedAt: MAX_DISCS_PER_REFRESH, bboxCovered: true });
    assert.equal(tiledResp.count, 3, "shared hex deduped across discs");
    assert.equal(tiledResp.source, "adsb.lol (ADS-B, community)");

    // BACKWARD COMPAT: no bbox params -> exactly one legacy query.
    calls.length = 0;
    const legacyResp = await (await fetch(`http://127.0.0.1:${port}/api/data/aircraft`)).json() as any;
    assert.equal(calls.length, 1, "absent bbox keeps today's single point-query");
    assert.equal(legacyResp.tiling.discs, 1);
    assert.equal(legacyResp.tiling.bboxCovered, false);
    assert.match(legacyResp.coverage_note, /feed covers ~250nm around view center/);

    // Partial params (old callers) also stay legacy.
    calls.length = 0;
    await (await fetch(`http://127.0.0.1:${port}/api/data/aircraft?lamin=30&lamax=45`)).json();
    assert.equal(calls.length, 1, "incomplete bbox never tiles");
  } finally {
    srv.close();
  }
});

// ── wiring pins: routes.ts actually runs on these building blocks ──────────

test("routes.ts wiring: aircraft route uses planDiscs/fetchDiscs/tilingEnvelope with the bbox opt-in gate", () => {
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(src.includes('from "./aircraftTiling"'), "tiling module not imported");
  assert.ok(src.includes("planDiscs(lamin, lamax, lomin, lomax, tiled ? MAX_DISCS_PER_REFRESH : 1)"),
    "fetchAircraft must plan discs, capped for tiled requests and single for legacy");
  assert.ok(src.includes("fetchDiscs(plan"), "fetchAircraft must fetch through the shared disc orchestrator");
  assert.ok(src.includes("tilingEnvelope(plan, discsOk, !tiled)"),
    "response must carry the truthful coverage/tiling envelope");
  assert.ok(src.includes("fetchAircraft(lamin, lamax, lomin, lomax, tiled)"),
    "route handler must pass the bbox opt-in flag into the fetch");
  const handler = src.slice(src.indexOf('app.get("/api/data/aircraft"'), src.indexOf('app.get("/api/data/aircraft"') + 3200);
  assert.ok(/const tiled = \[req\.query\.lamin, req\.query\.lamax, req\.query\.lomin, req\.query\.lomax\]/.test(handler),
    "tiling is opt-in by presence of a FULL explicit bbox (backward compat)");
  assert.ok(handler.includes("${tiled ?"), "cache key must separate tiled and legacy snapshots");
});

// ── GNSS integrity passthrough (2026-08-11): mapPointAircraft must now CARRY
//    the navigation-integrity/accuracy fields it used to discard, with the
//    null-is-not-zero rule and schema-drift resilience. ──
const RAW_FULL = {
  hex: "4cada4", flight: "RYR8779 ", r: "EI-IFV", lat: 57.0, lon: 12.6,
  alt_baro: 12425, alt_geom: 13000, gs: 306, track: 327, t: "B38M", category: "A3",
  nic: 8, nac_p: 10, nac_v: 2, sil: 3, sil_type: "perhour", rc: 186, gva: 2, sda: 2,
  nic_baro: 1, type: "adsb_icao", mlat: [], tisb: [], seen_pos: 0.4,
  gpsOkLat: 56.9, gpsOkLon: 12.5, gpsOkBefore: 3.2,
};

test("mapPointAircraft carries integrity + origin + provenance", () => {
  const [p] = mapPointAircraft({ ac: [RAW_FULL] }, "ac", "adsblol");
  assert.equal(p.nic, 8); assert.equal(p.nac_p, 10); assert.equal(p.nac_v, 2);
  assert.equal(p.sil, 3); assert.equal(p.sil_type, "perhour"); assert.equal(p.rc, 186);
  assert.equal(p.gva, 2); assert.equal(p.sda, 2); assert.equal(p.nic_baro, 1);
  assert.equal(p.pos_type, "adsb_icao");
  assert.equal(p.lkg_lat, 56.9); assert.equal(p.lkg_lon, 12.5); assert.equal(p.lkg_before, 3.2);
  assert.equal(p.seen_pos, 0.4);
  assert.equal(p.provider, "adsblol");
});

test("mapPointAircraft: pos_type is a.type, NOT a.t (the model designator)", () => {
  const [p] = mapPointAircraft({ ac: [{ ...RAW_FULL, t: "B738", type: "mlat" }] }, "ac", "adsblol");
  assert.equal(p.type, "B738", "ICAO model designator stays on `type`");
  assert.equal(p.pos_type, "mlat", "position-source discriminator lands on pos_type");
});

test("mapPointAircraft: a REPORTED 0 survives, a MISSING field is null (never 0)", () => {
  // nic:0 is a real integrity category — must be kept as 0, not dropped
  const [zero] = mapPointAircraft({ ac: [{ ...RAW_FULL, nic: 0, nac_p: 0, sil: 0 }] }, "ac", "adsblol");
  assert.equal(zero.nic, 0); assert.equal(zero.nac_p, 0); assert.equal(zero.sil, 0);
  // absent fields → null (silence), NEVER 0
  const bare = { hex: "abc123", flight: "X", lat: 1, lon: 2, alt_baro: 30000 };
  const [n] = mapPointAircraft({ ac: [bare] }, "ac", "adsblol");
  for (const k of ["nic", "nac_p", "nac_v", "sil", "sil_type", "rc", "gva", "sda", "nic_baro",
                   "pos_type", "mlat_fields", "tisb_fields", "lkg_lat", "lkg_lon", "lkg_before", "seen_pos"]) {
    assert.equal(n[k], null, `${k} must be null when absent, not 0/undefined`);
    assert.notEqual(n[k], 0, `${k} must never fabricate a 0 from silence`);
  }
});

test("mapPointAircraft: schema drift (garbage/renamed fields) → null, never a crash", () => {
  const junk = { hex: "z", flight: "Y", lat: 1, lon: 2, alt_baro: 30000,
    nic: "eight", nac_p: null, sil: {}, mlat: "notanarray", tisb: 5, type: 99,
    gpsOkLat: "n/a", seen_pos: NaN };
  let p: any;
  assert.doesNotThrow(() => { [p] = mapPointAircraft({ ac: [junk] }, "ac", "adsblol"); });
  assert.equal(p.nic, null); assert.equal(p.nac_p, null); assert.equal(p.sil, null);
  assert.equal(p.mlat_fields, null); assert.equal(p.tisb_fields, null);
  assert.equal(p.pos_type, null); assert.equal(p.lkg_lat, null); assert.equal(p.seen_pos, null);
});

test("mapPointAircraft: mlat/tisb derivation arrays pass through (string-filtered)", () => {
  const [p] = mapPointAircraft({ ac: [{ ...RAW_FULL, mlat: ["lat", "lon", 7], tisb: ["alt_baro"] }] }, "ac", "adsblol");
  assert.deepEqual(p.mlat_fields, ["lat", "lon"], "non-string entries dropped");
  assert.deepEqual(p.tisb_fields, ["alt_baro"]);
});
