/**
 * viewport.test.ts — SCALE PROGRAM S1(a) proof.
 *
 * Hermetic (no network, no clock): synthetic feature fixtures only. Proves:
 *  1. parseBbox validation (garbage / range / antimeridian / degenerate).
 *  2. inBbox boundary inclusivity + antimeridian wrap.
 *  3. filterByViewport is single-pass O(n) — the accessor is called exactly
 *     once per feature (no quadratic scan), and the RENDERED payload stays
 *     ~flat as TOTAL feature count scales 100x, because off-screen data is
 *     free. This is the whole point of the program, asserted as numbers.
 *  4. Backward compatibility: the aircraft-route filter is a no-op when no
 *     bbox is supplied (null bbox => original set, byte-for-byte).
 *  5. No-silent-caps envelope fields.
 *
 * Run: npx tsx --test server/viewport.test.ts
 */
import { test } from "node:test";
import assert from "node:assert";
import { parseBbox, inBbox, filterByViewport, applyViewport, Bbox } from "./viewport";

// ── parseBbox validation ──────────────────────────────────────────────────
test("parseBbox: valid box parses; whitespace tolerated", () => {
  assert.deepEqual(parseBbox("-10,-5,10,5"), { minLon: -10, minLat: -5, maxLon: 10, maxLat: 5 });
  assert.deepEqual(parseBbox(" -10 , -5 , 10 , 5 "), { minLon: -10, minLat: -5, maxLon: 10, maxLat: 5 });
});

test("parseBbox: garbage returns null, never throws", () => {
  for (const g of ["", "1,2,3", "1,2,3,4,5", "a,b,c,d", "1,2,3,x", "NaN,0,1,1",
                   undefined, null, 42, {}, [], "1,2,3,Infinity"]) {
    assert.equal(parseBbox(g as any), null, `expected null for ${JSON.stringify(g)}`);
  }
});

test("parseBbox: out-of-range coordinates rejected", () => {
  assert.equal(parseBbox("-181,0,10,5"), null); // lon < -180
  assert.equal(parseBbox("0,-91,10,5"), null);  // lat < -90
  assert.equal(parseBbox("0,0,181,5"), null);   // lon > 180
  assert.equal(parseBbox("0,0,10,91"), null);   // lat > 90
});

test("parseBbox: degenerate boxes rejected (zero height / zero width / inverted lat)", () => {
  assert.equal(parseBbox("0,5,10,5"), null);  // minLat == maxLat
  assert.equal(parseBbox("0,6,10,5"), null);  // minLat > maxLat (no lat wrap)
  assert.equal(parseBbox("10,0,10,5"), null); // minLon == maxLon
});

test("parseBbox: antimeridian-wrapping box (minLon > maxLon) is ACCEPTED", () => {
  assert.deepEqual(parseBbox("170,-5,-170,5"), { minLon: 170, minLat: -5, maxLon: -170, maxLat: 5 });
});

// ── inBbox boundary + wrap ────────────────────────────────────────────────
test("inBbox: boundary-inclusive on all four edges", () => {
  const b: Bbox = { minLon: -10, minLat: -5, maxLon: 10, maxLat: 5 };
  assert.ok(inBbox(-10, -5, b)); // SW corner
  assert.ok(inBbox(10, 5, b));   // NE corner
  assert.ok(inBbox(-10, 0, b));  // W edge
  assert.ok(inBbox(0, 5, b));    // N edge
  assert.ok(inBbox(0, 0, b));    // interior
  assert.ok(!inBbox(-10.0001, 0, b)); // just outside W
  assert.ok(!inBbox(0, 5.0001, b));   // just outside N
});

test("inBbox: non-finite coordinates are never inside", () => {
  const b: Bbox = { minLon: -10, minLat: -5, maxLon: 10, maxLat: 5 };
  assert.ok(!inBbox(NaN, 0, b));
  assert.ok(!inBbox(0, Infinity, b));
});

test("inBbox: antimeridian wrap matches both sides of the seam, excludes the middle", () => {
  const b = parseBbox("170,-5,-170,5")!; // covers 170..180 and -180..-170
  assert.ok(inBbox(175, 0, b));   // east side
  assert.ok(inBbox(-175, 0, b));  // west side
  assert.ok(inBbox(180, 0, b));   // seam edge (inclusive)
  assert.ok(inBbox(-180, 0, b));  // seam edge (inclusive)
  assert.ok(!inBbox(0, 0, b));    // middle of the globe excluded
  assert.ok(!inBbox(160, 0, b));  // just west of east edge excluded
});

// ── PERF / CORRECTNESS: flat rendered payload as total data scales ─────────
type Pt = { lon: number; lat: number; id: number };
const getLL = (p: Pt): [number, number] => [p.lon, p.lat];

/**
 * Build a fixture with a FIXED cluster of in-view points plus `outside`
 * points spread deterministically across the rest of the globe (all outside
 * the small central bbox). This models "add more streams / more global data":
 * the local density of the viewport is fixed; total volume scales.
 */
function buildFixture(inView: number, outside: number): Pt[] {
  const pts: Pt[] = [];
  let id = 0;
  // In-view cluster inside bbox [-1,-1,1,1].
  for (let i = 0; i < inView; i++) {
    pts.push({ lon: -0.9 + (1.8 * i) / inView, lat: -0.9 + (1.8 * i) / inView, id: id++ });
  }
  // Off-screen spread across lon [20..180]/[-180..-20], lat [-80..80].
  for (let i = 0; i < outside; i++) {
    const t = i / outside;
    const lon = (i % 2 === 0 ? 1 : -1) * (20 + t * 160); // |lon| in [20,180], never in [-1,1]
    const lat = -80 + 160 * ((i * 37) % outside) / outside;
    pts.push({ lon, lat, id: id++ });
  }
  return pts;
}

test("PERF: rendered payload stays flat as TOTAL scales 100x (off-screen is free)", () => {
  const bbox = parseBbox("-1,-1,1,1")!;
  const IN_VIEW = 50;

  const small = buildFixture(IN_VIEW, 1_000);   // total 1,050
  const large = buildFixture(IN_VIEW, 100_000); // total 100,050

  const rSmall = filterByViewport(small, bbox, getLL);
  const rLarge = filterByViewport(large, bbox, getLL);

  // The rendered set is bounded and CONSTANT despite 95x more total data.
  assert.equal(rSmall.length, IN_VIEW, "small fixture: exactly the in-view cluster");
  assert.equal(rLarge.length, IN_VIEW, "large fixture: SAME in-view count despite 100x total");

  // Serialized payload size is flat too (what actually crosses the wire).
  const bytesSmall = JSON.stringify(rSmall).length;
  const bytesLarge = JSON.stringify(rLarge).length;
  assert.equal(bytesSmall, bytesLarge, "rendered payload bytes independent of total data volume");
});

test("PERF: filterByViewport is single-pass O(n) — accessor called exactly once per feature", () => {
  const bbox = parseBbox("-1,-1,1,1")!;
  for (const total of [1_050, 100_050]) {
    const outside = total - 50;
    const pts = buildFixture(50, outside);
    let calls = 0;
    const counting = (p: Pt): [number, number] => { calls++; return [p.lon, p.lat]; };
    const out = filterByViewport(pts, bbox, counting);
    assert.equal(calls, pts.length, `accessor called once per feature (n=${pts.length}), no quadratic scan`);
    assert.equal(out.length, 50, "in-view count constant");
  }
});

test("filterByViewport: skips features whose accessor returns null/undefined", () => {
  const bbox = parseBbox("-1,-1,1,1")!;
  const pts = [{ lon: 0, lat: 0, id: 1 }, { lon: 0, lat: 0, id: 2 }, { lon: 0, lat: 0, id: 3 }];
  const out = filterByViewport(pts, bbox, (p) => (p.id === 2 ? null : [p.lon, p.lat]));
  assert.deepEqual(out.map((p) => p.id), [1, 3]);
});

// ── BACKWARD COMPAT: the route's applyView is a no-op without a bbox ───────
// Mirror the exact closure used in routes.ts /api/data/aircraft so the
// backward-compat guarantee is pinned by a test, not just by inspection.
function applyView(payload: any, viewBbox: Bbox | null) {
  if (!viewBbox || !Array.isArray(payload?.aircraft)) return payload;
  const before = payload.aircraft.length;
  const inView = filterByViewport(payload.aircraft, viewBbox, (a: any) => [a.lon, a.lat]);
  return {
    ...payload,
    aircraft: inView,
    count: inView.length,
    viewport_filtered: true,
    count_before_viewport: before,
    count_dropped_offscreen: before - inView.length,
  };
}

test("BACKWARD-COMPAT: no bbox => payload returned by reference, byte-for-byte identical", () => {
  const data = {
    source: "adsb.lol (ADS-B, community)", kind: "raw", time: 1234, coverage: "full",
    count: 3, aircraft: [{ icao24: "a", lon: 0, lat: 0 }, { icao24: "b", lon: 100, lat: 40 }, { icao24: "c", lon: -100, lat: -40 }],
  };
  const before = JSON.stringify(data);
  const out = applyView(data, null); // parseBbox("garbage") also yields null → same path
  assert.strictEqual(out, data, "no-bbox returns the SAME object reference (no copy, no filter)");
  assert.equal(JSON.stringify(out), before, "no-bbox output is byte-for-byte the original");
  assert.equal((out as any).viewport_filtered, undefined, "no viewport fields added when no bbox");
});

test("NO-SILENT-CAPS: with bbox, envelope states filtered count + pre-filter count + drop", () => {
  const data = {
    source: "adsb.lol (ADS-B, community)", kind: "raw", time: 1234, count: 3,
    aircraft: [
      { icao24: "in", lon: 0.5, lat: 0.5 },      // inside [-1,-1,1,1]
      { icao24: "out1", lon: 100, lat: 40 },     // outside
      { icao24: "out2", lon: -100, lat: -40 },   // outside
    ],
  };
  const out = applyView(data, parseBbox("-1,-1,1,1"));
  assert.equal(out.viewport_filtered, true);
  assert.equal(out.count, 1, "count reflects the rendered (in-view) set");
  assert.equal(out.count_before_viewport, 3, "pre-filter total is stated");
  assert.equal(out.count_dropped_offscreen, 2, "dropped-off-screen count is stated (no silent cap)");
  assert.deepEqual(out.aircraft.map((a: any) => a.icao24), ["in"]);
  // Untouched envelope fields survive.
  assert.equal(out.source, "adsb.lol (ADS-B, community)");
  assert.equal(out.time, 1234);
});

// ── applyViewport: the serve-time route helper (kept out of the handler body
//    so the raceDeadline guard window is untouched) ───────────────────────────

test("applyViewport: valid bbox filters payload[arrayKey] + rewrites count fields, no silent caps", () => {
  const payload = {
    time: "t1", cached: true,
    aircraft: [
      { id: "in", lon: 0, lat: 0 },
      { id: "out", lon: 40, lat: 40 },
      { id: "edge", lon: 1, lat: 1 },
    ],
    count: 3,
  };
  const out = applyViewport(payload, "-1,-1,1,1", "aircraft", (a: any) => [a.lon, a.lat]);
  assert.deepEqual(out.aircraft.map((a: any) => a.id), ["in", "edge"]);
  assert.equal(out.count, 2);
  assert.equal(out.count_before_viewport, 3);
  assert.equal(out.count_dropped_offscreen, 1);
  assert.equal(out.viewport_filtered, true);
  assert.equal(out.time, "t1"); // other envelope fields preserved
});

test("applyViewport: BACKWARD-COMPAT — absent or invalid bbox returns the SAME object reference, no annotation", () => {
  const payload = { time: "t1", aircraft: [{ id: "a", lon: 0, lat: 0 }], count: 1 };
  for (const bad of [undefined, "", "garbage", "1,2,3", "0,0,0,0", "200,0,201,1"]) {
    const out = applyViewport(payload, bad, "aircraft", (a: any) => [a.lon, a.lat]);
    assert.strictEqual(out, payload, `bbox=${JSON.stringify(bad)} must pass through unchanged`);
    assert.equal(out.viewport_filtered, undefined);
  }
});

test("applyViewport: no array at key (delta/unchanged envelope) is passed through untouched", () => {
  const delta = { unchanged: true, time: "t1", count: 42 };
  const out = applyViewport(delta, "-1,-1,1,1", "aircraft", (a: any) => [a.lon, a.lat]);
  assert.strictEqual(out, delta);
});
