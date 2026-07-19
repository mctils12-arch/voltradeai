// gemMethaneProximity — plume-to-nearest-GEM-asset join (gate-2(a) of the
// GEM METHANE-PLUME × EXTRACTION-REGISTRY PROXIMITY hypothesis,
// research/open_questions.md). Pins the match-radius honesty (unmatched
// beyond MATCH_RADIUS_KM, never a forced guess), the ambiguous-cluster
// flag, the grid index returning the same answer as brute force, and the
// cached-join degrade path when the plume file itself is unavailable.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  haversineKm, buildAssetGrid, joinPlumesToAssets,
  cachedGemMethaneProximity, _resetGemMethaneProximityCacheForTests,
  MATCH_RADIUS_KM,
} from "./gemMethaneProximity";
import { _resetGemMethaneCacheForTests } from "./gemMethane";
import { _resetGemAssetsCacheForTests } from "./gemMethaneAssets";
import type { MethanePlume } from "./gemMethane";
import type { GemAsset } from "./gemMethaneAssets";

function plume(id: string, lat: number, lon: number): MethanePlume {
  return {
    id, name: null, wiki: null, provider: null, instrument: null, observedAt: null,
    emissionsKgHr: null, emissionsUncertaintyKgHr: null, infrastructureType: null,
    infrastructureNotes: null, subnationalUnit: null, country: null, lat, lon,
  };
}

function asset(id: string, kind: GemAsset["kind"], lat: number, lon: number): GemAsset {
  return { id, kind, name: `${id} name`, operator: null, owner: null, parent: null, lat, lon };
}

test("haversineKm: zero distance to itself, ~111km per degree of latitude", () => {
  assert.equal(haversineKm(35, -119, 35, -119), 0);
  const d = haversineKm(0, 0, 1, 0);
  assert.ok(d > 110 && d < 112, `expected ~111km, got ${d}`);
});

test("buildAssetGrid: buckets assets by 0.5° cell", () => {
  const grid = buildAssetGrid([asset("A1", "coal_mine", 35.0, -119.0), asset("A2", "coal_mine", 35.9, -119.9)]);
  assert.equal(grid.size, 2, "two assets a full degree apart land in different cells");
});

test("joinPlumesToAssets: a plume within MATCH_RADIUS_KM matches the nearest asset", () => {
  const p = plume("P1", 35.0, -119.0);
  const near = asset("A1", "coal_mine", 35.005, -119.0); // ~0.55km away
  const far = asset("A2", "oil_gas_extraction", 36.0, -119.0); // far
  const out = joinPlumesToAssets([p], [near, far]);
  assert.equal(out[0].nearestAsset?.id, "A1");
  assert.equal(out[0].nearestAsset?.kind, "coal_mine");
  assert.ok(out[0].nearestAsset!.distanceKm < MATCH_RADIUS_KM);
  assert.equal(out[0].ambiguousMatch, false);
});

test("joinPlumesToAssets: a plume beyond MATCH_RADIUS_KM stays honestly unmatched", () => {
  const p = plume("P1", 35.0, -119.0);
  const distant = asset("A1", "coal_mine", 35.2, -119.0); // ~22km away, well beyond the 2km radius
  const out = joinPlumesToAssets([p], [distant]);
  assert.equal(out[0].nearestAsset, null);
  assert.equal(out[0].ambiguousMatch, false);
});

test("joinPlumesToAssets: no assets at all near the plume degrades to unmatched, never throws", () => {
  const out = joinPlumesToAssets([plume("P1", 35.0, -119.0)], []);
  assert.equal(out[0].nearestAsset, null);
});

test("joinPlumesToAssets: two similarly-close assets flag ambiguousMatch instead of an arbitrary pick", () => {
  const p = plume("P1", 35.0, -119.0);
  const a1 = asset("A1", "coal_mine", 35.005, -119.0);   // ~0.555km
  const a2 = asset("A2", "oil_gas_extraction", 35.0, -119.006); // ~0.546km — within AMBIGUOUS_MARGIN_KM of a1
  const out = joinPlumesToAssets([p], [a1, a2]);
  assert.ok(out[0].nearestAsset, "still reports the nearest one");
  assert.equal(out[0].ambiguousMatch, true);
});

test("joinPlumesToAssets: a plume with null coordinates never matches (nothing to place)", () => {
  const p: MethanePlume = { ...plume("P1", 0, 0), lat: null, lon: null };
  const out = joinPlumesToAssets([p], [asset("A1", "coal_mine", 0, 0)]);
  assert.equal(out[0].nearestAsset, null);
  assert.equal(out[0].ambiguousMatch, false);
});

test("joinPlumesToAssets: grid search matches brute force on a scattered fixture", () => {
  const rng = (() => { let s = 42; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();
  const assets: GemAsset[] = Array.from({ length: 200 }, (_, i) =>
    asset(`A${i}`, i % 2 ? "coal_mine" : "oil_gas_extraction", rng() * 10 - 5, rng() * 10 - 5));
  const plumes: MethanePlume[] = Array.from({ length: 30 }, (_, i) => plume(`P${i}`, rng() * 10 - 5, rng() * 10 - 5));
  const gridResult = joinPlumesToAssets(plumes, assets);
  for (let i = 0; i < plumes.length; i++) {
    const p = plumes[i];
    let bestId: string | null = null, bestKm = Infinity;
    for (const a of assets) {
      const d = haversineKm(p.lat!, p.lon!, a.lat, a.lon);
      if (d < bestKm) { bestKm = d; bestId = a.id; }
    }
    const expectMatch = bestKm <= MATCH_RADIUS_KM;
    assert.equal(gridResult[i].nearestAsset !== null, expectMatch, `plume ${p.id} match/no-match disagrees with brute force`);
    if (expectMatch) assert.equal(gridResult[i].nearestAsset?.id, bestId, `plume ${p.id} nearest-asset id disagrees with brute force`);
  }
});

test("cachedGemMethaneProximity: joins the real repo fixtures and caches across calls", () => {
  _resetGemMethaneProximityCacheForTests();
  _resetGemMethaneCacheForTests();
  _resetGemAssetsCacheForTests();
  try {
    const first = cachedGemMethaneProximity();
    const second = cachedGemMethaneProximity();
    assert.ok(first, "expected the real repo fixtures to load");
    assert.ok(first!.plumes.length > 0);
    assert.ok(first!.matchedCount > 0, "expected at least some real plumes to match a real asset");
    assert.ok(first!.matchedCount <= first!.plumes.length);
    assert.ok(first!.ambiguousCount <= first!.matchedCount);
    assert.equal(first, second, "second call must return the cached object, not re-join");
  } finally {
    _resetGemMethaneProximityCacheForTests();
    _resetGemMethaneCacheForTests();
    _resetGemAssetsCacheForTests();
  }
});
