// aeroCharts.test.ts — chart base-view state, persistence, registry merge,
// MapLibre builders and the honesty of the freshness badge.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  AERO_CHART_DEFAULTS, AERO_OPACITY_MIN, AERO_VIEW_DEFAULT, AERO_VIEW_PREF_KEY, aeroBadge, aeroPaint,
  aeroSourceSpec, aeroViewReducer, beforeIdAbove, clampOpacity, mergeAeroMeta, readAeroViewPref, writeAeroViewPref,
} from "./aeroCharts.ts";
import { RASTER_FADE_MS, rasterFadeViolations } from "../render/mapBaseConfig.ts";

function memStore(init: Record<string, string> = {}) {
  const m = new Map(Object.entries(init));
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => { m.set(k, v); }, m };
}

test("reducer: view switch, opacity clamp, no-op identity, reset", () => {
  const s1 = aeroViewReducer(AERO_VIEW_DEFAULT, { type: "setView", view: "sectional" });
  assert.equal(s1.view, "sectional");
  assert.equal(aeroViewReducer(s1, { type: "setView", view: "sectional" }), s1, "same view = same object (no re-render)");
  assert.equal(aeroViewReducer(s1, { type: "setView", view: "bogus" as never }), s1);
  assert.equal(aeroViewReducer(s1, { type: "setOpacity", opacity: 55.4 }).opacity, 55);
  assert.equal(aeroViewReducer(s1, { type: "setOpacity", opacity: 0 }).opacity, AERO_OPACITY_MIN,
    "never fully transparent — a chart view you cannot see is a bug report");
  assert.equal(aeroViewReducer(s1, { type: "setOpacity", opacity: 400 }).opacity, 100);
  assert.equal(clampOpacity(Number.NaN), 100);
  assert.deepEqual(aeroViewReducer(s1, { type: "reset" }), AERO_VIEW_DEFAULT);
});

test("persistence: round-trips; corrupt / hostile / throwing storage falls back to the default", () => {
  const st = memStore();
  assert.ok(writeAeroViewPref({ view: "ifrhigh", opacity: 60 }, st));
  assert.deepEqual(readAeroViewPref(st), { view: "ifrhigh", opacity: 60 });
  assert.deepEqual(readAeroViewPref(memStore({ [AERO_VIEW_PREF_KEY]: "{not json" })), AERO_VIEW_DEFAULT);
  assert.deepEqual(readAeroViewPref(memStore({ [AERO_VIEW_PREF_KEY]: '{"view":"<script>","opacity":"x"}' })), AERO_VIEW_DEFAULT);
  const throwing = { getItem: () => { throw new Error("SecurityError"); }, setItem: () => { throw new Error("QuotaExceeded"); } };
  assert.deepEqual(readAeroViewPref(throwing), AERO_VIEW_DEFAULT);
  assert.equal(writeAeroViewPref(AERO_VIEW_DEFAULT, throwing), false);
  assert.deepEqual(readAeroViewPref(null), AERO_VIEW_DEFAULT);
});

test("registry merge: server editions land; a tile URL off our origin is refused (Law II.8)", () => {
  const m = mergeAeroMeta([
    { id: "sectional", edition: "2026-07-09", effective: "2026-07-09", expires: "2026-09-03", expired: true,
      behindCurrentCycle: true, minzoom: 8, maxzoom: 12, tiles: "/tiles/aero/sectional/{z}/{x}/{y}?e=2026-07-09" },
    { id: "ifrhigh", tiles: "https://tiles.arcgis.com/evil/{z}/{y}/{x}", minzoom: "5" },
    { id: "satellite" }, null, { id: "nope" },
  ]);
  assert.equal(m.sectional.edition, "2026-07-09");
  assert.equal(m.sectional.tiles, "/tiles/aero/sectional/{z}/{x}/{y}?e=2026-07-09");
  assert.equal(m.ifrhigh.tiles, AERO_CHART_DEFAULTS.ifrhigh.tiles, "upstream URL never reaches the map");
  assert.equal(m.ifrhigh.minzoom, 5);
  assert.deepEqual(mergeAeroMeta(undefined), AERO_CHART_DEFAULTS);
  for (const meta of Object.values(AERO_CHART_DEFAULTS)) assert.match(meta.tiles, /^\/tiles\/aero\//);
});

test("source + paint: our origin, real LOD band (overzoom past max), shared Law II crossfade", () => {
  const src = aeroSourceSpec(AERO_CHART_DEFAULTS.ifrhigh);
  assert.deepEqual(src.tiles, ["/tiles/aero/ifrhigh/{z}/{x}/{y}"]);
  assert.equal(src.maxzoom, 9);
  assert.equal(src.tileSize, 256, "the FAA caches are 256px — declaring 512 would request a coarser level");
  assert.match(src.attribution, /NOT FOR NAVIGATION/);
  const p = aeroPaint(40);
  assert.equal(p["raster-opacity"], 0.4);
  assert.equal(p["raster-fade-duration"], RASTER_FADE_MS);
  assert.deepEqual(rasterFadeViolations(p), []);
});

test("layer order: the chart sits directly above the imagery, under every data layer", () => {
  assert.equal(beforeIdAbove(["bg", "blackmarble", "imagery", "hillshade", "aircraft"], "imagery"), "hillshade");
  assert.equal(beforeIdAbove(["bg", "imagery"], "imagery"), undefined);
  assert.equal(beforeIdAbove(["bg"], "imagery"), undefined);
});

test("badge: current, superseded/expired, and unverified editions are each stated honestly", () => {
  const now = Date.parse("2026-09-30T12:00:00Z");
  const cur = aeroBadge({ ...AERO_CHART_DEFAULTS.sectional, edition: "2026-09-03", effective: "2026-09-03", expires: "2026-10-29" }, now);
  assert.equal(cur.tone, "ok");
  assert.match(cur.edition, /Sep 3, 2026 – Oct 29, 2026/);
  const old = aeroBadge({ ...AERO_CHART_DEFAULTS.sectional, edition: "2026-07-09", effective: "2026-07-09",
    expires: "2026-09-03", expired: true, behindCurrentCycle: true }, now);
  assert.equal(old.tone, "warn");
  assert.match(old.edition, /expired/);
  // expiry is re-checked client-side even if the server flag is stale
  const lapsed = aeroBadge({ ...AERO_CHART_DEFAULTS.tac, edition: "2026-07-09", effective: "2026-07-09", expires: "2026-09-03" }, now);
  assert.equal(lapsed.tone, "warn");
  const unk = aeroBadge(AERO_CHART_DEFAULTS.ifrlow, now);
  assert.equal(unk.tone, "unknown");
  assert.match(unk.edition, /unverified/);
  assert.match(unk.coverage, /zoom 7–12/);
});
