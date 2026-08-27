// Layer-registry invariants — every /data map layer must carry the labels the
// RAW-vs-SIGNAL surface rules and DESIGN.md attribution rule depend on.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const registry = JSON.parse(
  fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"),
);

test("every layer carries kind (raw|signal), status, source attribution, and a description", () => {
  assert.ok(Array.isArray(registry.layers) && registry.layers.length >= 8);
  for (const l of registry.layers) {
    assert.ok(l.id && typeof l.id === "string", `layer missing id`);
    assert.ok(["raw", "signal"].includes(l.kind), `${l.id}: kind must be raw|signal`);
    assert.ok(["live", "awaiting_key", "planned"].includes(l.status), `${l.id}: bad status`);
    assert.ok(l.source && l.source.length > 3, `${l.id}: source attribution missing`);
    assert.ok(l.description && l.description.length > 10, `${l.id}: description missing`);
  }
});

test("signal-class layers never ship live before gating (planned until gate 2)", () => {
  for (const l of registry.layers) {
    if (l.kind === "signal") {
      assert.equal(l.status, "planned", `${l.id}: SIGNAL layer must stay planned until ladder gate 2`);
    }
  }
});

test("terrain layer registered with Mapterhorn attribution (Tier-1(a), licensing register 2026-07-04)", () => {
  const t = registry.layers.find((x: any) => x.id === "terrain");
  assert.ok(t, "terrain layer missing");
  assert.equal(t.kind, "raw");
  assert.ok(t.source.includes("Mapterhorn"), "attribution must name Mapterhorn");
});

test("surface-water layer: JRC attribution + static-vintage honesty + opacity inheritance (atlas parity 1)", () => {
  const s = registry.layers.find((x: any) => x.id === "surfacewater");
  assert.ok(s, "surfacewater layer missing");
  assert.equal(s.kind, "raw");
  assert.ok(s.source.includes("JRC"), "attribution must name EC JRC");
  assert.ok(/1984|2021|static/i.test(s.description), "description must state the static 1984–2021 vintage (imagery-date honesty)");
  assert.equal(s.field, true, "atlas rasters inherit the registry opacity slider");
});

test("forest layer: JRC attribution + static-vintage honesty + opacity inheritance (atlas parity 2)", () => {
  const f = registry.layers.find((x: any) => x.id === "forest");
  assert.ok(f, "forest layer missing");
  assert.equal(f.kind, "raw");
  assert.ok(f.source.includes("JRC"), "attribution must name EC JRC");
  assert.ok(/2020/.test(f.description), "description must state the static 2020 vintage (imagery-date honesty)");
  assert.equal(f.field, true, "atlas rasters inherit the registry opacity slider");
});

test("toggle-desync guard pinned: version skew detected, unwired rows disabled, delta cursor cleared on teardown", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("server_version: pkgVersion"), "registry response must carry server_version");
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  assert.ok(page.includes("CLIENT_VERSION"), "client must bake in its build version");
  assert.ok(page.includes("setVersionSkew"), "client must detect registry-vs-bundle skew");
  assert.ok(page.includes("reload the page to enable this new layer"), "unwired rows must say why they cannot toggle");
  assert.ok(page.includes("disabled={!toggleable(l) || unwired}"), "unwired rows must not render a functional-looking toggle");
  assert.ok(page.includes("delete sinceRef.current[id]"), "live-points teardown must clear the delta cursor (remount invisibility bug)");
});

test("legend rule pinned: DESIGN.md carries the approved text; legend renders from the shared registry", () => {
  const design = fs.readFileSync(path.join(here, "..", "DESIGN.md"), "utf8");
  assert.ok(
    design.includes("Every map symbol ships with its legend entry in the same PR, drawn from\nthe shared icon registry"),
    "DESIGN.md must carry the approved legend rule verbatim",
  );
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  assert.ok(page.includes("iconDataURL"), "legend must render registry shapes via iconDataURL");
  assert.ok(page.includes("data-vt-icon"), "legend entries must carry the parity hook");
  // the divergence the rule kills: hand-drawn SVG copies of registry shapes
  assert.ok(!/vt-legend[\s\S]{0,400}<svg/.test(page), "no hand-drawn SVG icon duplicates inside the legend");
});

test("boundaries layer: Natural Earth public domain + generalized-resolution honesty (atlas parity 3)", () => {
  const b = registry.layers.find((x: any) => x.id === "boundaries");
  assert.ok(b, "boundaries layer missing");
  assert.equal(b.kind, "raw");
  assert.ok(/Natural Earth/i.test(b.source), "attribution must name Natural Earth");
  assert.ok(/public domain/i.test(b.source), "source must state public domain");
  assert.ok(/generalized|110m/i.test(b.description), "description must state the generalized resolution");
});

test("boundaries_admin1 layer: Natural Earth public domain + generalized-resolution honesty + route + default-off (sprint W3)", () => {
  const b = registry.layers.find((x: any) => x.id === "boundaries_admin1");
  assert.ok(b, "boundaries_admin1 layer missing");
  assert.equal(b.kind, "raw");
  assert.equal(b.status, "live");
  assert.equal(b.group, "base");
  assert.ok(/Natural Earth/i.test(b.source), "attribution must name Natural Earth");
  assert.ok(/public domain/i.test(b.source), "source must state public domain");
  assert.ok(/generalized|50m/i.test(b.description), "description must state the generalized resolution");
  // artifact exists, is slim, and carries its compile provenance banner
  const art = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "boundaries", "ne_50m_admin1_lines.json"), "utf8"));
  assert.ok(Array.isArray(art.features) && art.features.length > 500, "admin-1 artifact must carry the ~581 line features");
  assert.ok(/Natural Earth 1:50m admin-1/.test(art._doc), "artifact must carry its provenance _doc");
  // served route + client wiring pinned; never defaulted on (state lines are opt-in)
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("/api/data/boundaries_admin1"), "server must serve the admin-1 route");
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  assert.ok(page.includes("enabled.boundaries_admin1"), "client effect must exist");
  assert.ok(!/boundaries_admin1["']?\s*:\s*true/.test(page), "must not be defaulted on");
});

test("celestial_paths layer: computed-ephemeris provenance + reference-not-observation honesty + default-off (sprint W1)", () => {
  const c = registry.layers.find((x: any) => x.id === "celestial_paths");
  assert.ok(c, "celestial_paths layer missing");
  assert.equal(c.kind, "raw");
  assert.equal(c.status, "live");
  assert.equal(c.group, "base");
  assert.ok(/astronomy-engine/.test(c.source), "source must name the ephemeris engine");
  assert.ok(/computed/i.test(c.source), "source must state it is computed, not a feed");
  assert.ok(/not observations/i.test(c.description), "description must carry the reference-lines honesty rail");
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  assert.ok(page.includes("enabled.celestial_paths"), "client toggle effect must exist");
  assert.ok(!/celestial_paths["']?\s*:\s*true/.test(page), "must not be defaulted on");
  // the cartoon sun/moon DOM markers must never return (W1 spec: removed)
  assert.ok(!page.includes("vt-celestial-marker"), "cartoon celestial DOM markers must stay removed");
  assert.ok(page.includes("mountCelestialSky"), "always-on celestial sky must be mounted");
});

test("firetemp layer: GOES-East attribution + irregular-cadence honesty + FIRMS-complement note (G2b)", () => {
  const f = registry.layers.find((x: any) => x.id === "firetemp");
  assert.ok(f, "firetemp layer missing");
  assert.equal(f.kind, "raw");
  assert.ok(f.source.includes("GOES-East"), "attribution must name GOES-East");
  assert.ok(/10-min/i.test(f.description), "description must state the ~10-min sub-daily cadence");
  assert.ok(/FIRMS/.test(f.description), "description must state how this differs from the existing FIRMS layer");
  assert.equal(f.field, true, "raster inherits the registry opacity slider");
});

test("weather layer states US-only coverage honestly (Tier-1(b), licensing register 2026-07-04)", () => {
  const w = registry.layers.find((x: any) => x.id === "weather");
  assert.ok(w, "weather layer missing");
  assert.equal(w.kind, "raw");
  assert.ok(w.source.includes("NOAA"), "attribution must name NOAA");
  assert.ok(/US.+only|only.+US/i.test(w.description), "description must state the US-only coverage limit");
});

test("earthquakes layer: USGS attribution + magnitude-scale honesty (map-layer wiring for usgsQuakes.ts)", () => {
  const q = registry.layers.find((x: any) => x.id === "earthquakes");
  assert.ok(q, "earthquakes layer missing");
  assert.equal(q.kind, "raw");
  assert.equal(q.status, "live");
  assert.ok(q.source.includes("USGS"), "attribution must name USGS");
  assert.ok(/M2\.5/.test(q.description), "description must state the M2.5+ threshold");
  assert.ok(/magnitude/i.test(q.description), "description must state magnitude drives the marker visual");
  assert.ok(/safety-of-life/i.test(q.description), "description must carry the not-for-safety-of-life caveat");
});

test("floodzones layer: FEMA attribution + zero-server-cost honesty + opacity inheritance (location_context_engine.md hazard #3)", () => {
  const f = registry.layers.find((x: any) => x.id === "floodzones");
  assert.ok(f, "floodzones layer missing");
  assert.equal(f.kind, "raw");
  assert.equal(f.status, "live");
  assert.equal(f.group, "hazards");
  assert.ok(f.source.includes("FEMA"), "attribution must name FEMA");
  assert.ok(/zero server cost/i.test(f.description), "description must state the zero-server-cost live-render pattern");
  assert.ok(/never a property risk claim/i.test(f.description), "description must carry the RAW-not-a-risk-claim honesty rail");
  assert.equal(f.field, true, "raster inherits the registry opacity slider");
});

test("buoys layer: NDBC attribution + no-fabricated-zero honesty (map-layer wiring for ndbcBuoys.ts)", () => {
  const b = registry.layers.find((x: any) => x.id === "buoys");
  assert.ok(b, "buoys layer missing");
  assert.equal(b.kind, "raw");
  assert.equal(b.status, "live");
  assert.ok(b.source.includes("National Data Buoy Center"), "attribution must name the National Data Buoy Center");
  assert.ok(/no.?data|missing/i.test(b.description), "description must state missing sensors are never coerced to zero");
});

test("seafloor layer: ETOPO1/NOAA attribution + interpolation & not-for-navigation honesty + opacity inheritance (EARTH TWIN E2-1)", () => {
  const s = registry.layers.find((x: any) => x.id === "seafloor");
  assert.ok(s, "seafloor layer missing");
  assert.equal(s.kind, "raw");
  assert.equal(s.status, "live");
  assert.ok(/ETOPO1/.test(s.source) && /NOAA/.test(s.source), "attribution must name NOAA ETOPO1");
  assert.ok(/interpolation/i.test(s.description), "description must state the soundings + satellite-gravity interpolation blend");
  assert.ok(/never navigational|not.{0,10}navigation/i.test(s.description), "description must carry the not-for-navigation caveat");
  assert.equal(s.field, true, "depth raster inherits the registry opacity slider");
  assert.equal(s.altitudeRef, "depth", "v2: z means depth for this layer");
});

test("seafloor_confidence layer: GEBCO TID attribution + measured-vs-predicted honesty + regional-coverage statement (EARTH TWIN E2 v2)", () => {
  const s = registry.layers.find((x: any) => x.id === "seafloor_confidence");
  assert.ok(s, "seafloor_confidence layer missing");
  assert.equal(s.kind, "raw");
  assert.equal(s.status, "live", "datamap wiring slice shipped (EARTH TWIN E2 v2 wiring session)");
  assert.ok(/GEBCO/.test(s.source) && /TID/.test(s.source), "attribution must name the GEBCO TID grid");
  assert.ok(/direct measurements/i.test(s.description), "description must name GEBCO's direct-measurement class");
  assert.ok(/predict|indirect/i.test(s.description), "description must name the predicted/indirect class");
  assert.ok(/regional|region/i.test(s.provenance.coverage), "coverage must state the regional demo honestly");
  assert.equal(s.altitudeRef, "depth");
  assert.equal(s.provenance.commercialOk, true, "GEBCO terms: public domain, commercial use allowed");
  assert.equal(s.field, true, "confidence raster inherits the registry opacity slider");
});

// ── REGISTRY v2 (EARTH TWIN E0-1, research/earth_twin_program.md A2) ──
// altitudeRef/time/provenance stay OPTIONAL and additive: entries that omit
// them stay valid forever; entries that carry them must carry them
// WELL-FORMED so the LOD director, global time axis, and license tripwire
// can trust what they read. renderKind + lod are the same "well-formed when
// present" story below, PLUS a Track 4 pinned-gap test further down that
// starts tracking their eventual graduation to required (see that test's
// own comment). Extending a vocabulary below is a deliberate, reviewed act:
// add the value here and in layers.json's _doc in the same PR.

const V2_ALTITUDE_REFS = new Set(["surface", "agl", "msl", "orbit", "depth", "underground"]);
const V2_TIME_MODES = new Set(["live", "dated-daily", "dated-subdaily", "archive", "static"]);
const V2_CONFIDENCES = new Set(["verified", "derived", "estimated", "inferred", "placeholder"]);
const V2_RENDER_KINDS = new Set(["point-symbol", "raster-field", "vector", "track", "grid", "custom"]);

test("registry v2: altitudeRef, when present, is a known vertical datum", () => {
  for (const l of registry.layers) {
    if (!("altitudeRef" in l)) continue;
    assert.ok(V2_ALTITUDE_REFS.has(l.altitudeRef),
      `${l.id}: altitudeRef must be one of ${[...V2_ALTITUDE_REFS].join("|")} (got ${l.altitudeRef})`);
  }
});

test("registry v2: time block, when present, is well-formed for the global time axis", () => {
  for (const l of registry.layers) {
    if (!("time" in l)) continue;
    const t = l.time;
    assert.ok(t && typeof t === "object" && !Array.isArray(t), `${l.id}: time must be an object`);
    assert.ok(V2_TIME_MODES.has(t.mode),
      `${l.id}: time.mode must be one of ${[...V2_TIME_MODES].join("|")} (got ${t.mode})`);
    if ("latencyDays" in t) {
      assert.ok(typeof t.latencyDays === "number" && t.latencyDays >= 0,
        `${l.id}: time.latencyDays must be a number >= 0`);
    }
    if ("historyStart" in t) {
      assert.ok(typeof t.historyStart === "string" && /^\d{4}-\d{2}(-\d{2})?$/.test(t.historyStart),
        `${l.id}: time.historyStart must be YYYY-MM or YYYY-MM-DD (the honest archive floor)`);
    }
  }
});

test("registry v2: lod block, when present, is a sane camera-altitude envelope", () => {
  for (const l of registry.layers) {
    if (!("lod" in l)) continue;
    const lod = l.lod;
    assert.ok(lod && typeof lod === "object" && !Array.isArray(lod), `${l.id}: lod must be an object`);
    for (const k of ["camMinKm", "camMaxKm", "fadeBandKm"]) {
      if (k in lod) assert.ok(typeof lod[k] === "number" && lod[k] >= 0,
        `${l.id}: lod.${k} must be a number >= 0`);
    }
    assert.ok(("camMinKm" in lod) || ("camMaxKm" in lod),
      `${l.id}: an lod block without camMinKm or camMaxKm gates nothing — remove it or bound it`);
    if ("camMinKm" in lod && "camMaxKm" in lod) {
      assert.ok(lod.camMinKm < lod.camMaxKm, `${l.id}: lod.camMinKm must be < lod.camMaxKm`);
    }
    if ("ramp" in lod) {
      assert.ok(Array.isArray(lod.ramp) && lod.ramp.every((s: unknown) => typeof s === "string"),
        `${l.id}: lod.ramp must be an array of stage names`);
    }
  }
});

test("registry v2: provenance block, when present, is well-formed (confidence tier + string facts)", () => {
  for (const l of registry.layers) {
    if (!("provenance" in l)) continue;
    const p = l.provenance;
    assert.ok(p && typeof p === "object" && !Array.isArray(p), `${l.id}: provenance must be an object`);
    if ("confidence" in p) {
      assert.ok(V2_CONFIDENCES.has(p.confidence),
        `${l.id}: provenance.confidence must be one of ${[...V2_CONFIDENCES].join("|")} (got ${p.confidence})`);
    }
    for (const k of ["license", "updateFreq", "resolution", "coverage"]) {
      if (k in p) assert.equal(typeof p[k], "string", `${l.id}: provenance.${k} must be a string`);
    }
    if ("commercialOk" in p) {
      assert.equal(typeof p.commercialOk, "boolean", `${l.id}: provenance.commercialOk must be boolean`);
    }
  }
});

test("registry v2: renderKind, when present, names a known engine", () => {
  for (const l of registry.layers) {
    if (!("renderKind" in l)) continue;
    assert.ok(V2_RENDER_KINDS.has(l.renderKind),
      `${l.id}: renderKind must be one of ${[...V2_RENDER_KINDS].join("|")} (got ${l.renderKind})`);
  }
});

test("LICENSE RATCHET: no layer ships with a declared non-commercial license (monetization tripwire, machine-checked)", () => {
  const violations = registry.layers
    .filter((l: any) => l.provenance && l.provenance.commercialOk === false)
    .map((l: any) => l.id);
  assert.deepEqual(violations, [],
    `layers declaring provenance.commercialOk=false may never ship (EARTH TWIN charter: NC data is a ` +
    `guaranteed rip-out at billing activation): ${violations.join(", ")} — use the build-first alternative ` +
    `(OSM/ODbL, US-gov public domain) or file a paid-license wishlist entry instead`);
});

// ── REGISTRY v2 TRACK 4 (T4.1, research/PROGRAM_STATE.md Q11 / earth_twin_
// program.md A2) — renderKind + lod start graduating out of pure-optional.
// Requiring them on the full, already-249-layer registry today would mean
// backfilling metadata for every pre-existing layer in one PR — exactly the
// "big-bang rewrite" A2 forbids ("MIGRATION IS OPPORTUNISTIC ... never a
// big-bang rewrite of 113 layers ... every layer TOUCHED afterward migrates
// in that PR"). So the requirement is enforced as a PINNED COUNT instead of
// a per-layer assertion: the gap is real, tracked, and can only move if a
// PR consciously updates the pin — silent drift in either direction fails.
test("registry v2 Track 4 (T4.1): renderKind + lod required — the migration gap is pinned, not silent", () => {
  const missing = registry.layers.filter((l) => !("renderKind" in l) || !("lod" in l));
  const PINNED_GAP = 248; // lower this in the SAME PR that migrates a layer; a rise means a new/edited layer shipped without the v2 fields it should now carry
  assert.equal(
    missing.length,
    PINNED_GAP,
    `${missing.length}/${registry.layers.length} layers are missing renderKind and/or lod (pinned at ${PINNED_GAP}). ` +
    `Per EARTH TWIN A2 this is an opportunistic migration, not a rewrite — every layer touched from here on should ` +
    `gain both fields in that same PR, lowering this pin. First few offenders: ` +
    `${missing.slice(0, 8).map((l) => l.id).join(", ")}${missing.length > 8 ? ", ..." : ""}`,
  );
});

test("registry v2 exemplars: the five annotated layers keep their contract (E0-1 ships wired, not vacuous)", () => {
  const byId = new Map(registry.layers.map((l: any) => [l.id, l]));
  const sats: any = byId.get("orbital_sats");
  assert.equal(sats?.altitudeRef, "orbit", "orbital_sats must declare altitudeRef=orbit");
  assert.ok(sats?.lod && typeof sats.lod.camMinKm === "number",
    "orbital_sats must carry the LOD camera-altitude envelope (the directive's zoom-gated satellites)");
  assert.equal(byId.get("terrain")?.time?.mode, "static");
  assert.equal(byId.get("aircraft")?.renderKind, "point-symbol");
  assert.equal(byId.get("nightlights")?.time?.mode, "dated-daily");
  const smap: any = byId.get("soilmoisture");
  assert.ok(smap?.time?.latencyDays >= 5,
    "soilmoisture must declare its ~6-day processing lag so the time axis defaults honestly");
});
