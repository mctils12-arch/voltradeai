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
