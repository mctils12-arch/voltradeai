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

test("weather layer states US-only coverage honestly (Tier-1(b), licensing register 2026-07-04)", () => {
  const w = registry.layers.find((x: any) => x.id === "weather");
  assert.ok(w, "weather layer missing");
  assert.equal(w.kind, "raw");
  assert.ok(w.source.includes("NOAA"), "attribution must name NOAA");
  assert.ok(/US.+only|only.+US/i.test(w.description), "description must state the US-only coverage limit");
});
