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
