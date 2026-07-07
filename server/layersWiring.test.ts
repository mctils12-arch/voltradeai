// layersWiring — [REPAIR R15 2026-07-07] ratchet: a registry layer can
// never ship permanently un-enableable.
//
// The defect class: datamap.tsx marks any live registry id missing from
// its LAYER_GROUP map as "reload to enable" (the honest state for a
// registry NEWER than the bundle, mid-deploy). But when a layer ships
// WITH wiring and WITHOUT a LAYER_GROUP entry, the current bundle
// declares its own layer unknown — a "reload" that no reload fixes
// (powergrid, v1.0.166→177). This test makes that unrepresentable:
// every non-signal, non-planned id in datacore/layers.json must appear
// in LAYER_GROUP, so the panel's unwired detection can only ever fire
// for genuinely newer registries.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

test("RATCHET: every live registry layer is declared in datamap.tsx LAYER_GROUP (no permanent 'reload to enable')", () => {
  const registry = JSON.parse(
    fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  const layers: Array<{ id: string; kind?: string; status?: string }> =
    registry.layers || registry;
  const src = fs.readFileSync(
    path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  const m = src.match(/const LAYER_GROUP: Record<string, string> = \{([\s\S]*?)\};/);
  assert.ok(m, "LAYER_GROUP map not found in datamap.tsx — if it was renamed, update this scrape AND the unwired detection together");
  const wired = new Set(Array.from(m![1].matchAll(/(\w+):\s*"/g)).map((x) => x[1]));
  assert.ok(wired.size >= 20, `LAYER_GROUP scrape looks broken (found ${wired.size} ids)`);

  const stuck = layers
    .filter((l) => l.kind !== "signal" && l.status !== "planned")
    .filter((l) => !wired.has(l.id))
    .map((l) => l.id);
  assert.deepEqual(stuck, [],
    `registry layers missing from datamap.tsx LAYER_GROUP would render PERMANENTLY "reload to enable": ${stuck.join(", ")} — ` +
    "add each to LAYER_GROUP in the same PR that adds its registry entry (R15, powergrid precedent)");
});
