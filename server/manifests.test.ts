// Universal archive envelope — enforcement (approved 2026-07-04, wishlist
// UNIVERSAL ARCHIVE ENVELOPE proposal, PR 1). Every archived stream carries a
// dataset-level manifest in datacore/manifests/; a NEW stream without one
// fails here, mechanically.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const MANIFEST_DIR = path.join(here, "..", "datacore", "manifests");

const REQUIRED = [
  "stream", "storage", "written_by", "source", "license", "attribution",
  "schema_version", "field_map", "confidence_model", "geo_fields",
  "entity_key", "started", "cadence",
];

function manifests(): Record<string, any> {
  const out: Record<string, any> = {};
  for (const f of fs.readdirSync(MANIFEST_DIR)) {
    if (!f.endsWith(".json")) continue;
    out[f.replace(/\.json$/, "")] = JSON.parse(fs.readFileSync(path.join(MANIFEST_DIR, f), "utf8"));
  }
  return out;
}

test("every manifest carries the full envelope", () => {
  const m = manifests();
  assert.ok(Object.keys(m).length >= 7, "expected at least the 7 streams manifested at adoption");
  for (const [name, doc] of Object.entries(m)) {
    for (const k of REQUIRED) {
      assert.ok(k in (doc as any), `manifest '${name}' missing envelope field '${k}'`);
    }
    assert.equal((doc as any).stream, name, `manifest '${name}' stream field must match filename`);
    assert.ok(Object.keys((doc as any).field_map).length >= 3, `manifest '${name}' field_map too thin`);
  }
});

test("FORWARD ENFORCEMENT: every archive directory referenced in server code has a manifest", () => {
  // Any module archiving under archiveBaseDir() must name its dir like
  // path.join(<base>, "name") — scrape those names and require manifests.
  const m = manifests();
  const dirs = new Set<string>();
  for (const f of fs.readdirSync(here)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const src = fs.readFileSync(path.join(here, f), "utf8");
    // matches: path.join(base, "aircraft") / path.join(baseDir || archiveBaseDir(), "fires") etc.
    for (const g of src.matchAll(/path\.join\([^)]*archiveBaseDir\(\)[^)]*,\s*"([a-z0-9_]+)"/g)) dirs.add(g[1]);
    for (const g of src.matchAll(/path\.join\(base(?:Dir)?,\s*"([a-z0-9_]+)"/g)) dirs.add(g[1]);
  }
  // track-derivative dirs are recomputable views over a manifested stream,
  // not independent streams (archive-ingredients doctrine)
  const DERIVED = new Set(["aircraft_tracks", "vessels_tracks", "trains_tracks"]);
  const missing = [...dirs].filter((d) => !DERIVED.has(d) && !m[d]);
  assert.deepEqual(missing, [],
    `archive dirs without manifests: ${missing.join(", ")} — new streams must ship datacore/manifests/<stream>.json (approved envelope, 2026-07-04)`);
  // Position streams (aircraft/vessels/trains) build their dirs from the
  // ArchiveKind VARIABLE, not literals — they're enforced by the union test
  // below. The literal scrape covers event-stream modules.
  assert.ok(dirs.size >= 4, `dir scrape looks broken (found only: ${[...dirs].join(", ")})`);
});

test("ArchiveKind union members are all manifested (position streams)", () => {
  const src = fs.readFileSync(path.join(here, "datacoreArchive.ts"), "utf8");
  const um = src.match(/export type ArchiveKind = ([^;]+);/);
  assert.ok(um, "ArchiveKind union not found");
  const kinds = [...um![1].matchAll(/"([a-z0-9_]+)"/g)].map((x) => x[1]);
  const m = manifests();
  for (const k of kinds) assert.ok(m[k], `ArchiveKind '${k}' has no manifest`);
});
