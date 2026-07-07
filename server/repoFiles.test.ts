// repoFiles tests — [REPAIR 2026-07-07] regression battery for the
// prod packaging defect: the image ships dist/ only, so runtime disk
// reads of repo datacore/ files silently missed on Railway
// (sentinel2_last_reading:null since that surface shipped; empty
// /api/data/streams on deploy day). These tests pin (1) the resolver's
// working-tree/dist fallback, (2) the build-script copy step, and
// (3) the never-silently-empty inventory payload.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import { fileURLToPath } from "node:url";
import { repoDataPath } from "./repoFiles";
import { buildStreamsInventory } from "./streamsInventory";

const here = path.dirname(fileURLToPath(import.meta.url));

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "repofiles-"));
}

test("resolver prefers the working tree when present", () => {
  const cwd = tmp();
  fs.mkdirSync(path.join(cwd, "datacore", "manifests"), { recursive: true });
  assert.equal(
    repoDataPath(path.join("datacore", "manifests"), cwd),
    path.join(cwd, "datacore", "manifests"),
  );
});

test("resolver falls back to dist/ copy — the production image layout", () => {
  const cwd = tmp();
  fs.mkdirSync(path.join(cwd, "dist", "datacore", "manifests"), { recursive: true });
  assert.equal(
    repoDataPath(path.join("datacore", "manifests"), cwd),
    path.join(cwd, "dist", "datacore", "manifests"),
  );
});

test("resolver returns the direct path when both missing — callers surface it, never mask it", () => {
  const cwd = tmp();
  assert.equal(
    repoDataPath(path.join("datacore", "manifests"), cwd),
    path.join(cwd, "datacore", "manifests"),
  );
});

test("RATCHET: script/build.ts stages the runtime datacore files into dist/", () => {
  // The frozen Dockerfile ships dist/ only; if this copy step is ever
  // removed, prod regresses to empty inventory + null sentinel2
  // freshness with NO other failing signal — pin it at the source.
  const src = fs.readFileSync(path.join(here, "..", "script", "build.ts"), "utf8");
  assert.ok(src.includes('cp("datacore/manifests", "dist/datacore/manifests"'),
    "script/build.ts must copy datacore/manifests into dist/ (prod reads it from there)");
  assert.ok(src.includes('cp("datacore/sentinel2/readings.jsonl"'),
    "script/build.ts must copy sentinel2 readings into dist/ (platformStats freshness)");
});

test("inventory payload is never silently empty: missing manifest dir is flagged as a packaging defect", async () => {
  const emptyBase = tmp();
  const missingManifests = path.join(tmp(), "nope");
  const inv = await buildStreamsInventory(emptyBase, missingManifests, 1);
  assert.equal(inv.count, 0);
  assert.equal(inv.manifest_dir_found, false);
  assert.ok(inv.note && inv.note.includes("packaging defect"), `note must name the defect class, got: ${inv.note}`);
});

test("inventory payload flags manifest_dir_found true on the real repo checkout", async () => {
  const inv = await buildStreamsInventory(tmp(), undefined, 1);
  assert.equal(inv.manifest_dir_found, true);
  assert.ok(inv.count >= 41);
});
