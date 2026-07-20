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
  // [REPAIR 2026-07-20] gemMethaneAssets.ts's oil_gas_extraction.json.gz /
  // coal_mine_tracker.json.gz were added to server/ 2026-07-19 but never
  // added HERE — this exact ratchet existed and did not catch it, because
  // it only pinned the two files present when it was written. Every
  // datacore/gem/*.json.gz repoDataPath() call site is asserted below so
  // the next file added to that directory fails this test until staged,
  // instead of silently degrading to zero matches on prod like this one did.
  for (const gemFile of [
    "datacore/gem/ownership.json.gz", "datacore/gem/methane_emitters.json.gz",
    "datacore/gem/oil_gas_extraction.json.gz", "datacore/gem/coal_mine_tracker.json.gz",
  ]) {
    assert.ok(src.includes(`cp("${gemFile}", "dist/${gemFile}"`),
      `script/build.ts must copy ${gemFile} into dist/ (else its repoDataPath() reader silently empties on prod)`);
  }
  // Generalizes the pin above from a hand-maintained list to every ACTUAL
  // read site: datacore/gem/ also holds coal_finance.json.gz/power_units.json.gz
  // (ingested by the census #5 GEM build, per research/wishlist.md) that no
  // server module reads yet — build.ts is deliberately NOT required to stage
  // those (its own comment states the image should carry only what's actually
  // read: "datacore/ is 254MB ... the runtime set is ~184KB"). So this scans
  // server/*.ts for real `repoDataPath("datacore/...")` call sites instead of
  // the directory listing, and would have caught the exact oil_gas/coal_mine
  // gap by construction, without over-demanding unread files ever get staged.
  const serverDir = path.join(here, "..", "server");
  const repoDataPathCalls = new Set<string>();
  for (const f of fs.readdirSync(serverDir)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const body = fs.readFileSync(path.join(serverDir, f), "utf8");
    for (const m of body.matchAll(/repoDataPath\(\s*["']([^"']+)["']/g)) repoDataPathCalls.add(m[1]);
  }
  assert.ok(repoDataPathCalls.size > 0, "sanity: the scan itself must find at least one repoDataPath() call site");
  for (const rel of repoDataPathCalls) {
    assert.ok(src.includes(`cp("${rel}", "dist/${rel}"`),
      `server/*.ts reads "${rel}" via repoDataPath() at runtime but script/build.ts does not stage it ` +
      "into dist/ — it will silently resolve to empty/missing data on prod (the frozen Dockerfile ships dist/ only)");
  }
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
