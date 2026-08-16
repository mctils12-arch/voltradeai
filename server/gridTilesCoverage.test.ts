// THE STATE+NATIONAL+WORLD POWER-TILE CORPUS (Q12, then de-quarantined
// 2026-08-16). Originally split out of server/gridTiles.test.ts on
// 2026-08-14 asserting `client/public/tiles` held >=50 committed
// `power_*.pmtiles` files — quarantined because it never could pass:
// scripts/build_power_tiles.sh forbids committing US-scale tiles, and its
// ODbL note forbids offering the .pmtiles file itself for download.
//
// THE CORPUS WAS NEVER MISSING — it was asserted in the wrong place. The
// 2026-07-31 "GRID VISION world rollout" migration (server/routes.ts's
// `/tiles-r2/:name` passthrough, datamap.tsx's `tilePath()`) moved every
// power-grid tile — US master, all 50 states + DC, and dozens of non-US
// countries — off the repo/Docker image and onto a public R2 bucket,
// streamed same-origin through the proxy (~830MB removed from the repo,
// per datamap.tsx's own migration comment). `research/wishlist.md`'s
// DATACORE MAXIMUS resume block still read "US-full: NOT built" because
// it was last updated 2026-07-07, three weeks before the R2 migration
// shipped the exact capability it described (under a different program,
// GRID VISION, and a different serving design than the block's own
// GitHub-Release-boot-fetch plan) — corrected in the same commit as this
// file. A scheduled session nearly re-built and re-downloaded ~12GB of
// already-shipped tiles on the strength of that stale doc; this file's
// original premise (files under client/public/tiles) was the same stale
// assumption, just encoded as a test instead of a wishlist paragraph.
//
// So: re-pointed at what the server actually serves (repo option (a) from
// the prior quarantine note), not deleted — the feature was never
// cancelled, it shipped via R2 instead of the local volume.
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.join(import.meta.dirname, "..");

test("datamap.tsx tilePath() routes every power_*.pmtiles reference through the R2 proxy, never the local /tiles path", () => {
  const src = fs.readFileSync(path.join(ROOT, "client", "src", "pages", "datamap.tsx"), "utf8");
  assert.match(
    src,
    /const tilePath = \(file: string\): string =>\s*\n\s*file\.startsWith\("power"\) \? `\/tiles-r2\/\$\{file\}` : `\/tiles\/\$\{file\}`;/,
    "tilePath()'s power-vs-local routing split not found in its expected shape — if the R2 serving design changed, re-verify " +
    "the corpus still isn't meant to be committed under client/public/tiles before updating this scrape",
  );
});

test("the state+national+world power-grid tile corpus is referenced in datamap.tsx (R2-served, not locally committed)", () => {
  const src = fs.readFileSync(path.join(ROOT, "client", "src", "pages", "datamap.tsx"), "utf8");
  const files = new Set(Array.from(src.matchAll(/power_[a-z_]+\.pmtiles/g)).map((m) => m[0]));
  assert.ok(
    files.size >= 50,
    `expected the state+national+world power tile corpus (>=50 distinct files, the same bar the original ` +
    `committed-file assertion used), found only ${files.size} referenced in datamap.tsx`,
  );
  // The master US file and at least one real state file must both be present —
  // a regex that only matched country codes (or only the master) would still
  // clear >=50 without actually covering the "all 50 states + DC" claim.
  assert.ok(files.has("power_us.pmtiles"), "master US power-grid tile (power_us.pmtiles) not referenced");
  assert.ok(files.has("power_ak.pmtiles") || files.has("power_tx.pmtiles"), "no per-state power tile file found");
});

test("server /tiles-r2/:name proxy exists and its allowlist permits power_*.pmtiles names", () => {
  const src = fs.readFileSync(path.join(ROOT, "server", "routes.ts"), "utf8");
  assert.match(src, /app\.get\("\/tiles-r2\/:name"/, "R2 tile passthrough route not found");
  const m = src.match(/if \(!(\/\^\[a-z0-9_\]\+\\\.pmtiles\$\/)\.test\(name\)\)/);
  assert.ok(m, "R2 proxy name-allowlist regex not found in its expected shape");
  const allowlist = new RegExp(m[1].slice(1, -1));
  assert.ok(allowlist.test("power_us.pmtiles"), "R2 proxy allowlist must permit power_us.pmtiles");
  assert.ok(!allowlist.test("../../etc/passwd"), "R2 proxy allowlist must reject path traversal");
});

interface LayerEntry {
  id: string;
  status?: string;
  description?: string;
}

test("datacore/layers.json powergrid entry is live and honestly claims full US coverage", () => {
  const registry = JSON.parse(fs.readFileSync(path.join(ROOT, "datacore", "layers.json"), "utf8"));
  const layers: LayerEntry[] = registry.layers || registry;
  const layer = layers.find((l) => l.id === "powergrid");
  assert.ok(layer, "powergrid layer entry missing from datacore/layers.json");
  assert.equal(layer.status, "live", "powergrid registry entry should report live status — the R2-served corpus is deployed");
  assert.match(layer.description, /50 states/i, "powergrid description should honestly state its all-50-states coverage");
});
