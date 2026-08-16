// Guard: every committed power-grid tile must be a real PMTiles file (magic
// "PMTiles"), not an MBTiles/SQLite blob. Regression for v1.0.251 where
// power_us.pmtiles shipped as SQLite and the pmtiles:// protocol silently
// rendered nothing — the visual harness missed it (layers default off).
//
// SPLIT 2026-08-14 (Q12). This file used to open with
// `assert.ok(files.length >= 50, "expected the state+national tiles")`, which
// failed on every run — three .pmtiles are committed and `git log --all` finds
// no `power_*.pmtiles` EVER committed. Because that assertion ran FIRST, the
// magic-byte loop below — the entire reason this file exists — never executed.
// **A guard against a real, previously-shipped defect had been dead since the
// day it was written.**
//
// The corpus assertion is not deleted or weakened: it moves verbatim to
// server/gridTilesCoverage.test.ts, which is the quarantined entry. That file
// carries the reason it cannot currently pass — scripts/build_power_tiles.sh
// line 53 says of exactly these artifacts:
//
//     US scale (est. 60-150MB): DO NOT commit — boot-fetch from a
//     GitHub Release asset into the volume, serve from there.
//
// So the >=50 assertion demands a state the repo's own build script forbids.
// Splitting lets THIS guard gate the build (it passes, and always would have)
// while that one stays honestly quarantined pending the boot-fetch design.
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";

const TILE_DIR = path.join(import.meta.dirname, "..", "client", "public", "tiles");

test("every client/public/tiles/*.pmtiles has the PMTiles magic", () => {
  const files = fs.readdirSync(TILE_DIR).filter((f) => f.endsWith(".pmtiles"));

  // Not vacuous: with zero committed tiles the loop below would pass trivially
  // and this file would go back to guarding nothing — a quieter version of the
  // same failure the split just fixed.
  assert.ok(
    files.length >= 1,
    `no .pmtiles committed under client/public/tiles — the magic-byte guard ` +
    `would pass vacuously. If the tiles genuinely moved to boot-fetch, this ` +
    `guard must move with them, not silently stop checking.`,
  );

  for (const f of files) {
    const fd = fs.openSync(path.join(TILE_DIR, f), "r");
    const buf = Buffer.alloc(7);
    fs.readSync(fd, buf, 0, 7, 0);
    fs.closeSync(fd);
    assert.equal(buf.toString("latin1"), "PMTiles", `${f} is not a PMTiles file (got "${buf.toString("latin1")}")`);
  }
});
