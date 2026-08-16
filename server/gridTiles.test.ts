// Guard: every committed power-grid tile must be a real PMTiles file (magic
// "PMTiles"), not an MBTiles/SQLite blob. Regression for v1.0.251 where
// power_us.pmtiles shipped as SQLite and the pmtiles:// protocol silently
// rendered nothing — the visual harness missed it (layers default off).
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";

// [REPAIR 2026-08-16] The `files.length >= 50` assertion below was STALE and
// had been failing CI on a clean tree (Q12). It predates the 2026-07-31
// migration that moved every power tile OUT of the repo and into R2 (~830 MB
// off the repo and Docker image, each copy verified byte-identical in the
// bucket before removal). Only the small page-critical non-power tiles are
// committed now, so demanding 50+ asserted a world that deliberately no
// longer exists.
//
// The MAGIC-BYTE GUARD IS NOT WEAKENED — it still runs over every committed
// .pmtiles, which is the actual regression this file exists for (v1.0.251
// shipped power_us.pmtiles as SQLite and pmtiles:// silently rendered
// nothing). What changed is only the obsolete population count: assert the
// directory is non-empty and that each file present is genuinely PMTiles.
// The R2-served tiles are covered separately by the bucket<->registry parity
// test in client/src/lib/gridMaster.test.ts.
test("every client/public/tiles/*.pmtiles has the PMTiles magic", () => {
  const dir = path.join(import.meta.dirname, "..", "client", "public", "tiles");
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".pmtiles"));
  assert.ok(files.length > 0, `no committed .pmtiles found in ${dir}`);
  for (const f of files) {
    const fd = fs.openSync(path.join(dir, f), "r");
    const buf = Buffer.alloc(7);
    fs.readSync(fd, buf, 0, 7, 0);
    fs.closeSync(fd);
    assert.equal(buf.toString("latin1"), "PMTiles", `${f} is not a PMTiles file (got "${buf.toString("latin1")}")`);
  }
});
