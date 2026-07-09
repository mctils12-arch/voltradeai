// Guard: every committed power-grid tile must be a real PMTiles file (magic
// "PMTiles"), not an MBTiles/SQLite blob. Regression for v1.0.251 where
// power_us.pmtiles shipped as SQLite and the pmtiles:// protocol silently
// rendered nothing — the visual harness missed it (layers default off).
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";

test("every client/public/tiles/*.pmtiles has the PMTiles magic", () => {
  const dir = path.join(import.meta.dirname, "..", "client", "public", "tiles");
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".pmtiles"));
  assert.ok(files.length >= 50, `expected the state+national tiles, found ${files.length}`);
  for (const f of files) {
    const fd = fs.openSync(path.join(dir, f), "r");
    const buf = Buffer.alloc(7);
    fs.readSync(fd, buf, 0, 7, 0);
    fs.closeSync(fd);
    assert.equal(buf.toString("latin1"), "PMTiles", `${f} is not a PMTiles file (got "${buf.toString("latin1")}")`);
  }
});
