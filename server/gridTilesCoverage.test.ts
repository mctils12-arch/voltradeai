// THE STATE+NATIONAL POWER-TILE CORPUS (Q12). Split out of
// server/gridTiles.test.ts on 2026-08-14 — this is the assertion that file
// opened with, moved here VERBATIM rather than deleted or weakened.
//
// WHY IT WAS SPLIT. It fails on every run: three .pmtiles are committed and
// `git log --all` finds no `power_*.pmtiles` ever committed. Because it ran
// first, it prevented the magic-byte loop in gridTiles.test.ts from executing
// at all — so a guard against a REAL previously-shipped defect (v1.0.251, where
// power_us.pmtiles shipped as SQLite and the pmtiles:// protocol silently
// rendered nothing) was dead from the day it was written. Splitting revives
// that guard while keeping this claim intact and honestly quarantined.
//
// WHY IT CANNOT SIMPLY BE MADE TO PASS. The obvious fix — build the tiles and
// commit them — is forbidden by the repo's own build script.
// scripts/build_power_tiles.sh line 53, of exactly these artifacts:
//
//     TX pilot (16MB): commit under client/public/tiles/ -> bundled to
//     dist/public, served with range requests by express static.
//     US scale (est. 60-150MB): DO NOT commit — boot-fetch from a
//     GitHub Release asset into the volume, serve from there.
//
// and the same file's ODbL note: "do NOT offer the .pmtiles file itself for
// download (share-alike boundary)". wishlist.md A4 PHASE 2 item 2 records the
// intended design as boot-fetch-from-Release, and marks it "filed above, NOT
// built".
//
// So this assertion asks for a committed corpus that the architecture
// deliberately does not commit. It is not wrong to WANT the coverage — the
// grid layer genuinely needs those tiles to exist somewhere — but asserting
// them in `client/public/tiles` is asserting the one place the design says
// they must not be.
//
// HOW TO RESOLVE IT — one of these, not a weakening of the assertion:
//   (a) build the boot-fetch path (A4 PHASE 2 item 2), publish the tiles as a
//       Release asset, and re-point this test at the manifest/volume the
//       server actually serves from; or
//   (b) if the decision is that state-level tiles are not being built at all,
//       delete this test WITH that decision recorded in experiments.md —
//       deleting a test because the feature was cancelled is legitimate;
//       deleting it because it is red is not.
//
// Quarantined in ci/quarantine.txt, review by 2026-09-13. It still RUNS in the
// gate's non-blocking pass, which prints "NOW PASSING — remove from
// ci/quarantine.txt" the day the corpus appears.
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";

test("the state+national power tiles are present under client/public/tiles", () => {
  const dir = path.join(import.meta.dirname, "..", "client", "public", "tiles");
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".pmtiles"));
  assert.ok(files.length >= 50, `expected the state+national tiles, found ${files.length}`);
});
