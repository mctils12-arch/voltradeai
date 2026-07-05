// OOM ratchet (Priority-1 postmortem 2026-07-05): archive analytics must
// FOLD (bounded state) rather than SLURP (materialize every point). Prod
// crash-looped every ~61s with "JavaScript heap out of memory" once the
// vessel archive outgrew what readVesselTracksAsync's Map-of-everything
// could hold under the 512MB cap.
//
// Two ratchets:
//  1. BEHAVIORAL: fold a synthetic multi-hundred-thousand-point archive and
//     assert the retained heap delta stays far below what materializing
//     the same points costs (measured: slurp ~8-10x the fold's footprint).
//  2. SOURCE: no runtime module may call the materializing readers
//     (readVesselTracks/readVesselTracksAsync) outside tests — they exist
//     only so the equivalence ratchets can pin the online folds.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import { computeShadowStatsAsync } from "./shadowFleet";
import { computePortDwellAsync } from "./portDwell";

const SERVER = path.dirname(fileURLToPath(import.meta.url));

function buildSyntheticArchive(base: string, nowMs: number, hours: number,
                               vessels: number, ptsPerVesselPerHour: number): number {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  let total = 0;
  for (let h = hours; h >= 1; h--) {
    const stamp = new Date(nowMs - h * 3600_000);
    const name = `${stamp.toISOString().slice(0, 10)}-${String(stamp.getUTCHours()).padStart(2, "0")}.jsonl`;
    const lines: string[] = [];
    for (let v = 0; v < vessels; v++) {
      for (let k = 0; k < ptsPerVesselPerHour; k++) {
        const t = Math.floor((nowMs - h * 3600_000) / 1000) + k * Math.floor(3600 / ptsPerVesselPerHour);
        lines.push(JSON.stringify({
          i: `MMSI${v}`, t,
          la: 33 + (v % 50) * 0.01 + k * 0.0001,
          lo: -118 - (v % 50) * 0.01 - k * 0.0001,
          v: (v + k) % 14, c: `VESSEL ${v}`,
        }));
        total++;
      }
    }
    // mix plain and gz like the real archive
    if (h % 2 === 0) fs.writeFileSync(path.join(dir, `${name}.gz`), zlib.gzipSync(lines.join("\n") + "\n"));
    else fs.writeFileSync(path.join(dir, name), lines.join("\n") + "\n");
  }
  return total;
}

test("archive folds retain bounded heap on a 300K-point archive (slurp would not)", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "foldmem-"));
  const now = Date.parse("2026-07-05T12:00:00Z");
  // 12h x 500 vessels x 50 pts = 300,000 points (~35MB of JSONL)
  const total = buildSyntheticArchive(base, now, 12, 500, 50);
  assert.equal(total, 300_000);

  global.gc?.(); // when run with --expose-gc; harmless otherwise
  const before = process.memoryUsage().heapUsed;
  const stats = await computeShadowStatsAsync(
    [{ id: "z1", name: "Test Zone", lat: 33.2, lon: -118.2, radius_km: 30 }],
    72, base, now);
  const dwell = await computePortDwellAsync(
    [{ id: "p1", name: "Test Port", lat: 33.2, lon: -118.2, radius_km: 20 }],
    168, base, now);
  const deltaMb = (process.memoryUsage().heapUsed - before) / 1048576;

  assert.equal(stats.points_read, 300_000, "every point was actually visited");
  assert.equal(stats.vessels_seen, 500);
  assert.equal(dwell.vessels_seen, 500);
  // Materializing 300K Pt objects into a Map measures >40MB retained; the
  // online folds keep per-vessel state only. Generous bound to avoid GC
  // flakiness while still failing hard on any slurp regression.
  assert.ok(deltaMb < 30,
    `fold retained ${deltaMb.toFixed(1)}MB heap — a materializing reader has crept back into the analytics path`);
  fs.rmSync(base, { recursive: true, force: true });
});

test("no runtime module calls the materializing vessel readers (tests only)", () => {
  const offenders: string[] = [];
  for (const f of fs.readdirSync(SERVER)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const src = fs.readFileSync(path.join(SERVER, f), "utf8");
    // definition site is allowed; call sites elsewhere are not
    const CALL = /(?<!function )readVesselTracks(Async)?\(/; // definition lines excluded
    if (f === "shadowFleet.ts" || f === "portDwell.ts") {
      // within these, only the sync compute* wrappers may call them —
      // assert the ASYNC paths do not
      const asyncBodies = src.match(/export (?:async )?function \w*Async[\s\S]*?\n}/g) || [];
      for (const body of asyncBodies) {
        if (CALL.test(body)) offenders.push(`${f} (async path)`);
      }
      continue;
    }
    if (CALL.test(src)) offenders.push(f);
  }
  assert.deepEqual(offenders, [],
    `materializing reader used outside tests/sync-test-paths: ${offenders.join(", ")}`);
});
