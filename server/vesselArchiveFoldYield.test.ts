// [REPAIR] 2026-11-21, KNOWN BROKEN #57: a fold cycle whose hours are all
// cached merged them in one block with no turn for the event loop (2.6 s on
// production). The cycle now yields between merges.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  runIncrementalFold, PartialCache, FoldStatsTracker, FOLD_ABSORB_SLICE_MS,
  type CachedPartial, type IncrementalFoldSpec, type FoldMode,
} from "./vesselArchiveFold";

const NOW = Date.UTC(2026, 6, 4, 12, 30, 0);
const HOURS = 40;
const BURN_MS = 5; // one merge; 38 cached hours in a row = about 190 ms

function writeArchive(base: string): void {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const endHour = Math.floor(NOW / 3600_000) * 3600_000;
  for (let h = HOURS - 1; h >= 0; h--) {
    const hourMs = endHour - h * 3600_000;
    const d = new Date(hourMs).toISOString();
    const lines = [0, 1, 2].map((k) => JSON.stringify({ t: Math.floor(hourMs / 1000) + 60 + k * 600, i: `7000${k}`, la: 33.7, lo: -118.2, v: 1 }));
    fs.writeFileSync(path.join(dir, `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl`), lines.join("\n") + "\n");
  }
}

interface P extends CachedPartial { n: number }
interface S { n: number; merges: number }

function spec(base: string, cache: PartialCache<P>, mode: Exclude<FoldMode, "full">, merged: { at: number[] }): IncrementalFoldSpec<S, P, number> {
  return {
    stream: "yieldtest", mode, windowHours: HOURS - 1, baseDir: base, nowMs: NOW, cache,
    stats: new FoldStatsTracker(),
    registryTotals: () => ({ partialsCached: 0, partialBytesEstimate: 0, uncacheableFiles: 0, compactDrops: 0 }),
    newState: () => ({ n: 0, merges: 0 }),
    push: (st) => { st.n++; },
    newBuilder: () => { let n = 0; return { push: () => { n++; }, finish: () => ({ n, bytes: 8, remapIds: () => {} }) }; },
    absorb: (st, p) => {
      const end = performance.now() + BURN_MS;
      while (performance.now() < end) { /* a merge that takes real time */ }
      st.n += p.n; st.merges++; merged.at.push(performance.now());
      return true;
    },
    finish: (st) => st.n,
  };
}

/** Runs `fn` while a setImmediate chain counts the loop's turns; returns the longest wait between two turns. */
async function longestBlockDuring(fn: () => Promise<unknown>): Promise<number> {
  let last = performance.now(), longest = 0, stop = false;
  const spin = (): void => {
    const now = performance.now();
    longest = Math.max(longest, now - last);
    last = now;
    if (!stop) setImmediate(spin);
  };
  setImmediate(spin);
  await fn();
  stop = true;
  await new Promise<void>((r) => setImmediate(r));
  return longest;
}

test("a cycle over cached hours gives the event loop a turn between merges, and the result is unchanged", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fold-yield-"));
  try {
    writeArchive(base);
    const cache = new PartialCache<P>();
    const first = { at: [] as number[] };
    const total = await runIncrementalFold(spec(base, cache, "incremental", first)); // builds the partials
    const second = { at: [] as number[] };
    let served = -1;
    const longest = await longestBlockDuring(async () => { served = await runIncrementalFold(spec(base, cache, "incremental", second)); });
    assert.equal(served, total, "the cached cycle serves what the building cycle served");
    assert.ok(second.at.length >= HOURS - 4, `cached merges in the second cycle: ${second.at.length}`);
    const merging = second.at.length * BURN_MS;
    assert.ok(merging >= 150, `the merges alone take ${merging} ms`);
    // one slice plus one merge, with room for a slow machine; one block of every merge would be 150 ms or more
    const limit = (FOLD_ABSORB_SLICE_MS + BURN_MS) * 4;
    assert.ok(longest < limit, `longest stretch without a turn: ${longest.toFixed(1)} ms (limit ${limit} ms, merges ${merging} ms)`);
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("the slice is the one the DTCC repair uses, and only the cached branch waits on it", () => {
  assert.equal(FOLD_ABSORB_SLICE_MS, 8);
  const src = fs.readFileSync(path.join(path.dirname(new URL(import.meta.url).pathname), "vesselArchiveFold.ts"), "utf8");
  const cycle = src.slice(src.indexOf("async function foldCycle<"));
  assert.match(cycle, /absorbTimed\(cached\);\s+await breathe\(\);\s+continue;/);
  assert.equal((cycle.match(/await breathe\(\)/g) || []).length, 1);
});
