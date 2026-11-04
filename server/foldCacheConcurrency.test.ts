// [REPAIR 2026-11-04] Overlapping incremental fold cycles on ONE partial cache.
//
// Live 2026-11-04 14:33Z onward (v1.0.1271, VOLTRADE_PORTDWELL_FOLD=shadow):
// `FOLD-MISMATCH portdwell: vesselsSeen: 60448 vs 60396` — the incremental
// fold counted 52 fewer vessels than the full fold over the same bytes.
// foldPortVisitsAsync has several callers sharing one PartialCache per
// (archive dir, ports): the 10-min /data refresh (168h), the entity graph's
// calls_at builder (168h), the Tier-3 weekly capture (168h ending at a past
// week) and the portdwell_window diag probe (any window, any end). They run
// concurrently. One cycle's prune() evicts and compact()s the cache's string
// table while another cycle is still mid-pass, holding ids minted by the
// table that was just replaced: a builder that captured the old interner,
// or a partial it fetched before an await that the other cycle evicted
// (evicted partials are not remapped). Those ids then resolve against the
// NEW table: wrong or undefined MMSIs, i.e. merged vessels.
//
// These tests run overlapping cycles with different windows/ends over one
// archive and require (a) every shadow compare to match, and (b) a later
// incremental-mode cycle on the same (possibly poisoned) cache to equal the
// full fold exactly.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  foldPortVisitsAsync, portDwellFoldStats, _resetPortDwellFoldForTests, PORTDWELL_FOLD_ENV,
  type PortDef,
} from "./portDwell";
import {
  computeShadowStatsAsync, shadowFleetFoldStats, _resetShadowFoldForTests, SHADOWFLEET_FOLD_ENV,
  type ShadowZone,
} from "./shadowFleet";
import { PartialCache, type CachedPartial, type WindowFile } from "./vesselArchiveFold";

const NOW = Date.UTC(2026, 6, 4, 12, 30, 0); // 12:30Z — hours <= 11 closed, hour 12 open
const LA: PortDef = { id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, radius_km: 5 };
const LB: PortDef = { id: "port_lb", name: "Port of Long Beach", lat: 33.7515, lon: -118.213, radius_km: 5 };
const PORTS = [LA, LB];

const hourName = (ms: number) => { const d = new Date(ms).toISOString(); return `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl`; };

/** `hours` hour files ending at the hour containing `endMs`. Every hour has
 *  its own vessels (fresh MMSIs, so the string table changes a lot when old
 *  hours are evicted) plus a few long-moored ones crossing many hours. */
function writeArchive(base: string, endMs: number, hours: number, perHour = 120): void {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const endHour = Math.floor(endMs / 3600_000) * 3600_000;
  for (let h = hours - 1; h >= 0; h--) {
    const hourMs = endHour - h * 3600_000;
    const lines: string[] = [];
    for (let k = 0; k < perHour; k++) {
      const tSec = Math.floor((hourMs + ((k * 29) % 3600) * 1000) / 1000);
      if (tSec * 1000 > endMs) continue;
      const inPort = k % 3 === 0;
      const port = k % 2 === 0 ? LA : LB;
      lines.push(JSON.stringify({
        t: tSec, i: `${700000 + h * 1000 + k}`, c: `V${h}_${k}`,
        la: inPort ? port.lat + 0.004 : port.lat + 1.5, lo: inPort ? port.lon + 0.004 : port.lon - 1.5, v: inPort ? 0.3 : 11.2,
      }));
    }
    for (let m = 0; m < 6; m++) {
      for (let s = 0; s < 3; s++) {
        const tSec = Math.floor((hourMs + (600 + s * 1200 + m * 7) * 1000) / 1000);
        if (tSec * 1000 > endMs) continue;
        lines.push(JSON.stringify({ t: tSec, i: `55500${m}`, c: `MOORED${m}`, la: LA.lat + 0.003, lo: LA.lon + 0.003, v: 0.1 }));
      }
    }
    fs.writeFileSync(path.join(dir, hourName(hourMs)), lines.join("\n") + "\n");
  }
}

async function withMode<T>(mode: string, fn: () => Promise<T>, envVar = PORTDWELL_FOLD_ENV): Promise<T> {
  const prev = process.env[envVar];
  process.env[envVar] = mode;
  try { return await fn(); } finally {
    if (prev === undefined) delete process.env[envVar]; else process.env[envVar] = prev;
  }
}

/** Order-independent canonical form of a fold result. */
function canon(r: { visitsByPort: Map<string, unknown[]>; vesselsSeen: number; pointsScanned: number }): string {
  const ports = Array.from(r.visitsByPort.entries()).sort(([a], [b]) => a.localeCompare(b))
    .map(([id, visits]) => [id, visits.map((v) => JSON.stringify(v)).sort()]);
  return JSON.stringify({ vesselsSeen: r.vesselsSeen, pointsScanned: r.pointsScanned, ports });
}

function tmpBase(): string { return fs.mkdtempSync(path.join(os.tmpdir(), "fold-cache-conc-")); }

test("overlapping port-dwell cycles with different windows never produce a shadow mismatch", async () => {
  _resetPortDwellFoldForTests();
  const base = tmpBase();
  try {
    writeArchive(base, NOW + 2 * 3600_000, 40);
    await withMode("shadow", async () => {
      // Warm the cache over a long window, then move `now` so hours fall out
      // and new hours close, and overlap a long-window cycle with short and
      // historical ones (the diag probe / weekly capture shapes).
      await foldPortVisitsAsync(PORTS, 30, base, NOW);
      for (let step = 1; step <= 4; step++) {
        const now = NOW + step * 1800_000;
        await Promise.all([
          foldPortVisitsAsync(PORTS, 30, base, now),
          foldPortVisitsAsync(PORTS, 4, base, now),
          foldPortVisitsAsync(PORTS, 30, base, now),
          foldPortVisitsAsync(PORTS, 8, base, now - 20 * 3600_000),
        ]);
      }
    });
    const s = portDwellFoldStats();
    assert.equal(s.mismatches, 0, `shadow compares mismatched ${s.mismatches} time(s) under overlapping cycles`);
    assert.equal(s.compactDrops, 0, "no partial was ever unresolvable at compaction");
    assert.ok(s.matches > 0, "the shadow compare must actually have run");
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
    _resetPortDwellFoldForTests();
  }
});

test("after overlapping cycles the cached partials still give the exact full result", async () => {
  _resetPortDwellFoldForTests();
  const base = tmpBase();
  try {
    writeArchive(base, NOW + 2 * 3600_000, 40);
    const now = NOW + 4 * 1800_000;
    await withMode("incremental", async () => {
      await foldPortVisitsAsync(PORTS, 30, base, NOW);
      for (let step = 1; step <= 4; step++) {
        const at = NOW + step * 1800_000;
        // Every cycle's served result must equal the full fold too, not only the last one.
        const served = await Promise.all([
          foldPortVisitsAsync(PORTS, 30, base, at),
          foldPortVisitsAsync(PORTS, 3, base, at),
          foldPortVisitsAsync(PORTS, 30, base, at),
          foldPortVisitsAsync(PORTS, 6, base, at - 15 * 3600_000),
        ]);
        const fullLong = await withMode("full", () => foldPortVisitsAsync(PORTS, 30, base, at));
        assert.equal(canon(served[0]), canon(fullLong), `step ${step}: first long-window cycle`);
        assert.equal(canon(served[2]), canon(fullLong), `step ${step}: second long-window cycle`);
      }
    });
    const inc = await withMode("incremental", () => foldPortVisitsAsync(PORTS, 30, base, now));
    const full = await withMode("full", () => foldPortVisitsAsync(PORTS, 30, base, now));
    assert.equal(canon(inc), canon(full));
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
    _resetPortDwellFoldForTests();
  }
});

test("overlapping shadow-fleet cycles (same machinery) never produce a shadow mismatch", async () => {
  _resetShadowFoldForTests();
  const base = tmpBase();
  const zones: ShadowZone[] = [{ id: "la_anchorage", name: "LA anchorage", lat: LA.lat, lon: LA.lon, radius_km: 45 }];
  try {
    writeArchive(base, NOW + 2 * 3600_000, 40);
    await withMode("shadow", async () => {
      await computeShadowStatsAsync(zones, 30, base, NOW);
      for (let step = 1; step <= 4; step++) {
        const now = NOW + step * 1800_000;
        await Promise.all([
          computeShadowStatsAsync(zones, 30, base, now),
          computeShadowStatsAsync(zones, 4, base, now),
          computeShadowStatsAsync(zones, 30, base, now),
          computeShadowStatsAsync(zones, 8, base, now - 20 * 3600_000),
        ]);
      }
    }, SHADOWFLEET_FOLD_ENV);
    const s = shadowFleetFoldStats();
    assert.equal(s.mismatches, 0, `shadow compares mismatched ${s.mismatches} time(s) under overlapping cycles`);
    assert.equal(s.compactDrops, 0, "no partial was ever unresolvable at compaction");
    assert.ok(s.matches > 0, "the shadow compare must actually have run");
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
    _resetShadowFoldForTests();
  }
});

interface FakePartial extends CachedPartial { ids: number[] }
const fake = (ids: number[]): FakePartial => ({
  ids, bytes: 10,
  remapIds(map) { for (let i = 0; i < this.ids.length; i++) this.ids[i] = map(this.ids[i]); },
});
const wf = (name: string, stampSec: number): WindowFile => ({ name, fp: `/x/${name}`, stampSec, size: 1, mtimeMs: 1, cacheable: true });

test("PartialCache: a prune inside an in-flight cycle defers compaction until the last cycle ends; held ids keep resolving", () => {
  const cache = new PartialCache<FakePartial>();
  const a = cache.interner.id("MMSI-A"), b = cache.interner.id("MMSI-B");
  cache.set(wf("h1", 1000), fake([a]));
  cache.set(wf("h2", 2000), fake([b]));
  const table = cache.interner;
  cache.beginCycle();
  cache.beginCycle();
  const held = cache.get(wf("h1", 1000))!; // fetched by a cycle that then awaits
  assert.equal(cache.prune(new Set(["h2"]), 10), 1, "h1 evicted");
  assert.equal(cache.interner, table, "table not replaced while cycles are in flight");
  assert.equal(cache.interner.str(held.ids[0]), "MMSI-A", "the evicted partial a cycle still holds resolves to its own string");
  cache.endCycle();
  assert.equal(cache.interner, table, "one cycle still in flight: still not compacted");
  cache.endCycle();
  assert.notEqual(cache.interner, table, "the last cycle out compacts");
  assert.equal(cache.interner.size, 1);
  assert.equal(cache.interner.str(cache.get(wf("h2", 2000))!.ids[0]), "MMSI-B", "the survivor was remapped onto the new table");
  assert.equal(cache.cyclesInFlight, 0);
  // outside any cycle prune still compacts at once (the pre-existing contract)
  cache.set(wf("h3", 3000), fake([cache.interner.id("MMSI-C")]));
  cache.prune(new Set(["h3"]), 10);
  assert.equal(cache.interner.size, 1);
  assert.equal(cache.interner.str(cache.get(wf("h3", 3000))!.ids[0]), "MMSI-C");
});

test("PartialCache.compact drops a partial holding an id the table does not know instead of throwing", () => {
  const cache = new PartialCache<FakePartial>();
  cache.set(wf("good", 1000), fake([cache.interner.id("MMSI-A")]));
  cache.set(wf("bad", 2000), fake([cache.interner.size + 50]));
  const errors: string[] = [];
  const orig = console.error;
  console.error = (...args: unknown[]) => { errors.push(args.map(String).join(" ")); };
  try { assert.doesNotThrow(() => cache.compact()); } finally { console.error = orig; }
  assert.equal(cache.size, 1);
  assert.equal(cache.peek("bad"), undefined, "the unresolvable partial is dropped (re-folded next cycle)");
  assert.equal(cache.interner.str(cache.peek("good")!.ids[0]), "MMSI-A");
  assert.equal(cache.compactDrops, 1);
  assert.ok(errors.some((e) => /compact dropped the partial for bad/.test(e)), "the drop is logged, not silent");
});
