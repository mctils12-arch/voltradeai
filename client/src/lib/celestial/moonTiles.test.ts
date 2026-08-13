// moonTiles.test — pure planning + stitch, and the streaming manager driven
// headless through the injected fetch seam (mosaic assembly, window fields,
// bounded bands, eviction, rebuild).

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  planMoonTarget,
  targetKey,
  tilePlacements,
  blitTile,
  createMoonTileManager,
  MOON_MAX_TILES,
  TILE_CACHE_MAX,
  type TexImageLike,
} from "./moonTiles.ts";
import { mosaicWindow, tilesForBbox, MOON_TREK } from "./lroc.ts";

// ── planning ────────────────────────────────────────────────────────────────

test("planMoonTarget stays within the tile budget", () => {
  const t = planMoonTarget(0, 0, 1e6 /* demand max z */, 40)!;
  assert.ok(t);
  assert.ok(t.tiles.length <= MOON_MAX_TILES, `tiles ${t.tiles.length}`);
  // backed off from z=8 (which over 80° would be hundreds of tiles)
  assert.ok(t.z < 8);
  assert.equal(t.mosaic.z, t.z);
});

test("planMoonTarget honours a modest resolution at a small span", () => {
  const t = planMoonTarget(20, -10, 6, 8)!;
  assert.ok(t.tiles.length >= 1 && t.tiles.length <= MOON_MAX_TILES);
  // the requested sub-point lies inside the mosaic window
  const m = t.mosaic;
  assert.ok(m.latMax >= -10 - 1e-6);
  assert.ok(m.latMax - m.latSpan <= -10 + 1e-6);
});

test("planMoonTarget clamps latitude near the pole", () => {
  const t = planMoonTarget(0, 88, 20, 10)!;
  assert.ok(t.bbox.latMax <= 90 && t.bbox.latMin >= -90);
});

test("targetKey changes with the sub-point / zoom", () => {
  const a = planMoonTarget(0, 0, 20, 10)!;
  const b = planMoonTarget(60, 0, 20, 10)!;
  assert.notEqual(targetKey(a), targetKey(b));
});

// ── stitch ──────────────────────────────────────────────────────────────────

test("tilePlacements grids row-major from the NW corner", () => {
  const tiles = tilesForBbox({ lonMin: -10, lonMax: 10, latMin: -10, latMax: 10 }, 5);
  const places = tilePlacements(tiles);
  assert.equal(places[0].col, 0);
  assert.equal(places[0].row, 0);
  for (const p of places) {
    assert.ok(p.col >= 0 && p.row >= 0);
  }
});

test("blitTile writes a tile into its mosaic cell (banded == whole)", () => {
  const tiles = tilesForBbox({ lonMin: -5, lonMax: 5, latMin: -5, latMax: 5 }, 6);
  const m = mosaicWindow(tiles)!;
  const places = tilePlacements(tiles);
  const tilePx = MOON_TREK.tilePx;
  // one solid tile per placement (colour = 10 + index)
  const rgbaFor = (i: number): TexImageLike => {
    const d = new Uint8ClampedArray(tilePx * tilePx * 4);
    for (let p = 0; p < tilePx * tilePx; p++) {
      d[p * 4] = 10 + i;
      d[p * 4 + 1] = 20;
      d[p * 4 + 2] = 30;
      d[p * 4 + 3] = 255;
    }
    return { data: d, width: tilePx, height: tilePx };
  };
  const whole = new Uint8ClampedArray(m.pxW * m.pxH * 4);
  const banded = new Uint8ClampedArray(m.pxW * m.pxH * 4);
  places.forEach((p, i) => blitTile(rgbaFor(i), p, m, tilePx, whole, 0, m.pxH));
  for (let y = 0; y < m.pxH; y += 50) {
    places.forEach((p, i) => blitTile(rgbaFor(i), p, m, tilePx, banded, y, Math.min(y + 50, m.pxH)));
  }
  assert.deepEqual(Array.from(whole), Array.from(banded));
  // the first tile's top-left pixel is its colour, opaque
  assert.equal(whole[0], 10);
  assert.equal(whole[3], 255);
});

// ── the manager (headless via the fetch seam) ───────────────────────────────

function fakeDeps() {
  const fetched: string[] = [];
  const fetchTile = async (url: string): Promise<TexImageLike> => {
    fetched.push(url);
    const tilePx = MOON_TREK.tilePx;
    const d = new Uint8ClampedArray(tilePx * tilePx * 4);
    for (let p = 0; p < tilePx * tilePx; p++) {
      d[p * 4] = 123;
      d[p * 4 + 1] = 200;
      d[p * 4 + 2] = 77;
      d[p * 4 + 3] = 255;
    }
    return { data: d, width: tilePx, height: tilePx };
  };
  return { fetched, deps: { fetchTile, bandRows: 128 } };
}

const settle = async (pred: () => boolean, ms = 2000): Promise<void> => {
  const t0 = Date.now();
  while (!pred() && Date.now() - t0 < ms) await new Promise((r) => setTimeout(r, 5));
};

test("manager streams tiles, stitches a windowed mosaic, records bounded bands", async () => {
  const { fetched, deps } = fakeDeps();
  const mgr = createMoonTileManager(deps);
  let updates = 0;
  mgr.onUpdate(() => { updates++; });
  assert.equal(mgr.current(), null, "nothing before request (zero startup cost)");

  mgr.request(0, 0, 20, 8);
  await settle(() => mgr.current() !== null);
  const mos = mgr.current()!;
  assert.ok(mos, "mosaic built");
  assert.ok(fetched.length > 0, "tiles fetched from trek urls");
  assert.match(fetched[0], /trek\.nasa\.gov/);
  assert.ok(mos.tex.data.byteLength > 0);
  assert.equal(mos.tex.width * mos.tex.height * 4, mos.tex.data.byteLength);
  // window fields are the DetailOverlay contract
  assert.ok(Number.isFinite(mos.lonMin) && mos.lonSpan > 0 && mos.latSpan > 0);
  // a mosaic pixel carries the fetched colour
  assert.equal(mos.tex.data[0], 123);
  assert.ok(updates >= 1);
  const st = mgr.stats();
  assert.ok(st.bytes > 0 && st.tiles > 0);
  assert.ok(st.z !== null);

  mgr.dispose();
});

test("clear() evicts the mosaic (bytes drop to 0), then a new request rebuilds", async () => {
  const { deps } = fakeDeps();
  const mgr = createMoonTileManager(deps);
  mgr.request(0, 0, 20, 8);
  await settle(() => mgr.current() !== null);
  assert.ok(mgr.stats().bytes > 0);

  mgr.clear();
  assert.equal(mgr.current(), null);
  assert.equal(mgr.stats().bytes, 0);
  assert.equal((globalThis as any).__vtMoonTileBytes, 0);

  mgr.request(0, 0, 20, 8);
  await settle(() => mgr.current() !== null);
  assert.ok(mgr.stats().bytes > 0, "rebuilt after eviction");
  mgr.dispose();
});

test("clear() during an in-flight build does NOT repopulate the mosaic (B5 evict race)", async () => {
  // a fetch that only resolves when we release it — lets us evict mid-build
  const gates: Array<() => void> = [];
  let fetches = 0;
  const fetchTile = async (): Promise<TexImageLike> => {
    fetches++;
    await new Promise<void>((res) => gates.push(res));
    const px = MOON_TREK.tilePx;
    const d = new Uint8ClampedArray(px * px * 4);
    for (let p = 0; p < px * px; p++) { d[p * 4] = 50; d[p * 4 + 3] = 255; }
    return { data: d, width: px, height: px };
  };
  const mgr = createMoonTileManager({ fetchTile, bandRows: 128 });
  mgr.request(0, 0, 20, 8);
  await settle(() => fetches > 0); // build is now awaiting the gated fetches
  // EVICT while the fetches are still outstanding
  mgr.clear();
  assert.equal(mgr.current(), null);
  assert.equal((globalThis as any).__vtMoonTileBytes, 0);
  // release every gated fetch → the stale build must bail, not set a mosaic
  for (const g of gates) g();
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(mgr.current(), null, "evicted build never repopulated the mosaic");
  assert.equal((globalThis as any).__vtMoonTileBytes, 0, "__vtMoonTileBytes stayed 0 after the race");
  // a fresh request after the race still works
  mgr.request(0, 0, 20, 8);
  await settle(() => gates.length > fetches - 1); // new fetches queued
  for (const g of gates) g();
  await settle(() => mgr.current() !== null);
  assert.ok(mgr.stats().bytes > 0, "manager still usable after an evict-during-build race");
  mgr.dispose();
});

test("a new sub-point supersedes and rebuilds a different mosaic", async () => {
  const { deps } = fakeDeps();
  const mgr = createMoonTileManager(deps);
  mgr.request(0, 0, 20, 8);
  await settle(() => mgr.current() !== null);
  const z0 = mgr.stats().z;
  mgr.request(120, 30, 20, 8);
  await settle(() => {
    const m = mgr.current();
    return !!m && Math.abs(m.lonMin - 0) > 1; // window moved east
  });
  const m = mgr.current()!;
  assert.ok(Math.abs(m.lonMin) > 1 || m.z !== z0, "mosaic re-planned for the new sub-point");
  mgr.dispose();
});

test("dispose frees everything and stops updates", async () => {
  const { deps } = fakeDeps();
  const mgr = createMoonTileManager(deps);
  mgr.request(0, 0, 20, 8);
  await settle(() => mgr.current() !== null);
  mgr.dispose();
  assert.equal(mgr.current(), null);
  assert.equal((globalThis as any).__vtMoonTileBytes, 0);
});

// ── alpha preservation (NAC strips are gray+alpha PNG, 2026-08-12) ──────────

test("blitTile copies REAL source alpha — transparent strip margins stay holes", async () => {
  const { blitTile } = await import("./moonTiles.ts");
  const tilePx = 2;
  // 2×2 tile: top-left opaque, top-right transparent
  const tile = {
    width: 2, height: 2,
    data: new Uint8ClampedArray([
      100, 100, 100, 255,   50, 50, 50, 0,
      100, 100, 100, 255,  100, 100, 100, 255,
    ]),
  };
  const mosaic = { pxW: 2, pxH: 2 } as any;
  const out = new Uint8ClampedArray(2 * 2 * 4);
  blitTile(tile as any, { tile: { z: 1, x: 0, y: 0 }, col: 0, row: 0 }, mosaic, tilePx, out, 0, 2);
  assert.equal(out[3], 255, "opaque texel keeps alpha 255");
  assert.equal(out[7], 0, "transparent texel keeps alpha 0 (falls through to the next tier)");
  assert.equal(out[0], 100);
});

// ── PR-6 hardening (2026-08-13): the three defects that would have ruined the
//    Apollo landing view — a blanket cache wipe mid-descent, a supersede bail
//    that starved the mosaic during any camera motion, and a silent permanent
//    hole on a single failed tile. Each test fails if its fix is reverted. ──

const solidTile = (v: number): TexImageLike => {
  const px = MOON_TREK.tilePx;
  const d = new Uint8ClampedArray(px * px * 4);
  for (let p = 0; p < px * px; p++) { d[p * 4] = v; d[p * 4 + 1] = v; d[p * 4 + 2] = v; d[p * 4 + 3] = 255; }
  return { data: d, width: px, height: px };
};

test("PR6: the tile cache evicts LRU and never blanket-wipes (bounded, not emptied)", async () => {
  let fetches = 0;
  const fetchTile = async (): Promise<TexImageLike> => { fetches++; return solidTile(60); };
  const mgr = createMoonTileManager({ fetchTile, bandRows: 128 });
  // build several DIFFERENT mosaics to accumulate tiles well past one plan
  for (const [lon, lat] of [[0, 0], [30, 10], [-40, -20], [80, 30], [-120, 40]] as const) {
    mgr.request(lon, lat, 20, 8);
    await settle(() => mgr.current() !== null && !mgr.stats().building, 4000);
  }
  const s = mgr.stats();
  assert.ok(s.cachedTiles > 0, "cache must retain tiles across mosaics — a blanket clear would drop to ~0");
  assert.ok(s.cachedTiles <= TILE_CACHE_MAX, `cache must stay bounded (${s.cachedTiles} > ${TILE_CACHE_MAX})`);
  assert.ok(fetches > 0);
  mgr.dispose();
});

test("PR6: a moving camera still gets a mosaic — supersede must not starve the first build", async () => {
  // Every request supersedes the last, exactly like a fly-to descent. Before
  // the fix the build bailed unstitched every pass and current() stayed null
  // for the whole descent, leaving the base texture magnified ~5,300x.
  const gates: Array<() => void> = [];
  const fetchTile = async (): Promise<TexImageLike> => {
    await new Promise<void>((res) => gates.push(res));
    return solidTile(90);
  };
  const mgr = createMoonTileManager({ fetchTile, bandRows: 128 });
  mgr.request(0, 0, 20, 8);
  await settle(() => gates.length > 0, 3000);
  // camera keeps moving: queue a DIFFERENT target while the first is in flight
  mgr.request(5, 5, 20, 8);
  for (const g of gates) g();
  await settle(() => mgr.current() !== null, 5000);
  assert.ok(mgr.current(), "a superseded first build must still stitch — never leave the screen with no mosaic");
  mgr.dispose();
});

test("PR6: a failing tile is retried and then COUNTED, never a silent permanent hole", async () => {
  let attempts = 0;
  const fetchTile = async (): Promise<TexImageLike> => {
    attempts++;
    throw new Error("simulated 404");
  };
  const mgr = createMoonTileManager({ fetchTile, bandRows: 128 });
  mgr.request(0, 0, 6, 8);
  await settle(() => mgr.stats().fetchFailures > 0, 8000);
  const s = mgr.stats();
  assert.ok(s.fetchFailures > 0, "an exhausted tile must be counted so the failure is observable");
  // retried, not given up on after one try
  assert.ok(attempts > s.fetchFailures,
    `each failure must include retries (attempts ${attempts} vs failures ${s.fetchFailures})`);
  mgr.dispose();
});

test("PR6: a tile that succeeds on retry is NOT counted as a failure", async () => {
  let n = 0;
  const fetchTile = async (): Promise<TexImageLike> => {
    n++;
    if (n === 1) throw new Error("transient");
    return solidTile(70);
  };
  const mgr = createMoonTileManager({ fetchTile, bandRows: 128 });
  mgr.request(0, 0, 6, 8);
  await settle(() => mgr.current() !== null, 6000);
  assert.equal(mgr.stats().fetchFailures, 0, "a transient error that recovers on retry is not a failure");
  mgr.dispose();
});
