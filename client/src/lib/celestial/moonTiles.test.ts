// moonTiles.test — pure planning + stitch, and the streaming manager driven
// headless through the injected fetch seam (mosaic assembly, window fields,
// bounded bands, eviction, rebuild).

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  planMoonTarget,
  spanLadder,
  targetKey,
  tilePlacements,
  blitTile,
  createMoonTileManager,
  MOON_MAX_TILES,
  MOON_MOSAIC_MAX_PX,
  type TexImageLike,
} from "./moonTiles.ts";
import { mosaicWindow, tilesForBbox, degPerTile, MOON_TREK } from "./lroc.ts";

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

// ── 3b: margin is spent before resolution (Law II.3/II.4, 2026-08-13) ───────
//
// The Moon was rendering 1-3 zoom levels below what the screen deserved
// (2x-8x soft). Cause: spaceFrame asks for a span 2.8x wider than the visible
// disc (MOON_PATCH_COVER_MARGIN — 7.8x the AREA, of which only ~36% of the
// span is ever on screen), and the planner used to hold that whole margin and
// drop zoom LEVELS until the inflated mosaic fit MOON_MOSAIC_MAX_PX.
//
// The fix inverts the order of sacrifice: give back prefetch margin first,
// down to the visible region plus a one-tile ring, and only then a level.
// These tests pin both the recovery AND the safety property that makes it
// legal — the mosaic must never stop covering what is actually on screen.

const MARGIN = 2.8; // spaceFrame.MOON_PATCH_COVER_MARGIN

test("spanLadder collapses to a single rung when there is nothing to give back", () => {
  // want <= must + ring: no headroom, so behaviour is exactly as before.
  assert.deepEqual(spanLadder(5, 5, 4), [5]);
  assert.deepEqual(spanLadder(3, 90, 4), [3]);
});

test("spanLadder descends and ends exactly at the visible span plus one tile", () => {
  const want = 56;
  const must = 10;
  const z = 5;
  const rungs = spanLadder(want, must, z);
  assert.ok(rungs.length >= 2, "a ladder with headroom must offer alternatives");
  assert.equal(rungs[0], want, "the widest rung is tried first — margin is kept when it fits");
  assert.equal(
    rungs[rungs.length - 1],
    must + degPerTile(z),
    "the cheapest rung is the visible region plus a one-tile pan ring (Law II.4)",
  );
  for (let i = 1; i < rungs.length; i++) {
    assert.ok(rungs[i] < rungs[i - 1], `ladder must descend: ${rungs[i - 1]} -> ${rungs[i]}`);
  }
});

test("the ring floor scales with the level, so coarse levels keep a wider ring", () => {
  const a = spanLadder(60, 10, 4).at(-1)!;
  const b = spanLadder(60, 10, 7).at(-1)!;
  assert.ok(a > b, "a z4 tile spans more degrees than a z7 tile, so its ring is wider");
});

test("telling the planner what is visible RECOVERS zoom levels", () => {
  // The measured regression case: a 1200px disc over a 20-deg span used to
  // land 2 levels under ideal. Pinning that it no longer does.
  const discPx = 1200;
  const spanDeg = 20;
  const pxPerDeg = discPx / spanDeg;
  const visHalf = spanDeg / 2;
  const wantHalf = visHalf * MARGIN;
  const before = planMoonTarget(0, 0, pxPerDeg, wantHalf, {})!;
  const after = planMoonTarget(0, 0, pxPerDeg, wantHalf, { minHalfSpanDeg: visHalf })!;
  assert.ok(
    after.z > before.z,
    `expected a sharper level once the visible span is known (was z${before.z}, got z${after.z})`,
  );
});

test("the recovery holds across the whole viewport matrix, never regressing", () => {
  let recovered = 0;
  for (const discPx of [400, 800, 1200, 1600, 2000]) {
    for (const spanDeg of [20, 60]) {
      const pxPerDeg = discPx / spanDeg;
      const visHalf = spanDeg / 2;
      const wantHalf = visHalf * MARGIN;
      const before = planMoonTarget(0, 0, pxPerDeg, wantHalf, {})!;
      const after = planMoonTarget(0, 0, pxPerDeg, wantHalf, { minHalfSpanDeg: visHalf })!;
      assert.ok(
        after.z >= before.z,
        `${discPx}px/${spanDeg}deg REGRESSED: z${before.z} -> z${after.z}`,
      );
      recovered += after.z - before.z;
    }
  }
  assert.ok(recovered >= 8, `expected >=8 levels recovered across the matrix, got ${recovered}`);
});

test("SAFETY: the mosaic never stops covering what is on screen", () => {
  // The only way shrinking the span could be a bug. If the returned window
  // ever failed to contain the visible disc, the user would see an untextured
  // edge — strictly worse than the fuzz this replaces.
  for (const discPx of [400, 1200, 2000]) {
    for (const spanDeg of [20, 60]) {
      for (const lat of [0, 45, -70]) {
        const pxPerDeg = discPx / spanDeg;
        const visHalf = spanDeg / 2;
        const t = planMoonTarget(0, lat, pxPerDeg, visHalf * MARGIN, { minHalfSpanDeg: visHalf })!;
        assert.ok(t, `no target for ${discPx}/${spanDeg}/${lat}`);
        const m = t.mosaic;
        // longitude: the window must span at least the visible arc
        assert.ok(
          m.lonSpan >= visHalf * 2 - 1e-9,
          `lon coverage ${m.lonSpan} < visible ${visHalf * 2} at ${discPx}/${spanDeg}/${lat}`,
        );
        // latitude: same, but the poles legitimately clamp the demand
        const wantLatTop = Math.min(90, lat + visHalf);
        const wantLatBot = Math.max(-90, lat - visHalf);
        assert.ok(
          m.latMax >= wantLatTop - 1e-9,
          `north edge ${m.latMax} does not reach ${wantLatTop}`,
        );
        assert.ok(
          m.latMax - m.latSpan <= wantLatBot + 1e-9,
          `south edge ${m.latMax - m.latSpan} does not reach ${wantLatBot}`,
        );
      }
    }
  }
});

test("the budget is still respected — recovery may not overspend VRAM", () => {
  for (const discPx of [400, 1200, 2000]) {
    for (const spanDeg of [20, 60]) {
      const pxPerDeg = discPx / spanDeg;
      const visHalf = spanDeg / 2;
      const t = planMoonTarget(0, 0, pxPerDeg, visHalf * MARGIN, { minHalfSpanDeg: visHalf })!;
      assert.ok(t.mosaic.pxW <= MOON_MOSAIC_MAX_PX, `pxW ${t.mosaic.pxW} over cap`);
      assert.ok(t.mosaic.pxH <= MOON_MOSAIC_MAX_PX, `pxH ${t.mosaic.pxH} over cap`);
      assert.ok(t.tiles.length <= MOON_MAX_TILES, `tiles ${t.tiles.length} over cap`);
    }
  }
});

test("BACKWARD COMPATIBLE: omitting the visible span changes nothing", () => {
  // Every pre-existing caller must be byte-identical, so this change cannot
  // have moved anything except the one call site that opts in.
  for (const discPx of [400, 900, 1500]) {
    for (const spanDeg of [15, 45, 75]) {
      const pxPerDeg = discPx / spanDeg;
      const wantHalf = (spanDeg / 2) * MARGIN;
      const plain = planMoonTarget(10, -5, pxPerDeg, wantHalf, {})!;
      // passing the DESIRED span as the floor is the degenerate case and must
      // reproduce the old plan exactly
      const degenerate = planMoonTarget(10, -5, pxPerDeg, wantHalf, { minHalfSpanDeg: wantHalf })!;
      assert.equal(degenerate.z, plain.z);
      assert.equal(targetKey(degenerate), targetKey(plain));
    }
  }
});
