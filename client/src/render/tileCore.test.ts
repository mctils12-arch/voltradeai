// tileCore.test.ts — Law II, mechanically.
//
// The high-value cases are the ones that would have caught the moon fuzz:
// an unready node is never drawable, a decode that lands after its node
// merged is discarded rather than painted, and every ImageBitmap is closed
// on every path including success.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CACHE,
  NodeState,
  STREAM,
  TILE,
  TileNode,
  TileStreamer,
  baseGeometricError,
  canTransition,
  canUpload,
  childrenOf,
  decodeWorkerCount,
  evictionPlan,
  fadeProgress,
  float32PrecisionAt,
  geometricError,
  isAncestorOf,
  isDrawable,
  isLowPowerDevice,
  lodDecision,
  parentOf,
  parseTileKey,
  relativeToEye,
  retryDelayMs,
  screenSpaceError,
  skirtDepth,
  sortRequests,
  tileKey,
  uvSubsetIn,
  vramBudgetBytes,
  compressedTileBytes,
  uncompressedTileBytes,
  type ImageBitmapLike,
  type TileId,
  type TileSink,
  type TileSource,
} from "./tileCore.ts";
import { METRIC, getCounter, getGauge, resetMetrics, textureBytes } from "./perfMetrics.ts";

const MOON_RADIUS_KM = 1737.4;

test("TILE/STREAM/CACHE constants match the Rendering & Motion Law article", () => {
  assert.equal(TILE.SIZE_PX, 512);
  assert.equal(TILE.SSE_SPLIT_PX, 2.0);
  assert.equal(TILE.SSE_MERGE_PX, 1.0);
  assert.equal(TILE.FRUSTUM_PAD_TILES, 1);
  assert.equal(TILE.PREFETCH_LEVELS_AHEAD, 1);
  assert.equal(TILE.SKIRT_DEPTH_FACTOR, 2.0);

  assert.equal(STREAM.MAX_INFLIGHT, 8);
  assert.equal(STREAM.MAX_UPLOADS_PER_FRAME, 2);
  assert.equal(STREAM.UPLOAD_BUDGET_MS, 4);
  assert.equal(STREAM.DECODE_WORKERS, 3);
  assert.equal(STREAM.CROSSFADE_MS, 200);
  assert.ok(STREAM.CROSSFADE_MS >= 150 && STREAM.CROSSFADE_MS <= 300, "crossfade must stay in the 150-300ms band");
  assert.equal(STREAM.REQUEST_TIMEOUT_MS, 15000);
  assert.equal(STREAM.MAX_RETRIES, 2);
  assert.equal(STREAM.RETRY_BASE_MS, 400);

  assert.equal(CACHE.VRAM_BUDGET_MB_MOBILE, 256);
  assert.equal(CACHE.VRAM_BUDGET_MB_DESKTOP, 768);
  assert.equal(CACHE.EVICT_TO_FRACTION, 0.9);
  assert.equal(CACHE.PINNED_BASE_LEVEL, 0);
});

// ── quadtree ────────────────────────────────────────────────────────────────

test("tile keys round-trip and parent/child are inverses", () => {
  const t: TileId = { level: 5, x: 13, y: 7 };
  assert.equal(tileKey(t), "5/13/7");
  assert.deepEqual(parseTileKey("5/13/7"), t);
  assert.equal(parseTileKey("nope"), null);
  const kids = childrenOf(t);
  assert.equal(kids.length, 4);
  for (const k of kids) assert.deepEqual(parentOf(k), t);
  assert.equal(parentOf({ level: 0, x: 0, y: 0 }), null, "the root has no parent");
});

test("isAncestorOf is strict and correct across several levels", () => {
  const root = { level: 0, x: 0, y: 0 };
  assert.equal(isAncestorOf(root, { level: 3, x: 5, y: 6 }), true);
  assert.equal(isAncestorOf(root, root), false, "a node is not its own ancestor");
  assert.equal(isAncestorOf({ level: 2, x: 1, y: 1 }, { level: 3, x: 2, y: 2 }), true);
  assert.equal(isAncestorOf({ level: 2, x: 1, y: 1 }, { level: 3, x: 4, y: 2 }), false);
  assert.equal(isAncestorOf({ level: 3, x: 0, y: 0 }, { level: 2, x: 0, y: 0 }), false, "levels must not invert");
});

test("uvSubsetIn places a descendant inside its ancestor — the parent-hold mechanism", () => {
  const anc = { level: 0, x: 0, y: 0 };
  assert.deepEqual(uvSubsetIn(anc, anc), { scale: 1, offsetU: 0, offsetV: 0 });
  assert.deepEqual(uvSubsetIn(anc, { level: 1, x: 0, y: 0 }), { scale: 0.5, offsetU: 0, offsetV: 0 });
  assert.deepEqual(uvSubsetIn(anc, { level: 1, x: 1, y: 1 }), { scale: 0.5, offsetU: 0.5, offsetV: 0.5 });
  assert.deepEqual(uvSubsetIn(anc, { level: 2, x: 3, y: 1 }), { scale: 0.25, offsetU: 0.75, offsetV: 0.25 });
  assert.equal(uvSubsetIn({ level: 1, x: 0, y: 0 }, { level: 2, x: 3, y: 3 }), null, "not a descendant");
});

test("uvSubsetIn tiles the ancestor exactly — four children cover it with no gap or overlap", () => {
  const anc = { level: 4, x: 3, y: 2 };
  const subs = childrenOf(anc).map((c) => uvSubsetIn(anc, c)!);
  const area = subs.reduce((a, s) => a + s.scale * s.scale, 0);
  assert.ok(Math.abs(area - 1) < 1e-12, `children cover ${area} of the parent`);
  const corners = new Set(subs.map((s) => `${s.offsetU},${s.offsetV}`));
  assert.equal(corners.size, 4, "each child occupies a distinct quadrant");
});

// ── screen-space error ──────────────────────────────────────────────────────

test("geometric error halves every level, from the documented base", () => {
  const base = baseGeometricError(MOON_RADIUS_KM);
  assert.ok(Math.abs(base - (2 * Math.PI * MOON_RADIUS_KM) / 512) < 1e-12);
  assert.equal(geometricError(0, MOON_RADIUS_KM), base);
  assert.equal(geometricError(1, MOON_RADIUS_KM), base / 2);
  assert.equal(geometricError(10, MOON_RADIUS_KM), base / 1024);
});

test("sse follows the documented formula and falls with distance", () => {
  const ge = geometricError(4, MOON_RADIUS_KM);
  const fov = (36.87 * Math.PI) / 180;
  const near = screenSpaceError(ge, 100, 900, fov);
  const far = screenSpaceError(ge, 10000, 900, fov);
  assert.ok(near > far, "a closer node has more screen-space error");
  const expected = (ge * 900) / (100 * 2 * Math.tan(fov / 2));
  assert.ok(Math.abs(near - expected) < 1e-9);
  assert.ok(Math.abs(near / far - 100) < 1e-9, "sse is inversely proportional to distance");
});

test("sse is Infinity (i.e. always split) for degenerate camera inputs", () => {
  assert.equal(screenSpaceError(1, 0, 900, 1), Infinity);
  assert.equal(screenSpaceError(1, -5, 900, 1), Infinity);
  assert.equal(screenSpaceError(1, 100, 0, 1), Infinity);
  assert.equal(screenSpaceError(1, 100, 900, 0), Infinity);
  assert.equal(lodDecision(Infinity), "hold", "but the LOD decision refuses to act on a degenerate reading");
});

test("LOD hysteresis: there is a band where neither split nor merge fires", () => {
  // Without the band, a node parked at the threshold splits (lowering its
  // sse), then merges (raising it), forever, with the camera stationary.
  assert.equal(lodDecision(3), "split");
  assert.equal(lodDecision(0.5), "merge");
  assert.equal(lodDecision(1.5), "hold");
  assert.equal(lodDecision(TILE.SSE_SPLIT_PX), "hold", "exactly at split is not a split");
  assert.equal(lodDecision(TILE.SSE_MERGE_PX), "hold", "exactly at merge is not a merge");
  assert.ok(TILE.SSE_SPLIT_PX > TILE.SSE_MERGE_PX, "the hysteresis gap must be nonzero");
});

test("splitting a node lands it inside the hold band, not back at merge (no thrash)", () => {
  // A split halves the geometric error, so the child's sse is half the
  // parent's. If the parent split at just over SPLIT, the child sits at
  // just over SPLIT/2 — which must be at or above MERGE, or the child
  // immediately merges back.
  assert.ok(TILE.SSE_SPLIT_PX / 2 >= TILE.SSE_MERGE_PX, "split/merge thresholds admit a thrash cycle");
});

test("skirt depth scales with geometric error and is never negative", () => {
  const ge = geometricError(6, MOON_RADIUS_KM);
  assert.equal(skirtDepth(ge), ge * TILE.SKIRT_DEPTH_FACTOR);
  assert.ok(skirtDepth(ge) > skirtDepth(geometricError(7, MOON_RADIUS_KM)), "coarser levels need deeper skirts");
  assert.equal(skirtDepth(-1), 0);
});

// ── the ready-gate ──────────────────────────────────────────────────────────

test("a node is drawable ONLY in FADING or RESIDENT", () => {
  assert.equal(isDrawable(NodeState.FADING), true);
  assert.equal(isDrawable(NodeState.RESIDENT), true);
  for (const s of [NodeState.PENDING, NodeState.REQUESTED, NodeState.DECODING, NodeState.UPLOADED, NodeState.FAILED]) {
    assert.equal(isDrawable(s), false, `${s} must not be drawable`);
  }
});

test("UPLOADED is deliberately not drawable — the texture exists, the fade has not begun", () => {
  const n = new TileNode({ id: { level: 2, x: 1, y: 1 } });
  n.transition(NodeState.REQUESTED);
  n.transition(NodeState.DECODING);
  n.transition(NodeState.UPLOADED);
  assert.equal(n.drawable, false, "drawing here is the hard pop the crossfade exists to remove");
  n.transition(NodeState.FADING, 0);
  assert.equal(n.drawable, true);
});

test("the state machine refuses to jump the ready-gate", () => {
  assert.equal(canTransition(NodeState.PENDING, NodeState.RESIDENT), false);
  assert.equal(canTransition(NodeState.PENDING, NodeState.FADING), false);
  assert.equal(canTransition(NodeState.REQUESTED, NodeState.UPLOADED), false);
  assert.equal(canTransition(NodeState.DECODING, NodeState.UPLOADED), true);
  const n = new TileNode({ id: { level: 1, x: 0, y: 0 } });
  assert.equal(n.transition(NodeState.RESIDENT), false, "an illegal transition is refused, not silently applied");
  assert.equal(n.state, NodeState.PENDING);
});

test("every state can be reset to PENDING (eviction must always be possible)", () => {
  for (const s of Object.values(NodeState)) assert.equal(canTransition(s, NodeState.PENDING), true, `${s} → PENDING`);
});

test("fade progresses over CROSSFADE_MS and clamps at both ends", () => {
  assert.equal(fadeProgress(-10), 0);
  assert.equal(fadeProgress(0), 0);
  assert.equal(fadeProgress(STREAM.CROSSFADE_MS / 2), 0.5);
  assert.equal(fadeProgress(STREAM.CROSSFADE_MS), 1);
  assert.equal(fadeProgress(99999), 1);
  const n = new TileNode({ id: { level: 1, x: 0, y: 0 } });
  n.transition(NodeState.REQUESTED);
  n.transition(NodeState.DECODING);
  n.transition(NodeState.UPLOADED);
  n.transition(NodeState.FADING, 1000);
  assert.equal(n.fade(1100), 0.5);
  n.transition(NodeState.RESIDENT);
  assert.equal(n.fade(1100), 1, "a resident node is fully faded in regardless of clock");
});

test("invalidate bumps the epoch and aborts in flight work", () => {
  const n = new TileNode({ id: { level: 1, x: 0, y: 0 } });
  const c = new AbortController();
  n.controller = c;
  const before = n.epoch;
  n.invalidate();
  assert.equal(n.epoch, before + 1);
  assert.equal(c.signal.aborted, true);
  assert.equal(n.controller, null);
});

test("level 0 is pinned, deeper levels are not (Law II.2)", () => {
  assert.equal(new TileNode({ id: { level: 0, x: 0, y: 0 } }).pinned, true);
  assert.equal(new TileNode({ id: { level: 1, x: 0, y: 0 } }).pinned, false);
});

// ── request priority ────────────────────────────────────────────────────────

test("requests go nearest-to-screen-centre first, then COARSE levels first", () => {
  const sorted = sortRequests([
    { key: "far-fine", level: 8, screenDistance: 500 },
    { key: "near-fine", level: 8, screenDistance: 10 },
    { key: "near-coarse", level: 3, screenDistance: 10 },
  ]);
  assert.deepEqual(
    sorted.map((s) => s.key),
    ["near-coarse", "near-fine", "far-fine"],
  );
});

test("request order is total and deterministic (no dependence on input order)", () => {
  const items = [
    { key: "b", level: 4, screenDistance: 10 },
    { key: "a", level: 4, screenDistance: 10 },
  ];
  assert.deepEqual(
    sortRequests(items).map((s) => s.key),
    ["a", "b"],
  );
  assert.deepEqual(
    sortRequests(items.slice().reverse()).map((s) => s.key),
    ["a", "b"],
  );
});

test("retry backoff is exponential with full jitter, and bounded", () => {
  const lo = retryDelayMs(0, () => 0);
  const hi = retryDelayMs(0, () => 1);
  assert.equal(lo, STREAM.RETRY_BASE_MS * 0.5);
  assert.equal(hi, STREAM.RETRY_BASE_MS);
  assert.equal(retryDelayMs(1, () => 1), STREAM.RETRY_BASE_MS * 2);
  assert.equal(retryDelayMs(2, () => 1), STREAM.RETRY_BASE_MS * 4);
  // Jitter must actually spread — a whole viewport failing at once must
  // not retry in lockstep.
  assert.ok(retryDelayMs(0, () => 0.25) !== retryDelayMs(0, () => 0.75));
});

// ── device tiering + budgets ────────────────────────────────────────────────

test("device tiering is capability-based and never sniffs a user agent", () => {
  assert.equal(isLowPowerDevice({ deviceMemoryGB: 8, maxTextureSize: 16384, hardwareConcurrency: 8 }), false);
  assert.equal(isLowPowerDevice({ deviceMemoryGB: 2, maxTextureSize: 16384, hardwareConcurrency: 8 }), true);
  assert.equal(isLowPowerDevice({ deviceMemoryGB: 8, maxTextureSize: 4096, hardwareConcurrency: 8 }), true);
  assert.equal(isLowPowerDevice({ deviceMemoryGB: 8, maxTextureSize: 16384, hardwareConcurrency: 4 }), true);
  // Absent hints (Safari/Firefox report no deviceMemory) must not force the
  // low tier — a wrong "low" permanently softens a capable machine.
  assert.equal(isLowPowerDevice({}), false);
});

test("a software rasterizer is low-power even with generous specs", () => {
  assert.equal(
    isLowPowerDevice({
      deviceMemoryGB: 16,
      maxTextureSize: 16384,
      hardwareConcurrency: 16,
      renderer: "Google SwiftShader",
      devicePixelRatio: 1,
    }),
    true,
  );
});

test("VRAM budgets and the tile arithmetic the work order sizes against", () => {
  assert.equal(vramBudgetBytes(true), 256 * 1024 * 1024);
  assert.equal(vramBudgetBytes(false), 768 * 1024 * 1024);
  const perTile = textureBytes(TILE.SIZE_PX, TILE.SIZE_PX, "compressed4bpp", true);
  assert.equal(Math.floor(vramBudgetBytes(true) / perTile), 1536, "~1,500 resident tiles on a 256MB mobile budget");
});

test("decode workers clamp to [2,4]", () => {
  assert.equal(decodeWorkerCount(1), 2);
  assert.equal(decodeWorkerCount(4), 3);
  assert.equal(decodeWorkerCount(8), 4);
  assert.equal(decodeWorkerCount(32), 4);
  assert.equal(decodeWorkerCount(), 4);
});

// ── eviction ────────────────────────────────────────────────────────────────

const ev = (key: string, over: Partial<Parameters<typeof evictionPlan>[0][number]> = {}) => ({
  key,
  bytes: 100,
  pinned: false,
  cameraDistance: 10,
  lastUsedMs: 0,
  resident: true,
  ...over,
});

test("under budget, nothing is evicted", () => {
  const plan = evictionPlan([ev("a"), ev("b")], 1000);
  assert.deepEqual(plan.evict, []);
  assert.equal(plan.bytesBefore, 200);
  assert.equal(plan.bytesAfter, 200);
});

test("over budget, the FARTHEST nodes go first and it undershoots to 90%", () => {
  const nodes = [ev("near", { cameraDistance: 1 }), ev("mid", { cameraDistance: 50 }), ev("far", { cameraDistance: 99 })];
  const plan = evictionPlan(nodes, 250); // 300 bytes resident, target 225
  assert.deepEqual(plan.evict, ["far"]);
  assert.equal(plan.bytesAfter, 200);
  assert.ok(plan.bytesAfter <= 250 * CACHE.EVICT_TO_FRACTION, "must undershoot, or the next upload re-triggers a pass");
});

test("the pinned base level is NEVER evicted, even when it is the farthest", () => {
  // Law II.2. Losing the base is how a region goes blank instead of soft.
  const nodes = [ev("base", { pinned: true, cameraDistance: 1e9, bytes: 500 }), ev("detail", { cameraDistance: 1 })];
  const plan = evictionPlan(nodes, 100);
  assert.ok(!plan.evict.includes("base"));
  assert.deepEqual(plan.evict, ["detail"]);
});

test("eviction never touches nodes that are still streaming (they hold no VRAM yet)", () => {
  const nodes = [ev("streaming", { resident: false, cameraDistance: 1e9 }), ev("resident", { bytes: 1000 })];
  const plan = evictionPlan(nodes, 500);
  assert.ok(!plan.evict.includes("streaming"));
  assert.equal(plan.bytesBefore, 1000, "a non-resident node contributes no bytes");
});

test("equidistant nodes break the tie on least-recently-used", () => {
  const nodes = [ev("fresh", { lastUsedMs: 900 }), ev("stale", { lastUsedMs: 100 })];
  const plan = evictionPlan(nodes, 150);
  assert.deepEqual(plan.evict, ["stale"]);
});

// ── upload budget ───────────────────────────────────────────────────────────

test("upload budget stops on count OR time, whichever hits first", () => {
  assert.equal(canUpload({ uploaded: 0, elapsedMs: 0 }), true);
  assert.equal(canUpload({ uploaded: STREAM.MAX_UPLOADS_PER_FRAME, elapsedMs: 0 }), false, "count cap");
  assert.equal(canUpload({ uploaded: 0, elapsedMs: STREAM.UPLOAD_BUDGET_MS }), false, "time cap");
  // The time cap is the one that matters: a driver stall on a single
  // compressed upload can blow the frame budget, and the count cap alone
  // would let it through.
  assert.equal(canUpload({ uploaded: 1, elapsedMs: 3.9 }), true);
  assert.equal(canUpload({ uploaded: 1, elapsedMs: 4.1 }), false);
});

// ── precision ───────────────────────────────────────────────────────────────

test("camera-relative rendering is required at 100m over a 1737km body", () => {
  // float32 ULP at the body's surface, in km. Anything above ~0.0001 km
  // (10cm) is visible jitter at 100m altitude.
  const ulpKm = float32PrecisionAt(MOON_RADIUS_KM);
  assert.ok(ulpKm > 1e-4, `float32 ULP at the surface is ${ulpKm} km — absolute coords would jitter`);
  // Camera-relative offsets are tiny, so their ULP is negligible.
  const relUlpKm = float32PrecisionAt(0.1);
  assert.ok(relUlpKm < 1e-7, `camera-relative ULP ${relUlpKm} km`);
});

test("relativeToEye subtracts the camera origin and can write into a reused buffer", () => {
  const out = new Float32Array(3);
  const r = relativeToEye([1737400.5, 10.25, -3.5], [1737400, 10, 0], out);
  assert.equal(r, out, "reuses the caller's buffer — no per-frame allocation");
  assert.ok(Math.abs(r[0] - 0.5) < 1e-6);
  assert.ok(Math.abs(r[1] - 0.25) < 1e-6);
  assert.ok(Math.abs(r[2] + 3.5) < 1e-6);
  assert.equal(float32PrecisionAt(0), Infinity);
});

// ── the streamer ────────────────────────────────────────────────────────────

class FakeBitmap implements ImageBitmapLike {
  closed = false;
  constructor(
    readonly width = TILE.SIZE_PX,
    readonly height = TILE.SIZE_PX,
  ) {}
  close() {
    this.closed = true;
  }
}

interface Rig {
  streamer: TileStreamer;
  bitmaps: FakeBitmap[];
  uploads: string[];
  releases: string[];
  failures: TileId[];
  /** Resolve the pending fetch/decode for a key. */
  settle(): Promise<void>;
  clock: { t: number };
  urls: string[];
}

function rig(over: { failKeys?: Set<string>; decodeDelayKeys?: Map<string, () => void> } = {}): Rig {
  const bitmaps: FakeBitmap[] = [];
  const uploads: string[] = [];
  const releases: string[] = [];
  const failures: TileId[] = [];
  const urls: string[] = [];
  const clock = { t: 0 };

  const source: TileSource = {
    id: "test",
    maxLevel: 8,
    bodyRadius: MOON_RADIUS_KM,
    url: (t) => `https://cdn.voltradeai.com/tiles/${tileKey(t)}.ktx2`,
  };
  const sink: TileSink = {
    upload(node) {
      uploads.push(node.key);
      return { tex: node.key };
    },
    release(node) {
      releases.push(node.key);
    },
  };

  const streamer = new TileStreamer({
    source,
    sink,
    now: () => clock.t,
    rand: () => 0.5,
    onFailure: (id) => failures.push(id),
    fetchImpl: async (url, init) => {
      urls.push(url);
      if (init.signal.aborted) throw new Error("aborted");
      const key = url.split("/tiles/")[1].replace(".ktx2", "");
      if (over.failKeys?.has(key)) return { ok: false, status: 500, blob: async () => ({}) };
      return { ok: true, status: 200, blob: async () => ({ key }) };
    },
    decodeImpl: async (blob) => {
      const key = (blob as { key: string }).key;
      const hold = over.decodeDelayKeys?.get(key);
      if (hold) await new Promise<void>((r) => over.decodeDelayKeys!.set(key, r as () => void));
      const b = new FakeBitmap();
      bitmaps.push(b);
      return b;
    },
  });

  return {
    streamer,
    bitmaps,
    uploads,
    releases,
    failures,
    clock,
    urls,
    // Drain the microtask queue enough times for fetch→blob→decode to land.
    async settle() {
      for (let i = 0; i < 12; i++) await Promise.resolve();
    },
  };
}

test("streamer: a selected tile streams through the full gate and becomes drawable only at FADING", async () => {
  resetMetrics();
  const r = rig();
  const id = { level: 2, x: 1, y: 1 };
  r.streamer.select({ wanted: [{ id, screenDistance: 0, cameraDistance: 100 }] });
  const node = r.streamer.node(id)!;
  assert.equal(node.state, NodeState.REQUESTED);
  assert.equal(node.drawable, false);

  await r.settle();
  assert.equal(node.state, NodeState.DECODING, "decoded but not yet uploaded — still not drawable");
  assert.equal(node.drawable, false);

  r.streamer.pumpUploads();
  assert.equal(node.state, NodeState.FADING);
  assert.equal(node.drawable, true);
  assert.deepEqual(r.uploads, ["2/1/1"]);

  r.clock.t += STREAM.CROSSFADE_MS;
  r.streamer.pumpUploads();
  assert.equal(node.state, NodeState.RESIDENT);
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: EVERY ImageBitmap is closed, including on the success path", async () => {
  // A retained bitmap is a GPU allocation the sink's release() knows
  // nothing about — invisible to dispose(), invisible to the leak harness
  // until the heap assertion fails.
  resetMetrics();
  const r = rig();
  r.streamer.select({ wanted: [{ id: { level: 1, x: 0, y: 1 }, screenDistance: 0, cameraDistance: 10 }] });
  await r.settle();
  r.streamer.pumpUploads();
  assert.equal(r.bitmaps.length, 1);
  assert.equal(r.bitmaps[0].closed, true, "bitmap must be closed after a successful upload");
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: a decode that lands AFTER its node merged is discarded, not painted", async () => {
  // The epoch guard, which is the real one: aborting the fetch does not
  // cancel an in-flight createImageBitmap.
  resetMetrics();
  const hold = new Map<string, () => void>([["2/0/0", () => {}]]);
  const r = rig({ decodeDelayKeys: hold });
  const id = { level: 2, x: 0, y: 0 };
  r.streamer.select({ wanted: [{ id, screenDistance: 0, cameraDistance: 10 }] });
  await r.settle();
  const node = r.streamer.node(id)!;
  assert.equal(node.state, NodeState.DECODING);

  // The camera pulls back: the subtree merges into its grandparent while
  // the decode is still running.
  r.streamer.merge({ level: 0, x: 0, y: 0 });
  assert.equal(r.streamer.node(id), undefined, "merged nodes leave the table");

  hold.get("2/0/0")!(); // the decode finally resolves
  await r.settle();
  r.streamer.pumpUploads();
  assert.deepEqual(r.uploads, [], "a superseded decode must never be uploaded");
  assert.equal(r.bitmaps[0].closed, true, "and its bitmap must still be closed");
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: in-flight requests are capped at MAX_INFLIGHT", async () => {
  resetMetrics();
  const hold = new Map<string, () => void>();
  const wanted = [];
  for (let i = 0; i < 20; i++) {
    hold.set(`3/${i}/0`, () => {});
    wanted.push({ id: { level: 3, x: i, y: 0 }, screenDistance: i, cameraDistance: 10 });
  }
  const r = rig({ decodeDelayKeys: hold });
  r.streamer.select({ wanted });
  assert.equal(r.streamer.inflightCount, STREAM.MAX_INFLIGHT);
  assert.equal(r.urls.length, STREAM.MAX_INFLIGHT);
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: uploads are budgeted — no more than MAX_UPLOADS_PER_FRAME land per frame", async () => {
  resetMetrics();
  const r = rig();
  const wanted = [0, 1, 2, 3, 4].map((i) => ({
    id: { level: 3, x: i, y: 0 },
    screenDistance: i,
    cameraDistance: 10,
  }));
  r.streamer.select({ wanted });
  await r.settle();
  assert.equal(r.streamer.pumpUploads(), STREAM.MAX_UPLOADS_PER_FRAME);
  assert.equal(r.uploads.length, 2);
  assert.equal(r.streamer.pumpUploads(), STREAM.MAX_UPLOADS_PER_FRAME);
  assert.equal(r.uploads.length, 4);
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: requests come from OUR CDN, never an upstream WMTS (Law II.8)", async () => {
  resetMetrics();
  const r = rig();
  r.streamer.select({ wanted: [{ id: { level: 1, x: 0, y: 0 }, screenDistance: 0, cameraDistance: 10 }] });
  assert.ok(r.urls.length > 0);
  for (const u of r.urls) assert.match(u, /^https:\/\/cdn\.voltradeai\.com\//);
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: drawableFor falls back to the nearest ready ANCESTOR with the right UV subset", async () => {
  resetMetrics();
  const r = rig();
  const root = { level: 0, x: 0, y: 0 };
  r.streamer.select({ wanted: [{ id: root, screenDistance: 0, cameraDistance: 10 }] });
  await r.settle();
  r.streamer.pumpUploads();

  // A deep child nobody has fetched yet: the layer still gets something
  // safe to draw, which is the entire parent-hold mechanism.
  const child = { level: 2, x: 3, y: 1 };
  const got = r.streamer.drawableFor(child)!;
  assert.equal(got.node.key, "0/0/0");
  assert.deepEqual(got.uv, { scale: 0.25, offsetU: 0.75, offsetV: 0.25 });

  // With nothing ready at all, it returns null rather than an unready node.
  const r2 = rig();
  assert.equal(r2.streamer.drawableFor(child), null, "never hand back an unready node");
  r.streamer.dispose();
  r2.streamer.dispose();
  resetMetrics();
});

test("streamer: a failing tile retries MAX_RETRIES times then reports loudly (Law V)", async () => {
  resetMetrics();
  const errs = console.error;
  console.error = () => {};
  try {
    const r = rig({ failKeys: new Set(["1/1/1"]) });
    const id = { level: 1, x: 1, y: 1 };
    r.streamer.select({ wanted: [{ id, screenDistance: 0, cameraDistance: 10 }] });
    // Retries are timer-driven; drive them by hand.
    for (let i = 0; i < 4; i++) {
      await r.settle();
      await new Promise((res) => setTimeout(res, 5));
      const n = r.streamer.node(id)!;
      if (n.state === NodeState.PENDING) await (r.streamer as never as { request(n: unknown): Promise<void> }).request(n);
    }
    await r.settle();
    const node = r.streamer.node(id)!;
    assert.equal(node.state, NodeState.FAILED);
    assert.equal(node.attempts, STREAM.MAX_RETRIES + 1, "one initial try plus MAX_RETRIES");
    assert.equal(r.failures.length, 1, "failure is reported, never silent");
    r.streamer.dispose();
  } finally {
    console.error = errs;
  }
  resetMetrics();
});

test("streamer: eviction releases the sink payload and resets the node", async () => {
  resetMetrics();
  const r = rig();
  const id = { level: 3, x: 5, y: 5 };
  r.streamer.select({ wanted: [{ id, screenDistance: 0, cameraDistance: 10 }] });
  await r.settle();
  r.streamer.pumpUploads();
  const node = r.streamer.node(id)!;
  const epochBefore = node.epoch;

  r.streamer.evict(node.key);
  assert.deepEqual(r.releases, ["3/5/5"]);
  assert.equal(node.payload, null);
  assert.equal(node.state, NodeState.PENDING);
  assert.equal(node.epoch, epochBefore + 1, "eviction must bump the epoch or in-flight work can resurrect it");
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: the pinned base level survives an evict() call", async () => {
  resetMetrics();
  const r = rig();
  const id = { level: 0, x: 0, y: 0 };
  r.streamer.select({ wanted: [{ id, screenDistance: 0, cameraDistance: 1e9 }] });
  await r.settle();
  r.streamer.pumpUploads();
  r.streamer.evict("0/0/0");
  assert.deepEqual(r.releases, [], "the base level is exempt from eviction");
  assert.equal(r.streamer.node(id)!.drawable, true);
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: dispose is total — payloads released, queued bitmaps closed, table empty", async () => {
  resetMetrics();
  const r = rig();
  const wanted = [0, 1, 2, 3].map((i) => ({
    id: { level: 3, x: i, y: 0 },
    screenDistance: i,
    cameraDistance: 10,
  }));
  r.streamer.select({ wanted });
  await r.settle();
  r.streamer.pumpUploads(); // uploads 2, leaves 2 queued
  assert.equal(r.uploads.length, 2);

  r.streamer.dispose();
  assert.equal(r.streamer.nodes.size, 0);
  assert.equal(r.releases.length, 2, "every uploaded payload released");
  for (const b of r.bitmaps) assert.equal(b.closed, true, "queued bitmaps closed too, not just uploaded ones");
  assert.equal(getGauge(METRIC.VRAM_BYTES), 0);
  assert.equal(getGauge(METRIC.TILES_RESIDENT), 0);
  assert.equal(getGauge(METRIC.IN_FLIGHT), 0);
  resetMetrics();
});

test("streamer: work started before dispose cannot land after it", async () => {
  resetMetrics();
  const hold = new Map<string, () => void>([["2/1/0", () => {}]]);
  const r = rig({ decodeDelayKeys: hold });
  r.streamer.select({ wanted: [{ id: { level: 2, x: 1, y: 0 }, screenDistance: 0, cameraDistance: 10 }] });
  await r.settle();
  r.streamer.dispose();
  hold.get("2/1/0")!();
  await r.settle();
  assert.deepEqual(r.uploads, []);
  assert.equal(r.bitmaps[0].closed, true);
  resetMetrics();
});

test("streamer: publishes the gauges the perf HUD reads", async () => {
  resetMetrics();
  const r = rig();
  r.streamer.select({ wanted: [{ id: { level: 2, x: 2, y: 2 }, screenDistance: 0, cameraDistance: 10 }] });
  assert.equal(getGauge(METRIC.TILES_PENDING), 1);
  assert.equal(getGauge(METRIC.TILES_RESIDENT), 0);
  await r.settle();
  r.streamer.pumpUploads();
  assert.equal(getGauge(METRIC.TILES_RESIDENT), 1);
  assert.equal(getGauge(METRIC.TILES_PENDING), 0);
  assert.ok(getGauge(METRIC.VRAM_BYTES) > 0);
  assert.equal(getCounter(METRIC.TILES_UPLOADED), 1);
  r.streamer.dispose();
  resetMetrics();
});

test("streamer: selecting the same tile twice does not double-request it", async () => {
  resetMetrics();
  const hold = new Map<string, () => void>([["4/1/1", () => {}]]);
  const r = rig({ decodeDelayKeys: hold });
  const w = { id: { level: 4, x: 1, y: 1 }, screenDistance: 0, cameraDistance: 10 };
  r.streamer.select({ wanted: [w] });
  r.streamer.select({ wanted: [w] });
  r.streamer.select({ wanted: [w] });
  assert.equal(r.urls.length, 1, "an in-flight node must not be re-requested every frame");
  r.streamer.dispose();
  resetMetrics();
});

// ── the VRAM-accounting default (found 2026-08-12 by the moon-bake recon) ──
// PR2 defaulted bytesPerTile to the COMPRESSED size. That is the optimistic
// assumption, and for a memory budget the failure modes are not symmetric:
// over-estimating evicts a little early (harmless); under-estimating makes
// evictionPlan believe it has headroom it does not have, so it never evicts
// and the layer blows the very budget it exists to enforce. On a 256MB
// mobile budget an 8x under-report is a context loss, not a slowdown.

test("the default tile size assumption is PESSIMISTIC, not optimistic", () => {
  const src: TileSource = {
    id: "undeclared",
    maxLevel: 8,
    bodyRadius: MOON_RADIUS_KM,
    url: (t) => `https://cdn.voltradeai.com/${tileKey(t)}.jpg`,
    // deliberately does NOT declare bytesPerTile
  };
  const sink: TileSink = { upload: () => ({}), release: () => {} };
  const s = new TileStreamer({
    source: src,
    sink,
    now: () => 0,
    fetchImpl: async () => ({ ok: true, status: 200, blob: async () => ({}) }),
    decodeImpl: async () => new FakeBitmap(),
  });
  s.select({ wanted: [{ id: { level: 1, x: 0, y: 0 }, screenDistance: 0, cameraDistance: 10 }] });
  const node = s.node({ level: 1, x: 0, y: 0 })!;
  assert.equal(
    node.bytes,
    uncompressedTileBytes(TILE.SIZE_PX),
    "an undeclared source must be budgeted as RGBA8, the worst case — not as compressed",
  );
  assert.equal(node.bytes, compressedTileBytes(TILE.SIZE_PX) * 8, "RGBA8 is exactly 8x a 4bpp compressed tile");
  s.dispose();
});

test("a source that declares its size is trusted", () => {
  const src: TileSource = {
    id: "declared",
    maxLevel: 8,
    bodyRadius: MOON_RADIUS_KM,
    url: (t) => `https://cdn.voltradeai.com/${tileKey(t)}.ktx2`,
    bytesPerTile: () => compressedTileBytes(TILE.SIZE_PX),
  };
  const sink: TileSink = { upload: () => ({}), release: () => {} };
  const s = new TileStreamer({
    source: src,
    sink,
    now: () => 0,
    fetchImpl: async () => ({ ok: true, status: 200, blob: async () => ({}) }),
    decodeImpl: async () => new FakeBitmap(),
  });
  s.select({ wanted: [{ id: { level: 2, x: 1, y: 1 }, screenDistance: 0, cameraDistance: 10 }] });
  assert.equal(s.node({ level: 2, x: 1, y: 1 })!.bytes, compressedTileBytes(TILE.SIZE_PX));
  s.dispose();
});

test("the mobile budget holds ~8x fewer uncompressed tiles — the number the Law is about", () => {
  // Law II.7's rationale, quantified. 256MB / 175KB compressed = 1,536
  // tiles; / 1.4MB uncompressed = 192. This is why the format matters, and
  // why mis-declaring it silently breaks the budget.
  const budget = 256 * 1024 * 1024;
  assert.equal(Math.floor(budget / compressedTileBytes(TILE.SIZE_PX)), 1536);
  assert.equal(Math.floor(budget / uncompressedTileBytes(TILE.SIZE_PX)), 192);
});
