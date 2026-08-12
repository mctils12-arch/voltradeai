// tileCore — THE RASTER STREAMER (Rendering & Motion Law, Law II).
//
// Source-agnostic on purpose. The Moon surface, the Earth base, NOAA radar
// and any future image-tile layer plug in through `TileSource` (where the
// bytes come from) and `TileSink` (what "uploaded" means for that
// consumer). This repo has two shapes of consumer and both must work:
//   * a WebGL2 custom layer, where upload is texImage2D into a texture;
//   * the CPU-rasterized Moon path, where "upload" is a blit into the
//     stitched mosaic that moonSurface samples per pixel.
// Nothing in this module knows which — it owns WHEN a tile may be shown,
// never HOW it is drawn.
//
// The single idea the whole module exists to enforce is the READY-GATE:
//
//   PENDING → REQUESTED → DECODING → UPLOADED → FADING → RESIDENT
//
// A node is drawable ONLY in FADING or RESIDENT. Until a child reaches
// FADING its parent stays visible and is UV-subset-sampled for the child's
// footprint. That is the difference between "a softer picture for 200ms"
// and "a fuzzy picture that later snaps sharp" — the moon fuzz bug.
//
// The second idea is the EPOCH GUARD. Aborting a fetch does not cancel an
// in-flight createImageBitmap, and a decode that lands after its node was
// merged or evicted will happily paint onto a node that no longer exists.
// Every node carries an epoch, bumped on merge/eviction; every async
// continuation re-checks it and closes the bitmap instead of using it.

import { classifyDevice } from "../lib/deviceTier.ts";
import { METRIC, bump, setGauge, textureBytes } from "./perfMetrics.ts";

/** Tile geometry + LOD constants. Law, not tune values. */
export const TILE = {
  SIZE_PX: 512, // 1/4 the request count of 256
  SSE_SPLIT_PX: 2.0,
  SSE_MERGE_PX: 1.0, // hysteresis gap prevents split/merge thrash
  FRUSTUM_PAD_TILES: 1,
  PREFETCH_LEVELS_AHEAD: 1,
  SKIRT_DEPTH_FACTOR: 2.0, // × geometricError, seals LOD cracks
} as const;

/** Network + upload budgets. */
export const STREAM = {
  MAX_INFLIGHT: 8,
  MAX_UPLOADS_PER_FRAME: 2,
  UPLOAD_BUDGET_MS: 4, // whichever limit hits first
  DECODE_WORKERS: 3, // clamp(hardwareConcurrency-1, 2, 4)
  CROSSFADE_MS: 200, // valid range 150–300
  REQUEST_TIMEOUT_MS: 15000,
  MAX_RETRIES: 2,
  RETRY_BASE_MS: 400, // exponential + jitter
} as const;

/** Residency budgets. */
export const CACHE = {
  VRAM_BUDGET_MB_MOBILE: 256,
  VRAM_BUDGET_MB_DESKTOP: 768,
  EVICT_TO_FRACTION: 0.9, // when over budget, evict down to 90%
  PINNED_BASE_LEVEL: 0, // never evicted
} as const;

// ── tile identity + quadtree (pure) ─────────────────────────────────────────

export interface TileId {
  level: number;
  x: number;
  y: number;
}

/** Stable string key. Used for cache maps and for test readability. */
export function tileKey(t: TileId): string {
  return `${t.level}/${t.x}/${t.y}`;
}

export function parseTileKey(key: string): TileId | null {
  const m = /^(\d+)\/(\d+)\/(\d+)$/.exec(key);
  if (!m) return null;
  return { level: +m[1], x: +m[2], y: +m[3] };
}

export function parentOf(t: TileId): TileId | null {
  if (t.level <= 0) return null;
  return { level: t.level - 1, x: t.x >> 1, y: t.y >> 1 };
}

export function childrenOf(t: TileId): TileId[] {
  const l = t.level + 1;
  const x = t.x * 2;
  const y = t.y * 2;
  return [
    { level: l, x, y },
    { level: l, x: x + 1, y },
    { level: l, x, y: y + 1 },
    { level: l, x: x + 1, y: y + 1 },
  ];
}

export function isAncestorOf(ancestor: TileId, node: TileId): boolean {
  if (ancestor.level >= node.level) return false;
  const shift = node.level - ancestor.level;
  return node.x >> shift === ancestor.x && node.y >> shift === ancestor.y;
}

export interface UVSubset {
  /** Scale applied to the descendant's [0,1] UVs. */
  scale: number;
  offsetU: number;
  offsetV: number;
}

/**
 * Where `node` sits inside `ancestor`, as a UV sub-rectangle.
 *
 * This is what lets the parent KEEP DRAWING the child's footprint at lower
 * resolution while the child streams — Law II.1's "parent stays visible"
 * clause, implemented. Without it the only options are a blank hole or an
 * unready tile, and both are the bug.
 */
export function uvSubsetIn(ancestor: TileId, node: TileId): UVSubset | null {
  if (tileKey(ancestor) === tileKey(node)) return { scale: 1, offsetU: 0, offsetV: 0 };
  if (!isAncestorOf(ancestor, node)) return null;
  const shift = node.level - ancestor.level;
  const span = 1 << shift;
  const scale = 1 / span;
  return {
    scale,
    offsetU: (node.x - (ancestor.x << shift)) * scale,
    offsetV: (node.y - (ancestor.y << shift)) * scale,
  };
}

// ── screen-space error (pure) ───────────────────────────────────────────────

/**
 * Geometric error of the root tile for a body, in metres-per-... whatever
 * unit `bodyRadius` is in. The full circumference divided across one tile's
 * pixels: the ground distance a single texel of the root tile covers.
 */
export function baseGeometricError(bodyRadius: number, tileSizePx = TILE.SIZE_PX): number {
  return (2 * Math.PI * bodyRadius) / tileSizePx;
}

/** Geometric error at a level — halves every level down. */
export function geometricError(level: number, bodyRadius: number, tileSizePx = TILE.SIZE_PX): number {
  return baseGeometricError(bodyRadius, tileSizePx) / Math.pow(2, level);
}

/**
 * Screen-space error, in pixels, of a node at `distance` from the camera.
 *
 *   sse = (geometricError * viewportHeightPx) / (distance * 2 * tan(fov/2))
 *
 * This is the ONLY LOD criterion in this module. Integer zoom levels are
 * never consulted: a zoom level is a property of a tile pyramid, not of
 * what the eye can resolve, and keying LOD off it is what makes tiles pop
 * at zoom-step boundaries instead of resolving continuously.
 */
export function screenSpaceError(
  geometricErrorAtLevel: number,
  distance: number,
  viewportHeightPx: number,
  fovRad: number,
): number {
  if (!(distance > 0) || !(viewportHeightPx > 0) || !(fovRad > 0)) return Infinity;
  const denom = distance * 2 * Math.tan(fovRad / 2);
  if (!(denom > 0)) return Infinity;
  return (geometricErrorAtLevel * viewportHeightPx) / denom;
}

export type LodAction = "split" | "merge" | "hold";

/**
 * Split/merge with MANDATORY hysteresis.
 *
 * A single threshold makes a node at exactly the boundary split, which
 * lowers its sse, which makes it merge, which raises its sse — the camera
 * sits still and the layer thrashes. The gap between SSE_MERGE_PX and
 * SSE_SPLIT_PX is what makes that impossible.
 */
export function lodDecision(sse: number, splitPx = TILE.SSE_SPLIT_PX, mergePx = TILE.SSE_MERGE_PX): LodAction {
  if (!Number.isFinite(sse)) return "hold";
  if (sse > splitPx) return "split";
  if (sse < mergePx) return "merge";
  return "hold";
}

/** Skirt depth for a node — extruded inward to seal T-junction cracks
 *  between adjacent tiles at differing levels. Sized off the geometric
 *  error so it always covers the worst-case height discontinuity, and
 *  generated WITH the geometry rather than patched afterwards. */
export function skirtDepth(geometricErrorAtLevel: number, factor = TILE.SKIRT_DEPTH_FACTOR): number {
  return Math.max(0, geometricErrorAtLevel * factor);
}

// ── node lifecycle ──────────────────────────────────────────────────────────

/**
 * The ready-gate, as a type. Order is meaningful: `DRAWABLE_FROM` is the
 * first state a node may be rendered in.
 */
export const NodeState = {
  PENDING: "PENDING",
  REQUESTED: "REQUESTED",
  DECODING: "DECODING",
  UPLOADED: "UPLOADED",
  FADING: "FADING",
  RESIDENT: "RESIDENT",
  FAILED: "FAILED",
} as const;

export type NodeStateName = (typeof NodeState)[keyof typeof NodeState];

/**
 * May this state be drawn? FADING and RESIDENT only.
 *
 * UPLOADED is deliberately NOT drawable: the texture exists but the
 * crossfade has not started, so drawing it would be the hard pop the fade
 * exists to remove. The harness asserts zero draws outside this set.
 */
export function isDrawable(state: NodeStateName): boolean {
  return state === NodeState.FADING || state === NodeState.RESIDENT;
}

/** Legal transitions. A layer that jumps the gate fails here rather than
 *  producing a frame nobody can explain. */
const LEGAL_NEXT: Record<NodeStateName, readonly NodeStateName[]> = {
  PENDING: [NodeState.REQUESTED, NodeState.FAILED],
  REQUESTED: [NodeState.DECODING, NodeState.FAILED],
  DECODING: [NodeState.UPLOADED, NodeState.FAILED],
  UPLOADED: [NodeState.FADING],
  FADING: [NodeState.RESIDENT],
  RESIDENT: [],
  FAILED: [],
};

export function canTransition(from: NodeStateName, to: NodeStateName): boolean {
  // Reset to PENDING is legal from EVERY state, including PENDING itself.
  // Eviction and merge can hit a node at any point in its life and must
  // never be refused — a node that cannot be reset is a node that cannot
  // be freed, which is a leak with a state machine in front of it.
  if (to === NodeState.PENDING) return true;
  return LEGAL_NEXT[from].includes(to);
}

/** Crossfade progress in [0,1] over STREAM.CROSSFADE_MS. Drives the single
 *  `uFade` uniform of a two-sampler shader. NOT two draw passes: drawing
 *  parent and child separately z-fights and doubles overdraw at exactly
 *  the moment the frame is most fill-bound. */
export function fadeProgress(elapsedMs: number, durationMs = STREAM.CROSSFADE_MS): number {
  if (!(durationMs > 0)) return 1;
  if (!(elapsedMs > 0)) return 0;
  return Math.min(1, elapsedMs / durationMs);
}

export interface TileNodeInit {
  id: TileId;
  /** Bytes this node will occupy once uploaded — for the VRAM budget. */
  bytes?: number;
}

export class TileNode {
  readonly id: TileId;
  readonly key: string;
  state: NodeStateName = NodeState.PENDING;

  /**
   * Bumped whenever this node is merged, evicted, or otherwise reset.
   * Every async continuation captures the epoch it started under and
   * discards its result if the epoch moved. This — not AbortController —
   * is the real guard: aborting a fetch does NOT cancel an in-flight
   * createImageBitmap, so the decode can still resolve after the node is
   * gone.
   */
  epoch = 0;

  /** Whatever the sink handed back for this node (a WebGLTexture, a canvas
   *  region handle, …). tileCore never inspects it. */
  payload: unknown = null;

  bytes = 0;
  attempts = 0;
  controller: AbortController | null = null;

  /** Distance from the camera at the last selection pass — the LRU key. */
  cameraDistance = Infinity;
  /** Distance from screen centre — the request priority key. */
  screenDistance = Infinity;
  /** frameCore time the fade began. */
  fadeStartMs = 0;
  /** frameCore time this node was last selected — LRU tiebreak. */
  lastUsedMs = 0;

  constructor(init: TileNodeInit) {
    this.id = init.id;
    this.key = tileKey(init.id);
    this.bytes = init.bytes ?? 0;
  }

  get drawable(): boolean {
    return isDrawable(this.state);
  }

  get pinned(): boolean {
    return this.id.level <= CACHE.PINNED_BASE_LEVEL;
  }

  /** Transition, refusing anything that jumps the ready-gate. */
  transition(to: NodeStateName, nowMs = 0): boolean {
    if (!canTransition(this.state, to)) return false;
    this.state = to;
    if (to === NodeState.FADING) this.fadeStartMs = nowMs;
    return true;
  }

  /** Invalidate: bump the epoch and abort in-flight work. Anything that
   *  lands after this is discarded by its own epoch check. */
  invalidate(): void {
    this.epoch++;
    this.controller?.abort();
    this.controller = null;
  }

  fade(nowMs: number): number {
    if (this.state === NodeState.RESIDENT) return 1;
    if (this.state !== NodeState.FADING) return 0;
    return fadeProgress(nowMs - this.fadeStartMs);
  }
}

// ── request priority (pure) ─────────────────────────────────────────────────

export interface RequestCandidate {
  key: string;
  level: number;
  screenDistance: number;
}

/**
 * Request order: nearest to screen centre first, then COARSE LEVELS FIRST.
 *
 * The level tiebreak is the important half. Coarse-first means there is
 * always something to show for a region before its detail arrives, which
 * is what makes the parent-hold in Law II.1 possible at all. Fine-first
 * looks faster on a fast connection and produces holes on a slow one.
 */
export function sortRequests<T extends RequestCandidate>(candidates: T[]): T[] {
  return candidates.slice().sort((a, b) => {
    if (a.screenDistance !== b.screenDistance) return a.screenDistance - b.screenDistance;
    if (a.level !== b.level) return a.level - b.level;
    return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
  });
}

/** Exponential backoff with jitter. `rand` is injectable so the test is
 *  deterministic — Math.random in a retry path is otherwise untestable. */
export function retryDelayMs(attempt: number, rand: () => number = Math.random, baseMs = STREAM.RETRY_BASE_MS): number {
  const exp = baseMs * Math.pow(2, Math.max(0, attempt));
  // Full jitter: uniform in [exp/2, exp]. Halves the thundering herd when a
  // whole viewport's worth of tiles fails at once (a CDN blip).
  return exp * 0.5 * (1 + Math.max(0, Math.min(1, rand())));
}

// ── budgets + eviction (pure) ───────────────────────────────────────────────

export interface DeviceCaps {
  /** navigator.deviceMemory, GB. Absent on Safari/Firefox. */
  deviceMemoryGB?: number;
  /** gl.getParameter(gl.MAX_TEXTURE_SIZE). */
  maxTextureSize?: number;
  hardwareConcurrency?: number;
  /** UNMASKED_RENDERER_WEBGL where available — used only to detect a
   *  software rasterizer, never to identify a device model. */
  renderer?: string;
  devicePixelRatio?: number;
}

/**
 * Low-power classification from CAPABILITY, never from user-agent.
 * WEBGL_debug_renderer_info is deprecated and blocked in more browsers
 * every year, so `renderer` is optional and only ever narrows.
 *
 * Chosen ONCE at load — a budget that flips mid-session would evict half
 * the cache at the worst possible moment.
 */
export function isLowPowerDevice(caps: DeviceCaps): boolean {
  const mem = caps.deviceMemoryGB ?? 4;
  const maxTex = caps.maxTextureSize ?? 8192;
  const cores = caps.hardwareConcurrency ?? 8;
  if (mem < 4 || maxTex < 8192 || cores <= 4) return true;
  if (caps.renderer) {
    const tier = classifyDevice({
      renderer: caps.renderer,
      deviceMemoryGB: caps.deviceMemoryGB,
      cores: caps.hardwareConcurrency,
      devicePixelRatio: caps.devicePixelRatio ?? 1,
    });
    if (tier.tier !== "full") return true;
  }
  return false;
}

/** VRAM budget in bytes for this device. */
export function vramBudgetBytes(lowPower: boolean): number {
  return (lowPower ? CACHE.VRAM_BUDGET_MB_MOBILE : CACHE.VRAM_BUDGET_MB_DESKTOP) * 1024 * 1024;
}

/** Decode-worker count, clamped per the work order. */
export function decodeWorkerCount(hardwareConcurrency = 8): number {
  return Math.max(2, Math.min(4, hardwareConcurrency - 1));
}

export interface EvictableNode {
  key: string;
  bytes: number;
  pinned: boolean;
  cameraDistance: number;
  lastUsedMs: number;
  /** Nodes still streaming are not evictable — they hold no VRAM yet. */
  resident: boolean;
}

export interface EvictionPlan {
  evict: string[];
  bytesBefore: number;
  bytesAfter: number;
  budgetBytes: number;
}

/**
 * LRU-by-camera-distance eviction, down to EVICT_TO_FRACTION of budget.
 *
 * Evicting exactly to the budget means the very next uploaded tile pushes
 * over again and triggers another pass — the cache spends the whole
 * gesture evicting. Undershooting to 90% buys headroom for a frame's worth
 * of uploads.
 *
 * The pinned base level is EXEMPT (Law II.2). Losing it is how a region
 * goes blank instead of soft, which is the one outcome the law forbids.
 */
export function evictionPlan(nodes: readonly EvictableNode[], budgetBytes: number): EvictionPlan {
  let total = 0;
  for (const n of nodes) if (n.resident) total += n.bytes;
  const plan: EvictionPlan = { evict: [], bytesBefore: total, bytesAfter: total, budgetBytes };
  if (total <= budgetBytes) return plan;

  const target = budgetBytes * CACHE.EVICT_TO_FRACTION;
  const candidates = nodes
    .filter((n) => n.resident && !n.pinned)
    // Farthest from the camera goes first; ties break on least-recently
    // selected, so two equidistant nodes evict the stale one.
    .sort((a, b) => b.cameraDistance - a.cameraDistance || a.lastUsedMs - b.lastUsedMs);

  for (const n of candidates) {
    if (total <= target) break;
    plan.evict.push(n.key);
    total -= n.bytes;
  }
  plan.bytesAfter = total;
  return plan;
}

// ── upload budgeting (pure) ─────────────────────────────────────────────────

export interface UploadBudgetState {
  uploaded: number;
  elapsedMs: number;
}

/**
 * May another texture upload run this frame? Capped by BOTH count and
 * time, whichever hits first — a 512² upload is usually sub-millisecond,
 * but a driver stall on a compressed-format upload can blow the whole
 * frame budget on one, and the count cap alone would not catch it.
 */
export function canUpload(
  st: UploadBudgetState,
  maxPerFrame = STREAM.MAX_UPLOADS_PER_FRAME,
  budgetMs = STREAM.UPLOAD_BUDGET_MS,
): boolean {
  return st.uploaded < maxPerFrame && st.elapsedMs < budgetMs;
}

// ── camera-relative rendering (pure) ────────────────────────────────────────

/**
 * float32 has ~7 decimal digits. At 100m altitude over a 1737.4km body,
 * a world coordinate needs ~10 digits to place a vertex sub-pixel — so
 * uploading absolute world positions as float32 makes the surface jitter
 * visibly as the camera moves. The fix is not optional: keep positions in
 * double on the CPU, subtract the camera origin, upload the (small) offset
 * as float32.
 */
export function relativeToEye(
  position: readonly [number, number, number],
  cameraOrigin: readonly [number, number, number],
  out?: Float32Array,
): Float32Array {
  const o = out ?? new Float32Array(3);
  o[0] = position[0] - cameraOrigin[0];
  o[1] = position[1] - cameraOrigin[1];
  o[2] = position[2] - cameraOrigin[2];
  return o;
}

/** Digits of float32 precision left at a given distance from the origin —
 *  a diagnostic for deciding whether camera-relative is needed at all. */
export function float32PrecisionAt(magnitude: number): number {
  if (!(magnitude > 0)) return Infinity;
  // 24-bit mantissa: the ULP at `magnitude` is magnitude * 2^-23.
  return magnitude * Math.pow(2, -23);
}

// ── source + sink interfaces ────────────────────────────────────────────────

export interface TileSource {
  /** Human name, for logs and the HUD. */
  readonly id: string;
  /** URL for a tile. MUST point at our own CDN — Law II.8: the runtime
   *  never touches an upstream WMTS. */
  url(t: TileId): string;
  /** Deepest level this source publishes. */
  readonly maxLevel: number;
  /** Body radius, for the geometric-error scale. */
  readonly bodyRadius: number;
  /** Bytes a tile of this source occupies once uploaded. Defaults to a
   *  4bpp compressed 512² with mips. */
  /**
   * VRAM a single uploaded tile of this source occupies, in bytes.
   *
   * DECLARE THIS. The default is deliberately PESSIMISTIC (uncompressed
   * RGBA8 + mips) because this number is the only input to the Law IV
   * eviction budget, and the failure modes are not symmetric:
   *   - over-estimate  -> evict sooner than strictly necessary (a little
   *     extra re-fetching, no user-visible break)
   *   - under-estimate -> evictionPlan believes it has headroom it does
   *     not have, never evicts, and the layer blows the VRAM budget it
   *     was written to enforce. On a 256MB mobile budget an 8x
   *     under-report is a context loss, not a slowdown.
   * A source that genuinely ships compressed tiles should say so via
   * `compressedTileBytes()`.
   */
  bytesPerTile?(): number;
}

/** Convenience for a source that ships 4bpp compressed tiles WITH baked
 *  mips (ETC1S / BC1 / ETC2-RGB). Use this rather than hand-computing, so
 *  the mip factor cannot be forgotten. */
export function compressedTileBytes(tileSizePx = TILE.SIZE_PX): number {
  return textureBytes(tileSizePx, tileSizePx, "compressed4bpp", true);
}

/** Convenience for a source that ships JPEG/PNG decoded to RGBA8 with
 *  runtime-generated mips — the no-Basis fallback path. */
export function uncompressedTileBytes(tileSizePx = TILE.SIZE_PX): number {
  return textureBytes(tileSizePx, tileSizePx, "rgba8", true);
}

export interface TileSink {
  /**
   * Make the decoded image available to the renderer. Whatever this
   * returns becomes `node.payload`. Synchronous and budgeted: it runs on
   * the main thread inside the per-frame upload budget.
   */
  upload(node: TileNode, bitmap: ImageBitmapLike): unknown;
  /** Free whatever `upload` allocated. Called on eviction and teardown. */
  release(node: TileNode): void;
}

/** The subset of ImageBitmap this module uses — so tests need no DOM. */
export interface ImageBitmapLike {
  readonly width: number;
  readonly height: number;
  close(): void;
}

export interface FetchLike {
  (url: string, init: { signal: AbortSignal }): Promise<{ ok: boolean; status: number; blob(): Promise<unknown> }>;
}

export interface DecodeLike {
  (blob: unknown, signal: AbortSignal): Promise<ImageBitmapLike>;
}

// ── the streamer ────────────────────────────────────────────────────────────

export interface TileStreamerOptions {
  source: TileSource;
  sink: TileSink;
  fetchImpl: FetchLike;
  decodeImpl: DecodeLike;
  /** Monotonic clock — pass frameCore().now so there is ONE clock. */
  now: () => number;
  lowPower?: boolean;
  maxInflight?: number;
  rand?: () => number;
  /** Called when the chain gives up on a tile. Law V: loud, not silent. */
  onFailure?: (id: TileId, err: unknown) => void;
}

export interface SelectionInput {
  /** Nodes the camera wants this frame, nearest-first is not required. */
  wanted: { id: TileId; screenDistance: number; cameraDistance: number }[];
}

/**
 * Owns the node table, the request queue, the upload budget and the
 * eviction pass. Deliberately does NOT own a rAF loop: it exposes
 * `beginFrame`/`pumpUploads` for a STREAM-priority frameCore callback, so
 * there stays exactly one loop per view (Law I).
 */
export class TileStreamer {
  readonly nodes = new Map<string, TileNode>();
  private readonly opts: TileStreamerOptions;
  private readonly bytesPerTile: number;
  private readonly budgetBytes: number;
  private readonly maxInflight: number;

  /** Decoded and waiting for a budgeted upload slot. */
  private readonly uploadQueue: { node: TileNode; bitmap: ImageBitmapLike; epoch: number }[] = [];
  private inflight = 0;
  private disposed = false;

  constructor(opts: TileStreamerOptions) {
    this.opts = opts;
    // Pessimistic default — see TileSource.bytesPerTile. Assuming the
    // COMPRESSED size here (as this originally did) under-reports an
    // uncompressed source by 8x and silently disables the eviction budget.
    this.bytesPerTile = opts.source.bytesPerTile?.() ?? uncompressedTileBytes(TILE.SIZE_PX);
    this.budgetBytes = vramBudgetBytes(opts.lowPower ?? false);
    this.maxInflight = opts.maxInflight ?? STREAM.MAX_INFLIGHT;
  }

  get inflightCount(): number {
    return this.inflight;
  }

  get residentBytes(): number {
    let n = 0;
    this.nodes.forEach((node) => {
      if (node.drawable) n += node.bytes;
    });
    return n;
  }

  node(id: TileId): TileNode | undefined {
    return this.nodes.get(tileKey(id));
  }

  /**
   * The best DRAWABLE stand-in for `id`: the node itself if ready, else
   * the nearest ready ancestor plus the UV subset to sample it through.
   *
   * This is the ready-gate's public face. A layer asks for what it should
   * draw and gets back something that is always safe to draw — it is
   * structurally unable to render an unready tile.
   */
  drawableFor(id: TileId): { node: TileNode; uv: UVSubset } | null {
    let cur: TileId | null = id;
    while (cur) {
      const n = this.nodes.get(tileKey(cur));
      if (n?.drawable) {
        const uv = uvSubsetIn(n.id, id);
        if (uv) return { node: n, uv };
      }
      cur = parentOf(cur);
    }
    return null;
  }

  /**
   * Declare what the camera wants this frame. Creates missing nodes,
   * refreshes LRU keys, and issues requests up to MAX_INFLIGHT in
   * priority order. Call from a STREAM-priority frame callback, AFTER sim
   * has advanced — so `wanted` is derived from the spring's TARGET, not
   * from where the camera happens to be (Law II.4).
   */
  select(input: SelectionInput): void {
    if (this.disposed) return;
    const now = this.opts.now();
    const candidates: RequestCandidate[] = [];

    for (const w of input.wanted) {
      const key = tileKey(w.id);
      let node = this.nodes.get(key);
      if (!node) {
        node = new TileNode({ id: w.id, bytes: this.bytesPerTile });
        this.nodes.set(key, node);
      }
      node.cameraDistance = w.cameraDistance;
      node.screenDistance = w.screenDistance;
      node.lastUsedMs = now;
      if (node.state === NodeState.PENDING || node.state === NodeState.FAILED) {
        candidates.push({ key, level: w.id.level, screenDistance: w.screenDistance });
      }
    }

    for (const c of sortRequests(candidates)) {
      if (this.inflight >= this.maxInflight) break;
      const node = this.nodes.get(c.key);
      if (node) void this.request(node);
    }

    this.publishMetrics();
  }

  private async request(node: TileNode): Promise<void> {
    if (node.state !== NodeState.PENDING && node.state !== NodeState.FAILED) return;
    if (node.state === NodeState.FAILED && !node.transition(NodeState.PENDING)) return;

    const epoch = node.epoch;
    const controller = new AbortController();
    node.controller = controller;
    node.transition(NodeState.REQUESTED);
    this.inflight++;

    const timeout = setTimeout(() => controller.abort(), STREAM.REQUEST_TIMEOUT_MS);
    try {
      const res = await this.opts.fetchImpl(this.opts.source.url(node.id), { signal: controller.signal });
      // EPOCH CHECK #1. The node may have merged while the request was out.
      if (this.disposed || node.epoch !== epoch) return;
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      if (this.disposed || node.epoch !== epoch) return;

      node.transition(NodeState.DECODING);
      const bitmap = await this.opts.decodeImpl(blob, controller.signal);
      // EPOCH CHECK #2, the load-bearing one. Aborting the fetch does NOT
      // cancel an in-flight createImageBitmap, so a decode can land here
      // long after the node was merged. Close it rather than paint it —
      // skipping the close leaks GPU memory that dispose() cannot recover.
      if (this.disposed || node.epoch !== epoch) {
        bitmap.close();
        return;
      }
      this.uploadQueue.push({ node, bitmap, epoch });
    } catch (err) {
      if (this.disposed || node.epoch !== epoch) return;
      node.attempts++;
      if (node.attempts <= STREAM.MAX_RETRIES) {
        node.transition(NodeState.PENDING);
        const delay = retryDelayMs(node.attempts - 1, this.opts.rand);
        setTimeout(() => {
          if (!this.disposed && node.epoch === epoch && node.state === NodeState.PENDING) void this.request(node);
        }, delay);
      } else {
        node.transition(NodeState.FAILED);
        // Law V: a chain that gives up says so. Silent degradation is why
        // staleness recurs.
        this.opts.onFailure?.(node.id, err);
        // eslint-disable-next-line no-console
        console.error(`[tileCore] ${this.opts.source.id} gave up on ${node.key} after ${node.attempts} attempts`, err);
      }
    } finally {
      clearTimeout(timeout);
      this.inflight--;
      if (node.controller === controller) node.controller = null;
    }
  }

  /**
   * Spend this frame's upload budget. Returns how many textures were
   * uploaded. Call from a STREAM-priority frame callback.
   */
  pumpUploads(): number {
    if (this.disposed) return 0;
    const start = this.opts.now();
    const st: UploadBudgetState = { uploaded: 0, elapsedMs: 0 };

    while (this.uploadQueue.length && canUpload(st)) {
      const item = this.uploadQueue.shift()!;
      const { node, bitmap, epoch } = item;
      // EPOCH CHECK #3: the node can merge between decode and upload too.
      if (node.epoch !== epoch) {
        bitmap.close();
        continue;
      }
      try {
        node.payload = this.opts.sink.upload(node, bitmap);
        node.transition(NodeState.UPLOADED);
        node.transition(NodeState.FADING, this.opts.now());
        st.uploaded++;
        bump(METRIC.TILES_UPLOADED);
      } finally {
        // ALWAYS close, on the success path too. A retained ImageBitmap is
        // a GPU allocation the sink's release() knows nothing about.
        bitmap.close();
      }
      st.elapsedMs = this.opts.now() - start;
    }

    this.advanceFades();
    this.enforceBudget();
    this.publishMetrics();
    return st.uploaded;
  }

  /** Promote finished crossfades to RESIDENT. */
  private advanceFades(): void {
    const now = this.opts.now();
    this.nodes.forEach((node) => {
      if (node.state === NodeState.FADING && node.fade(now) >= 1) node.transition(NodeState.RESIDENT);
    });
  }

  private enforceBudget(): void {
    const plan = evictionPlan(
      Array.from(this.nodes.values()).map((n) => ({
        key: n.key,
        bytes: n.bytes,
        pinned: n.pinned,
        cameraDistance: n.cameraDistance,
        lastUsedMs: n.lastUsedMs,
        resident: n.drawable,
      })),
      this.budgetBytes,
    );
    for (const key of plan.evict) this.evict(key);
  }

  /** Drop a node's GPU payload and reset it to PENDING. The epoch bump is
   *  what makes any in-flight work for it harmless. */
  evict(key: string): void {
    const node = this.nodes.get(key);
    if (!node || node.pinned) return;
    node.invalidate();
    if (node.payload !== null) {
      this.opts.sink.release(node);
      node.payload = null;
    }
    node.state = NodeState.PENDING;
    node.attempts = 0;
    bump(METRIC.TILES_EVICTED);
  }

  /**
   * Merge a subtree: the children are gone, the parent covers their
   * footprint. Bumps every descendant's epoch so nothing in flight for
   * them can land on a node that no longer exists.
   */
  merge(parent: TileId): void {
    Array.from(this.nodes.values()).forEach((node) => {
      if (isAncestorOf(parent, node.id)) {
        this.evict(node.key);
        this.nodes.delete(node.key);
        bump(METRIC.REQUESTS_ABORTED);
      }
    });
  }

  private publishMetrics(): void {
    let resident = 0;
    let pending = 0;
    this.nodes.forEach((n) => {
      if (n.drawable) resident++;
      else if (n.state !== NodeState.FAILED) pending++;
    });
    setGauge(METRIC.TILES_RESIDENT, resident);
    setGauge(METRIC.TILES_PENDING, pending);
    setGauge(METRIC.IN_FLIGHT, this.inflight);
    setGauge(METRIC.VRAM_BYTES, this.residentBytes);
    setGauge(METRIC.TEXTURES, resident);
  }

  /** Law IV teardown. Aborts everything, releases every payload, closes
   *  every queued bitmap, and empties the table. */
  dispose(): void {
    this.disposed = true;
    for (const item of this.uploadQueue) item.bitmap.close();
    this.uploadQueue.length = 0;
    this.nodes.forEach((node) => {
      node.invalidate();
      if (node.payload !== null) {
        this.opts.sink.release(node);
        node.payload = null;
      }
    });
    this.nodes.clear();
    this.inflight = 0;
    setGauge(METRIC.TILES_RESIDENT, 0);
    setGauge(METRIC.TILES_PENDING, 0);
    setGauge(METRIC.IN_FLIGHT, 0);
    setGauge(METRIC.VRAM_BYTES, 0);
    setGauge(METRIC.TEXTURES, 0);
  }
}
