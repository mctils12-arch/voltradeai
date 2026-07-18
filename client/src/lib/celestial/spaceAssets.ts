// spaceAssets — TEXTURE ASSETS, TIERS, LOADING & VIEW PREFS for the space
// frame (SPACE VIEW VISUAL UPGRADE, human directive 2026-07-18:
// research/directives/space_view_handoff_2026-07-18.md + the working
// reference research/directives/space_view_reference_2026-07-18.html).
//
// WHAT SHIPS WHERE: static texture binaries live in client/public/space/
// (the client/public/models/ ISS-mesh precedent). NOTHING here loads at
// page startup — the manager is created inside mountSpaceFrame, so the
// map pays zero cost until a user actually leaves Earth (B0 freeze
// discipline: no celestial asset cost at load).
//
// LICENSES (surfaced in the panel + the in-frame caption — the visible
// credit for the Milky Way panorama is REQUIRED by its CC-BY terms):
//  · Planet maps (mercury/venus/mars/jupiter/saturn/uranus/neptune, sun,
//    Saturn ring strips): NASA/JPL planetary mosaics (public domain),
//    distributed via the threex.planets texture pack.
//  · Moon (moonmap1k / moonbump1k / 2k_moon / moon_8k): NASA LRO WAC
//    global mosaic (public domain).
//  · Milky Way panorama (8k_stars_milky_way): © Solar System Scope,
//    CC-BY 4.0 — attribution "solarsystemscope.com" must stay visible
//    wherever the panorama renders.
//
// PROGRESSIVE TIERS (perf mandate: no >16ms main-thread task):
//  · low tier first (≤1024×512 per body — ~2 MB RGBA each), decoded
//    OFF-THREAD via createImageBitmap, then extracted to ImageData in
//    row bands small enough that each getImageData readback stays well
//    under a frame;
//  · the Moon alone has higher tiers: 2k while its disc is large, the
//    8k source only while the Moon is FOCUSED and close. The 8k file is
//    decoded at a 4096×2048 working resolution — a stated cap: the
//    sprite renderer shades at most a ~448 px disc, which can display
//    ≤ ~1800 texels of the visible hemisphere, so a full 8192×4096 RGBA
//    decode would cost 134 MB of heap for detail the renderer cannot
//    show (4096×2048 costs 33 MB and is visually identical here);
//  · the 8k tier is EVICTED the moment the Moon loses focus, and every
//    texture is freed on dispose (leaving the space view unloads all).
//
// Hermetic surface: everything above the manager (manifest, licenses,
// fade curve, tier chooser, prefs) is pure/no-DOM and `npx tsx --test`-
// able. The manager itself touches fetch/canvas and is only constructed
// from the mounted frame.

export interface TexImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  /** which asset tier this pixel buffer came from (sprite cache keys). */
  tier: string;
}

// ── manifest ────────────────────────────────────────────────────────────────

export const SPACE_TEXTURE_BASE = "/space/";

/** One equirectangular color map per textured body (single tier). Earth is
 *  deliberately ABSENT: Earth is the live map canvas (handoff integration
 *  point 1), never a texture. */
export const SPACE_TEXTURE_FILES: Record<string, string> = {
  sun: "sunmap.jpg",
  mercury: "mercurymap.jpg",
  venus: "venusmap.jpg",
  mars: "marsmap1k.jpg",
  jupiter: "jupitermap.jpg",
  saturn: "saturnmap.jpg",
  uranus: "uranusmap.jpg",
  neptune: "neptunemap.jpg",
  moon: "moonmap1k.jpg",
};

export const MOON_TIER_FILES: Record<"1k" | "2k" | "8k", string> = {
  "1k": "moonmap1k.jpg",
  "2k": "2k_moon.jpg",
  "8k": "moon_8k.jpg",
};

export const MOON_BUMP_FILE = "moonbump1k.jpg";

export const SATURN_RING_FILES = {
  color: "saturnringcolor.jpg",
  alpha: "saturnringpattern.gif",
} as const;

export const MILKY_WAY_FILE = "8k_stars_milky_way.jpg";

/** Decode caps (working resolutions), per asset kind — see header. */
export const DECODE_CAPS: Record<string, { w: number; h: number }> = {
  low: { w: 1024, h: 512 },
  "2k": { w: 2048, h: 1024 },
  "8k": { w: 4096, h: 2048 },
  milkyway: { w: 2048, h: 1024 },
  ring: { w: 1024, h: 128 },
};

// ── licenses (the visible credit strings) ───────────────────────────────────

export const MILKY_WAY_CREDIT = "Milky Way: solarsystemscope.com CC-BY 4.0";
export const SPACE_IMAGERY_CREDIT =
  "imagery: NASA/JPL planetary mosaics (public domain) via threex.planets · Moon: NASA LRO";

// ── Milky Way fade (reference: clamp((camAU−8)/25,0,1) of the camera's TRUE
// heliocentric distance — invisible near Earth (1 AU), saturated past 33 AU
// and at thousands of AU, exactly the confirmed far-zoom screenshot) ────────

export const MILKY_WAY_FADE_START_AU = 8;
export const MILKY_WAY_FADE_SPAN_AU = 25;

export function milkyWayOpacity(camAU: number): number {
  if (!Number.isFinite(camAU)) return 0;
  const t = (camAU - MILKY_WAY_FADE_START_AU) / MILKY_WAY_FADE_SPAN_AU;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

// ── Moon texture tier chooser ───────────────────────────────────────────────

/** 2k when the rendered disc reads at this size (a 1k map is visibly soft
 *  past ~90 px of disc). */
export const MOON_TIER_2K_MIN_DISC_PX = 90;
/** The 8k source only while the Moon is FOCUSED and genuinely close — the
 *  directive's "only while the Moon is focused/close". */
export const MOON_TIER_8K_MIN_DISC_PX = 260;

export function moonTextureTier(discPx: number, focused: boolean): "1k" | "2k" | "8k" {
  if (focused && discPx >= MOON_TIER_8K_MIN_DISC_PX) return "8k";
  if (discPx >= MOON_TIER_2K_MIN_DISC_PX) return "2k";
  return "1k";
}

// ── view prefs (the scaleModel/orbitPath localStorage pattern: get/set/
// subscribe; node-safe when localStorage is absent) ─────────────────────────

function makeBoolPref(key: string, def: boolean): {
  get: () => boolean;
  set: (on: boolean) => void;
  subscribe: (fn: () => void) => () => void;
} {
  let cur = def;
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (raw === "0") cur = false;
    else if (raw === "1") cur = true;
  } catch { /* no storage — default */ }
  const listeners = new Set<() => void>();
  return {
    get: () => cur,
    set: (on: boolean) => {
      if (on === cur) return;
      cur = on;
      try { globalThis.localStorage?.setItem(key, on ? "1" : "0"); } catch { /* best-effort */ }
      for (const fn of Array.from(listeners)) {
        try { fn(); } catch { /* listener owns its errors */ }
      }
    },
    subscribe: (fn: () => void) => {
      listeners.add(fn);
      return () => { listeners.delete(fn); };
    },
  };
}

/** Milky Way panorama backdrop (default ON — fades with camera altitude). */
const milkyPref = makeBoolPref("vt-celestial-milkyway", true);
export const getMilkyWayPref = milkyPref.get;
export const setMilkyWayPref = milkyPref.set;
export const subscribeMilkyWayPref = milkyPref.subscribe;

/** Ecliptic grid — AU range rings + bearing spokes (default OFF, matching
 *  the reference panel). */
const gridPref = makeBoolPref("vt-celestial-grid", false);
export const getEclipticGridPref = gridPref.get;
export const setEclipticGridPref = gridPref.set;
export const subscribeEclipticGridPref = gridPref.subscribe;

/** 60° trailing motion arcs (default ON, the reference's "Motion trails"). */
const trailsPref = makeBoolPref("vt-celestial-trails", true);
export const getMotionTrailsPref = trailsPref.get;
export const setMotionTrailsPref = trailsPref.set;
export const subscribeMotionTrailsPref = trailsPref.subscribe;

/** Body labels (default ON). Turning labels OFF never hides the sub-pixel
 *  MARKER dots — nothing is ever invisible (the honesty rail); only the
 *  text is suppressed. */
const labelsPref = makeBoolPref("vt-celestial-labels", true);
export const getBodyLabelsPref = labelsPref.get;
export const setBodyLabelsPref = labelsPref.set;
export const subscribeBodyLabelsPref = labelsPref.subscribe;

// ── the texture manager (DOM side — constructed only by the mounted frame) ──

export interface SpaceTexStats {
  /** color textures fully loaded (bodies + ring + milky way). */
  loaded: number;
  ids: string[];
  /** highest Moon tier currently resident. */
  moonTier: "1k" | "2k" | "8k" | "none";
  /** resident texture bytes (all Uint8ClampedArray buffers). */
  bytes: number;
  /** worst single main-thread readback band, ms (the ≤16ms budget proof). */
  maxChunkMs: number;
  /** true when the createImageBitmap path was unavailable (img fallback —
   *  decode then happens on the main thread; stated, never hidden). */
  mainThreadDecode: boolean;
}

export interface SpaceTextureManager {
  /** begin loading the low tier (serial queue, one asset in flight). */
  start(): void;
  /** best resident color texture for a body id (moon: highest tier). */
  get(id: string): TexImage | null;
  getMoonBump(): TexImage | null;
  getRing(): { color: TexImage; alpha: TexImage } | null;
  getMilkyWay(): TexImage | null;
  /** per-frame moon view report: drives 2k/8k fetch + 8k eviction. */
  noteMoonView(discPx: number, focused: boolean): void;
  stats(): SpaceTexStats;
  /** repaint hook — fired whenever a texture becomes resident/evicted. */
  onUpdate(cb: () => void): void;
  dispose(): void;
}

interface Job {
  key: string;
  file: string;
  cap: { w: number; h: number };
  tier: string;
}

export function createSpaceTextureManager(base: string = SPACE_TEXTURE_BASE): SpaceTextureManager {
  const store = new Map<string, TexImage>(); // key: body id | moon:2k | moon:8k | moon:bump | ring:color | ring:alpha | milkyway
  let disposed = false;
  let started = false;
  let running = false;
  let maxChunkMs = 0;
  let mainThreadDecode = false;
  let update: () => void = () => {};
  const abort = new AbortController();
  const queue: Job[] = [];
  const queued = new Set<string>();

  const bytes = (): number => {
    let b = 0;
    for (const t of Array.from(store.values())) b += t.data.byteLength;
    return b;
  };
  const mirrorBytes = (): void => {
    // drive/debug seam (prod-inert, the __vtMap precedent): eviction and
    // dispose are observable even after the handle is gone
    try { (globalThis as any).__vtSpaceTexBytes = bytes(); } catch { /* sandboxed */ }
  };

  async function decodeToTex(blob: Blob, cap: { w: number; h: number }, tier: string): Promise<TexImage> {
    // decode (and rescale) OFF the main thread when createImageBitmap
    // exists; the main thread then only ever touches one BAND at a time —
    // a small band canvas is blitted 1:1 and read back, so each raster
    // flush + copy is a bounded task (no whole-image flush ever happens).
    let bmp: ImageBitmap | HTMLImageElement;
    if (typeof createImageBitmap === "function") {
      let full = await createImageBitmap(blob);
      const scale = Math.min(1, cap.w / full.width, cap.h / full.height);
      if (scale < 1) {
        const scaled = await createImageBitmap(full, {
          resizeWidth: Math.max(1, Math.round(full.width * scale)),
          resizeHeight: Math.max(1, Math.round(full.height * scale)),
          resizeQuality: "high",
        });
        try { full.close(); } catch { /* closed */ }
        full = scaled;
      }
      bmp = full;
    } else {
      // fallback (no createImageBitmap): <img> decode — main thread, stated
      mainThreadDecode = true;
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.src = url;
      await img.decode();
      URL.revokeObjectURL(url);
      bmp = img;
    }
    const srcW = "width" in bmp ? bmp.width : 1;
    const srcH = "height" in bmp ? bmp.height : 1;
    const scale = Math.min(1, cap.w / srcW, cap.h / srcH);
    const w = Math.max(1, Math.round(srcW * scale));
    const h = Math.max(1, Math.round(srcH * scale));
    const oneToOne = w === srcW && h === srcH;
    const out = new Uint8ClampedArray(w * h * 4);
    const bandRows = Math.max(16, Math.floor(300_000 / Math.max(1, w)));
    const cv = document.createElement("canvas");
    cv.width = w;
    cv.height = Math.min(h, bandRows);
    const ctx = cv.getContext("2d", { willReadFrequently: true })!;
    for (let y = 0; y < h; y += bandRows) {
      const rows = Math.min(bandRows, h - y);
      const t0 = performance.now();
      ctx.clearRect(0, 0, w, rows);
      if (oneToOne) {
        // 1:1 band blit (bitmap already at target size — the normal path)
        ctx.drawImage(bmp as CanvasImageSource, 0, y, w, rows, 0, 0, w, rows);
      } else {
        // <img> fallback only: scale the matching source strip
        ctx.drawImage(bmp as CanvasImageSource, 0, y / scale, srcW, rows / scale, 0, 0, w, rows);
      }
      const band = ctx.getImageData(0, 0, w, rows);
      out.set(band.data.subarray(0, rows * w * 4), y * w * 4);
      const ms = performance.now() - t0;
      if (ms > maxChunkMs) maxChunkMs = ms;
      if (y + rows < h) await new Promise((r) => setTimeout(r, 0)); // yield
    }
    if ("close" in bmp) {
      try { (bmp as ImageBitmap).close(); } catch { /* closed */ }
    }
    return { data: out, width: w, height: h, tier };
  }

  function enqueue(key: string, file: string, cap: { w: number; h: number }, tier: string): void {
    if (disposed || store.has(key) || queued.has(key)) return;
    queued.add(key);
    queue.push({ key, file, cap, tier });
    void run();
  }

  async function run(): Promise<void> {
    if (running || disposed) return;
    running = true;
    while (queue.length && !disposed) {
      const job = queue.shift()!;
      try {
        const res = await fetch(base + job.file, { signal: abort.signal });
        if (!res.ok) throw new Error(`${res.status}`);
        const blob = await res.blob();
        if (disposed) break;
        const tex = await decodeToTex(blob, job.cap, job.tier);
        if (disposed) break;
        store.set(job.key, tex);
        mirrorBytes();
        update();
      } catch {
        // missing/failed asset: the body keeps its honest Lambert sphere —
        // degrade, never break (and never retry-loop: queued stays set)
      }
    }
    running = false;
  }

  return {
    start(): void {
      if (started || disposed) return;
      started = true;
      // hero-first order: sun + moon carry the entry view, ring strips are
      // tiny, planets follow, the (largest) milky way panorama last
      enqueue("sun", SPACE_TEXTURE_FILES.sun, DECODE_CAPS.low, "1k");
      enqueue("moon", MOON_TIER_FILES["1k"], DECODE_CAPS.low, "1k");
      enqueue("moon:bump", MOON_BUMP_FILE, DECODE_CAPS.low, "1k");
      enqueue("ring:color", SATURN_RING_FILES.color, DECODE_CAPS.ring, "ring");
      enqueue("ring:alpha", SATURN_RING_FILES.alpha, DECODE_CAPS.ring, "ring");
      for (const id of ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]) {
        enqueue(id, SPACE_TEXTURE_FILES[id], DECODE_CAPS.low, "1k");
      }
      enqueue("milkyway", MILKY_WAY_FILE, DECODE_CAPS.milkyway, "2k");
    },
    get(id: string): TexImage | null {
      if (id === "moon") {
        return store.get("moon:8k") ?? store.get("moon:2k") ?? store.get("moon") ?? null;
      }
      return store.get(id) ?? null;
    },
    getMoonBump(): TexImage | null {
      return store.get("moon:bump") ?? null;
    },
    getRing(): { color: TexImage; alpha: TexImage } | null {
      const color = store.get("ring:color");
      const alpha = store.get("ring:alpha");
      return color && alpha ? { color, alpha } : null;
    },
    getMilkyWay(): TexImage | null {
      return store.get("milkyway") ?? null;
    },
    noteMoonView(discPx: number, focused: boolean): void {
      if (disposed || !started) return;
      const tier = moonTextureTier(discPx, focused);
      if (tier === "8k") enqueue("moon:8k", MOON_TIER_FILES["8k"], DECODE_CAPS["8k"], "8k");
      if (tier === "8k" || tier === "2k") enqueue("moon:2k", MOON_TIER_FILES["2k"], DECODE_CAPS["2k"], "2k");
      if (!focused && store.has("moon:8k")) {
        // the eviction contract: the 33 MB working buffer never outlives
        // Moon focus (and dispose() below frees everything on exit)
        store.delete("moon:8k");
        queued.delete("moon:8k");
        mirrorBytes();
        update();
      }
    },
    stats(): SpaceTexStats {
      const ids = Array.from(store.keys());
      const moonTier = store.has("moon:8k") ? "8k" : store.has("moon:2k") ? "2k" : store.has("moon") ? "1k" : "none";
      return { loaded: ids.length, ids, moonTier, bytes: bytes(), maxChunkMs, mainThreadDecode };
    },
    onUpdate(cb: () => void): void {
      update = cb;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      try { abort.abort(); } catch { /* already */ }
      queue.length = 0;
      queued.clear();
      store.clear();
      mirrorBytes(); // __vtSpaceTexBytes → 0: leaving space unloads everything
      update = () => {};
    },
  };
}
