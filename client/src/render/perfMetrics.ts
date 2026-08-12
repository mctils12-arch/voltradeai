// perfMetrics — the shared counter store the `?perf=1` HUD reads.
//
// Split from perfHud.ts on purpose: PUBLISHERS (tileCore, satLayer, the
// trail layer, any streamer) import this and nothing else. It has no DOM,
// no frame loop, and no dependency on the HUD existing — a layer must not
// need to know whether anyone is watching. When the HUD is off, publishing
// is a Map.set and costs nothing measurable.
//
// Everything here is a GAUGE (current value) or a COUNTER (monotonic
// total). Rates are derived in the HUD, not stored, so a publisher never
// has to know the sampling interval.

/** Canonical metric names. Free-form strings are accepted too, but the HUD
 *  gives these a fixed row and a unit, so use them where they fit. */
export const METRIC = {
  /** Gauges — current state. */
  DRAW_CALLS: "drawCalls",
  TEXTURES: "textures",
  VRAM_BYTES: "vramBytes",
  IN_FLIGHT: "inFlight",
  TILES_RESIDENT: "tilesResident",
  TILES_PENDING: "tilesPending",
  FEATURES: "features",
  /** Counters — monotonic totals. */
  TILES_UPLOADED: "tilesUploaded",
  TILES_EVICTED: "tilesEvicted",
  REQUESTS_ABORTED: "requestsAborted",
  /** Frames that drew a node which was not FADING/RESIDENT. Law II.1 says
   *  this must be zero; the harness asserts on it (self-see #2). */
  UNREADY_DRAWS: "unreadyDraws",
} as const;

export type MetricName = (typeof METRIC)[keyof typeof METRIC] | (string & {});

const gauges = new Map<string, number>();
const counters = new Map<string, number>();

/** Set a gauge to its current value. Idempotent, last write wins. */
export function setGauge(name: MetricName, value: number): void {
  gauges.set(name, value);
}

/** Add to a gauge (e.g. a layer contributing its own draw calls). Layers
 *  that share a gauge must reset it once per frame — see `resetGauge`. */
export function addGauge(name: MetricName, delta: number): void {
  gauges.set(name, (gauges.get(name) ?? 0) + delta);
}

export function resetGauge(name: MetricName): void {
  gauges.set(name, 0);
}

export function getGauge(name: MetricName): number {
  return gauges.get(name) ?? 0;
}

/** Bump a monotonic counter. */
export function bump(name: MetricName, delta = 1): void {
  counters.set(name, (counters.get(name) ?? 0) + delta);
}

export function getCounter(name: MetricName): number {
  return counters.get(name) ?? 0;
}

export interface MetricsSnapshot {
  gauges: Record<string, number>;
  counters: Record<string, number>;
}

export function snapshot(): MetricsSnapshot {
  return {
    gauges: Object.fromEntries(gauges),
    counters: Object.fromEntries(counters),
  };
}

/** Drop every metric. Used by the harness between layer-toggle cases so a
 *  previous case's residue cannot mask a leak. */
export function resetMetrics(): void {
  gauges.clear();
  counters.clear();
}

// ── VRAM accounting (pure) ───────────────────────────────────────────────────

/** Bytes per pixel for the texture formats this renderer uploads. ETC1S /
 *  BC1 are 4bpp block formats; RGBA8 is the uncompressed fallback. */
export const BYTES_PER_PIXEL = {
  /** ETC1S / BC1 / ETC2-RGB — 4 bits per pixel. */
  compressed4bpp: 0.5,
  /** BC7 / ASTC 4x4 / ETC2-RGBA — 8 bits per pixel. */
  compressed8bpp: 1,
  rgba8: 4,
  rgb8: 3,
} as const;

export type TextureFormatClass = keyof typeof BYTES_PER_PIXEL;

/**
 * VRAM a texture occupies, in bytes. `mipped` adds the full mip chain,
 * which converges to 1/3 extra (sum of 4^-n) — the ×1.33 the tile budget
 * math assumes.
 */
export function textureBytes(
  widthPx: number,
  heightPx: number,
  format: TextureFormatClass = "rgba8",
  mipped = false,
): number {
  if (!(widthPx > 0) || !(heightPx > 0)) return 0;
  const base = widthPx * heightPx * BYTES_PER_PIXEL[format];
  return mipped ? base * (4 / 3) : base;
}

/** Human-readable byte count for the HUD. Binary units, one decimal. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}
