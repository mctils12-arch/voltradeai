// zoomInput — ONE zoom model for every view (Rendering & Motion Law, Law I).
//
// Replaces per-layer wheel handling. Three things it fixes at once:
//
//  1. ZOOM OPERATES ON log(altitude), NOT altitude. Linear steps in linear
//     altitude are unusable across a 300,000× range: one notch that moves
//     you 500km at 30,000km altitude teleports you through the surface at
//     100m. In log space a notch is always the same *proportional* move, so
//     the feel is identical at every scale.
//  2. WHEEL AND PINCH NORMALIZE INTO THE SAME log-altitude delta, so
//     desktop and phone feel the same. Pointer Events only — no
//     touch/mouse branches, no double-fired gestures.
//  3. THE TARGET IS A TARGET. Input never assigns a camera position; it
//     moves a spring's destination and the frame loop flies there. This is
//     Law I at the input boundary.
//
// Anchoring: zoom scales about the point under the cursor (or the pinch
// centroid), never the screen centre. `anchorScale` is the pure 1-D form —
// apply it per axis for a 2-D map or per component for a 3-D camera offset.
//
// Pure module: no DOM at import, every handler takes plain values so the
// whole gesture model is exercisable headless.

import { FRAME, SPRING, Spring } from "./frameCore.ts";

/** Zoom feel + envelope. Fixed by the Rendering & Motion Law article. */
export const ZOOM = {
  WHEEL_RATE: 1 / 450, // slower than default — buys prefetch time
  PINCH_RATE: 1.0,
  ALT_MIN_KM: 0.1, // 100 m
  ALT_MAX_KM: 30000,
  MOON_RADIUS_KM: 1737.4,
  EARTH_RADIUS_KM: 6371.0,
} as const;

/** `WheelEvent.deltaMode` values. A wheel event may report pixels, lines or
 *  pages depending on browser and device; normalizing them is the
 *  difference between a Firefox notch moving 3× a Chrome notch. */
export const DELTA_MODE = { PIXEL: 0, LINE: 1, PAGE: 2 } as const;

/** Pixels a "line" is worth when deltaMode is LINE. Matches the 16px line
 *  box browsers themselves assume. */
export const LINE_HEIGHT_PX = 16;

/** Largest single wheel event we honour, in pixels. A trackpad flick or a
 *  driver hiccup can deliver a 4-digit deltaY in one event; without this
 *  the target jumps the whole envelope in one frame and every tile in
 *  flight is superseded at once. */
export const MAX_WHEEL_DELTA_PX = 240;

// ── log-altitude space (pure) ────────────────────────────────────────────────

/**
 * Clamp a value into [lo, hi], treating ±Infinity as the corresponding
 * bound and NaN as `lo`.
 *
 * The distinction matters: `Math.exp` of a large log-altitude overflows to
 * +Infinity, and a naive `Number.isFinite` guard would then return the
 * MINIMUM altitude — i.e. zooming out past the envelope would slam the
 * camera onto the surface. Caught by the envelope test.
 */
function clampFinite(v: number, lo: number, hi: number): number {
  if (Number.isNaN(v)) return lo;
  if (v === Infinity) return hi;
  if (v === -Infinity) return lo;
  return Math.min(hi, Math.max(lo, v));
}

/** Clamp an altitude to the navigable envelope, in km. */
export function clampAltKm(altKm: number): number {
  return clampFinite(altKm, ZOOM.ALT_MIN_KM, ZOOM.ALT_MAX_KM);
}

/** Altitude (km) → the linear coordinate zoom actually operates in. */
export function altToLog(altKm: number): number {
  return Math.log(clampAltKm(altKm));
}

/** Inverse of `altToLog`, clamped back into the envelope. */
export function logToAlt(logAlt: number): number {
  if (Number.isNaN(logAlt)) return ZOOM.ALT_MIN_KM;
  return clampAltKm(Math.exp(logAlt));
}

/** The envelope's extent in log space — the full travel of one zoom axis. */
export const LOG_ALT_MIN = Math.log(ZOOM.ALT_MIN_KM);
export const LOG_ALT_MAX = Math.log(ZOOM.ALT_MAX_KM);
export const LOG_ALT_SPAN = LOG_ALT_MAX - LOG_ALT_MIN;

export function clampLogAlt(logAlt: number): number {
  return clampFinite(logAlt, LOG_ALT_MIN, LOG_ALT_MAX);
}

// ── input normalization (pure) ───────────────────────────────────────────────

/**
 * A wheel event → a delta in log-altitude.
 *
 * Positive deltaY (scroll down / pull toward you) = zoom OUT = altitude
 * up = positive log delta. deltaMode is normalized to pixels first, then
 * a single event is capped at MAX_WHEEL_DELTA_PX.
 */
export function wheelToLogDelta(deltaY: number, deltaMode: number = DELTA_MODE.PIXEL, viewportHeightPx = 800): number {
  if (!Number.isFinite(deltaY) || deltaY === 0) return 0;
  let px = deltaY;
  if (deltaMode === DELTA_MODE.LINE) px = deltaY * LINE_HEIGHT_PX;
  else if (deltaMode === DELTA_MODE.PAGE) px = deltaY * Math.max(1, viewportHeightPx);
  const capped = Math.sign(px) * Math.min(MAX_WHEEL_DELTA_PX, Math.abs(px));
  return capped * ZOOM.WHEEL_RATE;
}

/**
 * A pinch scale → a delta in log-altitude, in the SAME units wheel
 * produces. `scale` is currentSpread / previousSpread: >1 means fingers
 * spreading = zoom IN = altitude down = negative log delta.
 *
 * Why -ln(scale): halving the altitude is a fixed step in log space, and
 * spreading the fingers to twice the spread should halve the altitude. The
 * two rates therefore agree by construction rather than by tuning.
 */
export function pinchToLogDelta(scale: number): number {
  if (!Number.isFinite(scale) || scale <= 0) return 0;
  const d = -Math.log(scale) * ZOOM.PINCH_RATE;
  // Normalize -0 to 0: a caller comparing `delta === 0` to detect a no-op
  // gesture would otherwise miss the scale-1 case under Object.is.
  return d === 0 ? 0 : d;
}

/** Distance between two pointer positions — the pinch spread. */
export function spread(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Midpoint of two pointers — the pinch anchor. */
export function centroid(a: { x: number; y: number }, b: { x: number; y: number }): { x: number; y: number } {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/**
 * Scale a coordinate about an anchor so the anchor stays put.
 *
 * `ratio` is newAltitude / oldAltitude. At ratio 1 nothing moves; at ratio
 * 0.5 (zoomed in 2×) everything is pulled halfway toward the anchor. Apply
 * per axis for a map centre, per component for a camera offset vector.
 */
export function anchorScale(coord: number, anchor: number, ratio: number): number {
  if (!Number.isFinite(ratio)) return coord;
  return anchor + (coord - anchor) * ratio;
}

/** 2-D convenience over `anchorScale`. */
export function anchorScale2(
  coord: { x: number; y: number },
  anchor: { x: number; y: number },
  ratio: number,
): { x: number; y: number } {
  return { x: anchorScale(coord.x, anchor.x, ratio), y: anchorScale(coord.y, anchor.y, ratio) };
}

// ── the gesture controller ───────────────────────────────────────────────────

export interface ZoomAnchor {
  x: number;
  y: number;
}

export interface ZoomChange {
  /** Log-altitude the camera is now flying toward. */
  targetLogAlt: number;
  /** Altitude in km the camera is flying toward. */
  targetAltKm: number;
  /** newTargetAlt / previousTargetAlt — feed to `anchorScale`. */
  ratio: number;
  /** Where the gesture wants to keep fixed, in CSS pixels. */
  anchor: ZoomAnchor;
}

export interface ZoomInputOptions {
  /** Altitude the view starts at, km. */
  initialAltKm?: number;
  /** Spring the target drives. One is created if not supplied. */
  spring?: Spring;
  /** Called whenever the TARGET moves. Consumers re-anchor here; they must
   *  NOT move the camera — the frame loop does that from the spring. */
  onTarget?: (change: ZoomChange) => void;
  /** Viewport height, for PAGE-mode wheel normalization. */
  viewportHeightPx?: number;
  /** Envelope override — the Moon view and the Earth view differ. */
  minAltKm?: number;
  maxAltKm?: number;
}

/**
 * Turns raw pointer/wheel input into a single moving log-altitude target.
 *
 * Deliberately does NOT touch a camera, a map, or the DOM. Wire it up by
 * feeding it events and reading `spring.value` inside a SIM-priority frame
 * callback — that split is what keeps Law I true no matter which view
 * consumes it.
 */
export class ZoomInput {
  readonly spring: Spring;
  private readonly onTarget?: (change: ZoomChange) => void;
  private readonly viewportHeightPx: number;
  private readonly minLog: number;
  private readonly maxLog: number;

  /** Active pointers, keyed by pointerId. Pointer Events only. */
  private readonly pointers = new Map<number, ZoomAnchor>();
  private pinchSpread = 0;

  /** Timestamp of the last target change, for `isSettled`. */
  private lastInputMs = -Infinity;

  constructor(opts: ZoomInputOptions = {}) {
    const minAlt = opts.minAltKm ?? ZOOM.ALT_MIN_KM;
    const maxAlt = opts.maxAltKm ?? ZOOM.ALT_MAX_KM;
    this.minLog = Math.log(Math.max(1e-6, Math.min(minAlt, maxAlt)));
    this.maxLog = Math.log(Math.max(minAlt, maxAlt));
    const start = this.clamp(Math.log(Math.max(1e-6, opts.initialAltKm ?? ZOOM.ALT_MAX_KM)));
    this.spring =
      opts.spring ??
      new Spring(start, { stiffness: SPRING.ZOOM_STIFFNESS, damping: SPRING.ZOOM_DAMPING, epsilon: FRAME.EPSILON });
    if (!opts.spring) this.spring.snap(start);
    this.onTarget = opts.onTarget;
    this.viewportHeightPx = opts.viewportHeightPx ?? 800;
  }

  private clamp(logAlt: number): number {
    return clampFinite(logAlt, this.minLog, this.maxLog);
  }

  /** Where the camera is right now (spring value), in km. */
  get altKm(): number {
    return Math.exp(this.spring.value);
  }

  /** Where the camera is flying to, in km. Streaming prefetches from THIS
   *  (Law II.4), never from `altKm`. */
  get targetAltKm(): number {
    return Math.exp(this.spring.target);
  }

  get targetLogAlt(): number {
    return this.spring.target;
  }

  /** Number of pointers currently down — 2 means a pinch is in progress. */
  get pointerCount(): number {
    return this.pointers.size;
  }

  /**
   * Has the zoom TARGET stopped moving? True after SPRING.ZOOM_SETTLE_MS of
   * input silence, even while the camera is still flying. This is the
   * signal streaming waits on before committing an expensive prefetch: it
   * means the destination is final, so the tiles fetched for it will still
   * be the right ones when the camera arrives.
   */
  isSettled(nowMs: number, quietMs = SPRING.ZOOM_SETTLE_MS): boolean {
    return nowMs - this.lastInputMs >= quietMs;
  }

  /** Teleport target and value together (init, hard cut, context restore). */
  snapToAltKm(altKm: number): void {
    this.spring.snap(this.clamp(Math.log(Math.max(1e-6, altKm))));
  }

  /** Move the target directly, in km. Used by "fly to" buttons and the
   *  altitude slider — same path as a wheel notch, so they cannot diverge. */
  setTargetAltKm(altKm: number, anchor: ZoomAnchor, nowMs: number): ZoomChange | null {
    return this.applyLogDelta(this.clamp(Math.log(Math.max(1e-6, altKm))) - this.spring.target, anchor, nowMs);
  }

  /** A wheel notch. Returns the change, or null if it was a no-op (already
   *  pinned at an envelope end). */
  wheel(deltaY: number, deltaMode: number, anchor: ZoomAnchor, nowMs: number): ZoomChange | null {
    return this.applyLogDelta(wheelToLogDelta(deltaY, deltaMode, this.viewportHeightPx), anchor, nowMs);
  }

  // ── pointer events (the only input path) ───────────────────────────────

  pointerDown(pointerId: number, x: number, y: number): void {
    this.pointers.set(pointerId, { x, y });
    this.pinchSpread = this.currentSpread();
  }

  /** Returns a change when this move produced a pinch zoom, else null. */
  pointerMove(pointerId: number, x: number, y: number, nowMs: number): ZoomChange | null {
    if (!this.pointers.has(pointerId)) return null;
    this.pointers.set(pointerId, { x, y });
    if (this.pointers.size !== 2) return null;
    const now = this.currentSpread();
    if (this.pinchSpread <= 0 || now <= 0) {
      this.pinchSpread = now;
      return null;
    }
    const scale = now / this.pinchSpread;
    this.pinchSpread = now;
    return this.applyLogDelta(pinchToLogDelta(scale), this.currentCentroid(), nowMs);
  }

  pointerUp(pointerId: number): void {
    this.pointers.delete(pointerId);
    this.pinchSpread = this.currentSpread();
  }

  /** Drop all pointers — pointercancel, blur, or teardown. */
  clearPointers(): void {
    this.pointers.clear();
    this.pinchSpread = 0;
  }

  private currentSpread(): number {
    if (this.pointers.size !== 2) return 0;
    const [a, b] = [...this.pointers.values()];
    return spread(a, b);
  }

  private currentCentroid(): ZoomAnchor {
    if (this.pointers.size !== 2) return { x: 0, y: 0 };
    const [a, b] = [...this.pointers.values()];
    return centroid(a, b);
  }

  private applyLogDelta(deltaLog: number, anchor: ZoomAnchor, nowMs: number): ZoomChange | null {
    if (!Number.isFinite(deltaLog) || deltaLog === 0) return null;
    const prev = this.spring.target;
    const next = this.clamp(prev + deltaLog);
    if (next === prev) return null;
    this.spring.setTarget(next);
    this.lastInputMs = nowMs;
    const change: ZoomChange = {
      targetLogAlt: next,
      targetAltKm: Math.exp(next),
      // exp(next-prev) rather than exp(next)/exp(prev): one exp, and it
      // stays exact at the envelope ends where the two differ in float.
      ratio: Math.exp(next - prev),
      anchor,
    };
    this.onTarget?.(change);
    return change;
  }
}

// ── DOM binding (thin) ───────────────────────────────────────────────────────

export interface BindZoomOptions {
  /** Passive:false is required — we preventDefault to stop page zoom. */
  passive?: boolean;
}

/**
 * Attach a `ZoomInput` to an element using Pointer Events + wheel only.
 * Returns an unbind function; call it in the layer's `dispose()`.
 *
 * No touch* or mouse* listeners: a browser fires BOTH touch and mouse for
 * one finger, and a view listening to both double-applies every gesture.
 */
export function bindZoomInput(
  el: HTMLElement,
  input: ZoomInput,
  nowFn: () => number,
  opts: BindZoomOptions = {},
): () => void {
  const rectAnchor = (ev: { clientX: number; clientY: number }): ZoomAnchor => {
    const r = el.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  };

  const onWheel = (ev: WheelEvent) => {
    ev.preventDefault();
    input.wheel(ev.deltaY, ev.deltaMode, rectAnchor(ev), nowFn());
  };
  const onDown = (ev: PointerEvent) => {
    input.pointerDown(ev.pointerId, ev.clientX, ev.clientY);
    if (input.pointerCount === 2) el.setPointerCapture?.(ev.pointerId);
  };
  const onMove = (ev: PointerEvent) => {
    if (input.pointerCount === 2) ev.preventDefault();
    input.pointerMove(ev.pointerId, ev.clientX, ev.clientY, nowFn());
  };
  const onUp = (ev: PointerEvent) => {
    input.pointerUp(ev.pointerId);
    el.releasePointerCapture?.(ev.pointerId);
  };
  const onCancel = () => input.clearPointers();

  const passive = opts.passive ?? false;
  el.addEventListener("wheel", onWheel, { passive });
  el.addEventListener("pointerdown", onDown);
  el.addEventListener("pointermove", onMove, { passive });
  el.addEventListener("pointerup", onUp);
  el.addEventListener("pointercancel", onCancel);
  el.addEventListener("lostpointercapture", onUp);

  return () => {
    el.removeEventListener("wheel", onWheel);
    el.removeEventListener("pointerdown", onDown);
    el.removeEventListener("pointermove", onMove);
    el.removeEventListener("pointerup", onUp);
    el.removeEventListener("pointercancel", onCancel);
    el.removeEventListener("lostpointercapture", onUp);
    input.clearPointers();
  };
}
