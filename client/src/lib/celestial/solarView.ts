// Solar-system view — EARTH TWIN O6-7 tier 2 spike renderer (charter: "hand
// off to a celestial scene at LITERALLY ACCURATE SCALE — Earth shrinks to
// truth, the Moon a real 30-earth-diameter distance away at true size,
// Sun/planets at real ephemeris positions; the emptiness IS the honesty").
//
// WHAT THIS IS: a self-contained 2D-canvas prototype that mounts over the
// map container when the user zooms out past the globe's minimum zoom
// (handoff design: research/solar_view_spike.md). No React, no map
// dependency — mount(container, opts) / handle.setTime() / handle.dispose().
//
// TRUE SCALE, LITERALLY: one number (metersPerPixel) maps the real world to
// the screen. Every body is drawn at its true physical radius and its true
// ephemeris position (lib/celestial/solarSystem.ts — Schlyter/van Flandern
// low-precision theory, arcmin-class) in a top-down orthographic projection
// of the ecliptic plane (z discarded in projection only — never in the
// data). At solar-system scale almost every body is honestly SUB-PIXEL and
// is drawn as a single dim pixel — that emptiness is the point.
//
// ANNOTATION MODE (clearly labeled, on by default so the view is usable):
// ring markers, name + true-distance labels, and orbit paths SAMPLED from
// the same ephemeris. Markers annotate positions; they NEVER inflate a
// body's drawn size or move it. Turning annotation off leaves only the
// literal truth. Distances render through lib/units.ts (imperial | metric
// user preference) below 0.01 AU; in AU above (AU is a fixed domain
// convention, like knots for vessels).
//
// Display colors are presentation only (approximate naked-eye appearance),
// not data. No starfield: a decorative random sky would violate "real
// position or absent" — a real bright-star catalog is future work.

import {
  solarSystemState,
  BODY_RADIUS_M,
  AU_M,
  type BodyId,
  type BodyState,
} from "./solarSystem.js";
import { fmtKm } from "../units.js";

// ── pure view math (exported for tests) ─────────────────────────────────────

export interface ViewCamera {
  /** meters per CSS pixel — THE scale; true for every body drawn. */
  metersPerPixel: number;
  /** camera center, offset from Earth's current position, meters (ecliptic
   * x/y). 0/0 keeps Earth centered while time advances — the handoff pose. */
  centerOffsetXM: number;
  centerOffsetYM: number;
}

/** Ecliptic-plane world (meters, relative camera center) → CSS px. Top-down
 * orthographic: +x right (vernal-equinox direction), +y up = screen-up. */
export function worldToScreen(
  xM: number,
  yM: number,
  cam: ViewCamera,
  widthPx: number,
  heightPx: number,
): { x: number; y: number } {
  return {
    x: widthPx / 2 + (xM - cam.centerOffsetXM) / cam.metersPerPixel,
    y: heightPx / 2 - (yM - cam.centerOffsetYM) / cam.metersPerPixel,
  };
}

/** A body's TRUE on-screen radius in px. May be (honestly) sub-pixel. */
export function trueRadiusPx(radiusM: number, metersPerPixel: number): number {
  return radiusM / metersPerPixel;
}

/**
 * Entry scale for the handoff: the metersPerPixel at which Earth's true
 * disc spans `earthDiscPx` on screen. The parent measures the globe's
 * apparent diameter at max zoom-out and passes it here so the Earth the
 * user sees does not jump size across the switch.
 */
export function entryScaleForEarthDisc(earthDiscPx: number): number {
  return (2 * BODY_RADIUS_M.earth) / Math.max(1, earthDiscPx);
}

/**
 * Exit test (view → globe): true when zooming IN has made Earth's true disc
 * larger than `exitFraction` of the viewport's smaller side. Set
 * exitFraction ABOVE the entry disc fraction for hysteresis so the
 * boundary can't flip-flop on one wheel notch.
 */
export function shouldExitToGlobe(
  metersPerPixel: number,
  viewportMinPx: number,
  exitFraction = 0.6,
): boolean {
  return 2 * trueRadiusPx(BODY_RADIUS_M.earth, metersPerPixel) > viewportMinPx * exitFraction;
}

/** Hard scale clamp: Earth-fills-screen … well past Neptune's orbit. */
export function clampScale(metersPerPixel: number): number {
  return Math.min(1e11, Math.max(1e3, metersPerPixel));
}

/**
 * Scale-bar picker: a 1/2/5·10^n meters length whose drawn width lands in
 * [minPx, 2·minPx). Returns real meters — the label prints the truth.
 */
export function pickScaleBarMeters(metersPerPixel: number, minPx = 90): number {
  const target = metersPerPixel * minPx;
  const decade = Math.pow(10, Math.floor(Math.log10(target)));
  for (const m of [1, 2, 5, 10]) {
    if (decade * m >= target) return decade * m;
  }
  return decade * 10;
}

/** Distance label: km/mi via the units preference below 0.01 AU, AU above. */
export function fmtDistanceFromEarth(meters: number): string {
  if (meters < 0.01 * AU_M) return fmtKm(meters / 1000, 0);
  const au = meters / AU_M;
  return `${au.toFixed(au < 10 ? 3 : 2)} AU`;
}

// ── presentation constants ───────────────────────────────────────────────────

/** Approximate naked-eye display colors — presentation only, not data. */
const BODY_COLOR: Record<BodyId, string> = {
  sun: "#ffd76e",
  mercury: "#b5a9a0",
  venus: "#e6d5a8",
  earth: "#5b8fd9",
  moon: "#c9c9c9",
  mars: "#d97b5b",
  jupiter: "#d9b08c",
  saturn: "#e0cba8",
  uranus: "#9fd4d9",
  neptune: "#6e8fd9",
};

/** Sidereal orbital periods in days — Kepler's 3rd law from the theory's own
 * semi-major axes (365.256·a^1.5); used only to close orbit-path sampling. */
const ORBIT_PERIOD_DAYS: Partial<Record<BodyId, number>> = {
  mercury: 365.256 * Math.pow(0.387098, 1.5),
  venus: 365.256 * Math.pow(0.72333, 1.5),
  earth: 365.256,
  mars: 365.256 * Math.pow(1.523688, 1.5),
  jupiter: 365.256 * Math.pow(5.20256, 1.5),
  saturn: 365.256 * Math.pow(9.55475, 1.5),
  uranus: 365.256 * Math.pow(19.18171, 1.5),
  neptune: 365.256 * Math.pow(30.05826, 1.5),
  moon: 27.321661, // sidereal month
};

const ORBIT_SAMPLES = 192;
const MS_PER_DAY = 86_400_000;

// ── the view ─────────────────────────────────────────────────────────────────

export interface SolarViewOptions {
  /** initial time (default: now). */
  timeMs?: number;
  /** initial scale; default = Earth disc at 40% of the viewport min side. */
  metersPerPixel?: number;
  /** annotation mode on mount (default true — labeled on-canvas). */
  annotate?: boolean;
  /** wheel-zoom / drag-pan / dblclick-recenter handlers (default true). */
  interactive?: boolean;
  /** Earth-disc viewport fraction above which the view asks to hand the
   * camera back to the globe (default 0.6 — see shouldExitToGlobe). */
  exitFraction?: number;
  /** called once when the user zooms in past exitFraction: the parent
   * should dispose this view and restore the map camera. */
  onZoomIntoEarth?: () => void;
}

export interface SolarViewHandle {
  setTime(date: Date | number): void;
  setAnnotate(on: boolean): void;
  setScale(metersPerPixel: number): void;
  getState(): {
    timeMs: number;
    metersPerPixel: number;
    annotate: boolean;
    centerOffsetXM: number;
    centerOffsetYM: number;
  };
  /** re-render at the current state (e.g. after a units-preference change). */
  render(): void;
  dispose(): void;
}

interface OrbitCacheEntry {
  builtAtMs: number;
  /** heliocentric ecliptic x/y in meters (moon: geocentric). */
  points: Array<{ x: number; y: number }>;
}

export function mount(container: HTMLElement, opts: SolarViewOptions = {}): SolarViewHandle {
  const canvas = document.createElement("canvas");
  canvas.className = "vt-solar-view";
  // Overlay styling is inline so the spike is drop-in (no CSS file edit);
  // the integration PR may move this into index.css.
  canvas.style.position = "absolute";
  canvas.style.inset = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.background = "#020409"; // site map background family
  canvas.style.cursor = "grab";
  canvas.style.touchAction = "none";
  container.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  let timeMs = opts.timeMs ?? Date.now();
  let annotate = opts.annotate ?? true;
  const interactive = opts.interactive ?? true;
  const exitFraction = opts.exitFraction ?? 0.6;
  let exited = false;
  let disposed = false;

  const initialMinSide = Math.max(
    1,
    Math.min(container.clientWidth || 800, container.clientHeight || 600),
  );
  const cam: ViewCamera = {
    metersPerPixel: clampScale(
      opts.metersPerPixel ?? entryScaleForEarthDisc(initialMinSide * 0.4),
    ),
    centerOffsetXM: 0,
    centerOffsetYM: 0,
  };

  const orbitCache = new Map<BodyId, OrbitCacheEntry>();

  function cssSize(): { w: number; h: number } {
    return { w: canvas.clientWidth || 1, h: canvas.clientHeight || 1 };
  }

  function resizeBacking(): void {
    const dpr = (globalThis.devicePixelRatio as number | undefined) || 1;
    const { w, h } = cssSize();
    const bw = Math.max(1, Math.round(w * dpr));
    const bh = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** Orbit path from the SAME ephemeris (one period sampled around the
   * current time) — a computed path, cached for 6h of set time. */
  function orbitPoints(id: BodyId): Array<{ x: number; y: number }> {
    const period = ORBIT_PERIOD_DAYS[id];
    if (!period) return [];
    const cached = orbitCache.get(id);
    if (cached && Math.abs(cached.builtAtMs - timeMs) < 6 * 3_600_000) return cached.points;
    const points: Array<{ x: number; y: number }> = [];
    for (let k = 0; k <= ORBIT_SAMPLES; k++) {
      const t = timeMs + ((k / ORBIT_SAMPLES) - 0.5) * period * MS_PER_DAY;
      const body = solarSystemState(t).find((b) => b.id === id)!;
      const p = id === "moon" ? body.geoAu : body.helioAu; // moon path is geocentric
      points.push({ x: p.x * AU_M, y: p.y * AU_M });
    }
    orbitCache.set(id, { builtAtMs: timeMs, points });
    return points;
  }

  function render(): void {
    if (disposed || !ctx) return;
    resizeBacking();
    const { w, h } = cssSize();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#020409";
    ctx.fillRect(0, 0, w, h);

    const state = solarSystemState(timeMs);
    const earth = state.find((b) => b.id === "earth")!;
    const earthXM = earth.helioAu.x * AU_M;
    const earthYM = earth.helioAu.y * AU_M;
    // camera center in heliocentric meters = Earth + offset
    const camHelio: ViewCamera = {
      metersPerPixel: cam.metersPerPixel,
      centerOffsetXM: earthXM + cam.centerOffsetXM,
      centerOffsetYM: earthYM + cam.centerOffsetYM,
    };

    // Orbit paths first (annotation layer, beneath the bodies).
    if (annotate) {
      ctx.lineWidth = 1;
      for (const body of state) {
        const pts = orbitPoints(body.id);
        if (!pts.length) continue;
        ctx.strokeStyle = "rgba(140,160,190,0.18)";
        ctx.beginPath();
        for (let k = 0; k < pts.length; k++) {
          // Moon's sampled path is geocentric → anchor to Earth's CURRENT
          // position (labeled: instantaneous orbit, not a trail).
          const px = body.id === "moon" ? pts[k].x + earthXM : pts[k].x;
          const py = body.id === "moon" ? pts[k].y + earthYM : pts[k].y;
          const s = worldToScreen(px, py, camHelio, w, h);
          if (k === 0) ctx.moveTo(s.x, s.y);
          else ctx.lineTo(s.x, s.y);
        }
        ctx.stroke();
      }
    }

    // Bodies at TRUE scale and TRUE position.
    const margin = 40;
    for (const body of state) {
      const s = worldToScreen(body.helioAu.x * AU_M, body.helioAu.y * AU_M, camHelio, w, h);
      if (s.x < -margin || s.x > w + margin || s.y < -margin || s.y > h + margin) {
        // annotation mode still points at off-screen bodies via edge labels
        if (annotate) drawEdgePointer(ctx, s.x, s.y, w, h, body);
        continue;
      }
      const r = trueRadiusPx(body.radiusM, cam.metersPerPixel);
      ctx.fillStyle = BODY_COLOR[body.id];
      if (r >= 0.5) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // honestly sub-pixel: one dim pixel, never inflated
        ctx.globalAlpha = 0.9;
        ctx.fillRect(Math.round(s.x), Math.round(s.y), 1, 1);
        ctx.globalAlpha = 1;
      }
      if (annotate) drawAnnotation(ctx, s.x, s.y, r, body);
    }

    drawChrome(ctx, w, h, state);
  }

  function drawAnnotation(
    c: CanvasRenderingContext2D,
    x: number,
    y: number,
    truePx: number,
    body: BodyState,
  ): void {
    // Ring OUTSIDE the true disc — an annotation, never the body itself.
    const ring = Math.max(truePx + 6, 9);
    c.strokeStyle = "rgba(190,205,225,0.55)";
    c.lineWidth = 1;
    c.beginPath();
    c.arc(x, y, ring, 0, Math.PI * 2);
    c.stroke();
    c.fillStyle = "rgba(210,222,238,0.92)";
    c.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    c.textBaseline = "middle";
    const dist = body.id === "earth" ? "" : ` · ${fmtDistanceFromEarth(body.rEarthM)}`;
    const sub = truePx < 0.5 ? " · sub-pixel at this scale" : "";
    c.fillText(`${body.name}${dist}${sub}`, x + ring + 6, y);
  }

  function drawEdgePointer(
    c: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    body: BodyState,
  ): void {
    // Clamp the off-screen position to the viewport edge and label it there —
    // a direction pointer, not a position claim.
    const pad = 14;
    const cx = Math.max(pad, Math.min(w - pad, x));
    const cy = Math.max(pad, Math.min(h - pad, y));
    c.fillStyle = "rgba(150,165,190,0.6)";
    c.beginPath();
    c.arc(cx, cy, 2.5, 0, Math.PI * 2);
    c.fill();
    c.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    c.textBaseline = "middle";
    const tx = cx > w - 120 ? cx - 108 : cx + 6;
    c.fillText(`${body.name} ↗ ${fmtDistanceFromEarth(body.rEarthM)}`, tx, cy);
  }

  function drawChrome(
    c: CanvasRenderingContext2D,
    w: number,
    h: number,
    state: BodyState[],
  ): void {
    c.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    c.textBaseline = "alphabetic";
    // Scale bar (bottom-left): a real length at the current scale.
    const barM = pickScaleBarMeters(cam.metersPerPixel);
    const barPx = barM / cam.metersPerPixel;
    const bx = 16;
    const by = h - 22;
    c.strokeStyle = "rgba(210,222,238,0.85)";
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(bx, by);
    c.lineTo(bx + barPx, by);
    c.moveTo(bx, by - 4);
    c.lineTo(bx, by + 4);
    c.moveTo(bx + barPx, by - 4);
    c.lineTo(bx + barPx, by + 4);
    c.stroke();
    c.fillStyle = "rgba(210,222,238,0.85)";
    const barLabel = barM >= 0.1 * AU_M ? `${(barM / AU_M).toFixed(barM / AU_M < 3 ? 1 : 0)} AU` : fmtKm(barM / 1000, 0);
    c.fillText(barLabel, bx, by - 8);
    // Provenance + honesty caption (bottom-right) — always on, both modes.
    const cap1 = "COMPUTED EPHEMERIS · TRUE SCALE";
    const cap2 = "Schlyter/van Flandern low-precision (~arcmin) · sub-pixel bodies drawn as 1px";
    c.textAlign = "right";
    c.fillStyle = "rgba(160,175,198,0.8)";
    c.fillText(cap1, w - 16, h - 30);
    c.fillStyle = "rgba(140,155,178,0.65)";
    c.fillText(cap2, w - 16, h - 16);
    // Time + mode (top-left).
    c.textAlign = "left";
    c.fillStyle = "rgba(210,222,238,0.85)";
    c.fillText(new Date(timeMs).toISOString().replace(".000Z", "Z"), 16, 22);
    if (annotate) {
      c.fillStyle = "rgba(160,175,198,0.8)";
      c.fillText("ANNOTATED — markers/labels only; sizes & distances never scaled", 16, 38);
    }
    // Earth-disc size readout keeps the honesty visible while zooming.
    const earthPx = 2 * trueRadiusPx(BODY_RADIUS_M.earth, cam.metersPerPixel);
    c.fillStyle = "rgba(140,155,178,0.65)";
    c.fillText(
      `Earth disc: ${earthPx >= 1 ? earthPx.toFixed(0) + " px" : earthPx.toFixed(3) + " px (sub-pixel)"}`,
      16,
      annotate ? 54 : 38,
    );
    void state;
  }

  // ── interaction ────────────────────────────────────────────────────────────

  function zoomAt(factor: number, px: number, py: number): void {
    const { w, h } = cssSize();
    const before = cam.metersPerPixel;
    const after = clampScale(before * factor);
    if (after === before) return;
    // keep the world point under the cursor fixed
    const wx = cam.centerOffsetXM + (px - w / 2) * before;
    const wy = cam.centerOffsetYM - (py - h / 2) * before;
    cam.metersPerPixel = after;
    cam.centerOffsetXM = wx - (px - w / 2) * after;
    cam.centerOffsetYM = wy + (py - h / 2) * after;
    render();
    const minSide = Math.min(w, h);
    if (!exited && shouldExitToGlobe(after, minSide, exitFraction)) {
      exited = true; // fire once; parent disposes us
      opts.onZoomIntoEarth?.();
    }
  }

  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomAt(Math.pow(1.0015, e.deltaY), e.clientX - rect.left, e.clientY - rect.top);
  };

  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  const onPointerDown = (e: PointerEvent): void => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.style.cursor = "grabbing";
    try { canvas.setPointerCapture(e.pointerId); } catch { /* older engines */ }
  };
  const onPointerMove = (e: PointerEvent): void => {
    if (!dragging) return;
    cam.centerOffsetXM -= (e.clientX - lastX) * cam.metersPerPixel;
    cam.centerOffsetYM += (e.clientY - lastY) * cam.metersPerPixel;
    lastX = e.clientX;
    lastY = e.clientY;
    render();
  };
  const onPointerUp = (): void => {
    dragging = false;
    canvas.style.cursor = "grab";
  };
  const onDblClick = (): void => {
    cam.centerOffsetXM = 0;
    cam.centerOffsetYM = 0;
    render();
  };

  if (interactive) {
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("dblclick", onDblClick);
  }

  let ro: ResizeObserver | null = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => render());
    ro.observe(container);
  }

  render();

  return {
    setTime(date: Date | number): void {
      timeMs = typeof date === "number" ? date : date.getTime();
      render();
    },
    setAnnotate(on: boolean): void {
      annotate = on;
      render();
    },
    setScale(metersPerPixel: number): void {
      cam.metersPerPixel = clampScale(metersPerPixel);
      render();
    },
    getState() {
      return {
        timeMs,
        metersPerPixel: cam.metersPerPixel,
        annotate,
        centerOffsetXM: cam.centerOffsetXM,
        centerOffsetYM: cam.centerOffsetYM,
      };
    },
    render,
    dispose(): void {
      if (disposed) return;
      disposed = true;
      try { ro?.disconnect(); } catch { /* already gone */ }
      if (interactive) {
        canvas.removeEventListener("wheel", onWheel);
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
        canvas.removeEventListener("dblclick", onDblClick);
      }
      try { canvas.remove(); } catch { /* detached */ }
    },
  };
}
