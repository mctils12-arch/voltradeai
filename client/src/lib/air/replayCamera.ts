// REPLAY CAMERA — the Time Machine's field-of-view query box (FLIGHT
// PROGRAM, human directive 2026-09-28: "rewind and see ALL planes in my
// field of view … at any pan/tilt/zoom").
//
// Pure math, no DOM, no MapLibre import — unit-tested with `npx tsx --test`.
//
// WHY NOT map.getBounds(): the bounds of the CURRENT transform are wrong
// twice over for a replay query. (1) Mid-animation they describe a
// transient view the camera is flying THROUGH (Law II.4: "prefetch from the
// target, not the current position") — so this module computes the box
// from a camera POSE the caller supplies: the destination of a flight we
// initiated, or the settled pose once the gesture has stopped (see
// CameraTargetTracker). (2) With pitch, the visible ground stretches toward
// the horizon; getBounds() of a steeply pitched view either under-covers the
// far field or (above the horizon) runs off to absurd extents. Here the
// footprint is the ground intersection of rays through the screen's corners
// and edge midpoints, each clamped at the geometric HORIZON distance for the
// camera height and at REPLAY_CAMERA.MAX_DIST_KM.
//
// STACK NOTE (factual): the /data map's camera is MapLibre-owned — the
// render/zoomInput spring drives the celestial views, not this map — so the
// "target" is (a) the explicit destination of our own easeTo (close-approach
// framing), else (b) the pose once it has been still for SETTLE_MS. Both
// are reads of intent, never of a transient frame.
//
// Camera model = MapLibre's: vertical FOV 36.87°, 512-px world tiles,
// center-to-camera distance 0.5·H / tan(fov/2) px, pitch from nadir.

import { GLOBE_RADIUS_M } from "../orbital/occlusion.ts";

export interface CameraPose {
  lon: number;
  lat: number;
  zoom: number;
  /** degrees from nadir (0 = straight down) */
  pitch: number;
  /** degrees clockwise from north of the screen's up direction */
  bearing: number;
}

export interface ViewportPx {
  widthPx: number;
  heightPx: number;
}

/** w > e means the box crosses the antimeridian (server/aircraftWindow.ts
 *  lonInBBox supports the seam window). */
export interface BBox { w: number; s: number; e: number; n: number }

export const REPLAY_CAMERA = {
  /** MapLibre's default vertical field of view */
  FOV_DEG: 36.87,
  /** hard far-field clamp for pitched views (km along the ground) */
  MAX_DIST_KM: 1500,
  /** margin ring added around the visible box (fraction of each span) */
  MARGIN_FRAC: 0.25,
  /** below this zoom the whole visible globe is the query (world box) */
  WORLD_ZOOM: 2.5,
  /** a pose unchanged this long is the settled target */
  SETTLE_MS: 250,
  /** latitude clamp (web-mercator) */
  MAX_LAT: 85,
} as const;

// ONE mean radius for the repo (orbital/occlusion.ts, MapLibre's own value —
// dup_precise_literal ratchet); the tile-scale circumference is WGS84
// equatorial (Web Mercator), which is what MapLibre's worldSize uses.
const R_EARTH_M = GLOBE_RADIUS_M;
const WGS84_EQUATORIAL_M = 6378137;
const EARTH_CIRC_M = 2 * Math.PI * WGS84_EQUATORIAL_M;
const D2R = Math.PI / 180;

export const WORLD_BBOX: BBox = { w: -180, s: -REPLAY_CAMERA.MAX_LAT, e: 180, n: REPLAY_CAMERA.MAX_LAT };

/** meters per CSS pixel at the pose's center (MapLibre 512-px tiles) */
export function metersPerPixel(lat: number, zoom: number): number {
  return (EARTH_CIRC_M * Math.cos(lat * D2R)) / (512 * Math.pow(2, zoom));
}

/** geometric horizon distance (m) for an eye height h (m): sqrt(2Rh + h²) */
export function horizonDistM(heightM: number): number {
  const h = Math.max(0, heightM);
  return Math.sqrt(2 * R_EARTH_M * h + h * h);
}

export interface FootprintPoint {
  lon: number;
  lat: number;
  /** true when the ray was above the horizon or beyond the clamp distance */
  clamped: boolean;
}

/**
 * Ground footprint of the view: rays through the 4 screen corners and the 4
 * edge midpoints, intersected with a flat ground plane at the pose center,
 * clamped at min(horizon, maxDistKm) measured from the camera's ground point.
 */
export function groundFootprint(
  pose: CameraPose,
  vp: ViewportPx,
  opts: { fovDeg?: number; maxDistKm?: number } = {},
): { points: FootprintPoint[]; cameraHeightM: number; clampDistM: number } {
  const fov = (opts.fovDeg ?? REPLAY_CAMERA.FOV_DEG) * D2R;
  const W = Math.max(1, vp.widthPx), H = Math.max(1, vp.heightPx);
  const p = Math.min(89, Math.max(0, pose.pitch)) * D2R;
  const b = pose.bearing * D2R;
  const mpp = metersPerPixel(pose.lat, pose.zoom);
  const fPx = (H / 2) / Math.tan(fov / 2);
  const Dm = fPx * mpp; // camera → center distance, meters
  const camH = Dm * Math.cos(p);
  const camBack = Dm * Math.sin(p); // camera ground point sits this far BEHIND center
  const clampDistM = Math.min((opts.maxDistKm ?? REPLAY_CAMERA.MAX_DIST_KM) * 1000, horizonDistM(camH));

  const samples: Array<[number, number]> = [
    [-W / 2, H / 2], [0, H / 2], [W / 2, H / 2],      // top edge (far field)
    [W / 2, 0], [W / 2, -H / 2], [0, -H / 2],          // right, bottom
    [-W / 2, -H / 2], [-W / 2, 0],                     // bottom-left, left
  ];
  const points: FootprintPoint[] = [];
  for (const [sx, sy] of samples) {
    // ray direction in (forward, right, up) — px units
    const dUp = -fPx * Math.cos(p) + sy * Math.sin(p);
    const dFwd = fPx * Math.sin(p) + sy * Math.cos(p);
    const dRight = sx;
    const horizLen = Math.hypot(dFwd, dRight);
    let gF: number, gR: number; // ground offset from the CAMERA ground point, meters
    let clamped = false;
    if (dUp < -1e-9) {
      const t = camH / -dUp; // meters per px-unit
      gF = t * dFwd;
      gR = t * dRight;
      const dist = Math.hypot(gF, gR);
      if (dist > clampDistM) {
        const k = clampDistM / dist;
        gF *= k; gR *= k;
        clamped = true;
      }
    } else {
      // at/above the horizon: the ground along this azimuth, out to the clamp
      const k = horizLen > 1e-9 ? clampDistM / horizLen : 0;
      gF = dFwd * k;
      gR = dRight * k;
      clamped = true;
    }
    const fwd = gF - camBack; // relative to the pose center
    const east = fwd * Math.sin(b) + gR * Math.cos(b);
    const north = fwd * Math.cos(b) - gR * Math.sin(b);
    const lat = pose.lat + (north / R_EARTH_M) / D2R;
    const lon = pose.lon + (east / (R_EARTH_M * Math.max(0.01, Math.cos(pose.lat * D2R)))) / D2R;
    points.push({ lon, lat, clamped });
  }
  return { points, cameraHeightM: camH, clampDistM };
}

/** the camera eye: its ground point (behind the center by D·sin(pitch)
 *  along the view bearing) and height D·cos(pitch) — the LOD nearness origin */
export function cameraEye(pose: CameraPose, vp: ViewportPx, fovDeg: number = REPLAY_CAMERA.FOV_DEG): {
  lat: number; lon: number; heightM: number;
} {
  const p = Math.min(89, Math.max(0, pose.pitch)) * D2R;
  const b = pose.bearing * D2R;
  const fPx = (Math.max(1, vp.heightPx) / 2) / Math.tan((fovDeg * D2R) / 2);
  const Dm = fPx * metersPerPixel(pose.lat, pose.zoom);
  const back = Dm * Math.sin(p);
  const north = -back * Math.cos(b), east = -back * Math.sin(b);
  return {
    lat: pose.lat + (north / R_EARTH_M) / D2R,
    lon: wrapLon(pose.lon + (east / (R_EARTH_M * Math.max(0.01, Math.cos(pose.lat * D2R)))) / D2R),
    heightM: Dm * Math.cos(p),
  };
}

/** wrap a longitude into [-180, 180] */
export function wrapLon(lon: number): number {
  const x = ((lon + 180) % 360 + 360) % 360 - 180;
  return x === -180 && lon > 0 ? 180 : x;
}

/** normalize an unwrapped [w, e] (possibly beyond ±180) into a BBox lon pair */
function normLonRange(w: number, e: number): { w: number; e: number } {
  if (e - w >= 360) return { w: -180, e: 180 };
  return { w: wrapLon(w), e: wrapLon(e) };
}

export interface QueryBoxes {
  /** what is on screen (at the target pose) */
  visible: BBox;
  /** visible + margin ring — what to fetch */
  query: BBox;
  /** true when the far field was clamped (horizon / max distance) */
  clamped: boolean;
  /** true when the view is zoomed out to the whole globe */
  world: boolean;
}

/**
 * The replay query boxes for a camera pose: the visible footprint's bbox and
 * the same expanded by a margin ring. World view (low zoom or a footprint
 * spanning > 120° of latitude) → the whole-world box.
 */
export function cameraQueryBBox(
  pose: CameraPose,
  vp: ViewportPx,
  opts: { fovDeg?: number; maxDistKm?: number; marginFrac?: number } = {},
): QueryBoxes {
  if (!(pose.zoom >= REPLAY_CAMERA.WORLD_ZOOM)) {
    return { visible: { ...WORLD_BBOX }, query: { ...WORLD_BBOX }, clamped: false, world: true };
  }
  const fp = groundFootprint(pose, vp, opts);
  let s = Infinity, n = -Infinity, w = Infinity, e = -Infinity;
  // unwrap longitudes around the pose center so a seam-crossing view stays
  // one contiguous interval
  for (const pt of fp.points) {
    const lon = pose.lon + wrapLon(pt.lon - pose.lon);
    s = Math.min(s, pt.lat); n = Math.max(n, pt.lat);
    w = Math.min(w, lon); e = Math.max(e, lon);
  }
  if (n - s > 120) {
    return { visible: { ...WORLD_BBOX }, query: { ...WORLD_BBOX }, clamped: true, world: true };
  }
  const clamped = fp.points.some((p) => p.clamped);
  const m = opts.marginFrac ?? REPLAY_CAMERA.MARGIN_FRAC;
  const dLat = Math.max(0.02, (n - s) * m);
  const dLon = Math.max(0.02, (e - w) * m);
  const L = REPLAY_CAMERA.MAX_LAT;
  const vis = normLonRange(w, e);
  const q = normLonRange(w - dLon, e + dLon);
  return {
    visible: { w: vis.w, s: Math.max(-L, s), e: vis.e, n: Math.min(L, n) },
    query: { w: q.w, s: Math.max(-L, s - dLat), e: q.e, n: Math.min(L, n + dLat) },
    clamped,
    world: false,
  };
}

/** lon interval [w, e'] with e' >= w (seam boxes unwrapped by +360) */
function lonSpan(b: BBox): [number, number] {
  if (b.w <= b.e) return [b.w, b.e];
  return [b.w, b.e + 360];
}

/** does `outer` fully contain `inner`? (antimeridian-aware) */
export function bboxContains(outer: BBox, inner: BBox): boolean {
  if (inner.s < outer.s - 1e-9 || inner.n > outer.n + 1e-9) return false;
  const [ow, oe] = lonSpan(outer);
  if (oe - ow >= 360 - 1e-9) return true;
  const [iw, ie] = lonSpan(inner);
  for (const k of [-360, 0, 360]) {
    if (iw + k >= ow - 1e-9 && ie + k <= oe + 1e-9) return true;
  }
  return false;
}

/** area in square degrees (lon span × lat span — a ratio metric only) */
export function bboxAreaDeg2(b: BBox): number {
  const [w, e] = lonSpan(b);
  return Math.max(0, e - w) * Math.max(0, b.n - b.s);
}

/**
 * Does the target view need a new window read? Yes when (a) nothing was
 * fetched, (b) the visible box leaves the fetched (margin-padded) box, or
 * (c) the fetched response was hex-CAPPED and the view has zoomed in to
 * under a quarter of its area (the server dropped planes there that a
 * narrower read would return — honest detail, not decoration).
 */
export function needsRefetch(
  fetched: { query: BBox; capped: boolean } | null,
  target: QueryBoxes,
): boolean {
  if (!fetched) return true;
  if (!bboxContains(fetched.query, target.visible)) return true;
  if (fetched.capped && bboxAreaDeg2(target.visible) < 0.25 * bboxAreaDeg2(fetched.query)) return true;
  return false;
}

/** "w,s,e,n" at 4 decimals — the route's bbox param */
export function bboxParam(b: BBox): string {
  return [b.w, b.s, b.e, b.n].map((x) => x.toFixed(4)).join(",");
}

const poseEq = (a: CameraPose, b: CameraPose): boolean =>
  Math.abs(a.lon - b.lon) < 1e-7 && Math.abs(a.lat - b.lat) < 1e-7 &&
  Math.abs(a.zoom - b.zoom) < 1e-4 && Math.abs(a.pitch - b.pitch) < 0.01 &&
  Math.abs(a.bearing - b.bearing) < 0.01;

/**
 * The camera TARGET, read once per frame (frameCore STREAM priority). Returns
 * the pose the query should be computed from, or null while the destination
 * is unknown (a user gesture still in progress).
 */
export class CameraTargetTracker {
  private last: CameraPose | null = null;
  private stableSince = 0;
  private explicit: { pose: CameraPose; until: number } | null = null;

  constructor(private readonly settleMs: number = REPLAY_CAMERA.SETTLE_MS) {}

  /** we initiated a camera flight: its DESTINATION is the target from now
   *  (for up to holdMs, after which the settled pose takes over) */
  setExplicitTarget(pose: CameraPose, nowMs: number, holdMs: number): void {
    this.explicit = { pose: { ...pose }, until: nowMs + Math.max(0, holdMs) };
  }

  update(pose: CameraPose, nowMs: number): CameraPose | null {
    if (this.explicit) {
      if (nowMs < this.explicit.until) {
        this.last = { ...pose };
        this.stableSince = nowMs;
        return this.explicit.pose;
      }
      this.explicit = null;
    }
    if (!this.last || !poseEq(this.last, pose)) {
      this.last = { ...pose };
      this.stableSince = nowMs;
    }
    return nowMs - this.stableSince >= this.settleMs ? this.last : null;
  }
}
