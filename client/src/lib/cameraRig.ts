// DAMPED ORBIT CAMERA RIG — the handoff-approved button-driven navigation
// model (design_handoff_flight_track_3d, installed 2026-07-20), expressed
// on MapLibre's camera: {bearing, pitch, zoom, center} each carry a GOAL,
// and every frame the value moves toward its goal by the prototype's
// damping law  v += (goal − v) × (1 − e^(−7·dt)).
//
// Mapping from the prototype's {target, heading, pitch, distance} rig:
//  - heading (rad, direction the camera faces) ≡ MapLibre bearing (deg).
//  - pitch: the prototype measures ELEVATION from horizontal (2° = grazing,
//    88° = top-down); MapLibre pitch measures tilt from vertical. mapPitch
//    = 90° − elevation, so the prototype's 2°–88° elevation clamp is the
//    SAME 2°–88° range in MapLibre pitch (map must allow maxPitch 88).
//  - distance ↔ zoom: MapLibre keeps the camera→center distance a fixed
//    pixel count (1.5 × viewport height at the default fov), so distance
//    in meters = 1.5·h·(mercator meters per pixel). A multiplicative
//    distance change ×e^k is EXACTLY a zoom change of −k/ln2 — the
//    prototype's exponential zoom rates translate losslessly.
//  - pan: 0.9 × distance per second, screen-relative (up = away from the
//    camera), converted to ground degrees at the current latitude.
//
// Exact prototype rates (tuned, final per the handoff):
//  rotate ±1.5 rad/s · tilt ±0.9 rad/s · zoom ×e^(∓1.6·dt) ·
//  pan 0.9×dist/s · wheel ×e^(0.0011·ΔY) · drag-rotate 0.005 rad/px ·
//  drag-tilt 0.004 rad/px · damping 7/s.
//
// Pure math module — no MapLibre import; the caller owns the map handle
// and the rAF loop. Unit-testable with `npx tsx --test`.

const R2D = 180 / Math.PI;
const LN2 = Math.LN2;

/** damping factor per second (prototype: k = 1 − e^(−7·dt)). */
export const RIG_DAMPING_PER_S = 7;

/** rotate rate, deg/s (prototype 1.5 rad/s). */
export const RIG_ROTATE_DEG_S = 1.5 * R2D;
/** tilt rate, deg/s (prototype 0.9 rad/s). */
export const RIG_TILT_DEG_S = 0.9 * R2D;
/** zoom rate in zoom-levels/s (prototype dist ×e^(∓1.6·dt)). */
export const RIG_ZOOM_LEVELS_S = 1.6 / LN2;
/** pan rate: fraction of camera distance per second. */
export const RIG_PAN_DIST_PER_S = 0.9;
/** wheel: zoom-level change per deltaY unit (prototype ×e^(0.0011·ΔY)). */
export const RIG_WHEEL_ZOOM_PER_DY = 0.0011 / LN2;
/** drag rotate, deg/px (prototype 0.005 rad/px). */
export const RIG_DRAG_ROTATE_DEG_PX = 0.005 * R2D;
/** drag tilt, deg/px (prototype 0.004 rad/px). */
export const RIG_DRAG_TILT_DEG_PX = 0.004 * R2D;

/** pitch clamp in MapLibre degrees (prototype elevation 2°–88° = the same
 *  numeric range after the 90°−el mapping; 2 = near-top-down here). The
 *  UPPER bound is the extreme grazing tilt the curtain must survive. */
export const RIG_PITCH_MIN = 2;
export const RIG_PITCH_MAX = 88;

export interface RigState {
  bearing: number;
  pitch: number;
  zoom: number;
  lng: number;
  lat: number;
}

export interface Rig {
  cur: RigState;
  goal: RigState;
}

export const clampPitch = (p: number): number =>
  Math.min(RIG_PITCH_MAX, Math.max(RIG_PITCH_MIN, p));

export const clampLat = (lat: number): number => Math.min(85, Math.max(-85, lat));

/** One damping step toward the goals (mutates cur; returns true while any
 *  channel is still visibly moving — the caller's rAF keep-alive). */
export function stepRig(rig: Rig, dtSec: number): boolean {
  const k = 1 - Math.exp(-RIG_DAMPING_PER_S * Math.max(0, dtSec));
  const c = rig.cur, g = rig.goal;
  c.bearing += (g.bearing - c.bearing) * k;
  c.pitch += (g.pitch - c.pitch) * k;
  c.zoom += (g.zoom - c.zoom) * k;
  c.lng += (g.lng - c.lng) * k;
  c.lat += (g.lat - c.lat) * k;
  return (
    Math.abs(g.bearing - c.bearing) > 0.01 ||
    Math.abs(g.pitch - c.pitch) > 0.01 ||
    Math.abs(g.zoom - c.zoom) > 0.0005 ||
    Math.abs(g.lng - c.lng) > 1e-7 ||
    Math.abs(g.lat - c.lat) > 1e-7
  );
}

const EARTH_CIRCUM_M = 40075016.686;
const M_PER_DEG_LAT = 111319.49;

/** Camera→center distance in meters at a zoom/latitude/viewport-height —
 *  MapLibre's fixed 1.5×height-px camera distance × ground meters/px
 *  (512px world tiles). The prototype's "0.9 × distance per second" pan
 *  and its 900 m dolly feel translate through this. */
export function camDistanceMeters(zoom: number, latDeg: number, viewportHeightPx: number): number {
  const mpp = (Math.cos((latDeg * Math.PI) / 180) * EARTH_CIRCUM_M) / (512 * Math.pow(2, zoom));
  return 1.5 * viewportHeightPx * mpp;
}

/** Screen-relative pan: fx = +right, fz = +toward viewer (prototype panFn
 *  signs: up = (0,−1) = away from camera). Returns the lng/lat goal delta
 *  for one tick of `dtSec` at the given camera state. */
export function panDelta(
  bearingDeg: number,
  fx: number,
  fz: number,
  zoom: number,
  latDeg: number,
  viewportHeightPx: number,
  dtSec: number,
): { dLng: number; dLat: number } {
  const dist = camDistanceMeters(zoom, latDeg, viewportHeightPx);
  const s = dist * RIG_PAN_DIST_PER_S * dtSec;
  const b = (bearingDeg * Math.PI) / 180;
  // world-frame meters: screen-up at bearing b points toward (sin b, cos b)
  const east = (fx * Math.cos(b) - fz * Math.sin(b)) * s;
  const north = (-fx * Math.sin(b) - fz * Math.cos(b)) * s;
  const cosLat = Math.max(0.05, Math.cos((latDeg * Math.PI) / 180));
  return { dLng: east / (M_PER_DEG_LAT * cosLat), dLat: north / M_PER_DEG_LAT };
}

/** Shortest-way bearing delta that lands on north (compass click). */
export function bearingDeltaToNorth(bearingDeg: number): number {
  const w = ((bearingDeg % 360) + 540) % 360 - 180;
  return -w;
}

/** Compass-dial drag: pointer angle around the dial center → new bearing
 *  goal (grab angle → heading follows the delta, full 360°). The ring
 *  renders at rotate(−bearing), and the ring must follow the pointer:
 *  a clockwise screen drag (atan2 angle increasing, y-down) decreases the
 *  bearing — the prototype's `gheading = grabHeading − delta` exactly. */
export function dialBearing(
  startBearingDeg: number,
  grabAngleRad: number,
  nowAngleRad: number,
): number {
  return startBearingDeg - (nowAngleRad - grabAngleRad) * R2D;
}
