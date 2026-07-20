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
//    SAME numeric range in MapLibre pitch (upper bound tightened to
//    RIG_PITCH_MAX below — see the freeze note there).
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

/** pitch clamp in MapLibre degrees (2 = near-top-down). The UPPER bound is
 *  the extreme grazing tilt the curtain must survive. The prototype's 88°
 *  (2° elevation) FROZE production (2026-07-20 live report): MapLibre marks
 *  pitch >~85° experimental, and near-horizon views explode the visible
 *  tile cover — every frame queues hundreds of tile loads. 84° stays inside
 *  the supported envelope and the curtain's double-sided/no-depth-write fix
 *  is angle-independent (probe-verified at 84°). */
export const RIG_PITCH_MIN = 2;
export const RIG_PITCH_MAX = 84;

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
 *  channel is still visibly moving — the caller's rAF keep-alive). Channels
 *  SNAP onto the goal once inside their visible threshold: exponential
 *  damping never actually arrives, and a sub-pixel asymptote kept the rAF
 *  loop alive for seconds after every input (settle-flake, 2026-07-20). */
export function stepRig(rig: Rig, dtSec: number): boolean {
  const k = 1 - Math.exp(-RIG_DAMPING_PER_S * Math.max(0, dtSec));
  const c = rig.cur, g = rig.goal;
  const chase = (cur: number, goal: number, eps: number): number => {
    const next = cur + (goal - cur) * k;
    return Math.abs(goal - next) <= eps ? goal : next;
  };
  c.bearing = chase(c.bearing, g.bearing, 0.01);
  c.pitch = chase(c.pitch, g.pitch, 0.01);
  c.zoom = chase(c.zoom, g.zoom, 0.0005);
  c.lng = chase(c.lng, g.lng, 1e-7);
  c.lat = chase(c.lat, g.lat, 1e-7);
  return (
    c.bearing !== g.bearing ||
    c.pitch !== g.pitch ||
    c.zoom !== g.zoom ||
    c.lng !== g.lng ||
    c.lat !== g.lat
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

/** Button/wheel zoom ceiling — beyond ~z18 the imagery basemap serves
 *  "map data not yet available" placeholder tiles (live report 2026-07-20:
 *  hold-zoom flew uncontrollably into the blank void). The buttons stop
 *  where real imagery still exists; pinch/native zoom keeps the map's own
 *  maxZoom for users who deliberately go deeper. */
export const RIG_BUTTON_MAX_ZOOM = 17.5;

/** Max zoom-levels the GOAL may run ahead of the camera. The prototype
 *  accumulates held input into the goal and lets damping chase it — over a
 *  40km demo world that coast was bounded; over a planet a held button (or
 *  trackpad momentum) piled levels onto the goal and the camera kept flying
 *  long after release. 0.9 levels keeps the damped feel with an instant,
 *  controllable stop. */
export const RIG_ZOOM_MAX_AHEAD = 0.9;

/** One button/wheel zoom-goal step: bounded by the map's own zoom range,
 *  the button ceiling, and the goal-ahead cap. Pure — the runaway-zoom
 *  regression test lives on this. */
export function zoomGoalStep(
  curZoom: number,
  goalZoom: number,
  delta: number,
  minZoom: number,
  maxZoom: number,
): number {
  const ceiling = Math.min(maxZoom, RIG_BUTTON_MAX_ZOOM);
  const bounded = Math.min(ceiling, Math.max(minZoom, goalZoom + delta));
  return Math.min(curZoom + RIG_ZOOM_MAX_AHEAD, Math.max(curZoom - RIG_ZOOM_MAX_AHEAD, bounded));
}

/** Hold-button registry keyed by BUTTON, not by function identity.
 *  THE LATCH BUG (live report 2026-07-20 round 3: "clicked rotate left and
 *  it started rotating constantly and would not stop"): the page re-renders
 *  every data tick, which re-creates the button handlers — press stored
 *  closure A, release tried to remove closure B, and A spun forever. Keys
 *  survive re-renders; release/clear can never miss. */
export interface HoldRegistry {
  press(key: string, fn: (dt: number) => void): void;
  release(key: string): void;
  /** global failsafe: any pointerup/blur/hidden clears everything. */
  clear(): void;
  tick(dt: number): void;
  size(): number;
}

export function makeHoldRegistry(): HoldRegistry {
  const m = new Map<string, (dt: number) => void>();
  return {
    press: (key, fn) => m.set(key, fn),
    release: (key) => m.delete(key),
    clear: () => m.clear(),
    tick: (dt) => m.forEach((fn) => fn(dt)),
    size: () => m.size,
  };
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
