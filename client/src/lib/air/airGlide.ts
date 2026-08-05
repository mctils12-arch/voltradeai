// AIRCRAFT DEAD-RECKONING GLIDE — the SMOOTH SKY precedent applied to planes
// (2026-07-18 human report: "the planes stopped moving … sharp angles where
// the plane would make a curve"). Between 15s feed polls each aircraft is
// extrapolated along its BROADCAST track heading at its BROADCAST ground
// speed from its last REAL position — exactly the honesty model the
// satellite layer uses (satLayer.ts MAX_GLIDE_SEC): first-order motion from
// real data, display-only, never an invented turn. Positions are frozen (not
// glided) when the feed didn't broadcast the needed inputs, and the whole
// glide clamps at MAX_AIR_GLIDE_SEC — past a missed poll the extrapolation
// would be fiction (planes turn), so the display HOLDS rather than flies on.
//
// Pure module: no DOM, no map, hermetic to unit-test. Consumed by
// airLayer.ts (3D silhouettes — shader glide, satLayer pattern) and
// datamap's wireLivePoints (2D symbols — periodic setData of dead-reckoned
// coordinates).

import { lonLatToMercator, wrapMercDelta } from '../orbital/satBuffer.js';
import { metersPerPixel } from '../lod.js';

/** HONESTY CAP (the satellite MAX_GLIDE_SEC, sized for a 15s poll): beyond
 *  ~1.7 missed polls a straight-line extrapolation stops being physics —
 *  planes turn, climb, and land. The display freezes at the cap and the next
 *  real payload snaps everything back to truth. */
export const MAX_AIR_GLIDE_SEC = 25;

/** 2D glide floor: below this zoom a whole 15s poll gap moves a fast plane
 *  under ~3 screen pixels (250 m/s × 15 s ≈ 3.8 km ≈ 2.8 px at z5.5/lat 40)
 *  — re-shipping the source there is pure cost for sub-pixel motion. */
export const AIR_GLIDE_2D_MIN_ZOOM = 5.5;

/** Glide step cadence (ms) — the 2D setData tick and the 3D low-rate
 *  repaint tick share it (~3.3 Hz; steps are sub-pixel until the zoom where
 *  the 3D layer switches to per-frame self-repaint, see
 *  shouldGlidePerFrame). */
export const AIR_GLIDE_STEP_MS = 300;

/** Conservative worst-case ground speed (m/s) for repaint-visibility math —
 *  jetstream-assisted airliners peak near ~340 m/s over the ground. Bound
 *  only, never a per-object clamp (satLayer's MAX_GROUND_TRACK_SPEED_MPS
 *  precedent). */
export const MAX_AIR_SPEED_MPS = 350;

const METERS_PER_DEG_LAT = 111_320;

/**
 * Seconds of glide to apply now: elapsed since the payload anchor, clamped
 * to [0, MAX_AIR_GLIDE_SEC]. Non-finite input (no anchor, broken clock)
 * glides zero — raw tick positions render, the pre-glide behavior.
 * (Mirror of satLayer.glideDtSec with the aircraft cap.)
 */
export function airGlideDtSec(nowMs: number, anchorMs: number): number {
  const dt = (nowMs - anchorMs) / 1000;
  if (!Number.isFinite(dt) || dt <= 0) return 0;
  return dt < MAX_AIR_GLIDE_SEC ? dt : MAX_AIR_GLIDE_SEC;
}

/**
 * Broadcast track + ground speed → degrees-per-second ground velocity for
 * the 2D geojson glide. Returns NULL when the row must stay FROZEN — on
 * ground (taxi motion is sub-pixel and gs is often junk there), no broadcast
 * heading/speed, or outside the mercator band: a frozen plane is honest, a
 * guessed vector is not. Equirectangular local step — over ≤25 s (≤ ~9 km)
 * the great-circle divergence is centimeters.
 */
export function glideDegPerSec(
  lat: number | null | undefined,
  headingDeg: number | null | undefined,
  velocityMs: number | null | undefined,
  onGround: boolean | null | undefined,
): { dLon: number; dLat: number } | null {
  if (onGround) return null;
  if (lat == null || headingDeg == null || velocityMs == null) return null;
  if (!Number.isFinite(lat) || !Number.isFinite(headingDeg) || !Number.isFinite(velocityMs)) return null;
  if (velocityMs <= 0 || lat < -85 || lat > 85) return null;
  const rad = (headingDeg * Math.PI) / 180;
  const cosLat = Math.max(0.05, Math.cos((lat * Math.PI) / 180));
  return {
    dLat: (velocityMs * Math.cos(rad)) / METERS_PER_DEG_LAT,
    dLon: (velocityMs * Math.sin(rad)) / (METERS_PER_DEG_LAT * cosLat),
  };
}

/**
 * The SAME dead-reckoning as glideDegPerSec, expressed as normalized-
 * mercator units per second for the 3D instance buffer (satBuffer velX/velY
 * semantics): a finite difference of lonLatToMercator over one second of
 * travel, antimeridian-wrapped — so the 2D symbols and the 3D silhouettes
 * glide along IDENTICAL paths by construction. {0,0} = frozen.
 */
export function mercVelPerSec(
  lon: number | null | undefined,
  lat: number | null | undefined,
  headingDeg: number | null | undefined,
  velocityMs: number | null | undefined,
  onGround: boolean | null | undefined,
): { vx: number; vy: number } {
  const v = glideDegPerSec(lat, headingDeg, velocityMs, onGround);
  if (!v || lon == null || !Number.isFinite(lon)) return { vx: 0, vy: 0 };
  const m0 = lonLatToMercator(lon, lat as number);
  const m1 = lonLatToMercator(lon + v.dLon, (lat as number) + v.dLat);
  return { vx: wrapMercDelta(m1.x - m0.x), vy: m1.y - m0.y };
}

/**
 * True when the low-rate AIR_GLIDE_STEP_MS tick would move a worst-case
 * plane a VISIBLE step (≥1 px) at this camera — the zoom band where stepped
 * motion reads as jank, so the 3D layer switches to per-frame self-repaint
 * (60 fps glide, the satLayer shouldRequestGlideFrame pattern). Below it the
 * cheap step tick is already sub-pixel-smooth. Fails closed (false) on
 * broken input — the step tick still runs.
 */
export function shouldGlidePerFrame(latDeg: number, zoom: number): boolean {
  if (!Number.isFinite(latDeg) || !Number.isFinite(zoom)) return false;
  return metersPerPixel(latDeg, zoom) < MAX_AIR_SPEED_MPS * (AIR_GLIDE_STEP_MS / 1000);
}

/**
 * dt (seconds) for the curtain's LIVE TAIL — must match what the PLANE
 * renderer is actually showing (repair 2026-08-05, "the curtain is not up
 * to the plane"). In the 2D icon band under the terrain-saturation gate
 * (2026-07-31: overloaded + terrain freezes the 2D glide setData at raw
 * poll positions) the icons are frozen — the tail must freeze WITH them
 * (dt = 0, ending exactly at the raw fix) or it glides ahead of its own
 * plane. Above the hand-off (3D silhouettes glide in-shader every natural
 * frame) wall-clock dt applies as before. No velocity → no glide, ever.
 */
export function tailGlideDtSec(
  nowMs: number,
  anchorMs: number,
  hasVel: boolean,
  frozen2d: boolean,
): number {
  if (!hasVel || frozen2d) return 0;
  return airGlideDtSec(nowMs, anchorMs);
}
