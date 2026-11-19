// SELECTED-PLANE MOTION — one model for where the selected aircraft is DRAWN
// (owner 2026-11-19: "I don't like that the bluish line runs in front of the
// plane and the lag that it jumps when it catches signal on the plane ... we
// know what speed and heading it going").
//
// Before this module the selected plane had TWO positions. Its live track was
// extended to the newest fix of the one-plane lookup, dead-reckoned from that
// fix's own time; the plane itself was drawn from its row in the area feed,
// which can be tens of seconds older, was treated as true at the moment it
// arrived, and froze 25 s later. The line's head therefore ran ahead of the
// plane, and each new feed answer moved the plane in one step.
//
// Now there is one position, and three rules:
//
//   1. RECKON from the newest REAL fix, at that fix's OWN time: last position
//      + last reported ground speed along the last reported track x the time
//      since the fix. Straight ahead only — a turn is never invented.
//   2. A new fix is a new TARGET, never an assignment. The drawn position
//      keeps its place and its velocity and closes the difference as a
//      critically damped spring on the offset (drawn - reckoned). The closing
//      speed is limited, and while the aircraft is moving the drawn position
//      never travels backward along its track: when the reckoning over-ran,
//      the plane slows and the real position catches up.
//   3. HONESTY by age of the last real fix (the limits are the server
//      estimate's own, shared/aircraftEstimate.ts, so the two cannot drift):
//        up to 30 s    live: motion between fixes.
//        30 s - 120 s  still reckoned, and MARKED an estimate (card text,
//                      drawn faded). Under 40 kt nothing is extrapolated past
//                      30 s (taxi, hover, junk speeds).
//        past 120 s    HOLDS at the 120 s point and says so. A fix that is
//                      already past its limit when the plane is selected is
//                      not extrapolated at all: it is drawn where it was seen.
//
// The integrator is this module's own because the spring is a 2-D vector with
// two constraints (speed limit, no reversal) that frameCore's scalar Spring
// cannot express. It uses the same semi-implicit Euler step at the same fixed
// sub-step, and it is DRIVEN by frameCore's loop (datamap registers one
// callback) — it owns no timer and no requestAnimationFrame.
//
// Pure: no DOM, no map, no clock of its own. Every time is passed in, in ONE
// monotonic domain (performance.now()).

import { glideDegPerSec } from './airGlide.js';
import { FRAME, lerpTowards } from '../../render/frameCore.js';
import {
  DEAD_RECKONING_MAX_SEC, DEAD_RECKONING_SHORT_SEC, ESTIMATE_MIN_GS_KT,
} from '../../../../shared/aircraftEstimate.js';

/** Past this age of the last real fix the drawn position is labelled an estimate. */
export const PLANE_ESTIMATED_AFTER_SEC = DEAD_RECKONING_SHORT_SEC;
/** Past this age the reckoning stops and the drawn position holds. */
export const PLANE_HOLD_AFTER_SEC = DEAD_RECKONING_MAX_SEC;
/** Below this ground speed (m/s) nothing is extrapolated past the short window. */
export const PLANE_SLOW_MPS = ESTIMATE_MIN_GS_KT * 0.514444;

/** Spring rate of the correction (1/s): critically damped, settles in ~1.5 s. */
export const MOTION_OMEGA = 3;
/** An ordinary correction closes no faster than this fraction of ground speed,
 *  so the drawn plane never exceeds (1 + this) x its ground speed. */
export const MOTION_MAX_CORR_FRAC = 0.75;
/** While correcting backward the drawn plane keeps at least this fraction of
 *  its ground speed along its track: it slows, it never reverses. */
export const MOTION_MIN_FWD_FRAC = 0.25;
/** A gap too large for the ordinary limit (a plane re-acquired after a long
 *  silence) is closed over about this long instead: fast, but continuous. */
export const MOTION_MAX_CATCHUP_SEC = 4;
/** Floor of the closing speed (m/s), so a parked plane's correction ends. */
export const MOTION_MIN_CORR_MPS = 15;
/** Drawn heading and altitude ease to a new fix's values (half-lives, ms). */
export const MOTION_HEADING_HALF_LIFE_MS = 150;
export const MOTION_ALT_HALF_LIFE_MS = 300;
/** A step longer than this (a resumed tab) ends any correction outright. */
export const MOTION_MAX_STEP_MS = 2000;

const M_PER_DEG = 111_320;
const OFF_EPS_M = 0.05;

/** One REAL position report. */
export interface MotionFix {
  lon: number;
  lat: number;
  /** metres MSL; null = not reported */
  altM: number | null;
  /** broadcast track, degrees; null = not reported */
  headingDeg: number | null;
  /** broadcast ground speed, m/s; null = not reported */
  speedMs: number | null;
  onGround: boolean;
}

export type MotionPhase = 'live' | 'estimated' | 'held';

export interface PlaneMotion {
  fix: MotionFix;
  /** when the fix was TRUE, in the caller's monotonic clock (ms) */
  anchorMs: number;
  /** degrees per second; null = this fix is not extrapolated */
  vel: { dLon: number; dLat: number } | null;
  /** the same velocity in m/s east / north (0 when vel is null) */
  vE: number;
  vN: number;
  /** how long this fix may be reckoned forward, seconds */
  limitSec: number;
  /** correction offset (drawn - reckoned), metres east / north, and its rate */
  offE: number;
  offN: number;
  offVE: number;
  offVN: number;
  /** closing-speed limit chosen when the current target was set, m/s */
  rateMps: number;
  /** drawn heading (degrees) and altitude (m): eased, never stepped */
  headingDeg: number | null;
  altM: number | null;
  /** the last drawn position */
  lon: number;
  lat: number;
  lastMs: number;
}

export interface MotionOut {
  lon: number;
  lat: number;
  altM: number | null;
  headingDeg: number | null;
  phase: MotionPhase;
  /** age of the last real fix, seconds */
  ageSec: number;
  /** the fix carried a usable speed and track (it is being, or was, reckoned) */
  reckoned: boolean;
  /** the drawn position is still closing on the reckoned one */
  correcting: boolean;
  limitSec: number;
}

/** How long a fix may be flown straight ahead: 0 without a usable vector,
 *  the short window for a slow aircraft, else the full one. */
export function reckonLimitSec(speedMs: number | null | undefined, hasVector: boolean): number {
  if (!hasVector || speedMs == null || !Number.isFinite(speedMs)) return 0;
  return speedMs >= PLANE_SLOW_MPS ? PLANE_HOLD_AFTER_SEC : PLANE_ESTIMATED_AFTER_SEC;
}

function ageSecOf(anchorMs: number, nowMs: number): number {
  const a = (nowMs - anchorMs) / 1000;
  return Number.isFinite(a) && a > 0 ? a : 0;
}

/** The reckoned position at nowMs: straight ahead from the fix for
 *  min(age, limit) seconds. `moving` = the reckoning is still advancing. */
export function reckonAt(m: PlaneMotion, nowMs: number): { lon: number; lat: number; ageSec: number; moving: boolean } {
  const ageSec = ageSecOf(m.anchorMs, nowMs);
  if (!m.vel) return { lon: m.fix.lon, lat: m.fix.lat, ageSec, moving: false };
  const t = Math.min(ageSec, m.limitSec);
  return { lon: m.fix.lon + m.vel.dLon * t, lat: m.fix.lat + m.vel.dLat * t, ageSec, moving: ageSec < m.limitSec };
}

/** live / estimated / held from the age of the last real fix. */
export function motionPhase(ageSec: number, reckoned: boolean, limitSec: number): MotionPhase {
  if (ageSec <= PLANE_ESTIMATED_AFTER_SEC) return 'live';
  return reckoned && ageSec <= limitSec ? 'estimated' : 'held';
}

function setTarget(m: PlaneMotion, fix: MotionFix, anchorMs: number, nowMs: number, allowStale: boolean): void {
  const vel = glideDegPerSec(fix.lat, fix.headingDeg, fix.speedMs, fix.onGround);
  let limit = reckonLimitSec(fix.speedMs, vel != null);
  // a fix already past its limit when first seen is not extrapolated at all
  const tooOld = !allowStale && ageSecOf(anchorMs, nowMs) > limit;
  m.fix = fix;
  m.anchorMs = anchorMs;
  m.vel = tooOld ? null : vel;
  if (!m.vel) limit = 0;
  m.limitSec = limit;
  const rad = ((fix.headingDeg ?? 0) * Math.PI) / 180;
  const s = m.vel ? (fix.speedMs as number) : 0;
  m.vE = s * Math.sin(rad);
  m.vN = s * Math.cos(rad);
}

function wrapLonDeg(d: number): number {
  return d > 180 ? d - 360 : d < -180 ? d + 360 : d;
}

function setOffsetFromDrawn(m: PlaneMotion, drawnLon: number, drawnLat: number, nowMs: number): void {
  const r = reckonAt(m, nowMs);
  const cos = Math.max(0.05, Math.cos((r.lat * Math.PI) / 180));
  m.offE = wrapLonDeg(drawnLon - r.lon) * M_PER_DEG * cos;
  m.offN = (drawnLat - r.lat) * M_PER_DEG;
  const gap = Math.hypot(m.offE, m.offN);
  const speed = Math.hypot(m.vE, m.vN);
  m.rateMps = Math.max(MOTION_MAX_CORR_FRAC * speed, gap / MOTION_MAX_CATCHUP_SEC, MOTION_MIN_CORR_MPS);
}

/**
 * Start the model from a first fix. `drawnAt` is where the plane is on screen
 * right now (its feed icon), so selecting it never moves it in one step: the
 * model starts there and closes on the reckoned position.
 */
export function createPlaneMotion(
  fix: MotionFix,
  anchorMs: number,
  nowMs: number,
  drawnAt?: { lon: number; lat: number } | null,
): PlaneMotion {
  const m: PlaneMotion = {
    fix, anchorMs, vel: null, vE: 0, vN: 0, limitSec: 0,
    offE: 0, offN: 0, offVE: 0, offVN: 0, rateMps: MOTION_MIN_CORR_MPS,
    headingDeg: fix.headingDeg, altM: fix.altM,
    lon: fix.lon, lat: fix.lat, lastMs: nowMs,
  };
  setTarget(m, fix, anchorMs, nowMs, false);
  const r = reckonAt(m, nowMs);
  m.lon = r.lon;
  m.lat = r.lat;
  if (drawnAt && Number.isFinite(drawnAt.lon) && Number.isFinite(drawnAt.lat)) {
    setOffsetFromDrawn(m, drawnAt.lon, drawnAt.lat, nowMs);
    m.lon = drawnAt.lon;
    m.lat = drawnAt.lat;
  }
  return m;
}

/**
 * A newer REAL fix arrived. The drawn position and its velocity are kept; only
 * the target changes. Call stepPlaneMotion first if frames may have elapsed.
 */
export function retargetPlaneMotion(m: PlaneMotion, fix: MotionFix, anchorMs: number, nowMs: number): void {
  stepPlaneMotion(m, nowMs); // bring the drawn position up to this instant
  const before = reckonAt(m, nowMs);
  const dVE = (before.moving ? m.vE : 0) + m.offVE; // drawn velocity, m/s
  const dVN = (before.moving ? m.vN : 0) + m.offVN;
  // once reckoning, a late fix is still reckoned to its limit: dropping the
  // vector here would pull the drawn plane back to a stale point
  setTarget(m, fix, anchorMs, nowMs, m.vel != null);
  setOffsetFromDrawn(m, m.lon, m.lat, nowMs);
  const after = reckonAt(m, nowMs);
  m.offVE = dVE - (after.moving ? m.vE : 0);
  m.offVN = dVN - (after.moving ? m.vN : 0);
  if (m.headingDeg == null) m.headingDeg = fix.headingDeg;
  if (m.altM == null || fix.altM == null) m.altM = fix.altM;
}

function settle(m: PlaneMotion): void {
  m.offE = m.offN = m.offVE = m.offVN = 0;
}

/**
 * Advance the drawn position to nowMs and return it. Safe to call any number
 * of times per frame (a second call at the same instant integrates nothing).
 */
export function stepPlaneMotion(m: PlaneMotion, atMs: number): MotionOut {
  // callers mix the frame's timestamp with performance.now(): time never runs backward here
  const nowMs = atMs > m.lastMs ? atMs : m.lastMs;
  const elapsed = nowMs - m.lastMs;
  const r = reckonAt(m, nowMs);
  if (Number.isFinite(elapsed) && elapsed > 0) {
    m.lastMs = nowMs;
    if (elapsed > MOTION_MAX_STEP_MS) {
      settle(m);
      m.headingDeg = m.fix.headingDeg ?? m.headingDeg;
      m.altM = m.fix.altM;
    } else {
      if (m.offE !== 0 || m.offN !== 0 || m.offVE !== 0 || m.offVN !== 0) {
        const n = Math.max(1, Math.ceil(elapsed / FRAME.SPRING_STEP_MS));
        const h = elapsed / n / 1000;
        const speed = r.moving ? Math.hypot(m.vE, m.vN) : 0;
        const uE = speed > 0 ? m.vE / speed : 0;
        const uN = speed > 0 ? m.vN / speed : 0;
        const minAlong = -(1 - MOTION_MIN_FWD_FRAC) * speed;
        for (let i = 0; i < n; i++) {
          m.offVE += (-MOTION_OMEGA * MOTION_OMEGA * m.offE - 2 * MOTION_OMEGA * m.offVE) * h;
          m.offVN += (-MOTION_OMEGA * MOTION_OMEGA * m.offN - 2 * MOTION_OMEGA * m.offVN) * h;
          const v = Math.hypot(m.offVE, m.offVN);
          if (v > m.rateMps) { m.offVE *= m.rateMps / v; m.offVN *= m.rateMps / v; }
          if (speed > 0) {
            const along = m.offVE * uE + m.offVN * uN;
            if (along < minAlong) { m.offVE += (minAlong - along) * uE; m.offVN += (minAlong - along) * uN; }
          }
          m.offE += m.offVE * h;
          m.offN += m.offVN * h;
          if (Math.hypot(m.offE, m.offN) < OFF_EPS_M && Math.hypot(m.offVE, m.offVN) < OFF_EPS_M) { settle(m); break; }
        }
      }
      const th = m.fix.headingDeg;
      if (th != null && Number.isFinite(th)) {
        if (m.headingDeg == null) m.headingDeg = th;
        else {
          const d = ((th - m.headingDeg + 540) % 360) - 180; // shortest arc
          m.headingDeg = (th - lerpTowards(d, 0, MOTION_HEADING_HALF_LIFE_MS, elapsed) + 360) % 360;
        }
      }
      const ta = m.fix.altM;
      m.altM = ta == null || m.altM == null ? ta : lerpTowards(m.altM, ta, MOTION_ALT_HALF_LIFE_MS, elapsed);
    }
  }
  const cos = Math.max(0.05, Math.cos((r.lat * Math.PI) / 180));
  m.lon = r.lon + m.offE / (M_PER_DEG * cos);
  m.lat = r.lat + m.offN / M_PER_DEG;
  return {
    lon: m.lon, lat: m.lat, altM: m.altM, headingDeg: m.headingDeg,
    phase: motionPhase(r.ageSec, m.vel != null, m.limitSec),
    ageSec: r.ageSec, reckoned: m.vel != null,
    correcting: m.offE !== 0 || m.offN !== 0,
    limitSec: m.limitSec,
  };
}

/**
 * The flight card's sentence about the drawn position; null while it is live
 * (the card's "last position Ns ago" already says how old the real fix is).
 */
export function planeMotionNote(o: Pick<MotionOut, 'phase' | 'reckoned' | 'limitSec'>): string | null {
  if (o.phase === 'live') return null;
  if (o.phase === 'estimated') {
    return 'Drawn position is an estimate (shown faded): straight ahead from the last real position at its last reported speed and track. No turn is assumed.';
  }
  if (!o.reckoned) return 'Position held (shown faded): drawn at the last real position. Nothing newer has been received.';
  const limit = o.limitSec >= 60 ? `${Math.round(o.limitSec / 60)} min` : `${Math.round(o.limitSec)} s`;
  return `Position held (shown faded): an estimate, stopped ${limit} past the last real position because nothing newer has been received.`;
}

/**
 * How often the selected plane's tail, marker and tag are refreshed (ms; 0 =
 * every frame). Every frame only where a 300 ms step would be visible; the
 * terrain-saturation gate keeps its ~1 Hz pace (see datamap's glide tick).
 */
export function selectedPlaneCadenceMs(o: { stepVisible: boolean; correcting: boolean; overloaded: boolean; terrain: boolean }, stepMs: number): number {
  if (o.overloaded) return o.terrain ? stepMs * 3 : stepMs;
  return o.stepVisible || o.correcting ? 0 : stepMs;
}

/** A feed row as a MotionFix (on the ground the altitude is 0, as the track's fixes have it). */
export function motionFixFromRow(row: {
  lon: number; lat: number; altitude_m?: number | null; heading?: number | null;
  velocity_ms?: number | null; on_ground?: boolean | null;
}): MotionFix {
  return {
    lon: row.lon, lat: row.lat, altM: row.on_ground ? 0 : (row.altitude_m ?? null),
    headingDeg: row.heading ?? null, speedMs: row.velocity_ms ?? null, onGround: !!row.on_ground,
  };
}

/**
 * A feed row's own fix time in the FEED's clock (epoch seconds), or null when
 * the feed did not say: worldwide = the answer's `at` minus the row's age;
 * area = the snapshot's `time` minus the row's `seen_pos`. Used to order fixes
 * from different sources by when they were true, not by when they arrived.
 */
export function feedRowFixTimeSec(
  payload: { global?: unknown; time?: unknown; at?: unknown },
  row: { seen_pos?: number | null },
): number | null {
  const seen = row.seen_pos != null && Number.isFinite(row.seen_pos) ? Math.max(0, row.seen_pos) : null;
  if (payload.global === true) {
    return typeof payload.at === 'number' && Number.isFinite(payload.at) && seen != null ? payload.at / 1000 - seen : null;
  }
  return typeof payload.time === 'number' && Number.isFinite(payload.time) ? payload.time - (seen ?? 0) : null;
}

/**
 * Age (ms) of a feed row's position at `nowWallMs`, or null when the feed did
 * not say. Worldwide rows carry their age at the server's own clock, so only
 * the time since the answer arrived is added (no device-vs-server clock
 * comparison). Area rows carry `seen_pos` at the snapshot's `time` (epoch
 * seconds); the snapshot's own age is added when the two clocks plausibly
 * agree (0 to 10 min), else only the time since arrival.
 */
export function feedRowAgeMs(
  payload: { global?: unknown; time?: unknown; recvWallMs: number },
  row: { seen_pos?: number | null },
  nowWallMs: number,
): number | null {
  const sinceRecv = Math.max(0, nowWallMs - payload.recvWallMs);
  const seen = row.seen_pos != null && Number.isFinite(row.seen_pos) ? Math.max(0, row.seen_pos) * 1000 : null;
  if (payload.global === true) return seen == null ? null : seen + sinceRecv;
  const t = typeof payload.time === 'number' && Number.isFinite(payload.time) ? payload.time * 1000 : null;
  const snapAge = t == null ? null : payload.recvWallMs - t;
  if (snapAge != null && snapAge >= 0 && snapAge <= 600_000) return snapAge + (seen ?? 0) + sinceRecv;
  return seen == null ? null : seen + sinceRecv;
}
