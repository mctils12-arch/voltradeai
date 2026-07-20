// FLIGHT TRACK MODEL — the data core of the 3D flight-track visualization
// (design_handoff_flight_track_3d, human-approved prototype, installed
// 2026-07-20). Pure math, no GL, no React — unit-testable with `npx tsx
// --test` like every lib in this directory.
//
// One model feeds four consumers (the prototype's "State Management"
// contract: samples {t, lon, lat, altM, terrainM, gsKt, vsFpm} shared by
// the curtain, the altitude line, the profile chart and the flight-card
// readouts — terrain sampled ONCE per point, reused everywhere):
//
// - buildTrackSamples: archived fixes (1–5 min apart at cruise) densified
//   to ~TRACK_DENSIFY_M meter spacing by LINEAR interpolation between real
//   fixes. HONESTY: linear only — straight segments join real recorded
//   fixes, never smoothed into invented curves (the existing card promise;
//   the prototype's Catmull-Rom smoothing is for its synthetic demo track,
//   not for real ADS-B data). Fixes without a broadcast altitude become
//   honest GAPS (gap: true) — the curtain and 3D line break there.
// - altRampColor: the handoff's altitude ramp — low teal #38d1c1 → mid
//   blue #4da3ff → high violet #a06bff, linear in two halves, mapped
//   min→max across the visible track.
// - sampleAt: time-interpolated sample for playback/scrub (the profile
//   playhead and the 3D replay marker are the same position by
//   construction).
// - gs/vs derivation: the archive stores {t, lat, lon, alt} only — ground
//   speed and vertical speed are DERIVED from neighboring fixes exactly
//   like the prototype derives them from its samples (label: derived, not
//   broadcast; the live card prefers the feed's broadcast velocity when
//   present).

/** Archived track point (client/src/lib/air/breadcrumbs.ts TrackPoint):
 *  t epoch seconds, la/lo degrees, al meters MSL (null/undefined = the
 *  altitude wasn't broadcast — an honest gap, never invented). t is
 *  optional to match the wire type; fixes without a time are dropped. */
export interface RawTrackPoint {
  t?: number;
  la: number;
  lo: number;
  al?: number | null;
}

/** One densified sample — everything downstream consumers need. */
export interface TrackSample {
  /** epoch seconds */
  t: number;
  lon: number;
  lat: number;
  /** meters MSL (REAL, unexaggerated). NaN when inside an altitude gap. */
  altM: number;
  /** true = this sample sits in a no-altitude gap (curtain/line break). */
  gap: boolean;
  /** derived ground speed, knots (from neighboring fixes). */
  gsKt: number;
  /** derived vertical speed, feet per minute (from neighboring fixes). */
  vsFpm: number;
}

/** ~meter spacing the terrain under the track is re-sampled at (the
 *  handoff's "every ~120 m" — the drape rule that keeps ridges from
 *  opening gaps under the curtain). */
export const TRACK_DENSIFY_M = 120;

/** Hard cap on densified samples per track (a 500-fix archive track can
 *  span >1000 km; spacing degrades gracefully instead of unbounded
 *  geometry). 6000 samples ≈ 720 km at full 120 m fidelity. */
export const TRACK_MAX_SAMPLES = 6000;

const R_EARTH_M = 6371008.8;
const D2R = Math.PI / 180;

// ── current-flight trim (human 2026-07-20: the ALTITUDE/TIME chart "doesn't
// have that data" — the archive serves up to 48h, so a plane that flew,
// PARKED for hours (no broadcast altitude on the ground), then flew again
// rendered as two slivers around a giant honest gap. The display track —
// chart AND 3D trail — shows the CURRENT flight; the archive keeps
// everything.) ────────────────────────────────────────────────────────────

/** a fix-to-fix hole longer than this = out of coverage / a different
 *  flight — the newest contiguous block wins. */
export const FLIGHT_SPLIT_GAP_SEC = 45 * 60;
/** a sustained no-altitude dwell this long (parked at the gate — ground
 *  fixes rarely carry baro altitude) splits flights when airborne fixes
 *  exist after it. */
export const FLIGHT_GROUND_DWELL_SEC = 15 * 60;
/** an entirely ground-dwelling track (never airborne in the window) keeps
 *  only this trailing slice — 48h of parked dots is noise, not a flight. */
export const FLIGHT_PARKED_KEEP_SEC = 60 * 60;

/**
 * Trim raw archived fixes to the CURRENT flight. Pure; returns the input
 * array when nothing needs trimming. Boundaries, newest wins:
 *  1. a time hole > FLIGHT_SPLIT_GAP_SEC;
 *  2. a no-altitude dwell ≥ FLIGHT_GROUND_DWELL_SEC with airborne fixes
 *     after it (the dwell's last fix is KEPT — one ground point leads into
 *     the takeoff);
 *  3. never airborne at all → the trailing FLIGHT_PARKED_KEEP_SEC.
 */
export function trimToCurrentFlight(raw: RawTrackPoint[]): RawTrackPoint[] {
  const pts = raw.filter((p) => p != null && Number.isFinite(p.t));
  if (pts.length < 2) return raw;
  let start = 0;
  // newest hard time hole
  for (let i = pts.length - 1; i > 0; i--) {
    if ((pts[i].t as number) - (pts[i - 1].t as number) > FLIGHT_SPLIT_GAP_SEC) { start = i; break; }
  }
  // newest sustained ground dwell with flight after it
  let runEnd = -1; // newest index of the current no-altitude run
  for (let i = pts.length - 1; i >= start; i--) {
    if (pts[i].al == null) {
      if (runEnd < 0) runEnd = i;
    } else if (runEnd >= 0) {
      const span = (pts[runEnd].t as number) - (pts[i + 1].t as number);
      if (span >= FLIGHT_GROUND_DWELL_SEC && runEnd < pts.length - 1) {
        start = Math.max(start, runEnd); // keep the dwell's LAST fix
        break;
      }
      runEnd = -1;
    }
  }
  let out = start > 0 ? pts.slice(start) : pts;
  // never airborne → trailing slice only
  if (!out.some((p) => p.al != null)) {
    const tEnd = out[out.length - 1].t as number;
    const cut = out.findIndex((p) => (p.t as number) >= tEnd - FLIGHT_PARKED_KEEP_SEC);
    if (cut > 0) out = out.slice(cut);
  }
  return out.length === raw.length ? raw : out;
}

/** Great-circle meters between two fixes (haversine — the spacing ruler). */
export function distMeters(la1: number, lo1: number, la2: number, lo2: number): number {
  const phi1 = la1 * D2R, phi2 = la2 * D2R;
  const dphi = (la2 - la1) * D2R, dlam = (lo2 - lo1) * D2R;
  const a = Math.sin(dphi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dlam / 2) ** 2;
  return 2 * R_EARTH_M * Math.asin(Math.min(1, Math.sqrt(a)));
}

const MS_TO_KT = 1.94384;
const M_TO_FT = 3.28084;

/**
 * Densify raw archived fixes to ~TRACK_DENSIFY_M spacing (linear lat/lon/alt
 * interpolation between consecutive REAL fixes; per-fix sample count comes
 * from great-circle distance) and derive gs (kt) / vs (fpm) per sample from
 * a ±2-fix window over the RAW fixes (the prototype's derivation, applied
 * to real data). Consecutive-fix pairs where either end lacks altitude
 * produce gap samples: position/time still present (the profile chart's
 * terrain floor continues) but altM = NaN and gap = true.
 *
 * Returns samples plus the altitude range over non-gap samples (the ramp's
 * min→max mapping domain).
 */
export function buildTrackSamples(raw: RawTrackPoint[]): {
  samples: TrackSample[];
  altMin: number;
  altMax: number;
} {
  const fixes = raw
    .filter((p) => p != null && Number.isFinite(p.la) && Number.isFinite(p.lo) && Number.isFinite(p.t))
    .map((p) => ({ t: p.t as number, la: p.la, lo: p.lo, al: p.al ?? null }));
  if (fixes.length === 0) return { samples: [], altMin: 0, altMax: 0 };

  // derived rates at each RAW fix (±2 window like the prototype)
  const gs: number[] = new Array(fixes.length).fill(0);
  const vs: number[] = new Array(fixes.length).fill(0);
  for (let i = 0; i < fixes.length; i++) {
    const a = fixes[Math.max(0, i - 2)];
    const b = fixes[Math.min(fixes.length - 1, i + 2)];
    const dt = b.t - a.t;
    if (dt > 0) {
      gs[i] = (distMeters(a.la, a.lo, b.la, b.lo) / dt) * MS_TO_KT;
      if (a.al != null && b.al != null) vs[i] = ((b.al - a.al) / dt) * M_TO_FT * 60;
    }
  }

  // total distance → adaptive spacing under the sample cap
  let total = 0;
  for (let i = 1; i < fixes.length; i++) {
    total += distMeters(fixes[i - 1].la, fixes[i - 1].lo, fixes[i].la, fixes[i].lo);
  }
  const spacing = Math.max(TRACK_DENSIFY_M, total / Math.max(1, TRACK_MAX_SAMPLES - fixes.length));

  const out: TrackSample[] = [];
  const push = (t: number, lon: number, lat: number, altM: number | null, g: number, v: number) => {
    out.push({
      t, lon, lat,
      altM: altM == null ? NaN : altM,
      gap: altM == null,
      gsKt: g, vsFpm: v,
    });
  };

  for (let i = 0; i < fixes.length; i++) {
    const f = fixes[i];
    push(f.t, f.lo, f.la, f.al ?? null, gs[i], vs[i]);
    if (i + 1 >= fixes.length) break;
    const n = fixes[i + 1];
    const segM = distMeters(f.la, f.lo, n.la, n.lo);
    const steps = Math.min(512, Math.floor(segM / spacing));
    const segGap = f.al == null || n.al == null; // either end silent → gap
    for (let s = 1; s <= steps; s++) {
      const u = s / (steps + 1);
      push(
        f.t + (n.t - f.t) * u,
        f.lo + (n.lo - f.lo) * u,
        f.la + (n.la - f.la) * u,
        segGap ? null : (f.al as number) + ((n.al as number) - (f.al as number)) * u,
        gs[i] + (gs[i + 1] - gs[i]) * u,
        segGap ? 0 : vs[i] + (vs[i + 1] - vs[i]) * u,
      );
    }
  }

  let altMin = Infinity, altMax = -Infinity;
  for (const s of out) {
    if (s.gap) continue;
    if (s.altM < altMin) altMin = s.altM;
    if (s.altM > altMax) altMax = s.altM;
  }
  if (!Number.isFinite(altMin)) { altMin = 0; altMax = 0; }
  return { samples: out, altMin, altMax };
}

// ── altitude color ramp (the handoff's exact stops) ─────────────────────────
/** low teal #38d1c1 */
export const RAMP_LO: [number, number, number] = [0x38 / 255, 0xd1 / 255, 0xc1 / 255];
/** mid blue #4da3ff */
export const RAMP_MID: [number, number, number] = [0x4d / 255, 0xa3 / 255, 0xff / 255];
/** high violet #a06bff */
export const RAMP_HI: [number, number, number] = [0xa0 / 255, 0x6b / 255, 0xff / 255];

/** Curtain fill opacity (handoff: "same color at 34% opacity"). */
export const CURTAIN_ALPHA = 0.34;

/** Curtain bottom-vertex color multiplier (handoff: top × [0.25, 0.28, 0.45]
 *  — darkens toward the ground so the wall reads as a shaded curtain). */
export const CURTAIN_BOTTOM_MUL: [number, number, number] = [0.25, 0.28, 0.45];

/** Ground trace color #1e5fd6 (the draped topo-following line). */
export const TRACE_RGBA: [number, number, number, number] =
  [0x1e / 255, 0x5f / 255, 0xd6 / 255, 1];

/** meters the curtain bottom sits BELOW the terrain surface (drape rule 1:
 *  overlap into the ground so ridges never open gaps). */
export const CURTAIN_BELOW_TERRAIN_M = 40;

/** meters the ground trace floats above the terrain surface. */
export const TRACE_ABOVE_TERRAIN_M = 16;

/** altitude line ribbon width, px (handoff: "~4 px"). */
export const ALT_LINE_WIDTH_PX = 4;

/** ground trace ribbon width, px (handoff: "width ~3 px"). */
export const TRACE_WIDTH_PX = 3;

/**
 * Altitude (m MSL) → [r,g,b] on the handoff ramp, min→max normalized across
 * the visible track, linear in two halves (lo→mid then mid→hi).
 */
export function altRampColor(altM: number, altMin: number, altMax: number): [number, number, number] {
  const span = altMax - altMin;
  const t = span > 0 ? Math.min(1, Math.max(0, (altM - altMin) / span)) : 0.5;
  const lerp = (a: [number, number, number], b: [number, number, number], u: number):
      [number, number, number] =>
    [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
  return t < 0.5 ? lerp(RAMP_LO, RAMP_MID, t * 2) : lerp(RAMP_MID, RAMP_HI, (t - 0.5) * 2);
}

/**
 * Time-interpolated sample for playback/scrub. Clamps to the track's time
 * range. Interpolating INTO or OUT OF a gap keeps position but reports
 * gap=true (readouts show "—", the marker keeps moving — position is real,
 * altitude honestly unknown).
 */
export function sampleAt(samples: TrackSample[], t: number): TrackSample | null {
  const n = samples.length;
  if (n === 0) return null;
  if (t <= samples[0].t) return samples[0];
  if (t >= samples[n - 1].t) return samples[n - 1];
  // binary search for the bracketing pair
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t <= t) lo = mid; else hi = mid;
  }
  const a = samples[lo], b = samples[hi];
  const dt = b.t - a.t;
  const u = dt > 0 ? (t - a.t) / dt : 0;
  const gap = a.gap || b.gap;
  return {
    t,
    lon: a.lon + (b.lon - a.lon) * u,
    lat: a.lat + (b.lat - a.lat) * u,
    altM: gap ? NaN : a.altM + (b.altM - a.altM) * u,
    gap,
    gsKt: a.gsKt + (b.gsKt - a.gsKt) * u,
    vsFpm: gap ? 0 : a.vsFpm + (b.vsFpm - a.vsFpm) * u,
  };
}

/** Track heading (deg, 0=N clockwise) at time t — the tangent direction the
 *  replay marker's nose points along (handoff §4: "nose along track
 *  tangent"). Looks a few seconds ahead like the prototype. */
export function headingAt(samples: TrackSample[], t: number, aheadSec = 4): number | null {
  const a = sampleAt(samples, t);
  const b = sampleAt(samples, t + aheadSec);
  if (!a || !b) return null;
  const dLon = (b.lon - a.lon) * Math.cos(((a.lat + b.lat) / 2) * D2R);
  const dLat = b.lat - a.lat;
  if (dLon === 0 && dLat === 0) return null;
  return (Math.atan2(dLon, dLat) / D2R + 360) % 360;
}
