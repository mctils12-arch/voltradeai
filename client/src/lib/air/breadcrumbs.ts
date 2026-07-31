// SESSION BREADCRUMBS — trail/curtain continuity for the FOLLOWED plane
// (2026-07-18 human report: "the data is cut off ... sharp angles where the
// plane would make a curve"). The archive samples cruise aircraft every
// 1-5 minutes (datacoreArchive adaptive thinning), so the archived trail
// ends minutes BEHIND the live plane and its segments are long and
// straight. While a plane's card is open, every fresh live poll (~15s)
// appends a breadcrumb here; the merged track densifies the RECENT segment
// and carries the 3D trail + altitude curtain all the way to the plane's
// current position. HONESTY: crumbs are REAL broadcast positions from the
// live feed — never interpolated, never smoothed; the archived section
// keeps its straight segments between real recorded points (a straight
// line between real fixes is honest; an invented curve is not). Crumbs are
// session-only display state — the archive stays the single recorded truth.
// One deliberate display exception, mirroring the glide tail's documented
// "altitude held at the last broadcast value" policy: a fix that omits
// ALTITUDE (position still real) merges at the last real broadcast
// altitude for a bounded window (ALT_HOLD_MAX_AGE_SEC), flagged `held` —
// the stored crumb keeps al:null, so the buffer never lies about what was
// broadcast.
//
// Pure module: no DOM, no map, hermetic to unit-test.

/** One live-feed fix for the followed entity. `al` null = altitude not
 *  broadcast at this fix (renders as an honest gap in the 3D line). */
export interface Crumb {
  lo: number;
  la: number;
  al: number | null;
  /** epoch seconds (the payload's snapshot time). */
  t: number;
}

/** Max crumbs kept (240 ≈ 1h of 15s polls) — a card left open all day
 *  stays bounded; the archive owns long history. */
export const CRUMB_CAP = 240;

/** Min seconds between kept crumbs — the server cache means most 15s polls
 *  repeat the same upstream snapshot; identical/near-duplicate fixes add
 *  vertices without information. */
export const CRUMB_MIN_DT_SEC = 5;

/** Crumbs newer than the newest archived point by LESS than this are
 *  dropped on merge — the archive may have recorded the very same fix; a
 *  1s guard avoids double vertices at the seam. */
export const CRUMB_MERGE_GUARD_SEC = 1;

/** ALTITUDE HOLD bound (2026-07-31 video: the curtain stopped at the last
 *  archived fix while the plane flew on — its live fixes carried position
 *  and speed but no altitude, and one null NaN-gapped the whole live
 *  bridge). When a live fix omits altitude, the merge carries the last
 *  REAL broadcast altitude forward — the same held-level policy the glide
 *  tail has always documented ("altitude held at the last broadcast
 *  value") — but only within this bound, measured from that real fix.
 *  Beyond it the honest gap returns: minutes without any altitude is a
 *  data outage, not a level-flight assumption. */
export const ALT_HOLD_MAX_AGE_SEC = 240;

/**
 * Append a live fix to the session buffer. Returns the (possibly new)
 * buffer; the input array is never mutated. Rejects junk (non-finite
 * position/time), out-of-order or too-frequent fixes, and enforces the cap
 * by dropping the oldest.
 */
export function pushCrumb(
  crumbs: readonly Crumb[],
  c: Crumb,
  cap: number = CRUMB_CAP,
  minDtSec: number = CRUMB_MIN_DT_SEC,
): Crumb[] {
  if (!Number.isFinite(c.lo) || !Number.isFinite(c.la) || !Number.isFinite(c.t)) return crumbs as Crumb[];
  const last = crumbs.length ? crumbs[crumbs.length - 1] : null;
  if (last && c.t - last.t < minDtSec) return crumbs as Crumb[];
  const next = crumbs.concat(c);
  return next.length > cap ? next.slice(next.length - cap) : next;
}

/** Track point shape served by /api/data/track (al optional). `held` marks
 *  a display point whose altitude is the ALTITUDE HOLD carrying the last
 *  real broadcast forward (never in archived data — merge-time only). */
export interface TrackPoint {
  lo: number;
  la: number;
  t?: number;
  al?: number | null;
  held?: boolean;
}

/**
 * Archived track + session crumbs → one display track. Only crumbs
 * STRICTLY newer than the newest archived point (plus the seam guard) are
 * appended — the archived history is never reordered, thinned, or smoothed.
 * With no archive yet (id never archived / archive warming), the crumbs
 * alone form the track: the live segment is real data in its own right.
 *
 * ALTITUDE HOLD: a crumb whose fix omitted altitude (al null — the buffer
 * always stores what was actually broadcast) is displayed at the last real
 * broadcast altitude, flagged `held`, for at most `holdMaxAgeSec` after
 * that real fix. One missing field no longer NaN-gaps the curtain's live
 * bridge; a sustained outage still renders as an honest gap.
 */
export function mergeTrackWithCrumbs(
  archived: readonly TrackPoint[],
  crumbs: readonly Crumb[],
  holdMaxAgeSec: number = ALT_HOLD_MAX_AGE_SEC,
): TrackPoint[] {
  let lastT = -Infinity;
  for (const p of archived) if (p.t != null && p.t > lastT) lastT = p.t;
  const cut = lastT + CRUMB_MERGE_GUARD_SEC;
  // hold source: the newest archived point with a real altitude AND a time
  // (an untimed altitude can't be age-bounded, so it never seeds the hold)
  let holdAl: number | null = null;
  let holdT = -Infinity;
  for (const p of archived) {
    if (p.al != null && p.t != null && p.t >= holdT) { holdAl = p.al; holdT = p.t; }
  }
  const fresh: TrackPoint[] = [];
  for (const c of crumbs) {
    if (c.t <= cut) continue;
    if (c.al != null) {
      holdAl = c.al; holdT = c.t; // a real broadcast resets the hold source
      fresh.push({ lo: c.lo, la: c.la, t: c.t, al: c.al });
    } else if (holdAl != null && c.t - holdT <= holdMaxAgeSec) {
      fresh.push({ lo: c.lo, la: c.la, t: c.t, al: holdAl, held: true });
    } else {
      fresh.push({ lo: c.lo, la: c.la, t: c.t, al: null });
    }
  }
  return fresh.length ? archived.concat(fresh) : (archived as TrackPoint[]);
}
