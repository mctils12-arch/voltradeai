// Satellite find & group — EARTH TWIN O6-3 (human directive: "you can't
// find the international space station with all 12000 sats … a feature
// where you can see all Starlink sats and a toggle for their orbits").
//
// Pure helpers. GROUPS are NAME-PREFIX decodes of the catalog's own naming
// (CelesTrak GP names) — a broadcastable, checkable fact, never an inferred
// operator. Groups whose constellations live in deep space (GPS/GLONASS…)
// are deliberately absent: the near-earth SGP4 kernel gives them no live
// position, so a "group" of them would show an empty sky and read as a bug.

import type { GpRecord } from './tle.js';
import { SAT_STRIDE } from './satBuffer.js';

export interface SatSearchHit {
  index: number;
  noradId: number;
  name: string;
}

/** Case-insensitive name-substring (or exact NORAD id) search. */
export function searchSats(gp: GpRecord[] | null, query: string, limit = 8): SatSearchHit[] {
  if (!gp || !gp.length) return [];
  const q = query.trim().toUpperCase();
  if (q.length < 2) return [];
  const asId = /^\d{3,}$/.test(q) ? Number(q) : null;
  const hits: SatSearchHit[] = [];
  for (let i = 0; i < gp.length && hits.length < limit; i++) {
    const g = gp[i];
    const name = (g.name || '').toUpperCase();
    if ((asId != null && g.noradId === asId) || (q.length >= 2 && name.includes(q))) {
      hits.push({ index: i, noradId: g.noradId, name: g.name || `NORAD ${g.noradId}` });
    }
  }
  return hits;
}

export interface SatGroup {
  key: string;
  label: string;
  /** UPPERCASE name test against the catalog name. */
  test: (nameUpper: string) => boolean;
}

export const SAT_GROUPS: SatGroup[] = [
  { key: 'iss', label: 'ISS', test: (n) => n.startsWith('ISS (') },
  { key: 'starlink', label: 'Starlink', test: (n) => n.startsWith('STARLINK') },
  { key: 'oneweb', label: 'OneWeb', test: (n) => n.startsWith('ONEWEB') },
  { key: 'iridium', label: 'Iridium', test: (n) => n.startsWith('IRIDIUM') },
  { key: 'globalstar', label: 'Globalstar', test: (n) => n.startsWith('GLOBALSTAR') },
  { key: 'planet', label: 'Planet (Flock/SkySat)', test: (n) => n.startsWith('FLOCK') || n.startsWith('SKYSAT') },
  { key: 'spire', label: 'Spire (Lemur)', test: (n) => n.startsWith('LEMUR') },
];

/** 1 = member, index-aligned with gp (and therefore the worker buffer). */
export function groupMask(gp: GpRecord[], key: string): Uint8Array | null {
  const grp = SAT_GROUPS.find((g) => g.key === key);
  if (!grp) return null;
  const mask = new Uint8Array(gp.length);
  for (let i = 0; i < gp.length; i++) {
    if (grp.test((gp[i].name || '').toUpperCase())) mask[i] = 1;
  }
  return mask;
}

export function maskCount(mask: Uint8Array | null): number {
  if (!mask) return 0;
  let n = 0;
  for (let i = 0; i < mask.length; i++) n += mask[i];
  return n;
}

/**
 * Filter the live position buffer to a group by writing the SENTINEL class
 * code (-1) into non-members — the exact semantics the layer/picker already
 * honor for deep-space/invalid slots. Returns a COPY; the worker's buffer
 * is never mutated (the next tick replaces it anyway).
 */
export function applyGroupSentinel(buf: Float32Array, mask: Uint8Array | null): Float32Array {
  if (!mask) return buf;
  const out = buf.slice();
  const n = Math.min(Math.floor(out.length / SAT_STRIDE), mask.length);
  for (let i = 0; i < n; i++) {
    if (!mask[i]) out[i * SAT_STRIDE + 3] = -1;
  }
  return out;
}

/** Max simultaneous group orbit arcs — beyond this the sky is unreadable
 *  and the one-shot SGP4 sampling cost climbs; the UI must SAY when the
 *  cap bites (sampled evenly across the group, never silently). */
export const GROUP_ARC_CAP = 40;

/** Evenly-spread member indices under the cap (deterministic). */
export function spreadIndices(members: number[], cap: number = GROUP_ARC_CAP): number[] {
  if (members.length <= cap) return members;
  const step = members.length / cap;
  const out: number[] = [];
  for (let k = 0; k < cap; k++) out.push(members[Math.floor(k * step)]);
  return out;
}
