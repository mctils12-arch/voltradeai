// Satellite find & group — EARTH TWIN O6-3 (human directive: "you can't
// find the international space station with all 12000 sats … a feature
// where you can see all Starlink sats and a toggle for their orbits").
//
// Pure helpers. GROUPS are NAME-PREFIX decodes of the catalog's own naming
// (CelesTrak GP names) — a broadcastable, checkable fact, never an inferred
// operator. Deep-space constellations (GPS/Galileo/BeiDou) joined the list
// with the SDP4 port (O6-5) — they have real live positions now. GLONASS
// stays absent HONESTLY: its catalog names are bare "COSMOS NNNN", a prefix
// shared with many non-GLONASS objects — a name decode would lie.

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
  // deep-space constellations — live since the SDP4 port (O6-5)
  { key: 'gps', label: 'GPS', test: (n) => n.startsWith('NAVSTAR') || n.startsWith('GPS ') },
  { key: 'galileo', label: 'Galileo', test: (n) => n.includes('GALILEO') },
  { key: 'beidou', label: 'BeiDou', test: (n) => n.startsWith('BEIDOU') },
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

/** Max simultaneous group orbit arcs — beyond this the one-shot SGP4
 *  sampling cost climbs and the sky saturates; the UI must SAY when the
 *  cap bites (sampled evenly ACROSS ORBITAL PLANES, never silently).
 *  Raised 40→150 after live feedback ("a fraction of the orbits"):
 *  callers sort members by RAAN first, so the sample covers every plane. */
export const GROUP_ARC_CAP = 150;

/** Evenly-spread member indices under the cap (deterministic). */
export function spreadIndices(members: number[], cap: number = GROUP_ARC_CAP): number[] {
  if (members.length <= cap) return members;
  const step = members.length / cap;
  const out: number[] = [];
  for (let k = 0; k < cap; k++) out.push(members[Math.floor(k * step)]);
  return out;
}

/**
 * STATION COMPLEX COLLAPSE (human-directed 2026-07-16: "I just want the
 * one full [station] that is up there currently and get rid of the rest"):
 * the ISS and CSS are each ONE physical station, but their modules are
 * cataloged as separate objects (ISS (ZARYA)/(UNITY)/(NAUKA)…,
 * CSS (TIANHE)/(WENTIAN)/(MENGTIAN)) — rendering several co-orbiting
 * "satellites" for one station is LESS truthful, not more. Entries whose
 * name matches a station prefix collapse to ONE keeper: the lowest NORAD
 * id present, i.e. the station's first-launched core module (ISS → ZARYA
 * 25544). Genuinely separate visiting vehicles (Dragon, Soyuz, Progress,
 * Cygnus) have their own catalog names and are untouched. The detail card
 * states the collapse; the keeper carries the real station model.
 */
const STATION_PREFIXES = [/^ISS \(/i, /^CSS \(/i];

export function collapseStationComplexes(gp: GpRecord[]): GpRecord[] {
  const keeper = new Map<number, number>();
  for (const g of gp) {
    const i = STATION_PREFIXES.findIndex((re) => re.test((g.name ?? '').trim()));
    if (i < 0) continue;
    const cur = keeper.get(i);
    if (cur === undefined || g.noradId < cur) keeper.set(i, g.noradId);
  }
  if (!keeper.size) return gp;
  return gp.filter((g) => {
    const i = STATION_PREFIXES.findIndex((re) => re.test((g.name ?? '').trim()));
    return i < 0 || keeper.get(i) === g.noradId;
  });
}

/** True when this entry is a station-complex keeper (for the card note). */
export function isStationComplex(name: string | null | undefined): boolean {
  return STATION_PREFIXES.some((re) => re.test((name ?? '').trim()));
}
