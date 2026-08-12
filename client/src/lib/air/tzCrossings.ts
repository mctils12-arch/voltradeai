// TIME-ZONE CROSSINGS along a flight track (human directive 2026-08-11:
// "if it passed a time zone display it on the slider and the curtain as a
// different color line … it come on automatically if your click on plane
// and it passes a time zone or has in the log data"). The map-lines half
// shipped first (tz-borders/tz-dateline layers); this module is the
// per-flight half: WHERE along the track the local clock actually changed.
//
// HONESTY: a "crossing" is a change in the LOCAL UTC OFFSET between two
// consecutive recorded fixes — the thing a passenger's watch experiences —
// not merely a boundary line on the map (two zones with the same offset,
// e.g. adjacent regions both at UTC−5 in winter, produce no crossing;
// that matches what the clock does). Zone identity comes from
// @photostructure/tz-lookup (CC0, timezone-boundary-builder data); offsets
// come from the browser's own IANA database via Intl, evaluated AT THE
// FIX'S OWN TIME so DST is correct for the day of the flight.
//
// Boundary jitter: a plane flying along a border can flap between zones
// sample-to-sample. A crossing only commits when the NEW offset holds for
// CONFIRM_SAMPLES consecutive samples (or the track ends inside it).

import tzlookup from "@photostructure/tz-lookup";
import type { TrackSample } from "./trackModel";

export interface TzCrossing {
  /** epoch seconds of the first sample inside the new zone. */
  t: number;
  /** index of that sample in the input array. */
  idx: number;
  fromZone: string;
  toZone: string;
  /** minutes east of UTC at the crossing time (e.g. EDT = −240). */
  fromOffsetMin: number;
  toOffsetMin: number;
  /** viewer-facing label, e.g. "EDT → CDT" or "UTC+5:30 → UTC+6". */
  label: string;
}

/** consecutive samples the new offset must hold before a crossing commits */
export const CONFIRM_SAMPLES = 2;

// ── zone/offset lookups (cached — Intl.DateTimeFormat construction is slow
// and a densified track re-queries the same few zones thousands of times) ──

const fmtCache = new Map<string, Intl.DateTimeFormat | null>();
function fmtFor(zone: string): Intl.DateTimeFormat | null {
  let f = fmtCache.get(zone);
  if (f === undefined) {
    try {
      f = new Intl.DateTimeFormat("en-US", {
        timeZone: zone,
        timeZoneName: "shortOffset",
        year: "numeric", month: "numeric", day: "numeric",
        hour: "numeric", minute: "numeric",
      });
    } catch { f = null; }
    fmtCache.set(zone, f);
  }
  return f;
}

const abbrCache = new Map<string, Intl.DateTimeFormat | null>();
function abbrFmtFor(zone: string): Intl.DateTimeFormat | null {
  let f = abbrCache.get(zone);
  if (f === undefined) {
    try {
      f = new Intl.DateTimeFormat("en-US", { timeZone: zone, timeZoneName: "short" });
    } catch { f = null; }
    abbrCache.set(zone, f);
  }
  return f;
}

/** Parse "GMT-4" / "GMT+5:30" / "GMT" → minutes east of UTC, or null. */
export function parseGmtOffset(name: string): number | null {
  const m = /^(?:GMT|UTC)(?:([+-])(\d{1,2})(?::(\d{2}))?)?$/.exec(name.trim());
  if (!m) return null;
  if (!m[1]) return 0;
  const sign = m[1] === "-" ? -1 : 1;
  return sign * (parseInt(m[2], 10) * 60 + (m[3] ? parseInt(m[3], 10) : 0));
}

// offset varies with DST, so the cache key buckets by zone + UTC hour
const offCache = new Map<string, number | null>();
export function offsetMinAt(zone: string, tSec: number): number | null {
  const key = `${zone}|${Math.floor(tSec / 3600)}`;
  let v = offCache.get(key);
  if (v === undefined) {
    const f = fmtFor(zone);
    const name = f?.formatToParts(new Date(tSec * 1000))
      .find((p) => p.type === "timeZoneName")?.value;
    v = name != null ? parseGmtOffset(name) : null;
    offCache.set(key, v);
  }
  return v;
}

/** Short zone name at a moment ("EDT", "CET"); many zones only have a
 *  GMT-style name — the caller falls back to the offset form then. */
export function zoneAbbrAt(zone: string, tSec: number): string {
  try {
    return abbrFmtFor(zone)?.formatToParts(new Date(tSec * 1000))
      .find((p) => p.type === "timeZoneName")?.value || "";
  } catch { return ""; }
}

function offsetLabel(min: number): string {
  const sign = min < 0 ? "−" : "+";
  const a = Math.abs(min);
  const h = Math.floor(a / 60);
  const mm = a % 60;
  return `UTC${min === 0 ? "" : sign + h + (mm ? ":" + String(mm).padStart(2, "0") : "")}`;
}

/** One side of the crossing label: real abbreviation if Intl has one,
 *  otherwise the UTC-offset form (never a raw IANA path — "America/…"
 *  is a database key, not a clock). */
export function sideLabel(zone: string, tSec: number, offMin: number): string {
  const ab = zoneAbbrAt(zone, tSec);
  return ab && !/^(GMT|UTC)/.test(ab) ? ab : offsetLabel(offMin);
}

/** IANA zone of a fix, or null over international waters where the lookup
 *  falls back to an Etc/GMT± nautical zone (still a real offset — kept). */
export function zoneOf(lat: number, lon: number): string | null {
  try { return tzlookup(lat, lon); } catch { return null; }
}

/**
 * Pure: track samples → committed offset crossings, in time order.
 * Jitter-guarded: the new offset must hold CONFIRM_SAMPLES consecutive
 * samples (or reach the end of the track) before it commits — a border-
 * hugging flight doesn't spray marks.
 */
export function computeTzCrossings(samples: TrackSample[]): TzCrossing[] {
  const out: TzCrossing[] = [];
  if (samples.length < 2) return out;
  let curZone: string | null = null;
  let curOff: number | null = null;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const z = zoneOf(s.lat, s.lon);
    if (!z) continue;
    const off = offsetMinAt(z, s.t);
    if (off == null) continue;
    if (curOff == null) { curZone = z; curOff = off; continue; }
    if (off === curOff) { curZone = z; continue; } // zone rename, same clock
    // candidate crossing at i — confirm it holds
    let holds = 0;
    for (let j = i; j < samples.length && holds < CONFIRM_SAMPLES; j++) {
      const zj = zoneOf(samples[j].lat, samples[j].lon);
      const oj = zj ? offsetMinAt(zj, samples[j].t) : null;
      if (oj !== off) break;
      holds++;
    }
    const confirmed = holds >= CONFIRM_SAMPLES
      || (holds >= 1 && i + holds >= samples.length); // track ends inside it
    if (confirmed) {
      out.push({
        t: s.t, idx: i,
        fromZone: curZone!, toZone: z,
        fromOffsetMin: curOff, toOffsetMin: off,
        label: `${sideLabel(curZone!, s.t, curOff)} → ${sideLabel(z, s.t, off)}`,
      });
      curZone = z; curOff = off;
    }
    // unconfirmed flap: keep the current zone, keep scanning
  }
  return out;
}
