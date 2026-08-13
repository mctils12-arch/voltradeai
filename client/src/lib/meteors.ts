// LARGE METEORS — pure client helpers for the CNEOS layer (severity ramp,
// compass wording, direction streak geometry, coverage links). No DOM, no
// network — `npx tsx --test`-able like every lib here.
//
// COVERAGE LINKS HONESTY (human-approved mock 2026-08-13): the card links
// are SEARCHES and report-browse pages built from the event's date +
// region, never a claim that a specific story or video matches — the card
// says so. AMS = the American Meteor Society's public sighting-report
// browser (plain deep link, no API); news/video are prefilled searches.

import tzlookup from "@photostructure/tz-lookup";

export interface MeteorSeverity {
  key: "minor" | "moderate" | "major";
  color: string;
  label: string;
}

/** blast-energy severity ramp (kt TNT) — the mock's three classes. */
export function meteorSeverity(impKt: number): MeteorSeverity {
  if (impKt >= 1) return { key: "major", color: "#ff5a6e", label: "≥ 1 kt (rare; Hiroshima ≈ 15)" };
  if (impKt >= 0.1) return { key: "moderate", color: "#fbb24c", label: "0.1 – 1 kt" };
  return { key: "minor", color: "#7cc4ff", label: "< 0.1 kt (common)" };
}

const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"] as const;

export function compassPoint(deg: number): string {
  return COMPASS[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16];
}

/** symbol size (icon-size units) from flash energy — log-scaled like the
 *  approved mock; bounded so a Chelyabinsk-class event can't dwarf the map. */
export function meteorIconSize(flashE: number): number {
  return Math.min(1.05, 0.34 + Math.log10(1 + Math.max(0, flashE)) * 0.22);
}

/** Direction streak: a short trail BEHIND the burst point (the meteor
 *  arrived from the reciprocal bearing), as [ [lon,lat] tail, [lon,lat]
 *  head ]. Length grows with entry speed; degrees are geodesically naive
 *  with a cos(lat) longitude correction — fine at streak scale. Returns
 *  null when heading is unpublished (never an invented direction). */
export function meteorStreak(
  la: number, lo: number, hdg: number | null, vel: number | null,
): [[number, number], [number, number]] | null {
  if (hdg == null) return null;
  const lenDeg = Math.min(6, 1.2 + (vel ?? 15) * 0.12);
  const back = ((hdg + 180) * Math.PI) / 180;
  const dLat = lenDeg * Math.cos(back);
  const cos = Math.max(0.2, Math.cos((la * Math.PI) / 180));
  const dLon = (lenDeg * Math.sin(back)) / cos;
  const tailLa = Math.max(-89, Math.min(89, la + dLat));
  let tailLo = lo + dLon;
  if (tailLo > 180) tailLo -= 360;
  if (tailLo < -180) tailLo += 360;
  return [[tailLo, tailLa], [lo, la]];
}

/** Region search keyword from the event position via the same tz database
 *  the tz-crossings feature ships: "Asia/Tokyo" → "Tokyo". Nautical Etc/*
 *  zones (open ocean) yield null — the search runs date-only then, and the
 *  card's expectations note explains ocean events rarely have coverage. */
export function meteorRegion(la: number, lo: number): string | null {
  try {
    const zone = tzlookup(la, lo);
    if (!zone || zone.startsWith("Etc/")) return null;
    const part = zone.split("/").pop() || "";
    return part.replace(/_/g, " ") || null;
  } catch { return null; }
}

export interface MeteorLink { label: string; href: string }

/** The card's coverage links — AMS report browser (±1 day window) + news
 *  and video searches from date + region. All plain links, all verified
 *  URL shapes (2026-08-13 probes; IMO left out — unreachable then). */
export function meteorCoverageLinks(dateUtc: string, la: number, lo: number): MeteorLink[] {
  const day = dateUtc.slice(0, 10);
  const before = new Date(Date.parse(day + "T00:00:00Z") - 86_400_000).toISOString().slice(0, 10);
  const after = new Date(Date.parse(day + "T00:00:00Z") + 86_400_000).toISOString().slice(0, 10);
  const region = meteorRegion(la, lo);
  const q = encodeURIComponent(["meteor fireball", region ?? "", day].filter(Boolean).join(" "));
  return [
    { label: "Eyewitness reports · AMS", href: `https://www.amsmeteors.org/fireballs/?start_date=${before}&end_date=${after}` },
    { label: "News search", href: `https://news.google.com/search?q=${q}` },
    { label: "Video search", href: `https://www.youtube.com/results?search_query=${q}` },
    { label: "NASA CNEOS fireball data", href: "https://cneos.jpl.nasa.gov/fireballs/" },
  ];
}

/** "23:08 at the site" — the event's local wall clock via the tz database;
 *  null over nautical zones / lookup failure (the card omits the phrase
 *  rather than faking a zone). */
export function siteLocalTime(tSec: number, la: number, lo: number): string | null {
  try {
    const zone = tzlookup(la, lo);
    if (!zone || zone.startsWith("Etc/")) return null;
    const s = new Intl.DateTimeFormat("en-US", {
      timeZone: zone, hour: "2-digit", minute: "2-digit", hour12: false,
      timeZoneName: "short",
    }).format(new Date(tSec * 1000));
    return s;
  } catch { return null; }
}
