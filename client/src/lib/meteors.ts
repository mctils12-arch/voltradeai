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

const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"] as const;

/** "2016-08-05 …" → "August 5 2016" — search engines and video titles use
 *  human dates; the machine form is exactly what returned nothing when the
 *  human tried it (2026-08-13 report, Google News zero results). */
export function naturalDate(dateUtc: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(dateUtc);
  if (!m) return dateUtc.slice(0, 10);
  return `${MONTHS[Number(m[2]) - 1]} ${Number(m[3])} ${m[1]}`;
}

/** "August 2016" — the loose form video titles actually use. */
export function naturalMonthYear(dateUtc: string): string {
  const m = /^(\d{4})-(\d{2})/.exec(dateUtc);
  if (!m) return dateUtc.slice(0, 7);
  return `${MONTHS[Number(m[2]) - 1]} ${m[1]}`;
}

/** True when the sun was below the horizon at the site — the plain solar
 *  hour-angle test (local solar time from longitude, night = before 6h or
 *  after 18h solar). An honest approximation for "was the sky dark":
 *  ignores twilight and season, which is fine for a likelihood verdict. */
export function nightAtSite(tSec: number, lo: number): boolean {
  const utcHours = (tSec % 86_400) / 3600;
  const solar = (((utcHours + lo / 15) % 24) + 24) % 24;
  return solar < 6 || solar >= 18;
}

export interface MeteorVerdict {
  key: "likely" | "possible" | "unlikely";
  label: string;
}

/** Per-event coverage likelihood, computed from what we actually know —
 *  the design answer to "the links bring back nothing" (2026-08-13): a
 *  blast over the open ocean has no witnesses, so the card SAYS so instead
 *  of offering dead searches; Chelyabinsk had videos because it happened
 *  over a city at rush hour. */
export function meteorCoverageVerdict(tSec: number, la: number, lo: number): MeteorVerdict {
  const region = meteorRegion(la, lo);
  if (!region) return { key: "unlikely", label: "🌊 Remote ocean, no witnesses — footage unlikely" };
  if (nightAtSite(tSec, lo)) {
    return { key: "likely", label: `✦ Night sky near ${region} — sightings likely` };
  }
  return { key: "possible", label: `Daytime near ${region} — sightings possible (daylight fireballs need to be BIG)` };
}

/** The card's coverage links, verdict-aware and age-aware:
 *  - ocean (no region): NASA's record + ONE de-emphasized "search anyway"
 *    web link — never a wall of dead searches;
 *  - land: AMS eyewitness browser (±1 day), a video search in the loose
 *    month-year wording titles use, and a text search — Google NEWS only
 *    while the news index still has the event (~1 year), the general WEB
 *    after that (news indexes decay; archives don't). NASA's record last.
 *  All plain searches/browse pages, never claimed stories. */
export function meteorCoverageLinks(
  dateUtc: string, la: number, lo: number, nowMs?: number,
): MeteorLink[] {
  const day = dateUtc.slice(0, 10);
  const dayMs = Date.parse(day + "T00:00:00Z");
  const region = meteorRegion(la, lo);
  const nd = naturalDate(dateUtc);
  const cneos = { label: "NASA CNEOS record", href: "https://cneos.jpl.nasa.gov/fireballs/" };
  if (!region) {
    const q = encodeURIComponent(`meteor fireball ${nd}`);
    return [
      cneos,
      { label: "Search the web anyway", href: `https://www.google.com/search?q=${q}` },
    ];
  }
  const before = new Date(dayMs - 86_400_000).toISOString().slice(0, 10);
  const after = new Date(dayMs + 86_400_000).toISOString().slice(0, 10);
  const recent = (nowMs ?? Date.now()) - dayMs < 365 * 86_400_000;
  const qFull = encodeURIComponent(`meteor fireball ${region} ${nd}`);
  const qLoose = encodeURIComponent(`meteor ${region} ${naturalMonthYear(dateUtc)}`);
  return [
    { label: "Eyewitness reports · AMS", href: `https://www.amsmeteors.org/fireballs/?start_date=${before}&end_date=${after}` },
    { label: `Videos: "meteor ${region} ${naturalMonthYear(dateUtc)}"`, href: `https://www.youtube.com/results?search_query=${qLoose}` },
    recent
      ? { label: "News search", href: `https://news.google.com/search?q=${qFull}` }
      : { label: "Web search (news indexes decay — this event is old)", href: `https://www.google.com/search?q=${qFull}` },
    cneos,
  ];
}

// ── compact stat formatting (2026-08-13: "106299…" truncated in the chip —
// values must FIT; unit system respected, storage stays native) ────────────

export function fmtBlastAlt(altKm: number | null, system: "imperial" | "metric"): string {
  if (altKm == null) return "—";
  if (system === "imperial") return `${Math.round(altKm * 3280.84 / 1000)}k ft`;
  return `${Math.round(altKm)} km`;
}

export function fmtEntrySpeed(velKms: number | null, system: "imperial" | "metric"): string {
  if (velKms == null) return "—";
  if (system === "imperial") return `${Math.round(velKms * 2236.94 / 1000)}k mph`;
  return `${Math.round(velKms)} km/s`;
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
