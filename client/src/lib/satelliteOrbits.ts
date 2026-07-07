/**
 * satelliteOrbits.ts — pure helpers for the /data map's SATELLITE ORBITS
 * layer (ANALYST CONSOLE W2 client half, research/console_charter.md).
 *
 * server/satellites.ts serves the latest CelesTrak GP element set (OMM
 * JSON — NOT TLE, see that file's header for why) per NORAD_CAT_ID. This
 * module turns those mean elements into a map position, client-side,
 * with satellite.js doing the SGP4/SDP4 math.
 *
 * VERSION JUDGMENT CALL: satellite.js's npm `latest` (7.x, published
 * 2026-05) is a recent rewrite that ships an optional WASM propagator —
 * extra bundle/build-tool surface this repo's esbuild/Vite pipeline has
 * no precedent for. This module targets satellite.js@4.1.4: the
 * dependency-free, pure-JS major that has shipped unchanged since 2022
 * and is what most existing satellite.js integrations (and this repo's
 * "no key, client-side SGP4/SDP4" requirement) actually mean. Its public
 * API is TLE-line-pair only (`twoline2satrec(line1, line2)` — no JSON/OMM
 * constructor), so `ommToTle` below does the conversion ourselves,
 * verified against a real CelesTrak-published ISS OMM/TLE pair (see
 * satelliteOrbits.test.ts) rather than guessed from the TLE spec alone.
 *
 * TLE column layout verified against the NORAD/Vallado spec AND a live
 * CelesTrak ISS TLE fetched 2026-07-07 (both agree). Fields satellite.js's
 * twoline2satrec doesn't even read (classification, intl designator,
 * ephemeris type, element/rev numbers) are still filled in with
 * spec-shaped placeholders when the source OMM omits them — propagation
 * never depends on them, but the string must stay a syntactically valid
 * 69-column TLE line with a correct checksum.
 */

export type SatGroup = "stations" | "starlink" | "gps-ops" | "geo";

/** Mirrors server/satellites.ts GROUPS — kept in sync by hand: the client
 *  bundle cannot import server/*.ts (Node-only deps: fs, zlib, ...). */
export const SAT_GROUPS: readonly SatGroup[] = ["stations", "starlink", "gps-ops", "geo"];

export const SAT_GROUP_LABEL: Record<SatGroup, string> = {
  stations: "Space stations",
  starlink: "Starlink",
  "gps-ops": "GPS operational",
  geo: "Active geosynchronous",
};

/** Distinct per-group tint for the map + legend — chosen to stay visually
 *  apart from colors already in use elsewhere on the map (aircraft blue,
 *  fire/alert red-orange-yellow ramps, vessel greens). */
export const SATELLITE_GROUP_COLOR: Record<SatGroup, string> = {
  stations: "#fbbf24",
  starlink: "#38bdf8",
  "gps-ops": "#a78bfa",
  geo: "#fb7185",
};

/** Mirrors server/satellites.ts ATTRIBUTION exactly (same hand-sync
 *  constraint as SAT_GROUPS above). */
export const SATELLITE_ATTRIBUTION =
  "Orbital data courtesy of CelesTrak (celestrak.org), Dr. T.S. Kelso";

/** The OMM/GP fields CelesTrak's gp.php (FORMAT=json) returns and this
 *  module needs — a structural subset of server/satellites.ts's GpRecord
 *  (duplicated, not imported, for the same client/server boundary reason
 *  as the constants above). Numeric fields may arrive as number or string
 *  (CelesTrak uses number; some other OMM producers use string). */
export interface OmmLike {
  NORAD_CAT_ID: number | string;
  OBJECT_NAME?: string;
  OBJECT_ID?: string;
  EPOCH: string;
  CLASSIFICATION_TYPE?: string;
  MEAN_MOTION: number | string;
  ECCENTRICITY: number | string;
  INCLINATION: number | string;
  RA_OF_ASC_NODE: number | string;
  ARG_OF_PERICENTER: number | string;
  MEAN_ANOMALY: number | string;
  EPHEMERIS_TYPE?: number | string;
  ELEMENT_SET_NO?: number | string;
  REV_AT_EPOCH?: number | string;
  BSTAR: number | string;
  MEAN_MOTION_DOT: number | string;
  MEAN_MOTION_DDOT: number | string;
  [k: string]: unknown;
}

// ── TLE encoding primitives (NORAD/Vallado two-line element spec) ──────────

/** TLE checksum: sum of all digits in cols 1-68 (a bare '-' counts as 1,
 *  everything else — letters, spaces, '.', '+' — counts as 0), mod 10. */
export function tleChecksum(line68: string): number {
  let total = 0;
  for (let i = 0; i < line68.length; i++) {
    const ch = line68[i];
    if (ch >= "0" && ch <= "9") total += ch.charCodeAt(0) - 48;
    else if (ch === "-") total += 1;
  }
  return total % 10;
}

/** TLE's signed decimal-exponent field (8 chars) used for the second
 *  mean-motion derivative and BSTAR: [sign or space][5-digit mantissa]
 *  [exponent sign][1-digit exponent], meaning value = sign * 0.mantissa *
 *  10^exponent. Zero encodes as " 00000+0" (CelesTrak's own convention). */
function expField(value: number): string {
  const sign = value < 0 ? "-" : " ";
  let v = Math.abs(value);
  if (v === 0) return " 00000+0";
  let exp = 0;
  while (v >= 1) { v /= 10; exp++; }
  while (v < 0.1) { v *= 10; exp--; }
  let mantissa = Math.round(v * 1e5);
  if (mantissa >= 100000) { mantissa = Math.round(mantissa / 10); exp++; } // rounding overflow (v -> 1.0)
  const expSign = exp < 0 ? "-" : "+";
  return `${sign}${String(mantissa).padStart(5, "0")}${expSign}${Math.abs(exp)}`;
}

/** TLE's signed fixed-decimal field (10 chars) used for the first
 *  mean-motion derivative: [sign or space].{8 digits}, no leading zero. */
function signedDecimalField(value: number, decimals = 8): string {
  const sign = value < 0 ? "-" : " ";
  const digits = Math.round(Math.abs(value) * 10 ** decimals).toString().padStart(decimals, "0");
  return `${sign}.${digits}`;
}

/** Right-justified fixed-decimal field, space-padded (inclination/RAAN/
 *  arg-of-pericenter/mean-anomaly/mean-motion columns). */
function padDecimal(value: number, totalWidth: number, decimals: number): string {
  return value.toFixed(decimals).padStart(totalWidth, " ");
}

/** International designator (cols 10-17): "YYYY-NNNA" (OBJECT_ID) splits
 *  into 2-digit launch year, 3-digit launch number, up to 3-letter piece.
 *  Unparseable/missing OBJECT_ID gets spec-shaped placeholders — this
 *  field is descriptive only, never read by twoline2satrec. */
function intlDesignator(objectId?: string): { yr: string; num: string; piece: string } {
  const m = /^(\d{4})-(\d{3})([A-Za-z]*)$/.exec(String(objectId || "").trim());
  if (!m) return { yr: "00", num: "000", piece: "   " };
  return { yr: m[1].slice(2), num: m[2], piece: (m[3] || "").padEnd(3, " ").slice(0, 3) };
}

const CUM_DAYS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
function isLeapYear(y: number): boolean {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

/** TLE epoch fields (cols 19-32): 2-digit year + "DDD.DDDDDDDD" day-of-year
 *  with fractional day. Parsed from the ISO EPOCH string by hand (not via
 *  `Date`, whose millisecond resolution would round away the source's
 *  microsecond precision and drift the last output digit — verified
 *  against a live CelesTrak ISS epoch during development). */
function epochToTleFields(epochIso: string): { yy: string; dayFrac: string } {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)/.exec(String(epochIso));
  if (!m) return { yy: "00", dayFrac: "001.00000000" };
  const year = Number(m[1]), month = Number(m[2]), day = Number(m[3]);
  const hours = Number(m[4]), minutes = Number(m[5]), seconds = Number(m[6]);
  const yy = String(year % 100).padStart(2, "0");
  let doy = CUM_DAYS[month - 1] + day;
  if (isLeapYear(year) && month > 2) doy += 1;
  const frac = (hours * 3600 + minutes * 60 + seconds) / 86400;
  return { yy, dayFrac: `${String(doy).padStart(3, "0")}.${frac.toFixed(8).slice(2)}` };
}

/** Convert one CelesTrak OMM/GP record into a [line1, line2] TLE pair
 *  satellite.js@4.1.4's `twoline2satrec` can parse. Round-trip-verified
 *  against a live CelesTrak ISS OMM+TLE pair in satelliteOrbits.test.ts. */
export function ommToTle(rec: OmmLike): [string, string] {
  const satnum = String(rec.NORAD_CAT_ID).padStart(5, "0").slice(-5);
  const cls = String(rec.CLASSIFICATION_TYPE || "U").slice(0, 1);
  const { yr, num, piece } = intlDesignator(rec.OBJECT_ID);
  const { yy, dayFrac } = epochToTleFields(rec.EPOCH);
  const ndot = signedDecimalField(Number(rec.MEAN_MOTION_DOT ?? 0));
  const nddot = expField(Number(rec.MEAN_MOTION_DDOT ?? 0));
  const bstar = expField(Number(rec.BSTAR ?? 0));
  const ephem = String(rec.EPHEMERIS_TYPE ?? 0).slice(0, 1);
  const elset = String(rec.ELEMENT_SET_NO ?? 0).padStart(4, " ").slice(-4);
  const body1 = `1 ${satnum}${cls} ${yr}${num}${piece} ${yy}${dayFrac} ${ndot} ${nddot} ${bstar} ${ephem} ${elset}`;
  const line1 = `${body1}${tleChecksum(body1)}`;

  const eccField = String(Math.round(Number(rec.ECCENTRICITY) * 1e7)).padStart(7, "0").slice(-7);
  const revnum = String(rec.REV_AT_EPOCH ?? 0).padStart(5, " ").slice(-5);
  const body2 = `2 ${satnum} ${padDecimal(Number(rec.INCLINATION), 8, 4)} ` +
    `${padDecimal(Number(rec.RA_OF_ASC_NODE), 8, 4)} ${eccField} ` +
    `${padDecimal(Number(rec.ARG_OF_PERICENTER), 8, 4)} ${padDecimal(Number(rec.MEAN_ANOMALY), 8, 4)} ` +
    `${padDecimal(Number(rec.MEAN_MOTION), 11, 8)}${revnum}`;
  const line2 = `${body2}${tleChecksum(body2)}`;
  return [line1, line2];
}

// ── display helpers ─────────────────────────────────────────────────────────

/** MEAN_MOTION is revolutions/day (OMM convention) — period is the simple
 *  inverse, in minutes. */
export function periodMinutesFromMeanMotion(meanMotionRevPerDay: number): number {
  if (!Number.isFinite(meanMotionRevPerDay) || meanMotionRevPerDay <= 0) return NaN;
  return 1440 / meanMotionRevPerDay;
}

/** Honesty chip text for how stale a mean-element set is, matching this
 *  file's other freshness-age formatting (datamap.tsx's `formatAge`) —
 *  extended to days since GP epochs can run hours-to-a-day+ behind our
 *  6h poll cadence. */
export function formatElementAge(epochIso: string, nowMs: number = Date.now()): string {
  const t = Date.parse(/[zZ]$|[+-]\d\d:\d\d$/.test(epochIso) ? epochIso : `${epochIso}Z`);
  if (!Number.isFinite(t)) return "epoch unknown";
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 36 * 3600) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

/** Public CelesTrak catalog lookup for an object name — a real, live
 *  endpoint (verified 2026-07-07), not a guessed deep link. */
export function celestrakSatcatUrl(objectName: string): string {
  return `https://celestrak.org/satcat/table-satcat.php?NAME=${encodeURIComponent(objectName)}`;
}
