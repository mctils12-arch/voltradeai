// CelesTrak client-fetch layer (BROWSER-side).
//
// WHY BROWSER-SIDE: CelesTrak (celestrak.org) is firewalled from the
// Railway datacenter IP range (R17, verified). A visitor's browser reaches
// celestrak.org fine (residential/mobile IPs), so TLE/OMM + SATCAT are
// fetched CLIENT-SIDE and propagated in the browser with SGP4. This module
// is pure TS with zero React/DOM/server dependencies so it is unit-testable
// under `npx tsx --test`.
//
// FIELD NAMES here mirror the REAL CelesTrak schemas, verified live
// 2026-07-07 against:
//   - GP/OMM JSON:  https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json
//                   (OMM keywords per CCSDS 502.0-B-3)
//   - SATCAT CSV:   https://celestrak.org/pub/satcat.csv
//
// HONESTY RULE (from the ORBITAL charter): NEVER fabricate a field. A value
// that is absent, blank, or unparseable is `null` — never guessed, never
// defaulted to a plausible-looking number.

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const CELESTRAK_ORIGIN = 'https://celestrak.org';

/**
 * CelesTrak GP (General Perturbations) element query URL.
 * `group` is a CelesTrak GROUP name; `active` is the full active catalog
 * (~10k+ objects) and the default. Other useful groups: `starlink`,
 * `gps-ops`, `geo`, `oneweb`, `galileo`, `science`, `stations`.
 *
 * `format` selects the wire encoding. Both `json` (OMM) and `csv` carry the
 * SAME OMM keywords/fields; CSV is ~2.4 MB vs JSON's ~6.7 MB for the full
 * active catalog (identical data, no repeated keys), so the browser layer
 * fetches CSV — a 64% smaller transfer that loads far faster on mobile and
 * is much less likely to trip CelesTrak's courtesy rate limit mid-download.
 *
 * NOTE (charter): CelesTrak has exhausted 5-digit catalog numbers
 * (~2026-07-12), so OMM (json OR csv) — NOT legacy TLE text — is mandated.
 */
export function gpUrl(group: string = 'active', format: 'json' | 'csv' = 'json'): string {
  return `${CELESTRAK_ORIGIN}/NORAD/elements/gp.php?GROUP=${encodeURIComponent(
    group,
  )}&FORMAT=${format}`;
}

/** CelesTrak full SATCAT metadata catalog, CSV format (~6 MB). */
export function satcatUrl(): string {
  return `${CELESTRAK_ORIGIN}/pub/satcat.csv`;
}

// ---------------------------------------------------------------------------
// Raw wire shapes (what CelesTrak actually returns)
// ---------------------------------------------------------------------------

/** Raw OMM JSON record as returned by the GP query (FORMAT=json). */
export interface OmmJson {
  OBJECT_NAME?: string;
  OBJECT_ID?: string;
  EPOCH?: string;
  MEAN_MOTION?: number | string;
  ECCENTRICITY?: number | string;
  INCLINATION?: number | string;
  RA_OF_ASC_NODE?: number | string;
  ARG_OF_PERICENTER?: number | string;
  MEAN_ANOMALY?: number | string;
  EPHEMERIS_TYPE?: number | string;
  CLASSIFICATION_TYPE?: string;
  NORAD_CAT_ID?: number | string;
  ELEMENT_SET_NO?: number | string;
  REV_AT_EPOCH?: number | string;
  BSTAR?: number | string;
  MEAN_MOTION_DOT?: number | string;
  MEAN_MOTION_DDOT?: number | string;
}

// ---------------------------------------------------------------------------
// Normalized records
// ---------------------------------------------------------------------------

/** Normalized GP element record. Angles in DEGREES, meanMotion in REV/DAY. */
export interface GpRecord {
  noradId: number;
  name: string | null;
  /** ISO-8601 UTC epoch string, exactly as CelesTrak provides it. */
  epoch: string | null;
  inclination: number | null; // degrees
  raan: number | null; // degrees (RA_OF_ASC_NODE)
  ecc: number | null; // eccentricity, dimensionless
  argp: number | null; // degrees (ARG_OF_PERICENTER)
  meanAnomaly: number | null; // degrees
  meanMotion: number | null; // revolutions per day
  bstar: number | null; // drag term, 1/earth-radii
}

export type ObjectType = 'PAYLOAD' | 'ROCKET BODY' | 'DEBRIS' | 'UNKNOWN';
export type OrbitClass = 'LEO' | 'MEO' | 'GEO' | 'HEO';
export type RcsSize = 'SMALL' | 'MEDIUM' | 'LARGE';

/** Normalized SATCAT metadata record. */
export interface SatcatRecord {
  noradId: number;
  name: string | null;
  intlDes: string | null; // OBJECT_ID, international designator
  /**
   * OWNER — CelesTrak's source/ownership code (e.g. "US", "PRC", "CIS",
   * "ESA", "ISS"). SATCAT has ONE ownership column; it encodes the owning
   * country/agency, NOT a distinct operator-company name. `owner` and
   * `country` therefore surface the SAME underlying datum (the operator→
   * company→ticker mapping the entity graph wants is a downstream lookup,
   * not present in SATCAT).
   */
  owner: string | null;
  country: string | null; // == OWNER code (see note above)
  launchDate: string | null; // YYYY-MM-DD
  objectType: ObjectType | null; // PAY/R/B/DEB/UNK normalized
  opStatus: string | null; // OPS_STATUS_CODE, raw single-char code
  /**
   * DERIVED orbit class. SATCAT has no LEO/MEO/GEO column (ORBIT_TYPE is a
   * disposition code: ORB/IMP/LAN/DOC/R/T). Classified here from APOGEE /
   * PERIGEE altitudes — labeled derived, not a CelesTrak-provided field.
   */
  orbitClass: OrbitClass | null;
  rcsMeters2: number | null; // raw RCS in square meters (or null if blank)
  rcsSize: RcsSize | null; // derived bucket from rcsMeters2
}

/** GP + SATCAT joined on NORAD ID. Union of both sides; nothing dropped. */
export interface JoinedSat {
  noradId: number;
  name: string | null;
  // --- orbital (from GP); null when no GP element for this NORAD id ---
  epoch: string | null;
  inclination: number | null;
  raan: number | null;
  ecc: number | null;
  argp: number | null;
  meanAnomaly: number | null;
  meanMotion: number | null;
  bstar: number | null;
  // --- metadata (from SATCAT); null when no SATCAT row for this NORAD id ---
  intlDes: string | null;
  owner: string | null;
  country: string | null;
  launchDate: string | null;
  objectType: ObjectType | null;
  opStatus: string | null;
  orbitClass: OrbitClass | null;
  rcsMeters2: number | null;
  rcsSize: RcsSize | null;
  // --- provenance flags ---
  hasGp: boolean;
  hasSatcat: boolean;
}

// ---------------------------------------------------------------------------
// Coercion helpers — the ONLY place blank/missing becomes null
// ---------------------------------------------------------------------------

function numOrNull(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string') {
    const t = v.trim();
    if (t === '') return null;
    const n = Number(t);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function strOrNull(v: unknown): string | null {
  if (typeof v === 'string') {
    const t = v.trim();
    return t === '' ? null : t;
  }
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  return null;
}

// ---------------------------------------------------------------------------
// parseGp
// ---------------------------------------------------------------------------

/**
 * Normalize CelesTrak GP JSON (an array of OMM records, or its JSON text)
 * into GpRecord[]. Records with no usable NORAD_CAT_ID are dropped (they
 * cannot be joined or propagated) — every other missing field becomes null.
 */
export function parseGp(json: unknown): GpRecord[] {
  let arr: unknown = json;
  if (typeof json === 'string') {
    try {
      arr = JSON.parse(json);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(arr)) return [];

  const out: GpRecord[] = [];
  for (const raw of arr) {
    if (raw == null || typeof raw !== 'object') continue;
    const o = raw as OmmJson;
    const noradId = numOrNull(o.NORAD_CAT_ID);
    if (noradId == null) continue; // no join key => unusable
    out.push({
      noradId,
      name: strOrNull(o.OBJECT_NAME),
      epoch: strOrNull(o.EPOCH),
      inclination: numOrNull(o.INCLINATION),
      raan: numOrNull(o.RA_OF_ASC_NODE),
      ecc: numOrNull(o.ECCENTRICITY),
      argp: numOrNull(o.ARG_OF_PERICENTER),
      meanAnomaly: numOrNull(o.MEAN_ANOMALY),
      meanMotion: numOrNull(o.MEAN_MOTION),
      bstar: numOrNull(o.BSTAR),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// parseGpCsv
// ---------------------------------------------------------------------------

/**
 * Normalize CelesTrak GP CSV (gp.php?FORMAT=csv) into GpRecord[]. Same OMM
 * fields as parseGp, just CSV-encoded (~64% smaller on the wire). Columns are
 * located by HEADER NAME (robust to reordering); rows with no usable
 * NORAD_CAT_ID are dropped, every other missing/blank field becomes null.
 * Garbage or a non-GP CSV yields [] and never throws — the fetch layer turns
 * a rate-limit/error body (which is not GP CSV) into a proper thrown error via
 * res.ok, so an empty parse here means "genuinely no elements", not "HTTP error".
 */
export function parseGpCsv(csv: string): GpRecord[] {
  if (typeof csv !== 'string' || csv.trim() === '') return [];
  const lines = csv.split(/\r?\n/);
  let headerIdx = 0;
  while (headerIdx < lines.length && lines[headerIdx].trim() === '') headerIdx++;
  if (headerIdx >= lines.length) return [];

  const header = parseCsvLine(lines[headerIdx]).map((h) => h.trim());
  const col = (name: string): number => header.indexOf(name);
  const iNorad = col('NORAD_CAT_ID');
  if (iNorad < 0) return []; // not a GP CSV (e.g. an error/HTML body)
  const iName = col('OBJECT_NAME');
  const iEpoch = col('EPOCH');
  const iIncl = col('INCLINATION');
  const iRaan = col('RA_OF_ASC_NODE');
  const iEcc = col('ECCENTRICITY');
  const iArgp = col('ARG_OF_PERICENTER');
  const iMeanAnom = col('MEAN_ANOMALY');
  const iMeanMotion = col('MEAN_MOTION');
  const iBstar = col('BSTAR');

  const at = (fields: string[], idx: number): string | null =>
    idx >= 0 && idx < fields.length ? strOrNull(fields[idx]) : null;

  const out: GpRecord[] = [];
  for (let i = headerIdx + 1; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    const f = parseCsvLine(lines[i]);
    const noradId = numOrNull(at(f, iNorad));
    if (noradId == null) continue; // no join key => unusable
    out.push({
      noradId,
      name: at(f, iName),
      epoch: at(f, iEpoch),
      inclination: numOrNull(at(f, iIncl)),
      raan: numOrNull(at(f, iRaan)),
      ecc: numOrNull(at(f, iEcc)),
      argp: numOrNull(at(f, iArgp)),
      meanAnomaly: numOrNull(at(f, iMeanAnom)),
      meanMotion: numOrNull(at(f, iMeanMotion)),
      bstar: numOrNull(at(f, iBstar)),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// parseSatcat
// ---------------------------------------------------------------------------

/** Split one CSV line into fields, honoring RFC-4180 double-quoting. */
function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ',') {
      out.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
}

function normalizeObjectType(raw: string | null): ObjectType | null {
  if (raw == null) return null;
  switch (raw.toUpperCase()) {
    case 'PAY':
      return 'PAYLOAD';
    case 'R/B':
      return 'ROCKET BODY';
    case 'DEB':
      return 'DEBRIS';
    case 'UNK':
      return 'UNKNOWN';
    default:
      return null; // unknown code => honest null, not a guess
  }
}

/**
 * CelesTrak's standard RCS size buckets: SMALL < 0.1 m², MEDIUM 0.1–1.0 m²,
 * LARGE > 1.0 m². Derived from the numeric RCS column.
 */
function rcsSizeFromM2(m2: number | null): RcsSize | null {
  if (m2 == null) return null;
  if (m2 < 0.1) return 'SMALL';
  if (m2 <= 1.0) return 'MEDIUM';
  return 'LARGE';
}

/**
 * Classify orbit from apogee/perigee altitudes (km above mean earth radius).
 * SATCAT provides no orbit-class column, so this is a labeled DERIVATION:
 *   HEO  — perigee low (<2000) but apogee high (>2000) with a large gap
 *          (Molniya, GTO, most transfer/eccentric orbits)
 *   LEO  — mean altitude < 2000 km
 *   MEO  — mean altitude 2000 .. ~33786 km (GPS/GLONASS/Galileo band)
 *   GEO  — mean altitude ~35786 ± 2000 km (geosynchronous band)
 *   HEO  — anything above the GEO band (super-synchronous)
 */
export function classifyOrbit(
  apogeeKm: number | null,
  perigeeKm: number | null,
): OrbitClass | null {
  if (apogeeKm == null || perigeeKm == null) return null;
  const GEO = 35786;
  if (apogeeKm > 2000 && perigeeKm < 2000 && apogeeKm - perigeeKm > 5000) {
    return 'HEO';
  }
  const meanAlt = (apogeeKm + perigeeKm) / 2;
  if (meanAlt < 2000) return 'LEO';
  if (meanAlt < GEO - 2000) return 'MEO';
  if (meanAlt <= GEO + 2000) return 'GEO';
  return 'HEO';
}

/**
 * Parse CelesTrak satcat.csv text into SatcatRecord[]. Columns are located by
 * HEADER NAME (robust to column reordering). Rows with no usable NORAD_CAT_ID
 * are dropped; all other blank fields become null.
 */
export function parseSatcat(csv: string): SatcatRecord[] {
  if (typeof csv !== 'string' || csv.trim() === '') return [];
  const lines = csv.split(/\r?\n/);
  // find the header line (first non-empty)
  let headerIdx = 0;
  while (headerIdx < lines.length && lines[headerIdx].trim() === '') headerIdx++;
  if (headerIdx >= lines.length) return [];

  const header = parseCsvLine(lines[headerIdx]).map((h) => h.trim());
  const col = (name: string): number => header.indexOf(name);
  const iNorad = col('NORAD_CAT_ID');
  if (iNorad < 0) return []; // not a SATCAT CSV
  const iName = col('OBJECT_NAME');
  const iIntl = col('OBJECT_ID');
  const iType = col('OBJECT_TYPE');
  const iOps = col('OPS_STATUS_CODE');
  const iOwner = col('OWNER');
  const iLaunch = col('LAUNCH_DATE');
  const iApogee = col('APOGEE');
  const iPerigee = col('PERIGEE');
  const iRcs = col('RCS');

  const at = (fields: string[], idx: number): string | null =>
    idx >= 0 && idx < fields.length ? strOrNull(fields[idx]) : null;

  const out: SatcatRecord[] = [];
  for (let i = headerIdx + 1; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    const f = parseCsvLine(lines[i]);
    const noradId = numOrNull(at(f, iNorad));
    if (noradId == null) continue;
    const owner = at(f, iOwner);
    const rcsMeters2 = numOrNull(at(f, iRcs));
    out.push({
      noradId,
      name: at(f, iName),
      intlDes: at(f, iIntl),
      owner,
      country: owner, // SATCAT OWNER is a country/agency code (see interface)
      launchDate: at(f, iLaunch),
      objectType: normalizeObjectType(at(f, iType)),
      opStatus: at(f, iOps),
      orbitClass: classifyOrbit(
        numOrNull(at(f, iApogee)),
        numOrNull(at(f, iPerigee)),
      ),
      rcsMeters2,
      rcsSize: rcsSizeFromM2(rcsMeters2),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// joinGpSatcat
// ---------------------------------------------------------------------------

/**
 * Join GP elements to SATCAT metadata on NORAD ID. Returns the UNION: every
 * NORAD id present in either input yields exactly one JoinedSat. A GP record
 * with no SATCAT match keeps its orbital fields and nulls the metadata (and
 * hasSatcat=false); a SATCAT record with no GP match nulls the orbital fields
 * (hasGp=false). NOTHING is dropped — the no-silent-drops rule.
 */
export function joinGpSatcat(
  gp: GpRecord[],
  satcat: SatcatRecord[],
): JoinedSat[] {
  const satById = new Map<number, SatcatRecord>();
  for (const s of satcat) satById.set(s.noradId, s);
  const gpById = new Map<number, GpRecord>();
  for (const g of gp) gpById.set(g.noradId, g);

  const ids: number[] = [];
  const seen = new Set<number>();
  for (const g of gp)
    if (!seen.has(g.noradId)) {
      seen.add(g.noradId);
      ids.push(g.noradId);
    }
  for (const s of satcat)
    if (!seen.has(s.noradId)) {
      seen.add(s.noradId);
      ids.push(s.noradId);
    }

  const out: JoinedSat[] = [];
  for (const id of ids) {
    const g = gpById.get(id) ?? null;
    const s = satById.get(id) ?? null;
    out.push({
      noradId: id,
      name: g?.name ?? s?.name ?? null,
      epoch: g?.epoch ?? null,
      inclination: g?.inclination ?? null,
      raan: g?.raan ?? null,
      ecc: g?.ecc ?? null,
      argp: g?.argp ?? null,
      meanAnomaly: g?.meanAnomaly ?? null,
      meanMotion: g?.meanMotion ?? null,
      bstar: g?.bstar ?? null,
      intlDes: s?.intlDes ?? null,
      owner: s?.owner ?? null,
      country: s?.country ?? null,
      launchDate: s?.launchDate ?? null,
      objectType: s?.objectType ?? null,
      opStatus: s?.opStatus ?? null,
      orbitClass: s?.orbitClass ?? null,
      rcsMeters2: s?.rcsMeters2 ?? null,
      rcsSize: s?.rcsSize ?? null,
      hasGp: g != null,
      hasSatcat: s != null,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Thin browser fetch helpers (NOT exercised by the hermetic test path)
// ---------------------------------------------------------------------------

type FetchLike = (
  url: string,
) => Promise<{ ok?: boolean; status?: number; json(): Promise<unknown>; text(): Promise<string> }>;

/**
 * Fetch + parse a GP group in the browser. `fetchImpl` is injectable for tests.
 *
 * Uses CSV (see gpUrl): ~64% smaller than JSON, so the ~16k-object active
 * catalog downloads far faster on mobile. Crucially, a non-2xx response (e.g.
 * CelesTrak's HTTP 403 courtesy rate-limit, whose body is NOT GP data) now
 * THROWS with its status instead of being silently parsed to [] — that empty
 * parse used to read as "no elements returned", trapping the layer in an
 * endless "still retrying automatically…" loop. A real error surfaces to the
 * resilient-load backoff, which is what the retry ladder is for.
 */
export async function fetchGp(
  group: string = 'active',
  fetchImpl?: FetchLike,
): Promise<GpRecord[]> {
  const f = fetchImpl ?? (globalThis.fetch as unknown as FetchLike);
  const res = await f(gpUrl(group, 'csv'));
  if (res.ok === false) {
    throw new Error(`CelesTrak GP fetch failed: HTTP ${res.status ?? 'error'}`);
  }
  return parseGpCsv(await res.text());
}

/** Fetch + parse the SATCAT CSV in the browser. `fetchImpl` is injectable for tests. */
export async function fetchSatcat(fetchImpl?: FetchLike): Promise<SatcatRecord[]> {
  const f = fetchImpl ?? (globalThis.fetch as unknown as FetchLike);
  const res = await f(satcatUrl());
  return parseSatcat(await res.text());
}
