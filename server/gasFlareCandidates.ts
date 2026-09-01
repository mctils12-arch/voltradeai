/**
 * gasFlareCandidates.ts — persistent-hotspot gas-flare CANDIDATE detector,
 * built entirely over the NASA FIRMS active-fire archive we already ingest
 * (nasaFirms.ts) instead of registering for EOG/Colorado School of Mines'
 * VIIRS Nightfire product.
 *
 * BUILD-FIRST (CLAUDE.md): research/wishlist.md's GEOSPATIAL registry had
 * flagged VIIRS Nightfire as "the strongest pure EDGE-DOCTRINE candidate on
 * the board" but blocked from even entering the wishlist without a
 * free-alternative writeup first (eogdata.mines.edu gates the raw
 * flare-radiant-heat download behind free registration). Rule item 1 ("do
 * we already receive the raw material?") applies directly: VIIRS active-fire
 * detections — the same satellite, a public, keyless-beyond-MAP_KEY,
 * already-flowing feed (nasaFirms.ts) — are a strict superset of what
 * Nightfire's own flare-classification step consumes as input. Full
 * analysis: research/open_questions.md, dated entry "GAS FLARE CANDIDATES".
 *
 * METHOD (published precedent, not invented here): a gas flare is a
 * near-continuous point source that re-triggers the active-fire detector at
 * the SAME pixel night after night. A wildfire spreads, moves, or
 * extinguishes within days. Temporal persistence at a fixed location is the
 * standard technique for separating flares from wildfires in VIIRS/MODIS
 * active-fire products that predates the dedicated Nightfire algorithm (see
 * e.g. "Application of machine learning to gas flaring", arXiv:2301.04141,
 * and NOAA/NESDIS's own public explainer on VIIRS flare detection).
 *
 * HONESTY (BUILD-FIRST clause): this is a CANDIDATE/estimate product, not a
 * Nightfire-equivalent. It does not reproduce Nightfire's low-gain
 * multi-band radiant-heat/temperature retrieval (that needs data this
 * module does not have) — it flags recurring hotspot sites and a coarse
 * activity proxy (nights active, mean FRP). Never present this as
 * ground-truth flare temperature/volume; label it "candidate"/"inferred"
 * wherever it surfaces.
 *
 * GATE 1 (DATA) PLAN, documented not executed this session (no production
 * /data/voltrade archive exists in this sandbox — the same constraint the
 * 2026-08-29 shadow-fleet gate-1 session logged): once >=30 days of real
 * FIRMS archive has accumulated, aggregate candidate sites per country
 * (needs a country-boundary join — candidatesByRegion() below takes that
 * join as an injected function so a future session can supply it without
 * touching this module) and rank-correlate the per-country candidate count
 * against the World Bank Global Gas Flaring Reduction Partnership's public,
 * free annual country flaring-volume rankings
 * (datacatalog.worldbank.org/search/dataset/0037743) — a genuine external
 * ground-truth source at country-aggregate resolution. Pre-registered pass
 * bar for a future gate-1 script: Spearman rank correlation against GGFR's
 * published top-15 flaring countries, same discipline as this repo's other
 * gate-1 designs (dtcc_swaps_gate1.ts, shadowFleetGate1.ts).
 *
 * Pure functions only — no fs/network access. Consumes a FIRMS-detection
 * shape (structurally compatible with nasaFirms.ts's FireDetection, not
 * imported from it) so this module stays testable with synthetic fixtures
 * and callers pass readFireHistory()'s output straight through.
 */

export interface FlareCandidateInput {
  lat: number;
  lon: number;
  acq_date: string;             // YYYY-MM-DD, UTC
  daynight: "D" | "N" | null;
  frp: number | null;           // fire radiative power, MW
}

export interface GasFlareCandidate {
  siteKey: string;
  lat: number;                  // centroid of all detections assigned to this site
  lon: number;
  nightsActive: number;         // distinct nighttime acq_date with a detection at this site
  nightsInWindow: number;       // distinct nighttime acq_date present ANYWHERE in the input
  persistence: number;          // nightsActive / nightsInWindow
  meanFrp: number | null;
  firstSeen: string;
  lastSeen: string;
  detectionCount: number;
}

// ~830m at the equator — coarser than a VIIRS 375m pixel to absorb
// pass-to-pass geolocation jitter, fine enough that distinct well-pad/
// flare-stack sites (typically spaced well beyond this) stay separate.
export const GRID_DEG = 0.0075;

export function gridKey(lat: number, lon: number, gridDeg = GRID_DEG): string {
  const gLat = Math.round(lat / gridDeg) * gridDeg;
  const gLon = Math.round(lon / gridDeg) * gridDeg;
  return `${gLat.toFixed(4)}:${gLon.toFixed(4)}`;
}

export interface FindCandidatesOptions {
  gridDeg?: number;
  minNights?: number;      // minimum distinct nights active to qualify (default 3)
  minPersistence?: number; // minimum nightsActive / nightsInWindow to qualify (default 0.5)
}

/** Nighttime-only by design: flares burn continuously so recur at night
 *  regardless of season, while a genuine wildfire's daytime-dominant
 *  detections would otherwise dilute the persistence signal this depends
 *  on. Daytime detections are not consulted at all here. */
export function findGasFlareCandidates(
  detections: readonly FlareCandidateInput[],
  opts: FindCandidatesOptions = {},
): GasFlareCandidate[] {
  const gridDeg = opts.gridDeg ?? GRID_DEG;
  const minNights = opts.minNights ?? 3;
  const minPersistence = opts.minPersistence ?? 0.5;

  const nightDates = new Set<string>();
  for (const d of detections) if (d.daynight === "N") nightDates.add(d.acq_date);
  const nightsInWindow = nightDates.size;

  interface Acc {
    latSum: number; lonSum: number; n: number;
    nights: Set<string>; frpSum: number; frpN: number; dates: string[];
  }
  const sites = new Map<string, Acc>();
  for (const d of detections) {
    if (d.daynight !== "N") continue;
    const key = gridKey(d.lat, d.lon, gridDeg);
    let acc = sites.get(key);
    if (!acc) {
      acc = { latSum: 0, lonSum: 0, n: 0, nights: new Set(), frpSum: 0, frpN: 0, dates: [] };
      sites.set(key, acc);
    }
    acc.latSum += d.lat;
    acc.lonSum += d.lon;
    acc.n++;
    acc.nights.add(d.acq_date);
    acc.dates.push(d.acq_date);
    if (d.frp != null) { acc.frpSum += d.frp; acc.frpN++; }
  }

  const out: GasFlareCandidate[] = [];
  for (const [key, acc] of sites) {
    const nightsActive = acc.nights.size;
    const persistence = nightsInWindow > 0 ? nightsActive / nightsInWindow : 0;
    if (nightsActive < minNights || persistence < minPersistence) continue;
    const sortedDates = [...acc.dates].sort();
    out.push({
      siteKey: key,
      lat: acc.latSum / acc.n,
      lon: acc.lonSum / acc.n,
      nightsActive,
      nightsInWindow,
      persistence,
      meanFrp: acc.frpN > 0 ? acc.frpSum / acc.frpN : null,
      firstSeen: sortedDates[0],
      lastSeen: sortedDates[sortedDates.length - 1],
      detectionCount: acc.n,
    });
  }
  return out.sort((a, b) => b.persistence - a.persistence || (b.meanFrp ?? 0) - (a.meanFrp ?? 0));
}

/** Aggregates candidates into caller-supplied regions (e.g. countries) for
 *  the GATE 1 PLAN's rank-correlation step. `regionOf` is injected rather
 *  than implemented here — this module has no country-boundary data, and
 *  making that a hard dependency would block everything above it on a
 *  separate build. Returns null-region candidates uncounted. */
export function candidatesByRegion(
  candidates: readonly GasFlareCandidate[],
  regionOf: (lat: number, lon: number) => string | null,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of candidates) {
    const region = regionOf(c.lat, c.lon);
    if (!region) continue;
    out[region] = (out[region] || 0) + 1;
  }
  return out;
}
