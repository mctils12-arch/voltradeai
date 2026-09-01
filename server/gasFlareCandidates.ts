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
 * GATE 1 (DATA): RUN 2026-09-01, a same-day session with live production
 * DIAG_TOKEN access (scripts/gasflare_gate1.ts, server/countryLookup.ts —
 * the country-boundary join this module's own candidatesByRegion() was
 * built to accept). RESULT: GATE 1 FAIL — Spearman rho -0.4762 (n=8,
 * critical value +0.738) between per-country candidate density and the
 * World Bank 2026 Global Gas Flaring Tracker Report's published country
 * order; USA's candidate count led the sample despite ranking lowest of
 * the 8 tested, with evidence of wildfire-season contamination the
 * persistence filter did not exclude. Full account, including the scope
 * cut (Russia untestable via a single bbox — its polygon crosses the
 * antimeridian) and the diagnosis: datacore/signal_ladder.json's
 * `gas_flare_candidates` entry and research/open_questions.md's GAS FLARE
 * CANDIDATES entry, 2026-09-01 addendum.
 *
 * WILDFIRE DISCRIMINATOR (added same day, NEXT step (1) from that
 * addendum): VIIRS Nightfire's own published methodology (Elvidge et al.,
 * "VIIRS Nightfire: Satellite Pyrometry at Night") separates flares from
 * biomass burning on "temperature AND persistence" — flares hold a
 * stable, persistent retrieved temperature; wildfires are far more
 * variable. This module has no Planck-fit temperature (that needs bands
 * Nightfire consumes that FIRMS doesn't expose) — FRP is the nearest
 * available proxy for radiant-output stability, so `frpCV` (the
 * coefficient of variation of a site's PER-NIGHT mean FRP across its
 * active nights) is added as an opt-in second filter, `maxFrpCV`. A
 * flare's night-to-night FRP should cluster tightly (low CV); a spreading
 * or intensifying wildfire fragment that happens to persist in one grid
 * cell for a few nights should not. This is a candidate refinement, not a
 * validated result — see the gate-1 rerun in `scripts/gasflare_gate1.ts`
 * for whether it actually moves the correlation.
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
  frpCV: number | null;         // coefficient of variation of per-night mean FRP across active nights; null if <2 nights have FRP data
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
  // Opt-in wildfire discriminator (see module header): reject sites whose
  // per-night mean FRP coefficient of variation exceeds this. Undefined
  // (default) applies no CV filter — existing default detector behavior is
  // unchanged unless a caller explicitly opts in. A site with frpCV===null
  // (fewer than 2 nights carried FRP data) is never rejected by this filter
  // alone — insufficient data judges neither way, fail-open rather than
  // fail-closed against thin records.
  maxFrpCV?: number;
}

function coefficientOfVariation(vals: number[]): number | null {
  if (vals.length < 2) return null;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  if (mean === 0) return null;
  const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length;
  return Math.sqrt(variance) / mean;
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
  const maxFrpCV = opts.maxFrpCV;

  const nightDates = new Set<string>();
  for (const d of detections) if (d.daynight === "N") nightDates.add(d.acq_date);
  const nightsInWindow = nightDates.size;

  interface Acc {
    latSum: number; lonSum: number; n: number;
    nights: Set<string>; frpSum: number; frpN: number; dates: string[];
    nightlyFrp: Map<string, { sum: number; n: number }>;
  }
  const sites = new Map<string, Acc>();
  for (const d of detections) {
    if (d.daynight !== "N") continue;
    const key = gridKey(d.lat, d.lon, gridDeg);
    let acc = sites.get(key);
    if (!acc) {
      acc = { latSum: 0, lonSum: 0, n: 0, nights: new Set(), frpSum: 0, frpN: 0, dates: [], nightlyFrp: new Map() };
      sites.set(key, acc);
    }
    acc.latSum += d.lat;
    acc.lonSum += d.lon;
    acc.n++;
    acc.nights.add(d.acq_date);
    acc.dates.push(d.acq_date);
    if (d.frp != null) {
      acc.frpSum += d.frp; acc.frpN++;
      const night = acc.nightlyFrp.get(d.acq_date) ?? { sum: 0, n: 0 };
      night.sum += d.frp; night.n++;
      acc.nightlyFrp.set(d.acq_date, night);
    }
  }

  const out: GasFlareCandidate[] = [];
  for (const [key, acc] of sites) {
    const nightsActive = acc.nights.size;
    const persistence = nightsInWindow > 0 ? nightsActive / nightsInWindow : 0;
    if (nightsActive < minNights || persistence < minPersistence) continue;
    const nightlyMeans = [...acc.nightlyFrp.values()].map((v) => v.sum / v.n);
    const frpCV = coefficientOfVariation(nightlyMeans);
    if (maxFrpCV != null && frpCV != null && frpCV > maxFrpCV) continue;
    const sortedDates = [...acc.dates].sort();
    out.push({
      siteKey: key,
      lat: acc.latSum / acc.n,
      lon: acc.lonSum / acc.n,
      nightsActive,
      nightsInWindow,
      persistence,
      meanFrp: acc.frpN > 0 ? acc.frpSum / acc.frpN : null,
      frpCV,
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
