/**
 * gemMethaneAssetRates.ts — gate-2(b) of the GEM METHANE-PLUME ×
 * EXTRACTION-REGISTRY PROXIMITY hypothesis (research/open_questions.md):
 * groups the gate-2(a) proximity join (server/gemMethaneProximity.ts) by
 * matched asset and computes a REPEAT-DETECTION count/rate per asset — a
 * single detection is noise (per the hypothesis's own stated ladder path),
 * so this is the smallest next slice toward gate 2(c) (same-universe base
 * rate) and 2(d) (disclosed-emissions match), neither built here.
 *
 * STILL NOT A SIGNAL: a higher detection count/rate at a catalogued asset
 * is a descriptive fact about repeat satellite pass-over detections — not
 * yet compared against a same-universe base rate or an operator's own
 * disclosed methane intensity. No trading or scoring code reads this.
 *
 * Ambiguous matches (gemMethaneProximity's ambiguousMatch: true — a second
 * asset sits nearly as close as the nearest) are EXCLUDED from every
 * asset's count: attributing an ambiguous plume to one specific asset
 * would bias that asset's rate on what is really a coin-flip geometric
 * call, not a real repeat-detection.
 */
import type { PlumeWithAsset } from "./gemMethaneProximity";
import { cachedGemMethaneProximity } from "./gemMethaneProximity";
import type { GemAsset } from "./gemMethaneAssets";

const MS_PER_DAY = 86_400_000;
const DAYS_PER_YEAR = 365.25;
// A short observed span multiplies up into a wild annualized rate (e.g. 2
// detections a day apart -> ~730/yr extrapolated from one data point's
// worth of information) — exactly the "single detection is noise" problem
// restated. Below this many days of observed span, detectionsPerYear stays
// null (count is still reported) rather than publishing an extrapolation
// nobody should trust. Chosen conservatively, not tuned against outcomes.
export const MIN_SPAN_DAYS_FOR_RATE = 30;

export interface AssetDetectionRate {
  assetKind: GemAsset["kind"];
  assetId: string;
  name: string | null;
  operator: string | null;
  owner: string | null;
  parent: string | null;
  detectionCount: number;
  firstObservedAt: string | null;
  lastObservedAt: string | null;
  /** days between the earliest and latest dated detection; null if <2 dated detections */
  spanDays: number | null;
  /** count / (spanDays / 365.25); null for single-detection assets and for
   *  any asset observed over less than MIN_SPAN_DAYS_FOR_RATE — both cases
   *  are "not enough history to annualize", not a zero or a real rate */
  detectionsPerYear: number | null;
  meanEmissionsKgHr: number | null;
}

// GEM's "Observation Date" column ships UTC offsets as a BARE 2-digit
// offset with no minutes/colon (e.g. "2020-11-09T18:47:16+00") — every one
// of the 3,473 dated plumes in the release uses this exact suffix, and it
// is not valid ISO 8601: V8's Date.parse silently returns NaN on it rather
// than throwing, which (before this fix) meant every observedAt failed to
// parse and every asset's spanDays/detectionsPerYear stayed null. Anchored
// to a full "T HH:MM:SS[+-]OO" time component so a plain "YYYY-MM-DD" date
// (whose trailing "-09" would otherwise false-match a bare offset) is never
// touched.
const BARE_UTC_OFFSET_RE = /^(.*T\d{2}:\d{2}:\d{2}(?:\.\d+)?)([+-]\d{2})$/;

function parseDateMs(iso: string | null): number | null {
  if (!iso) return null;
  const m = iso.match(BARE_UTC_OFFSET_RE);
  const normalized = m ? `${m[1]}${m[2]}:00` : iso;
  const t = Date.parse(normalized);
  return Number.isFinite(t) ? t : null;
}

/** Groups matched, unambiguous plumes by nearest asset and computes a
 *  repeat-detection count/rate. Pure function — no I/O, fully unit-testable. */
export function computeAssetDetectionRates(plumes: PlumeWithAsset[]): AssetDetectionRate[] {
  const groups = new Map<string, PlumeWithAsset[]>();
  for (const p of plumes) {
    if (!p.nearestAsset || p.ambiguousMatch) continue;
    const key = `${p.nearestAsset.kind}:${p.nearestAsset.id}`;
    const bucket = groups.get(key);
    if (bucket) bucket.push(p);
    else groups.set(key, [p]);
  }

  const out: AssetDetectionRate[] = [];
  for (const bucket of Array.from(groups.values())) {
    const a = bucket[0].nearestAsset!;
    const dateMsList = bucket
      .map((p) => parseDateMs(p.observedAt))
      .filter((d): d is number => d !== null)
      .sort((x, y) => x - y);
    const firstMs = dateMsList.length ? dateMsList[0] : null;
    const lastMs = dateMsList.length ? dateMsList[dateMsList.length - 1] : null;
    // spanDays needs >=2 DATED detections to mean anything — a single dated
    // detection has first===last, which is "no span", not "span of zero".
    const spanDays = dateMsList.length >= 2 ? (lastMs! - firstMs!) / MS_PER_DAY : null;
    const detectionsPerYear =
      bucket.length >= 2 && spanDays !== null && spanDays >= MIN_SPAN_DAYS_FOR_RATE
        ? bucket.length / (spanDays / DAYS_PER_YEAR)
        : null;
    const emissions = bucket.map((p) => p.emissionsKgHr).filter((v): v is number => v !== null);
    const meanEmissionsKgHr = emissions.length
      ? emissions.reduce((s, v) => s + v, 0) / emissions.length
      : null;

    out.push({
      assetKind: a.kind,
      assetId: a.id,
      name: a.name,
      operator: a.operator,
      owner: a.owner,
      parent: a.parent,
      detectionCount: bucket.length,
      firstObservedAt: firstMs !== null ? new Date(firstMs).toISOString() : null,
      lastObservedAt: lastMs !== null ? new Date(lastMs).toISOString() : null,
      spanDays,
      detectionsPerYear,
      meanEmissionsKgHr,
    });
  }
  return out;
}

export type AssetRateSort = "count" | "rate";

/** Sorts a copy (never mutates input) — "count" (default) breaks ties by
 *  nothing further; "rate" falls single-detection (null-rate) assets to the
 *  bottom rather than treating null as zero, then breaks ties by count. */
export function sortAssetDetectionRates(
  rates: AssetDetectionRate[],
  by: AssetRateSort,
): AssetDetectionRate[] {
  const copy = [...rates];
  if (by === "rate") {
    copy.sort((x, y) => {
      if (x.detectionsPerYear === null && y.detectionsPerYear === null) return y.detectionCount - x.detectionCount;
      if (x.detectionsPerYear === null) return 1;
      if (y.detectionsPerYear === null) return -1;
      return y.detectionsPerYear - x.detectionsPerYear || y.detectionCount - x.detectionCount;
    });
  } else {
    copy.sort((x, y) => y.detectionCount - x.detectionCount);
  }
  return copy;
}

export interface AssetDetectionRatesResult {
  rates: AssetDetectionRate[];
  excludedAmbiguousCount: number;
  singleDetectionAssetCount: number;
}

// In-memory cache — same rationale as gemMethaneProximity.ts's cached join:
// derived from static reference data, computed once per process lifetime.
let cached: AssetDetectionRatesResult | null | undefined;

export function cachedGemMethaneAssetRates(): AssetDetectionRatesResult | null {
  if (cached !== undefined) return cached;
  const hit = cachedGemMethaneProximity();
  if (!hit) {
    cached = null;
    return cached;
  }
  const rates = computeAssetDetectionRates(hit.plumes);
  cached = {
    rates,
    excludedAmbiguousCount: hit.plumes.filter((p) => p.ambiguousMatch).length,
    singleDetectionAssetCount: rates.filter((r) => r.detectionCount === 1).length,
  };
  return cached;
}

/** Test-only: clears the module cache so a test can inject different upstream state. */
export function _resetGemMethaneAssetRatesCacheForTests(): void {
  cached = undefined;
}
