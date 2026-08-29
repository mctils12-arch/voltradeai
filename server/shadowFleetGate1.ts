/**
 * shadowFleetGate1.ts — GATE 1 (DATA) statistical test for the shadow-fleet
 * signal (research/open_questions.md "SHADOW-FLEET SIGNAL"): "Gate passes
 * if our gap/loiter detections are significantly ENRICHED for
 * reference-list vessels vs a size-matched random tanker sample from the
 * same archive window (odds ratio with CI, not eyeballing)."
 *
 * This is a case-control design: cases = MMSIs our own detectors
 * (server/shadowFleet.ts detectGapEvents/detectLoitering) flag as
 * gap/loiter candidates; controls = a size-matched random sample of the
 * same tanker population, drawn WITHOUT the cases. Both cohorts are
 * checked for membership on the OFAC reference list
 * (server/shadowFleetReference.ts) — "significantly enriched" means the
 * odds of reference-list membership are higher among cases than controls,
 * beyond what a random draw from the same population would produce.
 *
 * PASS criterion (pre-stated, matching the open_questions.md plan's own
 * "odds ratio with CI, not eyeballing"): the two-sided 95% Woolf log-odds
 * CI for the odds ratio lies entirely above 1. A CI that straddles or sits
 * below 1 is FAIL/INCONCLUSIVE, never reported as a pass on the point
 * estimate alone (REASONING STANDARD #4 — the same discipline
 * hazard_rate_probe.py's bootstrap range already applies to this codebase's
 * other small-sample tests).
 *
 * Pure statistics module — no fs/network/archive access. Consumes MMSI
 * sets built elsewhere: in production, from server/shadowFleet.ts's real
 * archive reads; in tests, from synthetic fixtures. Terrestrial-AIS
 * coverage-loss ambiguity (a "gap" can be a real dark period or just a
 * receiver blind spot) is controlled by the case-control DESIGN itself,
 * not by anything in this module — both cohorts are drawn from the same
 * archive window and suffer identical coverage loss, so it cancels out of
 * the comparison rather than needing its own correction term here.
 */

export interface Contingency { a: number; b: number; c: number; d: number }

/** 2x2 table over `universe` (every MMSI eligible to be counted — the
 *  cases plus the size-matched control sample): rows = in `candidates`
 *  (gap/loiter-flagged) or not; columns = in `reference` (OFAC SDN list)
 *  or not. Duplicate MMSIs in `universe` are counted once — a caller
 *  passing a raw archive scan (which can repeat an MMSI across windows)
 *  should not silently inflate one cell. */
export function contingencyCounts(
  candidates: ReadonlySet<string>,
  reference: ReadonlySet<string>,
  universe: readonly string[],
): Contingency {
  let a = 0, b = 0, c = 0, d = 0;
  const seen = new Set<string>();
  for (const mmsi of universe) {
    if (seen.has(mmsi)) continue;
    seen.add(mmsi);
    const inCandidates = candidates.has(mmsi);
    const inReference = reference.has(mmsi);
    if (inCandidates && inReference) a++;
    else if (inCandidates && !inReference) b++;
    else if (!inCandidates && inReference) c++;
    else d++;
  }
  return { a, b, c, d };
}

/** Haldane-Anscombe correction: add 0.5 to every cell when ANY cell is
 *  zero. A small reference-list overlap makes a zero cell likely (this
 *  reference list has hundreds, not thousands, of MMSIs against a whole
 *  tanker population), and the uncorrected ratio is undefined/infinite the
 *  moment one cell is empty — the standard small-sample fix (Haldane 1940,
 *  Anscombe 1956), not a bespoke one. */
function correctedCells(t: Contingency): { a: number; b: number; c: number; d: number } {
  const zero = t.a === 0 || t.b === 0 || t.c === 0 || t.d === 0;
  const k = zero ? 0.5 : 0;
  return { a: t.a + k, b: t.b + k, c: t.c + k, d: t.d + k };
}

export function oddsRatio(t: Contingency): number {
  const { a, b, c, d } = correctedCells(t);
  return (a * d) / (b * c);
}

/** Woolf's method: CI for the log-odds-ratio, then exponentiated back.
 *  z=1.96 is the standard two-sided 95% critical value. */
export function oddsRatioCI(t: Contingency, z = 1.96): { lower: number; upper: number } {
  const { a, b, c, d } = correctedCells(t);
  const logOr = Math.log((a * d) / (b * c));
  const se = Math.sqrt(1 / a + 1 / b + 1 / c + 1 / d);
  return { lower: Math.exp(logOr - z * se), upper: Math.exp(logOr + z * se) };
}

/** Deterministic xorshift32 PRNG — no crypto or external RNG dependency,
 *  reproducible given the same seed so a gate-1 run can be independently
 *  re-verified from a logged seed (same reproducibility standard
 *  hazard_rate_probe.py's bootstrap already holds this codebase to). */
function xorshift32(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 0xffffffff;
  };
}

/** Fisher-Yates sample of `size` items from `pool`, seeded. */
export function seededSample<T>(pool: readonly T[], size: number, seed: number): T[] {
  const next = xorshift32(seed);
  const arr = pool.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.slice(0, Math.min(size, arr.length));
}

/** Builds the case-control universe: every candidate (case) plus a
 *  size-matched random sample of the tanker pool EXCLUDING the candidates
 *  (controls never overlap cases by construction). If the pool has fewer
 *  eligible controls than candidates, every eligible control is used (a
 *  smaller-than-requested control group, never a silent duplicate draw). */
export function buildCaseControlUniverse(
  candidates: ReadonlySet<string>,
  tankerPool: readonly string[],
  seed: number,
): string[] {
  const cases = Array.from(candidates);
  const controlPool = tankerPool.filter((mmsi) => !candidates.has(mmsi));
  const controls = seededSample(controlPool, cases.length, seed);
  return [...cases, ...controls];
}

const MIN_REFERENCE_HITS = 5; // REASONING STANDARD #4 small-n floor, same value hazard_rate_probe.py uses

export interface Gate1Verdict {
  n_universe: number;
  n_candidates: number;
  n_controls: number;
  n_reference_hits: number;
  contingency: Contingency;
  odds_ratio: number;
  ci_95: { lower: number; upper: number };
  enriched: boolean; // PASS: CI lies entirely above 1
  insufficient_n: boolean; // fewer than MIN_REFERENCE_HITS reference-list hits in the whole universe
  seed: number;
}

/** Runs the full gate-1 case-control enrichment test and returns the
 *  verdict object described at the top of this file. `candidates` are the
 *  MMSIs our own detectors flagged in the archive window; `tankerPool` is
 *  every tanker MMSI seen in that same window (the population controls are
 *  drawn from); `reference` is the OFAC SDN vessel MMSI set. */
export function evaluateEnrichment(
  candidates: ReadonlySet<string>,
  reference: ReadonlySet<string>,
  tankerPool: readonly string[],
  seed: number,
  z = 1.96,
): Gate1Verdict {
  const universe = buildCaseControlUniverse(candidates, tankerPool, seed);
  const t = contingencyCounts(candidates, reference, universe);
  const ci = oddsRatioCI(t, z);
  return {
    n_universe: universe.length,
    n_candidates: t.a + t.b,
    n_controls: t.c + t.d,
    n_reference_hits: t.a + t.c,
    contingency: t,
    odds_ratio: oddsRatio(t),
    ci_95: ci,
    enriched: ci.lower > 1,
    insufficient_n: t.a + t.c < MIN_REFERENCE_HITS,
    seed,
  };
}
