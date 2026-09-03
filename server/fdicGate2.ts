/**
 * fdicGate2.ts — GATE 2 (SIGNAL) statistical test for fdic_bank_failures
 * (server/fdicBanks.ts's own docstring: "gate 2 = failure events vs
 * forward KRE / regional-bank returns").
 *
 * Event-study design, not a bare before/after read: does KRE's forward
 * return following a REAL FDIC failure date differ from what entering on
 * an ordinary (non-event) date, same horizon, same historical window,
 * would have returned? REASONING STANDARD #3 ("what would random entry
 * with the same holding period ... have returned? Alpha is the excess
 * over that") is the base rate this module tests against, via a seeded
 * bootstrap of the null (non-event) distribution rather than eyeballing a
 * point estimate (REASONING STANDARD #4 — same discipline
 * shadowFleetGate1.ts's odds-ratio CI and hazard_rate_probe.py's
 * bootstrap CV range already hold this codebase to).
 *
 * Pure statistics module — no fs/network access. Consumes a daily bar
 * series and an event-date list built elsewhere (production: real KRE
 * bars from Alpaca + real FDIC failure dates; tests: synthetic fixtures).
 */

export interface Bar { date: string; close: number } // "YYYY-MM-DD", ascending, trading days only

const MIN_EVENTS = 5; // same n>=5 small-sample floor as shadowFleetGate1.ts's MIN_REFERENCE_HITS / hazard_rate_probe.py's MIN_GAPS_FOR_STATS
const DEFAULT_N_BOOT = 2000; // same bootstrap iteration count hazard_rate_probe.py uses

function mean(xs: readonly number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

/** index of the first bar with date >= target, or -1 if target is after every bar in the series. */
function firstIndexOnOrAfter(bars: readonly Bar[], date: string): number {
  for (let i = 0; i < bars.length; i++) if (bars[i].date >= date) return i;
  return -1;
}

/** Forward `horizonDays` TRADING-DAY (bar-index, not calendar-day) return
 *  entering at the first bar on/after `entryDate`. null when there isn't
 *  enough future bar depth yet (a recent event near the series end) or the
 *  entry date is entirely outside the series. */
export function forwardReturn(bars: readonly Bar[], entryDate: string, horizonDays: number): number | null {
  const i = firstIndexOnOrAfter(bars, entryDate);
  if (i < 0 || i + horizonDays >= bars.length) return null;
  const entry = bars[i].close;
  if (!(entry > 0)) return null;
  return bars[i + horizonDays].close / entry - 1;
}

/** Deterministic xorshift32 PRNG — no crypto/external RNG dependency, same
 *  reproducibility standard as shadowFleetGate1.ts's own copy (kept local
 *  rather than shared, matching that module's own "no fs/network access"
 *  pure-module scope). */
function xorshift32(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 0xffffffff;
  };
}

export interface EventStudyVerdict {
  horizon_days: number;
  n_events_total: number;      // deduped event dates passed in
  n_events_valid: number;      // events with enough forward bar depth to score
  event_mean_return: number | null;
  base_rate_mean_return: number | null;   // mean forward return over every eligible non-event entry point
  base_rate_n: number;
  bootstrap_ci_95: { lower: number; upper: number } | null; // 95% band of the null (bootstrap) mean-return distribution at n=n_events_valid
  two_sided_p: number | null; // fraction of bootstrap draws at least as far from the base rate as the real event mean
  signal_detected: boolean;   // PASS: event_mean_return falls outside bootstrap_ci_95 (equivalently two_sided_p < 0.05)
  insufficient_n: boolean;
  seed: number;
}

/** Runs the event study. `bars` must be sorted ascending by date with no
 *  duplicate dates. `eventDates` are deduped and sorted internally — a
 *  caller passing a raw multi-fetch list should not silently double-count
 *  a repeated event. */
export function eventStudy(
  bars: readonly Bar[],
  eventDates: readonly string[],
  horizonDays: number,
  seed: number,
  nBoot: number = DEFAULT_N_BOOT,
): EventStudyVerdict {
  const dedupedEvents = Array.from(new Set(eventDates)).sort();
  const eventReturns: number[] = [];
  const eventEntryIdx: number[] = [];
  for (const d of dedupedEvents) {
    const i = firstIndexOnOrAfter(bars, d);
    if (i < 0 || i + horizonDays >= bars.length) continue;
    eventReturns.push(bars[i + horizonDays].close / bars[i].close - 1);
    eventEntryIdx.push(i);
  }
  const nEventsValid = eventReturns.length;

  if (nEventsValid < MIN_EVENTS) {
    return {
      horizon_days: horizonDays,
      n_events_total: dedupedEvents.length,
      n_events_valid: nEventsValid,
      event_mean_return: nEventsValid ? mean(eventReturns) : null,
      base_rate_mean_return: null,
      base_rate_n: 0,
      bootstrap_ci_95: null,
      two_sided_p: null,
      signal_detected: false,
      insufficient_n: true,
      seed,
    };
  }

  // Eligible base-rate entry points: every index with a valid forward
  // return, EXCLUDING any index within `horizonDays` of an event's own
  // entry index in either direction — an un-excluded overlap would let an
  // event's own price move leak into the "random" pool it's being tested
  // against, biasing the test toward finding no effect (the same
  // "identical coverage loss cancels out" design shadowFleetGate1.ts's
  // case-control comparison relies on, applied here to date overlap
  // instead of AIS coverage gaps).
  const excluded = new Set<number>();
  for (const idx of eventEntryIdx) {
    for (let k = idx - horizonDays; k <= idx + horizonDays; k++) excluded.add(k);
  }
  const eligibleReturns: number[] = [];
  for (let i = 0; i + horizonDays < bars.length; i++) {
    if (excluded.has(i)) continue;
    if (!(bars[i].close > 0)) continue;
    eligibleReturns.push(bars[i + horizonDays].close / bars[i].close - 1);
  }
  const baseRateMean = mean(eligibleReturns);
  const eventMean = mean(eventReturns);

  const rng = xorshift32(seed);
  const bootMeans: number[] = [];
  for (let b = 0; b < nBoot; b++) {
    let s = 0;
    for (let k = 0; k < nEventsValid; k++) s += eligibleReturns[Math.floor(rng() * eligibleReturns.length)];
    bootMeans.push(s / nEventsValid);
  }
  bootMeans.sort((a, z) => a - z);
  const lower = bootMeans[Math.floor(0.025 * nBoot)];
  const upper = bootMeans[Math.min(nBoot - 1, Math.floor(0.975 * nBoot))];
  const observedGap = Math.abs(eventMean - baseRateMean);
  const twoSidedP = bootMeans.filter((m) => Math.abs(m - baseRateMean) >= observedGap).length / nBoot;

  return {
    horizon_days: horizonDays,
    n_events_total: dedupedEvents.length,
    n_events_valid: nEventsValid,
    event_mean_return: eventMean,
    base_rate_mean_return: baseRateMean,
    base_rate_n: eligibleReturns.length,
    bootstrap_ci_95: { lower, upper },
    two_sided_p: twoSidedP,
    signal_detected: eventMean < lower || eventMean > upper,
    insufficient_n: false,
    seed,
  };
}
