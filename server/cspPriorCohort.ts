// cspPriorCohort.ts — the PRE-REGISTERED re-measure of the csp prior
// ([RULE-REVIEW] measurement, 2026-11-12). HONESTY METRIC: the sleeve ledger's
// 316 closed short-option lots won 20.6 % and returned −19.2 % of the premium
// per lot, against the csp prior the sizers and the tax filter price every
// premium-selling trade with (position_sizing.py: 71.6 % win rate). Most of
// those lots predate the exit fixes of 2026-11-05/06, so
// research/open_questions.md ("2026-11-06 — findings from the refocus build
// night", first item) registered the re-measure BEFORE a new lot existed:
//
//   only short-option lots OPENED on or after 2026-11-06; at n ≥ 30 closed
//   lots, if the Wilson 95 % interval of their win rate excludes 71.6 %, the
//   csp prior is rebuilt from live lots in its own [RULE-REVIEW] PR.
//
// This module computes that cohort, its n and the interval, and says which
// side of the rule the numbers are on. It changes no prior, size, threshold
// or order: the rebuild is a later, separate step. Pure. bot.ts serves the
// block inside GET /api/bot/tax (server/taxAware.ts `cspCohort`), so the
// `tax` diag probe carries it too.
//
// ONE DEFINITION. The lots are the sleeve ledger's RealizedEvents, the list
// /api/bot/tax's liveExpectancy and the `lots` diag probe are computed from.
// Which lots are OPTION lots, which are priced, what a win is (pnl > 0) and
// the return on the premium (pnl / basis, the basis of a short lot being the
// credit received) are edgeLedger's: the cohort's row IS edgeLedger(...)'s
// `options` row over the selected lots, never a second formula. This module
// adds two selections: SHORT (the closing fill bought the lot back: closeDir
// +1, the `lots` probe's side "short") and OPENED from the cutoff (openedAt,
// the fill that took the position from flat). With no cutoff the same
// selection is the 316 lots quoted above (65 of them won).
//
// WHAT A LOT IS. One RealizedEvent per closing fill. A position closed in two
// fills is two lots; the two short legs of an iron condor are two lots (its
// long wings are long lots and are not in this cohort). The prior is per
// backtest trade. The 20.6 % was counted per lot, so the re-measure is too.
//
// BIASES (MEASUREMENT INTEGRITY; research/experiments.md 2026-11-12):
//   - only CLOSED lots count, so an early read leans toward whichever
//     outcome closes first. Before the cutoff the losers closed first, which
//     pulls an early read down; exits that take profit before a stop fires
//     pull it up.
//   - a lot is closed by a FILL. One that ends by expiry or assignment has
//     no closing fill (Alpaca books those as non-trade activities), so the
//     walk never emits it: a put left to expire worthless is a win this
//     count misses. None had happened by 2026-11-12. /api/bot/sleeves shows
//     them: the options sleeve's unpricedCloses, and the OPEXP / OPASN rows
//     of nonTrade.types.
//   - the interval assumes independent lots. Fills of one position, legs of
//     one structure and lots closed in the same market move are not, so the
//     true interval is wider than the one shown.
//   - the verdict is recomputed on every read from n = 30 on; looking again
//     at every new lot excludes a true prior more often than 5 % of the time.
import type { RealizedEvent } from "./sleeveLedger";
import { edgeLedger } from "./edgeLedger";

/** PRE-REGISTERED 2026-11-06 (research/open_questions.md, "findings from the
 *  refocus build night", [HONESTY METRIC, top priority]): only lots OPENED on
 *  or after this date are re-measured, because the older lots predate that
 *  night's exit fixes (#1635, #1642, #1644, #1645, #1655). Changing it after
 *  seeing lots is the fishing the registration exists to prevent. Read as a
 *  UTC date, like every day in the sleeve ledger. No option fills happen
 *  between the New York close and midnight, so reading the date in UTC or in
 *  New York time selects the same lots. */
export const CSP_COHORT_OPENED_FROM = "2026-11-06";
export const CSP_COHORT_OPENED_FROM_MS = Date.parse(`${CSP_COHORT_OPENED_FROM}T00:00:00Z`);
/** PRE-REGISTERED with the date: the rule is read from this many closed lots on. */
export const CSP_COHORT_MIN_LOTS = 30;
/** The csp prior's win rate: position_sizing.py _get_historical_stats(), the
 *  default by_strategy["csp_options"]["win_rate"] (backtest NEW_CSP_ONLY, 1,226
 *  trades). MUST mirror it. Python owns the value, and the only thing that
 *  carries it to Node is a priced option row of the tax decision log, which
 *  may not exist; cspPriorCohort.test.ts fails when position_sizing.py
 *  changes it. */
export const CSP_PRIOR_WIN_RATE = 0.716;
/** Two-sided 95 % normal quantile. */
export const WILSON_Z_95 = 1.959964;

/** Wilson score interval (no continuity correction) of a binomial proportion,
 *  as fractions [lo, hi] within [0, 1]. null when there is nothing to
 *  estimate (n = 0) or the counts are not counts — never NaN. */
export function wilsonInterval(wins: number, n: number): [number, number] | null {
  if (!Number.isFinite(wins) || !Number.isFinite(n) || n <= 0 || wins < 0 || wins > n) return null;
  const z2 = WILSON_Z_95 * WILSON_Z_95;
  const p = wins / n;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (WILSON_Z_95 * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  // with no win the lower bound is exactly 0 and with no loss the upper bound exactly 1; float error must not show as −1e-17
  return [wins === 0 ? 0 : Math.max(0, center - half), wins === n ? 1 : Math.min(1, center + half)];
}

export interface CspPriorCohort {
  /** the pre-registered cutoff (ISO, UTC): only lots OPENED at or after it count */
  since: string;
  /** closed short-option lots opened since then */
  n: number;
  wins: number;
  /** percent of them with pnl > 0 (0–100); null without lots */
  winRatePct: number | null;
  /** Wilson 95 % score interval of that win rate, percent [lo, hi]; null without lots */
  wilson95Pct: [number, number] | null;
  /** mean pnl / premium received per lot, percent; null without lots */
  meanReturnPct: number | null;
  /** the csp prior's win rate the interval is compared with, percent */
  priorWinRatePct: number;
  /** closed lots the rule waits for */
  needed: number;
  /** n >= needed */
  enough: boolean;
  /** enough AND the interval does not contain the prior: the pre-registered trigger to rebuild the prior */
  priorExcluded: boolean;
  /** one line of plain English */
  verdict: string;
  /** short-option lots with no known open time (opened before the fetched history): cannot be placed, never guessed in */
  undated: number;
  lastCloseAt: string | null;
  /** when the sleeve ledger these lots come from was fetched (ISO); null when the caller does not say */
  fetchedAt: string | null;
  basis: string;
}

const r2 = (x: number) => Math.round(x * 100) / 100;
const pct = (x: number) => `${r2(x)}%`;
// edgeLedger sorts a lot into the options sleeve before it looks at the floor set, and the cohort has no unpriced closes to add
const NO_FLOOR: ReadonlySet<string> = new Set<string>();
const NO_UNPRICED: ReadonlyMap<string, number> = new Map<string, number>();
const optionsRow = (lots: readonly RealizedEvent[]) => edgeLedger(lots, NO_UNPRICED, NO_FLOOR).options;

/** Pure: the sleeve ledger's closed lots (any order) → the pre-registered
 *  cohort and its verdict. `fetchedAtMs`: when that ledger was fetched. */
export function cspPriorCohort(events: readonly RealizedEvent[], fetchedAtMs?: number | null): CspPriorCohort {
  const short = events.filter((e) => e.closeDir === 1);
  const row = optionsRow(short.filter((e) => e.openedAt !== null && e.openedAt >= CSP_COHORT_OPENED_FROM_MS));
  const undated = optionsRow(short.filter((e) => e.openedAt === null)).n;
  const { n, wins } = row;
  const ci = wilsonInterval(wins, n);
  const enough = n >= CSP_COHORT_MIN_LOTS;
  const priorExcluded = enough && ci !== null && (CSP_PRIOR_WIN_RATE < ci[0] || CSP_PRIOR_WIN_RATE > ci[1]);
  const prior = pct(CSP_PRIOR_WIN_RATE * 100);
  let verdict = `${n} of ${CSP_COHORT_MIN_LOTS} needed lots closed; too few to judge`;
  if (enough && ci !== null) {
    const shown = `the live 95% interval [${pct(ci[0] * 100)}, ${pct(ci[1] * 100)}]`;
    const won = `${wins} of ${n} lots won`;
    verdict = priorExcluded
      ? `prior ${prior} is outside ${shown} (${CSP_PRIOR_WIN_RATE > ci[1] ? "above" : "below"} it; ${won}): rebuild the csp prior (RULE-REVIEW)`
      : `prior ${prior} is inside ${shown} (${won}): the pre-registered rebuild is not triggered`;
  }
  return {
    since: new Date(CSP_COHORT_OPENED_FROM_MS).toISOString(),
    n, wins,
    winRatePct: row.winRate,
    wilson95Pct: ci ? [r2(ci[0] * 100), r2(ci[1] * 100)] : null,
    meanReturnPct: row.expectancyPct,
    priorWinRatePct: r2(CSP_PRIOR_WIN_RATE * 100),
    needed: CSP_COHORT_MIN_LOTS,
    enough, priorExcluded, verdict, undated,
    lastCloseAt: row.lastCloseAt,
    fetchedAt: typeof fetchedAtMs === "number" && Number.isFinite(fetchedAtMs) ? new Date(fetchedAtMs).toISOString() : null,
    basis: "closed short-option lots opened on or after the cutoff, per closed lot at the broker (server/edgeLedger.ts); a lot still open, or ended by expiry or assignment with no closing fill, is not counted; the prior is per backtest trade (position_sizing.py csp_options)",
  };
}
