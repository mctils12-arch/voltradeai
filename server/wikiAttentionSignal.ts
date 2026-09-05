/**
 * wikiAttentionSignal.ts — the SECOND root to reach the "gated-SIGNAL
 * /data surface" milestone gnssIntegritySignal.ts established first
 * (research/open_questions.md's gnss_integrity_adsb entry). GATE 2
 * (SIGNAL) for wikimedia_pageviews_attention PASSED 2026-09-04 for the
 * VOLUME channel specifically (datacore/signal_ladder.json, current_gate
 * 2, status gate2_pass): a pageview attention spike (z>=2.0 vs a trailing
 * baseline) on a small/mid-cap seed ticker's article is followed by
 * elevated forward trading volume, net of a same-day-or-prior-day SEC
 * 8-K (scripts/wikiattention_gate2.py + scripts/wikiattention_gate2_
 * newsfree.py — the pooled, Bonferroni-corrected, news-free-controlled
 * study). This module is the LIVE, public, no-token path: it reads
 * THIS repo's own rolling pageviews archive (server/wikiAttention.ts)
 * to report which seed tickers are CURRENTLY showing that same
 * statistical profile (a live z-score board), packaged with the
 * validated study's own frozen result table and full honesty caveats —
 * PREMIUM EXPERIENCE STANDARD (c): "premium presentation of wrong
 * numbers is fraud with good typography."
 *
 * WHAT THIS IS NOT: it does not re-run the study, does not re-check any
 * live spike against EDGAR for the news-free control (that would need a
 * live per-ticker CIK-mapped 8-K fetch this module does not perform —
 * left as a caveat and a future NEXT, not silently assumed clean), and
 * makes no volatility or directional-price claim (the validated study
 * found no significant realized-volatility elevation at any horizon).
 * GATE 3 (a backtested, cost-net entry/exit rule) has not been
 * attempted — nothing here is a trading signal.
 *
 * ARCHIVE DEPTH CAVEAT: the validated study fetched pageviews directly
 * from Wikimedia's own multi-year history (2 real years); THIS module
 * reads only this repo's own rolling archive, live since 2026-07-05
 * (server/wikiAttention.ts's header) — under the full TRAILING_WINDOW_
 * DAYS for every ticker until the archive itself turns 90 days old.
 * Every ticker's baseline_days/baseline_complete fields say so per-row
 * rather than silently computing a shorter baseline unlabeled.
 */
import { ARTICLES, lookupTickerHistory, type AttentionTickerPoint } from "./wikiAttention";

// Mirrors scripts/wikiattention_gate2.py's own pre-registered constants
// (DEFAULT_Z_THRESHOLD, DEFAULT_TRAILING_WINDOW, MEGA_CAP_TICKERS) —
// duplicated by hand because that script is Python and this module is
// TypeScript with no shared import path between the two runtimes; keep
// both in sync if the study's own split or threshold ever changes.
export const Z_THRESHOLD = 2.0;
export const TRAILING_WINDOW_DAYS = 90;
export const MEGA_CAP_TICKERS = ["NVDA", "AAPL", "TSLA", "AMD"] as const;
// Below this many trailing days, a z-score is too noisy to call a
// "spike" candidate at all — a display-honesty floor, not itself a
// validated statistical threshold.
export const MIN_BASELINE_DAYS = 20;

export function capTier(ticker: string): "small_mid" | "mega" {
  return (MEGA_CAP_TICKERS as readonly string[]).includes(ticker.toUpperCase()) ? "mega" : "small_mid";
}

export interface LatestZScore {
  latest_date: string;
  current_views: number;
  baseline_mean: number;
  baseline_stdev: number;
  baseline_days: number;
  z_score: number | null; // null when baseline_stdev is 0 (no variance to divide by)
}

/**
 * NO-LOOKAHEAD by construction: `series` is ascending by date; the
 * "latest" point is the most recent entry, and the baseline is built
 * ONLY from the strictly-prior entries within `window` — mirrors the
 * validated study's own z-score definition (scripts/wikiattention_
 * gate2.py's zscore_series: "z-scores at day i use only views[i-window:i]").
 */
export function latestZScore(series: AttentionTickerPoint[], window: number = TRAILING_WINDOW_DAYS): LatestZScore | null {
  if (series.length < 2) return null;
  const latest = series[series.length - 1];
  const priorAll = series.slice(0, series.length - 1);
  const prior = priorAll.slice(Math.max(0, priorAll.length - window));
  if (prior.length === 0) return null;
  const mean = prior.reduce((s, p) => s + p.views, 0) / prior.length;
  const variance = prior.reduce((s, p) => s + (p.views - mean) ** 2, 0) / prior.length;
  const stdev = Math.sqrt(variance);
  return {
    latest_date: latest.date,
    current_views: latest.views,
    baseline_mean: mean,
    baseline_stdev: stdev,
    baseline_days: prior.length,
    z_score: stdev > 0 ? (latest.views - mean) / stdev : null,
  };
}

export interface TickerZRow {
  ticker: string;
  article: string;
  cap_tier: "small_mid" | "mega";
  latest_date: string | null;
  current_views: number | null;
  baseline_mean: number | null;
  baseline_days: number;
  baseline_complete: boolean;
  z_score: number | null;
  spike: boolean;
}

/** One row per seed ticker, sorted by z-score descending (nulls last) —
 *  a live "attention board", not a filtered candidate list, so an
 *  absence of spikes today is visibly "23 tickers, none elevated", not
 *  an empty page indistinguishable from a broken feed. */
export function computeTickerRows(baseDir?: string): TickerZRow[] {
  const rows: TickerZRow[] = Object.entries(ARTICLES).map(([ticker, article]) => {
    const series = lookupTickerHistory(ticker, TRAILING_WINDOW_DAYS + 1, baseDir);
    const z = latestZScore(series, TRAILING_WINDOW_DAYS);
    if (!z) {
      return {
        ticker, article, cap_tier: capTier(ticker),
        latest_date: null, current_views: null, baseline_mean: null,
        baseline_days: 0, baseline_complete: false, z_score: null, spike: false,
      };
    }
    const zRounded = z.z_score == null ? null : Math.round(z.z_score * 100) / 100;
    return {
      ticker, article, cap_tier: capTier(ticker),
      latest_date: z.latest_date,
      current_views: z.current_views,
      baseline_mean: Math.round(z.baseline_mean * 10) / 10,
      baseline_days: z.baseline_days,
      baseline_complete: z.baseline_days >= TRAILING_WINDOW_DAYS,
      z_score: zRounded,
      spike: zRounded != null && z.baseline_days >= MIN_BASELINE_DAYS && zRounded >= Z_THRESHOLD,
    };
  });
  rows.sort((a, b) => (b.z_score ?? -Infinity) - (a.z_score ?? -Infinity));
  return rows;
}

export interface ValidatedEffectRow { horizon_days: number; mean_ratio: number; baseline_ratio: number; p_value: number; }

// Frozen result table from the validated news-free-controlled study
// (datacore/signal_ladder.json's wikimedia_pageviews_attention entry,
// "POOLED RESULT ON THE NEWS-FREE SUBSET" table, research/open_questions.md
// 2026-09-04 NEWS-FREE CONTROL entry) — a historical study result, not
// recomputed live. A future re-run should update these constants AND the
// study_date below, never silently drift them apart.
export const VALIDATED_STUDY_DATE = "2026-09-04";
export const VALIDATED_SMALL_MID: ValidatedEffectRow[] = [
  { horizon_days: 1, mean_ratio: 1.279, baseline_ratio: 1.039, p_value: 0.0001 },
  { horizon_days: 3, mean_ratio: 1.226, baseline_ratio: 1.053, p_value: 0.0001 },
  { horizon_days: 5, mean_ratio: 1.195, baseline_ratio: 1.063, p_value: 0.0014 },
];
export const VALIDATED_MEGA: ValidatedEffectRow[] = [
  { horizon_days: 1, mean_ratio: 1.119, baseline_ratio: 0.995, p_value: 0.0083 },
  { horizon_days: 3, mean_ratio: 1.100, baseline_ratio: 0.997, p_value: 0.0050 },
  { horizon_days: 5, mean_ratio: 1.101, baseline_ratio: 0.998, p_value: 0.0023 },
];
// The pre-registered Bonferroni bar the study applied across its own
// 10-cell family (alpha/10) — p-values above are compared against this,
// not the conventional 0.05.
export const BONFERRONI_ALPHA = 0.005;

const METHODOLOGY_NOTE =
  "Live board: each seed ticker's latest archived daily pageview count, z-scored against its own trailing " +
  `(up to ${TRAILING_WINDOW_DAYS}-day) baseline in THIS repo's own rolling archive (no lookahead — the ` +
  "baseline uses only days strictly before the latest one). A row is flagged a 'spike' at z>=2.0 with a " +
  `baseline of at least ${MIN_BASELINE_DAYS} days — the same z-score definition and threshold the validated ` +
  "study below used, applied here live rather than re-running the study itself. The validated effect table is " +
  "a frozen historical result (small/mid-cap forward trading-volume ratio, 23 seed tickers, 2 real years, " +
  "Bonferroni-corrected, net of a same-day-or-prior-day SEC 8-K) — it is NOT re-computed per request and does " +
  "not itself confirm that any ticker flagged as spiking right now is news-free.";

const CAVEATS = [
  "The validated VOLUME effect applies to the POOLED small/mid-cap group over the study's own historical " +
    "sample — not to any single ticker, and not specifically to whichever spike is flagged live above. A " +
    "ticker crossing the z-threshold today matches the statistical PROFILE the study tested, not a fresh " +
    "confirmation of this exact spike.",
  "This live board does NOT re-check today's spikes against EDGAR for a same-day-or-prior-day 8-K — the " +
    "validated study's own news-free control was a historical, offline re-run (scripts/wikiattention_gate2_ " +
    "newsfree.py), not a live per-spike classifier. A spike shown here could be news-driven.",
  "Realized volatility showed NO significant elevation at any horizon in the validated study (filtered or " +
    "unfiltered) — this is a VOLUME signal only, never a volatility or directional-price signal.",
  "Per-ticker sign-consistency weakened under the news-free control (11 of 16 eligible small/mid tickers " +
    "individually positive at the 3-day horizon, not itself significant) versus the strong unfiltered check " +
    "(21 of 23) — the effect is a real pooled-group finding, not yet confirmed to generalize evenly to every " +
    "individual ticker.",
  "GATE 3 (a backtested, cost-net entry/exit rule) has not been attempted for this root — nothing on this " +
    "page is a trading signal or a sizing input.",
  `A ticker's baseline_complete is false until this repo's own archive (live since 2026-07-05) reaches ` +
    `${TRAILING_WINDOW_DAYS} days — its z-score still computes on the shorter window available, labeled honestly.`,
];

const LICENSE_NOTE =
  "Wikimedia pageviews API (CC0/CC-BY, keyless, en.wikipedia all-access/user) — attribution 'Wikimedia " +
  "pageviews API'. SEC EDGAR submissions data (public domain) underlies the validated study's news-free " +
  "control but is not queried live by this endpoint.";

export interface WikiAttentionSignalSummary {
  kind: "signal";
  root_id: "wikimedia_pageviews_attention";
  generated_at: string;
  gate: { current_gate: 2; status: "gate2_pass"; channel: "trading_volume_elevation" };
  z_threshold: number;
  trailing_window_days: number;
  min_baseline_days: number;
  tickers: TickerZRow[];
  spike_count: number;
  validated_effect: {
    study_date: string;
    bonferroni_alpha: number;
    small_mid: ValidatedEffectRow[];
    mega: ValidatedEffectRow[];
  };
  methodology_note: string;
  caveats: string[];
  license: { source: string; note: string };
}

/** Live computation: reads this repo's own wikiattention archive (cheap,
 *  local, bounded to the ~23-ticker seed x <=91 rows each) — no network,
 *  so this can run synchronously per request, unlike gnssIntegritySignal's
 *  larger multi-day AIS scan which needs an eager-poller cache. */
export function computeWikiAttentionSignal(baseDir?: string, now: number = Date.now()): WikiAttentionSignalSummary {
  const tickers = computeTickerRows(baseDir);
  return {
    kind: "signal",
    root_id: "wikimedia_pageviews_attention",
    generated_at: new Date(now).toISOString(),
    gate: { current_gate: 2, status: "gate2_pass", channel: "trading_volume_elevation" },
    z_threshold: Z_THRESHOLD,
    trailing_window_days: TRAILING_WINDOW_DAYS,
    min_baseline_days: MIN_BASELINE_DAYS,
    tickers,
    spike_count: tickers.filter((t) => t.spike).length,
    validated_effect: {
      study_date: VALIDATED_STUDY_DATE,
      bonferroni_alpha: BONFERRONI_ALPHA,
      small_mid: VALIDATED_SMALL_MID,
      mega: VALIDATED_MEGA,
    },
    methodology_note: METHODOLOGY_NOTE,
    caveats: CAVEATS,
    license: { source: "Wikimedia pageviews API (CC0/CC-BY) + SEC EDGAR (public domain, offline study only)", note: LICENSE_NOTE },
  };
}
