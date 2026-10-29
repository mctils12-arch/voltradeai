# Long-term walk-forward of the stock score (2016+, point-in-time)

[RULE-REVIEW] measurement, 2026-10-29. Code: `price_panel.py` (reader),
`walkforward.py` (harness), `server/walkforward.ts` (weekly job + diag probe
`walkforward`). Input: the price panel built nightly by `server/pricePanel.ts`
(research/price_panel_coverage.md). Output: `DATA_DIR/walkforward/latest.json`
(`storage_config.WALKFORWARD_DIR`). It is the Brain page's long-term column for
the stock score and its inputs.

## Prior (stated 2026-10-29, before any real-panel run)

Most price-based inputs' rank IC is near 0 net of costs (|mean IC| < 0.02, none
surviving Bonferroni). Momentum is slightly positive in the least liquid
tercile outside crash regimes. The short-horizon inputs that reward large
recent moves (quick_score, volume, rel_strength) are flat to negative at 5 days
(short-term reversal). The price composite's top-10 does not beat the
random-entry base rate net of costs.

The same text is in every `latest.json` as `prior`. The first real run is
judged against it in research/experiments.md.

## What is measurable long-term, and what is not

Measurable: the parts of the live score that are pure functions of daily
bars. Each is recomputed as the live code computes it, and the tests pin the
mirror to the live function or to the live source lines
(`test_walkforward.py`).

| Input (Brain id) | Live definition |
|---|---|
| `quick_score` | bot_engine quick score: capped absolute daily change, day volume, direction bonus, extreme-move penalty |
| `momentum` | `strategies.momentum.score(12-1 momentum, 1-month return, day share volume)` on deep_score's 400-day bars |
| `mean_reversion` | `strategies.mean_reversion.score(analyze RSI-14, deep 5-day change, analyze 5d/90d volume ratio)`; 50 when RSI is missing |
| `volume` | deep_score volume sub-score: 90/75/60/50/35 by volume ratio |
| `vol_premium` | deep_score EWMA(0.94, 60) vs RV20 bonus: +8 / +5 / 0 |
| `adx` | deep_score Wilder DX on the 400-day bars: +6 / +3 / -4 |
| `obv` | deep_score OBV: +5 / -5 |
| `rel_strength` | deep_score: +6 when analyze's 19-day return beats SPY's by more than 5 points |
| `price_composite` | the rules composite with every non-reconstructable term held at its neutral value (VRP 50, squeeze 50, every adjustment 0), clamped 0..100 |

Not measurable long-term (no point-in-time history exists): VRP (implied
vol), squeeze (short interest, sentiment, Reddit), news, intel and insider
(Form 4, MSPR, analyst), alt data, social, Google Trends (off in live),
fundamentals (today's values only, which would be lookahead), the
contrarian and recommendation boosts, the macro adjustments (VIX regime,
sector momentum, risk on/off), and the ML blend (it has its own honest CV,
research/ml_v3_eval.md). These stay "not measurable long-term". The list is
`not_measurable` in the output, keyed by Brain component id.

## Method

- **Universe (point-in-time).** On each date, a series is in the universe only
  if it has a bar on that date. The price floor is a raw close of at least $5
  (the adjusted close is used only where no raw close was recorded, and those
  rows are counted). The day's estimated raw shares must be at least 500k, and
  the trailing 20-bar dollar volume at least $2.5M. The symbol rule is the
  live scanner's (no ".", at most 5 characters). Delisted names are included
  up to their last bar. Identical (date, close, volume) rows are deduplicated.
- **Series policy** (price_panel.py). Raw (asof=-) lines replace a symbol's
  default lines. A raw segment is split at its corporate-action date.
- **Rebalance.** Weekly, on the last trading day of each ISO week, from 2017
  (one year of warm-up for 12-1 momentum).
- **Outcomes.** Entry is the next bar's open. Exit is the close 5 or 20
  trading days after the decision, net of costs. Each input's top-10 is also
  run under the live exit model (`live_exit_policy`, walked as backtest_v2
  `exit_model="live"` walks it), capped at 60 bars.
- **Costs.** `backtest_v2.liquidity_cost_pct` per side: the volume tier on
  trailing 20-bar raw-share volume, or the half-tick floor at the raw price,
  whichever is larger.
- **Delistings.** A series that ends before the exit exits at its last close
  times (1 + r). r = 0 at a dated merger or rename. r = -30% otherwise
  (Shumway 1997).
- **Benchmarks.** SPY over the same holding. The random-entry base rate is
  the equal-weight universe mean on the same dates and holding (its exact
  expectation). For live exits it is 3 seeded random draws of 10.
- **Statistics.** Per-date Spearman rank IC, then the mean with a Newey-West
  t. Quintile means, and the Q5-Q1 spread net of both legs' costs. Top-10 vs
  random and vs SPY. Hit rates. Splits by regime (regime_util 5-level labels
  on panel SPY + VXX, using the backtest_v2 rules), by SPY vs its 200-day
  average, by year, and by liquidity tercile.
- **Multiple testing.** Bonferroni over every input x horizon IC test (9 x 2 =
  18). Deflated Sharpe (Bailey and Lopez de Prado 2014) over every top-10
  excess-vs-random Sharpe (18 trials), using the cross-trial variance of the
  annualized Sharpes.
- **Verdict per input.** At the primary horizon (5 days), "positive" or
  "negative" when the IC survives Bonferroni. "null" otherwise.
  "insufficient" below 52 dated ICs.

## Bias directions (MEASUREMENT INTEGRITY)

- **Survivorship-reduced, not free.** Delisted names missing from both the
  asset list and the corporate-actions feed are absent. Optimistic.
- **Delisting returns.** Assumed, not observed. The -30% is pessimistic for
  quiet mergers the feed missed, and optimistic for true bankruptcies.
- **Costs.** Mid-points of the live haircut tiers, with no market impact.
  Optimistic for the thinnest names.
- **Live-input deviations that cannot be reproduced.** deep_score uses an
  intraday partial bar. analyze.py uses split-only bars. Here every input is
  computed at the close on dividend-adjusted bars. The effect is small and its
  direction is mixed.
- **Universe floors.** They approximate the live snapshot floors. Dollar
  volume is understated by later dividend factors, so the floor is slightly
  stricter than nominal.
- **Tied scores.** The top-10 among tied discrete scores is a seeded random
  draw. Live breaks ties by scan order, which no history can reproduce.

## Output contract: `latest.json` (schema `walkforward/v1`)

Top level: `schema`, `status` ("ok"), `generated_at`, `as_of` (the panel's),
`code_version` (package.json), `panel {build_id, as_of, completed_at}`,
`window`, `universe`, `cost_model`, `delisting_assumption`, `regime`,
`multiple_testing`, `baselines`, `prior`, `inputs`,
`composite` ("price_composite"), `not_measurable`, `brain_long_term`,
`caveats`, `elapsed_s` and `peak_rss_mb`.

- **`window`:** `first_rebalance` and `last_rebalance`, `n_rebalances`,
  `horizons_days`, `primary_horizon_days`.
- **`universe`:** `n_symbols`, `delisted_symbol_share`, `delisted_row_share`,
  `rows`, `names_per_date`, `duplicates_removed`, `rows_without_raw_close`,
  `delisting_return_applied`, `filters`.
- **`regime`:** `quality`, `vxx_missing_share`.

`inputs.<id>` has `label`, `live_definition`, `measurable`, `verdict`,
`verdict_note`, `horizons` and `live_exit`.

- `horizons` is keyed `"5"` and `"20"`. Each horizon has:
  - `ic`: `mean`, `t_nw`, `p`, `p_bonferroni`, `hit_rate`, `n_dates`.
  - `quintiles`: `mean_net_pct[5]`, `spread_net_long_short_pct`.
  - `top_n`: `mean_net_pct`, `random_entry_net_pct`, `spy_pct`,
    `excess_vs_random_pct`, `excess_vs_spy_pct`, `hit_rate`,
    `random_hit_rate`, `dsr`.
  - `by_regime`, `by_spy_200d`, `by_year`, `by_liquidity`.
- `live_exit` has `n_picks`, `mean_net_pct`, `random_entry_net_pct`,
  `excess_vs_random_pct`, `t_nw_vs_random`, `hit_rate`, `mean_hold_bars` and
  `exit_reasons`.

`brain_long_term.<id>` is already in the Brain page's `LongItem` shape: `id`,
`live: true`, `title`, `source_file`, `date`, `horizon`, `finding`,
`compared_to`, `sign` (= verdict), `variants_tried` and `quotes: []`.

`last_run.json` records every run, including skipped and failed ones: `status`,
`reason` or the summary, and `finished_at`. A failed run never overwrites
`latest.json`.

## How the Brain page consumes it

`brain_evidence.py` was not on main when this shipped. Its branch exists, and
its `NO_LONG_SCORE` reason says this harness was missing. When it lands it
should do the following:

1. Read `os.path.join(storage_config.WALKFORWARD_DIR, "latest.json")`. If the
   file is absent or its `schema` is not `walkforward/v1`, keep today's "not
   yet measured" reason.
2. Put `latest["brain_long_term"][component_id]` at the front of
   `long_term.items` for the components `momentum`, `mean_reversion`,
   `volume`, `vol_premium`, `adx`, `obv`, `rel_strength`, `snapshots` /
   quick score (`quick_score`) and `final_score` (`price_composite`, with its
   `title` saying "price-measurable part"). Use `sign` as the long-term sign.
3. For the components named in `latest["not_measurable"]`, show its text as
   the long-term reason.
4. Surface `generated_at`, `panel.build_id` and `universe.delisted_symbol_share`
   as provenance. The output carries derived statistics only, so no ticker or
   price is displayed (Alpaca licence).

## Operations

- **Schedule.** Saturday or Sunday, 06:00-16:00 ET, once per ISO week. The
  job runs only if a complete panel build exists and the last result is not
  already for that build and code version, or is older than 27 days.
  Otherwise the week gets a "skipped" heartbeat beat. Job id:
  `walkforward_weekly`.
- **Lane.** Its own single-flight long-job lane: not the 90 s Python mutex,
  not the daemon. The process group is killed after 3 h. It runs with
  niceness 19 and one thread per BLAS. Python aborts above 900 MB peak RSS.
- **Flag.** `VOLTRADE_WALKFORWARD` (default on; `0` or `off` disables).
- **Status.** `/api/diag/walkforward`.
