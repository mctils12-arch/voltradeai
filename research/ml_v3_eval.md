# ML v3 candidate — honest out-of-sample evaluation (2026-10-27)

Human Phase 4 "Fix the model". The v3 candidate is **SHADOW ONLY**: the live
score blend still uses `ml_model_v2` (`voltrade_ml_v2.pkl`); v3 writes
`DATA_DIR/ml_v3/voltrade_ml_v3_candidate.pkl` and its probability is logged on
every shadow record (`ml.v3_win_probability`), never blended. Promotion is a
separate, later decision.

## Verdict

**No out-of-sample skill.** On 55,775 out-of-time test rows the candidate's
AUC is 0.499 (95% date-block bootstrap CI 0.479–0.516) and its log-loss
(0.6949) is *worse* than a constant forecast of the base rate (0.6928). Its
top predicted quintile returned **less** than its bottom quintile
(-0.13 pts net per trade, CI -0.79..+0.55). Early stopping on the inner
validation slice stops after 1–29 trees in every fold — the training window
itself finds nothing that generalises a few weeks forward.

The v2 procedure on the same kind of data *reports* AUC 0.658 / 61.9%
accuracy. That number is the CV leak, not skill: with the leak removed the
same v2 dataset scores AUC 0.515 and log-loss 0.690 vs 0.684 for the base
rate. The current production `last_accuracy: 58` should be read the same way.

**Do not promote v3. Do not raise the v2 ML blend weight.** The evidence
supports the opposite question: whether the ML blend (0.28–0.40 weight) should
be in the live score at all — a rule review with counterfactual evidence,
filed for a human decision (see "Needs a decision" below).

## What was run, and on what

| | |
|---|---|
| Data | Yahoo daily bars (split/dividend adjusted, `backtest_v2._yahoo_bars`), 97 stocks + SPY/VXX/TLT/HYG, 2022-12-27 → 2026-10-26 (today's partial session stripped). **Not** Alpaca and **not** production shadow records — neither is reachable from the session sandbox (no Alpaca keys; shadow log lives on the Railway volume). |
| Universe | Fixed list in `ml_model_v3.UNIVERSE` (liquid names across sectors + the bot's core list), chosen without looking at outcomes. **Survivorship-biased** (all listed today). |
| Rows | 66,833 decisions (2023-12-28 → 2026-09-28; the first 252 bars are feature warm-up, the last 20 bars have no complete label window) |
| Features | 24, `ml_features_v3.V3_FEATURE_COLS` — one PIT definition for train and serve |
| Label | `exit_aligned_v1_next_open`: entry at next session's open, walked under the live exit policy (`live_exit_policy`, mirror of `bot.ts checkPositionOnTick`), 20-bar cap, label = net return > 0 after 0.10% round trip |
| CV | `ml_cv.honest_cv`: 5 date-block walk-forward folds, 21-trading-date embargo (> 20-bar label horizon) + purge on label end date, early stopping on an inner slice of each training window |
| Model | LightGBM, `ml_cv.LGBM_V3_PARAMS`, fixed a priori (num_leaves 15, min_child_samples 100, lr 0.03, L2 5) |
| Variants tried | **1** model configuration on the v3 dataset (plus the v2-dataset before/after below). No tuning on results; the Bonferroni-adjusted CI equals the 95% CI. |
| Reproduce | `python3 scripts/ml_v3_eval.py --bars bars.json --out report.json` (Yahoo/Alpaca fetch if `--bars` omitted; `--v2-compare` adds the legacy-vs-honest table) |

## Prior (stated before the run)

Expected v3 OOS AUC 0.50–0.53 and no significant top-minus-bottom spread:
daily-bar technicals on liquid large caps are the most arbitraged features
there are, and the honest v2 number (0.515) was already near chance.
Result: inside the prior, at its low end.

## 1. CV leak: before vs after on identical inputs (v2 dataset)

v2 generic builder (`_build_training_data`, triple-barrier label), 97 tickers,
90,210 rows (decisions 2023-02-02 → 2026-10-16), base rate 0.427, same LightGBM params in both arms.

| Procedure | pooled AUC | log-loss | base-rate log-loss | accuracy |
|---|---|---|---|---|
| legacy (row-index folds on ticker-major rows, 5×n_tickers-row embargo, early stop on test fold) | **0.658** | 0.648 | 0.681 | 61.9% |
| honest (date folds, 5-date embargo + purge, inner-validation early stop) | **0.515** | 0.690 | 0.684 | 56.0% |

Per fold, legacy AUC 0.57–0.72; honest 0.50–0.54. Honest accuracy 56.0% is below
the 56.8% you get by always predicting the majority class (loss).

## 2. v3 candidate, pooled out-of-sample

| Metric | v3 OOS | constant base-rate forecast |
|---|---|---|
| AUC | 0.499 (CI 0.479–0.516) | 0.500 |
| Log-loss | 0.6949 | 0.6928 |
| Brier | 0.2508 | 0.2498 |
| Accuracy @0.5 | 51.8% | 51.3% (always "win") |
| Mean prediction / base rate | 0.517 / 0.513 | |

Per fold (test window → AUC, log-loss vs base-rate log-loss, trees):

| Fold | test window | AUC | log-loss | base LL | trees |
|---|---|---|---|---|---|
| 0 | 2024-06-12 → 2024-11-21 | 0.536 | 0.6874 | 0.6867 | 1 |
| 1 | 2024-11-22 → 2025-05-09 | 0.499 | 0.6974 | 0.6930 | 7 |
| 2 | 2025-05-12 → 2025-10-22 | 0.502 | 0.6912 | 0.6870 | 1 |
| 3 | 2025-10-23 → 2026-04-08 | 0.479 | 0.7037 | 0.6919 | 29 |
| 4 | 2026-04-09 → 2026-09-28 | 0.505 | 0.6946 | 0.6929 | 13 |

## 3. Realized return by predicted quintile (net of 0.10% round trip)

Base rate (random entry, same universe, same live exit rule): **+0.364% per
trade**, win rate 51.3%. Any model claim is the excess over this.

| Quintile | mean pred | mean net ret % | median % | win rate |
|---|---|---|---|---|
| Q1 (lowest) | 0.469 | 0.330 | -0.100 | 49.2% |
| Q2 | 0.508 | 0.513 | 0.419 | 54.0% |
| Q3 | 0.520 | 0.352 | 0.002 | 50.0% |
| Q4 | 0.533 | 0.419 | 0.430 | 53.9% |
| Q5 (highest) | 0.554 | 0.205 | -0.064 | 49.5% |

Top minus bottom: **-0.125 pts** (95% date-block CI -0.79..+0.55). Not
monotone; the top quintile is the worst.

## 4. Per regime (exit-policy regime class at decision)

| Regime | n | AUC | log-loss | base LL | base rate | mean net ret % | Q5−Q1 |
|---|---|---|---|---|---|---|---|
| bullish | 28,615 | 0.513 | 0.6928 | 0.6929 | 0.511 | 0.368 | +0.11 |
| neutral | 16,490 | 0.492 | 0.6965 | 0.6931 | 0.497 | 0.074 | -0.37 |
| bearish | 10,670 | 0.463 | 0.6978 | 0.6894 | 0.543 | 0.800 | +0.49 |

Nothing clears chance after the 3-way split (three looks at the same data =
another multiple-testing discount). Note the *base rates* differ by regime
(bearish-tape entries earned the most under the live exits, neutral the
least) — a regime fact, not a model fact.

## 5. Calibration (raw OOS probabilities, 10 equal-count bins)

Predicted 0.44→0.57 vs realized 0.50, 0.49, 0.56, 0.52, 0.51, 0.49, 0.54,
0.54, 0.53, 0.46: flat. The isotonic calibrator stored with the candidate is
fitted on these honest OOS probabilities, so it maps nearly everything to
~0.51 — the served v3 probability will be correctly unconfident.

## 6. What the model leans on

Gain importance of the final fit (12 trees): vxx_rank_252, spy_vs_ma50,
vol_of_vol, vxx_ratio, credit_spread, spy_ret_5d, atr_pct — the
market-wide, date-level features. Those can only time the whole market (every
stock on a date gets the same value), which is exactly the axis on which the
legacy CV leaked.

## Caveats (bias direction stated)

* Yahoo ≠ Alpaca bars (adjustment and volume conventions differ slightly);
  the production job retrains on Alpaca bars, the same source as serving.
  Direction: unknown, likely small.
* Survivorship (today's names only): biases mean returns UP; affects the base
  rate more than the model's ranking.
* Overlapping labels (each row's path spans up to 20 bars, same-day rows
  share the market move): rows are not independent — this is why CIs are
  date-block bootstraps, and why "55,775 rows" overstates the evidence.
* Live-exit mirror limitations (daily-bar touches, sampled peaks) are listed
  in `live_exit_policy.py`; direction: slightly optimistic returns.
* Shadow records (the population the bot actually scores) were not available
  here. The production job and `/api/diag/ml_v3` give the Alpaca-based
  version; the forward test is `ml.v3_win_probability` vs the exit-aligned
  shadow outcome, which accumulates from deploy.

## What could and could not be run

* Run here: everything above, on Yahoo bars, ~100 s end to end.
* Not run here: training on production shadow records or Alpaca bars; the
  production job. Production path: `server/bot.ts checkMlV3CandidateJob`
  runs `ml_model_v3.run_job(time_budget_s=60)` every 5 min during 03–04 UTC
  (staged and checkpointed under the 90 s kill cap), weekly
  (`ML_V3_RETRAIN_DAYS`). Read the result at `/api/diag/ml_v3`.

## Needs a decision (human)

1. The v2 ML blend (rule/ML 0.60–0.72 / 0.28–0.40) feeds a model whose
   honest skill on this evidence is ~zero. Removing or shrinking it is a
   rule change needing counterfactual evidence: compare `rules_only_score`
   vs `final_score` against exit-aligned outcomes in the shadow log.
2. v2's own training path still uses the leaky CV for its reported accuracy
   and its isotonic calibrator; switching it changes live scores.
