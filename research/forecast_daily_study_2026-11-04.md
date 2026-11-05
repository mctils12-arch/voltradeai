# Daily forecast range model — HAR-RV + implied vol, walk-forward study (Part C, 2026-11-04)

Tag: [RESEARCH] — the study behind `forecast_range.py` (model version **fc-range-har-v1**). Nothing trades on it: the forecast page is shadow.
Human-approved batch, Part C item 2: "Range model — the part with real skill. Blend chain implied vol (ATM IV / VIX for SPY) with a HAR-RV realized-vol model (daily/weekly/monthly realized vol). Fit the blend weights out-of-sample, walk-forward, no lookahead. Update intraday as IV and price move. Score range coverage daily against the 68% target."
Run: `python3 scripts/forecast_daily_study.py fetch --data-dir DIR` then `python3 scripts/forecast_daily_study.py range --data-dir DIR --md OUT.md --json research/forecast_daily_study_2026-11-04.json`. That takes about 40 minutes on one core. Most of it is re-fitting the dossier HAR at every origin, as it is served. Machine-readable output: `research/forecast_daily_study_2026-11-04.json` (derived statistics only). `forecast_range.RANGE_COEF` and `REMAINING_VARIANCE` are pinned to it by `test_forecast_range.py`.

## Verdict

- **The HAR + implied-vol blend ships as the daily band (fc-range-har-v1), in all three fit groups.**
  - Every group passes the pre-registered ship rule R2 on the 2019-01-02 → 2026-11-02 holdout:
    - the interval score is lower than the current path's (V0), with a day-block-bootstrap 95% CI that excludes 0;
    - coverage is within ±5 points of 68.27%.

  | fit group (production use) | chosen (R1, on 2013–2018) | holdout coverage [95% CI] | interval score vs current path | interval-score skill vs a constant band |
  |---|---|---|---|---|
  | index: SPY with VIX | V4 HAR + IV | 68.6% [66.9, 70.5] (SPY alone 70.8%) | −4.8% (−0.186 [−0.218, −0.149] pct-pts) | +7.0% (SPY +7.9%) |
  | names with chain IV30 | V4 HAR + IV | 68.4% [66.8, 69.8] | −4.0% (−0.230 [−0.270, −0.187]) | +4.4% |
  | names without IV (41-name panel) | V3 HAR only | 65.8% [64.7, 66.9] | −1.2% (−0.065 [−0.093, −0.039]) | +3.9% |

- **The current path (V0) is badly calibrated at a one-day horizon.**
  - It covers 75.5% (index) / 75.1% (stocks) instead of 68.3%: its bands are 20–25% too wide.
  - On Fridays it covers 92.9% for the index group. `sigma_for` scales implied vol by calendar days, so a Friday → Monday band uses 3 days of variance for one session.
  - Weekly (h = 5) still over-covers (79%), less dramatically.
- **Implied vol carries most of the weight where it exists.**
  - The final blend puts 71% of its weight on VIX for the index group and 78% on single-stock implied vol. The rest is HAR, mostly the weekly component.
  - Adding IV beats HAR alone on SELECTION by a margin the CI excludes from zero:
    - index: −0.054 [−0.072, −0.032];
    - stocks: −0.118 [−0.152, −0.081].
- **The weekly would benefit too (R3): reported, not switched.** At h = 5 the same models beat the current weekly band by R2's rule in all three groups:
  - index −2.6%, CI [−0.358, −0.070];
  - stocks −1.0%, CI [−0.242, −0.005], barely;
  - HAR −0.8%.
  The weekly band is left on its current path, per the pre-registration. Switching it is a separate change with its own version bump; filed as a follow-up below.
- **Regime honesty: the band is not calibrated in every regime.**
  - It is fit to hit 68.27% pooled, so it over-covers in calm tapes and under-covers in stress.
  - Index group, holdout: low-VIX tercile 80.0%, BULL 75.6%, BEAR 60.4%, PANIC 61.7% (n = 256 days).
  - The current path has the same shape, only shifted up.
  - The page shows coverage live, by n. Splitting it by regime once n allows is a follow-up.
- **Intraday (live band).**
  - The pre-registered variance-ratio scaling FAILED its prior: the second-half live coverage was 77–87% for index ETFs.
  - A declared POST-HOC quantile-ratio scaling tracks the daily band's own coverage within about 3 points at every mark up to 14:30. It ships for the display-only live band. Details below.
- **No intraday IV.** The repo's VIX (Cboe CSV) and chain-IV30 (options archive) sources are end-of-day series. The live band therefore re-centres on the latest price and decays with the time left, but cannot react to an intraday IV move. Saying so on the page is required, and it is said.

## Pre-registration (verbatim; written 2026-11-04 ~03:50 UTC, before any result was computed)

> Question: for the next-session close (h = 1), does a HAR-RV model blended
> with implied vol give a better 68.27% band than the current path, out of
> sample, walk-forward, by coverage AND by a proper interval score?
>
> Data (research only; production keeps Alpaca bars + the existing VIX/chain IV):
> - Yahoo daily OHLC, split-adjusted, NOT dividend-adjusted (mirrors Alpaca
>   `adjustment=split`). End fixed at 2026-11-03 (last completed session);
>   last origin 2026-11-02.
> - Cboe 30-day implied-vol indices (cdn.cboe.com daily_prices CSVs):
>   index group  SPY/VIX, QQQ/VXN, IWM/RVX, DIA/VXD
>   stock group  AAPL/VXAPL, AMZN/VXAZN, GOOG/VXGOG, IBM/VXIBM, GS/VXGS
>   other ETFs   GLD/GVZ, USO/OVX, EEM/VXEEM, SLV/VXSLV, EWZ/VXEWZ, FXI/VXFXI
>   (transfer check only)
> - HAR-only panel: the 15 TRAIN_BASKET ETFs + 20 fixed single names
>   (AAPL AMZN GOOG IBM GS MSFT JPM XOM PFE KO WMT NVDA AMD TSLA NFLX BA CAT F
>   INTC DIS) + the 6 other ETFs. Survivorship: every name exists today.
>
> Timing: origin t uses closes <= t and the IV index close of t (VIX settles
> 16:15 ET, 15 min after SPY — skew on the implied side only, as in P1).
> Outcome r_{t+1} = log(C_{t+1}/C_t).
>
> Regressors (daily-vol units): v_d = |r_t|; v_w = sqrt(mean r^2 over t-4..t);
> v_m = sqrt(mean r^2 over t-21..t); v_iv = IV_t/100/sqrt(252).
>
> Variants (ALL of them; nothing else is tried):
> - V_clim  constant band: per-name 68.27% quantile of |r| over the expanding
>           training window (the no-skill reference for ranges).
> - V0      current path at h = 1: sigma_for analog — IV * sqrt(calendar days to
>           the next session / 365) when IV exists, else 20-session realized
>           vol (sample std) * sqrt(1); z = 1.
> - V1      dossier HAR (dossier_ranges.sigma_forecast 1-session sigma: HAR on
>           r^2 levels, own 750 sessions, EWMA fallback), z = 1.
> - V2      IV only:   q = th_iv * v_iv
> - V3      HAR only:  q = th_d v_d + th_w v_w + th_m v_m
> - V4      HAR + IV:  q = th_d v_d + th_w v_w + th_m v_m + th_iv v_iv  (the blend)
> V2-V4: linear quantile regression of |r_{t+1}| at tau = 0.6827, no
> intercept (scale-equivariant, so one fit pools names of different vol),
> th >= 0, solved as a linear program. Minimizing that pinball loss is
> exactly minimizing the expected interval score of the symmetric band, so
> the fit targets the metric. Blend weight on IV = th_iv / sum(th).
>
> Fit groups: index (4 index ETFs, with IV), stock-IV (5 stocks, with IV),
> HAR-only (whole HAR panel, no IV). Walk-forward: expanding window, refit
> every 21 sessions; training labels end >= 1 session (the horizon) before
> the first origin the refit forecasts (purge + 1-session embargo); >= 504
> training sessions before the first evaluated origin.
>
> Segments: SELECTION origins 2013-01-02 .. 2018-12-31; HOLDOUT 2019-01-02 ..
> 2026-11-02. Model choice uses SELECTION only.
>
> Metrics: coverage of the 68.27% band (block-bootstrap 95% CI, 20-session
> blocks, resampling whole days); mean interval score
> IS = 2q + (2/alpha)(|r| - q)+ with alpha = 0.3173 (percent units);
> sharpness = mean width 2q; interval-score skill vs V_clim =
> 1 - IS/IS_clim. Splits: VIX terciles (cut points from the SELECTION
> segment), the bot's regime labels (regime_util.classify_regime_5level with
> VIX/30-day-mean as the VXX-ratio proxy, as in P1), weekday (diagnostic).
>
> Decision rules:
> - R1 (choose): per fit group, among V2/V3/V4 pick the lowest pooled mean
>   interval score on SELECTION (coverage there must be within +/-3 pts of
>   68.27, else the next best).
> - R2 (ship): the chosen variant ships as the daily band (fc-range-har-v1)
>   iff on HOLDOUT its pooled mean IS is below V0's with the block-bootstrap
>   95% CI of the difference excluding 0, AND holdout coverage is within
>   +/-5 pts of 68.27. Otherwise the daily band ships on V0 and the doc says so.
> - R3 (weekly): repeat at h = 5 (|r_{t+1..t+5}|, non-overlapping origins for
>   evaluation). If the chosen h = 5 variant beats the current weekly path by
>   R2's rule, REPORT it as a follow-up; the weekly is not switched in this batch.
> Production uses the coefficients of the LAST walk-forward refit (data
> through 2026-11-02) frozen under the model version; a refit = a new
> version + this study rerun.
>
> Intraday (live band) sub-study: Yahoo 60m bars (~725 sessions). f(tau) =
> remaining share of close-to-close variance at ET marks 09:30, 10:30, ...,
> 15:30 (16:00 -> 0), estimated as mean (log C/S_tau)^2 / mean r^2 on the
> FIRST half of the sessions, two fixed groups (index = SPY QQQ IWM DIA;
> other = everything else in the 60m panel). Live band = S_tau exp(+/- q_t
> sqrt(f(tau))). Evaluate coverage per mark on the SECOND half.
>
> Priors (stated before running):
> - P1 V4 beats V0 on holdout IS for the index group: 85%.
> - P2 adding IV helps (V4 beats V3 on SELECTION, index group): 75%.
> - P3 adding IV helps single stocks (stock-IV group): 70%.
> - P4 V4 holdout coverage within +/-3 pts for SPY: 60%; within +/-5: 85%.
> - P5 the chosen band under-covers by >= 3 pts in the high-VIX tercile or
>   BEAR regime: 50%.
> - P6 z = 1 Gaussian bands (V0, V1) over-cover (> 72%): 70%.
> - P7 the weekly (h = 5) would benefit by R2's rule: 60%.
> - P8 intraday f(tau) band coverage within +/-5 pts at every mark: 55%.

## Decisions made while building, before the run

- **Data start 2009-09-18.** That is the first day every Cboe index-group CSV (VXN, RVX, VXD) exists. Earlier closes feed only the regressors' look-back.
- **Single-stock IV.** It starts 2011-01-07, so the stock-IV group's first walk-forward refit has fewer than 504 IV rows until early 2013. The refit is skipped until it has them.
- **Common rows.** Every comparison uses the rows where every variant of the group has a band. In the IV groups, all fits, V3 included, train on rows where IV exists, so "adding IV" is a like-for-like comparison.
- **Quantile solver.** The quantile regression is solved by majorize-minimize (`forecast_range.fit_quantile_nonneg`). It is exact under θ ≥ 0 at each step, and it agrees with scipy/HiGHS's exact LP to ~1e-9 in the pinball loss on 20k rows (test-pinned). The LP itself took 13 s per fit at 20k rows — too slow for 166 refits × 3 groups × 3 variants.

## Data (free, keyless; nothing raw committed)

| input | source | span |
|---|---|---|
| daily OHLC, 42 symbols | Yahoo v8 chart API (the keyless endpoint `backtest_v2._yahoo_bars` already uses); `close` is split-adjusted, NOT dividend-adjusted — Alpaca's `adjustment=split` convention, so ex-dividend drops sit in both | 1999-01-04 → 2026-11-03 |
| 30-day implied-vol indices | Cboe `daily_prices/<INDEX>_History.csv` (the host `server/cboeVix.ts` archives VIX from): VIX VXN RVX VXD; VXAPL VXAZN VXGOG VXIBM VXGS; GVZ OVX VXEEM VXSLV VXEWZ VXFXI | 2009-09 / 2011 → 2026-11-03 |
| 60-minute bars, 30 symbols | Yahoo v8 chart API, regular session | 2023-12-06 → 2026-11-03 (722 complete sessions) |

- **Panels.**
  - HAR-only: the 15 `forecast.TRAIN_BASKET` ETFs, plus 20 fixed single names (AAPL AMZN GOOG IBM GS MSFT JPM XOM PFE KO WMT NVDA AMD TSLA NFLX BA CAT F INTC DIS), plus the 6 other ETFs.
  - Every name exists today, so the panel is survivorship-biased. That makes the vol panel slightly calmer than a point-in-time universe would be. The bias is in the sample, not in any look-ahead.
- **Alpaca.** No keys in this sandbox. Production computes the same regressors from Alpaca's split-adjusted closes. The study uses closes only, never highs or lows, because production bars are IEX-only when the SIP entitlement is downgraded (`alpaca_feed.bars_feed`), and IEX highs and lows would bias a range estimator.

## Results — next session (h = 1)

Coverage CIs and difference CIs are 2,000-resample moving day-block bootstraps (20-session blocks; all names of a day move together). The interval score IS = 2q + (2/α)·max(|y| − q, 0) is in percent (α = 0.3173); lower is better.

### Index group (SPY QQQ IWM DIA; IV = VIX VXN RVX VXD) → production: SPY with VIX

| | SELECTION 2013–2018: coverage | IS | width | HOLDOUT 2019–2026: coverage [95% CI] | IS | width |
|---|---|---|---|---|---|---|
| V_clim constant band | 74.9% | 2.796 | 1.80 | 62.3% [59.3, 65.6] | 3.942 | 1.81 |
| V0 current path | 77.0% | 2.739 | 1.97 | 75.5% [73.6, 77.5] | 3.852 | 2.70 |
| V1 dossier HAR | 73.4% | 2.678 | 1.77 | 72.7% [70.8, 74.6] | 3.739 | 2.49 |
| V2 IV only | 69.0% | 2.610 | 1.55 | 69.0% [66.9, 71.1] | 3.679 | 2.21 |
| V3 HAR only | 66.6% | 2.655 | 1.56 | 65.9% [64.0, 67.6] | 3.724 | 2.24 |
| **V4 HAR + IV** | 69.0% | **2.602** | 1.57 | **68.6% [66.9, 70.5]** | **3.667** | 2.23 |

- HOLDOUT dIS vs V0:
  - V4 −0.186 [−0.218, −0.149]
  - V2 −0.174
  - V3 −0.129
  - V1 −0.114
  - V_clim +0.089 [−0.053, +0.264]
- Per name, HOLDOUT, V4 vs V0 (coverage, IS):

  | name | V4 coverage | V4 IS | V0 coverage | V0 IS |
  |---|---|---|---|---|
  | SPY | 70.8% | 3.159 | 77.3% | 3.362 |
  | QQQ | 67.0% | 4.195 | 74.0% | 4.366 |
  | IWM | 66.2% | 4.332 | 73.0% | 4.520 |
  | DIA | 70.6% | 2.981 | 77.7% | 3.161 |

- SPY's constant-band IS is 3.432, so SPY's range skill is +7.9%.
- Blend weights:
  - First refit (2013-01-02): v_d 0, v_w 0.107, v_m 0.110, v_iv 0.584 (IV weight 0.73).
  - Last refit (2026-10-13, shipped): v_d 0, v_w 0.219, v_m 0.028, v_iv 0.604 (IV weight 0.71).
  - The IV share is stable. The realized part moved from the monthly to the weekly component over the decade.

### Names with implied vol (AAPL AMZN GOOG IBM GS; IV = Cboe's VIX-method single-stock indices) → production: any name with a chain IV30

| | SELECTION coverage | IS | HOLDOUT coverage [95% CI] | IS | width |
|---|---|---|---|---|---|
| V_clim | 72.3% | 4.449 | 62.7% [60.3, 65.2] | 5.806 | 2.71 |
| V0 | 76.5% | 4.488 | 75.1% [73.4, 76.6] | 5.778 | 3.87 |
| V1 | 77.5% | 4.481 | 74.8% [73.3, 76.3] | 5.703 | 3.73 |
| V2 | 68.0% | 4.296 | 68.1% [66.5, 69.7] | 5.561 | 3.19 |
| V3 | 65.9% | 4.402 | 65.9% [64.5, 67.3] | 5.680 | 3.18 |
| **V4** | 68.0% | **4.284** | **68.4% [66.8, 69.8]** | **5.548** | 3.21 |

- Blend weights:
  - First refit: 0.025 / 0 / 0.045 / 0.702 (IV weight 0.91).
  - Shipped: 0.021 / 0.097 / 0.061 / 0.647 (IV weight 0.78).
- Per name, HOLDOUT, V4 coverage: AAPL 67.3%, AMZN 67.6%, GOOG 68.3%, IBM 71.4%, GS 67.3%. V4 beats V0's IS on all five.

### Names without implied vol (41-name panel, HAR only) → production: every name without a chain IV30 for the base day

| | SELECTION coverage | IS | HOLDOUT coverage [95% CI] | IS | width |
|---|---|---|---|---|---|
| V_clim | 72.8% | 4.215 | 62.5% [60.3, 64.8] | 5.585 | 2.57 |
| V0 (20-day RV × 1, z = 1) | 70.3% | 4.168 | 70.1% [68.8, 71.3] | 5.434 | 3.42 |
| V1 dossier HAR | 75.0% | 4.177 | 73.7% [72.5, 75.0] | 5.403 | 3.57 |
| **V3** | 66.2% | **4.119** | **65.8% [64.7, 66.9]** | **5.370** | 3.08 |

- Shipped weights: v_d 0.022, v_w 0.262, v_m 0.634.
- **Honest weakness.** The HAR-only band under-covers on the holdout: 65.8%, about 2.5 points short, and 62–65% on the index ETFs. HAR alone reacts to realized vol with a lag, and the 2019–2026 holdout had more vol-of-vol than its training years. It still beats the current path's interval score with a CI that excludes zero, because the current path buys its coverage with bands 11% wider.
- The live page splits coverage by `range_source`, so this group's shortfall stays visible.

### Transfer check: ETFs with their own Cboe vol index (GLD USO EEM SLV EWZ FXI), HOLDOUT

| variant | coverage [95% CI] | IS | dIS vs V0 |
|---|---|---|---|
| V0 | 74.4% [73.2, 75.5] | 5.347 | — |
| V4 with the index coefficients | 68.2% [67.1, 69.3] | 5.170 | −0.177 [−0.214, −0.142] |
| V4 with the single-stock coefficients | 67.7% [66.5, 68.8] | 5.166 | −0.181 [−0.219, −0.144] |
| V3 with the HAR coefficients | 65.5% [64.3, 66.6] | 5.247 | −0.100 [−0.146, −0.059] |

- The single-stock blend transfers to ETFs it never saw, at the right coverage.
- That supports production's rule: any name, ETF or stock, with a chain IV30 for its base day uses the `stock_iv` set.

### By regime and weekday (HOLDOUT coverage; the shipped model vs the current path)

VIX tercile cut points come from the SELECTION segment: 12.81 / 15.07. 2013–2018 was calm, so most holdout days fall in the "high" tercile. Labels come from the bot's 5-level classifier, fed as in P1.

| | index V4 | index V0 | stocks V4 | stocks V0 | no-IV V3 | no-IV V0 |
|---|---|---|---|---|---|---|
| VIX low | 80.0% | 87.5% | 73.2% | 84.8% | 69.8% | 72.9% |
| VIX mid | 70.3% | 76.8% | 70.7% | 77.4% | 67.5% | 71.8% |
| VIX high | 67.5% | 74.4% | 67.5% | 74.0% | 65.2% | 69.5% |
| BULL | 75.6% | 82.2% | 72.4% | 79.6% | 70.2% | 75.0% |
| NEUTRAL | 69.1% | 76.9% | 69.2% | 75.8% | 64.3% | 68.6% |
| CAUTION | 66.3% | 73.3% | 67.6% | 74.5% | 63.1% | 67.4% |
| BEAR | 60.4% | 67.1% | 63.7% | 70.1% | 63.5% | 67.4% |
| PANIC (n 256 / 320 / 2,624) | 61.7% | 62.9% | 56.9% | 61.3% | 56.4% | 57.1% |
| Mon | 72.4% | 74.5% | 71.9% | 73.3% | 67.7% | 72.0% |
| Thu | 64.5% | 68.1% | 66.4% | 69.4% | 64.9% | 69.2% |
| Fri | 70.7% | **92.9%** | 66.7% | **90.9%** | 65.3% | 69.3% |

- At h = 1 the shipped band has a lower interval score than the current path in every group, every VIX tercile and every regime label, BEAR and PANIC included (per-regime IS in the json). At h = 5 it does not in BEAR (index 11.00 vs 10.88, stocks 16.15 vs 15.91).
- The calibration gap by regime is the remaining defect: about 10–15 points between BULL and BEAR. A regime-conditional quantile, or a VIX-term-structure input, is the obvious next variant. It is not tried here; trying it would be another variant, discounted accordingly.

## Weekly horizon (h = 5) — R3, reported only

Evaluation uses non-overlapping origins (every 5th session).

| group | chosen | HOLDOUT coverage | IS vs current weekly band | CI |
|---|---|---|---|---|
| index | V4 | 68.1% (current path 79.3%) | −2.6% | [−0.358, −0.070] |
| stocks | V4 | 66.1% (current path 74.4%) | −1.0% | [−0.242, −0.005] |
| no IV | V3 | 64.7% (current path 68.1%) | −0.8% | [−0.165, −0.037] |

- By R2's rule the weekly would benefit in all three groups. The stock group only barely passes, and on the no-IV group the dossier HAR (V1) beats V3 on IS at h = 5 (12.47 vs 12.52).
- Pre-registered outcome: report, do not switch. Proposal (follow-up, its own PR and version): move the weekly band to the h = 5 fits from this same study.

## Intraday live band (60-minute bars)

- **Method.** The live band re-centres on the latest price S_τ and scales the pre-open band by sqrt(f(τ)).
- **Fit split.** f is fit on the first half of the 722 sessions (to 2025-05-20). Live coverage is measured on the second half (2025-05-21 → 2026-11-03).
- **Reference.** On those second-half sessions the pre-open band itself covered 73.5% for index ETFs and 67.0% for the other 26 names.

| mark (ET) | PRE-REGISTERED f (variance ratio) — live coverage, index / others | POST-HOC f (quantile ratio) — live coverage, index / others | shipped f (quantile, full sample), index / others |
|---|---|---|---|
| 09:30 | 77.3% / 66.2% | 72.4% / 67.5% | 0.599 / 0.615 |
| 10:30 | 82.7% / 69.8% | 74.1% / 68.2% | 0.350 / 0.306 |
| 11:30 | 84.9% / 72.7% | 74.4% / 69.0% | 0.255 / 0.203 |
| 12:30 | 87.3% / 75.5% | 75.9% / 69.4% | 0.174 / 0.148 |
| 13:30 | 81.5% / 70.7% | 75.2% / 68.7% | 0.123 / 0.100 |
| 14:30 | 77.8% / 68.2% | 73.8% / 67.7% | 0.075 / 0.063 |
| 15:30 | 83.8% / 70.5% | 81.3% / 70.0% | 0.035 / 0.030 |

- **The pre-registered method failed its prior P8** (coverage within ±5 at every mark). It over-covered the index group by up to 14 points at midday.
  - Diagnosis: a variance ratio is a ratio of MEAN squared moves. It was dominated by a handful of event hours in the fit half — 2025-04-09's tariff-pause hour alone was a +4.8% SPY move between 12:30 and 13:30 — so it overstated the typical remaining move.
  - The band is a quantile, so it should shrink with the quantile of the remaining move. That is the post-hoc variant: f = [Q68(|log C/S_τ|/q) / Q68(|log C/C_prev|/q)]², with a running minimum so f only falls through the day.
- **It is a SECOND variant measured on the same second half.** It is recorded as post-hoc here and in `forecast_range.py`.
  - It ships because it is the consistent choice for a quantile band, not because it scored better. It also scored better: within 3 points of the band's own coverage at every mark through 14:30.
  - The last half hour is noisy (index 81% at 15:30, where the band is a fifth of its pre-open width).
  - The live band is display-only and never scored, so the residual risk is a misleading width on screen, not corrupted learning.
- **Half days.** A half day's 3.5 hours are stretched onto the 09:30–16:00 clock (`forecast.session_minutes`). That is an approximation, untested: the 60-minute sample excludes half days.

## Priors vs outcomes

| prior (stated before running) | outcome |
|---|---|
| P1 V4 beats V0 on holdout IS, index group: 85% | yes: −0.186 [−0.218, −0.149] |
| P2 adding IV helps on SELECTION, index: 75% | yes: −0.054 [−0.072, −0.032] |
| P3 adding IV helps single stocks: 70% | yes: −0.118 [−0.152, −0.081] |
| P4 V4 SPY holdout coverage within ±3 pts: 60%; within ±5: 85% | 70.8%: within ±3, barely (+2.5) |
| P5 under-covers by ≥ 3 pts in the high-VIX tercile or BEAR: 50% | yes: BEAR 60.4% (−7.9); the high-VIX tercile is close (67.5%) |
| P6 z = 1 Gaussian bands (V0, V1) over-cover (> 72%): 70% | yes for V0 (75.5%) and V1 (72.7%) in the index group; V0 70.1% on the no-IV panel |
| P7 the weekly would benefit by R2's rule: 60% | yes, all three groups (stocks barely) |
| P8 intraday coverage within ±5 at every mark (pre-registered f): 55% | **no** (index 77–87%) → the post-hoc quantile scaling above |

## What ships (fc-range-har-v1, `forecast_range.py`)

- **`RANGE_COEF`**: the last refit of each chosen variant. `index_iv` (SPY with VIX), `stock_iv` (any name with a chain IV30 dated the base day), `har` (everything else). The source is stored on every call as `range_source` ∈ {`har+vix`, `har+chain_iv30`, `har`}.
- **`REMAINING_VARIANCE`**: the quantile-ratio f by ET mark, for the index ETFs (SPY QQQ IWM DIA) and for everything else.
- **`interval_score`**: the measurement the daily scorer stores per call, together with the constant-band reference, so the page shows range skill live and not only coverage.
- **Production inputs are unchanged sources**: Alpaca split-adjusted closes (`shadow_portfolio._fetch_bars_paginated`), VIX from the Cboe archive the Node side already keeps (`server/cboeVix.ts`), chain IV30 from the options archive (`server/chainIv.ts`). Production never reads Yahoo.

## Caveats and risks

1. **IV definition transfer.** The `stock_iv` weights were fit on Cboe's VIX-method 30-day indices, which include the skew. Production feeds a delta-ATM chain IV30 from Alpaca's indicative feed, which is typically a little lower for names with steep put skews. Expect the `har+chain_iv30` bands to be slightly narrow; the live `by_range_source` coverage will show it. SPY uses VIX itself, so it has no transfer gap.
2. **No intraday IV.** The repo has no intraday VIX or chain-IV source; both feeds are end-of-day. Making the live band follow IV intraday needs a new source (e.g. Cboe's delayed VIX quote), with its own gate-1 check. It is not built here.
3. **Scheduled events.** A name without IV gets no warning of earnings inside the session. Where IV exists, the 30-day IV only partly carries the event.
4. **Regime calibration.** About 10–15 points of coverage spread between calm and stressed regimes (above).
5. **IEX bars.** When the SIP entitlement is downgraded, production closes are IEX's last prints. For liquid names those are within cents of the consolidated close; for thin names the close-to-close noise is larger than in this study.
6. **Survivorship.** The panel is today's names. The 41-name fit is a mega/large-cap fit. Small caps were not measured.

## Follow-ups (for the parent session to file; not done here)

- Move the weekly band to the h = 5 fits (R3 passed), as its own PR and version.
- Pre-register one regime-conditional variant (e.g. a VIX/VIX3M term-structure regressor, from the archive `server/cboeVix.ts` already keeps) aimed at the BULL/BEAR coverage spread. One variant, discounted.
- Score the live band out of sample: log the 12:30 ET live band once per session and score it at the close. That would make the post-hoc f a measured, not a once-validated, claim.
