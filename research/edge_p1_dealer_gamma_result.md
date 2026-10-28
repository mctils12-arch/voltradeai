# Edge P1 — dealer gamma vs realized/implied vol: pre-registered gate-2 result (2026-10-28)

Tag: [RESEARCH]. Ladder gate 2 (SIGNAL) only. Nothing trades, no trading code changed.
Spec: `research/edge_preregistration.md` § "P1 — Dealer gamma exposure → realized-vs-implied volatility", registered 2026-10-27 before any outcome was seen. Run with `python3 scripts/edge_p1_dealer_gamma.py` (downloads to a temp dir; `--data-dir` re-reads saved inputs). Human order 2026-10-28: "Run the P1 dealer-gamma test on free Alpaca data. No paid data."

**VERDICT: PASS** on the pre-registered rule. Full window b = −0.858, NW t = −9.61. Holdout b = −0.904, NW t = −7.25. The slope is about 4.5× the pre-registered MDE of 0.19. Caveats that limit what this licenses are below: gate 1 is still open, part of the slope comes from the VIX-vs-5-day horizon, and the outcome is not a tradeable P&L.

## Prior (copied verbatim from the pre-registration, stated before running)

70% that b < 0 with t ≤ −2.5 in-sample (mechanism well documented). 45% that it also clears the holdout bar (VIX may have learned to price positioning, especially since 2021). 35% that the effect is large enough (≥ 0.19) to matter for the sleeve.

Outcome vs prior: all three came true, including the 45% and 35% legs. The holdout slope is no weaker than the pre-2019 slope (−0.90 vs −0.74), so the "VIX has learned to price it" worry did not show up.

## Exact spec (unchanged from the pre-registration)

- `G_t` = SqueezeMetrics `gex` for date t. `R_t` = percentile rank of `G_t` within the trailing 252 GEX rows ending at t, inclusive. The rank is (count below + ½·ties) / 251, so it runs from 0 to 1. The first 252 rows are burn-in.
- `y_t = log(RV_{t+1..t+5} / IV_t)`. RV = Σ r² × 252/5 over SPY close-to-close log returns on days t+1..t+5. IV = (VIX_t/100)².
- Regression: OLS `y = a + b·R + c·log IV`, with Newey-West (Bartlett) lag 5 and no small-sample correction. In a split subsample, NW pairs are formed by SPY-day distance, not by row adjacency.
- PASS requires both of these:
  - b < 0 with t ≤ −2.5 on 2012-05 → 2026-10.
  - b < 0 with t ≤ −1.5 on the holdout, 2019-01 → 2026-10.
- KILL if any of these holds: b ≥ 0, full-window t > −2.5, or the holdout sign flips. The pre-registration does not say what happens when the full window passes and the holdout has the same sign but t > −1.5. The script reports that case as INCONCLUSIVE. It did not happen.

## Data (free, keyless; nothing raw committed)

| input | source | range used | rows |
|---|---|---|---|
| GEX (SPX, $ gamma per 1% move — an ESTIMATE) | SqueezeMetrics `DIX.csv` (squeezemetrics.com/monitor/static/DIX.csv) | 2011-05-02 → 2026-10-28 | 3,897 (pre-reg saw 3,895 through 10-26) |
| VIX close | Cboe `VIX_History.csv` (cdn.cboe.com, keyless; same source as `server/cboeVix.ts`) | 2012-05 → 2026-10-27 | 9,303 since 1990 |
| SPY daily OHLC | Yahoo v8 chart API, dividend- and split-adjusted close. This is the keyless fallback endpoint already used by `backtest_v2._yahoo_bars`. | 2011-01-03 → 2026-10-28 | 3,979 |
| VIX9D (post-hoc diagnostic only) | Cboe `VIX9D_History.csv` | 2011-01-04 → 2026-10-28 | 3,978 |

- **Alpaca:** this sandbox has no Alpaca keys, which exist only in production, so no Alpaca data was read.
  - SPY daily bars are consolidated-tape closes, and Yahoo's SPY series is vendor-equivalent to Alpaca's SIP daily bars for close-to-close returns.
  - Alpaca's free IEX-only feed would report different intraday highs and lows. That would affect only the Parkinson secondary.
  - Dividend adjustment matters only on the roughly 4 ex-dividend days per year. The test uses adjusted closes so those days do not create fake returns.
- **Panel:** 3,640 days (2012-05-01 → 2026-10-21; t+5 = 2026-10-28). The holdout has 1,962 days. Every SPY day in the window had a GEX row and a VIX close.
- **Date alignment:** this checks that a DIX row's date is the same session as SPY's close. The correlation of daily SPY returns with DIX `price` (SPX) returns is +0.999 at lag 0, and −0.11 at lags −1 and +1.
- **Input hashes:** the script prints sha256 of the inputs. The `spy.json` hash changes on every fetch because Yahoo embeds live metadata. The DIX and VIX hashes identify the data vintage.

## Results

### Primary (pre-registered)

| spec | n | b (R) | NW se | t | c (log IV) |
|---|---|---|---|---|---|
| **PRIMARY full 2012-05 → 2026-10** | 3640 | **−0.858** | 0.089 | **−9.61** | +0.146 (t +2.9) |
| **PRIMARY holdout 2019-01 → 2026-10** | 1962 | **−0.904** | 0.125 | **−7.25** | +0.047 (t +0.7) |
| pre-2019 half (info) | 1678 | −0.736 | 0.151 | −4.88 | +0.339 (t +3.4) |
| robustness (declared): R lagged 1 extra day, full | 3640 | −0.721 | 0.085 | −8.52 | +0.193 |
| robustness (declared): R lagged 1 extra day, holdout | 1962 | −0.730 | 0.118 | −6.19 | +0.108 |

- **Uncertainty:** the 95% CI on the full-window b is [−1.03, −0.68] log points. Read from the lowest to the highest gamma rank, that is realized/implied *variance* about 2.4× higher at the bottom, or vol about 1.5× higher.
- **Noise:** residual sd is 0.84 log units, against the 0.6 assumed in the power calculation. The test was noisier than planned and still far beyond the bar.
- **Non-overlapping check:** this checks the inference, not a different spec. Each run uses every 5th day, 728 windows, with classical SE. Across the 5 phase offsets, b ranges from −0.79 to −0.95 and t from −6.80 to −7.81. Those t-stats agree with the NW t, so the result is not an artefact of overlapping windows.
- **Publication timing:** the lagged-1-day robustness result (t −8.52) shows it is not driven by GEX timing. SqueezeMetrics builds GEX from open interest that OCC publishes the next morning. Even a full extra day of delay leaves the result in place.

### Base rate (what the test conditions against)

| sample | n | mean y | share RV > IV | RV/IV vol ratio (exp(mean y/2)) |
|---|---|---|---|---|
| full | 3640 | −0.849 | 15.2% | 0.654 |
| holdout | 1962 | −0.815 | 15.0% | 0.665 |
| GEX < 0 days (8.7% of the sample) | 317 | −0.270 | 25.6% | 0.874 |

Descriptive table by R quintile, full window. These are not tests.

| R bucket | n | mean y | share RV > IV | vol ratio |
|---|---|---|---|---|
| [0.0, 0.2) | 690 | −0.393 | 24.9% | 0.822 |
| [0.2, 0.4) | 622 | −0.639 | 19.3% | 0.726 |
| [0.4, 0.6) | 663 | −0.824 | 15.1% | 0.662 |
| [0.6, 0.8) | 677 | −0.958 | 12.0% | 0.619 |
| [0.8, 1.0] | 988 | −1.241 | 8.1% | 0.538 |

The holdout quintiles are the same shape: bottom quintile 24.8% / 0.835, top quintile 8.4% / 0.560. The ordering is monotone in all 5 buckets in both samples.

- **The bucket counts are uneven.** The top bucket holds 988 days because dollar GEX trends up with SPX's level, so new 252-day highs (R = 1) are common. That is a property of the pre-registered signal and is not corrected here.
- **The signal predicts the size of the premium, not its sign.** Even in the lowest-gamma quintile, realized vol averages only 82% of VIX.

### Secondaries (reported only; never used to pass)

**(i) Parkinson range RV instead of close-to-close RV:**
- Full: b −0.598, t −8.53.
- Holdout: b −0.591, t −5.91.
- The slope is smaller because Parkinson ignores overnight gaps.

**(ii) The 30-minute intraday-momentum split of [S2] by GEX sign:** NOT RUN. It needs 1-minute SPY history. Alpaca keys are absent here, and Yahoo serves 1-minute bars for only about 30 days. This can run in production, where the Alpaca keys exist.

**(iii) Split by VIX tercile:** cut points are from the full sample, so this is a descriptive split.

| VIX tercile | n | b | t |
|---|---|---|---|
| low (< 14.32) | 1212 | −0.577 | −3.54 |
| mid [14.32, 18.36) | 1211 | −0.903 | −6.48 |
| high (≥ 18.36) | 1217 | −1.038 | −6.85 |

**(iii) Split by the bot's regime labels:**
- Labels come from `regime_util.classify_regime_5level`, which is the live classifier.
- Inputs are SPY/MA50, SPY's 200-day trend, and VIX divided by its 30-day mean as a stand-in for the VXX ratio. This follows `scripts/regime_backtest.py`, because VXX history is unreliable before 2018.
- These are raw daily labels with no dwell or confirmation.

| regime | n | b | t | base mean y | share RV > IV |
|---|---|---|---|---|---|
| BULL | 1385 | −0.829 | −5.13 | −1.138 | 10.1% |
| NEUTRAL | 925 | −0.537 | −3.50 | −0.914 | 12.4% |
| CAUTION | 478 | −0.670 | −3.20 | −0.685 | 18.8% |
| BEAR | 737 | −0.405 | −2.22 | −0.439 | 22.7% |
| PANIC | 115 | −1.292 | −1.18 (n too small) | −0.141 | 35.7% |
| BULL+NEUTRAL | 2310 | −0.752 | −6.41 | | |
| CAUTION+BEAR+PANIC | 1330 | −0.635 | −4.28 | | |

The sign is negative in every VIX tercile and every regime. The effect is weakest in BEAR, where realized vol already runs closest to implied, and largest in calm, high-VIX-tercile states. It is not an artefact of one regime dominating the sample (REASONING #2).

### Post-hoc diagnostics (NOT pre-registered; never used for the verdict)

These were added after seeing the primary result. Each could only make the primary look *worse*. They count as variant #2 and #3 against this root for any later decision.

**(a) Vol clustering.** Low GEX follows selloffs: corr(R, log trailing 5-day RV) = −0.46. R might therefore only proxy recent realized vol that VIX under-prices.

**(b) Horizon mismatch.** VIX is a 30-day measure but the outcome covers 5 days. In calm, contango tapes, VIX overstates 5-day implied vol, and high GEX coincides with calm tapes. VIX9D is closer to the outcome horizon.

| diagnostic | n | b | t |
|---|---|---|---|
| + log trailing 5d RV control, full | 3640 | −0.842 | −9.44 |
| + log trailing 5d RV control, holdout | 1962 | −0.867 | −7.19 |
| VIX9D as implied (y and control), full | 3640 | −0.556 | −6.36 |
| VIX9D as implied, holdout | 1962 | −0.594 | −5.15 |
| VIX9D + trailing RV, full | 3640 | −0.557 | −6.35 |
| VIX9D + trailing RV, holdout | 1962 | −0.594 | −5.16 |

- **(a) does not explain the effect.** The trailing-RV control barely moves b.
- **(b) explains about a third of it.** Against the 9-day implied, b shrinks from −0.86 to −0.56 but still clears the bar.
- **What this means:** dealer gamma predicts 5-day SPY variance beyond both VIX and VIX9D. About 35% of the pre-registered slope is the term-structure gap between a 30-day implied and a 5-day outcome, not gamma.

## Verdict against the pre-registered rule

**PASS (gate 2, SIGNAL).**
- Full window: b −0.858 < 0 and t −9.61 ≤ −2.5.
- Holdout: b −0.904 has the same sign, and t −7.25 ≤ −1.5.
- None of the kill criteria hold.
- Round-level Bonferroni bar: |t| ≥ 2.5. The primary clears it by a wide margin.

## What it means for the short-premium sleeve (gate 2 only — nothing trades)

- **Effect on losing windows.** The bottom-quintile gamma state (R < 0.2, the gate-3 candidate rule) roughly triples the share of SPY 5-day windows where realized beats VIX-implied: 24.9% vs 8.1% in the top quintile.
- **Effect on the premium.** It also shrinks the average premium: the vol ratio is 0.82 vs 0.54.
- **The sign does not flip.** Even at low gamma, VIX on average still over-prices the next 5 days. A "no new short premium when R_{t−1} < 0.2" gate therefore trades forgone premium against avoided losses. Only the gate-3 simulation can say which side wins. That simulation is `scripts/csp_manager_rule_sim.py`, run with and without the rule.
- **The outcome is not P&L.** y compares a 5-day realized variance with a 30-day implied index. Nothing in the sleeve settles on that difference.
- **The sleeve trades other underlyings.** It sells single-name CSPs and QQQ puts. This test is SPY-index only, so transfer to single names is unproven. Single-name VRP was already found weak (research test 2).
- **Second-order: why has nobody arbitraged it?** Dealer hedging is mandatory risk reduction, not discretionary. The signal also mostly predicts the *size* of a premium that stays positive on average, not a mispricing anyone can trade cleanly. That fits a persistent effect, and fits the holdout not decaying since 2019.

## Caveats

1. **Gate 1 (DATA) is not passed.**
   - SqueezeMetrics GEX is a model ESTIMATE. It assumes dealers are long calls and short puts, and its methodology is proprietary.
   - The registered gate-1 cross-check has not run. That check compares SqueezeMetrics against our own Black-Scholes GEX built from Alpaca open interest, with the kill rule "sign disagrees on > 30% of ≥ 40 overlapping days". It needs OI history we do not yet have; see the `options_entitlements` probe.
   - The only gate-1-like evidence here is the date-alignment check.
   - By the ladder, this root may not advance to gate-3 *trading* until gate 1 passes. Gate-3 *simulation* may be prepared.
2. **Vintage risk.**
   - A SqueezeMetrics methodology backfill would be invisible to us.
   - SqueezeMetrics published its GEX methodology around 2017, so the in-sample history may partly reflect choices made with hindsight.
   - The 2019+ holdout was live after that publication, unless the history was revised. We cannot tell.
3. **Licence.**
   - DIX.csv terms are unverified, so it is for research use only. The raw CSV is not committed, since the repo is public.
   - Only derived statistics appear here. Do not redistribute it on /data. A production feed would need licence clarity, or our own gate-1 estimate in its place.
4. **Survivorship and lookahead.**
   - Index-level data, so there is no survivorship bias.
   - The rank uses GEX rows ≤ t, and the outcome uses t+1..t+5. Tests pin both.
   - VIX is published at 16:15 ET and SPY closes at 16:00. This 15-minute skew sits on the implied side only.
   - The VIX tercile cut points use the full sample, so that split is descriptive only.
5. **Rank drift.** Dollar GEX scales with SPX's level, so the trailing rank sits near 1 more often than uniform would. The spec is unchanged here, and a level-normalised GEX would be a new variant.
6. **Secondary (ii) was not run** (no free 1-minute history).

## Next steps on the ladder (proposals, not done here)

- **Gate 1.** Once Alpaca OI is confirmed populated, build our own GEX from the archive's IV and OI, then run the registered sign-agreement and rank-correlation check.
- **Gate 3.** Run `csp_manager_rule_sim.py` with and without "no new short premium when R_{t−1} < 0.2", plus the QQQ-put overlay sized up in that state.
- **Shadow logging.** The registered T-DATACORE `squeeze_gex` stream and the T-BOT shadow fields are still unbuilt.
