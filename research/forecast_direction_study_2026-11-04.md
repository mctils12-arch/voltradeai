# Daily forecast direction — pre-registered feature study (Part C, 2026-11-04)

Tag: [RESEARCH]. Ladder gate 2 (SIGNAL) only — nothing trades, no trading code changed.
Human-approved batch, Part C item 3: "Direction. Keep the probability shrunk toward the base rate. Candidate features: overnight gap, intraday momentum, VIX change, breadth, and dealer gamma if the GEX work exists. Ship a feature only if it beats the base-rate Brier on purged walk-forward holdout. The UI must keep showing skill vs. coin flip honestly."
Run: `python3 scripts/forecast_direction_study.py --data-dir DIR --dix PATH/DIX.csv` (DIR from `python3 scripts/forecast_daily_study.py fetch --data-dir DIR`). Machine-readable result: `research/forecast_direction_study_2026-11-04.json` (derived statistics only).

**VERDICT: NO FEATURE PASSES.** 13 of the 14 pre-registered tests ran; all 13 FAIL, and the combination test (T11) never qualified. Not one test came close to the multiple-testing bar (best one-sided p = 0.33 against a bar of 0.0036), and no feature improved the holdout Brier on both the selection and the holdout segment. **The daily direction ships as the shrunk base rate** — P(up) is the point-in-time up-frequency, the same for every name, and `forecast.DAILY_DIRECTION_FEATURES` is empty (pinned by `test_forecast_direction_study.py` and, once the daily horizon lands, by `test_forecast.py`).

What "skill vs coin flip" means for this call, read off the holdout (2019-01-02 → 2026-11-02):

| universe | sessions / rows | up-frequency | base-rate Brier | coin flip | base rate's skill vs coin flip |
|---|---|---|---|---|---|
| SPY, next close | 1,970 | 55.08% | 0.24767 | 0.25000 | +0.93% |
| SPY, rest of day from the open | 1,970 | 54.21% | 0.24852 | 0.25000 | +0.59% |
| panel (15 ETFs + 20 names), next close | 68,950 | 52.60% | 0.24950 | 0.25000 | +0.20% |
| panel, rest of day from the open | 68,950 | 51.45% | 0.24996 | 0.25000 | +0.02% |

So the honest daily direction is "up, about 52–55%", and its entire edge over a coin flip is the market's drift in this sample: under 1% Brier skill for SPY, about 0.2% for single names. Any page that shows this call must say exactly that.

## Pre-registration (verbatim, written 2026-11-04 ~03:50 UTC before any result was computed)

> Question: does any candidate feature beat the point-in-time base rate (Brier) on a purged walk-forward holdout, after a multiple-testing correction?
>
> Targets:
> - PRIMARY y = 1[C_{t+1} > C_t], decision at the t close (the scored daily call). Ties (flat) count as not-up, as in the weekly scorer.
> - INTRADAY y = 1[C_{t+1} > O_{t+1}] (rest of day from the open) and y = 1[C_{t+1} > P_{10:30}] (rest of day from the first hour's close).
>
> Features (point-in-time; vol-scaled by the name's 20-session daily vol where they are returns; each clipped at +/-4):
> - F1 gap_t = log(O_t / C_{t-1}) (the latest overnight gap; known at the t close)
> - F2 imom_t = log(C_t / O_t) (the latest session's open-to-close move)
> - F3 dvix_t = log(VIX_t / VIX_{t-1}) (market-level)
> - F4 breadth_t = share of regime_engine.BREADTH_BASKET (11 sector SPDRs + IWM + MDY, members with >= 200 sessions) above their 200-day SMA, - 0.5 (the repo's existing definition, not a new one)
> - F5 gex_t = SqueezeMetrics GEX trailing-252 percentile rank R_t - 0.5 (P1's definition; index-level; 2012-05 onward after burn-in)
> - G1 ovn_gap = log(O_{t+1}/C_t) for the rest-of-day-from-open target
> - G2 first-hour momentum = log(P_{10:30}/O_{t+1}) for the rest-of-day-from-10:30 target (60m bars, ~725 sessions only)
>
> Model: ridge logistic (forecast.fit_logistic, l2 = 1) on one feature; reference = the intercept-only model (training up-frequency = the point-in-time base rate). Walk-forward expanding window, refit every 21 sessions, 5-session embargo (>= the 1-session label horizon), >= 504 training sessions. SELECTION origins up to 2018-12-31, HOLDOUT 2019-01-02 .. 2026-11-02 (GEX from its first possible date; G2 = first half / second half of the 60m sample).
>
> Universes: SPY alone; pooled panel (15 TRAIN_BASKET ETFs + 20 names).
>
> Tests (m = 14): T1-T5 F1..F5 on SPY; T6-T10 F1..F5 on the panel; T11 the combination of all features that pass individually (only if >= 2 pass; counted anyway); T12 G1 SPY; T13 G1 panel; T14 G2 SPY + 60m panel.
>
> PASS (per test): holdout dBrier = Brier(model) - Brier(base) < 0, AND one-sided block-bootstrap p < 0.05/14 = 0.0036 (resample 20-session blocks of whole days, so the panel's cross-sectional correlation is kept), AND dBrier < 0 on SELECTION too (sign consistency). Ship only features that PASS. If none pass, the daily direction stays at the shrunk base rate and a test pins it.
>
> Priors (stated before running): T1 5%, T2 5%, T3 7%, T4 5%, T5 5%; panel versions about the same; T12 10%; T14 10%. P(at least one honest pass) ~25%. Expected scale: SPY base-rate Brier ~0.249 (up-frequency ~0.53-0.54); a detectable improvement needs dBrier around -0.0015 or better.

Implementation details fixed while writing the script, before it ran: F3 is divided by the trailing-252-session std of daily VIX log changes, and F4/F5 are scaled to [-1, 1] (2 x (share - 0.5)), so the ridge penalty treats every feature on a comparable scale. One forced deviation, logged before G2 was run: the 60-minute sample is only 722 sessions, so T14 cannot meet the 504-session minimum inside a walk-forward. It trains on the first half (expanding, refit every 21 sessions) and is scored on the second half (2025-05-21 → 2026-11-03). Its SELECTION sign check trains on the first quarter and scores the second.

## Data (free, keyless; nothing raw committed)

| input | source | span used |
|---|---|---|
| daily OHLC, split-adjusted (not dividend-adjusted, the Alpaca `adjustment=split` convention) | Yahoo v8 chart API — the keyless endpoint `backtest_v2._yahoo_bars` already uses | 1999-01-04 → 2026-11-03 |
| VIX close | Cboe `VIX_History.csv` (the series `server/cboeVix.ts` archives) | 1999 → 2026-11-03 |
| GEX | SqueezeMetrics `DIX.csv` — research use only, licence unverified, never committed (same posture as P1) | 2011-05-02 → 2026-11-03 |
| 60-minute bars | Yahoo v8 chart API, regular session | 2023-12-06 → 2026-11-03 (722 complete sessions) |

- Alpaca: no keys in this sandbox, so no Alpaca data was read. Production direction uses no market data beyond Alpaca's daily bars (the base rate), so nothing here needs an Alpaca replication before shipping — nothing shipped.
- Survivorship: the 20 single names exist today (AAPL AMZN GOOG IBM GS MSFT JPM XOM PFE KO WMT NVDA AMD TSLA NFLX BA CAT F INTC DIS). That biases the panel's base rate up, not any feature's dBrier in a known direction. SPY tests carry no survivorship bias.
- Lookahead: every feature at origin t uses closes/opens <= t (G1 and G2 use the target day's open / 10:30 price, known at decision time for their rest-of-day targets). VIX settles at 16:15 ET, after the 16:00 equity close; a decision "after the close" can use it. GEX_t is built from open interest OCC publishes the next morning, so F5 is known before the t+1 open but not at the t close. That is a production-timing problem only, and it is moot because F5 failed.

## Results — every test (HOLDOUT unless stated; dBrier = model − base rate, negative = better)

| test | feature | universe | target | holdout rows | Brier base | Brier model | dBrier ×1e4 [95% CI] | one-sided p | SELECTION dBrier ×1e4 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | F1 overnight gap | SPY | next close | 1,970 | 0.24767 | 0.24772 | +0.45 [−0.80, +1.64] | 0.740 | +1.33 | FAIL |
| T2 | F2 intraday momentum | SPY | next close | 1,970 | 0.24767 | 0.24790 | +2.32 [−2.77, +7.73] | 0.810 | −2.10 | FAIL |
| T3 | F3 VIX change | SPY | next close | 1,970 | 0.24763 | 0.24777 | +1.40 [−0.63, +3.35] | 0.913 | +0.43 | FAIL |
| T4 | F4 breadth | SPY | next close | 1,970 | 0.24762 | 0.24764 | +0.20 [−2.16, +2.02] | 0.486 | +2.75 | FAIL |
| T5 | F5 dealer gamma | SPY | next close | 1,970 | 0.24754 | 0.24772 | +1.79 [−2.18, +6.80] | 0.829 | +0.85 | FAIL |
| T6 | F1 overnight gap | panel | next close | 68,950 | 0.24950 | 0.24951 | +0.19 [−0.06, +0.53] | 0.935 | +0.25 | FAIL |
| T7 | F2 intraday momentum | panel | next close | 68,950 | 0.24950 | 0.24955 | +0.57 [−1.40, +2.77] | 0.740 | −1.62 | FAIL |
| T8 | F3 VIX change | panel | next close | 68,950 | 0.24947 | 0.24954 | +0.64 [−1.07, +2.27] | 0.785 | −0.21 | FAIL |
| T9 | F4 breadth | panel | next close | 68,950 | 0.24947 | 0.24949 | +0.14 [+0.01, +0.24] | 0.985 | +1.22 | FAIL |
| T10 | F5 dealer gamma | panel | next close | 68,950 | 0.24936 | 0.24928 | −0.77 [−3.61, +2.47] | 0.332 | +0.93 | FAIL |
| T11 | combination of passes | — | — | — | — | — | — | — | — | NOT RUN (no individual pass) |
| T12 | G1 today's gap | SPY | rest of day from the open | 1,970 | 0.24853 | 0.24840 | −1.29 [−6.96, +4.50] | 0.340 | +1.88 | FAIL |
| T13 | G1 today's gap | panel | rest of day from the open | 68,950 | 0.24995 | 0.25006 | +1.03 [−0.02, +2.14] | 0.972 | +0.39 | FAIL |
| T14 | G2 first-hour momentum | SPY QQQ IWM DIA + 20 names (60m) | rest of day from 10:30 | 8,664 | 0.24989 | 0.24993 | +0.35 [−0.14, +0.95] | 0.909 | +1.40 | FAIL |

- Bonferroni bar: one-sided p < 0.05 / 14 = 0.0036 (10,000 day-block resamples). Nothing came within two orders of magnitude of it.
- The three tests with a negative holdout dBrier (T10 GEX panel −0.77e-4, T12 SPY gap −1.29e-4; T14's SPY-only slice −0.14e-4) all had a POSITIVE dBrier on SELECTION, so even their sign is not stable. They are noise, not a near-miss. Read per REASONING #4: 13 variants were run, and the best holdout number of 13 noisy draws is expected to look like this.
- Fitted slopes are tiny (last refit): F1 SPY −0.014, F2 SPY −0.059 (a faint reversal of the latest open-to-close move), F3 SPY +0.009, F4 SPY +0.037, F5 SPY −0.056, G1 SPY +0.069 (a gap that continues rather than fades). None of them moves a probability by more than ~1 point for a typical feature value. The ridge already shrinks them, and the holdout still prefers zero.
- Moving-block bootstrap note: the moving (non-circular) block bootstrap under-samples the first and last 19 days of the holdout. That is why T4's p (0.486) sits slightly below 0.5 although its point estimate is positive. No verdict is anywhere near the edge where it matters.

### Base rates, and the production definition

- The table at the top is the study's own reference: each universe's expanding-window up-frequency.
- The production base rate is a fixed definition: the pooled up-frequency of `forecast.TRAIN_BASKET`'s last 756 daily returns. It was chosen in the design, not from these results. Diagnostic on the same holdout:

| reference | SPY Brier | panel Brier |
|---|---|---|
| production basket-756 base rate | 0.24794 | 0.24974 |
| the name's own expanding up-frequency | 0.24767 | 0.24950 |
| coin flip | 0.25000 | 0.25000 |

- The definitions differ by about 3e-4 Brier, which is invisible next to a single day's noise.
- In the holdout the production base rate ranged 51.5% → 55.9%. It read 54.5% on 2026-11-02.
- The page shows the base rate's skill vs a coin flip live, by n, with the `thin` flag. It does not claim the study's +0.93%.

### By regime (HOLDOUT dBrier ×1e4, descriptive; negative = the feature helped)

VIX tercile cut points come from the pre-2019 sample: 14.99 / 21.71. Regime labels are the bot's 5-level classifier (`regime_util.classify_regime_5level`), fed as in P1.

| test | VIX low | VIX mid | VIX high | BULL | NEUTRAL | CAUTION | BEAR | PANIC |
|---|---|---|---|---|---|---|---|---|
| T1 F1 SPY | +1.2 (428) | +0.9 (977) | −0.9 (565) | +1.1 (700) | +0.7 (471) | −0.5 (246) | −0.4 (489) | +1.5 (64) |
| T2 F2 SPY | −1.7 | +6.2 | −1.4 | +1.9 | +3.4 | +11.9 | +1.5 | −31.1 |
| T3 F3 SPY | +2.7 | +2.5 | −1.5 | +3.4 | +1.9 | +4.9 | −0.6 | −22.7 |
| T4 F4 SPY | +0.1 | +0.6 | −0.4 | −0.1 | −0.1 | +1.4 | −0.0 | +2.8 |
| T5 F5 SPY | +6.4 | −3.0 | +6.6 | +5.8 | −3.0 | −2.0 | +0.8 | +14.6 |
| T6 F1 panel | +0.1 | +0.2 | +0.3 | −0.0 | +0.1 | −0.2 | +0.4 | +3.3 |
| T7 F2 panel | +1.1 | +2.7 | −3.5 | +0.8 | +2.8 | +2.2 | +0.2 | −21.5 |
| T8 F3 panel | +2.2 | +1.6 | −2.2 | +1.8 | +2.0 | +4.1 | −1.1 | −22.1 |
| T9 F4 panel | +0.0 | +0.2 | +0.2 | +0.0 | +0.0 | −0.1 | +0.5 | +0.1 |
| T10 F5 panel | +2.4 | −3.1 | +0.9 | +0.7 | −2.3 | −3.1 | −1.3 | +8.1 |
| T12 G1 SPY | +10.0 | −2.4 | −8.0 | +4.6 | −1.2 | −1.4 | −13.9 | +30.8 |
| T13 G1 panel | −0.7 | +1.1 | +2.3 | +0.2 | −0.2 | +0.8 | +3.4 | +2.4 |

- No feature helps consistently in any regime.
- The only large cells are PANIC, at n = 64 days. In PANIC, F2 (intraday momentum) and F3 (VIX change) show −31 / −22 ×1e4 for SPY and −21 / −22 for the panel. That is a plausible mechanism: reversal after a VIX spike or a hard down day in a panic tape. But it is 64 days, post-hoc, and one cell of 8 × 13. It is recorded as a hypothesis to test on a different sample, not as a finding.

## What ships

- `DAILY_DIRECTION_FEATURES = ()`. The daily direction is the base rate, shrunk all the way.
- The weekly model (fc-logit-v1, holdout Brier 0.2511 vs 0.2508 base rate) is untouched by this study.
- The UI reports the daily call's Brier against BOTH the base-rate reference and 0.25, with n and the thin flag. That makes "no skill beyond the drift" the visible default, not something to dig for.

## Caveats

1. Free daily data only. Production bars are Alpaca's IEX/SIP split-adjusted bars. Gaps (F1/G1) built from an IEX-only open would be noisier than these consolidated ones, which is another reason not to ship them.
2. The 20-name panel is mega/large-cap and survivorship-biased. "Fish where whales can't" names (small caps) were not in the panel. A gap/momentum effect could exist there, but it would need a point-in-time small-cap universe with delisted names — a different study, with the price panel's survivorship-free bars.
3. GEX: licence unverified and ladder gate 1 still open (P1 caveats). It would not have been shippable on /data even if it had passed. It did not pass.
4. T14 has low power: 361 holdout sessions.

## Follow-up hypotheses (filed as text here for the parent session to log; not tested)

- PANIC-regime reversal: does F2 or F3 predict the next close in the bot's PANIC regime on a sample disjoint from 2019–2026? Use the 2000–2012 PANIC days (rough expectation 150–250 days), pre-registered, one test. Prior 15%.
- Small-cap gap fade: G1 on a point-in-time small-cap panel from the survivorship-free price panel (`server/pricePanel.ts`). Prior 15%. Ladder path: gate 2 before any display.
