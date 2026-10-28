# Floor basket: keep, replace or drop — decision memo (2026-10-27, Phase 6 part C)

Provenance: human-directed Phase 6 research. Docs only; no config changed. The basket's COMPOSITION is the human's stated thesis (system_config.py FLOOR_BASKET: QQQ .50 / SMH .15 / KWEB .10 / VXUS .15 / CASH .10, "tech-dominance thesis"); this memo is the evidence and a recommendation, not a change. Inputs: research test 4 (`research/floor_basket_vs_index_2026-10-20.md`, experiments.md 2026-10-20 ~22:30Z), the open questions filed 2026-10-14 (tracking error) and 2026-10-20 (exposure schedule), and the live sleeve P&L (audit 2026-10-27 §7.3, experiments.md 2026-10-20 / 10-26 reads).

## Recommendation

| Option | Decision | Strength of evidence |
|---|---|---|
| **Drop** the floor | **No.** | Strong in direction: it is the only positive sleeve live, and every floor arm in the 12-year simulation compounded positively at Sharpe 0.67-0.84; dropping it turns the account into cash plus the sleeves that are losing. |
| **Replace** the composition (e.g. SPY-only or QQQ-only) | **No change on current evidence — keep the human's basket.** | Insufficient: the composition arms are statistically indistinguishable on Sharpe, and the live sample cannot separate them for decades (below). |
| **Keep**, and test the EXPOSURE SCHEDULE instead | **Yes — this is where the measurable gap is.** | Moderate (one 12-year simulated history): the regime gating, not the basket, cost ~6 pts of CAGR per year at the same Sharpe. Settled by the pre-registered 4-arm ablation below, not by argument. |

## What the evidence says

### 1. Composition (research test 4, simulation 2014-11 → 2026-10, 2,990 days, 5 bp/side, live exposure schedule applied to every arm)

| Arm | CAGR | Max DD | Sharpe ± SE | Sharpe − same-exposure SPY [LW 95%] |
|---|---|---|---|---|
| basket (live) | 7.82% | −21.6% | 0.733 ± 0.291 | −0.109 [−0.375, +0.156] |
| qqq_only | 9.76% | −23.6% | 0.838 ± 0.291 | −0.005 [−0.247, +0.238] |
| fifty_fifty | 6.26% | −20.1% | 0.698 ± 0.291 | −0.145 [−0.341, +0.051] |
| spy_core (QQQ+SMH → SPY, intl kept) | 5.46% | −19.4% | 0.667 ± 0.290 | −0.175 [−0.342, −0.008] |
| SPY at the same exposure | 7.41% | −20.1% | 0.843 ± 0.291 | 0 |
| SPY buy-and-hold | 13.58% | −33.7% | 0.816 ± 0.291 | — |

Read:
- Every composition arm's Sharpe is within one standard error of every other; only spy_core's interval excludes zero (it is WORSE than plain SPY). The DSR bar for 4 arms (0.08) is cleared only by qqq_only.
- The CAGR ordering (qqq_only > basket > fifty_fifty > spy_core) is this decade's US-tech outperformance. The basket's cost vs a pure US core is its 25% international slice (KWEB + VXUS): about −1.9 pts of CAGR per year in-sample at roughly the same Sharpe. That is one decade, one regime of relative performance (2000-2010 ran the other way); it is the diversification bet the human chose, not a measured flaw.
- Discounts (from the test's own HONESTY section): pre-2018 regime labels degraded (VXX history), costs stated not measured, one history, four correlated arms.

### 2. The exposure schedule — the big number

SPY held at the floor's own regime-driven exposure (.70 / .90 / .35 / 0 / 0 by regime, plus the GLD leg and trend caps) made **7.41%/yr vs 13.58%** buy-and-hold, at Sharpe **0.843 vs 0.816** — the gating roughly halved compounding for no risk-adjusted gain, because the BEAR label (639 days, floor at 0) coincided with the strongest rebound days (+22.9%/yr annualized for buy-and-hold on those days). Churn: 479 rebalances, turnover 167× over the decade (≈ 1.2%/yr at 5 bp/side); live, the floor made ~2,755 fills between May and October. Under the GOAL's log-wealth-vs-SPY framing, equal Sharpe at higher exposure is the better arm. This is a RULE-REVIEW threshold family (open_questions.md 2026-10-20), not a composition question.

### 3. Live (paper account, window 2026-05-07 → 2026-10-27, ~120 trading days)

- Floor sleeve, `/api/diag/sleeves`: realized **+$1,554**, unrealized **+$1,199** (audit §7.3, 2026-10-27 ~19:10Z) — net ≈ +$2.75k. The sleeves it sits beside: options −$9,177 realized, stocks −$499.
- The number moves a lot week to week: the 2026-10-20 read (full-history basis, v1.0.1129) was realized +$5,628 / open −$2,092 (net +$3.5k); 10-26 read realized +$1,554 / open +$141 (net +$1.7k); October's month went from +$1,199 to −$2,875. About $4k of realization in a week on a ~$75k book is what one regime flip selling down losing lots looks like (a 1% day is ~$750); it is NOT reconciled here and the recommendation does not depend on it.
- Against SPY: net ≈ +3.4-3.9% on ~$70-80k of floor capital vs SPY total return +5.46% over the same window, but the floor's exposure was 70-90% plus 8% GLD, so the like-for-like gap vs same-exposure SPY is roughly −1 pt — a single path of 120 days.

## Sample-size honesty — why live data cannot settle composition

- Tracking error of the basket vs same-exposure SPY is ~5.1%/yr (sim). Over the live 0.48 years that is ~3.5 pts of 1-sd noise; the observed ~1-2 pt gap is under 0.6 sd. Nothing can be read from it in either direction.
- To show a 2 pt/yr composition shortfall at t = 2 with 5% TE needs (2 × 5 / 2)² = **25 years** of live data. To separate two correlated arms (ρ ≈ 0.9) by 0.1 in annual Sharpe needs SE(ΔSR) ≈ √(2(1−ρ)/T) = 0.05 → **T ≈ 80 years**. Composition is a prior/thesis decision, not a statistical one, at any horizon this account will see.
- The live sleeve P&L is useful for one thing: the HONESTY METRIC (live vs simulated divergence). There is no standing metric for it yet — the sleeve ledger reports dollars, not a daily return series vs a same-exposure benchmark.

## The test that settles the open part (pre-registered here, restating the 2026-10-20 filing with exact pass rules)

**Question:** does the exposure schedule cost compounding without buying risk reduction?
**Engine:** `scripts/floor_basket_vs_index.py` (research test 4's engine), composition FIXED at the live basket (the human's choice), 2014-11 → latest, regime labels from `backtest_v2.regime_series`.
**Arms (N = 4, declared now):** (a) live schedule (.70/.90/.35/0/0); (b) flat 70% floor, no regime gating, trend caps kept; (c) live schedule with SPY_FLOOR_BEAR 0.35 and SPY_FLOOR_PANIC 0.20 ("never fully out"); (d) live schedule with the 2026-10-13 debounce modelled (3-day confirmation).
**Costs:** 2 bp and 5 bp per side, both reported; results must hold at 5 bp.
**Pass rule for an arm vs (a):** CAGR ≥ (a) + 2.0 pts AND Sharpe ≥ (a) − 0.10 AND max drawdown no worse than (a) by more than 5 pts AND the arm clears the DSR bar for N = 4; reported per regime (BULL / NEUTRAL / CAUTION / BEAR / PANIC) with Lo standard errors. Arm (d) passes on churn alone if it cuts rebalances by ≥ 1/3 with CAGR and Sharpe within ±0.3 pt / ±0.05 of (a).
**Priors (from the filing):** 70% that "never fully out" (c) lifts CAGR ≥ 2 pts at a Sharpe within 0.1; 60% that the debounce (d) removes a third of the churn.
**If an arm passes:** one threshold at a time (RULE REVIEW for risk limits), prior value logged in the commit, rollback trigger = live max drawdown exceeding the backtest's by 5 pts within 90 days. The frozen kill switches remain the survival mechanism.
**If none passes:** keep the schedule; record the verdict.

**Measurement to add (own [RULE-REVIEW] PR, before any change):** a daily floor-sleeve return series vs SPY at the floor's own exposure, on the sleeves probe, so live-vs-simulation divergence of the floor is a number (expected band: ±5%/yr TE). This monitors honesty; it does not decide composition.

## For the human

- Composition: the evidence neither supports nor rejects your basket on risk-adjusted terms. The in-sample price of the international slice was about 1.9 pts of CAGR per year over a decade that favored US tech. Keeping it is a diversification bet against that decade repeating. Replacing it with QQQ would have been the better decade, with a deeper drawdown. No session will change it.
- The floor is not the account's problem: it is the only sleeve making money. The account's gap to SPY is the options sleeve (−$9.2k) and, structurally, the floor's exposure schedule, which the declared ablation above settles.
