# Floor study (2026-11-05): which passive core beats buy-and-hold SPY?

Human, 2026-11-05: "why have a floor like the QQQ could we not build our own floor for the system with leverage like the qqq that is better? ... the goal [is to] beat the S&P".

Script: `scripts/floor_study.py`. Results: `research/floor_study_2026-11-05.json`. The data is Yahoo daily adjusted closes (total return) plus ^IRX T-bill yields. It is not committed; the script records the exact inputs.

## Verdict

**Volatility-managed SPY (V1) is the only design that passes the pre-registered ship rule.**
- Mean log growth: 10.41% vs SPY's 10.26%.
- Sharpe: 0.573 vs 0.513.
- Max drawdown: −48.4% vs −55.2%.
- Worst year: 2008 at −22.6% vs −36.8%.
- It beat SPY in 2 of 3 subperiods: 2008–16 and 2017–26, but not 1993–2007.
- 2013-08 → 2026-11, the window where the current basket can be tested: CAGR 14.2% vs SPY 13.9%, max drawdown **−19.5%** vs −33.7%.

**Leverage on its own adds return but not risk-adjusted return.**
- Leverage rotation (V2): +1.4 pp/yr of log growth.
- Constant 1.25x (V3): +1.2 pp/yr.
- Both have Sharpe ≈ SPY's (0.50 vs 0.51), and V3's max drawdown is −65%. Both fail the Sharpe test.

**The current floor's basket is not what lost money.**
- Static, 2013+: CAGR 15.2% vs SPY 13.9%, Sharpe 0.80 vs 0.75.
- The live floor's 2026 under-performance comes from its regime gate (90% → 70% → 35% → 0% plus trend caps), which re-trades the basket (2,916 floor fills since May), and from its 65% tech/semis tilt on tech-led down days.
- QQQ alone since 1999: +1.9 pp/yr over SPY, at −83% max drawdown and the same Sharpe (0.43 vs 0.42). That is concentration, not a better risk-adjusted core.

## Results (1993-02-01 → 2026-11-04, 8,499 sessions, after financing and costs)

| design | CAGR | vol | Sharpe | max DD | log growth | worst year | rolling 3y beats SPY |
|---|---|---|---|---|---|---|---|
| B0 SPY | 10.81% | 18.51% | 0.513 | −55.2% | 10.26% | 2008 −36.8% | — |
| **V1 vol-managed SPY (16% target, ≤1.5x)** | 10.97% | 16.13% | **0.573** | **−48.4%** | 10.41% | 2008 **−22.6%** | 52.3% |
| V2 2x above 200-day SMA, else T-bills | 12.35% | 23.96% | 0.503 | −52.1% | 11.64% | 2000 −35.1% | 69.5% |
| V3 constant 1.25x | 12.09% | 23.14% | 0.502 | −64.8% | 11.41% | 2008 −45.5% | 74.7% |

**By subperiod** (log growth / Sharpe / max DD; SPY first):

| subperiod | SPY | V1 | V2 | V3 |
|---|---|---|---|---|
| 1993–2007 | 9.81 / 0.44 / −47.5 | 8.93 / 0.40 / −48.4 | 8.18 / 0.30 / −52.1 | 10.59 / 0.43 / −57.6 |
| 2008–2016 | 6.83 / 0.42 / −52.3 | 8.03 / 0.56 / −31.3 | 11.32 / 0.61 / −37.9 | 7.50 / 0.41 / −61.8 |
| 2017–2026 | 14.10 / 0.73 / −33.7 | 14.84 / 0.85 / −19.5 | 17.20 / 0.73 / −36.6 | 16.23 / 0.72 / −40.8 |

**2013-08 → 2026-11** (the current basket exists only from here):

| design | CAGR | Sharpe | max DD |
|---|---|---|---|
| current basket (static NEUTRAL weights) | 15.20% | 0.798 | −31.3% |
| SPY | 13.93% | 0.746 | −33.7% |
| QQQ | 19.06% | 0.844 | −35.1% |
| V1 | 14.19% | 0.796 | −19.5% |
| V2 | 17.19% | 0.717 | −36.6% |
| V3 | 16.33% | 0.733 | −40.8% |

**Exposure.**
- V1: mean 1.13x. Levered on 64% of days. 800 rebalances in 33 years, about 24 a year.
- V2: on 75.6% of days, with 215 switches.

## Robustness (post-hoc, reported, not used to choose)

V1's two parameters, swept across 5 targets × 3 caps:
- **All 15 points have Sharpe 0.547–0.575**, above SPY's 0.513. The improvement is the volatility management, not a lucky setting.
- Log growth rises with the target:

| target vol | cap | log growth | Sharpe | max DD | realised vol |
|---|---|---|---|---|---|
| 12% | any | ≈ 9.0% | ~0.56 | −36% | 12.5–13.4% |
| 16% | 1.5x | 10.4% | 0.573 | −48% | 16.1% |
| 18% | 2x | 11.3% | 0.561 | −54% | 18.9% (≈ SPY's) |
| 20% | 2x | 11.8% | 0.558 | −59% | 20.5% |

So "the same risk as SPY, more return" sits near an 18% target with a 2x cap. That was not pre-registered. It is a candidate for its own pre-registered follow-up, and a risk-appetite choice for the human.

## Caveats

- **Daily close-to-close simulation.** Signals at close t, held from t to t+1, no lookahead. It ignores intraday execution at the close.
- **Financing** is the T-bill rate + 1% on any exposure above 1x. Trading cost is 2 bp of notional traded. V2 also pays a 0.9%/yr fund fee.
- **The current basket is simulated static** at its NEUTRAL weights. The live regime gate and trend caps are NOT simulated (stated in the pre-registration).
- **Survivorship and lookahead:** index ETFs only, no stock selection. The scaling constant is a fixed 16% chosen a priori, not fitted on the full sample. This answers the Liu–Tang–Zhou critique of volatility-managed portfolios.
- **Out-of-sample literature:** volatility timing of the MARKET portfolio is the most robust case in Moreira & Muir (2017). Cederburg et al. (2020) find the benefit weaker or absent for many factor portfolios. This study only claims the market case.

## Pre-registration (verbatim; written 2026-11-05 ~15:00 UTC, before any long-history result)

> Only data seen before writing this: the last 3 months of daily closes for SPY QQQ SMH VXUS KWEB GLD QLD SSO TQQQ UPRO (used for the live attribution, not for any variant below).
>
> **Variants (all of them; nothing else is tried)**
> - B0 SPY buy-and-hold (the benchmark).
> - B1 QQQ buy-and-hold (reference: the current floor's core).
> - C0 current floor, static approximation: 45% QQQ, 13.5% SMH, 9% KWEB, 13.5% VXUS, 8% GLD, 11% cash at the T-bill rate, rebalanced monthly. The live regime gating is NOT simulated. 2013+ only.
> - V1 volatility-managed SPY: w_t = min(1.5, 0.16 / sigma_t), sigma_t from the last 20 daily returns, known at close t. Rebalance only when |w_target − w_held| > 0.10.
> - V2 leverage rotation: 2x SPY (synthetic, 0.9%/yr fee) when SPY's close is above its 200-day SMA, else T-bills.
> - V3 constant 1.25x SPY, rebalanced daily.
>
> Financing above 1x: the T-bill rate + 1.0%/yr. Cash earns the T-bill rate. Trading cost: 2 bp of traded notional.
>
> **Priors**
> - V1: CAGR ≈ SPY + 1–2 pp, Sharpe ≈ SPY + 0.10–0.15, max DD ≈ SPY's or a few pp better. Beats SPY's Sharpe in all three subperiods with probability ~50%.
> - V2: CAGR ≈ SPY + 2–5 pp; smaller 2008 drawdown; whipsaw in choppy years; a large hit in fast crashes; Sharpe ≈ SPY + 0.10.
> - V3: CAGR ≈ SPY + 1.5–2 pp; Sharpe ≈ SPY's or slightly lower; max DD ≈ 1.25 x SPY's.
> - C0 (2013+): CAGR within ±2 pp of SPY; higher volatility and drawdown than SPY; lower Sharpe than QQQ alone.
>
> **Ship rule:** a design may replace the live floor only if, over its full window:
> 1. mean log growth > SPY's;
> 2. Sharpe ≥ SPY's;
> 3. max DD no more than 10 pp worse than SPY's;
> 4. it beats SPY's log growth in at least 2 of the 3 subperiods (1993–2007, 2008–2016, 2017–2026).
>
> Among designs that pass, pick the highest log growth. Exploratory: kill date 2027-02-05; rollback trigger: live floor tracking vs its own rule > 3 pp over any 20 sessions, or account drawdown beyond the backtest's 95th-percentile 3-month drawdown. Four candidate designs, one fixed parameter set each; any tuned variant is a new study.

**How the priors held up.**
- V1 delivered smaller gains than the prior (+0.15 pp/yr CAGR, +0.06 Sharpe) and beat SPY in 2 of 3 subperiods rather than all three. Its drawdown improvement was larger than expected.
- V2's CAGR gain came in at +1.5 pp, below the prior's 2–5 pp, and its Sharpe is at SPY's rather than above it.
- V3 matched its prior.
- C0 beat its prior: +1.3 pp CAGR with a higher Sharpe than SPY over 2013+.
