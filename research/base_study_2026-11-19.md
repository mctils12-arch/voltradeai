# Base study (2026-11-19): is there a better base than the S&P 500, and what should the idle cash do?

Owner, 2026-11-19 (verbatim): "why not hold the top names or the best names in the market and build a better base of the s&p or am i thinking of this wrong and why hold that much cash on the side why not sell out of positions".

Script: `scripts/base_study.py`. Tests: `test_base_study.py`. Precedent: `scripts/floor_study.py` and `research/floor_study_2026-11-05.md` (same loader, same metrics, same style).

STATUS OF THIS FILE: the section below was written and committed BEFORE any price was fetched. The results sections are appended after it and never edit it.

## Pre-registration (written 2026-11-19 ~20:00 UTC, before any price was fetched)

### What I already knew when writing this (not blind)

- No price series was fetched or opened for this study before this commit.
- I carry general knowledge of the period: 2013-2026 was one long bull market led by a handful of mega-cap growth stocks; equal weight and low volatility lagged; momentum funds had a strong 2023-24; growth fell harder than the index in 2022. The priors below are informed by that. They are priors about SIZE, and the decision rule is what protects against reading the sample as proof.
- From the floor study (2026-11-05, total-return data): vol-managed SPY 1993-2026 log growth 10.41% vs SPY 10.26%, Sharpe 0.573 vs 0.513, max drawdown -48.4% vs -55.2%. With its idle cash at 0% the log-growth edge disappears (10.16% vs 10.25%, system_config FLOOR_VM_CASH note).

### The menu (fixed; nothing is added after results are seen)

Every candidate is one or two listed ETFs, so the test has no survivorship bias (a fund's price already contains its deletions) and is net of real fund costs. No stock-level backtest from today's index members.

| id | base | tickers |
|---|---|---|
| B0 | S&P 500, the benchmark | SPY 100% |
| B1 | the live rule | vol-managed SPY/SSO on 85% of equity, the floor's own idle part in a T-bill fund, the other 15% in cash at 0% |
| B2 | "top names" (mega-cap) | OEF, XLG, MGK (three separate candidates) |
| B3 | momentum | MTUM, SPMO (two) |
| B4 | quality | QUAL, SPHQ (two) |
| B5 | momentum + quality | 50% MTUM + 50% QUAL, rebalanced on the last session of each calendar year |
| B6 | equal weight (the anti-"top names" control) | RSP |
| B7 | minimum volatility (control) | USMV, SPLV (two) |

The cash question, separately, as a 2 x 3 grid run through one simulator:

| | (i) 15% in cash at 0% (live) | (ii) no reserve: 100% of equity in the base | (iii) the 15% in a T-bill fund |
|---|---|---|---|
| vol rule OFF (E* = 1 always) | B0-i | B0-ii = SPY buy-and-hold = the benchmark | B0-iii |
| vol rule ON (E* = min(1.5, 16% / sigma20)) | B1-i = B1, the live rule | B1-ii | B1-iii |

"Does the volatility rule earn its keep" = ON vs OFF in the same column, on the modern sample (real SSO, from SSO's 2006 inception) and on the long 1993+ SPY sample the floor study used (a synthetic 2x fund, formula below).

COUNT: 11 base candidates + 5 grid cells that are not the benchmark = **16 series tested against SPY**. Each is shown on up to three samples (its own longest, the all-candidate common sample, and for the grid the long 1993+ sample). Significance is judged ONCE per series, on its own longest sample, in one family of 16.

### Data

- Source: the Yahoo chart API, daily, the same endpoint `backtest_v2._yahoo_bars` calls. `backtest_v2.fetch_bars` itself is Alpaca-first with `adjustment: split` (price-only), Yahoo fallback with DIVIDEND-ADJUSTED closes. This session has no Alpaca keys, so `fetch_bars` would silently return total-return closes. The study therefore saves the raw Yahoo JSON and reads it with `floor_study.load(path, adjusted=False)` for PRICE-ONLY (split-adjusted, dividends excluded) closes and `adjusted=True` for total return.
- DECISION BASIS: price-only on both sides. The paper account receives no dividends and its benchmark is SPY price return (research/open_questions.md [DECIDED 2026-11-05]). Total return is reported next to it as a check, never used to pick.
- What price-only does: it removes each fund's dividend yield. A low-yield fund (mega-cap growth, momentum) keeps more of its return than SPY does; a high-yield fund (low volatility, equal weight) loses more. That tilt is real for THIS paper account and would NOT exist in a real account. A candidate that passes on price-only but not on total return is labelled a PAPER ARTIFACT and is not recommended.
- T-bill rate: ^IRX (13-week yield), forward-filled, `floor_study.daily_rf`. T-bill fund = that rate minus 0.19% a year (BOXX's fee; BOXX itself only exists from late 2022, so the fund is modelled, not read). No floor at zero: in zero-rate years the modelled fund loses its fee.
- Last session used: 2026-11-18. Every ticker's first date is reported. A ticker whose fetch fails is DROPPED and named, never substituted.
- Samples: (a) each candidate's OWN LONGEST sample = from the first session on which the candidate and SPY (and SSO for B1) all have a price; (b) the ALL-COMMON sample = from the latest inception among all tickers (expected: SPMO, October 2015); (c) the long sample = SPY from 1993 (grid only).
- Every sample is a fresh start: fully invested at the first close, no entry cost for anyone.

### Simulation rules

- Timing as the floor study: every decision at close t uses closes <= t and is held from close t to close t+1.
- B1 and the grid: the repo's own planning code, unchanged — `floor_vol_managed.plan()` (realized_vol, target_exposure, holdings_for, the 0.10 band) and `floor_vol_managed.cash_plan()` (the T-bill-fund parking with its 5% / $2,000 band), parameters read from `system_config` FLOOR_VM_*. Start equity $100,000. Vol rule OFF = the same code with the target pinned at 1.0. Column (ii) = FLOOR_VM_CAPITAL_PCT 1.0. Column (iii) = everything outside SPY/SSO goes to the T-bill fund. The flow-model multiplier is 1.0 (not simulated). The reserve is assumed never used by an option trade, which is what has happened live.
- Long sample (1993+): SSO did not exist before 2006, so the 2x fund is synthetic there: daily return = 2 x SPY total return - (T-bill + 1%/yr) - 0.9%/yr fee (the floor study's financing and fee constants). Its error is measured against real SSO over 2006+ and reported.
- Trading cost: 2 bp of traded notional (the floor study's COST_BP) on every rebalance trade of B1, the grid, and B5.
- Tax (an ESTIMATE): FIFO lots; a lot held more than 365 days is long-term (15%), otherwise short-term (24%) — server/taxLedger.ts DEFAULT_TAX_RATES and LONG_TERM_MS. Each calendar year's realized gains and losses are netted (short against long, losses carried forward); the tax is paid on the first session of the next year. The T-bill fund's yearly return is taxed at the short-term rate. No dividends are received, so none are taxed. Ignored: wash sales, the $3,000 ordinary-income offset, state tax.
- Two after-tax numbers: "still holding" (taxes paid along the way only) and "LIQUIDATED" (everything also sold on the last day, each lot taxed by its age; a net loss earns no refund). A buy-and-hold ETF pays nothing until liquidation.

### Metrics (the floor study's, plus the comparison ones)

Annual log growth, volatility, Sharpe over T-bills, max drawdown, worst calendar year (`floor_study.metrics`, `worst_year`); % of rolling 3-year windows that beat SPY (`floor_study.rolling_beat`); tracking error and information ratio vs SPY (daily, annualised); the longest stretch of underperformance vs SPY in months (the longest time the candidate's wealth relative to SPY stays below an earlier high, month-end data, an unfinished stretch counts); the t-statistic of the monthly excess return over SPY. Risk metrics and the t-statistic are on the pre-tax, after-cost path.

- Regime split: each calendar month is HIGH-VOL or LOW-VOL by SPY's 20-day realized volatility at the last close of the PREVIOUS month (known in advance), against the median of that reading over the sample being evaluated.
- Sub-periods: the sample's sessions cut into three equal consecutive thirds.
- Multiple comparisons: two-sided normal p-values; Bonferroni at family alpha 0.05 over the 16 series (`signal_attribution.bonferroni_z`: |t| must exceed about 2.96) is the test the decision uses; Benjamini-Hochberg q-values (`source_validation.benjamini_hochberg`) are reported beside it.

### Priors (price-only, each vs SPY on its own longest sample; written before any data)

| candidate | (a) annual log growth vs SPY | (b) Sharpe vs SPY | (c) max drawdown vs SPY |
|---|---|---|---|
| OEF (S&P 100) | -0.5 to +0.5 pp | about equal (+-0.03) | about equal (+-3 pts) |
| XLG (top 50) | +0.5 to +1.5 pp | 0 to +0.05 | about equal (+-3 pts) |
| MGK (mega-cap growth) | +2 to +4 pp (about 1 pp of it is the missing dividend) | +0.05 to +0.15 | equal to 5 pts worse |
| MTUM | 0 to +1 pp | about equal | about equal |
| SPMO | +1 to +3 pp | +0.05 to +0.15 | about equal |
| QUAL | -0.5 to +0.5 pp | about equal | about equal |
| SPHQ | -1 to +0.5 pp | about equal | about equal |
| B5 MTUM+QUAL | 0 to +0.7 pp before tax, about 0.1 pp less after | about equal | about equal |
| RSP (equal weight) | -0.5 to +0.5 pp on its 2003+ sample; -2 to -3 pp on the common sample | lower by 0.03 to 0.10 | 3 to 8 pts worse |
| USMV | -1.5 to -3 pp | -0.05 to +0.05 | 5 to 10 pts better |
| SPLV | -3 to -4.5 pp | lower by 0.05 to 0.15 | 3 to 8 pts better |
| B1 the live rule (2006+) | -1.5 to -0.3 pp | +0.03 to +0.08 | 15 to 25 pts better |

- Cash, vol rule off: (i) 15% at 0% costs 1.3 to 1.8 pp a year of log growth against (ii) fully invested, and shrinks the drawdown by about 15% of itself (-55% becomes about -48%). (iii) the T-bill fund gives back 0.15 to 0.30 pp a year of that at no extra drawdown. (iii) beats (i) in every sub-period with rates above the 0.19% fee.
- Vol rule ON vs OFF, same cash column: modern sample (2006+) log growth -0.3 to +0.8 pp, Sharpe +0.05 to +0.10, max drawdown at least 15 pts better. Long sample (1993+, price-only): log growth -0.3 to +0.3 pp, Sharpe +0.03 to +0.07, max drawdown 5 to 8 pts better. Probability the rule passes its own test below on the long sample: about 45% (price-only removes more from a 1.13x position than from a 1.0x one, and the floor study's edge was only +0.15 pp).
- Tax: B1's rebalancing costs 0.3 to 0.8 pp a year "still holding", 0.1 to 0.4 pp "liquidated". B5: under 0.1 pp.
- Statistics: tracking error of these funds vs SPY is 2 to 6% a year. With 11 to 20 years of months, an edge of 1 pp a year gives |t| of roughly 0.6 to 1.2. Clearing the Bonferroni bar (about 2.96) needs an edge of roughly 3 to 4 pp a year held for the whole sample. **This study cannot prove a small edge; it can only fail to find one.** Probability that at least one candidate meets conditions 1-5 below: about 35% (MGK, SPMO or XLG). Probability that one also survives Bonferroni: about 3%. Expected decision: SPY stays.

### Decision rules (fixed now)

**Rule A — the base.** A candidate REPLACES SPY as the base only if ALL of these hold on PRICE-ONLY data:

1. After-tax LIQUIDATED log growth is higher than SPY's on the candidate's own longest sample AND on the all-common sample.
2. In at least 2 of the 3 equal sub-periods of its own longest sample, its after-tax path log growth (after costs and taxes paid along the way) is higher than SPY's.
3. In BOTH the high-vol and the low-vol months of its own longest sample, its after-tax path log growth is higher than SPY's.
4. Its max drawdown is not worse than SPY's by more than 5 points, on its own longest sample AND on the all-common sample.
5. Its Sharpe is not below SPY's on its own longest sample.
6. Its monthly excess return over SPY survives Bonferroni over the 16 series.

Outcomes: all six -> REPLACE. Conditions 1-5 but not 6 -> **SPY stays**; the candidate is recorded as NOT DISPROVED, and whether to run it as a separately pre-registered paper trial is the owner's call, not this study's. Anything else -> SPY stays. A candidate meeting 1-5 on price-only but failing condition 1 on total return is labelled PAPER ARTIFACT.

**Rule B — the volatility rule earns its keep** if, ON vs OFF in column (iii) (idle money in the T-bill fund, the live design for the floor's own idle part), it meets the floor study's own ship rule on price-only data, on BOTH the modern sample and the long sample: log growth higher; Sharpe not lower; max drawdown no more than 10 points worse; higher log growth in at least 2 of 3 sub-periods. Passing on one sample only = "earns it on risk (Sharpe and drawdown), not on growth" if Sharpe is higher and drawdown smaller on both; otherwise "does not earn it on this evidence". The study changes nothing live either way: the floor is an EXPLORATORY experiment with its own kill date (2027-02-05).

**Rule C — the idle 15%.** (iii) the T-bill fund is preferred to (i) cash at 0% if its log growth is higher and its max drawdown is not worse by more than 0.5 point, for vol rule ON and OFF, modern sample. (ii) fully invested is preferred to (iii) if its after-tax liquidated log growth is higher on the full modern sample and in at least 2 of 3 sub-periods AND its max drawdown is no more than 5 points worse than the benchmark's (SPY buy-and-hold), for the vol rule ON row (the live row). What this study CANNOT measure, and says so: whether an option trade can actually be funded by selling the base in time, and what the reserve would have earned in the option trades it is kept for (nothing so far: it has sat idle).

### Limits known in advance

- 2013-2026 is one bull market led by mega-cap growth. "Top names" and momentum funds are, by construction, long exactly what led. A win for them on this sample is the sample describing itself, not evidence the tilt persists; the 2000-2012 stretch (where OEF and XLG have data) is the only counter-regime available and momentum/quality ETFs have none.
- SPHQ changed its index in 2010 and 2016; its early history is not today's fund. Fund fees fell over time.
- Daily closes; no intraday execution, no bid/ask on the ETFs beyond the 2 bp.
- The tax numbers are an estimate under one fixed set of assumptions.
- The monthly t-statistic ignores autocorrelation and fat tails.
