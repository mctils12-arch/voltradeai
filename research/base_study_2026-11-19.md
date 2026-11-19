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

---

## Results (appended 2026-11-19 after the run; the pre-registration above is unchanged since commit 2745a0b2)

Run: `python3 scripts/base_study.py <data dir> research/base_study_2026-11-19.json`, then `--md` for every table below (no number in a table is typed by hand; `test_base_study.py` checks each table line against the JSON). Data: 12 ETFs and ^IRX from Yahoo, all fetched, none dropped, through 2026-11-18.

### The answer in plain words

**Decision under the rules fixed in advance: SPY stays as the base.** No candidate's edge over SPY is distinguishable from luck once 16 series are tested (the bar is |t| > 2.96; the best is 2.25).

1. **"Hold the top names" — the idea is not wrong, but it is not an edge.** The S&P 500 is weighted by size, so it already holds the top names in proportion to their size. Holding ONLY them is OEF (top 100) or XLG (top 50):
   - OEF, 26 years: 0.21 points a year BEHIND SPY. It lagged in its first two thirds (2000-2018) and led only in the last. Its wealth relative to SPY has never got back to where it started in October 2000 (311 months).
   - XLG, 21.5 years: 0.18 points a year ahead, t = 0.23 (nothing). Behind in 2005-2012 and 2012-2019, ahead by 1.67 points a year since 2019.
   - On the 2015+ common sample both are ahead by 1.0 to 1.4 points a year. That is the last decade's mega-cap run. Before it, the same funds lost to SPY.
   - MGK (mega-cap GROWTH) is the one that looks good: +2.70 points a year over 18.9 years, higher Sharpe (0.567 vs 0.456), ahead in all three thirds and in both volatility regimes, t = 2.25. It meets conditions 1-5 and fails the significance bar (p = 0.025 before correction; with 16 tests about 0.4 results this strong are expected by luck; Benjamini-Hochberg q = 0.22). About 0.8 of the 2.70 points is the dividend the paper account does not receive (total-return edge: +1.91). Its sample starts in December 2007, so it contains no 2000-2002 (the floor study measured QQQ at −83% there), and in its last third its drawdown was −36.4% against SPY's −25.4%.
2. **"Hold the best names by a rule" (momentum, quality) — no.**
   - Quality: QUAL +0.03 points a year (t = 0.06), SPHQ −0.46. Nothing.
   - Momentum: MTUM +0.58 (t = 0.39) with a LOWER Sharpe than SPY (0.602 vs 0.654); ahead in one third of three; in high-volatility months it lost 3.62 points a year to SPY.
   - SPMO +2.96 points a year, higher Sharpe, t = 1.19 over only 11.1 years; behind SPY in its middle third (2019-2023), and +6.48 a year since March 2023. It meets conditions 1-5 and is nowhere near significant.
   - The 50/50 momentum + quality blend: +0.51 before tax, +0.43 after, t = 0.46; on the common sample it is behind SPY.
   - The controls say the same thing from the other side: equal weight (RSP) is −2.39 points a year on the common sample after +3.95 in 2003-2011; minimum volatility lost 2.7 to 3.6 points a year. Whatever led in one decade lagged in another.
3. **The idle cash — yes, it costs, and the rule fixed in advance says put it to work.** In the live design (vol rule on, 2006-2026):
   - 15% in cash at 0% (live): 8.36% a year. The same 15% in the base: 9.59%. **The idle 15% costs 1.23 points a year before tax** (0.99 after the tax estimate).
   - Parking it in a T-bill fund instead gives back 0.20 of that (8.56%) with no extra drawdown (−33.7% vs −34.1%).
   - Fully invested has a deeper drawdown, −38.8% against −33.7%. SPY's own was −56.5%.
   - Rule C: the T-bill fund is preferred to cash at 0% (yes, with the vol rule on and off); fully invested is preferred to the T-bill fund (yes: higher after-tax growth on the full sample and in all three thirds, drawdown far inside the limit).
   - What the study cannot say: whether an option trade can be funded in time by selling the base, and what selling costs in tax each time (under FIFO most of the floor's lots are short-term). The reserve has not been used by an option trade so far, so its option-side value is unmeasured, not zero.
4. **The volatility rule — by Rule B: "earns it on risk (Sharpe and drawdown), not on growth".**
   - ON vs OFF, idle money in the T-bill fund, before tax: 2006-2026 growth +0.41 points a year, Sharpe 0.572 vs 0.470, max drawdown −33.7% vs −49.4%. 1993-2026: growth +0.12, Sharpe 0.456 vs 0.397, drawdown −43.6% vs −51.9%, but ahead in only one of three thirds (−0.85, +1.27, −0.08), so the growth test fails on the long sample.
   - On total return the growth edge is gone: 0.00 (2006+) and −0.64 (1993+). Part of the price-only edge is a paper artifact: SSO keeps the index's dividends inside its price, SPY pays them out and paper does not credit them.
   - **After the tax estimate the rule LOSES growth**: ON minus OFF, liquidated, −0.71 points a year (2006+) and −1.15 (1993+). See the next point.
5. **The surprise, and it matters for an after-tax goal: the floor's rebalancing is taxed heavily in this estimate.** The live rule rebalances about 25 times a year and trades 6.6 times its equity a year (moving between SPY and SSO doubles the notional). The lots it sells are 103 days old on average, so under FIFO 87% of the gains it realizes are short-term (328% of starting equity short-term, 51% long-term, 2006-2026). Estimated cost: 1.79 points a year. After tax, liquidated: the live rule 6.60% a year against SPY buy-and-hold 8.01% (−1.41); with the reserve invested 7.59% (−0.42). My prior was 0.3 to 0.8 points; it was wrong by a factor of two to five. Paper pays no tax, so this is not in the account's equity curve; it is in its stated goal.

What stays true for the live floor: before tax it gave SPY-like growth (−0.30 points a year with 15% idle, +0.93 with the reserve invested) at a 0.10 higher Sharpe and a max drawdown of −34% (−39% with the reserve invested) against SPY's −56%.

### Tables (generated by `scripts/base_study.py --md`)

#### Data (every ticker's first date; Yahoo daily, through 2026-11-18)

| ticker | first date | sessions | largest 1-day move | price-only growth/yr | total-return growth/yr | dividend part |
|---|---|---|---|---|---|---|
| SPY | 1993-01-29 | 8510 | 14.52% | 8.32% | 10.11% | 1.78 pp |
| SSO | 2006-06-21 | 5136 | 23.36% | 12.97% | 13.98% | 1.01 pp |
| OEF | 2000-10-27 | 6553 | 11.62% | 6.16% | 7.91% | 1.75 pp |
| XLG | 2005-05-10 | 5417 | 11.05% | 8.72% | 10.43% | 1.71 pp |
| MGK | 2007-12-27 | 4754 | 12.55% | 11.18% | 12.20% | 1.02 pp |
| MTUM | 2013-04-18 | 3419 | 12.37% | 12.04% | 13.20% | 1.16 pp |
| SPMO | 2015-10-12 | 2793 | 15.35% | 14.58% | 15.72% | 1.14 pp |
| QUAL | 2013-07-18 | 3356 | 10.36% | 11.03% | 12.45% | 1.42 pp |
| SPHQ | 2005-12-06 | 5271 | 12.89% | 7.91% | 9.19% | 1.28 pp |
| RSP | 2003-05-01 | 5927 | 12.04% | 9.08% | 10.61% | 1.53 pp |
| USMV | 2011-10-20 | 3792 | 10.08% | 9.20% | 11.06% | 1.86 pp |
| SPLV | 2011-05-05 | 3909 | 12.39% | 7.33% | 9.63% | 2.30 pp |

#### Each base against SPY on its OWN LONGEST sample (price-only; the decision basis)

| base | from | yrs | growth/yr (SPY) | after-tax liquidated (SPY) | Sharpe (SPY) | max DD (SPY) | worst year | track. err | info ratio | 3y windows beating SPY | longest lag (months) | t | p | BH q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OEF | 2000-10-30 | 26.0 | 6.16% (6.37%) | 5.67% (5.87%) | 0.323 (0.334) | −57.3% (−56.5%) | 2008 −37.2% | 4.7% | −0.04 | 39.3% | 311 | −0.37 | 0.710 | 0.874 |
| XLG | 2005-05-11 | 21.5 | 8.72% (8.54%) | 8.09% (7.91%) | 0.465 (0.451) | −53.8% (−56.5%) | 2008 −35.4% | 4.4% | +0.03 | 58.8% | 178 | +0.23 | 0.819 | 0.874 |
| MGK | 2007-12-28 | 18.9 | 11.18% (8.47%) | 10.43% (7.80%) | 0.567 (0.456) | −48.5% (−53.9%) | 2008 −36.9% | 6.6% | +0.45 | 95.6% | 30 | +2.25 | 0.025 | 0.225 |
| MTUM | 2013-04-19 | 13.6 | 12.04% (11.47%) | 11.10% (10.54%) | 0.602 (0.654) | −34.1% (−34.1%) | 2022 −19.7% | 10.0% | +0.12 | 63.9% | 64 | +0.39 | 0.695 | 0.874 |
| SPMO | 2015-10-13 | 11.1 | 14.58% (11.62%) | 13.42% (10.58%) | 0.702 (0.618) | −31.3% (−34.1%) | 2022 −12.0% | 11.2% | +0.31 | 79.9% | 40 | +1.19 | 0.236 | 0.629 |
| QUAL | 2013-07-19 | 13.3 | 11.03% (11.00%) | 10.11% (10.08%) | 0.623 (0.623) | −34.1% (−34.1%) | 2022 −21.7% | 3.6% | +0.01 | 48.3% | 129 | +0.06 | 0.950 | 0.950 |
| SPHQ | 2005-12-07 | 20.9 | 7.91% (8.37%) | 7.29% (7.74%) | 0.419 (0.443) | −57.8% (−56.5%) | 2008 −46.4% | 8.2% | −0.06 | 53.4% | 221 | −0.27 | 0.785 | 0.874 |
| MTUM+QUAL | 2013-07-19 | 13.3 | 11.51% (11.00%) | 10.51% (10.08%) | 0.623 (0.623) | −34.1% (−34.1%) | 2022 −20.7% | 5.3% | +0.14 | 64.9% | 68 | +0.46 | 0.649 | 0.874 |
| RSP | 2003-05-02 | 23.5 | 9.08% (8.81%) | 8.47% (8.22%) | 0.471 (0.475) | −60.9% (−56.5%) | 2008 −41.1% | 5.8% | +0.08 | 50.8% | 140 | +0.53 | 0.597 | 0.874 |
| USMV | 2011-10-21 | 15.0 | 9.20% (11.91%) | 8.41% (11.02%) | 0.629 (0.696) | −33.1% (−34.1%) | 2022 −10.9% | 8.0% | −0.40 | 25.2% | 125 | −1.61 | 0.107 | 0.396 |
| SPLV | 2011-05-06 | 15.5 | 7.33% (10.95%) | 6.63% (10.11%) | 0.468 (0.630) | −36.6% (−34.1%) | 2022 −6.9% | 10.5% | −0.39 | 15.4% | 182 | −1.54 | 0.124 | 0.396 |
| B1-i | 2006-06-22 | 20.4 | 8.36% (8.66%) | 6.60% (8.01%) | 0.558 (0.459) | −34.1% (−56.5%) | 2008 −21.0% | 10.3% | −0.12 | 33.7% | 213 | −0.50 | 0.620 | 0.874 |

#### Every base on the ALL-COMMON sample (from 2015-10-12; price-only)

| base | from | yrs | growth/yr (SPY) | after-tax liquidated (SPY) | Sharpe (SPY) | max DD (SPY) | worst year | track. err | info ratio | 3y windows beating SPY | longest lag (months) | t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OEF | 2015-10-13 | 11.1 | 12.57% (11.62%) | 11.49% (10.58%) | 0.658 (0.618) | −31.4% (−34.1%) | 2022 −22.2% | 2.9% | +0.36 | 84.8% | 47 | +1.32 |
| XLG | 2015-10-13 | 11.1 | 13.03% (11.62%) | 11.93% (10.58%) | 0.673 (0.618) | −30.8% (−34.1%) | 2022 −25.2% | 4.4% | +0.36 | 95.1% | 35 | +1.25 |
| MGK | 2015-10-13 | 11.1 | 14.68% (11.62%) | 13.52% (10.58%) | 0.685 (0.618) | −36.4% (−34.1%) | 2022 −34.0% | 7.4% | +0.52 | 92.2% | 30 | +1.70 |
| MTUM | 2015-10-13 | 11.1 | 11.75% (11.62%) | 10.71% (10.58%) | 0.548 (0.618) | −34.1% (−34.1%) | 2022 −19.7% | 10.7% | +0.08 | 52.8% | 64 | +0.21 |
| SPMO | 2015-10-13 | 11.1 | 14.58% (11.62%) | 13.42% (10.58%) | 0.702 (0.618) | −31.3% (−34.1%) | 2022 −12.0% | 11.2% | +0.31 | 79.9% | 40 | +1.19 |
| QUAL | 2015-10-13 | 11.1 | 11.18% (11.62%) | 10.16% (10.58%) | 0.592 (0.618) | −34.1% (−34.1%) | 2022 −21.7% | 3.6% | −0.12 | 37.2% | 129 | −0.46 |
| SPHQ | 2015-10-13 | 11.1 | 11.40% (11.62%) | 10.37% (10.58%) | 0.608 (0.618) | −32.0% (−34.1%) | 2022 −17.3% | 5.0% | −0.05 | 47.3% | 121 | −0.20 |
| MTUM+QUAL | 2015-10-13 | 11.1 | 11.57% (11.62%) | 10.48% (10.58%) | 0.587 (0.618) | −34.1% (−34.1%) | 2022 −20.7% | 5.6% | +0.03 | 55.2% | 68 | +0.05 |
| RSP | 2015-10-13 | 11.1 | 9.23% (11.62%) | 8.32% (10.58%) | 0.478 (0.618) | −39.6% (−34.1%) | 2022 −13.2% | 6.5% | −0.36 | 11.7% | 120 | −1.17 |
| USMV | 2015-10-13 | 11.1 | 8.11% (11.62%) | 7.27% (10.58%) | 0.483 (0.618) | −33.1% (−34.1%) | 2022 −10.9% | 8.5% | −0.48 | 12.3% | 125 | −1.65 |
| SPLV | 2015-10-13 | 11.1 | 6.51% (11.62%) | 5.79% (10.58%) | 0.358 (0.618) | −36.6% (−34.1%) | 2022 −6.9% | 11.6% | −0.48 | 3.7% | 125 | −1.69 |
| B1-i | 2015-10-13 | 11.1 | 10.49% (11.62%) | 8.37% (10.58%) | 0.674 (0.618) | −17.0% (−34.1%) | 2022 −13.6% | 8.4% | −0.21 | 41.5% | 80 | −0.81 |

#### Sub-periods and volatility regimes (own longest sample, price-only; growth/yr minus SPY's, in points; after tax paid along the way)

| base | first third | middle third | last third | thirds won | high-vol months | low-vol months |
|---|---|---|---|---|---|---|
| OEF | −1.16 (2000-10→2009-07) | −0.70 (2009-07→2018-03) | +1.23 (2018-03→2026-11) | 1/3 | +0.03 | −0.45 |
| XLG | −0.65 (2005-05→2012-07) | −0.48 (2012-07→2019-09) | +1.67 (2019-09→2026-11) | 1/3 | +0.70 | −0.33 |
| MGK | +1.90 (2007-12→2014-04) | +5.10 (2014-04→2020-07) | +1.11 (2020-07→2026-11) | 3/3 | +3.28 | +2.13 |
| MTUM | +3.60 (2013-04→2017-10) | −1.62 (2017-10→2022-05) | −0.24 (2022-05→2026-11) | 1/3 | −3.62 | +4.70 |
| SPMO | +2.97 (2015-10→2019-06) | −0.57 (2019-06→2023-03) | +6.48 (2023-03→2026-11) | 2/3 | +1.44 | +4.47 |
| QUAL | +1.02 (2013-07→2017-12) | −0.86 (2017-12→2022-06) | −0.07 (2022-06→2026-11) | 1/3 | −0.76 | +0.79 |
| SPHQ | −1.49 (2005-12→2012-11) | +0.22 (2012-11→2019-11) | −0.12 (2019-11→2026-11) | 1/3 | −2.42 | +1.51 |
| MTUM+QUAL | +2.66 (2013-07→2017-12) | −1.54 (2017-12→2022-06) | −0.07 (2022-06→2026-11) | 1/3 | −2.02 | +2.65 |
| RSP | +3.95 (2003-05→2011-03) | −0.30 (2011-03→2019-01) | −2.87 (2019-01→2026-11) | 1/3 | +0.72 | −0.20 |
| USMV | −0.28 (2011-10→2016-10) | −4.09 (2016-10→2021-11) | −3.76 (2021-11→2026-11) | 0/3 | −4.52 | −0.92 |
| SPLV | +1.75 (2011-05→2016-07) | −6.88 (2016-07→2021-09) | −5.74 (2021-09→2026-11) | 1/3 | −6.37 | −0.92 |
| B1-i | +0.96 (2006-06→2013-04) | −2.73 (2013-04→2020-01) | −4.30 (2020-01→2026-11) | 1/3 | −3.15 | −0.90 |

#### Total-return check (dividends included on both sides; not the decision basis)

| base | own sample: growth/yr minus SPY (price-only) | same, total return | common sample (price-only) | same, total return |
|---|---|---|---|---|
| OEF | −0.21 | −0.24 | +0.95 | +0.87 |
| XLG | +0.18 | +0.07 | +1.41 | +1.10 |
| MGK | +2.70 | +1.91 | +3.06 | +2.24 |
| MTUM | +0.58 | +0.05 | +0.13 | −0.34 |
| SPMO | +2.96 | +2.48 | +2.96 | +2.48 |
| QUAL | +0.03 | −0.23 | −0.44 | −0.62 |
| SPHQ | −0.46 | −1.01 | −0.22 | −0.33 |
| MTUM+QUAL | +0.51 | +0.12 | −0.05 | −0.38 |
| RSP | +0.26 | −0.02 | −2.39 | −2.33 |
| USMV | −2.71 | −2.59 | −3.51 | −3.32 |
| SPLV | −3.62 | −3.07 | −5.11 | −4.61 |
| B1-i | −0.30 | −0.91 | −1.13 | −1.72 |

#### Rule A, check by check (Bonferroni bar: |t| > 2.96 over 16 series)

| base | 1 after-tax growth, own + common | 2 two of three thirds | 3 both vol regimes | 4 drawdown within 5 pts | 5 Sharpe | 6 Bonferroni | condition 1 on total return | outcome |
|---|---|---|---|---|---|---|---|---|
| OEF | NO | NO | NO | yes | NO | NO | NO | SPY STAYS |
| XLG | yes | NO | NO | yes | yes | NO | yes | SPY STAYS |
| MGK | yes | yes | yes | yes | yes | NO | yes | SPY STAYS (not disproved) |
| MTUM | yes | NO | NO | yes | NO | NO | NO | SPY STAYS |
| SPMO | yes | yes | yes | yes | yes | NO | yes | SPY STAYS (not disproved) |
| QUAL | NO | NO | NO | yes | yes | NO | NO | SPY STAYS |
| SPHQ | NO | NO | NO | yes | NO | NO | NO | SPY STAYS |
| MTUM+QUAL | NO | NO | NO | yes | yes | NO | NO | SPY STAYS |
| RSP | NO | NO | NO | NO | NO | NO | NO | SPY STAYS |
| USMV | NO | NO | NO | yes | NO | NO | NO | SPY STAYS |
| SPLV | NO | NO | NO | yes | NO | NO | NO | SPY STAYS |
| B1-i | NO | NO | NO | yes | yes | NO | NO | SPY STAYS |

**Rule A decision: SPY stays.**

#### Differences from SPY, own longest sample (price-only; what the priors were about)

| base | growth/yr, points | after-tax liquidated growth/yr, points | Sharpe | max drawdown, points (+ = smaller) |
|---|---|---|---|---|
| OEF | −0.21 | −0.20 | −0.011 | −0.8 |
| XLG | +0.18 | +0.18 | +0.014 | +2.7 |
| MGK | +2.70 | +2.63 | +0.111 | +5.3 |
| MTUM | +0.58 | +0.56 | −0.052 | +0.0 |
| SPMO | +2.96 | +2.84 | +0.084 | +2.8 |
| QUAL | +0.03 | +0.03 | +0.000 | +0.0 |
| SPHQ | −0.46 | −0.45 | −0.024 | −1.4 |
| MTUM+QUAL | +0.51 | +0.43 | +0.000 | +0.0 |
| RSP | +0.26 | +0.26 | −0.004 | −4.4 |
| USMV | −2.71 | −2.61 | −0.067 | +1.0 |
| SPLV | −3.62 | −3.47 | −0.162 | −2.5 |
| B1-i | −0.30 | −1.41 | +0.099 | +22.4 |

#### Tax ESTIMATE for the bases that trade (price-only; FIFO lots, 24% short-term, 15% long-term)

| base | sample | growth/yr before tax | after tax, still holding | after tax, liquidated | SPY liquidated | realized short-term gains | realized long-term gains | tax paid | mean age of lots sold | rebalances |
|---|---|---|---|---|---|---|---|---|---|---|
| MTUM+QUAL | own, 2013+ | 11.51% | 11.34% | 10.51% | 10.08% | 0% | 42% | 6% | 3244 days | 13 |
| MTUM+QUAL | common, 2015+ | 11.57% | 11.41% | 10.48% | 10.58% | 0% | 27% | 4% | 2529 days | 11 |
| B1-i | modern, 2006+ | 8.36% | 6.57% | 6.60% | 8.01% | 328% | 51% | 82% | 103 days | 505 |
| B1-ii | modern, 2006+ | 9.59% | 7.56% | 7.59% | 8.01% | 432% | 63% | 107% | 103 days | 499 |
| B1-iii | modern, 2006+ | 8.56% | 6.73% | 6.75% | 8.01% | 333% | 51% | 86% | 102 days | 505 |
| B1-i | common, 2015+ | 10.49% | 8.33% | 8.37% | 10.58% | 170% | 34% | 43% | 104 days | 265 |
| B1-i | long, 1993+ | 7.49% | 5.92% | 5.94% | 7.87% | 731% | 114% | 186% | 104 days | 833 |

(Gains and tax are in % of the starting equity, summed over the sample.)

#### The cash and volatility-rule grid, modern sample (real SSO; 2006-06-22 → 2026-11-18, 20.4 years; price-only)

| cell | growth/yr | after tax, still holding | after tax, liquidated | Sharpe | max DD | worst year | mean exposure | mean cash | mean T-bill fund | rebalances/yr | turnover/yr | t vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-i | 7.98% | 7.94% | 7.34% | 0.462 | −49.7% | 2008 −33.3% | 0.88x | 11.7% | 0.1% | 0.1 | 1% | −2.20 |
| B0-ii | 8.66% | 8.66% | 8.01% | 0.459 | −56.5% | 2008 −38.3% | 1.00x | 0.0% | 0.0% | 0.0 | 0% | +0.00 |
| B0-iii | 8.15% | 8.04% | 7.46% | 0.470 | −49.4% | 2008 −31.8% | 0.89x | 0.0% | 11.2% | 0.1 | 2% | −1.85 |
| B1-i | 8.36% | 6.57% | 6.60% | 0.558 | −34.1% | 2008 −21.0% | 0.97x | 14.8% | 8.5% | 24.7 | 662% | −0.50 |
| B1-ii | 9.59% | 7.56% | 7.59% | 0.574 | −38.8% | 2008 −24.3% | 1.14x | 0.3% | 9.8% | 24.5 | 783% | +0.56 |
| B1-iii | 8.56% | 6.73% | 6.75% | 0.572 | −33.7% | 2008 −20.8% | 0.97x | 0.7% | 22.6% | 24.7 | 668% | −0.36 |

#### The same grid on total return (check)

| cell | growth/yr | after tax, still holding | after tax, liquidated | Sharpe | max DD | worst year | mean exposure | mean cash | mean T-bill fund | rebalances/yr | turnover/yr | t vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-i | 9.63% | 9.55% | 8.95% | 0.555 | −48.5% | 2008 −32.1% | 0.89x | 10.6% | 0.1% | 0.1 | 1% | −2.78 |
| B0-ii | 10.48% | 10.48% | 9.78% | 0.554 | −55.2% | 2008 −36.8% | 1.00x | 0.0% | 0.0% | 0.0 | 0% | +0.00 |
| B0-iii | 9.78% | 9.69% | 9.05% | 0.567 | −48.1% | 2008 −31.6% | 0.89x | 0.0% | 11.2% | 0.1 | 2% | −2.34 |
| B1-i | 9.57% | 7.54% | 7.57% | 0.647 | −32.9% | 2008 −20.2% | 0.97x | 14.8% | 8.5% | 24.5 | 663% | −0.92 |
| B1-ii | 11.01% | 8.70% | 8.73% | 0.663 | −37.6% | 2008 −23.4% | 1.14x | 0.3% | 9.8% | 24.4 | 785% | +0.28 |
| B1-iii | 9.77% | 7.70% | 7.72% | 0.662 | −32.6% | 2008 −20.1% | 0.97x | 0.7% | 22.6% | 24.5 | 669% | −0.78 |

#### The grid on the long sample (synthetic 2x fund; 1993-03-02 → 2026-11-18, 33.7 years; price-only)

| cell | growth/yr | after tax, still holding | after tax, liquidated | Sharpe | max DD | worst year | mean exposure | mean cash | mean T-bill fund | rebalances/yr | turnover/yr | t vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-i | 7.50% | 7.40% | 7.01% | 0.394 | −49.9% | 2008 −33.5% | 0.88x | 10.4% | 1.8% | 0.1 | 1% | −3.15 |
| B0-ii | 8.32% | 8.32% | 7.87% | 0.407 | −56.5% | 2008 −38.3% | 1.00x | 0.0% | 0.0% | 0.0 | 0% | +0.00 |
| B0-iii | 7.68% | 7.71% | 7.33% | 0.397 | −51.9% | 2008 −35.0% | 0.89x | 0.0% | 10.5% | 0.1 | 2% | −3.16 |
| B1-i | 7.49% | 5.92% | 5.94% | 0.433 | −44.3% | 2008 −20.9% | 0.97x | 14.9% | 8.5% | 24.6 | 657% | −1.16 |
| B1-ii | 8.54% | 6.79% | 6.80% | 0.456 | −49.9% | 2008 −24.3% | 1.13x | 0.3% | 9.7% | 24.4 | 778% | +0.14 |
| B1-iii | 7.80% | 6.16% | 6.18% | 0.456 | −43.6% | 2008 −20.8% | 0.97x | 0.9% | 22.4% | 24.6 | 662% | −0.85 |

#### The long-sample grid on total return (check; the floor study's basis)

| cell | growth/yr | after tax, still holding | after tax, liquidated | Sharpe | max DD | worst year | mean exposure | mean cash | mean T-bill fund | rebalances/yr | turnover/yr | t vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-i | 9.05% | 9.00% | 8.57% | 0.485 | −50.2% | 2008 −33.3% | 0.88x | 11.6% | 0.0% | 0.1 | 1% | −4.17 |
| B0-ii | 10.11% | 10.11% | 9.64% | 0.504 | −55.2% | 2008 −36.8% | 1.00x | 0.0% | 0.0% | 0.0 | 0% | +0.00 |
| B0-iii | 9.43% | 9.32% | 8.90% | 0.510 | −50.2% | 2008 −33.2% | 0.88x | 0.0% | 11.9% | 0.1 | 2% | −2.86 |
| B1-i | 8.48% | 6.71% | 6.73% | 0.506 | −43.1% | 2008 −20.2% | 0.97x | 14.9% | 8.5% | 24.5 | 657% | −1.93 |
| B1-ii | 9.69% | 7.70% | 7.72% | 0.528 | −48.7% | 2008 −23.4% | 1.13x | 0.4% | 9.7% | 24.3 | 782% | −0.50 |
| B1-iii | 8.79% | 6.95% | 6.97% | 0.528 | −42.6% | 2008 −20.1% | 0.97x | 0.9% | 22.4% | 24.5 | 663% | −1.64 |

Volatility rule ON minus OFF by third, modern sample, column (iii), growth/yr in points: +1.50 (2006–2013), +0.28 (2013–2020), −0.54 (2020–2026).

Volatility rule ON minus OFF by third, long sample, column (iii), growth/yr in points: −0.85 (1993–2004), +1.27 (2004–2015), −0.08 (2015–2026).

**Rule B (the volatility rule, ON vs OFF, column iii):** modern sample log_growth_higher yes, sharpe_not_lower yes, max_dd_within_10_points yes, higher_growth_in_2_of_3_subperiods yes; long sample log_growth_higher yes, sharpe_not_lower yes, max_dd_within_10_points yes, higher_growth_in_2_of_3_subperiods NO. Verdict: **earns it on risk (Sharpe and drawdown), not on growth**.

**Rule C (the idle 15%):** T-bill fund preferred to cash at 0%: **yes** (vol rule off yes, on yes). Fully invested preferred to the T-bill fund: **yes** (after_tax_liquidated_growth_higher yes, higher_in_2_of_3_subperiods yes, max_dd_within_5_points_of_spy yes).

#### Data checks

- The floor study's own V1 code on today's download, through 2026-11-04: log growth 10.41%, Sharpe 0.572, max DD -48.41% (published: 10.41%, 0.573, −48.41%). SPY: 10.26%, 0.513, -55.19% (published 10.26%, 0.513, −55.2%).
- The synthetic 2x fund against real SSO from 2006-06-21: synthetic 13.64%/yr, SSO price-only 12.97%/yr, SSO total return 13.98%/yr; daily tracking error 3.91%/yr.


### How the priors held up

- **Base candidates, growth vs SPY** (prior → result, points a year): OEF −0.5..+0.5 → −0.21. XLG +0.5..+1.5 → +0.18 (below the range). MGK +2..+4 → +2.70. MTUM 0..+1 → +0.58. SPMO +1..+3 → +2.96. QUAL −0.5..+0.5 → +0.03. SPHQ −1..+0.5 → −0.46. Blend 0..+0.7 → +0.51. RSP −0.5..+0.5 → +0.26 (common sample −2..−3 → −2.39). USMV −1.5..−3 → −2.71. SPLV −3..−4.5 → −3.62. Ten of eleven inside the range.
- **Sharpe:** MTUM was expected about equal and came in 0.05 lower. RSP was expected lower and was equal. SPLV was 0.16 lower (prior 0.05 to 0.15).
- **Drawdown, where I was wrong:** minimum volatility did NOT protect on this sample. USMV was 1.0 point better than SPY (prior 5 to 10); SPLV was 2.5 points WORSE (prior 3 to 8 better). MGK was 5.3 points better on its own sample (prior: equal to 5 worse), 2.3 worse on the common sample.
- **The live rule (B1):** growth −0.30 (prior −1.5..−0.3, at the edge), Sharpe +0.099 (prior +0.03..+0.08), drawdown 22.4 points better (prior 15 to 25).
- **Cash:** the T-bill fund gives back 0.17 to 0.20 points (prior 0.15 to 0.30). Cash at 0% costs 0.68 (vol rule off) and 1.23 (on) points a year against fully invested (prior 1.3 to 1.8 for the rule-off row: too high, because with the rule off the simulator almost never rebalances and the 15% drifts down to about 12% of the account, and because price-only SPY grew 8.7% a year, not 11%).
- **Vol rule ON vs OFF:** 2006+ growth +0.41 (prior −0.3..+0.8), Sharpe +0.10 (prior +0.05..+0.10), drawdown 15.6 points better (prior at least 15). 1993+ growth +0.12 (prior −0.3..+0.3), Sharpe +0.06 (prior +0.03..+0.07), drawdown 8.3 points better (prior 5 to 8). I gave it a 45% chance of passing its own rule on the long sample; it failed on the sub-period test.
- **Tax: the big miss.** B1 "still holding" 1.79 points a year (prior 0.3 to 0.8); liquidated 1.76 (prior 0.1 to 0.4). I had not traced that the SPY-to-SSO swap turns the whole book over several times a year, so FIFO lots are almost never a year old. The blend: 0.08 liquidated (prior under 0.1), 0.17 still holding.
- **Statistics:** prior 35% that something meets conditions 1-5 → two did (MGK, SPMO, two of the three I named). Prior 3% that one survives Bonferroni → none. Expected decision "SPY stays" → SPY stays.

### How many comparisons

16 series were tested against SPY, one significance test each, one family (Bonferroni bar 2.96; nothing passes; nothing passes Benjamini-Hochberg at 10% either). The tables show 33 series-by-sample comparisons on price-only (11 own + 12 common + 5 modern grid + 5 long grid) and the same 33 on total return as a check. No variant was added, removed or re-parameterised after the first result.

### What differs from the pre-registration

- The five grid cells' significance tests use the modern sample (2006+), the sample on which the grid is defined. The pre-registration said "own longest sample"; for the two rule-off cells that could be read as 1993+. Their t-statistics there are shown in the long-sample table (−3.15, −3.16). No decision uses them.
- Two reporting tables were added after the first run: "Differences from SPY" and "Tax ESTIMATE" (realized short- and long-term gains, the age of the lots sold). They add columns, not variants.
- BOXX itself was not fetched; the T-bill fund is modelled (as pre-registered).

### Limits — reasons to distrust this

1. **One regime.** The momentum, quality and MGK samples are 11 to 19 years of a bull market led by mega-cap growth. MGK and SPMO are long exactly what led. Their result is the sample describing itself. OEF and XLG, which have data before the run, lost to SPY before it.
2. **Price-only is a paper-account basis.** It removes 1.8 points a year from SPY and 1.0 to 1.1 from MGK, SPMO and SSO. It flatters low-yield funds by 0.5 to 0.8 points a year and flatters the vol rule (SSO) against plain SPY by 0.4 to 0.8. None of that exists in a real account.
3. **The tax numbers are an estimate under FIFO.** Choosing lots (highest cost first) would realize far less. Wash sales are ignored (the floor re-buys SPY within 30 days all the time; counting them would defer losses and raise the tax). State tax and the $3,000 offset are ignored. The repo's own tax ledger walks average cost, not FIFO. Paper pays no tax.
4. **Execution is simplified.** Decisions at the close on that close's data, filled at the close, 2 bp. The live floor trades intraday on the previous sessions' closes. The flow-model multiplier is not simulated. The reserve is assumed never used. The T-bill fund is the 13-week rate minus 0.19%, with no floor in zero-rate years.
5. **The 2x fund before 2006 is synthetic.** Against real SSO since 2006 it grows 0.67 points a year faster than SSO's price and 0.34 slower than SSO's total return, with 3.9% a year of daily tracking error. The long-sample grid inherits that.
6. **Rule-off rows are a loose control.** With the target pinned at 1.0 the 0.10 band is crossed about once a decade, so "15% cash" drifts to 10 to 12% and the cash drag in those rows is understated. The taxed and untaxed runs can rebalance on different days (long sample, B0-iii: 4 rebalances taxed, 2 untaxed), which is why that cell's after-tax growth is 0.03 above its pre-tax growth. The rule-ON rows also hold more market on average (0.97x of equity against 0.89x in column iii, 1.14x against 1.00x in column ii), so part of ON's growth is more exposure, not better timing.
7. **The statistics are coarse.** Monthly excess returns, normal p-values, no correction for autocorrelation or fat tails. Daily tracking error includes closing-print noise, so information ratios are understated for thinly traded funds.
8. **Fund history is not today's fund.** SPHQ changed its index in 2010 and 2016. Fees fell over time.
9. **One data source.** Yahoo only; no Alpaca keys in this session, so the price-only closes were not cross-checked against the feed the bot trades on. The one external check available passed: the floor study's own V1 code on this download reproduces its published numbers (10.41%, Sharpe 0.572 vs 0.573, −48.41%).
10. **ETFs are not "the best names".** These are rule-based index funds. The study says nothing about a stock-picking base, and it deliberately does not test one, because a list of today's best names tested on yesterday's prices is survivorship bias.

### What this hands to the owner (nothing here changes the live account)

- Rule A: SPY stays. MGK and SPMO are "not disproved"; running either as a separately pre-registered paper trial is the owner's call. The honest expectation is that their edge is the last decade, not a property.
- Rule C: the reserve at 0% is the most expensive choice measured. The T-bill fund costs nothing in drawdown. Investing it costs 5 points of drawdown and adds about 1 point a year. Either is a risk-limit / reserve-design decision (RULE REVIEW), not made here.
- The tax estimate on the floor's rebalancing is new and large. Before any decision rests on it, it needs its own measurement PR: the live floor's realized gains since 2026-11-06 against this simulator, and a lot-selection comparison (FIFO vs highest-cost-first).
