# Edge pre-registration — Phase 6 part B (2026-10-27)

Provenance: human-directed Phase 6 research ("review published literature on equity and options edges that fit our data; pick the 3 most promising; pre-register each"). Docs only: nothing in this file has been run against returns, no trading code changed. Every statistic, threshold, horizon and kill rule below is fixed BEFORE any outcome is seen (REASONING STANDARD #10). A test that is later run with any parameter changed from this file is a NEW variant and must say so and raise its variant count.

Context that shaped the choice (all in this repo):
- The account trails SPY by ~13 pts since 2026-05-06; the options sleeve carries the loss (−$9,177 realized, audit 2026-10-27 §7); the composite score is anti-predictive (open_questions.md, 2026-10-26 CRITICAL). New signals are therefore tested STANDALONE, never blended into deep_score.
- Data we actually have: daily equity bars (Alpaca IEX / Yahoo; active symbols only — survivor-biased), ~81 days of once-daily Alpaca INDICATIVE option chains (iv, delta, bid/ask, no OI, no volume, SPY/QQQ/IWM pinned only since 2026-10-21, day labels shifted one session per audit §2.4 until fixed), SEC Form 3/4/5 bulk data sets (2006 →, archive keeps 8 quarters by default), EDGAR 10-K/8-K text (Lazy Prices pipeline), Cboe VIX family history, the alt-data archives. No historical option quotes, no OI history; Polygon options entitlement unknown until the `options_entitlements` probe (Phase 6 part A) reads live.
- Capacity: a ~$100k paper account. Positions of $2-5k are under 1% of ADV for any name with ADV ≥ $0.5M, so small/illiquid names are open to us (EDGE DOCTRINE #2) — but measured costs are unreliable (synthetic fills, audit §7.1), so every test below charges stated costs and treats a result that lives only in the illiquid tail as unproven until gate 5.

## Post-publication decay — the discount applied to every published number

McLean & Pontiff (2016, JF 71(1), 97 predictors): portfolio returns are **26% lower out-of-sample and 58% lower post-publication**; decay is **larger for predictors with higher in-sample returns** and returns stay **higher in high-idiosyncratic-risk, low-liquidity stocks** [S1]. Rule used below: expected post-publication effect = 0.42 × published, and the power calculation is run on that decayed size, not the headline. Where a post-2015 replication exists it overrides the blanket discount.

## The round's candidate list and why three were picked

12 candidates were considered. The three picked fit the data we have or can get for free, have a structural "who is on the other side" story, and each targets a known failure of this account (options sleeve losses; the killed Form 4 root; regime timing). The other nine are recorded with the reason they lost, so a later session does not re-derive them.

| Candidate | Published effect | Post-publication evidence | Fit to our data / capacity | Verdict |
|---|---|---|---|---|
| **Dealer gamma (GEX) → next-days realized vs implied vol** (Baltussen-Da-Lammers-Martens 2021 [S2]; Ni-Pearson-Poteshman-White 2021 [S3]) | Short-gamma dealers trade with the move: intraday momentum present only when net gamma is negative, stronger as it gets more negative (S&P options, 1974-2020 futures) | Published 2021; gamma positioning is widely watched by practitioners (risk that VIX already prices it) | **Free 15-year SPX GEX history exists** (SqueezeMetrics DIX.csv, 2011-05-02 → 2026-10-26, 3,895 rows, fetched this session); index-level so no survivorship; directly informs the short-premium sleeve | **PICK #1** |
| **Opportunistic insider purchases** (Cohen-Malloy-Pomorski 2012 [S4]) | Opportunistic (non-calendar-routine) insiders: **82 bp/month** value-weighted abnormal; routine ≈ 0 | No post-2012 replication found this session; blanket discount → ~35 bp/month | SEC bulk Form 4 data sets back to 2006 (free, `sec_form4_bulk.py`); small caps (EDGE #2); **directly addresses the confound that killed this repo's naive Form 4 test** (experiments.md 2026-07-22: routine 10b5-1/tax trades at price extremes) | **PICK #2** |
| **Implied-volatility spread / put-call parity deviation** (Cremers-Weinbaum 2010 [S5]) | Stocks with relatively expensive calls beat those with expensive puts by **50 bp/week**; stronger where option liquidity high and stock liquidity low; decayed within their own sample | Li-Wang 2026 (2015-2026 panel, 12.4M firm-days): IV spread **significant across regimes** [S6]; a 2024 extension reportedly confirms at 1- and 4-week horizons [S7]; Muravyev-Pearson-Pollet: ~2/3 of option-signal predictability is the stock-borrow fee [S8] | Computable from the chain archive (iv per strike/type) — **if** the indicative feed does not impose parity (gate 1); the CSP sleeve is literally the counterparty to informed put demand | **PICK #3** |
| IV smirk (Xing-Zhang-Zhao 2010 [S9]) | Steepest-smirk decile underperforms by 10.9%/yr | Li-Wang 2026: next-month coefficient −0.023 (t −5.5) 2015-19 → −0.006 (t −1.5) 2023-26, sign flips in the joint model [S6] | Same data as CW | Rejected: decayed to insignificance; CW is the surviving sibling. Logged as a descriptive column in P3, never tested |
| Variance risk premium | Index VRP robust; single-name weak (Driessen-Maenhout-Vilkov, Carr-Wu — bot_logic_research_2026-10-20.md) | — | Already tested here: research test 2 (single-name t 2.15); index leg accruing since 2026-10-21 | Not re-registered (in flight; re-read ≈ 2027-01 per experiments.md 2026-10-22) |
| Option/stock volume O/S (Johnson-So 2012 [S10]) | Low-O/S decile beats high by 0.34%/week | Mixed replications (positive in Korea, none in DAX) [S10] | The archive stores **no option volume** (audit §10.2); same short-sale-cost channel as CW | Deferred: add `dailyBar.v` to the archive first; revisit after ≥ 60 days of volume |
| Term-structure slope (Vasquez 2017) | Slope predicts option (straddle) returns | Not verified this session | Archive caps at 60 DTE and truncates SPY at ~39 DTE (audit §2.4); slope needs a long leg we do not keep | Deferred until the archive keeps a ≥ 90-DTE expiry per name |
| Put-call parity deviations (generic) | = CW volatility spread | — | — | Folded into PICK #3 |
| Lazy Prices (Cohen-Malloy-Nguyen 2020) | Least-changed minus most-changed 10-K language earns positive alpha | — | Harness built and pre-registered (`sec_filings_gate2.py`, v1.0.1100) | Not duplicated: blocked only on the PIT price panel (wishlist 2026-10-18) — the SAME blocker as PICK #2 |
| Post-earnings drift | SUE drift | Martineau 2022: gone from non-microcaps by 2006 [S11]; Subrahmanyam to 2024: t 2.18 all stocks, **t 1.43 ex-microcaps** [S11] | 8-K earnings archive exists | Rejected: survives only in microcaps where our cost measurement is fiction (synthetic fills) |
| Short-term reversal | Strong gross | Profits concentrated overnight (Lou-Polk-Skouras 2019 [S12]); cost-dominated | Daily bars | Rejected: turnover × unmeasured costs; nothing structural keeps it from faster players |
| Overnight vs intraday split (Lou-Polk-Skouras 2019 [S12]) | Momentum earned overnight, reversal intraday; persistent for years | Practitioner replications mixed [S12] | Daily O/C bars exist | Deferred: trading it needs open/close execution twice a day; usable later as a FEATURE decomposition of the momentum score, not a strategy |

Multiple-testing accounting for the round: 12 considered, 3 registered, each with exactly ONE primary statistic. Round-level Bonferroni on three primaries at two-sided 5% gives |t| ≥ 2.39; rounded up to **|t| ≥ 2.5** as the default bar, raised where a root has prior variants (PICK #2). Secondaries are reported, never used to pass.

---

## P1 — Dealer gamma exposure → realized-vs-implied volatility (short-premium timing)

**Hypothesis.** When aggregate dealer gamma in S&P 500 options is low or negative, dealers' delta hedging amplifies moves, so SPY's realized volatility over the next 5 trading days exceeds what VIX implied by more than on high-gamma days. Directional form: the coefficient of the GEX rank on log(realized / implied) is NEGATIVE.

**Why it exists / who is on the other side.** Not mispricing in the classic sense: a mechanical flow. Dealers who are short gamma must buy rallies and sell drops to stay hedged [S2, S3]. The counterparty to our use is the option seller who prices vol off VIX alone. The honest counter: vol desks watch gamma positioning, so VIX may already embed it — the test asks precisely whether it does (incremental to VIX). Why it may persist: hedging is risk-reducing for dealers, not optional, and the flow is predictable without being arbitrageable by the dealers themselves.

**Why it fits this account.** The options sleeve sells premium (CSPs, the QQQ-put overlay) and loses on the days realized beats implied. A gate that holds short-premium entries when gamma is negative is a WHAT-is-traded filter (never an order-path change).

**Data — have it?**
- SqueezeMetrics GEX (SPX, $ gamma per 1% move, dealers assumed long calls / short puts — an ESTIMATE) daily 2011-05-02 → 2026-10-26: **yes, free CSV** (https://squeezemetrics.com/monitor/static/DIX.csv, fetched this session, 3,895 rows). Licence terms not verified: research use only, never redistributed on /data.
- SPY daily OHLC 2011 → : yes (Alpaca SIP history / Yahoo; ETF, no survivorship).
- VIX close 2011 → : yes (Cboe VIX_History.csv, keyless; `server/cboeVix.ts` archives forward).
- Our own GEX (gate 1 cross-check): needs per-contract gamma (recomputable by Black-Scholes from the archive's iv) AND open interest — the archive has no OI; Alpaca's `/v2/options/contracts` schema carries `open_interest` (+date): the `options_entitlements` probe confirms whether it is populated.

**Exact signal.** `G_t` = SqueezeMetrics `gex` on day t (published after the close of t). `R_t` = percentile rank of `G_t` within the trailing 252 trading days ending at t (no lookahead; first 252 days are burn-in, so the test window starts 2012-05).

**Outcome.** `y_t = log( RV_{t+1..t+5} / IV_t )`, where RV = annualized realized variance from SPY close-to-close log returns over days t+1..t+5 (×252/5), IV = (VIX_t/100)². Uses the close of t as the last known price.

**Universe / holding.** SPY only. Horizon 5 trading days (one CSP-management cycle; matches the gamma-hedging literature's days-scale reversion).

**Test (ladder gate 2, SIGNAL).** OLS `y_t = a + b·R_t + c·log(IV_t) + e_t`, Newey-West SE with lag 5 (overlapping windows). Primary: `b` with **t ≤ −2.5** on the full window 2012-05 → 2026-10, AND the same sign with t ≤ −1.5 on the holdout half 2019-01 → 2026-10 (includes post-publication of [S2] and the 0DTE era). Secondary (reported only): (i) the same with Parkinson range instead of close-to-close RV; (ii) the 30-minute intraday-momentum split of [S2] on 1-minute SPY bars by GEX sign; (iii) split by VIX tercile and by the bot's regime labels (REASONING #2). Robustness (declared now): lag `R` one extra day.

**Sample size / power.** ~3,600 usable days, ~720 non-overlapping 5-day windows. With residual sd of y ≈ 0.6 (log units) and sd(R) = 0.29, SE(b) ≈ 0.6 / (0.29·√720) ≈ 0.077, so the minimum detectable slope at t = 2.5 is **≈ 0.19 log points** from the lowest to the highest gamma rank (realized/implied ~21% higher at the bottom). Below that the test cannot distinguish, and says so.

**Base rate.** Unconditionally realized < implied on most SPY 5-day windows (the index VRP); the base rate the test conditions against is the unconditional mean of y and its share of positive windows, reported beside every result.

**Prior (stated before running).** 70% that b < 0 with t ≤ −2.5 in-sample (mechanism well documented); 45% that it also clears the holdout bar (VIX may have learned to price positioning, especially since 2021); 35% that the effect is large enough (≥ 0.19) to matter for the sleeve.

**Kill criteria.** Kill if b ≥ 0, or t > −2.5 on the full window, or the holdout sign flips. Kill at gate 1 instead (data dies, not the idea) if our own BS-recomputed GEX from Alpaca OI disagrees in sign with SqueezeMetrics on > 30% of the overlapping days once ≥ 40 such days exist.

**Ladder path.** gate 1 DATA: SqueezeMetrics vs our own estimate (sign agreement, rank correlation) on overlapping days; the SqueezeMetrics series' own vintage risk stated (a methodology backfill would be invisible to us). gate 2 SIGNAL: above, runnable now. gate 3 LOGIC: the CSP manager-rule simulation (`scripts/csp_manager_rule_sim.py`) with and without "no new short premium when R_{t-1} < 0.2", plus the QQQ-put overlay sized up in the same state. gate 4 SIZING: vs equal size. gate 5 EXECUTION: n/a until it trades.

**Shadow-mode logging spec (no trading change).**
- T-DATACORE: a daily (after 18:30 ET) fetch of DIX.csv's latest row into a `squeeze_gex` datacore stream (date, gex, dix, price, fetched_at; raw overlay label, signal gated), plus a diag probe returning the rank series (aggregate only).
- T-BOT (logging only): every short-premium candidate the options scanner evaluates logs `{gex_rank_prev, gex_neg: bool, gex_date}` into `shadow_portfolio.log_candidate(features=…)`; every executed short-premium entry carries the same three fields in its trade-feedback record. Outcome labels: the existing CSP manager-rule sim result for that entry and SPY's realized/implied ratio over the entry's first 5 days.
- Read date for the shadow book: ≥ 60 short-premium entries in each GEX state (expected ≈ 6 months at the current pace).

---

## P2 — Opportunistic insider purchases (Cohen-Malloy-Pomorski classification)

**Hypothesis.** Open-market purchases (Form 4, code P) by officers/directors who are OPPORTUNISTIC — not calendar-routine — are followed by positive abnormal returns over the next month; the same purchases by ROUTINE insiders are not.

**Why it exists / who is on the other side.** Opportunistic insiders trade on firm-specific news they anticipate (CMP show they predict news, analyst and earnings announcement returns, and SEC enforcement) [S4]. The other side is the market maker / passive seller in small, thinly covered names. Why not arbitraged: the classification needs each insider's 3-year trading calendar (a per-person join most screens skip), the names are small (EDGE #2: capacity-constrained for large funds; McLean-Pontiff find surviving predictability concentrated in high-idio-risk, illiquid names [S1]), and Form 4's two-business-day lag leaves the slow part of the drift. Honest counter: 13 years post-publication and commercial insider screens now ship "routine vs opportunistic" flags — decay is the base case.

**Why this and not a re-run of the killed test.** This repo's Form 4 gate 2 (experiments.md 2026-07-19/07-22) found buys UNDERperforming their own-ticker baseline at 60d (t −3.8) and sales OUTperforming — and diagnosed it as non-informational routine trades (10b5-1, tax, diversification) clustering at price extremes plus a hot own-ticker baseline. CMP's routine filter is the published, theory-motivated remedy for exactly that confound. It is a NEW variant of a killed root, so it carries the root's variant count (see threshold).

**Data — have it?**
- Form 4 transactions with owner identity (REPORTINGOWNER.RPTOWNERCIK), role, code, date, shares, price: **yes, free**, SEC Insider Transactions Data Sets 2006 → (`sec_form4_bulk.py`; raise `BACKFILL_QUARTERS` from 8 to the full history for this test — research run only, not the live archive's retention).
- Daily prices INCLUDING delisted names, keyed by CIK: **no** — the same point-in-time panel Lazy Prices is blocked on (wishlist.md 2026-10-18: free Alpaca inactive-symbol probe first, else EODHD ~$20 / Polygon). One purchase unblocks two pre-registered tests.
- Market cap / size for matching: shares outstanding from EDGAR XBRL company facts (free) × price.

**Exact signal.**
- Insider i trading in year y is CLASSIFIABLE if i has ≥ 1 open-market trade (P or S) in each of the 3 calendar years y−3..y−1. ROUTINE if i traded in the same calendar month in each of those 3 years; OPPORTUNISTIC otherwise [S4]. Unclassifiable insiders are excluded (not pooled).
- Event = a code-P Form 4 (or 4/A) by an officer or director (10% owners / funds excluded, as in the repo's prior), price × shares ≥ $10,000.
- Entry = the OPEN of the first trading day after the SEC FILING_DATE (the bulk set has no acceptance time, so same-day entry is never allowed).

**Universe.** US common stocks with price ≥ $2 and 20-day ADV ≥ $250k at entry (our $2-5k positions stay < 2% of ADV). No size cap: small caps are the point.

**Holding period.** 21 trading days (CMP's monthly portfolio). Secondary: 63 days (comparability with the killed test).

**Test (ladder gate 2, SIGNAL).** Calendar-time monthly portfolio, equal-weighted across all names with ≥ 1 opportunistic event in the prior month, held one month; abnormal return = portfolio minus a characteristic-matched benchmark (size decile × prior-12-month-return tercile, formed from the same panel — momentum matching is mandatory given the repo's confound). Primary statistic: mean monthly abnormal return, Newey-West t (lag 3), **OOS window 2012-06 → latest** (post-publication; 2009-2012 is reported as context only). Threshold **t ≥ 2.8**: the Form 4 root already spent 8 tested cells, so 9 variants two-sided Bonferroni (0.05/9 → |z| 2.77). Secondaries (reported, never used to pass): routine-purchase portfolio (falsification: should be ≈ 0; if routine is as strong as opportunistic, the classification is not doing the work and the result is a size/momentum artifact), value-weighted version, 63-day horizon, opportunistic sales (CMP find these weaker).
**Survivorship rule (declared now).** A run on a survivor-only panel (active tickers) may KILL the hypothesis (a survivor panel flatters purchases, so failing on it is decisive) but may NEVER pass it.

**Sample size / power.** Expected decayed effect ≈ 0.42 × 82 ≈ **35 bp/month**. OOS window ≈ 160 months. Minimum detectable mean at t = 2.8 is 2.8 · sd / √160; it is ≤ 35 bp only if the abnormal-return sd ≤ ~1.6%/month (needs ≥ ~100 names per month). Report the realized sd and MDE beside the result: a non-significant result with MDE > 35 bp is INCONCLUSIVE, not a kill.

**Base rate.** Random entry with the same 21-day hold on the same filtered universe and calendar (size/momentum-matched benchmark) — the benchmark IS the base rate; also report the raw (unmatched) excess so the matching's effect is visible.

**Prior.** 70% the opportunistic-minus-matched mean is positive; 35% it clears t ≥ 2.8 OOS (decay plus power); 75% the routine portfolio is within ±15 bp/month of zero.

**Kill criteria.** Kill if (a) the survivor-only run shows a mean ≤ 0, or (b) the PIT-panel run shows t < 2.8 with MDE ≤ 35 bp, or (c) the routine falsification fails (routine ≥ opportunistic). Archive stays (it is gate-1 data); only the signal dies.

**Ladder path.** gate 1 DATA: already passed for the bulk archive (experiments.md 2026-07-19); add a 25-insider spot-check of the routine/opportunistic labels against the raw filings. gate 2 SIGNAL: above. gate 3 LOGIC: entry/exit ablation in `backtest_v2.py` on the event list (21-day time exit vs the live R-ladder exits — the live stops may cut the slow drift). gate 4 SIZING: vs equal weight. gate 5 EXECUTION: small-cap fills on the open — real fills only (the synthetic haircut cannot validate this).

**Shadow-mode logging spec (no trading change).**
- T-DATACORE: a daily job joins new Form 4 P filings from the live EDGAR feed (`server/edgarForm4.ts`) to each insider's 3-year history (bulk archive + live feed) and writes `insider_opportunistic` JSONL rows `{filing_date, accession, issuer_cik, ticker, owner_cik_hash, role, classification, value_usd, entry_ref: next-open date}` — owner CIK hashed in any served surface; aggregate-only diag probe.
- Labels: +21d / +63d returns vs the matched benchmark, filled by a scheduled labeler; a ledger of would-buys, never orders. Not consumed by deep_score.
- Read date for the shadow book: 12 months of live rows (≈ the first live calendar-time year), compared to the historical estimate (HONESTY METRIC: claimed vs ground truth).

---

## P3 — Option-implied volatility spread (Cremers-Weinbaum) as a short-put screen

**Hypothesis.** Names whose puts are expensive relative to same-strike calls (most negative IV spread) underperform over the next week; for this account, CSPs sold on the lowest-spread quintile lose more than CSPs on the rest.

**Why it exists / who is on the other side.** Informed traders with bad news buy puts when shorting the stock is costly or slow; the options market leads the stock (CW: stronger where option liquidity is high and stock liquidity low [S5]). Muravyev-Pearson-Pollet: about two-thirds of the predictability is the stock-borrow fee, i.e. low-spread names are hard to short [S8]. **The bot's CSP sleeve is the natural counterparty to that informed put demand**: selling puts where puts are richest is being the liquidity provider to the informed side. The useful action needs no shorting: do not sell puts there. Why not arbitraged: the long-short form needs borrow on the very names that are hard to borrow; the screen form costs us only foregone premium.

**Data — have it?**
- Per-contract iv with strike/type/expiry, daily: **yes** for the archived names (indicative feed). **Gate-1 risk:** Alpaca's indicative quotes are "derived values, not actual OPRA quotes" (Alpaca historical-option-data docs). If the derivation imposes put-call parity, the spread is noise by construction and the dataset — not the hypothesis — dies at gate 1.
- Breadth: the archive's list rotates (CSP candidates, 1,717 names in 76 days) — a time series per name needs a FIXED panel. **Not yet** (pipeline change below).
- Open interest weights: no (equal weights primary; OI-weighted secondary once the contracts OI is confirmed by the probe).
- Historical chains for a pre-forward test: no; Databento OPRA (priced 2026-07-04 at ~$740 for 100 underlyings 2016 →; ≈ $2.2k extrapolated for 300) or Polygon if the probe shows entitlement.

**Exact signal.** For underlying u on day d: `VS_{u,d}` = mean over matched pairs j (same strike and expiry, both iv present, expiry 14-60 DTE, call delta 0.25-0.75) of `(iv_call_j − iv_put_j)`. Requires ≥ 3 matched pairs, else missing. Day d is the session whose close the archived snapshot carries (correcting the one-session label shift, audit §2.4 defect 1, until it is fixed at source). XZZ smirk computed beside it as a descriptive column only.

**Universe.** A FIXED, pre-declared panel frozen at the start of logging: SPY/QQQ/IWM plus the 300 most liquid optionable US stocks by 60-day median option-chain quote count at panel formation, plus every name the bot actually sells a CSP on (tagged separately).

**Holding period.** Primary 5 trading days (CW-native). Secondary 21 trading days (the CSP holding period) and the tail outcome P(r_21d < −15%).

**Test (ladder gate 2, SIGNAL).** Daily cross-sections; per day, Spearman rank correlation between VS and the forward 5-day return; the series' mean with Newey-West t (lag 5). Primary bar **t ≥ 2.5**. Secondaries (reported only): bottom-quintile minus rest at 21d; crash probability by quintile (AUC); the CSP-candidate subset; OI-weighted VS.

**Sample size / power — the honest problem.** Decayed effect ≈ 0.42 × 50 ≈ **21 bp/week** top-minus-bottom. With 300 names (60 per quintile) the weekly spread sd is ≈ 1.5%, so t = 2.5 needs (2.5·1.5/0.21)² ≈ **320 weeks (~6 years) forward**. The forward archive can detect only the FULL published size (50 bp) at ≈ 56 weeks. Therefore: (1) the forward test reads ONCE at 52 weeks of the fixed panel — pass only if t ≥ 2.5; t < 2.5 with MDE > 21 bp is INCONCLUSIVE; (2) a powered historical test needs purchased chain history (Databento closing NBBO, 300 names × 2016 → ≈ $2.2k, MDE ≈ 16 bp/week over 520 weeks) — a human spend decision, filed in wishlist terms in open_questions.md, BUILD-FIRST analysis: the free version is the forward archive and needs ~6 years.

**Base rate.** Equal-weight forward return of the whole panel over the same days (the test is cross-sectional, so the base rate cancels in the rank correlation; it is reported for the quintile secondaries and the CSP subset).

**Prior.** 55% that the indicative feed carries a real (non-parity-imposed) spread at gate 1; given gate 1 passes, 60% the sign is right and 25% the 52-week forward read clears t ≥ 2.5 (power); 50% that the lowest-spread CSP candidates show a worse 21-day tail.

**Kill criteria.** Gate 1 kill (dataset): VS is identically ~0 or its cross-sectional sd is under 0.5 vol points on ≥ 80% of days, or its rank correlation with a real-quote reference (Cboe-sourced yfinance chains, or a ~$5 Databento sample, 20 names × 5 days) is < 0.3. Gate 2 kill (signal): the 52-week read has the wrong sign, or t < 2.5 with MDE ≤ 21 bp. Inconclusive is not a kill: it routes to the spend decision.

**Ladder path.** gate 1 DATA: indicative vs real-quote spread (above). gate 2 SIGNAL: above. gate 3 LOGIC: the CSP manager-rule simulation with and without "skip CSP when VS rank is in the bottom quintile". gate 4 SIZING. gate 5 EXECUTION.

**Shadow-mode logging spec (no trading change).**
- T-DATACORE: `server/chainVs.ts` derives `optionchains/derived/vs/<day>.json` (one row per underlying-day: u, session_day, vs, n_pairs, smirk_xzz, atm_iv) like `chainIv.ts`; diag probe `chain_vs` same posture as `chain_iv30`. A separate fixed-panel snapshot job (or a widened, panel-first underlying list in `optionsChainArchive.underlyingsFor`) so the panel is archived every day — its own [PIPELINE] PR.
- T-BOT (logging only): every CSP candidate the scanner evaluates logs `{vs, vs_rank_in_panel, vs_session_day}` in `shadow_portfolio.log_candidate(features=…)`; every executed CSP carries the same in its trade-feedback record. Outcome labels from the CSP manager-rule sim.

---

## SOURCES (opened or abstract/snippet-verified this session, 2026-10-27)

- [S1] McLean, R.D. & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? Journal of Finance 71(1), 5-32 — abstract read from the published PDF (26% / 58% / 32%; decay larger for higher in-sample returns; higher returns in high-idio-risk, low-liquidity stocks).
- [S2] Baltussen, G., Da, Z., Lammers, S. & Martens, M. (2021). Hedging demand and market intraday momentum. Journal of Financial Economics 142, 377-403 — abstract-level (search results; EUR repository).
- [S3] Ni, S.X., Pearson, N.D., Poteshman, A.M. & White, J. — option-trading mechanical effects / net gamma and stock price behavior (working-paper version found; exact RFS 2021 title not confirmed this session — cited for mechanism only).
- [S4] Cohen, L., Malloy, C. & Pomorski, L. (2012). Decoding Inside Information. Journal of Finance 67(3), 1009-1043 — 82 bp/month opportunistic, routine ≈ 0, classification by same-calendar-month trading (NBER digest, SSRN abstract).
- [S5] Cremers, M. & Weinbaum, D. (2010). Deviations from Put-Call Parity and Stock Return Predictability. JFQA 45(2), 335-367 — 50 bp/week; stronger with high option / low stock liquidity; predictability decreases over the sample (IDEAS abstract).
- [S6] Li & Wang (2026). Option-Implied Signals and Crash Risk. arXiv 2608.26115 (preprint, unrefereed) — 2015-2026 panel; smirk decays to insignificance 2023-26; IV spread and risk-neutral skewness significant across regimes.
- [S7] Secondary report of a master's-thesis extension of CW to 2024 (search snippet only — weakest source here, not relied on for numbers).
- [S8] Muravyev, D., Pearson, N.D. & Pollet, J.M. — Why Does Options Market Information Predict Stock Returns? (replication package listing; ~two-thirds of predictability explained by borrow fees).
- [S9] Xing, Y., Zhang, X. & Zhao, R. (2010). What Does the Individual Option Volatility Smirk Tell Us About Future Equity Returns? JFQA 45(3), 641-662 — 10.9%/yr.
- [S10] Johnson, T.L. & So, E.C. (2012). The option to stock volume ratio and future returns. JFE 106(2), 262-286 — 0.34%/week; mixed international replications.
- [S11] Martineau, C. (2022). Rest in Peace Post-Earnings Announcement Drift. Critical Finance Review 11(3-4), 613-646; UCLA Anderson Review summary of Subrahmanyam's 2001-2024 replication.
- [S12] Lou, D., Polk, C. & Skouras, S. (2019). A tug of war: Overnight versus intraday expected returns. JFE 134(1), 192-213.
- Data: SqueezeMetrics DIX.csv (https://squeezemetrics.com/monitor/static/DIX.csv), Alpaca historical option data docs (indicative = derived quotes; history from 2024-02), Massive/Polygon knowledge base (open interest only on snapshots, latest value only, Starter tier and up; no historical OI).
