# Bot logic research round — 2026-10-20 (RESEARCH-ONLY; nothing built, nothing run)

Provenance: human directive 2026-10-20 — "Before we go implementing new things
in the logic, we need to research what is best, like we did when we started
the program." This file is the round's only artifact: no code, config, test or
other research file was touched, the bot was not run, no backtest was run.
Every number is a cited source, a file:line read this session, or a formula
computed in the session scratchpad and restated inline. It follows CLAUDE.md
REASONING STANDARD (priors first, base rates, regime splits, variant discount,
costs first, "who is on the other side"), the ROOT VALIDATION LADDER, RULE
REVIEW and MEASUREMENT INTEGRITY. Source grades: A = peer-reviewed or primary
spec/vendor doc opened; B = working paper or sponsored study opened; C =
independent secondary summary opened; D = practitioner marketing or paywalled
page, partially opened. Unopened sources are listed at the end and never cited
for a number. Keys [S1]..[S36] resolve in SOURCES.

## EXECUTIVE SUMMARY

The six questions share one finding: the system today runs rules that were
either selected by large unreported searches (the configured exits are the best
of a "324-combo backtest", system_config.py:264-267, and are read by nothing),
copied from option-selling folklore whose published evidence is index-level and
gross of costs (21-DTE / 50% / 2x-credit, options_manager.py:50-59), or adapted
from live samples far too small to carry information (strategy weights move on
5-20 trades, server/bot.ts:999-1024; Kelly priors flip after 10 trades,
position_sizing.py:286-289). The literature that exists is clearest on three
points: stops add value only where returns have momentum and subtract it under
random-walk or mean-reverting dynamics (Kaminski-Lo [S1,S2]; Clare et al. [S4]);
the volatility risk premium is a reliable INDEX phenomenon and mostly absent in
single-name options (Driessen-Maenhout-Vilkov [S11], Carr-Wu [S12]), so the
bot's single-name CSP sleeve is selling something the literature says is not
there; and any number produced by selecting among many variants or from a short
sample must be deflated (Bailey-Lopez de Prado DSR [S17], Harvey-Liu [S20],
Lo [S22]) — by amounts that, for the sample sizes the bot actually has, swallow
every edge it has claimed. The five highest-value tests, in priority order, each
with its stated prior: (1) EXIT-MODEL RECONCILIATION on backtest_v2 — legacy vs
live exit engine vs two controls (time-exit-only; wide 3xATR stop + time exit),
per regime, against a random-entry same-holding base rate; prior 60% that the
live R-ladder has LOWER net expectancy than the wide-stop control on momentum
entries and 75% that any stop tighter than 2xATR hurts mean-reversion entries
(gate 3; exploratory arms die 2026-12-31). (2) SINGLE-NAME vs INDEX SHORT
PREMIUM — counterfactual from the fills ledger plus a manager-rule simulation on
historical chains; prior 65% that the single-name CSP sleeve's realized P&L is
explained by equity beta plus spread cost with no VRP residual, and 75% that the
weekly qqq_iron_condor as wired is net negative (gate 2 for the premium, gate 3
for the management rules). (3) BOUNDED EXPLORATION BUDGET for the Kelly-blocked
stocks bucket — at most 2 entries/week at 1% of equity, Thompson-sampled across
momentum/mean-reversion/squeeze, pre-registered stopping rule, 90-day kill;
prior 70% that it CONFIRMS the block (its value is the measurement, not the
P&L). (4) BENCHMARK-RELATIVE FLOOR EVALUATION — floor modelling in backtest_v2,
basket vs SPY vs QQQ-only vs 50/50, reported as IR, tracking error, Ledoit-Wolf
Sharpe-difference interval and DSR with N=4 declared trials, split by regime;
prior 55% that a SPY-core floor has the better benchmark-relative Sharpe and
35% the higher CAGR (unchanged from the 2026-10-14 entry). (5) RESTART-SAFETY
INVARIANTS — no research needed, only measurement: count phantom scale-outs
(#47) from broker sells vs stop_state since 2026-09-01 (prior 60% that at least
one has occurred live), and make the order book broker-derived at boot (#46);
client_order_id on every POST is the standard remedy but the POST path is a
FROZEN PATH, so it goes to wishlist.md for the human. Everything else in this
file is either a lower-priority test or a reason NOT to build.

## Q1. EXIT MODELS for systematic equity swing entries

QUESTION. Which exit shape (ATR vs fixed-% stop, time stop, trailing stop,
partial scale-out, profit target) adds expectancy for momentum, mean-reversion
and squeeze entries, how it interacts with the triple-barrier labeler, and
which ONE model the three divergent models (#48) should collapse to.

WHAT WE DO TODAY (three models, none validated against each other):
- CONFIGURED, read by nothing: system_config.py:268-274 — stop clamp(1.5xATR,
  2-6%), TP clamp(4x stop, 12-25%), TIME_STOP_DAYS 10; regime overrides at
  :462-492 (PANIC MAX_STOP 5%/3d, BEAR 5d, BULL 2.0xATR/10d); hard stop -20%
  at :325; all selected by a "324-combo backtest" (:264-267). KB #48 confirms
  no reader (open_questions.md:7462-7465). KELLY_DIVISOR at :113 is likewise
  read by nothing (grep this session; _kelly_fraction's own default 3.0 rules).
- BACKTEST default: backtest_v2.py:141-144 legacy = clamp(1.5xATR, 2-6%) stop,
  2R target, regime time stop (REGIME_PARAMS :134-140), checked on closes
  (:502-512). A selectable exit_model="live" exists since v1.0.1088
  (:160-161, live_exit_policy.py) but DEFAULT_EXIT_MODEL is still "legacy"
  and the before/after on identical inputs has not been judged
  (experiments.md:108609-108619).
- LIVE, the only code that sells stock: server/bot.ts checkPositionOnTick
  :6886-7256 — R unit = 2xATR% (:6721), phase-1 stop clamp(2xATR%, 1.5-8%)
  (:6736), scale out 1/3 at 1R (:6941) and 1/3 at 3R/2R/1.5R by regime (:6925,
  :6946), breakeven after the first scale-out (:7065), phase-2/3 trails
  max(1.5,1.5xATR)/max(1,1xATR) from PEAK P&L (:7126-7128), phase-4 trail
  ATRx{3,2,1.5} from the high (:7123), TP max(4,min(3xATR,15)) only while
  phase<3 AND both scale-outs done (:7178, i.e. almost never), time stop
  >= 7 CALENDAR days with |pnl| < 2% (:7188). bot_engine.manage_positions
  :2132-2147 computes the same ladder and marks scales_completed=1 /
  breakeven_active without an order (#47).
- LABELS: ml_model_v2._triple_barrier_label :622-660 uses PT 1.5xATR, SL
  1.0xATR, 5-day vertical barrier — a FOURTH exit model; the shadow labeler
  uses +2%/-4% (shadow_portfolio.py:83-84) with an exit-aligned variant at
  :1392. The ML therefore learns "which entries reach +1.5 ATR before -1 ATR
  within 5 days", while the live engine risks 2 ATR and holds 7+ days.

EVIDENCE.
- Kaminski & Lo (JFM 2014) [S1 abstract; S2 for the 2008-draft numbers]:
  under a random walk a simple stop-loss ALWAYS lowers expected return; with
  momentum or regime switching it can add value; on US equity/bond indexes
  1950-2004 (monthly) the rule beat buy-and-hold by ~0.5-1.0%/month during
  stopped-out periods, gross of frictions [S2]; the published abstract says
  only that "at longer sampling frequencies, certain stop-loss policies can
  increase expected return while substantially reducing volatility" [S1].
- Han, Zhou & Zhu, "Taming Momentum Crashes" (2016 WP, via [S3]): US stocks
  1926-2013, monthly momentum deciles, 15% stop via a limit at 1.5x the
  threshold: EW monthly return 0.99% -> 1.93%, SD 6.01% -> 4.61%, Sharpe
  0.17 -> 0.40, worst month -49.8% -> -17.4%; turnover 39% -> 49% (winners),
  45% -> 59% (losers); most of the gain is on the SHORT side; GROSS; the
  reviewer flags illiquid stop executions and hindsight-chosen thresholds.
  Limited relevance: the bot is long-only with intraday 1.5-8% stops.
- Clare, Seaton, Smith & Thomas (J. Asset Mgmt 2013) [S4]: S&P 500 daily
  Jul 1988-Jun 2011, 0.2% switching cost; trailing stops of 3-15% after a
  200-day MA entry: "In both cases stop-loss rules would seem to make
  performance worse"; purchase-cost stops "have no beneficial impact"; "a
  change of trend is the best stop loss"; and for mean reversion, "the
  investor is stopped out after a fall only to be left stranded as the
  portfolio recovers" [S4 §4].
- Mean-reversion entries: one independent blog test [S5] (SPY long-only MR,
  2000-2026, costs modelled): no stop CAGR 8.22% / max DD -18.63% / WR
  66.7%; same entries with a 5% stop: 1.05% / -41.78% / 49.1%. Grade D
  (entry rules paywalled, one system) — consistent with [S1,S4], not
  evidence alone. Connors/Alvarez RSI(2) stop tests could not be opened.
- Short-DTE option stops [S6] (same mechanism: a stop realizes the loss the
  premium is paid for): a 200%-of-credit stop turned a 30-DTE 45-delta cell
  from +30.6% to -16.5%; mid fills "overstate CAGR by 30-60%". Grade D —
  cited for the direction of the cost bias only.
- Triple-barrier labels (ml_model_v2.py:622-660): +1 if PT (1.5 ATR) is
  touched first, -1 if SL (1.0 ATR), 0 at the 5-day barrier; the docstring's
  claim that it "maps directly to actual bot behavior" is false (see above).
  The labeler's own purpose is to mirror the deployed exit; the 2026-10-15
  exit-aligned labeler PR acts on that. Lopez de Prado's book not opened.
- Partial scale-outs / R ladders: no study isolating them was found; under
  [S1]'s framework a scale-out is a partial profit target (hurts under
  momentum, helps under mean reversion). UNKNOWN empirically.

ESTABLISHED: stops' value is a property of the return process, not of the
stop; they subtract expectancy under random walk and mean reversion and add it
only with persistence [S1,S2,S4]. Costs scale with the number of stop events;
every gross stop study overstates net value [S3,S4,S6]. CONTESTED: whether
daily-frequency equity stops add value even under momentum — [S3] says yes at
15%/monthly for long-short deciles, [S4] says no at 3-15% trailing on a
trend-following index position. UNKNOWN: scale-out ladders; squeeze entries;
anything at the bot's 45-second cadence; the interaction of breakeven stops
with intraday noise (a breakeven stop armed at 1R on a 2xATR R unit is a
stop 2 ATR above the original entry risk, fired by the first mean reversion).

RECOMMENDATION (testable form, one PR per arm):
1. Ladder gate 3 (LOGIC), [RULE-REVIEW]: run backtest_v2 with exit_model=
   legacy and =live on the same universe/period, plus two CONTROL arms added
   to live_exit_policy as opt-in variants: (a) time-exit only (no stop, no
   TP, exit at the regime time stop); (b) 3xATR stop + the same time exit, no
   scale-outs. Report per arm and regime (BULL/NEUTRAL/CAUTION/BEAR): net
   expectancy per trade, share of each exit reason, holding days, Sharpe with
   Lo's SE [S22], max DD, turnover-implied cost. BASE RATE: random entries in
   the same universe with each arm's holding-day distribution (REASONING
   STANDARD #3). Prior: 60% the live ladder underperforms arm (b) net on
   momentum entries; 75% arms (a)/(b) beat it on mean-reversion entries; 50%
   on squeeze (no literature). Exploratory arms die 2026-12-31. DISCOUNT: 4
   arms x 4 regimes; a winner must clear the DSR threshold for N=4 (~1.0 x
   sqrt(V[SR]) [S17 eq. 1], scratchpad) before it is believed.
2. Reconciliation rule for #48: after test 1, the model that wins net, per
   regime, becomes THE model; live_exit_policy.py is already the single
   source for the live rules, so the losing code (the unread system_config
   block, or the live ladder) is deleted under STALENESS, and
   DEFAULT_EXIT_MODEL flips to whatever live runs. Until then every stored
   backtest number remains a measurement of exits the bot does not run.
3. Label alignment: the ML labeler's barriers must equal the deployed model's
   (PT/SL/horizon), or the ML is trained on a different game; keep the old
   labels alongside for the before/after (the 2026-10-15 PR's design).

DISCARDED: fixed-% stops (no study found favouring them over ATR-scaled
stops, and the labeler's own cross-sectional-volatility rationale argues
against them); re-tuning the 324-combo grid (E[max] ~2.9 x sqrt(V) for N=324
[S17] — any new optimum is noise unless deflated); re-enabling the -20% hard
stop (a gap guard is a kill-switch mechanism question, not an expectancy rule).

## Q2. SHORT-PREMIUM OPTIONS MANAGEMENT

QUESTION. Is the 30-45 DTE entry / 21-DTE management / 50% profit / 2x-credit
loss / IC 50%-of-max-loss rule set supported, and are weekly condors and
high-IV single-name put selling worth it net of spread and assignment risk?

WHAT WE DO TODAY: options_manager.py:50-59 — DTE_EXIT_THRESHOLD 21,
DTE_CRITICAL 5, PROFIT_TARGET_PCT 0.50, LOSS_LIMIT_MULTIPLIER 2.0,
IC_MAX_LOSS_EXIT_PCT 0.50, MIN_HOLD_MINUTES 60; grouped exits :636-808 (rule
order: critical DTE, min hold, 50% profit, IC 50%-of-max-loss, 2x-credit loss,
21 DTE, SABR edge-evaporation); standalone 21-DTE roll-or-close :1143-1165.
Entry floors: options_execution.py:104-135 — CSP_ENTRY_MIN_DTE = max(T1 30,
21+1) = 30 (tiered_strategy.py:182-183), MULTI_LEG_ENTRY_MIN_DTE = 30 as of
v1.0.1117; the weekly qqq_iron_condor (3 DTE), covered_call, buy_call/buy_put
(7 DTE) remain inside the exit zone (ENTRY/EXIT DTE CONFLICTS entry,
open_questions.md:22924-22931). Liquidity gates MIN_OPEN_INTEREST 200,
MAX_SPREAD_PCT 0.10 (:88-90). The IC rule's comment cites "Tastylive research"
(:706-712) — a source this session could not open (see NOT OPENED). The
csp_options Kelly prior is 71.6% WR / +0.69 / -0.46 from the 10-year backtest
(position_sizing.py:270-284); live CSPs are on single names (XLF/HPE/NOK/NU in
KB #46 and the v1.0.1056 entry).

EVIDENCE.
- Index put-writing, monthly vs weekly (Bondarenko 2016, Cboe-sponsored [S7];
  independent summary [S8]): PUT (monthly ATM, T-bill collateral) vs WPUT
  (weekly), Feb 2006-Dec 2015, GROSS: annual return PUT 6.6%, WPUT 5.6%, S&P
  500 TR 7.1%; SD 11.5% / 9.8% / 15.1%; Sharpe 0.52 / 0.50 / 0.46; max DD
  -33% / -24% / -51% [S8]; gross premium collected 24.1%/yr (PUT) vs 39.3%/yr
  (WPUT) [S7] — the weekly index collected 63% more premium and returned
  LESS, before costs that scale with 52 vs 12 rolls. PUT Jul 1986-Dec 2015:
  10.1% vs 9.8%, SD 10.1% vs 15.3% [S7,S8]; pre-launch history is back-filled
  and BXM-tracking products "substantially underperformed" their index [S8].
  COI: Cboe sells the listed products.
- Which index options to sell (Israelov & Tummala, AQR WP 2017 [S9]): S&P 500
  options Mar 1996-Dec 2015, delta-hedged daily; puts beat calls; shorter
  maturities earn more per trade but longer ones carry lower volatility and
  stress loss; near-the-money beats deep OTM on return; "most compensated per
  unit of stress-test loss" = front-month, near/moderately OTM; GROSS;
  "different risk metrics can lead to significantly different conclusions".
  COI: AQR runs volatility-selling funds. Israelov & Nielsen (FAJ 2014 [S10]):
  an index covered call = long equity + short volatility (beta 1.00 -> 0.64)
  and "makes sense only if the investor is confident that the volatility
  risk premium exists" — an INDEX result; the bot's covered_call is a weekly
  single-name roll.
- Single-name VRP, the decisive point: Driessen, Maenhout & Vilkov [S11]:
  S&P 500 implied vol 23.09% vs realized 20.69% (equality strongly rejected);
  constituents implied 43.38% vs realized 41.99%, NOT significantly
  different; zero individual VRP not rejected for 503 of 919 S&P 500 stocks
  at 30 days (27 of 43 DJ30), point estimate negative at 91 days; they read
  Carr & Wu and their own JF 2009 paper as unable to reject a zero premium
  "for the large majority of stocks". Carr & Wu [S12]: five indexes + 35
  most-active stocks; premia "strongly negative" for S&P 500/100 and Dow;
  single stocks "negative, but with a smaller absolute magnitude", mean VRP
  "insignificant for all but three of the 35". Second-order: the index
  premium is a CORRELATION premium [S11] — diversification insurance that a
  single name does not contain, so there is no structural seller's edge.
- Management rules (50% / 21 DTE / 2x credit): the only sources are
  tastylive-style brokerage-affiliate studies that could not be opened —
  grade D, UNVERIFIED. The one opened practitioner sweep [S6] found a
  2x-credit stop destructive at 30 DTE and mid fills overstating CAGR by
  30-60%. No peer-reviewed test of the 21-DTE rule was found.

ESTABLISHED: a negative VRP exists for S&P 500 index options and is
economically large [S11,S12,S7]; short-dated index puts earn the most per
trade but with the worst tails [S9]; everything published is gross and the
strategies turn over far more than the underlying [S8]. CONTESTED: whether
weekly selling beats monthly after costs — the sponsored study's own numbers
say no on return and yes on drawdown [S7,S8]. UNKNOWN: the 21-DTE / 50% /
2x-credit management triplet (no openable evidence either way); whether the
liquidity gates (OI 200, spread 10%) keep the bot in contracts where the
spread exceeds the premium edge; assignment frequency on the bot's names.

RECOMMENDATION (testable form):
1. Gate 2 (SIGNAL), [RULE-REVIEW]: for every single-name CSP/condor
   underlying the bot has traded since 2026-05-06, compute the 30-day implied
   vs subsequently realized variance from the chain snapshots the scanner
   already stores (or from the fills' entry IVs), and the same for QQQ/SPY;
   report mean VRP with Newey-West SEs as in [S11]. Prior: 65% that the
   single-name average is indistinguishable from zero and 85% that the index
   average is negative. If the prior holds, the single-name short-premium
   sleeve has no gate-2 signal and may not be sized by a VRP-harvest prior.
2. Gate 3 (LOGIC): a manager-rule simulation on historical chains for the
   names actually traded, arms = {hold to expiry; 50% profit only; 50% + 21
   DTE; 50% + 21 DTE + 2x-credit stop} (declared N=4), each at mid fills and
   at ask+1 tick / bid-1 tick, assignment modelled at expiry; base rate = the
   same-dated short-SPY-put P&L scaled to the name's beta. Prior: 60% the
   2x-credit stop lowers expectancy (it realizes the left tail the premium
   pays for, cf. [S6]); 55% the 21-DTE close is neutral-to-negative net of a
   second spread crossing. Exploratory arms die 2026-12-31.
3. Measurement first (KB #12c): options exits must carry realized P&L in
   trade_feedback before any of the above can be judged live; the fills
   ledger (v1.0.1121) is the ground truth until then.
4. ENTRY/EXIT conflicts: retire qqq_iron_condor (3 DTE into a 5-DTE critical
   close is unreachable by construction) and floor buy_call/buy_put at the
   shared 30-DTE floor — prior 70% net-positive vs status quo (unchanged from
   the 2026-10-20 entry); this is a design call for the human, logged there.

DISCARDED: "high-IV single names" as an edge (the premium is an index /
correlation premium [S11,S12]; high single-name IV is an event forecast and
the earnings gate is the instrument, not a sizing bonus); copying tastylive
thresholds (unopened marketing); weekly QQQ condors (more premium, lower
return, 52 spread crossings/year [S7,S8]).

## Q3. STRATEGY/SLEEVE ALLOCATION under uncertainty and the LEARNING DEADLOCK

QUESTION. How should fractional Kelly be set when win-rate and payoff are
estimates, how should a probation strategy be sized and judged, and what
exploration budget breaks the stocks-bucket deadlock at bounded cost?

WHAT WE DO TODAY: position_sizing._kelly_fraction :153-208 — f = (p*W -
q*L)/W over a fixed divisor 3 (the KELLY_DIVISOR key at system_config.py:113
is never passed); negative expectancy returns 0.0 and calculate_position
hard-blocks (:950-955); avg_loss <= 0 gives a 0.5% "starter" (:195). Priors
(:262-290) are plug-in point estimates from the 10-year backtest: stocks
0.49 / 4.19 / 4.25 (negative by construction: 0.49*4.19 - 0.51*4.25 = -0.11,
though the comment expected a starter size), csp_options 0.716 / 0.69 / 0.46,
etf 0.558 / 3.82 / 3.55; live stats replace the prior at >= 10 closed trades
(:286-289, :305-310). Live stocks: WR 27.8% over 36 trades
(open_questions.md:7647-7648); deadlock and prior filed at :7642-7665.

EVIDENCE.
- MacLean, Thorp & Ziemba [S13]: Kelly maximizes long-run growth and never
  risks ruin, but "given the extreme sensitivity of E log calculations to
  errors in mean estimates... the size of the wagers should be reduced";
  double Kelly grows at the risk-free rate; 700 wagers at a 14% edge can
  still turn $1000 into $18. Thorp 2006 [S14 §7-8]: fraction c of Kelly
  keeps c(2-c) of the growth with risk scaling as c; half Kelly keeps 3/4 of
  the growth while P(ever losing half) falls from 1/2 to 1/8; "overbetting
  is much more harmful than underbetting". Third Kelly (the bot) keeps 5/9.
- Baker & McHale (Decision Analysis 2013) [S15 abstract]: plug-in Kelly
  "ignores uncertainty in the probability of winning"; shrinking the stake
  for estimation error beats raw Kelly in simulation and on tennis data.
  Scratchpad check of the mechanism (true WR 55%, even money, WR estimated
  from n past trades, formula in calc.py): n=20 plug-in full Kelly has
  NEGATIVE realized growth (-0.011/bet vs oracle +0.005); half/third Kelly
  stay positive (+0.0013/+0.0023); at n=200 half Kelly is best (+0.0033).
  The "right" divisor depends on n: small samples want a larger one.
- Sample sizes (Wilson 95% intervals, scratchpad): live stocks 10/36 -> WR in
  [15.8%, 44.0%] — the block is justified at 95%; the backtest prior 37/83 ->
  [34.4%, 55.3%] — it cannot distinguish a losing bucket from a winning one;
  5 trades at 2/5 -> [11.8%, 76.9%]. Trades needed to tell WR 55% from 50%
  at alpha 5% one-sided / power 80%: ~616; 60% from 50%: ~152.
- Thompson sampling [S16]: Beta-Bernoulli posterior per arm, one draw per
  arm per period, play the max; under drift "the agent should never stop
  exploring" — use a sliding window or discount the posterior toward the
  prior; epsilon-greedy "wastes resources by failing to write off actions".
  No source on how funds run "probation" books was found (UNKNOWN); the
  design below is assembled from the pieces above.
- Minimum track record [S17 citing Bailey & LdP 2012; formula via S19]:
  MinTRL = (1 - skew*SR + (kurt-1)/4*SR^2) * (z/(SR - c))^2 periods;
  scratchpad (Gaussian, c=0, 95%, daily): annualized SR 0.5 needs ~10.8
  years, 1.0 ~2.7 years, 1.5 ~1.2 years. A probation sleeve cannot prove
  itself on P&L inside a quarter; it can only rule out a LARGE edge and
  feed labelled entries to the counterfactual logger.

ESTABLISHED: fractional Kelly with the fraction tied to estimation error, not
a fixed 1/3 [S13,S14,S15]; Kelly on a negative point estimate from n=36 is
correctly zero and the deadlock is real. CONTESTED: nothing material.
UNKNOWN: practitioner probation sizing; whether the bot's three stock
strategies have different enough edges for a bandit to separate within a year.

RECOMMENDATION (testable form; gate 3; the sizing change itself is a
RULE-REVIEW PR, one threshold at a time):
1. EXPLORATION BUDGET, pre-registered: when a bucket is Kelly-blocked, allow
   at most k=2 entries/week at a FIXED 1.0% of equity (not Kelly-sized),
   tagged exploratory, the strategy chosen by Thompson sampling over
   momentum / mean reversion / squeeze with Beta(1,1) priors discounted
   weekly (gamma 0.9 [S16]). Bounded cost: 2/week x 1% x 13 weeks x the live
   avg loss 5.16% (open_questions.md:7648) ~ 1.3% of equity per quarter if
   every entry loses. Stopping rule, stated now: at 90 days (kill 2027-01-20)
   or 40 entries, compute the Wilson interval of the exploratory WR per
   strategy and the +5d/+20d counterfactual P&L of the rejected_sizing
   shadow bucket; unblock only if the interval's LOWER bound exceeds the
   breakeven WR implied by the realized payoff ratio (36.2% at +9.08/-5.16;
   66.7% under the +2/-4 labeler). Prior: 70% the block is CONFIRMED; 20%
   that one strategy (ordering mean reversion > squeeze > momentum, from the
   2026-10-16 shadow finding that taken candidates win less than rejected
   ones, shadow_portfolio.py:1871) clears it in BULL only.
2. KELLY DIVISOR BY SAMPLE SIZE: replace the constant 3 with a divisor that
   rises with estimation error, e.g. divisor = 3 at n >= 200 closed trades
   and 3 * sqrt(200/n) below, floor 3 — the scratchpad bootstrap shows that at
   n=20 even half Kelly over-bets. Test on backtest_v2 per bucket vs the
   constant; prior 60% that growth is unchanged and drawdown lower.
3. Report the MinTRL for every sleeve on the Bot page next to its Sharpe so
   "unproven" is visible, not inferred.

DISCARDED: re-raising the stocks prior to force trading (the live interval
already excludes WR >= 44%); epsilon-greedy exploration (uniform waste
[S16]); full Thompson sampling over CAPITAL (Kelly already sizes; the bandit
should only choose WHICH blocked strategy gets the fixed exploratory slot).

## Q4. BENCHMARK-RELATIVE EVALUATION of the passive ETF floor vs SPY

QUESTION. How should the floor (NEUTRAL: QQQ 45 / SMH 13.5 / KWEB 9 / VXUS
13.5 of equity = 90% x FLOOR_BASKET weights at system_config.py:151-157, plus
GLD 8 from DEFENSIVE_FLOOR_NEUTRAL :196) be compared with SPY so the
comparison is fair, regime-conditioned and deflated for the number of
variants tried?

WHAT WE DO TODAY: /api/bot/benchmark (v1.0.1118) reports account vs SPY total
return since inception and over 30 days (since 2026-05-06: -6.90% vs +5.24%,
gap -12.15 pt; experiments.md:109053); the sleeve ledger (v1.0.1121) splits
realized P&L by options / floor / stocks. No tracking error, information
ratio, beta-adjustment or Sharpe-difference test exists. The floor weights
were chosen from an 8-config 10-year sweep (system_config.py:118-125: "QQQ
floor + VRP30/Sector25 = best: 18.7% CAGR vs SPY 12.3%") and later a note that
the active system's best config returned 5.68% CAGR vs SPY 13.97% (:128-134).
backtest_v2 does not model the floors (open_questions.md:22920).

EVIDENCE.
- Information ratio: Goodwin (FAJ 1998) [S24, abstract]: "the simplest form
  and interpretation of the ratio is the most useful", IR relates directly to
  a t-statistic. Standard definition (not a citation): IR = mean(active
  return) / SD(active return), active = portfolio - benchmark, annualized;
  tracking error = SD(active return).
- Sharpe precision: Lo (FAJ 2002) [S22, eq. 9]: SE(SR) = sqrt((1 + SR^2/2)/T)
  per period under IID; with 115 daily observations (the ~5.5 months since
  inception) SE ~ 0.093 per day, i.e. ~1.5 annualized — the account's Sharpe
  cannot yet be distinguished from zero or from SPY's. Annualizing by sqrt(12)
  or sqrt(252) is wrong under serial correlation [S22].
- Comparing two Sharpes: Ledoit & Wolf (JEF 2008) [S23]: the Jobson-Korkie/
  Memmel test "is not valid when returns have tails heavier than the normal
  distribution or are of time series nature"; use a studentized time-series
  bootstrap confidence interval for the DIFFERENCE and call them different
  only if zero is outside it [S23 abstract].
- Selection bias: Bailey & Lopez de Prado DSR [S17 eq. 1]: E[max SR] of N
  zero-skill trials = sqrt(V[SR]) * ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e))),
  g = 0.5772; scratchpad: N=8 (the floor sweep) ~1.4, N=324 (the exit grid)
  ~2.9, N=1000 ~3.3 SDs of the trial dispersion. DSR deflates the chosen SR
  by that threshold using N, V[SR], T, skew and kurtosis [S17 eq. 2];
  "Investment theory, not computational power, should motivate what
  experiments are worth conducting" [S17]. PBO [S18]: hold-out "tends to be
  unreliable and inaccurate" for backtests; CSCV estimates the probability
  the in-sample best is out-of-sample below median.
- Haircuts: Harvey & Liu (JPM 2015) [S20]: the flat 50% haircut "is a serious
  mistake"; the multiple-testing haircut is nonlinear — "the highest Sharpe
  ratios are only moderately penalized, while the marginal Sharpe ratios are
  heavily penalized"; Bonferroni / Holm / BHY given; Harvey, Liu & Zhu
  (RFS 2016) [S21]: a new factor "needs to clear a much higher hurdle, with a
  t-statistic greater than 3.0".
- Decay: McLean & Pontiff (JF 2016) [S25]: 97 published predictors earn 26%
  less out-of-sample and 58% less post-publication; declines are larger for
  predictors with higher in-sample returns. The floor's 2016-2025 tech-tilt
  premium is an in-sample number of exactly this kind.

ESTABLISHED: report IR/TE, Sharpe SEs, a bootstrap interval for the Sharpe
difference, and a DSR with the trial count declared; a 10-year sweep of 8
configs selecting "best" is a biased estimate by ~1.4 SD of its dispersion.
CONTESTED: none of the statistics; only the floor thesis itself, which is the
human's. UNKNOWN: the floor's live beta to SPY (not yet measured); the share
of the -12 pt gap due to the floor vs the active sleeves (the sleeve ledger
will show it once deployed).

RECOMMENDATION (testable form; gate 3 then 4; the composition is the human's
decision — this is the evidence request):
1. Floor modelling in backtest_v2 (measurement PR, [RULE-REVIEW], with its
   own before/after — account-level numbers will move; state the direction).
   Arms declared in advance (N=4): basket as configured; SPY-core (QQQ/SMH
   weight -> SPY); 50/50; QQQ-only (the v1.0.29 original). 2016-01 to date,
   regime labels from the engine's classifier. Per arm and regime: CAGR, max
   DD, Sharpe +/- Lo SE, beta to SPY, Jensen alpha, TE, IR, Ledoit-Wolf 95%
   interval of (Sharpe_arm - Sharpe_SPY), DSR with N=4. "Same exposure"
   means the beta-matched SPY position (the floor is ~77% invested live, per
   the 2026-10-14 entry), not 100% SPY. Prior (2026-10-14 entry, unchanged):
   55% SPY-core has the better benchmark-relative Sharpe; 35% the higher CAGR.
2. Live: add TE and IR (30/90-day) and beta to the benchmark tile next to the
   gap, with the SE so a one-month number cannot read as a verdict.
3. Any future weight change that passes test 1 ships as one threshold change
   with a rollback trigger (TE > 2x the backtest TE over 60 days, or IR
   negative over 120 days).

DISCARDED: judging the floor on the 30-day gap (SE swamps it [S22]); a new
sweep of weights (selection bias [S17]); replacing the human's thesis by a
session decision (constitutional: basket composition is the human's call).

## Q5. ORDER-STATE RECONCILIATION and RESTART SAFETY

QUESTION. What is standard practice for idempotent submission, reconciling
local state with the broker after restarts, never inferring a fill, and
avoiding cancel/replace loops across restarts (#46, #47)?

WHAT WE DO TODAY: every order POST in server/bot.ts (:3301, :4045, :4181,
:5051, :5544, :5749, :6142, :6563) and the Python submit path send NO
client_order_id (grep this session: the only mentions are server/diag.ts:336
dropping it from display and its test); Alpaca then auto-generates one [S27].
gracefulShutdown :826-838 cancels ALL open orders on every SIGTERM (DELETE
/v2/orders, also at :1253, :2456, :4026); the F7 cooldown lives in process
memory (KB #46, open_questions.md:7378-7386); replaceIfBetter treats the 204
of a successful cancel as a failure (#46 secondary (b)); the open-orders fetch
feeding the CSP gate fails OPEN (#46 (c)); Python marks scale-out 1 done with
no order (bot_engine.py:2141-2147) and Node adopts the larger
scales_completed / smaller remaining_qty (bot.ts:6746-6768) — KB #47.

EVIDENCE (vendor docs and standards, grade A/B unless noted):
- Alpaca: client_order_id is "a unique identifier for the order", max 128
  chars, auto-generated when absent, retrievable via GET
  /v2/orders:by_client_order_id [S27,S30]; what a DUPLICATE submission
  returns is not documented (UNKNOWN — probe on paper). On a submission
  timeout the order "may already have reached the market. Don't resend it
  or mark it canceled until Alpaca confirms its status" [S29]. States are
  explicit: pending_cancel = "waiting to be canceled"; pending_replace
  "will reject cancel request"; aged GTC orders can sit in pending_cancel;
  filled/canceled/expired are terminal [S28]. trade_updates delivers
  new/fill/partial_fill/canceled/expired/replaced/rejected/pending_* with
  position_qty after each fill, and gives NO reconnection or replay
  guidance — events missed during a disconnect are missed [S31]. Cancel-all
  returns HTTP 207 with a per-order sub-status; a failed cancel leaves the
  order accepted [S32].
- FIX: ClOrdID uniqueness is the SENDER's job; receivers key on
  SenderCompID + ClOrdID [S33]. The order-state matrices rank statuses
  (Pending Cancel 12 > Pending Replace 11 > Done for Day 10 > Filled 8 >
  Stopped 7 > Canceled 5 > Partially Filled 4 > New/Rejected/Pending New 2)
  and an unknown-order status request gets a reject, not silence [S34]. A
  venue's recovery procedure [S35]: after reconnect the gateway replays
  missed execution reports "including execution reports for any canceled
  orders" and "Platforms must process these reports to accurately
  reconstruct their order state"; replayed NEW orders must never be
  re-executed; from 2026-10-19 that venue rejects a replayed NewOrderSingle
  whose ClOrdID was already used — client idempotency keys are the standard.
- General engineering (Stripe [S36]): a request can fail ambiguously after
  the server completed the work; the client sends a key per logical
  operation, the server returns the cached result on replay, and clients
  retry with backoff until success is verified — never infer the outcome
  from the absence of a response.

ESTABLISHED: (i) every submission carries a client-generated unique id;
(ii) state is reconstructed from the broker's order and fill records on every
start, never from local memory; (iii) a fill exists only when the broker
reports it (fill event or order status), never when a price threshold is
crossed; (iv) cancel is asynchronous — act on canceled/expired, not on the
HTTP code of the cancel request; (v) a timeout is "unknown", resolved by
lookup (by client id), not by resubmission [S27-S36]. CONTESTED: nothing.
UNKNOWN: Alpaca's behaviour on a duplicate client_order_id (probe on paper);
whether trade_updates can be replayed at all (the docs say nothing).

RECOMMENDATION (engineering, gate-free; each its own [REPAIR] PR with a
regression test; items touching the raw POST path are FROZEN and go to
wishlist.md):
1. MEASURE #47 before fixing: for every closed stock position since
   2026-09-01, join broker sells (fills ledger) to stop_state scale history;
   count (a) scales_completed >= 1 with no matching sell and (b) residual
   shares after a "full" exit. Prior 60% that (a) > 0. If so, the fix is the
   single-owner invariant in #47: Python never writes scale-out STATE; Node
   writes scales_completed only after the broker ACCEPTS the order and
   remaining_qty only after the FILL event; a fuzz test asserts
   manage_positions never mutates either field.
2. BOOT RECONCILIATION (#46 NEXT): at start rebuild openOrders and the CSP
   brake from GET /v2/orders (open + closed since now - cooldown - 12m)
   [S28,S30]; derive "rested >= STALE_ORDER_MINUTES then canceled within the
   cooldown" from broker timestamps (indifferent to who canceled); fail
   CLOSED when the fetch fails; re-read the clock at dispatch. Regression
   test as written in #46 (open_questions.md:7431-7432).
3. CANCEL SEMANTICS: a 204 means "cancel requested"; confirm via status in
   {canceled, expired} or a canceled event [S28,S31]; reuse
   confirmCancelAfterDeleteError in replaceIfBetter (#46 (b)).
4. WISHLIST (frozen POST path): a deterministic client_order_id (e.g.
   `vt-<ticker>-<intent>-<minute-bucket>`, <= 128 chars [S27]) on every POST,
   timeouts resolved by GET by_client_order_id instead of resubmission
   [S29,S30] after probing the duplicate-id behaviour on paper; and whether
   gracefulShutdown should cancel only ENTRY orders and leave resting exits
   (a cancel-all on every docs-only deploy is the generator of #46; 17
   merges -> 17 SIGTERM cancel-alls in market hours on 2026-10-14).
   railway.json watchPatterns / deploy windows are likewise wishlist items.

DISCARDED: inferring fills from price (the phantom scale-out is exactly this);
retrying a timed-out POST without an id; persisting the F7 cooldown to disk
alone (a container swap can still lose the volume write race; the broker's
own timestamps are the durable record).

## Q6. LEARNING HYGIENE for a self-adjusting bot

QUESTION. When is it legitimate to adapt weights/thresholds from live results
(sample size, stationarity, multiple-variant discount) versus running
pre-registered experiments with kill dates?

WHAT WE DO TODAY: adjustStrategyWeights (server/bot.ts:999-1024) moves
strategy weights every cycle on the win rate of the last 5-20 post-deploy
trades, computed on `pnl`, which trackClosedTrades never fills — a constant
0% that nothing reads (experiments.md v1.0.1120 entry; removal parked behind
a test assertion). Kelly priors yield to live stats at >= 10 closed trades
(position_sizing.py:286-289). ML retrains every 20 real trades
(system_config.py:279) on triple-barrier labels with code_version gating.
Exits were selected from 324 combinations (system_config.py:264), the floor
from 8 (:118). REASONING STANDARD #4 and #10 are the bar; the counterfactual
logger and shadow_portfolio.py are the instruments.

EVIDENCE.
- Preregistration (Nosek et al., PNAS 2018 [S26]): "Mistaking generation of
  postdictions with testing of predictions reduces the credibility of
  research findings... An effective solution is to define the research
  questions and analysis plan prior to observing the research outcomes";
  practical variants exist "when the data are pre-existing" (i.e. for
  backtests: declare the arms, the metric and the stopping rule before
  looking). This is REASONING STANDARD #10 with a literature behind it.
- Multiple testing: Harvey, Liu & Zhu t > 3.0 [S21]; Harvey & Liu's
  nonlinear haircut with 316 prior tests [S20]; DSR needs N and V[SR]
  declared [S17]; hold-out alone is not enough [S18].
- Sample size and drift: Wilson intervals (scratchpad): n=5 -> [12%, 77%];
  n=20 at 40% -> [22%, 61%] — the adjuster's 40%/65% thresholds sit INSIDE
  the noise of its own sample; ~616 trades separate WR 55% from 50%. Under
  drift a learner must keep exploring and discount old evidence [S16];
  published edges decay 26% out of sample and 58% post-publication [S25], so
  a live-vs-backtest gap of that order is the BASE RATE, not an anomaly.
- Estimation sensitivity: Kelly's "extreme sensitivity... to errors in mean
  estimates" [S13] applies to every parameter re-fit from a short window.

ESTABLISHED: adapt only (a) with a pre-registered rule, (b) on samples whose
confidence interval excludes the status quo, (c) with the number of variants
declared and the result deflated, (d) with a kill date, (e) on a metric the
change could not have biased (MEASUREMENT INTEGRITY). CONTESTED: how much to
discount old regimes (window vs decay — [S16] offers both). UNKNOWN: the
true stationarity horizon of this bot's edges (nothing measured yet).

RECOMMENDATION (testable form):
1. Freeze all online adaptation that is not pre-registered: remove the
   strategy-weight adjuster (dead code with a false "LEARN" audit trail) once
   the human releases the 2026-08-07 test assertion; require n >= 30 closed
   trades per bucket AND a Wilson lower bound above the breakeven WR before a
   live stat replaces a Kelly prior (prior 80% that this changes no sizing
   today — every bucket is blocked or under 30).
2. One EXPERIMENT REGISTER format for every live adaptation and backtest
   sweep: hypothesis, arms (N), metric, regime splits, base rate, prior,
   sample needed (MinTRL or the binomial n), kill date, DSR threshold for
   that N; an undeclared N is not evidence (REASONING STANDARD #4 made
   mechanical). Budget by the secretary rule [S17]: sample ~37% of the
   theory-justified configurations at random, then stop at the first that
   beats all previous — never "try everything".
3. ML retraining: keep the 20-trade retrain only for the code_version-gated
   feedback blend; add a drift test (two-sample KS on label distribution and
   feature means, last 60 days vs prior 180) and retrain on drift, not on a
   trade count. Prior 55% the trade-count trigger fires more often with no
   measurable accuracy change.

DISCARDED: "learning" from post-deploy win rates on < 30 trades; any
self-applied threshold move without counterfactual data (RULE REVIEW); a
second adaptive layer on top of Kelly (two learners on one noisy sample).

## WHAT NOT TO BUILD (until the tests above read out)

- A fourth exit model or any re-tune of stop/TP constants: the three
  existing ones have never been compared on the same inputs, and a new grid
  "best" repeats the 324-combo error (E[max] ~2.9 SD [S17]).
- Weekly iron condors / weekly covered calls on QQQ or single names: lower
  return for more premium and 4x the spread crossings [S7,S8]; single-name
  VRP is statistically zero [S11,S12].
- A high-IV single-name put scanner or IV-rank sizing bonus: high single-name
  IV is an event forecast; the earnings gate is the instrument.
- A larger Kelly fraction, smaller divisor, or forced positive stocks prior
  (the live interval excludes WR >= 44%); only the bounded budget is justified.
- Floor-weight changes from the 30-day gap or a new sweep; the IR/TE/
  bootstrap/DSR evidence request comes first and the decision is the human's.
- Any adaptive logic acting on < 30 trades, including reviving the dead
  strategy-weight adjuster.
- Local persistence of the F7 cooldown as the #46 fix: the broker's order
  timestamps are the only restart-proof record.
- Measurement-engine changes bundled with strategy changes (MEASUREMENT
  INTEGRITY), including flipping DEFAULT_EXIT_MODEL before the before/after
  is logged.

## SOURCES (opened this session; grade in brackets)

- [S1] Kaminski & Lo, "When do stop-loss rules stop losses?", J. Financial Markets 18 (2014) — abstract, https://ideas.repec.org/a/eee/finmar/v18y2014icp234-254.html [A, abstract].
- [S2] CXO Advisory on the 2008 draft of [S1], https://www.cxoadvisory.com/?p=1809 [C].
- [S3] CXO Advisory on Han, Zhou & Zhu, "Taming Momentum Crashes" (2016 WP), https://www.cxoadvisory.com/?p=25071 [C; SSRN 2407199 not opened].
- [S4] Clare, Seaton, Smith & Thomas, J. Asset Management 14(3) 2013, https://openaccess.city.ac.uk/17842/8/BLACKBOX%20%20%20SSRN-id2126476.pdf [A].
- [S5] SetupAlpha, stop-loss mean-reversion backtest, https://setup4alpha.substack.com/p/stop-loss-mean-reversion-backtest [D, paywalled].
- [S6] FlashAlpha, SPY put-spread sweep, https://flashalpha.com/articles/spy-put-credit-spread-active-backtest-mm-fills-vrp-signal-drawdown-breaker [D, promotional].
- [S7] Cboe IR release on Bondarenko (2016), https://ir.cboe.com/news/news-details/2016/New-Study-On-Weekly-Monthly-SP-500-PutWrite-Indexes-Released-01-27-2016/default.aspx [B; sponsor COI].
- [S8] CXO Advisory on Bondarenko (2016), https://www.cxoadvisory.com/equity-options/performance-of-cboe-putwrite-indexes [C].
- [S9] Israelov & Tummala, "Which Index Options Should You Sell?" (AQR 2017), https://www.aqr.com/Insights/Research/Working-Paper/Which-Index-Options-Should-You-Sell + https://www.cxoadvisory.com/equity-options/best-index-options-to-sell [B; AQR COI].
- [S10] Israelov & Nielsen, "Covered Call Strategies: One Fact and Eight Myths", FAJ 70(6) 2014, https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/FAJ-Covered-Call-Strategies-One-Fact-and-Eight-Myths.pdf [A; AQR COI].
- [S11] Driessen, Maenhout & Vilkov, "Option-Implied Correlations and the Price of Correlation Risk" (Netspar WP), https://www.netspar.nl/wp-content/uploads/061_Driessen.pdf [B].
- [S12] Carr & Wu, "Variance Risk Premia" (WP 2004; RFS 2009), https://econwpa.ub.uni-muenchen.de/econ-wp/fin/papers/0409/0409015.pdf [B].
- [S13] MacLean, Thorp & Ziemba, "Good and bad properties of the Kelly criterion", https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf [A].
- [S14] Thorp, "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" (2006), https://www.gwern.net/doc/statistics/decision/2006-thorp.pdf [A].
- [S15] Baker & McHale, Decision Analysis 10(3) 2013 — abstract, https://ideas.repec.org/a/inm/ordeca/v10y2013i3p189-199.html [A, abstract].
- [S16] Russo et al., "A Tutorial on Thompson Sampling", https://arxiv.org/abs/1707.02038 and https://ar5iv.labs.arxiv.org/html/1707.02038 (first ~45%) [A].
- [S17] Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", JPM 2014, https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf [A].
- [S18] Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest Overfitting", https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf [A].
- [S19] Portfolio Optimizer blog, PSR/MinTRL formulas, https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-bias-adjustment-confidence-intervals-hypothesis-testing-and-minimum-track-record-length/ [C; Bailey & LdP 2012 not opened].
- [S20] Harvey & Liu, "Backtesting", JPM 41(1) 2015, https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF [A].
- [S21] Harvey, Liu & Zhu, RFS 29(1) 2016 — abstract, https://ideas.repec.org/a/oup/rfinst/v29y2016i1p5-68..html [A, abstract].
- [S22] Lo, "The Statistics of Sharpe Ratios", FAJ 2002, https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf [A].
- [S23] Ledoit & Wolf, JEF 15(5) 2008, https://uzh.ch/cmsssl/econ/dam/jcr:ffffffff-935a-b0d6-0000-00007214c2bc/jef_2008pdf.pdf [A].
- [S24] Goodwin, "The Information Ratio", FAJ 1998 — abstract, https://rpc.cfainstitute.org/research/financial-analysts-journal/1998/the-information-ratio [A, abstract].
- [S25] McLean & Pontiff, JF 71(1) 2016, https://www.gwern.net/doc/economics/2016-mclean.pdf [A].
- [S26] Nosek, Ebersole, DeHaven & Mellor, "The preregistration revolution", PNAS 2018, https://errorstatistics.com/wp-content/uploads/2017/07/nosek-et-al-preregistration-revolution.pdf [A].
- [S27] Alpaca, Create an order, https://docs.alpaca.markets/reference/postorder [A].
- [S28] Alpaca, Orders at Alpaca, https://docs.alpaca.markets/docs/orders-at-alpaca [A].
- [S29] Alpaca, Working with /orders, https://docs.alpaca.markets/docs/working-with-orders [A].
- [S30] Alpaca, Get order by client order id, https://docs.alpaca.markets/reference/getorderbyclientorderid [A].
- [S31] Alpaca, WebSocket streaming, https://docs.alpaca.markets/docs/websocket-streaming [A].
- [S32] Alpaca blog, cancel-all / liquidate, https://alpaca.markets/blog/position-liquidation-cancel-orders/ [B].
- [S33] FIX Trading Community forum, "ClOrdID uniqueness", https://forum.fixtrading.org/t/clordid-uniqueness/2519 [C].
- [S34] OnixS FIX dictionary, Appendix D, https://www.onixs.biz/fix-dictionary/latest/app_d.html [B].
- [S35] Zero Hash, CLOB FIX session management, https://docs.zerohash.com/reference/session-management-clob [B].
- [S36] Stripe, "Designing robust and predictable APIs with idempotency", https://stripe.com/blog/idempotency [B].

NOT OPENED (never cited for a number): tastylive study pages (JS landing
only); Spintwig / Early Retirement Now posts (404); SSRN pages (403; mirrors
used); Kaminski-Lo full text (rate-limited); Alpha Architect, PNAS and AMS
pages (403); Lopez de Prado's AFML; Carver's "Systematic Trading"; Connors &
Alvarez; the Alvarez podcast page (empty); Alpaca's single-order-cancel and
cancel-all reference pages (404; [S32] and [S28] cover the semantics).
Scratchpad arithmetic: /tmp/claude-0/-home-user-voltradeai/1c64125e-f4dc-5e5f-9984-1cb6a5ea06e4/scratchpad/calc.py (formulas restated inline; not in the repo).

## FILES / LINES READ THIS SESSION (all under /home/user/voltradeai)

- CLAUDE.md: REASONING STANDARD, EDGE DOCTRINE, ROOT VALIDATION LADDER, RULE
  REVIEW, MEASUREMENT INTEGRITY, HEALTH OF THE LOOP, KNOWN STATE.
- research/open_questions.md 7368-7492 (KB #46/#47/#48), 7642-7665 (LEARNING
  DEADLOCK), 22907-22922 (floor tracking error), 22924-22931 (DTE CONFLICTS).
- research/experiments.md 108609-108619 (v1.0.1088 exit model), 108991-109096
  (all 2026-10-20 entries).
- system_config.py 105-165, 188-200, 225-330, 452-500; options_manager.py
  45-80, 630-830, 1068-1165; options_execution.py 85-135; position_sizing.py
  150-310, 925-960; bot_engine.py 2040-2205; backtest_v2.py 125-161, 433-470,
  498-565; live_exit_policy.py 1-80; ml_model_v2.py 619-660;
  shadow_portfolio.py 80-86 (+ grep 1392, 1871); tiered_strategy.py 182-183.
- server/bot.ts 815-840, 999-1040, 6718-6780, 6876-6960, 7100-7200; grep of
  every "/v2/orders" POST and of client_order_id (none); server/diag.ts:336.
