# Open Questions

## KNOWN BROKEN — fix these first (repair mandate)

1. **[RESOLVED 2026-07-03 — backtest_v2.py]** ~~Backtest engine missing.~~
   Rebuilt: see experiments.md entry. Original text: `backtest.py` is a stub; the real engine
   that produced `backtest_10yr_results.json` was never ported from the
   workbench. Nothing can be evaluated until this exists. Reproduce the
   output schema in that JSON; invoke signature is
   `python3 backtest.py <ticker> <strategy> <years>` (bot.ts JSON-parses
   stdout). STANDING TOP PRIORITY.

2. **[RESOLVED 2026-07-03]** ~~2 failing tests reference missing backtest_v2.py.~~
   The backtest_v2 gated test now runs and passes; the backtest_v1028_full
   test remains skip-with-reason (legacy file superseded by backtest_v2,
   never ported).

3. **CSP execution cascade** (per CHANGELOG 2026-05-22): CSP trades were
   failing on three modes — insufficient capital for high-priced
   underlyings, no suitable puts in chain, liquidity filters rejecting
   everything. Fixes were applied (dynamic price ceiling etc.) — VERIFY
   in current audit logs that Tier 2 CSP trades actually fire now. If
   still zero fills, the fix pack didn't take.

4. **Human-reported: bot "doesn't work right" overall.** Symptoms to
   collect from audit logs: are trades firing at expected frequency? Are
   fills tracked into trade_feedback? Is the ML retrain loop (Tier 3)
   completing or erroring? Diagnose from `/data/voltrade` state files and
   the persisted audit log before assuming any subsystem works.

5. **Data modules not wired to live scoring.** The repo contains
   `alphadesk/` (EDGAR filings, catalyst detection, LLM filing reader),
   `macro_data.py`, `alt_data.py`, `social_data.py`,
   `institutional_data.py` — audit which of these actually feed
   `deep_score`/tier decisions vs. which are orphaned. Wire or retire.

6. **Full-repo pytest is broken at collection (pre-existing).**
   `test_auto_discovery.py` calls sys.exit() at module level, killing
   collection for everything; excluding it, 7 failures + 1 error remain in
   network/keys-dependent files (test_options_fixes, test_options_v134_fixes,
   test_fixes_pr8/11, test_full_system) — verified identical with and without
   the backtest change. CI's 4-file offline subset is the real gate and is
   green. Fix candidates: convert test_auto_discovery to proper pytest tests;
   mark network suites with a skip-if-no-keys guard.

## RULE COST AUDIT — after counterfactual logging exists

- Is MIN_SCORE=63 leaving winners on the table or blocking losers?
- SCORE_BAND_MAX=75 ("fake breakout" ceiling) — measure prevention-P&L.
- MAX_CHANGE_PCT=35 ("easy money gone") — verify against outcomes.
- Spread filter 0.5% — how many blocked names would have filled fine?
- Correlation/sector blocks — cost vs. protection in current regime.
- Kill-switch drawdown thresholds — sized for real-money caution; is
  that optimal for a paper account whose goal is learning speed?

## OPEN RESEARCH QUESTIONS

- **Options fill realism.** The synthetic slippage haircut in bot.ts is
  volume-tiered with a random component — good for stocks, weak for
  options. Replace for options with quote-based fills: short premium
  fills at the BID, long premium at the ASK, using the contract's actual
  quote at fill time (the liquidity filters already fetch it). Also cap
  simulated fill quantity at a sane multiple of the contract's real
  volume/open interest — Alpaca paper fills unlimited size, which is
  fiction for thin chains. Validate the existing stock slippage tiers
  against recorded bid/ask spreads in the fills tracker.
- **Strategy tournament.** Run strategies as isolated, tagged competitors
  (strategies/ modules are already shaped for this) with buy-and-hold SPY
  as a permanent benchmark entrant. Allocate more to winners, retire
  losers, log every promotion/retirement decision with evidence. Answers
  "is any of this beating doing nothing" continuously. Requires backtest
  engine (#1) first.
- Live-vs-backtest divergence: unmeasurable until #1 done. Then it is
  the standing honesty metric.
- Which regime detector (markov_regime vs. VXX-ratio heuristics) actually
  predicts forward volatility better? They currently coexist.
- Earnings/FOMC calendar awareness: verify positions are actually
  gated around scheduled events, not just theoretically supported.

- **Dual-momentum SPY/QQQ** (from 2026-07-03 harness run, `bot_backtest.py`):
  in-sample 2016-2026 beat SPY (16.3% vs 14.1% CAGR, Sharpe 0.90 vs 0.83,
  DD -28.6% vs -33.7%). 1-of-~7 variants tried — discount per REASONING
  STANDARD #4. PRIOR stated before any out-of-sample run (#10): edge shrinks
  but survives ~+1% CAGR over SPY ex-2020-21; kill if negative in >=2
  sub-periods. Candidate tournament entrant once #1 lands.

## OPS GOTCHAS (avoid re-learning)

- A `mergeable_state: "dirty"` claude/* PR stalls SILENTLY: no merge ref ->
  pull_request workflows never start -> no checks, no automerge, no error.
  Check mergeability FIRST, not CI logs. Cause: reusing one branch across
  squash-merged PRs; scheduled sessions (fresh branch each run) are immune,
  interactive sessions must reset the branch onto main after each merge.
