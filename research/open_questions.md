# Open Questions

Current hypotheses, ranked by expected value (CLAUDE.md — MEMORY PROTOCOL).

## KNOWN BROKEN — repair before research (CLAUDE.md: REASONING STANDARD)

A broken pipeline generates poisoned learning data. These outrank every
open question below:

1. **`backtest.py` is a stub.** The real engine was never ported; promotion
   rule #3 (backtest before shipping strategy changes) is currently
   unsatisfiable. This blocks ALL strategy/parameter changes from shipping
   legitimately. Standing top-priority task.
2. **Counterfactual logging does not exist.** RULE REVIEW mandates it as the
   evidence basis for any rule-threshold change; until built, no rule change
   can ship on evidence. Build: log every blocked candidate trade
   {date, ticker, rule, entry price, score}; scheduled job scores outcomes
   at +1d/+5d/+20d.
3. **Options-leg drag (suspected primary performance break).** Live
   options-heavy bot: 0.27% CAGR vs SPY 13.97% (static artifact); options
   avg -0.75%/trade over 328+ trades. The same momentum signal run cleanly
   as an ETF rotation does ~10% CAGR. Needs the rebuilt backtest engine +
   options data (wishlist) to diagnose/fix honestly.
4. **2 regime-consistency tests are skipUnless-gated** on the un-ported
   `backtest_v2.py` / `backtest_v1028_full.py`; they re-activate
   automatically when the backtest rebuild lands (no action needed beyond
   the rebuild itself).

## Open questions (ranked by expected value)

1. **Does dual-momentum SPY/QQQ hold out-of-sample?** In-sample (2016–2026) it
   beat SPY (16.3% vs 14.1% CAGR, Sharpe 0.90 vs 0.83, DD -28.6% vs -33.7%).
   Test walk-forward across sub-periods and other universes/date ranges before
   trusting it. Overfit risk: leans on the tech-led decade. Per REASONING
   STANDARD #4: this was one of ~7 variants tried — discount accordingly.
   PRIOR (stated before the out-of-sample run, per REASONING STANDARD #10):
   edge shrinks but survives at roughly +1% CAGR over SPY ex-2020-2021;
   if it flips negative in ≥2 sub-periods, kill it.
2. **Quantify and cut the options-leg drag** (see KNOWN BROKEN #3) — after
   the backtest engine exists, run the ablation: options leg on vs. off.
3. **Regime filter paradox**: the 200dma filter HURT in-sample (cash drag in a
   bull decade) but is exactly what protects a bear tape (REASONING STANDARD
   #2). Evaluate regime-conditioned, not pooled: what does it cost per regime?

## Ops gotchas (avoid re-learning)

- **A `mergeable_state: "dirty"` PR stalls silently**: GitHub cannot build the
  merge ref, so `pull_request` workflows never start — zero checks, no
  automerge, no error. If a claude/* PR sits unmerged with no check runs,
  check mergeability FIRST, not CI logs. Cause here: reusing one branch across
  squash-merged PRs. Scheduled sessions use a fresh branch each time, which
  avoids it; interactive sessions must reset the branch onto main after each
  merge.
