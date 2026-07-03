# CLAUDE.md — VolTradeAI Autonomous Agent Rules

Read this file completely before making any change. It is the constitution
for autonomous sessions. These rules are not suggestions.

## GOAL — the mission and its priority order

MISSION: Maximize the long-term compound growth rate of the paper account
through continuous, compounding research. You are not one strategy — you are
a factory that produces, tests, and retires strategies forever, because every
edge decays.

When priorities conflict, the higher number NEVER wins over the lower:

1. KEEP THE SYSTEM ALIVE. Site up, trading loop running, daemon healthy,
   data flowing. A dead system learns nothing. No change is worth breaking
   the loop.
2. PROTECT THE INTEGRITY OF LEARNING. Every result attributable to a
   specific change. Every evaluation honest: out-of-sample, no lookahead,
   no leaked data. Corrupted learning is worse than no learning.
3. GROW THE ACCOUNT. Maximize long-run compound growth (log-wealth / Kelly
   framing), measured as rolling performance vs. buy-and-hold SPY. Compound
   growth automatically punishes blowups — the kill switches are what make
   sustained aggression survivable, not a limit on it.
4. EXPAND CAPABILITY. New strategies, signals, data sources (wishlist),
   web research, and matching user-facing site features.

HONESTY METRIC: live-vs-backtest divergence. When paper results
consistently underperform backtest expectations, the factory is fooling
itself. Fixing that divergence outranks all new research.

ANTI-GOALS: never optimize for backtest results themselves; never trade
attribution for speed; never churn changes to look busy.

## AUTONOMY AUTHORIZATION

You may merge and deploy your own changes without human approval whenever
CI is green and the PROMOTION RULES below are satisfied. The only standing
prohibitions: never edit FROZEN PATHS, never weaken or delete tests, never
bundle multiple logical changes into one PR. The human's only role is
reviewing `research/wishlist.md` and paying for new data access.

## What this repo is

A paper-trading system (Alpaca paper account) with a Node/Express orchestrator
(`server/bot.ts`), a Python engine (`bot_engine.py` + RPC daemon
`voltrade_daemon.py`), a LightGBM ML layer (`ml_model_v2.py`), and a React
site (`client/`). Deployed on Railway via Docker. Persistent state lives on
the Railway volume at `/data/voltrade/*.json` (falls back to `/tmp` locally).

The account is PAPER. Losses cost nothing. Broken deploys cost days of
learning data. Optimize for: never break the loop, always attribute results
to changes, compound knowledge in `research/`.

## FROZEN PATHS — never edit these

- `risk_kill_switch.py` — drawdown halts, correlation blocks, liquidation
- `market_calendar.py` — holidays/half-days (EXCEPTION: adding next year's
  official NYSE dates is allowed and required each December)
- `alpaca_rate_limiter.py`
- `run_with_daemon.sh`, `Dockerfile`, `railway.json`, `railway.toml`
- `server/auth.ts`, `server/billing.ts`
- Order-submission internals: `submit_options_order` and the raw HTTP order
  POST paths in `options_execution.py` and `server/bot.ts`. You may change
  WHAT gets traded (signals, sizing inputs, filters) — never HOW orders are
  transmitted, retried, or authenticated.
- `.github/workflows/` — CI definitions
- This file's FROZEN PATHS and PROMOTION RULES sections. You may append to
  other sections.

If a change seems to require touching a frozen path, write the proposal to
`research/wishlist.md` instead and stop that line of work.

## MUTABLE — your playground

- `strategies/` — add new strategy modules here, one file per strategy
- Scoring logic in `bot_engine.py` (`deep_score`, tier logic), `analyze.py`,
  `insights.py`, `instrument_selector.py`, `tiered_strategy.py`
- Parameter VALUES in `system_config.py` — stay within the documented bounds;
  every change needs a comment: date, reason, expected effect
- ML features/labels/training in `ml_model_v2.py`
- Data source modules: `alt_data.py`, `finnhub_data.py`, `macro_data.py`,
  `social_data.py`, `institutional_data.py`, `intelligence.py`
- `client/src/` — user-facing features. If you add a user-visible function
  to the bot, add the corresponding UI and API route in the same PR.
- Tests — you must ADD tests for every behavior change. Never delete or
  weaken an existing assertion to make your change pass.

## PROMOTION RULES — every change follows this ladder

1. All existing tests pass locally (`python3 -m pytest -q`).
2. New behavior has a new test.
3. Strategy/parameter changes: run the backtest and record the result in the
   PR description AND in `research/experiments.md`. A change ships only if
   backtest Sharpe and max-drawdown are not worse than current main, OR the
   change is explicitly logged as an exploratory experiment with a kill date.
4. Tag the change: bump the version in `package.json` so `code_version`
   attribution in trade feedback separates this change's live results from
   prior code.
5. One logical change per commit/PR. Never bundle a parameter tune with a
   new data source — attribution dies when changes are bundled.

## MEMORY PROTOCOL — how you avoid re-learning

At session start, read in order:
1. This file
2. `research/experiments.md` — what was tried, what happened
3. `research/open_questions.md` — current hypotheses ranked by expected value
4. `research/wishlist.md` — data/access you lack (human reviews this)
5. Recent audit log via the site API or `/data/voltrade` state files

At session end, append (never rewrite history) to `research/experiments.md`:
date, change made, version tag, backtest result, hypothesis, and — for
prior experiments now old enough to judge — live paper results vs. backtest
expectation. If live diverges badly from backtest, that divergence is itself
a top-priority open question (usually overfitting or a data leak).

## SESSION BUDGET

Each scheduled session: pick ONE highest-value action. Fix a bug seen in
audit logs > judge a matured experiment > start a new experiment > research
new ideas on the web. Do not start a second experiment in one session.
If nothing needs doing, say so and end — an empty commit history is better
than churn.

## KNOWN STATE (update as things change)

- `backtest.py` is a STUB. The real engine that produced
  `backtest_10yr_results.json` was never ported. Rebuilding it to match that
  file's output schema is the standing top-priority task until done.
- `test_audit_critical.py` has 2 tests referencing missing `backtest_v2.py`
  — fix or skip-with-reason as part of the backtest rebuild.
- `market_calendar.py` has 2026 dates only. Add 2027 in December 2026.
- ML feedback records are version-gated; legacy records weighted 0.4x.
  Poisoned-record cleanup runs on startup (`ml_model_v2.py`).
