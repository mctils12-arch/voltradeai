# Open Questions

Current hypotheses, ranked by expected value (CLAUDE.md — MEMORY PROTOCOL).

1. **Does dual-momentum SPY/QQQ hold out-of-sample?** In-sample (2016–2026) it
   beat SPY (16.3% vs 14.1% CAGR, Sharpe 0.90 vs 0.83, DD -28.6% vs -33.7%).
   Test walk-forward across sub-periods and other universes/date ranges before
   trusting it. Overfit risk: leans on the tech-led decade.

2. **Rebuild the real backtest engine** (standing top-priority task; `backtest.py`
   is a stub). Reproduce the `backtest_10yr_results.json` schema, invoked as
   `python3 backtest.py <ticker> <strategy> <years>`, using `_fetch_alpaca_bars`
   from `analyze.py` so backtest and live see identical data. On completion,
   the 2 skipped regime-consistency tests (`backtest_v2.py`,
   `backtest_v1028_full.py`) re-activate automatically (skipUnless-gated).

3. **Quantify and cut the options-leg drag.** The static artifact pins options at
   -0.75%/trade. Measure directly, then test curtailing/removing the options leg
   vs the equity-momentum core.
