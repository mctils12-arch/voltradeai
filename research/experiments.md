# Experiment Log

Append-only. Newest at top. Never rewrite history (CLAUDE.md — MEMORY PROTOCOL).
Each entry: date · change · version tag · backtest result · hypothesis · (later) live-vs-backtest.

## 2026-07-03 — Equity-momentum backtest harness (`bot_backtest.py`)
- Change: added a reproducible backtest of the bot's OWN momentum scoring
  (`strategies/momentum.py`) run as a monthly ETF rotation, vs SPY buy&hold.
  Yahoo daily adj-close, stdlib only, cached under `.bt_cache/`.
- Window: 2016-01 → 2026-04 (matches `backtest_10yr_results.json`).
- Result (CAGR / Sharpe / maxDD):
  - SPY buy & hold ............ 14.13% / 0.83 / -33.7%
  - QQQ buy & hold ............ 18.67% / 0.88 / -35.1%
  - Bot momentum top-3 rot. ... 10.09% / 0.63 / -31.4%
  - **Dual-momentum SPY/QQQ ... 16.31% / 0.90 / -28.6%**  (beats SPY; higher Sharpe, lower DD)
  - 200dma regime filter ...... HURT (cash drag in a bull decade)
- Reference: live options-heavy bot (static artifact) = **0.27% CAGR**.
- Hypothesis: the live bot's drag is **options overtrading** (options avg
  -0.75%/trade over 328+ trades), NOT the core signal. A simple rules-based
  equity/ETF momentum core is a large improvement. Dual-momentum SPY/QQQ is a
  candidate but is IN-SAMPLE over a tech-led decade — needs out-of-sample /
  walk-forward validation before any live change (HONESTY METRIC risk).
- Caveat: equity logic only; the options leg can't be backtested without
  historical option prices (see wishlist).
