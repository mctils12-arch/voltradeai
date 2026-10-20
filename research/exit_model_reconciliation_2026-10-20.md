# Exit-model reconciliation — research test 1 (2026-10-20)

[RULE-REVIEW] measurement run of scripts/exit_model_reconciliation.py. Nothing live changed. research/bot_logic_research_2026-10-20.md Q1; KNOWN BROKEN #48.

- Universe (declared, N=30): SPY, QQQ, IWM, DIA, XLF, XLE, XLK, XLV, XLI, XLY, AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, XOM, CVX, JNJ, PFE, UNH, HD, WMT, COST, CAT, BA, DIS, INTC
- Years: 10.0 · arms declared: legacy, live, time_only, wide_stop (N=4) · random seeds per cell: 10
- Data: Yahoo (no Alpaca keys) · tickers with degraded VXX coverage: 30
- Runtime: 26.0 s on 4 workers · generated 2026-10-20T20:42:06+00:00

PRIOR (stated before the run): 60% that the live R-ladder has LOWER net expectancy than the wide_stop control on momentum entries; 75% that any stop tighter than 2xATR hurts mean-reversion entries (time_only / wide_stop beat legacy and live there); 50% on squeeze (not simulated: no short-interest history). Exploratory arms die 2026-12-31 if nothing clears the DSR threshold.

## Equity-curve view (mean over tickers; single position per ticker)

| strategy | arm | tickers | Sharpe (mean) | Lo SE (mean) | SD across tickers | CAGR % | max DD % |
|---|---|---|---|---|---|---|---|
| mean_reversion | legacy | 30 | 0.208 | 0.301 | 0.297 | 0.06 | 1.01 |
| mean_reversion | live | 30 | 0.146 | 0.301 | 0.32 | 0.03 | 0.83 |
| mean_reversion | time_only | 30 | 0.235 | 0.301 | 0.293 | 0.08 | 1.05 |
| mean_reversion | wide_stop | 30 | 0.235 | 0.301 | 0.29 | 0.07 | 1.02 |
| momentum | legacy | 30 | 0.172 | 0.301 | 0.267 | 0.39 | 4.59 |
| momentum | live | 30 | 0.051 | 0.301 | 0.301 | 0.16 | 4.74 |
| momentum | time_only | 30 | 0.16 | 0.301 | 0.307 | 0.4 | 5.28 |
| momentum | wide_stop | 30 | 0.173 | 0.301 | 0.304 | 0.42 | 5.03 |

- DSR bar, momentum: with N=4 arms and Sharpe dispersion V=0.00261, a zero-skill best-of-N would show E[max SR] ≈ 0.054; an arm is believed only if its mean Sharpe clears that AND its Lo SE.
- DSR bar, mean_reversion: with N=4 arms and Sharpe dispersion V=0.00132, a zero-skill best-of-N would show E[max SR] ≈ 0.038; an arm is believed only if its mean Sharpe clears that AND its Lo SE.
- Best arm by ticker (count of tickers each arm wins on Sharpe): {"momentum": {"legacy": 12, "wide_stop": 6, "time_only": 9, "live": 3}, "mean_reversion": {"time_only": 12, "legacy": 11, "live": 5, "wide_stop": 2}}

## Per-trade view by regime at entry (pooled across tickers)

`alpha` = strategy expectancy − random-entry expectancy for the SAME arm, regime gate, sizing and costs.

### momentum

| arm | regime | n | mean net % | ± SE | median % | win % | hold d | exit mix | random n | random mean % | alpha %/trade |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | ALL | 2876 | 0.364 | 0.087 | 0.23 | 51.6 | 7.4 | end_of_backtest:0% stop_loss:36% take_profit:15% time_stop:49% | 20240 | 0.429 | -0.065 |
| legacy | BULL | 2538 | 0.247 | 0.089 | 0.135 | 50.8 | 7.5 | end_of_backtest:0% stop_loss:36% take_profit:14% time_stop:50% | 16567 | 0.387 | -0.14 |
| legacy | CAUTION | 338 | 1.244 | 0.324 | 1.385 | 57.4 | 6.9 | stop_loss:34% take_profit:23% time_stop:43% | 3673 | 0.619 | 0.626 |
| live | ALL | 3773 | 0.117 | 0.059 | 0.17 | 52.1 | 5.0 | end_of_backtest:0% stop_loss:22% take_profit:0% time_stop:61% trailing_stop:17% | 27087 | 0.181 | -0.063 |
| live | BULL | 3357 | 0.07 | 0.059 | 0.15 | 51.8 | 5.0 | end_of_backtest:0% stop_loss:21% take_profit:0% time_stop:63% trailing_stop:16% | 22196 | 0.192 | -0.123 |
| live | CAUTION | 416 | 0.504 | 0.236 | 0.505 | 54.1 | 5.1 | stop_loss:26% take_profit:3% time_stop:45% trailing_stop:26% | 4891 | 0.129 | 0.375 |
| time_only | ALL | 2401 | 0.422 | 0.108 | 0.48 | 55.7 | 10.0 | end_of_backtest:1% time_stop:99% | 16692 | 0.586 | -0.164 |
| time_only | BULL | 2171 | 0.344 | 0.111 | 0.41 | 54.9 | 10.0 | end_of_backtest:1% time_stop:99% | 13647 | 0.423 | -0.079 |
| time_only | CAUTION | 230 | 1.161 | 0.405 | 1.54 | 63.5 | 10.0 | time_stop:100% | 3045 | 1.316 | -0.154 |
| wide_stop | ALL | 2470 | 0.42 | 0.104 | 0.49 | 55.8 | 9.3 | end_of_backtest:0% stop_loss:16% time_stop:84% | 17221 | 0.58 | -0.16 |
| wide_stop | BULL | 2230 | 0.361 | 0.107 | 0.43 | 55.1 | 9.4 | end_of_backtest:0% stop_loss:16% time_stop:84% | 14074 | 0.491 | -0.13 |
| wide_stop | CAUTION | 240 | 0.963 | 0.391 | 1.47 | 62.5 | 9.3 | stop_loss:15% time_stop:85% | 3147 | 0.976 | -0.012 |

### mean_reversion

| arm | regime | n | mean net % | ± SE | median % | win % | hold d | exit mix | random n | random mean % | alpha %/trade |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | ALL | 110 | 1.593 | 0.617 | 2.27 | 59.1 | 7.5 | stop_loss:30% take_profit:20% time_stop:50% | 1302 | 0.317 | 1.275 |
| legacy | BULL | 52 | 1.142 | 0.985 | 2.28 | 57.7 | 7.5 | stop_loss:35% take_profit:17% time_stop:48% | 1066 | 0.508 | 0.633 |
| legacy | CAUTION | 58 | 1.997 | 0.763 | 2.17 | 60.3 | 7.5 | stop_loss:26% take_profit:22% time_stop:52% | 236 | -0.545 | 2.542 |
| live | ALL | 111 | 0.855 | 0.464 | 0.66 | 54.1 | 5.7 | stop_loss:19% take_profit:3% time_stop:49% trailing_stop:30% | 1323 | 0.116 | 0.738 |
| live | BULL | 53 | -0.013 | 0.679 | -0.65 | 45.3 | 5.7 | stop_loss:24% time_stop:47% trailing_stop:28% | 1106 | 0.218 | -0.231 |
| live | CAUTION | 58 | 1.647 | 0.616 | 1.1 | 62.1 | 5.8 | stop_loss:14% take_profit:5% time_stop:50% trailing_stop:31% | 217 | -0.403 | 2.05 |
| time_only | ALL | 102 | 2.273 | 0.608 | 2.39 | 63.7 | 10.0 | time_stop:100% | 1200 | 0.529 | 1.744 |
| time_only | BULL | 46 | 1.311 | 0.906 | 2.28 | 63.0 | 10.0 | time_stop:100% | 956 | 0.663 | 0.648 |
| time_only | CAUTION | 56 | 3.063 | 0.806 | 3.285 | 64.3 | 10.0 | time_stop:100% | 244 | 0.006 | 3.057 |
| wide_stop | ALL | 104 | 2.118 | 0.627 | 2.41 | 63.5 | 9.6 | stop_loss:9% time_stop:91% | 1230 | 0.548 | 1.57 |
| wide_stop | BULL | 48 | 1.11 | 0.948 | 2.28 | 62.5 | 9.2 | stop_loss:15% time_stop:85% | 990 | 0.667 | 0.443 |
| wide_stop | CAUTION | 56 | 2.983 | 0.816 | 3.285 | 64.3 | 10.0 | stop_loss:4% time_stop:96% | 240 | 0.058 | 2.925 |

## Caveats (bias direction stated)

- SURVIVORSHIP: every universe name is a large, surviving ticker; the live scanner's small caps are absent → OPTIMISTIC for every arm alike, and silent on where the bot actually trades.
- The control arms are close-checked (legacy convention); the live arm walks bars intrabar → the controls are slightly OPTIMISTIC on stops relative to live.
- One position per ticker, regime-gated entries, backtest_v2's cost model; squeeze is not simulated (no short-interest history).
- The random baseline's rate is calibrated per (ticker, strategy, arm) to the strategy's own trade count over the regime-eligible bars.
- Multiple testing: 4 arms × 2 strategies × regimes; read every cell against the DSR bar and the Lo SE, not against zero. Per-trade `± SE` is the pooled standard error (trades treated as independent; overlapping regimes and shared market days make the true SE larger).
