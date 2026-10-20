# Research test 4: the ETF floor's composition vs SPY under the live exposure schedule — 2026-10-20

Generated 2026-10-20T22:24:17Z by scripts/floor_basket_vs_index.py — 2014-11-28 → 2026-10-20 (2990 trading days), regime days {'BULL': 1029, 'NEUTRAL': 1115, 'CAUTION': 178, 'BEAR': 639, 'PANIC': 29}, carried-forward closes {'SPY': 0, 'QQQ': 0, 'SMH': 0, 'KWEB': 0, 'VXUS': 0, 'GLD': 0}, cost 5.0 bp/side, data backtest_v2.fetch_bars (regime quality: degraded).

## PRIOR (stated before the run)

55% that SPY-core has the better benchmark-relative Sharpe than the configured basket; 35% that it has the higher CAGR (the 2026-10-14 entry, unchanged).

## VERDICT

benchmark-relative Sharpe: spy_core -0.175 vs basket -0.109 — basket better (basket's LW 95% interval vs the same-exposure SPY: [-0.375, +0.156] — includes 0); CAGR: spy_core 5.46% vs basket 7.82% — basket higher; DSR bar (N=4) 0.08: best arm qqq_only clears it.

## Arms (whole-equity daily returns; the floor target, GLD leg, trend caps and cash are identical across arms)

| arm | CAGR % | max DD % | vol % | Sharpe ± SE | beta SPY | alpha %/y | TE % | IR | Sharpe − same-exposure SPY [LW 95%] | rebalances | turnover | costs % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| basket | 7.82 | -21.56 | 11.12 | 0.733 ± 0.291 | 0.343 | 3.27 | 5.12 | 0.117 | -0.109 [-0.375, 0.156] | 479 | 166.6 | 14.509 |
| spy_core | 5.46 | -19.36 | 8.51 | 0.667 ± 0.290 | 0.277 | 1.73 | 2.60 | -0.719 | -0.175 [-0.342, -0.008] | 479 | 166.7 | 12.170 |
| fifty_fifty | 6.26 | -20.08 | 9.33 | 0.698 ± 0.291 | 0.300 | 2.23 | 3.17 | -0.328 | -0.145 [-0.341, 0.051] | 479 | 166.7 | 13.030 |
| qqq_only | 9.76 | -23.57 | 11.97 | 0.838 ± 0.291 | 0.380 | 4.61 | 5.34 | 0.465 | -0.005 [-0.247, 0.238] | 480 | 182.3 | 18.312 |
| spy_same_exposure | 7.41 | -20.06 | 8.96 | 0.843 ± 0.291 | 0.309 | 3.15 | 0.00 | — | 0.000 [0.000, 0.000] | 478 | 182.3 | 14.940 |
| spy_buy_hold | 13.58 | -33.72 | 17.48 | 0.816 ± 0.291 | 1.000 | 0.00 | 14.04 | 0.478 | -0.027 [-0.535, 0.481] | 4 | 1.0 | 0.050 |

DSR: N = 4 arms, Sharpe variance across arms 0.00553, E[max SR] bar 0.078, best arm qqq_only clears the bar.

## By regime (the regime that set the allocation earning each day's return)

| arm | BULL (days / ann. mean % / Sharpe) | NEUTRAL (days / ann. mean % / Sharpe) | CAUTION (days / ann. mean % / Sharpe) | BEAR (days / ann. mean % / Sharpe) | PANIC (days / ann. mean % / Sharpe) |
|---|---|---|---|---|---|
| basket | 1029 / 7.78 / 0.73 | 1114 / 11.07 / 0.79 | 178 / 14.83 / 1.73 | 639 / 1.64 / 0.33 | 29 / 12.02 / 0.85 |
| spy_core | 1029 / 4.50 / 0.58 | 1114 / 8.04 / 0.75 | 178 / 11.28 / 1.68 | 639 / 1.64 / 0.33 | 29 / 12.02 / 0.85 |
| fifty_fifty | 1029 / 5.75 / 0.67 | 1114 / 8.94 / 0.76 | 178 / 12.35 / 1.68 | 639 / 1.64 / 0.33 | 29 / 12.02 / 0.85 |
| qqq_only | 1029 / 13.07 / 1.14 | 1114 / 11.08 / 0.73 | 178 / 15.91 / 1.73 | 639 / 1.60 / 0.32 | 29 / 12.01 / 0.85 |
| spy_same_exposure | 1029 / 9.28 / 1.17 | 1114 / 8.45 / 0.74 | 178 / 12.64 / 1.80 | 639 / 1.60 / 0.32 | 29 / 12.01 / 0.85 |
| spy_buy_hold | 1029 / 12.74 / 1.16 | 1114 / 7.42 / 0.56 | 178 / 31.07 / 1.71 | 639 / 22.92 / 1.00 | 29 / 37.56 / 0.44 |

## HONESTY

- The regime classifier runs on VXX's series-B history (2018-01 →); earlier days carry a neutral VXX ratio, so pre-2018 labels lean on SPY's moving averages alone (the classifier's own degraded path).
- Costs are a stated 5 bp per side on traded value, not the measured fills; the live floor pays real spreads on regime flips (2,755 fills since May) that this does not model beyond that charge.
- The active sleeves are not simulated: every arm holds the same cash remainder, so the comparison isolates the floor's composition under the live exposure schedule, not the account.
- Ledoit–Wolf intervals use the iid approximation with the realized correlation; the DSR bar is the E[max SR] of N=4 arms over their Sharpe dispersion — four correlated arms, so the bar is conservative in one direction and the overlap in the other.
- One history, one draw: ten years contain two bear phases; per-regime rows are the regime-conditioning the REASONING STANDARD demands, and the small-day regimes are noise.
