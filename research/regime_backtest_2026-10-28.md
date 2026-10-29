# Regime-layer backtest — 2026-10-28

Index-level, isolated (scripts/regime_backtest.py). Window 2007-08-28 → 2026-10-26, lag 2 session(s), 5 bp per unit of turnover, leverage capped at 1, cash earns 0. Exposure decided on SPY closes and applied to each column's index. NOT a strategy backtest (docs/regime/FINDINGS.md): no universe, no fills, no options.

## Full window

| Arm | SPY CAGR / max DD / Sharpe | IWM CAGR / max DD / Sharpe | flips/yr | time in mkt |
|---|---|---|---|---|
| buy_and_hold | 11.05% / -55.2% / 0.63 | 8.26% / -58.4% / 0.45 | 0.0 | 100% |
| sma200_daily | 8.26% / -24.2% / 0.71 | 5.39% / -37.7% / 0.39 | 4.8 | 81% |
| faber_monthly | 7.93% / -27.7% / 0.66 | 5.38% / -41.8% / 0.39 | 1.6 | 80% |
| vol_target_10 | 8.03% / -25.4% / 0.78 | 5.75% / -25.7% / 0.46 | 125.4 | 100% |
| vol_target_15 | 10.13% / -34.7% / 0.78 | 7.79% / -34.0% / 0.51 | 71.6 | 100% |
| vol_target_20 | 11.05% / -40.2% / 0.76 | 8.76% / -40.1% / 0.52 | 38.8 | 100% |
| faber_x_vt15_floor0 | 7.05% / -24.3% / 0.65 | 4.38% / -39.5% / 0.35 | 40.7 | 80% |
| faber_x_vt15_floor50 | 8.64% / -26.4% / 0.75 | 6.17% / -34.9% / 0.45 | 66.8 | 100% |
| current_proxy | 4.30% / -20.2% / 0.56 | 1.36% / -28.6% / 0.17 | 22.9 | 61% |
| engine_quantized | 8.34% / -20.5% / 0.76 | 5.43% / -28.1% / 0.41 | 9.4 | 89% |
| engine_continuous | 8.06% / -20.5% / 0.74 | 5.32% / -27.9% / 0.41 | 33.6 | 90% |
| engine_core | 9.26% / -36.7% / 0.72 | 6.55% / -39.2% / 0.44 | 23.0 | 100% |

## Episodes (SPY)

| Arm | GFC 2007-08..2009-06 CAGR / DD | 2015-16 CAGR / DD | late 2018 CAGR / DD | 2020 CAGR / DD | 2022 CAGR / DD | 2023-26 CAGR / DD |
|---|---|---|---|---|---|---|
| buy_and_hold | -20.5% / -55.2% | -0.7% / -13.0% | -34.9% / -19.3% | -6.4% / -33.7% | -18.2% / -24.5% | 21.5% / -18.8% |
| sma200_daily | -4.5% / -17.8% | -13.7% / -14.5% | -36.0% / -14.5% | -8.0% / -12.5% | -18.9% / -19.3% | 14.5% / -12.0% |
| faber_monthly | -6.0% / -17.8% | -14.6% / -13.8% | -39.1% / -21.1% | -5.5% / -12.4% | -22.1% / -23.6% | 11.9% / -10.0% |
| vol_target_10 | -7.6% / -25.4% | -5.4% / -11.4% | -25.5% / -12.6% | -4.4% / -12.2% | -10.6% / -13.2% | 13.7% / -12.3% |
| vol_target_15 | -11.9% / -34.7% | -2.6% / -12.2% | -30.1% / -15.6% | -7.7% / -17.1% | -16.2% / -19.8% | 17.9% / -15.9% |
| vol_target_20 | -14.2% / -40.2% | 0.1% / -12.3% | -35.3% / -18.8% | -9.3% / -20.1% | -17.9% / -22.8% | 19.5% / -18.2% |
| faber_x_vt15_floor0 | -6.4% / -17.8% | -14.4% / -13.8% | -34.1% / -17.2% | -7.0% / -11.3% | -18.6% / -19.9% | 11.1% / -10.0% |
| faber_x_vt15_floor50 | -9.0% / -26.4% | -8.6% / -12.6% | -32.1% / -16.4% | -7.2% / -13.1% | -17.2% / -18.5% | 14.5% / -12.1% |
| current_proxy | -6.5% / -18.9% | -8.8% / -10.0% | -6.8% / -2.2% | -2.2% / -7.0% | -4.6% / -7.1% | 8.0% / -11.9% |
| engine_quantized | -4.1% / -17.1% | -7.5% / -11.9% | -17.0% / -7.6% | -18.5% / -15.2% | -9.4% / -11.0% | 14.4% / -9.2% |
| engine_continuous | -4.1% / -17.1% | -7.6% / -11.9% | -17.1% / -7.6% | -18.3% / -16.8% | -10.9% / -11.6% | 13.7% / -8.9% |
| engine_core | -11.9% / -36.7% | -4.4% / -11.9% | -24.4% / -12.6% | -6.5% / -22.3% | -11.3% / -15.1% | 16.9% / -11.9% |

## Exposure levels (days at each level, gating index)

- buy_and_hold: 1.00 → 4821
- sma200_daily: 1.00 → 3916, 0.00 → 905
- faber_monthly: 1.00 → 3876, 0.00 → 945
- vol_target_10: 1.00 → 1487, 0.56 → 71, 0.50 → 70, 0.70 → 66, 0.61 → 64, 0.71 → 63
- vol_target_15: 1.00 → 2981, 0.84 → 51, 0.75 → 50, 0.92 → 48, 0.98 → 47, 0.77 → 43
- vol_target_20: 1.00 → 3853, 0.68 → 35, 0.98 → 31, 0.72 → 31, 0.93 → 31, 0.99 → 29
- faber_x_vt15_floor0: 1.00 → 2914, 0.00 → 945, 0.98 → 43, 0.84 → 42, 0.92 → 37, 0.97 → 34
- faber_x_vt15_floor50: 1.00 → 2914, 0.50 → 73, 0.26 → 55, 0.27 → 48, 0.98 → 43, 0.84 → 42
- current_proxy: 0.00 → 1895, 1.00 → 1673, 0.50 → 644, 1.29 → 609
- engine_quantized: 1.00 → 2972, 0.75 → 569, 0.00 → 533, 0.25 → 386, 0.50 → 361
- engine_continuous: 1.00 → 2825, 0.00 → 493, 0.75 → 381, 0.50 → 192, 0.25 → 144, 0.24 → 22
- engine_core: 1.00 → 2825, 0.50 → 1222, 0.75 → 381, 0.96 → 19, 0.68 → 18, 0.97 → 18

## Pre-registered claims (gating index, full window)

- trend gated arms trail buy and hold cagr: **holds**
- loose vol dial is cagr neutral within 1pp: **holds**
- gated arms cut max drawdown: **holds**
- never fully out beats fully out cagr: **holds**
- current schedule proxy has lowest cagr: **holds**

## ±20% sensitivity (gating index, full window)

| Arm | Variant | CAGR | max DD | Sharpe | pairwise swaps vs base | claims |
|---|---|---|---|---|---|---|
| sma200_daily | window-20% | 9.08% | -22.5% | 0.78 | 7 | hold |
| sma200_daily | window+20% | 7.83% | -24.4% | 0.65 | 5 | hold |
| faber_monthly | months-20% | 9.52% | -27.7% | 0.79 | 14 | hold |
| faber_monthly | months+20% | 8.77% | -21.5% | 0.68 | 9 | hold |
| vol_target_10 | target-20% | 6.60% | -20.8% | 0.76 | 5 | hold |
| vol_target_10 | target+20% | 9.11% | -29.4% | 0.79 | 7 | hold |
| vol_target_10 | window-20% | 8.15% | -25.4% | 0.79 | 2 | hold |
| vol_target_10 | window+20% | 7.96% | -24.3% | 0.78 | 0 | hold |
| vol_target_15 | target-20% | 9.11% | -29.4% | 0.79 | 1 | hold |
| vol_target_15 | target+20% | 10.75% | -38.6% | 0.77 | 2 | hold |
| vol_target_15 | window-20% | 10.25% | -34.3% | 0.79 | 0 | hold |
| vol_target_15 | window+20% | 10.17% | -33.8% | 0.79 | 0 | hold |
| vol_target_20 | target-20% | 10.36% | -36.2% | 0.78 | 1 | hold |
| vol_target_20 | target+20% | 11.32% | -42.9% | 0.74 | 3 | hold |
| vol_target_20 | window-20% | 11.28% | -39.0% | 0.78 | 1 | hold |
| vol_target_20 | window+20% | 10.94% | -39.7% | 0.75 | 1 | hold |
| faber_x_vt15_floor0 | floor-20% | 7.05% | -24.3% | 0.65 | 0 | hold |
| faber_x_vt15_floor0 | floor+20% | 7.05% | -24.3% | 0.65 | 0 | hold |
| faber_x_vt15_floor0 | target-20% | 6.50% | -19.7% | 0.65 | 4 | hold |
| faber_x_vt15_floor0 | target+20% | 7.39% | -26.6% | 0.65 | 2 | hold |
| faber_x_vt15_floor0 | months-20% | 8.53% | -24.3% | 0.78 | 12 | hold |
| faber_x_vt15_floor0 | months+20% | 7.65% | -21.6% | 0.67 | 2 | hold |
| faber_x_vt15_floor0 | window-20% | 7.21% | -23.9% | 0.66 | 2 | hold |
| faber_x_vt15_floor0 | window+20% | 7.24% | -23.1% | 0.66 | 2 | hold |
| faber_x_vt15_floor50 | floor-20% | 8.33% | -24.6% | 0.73 | 3 | hold |
| faber_x_vt15_floor50 | floor+20% | 8.95% | -28.1% | 0.76 | 2 | hold |
| faber_x_vt15_floor50 | target-20% | 7.84% | -22.4% | 0.75 | 8 | hold |
| faber_x_vt15_floor50 | target+20% | 9.15% | -29.0% | 0.75 | 1 | hold |
| faber_x_vt15_floor50 | months-20% | 9.40% | -23.2% | 0.81 | 8 | hold |
| faber_x_vt15_floor50 | months+20% | 8.93% | -26.5% | 0.75 | 0 | hold |
| faber_x_vt15_floor50 | window-20% | 8.78% | -25.9% | 0.76 | 1 | hold |
| faber_x_vt15_floor50 | window+20% | 8.76% | -25.6% | 0.75 | 0 | hold |
| engine_quantized | band-20% | 7.91% | -22.0% | 0.73 | 6 | hold |
| engine_quantized | band+20% | 8.23% | -20.5% | 0.75 | 1 | hold |
| engine_quantized | filter_days-20% | 7.70% | -17.2% | 0.72 | 9 | hold |
| engine_quantized | filter_days+20% | 8.26% | -20.5% | 0.75 | 0 | hold |
| engine_quantized | vol_halflife-20% | 8.24% | -20.5% | 0.75 | 1 | hold |
| engine_quantized | vol_halflife+20% | 8.04% | -21.4% | 0.73 | 4 | hold |
| ALL | lag 1, cost 5 bp | — | — | — | 19 | hold |
| ALL | lag 2, cost 10 bp | — | — | — | 3 | hold |

**Every pre-registered claim holds under all 40 variants** (±20% on each parameter, lag 1, doubled cost).
Pairwise swaps between near-tied reference arms: 148 (diagnostic only; the arms are not the claim).

Verification: `python3 -m pytest -q test_regime_backtest.py`; rerun with the same flags to reproduce (cached bars refresh after 24 h, so the end date may advance).
