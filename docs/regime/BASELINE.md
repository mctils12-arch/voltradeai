# BASELINE — frozen reference before any regime change

Snapshot taken 2026-10-27 (UTC), repo HEAD `786c88ce` (tree identical to `main` after PR #1471), `package.json` 1.0.1168, live `server_version` 1.0.1168 (`/api/data/layers`), `/api/health` status `ok`. Every later change in this program is measured against this file. Nothing here was changed to produce it; all live numbers came from the token-gated diag probes (`account`, `positions`, `benchmark`, `sleeves`, `equity_curve`, `ml`, `shadow`, `scanner`) and the config was dumped by importing `system_config` / `regime_util` / `position_sizing` read-only. Raw JSON copies sit in the session scratchpad (`scratchpad/regime/baseline/*.json`); they are not committed because they carry account state.

## 1. Regime classifier in force (`regime_util.py:41-48`, consumed through `system_config.get_market_regime`)

| Threshold | Value | Meaning |
|---|---|---|
| `PANIC_VXX_THRESHOLD` | 1.40 | VXX close / VXX 30-day mean ≥ 1.40 → PANIC |
| `BEAR_VXX_THRESHOLD` | 1.15 | ≥ 1.15 → BEAR |
| `BEAR_SPY_THRESHOLD` | 0.94 | SPY / SPY 50-day MA < 0.94 → BEAR |
| slow-bear streak | 10 | ≥ 10 consecutive SPY closes below the 200-day MA → BEAR |
| `CAUTION_VXX_THRESHOLD` | 1.05 | ≥ 1.05 → CAUTION |
| `BULL_VXX_THRESHOLD` / `BULL_SPY_THRESHOLD` | 0.95 / 0.98 | ≤ 0.95 AND SPY/MA50 ≥ 0.98 AND above the 200-day → BULL; else NEUTRAL |

No hysteresis, no dwell, no confirmation anywhere except the floor managers (`FLOOR_REGIME_CONFIRM_CYCLES = 3` distinct macro readings for calming/lateral changes; escalations immediate; a fallback VXX reading holds the confirmed regime — `bot_engine.py:5047-5167`).

Inputs: `macro_data.get_macro_snapshot()` (5-minute cache) — VXX from 40 Alpaca daily bars, SPY from 220 Alpaca daily bars, spot `^VIX` from yfinance (5-day, **defaults to 20.0 on failure**), 11 SPDR one-day changes.

## 2. Adaptive caps per regime (`system_config.get_adaptive_params`, at account equity $96,590 — size-tier overlay applied)

| Regime | MAX_POSITIONS | MAX_POSITION_PCT | MAX_TOTAL_EXPOSURE | MIN_SCORE | MAX_STOP_PCT | TIME_STOP_DAYS (dead key) | MAX_OPTIONS_PCT | MAX_OPTIONS_POSITIONS |
|---|---|---|---|---|---|---|---|---|
| BULL | 6 | 0.15 | 0.95 | 63 | 6.0 | 10 | 0.08 | 6 |
| NEUTRAL | **0** | 0.08 | 0.95 | 63 | 6.0 | 10 | 0.08 | 6 |
| CAUTION | 6 | 0.10 | 0.60 | 67 | 6.0 | 10 | 0.08 | 6 |
| BEAR | 0 | 0.08 | 0.50 | 75 | 6.0 | 5 | 0.08 | 6 |
| PANIC | 0 | 0.06 | 0.30 | 75 | 5.0 | 3 | 0.08 | 6 |
| BEAR via 200-day streak | 0 | 0.08 | 0.50 | 75 | 6.0 | 5 | 0.08 | 6 |

Note the non-monotonic stock slots: NEUTRAL (calm, VXX ratio 0.95–1.05) allows **zero** stock positions while CAUTION (VXX ratio 1.05–1.15) allows six.

## 3. Floor exposure schedule (`system_config.py:135-200`)

| Regime | Basket floor (`SPY_FLOOR_*`) | GLD (`DEFENSIVE_FLOOR_*`) |
|---|---|---|
| BULL | 0.70 | 0.08 |
| NEUTRAL | 0.90 | 0.08 |
| CAUTION | 0.35 | 0.15 |
| BEAR | 0.00 | 0.30 |
| PANIC | 0.00 | 0.40 |

Basket: QQQ 0.50 / SMH 0.15 / KWEB 0.10 / VXUS 0.15 / CASH 0.10 (`FLOOR_BASKET_ENABLED True`), drift band 3 %, QQQ trend caps: death cross → 0.20, price below 200-day → 0.50; `DEFENSIVE_FLOOR_DEATHCROSS_MIN 0.25`. Convexity overlay `enabled: False` in config (but the code never reads the flag — see AUDIT.md §6).

## 4. Sizing scalars (`position_sizing.py`)

| Scalar | Values |
|---|---|
| `_regime_scalar` by spot VIX | <15 → 1.10, 15–20 → 1.00, 20–25 → 0.80, 25–30 → 0.65, 30–40 → 0.50, >40 → 0.30 (macro `None` on the CSP path → 1.0) |
| `_volatility_scalar` by daily ATR % | 1 % → 1.20, 2 % → 1.00, 3 % → 0.667, 5 % → 0.40 (clamp 0.25–1.20) |
| stress-index multiplier | calm 1.0 / elevated 0.75 / stressed 0.5 / crisis 0.0 |
| drawdown scalar | 0–5 % 1.0 / 5–10 % 0.8 / 10–15 % 0.6 / >15 % 0.4 (reads `peak_equity`; dead on a key mismatch per the cockpit audit) |
| `ABSOLUTE_MAX_POSITION_PCT` | 0.08 |
| Tier-1 CSP `size_scalar` by VXX ratio | 1.00 / 0.85 (≥1.05) / 0.60 (≥1.20) / 0.30 (≥1.40), ×0.75 at −10 % DD, ×0.50 at −15 % DD — **not passed to `select_contract`** |

Node constants (`server/bot.ts:360-378`): `MAX_POSITION_SIZE 0.20`, `MAX_TOTAL_EXPOSURE 0.30` (non-floor), `MAX_POSITIONS 8`, `MAX_OPTIONS_POSITIONS 6`, `minScoreThreshold 65` (raised to 75 by the win-rate diagnostic), drawdown kill −10 % from `equityPeak`, Python DD halt 18 %, `risk_kill_switch` −20 % / −3 % daily.

## 5. Universe definition at the baseline (see AUDIT.md §3 for the full funnel)

- Stock scan: every active, tradable Alpaca US equity (~11.6–12.1k symbols, 24-hour cache) → price ≥ $5 and volume ≥ 500k (100k in the first half hour) → quick score ≈ 3·min(|Δ%|,15) + 2·min(vol/1M,5) + mover bonuses → **top 5 deep-scored per scan** (`VOLTRADE_DEEP_SCORE_CAP`, trimmed to 3 under memory pressure) → post-score gates (slots, cooldown, regime, adaptive MIN_SCORE, sector veto, $50M dollar-volume floor, blocked list, spread ≤ 0.5 %, Kelly) → Node admission gates.
- CSP universe: same list → price $10..≈$255 (equity × 0.25 / 95), volume ≥ 1M, dollar volume ≥ $20M, not leveraged/blocked → 7-factor score → top 200 ∪ 17 anchors → Tier-1 affordability ≈ price ≤ $81 → first 3 unheld names.
- Options scanner: price ≥ $10, volume ≥ 1M, |Δ%| > 2 % → top 400 by |Δ%| + 20 anchors.
- No market-cap field exists on any selection path; no point-in-time copy of yesterday's universe is stored.

## 6. Live results at the baseline

| Item | Value |
|---|---|
| Equity / last equity / peak | $96,589.92 / $96,529.29 / $99,962.18 |
| Cash / long MV / short MV | $13,715 / $87,915 / −$5,040 |
| Positions | 12 (5 stocks incl. floor ETFs, 7 options); gross $92,955, net $82,875 |
| Benchmark since 2026-05-06 (174 d) | account −9.33 % vs SPY +4.82 % (gap −14.14 pp); after tax −12.55 % vs +3.66 % |
| Last 30 days | account −7.21 % vs SPY −0.79 % (gap −6.41 pp) |
| Equity curve (30 rows) | 2026-09-28 $103,811 → 2026-10-27 $96,787 |
| Sleeves (fills since 2026-05-07) | options 822 fills / 298 symbols / realized −$9,177 / unrealized −$657; floor 2,834 fills / +$1,554 realized; stocks 178 fills / 28 symbols / −$499 |
| Options by hold time | <1h 168 closes −$4,812; <1d 142 −$4,922; 1–7d 54 +$64; 8–30d 4 +$389; >30d 1 +$104 |
| Live stock trades (ML feedback) | 36 closed, win rate 27.8 %, avg win +9.08 %, avg loss −5.16 %, recent-20 win rate 20 % |
| Shadow portfolio | 20,000 records 2026-08-10 → 2026-10-27; decisions taken 8,102 / rejected_score 11,081 / rejected_heat 163 / rejected_other 591 / rejected_stock_slots 47 / rejected_masterkill 16; regime labels NEUTRAL_BULL 17,630 / NEUTRAL 2,370 |
| Shadow win rates +5d / +10d / +20d | taken 42.6 / 51.6 / 44.9 %; rejected_score 49.6 / 50.8 / 55.4 %; rejected_other 49 / 50 / 56.8 % |
| Taken same-day move | median 17.4 %, p95 25.3 %, 2,835 of 8,102 over 20 % |
| ML | model age 14.3 h; feedback 1,722 records (396 live); live outcome breakdown orphan_exit 340 / win 10 / loss 26 / open 20 |
| Scanner health | not degraded, 0 consecutive failures |

## 7. Test suite at the baseline

`python3 -m pytest -q` on this checkout: **2,833 passed, 1 failed, 2 skipped** in 67 s. The one failure (`test_insider_cusum_probe.py::TestRunProbeGracefulNoArchive::test_no_archived_quarters_returns_error_not_exception`) reads a leftover Form-4 quarter archive present in this sandbox's data directory and is unrelated to regime, volatility or breadth; it passes in CI's clean environment. Node tests and the counter ratchet are run per PR (`scripts/gated_tests.sh`, `scripts/counter_ratchet.sh`); the ratchet pins are in `ci/counter_baseline.txt` (`assertions 22139`, `silent_py_handlers 254`, `tests_run_in_ci 626`).

## 8. Reference artifacts this program will compare against

- `research/experiments.md` 2026-10-20 entries (exit-model reconciliation; floor basket vs index; sleeve ledger), 2026-10-26 entries (score anti-predictivity; gate relabels; CSP manager-rule simulation).
- `research/floor_basket_vs_index_2026-10-20.md`, `research/exit_model_reconciliation_2026-10-20.md`, `research/csp_manager_rule_sim_2026-10-26.md`.
- The shadow-bucket first read scheduled for 2026-11-16 (`rejected_regime_gate`, `rejected_stock_slots`, adaptive `rejected_score`).

Forward measurement against this baseline is defined in FINDINGS.md §Validation and in RUNBOOK.md.
