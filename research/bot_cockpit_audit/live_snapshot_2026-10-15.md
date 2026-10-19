# Live snapshot — production read-only probes, 2026-10-15 ~03:12–03:25Z (23:12 ET Oct 14, market closed)

Source: `GET https://voltradeai.com/api/health`, `/api/bot/market-status`, `/api/ml/status` (public) and the token-gated read-only `/api/diag/<probe>` endpoints (`server/bot.ts:2426`). Values are the paper account. This file is a digest; it anchors the HEALTHY VALUES column of the audit to reality on one date. Raw JSON is not committed.

## /api/health (public)
`status ok`; `checks.bot = {status: active, equityPeak: 110727.04, drawdownPct: "-10.0", liveness: {dark: false}, autoResume.outcome: NOT_EVALUATED}`; `checks.scanner.consecutiveFailures 0`; `checks.server {uptime_s 1829, heap_used_mb 1318, rss_mb 1741}`; `checks.memory.pressure ok`; `checks.process` counters 0; `serving.gates [server, database]`.

## /api/bot/market-status (public)
`{isOpen: false, nextOpen: 2026-10-15T09:30:00-04:00, nextClose: 2026-10-15T16:00:00-04:00}`

## /api/ml/status (public)
`model_exists true, model_age_hours 19.6, enabled true, last_accuracy 58, last_samples 10000, last_status "trained", last_steps [python_started, numpy_ok, lightgbm_ok, import_ok, training_start, training_done], last_train_time null` — NOTE `ml_status.json` is written only by `POST /api/ml/retrain` (`server/routes.ts:7171-7191`), so `last_*` reflect the last MANUAL retrain, not Tier 3.

## /api/diag/account
```
{
 "equity": 99615.03,
 "last_equity": 102226.48,
 "cash": 17601.88,
 "portfolio_value": 99615.03,
 "long_market_value": 83520.15,
 "short_market_value": -1507,
 "initial_margin": 74160.08,
 "maintenance_margin": 57456.05,
 "buying_power": 174663.94,
 "daytrading_buying_power": null,
 "multiplier": 4,
 "status": "ACTIVE",
 "equityPeak": 110727.04,
 "drawdownPct": -10.035498104166782,
 "killSwitch": false
}
```
Drawdown from the Node peak: (99615.03 − 110727.04) / 110727.04 = **−10.035%** — below the `maxDrawdownPct = -10` kill threshold with `killSwitch` still `false` (the Tier-1 kill check only runs while `/v2/clock.is_open`; see audit §9 / V37).

## /api/diag/positions-detail
count 11 (stocks 5, options 6); grossExposure 85024; netExposure 82010

| symbol | class | side | qty | avg_entry | current | unrealized_pl | unrealized_plpc |
|---|---|---|---|---|---|---|---|
| CRWV261120P00055000 | us_option | short | -1 | 2.47 | 2.99 | -52 | -0.21053 |
| GLD | us_equity | long | 21 | 381.047619 | 380.65 | -8.35 | -0.00104 |
| KWEB | us_equity | long | 388 | 26.5266 | 24.21 | -898.84 | -0.08733 |
| PFE261120P00027000 | us_option | short | -1 | 0.48 | 0.32 | 16 | 0.33333 |
| QQQ | us_equity | long | 61 | 741.319345 | 720.88 | -1246.8 | -0.02757 |
| SMH | us_equity | long | 20 | 551.427288 | 555 | 71.45 | 0.00648 |
| TEM261120P00070000 | us_option | short | -1 | 7.6 | 8.4 | -80 | -0.10526 |
| VXUS | us_equity | long | 133 | 86.813308 | 83.13 | -489.88 | -0.04243 |
| VZ261120P00040000 | us_option | short | -1 | 0.49 | 0.32 | 17 | 0.34694 |
| WFC261120P00080000 | us_option | short | -1 | 2.48 | 2.63 | -15 | -0.06048 |
| XLF261120P00052000 | us_option | short | -1 | 0.41 | 0.41 | 0 | 0 |

Equity-position market value (QQQ+SMH+KWEB+VXUS floor basket + GLD defensive floor) = **$83,517.10 = 83.8% of equity**. Zero non-floor stock positions. Six short puts (one contract each, all Nov-20 expiry) are the only active-sleeve positions.

## /api/diag/ml
```
{
 "model_exists": true,
 "model_age_hours": 19.4,
 "feedback_count": 1674,
 "fills_count": 55,
 "feedback_seeded_count": 1326,
 "feedback_live_count": 348,
 "retrain_needed": false,
 "retrain_overdue": false,
 "live_performance": {
  "total_trades": 36,
  "win_rate": 27.8,
  "avg_win": 9.08,
  "avg_loss": -5.16,
  "recent_win_rate_20": 20
 },
 "live_outcome_breakdown": {
  "orphan_exit": 293,
  "win": 10,
  "loss": 26,
  "open": 19
 },
 "live_options_outcome_breakdown": {
  "orphan_exit": 30
 },
 "live_sessions": {
  "None": 293,
  "regular": 55
 },
 "live_entry_features_count": 55,
 "live_pnl_null_count": 312,
 "live_pnl_zero_count": 0,
 "live_record_date_range": [
  "2026-07-10",
  "2026-10-14"
 ]
}
```
`live_key_signatures` (top 4 record shapes among the 348 live records): 225× `{code_version,fill_price,note,outcome,pnl_pct,qty,side,ticker,time_filled}` (orphan_exit), 68× the same + `exit_reason`, 36× full closed entry (`…entry_features, exit_context, exit_price, exit_reason, exit_time, expected_price, fill_price, outcome, pnl_pct, score, session, slippage_pct, volume…`), 19× open entry. `live_code_versions` spans 1.0.34 → 1.0.1031 (≈100 distinct versions).

Edge arithmetic on the 36 closed live trades (the bot does not compute this): win rate 27.8%, avg win +9.08%, avg loss −5.16% → payoff ratio 1.76, breakeven win rate = 5.16/(9.08+5.16) = **36.2%**, expectancy = 0.278×9.08 − 0.722×5.16 = **−1.20% per trade** (gross of the synthetic haircut already inside fill_price; n = 36, so wide confidence bands).

## /api/diag/timings (scan started 2026-10-14T23:52:39Z)
| phase | duration_sec | total_sec |
|---|---|---|
| universe_load | 0.6 | 0.6 |
| snapshot_fetch | 4.45 | 5.05 |
| quick_scan | 0.16 | 5.21 |
| deep_score | 35.67 | 40.88 |
| step6a_trade_loop_and_covered_calls | 0.09 | 40.97 |
| step6_trade_loop_and_options | 0.05 | 41.02 |
| step7_manage_positions | 0.03 | 41.05 |
| step8b_options_manager | 0.32 | 41.37 |
| step9_third_leg | 0.12 | 41.49 |
| step9b_convexity_overlay | 0.34 | 41.83 |
| step10_spy_floor | 0.12 | 41.95 |
| step10b_defensive_floor | 0.13 | 42.08 |
| tier_engine_start | 0 | 42.08 |
| tier_engine_tier1 | 57.04 | 99.12 |
| tier_engine_tier2 | 0 | 99.12 |
| tier_engine_tier3 | 0 | 99.12 |
| tier_engine_tier4 | 0 | 99.12 |
| tier_engine_allocator | 0 | 99.12 |
| tier_engine_complete | 0 | 99.12 |
| scan_complete | 0 | 99.12 |

tier_engine_breakdown.tier_timings: `{"tier1_sec": 56.91, "csp_layer2_prefetch": {"cache_hit": false, "completed": 106, "total": 150, "elapsed_sec": 46.12, "budget_exceeded": true, "checked_at": 1792022048.8279104}, "tier2_sec": 0, "tier3_sec": 0, "tier4_sec": 0, "allocator_sec": 0, "total_sec": 56.92}`

status completed, total_sec 99.12. deep_score at 35.67 s = the 35 s wall cap is binding; tier1 57 s is the CSP Layer-2 prefetch with `budget_exceeded: true` (106/150) even though options slots are 6/6 and no CSP can be placed.

## /api/diag/daemon
```
{
 "alive": true,
 "uptime_seconds": 1846,
 "rss_mb": 166,
 "max_rss_mb": 1024,
 "modules_loaded": [
  "storage_config",
  "position_sizing",
  "options_execution",
  "ml_model_v2",
  "bot_engine"
 ],
 "active_dispatches": 1,
 "active_dispatch_detail": [
  {
   "method": "health",
   "elapsed_sec": 0
  }
 ],
 "layer2_prefetch": {},
 "layer1_stats": {},
 "surface_score_stats": {},
 "pid": 5
}
```

## /api/diag/scanner
`{"degraded": false, "consecutiveFailures": 0, "lastFailureDetail": null, "dataSourceErrors": {}}`  — reads healthy although several sub-feeds are structurally dead (audit §S1).

## /api/diag/shadow (counterfactual log)
total_records 20000, oldest 2026-08-03T11:39:54.093434+00:00, newest 2026-10-14T23:53:18.527098+00:00
by_decision: `{"rejected_score": 10438, "taken": 8372, "rejected_other": 681, "rejected_heat": 493, "rejected_masterkill": 16}`; by_regime: `{"NEUTRAL_BULL": 19984, "NEUTRAL": 16}`
labeled_by_horizon: `{"+5d": {"wins": 8193, "losses": 7889, "unknown": 3918}, "+10d": {"wins": 8086, "losses": 7408, "unknown": 4506}, "+20d": {"wins": 3491, "losses": 3212, "unknown": 13297}}`
win_rate_by_decision (+5d): rejected_score 51.8% (n=8098), taken 51.5% (n=6938), rejected_other 52.2% (n=602), rejected_heat 24.1% (n=428), rejected_masterkill 37.5% (n=16)

## /api/diag/shadow_bands
labeling: `{"method": "path_dependent_v2", "win_pct": 2, "stop_pct": -4, "note": "return_pct is entry->exit under the labeler's target/stop/timeout rules, not a raw forward return; 'all' is the base rate across every scored candidate. Labels are LONG-stock outcomes: they mean what they say only for action BUY. Records before 2026-10-06 carry no action ('unrecorded')."}`; score_quintile_edges [48.15, 56.592, 66.478, 77.4]
| quintile | +5d n | +5d win% | +5d mean% | +20d n | +20d win% | +20d mean% |
|---|---|---|---|---|---|---|
| Q1 | 3275 | 56.2 | -0.617 | 450 | 83.3 | 1.038 |
| Q2 | 2873 | 52.5 | -0.821 | 792 | 76.5 | 0.655 |
| Q3 | 3167 | 48.1 | -1.092 | 1373 | 57.4 | -0.51 |
| Q4 | 3103 | 56.8 | -0.589 | 1719 | 54.5 | -0.733 |
| Q5 | 3664 | 42.5 | -1.448 | 2369 | 33.2 | -2.009 |

`all` (base rate across every scored candidate): `{"all": {"+5d": {"n": 16082, "win_rate": 50.9, "mean_return_pct": -0.931, "median_return_pct": 2}, "+10d": {"n": 15494, "win_rate": 52.2, "mean_return_pct": -0.86, "median_return_pct": 2}, "+20d": {"n": 6703, "win_rate": 52.1, "mean_return_pct": -0.855, "median_return_pct": 2}}}`

## /api/diag/equity_curve (30-day window, newest first)
| date (UTC) | value | pnl |
|---|---|---|
| 2026-10-15 | 99655.03 | -2571.45 |
| 2026-10-14 | 99810.31 | -2416.17 |
| 2026-10-13 | 102180.71 | -560.99 |
| 2026-10-12 | 102592.64 | -1079.52 |
| 2026-10-11 | 103553.16 | -119 |
| 2026-10-10 | 103553.16 | -119 |
| 2026-10-09 | 103563.43 | 452.02 |
| 2026-10-08 | 103182.49 | -1572.5 |
| 2026-10-07 | 104846.67 | -834.21 |
| 2026-10-06 | 105773.85 | 283.33 |
| 2026-10-05 | 105592.66 | 713.52 |
| 2026-10-04 | 104879.14 | 0 |
| 2026-10-03 | 104879.14 | 0 |
| 2026-10-02 | 104882.95 | 786.88 |
| 2026-10-01 | 104239.32 | 332.45 |
| 2026-09-30 | 104063.56 | 221.1 |
| 2026-09-29 | 103996.93 | 312.08 |
| 2026-09-28 | 103811.36 | -495.7 |
| 2026-09-27 | 104306.06 | -1 |
| 2026-09-26 | 104306.06 | -1 |
| 2026-09-25 | 104389.7 | 458.65 |
| 2026-09-24 | 103929.22 | -158.55 |
| 2026-09-23 | 104103.44 | -817.13 |
| 2026-09-22 | 104964.45 | 670.02 |
| 2026-09-21 | 104467.18 | 1904.73 |
| 2026-09-20 | 102560.45 | -2 |
| 2026-09-19 | 102560.45 | -2 |
| 2026-09-18 | 102661.44 | 634.87 |
| 2026-09-17 | 101987.5 | 1252.92 |
| 2026-09-16 | 101068.09 | 337.27 |

## /api/diag/orders (last 100 closed/open orders)
window 2026-10-12T15:26 → 2026-10-14T14:14; by (type,status): `{('market', 'filled'): 32, ('limit', 'filled'): 30, ('limit', 'expired'): 4, ('limit', 'canceled'): 34}`; by asset_class: `{'us_equity': 42, 'us_option': 58}`. Equity orders are `market`; option orders are `limit` (some expire unfilled at the close, e.g. XLF put 0.29 limit expired 2026-10-13T20:00Z, refilled at 0.41 next morning). No QQQ option orders (the convexity overlay's would be).

## Audit log (SQLite `audit_log`, last 2000 rows ≈ 2026-10-14T13:58Z → 2026-10-15T03:15Z) — rows per type
From `/api/diag/audit?type=<T>&limit=500`. Window covers the Oct-14 afternoon RTH session (10:00–16:00 ET) and the evening.

```
BACKUP-ERROR           n=0
BACKUP                 n=0
BLOCKED                n=0
CIRCUIT-BREAKER        n=0
CORR-CAP               n=0
CSP-SKIP               n=0
DB-SLOW-WRITE          n=0
DD-HALT                n=0
DIAGNOSTIC             n=   7  2026-10-14T14:46 -> 2026-10-14T19:37
         7x [MEDIUM] Low win rate: 27.8% over 36 trades
DRAWDOWN-KILL          n=0
DUP-SKIP               n=0
EQUITY-READ-INVALID    n=0
EVENTLOOP-LAG          n= 106  2026-10-14T13:58 -> 2026-10-15T03:03
         2x Node event loop stalled ~1221ms past its 2000ms tick — check for a coi
         1x Node event loop stalled ~604ms past its 2000ms tick — check for a coin
         1x Node event loop stalled ~1204ms past its 2000ms tick — check for a coi
         1x Node event loop stalled ~6218ms past its 2000ms tick — check for a coi
EXECUTION              n= 125  2026-10-14T14:25 -> 2026-10-15T02:41
        25x Smart Execution: Morning queue — overnight research executed at 9:30am
        25x Smart Execution: Score-based replacement — better picks replace weaker
        25x Smart Execution: Stale order sweeper — cancel unfilled limits after 12
        25x Smart Execution: Regular hours → market orders for instant fill
FILL-CHECK-WARN        n=0
HEAT-CAP               n=0
INSTRUMENT             n=0
KILL-SWITCH            n=0
KILL-WARN              n=0
LEARN                  n=  17  2026-10-14T14:01 -> 2026-10-14T19:54
        17x Win rate 0% (20 post-deploy trades) — shifting weight toward VRP/squee
MANIPULATION           n= 150  2026-10-14T14:25 -> 2026-10-15T03:15
        29x SDST: Spiked to $0.03 then rejected — possible bull trap / stop hunt (
        28x TCRT: Spiked to $4.40 then rejected — possible bull trap / stop hunt (
        25x DKI: Spiked to $0.09 then rejected — possible bull trap / stop hunt (c
        25x SKYD: Spiked to $7.98 then rejected — possible bull trap / stop hunt (
MAX-RETRY              n=0
MORNING-SKIP           n=0
MORNING-STALE          n=0
MORNING-TRADE          n=0
MORNING                n=0
OPTIONS-SKIP           n=0
OPTIONS-SLOT-FULL      n= 158  2026-10-14T14:18 -> 2026-10-14T20:00
        49x VXUS: options slots full (6/6)
        49x KWEB: options slots full (6/6)
        38x TSLA: options slots full (6/6)
         3x APH: options slots full (6/6)
OPTIONS-TRADE          n=0
ORDER-REJECTED         n=0
PARTIAL-FILL           n=0
POS-KILL               n=0
POS-MONITOR-SYNC       n=0
POS-WARN               n=  34  2026-10-14T14:25 -> 2026-10-15T03:15
        21x CRWV261120P00055000: down -21.1% (warn threshold -15%)
         2x CRWV261120P00055000: down -31.6% (warn threshold -15%)
         1x CRWV261120P00055000: down -20.6% (warn threshold -15%)
         1x CRWV261120P00055000: down -27.5% (warn threshold -15%)
QUEUE                  n=  26  2026-10-14T20:14 -> 2026-10-14T23:37
        13x VXUS queued for market open (score 75)
        13x KWEB queued for market open (score 75)
REPLACE                n=0
RESEARCH-ERROR         n=0
RESEARCH               n=0
RETRAIN-ERROR          n=0
RETRAIN                n=0
SHADOW-BACKFILL-ERROR  n=0
SHADOW-BACKFILL        n=   3  2026-10-14T22:00 -> 2026-10-14T22:49
         1x {"updated":97,"already_filled":41141,"missing_price":85,"permanently_u
         1x {"updated":74,"already_filled":41085,"missing_price":176,"permanently_
         1x {"updated":83,"already_filled":41020,"missing_price":250,"permanently_
SHORT-BLOCKED          n=0
SHUTDOWN               n=  77  2026-10-14T14:25 -> 2026-10-15T02:41
        24x Graceful shutdown on SIGTERM
        24x All open orders cancelled
        24x Received SIGTERM — cancelling open orders and shutting down...
         4x Resolved in-flight refresh markers (clean stop, not a crash): shadowst
SIZE-CAP               n=0
SKIP                   n=0
STARTUP                n=  50  2026-10-14T14:25 -> 2026-10-15T02:41
        25x SQLite journal_mode=wal (KNOWN BROKEN #18 follow-up)
         2x Server boot — code_version 1.0.1084, pid 1
         2x Server boot — code_version 1.0.1081, pid 1
         2x Server boot — code_version 1.0.1071, pid 1
STREAM-DISCONNECT      n=   1  2026-10-15T00:20 -> 2026-10-15T00:20
         1x Feed disconnected — reconnecting in 10s
STREAM-ERROR           n=0
SWEEP                  n=0
SYSTEM                 n= 109  2026-10-14T14:25 -> 2026-10-15T03:15
        26x FEEDBACK: 1674 trades, all clean
        26x Mode: PAPER TRADING (Alpaca)
        26x === VolTradeAI Bot v2.0 Initialized ===
        21x Market is in a trading window. Tier 2 scan starting...
T1-ERR                 n=0
T1-FAIL                n=0
T1                     n=0
T3-SKIP                n=0
T3                     n=0
TIER-DEFER             n=0
TIER2-BACKOFF          n=0
TIER2-ERROR            n=0
TIER2-FAIL             n=0
TIER2-LIMIT            n=0
TIER2                  n= 200  2026-10-14T14:38 -> 2026-10-14T23:54
        67x Starting intelligence scan...
        22x Starting scan (interval: 2min based on market time)
        20x Starting scan (interval: 15min based on market time)
        18x Scanned 12138 stocks, 2 trade candidates (via daemon)
TIER3-DIAG             n=  31  2026-10-14T14:17 -> 2026-10-15T02:42
        21x System health: warning — 1 issues (node_uptime_s=32, tier2_ran_since_b
         5x System health: warning — 1 issues (node_uptime_s=3602, tier2_ran_since
         2x System health: warning — 1 issues (node_uptime_s=33, tier2_ran_since_b
         2x System health: warning — 1 issues (node_uptime_s=31, tier2_ran_since_b
TIER3-ML-ERROR         n=0
TIER3                  n=  94  2026-10-14T14:17 -> 2026-10-15T03:15
        32x Strategic scan complete
        31x Starting strategic scan...
         2x ML model fresh (13.7h old) — skipping retrain
         1x ML model fresh (19.4h old) — skipping retrain
TIME-EXIT              n=0
TRADE                  n=0
WS-EXIT                n=0
WS-SCALE-OUT           n=0
```

Reading: 25 process boots (24 redeploys, code_version 1.0.1064 → 1.0.1084) in 13 h, each cancelling all open orders on SIGTERM; zero `TRADE`, zero tier-1 CSP dispatch (`T1`), zero `SWEEP`, zero stock exits during a full market afternoon; `OPTIONS-SLOT-FULL` ×158 (the only thing the scan tried to do); `LEARN "Win rate 0% (20 post-deploy trades)"` ×17 (structurally false, audit §S11); shadow backfill ran 3× in one evening (once per boot, in-memory once-per-day flag).
