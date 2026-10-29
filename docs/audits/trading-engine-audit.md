# VolTradeAI trading-engine audit (read-only)

Date: 2026-10-27. Checkout: `main` @ v1.0.1168 (`f9892473`). Method: five parallel read-only passes (architecture+UI; data sources; universe/features/model; trading rules; performance/dead code/options gaps), every claim cited to file + function + approximate line; live numbers from read-only `/api/diag/*` GETs (no order endpoint was called, no repo code changed). Headline claims were spot-verified by the assembling session (account return, the tier-regime scope bug, the ticker-ordered CV leak). Items marked *inferred* or *unverified* were traced in code but not confirmed against live data. Line numbers drift; search by function name.

Prior audits this builds on: `research/bot_cockpit_audit.md` (+ `research/bot_cockpit_audit/`), `research/open_questions.md` KNOWN BROKEN (#41-#48), `research/experiments.md` 2026-10-13..10-27 entries.

## Executive summary (one page)

**What the system does today.** A paper-trading bot on Alpaca. A Node/Express orchestrator (`server/bot.ts`) runs a ~45 s position monitor, adaptive Tier-2 scans and hourly Tier-3 jobs, and calls a Python engine (`bot_engine.py`, via the Unix-socket daemon `voltrade_daemon.py` or subprocess fallback). Each scan pulls ~12k Alpaca snapshots, filters by price/volume, ranks with a direction-blind quick score, and fully scores only the top ~5 (`deep_score`, rules blended 60/40 with a LightGBM model). Trades are stocks (long/short), options (cash-secured puts, spreads, QQQ-put overlay) and a passive "floor" basket (QQQ/SMH/KWEB/VXUS ~70-90% of equity, GLD defensive). Live exits are executed only by `server/bot.ts checkPositionOnTick` (2×ATR stop clamped 1.5-8%, 1/3 scale-outs, breakeven, trails, 7-day conditional time stop); options exits by `options_manager.py`. Eight separate halt/kill mechanisms with four separate equity-peak stores.

**What it measurably achieves.** It does **not** beat SPY. 2026-05-06 → 10-27: account **−7.71%** vs SPY **+5.46%** (total return) — a 13.2-point shortfall; Sharpe **−0.40** vs 0.99; max drawdown **−18.6%** vs −4.5%; beta ~1.21, annualized alpha ~−27%. The options sleeve has **lost $9,177 realized** (all sleeves −$8,122; §7); the floor basket is the only positive sleeve. Live per-trade expectancy ≈ **−1.2%** (36 attributable trades, 27.8% win rate, +9.1% avg win vs −5.2% avg loss). Shadow logs show taken candidates doing worse than rejected ones and higher scores doing worse than lower scores (on overlapping per-scan records; a deduplicated probe is pending). There is **no evidence of an edge after costs**; measured costs are themselves unreliable because recorded fill prices are synthetic.

**Three biggest weaknesses.**
1. **The model's validation is not out-of-sample, and the model does not see what it claims to.** Training rows are grouped by ticker and never sorted by date, so "walk-forward" test folds overlap training dates (§5); early stopping uses the test fold; the earnings-surprise feature leaks future data; production now trains only on shadow records labelled with a +2%/−4% rule that matches no live exit; 9 of 34 features are constant at serve time and 8 differ between training and serving; the hourly retrain is SIGKILLed at 10 s and the 4am retrain/reset branch can never run (§1.x D1-D2, §5). Reported accuracy (58%) is not evidence of skill.
2. **Rules fight each other and the model.** At least six exit models (KB #48 extended, §6.8 N5), a tier engine that always sees regime "NEUTRAL" (Tier 3 can never fire, §6.8 N1), signals executed with the wrong instrument or direction (a "SELL OPTIONS" signal can open a naked stock short — inferred; an RSI-overbought "SELL" executes as a BUY; Node re-decides options into a bear put spread — §6.8 N2-N4), sizing that never reaches the options path (CSP collateral up to 20% vs an 8% cap), an unstopped SVXY position up to 40% of equity, and rule scores that re-count ~9 of the model's own features (§6.9). Deploys cancel every open order and wipe in-memory state (KB #46).
3. **Measurement is too weak to tell good from bad.** No live Sortino/Calmar/beta/rolling-vs-SPY; synthetic fill prices; 86% of trade-feedback records are orphan exits; backtests use Yahoo prices, a dead-config exit model by default and no options simulation; the audit log holds ~3 hours. The system cannot currently prove or disprove any edge, including a new options one.

**For an options layer.** Options data today is limited to Alpaca *indicative* snapshot chains (OPRA real-time was rejected with HTTP 403 in July) plus yfinance chains in `analyze.py`; an 81-day archive exists (`server/optionsChainArchive.ts`) but has no open interest, a midnight day-label bug and truncated index chains (§2.4). Polygon options entitlement is unknown (no key in the audit sandbox). Least-disruptive insertion: extend the existing chain archive → derive GEX/put-call/skew/term-structure/expected-move server-side → expose via a diag probe and shadow-log features for gate-2 testing before touching the 34-column model or rules (§10).

---

## Contents
1. Architecture map (+ 1.x defects) · 2. Data sources (2.4 options data) · 3. Universe and scanner · 4. Features · 5. Model · 6. Trading rules (6.7 known conflicts, 6.8 new conflicts, 6.9 duplication with the model) · 7. Performance tracking · 8. UI · 9. Dead code and bloat · 10. Gaps for an options layer

---

## 1. Architecture map

### 1.0 Process topology

- **Start-up.** `railway.json` / `railway.toml` `startCommand = "./run_with_daemon.sh"`. The Dockerfile `CMD ["node","dist/index.cjs"]` (Dockerfile ~98) is overridden by that startCommand. WORKDIR is `/app` (Dockerfile ~12/39).
- **`run_with_daemon.sh`** (FROZEN) does three things:
  - It reads the cgroup memory limit and runs `node --max-old-space-size=<60% of limit, floor 512, cap 6144>` (lines ~19-29).
  - It starts the daemon in a background loop, `while true; python3 voltrade_daemon.py; sleep 2` (~44-51). Any exit, including the RSS self-kill `os._exit(3)`, is respawned after 2 s.
  - It `exec`s Node in the foreground.
  - `VOLTRADE_DAEMON_ENABLED=false`, a missing `voltrade_daemon.py`, or a missing `python3` each mean Node runs alone (~31-42).
- **Node entry.**
  - `server/index.ts` calls `registerBillingRoutes` (~60), `registerNewsletterRoutes` (~69) and `registerRoutes` (~110).
  - `server/routes.ts:registerRoutes` (~570) calls `registerAuthRoutes(app)` (~604) and then `registerBotRoutes(app)` (~605). Bot routes are therefore registered before the rest of the routes in `routes.ts`.
  - It also registers globalAircraft, flightPlans, procedures, archiveOffload and aeroCharts routes (~1413-1566).
- **Persistence.**
  - JSON state lives under `DATA_DIR`. That is `/data/voltrade` when `/data` exists, otherwise `/tmp` (`storage_config.py` ~10-15). The cache dir is `/tmp/voltrade_alt_cache` (~18).
  - SQLite uses `better-sqlite3`. The DB path is `process.env.DB_PATH || DATA_DIR/voltrade.db` (`server/auth.ts` ~13-16).
  - Tables:
    - `users`, `sessions` and `password_resets` (auth.ts ~20-51)
    - `audit_log` (bot.ts ~1154, created at module load in bot.ts)
    - `watchlists` (routes.ts ~609)
    - `api_keys` (apiKeyAccounts.ts ~97)
    - billing has its own `new Database(dbPath)` (billing.ts ~35)
  - bot.ts sets `journal_mode=WAL` on the shared connection (~1169).

### 1.1 Python RPC daemon: `voltrade_daemon.py`

**Protocol** [code]
- Transport is a Unix stream socket at `VOLTRADE_DAEMON_SOCKET` (default `/tmp/voltrade_daemon.sock`, ~76), served by `ThreadingUnixServer(ThreadingMixIn, UnixStreamServer)` (~667).
- The socket is chmod 0666 (~766).
- Requests are one line of JSON, `{"method","args"}`, and responses are one line of JSON, `{"status":"ok","result"}` or `{"status":"error","error_message",["traceback"]}` (`RPCHandler.handle` ~584-664).
- The module docstring's `curl ... http://localhost/rpc` usage example (~36-40) is inaccurate: the server speaks raw line-delimited JSON, not HTTP.

**Limits** [code]

| Limit | Value | Source |
|---|---|---|
| `MAX_RSS_MB` | env `VOLTRADE_DAEMON_MAX_RSS_MB`, default 1024 | ~77 |
| `_memory_watchdog` | Every 30 s reads `ru_maxrss` (lifetime peak RSS, not current) and calls `os._exit(3)` when above `MAX_RSS_MB` | ~674-687 |
| `REQUEST_TIMEOUT_SEC` | 300 | ~78 |
| `MAX_INFLIGHT_REQUESTS` | env, default 8, enforced by `BoundedSemaphore` | ~86-87 |

- The handler waits up to 300 s for a semaphore slot (~597). It then runs the dispatch in a daemon thread and calls `join(300)` (~613-629).
- On timeout it answers `Request timed out after 300s`, but the thread keeps running because Python threads cannot be killed.
- `_active_dispatch_count` / `_active_dispatch_detail` track threads that are still running (~101-152). `_health()` exposes them (~470-489).

**Dispatch** (`RPCDispatcher.dispatch` ~334-468)
- Modules load lazily through `__import__` (`_lazy_import` ~245-255).
- Kwargs versus positional calling is decided with `inspect.signature(fn).bind(**args)` (~433-446).
- `run_full_scan` writes phase breadcrumbs to `/tmp/voltrade_daemon_trace.json` (~339-357, ~390-459).

**Startup** (`main` ~695-779)
1. Removes a stale socket.
2. Auto-seeds `trade_feedback.json` via `seed_feedback_from_backtest.py` when the file has fewer than 100 records (~715-745; set `VOLTRADE_AUTOSEED=false` to disable).
3. Starts the watchdog.
4. Warms `storage_config`, `position_sizing`, `options_execution`, `ml_model_v2` and `bot_engine` in a background thread (~752-758).
5. Calls `serve_forever`.

Numpy, pandas and requests are imported at module load (~230-238).

**Method table** (`RPCDispatcher.__init__` ~265-332)

| RPC method | Python target | Called from bot.ts? |
|---|---|---|
| `health`, `ping` | local `_health` / `_ping` | `health`: yes (~2300, 2517, 2805, 3670, 3697, 5516) |
| `ml_status` | module `ml_status_impl`. **The file does not exist** (no `ml_status_impl*.py`), so the dispatcher always falls back to the inline `_ml_status_fallback` (~375-376, ~518-556) | yes, `/api/monitoring/overview` ~3729 |
| `ml_toggle` | module `ml_toggle_impl` (does not exist), so `_ml_toggle_fallback` | not found in bot.ts |
| `track_fill` | `ml_model_v2.track_fill` | yes (~214, 5954, 6243, 4496) |
| `check_halt` | `position_sizing.check_halt_status` | yes ~5849 |
| `select_contract`, `submit_options_order` | `options_execution.*` | **not via RPC**. bot.ts calls them through `python3 -c` subprocesses (~5270, 5407, 5987) |
| `evaluate_and_execute` | `options_execution.evaluate_and_execute` | yes ~6065 |
| `risk_status` | `risk_kill_switch.get_kill_switch_status` | yes ~3705 |
| `get_peak_equity` | `risk_kill_switch.get_peak_equity` (returns a **float**) | yes ~3863 (see defect D3) |
| `check_position_risk` | `risk_kill_switch.check_position_risk` | not found |
| `check_correlation_pre_trade` | `risk_kill_switch.check_correlation_pre_trade` | yes ~5831 |
| `paper_resume_sync` | `paper_resume_sync.rebaseline_python_halts` | yes ~8118 |
| `macro_snapshot` | `macro_data.get_macro_snapshot` | not found as an RPC (bot.ts ~2397 uses inline python) |
| `shadow_stats` | `shadow_portfolio.get_shadow_stats` | not found |
| `shadow_backfill_exit_aligned` | `shadow_portfolio.backfill_exit_aligned` | registered but deliberately unused (comment ~296-307, bot.ts ~4695) |
| `shadow_update_decisions` | `shadow_portfolio.update_decisions_batch` | bot.ts ~6178 calls `pythonCall("shadow_update_decisions",...)` [code] |
| `csp_gate_shadow` | `tiered_strategy.log_csp_gate_shadow` | yes ~5455 |
| `cache_inventory` | local `_cache_inventory` | yes ~3888 |
| `run_full_scan` | `bot_engine.scan_market` | yes ~4840 (Tier 2) |
| `lookup_ticker`, `lookup_ledger_report` | `lookup_ticker.lookup`, `lookup_ledger.report` | yes ~2035, ~2069 |

**[DEFECT] Two RPC names that bot.ts calls are not registered.** Every call hits `Unknown method` and then runs the subprocess fallback, with a `console.warn` each time (`pythonCall` ~183):
- `strategy_pnl_stats` (`/api/monitoring/tier-pnl`, bot.ts ~3845)
- `snapshot_python_state` (`/api/system/snapshot`, bot.ts ~2356)

### 1.2 Node to Python bridge (`server/bot.ts`)

**`pythonRpc(method, args, timeoutMs?)`** (~117-154)
- Opens a `net.createConnection(DAEMON_SOCKET)` per call and writes one JSON line.
- Resolves on the first `\n` in the response.
- Default timeout `DAEMON_TIMEOUT_MS = 30000` (~115). On timeout it destroys the socket and returns `{status:"error","Daemon timeout"}`.
- `DAEMON_ENABLED = VOLTRADE_DAEMON_ENABLED !== "false"` (~114).

**`pythonCall(daemonMethod, daemonArgs, subprocessCmd, subprocessOpts, daemonTimeoutMs?)`** (~162-203)
1. Tries the daemon first.
2. `HEAVY_DAEMON_ONLY = {run_full_scan, scan_market, manage_positions}` never falls back to a subprocess when the daemon fails or throws. The result is `via:"daemon-failed"` or `"daemon-error"` (~173-194).
3. Light methods fall back to `execPythonSerialized(subprocessCmd)` and then `JSON.parse(stdout)` (~196-202).
4. When the daemon is **disabled by env**, even heavy methods go straight to the subprocess.

**`execPythonSerialized(cmd, opts)`** (~250-290)
- `hardTimeout = min(opts.timeout || 90000, 90000)` (~252). **Every serialized Python subprocess is SIGKILLed after at most 90 s, whatever timeout the caller asks for.**
- Non-trivial commands wait for a global mutex (`pythonRunning`), polling every 500 ms, up to `maxWait = opts.timeout || 30000`. The mutex is force-released when held longer than `hardTimeout + 10 s` (~259-277). A separate 30 s watchdog interval does the same (~296-307).

**`isTrivialPython(cmd)`** (~244-248): a command is "trivial" when `cmd.length <= 150` and the command string does not match `/import\s+(numpy|pandas|sklearn|lightgbm|scipy|torch)/`. Trivial commands **bypass the mutex** and run with `timeout = min(hardTimeout, 10000)` and `SIGKILL` (~255-257).
- The classifier looks only at the command text, not at what the imported module loads. A short `python3 x.py` or a short `python3 -c "from bot_engine import ..."` therefore counts as trivial.
- I evaluated the classifier on the real command strings with the same function (scratch `triv.js`, assuming the `/app/` path). These commands are trivial, so they are **killed at 10 s with no mutex**:

| Command | Caller | Requested timeout | Effective |
|---|---|---|---|
| `python3 -c "from bot_engine import manage_positions ..."` (120 chars) | `tier1Reflex` ~4557 | 15 s | 10 s |
| `python3 ml_retrain_safe.py` (26) | `tier3Strategic` ~6293; 4am retrain ~6775 | 900 s | **10 s** |
| `python3 -W ignore "/app/bot_engine.py" full` (43) | Tier-2 subprocess fallback ~4843 (used only when the daemon is disabled by env) | 300 s | **10 s** |
| `python3 -W ignore ".../bot_engine.py" full 2>/dev/null` (55) | overnight evening scan ~6582 | 300 s | 10 s |
| `python3 ".../bot_engine.py" scan` (33) | pre-market report ~6747 | 180 s | 10 s |
| `python3 ".../backtest.py" SPY all 2` (36) | overnight backtest ~6710 | 120 s | 10 s |
| `python3 -c "from shadow_portfolio import backfill_outcomes ... (3000)"` (116) | `checkShadowBackfill` ~4630 | 120 s | **10 s** |
| `python3 -c "... from ml_insight import ml_insight ..."` (123) | `/api/bot/insight` ~8572 | 40 s | **10 s** |
| `python3 -c "from intraday_shorts import close_open_shorts ..."` (111) | EOD cover ~5100 | 30 s | 10 s |
| halt check fallback (117) | ~5851 | 8 s | 8 s |

- Not trivial, so behind the mutex with a 90 s cap: `get_auto_fix_params` (159 chars) and `check_weekly_loss` (170), plus the multi-line SELL_CSP, BUY_PUT and scanner-options scripts (all over 150 chars).
- **[DEFECT, unverified-runtime]** The ML retrain (Tier 3 and 4am) asks for 900 s and is SIGKILLed at 10 s. The bot.ts code comments (~6286-6293, ~6772-6776) document 900 s. This was already reported in `research/bot_cockpit_audit/A_orchestration_health.md` §1 (lines 11-27, 1290), and the current code still does it.
- **[DEFECT, unverified-runtime]** The nightly shadow backfill (`backfill_outcomes(3000)`, ~4630) and the `/api/bot/insight` band report (comment ~8555 says "~20 s cold", asks for 40 s) both hit the same 10 s cap. The insight route answers 503 with `stale` on failure (~8574-8577).
  - Contrast `checkShadowBackfillExitAligned`. Its command was made longer than 150 chars on purpose to escape this trap (comment ~4680-4693), and `server/shadowBackfillExitAligned.test.ts` pins that.

**Other Python spawn paths that bypass the mutex**
- `routes.ts` uses its own `execAsync = promisify(exec)` (~168) with no mutex, no SIGKILL policy and no thread-capping env.
  - `/api/analyze/:ticker` → `python3 analyze.py TICKER` (~678, 120 s)
  - `/api/insights/:ticker` → `insights.py` (~716, 90 s)
  - `/api/research/:ticker` → `python3 -m alphadesk` (~773, 120 s)
  - `/api/scan` → `scanBatch` → `scanSingleTicker` → `python3 analyze.py T --mode=scan` (~279-284), **20 at a time through `Promise.allSettled`** (~320, ~336-341)
  - `/api/ml/status` → `python3 ml_status.py` (~7339); `/api/ml/toggle` (~7362); `/api/ml/retrain` (~7378, 300 s)
  - `/api/bot/system-status` → `system_status.py` (~6785)
  - `/api/shorts/dashboard` (~7201)
  - `/api/etf/:ticker` → `etf_analyzer.py` (~6634)
- These routes are not owner-gated (`/api/analyze`, `/api/insights`, `/api/research`, `/api/scan`), so any visitor can trigger Python processes [code].
- `bot_engine.get_stock_details` (~475-509) itself spawns `python3 analyze.py TICKER --mode=scan` through `subprocess.run(timeout=30)`. `deep_score` also spawns a `python3 -c` yfinance fundamentals subprocess with a 5 s hard timeout (~945-996).
- **`--mode=scan` is ignored.** `analyze.py __main__` (~2642-2663) reads only `sys.argv[1]` and runs the full `analyze_ticker` for any non-"SCAN" argument [code].

### 1.3 Node/Express scheduling (`server/bot.ts`, all inside `registerBotRoutes` ~1211)

| Loop | Cadence | Gate | Body | Source |
|---|---|---|---|---|
| Tier 0 stream | WebSocket `wss://stream.data.alpaca.markets/v2/iex`, 1-minute bars for `STREAM_TICKERS` (~6848) | `state.active && !killSwitch` (reconnect after a flat 10 s, ~7906-7920) | `checkPositionOnTick` for monitored tickers. Volume/momentum spikes go to `streamSignalQueue` (≥2.5x vol, ≥0.8% bar, ≥0.3% mom5, 10-min cooldown) | `startStreaming` ~7793-7921 |
| Position sync | 60 s | active, not killed | `syncMonitoredPositions` (TIME-EXIT ~6955, POS-KILL ~7047) | ~7788-7791 |
| **Tier 1** | **45 s** | If killed it runs `killSwitchAutoResumeTick` instead; otherwise needs active, no circuit breaker, `clock.is_open` | `tier1Reflex` (below) | ~8165-8180 |
| **Tier 2** | adaptive `getTier2Interval`: 9:30-10:30 60 s, -12:00 3 min, -14:00 7 min, -16:00 2 min, 16-20 15 min, 4-9:30 5 min, else 30 min. Back-off 60 s→600 s on failure. First run at +10 s | active, not killed, no circuit breaker, and `isAnyWindow = clock.is_open \|\| 4 ≤ ET < 20` | `tier2Intelligence(isOpen, etH)` | ~8182-8298 |
| **Tier 3** | 3 600 s; plus one run at +30 s after startup | `state.active` (no market gate) | `tier3Strategic` | ~8300-8333 |
| Shadow backfill | 5-min poll, runs in UTC hour 22 | none | `checkShadowBackfill` and the lookup-ledger backfill | ~7949, ~4625-4672 |
| Exit-aligned shadow backfill | 5-min poll | none | `checkShadowBackfillExitAligned` | ~7952, ~4720 |
| Event-loop lag, tmp cleanup, mutex watchdog | various | none | | ~7935-7943, ~321-327, ~296-307 |

- Tier-2 intervals are **hardcoded in bot.ts**. They mirror `system_config.BASE_CONFIG TIER2_*_MS` (system_config.py ~39-45) by comment only; they are not read from it.
- CLAUDE.md's CODEBASE MAP says "Tier 1 stop monitoring ~30s". The code is 45 s (bot.ts ~8180). This is a documentation drift.

**`tier1Reflex`** (~4535-4593)
1. `checkDrawdownKillSwitch` (~4265-4331): validated `evaluateDrawdown`; kill at `state.maxDrawdownPct = -10`. On kill it sends `DELETE /v2/orders` (~4286). It also has an optional `VOLTRADE_LIQUIDATE_ON_KILL` liquidation at -25% or worse (~4294-4321).
2. `sweepStaleOrders` (~4152).
3. `manage_positions()` via a 10 s trivial subprocess. Its actions are **only audited (`POS-MONITOR-SYNC`), not executed**; the call refreshes `voltrade_stop_state.json` (~4556-4575).
4. `syncMonitoredPositions`.
5. `executeMorningQueue` on the first open cycle (~4580-4585, ~4333).
6. `trackClosedTrades` (~867).

**`tier2Intelligence`** (~4769-5711)
1. Alpaca `/v2/account` ping.
2. Every 5th cycle: `diagnostics.get_auto_fix_params`, which sets `positionSizeMultiplier` and `minScoreThreshold || 65` and may pause.
3. Drains the stream queue (audited only; the scan does not consume the queued tickers).
4. **`pythonCall("run_full_scan", {}, "python3 -W ignore bot_engine.py full", {timeout:300000}, 300000)`** (~4840-4846).
5. Audits SIZE-BLOCK, options management, DD-HALT and SHADOW-GATES. Stores `lastScanResult` (in memory only, ~4525/4929). Rebuilds the in-memory `signals` list (max 30, 1 h expiry) from `top_10` (score ≥ 60), `new_trades`, `position_actions` and `tier_actions` (~4931-5010).
6. Daily loss check at **≤ -3%** with validated reads (~5061-5072), then weekly loss check (~5076-5092).
7. EOD intraday-short cover at ≥ 15:50 ET (~5097-5110).
8. If the market is open, `executeTrades(result.new_trades, equity)`; otherwise the trades are queued in `morningQueue` (~5112-5122).
9. **Tier dispatcher** (~5127-5473): re-fetches positions, open orders and closed orders, then re-checks the options slot cap of 6 and the cash budget.
   - SELL_CSP goes to an inline `python3 -c` that calls `options_execution.select_contract` and then `submit_options_order` (~5255-5340).
   - BUY (Tier 3) is a direct `alpaca("/v2/orders", POST)` (~5344-5391).
   - BUY_PUT (Tier 4) uses the same inline `select_contract`/`submit_options_order` (~5394-5445).
   - CSP gate rejections go to `csp_gate_shadow` (~5451-5469).
10. Kill-status and tier-kill audits (~5476-5492).
11. **Overnight research** is called at the end only when `!isMarketOpen && (ET ≥ 20 || ET < 4)` (~5702-5710).

**[DEFECT, code] Most of `runOvernightResearch` is unreachable.**
- `scheduleTier2` only enters `tier2Intelligence` when `4 ≤ ET < 20` or the market is open (~8245-8249). So the `(ET ≥ 20 || ET < 4)` condition at ~5704 can only hold when a scan that started just before 20:00 finishes after it, or when the owner calls `/api/bot/run-now` at night (~8520-8530).
- Inside `runOvernightResearch` (~6571-6840) the `Math.floor(etHour) === 4` branch (~6758) can never run, because the caller requires `ET < 4` or `ET ≥ 20`. That branch holds:
  - the **daily reset** of `dailyBlockedTickers`, `consecutiveStopLosses`, `morningQueueExecuted=false`, `orderAttemptCounts` and `sweptCspAt` (~6760-6770)
  - the 4am ML retrain
  - the nightly backup (~6785-6833)
- `grep` finds no other reset of `state.morningQueueExecuted = false` or `orderAttemptCounts = {}` in bot.ts (only ~6766/6768).
- Consequence: after the first morning-queue run following a boot, `morningQueueExecuted` stays true until the next process restart. Per-ticker attempt counts (`MAX_ORDER_ATTEMPTS_PER_TICKER`, ~5866) are only cleared by a restart. **[unverified-runtime]** Frequent deploys may hide this.

**`executeTrades(trades, equity)`** (~5715-6260): Node-side admission gates, in order.
1. Options slot cap of 6 (OPTIONS-SLOT-FULL).
2. Score below `state.minScoreThreshold` (SKIP).
3. Tickers blocked for the day.
4. Size ceiling `MAX_POSITION_SIZE=0.20` (SIZE-CAP).
5. Heat cap `MAX_TOTAL_EXPOSURE=0.30`, which excludes floor and leg tickers (HEAT-CAP).
6. `check_correlation_pre_trade` with a 0.40 sector cap (CORR-CAP, fail-open).
7. `check_halt` (HALT-SKIP, fail-open).
8. Max attempts; duplicate open order (DUP-SKIP); buying power, else `replaceIfBetter`.

Then by instrument:
- **ETF branch:** `alpaca("/v2/orders", POST)` for half the shares (~5907-5970).
- **Options branch:**
  - The scanner path (`regime_at_entry==="OPTIONS_SCANNER"`) runs an inline `select_contract` + `submit_options_order` + `register_options_entry` + `merge_options_meta` (~5977-6053).
  - Otherwise `pythonCall("evaluate_and_execute")` (~6056-6077).
- **Stock branch:** the HARD shorting block (~6115-6119), then `alpaca("/v2/orders", POST)` with `getOrderParams` (DAY limit at ask+0.1% during RTH, `server/orderParams.ts:getOrderParams` ~19-49) (~6107-6164).
- Gate rejections are batch-relabelled into the shadow log (~6170-6189). Fills are confirmed and `track_fill` is recorded (~6195-6258).

**`tier3Strategic`** (~6263-6510)
1. ML model age check, then a retrain if older than 24 h (10 s cap, see 1.2).
2. `manipulation_detect.scan_for_manipulation`.
3. `diagnostics.run_diagnostics` (TIER3-DIAG).
4. `recordDailyEquity()` (~6431, see 1.4).
5. `cftc_cot.run_daily_update`.
6. `sec_form4_bulk.run_update`.
7. Port-dwell weekly capture.

**Constant drift between Node and Python** [code, inconsistency]
- bot.ts `MAX_POSITIONS = 8` (~364) versus `system_config BASE_CONFIG MAX_POSITIONS = 6` (~92). Python's adaptive value is 8 in BULL and 0 in NEUTRAL/BEAR/PANIC (system_config.py ~459-509).
- bot.ts `state.minScoreThreshold` defaults to 65 (~507, ~4800) versus `MIN_SCORE 63` in system_config (~62).
- `/api/bot/status` returns hardcoded `maxPositionPct: 8, maxExposurePct: 95, dailyLossLimitPct: 5` (bot.ts ~2578-2581). The enforced Node daily loss halt is **-3%** (`DAILY_LOSS_LIMIT=-3` ~361; check ~5067). **[DEFECT, minor]** The status API misreports the active daily-loss limit. The client `BotStatus` type (useBotData.ts ~22-27) does not read these fields, so this is not visible in the UI today.

### 1.4 Output surfaces (Node to UI)

- **Audit log.** `audit(action, detail)` (bot.ts ~526-533) does three things:
  - writes `console.log`
  - calls `broadcastSSE(entry)`
  - calls `persistAudit`, an SQLite `INSERT` into `audit_log` (500-char message) followed by a `DELETE` that keeps the last 2000 rows (~1176-1196)
- `/api/bot/audit` returns the last 100 rows (~2651-2654).
- **SSE.** `GET /api/bot/stream` (owner) registers the response in `sseClients` (~1228-1236), and `broadcastSSE` writes every audit entry (~520-524).
  - **No client code uses `EventSource` or `/api/bot/stream`** (grep over `client/src`). The UI polls `/api/bot/audit` every 10 s instead (useBotData.ts ~149). The SSE fan-out runs but nothing consumes it [code].
- **Equity curve.**
  - Loaded at boot from `/data/voltrade/voltrade_equity_curve.json` or the `/tmp` fallback (~576-597).
  - Updated by `recordDailyEquity` (~1037-1056), called only from `tier3Strategic` (~6431), so hourly and only while `state.active`.
  - The key is the **UTC** date (`toISOString().split("T")[0]`). A new day therefore starts at 20:00/19:00 ET.
  - Served in `/api/bot/performance.equityCurve` (~1964) and used by `/api/bot/benchmark` and `/api/bot/sleeves` reconciliation.
- **Equity peak and kill switch.** Persisted to `/data/voltrade/voltrade_equity_peak.json` (~620-660) and the kill-switch file (`saveKillSwitch` ~691). Liveness lives in its own file (`loadLiveness` / `saveLiveness` ~748-780).
- **In-memory only (lost on every restart):** `lastScanResult` (~4525), `signals` (~3927), `notifications` (~543), `morningQueue` (~4059), `openOrders` tracking, `monitoredPositions`.

### 1.5 Python trading engine components

**`bot_engine.py`** (295 KB). All reached at runtime via `run_full_scan` and Tier 1 [code].
- `scan_market()` (~2551-2612): wrapper. Arms SIGALRM 55 s only on the main thread, which means only in subprocess mode; in the daemon worker thread the alarm is skipped. Returns `_partial_scan_result` on `TimeoutError`.
- `_scan_market_inner()` (~2700-4298), in order:
  1. `VOLTRADE_ULTRA_SURVIVAL_MODE` early return (default off, ~2773-2805).
  2. Portfolio DD halt: `update_equity_peak` with regime from `macro_data.get_macro_snapshot` + `system_config.get_market_regime`. When halted it returns `{halted, halt_reason, ...}` (~2809-2850, `DRAWDOWN_HALT_PCT 18`, system_config ~308).
  3. Universe: `_get_full_universe()` (~2471), plus the Alpaca screener movers top 50 (~2853-2883).
  4. Snapshots: sequential batches of 400 from `/v2/stocks/snapshots?feed=data_feed()`, trimmed to `(c,o,v,pc)` (~2887-2934).
  5. Quick score: price/volume floor (`MIN_PRICE`, `MIN_VOLUME`, or 100k in the opening half hour), formula `min(|chg|,15)*3 + vol + direction bonus + extreme penalty` (~2950-2995). Then sort.
  6. `SURVIVAL_MODE` RSS check (700 MB, ~3044).
  7. Deep-score guard: `_deep_score_guard_decision` skips at ≥ 550 MB, trims to 3 at ≥ 400 MB (~3116-3146). Deep cap 5 by default, max 10, hard ceiling 15.
  8. **Serial** `deep_score` per candidate with a 35 s total cap (~3174-3214).
  9. Tie-breaking (~3219-3311).
  10. Trade loop (~3325-3830). Account and positions; `get_adaptive_params` slots; per-candidate gates (held, regime-aware stop cooldown from `voltrade_stop_cooldown.json`, regime gate, sector correlation, dollar-volume Fix D, 0.5% spread check). Then `instrument_selector.select_instrument` (fallback `options_execution.should_use_options`, ~3642-3669), `position_sizing.calculate_position` (~3680-3713), the volume-share cap (~3731-3753), and finally `update_decisions_batch` for gate relabels (~3805).
  11. Covered-call sweep (~3835-3930).
  12. `options_scanner.get_options_trades(min_score=60)` adds options trades, marked `regime_at_entry:"OPTIONS_SCANNER"` (~3947-4020).
  13. `manage_positions()` (~4029).
  14. Intraday shorts **permanently disabled** (~4036-4039).
  15. `options_manager.manage_options_positions` (~4044-4048).
  16. `_run_third_leg` (~4057).
  17. `_run_convexity_overlay` (~4068).
  18. `_resolve_floor_regime`, then `_manage_spy_floor` and `_manage_defensive_floor` (~4081-4103).
  19. Tier engine: `check_kill_switches`, then `TieredStrategy().run_tiers(ctx)`. When the kill fires, `suppress_entries_on_kill` removes scanner entries (~4118-4212).
  20. Return dict (~4226-4298).
- `deep_score(ticker, quick_result, _diag, min_score)` (~807-1958). Steps:
  1. `get_stock_details` (analyze.py subprocess, ~825).
  2. A parallel `ThreadPoolExecutor(5)` with 15 s per source: `macro_data.get_macro_snapshot` + `get_news_sentiment`, `intelligence.get_full_intelligence`, `alt_data.get_alt_data_score`, `social_data.get_social_intelligence`, and `finnhub_data.get_insider_sentiment` + `get_recommendation_trends` (~836-936).
  3. yfinance fundamentals subprocess (5 s).
  4. Alpaca 400-day bars for ADX/OBV (~1013-1077).
  5. Strategy modules `strategies/momentum.score`, `mean_reversion.score` and `squeeze.score`, loaded through `sys.path.insert` of `strategies/` at import (~139-161, ~1086-1170).
  6. EWMA/GARCH, macro/news, intelligence, alt-data, social and analyst/valuation factors (~1192-1475).
  7. `rules_only_score` (~1476).
  8. ML blend: `ml_model_v2.ml_score`. `rule_weight`/`ml_weight` come from `BLEND_TRACKER_PATH`, default 0.6/0.4; when conviction < 8 the ML term is neutralised to 50 (~1479-1850).
  9. Side (~1852).
  10. `shadow_portfolio.log_candidate` (~1881-1900).
- `manage_positions()` (~2045-2420): ATR/R-multiple phase logic and `voltrade_stop_state.json` persistence. It **returns actions only and never submits orders** (no POST in the function).
- `check_sector_correlation` (~1962), `update_equity_peak` / `is_trading_halted` (~584-745).
- **Python-side order POSTs** in bot_engine (frozen category "raw HTTP order POST paths"; listed, not changed):
  - third leg ~4531/4562
  - SQQQ close ~4620
  - convexity ~4758/4885
  - covered-call unwind ~4958
  - SPY floor ~5332/5368/5427/5537/5564
  - defensive floor ~5714/5784/5807
- Floor basket `{QQQ .50, SMH .15, KWEB .10, VXUS .15, CASH .10}` (system_config ~151-157). Defensive floor ticker GLD (~189).

**Other components**

| Component | Role | Reached at runtime? |
|---|---|---|
| `ml_model_v2.py` | LightGBM training (`train_model` ~1251, triple-barrier ~622, purged walk-forward CV ~665/711, poisoned-record cleanup ~998), `ml_score` (~1573), `track_fill` (~2355), `fills_slippage_stats` (~2627), options model (~2056/2110) | `ml_score` via deep_score; `track_fill` via RPC; retrain via `ml_retrain_safe.py` (10 s cap, see 1.2); `fills_slippage_stats` via `/api/bot/performance` and `/ml-status` |
| `system_config.py` | `BASE_CONFIG` (~30), `get_market_regime` (~386), `get_adaptive_params` (~441-600) | Yes, from bot_engine, tiered_strategy and others |
| `risk_kill_switch.py` (frozen mechanisms) | `check_kill_switches` (~154), `check_correlation_pre_trade` (~577), `get_kill_switch_status` (~708), `get_peak_equity` (~660), `should_liquidate_all` (~630) | Yes: scan (~4128), RPC, monitoring routes |
| `tiered_strategy.py` | `TieredStrategy.run_tiers` (~851): `master_kill_switch` (~279), `tier1_csp_core` (~466, universe from `csp_universe.get_top_csp_candidates` ~163), `tier2_leverage_multiplier` (~604), `tier3_trend_capture` (~682), `tier4_tail_hedge` (~754), `_enforce_exposure_cap` (~800). Returns actions only; Node dispatches them | Yes (bot_engine ~4157) |
| `csp_universe.py` | Layer-1 hard gates (~203) and Layer-2 scoring (~618: IV rank, VRP, skew, liquidity, earnings, history, stability); `get_top_csp_candidates` (~842) | Yes, via tier1 |
| `options_scanner.py` | `get_options_trades` (~1771) → `scan_options` (~1501): earnings IV crush, VXX panic put sale, high-IV premium sale, CSP normal-market setups | Yes (bot_engine ~3962) |
| `options_execution.py` | `should_use_options` (~251), `select_contract` (~377), per-strategy `_select_*`, **`submit_options_order` (~2286, POST ~2328) and `_submit_mleg_order` (~2369, POST ~2411): FROZEN**, `evaluate_and_execute` (~2548) | Yes (bot.ts inline scripts and RPC) |
| `options_manager.py` | `manage_options_positions` (~855): DTE / profit / roll; `_submit_close_order` POST ~313, `_close_strategy_mleg` POST ~543; `register_options_entry` (~1346), `merge_options_meta` (~1415) | Yes (bot_engine ~4046, bot.ts inline) |
| `instrument_selector.py` | `select_instrument` (~1277) scores stock vs ETF vs options (~475/546/720) | Yes (bot_engine ~3643) |
| `position_sizing.py` | `calculate_position` (~878), Kelly (~153), scalars, `check_halt_status` (~783) | Yes |
| `strategies/` (momentum, mean_reversion, squeeze) | `score()` functions | Yes, inside deep_score |
| `analyze.py` | Bennett-style vol engine; `analyze_ticker` (~1930) uses Alpaca bars first, then yfinance; `quick_scan` (~2471) | Yes: deep_score subprocess, `/api/analyze`, `/api/scan`, `generateSignals`, overnight `quick_scan` (~6619) |
| `insights.py` | `get_insights` (~694) using institutional_data, finnhub_data, analyze | UI only (`/api/insights/:ticker`) |
| `shadow_portfolio.py` | `log_candidate` (~270), `update_decisions_batch` (~426), `backfill_outcomes` (~651), `backfill_exit_aligned` (~1808), band reports (~2151-2271) | Yes (deep_score, Node relabels, nightly jobs, ml_insight) |
| `alt_data.py` | wiki attention, FRED, geopolitical, short interest, congressional, patents, FTD → `get_alt_data_score` (~517) | Yes (deep_score) |
| `macro_data.py` | `get_macro_snapshot` (~76), `get_news_sentiment` (~402) | Yes (scan, deep_score, sizing, instrument selector) |
| `social_data.py` | Reddit RSS, Google Trends, news RSS → `get_social_intelligence` (~415) | Yes (deep_score) |
| `intelligence.py` | `get_full_intelligence` (~527): news classification, insider, earnings pattern, event memory | Yes (deep_score, ml_model_v2) |
| `finnhub_data.py` | insider sentiment, recommendations, earnings | Yes (deep_score, insights) |
| `institutional_data.py` | 13F signals (`get_institutional_score` ~512) | **Only via `insights.py`** (UI); not in the trading path |
| `etf_data_sources.py` | ETF Builder data layer | **Only via `etf_analyzer.py`** (`/api/etf/:ticker`); not in the trading path |
| `alphadesk/` | separate package; `python -m alphadesk` | **Only `/api/research/:ticker`** (routes.ts ~747-795, "never places orders"); not in the trading path |
| `intraday_shorts.py` | `close_open_shorts` at EOD (bot.ts ~5100), dashboard (routes ~7202) | Entry disabled in scan (~4036); EOD cover still runs |
| `diagnostics.py`, `manipulation_detect.py`, `cftc_cot.py`, `sec_form4_bulk.py`, `ml_insight.py`, `lookup_ticker.py`, `paper_resume_sync.py`, `probability_engine.py`, `stress_index.py` | invoked from bot.ts inline Python or RPC | Yes, as noted at their call sites |

### 1.6 End-to-end data flow (one Tier-2 cycle, market open)

1. **Trigger.** `armTier2Timer` → `scheduleTier2` (bot.ts ~8213-8295) → `tier2Intelligence` (~4769).
2. **Ingestion.**
   - Alpaca account ping.
   - Daemon RPC `run_full_scan` → `bot_engine.scan_market` → `_scan_market_inner`.
   - Universe and screener movers (Alpaca); `/v2/stocks/snapshots` batches (~2853-2934).
   - Macro snapshot (`macro_data`, VXX ratio, SPY vs MA50/200).
3. **Quick filter and score.** Price/volume floor and quick_score over the whole universe (~2950-3010).
4. **Features and deep score** for the top ≤5 (memory guard; ≤3 when trimmed; 0 when skipped).
   - analyze.py subprocess (Alpaca bars, yfinance options/fundamentals).
   - Five parallel enrichment fetchers; the yfinance fundamentals subprocess; 400-day Alpaca bars → ADX/OBV/momentum.
   - Strategy module scores → rules score → LightGBM `ml_score` blend (0.6/0.4) → side.
   - The shadow record is logged as "taken" (~807-1958).
5. **Candidate generation.** Python gates (held, cooldown, regime, sector correlation, dollar volume, spread) → `select_instrument` (stock / 2x ETF / options) → `calculate_position` (Kelly/vol/heat/regime scalars; blocked → `sizing_blocks`) → volume-share cap → `new_trades` (~3409-3830).
   - The options scanner appends options trades (~3947-4020).
   - The tier engine produces `tier_actions`, a separate list (~4118-4212).
   - Kill-switch suppression strips entries when it fires.
6. **Python-side management and positioning within the same call.** These place orders themselves, from Python:
   - `manage_options_positions` (closes/rolls, POST in options_manager)
   - third leg, convexity overlay (QQQ puts), SPY floor basket, GLD defensive floor (POSTs in bot_engine)
   - `manage_positions` returns stock exit actions only.
7. **Return to Node.** The result dict comes back over the socket. Node audits it, stores `lastScanResult`, refreshes `signals`, then checks the daily/weekly loss limits (~4882-5092).
8. **Admission and order submission.** `executeTrades` Node gates (1.3) → instrument branch:
   - Stock and ETF: `alpaca("/v2/orders", POST)` in bot.ts (~6123, ~5912).
   - Options: inline Python `options_execution.select_contract` → **`submit_options_order` (FROZEN)**, or RPC `evaluate_and_execute`.
   - Tier dispatcher: SELL_CSP / BUY_PUT through the same frozen `submit_options_order`; Tier-3 BUY via the bot.ts POST (~5357).
   - **Frozen order paths, not modified here:**
     - `options_execution.submit_options_order` / `_submit_mleg_order`
     - every `alpaca("/v2/orders", {method:"POST"})` in bot.ts: ~4305, 4441, 5357, 5912, 6123, 6537, 6958, 7050, 7396, 7423, 7610, 7657
     - `DELETE /v2/orders` at ~2635, 4167, 4237, 4286
   - `placeOptionsOrder` (bot.ts ~1081) **has no callers** (grep), so it is dead code inside a frozen path.
   - Entries are tracked in `openOrders`, `addPositionToMonitor` and `track_fill`. Rejections are relabelled in the shadow log.
9. **Position management and exits.**
   - Real-time path: the IEX 1-minute bar stream → `checkPositionOnTick` (scale-out 1/3 at 1R and 2-3R, ATR trailing, stops/TP; options are skipped and left to options_manager) → bot.ts exit POSTs (~7423, ~7657) → `recordExitFill` → `track_fill`.
   - Periodic: `syncMonitoredPositions` TIME-EXIT and POS-KILL (~6955, ~7047).
   - Tier 1 drawdown kill (-10%) and auto-resume.
   - Python: options manager, floors, convexity roll. Python `manage_positions` only refreshes stop state.
10. **Learning.** `track_fill` writes `trade_feedback.json`. The nightly shadow backfill labels outcomes (10 s cap, see 1.2). ML retrain is hourly-if-stale in Tier 3 (10 s cap).
11. **UI.** Polling REST (section 8): status, account, positions, audit (SQLite), last-scan (in memory), performance (feedback file + equity curve), insight (ml_insight.py), sleeves and benchmark (Alpaca fills/activities/bars). The SSE stream exists but has no consumer.

---

### 1.x Architecture/UI defects flagged (facts only)

| # | Severity | Location | Fact |
|---|---|---|---|
| D1 | High | bot.ts `isTrivialPython` ~244 + callers ~6293, ~6775, ~4630, ~8572 | ML retrain (asks 900 s), nightly shadow backfill (120 s) and `/api/bot/insight` (40 s) all classify as "trivial" and are SIGKILLed at 10 s. All non-trivial serialized calls are capped at 90 s (~252). Previously reported in `research/bot_cockpit_audit/A_orchestration_health.md` §1; still present. |
| D2 | High | bot.ts ~5704 vs ~8245; ~6758 | Overnight research is effectively unreachable from the scheduler. The 4am branch (daily reset of `morningQueueExecuted`, `orderAttemptCounts`, `dailyBlockedTickers`, `consecutiveStopLosses`, `sweptCspAt`; 4am retrain; nightly backup) can never run. No other reset of `morningQueueExecuted`/`orderAttemptCounts` exists. |
| D3 | Medium | bot.ts `/api/monitoring/drawdown` ~3863-3868 | Through the daemon, `get_peak_equity` returns a float. The code reads `.peak`, gets `undefined`, and falls back to current equity, so the reported drawdown is about 0 whenever the daemon is up. |
| D4 | Medium | routes.ts ~7168 vs useBotData.ts ~156, panels.tsx ~661-664, BotPage ~130 | `/api/orders/open` returns `{orders}`, but the cockpit expects an array. The Orders panel always says "none working" and the stage row says "? open". |
| D5 | Low | bot.ts ~2356, ~3845 vs voltrade_daemon.py ~265-332 | RPC methods `snapshot_python_state` and `strategy_pnl_stats` are not registered, so every call falls back to a subprocess and logs a warning. |
| D6 | Low | voltrade_daemon.py ~271-272 | `ml_status_impl` / `ml_toggle_impl` modules do not exist; the inline fallbacks always run. |
| D7 | Low | bot.ts ~2578-2581 | `/api/bot/status` reports `dailyLossLimitPct: 5`, but the enforced halt is -3% (~361, ~5067). `MAX_POSITIONS` 8 (Node) vs 6 (system_config); `minScoreThreshold` 65 vs `MIN_SCORE` 63. |
| D8 | Info | bot.ts `/api/bot/stream` ~1228, `/api/bot/signals` ~4008; `components/TradingActivity.tsx`; bot.ts `placeOptionsOrder` ~1081 | Server SSE stream and signals list have no client consumer. TradingActivity is an orphan component. `placeOptionsOrder` has no callers. |
| D9 | Info | `analyze.py __main__` ~2642; `bot_engine.get_stock_details` ~491; routes.ts `scanSingleTicker` ~283 | The `--mode=scan` flag is ignored; the full `analyze_ticker` runs. `/api/scan` fans out 20 concurrent analyze.py processes outside the mutex, on public routes. |
| D10 | Info | CLAUDE.md CODEBASE MAP vs bot.ts ~8180 | The documentation says Tier 1 runs every ~30 s; the code runs it every 45 s. |
| D11 | Info | `voltrade_daemon.py` docstring ~36-40 | Documents HTTP over the socket; the protocol is raw line JSON. |

## 2. Data sources

Read-only audit of `/home/user/voltradeai` at 2026-10-27. Line numbers are approximate (±5). "Prod" means read-only GETs against `https://voltradeai-production.up.railway.app/api/diag/*` with the diag token, made during this audit. No provider keys are in this sandbox (only `EIA_API_KEY` and `DIAG_TOKEN` are set), so no provider endpoint was called directly.

## 2.0 Checking the known context

| Claim | Verdict | Evidence |
|---|---|---|
| There is no central data router | **Confirmed, with one nuance.** Nothing routes the fetches: every module builds its own `requests.get` or `fetch`. The only shared piece is the Alpaca **feed selector**. `alpaca_feed.py` has `data_feed()` (~l.71), `bars_feed()` (~l.115) and `options_feed()` (~l.173). It picks the `feed=` value and nothing else. Node has its own, separate selector: `server/alpacaFeed.ts:resolveAlpacaFeed` (l.11) defaults to `delayed_sip`. | Per-module URLs are listed in 2.1 |
| Every backtest falls back to Yahoo (`backtest_v2._yahoo_bars`) | **Partly true.** `backtest_v2.fetch_bars` (l.240-252) tries Alpaca first through `analyze._fetch_alpaca_bars` and only falls back to `_yahoo_bars` (l.181) when the keys are missing or the fetch returns empty. With no keys (this sandbox, CI) it is **always Yahoo**. **Side effect:** the two sources adjust prices differently. Alpaca uses `adjustment=split` only (`analyze.py:_fetch_alpaca_bars` l.80-118). Yahoo is rescaled for dividends and splits (`_yahoo_bars` l.207-214). So results depend on which source answered. `_fetch_alpaca_bars` also sends **no `feed=` param** (so the account default applies), uses a 5s timeout, and has no pagination. | backtest_v2.py l.24-25, 181-252 |
| `bot_backtest.py` is Yahoo-only | **Confirmed.** `fetch()` (l.47-70) calls `query1.finance.yahoo.com/v8/finance/chart`. Fixed universe of 12 ETFs (`UNIVERSE` l.35), window `2016-01-01`→`2026-04-07`, cached in `.bt_cache/`. `bot_backtest_subperiods.py` reuses it (l.26). | |
| Polygon is used for single-ticker reference + grouped daily + news | **Confirmed. Only one of these touches trading.** Reference (`/v3/reference/tickers/{t}`): `alt_data.get_ftd_signal` (l.495) is **live in deep_score** via `get_alt_data_score`. `macro_data.get_sector_for_ticker` (l.477-480) is **dead** (only `__main__` calls it). `etf_data_sources.py` (l.295) serves the site's ETF page. Grouped daily (`/v2/aggs/grouped/...`): `server/routes.ts /api/market-snapshot` (l.6858-6886) is **site only**; `ml_model.py:207` is an **orphan** (nothing imports `ml_model.py`). News (`/v2/reference/news`): `routes.ts /api/news` (l.6890-6906), **site only**. **Env-var split:** Python reads `POLYGON_KEY` (`macro_data.py:21`, `alt_data.py:14,495`, `social_data.py:20`, `intelligence.py:23`); the server reads `POLYGON_API_KEY` (`routes.ts:6859,6891`); only `etf_data_sources.py:101` accepts both. If prod sets only one name, the other side calls Polygon keyless and the error is swallowed. **Which names are set in prod is unknown.** | |
| No code fetches delisted tickers | **Confirmed.** Live universe = `GET /v2/assets?status=active&asset_class=us_equity` (`bot_engine._get_full_universe` l.2471-2516). ML training tickers = today's most-actives top 50 plus 30 hard-coded names (`ml_model_v2._fetch_training_bars` ~l.244-256). Backtests use explicit current tickers. A repo-wide grep finds no `status=inactive` or delisted fetch. The SEC research harness only *handles* delisted exits (`sec_filings_gate2.py` l.128-145). `research/wishlist.md` l.3-14 records the gate blocked on exactly this missing panel. | |

## 2.1 External APIs used by the trading engine and the server (finance)

"Trading" = the import closure of `bot_engine.py`, `ml_model_v2.py`, `options_*.py`, `csp_universe.py`, `vol_surface.py`, `position_sizing.py`, `instrument_selector.py`, `tiered_strategy.py`, `stress_index.py`, plus the `analyze.py` subprocess. The subprocess is launched by `bot_engine.get_stock_details` (l.474-509) for every deep-scored ticker. It passes `--mode=scan`, but `analyze.py __main__` (l.2642-2662) ignores that flag and runs the full `analyze_ticker`.

| Provider | Endpoints called (file:function ~line) | Plan/tier (as far as the repo shows) | Rate limiting | Fields kept | Storage | History depth fetched |
|---|---|---|---|---|---|---|
| **Alpaca Trading API** (`paper-api.alpaca.markets`, env `ALPACA_KEY`/`ALPACA_SECRET`/`ALPACA_BASE_URL`) | `/v2/account`, `/v2/positions` (`bot_engine.get_alpaca_account/positions` l.512-531, also l.4476-4487, 4675-4685); `/v2/orders` (frozen submit paths); `/v2/assets?status=active` (`_get_full_universe` l.2488); `/v2/assets/{t}` (`position_sizing.py` ~l.791); `/v2/options/contracts` (QQQ put discovery in `_run_convexity_overlay` l.4803) | Paper account | Python token bucket (see 2.2) | Account equity/last_equity, positions, asset symbol + name, contract symbol/strike/expiry | `universe_cache.json` (24h, `DATA_DIR`), `universe_names.json` (via `leveraged_etf_filter.write_universe_names`) | n/a |
| **Alpaca Market Data, stocks** (`data.alpaca.markets`) | Snapshots `/v2/stocks/snapshots` (full-universe batches of 400: `_scan_market_inner` l.2909-2925; `csp_universe._fetch_snap_data_for_universe` l.354; `macro_data` sector ETFs l.151; floors l.5501, 5758; QQQ l.4790) and `/v2/stocks/{t}/snapshot` (options_execution l.407, 913; bot.ts l.4398 with hard-coded `feed=delayed_sip`). Bars `/v2/stocks/bars` and `/v2/stocks/{t}/bars` (deep_score 400d/limit 300 l.1015-1021; VXX l.1554; credit-spread l.1668; floors l.5244, 5649; `macro_data` SPY 300d/220 and VXX 45d/40 l.210-300; `options_scanner._fetch_iv_rank` 400d/300 l.559-565; `vol_surface._fetch_historical_bars` 90d×1.6, paginated l.300-335; `alt_data` 15-30d l.307, 486; `ml_model_v2._fetch_training_bars` l.281). Latest quote `/v2/stocks/{t}/quotes/latest` (l.3612). Latest trade `/v2/stocks/trades/latest` (`vol_surface._fetch_spot_price` l.250). Screener `/v1beta1/screener/stocks/movers` and `most-actives` (scan l.2860, 2879; ML l.244). News `/v1beta1/news` (`intelligence.classify_news` l.125 limit 15; l.290, 375 limit 5; `macro_data.get_news_sentiment` l.419; `social_data` l.342; `ml_model_v2._fetch_historical_news_sentiment` l.315 is **dead**). WebSocket `wss://stream.data.alpaca.markets/v2/iex` for 1-minute bars of held tickers (`server/bot.ts` l.7809). Server proxy routes: snapshots/news/screener (`routes.ts` l.6908+), `/api/bars` and `fetchDailyBarsRange` with `feed=iex` (`bot.ts` l.2664, 3634) | **SIP real-time rejected with HTTP 403 since 2026-07-06** (`alpaca_feed.py` docstring l.1-30; `research/wishlist.md` ~l.1981-1997, "Algo Trader Plus ~$99/mo" not active). Snapshots/quotes run on `delayed_sip` (15-min delay). **Bars run on `iex`**, because the bars endpoint rejects `delayed_sip` with HTTP 400 (`bars_feed` l.115-139). Current prod feed state is **not exposed** by any diag probe (`feed_status()` l.141 has no caller in bot.ts or the daemon). | Python token bucket 180/min (2.2). Node: none | Snapshots trimmed to (c, o, v, prev_c) per symbol (l.2895-2925). Bars: OHLCV. News: headline/summary only for keyword scoring | Mostly in-memory or `/tmp` caches. `macro_data` `/tmp/voltrade_macro_cache.json` TTL 300s (l.21-22). `csp_universe` L1/L2 caches overwritten every 15 min (`DATA_DIR/voltrade_csp_universe_cache.json`, `..._scores_cache.json`, l.51-54, 314, 807). The stock price history is **not archived** | ML: **365 days** in `fast_mode` (prod path: `ml_retrain_safe.py:69` → `train_model(fast_mode=True)`; `start = now − max(days,365)` l.268; `ML_LOOKBACK_DAYS=60` `system_config.py:281`). ≤80 tickers. Batches of 10 symbols with `limit=10000` and **no `next_page_token` handling**: fine at 1 year, but it would truncate in the non-fast `2020-01-01` path (10 × ~1,700 bars > 10k). deep_score ~300 bars. Live stops: 1-minute IEX bars |
| **Alpaca Market Data, options** | See 2.4 | OPRA real-time **rejected (403) since ~2026-07-23/25**; downgraded to free `indicative` | Same Python bucket. The Node archive sleeps 350ms between underlyings | See 2.4 | See 2.4 | See 2.4 |
| **Yahoo / yfinance** (no key) | `yf.Ticker(t).info` and `.recommendations` in a 5s-timeout subprocess inside deep_score (`bot_engine.deep_score` l.938-996, 5-min in-memory `_yf_cache`). `analyze_ticker` (l.1930+): yfinance history fallback, `.info`, **`.options` + `.option_chain(exp)` for up to 8 expiries (+1 for GEX)** (l.1994-2107). `^VIX`, `^TNX`, `DX-Y.NYB` 5d history (`macro_data.get_macro_snapshot` l.96-139). `.calendar` earnings date (`position_sizing._fetch_earnings_date_with_status` l.637-690, 12h cache; drives `_earnings_scalar` and the options earnings guard `options_execution._check_earnings_guard` l.154). Backtests: `/v8/finance/chart` (`backtest_v2._yahoo_bars`, `bot_backtest.fetch`) | Unofficial and unkeyed. No SLA, no license for commercial use | None (timeouts only) | IV/bid/ask/OI per contract (analyze), fundamentals, VIX/TNX/DXY last close, next earnings date | In memory. Backtests: `.bt_cache/`, `CACHE_DIR/bt2_*.json` (24h) | 1y history fallback; backtest `years` param (default 3) |
| **Finnhub** (`FINNHUB_KEY`; comment calls it the "free API key", `finnhub_data.py:7`) | `/stock/insider-sentiment` and `/stock/recommendation` (deep_score `_fetch_finnhub` l.865-875 → `finnhub_data` l.139, 406). `/calendar/earnings` (`options_scanner._fetch_earnings_calendar_with_status` l.641-680 → CSP earnings gate). `/stock/earnings` (`ml_model_v2._fetch_earnings_surprises` l.297, ≤30 tickers per retrain). `/stock/peers` | Free tier implied | `finnhub_data._TokenBucket` **55/min** (l.62-100), **but only for calls through `_finnhub_get`**. `options_scanner` l.657 and `ml_model_v2` l.302 call `requests.get` directly and **bypass the bucket** | Insider MSPR, rec-trend counts, earnings dates, surprise sign | `/tmp/voltrade_alt_cache/*.json`, TTL 4-24h (l.147-483) | Insider 365d window. Calendar 10 days ahead |
| **SEC EDGAR** (UA `VolTradeAI research@voltradeai.com`) | `efts.sec.gov/LATEST/search-index` Form 4 hit counts (`intelligence.get_insider_activity` l.224-260, cached in `INSIDER_CACHE_PATH`; `alt_data.get_congressional_signal` l.364-392, 12h). `institutional_data.py` (data.sec.gov submissions, Archives) is used **only by `insights.py`** (site) | Free; fair-access ~10 req/s (not enforced in code) | None | Hit counts only (no parsed transactions in the scoring path) | `DATA_DIR/voltrade_insider_cache.json`, `/tmp` cache | 14-60 days |
| **FRED** | Keyless `fred.stlouisfed.org/graph/fredgraph.csv` for UNRATE, UMCSENT, ICSA, BAMLC0A0CM, MICH, T10Y2Y, FEDFUNDS (`alt_data.get_fred_macro` l.124-160, 6h cache) → `stress_index` (l.38-57). Separately, the keyed `api.stlouisfed.org` (`FRED_API_KEY`) in `server/fredMacro.ts` is archive only | Keyless CSV, API key for the datacore stream | None | Latest value per series | `/tmp` cache. The datacore archive has vintage `rt` (fredMacro.ts header) | 90 days |
| **Polygon** (`POLYGON_KEY` vs `POLYGON_API_KEY`) | Trading: `/v3/reference/tickers/{t}` → `share_class_shares_outstanding` (`alt_data.get_ftd_signal` l.495). Everything else is site-only (see 2.0) | **Unknown.** Every call used works on Polygon's free tier. **No Polygon options, quotes, trades, or aggregates-per-ticker endpoint exists anywhere in the code** | None ("not rate-limited" comment l.494 is an assumption) | Shares outstanding | `/tmp` cache 6h | n/a |
| **Wikimedia pageviews** | `alt_data.get_wiki_attention` l.89 | Free | None | Daily views | `/tmp` 4h | ~recent days |
| **GDELT DOC API** | `alt_data.get_geopolitical_risk` l.226 | Free | None | 5 article titles | `/tmp` 2h | n/a |
| **USPTO** | `developer.uspto.gov/ibd-api/...` (`alt_data.get_patent_activity` l.442). Endpoint liveness unverified | Free | None | Count | `/tmp` 24h | n/a |
| **Reddit / StockTwits / Google Trends / RSS** | Reddit RSS (`social_data.get_reddit_sentiment` l.112). pytrends (`social_data.get_google_trends` l.244, 6h). Reuters/CNBC/MarketWatch RSS (`social_data` l.301-303). In `analyze.get_sentiment` (l.878-950): StockTwits `api.stocktwits.com/api/2/streams/symbol`, `reddit.com/search.json`, pytrends | Unofficial, unkeyed | None | Counts and keyword sentiment | `/tmp` caches 1-6h. analyze: none | 7 days |
| **CFTC Socrata** | Python `cftc_cot.run_daily_update` (bot.ts Tier 3 l.6436-6452) and TS `cftcCot.ts`/`cftcTff.ts`. **Archive only.** bot.ts comment: "feeds nothing into scoring yet" | Public domain, keyless | ~1 call/20h self-guard | Disaggregated COT fields | `COT_ARCHIVE_PATH` (`storage_config.py`), datacore archive | Weekly |
| **SEC Form 3/4/5 bulk** | `sec_form4_bulk.run_update` (bot.ts Tier 3 l.6456-6475). Archive only, gate 1 | Free | 1 network pass per ~12h | Insider transactions | `FORM4_BULK_DIR` | Quarterly backfill |
| **Anthropic** | `server/analyst.ts` (console chat), not trading | Key-gated | — | — | — | — |

Present in the code but **not used by any trading or server path**: **NASDAQ/nasdaqtrader** (only `scripts/illiquid_universe_probe*.py` and `scripts/ats_resolver_gate0.py`, which are research scripts that the Dockerfile never ships per KB #44 in `bot.ts` ~l.3490; `market_calendar.py:11` is a source comment). **api.nasdaq.com** (`etf_data_sources.py:587`, the site ETF page). **Stooq** (only a legacy alias name, `etf_data_sources.py:946`; no HTTP). **Tradier, Tiingo, Alpha Vantage, IEX Cloud, Quandl**: zero hits. **EIA**: `server/gridDemand.ts`, `gridGeneration.ts` and `scripts/eia*` are map/research only. **CBOE** (`server/cboeVix.ts`, header: "no trading logic reads this yet"), **OCC** (`server/occVolume.ts`), **FINRA** (`finraShortVolume.ts`, `finraQuery.ts`) and **OpenFIGI** (`cusipResolver.ts`) are datacore archives only. **Databento**: no client code; the pilot was priced but never pulled (`wishlist.md` ~l.2413-2500).

### Side finding (data integrity, trading ML)

`ml_model_v2._fetch_earnings_surprises` (l.297-312) fetches each ticker's **latest** EPS surprise today. `_build_training_data` (l.842) then applies it as a constant feature `earnings_surprise` (FEATURE list l.165, l.597) to **every historical training row** of that ticker. That is lookahead: rows dated before the surprise carry information they could not have had.

## 2.2 Rate limiting: what actually exists

- `alpaca_rate_limiter.py`: token bucket at **180/min** (l.69), justified as "paid tier 200/min" (l.6-8). Alpaca's public plans are about 200/min on Basic and much higher on Algo Trader Plus (from outside knowledge, not verified here); the 180 figure matches the free tier.
- `install_global_throttle()` (l.134-171) monkey-patches `requests.get/post/delete` for any URL containing `alpaca.markets`. It is installed only when `bot_engine.py` (l.75) or `ml_model_v2.py` (l.55) is imported. **The bucket is per process:**
  - the daemon,
  - each `python3 -c` subprocess from bot.ts,
  - the `analyze.py` subprocess (which never installs it),
  - Node (no limiter on its Alpaca REST proxy, archive, or bars).

  All of these draw on the same account limit, each with its own budget or none. Calls made through `requests.Session` would also bypass the patch.
- Finnhub: 55/min bucket, bypassed by two direct callers (see the Finnhub row).
- The Node options archive sleeps 350ms between underlyings (`optionsChainArchive.ts:runDailyChainArchive` l.250).
- Polygon, SEC, Yahoo, Reddit, StockTwits, pytrends, GDELT and Wikimedia have **no limiter**, only caches and timeouts.

## 2.3 Map-layer and alt-data feeds: do they reach the trading engine?

None of the trading Python modules read the datacore archive or `/api/data` (grep of `bot_engine`, `ml_model_v2`, `analyze`, `options_*`, `csp_universe`, `position_sizing`, `stress_index` and the enrichment modules finds no `datacore_archive`, `/api/data/` or localhost reads). `server/bot.ts` imports datacore modules (l.22-37: `datacoreArchive`, `portDwell`, `shadowFleet`, `gnssIntegrity*`, `secMidas`, `fdicBanks`, `spaceWeather`, `chainIv`, `chainCspEntry`), but every use is inside the `/api/diag/:probe` switch (~l.3030-3480). That is research read-back, not scoring.

| Feed family (server module) | Feeds the trading engine? | Notes |
|---|---|---|
| ADS-B (`aircraftFastLane`, `globalSweep`, `trackedPlanes`, `flightPlans`, `coverageGapFill`, `openskyGlobal`: adsb.lol/adsb.fi/airplanes.live/OpenSky) | No (map/archive) | |
| AIS vessels (`aisFeed` digitraffic), `shadowFleet`, `portDwell` | No (map, diag probes) | |
| Satellites (`satellites` CelesTrak) | No | |
| Weather/hazards (`weatherGrid`/`owmTiles` OWM, `nwsAlerts`, `usgsQuakes`, `usgsVolcanoes`, `usgsWater`, `ndbcBuoys`, `spaceWeather`, `droughtMonitor`, `femaFlood`, `ambientRadiation`, `airQuality`, `nasaFirms`) | No | |
| Grid/energy (`gridDemand`/`gridGeneration` EIA, `euLoad`/`euGenerationMix`/`euDayAheadPrices` ENTSO-E, `epaCamd`, `nrcReactorStatus`) | No | |
| Finance datacore (`cboeVix`, `occVolume`, `finraShortVolume`, `finraQuery`, `cftcCot`/`cftcTff`, `fredMacro`, `edgarForm4`, `edgar13f`, `sec8kEarnings`, `secFtd`, `secMidas`, `treasuryDts`, `treasuryAuctions`, `dtccSwaps`, `usaSpending`, `fdicBanks`, `euMacro`) | No (archive + gate-locked research) | The Python `cftc_cot.py`/`sec_form4_bulk.py` archives are also gate 1 only |
| Other (`gdeltEvents`, `clinicalTrials`, `fdaEvents`, `nhtsaComplaints`, `censusImports`, `cropConditions`, `appStoreRankings`, `cbpBorderWait`, `trainsFeed`, `fmcsaOutOfService`, `superfund`, `waterViolators`, `wikiAttention`, FAA charts/procedures) | No | The Python enrichment modules hit **their own** Wikimedia/GDELT endpoints (2.1), separate from the datacore streams |
| `optionsChainArchive` | **No** (research only: `chain_iv30`/`chain_csp_entry` diag probes) | See 2.4 |

## 2.4 Options data

### What is fetched today (all from Alpaca `data.alpaca.markets`, except analyze.py)

| Code path | Endpoint and params | Fields used | Cadence | Stored? |
|---|---|---|---|---|
| `options_execution._fetch_option_chain` (l.566-717): CSP / spread / condor contract selection | `GET /v1beta1/options/snapshots/{ticker}`, `feed=options_feed()`, **`limit=100`, single page (no pagination)**, expiry `min_dte..50d`, strike band (puts 0.80-1.05× spot), optional `type` | `latestQuote` bp/ap/bs/as, `latestTrade.p` (mid fallback), `dailyBar.v` (volume), `openInterest` (**falls back to `max(bid_size, ask_size)×10` as a proxy stored under the key `open_interest`**, l.681-700), `impliedVolatility`, greeks delta/theta/gamma | Per trade attempt (Tier 1/2) | No (in memory) |
| `options_scanner._fetch_options_chain` (l.283-470) | Same endpoint, `options_feed()`, `limit=1000`, ≤5 pages, strikes ±30%, expiry window per setup | Same, plus a **moneyness-table delta proxy** when greeks.delta is missing (l.380-415) and the same OI proxy (l.440-443) | 180s in-memory cache (`_CHAIN_CACHE_TTL` l.281) | No |
| `options_scanner._fetch_iv_rank` (l.532-636) | Same endpoint, **hard-coded `feed=indicative`, `limit=100`, no strike filter**, 7-45 DTE | `impliedVolatility` of the nearest ATM contract within 5% of spot | Per candidate | No. **Risk:** with symbol-ordered pages and no strike filter, the first 100 snapshots may contain no ATM strike, so it silently falls back to HV rank (l.627-633) |
| `options_manager._get_option_snapshot` (l.182-225) | `GET /v1beta1/options/snapshots/{underlying}?symbols={occ}`, `options_feed()` | bid/ask/mid, delta/gamma/theta/vega, IV | Per position-management pass | `voltrade_options_state.json` holds position state, not market data. Unverified whether the per-underlying endpoint honors `symbols=`; if it ignores it, the target OCC may be missing from the default page. The root parse ("everything before the first digit", l.186-192) breaks on adjusted roots like `IONQ1` |
| `options_manager._attempt_roll` (l.353-410) | Same endpoint, 30-50 DTE, strike ±1, `limit=20` | Quotes | On roll | No |
| `vol_surface._fetch_options_chain` (l.187-240) → `build_surface` (l.829) | Same endpoint, `options_feed()`, `limit=200`, ≤10 pages (**2,000-contract cap, no strike filter**), SPY/QQQ/IWM/DIA capped at 30 DTE (l.184, 201) | `impliedVolatility`, greeks (l.898-899), quotes | `CACHE_TTL=3600` (l.35), in memory | No |
| `csp_universe` (L1/L2 ranking, 15-min TTL l.53-54) | Reuses `options_scanner._fetch_options_chain`, `_fetch_iv_rank` and `vol_surface.get_surface_score` | Derived scores | 15 min | `voltrade_csp_scores_cache.json` (overwritten, no history) |
| `bot_engine._run_convexity_overlay` (l.4636-4870) | Trading API `GET /v2/options/contracts` (QQQ puts 45-75 DTE, strikes 0.75-0.90× spot, limit 50) + **`GET /v1beta1/options/quotes/latest?symbols={occ}&feed=indicative`** | strike/expiry, bp/ap. If there is no quote, the price is **estimated as 0.7% of QQQ** (l.4861-4863) | Per overlay run | No |
| `server/optionsChainArchive.ts` (`bootChainArchive` l.290-322, wired at `routes.ts:2047`) | `GET /v1beta1/options/snapshots/{u}?feed=indicative&limit=1000`, ≤5 pages, **no server-side expiry/strike filters** (filtered client-side: ≤60 DTE, ±20% only when spot is known) | `bid, ask, bsz, asz, iv, delta, oi` + `feed:"indicative"` (`compactContract` l.106-117) | Meant to run once per trading day after 16:15 ET (but see below) | **YES, the only options data that is stored.** `DATA_DIR/datacore_archive/optionchains/YYYY-MM-DD.jsonl(.gz)` (`chainDir` l.119, `archiveChainRecords` l.124). Derived caches go to `optionchains/derived/{iv30,csp_entry}/` (`chainIv.ts`, `chainCspEntry.ts`) |
| `analyze.analyze_ticker` (l.1930-2140), run as a subprocess by deep_score | **yfinance** `.options` + `.option_chain(exp)` for ≤8 expiries (+1 for GEX) | strike, bid, ask, `impliedVolatility`, `openInterest` → `atm_iv`, VRP, skew, GEX, put/call OI, spreads (l.2028-2110, 1502-1523, 1735) | Every deep-scored ticker, every Tier 2 scan; no cache | No |
| `insights.py` (site) | yfinance `option_chain` (l.579-593) | — | On page view | No |

**Not fetched anywhere:**
- option **trades or tick history**: no `/v1beta1/options/trades` or `/v1beta1/options/bars`. `latestTrade` is read only as a mid fallback;
- historical option **quotes**;
- OCC open-interest files: `occVolume.ts` is cleared *volume* by origin, archived, not traded;
- CBOE option data: the put/call CSVs are frozen per the `cboeVix.ts` header;
- any **Polygon options endpoint**: no `/v3/snapshot/options` or `/v3/reference/options/contracts`.

The backtests have **no options leg** (`backtest_v2.py` l.20-22: "options/squeeze legs are NOT simulated").

### What is stored, and how deep

- **Archive start:** manifest `started: 2026-07-05` (`datacore/manifests/optionchains.json`).
- **Prod depth now:** `chain_iv30` probe reports **81 archived days, 2026-07-06 → 2026-10-27**, feed `indicative`.
- **SPY/QQQ/IWM** only since **2026-10-20/21** (`CHAIN_ALWAYS` added 2026-10-20, l.37-41; SPY rows begin 2026-10-21 in prod).
- **Everything else** in the trading engine's options path is fetched and thrown away.

### Defects found in the stored chain data (prod-verified)

1. **The day labels are shifted by one trading day.**
   - **Evidence:** every sampled archive day (2026-10-20, 21, 22, 23, 26, 27) has snapshot `t` ≈ **04:36-04:47 UTC, i.e. about 00:40 ET**, not after 16:15 ET (prod `archive?stream=optionchains&day=…`).
   - **Cause:** `shouldRunNow` (l.181-190) reads the hour from `Intl.DateTimeFormat(... hour12:false)` (l.141-143). On **Node 20** (the Dockerfile uses `node:20-slim`) that returns `"24"` at midnight. I verified this with `npx node@20`: it yields `["Wed"," ","24",":","46"]` for 2026-10-21T04:46Z, while Node 22 yields `"00"`. So at 00:xx ET, `mins` = 1440+ ≥ 975, the run fires, and `writeLastRunDay(etDay)` then blocks the real 16:15 run.
   - **Effect:** file `D` contains the **previous session's closing indicative quotes**. Monday's file holds Friday's close.
   - **Consumers affected:** any join of label `D` against the stock's day-`D` close (`chainIv.ts`, `chainCspEntry.ts`, `scripts/csp_manager_rule_sim.py`) is misaligned by one session.
   - **Also affected:** `server/liveness.ts` l.47-58 uses the same formatter. Harmless there, because midnight counts as "closed" either way.
2. **Heavy chains are truncated.**
   - Prod day 2026-10-26: **SPY has 4,151 rows across 15 expiries, all ending at 2026-12-04 (~39 DTE)**, even though the cap is 60 DTE.
   - The 2026-12-18 monthly is absent. That is consistent with the 5 × 1000 page cap being reached before the DTE filter applies.
   - SPY spot is null, so all strikes are kept (~258 contracts per expiry).
   - Note: the diag probe returns at most 5,000 rows, so I saw SPY plus part of QQQ only.
3. **Open interest is never present.** 0 of 5,000 sampled rows have `oi` (`iv` is present on 3,340/4,151 SPY rows, `delta` on 3,767/5,000). The indicative snapshot evidently does not return `openInterest`. The live code therefore runs its OI ≥ 50/100/300 liquidity gates (`options_execution` l.1700, 1799; `_is_liquid` l.719-760) on the **quote-size×10 proxy**, not on real OI.

### Do the current plans include options trades/quotes?

**What the repo and prod show:**
- **Alpaca:**
  - OPRA (real-time consolidated options) was **rejected with HTTP 403 "subscription does not permit querying OPRA data"** as of 2026-07-25 (`alpaca_feed.options_feed` docstring l.173-197; `wishlist.md` ~l.1961-1977; `experiments.md` ~l.78860).
  - Every options call now uses `options_feed()` (downgrades to `indicative` on 403) or hard-codes `indicative` (archive, IV-rank, convexity quote).
  - Prod archive rows are all `indicative`.
  - `wishlist.md` ~l.2430 notes "Alpaca's free [options] history starts Feb 2024".
- **Polygon:** **nothing in the repo proves any Polygon plan.** No code calls a Polygon options or quotes endpoint, so options entitlement there is untested. `wishlist.md` ~l.2443 lists "Polygon $29-199/mo (quotes only on upper tiers)" only as a purchase candidate.

**What cannot be proven:**
- whether the 403 still holds today. Neither `options_feed_status()` (l.230) nor `feed_status()` (l.141) is surfaced by any diag probe, and the `audit` probe shows no T2-FAIL entries;
- whether `ALPACA_OPTIONS_FEED` / `ALPACA_DATA_FEED` are forced in prod;
- the Alpaca account's subscription tier (only "SIP 403" and "OPRA 403" as of July);
- whether `indicative` includes OI on any underlying (sample: none);
- whether the Polygon key exists, under which env-var name, and at which tier.

**Bottom line:**
- No consolidated (NBBO) option quotes, no option trades, and no real open interest are fetched.
- The only stored options data is about 81 days of once-daily indicative snapshots, with one-day-shifted labels and truncated index chains.
- All historical options backtesting remains impossible. Databento was priced at ~$740 for 2016→present but never pulled.

## 3. Universe and scanner

### 3.1 Pipeline (stock path): `bot_engine.scan_market` → `_scan_market_inner`

| Stage | What happens | Code | Live count |
|---|---|---|---|
| 0. DD halt | `update_equity_peak()`. If the portfolio is halted, the scan returns no trades. | `_scan_market_inner` ~2815-2850 | – |
| 1. Universe | `_get_full_universe()`: all Alpaca `/v2/assets?status=active&asset_class=us_equity` that are tradable, contain no ".", do not end in "W", and are 5 characters or fewer. Cached 24h in `universe_cache.json`. Today's movers (`/v1beta1/screener/stocks/movers?top=50`) are appended. If the universe is empty, it falls back to most-actives top 100. | `_get_full_universe` 2471-2519; `_scan_market_inner` 2852-2883 | – |
| 2. Snapshots | Batches of 400 are fetched one after another from `/v2/stocks/snapshots?feed=data_feed()` and trimmed to `(c,o,v,prev_c)` from `dailyBar`/`prevDailyBar`. A symbol with `c<=0` is dropped. | `_parse_snapshot_batch` 2615-2658; loop 2899-2931 | **~12,156 symbols per scan** (audit `TIER2 "Scanned 12156 stocks…"`, 66 lines today). `data_feed()` is currently **delayed_sip** (15-min delay; audit `TIER3-ML-ERROR` text "[FEED] SIP entitlement rejected (HTTP 403) — downgrading to delayed_sip") |
| 3. Price/volume floor | `c < MIN_PRICE (5)` or `v < _min_vol` is dropped. `_min_vol = 100_000` from 9:30 to 9:59 ET and `MIN_VOLUME = 500_000` otherwise. `v` is **cumulative intraday volume** from the snapshot. | 2936-2968; constants `bot_engine.py` 208-209 | Not exposed (`filtered` lives only in the owner-only last-scan) |
| 4. Quick score | `quick_score = min(|chg|,15)*3 + min(v/1e6,5)*2 + {5 if |chg|>10, 2 if >5} − {30 if |chg|>50, 15 if >30}`. The result is sorted descending. It is **direction-agnostic**: big losers rank as high as big gainers. | 2969-3009 | – |
| 5. Memory guards | If RSS is 700MB or more, survival mode returns the quick-scan top 10 with no deep scoring. From 550MB it skips deep scoring; from 400MB it trims to 3. | 3044-3146 | `scanner` probe: `degraded:false`, `dataSourceErrors:{}` |
| 6. Deep score | The top **`VOLTRADE_DEEP_SCORE_CAP` = 5** (max 10, hard ceiling 15) are scored serially with a 35s total cap. Each goes through `deep_score(ticker, quick, min_score=adaptive MIN_SCORE)`. | 3098-3214 | 5 per scan (3 under trim) |
| 7. Tie-break | Groups within 3 points are re-ranked by log10(dollar_vol/spread proxy)*5 + 5 per factor ≥ 70 − 15 for a sector clash. **`current_positions` is referenced before assignment** (assigned at ~3336), so this always raises NameError, gets caught, and logs "tie-breaking failed". The tie-break never runs (code reading). | 3229-3310 vs 3336 | – |
| 8. Entry gates (in order) | (a) stock slots = adaptive `MAX_POSITIONS` − held. (b) already held. (c) regime-aware stop cooldown of 1.5–4h. (d) **BEAR/PANIC: no stock longs**. (e) `final_score < get_adaptive_params().MIN_SCORE` (63 BULL / 67 CAUTION / 75 BEAR,PANIC; NEUTRAL keeps 63 but has MAX_POSITIONS 0). (f) `check_sector_correlation` (MAX_SECTOR_POSITIONS from size tier). (g) **dollar volume < $50M** (hard-coded `_MIN_DOLLAR_VOL`). (h) `BLOCKED_TICKERS` ∪ {UBER}. (i) change > 50% goes to the overnight watchlist. (j) shorts are forced to buy. (k) quote spread > 0.5%. (l) `select_instrument`. (m) `calculate_position` (blocked or 0 shares). | 3409-3720 | Shadow-log decisions out of 20,000 deep-scored records (2026-08-11 to 10-27): taken 7,710, rejected_score 11,285, rejected_other (spread) 591, rejected_stock_slots 235, rejected_heat 163, masterkill 16 (`/api/diag/shadow`). "taken" is deep_score's provisional label, not an executed order. |
| 9. Node-side gate | `trade.score < state.minScoreThreshold` (diagnostics `get_auto_fix_params`, default **65**, raised to 75 on "increase_score_threshold") plus Node-side slot caps. | `server/bot.ts` ~5795; `diagnostics.py` 510-555 | Audit: "3 trade candidates" on every scan today |

**Regime gating means stock longs are possible only in BULL or CAUTION.**
- `get_adaptive_params` sets `MAX_POSITIONS = 0` for PANIC, BEAR and NEUTRAL (`system_config.py` 457-507), and a size tier never raises a zero (`system_config.py` 567-572).
- The scan computes its regime from `get_market_regime` (delegated to `regime_util.classify_regime_5level`) using the macro snapshot (`bot_engine.py` 3455-3472).

**Config drift in the floors.**
- `bot_engine` uses its own module constants `MIN_VOLUME=500000` and `MIN_PRICE=5` (208-209). It does not read `system_config.BASE_CONFIG["MIN_VOLUME"/"MIN_PRICE"]` (66-67).
- At the open, system_config says **1,000,000** (`OPEN_HOUR_MIN_VOLUME`, line 68, applied in `get_adaptive_params` ~518). The scanner instead *lowers* the floor to **100,000** (2946). The two are opposite policies.
- The $50M dollar-volume floor and the 0.5% spread cap are hard-coded in `_scan_market_inner`. They do not live in system_config.

**Polygon grouped daily is not on any live path.**
- `_fetch_grouped_day` exists only in the legacy `ml_model.py` (~203-243), and no `.py` or `.ts` file imports `ml_model` (grep).
- `POLYGON_KEY` is read in `bot_engine.py` 190 but is not used by the scanner.

### 3.2 Scan frequency (`server/bot.ts`)

`getTier2Interval()` (~8183-8200) sets the delay by ET time:

| ET window | Delay |
|---|---|
| 9:30–10:30 | 60s |
| 10:30–12:00 | 3 min |
| 12:00–14:00 | 7 min |
| 14:00–16:00 | 2 min |
| 16:00–20:00 | 15 min |
| 04:00–09:30 | 5 min |
| Otherwise (overnight) | 30 min |

These values are duplicated in `system_config.py` 39-45 (`TIER2_*_MS`), but the Node code does not read them.

How scheduling works:
- `scheduleTier2()` (~8223-8295) re-arms the timer **after** a scan finishes, in the `finally` block (`armTier2Timer(delay)`). The real period is therefore interval + scan duration. Failures back off exponentially from 60s up to 600s.
- Live: power-hour scans were about 5.5–6.5 min apart (18:05Z to 19:09Z). The nominal interval is 2 min.
- Overnight scans happen only inside a "clock open or 4–20 ET" window (~8243-8248). Outside that window the timer is just re-armed.
- The scan is dispatched through daemon RPC `run_full_scan` → `bot_engine.scan_market` (`voltrade_daemon.py` ~324; `bot.ts` ~4840).

### 3.3 Options universes

**`csp_universe` (feeds T1)**

Layer 1 is `_layer1_hard_gates` (~203-326), with a 15-min cache:
- Drops non-plain symbols, `CSP_BLOCKED_TICKERS` (leveraged seed list, T-bill ETFs, commodity ETFs) and `leveraged_symbols()`.
- Keeps names with price ≥ $10 and ≤ an affordability cap of equity × 0.25 / 95 (default $250), volume ≥ 1M, and dollar volume ≥ $20M.
- Always adds `CSP_ANCHOR_TICKERS` (17 names).

Layer 2 is `_layer2_score` (~618-820):
- Candidates are sorted by dollar volume. Only the **top 150** get full factor scoring (`deep_score_limit=150`, with a 45s prefetch budget). The rest get baseline values: IVR 40, VRP 40, skew 50.
- Composite score = 0.25·IVR + 0.20·VRP + 0.15·liquidity + 0.10·put skew + 0.10·earnings + 0.10·historical + 0.10·stability.
- A name is dropped if its earnings score is ≤ 10 or if the earnings lookup failed.
- `get_top_csp_candidates(n=200)` (~842).

**`tiered_strategy` T1**
- `_get_t1_universe()` (148-176) returns the dynamic top 200 when it has at least 17 names. Otherwise it uses `T1_TICKERS_FALLBACK` (17 large caps, line 142).
- `tier1_csp_core` (466-601) walks the universe in **rank order**. It skips names already held (by OCC underlying) and names over the per-position affordability limit. It emits `SELL_CSP` until the options slots are used up and returns **at most 3 per scan**.
- `T1_MIN_IV_RANK = 40` is only *passed in metadata* (~584). It is not applied as a filter here.

**Regime context bug**
- The tier engine receives `regime=_regime_ctx.get(...) if '_regime_ctx' in locals() else "NEUTRAL"` (`bot_engine.py` ~4140). `_regime_ctx` is a local of `deep_score`, not of `_scan_market_inner` (grep: there is no other assignment after line 2700), so **TierContext.regime is always "NEUTRAL"** (code reading).
- T1's own caps use `get_regime_caps(vxx_ratio, spy_vs_ma50)`, so the impact is limited to consumers of `ctx.regime`.

### 3.4 Ranking logic in `deep_score` (`bot_engine.py` 807-1958)

1. **Rules composite** (~1259-1266): momentum·0.25 + mean reversion·0.20 + VRP·0.25 + squeeze·0.15 + volume·0.15, plus the vol premium bonus. Additive adjustments follow:
   - macro (VIX regime, sector ETF day %, risk on/off)
   - news (`get_news_sentiment`)
   - ADX (+6/+3/−4)
   - OBV (±5)
   - intel (±15, plus Finnhub MSPR ±5 and analyst +3)
   - alt (±20)
   - social (±15)
   - yfinance fundamentals (analyst rating, target upside, forward P/E, EV/EBITDA)
   - analyze.py recommendation (+8/+6)
   - contrarian flags (+10/+8/+4)
   - relative strength > 5 (+6)

   The result is clamped to 0–100 and stored as `rules_only_score` (~1476).
2. **ML blend** (~1805-1836). It applies only if `_ml_result_is_model()`, which excludes `fallback_rules`.
   - If |ml−50| < 8, the ML value is set to 50.
   - `final = rules·rule_w + ml·ml_w`. The default weights are 0.6/0.4, read from `voltrade_blend_tracker.json`.
   - That file is written by `diagnostics.run_diagnostics` (~476-502): ml_w = 0.4, adjusted by recent win rate, and floored at 0.2.
   - Live `recent_win_rate_20 = 20` (`/api/diag/ml`) gives about **0.72 rules / 0.28 ML**, assuming diagnostics reads the same perf block.
3. **Side**: if the score is > 65 and VRP > 5, the trade becomes SELL OPTIONS. If RSI > 70 it becomes SELL. Otherwise BUY (~1853-1875). Stock shorts are disabled downstream (~3605).
4. **Known result from the shadow log** (open_questions.md ~23077, 2026-10-26): the score ranking is **inverted**.
   - The highest-score quintile wins 42.5% at +5d; the lowest quintile wins 56.2%.
   - Live `/api/diag/shadow` agrees: `taken` wins 41.1% at +5d versus 49.5% for `rejected_score`.

---

## 4. Features

### 4.1 Model features (`ml_model_v2.FEATURE_COLS`, 140-186, asserted at 34)

How to read the tables:
- **Train def** is `_compute_features` (364-615), called from `_build_training_data` (876-885) with `cross_sec_rank` overwritten at ~896.
- **Serve def** is the `ml_features` dict in `deep_score` (~1731-1783). The same dict is also persisted as shadow-record features (`log_candidate`, ~1909).
- **Live?** describes the value that actually reaches `ml_score()` in production.

A key finding (section 5.6): **the current production model is trained only on shadow rows, which carry the *serve* definitions.** As a result, train/serve skew against the generic definitions currently matters only for the `_rule_based_fallback` path and for any future run where shadow rows fall below 10k.

#### Price / technical (12)

| Feature | Train def (lookback, source: Alpaca daily bars, adjustment=all) | Serve def (source) | Live? |
|---|---|---|---|
| momentum_1m | (c−c[−21])/c[−21]·100 | `_deep_closes[-1]/[-21]−1` (20-bar), Alpaca 400d bars (~1093-1106) | Used, about aligned |
| momentum_3m | 63-bar return | `[-1]/[-63]` (62-bar) (~1725-1728) | Used |
| rsi_14 | Simple-average RSI over `closes[idx-15:idx]` (**excludes today**) | `analyze.compute_rsi` via `vol_metrics['rsi_14']` (analyze.py ~2119/2144; analyze_ticker runs the full path because `--mode=scan` is ignored by `__main__` ~2642-2663) | Used, def differs |
| price_vs_52w_high | (c − max high of 252 bars)/max·100 | `quick_result.get("high_52w", price)`. **`high_52w` is never set** on quick results (only ref is ~1508) | **Always 0 live** |
| vwap_position | 1 if close > open | 1 if snapshot c > o (`above_vwap`) | Used, aligned (delayed feed) |
| adx | Simplified \|ΣDM+ − ΣDM−\|/(ΣDM+ + ΣDM−)·100, 14 bars | Wilder-smoothed **DX** (not ADX) over about 300 bars (~1042-1056) | Used, def differs |
| range_pct | (h−l)/c·100 of bar idx | `quick_result["range_pct"]`, hard-coded **0** (~2990) | **Always 0 live** |
| change_pct_today | Full-day close-to-close | Intraday partial (snapshot c vs prev close) | Used, def differs (partial day) |
| above_ma10 | c > mean of prior 10 closes | `_deep_closes[-1] > mean(last 10)` (includes today) | Used |
| trend_strength | min(\|chg\|/ewma_vol, 5) | min(\|chg\|/2, 5), because ewma_vol is constant 2 | Used, distorted |
| frac_diff_price | `_frac_diff(closes[idx-20:idx])` (d=0.4, 20-weight FFD) | `_frac_diff(_deep_closes)` (last 20) | Used, aligned |
| idiosyncratic_ret | 21d stock ret − β·SPY 21d ret, β = 1.2 if vol ratio > 2 else 0.8 | Today's chg% − `macro["spy_change_pct"]`. **macro_data never emits `spy_change_pct`** (grep), so the feature equals change_pct_today | Used, a different quantity |

#### Volume (1)

| Feature | Train | Serve | Live? |
|---|---|---|---|
| volume_ratio | Today's volume / mean of prior 20 days (cap 10). Rows with volume < 300k are dropped (386). | analyze `volume_ratio_5d` = **5-day avg / 90-day avg** (analyze.py 251-254) | Used, def differs |

#### Volatility (7)

| Feature | Train | Serve | Live? |
|---|---|---|---|
| ewma_vol | std of prior 20 daily returns ·100 (not EWMA) | `vol_metrics.get("ewma_vol",2)`. analyze emits `ewma_rv`, never `ewma_vol` | **Always 2 live** |
| atr_pct | Mean TR of prior 14 bars / c ·100 | Wilder ATR14 / price (~1533-1535) | Used, about aligned |
| vrp | (vxx_ratio−1)·50, clipped ±20 (**market-wide**) | analyze `vrp` = ATM IV − RV20 (**per stock**, yfinance chain) | Used, a different quantity |
| iv_rank_proxy | VXX 52-week rank 0–100 (market-wide) | `_compute_stock_iv_rank` = per-stock 30d HV rank (~20-33) | Used, a different quantity |
| iv_rank_stock | iv_rank_proxy · min(vol_ratio,3)/2 | Same `_compute_stock_iv_rank`, so it duplicates iv_rank_proxy | Used, duplicate |
| vol_of_vol | std of 10 VXX daily returns ·100 | Same, from `macro.vxx_closes` (~1544-1571) | Used, aligned |
| put_call_proxy | −1 if vrp > 8, +1 if vrp < −5, else 0 (from the market vrp) | Same rule on the **per-stock** vrp | Used, semantics differ |

#### Regime / macro (9)

| Feature | Train | Serve | Live? |
|---|---|---|---|
| vxx_ratio | VXX / prior 30d avg (date-aligned via `_aligned`) | `regime_inputs(macro)` (regime_inputs.py). **Before 2026-10-16 it was pinned at 1.0 on 20,000 of 20,000 shadow records** (module docstring) | Used, aligned since 10-16 |
| spy_vs_ma50 | SPY / prior 50d MA | regime_inputs | Used |
| markov_state | Mean of 5 SPY returns vs ±0.2% → 0/1/2 | regime_inputs (same formula since 10-16; before that the Markov module was fed an empty list) | Used |
| regime_score | Closed form of spy_vs_ma50 and vxx_ratio, 0–100 | regime_inputs | Used |
| sector_momentum | Stock 5d − SPY 5d | Sector-ETF **1-day** % for `SECTOR_MAP.get(ticker,"Other")` (a static map; "Other" gives 0) (~1224-1226, 1751) | Used, a different quantity |
| cross_sec_rank | Percentile of today's change across the training universe on the same date (~889-896) | Percentile vs `most-actives?top=100` items' `"change"` field (~1583-1602). Code reading: Alpaca most-actives items are documented as symbol/volume/trade_count only. If so, the list is empty and the value is **0.5 always** (not verified live, no Alpaca key) | Probably constant |
| credit_spread | TLT 21d ret − HYG 21d ret | Same, fetched with a 60s cache (~1660-1686) | Used, aligned |
| market_breadth | % of training tickers above their 50d MA on that date (~816-833) | `macro["market_breadth_ma50"]` (**never emitted**, grep), so it falls back to 0.5 + (spy_vs_ma50−1)·6 (~1699-1712) | Used, a proxy |
| put_call_ratio | VXX close / 20 | `macro.vxx_latest` / 20 | Used, aligned (a level feature on a decaying ETN, non-stationary) |

#### Fundamentals / news / sentiment / events (5)

| Feature | Train | Serve | Live? |
|---|---|---|---|
| earnings_surprise | Finnhub **latest** surprise sign for the first 30 tickers, applied to **every historical row** of that ticker (`_fetch_earnings_surprises` 297-309; `_train_model_impl` ~1327). This is a **lookahead leak**. | `intel.get("earnings_surprise")`. intelligence.py never emits it (grep) | **Always 0 live** |
| intel_score | 0.4·news + 0.3·insider + 0.3·earn, where news and insider are fixed at 0 (`_build_training_data` 880-882) | Hard-coded 0.0 (~1772) | **Dead (constant)** |
| news_sentiment | 0.0 (`_fetch_historical_news_sentiment` exists at 315 but is never called) | 0.0 | **Dead** |
| insider_signal | 0.0 (`_fetch_insider_signal` 348 is never called) | 0.0 | **Dead** |
| days_to_earnings | 0.5 constant (555) | 0.5 constant (~1779). The computed value at ~1639-1655 is discarded. | **Dead** |

#### Alternative data / map layers

**None of the 34 ML features comes from datacore, map layers, or any EDGE-DOCTRINE pipeline.** grep finds no `datacore` or `/api/v1` reference in `bot_engine.py` or `ml_model_v2.py`.

Summary of what is live:
- **Constant or dead at serve: 9 of 34.** These are price_vs_52w_high, range_pct, ewma_vol, earnings_surprise, intel_score, news_sentiment, insider_signal, days_to_earnings, and probably cross_sec_rank. trend_strength is degraded too.
- **Pure functions of the date: 11 of 34.** These are vxx_ratio, spy_vs_ma50, markov_state, regime_score, iv_rank_proxy (train), vrp (train), put_call_proxy (train), vol_of_vol, credit_spread, market_breadth and put_call_ratio. This matters for 5.5.

### 4.2 `deep_score` rule inputs (not model features)

| Group | Input | Definition / source | Effect on score |
|---|---|---|---|
| Price/technical | Momentum (Jegadeesh-Titman 12-1) | 252-bar return − 21-bar return from Alpaca 400d bars. With fewer than 252 bars it is 1m·0.5 (~1089-1120). Passed to `strategies.momentum.score` | Weight 0.25 |
| Price/technical | Mean reversion | RSI (analyze), real 5-day change, volume_ratio_5d → `strategies.mean_reversion.score`. Skipped if RSI is None (~1123-1142) | Weight 0.20 |
| Price/technical | ADX | Wilder DX of the last bar (~1031-1056) | +6 (>40), +3 (>25), −4 (<15) |
| Price/technical | Relative strength | analyze `edge_factors.relative_strength` | +6 if > 5 |
| Volume | Volume score | volume_ratio_5d: >3→90, >2→75, >1.5→60, >1→50, else 35 | Weight 0.15 |
| Volume | OBV | 5-day vs prior 5-day OBV average, ±5% | ±5 |
| Volatility | VRP score | analyze ATM IV − RV20: >8→85, >5→70, >0→55, <−3→65, else 45 | Weight 0.25 |
| Volatility | EWMA vs RV20 | analyze `ewma_rv − rv20` | +8 / +5 |
| Squeeze/short | Squeeze | analyze edge `short_float_pct`, `days_to_cover`, sentiment score, reddit buzz → `strategies.squeeze.score` | Weight 0.15 |
| Macro | VIX regime, sector ETF day %, risk on/off | `macro_data.get_macro_snapshot` (~1213-1241) | ±5 / ±8 / −5,+3 |
| News/sentiment | Headline sentiment | `macro_data.get_news_sentiment` (`sentiment_score`) | +10/+5/−6/−12 |
| News/sentiment | Intel | `intelligence.get_full_intelligence` (`intel_score` = classify_news score ± insider direction ± earnings pattern), trap warning | Clamp ±15, trap ≤ −10, sell-the-news −5 |
| News/sentiment | Finnhub | Insider MSPR (±30 → ±5), recommendation consensus "buy" (+3). Applied only inside `if intel:` | ±5, +3 |
| News/sentiment | Social | `social_data.get_social_intelligence.combined_signal` = 0.35·Trends + 0.25·Reddit + 0.25·news + 0.15·Reddit direction, clamp ±15. **Google Trends is stubbed to signal 0** (social_data.py ~429). Halved at low confidence. | ±15 |
| News/sentiment | Contrarian flag, sentiment score | analyze `sentiment` | +10 / +8 / +4 |
| Fundamentals | yfinance (5s subprocess, 5-min cache) | Recommendation key, target-mean upside, forward P/E, EV/EBITDA (~938-996, 1403-1446) | About −18 to +16 |
| Fundamentals | analyze recommendation | `recommendation.action` | +8 / +6 |
| Alternative data | `alt_data.get_alt_data_score` | Sum of Wikipedia attention, short interest, congressional/Form-4, patents, FTD signals, plus geopolitical `risk_score` and FRED flags (−5/−3/−8) (alt_data.py 517-556) | Clamp ±20 |
| ML | `ml_score` | See section 5 | Blend at 0.28–0.40 |

---

## 5. Model

### 5.1 Learner (`_train_single_lgbm`, 1197-1248)

**Preprocessing:** StandardScaler, fit on the train split.

**LightGBM settings:**

| Parameter | Value |
|---|---|
| objective | binary |
| metric | binary_logloss |
| learning_rate | 0.04 |
| num_leaves | 24 |
| min_child_samples | 10 |
| feature_fraction | 0.75 |
| bagging_fraction | 0.75 |
| bagging_freq | 5 |
| scale_pos_weight | max(1, n_neg/n_pos) |
| min_gain_to_split | 0.01 |
| num_boost_round | 300 |
| early stopping | `early_stopping(30)` with **`valid_sets=[dtest]` (the test split)** |

**Fallback.** If the p90−p10 spread of the test probabilities is below 0.08, the model is replaced with `GradientBoostingClassifier(150, depth 4, lr 0.08, subsample 0.8, min_samples_leaf 8)`. The fallback is chosen by looking at test-set output.

**Reported metric:** accuracy at a 0.5 threshold only. There is no AUC, logloss or Brier score.

**Architecture** (`_train_model_impl` 1287-1566):
- One global model, plus bull/bear/neutral models wherever a regime has ≥ 50 train and ≥ 10 test rows.
- An isotonic calibrator is fitted on the pooled walk-forward OOS probabilities (~1454-1464).
- Inference (`ml_score` 1573-1686): the probability is clamped to 0.10–0.90 and calibrated. The rule fallback applies if the model file is older than `ML_MAX_AGE_HOURS` (config **24**).

**Other model code that does not run live:**
- `meta_score` and `train_meta_model` have **no callers** (grep).
- The options model is trained only by `bot_engine.py train` (~5898). `options_ml_score` is called by options_scanner (~834+).

### 5.2 Labels

Generic rows use `_triple_barrier_label` (622-658, with defaults `pt_mult=1.5, sl_mult=1.0, max_days=5`):

- **Entry:** `close[idx]`.
- **Barriers:**
  - PT = +max(0.5%, 1.5·ATR%)
  - SL = −max(0.3%, 1.0·ATR%)
  - Vertical barrier: bars idx+1 to idx+5.
- **Touch test:** uses high and low. **PT is checked before SL on the same bar**, so a double touch counts as a win. This is an optimistic bias.
- **Timeout relabel** (902-914): at idx+5, pnl > +0.3·ATR% gives +1, pnl < −0.3·ATR% gives −1, otherwise 0.
- **Binarization** (917): y = 1 only for +1. Both 0 and −1 become 0.

The training mix uses three more label definitions:

| Source | Label rule | Code |
|---|---|---|
| Feedback | y = 1 if pnl_pct > 1.5·entry ATR%. A None pnl becomes 0, which labels it 0. | 1134-1190 |
| Shadow | +2% target / −4% stop within **10** trading days (double touch counts as 0). On timeout, y = 1 if close ≥ +2%. | `shadow_portfolio._label_from_path` 553-620; `load_shadow_data(horizon_days=10, weight 0.3)` 841-917 |

**Disagreement with live exits** (KNOWN BROKEN #48, open_questions.md ~7460).

The configured exits in system_config (268-274: `ATR_STOP_MULTIPLIER` 1.5, `MIN/MAX_STOP_PCT` 2/6, `ATR_TP_MULTIPLIER` 4, `MIN/MAX_TP_PCT` 12/25, `TIME_STOP_DAYS` 10, plus regime overrides at 457-492) have **no reader** outside system_config. grep finds only comments, and the constants are mirrored in `live_exit_policy.py` and `shared/positionLevels.ts:112`.

The live exits are in `server/bot.ts checkPositionOnTick`, mirrored in `live_exit_policy.py` 61-71:
- Phase-1 stop: clamp(2·ATR%, 1.5–8%).
- R = 2·ATR%.
- Scale out a third at 1R, and a third at 3R / 2R / 1.5R depending on regime.
- Trailing stops after that.
- TP = clamp(3·ATR%, 4–15%), only while phase < 3 and both scale-outs are done.
- Time stop: ≥ 7 **calendar** days and |pnl| < 2%.

Label against live, point by point:

| Aspect | Generic label | Live exit |
|---|---|---|
| Stop | 1·ATR | 2·ATR, floored at 1.5% |
| Profit point | 1.5·ATR | First profit event at 2·ATR (1R), a partial exit |
| Horizon | 5 trading days, hard | Positions with \|pnl\| ≥ 2% are never time-stopped |
| Double touch | Win | – |

The shadow label (+2/−4, 10d) and the feedback label (> 1.5 ATR) are two further definitions. The model therefore learns a target that no live exit implements.

### 5.3 Validation splits

**`_walk_forward_cv(X, y, n_splits=5, embargo_days=5, n_tickers_approx=_n_tickers)`** (711-740):
- `fold = n//6`.
- Fold k trains on rows `[0, k·fold)` and tests on rows `[k·fold + 5·n_tickers, (k+1)·fold)`.
- A fold is skipped if train < max(500, fold).
- `_n_tickers = len(ticker_list)` (~1416). This is about 74: most-actives top 50 plus 30 core names, minus 6 ETFs.

**`_purged_train_test_split`** (665-708, with the call at 1467):
- Test = the last 20% of rows.
- Train = rows before `test_start − min(5·n_tickers, n//5)`.
- Overlapping label windows are not purged.
- The production model trains on this split, and its early stopping uses the holdout.

### 5.4 Feature assembly (`_build_training_data` 747-936) and data blend

**Bars** (`_fetch_training_bars` 206-294):
- Universe: today's `most-actives?top=50` plus 30 hard-coded core names, capped at 100 in fast mode. Choosing names by today's activity is survivorship-biased.
- Window: `fast_mode` (the only automatic path, `ml_retrain_safe.py` → `train_model(fast_mode=True)`) uses `days = min(ML_LOOKBACK_DAYS=60, 730)`, then `start = now − max(days, 365)`. That is **365 days**, not the 60 the config says.
- Feed: `bars_feed()`.

**Reference series.** VXX, SPY, TLT and HYG are aligned to each stock's dates (`_aligned` 847-874). Breadth and cross-sectional rank are computed per calendar date (794-833, 889-896).

**Row order.** Rows are emitted **ticker by ticker** (outer loop `for ticker, bars_raw in all_bars.items()` at 835, inner loop `for idx in range(25, len(bars)-6)` at 876).

**Blend order** (1344-1385): generic, then feedback, then shadow.
- **Feedback** comes from `_load_trade_feedback` (1085-1131). It needs `entry_features`. `code_version < MIN_FEEDBACK_VERSION "1.0.33"` gets `_legacy_weight 0.4`. Weight = 3·2^(−age/30)·legacy. It is capped at 15% of the generic rows and needs at least 15 records.
- **Shadow** has weight 0.3·2^(−age/30), needs at least 50 labelled records, and every shadow row gets regime "neutral".
- **Fast mode then keeps the last 10,000 rows** (1404-1410).
- `_config_similarity_weight` (979) has no caller.

### 5.5 Verdict on the suspected leak: CONFIRMED for generic data

The code evidence:
1. Rows are grouped by ticker (835/876 above).
2. Nothing in `_build_training_data` or `_train_model_impl` (747-1566) sorts or reorders rows by date. The only reordering is the tail slice `X_all[-keep:]` at 1406, which keeps the order.
3. `_walk_forward_cv` (726-736) and `_purged_train_test_split` (691-707) split purely by **row index**.

So each walk-forward test fold, and the 20% holdout, is a **block of whole tickers covering the same calendar window** as the training rows. It is not a later time period. The "embargo" of 5·n_tickers rows (about 370) removes roughly 1.7 tickers' worth of rows. It is not 5 days.

Why the reported metrics are optimistic:
- 11 of the 34 features are pure functions of the date (4.1). Together they fingerprint each day.
- Triple-barrier outcomes across stocks over the same 5-day window are strongly correlated through market moves.
- So the model can learn "these dates went on to hit PT" from the training tickers and get credit for applying it to the test tickers on those same dates. Training labels also cover the test rows' future windows.
- Early stopping, and the LGB-vs-GB choice, are both made on the test fold (1225-1236), which adds selection bias.
- The Finnhub earnings_surprise feature is a further lookahead (4.1).

**What the reported numbers mean.** `wf_accuracy` (the bundle `accuracy`, ~1517), `regime_accs`, `wf_fold_accs`, and the isotonic calibrator fitted on those "OOS" probabilities are **contemporaneous cross-sectional fit, not out-of-time skill**. They should be treated as upper bounds with no evidential value for live trading. The calibrator learns a mapping from leaked probabilities, so the served `win_probability` is miscalibrated in the direction of overconfidence.

### 5.6 What production actually trains on today

There is an additional finding beyond the suspected leak: shadow rows now swamp the 10k tail.

- `/api/diag/shadow` reports **11,948 labelled +10d records** (5,984 wins and 5,964 losses, a 50.1% base rate).
- `load_shadow_data` appends them **after** the roughly 15–16k generic rows and the feedback rows (1372-1383).
- Fast mode keeps `X_all[-10000:]` (1404-1410).

So, as of 2026-10-27, **all 10,000 training rows are shadow records**. Generic triple-barrier data and real-trade feedback are discarded entirely. This agrees with `/api/ml/status` `last_samples: 10000`, though that file can be stale (see 5.7).

Consequences:

1. **Labels** are the shadow +2%/−4%/10-day definition only. The bundle still claims `label_method: triple_barrier…` (~1524).
2. **Rows are chronological**, because shadow records are appended in `log_candidate` (~343-345). The CV is therefore roughly out-of-time, so the ticker-major leak in 5.5 is not what drives today's metric. But:
   - The embargo is about 370 rows, roughly 1.4 days at ~260 records/day. The labels span 10 trading days, so label windows straddle every fold boundary.
   - The same ticker is re-scored many times per day with near-identical features and an identical label. Those near-duplicates leak across the boundaries.
   - Early stopping still runs on the test fold.
   - The metric is still optimistic, though less so than in 5.5.
3. **Regime models.** Every row is regime "neutral", so only the neutral model can train. Bull and bear models never exist. `regime_accs` describes only neutral.
4. **Feature distributions shift mid-sample.**
   - Rows from 08-11 to 10-16 carry `vxx_ratio = 1.0` and the old Markov inputs.
   - Later rows carry the real values.
   - The 9 constant features carry no information.
   - Because the shadow features are the serve-time ones, the skew in 4.1 does *not* affect this model. It does affect any run where generic rows dominate (fewer than 10k shadow rows, or `fast_mode=False`).
5. **Sample weights.** The walk-forward folds are unweighted. The production fit uses `sample_weights[:len(X_tr)]` (1474), which are aligned because there is no shuffle.

### 5.7 Retraining cadence and latest metrics

**Retrain paths**

- **Tier 3 hourly** (`bot.ts` `tier3Strategic` ~6263-6360, started by `setInterval(…,3600000)` ~8300 and once 30s after boot ~8332):
  - It retrains only if the model file is more than 86,400s old (~6267-6273).
  - It runs `execPythonSerialized("python3 ml_retrain_safe.py", {timeout: 900000})` (~6293).
  - `execPythonSerialized` caps the timeout at 90s (`Math.min(opts.timeout, 90000)`, ~252). The command string is short and contains no `import numpy`, so `isTrivialPython()` (~241-246) treats it as trivial. **It is then SIGKILLed after 10s** (~255).
  - Live confirmation: `TIER3 "ML model stale…triggering retrain"` at 13:19:58Z, then `TIER3-ML-ERROR "(code=? signal=SIGKILL)"` at **13:20:08Z**.
  - Even so, `/api/diag/ml` and `/api/ml/status` show `model_age_hours 5.8` at 19:10Z, so the file was written around 13:22Z. That is consistent with the Python child surviving the shell kill as an orphan and finishing training. This is inferred, not proven. Its JSON result is never audited.
- **4am ET daily retrain** (`bot.ts` ~6760, `Math.floor(etHour) === 4`) is **unreachable**. `runOvernightResearch` is called only when `etHour >= 20 || etHour < 4` (~5703). Live audit has **zero** `RETRAIN` or `RETRAIN-ERROR` entries.
- **Manual `POST /api/ml/retrain`** (routes.ts ~7365) is the only writer of `ml_status.json`.
- **`bot_engine.py train`** is opt-in only (`_inline_train_allowed`, ~331).
- **Stale window.** `ML_MAX_AGE_HOURS = 24` matches the retrain trigger. Between 24h and the next hourly check, `ml_score` returns `fallback_rules`, which is excluded from the blend, so scoring is rules-only.

**Latest recorded metrics and where they live**

| Location | Content | Readable? |
|---|---|---|
| `/data/voltrade/voltrade_ml_v2.pkl` (`MODEL_PATH` 92) | Bundle with `accuracy` (walk-forward mean), `regime_accs`, `samples`, `data_source`, `feature_importance`, `timestamp` (~1512-1528) | Not reachable from this session |
| `/data/voltrade/voltrade_ml_calibrator.pkl` | Isotonic calibrator | Not reachable |
| `/data/voltrade/voltrade_last_retrain.json` | Full result of `ml_retrain_safe.py` (`wf_fold_accs`, etc.). Nothing reads it (`ml_retrain_safe.py` ~95-108). | Not reachable |
| `/data/voltrade/ml_status.json` → public `GET /api/ml/status` | Live: `last_accuracy: 58`, `last_samples: 10000`, `last_status: "trained"`, `last_train_time: null` | Read live. This is from the last **manual** retrain and its date is unknown, so it is **not necessarily the current bundle**. |
| `GET /api/diag/ml` | `model_exists: true`, `model_age_hours: 5.8`, `feedback_count: 1724` (1,326 seeded, 398 live), `live_entry_features_count: 56`, `live_performance` (36 trades, **win rate 27.8%**, avg win +9.08, avg loss −5.16, `recent_win_rate_20: 20`), `live_pnl_null_count: 362` | Read live. It carries no model accuracy. |
| `ml_toggle.json` "enabled" | Shown by `/api/ml/status` | **Not consulted by `deep_score` or `ml_score`** (grep), so it is cosmetic |

**Interpretation.** About 58% accuracy against a roughly 50% shadow base rate, measured with the leaks in 5.5 and 5.6, is not evidence of skill. The live picture is a 27.8% win rate on 36 closed trades and an inverted score-to-outcome relationship in the shadow log (3.4).

### 5.8 Prior art

`research/bot_cockpit_audit.md` (~440-520) already documents:
- the ticker-major split (its V15),
- the 10s retrain kill (V01),
- the unreachable 4am retrain (V02),
- the three-label mix.

What this audit adds:
1. Shadow rows now exceed the 10k fast-mode tail, so production trains on shadow rows only (5.6).
2. Four serve-time features are hard-wired to constants: `high_52w`, `range_pct`, `ewma_vol`, `spy_change_pct` (4.1).
3. The tie-break block never runs because of a NameError (3.1, stage 7).
4. `TierContext.regime` is always NEUTRAL (3.3).
5. The scanner's open-hour volume floor contradicts system_config (3.1).

## 6. Trading rules

Scope: every rule that selects an instrument, admits an entry, sizes it, exits it, gates it by regime, or halts trading. Each row cites file + function + approx line. "Live?" uses these values:

- **LIVE**: on the executing path; places, blocks or exits orders.
- **LIVE (Py-order)**: Python POSTs the order directly. It bypasses Node's openOrders, sweeper and audit trail.
- **COMPUTED-ONLY**: computed and returned, but nothing executes it.
- **DEAD**: no reader, unreachable, or always false.
- **DISABLED**: turned off by a flag.
- **DISPLAY**: UI or audit text only.

"Verified" means I read the code path this session. "Inferred" means it follows from the code but was not checked against live orders.

---

## 6.0 How one candidate flows (the real execution chain)

1. **Node `scheduleTier2`** (server/bot.ts ~8223-8300) runs whenever `clock.is_open || 4<=ET<20`. That condition is weekday-blind (`getETHour`, server/orderParams.ts). `tier2Intelligence(clock.is_open, etH)` (~8254) captures `isMarketOpen` once, at scan start.
2. **Python `bot_engine._scan_market_inner`** (bot_engine.py ~2700-4298) does all of the following in one call, in this order:
   - DD-halt check (~2815)
   - quick scan (~2952)
   - deep score on the top **5** (~3099, `VOLTRADE_DEEP_SCORE_CAP`)
   - entry filters and sizing (~3409-3798)
   - covered-call sweep (~3841)
   - options scanner (~3962)
   - `manage_positions` (~4030; computes only)
   - `manage_options_positions` (~4046; **Python POSTs option closes**)
   - third leg (~4057; **Python POSTs SVXY**)
   - convexity (~4068; disabled)
   - SPY floor (~4089; **Python POSTs**)
   - defensive floor (~4101; **Python POSTs**)
   - risk kill switch + tier engine (~4118-4212)
3. **Node `tier2Intelligence`** (bot.ts ~4769-5711) then applies:
   - daily -3% halt (~5067)
   - EOD short cover (~5097)
   - `executeTrades` if open, otherwise `morningQueue` (~5112-5122)
   - **tier dispatcher** (~5127-5473)
4. **Node Tier 1** (every 45s, market hours only, bot.ts ~8165-8180) runs:
   - drawdown kill (~4265)
   - stale-order sweep (~4152)
   - `manage_positions` refresh (~4557)
   - `syncMonitoredPositions` (~6906; POS-KILL, POS-WARN, TIME-EXIT)
   - morning queue (~4580)
5. **Node WebSocket 1-min IEX bars** call `checkPositionOnTick` (~7281). This is **the only live stock exit engine**.

---

## 6.1 Instrument selection — what gets traded and how the system decides

| Rule | Where | Threshold / logic | Applies to | Live? |
|---|---|---|---|---|
| deep_score side / trade_type | bot_engine.py `deep_score` ~1852-1875 | `combined_score > 65` and VRP > 5 gives side=sell, trade_type=options, "SELL OPTIONS". Otherwise, score > 65 and RSI > 70 gives side=sell, label "SELL" (stock). Otherwise BUY. Stock shorting is commented out. | every deep-scored stock | LIVE (label feeds everything below) |
| Stock-short conversion | bot_engine.py `_scan_market_inner` ~3601-3604 | side sell/short with trade_type != options is **rewritten to BUY** | RSI>70 "SELL" candidates | LIVE (see conflict N3) |
| Unified 3-way selector | instrument_selector.py `select_instrument` ~1277-1530 | Scores stock vs 2x ETF vs options. Highest score wins. Stock is always a candidate. | every admitted scanner candidate | LIVE (choice only; its `sizing` / `stop_config` are discarded) |
| Options eligibility | instrument_selector.py ~1430-1437, `_score_options` ~750-755 | regular hours, score >= `MIN_OPTIONS_SCORE` **65**, options exposure < `MAX_TOTAL_OPTIONS_PCT` **16%**, never `sell_call` | options branch | LIVE |
| Sell-premium trigger | instrument_selector.py `_score_options` ~773-781 | "SELL OPTIONS" **or VRP > 5 or IVR > 60**. Market IVR is injected from the VXX ratio (≥1.10 gives IVR 70) at ~1358-1374. Result: sell CSP; upgraded to `bull_put_credit_spread` if score ≥ 70, IVR > 40, VRP > 5, VXX > 1.0, GEX not explosive, no earnings within 7 days (~856). | options | LIVE as a choice (re-decided in Node, see N4) |
| QQQ weekly iron condor | instrument_selector.py ~828 | ticker QQQ, Monday, VXX 0.80-1.05 | QQQ | LIVE as a choice |
| Covered call | instrument_selector.py ~834 | shares_held ≥ 100, no earnings, regime not BEAR/PANIC (uses `vix_regime` text, wrong field) | held names | Effectively DEAD: held tickers are skipped earlier (~3425) |
| Buy calls / puts, debit spreads | instrument_selector.py ~861-888; options_execution.py `should_use_options` ~342-361 | "disabled by backtest". Routed to stock. | options | DISABLED in selector (but see N4: `bear_put_spread` still reachable) |
| 2x ETF | instrument_selector.py `_score_etf` ~546-717 | Mapping in analyze.py `LEVERAGED_ETFS` ~1323 (TSLL, NVDL, SSO, QLD, … **TNA and FAS are 3x**). Requires the 200d-MA gate. Prefers a 2-3 day hold (expected hold defaults to 3). VIX penalty. PANIC block reads `trade["macro"]`, which is never set (DEAD sub-rule). | stock candidates with a mapped ETF | LIVE |
| ETF execution | server/bot.ts `executeTrades` ~5907-5970 | buys `floor(trade.shares/2)` of the ETF at `getOrderParams(trade.price)`, i.e. **the underlying's price** | ETF choice | LIVE (bugged sizing and pricing, N6) |
| Stock-vs-options re-decision | options_execution.py `should_use_options` ~251-370 via `evaluate_and_execute` ~2548 | score ≥ 65; regular hours; options exposure ≤ 16%; "SELL OPTIONS" or **VRP > 2** gives `sell_cash_secured_put` if side=buy, **`bear_put_spread` if side=sell**; VRP < -2 gives stock | every non-scanner options trade in Node | LIVE (N4) |
| Scanner-direct options | bot.ts `executeTrades` ~5977-6053 | trades with `regime_at_entry == "OPTIONS_SCANNER"` skip `should_use_options` and go straight to `select_contract` + submit | options_scanner output | LIVE |
| Options scanner setups | options_scanner.py `get_options_trades` ~1771-1915 | only `HIGH_EDGE_SETUPS` = earnings_iv_crush (always `iron_condor`, ~811), vxx_panic_put_sale (`bull_put_credit_spread`, ~958), high_iv_premium_sale (iron condor only). `min_score = max(60, …)`, +5 in a short week. Skips half-days and pre-long-weekends (~1797). | market-wide | LIVE |
| Tier 1 CSP core | tiered_strategy.py `tier1_csp_core` ~466-597 | top of `csp_universe` ranking, de-duped by **underlying** (F5), affordability price ≤ equity × 8% × scalar / 95, **max 3 per scan**, slots `MAX_OPTIONS_POSITIONS` 6 | universe of up to 200 names | LIVE |
| Tier 2 leverage | tiered_strategy.py `tier2_leverage_multiplier` ~604-675 | requires PM (multiplier ≥ 4), DD > -8%, VXX < 1.40 | T1 actions | DEAD at execution: the dispatcher never passes `size_pct` (N8) |
| Tier 3 trend long SPY/QQQ | tiered_strategy.py `tier3_trend_capture` ~682-747 | VXX ≤ 0.95, SPY/MA50 ≥ 1.01, **`ctx.regime` in BULL**, ≤ 40% | SPY, QQQ | DEAD: `ctx.regime` is always "NEUTRAL" (N1) |
| Tier 4 tail hedge (SPY puts) | tiered_strategy.py ~754; `TIERS_ENABLED[4]=False` ~85 | 1.5%/yr, 60 DTE, 10-delta | SPY | DISABLED |
| QQQ put convexity overlay | system_config.py `CONVEXITY_OVERLAY.enabled=False` ~242; bot_engine.py `_run_convexity_overlay` ~4636 | 2%/4% budget, 60 DTE, 20% OTM, roll at 21 DTE | QQQ | DISABLED (Node still excludes QQQ puts from slot counts, bot.ts ~468, ~493) |
| Intraday stock shorts | bot_engine.py ~4036-4039 (disabled); bot.ts ~5097 still calls `close_open_shorts` at 15:50 ET | — | — | DISABLED (cover call harmless) |
| Stock short hard blocks | bot.ts `executeTrades` ~6115, morning queue ~4417, `/api/bot/trade` ~3540 | sell/short gets converted to BUY. **The morning-queue block skips `trade_type=="options"`.** | stock orders | LIVE (hole, N2) |
| Leveraged / inverse exclusion | csp_universe.py `_layer1_hard_gates` ~273; options_scanner.py ~1403 (`leveraged_etf_filter`) | excluded from CSP and scanner | options only | LIVE (stocks are not excluded; ETFs are actively bought, N6) |

---

## 6.2 Entry rules

### Stock / ETF (scanner path, Python then Node)

| Rule | Where | Threshold | Live? |
|---|---|---|---|
| Price / volume floor (quick scan) | bot_engine.py ~2946, ~2965 | price ≥ **5** (module const `MIN_PRICE`); volume ≥ **500k**, but **100k** in 9:30-9:59 ET; symbol has no "." and ≤ 5 characters | LIVE |
| Quick-score rank, then deep-score cap | bot_engine.py ~2970-2988, ~3099 | rank = \|chg%\| capped at 15 × 3 + volume term + direction bonus; penalty -15 above 30%, -30 above 50%; only the **top 5** are deep-scored (3 on "trim", 0 on "skip": memory guard 400/550 MB, ~3117-3146; survival mode ≥ 700 MB returns no trades, ~3044-3088) | LIVE |
| Portfolio DD halt (Python) | bot_engine.py `update_equity_peak` ~584-725, gate ~2832 | DD ≥ **18%** halts **the whole scan**. Confirmation ≥ 60s later at ≥ 9%, otherwise released as bad data. Resumes only in BULL/NEUTRAL **and** within 5% of the peak. | LIVE (scope conflict N9) |
| Held-ticker skip | ~3425 | ticker already in positions | LIVE |
| Stop cooldown (regime-aware) | ~3447-3480 | PANIC 4h, BEAR 3h, CAUTION 2.5h, NEUTRAL 2h, BULL 1.5h after a stop. File is written by **both** Node (~7713) and Python `manage_positions` (~2362). Config `STOP_COOLDOWN_SECONDS` is unread. | LIVE |
| Regime gate | ~3483 | no stock longs in **BEAR/PANIC** (`get_market_regime` with 200d inputs) | LIVE |
| Adaptive MIN_SCORE | ~3490-3507; system_config.py `get_adaptive_params` ~441 | BULL **63**, NEUTRAL 63 (but 0 slots), CAUTION **67**, BEAR/PANIC **75** | LIVE |
| Stock slots | ~3367-3379, ~3410 | `MAX_POSITIONS` from `get_adaptive_params`: NEUTRAL/BEAR/PANIC give 0; BULL and CAUTION are **overwritten by size tier** (8 at $100k-250k). Counts **all equity positions incl. floor ETFs**. | LIVE (N10, N11) |
| Sector / beta | `check_sector_correlation` ~1962-2041 | ≤ `MAX_SECTOR_POSITIONS` (tier: 2) per sector; if ≥ 3 holdings, block when there are already 3 high-beta (≥1.5) names or avg beta > 1.6. Fail-open on exception. | LIVE |
| Dollar volume | ~3557-3564 | ≥ **$50M** (hardcoded) | LIVE |
| Blocked tickers | ~3570-3576 | `BASE_CONFIG.BLOCKED_TICKERS` + UBER | LIVE (bypassed by ETF substitution, N6) |
| Extreme mover | ~3581-3599 | change > **+50%** goes to the overnight watchlist | LIVE |
| Bid/ask spread | ~3610-3631 | spread > **0.5%** rejects; fail-open on quote error. Config `MAX_SPREAD_PCT` is unread. | LIVE |
| Sizing block | ~3672-3718 | `calculate_position` blocked (halt, negative Kelly, 0 shares) | LIVE |
| Volume-share cap | ~3731-3753 | position / $-volume ≤ tier cap (3% at $100k) | LIVE |
| Risk kill switch entry suppression | ~4128-4209, `suppress_entries_on_kill` ~2523 | `check_kill_switches` killed: manual file, DD ≤ -20%, 5 consecutive losses (24h), or prior kill not resumed | LIVE |
| Node: kill switch | bot.ts ~5777 | loop ends | LIVE |
| Node: slots | bot.ts ~5783-5794 | stocks `MAX_POSITIONS` **8** (counts all `us_equity` incl. floor); options **6** (`countOptionsPositions` + resting sell orders, excludes QQQ puts) | LIVE |
| Node: score floor | bot.ts ~5795; `state.minScoreThreshold` ~507, set from `diagnostics.get_auto_fix_params` ~4800 (65-80) | score < **65** skip | LIVE (N12) |
| Node: daily ticker block | ~5803 | 3 stops on the same ticker blocks it for the day | LIVE |
| Node: size ceiling | ~5810 | value > equity × **20%** × diag multiplier | LIVE |
| Node: active heat cap | ~5818 | non-floor market value + trade > **30%** of equity: loop ends | LIVE |
| Node: sector concentration | ~5830-5845 → risk_kill_switch.py `check_correlation_pre_trade` ~577 | sector > **40%** of total exposure (≈25-name map); fail-open | LIVE |
| Node: halt / retry / dup / BP | ~5848-5893 | halted symbol; ≥ **10** attempts per ticker per day; open order on the same symbol; value > buying power → replace the weakest entry if new score ≥ old + 10 | LIVE |
| Node: daily loss | bot.ts ~5061-5072 | (equity − last_equity)/last_equity ≤ **-3%** → return before executeTrades / dispatcher (but after Python's own order paths ran) | LIVE |
| Node: weekly loss | bot.ts ~5075-5092 → diagnostics.py `check_weekly_loss` ~38 | -8% pauses, -4% halves multiplier, but it is called with a **1-row history**, so it returns "continue" at ~44 | DEAD |
| Off-hours deferral | bot.ts ~5112-5122 | market closed: up to 8 `new_trades` go to `morningQueue` | LIVE |
| Morning queue | bot.ts `executeMorningQueue` ~4333-4520 | entries older than 16h are dropped; **slots = ALL positions** vs 8; size ≤ 20%; gap > 5% skips (2-5% re-sizes); qty ≥ 1. **No score, regime, heat, sector, earnings or options check.** | LIVE (N2) |
| Stream (Tier 0) signals | bot.ts ~7860-7890, ~4815-4822 | 2.5x volume, ≥ 0.8% bar, ≥ 0.3% 5-bar move, 10-min cooldown, queued "for priority" — but `run_full_scan` is called with `{}` and **never receives the tickers** | DEAD (audit only) |

### Options entries

| Rule | Where | Threshold | Live? |
|---|---|---|---|
| CSP universe layer 1 | csp_universe.py `_layer1_hard_gates` ~203-300 | price ≥ $10 and ≤ equity × 25% / 95 ($250 default); volume ≥ 1M; $-volume ≥ $20M; not blocked or leveraged | LIVE |
| CSP universe layer 2 | csp_universe.py `_layer2_score` ~618-800 | 7-factor composite. Excluded if earnings ≤ **7d** (score ≤ 10) or the calendar lookup failed (KB #45 fail-closed). | LIVE |
| T1 IV-rank floor | tiered_strategy.py `T1_MIN_IV_RANK=40` ~181 | carried in metadata only | DEAD |
| T1 target DTE / delta | tiered_strategy.py ~182-184 | 30-45 DTE, 0.30 delta in metadata; `select_contract` ignores metadata and uses CSP_ENTRY_MIN_DTE 30 to fetch max 50, delta -0.30 hardcoded (options_execution.py ~1043) | metadata DEAD, behaviour LIVE |
| CSP entry DTE floor (F3) | options_execution.py `CSP_ENTRY_MIN_DTE` ~107, `select_contract` ~432-440, `_select_sell_put` ~938-955 | ≥ max(30, 21+1) = **30**; puts ≤ 21 DTE dropped | LIVE |
| Multi-leg entry DTE floor (F3 recurrence, v1.0.1117) | options_execution.py ~124-148 | all grouped multi-leg except the weekly qqq_iron_condor fetched from **30** DTE; `ENTRY_EXIT_ZONE_CONFLICTS` lists 6 strategies still inside the zone | LIVE |
| Contract liquidity | options_execution.py ~88-90, `_is_liquid` ~719 | OI ≥ 200, volume ≥ 10, spread ≤ 10% of mid | LIVE |
| Market calendar skip | options_scanner.py ~1797-1813 | half-day or pre-long-weekend: no scanner trades | LIVE (**not** applied to T1 CSP dispatch) |
| Off-hours tier defer (F6) | server/tierDispatchGate.ts; bot.ts ~5222 | SELL_CSP / BUY / BUY_PUT deferred while `isMarketOpen=false` | LIVE |
| Options slot cap at dispatch | bot.ts ~5146-5161, ~5226 | positions + resting sell-to-open orders ≥ **6** | LIVE |
| CSP re-submit brake (F7 + KB #46) | bot.ts ~5155-5156, ~5235-5249; server/cspSweepCooldown.ts `shouldDispatchCsp` ~154 | skip an underlying swept < 60 min ago (in memory), with a resting sell-to-open, or with a broker-closed never-filled order that rested ≥ 12 min and ended < 60 min ago | LIVE |
| Live capital | bot.ts ~5188-5192, ~5311 | `options_buying_power`, decremented per SELL_CSP | LIVE |
| Earnings guard (covered calls) | bot_engine.py ~3868-3874; options_execution.py `_check_earnings_guard` ~154 | `_earnings_scalar` ≥ 0.85; fail-open | LIVE (sweep never executes, N4) |
| Earnings IV-crush setup | options_scanner.py `_setup_earnings_iv_crush` ~720-860 | **sells an iron condor 0-7 days before earnings** on IV ≥ 40% | LIVE (contradicts the CSP earnings exclusion; DTE floor distorts it, N7) |

---

## 6.3 Position sizing

| Rule | Where | Value | Live? |
|---|---|---|---|
| Kelly base | position_sizing.py `_kelly_fraction` ~153-208; `calculate_position` ~931-954 | per-bucket win rate / avg win / avg loss; **third-Kelly** (default arg 3.0); negative EV or zero wins blocks; no losses gives 0.5% starter; clamped 1%-8% | LIVE |
| Scalars | `calculate_position` ~967-993 | volatility × confidence × regime (VIX) × earnings × time of day × heat × liquidity | LIVE |
| DD scalar | ~995-1026 | 0.8 / 0.6 / 0.4 at -5 / -10 / -15%. Reads `voltrade_equity_peak.json["peak_equity"]`, but Node writes `{"equityPeak": …}` (bot.ts `saveEquityPeak` ~638-648), so `_peak` falls back to `equity` and **never applies** | DEAD (N13) |
| Stress multiplier | ~1028-1047 | `stress_index.get_stress_multiplier()` | LIVE |
| Absolute clamp | position_sizing.py `ABSOLUTE_MAX_POSITION_PCT=0.08` ~42 | 8% per stock. Regime `MAX_POSITION_PCT` (BULL 15%, CAUTION 10%, tier 12%) is **never applied to stock sizing**. | LIVE (config DEAD for stocks) |
| Entry stop / TP from sizer | ~1066-1115 | stock 2.5×ATR, clamp 5-12%, TP 6-18%; ETF 1.5×, 4-9% | COMPUTED-ONLY (6th exit model, N5) |
| Options size | options_execution.py `_dynamic_options_size` ~177-237 | csp_options Kelly × scalars × **0.5**, clamp 1%-8% | LIVE |
| CSP budget | options_execution.py `_select_sell_put` ~964-1060 | max(size, **2%**) ≤ cash; **stretch to 20% of equity** for the smallest strike; qty ≤ **2** | LIVE (N8) |
| T1 size scalars | tiered_strategy.py ~486-500 | VXX ≥ 1.40 gives 0.30; ≥ 1.20 gives 0.60; ≥ 1.05 gives 0.85; DD ≤ -15% halves, ≤ -10% gives ×0.75 | DEAD at execution: `size_pct` never reaches `select_contract` (bot.ts ~5276) |
| Scanner base size | options_scanner.py ~1873-1881 | 6 / 7 / 8% by setup | DEAD: Node passes no `trade` to `select_contract` (bot.ts ~5992) |
| Regime / tier caps | system_config.py `get_adaptive_params` ~457-598; position_sizing.py `SIZE_TIER_TABLE` ~74 | tier sets `MAX_POSITIONS`, `MAX_POSITION_PCT`=min(regime, tier), sector cap, volume share; `MIN_POSITION_DOLLAR` unread | PARTIAL |
| Node ceilings | bot.ts ~361-364 | 20% per trade, 30% active exposure, 8 positions, diag multiplier | LIVE |
| Config keys with zero readers | system_config.py | `KELLY_DIVISOR`, `OPTIONS_SCALE`, `MIN_POSITION_PCT`, `MAX_TOTAL_CAPITAL_PCT`, `ML_CONFIDENCE_THRESH`, `OPEN_HOUR_MIN_VOLUME` (only used by `time_of_day="open"`, never passed), `STREAM_*`, `REGIME_VXX_*`, `REGIME_BEAR_THRESHOLD`, `CIRCUIT_BREAKER_*`, `STOP_COOLDOWN_SECONDS`, `DAILY_LOSS_LIMIT_PCT`, `MAX_SPREAD_PCT` (Python const of the same name is separate), `MIN_TP_PCT`, `MAX_TP_PCT`, `ATR_TP_MULTIPLIER` | DEAD (grep, excluding tests and `.claude` worktrees) |

---

## 6.4 Exits and stops

### LIVE stock exits: server/bot.ts `checkPositionOnTick` (~7281-7785), fed by `syncMonitoredPositions` (~6906-7214)

| Rule | Line | Threshold | Notes |
|---|---|---|---|
| Phase-1 stop | ~7131, ~7525, ~7564 | `clamp(2×ATR%, 1.5, 8)`; R = `initial_risk_pct` (Python: 2×ATR%, unclamped) | ATR% is rewritten each cycle by Python |
| Scale-out 1 | ~7333-7340 | 1/3 at **1R**, arms breakeven | only if `remainingQty > 1` and `floor(orig/3) ≥ 1` |
| Scale-out 2 | ~7320, ~7341 | 1/3 at 3R / 2R / 1.5R (bullish / neutral / bearish, **current** regime) | |
| Trails | ~7516-7555 | phase 2: max(1.5, 1.5×ATR) from peak P&L; phase 3: max(1, ATR); phase 4: ATR × {3, 2, 1.5} from high price | peaks come from Python's 45s samples |
| Breakeven | ~7557 | P&L < 0 after scale-out 1 | |
| Take profit | ~7133, ~7573 | max(4, min(3×ATR, 15)), only when phase < 3 **and** both scale-outs are done | effectively bearish-regime only |
| Time stop | ~7581-7585 | ≥ **7 calendar days** and \|P&L\| < 2% | |
| Order type | orderParams.ts `getOrderParams` | stop/trail: market in regular hours; TP and scale-outs: limit; extended hours: limit ±0.5% | |
| Covered-call unwind before sale | ~7378-7417, ~7594-7632 | blocks the exit if the buy-back fails | |
| Circuit breaker | ~7726-7770 | 3 consecutive stops: if `state.dailyPnL`/peak ≤ -5%, halt to 20:00 *server-local*; else 2+ tickers gives 1h; else block the ticker for the day | `dailyPnL` is only updated by the UI `/api/bot/account` route (~1246), so the day-halt branch is stale. Config `CIRCUIT_BREAKER_*` is unread. |
| POS-KILL (stock) | ~6979, ~7011-7101 | unrealized ≤ **-25%**: market liquidation, with a duplicate guard | |
| POS-KILL option backstop (F4) | ~446-457, ~6995-7010 | short ≤ **-150%**, long ≤ **-75%**; between -25% and the backstop, one HOLD audit line per day | |
| POS-WARN | ~6980, ~7102 | ≤ -15%, hourly audit | |
| TIME-EXIT | ~6943-6972 | > 14d and P&L in [-5%, +2%] → market close | reads `pos.created_at` / `submitted_at`, which Alpaca positions do not carry. **Probably DEAD** (KB #48 says verify). |
| Exemption | ~6929, `FLOOR_AND_LEG_TICKERS` ~409 | QQQ, SVXY, SPY, SMH, KWEB, VXUS, GLD skip **all** of the above, including POS-KILL | SVXY / SPY / QQQ have no stop at all (N14) |
| Kill-switch scope | ~7285, ~7789, ~8169 | `state.killSwitch` or `!active` stops **all** WS exits, syncs and POS-KILL | N15 |

### Python stock exits: bot_engine.py `manage_positions` (~2045-2419). These compute only.

Node discards the actions (bot.ts ~4564-4572) and only reads the stop-state file. The following rules **never execute live**:

| Rule | Line | Threshold |
|---|---|---|
| Minimum hold | ~2131, ~2303-2306 | 60 minutes unless P&L ≤ -10% |
| Short-hold grace | ~2321-2333 | phase-1 stops suppressed for < 4 days unless ≤ -15%, hard floor, or breakeven |
| Hard floor | ~2243-2250 | `POSITION_HARD_STOP_PCT` **-20%** (phase 1) |
| Regime stop scalar | ~2194-2217 | `STOP_SCALAR_*` 0.7-1.2 |
| Phase-1 clamp | ~2215 | 1.5-**10%** (Node uses 8%) |
| Python stop cooldown write | ~2361-2369 | written even though Python places no order |
| **Phantom scale-out state** (KB #47) | ~2164-2191, persisted ~2414 | `scales_completed`, `remaining_qty`, `breakeven_active` written without an order; Node adopts via `Math.max` / `Math.min` (bot.ts ~7139-7164). **Still present.** |

### Options exits: options_manager.py `manage_options_positions` (~855-1340) and `_manage_strategy_group` (~571-855)

These are LIVE (Py-order): Python POSTs the limit closes and rolls itself (`_submit_close_order` ~303).

| Rule | Line | Threshold |
|---|---|---|
| Minimum hold | Check 0 ~1058, ~1094 | `MIN_HOLD_MINUTES` 60, except critical DTE |
| Critical DTE | Check 1 ~1069 | ≤ **5 DTE**: close |
| Assignment risk | Check 2 ~1097 | short with \|Δ\| > 0.80: roll, else close |
| 21-DTE management | Check 3 ~1144 | ≤ **21 DTE**: short rolls (30-50 DTE, net credit ≥ 0.05) or closes; long closes |
| Profit target | Check 4 ~1202 | **50%** of credit |
| Loss limit | Check 5 ~1232 | short: cost ≥ **2×** credit; long: -**50%** |
| Gamma | Check 6 ~1282 | \|Γ\| > **0.08** and DTE ≤ 30: close (**not price-normalised, not side-aware**, N16) |
| Delta drift | Check 7 ~1304 | > 0.25: warning only |
| Groups | ~636-845 | hold, critical DTE, 50% profit, iron-condor 50%-of-max-loss, 2× credit / -50% debit, 21 DTE, VRP-flip "edge evaporated" |
| Hours gate | — | **None.** Runs from pre-market to 20:00 ET, including weekends (N17). |
| Duplicated (dead) T1 exit metadata | tiered_strategy.py ~185-187 | `take_profit_pct` 0.5, `stop_loss_mult` 2, `force_close_dte` 21 — never read |

### Floors and third leg (Py-order; no Node stops)

| Rule | Where | Threshold | Live? |
|---|---|---|---|
| SPY/QQQ floor basket | bot_engine.py `_manage_spy_floor` ~5170-5590 | target = `SPY_FLOOR_{regime}` (BULL 70%, NEUTRAL 90%, CAUTION 35%, BEAR/PANIC 0) × basket weights (QQQ 0.5, SMH 0.15, KWEB 0.10, VXUS 0.15, CASH 0.10); drift ≥ 3% triggers rebalance; trend caps: QQQ < 200d gives ≤ 50%, death cross gives ≤ 20% (~5255-5268); market orders only while 9:30-16:00 ET (hour-only check `_is_regular_hours` ~4976) | LIVE (N18, N19) |
| Floor regime dwell (F2) | `_resolve_floor_regime` ~5047; `FLOOR_REGIME_CONFIRM_CYCLES` 3 | calming changes need 3 distinct macro snapshots; risk-off is immediate | LIVE |
| Zero-target exit | ~5400-5449 | BEAR/PANIC: sells **only `FLOOR_TICKER` (QQQ)**, after a covered-call unwind | LIVE (N18) |
| Defensive GLD floor | `_manage_defensive_floor` ~5592 | 8 / 8 / 15 / 30 / 40% (BULL→PANIC); death cross gives ≥ 25% | LIVE |
| Third leg: SVXY VRP harvest | `_run_third_leg` ~4440-4587 | buys `min(LEG3_VRP_PCT 0.40, 0.6 × remaining capacity)` of equity in SVXY when VXX ratio 1.05-1.25 and the 5-day trend is declining; limit +0.1% | LIVE (no hours gate; N14, N20) |
| Third-leg exit (F1) | ~4555-4575 | NEUTRAL/BULL: market-sell `{SVXY, ITA, XOM, LMT}` (GLD removed by F1) | LIVE (no dwell) |

---

## 6.5 Regime filters (how many "regimes" exist)

| Classifier | Where | Inputs / thresholds | Consumers |
|---|---|---|---|
| `regime_util.classify_regime_5level` (canonical) | regime_util.py ~74-126 | PANIC VXX ≥ 1.40; BEAR VXX ≥ 1.15 or SPY/MA50 < 0.94 or ≥ 10 days below 200d (or < 200d and ≥ 5 days); CAUTION VXX ≥ 1.05; BULL VXX ≤ 0.95 and SPY ≥ 0.98 and above 200d; else NEUTRAL | `system_config.get_market_regime` ~386 |
| Legacy fallback | system_config.py ~414-438 | uses unread `REGIME_*` keys (0.90 / 1.05 / 1.15 / 1.40, SPY 0.96 / 0.94) | only if the import fails |
| Inline fallback | bot_engine.py ~3467-3472 | PANIC at **1.30** (stale) | only on exception |
| `get_adaptive_params` | system_config.py ~453 | **does not take `spy_above_200d`**; callers in the scan omit `spy_below_200_days` (~3371) | MIN_SCORE, MAX_POSITIONS, caps (N10) |
| `get_regime_caps` | tiered_strategy.py ~105-131 | forces `spy_below_200_days=0`, `markov_state=1` | master kill, T1-T3 |
| TierContext.regime | bot_engine.py ~4140 | `_regime_ctx` exists only inside `deep_score` (~1489), so the value is always **"NEUTRAL"** | T3 gate (N1) |
| Node exit regime class | bot.ts ~7314-7323 | BULL / NEUTRAL_BULL bullish; BEAR / PANIC / CAUTION bearish (from Python stop state, current regime) | scale-out 2, trail width |
| deep_score macro factors | bot_engine.py ~1213-1241 | `vix_regime` extreme / low, `market_regime` risk_on / off, sector momentum ±1.5% | score ± 3-10 |
| Instrument market IVR | instrument_selector.py ~1358-1374 | VXX ratio maps to IVR 20-85 | sell-premium switch |
| position_sizing regime | position_sizing.py ~1073 | reads `macro["regime"]`, which macro_data does not produce, so the scalar is always 1.0 | informational stop |
| Kill-switch regimes | risk_kill_switch.py ~64, ~271; tiered_strategy.py ~95 | VXX ≥ 1.40 kills T2-T4; resume needs VXX < 1.20 | tiers |

---

## 6.6 Kill switches and halts (eight independent mechanisms, four peaks)

| Mechanism | Where | Threshold | Peak source | Effect | Live? |
|---|---|---|---|---|---|
| Node drawdown kill | bot.ts `checkDrawdownKillSwitch` ~4265-4327 (also `/api/bot/account` ~1251) | **-10%** from peak (`state.maxDrawdownPct` ~514), validated read | `voltrade_equity_peak.json {equityPeak}` | cancel all orders; latch; **stops everything incl. exits** | LIVE |
| Liquidate on kill | same ~4294 | -25% and `VOLTRADE_LIQUIDATE_ON_KILL=true` | — | market-close all | LIVE if env set |
| Paper auto-resume | server/killSwitchAutoResume.ts `AUTO_RESUME` ~69-98 | ≥ 30 min, 3 consecutive OK reads, basis RECOVERED (> -8%), DATA_ANOMALY (< 25% explained), or PAPER_REBASE (2 market days) | rebases Node peak + `paper_resume_sync.py` | clears the latch | LIVE |
| Python DD halt | bot_engine.py `update_equity_peak` ~584 | **18%**, confirm ≥ 60s at ≥ 9%; resume in BULL/NEUTRAL within 5% | `voltrade_portfolio_dd.json` | scan returns empty, **including floors, options exits, third leg** | LIVE |
| risk_kill_switch | risk_kill_switch.py `check_kill_switches` ~154-365 | manual file; DD **-20%**; **5** consecutive losses (24h); resume ≥ 4h + 95% of peak + VXX < 1.20; daily **-3%** and VXX ≥ 1.40 only set `tier_blocks` / `blocks_entries`, which the caller ignores (bot_engine.py ~4133 reads `killed` only) | `voltrade_killswitch.json` + `voltrade_peak_equity.json` | suppress scanner entries + tiers | LIVE (partially DEAD fields) |
| Tier master kill | tiered_strategy.py `master_kill_switch` ~278-303 | DD **-20%**; daily **-3%**; invested (all positions, incl. floor) ≥ regime `MAX_TOTAL_EXPOSURE` (95 / 95 / 60 / 50 / 30%) | `voltrade_peak_equity.json` | kills **all** tiers incl. T1 CSP (KB #20) | LIVE |
| Node daily loss | bot.ts ~5061-5072; `checkTradeAllowed` ~1060 (manual route only) | **-3%** | last_equity | skip executeTrades + dispatcher | LIVE |
| Node circuit breaker | bot.ts ~7743-7770 | 3 stops | — | blocks tier1 / tier2 timers (~8171, ~8235) | LIVE |
| `check_position_risk` -25% | risk_kill_switch.py ~505-550; registered in voltrade_daemon.py ~284 | -25% / -15% (no option scaling) | — | no Node caller | DEAD (dangerous if wired: re-opens the F4 bug) |
| Weekly loss -8% | diagnostics.py ~34, ~38 | — | — | always "continue" | DEAD |
| Status / display values | bot.ts `/api/bot/status` ~2579-2582; startup RULES audit ~8466-8474 | says 8% / 95% / **5%** daily and "emergency backstop 15% / TP ceiling 25%" (`STOP_LOSS_PCT`, `TAKE_PROFIT_PCT` ~378-379, never enforced) | — | — | DISPLAY (wrong) |

---

## 6.7 Known conflicts: verification against current code

| ID | Claim | Status in code today |
|---|---|---|
| **KB #46** (deploy cancel-all + in-memory CSP brake) | open_questions.md ~7370-7434 | **PARTIALLY FIXED.** v1.0.1158 (experiments.md ~109882) added the broker-derived brake (bot.ts ~5155-5156, ~5241; cspSweepCooldown.ts `isRecentUnfilledRestedCsp` ~129). **Still open:** `gracefulShutdown` cancels all orders (bot.ts ~846, which also cancels resting *exit* limits); `isMarketOpen` captured at scan start (~8254); open-orders fetch fails open `.catch(() => [])` (~5153, and ~5156 for closed orders); `replaceIfBetter` still treats the 204-parse throw as failure (~4236-4244); `openOrders` is never rehydrated (`const openOrders` ~4097, no boot load). |
| **KB #47** (phantom first scale-out) | bot_engine.py ~2166-2179 writes scale state; bot.ts ~7141-7164 adopts it | **STILL PRESENT, unmeasured.** `live_exit_policy.py` (docstring ~25-27) lists it as unmodelled. |
| **KB #48** (three exit models) | config / backtest / live | **Steps 1-2 shipped**: exit-aligned labeler v1.0.1087; `backtest_v2` `exit_model="live"` v1.0.1088 via `live_exit_policy.py`. **But `DEFAULT_EXIT_MODEL="legacy"`** (backtest_v2.py ~169), so PROMOTION RULE 3 backtests still score the dead config exits. Config keys `ATR_STOP_MULTIPLIER`, `MIN/MAX_STOP_PCT` are read only by backtest_v2 legacy (~141-143); `MIN/MAX_TP_PCT` and `ATR_TP_MULTIPLIER` have zero readers; `TIME_STOP_DAYS` is read only by mirrors / comments. Step 3 (pick and delete the loser) is open. **Now at least six models, see N5.** |
| F1 GLD ownership | bot_engine.py `_floor_owned_tickers` ~4410, `_third_leg_exit_tickers` ~4429 | present. Three ownership lists still exist: bot.ts ~409, bot_engine.py ~123, ~4410. **T3 SPY/QQQ vs the floor is a new ownership clash (N1, N19).** |
| F2 floor dwell | system_config.py ~176; bot_engine.py `_resolve_floor_regime` ~5047 | present. The third-leg exit and SVXY buy have **no** dwell (N20). |
| F3 CSP DTE vs 21-DTE exit | options_execution.py ~105-148, ~432-440 | present, plus multi-leg recurrence fix. **The gamma check is a new same-class conflict (N16).** |
| F4 POS-KILL options | bot.ts ~446-457, ~6995-7010 | present |
| F5 T1 OCC de-dupe | tiered_strategy.py `_option_underlying` ~440, ~508 | present |
| F6 off-hours tier dispatch | server/tierDispatchGate.ts; bot.ts ~5222 | present. **Python order paths (floors, third leg, options_manager) have no equivalent gate (N17).** |
| F7 sweep loop | bot.ts ~4152-4218, ~4120 | present (with KB #46 caveats) |
| `check_position_risk` -25% | risk_kill_switch.py ~509 | confirmed unused (daemon-registered only) |
| Python-only rules (60 min, 4 days, -20%) | bot_engine.py ~2131, ~2321, ~2243 | confirmed computed-only. **`shared/positionLevels.ts` (header ~1-12) draws these non-executing levels on the Bot page as if they were the engine's (N5).** |

---

## 6.8 NEW conflicts / defects found this audit

Ordered by expected impact. All are verified in code unless marked *inferred*.

- **N1. Tier 3 can never fire, and KB #20's premise is wrong.**
  - bot_engine.py ~4140 passes `regime=_regime_ctx.get(...) if '_regime_ctx' in locals() else "NEUTRAL"`. `_regime_ctx` is assigned only inside `deep_score` (~1489, ~1502), so in `_scan_market_inner` it is always "NEUTRAL".
  - `tier3_trend_capture` requires BULL (tiered_strategy.py ~700), so the T3 dispatcher branch (bot.ts ~5344-5391) is dead.
  - KB #20(a) (open_questions.md ~112-121) says bot_engine passes "the richer, already-computed regime". It does not.
  - T3's `exit_on_ma_break` has no implementation anywhere (grep). If T3 is ever revived, its SPY/QQQ would have no exit (FLOOR_AND_LEG set), and its QQQ would be rebalanced away by the floor basket (shared ticker).

- **N2. The morning queue can open a naked stock short (inferred from code; check Alpaca history).**
  - deep_score "SELL OPTIONS" trades carry `trade_type="options"`, `side="sell"` and shares > 0 (bot_engine.py ~1862-1866). The stock-short rewrite skips options (~3602).
  - Off-hours they enter `morningQueue` (bot.ts ~5116). `executeMorningQueue`'s short block only converts when `trade_type !== "options"` (~4417), then POSTs `side: "sell"` on the **stock** (~4441-4449) and monitors it as "long" (~4509).
  - The same trade in hours becomes a stock **BUY** instead (bot.ts ~6115), or a bear put spread (N4). One signal has three opposite executions depending on time and path.

- **N3. RSI-overbought "SELL" is executed as a BUY.**
  - deep_score labels score > 65 with RSI > 70 as side=sell / "SELL" (~1872-1874). The scan converts it to BUY (~3602-3604).
  - So the overbought exit signal buys the stock.

- **N4. A second options decision overrides instrument_selector.**
  - Non-scanner options trades go through `evaluate_and_execute`, which re-runs `should_use_options` (options_execution.py ~2561). That function has different thresholds (VRP > 2, side-based) from instrument_selector (VRP > 5 or IVR > 60) and deep_score (VRP > 5).
  - For a "SELL OPTIONS" trade, Node passes `side:"sell"` (bot.ts ~6058), so `should_use_options` returns **`bear_put_spread`** (~336-339). That is a debit spread both selectors call "disabled by backtest".
  - When `should_use_options` declines, `trade_type=="stock"` trades fall back to buying stock (~6090-6097). These were counted as *options* slots, so they bypass the stock-slot cap (~5783-5794).
  - The covered-call sweep (bot_engine.py ~3841-3922; score 75, side sell, VRP None) always gets `use_options=False` in `should_use_options`, so it ends as OPTIONS-SKIP. **The covered-call sweep never executes.**
  - Scanner-path `select_contract` is called with no `trade` (bot.ts ~5992). A `covered_call` there would hit `trade.get` on None (options_execution.py ~514).

- **N5. There are at least six exit models (extends KB #48).**
  1. live Node `checkPositionOnTick`
  2. Python `manage_positions` (computes only)
  3. `system_config` dead keys
  4. `backtest_v2` legacy (still the default)
  5. `position_sizing` entry stop / TP: 2.5×ATR 5-12%, TP 6-18% (~1093-1098), shipped as `stop_loss` / `take_profit` on every trade (~3701) but unused
  6. instrument_selector `_build_stop_config` (~1165)

  In addition, `shared/positionLevels.ts` mirrors model 2. The Bot-page chart therefore shows levels (60-min / 4-day suppression, 10% clamp, regime scalar, -20% floor) that the live engine does not honour.

- **N6. ETF substitution is mis-sized, mis-priced, bypasses blocks, and has no max hold.**
  - Node buys `floor(trade.shares/2)` of the ETF, where `shares` is the underlying share count. This is not 50% of the dollars.
  - The limit price is the **underlying's** price (bot.ts ~5909-5911). Example: XLF→FAS (3x, ≈3× the price) never fills; most others are effectively market orders.
  - Selector ETF sizing (instrument_selector.py `_build_sizing` ~1106-1145) is discarded.
  - TNA and FAS are 3x, yet the config blocks 3x ETFs for 22% win rate (`BLOCKED_TICKERS`, checked only on the underlying ~3574).
  - `ETF_MAX_HOLD_DAYS=5` ("never hold leveraged ETF longer than 5 days", ~158) has no exit anywhere; the live time stop is 7 days and only if flat.
  - Options exclude leveraged ETFs, while stocks actively buy them.

- **N7. Earnings rules contradict each other.**
  - CSP excludes earnings ≤ 7 days (csp_universe.py ~776).
  - The scanner *sells* iron condors 0-7 days before earnings (options_scanner.py ~750-811).
  - instrument_selector penalises sellers at ≤ 2 days; `should_use_options` blocks selling when the earnings scalar < 0.6.
  - Since v1.0.1117 the earnings condor is fetched from **≥ 30 DTE** (options_execution.py ~437-440), not the `days+14` window the setup priced (options_scanner.py ~755). The setup's thesis and its execution diverge.
  - A 30-45 DTE CSP can still hold through earnings 8-45 days out.

- **N8. Tier sizing never reaches execution.**
  - The dispatcher calls `select_contract(ticker, "sell_cash_secured_put", 0, equity, cash_available)` (bot.ts ~5276), with no `size_pct`, metadata, trade, positions or macro.
  - T1 stress and DD de-risking (0.30-1.00 × 0.5-0.75), T2 leverage, T1 delta / DTE / IVR metadata, the heat scalar, and the scanner's 6-8% sizes are all ignored. `_dynamic_options_size` runs on default score 70.
  - CSP collateral can **stretch to 20% of equity** (~1000) despite `MAX_OPTIONS_PCT` 8%.
  - There are three different CSP affordability ceilings: T1 8% / 95 (tiered_strategy.py ~529-530), layer 1 25% / 95 (csp_universe.py ~268), selector 20% stretch.

- **N9. Halts have inconsistent scope.**
  - The Python DD halt returns before **options exits, floors and defensive GLD rotation** (bot_engine.py ~2833-2846). Its own comment says "only entries are gated". So an 18% drawdown freezes de-risking.
  - Node's daily-loss halt returns *after* Python's floor, third-leg and options orders have already been placed in the same scan (bot.ts ~5013-5072 runs after the scan returns).

- **N10. One scan computes the regime three ways.**
  - Slot params omit both 200d inputs (bot_engine.py ~3371). `get_adaptive_params` cannot take `spy_above_200d` (system_config.py ~453).
  - With SPY below its 200d but VXX calm, the gate regime is NEUTRAL (stock trading intended off), while the slots regime is BULL (8 slots). The NEUTRAL "no stock trades" rule leaks.

- **N11. Size tier raises the CAUTION cap and floor ETFs consume stock slots.**
  - The size tier overwrites the CAUTION `MAX_POSITIONS` 4 with 8 at $100k (system_config.py ~567-572). The comment only promises not to raise zero caps.
  - Both Node (~5721) and Python (~3342) count floor and leg ETFs (5-7 tickers) as stock positions against 8. The morning queue counts *all* positions, options included (~4363-4364).
  - Python's options-slot math uses the module constant `MAX_OPTIONS_POSITIONS = 8` (bot_engine.py ~200, ~3958) and counts QQQ puts. Node and config use 6 and exclude QQQ puts.

- **N12. Node's score floor silently overrides Python thresholds.**
  - `state.minScoreThreshold` is ≥ **65** (bot.ts ~507, ~4800). Python admits 63 in BULL; the scanner admits 60 ("ALPHA-TUNE 65→60", options_scanner.py ~1764-1768).
  - Candidates scoring 60-64.9 are always rejected in Node (`rejected_node_score`).

- **N13. The drawdown sizing scalar is dead.**
  - position_sizing.py ~1000-1005 reads key `peak_equity`; Node writes `equityPeak` (bot.ts ~643).
  - There are four separate peak stores: Node; bot_engine DD; tiered / risk `voltrade_peak_equity.json`; `voltrade_killswitch.json`.

- **N14. Unstopped tail positions.**
  - SVXY (short-vol ETF) is bought in CAUTION/BEAR-range VXX (1.05-1.25) and exited only on NEUTRAL/BULL. It is in `FLOOR_AND_LEG_TICKERS`, so it gets no WS stop and no POS-KILL (bot.ts ~6929).
  - Python `manage_positions` also skips it (~2104).
  - Leg size is up to 40% of equity (`LEG3_VRP_PCT` 0.40, whose comment says "15% sweet spot"). There is no exit during a vol spike.

- **N15. The Node drawdown kill also kills every exit.**
  - `checkPositionOnTick` (~7285), the 60s sync including POS-KILL (~7789), Tier 1 (~8169) and Tier 2 (so `options_manager` too) all stop while killed.
  - Python mechanisms explicitly keep exits running during a halt. F4's entry already notes this ("pre-existing"); it is not fixed.
  - The circuit breaker also skips Tier 1, which suspends the -10% drawdown check itself (~8171) unless the UI polls `/api/bot/account`.

- **N16. Gamma exit vs CSP entry (F3 class, inferred analytically).**
  - Check 6 closes any option with |Γ| > 0.08 once DTE ≤ 30 (options_manager.py ~1282). The threshold is per-share, so it is not normalised for price, and it is applied to longs too.
  - CSPs enter at 30-38 DTE. T1's affordability gate pushes toward cheap underlyings (≲ $84 at $100k).
  - Example: a 30-delta put on a $20 stock at σ = 0.4 and 30 DTE has Γ ≈ 0.15. Such CSPs would be gamma-closed on the first cycle after the 60-minute hold once DTE ≤ 30.
  - Verify against the options-manager audit (`gamma_risk`).

- **N17. Python order paths lack an hours, weekday or holiday gate.**
  - `options_manager` has no hours check. It POSTs DAY limits, then `state.pop()` drops the position's state even if the order never fills (~1287-1301).
  - The third leg has no hours check (~4504-4575).
  - The floors' `_is_regular_hours` checks the hour only (~4976-4995), and Tier 2 runs on weekends 04-20 ET (bot.ts ~8245). *Inferred:* weekend market orders queued for Monday, repeated each scan.
  - None of these are registered in Node's `openOrders`, so `sweepStaleOrders` never cancels them.

- **N18. Basket members are orphaned in BEAR/PANIC.**
  - At target 0%, `_manage_spy_floor` sells only `floor_ticker` (QQQ) (~5413-5431). The basket branch requires target > 0 (~5289).
  - So SMH, KWEB and VXUS (about 28-36% of equity) are held with no exit through the bear phase. This also pushes invested % over the BEAR/PANIC caps and fires the master kill (KB #20 chain).
  - The basket rebalance path does not unwind covered calls; the single-ticker path does (~5420).

- **N19. Tickers have multiple owners.**
  - QQQ is owned by the floor basket, T3 and the (disabled) convexity overlay.
  - SPY is owned by T3, T4 and FLOOR_AND_LEG.
  - The CSP underlying is shared by executeTrades CSPs and T1. `executeTrades` checks `held.includes(stock ticker)` against OCC symbols, so it cannot see an existing short put on the same underlying. T1's snapshot predates `executeTrades`.

- **N20. No dwell on the third leg.**
  - The SVXY buy (VXX 1.05-1.25) and the exit (regime NEUTRAL/BULL) share the CAUTION/NEUTRAL boundary at VXX 1.05 with no hysteresis. This is the same flicker class F2 fixed for the floors.

- **N21. Smaller dead or duplicate rules.**
  - The ML-uncertain branch (|ml−50| < 8) blends **50** at 40% weight. Its comment says "no drag", but it pulls a 70 rules score to 62, below `MIN_SCORE` (bot_engine.py `deep_score` ~1825-1832).
  - The ETF PANIC sub-rule reads a missing field.
  - The open-hour volume rule inverts the config: code uses **100k at open**, config says "need even more" (1M, `OPEN_HOUR_MIN_VOLUME`).
  - Weekly loss is dead. Stream priority is dead. The status endpoint numbers are wrong. `MIN_POSITION_DOLLAR` is dead.

---

## 6.9 Rules that duplicate the ML model's features (double counting)

The LightGBM model's 34 features are built inside deep_score (bot_engine.py ~1731-1781). They include:

`momentum_1m/3m, rsi_14, volume_ratio, adx, ewma_vol, vrp, iv_rank_proxy/iv_rank_stock, atr_pct, vxx_ratio, spy_vs_ma50, markov_state, regime_score, sector_momentum, cross_sec_rank, news_sentiment, insider_signal, intel_score, earnings_surprise, put_call_proxy/ratio, trend_strength, credit_spread, market_breadth, days_to_earnings`

The rules score blended with it 60/40 (~1810-1832) re-encodes the same inputs:

| Rules term | Where in deep_score | Duplicated ML feature |
|---|---|---|
| Momentum strategy, weight 0.25 | ~1089-1120 | `momentum_1m/3m` |
| Mean reversion (RSI), weight 0.20 | ~1123-1142 | `rsi_14` |
| VRP score, weight 0.25, plus EWMA-premium bonus +8 / +5 and VIX-extreme ±5 / -10 | ~1145-1220 | `vrp`, `ewma_vol`, `vxx_ratio` |
| Volume score, weight 0.15 | ~1180-1190 | `volume_ratio` |
| ADX ±3-6 | ~1269-1281 | `adx`, `trend_strength` |
| Sector momentum ±8 | ~1224-1232 | `sector_momentum` |
| risk_on / off ±3-5 | ~1235-1241 | `regime_score`, `markov_state` |
| News ±5-12 | ~1244-1256 | `news_sentiment` |
| Intel ±15 incl. insider MSPR ±5 | ~1295-1340 | `intel_score`, `insider_signal` |

The same input enters once directly and once through the model. The 60/40 weights were never fit for this overlap.

**Squeeze is counted three or four times:**
- squeeze strategy (weight 0.15)
- "Squeeze Watch" +10 (~1460)
- analyze `get_recommendation` "BUY CALLS" on squeeze +8 (~1449-1453; analyze.py ~1388)
- alt-data `squeeze_potential` through `alt_score`

**The "AI recommends" boost is direction-blind.** It adds +8 whenever `"BUY" in action` (bot_engine.py ~1451). That includes analyze's bearish **"BUY PUTS"** (Danger Zone, analyze.py ~1418) and the disabled "BUY CALLS" / "BUY CALL SPREAD". A bearish recommendation raises the long-stock score. The "+6 for SELL OPTIONS" branch can never match, because analyze emits "SELL PREMIUM".

**VRP raises a stock-buy score.** A high VRP adds up to 85 × 0.25 to the score and then labels the trade "SELL OPTIONS". The same evidence both qualifies the stock and diverts it to options.

**Sector and correlation are modelled four times:**
1. bot_engine `SECTOR_MAP` + yfinance count cap
2. risk_kill_switch `_SECTOR_MAP` 40% exposure
3. risk_kill_switch `CORRELATION_GROUPS` 60% (warning only)
4. position_sizing heat scalar

`is_ticker_blocked_by_correlation` is unused (risk_kill_switch.py ~399).

---

## 6.10 Suggested evidence checks (no changes made)

1. **N2:** Alpaca orders with `side=sell` on equities submitted 13:30-13:35Z with no prior long. Also look for MORNING-TRADE audit lines with "SELL".
2. **N16:** count `gamma_risk` closes vs CSP entries since 2026-10-13. Record the underlying price at each.
3. **N17:** Alpaca orders submitted on Sat/Sun or NYSE holidays for floor tickers, SVXY and OCC closes.
4. **N18:** holdings of SMH, KWEB and VXUS during any BEAR/PANIC floor regime (voltrade_floor_state.json history).
5. **KB #47:** same as open_questions.md ~7452. Note the audit retains about 3 hours, so use Alpaca fills.

## 7. Performance tracking

### 7.1 What is measured today, and where

| Metric | Where computed | Basis / caveats |
|---|---|---|
| Daily equity curve `{date, value, pnl}` | `server/bot.ts` `recordDailyEquity()` ~L1037-1058 | One row per UTC date from Alpaca `/v2/account` `portfolio_value`. `pnl = equity - last_equity`. Capped at 365 rows. **Weekend/holiday rows are written too** (175 rows vs 121 sessions since 2026-05-06), and their `pnl` repeats a stale broker delta (e.g. 10-24/10-25 both `-350`). The latest row is overwritten intraday. |
| Win rate, avg gain/loss, profit factor, best/worst, "totalPnlPct", per-strategy W/L | `server/bot.ts` `/api/bot/performance` inline `python3 -c` ~L1705-1800 (reads `voltrade_trade_feedback.json`) | Excludes `_seed` records and `pnl_pct is None`. **`totalPnlPct` is a sum of per-trade percents** (not compounded, not capital-weighted). "realisticPnlPct" subtracts `avg_slippage_pct * 2 * n` (L1776). No Sharpe or drawdown here; `currentDrawdown` comes from `state.equityPeak` (L1966), which was reset on 2026-10-06. |
| Expectancy | Not shown anywhere as one number; derivable from `/api/diag/ml` `live_performance` (`ml_model_v2.py` status path) | Live: 36 closed trades, win rate 27.8%, avg win +9.08%, avg loss −5.16% → **expectancy ≈ −1.20% per trade** (0.278×9.08 − 0.722×5.16). |
| Fill slippage | `ml_model_v2.py` `track_fill()` ~L2355 (computes `abs(fill−expected)/expected`) and `fills_slippage_stats()` ~L2627 | **The "fill_price" is synthetic.** Regular path `server/bot.ts` ~L6213-6225 adds a random volume-tiered haircut (>20M 0.02-0.05%, >5M 0.05-0.10%, >1M 0.08-0.15%, else 0.12-0.25%) on top of Alpaca's `filled_avg_price`. The morning-queue path ~L4474-4482 never reads a real fill: it computes `trade.price × (1 ± 0.05-0.25%)`. Slippage also uses `abs()`, so favorable fills count as cost. The "fills tracker" therefore measures the model's own random number plus quote drift, not execution quality (ROOT LADDER gate 5 is not measurable live). |
| SPY benchmark (since-start and 30-day) | `server/spyBenchmark.ts` `benchmarkVsSpy()`; `server/bot.ts` `benchmarkBody()` ~L1979 (owner `/api/bot/benchmark` + diag `benchmark`) | Total-return SPY (Alpaca `adjustment=all`). Only two windows: `since` and `d30`. No beta, alpha, IR, or tracking error. |
| After-tax benchmark, wash sales | `server/bot.ts` `afterTaxFor()` ~L2000 → `server/afterTaxBenchmark.ts`; `sleevesBody()` ~L2149 | Live: 775 wash sales, $19,855 disallowed. Estimated ST tax $2,816. |
| P&L by sleeve (options / floor / stocks), by options underlying, non-trade cash, equity reconciliation | `server/bot.ts` `sleevesBody()` ~L2149 (diag `sleeves`) | Realized plus unrealized from Alpaca fills. The reconciliation leaves a residual of −$1,292. |
| Shadow / counterfactual: win rate by decision bucket, masterkill reason, change-% band | `shadow_portfolio.py` `get_shadow_stats()` ~L1074 (diag `shadow`) | Old ruler: path-dependent +2%/−4% labels at +5/+10/+20d (`_label_from_path` ~L553). |
| Shadow bands: decision × band × score quintile × regime × action, plus `exit_aligned` (mirror of live exits) | `shadow_portfolio.py` `shadow_band_report()` ~L2151, `_exit_aligned_tables()` ~L2054, `backfill_exit_aligned()` ~L1808 (diag `shadow_bands`) | The exit-aligned ruler mirrors `checkPositionOnTick` and charges a 0.1% round-trip cost. |
| Shadow bands deduped (one first-seen record per ticker × ET day) | `shadow_portfolio.py` `shadow_band_report_deduped()` ~L2242, `get_shadow_band_report_deduped()` ~L2267; `server/diag.ts` `"shadow_bands_dedup"` ~L330 | **Not live.** It exists only in local commit `61938df9`, which is not on `origin/main`, and production returns **404 "unknown probe"**. |
| Backtest metrics | `backtest_v2.py` `compute_metrics()` ~L333-352; `simulate()` out dict ~L630-655; `run_backtest()` ~L657-727 (`spy_cagr`, `alpha` = CAGR difference) | **Only total return, CAGR, Sharpe and max DD**, plus `win_rate_pct` and `n_trades`. Sharpe uses population SD and no risk-free rate. No Sortino, Calmar, deflated Sharpe, turnover or exposure. There is no options model (equities only). Default `exit_model='legacy'` still simulates the dead config exits (KB #48). |
| Exit-model reconciliation with DSR bar and random-entry baseline | `scripts/exit_model_reconciliation.py` → `research/exit_model_reconciliation_2026-10-20.md` | This is the only place where a deflated-Sharpe-style bar and a base-rate "alpha vs random entry" exist. It is a one-off research script, not a standing metric. |
| Strategy-weight "LEARN" loop | `server/bot.ts` `adjustStrategyWeights()` ~L1010-1035 | Changes `strategyWeights` and audits "shifting weight toward VRP". **`strategyWeights` is never read** outside this function (grep: L859 definition plus writes only), so this is dead feedback (see §9). |

### 7.2 What is missing

1. **Risk-adjusted live metrics.** There is no live Sharpe, Sortino, Calmar, volatility, beta, alpha or information ratio anywhere: grep `sortino|calmar` hits only comments in `options_*.py`, and the client has no Sharpe. A session that wants them has to compute them by hand, as this audit did.
2. **Rolling windows.** The benchmark has only `since` and `d30` (`spyBenchmark.ts`). There is no 63d/126d rolling series, so CLAUDE.md GOAL #3 ("rolling performance vs. buy-and-hold SPY") is only partly met.
3. **Capital-weighted trade stats.** `/api/bot/performance` sums trade percents. A 1-share and a 500-share trade weigh the same.
4. **Attribution coverage.** Of 398 live feedback records, 342 are `orphan_exit` (86%) and 362 have null `pnl_pct` (`/api/diag/ml`). Per-trade win rate and expectancy rest on n=36, while sleeves (fills-based) show −$8.1k realized. The two rulers measure different populations.
5. **Real execution-quality measurement.** The synthetic haircut (see 7.1) overwrites `fill_price`. `alpaca_fill_price` is stored but no metric reads it against the decision-time quote.
6. **Live-vs-backtest divergence tracker (HONESTY METRIC).** No standing job compares live sleeve results with backtest expectations. Options-sleeve results stand against unreproducible backtest claims in comments: `options_execution.py` ~L317-321 and ~L1490-1494 cite "backtest_scenario_c_wf" / "backtest_cc_investigation" with OOS CAGR 40-42% and Sortino 3.3-3.4. **Neither script exists in the repo** (grep finds only those comments and tests).
7. **Regime-split live performance.** None exists. Shadow regime is 86% NEUTRAL_BULL and 14% NEUTRAL, so it cannot be regime-conditioned either.
8. **Backtest statistics.** No Sortino, Calmar, deflated Sharpe or Lo SE in `compute_metrics()`, and no options or CSP simulation. Pre-April history is also missing: `sleeves.preWindowFills = 372` (first fill 2026-04-06) while the equity curve starts 2026-05-06, so the first month of live trading has no equity-curve record.
9. **Audit retention.** `persistAudit` (`server/bot.ts` ~L1176) keeps about 2,000 rows (~3 market hours, KB #46), so "did X happen yesterday" cannot be answered from the audit log.

### 7.3 Live numbers (pulled 2026-10-27 ~19:10Z, read-only)

**Account (`/api/diag/account`):** equity $98,590.14, cash $13,714, long MV $89,109, short MV −$4,233, `equityPeak` (reset) $99,962, drawdown vs reset peak −1.37%, kill switch off.

**Benchmark (`/api/diag/benchmark`, server-computed):** since 2026-05-06 (174 days): **account −7.71% vs SPY +5.46% → gap −13.17 pp**. Last 30 days: account −5.55% vs SPY −0.18% → gap −5.37 pp.

**Own computation** (`equity_curve?days=365`, 175 rows filtered to the 121 SPY sessions; SPY from yfinance auto-adjusted closes; daily returns, √252, sample SD; script at `.../scratchpad/audit/eq.py`):

| 2026-05-06 → 2026-10-27 (120 daily returns) | Account | SPY (TR) |
|---|---|---|
| Total return | **−7.71%** | **+5.47%** |
| Annualized | −15.5% | +11.8% |
| Ann. volatility | 30.6% | 12.1% |
| Sharpe (rf=0) | −0.40 | 0.99 |
| Sortino | −0.55 | 1.46 |
| Max drawdown | **−18.63%** (peak $111,613 on 06-02 → trough $90,814 on 09-09) | −4.49% |
| Calmar | −0.83 | 2.63 |
| Beta to SPY / correlation | 1.21 / 0.48 | — |
| Jensen alpha (ann.) | **−26.6%** | — |
| Excess-return t-stat / IR | −0.62 / −0.89 | — |

Monthly (account vs SPY): May +3.17/+3.09, Jun +0.38/−1.03, Jul −3.81/+0.03, Aug −3.16/+2.68, Sep +1.06/−0.33, Oct (MTD) −5.33/+0.98. The single worst day was 2026-09-09: −$12,060, the KNOWN BROKEN #42 incident.

**Where the loss comes from (`/api/diag/sleeves`, window from 2026-05-07):**

| Sleeve | Realized | Unrealized | Note |
|---|---|---|---|
| options | **−$9,177** | +$108 | 822 fills, 298 symbols; Aug alone −$5,437. Worst roots: MUU −1,700, NVDA −1,126, PATH −972 |
| floor (QQQ/SMH/GLD/VXUS/KWEB basket) | +$1,554 | +$1,199 | The only positive sleeve. Oct −$2,875 |
| stocks | −$499 | 0 | 178 fills |
| non-trade (fees) | −$119 | — | |
| **Total** | **−$8,122** | +$1,307 | residual vs equity change −$1,292; after-tax realized −$10,938 |

**Shadow / counterfactual (`/api/diag/shadow`, `/api/diag/shadow_bands`; 20,000 records 2026-08-11 → 10-27):**
- Old ruler, +5d win rate: taken 41.1% (n 6,170) vs rejected_score 49.5% (n 8,995); +20d: 42.6% vs 55.3%.
- Exit-aligned ruler (mirrors live exits, net of 0.1%): taken mean net **−3.01%/candidate**, win 27.4% (n 4,154) vs rejected_score −2.51%, win 38.4% (n 4,098). Base rate across all candidates is −2.42%.
- Score quintile is **inverted**: Q1 (lowest scores) has win 51.5% and mean −0.77%; Q2-Q5 range from −1.6% to −3.1%. On the old ruler, Q5 +5d win is 33.6% vs Q1 56.2%.
- Caveats: (i) records are logged on every scan, so n counts copies. Exit-aligned "taken" is 4,154 labeled records but only **123 unique ticker × entry-days** (`coverage.taken.unique_ticker_entry_days`), and the dedup probe that fixes this is not deployed (404). (ii) The labels are long-stock outcomes, while many "taken" are option sales (action `unrecorded` before 2026-10-06). (iii) There is a single regime. (iv) The 20k cap means rotation is already discarding history: oldest is 08-11, though the comment at `shadow_portfolio.py` L81 says "~6 months". (v) `rejected_stock_slots` has 235 records with 0 labeled (all `not_yet_attempted`), probably because the bucket is new.

### 7.4 Honest verdict: does the system beat SPY after costs?

**No. There is no evidence that it does, and every available measure points the other way.**
- **Live, net of real paper fills and fees:** −7.71% vs SPY +5.46% over 5.7 months (−13.2 pp). It carried 1.2 beta and 2.5× SPY's volatility, and its max drawdown was 4× SPY's. Beta-adjusted alpha is about −27%/yr. The excess-return t-stat (−0.62) means the window is too short to prove underperformance statistically. It is far too short, and wrongly signed, to suggest outperformance. The losses are concentrated in the options sleeve, and the floor basket is the only positive contributor. After estimated tax the gap widens (after-tax realized −$10.9k).
- **Costs make it worse, not better.** Paper fills are optimistic relative to real execution, especially for options on the indicative feed. Live options and CSP results also stand against comment-cited backtests of 40% CAGR that cannot be reproduced.
- **Backtest history agrees.** `system_config.py` ~L127-134: the active system's best config (NEW_CSP_ONLY) returned **5.68% CAGR vs SPY 13.97%** over 10 years (`backtest_10yr_results.json`). The only "beats SPY" config in that comment (~L121-124, QQQ floor 18.7% vs 12.3%) is a passive QQQ-tilt result, which is beta and sector exposure, not alpha. The 2026-10-16 entry ("backtest_v2 GAINS A SELECTABLE LIVE EXIT MODEL", `research/experiments.md` ~L108777, v1.0.1088) showed that **stored backtests measured exits the bot does not run and were optimistic**. Switching to the live exit model lowered SPY momentum Sharpe 0.180→−0.005 (2y), 0.764→0.496 (3y), 0.431→0.312 (10y), and liquid-7 mean 0.476→0.400. The 10-20 reconciliation (`research/exit_model_reconciliation_2026-10-20.md`) found momentum alpha vs random entry **negative in every arm** (live arm −0.063%/trade) on a survivorship-biased large-cap universe that is optimistic for every arm.
- **The shadow book agrees.** Candidates the bot takes underperform candidates it rejects under both rulers, and higher scores do worse. The most likely reading is that the composite score has no edge, or a negative one, in the current regime. Deduplication and action-splitting are still needed before treating the magnitude as reliable.

---

## 8. UI

### 8.1 Shell and pages

- **Routing.** `client/src/App.tsx:AppShell` (~141-167) is a wouter router with public marketing routes plus `/app` and `/app/:rest*` → `AppDashboard`.
  - `AppDashboard` (~78-122) checks `/api/auth/me` and renders `pages/home.tsx:Home`.
- **Tabs.** `pages/home.tsx` defines the tabs: `TABS` (~29-39) and the switch (~510-556).

| Tab | Component | Auth | What it shows |
|---|---|---|---|
| analyze | `pages/analyze.tsx` (with sections options, smart-money via `InsightsView`, structure, etf-builder via `ETFBuilderView`) | public | single-ticker analytics and a scanner list |
| research | `pages/research.tsx` | public | AlphaDesk verdict |
| scanner | `pages/scanner.tsx` (+ `components/SectorHeatmap`) | public | raw Alpaca most-actives/movers snapshot and sector ETF heatmap. **Not the bot's picks** |
| news | `pages/news.tsx` | public | Alpaca news |
| watchlist | `pages/watchlist.tsx` | public (writes need a session) | user tickers + Alpaca snapshots |
| data | `pages/datamap.tsx` (16.6k lines; imports 50 data views ~28-77) | public | /data map and data-product views |
| planner, taxes | `pages/planner.tsx`, `pages/taxes.tsx` | public | `/api/plan`, `/api/tax/estimate` |
| bot ("AI Engine") | `pages/bot.tsx` → `bot/BotPage.tsx` | `requiresAuth` + `requiresOwner`; also gated in render (~535-536) | the trading cockpit |

**Orphan:** `components/TradingActivity.tsx` (polls `/api/trades/today` and `/api/orders/open`, ~228-238) is **not imported anywhere** in `client/src` (grep).

### 8.2 Bot cockpit (picks and signals as the owner sees them)

**Data hook.** `bot/useBotData.ts:useBotData` (~145-161). All reads are **polling REST**; none use SSE.

| Endpoint | Poll interval |
|---|---|
| status | 5 s |
| account | 10 s |
| positions | 15 s |
| audit | 10 s |
| performance | 60 s |
| market-status | 60 s |
| calendar | 10 min |
| `/api/ml/status` | 60 s |
| insight | 5 min (60 s client timeout) |
| last-scan | 60 s |
| `/api/orders/open` | 30 s |
| benchmark | 10 min |
| sleeves | 10 min, or 20 s while pending |

Each panel prints its own data age from `dataUpdatedAt` (BotPage ~300).

**Panels** (`bot/BotPage.tsx` ~302-314, bodies in `bot/panels.tsx`)
- **Watch** (`WatchBody` ~543; model in `bot/watch.ts`). The "picks" view. It renders `/api/bot/last-scan.top_10`: rank, score, factor breakdown (momentum / mean_reversion / vrp / squeeze / volume), sector clash, `new_trades`, `sizing_blocks`. It joins these with audit refusal/queue rows (SKIP, SIZE-CAP, HEAT-CAP, CORR-CAP, HALT-SKIP, DUP-SKIP, OPTIONS-SLOT-FULL, OPTIONS-SKIP, SIZE-BLOCK, QUEUE) and compares against `status.minScoreThreshold` (watch.ts header ~1-7).
  - Live versus stale: `lastScanResult` is in memory, so after any restart the route returns `{message:"No scan run yet..."}` until the first Tier-2 scan (bot.ts ~8336-8338). Overnight it shows the last daytime scan, because Tier 2 is dark from 20:00 to 04:00 ET.
- **Look-up** (`LookupBox`/`LookupCard` ~207-262). `/api/bot/lookup/:symbol` (daemon `lookup_ticker`, 90 s RPC, 120 s client) and `/api/bot/lookup-ledger` (BotPage ~186-190).
- **State / Decision / Scan strip / Incident / Day review** (~315-411). Built from last-scan, audit, status and benchmark.
- **Model** (`ModelBody` ~415). `/api/bot/insight` (`ml_insight.py`: model age, blend weights, recent ML trace, shadow score bands).
- **Chart** (`ChartBody` ~490, `bot/chart/BotChart.tsx` ~44-64). `/api/bot/bars/:ticker` (Alpaca bars; empty for OCC symbols, bot.ts ~3608-3660) for positions and watch candidates.
- **Book** (`BookBody` ~634, `SleeveSummary` ~577). `/api/bot/positions` (Alpaca positions + `voltrade_stop_state.json` levels via an inline Python subprocess, bot.ts ~1287-1445) and `/api/bot/sleeves`.
- **Orders** (`OrdersBody` ~661). `/api/orders/open`.
  - **[DEFECT, code]** `routes.ts` `/api/orders/open` (~7154-7172) responds `{ orders: [...] }`. `useBotData` types the response as `OpenOrder[]` and passes it through unchanged (~156).
  - `OrdersBody` checks `orders.length`. On the wrapper object that is `undefined`, so `!orders.length` is true and the panel always shows "none working".
  - `BotPage` computes `openOrders: d.orders.data.length` → `undefined`, so the stage row shows "? open" (BotPage ~130, derive.ts ~92).
  - Net effect: working orders never display in the cockpit. **[unverified-runtime]**
- **Timeline / Stages / Issues** (~263-314, ~674). From `/api/bot/audit` (last 100 SQLite rows).
- **KPI strip** (~84). perf, account, status, market, calendar, benchmark.
- **Controls.** `/api/bot/start`, `/api/bot/stop`, `/api/bot/kill` (a toggle) (BotPage ~207-211).

**Signals list.** The in-memory `signals` array, refreshed every Tier 2 (bot.ts ~4931-5010), is served at `GET /api/bot/signals` (~4008). **No client code reads `/api/bot/signals`**. The cockpit derives picks from last-scan instead.

### 8.3 Non-bot analytics pages (public)

- **Analyze tab** (`pages/analyze.tsx`).
  - `/api/analyze/:ticker` (~2077-2079) → `python3 analyze.py TICKER`, computed live per request.
  - `/api/scan?top_n=10` (~1872-1874) and `/api/scan/progress` (~1733-1735) → routes.ts in-memory `tier1Cache` / `fullUniverseCache`. These are pre-warmed with 10 tickers at +3 s (~7320-7329) and refreshed in the background only when a request finds the cache older than 5 minutes. The periodic `setInterval(refreshTier1)` is commented out (~564).
  - This scanner is **separate from the bot's scan** and does not show bot picks.
- **Smart-money** (`InsightsView` ~672-674) → `/api/insights/:ticker` → `insights.py`.
- **ETF Builder** → `/api/etf/:ticker` → `etf_analyzer.py`.
- **Research** → `/api/research/:ticker` → alphadesk.
- **Scanner** → `/api/market/scanner` (Alpaca screener + snapshots, ~6918-6966) and `/api/market/sectors` (~6969-6980).
- **News** → `/api/market/news` (~7000-7023).
- **Watchlist** → `/api/watchlist*` (SQLite) and `/api/market/snapshots`.

### 8.4 /data surface (signals vs raw overlays)

- `datamap.tsx` has 87 literal `/api/data/` references. It mounts per-stream views (~28-77): signal ladder, streams, quality dashboard, pipeline health, COT, TFF, attention, wiki-attention signal, earnings language, filings, etc.
- Signal-strength and health views read pre-compiled or cached aggregates:
  - `/api/data/signal-ladder` → `loadSignalLadder()` over `datacore/signal_ladder.json` (routes ~1739-1743)
  - `/api/data/streams` → `getStreamsInventoryCached()` (~1707-1713)
  - `/api/data/quality-dashboard` (~1721-1733)
  - `/api/data/pipeline-health-dashboard` → JSONL health history (~1752-1765)
- None of these feed the trading bot. The datacore boundary is enforced separately (`test_spinout_boundary.py`, not audited here).
- Route counts in routes.ts: 125 `/api/data/*`, 60 `/api/v1/*` (external API product), 36 other.
- Thirteen `/api/data/*` paths have no literal string match in `client/src`. They may be built dynamically, so **[unverified]**: `aircraft/global_scopes`, `archive/stats`, `plants-under-alerts`, `3dtiles/{status,root,proxy}`, `aircraft/hexes`, `query`, `satellites`, `rail-traffic`, and the `crop-conditions`, `appstore-rankings` and `github-activity` `/history` routes.

### 8.5 Route table (trading, bot and analysis routes)

Handler location is `bot.ts` unless marked `r:` (routes.ts). "Owner" means `requireOwner`; "session" means `_checkSession`.

| Route | Handler (~line) | Data source | UI consumer |
|---|---|---|---|
| GET /api/bot/stream (owner) | ~1228 | SSE of `audit()` entries | **none** |
| GET /api/bot/account (owner) | ~1239 | Alpaca `/v2/account`; also evaluates the DD kill | useBotData account |
| GET /api/bot/positions (owner) | ~1287 | Alpaca `/v2/positions` + stop_state/BASE_CONFIG (inline py) + options state | useBotData positions (Book, Chart) |
| GET /api/bot/history (owner) | ~1450 | Alpaca closed orders | none found |
| GET /api/bot/notifications, POST …/read (owner) | ~1474/1478 | in-memory `notifications` | none found |
| GET /api/health | ~1484 | composite checks; HTTP code from `healthGate.ts` | Railway healthcheck |
| GET /api/bot/performance (owner) | ~1705 | `trade_feedback.json` (seeds excluded) + `fills_slippage_stats` + equityCurve | useBotData performance (KPI) |
| GET /api/bot/benchmark (owner) | ~2009 | equityCurve + Alpaca SPY daily bars (`adjustment=all`) | useBotData benchmark |
| GET /api/bot/lookup/:symbol (owner) | ~2050 | daemon `lookup_ticker` (subprocess fallback) | BotPage look-up |
| GET /api/bot/lookup-ledger (owner) | ~2067 | daemon `lookup_ledger_report` | BotPage |
| GET /api/bot/sleeves (owner) | ~2163 | Alpaca FILL activities + positions (sleeveLedger.ts), tax ledger | useBotData sleeves |
| GET /api/bot/export-trades, POST /api/bot/backup | ~2173/2211 | files | none found |
| GET /api/system/snapshot | ~2255 | daemon health; inline py (macro, stress, probability, csp universe); RPC `snapshot_python_state` (unregistered, subprocess fallback) | none found |
| GET /api/system/health-check, POST /api/system/verify-change | ~2494/2535 | composite | none found |
| GET /api/bot/status (owner) | ~2560 | in-memory `state`, autoResume, SQLite count | useBotData status (5 s) |
| POST /api/bot/start, /stop, /kill, /circuit-breaker/reset (owner) | ~2606-2641 | state mutation; kill also `DELETE /v2/orders` | BotPage controls (start/stop/kill) |
| GET /api/bot/audit (owner) | ~2651 | SQLite `audit_log`, last 100 | useBotData audit (10 s) |
| GET /api/diag/:probe (`DIAG_TOKEN`) | ~2679 | whitelist in `server/diag.ts:DIAG_PROBES` (~20-60) | sessions, not UI |
| POST /api/bot/trade, /api/bot/close (owner) | ~3520/3593 | manual orders | none found in client |
| GET /api/bot/bars/:ticker (owner) | ~3608 | Alpaca bars | BotChart |
| GET /api/daemon/health | ~3665 | RPC `health` | none found |
| GET /api/monitoring/overview, /tier-pnl, /drawdown, /caches (owner) | ~3684/3753/3858/3886 | RPC `risk_status`, `ml_status`, `strategy_pnl_stats` (unregistered), `get_peak_equity`, `cache_inventory` | none found |
| GET /api/bot/market-status | ~3912 | Alpaca clock | useBotData market |
| GET /api/bot/signals; POST /api/bot/refresh-signals (owner) | ~4008/4013 | in-memory `signals`; refresh runs analyze.py on 10 tickers | **none** |
| POST /api/bot/backtest (owner) | ~4021 | backtest.py | none found |
| GET /api/bot/last-scan (owner) | ~8336 | in-memory `lastScanResult` | useBotData lastScan (Watch, Decision, Scan strip) |
| GET /api/bot/calendar (owner) | ~8341 | Alpaca `/v2/calendar` + hardcoded 2026 holiday names | useBotData calendar |
| POST /api/bot/run-now (owner) | ~8520 | runs `tier2Intelligence` directly | none found |
| GET /api/bot/diagnostics (owner) | ~8533 | diagnostics.py | none found |
| GET /api/bot/insight (owner) | ~8559 | `ml_insight.py` (2-min cache; 10 s trivial-kill, see 1.2) | useBotData insight (Model) |
| GET /api/bot/ml-status (owner) | ~8580 | inline py over model/feedback files | none found (UI uses `/api/ml/status`) |
| GET /api/watchlist, POST …/add, …/remove | r:~612/626/649 | SQLite `watchlists` | watchlist.tsx |
| GET /api/analyze/:ticker | r:~668 | analyze.py (no mutex) | analyze.tsx |
| GET /api/insights/:ticker | r:~706 | insights.py | InsightsView.tsx |
| GET /api/research/:ticker | r:~747 | alphadesk | research.tsx |
| GET /api/bot/system-status | r:~6782 | system_status.py (not owner-gated) | none found |
| GET /api/scan, /api/scan/progress | r:~6804/6848 | analyze.py fan-out cache | analyze.tsx |
| GET /api/market-snapshot, /api/news | r:~6858/6890 | Polygon | none found for market-snapshot; `/api/news` string appears only in newsletter pages (likely a newsletter route prefix; unverified) |
| GET /api/market/scanner, /sectors, /snapshots, /news | r:~6918-7000 | Alpaca data API | scanner.tsx, SectorHeatmap.tsx, watchlist.tsx, news.tsx |
| GET /api/trades/today (session) | r:~7052 | Alpaca closed orders + positions | TradingActivity.tsx (**orphan**) |
| GET /api/orders/open (session) | r:~7154 | Alpaca open orders, **wrapped `{orders}`** | useBotData orders (**shape mismatch, see 8.2**), TradingActivity (orphan) |
| GET /api/positions, /api/trades/history, /api/shorts/dashboard (session) | r:~7175/7219/7196 | Alpaca / intraday_shorts | none found |
| GET /api/ml/status | r:~7334 | `python3 ml_status.py` | useBotData ml |
| POST /api/ml/toggle, /api/ml/retrain (session) | r:~7352/7369 | ml_toggle.py / ml_retrain_safe.py (300 s, plain `execAsync`) | none found |
| GET /api/etf/:ticker | r:~6627 | etf_analyzer.py → etf_data_sources | ETFBuilderView.tsx |
| POST /api/plan, /api/tax/estimate | r:~6601/6574 | server modules | planner.tsx, taxes.tsx |

"none found" means no literal match for the path in `client/src` (excluding tests).

---

## 9. Dead code and bloat

### 9.1 Config keys with no reader (`system_config.py` `BASE_CONFIG`)

Method: every `"KEY":` in `system_config.py` was grepped across `*.py/*.ts/*.tsx` (excluding tests, node_modules and .claude). Dynamic f-string readers were then removed from the list: `SPY_FLOOR_{regime}` (`bot_engine.py` ~L5223), `DEFENSIVE_FLOOR_{regime}` (~L5638), `STOP_SCALAR_{regime}` (`bot_engine.py` ~L2196, `position_sizing.py` ~L1074, `shared/positionLevels.ts` ~L252), and the `REGIME_VXX_PANIC/ELEVATED/CAUTION`/`REGIME_BEAR_THRESHOLD` keys that `system_config.get_market_regime()` (~L418-426) reads internally. **36 of 117 keys** remain with no reader anywhere outside their own definition:

| Group | Keys with zero readers | Evidence |
|---|---|---|
| Exit config (KB #48) | `ATR_TP_MULTIPLIER`, `MIN_TP_PCT`, `MAX_TP_PCT` | Zero hits outside `system_config.py`. `ATR_STOP_MULTIPLIER`/`MIN_STOP_PCT`/`MAX_STOP_PCT` are referenced only as *copied constants* in `backtest_v2.py` ~L141-148 (legacy model), and `TIME_STOP_DAYS` only in mirrors (`live_exit_policy.py` L70 uses its own 7). The live executor `server/bot.ts` `checkPositionOnTick` (~L7281) reads none of them. |
| Scheduler cadences | `TIER1_INTERVAL_MS`, `TIER2_AFTERHOURS_MS`, `TIER2_OVERNIGHT_MS`, `TIER3_INTERVAL_MS` | bot.ts hard-codes its own cadences |
| Stream tuning | `STREAM_BAR_CHANGE_PCT`, `STREAM_COOLDOWN_MS`, `STREAM_MOMENTUM_PCT`, `STREAM_TICKERS_N`, `STREAM_VOL_SPIKE_RATIO` | `STREAM_VOL_SPIKE_RATIO` is even set per regime in `get_adaptive_params()` ~L464, and nothing reads it |
| Cache TTLs | `CACHE_ALT_DATA_HOURS`, `CACHE_INSIDER_HOURS`, `CACHE_INTEL_SECONDS`, `CACHE_MACRO_SECONDS`, `CACHE_SOCIAL_HOURS`, `CACHE_YF_SECONDS` | modules hard-code their own TTLs |
| Sizing | `KELLY_DIVISOR`, `MAX_TOTAL_CAPITAL_PCT`, `MIN_POSITION_PCT` | |
| ML | `ML_CONFIDENCE_THRESH`, `ML_RETRAIN_TRADES`, `ML_TARGET_HORIZON` | (`ML_FEATURE_COUNT` is read by a test only) |
| Options | `OPTIONS_SCALE`, `OPTIONS_SLIPPAGE` | |
| Misc | `CIRCUIT_BREAKER_HOURS`, `CIRCUIT_BREAKER_STOPS`, `STOP_COOLDOWN_SECONDS`, `DEEP_SCORE_TOP_N`, `DEEP_SCORE_WORKERS`, `SCREENER_TOP_N`, `SCREENER_MOVERS_N`, `OPEN_HOUR_MIN_VOLUME`, `REGIME_SPY_MA_PERIOD`, `REGIME_VXX_BULL` | |
| Partly dead | `MAX_CHANGE_PCT` | read only by `shadow_portfolio.py` (reporting band) and `server/diag.ts` comments, never as a live gate (KB #10 history). `SCORE_BAND_MAX` was removed from config but still appears in `diag.ts` comments. |

### 9.2 Rules and logic that exist but never execute (or are fed constants)

| Item | Location | Evidence |
|---|---|---|
| Python-only exit rules: 60-min min hold, 4-day short-hold grace, −20% `POSITION_HARD_STOP_PCT`, regime stop scalar | `bot_engine.py` `manage_positions` (~L2085-2197) | KB #48 (`research/open_questions.md` ~L7460): Node's `checkPositionOnTick` places every stock exit, and these Python rules never trigger an order. `POSITION_HARD_STOP_PCT`/`STOP_SCALAR_*` reach bot.ts only for **display** (`server/bot.ts` ~L1316 → `shared/positionLevels.ts`). |
| Sync TIME-EXIT (>14d, −5..+2%) | `server/bot.ts` sync path | KB #48 says it is "probably dead": it reads `pos.created_at`/`submitted_at`, which Alpaca position objects lack. Unverified live. |
| Phantom first scale-out (state written with no order) | `bot_engine.py` ~L2085-2091, adopted by bot.ts ~L6580-6595 | KB #47. Live frequency not measured. |
| `strategyWeights` "LEARN" loop | `server/bot.ts` L859 definition, `adjustStrategyWeights()` ~L1010-1035 | The weights are mutated and audited ("shifting weight toward VRP/squeeze") but **read by nothing**. The audit message misleads. |
| ML features pinned to constants at inference | `bot_engine.py` `ml_features` ~L1767-1779: `intel_score`, `news_sentiment`, `insider_signal` = 0.0 and `days_to_earnings` = 0.5 | Training also uses constants (`ml_model_v2.py` `_compute_features` ~L546, ~L554). **4 of 34 features carry no information.** The real `_days_to_earnings` is computed (~L1636-1655) and then discarded. |
| ML "options" features that are VXX proxies | `ml_model_v2.py` ~L459-461, ~L543, ~L550-551, ~L576 | `vrp`, `iv_rank_proxy`, `put_call_proxy`, `iv_rank_stock`, `put_call_ratio` are all derived from VXX in training. `put_call_ratio = vxx_close/20` is not a put/call ratio. **Train/serve skew:** at inference `vrp` = the per-stock ATM IV − RV20 from `analyze.compute_vrp()` (`analyze.py` ~L559) via `get_stock_details` (`bot_engine.py` L999, L1743), and `iv_rank_proxy`/`iv_rank_stock` = per-stock HV rank (`bot_engine.py` `_compute_stock_iv_rank` L20, L1744, L1629). Training uses (vxx_ratio−1)×50 clamped to ±20 and the VXX 52w rank. |
| ML retrain subprocess under the 10s "trivial" kill | `server/bot.ts` ~L6293 and ~L6776 `execPythonSerialized("python3 ml_retrain_safe.py", {timeout: 900000})` | `isTrivialPython()` (~L244) returns true for any command of 150 chars or fewer without an `import numpy…` substring, so this 26-char command gets `min(90000,10000)` = **10s SIGKILL**. The "15 minutes headroom" comment is false. It was flagged "unconfirmed" in `research/experiments.md` ~L108772. The model is fresh live (`model_age_hours` 5.8), so another path (the daemon) probably carries the retrain. |
| Unlabeled counterfactual bucket | `rejected_stock_slots` (`bot_engine.py` ~L3416, `server/bot.ts` ~L5791) | 235 records, 0 labeled at any horizon (`/api/diag/shadow`). Probably recency, but worth confirming. |
| Kill-switch thresholds | `risk_kill_switch.py` | Still **not counterfactual-logged** (`open_questions.md` RULE COST AUDIT ~L7530). |

### 9.3 Modules never imported at runtime

| Module | Lines | Runtime importers | Status |
|---|---|---|---|
| `ml_model.py` (v1) | 1,149 | **none**. Only `test_auto_discovery.py` L131 references it. `diagnostics.py` uses the *name* "ml_model" for a file-freshness check. | Dead. Superseded by `ml_model_v2.py` (`track_fill` compat comment ~L2259, ~L2574). |
| `sim_trading_day_v2.py` | 334 | none (CLI) | No test references it. |
| `bot_backtest.py`, `bot_backtest_subperiods.py` | 239 + 121 | each other only (CLI) | Superseded by `backtest_v2.py`, though the tests (`test_bot_backtest_subperiods.py`) still exist. |
| `cleanup_feedback.py` | 90 | none | Superseded by `feedback_boot_cleanup.py` (used at `server/bot.ts` ~L8450). |
| `system_status.py` | 278 | `server/routes.ts` only | **No test** references it. |
| `scripts/` (179 files) | — | **not in the Docker image** (KB #44, `Dockerfile` production stage copies `*.py`, `strategies/`, `alphadesk/` only) | Any runtime/diag path importing `scripts/*.py` 500s (`insider_cusum_gate2`). Guard at `server/bot.ts` ~L3498-3507. |
| `strategies/{momentum,mean_reversion,squeeze}.py` | — | `backtest_v2.py` `score_candidate()` and `bot_backtest.py` only; **not imported by `bot_engine.py`** | The live scorer is `deep_score()`. The "tournament home" is a backtest-only scorer, which is another source of live/backtest divergence. |

### 9.4 Data modules: reached, but unvalidated

| Module | Reached? | Effect on trading | Ladder status |
|---|---|---|---|
| `intelligence.py` | Yes, in `bot_engine.deep_score` `_fetch_intel` ~L844 | `intel_adjustment` up to ±15 on `combined_score` (~L1295-1340, plus finnhub ±5/+3) | No gate-2 validation for the composite. ML feature versions are pinned to 0. |
| `alt_data.py` | Yes, `_fetch_alt` ~L851 | `alt_adjustment` ±20 (~L1343-1371): wiki, short interest, congressional, geopolitical, FRED | Unvalidated in live scoring. This violates the ladder doctrine for live trading. |
| `social_data.py` | Yes, `_fetch_social` ~L858 | `social_adjustment` ±15 (~L1374-1400): reddit, Google Trends, news | Unvalidated |
| `institutional_data.py` | Only via `insights.py` (~L151) → `server/routes.ts` (user-facing insights) | none on trading | 13F display only |
| `finnhub_data.py`, `macro_data.py` | Yes | insider/analyst ±5/+3; macro adjustments | Mixed |

Live `/api/diag/scanner` shows `dataSourceErrors: {}`, so these fetchers are returning data and moving scores by up to roughly ±55 points combined on a 0-100 scale. None has passed ROOT VALIDATION LADDER gate 2. This contradicts the ANTI-GOALS and gate rules, and may help explain the inverted score-quintile result in §7.3.

### 9.5 Duplicate implementations

| Concern | Copies | Evidence |
|---|---|---|
| **Stock exit model** | **5**: (1) `server/bot.ts` `checkPositionOnTick` ~L7281 (the only executor); (2) `bot_engine.py` `manage_positions` (writes state, KB #47); (3) `live_exit_policy.py` (backtest mirror, imported by `backtest_v2.py` L39); (4) `shadow_portfolio.py` `XA_*` constants and `_label_exit_aligned` ~L1198-1537 (shadow mirror, **does not import** `live_exit_policy`); (5) `shared/positionLevels.ts` (display, mirrors *manage_positions*, not the executor). Plus the dead config block and the legacy backtest constants. | KB #48. The 10-16 entry's FOLLOW-UP (2) "de-duplicate the two mirrors" is still open. Drift is held off only by fuzz tests (`test_shadow_exit_aligned.py`, `test_backtest_v2_exit_model.py`). |
| **EDGAR / SEC HTTP clients** | ~13 runtime: TS `server/edgarForm4.ts`, `server/sec8kEarnings.ts`, `server/edgar13f.ts`, `server/secMidas.ts`; Py `alt_data.py`, `intelligence.py`, `institutional_data.py`, `sec_form4_bulk.py`, `sec_filings_text.py`, `sec_filings_universe.py`, `sec_filings_classify.py`, `alphadesk/alphadesk/edgar.py` + `providers.py`; plus 5+ in `scripts/` | Each has its own UA, rate limiting and timestamp handling. `sec_filings_text.py` found that `acceptanceDateTime` "Z" is not reliably UTC (experiments 2026-10-16). The other clients have not been checked for the same lookahead. |
| **IV / skew / VRP math** | `analyze.py` (`black_scholes`, `compute_vrp` ~L559, `compute_skew` ~L605); `vol_surface.py` (`implied_vol`, SABR, `_compute_atm_iv` ~L986, `analyze_skew` ~L1021); `options_scanner._fetch_iv_rank` ~L532; `bot_engine._compute_stock_iv_rank` L20 (HV rank labeled "IV rank"); `server/chainIv.ts` (IV30); `alphadesk/alphadesk/options.py` | Four or more ATM-IV definitions. "iv_rank" means HV-rank in one place and IV/HV blend in another. |
| Regime classifiers | `system_config.get_market_regime`, `regime_util.py`, `markov_regime.py`, `regime_inputs.py`, `ml_model_v2` regime classification, `backtest_v2.regime_series` | The shadow labeler calls regime without 200-day inputs (noted in its own `not_modelled`). |
| P&L rulers | `/api/bot/performance` (feedback, sum of %), `sleevesBody` (fills), `benchmarkBody` (equity curve), `reconstruct_pnl` (`server/reconstructPnl.ts`), ML probe `live_performance` | These give different answers. Example: feedback n=36 vs 1,000+ fills. |

### 9.6 Known-broken items still open (from `research/open_questions.md` KNOWN BROKEN)

#12 (2 gated follow-ups, ~L757), #20 master_kill_switch over-kill (~L107), #30 options slot overflow, 3rd recurrence, structural fix in wishlist (~L4713), #35 POS-KILL no duplicate guard (~L5232), #36 experiments.md ordering breaks the thrash-ratio check (~L5322), #40 (`test_options_v134_fixes` ~L6007), #45 CSP earnings gate fail-open, path (c) unconfirmed (~L7328), #46 deploy SIGTERM cancel-all defeats the CSP brake (~L7370), #47 phantom scale-out (~L7436), #48 three exit models (~L7460).

---

## 10. Gaps for an options layer (dealer GEX, P/C, IV skew/term, expected move, OI walls, unusual volume)

### 10.1 What already exists to build on

- **Raw archive (gate-1 material):** `server/optionsChainArchive.ts` stores a daily post-close (16:15 ET) snapshot of up to 120 underlyings, expiries of 60 days or less, strikes within ±20% of spot. That is about 30k rows/day since **2026-07-05**, with fields `s,u,t,bid,ask,bsz,asz,iv,delta,oi,feed="indicative"` (`ChainRecord` ~L96-108, `compactContract` ~L112-124; manifest `datacore/manifests/optionchains.json`). SPY/QQQ/IWM were pinned only from 2026-10-20 (`CHAIN_ALWAYS` ~L39).
- **Derived-feature pattern:** `server/chainIv.ts` (IV30 by total-variance interpolation, write-once derived cache `optionchains/derived/iv30/<day>.json`, `IV30_MAX_COMPUTE_PER_CALL=15`) and `server/chainCspEntry.ts`. They are exposed via diag probes `chain_iv30` and `chain_csp_entry` (`server/diag.ts` ~L60-75). Research test 2 used this route (`research/vrp_single_name_vs_index_2026-10-20.md`: single-name VRP t=2.15; index VRP not measurable).
- **Live chain/IV code in the trading path:** `options_scanner.py` (`_fetch_options_chain` ~L283, `_fetch_iv_rank` ~L532, setups ~L720-1297), `vol_surface.py` (surface, skew, SABR), `instrument_selector._score_options` ~L720, `options_execution.select_contract` ~L377.

### 10.2 Missing raw fields (the cheapest, highest-leverage fix)

| Feature | Needs | Archive today | Gap |
|---|---|---|---|
| Dealer gamma / GEX, gamma flip | per-contract gamma, OI, spot | delta, iv, oi; **no gamma, no spot** (spot is `null` for pinned/position names, `underlyingsFor` ~L45) | Keep `snap.greeks.gamma` (and vega) in `compactContract`, and record spot per underlying. Gamma can otherwise be recomputed from iv + spot with BS (`vol_surface.bs_*`). GEX sign convention (dealers short puts, long calls) is an **assumption** and must be labeled an estimate (BUILD-FIRST HONESTY CLAUSE). |
| Put/call ratio (volume and OI) | contract daily volume, OI | OI only | Add `snap.dailyBar.v` (volume). P/C *OI* ratio is computable today. |
| IV skew (25Δ risk reversal), term structure (IV30/IV60, front/back) | iv, delta per expiry | yes | Computable today, following the `chainIv.ts` pattern |
| Expected move | ATM straddle mid or IV30·√T, spot | bid/ask/iv | Needs spot. Straddle-based version is computable today. |
| OI walls / max pain | strike, OI | yes (strike from OCC) | Computable today |
| Unusual volume (vol/OI, vol vs trailing) | volume history | **no volume** | Needs the volume field, then ≥20 days of accumulation before z-scores mean anything |

Universe caveat: the daily list comes from the **CSP candidate cache plus open positions**. It rotates (1,717 distinct underlyings across 76 days), so per-name time series are sparse and selection-biased toward names the bot already favors. A feature panel needs a **fixed panel** (index ETFs plus a stable liquid set plus a declared small-cap set), which competes with the 120 cap.

### 10.3 Candidate insertion points, ranked by least disruption

| # | Insertion point | Territory | Disruption | Constraints / why |
|---|---|---|---|---|
| 1 | **Datacore derived stream**: extend `compactContract` (+gamma, +vol, +spot; bump `schema_version` in `datacore/manifests/optionchains.json`), and add `server/chainFeatures.ts` (GEX, P/C OI and volume, RR25, term slope, expected move, OI walls) with a write-once derived cache like `chainIv.ts` | T-DATACORE | **Lowest.** Pure TS, no trading imports (`test_spinout_boundary.py` enforces this), no Python, no subprocess caps. Server-side aggregation avoids the diag 5,000-row cap. | Features are as-of the **16:15 ET snapshot**, so trading use must be T+1 (no same-day lookahead). Feed is INDICATIVE, not NBBO: label it. Stays raw/gate 0 until gate 2. Adding fields to new rows only leaves old rows without gamma/vol (forward-only history). |
| 2 | **Read surface**: a diag probe `chain_features` (one row per underlying-day, same posture as `chain_iv30`) for gate-2 research, then `/api/data/options-features` (kind:"raw" → "signal" only after gate 2) and an `/api/v1` mirror | shared `server/routes.ts`, `server/diag.ts` | Low | RAW vs SIGNALS rule: interpreted readings like GEX "regime" are SIGNALS, so they stay gated until gate 2. `scripts/` is not in Docker (KB #44): the gate-2 harness runs in a sandbox against the probe (the pattern of `scripts/vrp_single_name_vs_index.py`). It must never be imported by a runtime probe. |
| 3 | **Gate-2 statistical test** (e.g. GEX sign vs next-5d realized vol, RR25 vs forward return, expected move vs realized move, which also serves as a calibration check) | research scripts | Low | ROOT LADDER gate 2 must come before any trading use. With about 80 sessions of history, power is low. Pre-register one theory-motivated spec per feature (REASONING #4). An index-level test is possible only from 2026-10-20. Databento history is still a wishlist/pilot decision (`research/wishlist.md` ~L2037, ~L2413). |
| 4 | **Shadow testing**: attach options features to `shadow_portfolio.log_candidate(features=…)` (~L270-345; `features` is a free-form dict, no column assert) | T-BOT | Low-medium | The bot must consume the datacore value the way an external customer would (boundary rule), via an HTTP/internal API read, cached per scan; it cannot import TS. The shadow log is capped at 20k and already rotating (~2.5 months), with one record per scan (duplicates), so evaluate on `shadow_band_report_deduped` (not yet deployed). Labels are long-stock outcomes, which is a mismatch for option-sale candidates. Use `exit_aligned` with action split. |
| 5 | **Options execution choices** (WHAT, not HOW): `options_scanner` setups (`_setup_high_iv_premium_sale` ~L977, `_setup_csp_normal_market` ~L1124), `instrument_selector._score_options` ~L720, strike choice in `options_execution.select_contract` ~L377 (e.g. avoid strikes inside an OI wall, size by expected move) | T-BOT | Medium | `submit_options_order` and the raw order POSTs are **FROZEN**. RULE REVIEW needs counterfactual or ablation evidence. **`backtest_v2.py` has no options model**, so LOGIC gate 3 cannot run without building an options sim over the chain archive (forward-only) or Databento. The cited options backtests in `options_execution.py` comments are not in the repo. Deploys SIGTERM-cancel all open orders (KB #46, `gracefulShutdown` ~L821-836), so any resting-limit options logic is disrupted by every merge. |
| 6 | **`deep_score` adjustment** (the ±15 intel/alt/social block pattern, `bot_engine.py` ~L1295-1400) | T-BOT | Medium (code is easy) | It is the **easiest code insertion and the worst practice**: the existing alt/social/intel adjustments are exactly the unvalidated-signal pattern not to copy. Gate 2 and gate 3 first. MEASUREMENT INTEGRITY: a scoring change and any ruler change go in separate PRs. Bump `package.json` for `code_version` attribution. |
| 7 | **ML features**: `ml_model_v2.FEATURE_COLS` (L140-184, `assert len == 34` at L186; pinned by `test_audit_critical.py` L21-27 and `system_config` `ML_FEATURE_COUNT` L278), `_compute_features` ~L364 (training), `bot_engine.ml_features` ~L1730-1782 (inference), `OPTIONS_FEATURE_COLS` ~L1905 (trained on *synthetic* VXX/SPY data, `_build_options_synthetic_training`) | T-BOT | **Highest** | Adding columns breaks the 34-count assert and test pins (changing a pin is not weakening, but must be explicit). The less disruptive path is **replacing the 5 VXX proxy slots and the 4 constant slots** (§9.2) with real options features, keeping 34. That still needs per-ticker options history for every training bar (6 years × 200 tickers), and the archive has about 80 days. Without that, real values at inference would recreate the train/serve skew that already exists for `vrp`/`iv_rank_*`. So ML integration has to wait for Databento-grade history or for an ML-on-archive window, and must follow the gates. Retrain heavy work: daemon 1 GB RSS cap (`voltrade_daemon.py` L77), 90s subprocess cap and a 10s kill for short "trivial" `python3 -c` commands (`server/bot.ts` L244-256). |

### 10.4 Hard constraints checklist for any options-layer PR

- **Spinout:** chain features live in `server/*.ts` / `datacore/` with zero imports from `bot.ts`, `bot_engine`, `system_config` or `strategies` (`test_spinout_boundary.py`).
- **Ladder:** data (indicative quotes vs the Databento pilot ground truth) → signal → logic → sizing → execution. Nothing reaches `deep_score`, the ML layer or `select_contract` before gate 2. Free estimates such as GEX are labeled "estimate".
- **Measurement integrity:** any new ruler (options sim, P/C label) gets its own [RULE-REVIEW] PR with before/after on identical inputs.
- **Frozen:** `submit_options_order`, order POST paths, `Dockerfile` (so `scripts/` stays out of runtime), `run_with_daemon.sh`, `risk_kill_switch` mechanisms.
- **Runtime budgets:** Python subprocess hard cap 90s, and short commands under 150 chars without an `import numpy…` substring are killed at 10s (`isTrivialPython`). The daemon self-kills over 1 GB RSS. Heavy per-day aggregation (30k rows) therefore belongs in Node with derived caches, not in the scan loop.
- **Operational:** every deploy cancels all open orders (KB #46), and audit retention is about 3 hours (KB #46). Options features must be T+1 relative to the 16:15 ET snapshot.
