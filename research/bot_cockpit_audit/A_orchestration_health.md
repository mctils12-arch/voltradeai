# A — Orchestration, Scheduling, Liveness, Health, Alerting, Broker/Daemon/WebSocket Plumbing

Ground-truth audit of `/home/user/voltradeai` at HEAD `8998328` (2026-10-15). Everything below was read from code. Where a
comment or doc disagrees with the code, the code wins and the disagreement is called out. Line numbers are `file:line`.
"(code-derived)" marks conclusions reached by tracing the code that have not been confirmed against production.

---

## 0. Top findings (read these first)

1. **Most "heavy" Python subprocesses are classified "trivial" and get a hard 10 s SIGKILL with no mutex.**
   `isTrivialPython()` (`server/bot.ts:233-237`) calls any command up to 150 characters "trivial" unless it literally
   contains `import numpy|pandas|sklearn|lightgbm|scipy|torch`. Trivial commands skip the Python mutex and run with
   `timeout = min(hardTimeout, 10000)` (`bot.ts:244-246`), and `hardTimeout` is already capped at 90 s (`bot.ts:241`).
   I re-ran the classifier against the real command strings (scratch script). These calls all end up "trivial / no mutex / 10 s":
   - Tier-3 ML retrain `python3 ml_retrain_safe.py` (asks for 900000 ms, `bot.ts:5739`)
   - the 4am retrain (`bot.ts:6214`)
   - Tier-1 `manage_positions` (asks for 15 s, `bot.ts:4213`)
   - shadow backfill (asks for 120 s, `bot.ts:4285`)
   - `sec_form4_bulk` (asks for 120 s, `bot.ts:5900`)
   - `cftc_cot` (asks for 45 s, `bot.ts:5879`)
   - manipulation scan (`bot.ts:5802`)
   - the overnight `bot_engine.py full` scans (`bot.ts:6021`, `6186`)
   - backtest (`bot.ts:6149`)
   - backup (`bot.ts:6254`)

   The `[likely OOM kill ...]` classification on TIER3-ML-ERROR SIGKILLs (`bot.ts:5785-5786`) can therefore be the 10 s
   exec cap. This is not recorded anywhere in `research/` (grep found no match). (code-derived)
2. **Overnight research and the daily 4am reset effectively never run from the scheduler.**
   - `runOvernightResearch()` is only called from the end of `tier2Intelligence()` when `!isMarketOpen && (etHour >= 20 || etHour < 4)` (`bot.ts:5207-5215`).
   - `scheduleTier2()` only runs Tier 2 when `clock.is_open || (4 <= etH < 20)` (`bot.ts:7681`).
   - Together these mean the research function is only reached by a scan that starts before 20:00 ET and finishes after
     it, or by a manual `/api/bot/run-now` at night.
   - Its 4am branch (`Math.floor(etHour) === 4`, `bot.ts:6199`) can never be reached, because the caller requires
     `etHour < 4 || >= 20`. That branch holds the daily reset (`morningQueueExecuted`, `orderAttemptCounts`, `sweptCspAt`,
     `dailyBlockedTickers`, `consecutiveStopLosses`), the 4am retrain and the nightly backup.
   - Consequence: those counters only reset when the process restarts. On a day with no redeploy,
     `state.morningQueueExecuted` stays `true`, so the next day's morning queue never executes, and
     `orderAttemptCounts` keeps accumulating toward MAX-RETRY=10. (code-derived)
3. **`/api/system/health-check` is permanently "critical" and `/api/system/snapshot` scan stats are permanently zero.**
   - Both look for a TIER2 audit message that contains `"complete"` (`bot.ts:2071`, `2079`, `2254-2258`).
   - No TIER2 audit message contains that word. The success line is `Scanned N stocks, M trade candidates (via …)`
     (`bot.ts:4448`).
   - So `no_successful_scans` is always pushed as critical, `last_scan_success` is always `null`, and `scan_stats.success`
     is always 0.
   - Its RSS thresholds (critical above 450 MB, warning above 380 MB, `bot.ts:2264-2265`) also sit below normal production
     RSS (~500–990 MB per the comments in `memoryCeiling.ts`/`bot.ts`).
4. **`/api/health` never checks the daemon.**
   - The `python` check runs `python3 -c "print('ok')"` as a subprocess (`bot.ts:1485`). It proves the binary exists,
     not that the RPC daemon is alive.
   - Daemon liveness is only visible on the owner-only `/api/daemon/health`, `/api/system/snapshot`,
     `/api/monitoring/overview`, and the token-gated `/api/diag/daemon`.
5. **The Node kill switch is invisible to Python, and there are four independent drawdown halts.**
   - Node latch: `voltrade_kill_switch.json`, -10%.
   - `bot_engine` DD halt: `voltrade_portfolio_dd.json`, 18%.
   - `risk_kill_switch`: `voltrade_killswitch.json`, -20%, plus a `voltrade_MANUAL_KILL` touch file.
   - `tiered_strategy` master kill.
   - No Python file reads the Node latch (grep `voltrade_kill_switch` in `*.py` → none).
   - `/api/monitoring/drawdown` reports a **-20%** kill threshold using the Python peak (`bot.ts:3526-3541`).
     `/api/health` and `/api/monitoring/overview` use the Node peak and **-10%**.
6. **Paper vs live is not a single switch.**
   - Node trades through a hard-coded `ALPACA_BASE = "https://paper-api.alpaca.markets"` (`bot.ts:319`).
   - Python reads env `ALPACA_BASE_URL`, default paper (`bot_engine.py:185` and others).
   - `csp_universe.py:180` reads a different env var, `ALPACA_TRADING_URL`.
   - `mode: "paper"` in `/api/bot/account` and `/api/bot/status` is a string literal (`bot.ts:1263`, `2317`), not derived
     from config.
7. **Restart semantics hide owner intent.**
   - `state.active` boots `true` (`bot.ts:486`), and `ownerStopped` is in-memory (`bot.ts:2341`). An owner STOP is undone
     by every deploy.
   - Turning the kill switch OFF via `/api/bot/kill` does **not** set `state.active = true` (`bot.ts:2381-2384`). The bot
     stays stopped (and the liveness alarm stays dark) until `/api/bot/start` is called. The UI message reads "trading can resume".
8. **No per-stage heartbeat timestamps exist in memory or in any API.**
   - Tier 1 emits no audit line on a healthy cycle, and its failures go to `console.error` only (`bot.ts:4246-4248`, `7613-7615`).
   - Stream connection state (`streamConnected`, `bot.ts:6299`) is not exposed anywhere.
   - No metric records Alpaca latency, RPC round-trip latency, or Tier-2 cycle duration.

---

## 1. Process topology and boot sequence

**Supervisor** (`run_with_daemon.sh`, FROZEN, read-only):
- Railway start command is `./run_with_daemon.sh` (`railway.json` startCommand).
  Railway settings: healthcheckPath `/api/health`, healthcheckTimeout 60, restartPolicyType ALWAYS, maxRetries 10, numReplicas 1.
- Node heap is set from the cgroup limit: 60%, floor 512 MB, cap 6144 MB, override `VOLTRADE_NODE_HEAP_MB`
  (`run_with_daemon.sh:20-31`). The chosen value is printed once to stdout only.
- If `VOLTRADE_DAEMON_ENABLED=false`, or `voltrade_daemon.py` or `python3` is missing, the script execs Node only
  (`run_with_daemon.sh:33-46`).
- Otherwise a background loop runs `python3 voltrade_daemon.py`, then on exit prints `daemon exited ($?) — restart in 2s`,
  sleeps 2 s and restarts it, forever (`run_with_daemon.sh:49-56`).
- `trap 'kill $DAEMON_PID…' SIGTERM SIGINT` (`run_with_daemon.sh:59`) is **discarded** by the `exec $NODE_CMD` on line 62,
  because exec replaces the shell. Node becomes PID 1's process and handles SIGTERM itself (`bot.ts:839-840`). The daemon
  loop is never signalled; it dies when the container is torn down.

**Node boot order:**
- **Module load of `server/bot.ts`** (before any route registration):
  - mutex watchdog interval (`bot.ts:285`)
  - tmp cleanup interval (`bot.ts:310`)
  - `state.equityPeak = loadEquityPeak()` (`bot.ts:638`)
  - `state.killSwitch = loadKillSwitch()` (`bot.ts:696`)
  - `livenessState = loadLiveness()` (`bot.ts:764`)
  - liveness 60 s interval (`bot.ts:766-791`)
  - SIGTERM/SIGINT handlers (`bot.ts:839-840`)
  - `audit_log` table creation and the WAL pragma (`bot.ts:1137-1158`)
- **`registerRoutes`** (`server/routes.ts:603`) → **`registerBotRoutes(app)`** (`bot.ts:1195`):
  - `STARTUP` audit `Server boot — code_version X, pid N` (`bot.ts:1201`)
  - crash handlers (`bot.ts:1209`)
  - route registration
  - interval: position sync 60 s (`bot.ts:7227`)
  - interval: event-loop lag 2 s (`bot.ts:7375`)
  - interval: shadow backfill 5 min (`bot.ts:7388`)
  - interval: Tier 1 at 45 s (`bot.ts:7602`)
  - `armTier2Timer(10000)`, so Tier 2's first run is at +10 s (`bot.ts:7734`)
  - interval: Tier 3 at 1 h (`bot.ts:7737`)
  - startup Tier 3 at +30 s (`bot.ts:7768`)
  - boot block at +5 s (`bot.ts:7864`), which:
    - runs the feedback cleanup subprocess
    - writes about 30 `SYSTEM`/`RULES`/`EXECUTION`/`SCHEDULE` audit lines
    - reads `/v2/clock`, then calls `startStreaming()` at +3 s unconditionally, even when killed (`bot.ts:7939`)
- `httpServer.listen(PORT || 5000)` (`server/index.ts:139-148`) happens after `registerRoutes` resolves.

Initial in-memory `state` (`bot.ts:485-499`):

| field | initial value |
|---|---|
| `active` | `true` |
| `killSwitch` | `false`, then overwritten from disk |
| `dailyPnL` | 0 |
| `dailyLossLimit` | -3 |
| `positionSizeMultiplier` | 1.0 |
| `minScoreThreshold` | 65 |
| `diagCycleCount` | 0 |
| `consecutiveStopLosses` | 0 |
| `circuitBreakerUntil` | 0 |
| `alpacaFailCount` | 0 |
| `morningQueueExecuted` | `false` |
| `equityPeak` | 0, then overwritten from disk |
| `maxDrawdownPct` | -10 |

Ad-hoc fields attached later: `streamSignalQueue`, `dailyBlockedTickers`, `recentStopTickers`, `lastStopTimes`.

> NOTE: when the kill latch is restored ON at boot, `state.active` stays `true` (`bot.ts:696-699` never touches `active`).
> Tier 3's gate is `!state.active || tier3Running` (`bot.ts:7745`), so after a restart **Tier 3 runs hourly while killed**.
> When the kill fires in-process, `state.active = false` (`bot.ts:3939`), and Tier 3 stops instead.

---

## 2. Sub-stages

### 2.1 Python RPC daemon (`voltrade_daemon.py`)

**What it does.** A long-lived Unix-socket JSON-line RPC server. It holds numpy, pandas and requests resident, lazily
imports trading modules, and warms up `storage_config, position_sizing, options_execution, ml_model_v2, bot_engine` in a
background thread (`voltrade_daemon.py:724-730`).

**Socket:** `VOLTRADE_DAEMON_SOCKET`, default `/tmp/voltrade_daemon.sock` (`:76`). Mode 0666 (`:737-740`).

**Method table** (`:265-304`):

| RPC method | target | notes |
|---|---|---|
| `health` | local `_health` | |
| `ping` | local `_ping` | |
| `ml_status` | module `ml_status_impl` | that module does not exist in the repo, so it always uses `_ml_status_fallback` (`:346-348`) |
| `ml_toggle` | module `ml_toggle_impl` | missing → `_ml_toggle_fallback` |
| `track_fill` | `ml_model_v2.track_fill` | |
| `check_halt` | `position_sizing.check_halt_status` | |
| `select_contract` | `options_execution.select_contract` | |
| `submit_options_order` | `options_execution.submit_options_order` | |
| `evaluate_and_execute` | `options_execution.evaluate_and_execute` | |
| `risk_status` | `risk_kill_switch.get_kill_switch_status` | |
| `get_peak_equity` | `risk_kill_switch.get_peak_equity` | |
| `check_position_risk` | `risk_kill_switch.check_position_risk` | |
| `check_correlation_pre_trade` | `risk_kill_switch.check_correlation_pre_trade` | |
| `paper_resume_sync` | `paper_resume_sync.rebaseline_python_halts` | |
| `macro_snapshot` | `macro_data.get_macro_snapshot` | |
| `shadow_stats` | `shadow_portfolio.get_shadow_stats` | |
| `cache_inventory` | local `_cache_inventory` | |
| `run_full_scan` | `bot_engine.scan_market` | |

- Node also calls `snapshot_python_state` (`bot.ts:2107`) and `strategy_pnl_stats` (`bot.ts:3508`). **Neither is in the
  table**, so the daemon answers `{"status":"error","error_message":"Unknown method: …"}` (`:331-333`) and `pythonCall`
  silently falls through to the subprocess path every time, logging only a `console.warn`.

**Timeouts and concurrency:**
- `REQUEST_TIMEOUT_SEC = 300` (`:78`) applies twice:
  - to acquiring the in-flight semaphore (`:569`)
  - to `t.join()` on the per-request worker thread (`:601`)
- `MAX_INFLIGHT_REQUESTS = env VOLTRADE_DAEMON_MAX_INFLIGHT or 8` (`:86-87`).
- **Busy:** if the semaphore is not acquired within 300 s, the response is
  `{"status":"error","error_message":"Daemon busy: >8 in-flight requests"}` (`:569-579`).
- **Thread creation failure:** `"Daemon thread creation failed: …"` (`:602-614`).
- **Timeout:** `"Request timed out after 300s"` (`:616-620`). The worker thread is **abandoned, not killed**. It keeps
  running and is counted in `active_dispatches` (`:89-144`).
- Dispatch exceptions return `{"status":"error","error_message":str(e)[:500],"traceback":…[:2000]}` (`:436-440`).
- `scan_market` skips its own 55 s SIGALRM cap when it runs in a daemon worker thread (`bot_engine.py:2493-2502`). In the
  daemon it is bounded only by the 300 s outer timeouts. The comment there saying "daemon REQUEST_TIMEOUT_SEC=60" is stale;
  the code value is 300.

**RSS self-kill** (`_memory_watchdog`, `:646-659`):
- Every 30 s it reads `resource.getrusage(RUSAGE_SELF).ru_maxrss`. On Linux this is the **peak** RSS since start, not current.
- If above `MAX_RSS_MB` (env `VOLTRADE_DAEMON_MAX_RSS_MB`, default 1024), it logs to stderr and calls `os._exit(3)`.
- The supervisor restarts it 2 s later.

**Startup cost before the socket listens** (`main()`, `:667-751`):
- stale-socket unlink
- an optional one-shot feedback seed subprocess (`seed_feedback_from_backtest.py`, **up to 120 s**, only when
  `trade_feedback.json` is missing or has fewer than 100 records, and `VOLTRADE_AUTOSEED != "false"`, `:687-717`)
- watchdog start, warmup thread start, then listen

**Payloads:**
- `health` result (`:442-461`), keys:
  - `alive` (always `true`)
  - `uptime_seconds`
  - `rss_mb` (peak maxrss)
  - `max_rss_mb`
  - `modules_loaded[]`
  - `active_dispatches`
  - `active_dispatch_detail[{method, elapsed_sec}]`
  - `layer2_prefetch{cache_hit, completed, total, elapsed_sec, budget_exceeded, checked_at, age_sec}`
  - `layer1_stats{cache_hit, universe_size, elapsed_sec, checked_at, age_sec}`
  - `surface_score_stats{calls, cache_misses, cumulative_elapsed_sec, skipped_budget, checked_at, age_sec}` (each is `{}` if not loaded yet)
  - `pid`
- `ping` result: `{pong: true, t: <epoch s>}` (`:463-465`). No Node caller uses it.
- Trace file `/tmp/voltrade_daemon_trace.json`, written only for `run_full_scan` (`:308-329`, `362-372`, `420-431`):
  - `last_request_received`
  - `last_method`
  - `last_status` (`received` | `calling_scan_market` | `scan_market_returned`)
  - `call_start`, `call_end`, `call_duration`
  - It is not updated on exception, so a failed scan leaves `last_status: "calling_scan_market"`.

**Healthy state:**
- `alive: true` with `uptime_seconds` monotonically growing (a drop means a restart).
- `rss_mb` (peak) below 1024.
- `active_dispatches` 0–1, with a `run_full_scan` `elapsed_sec` below 300.

**Failure signature:**
- **Daemon restarts are not audited anywhere.** Only supervisor stdout shows them. They are visible from outside only as
  an `uptime_seconds` reset or `pid` change, if someone polls health.
- While it is down, `run_full_scan` produces
  `TIER2-ERROR "Scan failed via daemon: daemon run_full_scan failed: Socket: connect ENOENT … — daemon health returned non-alive: {…}"`
  (`bot.ts:5150-5158`).
- Abandoned zombie threads are visible only through `active_dispatch_detail`.

### 2.2 Daemon RPC client — `pythonRpc()` (`bot.ts:106-143`)

- **What:** opens a fresh `net.createConnection(DAEMON_SOCKET)` per call, writes `{"method","args"}\n`, resolves on the
  first `\n`-terminated line.
- **Timeout:** `timeoutMs` if given, else `DAEMON_TIMEOUT_MS = 30000` (`bot.ts:104`).
- **Configuration:**
  - `DAEMON_SOCKET = env VOLTRADE_DAEMON_SOCKET || "/tmp/voltrade_daemon.sock"` (`bot.ts:102`)
  - `DAEMON_ENABLED = env VOLTRADE_DAEMON_ENABLED !== "false"` (`bot.ts:103`)
- **It never rejects.** Errors resolve as envelopes:
  - `{status:"error", error_message:"Daemon disabled via env" | "Daemon timeout" | "Socket: <msg>" | "Parse: <msg>"}`
  - success is the daemon's `{status:"ok", result}`
- Every call opens and closes a socket. There is no connection pooling and no latency measurement.

### 2.3 `pythonCall()` — daemon-first, subprocess fallback (`bot.ts:151-192`)

- **Returns** `{success, result, via, error?}` where `via` is one of:
  - `"daemon"`
  - `"subprocess"`
  - `"daemon-failed"` (heavy method, daemon answered non-ok)
  - `"daemon-error"` (heavy method, RPC threw — practically unreachable, because `pythonRpc` never throws)
  - `"none"` (subprocess failed)
- `HEAVY_DAEMON_ONLY = {run_full_scan, scan_market, manage_positions}` (`bot.ts:162-164`). These never fall back, to
  avoid OOM.
- **Comment vs code:** Tier 1 calls `manage_positions` through `execPythonSerialized` **as a subprocess** every 45 s
  (`bot.ts:4213-4218`), bypassing both `pythonCall` and the daemon-only intent.
- Daemon non-ok responses log `console.warn` `[pythonCall] daemon X status=… error=…` only (`bot.ts:172-173`). Nothing is audited.
- **Duplicate-execution risk (code-derived):** a light method that times out on the Node side (30 s, or busy-waiting on
  the daemon semaphore) falls back to a subprocess. The daemon may still execute the same request later, because the
  abandoned or queued thread runs on. For `track_fill` (`bot.ts:203-211`, `4153-4161`, `5413`, `6530`, `7134`) this can
  double-record a fill in trade_feedback.

**Call sites by method:**

| method | call site |
|---|---|
| `track_fill` | ×4 |
| `snapshot_python_state` | `/api/system/snapshot` |
| `risk_status` | `/api/monitoring/overview` |
| `ml_status` | `/api/monitoring/overview` |
| `strategy_pnl_stats` | `/api/monitoring/tier-pnl` |
| `get_peak_equity` | `/api/monitoring/drawdown` |
| `cache_inventory` | `/api/monitoring/caches` |
| `run_full_scan` | Tier 2 |
| `paper_resume_sync` | auto-resume |
| `check_halt`, `check_correlation_pre_trade`, `evaluate_and_execute` | `executeTrades` |

### 2.4 Subprocess runner — `execPythonSerialized()` and the mutex (`bot.ts:227-296`)

- **Environment:** every child gets `OPENBLAS/MKL/OMP/NUMEXPR/VECLIB = 2` threads (`bot.ts:56-63`) and `maxBuffer` 32 MB
  (`bot.ts:68`, `90-91`).
- **Non-trivial commands** (more than 150 characters, or one that literally imports a heavy library) wait for the
  module-level mutex:
  - flags: `pythonRunning`, `pythonLockedAt`, `pythonLockHardTimeout`
  - they poll every 500 ms, for up to `opts.timeout || 30000`
  - on wait timeout they throw `"Python mutex timeout — another Python process is holding the lock"` (`bot.ts:262-264`)
  - a stale lock (held longer than holder's `hardTimeout + 10 s`) is force-released with
    `console.error "[python-mutex] Force-releasing stale lock…"` (`bot.ts:252-260`)
  - they then run with `timeout = hardTimeout = min(opts.timeout || 90000, 90000)` and SIGKILL (`bot.ts:241`, `271`)
  - the lock is always released in `finally`
- **Mutex watchdog:** `setInterval` **30000 ms** (`bot.ts:285-296`). It releases stale locks and logs to console only.
- **Trivial path** (`bot.ts:243-246`): no mutex, timeout `min(hardTimeout, 10000)`, SIGKILL. See Top finding 1 for which
  heavy jobs land here.
- **Effective ceiling:** nothing launched through this function can run longer than 90 s, regardless of the requested timeout:
  - Tier-3 retrain asks for 900000 (`bot.ts:5739`)
  - Tier-2 subprocess fallback asks for 300000 (`bot.ts:4410`; never actually used, because `run_full_scan` is daemon-only)
  - overnight scans ask for 300000 / 180000
- **Silent:** mutex timeouts in `scheduleTier2` are swallowed into a console line plus a 60 s back-off; no audit is written
  (`bot.ts:7695-7701`).

### 2.5 Broker client — `alpaca()` (`bot.ts:318-346`)

- `ALPACA_BASE` is the hard-coded paper URL. `ALPACA_KEY` and `ALPACA_SECRET` come from env. If either is missing, a
  console warning says "trading disabled" (`bot.ts:322`), but nothing actually disables trading; calls simply fail with 401.
- **Timeout:** 15 s via AbortController (`bot.ts:325-326`).
- **Non-2xx:** throws `Error("Alpaca <status>: <body|statusText>")` (`bot.ts:338-341`).
- **2xx:** returns `r.json()`. A 204 No Content (single-order `DELETE`) therefore **throws even though it succeeded**. This
  is handled after the fact for the stale sweep only (`bot.ts:3835-3846`, `cspSweepCooldown.confirmCancelAfterDeleteError`).
- **No** retry, back-off, 429/`Retry-After` handling, rate limiter, or latency measurement on the Node side.
- Other direct Alpaca fetches bypass `alpaca()`:
  - `fetchDailyBarsRange` → `data.alpaca.markets` (`bot.ts:2410-2419`)
  - the WebSocket (`bot.ts:7248`)
  - morning-queue price checks via a Python `requests` one-liner (`bot.ts:4055`)

### 2.6 Tier 1 Reflex (`tier1Reflex`, `bot.ts:4191-4249`; interval `bot.ts:7601-7616`)

**Cadence:** `setInterval(…, 45000)` (`bot.ts:7616`). The gates are evaluated in this order:
1. If `state.killSwitch` → run `killSwitchAutoResumeTick()` instead and return (`bot.ts:7605`).
2. If `!state.active` → return.
3. If `state.circuitBreakerUntil > now` → return. **This pauses the drawdown-kill check during a circuit breaker.**
4. `await alpaca("/v2/clock")`. If `!clock.is_open` → return. Tier 1 runs only in Alpaca-reported RTH, so it correctly
   skips holidays and half-day afternoons.

**Comment and config conflicts:**
- `system_config.py:35` has `TIER1_INTERVAL_MS: 30_000`, but bot.ts hard-codes 45 s and never reads it.
- CLAUDE.md says "Tier 1 stop monitoring ~30s".
- The boot audit line `RULES "Scan interval: 45 seconds during trading hours"` (`bot.ts:7901`) is really the Tier-1
  interval, not a scan interval.

**Phases:**
0. `checkDrawdownKillSwitch()` (`bot.ts:3922-3984`), in its own try/catch:
   - `/v2/account` → `evaluateDrawdown(equity, equityPeak, -10)`
   - an invalid read audits `EQUITY-READ-INVALID "tier1: equity=…"`
   - a new peak triggers `saveEquityPeak()`
   - on a kill:
     - `state.killSwitch = true`, `saveKillSwitch("drawdown-kill (tier1)")`, `state.active = false`
     - `audit DRAWDOWN-KILL`
     - email `"MAX DRAWDOWN — Trading Stopped"`
     - `DELETE /v2/orders`
     - optional liquidate-all when `VOLTRADE_LIQUIDATE_ON_KILL === "true"` and drawdown ≤ -25%, auditing
       `LIQUIDATE-ALL` / `LIQUIDATE` / `LIQUIDATE-ERROR` / `LIQUIDATE-FAIL`, with email "ALL POSITIONS LIQUIDATED"
   - errors go to `console.error "[tier1-drawdown]"` only
1. `sweepStaleOrders()` (`bot.ts:3809-3875`):
   - cancels tracked `openOrders` older than `STALE_ORDER_MINUTES = 12` (`bot.ts:3768`)
   - `audit SWEEP`
   - re-syncs against `/v2/orders?status=open`
2. `manage_positions` subprocess (trivial path, so 10 s): refreshes `voltrade_stop_state.json`. Audits
   `POS-MONITOR-SYNC` when bot_engine flags an action. Failures go to `console.error "[tier1-sync]"` only.
3. `syncMonitoredPositions()`.
4. A second `/v2/clock` read. If open, not yet executed today, and the queue is non-empty → `executeMorningQueue(t1Equity)`,
   then `state.morningQueueExecuted = true` (`bot.ts:4237-4241`).
5. `trackClosedTrades()` (`/v2/orders?status=closed&limit=50`, `bot.ts:851-937`). Failures audit `LEARN-ERROR`.

**Emitted fields:** audit types `DRAWDOWN-KILL`, `EQUITY-READ-INVALID`, `SWEEP`, `POS-MONITOR-SYNC`, `MORNING*`,
`LEARN`, `LEARN-ERROR`, `LIQUIDATE*`. There is no per-cycle audit or heartbeat.

**Healthy state:** a cycle every 45 s during RTH while active, not killed, and not circuit-broken. Nothing in the audit
log proves it ran.

**Failure signature:**
- Exceptions in the interval or in `tier1Reflex` go to `console.error("[tier1]")` only (`bot.ts:4246-4248`,
  `7613-7615`), so an Alpaca outage or 401 during Tier 1 is **silent in the audit log**.
- **No re-entrancy guard** (unlike Tier 2/3/auto-resume). `setInterval` does not await, so if a cycle takes more than
  45 s, cycles overlap.
- `executeMorningQueue` only sets `morningQueueExecuted` at the end, and clears the queue at the end (`bot.ts:4175`). A
  queue taking more than 45 s (per trade: an 8 s price-check subprocess, the order, and a 500 ms sleep) can therefore be
  executed twice by overlapping cycles. (code-derived)

### 2.7 Morning queue (`executeMorningQueue`, `bot.ts:3990-4177`)

**Where the queue is fed:**
- Tier 2 when `!isMarketOpen`: `morningQueue.push({...trade, queuedAt})`, deduplicated by ticker, at most
  `MAX_POSITIONS = 8` per scan, audit `QUEUE` (`bot.ts:4664-4670`).
- The overnight extreme-mover research (`bot.ts:6097-6118`), which is effectively unreachable (see 2.10).
- The queue is **in-memory only**, so it is lost on any restart.

**Execution** (first Tier-1 cycle with `clock.is_open` while `!state.morningQueueExecuted`):
1. Drop entries older than 16 h (`MORNING-STALE`), then audit `MORNING "Executing N queued trades…"`.
2. Sort by score.
3. Enforce slot cap: `MORNING "Max positions (8) reached"`.
4. Enforce size cap: `MORNING-BLOCKED`.
5. Gap check via a Python one-liner on `delayed_sip`: skip on a gap above 5% (`MORNING-GAP`), resize on a gap above 2%
   (`MORNING-ADJUST`).
6. Convert any short to a buy (`SHORT-BLOCKED`).
7. Skip qty < 1 (`MORNING-SKIP`).
8. `POST /v2/orders` using `getOrderParams()`, which is a DAY limit (`MORNING-TRADE`).
9. Register the order in `openOrders`.
10. `track_fill` with a **synthetic random slippage** (`bot.ts:4130-4161`).
11. `addPositionToMonitor`.
12. Errors audit `MORNING-ERROR`.
13. At the end: `MORNING "Queue cleared."`

**Daily reset:** `morningQueueExecuted` only resets in the unreachable 4am branch (`bot.ts:6205`) or on process restart.
See Top finding 2.

### 2.8 Kill switch latch and paper auto-resume

**Latch file:**
- Path: `/data/voltrade/voltrade_kill_switch.json`, fallback `/tmp/voltrade_kill_switch.json` (`bot.ts:647-648`).
- Keys: `{killSwitch: bool, reason: string, savedAt: ISO, lastAutoResumeAt?: ISO}` (`bot.ts:688`).
- Parsed by `parseKillSwitchRecord` (`killSwitchAutoResume.ts:156-166`), which accepts the older
  `{killSwitch}` and `{killSwitch, reason, savedAt}` shapes.
- `reason` strings written:
  - `"drawdown-kill (account route)"` (`bot.ts:1242`)
  - `"drawdown-kill (tier1)"` (`bot.ts:3938`)
  - `"owner toggle ON"` / `"owner toggle OFF"` (`bot.ts:2374`)
  - `"auto-resume: <RECOVERED|DATA_ANOMALY|PAPER_REBASE>"` (`bot.ts:7573`)
- `saveKillSwitch` writes the volume path and falls back to `/tmp` **only if the write throws** (`bot.ts:675-694`).
- `loadKillSwitch` reads the first path that parses (`bot.ts:658-673`).
- **Fail-open:** if the file is missing or unparseable on both paths, it returns `false`, so a corrupted latch boots the
  bot **un-killed and active**.

**Who trips it:**
- Tier-1 `checkDrawdownKillSwitch` (see 2.6).
- The **owner `/api/bot/account` route** (`bot.ts:1235-1250`), which is a dashboard read with side effects. Same -10%
  rule, also emails, and also cancels all orders.
- The owner toggle `POST /api/bot/kill` (`bot.ts:2372-2386`). It is a **toggle, not an idempotent set**, so a double
  click flips it back.
  - ON: email "Kill Switch Activated", `state.active = false`, `audit "KILL SWITCH ON"`, `notify("alert")`, `DELETE /v2/orders`.
  - OFF: `audit "KILL SWITCH OFF"`. Does **not** reactivate.

**Auto-resume** (paper-only, human-directed 2026-09-28):
- `killSwitchAutoResumeTick()` (`bot.ts:7480-7534`) runs on each Tier-1 tick while killed (45 s), guarded by
  `autoResumeRunning`.
- **Eligibility** (`killSwitchAutoResume.ts:413-430`):
  - `VOLTRADE_KILL_AUTO_RESUME` not in `off/0/false/disabled/no`
  - `allPaperAlpacaUrls([ALPACA_BASE, env ALPACA_BASE_URL || ALPACA_BASE])`
  - latch reason starts with `drawdown-kill`
- **Evaluation:**
  - reads `/v2/clock`; it only evaluates while the market is open
  - reads `/v2/account`, `/v2/positions`, and FILL activities from latch minus 10 days (100 per page, up to 5 pages)
  - computes `evaluateAutoResume()` (`killSwitchAutoResume.ts:434-529`)
- **Tunables** (`AUTO_RESUME`, `killSwitchAutoResume.ts:69-98`):

  | tunable | value |
  |---|---|
  | `COOLDOWN_MINUTES` | 30 |
  | `CONSECUTIVE_OK_REQUIRED` | 3 |
  | `ANOMALY_EXPLAINED_RATIO` | 0.25 |
  | `RECOVERY_HYSTERESIS_PP` | 2 |
  | `PAPER_REBASE_AFTER_TRADING_DAYS` | 2 (× 6.5 market hours) |
  | `FILLS_LOOKBACK_DAYS` | 10 |
  | `FILLS_PAGE_SIZE` | 100 |
  | `FILLS_MAX_PAGES` | 5 |
  | `ANTI_FLAP_HOURS` | 24 |

- **On resume** (`performKillSwitchAutoResume`, `bot.ts:7536-7599`):
  1. `pythonCall("paper_resume_sync", …)` re-baselines the Python halts.
  2. Possibly re-baselines `equityPeak`.
  3. `state.killSwitch = false`, `saveKillSwitch("auto-resume: …", {autoResume: true})`.
  4. `audit "KILL SWITCH AUTO-RESUME"`, `notify("alert")`, email `Kill Switch Auto-Resumed (paper) — <label>`.
  5. `startBotActivity("kill-switch auto-resume")`, which refuses if `ownerStopped`.

**Status object** `autoResumeStatus` (`bot.ts:703-722`):
- Keys: `eligible`, `latchReason`, `outcome` (`NOT_EVALUATED|NOT_LATCHED|INELIGIBLE|MARKET_CLOSED|INVALID_READ|EVALUATED|RESUMED`),
  `classification` (`NO_DRAWDOWN|DATA_ANOMALY|REAL_LOSS|UNRECONCILED|null`), `basis` (`RECOVERED|DATA_ANOMALY|PAPER_REBASE|null`),
  `reason`, `consecutiveOk`, `required`, `lastEvaluation`, `nextCheck`, `numbers`, `lastResume`.
- `numbers{equity, peak, drawdownPct, resumeAbovePct, latchedMinutes, marketHoursSinceLatch, rebaseAfterMarketHours, reportedDrop, unrealizedPl, realizedPl, explainedLoss, explainedRatio}`.
- `lastResume{at, basis, priorPeak, newPeak}`.
- Audit `KILL-AUTO-RESUME-EVAL` is written **on change only** (`bot.ts:7472-7477`).
- Exposed in full on `/api/bot/status.autoResume`. `/api/health.checks.bot.autoResume` omits `numbers` and
  `lastResume.priorPeak/newPeak`.

**Failure signature:**
- The auto-resume streak lives in memory, so a restart resets `consecutiveOk` and `firstObservedAt`.
- Read failures go to `console.error "[auto-resume] <label> read failed"` and surface as `outcome: "INVALID_READ"`.
- The owner toggle OFF does not run `paper_resume_sync`, so Python halts can still block entries after a manual un-kill.

### 2.9 Tier 2 scheduler (`getTier2Interval`, `armTier2Timer`, `scheduleTier2`; `bot.ts:7619-7734`)

**Interval** — `getTier2Interval()` uses `getETHour()`, which is wall-clock ET, **weekday-blind and holiday-blind**
(`server/orderParams.ts`):

| ET window | interval |
|---|---|
| 09:30–10:30 | 60000 ms |
| 10:30–12:00 | 180000 ms |
| 12:00–14:00 | 420000 ms |
| 14:00–16:00 | 120000 ms |
| 16:00–20:00 | 900000 ms |
| 04:00–09:30 | 300000 ms |
| otherwise | 1800000 ms |

These mirror `system_config.py:39-45`, which bot.ts does **not** read.

**Single-chain timer:** `armTier2Timer(delay)` refuses to arm a second timer while `tier2TimerArmed` is set
(`bot.ts:7649-7657`). The first run is at +10 s (`bot.ts:7734`).

**`scheduleTier2()` gates** (`bot.ts:7659-7685`):
- `TIER2_DISABLED` (env `VOLTRADE_DISABLE_TIER2 === "1"`) → audit `TIER2-DISABLED` once, and **the chain stops for good**.
- `!active || killSwitch || tier2Running` or circuit breaker active → re-arm after the normal interval.
- `/v2/clock`, then `isAnyWindow = clock.is_open || (4 <= etH < 20)`. Otherwise re-arm (30 min overnight, so the first
  pre-market scan starts between 04:00 and 04:30 ET).

**Run:**
- `tier2Running = true`
- `audit TIER2 "Starting scan (interval: Nmin based on market time)"`
- `tier2Intelligence(clock.is_open, etH)`
- `tier2CompletedSinceBoot = true`

**Back-off** (`bot.ts:7702-7730`):
- `failed = schedulerThrew || tier2LastScanFailed`; failure increments `tier2ConsecutiveFailures`, success resets it to 0.
- Mutex timeout: delay `max(60000, interval)`, with no audit.
- Failure: delay `max(min(600000, 60000 × 2^(min(n,5)−1)), interval)` and audit `TIER2-BACKOFF "failure #n, next scan in Ns"`.
- Scheduler throw: audit `TIER2-ERROR <msg>`.

**Healthy state:** during RTH, a pair of `TIER2` lines ("Starting scan…", "Starting intelligence scan…") plus
`"Scanned N stocks, M trade candidates (via daemon)"` every 1/3/7/2 min, no `TIER2-BACKOFF`, and
`tier2ConsecutiveFailures = 0`.

**Off-hours behavior:**
- Tier 2 **also runs full daemon scans on weekends and market holidays from 04:00 to 20:00 ET**, every 5–15 min, because
  `getETHour` is weekday-blind and `clock.is_open` is false.
- Those scans queue morning trades, and tier order actions are deferred (`TIER-DEFER`).

**Failure signatures:**
- Classified daemon/subprocess failures: see 2.10.
- `/api/health.checks.scanner` degrades at `consecutiveFailures >= 6` (`scannerHealth.ts:21`).
- **Not counted as failures:**
  - `"TIER2 Alpaca not responding — skipping cycle"` (equity missing)
  - `TIER2-FAIL "Alpaca down (n consecutive)"`
  - `TIER2-LIMIT` (daily loss)

  These return early without setting `tier2LastScanFailed` (`bot.ts:4341-4353`, `4616-4620`), so the scanner check stays
  "ok" through them. An Alpaca outage still usually counts, because `scheduleTier2`'s own `/v2/clock` call throws first.

### 2.10 Tier 2 Intelligence phases (structure only; `tier2Intelligence`, `bot.ts:4335-5216`)

1. Reset `tier2LastScanFailed` and `tier2LastFailureDetail`. `audit TIER2 "Starting intelligence scan..."` (`4336-4338`).
2. Alpaca ping `/v2/account`. Increments or resets `state.alpacaFailCount`, which is **exposed nowhere** (`4341-4353`).
3. Every 5th cycle (`state.diagCycleCount % 5`): `diagnostics.get_auto_fix_params(...)` subprocess (non-trivial path,
   15 s).
   - Sets `state.positionSizeMultiplier` and `state.minScoreThreshold`.
   - Audits `DIAGNOSTIC <problems_summary>` unless the summary is "All systems healthy".
   - The `should_pause` → `DIAGNOSTIC-PAUSE` branch is **dead**: `get_auto_fix_params` never sets `should_pause = True`
     (`diagnostics.py:526-565`) (`4355-4377`).
4. Drain the stream-signal queue (signals under 5 min old) → `STREAM-PRIORITY` audit (`4381-4388`). The drained tickers
   are not passed to the scan.
5. `pythonCall("run_full_scan", {}, "python3 -W ignore bot_engine.py full", {timeout: 300000}, 300000)` (`4406-4412`).
   Daemon only.
   - Failure with `result.error` → `audit TIER2 "Scan returned error: … (via …)"`. The audit **type is TIER2, not TIER2-ERROR**.
   - Success → `audit TIER2 "Scanned … (via daemon)"`.
   - `result.halted` → `DD-HALT`.
   - `dd_event.kind === "anomaly_release"` → `DD-HALT RELEASED…`.
   - Stores `tier2LastDataSourceErrors` and `lastScanResult`.
6. Signal-list maintenance (in-memory `signals`, at most 30 entries, 1 h expiry) (`4480-4559`).
7. Daily-loss guard: `evaluateDailyPnl(acct.equity, acct.last_equity)`.
   - Invalid read → `EQUITY-READ-INVALID`.
   - At or below **-3%** → `TIER2-LIMIT` plus `notify("alert")`, then return for this cycle only. It does not latch.
   - (`4561-4621`)
8. Weekly-loss check (`4623-4641`) is **dead**. It passes `check_weekly_loss` a history of length 1, which
   `diagnostics.py:44-45` treats as "continue". `WEEKLY-LIMIT` and `WEEKLY-WARNING` can never fire.
9. EOD short cover when `isMarketOpen && ET >= 15.83` (`4645-4659`). On half-days the market is closed by then, so this
   never runs.
10. If market open: `executeTrades(new_trades, equity)`. Else: queue for the morning (`4661-4671`).
11. Tier dispatcher (`4676-4978`):
    - Re-fetches positions, open orders and the account.
    - Defers order actions while closed (`tierDispatchGate.ts:26-48`, single `TIER-DEFER` line).
    - `SELL_CSP` checks: slot cap (`OPTIONS-SLOT-FULL`), cooldown (`CSP-SKIP`).
    - Dispatch audits `T<tier>`, `T<tier>-FAIL`, `T<tier>-ERR`, `T3`, `T3-FAIL`, `T3-SKIP`, `T4`, `T4-FAIL`, `T4-ERR`,
      `TIER-DISPATCH-ERR`.
12. `KILL-SWITCH`, `KILL-WARN`, `TIER-KILL` from the Python kill status fields (`4981-4997`).
13. Catch block (`4999-5205`):
    - Daemon failure → health probe (5 s), then `TIER2-ERROR "Scan failed via daemon: <msg> — daemon rss=…MB active_dispatches=… [method:secs] uptime=…s layer2_prefetch={…} layer1_stats={…} surface_score_stats={…} scan_timings={status,last_phase,age} preamble={total, slowest}"`.
    - Subprocess failure → `TIER2-ERROR "Scan failed (code=… signal=… rss=…MB heap=…/…MB)[classification]: <stderr tail>"`.
14. Overnight research hook (`5207-5215`). Effectively unreachable; see Top finding 2.

**Overnight research** (`runOvernightResearch`, `bot.ts:6010-6280`):
- Branches by ET hour: 20–22, 22–24 (`etHour >= 22 || etHour < 0`), 0–2, 2–4, and `Math.floor(etHour) === 4`
  (daily reset, retrain, backup).
- Throttled to once per 55 min via `lastResearchRun`. Sets `autoRunning`, a write-only flag.
- **Reachability:**
  - Scheduled path: only the 20–22 branch, only for a scan straddling 20:00 ET.
  - `/api/bot/run-now` at night: the 20–22, 22–24, 0–2 and 2–4 branches.
  - The 4am branch: never.
- Audits: `RESEARCH`, `RESEARCH-ERROR`, `QUEUE`, `SYSTEM "Daily reset…"`, `RETRAIN`, `RETRAIN-ERROR`, `BACKUP`,
  `BACKUP-WARN`, `BACKUP-ERROR`.
- The boot `SCHEDULE` audit lines (`bot.ts:7926-7932`) advertise this 8pm–4am schedule as if it ran.

### 2.11 Tier 3 Strategic (`tier3Strategic`, `bot.ts:5709-5949`; interval `bot.ts:7737-7756`; startup `bot.ts:7762-7769`)

**Cadence:**
- `setInterval(…, 3600000)`.
- Gate: `TIER3_DISABLED` (env `VOLTRADE_DISABLE_TIER3 === "1"`, `TIER3-DISABLED` audited once) or `!state.active || tier3Running`.
- Not gated by market hours or the kill switch, but see the boot note in §1.
- One extra run at +30 s after boot, gated only by `TIER3_DISABLED`.

**Steps:**
1. Model-age check (5 s). If the model is older than 24 h → `ml_retrain_safe.py` (trivial path, 10 s, see Top finding 1).
   - Audits: `TIER3 "ML model stale…"`, `TIER3 "ML retrain complete — status…"`, `TIER3-ML-ERROR`, `TIER3 "ML model fresh (Xh old)…"`.
2. Manipulation scan → `MANIPULATION` (up to 5 alerts) or `TIER3-MANIP-ERROR`.
3. `diagnostics.run_diagnostics(...)` → `TIER3-DIAG "System health: <overall_status> — N issues (node_uptime_s=…, tier2_ran_since_boot=…): [SEV] system: msg | …"`
   when not healthy.
4. `recordDailyEquity()`. **This is the only writer of the equity curve**, so equity-curve days are missing whenever Tier 3
   is not running (stopped or killed in-process).
5. `cftc_cot.run_daily_update` → `TIER3-COT` on per-symbol errors.
6. `sec_form4_bulk.run_update` → `TIER3-FORM4BULK`.
7. Port-dwell weekly capture (in-process TypeScript) → `TIER3-PORTDWELL`.
8. `TIER3 "Strategic scan complete"`.

Errors in steps 1 (outer), 3, 5, 6 and 7 go to console only (`[tier3-ml]`, `[tier3-diag]`, `[tier3-cot]`,
`[tier3-form4bulk]`, `[tier3-portdwell]`). The interval-level catch audits `TIER3-ERROR`.

**Coupling:** datacore archive updates (COT, Form4, port-dwell) **stop whenever the trading bot is stopped or killed
in-process**.

**Healthy state:** `TIER3 "Starting strategic scan..."` … `"Strategic scan complete"` once per hour.

### 2.12 Position-sync interval (`bot.ts:7226-7230`)

- `setInterval(…, 60000)`, gated only by `state.active && !state.killSwitch`. **Not market-hours gated.**
- Calls `syncMonitoredPositions()` (`bot.ts:6345-…`):
  - `/v2/positions`, plus a stop-state subprocess (5 s)
  - can place **TIME-EXIT market orders** and **POS-KILL** orders at any hour
  - audits `TIME-EXIT`, `TIME-EXIT-ERROR`, `POS-KILL*`, `POS-WARN` (every 60 s per position at or below -15%),
    `POS-QTY-CORRECTED`, `POS-MONITOR`
- Also called from Tier 1 (step 3) and on stream subscription. These can run concurrently with each other.

### 2.13 WebSocket market-data stream (`startStreaming` / `stopStreaming`, `bot.ts:7232-7365`)

**What:**
- Connects to `wss://stream.data.alpaca.markets/v2/iex` (`bot.ts:7248`). Auths with `ALPACA_KEY/SECRET`.
- On `{"T":"success","msg":"authenticated"}`, subscribes `bars` for `STREAM_TICKERS`: 43 tickers (`bot.ts:6287-6293`).
- On `{"T":"subscription"}`: `streamConnected = true`, `audit STREAM "Real-time feed live — <n> tickers"`, and
  `syncMonitoredPositions()` (`bot.ts:7267-7272`).
- `{"T":"error"}` → `STREAM-ERROR "Alpaca stream error code=… msg=…"` (`bot.ts:7281-7283`).

**Bar handling** (`{"T":"b"}`):
- If the ticker is monitored → `checkPositionOnTick(ticker, close)`. Inactive when killed or not active (`bot.ts:6724`).
- Signal detection: volume ≥ 2.5× average, bar change ≥ 0.8%, 5-bar momentum ≥ 0.3%, 10 min cooldown per ticker, gated
  on active / not killed / no circuit breaker → `STREAM-SIGNAL` and `state.streamSignalQueue` (`bot.ts:7314-7335`).

**Errors and reconnect:**
- `ws.on("error")` → `STREAM-ERROR <msg[:100]>`.
- `ws.on("close")` → `STREAM-DISCONNECT "Feed disconnected — reconnecting in 10s"` or
  `"Feed closed before ever reaching subscribed state — reconnecting in 10s"`, then
  `setTimeout(startStreaming, 10000)` if `state.active && !state.killSwitch` (`bot.ts:7345-7359`).
- The reconnect is a flat 10 s: no back-off, no retry cap, **no ping/pong heartbeat, no bar-staleness watchdog**.
- A half-open socket that never fires `close` would leave `streamConnected = true` with no bars, and nothing detects it.

**Lifecycle:**
- Started at boot (+8 s, even when killed), by `/api/bot/start`, and by auto-resume through `startBotActivity`
  (`bot.ts:2349`, +2 s).
- `stopStreaming()` only from `/api/bot/stop` (`bot.ts:2365`). The kill switch does not stop the stream; it simply stops
  reconnecting after the next close.

**Reconnect-churn hazard (code-derived):**
- `startStreaming()` closes the previous socket (`bot.ts:7233`).
- The old socket's `close` handler does not check `ws === streamWs`. It schedules another `startStreaming()` 10 s later
  whenever the bot is active.
- So calling `startStreaming()` while a socket is already open (e.g. `/api/bot/start` while active, or auto-resume while
  the boot socket is still open) can set up a self-perpetuating close/reopen cycle about every 10 s, each cycle writing
  a `STREAM-DISCONNECT` line.

**Exposure:** `streamConnected`, `positionMonitorInitialized`, subscribed tickers and last-bar time are **not exposed by
any endpoint**.

**Healthy state:** one `STREAM "Real-time feed live — 43 tickers"` after boot, then no STREAM-DISCONNECT churn.

### 2.14 SSE, audit log, and notifications

- **`audit(action, detail)`** (`bot.ts:510-517`):
  - writes console `[BOT] action: detail`
  - calls `broadcastSSE(entry)`
  - calls `persistAudit(action, detail)`
  - entry keys: `{time, type, action, detail, message}`, where `type === action` and `message === detail`
- **SQLite storage:** `audit_log(id, time, type, message)` in `db` from `server/auth.ts`, at
  `DB_PATH || /data/voltrade.db` (note: `/data`, not `/data/voltrade`; `auth.ts:12-16`). WAL mode (`bot.ts:1153-1158`).
- **`persistAudit`** (`bot.ts:1160-1180`):
  - truncates the message to 500 characters
  - **keeps only the last 2000 rows**: every insert runs `DELETE … NOT IN (SELECT … LIMIT 2000)`
  - a slow insert or delete (≥ 500 ms, `dbWriteTiming.ts:58`) audits `DB-SLOW-WRITE`
  - retention is by row count, not time; a chatty day (boot alone writes about 35 lines) can push out history within hours
- **Readers:**
  - `GET /api/bot/audit`: last 100 rows mapped to `{time, action, type, detail, message}` (`bot.ts:2399-2402`)
  - `/api/diag/audit?type=&limit≤500` (token-gated)
- **SSE:**
  - `GET /api/bot/stream` (owner) sets `text/event-stream` and adds the response to `sseClients` (`bot.ts:1212-1220`)
  - `broadcastSSE` writes `data: <json>\n\n` (`bot.ts:504-508`)
  - only default unnamed events, no heartbeat or keep-alive comments, and the audit entry is the only payload
  - **no client code subscribes** (grep `bot/stream` / `EventSource` in `client/src`: none)
- **`notify(type, message)`** (`bot.ts:529-533`):
  - in-memory array, newest first, maximum 100, `{time, type, message, read}`
  - types used: `alert`, `critical`, `earnings`, `exit`, `research`, `system`, `trade`, `upgrade`
  - read via `/api/bot/notifications`; `/api/bot/notifications/read` marks all read; the unread count is in `/api/bot/status`
  - lost on restart, and it is **not** an outbound alert channel

### 2.15 Liveness alarm (Amendment 2 runtime half)

**Pure logic** (`server/liveness.ts`):
- `LOOP_DARK_MARKET_HOURS = 2` (`:16`), `LOOP_DARK_WALLCLOCK_HOURS = 24` (`:17`), `LIVENESS_REMINDER_INTERVAL_HOURS = 24` (`:26`).
- `nextLiveness(prev, activeNow, now)` (`:39-45`):
  - active → `{lastActiveAt: now}`
  - inactive with no prior record → seed `now`
  - otherwise keep the prior record
- `marketHoursBetween` (`:64-71`): weekdays 09:30–16:00 ET via Intl, in 5-minute steps, capped at 14 days. **Holidays and
  half-days are not excluded.**
- `loopDark` (`:74-95`): dark if market hours > 2 or wall hours > 24.
  `detail = "LIVENESS ALARM: trading loop dark for X market hours (Yh wall-clock) since <ISO> — …"`.

**"Active" definition:** `state.active && !state.killSwitch` only (`bot.ts:770`, `1496`).
- **Broker-unreadable is not an input**, despite CLAUDE.md's definition ("paused, halted, or broker-account-unreadable").
- A loop that is "active" but failing every scan is caught separately by `checks.scanner`.
- Python-side halts (`DD-HALT`, `TIER-KILL`) do not count as dark.

**File:**
- Path: `/data/voltrade/voltrade_liveness.json`, fallback `/tmp/voltrade_liveness.json` (`bot.ts:729-730`).
- Keys: `{lastActiveAt: epoch ms, lastReminderAt?: epoch ms, savedAt: ISO}`.
- Corrupt or missing → `null` → re-seeded at now. **Fail-open: the darkness clock resets.**

**Cadence:**
- Module-level `setInterval(…, 60_000)` (`bot.ts:766-791`):
  - stamps when active, at one disk write per minute
  - while dark, sends at most one email every 24 h: `"Trading Loop Still Dark — Resume Decision Needed"`
- `/api/health` also advances the stamp on every hit (`bot.ts:1497-1498`).

**Exposure:**
- `/api/health.checks.bot.liveness = {dark: false}` or `{dark: true, marketHours, wallHours, detail}`.
- When dark, `status` flips to `"degraded"` (`bot.ts:1518-1541`).
- It never touches the HTTP code (see 2.16).

**Failure signature:** an email (if `RESEND_KEY` is set) plus a degraded health payload. The 60 s interval's errors are
swallowed by `catch {}` (`bot.ts:790`).

### 2.16 `/api/health` and `healthGate` (deploy gate)

**`GET /api/health`** (unauthenticated, `bot.ts:1445-1657`). Response keys:
- `timestamp`
- `status` (`"ok" | "degraded"`)
- `serving{ok, gates, failing, note}`
- `checks.server{status:"ok", uptime_s, heap_used_mb, rss_mb}`
- `checks.database{status, detail?}`: `SELECT 1`. Error → degraded **and HTTP 503**.
- `checks.alpaca{status, account_status | detail}`: live `/v2/account` (up to 15 s).
- `checks.python{status, detail?}`: trivial subprocess, 5 s. **Not the daemon.**
- `checks.bot`:
  - `status`: `"killed" | "active" | "stopped"`
  - `equityPeak`
  - `drawdownPct`: string `toFixed(1)` or `"N/A"`
  - `liveness`
  - `autoResume{eligible, outcome, classification, basis, reason, consecutiveOk, required, lastEvaluation, nextCheck, lastResume{at, basis}|null}`
- `checks.scanner{status: "ok"|"degraded", consecutiveFailures}`
- `checks.feeds{status, dead[], feeds{aircraft|vessels|trains: {silent_hours, dead}}, detail?, gates_top_level_status: false}`:
  archive throughput with a 3 h threshold (`feedDeadAir.ts`). **Never flips `status`.**
- `checks.licensing{status: "ok"|"warning", detail?}`
- `checks.process{unhandledRejections, uncaughtExceptions, handlerFaults, lastRejection{at, detail}|null, lastException{at, detail}|null}`
  (`crashVisibility.ts`)
- `checks.memory{heapUsedMB, heapTotalMB, rssMB, externalMB, heapLimitMB, cgroupLimitMB, cgroupUsageMB, cgroupHeadroomMB, cgroupSource, pressure: "ok"|"heap-near-limit"|"cgroup-near-limit"|"unknown"}`
  (`memoryCeiling.ts`, near-limit fraction 0.85)

What flips `status` to "degraded": database error, alpaca error, python error, liveness dark, scanner degraded (6 or
more consecutive failures), licensing warning.

**Deploy gate:**
- `servingVerdict` and `healthHttpCode` (`server/healthGate.ts:45`, `73-85`).
- `SERVING_CHECKS = ["server", "database"]`, so HTTP **200 unless the server or database check is not ok, else 503**.
- `serving.note` explains the split.
- Railway's healthcheck (60 s) reads only the code.
- CI's `scripts/deploy_gate_smoke.mjs` boots the built bundle with the kill latched and a stale liveness stamp, and
  requires a 200 within 60 s.

**History:**
- `recordHealthSnapshot(checks)` (`bot.ts:1643`, `pipelineHealthHistory.ts:85-101`) appends at most one row per 5 min to
  `<DATA_DIR>/datacore_archive/pipeline_health/YYYY-MM-DD.jsonl`.
- Row keys: `{t, status, database_ok, alpaca_ok, python_ok, scanner_ok, scanner_consecutive_failures, licensing_ok, bot_status, bot_liveness_dark, heap_used_mb, rss_mb}`.
- Rows are only written **when someone hits `/api/health`**, so there is no sampling when nobody polls and no sampling
  while the process is down.
- `GET /api/data/pipeline-health-dashboard` (`routes.ts:1750-1760`) returns
  `{generated_at, warming_up, window_24h, window_7d}`. Each window is
  `{window_hours, sample_count, uptime_pct, current, degraded_counts{database, alpaca, python, scanner, licensing, bot_liveness_dark}, timeline[{t, status}]}`.
- **`uptime_pct` is really "% of samples with status ok"**, not process uptime.

**Healthy state:** HTTP 200, `status: "ok"`, `checks.bot.status: "active"`, `liveness.dark: false`,
`scanner.consecutiveFailures: 0`, `memory.pressure: "ok"`, `server.uptime_s` growing across reads, `process` counters 0.

**Cost:** each hit makes a live Alpaca account call and spawns one Python subprocess, with no auth or rate limit.

### 2.17 Other status and health endpoints

**`GET /api/bot/status`** (owner, `bot.ts:2311-2332`). Keys:
- `active`, `killSwitch`
- `dailyLossLimit` (-3)
- `auditLogCount`
- `mode: "paper"` (literal)
- `equityPeak`, `maxDrawdownPct` (-10)
- `autoResume` (full status object)
- `unreadNotifications`
- `circuitBreakerActive`, `circuitBreakerUntil` (ISO|null), `consecutiveStopLosses`
- `maxPositionPct: 8`, `maxExposurePct: 95`, `dailyLossLimitPct: 5` — **hard-coded literals**.
  `dailyLossLimitPct: 5` contradicts `dailyLossLimit: -3` and the -3% `TIER2-LIMIT`.

**`GET /api/daemon/health`** (owner, `bot.ts:3328-3342`): `{alive: true, ...daemon health result}` or
`{alive: false, reason}`. Timeout 30 s.

**`GET /api/system/snapshot`** (owner, downloaded as an attachment, `bot.ts:2006-2243`). Keys:
- `snapshot_version: "1.1"`, `generated_at`
- `git_head_expected: "5d2add3 or later"` (stale literal)
- `server{uptime_seconds, node_version, memory_mb, heap_used_mb, heap_total_mb}`
- `bot_state{active, kill_switch, daily_pnl, daily_loss_limit, position_size_multiplier, min_score_threshold, max_drawdown_pct, equity_peak, consecutive_stop_losses, circuit_breaker_until}`
- `daemon{enabled, alive, …health}`
- `audit_log{total_entries, recent_errors[≤10], recent_backoffs, last_scan_success (always null, see Top finding 3), last_scan_failure}`
- `scan_stats{success (always 0), error, sigkill, last_24h}`
- `positions{count, deployment_pct, by_ticker, equity, cash, buying_power}`
- `kill_switches`, `ml`, `config_files{base_config, current_adaptive, macro}`, `stress_index`, `floor_state`,
  `tier_engine{tiers_enabled}`, `probability_engine`
- `last_scan_timings` (contents of `voltrade_scan_timings.json`), `daemon_trace`
- The Python part always goes through the subprocess (`snapshot_python_state` is not a daemon route), with a 20 s timeout.

**`GET /api/system/health-check`** (owner, `bot.ts:2245-2284`):
- Payload: `{timestamp, overall: "healthy"|"warning"|"critical", issues_count, warnings_count, issues[{severity, category, detail}], warnings[...]}`.
- Categories: `scan_failures` (5 or more TIER2-ERROR/BACKOFF in the last 200 audits), `no_successful_scans` (**always**),
  `stale_scans`, `memory_pressure` (RSS > 450 critical, > 380 warning), `daemon_down`, `health_check_failed`.
- **Effectively always "critical".**

**`GET /api/monitoring/overview`** (owner, `bot.ts:3347-3401`). Keys:
- `timestamp`
- `daemon`
- `kill_switches` (the `risk_kill_switch.get_kill_switch_status` shape:
  `{peak_equity, consecutive_losses, consecutive_loss_limit, daily_pnl_pct, daily_loss_limit_pct, portfolio_dd_kill_pct, vxx_regime_kill_threshold, min_free_bp}`;
  the default is `{active: 0, warnings: []}`)
- `memory{rss_mb}`
- `drawdown{current_pct, kill_threshold_pct, proximity_pct, status: OK|WARNING|CRITICAL|UNKNOWN}` (Node peak, -10%;
  `drawdownGuard.ts drawdownStatus`)
- `ml` (`{model_exists, model_age_hours, enabled, contributes_to_cagr, last_status}` from the daemon fallback)

**`GET /api/monitoring/drawdown`** (owner, `bot.ts:3521-3546`):
- `{equity, peak (risk_kill_switch peak), drawdown_pct, kill_threshold_pct: -20, status: KILLED|CRITICAL|WARNING|NOTICE|OK}`.
- **Disagrees with the -10% kill that actually halts Node.**

**`GET /api/bot/market-status`** (**unauthenticated**, `bot.ts:3575-3587`):
- `{isOpen, nextOpen, nextClose, timestamp}` straight from `/v2/clock`, or 500 `{error}`. No early-close flag.

**`GET /api/bot/calendar`** (owner, `bot.ts:7777-7861`):
- `{tomorrow{status: open|weekend|holiday|early_close, note, date}, holidays[{date, name}], earlyCloses[{date, close}], tradingDaysNext30}`.
- Dates come from Alpaca `/v2/calendar`; holiday names are a hard-coded 2026 table.

**`GET /api/bot/diagnostics`** (owner, `bot.ts:7969-7984`):
- `{report: run_diagnostics(), auto_fix: get_auto_fix_params()}`, or `{error}` with HTTP 200.
- **The GET has side effects:**
  - `cleanup_disk()` truncates `/tmp/voltrade_*.json` files above 5 MB and deletes old caches when the total exceeds 50 MB
  - writes the blend-weight tracker
  - appends to `voltrade_health_log.json`
  - runs `run_diagnostics` **twice** (`diagnostics.py:512-524`)
- `report` keys: `{timestamp, overall_status: healthy|warning|degraded, problems[{system, severity, message, auto_fix, fix_params}], warnings[{system, message}], auto_fixes[], recommendations[], blend_weights?}`.
- `auto_fix` keys: `{position_size_multiplier, min_score_threshold, should_pause (always false), force_retrain, problems_summary}`.

**`GET /api/bot/system-status`** (**unauthenticated**, `routes.ts:6575-6593`, `system_status.py`):
- `{generated_at, schema_version: "system-status-v1", overall: ok|degraded|broken, summary{total, passed, failed, warned}, checks{size_tier_module, adaptive_params_overlay, size_portfolio_respects_tier, sector_correlation_override, top_10_enrichment, tie_breaking_present, scan_caches}}`.
- Comment vs code:
  - the docstring says it is gated by `DEBUG_ENABLED`; it is not
  - the routes.ts comment says warnings make it "degraded"; in code, only failures do (`system_status.py:250`)

**`GET /api/diag/:probe`** (token-gated: `DIAG_TOKEN` of length ≥ 24, header `x-diag-token` or `?token=`; 404 when
unset, `bot.ts:2426-…`, `diag.ts:293-303`). Orchestration-relevant probes:
- `audit` (`?type=`, `?limit≤500`)
- `daemon` (`{probe, alive, …health}`)
- `scanner` (`{probe, degraded, consecutiveFailures, lastFailureDetail, dataSourceErrors}`)
- `timings` (`{probe, found, …voltrade_scan_timings.json}`)
- `account` (`{probe, …accountRow, equityPeak, drawdownPct, killSwitch}`)
- `equity_curve`, `positions`, `positions-detail`, `orders`, `ml`

**Scan timings file** (`bot_engine.py:2584-2614`, `3995-4003`):
- Path: `<DATA_DIR>/voltrade_scan_timings.json`.
- Keys: `{scan_started, phases[{phase, duration_sec, total_sec}], status: "in_progress"|"completed", last_phase_completed, last_phase_at, total_sec (when completed)}`.
- Phase names: `universe_load, snapshot_fetch, quick_scan, deep_score, step6a_trade_loop_and_covered_calls, step6_trade_loop_and_options, step7_manage_positions, step8b_options_manager, step9_third_leg, step9b_convexity_overlay, step10_spy_floor, step10b_defensive_floor, tier_engine_start, tier_engine_<p>, tier_engine_complete, scan_complete`.
- This is the **only per-stage timing that exists**, and it is overwritten every scan.

### 2.18 Owner controls

**`POST /api/bot/start`** (`bot.ts:2354-2359`):
- Goes through `startBotActivity("")` (`bot.ts:2342-2351`).
- 400 `{error: "Kill switch is ON. Disable it first."}` if killed.
- Otherwise: `active = true`, `audit START "Bot activated"`, `notify system`, `startStreaming` at +2 s, `ownerStopped = false`.
- Returns `{ok: true, active: true}`.

**`POST /api/bot/stop`** (`bot.ts:2362-2369`): `ownerStopped = true`, `active = false`, `stopStreaming()`, `audit STOP`.
Returns `{ok: true, active: false}`. **In memory only, so a redeploy re-activates the bot.**

**`POST /api/bot/kill`:** a toggle; see 2.8. Returns `{ok: true, killSwitch}`.

**`POST /api/bot/circuit-breaker/reset`** (`bot.ts:2389-2395`): `circuitBreakerUntil = 0`, `consecutiveStopLosses = 0`,
`audit CIRCUIT-BREAKER-RESET`. Returns `{ok: true, circuitBreakerUntil: 0, consecutiveStopLosses: 0}`.

**`POST /api/bot/run-now`** (`bot.ts:7956-7966`):
- If `tier2Running` → `{message: "Tier 2 scan already running..."}`.
- Otherwise it locks, responds `{message: "Tier 2 intelligence scan starting..."}`, and runs `tier2Intelligence` directly.
- It bypasses `active`, `circuitBreakerUntil`, the time-window gate, and the back-off counter.
- The kill switch is only checked inside the order loops. The Python `scan_market` it triggers (which can itself place
  floor/overlay orders) does not know about the Node kill.

### 2.19 Circuit breaker (in WS exit handling, `bot.ts:7165-7212`)

**Trigger:** a stop-loss or trailing-stop WS exit (stock positions only) increments `consecutiveStopLosses`, with a 5 min
per-ticker dedupe. At 3:
- If `dailyPnL / equityPeak ≤ -5%`: pause until `setHours(20,0,0,0)` in **server local time (UTC)** = 16:00 EDT /
  15:00 EST, minimum 1 h. Email "Circuit Breaker — Day Halted", `audit CIRCUIT-BREAKER`.
- Else if the last 3 stops hit at least 2 tickers: pause 1 h. Email "Circuit Breaker Triggered", `audit CIRCUIT-BREAKER`.
- Else: `TICKER-BLOCKED` for the day.

**Caveats:**
- `state.dailyPnL` is **only updated by the owner `/api/bot/account` route** (`bot.ts:1230`). It is stale unless someone
  loads the dashboard.
- In-memory only, so it resets on restart.
- While active, it suspends Tier 1 entirely, including the drawdown-kill check, and Tier 2.

### 2.20 Email alerts (`sendEmailAlert`, `bot.ts:793-818`)

- Sends via Resend (`https://api.resend.com/emails`) from `EMAIL_FROM` to `env ALERT_EMAIL` or a hard-coded fallback
  address. Subject is prefixed `[VolTradeAI]`.
- Silently does nothing when `RESEND_KEY` is unset; no audit.
- **The HTTP status is not checked.** Any non-throwing response, including a 401/422 rejection, writes
  `audit EMAIL "Alert sent: <subject>"`. Network errors go to `console.error` only.

| trigger | site |
|---|---|
| drawdown kill (account route / tier1) | `bot.ts:1247`, `3942` |
| owner kill ON | `bot.ts:2376` |
| liquidate-all | `bot.ts:3976` |
| circuit breaker ×2 | `bot.ts:7194`, `7198` |
| auto-resume | `bot.ts:7588` |
| liveness reminder, every 24 h while dark | `bot.ts:785` |

Tier-2 failures, daemon death, stream outage, scanner degraded, feed dead-air and crashes **send no email**.

### 2.21 Graceful shutdown and crash handlers

**`gracefulShutdown(signal)`** (`bot.ts:821-840`), on SIGTERM or SIGINT:
1. `resolveInflightRefreshes()` → `audit SHUTDOWN "Resolved in-flight refresh markers…"`.
2. `audit SHUTDOWN "Received SIGTERM — cancelling open orders and shutting down..."`.
3. `DELETE /v2/orders` (**cancels every open order**, including resting CSP and exit limits) → `SHUTDOWN "All open orders cancelled"` or `SHUTDOWN-ERROR`.
4. `persistAudit("SHUTDOWN", "Graceful shutdown on SIGTERM")`.
5. `process.exit(0)`.

There is no time bound beyond the 15 s Alpaca timeout. **Every Railway redeploy cancels all resting orders.**

**Crash handlers** (`crashVisibility.ts`, installed at `bot.ts:1209`):
- `unhandledRejection` → `audit FATAL-REJECTION`, and the process keeps running.
- `uncaughtException` → `audit FATAL-EXCEPTION`, then `exit(1)` after 250 ms. **No order cancel.** Orders resting at the
  crash are not tracked by the next process: `openOrders` is in memory, so they are never swept.
- Counts surface in `checks.process`.

### 2.22 Miscellaneous intervals

- **tmp cleanup:** every `TMP_CLEANUP_INTERVAL_MS = 600000` (`tmpCleanup.ts:29`). Deletes `/tmp/{fb_,fill_,opt_}*` older
  than 300000 ms. Audits `TMP-CLEANUP` only when ≥ 200 candidates are scanned (`bot.ts:310-316`). `tier_csp_*` and
  `tier_hedge_*` temp files are not covered; Python deletes them itself.
- **Event-loop lag:** every `EVENTLOOP_LAG_CHECK_MS = 2000`. Audits `EVENTLOOP-LAG "Node event loop stalled ~Nms…"` when
  lag ≥ 500 ms (`eventLoopLag.ts:33-34`, `bot.ts:7374-7382`). **This is the only continuous latency metric, and it is
  stored only as audit lines.**
- **Shadow backfill:** checked every 300000 ms. Runs once when the UTC hour is 22 (18:00 EDT / 17:00 EST), resets at UTC
  hour 0, with no market or kill gate. Audits `SHADOW-BACKFILL <stats json>` or `SHADOW-BACKFILL-ERROR`. Trivial path, so
  the subprocess actually gets 10 s (`bot.ts:4281-4313`, `7388`).

### 2.23 Python helper modules

**`diagnostics.py`:**
- `check_weekly_loss` (`:38-81`; a -8% limit that is dead from bot.ts).
- `check_cache_freshness` (`:96-140`): multiplier 0.5 / 0.7 / 0.85 / 1.0.
- `cleanup_disk` (`:145-216`): mutates files.
- `check_model_health` (`:221-280`): retrain needed after 7 days, overdue after 10 days.
- `run_diagnostics` (`:303-509`): API-down grace via `tier2_ran_since_boot`; writes `voltrade_health_log.json` (last 100)
  and the blend tracker.
- `get_auto_fix_params` (`:512-565`): `should_pause` is never true.
- `DIAG_PATH` (`:29`) is defined but never used.

**`system_status.py`:** a self-test only. See 2.17.

**`market_calendar.py`** (FROZEN facts):
- 2026 only: 10 holidays, half-days 2026-11-27 and 2026-12-24 (`:20-37`).
- API: `is_market_holiday`, `is_half_day`, `is_short_week`, `trading_days_this_week`, `next_trading_day`,
  `days_until_next_holiday`, `is_pre_long_weekend`, `should_skip_new_options` (`:44-144`).
- **The only runtime consumer is `options_scanner.py:1765`** (skip new options on half-days and before long weekends),
  plus `server/optionsChainArchive.ts:154`, which parses the file for archive scheduling.
- The trading loop's open/closed decisions use Alpaca `/v2/clock`; Tier-2 cadence and order types use wall-clock ET; the
  liveness alarm uses weekday 09:30–16:00.
- **From 2027-01-01 the file knows no holidays** (see the December rule in CLAUDE.md).

**`alpaca_rate_limiter.py`** (FROZEN):
- A token bucket of 180 per 60 s, **per process** (`:75`).
- `install_global_throttle()` monkey-patches `requests.get/post/delete` for URLs containing `alpaca.markets` (`:134-171`).
- The daemon and every subprocess each get their own 180/min bucket. The **Node side has none**. Nothing coordinates
  against Alpaca's account-wide 200/min.
- No 429 counter or metric is emitted.

### 2.24 `research/liveness_notify_state.json` and `research/outage_state.json`

- **Not runtime state.** Both are written by `scripts/session_health_check.py` (`:52-56`, `319-432`, `753-765`), the
  session-side checker that reads the public `/api/health` and the token-gated `/api/diag/*`. They are git-committed so
  sessions share them.
- `outage_state.json` keys: `{down_since, last_outage_started_utc, last_recovered_utc}`.
  - Current value: `down_since: null`, last outage `2026-09-10T20:18:00Z` → recovered `2026-09-16T02:40:06Z`.
  - Tracks **site unreachable** (no `/api/health` payload).
- `liveness_notify_state.json` keys: `{dark_since, last_notified_utc, last_notified_wall_hours}`.
  - Current value: all `null`, meaning not dark.
  - Tracks **loop dark while the site is up**. A session is told to notify again when wall-hours have at least doubled
    since the last notification (`compute_liveness_notify_state`, `:366-397`).

---

## 3. Master timer table (`server/bot.ts`)

| line | kind | period / delay | gate | does |
|---|---|---|---|---|
| 119 | setTimeout | `timeoutMs` or 30000 | — | pythonRpc timeout |
| 285 | setInterval | 30000 | — | python mutex stale-lock release (console only) |
| 310 | setInterval | 600000 | — | /tmp orphan cleanup |
| 326 | setTimeout | 15000 | — | `alpaca()` abort |
| 766 | setInterval | 60000 | — (module scope) | liveness stamp + 24 h dark reminder email |
| 2349 | setTimeout | 2000 | via `startBotActivity` | `startStreaming` |
| 7227 | setInterval | 60000 | active && !kill | `syncMonitoredPositions` (24/7) |
| 7358 | setTimeout | 10000 | active && !kill at fire time | stream reconnect |
| 7375 | setInterval | 2000 | — | event-loop lag probe |
| 7388 | setInterval | 300000 | — | shadow backfill (UTC hour 22) |
| 7602 | setInterval | 45000 | kill → auto-resume tick; else active, no CB, clock open | Tier 1 |
| 7653 | setTimeout chain | `getTier2Interval()` / back-off; first 10000 (7734) | see 2.9 | Tier 2 |
| 7737 | setInterval | 3600000 | !TIER3_DISABLED, active, !tier3Running | Tier 3 |
| 7768 | setTimeout | 30000 | !TIER3_DISABLED | startup Tier 3 |
| 7864 | setTimeout | 5000 | — | boot banner, feedback cleanup, stream start (+3000 at 7939) |
| crashVisibility | setTimeout | 250 | uncaughtException | exit(1) |

Daemon side: the 30 s RSS watchdog (`voltrade_daemon.py:650`) and the 300 s request timeout. Supervisor: 2 s daemon
restart delay.

## 4. In-memory state flags

| flag | where set | meaning | exposed? |
|---|---|---|---|
| `state.active` | 486 (true), 1243, 2345, 2364, 2377, 3939, 4635 (dead) | loop on/off | `/api/bot/status.active`, `/api/health.checks.bot.status`, snapshot |
| `state.killSwitch` | 696, 1241, 2373, 3937, 7572 | latch | status, health, snapshot, diag/account |
| `ownerStopped` | 2341, 2357, 2363 | owner STOP blocks auto re-activate | no |
| `tier2Running` | 7687, 7703, 7958, 7962/7964 | scan lock | no |
| `tier2TimerArmed` | 7651-7654 | one timer chain | no |
| `tier2ConsecutiveFailures` | 7709/7711 | back-off counter | `/api/health.checks.scanner.consecutiveFailures`, diag/scanner |
| `tier2LastScanFailed` / `tier2LastFailureDetail` / `tier2LastDataSourceErrors` | 4336-4337, 4415-4442, 4475, 5000-5202 | last scan outcome | diag/scanner only |
| `tier2CompletedSinceBoot` | 7691, 7962 | first scan done | passed to diagnostics only |
| `tier3Running` | 7748/7754 | lock | no |
| `autoRunning` | 6014/6279 | overnight research running | no (write-only) |
| `autoResumeRunning` | 7482/7532 | lock | no |
| `pythonRunning` / `pythonLockedAt` | 267-277 | subprocess mutex | no |
| `streamConnected` | 7268, 7353, 7364 | WS subscribed | **no** |
| `positionMonitorInitialized` | 6648 | first sync done | no |
| `state.circuitBreakerUntil` / `consecutiveStopLosses` | 7193/7197, 2390 | CB | status, snapshot |
| `state.morningQueueExecuted` | 4240, 6205 (unreachable) | daily gate | no |
| `state.alpacaFailCount` | 4345-4350 | Tier-2 Alpaca ping failures | **no** |
| `state.positionSizeMultiplier` / `minScoreThreshold` | 4365-4366, 4639 | diagnostics auto-fix | snapshot only |
| `state.dailyPnL` | 1230 (account route only) | CB / `checkTradeAllowed` input | snapshot |
| `_shadowBackfilledToday` | 4290, 4311 | daily job flag | no |

## 5. Persisted state files (Node-written unless noted)

| file | keys | writer / reader |
|---|---|---|
| `/data/voltrade/voltrade_kill_switch.json` (fallback `/tmp`) | `killSwitch, reason, savedAt, lastAutoResumeAt?` | bot.ts 675 / 658 |
| `/data/voltrade/voltrade_liveness.json` | `lastActiveAt, lastReminderAt?, savedAt` | bot.ts 751 / 732 |
| `/data/voltrade/voltrade_equity_peak.json` | `equityPeak, savedAt` | bot.ts 622 / 607 |
| `/data/voltrade/voltrade_equity_curve.json` | `[{date, value, pnl}]`, newest first, ≤365 | bot.ts 583 / 563 |
| `/data/voltrade/voltrade_deploy_timestamp.json` | `deployTimestamp` | bot.ts 965 |
| `/data/voltrade.db` table `audit_log` | `id, time, type, message` (last 2000) | bot.ts 1160 |
| `<DATA_DIR>/datacore_archive/pipeline_health/YYYY-MM-DD.jsonl` | see 2.16 | pipelineHealthHistory.ts |
| `<DATA_DIR>/voltrade_scan_timings.json` | see 2.17 | bot_engine.py (Python) |
| `/tmp/voltrade_daemon_trace.json` | see 2.1 | voltrade_daemon.py (lost on restart) |
| `<DATA_DIR>/voltrade_stop_state.json`, `voltrade_options_state.json`, `voltrade_stop_cooldown.json` | per-ticker | Python + bot.ts subprocess |
| `<DATA_DIR>/voltrade_health_log.json` | last 100 diagnostics reports | diagnostics.py |
| `<DATA_DIR>/voltrade_portfolio_dd.json` | `peak, dd_pct, halted, halt_reason…` | bot_engine.py:534 (18% DD halt) |
| `<DATA_DIR>/voltrade_killswitch.json`, `voltrade_kill_history.json`, `voltrade_MANUAL_KILL`, `voltrade_peak_equity.json` | risk_kill_switch | risk_kill_switch.py:46-48, 657 |
| `research/outage_state.json`, `research/liveness_notify_state.json` | see 2.24 | scripts/session_health_check.py |

> Equity-curve bug (code-derived): `recordDailyEquity` does `unshift(new)` and then, past 365 entries, `shift()`
> (`bot.ts:1034-1035`). `shift()` removes index 0, which is the **entry it just added**. After 365 entries no new day is
> ever recorded. The `/api/diag/equity_curve` comment says "oldest-day pruned", which is not what happens.

---

## 6. FAILURE VARIABLES

### Daemon down or socket missing
- **Behavior:**
  - `pythonRpc` resolves `{status: "error", error_message: "Socket: connect ENOENT|ECONNREFUSED …"}`.
  - Light methods fall back to subprocesses silently (console.warn only).
  - `run_full_scan` (daemon-only) fails, producing `TIER2-ERROR "Scan failed via daemon: … — daemon health returned non-alive: {…}"` and `TIER2-BACKOFF`.
  - After 6 consecutive failures, `/api/health.checks.scanner` goes "degraded" and `status` "degraded".
  - Tier 1 is unaffected: it uses subprocesses.
- **Recovery time:** about 2 s supervisor delay, plus import and warm-up time, plus **up to 120 s** if the feedback seed
  runs before the socket listens.
- **Visibility:** daemon restarts themselves are audited nowhere. `/api/health` has no daemon check. Owner/diag endpoints
  show `alive: false, reason`.

### Daemon busy or timeouts
- **Behavior:**
  - Node gives up after 30 s (default) or 300 s (scan); the daemon gives up after 300 s.
  - Requests beyond 8 in flight wait up to 300 s for the semaphore, then get "Daemon busy".
  - Timed-out work keeps running as zombie threads, visible in `active_dispatches` and `active_dispatch_detail`.
  - Light-method fallbacks can **double-execute** (e.g. `track_fill`).
- **Visibility:** `TIER2-ERROR` carries daemon rss, dispatch detail, layer stats, and scan phase/age. Other methods'
  timeouts are console-only.

### WebSocket drop or reconnect
- **Behavior:** a flat 10 s reconnect when active and not killed. No heartbeat and no stale-bar detection, so a silent
  half-open socket goes unnoticed. Possible churn when `startStreaming` is called on a live socket (2.13).
- **Effect while down:** real-time WS exits (`checkPositionOnTick`) stop. Stops then rely on Tier-1 `manage_positions`
  (10 s trivial subprocess) and the 60 s `syncMonitoredPositions` POS-KILL/TIME-EXIT.
- **Visibility:** `STREAM-DISCONNECT`, `STREAM-ERROR` and `STREAM` audit lines only. **No stream state in any API**, and no alert.

### Broker (Alpaca) API failures: 401, 429, 5xx, timeouts
- **Node:**
  - every `alpaca()` call throws `"Alpaca <code>: <body>"`; a timeout throws an AbortError after 15 s
  - no retry, no 429 back-off, no `Retry-After` handling
  - a 204 success on single-order DELETE throws (handled only in the sweep)
- **Where each Node failure shows up:**

  | caller | how the failure surfaces |
  |---|---|
  | Tier 1 | **console only** (silent) |
  | Tier 2 | `TIER2-ERROR <alpaca msg>` from the scheduler's `/v2/clock`, plus back-off; or `TIER2-FAIL` / "Alpaca not responding" (not counted as failures) |
  | `/api/health` | `checks.alpaca{status: "error", detail}` and `status: "degraded"`; HTTP code unaffected |
  | auto-resume | `outcome: INVALID_READ` |
  | sync interval | silent catch |
  | owner routes | 500 `{error}` |
  | `gracefulShutdown` | `SHUTDOWN-ERROR` |

- **Python:** the per-process 180/min throttle (`alpaca_rate_limiter.py`). Errors surface through the scan result
  (`TIER2 "Scan returned error…"`, `debug_detail` in `diag/scanner.lastFailureDetail`).
- **Not counted anywhere:** the number of 429s, 401s, or 5xx responses.

### Latency spikes
**Measured:**
- `EVENTLOOP-LAG` audit (≥ 500 ms over a 2 s tick)
- `DB-SLOW-WRITE` audit (≥ 500 ms SQLite insert/delete)
- scan phase durations in `voltrade_scan_timings.json` (latest scan only)
- daemon `active_dispatch_detail[].elapsed_sec` and the trace file's `call_duration`
- per-HTTP-request duration printed to stdout only (`server/index.ts:94-103`)
- `/api/data/*` latency measured externally by `scripts/session_health_check.py`

**Not measured:** Alpaca REST latency, daemon RPC round-trip, Tier-1 or Tier-2 cycle duration, order submit-to-fill
latency (only synthetic slippage), WS bar latency, `/api/health` handler time.

### Market closed, holiday, half-day
- **Tier 1:** Alpaca clock, so correct on holidays and half-days.
- **Tier 2:** weekday- and holiday-blind 04:00–20:00 ET window. Runs full scans on weekends and holidays and queues
  morning trades; tier order actions are deferred (`TIER-DEFER`).
- **Half-days (13:00 close):**
  - `getETHour`-based logic (Tier-2 cadence; `getOrderParams` "regular hours" from 09:30 to 16:00) treats 13:00–16:00 as
    regular hours, so stop/trailing exits would be built as market/day orders without `extended_hours`
  - the EOD short cover (gated on `isMarketOpen`) never runs
- **Position sync:** the 60 s sync (POS-KILL, TIME-EXIT) runs 24/7.
- **Liveness alarm:** counts holiday and half-day afternoons as market hours, so it can alarm early.
- **`market_calendar.py`:** 2026 only, used only for options skip logic.
- **Calendar data:** `/api/bot/calendar` and `/api/bot/market-status` come from Alpaca.

### Conflicting or missing state files
**Fail-open on missing or corrupt files:**
- kill latch → boots **active**
- liveness → clock **reset**
- equity peak → 0, then re-seeded from current equity, so drawdown protection **re-bases**
- deploy timestamp → re-minted

**Volume problems:** writes fall back to `/tmp` only when the volume write throws. Reads take the volume copy first, so
a stale volume copy wins over a fresher `/tmp` copy.

**Four independent drawdown or kill states with different thresholds and peaks** (§0.5):
- `/api/monitoring/drawdown` (-20%, Python peak) vs `/api/health` and overview (-10%, Node peak)
- `/api/bot/status.dailyLossLimitPct: 5` vs the actual -3% guard

The owner un-kill does not clear Python halts. Python never reads the Node latch.

### Process restart mid-day
**Lost (in memory):**
- `morningQueue`
- `openOrders`: orders from a crash become unswept; a graceful SIGTERM cancels all orders anyway
- `monitoredPositions`: rebuilt from Alpaca plus `stop_state`
- circuit breaker, `consecutiveStopLosses`, `dailyBlockedTickers`, `orderAttemptCounts`, `sweptCspAt`
- `streamSignalQueue`, `signals`, `notifications`, `tradeResults`, `strategyWeights`, `lastScanResult`
- `tier2ConsecutiveFailures`
- `ownerStopped`: a stopped bot comes back **active**
- the auto-resume streak
- `state.dailyPnL`

**Kept:** kill latch, equity peak, liveness stamp, equity curve, deploy timestamp, audit log, Python state files.

**Boot behavior:** an owner-visible `STARTUP` audit with `code_version` and pid. `/api/health.checks.server.uptime_s`
resets, which is the only restart signal on the public surface. Every deploy also cancels all open orders through SIGTERM.

---

## 7. PAPER vs LIVE

**Node (`server/bot.ts`):**
- `const ALPACA_BASE = "https://paper-api.alpaca.markets"` is **hard-coded** (`bot.ts:319`). There is no env override;
  every Node order, account read, kill and auto-resume decision goes to paper.

**Node (`server/routes.ts`):**
- `ALPACA_BASE_URL = process.env.ALPACA_BASE_URL || paper` (`routes.ts:600`), used for some read routes
  (`routes.ts:6855-7030`).

**Python:**
- `os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")` in:
  - `bot_engine.py:185` (order POSTs at 4736, 5111, 5147, 5205, 5315, 5342, 5493, 5563, 5586)
  - `options_execution.py:61`
  - `options_manager.py:46`
  - `intraday_shorts.py:52`
  - `tiered_strategy.py:996`
  - `position_sizing.py:791`
  - `options_scanner.py:1346`
- `options_scanner.py:165` hard-codes `ALPACA_TRADE` to paper.
- **`csp_universe.py:180` uses a different variable, `ALPACA_TRADING_URL`.**

**Keys:**
- `ALPACA_KEY` / `ALPACA_SECRET` in Node and Python.
- One Python site reads `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`.
- Data-feed env vars: `ALPACA_DATA_FEED`, `ALPACA_OPTIONS_FEED`, `ALPACA_DATA_URL`.

**Consequence (code-derived):** setting `ALPACA_BASE_URL` to the live host would send **Python** orders live while
**Node** kept monitoring, killing and sweeping the **paper** account. `mode` would still read `"paper"`.

**Where mode is exposed:**
- `/api/bot/account.mode: "paper"` (`bot.ts:1263`) — literal.
- `/api/bot/status.mode: "paper"` (`bot.ts:2317`) — literal.
- boot audit `SYSTEM "Mode: PAPER TRADING (Alpaca)"` (`bot.ts:7900`) — literal.
- The only code that actually inspects the URLs is auto-resume eligibility (`allPaperAlpacaUrls([ALPACA_BASE, env ALPACA_BASE_URL])`,
  `bot.ts:7498`). It surfaces as `autoResume.outcome: "INELIGIBLE"` with reason
  `"Alpaca base URL is not verifiably the PAPER endpoint…"`. It ignores `ALPACA_TRADING_URL`.
- `paper_resume_sync.py:95` independently refuses writes unless `bot_engine.ALPACA_BASE_URL` is paper.
- `/api/health` exposes no mode.

---

## 8. GAPS FOR A COCKPIT (not recorded or not exposed today)

1. **Per-stage heartbeat timestamps:**
   - `lastTier1At`, `lastTier2StartAt`, `lastTier2EndAt`, `lastTier2Result`, `lastTier3At`, `nextTier2At`, current
     interval, and the back-off delay do not exist as fields.
   - Today they can only be inferred from audit text, which holds 2000 rows maximum and has no successful-scan marker
     that the existing checkers recognise.
2. **Cycle durations and latency:**
   - No Tier-1, Tier-2 or Tier-3 cycle duration.
   - No Alpaca REST latency or error-rate counters (by status: 401, 429, 5xx, timeout).
   - No daemon RPC round-trip latency (`_ping` exists but is unused).
   - No order submit→ack→fill latency.
3. **Stream state:** `streamConnected`, connect and subscribe time, last bar received per ticker, bars per minute,
   reconnect count, subscribed ticker list.
4. **Daemon on the public health payload:** daemon alive, uptime, restart count, current (not peak) RSS, zombie dispatch
   count. `/api/health.python` only proves the binary runs.
5. **Python subprocess telemetry:**
   - mutex holder, wait time, queue depth
   - which jobs are on the trivial 10 s path
   - SIGKILL counts per job (only Tier-2 and Tier-3 retrain classify signals)
6. **Morning queue:** contents, size and `morningQueueExecuted` are not exposed.
7. **Order tracking:** `openOrders` (tracked resting orders, age, sweep ETA) is not exposed.
8. **Daily reset:** no timestamp, and the reset itself never runs (Top finding 2). `orderAttemptCounts`, `sweptCspAt`
   and `dailyBlockedTickers` are not exposed.
9. **Unified halt view:** Node latch, bot_engine DD halt, risk_kill_switch kill and manual file, tier master kill,
   circuit breaker, `TIER2-LIMIT` daily-loss skip — each with threshold, peak and since-time. Today they are spread over
   `/api/bot/status`, `/api/monitoring/*`, snapshot and audit lines, with conflicting thresholds.
10. **Effective configuration:**
    - flags: `VOLTRADE_DAEMON_ENABLED`, `VOLTRADE_DISABLE_TIER2/3`, `VOLTRADE_LIQUIDATE_ON_KILL`,
      `VOLTRADE_KILL_AUTO_RESUME`, `VOLTRADE_DAEMON_MAX_RSS_MB`, `VOLTRADE_DAEMON_MAX_INFLIGHT`
    - the effective Alpaca base URL for Node and for each Python module
    - `code_version` on `/api/health` (today it is only on `/api/data/layers.server_version` and the `STARTUP` audit)
11. **Restart history:** restart count, last boot time and cause. Only `uptime_s` and the `STARTUP` audit exist;
    `pipeline_health` snapshots are only written when `/api/health` is polled.
12. **Alert delivery status:** Resend HTTP result, last email sent, and suppressed alerts (no `RESEND_KEY`). Today
    `EMAIL "Alert sent"` is audited even on a rejected send.
13. **Market-session context:** early-close flag and today's actual close time on `market-status`. Today's session is
    not exposed as holiday / half-day / weekend for the loop's own decisions.
14. **Liveness inputs:** the alarm ignores broker-unreadable and Python halts. A cockpit needs "effective trading
    capability", not just `active && !killSwitch`.
15. **Owner intent:** `ownerStopped` is not persisted or exposed. The cockpit cannot tell "stopped by owner" from
    "stopped by weekly limit" (dead) or "inactive after kill OFF".
16. **SSE:** no consumer, no named event types, no heartbeat. It carries only audit entries, so there is no structured
    state push (tier state, stream state, positions).
17. **Rate-limit budget:** no shared Alpaca budget or usage gauge across Node, the daemon and subprocesses.

## 9. Comment/doc vs code discrepancies (quick list)

- CLAUDE.md "Tier 1 stop monitoring ~30s" and `system_config.TIER1_INTERVAL_MS 30_000` vs code **45000** (`bot.ts:7616`).
- CLAUDE.md "self-kills over 1GB RSS": true, but the check uses **peak** RSS (`ru_maxrss`).
- `pythonCall` HEAVY_DAEMON_ONLY says `manage_positions` must not run as a subprocess; Tier 1 runs it as a subprocess
  every 45 s.
- Tier-3 and 4am retrain comments say "900s"; the effective timeout is 10 s (trivial path), and 90 s at most otherwise.
- Overnight schedule (`SCHEDULE` boot audits; the "4am ET: Daily ML retrain + reset" comment) does not run.
- `ml-status` says `retrain_schedule: "Daily at 4am ET"` (`bot.ts:8000`); that never runs. Tier 3 retrains when the
  model is older than 24 h.
- `/api/system/health-check` and snapshot look for `"complete"` in TIER2 lines; that text is never written.
- `/api/bot/status` hard-coded `maxPositionPct: 8`, `maxExposurePct: 95`, `dailyLossLimitPct: 5` disagree with the
  enforced -3% daily guard and the regime-adaptive config.
- `system_status.py` docstring says the route is gated by `DEBUG_ENABLED`; it is ungated and unauthenticated.
- The `bot_engine.scan_market` comment says the daemon request timeout is 60 s; it is 300 s.
- `voltrade_daemon.py` docstring says "bot.ts respawns it" / "supervised by bot.ts or systemd"; `run_with_daemon.sh`
  respawns it.
- Liveness doc (CLAUDE.md) includes "broker-account-unreadable"; the implementation does not.
- The `/api/diag/equity_curve` comment says the oldest day is pruned past 365; the code drops the newest entry.
- Boot `RULES` lines (`bot.ts:7901-7920`), such as "Regular hours → market orders for instant fill", contradict
  `getOrderParams` (new entries are always limit orders).
