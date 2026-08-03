import { Express } from "express";
import { requireAuth, requireOwner } from "./auth";
import { exec, execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";
import WebSocket from "ws";
import { getDisplaySide } from "../shared/inverseEtfs";
import { evaluateDrawdown } from "./drawdownGuard";
import { nextLiveness, loopDark, type LivenessFile } from "./liveness";
import { scannerDegraded } from "./scannerHealth";
import { diagEnabled, checkDiagToken, positionsSummary, sanitizeDiag, orderRow, positionRow, DIAG_PROBES } from "./diag";
import { readArchiveDay } from "./datacoreArchive";
import { recordHealthSnapshot } from "./pipelineHealthHistory";
import * as net from "net";
import { getETHour, getOrderParams, OrderContext } from "./orderParams";
import { buildExitFillPayload } from "./exitFill";
import { buildEntryFillPayload } from "./entryFill";
import { aircraftProviderCompliance } from "./providerCompliance";
import { computeLagMs, lagExceedsThreshold, EVENTLOOP_LAG_CHECK_MS } from "./eventLoopLag";
import { cleanupOrphanedTempFiles, TMP_CLEANUP_INTERVAL_MS, TMP_CLEANUP_AUDIT_THRESHOLD } from "./tmpCleanup";
import { isSlowDbWrite, formatSlowWriteMessage } from "./dbWriteTiming";
import { version as pkgVersion } from "../package.json";
const _execRaw = promisify(exec);
// Force-cap OpenBLAS/MKL threads for ALL child Python processes
// (Railway's container can't handle 32 threads per numpy import)
const _pyEnv = {
  ...process.env,
  OPENBLAS_NUM_THREADS: "2",
  MKL_NUM_THREADS: "2",
  OMP_NUM_THREADS: "2",
  NUMEXPR_MAX_THREADS: "2",
  VECLIB_MAXIMUM_THREADS: "2",
};
// Bump maxBuffer from 1MB (Node default) to 32MB. Full scans log many
// tickers to stdout; hitting 1MB caused exec to SIGKILL the child and throw
// "Command failed" with EMPTY stderr — making root causes impossible to
// diagnose. 32MB handles any realistic scan output comfortably.
const DEFAULT_MAX_BUFFER = 32 * 1024 * 1024;
const execAsync = (cmd: string, opts?: any) =>
  _execRaw(cmd, { env: _pyEnv, maxBuffer: DEFAULT_MAX_BUFFER, ...opts });

// ══════════════════════════════════════════════════════════════════════
// Python Daemon RPC client (OPTIMIZATION 2026-04-20)
// ══════════════════════════════════════════════════════════════════════
// Calls the long-running Python daemon over a Unix socket. Daemon keeps
// numpy/pandas/LightGBM resident in memory so each RPC call is ~3ms
// instead of the ~450ms subprocess startup cost. Safe — if daemon is off
// or unresponsive, callers fall back to the subprocess pattern.
//
// To disable: set env VOLTRADE_DAEMON_ENABLED=false
const DAEMON_SOCKET = process.env.VOLTRADE_DAEMON_SOCKET || "/tmp/voltrade_daemon.sock";
const DAEMON_ENABLED = process.env.VOLTRADE_DAEMON_ENABLED !== "false";
const DAEMON_TIMEOUT_MS = 30000;

async function pythonRpc(method: string, args: any = {}, timeoutMs?: number): Promise<any> {
  if (!DAEMON_ENABLED) {
    return { status: "error", error_message: "Daemon disabled via env" };
  }
  // Per-call timeout override — heavy methods like run_full_scan need more
  // than the default 30s (full scans run up to ~55s under the Python-side
  // SIGALRM cap). Caller passes timeoutMs; falls back to DAEMON_TIMEOUT_MS.
  const effectiveTimeout = timeoutMs && timeoutMs > 0 ? timeoutMs : DAEMON_TIMEOUT_MS;
  return new Promise((resolve) => {
    let buf = "";
    let settled = false;
    const done = (v: any) => { if (!settled) { settled = true; resolve(v); } };
    const client = net.createConnection(DAEMON_SOCKET);
    const timer = setTimeout(() => {
      try { client.destroy(); } catch {}
      done({ status: "error", error_message: "Daemon timeout" });
    }, effectiveTimeout);
    client.on("connect", () => {
      client.write(JSON.stringify({ method, args }) + "\n");
    });
    client.on("data", (chunk) => {
      buf += chunk.toString();
      if (buf.includes("\n")) {
        clearTimeout(timer);
        try {
          done(JSON.parse(buf.split("\n")[0]));
        } catch (e: any) {
          done({ status: "error", error_message: `Parse: ${e.message}` });
        }
        try { client.destroy(); } catch {}
      }
    });
    client.on("error", (err) => {
      clearTimeout(timer);
      done({ status: "error", error_message: `Socket: ${err.message}` });
    });
  });
}

// pythonCall: try daemon first, fall back to subprocess.
// Same return shape regardless of which path was taken.
// Returns { success: boolean, result: any, via: "daemon"|"subprocess"|"none", error?: any }
// On subprocess failure the raw exec error (with .stderr/.stdout/.signal/.code)
// is attached as `error` so callers can preserve the existing diagnostic
// classification instead of getting only an error message string.
async function pythonCall(
  daemonMethod: string,
  daemonArgs: any,
  subprocessCmd: string,
  subprocessOpts: any = {},
  daemonTimeoutMs?: number
): Promise<{ success: boolean; result: any; via: string; error?: any }> {
  // DAEMON-DIAG 2026-04-23: explicit logging of daemon RPC outcomes
  // so we can see why fallthrough happens instead of silent subprocess.
  // Heavy scan methods (run_full_scan) are daemon-ONLY — subprocess
  // path OOMs at ~300MB cold-start on Railway's tight container cap.
  const HEAVY_DAEMON_ONLY = new Set([
    "run_full_scan", "scan_market", "manage_positions",
  ]);
  if (DAEMON_ENABLED) {
    try {
      const r = await pythonRpc(daemonMethod, daemonArgs, daemonTimeoutMs);
      if (r.status === "ok") {
        return { success: true, result: r.result, via: "daemon" };
      }
      // Daemon responded but with non-ok status — log it
      console.warn(`[pythonCall] daemon ${daemonMethod} status=${r.status} ` +
        `error=${(r.error_message || "").slice(0, 200)}`);
      if (HEAVY_DAEMON_ONLY.has(daemonMethod)) {
        // Don't fall back to subprocess for heavy methods — will OOM
        return { success: false, result: { error: `daemon ${daemonMethod} failed: ${r.error_message}` }, via: "daemon-failed" };
      }
    } catch (rpcErr: any) {
      console.warn(`[pythonCall] daemon ${daemonMethod} threw: ${(rpcErr?.message || "").slice(0, 200)}`);
      if (HEAVY_DAEMON_ONLY.has(daemonMethod)) {
        return { success: false, result: { error: `daemon RPC threw: ${rpcErr?.message}` }, via: "daemon-error" };
      }
    }
  }
  // Subprocess fallback for LIGHT methods only (kill_switches, ml_status, etc.)
  try {
    const { stdout } = await execPythonSerialized(subprocessCmd, subprocessOpts);
    return { success: true, result: JSON.parse(stdout.trim()), via: "subprocess" };
  } catch (e: any) {
    return { success: false, result: { error: e.message }, via: "none", error: e };
  }
}

// REPAIR 2026-07-06 (R12-D2): record a FULL exit into trade_feedback so the
// matching open entry record gets a real outcome + pnl_pct (Bug #13's exit
// machinery in ml_model_v2.track_fill finally gets a caller). Mirrors the
// entry-side tmp-file + daemon-first pattern used at both entry sites.
// Recording only — runs strictly AFTER the (frozen) order POST succeeded.
function recordExitFill(payload: ReturnType<typeof buildExitFillPayload>) {
  try {
    const tmp = `/tmp/fill_x_${payload.ticker}_${Date.now()}.json`;
    fs.writeFileSync(tmp, JSON.stringify(payload));
    pythonCall(
      "track_fill", payload,
      `python3 -c "import json, os; from ml_model_v2 import track_fill; d=json.load(open('${tmp}')); os.remove('${tmp}'); track_fill(d)"`,
      { timeout: 5000 }
    ).then((r) => {
      if (r.via === "daemon") {
        try { fs.unlinkSync(tmp); } catch {}
      }
    }).catch(() => {});
  } catch (err: any) { console.error("[bot]", err?.message || err); }
}



// ─── Python Subprocess Serialization (OOM fix) ─────────────────────────────
// At most 1 heavy Python subprocess at a time. Each Python invocation imports
// numpy/pandas/sklearn/lightgbm (~100-150MB). Concurrent subprocesses push the
// container past its memory limit. This mutex ensures sequential execution.
//
// LOCK SAFETY CONTRACT:
//   - Lock is ALWAYS released in finally{} even if execAsync throws/hangs
//   - Watchdog force-releases stale locks every 30s (was 60s)
//   - Stale threshold is hardTimeout + 10s grace (was fixed 120s)
//   - TRIVIAL_PYTHON commands bypass the mutex entirely
let pythonRunning = false;
let pythonLockedAt = 0;
let pythonLockHardTimeout = 90000;  // tracks the hardTimeout of current holder

// Trivial commands that don't import heavy libs — bypass mutex.
// Pattern match: `python3 -c "print('ok')"` and similar <100-byte pings
function isTrivialPython(cmd: string): boolean {
  if (cmd.length > 150) return false;
  if (/import\s+(numpy|pandas|sklearn|lightgbm|scipy|torch)/.test(cmd)) return false;
  return true;
}

async function execPythonSerialized(cmd: string, opts?: any) {
  const maxWait = opts?.timeout || 30000;
  const hardTimeout = Math.min(opts?.timeout || 90000, 90000);

  // Bypass mutex for trivial pings (e.g., /api/health)
  if (isTrivialPython(cmd)) {
    return execAsync(cmd, { ...opts, timeout: Math.min(hardTimeout, 10000), killSignal: 'SIGKILL' });
  }

  const start = Date.now();
  while (pythonRunning) {
    // Eager stale-lock detection: if the current holder's hardTimeout + 10s grace
    // has elapsed, force-release. This is tighter than the old fixed 120s.
    const staleAfter = pythonLockHardTimeout + 10000;
    if (pythonLockedAt > 0 && Date.now() - pythonLockedAt > staleAfter) {
      console.error("[python-mutex] Force-releasing stale lock held for",
        Math.round((Date.now() - pythonLockedAt) / 1000),
        "seconds (stale threshold:", Math.round(staleAfter/1000), "s)");
      pythonRunning = false;
      pythonLockedAt = 0;
      pythonLockHardTimeout = 90000;
      break;
    }
    if (Date.now() - start > maxWait) {
      throw new Error("Python mutex timeout — another Python process is holding the lock");
    }
    await new Promise(r => setTimeout(r, 500));
  }
  pythonRunning = true;
  pythonLockedAt = Date.now();
  pythonLockHardTimeout = hardTimeout;
  try {
    return await execAsync(cmd, { ...opts, timeout: hardTimeout, killSignal: 'SIGKILL' });
  } finally {
    // CRITICAL: always release — even if execAsync throws, hangs, or the
    // SIGKILL fires before Node cleans up the child's stdio.
    pythonRunning = false;
    pythonLockedAt = 0;
    pythonLockHardTimeout = 90000;
  }
}

// ─── Python Mutex Watchdog ──────────────────────────────────────────────────
// Safety net: force-release stale Python locks every 30s (was 60s).
// Tighter interval catches stuck locks faster. Stale threshold scales to the
// lock holder's declared hardTimeout rather than a fixed 120s.
setInterval(() => {
  if (!pythonRunning || pythonLockedAt === 0) return;
  const heldMs = Date.now() - pythonLockedAt;
  const staleAfter = pythonLockHardTimeout + 10000;
  if (heldMs > staleAfter) {
    console.error("[python-mutex] Watchdog: force-releasing stale lock held for",
      Math.round(heldMs / 1000), "seconds (stale threshold:", Math.round(staleAfter/1000), "s)");
    pythonRunning = false;
    pythonLockedAt = 0;
    pythonLockHardTimeout = 90000;
  }
}, 30000);

// ─── Temp File Cleanup (OOM fix) ────────────────────────────────────────────
// Orphaned temp files from fire-and-forget Python subprocesses that timed out
// or failed. Clean up every 10 minutes; delete files older than 5 minutes.
// REPAIR 2026-07-12 (KNOWN BROKEN #18 follow-up): this used to run the scan
// with fs.readdirSync/statSync/unlinkSync directly on the event loop — a
// synchronous fs sweep on an exact 600000ms period matching the cadence
// observed between production EVENTLOOP-LAG stalls (59-75s!) and their
// coincident STREAM-DISCONNECT entries. Moved to server/tmpCleanup.ts's
// fs.promises implementation, which cannot block the event loop regardless
// of how many orphaned files have accumulated. TMP_CLEANUP_AUDIT_THRESHOLD
// diagnostic: an audit line only fires when the scan is anomalously large,
// so a future session can see whether a real backlog was driving this.
setInterval(() => {
  cleanupOrphanedTempFiles().then(({ scanned, deleted }) => {
    if (scanned >= TMP_CLEANUP_AUDIT_THRESHOLD) {
      audit("TMP-CLEANUP", `Scanned ${scanned} orphan-candidate temp files in /tmp, deleted ${deleted} — anomalously large (KNOWN BROKEN #18 diagnostic)`);
    }
  }).catch(() => {});
}, TMP_CLEANUP_INTERVAL_MS);

// ─── Alpaca Config ──────────────────────────────────────────────────────────
const ALPACA_BASE = "https://paper-api.alpaca.markets";
const ALPACA_KEY = process.env.ALPACA_KEY || "";
const ALPACA_SECRET = process.env.ALPACA_SECRET || "";
if (!ALPACA_KEY || !ALPACA_SECRET) console.warn("[WARN] ALPACA_KEY/ALPACA_SECRET not set — trading disabled");

async function alpaca(path: string, opts: any = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000); // 15s timeout
  try {
    const r = await fetch(`${ALPACA_BASE}${path}`, {
      ...opts,
      signal: controller.signal,
      headers: {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
        ...opts.headers,
      },
    });
    if (!r.ok) {
      const body = await r.text().catch(() => "");
      throw new Error(`Alpaca ${r.status}: ${body || r.statusText}`);
    }
    return r.json();
  } finally {
    clearTimeout(timeout);
  }
}

// ─── Bot State ──────────────────────────────────────────────────────────────
// Phase 6: Security guardrails (absolute ceilings — dynamic sizing handles the real math)
const DAILY_LOSS_LIMIT = -3; // percent — hard circuit breaker
const MAX_POSITION_SIZE = 0.20; // Safety ceiling — real sizing from system_config.py (3-15%)
const MAX_TOTAL_EXPOSURE = 0.30; // 30% of equity for ACTIVE trades (excludes QQQ floor + third leg)
const MAX_POSITIONS = 8; // absolute ceiling (dynamic sizing uses portfolio heat)
const STOP_LOSS_PCT = 0.15; // Emergency backstop only — real stops from system_config.py (6% ATR-based)
const TAKE_PROFIT_PCT = 0.25; // Emergency ceiling — real TP from system_config.py (12% ATR-based)
// NOTE: These are EMERGENCY SAFETY NETS only. The real limits come from
// system_config.py's get_adaptive_params() which adapts by regime:
//   BULL: 95% max exposure, 15% position size, 12% TP, 6% SL
//   NEUTRAL: 95% (QQQ floor handles it), no active stock trades
//   CAUTION: 60% exposure, 10% position size
//   BEAR: 50% exposure (third leg only)
//   PANIC: 30% exposure
// The QQQ floor deploys 70-90% of equity — these safety nets MUST
// be above that or they block the core strategy.
// MAX_TOTAL_EXPOSURE here (0.30) is for ACTIVE satellite trades only
// (floor + legacy leg tickers excluded via FLOOR_AND_LEG_TICKERS below).
// system_config.py's 0.95 controls total portfolio including floor.

// Passive floor-basket + legacy leg tickers — managed by bot_engine.py's
// _manage_spy_floor()/_manage_defensive_floor() regime-driven rebalancing,
// not by the active-trade stop-loss/take-profit/scale-out/time-stop/heat-cap
// machinery below. MUST mirror bot_engine.py's FLOOR_AND_LEG_TICKERS
// (system_config.BASE_CONFIG's FLOOR_BASKET keys + FLOOR_TICKER +
// DEFENSIVE_FLOOR_TICKER, plus legacy SVXY) — update both together whenever
// FLOOR_BASKET changes.
// R-FIX 2026-07-17: this was hardcoded to {"QQQ","SVXY","SPY"} in three
// separate places here and never updated when FLOOR_BASKET (SMH/KWEB/VXUS)
// shipped 2026-04-22, nor for DEFENSIVE_FLOOR_TICKER (GLD) — syncMonitoredPositions
// was letting those into monitoredPositions, so checkPositionOnTick's WS-driven
// exit logic applied active TIME STOP/STOP LOSS checks to them, causing
// repeated erroneous round-trip sells on the passive floor basket (5x VXUS +
// 1x SMH in one live session). executeTrades' totalDeployed heat-cap also
// over-counted floor market value as active capital. See
// research/experiments.md 2026-07-17.
const FLOOR_AND_LEG_TICKERS = new Set(["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"]);

// Defensive-floor candidates that were evaluated and rejected in favor of GLD
// (see system_config.py DEFENSIVE_FLOOR_TICKER comment: "TLT/VIXY/SH/SQQQ all
// decay") — used only by syncMonitoredPositions' TIME-EXIT staleness check, in
// case one is ever held again. Not part of FLOOR_AND_LEG_TICKERS itself.
const LEGACY_DEFENSIVE_CANDIDATE_TICKERS = new Set(["VTI", "IWM", "TLT", "IEF", "SCHP"]);

// OCC option symbol detector (e.g. XLK260424P00149000) — same shape check
// already used by /api/bot/bars/:ticker below. The equity market-data stream
// (wss://stream.data.alpaca.markets/v2/iex) only accepts stock symbols; a
// "subscribe"/"unsubscribe" message containing an option symbol makes Alpaca
// reject the WHOLE batched message with code=400 "invalid syntax" — silently
// dropping any legitimate equity tickers bundled in the same call. Used to
// keep option positions out of the WS bars subscription (they still get
// full position-monitor tracking — kill/warn/time-exit — just not a live
// bars stream, which doesn't exist for options on this feed anyway).
function isOptionSymbol(ticker: string): boolean {
  const t = String(ticker).toUpperCase();
  return t.length > 10 || /^[A-Z]+\d{6}[CP]\d{8}$/.test(t);
}

// ─── ET Hour Helper + Order Params — extracted to ./orderParams (see top imports) for unit testing ──

const state = {
  active: true,  // Bot starts automatically — always on unless killed
  killSwitch: false,
  dailyPnL: 0,
  dailyLossLimit: DAILY_LOSS_LIMIT,
  positionSizeMultiplier: 1.0, // Legacy — diagnostics can still reduce this as emergency brake
  minScoreThreshold: 65,
  diagCycleCount: 0,
  consecutiveStopLosses: 0,
  circuitBreakerUntil: 0,  // epoch ms — paused until this time
  alpacaFailCount: 0,       // consecutive Alpaca ping failures
  morningQueueExecuted: false, // Track if we ran the morning queue today
  equityPeak: 0,              // High water mark for max drawdown kill switch
  maxDrawdownPct: -10,        // Kill switch triggers at -10% from peak
};

// ─── SSE Clients ────────────────────────────────────────────────────────────
const sseClients: Set<any> = new Set();

function broadcastSSE(data: any) {
  for (const client of sseClients) {
    try { client.write(`data: ${JSON.stringify(data)}\n\n`); } catch {}
  }
}

function audit(action: string, detail: string) {
  const entry = { time: new Date().toISOString(), type: action, action, detail, message: detail };
  // OOM fix: removed in-memory auditLog accumulation — always read from SQLite instead
  console.log(`[BOT] ${action}: ${detail}`);
  broadcastSSE(entry);
  // Persist to database (survives deploys)
  try { persistAudit(action, detail); } catch {}
}

// ─── Notifications ──────────────────────────────────────────────────────────
interface Notification {
  time: string;
  type: string;
  message: string;
  read: boolean;
}

const notifications: Notification[] = [];

function notify(type: string, message: string) {
  notifications.unshift({ time: new Date().toISOString(), type, message, read: false });
  if (notifications.length > 100) notifications.length = 100;
  console.log(`[NOTIFY] ${type}: ${message}`);
}

// ─── Performance / Learning Loop ────────────────────────────────────────────
interface TradeResult {
  ticker: string;
  side: string;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  pnlPct: number;
  strategy: string;
  score: number;
  rulesScore: number | null;  // Score before ML blend (for attribution)
  mlScore: number | null;     // ML-only score (for attribution)
  instrument: string;         // stock / etf / options
  entryFeatures: any;         // 52-feature snapshot at entry (for ML exit model)
  exitContext: any;           // Stop phase, R-multiple, ATR at exit
  holdingDays: number;
  timestamp: string;
}

const tradeResults: TradeResult[] = [];

// In-memory equity curve (keeps up to 1 year of daily data)
// UI FIX 2026-05-05 batch 9: equity curve was in-memory only. Every Railway
// redeploy wiped it, so the dashboard chart only ever showed "today" or
// at most 1-2 days of history. Now load from disk on startup, persist
// after each daily snapshot. Stays in /data/voltrade so it survives.
const EQUITY_CURVE_PATH = "/data/voltrade/voltrade_equity_curve.json";
const EQUITY_CURVE_FALLBACK = "/tmp/voltrade_equity_curve.json";
const equityCurve: Array<{ date: string; value: number; pnl: number }> = (() => {
  for (const p of [EQUITY_CURVE_PATH, EQUITY_CURVE_FALLBACK]) {
    try {
      if (fs.existsSync(p)) {
        const raw = fs.readFileSync(p, "utf8");
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          // Sanity-check shape — only keep records that look like equity points
          return parsed.filter((d: any) =>
            d && typeof d.date === "string" && typeof d.value === "number"
          ).slice(0, 365);
        }
      }
    } catch (e: any) {
      console.error(`[equity-curve] could not load ${p}:`, e?.message || e);
    }
  }
  return [];
})();

function saveEquityCurve() {
  for (const p of [EQUITY_CURVE_PATH, EQUITY_CURVE_FALLBACK]) {
    try {
      // Make sure parent dir exists; ignore if already there
      const dir = p.substring(0, p.lastIndexOf("/"));
      try { fs.mkdirSync(dir, { recursive: true }); } catch {}
      fs.writeFileSync(p, JSON.stringify(equityCurve));
      return; // Stop on first successful write
    } catch (e: any) {
      // Try fallback path
      console.error(`[equity-curve] could not save to ${p}:`, e?.message || e);
    }
  }
}

// KNOWN BROKEN #7 FIX 2026-07-03 (human-approved): the max-drawdown kill
// switch's high-water mark (state.equityPeak) was in-memory only, so every
// deploy/restart re-based drawdown protection from current equity — frequent
// autonomous deploys silently defanged it. Persist the peak exactly like the
// equity curve above (volume path + /tmp fallback), restore on boot. The
// halt logic itself is untouched — this only preserves its input.
const EQUITY_PEAK_PATH = "/data/voltrade/voltrade_equity_peak.json";
const EQUITY_PEAK_FALLBACK = "/tmp/voltrade_equity_peak.json";

function loadEquityPeak(): number {
  for (const p of [EQUITY_PEAK_PATH, EQUITY_PEAK_FALLBACK]) {
    try {
      if (fs.existsSync(p)) {
        const parsed = JSON.parse(fs.readFileSync(p, "utf8"));
        const peak = parseFloat(parsed?.equityPeak);
        if (Number.isFinite(peak) && peak > 0) return peak;
      }
    } catch (e: any) {
      console.error(`[equity-peak] could not load ${p}:`, e?.message || e);
    }
  }
  return 0;
}

function saveEquityPeak() {
  for (const p of [EQUITY_PEAK_PATH, EQUITY_PEAK_FALLBACK]) {
    try {
      const dir = p.substring(0, p.lastIndexOf("/"));
      try { fs.mkdirSync(dir, { recursive: true }); } catch {}
      fs.writeFileSync(p, JSON.stringify({ equityPeak: state.equityPeak, savedAt: new Date().toISOString() }));
      return;
    } catch (e: any) {
      console.error(`[equity-peak] could not save to ${p}:`, e?.message || e);
    }
  }
}

// Restore the high-water mark on boot so restarts don't reset drawdown
// protection. If no file exists yet (first boot), the existing lazy seeding
// at the kill-switch sites behaves exactly as before.
state.equityPeak = loadEquityPeak();

// R16 2026-07-07: the kill-switch LATCH was memory-only — every deploy booted
// the bot ACTIVE, opening a live trading window until the next tier-1 equity
// read re-killed it (observed 2026-07-07 during the Alpaca account-state
// incident: ~50-minute window, $21.6k of buys, on an account whose state was
// in dispute). Persist the latch exactly like the peak above: restore on
// boot, save on every transition. The owner /api/bot/kill toggle remains the
// ONLY clear path — a deploy can no longer un-kill the bot.
const KILL_SWITCH_PATH = "/data/voltrade/voltrade_kill_switch.json";
const KILL_SWITCH_FALLBACK = "/tmp/voltrade_kill_switch.json";

function loadKillSwitch(): boolean {
  for (const p of [KILL_SWITCH_PATH, KILL_SWITCH_FALLBACK]) {
    try {
      if (fs.existsSync(p)) {
        const parsed = JSON.parse(fs.readFileSync(p, "utf8"));
        if (typeof parsed?.killSwitch === "boolean") return parsed.killSwitch;
      }
    } catch (e: any) {
      console.error(`[kill-switch] could not load ${p}:`, e?.message || e);
    }
  }
  return false;
}

function saveKillSwitch(reason: string) {
  for (const p of [KILL_SWITCH_PATH, KILL_SWITCH_FALLBACK]) {
    try {
      const dir = p.substring(0, p.lastIndexOf("/"));
      try { fs.mkdirSync(dir, { recursive: true }); } catch {}
      fs.writeFileSync(p, JSON.stringify({ killSwitch: state.killSwitch, reason, savedAt: new Date().toISOString() }));
      return;
    } catch (e: any) {
      console.error(`[kill-switch] could not save to ${p}:`, e?.message || e);
    }
  }
}

state.killSwitch = loadKillSwitch();
if (state.killSwitch) {
  console.log("[kill-switch] restored ON from disk — bot boots halted; only the owner /api/bot/kill toggle clears it");
}

// ─── LIVENESS ALARM (Amendment 2 runtime half, human-approved 2026-07-04) ──
// The loop paused/halted >2 market hours (or 24h wall-clock) degrades
// /api/health. Heartbeat persisted like equityPeak so deploys never reset
// the darkness clock; Railway's healthcheck polling drives the assessment,
// the 60s touch below keeps the stamp fresh regardless.
const LIVENESS_PATH = "/data/voltrade/voltrade_liveness.json";
const LIVENESS_FALLBACK = "/tmp/voltrade_liveness.json";

function loadLiveness(): LivenessFile | null {
  for (const p of [LIVENESS_PATH, LIVENESS_FALLBACK]) {
    try {
      if (fs.existsSync(p)) {
        const parsed = JSON.parse(fs.readFileSync(p, "utf8"));
        if (Number.isFinite(parsed?.lastActiveAt)) return { lastActiveAt: parsed.lastActiveAt };
      }
    } catch (e: any) {
      console.error(`[liveness] could not load ${p}:`, e?.message || e);
    }
  }
  return null;
}

function saveLiveness(v: LivenessFile) {
  for (const p of [LIVENESS_PATH, LIVENESS_FALLBACK]) {
    try {
      const dir = p.substring(0, p.lastIndexOf("/"));
      try { fs.mkdirSync(dir, { recursive: true }); } catch {}
      fs.writeFileSync(p, JSON.stringify({ ...v, savedAt: new Date().toISOString() }));
      return;
    } catch (e: any) {
      console.error(`[liveness] could not save to ${p}:`, e?.message || e);
    }
  }
}

let livenessState: LivenessFile | null = loadLiveness();

setInterval(() => {
  try {
    // active → fresh stamp (new object) → saved each minute; inactive →
    // same object back (or a one-time seed) → no disk churn while dark.
    const next = nextLiveness(livenessState, state.active && !state.killSwitch, Date.now());
    if (next !== livenessState) { livenessState = next; saveLiveness(next); }
  } catch {}
}, 60_000);

// ─── Email Alerts (Resend) ────────────────────────────────────────────────────
const RESEND_KEY = process.env.RESEND_KEY || "";
const ALERT_EMAIL = process.env.ALERT_EMAIL || "mctils12@gmail.com";

async function sendEmailAlert(subject: string, body: string) {
  if (!RESEND_KEY) return;
  try {
    await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { "Authorization": `Bearer ${RESEND_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        from: "VolTradeAI <onboarding@resend.dev>",
        to: [ALERT_EMAIL],
        subject: `[VolTradeAI] ${subject}`,
        html: `<div style="font-family:monospace;background:#0a0a0a;color:#00ffd5;padding:20px;border-radius:8px;">
          <h2 style="color:#ff4444;">${subject}</h2>
          <p>${body}</p>
          <p style="color:#666;margin-top:20px;">— VolTradeAI Automated Alert</p>
        </div>`,
      }),
    });
    audit("EMAIL", `Alert sent: ${subject}`);
  } catch (err: any) {
    console.error("[email-alert]", err?.message || err);
  }
}

// ─── Graceful Shutdown ───────────────────────────────────────────────────────
async function gracefulShutdown(signal: string) {
  audit("SHUTDOWN", `Received ${signal} — cancelling open orders and shutting down...`);
  try {
    await alpaca("/v2/orders", { method: "DELETE" }); // Cancel ALL open orders
    audit("SHUTDOWN", "All open orders cancelled");
  } catch (err: any) {
    audit("SHUTDOWN-ERROR", `Failed to cancel orders: ${err?.message}`);
  }
  persistAudit("SHUTDOWN", `Graceful shutdown on ${signal}`);
  process.exit(0);
}

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

// Strategy weights — start equal, adjusted by learning loop
const strategyWeights = {
  momentum: 0.25,
  mean_reversion: 0.20,
  vrp: 0.25,
  squeeze: 0.15,
  volume: 0.15,
};

async function trackClosedTrades() {
  try {
    const orders = await alpaca("/v2/orders?status=closed&limit=50");
    if (!Array.isArray(orders)) return;

    const knownIds = new Set(tradeResults.map(t => t.ticker + t.timestamp));

    for (const order of orders) {
      if (!order.filled_at) continue;
      const key = order.symbol + order.filled_at;
      if (knownIds.has(key)) continue;

      const entryPrice = parseFloat(order.filled_avg_price || 0);
      const exitPrice = parseFloat(order.filled_avg_price || 0); // simplified — same fill price

      // OOM fix: strip entryFeatures/exitContext from in-memory array — they're
      // only needed for ML feedback (written to file), not for the dashboard
      tradeResults.unshift({
        ticker: order.symbol,
        side: order.side,
        entryPrice,
        exitPrice,
        pnl: 0, // Will be updated when matched buy/sell pairs found
        pnlPct: 0,
        strategy: "auto",
        score: 0,
        rulesScore: null,  // Populated when trade originates from scan
        mlScore: null,     // Populated when trade originates from scan
        instrument: "stock",
        entryFeatures: null,
        exitContext: null,
        holdingDays: 0,
        timestamp: order.filled_at,
      });

      if (tradeResults.length > 200) tradeResults.length = 200;
    }

    // FIX 2026-04-20 (Bug #25 v2 — COMPLETE): Compute real pnl_pct by pairing
    // buys and sells on the same symbol. BOTH sides get backfilled (previously
    // only sells did). Also cross-reference scan-origin trades to recover
    // entry_features for ML training labels.
    try {
      const bySymbol: Record<string, { buys: any[]; sells: any[] }> = {};
      for (const t of tradeResults.slice(0, 100)) {
        const sym = t.ticker;
        if (!bySymbol[sym]) bySymbol[sym] = { buys: [], sells: [] };
        const sideLower = (t.side || "").toLowerCase();
        if (sideLower === "buy") bySymbol[sym].buys.push(t);
        else if (sideLower === "sell" || sideLower === "sell_short") bySymbol[sym].sells.push(t);
      }
      for (const sym of Object.keys(bySymbol)) {
        const { buys, sells } = bySymbol[sym];
        if (buys.length === 0 || sells.length === 0) continue;
        // Average entry from buys, average exit from sells
        const avgBuy = buys.reduce((s, b) => s + b.entryPrice, 0) / buys.length;
        const avgSell = sells.reduce((s, b) => s + b.exitPrice, 0) / sells.length;
        if (avgBuy > 0 && avgSell > 0) {
          const realPnlPct = ((avgSell - avgBuy) / avgBuy) * 100;
          const roundedPnl = Math.round(realPnlPct * 100) / 100;
          // Bug #25 completion 2026-04-20: backfill BOTH sides with real pnl
          // Previously only sells got pnl; buys stayed at 0 and couldn't be
          // used for ML training because they appeared as flat trades.
          for (const s of sells) {
            if (s.pnlPct === 0 || s.pnlPct === null) {
              s.pnlPct = roundedPnl;
              s.entryPrice = avgBuy;
              s.exitPrice = avgSell;
            }
          }
          for (const b of buys) {
            if (b.pnlPct === 0 || b.pnlPct === null) {
              b.pnlPct = roundedPnl;
              b.entryPrice = avgBuy;
              b.exitPrice = avgSell;
            }
          }
        }
      }
    } catch (pairErr: any) { /* non-critical */ }

    // Adjust strategy weights based on recent performance
    adjustStrategyWeights();

    // Self-improving: feed closed trades back to ML training data
    //
    // FIX 2026-04-20 (Bug #25b): Only write records with real pnl_pct.
    // Previously every record had pnl_pct=0 (sentinel for unknown), which
    // ML training interpreted as "loss" — poisoning the training set.
    // Also require entry_features to be present; without features, the
    // record is useless for training even if pnl_pct is correct.
    if (tradeResults.length > 0) {
      try {
        const feedbackData = tradeResults.slice(0, 20)
          .filter(t => t.pnlPct !== 0 && t.pnlPct !== null && t.entryFeatures != null)  // skip garbage
          .map(t => ({
          ticker: t.ticker,
          side: t.side,
          pnl_pct: t.pnlPct,
          holding_days: t.holdingDays,
          strategy: t.strategy,
          score: t.score,
          rules_score: t.rulesScore || null,
          ml_score: t.mlScore || null,
          blended_score: t.score,
          won: t.pnlPct > 0 ? 1 : 0,
          instrument: t.instrument || "stock",
          entry_features: t.entryFeatures || null,   // 52-feature snapshot at entry
          exit_context: t.exitContext || null,        // Stop phase, R-multiple, ATR at exit
          timestamp: t.timestamp,
          // REPAIR 2026-07-11: was hardcoded "1.0.34" (the version at
          // Bug-25's fix) forever — same bug as track_fill's stamping,
          // fixed there too. This block is currently dead code (KNOWN
          // BROKEN #12b: entryFeatures is hardcoded null upstream), but
          // fixing the stale literal costs nothing and avoids leaving a
          // second copy of the same bug pattern in the repo.
          code_version: pkgVersion,
        }));
        if (feedbackData.length === 0) {
          // Nothing worth training on this cycle
          return;
        }
        const fbTmpPath = `/tmp/fb_${Date.now()}.json`;
        fs.writeFileSync(fbTmpPath, JSON.stringify(feedbackData));
        execPythonSerialized(`python3 -c "
import json, os
try:
    from storage_config import TRADE_FEEDBACK_PATH
except ImportError:
    TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'
feedback = json.load(open('${fbTmpPath}'))
os.remove('${fbTmpPath}')
existing = []
if os.path.exists(TRADE_FEEDBACK_PATH):
    try:
        with open(TRADE_FEEDBACK_PATH) as f:
            existing = json.load(f)
    except: pass
existing.extend(feedback)
existing = existing[-500:]
with open(TRADE_FEEDBACK_PATH, 'w') as f:
    json.dump(existing, f)
print(json.dumps({'saved': len(feedback), 'total': len(existing)}))
"`, { timeout: 5000 }).catch(() => {});
      } catch (err: any) { console.error("[bot]", err?.message || err); }
    }
  } catch (e: any) {
    audit("LEARN-ERROR", `Track closed trades failed: ${e.message}`);
  }
}

let lastWeightAdjustLog = 0; // Throttle LEARN audit log — max once per 30 min
// Deployment timestamp — only trades AFTER this should influence weight adjustments.
// All pre-deploy trades ran on broken code (pre-27-bug-fix, pre-scale-out,
// pre-HEAT-CAP fix) and their outcomes reflect bugs, not strategy quality.
const DEPLOY_TIMESTAMP = new Date().toISOString();

function adjustStrategyWeights() {
  // Only learn from trades that closed AFTER this deployment (clean code)
  // Pre-deploy trades ran on broken code — their win/loss reflects bugs, not signals
  const recent = tradeResults
    .filter(t => t.timestamp && t.timestamp >= DEPLOY_TIMESTAMP)
    .slice(0, 20);
  if (recent.length < 5) return; // Need 5+ post-deploy trades before adjusting

  const winRate = recent.filter(t => t.pnl > 0).length / recent.length;
  if (winRate < 0.4) {
    // Adjust weights silently every cycle, but only LOG once per 30 minutes
    strategyWeights.vrp = Math.min(0.40, strategyWeights.vrp + 0.02);
    strategyWeights.momentum = Math.max(0.15, strategyWeights.momentum - 0.01);
    strategyWeights.volume = Math.max(0.10, strategyWeights.volume - 0.01);
    if (Date.now() - lastWeightAdjustLog > 1800000) { // 30 minutes
      audit("LEARN", `Win rate ${(winRate * 100).toFixed(0)}% (${recent.length} post-deploy trades) — shifting weight toward VRP/squeeze (momentum: ${(strategyWeights.momentum * 100).toFixed(0)}%, VRP: ${(strategyWeights.vrp * 100).toFixed(0)}%)`);
      lastWeightAdjustLog = Date.now();
    }
  } else if (winRate > 0.65) {
    strategyWeights.momentum = Math.min(0.30, strategyWeights.momentum + 0.01);
    if (Date.now() - lastWeightAdjustLog > 1800000) {
      audit("LEARN", `Win rate ${(winRate * 100).toFixed(0)}% (${recent.length} post-deploy trades) — restoring balanced weights (momentum: ${(strategyWeights.momentum * 100).toFixed(0)}%)`);
      lastWeightAdjustLog = Date.now();
    }
  }
}

function recordDailyEquity() {
  alpaca("/v2/account").then((acct: any) => {
    if (!acct || !acct.portfolio_value) return;
    const date = new Date().toISOString().split("T")[0];
    const value = parseFloat(acct.portfolio_value);
    const pnl = parseFloat(acct.equity) - parseFloat(acct.last_equity);

    // Only record once per day
    const last = equityCurve[0];
    if (last && last.date === date) {
      last.value = value;
      last.pnl = pnl;
    } else {
      equityCurve.unshift({ date, value, pnl });
      if (equityCurve.length > 365) equityCurve.shift();
    }
    // UI FIX batch 9: persist after every update so we don't lose history
    // on the next redeploy.
    saveEquityCurve();
  }).catch(() => {});
}

// Phase 6: Pre-trade security check
async function checkTradeAllowed(portfolioValue: number, tradeValue: number): Promise<{ allowed: boolean; reason?: string }> {
  // 1. Kill switch check
  if (state.killSwitch) {
    return { allowed: false, reason: "Kill switch is active. All trading halted." };
  }
  // 2. Daily loss limit check
  if (state.dailyPnL !== 0 && portfolioValue > 0) {
    const dailyPnLPct = (state.dailyPnL / portfolioValue) * 100;
    if (dailyPnLPct <= DAILY_LOSS_LIMIT) {
      return { allowed: false, reason: `Daily loss limit of ${DAILY_LOSS_LIMIT}% exceeded. Bot stopped for today.` };
    }
  }
  // 3. Position size check
  if (portfolioValue > 0 && tradeValue / portfolioValue > MAX_POSITION_SIZE) {
    return { allowed: false, reason: `Trade exceeds max position size of ${MAX_POSITION_SIZE * 100}% of portfolio.` };
  }
  // 4. Total exposure check (simplified)
  return { allowed: true };
}

// ─── Options order — LIVE execution via Alpaca (Level 3 approved) ────────────
async function placeOptionsOrder(
  ticker: string,
  optionType: "call" | "put",
  strike: number,
  expiry: string,
  side: "buy" | "sell",
  qty: number
) {
  // Alpaca options use OCC symbol format: AAPL260418C00250000
  const strikeStr = (strike * 1000).toFixed(0).padStart(8, "0");
  const dateStr = expiry.replace(/-/g, "").substring(2); // YYMMDD
  const typeChar = optionType === "call" ? "C" : "P";
  const occSymbol = `${ticker.toUpperCase()}${dateStr}${typeChar}${strikeStr}`;

  audit("OPTIONS", `${side.toUpperCase()} ${qty}x ${ticker} $${strike} ${optionType} exp:${expiry} [OCC: ${occSymbol}]`);

  try {
    const orderPayload = {
      symbol: occSymbol,
      qty: String(qty),
      side: side,
      type: "limit" as const,
      time_in_force: "day" as const,
      limit_price: "0",  // Will be set by Python execution path
    };

    // Use Python options_execution for smart pricing and submission
    const tradeData = {
      occ_symbol: occSymbol,
      ticker: ticker,
      option_type: optionType,
      strike: strike,
      expiry: expiry,
      side: side,
      qty: qty,
    };
    const optTmpPath = `/tmp/opt_order_${ticker}_${Date.now()}.json`;
    fs.writeFileSync(optTmpPath, JSON.stringify(tradeData));

    const { stdout, stderr } = await execPythonSerialized(
      `python3 -c "
import json, sys, os
sys.path.insert(0, '.')
from options_execution import submit_options_order
trade = json.load(open('${optTmpPath}'))
os.remove('${optTmpPath}')
result = submit_options_order(trade)
print(json.dumps(result))
"`,
      { timeout: 15000 }
    );

    const result = JSON.parse(stdout.trim());
    
    if (result.status === "submitted" || result.status === "filled") {
      audit("OPTIONS-EXEC", `FILLED: ${side.toUpperCase()} ${qty}x ${occSymbol} | order=${result.order_id || "?"}`);
      notify("trade", `OPTIONS: ${side.toUpperCase()} ${qty}x ${ticker} $${strike} ${optionType.toUpperCase()} (${expiry})`);
      return { status: result.status, occ_symbol: occSymbol, order_id: result.order_id, detail: result.detail };
    } else {
      audit("OPTIONS-ERROR", `Order rejected: ${result.detail || result.error || "unknown"}`);
      return { status: "error", occ_symbol: occSymbol, message: result.detail || "Order rejected" };
    }
  } catch (err: any) {
    const errMsg = err?.stderr?.slice(-200) || err?.message || "Unknown error";
    audit("OPTIONS-ERROR", `Execution failed: ${errMsg}`);
    return { status: "error", occ_symbol: occSymbol, message: errMsg };
  }
}

// ─── Routes ─────────────────────────────────────────────────────────────────
// ─── Persistent Audit Log (survives deploys) ────────────────────────────────
import { db } from "./auth";
try {
  db.prepare(`CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    type TEXT NOT NULL,
    message TEXT NOT NULL
  )`).run();
} catch {}

// REPAIR 2026-07-13 (KNOWN BROKEN #18 follow-up, see server/dbWriteTiming.ts
// for the full evidence trail): `db` (from server/auth.ts, a FROZEN PATH —
// this call configures the already-exported connection from bot.ts instead
// of editing that file) defaulted to SQLite's rollback-journal mode on
// `/data`, the Railway persistent volume. WAL avoids a per-transaction
// journal-file create/fsync/delete cycle; `synchronous` is left at its FULL
// default so durability for auth.ts's users/sessions tables is unchanged.
try {
  const mode = db.pragma("journal_mode = WAL", { simple: true });
  audit("STARTUP", `SQLite journal_mode=${mode} (KNOWN BROKEN #18 follow-up)`);
} catch (err: any) {
  audit("STARTUP", `SQLite WAL pragma failed: ${String(err?.message || err).slice(0, 200)}`);
}

function persistAudit(type: string, message: string) {
  try {
    const insertStart = Date.now();
    db.prepare("INSERT INTO audit_log (time, type, message) VALUES (?, ?, ?)").run(
      new Date().toISOString(), type, message.slice(0, 500)
    );
    const insertMs = Date.now() - insertStart;

    const deleteStart = Date.now();
    // Keep last 2000 entries
    db.prepare("DELETE FROM audit_log WHERE id NOT IN (SELECT id FROM audit_log ORDER BY id DESC LIMIT 2000)").run();
    const deleteMs = Date.now() - deleteStart;

    // Guard on `type` to bound recursion to one level (the nested
    // audit("DB-SLOW-WRITE", ...) call below re-enters persistAudit with
    // type="DB-SLOW-WRITE", which this check refuses to re-trigger on).
    if (type !== "DB-SLOW-WRITE" && (isSlowDbWrite(insertMs) || isSlowDbWrite(deleteMs))) {
      audit("DB-SLOW-WRITE", formatSlowWriteMessage(insertMs, deleteMs));
    }
  } catch {}
}

function getPersistedAuditLog(limit = 100): any[] {
  try {
    return db.prepare("SELECT time, type, message FROM audit_log ORDER BY id DESC LIMIT ?").all(limit) as any[];
  } catch { return []; }
}

function getAuditLogCount(): number {
  try {
    const row = db.prepare("SELECT COUNT(*) as cnt FROM audit_log").get() as any;
    return row?.cnt ?? 0;
  } catch { return 0; }
}

export function registerBotRoutes(app: Express) {
  // R13: every SIGTERM restart traced back (2026-07-06) to a Railway
  // redeploy, not a crash loop — 20/20 sampled restarts matched a merge
  // to main within ~2.5min. That correlation was done by hand against
  // git log; persist the version at boot so future audit queries
  // (?type=STARTUP) show it directly without re-deriving it.
  audit("STARTUP", `Server boot — code_version ${pkgVersion}, pid ${process.pid}`);

  // SSE for live bot audit log updates
  app.get("/api/bot/stream", requireOwner, (req, res) => {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    });
    sseClients.add(res);
    req.on("close", () => sseClients.delete(res));
  });

  // Account info from Alpaca
  app.get("/api/bot/account", requireOwner, async (_req, res) => {
    try {
      const acct = await alpaca("/v2/account");
      const equity = parseFloat(acct.equity);
      const lastEquity = parseFloat(acct.last_equity);
      const dailyPnL = equity - lastEquity;
      // Update bot state with current daily P&L
      state.dailyPnL = dailyPnL;

      // ── Max Drawdown Kill Switch (validated reads only — 2026-07-07
      // incident: a garbage equity read killed the loop at market open
      // with the account at peak; see server/drawdownGuard.ts) ──
      const dd = evaluateDrawdown(acct.equity, state.equityPeak, state.maxDrawdownPct);
      if (!dd.valid) {
        audit("EQUITY-READ-INVALID", `account route: equity=${JSON.stringify(acct.equity)} — drawdown eval skipped (no kill, no peak update)`);
      } else {
        if (dd.newPeak !== state.equityPeak) { state.equityPeak = dd.newPeak; saveEquityPeak(); }
        if (dd.kill && !state.killSwitch) {
          state.killSwitch = true;
          saveKillSwitch("drawdown-kill (account route)");
          state.active = false;
          const msg = `MAX DRAWDOWN KILL SWITCH: Equity $${dd.equity!.toFixed(0)} is ${dd.drawdownPct!.toFixed(1)}% below peak $${state.equityPeak.toFixed(0)}. All trading stopped.`;
          audit("DRAWDOWN-KILL", msg);
          notify("critical", msg);
          sendEmailAlert("MAX DRAWDOWN — Trading Stopped", msg);
          try { await alpaca("/v2/orders", { method: "DELETE" }); } catch {}
        }
      }

      res.json({
        accountNumber: acct.account_number,
        cash: parseFloat(acct.cash),
        portfolioValue: parseFloat(acct.portfolio_value),
        buyingPower: parseFloat(acct.buying_power),
        equity,
        lastEquity,
        dailyPnL,
        dailyPnLPct: ((equity - lastEquity) / lastEquity) * 100,
        status: acct.status,
        tradingBlocked: acct.trading_blocked,
        mode: "paper",
      });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // Current positions from Alpaca
  app.get("/api/bot/positions", requireOwner, async (_req, res) => {
    try {
      const positions = await alpaca("/v2/positions");
      if (!Array.isArray(positions)) {
        console.error("[bot] /v2/positions returned non-array:", JSON.stringify(positions).slice(0, 200));
        return res.json([]);
      }

      // Load stop state and options state in parallel — these are enrichment data,
      // so failures must not prevent returning positions
      let stopState: any = {};
      let optionsState: any = {};
      await Promise.all([
        (async () => {
          try {
            const { stdout: stopOut } = await execPythonSerialized(`python3 -c "
import json, os
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
stop_path = os.path.join(DATA_DIR, 'voltrade_stop_state.json')
try:
    with open(stop_path) as f: data = json.load(f)
except Exception: data = {}
print(json.dumps(data))
"`, { timeout: 5000 });
            stopState = JSON.parse(stopOut.trim() || "{}");
          } catch (e: any) {
            console.error("[bot] Python stop state failed, trying direct file read:", e.message);
            try {
              const DATA_DIR = fs.existsSync('/data/voltrade') ? '/data/voltrade' : '/tmp';
              const stopPath = `${DATA_DIR}/voltrade_stop_state.json`;
              if (fs.existsSync(stopPath)) {
                stopState = JSON.parse(fs.readFileSync(stopPath, 'utf8'));
              }
            } catch (fsErr: any) {
              console.error("[bot] Direct file read also failed:", fsErr.message);
            }
          }
        })(),
        (async () => {
          try {
            const { stdout: optOut } = await execPythonSerialized(`python3 -c "
import json, os
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
path = os.path.join(DATA_DIR, 'voltrade_options_state.json')
try:
    with open(path) as f: data = json.load(f)
except Exception: data = {}
print(json.dumps(data))
"`, { timeout: 5000 });
            optionsState = JSON.parse(optOut.trim() || "{}");
          } catch (e: any) {
            console.error("[bot] Python options state failed, trying direct file read:", e.message);
            try {
              const DATA_DIR = fs.existsSync('/data/voltrade') ? '/data/voltrade' : '/tmp';
              const optionsPath = `${DATA_DIR}/voltrade_options_state.json`;
              if (fs.existsSync(optionsPath)) {
                optionsState = JSON.parse(fs.readFileSync(optionsPath, 'utf8'));
              }
            } catch (fsErr: any) {
              console.error("[bot] Direct file read also failed:", fsErr.message);
            }
          }
        })(),
      ]);

      const mapped = (positions as any[]).map((p: any) => {
        const ss = stopState[p.symbol] || {};
        const entry = parseFloat(p.avg_entry_price);
        const current = parseFloat(p.current_price);
        const pnlPct = parseFloat(p.unrealized_plpc) * 100;
        const atrPct = ss.current_stop_pct || 2.5;
        const phase = ss.phase || 1;
        // Calculate live stop and TP prices from evolving stop state
        const stopPrice = phase >= 2
          ? current * (1 - atrPct / 100)  // Trailing from current
          : entry * (1 - atrPct / 100);    // Phase 1: from entry
        const tpPrice = phase >= 3 ? null : entry * (1 + atrPct * 1.5 / 100);

        // v1.0.33: Parse OCC symbol for options positions
        const isOption = (p.asset_class || "").toLowerCase().includes("option") || (p.symbol || "").length > 12;
        let optionMeta: any = null;
        if (isOption) {
          const sym = p.symbol || "";
          // OCC format: AAPL260418C00250000 = ticker + YYMMDD + C/P + 8-digit strike*1000
          const match = sym.match(/^([A-Z]+)(\d{6})([CP])(\d{8})$/);
          if (match) {
            const [, oTicker, dateStr, cp, strikeRaw] = match;
            const expiry = `20${dateStr.slice(0,2)}-${dateStr.slice(2,4)}-${dateStr.slice(4,6)}`;
            const strike = parseInt(strikeRaw, 10) / 1000;
            const optType = cp === "C" ? "CALL" : "PUT";
            // Look up strategy/setup from options state
            const os = optionsState[sym] || optionsState[oTicker] || {};
            optionMeta = {
              underlyingTicker: oTicker,
              expiry,
              strike,
              optionType: optType,
              strategy: os.strategy || os.options_strategy || null,
              setup: os.setup || null,
              daysToExpiry: Math.ceil((new Date(expiry).getTime() - Date.now()) / 86400000),
            };
          }
        }

        const rawSide = parseFloat(p.qty) > 0 ? "long" : "short";
        return {
          ticker: p.symbol,
          qty: parseFloat(p.qty),
          side: getDisplaySide(p.symbol, rawSide),
          entryPrice: entry,
          currentPrice: current,
          marketValue: parseFloat(p.market_value),
          pnl: parseFloat(p.unrealized_pl),
          pnlPct: pnlPct,
          stopPrice: Math.round(stopPrice * 100) / 100,
          takeProfitPrice: tpPrice ? Math.round(tpPrice * 100) / 100 : null,
          phase: phase,
          rMultiple: ss.r_multiple || 0,
          highestPnl: ss.highest_pnl || 0,
          daysHeld: ss.days_held || 0,
          asset_class: p.asset_class || "us_equity",
          isOption,
          optionMeta,
        };
      });
      res.json(mapped);
    } catch (e: any) {
      console.error("[bot] /api/bot/positions error:", e.message);
      res.status(500).json({ error: e.message });
    }
  });

  // Candlestick bars for trade charts
  // Trade history from Alpaca
  app.get("/api/bot/history", requireOwner, async (_req, res) => {
    try {
      const orders = await alpaca("/v2/orders?status=closed&limit=50");
      if (!Array.isArray(orders)) return res.json([]);
      const mapped = (orders as any[]).map((o: any) => ({
        ticker: o.symbol,
        side: o.side,
        qty: parseFloat(o.filled_qty || o.qty),
        price: parseFloat(o.filled_avg_price || 0),
        status: o.status,
        submittedAt: o.submitted_at,
        filledAt: o.filled_at,
        type: o.type,
      }));
      res.json(mapped);
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Performance endpoint ──────────────────────────────────────────────────
  // Removed: duplicate performance endpoint (empty handler shadowed real one)

  // ── Notifications endpoints ───────────────────────────────────────────────
  app.get("/api/bot/notifications", requireOwner, (_req, res) => {
    res.json(notifications);
  });

  app.post("/api/bot/notifications/read", requireOwner, (_req, res) => {
    notifications.forEach(n => (n.read = true));
    res.json({ ok: true });
  });

  // ── Health Check (for Railway and monitoring) ──────────────────────────────
  app.get("/api/health", async (_req, res) => {
    const checks: any = { timestamp: new Date().toISOString(), status: "ok", checks: {} };

    // Check 1: Node.js server is running (obviously)
    // uptime_s: a restart loop is invisible without it — prod cycled every
    // ~61s for 2+ hours on 2026-07-05 while every health probe read "ok"
    // (each probe hit a young-but-alive process). Low uptime on repeated
    // reads IS the restart-loop signal; the DAILY check must alarm on it.
    // heap/rss (OOM postmortem): the crash driver was heap exhaustion —
    // memory must be readable from outside so growth is seen BEFORE the cap.
    const mu = process.memoryUsage();
    checks.checks.server = {
      status: "ok",
      uptime_s: Math.round(process.uptime()),
      heap_used_mb: Math.round(mu.heapUsed / 1048576),
      rss_mb: Math.round(mu.rss / 1048576),
    };
    
    // Check 2: SQLite database
    try {
      db.prepare("SELECT 1").get();
      checks.checks.database = { status: "ok" };
    } catch (err: any) {
      checks.checks.database = { status: "error", detail: err?.message };
      checks.status = "degraded";
    }
    
    // Check 3: Alpaca API
    try {
      const acct = await alpaca("/v2/account");
      checks.checks.alpaca = { status: "ok", account_status: acct.status };
    } catch (err: any) {
      checks.checks.alpaca = { status: "error", detail: err?.message };
      checks.status = "degraded";
    }
    
    // Check 4: Python engine
    try {
      const { stdout } = await execPythonSerialized('python3 -c "print(\'ok\')"', { timeout: 5000 });
      checks.checks.python = { status: stdout.trim() === "ok" ? "ok" : "error" };
    } catch (err: any) {
      checks.checks.python = { status: "error", detail: err?.message };
      checks.status = "degraded";
    }
    
    // Check 5: Bot state + LIVENESS ALARM (Amendment 2 runtime half): a
    // loop paused/halted >2 market hours (or 24h wall-clock) DEGRADES
    // overall health — this exact gap let the bot sit paused unflagged
    // while every routine read "all ok".
    const activeNow = state.active && !state.killSwitch;
    const nextLv = nextLiveness(livenessState, activeNow, Date.now());
    if (nextLv !== livenessState) { livenessState = nextLv; saveLiveness(nextLv); }
    const lv = loopDark(livenessState, activeNow, Date.now());
    checks.checks.bot = {
      status: state.killSwitch ? "killed" : state.active ? "active" : "stopped",
      equityPeak: state.equityPeak,
      drawdownPct: state.equityPeak > 0 ? (((state.equityPeak - parseFloat(state.lastEquity || String(state.equityPeak))) / state.equityPeak) * 100).toFixed(1) : "N/A",
      liveness: lv.dark
        ? { dark: true, marketHours: +lv.marketHours.toFixed(1), wallHours: +lv.wallHours.toFixed(1), detail: lv.detail }
        : { dark: false },
    };
    if (lv.dark) checks.status = "degraded";

    // Check 5b: TIER-2 scan health (REPAIR 2026-07-06) — Check 5 above
    // catches the loop being paused/halted; it does NOT catch a loop that
    // reads "active" while its market-data scan has failed every cycle
    // for hours (found live: 10+ consecutive TIER2-BACKOFF failures over
    // ~1.5h, all "Could not fetch market data from Alpaca", while this
    // endpoint read clean the whole time). A scan that never succeeds
    // finds no new trade candidates — degraded in every sense that
    // matters even though the process itself is technically "active".
    // /api/health is UNAUTHENTICATED — status + a bare count only, same
    // sensitivity class as the existing bot/licensing checks; the free-text
    // failure reason (subprocess stderr tails, raw Alpaca error bodies) is
    // reserved for the token-gated /api/diag/scanner probe below.
    checks.checks.scanner = {
      status: scannerDegraded(tier2ConsecutiveFailures) ? "degraded" : "ok",
      consecutiveFailures: tier2ConsecutiveFailures,
    };
    if (scannerDegraded(tier2ConsecutiveFailures)) checks.status = "degraded";

    // Check 6: data-provider licensing (monetization tripwire, CLAUDE.md
    // KNOWN STATE 2026-07-03) — non-commercial providers must leave the
    // aircraft chain before billing activates; degrade health so the next
    // DAILY routine's health check sees a dashboard-only monetization flip.
    const licensing = aircraftProviderCompliance();
    if (licensing.status === "ok") {
      checks.checks.licensing = { status: "ok" };
    } else {
      checks.checks.licensing = { status: "warning", detail: licensing.detail };
      checks.status = "degraded";
    }
    
    // OOM fix: expose memory usage in health check for monitoring
    const mem = process.memoryUsage();
    checks.checks.memory = {
      heapUsedMB: Math.round(mem.heapUsed / 1048576),
      heapTotalMB: Math.round(mem.heapTotal / 1048576),
      rssMB: Math.round(mem.rss / 1048576),
      externalMB: Math.round(mem.external / 1048576),
    };

    // MAP V2 ROADMAP R6(c) PIPELINE-HEALTH dashboard time-series (2026-07-31):
    // no new poller — this handler already computes every check on a steady
    // external polling cadence (Railway + monitors), so recordHealthSnapshot
    // throttles itself to one disk write per window. Never let a history-
    // recording failure affect the health response itself.
    try { recordHealthSnapshot(checks); } catch {}

    const httpCode = checks.status === "ok" ? 200 : 503;
    res.status(httpCode).json(checks);
  });

  // ── Performance Dashboard Data ────────────────────────────────────────────
  app.get("/api/bot/performance", requireOwner, async (_req, res) => {
    try {
      let perf: any = { totalTrades: 0, totalFills: 0, winRate: 0, avgGain: 0, avgLoss: 0, totalPnlPct: 0, profitFactor: 0, byStrategy: {}, recentTrades: [], realisticPnlPct: 0, avgSlippagePct: 0, totalSlippageCost: 0, slippageGapPct: 0, bestTrade: null, worstTrade: null };

      try {
        // Get trade history from fills
        const { stdout: fillsOut } = await execPythonSerialized(`python3 -c "
import json, os
try:
    from storage_config import TRADE_FEEDBACK_PATH
except ImportError:
    TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'
from ml_model_v2 import fills_slippage_stats

feedback_raw = []
if os.path.exists(TRADE_FEEDBACK_PATH):
    with open(TRADE_FEEDBACK_PATH) as f: feedback_raw = json.load(f)

# DASHBOARD-FIX 2026-05-03 (alpha audit batch 3): filter out backtest-seeded
# records before computing PERFORMANCE numbers. Seeded records exist purely
# to give the Kelly gate a real per-bucket prior on day one — they are NOT
# trades the bot actually placed. Including them in the dashboard would
# tell users 'you have 1326 trades, +700% PnL' as soon as the daemon
# auto-seeds, which is dramatically misleading. Seeded records have
# _seed: True set by seed_feedback_from_backtest.py.
seeded_count = sum(1 for t in feedback_raw if t.get('_seed'))
feedback = [t for t in feedback_raw if not t.get('_seed')]

# Filter out corrupt records: require non-empty ticker
feedback = [t for t in feedback if t.get('ticker', '').strip() and not (t.get('pnl_pct', 0) == 0 and t.get('outcome') is None)]
# Win rate
wins = [t for t in feedback if t.get('pnl_pct', 0) > 0]
losses = [t for t in feedback if t.get('pnl_pct', 0) <= 0]
win_rate = len(wins) / len(feedback) * 100 if feedback else 0
avg_win = sum(t.get('pnl_pct', 0) for t in wins) / len(wins) if wins else 0
avg_loss = sum(t.get('pnl_pct', 0) for t in losses) / len(losses) if losses else 0
total_pnl = sum(t.get('pnl_pct', 0) for t in feedback)

# Profit factor
gross_profit = sum(t.get('pnl_pct', 0) for t in wins)
gross_loss = abs(sum(t.get('pnl_pct', 0) for t in losses))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

# By strategy
by_strat = {}
for t in feedback:
    s = t.get('strategy', 'unknown')
    if s not in by_strat: by_strat[s] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
    if t.get('pnl_pct', 0) > 0: by_strat[s]['wins'] += 1
    else: by_strat[s]['losses'] += 1
    by_strat[s]['total_pnl'] += t.get('pnl_pct', 0)

# Realistic P&L (slippage derived from entry-fill records themselves — see
# fills_slippage_stats in ml_model_v2.py: the old separate fills-tracker
# file has had no writer since a legacy module was retired; entry-fill
# slippage_pct/expected_price/qty have lived directly on trade_feedback
# records since v1.0.34)
_slip = fills_slippage_stats(feedback)
total_slippage_cost = _slip['total_slippage_cost']
avg_slippage_pct = _slip['avg_slippage_pct']
fills_count = _slip['count']

# Estimated realistic P&L = paper P&L minus slippage drag
realistic_total_pnl = total_pnl
if fills_count > 0:
    # Each trade has entry + exit slippage
    realistic_total_pnl = total_pnl - (avg_slippage_pct * 2 * len(feedback) / 100 * 100)

# Best / worst trades
best_trade = max(feedback, key=lambda t: t.get('pnl_pct', 0)) if feedback else None
worst_trade = min(feedback, key=lambda t: t.get('pnl_pct', 0)) if feedback else None

print(json.dumps({
    'totalTrades': len(feedback),
    'totalFills': fills_count,
    'seededTrades': seeded_count,
    'winRate': round(win_rate, 1),
    'avgGain': round(avg_win, 2),
    'avgLoss': round(avg_loss, 2),
    'totalPnlPct': round(total_pnl, 2),
    'profitFactor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
    'byStrategy': by_strat,
    'recentTrades': feedback[-20:][::-1],
    'realisticPnlPct': round(realistic_total_pnl, 2),
    'avgSlippagePct': round(avg_slippage_pct, 4),
    'totalSlippageCost': round(total_slippage_cost, 2),
    'slippageGapPct': round(total_pnl - realistic_total_pnl, 2),
    'bestTrade': {'ticker': best_trade.get('ticker', ''), 'pnlPct': round(best_trade.get('pnl_pct', 0), 2)} if best_trade else None,
    'worstTrade': {'ticker': worst_trade.get('ticker', ''), 'pnlPct': round(worst_trade.get('pnl_pct', 0), 2)} if worst_trade else None,
}))
"`, { timeout: 10000 });

        perf = JSON.parse(fillsOut.trim());
      } catch (pyErr: any) {
        console.error("[perf] Python execution failed, falling back to Alpaca:", pyErr?.message || pyErr);
      }

      // Fallback: if Python files yielded no trades, compute from Alpaca orders
      if (perf.totalTrades === 0) {
        try {
          // Compute trading-day start: 4 AM ET today (or yesterday if before 4 AM)
          const nowET = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
          if (nowET.getHours() < 4) nowET.setDate(nowET.getDate() - 1);
          nowET.setHours(4, 0, 0, 0);
          const etYear = nowET.getFullYear();
          const etMonth = String(nowET.getMonth() + 1).padStart(2, "0");
          const etDay = String(nowET.getDate()).padStart(2, "0");
          const jan = new Date(etYear, 0, 1);
          const jul = new Date(etYear, 6, 1);
          const stdOffset = Math.max(jan.getTimezoneOffset(), jul.getTimezoneOffset());
          const isDST = nowET.getTimezoneOffset() < stdOffset;
          const utcHour = isDST ? 8 : 9;
          const tradingDayStart = `${etYear}-${etMonth}-${etDay}T${String(utcHour).padStart(2, "0")}:00:00Z`;

          const [alpacaOrders, alpacaPositions, alpacaAccount] = await Promise.all([
            alpaca(`/v2/orders?status=closed&after=${encodeURIComponent(tradingDayStart)}&limit=200&direction=desc`),
            alpaca("/v2/positions"),
            alpaca("/v2/account"),
          ]);

          const filled = (alpacaOrders as any[]).filter((o: any) => o.status === "filled");
          const posMap: Record<string, number> = {};
          for (const p of (alpacaPositions as any[])) {
            posMap[p.symbol] = parseFloat(p.avg_entry_price) || 0;
          }

          // Pair buys/sells by symbol to compute P/L
          // UI FIX 2026-05-05 batch 9: include filled_at timestamp so the
          // recent-trades panel shows real dates instead of dashes. Strategy
          // is harder — Alpaca doesn't store our strategy tag, but we can
          // backfill from in-memory tradeResults if we have it.
          const buysBySymbol: Record<string, Array<{ price: number; ts: string }>> = {};
          const trades: Array<{ ticker: string; side: string; pnl_pct: number; pnl: number; timestamp?: string; strategy?: string }> = [];

          // Build a strategy lookup from in-memory tradeResults: ticker → strategy
          // (most recent trade for that ticker wins). Only includes strategies
          // recorded by the scan loop, not generic "auto" fallbacks.
          const stratByTicker: Record<string, string> = {};
          for (const tr of tradeResults) {
            if (tr.strategy && tr.strategy !== "auto" && tr.ticker) {
              stratByTicker[tr.ticker] = tr.strategy;
            }
          }

          // Process chronologically (oldest first)
          for (const o of [...filled].reverse()) {
            const sym = o.symbol || "";
            const side = (o.side || "").toLowerCase();
            const fillPrice = parseFloat(o.filled_avg_price) || 0;
            const qty = parseFloat(o.filled_qty || o.qty) || 0;
            const filledAt = o.filled_at || o.created_at || "";

            if (side === "buy") {
              if (!buysBySymbol[sym]) buysBySymbol[sym] = [];
              buysBySymbol[sym].push({ price: fillPrice, ts: filledAt });
            } else if (side === "sell") {
              const buyEntry = (buysBySymbol[sym] && buysBySymbol[sym].length > 0)
                ? buysBySymbol[sym].shift()!
                : { price: posMap[sym] || 0, ts: "" };
              const entryPrice = buyEntry.price;
              if (entryPrice > 0) {
                const pnlDollar = (fillPrice - entryPrice) * qty;
                const pnlPct = ((fillPrice - entryPrice) / entryPrice) * 100;
                trades.push({
                  ticker: sym,
                  side: "sell",
                  pnl_pct: pnlPct,
                  pnl: pnlDollar,
                  timestamp: filledAt,                       // exit timestamp
                  strategy: stratByTicker[sym] || "stock",   // best-effort backfill
                });
              }
            }
          }

          // Include open positions as unrealized "trades"
          for (const pos of (alpacaPositions as any[])) {
            const sym = pos.symbol || "";
            const unrealizedPl = parseFloat(pos.unrealized_pl) || 0;
            const unrealizedPlPct = (parseFloat(pos.unrealized_plpc) || 0) * 100;
            const side = (pos.side || "long").toLowerCase();
            // UI FIX batch 9: even open positions get a marker so the date
            // column doesn't show "—". Use today's date since the position
            // is current.
            trades.push({
              ticker: sym,
              side,
              pnl_pct: unrealizedPlPct,
              pnl: unrealizedPl,
              timestamp: new Date().toISOString(),
              strategy: stratByTicker[sym] || "open_position",
            });
          }

          // Account-level daily P/L from Alpaca (most accurate headline number)
          const acct = alpacaAccount as any;
          const equity = parseFloat(acct?.equity) || 0;
          const lastEquity = parseFloat(acct?.last_equity) || 0;
          const accountPnlDollar = lastEquity > 0 ? equity - lastEquity : 0;
          const accountPnlPct = lastEquity > 0 ? ((equity - lastEquity) / lastEquity) * 100 : 0;

          const wins = trades.filter(t => t.pnl_pct > 0);
          const losses = trades.filter(t => t.pnl_pct <= 0);
          const winRate = trades.length > 0 ? (wins.length / trades.length) * 100 : 0;
          const avgGain = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl_pct, 0) / wins.length : 0;
          const avgLoss = losses.length > 0 ? losses.reduce((s, t) => s + t.pnl_pct, 0) / losses.length : 0;
          const grossProfit = wins.reduce((s, t) => s + t.pnl_pct, 0);
          const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnl_pct, 0));
          const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

          const bestTrade = trades.length > 0 ? trades.reduce((a, b) => a.pnl_pct > b.pnl_pct ? a : b) : null;
          const worstTrade = trades.length > 0 ? trades.reduce((a, b) => a.pnl_pct < b.pnl_pct ? a : b) : null;

          perf = {
            totalTrades: trades.length,
            totalFills: filled.length,
            winRate: Math.round(winRate * 10) / 10,
            avgGain: Math.round(avgGain * 100) / 100,
            avgLoss: Math.round(avgLoss * 100) / 100,
            totalPnlPct: Math.round(accountPnlPct * 100) / 100,
            totalPnlDollar: Math.round(accountPnlDollar * 100) / 100,
            profitFactor: profitFactor === Infinity ? "inf" : Math.round(profitFactor * 100) / 100,
            byStrategy: {},
            recentTrades: trades.slice(-20).reverse().map(t => ({
              ticker: t.ticker,
              side: t.side,
              pnl_pct: Math.round(t.pnl_pct * 100) / 100,
              // UI FIX batch 9: include timestamp + strategy so the recent
              // trades panel shows real values instead of dashes.
              timestamp: t.timestamp || null,
              strategy: t.strategy || null,
            })),
            realisticPnlPct: Math.round(accountPnlPct * 100) / 100,
            avgSlippagePct: 0,
            totalSlippageCost: 0,
            slippageGapPct: 0,
            bestTrade: bestTrade ? { ticker: bestTrade.ticker, pnlPct: Math.round(bestTrade.pnl_pct * 100) / 100 } : null,
            worstTrade: worstTrade ? { ticker: worstTrade.ticker, pnlPct: Math.round(worstTrade.pnl_pct * 100) / 100 } : null,
          };
        } catch (alpacaErr: any) {
          console.error("[perf] Alpaca fallback failed:", alpacaErr?.message || alpacaErr);
        }
      }

      // Apply inverse ETF side mapping to recent trades
      if (Array.isArray(perf.recentTrades)) {
        for (const t of perf.recentTrades) {
          const ticker = t.ticker ?? t.symbol ?? "";
          const rawSide = t.side ?? t.direction ?? "long";
          t.side = getDisplaySide(ticker, rawSide);
        }
      }
      res.json({
        ...perf,
        equityCurve,
        equityPeak: state.equityPeak,
        currentDrawdown: state.equityPeak > 0 ? ((equityCurve[0]?.value || 0) - state.equityPeak) / state.equityPeak * 100 : 0,
      });
    } catch (err: any) {
      res.json({ error: err?.message, totalTrades: 0, equityCurve });
    }
  });

  // ── Trade History CSV Export ─────────────────────────────────────────────
  app.get("/api/bot/export-trades", requireOwner, async (_req, res) => {
    try {
      const { stdout } = await execPythonSerialized(`python3 -c "
import json, os, csv, io
try:
    from storage_config import TRADE_FEEDBACK_PATH
except ImportError:
    TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'

feedback = []
if os.path.exists(TRADE_FEEDBACK_PATH):
    with open(TRADE_FEEDBACK_PATH) as f: feedback = json.load(f)

out = io.StringIO()
w = csv.writer(out)
w.writerow(['Date', 'Ticker', 'Side', 'P&L %', 'Score', 'Strategy', 'Holding Days', 'Won'])
for t in feedback:
    w.writerow([
        t.get('timestamp', ''),
        t.get('ticker', ''),
        t.get('side', ''),
        round(t.get('pnl_pct', 0), 2),
        t.get('score', ''),
        t.get('strategy', ''),
        t.get('holding_days', ''),
        'Yes' if t.get('won') else 'No',
    ])
print(out.getvalue())
"`, { timeout: 10000 });
      res.setHeader("Content-Type", "text/csv");
      res.setHeader("Content-Disposition", `attachment; filename=voltrade_trades_${new Date().toISOString().slice(0,10)}.csv`);
      res.send(stdout);
    } catch (err: any) {
      res.status(500).json({ error: err?.message });
    }
  });

  // ── Data Backup Endpoint (manual trigger or cron) ──────────────────────────
  app.post("/api/bot/backup", requireOwner, async (_req, res) => {
    try {
      const { stdout } = await execPythonSerialized(`python3 -c "
import json, os, shutil, time
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = '/tmp'

backup_dir = os.path.join(DATA_DIR, 'backups')
os.makedirs(backup_dir, exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S')
backup_path = os.path.join(backup_dir, f'backup_{timestamp}')
os.makedirs(backup_path, exist_ok=True)

files_backed = []
for fname in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.isfile(fpath) and (fname.endswith('.json') or fname.endswith('.pkl') or fname.endswith('.db')):
        shutil.copy2(fpath, os.path.join(backup_path, fname))
        files_backed.append(fname)

# Keep only last 5 backups
all_backups = sorted([d for d in os.listdir(backup_dir) if d.startswith('backup_')])
while len(all_backups) > 5:
    old = all_backups.pop(0)
    shutil.rmtree(os.path.join(backup_dir, old), ignore_errors=True)

print(json.dumps({'backed_up': len(files_backed), 'files': files_backed, 'path': backup_path, 'total_backups': min(len(all_backups)+1, 5)}))
"`, { timeout: 30000 });
      const result = JSON.parse(stdout.trim());
      audit("BACKUP", `Data backed up: ${result.backed_up} files to ${result.path}`);
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err?.message });
    }
  });

  // ═════════════════════════════════════════════════════════════════
  // OBSERVABILITY — 2026-04-22
  // Read-only snapshot/health endpoints. Zero trading impact.
  // ═════════════════════════════════════════════════════════════════

  app.get("/api/system/snapshot", requireOwner, async (_req, res) => {
    const snapshot: any = {
      snapshot_version: "1.1",
      generated_at: new Date().toISOString(),
      git_head_expected: "5d2add3 or later",
      server: {
        uptime_seconds: Math.round(process.uptime()),
        node_version: process.version,
        memory_mb: Math.round(process.memoryUsage().rss / 1024 / 1024),
        heap_used_mb: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        heap_total_mb: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
      },
      bot_state: {
        active: state.active,
        kill_switch: state.killSwitch,
        daily_pnl: state.dailyPnL,
        daily_loss_limit: state.dailyLossLimit,
        position_size_multiplier: state.positionSizeMultiplier,
        min_score_threshold: state.minScoreThreshold,
        max_drawdown_pct: state.maxDrawdownPct,
        equity_peak: state.equityPeak,
        consecutive_stop_losses: state.consecutiveStopLosses,
        circuit_breaker_until: state.circuitBreakerUntil,
      },
      daemon: { enabled: DAEMON_ENABLED, alive: false },
      audit_log: {
        total_entries: 0,
        recent_errors: [] as any[],
        recent_backoffs: 0,
        last_scan_success: null as string | null,
        last_scan_failure: null as string | null,
      },
      scan_stats: { success: 0, error: 0, sigkill: 0, last_24h: 0 },
      positions: { count: 0, deployment_pct: 0, by_ticker: {} as any },
      kill_switches: {},
      ml: {},
      config_files: {},
      stress_index: {},
      floor_state: {},
      tier_engine: {},
      probability_engine: {},
    };

    if (DAEMON_ENABLED) {
      try {
        const r = await pythonRpc("health", {});
        if (r.status === "ok") snapshot.daemon = { enabled: true, alive: true, ...r.result };
        else snapshot.daemon = { enabled: true, alive: false, reason: r.error_message };
      } catch (e: any) { snapshot.daemon = { enabled: true, alive: false, reason: e.message }; }
    }

    try {
      const persisted = getPersistedAuditLog(500);
      snapshot.audit_log.total_entries = getAuditLogCount();
      let lastSuccess = null, lastFailure = null;
      let backoffCount = 0;
      const cutoff24h = Date.now() - 86400000;
      let last24h = 0;
      for (const e of persisted) {
        const t = new Date(e.time).getTime();
        if (t > cutoff24h) last24h++;
        if (e.type === "TIER2-ERROR" || e.type === "TIER2-BACKOFF") {
          if (e.type === "TIER2-BACKOFF") backoffCount++;
          if (!lastFailure) lastFailure = e.time;
        }
        if (e.type === "TIER2" && typeof e.message === "string" && e.message.includes("complete")) {
          if (!lastSuccess) lastSuccess = e.time;
        }
      }
      snapshot.audit_log.recent_errors = persisted.filter((e: any) => e.type?.includes("ERROR")).slice(0, 10);
      snapshot.audit_log.recent_backoffs = backoffCount;
      snapshot.audit_log.last_scan_success = lastSuccess;
      snapshot.audit_log.last_scan_failure = lastFailure;
      snapshot.scan_stats.success = persisted.filter((e: any) => e.type === "TIER2" && e.message?.includes("complete")).length;
      snapshot.scan_stats.error = persisted.filter((e: any) => e.type === "TIER2-ERROR").length;
      snapshot.scan_stats.sigkill = persisted.filter((e: any) => e.message?.includes("SIGKILL")).length;
      snapshot.scan_stats.last_24h = last24h;
    } catch (e: any) { snapshot.audit_log.error = e.message; }

    try {
      const positions = await alpaca("/v2/positions");
      if (Array.isArray(positions)) {
        snapshot.positions.count = positions.length;
        let totalValue = 0;
        const byTicker: Record<string, number> = {};
        for (const p of positions) {
          const mv = Math.abs(parseFloat(p.market_value || "0"));
          totalValue += mv;
          byTicker[p.symbol || ""] = (byTicker[p.symbol || ""] || 0) + mv;
        }
        const acct = await alpaca("/v2/account");
        const equity = parseFloat(acct.equity || "0");
        snapshot.positions.deployment_pct = equity > 0 ? Math.round((totalValue / equity) * 10000) / 100 : 0;
        snapshot.positions.by_ticker = byTicker;
        snapshot.positions.equity = equity;
        snapshot.positions.cash = parseFloat(acct.cash || "0");
        snapshot.positions.buying_power = parseFloat(acct.buying_power || "0");
      }
    } catch (e: any) { snapshot.positions.error = e.message; }

    try {
      const mlCall = await pythonCall(
        "snapshot_python_state", {},
        `python3 -c "
import json, os
result = {}
try:
    from ml_model_v2 import FEEDBACK_PATH, MODEL_PATH, MIN_FEEDBACK_VERSION
    result['feedback_path'] = FEEDBACK_PATH
    result['model_path'] = MODEL_PATH
    result['min_feedback_version'] = MIN_FEEDBACK_VERSION
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH) as f: fb = json.load(f)
        result['feedback_count'] = len(fb) if isinstance(fb, list) else 0
        # ALPHA AUDIT 2026-05-03 batch 3: distinguish seeded backtest records
        # from live records. Seeded records lack code_version + entry_features
        # so they don't pass ML training filters anyway, but counting them
        # together as 'feedback_count' obscures how much LIVE data exists.
        result['feedback_seeded_count'] = sum(1 for r in fb if r.get('_seed'))
        result['feedback_live_count'] = result['feedback_count'] - result['feedback_seeded_count']
        clean = [r for r in fb if r.get('pnl_pct') is not None and r.get('code_version')]
        result['feedback_clean_count'] = len(clean)
        with_fp = [r for r in fb if r.get('config_fingerprint')]
        result['feedback_with_fingerprint'] = len(with_fp)
    else:
        result['feedback_count'] = 0
        result['feedback_seeded_count'] = 0
        result['feedback_live_count'] = 0
    if os.path.exists(MODEL_PATH):
        import time
        result['model_age_hours'] = round((time.time() - os.path.getmtime(MODEL_PATH)) / 3600, 1)
    else:
        result['model_age_hours'] = None
except Exception as e:
    result['error'] = str(e)[:200]
try:
    from risk_kill_switch import get_kill_switch_status
    result['kill_switches'] = get_kill_switch_status()
except Exception as e:
    result['kill_switches_error'] = str(e)[:200]
try:
    from system_config import BASE_CONFIG, get_adaptive_params
    from macro_data import get_macro_snapshot
    m = get_macro_snapshot()
    result['base_config'] = BASE_CONFIG
    result['current_adaptive_params'] = get_adaptive_params(
        vxx_ratio=float(m.get('vxx_ratio', 1.0)),
        spy_vs_ma50=float(m.get('spy_vs_ma50', 1.0)),
        account_equity=100000,
    )
    result['macro'] = {
        'regime': m.get('regime'),
        'vxx_ratio': m.get('vxx_ratio'),
        'spy_vs_ma50': m.get('spy_vs_ma50'),
        'vix': m.get('vix'),
        'data_quality': m.get('data_quality'),
    }
except Exception as e:
    result['config_error'] = str(e)[:200]
try:
    from stress_index import compute_stress_index
    result['stress_index'] = compute_stress_index()
except Exception as e:
    result['stress_index_error'] = str(e)[:200]
try:
    from probability_engine import compute_all_dimensions
    result['probability_engine'] = compute_all_dimensions()
except Exception as e:
    result['probability_engine_error'] = str(e)[:200]
try:
    from csp_universe import get_universe_snapshot
    result['csp_universe'] = get_universe_snapshot()
except Exception as e:
    result['csp_universe_error'] = str(e)[:200]
try:
    from storage_config import DATA_DIR as _DD
    floor_path = os.path.join(_DD, 'voltrade_floor_state.json')
    if os.path.exists(floor_path):
        with open(floor_path) as f: result['floor_state'] = json.load(f)
except Exception as e:
    result['floor_state_error'] = str(e)[:200]
try:
    from tiered_strategy import TIERS_ENABLED
    result['tiers_enabled'] = TIERS_ENABLED
except Exception as e:
    result['tiers_enabled_error'] = str(e)[:200]
print(json.dumps(result, default=str))
"`,
        { timeout: 20000 }
      );
      if (mlCall.success) {
        snapshot.ml = { status: "ok", ...mlCall.result };
        if (mlCall.result.kill_switches) snapshot.kill_switches = mlCall.result.kill_switches;
        if (mlCall.result.base_config) snapshot.config_files.base_config = mlCall.result.base_config;
        if (mlCall.result.current_adaptive_params) snapshot.config_files.current_adaptive = mlCall.result.current_adaptive_params;
        if (mlCall.result.macro) snapshot.config_files.macro = mlCall.result.macro;
        if (mlCall.result.stress_index) snapshot.stress_index = mlCall.result.stress_index;
        if (mlCall.result.probability_engine) snapshot.probability_engine = mlCall.result.probability_engine;
        if (mlCall.result.floor_state) snapshot.floor_state = mlCall.result.floor_state;
        if (mlCall.result.tiers_enabled) snapshot.tier_engine.tiers_enabled = mlCall.result.tiers_enabled;
      } else {
        snapshot.ml = { status: "failed", error: mlCall.result?.error };
      }
    } catch (e: any) { snapshot.ml = { status: "error", error: e.message }; }

    // TIMING-FILE-DIRECT 2026-04-23: read timing JSON from Node side
    // directly so it works even when the Python subprocess fails.
    try {
      const timingPaths = [
        "/data/voltrade/voltrade_scan_timings.json",
        "/tmp/voltrade_scan_timings.json"
      ];
      for (const tpath of timingPaths) {
        if (fs.existsSync(tpath)) {
          const raw = fs.readFileSync(tpath, "utf8");
          snapshot.last_scan_timings = JSON.parse(raw);
          break;
        }
      }
    } catch (e: any) {
      snapshot.last_scan_timings_error = e.message;
    }

    // DAEMON-TRACE 2026-04-23: read daemon dispatch trace file
    try {
      if (fs.existsSync("/tmp/voltrade_daemon_trace.json")) {
        const raw = fs.readFileSync("/tmp/voltrade_daemon_trace.json", "utf8");
        snapshot.daemon_trace = JSON.parse(raw);
      }
    } catch (e: any) {
      snapshot.daemon_trace_error = e.message;
    }

    const filename = `voltrade_snapshot_${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.json`;
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
    res.json(snapshot);
  });

  app.get("/api/system/health-check", requireOwner, async (_req, res) => {
    const issues: any[] = [];
    const warnings: any[] = [];
    const now = Date.now();
    try {
      const persisted = getPersistedAuditLog(200);
      const recentFailures = persisted.filter((e: any) =>
        e.type === "TIER2-ERROR" || e.type === "TIER2-BACKOFF"
      );
      const recentSuccess = persisted.find((e: any) =>
        e.type === "TIER2" && typeof e.message === "string" && e.message.includes("complete")
      );
      if (recentFailures.length >= 5) issues.push({ severity: "critical", category: "scan_failures", detail: `${recentFailures.length} scan errors/backoffs in recent history` });
      if (!recentSuccess) issues.push({ severity: "critical", category: "no_successful_scans", detail: "No TIER2 scan complete in last 200 audit entries" });
      else {
        const lastSuccessAge = (now - new Date(recentSuccess.time).getTime()) / 60000;
        if (lastSuccessAge > 60) warnings.push({ severity: "warning", category: "stale_scans", detail: `Last successful scan was ${Math.round(lastSuccessAge)} minutes ago` });
      }
      const rssMb = process.memoryUsage().rss / 1024 / 1024;
      if (rssMb > 450) issues.push({ severity: "critical", category: "memory_pressure", detail: `Node RSS at ${Math.round(rssMb)}MB` });
      else if (rssMb > 380) warnings.push({ severity: "warning", category: "memory_pressure", detail: `Node RSS at ${Math.round(rssMb)}MB` });
      if (DAEMON_ENABLED) {
        try {
          const d = await pythonRpc("health", {});
          if (d.status !== "ok") issues.push({ severity: "critical", category: "daemon_down", detail: `Daemon: ${d.error_message}` });
        } catch {
          issues.push({ severity: "critical", category: "daemon_down", detail: "Daemon RPC failed" });
        }
      }
    } catch (e: any) {
      issues.push({ severity: "error", category: "health_check_failed", detail: e.message });
    }
    res.json({
      timestamp: new Date().toISOString(),
      overall: issues.length > 0 ? "critical" : warnings.length > 0 ? "warning" : "healthy",
      issues_count: issues.length,
      warnings_count: warnings.length,
      issues, warnings,
    });
  });

  app.post("/api/system/verify-change", requireOwner, async (req, res) => {
    const expected = req.body?.changes || [];
    if (!Array.isArray(expected)) return res.status(400).json({ error: "Body must be {changes: string[]}" });
    const checks: any = { verified: [], missing: [], checked_at: new Date().toISOString() };
    try {
      const files = [
        "./bot_engine.py", "./options_scanner.py", "./ml_model_v2.py",
        "./server/bot.ts", "./tiered_strategy.py", "./risk_kill_switch.py",
        "./position_sizing.py", "./macro_data.py", "./system_config.py",
        "./alt_data.py", "./options_execution.py", "./stress_index.py",
        "./probability_engine.py",
      ];
      let haystack = "";
      for (const f of files) {
        try { haystack += fs.readFileSync(f, "utf8") + "\n"; } catch {}
      }
      for (const exp of expected) {
        if (typeof exp === "string" && haystack.includes(exp)) checks.verified.push(exp);
        else checks.missing.push(exp);
      }
    } catch (e: any) { checks.error = e.message; }
    res.json(checks);
  });

  // Bot status
  app.get("/api/bot/status", requireOwner, (_req, res) => {
    res.json({
      active: state.active,
      killSwitch: state.killSwitch,
      dailyLossLimit: state.dailyLossLimit,
      auditLogCount: getAuditLogCount(),
      mode: "paper",
      equityPeak: state.equityPeak,
      maxDrawdownPct: state.maxDrawdownPct,
      unreadNotifications: notifications.filter(n => !n.read).length,
      // Circuit breaker status
      circuitBreakerActive: state.circuitBreakerUntil > Date.now(),
      circuitBreakerUntil: state.circuitBreakerUntil > 0 ? new Date(state.circuitBreakerUntil).toISOString() : null,
      consecutiveStopLosses: state.consecutiveStopLosses,
      // Pro-level security controls from system_config.py
      maxPositionPct: 8,      // MAX_POSITION_PCT 0.08 — hard cap per position
      maxExposurePct: 95,     // MAX_TOTAL_EXPOSURE 0.95 — max portfolio invested (regime engine controls actual)
      dailyLossLimitPct: 5,   // DAILY_LOSS_LIMIT_PCT 5.0 — halts trading
    });
  });

  // Start bot
  app.post("/api/bot/start", requireOwner, (_req, res) => {
    if (state.killSwitch) return res.status(400).json({ error: "Kill switch is ON. Disable it first." });
    state.active = true;
    audit("START", "Bot activated");
    notify("system", "Bot activated — scanning for opportunities");
    // Start real-time streaming feed
    setTimeout(() => startStreaming(), 2000);
    res.json({ ok: true, active: true });
  });

  // Stop bot
  app.post("/api/bot/stop", requireOwner, (_req, res) => {
    state.active = false;
    stopStreaming();
    audit("STOP", "Bot deactivated");
    notify("system", "Bot paused");
    res.json({ ok: true, active: false });
  });

  // Kill switch
  app.post("/api/bot/kill", requireOwner, async (_req, res) => {
    state.killSwitch = !state.killSwitch;
    saveKillSwitch(state.killSwitch ? "owner toggle ON" : "owner toggle OFF");
    if (state.killSwitch) {
      sendEmailAlert("Kill Switch Activated", "Kill switch manually activated. All trading stopped. Open orders cancelled.");
      state.active = false;
      audit("KILL SWITCH ON", "All trading halted. Cancelling open orders.");
      notify("alert", "KILL SWITCH ACTIVATED — all trading halted, open orders cancelled");
      try { await alpaca("/v2/orders", { method: "DELETE" }); } catch (err: any) { console.error("[bot]", err?.message || err); }
    } else {
      audit("KILL SWITCH OFF", "Trading can resume.");
      notify("system", "Kill switch deactivated — trading can resume");
    }
    res.json({ ok: true, killSwitch: state.killSwitch });
  });

  // Reset circuit breaker
  app.post("/api/bot/circuit-breaker/reset", requireOwner, async (_req, res) => {
    state.circuitBreakerUntil = 0;
    state.consecutiveStopLosses = 0;
    audit("CIRCUIT-BREAKER-RESET", "Manual reset by owner");
    notify("system", "Circuit breaker manually reset");
    res.json({ ok: true, circuitBreakerUntil: 0, consecutiveStopLosses: 0 });
  });

  // Audit log
  // OOM fix: always read from SQLite instead of in-memory array
  app.get("/api/bot/audit", requireOwner, (_req, res) => {
    const persisted = getPersistedAuditLog(100);
    res.json(persisted.map((e: any) => ({ time: e.time, action: e.type, type: e.type, detail: e.message, message: e.message })));
  });

  // ── Token-gated READ-ONLY diagnostics (wishlist option (d), human-
  // approved 2026-07-04). HARD WHITELIST: audit tail, ml status, daemon
  // health, positions SUMMARY (counts/exposure — never symbols). Closed
  // by default (no/short DIAG_TOKEN ⇒ 404); auth.ts untouched; every
  // response passes sanitizeDiag. Sessions send x-diag-token. ──
  app.get("/api/diag/:probe", async (req, res) => {
    if (!diagEnabled(process.env)) return res.status(404).json({ error: "not enabled" });
    const provided = String(req.headers["x-diag-token"] || req.query.token || "");
    if (!checkDiagToken(process.env, provided)) return res.status(401).json({ error: "unauthorized" });
    try {
      switch (req.params.probe) {
        case "audit": {
          // REPAIR 2026-07-06 (liveness): optional ?type= and ?limit= so
          // restart/incident PATTERNS are readable from outside — the fixed
          // 50-entry tail spans ~5 minutes of a chatty log and hid today's
          // recurring SIGTERM cycle. Same whitelist surface, same sanitize;
          // limit capped so the probe itself can't become a heap problem.
          const limit = Math.min(Math.max(parseInt(String(req.query.limit || "50"), 10) || 50, 1), 500);
          const typeFilter = String(req.query.type || "").trim();
          let entries: any[];
          if (typeFilter) {
            try {
              entries = db.prepare(
                "SELECT time, type, message FROM audit_log WHERE type = ? ORDER BY id DESC LIMIT ?"
              ).all(typeFilter, limit) as any[];
            } catch { entries = []; }
          } else {
            entries = getPersistedAuditLog(limit);
          }
          return res.json(sanitizeDiag({
            probe: "audit",
            entries: entries.map((e: any) => ({ time: e.time, type: e.type, message: e.message })),
          }));
        }
        case "ml": {
          const { stdout } = await execPythonSerialized(
            `python3 -c "
import json, os, time
try:
    from storage_config import ML_MODEL_PATH, TRADE_FEEDBACK_PATH
except ImportError:
    ML_MODEL_PATH = '/tmp/voltrade_ml_model.pkl'; TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'
from ml_model_v2 import fills_slippage_stats
from diagnostics import check_model_health
s = {'model_exists': os.path.exists(ML_MODEL_PATH)}
if s['model_exists']: s['model_age_hours'] = round((time.time() - os.path.getmtime(ML_MODEL_PATH)) / 3600, 1)
try:
    with open(TRADE_FEEDBACK_PATH) as f: _feedback = json.load(f)
except Exception:
    _feedback = []
s['feedback_count'] = len(_feedback)
s['fills_count'] = fills_slippage_stats(_feedback)['count']
# REPAIR 2026-07-06 (KNOWN BROKEN #3/#4 verification): feedback_count and
# fills_count above are computed over the RAW file (seeded backtest records
# + live records mixed) — cannot tell from them alone whether live trades
# are actually completing and recording fills. check_model_health() already
# does the exact seeded/corrupt filtering the dashboard perf endpoint uses
# (server/bot.ts ALPHA-AUDIT-2026-05-03 batch 3) — reuse it instead of
# re-deriving the filter here.
s['feedback_seeded_count'] = sum(1 for r in _feedback if isinstance(r, dict) and r.get('_seed'))
s['feedback_live_count'] = s['feedback_count'] - s['feedback_seeded_count']
_health = check_model_health()
s['retrain_needed'] = _health.get('retrain_needed', False)
s['retrain_overdue'] = _health.get('retrain_overdue', False)
s['live_performance'] = _health.get('performance', {})
# REPAIR 2026-07-06 pt.2: first live query (fills_count=0, feedback_live_count=500,
# live_performance.total_trades=0) is only possible if every live record has BOTH
# pnl_pct=None (excluded from total_trades) AND no expected_price/slippage_pct
# (excluded from fills_count) — track_fill() sets both of those on ENTRY creation
# unconditionally, so a normal entry record could not produce this combination.
# The one record shape that matches is track_fill()'s "orphan_exit" fallback
# (exit fill found no matching open entry -> logged with pnl_pct=None and no
# expected_price field at all). Breaking down by outcome (aggregate counts only,
# no ticker/price data) tells us directly whether that's what's happening instead
# of inferring it from two absences.
_outcomes = {}
for _r in _feedback:
    if not isinstance(_r, dict) or _r.get('_seed'):
        continue
    _k = _r.get('outcome') if _r.get('outcome') is not None else 'open'
    _outcomes[_k] = _outcomes.get(_k, 0) + 1
s['live_outcome_breakdown'] = _outcomes
# REPAIR 2026-07-06 pt.3: the pt.2 breakdown REFUTED the orphan_exit theory
# (500 live records, all bucketed 'open', zero orphan_exit) — but 'open' only
# says outcome-is-missing; it cannot say WHICH writer created the records.
# Four candidate writers each contradict one observed field: track_fill entry
# always stamps expected_price (fills_count would be >0); the seeder always
# stamps _seed; trackClosedTrades' feedback block filters pnlPct!==0/null
# before writing (total_trades would be >0); legacy ml_model.track_fill writes
# a different file. Shape aggregates (field-NAME signatures + code_version /
# session distributions + record time range — never tickers or prices) name
# the writer directly instead of a fifth round of inference-from-absences.
_sigs, _cvs, _sess = {}, {}, {}
_tmin = _tmax = None
_has_feat = _pnl_null = _pnl_zero = 0
for _r in _feedback:
    if not isinstance(_r, dict) or _r.get('_seed'):
        continue
    _sig = ','.join(sorted(_r.keys()))
    _sigs[_sig] = _sigs.get(_sig, 0) + 1
    _cv = str(_r.get('code_version'))
    _cvs[_cv] = _cvs.get(_cv, 0) + 1
    _sn = str(_r.get('session'))
    _sess[_sn] = _sess.get(_sn, 0) + 1
    if _r.get('entry_features'): _has_feat += 1
    if _r.get('pnl_pct') is None: _pnl_null += 1
    elif _r.get('pnl_pct') == 0: _pnl_zero += 1
    _t = str(_r.get('time_filled') or _r.get('timestamp') or '')[:10]
    if _t:
        _tmin = _t if _tmin is None or _t < _tmin else _tmin
        _tmax = _t if _tmax is None or _t > _tmax else _tmax
s['live_key_signatures'] = dict(sorted(_sigs.items(), key=lambda kv: -kv[1])[:5])
s['live_code_versions'] = _cvs
s['live_sessions'] = _sess
s['live_entry_features_count'] = _has_feat
s['live_pnl_null_count'] = _pnl_null
s['live_pnl_zero_count'] = _pnl_zero
s['live_record_date_range'] = [_tmin, _tmax]
print(json.dumps(s))
"`, { timeout: 15000 });
          return res.json(sanitizeDiag({ probe: "ml", ...JSON.parse(stdout.toString().trim() || "{}") }));
        }
        case "daemon": {
          if (!DAEMON_ENABLED) return res.json({ probe: "daemon", alive: false, reason: "disabled via env" });
          try {
            const r = await pythonRpc("health", {});
            return res.json(sanitizeDiag({ probe: "daemon", alive: r.status === "ok", ...(r.status === "ok" ? r.result : { reason: r.error_message }) }));
          } catch (e: any) {
            return res.json({ probe: "daemon", alive: false, reason: sanitizeDiag(String(e?.message || e)) });
          }
        }
        case "positions": {
          const positions = await alpaca("/v2/positions");
          return res.json(sanitizeDiag({ probe: "positions", ...positionsSummary(Array.isArray(positions) ? positions : []) }));
        }
        case "positions-detail": {
          // Whitelist widened 2026-07-07 (human-directed) — see diag.ts.
          const positions = await alpaca("/v2/positions");
          const arr = Array.isArray(positions) ? positions : [];
          return res.json(sanitizeDiag({
            probe: "positions-detail",
            ...positionsSummary(arr),
            positions: arr.map(positionRow),
          }));
        }
        case "orders": {
          // Whitelist widened 2026-07-07 (human-directed) — see diag.ts.
          const status = ["open", "closed", "all"].includes(String(req.query.status))
            ? String(req.query.status) : "all";
          const limit = Math.min(Math.max(parseInt(String(req.query.limit || "100"), 10) || 100, 1), 200);
          const after = String(req.query.after || "").trim();
          const qs = `status=${status}&limit=${limit}&direction=desc`
            + (after ? `&after=${encodeURIComponent(after)}` : "");
          const orders = await alpaca(`/v2/orders?${qs}`);
          const arr = Array.isArray(orders) ? orders : [];
          return res.json(sanitizeDiag({ probe: "orders", count: arr.length, orders: arr.map(orderRow) }));
        }
        case "scanner": {
          // REPAIR 2026-07-06: the free-text failure reason (subprocess
          // stderr tails, raw Alpaca error bodies) belongs behind the
          // token gate, not on public /api/health — see the "scanner"
          // check there for the bare status+count public surface.
          return res.json(sanitizeDiag({
            probe: "scanner",
            degraded: scannerDegraded(tier2ConsecutiveFailures),
            consecutiveFailures: tier2ConsecutiveFailures,
            lastFailureDetail: tier2LastFailureDetail,
            // REPAIR 2026-07-06 pt.2: deep_score's per-source enrichment
            // errors (macro/intel/alt/social/finnhub) — silent signal
            // degradation, distinct from the scan pass/fail fields above.
            dataSourceErrors: tier2LastDataSourceErrors,
          }));
        }
        case "timings": {
          // REPAIR 2026-07-18 (KNOWN BROKEN #18 recurrence investigation):
          // scan_market's daemon-thread path (bot_engine.py, TIMING-DISK
          // 2026-04-23) already persists per-phase wall-clock timings to
          // /data/voltrade/voltrade_scan_timings.json (or /tmp locally) on
          // every phase boundary — written incrementally, so it survives a
          // 300s daemon-timeout kill and shows exactly which phase a stuck
          // scan reached. Mirrors the read-only file lookup /api/system/
          // snapshot already does (TIMING-FILE-DIRECT 2026-04-23) but
          // without requiring the owner cookie, so a token-only session can
          // read it too.
          try {
            const timingPaths = [
              "/data/voltrade/voltrade_scan_timings.json",
              "/tmp/voltrade_scan_timings.json",
            ];
            for (const tpath of timingPaths) {
              if (fs.existsSync(tpath)) {
                const raw = fs.readFileSync(tpath, "utf8");
                return res.json(sanitizeDiag({ probe: "timings", found: true, ...JSON.parse(raw) }));
              }
            }
            return res.json({ probe: "timings", found: false });
          } catch (e: any) {
            return res.json({ probe: "timings", found: false, error: sanitizeDiag(String(e?.message || e)) });
          }
        }
        case "archive": {
          // ADDED 2026-07-26 (scheduled-routine PRODUCT session): generic,
          // read-only, one-day-per-call passthrough of a datacore archive
          // directory (see readArchiveDay in datacoreArchive.ts and the
          // DIAG_PROBES comment in diag.ts). Filed to unblock the
          // USAspending gate-2 statistical test, which needs the
          // multi-week historical archive — /api/data/contracts only
          // ever serves the in-memory recent-cache window, not history.
          //
          // Whitelist is "the directory already exists under
          // archiveBaseDir()" (stream regex blocks path traversal; a
          // directory only exists if some datacore writer already
          // created it) rather than a second hardcoded stream list —
          // every archived stream is either public/licensed source data
          // (see each stream's datacore/manifests/*.json) or already
          // displayed live on /data, so there is no additional secret
          // surface a new stream could introduce here.
          //
          // Row cap diverges from the other probes' 200-item convention
          // (matching sanitizeDiag's blanket array truncation): archive
          // rows are federal/economic/tracking records with no secrets
          // by construction, and a real gate-2 sample needs more than
          // 200 rows/day, so this probe sanitizes PER ROW (still scrubs
          // key-like strings/hex/base64/emails, still truncates long
          // strings) rather than passing the whole `rows` array through
          // sanitizeDiag's 200-item cap — and reports an explicit
          // `truncated` flag instead of silently dropping rows past 200.
          const stream = String(req.query.stream || "").trim();
          const day = String(req.query.day || "").trim();
          if (!/^[a-z0-9_]+$/.test(stream)) {
            return res.status(400).json({ error: "invalid stream (expected [a-z0-9_]+)" });
          }
          if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) {
            return res.status(400).json({ error: "invalid day (expected YYYY-MM-DD)" });
          }
          const limit = Math.min(Math.max(parseInt(String(req.query.limit || "1000"), 10) || 1000, 1), 5000);
          const result = await readArchiveDay(stream, day, undefined, limit);
          if (!result) return res.status(404).json({ error: "unknown stream (no archive directory on this instance)" });
          return res.json({
            probe: "archive", stream, day,
            files: result.files, count: result.rows.length, truncated: result.truncated,
            rows: result.rows.map((r) => sanitizeDiag(r)),
          });
        }
        default:
          return res.status(404).json({ error: "unknown probe", probes: DIAG_PROBES });
      }
    } catch (e: any) {
      return res.status(500).json({ error: sanitizeDiag(String(e?.message || e)) });
    }
  });

  // Place a trade (manual or from bot signals)
  app.post("/api/bot/trade", requireOwner, async (req, res) => {
    if (state.killSwitch) return res.status(400).json({ error: "Kill switch is ON" });
    const { ticker, side, qty, type = "market" } = req.body || {};
    if (!ticker || !side || !qty) return res.status(400).json({ error: "ticker, side, qty required" });

    // Phase 6: Security check before trading
    try {
      const acct = await alpaca("/v2/account");
      const portfolioValue = parseFloat(acct.portfolio_value);
      const currentPrice = parseFloat(acct.last_equity) / 100; // rough estimate
      const tradeValue = qty * (currentPrice || 1);
      const check = await checkTradeAllowed(portfolioValue, tradeValue);
      if (!check.allowed) {
        audit("BLOCKED", `Trade rejected: ${check.reason}`);
        return res.status(400).json({ error: check.reason });
      }
    } catch (err: any) { console.error("[bot]", err?.message || err); }

    try {
      // ══ HARD BLOCK: No stock shorting via manual API (PR #53) ══════════
      let orderSide = side;
      if (side === "short" || side === "sell") {
        // Check if this is a close (user holds a long position) or a new short
        try {
          const posCheck = await alpaca(`/v2/positions/${ticker.toUpperCase()}`);
          if (posCheck && posCheck.side === "long") {
            orderSide = "sell"; // Closing a long position — allowed
          } else {
            return res.status(400).json({ error: "Stock shorting is disabled. Use options for bearish plays." });
          }
        } catch {
          // No existing position — this would be a short sale. Block it.
          return res.status(400).json({ error: "Stock shorting is disabled. Use options for bearish plays." });
        }
      }
      const order = await alpaca("/v2/orders", {
        method: "POST",
        body: JSON.stringify({
          symbol: ticker.toUpperCase(),
          qty: String(qty),
          side: orderSide,
          type,
          time_in_force: "day",
        }),
      });
      audit("TRADE", `${side.toUpperCase()} ${qty} ${ticker} @ ${type}`);
      notify("trade", `${side.toUpperCase()} ${qty} ${ticker} @ ${type}`);
      res.json(order);
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // Close a position
  app.post("/api/bot/close", requireOwner, async (req, res) => {
    const { ticker } = req.body || {};
    if (!ticker) return res.status(400).json({ error: "ticker required" });
    try {
      await alpaca(`/v2/positions/${String(ticker).toUpperCase()}`, { method: "DELETE" });
      removePositionFromMonitor(String(ticker).toUpperCase());
      audit("CLOSE", `Closed position in ${ticker}`);
      notify("trade", `Position closed: ${ticker}`);
      res.json({ ok: true });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Chart data from Alpaca ──────────────────────────────────────────────────
  app.get("/api/bot/bars/:ticker", requireOwner, async (req, res) => {
    const { ticker } = req.params;

    // Detect OCC option symbols (e.g., XLK260424P00149000) and return empty bars
    const tickerStr = String(ticker).toUpperCase();
    if (isOptionSymbol(tickerStr)) {
      return res.json({ bars: [], ticker: tickerStr, timeframe: "1Day", isOptionSymbol: true });
    }

    const timeframe = (String(req.query.timeframe || "1Day")) || "1Day";
    const limit = parseInt(String(req.query.limit || "200")) || 200;

    // Map timeframe to Alpaca format
    const tfMap: Record<string, string> = {
      "1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "1Hour": "1Hour",
      "1Day": "1Day", "1Week": "1Week",
    };
    const tf = tfMap[timeframe] || "1Day";

    try {
      // 2026-07-20/27 [REPAIR]: the sip feed param is rejected by this
      // account's data subscription. BARS specifically also reject
      // delayed_sip — HTTP 400 "invalid feed: delayed_sip" (live evidence
      // 2026-07-18, compiled in alpaca_feed.py bars_feed(): delayed_sip is
      // a real-time-tape delay concept, not a bars-endpoint feed value).
      // iex is the always-accepted value for bars, per that precedent.
      const url = `https://data.alpaca.markets/v2/stocks/${String(ticker).toUpperCase()}/bars?timeframe=${tf}&limit=${limit}&adjustment=split&feed=iex`;
      const r = await fetch(url, {
        headers: {
          "APCA-API-KEY-ID": ALPACA_KEY,
          "APCA-API-SECRET-KEY": ALPACA_SECRET,
        },
      });
      const data = await r.json();
      const bars = (data.bars || []).map((b: any) => ({
        time: Math.floor(new Date(b.t).getTime() / 1000),
        open: b.o,
        high: b.h,
        low: b.l,
        close: b.c,
        volume: b.v,
      }));
      res.json({ bars, ticker: String(ticker).toUpperCase(), timeframe: tf });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Daemon health (OPT 2026-04-20) ─────────────────────────────────────────
  // Returns { alive, rss_mb, uptime_seconds } if daemon is running, or
  // { alive: false, reason: "..." } if down. Useful for monitoring.
  app.get("/api/daemon/health", requireOwner, async (_req, res) => {
    if (!DAEMON_ENABLED) {
      return res.json({ alive: false, reason: "Daemon disabled via env" });
    }
    try {
      const r = await pythonRpc("health", {});
      if (r.status === "ok") {
        res.json({ alive: true, ...r.result });
      } else {
        res.json({ alive: false, reason: r.error_message || "unknown" });
      }
    } catch (e: any) {
      res.json({ alive: false, reason: e.message });
    }
  });

  // ── Monitoring endpoints (OPT 2026-04-20) ─────────────────────────────────
  // Consolidated system health — drawdown proximity, ML status, kill switch
  // state, shadow portfolio divergence. Used for dashboard + alerting.
  app.get("/api/monitoring/overview", requireOwner, async (_req, res) => {
    const overview: any = {
      timestamp: new Date().toISOString(),
      daemon: { alive: false },
      kill_switches: { active: 0, warnings: [] },
      memory: { rss_mb: Math.round(process.memoryUsage().rss / 1024 / 1024) },
      drawdown: {},
      ml: {},
    };

    // Daemon
    if (DAEMON_ENABLED) {
      try {
        const d = await pythonRpc("health", {});
        if (d.status === "ok") overview.daemon = { alive: true, ...d.result };
        else overview.daemon = { alive: false, reason: d.error_message };
      } catch (e: any) { overview.daemon = { alive: false, reason: e.message }; }
    }

    // Kill switch status (via daemon or subprocess)
    try {
      const ks = await pythonCall(
        "risk_status", {},
        `python3 -c "from risk_kill_switch import get_kill_switch_status; import json; print(json.dumps(get_kill_switch_status()))"`,
        { timeout: 5000 }
      );
      if (ks.success) {
        overview.kill_switches = ks.result;
        // Compute drawdown proximity
        const dd = ks.result.current_dd_pct || 0;
        overview.drawdown = {
          current_pct: dd,
          kill_threshold_pct: -20,
          proximity_pct: Math.max(0, Math.min(100, ((-20 - dd) / -20) * 100)),
          status: dd <= -15 ? "CRITICAL" : dd <= -10 ? "WARNING" : "OK",
        };
      }
    } catch {}

    // ML status
    try {
      const ml = await pythonCall(
        "ml_status", {},
        `python3 ml_status.py`,
        { timeout: 10000 }
      );
      if (ml.success) overview.ml = ml.result;
    } catch {}

    res.json(overview);
  });

  // ── Per-strategy PnL breakdown ─────────────────────────────────────────────
  // ALPHA AUDIT 2026-05-03 BATCH 3: rewritten to match the new bucket names
  // produced by position_sizing._infer_strategy() and the seeded backtest
  // records. Original implementation looked for strategies named csp_core,
  // leveraged_csp, trend_long, tail_hedge — none of which the bot has ever
  // written. Result: dashboard tier-pnl tab showed zeros for everything
  // legitimate and dumped all real activity into 'legacy' (which the UI
  // probably hides). This was broken from day one — most visible now that
  // the trade_feedback seeder loads 1326 records.
  //
  // Also fixes the timestamp field — original code looked for r.timestamp
  // which neither the seeder NOR track_fill writes. Now reads time_filled
  // (which both write) with timestamp as a fallback for any legacy records.
  app.get("/api/monitoring/tier-pnl", requireOwner, async (req, res) => {
    const days = parseInt((req.query.days as string) || "30", 10);
    try {
      const cmd = `python3 -c "
import json, os, time
try:
    from storage_config import TRADE_FEEDBACK_PATH
except ImportError:
    TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'

cutoff = time.time() - ${days} * 86400

# Buckets aligned with position_sizing._infer_strategy():
#   csp_options — CSPs and other premium-selling structures
#   etf         — leveraged ETF trades (the alpha generator)
#   stocks      — single-name directional stocks
#   vrp         — VRP-dominant trades
#   other       — anything that doesn't match (data quality bucket)
bucket_stats = {
    'csp_options': {'count': 0, 'pnl_pct_sum': 0, 'wins': 0, 'seeded': 0},
    'etf':         {'count': 0, 'pnl_pct_sum': 0, 'wins': 0, 'seeded': 0},
    'stocks':      {'count': 0, 'pnl_pct_sum': 0, 'wins': 0, 'seeded': 0},
    'vrp':         {'count': 0, 'pnl_pct_sum': 0, 'wins': 0, 'seeded': 0},
    'other':       {'count': 0, 'pnl_pct_sum': 0, 'wins': 0, 'seeded': 0},
}

try:
    with open(TRADE_FEEDBACK_PATH) as f:
        records = json.load(f)
except Exception:
    records = []

from datetime import datetime
def to_unix(ts):
    if not ts: return 0
    if isinstance(ts, (int, float)): return float(ts)
    s = str(ts)
    try:
        if 'T' in s:
            return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
        # Bare YYYY-MM-DD from seeded records
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0

for r in records:
    # Try multiple time field names — seeder writes time_filled,
    # track_fill writes time_filled, legacy code may have written timestamp.
    rt = to_unix(r.get('time_filled') or r.get('timestamp') or r.get('time_placed'))
    if rt < cutoff: continue
    pnl = r.get('pnl_pct')
    if pnl is None: continue  # incomplete trade — exit hasn't filled

    strat = r.get('strategy', 'other')
    if strat not in bucket_stats:
        strat = 'other'
    is_seed = bool(r.get('_seed'))
    if is_seed:
        bucket_stats[strat]['seeded'] += 1
        # Track seeded PnL separately so we can show it for context
        bucket_stats[strat].setdefault('seeded_pnl_sum', 0)
        bucket_stats[strat].setdefault('seeded_wins', 0)
        bucket_stats[strat]['seeded_pnl_sum'] += float(pnl)
        if pnl > 0:
            bucket_stats[strat]['seeded_wins'] += 1
    else:
        bucket_stats[strat]['count'] += 1
        bucket_stats[strat]['pnl_pct_sum'] += float(pnl)
        if pnl > 0:
            bucket_stats[strat]['wins'] += 1

# Compute derived metrics
# DASHBOARD-FIX 2026-05-03: primary trade_count and total_pnl_pct are
# LIVE-only (seeded backtest records aren't real trades). Seeded numbers
# exposed as a separate object so the UI can show 'live: X | seeded: Y'.
result = {'days': ${days}, 'tiers': {}}
for bucket, s in bucket_stats.items():
    count = s['count']
    seed_count = s['seeded']
    result['tiers'][bucket] = {
        'trade_count':       count,                # LIVE only
        'total_pnl_pct':     round(s['pnl_pct_sum'], 2),
        'avg_pnl_pct':       round(s['pnl_pct_sum'] / count, 3) if count > 0 else 0,
        'win_rate':          round(s['wins'] / count * 100, 1) if count > 0 else 0,
        'seeded': {
            'count':         seed_count,
            'total_pnl_pct': round(s.get('seeded_pnl_sum', 0), 2),
            'win_rate':      round(s.get('seeded_wins', 0) / seed_count * 100, 1) if seed_count > 0 else 0,
        },
    }
print(json.dumps(result))
"`;
      const tierCall = await pythonCall("strategy_pnl_stats", { days },
        cmd, { timeout: 10000 });
      if (tierCall.success) {
        res.json(tierCall.result);
      } else {
        res.status(500).json({ error: "Could not compute strategy PnL" });
      }
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "Unknown error" });
    }
  });

  // Drawdown proximity alert — returns status + how close to kill threshold
  app.get("/api/monitoring/drawdown", requireOwner, async (_req, res) => {
    try {
      const acct = await alpaca("/v2/account");
      const equity = parseFloat(acct.equity || "0");
      // Peak equity from risk_kill_switch's persisted state
      const peakCall = await pythonCall(
        "get_peak_equity", {},
        `python3 -c "from risk_kill_switch import get_peak_equity; import json; print(json.dumps({'peak': get_peak_equity()}))"`,
        { timeout: 5000 }
      );
      const peak = peakCall.success ? (peakCall.result.peak || equity) : equity;
      const dd = peak > 0 ? ((equity - peak) / peak) : 0;
      res.json({
        equity,
        peak,
        drawdown_pct: Math.round(dd * 10000) / 100,  // in basis points, 2 decimals
        kill_threshold_pct: -20,
        status: dd <= -0.20 ? "KILLED" :
                dd <= -0.15 ? "CRITICAL" :
                dd <= -0.10 ? "WARNING" :
                dd <= -0.05 ? "NOTICE" : "OK",
      });
    } catch (e: any) {
      res.status(500).json({ error: e?.message?.slice(0, 200) });
    }
  });

  // Cache inventory — for operational visibility
  app.get("/api/monitoring/caches", requireOwner, async (_req, res) => {
    try {
      const cacheCall = await pythonCall(
        "cache_inventory", {},
        `python3 -c "
import os, json, glob
result = []
for pattern in ['/tmp/voltrade_*.json', '/tmp/voltrade_alt_cache/*.json']:
    for f in glob.glob(pattern):
        try:
            size = os.path.getsize(f)
            age = (os.path.getmtime(f))
            result.append({'path': f, 'size_kb': round(size/1024, 1)})
        except: pass
result.sort(key=lambda x: -x['size_kb'])
print(json.dumps(result[:20]))
"`,
        { timeout: 5000 }
      );
      res.json({ caches: cacheCall.success ? cacheCall.result : [] });
    } catch (e: any) {
      res.status(500).json({ error: e?.message?.slice(0, 200) });
    }
  });

  // ── Market status ───────────────────────────────────────────────────────────
  app.get("/api/bot/market-status", async (_req, res) => {
    try {
      const clock = await alpaca("/v2/clock");
      res.json({
        isOpen: clock.is_open,
        nextOpen: clock.next_open,
        nextClose: clock.next_close,
        timestamp: clock.timestamp,
      });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── AI Signals ────────────────────────────────────────────────────────────
  const signals: Array<{
    ticker: string; action: string; reason: string;
    confidence: number; timestamp: string; type: string;
  }> = [];

  // Generate signals for a list of tickers
  async function generateSignals(tickers: string[]) {
    const scriptPath = path.resolve("analyze.py");
    for (const ticker of tickers) {
      try {
        const { stdout } = await execPythonSerialized(
          `python3 "${scriptPath}" "${ticker}" --mode=scan`,
          { timeout: 30000 }
        );
        if (!stdout.trim()) continue;
        const raw = JSON.parse(stdout.trim());

        // Score the signal
        const vrp = raw.vrp ?? 0;
        const rec = raw.recommendation;
        const edge = raw.edge_factors;

        let action = "HOLD";
        let reason = "";
        let confidence = 50;
        let type = "neutral";

        // VRP signal
        if (vrp > 5) {
          action = "SELL OPTIONS";
          reason = `IV is ${vrp.toFixed(1)}% above realized vol — options are overpriced. Sell premium.`;
          confidence = Math.min(90, 50 + vrp * 3);
          type = "sell";
        } else if (vrp < -3) {
          action = "BUY OPTIONS";
          reason = `IV is ${Math.abs(vrp).toFixed(1)}% below realized vol — options are cheap. Buy for upside.`;
          confidence = Math.min(85, 50 + Math.abs(vrp) * 3);
          type = "buy";
        }

        // Override with recommendation if available
        if (rec?.action && rec.action !== "WAIT") {
          action = rec.action;
          reason = rec.reasoning || reason;
          confidence = rec.signal?.includes("🚀") ? 80 : rec.signal?.includes("⚠️") ? 40 : 60;
          type = rec.action.toLowerCase().includes("buy") ? "buy" : rec.action.toLowerCase().includes("sell") ? "sell" : "neutral";
        }

        // Squeeze boost
        if (edge?.squeeze_score > 70) {
          reason += ` | SQUEEZE ALERT: ${edge.squeeze_desc}`;
          confidence = Math.min(95, confidence + 15);
        }

        if (action !== "HOLD" && confidence > 45) {
          // Remove old signal for this ticker
          const idx = signals.findIndex(s => s.ticker === ticker);
          if (idx >= 0) signals.splice(idx, 1);

          signals.unshift({
            ticker,
            action,
            reason,
            confidence: Math.round(confidence),
            timestamp: new Date().toISOString(),
            type,
          });

          // Keep max 20 signals
          if (signals.length > 20) signals.length = 20;

          audit("SIGNAL", `${action} ${ticker} (${Math.round(confidence)}% confidence)`);
        }
      } catch (err: any) { console.error("[bot]", err?.message || err); }
    }
  }

  // Signals endpoint
  app.get("/api/bot/signals", requireOwner, (_req, res) => {
    res.json(signals);
  });

  // Manual signal refresh
  app.post("/api/bot/refresh-signals", requireOwner, async (_req, res) => {
    const topTickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "AMD", "GOOGL"];
    audit("REFRESH", "Generating fresh signals for top 10 tickers...");
    await generateSignals(topTickers);
    res.json({ ok: true, count: signals.length });
  });

  // ── Backtesting ───────────────────────────────────────────────────────────
  app.post("/api/bot/backtest", requireOwner, async (req, res) => {
    const { ticker = "SPY", strategy = "all", years = 3 } = req.body || {};
    const scriptPath = path.resolve("backtest.py");

    try {
      audit("BACKTEST", `Running ${strategy} on ${ticker} (${years}yr)`);
      const { stdout } = await execPythonSerialized(
        `python3 "${scriptPath}" "${ticker}" "${strategy}" "${years}"`,
        { timeout: 120000 }
      );
      const result = JSON.parse(stdout.trim());
      audit("BACKTEST", `Complete — ${result.results?.length || 0} strategies tested`);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Smart Execution Engine ──────────────────────────────────────────────────

  // Morning queue: trades discovered after-hours, executed at 9:30am market open
  interface QueuedTrade {
    ticker: string;
    side: string;
    shares: number;
    price: number;
    score: number;
    reasons: string[];
    trade_type: string;
    recommendation?: string;
    rec_reasoning?: string;
    position_value: number;
    queuedAt: string;
    vrp?: number;
  }
  const morningQueue: QueuedTrade[] = [];

  // Track open orders so we can sweep stale ones
  interface TrackedOrder {
    orderId: string;
    ticker: string;
    score: number;
    placedAt: number; // epoch ms
    side: string;
    qty: number;
    limitPrice: number;
  }
  const openOrders: TrackedOrder[] = [];

  // ── Per-Ticker Order Attempt Counter (Fix 2026-04-10, relaxed 2026-04-13) ─
  // Originally set to 3 to prevent the ELAB retry-loop bug (10 canceled orders
  // per scan cycle). Raised to 10 because the hard cap of 3 was blocking
  // legitimate re-entries throughout the day — other protections (spread filter,
  // limit orders, stale-order sweeper, stop cooldowns, circuit breakers) now
  // prevent the original churn problem. The counter resets daily at 4am ET.
  // NOTE: This only gates new ENTRY orders. Exits (stops, TP, time stops) use
  // the WebSocket handler and are never blocked by this counter.
  const MAX_ORDER_ATTEMPTS_PER_TICKER = 10;
  let orderAttemptCounts: Record<string, number> = {};

  // ── Stale Order Sweeper ──────────────────────────────────────────────────
  const STALE_ORDER_MINUTES = 12; // cancel unfilled limits after 12 minutes

  async function sweepStaleOrders() {
    if (openOrders.length === 0) return;

    const now = Date.now();
    const staleThreshold = STALE_ORDER_MINUTES * 60 * 1000;
    const toCancel: TrackedOrder[] = [];

    for (const tracked of openOrders) {
      if (now - tracked.placedAt > staleThreshold) {
        toCancel.push(tracked);
      }
    }

    for (const stale of toCancel) {
      try {
        await alpaca(`/v2/orders/${stale.orderId}`, { method: "DELETE" });
        audit("SWEEP", `Cancelled stale order: ${stale.side.toUpperCase()} ${stale.qty} ${stale.ticker} (unfilled for ${STALE_ORDER_MINUTES}+ min, score ${stale.score}) — buying power freed`);
        // Remove from tracked list
        const idx = openOrders.findIndex(o => o.orderId === stale.orderId);
        if (idx >= 0) openOrders.splice(idx, 1);
        // Clear pendingExit so the position can be re-evaluated for exit
        if (monitoredPositions[stale.ticker]) {
          monitoredPositions[stale.ticker].pendingExit = false;
        }
      } catch (e: any) {
        // Order may have already filled or been cancelled
        const idx = openOrders.findIndex(o => o.orderId === stale.orderId);
        if (idx >= 0) openOrders.splice(idx, 1);
        // Also clear pendingExit — if the order filled, the next sync will remove
        // the position entirely; if cancelled, we want to allow re-evaluation
        if (monitoredPositions[stale.ticker]) {
          monitoredPositions[stale.ticker].pendingExit = false;
        }
      }
    }

    // Also sync with Alpaca — remove tracked orders that Alpaca says are filled/cancelled
    try {
      const alpacaOrders = await alpaca("/v2/orders?status=open");
      const alpacaIds = new Set((alpacaOrders as any[]).map((o: any) => o.id));
      for (let i = openOrders.length - 1; i >= 0; i--) {
        if (!alpacaIds.has(openOrders[i].orderId)) {
          openOrders.splice(i, 1); // Filled or cancelled externally
        }
      }
    } catch (err: any) { console.error("[bot]", err?.message || err); }
  }

  // ── Score-Based Order Replacement ──────────────────────────────────────────
  async function replaceIfBetter(newTrade: any): Promise<boolean> {
    // Find the lowest-score open order
    if (openOrders.length === 0) return false;

    const weakest = openOrders.reduce((a, b) => a.score < b.score ? a : b);

    // Only replace if new trade is significantly better (10+ points)
    if (newTrade.score <= weakest.score + 10) return false;

    try {
      await alpaca(`/v2/orders/${weakest.orderId}`, { method: "DELETE" });
      audit("REPLACE", `Cancelled ${weakest.ticker} (score ${weakest.score}) → replacing with ${newTrade.ticker} (score ${newTrade.score})`);
      const idx = openOrders.findIndex(o => o.orderId === weakest.orderId);
      if (idx >= 0) openOrders.splice(idx, 1);
      return true;
    } catch {
      return false;
    }
  }

  // ── Morning Queue Execution ────────────────────────────────────────────────
  async function executeMorningQueue() {
    if (morningQueue.length === 0) return;

    // Filter out stale orders (queued more than 16 hours ago)
    const now = Date.now();
    const freshQueue = morningQueue.filter((t: any) => {
      const queuedTime = t.queuedAt ? new Date(t.queuedAt).getTime() : 0;
      const ageHours = (now - queuedTime) / 3600000;
      if (ageHours > 16) {
        audit("MORNING-STALE", `${t.ticker}: skipped — queued ${ageHours.toFixed(0)}h ago (max 16h)`);
        return false;
      }
      return true;
    });
    if (freshQueue.length < morningQueue.length) {
      audit("MORNING", `Dropped ${morningQueue.length - freshQueue.length} stale orders from queue`);
    }
    morningQueue.length = 0;
    morningQueue.push(...freshQueue);
    audit("MORNING", `Executing ${morningQueue.length} queued trades from overnight research...`);

    // Sort by score descending — best picks first
    morningQueue.sort((a, b) => b.score - a.score);

    const acct = await alpaca("/v2/account");

    // Max drawdown check on every Tier 1 cycle — VALIDATED reads only
    // (2026-07-07 incident: the old `parseFloat(acct.equity || "100000")`
    // let a garbage "0" through — computing as -100% — and fabricated
    // equity when the field was absent; see server/drawdownGuard.ts)
    const t1dd = evaluateDrawdown(acct.equity, state.equityPeak, state.maxDrawdownPct);
    if (!t1dd.valid) {
      audit("EQUITY-READ-INVALID", `tier1: equity=${JSON.stringify(acct.equity)} — drawdown eval skipped (no kill, no peak update)`);
    }
    const equity = t1dd.valid ? t1dd.equity! : state.equityPeak;
    if (t1dd.valid && t1dd.newPeak !== state.equityPeak) { state.equityPeak = t1dd.newPeak; saveEquityPeak(); }
    const t1Drawdown = t1dd.drawdownPct ?? 0;
    if (t1dd.valid && t1dd.kill && !state.killSwitch) {
      state.killSwitch = true;
      saveKillSwitch("drawdown-kill (tier1)");
      state.active = false;
      const msg = `MAX DRAWDOWN KILL SWITCH: Equity $${equity.toFixed(0)} is ${t1Drawdown.toFixed(1)}% below peak $${state.equityPeak.toFixed(0)}`;
      audit("DRAWDOWN-KILL", msg);
      sendEmailAlert("MAX DRAWDOWN — Trading Stopped", msg);
      try { await alpaca("/v2/orders", { method: "DELETE" }); } catch {}

      // ═══════════════════════════════════════════════════════════════
      // ITEM 15 FIX 2026-04-20: Liquidate-all on deep drawdown
      // If user opted in via env var VOLTRADE_LIQUIDATE_ON_KILL=true AND
      // drawdown is at -25% or worse, force-close all positions rather
      // than just blocking new trades. Mercy rule for catastrophic losses.
      // ═══════════════════════════════════════════════════════════════
      if (process.env.VOLTRADE_LIQUIDATE_ON_KILL === "true" && t1Drawdown <= -25.0) {
        audit("LIQUIDATE-ALL", `Drawdown ${t1Drawdown.toFixed(1)}% <= -25% + VOLTRADE_LIQUIDATE_ON_KILL=true — closing all positions`);
        try {
          const allPositions = await alpaca("/v2/positions");
          if (Array.isArray(allPositions)) {
            for (const pos of allPositions) {
              try {
                const psym = pos.symbol || "";
                const pqty = Math.abs(parseInt(pos.qty || "0"));
                const pside = pos.side === "long" ? "sell" : "buy";
                if (pqty > 0) {
                  await alpaca("/v2/orders", {
                    method: "POST",
                    body: JSON.stringify({
                      symbol: psym, qty: String(pqty), side: pside,
                      type: "market", time_in_force: "day",
                    }),
                  });
                  audit("LIQUIDATE", `Closing ${psym} qty=${pqty} side=${pside}`);
                }
              } catch (closeErr: any) {
                audit("LIQUIDATE-ERROR", `${pos.symbol}: ${closeErr?.message?.slice(0, 100)}`);
              }
            }
          }
          sendEmailAlert("ALL POSITIONS LIQUIDATED", `Drawdown hit -25%, all positions closed per VOLTRADE_LIQUIDATE_ON_KILL=true`);
        } catch (liqErr: any) {
          audit("LIQUIDATE-FAIL", `Could not fetch positions: ${liqErr?.message?.slice(0, 100)}`);
        }
      }
      return;
    }

    let slotsUsed = 0;

    try {
      const positions = await alpaca("/v2/positions");
      slotsUsed = Array.isArray(positions) ? positions.length : 0;
    } catch (err: any) { console.error("[bot]", err?.message || err); }

    // Track sectors of current positions for correlation check
    const positionSectors: Record<string, number> = {};
    try {
      const currentPos = await alpaca("/v2/positions");
      if (Array.isArray(currentPos)) {
        for (const p of currentPos) {
          // Simple sector mapping by first letter of symbol — will be refined by bot_engine
          const sym = p.symbol || "";
          positionSectors[sym] = 1;
        }
      }
    } catch (err: any) { console.error("[bot]", err?.message || err); }

    for (const trade of morningQueue) {
      if (state.killSwitch) break;
      if (slotsUsed >= MAX_POSITIONS) {
        audit("MORNING", `Max positions (${MAX_POSITIONS}) reached — skipping remaining queue`);
        break;
      }
      if (trade.position_value > equity * MAX_POSITION_SIZE * state.positionSizeMultiplier) {
        audit("MORNING-BLOCKED", `${trade.ticker}: Position too large`);
        continue;
      }

      // ── Pre-market price check: verify price hasn't gapped more than 5% ──
      try {
        const { stdout: priceCheck } = await execPythonSerialized(
          // 2026-07-20 [REPAIR]: the sip feed param returned {message:...} on this
          // subscription, so this parsed 0 and the 5% overnight-gap guard
          // silently never fired. delayed_sip restores a real price (15-min
          // delayed — fine for detecting an overnight gap).
          `python3 -c "import requests,json,os; r=requests.get('https://data.alpaca.markets/v2/stocks/${trade.ticker}/snapshot?feed=delayed_sip', headers={'APCA-API-KEY-ID':os.environ.get('ALPACA_KEY',''),'APCA-API-SECRET-KEY':os.environ.get('ALPACA_SECRET','')}, timeout=5); d=r.json(); print(d.get('dailyBar',{}).get('c',0) or d.get('latestTrade',{}).get('p',0))"` ,
          { timeout: 8000 }
        );
        const currentPrice = parseFloat(priceCheck.trim());
        if (currentPrice > 0 && trade.price > 0) {
          const gapPct = Math.abs((currentPrice - trade.price) / trade.price) * 100;
          if (gapPct > 5) {
            audit("MORNING-GAP", `${trade.ticker}: Price gapped ${gapPct.toFixed(1)}% overnight (was $${trade.price}, now $${currentPrice}) — skipping to avoid bad fill`);
            continue;
          }
          // Update share count based on current price
          if (gapPct > 2) {
            trade.shares = Math.floor(trade.position_value / currentPrice);
            audit("MORNING-ADJUST", `${trade.ticker}: Price moved ${gapPct.toFixed(1)}%, adjusted qty to ${trade.shares} shares`);
          }
        }
      } catch (err: any) { console.error("[bot]", err?.message || err); }

      // ══ HARD BLOCK: No stock shorting in morning queue (PR #53) ══════════
      if (trade.trade_type !== "options" && (trade.side === "short" || trade.side === "sell")) {
        audit("SHORT-BLOCKED", `${trade.ticker}: side='${trade.side}' blocked in morning queue — stock shorting disabled. Converting to BUY.`);
        trade.side = "buy";
      }

      // ALPHA AUDIT 2026-05-04 batch 5: guard against qty=0 orders.
      // When trade.position_value is small (e.g. $50) and currentPrice is
      // high (e.g. QQQ at $500+), Math.floor produces 0 shares. Submitting
      // qty:0 to Alpaca returns 422 "qty must be > 0" and trashes the
      // morning queue. This morning at 13:30:28Z we hit:
      //   MORNING-ERROR — Failed: QQQ — Alpaca 422: {"code":40010001,
      //                  "message":"qty must be > 0"}
      // Skip the trade gracefully instead of attempting an invalid order.
      const _qtyToSubmit = Math.floor(trade.shares);
      if (_qtyToSubmit < 1) {
        audit("MORNING-SKIP",
          `${trade.ticker}: skipped — computed qty=${_qtyToSubmit} ` +
          `(position_value=$${trade.position_value?.toFixed(2) ?? "?"}, ` +
          `price=$${trade.price?.toFixed(2) ?? "?"}). ` +
          `Position too small to buy a single share.`);
        continue;
      }

      try {
        const order = await alpaca("/v2/orders", {
          method: "POST",
          body: JSON.stringify({
            symbol: trade.ticker,
            qty: String(_qtyToSubmit),
            side: trade.side || "buy",
            ...getOrderParams(trade.price || 0), // Market during regular hours, limit during extended
          }),
        });

        audit("MORNING-TRADE", `MARKET ${(trade.side || "BUY").toUpperCase()} ${_qtyToSubmit} ${trade.ticker} @ market | Score: ${trade.score} | Queued from overnight research`);
        notify("trade", `Morning queue: ${(trade.side || "BUY").toUpperCase()} ${_qtyToSubmit} ${trade.ticker} (score: ${trade.score})`);

        // Track fill with realistic slippage (morning queue = market open = wider slippage)
        try {
          const mVol = trade.volume || 1000000;
          let mSlip = 0.08; // Morning fills are worse (opening volatility)
          if (mVol > 20000000) mSlip = 0.05 + Math.random() * 0.05;
          else if (mVol > 5000000) mSlip = 0.08 + Math.random() * 0.07;
          else mSlip = 0.15 + Math.random() * 0.10;
          const mSide = trade.side || "buy";  // side already normalized above (short blocked)
          const mDir = mSide === "buy" ? 1 : -1;
          const mFillPrice = Math.round((trade.price * (1 + mDir * mSlip / 100)) * 100) / 100;

          const morningFillPayload = {
            ticker: trade.ticker, order_type: "market", side: mSide,
            qty: Math.floor(trade.shares),
            expected_price: trade.price, fill_price: mFillPrice,
            slippage_applied_pct: mSlip,
            time_placed: trade.queuedAt || new Date().toISOString(),
            session: "morning_queue", volume: mVol, score: trade.score,
            instrument: trade.instrument || "stock",
            code_version: pkgVersion,
          };
          const mfTmp = `/tmp/fill_m_${trade.ticker}_${Date.now()}.json`;
          fs.writeFileSync(mfTmp, JSON.stringify(morningFillPayload));
          // OPT 2026-04-20: daemon first, subprocess fallback
          pythonCall(
            "track_fill", morningFillPayload,
            `python3 -c "import json, os; from ml_model_v2 import track_fill; d=json.load(open('${mfTmp}')); os.remove('${mfTmp}'); track_fill(d)"`,
            { timeout: 5000 }
          ).then((r) => {
            if (r.via === "daemon") {
              try { fs.unlinkSync(mfTmp); } catch {}
            }
          }).catch(() => {});
        } catch (err: any) { console.error("[bot]", err?.message || err); }

        slotsUsed++;
        // Auto-subscribe for real-time exit monitoring
        addPositionToMonitor(trade.ticker, "long", trade.price || 0, Math.floor(trade.shares));  // Always long (short blocked)
      } catch (e: any) {
        audit("MORNING-ERROR", `Failed: ${trade.ticker} — ${e.message}`);
      }

      await new Promise(r => setTimeout(r, 500));
    }

    // Clear the queue
    morningQueue.length = 0;
    audit("MORNING", "Queue cleared.");
  }

  // ── Three-Tier Engine ────────────────────────────────────────────────────

  let autoRunning = false; // used by runOvernightResearch
  let lastScanResult: any = null;
  let _shadowBackfilledToday = false;

  // ── Tier 1: Reflex — order management, morning queue (45s) ─────────────────
  // NOTE: Position monitoring (stops, trailing stops, take-profits) has been
  // moved to the WebSocket handler for real-time, event-driven exits.
  // Tier 1 only handles: stale orders, morning queue, trade tracking, and
  // syncing the position monitor's stop state.

  async function tier1Reflex() {
    try {
      // 1. Sweep stale orders
      await sweepStaleOrders();

      // 2. Sync position monitor stop state from bot_engine (refreshes ATR/phases)
      //    This keeps the WS monitor's levels accurate without blocking exits.
      try {
        const { stdout } = await execPythonSerialized(`python3 -c "
from bot_engine import manage_positions
import json
result = manage_positions()
print(json.dumps(result))
"`, { timeout: 15000 });
        const mgmt = JSON.parse(stdout.trim());
        // Bot_engine may still compute actions — but we do NOT execute them here.
        // The WS handler fires exits in real-time. We only use this to refresh
        // the on-disk stop_state.json which syncMonitoredPositions() reads.
        // Log if bot_engine found something the WS monitor should have caught
        for (const action of (mgmt.actions || [])) {
          if (monitoredPositions[action.ticker]) {
            audit("POS-MONITOR-SYNC", `bot_engine flagged ${action.ticker} (${action.type}) — WS monitor should handle`);
          }
        }
      } catch (err: any) {
        console.error("[tier1-sync]", err?.message || err);
      }

      // 3. Refresh monitored positions from Alpaca + stop state
      await syncMonitoredPositions();

      // 4. Execute morning queue on first market-open cycle
      const clockT1 = await alpaca("/v2/clock");
      if (clockT1.is_open && !state.morningQueueExecuted && morningQueue.length > 0) {
        await executeMorningQueue();
        state.morningQueueExecuted = true;
      }

      // 5. Track closed trades for learning
      await trackClosedTrades();

      // Once per day, backfill shadow portfolio outcomes.
      // Runs ~3-5 Alpaca batch calls (token-bucketed to stay within 200/min).
      const nowHour = new Date().getUTCHours();
      if (nowHour === 22 && !_shadowBackfilledToday) {  // 10pm UTC = 5pm EST
          execPythonSerialized(
              `python3 -c "from shadow_portfolio import backfill_outcomes; import json; print(json.dumps(backfill_outcomes(500)))"`,
              { timeout: 120000 }  // 2 min cap
          ).then((result) => {
              console.log("[SHADOW] Backfill result:", result.stdout);
              _shadowBackfilledToday = true;
          }).catch((e) => {
              console.error("[SHADOW] Backfill failed:", e);
          });
      }
      if (nowHour === 0) {
          _shadowBackfilledToday = false;  // reset at midnight UTC
      }

    } catch (err: any) {
      console.error("[tier1]", err?.message || err);
    }
  }

  // ── Tier 2: Intelligence — find and execute trades (5 min) ────────────────

  // Set by tier2Intelligence when the Python scan subprocess returns an error
  // or throws (e.g., "RuntimeError: can't start new thread"). The scheduler
  // reads this to apply an exponential back-off after repeated failures.
  let tier2LastScanFailed = false;
  // REPAIR 2026-07-06: the reason for the most recent failure, surfaced via
  // the token-gated /api/diag/scanner probe (never the public /api/health —
  // see that check's comment) once failures persist — a scan that fails
  // every cycle for hours previously left zero trace beyond a generic
  // "Could not fetch market data from Alpaca" repeated in the audit log
  // with no root cause. Cleared on any successful scan.
  let tier2LastFailureDetail: string | null = null;
  // REPAIR 2026-07-06 pt.2: deep_score's per-source enrichment-fetch errors
  // (macro/intel/alt/social/finnhub) from the most recent successful scan —
  // these degrade signal richness silently, not a scan failure, so they
  // ride alongside tier2LastFailureDetail on the same token-gated probe
  // rather than the pass/fail tier2LastScanFailed flag.
  let tier2LastDataSourceErrors: Record<string, string> = {};

  async function tier2Intelligence(isMarketOpen: boolean, etHour: number) {
    tier2LastScanFailed = false;
    tier2LastFailureDetail = null;
    audit("TIER2", "Starting intelligence scan...");

    // 1. Alpaca health check
    try {
      const pingRes = await alpaca("/v2/account");
      if (!pingRes.equity) {
        audit("TIER2", "Alpaca not responding — skipping cycle");
        state.alpacaFailCount = (state.alpacaFailCount || 0) + 1;
        return;
      }
      state.alpacaFailCount = 0;
    } catch (err: any) {
      state.alpacaFailCount = (state.alpacaFailCount || 0) + 1;
      audit("TIER2-FAIL", `Alpaca down (${state.alpacaFailCount} consecutive)`);
      return;
    }

    // 2. Diagnostic check (every 5th Tier 2 cycle)
    state.diagCycleCount++;
    if (state.diagCycleCount % 5 === 0) {
      try {
        const { stdout: diagOut } = await execPythonSerialized(`python3 -c "
from diagnostics import get_auto_fix_params
import json
print(json.dumps(get_auto_fix_params()))
"`, { timeout: 15000 });
        const diagParams = JSON.parse(diagOut.trim());
        state.positionSizeMultiplier = diagParams.position_size_multiplier || 1.0;
        state.minScoreThreshold = diagParams.min_score_threshold || 65;
        if (diagParams.problems_summary && diagParams.problems_summary !== "All systems healthy") {
          audit("DIAGNOSTIC", diagParams.problems_summary);
        }
        if (diagParams.should_pause) {
          audit("DIAGNOSTIC-PAUSE", "Critical issues — pausing");
          return;
        }
      } catch (err: any) {
        console.error("[tier2-diag]", err?.message || err);
      }
    }

    // 3. Run bot_engine scan (quick scan + deep analyze top 20)
    //    Priority: process streaming signals first (detected in real-time by Tier 0)
    const streamQueue: any[] = ((state as any).streamSignalQueue || []).filter(
      (s: any) => Date.now() - s.ts < 300_000  // Only signals < 5 min old
    );
    if (streamQueue.length > 0) {
      const tickers = streamQueue.map((s: any) => s.ticker).join(", ");
      audit("STREAM-PRIORITY", `${streamQueue.length} real-time signal(s) queued: ${tickers} — processing first`);
      (state as any).streamSignalQueue = [];  // Clear after reading
    }

    try {
      const enginePath = require("path").resolve(process.cwd(), "bot_engine.py");

      // DAEMON ROUTING 2026-04-22: Route through voltrade_daemon when alive
      // (daemon holds numpy/pandas/lightgbm/ml_model_v2 resident, so scan
      // runs in-process without the ~300 MB subprocess cold-start that was
      // getting SIGKILLed by Railway's cgroup OOM killer). Falls back to
      // spawning `python3 bot_engine.py full` if the daemon is down, disabled,
      // or returns a non-ok status.
      //
      // Daemon timeout: 90s (bot_engine.scan_market caps itself at 55s via
      // SIGALRM in main thread; in daemon worker thread it relies on this
      // outer bound). Subprocess timeout unchanged at 300s.
      // DAEMON-TIMEOUT-DIAG 2026-04-23: bump daemon timeout to 300s
      // temporarily so we can see full scan completion time in logs.
      // Previous 90s (or 60s if middleware overrode) was too tight.
      const callResult = await pythonCall(
        "run_full_scan",
        {},
        `python3 -W ignore "${enginePath}" full`,
        { timeout: 300000 },
        300000
      );

      if (!callResult.success) {
        tier2LastScanFailed = true;
        // If subprocess fallback ran and failed, rethrow its raw error object
        // so the existing catch block below can extract stderr/stdout/signal/
        // code and classify (SIGKILL/OOM/timeout/buffer) the same way it did
        // when scans went straight to execPythonSerialized. Otherwise surface
        // whatever error string we have.
        if (callResult.error) throw callResult.error;
        // DAEMON-TIMEOUT-VISIBILITY 2026-07-10: this branch covers
        // via==="daemon-failed"/"daemon-error" — a Unix-socket RPC outcome,
        // not a spawned subprocess. err.stderr/stdout/code/signal (what the
        // catch block below classifies on) are all meaningless here, which
        // is exactly why prior TIER2-ERROR audit lines for daemon timeouts
        // read as content-free "code=? signal=none" with a Node-process
        // memory snapshot that has nothing to do with the daemon. Tag the
        // error so the catch block can branch to a daemon health probe
        // instead.
        const e = new Error(callResult.result?.error || "scan call failed (no daemon, no subprocess)");
        (e as any).daemonFailure = true;
        throw e;
      }

      // Daemon path returns the parsed dict directly; subprocess path also
      // returns a parsed dict (pythonCall does JSON.parse on stdout).
      const result = callResult.result;

      if (!result || result.error) {
        tier2LastScanFailed = true;
        tier2LastFailureDetail = result?.debug_detail ? String(result.debug_detail).slice(0, 300) : null;
        const detailSuffix = tier2LastFailureDetail ? ` — ${tier2LastFailureDetail}` : "";
        audit("TIER2", `Scan returned error: ${result?.error || "unknown"}${detailSuffix} (via ${callResult.via})`);
        return;
      }

      audit("TIER2", `Scanned ${result.scanned || 0} stocks, ${(result.new_trades || []).length} trade candidates (via ${callResult.via})`);

      tier2LastDataSourceErrors = (result.data_source_errors && typeof result.data_source_errors === "object")
        ? result.data_source_errors : {};

      lastScanResult = result;

      // Auto-generate signals from scan results
      const topPicks = result.top_10 || [];
      for (const pick of topPicks) {
        if (pick.score >= 60) {
          const existing = signals.findIndex((s: any) => s.ticker === pick.ticker);
          if (existing >= 0) signals.splice(existing, 1);
          const side = pick.side || "buy";
          const actionLabel = pick.action_label || (pick.score >= 75 ? "STRONG BUY" : pick.score >= 65 ? "BUY" : "WATCH");
          // ALPHA AUDIT 2026-05-06 batch 11: pass through enriched fields so
          // the UI can show rank #1/#2/#3, per-factor breakdown, and sector
          // clash warnings. Lets the user verify the ranking is real, not
          // noise.
          signals.unshift({
            ticker: pick.ticker,
            action: actionLabel,
            reason: (pick.reasons || []).join(" | ") || `Score: ${pick.score}/100`,
            confidence: Math.min(95, pick.score),
            timestamp: new Date().toISOString(),
            type: side === "short" ? "sell" : side === "sell" ? "sell" : "buy",
            rank: pick.rank,
            factors: pick.factors,
            factor_strength: pick.factor_strength,
            sector_clash: pick.sector_clash,
            sector: pick.sector,
            tiebreak_score: pick.tiebreak_score,
          });
        }
      }
      // Add signals from new trades
      for (const trade of (result.new_trades || [])) {
        const existing = signals.findIndex((s: any) => s.ticker === trade.ticker);
        if (existing >= 0) signals.splice(existing, 1);
        const side = trade.side || "buy";
        signals.unshift({
          ticker: trade.ticker,
          action: `${trade.action || "BUY"} ${trade.shares} shares`,
          reason: (trade.reasons || []).join(" | ") || trade.rec_reasoning || `Score: ${trade.score}`,
          confidence: Math.min(95, trade.score),
          timestamp: new Date().toISOString(),
          type: side === "short" ? "sell" : "buy",
        });
      }
      // Add close signals
      for (const action of (result.position_actions || [])) {
        signals.unshift({
          ticker: action.ticker,
          action: "CLOSE",
          reason: action.reason,
          confidence: 90,
          timestamp: new Date().toISOString(),
          type: action.type === "take_profit" ? "sell" : "stop",
        });
      }

      // ITEM 20 FIX 2026-04-21: Surface tier-engine trades in signals panel
      // Previously, only score-based picks (>= 60) and stock trades flowed into
      // `signals`. Tier 1 CSP core, Tier 2 leveraged CSP, Tier 3 trend longs,
      // and Tier 4 tail hedges deployed silently — making the UI appear idle
      // even when the bot was actively trading. Now every tier action shows up.
      for (const action of (result.tier_actions || [])) {
        const existing = signals.findIndex((s: any) => s.ticker === action.ticker);
        if (existing >= 0) signals.splice(existing, 1);
        const tierLabel = action.tier ? `T${action.tier}` : "TIER";
        const actionLabel = (action.action || "DEPLOY").toUpperCase();
        signals.unshift({
          ticker: action.ticker,
          action: `${tierLabel}: ${actionLabel}`,
          reason: action.reason || action.strategy || `${tierLabel} tier engine action`,
          confidence: Math.min(95, Math.max(60, action.confidence || action.score || 75)),
          timestamp: new Date().toISOString(),
          type: action.side === "sell" || action.side === "short" || actionLabel.includes("SELL") || actionLabel.includes("SHORT") || actionLabel.includes("CSP") ? "sell" : "buy",
        });
      }

      if (signals.length > 30) signals.length = 30;
      // OOM fix: expire stale signals older than 1 hour
      const signalCutoff = Date.now() - 3600000;
      for (let i = signals.length - 1; i >= 0; i--) {
        if (new Date(signals[i].timestamp).getTime() < signalCutoff) signals.splice(i, 1);
      }

      // 4. Check daily/weekly limits
      const acct = await alpaca("/v2/account");
      const equity = parseFloat(acct.equity || "100000");
      // LIVE-CAPITAL CAP 2026-07-31: real uncommitted cash, threaded into the
      // Tier 1/2 SELL_CSP dispatch below so options_execution's CSP budget
      // check can cap itself against actual account capacity instead of a
      // pure equity percentage. See options_execution.py's cash_available
      // comment for the full evidence trail (repeated T2-FAIL "insufficient
      // options buying power" rejections on the same ticker every cycle).
      const cashAvailable = parseFloat(acct.cash || "0");
      const lastEquity = parseFloat(acct.last_equity || "100000");
      const dailyPnlPct = ((equity - lastEquity) / lastEquity) * 100;

      if (dailyPnlPct <= -3) {
        audit("TIER2-LIMIT", `Daily loss limit: ${dailyPnlPct.toFixed(1)}%`);
        notify("alert", `Daily loss: ${dailyPnlPct.toFixed(1)}%. Trading halted.`);
        return;
      }

      // Weekly loss check
      try {
        const { stdout: weeklyOut } = await execPythonSerialized(`python3 -c "
from diagnostics import check_weekly_loss
import json
history = [{'date': '${new Date().toISOString().split('T')[0]}', 'equity': ${equity}}]
print(json.dumps(check_weekly_loss(history)))
"`, { timeout: 5000 });
        const weeklyResult = JSON.parse(weeklyOut.trim());
        if (weeklyResult.action === "pause") {
          audit("WEEKLY-LIMIT", weeklyResult.reason);
          notify("alert", weeklyResult.reason);
          state.active = false;
          return;
        } else if (weeklyResult.action === "reduce") {
          audit("WEEKLY-WARNING", weeklyResult.reason);
          state.positionSizeMultiplier = Math.min(state.positionSizeMultiplier, 0.5);
        }
      } catch (err: any) { console.error("[tier2-weekly]", err?.message || err); }

      // 5. Execute trades or queue for morning
      // Cover intraday shorts at 3:50 PM (before 4 PM close)
      const etForShorts = getETHour();
      if (isMarketOpen && etForShorts >= 15.83) { // 3:50 PM ET
        try {
          const { stdout: shortsOut } = await execPythonSerialized(
            'python3 -c "from intraday_shorts import close_open_shorts; import json; print(json.dumps(close_open_shorts()))"',
            { timeout: 30000 }
          );
          const shortsResult = JSON.parse(shortsOut.trim());
          if (shortsResult.count > 0) {
            audit("SHORTS", `Covered ${shortsResult.count} intraday shorts at EOD`);
          }
        } catch (err: any) {
          audit("SHORTS-ERROR", `EOD cover failed: ${err?.message?.slice(0, 100)}`);
        }
      }

      if (isMarketOpen) {
        await executeTrades(result.new_trades || [], equity);
      } else {
        // Queue for morning
        for (const trade of (result.new_trades || []).slice(0, MAX_POSITIONS)) {
          if (!morningQueue.some((q: any) => q.ticker === trade.ticker)) {
            morningQueue.push({ ...trade, queuedAt: new Date().toISOString() });
            audit("QUEUE", `${trade.ticker} queued for market open (score ${trade.score})`);
          }
        }
      }

      // ──────────────────────────────────────────────────────────────────────
      // TIER DISPATCHER — routes tier_actions to the right execution path
      // ──────────────────────────────────────────────────────────────────────
      if (result.tier_actions && result.tier_actions.length > 0) {
        audit("TIERS", `${result.tier_actions.length} tier actions: ${JSON.stringify(result.tier_stats || {})}`);

        for (const action of result.tier_actions) {
          if (state.killSwitch) break;
          try {
            // ── TIER 1/2: SELL CSP ────────────────────────────────────────────
            // CSPs are options trades — use the Python options_execution path
            // via placeOptionsOrder. The Python side picks the actual contract
            // based on metadata (target_dte, target_delta).
            if (action.action === "SELL_CSP") {
              // Delegate to Python options_execution which handles strike/expiry
              // selection based on target_delta and target_dte_min/max in metadata
              const cspPayload = {
                ticker: action.ticker,
                strategy: "sell_cash_secured_put",  // matches options_execution.select_contract
                price: 0,  // Python will fetch current price
                equity: equity,
                cash: cashAvailable,
                size_pct: action.size_pct,
                metadata: action.metadata,
              };
              const cspTmpPath = `/tmp/tier_csp_${action.ticker}_${Date.now()}.json`;
              fs.writeFileSync(cspTmpPath, JSON.stringify(cspPayload));
              try {
                const { stdout } = await execPythonSerialized(
                  `python3 -c "
import json, sys; sys.path.insert(0, '.')
from options_execution import select_contract, submit_options_order
data = json.load(open('${cspTmpPath}'))
import os; os.remove('${cspTmpPath}')
contract = select_contract(data['ticker'], data['strategy'], data['price'], data['equity'], cash_available=data.get('cash'))
if contract.get('error'):
    print(json.dumps({'status': 'error', 'reason': contract['error']}))
else:
    result = submit_options_order(contract)
    print(json.dumps(result))
"`,
                  { timeout: 30000 }
                );
                const r = JSON.parse(stdout.trim());
                if (r.status === "submitted" || r.status === "filled") {
                  audit("T" + action.tier, `SELL_CSP ${action.ticker} | ${action.reason}`);
                } else {
                  audit("T" + action.tier + "-FAIL", `${action.ticker}: ${r.reason || r.detail || 'unknown'}`);
                }
              } catch (e: any) {
                audit("T" + action.tier + "-ERR", `${action.ticker}: ${e?.message?.slice(0,120)}`);
              }
            }

            // ── TIER 3: BUY SPY/QQQ at 2x ───────────────────────────────────────
            // Stock/ETF — use direct alpaca() call
            else if (action.action === "BUY") {
              const acctT3 = await alpaca("/v2/account");
              const buyingPower = parseFloat(acctT3.buying_power || "0");
              const targetDollars = equity * action.size_pct;

              // Fetch current price to compute shares
              const snap = await alpaca(`/v2/stocks/${action.ticker}/snapshot`).catch(() => null);
              const currentPrice = parseFloat(snap?.latestTrade?.p || snap?.dailyBar?.c || 0);
              if (currentPrice > 0 && targetDollars <= buyingPower) {
                const shares = Math.floor(targetDollars / currentPrice);
                if (shares > 0) {
                  const orderParams = getOrderParams(currentPrice);
                  try {
                    await alpaca("/v2/orders", {
                      method: "POST",
                      body: JSON.stringify({
                        symbol: action.ticker,
                        qty: String(shares),
                        side: "buy",
                        ...orderParams,
                      }),
                    });
                    audit("T3", `BUY ${shares} ${action.ticker} @ ~$${currentPrice.toFixed(2)} | ${action.reason}`);
                  } catch (e: any) {
                    audit("T3-FAIL", `${action.ticker}: ${e?.message?.slice(0,120)}`);
                  }
                }
              } else {
                audit("T3-SKIP", `${action.ticker}: insufficient BP or no price`);
              }
            }

            // ── TIER 4: BUY OTM SPY PUT (tail hedge) ───────────────────────────
            else if (action.action === "BUY_PUT") {
              // Tail hedge — same options execution path as CSP, different strategy
              const hedgePayload = {
                ticker: action.ticker,
                strategy: "buy_put",  // tail hedge = long OTM put; matches options_execution.select_contract
                price: 0,
                equity: equity,
                size_pct: action.size_pct,
                metadata: action.metadata,
              };
              const hedgeTmpPath = `/tmp/tier_hedge_${action.ticker}_${Date.now()}.json`;
              fs.writeFileSync(hedgeTmpPath, JSON.stringify(hedgePayload));
              try {
                const { stdout } = await execPythonSerialized(
                  `python3 -c "
import json, sys; sys.path.insert(0, '.')
from options_execution import select_contract, submit_options_order
data = json.load(open('${hedgeTmpPath}'))
import os; os.remove('${hedgeTmpPath}')
contract = select_contract(data['ticker'], data['strategy'], data['price'], data['equity'])
if contract.get('error'):
    print(json.dumps({'status': 'error', 'reason': contract['error']}))
else:
    result = submit_options_order(contract)
    print(json.dumps(result))
"`,
                  { timeout: 30000 }
                );
                const r = JSON.parse(stdout.trim());
                if (r.status === "submitted" || r.status === "filled") {
                  audit("T4", `BUY_PUT ${action.ticker} hedge | ${action.reason}`);
                } else {
                  audit("T4-FAIL", `${action.ticker}: ${r.reason || r.detail || 'unknown'}`);
                }
              } catch (e: any) {
                audit("T4-ERR", `${action.ticker}: ${e?.message?.slice(0,120)}`);
              }
            }
          } catch (e: any) {
            audit("TIER-DISPATCH-ERR", `T${action.tier} ${action.ticker}: ${e?.message?.slice(0,120)}`);
          }
        }
      }

      // Surface kill-switch warnings
      if (result.kill_status) {
        if (result.kill_status.killed) {
          audit("KILL-SWITCH", `FIRED: ${result.kill_status.kill_reason}`);
        }
        for (const warning of (result.kill_status.warnings || [])) {
          audit("KILL-WARN", warning);
        }
      }

      // Surface the tier engine's OWN internal kill switch (tiered_strategy.py
      // master_kill_switch — distinct from kill_status above). Without this,
      // tier_actions silently staying empty is indistinguishable from "no
      // eligible CSP candidates today" (see the tier_kill_status field added
      // in bot_engine.py's scan_market return, 2026-07-11).
      if (result.tier_kill_status?.killed) {
        audit("TIER-KILL", `Tier engine blocked: ${result.tier_kill_status.kill_reason}`);
      }

    } catch (err: any) {
      tier2LastScanFailed = true;
      console.error("[tier2-scan]", err?.message || err);

      if (err?.daemonFailure) {
        // DAEMON-TIMEOUT-VISIBILITY 2026-07-10: this failure came from the
        // Unix-socket RPC path (run_full_scan is HEAVY_DAEMON_ONLY, never
        // falls back to subprocess), so err.stderr/stdout/code/signal below
        // are all undefined and the classification logic they feed produces
        // the content-free "code=? signal=none" line seen recurring in the
        // audit log 2026-07-10 (18:22-19:57, 7x in ~90min) — plus a Node
        // process memory snapshot that describes the wrong process (bot.ts,
        // not the daemon). Probe the daemon's own health instead so the
        // audit trail can actually distinguish "daemon genuinely hung on a
        // slow scan" from "daemon piling up abandoned dispatch threads"
        // (voltrade_daemon.py's REQUEST_TIMEOUT_SEC join() abandons the
        // worker thread rather than killing it — active_dispatches surfaces
        // that pileup if it's happening). Short 5s timeout: if the daemon
        // is saturated enough to not even answer a trivial health call,
        // that itself is the finding.
        let daemonState = "";
        try {
          const h = await pythonRpc("health", {}, 5000);
          // BUGFIX 2026-07-11: pythonRpc's raw return is the RPC envelope
          // {status, result} (voltrade_daemon.py's dispatch() at "return
          // {status: ok, result: result}") — _health()'s own fields (uptime,
          // rss, the dispatch counter) live one level down, not on the
          // envelope itself. Reading the "alive" field off the top-level
          // envelope (the bug this replaces) was always undefined/falsy for
          // every successful health call, so this branch has logged
          // "non-alive: {full envelope}" for every single TIER2-ERROR
          // daemon-timeout entry since v1.0.266 shipped — the dispatch-count
          // evidence KNOWN BROKEN #18 needs to confirm or refute the
          // zombie-thread-pileup theory was never actually captured.
          const hr = h && h.status === "ok" ? h.result : null;
          // DAEMON-TIMEOUT-VISIBILITY 2026-07-20 (KNOWN BROKEN #18 continuation):
          // active_dispatches alone can't distinguish healthy 2x concurrency from
          // a self-perpetuating cascade — an abandoned run_full_scan zombie thread
          // from the PRIOR timed-out cycle still running and competing for the
          // shared alpaca_throttle bucket with a fresh run_full_scan that just
          // started would also read active_dispatches=2. voltrade_daemon.py now
          // tracks method name + elapsed time per active dispatch; surface it here
          // so the next TIER2-ERROR occurrence's audit line answers "what, and for
          // how long" directly instead of needing a live diag/health poll.
          const dispatchDetail = Array.isArray(hr?.active_dispatch_detail) && hr.active_dispatch_detail.length > 0
            ? ` [${hr.active_dispatch_detail.map((d: any) => `${d.method}:${d.elapsed_sec}s`).join(", ")}]`
            : "";
          // DAEMON-TIMEOUT-VISIBILITY 2026-07-21 (KNOWN BROKEN #18 continuation):
          // the item's own prior NEXT STEP asked for csp_layer2_prefetch's
          // cache_hit/budget_exceeded correlated with a live TIER2-ERROR, but
          // two sessions' live stakeouts (polling /api/diag/timings, which
          // only reflects a scan that already returned) both missed the
          // window. voltrade_daemon.py's health() now reads csp_universe's
          // module-level stats directly (live, even mid-hang), so the
          // correlation lands in the audit line itself the next time this
          // fires — no stakeout required.
          const l2 = hr?.layer2_prefetch;
          const layer2Detail = l2 && Object.keys(l2).length > 0
            ? ` layer2_prefetch={cache_hit=${l2.cache_hit} completed=${l2.completed}/${l2.total} elapsed=${l2.elapsed_sec}s budget_exceeded=${l2.budget_exceeded} age=${l2.age_sec}s}`
            : "";
          // DAEMON-TIMEOUT-VISIBILITY 2026-07-27 (KNOWN BROKEN #18 continuation):
          // every tier_engine_start-stuck occurrence checked this session (11/11
          // since v1.0.503's on_phase instrumentation shipped) read layer2_prefetch
          // cache_hit=true (Layer 2 fast) — Layer 2 was never the explanation for
          // THESE occurrences. Layer 1 (csp_universe._layer1_hard_gates) has its
          // own independent 15-min cache and its cache-miss path (full-universe
          // live snapshot fetch + account-equity fetch) has never been visible
          // here — surface it the same way so the next occurrence can show
          // whether Layer 1, not Layer 2, is what's still running.
          const l1 = hr?.layer1_stats;
          const layer1Detail = l1 && Object.keys(l1).length > 0
            ? ` layer1_stats={cache_hit=${l1.cache_hit} universe_size=${l1.universe_size} elapsed=${l1.elapsed_sec}s age=${l1.age_sec}s}`
            : "";
          // DAEMON-TIMEOUT-VISIBILITY 2026-07-28 (KNOWN BROKEN #18 continuation):
          // live occurrences today show step6_trade_loop_and_options (the options
          // scanner) as the preamble's slowest phase at 233-240s — RECURRING
          // despite v1.0.503's fix for this exact phase. Root cause found this
          // session: Setup 7 (options_scanner.py) calls vol_surface.get_surface_score()
          // for every high_iv/anchor candidate, and THAT function's own spot/chain/
          // bars fetches are not gated by alpaca_throttle — a cost none of this
          // item's prior sessions' models accounted for. Surfacing it the same way
          // so the next occurrence's audit line shows the call count and cumulative
          // elapsed time directly.
          const ss = hr?.surface_score_stats;
          const surfaceDetail = ss && Object.keys(ss).length > 0
            ? ` surface_score_stats={calls=${ss.calls} cache_misses=${ss.cache_misses} cumulative_elapsed=${ss.cumulative_elapsed_sec}s skipped_budget=${ss.skipped_budget ?? 0} age=${ss.age_sec}s}`
            : "";
          // DAEMON-TIMEOUT-VISIBILITY 2026-07-22 (KNOWN BROKEN #18 continuation):
          // layer2_prefetch only answers "was it Layer 2" — the storm's third
          // named mechanism (deep_score's ThreadPoolExecutor shutdown hazard,
          // v1.0.468) shipped and the storm continued unchanged through today's
          // full session (13 fresh TIER2-ERROR occurrences 13:37-19:55Z, same
          // active_dispatches=2/300s signature, all with high layer2_prefetch
          // age — Layer 2 stays refuted). Rather than guess a fourth specific
          // mechanism, read bot_engine.py's own TIMING-DISK file (2026-04-23;
          // already used by /api/diag/timings and /api/system/snapshot) HERE,
          // at the instant the timeout is caught — it is written progressively,
          // phase-by-phase, straight to shared disk (not returned via RPC), so
          // if the stuck run_full_scan dispatch is still mid-flight this read
          // lands on ITS OWN last-completed phase and elapsed time, generalizing
          // the "which phase" question to the whole pipeline instead of only
          // Layer 2 — closing the exact gap the 2026-07-21 session hit trying
          // to live-poll /api/diag/timings from a separate stakeout window
          // (which missed every occurrence because a fresh scan can overwrite
          // the file before a human-timed poll lands). No live stakeout
          // required going forward; the next occurrence's audit line carries it.
          let scanTimingsDetail = "";
          try {
            const timingPaths = ["/data/voltrade/voltrade_scan_timings.json", "/tmp/voltrade_scan_timings.json"];
            for (const tpath of timingPaths) {
              if (fs.existsSync(tpath)) {
                const st = JSON.parse(fs.readFileSync(tpath, "utf8"));
                const nowSec = Date.now() / 1000;
                const lastPhase = st.last_phase_completed ?? (Array.isArray(st.phases) && st.phases.length > 0 ? st.phases[st.phases.length - 1]?.phase : null) ?? "none";
                const anchor = st.last_phase_at ?? st.scan_started;
                const ageSec = typeof anchor === "number" ? Math.round((nowSec - anchor) * 10) / 10 : null;
                // DAEMON-TIMEOUT-VISIBILITY 2026-07-27 #2 (KNOWN BROKEN #18
                // continuation): last_phase+age only says WHERE the kill
                // landed, not why it took 300s to get there. Today's busy-hours
                // occurrences all show last_phase=tier_engine_start with a
                // small age (~20-40s) — i.e. tier1 itself only ran briefly
                // before the outer daemon timeout fired, so the PREAMBLE
                // (universe_load..step10b_defensive_floor) must already have
                // consumed the other ~260-280s. st.phases is written
                // progressively (survives the kill) and already holds each
                // preamble phase's own duration_sec — surface the preamble
                // total and its single slowest phase directly here instead of
                // requiring a future session to catch a live in-progress scan
                // (the last two sessions' stakeouts both missed the window).
                let preambleDetail = "";
                const phases = Array.isArray(st.phases)
                  ? st.phases.filter((p: any) => typeof p?.duration_sec === "number")
                  : [];
                if (phases.length > 0) {
                  const preamblePhases = phases.filter((p: any) => !String(p.phase).startsWith("tier_engine"));
                  const preambleTotal = Math.round(
                    preamblePhases.reduce((sum: number, p: any) => sum + p.duration_sec, 0) * 100
                  ) / 100;
                  const slowest = phases.reduce(
                    (max: any, p: any) => (p.duration_sec > (max?.duration_sec ?? -1) ? p : max),
                    null
                  );
                  preambleDetail = ` preamble={total=${preambleTotal}s slowest=${slowest?.phase}:${slowest?.duration_sec}s}`;
                }
                scanTimingsDetail = ` scan_timings={status=${st.status ?? "?"} last_phase=${lastPhase} age=${ageSec ?? "?"}s}${preambleDetail}`;
                break;
              }
            }
          } catch (timingErr: any) {
            scanTimingsDetail = ` scan_timings={read_error=${String(timingErr?.message || timingErr).slice(0, 80)}}`;
          }
          daemonState = hr && hr.alive
            ? `daemon rss=${hr.rss_mb}MB active_dispatches=${hr.active_dispatches ?? "?"}${dispatchDetail} uptime=${hr.uptime_seconds}s${layer2Detail}${layer1Detail}${surfaceDetail}${scanTimingsDetail}`
            : `daemon health returned non-alive: ${JSON.stringify(h).slice(0, 150)}`;
        } catch (healthErr: any) {
          daemonState = `daemon health probe itself failed: ${String(healthErr?.message || healthErr).slice(0, 120)}`;
        }
        const msg = String(err?.message || err);
        tier2LastFailureDetail = `${msg} — ${daemonState}`.slice(0, 300);
        audit("TIER2-ERROR", `Scan failed via daemon: ${msg} — ${daemonState}`);
      } else {
        // Gather everything useful: stderr, stdout tail (in case Python wrote
        // the error there), exit code, kill signal, and current memory. Without
        // these, OOM kills and buffer overruns both show as "Command failed"
        // with no context.
        const stderr = String(err?.stderr || "");
        const stdout = String(err?.stdout || "");
        // Node's child_process.exec overloads err.code: numeric exit status for
        // non-zero exits, string error name for spawn/buffer failures (e.g.
        // ERR_CHILD_PROCESS_STDIO_MAXBUFFER), and null when the child was killed
        // by a signal. Normalize so the activity log never prints "code=null"
        // or "code=undefined", which look like bugs to operators.
        const rawCode = err?.code;
        const code: string | number =
          rawCode === undefined || rawCode === null || rawCode === "" ? "?" : rawCode;
        const signal = err?.signal || "none";
        const msg = String(err?.message || err);

        // Prefer stderr (where Python tracebacks live), fall back to stdout tail
        const primary = stderr.trim() ? stderr : stdout;
        const tail = primary.length > 800 ? '…' + primary.slice(-800) : primary;

        // Memory snapshot — helps correlate OOM kills
        const mem = process.memoryUsage();
        const memStr = `rss=${Math.round(mem.rss/1024/1024)}MB heap=${Math.round(mem.heapUsed/1024/1024)}/${Math.round(mem.heapTotal/1024/1024)}MB`;

        // Classify: SIGKILL+empty-stderr is almost always OOM or buffer overflow.
        // Compare against rawCode (not the "?" placeholder) so classification
        // still fires when the exec error arrives as a string code.
        let classification = "";
        if (signal === "SIGKILL" && !stderr.trim()) {
          classification = " [likely OOM kill or maxBuffer exceeded]";
        } else if (rawCode === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER") {
          classification = " [stdout buffer exceeded — raise DEFAULT_MAX_BUFFER]";
        } else if (signal === "SIGTERM") {
          classification = " [timed out or killed externally]";
        } else if (rawCode === "ETIMEDOUT") {
          classification = " [exec timeout exceeded]";
        } else if (rawCode === "ENOENT") {
          classification = " [python3 not found on PATH]";
        }

        const detail = tail.trim() || msg;
        tier2LastFailureDetail = `code=${code} signal=${signal}${classification}: ${detail}`.slice(0, 300);
        audit("TIER2-ERROR", `Scan failed (code=${code} signal=${signal} ${memStr})${classification}: ${detail}`);
      }
    }

    // Overnight/pre-market research: runs during 8pm-4am ET window
    const _etHourNow = getETHour();
    if (!isMarketOpen && (_etHourNow >= 20 || _etHourNow < 4)) {
      try {
        await runOvernightResearch(_etHourNow);
      } catch (researchErr: any) {
        audit("RESEARCH-ERROR", `Overnight research failed: ${String(researchErr?.message || researchErr).slice(0, 200)}`);
      }
    }
  }

  // ── Tier 2 trade execution helper ─────────────────────────────────────────

  async function executeTrades(trades: any[], equity: number) {
    const positions = await alpaca("/v2/positions").catch(() => []);
    const held = Array.isArray(positions) ? positions.map((p: any) => p.symbol) : [];
    // Count stock and options positions separately
    // Exclude floor + third-leg positions from active trading heat check
    // (FLOOR_AND_LEG_TICKERS, module scope).
    const stockPositions = Array.isArray(positions)
      ? positions.filter((p: any) => (p.asset_class || "us_equity") === "us_equity").length
      : held.length;
    const optionsPositions = Array.isArray(positions)
      ? positions.filter((p: any) => {
          if (p.asset_class !== "us_option") return false;
          // Don't count convexity overlay puts against scanner slots
          const sym = p.symbol || "";
          if (sym.startsWith("QQQ") && sym.includes("P") && sym.length > 10) return false;
          return true;
        }).length
      : 0;
    // BUG FIX 2026-07-29: was hardcoded to 3, a stale local constant that
    // predates system_config.py's dedicated MAX_OPTIONS_POSITIONS key
    // (added 2026-07-11, KNOWN BROKEN #3 fix) which is explicitly
    // documented there as "Max total open OPTIONS (CSP) positions" and
    // "the single source of truth" at value 6, held constant across every
    // regime. This scanner-path slot check counts the SAME thing
    // (total open us_option positions) as tiered_strategy.py's
    // tier1_csp_core(), which already generates up to 6 CSP candidates —
    // so a stricter local cap of 3 here silently blocked legitimate
    // Tier-2-scanner options trades once the tier engine (or this path
    // itself) had filled 3 of the 6 intended slots. Live evidence
    // 2026-07-29: OPTIONS-SLOT-FULL fired for VXUS/KWEB at "4/3" while
    // the account correctly held 4 real CSP positions (within the 6-slot
    // budget) — the scanner path was starving itself against a ceiling
    // nothing else in the system agrees with. Mechanical fix restoring
    // the documented single-source-of-truth value; not a new threshold
    // policy (same class as the 2026-07-11 fix), no RULE REVIEW gate.
    const MAX_OPTIONS_POSITIONS = 6;  // Separate slots for options — mirrors system_config.py's MAX_OPTIONS_POSITIONS
    let slotsUsed = stockPositions;  // Only count stocks against MAX_POSITIONS
    let optionsSlotsUsed = optionsPositions;
    let totalDeployed = Array.isArray(positions)
      ? positions.reduce((sum: number, p: any) => {
          if (FLOOR_AND_LEG_TICKERS.has(p.symbol)) return sum; // Don't count floor/leg positions
          return sum + Math.abs(parseFloat(p.market_value || "0"));
        }, 0)
      : 0;
    const pendingOrderIds: Array<{orderId: string; trade: any; side: string; qty: number}> = [];

    for (const trade of trades) {
      if (state.killSwitch) break;
      const isOptionsTrade = trade.trade_type === "options" || (trade.use_options && trade.options_strategy !== "stock");
      // Options have separate slot limit from stocks
      if (isOptionsTrade) {
        if (optionsSlotsUsed >= MAX_OPTIONS_POSITIONS) {
          audit("OPTIONS-SLOT-FULL", `${trade.ticker}: options slots full (${optionsSlotsUsed}/${MAX_OPTIONS_POSITIONS})`);
          continue;  // Skip this options trade, but continue checking remaining trades
        }
      } else {
        if (slotsUsed >= MAX_POSITIONS) break;  // Stock slots full
      }
      if (trade.score < state.minScoreThreshold) {
        audit("SKIP", `${trade.ticker}: score ${trade.score} < threshold ${state.minScoreThreshold}`);
        continue;
      }
      if (held.includes(trade.ticker)) continue; // Already holding
      if ((trade as any)._consumed) continue;
      // Skip tickers blocked after 3 same-ticker stops today
      if (Array.isArray((state as any).dailyBlockedTickers) && (state as any).dailyBlockedTickers.includes(trade.ticker)) {
        audit("SKIP", `${trade.ticker}: blocked for today after repeated stop-losses`);
        continue;
      }

      // Dynamic sizing: trust position_sizing.py output but enforce absolute ceiling
      const effectiveCeiling = MAX_POSITION_SIZE * state.positionSizeMultiplier;
      if (trade.position_value > equity * effectiveCeiling) {
        audit("SIZE-CAP", `${trade.ticker}: $${trade.position_value} exceeds ceiling ${(effectiveCeiling * 100).toFixed(1)}%`);
        continue;
      }

      // Portfolio heat check — total deployed capital
      if (totalDeployed + trade.position_value > equity * MAX_TOTAL_EXPOSURE) {
        audit("HEAT-CAP", `${trade.ticker}: total exposure would exceed ${(MAX_TOTAL_EXPOSURE * 100)}%`);
        break;
      }

      // ══════════════════════════════════════════════════════════════
      // ITEM 14 FIX 2026-04-20: Pre-trade correlation check
      // Before adding a new position, check if it would over-concentrate
      // the book by sector. Prevents tech sell-offs from killing the book
      // when 10+ positions are all tech. Uses risk_kill_switch.check_correlation_pre_trade.
      // ══════════════════════════════════════════════════════════════
      try {
        const corrCall = await pythonCall(
          "check_correlation_pre_trade",
          { new_ticker: trade.ticker, positions: Array.isArray(positions) ? positions : [], max_sector_pct: 0.40 },
          `python3 -c "import json, sys; sys.path.insert(0, '.'); from risk_kill_switch import check_correlation_pre_trade; pos = ${JSON.stringify(Array.isArray(positions) ? positions : [])}; r = check_correlation_pre_trade('${trade.ticker}', pos, 0.40); print(json.dumps(r))"`,
          { timeout: 5000 }
        );
        if (corrCall.success && corrCall.result && corrCall.result.allowed === false) {
          audit("CORR-CAP", `${trade.ticker}: ${corrCall.result.reason}`);
          continue;  // Skip this trade — sector over-concentrated
        }
      } catch (corrErr: any) {
        // Fail-open — don't block trades on our bug
      }

      // Exchange halt check (via position_sizing.py)
      // OPT 2026-04-20: daemon first, subprocess fallback
      try {
        const haltCall = await pythonCall(
          "check_halt", { ticker: trade.ticker },
          `python3 -c "from position_sizing import check_halt_status; import json; print(json.dumps(check_halt_status('${trade.ticker}')))"`,
          { timeout: 8000 }
        );
        const haltResult = haltCall.success ? haltCall.result : { halted: false };
        if (haltResult.halted) {
          audit("HALT-SKIP", `${trade.ticker}: stock is halted (status: ${haltResult.status})`);
          continue;
        }
      } catch (err: any) {
        audit("HALT-CHECK-WARN", `${trade.ticker}: halt check failed, proceeding`);
      }

      // Max retry check — stop re-submitting orders for tickers that keep failing
      // (Fix 2026-04-10: ELAB had 10 canceled orders from this retry loop)
      const priorAttempts = orderAttemptCounts[trade.ticker] || 0;
      if (priorAttempts >= MAX_ORDER_ATTEMPTS_PER_TICKER) {
        audit("MAX-RETRY", `${trade.ticker}: skipped — already attempted ${priorAttempts} times today (max ${MAX_ORDER_ATTEMPTS_PER_TICKER})`);
        continue;
      }

      // Duplicate order check — skip if we already have a pending order for this ticker
      try {
        const openOrders = await alpaca("/v2/orders?status=open");
        if (Array.isArray(openOrders) && openOrders.some((o: any) => o.symbol === trade.ticker)) {
          audit("DUP-SKIP", `${trade.ticker}: already have a pending order`);
          continue;
        }
      } catch (err: any) { /* proceed if check fails */ }

      // Buying power check
      const acctCheck = await alpaca("/v2/account");
      const bp = parseFloat(acctCheck.buying_power || "0");
      if (trade.position_value > bp) {
        const replaced = await replaceIfBetter(trade);
        if (!replaced) continue;
        await new Promise(r => setTimeout(r, 1000));
      }

      // Track order attempt count per ticker (Fix 2026-04-10)
      orderAttemptCounts[trade.ticker] = (orderAttemptCounts[trade.ticker] || 0) + 1;

      try {
        // ── INSTRUMENT SCORE DEBUG LOG ──
        const iScores = (trade as any).instrument_scores || {};
        const sScore  = iScores.stock?.score   ?? "N/A";
        const eScore  = iScores.etf?.score     ?? "N/A";
        const oScore  = iScores.options?.score ?? "N/A";
        audit("INSTRUMENT", `${trade.ticker} → ${(trade.instrument || 'stock').toUpperCase()} chosen | scores: stock=${sScore} etf=${eScore} options=${oScore} | ${(trade.instrument_reasoning || '').slice(0, 120)}`);

        // ── ETF execution (instrument selector chose 2x leveraged ETF) ──
        if (trade.instrument === "etf" && trade.instrument_ticker && trade.instrument_ticker !== trade.ticker) {
          const etfTicker = trade.instrument_ticker;
          const etfShares = Math.max(1, Math.floor(trade.shares / 2)); // Half shares for 2x leverage
          try {
            const etfOrderParams = getOrderParams(trade.price || 0);
            await alpaca("/v2/orders", {
              method: "POST",
              body: JSON.stringify({
                symbol: etfTicker, qty: String(etfShares), side: "buy",  // Always buy (short blocked)
                ...etfOrderParams,
              }),
            });
            audit("ETF-TRADE", `${trade.side?.toUpperCase() || 'BUY'} ${etfShares} ${etfTicker} (2x ETF for ${trade.ticker}) | Score: ${trade.score} | ${trade.instrument_reasoning?.slice(0, 120)}`);
            notify("trade", `ETF: ${etfShares} ${etfTicker} (2x ${trade.ticker})`);
            slotsUsed++;
            totalDeployed += etfShares * trade.price; // Approximate
            // REPAIR 2026-07-31 (KNOWN BROKEN #12(c) contributor): this branch
            // monitored the position for WS exit but never wrote a track_fill
            // entry record, so its eventual exit always found no match and
            // became a permanent orphan_exit — see entryFill.ts for the full
            // trace. Record the entry the same way the regular/morning-queue
            // stock paths already do.
            try {
              const etfEntryPayload = buildEntryFillPayload({
                ticker: etfTicker, side: "buy", qty: etfShares,
                fillPrice: trade.price || 0, session: "regular",
                volume: trade.volume, score: trade.score,
                instrument: "etf", codeVersion: pkgVersion,
              });
              const etfTmp = `/tmp/fill_etf_${etfTicker}_${Date.now()}.json`;
              fs.writeFileSync(etfTmp, JSON.stringify(etfEntryPayload));
              pythonCall(
                "track_fill", etfEntryPayload,
                `python3 -c "import json, os; from ml_model_v2 import track_fill; d=json.load(open('${etfTmp}')); os.remove('${etfTmp}'); track_fill(d)"`,
                { timeout: 5000 }
              ).then((r) => {
                if (r.via === "daemon") {
                  try { fs.unlinkSync(etfTmp); } catch {}
                }
              }).catch(() => {});
            } catch (err: any) { console.error("[bot]", err?.message || err); }
            addPositionToMonitor(etfTicker, trade.side === "short" ? "short" : "long", trade.price || 0, etfShares);
            continue;
          } catch (etfErr: any) {
            audit("ETF-FALLBACK", `${etfTicker} order failed: ${etfErr?.message?.slice(0, 100)} — falling back to stock`);
            // Fall through to stock/options execution
          }
        }

        // ── Options vs Stock execution ──
        if (trade.use_options && trade.options_strategy !== "stock") {
          try {
            let optExec: any;

            if (trade.trade_type === "options" && trade.regime_at_entry === "OPTIONS_SCANNER") {
              // ── SCANNER PATH: Direct to select_contract + submit ──
              // Scanner already found the setup. Bypass should_use_options() which would
              // re-evaluate with incomplete data (no vrp/rsi) and potentially reject.
              const scannerPayload = {
                ticker: trade.ticker, strategy: trade.options_strategy || trade.instrument_strategy,
                price: trade.price, equity, setup: trade.setup || "",
              };
              const scanTmpPath = `/tmp/opt_scan_${trade.ticker}_${Date.now()}.json`;
              fs.writeFileSync(scanTmpPath, JSON.stringify(scannerPayload));
              const { stdout: scanResult } = await execPythonSerialized(
                `python3 -c "
import json, sys; sys.path.insert(0, '.')
from options_execution import select_contract, submit_options_order
data = json.load(open('${scanTmpPath}'))
contract = select_contract(data['ticker'], data['strategy'], data['price'], data['equity'])
if contract.get('error'):
    print(json.dumps({'instrument':'stock','order':None,'contract':None,'reasoning':contract['error']}))
else:
    order = submit_options_order(contract)
    if order.get('status') in ('submitted','filled','pending_new','accepted'):
        # P0-9 FIX: previous call passed delta as the 4th positional arg,
        # but register_options_entry's signature is
        # (occ_symbol, entry_price, side, strategy, delta=0, qty=1, ticker, setup, max_loss).
        # So strategy was never registered, every position defaulted to
        # strategy='' which broke grouping/profit-target/max-loss logic.
        # Use kwargs + iterate ALL leg keys so multi-leg states get saved.
        try:
            from options_manager import register_options_entry
            _strategy = data.get('strategy','')
            _ticker = data.get('ticker','')
            _setup = data.get('setup','')
            _max_loss = float(contract.get('max_loss', 0) or 0)
            _leg_keys = ['occ_symbol','short_call','long_call','short_put','long_put','call_leg','put_leg']
            _registered = set()
            for _lk in _leg_keys:
                _occ = contract.get(_lk,'')
                if not _occ or _occ in _registered:
                    continue
                _registered.add(_occ)
                # Each leg carries its own side / delta / qty if present,
                # falling back to top-level contract fields.
                _leg_side = contract.get(_lk + '_side') or contract.get('side','buy')
                _leg_delta = contract.get(_lk + '_delta')
                if _leg_delta is None: _leg_delta = contract.get('delta',0)
                _leg_qty = contract.get(_lk + '_qty') or contract.get('qty',1)
                _leg_price = contract.get(_lk + '_price') or contract.get('limit_price',0)
                register_options_entry(
                    occ_symbol=_occ,
                    entry_price=float(_leg_price or 0),
                    side=_leg_side,
                    strategy=_strategy,
                    delta=float(_leg_delta or 0),
                    qty=int(_leg_qty or 1),
                    ticker=_ticker,
                    setup=_setup,
                    max_loss=_max_loss,
                )
        except Exception as _e: pass
        # v1.0.33: Save options state for dashboard display (setup, strategy, expiry)
        try:
            import os
            try:
                from storage_config import DATA_DIR
            except ImportError:
                DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
            state_path = os.path.join(DATA_DIR, 'voltrade_options_state.json')
            try:
                with open(state_path) as _f: ostate = json.load(_f)
            except: ostate = {}
            # P0-9 FIX: iterate all possible multi-leg keys, not just call/put.
            for key in [contract.get('occ_symbol',''),
                        contract.get('short_call',''), contract.get('long_call',''),
                        contract.get('short_put',''),  contract.get('long_put',''),
                        contract.get('call_leg',''),   contract.get('put_leg','')]:
                if key:
                    ostate[key] = {'strategy': data.get('strategy',''), 'setup': data.get('setup',''), 'ticker': data.get('ticker','')}
            ostate[data.get('ticker','')] = {'strategy': data.get('strategy',''), 'setup': data.get('setup',''), 'options_strategy': data.get('strategy','')}
            with open(state_path, 'w') as _f: json.dump(ostate, _f)
        except: pass
    print(json.dumps({'instrument':'options','strategy':data['strategy'],'contract':contract,'order':order,'reasoning':'Scanner direct execution'}))
"`,
                { timeout: 30000 }
              );
              try { fs.unlinkSync(scanTmpPath); } catch (_) {}
              optExec = JSON.parse(scanResult.trim());
              audit("OPTIONS-SCANNER-EXEC", `${trade.ticker} | strategy=${trade.options_strategy} | setup=${trade.setup || '?'} | result=${optExec.order?.status || 'no_order'}`);

            } else {
              // ── STOCK→OPTIONS PATH: Full evaluate_and_execute pipeline ──
              const optionsPayload = {
                trade: { ticker: trade.ticker, price: trade.price, deep_score: trade.score, vrp: trade.vrp || 0, side: trade.side || "buy", action_label: trade.action || "BUY", rsi: trade.rsi || 50, ewma_rv: trade.ewma_rv || 2, garch_rv: trade.garch_rv || 2 },
                equity, positions: Array.isArray(positions) ? positions : [],
              };
              const tmpPath = `/tmp/opt_${trade.ticker}_${Date.now()}.json`;
              fs.writeFileSync(tmpPath, JSON.stringify(optionsPayload));
              // OPT 2026-04-20 (Bug #25 continuation): daemon first, subprocess fallback.
              // evaluate_and_execute is called once per options-candidate trade — hot path.
              const evalCall = await pythonCall(
                "evaluate_and_execute",
                { trade: optionsPayload.trade, equity: optionsPayload.equity, positions: optionsPayload.positions },
                `python3 -c "import json; from options_execution import evaluate_and_execute; d=json.load(open('${tmpPath}')); print(json.dumps(evaluate_and_execute(d['trade'], d['equity'], d['positions'])))"`,
                { timeout: 30000 }
              );
              try { fs.unlinkSync(tmpPath); } catch (_) {}
              if (evalCall.success) {
                optExec = evalCall.result;
              } else {
                optExec = { instrument: "stock", order: null, reasoning: "evaluate_and_execute failed" };
              }
            }

            if (optExec.instrument === "options" && ["submitted", "filled", "pending_new", "accepted"].includes(optExec.order?.status)) {
              audit("OPTIONS-TRADE", `${(trade.options_strategy || '').toUpperCase()} ${trade.ticker} | ${optExec.contract?.occ_symbol || ''} | ${(optExec.reasoning || '').slice(0, 120)}`);
              notify("trade", `OPTIONS: ${trade.options_strategy} on ${trade.ticker} (edge: ${(trade.options_edge_pct || 0).toFixed?.(1) || '?'}%)`);
              optionsSlotsUsed++;
              totalDeployed += optExec.contract?.max_cost || trade.position_value;
              // Options positions use the OCC symbol but monitor via underlying ticker
              addPositionToMonitor(trade.ticker, trade.side === "short" ? "short" : "long", trade.price || 0, trade.shares || 1);
              continue; // Skip stock execution
            } else if (optExec.instrument === "options" && optExec.order?.status === "error") {
              audit("OPTIONS-FALLBACK", `${trade.ticker}: ${(optExec.order.detail || '').slice(0, 150)} — falling back to stock`);
              // Fall through to stock execution
            } else {
              // Options engine said stock is better or scanner contract selection failed
              audit("OPTIONS-SKIP", `${trade.ticker}: ${(optExec.reasoning || '').slice(0, 100)}`);
              if (trade.trade_type === "options") continue; // Pure options trade — no stock fallback
            }
          } catch (optErr: any) {
            audit("OPTIONS-ERROR", `${trade.ticker}: ${(optErr?.message || '').slice(0, 100)} — falling back to stock`);
            if (trade.trade_type === "options") continue; // Pure options trade — no stock fallback
          }
        }

        // ── Stock execution (default or fallback from options) ──
        const qty = Math.floor(trade.shares);
        if (qty <= 0) continue;

        // ══ HARD BLOCK: No stock shorting (PR #52 + PR #53) ══════════════
        // Stock shorts destroyed value (-$419K backtest). All bearish plays
        // route through options. This is the final safety net — if ANY code
        // path produces side="sell" or side="short" for a stock, block it.
        if (trade.side === "short" || trade.side === "sell") {
          audit("SHORT-BLOCKED", `${trade.ticker}: side='${trade.side}' blocked — stock shorting disabled. Converting to BUY.`);
          trade.side = "buy";
          trade.action_label = "BUY";
        }
        const side = trade.side || "buy";

        const orderParams = getOrderParams(trade.price || 0);
        const orderResult = await alpaca("/v2/orders", {
          method: "POST",
          body: JSON.stringify({
            symbol: trade.ticker, qty: String(qty), side, ...orderParams,
          }),
        });
        const orderId = orderResult?.id || null;

        // Log with sizing details
        const sizingInfo = trade.sizing_reasoning ? ` | Sizing: ${trade.sizing_reasoning.slice(0, 150)}` : "";
        const scoreAttrib = trade.rules_score != null ? ` | Rules: ${trade.rules_score} ML: ${trade.ml_score_raw ?? 'N/A'} Blend: ${trade.score}` : "";
        audit("TRADE", `${orderParams.type.toUpperCase()} ${side.toUpperCase()} ${qty} ${trade.ticker} @ ~$${trade.price} | Score: ${trade.score}${scoreAttrib} | $${trade.position_value} (${((trade.position_value / equity) * 100).toFixed(1)}%)${sizingInfo}`);
        notify("trade", `${side.toUpperCase()} ${qty} ${trade.ticker} @ $${trade.price} (${((trade.position_value / equity) * 100).toFixed(1)}% of portfolio)`);
        slotsUsed++;
        totalDeployed += trade.position_value;
        addPositionToMonitor(trade.ticker, "long", trade.price || 0, qty);  // Always long (short blocked)

        // Collect order for batch confirmation
        if (orderId) {
          pendingOrderIds.push({ orderId, trade, side, qty });
        }

      } catch (err: any) {
        audit("TRADE-ERROR", `${trade.ticker}: ${err?.message}`);
      }

      await new Promise(r => setTimeout(r, 500));
    }

    // ── Batch fill confirmation (one 2s wait, then check all orders) ──
    if (pendingOrderIds.length > 0) {
      await new Promise(r => setTimeout(r, 2000)); // Single 2-second wait for all fills

      for (const pending of pendingOrderIds) {
        try {
          const orderStatus = await alpaca(`/v2/orders/${pending.orderId}`);
          const filledQty = parseInt(orderStatus?.filled_qty || "0");
          const requestedQty = pending.qty;
          const filledAvgPrice = parseFloat(orderStatus?.filled_avg_price || String(pending.trade.price));
          const status = orderStatus?.status || "unknown";
          const isPartial = filledQty > 0 && filledQty < requestedQty;

          if (isPartial) {
            audit("PARTIAL-FILL", `${pending.trade.ticker}: filled ${filledQty}/${requestedQty} shares @ $${filledAvgPrice} (${status})`);
          } else if (status === "rejected" || status === "canceled") {
            audit("ORDER-REJECTED", `${pending.trade.ticker}: order ${status} — ${orderStatus?.reject_reason || 'unknown reason'}`);
          }

          // Track fill with ACTUAL data (not assumed)
          const vol = pending.trade.volume || 1000000;
          let slippagePct = 0.03;
          if (vol > 20000000) slippagePct = 0.02 + Math.random() * 0.03;
          else if (vol > 5000000) slippagePct = 0.05 + Math.random() * 0.05;
          else if (vol > 1000000) slippagePct = 0.08 + Math.random() * 0.07;
          else slippagePct = 0.12 + Math.random() * 0.13;

          const slippageDirection = (pending.side === "buy") ? 1 : -1;
          const realisticFillPrice = Math.round((filledAvgPrice * (1 + slippageDirection * slippagePct / 100)) * 100) / 100;

          const fillPayload = {
            ticker: pending.trade.ticker, order_type: "market", side: pending.side,
            qty_requested: requestedQty,
            qty_filled: filledQty || requestedQty,
            partial_fill: isPartial,
            expected_price: pending.trade.price,
            alpaca_fill_price: filledAvgPrice,
            fill_price: realisticFillPrice,
            slippage_applied_pct: slippagePct,
            order_status: status,
            time_placed: new Date().toISOString(),
            session: "regular", volume: vol, score: pending.trade.score,
            instrument: pending.trade.instrument || "stock",
            entry_features: pending.trade.entry_features || null,
            code_version: pkgVersion,
          };
          const rfTmp = `/tmp/fill_r_${pending.trade.ticker}_${Date.now()}.json`;
          fs.writeFileSync(rfTmp, JSON.stringify(fillPayload));
          // OPT 2026-04-20: daemon first, subprocess fallback
          pythonCall(
            "track_fill", fillPayload,
            `python3 -c "import json, os; from ml_model_v2 import track_fill; d=json.load(open('${rfTmp}')); os.remove('${rfTmp}'); track_fill(d)"`,
            { timeout: 5000 }
          ).then((r) => {
            // Clean up temp file if daemon handled it (subprocess removes its own)
            if (r.via === "daemon") {
              try { fs.unlinkSync(rfTmp); } catch {}
            }
          }).catch(() => {});
        } catch (cfErr: any) {
          // Confirmation failed — record with best-guess data
          audit("FILL-CHECK-WARN", `${pending.trade.ticker}: could not confirm fill — recording expected values`);
        }
      }
    }
  }

  // ── Tier 3: Strategic — ML retrain, macro, manipulation scan (1h) ─────────

  async function tier3Strategic() {
    audit("TIER3", "Starting strategic scan...");

    // 1. ML model retrain (if needed)
    try {
      const { stdout: modelCheck } = await execPythonSerialized(`python3 -c "
import os, time, json
from storage_config import ML_MODEL_PATH as path
if not os.path.exists(path) or (time.time() - os.path.getmtime(path)) > 86400:
    print(json.dumps({'needs_retrain': True}))
else:
    age_h = (time.time() - os.path.getmtime(path)) / 3600
    print(json.dumps({'needs_retrain': False, 'age_hours': round(age_h, 1)}))
"`, { timeout: 5000 });
      const modelStatus = JSON.parse(modelCheck.trim());
      if (modelStatus.needs_retrain) {
        audit("TIER3", "ML model stale or missing — triggering retrain...");
        try {
          // batch 8 REVISED 2026-05-04: bumped timeout 300s → 900s.
          // Production SIGKILL events were classified as "likely OOM"
          // by the heuristic below, but with 8GB available and only
          // ~280MB used by the daemon, OOM is unlikely. Far more likely
          // cause is THIS timeout firing on the heavy retrain (200
          // tickers × 6 years of bars + 4 regime-conditional models +
          // LightGBM training). Node.js sends SIGTERM then SIGKILL on
          // exec timeout — looks identical to an OS OOM kill from
          // outside the process. 15 minutes gives the retrain real
          // headroom; if it still times out we'll get the new
          // steps_completed diagnostic from ml_retrain_safe to tell us
          // which step is actually slow.
          const { stdout: trainOut } = await execPythonSerialized("python3 ml_retrain_safe.py", { timeout: 900000 });
          const trainResult = JSON.parse(trainOut.trim());
          // ALPHA AUDIT 2026-05-07 batch 12: when the retrain returns a
          // status that starts with "error:", append error_location and
          // traceback tail so we can actually diagnose the failure. The
          // Python wrapper at ml_model_v2._train_model_impl already
          // captures these fields — they were just being dropped on the
          // floor at the audit boundary. Production logs were stuck
          // showing "error: 'NoneType' object is not subscriptable"
          // every hour with no file/line.
          const _statusStr = String(trainResult.status || "");
          if (_statusStr.startsWith("error:") || _statusStr.startsWith("failed")) {
            // KNOWN BROKEN #17 (open_questions.md, found 2026-07-09): several
            // _train_model_impl early returns use the plain shape
            // {"status": "failed", "reason": "..."} (no error_location/
            // traceback_tail — those only ever come from train_model()'s outer
            // exception wrapper) which collapsed this line to the content-free
            // "ML retrain failed: failed". `reason` carries the real cause
            // (e.g. "Could not fetch training bars") and was being dropped.
            const _reason = trainResult.reason ? ` — ${trainResult.reason}` : "";
            const _loc = trainResult.error_location ? ` @ ${trainResult.error_location}` : "";
            const _tbTail = trainResult.traceback_tail ? ` | tb: ${String(trainResult.traceback_tail).slice(-200).replace(/\n/g, " ")}` : "";
            audit("TIER3-ML-ERROR", `ML retrain failed: ${_statusStr}${_reason}${_loc}${_tbTail}`);
          } else {
            audit("TIER3", `ML retrain complete — status: ${trainResult.status}, accuracy: ${trainResult.accuracy || 'N/A'}, features: ${trainResult.feature_count || 'N/A'}, samples: ${trainResult.samples || trainResult.sample_count || 'N/A'}`);
          }
        } catch (trainErr: any) {
          // DIAGNOSTIC FIX 2026-04-22: previously logged only err.message
          // which produced useless entries like "Command failed: python3
          // ml_retrain_safe.py" with no signal/stderr/stdout context — so
          // OOM kills, import failures, and silent Python crashes all
          // looked identical. Mirror the Tier 2 scan classifier: grab
          // stderr (Python traceback), stdout tail (ml_retrain_safe.py
          // writes its JSON error payload here on caught exceptions),
          // exit code and kill signal.
          const _stderr = String(trainErr?.stderr || "");
          const _stdout = String(trainErr?.stdout || "");
          const _rawCode = trainErr?.code;
          const _code: string | number =
            _rawCode === undefined || _rawCode === null || _rawCode === "" ? "?" : _rawCode;
          const _signal = trainErr?.signal || "none";
          // Prefer stderr (Python traceback), fall back to stdout
          // (ml_retrain_safe prints structured JSON errors there).
          const _primary = _stderr.trim() ? _stderr : _stdout;
          const _tail = _primary.length > 500 ? "…" + _primary.slice(-500) : _primary;
          let _cls = "";
          if (_signal === "SIGKILL" && !_stderr.trim()) {
            _cls = " [likely OOM kill during import lightgbm/ml_model_v2]";
          } else if (_signal === "SIGTERM") {
            _cls = " [timed out or killed externally]";
          } else if (_rawCode === "ETIMEDOUT") {
            _cls = " [exec timeout exceeded]";
          }
          const _detail = _tail.trim() || String(trainErr?.message || trainErr);
          audit("TIER3-ML-ERROR", `ML retrain failed (code=${_code} signal=${_signal})${_cls}: ${_detail}`);
        }
      } else {
        audit("TIER3", `ML model fresh (${modelStatus.age_hours}h old) — skipping retrain`);
      }
    } catch (err: any) { console.error("[tier3-ml]", err?.message || err); }

    // 2. Manipulation detection scan
    try {
      const { stdout: manipOut } = await execPythonSerialized(`python3 -c "
from manipulation_detect import scan_for_manipulation
import json
print(json.dumps(scan_for_manipulation()))
"`, { timeout: 30000 });
      const manipResult = JSON.parse(manipOut.trim());
      if (manipResult.alerts && manipResult.alerts.length > 0) {
        for (const alert of manipResult.alerts.slice(0, 5)) {
          audit('MANIPULATION', alert.message || JSON.stringify(alert));
        }
      }
    } catch (err: any) {
      // R18 (KNOWN BROKEN #14, PR #327 follow-up): this catch previously
      // only console.error'd, leaving zero trail in the persisted audit
      // log whenever the scan failed — invisible from outside the
      // container. Mirror the ML-retrain catch block's audit-with-cause
      // pattern (same stdout-corruption failure class) so a future
      // manipulation-scan failure of any kind surfaces in the audit log.
      const _stderr = String(err?.stderr || "");
      const _stdout = String(err?.stdout || "");
      const _rawCode = err?.code;
      const _code: string | number =
        _rawCode === undefined || _rawCode === null || _rawCode === "" ? "?" : _rawCode;
      const _signal = err?.signal || "none";
      const _primary = _stderr.trim() ? _stderr : _stdout;
      const _tail = _primary.length > 500 ? "…" + _primary.slice(-500) : _primary;
      const _detail = _tail.trim() || String(err?.message || err);
      audit("TIER3-MANIP-ERROR", `Manipulation scan failed (code=${_code} signal=${_signal}): ${_detail}`);
      console.error("[tier3-manip]", err?.message || err);
    }

    // 3. Run full diagnostics
    try {
      const { stdout: diagFull } = await execPythonSerialized(`python3 -c "
from diagnostics import run_diagnostics
import json
print(json.dumps(run_diagnostics()))
"`, { timeout: 15000 });
      const diagReport = JSON.parse(diagFull.trim());
      if (diagReport.overall_status !== "healthy") {
        // R-DIAG (KNOWN BROKEN follow-up, research/experiments.md 2026-07-05/
        // 2026-07-06): this audit line used to report only the problem COUNT,
        // never WHICH system or WHAT the message was — "1 issues" recurred in
        // the audit log for a month with no session able to tell what it
        // actually was without re-running diagnostics.py by hand. diagReport
        // already carries system/severity/message per problem; just log it.
        const detail = (diagReport.problems || [])
          .map((p: any) => `[${String(p.severity || "?").toUpperCase()}] ${p.system || "?"}: ${p.message || ""}`)
          .join(" | ");
        audit("TIER3-DIAG", `System health: ${diagReport.overall_status} — ${diagReport.problems?.length || 0} issues: ${detail}`);
      }
    } catch (err: any) { console.error("[tier3-diag]", err?.message || err); }

    // 4. Record daily equity
    recordDailyEquity();

    // 5. CFTC Commitments of Traders archive update (EDGE DOCTRINE #1
    // free-data pipeline). COT only publishes weekly — cftc_cot.py
    // self-guards to hit the network at most once per ~20h, so this
    // hourly call is a cheap file-mtime check on 23 of 24 invocations.
    // DATA-LADDER GATE 1 ONLY: archives validated positioning, feeds
    // nothing into scoring yet (see research/open_questions.md).
    try {
      const { stdout: cotOut } = await execPythonSerialized(`python3 -c "
from cftc_cot import run_daily_update
import json
print(json.dumps(run_daily_update()))
"`, { timeout: 45000 });
      const cotResult = JSON.parse(cotOut.trim());
      if (cotResult.status === "done") {
        const errored = Object.entries(cotResult.symbols || {}).filter(([, v]: [string, any]) => v.status === "error");
        if (errored.length > 0) {
          audit("TIER3-COT", `COT update: ${errored.length} symbol(s) failed — ${errored.map(([s]) => s).join(", ")}`);
        }
      }
    } catch (err: any) { console.error("[tier3-cot]", err?.message || err); }

    // 6. SEC bulk-historical Form 3/4/5 insider-transaction archive update
    // (EDGE DOCTRINE #1 free-data pipeline; DATA-LADDER GATE 1 ONLY — see
    // sec_form4_bulk.py / research/open_questions.md's "Insider Form 4
    // clustering as a signal"). Quarterly-cadence dataset, so this hourly
    // call self-guards to hit the network at most once per ~12h and
    // downloads at most one quarter per invocation during backfill.
    try {
      const { stdout: form4Out } = await execPythonSerialized(`python3 -c "
from sec_form4_bulk import run_update
import json
print(json.dumps(run_update()))
"`, { timeout: 120000 });
      const form4Result = JSON.parse(form4Out.trim());
      if (form4Result.status === "error") {
        audit("TIER3-FORM4BULK", `Form4 bulk update failed: ${form4Result.reason}`);
      } else {
        const errored = Object.entries(form4Result.results || {}).filter(([, v]: [string, any]) => v.status === "error");
        if (errored.length > 0) {
          audit("TIER3-FORM4BULK", `Form4 bulk update: ${errored.length} quarter(s) failed — ${errored.map(([q]) => q).join(", ")}`);
        }
      }
    } catch (err: any) { console.error("[tier3-form4bulk]", err?.message || err); }

    audit("TIER3", "Strategic scan complete");
  }

  // ── Upgrade logic: sell weak positions if much better picks available ──────
  // (kept for compatibility with runOvernightResearch and manual calls)
  async function runUpgradeCandidates(upgradeCandidates: any[], newTrades: any[]) {
    if (upgradeCandidates.length === 0 || newTrades.length === 0) return;

    upgradeCandidates.sort((a: any, b: any) => (a.score || 0) - (b.score || 0));
    const sortedNew = [...newTrades].sort((a: any, b: any) => (b.score || 0) - (a.score || 0));

    for (const candidate of upgradeCandidates) {
      if (state.killSwitch) break;
      const betterPick = sortedNew.find((t: any) =>
        t.score >= (candidate.score || 50) + 20 &&
        !upgradeCandidates.some((uc: any) => uc.ticker === t.ticker)
      );
      if (betterPick) {
        try {
          await alpaca(`/v2/positions/${candidate.ticker}`, { method: "DELETE" });
          removePositionFromMonitor(candidate.ticker);
          audit("UPGRADE", `Sold ${candidate.ticker} → upgrading to ${betterPick.ticker} (score ${betterPick.score})`);
          notify("upgrade", `Upgraded: sold ${candidate.ticker} → buying ${betterPick.ticker}`);
          betterPick._consumed = true;
          await new Promise(r => setTimeout(r, 1000));
          const upgradeQty = Math.floor(candidate.market_value / betterPick.price);
          if (upgradeQty > 0) {
            await alpaca("/v2/orders", {
              method: "POST",
              body: JSON.stringify({ symbol: betterPick.ticker, qty: String(upgradeQty), side: "buy", ...getOrderParams(betterPick.price || 0) }),  // Always buy (short blocked)
            });
            addPositionToMonitor(betterPick.ticker, "long", betterPick.price || 0, upgradeQty);  // Always long (short blocked)
          }
        } catch (e: any) { audit("UPGRADE-ERROR", `Failed to upgrade: ${e.message}`); }
      }
    }
  }

  // ── Overnight Research Engine (8pm-4am ET) ───────────────────────────────
  let lastResearchRun = 0;

  async function runOvernightResearch(etHour: number) {
    // Only run once per hour to save resources
    if (Date.now() - lastResearchRun < 55 * 60 * 1000) return;
    lastResearchRun = Date.now();
    autoRunning = true;

    try {
      // 8pm-10pm: Scan market, find tomorrow's opportunities + analyze today's extreme movers
      if (etHour >= 20 && etHour < 22) {
        audit("RESEARCH", "Evening scan — analyzing today's movers for tomorrow's plays");
        const enginePath3 = path.resolve("bot_engine.py");
        const { stdout: scanOut3 } = await execPythonSerialized(`python3 -W ignore "${enginePath3}" full 2>/dev/null`, { timeout: 300000 });
        const jsonStart3 = scanOut3.indexOf('{');
        if (jsonStart3 !== -1) {
          lastScanResult = JSON.parse(scanOut3.slice(jsonStart3));
          audit("RESEARCH", `Scanned ${lastScanResult.scanned || 0} stocks — top picks identified for morning`);
        }

        // Process extreme movers from today — analyze for tomorrow's entry
        try {
          const { stdout: emOut } = await execPythonSerialized(`python3 -c "
import json, os, time
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'

em_path = os.path.join(DATA_DIR, 'extreme_movers_today.json')
if not os.path.exists(em_path):
    print(json.dumps({'movers': [], 'analyzed': 0}))
else:
    with open(em_path) as f: movers = json.load(f)
    # Only process from today
    today = time.strftime('%Y-%m-%d')
    today_movers = [m for m in movers if m.get('timestamp', '').startswith(today)]
    print(json.dumps({'movers': today_movers, 'analyzed': len(today_movers)}))
"`, { timeout: 10000 });
          const emResult = JSON.parse(emOut.trim());
          const todayMovers = emResult.movers || [];

          if (todayMovers.length > 0) {
            audit("RESEARCH", `${todayMovers.length} extreme movers from today — running overnight deep analysis for tomorrow's setups`);

            for (const mover of todayMovers.slice(0, 5)) { // Max 5 extreme movers analyzed
              try {
                // Deep analyze each extreme mover for tomorrow's setup
                const { stdout: deepOut } = await execPythonSerialized(
                  `python3 -W ignore -c "
from analyze import quick_scan
import json
result = quick_scan('${mover.ticker}')
print(json.dumps(result))
" 2>/dev/null`, { timeout: 60000 }
                );
                const analysis = JSON.parse(deepOut.trim());

                const vrp = analysis.vrp || 0;
                const changeToday = mover.change_pct || 0;

                // Determine tomorrow's setup
                let tomorrowSetup = '';
                let tomorrowDirection = '';
                let tomorrowScore = 0;

                if (vrp > 8) {
                  // IV still elevated → sell premium (put credit spread or cash secured put)
                  tomorrowSetup = `High VRP ${vrp.toFixed(1)}% — sell puts tomorrow if IV stays elevated`;
                  tomorrowDirection = 'SELL OPTIONS';
                  tomorrowScore = 72;
                } else if (vrp < -3) {
                  // IV crushed after spike → buy puts cheaply for mean reversion
                  tomorrowSetup = `IV crushed — cheap puts for mean reversion play`;
                  tomorrowDirection = 'BUY PUT';
                  tomorrowScore = 68;
                } else if (changeToday > 100 && (analysis.recommendation === 'BUY' || analysis.momentum_score > 70)) {
                  // Rare continuation setup
                  tomorrowSetup = `Momentum continuation — ran ${changeToday.toFixed(0)}%, still has strength`;
                  tomorrowDirection = 'BUY';
                  tomorrowScore = 65;
                } else {
                  // No clear setup tomorrow — skip
                  audit("RESEARCH", `${mover.ticker}: no clear tomorrow setup (VRP ${vrp.toFixed(1)}%, change ${changeToday.toFixed(0)}%) — skipping`);
                  continue;
                }

                audit("RESEARCH", `${mover.ticker}: ${tomorrowSetup} → queuing for tomorrow`);

                // Queue for tomorrow morning
                const tomorrowTrade = {
                  ticker: mover.ticker,
                  price: mover.price,
                  score: tomorrowScore,
                  side: tomorrowDirection === 'BUY' ? 'buy' : 'sell',
                  action: tomorrowDirection,
                  action_label: tomorrowDirection,
                  shares: Math.max(1, Math.floor(5000 / (mover.price || 10))),
                  position_value: 5000,
                  reasons: [tomorrowSetup],
                  setup_reason: `Extreme mover (+${changeToday.toFixed(0)}% yesterday) — ${tomorrowSetup}`,
                  vrp: vrp,
                  use_options: tomorrowDirection !== 'BUY',
                  options_strategy: tomorrowDirection === 'SELL OPTIONS' ? 'sell_cash_secured_put' : 'buy_put',
                  queuedAt: new Date().toISOString(),
                };

                if (!morningQueue.some((q: any) => q.ticker === mover.ticker)) {
                  morningQueue.push(tomorrowTrade as any);
                  audit("QUEUE", `${mover.ticker} queued for tomorrow: ${tomorrowSetup}`);
                }

              } catch (analyzeErr: any) {
                console.error(`[overnight-extreme-mover] ${mover.ticker}:`, analyzeErr?.message);
              }
            }

            // Clear extreme movers file after processing
            await execPythonSerialized(`python3 -c "
import os, json
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
em_path = os.path.join(DATA_DIR, 'extreme_movers_today.json')
if os.path.exists(em_path): os.remove(em_path)
print('cleared')
"`, { timeout: 5000 }).catch(() => {});
          } else {
            audit("RESEARCH", "No extreme movers from today to analyze");
          }
        } catch (emErr: any) {
          console.error("[overnight-extreme-movers]", emErr?.message);
        }
      }

      // 10pm-12am: Run backtests on current strategies to validate they still work
      if (etHour >= 22 || etHour < 0) {
        audit("RESEARCH", "Running backtest validation on active strategies...");
        const btPath = path.resolve("backtest.py");
        try {
          const { stdout: btOut } = await execPythonSerialized(
            `python3 "${btPath}" SPY all 2`, { timeout: 120000 }
          );
          const btResult = JSON.parse(btOut.trim());
          const results = btResult.results || [];
          for (const r of results) {
            const status = (r.sharpe || 0) >= 1.0 ? "PASSING" : (r.sharpe || 0) >= 0.5 ? "MARGINAL" : "FAILING";
            audit("RESEARCH", `Backtest ${r.strategy}: Sharpe ${r.sharpe}, Return ${r.totalReturn}%, Drawdown ${r.maxDrawdown}% — ${status}`);
          }
        } catch (err: any) { console.error("[bot]", err?.message || err); }
      }

      // 12am-2am: Deep analyze earnings reporters for tomorrow
      if (etHour >= 0 && etHour < 2) {
        audit("RESEARCH", "Analyzing upcoming earnings and overnight news...");
        const earningsTickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"];
        for (const t of earningsTickers) {
          try {
            const { stdout: aOut } = await execPythonSerialized(
              `python3 "${path.resolve('analyze.py')}" "${t}" --mode=scan`, { timeout: 30000 }
            );
            const analysis = JSON.parse(aOut.trim());
            const ei = analysis.earnings_intel;
            if (ei && ei.days_to_earnings >= 0 && ei.days_to_earnings <= 5) {
              audit("RESEARCH", `EARNINGS ALERT: ${t} reports in ${ei.days_to_earnings} days (${ei.timing}). Beat rate: ${ei.beat_pct}%. IV crush opportunity: ${ei.options_edge > 0 ? 'YES' : 'NO'}`);
              notify("earnings", `${t} earnings in ${ei.days_to_earnings} days — IV crush opportunity: ${ei.options_edge > 0 ? 'YES' : 'NO'}`);
            }
          } catch (err: any) { console.error("[bot]", err?.message || err); }
        }
      }

      // 2am-4am: Prepare pre-market report
      if (etHour >= 2 && etHour < 4) {
        audit("RESEARCH", "Compiling pre-market report...");
        // Re-scan to get freshest data
        const enginePath4 = path.resolve("bot_engine.py");
        try {
          const { stdout: scanOut } = await execPythonSerialized(`python3 "${enginePath4}" scan`, { timeout: 180000 });
          lastScanResult = JSON.parse(scanOut.trim());
          const topTrades = lastScanResult.new_trades || [];
          if (topTrades.length > 0) {
            audit("RESEARCH", `PRE-MARKET REPORT: ${topTrades.length} trade opportunities ready. Top pick: ${topTrades[0].ticker} (score ${topTrades[0].score})`);
            notify("research", `Pre-market report ready: ${topTrades.length} opportunities. Top: ${topTrades[0].ticker} @ $${topTrades[0].price}`);
          } else {
            audit("RESEARCH", "PRE-MARKET REPORT: No high-conviction trades found. Bot will monitor during pre-market.");
          }
        } catch (err: any) { console.error("[bot]", err?.message || err); }
      }

      // 4am ET: Daily ML retrain + reset daily state
      if (Math.floor(etHour) === 4) {
        // Clear daily blocked tickers, stop counters, and order attempt counts for new trading day
        (state as any).dailyBlockedTickers = [];
        state.consecutiveStopLosses = 0;
        (state as any).recentStopTickers = [];
        (state as any).lastStopTimes = {};
        state.morningQueueExecuted = false;
        // OOM fix: reassign instead of for...delete (V8 keeps hidden class slots with delete)
        orderAttemptCounts = {};
        audit("SYSTEM", "Daily reset: blocked tickers cleared, counters reset for new trading day");
        audit("RETRAIN", "4am daily ML retrain — training on yesterday's data before market open");
        try {
          // batch 8 REVISED 2026-05-04: bumped timeout 300s → 900s.
          // Same as the TIER3 retrain call earlier — see comment there.
          const { stdout: trainOut } = await execPythonSerialized(
            `python3 ml_retrain_safe.py`,
            { timeout: 900000 }
          );
          const trainResult = JSON.parse(trainOut.trim());
          audit("RETRAIN", `Daily retrain complete — status: ${trainResult.status}, accuracy: ${trainResult.accuracy || 'N/A'}, features: ${trainResult.feature_count || 'N/A'}, samples: ${trainResult.samples || trainResult.sample_count || 'N/A'}`);
        } catch (err: any) {
          audit("RETRAIN-ERROR", `Daily retrain failed: ${err?.message || err}`);
        }

        // Nightly auto-backup at 4am alongside retrain
        try {
          const { stdout: backupOut } = await execPythonSerialized(`python3 -c "
import json, os, shutil, time
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = '/tmp'
backup_dir = os.path.join(DATA_DIR, 'backups')
os.makedirs(backup_dir, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
backup_path = os.path.join(backup_dir, f'backup_{timestamp}')
os.makedirs(backup_path, exist_ok=True)
count = 0
for fname in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.isfile(fpath) and (fname.endswith('.json') or fname.endswith('.pkl') or fname.endswith('.db')):
        shutil.copy2(fpath, os.path.join(backup_path, fname))
        count += 1
all_b = sorted([d for d in os.listdir(backup_dir) if d.startswith('backup_')])
while len(all_b) > 5:
    shutil.rmtree(os.path.join(backup_dir, all_b.pop(0)), ignore_errors=True)
print(json.dumps({'files': count}))
"`, { timeout: 30000 });
          const bk = JSON.parse(backupOut.trim());
          audit("BACKUP", `Nightly auto-backup: ${bk.files} files saved (local)`);

          // Off-site backup: push data snapshot to GitHub data-backup branch
          try {
            const backupScript = path.resolve("backup_to_github.py");
            const { stdout: gitBackup } = await execPythonSerialized(
              `python3 "${backupScript}"`, { timeout: 60000 }
            );
            const gbResult = JSON.parse(gitBackup.trim());
            if (gbResult.status === "success") {
              audit("BACKUP", `Off-site: ${gbResult.count} files pushed to GitHub (data-backup branch)`);
            } else if (gbResult.status === "no_changes") {
              audit("BACKUP", `Off-site: data unchanged since last backup`);
            } else if (gbResult.status === "skipped") {
              audit("BACKUP-WARN", `Off-site skipped: ${gbResult.reason}`);
            } else {
              audit("BACKUP-ERROR", `Off-site failed: ${gbResult.reason || 'unknown'}`);
            }
          } catch (err: any) {
            audit("BACKUP-WARN", `Off-site backup failed: ${err?.message || err}`);
          }
        } catch (err: any) {
          audit("BACKUP-ERROR", `Nightly backup failed: ${err?.message || err}`);
        }
      }

    } catch (e: any) {
      audit("RESEARCH-ERROR", `Overnight research failed: ${e.message}`);
    }

    autoRunning = false;
  }

  // ── TIER 0: Real-Time Streaming ──────────────────────────────────────────────
  // WebSocket connection to Alpaca SIP feed — receives 1-minute bar updates the
  // instant each bar closes. Detects volume spikes and breakouts in real-time,
  // BEFORE they appear on the polling-based most-actives list (which lags 2-15 min).

  const STREAM_TICKERS = [
    "NVDA","AMD","META","MSFT","AAPL","GOOGL","AMZN","AVGO","TSLA","PLTR",
    "COIN","MSTR","HOOD","SOFI","AFRM","UPST","SQ","CRWD","NET","DDOG",
    "MDB","SNOW","PANW","ZS","ARM","SMCI","SHOP","PINS","SNAP","NFLX",
    "JPM","GS","V","MA","UNH","LLY","WMT","COST","XOM","CVX",
    "SPY","QQQ","IWM",
  ];

  const streamVolHistory: Record<string, number[]>   = {};
  const streamPriceHistory: Record<string, number[]> = {};
  const streamLastSignal: Record<string, number>     = {};
  let   streamWs: WebSocket | null = null;
  let   streamConnected = false;

  // ── WebSocket-Driven Position Monitor ──────────────────────────────────────
  // Tracks open positions with their stop/target levels. When a price tick
  // arrives for an owned stock, immediately evaluate exits — no waiting for
  // the scanner cycle.
  interface MonitoredPosition {
    ticker: string;
    side: 'long' | 'short';
    entryPrice: number;
    qty: number;
    phase: number;
    initialRiskPct: number;
    highestPnl: number;
    currentStopPct: number;
    tpPct: number;
    entryDate: string;
    lastCheckedPrice: number;
    // Scale-out tracking
    originalQty: number;       // Original position size at entry
    remainingQty: number;      // Shares still held (decreases with each scale-out)
    scalesCompleted: number;   // 0, 1, or 2 (of 3 thirds)
    breakevenActive: boolean;  // True after first scale-out — stop can't go below entry
    // Regime-aware exits
    regime: string;            // Current market regime (BULL, NEUTRAL, CAUTION, BEAR, PANIC)
    // ATR-based trailing (Phase 4 fix)
    atrPct: number;            // Current ATR as % of price
    highestPrice: number;      // Highest price reached (for ATR trailing from price, not P&L)
    pendingExit?: boolean;     // True when exit order placed but not yet filled
    // Options position tracking
    positionType?: 'stock' | 'credit_spread' | 'iron_condor' | 'covered_call' | 'csp';
    creditReceived?: number;    // For credit strategies
    maxLoss?: number;           // For defined-risk strategies
    legSymbols?: string[];      // OCC symbols of all legs
    strategyLabel?: string;     // Display label
  }

  const monitoredPositions: Record<string, MonitoredPosition> = {};
  let positionMonitorInitialized = false;
  // Tracks tickers we've subscribed to for position monitoring (not in STREAM_TICKERS)
  const positionSubscribedTickers: Set<string> = new Set();

  /**
   * Load positions + stop state from Alpaca + disk, build the monitoredPositions map.
   * Called on startup and periodically to stay in sync.
   */
  async function syncMonitoredPositions() {
    try {
      const positions = await alpaca("/v2/positions");
      if (!Array.isArray(positions)) return;

      // Load stop state from disk (same file bot_engine writes)
      let stopState: Record<string, any> = {};
      try {
        const { stdout } = await execPythonSerialized(`python3 -c "
import json, os
DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
p = os.path.join(DATA_DIR, 'voltrade_stop_state.json')
try:
    with open(p) as f: print(json.dumps(json.load(f)))
except: print('{}')
"`, { timeout: 5000 });
        stopState = JSON.parse(stdout.trim());
      } catch (_) {}

      const activeTickers = new Set<string>();

      for (const pos of positions) {
        const ticker = pos.symbol || "";
        if (FLOOR_AND_LEG_TICKERS.has(ticker)) continue;

        activeTickers.add(ticker);
        const current = parseFloat(pos.current_price || "0");
        const entry = parseFloat(pos.avg_entry_price || String(current));
        const pnlPct = parseFloat(pos.unrealized_plpc || "0") * 100;
        const qty = Math.abs(parseInt(pos.qty || "0"));
        const side = pos.side === "short" ? "short" as const : "long" as const;

        // ══════════════════════════════════════════════════════════════
        // TIME-EXIT 2026-04-22: close positions older than 14 days with
        // pnl in [-5%, +2%]. Stale + not working = opportunity cost.
        // Skips ETF floor holdings.
        // ══════════════════════════════════════════════════════════════
        try {
          // Positions in FLOOR_AND_LEG_TICKERS already `continue`d above and
          // never reach this loop body — LEGACY_DEFENSIVE_CANDIDATE_TICKERS
          // additionally exempts defensive-floor candidates (VTI/IWM/TLT/
          // IEF/SCHP) that were evaluated and rejected in favor of GLD
          // (system_config.py DEFENSIVE_FLOOR_TICKER) but are kept here in
          // case one is ever held again.
          if (!LEGACY_DEFENSIVE_CANDIDATE_TICKERS.has(ticker)) {
            const entryTime = pos.created_at || pos.submitted_at;
            if (entryTime) {
              const ageDays = (Date.now() - new Date(entryTime).getTime()) / (1000 * 60 * 60 * 24);
              if (ageDays > 14 && pnlPct >= -5.0 && pnlPct <= 2.0) {
                audit("TIME-EXIT", `${ticker}: ${Math.round(ageDays)}d old, pnl ${pnlPct.toFixed(1)}% — closing stale`);
                try {
                  const closeSide = side === "long" ? "sell" : "buy";
                  await alpaca("/v2/orders", {
                    method: "POST",
                    body: JSON.stringify({
                      symbol: ticker, qty: String(qty), side: closeSide,
                      type: "market", time_in_force: "day",
                    }),
                  });
                  continue;
                } catch (e: any) {
                  audit("TIME-EXIT-ERROR", `${ticker}: ${e?.message?.slice(0, 80)}`);
                }
              }
            }
          }
        } catch { /* non-critical */ }

        // ══════════════════════════════════════════════════════════════
        // ITEM 13 FIX 2026-04-20: Per-position risk kill
        // If this position has lost more than 25%, liquidate immediately.
        // Catches runaway single-name losses faster than portfolio DD kill.
        // ══════════════════════════════════════════════════════════════
        const POSITION_KILL_LOSS_PCT = -25.0;  // -25% per position
        const POSITION_WARN_LOSS_PCT = -15.0;  // -15% warn threshold
        if (pnlPct <= POSITION_KILL_LOSS_PCT) {
          audit("POS-KILL", `${ticker}: position down ${pnlPct.toFixed(1)}% — forcing liquidation (threshold: ${POSITION_KILL_LOSS_PCT}%)`);
          try {
            const closeSide = side === "long" ? "sell" : "buy";
            const orderParams = getOrderParams(current, 'stop_loss');
            await alpaca("/v2/orders", {
              method: "POST",
              body: JSON.stringify({
                symbol: ticker, qty: String(qty), side: closeSide,
                ...orderParams,
              }),
            });
            notify("alert", `POSITION KILL: ${ticker} at ${pnlPct.toFixed(1)}% — liquidated`);
            // REPAIR 2026-07-06 (R12-D2): a forced liquidation is a full
            // exit — record the outcome so the ML loop learns from the
            // worst trades too, not only the graceful ones.
            recordExitFill(buildExitFillPayload({
              ticker, exitSide: closeSide, qty: Number(qty) || 0,
              fillPrice: current, pnlPct, exitReason: "position_kill",
              codeVersion: pkgVersion,
            }));
            continue; // Skip the rest of the loop for this position
          } catch (killErr: any) {
            audit("POS-KILL-ERROR", `${ticker}: failed to liquidate — ${killErr?.message?.slice(0, 120)}`);
          }
        } else if (pnlPct <= POSITION_WARN_LOSS_PCT) {
          // Don't spam — only audit the first warn per position per hour
          const warnKey = `${ticker}_warn`;
          const lastWarn = (state as any)._lastPositionWarn?.[warnKey] || 0;
          if (Date.now() - lastWarn > 3600000) {
            audit("POS-WARN", `${ticker}: down ${pnlPct.toFixed(1)}% (warn threshold ${POSITION_WARN_LOSS_PCT}%)`);
            if (!(state as any)._lastPositionWarn) (state as any)._lastPositionWarn = {};
            (state as any)._lastPositionWarn[warnKey] = Date.now();
          }
        }

        // Read stop state for this ticker
        const ps = stopState[ticker] || {};
        const atrPct = ps.current_stop_pct ? ps.current_stop_pct : 2.0;
        const initialRiskPct = ps.initial_risk_pct || atrPct * 2.0;
        const highestPnl = Math.max(ps.highest_pnl || 0, pnlPct);
        const peakR = initialRiskPct > 0 ? highestPnl / initialRiskPct : 0;
        const phase = ps.phase || (peakR >= 3 ? 4 : peakR >= 2 ? 3 : peakR >= 1 ? 2 : 1);

        // Compute stop and TP levels (regime-aware, ATR-based Phase 4)
        const posRegime = ps.regime || "NEUTRAL";
        const posIsBullish = posRegime === "BULL" || posRegime === "NEUTRAL_BULL";
        const posIsBearish = posRegime === "BEAR" || posRegime === "PANIC" || posRegime === "CAUTION";
        const posTrailMult = posIsBullish ? 3.0 : (posIsBearish ? 1.5 : 2.0);

        let stopPct: number;
        if (peakR >= 3) stopPct = atrPct * posTrailMult;  // Phase 4: ATR-based, not 50%-of-peak
        else if (peakR >= 2) stopPct = Math.max(1.0, atrPct * 1.0);
        else if (peakR >= 1) stopPct = Math.max(1.5, atrPct * 1.5);
        else stopPct = Math.max(1.5, Math.min(atrPct * 2.0, 8.0));

        const tpPct = phase < 3 ? Math.max(4.0, Math.min(atrPct * 3.0, 15.0)) : 999;

        // Scale-out state from stop_state.json (persisted across restarts)
        // IMPORTANT: If the position already exists in memory, preserve the
        // in-memory values using Math.max — they may be more recent than the
        // file if a Python persist failed (e.g., mutex timeout).
        const existingPos = monitoredPositions[ticker];
        const originalQty = ps.original_qty || qty;
        const scalesCompleted = existingPos
          ? Math.max(existingPos.scalesCompleted, ps.scales_completed || 0)
          : (ps.scales_completed || 0);
        const remainingQty = existingPos
          ? Math.min(existingPos.remainingQty, ps.remaining_qty || qty)
          : (ps.remaining_qty || qty);
        const breakevenActive = existingPos
          ? (existingPos.breakevenActive || ps.breakeven_active || false)
          : (ps.breakeven_active || false);
        const regime = ps.regime || "NEUTRAL";
        const highestPrice = Math.max(ps.highest_price || current, current);

        monitoredPositions[ticker] = {
          ticker, side, entryPrice: entry, qty, phase,
          initialRiskPct, highestPnl, currentStopPct: stopPct, tpPct,
          entryDate: ps.entry_date || new Date().toISOString().split("T")[0],
          lastCheckedPrice: current,
          originalQty, remainingQty, scalesCompleted, breakevenActive,
          regime, atrPct: atrPct, highestPrice,
        };
        // Preserve pendingExit flag from in-memory state — prevents re-triggering
        // exit logic while a limit sell order is still open and unfilled
        if (existingPos?.pendingExit) {
          monitoredPositions[ticker].pendingExit = true;
        }
      }

      // Remove positions that are no longer held (order filled or closed externally)
      for (const ticker of Object.keys(monitoredPositions)) {
        if (!activeTickers.has(ticker)) {
          removePositionFromMonitor(ticker);
        }
      }

      // Auto-subscribe to position tickers not already in STREAM_TICKERS
      const streamSet = new Set(STREAM_TICKERS);
      const toSubscribe: string[] = [];
      Array.from(activeTickers).forEach((ticker) => {
        if (!streamSet.has(ticker) && !positionSubscribedTickers.has(ticker) && !isOptionSymbol(ticker)) {
          toSubscribe.push(ticker);
          positionSubscribedTickers.add(ticker);
        }
      });
      if (toSubscribe.length > 0 && streamWs && streamConnected) {
        try {
          streamWs.send(JSON.stringify({ action: "subscribe", bars: toSubscribe }));
          audit("POS-MONITOR", `Subscribed to ${toSubscribe.join(", ")} for real-time exit monitoring`);
        } catch (_) {}
      }

      if (!positionMonitorInitialized) {
        const count = Object.keys(monitoredPositions).length;
        if (count > 0) audit("POS-MONITOR", `Initialized — tracking ${count} positions via WebSocket`);
        positionMonitorInitialized = true;
      }
    } catch (err: any) {
      console.error("[pos-monitor-sync]", err?.message || err);
    }
  }

  /**
   * Called when a new position is opened. Adds it to monitoring and subscribes to its ticker.
   */
  function addPositionToMonitor(ticker: string, side: 'long' | 'short', entryPrice: number, qty: number) {
    const defaultRiskPct = 4.0; // 2x ATR estimate — will be refined on next sync
    monitoredPositions[ticker] = {
      ticker, side, entryPrice, qty, phase: 1,
      initialRiskPct: defaultRiskPct,
      highestPnl: 0,
      currentStopPct: defaultRiskPct,
      tpPct: Math.max(4.0, defaultRiskPct * 1.5),
      entryDate: new Date().toISOString().split("T")[0],
      lastCheckedPrice: entryPrice,
      originalQty: qty,
      remainingQty: qty,
      scalesCompleted: 0,
      breakevenActive: false,
      regime: "NEUTRAL",  // Refined on next sync from bot_engine
      atrPct: defaultRiskPct / 2.0,
      highestPrice: entryPrice,
    };
    // Subscribe to WebSocket if not already streaming (equity symbols only —
    // the v2/iex bars stream rejects option symbols, see isOptionSymbol above)
    const streamSet = new Set(STREAM_TICKERS);
    if (!streamSet.has(ticker) && !positionSubscribedTickers.has(ticker) && !isOptionSymbol(ticker)) {
      positionSubscribedTickers.add(ticker);
      if (streamWs && streamConnected) {
        try {
          streamWs.send(JSON.stringify({ action: "subscribe", bars: [ticker] }));
          audit("POS-MONITOR", `Auto-subscribed ${ticker} for exit monitoring`);
        } catch (_) {}
      }
    }
  }

  /**
   * Called when a position is closed. Removes from monitoring and unsubscribes.
   */
  function removePositionFromMonitor(ticker: string) {
    delete monitoredPositions[ticker];
    // OOM fix: clean up stream data for non-base tickers to prevent unbounded growth
    if (!STREAM_TICKERS.includes(ticker)) {
      delete streamVolHistory[ticker];
      delete streamPriceHistory[ticker];
      delete streamLastSignal[ticker];
    }
    if (positionSubscribedTickers.has(ticker)) {
      positionSubscribedTickers.delete(ticker);
      if (streamWs && streamConnected) {
        try { streamWs.send(JSON.stringify({ action: "unsubscribe", bars: [ticker] })); } catch (_) {}
      }
    }
  }

  // Track tickers currently being exited to prevent duplicate exit attempts
  const exitingTickers: Set<string> = new Set();

  /**
   * Core position check — called from WebSocket on every price tick for an owned stock.
   * Implements:
   *   - Scale-out in thirds (1/3 at 1R, 1/3 at 2-3R depending on regime, trail last 1/3)
   *   - Breakeven stop after first scale-out
   *   - Regime-aware exit targets (wider in BULL, tighter in BEAR/PANIC)
   *   - ATR-based Phase 4 trailing (replaces broken 50%-of-peak)
   */
  async function checkPositionOnTick(ticker: string, currentPrice: number) {
    const pos = monitoredPositions[ticker];
    if (!pos || exitingTickers.has(ticker)) return;
    if (pos.pendingExit) return; // Exit order already placed, waiting for fill
    if (!state.active || state.killSwitch) return;

    // Skip stock exit logic for options positions — options_manager.py handles those
    if (pos.positionType && pos.positionType !== 'stock') {
      return;
    }

    const entry = pos.entryPrice;
    if (entry <= 0 || currentPrice <= 0) return;

    // Calculate current P&L %
    const pnlPct = pos.side === 'long'
      ? ((currentPrice - entry) / entry) * 100
      : ((entry - currentPrice) / entry) * 100;

    // Update high water marks
    pos.highestPnl = Math.max(pos.highestPnl, pnlPct);
    pos.highestPrice = Math.max(pos.highestPrice || currentPrice, currentPrice);
    pos.lastCheckedPrice = currentPrice;

    // Recalculate R-multiple and phase
    const rMultiple = pos.initialRiskPct > 0 ? pnlPct / pos.initialRiskPct : 0;
    const peakR = pos.initialRiskPct > 0 ? pos.highestPnl / pos.initialRiskPct : 0;

    if (peakR >= 3 && pos.phase < 4) pos.phase = 4;
    else if (peakR >= 2 && pos.phase < 3) pos.phase = 3;
    else if (peakR >= 1 && pos.phase < 2) pos.phase = 2;

    // ── Regime-aware thresholds ──────────────────────────────────────────────
    const regime = pos.regime || "NEUTRAL";
    const isBullish = regime === "BULL" || regime === "NEUTRAL_BULL";
    const isBearish = regime === "BEAR" || regime === "PANIC" || regime === "CAUTION";

    // Scale-out R thresholds adapt to regime
    const scaleOut1R = 1.0;  // Always scale first third at 1R
    const scaleOut2R = isBullish ? 3.0 : (isBearish ? 1.5 : 2.0);

    // Trailing stop ATR multiplier adapts to regime
    const trailAtrMult = isBullish ? 3.0 : (isBearish ? 1.5 : 2.0);

    const atr = pos.atrPct || 2.0;

    // ── SCALE-OUT CHECK (before stop/TP logic) ──────────────────────────────
    // Scale out 1/3 at each threshold. Only if we haven't already scaled at this level.
    let shouldScaleOut = false;
    let scaleOutLabel = '';
    let scaleOutQty = 0;

    if (pos.scalesCompleted < 2 && pos.remainingQty > 1) {
      const thirdQty = Math.floor(pos.originalQty / 3);

      if (pos.scalesCompleted === 0 && rMultiple >= scaleOut1R && thirdQty >= 1) {
        // First scale-out: sell 1/3 at 1R
        shouldScaleOut = true;
        scaleOutQty = thirdQty;
        scaleOutLabel = `SCALE-OUT 1/3 at +${pnlPct.toFixed(1)}% (${rMultiple.toFixed(1)}R)`;
      } else if (pos.scalesCompleted === 1 && rMultiple >= scaleOut2R && thirdQty >= 1) {
        // Second scale-out: sell another 1/3 at 2-3R (regime-dependent)
        scaleOutQty = thirdQty;
        shouldScaleOut = true;
        scaleOutLabel = `SCALE-OUT 2/3 at +${pnlPct.toFixed(1)}% (${rMultiple.toFixed(1)}R)`;
      }
    }

    // ── DUPLICATE SELL ORDER GUARD ──────────────────────────────────────────
    // Primary defense: if there's already an open sell order for this ticker
    // (from a prior scale-out whose persist failed), skip to avoid duplicates.
    if (shouldScaleOut) {
      const exitSide = pos.side === 'long' ? 'sell' : 'buy';
      const existingSellOrder = openOrders.find(o => o.ticker === ticker && o.side === exitSide);
      if (existingSellOrder) {
        audit("WS-SCALE-OUT", `${ticker}: skipping — existing open ${exitSide} order found (id=${existingSellOrder.orderId})`);
        shouldScaleOut = false;
      }
    }
    if (shouldScaleOut) {
      // Double-check against Alpaca API in case local tracking missed something
      try {
        const exitSide = pos.side === 'long' ? 'sell' : 'buy';
        const existingOrders = await alpaca(`/v2/orders?status=open&symbols=${ticker}&side=${exitSide}`);
        const parsed = JSON.parse(typeof existingOrders === 'string' ? existingOrders : JSON.stringify(existingOrders));
        if (Array.isArray(parsed) && parsed.length > 0) {
          audit("WS-SCALE-OUT", `${ticker}: skipping duplicate — ${parsed.length} open ${exitSide} orders already exist on Alpaca`);
          shouldScaleOut = false;
        }
      } catch (_) {
        // If API check fails, proceed with local guard only (already checked above)
      }
    }

    if (shouldScaleOut) {
      exitingTickers.add(ticker);

      // ── COVERED CALL CHECK on scale-out ──────────────────────────────────
      // If scaling out would drop us below 100 shares, buy back any short calls first
      // to prevent becoming under-covered (partial naked exposure).
      const sharesAfterScale = pos.remainingQty - scaleOutQty;
      if (sharesAfterScale < 100) {
        try {
          const scalePositions = await alpaca("/v2/positions");
          const scaleAllPos = Array.isArray(scalePositions) ? scalePositions : JSON.parse(typeof scalePositions === 'string' ? scalePositions : '[]');
          const scaleCalls = scaleAllPos.filter((p: any) =>
            p.asset_class === "us_option"
            && p.symbol.startsWith(ticker)
            && parseInt(p.qty) < 0
            && p.symbol.slice(ticker.length).includes("C")
          );
          for (const sc of scaleCalls) {
            const buyBackQty = Math.abs(parseInt(sc.qty));
            audit("CC-UNWIND", `${ticker}: Scale-out would leave ${sharesAfterScale} shares — buying back ${buyBackQty}x ${sc.symbol}`);
            try {
              await alpaca("/v2/orders", {
                method: "POST",
                body: JSON.stringify({
                  symbol: sc.symbol,
                  qty: String(buyBackQty),
                  side: "buy",
                  type: "market",
                  time_in_force: "day",
                }),
              });
              audit("CC-UNWIND", `${ticker}: Buy-back order placed for ${sc.symbol}`);
            } catch (unwindErr: any) {
              audit("CC-UNWIND-ERROR", `${ticker}: Failed to buy back ${sc.symbol} on scale-out: ${unwindErr?.message}`);
              notify("system", `CRITICAL: Could not buy back covered call ${sc.symbol} — scale-out BLOCKED`);
              exitingTickers.delete(ticker);
              return;
            }
          }
        } catch (_) {
          audit("CC-UNWIND-WARN", `${ticker}: Could not check for open calls on scale-out`);
        }
      }

      try {
        const exitSide = pos.side === 'long' ? 'sell' : 'buy';
        const orderParams = getOrderParams(currentPrice, 'take_profit');

        await alpaca("/v2/orders", {
          method: "POST",
          body: JSON.stringify({
            symbol: ticker,
            qty: String(scaleOutQty),
            side: exitSide,
            ...orderParams,
          }),
        });

        pos.scalesCompleted++;
        pos.remainingQty -= scaleOutQty;
        pos.qty = pos.remainingQty;

        // After first scale-out: activate breakeven stop
        if (pos.scalesCompleted === 1) {
          pos.breakevenActive = true;
        }

        audit("WS-SCALE-OUT", `${scaleOutLabel} | ${orderParams.type.toUpperCase()} ${exitSide} ${scaleOutQty} of ${pos.originalQty} ${ticker} @ $${currentPrice.toFixed(2)} (${pos.remainingQty} remaining)`);
        notify("exit", `Scale-out ${ticker}: ${scaleOutLabel}`);

        // Persist scale-out state to stop_state.json
        let pythonPersistOk = false;
        try {
          await execPythonSerialized(`python3 -c "
import json, os
DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
p = os.path.join(DATA_DIR, 'voltrade_stop_state.json')
try:
    with open(p) as f: ss = json.load(f)
except: ss = {}
if '${ticker}' in ss:
    ss['${ticker}']['scales_completed'] = ${pos.scalesCompleted}
    ss['${ticker}']['remaining_qty'] = ${pos.remainingQty}
    ss['${ticker}']['original_qty'] = ${pos.originalQty}
    ss['${ticker}']['breakeven_active'] = ${pos.breakevenActive ? 'True' : 'False'}
    with open(p, 'w') as f: json.dump(ss, f)
"`, { timeout: 5000 });
          pythonPersistOk = true;
        } catch (persistErr: any) {
          audit("WS-SCALE-OUT-WARN", `${ticker}: Python persist failed (${persistErr?.message?.slice(0, 80)}), using Node.js fallback`);
        }

        // Fallback: persist directly from Node.js if Python mutex is stuck
        if (!pythonPersistOk) {
          try {
            const DATA_DIR = fs.existsSync('/data') ? '/data/voltrade' : '/tmp';
            const statePath = path.join(DATA_DIR, 'voltrade_stop_state.json');
            const state = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
            if (state[ticker]) {
              state[ticker].scales_completed = pos.scalesCompleted;
              state[ticker].remaining_qty = pos.remainingQty;
              state[ticker].original_qty = pos.originalQty;
              state[ticker].breakeven_active = pos.breakevenActive;
              fs.writeFileSync(statePath, JSON.stringify(state));
              audit("WS-SCALE-OUT", `${ticker}: scale-out state persisted via Node.js fallback`);
            }
          } catch (_) {}
        }

      } catch (err: any) {
        audit("WS-SCALE-OUT-ERROR", `${ticker}: ${err?.message?.slice(0, 150)}`);
      } finally {
        exitingTickers.delete(ticker);
      }
      return; // Don't evaluate full exit on the same tick as a scale-out
    }

    // ── STOP LEVEL CALCULATION ──────────────────────────────────────────────
    // Phase 4 fix: ATR-based trailing from highest PRICE, not 50% of peak P&L
    let stopPct: number;
    if (pos.phase >= 4) {
      // Trail at 2-3× ATR below the highest price reached (regime-aware)
      const trailFromHigh = atr * trailAtrMult;
      stopPct = trailFromHigh;
    } else if (pos.phase >= 3) {
      stopPct = Math.max(1.0, atr * 1.0);
    } else if (pos.phase >= 2) {
      stopPct = Math.max(1.5, atr * 1.5);
    } else {
      stopPct = pos.currentStopPct;
    }

    // Breakeven override: after first scale-out, stop can't be below entry
    // For phases 2+, this means drawdown from peak can't exceed the peak P&L itself
    // (i.e., P&L can't go negative)
    const breakevenFloor = pos.breakevenActive;

    // ── STOP TRIGGER CHECK ──────────────────────────────────────────────────
    let shouldStop = false;
    let stopType: 'stop_loss' | 'trailing_stop' = 'stop_loss';
    let stopReason = '';

    if (pos.phase >= 2) {
      if (pos.phase >= 4) {
        // Phase 4: ATR-based trailing from highest PRICE
        const drawdownFromHighPrice = ((pos.highestPrice - currentPrice) / pos.highestPrice) * 100;
        if (drawdownFromHighPrice >= stopPct) {
          shouldStop = true;
          stopType = 'trailing_stop';
          stopReason = `WS TRAILING STOP Phase 4: price $${currentPrice.toFixed(2)} dropped ${drawdownFromHighPrice.toFixed(1)}% from high $${pos.highestPrice.toFixed(2)} (${trailAtrMult.toFixed(0)}×ATR trail, regime=${regime})`;
        }
      } else {
        // Phase 2-3: trailing from peak P&L
        const drawdownFromPeak = pos.highestPnl - pnlPct;
        if (drawdownFromPeak >= stopPct) {
          shouldStop = true;
          stopType = 'trailing_stop';
          stopReason = `WS TRAILING STOP Phase ${pos.phase}: P&L ${pnlPct.toFixed(1)}% dropped ${drawdownFromPeak.toFixed(1)}% from peak ${pos.highestPnl.toFixed(1)}%`;
        }
      }

      // Breakeven enforcement: if active, stop triggers if P&L goes below 0
      if (!shouldStop && breakevenFloor && pnlPct < 0) {
        shouldStop = true;
        stopType = 'trailing_stop';
        stopReason = `WS BREAKEVEN STOP: P&L ${pnlPct.toFixed(1)}% dropped below entry after scale-out (breakeven active)`;
      }
    } else {
      if (pnlPct <= -stopPct) {
        shouldStop = true;
        stopType = 'stop_loss';
        stopReason = `WS STOP LOSS: ${pnlPct.toFixed(1)}% loss hit Phase 1 stop at -${stopPct.toFixed(1)}%`;
      }
    }

    // Check take-profit (only for remaining position, only in early phases before all scale-outs done)
    let shouldTP = false;
    if (!shouldStop && pnlPct >= pos.tpPct && pos.phase < 3 && pos.scalesCompleted >= 2) {
      // TP only fires for the last third after both scale-outs are done
      shouldTP = true;
    }

    // Time stop check (only if no other exit triggered)
    let shouldTimeStop = false;
    if (!shouldStop && !shouldTP) {
      try {
        const daysHeld = Math.floor((Date.now() - new Date(pos.entryDate).getTime()) / 86400000);
        if (daysHeld >= 7 && Math.abs(pnlPct) < 2.0) {
          shouldTimeStop = true;
        }
      } catch (_) {}
    }

    if (!shouldStop && !shouldTP && !shouldTimeStop) return;

    // ── FIRE EXIT ORDER (remaining position) ─────────────────────────────────
    exitingTickers.add(ticker);

    // ── COVERED CALL UNWIND: Buy back any short calls before selling stock ──
    // If we sell stock while holding a short call, it becomes naked (unlimited risk).
    try {
      const positionsResp = await alpaca("/v2/positions");
      const allPositions = Array.isArray(positionsResp) ? positionsResp : JSON.parse(typeof positionsResp === 'string' ? positionsResp : '[]');
      const shortCalls = allPositions.filter((p: any) =>
        p.asset_class === "us_option"
        && p.symbol.startsWith(ticker)
        && parseInt(p.qty) < 0
        && p.symbol.slice(ticker.length).includes("C")
      );

      for (const sc of shortCalls) {
        const buyBackQty = Math.abs(parseInt(sc.qty));
        audit("CC-UNWIND", `${ticker}: Buying back ${buyBackQty}x ${sc.symbol} before stock exit`);
        try {
          await alpaca("/v2/orders", {
            method: "POST",
            body: JSON.stringify({
              symbol: sc.symbol,
              qty: String(buyBackQty),
              side: "buy",
              type: "market",
              time_in_force: "day",
            }),
          });
          audit("CC-UNWIND", `${ticker}: Buy-back order placed for ${sc.symbol}`);
        } catch (unwindErr: any) {
          audit("CC-UNWIND-ERROR", `${ticker}: Failed to buy back ${sc.symbol}: ${unwindErr?.message}`);
          // DON'T proceed with stock sale if we can't close the call — that leaves us naked
          notify("system", `CRITICAL: Could not buy back covered call ${sc.symbol} — stock exit BLOCKED`);
          exitingTickers.delete(ticker);
          return;
        }
      }
    } catch (posErr: any) {
      // If we can't check positions, log but proceed (better to exit than hold)
      audit("CC-UNWIND-WARN", `${ticker}: Could not check for open calls: ${posErr?.message}`);
    }

    // Check for existing open sell orders on this ticker before placing exit
    try {
      const exitSideCheck = pos.side === 'long' ? 'sell' : 'buy';
      const existingResp = await alpaca(`/v2/orders?status=open&symbols=${ticker}&side=${exitSideCheck}`);
      const existingOrders = JSON.parse(typeof existingResp === 'string' ? existingResp : JSON.stringify(existingResp));
      if (Array.isArray(existingOrders) && existingOrders.length > 0) {
        const totalHeld = existingOrders.reduce((sum: number, o: any) => sum + Number(o.qty || 0), 0);
        audit("WS-EXIT", `${ticker}: skipping — ${existingOrders.length} open ${exitSideCheck} orders already exist (${totalHeld} shares held)`);
        exitingTickers.delete(ticker);
        return;
      }
    } catch (_) {
      // If the check fails, proceed cautiously
    }

    try {
      const exitSide = pos.side === 'long' ? 'sell' : 'buy';
      const exitQty = pos.remainingQty;
      const exitType = shouldStop ? stopType : (shouldTP ? 'take_profit' : 'time_stop');
      const exitContext: OrderContext = shouldStop ? stopType : (shouldTP ? 'take_profit' : 'stop_loss');
      const orderParams = getOrderParams(currentPrice, exitContext);

      // Submit sell/cover order for remaining shares
      const orderResult = await alpaca("/v2/orders", {
        method: "POST",
        body: JSON.stringify({
          symbol: ticker,
          qty: String(exitQty),
          side: exitSide,
          ...orderParams,
        }),
      });

      const scaleNote = pos.scalesCompleted > 0 ? ` (final ${exitQty}/${pos.originalQty} shares, ${pos.scalesCompleted} prior scale-outs)` : '';
      const reason = shouldStop ? stopReason + scaleNote
        : shouldTP ? `WS TAKE PROFIT: +${pnlPct.toFixed(1)}% hit target +${pos.tpPct.toFixed(1)}% (Phase ${pos.phase})${scaleNote}`
        : `WS TIME STOP: held since ${pos.entryDate}, P&L only ${pnlPct.toFixed(1)}%${scaleNote}`;

      audit("WS-EXIT", `${reason} | ${orderParams.type.toUpperCase()} ${exitSide} ${exitQty} ${ticker} @ $${currentPrice.toFixed(2)}`);
      notify("exit", `WS exit ${ticker}: ${reason}`);

      // REPAIR 2026-07-06 (R12-D2): close the ML feedback entry record with
      // the bot's own P&L accounting. Full exits only — scale-outs leave the
      // position (and its record) open by design.
      recordExitFill(buildExitFillPayload({
        ticker, exitSide, qty: exitQty, fillPrice: currentPrice,
        pnlPct, exitReason: exitType, entryDate: pos.entryDate,
        codeVersion: pkgVersion,
      }));

      // Write stop-loss cooldown
      if (exitType === 'stop_loss' || exitType === 'trailing_stop') {
        try {
          await execPythonSerialized(`python3 -c "
import json, os, time
DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
cd_path = os.path.join(DATA_DIR, 'voltrade_stop_cooldown.json')
try:
    with open(cd_path) as f: cd = json.load(f)
except: cd = {}
cd['${ticker}'] = time.time()
cd = {k: v for k, v in cd.items() if time.time() - v < 86400}
with open(cd_path, 'w') as f: json.dump(cd, f)
"`, { timeout: 5000 });
        } catch (_) {}

        // Circuit breaker logic — dedup same ticker within 5 min to prevent
        // duplicate exit attempts from inflating the stop counter
        if (!(state as any).lastStopTimes) (state as any).lastStopTimes = {};
        const lastStopTime = (state as any).lastStopTimes[ticker] || 0;
        if (Date.now() - lastStopTime < 300000) {
          audit("WS-EXIT", `${ticker}: skipping circuit breaker increment — same ticker stopped ${Math.round((Date.now() - lastStopTime) / 1000)}s ago`);
        } else {
          // Options exits don't trigger stock circuit breaker
          if (!pos.positionType || pos.positionType === 'stock') {
            state.consecutiveStopLosses++;
          }
        }
        (state as any).lastStopTimes[ticker] = Date.now();
        if (!Array.isArray((state as any).recentStopTickers)) (state as any).recentStopTickers = [];
        (state as any).recentStopTickers.push(ticker);
        if ((state as any).recentStopTickers.length > 5) (state as any).recentStopTickers.shift();

        if (state.consecutiveStopLosses >= 3) {
          const recentTickers: string[] = (state as any).recentStopTickers || [];
          const uniqueTickers = new Set(recentTickers.slice(-3)).size;
          const dailyLossHit = state.dailyPnL / (state.equityPeak || 100000) * 100 <= -5;

          if (dailyLossHit) {
            const msUntilClose = (() => {
              const now = new Date();
              const close = new Date(now); close.setHours(20, 0, 0, 0);
              return Math.max(close.getTime() - now.getTime(), 3600000);
            })();
            state.circuitBreakerUntil = Date.now() + msUntilClose;
            sendEmailAlert("Circuit Breaker — Day Halted", `3 stops + down ${(state.dailyPnL / (state.equityPeak || 100000) * 100).toFixed(1)}% today. Trading paused for rest of day.`);
            audit("CIRCUIT-BREAKER", `3 stops + -5%+ day loss — paused rest of day`);
          } else if (uniqueTickers >= 2) {
            state.circuitBreakerUntil = Date.now() + 3600000;
            sendEmailAlert("Circuit Breaker Triggered", `3 stops across ${uniqueTickers} different tickers — market conditions unfavorable. Paused 1 hour.`);
            audit("CIRCUIT-BREAKER", `3 stops on ${Array.from(new Set(recentTickers.slice(-3))).join(", ")} — broad market issue, paused 1 hour`);
          } else {
            const blockedTicker = recentTickers[recentTickers.length - 1];
            if (!Array.isArray((state as any).dailyBlockedTickers)) (state as any).dailyBlockedTickers = [];
            (state as any).dailyBlockedTickers.push(blockedTicker);
            // OOM fix: cap dailyBlockedTickers to prevent unbounded growth
            if ((state as any).dailyBlockedTickers.length > 20) (state as any).dailyBlockedTickers.length = 20;
            state.consecutiveStopLosses = 0;
            audit("TICKER-BLOCKED", `${blockedTicker}: 3 stops on same ticker — blocked for today`);
          }
        }
      } else if (exitType === 'take_profit') {
        state.consecutiveStopLosses = 0;
      }

      // Mark as pending exit — don't remove from monitoring until the order fills.
      // This prevents the position from being re-discovered on the next sync and
      // triggering a duplicate exit order (which fails with 403 insufficient qty).
      pos.pendingExit = true;

    } catch (err: any) {
      audit("WS-EXIT-ERROR", `${ticker}: ${err?.message?.slice(0, 150)}`);
    } finally {
      exitingTickers.delete(ticker);
    }
  }

  // Sync monitored positions every 60 seconds to stay aligned with Alpaca + stop state
  setInterval(async () => {
    if (!state.active || state.killSwitch) return;
    try { await syncMonitoredPositions(); } catch (_) {}
  }, 60000);

  function startStreaming() {
    if (streamWs) { try { streamWs.close(); } catch (_) {} }

    // REPAIR 2026-07-09 (R19): this was hardcoded to /v2/sip — the same
    // paid-entitlement tier whose 2026-07-06 rejection (wishlist.md #9)
    // required alpaca_feed.py's data_feed() resolver for every REST call
    // site. That fix never reached this file (TypeScript, outside the
    // Python-only ratchet in test_alpaca_feed.py's TestNoHardcodedFeeds),
    // so this stream kept connecting to /v2/sip regardless of entitlement
    // status. Unlike REST candidate-discovery (which needs SIP's true
    // consolidated volume — alpaca_feed.py deliberately rejects iex there),
    // this stream is consumed ONLY by checkPositionOnTick's `close` price
    // for real-time stop-loss/take-profit/trailing-stop triggers — volume
    // undercount doesn't apply. /v2/iex requires no special entitlement and
    // is never rejected, so it removes this path's dependency on SIP status
    // entirely rather than replicating the REST probe/downgrade machinery.
    const ws = new WebSocket("wss://stream.data.alpaca.markets/v2/iex");
    streamWs = ws;

    ws.on("open", () => {
      ws.send(JSON.stringify({ action: "auth", key: ALPACA_KEY, secret: ALPACA_SECRET }));
    });

    ws.on("message", async (raw: Buffer) => {
      try {
        const msgs = JSON.parse(raw.toString());
        const items = Array.isArray(msgs) ? msgs : [msgs];

        for (const item of items) {
          if (!item || typeof item !== "object") continue;

          // Auth success — subscribe to 1-min bars
          if (item.T === "success" && item.msg === "authenticated") {
            ws.send(JSON.stringify({ action: "subscribe", bars: STREAM_TICKERS }));
          }
          if (item.T === "subscription") {
            streamConnected = true;
            audit("STREAM", `Real-time feed live — ${(item.bars || []).length} tickers`);
            // Initialize position monitor — sync positions and auto-subscribe owned tickers
            syncMonitoredPositions().catch(() => {});
          }
          // REPAIR 2026-07-09 (R19): Alpaca reports auth/subscribe rejection
          // as a {"T":"error",...} frame, not a socket close — this was
          // previously unhandled by any branch here, so a rejection (e.g.
          // an entitlement problem) left checkPositionOnTick silently
          // starved of bars with zero audit trail (exactly how the /v2/sip
          // hardcoding above went undetected for days: zero "Real-time feed
          // live", zero WS-EXIT, zero error, ever, across 22+ hours of live
          // trading and 19 restarts, verified via /api/diag/audit).
          if (item.T === "error") {
            audit("STREAM-ERROR", `Alpaca stream error code=${item.code ?? "?"} msg=${String(item.msg || "").slice(0, 150)}`);
          }

          // 1-minute bar received — position check + signal detection
          if (item.T === "b" && item.S && item.v > 0 && item.c > 0) {
            const ticker = item.S as string;
            const volume = item.v as number;
            const close  = item.c as number;
            const open_  = item.o as number;

            // ── Position monitor: check exits on every tick for owned stocks ──
            if (monitoredPositions[ticker]) {
              checkPositionOnTick(ticker, close).catch((err: any) => {
                console.error(`[ws-pos-check] ${ticker}:`, err?.message || err);
              });
            }

            if (!streamVolHistory[ticker])   streamVolHistory[ticker]   = [];
            if (!streamPriceHistory[ticker]) streamPriceHistory[ticker] = [];
            streamVolHistory[ticker].push(volume);
            streamPriceHistory[ticker].push(close);
            if (streamVolHistory[ticker].length   > 20) streamVolHistory[ticker].shift();
            if (streamPriceHistory[ticker].length > 20) streamPriceHistory[ticker].shift();
            if (streamVolHistory[ticker].length < 5) continue;

            const volArr  = streamVolHistory[ticker].slice(0, -1);
            const avgVol  = volArr.reduce((a: number, b: number) => a + b, 0) / volArr.length;
            const volRatio = avgVol > 0 ? volume / avgVol : 1;
            const barChg   = open_ > 0 ? ((close - open_) / open_) * 100 : 0;
            const ph       = streamPriceHistory[ticker];
            const mom5     = ph.length >= 5 ? ((ph[ph.length-1] - ph[ph.length-5]) / ph[ph.length-5]) * 100 : 0;

            // Signal: volume 2.5x+ average AND directional price move
            if (volRatio >= 2.5 && Math.abs(barChg) >= 0.8 && Math.abs(mom5) >= 0.3) {
              const cooldown = 10 * 60 * 1000;
              if (Date.now() - (streamLastSignal[ticker] || 0) < cooldown) continue;
              if (!state.active || state.killSwitch || state.circuitBreakerUntil > Date.now()) continue;

              streamLastSignal[ticker] = Date.now();
              const dir = barChg > 0 ? "bullish" : "bearish";
              audit("STREAM-SIGNAL", `${ticker}: ${volRatio.toFixed(1)}x vol, ${barChg > 0 ? "+" : ""}${barChg.toFixed(2)}% bar (${dir}) — queued for fast scan`);

              // Queue for priority processing in next Tier 2 cycle
              if (!Array.isArray((state as any).streamSignalQueue)) (state as any).streamSignalQueue = [];
              const queue: any[] = (state as any).streamSignalQueue;
              // Deduplicate and cap
              if (!queue.find((s: any) => s.ticker === ticker)) {
                queue.push({ ticker, price: close, direction: dir, volRatio, barChg, ts: Date.now() });
              }
              // Expire signals older than 5 minutes
              (state as any).streamSignalQueue = queue
                .filter((s: any) => Date.now() - s.ts < 300_000)
                .slice(-20);
            }
          }
        }
      } catch (_) {}
    });

    ws.on("error", (err: Error) => {
      audit("STREAM-ERROR", err.message.slice(0, 100));
    });

    ws.on("close", () => {
      // REPAIR 2026-07-09 (R19): this reconnect loop was completely silent —
      // no audit trail distinguished "briefly blipped" from "has never once
      // subscribed successfully." Logging the wasConnected flag makes a
      // stuck reconnect loop (the /v2/sip failure mode this PR fixes)
      // visible on the very next occurrence instead of requiring a live
      // WS trace to notice.
      const wasConnected = streamConnected;
      streamConnected = false;
      audit("STREAM-DISCONNECT", wasConnected
        ? "Feed disconnected — reconnecting in 10s"
        : "Feed closed before ever reaching subscribed state — reconnecting in 10s");
      // Auto-reconnect after 10 seconds if bot is still active
      setTimeout(() => { if (state.active && !state.killSwitch) startStreaming(); }, 10000);
    });
  }

  function stopStreaming() {
    if (streamWs) { try { streamWs.close(); } catch (_) {} streamWs = null; }
    streamConnected = false;
  }

  // EVENT-LOOP LAG MONITOR (REPAIR 2026-07-12, KNOWN BROKEN #18 follow-up —
  // see server/eventLoopLag.ts header for the full correlation this tests:
  // both 2026-07-12 TIER2-ERROR daemon-timeout entries landed within
  // 17-35ms of a STREAM-DISCONNECT entry, closer than two independent
  // ~300s/~600s timers should coincidentally land. A shared cause — the
  // event loop stalling long enough to delay both an elapsed setTimeout and
  // a queued WS 'close' event until it catches up — is untested until now.
  let eventLoopLagLastTick = Date.now();
  setInterval(() => {
    const now = Date.now();
    const lag = computeLagMs(EVENTLOOP_LAG_CHECK_MS, now - eventLoopLagLastTick);
    eventLoopLagLastTick = now;
    if (lagExceedsThreshold(lag)) {
      audit("EVENTLOOP-LAG", `Node event loop stalled ~${lag}ms past its ${EVENTLOOP_LAG_CHECK_MS}ms tick — check for a coincident TIER2-ERROR/STREAM-DISCONNECT (KNOWN BROKEN #18)`);
    }
  }, EVENTLOOP_LAG_CHECK_MS);

  // ── Three-Tier Engine Intervals ─────────────────────────────────────────────
  let tier2Running = false;
  let tier3Running = false;
  // Cooldown after Tier 2 scan failures — prevents rapid-fire retries when
  // Python subprocess returns errors like "RuntimeError: can't start new thread".
  // Cleared on successful scan. Grows with consecutive failures (60s → 120s → 240s,
  // capped at 600s) so a thread-exhaustion crash loop can't hammer the daemon.
  let tier2ConsecutiveFailures = 0;

  // TIER 1: Reflex (every 45 seconds) — positions, stops, order execution
  setInterval(async () => {
    if (!state.active || state.killSwitch) return;
    if (state.circuitBreakerUntil > Date.now()) return;

    try {
      const clock = await alpaca("/v2/clock");
      if (!clock.is_open) return; // Only during market hours
      await tier1Reflex();
    } catch (err: any) {
      console.error("[tier1]", err?.message || err);
    }
  }, 45000);

  // TIER 2: Intelligence (adaptive interval based on market time)
  function getTier2Interval(): number {
    const etTime = getETHour();

    // Market open rush (9:30-10:30 ET): scan every 1 minute (matches system_config TIER2_OPEN_MS)
    if (etTime >= 9.5 && etTime < 10.5) return 60000;
    // Mid-morning (10:30-12 ET): every 3 minutes (matches system_config TIER2_MIDMORNING_MS)
    if (etTime >= 10.5 && etTime < 12) return 180000;
    // Quiet midday (12-2pm ET): every 7 minutes — deeper analysis window (matches system_config TIER2_MIDDAY_MS)
    if (etTime >= 12 && etTime < 14) return 420000;
    // Power hour (2-4pm ET): every 2 minutes (matches system_config TIER2_POWER_HOUR_MS)
    if (etTime >= 14 && etTime < 16) return 120000;
    // After hours (4-8pm ET): every 15 minutes — research only
    if (etTime >= 16 && etTime < 20) return 900000;
    // Pre-market (4-9:30am ET): every 5 minutes (matches system_config TIER2_PREMARKET_MS)
    if (etTime >= 4 && etTime < 9.5) return 300000;
    // Overnight: every 30 minutes
    return 1800000;
  }

  // BUG FIX 2026-05-06 batch 10: scheduleTier2 spawned PARALLEL timer chains
  // when its callsites raced with the tier2Running lock. Each defer-on-lock
  // (line "if (tier2Running) { setTimeout(scheduleTier2, ...); return; }")
  // started a NEW chain, while the in-progress scan ALSO scheduled a new
  // chain when it finished. Result: after enough lock contention there were
  // 5-10 chains all firing every 3 minutes, staggered so scans actually ran
  // every 15-30 seconds. Audit log showed scans firing constantly with
  // "interval: 3min" labels.
  //
  // Fix: track whether a tier2 timer is already armed. Refuse to arm a
  // second one. Anywhere we reschedule, go through this single helper.
  let tier2TimerArmed = false;
  function armTier2Timer(delayMs: number) {
    if (tier2TimerArmed) return; // a chain is already pending
    tier2TimerArmed = true;
    setTimeout(() => {
      tier2TimerArmed = false;
      scheduleTier2().catch(() => {});
    }, delayMs);
  }

  async function scheduleTier2() {
    if (!state.active || state.killSwitch || tier2Running) {
      armTier2Timer(getTier2Interval());
      return;
    }
    if (state.circuitBreakerUntil > Date.now()) {
      armTier2Timer(getTier2Interval());
      return;
    }

    let hitMutexTimeout = false;
    let schedulerThrew = false;
    try {
      const clock = await alpaca("/v2/clock");
      const etH = getETHour();
      const isAnyWindow = clock.is_open || (etH >= 4 && etH < 20);
      if (!isAnyWindow) {
        armTier2Timer(getTier2Interval());
        return;
      }

      tier2Running = true;
      const interval = getTier2Interval();
      audit("TIER2", `Starting scan (interval: ${Math.round(interval / 60000)}min based on market time)`);
      await tier2Intelligence(clock.is_open, etH);
    } catch (err: any) {
      schedulerThrew = true;
      const msg = String(err?.message || err);
      hitMutexTimeout = msg.includes("mutex timeout");
      if (!hitMutexTimeout) {
        console.error("[tier2]", msg);
        audit("TIER2-ERROR", msg.slice(0, 200));
      } else {
        console.error("[tier2] mutex timeout — backing off 60s");
      }
    } finally {
      tier2Running = false;
      // Track consecutive failures for back-off. Anything that reached the scan
      // and errored (scheduler throw, Python scan JSON error, subprocess crash)
      // contributes; a clean run resets the counter.
      const failed = schedulerThrew || tier2LastScanFailed;
      if (failed) {
        tier2ConsecutiveFailures += 1;
      } else {
        tier2ConsecutiveFailures = 0;
      }

      let delay: number;
      if (hitMutexTimeout) {
        delay = Math.max(60000, getTier2Interval());
      } else if (failed) {
        // Exponential back-off after scan failure: 60s, 120s, 240s, 480s,
        // capped at 600s. Prevents a rapid-fire retry loop when Python can't
        // spawn new threads or the container is OOM-thrashing. Always at least
        // as long as the normal cadence so we never scan *more* often on failure.
        const n = Math.min(tier2ConsecutiveFailures, 5);
        const backoff = Math.min(600000, 60000 * Math.pow(2, n - 1));
        delay = Math.max(backoff, getTier2Interval());
        audit("TIER2-BACKOFF", `failure #${tier2ConsecutiveFailures}, next scan in ${Math.round(delay/1000)}s`);
      } else {
        delay = getTier2Interval();
      }
      armTier2Timer(delay);
    }
  }

  // Start Tier 2 with adaptive scheduling
  armTier2Timer(10000); // First run after 10 seconds

  // TIER 3: Strategic (every 1 hour) — ML retrain, macro, manipulation scan
  setInterval(async () => {
    if (!state.active || tier3Running) return;

    try {
      tier3Running = true;
      await tier3Strategic();
    } catch (err: any) {
      console.error("[tier3]", err?.message || err);
      audit("TIER3-ERROR", String(err?.message || err).slice(0, 200));
    } finally {
      tier3Running = false;
    }
  }, 3600000); // 1 hour

  // Run Tier 3 once on startup after a delay (Tier 2 is handled by scheduleTier2)
  setTimeout(() => { tier3Strategic().catch(() => {}); }, 30000);

  // Route: Get last scan result
  app.get("/api/bot/last-scan", requireOwner, (_req, res) => {
    res.json(lastScanResult || { message: "No scan run yet. Activate the bot to start." });
  });

  // Route: Market calendar — holidays and early closes
  app.get("/api/bot/calendar", requireOwner, async (_req, res) => {
    try {
      const today = new Date().toISOString().split("T")[0];
      const future = new Date(Date.now() + 30 * 86400000).toISOString().split("T")[0];
      const cal = await alpaca(`/v2/calendar?start=${today}&end=${future}`);

      const tradingDays = new Set((cal as any[]).map((d: any) => d.date));

      const holidays: Array<{date: string; name: string}> = [];
      const earlyCloses: Array<{date: string; close: string}> = [];

      // ALPHA AUDIT 2026-05-04 batch 6: holiday name lookup updated for 2026.
      // Previous values were 2025 dates (MLK Jan 20, Presidents Feb 17,
      // Memorial May 26, Labor Sep 1, etc.) which never matched in 2026 —
      // so detected holidays from Alpaca's calendar fell through to the
      // default "Market Holiday" name. The DATES themselves come from
      // Alpaca's /v2/calendar so trading-day detection was correct; just
      // the human-readable name was generic.
      const holidayNames: Record<string, string> = {
        "01-01": "New Year's Day",
        "01-19": "Martin Luther King Jr. Day",     // 2026: 3rd Monday of January
        "02-16": "Presidents' Day",                // 2026: 3rd Monday of February
        "04-03": "Good Friday",                    // 2026: April 3
        "05-25": "Memorial Day",                   // 2026: last Monday of May
        "06-19": "Juneteenth",                     // Fixed date
        "07-03": "Independence Day (observed)",    // 2026: July 4 = Saturday → observed Friday
        "07-04": "Independence Day",               // Fixed (in case observed differently)
        "09-07": "Labor Day",                      // 2026: 1st Monday of September
        "11-26": "Thanksgiving Day",               // 2026: 4th Thursday of November
        "12-25": "Christmas Day",
      };

      for (let i = 0; i < 30; i++) {
        const d = new Date(Date.now() + i * 86400000);
        const dayOfWeek = d.getDay();
        if (dayOfWeek === 0 || dayOfWeek === 6) continue;
        const dateStr = d.toISOString().split("T")[0];
        if (!tradingDays.has(dateStr)) {
          const mmdd = dateStr.substring(5);
          holidays.push({
            date: dateStr,
            name: holidayNames[mmdd] || "Market Holiday",
          });
        }
      }

      for (const day of (cal as any[])) {
        if (day.close && day.close < "16:00") {
          earlyCloses.push({ date: day.date, close: day.close });
        }
      }

      const tomorrow = new Date(Date.now() + 86400000);
      const tomorrowStr = tomorrow.toISOString().split("T")[0];
      const tomorrowDay = tomorrow.getDay();
      let tomorrowStatus = "open";
      let tomorrowNote = "";

      if (tomorrowDay === 0 || tomorrowDay === 6) {
        tomorrowStatus = "weekend";
        tomorrowNote = "Market is closed — it's the weekend.";
      } else if (!tradingDays.has(tomorrowStr)) {
        tomorrowStatus = "holiday";
        const mmdd = tomorrowStr.substring(5);
        tomorrowNote = `Market is closed tomorrow for ${holidayNames[mmdd] || "a market holiday"}.`;
      } else {
        const tomorrowCal = (cal as any[]).find((d: any) => d.date === tomorrowStr);
        if (tomorrowCal?.close && tomorrowCal.close < "16:00") {
          tomorrowStatus = "early_close";
          tomorrowNote = `Market closes early tomorrow at ${tomorrowCal.close} ET.`;
        } else {
          tomorrowNote = "Market is open tomorrow, regular hours.";
        }
      }

      res.json({
        tomorrow: { status: tomorrowStatus, note: tomorrowNote, date: tomorrowStr },
        holidays,
        earlyCloses,
        tradingDaysNext30: tradingDays.size,
      });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // Auto-start: log all rules and begin
  setTimeout(() => {
    // ── FEEDBACK CLEANUP: Wipe pre-fix trade data on boot ─────────────────
    // All trades before v1.0.33 ran on broken code (27+ bugs, no scale-out,
    // no WebSocket stops, HEAT-CAP blocking all trades, GLD flooding, etc.).
    // Their outcomes reflect code bugs, not strategy quality.
    // The ML filter in ml_model_v2.py also skips them, but cleaning the file
    // prevents accumulating dead weight and speeds up future loads.
    try {
      execPythonSerialized(`python3 -c "
import json, os
try:
    from storage_config import TRADE_FEEDBACK_PATH
except ImportError:
    DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
    TRADE_FEEDBACK_PATH = os.path.join(DATA_DIR, 'voltrade_trade_feedback.json')
if os.path.exists(TRADE_FEEDBACK_PATH):
    with open(TRADE_FEEDBACK_PATH) as f:
        raw = json.load(f)
    # REPAIR 2026-07-06 (R12-D4): filter extracted to feedback_boot_cleanup.py
    # (unit-tested). Fixes: seeds no longer wiped every deploy; version compare
    # numeric (the old '1.0.153' < '1.0.34' string trap); the April-2026
    # pre-Bug-25b fossils (code_version 1.0.33, pnl 0) finally removed.
    from feedback_boot_cleanup import clean_feedback
    valid, removed = clean_feedback(raw)
    if removed > 0:
        with open(TRADE_FEEDBACK_PATH, 'w') as f:
            json.dump(valid, f)
        print(f'FEEDBACK CLEANUP: removed {removed} untrusted-era records, kept {len(valid)}')
    else:
        print(f'FEEDBACK: {len(raw)} trades, all clean')
"`, { timeout: 5000 }).then((r: any) => {
        if (r.stdout?.trim()) audit("SYSTEM", r.stdout.trim());
      }).catch(() => {});
    } catch (_) {}

    audit("SYSTEM", "=== VolTradeAI Bot v2.0 Initialized ===");
    audit("SYSTEM", `Mode: PAPER TRADING (Alpaca)`);
    audit("RULES", `Scan interval: 45 seconds during trading hours`);
    audit("RULES", `Max positions: ${MAX_POSITIONS} ceiling (dynamic sizing uses portfolio heat)`);
    audit("RULES", `Position sizing: DYNAMIC — Kelly Criterion × volatility × confidence × VIX × earnings × liquidity × time`);
    audit("RULES", `Position range: 1-${MAX_POSITION_SIZE * 100}% per trade (sized by conviction + risk)`);
    audit("RULES", `Max active satellite exposure: ${MAX_TOTAL_EXPOSURE * 100}% (QQQ floor + VRP excluded) | Total portfolio cap: 95% (regime-adaptive)`);
    audit("RULES", `Stop loss: ATR-based (1.5-8% dynamic) | Emergency backstop: ${STOP_LOSS_PCT * 100}%`);
    audit("RULES", `Take profit: ATR-based (4-15% dynamic) | Ceiling: ${TAKE_PROFIT_PCT * 100}%`);
    audit("RULES", `Earnings proximity: auto-reduces position size near earnings dates`);
    audit("RULES", `Exchange halt check: blocks orders on halted stocks`);
    audit("RULES", `Daily loss limit: ${DAILY_LOSS_LIMIT}% — all trading halts if hit`);
    audit("RULES", `Kill switch: OFF (manual emergency stop available)`);
    audit("RULES", `Options: Regular hours only (9:30am-4pm ET)`);
    audit("RULES", `Stocks: Extended hours OK (4am-8pm ET, limit orders only)`);
    audit("RULES", `Extended hours: No fractional shares, no market orders, no short selling`);
    audit("RULES", `PDT: If account < $25K, max 3 day trades per 5 business days`);
    audit("RULES", `Minimum score to trade: 65/100 combined edge score`);
    audit("RULES", `Minimum stock price: $5 | Minimum volume: 500K daily`);
    audit("RULES", `Strategy scoring: Momentum 25%, VRP 25%, MeanRev 20%, Squeeze 15%, Volume 15%`);
    audit("RULES", `Correlation check: Max ${2} stocks per sector`);
    audit("RULES", `Sell/Short: Negative momentum + bearish sentiment → SHORT | High VRP → SELL OPTIONS`);
    audit("EXECUTION", `Smart Execution: Extended hours → queue for morning, no chasing thin liquidity`);
    audit("EXECUTION", `Smart Execution: Regular hours → market orders for instant fill`);
    audit("EXECUTION", `Smart Execution: Stale order sweeper — cancel unfilled limits after ${STALE_ORDER_MINUTES} min`);
    audit("EXECUTION", `Smart Execution: Score-based replacement — better picks replace weaker unfilled orders`);
    audit("EXECUTION", `Smart Execution: Morning queue — overnight research executed at 9:30am market open`);
    audit("SCHEDULE", `4am-9:30am ET: Pre-market research → queue for market open`);
    audit("SCHEDULE", `9:30am-4pm ET: Full trading (market orders + options)`);
    audit("SCHEDULE", `4pm-8pm ET: After-hours research → queue for next open`);
    audit("SCHEDULE", `8pm-10pm ET: Evening research — analyze today's movers`);
    audit("SCHEDULE", `10pm-12am ET: Backtest validation — check strategies still work`);
    audit("SCHEDULE", `12am-2am ET: Earnings analysis — upcoming reporters`);
    audit("SCHEDULE", `2am-4am ET: Pre-market report — compile morning trade list`);

    alpaca("/v2/clock").then((clock: any) => {
      const etH3 = getETHour();
      const canTrade = clock.is_open || (etH3 >= 4 && etH3 < 20);

      // Always start real-time streaming (works during and after market hours)
      setTimeout(() => startStreaming(), 3000);
      audit("STREAM", "Real-time WebSocket feed starting...");

      if (canTrade) {
        audit("SYSTEM", "Market is in a trading window. Tier 2 scan starting...");
        // Tier 2/3 startup timeouts already scheduled above
      } else {
        const nextOpen = clock.next_open ? new Date(clock.next_open).toLocaleString("en-US", { timeZone: "America/New_York" }) : "unknown";
        audit("SYSTEM", `Market is closed. No trading until next session. Next open: ${nextOpen} ET`);
        audit("SYSTEM", "Bot is idle. Will auto-scan when market opens.");
      }
    }).catch(() => {
      audit("SYSTEM", "Could not check market status. Will retry on next cycle.");
    });
  }, 5000);

  // Route: Force immediate scan (respects the concurrency lock)
  app.post("/api/bot/run-now", requireOwner, async (_req, res) => {
    if (tier2Running) return res.json({ message: "Tier 2 scan already running..." });
    tier2Running = true;  // Lock BEFORE responding to prevent double-triggers
    res.json({ message: "Tier 2 intelligence scan starting..." });
    alpaca("/v2/clock").then(async (clock: any) => {
      const etH = getETHour();
      try { await tier2Intelligence(clock.is_open, etH); } finally { tier2Running = false; }
    }).catch(async () => {
      try { await tier2Intelligence(false, getETHour()); } finally { tier2Running = false; }
    });
  });

  // ── Self-Diagnostic Status ───────────────────────────────────────
  app.get("/api/bot/diagnostics", requireOwner, async (_req, res) => {
    try {
      const { stdout } = await execPythonSerialized(`python3 -c "
from diagnostics import run_diagnostics, get_auto_fix_params
import json
report = run_diagnostics()
params = get_auto_fix_params()
print(json.dumps({'report': report, 'auto_fix': params}))
"`, { timeout: 15000 });
      res.json(JSON.parse(stdout.trim()));
    } catch (e: any) {
      res.json({ error: e.message });
    }
  });


  // ── ML Model Status ───────────────────────────────────────────────────────
  app.get("/api/bot/ml-status", requireOwner, async (_req, res) => {
    try {
      const { stdout } = await execPythonSerialized(
        `python3 -c "
import json, os, time
try:
    from storage_config import ML_MODEL_PATH, TRADE_FEEDBACK_PATH, WEIGHTS_PATH
except ImportError:
    ML_MODEL_PATH = '/tmp/voltrade_ml_model.pkl'
    TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'
    WEIGHTS_PATH = '/tmp/voltrade_weights.json'
from ml_model_v2 import fills_slippage_stats
status = {'model_exists': os.path.exists(ML_MODEL_PATH), 'retrain_schedule': 'Daily at 4am ET', 'retrain_threshold_hours': 24}
if status['model_exists']:
    status['model_age_hours'] = round((time.time() - os.path.getmtime(ML_MODEL_PATH)) / 3600, 1)
    status['needs_retrain'] = status['model_age_hours'] > 24
if os.path.exists(TRADE_FEEDBACK_PATH):
    with open(TRADE_FEEDBACK_PATH) as f:
        _feedback = json.load(f)
    status['total_fills'] = fills_slippage_stats(_feedback)['count']
else:
    status['total_fills'] = 0
if os.path.exists(WEIGHTS_PATH):
    with open(WEIGHTS_PATH) as f:
        status['learned_weights'] = json.load(f)
print(json.dumps(status))
"`, { timeout: 10000 }
      );
      res.json(JSON.parse(stdout.trim()));
    } catch (e: any) {
      res.json({ error: e.message });
    }
  });

}
