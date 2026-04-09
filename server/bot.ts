import { Express } from "express";
import { requireAuth } from "./auth";
import { exec, execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";
import WebSocket from "ws";
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
const execAsync = (cmd: string, opts?: any) => _execRaw(cmd, { env: _pyEnv, ...opts });

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
    return r.json();
  } finally {
    clearTimeout(timeout);
  }
}

// ─── Bot State ──────────────────────────────────────────────────────────────
interface AuditEntry { time: string; action: string; detail: string; }

// Phase 6: Security guardrails (absolute ceilings — dynamic sizing handles the real math)
const DAILY_LOSS_LIMIT = -3; // percent — hard circuit breaker
const MAX_POSITION_SIZE = 0.20; // Safety ceiling — real sizing from system_config.py (3-15%)
const MAX_TOTAL_EXPOSURE = 0.30; // 30% of equity for ACTIVE trades (excludes QQQ floor + third leg)
const MAX_POSITIONS = 8; // absolute ceiling (dynamic sizing uses portfolio heat)
const STOP_LOSS_PCT = 0.15; // Emergency backstop only — real stops from system_config.py (6% ATR-based)
const TAKE_PROFIT_PCT = 0.25; // Emergency ceiling — real TP from system_config.py (12% ATR-based)
// NOTE: These are EMERGENCY SAFETY NETS only. The real limits come from
// system_config.py's get_adaptive_params() which adapts by regime:
//   BULL: 90% max exposure, 15% position size, 12% TP, 6% SL
//   NEUTRAL: 90% (QQQ floor handles it), no active stock trades
//   CAUTION: 60% exposure, 10% position size
//   BEAR: 50% exposure (third leg only)
//   PANIC: 30% exposure
// The QQQ floor deploys 70-90% of equity — these safety nets MUST
// be above that or they block the core strategy.

// ─── ET Hour Helper (DST-aware) ───────────────────────────────────────────────
function getETHour(): number {
  // Proper ET that handles DST automatically (EST = UTC-5, EDT = UTC-4)
  const now = new Date();
  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  return et.getHours() + et.getMinutes() / 60;
}

// ─── Market Hours Helper ──────────────────────────────────────────────────────
type OrderContext = 'stop_loss' | 'trailing_stop' | 'take_profit' | 'new_entry' | 'options_entry' | 'options_exit';

function getOrderParams(
  price: number,
  context: OrderContext = 'new_entry'
): { type: string; limit_price?: string; time_in_force: string } {
  const etTime = getETHour();
  const isRegularHours = etTime >= 9.5 && etTime < 16.0;

  // Options: ALWAYS limit, no exceptions (wide bid-ask spreads)
  if (context === 'options_entry' || context === 'options_exit') {
    const limitPrice = Math.round(price * 100) / 100;
    return { type: "limit", limit_price: String(limitPrice), time_in_force: "day" };
  }

  if (isRegularHours) {
    switch (context) {
      case 'stop_loss':
      case 'trailing_stop':
        // Speed matters — get out NOW. A limit that doesn't fill while price drops is catastrophic.
        return { type: "market", time_in_force: "day" };
      case 'take_profit': {
        // Not in a rush — want the exact target price
        const tpPrice = Math.round(price * 100) / 100;
        return { type: "limit", limit_price: String(tpPrice), time_in_force: "day" };
      }
      case 'new_entry':
      default: {
        // Limit at ask + 0.1% — fill priority while capping worst case
        const entryPrice = Math.round(price * 1.001 * 100) / 100;
        return { type: "limit", limit_price: String(entryPrice), time_in_force: "day" };
      }
    }
  } else {
    // Extended hours (4am-9:30am, 4pm-8pm ET): Alpaca requires limit orders
    switch (context) {
      case 'stop_loss':
      case 'trailing_stop': {
        // Bid - 0.5% to ensure fill in thin liquidity
        const stopPrice = Math.round(price * 0.995 * 100) / 100;
        return { type: "limit", limit_price: String(stopPrice), time_in_force: "day" };
      }
      case 'take_profit': {
        const tpPrice = Math.round(price * 100) / 100;
        return { type: "limit", limit_price: String(tpPrice), time_in_force: "day" };
      }
      case 'new_entry':
      default: {
        // Ask + 0.5% — wider buffer for thinner extended hours liquidity
        const entryPrice = Math.round(price * 1.005 * 100) / 100;
        return { type: "limit", limit_price: String(entryPrice), time_in_force: "day" };
      }
    }
  }
}

const state = {
  active: true,  // Bot starts automatically — always on unless killed
  killSwitch: false,
  dailyPnL: 0,
  dailyLossLimit: DAILY_LOSS_LIMIT,
  auditLog: [] as AuditEntry[],
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
  state.auditLog.unshift(entry);
  if (state.auditLog.length > 500) state.auditLog.length = 500;
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
const equityCurve: Array<{ date: string; value: number; pnl: number }> = [];

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

    // Adjust strategy weights based on recent performance
    adjustStrategyWeights();

    // Self-improving: feed closed trades back to ML training data
    if (tradeResults.length > 0) {
      try {
        const feedbackData = tradeResults.slice(0, 20).map(t => ({
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
        }));
        const fbTmpPath = `/tmp/fb_${Date.now()}.json`;
        fs.writeFileSync(fbTmpPath, JSON.stringify(feedbackData));
        execAsync(`python3 -c "
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
function adjustStrategyWeights() {
  // Simple heuristic: if recent win rate < 40%, reduce weight on low-confidence signals
  const recent = tradeResults.slice(0, 20);
  if (recent.length < 5) return;

  const winRate = recent.filter(t => t.pnl > 0).length / recent.length;
  if (winRate < 0.4) {
    // Adjust weights silently every cycle, but only LOG once per 30 minutes
    strategyWeights.vrp = Math.min(0.40, strategyWeights.vrp + 0.02);
    strategyWeights.momentum = Math.max(0.15, strategyWeights.momentum - 0.01);
    strategyWeights.volume = Math.max(0.10, strategyWeights.volume - 0.01);
    if (Date.now() - lastWeightAdjustLog > 1800000) { // 30 minutes
      audit("LEARN", `Win rate ${(winRate * 100).toFixed(0)}% — shifting weight toward VRP/squeeze (momentum: ${(strategyWeights.momentum * 100).toFixed(0)}%, VRP: ${(strategyWeights.vrp * 100).toFixed(0)}%)`);
      lastWeightAdjustLog = Date.now();
    }
  } else if (winRate > 0.65) {
    strategyWeights.momentum = Math.min(0.30, strategyWeights.momentum + 0.01);
    if (Date.now() - lastWeightAdjustLog > 1800000) {
      audit("LEARN", `Win rate ${(winRate * 100).toFixed(0)}% — restoring balanced weights (momentum: ${(strategyWeights.momentum * 100).toFixed(0)}%)`);
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

    const { stdout, stderr } = await execAsync(
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

function persistAudit(type: string, message: string) {
  try {
    db.prepare("INSERT INTO audit_log (time, type, message) VALUES (?, ?, ?)").run(
      new Date().toISOString(), type, message.slice(0, 500)
    );
    // Keep last 2000 entries
    db.prepare("DELETE FROM audit_log WHERE id NOT IN (SELECT id FROM audit_log ORDER BY id DESC LIMIT 2000)").run();
  } catch {}
}

function getPersistedAuditLog(limit = 100): any[] {
  try {
    return db.prepare("SELECT time, type, message FROM audit_log ORDER BY id DESC LIMIT ?").all(limit) as any[];
  } catch { return []; }
}

export function registerBotRoutes(app: Express) {

  // SSE for live bot audit log updates
  app.get("/api/bot/stream", requireAuth, (req, res) => {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    });
    sseClients.add(res);
    req.on("close", () => sseClients.delete(res));
  });

  // Account info from Alpaca
  app.get("/api/bot/account", requireAuth, async (_req, res) => {
    try {
      const acct = await alpaca("/v2/account");
      const equity = parseFloat(acct.equity);
      const lastEquity = parseFloat(acct.last_equity);
      const dailyPnL = equity - lastEquity;
      // Update bot state with current daily P&L
      state.dailyPnL = dailyPnL;

      // ── Max Drawdown Kill Switch ──
      if (state.equityPeak === 0) state.equityPeak = equity;
      if (equity > state.equityPeak) state.equityPeak = equity;
      const drawdownPct = ((equity - state.equityPeak) / state.equityPeak) * 100;
      if (drawdownPct <= state.maxDrawdownPct && !state.killSwitch) {
        state.killSwitch = true;
        state.active = false;
        const msg = `MAX DRAWDOWN KILL SWITCH: Equity $${equity.toFixed(0)} is ${drawdownPct.toFixed(1)}% below peak $${state.equityPeak.toFixed(0)}. All trading stopped.`;
        audit("DRAWDOWN-KILL", msg);
        notify("critical", msg);
        sendEmailAlert("MAX DRAWDOWN — Trading Stopped", msg);
        try { await alpaca("/v2/orders", { method: "DELETE" }); } catch {}
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
  app.get("/api/bot/positions", requireAuth, async (_req, res) => {
    try {
      const positions = await alpaca("/v2/positions");
      // Load evolving stop state for stop/TP levels
      const { stdout: stopOut } = await execAsync(`python3 -c "
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
"`, { timeout: 5000 }).catch(() => ({ stdout: "{}" }));
      const stopState: any = JSON.parse(stopOut.trim() || "{}");

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
        return {
          ticker: p.symbol,
          qty: parseFloat(p.qty),
          side: parseFloat(p.qty) > 0 ? "long" : "short",
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
        };
      });
      res.json(mapped);
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // Candlestick bars for trade charts
  // Trade history from Alpaca
  app.get("/api/bot/history", requireAuth, async (_req, res) => {
    try {
      const orders = await alpaca("/v2/orders?status=closed&limit=50");
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
  app.get("/api/bot/notifications", requireAuth, (_req, res) => {
    res.json(notifications);
  });

  app.post("/api/bot/notifications/read", requireAuth, (_req, res) => {
    notifications.forEach(n => (n.read = true));
    res.json({ ok: true });
  });

  // ── Health Check (for Railway and monitoring) ──────────────────────────────
  app.get("/api/health", async (_req, res) => {
    const checks: any = { timestamp: new Date().toISOString(), status: "ok", checks: {} };
    
    // Check 1: Node.js server is running (obviously)
    checks.checks.server = { status: "ok" };
    
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
      const { stdout } = await execAsync('python3 -c "print(\'ok\')"', { timeout: 5000 });
      checks.checks.python = { status: stdout.trim() === "ok" ? "ok" : "error" };
    } catch (err: any) {
      checks.checks.python = { status: "error", detail: err?.message };
      checks.status = "degraded";
    }
    
    // Check 5: Bot state
    checks.checks.bot = {
      status: state.killSwitch ? "killed" : state.active ? "active" : "stopped",
      equityPeak: state.equityPeak,
      drawdownPct: state.equityPeak > 0 ? (((state.equityPeak - parseFloat(state.lastEquity || String(state.equityPeak))) / state.equityPeak) * 100).toFixed(1) : "N/A",
    };
    
    const httpCode = checks.status === "ok" ? 200 : 503;
    res.status(httpCode).json(checks);
  });

  // ── Performance Dashboard Data ────────────────────────────────────────────
  app.get("/api/bot/performance", requireAuth, async (_req, res) => {
    try {
      // Get trade history from fills
      const { stdout: fillsOut } = await execAsync(`python3 -c "
import json, os
try:
    from storage_config import FILLS_PATH, TRADE_FEEDBACK_PATH
except ImportError:
    FILLS_PATH = '/tmp/voltrade_fills.json'
    TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'

fills = []
if os.path.exists(FILLS_PATH):
    with open(FILLS_PATH) as f: fills = json.load(f)

feedback = []
if os.path.exists(TRADE_FEEDBACK_PATH):
    with open(TRADE_FEEDBACK_PATH) as f: feedback = json.load(f)

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

# Realistic P&L (with slippage from fills tracker)
total_slippage_cost = 0
for f in fills:
    slip = f.get('slippage_pct', 0) or 0
    expected = f.get('expected_price', 0) or 0
    qty = f.get('qty', 0) or 0
    if expected > 0 and qty > 0:
        total_slippage_cost += expected * qty * slip / 100

# Estimated realistic P&L = paper P&L minus slippage drag
realistic_total_pnl = total_pnl
if len(fills) > 0:
    avg_slippage_pct = sum(f.get('slippage_pct', 0) or 0 for f in fills) / len(fills)
    # Each trade has entry + exit slippage
    realistic_total_pnl = total_pnl - (avg_slippage_pct * 2 * len(feedback) / 100 * 100)
else:
    avg_slippage_pct = 0

print(json.dumps({
    'total_trades': len(feedback),
    'total_fills': len(fills),
    'win_rate': round(win_rate, 1),
    'avg_win_pct': round(avg_win, 2),
    'avg_loss_pct': round(avg_loss, 2),
    'total_pnl_pct': round(total_pnl, 2),
    'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
    'by_strategy': by_strat,
    'recent_trades': feedback[-20:][::-1],
    'realistic_pnl_pct': round(realistic_total_pnl, 2),
    'avg_slippage_pct': round(avg_slippage_pct, 4),
    'total_slippage_cost': round(total_slippage_cost, 2),
    'slippage_gap_pct': round(total_pnl - realistic_total_pnl, 2),
}))
"`, { timeout: 10000 });
      
      const perf = JSON.parse(fillsOut.trim());
      res.json({
        ...perf,
        equityCurve,
        equityPeak: state.equityPeak,
        currentDrawdown: state.equityPeak > 0 ? ((equityCurve[0]?.value || 0) - state.equityPeak) / state.equityPeak * 100 : 0,
      });
    } catch (err: any) {
      res.json({ error: err?.message, total_trades: 0, equityCurve });
    }
  });

  // ── Trade History CSV Export ─────────────────────────────────────────────
  app.get("/api/bot/export-trades", requireAuth, async (_req, res) => {
    try {
      const { stdout } = await execAsync(`python3 -c "
import json, os, csv, io
try:
    from storage_config import TRADE_FEEDBACK_PATH, FILLS_PATH
except ImportError:
    TRADE_FEEDBACK_PATH = '/tmp/voltrade_trade_feedback.json'
    FILLS_PATH = '/tmp/voltrade_fills.json'

feedback = []
if os.path.exists(TRADE_FEEDBACK_PATH):
    with open(TRADE_FEEDBACK_PATH) as f: feedback = json.load(f)

fills = []
if os.path.exists(FILLS_PATH):
    with open(FILLS_PATH) as f: fills = json.load(f)

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
  app.post("/api/bot/backup", requireAuth, async (_req, res) => {
    try {
      const { stdout } = await execAsync(`python3 -c "
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

  // Bot status
  app.get("/api/bot/status", requireAuth, (_req, res) => {
    res.json({
      active: state.active,
      killSwitch: state.killSwitch,
      dailyLossLimit: state.dailyLossLimit,
      auditLogCount: state.auditLog.length,
      mode: "paper",
      equityPeak: state.equityPeak,
      maxDrawdownPct: state.maxDrawdownPct,
      unreadNotifications: notifications.filter(n => !n.read).length,
    });
  });

  // Start bot
  app.post("/api/bot/start", requireAuth, (_req, res) => {
    if (state.killSwitch) return res.status(400).json({ error: "Kill switch is ON. Disable it first." });
    state.active = true;
    audit("START", "Bot activated");
    notify("system", "Bot activated — scanning for opportunities");
    // Start real-time streaming feed
    setTimeout(() => startStreaming(), 2000);
    res.json({ ok: true, active: true });
  });

  // Stop bot
  app.post("/api/bot/stop", requireAuth, (_req, res) => {
    state.active = false;
    stopStreaming();
    audit("STOP", "Bot deactivated");
    notify("system", "Bot paused");
    res.json({ ok: true, active: false });
  });

  // Kill switch
  app.post("/api/bot/kill", requireAuth, async (_req, res) => {
    state.killSwitch = !state.killSwitch;
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

  // Audit log
  app.get("/api/bot/audit", requireAuth, (_req, res) => {
    // Return in-memory log, fall back to persistent DB log if empty (e.g. after redeploy)
    if (state.auditLog.length > 0) {
      res.json(state.auditLog.slice(0, 100));
    } else {
      const persisted = getPersistedAuditLog(100);
      res.json(persisted.map((e: any) => ({ time: e.time, action: e.type, type: e.type, detail: e.message, message: e.message })));
    }
  });

  // Place a trade (manual or from bot signals)
  app.post("/api/bot/trade", requireAuth, async (req, res) => {
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
      const orderSide = side === "short" ? "sell" : side;
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
  app.post("/api/bot/close", requireAuth, async (req, res) => {
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
  app.get("/api/bot/bars/:ticker", requireAuth, async (req, res) => {
    const { ticker } = req.params;
    const timeframe = (String(req.query.timeframe || "1Day")) || "1Day";
    const limit = parseInt(String(req.query.limit || "200")) || 200;

    // Map timeframe to Alpaca format
    const tfMap: Record<string, string> = {
      "1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "1Hour": "1Hour",
      "1Day": "1Day", "1Week": "1Week",
    };
    const tf = tfMap[timeframe] || "1Day";

    try {
      const url = `https://data.alpaca.markets/v2/stocks/${String(ticker).toUpperCase()}/bars?timeframe=${tf}&limit=${limit}&adjustment=split&feed=sip`;
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
        const { stdout } = await execAsync(
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
  app.get("/api/bot/signals", requireAuth, (_req, res) => {
    res.json(signals);
  });

  // Manual signal refresh
  app.post("/api/bot/refresh-signals", requireAuth, async (_req, res) => {
    const topTickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "AMD", "GOOGL"];
    audit("REFRESH", "Generating fresh signals for top 10 tickers...");
    await generateSignals(topTickers);
    res.json({ ok: true, count: signals.length });
  });

  // ── Backtesting ───────────────────────────────────────────────────────────
  app.post("/api/bot/backtest", requireAuth, async (req, res) => {
    const { ticker = "SPY", strategy = "all", years = 3 } = req.body || {};
    const scriptPath = path.resolve("backtest.py");

    try {
      audit("BACKTEST", `Running ${strategy} on ${ticker} (${years}yr)`);
      const { stdout } = await execAsync(
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
      } catch (e: any) {
        // Order may have already filled or been cancelled
        const idx = openOrders.findIndex(o => o.orderId === stale.orderId);
        if (idx >= 0) openOrders.splice(idx, 1);
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
    const equity = parseFloat(acct.equity || "100000");

    // Max drawdown check on every Tier 1 cycle
    if (state.equityPeak === 0) state.equityPeak = equity;
    if (equity > state.equityPeak) state.equityPeak = equity;
    const t1Drawdown = ((equity - state.equityPeak) / state.equityPeak) * 100;
    if (t1Drawdown <= state.maxDrawdownPct && !state.killSwitch) {
      state.killSwitch = true;
      state.active = false;
      const msg = `MAX DRAWDOWN KILL SWITCH: Equity $${equity.toFixed(0)} is ${t1Drawdown.toFixed(1)}% below peak $${state.equityPeak.toFixed(0)}`;
      audit("DRAWDOWN-KILL", msg);
      sendEmailAlert("MAX DRAWDOWN — Trading Stopped", msg);
      try { await alpaca("/v2/orders", { method: "DELETE" }); } catch {}
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
        const { stdout: priceCheck } = await execAsync(
          `python3 -c "import requests,json,os; r=requests.get('https://data.alpaca.markets/v2/stocks/${trade.ticker}/snapshot?feed=sip', headers={'APCA-API-KEY-ID':os.environ.get('ALPACA_KEY',''),'APCA-API-SECRET-KEY':os.environ.get('ALPACA_SECRET','')}, timeout=5); d=r.json(); print(d.get('dailyBar',{}).get('c',0) or d.get('latestTrade',{}).get('p',0))"` ,
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

      try {
        const order = await alpaca("/v2/orders", {
          method: "POST",
          body: JSON.stringify({
            symbol: trade.ticker,
            qty: String(Math.floor(trade.shares)),
            side: trade.side === "short" ? "sell" : (trade.side || "buy"),
            ...getOrderParams(trade.price || 0), // Market during regular hours, limit during extended
          }),
        });

        audit("MORNING-TRADE", `MARKET ${(trade.side || "BUY").toUpperCase()} ${Math.floor(trade.shares)} ${trade.ticker} @ market | Score: ${trade.score} | Queued from overnight research`);
        notify("trade", `Morning queue: ${(trade.side || "BUY").toUpperCase()} ${Math.floor(trade.shares)} ${trade.ticker} (score: ${trade.score})`);

        // Track fill with realistic slippage (morning queue = market open = wider slippage)
        try {
          const mVol = trade.volume || 1000000;
          let mSlip = 0.08; // Morning fills are worse (opening volatility)
          if (mVol > 20000000) mSlip = 0.05 + Math.random() * 0.05;
          else if (mVol > 5000000) mSlip = 0.08 + Math.random() * 0.07;
          else mSlip = 0.15 + Math.random() * 0.10;
          const mSide = trade.side === "short" ? "sell" : (trade.side || "buy");
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
          };
          const mfTmp = `/tmp/fill_m_${trade.ticker}_${Date.now()}.json`;
          fs.writeFileSync(mfTmp, JSON.stringify(morningFillPayload));
          execAsync(`python3 -c "import json, os; from ml_model_v2 import track_fill; d=json.load(open('${mfTmp}')); os.remove('${mfTmp}'); track_fill(d)"`, { timeout: 5000 }).catch(() => {});
        } catch (err: any) { console.error("[bot]", err?.message || err); }

        slotsUsed++;
        // Auto-subscribe for real-time exit monitoring
        addPositionToMonitor(trade.ticker, trade.side === "short" ? "short" : "long", trade.price || 0, Math.floor(trade.shares));
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
        const { stdout } = await execAsync(`python3 -c "
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

    } catch (err: any) {
      console.error("[tier1]", err?.message || err);
    }
  }

  // ── Tier 2: Intelligence — find and execute trades (5 min) ────────────────

  async function tier2Intelligence(isMarketOpen: boolean, etHour: number) {
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
        const { stdout: diagOut } = await execAsync(`python3 -c "
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
      const { stdout, stderr } = await execAsync(`python3 -W ignore "${enginePath}" full`, { timeout: 300000 }); // 5 min timeout
      // Robust JSON extraction: find the first '{' to skip any warning/debug text before JSON
      const cleanStdout = stdout.replace(/\r/g, '').trim();
      const jsonStart = cleanStdout.indexOf('{');
      if (jsonStart === -1) throw new Error(`No JSON in output. stdout: ${cleanStdout.slice(0, 200)} stderr: ${(stderr || '').slice(0, 200)}`);
      const result = JSON.parse(cleanStdout.slice(jsonStart));

      if (!result || result.error) {
        audit("TIER2", `Scan returned error: ${result?.error || "unknown"}`);
        return;
      }

      audit("TIER2", `Scanned ${result.scanned || 0} stocks, ${(result.new_trades || []).length} trade candidates`);

      lastScanResult = result;

      // Auto-generate signals from scan results
      const topPicks = result.top_10 || [];
      for (const pick of topPicks) {
        if (pick.score >= 60) {
          const existing = signals.findIndex((s: any) => s.ticker === pick.ticker);
          if (existing >= 0) signals.splice(existing, 1);
          const side = pick.side || "buy";
          const actionLabel = pick.action_label || (pick.score >= 75 ? "STRONG BUY" : pick.score >= 65 ? "BUY" : "WATCH");
          signals.unshift({
            ticker: pick.ticker,
            action: actionLabel,
            reason: (pick.reasons || []).join(" | ") || `Score: ${pick.score}/100`,
            confidence: Math.min(95, pick.score),
            timestamp: new Date().toISOString(),
            type: side === "short" ? "sell" : side === "sell" ? "sell" : "buy",
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
      if (signals.length > 30) signals.length = 30;

      // 4. Check daily/weekly limits
      const acct = await alpaca("/v2/account");
      const equity = parseFloat(acct.equity || "100000");
      const lastEquity = parseFloat(acct.last_equity || "100000");
      const dailyPnlPct = ((equity - lastEquity) / lastEquity) * 100;

      if (dailyPnlPct <= -3) {
        audit("TIER2-LIMIT", `Daily loss limit: ${dailyPnlPct.toFixed(1)}%`);
        notify("alert", `Daily loss: ${dailyPnlPct.toFixed(1)}%. Trading halted.`);
        return;
      }

      // Weekly loss check
      try {
        const { stdout: weeklyOut } = await execAsync(`python3 -c "
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
          const { stdout: shortsOut } = await execAsync(
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

    } catch (err: any) {
      console.error("[tier2-scan]", err?.message || err);
      audit("TIER2-ERROR", `Scan failed: ${String(err?.message || err).slice(0, 200)}`);
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
    // Exclude floor + third-leg positions from active trading heat check.
    const MANAGED_TICKERS = new Set(["QQQ", "GLD", "ITA", "SVXY", "SPY", "XOM", "LMT"]);
    const stockPositions = Array.isArray(positions)
      ? positions.filter((p: any) => (p.asset_class || "us_equity") === "us_equity").length
      : held.length;
    const optionsPositions = Array.isArray(positions)
      ? positions.filter((p: any) => p.asset_class === "us_option").length
      : 0;
    const MAX_OPTIONS_POSITIONS = 3;  // Separate slots for options
    let slotsUsed = stockPositions;  // Only count stocks against MAX_POSITIONS
    let optionsSlotsUsed = optionsPositions;
    let totalDeployed = Array.isArray(positions)
      ? positions.reduce((sum: number, p: any) => {
          if (MANAGED_TICKERS.has(p.symbol)) return sum; // Don't count floor/leg positions
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

      // Exchange halt check (via position_sizing.py)
      try {
        const { stdout: haltOut } = await execAsync(
          `python3 -c "from position_sizing import check_halt_status; import json; print(json.dumps(check_halt_status('${trade.ticker}')))"`,
          { timeout: 8000 }
        );
        const haltResult = JSON.parse(haltOut.trim());
        if (haltResult.halted) {
          audit("HALT-SKIP", `${trade.ticker}: stock is halted (status: ${haltResult.status})`);
          continue;
        }
      } catch (err: any) {
        // If halt check fails, proceed cautiously
        audit("HALT-CHECK-WARN", `${trade.ticker}: halt check failed, proceeding`);
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
                symbol: etfTicker, qty: String(etfShares), side: trade.side || "buy",
                ...etfOrderParams,
              }),
            });
            audit("ETF-TRADE", `${trade.side?.toUpperCase() || 'BUY'} ${etfShares} ${etfTicker} (2x ETF for ${trade.ticker}) | Score: ${trade.score} | ${trade.instrument_reasoning?.slice(0, 120)}`);
            notify("trade", `ETF: ${etfShares} ${etfTicker} (2x ${trade.ticker})`);
            slotsUsed++;
            totalDeployed += etfShares * trade.price; // Approximate
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
              const { stdout: scanResult } = await execAsync(
                `python3 -c "
import json, sys; sys.path.insert(0, '.')
from options_execution import select_contract, submit_options_order
data = json.load(open('${scanTmpPath}'))
contract = select_contract(data['ticker'], data['strategy'], data['price'], data['equity'])
if contract.get('error'):
    print(json.dumps({'instrument':'stock','order':None,'contract':None,'reasoning':contract['error']}))
else:
    order = submit_options_order(contract)
    if order.get('status') in ('submitted','filled'):
        try:
            from options_manager import register_options_entry
            register_options_entry(contract.get('occ_symbol',''), contract.get('limit_price',0), contract.get('side','buy'), contract.get('delta',0), contract.get('qty',1))
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
              const { stdout: optResult } = await execAsync(
                `python3 -c "import json; from options_execution import evaluate_and_execute; d=json.load(open('${tmpPath}')); print(json.dumps(evaluate_and_execute(d['trade'], d['equity'], d['positions'])))"`,
                { timeout: 30000 }
              );
              try { fs.unlinkSync(tmpPath); } catch (_) {}
              optExec = JSON.parse(optResult.trim());
            }

            if (optExec.instrument === "options" && optExec.order?.status === "submitted") {
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
        const isShort = trade.side === "short";
        const side = isShort ? "sell" : (trade.side || "buy");

        // Short stock: verify easy-to-borrow before submitting
        if (isShort) {
          try {
            const assetInfo = await alpaca(`/v2/assets/${trade.ticker}`);
            if (!assetInfo.easy_to_borrow) {
              audit("SHORT-SKIP", `${trade.ticker}: not easy to borrow — routing to puts`);
              continue; // Skip — options path should have handled this
            }
          } catch (borrowErr: any) {
            audit("SHORT-SKIP", `${trade.ticker}: borrow check failed — skipping short`);
            continue;
          }
        }

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
        addPositionToMonitor(trade.ticker, isShort ? "short" : "long", trade.price || 0, qty);

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
          };
          const rfTmp = `/tmp/fill_r_${pending.trade.ticker}_${Date.now()}.json`;
          fs.writeFileSync(rfTmp, JSON.stringify(fillPayload));
          execAsync(`python3 -c "import json, os; from ml_model_v2 import track_fill; d=json.load(open('${rfTmp}')); os.remove('${rfTmp}'); track_fill(d)"`, { timeout: 5000 }).catch(() => {});
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
      const { stdout: modelCheck } = await execAsync(`python3 -c "
import os, time, json
path = '/data/voltrade_ml_model.pkl' if os.path.isdir('/data') else '/tmp/voltrade_ml_model.pkl'
if not os.path.exists(path) or (time.time() - os.path.getmtime(path)) > 86400:
    print(json.dumps({'needs_retrain': True}))
else:
    age_h = (time.time() - os.path.getmtime(path)) / 3600
    print(json.dumps({'needs_retrain': False, 'age_hours': round(age_h, 1)}))
"`, { timeout: 5000 });
      const modelStatus = JSON.parse(modelCheck.trim());
      // ML auto-retrain disabled (v1.0.30) — ML doesn't contribute to CAGR yet.
      // Re-enable when training pipeline is optimized for Railway's constraints.
      audit("TIER3", "ML retrain: skipped (disabled until pipeline optimized)");
    } catch (err: any) { console.error("[tier3-ml]", err?.message || err); }

    // 2. Manipulation detection scan
    try {
      const { stdout: manipOut } = await execAsync(`python3 -c "
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
    } catch (err: any) { console.error("[tier3-manip]", err?.message || err); }

    // 3. Run full diagnostics
    try {
      const { stdout: diagFull } = await execAsync(`python3 -c "
from diagnostics import run_diagnostics
import json
print(json.dumps(run_diagnostics()))
"`, { timeout: 15000 });
      const diagReport = JSON.parse(diagFull.trim());
      if (diagReport.overall_status !== "healthy") {
        audit("TIER3-DIAG", `System health: ${diagReport.overall_status} — ${diagReport.problems?.length || 0} issues`);
      }
    } catch (err: any) { console.error("[tier3-diag]", err?.message || err); }

    // 4. Record daily equity
    recordDailyEquity();

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
              body: JSON.stringify({ symbol: betterPick.ticker, qty: String(upgradeQty), side: betterPick.side === "short" ? "sell" : "buy", ...getOrderParams(betterPick.price || 0) }),
            });
            addPositionToMonitor(betterPick.ticker, betterPick.side === "short" ? "short" : "long", betterPick.price || 0, upgradeQty);
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
        const { stdout: scanOut3 } = await execAsync(`python3 -W ignore "${enginePath3}" full 2>/dev/null`, { timeout: 300000 });
        const jsonStart3 = scanOut3.indexOf('{');
        if (jsonStart3 !== -1) {
          lastScanResult = JSON.parse(scanOut3.slice(jsonStart3));
          audit("RESEARCH", `Scanned ${lastScanResult.scanned || 0} stocks — top picks identified for morning`);
        }

        // Process extreme movers from today — analyze for tomorrow's entry
        try {
          const { stdout: emOut } = await execAsync(`python3 -c "
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
                const { stdout: deepOut } = await execAsync(
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
            await execAsync(`python3 -c "
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
          const { stdout: btOut } = await execAsync(
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
            const { stdout: aOut } = await execAsync(
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
          const { stdout: scanOut } = await execAsync(`python3 "${enginePath4}" scan`, { timeout: 180000 });
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
        // Clear daily blocked tickers and stop counters for new trading day
        (state as any).dailyBlockedTickers = [];
        state.consecutiveStopLosses = 0;
        (state as any).recentStopTickers = [];
        state.morningQueueExecuted = false;
        audit("SYSTEM", "Daily reset: blocked tickers cleared, counters reset for new trading day");
        audit("RETRAIN", "4am daily ML retrain — training on yesterday's data before market open");
        try {
          // ML retrain disabled (v1.0.30) — doesn't affect the 20.3% CAGR.
          // The system runs on regime detection + QQQ floor + VRP + sector rotation.
          audit("RETRAIN", "ML retrain: skipped (disabled until pipeline optimized)");
          if (false) { // Disabled
          const { stdout: trainOut } = await execAsync(
            `python3 ml_retrain_safe.py`,
            { timeout: 120000 }
          );
          const trainResult = JSON.parse(trainOut.trim());
          audit("RETRAIN", `Daily retrain complete — accuracy: ${trainResult.accuracy || 'N/A'}, features: ${trainResult.feature_count || 'N/A'}, samples: ${trainResult.sample_count || 'N/A'}`);
          } // end disabled
        } catch (err: any) {
          audit("RETRAIN-ERROR", `Daily retrain failed: ${err?.message || err}`);
        }

        // Nightly auto-backup at 4am alongside retrain
        try {
          const { stdout: backupOut } = await execAsync(`python3 -c "
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
            const { stdout: gitBackup } = await execAsync(
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
        const { stdout } = await execAsync(`python3 -c "
import json, os
DATA_DIR = '/data/voltrade' if os.path.isdir('/data') else '/tmp'
p = os.path.join(DATA_DIR, 'voltrade_stop_state.json')
try:
    with open(p) as f: print(json.dumps(json.load(f)))
except: print('{}')
"`, { timeout: 5000 });
        stopState = JSON.parse(stdout.trim());
      } catch (_) {}

      const MANAGED_TICKERS = new Set(["QQQ", "SVXY", "ITA", "SPY", "GLD", "XOM", "LMT"]);
      const activeTickers = new Set<string>();

      for (const pos of positions) {
        const ticker = pos.symbol || "";
        if (MANAGED_TICKERS.has(ticker)) continue;

        activeTickers.add(ticker);
        const current = parseFloat(pos.current_price || "0");
        const entry = parseFloat(pos.avg_entry_price || String(current));
        const pnlPct = parseFloat(pos.unrealized_plpc || "0") * 100;
        const qty = Math.abs(parseInt(pos.qty || "0"));
        const side = pos.side === "short" ? "short" as const : "long" as const;

        // Read stop state for this ticker
        const ps = stopState[ticker] || {};
        const atrPct = ps.current_stop_pct ? ps.current_stop_pct : 2.0;
        const initialRiskPct = ps.initial_risk_pct || atrPct * 2.0;
        const highestPnl = Math.max(ps.highest_pnl || 0, pnlPct);
        const peakR = initialRiskPct > 0 ? highestPnl / initialRiskPct : 0;
        const phase = ps.phase || (peakR >= 3 ? 4 : peakR >= 2 ? 3 : peakR >= 1 ? 2 : 1);

        // Compute stop and TP levels
        let stopPct: number;
        if (peakR >= 3) stopPct = highestPnl * 0.50;
        else if (peakR >= 2) stopPct = Math.max(1.0, atrPct * 1.0);
        else if (peakR >= 1) stopPct = Math.max(1.5, atrPct * 1.5);
        else stopPct = Math.max(1.5, Math.min(atrPct * 2.0, 8.0));

        const tpPct = phase < 3 ? Math.max(4.0, Math.min(atrPct * 3.0, 15.0)) : 999;

        monitoredPositions[ticker] = {
          ticker, side, entryPrice: entry, qty, phase,
          initialRiskPct, highestPnl, currentStopPct: stopPct, tpPct,
          entryDate: ps.entry_date || new Date().toISOString().split("T")[0],
          lastCheckedPrice: current,
        };
      }

      // Remove positions that are no longer held
      for (const ticker of Object.keys(monitoredPositions)) {
        if (!activeTickers.has(ticker)) {
          delete monitoredPositions[ticker];
          // Unsubscribe if we added it for monitoring
          if (positionSubscribedTickers.has(ticker)) {
            positionSubscribedTickers.delete(ticker);
            if (streamWs && streamConnected) {
              try { streamWs.send(JSON.stringify({ action: "unsubscribe", bars: [ticker] })); } catch (_) {}
            }
          }
        }
      }

      // Auto-subscribe to position tickers not already in STREAM_TICKERS
      const streamSet = new Set(STREAM_TICKERS);
      const toSubscribe: string[] = [];
      Array.from(activeTickers).forEach((ticker) => {
        if (!streamSet.has(ticker) && !positionSubscribedTickers.has(ticker)) {
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
    };
    // Subscribe to WebSocket if not already streaming
    const streamSet = new Set(STREAM_TICKERS);
    if (!streamSet.has(ticker) && !positionSubscribedTickers.has(ticker)) {
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
   * Extremely lightweight: just compares price to stored levels.
   */
  async function checkPositionOnTick(ticker: string, currentPrice: number) {
    const pos = monitoredPositions[ticker];
    if (!pos || exitingTickers.has(ticker)) return;
    if (!state.active || state.killSwitch) return;

    const entry = pos.entryPrice;
    if (entry <= 0 || currentPrice <= 0) return;

    // Calculate current P&L %
    const pnlPct = pos.side === 'long'
      ? ((currentPrice - entry) / entry) * 100
      : ((entry - currentPrice) / entry) * 100;

    // Update high water mark
    pos.highestPnl = Math.max(pos.highestPnl, pnlPct);
    pos.lastCheckedPrice = currentPrice;

    // Recalculate phase based on updated P&L
    const rMultiple = pos.initialRiskPct > 0 ? pnlPct / pos.initialRiskPct : 0;
    const peakR = pos.initialRiskPct > 0 ? pos.highestPnl / pos.initialRiskPct : 0;

    if (peakR >= 3 && pos.phase < 4) pos.phase = 4;
    else if (peakR >= 2 && pos.phase < 3) pos.phase = 3;
    else if (peakR >= 1 && pos.phase < 2) pos.phase = 2;

    // Recalculate stop level for current phase
    let stopPct: number;
    if (pos.phase >= 4) stopPct = pos.highestPnl * 0.50;
    else if (pos.phase >= 3) stopPct = Math.max(1.0, pos.currentStopPct);
    else if (pos.phase >= 2) stopPct = Math.max(1.5, pos.currentStopPct);
    else stopPct = pos.currentStopPct;

    // Determine if stop is triggered
    let shouldStop = false;
    let stopType: 'stop_loss' | 'trailing_stop' = 'stop_loss';
    let stopReason = '';

    if (pos.phase >= 2) {
      const drawdownFromPeak = pos.highestPnl - pnlPct;
      if (drawdownFromPeak >= stopPct) {
        shouldStop = true;
        stopType = 'trailing_stop';
        stopReason = `WS TRAILING STOP Phase ${pos.phase}: P&L ${pnlPct.toFixed(1)}% dropped ${drawdownFromPeak.toFixed(1)}% from peak ${pos.highestPnl.toFixed(1)}%`;
      }
    } else {
      if (pnlPct <= -stopPct) {
        shouldStop = true;
        stopType = 'stop_loss';
        stopReason = `WS STOP LOSS: ${pnlPct.toFixed(1)}% loss hit Phase 1 stop at -${stopPct.toFixed(1)}%`;
      }
    }

    // Check take-profit
    let shouldTP = false;
    if (!shouldStop && pnlPct >= pos.tpPct && pos.phase < 3) {
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

    // ── FIRE EXIT ORDER IMMEDIATELY ──
    exitingTickers.add(ticker);

    try {
      const exitSide = pos.side === 'long' ? 'sell' : 'buy';
      const exitType = shouldStop ? stopType : (shouldTP ? 'take_profit' : 'time_stop');
      const exitContext: OrderContext = shouldStop ? stopType : (shouldTP ? 'take_profit' : 'stop_loss');
      const orderParams = getOrderParams(currentPrice, exitContext);

      // Submit sell/cover order with context-aware order type
      const orderResult = await alpaca("/v2/orders", {
        method: "POST",
        body: JSON.stringify({
          symbol: ticker,
          qty: String(pos.qty),
          side: exitSide,
          ...orderParams,
        }),
      });

      const reason = shouldStop ? stopReason
        : shouldTP ? `WS TAKE PROFIT: +${pnlPct.toFixed(1)}% hit target +${pos.tpPct.toFixed(1)}% (Phase ${pos.phase})`
        : `WS TIME STOP: held since ${pos.entryDate}, P&L only ${pnlPct.toFixed(1)}%`;

      audit("WS-EXIT", `${reason} | ${orderParams.type.toUpperCase()} ${exitSide} ${pos.qty} ${ticker} @ $${currentPrice.toFixed(2)}`);
      notify("exit", `WS exit ${ticker}: ${reason}`);

      // Write stop-loss cooldown
      if (exitType === 'stop_loss' || exitType === 'trailing_stop') {
        try {
          await execAsync(`python3 -c "
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

        // Circuit breaker logic (same as old Tier 1)
        state.consecutiveStopLosses++;
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
            state.consecutiveStopLosses = 0;
            audit("TICKER-BLOCKED", `${blockedTicker}: 3 stops on same ticker — blocked for today`);
          }
        }
      } else if (exitType === 'take_profit') {
        state.consecutiveStopLosses = 0;
      }

      // Remove from monitoring (position is closed)
      removePositionFromMonitor(ticker);

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

    const ws = new WebSocket("wss://stream.data.alpaca.markets/v2/sip");
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
      streamConnected = false;
      // Auto-reconnect after 10 seconds if bot is still active
      setTimeout(() => { if (state.active && !state.killSwitch) startStreaming(); }, 10000);
    });
  }

  function stopStreaming() {
    if (streamWs) { try { streamWs.close(); } catch (_) {} streamWs = null; }
    streamConnected = false;
  }

  // ── Three-Tier Engine Intervals ─────────────────────────────────────────────
  let tier2Running = false;
  let tier3Running = false;

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

  async function scheduleTier2() {
    if (!state.active || state.killSwitch || tier2Running) {
      setTimeout(scheduleTier2, getTier2Interval());
      return;
    }
    if (state.circuitBreakerUntil > Date.now()) {
      setTimeout(scheduleTier2, getTier2Interval());
      return;
    }

    try {
      const clock = await alpaca("/v2/clock");
      const etH = getETHour();
      const isAnyWindow = clock.is_open || (etH >= 4 && etH < 20);
      if (!isAnyWindow) {
        setTimeout(scheduleTier2, getTier2Interval());
        return;
      }

      tier2Running = true;
      const interval = getTier2Interval();
      audit("TIER2", `Starting scan (interval: ${Math.round(interval / 60000)}min based on market time)`);
      await tier2Intelligence(clock.is_open, etH);
    } catch (err: any) {
      console.error("[tier2]", err?.message || err);
      audit("TIER2-ERROR", String(err?.message || err).slice(0, 200));
    } finally {
      tier2Running = false;
      setTimeout(scheduleTier2, getTier2Interval());
    }
  }

  // Start Tier 2 with adaptive scheduling
  setTimeout(scheduleTier2, 10000); // First run after 10 seconds

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
  app.get("/api/bot/last-scan", requireAuth, (_req, res) => {
    res.json(lastScanResult || { message: "No scan run yet. Activate the bot to start." });
  });

  // Route: Market calendar — holidays and early closes
  app.get("/api/bot/calendar", requireAuth, async (_req, res) => {
    try {
      const today = new Date().toISOString().split("T")[0];
      const future = new Date(Date.now() + 30 * 86400000).toISOString().split("T")[0];
      const cal = await alpaca(`/v2/calendar?start=${today}&end=${future}`);

      const tradingDays = new Set((cal as any[]).map((d: any) => d.date));

      const holidays: Array<{date: string; name: string}> = [];
      const earlyCloses: Array<{date: string; close: string}> = [];

      const holidayNames: Record<string, string> = {
        "01-01": "New Year's Day",
        "01-20": "Martin Luther King Jr. Day",
        "02-17": "Presidents' Day",
        "04-18": "Good Friday",
        "05-26": "Memorial Day",
        "06-19": "Juneteenth",
        "07-04": "Independence Day",
        "09-01": "Labor Day",
        "11-27": "Thanksgiving Day",
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
    audit("SYSTEM", "=== VolTradeAI Bot v2.0 Initialized ===");
    audit("SYSTEM", `Mode: PAPER TRADING (Alpaca)`);
    audit("RULES", `Scan interval: 45 seconds during trading hours`);
    audit("RULES", `Max positions: ${MAX_POSITIONS} ceiling (dynamic sizing uses portfolio heat)`);
    audit("RULES", `Position sizing: DYNAMIC — Kelly Criterion × volatility × confidence × VIX × earnings × liquidity × time`);
    audit("RULES", `Position range: 1-${MAX_POSITION_SIZE * 100}% per trade (sized by conviction + risk)`);
    audit("RULES", `Max total exposure: ${MAX_TOTAL_EXPOSURE * 100}% of portfolio invested`);
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
  app.post("/api/bot/run-now", requireAuth, async (_req, res) => {
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
  app.get("/api/bot/diagnostics", requireAuth, async (_req, res) => {
    try {
      const { stdout } = await execAsync(`python3 -c "
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
  app.get("/api/bot/ml-status", requireAuth, async (_req, res) => {
    try {
      const { stdout } = await execAsync(
        `python3 -c "
import json, os, time
try:
    from storage_config import ML_MODEL_PATH, FILLS_PATH, WEIGHTS_PATH
except ImportError:
    ML_MODEL_PATH = '/tmp/voltrade_ml_model.pkl'
    FILLS_PATH = '/tmp/voltrade_fills.json'
    WEIGHTS_PATH = '/tmp/voltrade_weights.json'
status = {'model_exists': os.path.exists(ML_MODEL_PATH), 'retrain_schedule': 'Daily at 4am ET', 'retrain_threshold_hours': 24}
if status['model_exists']:
    status['model_age_hours'] = round((time.time() - os.path.getmtime(ML_MODEL_PATH)) / 3600, 1)
    status['needs_retrain'] = status['model_age_hours'] > 24
if os.path.exists(FILLS_PATH):
    with open(FILLS_PATH) as f:
        fills = json.load(f)
    status['total_fills'] = len(fills)
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
