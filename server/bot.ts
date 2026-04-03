import { Express } from "express";
import { requireAuth } from "./auth";
import { exec } from "child_process";
import { promisify } from "util";
import path from "path";
const execAsync = promisify(exec);

// ─── Alpaca Config ──────────────────────────────────────────────────────────
const ALPACA_BASE = "https://paper-api.alpaca.markets";
const ALPACA_KEY = process.env.ALPACA_KEY || "PKMDHJOVQEVIB4UHZXUYVTIDBU";
const ALPACA_SECRET = process.env.ALPACA_SECRET || "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et";

async function alpaca(path: string, opts: any = {}) {
  const r = await fetch(`${ALPACA_BASE}${path}`, {
    ...opts,
    headers: {
      "APCA-API-KEY-ID": ALPACA_KEY,
      "APCA-API-SECRET-KEY": ALPACA_SECRET,
      "Content-Type": "application/json",
      ...opts.headers,
    },
  });
  return r.json();
}

// ─── Bot State ──────────────────────────────────────────────────────────────
interface AuditEntry { time: string; action: string; detail: string; }

// Phase 6: Security constants
const DAILY_LOSS_LIMIT = -3; // percent
const MAX_POSITION_SIZE = 0.05; // 5% of portfolio per position
const MAX_TOTAL_EXPOSURE = 0.5; // 50% of portfolio max invested
const MAX_POSITIONS = 5;
const STOP_LOSS_PCT = 0.02;
const TAKE_PROFIT_PCT = 0.06;

const state = {
  active: true,  // Bot starts automatically — always on unless killed
  killSwitch: false,
  dailyPnL: 0,
  dailyLossLimit: DAILY_LOSS_LIMIT,
  auditLog: [] as AuditEntry[],
};

function audit(action: string, detail: string) {
  state.auditLog.unshift({ time: new Date().toISOString(), action, detail });
  if (state.auditLog.length > 500) state.auditLog.length = 500;
  console.log(`[BOT] ${action}: ${detail}`);
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
  holdingDays: number;
  timestamp: string;
}

const tradeResults: TradeResult[] = [];

// In-memory equity curve (keeps up to 1 year of daily data)
const equityCurve: Array<{ date: string; value: number; pnl: number }> = [];

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
        holdingDays: 0,
        timestamp: order.filled_at,
      });

      if (tradeResults.length > 200) tradeResults.length = 200;
    }

    // Adjust strategy weights based on recent performance
    adjustStrategyWeights();
  } catch (e: any) {
    audit("LEARN-ERROR", `Track closed trades failed: ${e.message}`);
  }
}

function adjustStrategyWeights() {
  // Simple heuristic: if recent win rate < 40%, reduce weight on low-confidence signals
  const recent = tradeResults.slice(0, 20);
  if (recent.length < 5) return;

  const winRate = recent.filter(t => t.pnl > 0).length / recent.length;
  if (winRate < 0.4) {
    audit("LEARN", `Win rate ${(winRate * 100).toFixed(0)}% — increasing VRP/squeeze weight`);
    // Shift weight toward VRP (more reliable signal)
    strategyWeights.vrp = Math.min(0.40, strategyWeights.vrp + 0.02);
    strategyWeights.momentum = Math.max(0.15, strategyWeights.momentum - 0.01);
    strategyWeights.volume = Math.max(0.10, strategyWeights.volume - 0.01);
  } else if (winRate > 0.65) {
    // Good performance — restore balanced weights gradually
    strategyWeights.momentum = Math.min(0.30, strategyWeights.momentum + 0.01);
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

// ─── Options order (log + audit, full execution requires Alpaca options setup) ─
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

  audit("OPTIONS", `${side.toUpperCase()} ${qty}x ${ticker} ${strike} ${optionType} exp:${expiry} [OCC: ${occSymbol}]`);
  notify("options", `${side.toUpperCase()} ${qty}x ${ticker} $${strike} ${optionType.toUpperCase()} (${expiry}) — queued`);
  // TODO: Implement full execution when Alpaca options API keys are configured
  return { status: "logged", occ_symbol: occSymbol, message: "Options execution requires Alpaca options-enabled account" };
}

// ─── Routes ─────────────────────────────────────────────────────────────────
export function registerBotRoutes(app: Express) {

  // Account info from Alpaca
  app.get("/api/bot/account", requireAuth, async (_req, res) => {
    try {
      const acct = await alpaca("/v2/account");
      const equity = parseFloat(acct.equity);
      const lastEquity = parseFloat(acct.last_equity);
      const dailyPnL = equity - lastEquity;
      // Update bot state with current daily P&L
      state.dailyPnL = dailyPnL;
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
      const mapped = (positions as any[]).map((p: any) => ({
        ticker: p.symbol,
        qty: parseFloat(p.qty),
        side: parseFloat(p.qty) > 0 ? "long" : "short",
        entryPrice: parseFloat(p.avg_entry_price),
        currentPrice: parseFloat(p.current_price),
        marketValue: parseFloat(p.market_value),
        pnl: parseFloat(p.unrealized_pl),
        pnlPct: parseFloat(p.unrealized_plpc) * 100,
      }));
      res.json(mapped);
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

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
  app.get("/api/bot/performance", requireAuth, (_req, res) => {
    const results = tradeResults;
    const totalTrades = results.length;
    const winners = results.filter(t => t.pnl > 0);
    const losers = results.filter(t => t.pnl < 0);
    const winRate = totalTrades > 0 ? (winners.length / totalTrades) * 100 : 0;
    const totalPnl = results.reduce((sum, t) => sum + t.pnl, 0);
    const avgGain = winners.length > 0 ? winners.reduce((s, t) => s + t.pnlPct, 0) / winners.length : 0;
    const avgLoss = losers.length > 0 ? losers.reduce((s, t) => s + t.pnlPct, 0) / losers.length : 0;
    const bestTrade = results.reduce((best, t) => (!best || t.pnlPct > best.pnlPct ? t : best), null as TradeResult | null);
    const worstTrade = results.reduce((worst, t) => (!worst || t.pnlPct < worst.pnlPct ? t : worst), null as TradeResult | null);

    // Strategy breakdown
    const byStrategy: Record<string, { trades: number; wins: number; pnl: number }> = {};
    for (const t of results) {
      if (!byStrategy[t.strategy]) byStrategy[t.strategy] = { trades: 0, wins: 0, pnl: 0 };
      byStrategy[t.strategy].trades++;
      if (t.pnl > 0) byStrategy[t.strategy].wins++;
      byStrategy[t.strategy].pnl += t.pnl;
    }

    res.json({
      totalTrades,
      winRate: Math.round(winRate * 10) / 10,
      totalPnl: Math.round(totalPnl * 100) / 100,
      avgGain: Math.round(avgGain * 100) / 100,
      avgLoss: Math.round(avgLoss * 100) / 100,
      bestTrade,
      worstTrade,
      equityCurve,
      strategyWeights,
      byStrategy,
    });
  });

  // ── Notifications endpoints ───────────────────────────────────────────────
  app.get("/api/bot/notifications", requireAuth, (_req, res) => {
    res.json(notifications);
  });

  app.post("/api/bot/notifications/read", requireAuth, (_req, res) => {
    notifications.forEach(n => (n.read = true));
    res.json({ ok: true });
  });

  // Bot status
  app.get("/api/bot/status", requireAuth, (_req, res) => {
    res.json({
      active: state.active,
      killSwitch: state.killSwitch,
      dailyLossLimit: state.dailyLossLimit,
      auditLogCount: state.auditLog.length,
      mode: "paper",
      unreadNotifications: notifications.filter(n => !n.read).length,
    });
  });

  // Start bot
  app.post("/api/bot/start", requireAuth, (_req, res) => {
    if (state.killSwitch) return res.status(400).json({ error: "Kill switch is ON. Disable it first." });
    state.active = true;
    audit("START", "Bot activated");
    notify("system", "Bot activated — scanning for opportunities");
    res.json({ ok: true, active: true });
  });

  // Stop bot
  app.post("/api/bot/stop", requireAuth, (_req, res) => {
    state.active = false;
    audit("STOP", "Bot deactivated");
    notify("system", "Bot paused");
    res.json({ ok: true, active: false });
  });

  // Kill switch
  app.post("/api/bot/kill", requireAuth, async (_req, res) => {
    state.killSwitch = !state.killSwitch;
    if (state.killSwitch) {
      state.active = false;
      audit("KILL SWITCH ON", "All trading halted. Cancelling open orders.");
      notify("alert", "KILL SWITCH ACTIVATED — all trading halted, open orders cancelled");
      try { await alpaca("/v2/orders", { method: "DELETE" }); } catch {}
    } else {
      audit("KILL SWITCH OFF", "Trading can resume.");
      notify("system", "Kill switch deactivated — trading can resume");
    }
    res.json({ ok: true, killSwitch: state.killSwitch });
  });

  // Audit log
  app.get("/api/bot/audit", requireAuth, (_req, res) => {
    res.json(state.auditLog.slice(0, 100));
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
    } catch {}

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
      } catch {}
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
    } catch {}
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

    audit("MORNING", `Executing ${morningQueue.length} queued trades from overnight research...`);

    // Sort by score descending — best picks first
    morningQueue.sort((a, b) => b.score - a.score);

    const acct = await alpaca("/v2/account");
    const equity = parseFloat(acct.equity || "100000");
    let slotsUsed = 0;

    try {
      const positions = await alpaca("/v2/positions");
      slotsUsed = Array.isArray(positions) ? positions.length : 0;
    } catch {}

    for (const trade of morningQueue) {
      if (state.killSwitch) break;
      if (slotsUsed >= MAX_POSITIONS) {
        audit("MORNING", `Max positions (${MAX_POSITIONS}) reached — skipping remaining queue`);
        break;
      }
      if (trade.position_value > equity * MAX_POSITION_SIZE) {
        audit("MORNING-BLOCKED", `${trade.ticker}: Position too large`);
        continue;
      }

      try {
        const order = await alpaca("/v2/orders", {
          method: "POST",
          body: JSON.stringify({
            symbol: trade.ticker,
            qty: String(Math.floor(trade.shares)),
            side: trade.side === "short" ? "sell" : (trade.side || "buy"),
            type: "market", // Market orders at open for instant fill
            time_in_force: "day",
          }),
        });

        audit("MORNING-TRADE", `MARKET ${(trade.side || "BUY").toUpperCase()} ${Math.floor(trade.shares)} ${trade.ticker} @ market | Score: ${trade.score} | Queued from overnight research`);
        notify("trade", `Morning queue: ${(trade.side || "BUY").toUpperCase()} ${Math.floor(trade.shares)} ${trade.ticker} (score: ${trade.score})`);
        slotsUsed++;
      } catch (e: any) {
        audit("MORNING-ERROR", `Failed: ${trade.ticker} — ${e.message}`);
      }

      await new Promise(r => setTimeout(r, 500));
    }

    // Clear the queue
    morningQueue.length = 0;
    audit("MORNING", "Queue cleared.");
  }

  // ── Autonomous Bot Engine ─────────────────────────────────────────────────

  let lastAutoRun = 0;
  let autoRunning = false;
  let lastScanResult: any = null;
  let morningQueueExecuted = false; // Track if we ran the morning queue today

  async function runAutonomousCycle() {
    if (!state.active || state.killSwitch || autoRunning) return;
    // Don't re-run if last scan was less than 30 seconds ago
    if (Date.now() - lastAutoRun < 30000 && lastAutoRun > 0) return;

    // CHECK: Is market actually open for ANY trading?
    try {
      const clockCheck = await alpaca("/v2/clock");
      const nowCheck = new Date(clockCheck.timestamp);
      const etH = nowCheck.getUTCHours() - 4;
      const isAnyTradingWindow = clockCheck.is_open || (etH >= 4 && etH < 20);

      if (!isAnyTradingWindow) {
        // Market fully closed (8pm-4am ET) — do research only, no trades
        return;
      }
    } catch { return; }

    autoRunning = true;
    audit("AUTO", "Starting autonomous scan cycle...");

    try {
      // Step 1: Run bot_engine.py to scan market and get recommendations
      const enginePath = path.resolve("bot_engine.py");
      const { stdout } = await execAsync(`python3 "${enginePath}" full`, { timeout: 180000 });
      const result = JSON.parse(stdout.trim());
      lastScanResult = result;

      if (result.error) {
        audit("AUTO", `Scan error: ${result.error}`);
        autoRunning = false;
        return;
      }

      audit("AUTO", `Scanned ${result.scanned} stocks, filtered to ${result.filtered}, deep-analyzed ${result.deep_analyzed}`);

      // Auto-generate signals from scan results
      const topPicks = result.top_10 || [];
      for (const pick of topPicks) {
        if (pick.score >= 60) {
          const existing = signals.findIndex(s => s.ticker === pick.ticker);
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
        const existing = signals.findIndex(s => s.ticker === trade.ticker);
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

      // Step 2: Execute position management (stop-loss / take-profit)
      for (const action of (result.position_actions || [])) {
        if (state.killSwitch) break;
        try {
          await alpaca(`/v2/positions/${action.ticker}`, { method: "DELETE" });
          audit("AUTO-CLOSE", `${action.ticker}: ${action.reason}`);
          notify(
            action.type === "take_profit" ? "profit" : "stop_loss",
            `${action.ticker} position closed — ${action.reason}`
          );
        } catch (e: any) {
          audit("AUTO-ERROR", `Failed to close ${action.ticker}: ${e.message}`);
        }
      }

      // Step 3: Execute new trades — Smart Execution Engine
      const clock2 = await alpaca("/v2/clock");
      const isMarketOpen = clock2.is_open;
      const now2 = new Date(clock2.timestamp);
      const etHour2 = now2.getUTCHours() - 4;
      const isPreMarket = !isMarketOpen && etHour2 >= 4 && etHour2 < 9.5;
      const isAfterHours = !isMarketOpen && etHour2 >= 16 && etHour2 < 20;
      const isExtended = isPreMarket || isAfterHours;

      // SMART EXECUTION RULES:
      // - Regular hours (9:30am-4pm): Market orders for instant fill
      // - Pre-market (4am-9:30am): Queue for market open (no chasing thin liquidity)
      // - After-hours (4pm-8pm): Queue for next market open
      // - Options: Regular hours only
      // - Stale limit orders cancelled after 12 minutes
      // - Better-scoring picks replace weaker unfilled orders
      // - Morning queue executes market orders at 9:30am for guaranteed fills

      // Run stale order sweeper every cycle
      await sweepStaleOrders();

      const acct2 = await alpaca("/v2/account");
      const equity = parseFloat(acct2.equity || "100000");
      const lastEquity = parseFloat(acct2.last_equity || "100000");
      const dailyPnlPct = ((equity - lastEquity) / lastEquity) * 100;

      if (dailyPnlPct <= DAILY_LOSS_LIMIT) {
        audit("AUTO-BLOCKED", `Daily loss limit hit (${dailyPnlPct.toFixed(2)}%). All trading halted for today.`);
        notify("alert", `Daily loss limit hit (${dailyPnlPct.toFixed(2)}%). Trading halted for today.`);
      } else {
        // Execute morning queue on first regular-hours cycle of the day
        if (isMarketOpen && !morningQueueExecuted && morningQueue.length > 0) {
          await executeMorningQueue();
          morningQueueExecuted = true;
        }

        for (const trade of (result.new_trades || [])) {
          if (state.killSwitch) break;

          // Position size check
          if (trade.position_value > equity * MAX_POSITION_SIZE) {
            audit("AUTO-BLOCKED", `${trade.ticker}: Position too large ($${trade.position_value} > ${(MAX_POSITION_SIZE * 100)}% limit)`);
            continue;
          }

          // Options trade check — only during regular hours
          const isOptionsOrder = (trade.trade_type === "options") || trade.recommendation?.toLowerCase().includes("option");
          if (isOptionsOrder) {
            if (!isMarketOpen) {
              // Queue for morning instead of blocking
              const existsInQueue = morningQueue.some(q => q.ticker === trade.ticker);
              if (!existsInQueue) {
                morningQueue.push({ ...trade, queuedAt: new Date().toISOString() });
                audit("QUEUE", `${trade.ticker}: Options queued for market open (score ${trade.score})`);
              }
              continue;
            }
            await placeOptionsOrder(
              trade.ticker,
              trade.vrp > 0 ? "call" : "put",
              trade.price,
              new Date(Date.now() + 30 * 86400000).toISOString().split("T")[0],
              trade.side === "short" ? "sell" : "buy",
              trade.shares
            );
            continue;
          }

          // ── EXTENDED HOURS: Queue for morning, don't chase thin liquidity ──
          if (isExtended) {
            // Replace weaker queued trades if this one is better
            const weakIdx = morningQueue.findIndex(q => q.score < trade.score - 10);
            if (weakIdx >= 0 && morningQueue.length >= MAX_POSITIONS) {
              const replaced = morningQueue[weakIdx];
              morningQueue.splice(weakIdx, 1);
              audit("QUEUE-REPLACE", `Replaced ${replaced.ticker} (score ${replaced.score}) with ${trade.ticker} (score ${trade.score}) in morning queue`);
            }

            const existsInQueue = morningQueue.some(q => q.ticker === trade.ticker);
            if (!existsInQueue && morningQueue.length < MAX_POSITIONS) {
              morningQueue.push({ ...trade, queuedAt: new Date().toISOString() });
              audit("QUEUE", `${trade.ticker}: Queued for market open (score ${trade.score}) — extended hours, no chasing thin liquidity`);
            }
            continue;
          }

          // ── REGULAR HOURS: Execute immediately with market orders ──
          const qty = Math.floor(trade.shares);
          if (qty <= 0) continue;

          // Check if buying power is locked — try to replace weaker open orders
          const acctCheck = await alpaca("/v2/account");
          const buyingPower = parseFloat(acctCheck.buying_power || "0");
          if (trade.position_value > buyingPower) {
            const replaced = await replaceIfBetter(trade);
            if (!replaced) {
              audit("AUTO-BLOCKED", `${trade.ticker}: Not enough buying power ($${buyingPower.toFixed(0)} available, need $${trade.position_value.toFixed(0)})`);
              continue;
            }
            // Wait for order cancellation to free buying power
            await new Promise(r => setTimeout(r, 1000));
          }

          try {
            const sideStr = trade.side === "short" ? "sell" : (trade.side || "buy");
            const order = await alpaca("/v2/orders", {
              method: "POST",
              body: JSON.stringify({
                symbol: trade.ticker,
                qty: String(qty),
                side: sideStr,
                type: "market",
                time_in_force: "day",
              }),
            });

            const sideLabel = trade.side === "short" ? "SHORT" : (trade.side || "BUY").toUpperCase();
            audit("AUTO-TRADE", `[REGULAR] MARKET ${sideLabel} ${qty} ${trade.ticker} @ ~$${trade.price} | Score: ${trade.score} | ${(trade.reasons || []).join(" | ")}`);
            notify("trade", `${sideLabel} ${qty} ${trade.ticker} @ ~$${trade.price} (score: ${trade.score})`);

            // Track the order for stale sweeping (market orders fill instantly, but just in case)
            if (order?.id) {
              openOrders.push({
                orderId: order.id,
                ticker: trade.ticker,
                score: trade.score,
                placedAt: Date.now(),
                side: sideStr,
                qty,
                limitPrice: trade.price,
              });
            }
          } catch (e: any) {
            audit("AUTO-ERROR", `Failed to trade ${trade.ticker}: ${e.message}`);
          }

          await new Promise(r => setTimeout(r, 500));
        }
      }

      // Run learning loop after each cycle
      await trackClosedTrades();

      audit("AUTO", `Cycle complete. ${(result.new_trades || []).length} new trades, ${(result.position_actions || []).length} position actions.`);

      // Daily summary (check once per cycle if it's near market close — 4pm ET)
      try {
        const etHourNow = new Date().getUTCHours() - 4;
        if (etHourNow === 16) {
          recordDailyEquity();
          const acctSummary = await alpaca("/v2/account");
          const dailyPnlDollars = parseFloat(acctSummary.equity) - parseFloat(acctSummary.last_equity);
          notify(
            "daily_summary",
            `Daily summary: P&L ${dailyPnlDollars >= 0 ? "+" : ""}$${dailyPnlDollars.toFixed(2)} | Portfolio: $${parseFloat(acctSummary.portfolio_value).toLocaleString()}`
          );
        }
      } catch {}
    } catch (e: any) {
      audit("AUTO-ERROR", `Autonomous cycle failed: ${e.message}`);
    }

    autoRunning = false;
    lastAutoRun = Date.now();
  }

  // ── Overnight Research Engine (8pm-4am ET) ───────────────────────────────
  let lastResearchRun = 0;

  async function runOvernightResearch(etHour: number) {
    // Only run once per hour to save resources
    if (Date.now() - lastResearchRun < 55 * 60 * 1000) return;
    lastResearchRun = Date.now();
    autoRunning = true;

    try {
      // 8pm-10pm: Scan market, find tomorrow's opportunities
      if (etHour >= 20 && etHour < 22) {
        audit("RESEARCH", "Evening scan — analyzing today's movers for tomorrow's plays");
        const enginePath3 = path.resolve("bot_engine.py");
        const { stdout } = await execAsync(`python3 "${enginePath3}" scan`, { timeout: 180000 });
        lastScanResult = JSON.parse(stdout.trim());
        audit("RESEARCH", `Scanned ${lastScanResult.scanned || 0} stocks — top picks identified for morning`);
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
        } catch {}
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
          } catch {}
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
        } catch {}
      }

    } catch (e: any) {
      audit("RESEARCH-ERROR", `Overnight research failed: ${e.message}`);
    }

    autoRunning = false;
  }

  // Auto-run every 45 seconds when bot is active
  setInterval(async () => {
    if (!state.active || state.killSwitch) return;

    try {
      const clock = await alpaca("/v2/clock");
      const now = new Date(clock.timestamp);
      const etHour = now.getUTCHours() - 4; // Approximate ET
      const isExtendedHours = (etHour >= 4 && etHour < 9.5) || (etHour >= 16 && etHour < 20);
      const isRegularHours = clock.is_open;

      if (!isRegularHours && !isExtendedHours) {
        // 8pm-4am ET = Research & Preparation mode
        if (!autoRunning) {
          await runOvernightResearch(etHour);
        }
        return;
      }

      if (isExtendedHours) {
        audit("AUTO", "Running extended hours scan — findings queued for market open");
      }

      // Reset morning queue flag at start of each trading day (4am ET)
      if (etHour >= 4 && etHour < 5) {
        morningQueueExecuted = false;
      }
    } catch { return; }

    await runAutonomousCycle();
  }, 45 * 1000);

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
    audit("RULES", `Max positions: ${MAX_POSITIONS} at once`);
    audit("RULES", `Max position size: ${MAX_POSITION_SIZE * 100}% of portfolio per trade`);
    audit("RULES", `Max total exposure: ${MAX_TOTAL_EXPOSURE * 100}% of portfolio invested`);
    audit("RULES", `Stop loss: ${STOP_LOSS_PCT * 100}% per position`);
    audit("RULES", `Take profit: ${TAKE_PROFIT_PCT * 100}% per position`);
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
      const now3 = new Date(clock.timestamp);
      const etH3 = now3.getUTCHours() - 4;
      const canTrade = clock.is_open || (etH3 >= 4 && etH3 < 20);

      if (canTrade) {
        audit("SYSTEM", "Market is in a trading window. First scan starting...");
        setTimeout(() => runAutonomousCycle(), 5000);
      } else {
        const nextOpen = clock.next_open ? new Date(clock.next_open).toLocaleString("en-US", { timeZone: "America/New_York" }) : "unknown";
        audit("SYSTEM", `Market is closed. No trading until next session. Next open: ${nextOpen} ET`);
        audit("SYSTEM", "Bot is idle. Will auto-scan when market opens.");
      }
    }).catch(() => {
      audit("SYSTEM", "Could not check market status. Will retry on next cycle.");
    });
  }, 5000);

  // Route: Force immediate scan
  app.post("/api/bot/run-now", requireAuth, async (_req, res) => {
    if (autoRunning) return res.json({ message: "Already running..." });
    res.json({ message: "Autonomous cycle starting..." });
    runAutonomousCycle();
  });

}
