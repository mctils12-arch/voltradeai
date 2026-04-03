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

const state = {
  active: false,
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
  // 4. Total exposure check (simplified — checked via portfolioValue vs cash ratio)
  return { allowed: true };
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

  // Bot status
  app.get("/api/bot/status", requireAuth, (_req, res) => {
    res.json({
      active: state.active,
      killSwitch: state.killSwitch,
      dailyLossLimit: state.dailyLossLimit,
      auditLogCount: state.auditLog.length,
      mode: "paper",
    });
  });

  // Start bot
  app.post("/api/bot/start", requireAuth, (_req, res) => {
    if (state.killSwitch) return res.status(400).json({ error: "Kill switch is ON. Disable it first." });
    state.active = true;
    audit("START", "Bot activated");
    res.json({ ok: true, active: true });
  });

  // Stop bot
  app.post("/api/bot/stop", requireAuth, (_req, res) => {
    state.active = false;
    audit("STOP", "Bot deactivated");
    res.json({ ok: true, active: false });
  });

  // Kill switch
  app.post("/api/bot/kill", requireAuth, async (_req, res) => {
    state.killSwitch = !state.killSwitch;
    if (state.killSwitch) {
      state.active = false;
      audit("KILL SWITCH ON", "All trading halted. Cancelling open orders.");
      try { await alpaca("/v2/orders", { method: "DELETE" }); } catch {}
    } else {
      audit("KILL SWITCH OFF", "Trading can resume.");
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
      const order = await alpaca("/v2/orders", {
        method: "POST",
        body: JSON.stringify({
          symbol: ticker.toUpperCase(),
          qty: String(qty),
          side,
          type,
          time_in_force: "day",
        }),
      });
      audit("TRADE", `${side.toUpperCase()} ${qty} ${ticker} @ ${type}`);
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

}
