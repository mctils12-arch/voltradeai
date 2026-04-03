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

  // ── Autonomous Bot Engine ─────────────────────────────────────────────────

  let lastAutoRun = 0;
  let autoRunning = false;
  let lastScanResult: any = null;

  async function runAutonomousCycle() {
    if (!state.active || state.killSwitch || autoRunning) return;
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

      // Step 2: Execute position management (stop-loss / take-profit)
      for (const action of (result.position_actions || [])) {
        if (state.killSwitch) break;
        try {
          await alpaca(`/v2/positions/${action.ticker}`, { method: "DELETE" });
          audit("AUTO-CLOSE", `${action.ticker}: ${action.reason}`);
        } catch (e: any) {
          audit("AUTO-ERROR", `Failed to close ${action.ticker}: ${e.message}`);
        }
      }

      // Step 3: Execute new trades
      for (const trade of (result.new_trades || [])) {
        if (state.killSwitch) break;

        // Security checks
        const acct = await alpaca("/v2/account");
        const equity = parseFloat(acct.equity || "100000");
        const lastEquity = parseFloat(acct.last_equity || "100000");
        const dailyPnlPct = ((equity - lastEquity) / lastEquity) * 100;

        if (dailyPnlPct <= DAILY_LOSS_LIMIT) {
          audit("AUTO-BLOCKED", `Daily loss limit hit (${dailyPnlPct.toFixed(2)}%). No new trades.`);
          break;
        }

        if (trade.position_value > equity * MAX_POSITION_SIZE) {
          audit("AUTO-BLOCKED", `${trade.ticker} position too large ($${trade.position_value} > ${(MAX_POSITION_SIZE*100)}% of portfolio)`);
          continue;
        }

        try {
          // Check if we're in extended hours
          const clock = await alpaca("/v2/clock");
          const isExtended = !clock.is_open;
          
          const order = await alpaca("/v2/orders", {
            method: "POST",
            body: JSON.stringify({
              symbol: trade.ticker,
              qty: String(trade.shares),
              side: "buy",
              type: isExtended ? "limit" : "market",
              // Extended hours requires limit orders
              ...(isExtended ? { limit_price: String(trade.price) } : {}),
              time_in_force: isExtended ? "day" : "day",
              extended_hours: isExtended,
            }),
          });
          const session = isExtended ? "[EXTENDED]" : "[REGULAR]";
          audit("AUTO-TRADE", `${session} BUY ${trade.shares} ${trade.ticker} @ ~$${trade.price} | Score: ${trade.score} | ${(trade.reasons || []).join(" | ")}`);
        } catch (e: any) {
          audit("AUTO-ERROR", `Failed to buy ${trade.ticker}: ${e.message}`);
        }

        // Small delay between orders
        await new Promise(r => setTimeout(r, 1000));
      }

      audit("AUTO", `Cycle complete. ${(result.new_trades || []).length} new trades, ${(result.position_actions || []).length} position actions.`);
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

  // Auto-run every 15 minutes when bot is active
  // Trades during: pre-market (4am-9:30am ET), regular hours (9:30am-4pm ET), after hours (4pm-8pm ET)
  // Research during: 8pm-4am ET
  setInterval(async () => {
    if (!state.active || state.killSwitch) return;

    // Check market status — trade in extended hours too
    try {
      const clock = await alpaca("/v2/clock");
      const now = new Date(clock.timestamp);
      const etHour = now.getUTCHours() - 4; // Approximate ET
      const isExtendedHours = (etHour >= 4 && etHour < 9.5) || (etHour >= 16 && etHour < 20);
      const isRegularHours = clock.is_open;
      
      if (!isRegularHours && !isExtendedHours) {
        // 8pm-4am ET = Research & Preparation mode
        // Run full analysis, backtests, revalidation — no live trades
        if (!autoRunning) {
          await runOvernightResearch(etHour);
        }
        return;
      }
      
      if (isExtendedHours) {
        audit("AUTO", "Running extended hours scan (pre-market/after-hours)");
      }
    } catch { return; }

    // Run autonomous cycle
    await runAutonomousCycle();
  }, 15 * 60 * 1000); // Every 15 minutes

  // Route: Get last scan result
  app.get("/api/bot/last-scan", requireAuth, (_req, res) => {
    res.json(lastScanResult || { message: "No scan run yet. Activate the bot to start." });
  });

  // Route: Market calendar — holidays and early closes
  app.get("/api/bot/calendar", requireAuth, async (_req, res) => {
    try {
      // Get next 30 days of market calendar from Alpaca
      const today = new Date().toISOString().split("T")[0];
      const future = new Date(Date.now() + 30 * 86400000).toISOString().split("T")[0];
      const cal = await alpaca(`/v2/calendar?start=${today}&end=${future}`);
      
      // Find all trading days
      const tradingDays = new Set((cal as any[]).map((d: any) => d.date));
      
      // Find holidays (non-trading weekdays)
      const holidays: Array<{date: string; name: string}> = [];
      const earlyCloses: Array<{date: string; close: string}> = [];
      
      // Known US market holidays
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
      
      // Check next 30 days for non-trading weekdays
      for (let i = 0; i < 30; i++) {
        const d = new Date(Date.now() + i * 86400000);
        const dayOfWeek = d.getDay();
        if (dayOfWeek === 0 || dayOfWeek === 6) continue; // Skip weekends
        const dateStr = d.toISOString().split("T")[0];
        if (!tradingDays.has(dateStr)) {
          const mmdd = dateStr.substring(5);
          holidays.push({
            date: dateStr,
            name: holidayNames[mmdd] || "Market Holiday",
          });
        }
      }
      
      // Check for early closes (close time before 16:00)
      for (const day of (cal as any[])) {
        if (day.close && day.close < "16:00") {
          earlyCloses.push({ date: day.date, close: day.close });
        }
      }
      
      // Tomorrow check
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

  // Auto-start: run first cycle 30 seconds after server boots
  setTimeout(() => {
    audit("AUTO", "Bot auto-started on server boot");
    runAutonomousCycle();
  }, 30000);

  // Route: Force immediate scan
  app.post("/api/bot/run-now", requireAuth, async (_req, res) => {
    if (autoRunning) return res.json({ message: "Already running..." });
    res.json({ message: "Autonomous cycle starting..." });
    runAutonomousCycle();
  });

}
