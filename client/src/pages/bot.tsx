import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  ShieldAlert, Play, Square, TrendingUp, Activity, Clock,
  RefreshCw, Shield, Lock, BarChart2, Zap,
} from "lucide-react";
import ChartPage from "./chart";

// ─── Tooltip helper ──────────────────────────────────────────────────────────
const TIPS: Record<string, string> = {
  portfolio: "Total value of your account — cash plus the value of all open positions.",
  cash: "Money available to make new trades.",
  buyingPower: "How much you can trade with, including margin. For paper trading, this is usually 4x your cash.",
  dailyPnl: "How much you've made or lost today compared to yesterday's close.",
  killSwitch: "Emergency stop. Turns off all trading and cancels every open order immediately. Use if something looks wrong.",
  paperTrading: "Fake money trading. Tests your strategies safely before risking real money. Your account starts with $100,000 of practice money.",
  position: "A stock you currently own (long) or are betting against (short).",
  pnl: "Profit and Loss — how much money this position has made or lost since you opened it.",
  sharpe: "How much return you get per unit of risk. Above 1.0 is good. Above 2.0 is excellent.",
  drawdown: "The biggest peak-to-trough drop. If it says -15%, at worst you lost 15% before recovering.",
  winrate: "What percentage of trades made money. Above 60% is very good for this type of strategy.",
  backtest: "Testing a strategy on past data to see if it would have made money. Not a guarantee, but essential before trading real money.",
  vrpSelling: "Selling options when they're overpriced vs how much the stock actually moves. You collect money upfront and keep it if the stock stays calm.",
  momentum: "Buying stocks that have been going up recently. Research shows trends tend to continue for weeks or months.",
  pead: "Post-Earnings Announcement Drift — when a company beats earnings expectations, the stock usually keeps rising for weeks after.",
  regime: "What kind of market we're in right now: calm, choppy, trending up, or crashing. The bot uses different strategies for each.",
  confidence: "How sure the AI is about this signal. 80%+ means strong conviction based on multiple factors agreeing.",
  annualReturn: "Average return per year if you ran this strategy continuously.",
  totalReturn: "Total percentage gained (or lost) over the entire backtest period.",
  maxDrawdown: "Worst peak-to-trough decline during the backtest. Smaller is safer.",
  dailyLossLimit: "If the bot loses more than this percentage in a single day, all trading automatically halts until reset.",
  positionSize: "The bot will never put more than this percentage of your portfolio into a single trade.",
  totalExposure: "Maximum percentage of your portfolio that can be invested at any one time.",
};

function Tip({ id, children }: { id: string; children: React.ReactNode }) {
  const [show, setShow] = useState(false);
  const tip = TIPS[id];
  if (!tip) return <>{children}</>;
  return (
    <span style={{ position: "relative", cursor: "help", borderBottom: "1px dotted rgba(255,255,255,0.3)" }}
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span style={{ position: "absolute", bottom: "calc(100% + 8px)", left: "50%", transform: "translateX(-50%)",
          background: "rgba(30,30,30,0.95)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: "10px",
          padding: "10px 14px", fontSize: "12px", color: "#d1d1d6", lineHeight: 1.5, width: "260px", zIndex: 100,
          backdropFilter: "blur(12px)", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}>
          {tip}
        </span>
      )}
    </span>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const card: React.CSSProperties = { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "16px", padding: "20px", backdropFilter: "blur(20px)" };
const label: React.CSSProperties = { fontSize: "11px", color: "#6e6e73", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" };
const bigNum: React.CSSProperties = { fontSize: "28px", fontWeight: 700, fontFamily: "'SF Mono', 'Fira Code', monospace" };

// ─── Sharpe color helper ──────────────────────────────────────────────────────
function sharpeColor(v: number) {
  if (v >= 1) return "#30d158";
  if (v >= 0.5) return "#ff9f0a";
  return "#ff453a";
}

// ─── Signal type color ────────────────────────────────────────────────────────
function signalColor(type: string) {
  if (type === "buy") return "#30d158";
  if (type === "sell") return "#ff453a";
  return "#ff9f0a";
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function BotDashboard() {
  // ── Local state for backtest form ─────────────────────────────────────────
  const [btTicker, setBtTicker] = useState("SPY");
  const [btStrategy, setBtStrategy] = useState("all");
  const [btYears, setBtYears] = useState("3");
  const [btResults, setBtResults] = useState<any>(null);
  const [btLoading, setBtLoading] = useState(false);
  const [btError, setBtError] = useState("");

  // ── Queries ───────────────────────────────────────────────────────────────
  const { data: acct } = useQuery({
    queryKey: ["/api/bot/account"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/account"); return r.json(); },
    refetchInterval: 10000,
  });
  const { data: positions } = useQuery({
    queryKey: ["/api/bot/positions"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/positions"); return r.json(); },
    refetchInterval: 10000,
  });
  const { data: status } = useQuery({
    queryKey: ["/api/bot/status"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/status"); return r.json(); },
    refetchInterval: 5000,
  });
  const { data: audit } = useQuery({
    queryKey: ["/api/bot/audit"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/audit"); return r.json(); },
    refetchInterval: 15000,
  });
  const { data: signals, refetch: refetchSignals } = useQuery({
    queryKey: ["/api/bot/signals"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/signals"); return r.json(); },
    refetchInterval: 30000,
  });

  // ── Mutations ─────────────────────────────────────────────────────────────
  const startBot = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/start").then(r => r.json()),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); },
  });
  const stopBot = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/stop").then(r => r.json()),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); },
  });
  const killBot = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/kill").then(r => r.json()),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); },
  });
  const closePos = useMutation({
    mutationFn: (ticker: string) => apiRequest("POST", "/api/bot/close", { ticker }).then(r => r.json()),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/positions"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); },
  });
  const refreshSignals = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/refresh-signals").then(r => r.json()),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot/signals"] });
      queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] });
      refetchSignals();
    },
  });

  // ── Derived values ────────────────────────────────────────────────────────
  const dailyPnl = acct?.dailyPnL ?? 0;
  const dailyPnlPct = acct?.dailyPnLPct ?? 0;
  const isKilled = status?.killSwitch ?? false;
  const isActive = status?.active ?? false;

  // ── Backtest runner ───────────────────────────────────────────────────────
  async function runBacktest() {
    setBtLoading(true);
    setBtError("");
    setBtResults(null);
    try {
      const r = await apiRequest("POST", "/api/bot/backtest", {
        ticker: btTicker.toUpperCase().trim(),
        strategy: btStrategy,
        years: parseInt(btYears),
      });
      const data = await r.json();
      if (data.error) { setBtError(data.error); }
      else { setBtResults(data); }
    } catch (e: any) {
      setBtError(e.message || "Backtest failed");
    } finally {
      setBtLoading(false);
    }
  }

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "24px 16px" }}>

      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h1 style={{ fontSize: "1.4rem", fontWeight: 700, color: "#f5f5f7", margin: 0 }}>Trading Bot</h1>
          <div style={{ display: "flex", gap: "8px", marginTop: "6px", alignItems: "center" }}>
            <span style={{
              padding: "3px 10px", borderRadius: "20px", fontSize: "11px", fontWeight: 600,
              background: isKilled ? "rgba(255,69,58,0.15)" : isActive ? "rgba(48,209,88,0.15)" : "rgba(255,159,10,0.15)",
              color: isKilled ? "#ff453a" : isActive ? "#30d158" : "#ff9f0a",
              border: `1px solid ${isKilled ? "rgba(255,69,58,0.3)" : isActive ? "rgba(48,209,88,0.3)" : "rgba(255,159,10,0.3)"}`,
            }}>
              {isKilled ? "KILLED" : isActive ? "ACTIVE" : "PAUSED"}
            </span>
            <Tip id="paperTrading">
              <span style={{ padding: "3px 10px", borderRadius: "20px", fontSize: "11px", fontWeight: 600, background: "rgba(10,132,255,0.15)", color: "#0a84ff", border: "1px solid rgba(10,132,255,0.3)" }}>
                PAPER TRADING
              </span>
            </Tip>
          </div>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          {!isActive ? (
            <button onClick={() => startBot.mutate()} disabled={isKilled} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 16px", borderRadius: "10px", border: "none", background: isKilled ? "#333" : "#30d158", color: "white", fontSize: "13px", fontWeight: 600, cursor: isKilled ? "not-allowed" : "pointer" }}>
              <Play size={14} /> Start
            </button>
          ) : (
            <button onClick={() => stopBot.mutate()} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 16px", borderRadius: "10px", border: "none", background: "#ff9f0a", color: "black", fontSize: "13px", fontWeight: 600, cursor: "pointer" }}>
              <Square size={14} /> Pause
            </button>
          )}
          <Tip id="killSwitch">
            <button onClick={() => killBot.mutate()} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 16px", borderRadius: "10px", border: isKilled ? "2px solid #ff453a" : "2px solid rgba(255,69,58,0.3)", background: isKilled ? "rgba(255,69,58,0.2)" : "transparent", color: "#ff453a", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}>
              <ShieldAlert size={14} /> {isKilled ? "Unkill" : "Kill Switch"}
            </button>
          </Tip>
        </div>
      </div>

      {/* ── Account Overview ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px", marginBottom: "20px" }}>
        <div style={card}>
          <div style={label}><Tip id="portfolio">Portfolio Value</Tip></div>
          <div style={{ ...bigNum, color: "#f5f5f7" }}>${(acct?.portfolioValue ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
        </div>
        <div style={card}>
          <div style={label}><Tip id="cash">Cash Available</Tip></div>
          <div style={{ ...bigNum, color: "#a1a1a6" }}>${(acct?.cash ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
        </div>
        <div style={card}>
          <div style={label}><Tip id="dailyPnl">Today's P&L</Tip></div>
          <div style={{ ...bigNum, color: dailyPnl >= 0 ? "#30d158" : "#ff453a" }}>
            {dailyPnl >= 0 ? "+" : ""}{dailyPnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            <span style={{ fontSize: "14px", marginLeft: "6px" }}>({dailyPnlPct >= 0 ? "+" : ""}{Number(dailyPnlPct ?? 0).toFixed(2)}%)</span>
          </div>
        </div>
        <div style={card}>
          <div style={label}><Tip id="buyingPower">Buying Power</Tip></div>
          <div style={{ ...bigNum, color: "#a1a1a6", fontSize: "22px" }}>${(acct?.buyingPower ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
        </div>
      </div>

      {/* ── Open Positions ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
          <Activity size={14} style={{ color: "#0a84ff" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#f5f5f7" }}><Tip id="position">Open Positions</Tip></span>
          <span style={{ marginLeft: "auto", fontSize: "12px", color: "#6e6e73" }}>{Array.isArray(positions) ? positions.length : 0} positions</span>
        </div>
        {Array.isArray(positions) && positions.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <thead>
                <tr style={{ color: "#6e6e73", textAlign: "left", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                  <th style={{ padding: "8px 12px" }}>Ticker</th>
                  <th style={{ padding: "8px 6px" }}>Side</th>
                  <th style={{ padding: "8px 6px", textAlign: "right" }}>Qty</th>
                  <th style={{ padding: "8px 6px", textAlign: "right" }}>Entry</th>
                  <th style={{ padding: "8px 6px", textAlign: "right" }}>Current</th>
                  <th style={{ padding: "8px 6px", textAlign: "right" }}><Tip id="pnl">P&L</Tip></th>
                  <th style={{ padding: "8px 6px" }}></th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p: any) => (
                  <tr key={p.ticker} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <td style={{ padding: "8px 12px", fontWeight: 700, color: "#f5f5f7", fontFamily: "monospace" }}>{p.ticker}</td>
                    <td style={{ padding: "8px 6px", color: p.side === "long" ? "#30d158" : "#ff453a" }}>{p.side}</td>
                    <td style={{ padding: "8px 6px", textAlign: "right", fontFamily: "monospace" }}>{p.qty}</td>
                    <td style={{ padding: "8px 6px", textAlign: "right", fontFamily: "monospace", color: "#a1a1a6" }}>${Number(p.entryPrice ?? 0).toFixed(2)}</td>
                    <td style={{ padding: "8px 6px", textAlign: "right", fontFamily: "monospace" }}>${Number(p.currentPrice ?? 0).toFixed(2)}</td>
                    <td style={{ padding: "8px 6px", textAlign: "right", fontFamily: "monospace", fontWeight: 600, color: (p.pnl ?? 0) >= 0 ? "#30d158" : "#ff453a" }}>
                      {(p.pnl ?? 0) >= 0 ? "+" : ""}${Number(p.pnl ?? 0).toFixed(2)} ({(p.pnlPct ?? 0) >= 0 ? "+" : ""}{Number(p.pnlPct ?? 0).toFixed(2)}%)
                    </td>
                    <td style={{ padding: "8px 6px" }}>
                      <button onClick={() => closePos.mutate(p.ticker)} style={{ padding: "4px 10px", borderRadius: "6px", border: "1px solid rgba(255,69,58,0.3)", background: "transparent", color: "#ff453a", fontSize: "11px", cursor: "pointer" }}>Close</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: "#6e6e73", fontSize: "13px", textAlign: "center", padding: "20px 0" }}>No open positions. The bot will generate signals and trade when activated.</p>
        )}
      </div>

      {/* ── AI Signals Panel ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <Zap size={14} style={{ color: "#ff9f0a" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#f5f5f7" }}>
            <Tip id="confidence">AI Trade Signals</Tip>
          </span>
          <span style={{ marginLeft: "auto", fontSize: "11px", color: "#6e6e73" }}>
            {Array.isArray(signals) ? signals.length : 0} active signals
          </span>
          <button
            onClick={() => refreshSignals.mutate()}
            disabled={refreshSignals.isPending}
            style={{
              display: "flex", alignItems: "center", gap: "6px",
              padding: "6px 14px", borderRadius: "8px", border: "1px solid rgba(255,159,10,0.3)",
              background: "rgba(255,159,10,0.1)", color: "#ff9f0a", fontSize: "12px", fontWeight: 600,
              cursor: refreshSignals.isPending ? "not-allowed" : "pointer", opacity: refreshSignals.isPending ? 0.6 : 1,
            }}
          >
            <RefreshCw size={12} style={{ animation: refreshSignals.isPending ? "spin 1s linear infinite" : "none" }} />
            {refreshSignals.isPending ? "Scanning..." : "Refresh Signals"}
          </button>
        </div>

        {Array.isArray(signals) && signals.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {signals.map((sig: any, i: number) => (
              <div key={i} style={{
                background: "rgba(255,255,255,0.03)", border: `1px solid rgba(255,255,255,0.06)`,
                borderLeft: `3px solid ${signalColor(sig.type)}`,
                borderRadius: "10px", padding: "12px 14px",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
                  <span style={{ fontFamily: "monospace", fontWeight: 700, fontSize: "15px", color: "#f5f5f7" }}>{sig.ticker}</span>
                  <span style={{
                    padding: "2px 8px", borderRadius: "6px", fontSize: "11px", fontWeight: 700,
                    background: sig.type === "buy" ? "rgba(48,209,88,0.15)" : sig.type === "sell" ? "rgba(255,69,58,0.15)" : "rgba(255,159,10,0.15)",
                    color: signalColor(sig.type),
                  }}>{sig.action}</span>
                  <span style={{ marginLeft: "auto", fontSize: "11px", color: "#6e6e73" }}>
                    {new Date(sig.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p style={{ color: "#a1a1a6", fontSize: "12px", margin: "0 0 8px 0", lineHeight: 1.5 }}>{sig.reason}</p>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "11px", color: "#6e6e73", flexShrink: 0 }}>
                    <Tip id="confidence">Confidence</Tip>
                  </span>
                  <div style={{ flex: 1, height: "4px", background: "rgba(255,255,255,0.08)", borderRadius: "2px", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", width: `${sig.confidence}%`, borderRadius: "2px",
                      background: sig.confidence >= 70 ? "#30d158" : sig.confidence >= 50 ? "#ff9f0a" : "#ff453a",
                      transition: "width 0.5s ease",
                    }} />
                  </div>
                  <span style={{ fontSize: "11px", fontWeight: 700, color: "#f5f5f7", flexShrink: 0 }}>{sig.confidence}%</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "32px 0" }}>
            <Zap size={28} style={{ color: "#3a3a3c", marginBottom: "12px" }} />
            <p style={{ color: "#6e6e73", fontSize: "13px", margin: 0 }}>No signals yet. Click <strong style={{ color: "#ff9f0a" }}>Refresh Signals</strong> to scan top 10 tickers.</p>
          </div>
        )}
      </div>

      {/* ── Backtest Panel ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <BarChart2 size={14} style={{ color: "#0a84ff" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#f5f5f7" }}>
            <Tip id="backtest">Strategy Backtesting</Tip>
          </span>
          <span style={{ marginLeft: "auto", fontSize: "11px", color: "#6e6e73" }}>Historical walk-forward validation</span>
        </div>

        {/* Form row */}
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginBottom: "16px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <span style={{ ...label }}>Ticker</span>
            <input
              value={btTicker}
              onChange={e => setBtTicker(e.target.value.toUpperCase())}
              placeholder="SPY"
              style={{
                background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px", padding: "8px 12px", color: "#f5f5f7", fontSize: "13px",
                fontFamily: "monospace", width: "90px", outline: "none",
              }}
            />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <span style={{ ...label }}>Strategy</span>
            <select
              value={btStrategy}
              onChange={e => setBtStrategy(e.target.value)}
              style={{
                background: "rgba(30,30,35,0.95)", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px", padding: "8px 12px", color: "#f5f5f7", fontSize: "13px", outline: "none",
              }}
            >
              <option value="all">All Strategies</option>
              <option value="momentum">Momentum</option>
              <option value="mean_reversion">Mean Reversion</option>
              <option value="vol_selling">Vol Selling</option>
              <option value="combined">Combined</option>
            </select>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <span style={{ ...label }}>Years</span>
            <select
              value={btYears}
              onChange={e => setBtYears(e.target.value)}
              style={{
                background: "rgba(30,30,35,0.95)", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px", padding: "8px 12px", color: "#f5f5f7", fontSize: "13px", outline: "none",
              }}
            >
              <option value="1">1 year</option>
              <option value="2">2 years</option>
              <option value="3">3 years</option>
              <option value="5">5 years</option>
            </select>
          </div>
          <div style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end" }}>
            <button
              onClick={runBacktest}
              disabled={btLoading}
              style={{
                display: "flex", alignItems: "center", gap: "6px",
                padding: "8px 20px", borderRadius: "8px", border: "none",
                background: btLoading ? "#333" : "#0a84ff", color: "white",
                fontSize: "13px", fontWeight: 600, cursor: btLoading ? "not-allowed" : "pointer",
              }}
            >
              <TrendingUp size={14} />
              {btLoading ? "Running..." : "Run Backtest"}
            </button>
          </div>
        </div>

        {btError && (
          <div style={{ background: "rgba(255,69,58,0.1)", border: "1px solid rgba(255,69,58,0.2)", borderRadius: "8px", padding: "10px 14px", color: "#ff453a", fontSize: "13px", marginBottom: "12px" }}>
            Error: {btError}
          </div>
        )}

        {btLoading && (
          <div style={{ textAlign: "center", padding: "24px 0", color: "#6e6e73", fontSize: "13px" }}>
            Downloading historical data and running simulations... (may take 30–60s)
          </div>
        )}

        {btResults && Array.isArray(btResults.results) && (
          <div>
            <div style={{ fontSize: "12px", color: "#6e6e73", marginBottom: "10px" }}>
              {btResults.ticker} · {btResults.years}yr backtest
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
                <thead>
                  <tr style={{ color: "#6e6e73", textAlign: "left", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                    <th style={{ padding: "8px 10px" }}>Strategy</th>
                    <th style={{ padding: "8px 6px", textAlign: "right" }}><Tip id="annualReturn">Annual Return</Tip></th>
                    <th style={{ padding: "8px 6px", textAlign: "right" }}><Tip id="totalReturn">Total Return</Tip></th>
                    <th style={{ padding: "8px 6px", textAlign: "right" }}><Tip id="sharpe">Sharpe</Tip></th>
                    <th style={{ padding: "8px 6px", textAlign: "right" }}><Tip id="maxDrawdown">Max DD</Tip></th>
                    <th style={{ padding: "8px 6px", textAlign: "right" }}><Tip id="winrate">Win Rate</Tip></th>
                    <th style={{ padding: "8px 6px", textAlign: "right" }}>Trades</th>
                  </tr>
                </thead>
                <tbody>
                  {btResults.results.map((r: any, i: number) => (
                    <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "10px 10px", color: "#f5f5f7", fontWeight: 600, fontSize: "12px" }}>
                        {r.error ? (
                          <span style={{ color: "#ff453a" }}>{r.strategy} — {r.error}</span>
                        ) : r.strategy}
                      </td>
                      {!r.error && (<>
                        <td style={{ padding: "10px 6px", textAlign: "right", fontFamily: "monospace", color: (r.annualReturn ?? 0) >= 0 ? "#30d158" : "#ff453a" }}>
                          {(r.annualReturn ?? 0) >= 0 ? "+" : ""}{Number(r.annualReturn ?? 0).toFixed(1)}%
                        </td>
                        <td style={{ padding: "10px 6px", textAlign: "right", fontFamily: "monospace", color: (r.totalReturn ?? 0) >= 0 ? "#30d158" : "#ff453a" }}>
                          {(r.totalReturn ?? 0) >= 0 ? "+" : ""}{Number(r.totalReturn ?? 0).toFixed(1)}%
                        </td>
                        <td style={{ padding: "10px 6px", textAlign: "right", fontFamily: "monospace", fontWeight: 700, color: sharpeColor(r.sharpe ?? 0) }}>
                          {Number(r.sharpe ?? 0).toFixed(2)}
                        </td>
                        <td style={{ padding: "10px 6px", textAlign: "right", fontFamily: "monospace", color: "#ff453a" }}>
                          {Number(r.maxDrawdown ?? 0).toFixed(1)}%
                        </td>
                        <td style={{ padding: "10px 6px", textAlign: "right", fontFamily: "monospace", color: (r.winRate ?? 0) >= 55 ? "#30d158" : "#a1a1a6" }}>
                          {Number(r.winRate ?? 0).toFixed(1)}%
                        </td>
                        <td style={{ padding: "10px 6px", textAlign: "right", color: "#6e6e73" }}>
                          {r.totalTrades?.toLocaleString()}
                        </td>
                      </>)}
                      {r.error && <td colSpan={6} />}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: "10px", fontSize: "11px", color: "#3a3a3c", lineHeight: 1.6 }}>
              Sharpe: <span style={{ color: "#30d158" }}>green &gt; 1.0</span>, <span style={{ color: "#ff9f0a" }}>amber &gt; 0.5</span>, <span style={{ color: "#ff453a" }}>red &lt; 0.5</span>. Past performance does not guarantee future results.
            </div>
          </div>
        )}
      </div>

      {/* ── Chart ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <ChartPage />
      </div>

      {/* ── Security Controls ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <Shield size={14} style={{ color: "#30d158" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#f5f5f7" }}>Security Controls</span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px", marginBottom: "16px" }}>
          {/* Kill Switch status */}
          <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: "10px", padding: "14px", border: `1px solid ${isKilled ? "rgba(255,69,58,0.3)" : "rgba(255,255,255,0.06)"}` }}>
            <div style={{ ...label, marginBottom: "8px" }}><Tip id="killSwitch">Kill Switch</Tip></div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{
                width: "44px", height: "24px", borderRadius: "12px", position: "relative", cursor: "pointer",
                background: isKilled ? "#ff453a" : "rgba(255,255,255,0.1)",
                transition: "background 0.2s",
              }} onClick={() => killBot.mutate()}>
                <div style={{
                  position: "absolute", top: "3px", left: isKilled ? "23px" : "3px",
                  width: "18px", height: "18px", borderRadius: "50%", background: "white",
                  transition: "left 0.2s", boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
                }} />
              </div>
              <span style={{ fontSize: "13px", fontWeight: 600, color: isKilled ? "#ff453a" : "#6e6e73" }}>
                {isKilled ? "ACTIVE" : "OFF"}
              </span>
            </div>
          </div>

          {/* Daily loss limit */}
          <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: "10px", padding: "14px", border: "1px solid rgba(255,255,255,0.06)" }}>
            <div style={{ ...label, marginBottom: "4px" }}><Tip id="dailyLossLimit">Daily Loss Limit</Tip></div>
            <div style={{ fontSize: "22px", fontWeight: 700, fontFamily: "monospace", color: "#ff9f0a" }}>−3%</div>
            <div style={{ fontSize: "11px", color: "#6e6e73", marginTop: "4px" }}>Auto-halts trading if exceeded</div>
          </div>

          {/* Max position size */}
          <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: "10px", padding: "14px", border: "1px solid rgba(255,255,255,0.06)" }}>
            <div style={{ ...label, marginBottom: "4px" }}><Tip id="positionSize">Max Position Size</Tip></div>
            <div style={{ fontSize: "22px", fontWeight: 700, fontFamily: "monospace", color: "#0a84ff" }}>5%</div>
            <div style={{ fontSize: "11px", color: "#6e6e73", marginTop: "4px" }}>Per trade, of portfolio</div>
          </div>

          {/* Max exposure */}
          <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: "10px", padding: "14px", border: "1px solid rgba(255,255,255,0.06)" }}>
            <div style={{ ...label, marginBottom: "4px" }}><Tip id="totalExposure">Max Exposure</Tip></div>
            <div style={{ fontSize: "22px", fontWeight: 700, fontFamily: "monospace", color: "#bf5af2" }}>50%</div>
            <div style={{ fontSize: "11px", color: "#6e6e73", marginTop: "4px" }}>Max portfolio invested</div>
          </div>
        </div>

        {/* Mode & security badges */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
          <span style={{ padding: "4px 12px", borderRadius: "20px", fontSize: "11px", fontWeight: 600, background: "rgba(10,132,255,0.15)", color: "#0a84ff", border: "1px solid rgba(10,132,255,0.2)" }}>
            Paper Trading Mode
          </span>
          <span style={{ padding: "4px 12px", borderRadius: "20px", fontSize: "11px", fontWeight: 600, background: "rgba(48,209,88,0.1)", color: "#30d158", border: "1px solid rgba(48,209,88,0.2)" }}>
            {status?.auditLogCount ?? 0} Audit Entries
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "5px", padding: "4px 12px", borderRadius: "20px", fontSize: "11px", fontWeight: 600, background: "rgba(255,255,255,0.04)", color: "#a1a1a6", border: "1px solid rgba(255,255,255,0.08)" }}>
            <Lock size={10} /> API keys encrypted
          </span>
        </div>
      </div>

      {/* ── Audit Log ── */}
      <div style={{ ...card }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
          <Clock size={14} style={{ color: "#0a84ff" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#f5f5f7" }}>Activity Log</span>
          <span style={{ marginLeft: "auto", fontSize: "11px", color: "#6e6e73" }}>Every trade and decision is logged</span>
        </div>
        <div style={{ maxHeight: "250px", overflowY: "auto" }}>
          {Array.isArray(audit) && audit.length > 0 ? (
            audit.map((a: any, i: number) => (
              <div key={i} style={{ display: "flex", gap: "10px", padding: "6px 0", borderBottom: "1px solid rgba(255,255,255,0.04)", fontSize: "12px" }}>
                <span style={{ color: "#6e6e73", fontFamily: "monospace", flexShrink: 0, width: "130px" }}>{new Date(a.time).toLocaleString()}</span>
                <span style={{ color: "#0a84ff", fontWeight: 600, flexShrink: 0, width: "80px" }}>{a.action}</span>
                <span style={{ color: "#a1a1a6" }}>{a.detail}</span>
              </div>
            ))
          ) : (
            <p style={{ color: "#6e6e73", fontSize: "12px", textAlign: "center", padding: "16px 0" }}>No activity yet. Start the bot to begin.</p>
          )}
        </div>
      </div>

    </div>
  );
}
