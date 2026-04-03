import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { ShieldAlert, Play, Square, Power, TrendingUp, TrendingDown, DollarSign, Activity, Clock, AlertTriangle } from "lucide-react";

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

// ─── Component ───────────────────────────────────────────────────────────────
export default function BotDashboard() {
  // Fetch account
  const { data: acct } = useQuery({ queryKey: ["/api/bot/account"], queryFn: async () => { const r = await apiRequest("GET", "/api/bot/account"); return r.json(); }, refetchInterval: 10000 });
  // Fetch positions
  const { data: positions } = useQuery({ queryKey: ["/api/bot/positions"], queryFn: async () => { const r = await apiRequest("GET", "/api/bot/positions"); return r.json(); }, refetchInterval: 10000 });
  // Fetch bot status
  const { data: status } = useQuery({ queryKey: ["/api/bot/status"], queryFn: async () => { const r = await apiRequest("GET", "/api/bot/status"); return r.json(); }, refetchInterval: 5000 });
  // Fetch audit log
  const { data: audit } = useQuery({ queryKey: ["/api/bot/audit"], queryFn: async () => { const r = await apiRequest("GET", "/api/bot/audit"); return r.json(); }, refetchInterval: 15000 });

  // Mutations
  const startBot = useMutation({ mutationFn: () => apiRequest("POST", "/api/bot/start").then(r => r.json()), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); } });
  const stopBot = useMutation({ mutationFn: () => apiRequest("POST", "/api/bot/stop").then(r => r.json()), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); } });
  const killBot = useMutation({ mutationFn: () => apiRequest("POST", "/api/bot/kill").then(r => r.json()), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); } });
  const closePos = useMutation({ mutationFn: (ticker: string) => apiRequest("POST", "/api/bot/close", { ticker }).then(r => r.json()), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["/api/bot/positions"] }); queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] }); } });

  const dailyPnl = acct?.dailyPnL ?? 0;
  const dailyPnlPct = acct?.dailyPnLPct ?? 0;
  const isKilled = status?.killSwitch ?? false;
  const isActive = status?.active ?? false;

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "24px 16px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h1 style={{ fontSize: "1.4rem", fontWeight: 700, color: "#f5f5f7", margin: 0 }}>Trading Bot</h1>
          <div style={{ display: "flex", gap: "8px", marginTop: "6px", alignItems: "center" }}>
            <span style={{ padding: "3px 10px", borderRadius: "20px", fontSize: "11px", fontWeight: 600,
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

      {/* Account Overview */}
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
            <span style={{ fontSize: "14px", marginLeft: "6px" }}>({dailyPnlPct >= 0 ? "+" : ""}{dailyPnlPct.toFixed(2)}%)</span>
          </div>
        </div>
        <div style={card}>
          <div style={label}><Tip id="buyingPower">Buying Power</Tip></div>
          <div style={{ ...bigNum, color: "#a1a1a6", fontSize: "22px" }}>${(acct?.buyingPower ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
        </div>
      </div>

      {/* Positions */}
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
                    <td style={{ padding: "8px 6px", textAlign: "right", fontFamily: "monospace", color: "#a1a1a6" }}>${p.entryPrice?.toFixed(2)}</td>
                    <td style={{ padding: "8px 6px", textAlign: "right", fontFamily: "monospace" }}>${p.currentPrice?.toFixed(2)}</td>
                    <td style={{ padding: "8px 6px", textAlign: "right", fontFamily: "monospace", fontWeight: 600, color: (p.pnl ?? 0) >= 0 ? "#30d158" : "#ff453a" }}>
                      {(p.pnl ?? 0) >= 0 ? "+" : ""}${(p.pnl ?? 0).toFixed(2)} ({(p.pnlPct ?? 0) >= 0 ? "+" : ""}{(p.pnlPct ?? 0).toFixed(2)}%)
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

      {/* Audit Log */}
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
