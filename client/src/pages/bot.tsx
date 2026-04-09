import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  ShieldAlert, Play, Square, TrendingUp, Activity, Clock,
  RefreshCw, Shield, Lock, BarChart2, Zap, Bell, ArrowUp, ArrowDown,
  DollarSign, History, Sunrise,
} from "lucide-react";
import TradeCharts from "@/components/TradeChart";

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
  performance: "Live performance tracking — win rate, P&L, and equity curve across all bot trades.",
  notifications: "Real-time alerts for trades executed, stop losses hit, earnings events, and daily summaries.",
};

function Tip({ id, children }: { id: string; children: React.ReactNode }) {
  const [show, setShow] = useState(false);
  const tip = TIPS[id];
  if (!tip) return <>{children}</>;
  return (
    <span style={{ position: "relative", cursor: "help", borderBottom: "1px dotted rgba(0, 229, 255, 0.3)" }}
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span style={{ position: "absolute", bottom: "calc(100% + 8px)", left: "50%", transform: "translateX(-50%)",
          background: "rgba(3, 8, 15, 0.95)", border: "1px solid rgba(0, 229, 255, 0.15)", borderRadius: "4px",
          padding: "10px 14px", fontSize: "12px", color: "#c8d6e5", lineHeight: 1.5, width: "260px", zIndex: 100,
          backdropFilter: "blur(12px)", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}>
          {tip}
        </span>
      )}
    </span>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const card: React.CSSProperties = { background: "rgba(0, 20, 40, 0.5)", border: "1px solid rgba(0, 229, 255, 0.1)", borderRadius: "6px", padding: "20px", backdropFilter: "blur(20px)" };
const label: React.CSSProperties = { fontSize: "11px", color: "#4a5c70", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px", fontFamily: "'JetBrains Mono', monospace" };
const bigNum: React.CSSProperties = { fontSize: "clamp(18px, 3.5vw, 28px)", fontWeight: 700, fontFamily: "'JetBrains Mono', 'Fira Code', monospace", color: "#c8d6e5", wordBreak: "break-word", overflow: "hidden", textOverflow: "ellipsis" };

// ─── Sharpe color helper ──────────────────────────────────────────────────────
function sharpeColor(v: number) {
  if (v >= 1) return "#30d158";
  if (v >= 0.5) return "#d4a017";
  return "#ff453a";
}

// ─── Signal type color ────────────────────────────────────────────────────────
function signalColor(type: string) {
  if (type === "buy") return "#30d158";
  if (type === "sell") return "#ff453a";
  return "#d4a017";
}

// ─── Mini equity curve chart ──────────────────────────────────────────────────
function MiniEquityCurve({ data }: { data: Array<{ date: string; value: number; pnl: number }> }) {
  if (!data || data.length < 2) {
    return (
      <div style={{ height: "48px", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: "11px", color: "#2a3a4c" }}>Equity curve builds as trades complete</span>
      </div>
    );
  }

  const reversed = [...data].reverse();
  const values = reversed.map(d => d.value);
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const range = maxV - minV || 1;
  const w = 300;
  const h = 48;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - minV) / range) * h;
    return `${x},${y}`;
  }).join(" ");

  const isUp = values[values.length - 1] >= values[0];

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ display: "block" }}>
      <polyline
        points={pts}
        fill="none"
        stroke={isUp ? "#30d158" : "#ff453a"}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ─── Notification Bell ────────────────────────────────────────────────────────
function NotificationBell({ notifications, onMarkRead }: {
  notifications: Array<{ time: string; type: string; message: string; read: boolean }>;
  onMarkRead: () => void;
}) {
  const [open, setOpen] = useState(false);
  const unreadCount = notifications.filter(n => !n.read).length;

  const typeColor = (type: string) => {
    if (type === "alert" || type === "stop_loss") return "#ff453a";
    if (type === "profit" || type === "trade") return "#30d158";
    if (type === "earnings") return "#d4a017";
    return "#00e5ff";
  };

  const typeIcon = (type: string) => {
    if (type === "alert" || type === "stop_loss") return "🛑";
    if (type === "profit") return "✅";
    if (type === "trade") return "📈";
    if (type === "earnings") return "📊";
    if (type === "daily_summary") return "📋";
    return "ℹ️";
  };

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => {
          setOpen(o => !o);
          if (!open && unreadCount > 0) onMarkRead();
        }}
        style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          width: "36px", height: "36px", borderRadius: "4px",
          border: "1px solid rgba(0, 229, 255, 0.12)", background: "rgba(0, 229, 255, 0.08)",
          cursor: "pointer", position: "relative", color: "#a1a1a6",
        }}
        title="Notifications"
      >
        <Bell size={15} />
        {unreadCount > 0 && (
          <span style={{
            position: "absolute", top: "-4px", right: "-4px",
            background: "#ff453a", borderRadius: "50%",
            width: "16px", height: "16px", fontSize: "9px",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "white", fontWeight: 700, border: "2px solid #000",
          }}>
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div style={{
          position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 200,
          width: "320px", maxHeight: "400px", overflowY: "auto",
          background: "rgba(3, 8, 15, 0.97)", border: "1px solid rgba(0, 229, 255, 0.12)",
          borderRadius: "6px", boxShadow: "0 16px 48px rgba(0,0,0,0.6)",
          backdropFilter: "blur(20px)",
        }}>
          <div style={{
            padding: "12px 16px", borderBottom: "1px solid rgba(0, 229, 255, 0.1)",
            display: "flex", alignItems: "center", justifyContent: "space-between",
          }}>
            <span style={{ fontSize: "13px", fontWeight: 600, color: "#c8d6e5" }}>
              <Tip id="notifications">Notifications</Tip>
            </span>
            {notifications.length > 0 && (
              <button onClick={onMarkRead} style={{
                fontSize: "11px", color: "#00e5ff", background: "none", border: "none", cursor: "pointer",
              }}>
                Mark all read
              </button>
            )}
          </div>
          {notifications.length === 0 ? (
            <div style={{ padding: "24px", textAlign: "center", color: "#4a5c70", fontSize: "12px" }}>
              No notifications yet
            </div>
          ) : (
            notifications.map((n, i) => (
              <div key={i} style={{
                padding: "10px 16px", borderBottom: "1px solid rgba(0, 15, 30, 0.4)",
                background: n.read ? "transparent" : "rgba(0,229,255,0.05)",
                display: "flex", gap: "10px", alignItems: "flex-start",
              }}>
                <span style={{ fontSize: "14px", marginTop: "1px" }}>{typeIcon(n.type)}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: "12px", color: "#c8d6e5", lineHeight: 1.4, wordBreak: "break-word" }}>
                    {n.message}
                  </div>
                  <div style={{ fontSize: "10px", color: "#4a5c70", marginTop: "3px" }}>
                    {new Date(n.time).toLocaleString()}
                  </div>
                </div>
                {!n.read && (
                  <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#00e5ff", flexShrink: 0, marginTop: "5px" }} />
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

// ─── Calendar Banner component ────────────────────────────────────────────────
function CalendarBanner() {
  const { data } = useQuery({
    queryKey: ["/api/bot/calendar"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/calendar"); return r.json(); },
    staleTime: 600000, // 10 min
  });

  if (!data || !data.tomorrow) return null;

  const { tomorrow, holidays } = data;
  const isAlert = tomorrow.status === "holiday" || tomorrow.status === "early_close";
  const bgColor = isAlert ? "rgba(212,160,23,0.1)" : "rgba(0,255,65,0.06)";
  const borderColor = isAlert ? "rgba(212,160,23,0.2)" : "rgba(0,255,65,0.1)";
  const textColor = isAlert ? "#d4a017" : "#4a5c70";

  return (
    <div style={{ padding: "10px 14px", background: bgColor, border: `1px solid ${borderColor}`, borderRadius: "4px", marginBottom: "16px", fontSize: "12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "14px" }}>{isAlert ? "⚠️" : "📅"}</span>
        <span style={{ color: textColor, fontWeight: 600 }}>{tomorrow.note}</span>
      </div>
      {holidays && holidays.length > 0 && (
        <div style={{ marginTop: "6px", color: "#4a5c70", fontSize: "11px" }}>
          Upcoming closures: {holidays.map((h: any) => `${h.name} (${h.date})`).join(" \u00b7 ")}
        </div>
      )}
    </div>
  );
}

// ─── Performance Card ─────────────────────────────────────────────────────────
function PerformanceCard({ perf }: { perf: any }) {
  if (!perf) {
    return (
      <div style={card}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
          <TrendingUp size={14} style={{ color: "#00e5ff" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>
            <Tip id="performance">Performance</Tip>
          </span>
        </div>
        <p style={{ color: "#4a5c70", fontSize: "13px", textAlign: "center", padding: "16px 0" }}>
          Performance data builds as trades complete.
        </p>
      </div>
    );
  }

  const { totalTrades = 0, winRate = 0, totalPnl = 0, avgGain = 0, avgLoss = 0, bestTrade, worstTrade, equityCurve } = perf || {};
  const winRateColor = winRate >= 60 ? "#30d158" : winRate >= 45 ? "#d4a017" : "#ff453a";
  const pnlColor = totalPnl >= 0 ? "#30d158" : "#ff453a";

  return (
    <div style={{ ...card, marginBottom: "20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
        <TrendingUp size={14} style={{ color: "#00e5ff" }} />
        <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>
          <Tip id="performance">Performance</Tip>
        </span>
        <span style={{ marginLeft: "auto", fontSize: "11px", color: "#4a5c70" }}>
          {totalTrades} total trades
        </span>
      </div>

      {/* Stats row */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
        gap: "12px",
        marginBottom: "16px",
      }}>
        <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "12px" }}>
          <div style={label}>Total P&L</div>
          <div style={{ fontSize: "20px", fontWeight: 700, fontFamily: "monospace", color: pnlColor }}>
            {totalPnl >= 0 ? "+" : ""}${(totalPnl ?? 0).toFixed(2)}
          </div>
        </div>
        <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "12px" }}>
          <div style={label}><Tip id="winrate">Win Rate</Tip></div>
          <div style={{ fontSize: "20px", fontWeight: 700, fontFamily: "monospace", color: winRateColor }}>
            {(winRate ?? 0).toFixed(1)}%
          </div>
        </div>
        <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "12px" }}>
          <div style={label}>Avg Gain</div>
          <div style={{ fontSize: "20px", fontWeight: 700, fontFamily: "monospace", color: "#30d158" }}>
            +{(avgGain ?? 0).toFixed(2)}%
          </div>
        </div>
        <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "12px" }}>
          <div style={label}>Avg Loss</div>
          <div style={{ fontSize: "20px", fontWeight: 700, fontFamily: "monospace", color: "#ff453a" }}>
            {(avgLoss ?? 0).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Best / Worst trade */}
      {(bestTrade || worstTrade) && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "16px" }}>
          {bestTrade && (
            <div style={{ background: "rgba(0,255,65,0.06)", border: "1px solid rgba(0,255,65,0.15)", borderRadius: "4px", padding: "10px" }}>
              <div style={{ fontSize: "10px", color: "#4a5c70", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>
                Best Trade
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <ArrowUp size={12} style={{ color: "#30d158" }} />
                <span style={{ fontFamily: "monospace", fontWeight: 700, color: "#c8d6e5", fontSize: "13px" }}>
                  {bestTrade.ticker}
                </span>
                <span style={{ fontSize: "12px", color: "#30d158", marginLeft: "auto", fontWeight: 600 }}>
                  +{bestTrade.(pnlPct ?? 0).toFixed(2)}%
                </span>
              </div>
            </div>
          )}
          {worstTrade && (
            <div style={{ background: "rgba(255,51,51,0.06)", border: "1px solid rgba(255,51,51,0.15)", borderRadius: "4px", padding: "10px" }}>
              <div style={{ fontSize: "10px", color: "#4a5c70", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>
                Worst Trade
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <ArrowDown size={12} style={{ color: "#ff453a" }} />
                <span style={{ fontFamily: "monospace", fontWeight: 700, color: "#c8d6e5", fontSize: "13px" }}>
                  {worstTrade.ticker}
                </span>
                <span style={{ fontSize: "12px", color: "#ff453a", marginLeft: "auto", fontWeight: 600 }}>
                  {worstTrade.(pnlPct ?? 0).toFixed(2)}%
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Equity curve */}
      {equityCurve && equityCurve.length >= 2 && (
        <div>
          <div style={{ ...label, marginBottom: "8px" }}>Equity Curve</div>
          <MiniEquityCurve data={equityCurve} />
        </div>
      )}

      {/* No trades yet */}
      {totalTrades === 0 && (
        <p style={{ color: "#4a5c70", fontSize: "12px", textAlign: "center", padding: "8px 0" }}>
          Performance data builds as the bot executes trades.
        </p>
      )}
    </div>
  );
}

// ─── Performance Dashboard ──────────────────────────────────────────────────
function PerformanceDashboard({ perfData }: { perfData: any }) {
  const [open, setOpen] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Derive metrics from perfData
  const totalTrades: number = perfData?.totalTrades ?? 0;
  const winRate: number = perfData?.winRate ?? 0;
  const profitFactor: number = perfData?.profitFactor ?? 0;
  const avgGain: number = perfData?.avgGain ?? 0;
  const avgLoss: number = perfData?.avgLoss ?? 0;
  const drawdown: number = perfData?.currentDrawdown ?? perfData?.maxDrawdown ?? 0;
  const equityCurve: Array<{ date: string; value: number }> = Array.isArray(perfData?.equityCurve)
    ? perfData.equityCurve.map((d: any) => ({ date: d.date ?? "", value: d.value ?? 0 }))
    : [];
  const recentTrades: Array<any> = Array.isArray(perfData?.recentTrades)
    ? perfData.recentTrades.slice(0, 10)
    : Array.isArray(perfData?.trades)
    ? perfData.trades.slice(0, 10)
    : [];

  // Win rate color
  const winRateColor = winRate > 55 ? "#30d158" : winRate < 45 ? "#ff453a" : "#d4a017";
  // Profit factor color
  const pfColor = profitFactor > 1.5 ? "#30d158" : profitFactor < 1.0 ? "#ff453a" : "#d4a017";
  // Drawdown color (negative = bad)
  const ddColor = drawdown <= -10 ? "#ff453a" : drawdown <= -5 ? "#d4a017" : "#30d158";

  // ── Canvas equity curve ──
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Use display size
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.clearRect(0, 0, W, H);

    const PAD_LEFT = 56;
    const PAD_RIGHT = 16;
    const PAD_TOP = 16;
    const PAD_BOTTOM = 28;
    const chartW = W - PAD_LEFT - PAD_RIGHT;
    const chartH = H - PAD_TOP - PAD_BOTTOM;

    if (equityCurve.length < 2) {
      // Empty state
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#2a3a4c";
      ctx.textAlign = "center";
      ctx.fillText("Equity curve builds as trades complete", W / 2, H / 2);
      return;
    }

    const values = equityCurve.map(d => d.value);
    const minV = Math.min(...values);
    const maxV = Math.max(...values);
    const peak = Math.max(...values);
    const range = maxV - minV || 1;

    // Grid lines
    const gridLines = 4;
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = "rgba(0,229,255,0.07)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= gridLines; i++) {
      const y = PAD_TOP + (i / gridLines) * chartH;
      ctx.beginPath();
      ctx.moveTo(PAD_LEFT, y);
      ctx.lineTo(PAD_LEFT + chartW, y);
      ctx.stroke();
      // Y axis labels
      const val = maxV - (i / gridLines) * range;
      ctx.setLineDash([]);
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#2a3a4c";
      ctx.textAlign = "right";
      ctx.fillText(
        val >= 1000 ? `$${(val / 1000).toFixed(1)}k` : `$${val.toFixed(0)}`,
        PAD_LEFT - 4,
        y + 3
      );
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = "rgba(0,229,255,0.07)";
    }
    ctx.setLineDash([]);

    // Build path
    const toX = (i: number) => PAD_LEFT + (i / (values.length - 1)) * chartW;
    const toY = (v: number) => PAD_TOP + ((maxV - v) / range) * chartH;

    // Area fill
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(values[0]));
    for (let i = 1; i < values.length; i++) ctx.lineTo(toX(i), toY(values[i]));
    ctx.lineTo(toX(values.length - 1), PAD_TOP + chartH);
    ctx.lineTo(toX(0), PAD_TOP + chartH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, PAD_TOP, 0, PAD_TOP + chartH);
    grad.addColorStop(0, "rgba(0,255,213,0.18)");
    grad.addColorStop(1, "rgba(0,255,213,0.00)");
    ctx.fillStyle = grad;
    ctx.fill();

    // Equity line
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(values[0]));
    for (let i = 1; i < values.length; i++) ctx.lineTo(toX(i), toY(values[i]));
    ctx.strokeStyle = "#00ffd5";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();

    // Peak line (dotted amber)
    const peakY = toY(peak);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(251,191,36,0.6)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD_LEFT, peakY);
    ctx.lineTo(PAD_LEFT + chartW, peakY);
    ctx.stroke();
    ctx.setLineDash([]);
    // Peak label
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(251,191,36,0.7)";
    ctx.textAlign = "left";
    ctx.fillText("PEAK", PAD_LEFT + chartW - 36, peakY - 3);

    // X axis date labels (show up to 5)
    const dates = equityCurve.map(d => d.date);
    const labelCount = Math.min(5, dates.length);
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#2a3a4c";
    ctx.textAlign = "center";
    for (let i = 0; i < labelCount; i++) {
      const idx = Math.round((i / (labelCount - 1)) * (dates.length - 1));
      const x = toX(idx);
      const dateStr = dates[idx] ? dates[idx].slice(5) : ""; // MM-DD
      ctx.fillText(dateStr, x, PAD_TOP + chartH + 16);
    }
  }, [equityCurve]);

  useEffect(() => {
    if (open) {
      // Small delay to ensure layout
      const t = setTimeout(drawChart, 50);
      return () => clearTimeout(t);
    }
  }, [open, drawChart]);

  useEffect(() => {
    if (!open) return;
    const handleResize = () => drawChart();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [open, drawChart]);

  const sectionHeader: React.CSSProperties = {
    display: "flex", alignItems: "center", gap: "8px",
    cursor: "pointer", userSelect: "none",
    marginBottom: open ? "20px" : 0,
  };

  const metricBox: React.CSSProperties = {
    background: "rgba(0, 15, 30, 0.5)",
    border: "1px solid rgba(0, 229, 255, 0.08)",
    borderRadius: "4px",
    padding: "14px 16px",
  };

  return (
    <div style={{ ...card, marginBottom: "20px" }}>
      {/* Header — click to collapse */}
      <div style={sectionHeader} onClick={() => setOpen(o => !o)}>
        <TrendingUp size={14} style={{ color: "#00ffd5" }} />
        <span style={{ fontSize: "14px", fontWeight: 700, color: "#c8d6e5",
          fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em" }}>
          PERFORMANCE
        </span>
        <span style={{ fontSize: "10px", color: "#4a5c70", textTransform: "uppercase",
          letterSpacing: "0.5px", marginLeft: "4px" }}>
          {totalTrades} trades
        </span>
        <span style={{ marginLeft: "auto", fontSize: "18px", color: "#00ffd5", lineHeight: 1 }}>
          {open ? "−" : "+"}
        </span>
      </div>

      {open && (
        <>
          {/* ── Metrics grid ── */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
            gap: "10px",
            marginBottom: "20px",
          }}>
            {/* Total Trades */}
            <div style={metricBox}>
              <div style={label}>Total Trades</div>
              <div style={{ ...bigNum, color: "#c8d6e5" }}>{totalTrades}</div>
            </div>

            {/* Win Rate */}
            <div style={{ ...metricBox, borderColor: `${winRateColor}22` }}>
              <div style={label}><Tip id="winrate">Win Rate</Tip></div>
              <div style={{ ...bigNum, color: winRateColor }}>
                {(winRate ?? 0).toFixed(1)}%
              </div>
              <div style={{ marginTop: "6px", height: "3px", background: "rgba(0,229,255,0.08)", borderRadius: "2px", overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${Math.min(winRate, 100)}%`, background: winRateColor, borderRadius: "2px", transition: "width 0.5s ease" }} />
              </div>
            </div>

            {/* Profit Factor */}
            <div style={{ ...metricBox, borderColor: `${pfColor}22` }}>
              <div style={label}>Profit Factor</div>
              <div style={{ ...bigNum, color: pfColor }}>
                {profitFactor > 0 ? (profitFactor ?? 0).toFixed(2) : "—"}
              </div>
              <div style={{ fontSize: "10px", color: "#4a5c70", marginTop: "4px" }}>
                {profitFactor >= 1.5 ? "Excellent" : profitFactor >= 1.0 ? "Acceptable" : profitFactor > 0 ? "Below 1.0" : "No data"}
              </div>
            </div>

            {/* Avg Win */}
            <div style={metricBox}>
              <div style={label}>Avg Win</div>
              <div style={{ ...bigNum, color: "#30d158" }}>
                {avgGain > 0 ? `+${(avgGain ?? 0).toFixed(2)}%` : "—"}
              </div>
            </div>

            {/* Avg Loss */}
            <div style={metricBox}>
              <div style={label}>Avg Loss</div>
              <div style={{ ...bigNum, color: "#ff453a" }}>
                {avgLoss !== 0 ? `${avgLoss > 0 ? "-" : ""}${Math.abs(avgLoss).toFixed(2)}%` : "—"}
              </div>
            </div>

            {/* Drawdown */}
            <div style={{ ...metricBox, borderColor: drawdown < -5 ? "rgba(255,68,68,0.2)" : "rgba(0,229,255,0.08)" }}>
              <div style={label}><Tip id="drawdown">Drawdown</Tip></div>
              <div style={{ ...bigNum, color: ddColor }}>
                {drawdown !== 0 ? `${(drawdown ?? 0).toFixed(2)}%` : "0.00%"}
              </div>
              <div style={{ fontSize: "10px", color: "#4a5c70", marginTop: "4px" }}>From peak</div>
            </div>
          </div>

          {/* ── Equity Curve Canvas ── */}
          <div style={{ marginBottom: "20px" }}>
            <div style={{ ...label, marginBottom: "10px", display: "flex", alignItems: "center", gap: "12px" }}>
              Equity Curve
              <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                <span style={{ width: "18px", height: "2px", background: "#00ffd5", display: "inline-block" }} />
                <span style={{ fontSize: "9px", color: "#4a5c70" }}>Equity</span>
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                <span style={{ width: "18px", borderTop: "1px dashed rgba(251,191,36,0.6)", display: "inline-block" }} />
                <span style={{ fontSize: "9px", color: "#4a5c70" }}>Peak</span>
              </span>
            </div>
            <div style={{
              background: "rgba(0, 10, 20, 0.6)",
              border: "1px solid rgba(0, 229, 255, 0.08)",
              borderRadius: "4px",
              overflow: "hidden",
            }}>
              <canvas
                ref={canvasRef}
                style={{ width: "100%", height: "160px", display: "block" }}
              />
            </div>
          </div>

          {/* ── Recent Trades Table ── */}
          <div>
            <div style={{ ...label, marginBottom: "10px" }}>Recent Trades (Last 10)</div>
            {recentTrades.length > 0 ? (
              <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", minWidth: "420px" }}>
                  <thead>
                    <tr style={{ color: "#4a5c70", textAlign: "left", borderBottom: "1px solid rgba(0, 229, 255, 0.1)" }}>
                      <th style={{ padding: "7px 10px", fontWeight: 500 }}>Ticker</th>
                      <th style={{ padding: "7px 6px", fontWeight: 500 }}>Side</th>
                      <th style={{ padding: "7px 6px", textAlign: "right", fontWeight: 500 }}>P&L %</th>
                      <th style={{ padding: "7px 6px", fontWeight: 500 }}>Strategy</th>
                      <th style={{ padding: "7px 6px", fontWeight: 500 }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentTrades.map((t: any, i: number) => {
                      const pnlPct = t.pnlPct ?? t.pnl_pct ?? t.returnPct ?? 0;
                      const isProfit = pnlPct >= 0;
                      return (
                        <tr key={i} style={{
                          borderBottom: "1px solid rgba(0, 15, 30, 0.5)",
                          background: isProfit
                            ? "rgba(48, 209, 88, 0.04)"
                            : "rgba(255, 68, 68, 0.04)",
                        }}>
                          <td style={{ padding: "7px 10px", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#c8d6e5" }}>
                            {t.ticker ?? t.symbol ?? "—"}
                          </td>
                          <td style={{ padding: "7px 6px", color: (t.side ?? t.direction ?? "long") === "short" ? "#ff453a" : "#30d158", fontFamily: "monospace", fontSize: "11px", fontWeight: 600, textTransform: "uppercase" }}>
                            {t.side ?? t.direction ?? "long"}
                          </td>
                          <td style={{ padding: "7px 6px", textAlign: "right", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: isProfit ? "#30d158" : "#ff453a" }}>
                            {isProfit ? "+" : ""}{Number(pnlPct).toFixed(2)}%
                          </td>
                          <td style={{ padding: "7px 6px", color: "#a1a1a6", fontSize: "11px", maxWidth: "120px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {t.strategy ?? t.signal ?? "—"}
                          </td>
                          <td style={{ padding: "7px 6px", color: "#4a5c70", fontFamily: "monospace", fontSize: "11px", whiteSpace: "nowrap" }}>
                            {t.date ?? t.exitDate ?? t.time
                              ? new Date(t.date ?? t.exitDate ?? t.time).toLocaleDateString()
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: "24px 0", color: "#2a3a4c", fontSize: "12px" }}>
                No completed trades yet. Performance data builds as the bot executes trades.
              </div>
            )}
          </div>
              <MLModelPanel />
      </>
      )}
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────

// ─── ML Model Panel ──────────────────────────────────────────────────────────
function MLModelPanel() {
  const { data: mlStatus, isLoading } = useQuery({
    queryKey: ["/api/ml/status"],
    refetchInterval: 30000,
  });

  const toggleMutation = useMutation({
    mutationFn: async (enabled: boolean) => {
      const res = await apiRequest("POST", "/api/ml/toggle", { enabled });
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/api/ml/status"] }),
  });

  const s = mlStatus as any;
  const enabled = s?.enabled ?? false;
  const accuracy = s?.last_accuracy;
  const samples = s?.last_samples;
  const ageHours = s?.model_age_hours;
  const modelExists = s?.model_exists ?? false;

  return (
    <div style={{ background: "rgba(0, 20, 40, 0.5)", border: "1px solid rgba(0, 229, 255, 0.1)", borderRadius: "6px", padding: "20px", backdropFilter: "blur(20px)", marginTop: "1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <Zap size={18} color="#00e5ff" />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5", letterSpacing: "0.5px" }}>ML MODEL</span>
          <span style={{
            fontSize: "10px", padding: "2px 8px", borderRadius: "3px",
            background: enabled ? "rgba(48, 209, 88, 0.15)" : "rgba(255, 69, 58, 0.15)",
            color: enabled ? "#30d158" : "#ff453a", fontWeight: 600,
          }}>
            {enabled ? "ON" : "OFF"}
          </span>
        </div>
        <button onClick={() => toggleMutation.mutate(!enabled)} disabled={toggleMutation.isPending}
          style={{
            padding: "6px 16px", borderRadius: "4px", fontSize: "12px", fontWeight: 600, cursor: "pointer",
            border: `1px solid ${enabled ? "rgba(255, 69, 58, 0.3)" : "rgba(48, 209, 88, 0.3)"}`,
            background: enabled ? "rgba(255, 69, 58, 0.1)" : "rgba(48, 209, 88, 0.1)",
            color: enabled ? "#ff453a" : "#30d158",
          }}>
          {toggleMutation.isPending ? "..." : enabled ? "Disable ML" : "Enable ML"}
        </button>
      </div>

      <p style={{ fontSize: "11px", color: "#4a5c70", marginBottom: "16px", lineHeight: 1.6 }}>
        The ML model learns from market data to score trades. It does NOT affect the core system (QQQ floor, VRP harvest, sector rotation). Enable to test if it adds edge.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
        <div>
          <div style={{ fontSize: "11px", color: "#4a5c70", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>Status</div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: modelExists ? "#30d158" : "#ff453a" }}>
            {isLoading ? "..." : modelExists ? "Trained" : "No Model"}
          </div>
        </div>
        <div>
          <div style={{ fontSize: "11px", color: "#4a5c70", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>Accuracy</div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>{accuracy ? `${accuracy}%` : "\u2014"}</div>
        </div>
        <div>
          <div style={{ fontSize: "11px", color: "#4a5c70", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>Samples</div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>{samples ? Number(samples).toLocaleString() : "\u2014"}</div>
        </div>
        <div>
          <div style={{ fontSize: "11px", color: "#4a5c70", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>Model Age</div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: ageHours && ageHours > 48 ? "#d4a017" : "#c8d6e5" }}>
            {ageHours ? `${ageHours}h` : "\u2014"}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Enhanced Positions Component ────────────────────────────────────────────────────
function useMarketClock() {
  const { data } = useQuery({
    queryKey: ["/api/bot/clock"],
    queryFn: async () => {
      const r = await apiRequest("GET", "/api/bot/clock");
      return r.json();
    },
    refetchInterval: 60000,
    staleTime: 30000,
  });
  const clock = data as any;
  return {
    isOpen: clock?.is_open ?? false,
    nextOpen: clock?.next_open ?? "",
    nextClose: clock?.next_close ?? "",
  };
}

function EnhancedPositions({
  positions,
  closePos,
}: {
  positions: any[] | undefined;
  closePos: (ticker: string) => void;
}) {
  const [dollarView, setDollarView] = useState<Record<string, boolean>>({});
  const { isOpen: marketOpen } = useMarketClock();

  const toggleDollarView = (ticker: string) => {
    setDollarView(prev => ({ ...prev, [ticker]: !prev[ticker] }));
  };

  return (
    <div style={{ ...card, marginBottom: "20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
        <Activity size={14} style={{ color: "#00e5ff" }} />
        <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>
          <Tip id="position">Open Positions</Tip>
        </span>
        {!marketOpen && (
          <span
            style={{
              fontSize: "10px", padding: "2px 7px", borderRadius: "3px",
              background: "rgba(212,160,23,0.12)", color: "#d4a017",
              border: "1px solid rgba(212,160,23,0.25)", fontWeight: 600,
              letterSpacing: "0.3px",
            }}
          >
            AFTER HOURS
          </span>
        )}
        <span style={{ marginLeft: "auto", fontSize: "12px", color: "#4a5c70" }}>
          {Array.isArray(positions) ? positions.length : 0} positions
        </span>
      </div>

      {Array.isArray(positions) && positions.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {positions.map((p: any) => {
            const qty = Number(p.qty ?? 0);
            const entry = Number(p.entryPrice ?? p.avg_entry_price ?? 0);
            const current = Number(p.currentPrice ?? p.current_price ?? 0);
            const prevClose = Number(p.lastday_price ?? 0);
            const pnl = Number(p.pnl ?? p.unrealized_pl ?? 0);
            const rawPnlPct = Number(p.pnlPct ?? p.unrealized_plpc ?? 0);
            const pnlPct = p.unrealized_plpc !== undefined ? rawPnlPct * 100 : rawPnlPct;
            const costBasis = qty * entry;
            const marketValue = qty * current;
            const changeTodayRatio = Number(p.change_today ?? 0);
            const changeTodayPct = changeTodayRatio * 100;
            const changeTodayFromClose = prevClose > 0 ? current - prevClose : null;
            const changeTodayDollar = marketValue * changeTodayRatio;
            const pnlColor = pnl >= 0 ? "#30d158" : "#ff453a";
            const todayColor = changeTodayPct >= 0 ? "#30d158" : "#ff453a";
            const showDollar = dollarView[p.ticker] ?? false;
            const showAH = !marketOpen;
            // Asset type detection
            const assetClass: string = p.asset_class ?? "us_equity";
            const isStockShort = p.side === "short" && assetClass === "us_equity";
            const isOption = assetClass.toLowerCase().includes("option");

            return (
              <div
                key={p.ticker}
                data-testid={`position-row-${p.ticker}`}
                style={{
                  background: "rgba(0, 15, 30, 0.4)",
                  border: "1px solid rgba(0, 229, 255, 0.08)",
                  borderRadius: "4px",
                  padding: "12px 14px",
                }}
              >
                {/* Row 1: ticker / side / shares / price / AH badge / asset badge / close button */}
                <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap", marginBottom: "6px" }}>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: "15px", color: "#c8d6e5" }}>
                    {p.ticker}
                  </span>
                  <span style={{
                    fontSize: "11px", fontWeight: 600, padding: "2px 7px", borderRadius: "3px",
                    color: p.side === "long" ? "#30d158" : "#ff453a",
                    background: p.side === "long" ? "rgba(48,209,88,0.12)" : "rgba(255,68,68,0.12)",
                    border: `1px solid ${p.side === "long" ? "rgba(48,209,88,0.25)" : "rgba(255,68,68,0.25)"}`,
                    textTransform: "uppercase" as const,
                  }}>
                    {p.side}
                  </span>
                  {/* FIX 3: STOCK SHORT badge for equity shorts */}
                  {isStockShort && (
                    <span style={{
                      fontSize: "10px", fontWeight: 700, padding: "2px 6px", borderRadius: "3px",
                      color: "#d4a017",
                      background: "rgba(212,160,23,0.12)",
                      border: "1px solid rgba(212,160,23,0.3)",
                      letterSpacing: "0.4px",
                    }}>
                      STOCK SHORT
                    </span>
                  )}
                  {/* Option badge (future-proof) */}
                  {isOption && (
                    <span style={{
                      fontSize: "10px", fontWeight: 700, padding: "2px 6px", borderRadius: "3px",
                      color: "#bf5af2",
                      background: "rgba(191,90,242,0.12)",
                      border: "1px solid rgba(191,90,242,0.3)",
                      letterSpacing: "0.4px",
                    }}>
                      OPTION
                    </span>
                  )}
                  <span style={{ fontSize: "12px", color: "#a1a1a6", fontFamily: "monospace" }}>
                    {qty} shares
                  </span>
                  <span style={{ fontSize: "13px", fontFamily: "monospace", color: "#c8d6e5", marginLeft: "4px" }}>
                    @ ${(current ?? 0).toFixed(2)}
                  </span>
                  {showAH && (
                    <span
                      data-testid={`ah-badge-${p.ticker}`}
                      style={{
                        fontSize: "9px", padding: "1px 5px", borderRadius: "2px",
                        background: "rgba(212,160,23,0.15)", color: "#d4a017",
                        border: "1px solid rgba(212,160,23,0.3)", fontWeight: 700, letterSpacing: "0.5px",
                      }}
                    >
                      AH
                    </span>
                  )}
                  <button
                    data-testid={`close-pos-${p.ticker}`}
                    onClick={() => closePos(p.ticker)}
                    style={{
                      marginLeft: "auto", padding: "4px 10px", borderRadius: "4px",
                      border: "1px solid rgba(255,51,51,0.3)", background: "transparent",
                      color: "#ff453a", fontSize: "11px", cursor: "pointer",
                    }}
                  >
                    Close
                  </button>
                </div>

                {/* FIX 2: Price reference row — Entry | Prev Close | Now */}
                <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", fontSize: "12px", marginBottom: "6px" }}>
                  <span style={{ color: "#4a5c70" }}>
                    Entry:{" "}
                    <span style={{ color: "#a1a1a6", fontFamily: "monospace" }}>
                      ${(entry ?? 0).toFixed(2)}
                    </span>
                  </span>
                  {prevClose > 0 && (
                    <>
                      <span style={{ color: "#2a3a4c" }}>|</span>
                      <span style={{ color: "#4a5c70" }}>
                        Prev Close:{" "}
                        <span style={{ color: "#a1a1a6", fontFamily: "monospace" }}>
                          ${(prevClose ?? 0).toFixed(2)}
                        </span>
                      </span>
                    </>
                  )}
                  <span style={{ color: "#2a3a4c" }}>|</span>
                  <span style={{ color: "#4a5c70" }}>
                    Now:{" "}
                    <span style={{ color: "#c8d6e5", fontFamily: "monospace" }}>
                      ${(current ?? 0).toFixed(2)}
                    </span>
                  </span>
                </div>

                {/* Row 2: Cost Basis | Market Value | P&L */}
                <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", fontSize: "12px", marginBottom: "4px" }}>
                  <span style={{ color: "#4a5c70" }}>
                    Cost:{" "}
                    <span style={{ color: "#a1a1a6", fontFamily: "monospace" }}>
                      ${costBasis.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                  </span>
                  <span style={{ color: "#4a5c70" }}>
                    Value:{" "}
                    <span style={{ color: "#c8d6e5", fontFamily: "monospace" }}>
                      ${marketValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                  </span>
                  <span style={{ color: "#4a5c70" }}>
                    P&L:{" "}
                    <span style={{ color: pnlColor, fontFamily: "monospace", fontWeight: 600 }}>
                      {pnl >= 0 ? "+" : ""}${(pnl ?? 0).toFixed(2)} ({pnlPct >= 0 ? "+" : ""}{(pnlPct ?? 0).toFixed(2)}%)
                    </span>
                  </span>
                </div>

                {/* Row 3: Today change (toggle) */}
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px" }}>
                  <span style={{ color: "#4a5c70" }}>Today:</span>
                  <button
                    data-testid={`today-toggle-${p.ticker}`}
                    onClick={() => toggleDollarView(p.ticker)}
                    title="Tap to toggle between % and $ view"
                    style={{
                      background: "none", border: "none", cursor: "pointer", padding: 0,
                      color: todayColor, fontFamily: "monospace", fontWeight: 600, fontSize: "12px",
                      textDecoration: "underline dotted",
                    }}
                  >
                    {showDollar
                      ? `${changeTodayDollar >= 0 ? "+" : ""}$${(changeTodayDollar ?? 0).toFixed(2)}`
                      : `${changeTodayPct >= 0 ? "+" : ""}${(changeTodayPct ?? 0).toFixed(2)}%`}
                  </button>
                  {prevClose > 0 && changeTodayFromClose !== null && (
                    <span style={{ color: "#4a5c70", fontFamily: "monospace", fontSize: "11px" }}>
                      [{changeTodayFromClose >= 0 ? "+" : ""}
                      ${changeTodayFromClose.toFixed(2)}/sh from close]
                    </span>
                  )}
                  <span style={{ color: "#2a3a4c", fontSize: "10px" }}>(tap to toggle)</span>
                  {showAH && changeTodayPct !== 0 && (
                    <span style={{ color: "#d4a017", fontSize: "10px", marginLeft: "4px" }}>
                      \u00b7 AH price reflects extended hours
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p style={{ color: "#4a5c70", fontSize: "13px", textAlign: "center", padding: "20px 0" }}>
          No open positions. The bot will generate signals and trade when activated.
        </p>
      )}
    </div>
  );
}

// ─── Trade History Panel ──────────────────────────────────────────────────────
function TradeHistoryPanel() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["/api/trades/history"],
    queryFn: async () => {
      const r = await apiRequest("GET", "/api/trades/history");
      return r.json();
    },
    refetchInterval: 60000,
    staleTime: 30000,
  });

  const trades: any[] = (data as any)?.trades ?? [];

  return (
    <div style={{ ...card, marginBottom: "20px" }} data-testid="trade-history-panel">
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
        <History size={14} style={{ color: "#00e5ff" }} />
        <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em" }}>
          TRADE HISTORY
        </span>
        <span style={{ marginLeft: "auto", fontSize: "12px", color: "#4a5c70" }}>
          {isLoading ? "loading…" : `${trades.length} closed trades`}
        </span>
      </div>

      {isError && (
        <div style={{ color: "#ff453a", fontSize: "12px", textAlign: "center", padding: "12px 0" }}>
          Failed to load trade history.
        </div>
      )}

      {!isLoading && !isError && trades.length === 0 && (
        <div style={{ color: "#4a5c70", fontSize: "13px", textAlign: "center", padding: "20px 0" }}>
          No closed trades yet
        </div>
      )}

      {trades.length > 0 && (
        <div style={{ maxHeight: "300px", overflowY: "auto", WebkitOverflowScrolling: "touch" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", minWidth: "560px" }}>
            <thead style={{ position: "sticky", top: 0, background: "rgba(0, 10, 20, 0.95)", zIndex: 1 }}>
              <tr style={{ color: "#4a5c70", textAlign: "left", borderBottom: "1px solid rgba(0, 229, 255, 0.1)" }}>
                <th style={{ padding: "7px 10px", fontWeight: 500 }}>Symbol</th>
                <th style={{ padding: "7px 6px", fontWeight: 500 }}>Side</th>
                <th style={{ padding: "7px 6px", textAlign: "right", fontWeight: 500 }}>Shares</th>
                <th style={{ padding: "7px 6px", textAlign: "right", fontWeight: 500 }}>Entry</th>
                <th style={{ padding: "7px 6px", textAlign: "right", fontWeight: 500 }}>Exit</th>
                <th style={{ padding: "7px 6px", textAlign: "right", fontWeight: 500 }}>P&L $</th>
                <th style={{ padding: "7px 6px", textAlign: "right", fontWeight: 500 }}>P&L %</th>
                <th style={{ padding: "7px 6px", fontWeight: 500 }}>Date/Time</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any, i: number) => {
                const pnl = t.pnl ?? 0;
                const pnlPct = t.pnlPct ?? 0;
                const isPos = pnl >= 0;
                const pnlColor = isPos ? "#30d158" : "#ff453a";
                const sideColor = t.side === "BUY" ? "#30d158" : "#ff453a";
                return (
                  <tr
                    key={i}
                    data-testid="trade-history-row"
                    style={{
                      borderBottom: "1px solid rgba(0, 15, 30, 0.4)",
                      background: isPos ? "rgba(48,209,88,0.03)" : "rgba(255,68,68,0.03)",
                    }}
                  >
                    <td style={{ padding: "7px 10px", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#c8d6e5" }}>
                      {t.symbol}
                    </td>
                    <td style={{ padding: "7px 6px", color: sideColor, fontFamily: "monospace", fontSize: "11px", fontWeight: 600 }}>
                      {t.side}
                    </td>
                    <td style={{ padding: "7px 6px", textAlign: "right", fontFamily: "monospace", color: "#a1a1a6" }}>
                      {Number(t.shares).toFixed(2)}
                    </td>
                    <td style={{ padding: "7px 6px", textAlign: "right", fontFamily: "monospace", color: "#a1a1a6" }}>
                      ${Number(t.entryPrice).toFixed(2)}
                    </td>
                    <td style={{ padding: "7px 6px", textAlign: "right", fontFamily: "monospace", color: "#c8d6e5" }}>
                      ${Number(t.exitPrice).toFixed(2)}
                    </td>
                    <td style={{ padding: "7px 6px", textAlign: "right", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: pnlColor }}>
                      {isPos ? "+" : ""}{(pnl ?? 0).toFixed(2)}
                    </td>
                    <td style={{ padding: "7px 6px", textAlign: "right", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: pnlColor }}>
                      {isPos ? "+" : ""}{(pnlPct ?? 0).toFixed(2)}%
                    </td>
                    <td style={{ padding: "7px 6px", color: "#4a5c70", fontFamily: "monospace", fontSize: "11px", whiteSpace: "nowrap" }}>
                      {t.filledAt ? new Date(t.filledAt).toLocaleString() : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function BotDashboard() {
  // ── Queries ───────────────────────────────────────────────────────────────
  const { data: acct } = useQuery({
    queryKey: ["/api/bot/account"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/account"); return r.json(); },
    refetchInterval: 10000,
  });
  const { data: positions } = useQuery({
    queryKey: ["/api/bot/positions"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/positions"); return r.json(); },
    refetchInterval: 15000, // Refresh every 15s for live stop/TP updates
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
  const { data: signals } = useQuery({
    queryKey: ["/api/bot/signals"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/signals"); return r.json(); },
    refetchInterval: 30000,
  });
  const { data: perfData } = useQuery({
    queryKey: ["/api/bot/performance"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/performance"); return r.json(); },
    refetchInterval: 60000,
  });
  const { data: notifData, refetch: refetchNotifs } = useQuery({
    queryKey: ["/api/bot/notifications"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/notifications"); return r.json(); },
    refetchInterval: 15000,
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
  const markAllRead = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/notifications/read").then(r => r.json()),
    onSuccess: () => refetchNotifs(),
  });

  // ── Derived values ────────────────────────────────────────────────────────
  const dailyPnl = acct?.dailyPnL ?? 0;
  const dailyPnlPct = acct?.dailyPnLPct ?? 0;
  const isKilled = status?.killSwitch ?? false;
  const isActive = status?.active ?? false;
  const notifList = Array.isArray(notifData) ? notifData : [];
  const unreadCount = notifList.filter((n: any) => !n.read).length;

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "24px 16px" }}>

      {/* ── Header ── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: "20px", flexWrap: "wrap", gap: "12px",
      }}>
        <div>
          <h1 style={{ fontSize: "1.4rem", fontWeight: 700, color: "#c8d6e5", margin: 0 }}>AI Trading Engine</h1>
          <div style={{ display: "flex", gap: "8px", marginTop: "6px", alignItems: "center" }}>
            <span style={{
              padding: "3px 10px", borderRadius: "4px", fontSize: "11px", fontWeight: 600,
              background: isKilled ? "rgba(255,51,51,0.15)" : isActive ? "rgba(0,255,65,0.15)" : "rgba(212,160,23,0.15)",
              color: isKilled ? "#ff453a" : isActive ? "#30d158" : "#d4a017",
              border: `1px solid ${isKilled ? "rgba(255,51,51,0.3)" : isActive ? "rgba(0,255,65,0.3)" : "rgba(212,160,23,0.3)"}`,

            }}>
              {isKilled ? "KILLED" : isActive ? "ACTIVE" : "PAUSED"}
            </span>
            <Tip id="paperTrading">
              <span style={{ padding: "3px 10px", borderRadius: "4px", fontSize: "11px", fontWeight: 600, background: "rgba(0,229,255,0.15)", color: "#00e5ff", border: "1px solid rgba(0,229,255,0.3)" }}>
                PAPER TRADING
              </span>
            </Tip>
          </div>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {/* Notifications Bell */}
          <NotificationBell
            notifications={notifList}
            onMarkRead={() => markAllRead.mutate()}
          />

          {!isActive ? (
            <button onClick={() => startBot.mutate()} disabled={isKilled} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 16px", borderRadius: "4px", border: "none", background: isKilled ? "#0d1a28" : "#30d158", color: isKilled ? "#4a5c70" : "#0a1420", fontSize: "13px", fontWeight: 600, cursor: isKilled ? "not-allowed" : "pointer" }}>
              <Play size={14} /> Start
            </button>
          ) : (
            <button onClick={() => stopBot.mutate()} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 16px", borderRadius: "4px", border: "none", background: "#d4a017", color: "black", fontSize: "13px", fontWeight: 600, cursor: "pointer" }}>
              <Square size={14} /> Pause
            </button>
          )}
          <Tip id="killSwitch">
            <button onClick={() => killBot.mutate()} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 16px", borderRadius: "4px", border: isKilled ? "2px solid #ff453a" : "2px solid rgba(255,69,58,0.3)", background: isKilled ? "rgba(255,51,51,0.2)" : "transparent", color: "#ff453a", fontSize: "13px", fontWeight: 700, cursor: "pointer" }}>
              <ShieldAlert size={14} /> {isKilled ? "Unkill" : "Kill Switch"}
            </button>
          </Tip>
        </div>
      </div>

      {/* ── Calendar / Holiday Banner ── */}
      <CalendarBanner />

      {/* ── Account Overview ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "12px", marginBottom: "20px" }}>
        <div style={card}>
          <div style={label}><Tip id="portfolio">Portfolio Value</Tip></div>
          <div style={{ ...bigNum, color: "#c8d6e5", fontSize: "clamp(18px, 4vw, 28px)" }}>
            ${(acct?.portfolioValue ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>
        <div style={card}>
          <div style={label}><Tip id="cash">Cash Available</Tip></div>
          <div style={{ ...bigNum, color: "#a1a1a6", fontSize: "clamp(18px, 4vw, 28px)" }}>
            ${(acct?.cash ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>
        <div style={card}>
          <div style={label}><Tip id="dailyPnl">Today's P&L</Tip></div>
          <div style={{ ...bigNum, color: dailyPnl >= 0 ? "#30d158" : "#ff453a", fontSize: "clamp(18px, 4vw, 28px)" }}>
            {dailyPnl >= 0 ? "+" : ""}{dailyPnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            <span style={{ fontSize: "12px", marginLeft: "6px" }}>({dailyPnlPct >= 0 ? "+" : ""}{Number(dailyPnlPct ?? 0).toFixed(2)}%)</span>
          </div>
        </div>
        <div style={card}>
          <div style={label}><Tip id="buyingPower">Buying Power</Tip></div>
          <div style={{ ...bigNum, color: "#a1a1a6", fontSize: "clamp(16px, 3vw, 22px)" }}>
            ${(acct?.buyingPower ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
        </div>
      </div>

      {/* ── Performance Card ── */}
      <PerformanceCard perf={perfData} />

      {/* ── Open Positions (enhanced) ── */}
      <EnhancedPositions positions={positions} closePos={(t: string) => closePos.mutate(t)} />

      {/* ── Trade History Panel ── */}
      <TradeHistoryPanel />

      {/* ── AI Signals Panel ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <Zap size={14} style={{ color: "#d4a017" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>
            <Tip id="confidence">AI Trade Signals</Tip>
          </span>
          <span style={{ marginLeft: "auto", fontSize: "11px", color: "#4a5c70" }}>
            {Array.isArray(signals) ? signals.length : 0} active signals
          </span>
          <button
            style={{
              display: "flex", alignItems: "center", gap: "6px",
              padding: "6px 14px", borderRadius: "4px", border: "1px solid rgba(212,160,23,0.3)",
              background: "rgba(212,160,23,0.1)", color: "#d4a017", fontSize: "12px", fontWeight: 600,
            }}
          >
          </button>
        </div>

        {Array.isArray(signals) && signals.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {signals.map((sig: any, i: number) => (
              <div key={i} style={{
                background: "rgba(0, 20, 40, 0.4)", border: `1px solid rgba(0, 229, 255, 0.08)`,
                borderLeft: `3px solid ${signalColor(sig.type)}`,
                borderRadius: "4px", padding: "12px 14px",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
                  <span style={{ fontFamily: "monospace", fontWeight: 700, fontSize: "15px", color: "#c8d6e5" }}>{sig.ticker}</span>
                  <span style={{
                    padding: "2px 8px", borderRadius: "4px", fontSize: "11px", fontWeight: 700,
                    background: sig.type === "buy" ? "rgba(0,255,65,0.15)" : sig.type === "sell" ? "rgba(255,51,51,0.15)" : "rgba(212,160,23,0.15)",
                    color: signalColor(sig.type),
                  }}>{sig.action}</span>
                  <span style={{ marginLeft: "auto", fontSize: "11px", color: "#4a5c70" }}>
                    {new Date(sig.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p style={{ color: "#a1a1a6", fontSize: "12px", margin: "0 0 8px 0", lineHeight: 1.5 }}>{sig.reason}</p>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "11px", color: "#4a5c70", flexShrink: 0 }}>
                    <Tip id="confidence">Confidence</Tip>
                  </span>
                  <div style={{ flex: 1, height: "4px", background: "rgba(0, 229, 255, 0.1)", borderRadius: "2px", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", width: `${sig.confidence}%`, borderRadius: "2px",
                      background: sig.confidence >= 70 ? "#30d158" : sig.confidence >= 50 ? "#d4a017" : "#ff453a",
                      transition: "width 0.5s ease",
                    }} />
                  </div>
                  <span style={{ fontSize: "11px", fontWeight: 700, color: "#c8d6e5", flexShrink: 0 }}>{sig.confidence}%</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "32px 0" }}>
            <Zap size={28} style={{ color: "#2a3a4c", marginBottom: "12px" }} />
          </div>
        )}
      </div>

      {/* ── Live Trade Charts ── */}
      <div style={{ marginBottom: "20px" }}>
        <TradeCharts positions={positions || []} />
      </div>

      {/* ── Security Controls ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <Shield size={14} style={{ color: "#30d158" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>Security Controls</span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px", marginBottom: "16px" }}>
          {/* Kill Switch status */}
          <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "14px", border: `1px solid ${isKilled ? "rgba(255,51,51,0.3)" : "rgba(0, 229, 255, 0.08)"}` }}>
            <div style={{ ...label, marginBottom: "8px" }}><Tip id="killSwitch">Kill Switch</Tip></div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{
                width: "44px", height: "24px", borderRadius: "6px", position: "relative", cursor: "pointer",
                background: isKilled ? "#ff453a" : "rgba(0, 229, 255, 0.12)",
                transition: "background 0.2s",
              }} onClick={() => killBot.mutate()}>
                <div style={{
                  position: "absolute", top: "3px", left: isKilled ? "23px" : "3px",
                  width: "18px", height: "18px", borderRadius: "50%", background: "white",
                  transition: "left 0.2s", boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
                }} />
              </div>
              <span style={{ fontSize: "13px", fontWeight: 600, color: isKilled ? "#ff453a" : "#4a5c70" }}>
                {isKilled ? "ACTIVE" : "OFF"}
              </span>
            </div>
          </div>

          {/* Daily loss limit */}
          <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "14px", border: "1px solid rgba(0, 229, 255, 0.08)" }}>
            <div style={{ ...label, marginBottom: "4px" }}><Tip id="dailyLossLimit">Daily Loss Limit</Tip></div>
            <div style={{ fontSize: "22px", fontWeight: 700, fontFamily: "monospace", color: "#d4a017" }}>−3%</div>
            <div style={{ fontSize: "11px", color: "#4a5c70", marginTop: "4px" }}>Auto-halts trading if exceeded</div>
          </div>

          {/* Max position size */}
          <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "14px", border: "1px solid rgba(0, 229, 255, 0.08)" }}>
            <div style={{ ...label, marginBottom: "4px" }}><Tip id="positionSize">Max Position Size</Tip></div>
            <div style={{ fontSize: "22px", fontWeight: 700, fontFamily: "monospace", color: "#00e5ff" }}>5%</div>
            <div style={{ fontSize: "11px", color: "#4a5c70", marginTop: "4px" }}>Per trade, of portfolio</div>
          </div>

          {/* Max exposure */}
          <div style={{ background: "rgba(0, 15, 30, 0.4)", borderRadius: "4px", padding: "14px", border: "1px solid rgba(0, 229, 255, 0.08)" }}>
            <div style={{ ...label, marginBottom: "4px" }}><Tip id="totalExposure">Max Exposure</Tip></div>
            <div style={{ fontSize: "22px", fontWeight: 700, fontFamily: "monospace", color: "#a855f7" }}>50%</div>
            <div style={{ fontSize: "11px", color: "#4a5c70", marginTop: "4px" }}>Max portfolio invested</div>
          </div>
        </div>

        {/* Mode & security badges */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
          <span style={{ padding: "4px 12px", borderRadius: "4px", fontSize: "11px", fontWeight: 600, background: "rgba(0,229,255,0.15)", color: "#00e5ff", border: "1px solid rgba(0,229,255,0.2)" }}>
            Paper Trading Mode
          </span>
          <span style={{ padding: "4px 12px", borderRadius: "4px", fontSize: "11px", fontWeight: 600, background: "rgba(0,255,65,0.1)", color: "#30d158", border: "1px solid rgba(0,255,65,0.2)" }}>
            {status?.auditLogCount ?? 0} Audit Entries
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "5px", padding: "4px 12px", borderRadius: "4px", fontSize: "11px", fontWeight: 600, background: "rgba(0, 20, 40, 0.5)", color: "#c8d6e5", border: "1px solid rgba(0, 229, 255, 0.1)" }}>
            <Lock size={10} /> API keys encrypted
          </span>
          <span style={{ padding: "4px 12px", borderRadius: "4px", fontSize: "11px", fontWeight: 600, background: "rgba(168,85,247,0.1)", color: "#a855f7", border: "1px solid rgba(168,85,247,0.2)" }}>
            {unreadCount} unread alerts
          </span>
        </div>
      </div>

      {/* ── Audit Log ── */}
      <div style={{ ...card, marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
          <Clock size={14} style={{ color: "#00e5ff" }} />
          <span style={{ fontSize: "14px", fontWeight: 600, color: "#c8d6e5" }}>Activity Log</span>
          <span style={{ marginLeft: "auto", fontSize: "11px", color: "#4a5c70" }}>Every trade and decision is logged</span>
        </div>
        <div style={{ maxHeight: "250px", overflowY: "auto" }}>
          {Array.isArray(audit) && audit.length > 0 ? (
            audit.map((a: any, i: number) => (
              <div key={i} style={{ display: "flex", gap: "10px", padding: "6px 0", borderBottom: "1px solid rgba(0, 15, 30, 0.4)", fontSize: "12px", flexWrap: "wrap" }}>
                <span style={{ color: "#4a5c70", fontFamily: "monospace", flexShrink: 0, width: "130px" }}>{new Date(a.time).toLocaleString()}</span>
                <span style={{ color: "#00e5ff", fontWeight: 600, flexShrink: 0, width: "80px" }}>{a.action}</span>
                <span style={{ color: "#a1a1a6" }}>{a.detail}</span>
              </div>
            ))
          ) : (
            <p style={{ color: "#4a5c70", fontSize: "12px", textAlign: "center", padding: "16px 0" }}>No activity yet. Start the bot to begin.</p>
          )}
        </div>
      </div>

      {/* ── Performance Dashboard (detailed) ── */}
      <PerformanceDashboard perfData={perfData} />

    </div>
  );
}
