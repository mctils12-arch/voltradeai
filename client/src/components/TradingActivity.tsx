import { useState, useEffect, useCallback } from "react";
import { ArrowUpDown, BookOpen } from "lucide-react";
import { INVERSE_ETFS, getDisplaySide } from "../../../shared/inverseEtfs";

// ─── ETF sets ────────────────────────────────────────────────────────────────

const COMMON_ETFS = new Set([
  "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLI","XLU","XLC","XLY","XLP","XLRE","XLB",
  "VTI","VOO","VEA","VWO","BND","AGG","GLD","SLV","USO","UNG","TLT","HYG","LQD","EEM","EFA",
  "ARKK","ARKW","ARKF","ARKG","ARKQ","TQQQ","SOXL","TECL","UPRO","SPXL","UDOW","URTY",
  "SMH","XBI","IBB","KWEB","FXI","MCHI","INDA","EWZ","EWJ","VGK",
]);

// ─── Trade type classification ───────────────────────────────────────────────

interface TradeType {
  label: string;
  cssClass: string;
}

function getTradeType(order: any): TradeType {
  const sym = (order.symbol || "").toUpperCase();
  const side = (order.side || "").toLowerCase();
  const orderClass = (order.order_class || "").toLowerCase();
  const assetClass = (order.asset_class || "").toLowerCase();
  const legs = order.legs;

  if (legs && Array.isArray(legs) && legs.length >= 2) {
    return { label: "Iron Condor", cssClass: "type-iron-condor" };
  }
  if (orderClass === "oco" || orderClass === "bracket" || orderClass === "oto") {
    return { label: "Iron Condor", cssClass: "type-iron-condor" };
  }

  if (assetClass === "us_option" || sym.length > 8) {
    const match = sym.match(/[CP]\d{8}$/);
    if (match) {
      const optType = match[0][0];
      if (optType === "P") return { label: "Put", cssClass: "type-put" };
      if (optType === "C") return { label: "Call", cssClass: "type-call" };
    }
    return { label: "Option", cssClass: "type-option" };
  }

  if (INVERSE_ETFS.has(sym)) {
    return { label: "Short ETF", cssClass: "type-short-etf" };
  }
  if (COMMON_ETFS.has(sym)) {
    return { label: "ETF", cssClass: "type-etf" };
  }

  if (side === "sell" || side === "sell_short" || side === "short") {
    return { label: "Short", cssClass: "type-short" };
  }

  return { label: "Long", cssClass: "type-long" };
}

function getPositionType(position: any): TradeType {
  const sym = (position.symbol || "").toUpperCase();
  const rawSide = (position.side || "long").toLowerCase();
  const assetClass = (position.asset_class || "").toLowerCase();

  if (assetClass === "us_option" || sym.length > 8) {
    const match = sym.match(/[CP]\d{8}$/);
    if (match) {
      const optType = match[0][0];
      if (optType === "P") return { label: "Put", cssClass: "type-put" };
      if (optType === "C") return { label: "Call", cssClass: "type-call" };
    }
    return { label: "Option", cssClass: "type-option" };
  }

  if (INVERSE_ETFS.has(sym)) {
    return { label: "Short ETF", cssClass: "type-short-etf" };
  }
  if (COMMON_ETFS.has(sym)) {
    return { label: "ETF", cssClass: "type-etf" };
  }
  if (rawSide === "short") {
    return { label: "Short", cssClass: "type-short" };
  }

  return { label: "Long", cssClass: "type-long" };
}

// ─── Formatting helpers ──────────────────────────────────────────────────────

function formatCurrency(val: string | number) {
  const n = parseFloat(String(val));
  if (isNaN(n)) return "—";
  return n < 0
    ? "-$" + Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : "$" + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatTime(isoStr: string | undefined) {
  if (!isoStr) return "—";
  const d = new Date(isoStr);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// ─── Shared styles (matching bot.tsx card pattern exactly) ────────────────────

const card: React.CSSProperties = {
  background: "rgba(0, 20, 40, 0.5)",
  border: "1px solid rgba(0, 229, 255, 0.1)",
  borderRadius: "6px",
  padding: "20px",
  backdropFilter: "blur(20px)",
  marginBottom: "20px",
};

const headerRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "8px",
  marginBottom: "12px",
};

const titleStyle: React.CSSProperties = {
  fontSize: "14px",
  fontWeight: 600,
  color: "#c8d6e5",
  fontFamily: "'JetBrains Mono', monospace",
  letterSpacing: "0.05em",
};

const countStyle: React.CSSProperties = {
  marginLeft: "auto",
  fontSize: "12px",
  color: "#4a5c70",
};

const stickyHead: React.CSSProperties = {
  position: "sticky",
  top: 0,
  background: "rgba(0, 10, 20, 0.95)",
  zIndex: 1,
};

const thStyle: React.CSSProperties = {
  padding: "7px 10px",
  fontWeight: 500,
  color: "#4a5c70",
  textAlign: "left",
  fontSize: "12px",
  whiteSpace: "nowrap",
};

const thRight: React.CSSProperties = { ...thStyle, textAlign: "right" };

const tdStyle: React.CSSProperties = {
  padding: "7px 10px",
  fontFamily: "monospace",
  color: "#a1a1a6",
  fontSize: "12px",
  whiteSpace: "nowrap",
};

const tdSymbol: React.CSSProperties = {
  padding: "7px 10px",
  fontFamily: "'JetBrains Mono', monospace",
  fontWeight: 700,
  color: "#c8d6e5",
  fontSize: "12px",
};

const tdRight: React.CSSProperties = { ...tdStyle, textAlign: "right" };

const rowBorder: React.CSSProperties = {
  borderBottom: "1px solid rgba(0, 15, 30, 0.4)",
};

const emptyRow: React.CSSProperties = {
  color: "#4a5c70",
  fontSize: "13px",
  textAlign: "center",
  padding: "20px 0",
};

const tableStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "12px",
  minWidth: "560px",
};

// Badge styles — matches existing bot.tsx badge pattern (rectangular, bordered, 3px radius)
const badgeColors: Record<string, { bg: string; color: string; border: string }> = {
  "type-long":        { bg: "rgba(48,209,88,0.12)",   color: "#30d158", border: "rgba(48,209,88,0.25)" },
  "type-short":       { bg: "rgba(255,69,58,0.12)",   color: "#ff453a", border: "rgba(255,69,58,0.25)" },
  "type-short-etf":   { bg: "rgba(255,69,58,0.12)",   color: "#ff453a", border: "rgba(255,69,58,0.25)" },
  "type-etf":         { bg: "rgba(0,229,255,0.12)",   color: "#00e5ff", border: "rgba(0,229,255,0.25)" },
  "type-put":         { bg: "rgba(255,69,58,0.12)",   color: "#ff453a", border: "rgba(255,69,58,0.3)" },
  "type-call":        { bg: "rgba(48,209,88,0.12)",   color: "#30d158", border: "rgba(48,209,88,0.3)" },
  "type-iron-condor": { bg: "rgba(191,90,242,0.12)",  color: "#bf5af2", border: "rgba(191,90,242,0.3)" },
  "type-option":      { bg: "rgba(212,160,23,0.12)",  color: "#d4a017", border: "rgba(212,160,23,0.25)" },
};

function Badge({ type }: { type: TradeType }) {
  const colors = badgeColors[type.cssClass] || badgeColors["type-option"];
  return (
    <span style={{
      display: "inline-block",
      fontSize: "10px",
      fontWeight: 700,
      padding: "2px 6px",
      borderRadius: "3px",
      textTransform: "uppercase" as const,
      letterSpacing: "0.4px",
      whiteSpace: "nowrap",
      background: colors.bg,
      color: colors.color,
      border: `1px solid ${colors.border}`,
    }}>
      {type.label}
    </span>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function TradingActivity() {
  const [trades, setTrades] = useState<any[]>([]);
  const [orders, setOrders] = useState<any[]>([]);

  const fetchData = useCallback(async () => {
    try {
      const [tradesRes, ordersRes] = await Promise.all([
        fetch("/api/trades/today").then(r => r.json()).catch(() => ({ trades: [] })),
        fetch("/api/orders/open").then(r => r.json()).catch(() => ({ orders: [] })),
      ]);
      setTrades(tradesRes.trades || []);
      setOrders(ordersRes.orders || []);
    } catch (err) {
      console.error("[trading] Refresh failed:", err);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const pnlColor = (val: number) => {
    if (isNaN(val) || val === 0) return "#4a5c70";
    return val > 0 ? "#30d158" : "#ff453a";
  };

  const sideColor = (side: string) => {
    const s = side.toLowerCase();
    if (s === "buy" || s === "long") return "#30d158";
    if (s === "sell" || s === "short") return "#ff453a";
    return "#a1a1a6";
  };

  return (
    <>
      {/* ── Today's Trades ── */}
      <div style={card} data-testid="todays-trades-panel">
        <div style={headerRow}>
          <ArrowUpDown size={14} style={{ color: "#00e5ff" }} />
          <span style={titleStyle}>TODAY'S TRADES</span>
          <span style={countStyle}>{trades.length} fills</span>
        </div>

        {trades.length === 0 ? (
          <div style={emptyRow}>No trades today</div>
        ) : (
          <div style={{ maxHeight: "300px", overflowY: "auto", overflowX: "auto", WebkitOverflowScrolling: "touch" as any }}>
            <table style={tableStyle}>
              <thead style={stickyHead}>
                <tr style={{ color: "#4a5c70", textAlign: "left", borderBottom: "1px solid rgba(0, 229, 255, 0.1)" }}>
                  <th style={thStyle}>Time</th>
                  <th style={thStyle}>Symbol</th>
                  <th style={thStyle}>Side</th>
                  <th style={thStyle}>Type</th>
                  <th style={thRight}>Qty</th>
                  <th style={thRight}>Fill Price</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t: any, i: number) => {
                  const sym = t.symbol || "";
                  const rawSide = (t.side || "").toLowerCase();
                  const displaySide = INVERSE_ETFS.has(sym) && rawSide === "buy"
                    ? "SHORT"
                    : rawSide.toUpperCase();
                  const qty = t.filled_qty || t.qty || "0";
                  const price = t.filled_avg_price || "0";
                  const tradeType = getTradeType(t);

                  return (
                    <tr key={i} style={rowBorder}>
                      <td style={tdStyle}>{formatTime(t.filled_at || t.updated_at)}</td>
                      <td style={tdSymbol}>{sym}</td>
                      <td style={{ ...tdStyle, color: sideColor(displaySide), fontWeight: 600, fontSize: "11px" }}>{displaySide}</td>
                      <td style={tdStyle}><Badge type={tradeType} /></td>
                      <td style={tdRight}>{qty}</td>
                      <td style={tdRight}>{formatCurrency(price)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Open Orders ── */}
      <div style={card} data-testid="open-orders-panel">
        <div style={headerRow}>
          <BookOpen size={14} style={{ color: "#00e5ff" }} />
          <span style={titleStyle}>OPEN ORDERS</span>
          <span style={countStyle}>{orders.length} pending</span>
        </div>

        {orders.length === 0 ? (
          <div style={emptyRow}>No open orders</div>
        ) : (
          <div style={{ maxHeight: "250px", overflowY: "auto", overflowX: "auto", WebkitOverflowScrolling: "touch" as any }}>
            <table style={tableStyle}>
              <thead style={stickyHead}>
                <tr style={{ color: "#4a5c70", textAlign: "left", borderBottom: "1px solid rgba(0, 229, 255, 0.1)" }}>
                  <th style={thStyle}>Time</th>
                  <th style={thStyle}>Symbol</th>
                  <th style={thStyle}>Side</th>
                  <th style={thRight}>Qty</th>
                  <th style={thStyle}>Order Type</th>
                  <th style={thRight}>Limit Price</th>
                  <th style={thStyle}>Status</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((o: any, i: number) => {
                  const sym = o.symbol || "";
                  const side = (o.side || "").toUpperCase();
                  const qty = o.qty || "0";
                  const orderType = (o.type || "market").replace("_", " ");
                  const limitPrice = o.limit_price ? formatCurrency(o.limit_price) : "—";
                  const status = o.status || "unknown";

                  return (
                    <tr key={i} style={rowBorder}>
                      <td style={tdStyle}>{formatTime(o.submitted_at || o.created_at)}</td>
                      <td style={tdSymbol}>{sym}</td>
                      <td style={{ ...tdStyle, color: sideColor(side), fontWeight: 600, fontSize: "11px" }}>{side}</td>
                      <td style={tdRight}>{qty}</td>
                      <td style={{ ...tdStyle, textTransform: "capitalize" }}>{orderType}</td>
                      <td style={tdRight}>{limitPrice}</td>
                      <td style={tdStyle}>
                        <span style={{
                          display: "inline-block", fontSize: "10px", fontWeight: 700,
                          padding: "2px 6px", borderRadius: "3px",
                          background: "rgba(0, 229, 255, 0.12)", color: "#00e5ff",
                          border: "1px solid rgba(0, 229, 255, 0.25)",
                          textTransform: "capitalize" as const, letterSpacing: "0.3px",
                        }}>
                          {status.replace("_", " ")}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </>
  );
}
