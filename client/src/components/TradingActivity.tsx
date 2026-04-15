import { useState, useEffect, useCallback } from "react";
import { RefreshCw } from "lucide-react";
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

// ─── Styles ──────────────────────────────────────────────────────────────────

const sectionCard: React.CSSProperties = {
  background: "rgba(0, 20, 40, 0.5)",
  border: "1px solid rgba(0, 229, 255, 0.1)",
  borderRadius: "6px",
  overflow: "hidden",
  marginBottom: "16px",
  backdropFilter: "blur(20px)",
};

const cardHeader: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "14px 18px",
  borderBottom: "1px solid rgba(0, 229, 255, 0.08)",
};

const cardTitle: React.CSSProperties = {
  fontSize: "14px",
  fontWeight: 600,
  color: "#c8d6e5",
};

const countBadge: React.CSSProperties = {
  fontSize: "11px",
  fontWeight: 600,
  padding: "2px 10px",
  borderRadius: "9999px",
  background: "rgba(59, 130, 246, 0.15)",
  color: "#3b82f6",
  fontVariantNumeric: "tabular-nums",
};

const tableWrap: React.CSSProperties = {
  overflowX: "auto",
  WebkitOverflowScrolling: "touch",
};

const table: React.CSSProperties = {
  width: "100%",
  minWidth: "500px",
  fontVariantNumeric: "tabular-nums",
  borderCollapse: "collapse",
};

const th: React.CSSProperties = {
  textAlign: "left",
  fontSize: "11px",
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: "#4a5c70",
  padding: "10px 16px",
  whiteSpace: "nowrap",
  background: "rgba(0, 15, 30, 0.4)",
  fontFamily: "'JetBrains Mono', monospace",
};

const td: React.CSSProperties = {
  fontSize: "13px",
  padding: "10px 16px",
  borderTop: "1px solid rgba(0, 229, 255, 0.06)",
  whiteSpace: "nowrap",
  color: "#c8d6e5",
};

const emptyCell: React.CSSProperties = {
  textAlign: "center",
  color: "#4a5c70",
  padding: "28px 16px",
  fontStyle: "italic",
  fontSize: "13px",
};

// Badge styles for trade types
const badgeColors: Record<string, { bg: string; color: string }> = {
  "type-long": { bg: "rgba(34, 197, 94, 0.15)", color: "#22c55e" },
  "type-short": { bg: "rgba(239, 68, 68, 0.15)", color: "#ef4444" },
  "type-short-etf": { bg: "rgba(239, 68, 68, 0.15)", color: "#f87171" },
  "type-etf": { bg: "rgba(59, 130, 246, 0.15)", color: "#3b82f6" },
  "type-put": { bg: "rgba(249, 115, 22, 0.15)", color: "#f97316" },
  "type-call": { bg: "rgba(168, 85, 247, 0.15)", color: "#a855f7" },
  "type-iron-condor": { bg: "rgba(236, 72, 153, 0.15)", color: "#ec4899" },
  "type-option": { bg: "rgba(234, 179, 8, 0.15)", color: "#eab308" },
};

function Badge({ type }: { type: TradeType }) {
  const colors = badgeColors[type.cssClass] || badgeColors["type-option"];
  return (
    <span style={{
      display: "inline-block",
      fontSize: "10px",
      fontWeight: 600,
      padding: "2px 8px",
      borderRadius: "9999px",
      textTransform: "uppercase",
      letterSpacing: "0.03em",
      whiteSpace: "nowrap",
      background: colors.bg,
      color: colors.color,
    }}>
      {type.label}
    </span>
  );
}

function pnlStyle(val: number): React.CSSProperties {
  if (isNaN(val) || val === 0) return { color: "#4a5c70" };
  return { color: val > 0 ? "#22c55e" : "#ef4444", fontWeight: 600 };
}

function sideStyle(side: string): React.CSSProperties {
  const s = side.toLowerCase();
  if (s === "buy" || s === "long") return { color: "#22c55e", fontWeight: 600 };
  if (s === "sell" || s === "short") return { color: "#ef4444", fontWeight: 600 };
  return {};
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function TradingActivity() {
  const [trades, setTrades] = useState<any[]>([]);
  const [orders, setOrders] = useState<any[]>([]);
  const [positions, setPositions] = useState<any[]>([]);
  const [lastUpdate, setLastUpdate] = useState<string>("");
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    setRefreshing(true);
    try {
      const [tradesRes, ordersRes, positionsRes] = await Promise.all([
        fetch("/api/trades/today").then(r => r.json()).catch(() => ({ trades: [] })),
        fetch("/api/orders/open").then(r => r.json()).catch(() => ({ orders: [] })),
        fetch("/api/positions").then(r => r.json()).catch(() => ({ positions: [] })),
      ]);
      setTrades(tradesRes.trades || []);
      setOrders(ordersRes.orders || []);
      setPositions(positionsRes.positions || []);
      setLastUpdate("Updated " + new Date().toLocaleTimeString());
    } catch (err) {
      console.error("[trading] Refresh failed:", err);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return (
    <div style={{ marginTop: "20px" }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: "16px", flexWrap: "wrap", gap: "8px",
      }}>
        <div>
          <div style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "#00e5ff", marginBottom: "4px" }}>
            Live Paper Trading
          </div>
          <h2 style={{ fontSize: "18px", fontWeight: 700, color: "#c8d6e5", margin: 0 }}>
            Trading Activity
          </h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "11px", color: "#4a5c70", fontVariantNumeric: "tabular-nums" }}>
            {lastUpdate}
          </span>
          <button
            onClick={fetchData}
            disabled={refreshing}
            style={{
              display: "inline-flex", alignItems: "center", gap: "6px",
              padding: "6px 14px", borderRadius: "4px", fontSize: "11px", fontWeight: 600,
              background: "rgba(0, 229, 255, 0.08)", border: "1px solid rgba(0, 229, 255, 0.2)",
              color: "#00e5ff", cursor: refreshing ? "not-allowed" : "pointer",
              opacity: refreshing ? 0.5 : 1,
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: "0.04em", textTransform: "uppercase",
            }}
          >
            <RefreshCw size={12} style={refreshing ? { animation: "spin 1s linear infinite" } : {}} />
            Refresh
          </button>
        </div>
      </div>

      {/* Today's Trades */}
      <div style={sectionCard}>
        <div style={cardHeader}>
          <span style={cardTitle}>Today's Trades</span>
          <span style={countBadge}>{trades.length}</span>
        </div>
        <div style={tableWrap}>
          <table style={table}>
            <thead>
              <tr>
                <th style={th}>Time</th>
                <th style={th}>Symbol</th>
                <th style={th}>Side</th>
                <th style={th}>Qty</th>
                <th style={th}>Fill Price</th>
                <th style={th}>Type</th>
              </tr>
            </thead>
            <tbody>
              {trades.length === 0 ? (
                <tr><td colSpan={6} style={emptyCell}>No trades today</td></tr>
              ) : trades.map((t: any, i: number) => {
                const sym = t.symbol || "";
                const rawSide = (t.side || "").toLowerCase();
                const displaySide = INVERSE_ETFS.has(sym) && rawSide === "buy"
                  ? "Short"
                  : rawSide.charAt(0).toUpperCase() + rawSide.slice(1);
                const qty = t.filled_qty || t.qty || "0";
                const price = t.filled_avg_price || "0";
                const tradeType = getTradeType(t);

                return (
                  <tr key={i}>
                    <td style={td}>{formatTime(t.filled_at || t.updated_at)}</td>
                    <td style={{ ...td, fontWeight: 600 }}>{sym}</td>
                    <td style={{ ...td, ...sideStyle(displaySide) }}>{displaySide}</td>
                    <td style={td}>{qty}</td>
                    <td style={td}>{formatCurrency(price)}</td>
                    <td style={td}><Badge type={tradeType} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Open Orders */}
      <div style={sectionCard}>
        <div style={cardHeader}>
          <span style={cardTitle}>Open Orders</span>
          <span style={countBadge}>{orders.length}</span>
        </div>
        <div style={tableWrap}>
          <table style={table}>
            <thead>
              <tr>
                <th style={th}>Time</th>
                <th style={th}>Symbol</th>
                <th style={th}>Side</th>
                <th style={th}>Qty</th>
                <th style={th}>Type</th>
                <th style={th}>Limit Price</th>
                <th style={th}>Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.length === 0 ? (
                <tr><td colSpan={7} style={emptyCell}>No open orders</td></tr>
              ) : orders.map((o: any, i: number) => {
                const sym = o.symbol || "";
                const side = (o.side || "").charAt(0).toUpperCase() + (o.side || "").slice(1);
                const qty = o.qty || "0";
                const orderType = (o.type || "market").replace("_", " ");
                const limitPrice = o.limit_price ? formatCurrency(o.limit_price) : "—";
                const status = o.status || "unknown";

                return (
                  <tr key={i}>
                    <td style={td}>{formatTime(o.submitted_at || o.created_at)}</td>
                    <td style={{ ...td, fontWeight: 600 }}>{sym}</td>
                    <td style={{ ...td, ...sideStyle(side) }}>{side}</td>
                    <td style={td}>{qty}</td>
                    <td style={{ ...td, textTransform: "capitalize" }}>{orderType}</td>
                    <td style={td}>{limitPrice}</td>
                    <td style={td}>
                      <span style={{
                        display: "inline-block", fontSize: "11px", fontWeight: 500,
                        padding: "2px 8px", borderRadius: "9999px",
                        background: "rgba(59, 130, 246, 0.15)", color: "#3b82f6",
                        textTransform: "capitalize",
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
      </div>

      {/* Open Positions */}
      <div style={sectionCard}>
        <div style={cardHeader}>
          <span style={cardTitle}>Open Positions</span>
          <span style={countBadge}>{positions.length}</span>
        </div>
        <div style={tableWrap}>
          <table style={table}>
            <thead>
              <tr>
                <th style={th}>Symbol</th>
                <th style={th}>Side</th>
                <th style={th}>Type</th>
                <th style={th}>Qty</th>
                <th style={th}>Avg Entry</th>
                <th style={th}>Current</th>
                <th style={th}>Mkt Value</th>
                <th style={th}>P/L</th>
                <th style={th}>P/L %</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr><td colSpan={9} style={emptyCell}>No open positions</td></tr>
              ) : positions.map((p: any, i: number) => {
                const sym = p.symbol || "";
                const rawSide = (p.side || "long").toLowerCase();
                const displaySide = getDisplaySide(sym, rawSide);
                const sideLabel = displaySide.charAt(0).toUpperCase() + displaySide.slice(1);
                const qty = Math.abs(parseFloat(p.qty || "0"));
                const avgEntry = parseFloat(p.avg_entry_price || "0");
                const current = parseFloat(p.current_price || "0");
                const marketValue = parseFloat(p.market_value || "0");
                const unrealizedPl = parseFloat(p.unrealized_pl || "0");
                const unrealizedPlPct = parseFloat(p.unrealized_plpc || "0") * 100;
                const posType = getPositionType(p);

                return (
                  <tr key={i}>
                    <td style={{ ...td, fontWeight: 600 }}>{sym}</td>
                    <td style={{ ...td, ...sideStyle(sideLabel) }}>{sideLabel}</td>
                    <td style={td}><Badge type={posType} /></td>
                    <td style={td}>{qty}</td>
                    <td style={td}>{formatCurrency(avgEntry)}</td>
                    <td style={td}>{formatCurrency(current)}</td>
                    <td style={td}>{formatCurrency(marketValue)}</td>
                    <td style={{ ...td, ...pnlStyle(unrealizedPl) }}>{formatCurrency(unrealizedPl)}</td>
                    <td style={{ ...td, ...pnlStyle(unrealizedPlPct) }}>
                      {(unrealizedPlPct >= 0 ? "+" : "") + unrealizedPlPct.toFixed(2) + "%"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Spin animation for refresh button */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
