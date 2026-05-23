import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  Search, PieChart, DollarSign, AlertTriangle, TrendingUp, TrendingDown,
  Building2, Calendar, Layers, Info, Calculator, Percent,
  Briefcase, Globe, Hash, Activity, Sparkles, ChevronDown, ChevronUp,
} from "lucide-react";

// ────────────────────────────────────────────────────────────────────────────
// Types — mirror etf_analyzer.py output
// ────────────────────────────────────────────────────────────────────────────

type Period = "1mo" | "3mo" | "6mo" | "ytd" | "1y" | "5y";
const PERIODS: { key: Period; label: string }[] = [
  { key: "1mo", label: "1M" },
  { key: "3mo", label: "3M" },
  { key: "6mo", label: "6M" },
  { key: "ytd", label: "YTD" },
  { key: "1y", label: "1Y" },
  { key: "5y", label: "5Y" },
];

interface Holding {
  symbol: string;
  weight: number;
  name?: string | null;
  price?: number | null;
  currency?: string | null;
  exchange?: string | null;
  country?: string | null;
  sector?: string | null;
  industry?: string | null;
  market_cap?: number | null;
  shares_outstanding?: number | null;
  first_trade_date?: string | null;
  listing_age_years?: number | null;
  is_spac_heuristic?: boolean;
  trailing_pe?: number | null;
  forward_pe?: number | null;
  dividend_yield?: number | null;
  beta?: number | null;
  "52w_high"?: number | null;
  "52w_low"?: number | null;
  "52w_pos_pct"?: number | null;
  employees?: number | null;
  headquarters?: string | null;
  website?: string | null;
  long_business_summary?: string | null;
  performance?: Partial<Record<Period, number | null>>;
  contribution?: Partial<Record<Period, number | null>>;
  is_drag?: Partial<Record<Period, boolean | null>>;
  fetch_error?: string | null;
}

interface ETFInfo {
  ticker: string;
  long_name?: string | null;
  family?: string | null;
  category?: string | null;
  legal_type?: string | null;
  exchange?: string | null;
  currency?: string | null;
  total_assets?: number | null;
  nav_price?: number | null;
  current_price?: number | null;
  expense_ratio?: number | null;
  yield?: number | null;
  ytd_return?: number | null;
  three_year_avg_return?: number | null;
  five_year_avg_return?: number | null;
  inception_date?: string | null;
  management_style: string;
  tracks_index: string;
  rebalance_schedule: string;
  long_business_summary?: string | null;
}

interface ETFResult {
  ticker?: string;
  info?: ETFInfo;
  sector_weights?: Record<string, number>;
  holdings?: Holding[];
  holdings_coverage?: number;
  holdings_returned?: number;
  holdings_truncated?: boolean;
  etf_performance?: Partial<Record<Period, number | null>>;
  as_of?: string;
  data_source_notes?: string;
  from_cache?: boolean;
  error?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ────────────────────────────────────────────────────────────────────────────

const fmtPct = (v?: number | null, digits = 2): string =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;

const fmtBp = (v?: number | null): string =>
  v == null ? "—" : `${(v * 10000).toFixed(0)} bps`;

const fmtMoney = (v?: number | null, digits = 2): string =>
  v == null ? "—" : `$${v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;

const fmtBigMoney = (v?: number | null): string => {
  if (v == null) return "—";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toLocaleString()}`;
};

const fmtNum = (v?: number | null, digits = 2): string =>
  v == null ? "—" : v.toFixed(digits);

const perfColor = (v?: number | null): string => {
  if (v == null) return "#94a3b8";
  if (v > 0) return "#4ade80";
  if (v < 0) return "#ff5a6e";
  return "#94a3b8";
};

// ────────────────────────────────────────────────────────────────────────────
// Reusable section card
// ────────────────────────────────────────────────────────────────────────────

function Section({
  icon, title, subtitle, children, accent = "#4d9fff",
}: {
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  accent?: string;
}) {
  return (
    <div style={{
      marginTop: "1.5rem",
      padding: "1.25rem",
      background: "rgba(15, 23, 42, 0.6)",
      border: "1px solid rgba(148, 163, 184, 0.15)",
      borderRadius: "0.5rem",
      backdropFilter: "blur(8px)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <div style={{ color: accent, display: "flex" }}>{icon}</div>
        <div>
          <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "#e2e8f0" }}>{title}</h3>
          {subtitle && <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{subtitle}</div>}
        </div>
      </div>
      {children}
    </div>
  );
}

function StatTile({ label, value, sub, color }: {
  label: string; value: React.ReactNode; sub?: React.ReactNode; color?: string;
}) {
  return (
    <div style={{
      padding: "0.75rem 1rem",
      background: "rgba(10, 22, 40, 0.5)",
      border: "1px solid rgba(148, 163, 184, 0.1)",
      borderRadius: "0.375rem",
    }}>
      <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{
        fontSize: 16, fontWeight: 600, color: color || "#eef3fb",
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Investment calculator (slider + numeric input)
// ────────────────────────────────────────────────────────────────────────────

function InvestmentCalculator({
  holdings, etfPrice, etfTicker,
}: {
  holdings: Holding[];
  etfPrice?: number | null;
  etfTicker: string;
}) {
  const [amount, setAmount] = useState<number>(10000);
  const [allowFractional, setAllowFractional] = useState<boolean>(true);

  const etfShares = etfPrice && etfPrice > 0
    ? (allowFractional ? amount / etfPrice : Math.floor(amount / etfPrice))
    : 0;
  const etfActualCost = etfPrice ? etfShares * etfPrice : 0;

  const rows = holdings
    .filter(h => h.weight && h.price && h.price > 0)
    .map(h => {
      const dollars = (h.weight as number) * amount;
      const shares = allowFractional ? dollars / (h.price as number) : Math.floor(dollars / (h.price as number));
      const actualCost = shares * (h.price as number);
      return { ...h, dollars, shares, actualCost };
    });

  const replicatedTotal = rows.reduce((s, r) => s + r.actualCost, 0);
  const uninvested = amount - replicatedTotal;
  const coverageWeight = holdings.reduce((s, h) => s + (h.weight || 0), 0);

  return (
    <Section
      icon={<Calculator size={16} />}
      title="Replication Calculator"
      subtitle={`Buy individual holdings instead of ${etfTicker}. See exactly how many shares of each.`}
      accent="#4d9fff"
    >
      {/* Amount controls */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "1rem", alignItems: "center", marginBottom: "1rem" }}>
        <div>
          <input
            type="range"
            min={100}
            max={1000000}
            step={100}
            value={amount}
            onChange={(e) => setAmount(parseInt(e.target.value))}
            style={{ width: "100%", accentColor: "#4d9fff" }}
            aria-label="Investment amount slider"
          />
          <div style={{
            display: "flex", justifyContent: "space-between", marginTop: 4,
            fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace",
          }}>
            <span>$100</span><span>$10K</span><span>$100K</span><span>$1M</span>
          </div>
        </div>
        <div style={{ position: "relative" }}>
          <span style={{
            position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)",
            color: "#64748b", fontSize: 13, pointerEvents: "none",
          }}>$</span>
          <input
            type="number"
            value={amount}
            min={0}
            onChange={(e) => setAmount(Math.max(0, parseInt(e.target.value) || 0))}
            style={{
              padding: "0.5rem 0.75rem 0.5rem 1.5rem",
              background: "rgba(10, 22, 40, 0.8)",
              border: "1px solid rgba(77, 159, 255, 0.3)",
              borderRadius: "0.375rem",
              color: "#eef3fb",
              fontSize: 14,
              fontFamily: "'JetBrains Mono', monospace",
              width: 130,
            }}
            aria-label="Investment amount input"
          />
        </div>
      </div>

      {/* Comparison cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
        <div style={{
          padding: "1rem",
          background: "rgba(77, 159, 255, 0.06)",
          border: "1px solid rgba(77, 159, 255, 0.2)",
          borderRadius: "0.375rem",
        }}>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
            Buy {etfTicker} (one fund)
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#4d9fff", fontFamily: "'JetBrains Mono', monospace" }}>
            {allowFractional ? etfShares.toFixed(4) : etfShares} {etfTicker} shares
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            {fmtMoney(etfActualCost)} of ${amount.toLocaleString()} deployed
            {!allowFractional && etfPrice && ` · $${(amount - etfActualCost).toFixed(2)} left over`}
          </div>
        </div>

        <div style={{
          padding: "1rem",
          background: "rgba(192, 132, 252, 0.06)",
          border: "1px solid rgba(192, 132, 252, 0.2)",
          borderRadius: "0.375rem",
        }}>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
            Replicate with holdings ({rows.length} positions)
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#c084fc", fontFamily: "'JetBrains Mono', monospace" }}>
            {fmtMoney(replicatedTotal)} deployed
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            Covers {(coverageWeight * 100).toFixed(1)}% of fund weight ·{" "}
            {uninvested >= 0 ? `${fmtMoney(uninvested)} uninvested` : `over by ${fmtMoney(-uninvested)}`}
          </div>
        </div>
      </div>

      {/* Fractional toggle */}
      <label style={{
        display: "inline-flex", alignItems: "center", gap: "0.5rem",
        fontSize: 12, color: "#94a3b8", cursor: "pointer", marginBottom: "1rem",
      }}>
        <input
          type="checkbox"
          checked={allowFractional}
          onChange={(e) => setAllowFractional(e.target.checked)}
          style={{ accentColor: "#4d9fff" }}
        />
        Allow fractional shares (most brokerages support this)
      </label>

      {/* Per-holding share table */}
      <div style={{
        overflowX: "auto",
        background: "rgba(10, 22, 40, 0.4)",
        borderRadius: "0.375rem",
      }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: "rgba(148, 163, 184, 0.06)", textAlign: "left" }}>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600 }}>Symbol</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600 }}>Weight</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>Allocation</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>Price</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>Shares</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>Actual Cost</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.symbol} style={{ borderTop: "1px solid rgba(148, 163, 184, 0.06)" }}>
                <td style={{ padding: "0.5rem 0.75rem", color: "#eef3fb", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                  {r.symbol}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
                  {(r.weight * 100).toFixed(2)}%
                </td>
                <td style={{ padding: "0.5rem 0.75rem", color: "#cbd5e1", textAlign: "right", fontFamily: "'JetBrains Mono', monospace" }}>
                  {fmtMoney(r.dollars)}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", textAlign: "right", fontFamily: "'JetBrains Mono', monospace" }}>
                  {fmtMoney(r.price)}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", color: "#4d9fff", textAlign: "right", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                  {allowFractional ? r.shares.toFixed(4) : r.shares}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", color: "#cbd5e1", textAlign: "right", fontFamily: "'JetBrains Mono', monospace" }}>
                  {fmtMoney(r.actualCost)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{
        marginTop: "0.75rem", padding: "0.625rem 0.75rem",
        background: "rgba(251, 178, 76, 0.06)",
        border: "1px solid rgba(251, 178, 76, 0.2)",
        borderRadius: "0.375rem",
        fontSize: 11, color: "#cbd5e1", lineHeight: 1.5,
      }}>
        <AlertTriangle size={11} style={{ display: "inline", marginRight: 4, verticalAlign: "text-bottom", color: "#fbb24c" }} />
        This replicates only the top-N holdings Yahoo discloses. The remaining{" "}
        {(((1 - coverageWeight)) * 100).toFixed(1)}% of the fund isn't represented.
        Replicating perfectly requires the issuer's full holdings list.
      </div>
    </Section>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Performance & contribution table
// ────────────────────────────────────────────────────────────────────────────

function PerformanceTable({
  holdings, etfPerf, ticker,
}: {
  holdings: Holding[];
  etfPerf?: Partial<Record<Period, number | null>>;
  ticker: string;
}) {
  const [period, setPeriod] = useState<Period>("1y");
  const [sort, setSort] = useState<"weight" | "perf" | "contrib">("contrib");

  const etfReturn = etfPerf?.[period];

  const enriched = holdings.map(h => {
    const p = h.performance?.[period];
    const c = h.contribution?.[period];
    const d = etfReturn != null && p != null ? (p - etfReturn) : null;
    return { ...h, perf: p, contrib: c, vsEtf: d };
  });

  const sorted = [...enriched].sort((a, b) => {
    if (sort === "weight") return (b.weight || 0) - (a.weight || 0);
    if (sort === "perf") return (b.perf ?? -Infinity) - (a.perf ?? -Infinity);
    return (b.contrib ?? -Infinity) - (a.contrib ?? -Infinity);
  });

  // Best/worst contributors
  const withContrib = enriched.filter(h => h.contrib != null);
  const top = [...withContrib].sort((a, b) => (b.contrib ?? 0) - (a.contrib ?? 0)).slice(0, 3);
  const drags = [...withContrib].sort((a, b) => (a.contrib ?? 0) - (b.contrib ?? 0)).slice(0, 3);

  return (
    <Section
      icon={<TrendingUp size={16} />}
      title="Performance & Contribution"
      subtitle={`What's driving ${ticker} — and what's dragging it down.`}
      accent="#fbb24c"
    >
      {/* Period switcher */}
      <div style={{ display: "flex", gap: "0.25rem", marginBottom: "1rem", flexWrap: "wrap" }}>
        {PERIODS.map((p) => (
          <button
            key={p.key}
            onClick={() => setPeriod(p.key)}
            style={{
              padding: "0.4rem 0.85rem",
              background: period === p.key ? "rgba(251, 178, 76, 0.12)" : "rgba(15, 23, 42, 0.6)",
              border: period === p.key ? "1px solid rgba(251, 178, 76, 0.4)" : "1px solid rgba(148, 163, 184, 0.15)",
              borderRadius: "0.375rem",
              color: period === p.key ? "#fbb24c" : "#94a3b8",
              fontSize: 11,
              fontWeight: 600,
              cursor: "pointer",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* ETF return headline */}
      <div style={{
        padding: "0.75rem 1rem",
        background: "rgba(10, 22, 40, 0.5)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: "0.375rem",
        marginBottom: "1rem",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            {ticker} return — {PERIODS.find(p => p.key === period)?.label}
          </div>
          <div style={{
            fontSize: 22, fontWeight: 700, color: perfColor(etfReturn),
            fontFamily: "'JetBrains Mono', monospace", marginTop: 2,
          }}>
            {fmtPct(etfReturn)}
          </div>
        </div>
        <div style={{ fontSize: 11, color: "#64748b", textAlign: "right", maxWidth: 280 }}>
          Drag = holding underperformed the ETF this period.
          Contribution = weight × return (how much it moved the fund).
        </div>
      </div>

      {/* Top/drag callouts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
        <div style={{
          padding: "0.875rem",
          background: "rgba(74, 222, 128, 0.06)",
          border: "1px solid rgba(74, 222, 128, 0.25)",
          borderRadius: "0.375rem",
        }}>
          <div style={{ fontSize: 11, color: "#4ade80", marginBottom: "0.5rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            <TrendingUp size={11} style={{ display: "inline", marginRight: 4, verticalAlign: "text-bottom" }} />
            Top contributors
          </div>
          {top.length === 0 && <div style={{ fontSize: 11, color: "#64748b" }}>No data for this period.</div>}
          {top.map(h => (
            <div key={h.symbol} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "2px 0" }}>
              <span style={{ color: "#eef3fb", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                {h.symbol}
              </span>
              <span style={{ color: "#4ade80", fontFamily: "'JetBrains Mono', monospace" }}>
                +{fmtBp(h.contrib)}
              </span>
            </div>
          ))}
        </div>

        <div style={{
          padding: "0.875rem",
          background: "rgba(255, 90, 110, 0.06)",
          border: "1px solid rgba(255, 90, 110, 0.25)",
          borderRadius: "0.375rem",
        }}>
          <div style={{ fontSize: 11, color: "#ff5a6e", marginBottom: "0.5rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            <TrendingDown size={11} style={{ display: "inline", marginRight: 4, verticalAlign: "text-bottom" }} />
            Biggest drags
          </div>
          {drags.length === 0 && <div style={{ fontSize: 11, color: "#64748b" }}>No data for this period.</div>}
          {drags.map(h => (
            <div key={h.symbol} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "2px 0" }}>
              <span style={{ color: "#eef3fb", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                {h.symbol}
              </span>
              <span style={{ color: "#ff5a6e", fontFamily: "'JetBrains Mono', monospace" }}>
                {fmtBp(h.contrib)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Sort selector */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem", alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "#64748b" }}>Sort by:</span>
        {(["weight", "perf", "contrib"] as const).map(s => (
          <button
            key={s}
            onClick={() => setSort(s)}
            style={{
              padding: "0.25rem 0.625rem",
              background: sort === s ? "rgba(77, 159, 255, 0.12)" : "transparent",
              border: sort === s ? "1px solid rgba(77, 159, 255, 0.3)" : "1px solid rgba(148, 163, 184, 0.15)",
              borderRadius: "0.25rem",
              color: sort === s ? "#4d9fff" : "#94a3b8",
              fontSize: 11,
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            {s === "weight" ? "Weight" : s === "perf" ? "Return" : "Contribution"}
          </button>
        ))}
      </div>

      <div style={{ overflowX: "auto", background: "rgba(10, 22, 40, 0.4)", borderRadius: "0.375rem" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: "rgba(148, 163, 184, 0.06)", textAlign: "left" }}>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600 }}>Symbol</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600 }}>Name</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>Weight</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>Return</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>vs ETF</th>
              <th style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", fontWeight: 600, textAlign: "right" }}>Contribution</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((h) => (
              <tr key={h.symbol} style={{ borderTop: "1px solid rgba(148, 163, 184, 0.06)" }}>
                <td style={{ padding: "0.5rem 0.75rem", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: "#eef3fb" }}>
                  {h.symbol}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", color: "#94a3b8", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {h.name || "—"}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", color: "#cbd5e1", fontFamily: "'JetBrains Mono', monospace" }}>
                  {(h.weight * 100).toFixed(2)}%
                </td>
                <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", color: perfColor(h.perf), fontFamily: "'JetBrains Mono', monospace" }}>
                  {fmtPct(h.perf)}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", color: perfColor(h.vsEtf), fontFamily: "'JetBrains Mono', monospace" }}>
                  {h.vsEtf == null ? "—" : `${h.vsEtf >= 0 ? "+" : ""}${(h.vsEtf * 100).toFixed(2)}%`}
                </td>
                <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", color: perfColor(h.contrib), fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                  {h.contrib == null ? "—" : `${h.contrib >= 0 ? "+" : ""}${fmtBp(h.contrib)}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Holdings detail (expandable rows with full Yahoo info)
// ────────────────────────────────────────────────────────────────────────────

function HoldingDetail({ h }: { h: Holding }) {
  return (
    <div style={{
      padding: "1rem",
      background: "rgba(5, 10, 18, 0.6)",
      borderTop: "1px solid rgba(148, 163, 184, 0.1)",
      fontSize: 12,
    }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.875rem" }}>
        <div>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 6, letterSpacing: "0.05em" }}>Identity</div>
          <DetailRow label="Sector" value={h.sector} />
          <DetailRow label="Industry" value={h.industry} />
          <DetailRow label="Country" value={h.country} />
          <DetailRow label="Exchange" value={h.exchange} />
          <DetailRow label="Currency" value={h.currency} />
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 6, letterSpacing: "0.05em" }}>Listing</div>
          <DetailRow label="First trade date" value={h.first_trade_date} />
          <DetailRow label="Years listed" value={h.listing_age_years != null ? `${h.listing_age_years} yrs` : null} />
          <DetailRow label="SPAC heuristic" value={
            h.is_spac_heuristic
              ? <span style={{ color: "#fbb24c" }}>flagged by name pattern</span>
              : <span style={{ color: "#64748b" }}>no match</span>
          } />
          <DetailRow label="Headquarters" value={h.headquarters} />
          <DetailRow label="Employees" value={h.employees != null ? h.employees.toLocaleString() : null} />
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 6, letterSpacing: "0.05em" }}>Valuation & Size</div>
          <DetailRow label="Market cap" value={fmtBigMoney(h.market_cap)} />
          <DetailRow label="Shares outstanding" value={h.shares_outstanding != null ? fmtBigMoney(h.shares_outstanding).replace("$", "") : null} />
          <DetailRow label="Trailing P/E" value={fmtNum(h.trailing_pe)} />
          <DetailRow label="Forward P/E" value={fmtNum(h.forward_pe)} />
          <DetailRow label="Beta" value={fmtNum(h.beta)} />
          <DetailRow label="Dividend yield" value={h.dividend_yield != null ? fmtPct(h.dividend_yield) : null} />
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 6, letterSpacing: "0.05em" }}>52-Week Range</div>
          <DetailRow label="High" value={fmtMoney(h["52w_high"])} />
          <DetailRow label="Low" value={fmtMoney(h["52w_low"])} />
          <DetailRow label="Position" value={h["52w_pos_pct"] != null ? `${(h["52w_pos_pct"] * 100).toFixed(0)}% of range` : null} />
          {h["52w_pos_pct"] != null && (
            <div style={{ marginTop: 6 }}>
              <div style={{ position: "relative", height: 4, background: "rgba(148, 163, 184, 0.2)", borderRadius: 2 }}>
                <div style={{
                  position: "absolute", top: 0, left: `${h["52w_pos_pct"] * 100}%`,
                  width: 3, height: 4, background: "#4d9fff", boxShadow: "0 0 6px rgba(77, 159, 255, 0.6)",
                }} />
              </div>
            </div>
          )}
        </div>
      </div>

      {h.long_business_summary && (
        <div style={{ marginTop: "0.875rem", padding: "0.625rem 0.75rem", background: "rgba(15, 23, 42, 0.6)", borderRadius: "0.375rem" }}>
          <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", marginBottom: 4, letterSpacing: "0.05em" }}>About</div>
          <div style={{ fontSize: 11, color: "#cbd5e1", lineHeight: 1.5 }}>
            {h.long_business_summary}
          </div>
          {h.website && (
            <a href={h.website} target="_blank" rel="noopener noreferrer" style={{ fontSize: 11, color: "#4d9fff", marginTop: 6, display: "inline-block" }}>
              {h.website} ↗
            </a>
          )}
        </div>
      )}

      {h.fetch_error && (
        <div style={{ marginTop: "0.75rem", padding: "0.5rem 0.625rem", background: "rgba(255, 90, 110, 0.08)", border: "1px solid rgba(255, 90, 110, 0.2)", borderRadius: "0.375rem", fontSize: 11, color: "#fda4af" }}>
          Partial data: {h.fetch_error}
        </div>
      )}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: 11 }}>
      <span style={{ color: "#94a3b8" }}>{label}</span>
      <span style={{ color: value == null || value === "—" ? "#64748b" : "#cbd5e1", fontFamily: "'JetBrains Mono', monospace", textAlign: "right" }}>
        {value == null || value === "" ? "—" : value}
      </span>
    </div>
  );
}

function HoldingsExplorer({ holdings }: { holdings: Holding[] }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (sym: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(sym)) next.delete(sym);
      else next.add(sym);
      return next;
    });
  };

  return (
    <Section
      icon={<Layers size={16} />}
      title="Holdings Explorer"
      subtitle={`Click any holding for full Yahoo Finance detail — sector, industry, listing date, valuation, business summary.`}
      accent="#c084fc"
    >
      <div style={{ background: "rgba(10, 22, 40, 0.4)", borderRadius: "0.375rem", overflow: "hidden" }}>
        {holdings.map((h, i) => {
          const isOpen = expanded.has(h.symbol);
          return (
            <div key={h.symbol} style={{ borderTop: i > 0 ? "1px solid rgba(148, 163, 184, 0.08)" : "none" }}>
              <button
                onClick={() => toggle(h.symbol)}
                style={{
                  width: "100%",
                  display: "grid",
                  gridTemplateColumns: "auto 1fr auto auto",
                  alignItems: "center",
                  gap: "0.75rem",
                  padding: "0.75rem 1rem",
                  background: isOpen ? "rgba(192, 132, 252, 0.04)" : "transparent",
                  border: "none",
                  cursor: "pointer",
                  textAlign: "left",
                  fontFamily: "inherit",
                  color: "#eef3fb",
                }}
              >
                {isOpen ? <ChevronUp size={14} style={{ color: "#94a3b8" }} /> : <ChevronDown size={14} style={{ color: "#94a3b8" }} />}
                <div style={{ minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 13 }}>{h.symbol}</span>
                    {h.is_spac_heuristic && (
                      <span style={{
                        fontSize: 9, padding: "2px 6px",
                        background: "rgba(251, 178, 76, 0.15)",
                        color: "#fbb24c",
                        borderRadius: 3,
                        fontFamily: "'JetBrains Mono', monospace",
                        letterSpacing: "0.05em",
                      }} title="Name matches a SPAC pattern — heuristic only">SPAC?</span>
                    )}
                  </div>
                  <div style={{ fontSize: 11, color: "#94a3b8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {h.name || "—"} {h.sector && `· ${h.sector}`}
                  </div>
                </div>
                <div style={{ textAlign: "right", fontFamily: "'JetBrains Mono', monospace" }}>
                  <div style={{ fontSize: 12, color: "#cbd5e1" }}>{(h.weight * 100).toFixed(2)}%</div>
                  <div style={{ fontSize: 10, color: "#64748b" }}>weight</div>
                </div>
                <div style={{ textAlign: "right", fontFamily: "'JetBrains Mono', monospace", minWidth: 80 }}>
                  <div style={{ fontSize: 12, color: "#cbd5e1" }}>{fmtMoney(h.price)}</div>
                  <div style={{ fontSize: 10, color: "#64748b" }}>last</div>
                </div>
              </button>
              {isOpen && <HoldingDetail h={h} />}
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Sector breakdown
// ────────────────────────────────────────────────────────────────────────────

function SectorBreakdown({ sectors }: { sectors: Record<string, number> }) {
  const entries = Object.entries(sectors).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;

  const colors = ["#4d9fff", "#c084fc", "#fbb24c", "#4ade80", "#ff5a6e", "#7cc4ff", "#a78bfa", "#fb923c", "#22d3ee", "#f472b6", "#94a3b8"];

  return (
    <Section
      icon={<PieChart size={16} />}
      title="Sector Breakdown"
      subtitle="As reported by Yahoo Finance"
      accent="#4ade80"
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {entries.map(([sector, weight], i) => {
          const color = colors[i % colors.length];
          return (
            <div key={sector}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                <span style={{ color: "#cbd5e1" }}>{sector}</span>
                <span style={{ color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
                  {(weight * 100).toFixed(2)}%
                </span>
              </div>
              <div style={{ height: 6, background: "rgba(148, 163, 184, 0.1)", borderRadius: 3, overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${Math.min(100, weight * 100)}%`,
                  background: color, opacity: 0.8,
                }} />
              </div>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// ETF metadata header card
// ────────────────────────────────────────────────────────────────────────────

function ETFHeader({ info, perf, holdingsCount }: {
  info: ETFInfo;
  perf?: Partial<Record<Period, number | null>>;
  holdingsCount: number;
}) {
  const isPassive = info.management_style.toLowerCase().includes("passive");
  return (
    <div style={{
      marginTop: "1.5rem",
      padding: "1.5rem",
      background: "linear-gradient(135deg, rgba(77, 159, 255, 0.06), rgba(192, 132, 252, 0.04))",
      border: "1px solid rgba(77, 159, 255, 0.2)",
      borderRadius: "0.5rem",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.625rem", flexWrap: "wrap" }}>
            <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: "#eef3fb", fontFamily: "'JetBrains Mono', monospace" }}>
              {info.ticker}
            </h2>
            <span style={{
              padding: "3px 8px",
              fontSize: 10,
              background: isPassive ? "rgba(77, 159, 255, 0.15)" : "rgba(251, 178, 76, 0.15)",
              color: isPassive ? "#4d9fff" : "#fbb24c",
              borderRadius: 3,
              fontWeight: 600,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {isPassive ? "Passive" : "Active"}
            </span>
            {info.legal_type && (
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{info.legal_type}</span>
            )}
          </div>
          <div style={{ fontSize: 14, color: "#b3c2d8", marginTop: 4 }}>
            {info.long_name || "—"}
          </div>
          {info.family && (
            <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
              <Briefcase size={10} style={{ display: "inline", marginRight: 4, verticalAlign: "text-bottom" }} />
              {info.family}
            </div>
          )}
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 11, color: "#64748b" }}>Last price</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: "#eef3fb", fontFamily: "'JetBrains Mono', monospace" }}>
            {fmtMoney(info.current_price)}
          </div>
          {info.nav_price != null && (
            <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
              NAV: {fmtMoney(info.nav_price)}
            </div>
          )}
        </div>
      </div>

      {/* Key stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.625rem" }}>
        <StatTile
          label="Expense ratio"
          value={info.expense_ratio != null ? fmtPct(info.expense_ratio, 3) : "—"}
          sub={info.expense_ratio != null ? `${(info.expense_ratio * 10000).toFixed(0)} bps` : undefined}
          color={info.expense_ratio != null && info.expense_ratio < 0.001 ? "#4ade80" : undefined}
        />
        <StatTile
          label="Total AUM"
          value={fmtBigMoney(info.total_assets)}
        />
        <StatTile
          label="30d SEC yield"
          value={info.yield != null ? fmtPct(info.yield) : "—"}
        />
        <StatTile
          label="Tracks"
          value={<span style={{ fontSize: 11 }}>{info.tracks_index || "—"}</span>}
        />
        <StatTile
          label="Inception"
          value={<span style={{ fontSize: 13 }}>{info.inception_date || "—"}</span>}
        />
        <StatTile
          label="Holdings shown"
          value={holdingsCount}
          sub="top-N from Yahoo"
        />
      </div>

      {/* Rebalance / category info */}
      <div style={{ marginTop: "1rem", padding: "0.75rem 1rem", background: "rgba(10, 22, 40, 0.5)", border: "1px solid rgba(148, 163, 184, 0.1)", borderRadius: "0.375rem", fontSize: 12, color: "#cbd5e1" }}>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
          <Calendar size={13} style={{ color: "#94a3b8", marginTop: 2, flexShrink: 0 }} />
          <div>
            <strong style={{ color: "#eef3fb" }}>Rebalance schedule:</strong> {info.rebalance_schedule}
            {info.category && (
              <span style={{ color: "#64748b" }}> · Category: {info.category}</span>
            )}
          </div>
        </div>
      </div>

      {info.long_business_summary && (
        <div style={{ marginTop: "0.75rem", fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>
          {info.long_business_summary}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────────

interface ETFBuilderViewProps {
  initialTicker?: string;
}

const SUGGESTIONS = ["SPY", "QQQ", "VOO", "VTI", "IWM", "DIA", "ARKK", "XLK", "XLF", "JEPI"];

export default function ETFBuilderView({ initialTicker }: ETFBuilderViewProps = {}) {
  const [input, setInput] = useState(initialTicker ?? "");
  const [ticker, setTicker] = useState<string | null>(initialTicker ?? null);

  const { data, isLoading, isError, error } = useQuery<ETFResult>({
    queryKey: ["/api/etf", ticker],
    queryFn: async () => {
      const r = await apiRequest("GET", `/api/etf/${ticker}`);
      return r.json();
    },
    enabled: !!ticker,
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 min — backend already caches 15 min
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = input.trim().toUpperCase();
    if (t && /^[A-Z.]{1,10}$/.test(t)) setTicker(t);
  };

  const holdings = data?.holdings || [];
  const sortedHoldings = useMemo(
    () => [...holdings].sort((a, b) => (b.weight || 0) - (a.weight || 0)),
    [holdings]
  );

  return (
    <div data-testid="section-etf-builder">
      {/* Search */}
      <div style={{ display: "flex", justifyContent: "center", marginTop: "1rem" }}>
        <form onSubmit={handleSubmit} style={{ width: "100%", maxWidth: 600 }}>
          <div style={{
            display: "flex", gap: "0.5rem", padding: "0.5rem",
            background: "rgba(15, 23, 42, 0.8)",
            border: "1px solid rgba(77, 159, 255, 0.3)",
            borderRadius: "0.5rem",
          }}>
            <Search size={18} style={{ color: "#4d9fff", alignSelf: "center", marginLeft: "0.5rem" }} />
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value.toUpperCase())}
              placeholder="Enter ETF ticker — e.g. SPY, QQQ, VOO, ARKK"
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                color: "#eef3fb",
                fontSize: 16,
                fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: "0.05em",
              }}
              autoFocus
            />
            <button type="submit" style={{
              padding: "0.5rem 1.25rem",
              background: "rgba(77, 159, 255, 0.15)",
              border: "1px solid rgba(77, 159, 255, 0.4)",
              color: "#4d9fff",
              borderRadius: "0.375rem",
              fontWeight: 600,
              fontSize: 13,
              cursor: "pointer",
              fontFamily: "inherit",
              letterSpacing: "0.05em",
            }}>
              ANALYZE
            </button>
          </div>
        </form>
      </div>

      {/* Suggestions */}
      {!ticker && (
        <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: "0.75rem", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Quick start
          </div>
          <div style={{ display: "flex", justifyContent: "center", gap: "0.4rem", flexWrap: "wrap" }}>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => { setInput(s); setTicker(s); }}
                style={{
                  padding: "0.375rem 0.875rem",
                  background: "rgba(15, 23, 42, 0.6)",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  color: "#cbd5e1",
                  borderRadius: "0.25rem",
                  cursor: "pointer",
                  fontSize: 12,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontWeight: 600,
                }}
              >
                {s}
              </button>
            ))}
          </div>
          <div style={{
            marginTop: "2rem", padding: "1rem 1.25rem",
            background: "rgba(15, 23, 42, 0.5)",
            border: "1px solid rgba(148, 163, 184, 0.1)",
            borderRadius: "0.5rem",
            maxWidth: 640, margin: "2rem auto 0",
            fontSize: 12, color: "#94a3b8", lineHeight: 1.6,
          }}>
            <Sparkles size={13} style={{ display: "inline", marginRight: 6, color: "#fbb24c", verticalAlign: "text-bottom" }} />
            <strong style={{ color: "#cbd5e1" }}>ETF Builder</strong> breaks any ETF down into its underlying holdings, shows you
            how each one is performing across multiple periods, surfaces which holdings are dragging the fund down, and lets you
            calculate exactly how many shares of each you'd need to replicate the ETF with a given dollar amount.
            All data sourced from Yahoo Finance. Top-N holdings only (Yahoo doesn't expose full holdings lists).
          </div>
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div style={{ textAlign: "center", marginTop: "3rem", color: "#94a3b8" }}>
          <div style={{ fontSize: 13, marginBottom: 8 }}>Pulling ETF metadata and holdings…</div>
          <div style={{ fontSize: 11, color: "#64748b" }}>
            Fetching per-holding details in parallel — this can take 10-30 seconds on first run.
          </div>
        </div>
      )}

      {/* Error */}
      {(isError || data?.error) && !isLoading && (
        <div style={{
          marginTop: "2rem", padding: "1.25rem",
          background: "rgba(255, 90, 110, 0.08)",
          border: "1px solid rgba(255, 90, 110, 0.3)",
          borderRadius: "0.5rem",
          color: "#fda4af", fontSize: 13,
        }}>
          <AlertTriangle size={16} style={{ display: "inline", marginRight: 8, verticalAlign: "text-bottom" }} />
          {data?.error || (error as any)?.message || "ETF analysis failed. The ticker must be an ETF (e.g. SPY, not AAPL)."}
        </div>
      )}

      {/* Results */}
      {data && !data.error && data.info && (
        <>
          <ETFHeader
            info={data.info}
            perf={data.etf_performance}
            holdingsCount={data.holdings_returned || 0}
          />

          {/* Coverage warning if Yahoo only gave us a partial holdings list */}
          {data.holdings_truncated && (
            <div style={{
              marginTop: "1rem", padding: "0.75rem 1rem",
              background: "rgba(251, 178, 76, 0.06)",
              border: "1px solid rgba(251, 178, 76, 0.25)",
              borderRadius: "0.375rem",
              fontSize: 12, color: "#cbd5e1", lineHeight: 1.5,
            }}>
              <Info size={13} style={{ display: "inline", marginRight: 6, color: "#fbb24c", verticalAlign: "text-bottom" }} />
              <strong style={{ color: "#fbb24c" }}>Partial holdings:</strong>{" "}
              Yahoo Finance returned {data.holdings_returned} holdings covering{" "}
              <strong>{((data.holdings_coverage || 0) * 100).toFixed(1)}%</strong> of the fund weight.
              For complete holdings, see the issuer's site ({data.info.family || "fund issuer"}).
            </div>
          )}

          <PerformanceTable
            holdings={sortedHoldings}
            etfPerf={data.etf_performance}
            ticker={data.info.ticker}
          />

          <InvestmentCalculator
            holdings={sortedHoldings}
            etfPrice={data.info.current_price}
            etfTicker={data.info.ticker}
          />

          {data.sector_weights && Object.keys(data.sector_weights).length > 0 && (
            <SectorBreakdown sectors={data.sector_weights} />
          )}

          <HoldingsExplorer holdings={sortedHoldings} />

          {/* Sources footer */}
          <div style={{
            marginTop: "1.5rem", padding: "0.75rem 1rem",
            background: "rgba(77, 159, 255, 0.04)",
            border: "1px solid rgba(77, 159, 255, 0.15)",
            borderRadius: "0.375rem",
            fontSize: 11, color: "#94a3b8", lineHeight: 1.5,
          }}>
            <strong style={{ color: "#4d9fff" }}>Sources:</strong>{" "}
            {data.data_source_notes || "Yahoo Finance via yfinance."}
            {data.as_of && (
              <span style={{ color: "#64748b" }}> · Generated {new Date(data.as_of).toLocaleString()}</span>
            )}
          </div>

          <div style={{ marginTop: "1rem", textAlign: "center", fontSize: 10, color: "#64748b" }}>
            For educational purposes only. Not financial advice. Always read the prospectus before investing.
          </div>
        </>
      )}
    </div>
  );
}
