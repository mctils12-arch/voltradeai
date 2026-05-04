import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  TrendingUp, TrendingDown, Minus, AlertTriangle, Eye,
  Users, Building2, Layers, Target, Activity, Lock,
} from "lucide-react";

// ────────────────────────────────────────────────────────────────────────────
// Types — match insights.py output
// ────────────────────────────────────────────────────────────────────────────

interface InsiderSection {
  available: boolean;
  reason?: string;       // populated when available=false (API/network error)
  no_data?: boolean;     // populated when available=true but ticker has no insider activity
  mspr?: number;
  change?: number;
  signal?: "bullish" | "bearish" | "neutral" | "no_data";
  total_buys?: number;
  total_sells?: number;
  plain_english?: string;
}

interface InstitutionalSection {
  available: boolean;
  reason?: string;
  no_data?: boolean;
  direction?: "increasing" | "decreasing" | "stable";
  signal_strength?: number;
  major_holders?: string[];
  holders_added?: number;
  holders_reduced?: number;
  whales?: string[];
  whale_interest?: boolean;
  plain_english?: string;
}

interface FloatSection {
  available: boolean;
  shares_outstanding?: number;
  float_shares?: number;
  insider_locked_pct?: number;
  held_insiders_pct?: number;
  held_institutions_pct?: number;
  short_count?: number;
  short_prior?: number;
  short_pct_float?: number;
  short_ratio_days?: number;
  squeeze_setup?: "low" | "moderate" | "high";
  short_trend?: "rising_fast" | "rising" | "stable" | "covering" | "covering_fast";
  plain_english?: string;
}

interface FlowSection {
  available: boolean;
  obv_signal?: "accumulation" | "distribution" | "neutral";
  mfi?: number;
  volume_ratio?: number;
  up_down_volume_ratio?: number;
  plain_english?: string;
}

interface SRLevel {
  label: string;
  price: number;
  kind: "support" | "resistance" | "pivot";
  distance_pct: number;
}

interface SRSection {
  available: boolean;
  spot?: number;
  levels?: SRLevel[];
  nearest_support?: SRLevel | null;
  nearest_resistance?: SRLevel | null;
  poc?: number;
  poc_volume_pct?: number;
  plain_english?: string;
}

interface OptionsSection {
  available: boolean;
  expiration?: string;
  spot?: number;
  gamma_pin?: number;
  gamma_pin_oi?: number;
  gamma_pin_dist_pct?: number;
  max_pain?: number;
  max_pain_dist_pct?: number;
  put_call_ratio?: number;
  put_call_signal?: string;
  put_call_interp?: string;
  total_call_oi?: number;
  total_put_oi?: number;
  plain_english?: string;
}

interface InsightsResult {
  ticker: string;
  company_name?: string;
  spot?: number;
  generated_at?: string;
  insider: InsiderSection;
  institutional: InstitutionalSection;
  float: FloatSection;
  flow: FlowSection;
  support_resistance: SRSection;
  options: OptionsSection;
  error?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function fmtBig(n?: number): string {
  if (n === undefined || n === null) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

function fmtPct(n?: number, decimals = 2): string {
  if (n === undefined || n === null) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(decimals)}%`;
}

function fmtPrice(n?: number): string {
  if (n === undefined || n === null) return "—";
  return `$${n.toFixed(2)}`;
}

// Color by signal
function signalColor(signal?: string): string {
  if (!signal) return "#94a3b8";
  const s = signal.toLowerCase();
  if (s.includes("bull") || s.includes("accumulat") || s.includes("increas")) return "#30d158";
  if (s.includes("bear") || s.includes("distribut") || s.includes("decreas")) return "#ff453a";
  if (s.includes("greed")) return "#ff9f0a";  // contrarian sell
  if (s.includes("fear")) return "#00e5ff";   // contrarian buy
  return "#94a3b8";
}

// Section card shell — matches existing analyze.tsx aesthetic
function Section({
  icon, title, subtitle, children, accent = "#00e5ff",
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
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
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

// Statistic row
function StatRow({ label, value, color }: { label: string; value: React.ReactNode; color?: string }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "0.5rem 0",
      borderBottom: "1px solid rgba(148, 163, 184, 0.08)",
      fontSize: 13,
    }}>
      <span style={{ color: "#94a3b8" }}>{label}</span>
      <span style={{ color: color || "#e2e8f0", fontWeight: 500, fontFamily: "'JetBrains Mono', monospace" }}>{value}</span>
    </div>
  );
}

// Simple horizontal gauge for MSPR (-100 to +100)
function MsprGauge({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, ((value + 100) / 200) * 100));
  const color = value > 20 ? "#30d158" : value < -20 ? "#ff453a" : "#94a3b8";
  return (
    <div style={{ marginTop: "0.5rem" }}>
      <div style={{
        position: "relative",
        height: "6px",
        background: "rgba(148, 163, 184, 0.15)",
        borderRadius: "3px",
        overflow: "hidden",
      }}>
        <div style={{
          position: "absolute",
          top: 0, bottom: 0,
          left: "50%",
          width: "1px",
          background: "rgba(148, 163, 184, 0.4)",
          zIndex: 1,
        }} />
        <div style={{
          position: "absolute",
          top: 0, bottom: 0,
          left: value >= 0 ? "50%" : `${pct}%`,
          width: `${Math.abs(value) / 2}%`,
          background: color,
          opacity: 0.7,
        }} />
      </div>
      <div style={{
        display: "flex", justifyContent: "space-between", marginTop: "0.25rem",
        fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace",
      }}>
        <span>−100 (selling)</span>
        <span>0</span>
        <span>+100 (buying)</span>
      </div>
    </div>
  );
}

// Plain-english callout
function Callout({ text, signal }: { text: string; signal?: string }) {
  const color = signalColor(signal);
  return (
    <div style={{
      marginTop: "0.75rem",
      padding: "0.625rem 0.75rem",
      background: `${color}10`,
      border: `1px solid ${color}30`,
      borderRadius: "0.375rem",
      fontSize: 12,
      lineHeight: 1.5,
      color: "#cbd5e1",
    }}>
      {text}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Sub-sections
// ────────────────────────────────────────────────────────────────────────────

function InsiderCard({ data }: { data: InsiderSection }) {
  // State 1: API/network error — show what's wrong, don't pretend it's "neutral"
  if (!data?.available) {
    return (
      <Section icon={<Eye size={16} />} title="Insider Activity" subtitle="Form 4 filings, last 12 months">
        <div style={{
          padding: "0.75rem", fontSize: 12, color: "#fda4af",
          background: "rgba(255, 69, 58, 0.06)",
          border: "1px solid rgba(255, 69, 58, 0.2)",
          borderRadius: "0.375rem",
        }}>
          <strong style={{ display: "block", marginBottom: "0.25rem", color: "#ff453a" }}>
            Insider data unavailable
          </strong>
          <span style={{ color: "#cbd5e1" }}>
            {data?.reason || "Source did not respond. May indicate FINNHUB_KEY env var is not set on server."}
          </span>
        </div>
      </Section>
    );
  }

  // State 2: API succeeded but ticker has no insider activity (small caps, etc.)
  if (data.no_data) {
    return (
      <Section icon={<Eye size={16} />} title="Insider Activity" subtitle="Form 4 filings, last 12 months">
        <div style={{
          padding: "0.75rem", fontSize: 12, color: "#94a3b8",
          background: "rgba(148, 163, 184, 0.05)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: "0.375rem",
        }}>
          <strong style={{ display: "block", marginBottom: "0.25rem", color: "#cbd5e1" }}>
            No insider transactions this period
          </strong>
          <span>{data.plain_english}</span>
        </div>
      </Section>
    );
  }

  // State 3: Real data
  const mspr = data.mspr ?? 0;
  return (
    <Section
      icon={<Eye size={16} />}
      title="Insider Activity"
      subtitle="Form 4 filings, last 12 months"
      accent={signalColor(data.signal)}
    >
      <div style={{ marginBottom: "0.75rem" }}>
        <div style={{
          display: "flex", justifyContent: "space-between", alignItems: "baseline",
          marginBottom: "0.25rem",
        }}>
          <span style={{ fontSize: 12, color: "#94a3b8" }}>Monthly Share Purchase Ratio (MSPR)</span>
          <span style={{ fontSize: 20, fontWeight: 600, color: signalColor(data.signal),
                          fontFamily: "'JetBrains Mono', monospace" }}>
            {mspr >= 0 ? "+" : ""}{mspr.toFixed(1)}
          </span>
        </div>
        <MsprGauge value={mspr} />
      </div>
      <StatRow label="Total buy transactions (12mo)" value={fmtBig(data.total_buys)} color="#30d158" />
      <StatRow label="Total sell transactions (12mo)" value={fmtBig(data.total_sells)} color="#ff453a" />
      <StatRow label="Latest month change" value={fmtBig(data.change)} />
      <StatRow label="Signal" value={
        <span style={{ color: signalColor(data.signal), textTransform: "uppercase", fontSize: 11 }}>
          {data.signal}
        </span>
      } />
      {data.plain_english && <Callout text={data.plain_english} signal={data.signal} />}
    </Section>
  );
}

function InstitutionalCard({ data }: { data: InstitutionalSection }) {
  // State 1: API/network error
  if (!data?.available) {
    return (
      <Section icon={<Building2 size={16} />} title="Institutional Positioning" subtitle="13F filings via SEC EDGAR">
        <div style={{
          padding: "0.75rem", fontSize: 12, color: "#fda4af",
          background: "rgba(255, 69, 58, 0.06)",
          border: "1px solid rgba(255, 69, 58, 0.2)",
          borderRadius: "0.375rem",
        }}>
          <strong style={{ display: "block", marginBottom: "0.25rem", color: "#ff453a" }}>
            Institutional data unavailable
          </strong>
          <span style={{ color: "#cbd5e1" }}>
            {data?.reason || "SEC EDGAR did not respond. Try again in a moment."}
          </span>
        </div>
      </Section>
    );
  }

  // State 2: Successful query but no major holders found
  if (data.no_data) {
    return (
      <Section icon={<Building2 size={16} />} title="Institutional Positioning" subtitle="13F filings via SEC EDGAR">
        <div style={{
          padding: "0.75rem", fontSize: 12, color: "#94a3b8",
          background: "rgba(148, 163, 184, 0.05)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: "0.375rem",
        }}>
          <strong style={{ display: "block", marginBottom: "0.25rem", color: "#cbd5e1" }}>
            No major 13F holders found
          </strong>
          <span>{data.plain_english}</span>
        </div>
      </Section>
    );
  }

  // State 3: Real data
  return (
    <Section
      icon={<Building2 size={16} />}
      title="Institutional Positioning"
      subtitle="13F filings via SEC EDGAR — last reporting quarter"
      accent={signalColor(data.direction)}
    >
      <StatRow label="Direction" value={
        <span style={{ color: signalColor(data.direction), textTransform: "uppercase", fontSize: 11 }}>
          {data.direction}
        </span>
      } />
      <StatRow label="Signal strength" value={`${data.signal_strength ?? 0}/100`} />
      <StatRow label="Major holders added (QoQ)" value={data.holders_added ?? 0} color="#30d158" />
      <StatRow label="Major holders reduced (QoQ)" value={data.holders_reduced ?? 0} color="#ff453a" />
      {data.major_holders && data.major_holders.length > 0 && (
        <div style={{ marginTop: "0.5rem", paddingTop: "0.5rem", borderTop: "1px solid rgba(148, 163, 184, 0.08)" }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: "0.375rem" }}>TOP HOLDERS</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem" }}>
            {data.major_holders.slice(0, 8).map((h, i) => (
              <span key={i} style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 3,
                background: "rgba(0, 229, 255, 0.08)", color: "#00e5ff",
                border: "1px solid rgba(0, 229, 255, 0.2)",
              }}>{h}</span>
            ))}
          </div>
        </div>
      )}
      {data.whales && data.whales.length > 0 && (
        <div style={{ marginTop: "0.5rem", paddingTop: "0.5rem", borderTop: "1px solid rgba(148, 163, 184, 0.08)" }}>
          <div style={{ fontSize: 11, color: "#64748b", marginBottom: "0.375rem" }}>SUPER-INVESTOR WHALES HOLDING</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem" }}>
            {data.whales.map((w, i) => (
              <span key={i} style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 3,
                background: "rgba(212, 160, 23, 0.08)", color: "#d4a017",
                border: "1px solid rgba(212, 160, 23, 0.2)",
              }}>{w}</span>
            ))}
          </div>
        </div>
      )}
      {data.plain_english && <Callout text={data.plain_english} signal={data.direction} />}
    </Section>
  );
}

function FloatCard({ data }: { data: FloatSection }) {
  if (!data?.available) {
    return <Section icon={<Layers size={16} />} title="Float & Share Structure">
      <div style={{ color: "#64748b", fontSize: 12 }}>No float data available.</div>
    </Section>;
  }
  const squeezeColor = data.squeeze_setup === "high" ? "#ff9f0a"
                     : data.squeeze_setup === "moderate" ? "#ff453a"
                     : "#94a3b8";
  const trendColor = data.short_trend?.startsWith("rising") ? "#ff453a"
                   : data.short_trend?.startsWith("covering") ? "#30d158"
                   : "#94a3b8";
  return (
    <Section icon={<Layers size={16} />} title="Float & Share Structure" subtitle="Outstanding, float, short interest">
      <StatRow label="Shares outstanding" value={fmtBig(data.shares_outstanding)} />
      <StatRow label="Public float" value={fmtBig(data.float_shares)} />
      <StatRow label="Insider-locked %" value={data.insider_locked_pct !== undefined && data.insider_locked_pct !== null ? `${data.insider_locked_pct.toFixed(1)}%` : "—"} />
      <StatRow label="Insiders %" value={data.held_insiders_pct !== undefined && data.held_insiders_pct !== null ? `${data.held_insiders_pct.toFixed(2)}%` : "—"} />
      <StatRow label="Institutions %" value={data.held_institutions_pct !== undefined && data.held_institutions_pct !== null ? `${data.held_institutions_pct.toFixed(2)}%` : "—"} />
      <div style={{ marginTop: "0.5rem", paddingTop: "0.5rem", borderTop: "1px solid rgba(148, 163, 184, 0.08)" }}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: "0.375rem" }}>SHORT INTEREST</div>
        <StatRow label="Short shares" value={fmtBig(data.short_count)} />
        <StatRow label="Short % of float" value={data.short_pct_float !== undefined && data.short_pct_float !== null ? `${data.short_pct_float.toFixed(2)}%` : "—"} />
        <StatRow label="Days to cover" value={data.short_ratio_days ?? "—"} />
        <StatRow label="MoM trend" value={
          <span style={{ color: trendColor, textTransform: "uppercase", fontSize: 11 }}>
            {data.short_trend?.replace(/_/g, " ") ?? "—"}
          </span>
        } />
        <StatRow label="Squeeze setup" value={
          <span style={{ color: squeezeColor, textTransform: "uppercase", fontSize: 11, fontWeight: 600 }}>
            {data.squeeze_setup ?? "—"}
          </span>
        } />
      </div>
      {data.plain_english && <Callout text={data.plain_english} signal={data.squeeze_setup === "high" ? "bullish" : "neutral"} />}
    </Section>
  );
}

function FlowCard({ data }: { data: FlowSection }) {
  if (!data?.available) {
    return <Section icon={<Activity size={16} />} title="Buying / Selling Pressure">
      <div style={{ color: "#64748b", fontSize: 12 }}>Insufficient history.</div>
    </Section>;
  }
  const mfiColor = (data.mfi ?? 50) > 80 ? "#ff453a"
                 : (data.mfi ?? 50) < 20 ? "#30d158"
                 : "#94a3b8";
  return (
    <Section
      icon={<Activity size={16} />}
      title="Buying / Selling Pressure"
      subtitle="OBV, MFI, volume distribution"
      accent={signalColor(data.obv_signal)}
    >
      <StatRow label="OBV signal (last 5d vs prior 5d)" value={
        <span style={{ color: signalColor(data.obv_signal), textTransform: "uppercase", fontSize: 11 }}>
          {data.obv_signal ?? "—"}
        </span>
      } />
      <StatRow label="Money Flow Index (14d)" value={
        <span style={{ color: mfiColor, fontWeight: 600 }}>
          {data.mfi !== undefined && data.mfi !== null ? data.mfi.toFixed(1) : "—"}
        </span>
      } />
      <StatRow label="Today's volume vs 20d avg" value={
        data.volume_ratio !== undefined && data.volume_ratio !== null
          ? `${data.volume_ratio.toFixed(2)}×`
          : "—"
      } />
      <StatRow label="Up-day vol ÷ down-day vol (30d)" value={
        data.up_down_volume_ratio !== undefined && data.up_down_volume_ratio !== null
          ? `${data.up_down_volume_ratio.toFixed(2)}×`
          : "—"
      } />
      {data.plain_english && <Callout text={data.plain_english} signal={data.obv_signal} />}
    </Section>
  );
}

function SRCard({ data, spot }: { data: SRSection; spot?: number }) {
  if (!data?.available) {
    return <Section icon={<Target size={16} />} title="Support & Resistance">
      <div style={{ color: "#64748b", fontSize: 12 }}>Insufficient history.</div>
    </Section>;
  }
  const ref_spot = data.spot ?? spot;
  return (
    <Section
      icon={<Target size={16} />}
      title="Support & Resistance Levels"
      subtitle={`From price history + pivot points + 60d Volume Profile (POC)`}
    >
      {ref_spot && (
        <div style={{ fontSize: 12, color: "#64748b", marginBottom: "0.5rem" }}>
          Current spot: <span style={{ color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace" }}>${ref_spot.toFixed(2)}</span>
        </div>
      )}
      {data.nearest_resistance && (
        <div style={{
          padding: "0.5rem 0.75rem", marginBottom: "0.5rem",
          background: "rgba(255, 69, 58, 0.08)",
          borderLeft: "3px solid #ff453a",
          borderRadius: "0 4px 4px 0",
        }}>
          <div style={{ fontSize: 10, color: "#ff453a", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Nearest Resistance
          </div>
          <div style={{ fontSize: 14, color: "#e2e8f0", marginTop: 2, fontFamily: "'JetBrains Mono', monospace" }}>
            ${data.nearest_resistance.price.toFixed(2)}
            <span style={{ fontSize: 11, color: "#94a3b8", marginLeft: "0.5rem" }}>
              {data.nearest_resistance.label} · {data.nearest_resistance.distance_pct >= 0 ? "+" : ""}{data.nearest_resistance.distance_pct.toFixed(1)}%
            </span>
          </div>
        </div>
      )}
      {data.nearest_support && (
        <div style={{
          padding: "0.5rem 0.75rem", marginBottom: "0.75rem",
          background: "rgba(48, 209, 88, 0.08)",
          borderLeft: "3px solid #30d158",
          borderRadius: "0 4px 4px 0",
        }}>
          <div style={{ fontSize: 10, color: "#30d158", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Nearest Support
          </div>
          <div style={{ fontSize: 14, color: "#e2e8f0", marginTop: 2, fontFamily: "'JetBrains Mono', monospace" }}>
            ${data.nearest_support.price.toFixed(2)}
            <span style={{ fontSize: 11, color: "#94a3b8", marginLeft: "0.5rem" }}>
              {data.nearest_support.label} · {data.nearest_support.distance_pct.toFixed(1)}%
            </span>
          </div>
        </div>
      )}
      {data.poc !== undefined && data.poc !== null && (
        <StatRow label={`POC (60-day, ${data.poc_volume_pct?.toFixed(0) ?? "?"}% of volume)`} value={`$${data.poc.toFixed(2)}`} />
      )}
      <div style={{ marginTop: "0.75rem", paddingTop: "0.5rem", borderTop: "1px solid rgba(148, 163, 184, 0.08)" }}>
        <div style={{ fontSize: 11, color: "#64748b", marginBottom: "0.5rem" }}>ALL LEVELS</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.25rem 0.75rem" }}>
          {(data.levels ?? []).map((lvl, i) => (
            <div key={i} style={{
              fontSize: 11, padding: "2px 0",
              fontFamily: "'JetBrains Mono', monospace",
              color: lvl.kind === "resistance" ? "#fda4af" :
                     lvl.kind === "support" ? "#86efac" : "#94a3b8",
            }}>
              <span style={{ color: "#64748b" }}>{lvl.label}:</span> ${lvl.price.toFixed(2)} <span style={{ fontSize: 10, color: "#475569" }}>({lvl.distance_pct >= 0 ? "+" : ""}{lvl.distance_pct.toFixed(1)}%)</span>
            </div>
          ))}
        </div>
      </div>
      {data.plain_english && <Callout text={data.plain_english} />}
    </Section>
  );
}

function OptionsCard({ data }: { data: OptionsSection }) {
  if (!data?.available) {
    return <Section icon={<TrendingUp size={16} />} title="Options Smart Money">
      <div style={{ color: "#64748b", fontSize: 12 }}>No options data available.</div>
    </Section>;
  }
  return (
    <Section
      icon={<TrendingUp size={16} />}
      title="Options Smart Money"
      subtitle={`Nearest expiration: ${data.expiration ?? "—"}`}
      accent={signalColor(data.put_call_signal)}
    >
      <StatRow label="Gamma pin (highest OI strike)" value={
        data.gamma_pin
          ? `$${data.gamma_pin.toFixed(2)} · ${fmtPct(data.gamma_pin_dist_pct, 1)}`
          : "—"
      } />
      <StatRow label="Gamma pin total OI" value={fmtBig(data.gamma_pin_oi)} />
      <StatRow label="Max pain" value={
        data.max_pain
          ? `$${data.max_pain.toFixed(2)} · ${fmtPct(data.max_pain_dist_pct, 1)}`
          : "—"
      } />
      <StatRow label="Put/Call ratio (OI)" value={
        data.put_call_ratio !== undefined && data.put_call_ratio !== null ? data.put_call_ratio.toFixed(2) : "—"
      } color={signalColor(data.put_call_signal)} />
      <StatRow label="Sentiment" value={
        <span style={{ color: signalColor(data.put_call_signal), textTransform: "uppercase", fontSize: 11 }}>
          {data.put_call_signal ?? "—"}
        </span>
      } />
      <div style={{ marginTop: "0.5rem", paddingTop: "0.5rem", borderTop: "1px solid rgba(148, 163, 184, 0.08)" }}>
        <StatRow label="Total call OI" value={fmtBig(data.total_call_oi)} color="#30d158" />
        <StatRow label="Total put OI" value={fmtBig(data.total_put_oi)} color="#ff453a" />
      </div>
      {data.plain_english && <Callout text={data.plain_english} signal={data.put_call_signal} />}
    </Section>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Loading skeleton
// ────────────────────────────────────────────────────────────────────────────

function InsightsSkeleton() {
  return (
    <div style={{ marginTop: "1.5rem" }}>
      {[1, 2, 3].map(i => (
        <div key={i} style={{
          height: "180px",
          marginBottom: "1rem",
          background: "rgba(15, 23, 42, 0.4)",
          border: "1px solid rgba(148, 163, 184, 0.1)",
          borderRadius: "0.5rem",
          backgroundImage: "linear-gradient(90deg, transparent 0%, rgba(148, 163, 184, 0.05) 50%, transparent 100%)",
          backgroundSize: "200% 100%",
          animation: "shimmer 2s infinite",
        }} />
      ))}
      <style>{`
        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Main view component
// ────────────────────────────────────────────────────────────────────────────

interface InsightsViewProps {
  ticker: string | null;
}

export default function InsightsView({ ticker }: InsightsViewProps) {
  const { data, isLoading, isError, error } = useQuery<InsightsResult>({
    queryKey: ["/api/insights", ticker],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/insights/${ticker}`);
      return res.json();
    },
    enabled: !!ticker,
    retry: false,
    staleTime: 60_000,  // Cache 1 min — insights data isn't real-time
  });

  if (!ticker) {
    return (
      <div style={{ padding: "2rem", textAlign: "center", color: "#64748b" }}>
        Enter a ticker above to see smart-money insights.
      </div>
    );
  }

  if (isLoading) return <InsightsSkeleton />;

  if (isError || data?.error) {
    return (
      <div style={{
        marginTop: "1.5rem", padding: "1.25rem",
        background: "rgba(255, 69, 58, 0.08)",
        border: "1px solid rgba(255, 69, 58, 0.3)",
        borderRadius: "0.5rem",
        color: "#fda4af",
        fontSize: 13,
      }}>
        <AlertTriangle size={16} style={{ display: "inline", marginRight: "0.5rem", verticalAlign: "text-bottom" }} />
        {data?.error || (error as any)?.message || "Failed to load insights."}
      </div>
    );
  }

  if (!data) return null;

  return (
    <div data-testid="section-insights">
      {/* Page header note about source */}
      <div style={{
        marginTop: "1rem",
        padding: "0.75rem 1rem",
        background: "rgba(0, 229, 255, 0.04)",
        border: "1px solid rgba(0, 229, 255, 0.15)",
        borderRadius: "0.375rem",
        fontSize: 11,
        color: "#94a3b8",
        lineHeight: 1.5,
      }}>
        <strong style={{ color: "#00e5ff" }}>Sources:</strong>{" "}
        Insider data — Finnhub (Form 4 filings).{" "}
        Institutional — SEC EDGAR (13F filings, current quarter).{" "}
        Float / short — Yahoo Finance.{" "}
        Pressure / S&R — computed from Yahoo daily bars.{" "}
        Options — Yahoo options chain, nearest monthly expiration.{" "}
        Cached 1-24h depending on source. Not real-time.
      </div>

      <InsiderCard data={data.insider} />
      <InstitutionalCard data={data.institutional} />
      <FloatCard data={data.float} />
      <FlowCard data={data.flow} />
      <SRCard data={data.support_resistance} spot={data.spot} />
      <OptionsCard data={data.options} />
    </div>
  );
}
