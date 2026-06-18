import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Search, RefreshCw, Scale, ShieldAlert, FileText, Layers, Target, Megaphone, Newspaper } from "lucide-react";

// ── Types (mirror alphadesk JSON contract) ──────────────────────────────────

interface Factor {
  name: string;
  score: number;
  weight: number;
  detail: string;
}

interface Pillar {
  name: string;
  score: number;
  factors: Factor[];
  note?: string;
}

interface Horizon {
  label: string;
  horizon_days: number;
  expected_return: number;
  tax_rate: number;
  after_tax_return: number;
  note: string;
}

interface FilingRead {
  summary: string;
  sentiment: string;
  red_flags: string[];
  catalysts: string[];
  source: string;
}

interface OptionsView {
  snapshot: {
    iv_rank?: number | null;
    expected_move_30d?: number | null;
    hv_30d?: number | null;
    iv_30d?: number | null;
    put_call_skew?: number | null;
  };
  strategy: {
    structure: string;
    direction: string;
    rationale: string;
    legs: string[];
    max_risk_note: string;
  };
}

interface NewsItem {
  headline: string;
  summary?: string;
  url?: string;
  source?: string;
  date?: string;
}

interface CatalystView {
  catalyst: { kind: string; headline?: string; url?: string; deal_price?: number | null; cash?: boolean; status?: string };
  current_price?: number | null;
  upside_pct?: number | null;
  downside_pct?: number | null;
  downside_price?: number | null;
  implied_close_prob?: number | null;
  assessment?: string;
  news?: NewsItem[];
}

interface ResearchReport {
  ticker: string;
  as_of: string;
  price: number;
  verdict: string;
  conviction: number;
  composite_score: number;
  pillars: Pillar[];
  options?: OptionsView | null;
  catalyst?: CatalystView | null;
  supply_demand_read: string;
  filing_read: FilingRead;
  market_note: string;
  horizons: Horizon[];
  best_horizon: string;
  thesis: string;
  risks: string[];
  data_completeness: number;
  provider: string;
  disclaimer: string;
  error?: string;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function verdictColor(verdict: string): string {
  const v = verdict.toLowerCase();
  if (v.includes("strong buy")) return "#30d158";
  if (v.includes("buy")) return "#4ade80";
  if (v.includes("strong sell")) return "#ff453a";
  if (v.includes("sell")) return "#ff6b5a";
  return "#f5a623"; // Hold / neutral
}

function scoreColor(score: number): string {
  if (score >= 66) return "#30d158";
  if (score >= 45) return "#f5a623";
  return "#ff6b5a";
}


// ── Sub-components ────────────────────────────────────────────────────────────

function ScoreBar({ score }: { score: number }) {
  return (
    <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden", minWidth: 80 }}>
      <div style={{ height: "100%", width: `${Math.max(0, Math.min(100, score))}%`, background: scoreColor(score), borderRadius: 3 }} />
    </div>
  );
}

function PillarCard({ pillar }: { pillar: Pillar }) {
  return (
    <div style={{ background: "rgba(77,159,255,0.04)", border: "1px solid rgba(77,159,255,0.15)", borderRadius: 8, padding: "14px 16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
        <span style={{ color: "#eef3fb", fontWeight: 600, fontSize: 14 }}>{pillar.name}</span>
        <span style={{ color: scoreColor(pillar.score), fontWeight: 700, fontFamily: "Geist Mono, monospace", fontSize: 15 }}>
          {pillar.score.toFixed(0)}
          <span style={{ color: "#5a6b82", fontSize: 11 }}>/100</span>
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {pillar.factors.map((f) => (
          <div key={f.name} style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "2px 12px", alignItems: "center" }}>
            <span style={{ color: "#b3c2d8", fontSize: 12.5 }}>{f.name}</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <ScoreBar score={f.score} />
              <span style={{ color: scoreColor(f.score), fontFamily: "Geist Mono, monospace", fontSize: 12, width: 26, textAlign: "right" }}>
                {f.score.toFixed(0)}
              </span>
            </div>
            <span style={{ gridColumn: "1 / -1", color: "#6b7c93", fontSize: 11, fontFamily: "Geist Mono, monospace" }}>{f.detail}</span>
          </div>
        ))}
      </div>
      {pillar.note && (
        <p style={{ color: "#8a9bb3", fontSize: 11.5, marginTop: 10, fontStyle: "italic" }}>{pillar.note}</p>
      )}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ResearchPage({ onSelectTicker }: { onSelectTicker?: (t: string) => void }) {
  const [input, setInput] = useState("");
  const [ticker, setTicker] = useState<string>("");

  const { data, isFetching, error, refetch } = useQuery<ResearchReport>({
    queryKey: ["/api/research", ticker],
    queryFn: async () => {
      const r = await apiRequest("GET", `/api/research/${encodeURIComponent(ticker)}`);
      const json = await r.json();
      if (!r.ok) throw new Error(json?.error || "Research failed");
      return json;
    },
    enabled: !!ticker,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    const t = input.trim().toUpperCase();
    if (t) setTicker(t);
  };

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "8px 0 48px" }}>
      {/* Header */}
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ color: "#eef3fb", fontSize: 22, fontWeight: 700, margin: 0, display: "flex", alignItems: "center", gap: 8 }}>
          <Scale size={20} style={{ color: "#4d9fff" }} /> Research
        </h1>
        <p style={{ color: "#8a9bb3", fontSize: 13, margin: "6px 0 0" }}>
          Explainable buy/sell verdict from six weighted pillars — fundamentals, valuation, supply/demand,
          market context, SEC filings, and options — plus catalyst detection for special situations.
        </p>
      </div>

      {/* Search */}
      <form onSubmit={submit} style={{ display: "flex", gap: 8, marginBottom: 22 }}>
        <div style={{ position: "relative", flex: 1, maxWidth: 320 }}>
          <Search size={15} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "#5a6b82" }} />
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter a ticker (e.g. AAPL)"
            style={{
              width: "100%", padding: "10px 12px 10px 34px", background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(77,159,255,0.2)", borderRadius: 6, color: "#eef3fb", fontSize: 14,
              fontFamily: "Geist, sans-serif", outline: "none",
            }}
          />
        </div>
        <button
          type="submit"
          disabled={isFetching}
          style={{
            padding: "10px 20px", background: "rgba(77,159,255,0.1)", border: "1px solid rgba(77,159,255,0.35)",
            color: "#4d9fff", borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: isFetching ? "default" : "pointer",
            display: "flex", alignItems: "center", gap: 6, opacity: isFetching ? 0.6 : 1,
          }}
        >
          {isFetching ? <RefreshCw size={14} className="spin" /> : <Search size={14} />}
          {isFetching ? "Analyzing…" : "Research"}
        </button>
      </form>

      {/* Empty state */}
      {!ticker && !isFetching && (
        <div style={{ textAlign: "center", padding: "48px 0", color: "#5a6b82" }}>
          <Scale size={40} style={{ opacity: 0.3, marginBottom: 12 }} />
          <p style={{ fontSize: 14 }}>Enter a ticker to generate an explainable research verdict.</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ background: "rgba(255,69,58,0.06)", border: "1px solid rgba(255,69,58,0.3)", borderRadius: 8, padding: 16, color: "#ff6b5a", fontSize: 13 }}>
          {(error as Error).message}
        </div>
      )}

      {/* Report */}
      {data && !error && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Special-situation / catalyst banner */}
          {data.catalyst && data.catalyst.catalyst.kind === "pending_acquisition" && (
            <div style={{ background: "rgba(245,166,35,0.08)", border: "1px solid rgba(245,166,35,0.45)", borderRadius: 10, padding: "16px 20px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#f5a623", fontWeight: 700, fontSize: 13, letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: 8 }}>
                <Megaphone size={16} /> Special situation — pending acquisition
              </div>
              <p style={{ color: "#eef3fb", fontSize: 13.5, margin: "0 0 10px", lineHeight: 1.55 }}>{data.catalyst.assessment}</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 18, fontFamily: "Geist Mono, monospace", fontSize: 12.5 }}>
                {data.catalyst.catalyst.deal_price != null && (
                  <span style={{ color: "#b3c2d8" }}>deal <b style={{ color: "#eef3fb" }}>${data.catalyst.catalyst.deal_price.toFixed(2)}</b></span>
                )}
                {data.catalyst.upside_pct != null && (
                  <span style={{ color: "#30d158" }}>upside {(data.catalyst.upside_pct * 100).toFixed(0)}%</span>
                )}
                {data.catalyst.downside_pct != null && (
                  <span style={{ color: "#ff6b5a" }}>downside {(data.catalyst.downside_pct * 100).toFixed(0)}%</span>
                )}
                {data.catalyst.implied_close_prob != null && (
                  <span style={{ color: "#b3c2d8" }}>market-implied close odds <b style={{ color: "#f5a623" }}>~{(data.catalyst.implied_close_prob * 100).toFixed(0)}%</b></span>
                )}
              </div>
              {data.catalyst.catalyst.url && (
                <a href={data.catalyst.catalyst.url} target="_blank" rel="noopener noreferrer" style={{ color: "#4d9fff", fontSize: 11.5, marginTop: 8, display: "inline-block" }}>source ↗</a>
              )}
            </div>
          )}

          {/* Verdict banner */}
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 20, background: "rgba(77,159,255,0.05)", border: "1px solid rgba(77,159,255,0.2)", borderRadius: 10, padding: "18px 22px" }}>
            <div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                <span style={{ color: "#eef3fb", fontSize: 26, fontWeight: 800, fontFamily: "Geist Mono, monospace" }}>{data.ticker}</span>
                <span style={{ color: "#b3c2d8", fontSize: 15, fontFamily: "Geist Mono, monospace" }}>${data.price?.toFixed(2)}</span>
              </div>
              <span style={{ color: "#6b7c93", fontSize: 11 }}>as of {data.as_of}</span>
            </div>
            <div style={{ flex: 1 }} />
            <div style={{ textAlign: "right" }}>
              <div style={{ color: verdictColor(data.verdict), fontSize: 24, fontWeight: 800 }}>{data.verdict}</div>
              <div style={{ color: "#8a9bb3", fontSize: 12, fontFamily: "Geist Mono, monospace" }}>
                conviction {data.conviction?.toFixed(1)}/10 · composite {data.composite_score?.toFixed(0)}/100
              </div>
            </div>
          </div>

          {/* Thesis */}
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8, padding: "14px 16px" }}>
            <p style={{ color: "#dbe4f0", fontSize: 13.5, margin: 0, lineHeight: 1.55 }}>{data.thesis}</p>
          </div>

          {/* Recent news */}
          {data.catalyst && data.catalyst.news && data.catalyst.news.length > 0 && (
            <div>
              <SectionTitle icon={<Newspaper size={15} />} text="Recent news" />
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {data.catalyst.news.slice(0, 5).map((n, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, fontSize: 12.5, color: "#b3c2d8" }}>
                    {n.date && <span style={{ color: "#6b7c93", fontFamily: "Geist Mono, monospace", whiteSpace: "nowrap" }}>{n.date}</span>}
                    {n.url
                      ? <a href={n.url} target="_blank" rel="noopener noreferrer" style={{ color: "#cdd8e8", textDecoration: "none" }}>{n.headline}</a>
                      : <span>{n.headline}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Pillars */}
          <div>
            <SectionTitle icon={<Layers size={15} />} text="Pillars" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
              {data.pillars.map((p) => <PillarCard key={p.name} pillar={p} />)}
            </div>
          </div>

          {/* Options strategy */}
          {data.options && (
            <div>
              <SectionTitle icon={<Target size={15} />} text="Options strategy" />
              <div style={{ background: "rgba(77,159,255,0.04)", border: "1px solid rgba(77,159,255,0.15)", borderRadius: 8, padding: "14px 16px" }}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginBottom: 10, color: "#8a9bb3", fontSize: 11.5, fontFamily: "Geist Mono, monospace" }}>
                  {data.options.snapshot.iv_rank != null && <span>IV rank {data.options.snapshot.iv_rank.toFixed(0)}</span>}
                  {data.options.snapshot.expected_move_30d != null && <span>~{(data.options.snapshot.expected_move_30d * 100).toFixed(1)}% 30d expected move</span>}
                  {data.options.snapshot.hv_30d != null && <span>HV {(data.options.snapshot.hv_30d * 100).toFixed(0)}%</span>}
                </div>
                <div style={{ color: "#eef3fb", fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
                  {data.options.strategy.structure}
                  <span style={{ color: "#6b7c93", fontWeight: 400, fontSize: 12, marginLeft: 8 }}>({data.options.strategy.direction})</span>
                </div>
                <p style={{ color: "#cdd8e8", fontSize: 13, margin: "0 0 8px", lineHeight: 1.5 }}>{data.options.strategy.rationale}</p>
                {data.options.strategy.legs.length > 0 && (
                  <ul style={{ margin: "0 0 8px", paddingLeft: 18, color: "#b3c2d8", fontSize: 12.5, lineHeight: 1.6, fontFamily: "Geist Mono, monospace" }}>
                    {data.options.strategy.legs.map((leg, i) => <li key={i}>{leg}</li>)}
                  </ul>
                )}
                {data.options.strategy.max_risk_note && (
                  <p style={{ color: "#6b7c93", fontSize: 11, margin: 0 }}>{data.options.strategy.max_risk_note}</p>
                )}
              </div>
            </div>
          )}

          {/* Filings + reads */}
          <div>
            <SectionTitle icon={<FileText size={15} />} text="Filings & context" />
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <ReadRow label="Filings" value={data.filing_read?.summary} sub={`sentiment: ${data.filing_read?.sentiment} · source: ${data.filing_read?.source}`} />
              {data.filing_read?.catalysts?.length > 0 && (
                <ReadRow label="Catalysts" value={data.filing_read.catalysts.join("; ")} />
              )}
              {data.filing_read?.red_flags?.length > 0 && (
                <ReadRow label="Red flags" value={data.filing_read.red_flags.join("; ")} accent="#ff6b5a" />
              )}
              <ReadRow label="Supply/Demand" value={data.supply_demand_read} />
              <ReadRow label="Market" value={data.market_note} />
            </div>
          </div>

          {/* Risks */}
          {data.risks?.length > 0 && (
            <div>
              <SectionTitle icon={<ShieldAlert size={15} />} text="Risks" />
              <ul style={{ margin: 0, paddingLeft: 20, color: "#b3c2d8", fontSize: 13, lineHeight: 1.7 }}>
                {data.risks.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}

          {/* Footer meta + disclaimer */}
          <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 12 }}>
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap", color: "#6b7c93", fontSize: 11, fontFamily: "Geist Mono, monospace", marginBottom: 8 }}>
              <span>data source: {data.provider}</span>
              <span>completeness: {(data.data_completeness * 100).toFixed(0)}%</span>
            </div>
            <p style={{ color: "#5a6b82", fontSize: 11, margin: 0, lineHeight: 1.5 }}>{data.disclaimer}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function SectionTitle({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7, color: "#8a9bb3", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 10 }}>
      {icon}
      {text}
    </div>
  );
}

function ReadRow({ label, value, sub, accent }: { label: string; value?: string; sub?: string; accent?: string }) {
  if (!value) return null;
  return (
    <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6, padding: "10px 14px" }}>
      <div style={{ color: "#8a9bb3", fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: 3 }}>{label}</div>
      <div style={{ color: accent || "#cdd8e8", fontSize: 13, lineHeight: 1.5 }}>{value}</div>
      {sub && <div style={{ color: "#6b7c93", fontSize: 11, marginTop: 3, fontFamily: "Geist Mono, monospace" }}>{sub}</div>}
    </div>
  );
}
