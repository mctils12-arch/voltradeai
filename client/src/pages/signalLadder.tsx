// SignalLadderView — MAP V2 ROADMAP R6(a) SIGNAL-STRENGTH dashboard
// (#/data/signals). Charter directive 2026-07-04: "ladder position of
// every root (gate passed/date/next gate), from research/ bookkeeping
// made machine-readable." Reads datacore/signal_ladder.json (a
// hand-compiled, provenance-carrying snapshot — see its own _doc) via
// /api/data/signal-ladder: a funnel of how many roots reached each gate,
// then every root grouped by category with its status, note, and
// source_ref so a visitor can trace the claim back to the record.
// PREMIUM EXPERIENCE STANDARD (c): every number here carries provenance —
// this is the platform's own honesty ledger, made visible.
// Mobile-first (390px primary). Reuses the .vt-filings/.vt-streams-body
// shell + .vt-quality-* tokens, same pattern as qualityDashboard.tsx.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, RefreshCw, TrendingUp } from "lucide-react";

interface LadderRoot {
  id: string;
  name: string;
  category: string;
  status: string;
  current_gate: number;
  last_update_date: string;
  note: string;
  source_ref: string;
  detail_route?: string;
}
interface LadderSummary {
  total: number;
  by_status: Record<string, number>;
  by_category: Record<string, number>;
  gate_counts: { gate: number; count: number }[];
  killed_count: number;
  raw_only_count: number;
  furthest_gate_reached: number;
}
interface LadderPayload {
  generated_at: string;
  compiled: string;
  sources: string[];
  roots: LadderRoot[];
  summary: LadderSummary;
}

const GATE_LABEL: Record<number, string> = {
  0: "Gate 0 · raw / untested",
  1: "Gate 1 · DATA",
  2: "Gate 2 · SIGNAL",
  3: "Gate 3 · LOGIC",
  4: "Gate 4 · SIZING",
  5: "Gate 5 · EXECUTION",
};

const STATUS_META: Record<string, { label: string; cls: string }> = {
  raw_only: { label: "raw overlay", cls: "raw" },
  gate1_pending: { label: "gate 1 pending", cls: "pending" },
  gate1_pass: { label: "gate 1 passed", cls: "pass" },
  gate1_fail: { label: "gate 1 FAILED", cls: "fail" },
  gate2_pending: { label: "gate 2 pending", cls: "pending" },
  gate2_pass: { label: "gate 2 passed", cls: "pass" },
  gate2_fail: { label: "gate 2 FAILED", cls: "fail" },
  gate3_pass: { label: "gate 3 passed", cls: "pass" },
  gate4_pass: { label: "gate 4 passed", cls: "pass" },
  gate5_pass: { label: "gate 5 passed — traded", cls: "pass" },
  killed: { label: "killed", cls: "killed" },
};

const CATEGORY_LABEL: Record<string, string> = {
  power_grid: "Power grid",
  maritime: "Maritime",
  insider_trading: "Insider / institutional",
  options_flow: "Options & market structure",
  space_weather: "Space weather",
  environmental: "Environmental",
  government_spending: "Government spending",
  labor: "Labor",
  commodities: "Commodities",
  macro: "Macro",
  credit_markets: "Credit markets",
  other: "Other",
};

function statusMeta(status: string) {
  return STATUS_META[status] || { label: status, cls: "pending" };
}

export default function SignalLadderView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<LadderPayload | null>(null);
  const [error, setError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [activeCategory, setActiveCategory] = useState<string | null>(null);

  const load = async () => {
    setRefreshing(true);
    try {
      const r = await fetch("/api/data/signal-ladder");
      // crash-guard (repair 2026-08-06): a JSON-bodied 5xx parsed fine and
      // crashed downstream on missing fields — route errors to the error state.
      if (!r.ok) throw new Error(String(r.status));
      setData(await r.json());
      setError(false);
    } catch {
      setError(true);
    } finally {
      setRefreshing(false);
    }
  };
  useEffect(() => { void load(); }, []);

  const maxGateCount = useMemo(() => {
    if (!data) return 1;
    return Math.max(1, ...data.summary.gate_counts.map((g) => g.count));
  }, [data]);

  const categories = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.summary.by_category)
      .sort((a, b) => b[1] - a[1])
      .map(([cat, count]) => ({ cat, count, label: CATEGORY_LABEL[cat] || cat }));
  }, [data]);

  const visibleRoots = useMemo(() => {
    if (!data) return [];
    const roots = activeCategory ? data.roots.filter((r) => r.category === activeCategory) : data.roots;
    return [...roots].sort((a, b) => b.current_gate - a.current_gate || a.name.localeCompare(b.name));
  }, [data, activeCategory]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Signal strength dashboard">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <TrendingUp size={16} />
        <div>
          <div className="vt-filings-title">Signal strength</div>
          <div className="vt-filings-sub">
            {data ? "every data root's ladder position, compiled from the research log — not every root is a validated signal, and that's the point"
              : error ? "dashboard unavailable — retry below" : "loading…"}
          </div>
        </div>
        <button className="vt-icon-btn" aria-label="Refresh dashboard" onClick={() => void load()}
                style={{ marginLeft: "auto" }} disabled={refreshing}>
          <RefreshCw size={15} className={refreshing ? "vt-spin" : undefined} />
        </button>
      </div>

      <div className="vt-streams-body om-sb">
        {error && !data && (
          <div className="vt-filings-state">
            Could not load the dashboard. <button type="button" className="vt-graph-example" onClick={() => void load()}>Retry</button>
          </div>
        )}

        {data && (
          <>
            <div className="vt-filings-sub" style={{ padding: "0 0 4px" }}>
              Snapshot compiled {data.compiled} from research/experiments.md + research/open_questions.md — a
              point-in-time read of the ladder log, not a live-updating feed. Every entry below carries a
              source_ref so its claim can be re-checked.
            </div>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>Ladder funnel</span>
                <span className="vt-quality-section-sub">
                  {data.summary.total} roots tracked · {data.summary.raw_only_count} raw overlays (ladder N/A) ·
                  {" "}{data.summary.killed_count} killed · furthest a signal has reached: gate {data.summary.furthest_gate_reached}
                </span>
              </div>
              <div className="vt-ladder-funnel">
                {data.summary.gate_counts.map((g) => (
                  <div key={g.gate} className="vt-ladder-funnel-row">
                    <span className="vt-ladder-funnel-label">{GATE_LABEL[g.gate]}</span>
                    <span className="vt-quality-kind-bar-track" style={{ flex: 1 }}>
                      <span className="vt-quality-kind-bar-fill" style={{ width: `${Math.max(2, (g.count / maxGateCount) * 100)}%` }} />
                    </span>
                    <span className="vt-ladder-funnel-count">{g.count}</span>
                  </div>
                ))}
              </div>
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>By category</span>
                <span className="vt-quality-section-sub">tap to filter the list below</span>
              </div>
              <div className="vt-ladder-cat-chips">
                <button type="button" className={`vt-ladder-cat-chip${activeCategory === null ? " vt-ladder-cat-chip-on" : ""}`}
                        onClick={() => setActiveCategory(null)}>
                  All ({data.summary.total})
                </button>
                {categories.map((c) => (
                  <button key={c.cat} type="button"
                          className={`vt-ladder-cat-chip${activeCategory === c.cat ? " vt-ladder-cat-chip-on" : ""}`}
                          onClick={() => setActiveCategory(c.cat === activeCategory ? null : c.cat)}>
                    {c.label} ({c.count})
                  </button>
                ))}
              </div>
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>Roots</span>
                <span className="vt-quality-section-sub">{visibleRoots.length} shown, furthest gate first</span>
              </div>
              <div className="vt-ladder-list">
                {visibleRoots.map((r) => {
                  const meta = statusMeta(r.status);
                  return (
                    <div key={r.id} className="vt-ladder-row" data-vt-ladder-root={r.id}>
                      <div className="vt-ladder-row-top">
                        <span className="vt-ladder-row-name">{r.name}</span>
                        <span className={`vt-ladder-badge vt-ladder-badge-${meta.cls}`}>{meta.label}</span>
                      </div>
                      <div className="vt-ladder-row-note">{r.note}</div>
                      <div className="vt-ladder-row-foot">
                        <span>{CATEGORY_LABEL[r.category] || r.category}</span>
                        <span>·</span>
                        <span>updated {r.last_update_date}</span>
                        <span>·</span>
                        <span className="vt-ladder-row-src" title={r.source_ref}>source: {r.source_ref}</span>
                        {r.detail_route && (
                          <>
                            <span>·</span>
                            <a href={r.detail_route} className="vt-graph-example">view live signal →</a>
                          </>
                        )}
                      </div>
                    </div>
                  );
                })}
                {visibleRoots.length === 0 && (
                  <div className="vt-filings-state">No roots in this category.</div>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
