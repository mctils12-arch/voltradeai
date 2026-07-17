// GraphView — the Everything Graph panel (#/data/graph).
// DESIGN.md-governed: readable at 390/768/1440, designed empty/loading/error
// states, theme tokens only, reuses the .vt-filings-* shell (same pattern as
// filings.tsx/shortvol.tsx). Data: /api/data/graph (counts-only summary) and
// /api/data/graph?entity=<ticker|MMSI|CIK|facility id>&hops=1..3 (BFS
// neighborhood). RAW join over the Form4/entity_map/AIS-derived archives
// (datacore/EVERYTHING_GRAPH.md) — every edge carries
// {source, confidence, first_seen, last_seen}; no predictive claim.
//
// Scope note: the connections list below always shows DIRECT edges touching
// the searched entity (1 hop), regardless of the hops selector — showing
// edges further out as if they were direct would misrepresent the join.
// The hops selector instead controls how many entities/edges the reachable-
// network count includes, stated explicitly so it's never misleading.
import { useEffect, useState } from "react";
import { ArrowLeft, Search, Share2, Building2, User, Factory, Ship } from "lucide-react";

type NodeType = "company" | "person" | "facility" | "vessel";
type EdgeType = "insider_of" | "operates" | "calls_at" | "owns";

interface GNode { id: string; type: NodeType; label: string; attrs: Record<string, any>; }
interface GEdge {
  type: EdgeType; from: string; to: string; source: string;
  confidence: "high" | "medium" | "low" | "unverified";
  first_seen: string | null; last_seen: string | null; attrs: Record<string, any>;
}
interface Counts {
  nodes: number; edges: number;
  company: number; person: number; facility: number; vessel: number;
  insider_of: number; operates: number; calls_at: number; owns: number;
}
interface Summary { kind: string; counts?: Counts; caveat?: string; warming_up?: boolean; }
interface Neighborhood {
  kind: string; entity?: string; hops?: number; caveat?: string;
  nodes?: GNode[]; edges?: GEdge[]; error?: string;
}

const TYPE_ICON: Record<NodeType, any> = { company: Building2, person: User, facility: Factory, vessel: Ship };
const TYPE_LABEL: Record<NodeType, string> = { company: "Company", person: "Person", facility: "Facility", vessel: "Vessel" };
const REL_LABEL: Record<EdgeType, { fwd: string; rev: string }> = {
  insider_of: { fwd: "insider of", rev: "has insider" },
  operates: { fwd: "operates", rev: "operated by" },
  calls_at: { fwd: "calls at", rev: "visited by" },
  owns: { fwd: "owns", rev: "owned by" },
};
const CONF_COLOR: Record<string, string> = {
  high: "var(--accent-green)", medium: "var(--accent-orange)",
  low: "var(--text-tertiary)", unverified: "var(--text-tertiary)",
};

function facilitySub(attrs: Record<string, any>): string {
  const bits = [attrs.kind, attrs.category, attrs.fuel].filter(Boolean);
  const op = attrs.operator || attrs.owner;
  if (op) bits.push(`operator: ${op}`);
  return bits.join(" · ");
}

export default function GraphView({ onBack }: { onBack: () => void }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [summaryError, setSummaryError] = useState(false);
  const [query, setQuery] = useState("");
  const [hops, setHops] = useState(1);
  const [result, setResult] = useState<Neighborhood | null>(null);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/graph");
        const d = await r.json();
        if (!stop) setSummary(d);
      } catch {
        if (!stop) setSummaryError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const runSearch = async (entity: string, h = hops) => {
    const term = entity.trim();
    if (!term) return;
    setLoading(true);
    setSearched(true);
    try {
      const r = await fetch(`/api/data/graph?entity=${encodeURIComponent(term)}&hops=${h}`);
      const d = await r.json();
      setResult(d);
    } catch {
      setResult({ kind: "raw", error: "lookup failed — try again" });
    } finally {
      setLoading(false);
    }
  };

  const submit = (e?: React.FormEvent) => { e?.preventDefault(); runSearch(query); };
  const jumpTo = (nodeId: string, label: string) => { setQuery(label); runSearch(nodeId); };

  const nodesById = new Map((result?.nodes || []).map((n) => [n.id, n]));
  const centerId = result?.entity;
  const center = centerId ? nodesById.get(centerId) : null;
  const direct = (result?.edges || [])
    .filter((e) => e.from === centerId || e.to === centerId)
    .map((edge) => {
      const isFwd = edge.from === centerId;
      const otherId = isFwd ? edge.to : edge.from;
      return { edge, other: nodesById.get(otherId), otherId, isFwd };
    })
    .filter((c): c is typeof c & { other: GNode } => !!c.other);

  return (
    <div className="vt-filings-page" role="region" aria-label="Everything Graph">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Share2 size={16} />
        <div>
          <div className="vt-filings-title">Everything Graph</div>
          <div className="vt-filings-sub">
            RAW joins across insiders, facilities &amp; vessels — every edge carries source + confidence, no predictive claim ·{" "}
            {summary?.counts
              ? `${summary.counts.nodes.toLocaleString()} entities, ${summary.counts.edges.toLocaleString()} connections`
              : summary?.warming_up ? "first build in progress — retry shortly"
              : summaryError ? "summary unavailable" : "loading…"}
          </div>
        </div>
      </div>

      <div className="vt-graph-body">
        {summary?.counts && (
          <div className="vt-graph-counts" role="group" aria-label="Graph entity counts">
            <span><Building2 size={12} /> {summary.counts.company.toLocaleString()} companies</span>
            <span><User size={12} /> {summary.counts.person.toLocaleString()} people</span>
            <span><Factory size={12} /> {summary.counts.facility.toLocaleString()} facilities</span>
            <span><Ship size={12} /> {summary.counts.vessel.toLocaleString()} vessels</span>
          </div>
        )}

        <form className="vt-filings-filters" onSubmit={submit}>
          <input
            type="search" className="vt-earnings-search"
            placeholder="Search a ticker, MMSI, CIK, or facility id (e.g. STLD)…"
            aria-label="Search the Everything Graph"
            value={query} onChange={(e) => setQuery(e.target.value)}
          />
          {[1, 2, 3].map((h) => (
            <button key={h} type="button" role="tab" aria-selected={hops === h}
                    className={`vt-filings-filter${hops === h ? " active" : ""}`}
                    onClick={() => { setHops(h); if (result?.entity) runSearch(result.entity, h); }}>
              {h} hop{h > 1 ? "s" : ""}
            </button>
          ))}
          <button type="submit" className="vt-filings-filter" disabled={loading || !query.trim()}>
            <Search size={13} /> {loading ? "Searching…" : "Search"}
          </button>
        </form>

        {!searched && (
          <div className="vt-filings-state">
            Search a company ticker, insider CIK, vessel MMSI, or facility id to see everything the
            platform has joined about it. Try{" "}
            <button type="button" className="vt-graph-example" onClick={() => { setQuery("STLD"); runSearch("STLD"); }}>
              STLD
            </button>.
          </div>
        )}
        {searched && loading && <div className="vt-filings-state">Searching…</div>}
        {searched && !loading && result?.error && (
          <div className="vt-filings-state">{result.error} — try a company ticker, MMSI, CIK, or facility id.</div>
        )}
        {searched && !loading && !result?.error && center && (
          <div className="vt-graph-result">
            <div className="vt-graph-center">
              <span className="vt-graph-typebadge" data-type={center.type}>{TYPE_LABEL[center.type]}</span>
              <span className="vt-graph-centerlabel">{center.label}</span>
              {center.type === "facility" && <span className="vt-filings-sub">{facilitySub(center.attrs)}</span>}
              {center.type === "vessel" && <span className="vt-filings-sub">MMSI {center.attrs?.mmsi}</span>}
              {center.type === "person" && <span className="vt-filings-sub">CIK {center.attrs?.cik}</span>}
              {center.type === "company" && center.attrs?.parent && (
                <span className="vt-filings-sub">parent: {center.attrs.parent}</span>
              )}
            </div>
            <div className="vt-filings-sub">
              {(result!.nodes || []).length.toLocaleString()} entities, {(result!.edges || []).length.toLocaleString()} connections
              reachable within {hops} hop{hops > 1 ? "s" : ""} — direct connections shown below.
            </div>
            {direct.length === 0 && (
              <div className="vt-filings-state">No direct connections in the archive yet.</div>
            )}
            {direct.length > 0 && (
              <div className="vt-graph-connections">
                {direct.map(({ edge, other, otherId, isFwd }, i) => {
                  const Icon = TYPE_ICON[other.type];
                  const rel = REL_LABEL[edge.type];
                  return (
                    <button key={`${otherId}-${i}`} type="button" className="vt-graph-conn-row"
                            onClick={() => jumpTo(otherId, other.label)}>
                      <Icon size={14} />
                      <span className="vt-graph-conn-main">
                        <span className="vt-graph-conn-label">{other.label}</span>
                        <span className="vt-graph-conn-rel">{isFwd ? rel.fwd : rel.rev}</span>
                      </span>
                      <span className="vt-graph-conn-meta">
                        <span className="vt-graph-conf" style={{ color: CONF_COLOR[edge.confidence] }}>{edge.confidence}</span>
                        {edge.attrs?.filing_count != null && <span>{edge.attrs.filing_count} filings</span>}
                        {edge.attrs?.visit_count != null && <span>{edge.attrs.visit_count} calls</span>}
                        {edge.attrs?.share_pct != null && <span>{edge.attrs.share_pct}% owned</span>}
                        {Array.isArray(edge.attrs?.roles) && edge.attrs.roles.length > 0 && (
                          <span>{edge.attrs.roles.join(", ")}</span>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}
        {summary?.caveat && <div className="vt-filings-sub vt-graph-caveat">{summary.caveat}</div>}
      </div>
    </div>
  );
}
