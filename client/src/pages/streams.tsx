// StreamsView — the streams inventory (#/data/streams). DATACORE MAXIMUS
// Phase 4: every archived stream visible in one place — source, hypothesis,
// recording-since, count, freshness, health, latest-record peek.
// Mobile-first (390px is the primary viewport): stacked cards, no tables.
// Coverage is mechanical: the list renders /api/data/streams verbatim, which
// enumerates datacore/manifests/ at runtime — no hand-picked stream list
// exists anywhere in the client (inventory-coverage ratchet, Phase 5).
// DESIGN.md-governed: reuses the .vt-filings-* shell, theme tokens only,
// designed loading/warming/error/empty states.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Database, RefreshCw, Search } from "lucide-react";

interface StreamRow {
  stream: string;
  source: string;
  attribution: string;
  license: string;
  started: string;
  cadence: string;
  hypothesis: string;
  storage: string;
  written_by: string;
  files: number;
  bytes: number;
  newest_file: string | null;
  age_hours: number | null;
  records_newest_file: number | null;
  latest_peek: string | null;
  peek_note: string | null;
  health: "live" | "recent" | "stale" | "no-data";
  health_note: string;
}
interface Inventory { time?: number; count: number; streams: StreamRow[]; warming_up?: boolean; }

const HEALTH_LABEL: Record<StreamRow["health"], string> = {
  live: "live", recent: "recent", stale: "stale", "no-data": "no data",
};
const HEALTH_COLOR: Record<StreamRow["health"], string> = {
  live: "var(--accent-green)",
  recent: "var(--accent-blue, #4d9fff)",
  stale: "var(--accent-orange)",
  "no-data": "var(--text-tertiary)",
};

function fmtAge(h: number | null): string {
  if (h === null) return "never";
  if (h < 1) return `${Math.round(h * 60)}m ago`;
  if (h < 48) return `${h.toFixed(1)}h ago`;
  return `${(h / 24).toFixed(1)}d ago`;
}
function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(0)} KB`;
  return `${b} B`;
}

const FILTERS: Array<"all" | StreamRow["health"]> = ["all", "live", "recent", "stale", "no-data"];

export default function StreamsView({ onBack }: { onBack: () => void }) {
  const [inv, setInv] = useState<Inventory | null>(null);
  const [error, setError] = useState(false);
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all");
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [refreshing, setRefreshing] = useState(false);

  const load = async () => {
    setRefreshing(true);
    try {
      const r = await fetch("/api/data/streams");
      setInv(await r.json());
      setError(false);
    } catch {
      setError(true);
    } finally {
      setRefreshing(false);
    }
  };
  useEffect(() => { void load(); }, []);

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: inv?.streams.length || 0 };
    for (const s of inv?.streams || []) c[s.health] = (c[s.health] || 0) + 1;
    return c;
  }, [inv]);

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (inv?.streams || [])
      .filter((s) => filter === "all" || s.health === filter)
      .filter((s) => !needle
        || s.stream.includes(needle)
        || s.source.toLowerCase().includes(needle)
        || s.attribution.toLowerCase().includes(needle));
  }, [inv, filter, q]);

  const totalBytes = useMemo(() => (inv?.streams || []).reduce((a, s) => a + s.bytes, 0), [inv]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Streams inventory">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Database size={16} />
        <div>
          <div className="vt-filings-title">Streams inventory</div>
          <div className="vt-filings-sub">
            {inv && !inv.warming_up
              ? `${inv.count} archived streams · ${fmtBytes(totalBytes)} on the volume · every manifested stream listed — nothing hand-picked`
              : inv?.warming_up ? "first archive scan in progress — retry shortly"
              : error ? "inventory unavailable — retry below" : "loading…"}
          </div>
        </div>
        <button className="vt-icon-btn" aria-label="Refresh inventory" onClick={() => void load()}
                style={{ marginLeft: "auto" }} disabled={refreshing}>
          <RefreshCw size={15} className={refreshing ? "vt-spin" : undefined} />
        </button>
      </div>

      <div className="vt-streams-body">
        <div className="vt-filings-filters" role="tablist" aria-label="Filter by health">
          {FILTERS.map((f) => (
            <button key={f} type="button" role="tab" aria-selected={filter === f}
                    className={`vt-filings-filter${filter === f ? " active" : ""}`}
                    onClick={() => setFilter(f)}>
              {f === "all" ? "all" : HEALTH_LABEL[f]} {counts[f] ? `(${counts[f]})` : "(0)"}
            </button>
          ))}
          <label className="vt-streams-search">
            <Search size={13} aria-hidden />
            <input type="search" className="vt-earnings-search" placeholder="Filter streams…"
                   aria-label="Filter streams by name or source"
                   value={q} onChange={(e) => setQ(e.target.value)} />
          </label>
        </div>

        {error && !inv && (
          <div className="vt-filings-state">
            Could not load the inventory. <button type="button" className="vt-graph-example" onClick={() => void load()}>Retry</button>
          </div>
        )}
        {inv?.warming_up && (
          <div className="vt-filings-state">First archive scan is running on the server — refresh in a few seconds.</div>
        )}
        {inv && !inv.warming_up && shown.length === 0 && (
          <div className="vt-filings-state">No streams match this filter.</div>
        )}

        <div className="vt-streams-list">
          {shown.map((s) => (
            <div key={s.stream} className="vt-streams-card" data-health={s.health}>
              <button type="button" className="vt-streams-cardhead" aria-expanded={!!open[s.stream]}
                      onClick={() => setOpen((o) => ({ ...o, [s.stream]: !o[s.stream] }))}>
                <span className="vt-streams-name">{s.stream}</span>
                <span className="vt-streams-chip" style={{ color: HEALTH_COLOR[s.health], borderColor: HEALTH_COLOR[s.health] }}
                      title={s.health_note}>
                  {HEALTH_LABEL[s.health]}
                </span>
                <span className="vt-streams-age">{fmtAge(s.age_hours)}</span>
              </button>
              <div className="vt-streams-meta">
                <span>{s.files} file{s.files === 1 ? "" : "s"} · {fmtBytes(s.bytes)}</span>
                {s.records_newest_file !== null && <span>{s.records_newest_file.toLocaleString()} records in newest file</span>}
                <span>since {s.started || "?"}</span>
              </div>
              <div className="vt-streams-attr">{s.attribution}</div>
              {open[s.stream] && (
                <div className="vt-streams-detail">
                  <div><b>Source.</b> {s.source}</div>
                  <div><b>Hypothesis.</b> {s.hypothesis}</div>
                  <div><b>Cadence.</b> {s.cadence}</div>
                  <div><b>Storage.</b> {s.storage}</div>
                  <div><b>Writer.</b> {s.written_by}</div>
                  <div><b>License.</b> {s.license}</div>
                  <div><b>Health.</b> {s.health_note}</div>
                  {s.latest_peek ? (
                    <div className="vt-streams-peek">
                      <div className="vt-streams-peeklabel">Latest record ({s.newest_file})</div>
                      <pre>{s.latest_peek}</pre>
                    </div>
                  ) : (
                    <div><b>Peek.</b> {s.peek_note || "unavailable"}</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {inv && !inv.warming_up && (
          <div className="vt-filings-sub vt-streams-foot">
            Health is derived from newest-file age vs. each stream's cadence (raw age always shown).
            "no data" = key-gated, warming up, or a session-run pipeline that hasn't landed on this
            volume. Record counts shown are for the newest file; files/bytes cover the full archive.
          </div>
        )}
      </div>
    </div>
  );
}
