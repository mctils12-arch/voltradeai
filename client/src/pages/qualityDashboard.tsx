// QualityDashboardView — MAP V2 ROADMAP R6(b) DATA-QUALITY dashboard
// (#/data/quality). Charter directive 2026-07-04: "dashboards from
// monitoring we already emit, no new collection." Three honest readings,
// each already computed elsewhere and simply surfaced here in one place:
// (1) feed health across every archived stream, (2) archive growth by
// kind, (3) imagery-verification coverage for sites/ports/plants — the
// PREMIUM EXPERIENCE STANDARD's "every number visibly carries freshness,
// provenance, and confidence" clause applied at the platform level rather
// than the per-layer level (that's streams.tsx / layerFreshness.ts).
// Mobile-first (390px primary). DESIGN.md-governed: reuses the
// .vt-filings-* shell + theme tokens only.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ShieldCheck, RefreshCw, Database } from "lucide-react";

interface StreamHealthSummary {
  total: number;
  live: number;
  recent: number;
  stale: number;
  no_data: number;
}
interface ArchiveKindSummary { kind: string; files: number; bytes: number }
interface ArchiveSummary { total_bytes: number; total_files: number; kinds: ArchiveKindSummary[] }
interface VerificationCategory { verified: number; total: number; method: string }
interface VerificationCoverage { sites: VerificationCategory; ports: VerificationCategory; plants: VerificationCategory }
interface QualityPayload {
  generated_at: string;
  streams_warming_up: boolean;
  streams: StreamHealthSummary;
  archive: ArchiveSummary;
  verification: VerificationCoverage;
}

const HEALTH_COLOR: Record<"live" | "recent" | "stale" | "no_data", string> = {
  live: "var(--accent-green)",
  recent: "var(--accent-blue, #4d9fff)",
  stale: "var(--accent-orange)",
  no_data: "var(--text-tertiary)",
};
const HEALTH_LABEL: Record<"live" | "recent" | "stale" | "no_data", string> = {
  live: "live", recent: "recent", stale: "stale", no_data: "no data",
};

function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(0)} KB`;
  return `${b} B`;
}
function pct(verified: number, total: number): number {
  return total > 0 ? Math.round((verified / total) * 1000) / 10 : 0;
}

export default function QualityDashboardView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<QualityPayload | null>(null);
  const [error, setError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const load = async () => {
    setRefreshing(true);
    try {
      const r = await fetch("/api/data/quality-dashboard");
      setData(await r.json());
      setError(false);
    } catch {
      setError(true);
    } finally {
      setRefreshing(false);
    }
  };
  useEffect(() => { void load(); }, []);

  const healthRows = useMemo(() => {
    if (!data) return [];
    const s = data.streams;
    return (["live", "recent", "stale", "no_data"] as const).map((k) => ({
      key: k, label: HEALTH_LABEL[k], color: HEALTH_COLOR[k],
      count: s[k], pct: s.total > 0 ? Math.round((s[k] / s.total) * 100) : 0,
    }));
  }, [data]);

  const verifTiles = useMemo(() => {
    if (!data) return [];
    const v = data.verification;
    return [
      { name: "Strategic sites", cat: v.sites },
      { name: "Port geofences", cat: v.ports },
      { name: "US power plants", cat: v.plants },
    ];
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Data quality dashboard">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <ShieldCheck size={16} />
        <div>
          <div className="vt-filings-title">Data quality</div>
          <div className="vt-filings-sub">
            {data ? "feed health, archive growth, and imagery-verification coverage — all from monitoring we already run"
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
            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>Feed health</span>
                <span className="vt-quality-section-sub">
                  {data.streams_warming_up
                    ? "streams inventory still warming up — first archive scan in progress"
                    : `${data.streams.total} archived streams`}
                </span>
              </div>
              <div className="vt-quality-grid">
                {healthRows.map((h) => (
                  <div key={h.key} className="vt-quality-tile" data-vt-quality-health={h.key}>
                    <div className="vt-quality-tile-value" style={{ color: h.color }}>{h.count}</div>
                    <div className="vt-quality-tile-label">{h.label}</div>
                    <div className="vt-quality-bar-track">
                      <div className="vt-quality-bar-fill" style={{ width: `${h.pct}%`, background: h.color }} />
                    </div>
                  </div>
                ))}
              </div>
              <div className="vt-filings-sub vt-streams-foot">
                Health is derived from newest-file age vs. each stream's cadence — see the full{" "}
                <a href="#/data/streams" onClick={(e) => { e.preventDefault(); window.location.hash = "#/data/streams"; }}>
                  streams inventory
                </a> for a per-stream breakdown.
              </div>
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>Verification coverage</span>
                <span className="vt-quality-section-sub">position accuracy, checked against imagery — not every number is 100%, and that's stated honestly</span>
              </div>
              <div className="vt-quality-grid">
                {verifTiles.map((t) => (
                  <div key={t.name} className="vt-quality-tile" title={t.cat.method}>
                    <div className="vt-quality-tile-value">
                      {t.cat.verified.toLocaleString()}<span className="vt-quality-tile-of">/{t.cat.total.toLocaleString()}</span>
                    </div>
                    <div className="vt-quality-tile-label">{t.name}</div>
                    <div className="vt-quality-bar-track">
                      <div className="vt-quality-bar-fill" style={{ width: `${pct(t.cat.verified, t.cat.total)}%`, background: "var(--accent)" }} />
                    </div>
                    <div className="vt-quality-tile-pct">{pct(t.cat.verified, t.cat.total)}% verified</div>
                  </div>
                ))}
              </div>
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>Archive growth</span>
                <span className="vt-quality-section-sub">
                  <Database size={12} style={{ verticalAlign: -2, marginRight: 3 }} />
                  {fmtBytes(data.archive.total_bytes)} across {data.archive.total_files.toLocaleString()} files on the volume
                </span>
              </div>
              <div className="vt-quality-kinds">
                {data.archive.kinds.slice(0, 12).map((k) => {
                  const maxBytes = data.archive.kinds[0]?.bytes || 1;
                  return (
                    <div key={k.kind} className="vt-quality-kind-row">
                      <span className="vt-quality-kind-name">{k.kind}</span>
                      <span className="vt-quality-kind-bar-track">
                        <span className="vt-quality-kind-bar-fill" style={{ width: `${Math.max(2, (k.bytes / maxBytes) * 100)}%` }} />
                      </span>
                      <span className="vt-quality-kind-bytes">{fmtBytes(k.bytes)}</span>
                    </div>
                  );
                })}
                {data.archive.kinds.length > 12 && (
                  <div className="vt-filings-sub">+ {data.archive.kinds.length - 12} smaller archive kinds not shown</div>
                )}
                {data.archive.kinds.length === 0 && (
                  <div className="vt-filings-state">No archive activity recorded yet.</div>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
