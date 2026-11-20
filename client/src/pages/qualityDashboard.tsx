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
import { firesCheckLines, type FiresCheck } from "@/lib/archiveChecks";
import { archiveRows, fmtBytes, offsiteLines, type ArchiveStats, type OffloadStatus } from "@/lib/archiveView";

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

function pct(verified: number, total: number): number {
  return total > 0 ? Math.round((verified / total) * 1000) / 10 : 0;
}

/** archives listed before "show all": the largest, where the bytes are */
const ARCHIVES_SHOWN = 12;

export default function QualityDashboardView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<QualityPayload | null>(null);
  const [error, setError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  // Archive checks (2026-11-20): an archive compared with its source's own
  // record. Its own request and its own failure: the panel above does not
  // depend on it.
  const [firesCheck, setFiresCheck] = useState<FiresCheck | null>(null);
  const [firesCheckError, setFiresCheckError] = useState(false);
  // Archive files and the off-site copy (UI coverage batch C, second part):
  // two more reads, each allowed to fail without taking the panel down.
  const [stats, setStats] = useState<ArchiveStats | null>(null);
  const [offload, setOffload] = useState<OffloadStatus | null>(null);
  const [archiveError, setArchiveError] = useState({ stats: false, offload: false });
  const [allArchives, setAllArchives] = useState(false);
  const loadArchive = async () => {
    const get = async (url: string) => { const r = await fetch(url); if (!r.ok) throw new Error(String(r.status)); return r.json(); };
    const [st, off] = await Promise.allSettled([get("/api/data/archive/stats"), get("/api/data/archive/offload-status")]);
    if (st.status === "fulfilled") setStats(st.value);
    if (off.status === "fulfilled") setOffload(off.value);
    setArchiveError({ stats: st.status === "rejected", offload: off.status === "rejected" });
  };
  const loadChecks = async () => {
    try {
      const r = await fetch("/api/data/fires/completeness");
      if (!r.ok) throw new Error(String(r.status));
      setFiresCheck(await r.json());
      setFiresCheckError(false);
    } catch {
      setFiresCheckError(true);
    }
  };

  const load = async () => {
    setRefreshing(true);
    try {
      const r = await fetch("/api/data/quality-dashboard");
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
  useEffect(() => { void load(); void loadChecks(); void loadArchive(); }, []);
  const rows = useMemo(() => archiveRows(stats, offload, Date.now()), [stats, offload]);
  const shownRows = allArchives ? rows : rows.slice(0, ARCHIVES_SHOWN);

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
        <button className="vt-icon-btn" aria-label="Refresh dashboard" onClick={() => { void load(); void loadChecks(); void loadArchive(); }}
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

            <section className="vt-quality-section" data-vt-quality-checks>
              <div className="vt-quality-section-head">
                <span>Archive checks</span>
                <span className="vt-quality-section-sub">an archive compared with its source's own record, and what was done about the difference</span>
              </div>
              <div className="vt-quality-check">
                <div className="vt-quality-check-name">Active fires (NASA FIRMS)</div>
                {firesCheckError && !firesCheck && <div className="vt-filings-sub">The check's record did not load. Refresh asks again.</div>}
                {!firesCheckError && !firesCheck && <div className="vt-filings-sub">Loading…</div>}
                {firesCheck && (
                  <ul className="vt-quality-check-lines">
                    {firesCheckLines(firesCheck, Date.now()).map((line) => <li key={line}>{line}</li>)}
                  </ul>
                )}
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

            <section className="vt-quality-section" data-vt-quality-files>
              <div className="vt-quality-section-head">
                <span>Archive files</span>
                <span className="vt-quality-section-sub">
                  every archive on the volume: how many files, the first and the newest, and what is held off-site
                </span>
              </div>
              {archiveError.stats && !stats && <div className="vt-filings-sub">The archive listing did not load. Refresh asks again.</div>}
              {!archiveError.stats && !stats && <div className="vt-filings-sub">Loading…</div>}
              {stats && rows.length === 0 && <div className="vt-filings-sub">No archive holds a file yet.</div>}
              {rows.length > 0 && (
                <>
                  <div className="vt-filings-tablewrap vt-tablewrap-flow">
                    <table className="vt-filings-table">
                      <thead>
                        <tr><th>Archive</th><th className="num">Files</th><th className="num">Size</th><th>First file</th><th>Newest file</th><th>Off-site</th></tr>
                      </thead>
                      <tbody>
                        {shownRows.map((r) => (
                          <tr key={r.kind} data-vt-archive={r.kind}>
                            <td data-l="Archive">{r.kind}</td>
                            <td data-l="Files" className="num">{r.files.toLocaleString("en-US")}</td>
                            <td data-l="Size" className="num">{r.files ? fmtBytes(r.bytes) : "—"}</td>
                            <td data-l="First file">{r.first}</td>
                            <td data-l="Newest file">{r.newest}{r.age ? ` (${r.age})` : ""}</td>
                            <td data-l="Off-site">{r.offsite || (offload ? "—" : archiveError.offload ? "not loaded" : "…")}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {rows.length > ARCHIVES_SHOWN && (
                    <button type="button" className="vt-pua-more" data-vt-archive-more onClick={() => setAllArchives((v) => !v)}>
                      {allArchives ? `Show the ${ARCHIVES_SHOWN} largest only` : `Show all ${rows.length} archives`}
                    </button>
                  )}
                  <div className="vt-filings-sub vt-streams-foot">
                    The newest file's age is counted in UTC days from the date in its name. A source that publishes weekly or
                    quarterly is not late because its newest file is days old: the streams inventory judges each stream against
                    its own cadence. An archive with no file on the volume and files off-site has been moved there, not lost.
                  </div>
                </>
              )}
            </section>

            <section className="vt-quality-section" data-vt-quality-offsite>
              <div className="vt-quality-section-head">
                <span>Off-site copy</span>
                <span className="vt-quality-section-sub">a second copy of the archive in object storage, checked against the volume</span>
              </div>
              <div className="vt-quality-check">
                {archiveError.offload && !offload && <div className="vt-filings-sub">The off-site status did not load. Refresh asks again.</div>}
                {!archiveError.offload && !offload && <div className="vt-filings-sub">Loading…</div>}
                {offload && (
                  <ul className="vt-quality-check-lines">
                    {offsiteLines(offload, Date.now()).map((line) => <li key={line}>{line}</li>)}
                  </ul>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
