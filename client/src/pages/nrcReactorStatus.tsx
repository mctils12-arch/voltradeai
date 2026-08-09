// NrcReactorStatusView — NRC daily nuclear reactor status full view
// (#/data/nrc-reactor-status). The `nrc_reactor_status` map layer (built
// 2026-08-04, GATE 1 DATA passed same day — datacore/layers.json,
// server/nrcReactorStatus.ts) has only ever had a map-marker view; this is
// the standalone dashboard page queued since (research/experiments.md's
// 2026-08-08/2026-08-09 NEXT notes: "sec_form4_bulk_archive /
// nrc_outage_reports still without a standalone /data page — NRC has a
// map-layer toggle only, no dedicated dashboard"). Verified this session
// that sec_form4_bulk_archive (client/src/pages/filings.tsx) and
// sec_edgar_13f_institutional_clustering (client/src/pages/edgar13f.tsx)
// already got theirs in earlier sessions — those queue notes were stale;
// NRC was the one genuine remaining gap.
//
// Same "Open full view" launch pattern as methane_plumes (a spatial layer
// whose per-asset ranked table doesn't belong in the layer-toggle
// sidebar) rather than the page-top launcher list used by non-spatial
// clusters (VIX/FRED/13F/etc) — NRC already has a map layer, this page is
// a sortable-table complement to it, not a replacement launch surface.
// Sort/status-tally logic lives in client/src/lib/nrcReactorStatus.ts
// (behavioral-tested there) — this file is presentation only. Reuses the
// vt-filings-* table shell (13F/Form-4/crop-conditions precedent) — no
// new CSS. RAW display only (kind:"raw"): every value is the NRC's own
// reported percent-of-rated-thermal-power; the outage-adjacent SIGNAL
// hypothesis stays gate-2-locked, no predictive claim made here.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Zap } from "lucide-react";
import { nrcReactorStatusColor } from "../lib/mapIcons";
import {
  type PlantPowerStatus, type PlantReactorStatus,
  STATUS_LABEL, sortPlantsByStatus, statusCounts,
} from "../lib/nrcReactorStatus";

interface StatusPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: string;
  date?: string;
  count: number;
  note?: string;
  plantCount?: number;
  plants: PlantReactorStatus[];
}

type Filter = "all" | PlantPowerStatus;
const FILTERS: Filter[] = ["all", "outage", "reduced", "unknown", "full"];

function fmtPower(v: number | null): string {
  return v == null ? "—" : `${v.toFixed(1)}%`;
}

export default function NrcReactorStatusView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<StatusPayload | null>(null);
  const [error, setError] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/nrc-reactor-status");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const plants = data?.plants ?? [];
  const sorted = useMemo(() => sortPlantsByStatus(plants), [plants]);
  const counts = useMemo(() => statusCounts(plants), [plants]);
  const visible = filter === "all" ? sorted : sorted.filter((p) => p.status === filter);
  const active = useMemo(() => sorted.find((p) => p.idx === selected) ?? null, [sorted, selected]);

  return (
    <div className="vt-filings-page" role="region" aria-label="NRC daily reactor status">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Zap size={16} />
        <div>
          <div className="vt-filings-title">Nuclear reactor status — NRC daily report</div>
          <div className="vt-filings-sub">
            percent of rated thermal power, per plant · {data?.date ? `report date ${data.date}` : data?.warming_up ? "warming up…" : "loading…"} ·
            {" "}gate 1 (data) passed, outage-adjacent signal not attempted ·{" "}
            <a href="https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/"
               target="_blank" rel="noreferrer">NRC Power Reactor Status Reports <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      <div className="vt-filings-filters" role="tablist" aria-label="Filter by status">
        {FILTERS.map((f) => (
          <button key={f} role="tab" aria-selected={filter === f}
                  className={`vt-filings-filter${filter === f ? " active" : ""}`}
                  onClick={() => setFilter(f)}>
            {f === "all" ? "All" : STATUS_LABEL[f]}
            {f !== "all" && <span style={{ marginLeft: 4, color: nrcReactorStatusColor(f) }}>({counts[f]})</span>}
          </button>
        ))}
        <span className="vt-filings-count">{visible.length.toLocaleString()} plants</span>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data?.warming_up && data && plants.length === 0 && (
        <div className="vt-filings-state">
          No plants resolved yet — the poll runs every ~6 hours; NRC collects this report 4am-8am ET daily.
        </div>
      )}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && !data?.warming_up && plants.length > 0 && visible.length === 0 && (
        <div className="vt-filings-state">No plants match this filter today.</div>
      )}

      {visible.length > 0 && (
        <>
          <div className="vt-filings-tablewrap om-sb">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Plant</th><th>Owner</th>
                  <th className="num">Capacity (MW)</th><th>Status</th>
                  <th className="num">Avg power</th><th className="num">Units</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((p) => (
                  <tr key={p.idx} onClick={() => setSelected(p.idx === selected ? null : p.idx)}
                      style={{ cursor: "pointer" }} aria-selected={p.idx === selected}>
                    <td data-l="Plant"><span className="vt-filings-ticker">{p.name}</span></td>
                    <td data-l="Owner">{p.owner || "—"}</td>
                    <td data-l="Capacity (MW)" className="num">{p.mw ? Math.round(p.mw).toLocaleString() : "—"}</td>
                    <td data-l="Status">
                      <span className="vt-filings-kindtag" style={{ color: nrcReactorStatusColor(p.status) }}>
                        {STATUS_LABEL[(p.status as PlantPowerStatus)] || p.status}
                      </span>
                    </td>
                    <td data-l="Avg power" className="num">{fmtPower(p.avgPower)}</td>
                    <td data-l="Units" className="num">{p.units.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="vt-filings-sub">
            {active
              ? `Per-unit breakdown — ${active.name}`
              : "Click a plant above to see its per-unit power breakdown."}
          </div>

          {active && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead><tr><th>Unit</th><th className="num">Power</th></tr></thead>
                <tbody>
                  {active.units.map((u, i) => (
                    <tr key={`${u.unit}-${i}`}>
                      <td data-l="Unit">{u.unit}</td>
                      <td data-l="Power" className="num">{fmtPower(u.power)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {data?.note && <div className="vt-filings-sub vt-streams-foot">{data.note}</div>}
        </>
      )}
    </div>
  );
}
