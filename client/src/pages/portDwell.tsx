// PortDwellView — AIS port-dwell/transit full view (#/data/port-dwell).
// The `portdwell` map layer (server/routes.ts, server/portDwell.ts) has only
// ever had a map-marker + sidebar-summary view; this is the standalone
// ranked-table dashboard, closing the last remaining "gate1_pass root with
// no dedicated /data page" gap (research/experiments.md 2026-09-16 session's
// own NEXT — the other two named candidates, sec_form4_bulk_archive and
// entity_map_operator_ticker, already had pages under filings.tsx/graph.tsx).
//
// Same "Open full view" launch pattern as nrc-reactor-status/plant-
// operations (a spatial layer whose per-port ranked table doesn't belong in
// the layer-toggle sidebar) rather than a replacement for the map markers.
// Sort/filter logic lives in client/src/lib/portDwellView.ts (unit-tested
// there) — this file is presentation only. Reuses the vt-filings-* table
// shell (NRC/13F/Form-4 precedent) — no new CSS.
//
// RAW display only (kind:"raw"): every figure is a lower-bound reading off
// our own AIS archive (server/portDwell.ts's own documented caveat —
// terrestrial coverage, day-run merging on the rollup path). The dwell-
// anomaly-vs-forward-returns SIGNAL hypothesis stays gate-2-locked
// (datacore/signal_ladder.json, port_dwell_maritime_transit,
// current_gate: 1) — no predictive claim made here, anomaly flags are the
// server's own 3x-median rule, not a new claim invented in this view.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Anchor } from "lucide-react";
import {
  type PortDwellFilter, type PortDwellPortRow,
  sortPortsByActivity, filterPorts, totalAnomalies,
} from "../lib/portDwellView";

interface DwellPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  window_hours?: number;
  vessels_seen?: number;
  visits_completed?: number;
  in_port_now?: number;
  anomaly_count?: number;
  caveat?: string;
  generated_at?: string;
  ports: PortDwellPortRow[];
}

const FILTERS: PortDwellFilter[] = ["all", "active", "anomaly"];
const FILTER_LABEL: Record<PortDwellFilter, string> = {
  all: "All", active: "In port now", anomaly: "Anomalies",
};

function fmtHours(v: number | null | undefined): string {
  return v == null ? "—" : `${v.toFixed(1)}h`;
}

export default function PortDwellView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<DwellPayload | null>(null);
  const [error, setError] = useState(false);
  const [filter, setFilter] = useState<PortDwellFilter>("all");
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/portdwell");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const ports = data?.ports ?? [];
  const sorted = useMemo(() => sortPortsByActivity(ports), [ports]);
  const visible = useMemo(() => filterPorts(sorted, filter), [sorted, filter]);
  const anomalyTotal = useMemo(() => totalAnomalies(ports), [ports]);
  const active = useMemo(() => visible.find((p) => p.id === selected) ?? null, [visible, selected]);

  return (
    <div className="vt-filings-page" role="region" aria-label="AIS port dwell and transit activity">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Anchor size={16} />
        <div>
          <div className="vt-filings-title">Port dwell & transit — AIS-derived port activity</div>
          <div className="vt-filings-sub">
            {data?.window_hours ? `${data.window_hours}h rolling window` : data?.warming_up ? "warming up…" : "loading…"} ·
            {" "}gate 1 (data) passed, dwell-anomaly signal not attempted ·{" "}
            <a href="https://www.navcen.uscg.gov/ais-information" target="_blank" rel="noreferrer">
              AIS position archive <ExternalLink size={11} />
            </a>
          </div>
        </div>
      </div>

      <div className="vt-filings-filters" role="tablist" aria-label="Filter by activity">
        {FILTERS.map((f) => (
          <button key={f} role="tab" aria-selected={filter === f}
                  className={`vt-filings-filter${filter === f ? " active" : ""}`}
                  onClick={() => setFilter(f)}>
            {FILTER_LABEL[f]}
            {f === "anomaly" && <span style={{ marginLeft: 4 }}>({anomalyTotal})</span>}
          </button>
        ))}
        <span className="vt-filings-count">{visible.length.toLocaleString()} ports</span>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data?.warming_up && data && ports.length === 0 && (
        <div className="vt-filings-state">No ports resolved yet — the poller refreshes every ~10 minutes.</div>
      )}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First archive scan still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && !data?.warming_up && ports.length > 0 && visible.length === 0 && (
        <div className="vt-filings-state">No ports match this filter right now.</div>
      )}

      {visible.length > 0 && (
        <>
          <div className="vt-filings-tablewrap om-sb">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Port</th>
                  <th className="num">In port now</th>
                  <th className="num">Visits</th>
                  <th className="num">Unique vessels</th>
                  <th className="num">Dwell median</th>
                  <th className="num">Dwell p90</th>
                  <th className="num">Anomalies</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((p) => (
                  <tr key={p.id} onClick={() => setSelected(p.id === selected ? null : p.id)}
                      style={{ cursor: "pointer" }} aria-selected={p.id === selected}>
                    <td data-l="Port"><span className="vt-filings-ticker">{p.name}</span></td>
                    <td data-l="In port now" className="num">{p.in_port_now}</td>
                    <td data-l="Visits" className="num">{p.visits_completed}</td>
                    <td data-l="Unique vessels" className="num">{p.unique_vessels}</td>
                    <td data-l="Dwell median" className="num">{fmtHours(p.dwell_median_h)}</td>
                    <td data-l="Dwell p90" className="num">{fmtHours(p.dwell_p90_h)}</td>
                    <td data-l="Anomalies" className="num">
                      {p.anomaly_count > 0
                        ? <span className="vt-filings-kindtag" style={{ color: "var(--vt-danger, #e05252)" }}>{p.anomaly_count}</span>
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="vt-filings-sub">
            {active
              ? `Anomaly examples — ${active.name} (vessels dwelling 3x the port's own median)`
              : "Click a port above to see its flagged-vessel examples, if any."}
          </div>

          {active && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead><tr><th>Vessel</th><th className="num">Dwell</th><th className="num">Port median</th></tr></thead>
                <tbody>
                  {active.anomaly_examples.length === 0
                    ? <tr><td colSpan={3} className="vt-filings-state">No flagged vessels at this port right now.</td></tr>
                    : active.anomaly_examples.map((a, i) => (
                      <tr key={`${a.mmsi}-${i}`}>
                        <td data-l="Vessel">{a.name || a.mmsi}</td>
                        <td data-l="Dwell" className="num">{fmtHours(a.dwell_h)}</td>
                        <td data-l="Port median" className="num">{fmtHours(a.median_h)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}

          {data?.caveat && <div className="vt-filings-sub vt-streams-foot">{data.caveat}</div>}
        </>
      )}
    </div>
  );
}
