// PlantOperationsView — EPA CAMD CEMS per-facility power-plant utilization
// ground truth (#/data/plant-operations). server/epaCamd.ts (BUILT
// v1.0.385, /data map layer SHIPPED 2026-07-20 — datacore/layers.json
// "plant_operations", facilities group) has only ever had a map-marker
// view; this is the standalone ranked-table complement, same
// "shipped-data-no-client-page" gap the USAspending/DTCC-swaps/JODI
// precedents already closed for their own roots. RAW display only
// (kind:"raw", predictive:false, restated verbatim from the route): every
// value is the sum of EPA's own unit-level grossLoad/opTime reporting for
// the newest archived quarter — the ladder gate-1 truth source for the
// power vertical, not itself a trading signal (no threshold, ranking, or
// color implies one here). facilities[] arrives pre-sorted by sumGrossLoad
// descending (server/epaCamd.ts aggregateByFacility) — no client-side sort
// state needed, same "presentation only, server does the ordering" shape
// as dtccSwaps.tsx/usaspendingContracts.tsx.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Gauge } from "lucide-react";

interface FacilityRow {
  facilityId: number;
  facilityName: string | null;
  unitCount: number;
  sumOpTime: number;
  sumGrossLoad: number;
  primaryFuelInfo: string | null;
  lat: number | null;
  lon: number | null;
  ownerOperator: string | null;
}
interface PlantOpsPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  state?: string;
  note?: string;
  key_mode?: string;
  time?: number;
  year?: number;
  quarter?: number;
  unit_days?: number;
  facilities: FacilityRow[];
}

const fmtNum = (n: number, digits = 0) =>
  n.toLocaleString(undefined, { maximumFractionDigits: digits });

export default function PlantOperationsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<PlantOpsPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/plant-operations");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const facilities = data?.facilities ?? [];

  return (
    <div className="vt-filings-page" role="region" aria-label="EPA CAMD power-plant utilization">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Gauge size={16} />
        <div>
          <div className="vt-filings-title">Power-plant utilization — EPA CAMD CEMS (TX pilot)</div>
          <div className="vt-filings-sub">
            per-facility ground truth, ranked by gross load ·{" "}
            {data?.year && data?.quarter ? `Q${data.quarter} ${data.year}` : data?.warming_up ? "warming up…" : "loading…"} ·
            {" "}RAW, no predictive claim ·{" "}
            <a href="https://www.epa.gov/airmarkets/cedri-and-other-data-access-tools"
               target="_blank" rel="noreferrer">EPA Clean Air Markets Division <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">
          First quarter still loading — EPA CAMD is quarterly-cadence data, this can take a few minutes on a cold start.
        </div>
      )}
      {!error && data && !data.warming_up && facilities.length === 0 && (
        <div className="vt-filings-state">No facilities resolved for {data.state ?? "this state"} yet.</div>
      )}

      {!error && data && !data.warming_up && facilities.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data.state} · {facilities.length.toLocaleString()} facilities ·{" "}
            {data.unit_days?.toLocaleString()} unit-days in the archived quarter ·{" "}
            key: {data.key_mode}
          </div>

          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Facility</th>
                  <th>Owner / operator</th>
                  <th>Primary fuel</th>
                  <th className="num">Units</th>
                  <th className="num">Gross load (MW-days)</th>
                  <th className="num">Operating hours</th>
                </tr>
              </thead>
              <tbody>
                {facilities.map((f) => (
                  <tr key={f.facilityId}>
                    <td data-l="Facility">{f.facilityName || `Facility ${f.facilityId}`}</td>
                    <td data-l="Owner / operator">{f.ownerOperator || "—"}</td>
                    <td data-l="Primary fuel">{f.primaryFuelInfo || "—"}</td>
                    <td data-l="Units" className="num">{f.unitCount}</td>
                    <td data-l="Gross load (MW-days)" className="num">{fmtNum(f.sumGrossLoad, 1)}</td>
                    <td data-l="Operating hours" className="num">{fmtNum(f.sumOpTime, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="vt-filings-sub vt-streams-foot">
            {data.attribution} · {data.note}
          </div>
        </div>
      )}
    </div>
  );
}
