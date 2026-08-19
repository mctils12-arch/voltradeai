// FleetUtilizationView — corporate/LLC aircraft fleet utilization
// (aircraft position archive x FAA entity spine join), #/data/fleet-utilization.
// server/fleetUtilization.ts shipped the API-only route (/api/data/
// fleet-utilization, BUILD ORDER 3 #1, v1.0.578) with no client view — this
// is that follow-up, same wiring recipe as midas.tsx/secFtd.tsx. RAW display
// of a weekly session count (GATE 1 join accuracy PASSED 20/20, GATE 2
// utilization-x-earnings NOT attempted — nothing here is a signal, per the
// server module's own docstring). Reuses the generic .vt-filings-* CSS
// (atsSummary.tsx precedent) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Plane } from "lucide-react";

interface OwnerRow {
  owner: string;
  group: string;
  resolution: { callsign: number; registrant: number };
  trustee_airframes: number;
  registrant_type: string;
  n_airframes: number;
  weekly: Record<string, { f: number; h: number }>;
}
interface FleetResponse {
  kind?: string;
  source?: string;
  attribution?: string;
  as_of?: string;
  owners_total?: number;
  count?: number;
  note?: string;
  owners?: OwnerRow[];
  error?: string;
}

const num = (n: number) => n.toLocaleString();

/** Sum an owner's weekly {f,h} buckets into totals, honestly counting only
 *  the weeks the archive actually covers (weeks without coverage are
 *  absent, not zero — same contract the server route documents). */
function totals(o: OwnerRow): { weeks: number; flights: number; hours: number } {
  const weeks = Object.values(o.weekly);
  return {
    weeks: weeks.length,
    flights: weeks.reduce((s, w) => s + w.f, 0),
    hours: +weeks.reduce((s, w) => s + w.h, 0).toFixed(1),
  };
}

const basisLabel = (o: OwnerRow) => {
  if (o.registrant_type === "historical-archive-only") return "historical archive only";
  const { callsign, registrant } = o.resolution;
  if (callsign && registrant) return `${callsign} callsign / ${registrant} registrant`;
  if (callsign) return "callsign-resolved";
  return "FAA registrant";
};

export default function FleetUtilizationView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FleetResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/fleet-utilization");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const owners = data?.owners ?? [];
  // Rank by total airborne hours (the utilization the page is about), not
  // by n_airframes (fleet size) — the server's own sort order is by
  // airframe count, so this page re-sorts client-side for its own purpose.
  const ranked = [...owners]
    .map((o) => ({ o, t: totals(o) }))
    .sort((a, b) => b.t.hours - a.t.hours);

  return (
    <div className="vt-filings-page" role="region" aria-label="Corporate/LLC aircraft fleet utilization">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Plane size={16} />
        <div>
          <div className="vt-filings-title">Corporate/LLC aircraft fleet utilization</div>
          <div className="vt-filings-sub">
            weekly flight sessions per registered owner, ADS-B archive × FAA entity spine — RAW, no predictive claim ·{" "}
            {data?.as_of ? `as of ${data.as_of}` : error ? "" : "loading…"} ·{" "}
            <a href="https://registry.faa.gov/aircraftinquiry/" target="_blank" rel="noreferrer">FAA Aircraft Registry <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}

      {!error && data && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data.owners_total != null ? `${num(data.owners_total)} registered owners with 2+ tracked airframes · showing top ${num(data.count ?? ranked.length)} by airborne hours` : ""}
          </div>
          {data.note && <div className="vt-filings-sub">{data.note}</div>}
          <div className="vt-filings-sub">
            GATE 1 (spine-join accuracy) PASSED — 20/20 exact N-number matches vs. independent registrations. GATE 2
            (utilization × earnings timing) NOT attempted: nothing on this page is a validated signal.
          </div>

          {ranked.length > 0 ? (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Registrant / operator</th><th>Basis</th><th className="num">Airframes</th>
                    <th className="num">Weeks tracked</th><th className="num">Flights</th><th className="num">Airborne hours (lower bound)</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map(({ o, t }) => (
                    <tr key={o.owner}>
                      <td data-l="Registrant / operator">
                        <span className="vt-filings-ticker">{o.owner}</span>
                        {o.group !== o.owner && <span> ({o.group})</span>}
                        {o.trustee_airframes > 0 && <span title="FAA registrant is a trustee/leasing shell for at least one airframe"> · trustee-registered</span>}
                      </td>
                      <td data-l="Basis">{basisLabel(o)}</td>
                      <td data-l="Airframes" className="num">{num(o.n_airframes)}</td>
                      <td data-l="Weeks tracked" className="num">{num(t.weeks)}</td>
                      <td data-l="Flights" className="num">{num(t.flights)}</td>
                      <td data-l="Airborne hours (lower bound)" className="num">{num(t.hours)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="vt-filings-state">No owners with 2+ tracked airframes in the current archive window.</div>
          )}
        </div>
      )}
    </div>
  );
}
