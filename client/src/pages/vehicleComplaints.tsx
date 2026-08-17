// VehicleComplaintsView — NHTSA vehicle complaints watchlist
// (#/data/vehicle-complaints). server/nhtsaComplaints.ts (GATE 1 DATA,
// BUILD ORDER 6 #4) has archived this since 2026-07-06 with zero client
// view until now — one of the two remaining zero-wiring routes the
// 2026-08-16 TFF session's sweep found (bank-failures being the other),
// mechanical repeat of the App Store rankings / GitHub activity precedent
// (RAW display, .vt-filings-* shell, no new CSS). HYPOTHESIS (module's own
// header, gate-locked): complaint-rate acceleration per make/model —
// especially crash/fire flags — precedes recalls and NHTSA investigations.
// RAW display only here, no signal claim — GATE 2 (velocity vs forward
// recalls/returns) is unstarted per the module header and BUILD ORDER 6.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Car } from "lucide-react";

interface VehicleStat {
  ticker: string;
  make: string;
  model: string;
  model_year: number;
  total_complaints: number;
  crash_count: number;
  fire_count: number;
  newest_filed: string | null;
}

interface Payload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  count: number;
  note?: string;
  vehicles: VehicleStat[];
}

const fmtCount = (n: number | null) => (n == null ? "—" : n.toLocaleString());
const titleCase = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());

function sortedRows(vehicles: VehicleStat[]): VehicleStat[] {
  return [...vehicles].sort((a, b) => {
    const bc = b.total_complaints - a.total_complaints;
    if (bc !== 0) return bc;
    if (a.ticker !== b.ticker) return a.ticker.localeCompare(b.ticker);
    return a.model.localeCompare(b.model);
  });
}

export default function VehicleComplaintsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/vehicle-complaints");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = useMemo(() => sortedRows(data?.vehicles ?? []), [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="NHTSA vehicle complaints watchlist">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Car size={16} />
        <div>
          <div className="vt-filings-title">Vehicle complaints — NHTSA watchlist</div>
          <div className="vt-filings-sub">
            per make/model complaint counts + crash/fire flags, curated ticker-mapped watchlist — RAW, no predictive claim ·{" "}
            {data?.time ? `updated ${new Date(data.time).toLocaleString()}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.nhtsa.gov/nhtsa-datasets-and-apis" target="_blank" rel="noreferrer">NHTSA ODI complaints API <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a while on a cold start (curated make/model sweep).</div>
      )}
      {!error && !data?.warming_up && data && rows.length === 0 && (
        <div className="vt-filings-state">No vehicles with complaints on file yet.</div>
      )}

      {!error && rows.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub vt-shortvol-topheader">
            ranked by total complaints on file · curated watchlist only (not the full vehicle universe) ·
            counts are API-reported totals, not archived-window deltas
          </div>
          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Vehicle</th>
                  <th className="num">Total complaints</th>
                  <th className="num">Crash-flagged</th>
                  <th className="num">Fire-flagged</th>
                  <th>Newest filed</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={`${row.ticker}|${row.make}|${row.model}|${row.model_year}`}>
                    <td data-l="Ticker"><span className="vt-filings-ticker">{row.ticker}</span></td>
                    <td data-l="Vehicle">{titleCase(row.make)} {titleCase(row.model)} ({row.model_year})</td>
                    <td data-l="Total complaints" className="num">{fmtCount(row.total_complaints)}</td>
                    <td data-l="Crash-flagged" className="num">{fmtCount(row.crash_count)}</td>
                    <td data-l="Fire-flagged" className="num">{fmtCount(row.fire_count)}</td>
                    <td data-l="Newest filed">{row.newest_filed ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data?.note && <div className="vt-filings-sub vt-streams-foot">{data.note}</div>}
        </div>
      )}
    </div>
  );
}
