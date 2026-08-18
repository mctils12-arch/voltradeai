// VehicleComplaintsView — NHTSA vehicle complaints, curated ticker-mapped
// watchlist, #/data/vehicle-complaints.
// server/nhtsaComplaints.ts (BUILD ORDER 6 #4, 2026-07-06) shipped the
// API-only route (/api/data/vehicle-complaints, keyless NHTSA ODI) with no
// client view and no datacore/layers.json entry at all — the same
// "shipped-data-no-UI" gap class as treasury_dts/treasury_auctions/tff/
// fda_events, but this route additionally had ZERO registry wiring (those
// four already had a layers.json entry before their own client-view PR).
// RAW display only; the module's own hypothesis (complaint-rate
// acceleration per make/model, esp. crash/fire flags, precedes recalls and
// NHTSA investigations) stays gate-locked at gate 1 (not attempted).
// Reuses the generic .vt-filings-*/.vt-shortvol-* CSS — no new styles.
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
interface ComplaintsResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: string;
  count?: number;
  note?: string;
  warming_up?: boolean;
  vehicles?: VehicleStat[];
}

const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString());

export default function VehicleComplaintsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<ComplaintsResponse | null>(null);
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

  // Group by ticker (the watchlist maps several models per company) and
  // rank groups by combined crash+fire flags — the most safety-relevant
  // tickers surface first. This is a display ranking only, not a model
  // output: no score, no threshold, no trade implication.
  const groups = useMemo(() => {
    const byTicker = new Map<string, VehicleStat[]>();
    for (const v of data?.vehicles || []) {
      const list = byTicker.get(v.ticker) || [];
      list.push(v);
      byTicker.set(v.ticker, list);
    }
    return Array.from(byTicker.entries())
      .map(([ticker, vehicles]) => {
        const models = vehicles.slice().sort((a, b) => b.total_complaints - a.total_complaints);
        const totalComplaints = models.reduce((s, m) => s + m.total_complaints, 0);
        const totalCrash = models.reduce((s, m) => s + m.crash_count, 0);
        const totalFire = models.reduce((s, m) => s + m.fire_count, 0);
        return { ticker, models, totalComplaints, totalCrash, totalFire };
      })
      .sort((a, b) => (b.totalCrash + b.totalFire) - (a.totalCrash + a.totalFire));
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="NHTSA vehicle complaints">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Car size={16} />
        <div>
          <div className="vt-filings-title">Vehicle Complaints</div>
          <div className="vt-filings-sub">
            NHTSA complaint counts over a curated ticker-mapped watchlist (RAW, not a signal) ·{" "}
            {data && !data.warming_up ? `${data.count ?? 0} vehicles, ${groups.length} tickers` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.nhtsa.gov/nhtsa-datasets-and-apis" target="_blank" rel="noreferrer">NHTSA ODI complaints API <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — check back shortly.</div>
      )}

      {!error && !data?.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            covers a CURATED make/model/year watchlist mapped to public tickers, not the full NHTSA
            vehicle universe — a make/model absent here was never polled, not screened out for being
            clean. Groups below are ranked by combined crash+fire flag count, a display ordering only,
            no score or threshold applied.
          </div>
          <div className="vt-filings-sub">
            hypothesis under research (gate-locked, gate 1 not attempted): complaint-rate acceleration
            per make/model — especially crash/fire flags — precedes recalls and NHTSA investigations,
            which move automakers and their suppliers. Nothing below reflects that hypothesis yet — this
            is a raw per-vehicle count table, no velocity/rate model applied. "Total complaints" is the
            API's all-time cumulative count for that make/model/year, not a count of new complaints this
            window; crash/fire counts are flag totals within the same fetched set.
          </div>

          {!data && <div className="vt-filings-state">Loading latest complaint stats…</div>}
          {data && groups.length === 0 && <div className="vt-filings-state">No complaint data in the current watchlist window.</div>}

          {groups.map((g) => (
            <div key={g.ticker}>
              <div className="vt-filings-sub vt-shortvol-topheader">
                <span className="vt-filings-ticker">{g.ticker}</span> · {fmt(g.totalComplaints)} complaints ·{" "}
                {fmt(g.totalCrash)} crash · {fmt(g.totalFire)} fire
              </div>
              <div className="vt-filings-tablewrap">
                <table className="vt-filings-table">
                  <thead>
                    <tr>
                      <th>Model</th><th className="num">Total complaints</th>
                      <th className="num">Crash</th><th className="num">Fire</th><th>Newest filed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {g.models.map((m) => (
                      <tr key={`${m.make}-${m.model}-${m.model_year}`}>
                        <td data-l="Model">{m.make} {m.model} ({m.model_year})</td>
                        <td data-l="Total complaints" className="num">{fmt(m.total_complaints)}</td>
                        <td data-l="Crash" className="num">{fmt(m.crash_count)}</td>
                        <td data-l="Fire" className="num">{fmt(m.fire_count)}</td>
                        <td data-l="Newest filed">{m.newest_filed || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
