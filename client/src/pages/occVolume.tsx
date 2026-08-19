// OccVolumeView — OCC daily options cleared volume by trade origin,
// #/data/occ-volume. server/occVolume.ts shipped the API-only route
// (/api/data/occ-volume, DATACORE MAXIMUS census #1, v1.0.580) with no
// client view — this is that follow-up, same wiring recipe as
// gridDemand.tsx/fleetUtilization.tsx. RAW display only: GATE 1 (day
// totals vs OCC's own published daily volume page) PASSED (v1.0.580);
// GATE 2 (pre-registered customer call/put skew, ISEE-successor
// definition) was ATTEMPTED AND KILLED (v1.0.585) — the pre-stated pass
// bar failed in its REVERSED direction, stated here verbatim per the same
// honesty standard gridDemand.tsx applies to its own gate status. Reuses
// the generic .vt-filings-* CSS (atsSummary.tsx precedent) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, TrendingDown } from "lucide-react";

interface UnderlyingStat {
  underlying: string;
  cust_put: number;
  cust_call: number;
  mm_put: number;
  mm_call: number;
  total_cleared: number;
}
interface OccVolumeResponse {
  kind?: string;
  source?: string;
  attribution?: string;
  time?: number;
  report_date?: string;
  underlyings?: number;
  count?: number;
  note?: string;
  top?: UnderlyingStat[];
  warming_up?: boolean;
}

const num = (n: number) => Math.round(n).toLocaleString();

/** Customer call/put skew — the exact GATE 2 KILLED definition, shown here
 *  as an informational split only. "—" when there is no customer volume on
 *  either side (nothing to ratio). */
function custSkew(r: UnderlyingStat): string {
  const total = r.cust_call + r.cust_put;
  if (total === 0) return "—";
  return `${((r.cust_call / total) * 100).toFixed(0)}% call`;
}

export default function OccVolumeView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<OccVolumeResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/occ-volume");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const top = data?.top ?? [];

  return (
    <div className="vt-filings-page" role="region" aria-label="OCC daily options cleared volume">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <TrendingDown size={16} />
        <div>
          <div className="vt-filings-title">Options cleared volume — OCC daily</div>
          <div className="vt-filings-sub">
            top underlyings by daily cleared volume, customer vs market-maker put/call split — RAW, no predictive claim ·{" "}
            {data?.report_date ? `report date ${data.report_date}` : error ? "" : "loading…"} ·{" "}
            <a href="https://marketdata.theocc.com/volume-query" target="_blank" rel="noreferrer">OCC volume query <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}

      {!error && data && data.warming_up && (
        <div className="vt-filings-state">Warming up — first archive scan in progress.</div>
      )}

      {!error && data && !data.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data.underlyings != null ? `${num(data.underlyings)} underlyings traded, top ${top.length} shown by cleared volume` : ""}
          </div>
          {data.note && <div className="vt-filings-sub">{data.note}</div>}
          <div className="vt-filings-sub">
            GATE 1 (one day's totals vs. OCC's own published daily volume page) PASSED (v1.0.580). GATE 2
            (pre-registered customer call/put skew, an ISEE-successor definition) was ATTEMPTED AND KILLED
            (v1.0.585): the pre-stated pass bar (call-skewed names beat put-skewed names on both the 5-day and
            20-day horizon) FAILED — the ordering came back REVERSED, put-skewed names outperforming
            call-skewed names at both horizons. That reversal is not promoted as a new signal — it was one
            pre-registered test with a naive-pooled t-stat that likely overstates its own significance, never
            replicated out-of-sample. Nothing on this page is a validated signal — display only.
          </div>

          {top.length > 0 ? (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Underlying</th><th className="num">Total cleared</th>
                    <th className="num">Customer calls</th><th className="num">Customer puts</th>
                    <th className="num">Customer call/put skew</th>
                    <th className="num">MM calls</th><th className="num">MM puts</th>
                  </tr>
                </thead>
                <tbody>
                  {top.map((r) => (
                    <tr key={r.underlying}>
                      <td data-l="Underlying"><span className="vt-filings-ticker">{r.underlying}</span></td>
                      <td data-l="Total cleared" className="num">{num(r.total_cleared)}</td>
                      <td data-l="Customer calls" className="num">{num(r.cust_call)}</td>
                      <td data-l="Customer puts" className="num">{num(r.cust_put)}</td>
                      <td data-l="Customer call/put skew" className="num">{custSkew(r)}</td>
                      <td data-l="MM calls" className="num">{num(r.mm_call)}</td>
                      <td data-l="MM puts" className="num">{num(r.mm_put)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="vt-filings-state">No underlyings in the current archive window.</div>
          )}
        </div>
      )}
    </div>
  );
}
