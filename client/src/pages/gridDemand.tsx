// GridDemandView — EIA-930 hourly electric grid demand by balancing
// authority, #/data/grid-demand. server/gridDemand.ts shipped the API-only
// route (/api/data/grid-demand, DATACORE MAXIMUS Phase 0, v1.0.163) with no
// client view — this is that follow-up, same wiring recipe as
// fleetUtilization.tsx/secFtd.tsx. RAW display only: GATE 1 (revision
// stability — an independent later re-draw of the same API) PASSED; GATE 2
// (weather-adjusted demand residual vs forward industrial-sector returns)
// was ATTEMPTED AND KILLED 2026-08-09 (signs disagreed at both 20d/60d
// horizons, no comparison near significant) — stated here verbatim per the
// same honesty standard fleetUtilization.tsx applies to its own gate status,
// not softened to "not attempted". Reuses the generic .vt-filings-* CSS
// (atsSummary.tsx precedent) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Zap } from "lucide-react";

interface RespondentStat {
  respondent: string;
  latest_period: string;
  latest_mwh: number | null;
  latest_forecast_mwh: number | null;
  hours_in_window: number;
}
interface GridDemandResponse {
  kind?: string;
  enabled?: boolean;
  reason?: string;
  source?: string;
  attribution?: string;
  time?: number;
  count?: number;
  note?: string;
  respondents?: RespondentStat[];
  warming_up?: boolean;
  error?: string;
}

const RESPONDENT_LABELS: Record<string, string> = {
  US48: "Lower 48 (total)", CISO: "California ISO", ERCO: "ERCOT (Texas)",
  MISO: "Midcontinent ISO", PJM: "PJM Interconnection", NYIS: "New York ISO",
  ISNE: "ISO New England", SWPP: "Southwest Power Pool", FPL: "Florida Power & Light",
  SE: "Southeast region", NW: "Northwest region", SW: "Southwest region",
};

const num = (n: number | null) => (n == null ? "—" : Math.round(n).toLocaleString());

/** Forecast strain: realized minus day-ahead-forecast demand for the same
 *  hour, as a % of the forecast — server contract: null forecast means DF
 *  hasn't published for that hour yet, never coerced to 0. */
function strainPct(r: RespondentStat): number | null {
  if (r.latest_mwh == null || r.latest_forecast_mwh == null || r.latest_forecast_mwh === 0) return null;
  return ((r.latest_mwh - r.latest_forecast_mwh) / r.latest_forecast_mwh) * 100;
}

export default function GridDemandView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<GridDemandResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/grid-demand");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const respondents = data?.respondents ?? [];
  // Rank by latest demand (the number this page is about), not alphabetically
  // (the server's own default order) — descending, unknowns last.
  const ranked = [...respondents].sort((a, b) => (b.latest_mwh ?? -1) - (a.latest_mwh ?? -1));

  return (
    <div className="vt-filings-page" role="region" aria-label="EIA-930 hourly electric grid demand">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Zap size={16} />
        <div>
          <div className="vt-filings-title">Electric grid demand (EIA-930)</div>
          <div className="vt-filings-sub">
            hourly demand by balancing authority, ~1-2h publication lag — RAW, no predictive claim ·{" "}
            {data?.time ? `as of ${new Date(data.time).toUTCString()}` : error ? "" : "loading…"} ·{" "}
            <a href="https://www.eia.gov/electricity/gridmonitor/" target="_blank" rel="noreferrer">EIA-930 Grid Monitor <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}

      {!error && data && data.enabled === false && (
        <div className="vt-filings-state">Not enabled — {data.reason || "EIA_API_KEY not set"}.</div>
      )}
      {!error && data && data.enabled !== false && data.warming_up && (
        <div className="vt-filings-state">Warming up — first archive scan in progress.</div>
      )}

      {!error && data && data.enabled !== false && !data.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data.count != null ? `${num(data.count)} respondents (US48 total, major balancing authorities, and regional aggregates)` : ""}
          </div>
          {data.note && <div className="vt-filings-sub">{data.note}</div>}
          <div className="vt-filings-sub">
            GATE 1 (revision stability) PASSED — an independent later re-draw of the same API matched within 0.24%.
            GATE 2 (weather-adjusted demand residual vs. forward industrial-sector returns) was ATTEMPTED AND
            KILLED 2026-08-09: predicted signs disagreed with realized returns at both the 20-day and 60-day
            horizons, and no comparison came close to statistical significance. Nothing on this page is a
            validated signal — display only.
          </div>

          {ranked.length > 0 ? (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Balancing authority</th><th className="num">Latest hour (UTC)</th>
                    <th className="num">Demand (MWh)</th><th className="num">Day-ahead forecast (MWh)</th>
                    <th className="num">Forecast strain</th><th className="num">Hours in window</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map((r) => {
                    const strain = strainPct(r);
                    return (
                      <tr key={r.respondent}>
                        <td data-l="Balancing authority">
                          <span className="vt-filings-ticker">{r.respondent}</span>
                          {RESPONDENT_LABELS[r.respondent] && <span> ({RESPONDENT_LABELS[r.respondent]})</span>}
                        </td>
                        <td data-l="Latest hour (UTC)" className="num">{r.latest_period}</td>
                        <td data-l="Demand (MWh)" className="num">{num(r.latest_mwh)}</td>
                        <td data-l="Day-ahead forecast (MWh)" className="num">{num(r.latest_forecast_mwh)}</td>
                        <td data-l="Forecast strain" className="num">{strain == null ? "—" : `${strain > 0 ? "+" : ""}${strain.toFixed(1)}%`}</td>
                        <td data-l="Hours in window" className="num">{num(r.hours_in_window)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="vt-filings-state">No respondents in the current archive window.</div>
          )}
        </div>
      )}
    </div>
  );
}
