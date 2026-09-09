// GridGenerationView — EIA-930 hourly net generation by fuel type,
// #/data/grid-generation. server/gridGeneration.ts shipped the API-only
// route (/api/data/grid-generation, v1.0.870, the missing raw ingredient
// for CLAUDE.md's FUSION HYPOTHESIS (b)) with no client view — this is
// that follow-up, same wiring recipe as gridDemand.tsx (its own sibling
// series: demand vs. generation-by-source, same EIA-930 API, same
// EIA_API_KEY gate). RAW display only: `predictive:false` is asserted
// server-side and restated here — the fusion hypothesis (generation
// shifts x utility tickers, reconciled against registry capacity) stays
// gate-1-locked until archive depth accumulates. Reuses the generic
// .vt-filings-* CSS (atsSummary.tsx precedent) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Zap } from "lucide-react";

interface FuelMixEntry { fueltype: string; latest_mwh: number | null }
interface RespondentGenerationStat {
  respondent: string;
  latest_period: string;
  total_mwh: number | null;
  fuel_mix: FuelMixEntry[];
  hours_in_window: number;
}
interface GridGenerationResponse {
  kind?: string;
  enabled?: boolean;
  reason?: string;
  source?: string;
  attribution?: string;
  time?: number;
  count?: number;
  note?: string;
  respondents?: RespondentGenerationStat[];
  warming_up?: boolean;
  error?: string;
}

const RESPONDENT_LABELS: Record<string, string> = {
  US48: "Lower 48 (total)", CISO: "California ISO", ERCO: "ERCOT (Texas)",
  MISO: "Midcontinent ISO", PJM: "PJM Interconnection", NYIS: "New York ISO",
  ISNE: "ISO New England", SWPP: "Southwest Power Pool", FPL: "Florida Power & Light",
  SE: "Southeast region", NW: "Northwest region", SW: "Southwest region",
};

// EIA-930 fuel-type codes (electricity/rto/fuel-type-data facets doc).
// Storage types (BAT/OES/PS/UES) legitimately go negative — charging
// draws net power — never floored at zero, per server/gridGeneration.ts's
// own data-quality gate comment.
const FUEL_LABELS: Record<string, string> = {
  BAT: "battery storage", COL: "coal", GEO: "geothermal", NG: "natural gas",
  NUC: "nuclear", OES: "other storage", OIL: "oil", OTH: "other", PS: "pumped storage",
  SNB: "solar (small/behind-meter)", SUN: "solar", UES: "unknown storage",
  UNK: "unknown", WAT: "hydro", WNB: "wind (small/behind-meter)", WND: "wind",
};

const num = (n: number | null) => (n == null ? "—" : Math.round(n).toLocaleString());
const fuelLabel = (code: string) => FUEL_LABELS[code] ? `${code} (${FUEL_LABELS[code]})` : code;

/** Compact fuel-mix summary for a row: top 4 sources by magnitude, each
 *  with its share of the row's total_mwh (share is undefined, not 0%,
 *  when total_mwh is null or 0 — never divides by a falsy total). */
function fuelMixSummary(r: RespondentGenerationStat): string {
  if (!r.fuel_mix.length) return "—";
  const top = r.fuel_mix.slice(0, 4);
  return top.map((f) => {
    const share = r.total_mwh ? (((f.latest_mwh ?? 0) / r.total_mwh) * 100).toFixed(0) + "%" : null;
    return `${fuelLabel(f.fueltype)} ${num(f.latest_mwh)}${share ? ` (${share})` : ""}`;
  }).join(" · ") + (r.fuel_mix.length > 4 ? ` +${r.fuel_mix.length - 4} more` : "");
}

export default function GridGenerationView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<GridGenerationResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/grid-generation");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const respondents = data?.respondents ?? [];
  // Rank by total generation (the number this page is about), not
  // alphabetically (the server's own default order) — descending, unknowns last.
  const ranked = [...respondents].sort((a, b) => (b.total_mwh ?? -1) - (a.total_mwh ?? -1));

  return (
    <div className="vt-filings-page" role="region" aria-label="EIA-930 hourly electric grid generation by fuel type">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Zap size={16} />
        <div>
          <div className="vt-filings-title">Electric grid generation by fuel type (EIA-930)</div>
          <div className="vt-filings-sub">
            hourly net generation by fuel source, ~1-2h publication lag — RAW, no predictive claim ·{" "}
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
            RAW ingredient for CLAUDE.md's FUSION HYPOTHESIS (b) — generation shifts x utility tickers, reconciled
            against powerplant registry capacity. That reconciliation is a GATE 1 test that has not run yet: it
            needs several days of accumulated archive depth across a full diurnal generation cycle per region
            before a same-region comparison means anything. Nothing on this page is a validated signal — display only.
            Storage fuel types (battery, pumped, other/unknown storage) legitimately read negative while charging.
          </div>

          {ranked.length > 0 ? (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Balancing authority</th><th className="num">Latest hour (UTC)</th>
                    <th className="num">Total generation (MWh)</th><th>Fuel mix (top sources, share of total)</th>
                    <th className="num">Hours in window</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map((r) => (
                    <tr key={r.respondent}>
                      <td data-l="Balancing authority">
                        <span className="vt-filings-ticker">{r.respondent}</span>
                        {RESPONDENT_LABELS[r.respondent] && <span> ({RESPONDENT_LABELS[r.respondent]})</span>}
                      </td>
                      <td data-l="Latest hour (UTC)" className="num">{r.latest_period}</td>
                      <td data-l="Total generation (MWh)" className="num">{num(r.total_mwh)}</td>
                      <td data-l="Fuel mix (top sources, share of total)">{fuelMixSummary(r)}</td>
                      <td data-l="Hours in window" className="num">{num(r.hours_in_window)}</td>
                    </tr>
                  ))}
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
