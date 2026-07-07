// GridStressView — TX/ERCOT grid-stress reading (#/data/grid-stress).
// GRID VISION A1 gate-2 FAIL path (research/grid_vision_products.md +
// experiments.md 2026-07-07 v1.0.190): the v0 stress index was gate-2
// tested against two pre-stated outcome definitions and voided on both.
// Per the design's own locked FAIL path this ships as a DESCRIPTIVE
// dashboard surface, not a signal — still a product, honestly framed
// (Amendment 5c). Every number on this page is explicitly non-predictive;
// the page leads with that, not buries it in a footnote.
// DESIGN.md-governed: reuses the .vt-filings-* shell + streams-card idiom,
// theme tokens only, designed loading/warming/error/insufficient states.
import { useEffect, useState } from "react";
import { ArrowLeft, Zap, AlertTriangle, RefreshCw } from "lucide-react";

interface GridStressReading {
  date: string;
  respondent: string;
  demand_peak_mw: number | null;
  demand_percentile: number | null;
  demand_sample_days: number;
  forecast_strain_pct: number | null;
  strain_percentile: number | null;
  strain_sample_days: number;
  weather_degree_days: number | null;
  weather_percentile: number | null;
  weather_sample_days: number;
  composite_percentile: number | null;
}
interface GridStressPayload {
  kind: string;
  enabled?: boolean;
  reason?: string;
  warming_up?: boolean;
  predictive: false;
  validation_status?: string;
  note?: string;
  time?: number;
  history_depth_days?: number;
  reading: GridStressReading | null;
}

function fmtMw(v: number | null): string {
  return v == null ? "—" : `${Math.round(v).toLocaleString()} MW`;
}
function fmtPct(v: number | null, digits = 1): string {
  return v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function ordinal(n: number): string {
  const mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1: return `${n}st`;
    case 2: return `${n}nd`;
    case 3: return `${n}rd`;
    default: return `${n}th`;
  }
}
function fmtPercentile(v: number | null): string {
  return v == null ? "n/a — not enough history yet" : `${ordinal(v)} percentile`;
}
function barColor(v: number | null): string {
  if (v == null) return "var(--text-tertiary)";
  if (v >= 90) return "var(--accent-red)";
  if (v >= 70) return "var(--accent-orange)";
  return "var(--accent-blue, #4d9fff)";
}

function Tile({ label, value, percentile, sampleDays, sub }: {
  label: string; value: string; percentile: number | null; sampleDays: number; sub?: string;
}) {
  return (
    <div className="vt-gridstress-tile">
      <div className="vt-gridstress-tile-label">{label}</div>
      <div className="vt-gridstress-tile-value">{value}</div>
      <div className="vt-gridstress-bar" role="img" aria-label={`${label}: ${fmtPercentile(percentile)}`}>
        <div className="vt-gridstress-bar-fill"
             style={{ width: `${percentile ?? 0}%`, background: barColor(percentile) }} />
      </div>
      <div className="vt-gridstress-tile-sub">
        {fmtPercentile(percentile)}
        {sub ? ` · ${sub}` : ""}
        {" · "}{sampleDays} peer day{sampleDays === 1 ? "" : "s"} on file
      </div>
    </div>
  );
}

export default function GridStressView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<GridStressPayload | null>(null);
  const [error, setError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const load = async () => {
    setRefreshing(true);
    try {
      const r = await fetch("/api/data/grid-stress");
      setData(await r.json());
      setError(false);
    } catch {
      setError(true);
    } finally {
      setRefreshing(false);
    }
  };
  useEffect(() => { void load(); }, []);

  const reading = data?.reading ?? null;

  return (
    <div className="vt-filings-page" role="region" aria-label="TX/ERCOT grid-stress reading">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Zap size={16} />
        <div>
          <div className="vt-filings-title">Grid stress (TX/ERCOT)</div>
          <div className="vt-filings-sub">
            {data?.enabled === false ? data.reason
              : data?.warming_up ? "first archive fold in progress — retry shortly"
              : error ? "reading unavailable — retry below"
              : reading ? `reading for ${reading.date} · ${data?.history_depth_days ?? 0} archived days total`
              : "loading…"}
          </div>
        </div>
        <button className="vt-icon-btn" aria-label="Refresh reading" onClick={() => void load()}
                style={{ marginLeft: "auto" }} disabled={refreshing}>
          <RefreshCw size={15} className={refreshing ? "vt-spin" : undefined} />
        </button>
      </div>

      <div className="vt-streams-body">
        <div className="vt-gridstress-banner" role="note">
          <AlertTriangle size={15} aria-hidden />
          <div>
            <b>Descriptive only — not a trading signal.</b> This composite failed gate-2 out-of-sample
            validation twice (forecast-exceedance and pooled-peak outcome designs, both voided on their
            own pre-stated spot-validation rules — see the research log). It is shown for transparency,
            not prediction: nothing here is tradable or sellable.
          </div>
        </div>

        {error && !data && (
          <div className="vt-filings-state">
            Could not load the reading. <button type="button" className="vt-graph-example" onClick={() => void load()}>Retry</button>
          </div>
        )}
        {data?.enabled === false && (
          <div className="vt-filings-state">{data.reason}</div>
        )}
        {data?.warming_up && (
          <div className="vt-filings-state">First archive fold is running on the server — refresh in a few seconds.</div>
        )}
        {data && data.enabled !== false && !data.warming_up && !reading && (
          <div className="vt-filings-state">No complete ERCOT day found in the last 5 days of archive — the feed may still be warming up.</div>
        )}

        {reading && (
          <>
            <div className="vt-gridstress-grid">
              <Tile label="Demand" value={fmtMw(reading.demand_peak_mw)}
                    percentile={reading.demand_percentile} sampleDays={reading.demand_sample_days}
                    sub="same-month daily peak" />
              <Tile label="Forecast strain" value={fmtPct(reading.forecast_strain_pct)}
                    percentile={reading.strain_percentile} sampleDays={reading.strain_sample_days}
                    sub="realized vs. day-ahead" />
              <Tile label="Weather extremity" value={reading.weather_degree_days == null ? "—" : `${Math.round(reading.weather_degree_days)} HDD+CDD`}
                    percentile={reading.weather_percentile} sampleDays={reading.weather_sample_days}
                    sub="TX population-weighted" />
            </div>
            <div className="vt-gridstress-composite">
              <span className="vt-gridstress-composite-label">Equal-weighted composite</span>
              <span className="vt-gridstress-composite-value" style={{ color: barColor(reading.composite_percentile) }}>
                {fmtPercentile(reading.composite_percentile)}
              </span>
            </div>
          </>
        )}

        {data?.note && (
          <div className="vt-filings-sub vt-streams-foot">{data.note}</div>
        )}
      </div>
    </div>
  );
}
