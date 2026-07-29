// SpaceWeatherView — the full NOAA SWPC space-weather view (#/data/space-weather).
// Same non-geospatial standalone-page pattern as attention.tsx/cot.tsx:
// a status-badge poll on the map page + a full view here. RAW overlay —
// official readings + NOAA's own published classification formulas, no
// predictive claim. Mike's GIC hypothesis (research/open_questions.md
// GRID-ADJACENT FUTURE ROOTS #1) stays gate-1-unvalidated.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Zap } from "lucide-react";

interface KIndexPoint { time_tag: string; kp: number; a_running: number | null; station_count: number | null; }
interface XrayPoint { time_tag: string; satellite: number | null; energy: string; flux: number; electron_contamination: boolean | null; }
interface SpaceWeather {
  time: number;
  kindex: { latest: KIndexPoint | null; gScale: { gScale: number | null; label: string } | null; history: KIndexPoint[] };
  xray: { latest: XrayPoint | null; flare: { cls: string; magnitude: number; label: string } | null; history: XrayPoint[] };
}

/** Minimal inline sparkline — mirrors attention.tsx's, no charting dependency. */
function Sparkline({ points, width = 280, height = 48, log = false }: { points: Array<number | null>; width?: number; height?: number; log?: boolean }) {
  const vals = points
    .map((v, i) => (v == null ? null : { i, v: log && v > 0 ? Math.log10(v) : v }))
    .filter((p): p is { i: number; v: number } => p != null);
  if (vals.length < 2) return <div className="vt-shortvol-spark-empty">not enough history yet</div>;
  const lo = Math.min(...vals.map((p) => p.v));
  const hi = Math.max(...vals.map((p) => p.v));
  const span = hi - lo || 1;
  const x = (i: number) => (i / (points.length - 1)) * (width - 4) + 2;
  const y = (v: number) => height - 2 - ((v - lo) / span) * (height - 4);
  const d = vals.map((p, k) => `${k === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  return (
    <svg className="vt-shortvol-spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
         aria-label={`trend, ${vals.length} points`}>
      <path d={d} fill="none" stroke="var(--accent-bright)" strokeWidth={1.6} />
    </svg>
  );
}

const G_COLOR: Record<number, string> = { 1: "#ffd23f", 2: "#ffb020", 3: "#ff8c42", 4: "#ff5c3b", 5: "#ff3b3b" };
const FLARE_COLOR: Record<string, string> = { A: "#4d9fff", B: "#4d9fff", C: "#9aa4b2", M: "#ff8c42", X: "#ff3b3b" };

export default function SpaceWeatherView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<SpaceWeather | null>(null);
  const [warming, setWarming] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/space-weather");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setWarming(true); return; }
        setWarming(false);
        setData(d);
      } catch {
        if (!stop) setError(true);
      }
    };
    load();
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 120_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, []);

  const kp = data?.kindex.latest;
  const gScale = data?.kindex.gScale;
  const xray = data?.xray.latest;
  const flare = data?.xray.flare;

  return (
    <div className="vt-filings-page" role="region" aria-label="Space weather (NOAA SWPC)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Zap size={16} />
        <div>
          <div className="vt-filings-title">Space weather — NOAA SWPC</div>
          <div className="vt-filings-sub">
            planetary K-index/G-scale + GOES X-ray flux/flare class — official readings, RAW display, not a signal ·{" "}
            <a href="https://www.swpc.noaa.gov/" target="_blank" rel="noreferrer">swpc.noaa.gov <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && warming && <div className="vt-filings-state">Warming up — first poll can take a minute.</div>}

      {!error && !warming && (
        <div className="vt-shortvol-body">
          <div className="vt-shortvol-snapshot">
            <div className="vt-shortvol-statblock">
              <div className="vt-shortvol-statlabel">planetary Kp, latest 3h window</div>
              <div className="vt-shortvol-statval">{kp ? kp.kp.toFixed(2) : "—"}</div>
              {gScale && (
                <div className="vt-filings-count" style={gScale.gScale ? { color: G_COLOR[gScale.gScale] } : undefined}>
                  {gScale.label}
                </div>
              )}
              {kp && <div className="vt-shortvol-trendnote">as of {kp.time_tag} UTC · {kp.station_count ?? "—"} stations</div>}
            </div>
            <div className="vt-shortvol-trendblock">
              <div className="vt-shortvol-statlabel">last {data?.kindex.history.length || 0} readings (~{Math.round((data?.kindex.history.length || 0) * 3)}h)</div>
              <Sparkline points={(data?.kindex.history || []).map((h) => h.kp)} />
              <div className="vt-shortvol-trendnote">G1+ (Kp≥5) = official NOAA minor-storm threshold</div>
            </div>
          </div>

          <div className="vt-shortvol-snapshot">
            <div className="vt-shortvol-statblock">
              <div className="vt-shortvol-statlabel">GOES X-ray flare class, 0.1-0.8nm band</div>
              <div className="vt-shortvol-statval" style={flare ? { color: FLARE_COLOR[flare.cls] || undefined } : undefined}>
                {flare ? flare.label : "—"}
              </div>
              {xray && <div className="vt-filings-count">{xray.flux.toExponential(2)} W/m²</div>}
              {xray && <div className="vt-shortvol-trendnote">as of {xray.time_tag} UTC{xray.electron_contamination ? " · electron contamination flagged upstream" : ""}</div>}
            </div>
            <div className="vt-shortvol-trendblock">
              <div className="vt-shortvol-statlabel">last {data?.xray.history.length || 0} minutes (log scale)</div>
              <Sparkline points={(data?.xray.history || []).map((h) => h.flux)} log />
              <div className="vt-shortvol-trendnote">C/M/X = official NOAA GOES flare classes</div>
            </div>
          </div>

          <div className="vt-filings-sub vt-shortvol-topheader">
            GIC hypothesis (geomagnetic storms stress grid transformers, research/open_questions.md) is an open gate-1
            item — this view is the archiver's display, not a validated signal.
          </div>
        </div>
      )}
    </div>
  );
}
