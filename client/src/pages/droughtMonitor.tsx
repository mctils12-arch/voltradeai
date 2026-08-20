// DroughtMonitorView — U.S. Drought Monitor weekly severity, CONUS + 8
// ag/water belt states (#/data/drought). server/droughtMonitor.ts shipped
// the API-only route (/api/data/drought, BUILD ORDER 2 #5, v1.0.118,
// 2026-07-05) with no client view — this is that follow-up, the sixth
// "shipped-data-no-UI" gap closed this way (fleet_utilization/grid_demand/
// occ_volume/crop_conditions precedent). Not a spatial layer (state/CONUS
// aggregates, no per-row lat/lon) so it launches from the panel-top
// "Streams inventory" list like crop-conditions/grid-stress, not the
// layer-toggle sidebar. RAW display only: this root's own filed hypothesis
// (open_questions.md BUILD ORDER 6 #5 — belt-weighted drought delta vs.
// corn/soy futures forward returns) has NOT been gate-1 or gate-2 tested
// yet, stated here verbatim rather than left implicit. Reuses the
// .vt-cropcond-* bar/legend classes (cropConditions.tsx precedent, itself
// a stacked-severity-class bar) — zero new CSS.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Droplet } from "lucide-react";

interface DroughtRec {
  aoi: string;
  map_date: string;
  none: number;
  d0: number;
  d1: number;
  d2: number;
  d3: number;
  d4: number;
  dsci: number;
  valid_start: string | null;
  valid_end: string | null;
  rt: string;
}
interface DroughtPayload {
  kind?: string;
  source?: string;
  attribution?: string;
  time?: number;
  count?: number;
  note?: string;
  warming_up?: boolean;
  drought?: DroughtRec[];
}

// Display order matches droughtMonitor.ts's own DROUGHT_AOIS build order
// (CONUS first, then the 8 belt states as filed) rather than an alphabetical
// re-sort, so the page reads the same order the archive was built in.
const AOI_ORDER = ["CONUS", "IA", "IL", "NE", "KS", "MN", "TX", "OK", "CA"];
const AOI_LABEL: Record<string, string> = {
  CONUS: "CONUS (lower 48)", IA: "Iowa", IL: "Illinois", NE: "Nebraska", KS: "Kansas",
  MN: "Minnesota", TX: "Texas", OK: "Oklahoma", CA: "California",
};

// Official USDM severity-class colors (droughtmonitor.unl.edu's own map
// legend), applied to INCREMENTAL bands (each D-level here is "in this class
// only," not the cumulative area-or-worse percentage the API returns).
const BAND_COLOR: Record<string, string> = {
  none: "#2f7a3d",
  d0: "#ffff00",
  d1: "#fcd37f",
  d2: "#ffaa00",
  d3: "#e60000",
  d4: "#730000",
};
const BAND_LABEL: Record<string, string> = {
  none: "None", d0: "D0 Abnormally dry", d1: "D1 Moderate", d2: "D2 Severe", d3: "D3 Extreme", d4: "D4 Exceptional",
};

function latestByAoi(rows: DroughtRec[]): Map<string, DroughtRec> {
  const out = new Map<string, DroughtRec>();
  for (const r of rows) {
    const cur = out.get(r.aoi);
    if (!cur || r.map_date > cur.map_date) out.set(r.aoi, r);
  }
  return out;
}

// Nearest prior reading 21-35 days back (a ~4-week trend window tolerant of
// the weekly cadence landing a day or two off an exact 28), used for an
// honest DSCI delta rather than a fabricated trend.
function fourWeekAgo(rows: DroughtRec[], aoi: string, latest: DroughtRec): DroughtRec | null {
  const latestMs = Date.parse(latest.map_date + "T00:00:00Z");
  let best: DroughtRec | null = null;
  let bestDiff = Infinity;
  for (const r of rows) {
    if (r.aoi !== aoi || r.map_date === latest.map_date) continue;
    const days = (latestMs - Date.parse(r.map_date + "T00:00:00Z")) / 86400_000;
    if (days < 21 || days > 35) continue;
    const diff = Math.abs(days - 28);
    if (diff < bestDiff) { bestDiff = diff; best = r; }
  }
  return best;
}

function AoiBar({ label, r, priorDsci }: { label: string; r: DroughtRec; priorDsci: number | null }) {
  const bands: Array<{ key: string; v: number }> = [
    { key: "none", v: r.none },
    { key: "d0", v: Math.max(0, r.d0 - r.d1) },
    { key: "d1", v: Math.max(0, r.d1 - r.d2) },
    { key: "d2", v: Math.max(0, r.d2 - r.d3) },
    { key: "d3", v: Math.max(0, r.d3 - r.d4) },
    { key: "d4", v: Math.max(0, r.d4) },
  ];
  const delta = priorDsci != null ? r.dsci - priorDsci : null;
  return (
    <div className="vt-cropcond-commodity">
      <div className="vt-cropcond-commodity-head">
        <span className="vt-cropcond-commodity-name">{label}</span>
        <span className="vt-cropcond-commodity-gpe">
          DSCI {r.dsci.toFixed(0)}/500
          {delta != null && (
            <span style={{ marginLeft: 6, color: delta > 0 ? "#e6555a" : delta < 0 ? "#4ade80" : "var(--text-tertiary)" }}>
              ({delta > 0 ? "+" : ""}{delta.toFixed(0)} vs 4wk ago)
            </span>
          )}
        </span>
      </div>
      <div className="vt-cropcond-bar" role="img"
           aria-label={`${label} drought coverage: ${bands.map((b) => `${BAND_LABEL[b.key]} ${b.v.toFixed(0)}%`).join(", ")}`}>
        {bands.map((b) => b.v > 0 ? (
          <div key={b.key} className="vt-cropcond-seg" style={{ width: `${b.v}%`, background: BAND_COLOR[b.key] }}
               title={`${BAND_LABEL[b.key]}: ${b.v.toFixed(1)}%`} />
        ) : null)}
      </div>
      <div className="vt-filings-sub" style={{ marginTop: 2 }}>week of {r.map_date}</div>
    </div>
  );
}

export default function DroughtMonitorView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<DroughtPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/drought");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = data?.drought ?? [];
  const latest = latestByAoi(rows);
  const aois = AOI_ORDER.filter((a) => latest.has(a));

  return (
    <div className="vt-filings-page" role="region" aria-label="U.S. Drought Monitor weekly severity">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Droplet size={16} />
        <div>
          <div className="vt-filings-title">Drought severity — U.S. Drought Monitor</div>
          <div className="vt-filings-sub">
            weekly D0-D4 area coverage, CONUS + 8 ag/water belt states — RAW, no predictive claim ·{" "}
            {aois.length > 0 ? `week of ${latest.get(aois[0])!.map_date}` : error ? "" : "loading…"} ·{" "}
            <a href="https://droughtmonitor.unl.edu" target="_blank" rel="noreferrer">U.S. Drought Monitor <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data && data.warming_up && (
        <div className="vt-filings-state">Warming up — first archive scan in progress.</div>
      )}
      {!error && data && !data.warming_up && aois.length === 0 && (
        <div className="vt-filings-state">No readings in the current archive window.</div>
      )}

      {!error && aois.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data?.note ? data.note + " " : ""}
            DSCI (Drought Severity and Coverage Index, 0-500) is the USDM's own published area-weighted
            severity index, derived here from the published D0-D4 area percentages. Bars show each AOI's
            share of area in ONLY that severity class (not the cumulative area-or-worse figure the API
            returns) using the USDM's own map legend colors.
          </div>
          <div className="vt-filings-sub">
            HYPOTHESIS (filed, not yet tested): belt-weighted drought-severity deltas over these ag states
            lead ag-commodity (grains, cattle) forward returns by weeks. GATE 1 (spot-check vs. the
            published national map) and GATE 2 (drought delta vs. corn/soy futures forward returns) have
            NOT been run for this root — nothing on this page is a validated signal, display only.
          </div>

          <div className="vt-cropcond-body">
            {aois.map((a) => (
              <AoiBar key={a} label={AOI_LABEL[a] ?? a} r={latest.get(a)!} priorDsci={fourWeekAgo(rows, a, latest.get(a)!)?.dsci ?? null} />
            ))}
          </div>

          <div className="vt-cropcond-legend" style={{ marginTop: 4 }}>
            {(["none", "d0", "d1", "d2", "d3", "d4"] as const).map((k) => (
              <span key={k} className="vt-cropcond-legend-item">
                <i className="vt-cropcond-legend-swatch" style={{ background: BAND_COLOR[k] }} />
                {BAND_LABEL[k]}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
