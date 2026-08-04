// CropConditionsView — USDA NASS weekly crop condition ratings
// (#/data/crop-conditions). DATACORE MAXIMUS Phase 0b shipped the
// API-only route (/api/data/crop-conditions, server/cropConditions.ts,
// v1.0.164, 2026-07-06) with no client view; GATE 1 PASSED same-day
// 2026-08-04 (v1.0.589, 10/10 exact match vs USDA's own published Crop
// Progress bulletin) — this is that follow-up, named directly in
// experiments.md's NEXT queue ("best suited to a chart/table view
// rather than a map point layer"). RAW display of NASS's own weekly
// national condition percentages — no predictive claim; the
// condition-delta-vs-futures hypothesis stays gate-2-locked (see the
// module docstring in cropConditions.ts). Reuses the .vt-filings-*
// shell (ATS/grid-stress precedent) — only the bar-chart classes below
// are new.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Sprout } from "lucide-react";

interface ConditionRow {
  commodity: string;      // CORN | SOYBEANS
  week_ending: string;    // YYYY-MM-DD
  item: string;           // short_desc verbatim, carries the condition class
  pct: number | null;
  rt: string;
}
interface CropConditionsPayload {
  kind: string;
  enabled?: boolean;
  reason?: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: string;
  latest_week?: string;
  count: number;
  note?: string;
  rows: ConditionRow[];
}

// Worst-to-best, matching NASS's own published bulletin ordering.
const CLASSES = ["VERY POOR", "POOR", "FAIR", "GOOD", "EXCELLENT"] as const;
type ConditionClass = typeof CLASSES[number];
// Diverging red -> green ramp (worst -> best); not a map symbol, a
// chart color scale, so SYMBOLS NOT DOTS doesn't apply here.
const CLASS_COLOR: Record<ConditionClass, string> = {
  "VERY POOR": "#ff5a6e",
  "POOR": "#ff8a5a",
  "FAIR": "#fbb24c",
  "GOOD": "#a3d977",
  "EXCELLENT": "#4ade80",
};

// "VERY POOR" must be checked before "POOR" (substring collision).
function classOf(item: string): ConditionClass | null {
  const up = item.toUpperCase();
  for (const c of CLASSES) {
    if (up.includes(c)) return c;
  }
  return null;
}

function CommodityBar({ commodity, rows }: { commodity: string; rows: ConditionRow[] }) {
  const byClass = new Map<ConditionClass, number>();
  for (const r of rows) {
    if (r.pct == null) continue;
    const c = classOf(r.item);
    if (c) byClass.set(c, r.pct);
  }
  const total = CLASSES.reduce((s, c) => s + (byClass.get(c) ?? 0), 0);
  const goodPlusExcellent = (byClass.get("GOOD") ?? 0) + (byClass.get("EXCELLENT") ?? 0);
  return (
    <div className="vt-cropcond-commodity">
      <div className="vt-cropcond-commodity-head">
        <span className="vt-cropcond-commodity-name">{commodity}</span>
        <span className="vt-cropcond-commodity-gpe">
          {byClass.size ? `${goodPlusExcellent.toFixed(0)}% good/excellent` : "no reading"}
        </span>
      </div>
      <div className="vt-cropcond-bar" role="img"
           aria-label={`${commodity} condition: ${CLASSES.map((c) => `${c.toLowerCase()} ${(byClass.get(c) ?? 0).toFixed(0)}%`).join(", ")}`}>
        {CLASSES.map((c) => {
          const v = byClass.get(c) ?? 0;
          if (v <= 0) return null;
          return (
            <div key={c} className="vt-cropcond-seg" style={{ width: `${v}%`, background: CLASS_COLOR[c] }}
                 title={`${c}: ${v}%`} />
          );
        })}
      </div>
      <div className="vt-cropcond-legend">
        {CLASSES.map((c) => (
          <span key={c} className="vt-cropcond-legend-item">
            <i className="vt-cropcond-legend-swatch" style={{ background: CLASS_COLOR[c] }} />
            {c.charAt(0) + c.slice(1).toLowerCase()}
            <b className="vt-cropcond-legend-val">{byClass.has(c) ? `${byClass.get(c)!.toFixed(0)}%` : "—"}</b>
          </span>
        ))}
      </div>
      {total > 0 && Math.abs(total - 100) > 1.5 && (
        <div className="vt-cropcond-note">
          classes sum to {total.toFixed(0)}% (NASS rounds each class independently — not a data error)
        </div>
      )}
    </div>
  );
}

export default function CropConditionsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<CropConditionsPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/crop-conditions");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const commodities = Array.from(new Set((data?.rows ?? []).map((r) => r.commodity))).sort();

  return (
    <div className="vt-filings-page" role="region" aria-label="USDA crop condition ratings">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Sprout size={16} />
        <div>
          <div className="vt-filings-title">Crop conditions — USDA NASS</div>
          <div className="vt-filings-sub">
            national weekly condition ratings, corn &amp; soybeans — RAW, no predictive claim ·{" "}
            {data?.latest_week ? `week of ${data.latest_week}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://quickstats.nass.usda.gov" target="_blank" rel="noreferrer">USDA NASS QuickStats <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.enabled === false && (
        <div className="vt-filings-state">{data.reason}</div>
      )}
      {!error && data?.enabled !== false && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && data?.enabled !== false && !data?.warming_up && data && commodities.length === 0 && (
        <div className="vt-filings-state">No rows for the current week yet.</div>
      )}

      {!error && commodities.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-cropcond-body">
            {commodities.map((c) => (
              <CommodityBar key={c} commodity={c} rows={(data!.rows ?? []).filter((r) => r.commodity === c)} />
            ))}
          </div>
          {data?.note && <div className="vt-filings-sub vt-streams-foot">{data.note}</div>}
        </div>
      )}
    </div>
  );
}
