// EuMacroView — European macro regime cluster (#/data/eu-macro).
// server/euMacro.ts (GATE 1 DATA passed 2026-07-07, /api/data/eu-macro,
// DATACORE MAXIMUS census build #7) has archived EUR/USD, €STR, the
// Eurosystem balance sheet, EA20 industrial production, and the 10Y Bund
// yield since its build with no client view until now — this closes that
// gap, the same shape as the FRED-style precedents (VIX term structure,
// crop conditions). RAW display only (kind:"raw"): every value is the
// source's own published level; this is a REGIME INPUT feed, never a
// standalone traded signal (per the module's own docstring, restated here
// rather than implied). Tile grid is the VIX term-structure precedent;
// the per-series history picker is the ATS/OTC leaderboard tab precedent
// (vt-filings-filters). Reuses vt-filings-*/vt-gridstress-tile* CSS only —
// no new styles needed.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Euro } from "lucide-react";

interface EuObsPoint { d: string; v: number }
interface EuSeriesSnapshot {
  key: string;
  source: "ecb" | "eurostat" | "bbk";
  label: string;
  cadence: "daily" | "weekly" | "monthly";
  unit: string;
  attribution: string;
  latest: EuObsPoint | null;
  prev: EuObsPoint | null;
  history: EuObsPoint[];
}
interface EuMacroPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  note?: string;
  series: EuSeriesSnapshot[];
}

function fmtVal(v: number, unit: string): string {
  if (unit === "EUR millions") return `€${(v / 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}bn`;
  if (unit === "%") return `${v.toFixed(2)}%`;
  if (unit === "USD") return v.toFixed(4);
  if (unit === "index") return v.toFixed(1);
  return v.toLocaleString();
}

function fmtDelta(latest: EuObsPoint | null, prev: EuObsPoint | null, unit: string): string {
  if (!latest || !prev) return "—";
  const d = latest.v - prev.v;
  const sign = d >= 0 ? "+" : "";
  if (unit === "EUR millions") return `${sign}${(d / 1000).toFixed(0)}bn`;
  if (unit === "%") return `${sign}${d.toFixed(2)}pp`;
  if (unit === "USD") return `${sign}${d.toFixed(4)}`;
  if (unit === "index") return `${sign}${d.toFixed(1)}`;
  return `${sign}${d.toLocaleString()}`;
}

const SOURCE_LABEL: Record<EuSeriesSnapshot["source"], string> = {
  ecb: "ECB",
  eurostat: "Eurostat",
  bbk: "Bundesbank",
};

export default function EuMacroView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<EuMacroPayload | null>(null);
  const [error, setError] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/eu-macro");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const series = data?.series ?? [];
  const active = useMemo(
    () => series.find((s) => s.key === selected) ?? series.find((s) => s.latest) ?? series[0] ?? null,
    [series, selected],
  );
  const history = (active?.history ?? []).slice().reverse().slice(0, 30);

  return (
    <div className="vt-filings-page" role="region" aria-label="European macro regime cluster">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Euro size={16} />
        <div>
          <div className="vt-filings-title">European macro cluster — ECB · Eurostat · Bundesbank</div>
          <div className="vt-filings-sub">
            EUR/USD, €STR, Eurosystem balance sheet, EA20 industrial production, 10Y Bund — RAW regime input, never a
            standalone signal ·{" "}
            {series.some((s) => s.latest) ? "live" : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://data.ecb.europa.eu" target="_blank" rel="noreferrer">ECB Data Portal <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && !data?.warming_up && data && series.length === 0 && (
        <div className="vt-filings-state">No series archived yet.</div>
      )}

      {!error && series.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-gridstress-grid">
            {series.map((s) => (
              <div className="vt-gridstress-tile" key={s.key}>
                <div className="vt-gridstress-tile-label">{s.label}</div>
                <div className="vt-gridstress-tile-value">{s.latest ? fmtVal(s.latest.v, s.unit) : "—"}</div>
                <div className="vt-gridstress-tile-sub">
                  {SOURCE_LABEL[s.source]} · {s.cadence} · Δ {fmtDelta(s.latest, s.prev, s.unit)}
                  {s.latest ? ` · as of ${s.latest.d}` : ""}
                </div>
              </div>
            ))}
          </div>

          <div className="vt-filings-sub">
            {data?.note || "REGIME INPUT feed (never a direct signal). Sources revise in place — the archive keeps every vintage as-seen."}
          </div>

          <div className="vt-filings-filters" role="group" aria-label="Series history">
            {series.map((s) => (
              <button
                key={s.key}
                type="button"
                className="vt-filings-filter"
                aria-pressed={active?.key === s.key}
                disabled={!s.history.length}
                style={active?.key === s.key ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                onClick={() => setSelected(s.key)}
              >
                {s.label}
              </button>
            ))}
          </div>

          {active && history.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th className="num">{active.label} ({active.unit})</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((row) => (
                    <tr key={row.d}>
                      <td data-l="Date">{row.d}</td>
                      <td data-l={active.label} className="num">{fmtVal(row.v, active.unit)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {active && history.length === 0 && (
            <div className="vt-filings-state">No history archived yet for {active.label}.</div>
          )}

          <div className="vt-filings-sub vt-streams-foot">
            {active ? active.attribution : ""} · per-series attribution required for reuse, carried verbatim from each source's license
          </div>
        </div>
      )}
    </div>
  );
}
