// FredMacroView — US macro regime cluster (#/data/fred-macro).
// server/fredMacro.ts (GATE 1 DATA passed 2026-07-05, /api/data/macro,
// DATA STREAM EXPANSION stream #3; FRED_API_KEY set in Railway 2026-07-05)
// has archived 28 public FRED series (rates/curve, financial stress, labor,
// inflation, activity, money & liquidity, commodities & dollar) with no
// client view until now — this closes that gap, the same shape as the
// eu-macro/VIX-term-structure precedents. RAW display only (kind:"raw"):
// every value is FRED's own currently-published level; this is a REGIME
// INPUT feed per fredMacro.ts's own docstring, never a standalone traded
// signal, and license:"restricted" series (VIX, HY OAS, UMich sentiment)
// are excluded server-side before the payload ever reaches this page — see
// server/fredMacro.ts's LICENSING note. Tile grid is the VIX/eu-macro
// precedent; category section headers reuse the data-quality dashboard's
// vt-quality-section pattern (7 categories vs. eu-macro's single flat
// group of 5, so grouping stays readable at 28 series). Per-series history
// picker is the eu-macro/ATS-OTC leaderboard tab precedent. Reuses
// vt-filings-*/vt-gridstress-tile*/vt-quality-section* CSS only — no new
// styles needed.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Percent } from "lucide-react";

interface FredObsPoint { d: string; v: number }
interface FredSeriesSnapshot {
  id: string;
  label: string;
  cadence: "daily" | "weekly" | "monthly";
  unit: string;
  latest: FredObsPoint | null;
  prev: FredObsPoint | null;
  history: FredObsPoint[];
}
interface FredMacroPayload {
  kind: string;
  enabled?: boolean;
  warming_up?: boolean;
  reason?: string;
  source?: string;
  attribution?: string;
  time?: number;
  note?: string;
  series: FredSeriesSnapshot[];
}

// Category grouping mirrors server/fredMacro.ts's FRED_SERIES section
// comments — display-only, no ladder or licensing logic lives here.
const CATEGORY: Record<string, string> = {
  DGS3MO: "Rates & curve", DGS2: "Rates & curve", DGS10: "Rates & curve", DGS30: "Rates & curve",
  T10Y2Y: "Rates & curve", T10Y3M: "Rates & curve", FEDFUNDS: "Rates & curve", SOFR: "Rates & curve",
  T5YIE: "Rates & curve", T10YIE: "Rates & curve",
  STLFSI4: "Financial stress", NFCI: "Financial stress",
  ICSA: "Labor", CCSA: "Labor", UNRATE: "Labor", PAYEMS: "Labor",
  CPIAUCSL: "Inflation", CPILFESL: "Inflation", PCEPILFE: "Inflation",
  INDPRO: "Activity", HOUST: "Activity", PERMIT: "Activity", RSAFS: "Activity",
  M2SL: "Money & liquidity", WALCL: "Money & liquidity", RRPONTSYD: "Money & liquidity",
  DCOILWTICO: "Commodities & dollar", DTWEXBGS: "Commodities & dollar",
};

function fmtBillions(b: number, forceSign: boolean): string {
  const neg = b < 0;
  const abs = Math.abs(b);
  const sign = neg ? "-" : forceSign ? "+" : "";
  if (abs >= 1000) return `${sign}$${(abs / 1000).toFixed(2)}T`;
  if (abs >= 1) return `${sign}$${abs.toFixed(1)}B`;
  return `${sign}$${(abs * 1000).toFixed(0)}M`;
}

function normDollars(v: number, unit: string): number {
  return unit === "$M" ? v / 1000 : v; // $B already in billions
}

function fmtVal(v: number, unit: string): string {
  switch (unit) {
    case "%": return `${v.toFixed(2)}%`;
    case "index": return v.toFixed(2);
    case "claims": return `${(v / 1000).toFixed(0)}K`;
    case "thousands": return `${(v / 1000).toFixed(2)}M`;
    case "$/bbl": return `$${v.toFixed(2)}`;
    case "$M":
    case "$B": return fmtBillions(normDollars(v, unit), false);
    default: return v.toLocaleString();
  }
}

function fmtDelta(latest: FredObsPoint | null, prev: FredObsPoint | null, unit: string): string {
  if (!latest || !prev) return "—";
  const d = latest.v - prev.v;
  const sign = d >= 0 ? "+" : "";
  switch (unit) {
    case "%": return `${sign}${d.toFixed(2)}pp`;
    case "index": return `${sign}${d.toFixed(2)}`;
    case "claims": return `${sign}${(d / 1000).toFixed(0)}K`;
    case "thousands": return `${sign}${(d / 1000).toFixed(2)}M`;
    case "$/bbl": return `${sign}$${d.toFixed(2)}`;
    case "$M":
    case "$B": return fmtBillions(normDollars(d, unit), true);
    default: return `${sign}${d.toLocaleString()}`;
  }
}

export default function FredMacroView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FredMacroPayload | null>(null);
  const [error, setError] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/macro");
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
    () => series.find((s) => s.id === selected) ?? series.find((s) => s.latest) ?? series[0] ?? null,
    [series, selected],
  );
  const history = (active?.history ?? []).slice().reverse().slice(0, 30);

  const groups = useMemo(() => {
    const out: { category: string; items: FredSeriesSnapshot[] }[] = [];
    for (const s of series) {
      const cat = CATEGORY[s.id] || "Other";
      const last = out[out.length - 1];
      if (last && last.category === cat) last.items.push(s);
      else out.push({ category: cat, items: [s] });
    }
    return out;
  }, [series]);

  return (
    <div className="vt-filings-page" role="region" aria-label="US macro regime cluster">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Percent size={16} />
        <div>
          <div className="vt-filings-title">US macro regime cluster — FRED</div>
          <div className="vt-filings-sub">
            28 series — rates/curve, financial stress, labor, inflation, activity, money &amp; liquidity, commodities
            &amp; dollar — RAW regime input, never a standalone signal ·{" "}
            {series.some((s) => s.latest) ? "live" : data?.warming_up ? "warming up…" : data?.enabled === false ? "not configured" : "loading…"} ·{" "}
            <a href="https://fred.stlouisfed.org" target="_blank" rel="noreferrer">FRED — St. Louis Fed <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.enabled === false && (
        <div className="vt-filings-state">{data.reason || "FRED_API_KEY not set server-side."}</div>
      )}
      {!error && data?.enabled !== false && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && data?.enabled !== false && !data?.warming_up && data && series.length === 0 && (
        <div className="vt-filings-state">No series archived yet.</div>
      )}

      {!error && series.length > 0 && (
        <div className="vt-shortvol-body">
          {groups.map((g) => (
            <section className="vt-quality-section" key={g.category}>
              <div className="vt-quality-section-head">
                <span>{g.category}</span>
              </div>
              <div className="vt-gridstress-grid">
                {g.items.map((s) => (
                  <div className="vt-gridstress-tile" key={s.id}>
                    <div className="vt-gridstress-tile-label">{s.label}</div>
                    <div className="vt-gridstress-tile-value">{s.latest ? fmtVal(s.latest.v, s.unit) : "—"}</div>
                    <div className="vt-gridstress-tile-sub">
                      {s.cadence} · Δ {fmtDelta(s.latest, s.prev, s.unit)}
                      {s.latest ? ` · as of ${s.latest.d}` : ""}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}

          <div className="vt-filings-sub">
            {data?.note || "REGIME INPUT feed (never a direct signal). Values revise in place — our archive keeps every vintage as-seen."}
          </div>

          <div className="vt-filings-filters" role="group" aria-label="Series history">
            {series.map((s) => (
              <button
                key={s.id}
                type="button"
                className="vt-filings-filter"
                aria-pressed={active?.id === s.id}
                disabled={!s.history.length}
                style={active?.id === s.id ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                onClick={() => setSelected(s.id)}
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
            {data?.attribution || "Source: FRED, Federal Reserve Bank of St. Louis"} · public-domain US government/Fed
            series only — third-party copyrighted series (VIX, HY OAS, UMich sentiment) stay internal, never surfaced here
          </div>
        </div>
      )}
    </div>
  );
}
