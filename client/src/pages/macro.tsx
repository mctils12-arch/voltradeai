// MacroView — the REGIME INPUT dashboard (#/data/macro). Both
// server/fredMacro.ts (31 US series, gate-1 PASSED 2026-07-05, 10/10
// exact match vs FRED's own fredgraph.csv export) and server/euMacro.ts
// (5-series European cluster, gate-1 PASSED 2026-07-07, all live-verified
// against the pre-deploy workup) have been serving /api/data/macro and
// /api/data/eu-macro since their respective build sessions with NO client
// view — this is that follow-up (PRODUCT session, 2026-08-06), same
// shipped-data-no-UI gap class that midas.tsx/atsSummary.tsx closed
// earlier. RAW display only: the module docstrings are explicit that this
// is the REGIME CONDITIONING feed, never traded or sold as a standalone
// signal — the UI states that plainly rather than implying predictive
// value. Reuses the generic .vt-filings-*/.vt-shortvol-* shell (crop-
// conditions/cot precedent); only the card-grid + sparkline classes below
// are new (index.css .vt-macro-*).
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, LineChart, Search } from "lucide-react";

interface SeriesPoint { d: string; v: number }
interface FredSeries {
  id: string; label: string; cadence: "daily" | "weekly" | "monthly"; unit: string;
  latest: SeriesPoint | null; prev: SeriesPoint | null; history: SeriesPoint[];
}
interface FredPayload {
  kind: string; enabled?: boolean; reason?: string; warming_up?: boolean;
  source?: string; attribution?: string; time?: number; note?: string;
  series: FredSeries[];
}
interface EuSeries {
  key: string; source: "ecb" | "eurostat" | "bbk"; label: string;
  cadence: "daily" | "weekly" | "monthly"; unit: string; attribution: string;
  latest: SeriesPoint | null; prev: SeriesPoint | null; history: SeriesPoint[];
}
interface EuPayload {
  kind: string; warming_up?: boolean; source?: string; attribution?: string;
  time?: number; note?: string; series: EuSeries[];
}

// Unified shape both feeds map to, so one card grid renders both.
interface Card {
  key: string; group: "US — FRED" | "Europe"; label: string;
  unit: string; cadence: string; attribution: string;
  latest: SeriesPoint | null; prev: SeriesPoint | null; history: SeriesPoint[];
}

/** Minimal inline sparkline — mirrors cot.tsx's/attention.tsx's, no charting dependency. */
function Sparkline({ points, width = 160, height = 36 }: { points: Array<number | null>; width?: number; height?: number }) {
  const vals = points.map((v, i) => (v == null ? null : { i, v })).filter((p): p is { i: number; v: number } => p != null);
  if (vals.length < 2) return <div className="vt-shortvol-spark-empty">not enough history yet</div>;
  const lo = Math.min(...vals.map((p) => p.v));
  const hi = Math.max(...vals.map((p) => p.v));
  const span = hi - lo || 1;
  const x = (i: number) => (i / (points.length - 1)) * (width - 4) + 2;
  const y = (v: number) => height - 2 - ((v - lo) / span) * (height - 4);
  const d = vals.map((p, k) => `${k === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  const rising = vals[vals.length - 1].v >= vals[0].v;
  return (
    <svg className="vt-macro-spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
         aria-label={`trend from ${vals[0].v} to ${vals[vals.length - 1].v}`}>
      <path d={d} fill="none" stroke={rising ? "var(--accent-green, #4ade80)" : "var(--accent-orange, #ff8a5a)"} strokeWidth={1.6} />
    </svg>
  );
}

function fmtVal(v: number | null | undefined, unit: string): string {
  if (v == null) return "—";
  if (unit === "%") return `${v.toFixed(2)}%`;
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(2);
}

/** A 2-decimal delta on a small-magnitude series (e.g. EUR/USD ~1.09) can
 *  round a real, nonzero move to "+0.00" — misleading with the up/down
 *  color still attached. Widens precision only as far as needed to show a
 *  genuinely nonzero digit, never further. */
function fmtDelta(delta: number | null, unit: string): string | null {
  if (delta == null || delta === 0) return null;
  const sign = delta > 0 ? "+" : "";
  if (unit === "%") return `${sign}${delta.toFixed(2)}%`;
  if (Math.abs(delta) >= 1000) return `${sign}${delta.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  let decimals = 2;
  while (decimals < 6 && Number(delta.toFixed(decimals)) === 0) decimals++;
  return `${sign}${delta.toFixed(decimals)}`;
}

export default function MacroView({ onBack }: { onBack: () => void }) {
  const [fred, setFred] = useState<FredPayload | null>(null);
  const [eu, setEu] = useState<EuPayload | null>(null);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const [f, e] = await Promise.all([
          fetch("/api/data/macro").then((r) => r.json()),
          fetch("/api/data/eu-macro").then((r) => r.json()),
        ]);
        if (!stop) { setFred(f); setEu(e); }
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const cards = useMemo<Card[]>(() => {
    const out: Card[] = [];
    for (const s of fred?.series ?? []) {
      out.push({
        key: `fred:${s.id}`, group: "US — FRED", label: s.label, unit: s.unit,
        cadence: s.cadence, attribution: fred?.attribution || "FRED",
        latest: s.latest, prev: s.prev, history: s.history,
      });
    }
    for (const s of eu?.series ?? []) {
      out.push({
        key: `eu:${s.key}`, group: "Europe", label: s.label, unit: s.unit,
        cadence: s.cadence, attribution: s.attribution,
        latest: s.latest, prev: s.prev, history: s.history,
      });
    }
    return out;
  }, [fred, eu]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return cards;
    return cards.filter((c) => c.label.toLowerCase().includes(q));
  }, [cards, query]);

  const usCards = filtered.filter((c) => c.group === "US — FRED");
  const euCards = filtered.filter((c) => c.group === "Europe");
  const fredDisabled = fred?.enabled === false;
  const nothingLoadedYet = !fred && !eu && !error;

  return (
    <div className="vt-filings-page" role="region" aria-label="Macro regime series">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <LineChart size={16} />
        <div>
          <div className="vt-filings-title">Macro regime series</div>
          <div className="vt-filings-sub">
            REGIME INPUT feed — conditions other readings, never traded or sold alone · no predictive claim ·{" "}
            <a href="https://fred.stlouisfed.org" target="_blank" rel="noreferrer">FRED <ExternalLink size={11} /></a>
            {" "}+ ECB / Eurostat / Bundesbank
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && nothingLoadedYet && <div className="vt-filings-state">Loading…</div>}

      {!error && !nothingLoadedYet && (
        <div className="vt-shortvol-body">
          {fredDisabled && (
            <div className="vt-filings-state">US series unavailable: {fred?.reason || "FRED_API_KEY not set"}. European cluster below is keyless and unaffected.</div>
          )}
          {!fredDisabled && fred?.warming_up && (
            <div className="vt-filings-state">US series: first poll still in progress — check back shortly.</div>
          )}
          {eu?.warming_up && (
            <div className="vt-filings-state">European cluster: first poll still in progress — check back shortly.</div>
          )}

          <form className="vt-filings-filters" onSubmit={(e) => e.preventDefault()}>
            <input
              type="search"
              className="vt-earnings-search"
              placeholder="Filter series (e.g. 10-Year, CPI, jobless)…"
              aria-label="Filter macro series by name"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <span className="vt-filings-count">{filtered.length} of {cards.length} series</span>
          </form>

          {cards.length > 0 && filtered.length === 0 && (
            <div className="vt-filings-state">No series match "{query}".</div>
          )}

          {usCards.length > 0 && (
            <div className="vt-macro-section">
              <div className="vt-filings-sub vt-macro-sectionhead">United States — {usCards.length} series</div>
              <div className="vt-macro-grid">
                {usCards.map((c) => <MacroCard key={c.key} card={c} />)}
              </div>
            </div>
          )}

          {euCards.length > 0 && (
            <div className="vt-macro-section">
              <div className="vt-filings-sub vt-macro-sectionhead">Europe — {euCards.length} series</div>
              <div className="vt-macro-grid">
                {euCards.map((c) => <MacroCard key={c.key} card={c} />)}
              </div>
            </div>
          )}

          <div className="vt-filings-sub vt-streams-foot">
            values as currently published by each source; sources revise history in place — our archive keeps every
            vintage as-seen (point-in-time). Third-party-copyrighted series (VIX, ICE BofA credit spreads, UMich
            sentiment) are used internally for regime classification but excluded from this display per licensing terms.
          </div>
        </div>
      )}
    </div>
  );
}

function MacroCard({ card }: { card: Card }) {
  const { latest, prev, unit } = card;
  const delta = latest?.v != null && prev?.v != null ? latest.v - prev.v : null;
  const deltaLabel = fmtDelta(delta, unit);
  return (
    <div className="vt-macro-card">
      <div className="vt-macro-cardhead">
        <span className="vt-macro-label">{card.label}</span>
        <span className="vt-macro-cadence">{card.cadence}</span>
      </div>
      <div className="vt-macro-valrow">
        <span className="vt-macro-val">{fmtVal(latest?.v, unit)}</span>
        {deltaLabel && (
          <span className={`vt-macro-delta ${delta! >= 0 ? "up" : "down"}`}>{deltaLabel}</span>
        )}
      </div>
      <Sparkline points={card.history.map((h) => h.v)} />
      <div className="vt-macro-foot">
        {latest?.d ? `as of ${latest.d}` : "no reading yet"} · {card.attribution}
      </div>
    </div>
  );
}
