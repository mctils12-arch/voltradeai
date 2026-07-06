// ShortVolView — the full FINRA daily short-sale-volume view
// (#/data/short-volume). DESIGN.md-governed: readable at 390/768/1440
// (table on desktop, stacked cards on phone via the shared .vt-filings-table
// CSS), designed empty/loading/error states, theme tokens only.
//
// Data: /api/data/short-volume (today's live top-ratio-mover list, computed
// at poll time — never on the request path) + /api/data/short-volume/history
// (market-wide trend log, and on demand a single ticker's ratio series read
// directly from the 2024-06-17+ day archive). RAW flow proxy — NOT short
// interest, no predictive claim; short-ratio extremes are a BUILD ORDER 5 #1
// gate-2 hypothesis, still open.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, TrendingUp, Search } from "lucide-react";

interface TopRatioRow { symbol: string; short_ratio: number; total_vol: number; market: string; }
interface Summary {
  date: string; symbols: number; agg_short_ratio: number | null;
  top_ratio: TopRatioRow[]; floor_total_vol: number; top_cap: number;
}
interface TrendPoint { date: string; symbols: number; agg_short_ratio: number | null; }
interface SymbolPoint { date: string; short_vol: number; total_vol: number; short_ratio: number | null; market: string; }

const pct = (r: number | null) => (r == null ? "—" : `${(r * 100).toFixed(1)}%`);

/** Minimal inline sparkline — no charting dependency for a single trend
 *  line. Values that are null (a day the aggregate couldn't be computed)
 *  are skipped, not zero-filled. */
function Sparkline({ points, width = 240, height = 40 }: { points: Array<number | null>; width?: number; height?: number }) {
  const vals = points.map((v, i) => (v == null ? null : { i, v })).filter((p): p is { i: number; v: number } => p != null);
  if (vals.length < 2) return <div className="vt-shortvol-spark-empty">not enough history yet</div>;
  const lo = Math.min(...vals.map((p) => p.v));
  const hi = Math.max(...vals.map((p) => p.v));
  const span = hi - lo || 1;
  const x = (i: number) => (i / (points.length - 1)) * (width - 4) + 2;
  const y = (v: number) => height - 2 - ((v - lo) / span) * (height - 4);
  const d = vals.map((p, k) => `${k === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  return (
    <svg className="vt-shortvol-spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
         aria-label={`trend from ${pct(vals[0].v)} to ${pct(vals[vals.length - 1].v)}`}>
      <path d={d} fill="none" stroke="var(--accent-bright)" strokeWidth={1.6} />
    </svg>
  );
}

export default function ShortVolView({ onBack }: { onBack: () => void }) {
  const [today, setToday] = useState<Summary | null>(null);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [symbolResult, setSymbolResult] = useState<{ symbol: string; series: SymbolPoint[]; note?: string } | null>(null);
  const [symbolLoading, setSymbolLoading] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const [live, hist] = await Promise.all([
          fetch("/api/data/short-volume").then((r) => r.json()),
          fetch("/api/data/short-volume/history?days=60").then((r) => r.json()),
        ]);
        if (stop) return;
        if (live.summary) setToday(live.summary);
        setTrend(hist.trend || []);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const lookupSymbol = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const sym = query.trim();
    if (!sym) return;
    setSymbolLoading(true);
    try {
      const r = await fetch(`/api/data/short-volume/history?symbol=${encodeURIComponent(sym)}&days=90`);
      const d = await r.json();
      setSymbolResult({ symbol: d.symbol || sym.toUpperCase(), series: d.series || [], note: d.note });
    } catch {
      setSymbolResult({ symbol: sym.toUpperCase(), series: [], note: "lookup failed — try again" });
    } finally {
      setSymbolLoading(false);
    }
  };

  const trendVals = useMemo(() => trend.map((t) => t.agg_short_ratio), [trend]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Short-sale volume (FINRA Reg SHO)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <TrendingUp size={16} />
        <div>
          <div className="vt-filings-title">Short-sale volume — FINRA Reg SHO</div>
          <div className="vt-filings-sub">
            daily short-marked execution volume (flow proxy — NOT short interest, no predictive claim) ·{" "}
            {today ? `${today.symbols.toLocaleString()} symbols, ${today.date}` : "loading…"} ·{" "}
            <a href="https://cdn.finra.org/equity/regsho/daily/" target="_blank" rel="noreferrer">FINRA Reg SHO <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}

      {!error && (
        <div className="vt-shortvol-body">
          <div className="vt-shortvol-snapshot">
            <div className="vt-shortvol-statblock">
              <div className="vt-shortvol-statlabel">market-wide short ratio, today</div>
              <div className="vt-shortvol-statval">{today ? pct(today.agg_short_ratio) : "—"}</div>
            </div>
            <div className="vt-shortvol-trendblock">
              <div className="vt-shortvol-statlabel">last {trend.length || 0} recorded days</div>
              <Sparkline points={trendVals} />
              <div className="vt-shortvol-trendnote">
                {trend.length
                  ? "trend log started 2026-07-06 — grows one point per trading day"
                  : "trend log just started; check back after a few trading days"}
              </div>
            </div>
          </div>

          <form className="vt-filings-filters" onSubmit={lookupSymbol}>
            <input
              type="search"
              className="vt-earnings-search"
              placeholder="Look up a ticker's ratio history…"
              aria-label="Look up a ticker's short-volume history"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" className="vt-filings-filter" disabled={symbolLoading || !query.trim()}>
              <Search size={13} /> {symbolLoading ? "Searching…" : "Search"}
            </button>
            {symbolResult && (
              <span className="vt-filings-count">
                {symbolResult.series.length} day{symbolResult.series.length === 1 ? "" : "s"} for {symbolResult.symbol}
              </span>
            )}
          </form>

          {symbolResult && (
            <div className="vt-shortvol-symwrap">
              {symbolResult.series.length === 0 ? (
                <div className="vt-filings-state">{symbolResult.note || "no rows for this ticker in the archived window."}</div>
              ) : (
                <>
                  <Sparkline points={symbolResult.series.map((p) => p.short_ratio)} width={320} height={48} />
                  <div className="vt-filings-tablewrap">
                    <table className="vt-filings-table">
                      <thead>
                        <tr><th>Date</th><th className="num">Short ratio</th><th className="num">Total volume</th><th>Market</th></tr>
                      </thead>
                      <tbody>
                        {[...symbolResult.series].reverse().map((p) => (
                          <tr key={p.date}>
                            <td data-l="Date">{p.date}</td>
                            <td data-l="Short ratio" className="num">{pct(p.short_ratio)}</td>
                            <td data-l="Total volume" className="num">{p.total_vol.toLocaleString()}</td>
                            <td data-l="Market">{p.market || "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          <div className="vt-filings-sub vt-shortvol-topheader">
            top short-ratio movers, {today?.date || "today"} · volume ≥ {today ? today.floor_total_vol.toLocaleString() : "—"} shares ·
            {" "}capped at {today?.top_cap ?? "—"} symbols (stated, not hidden)
          </div>
          {!today && <div className="vt-filings-state">Loading today's summary…</div>}
          {today && today.top_ratio.length === 0 && <div className="vt-filings-state">No symbols cleared the volume floor today.</div>}
          {today && today.top_ratio.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr><th>Ticker</th><th className="num">Short ratio</th><th className="num">Total volume</th><th>Market</th></tr>
                </thead>
                <tbody>
                  {today.top_ratio.map((r) => (
                    <tr key={r.symbol}>
                      <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol}</span></td>
                      <td data-l="Short ratio" className="num">{pct(r.short_ratio)}</td>
                      <td data-l="Total volume" className="num">{r.total_vol.toLocaleString()}</td>
                      <td data-l="Market">{r.market}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
