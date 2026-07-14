// AttentionView — the full Wikipedia pageviews attention view
// (#/data/attention). Same shape as ShortVolView (shortvol.tsx): market-
// wide trend + per-ticker archive lookup + today's ranked table. Reuses
// the same generic .vt-filings-*/.vt-shortvol-* CSS — no new styles needed.
//
// Data: /api/data/attention (today's live ranked seed panel, computed at
// poll time) + /api/data/attention/history (market-wide seed-total trend,
// and on demand a single ticker's view series from the day archive). RAW
// attention PROXY — NOT a signal; spike/z-score claims stay gate-locked
// until the archive has trailing history and gate 1 passes.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Eye, Search } from "lucide-react";

interface TickerRow { ticker: string; article: string; views: number; }
interface Today { date: string; symbols?: number; count?: number; seed_size?: number; tickers: TickerRow[]; }
interface TrendPoint { date: string; tickers: number; total_views: number; }
interface TickerPoint { date: string; views: number; }

const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString());

/** Minimal inline sparkline — mirrors shortvol.tsx's, no charting dependency. */
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
         aria-label={`trend from ${fmt(vals[0].v)} to ${fmt(vals[vals.length - 1].v)}`}>
      <path d={d} fill="none" stroke="var(--accent-bright)" strokeWidth={1.6} />
    </svg>
  );
}

export default function AttentionView({ onBack }: { onBack: () => void }) {
  const [today, setToday] = useState<Today | null>(null);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [tickerResult, setTickerResult] = useState<{ ticker: string; article: string | null; series: TickerPoint[]; note?: string } | null>(null);
  const [tickerLoading, setTickerLoading] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const [live, hist] = await Promise.all([
          fetch("/api/data/attention").then((r) => r.json()),
          fetch("/api/data/attention/history?days=60").then((r) => r.json()),
        ]);
        if (stop) return;
        if (!live.warming_up) setToday({ date: live.date, count: live.count, seed_size: live.seed_size, tickers: live.tickers || [] });
        setTrend(hist.trend || []);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const lookupTicker = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const t = query.trim();
    if (!t) return;
    setTickerLoading(true);
    try {
      const r = await fetch(`/api/data/attention/history?ticker=${encodeURIComponent(t)}&days=90`);
      const d = await r.json();
      setTickerResult({ ticker: d.ticker || t.toUpperCase(), article: d.article ?? null, series: d.series || [], note: d.note });
    } catch {
      setTickerResult({ ticker: t.toUpperCase(), article: null, series: [], note: "lookup failed — try again" });
    } finally {
      setTickerLoading(false);
    }
  };

  const trendVals = useMemo(() => trend.map((t) => t.total_views), [trend]);
  const ranked = useMemo(() => [...(today?.tickers || [])].sort((a, b) => b.views - a.views), [today]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Attention proxy (Wikipedia pageviews)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Eye size={16} />
        <div>
          <div className="vt-filings-title">Attention proxy — Wikipedia pageviews</div>
          <div className="vt-filings-sub">
            daily article views for a curated ticker seed (attention PROXY — NOT a signal, no spike claims) ·{" "}
            {today ? `${today.count ?? today.tickers.length} of ${today.seed_size ?? today.tickers.length} tickers, ${today.date}` : "loading…"} ·{" "}
            <a href="https://wikimedia.org/api/rest_v1/" target="_blank" rel="noreferrer">Wikimedia pageviews API <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}

      {!error && (
        <div className="vt-shortvol-body">
          <div className="vt-shortvol-snapshot">
            <div className="vt-shortvol-statblock">
              <div className="vt-shortvol-statlabel">seed-total views, latest complete day</div>
              <div className="vt-shortvol-statval">{today ? fmt(trend[trend.length - 1]?.total_views ?? null) : "—"}</div>
            </div>
            <div className="vt-shortvol-trendblock">
              <div className="vt-shortvol-statlabel">last {trend.length || 0} recorded days</div>
              <Sparkline points={trendVals} />
              <div className="vt-shortvol-trendnote">
                {trend.length
                  ? "one point per day the panel archive holds — grows daily"
                  : "archive just started; check back after a few days"}
              </div>
            </div>
          </div>

          <form className="vt-filings-filters" onSubmit={lookupTicker}>
            <input
              type="search"
              className="vt-earnings-search"
              placeholder="Look up a ticker's view history…"
              aria-label="Look up a ticker's pageview history"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" className="vt-filings-filter" disabled={tickerLoading || !query.trim()}>
              <Search size={13} /> {tickerLoading ? "Searching…" : "Search"}
            </button>
            {tickerResult && (
              <span className="vt-filings-count">
                {tickerResult.series.length} day{tickerResult.series.length === 1 ? "" : "s"} for {tickerResult.ticker}
                {tickerResult.article ? ` (${tickerResult.article})` : ""}
              </span>
            )}
          </form>

          {tickerResult && (
            <div className="vt-shortvol-symwrap">
              {tickerResult.series.length === 0 ? (
                <div className="vt-filings-state">{tickerResult.note || "no rows for this ticker in the archived window."}</div>
              ) : (
                <>
                  <Sparkline points={tickerResult.series.map((p) => p.views)} width={320} height={48} />
                  <div className="vt-filings-tablewrap">
                    <table className="vt-filings-table">
                      <thead>
                        <tr><th>Date</th><th className="num">Views</th></tr>
                      </thead>
                      <tbody>
                        {[...tickerResult.series].reverse().map((p) => (
                          <tr key={p.date}>
                            <td data-l="Date">{p.date}</td>
                            <td data-l="Views" className="num">{fmt(p.views)}</td>
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
            ranked by views, {today?.date || "today"} · curated ticker→article seed, not the full market
          </div>
          {!today && <div className="vt-filings-state">Loading today's panel…</div>}
          {today && ranked.length === 0 && <div className="vt-filings-state">No tickers reported today.</div>}
          {today && ranked.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr><th>Ticker</th><th>Article</th><th className="num">Views</th></tr>
                </thead>
                <tbody>
                  {ranked.map((r) => (
                    <tr key={r.ticker}>
                      <td data-l="Ticker"><span className="vt-filings-ticker">{r.ticker}</span></td>
                      <td data-l="Article">{r.article}</td>
                      <td data-l="Views" className="num">{fmt(r.views)}</td>
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
