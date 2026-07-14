// CotView — the full CFTC Commitments of Traders (disaggregated) view
// (#/data/cot). Same shape as AttentionView/ShortVolView: market-wide
// trend + a market search (no curated ticker seed here, so search-then-
// pick replaces the exact ticker lookup) + today's ranked table. Reuses
// the same generic .vt-filings-*/.vt-shortvol-* CSS — no new styles.
//
// Data: /api/data/cot (latest cached week's full market list) +
// /api/data/cot/history (market-wide trend, market search, or one
// market's multi-week series). RAW weekly positioning — NOT a signal;
// positioning-extreme claims stay gate-locked until the archive has
// trailing history and gate 1 passes.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Scale, Search } from "lucide-react";

interface MarketRow {
  code: string;
  market: string;
  commodity: string | null;
  open_interest: number | null;
  m_money_long: number | null;
  m_money_short: number | null;
}
interface Today { report_date: string; count: number; markets: MarketRow[]; }
interface TrendPoint { report_date: string; markets: number; total_open_interest: number; }
interface MarketMatch { code: string; market: string; commodity: string | null; report_date: string; open_interest: number | null; }
interface MarketPoint { report_date: string; open_interest: number | null; m_money_long: number | null; m_money_short: number | null; m_money_net: number | null; m_money_net_pct_oi: number | null; }

const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString());
const pct = (n: number | null | undefined) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);

/** Minimal inline sparkline — mirrors attention.tsx's, no charting dependency. */
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

/** Floor open interest before ranking by |managed-money net % of OI| —
 *  same reasoning as shortvol's FLOOR_TOTAL_VOL: a tiny illiquid market
 *  can show a huge ratio on noise alone. */
const OI_FLOOR = 1000;
const TOP_CAP = 30;

export default function CotView({ onBack }: { onBack: () => void }) {
  const [today, setToday] = useState<Today | null>(null);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<MarketMatch[] | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [picked, setPicked] = useState<{ code: string; market: string; series: MarketPoint[]; note?: string } | null>(null);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const [live, hist] = await Promise.all([
          fetch("/api/data/cot").then((r) => r.json()),
          fetch("/api/data/cot/history?weeks=52").then((r) => r.json()),
        ]);
        if (stop) return;
        if (!live.warming_up) setToday({ report_date: live.report_date, count: live.count, markets: live.markets || [] });
        setTrend(hist.trend || []);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const search = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = query.trim();
    if (!q) return;
    setSearchLoading(true);
    setPicked(null);
    try {
      const r = await fetch(`/api/data/cot/history?q=${encodeURIComponent(q)}`);
      const d = await r.json();
      setMatches(d.matches || []);
    } catch {
      setMatches([]);
    } finally {
      setSearchLoading(false);
    }
  };

  const pickMarket = async (code: string, market: string) => {
    setPicked({ code, market, series: [] });
    try {
      const r = await fetch(`/api/data/cot/history?code=${encodeURIComponent(code)}&weeks=104`);
      const d = await r.json();
      setPicked({ code, market, series: d.series || [], note: d.note });
    } catch {
      setPicked({ code, market, series: [], note: "lookup failed — try again" });
    }
  };

  const trendVals = useMemo(() => trend.map((t) => t.total_open_interest), [trend]);
  const ranked = useMemo(() => {
    return (today?.markets || [])
      .map((m) => {
        const net = (m.m_money_long != null && m.m_money_short != null) ? m.m_money_long - m.m_money_short : null;
        const netPctOi = (net != null && m.open_interest) ? net / m.open_interest : null;
        return { ...m, net, netPctOi };
      })
      .filter((m) => (m.open_interest ?? 0) >= OI_FLOOR && m.netPctOi != null)
      .sort((a, b) => Math.abs(b.netPctOi!) - Math.abs(a.netPctOi!))
      .slice(0, TOP_CAP);
  }, [today]);

  return (
    <div className="vt-filings-page" role="region" aria-label="CFTC Commitments of Traders (disaggregated)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Scale size={16} />
        <div>
          <div className="vt-filings-title">CFTC Commitments of Traders — disaggregated</div>
          <div className="vt-filings-sub">
            weekly futures-only positioning by trader category (positioning PROXY — NOT a signal, no reversal claims) ·{" "}
            {today ? `${today.count} markets, ${today.report_date}` : "loading…"} ·{" "}
            <a href="https://publicreporting.cftc.gov/" target="_blank" rel="noreferrer">CFTC Public Reporting <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}

      {!error && (
        <div className="vt-shortvol-body">
          <div className="vt-shortvol-snapshot">
            <div className="vt-shortvol-statblock">
              <div className="vt-shortvol-statlabel">total open interest, latest archived week</div>
              <div className="vt-shortvol-statval">{today ? fmt(trend[trend.length - 1]?.total_open_interest ?? null) : "—"}</div>
            </div>
            <div className="vt-shortvol-trendblock">
              <div className="vt-shortvol-statlabel">last {trend.length || 0} recorded weeks</div>
              <Sparkline points={trendVals} />
              <div className="vt-shortvol-trendnote">
                {trend.length
                  ? "one point per report week the archive holds — grows weekly"
                  : "archive just started; check back after a few weekly publishes"}
              </div>
            </div>
          </div>

          <form className="vt-filings-filters" onSubmit={search}>
            <input
              type="search"
              className="vt-earnings-search"
              placeholder="Search a market or commodity (e.g. CRUDE OIL, GOLD)…"
              aria-label="Search CFTC markets by name or contract code"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" className="vt-filings-filter" disabled={searchLoading || !query.trim()}>
              <Search size={13} /> {searchLoading ? "Searching…" : "Search"}
            </button>
            {matches && (
              <span className="vt-filings-count">
                {matches.length} match{matches.length === 1 ? "" : "es"}
              </span>
            )}
          </form>

          {matches && matches.length === 0 && (
            <div className="vt-filings-state">No markets matched — CFTC market names are terse (e.g. "WTI-PHYSICAL"), try a shorter fragment.</div>
          )}

          {matches && matches.length > 0 && !picked && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr><th>Market</th><th>Commodity</th><th className="num">Open interest</th><th aria-label="Select" /></tr>
                </thead>
                <tbody>
                  {matches.map((m) => (
                    <tr key={m.code}>
                      <td data-l="Market">{m.market}</td>
                      <td data-l="Commodity">{m.commodity || "—"}</td>
                      <td data-l="Open interest" className="num">{fmt(m.open_interest)}</td>
                      <td><button className="vt-filings-filter" onClick={() => pickMarket(m.code, m.market)}>View →</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {picked && (
            <div className="vt-shortvol-symwrap">
              <div className="vt-filings-sub">{picked.market} — managed-money net position vs. open interest</div>
              {picked.series.length === 0 ? (
                <div className="vt-filings-state">{picked.note || "no rows for this market in the archived window."}</div>
              ) : (
                <>
                  <Sparkline points={picked.series.map((p) => p.m_money_net_pct_oi)} width={320} height={48} />
                  <div className="vt-filings-tablewrap">
                    <table className="vt-filings-table">
                      <thead>
                        <tr><th>Report date</th><th className="num">Open interest</th><th className="num">MM long</th><th className="num">MM short</th><th className="num">MM net</th><th className="num">Net % OI</th></tr>
                      </thead>
                      <tbody>
                        {[...picked.series].reverse().map((p) => (
                          <tr key={p.report_date}>
                            <td data-l="Report date">{p.report_date}</td>
                            <td data-l="Open interest" className="num">{fmt(p.open_interest)}</td>
                            <td data-l="MM long" className="num">{fmt(p.m_money_long)}</td>
                            <td data-l="MM short" className="num">{fmt(p.m_money_short)}</td>
                            <td data-l="MM net" className="num">{fmt(p.m_money_net)}</td>
                            <td data-l="Net % OI" className="num">{pct(p.m_money_net_pct_oi)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
              <button className="vt-filings-filter" onClick={() => { setPicked(null); setMatches(null); setQuery(""); }}>← New search</button>
            </div>
          )}

          <div className="vt-filings-sub vt-shortvol-topheader">
            largest |managed-money net % of open interest|, {today?.report_date || "latest week"} · open interest floored at {OI_FLOOR.toLocaleString()} contracts · RAW ratio, no model applied
          </div>
          {!today && <div className="vt-filings-state">Loading latest report week…</div>}
          {today && ranked.length === 0 && <div className="vt-filings-state">No markets cleared the open-interest floor this week.</div>}
          {today && ranked.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr><th>Market</th><th>Commodity</th><th className="num">Open interest</th><th className="num">MM net</th><th className="num">Net % OI</th></tr>
                </thead>
                <tbody>
                  {ranked.map((r) => (
                    <tr key={r.code}>
                      <td data-l="Market"><span className="vt-filings-ticker">{r.market}</span></td>
                      <td data-l="Commodity">{r.commodity || "—"}</td>
                      <td data-l="Open interest" className="num">{fmt(r.open_interest)}</td>
                      <td data-l="MM net" className="num">{fmt(r.net)}</td>
                      <td data-l="Net % OI" className="num">{pct(r.netPctOi)}</td>
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
