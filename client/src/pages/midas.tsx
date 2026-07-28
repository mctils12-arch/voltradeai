// MidasView — SEC MIDAS individual-security market-structure metrics
// (#/data/midas). DATACORE MAXIMUS census build #10 shipped the API-only
// route (/api/data/microstructure, server/secMidas.ts, v1.0.265, 2026-07-10)
// with no client view — wishlist.md's 2026-07-22 atsSummary/shortvol UI
// sweep named this as the one remaining shipped-data-no-UI gap; this is
// that follow-up, same wiring recipe as atsSummary.tsx. RAW display of
// SEC's own precomputed per-(date,ticker) ranks and ratios — no predictive
// claim; the small-cap HFT-colonization filter hypothesis this watchlist
// feeds is a separately gated [RESEARCH] item (gate 2 unattempted). Reuses
// the generic .vt-filings-* CSS (atsSummary.tsx precedent) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Radar } from "lucide-react";

interface MidasWatchRow {
  ticker: string;
  mcapRank: number;
  cancelToTrade: number | null;
  hiddenRatePct: number | null;
  oddLotRatePct: number | null;
}
interface MidasSummary {
  period: string;
  kind_counts: { stock: number; etf: number };
  newest_date: string;
  rows: number;
  smallcap_watch: MidasWatchRow[];
  smallcap_max_rank: number;
  min_trades_for_hidden: number;
  top_cap: number;
}
interface MidasResponse {
  kind: string;
  source?: string;
  attribution?: string;
  note?: string;
  warming_up?: boolean;
  summary?: MidasSummary;
}

const num = (n: number) => n.toLocaleString();
const ratio2 = (n: number | null) => (n == null ? "—" : n.toFixed(2));
const pct1 = (n: number | null) => (n == null ? "—" : `${n.toFixed(1)}%`);

export default function MidasView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<MidasResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/microstructure");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const s = data?.summary;

  return (
    <div className="vt-filings-page" role="region" aria-label="Market microstructure (SEC MIDAS)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Radar size={16} />
        <div>
          <div className="vt-filings-title">Market microstructure — SEC MIDAS</div>
          <div className="vt-filings-sub">
            per-(date, ticker) lit/hidden/odd-lot/cancel metrics, quarterly files — RAW, no predictive claim ·{" "}
            {s ? `${s.period}, newest date ${s.newest_date}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.sec.gov/data/foiadocsmidasdata" target="_blank" rel="noreferrer">SEC MIDAS <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — quarterly files are large, this can take a few minutes on a cold start.</div>
      )}

      {!error && !data?.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {s ? `${num(s.kind_counts.stock)} stock rows · ${num(s.kind_counts.etf)} ETF rows on the newest date (${num(s.rows)} total rows this quarter)` : ""}
          </div>
          <div className="vt-filings-sub">
            rank scale differs by kind — Stock ranks are deciles (1–10), ETF ranks are quartiles (1–4), never comparable;
            this watchlist covers Stock rows only.
          </div>

          {s && s.smallcap_watch.length > 0 ? (
            <>
              <div className="vt-filings-sub">
                small-cap watch — McapRank ≤ {s.smallcap_max_rank} of 10, floor of {s.min_trades_for_hidden} trades-for-hidden,
                ranked by SEC's own published cancel-to-trade ratio (candidate HFT-colonization filter, gate-2 unattempted — not a signal)
              </div>
              <div className="vt-filings-tablewrap">
                <table className="vt-filings-table">
                  <thead>
                    <tr>
                      <th>Ticker</th><th className="num">McapRank</th><th className="num">Cancel/Trade</th>
                      <th className="num">Hidden %</th><th className="num">Odd-lot %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.smallcap_watch.map((r) => (
                      <tr key={r.ticker}>
                        <td data-l="Ticker"><span className="vt-filings-ticker">{r.ticker}</span></td>
                        <td data-l="McapRank" className="num">{r.mcapRank}/10</td>
                        <td data-l="Cancel/Trade" className="num">{ratio2(r.cancelToTrade)}</td>
                        <td data-l="Hidden %" className="num">{pct1(r.hiddenRatePct)}</td>
                        <td data-l="Odd-lot %" className="num">{pct1(r.oddLotRatePct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="vt-filings-state">No small-cap watchlist rows archived yet.</div>
          )}
        </div>
      )}
    </div>
  );
}
