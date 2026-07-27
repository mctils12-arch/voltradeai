// MidasView — SEC MIDAS individual-security market-structure metrics
// (#/data/microstructure). server/secMidas.ts + /api/data/microstructure
// shipped API-only (census build #10, v1.0.265) with no client view — this
// is that follow-up, named directly in wishlist.md's DATACORE MAXIMUS
// resume block ("SEC MIDAS ... has the identical no-client-view gap ...
// model on atsSummary.tsx/shortvol.tsx"). RAW display of SEC's own
// cross-sectional lit/hidden/odd-lot/cancel metrics — the newest archived
// quarter's smallest-decile Stock rows by cancel-to-trade ratio, sorted
// purely by the source's own numbers (no model applied). The HFT-
// colonization FILTER hypothesis this feeds is a separately gated
// [RESEARCH] item (gate-2 unattempted) — never claimed as a signal here.
// Reuses the generic .vt-filings-*/.vt-shortvol-* CSS, same precedent as
// atsSummary.tsx/methaneHotspots.tsx — no new styles needed.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Activity } from "lucide-react";

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
  time?: string;
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
        <Activity size={16} />
        <div>
          <div className="vt-filings-title">Market microstructure — SEC MIDAS</div>
          <div className="vt-filings-sub">
            cross-sectional lit/hidden/odd-lot/cancel metrics, quarterly files — RAW, no predictive claim ·{" "}
            {s ? `quarter ${s.period}, as of ${s.newest_date}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.sec.gov/data-research/market-microstructure-data" target="_blank" rel="noreferrer">SEC MIDAS <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — MIDAS quarterly files run large; this can take a while on a cold start.</div>
      )}
      {!error && !data?.warming_up && !s && <div className="vt-filings-state">Loading…</div>}

      {!error && !data?.warming_up && s && (
        <div className="vt-shortvol-body">
          {data?.note && <div className="vt-filings-sub">{data.note}</div>}
          <div className="vt-filings-sub">
            {num(s.rows)} rows this quarter · {num(s.kind_counts.stock)} Stock (decile ranks) + {num(s.kind_counts.etf)} ETF (quartile ranks, never comparable to Stock ranks)
          </div>
          <div className="vt-filings-sub">
            smallest-decile watch: Stock rows with McapRank ≤ {s.smallcap_max_rank} of 10, floored at {s.min_trades_for_hidden}+ hidden-eligible trades — ranked by the source's own cancel-to-trade ratio, not a model
          </div>

          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Ticker</th><th className="num">McapRank</th><th className="num">Cancel/trade</th>
                  <th className="num">Hidden rate</th><th className="num">Odd-lot rate</th>
                </tr>
              </thead>
              <tbody>
                {s.smallcap_watch.length === 0 && (
                  <tr><td colSpan={5}><div className="vt-filings-state">No smallest-decile rows cleared the trade floor this date.</div></td></tr>
                )}
                {s.smallcap_watch.map((r) => (
                  <tr key={r.ticker}>
                    <td data-l="Ticker"><span className="vt-filings-ticker">{r.ticker}</span></td>
                    <td data-l="McapRank" className="num">{r.mcapRank}/10</td>
                    <td data-l="Cancel/trade" className="num">{ratio2(r.cancelToTrade)}</td>
                    <td data-l="Hidden rate" className="num">{pct1(r.hiddenRatePct)}</td>
                    <td data-l="Odd-lot rate" className="num">{pct1(r.oddLotRatePct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
