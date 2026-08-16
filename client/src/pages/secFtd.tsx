// SecFtdView — SEC CNS fails-to-deliver (half-month files), #/data/ftd.
// DATACORE MAXIMUS census build #6 shipped the API-only route
// (/api/data/ftd, server/secFtd.ts, v1.0.171, 2026-07-07; a paid-tier
// mirror followed at /api/v1/stats/secftd) with no client view — this is
// that follow-up, same wiring recipe as atsSummary.tsx/midas.tsx. RAW
// display of SEC's own trailer-checksummed half-month fail balances — no
// predictive claim; raw FTD spikes alone are a crowded signal per the
// route's own note, the settlement-stress composite (server/
// settlementStress.ts) is the gated hypothesis this feeds. Reuses the
// generic .vt-filings-* CSS (atsSummary.tsx precedent) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, TrendingDown } from "lucide-react";

interface FtdTopRow {
  symbol: string;
  name: string;
  qty: number;
  price: number | null;
}
interface FtdSummary {
  period: string;
  settlement_dates: number;
  newest_date: string;
  rows: number;
  top_fails: FtdTopRow[];
  qty_floor: number;
  top_cap: number;
}
interface FtdResponse {
  kind: string;
  source?: string;
  attribution?: string;
  note?: string;
  warming_up?: boolean;
  summary?: FtdSummary;
}

const num = (n: number) => n.toLocaleString();
const price2 = (n: number | null) => (n == null ? "—" : `$${n.toFixed(2)}`);

export default function SecFtdView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FtdResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/ftd");
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
    <div className="vt-filings-page" role="region" aria-label="Fails-to-deliver (SEC CNS)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <TrendingDown size={16} />
        <div>
          <div className="vt-filings-title">Fails-to-deliver — SEC CNS</div>
          <div className="vt-filings-sub">
            aggregate net fail balances per settlement date, half-month files — RAW, no predictive claim ·{" "}
            {s ? `${s.period}, newest date ${s.newest_date}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.sec.gov/data/foiadocsfailsdatahtm" target="_blank" rel="noreferrer">SEC fails-to-deliver <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — half-month files are large, this can take a few minutes on a cold start.</div>
      )}

      {!error && !data?.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {s ? `${num(s.settlement_dates)} settlement dates · ${num(s.rows)} total rows this half-month` : ""}
          </div>
          <div className="vt-filings-sub">
            QUANTITY is a fail BALANCE (a level), not a daily flow — published on a 2.5–4.5 week lag.
          </div>

          {s && s.top_fails.length > 0 ? (
            <>
              <div className="vt-filings-sub">
                top fails on the newest settlement date, floored at {num(s.qty_floor)} shares (stated) — raw FTD
                spikes alone are maximally crowded; the settlement-stress composite (threshold persistence x FTD x
                short volume) is the gated hypothesis, not this list alone
              </div>
              <div className="vt-filings-tablewrap">
                <table className="vt-filings-table">
                  <thead>
                    <tr>
                      <th>Ticker</th><th>Name</th><th className="num">Fail balance (shares)</th><th className="num">Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.top_fails.map((r) => (
                      <tr key={r.symbol}>
                        <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol}</span></td>
                        <td data-l="Name">{r.name}</td>
                        <td data-l="Fail balance (shares)" className="num">{num(r.qty)}</td>
                        <td data-l="Price" className="num">{price2(r.price)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="vt-filings-state">No fails above the floor on the newest settlement date.</div>
          )}
        </div>
      )}
    </div>
  );
}
