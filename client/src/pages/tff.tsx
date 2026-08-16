// TffView — CFTC Traders in Financial Futures (futures-only), #/data/tff.
// DATACORE MAXIMUS census build 6 #1 shipped the API-only route
// (/api/data/tff, server/cftcTff.ts, keyless Socrata sibling of /api/data/cot)
// with no client view — this is that follow-up, same "shipped-data-no-UI"
// gap class as secFtd.tsx (2026-08-16). Unlike cot.tsx's sibling route, TFF
// has no /history endpoint yet (single-week cache only, no archive query
// surface) — this page mirrors secFtd.tsx's simpler single-snapshot shape,
// not cot.tsx's search+trend shape. RAW display of the weekly financial-
// futures positioning report only — NOT a signal; the file's own
// hypothesis note (leveraged-money extremes mean-revert, dealer is the
// informed side) stays gate-locked pending trailing archive depth. Reuses
// the generic .vt-filings-*/.vt-shortvol-* CSS — no new styles.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Scale } from "lucide-react";

interface TffMarket {
  report_date: string;
  market: string;
  code: string;
  commodity: string | null;
  open_interest: number | null;
  dealer_long: number | null;
  dealer_short: number | null;
  asset_mgr_long: number | null;
  asset_mgr_short: number | null;
  lev_money_long: number | null;
  lev_money_short: number | null;
  other_rept_long: number | null;
  other_rept_short: number | null;
}
interface TffResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: string;
  report_date?: string;
  count?: number;
  note?: string;
  warming_up?: boolean;
  markets?: TffMarket[];
}

const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString());
const pct = (n: number | null | undefined) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);

/** Same floor-then-rank-by-|net %OI| idiom as cot.tsx — a tiny illiquid
 *  market can show a huge ratio on noise alone. */
const OI_FLOOR = 1000;
const TOP_CAP = 30;

export default function TffView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<TffResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/tff");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const ranked = useMemo(() => {
    return (data?.markets || [])
      .map((m) => {
        const levNet = (m.lev_money_long != null && m.lev_money_short != null) ? m.lev_money_long - m.lev_money_short : null;
        const levNetPctOi = (levNet != null && m.open_interest) ? levNet / m.open_interest : null;
        const dealerNet = (m.dealer_long != null && m.dealer_short != null) ? m.dealer_long - m.dealer_short : null;
        return { ...m, levNet, levNetPctOi, dealerNet };
      })
      .filter((m) => (m.open_interest ?? 0) >= OI_FLOOR && m.levNetPctOi != null)
      .sort((a, b) => Math.abs(b.levNetPctOi!) - Math.abs(a.levNetPctOi!))
      .slice(0, TOP_CAP);
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="CFTC Traders in Financial Futures">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Scale size={16} />
        <div>
          <div className="vt-filings-title">CFTC Traders in Financial Futures</div>
          <div className="vt-filings-sub">
            weekly financial-futures positioning by trader category (positioning PROXY — NOT a signal, no reversal claims) ·{" "}
            {data?.report_date ? `${data.count ?? 0} markets, ${data.report_date}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://publicreporting.cftc.gov/" target="_blank" rel="noreferrer">CFTC Public Reporting <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — check back shortly.</div>
      )}

      {!error && !data?.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            covers ES/NQ/rates/FX-class markets with financial trader categories (dealer/intermediary, asset manager,
            leveraged money, other reportables) — the futures-only financial sibling of the commodities-focused COT
            report; futures-ONLY, no options included.
          </div>
          <div className="vt-filings-sub">
            hypothesis under research (gate-locked, archive still accumulating trailing history): leveraged-money net-
            positioning extremes mean-revert in index futures; dealer positioning is the informed side. Nothing below
            reflects that hypothesis yet — it is a raw ranked table, no model applied.
          </div>

          <div className="vt-filings-sub vt-shortvol-topheader">
            largest |leveraged-money net % of open interest|, {data?.report_date || "latest week"} · open interest
            floored at {OI_FLOOR.toLocaleString()} contracts · RAW ratio, no model applied
          </div>
          {!data && <div className="vt-filings-state">Loading latest report week…</div>}
          {data && ranked.length === 0 && <div className="vt-filings-state">No markets cleared the open-interest floor this week.</div>}
          {ranked.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Market</th><th>Commodity</th><th className="num">Open interest</th>
                    <th className="num">Lev money net</th><th className="num">Net % OI</th><th className="num">Dealer net</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map((r) => (
                    <tr key={r.code}>
                      <td data-l="Market"><span className="vt-filings-ticker">{r.market}</span></td>
                      <td data-l="Commodity">{r.commodity || "—"}</td>
                      <td data-l="Open interest" className="num">{fmt(r.open_interest)}</td>
                      <td data-l="Lev money net" className="num">{fmt(r.levNet)}</td>
                      <td data-l="Net % OI" className="num">{pct(r.levNetPctOi)}</td>
                      <td data-l="Dealer net" className="num">{fmt(r.dealerNet)}</td>
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
