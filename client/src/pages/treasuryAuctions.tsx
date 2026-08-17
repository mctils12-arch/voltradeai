// TreasuryAuctionsView — US Treasury auction results, #/data/treasury-auctions.
// server/treasuryAuctions.ts (BUILD ORDER 2 #4, 2026-07-05) shipped the
// API-only route (/api/data/treasury-auctions, keyless TreasuryDirect TA_WS)
// with no client view — same "shipped-data-no-UI" gap class as secFtd.tsx/
// tff.tsx (2026-08-16). RAW display of primary-market results (bid-to-cover
// + bidder-class shares) only — NOT a signal; the module's own hypothesis
// note (bid-to-cover deterioration / rising dealer take precede rate-regime
// shifts) stays gate-locked. The classic tail-vs-when-issued stress metric
// needs a 1pm-ET WI quote, which is paid data — never faked here. Reuses the
// generic .vt-filings-*/.vt-shortvol-* CSS — no new styles.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Tag } from "lucide-react";

interface AuctionRec {
  cusip: string;
  auction_date: string;
  issue_date: string | null;
  maturity_date: string | null;
  type: string;
  term: string | null;
  reopening: boolean | null;
  bid_to_cover: number | null;
  high_yield: number | null;
  high_discount_rate: number | null;
  total_accepted: number | null;
  competitive_accepted: number | null;
  primary_dealer_accepted: number | null;
  direct_accepted: number | null;
  indirect_accepted: number | null;
  dealer_take: number | null;
}
interface AuctionsResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: string;
  count?: number;
  note?: string;
  warming_up?: boolean;
  auctions?: AuctionRec[];
}

const fmtRate = (n: number | null | undefined) => (n == null ? "—" : `${n.toFixed(3)}%`);
const fmtRatio = (n: number | null | undefined) => (n == null ? "—" : n.toFixed(2));
const pct = (n: number | null | undefined) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);
/** Same "compute a share the server already computes dealer_take from"
 *  idiom, applied to the two bidder classes the route doesn't precompute —
 *  identical inputs (accepted / competitive_accepted), just not shipped
 *  server-side because dealer_take is the only one the hypothesis names. */
const shareOf = (part: number | null, whole: number | null): number | null =>
  part != null && whole != null && whole > 0 ? part / whole : null;

const ROW_CAP = 60;

export default function TreasuryAuctionsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<AuctionsResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/treasury-auctions");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = useMemo(() => {
    return (data?.auctions || [])
      .slice()
      .sort((a, b) => b.auction_date.localeCompare(a.auction_date))
      .slice(0, ROW_CAP)
      .map((a) => ({
        ...a,
        rate: a.high_yield ?? a.high_discount_rate,
        rate_kind: a.high_yield != null ? "yield" : a.high_discount_rate != null ? "discount" : null,
        direct_take: shareOf(a.direct_accepted, a.competitive_accepted),
        indirect_take: shareOf(a.indirect_accepted, a.competitive_accepted),
      }));
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="US Treasury auction results">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Tag size={16} />
        <div>
          <div className="vt-filings-title">US Treasury Auctions</div>
          <div className="vt-filings-sub">
            primary-market results — bid-to-cover + bidder-class shares (stress PROXY, not a signal) ·{" "}
            {data?.count != null ? `${data.count} auctions, last 30 days` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.treasurydirect.gov/auctions/auction-query/" target="_blank" rel="noreferrer">TreasuryDirect <ExternalLink size={11} /></a>
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
            results-complete Bill/Note/Bond/TIPS/FRN/CMB auctions from the last 30 days, TreasuryDirect TA_WS.
            dealer/direct/indirect take are DERIVED shares of competitive accepted, not published fields.
          </div>
          <div className="vt-filings-sub">
            hypothesis under research (gate-locked, no trailing-history determination yet): bid-to-cover
            deterioration and a rising dealer take (dealers absorb what investors won't) precede rate-regime
            shifts. Nothing below reflects that hypothesis yet — it is a raw chronological table, no model
            applied. The classic tail-vs-when-issued stress metric needs a 1pm ET WI quote, which is PAID
            data and is not shown here.
          </div>

          <div className="vt-filings-sub vt-shortvol-topheader">
            most recent {ROW_CAP} results, newest first · RAW figures, no model applied
          </div>
          {!data && <div className="vt-filings-state">Loading latest auction results…</div>}
          {data && rows.length === 0 && <div className="vt-filings-state">No results-complete auctions in the last 30 days.</div>}
          {rows.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Date</th><th>Security</th><th className="num">Bid-to-cover</th>
                    <th className="num">High rate</th><th className="num">Dealer take</th>
                    <th className="num">Direct take</th><th className="num">Indirect take</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.cusip + r.auction_date}>
                      <td data-l="Date"><span className="vt-filings-ticker">{r.auction_date}</span></td>
                      <td data-l="Security">{r.type}{r.term ? ` ${r.term}` : ""}{r.reopening ? " (reopen)" : ""}</td>
                      <td data-l="Bid-to-cover" className="num">{fmtRatio(r.bid_to_cover)}</td>
                      <td data-l="High rate" className="num">{fmtRate(r.rate)}{r.rate_kind === "discount" ? " disc" : ""}</td>
                      <td data-l="Dealer take" className="num">{pct(r.dealer_take)}</td>
                      <td data-l="Direct take" className="num">{pct(r.direct_take)}</td>
                      <td data-l="Indirect take" className="num">{pct(r.indirect_take)}</td>
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
