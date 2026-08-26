// UsaspendingContractsView — federal contract awards, #/data/contracts.
// server/usaSpending.ts (DATA STREAM EXPANSION #4, 2026-07-05) has shipped
// the API-only route (/api/data/contracts) and a ticker-matched cross-join
// into the entity dossier's "related contracts" panel since 2026-07-24, but
// never a dedicated browse view — the same "shipped-data-no-client-page"
// gap the 2026-08-25 /api/v1 mirror sweep found and flagged as worth its
// own [PRODUCT] session (research/experiments.md, session #27).
//
// HONESTY: this is RAW display only. GATE 1 (recipient->ticker matcher)
// PASSED 2026-07-24. GATE 2's pre-registered hypothesis (high award/mcap
// ratio predicts BETTER forward small-cap returns) was REJECTED 2026-08-15
// — adequately powered, no positive separation at any horizon, and the one
// nominally-interesting result is WRONG-SIGNED and does not survive
// Bonferroni (datacore/signal_ladder.json usaspending_contracts entry). A
// market-cap-matched redesign is filed as a fresh, un-pre-registered
// candidate in open_questions.md — not promoted from this run. Nothing
// below implies a signal; it is the same raw feed the rejected hypothesis
// was tested against.
//
// Server floors at |amt| >= $25k and caps the response at the 500 most
// recent rows (server/routes.ts); this view ranks/searches within that
// same window, client-side, since the route has no server-side search or
// history endpoint (unlike cot.tsx's sibling /history route). Reuses the
// generic .vt-filings-*/.vt-shortvol-* CSS — no new styles.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Handshake, Search } from "lucide-react";

interface ContractTxn {
  aid: string;
  piid: string | null;
  ad: string | null;
  amt: number;
  r: string | null;
  pn: string | null;
  tkr: string | null;
  ag: string | null;
  sub: string | null;
  desc: string | null;
  rt: string;
}
interface ContractsResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: string;
  count?: number;
  note?: string;
  warming_up?: boolean;
  contracts?: ContractTxn[];
}

const TOP_CAP = 100;

const fmtUsd = (n: number | null | undefined) => {
  if (n == null || Number.isNaN(n)) return "—";
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`;
  return `${sign}$${abs.toFixed(0)}`;
};

export default function UsaspendingContractsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<ContractsResponse | null>(null);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [matchedOnly, setMatchedOnly] = useState(true);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/contracts");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = useMemo(() => {
    const q = query.trim().toUpperCase();
    return (data?.contracts || [])
      .filter((c) => !matchedOnly || !!c.tkr)
      .filter((c) => {
        if (!q) return true;
        return (c.tkr || "").toUpperCase().includes(q)
          || (c.r || "").toUpperCase().includes(q)
          || (c.pn || "").toUpperCase().includes(q)
          || (c.ag || "").toUpperCase().includes(q);
      })
      .slice()
      .sort((a, b) => Math.abs(b.amt) - Math.abs(a.amt))
      .slice(0, TOP_CAP);
  }, [data, query, matchedOnly]);

  const matchedCount = useMemo(
    () => (data?.contracts || []).filter((c) => !!c.tkr).length,
    [data],
  );

  return (
    <div className="vt-filings-page" role="region" aria-label="USAspending federal contract awards">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Handshake size={16} />
        <div>
          <div className="vt-filings-title">Federal Contract Awards</div>
          <div className="vt-filings-sub">
            USAspending.gov contracts A-D, |amount| ≥ $25K (RAW — GATE 2 hypothesis REJECTED, not a signal) ·{" "}
            {data && !data.warming_up ? `${data.count ?? rows.length} in window, ${matchedCount} ticker-matched` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.usaspending.gov/" target="_blank" rel="noreferrer">USAspending.gov <ExternalLink size={11} /></a>
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
            GATE 1 (recipient→ticker matcher) PASSED 2026-07-24. GATE 2's pre-registered hypothesis
            — a high award-value/market-cap ratio predicts BETTER forward small-cap returns — was
            REJECTED 2026-08-15: adequately powered (n=50 high-ratio, n=43 low-ratio at 5 days), no
            positive separation at any horizon tested, and the one nominally-interesting result is
            WRONG-SIGNED and fails the Bonferroni multi-comparison bar. A market-cap-matched
            redesign is a fresh, un-pre-registered candidate filed separately, not promoted from
            this run. Everything below is the raw feed that hypothesis was tested against — a
            listing, not a ranking by predicted impact.
          </div>
          <div className="vt-filings-sub">
            Ticker matching is precision-first: an unmatched recipient carries no ticker and is
            excluded by the "ticker-matched only" filter below rather than guessed. Dates (rt) are
            the archive's as-seen date — action_date is the contract's signature date, and DoD/USACE
            awards in particular publish roughly 90 days after signature, so a recent DoD row here
            can describe an older commitment. Amounts are this action's obligation only, not the
            award's lifetime total; a negative amount is a deobligation.
          </div>

          <form className="vt-filings-filters" onSubmit={(e) => e.preventDefault()}>
            <input
              type="search"
              className="vt-earnings-search"
              placeholder="Search ticker, recipient, or agency…"
              aria-label="Search contracts by ticker, recipient, or agency"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <label className="vt-filings-filter" style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
              <input type="checkbox" checked={matchedOnly} onChange={(e) => setMatchedOnly(e.target.checked)} />
              Ticker-matched only
            </label>
            <span className="vt-filings-count">
              {rows.length} shown{rows.length === TOP_CAP ? ` (top ${TOP_CAP} by |amount|)` : ""}
            </span>
          </form>

          {!data && <div className="vt-filings-state">Loading latest contract awards…</div>}
          {data && rows.length === 0 && (
            <div className="vt-filings-state">
              {matchedOnly && !query ? "No ticker-matched contracts in the current archive window." : "No contracts matched this search."}
            </div>
          )}

          {rows.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Date</th><th>Recipient</th><th>Ticker</th><th>Agency</th>
                    <th className="num">Amount</th><th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((c) => (
                    <tr key={c.aid + c.rt}>
                      <td data-l="Date">{c.rt}</td>
                      <td data-l="Recipient">{c.pn || c.r || "—"}</td>
                      <td data-l="Ticker">{c.tkr ? <span className="vt-filings-ticker">{c.tkr}</span> : "—"}</td>
                      <td data-l="Agency">{c.ag || "—"}</td>
                      <td data-l="Amount" className="num">{fmtUsd(c.amt)}</td>
                      <td data-l="Description">{c.desc ? (c.desc.length > 80 ? `${c.desc.slice(0, 80)}…` : c.desc) : "—"}</td>
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
