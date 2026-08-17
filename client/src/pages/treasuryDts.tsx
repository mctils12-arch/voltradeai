// TreasuryDtsView — Treasury Daily Statement, deposits & withdrawals of
// operating cash, #/data/treasury-dts.
// server/treasuryDts.ts (BUILD ORDER 6 #2, 2026-07-06) shipped the
// API-only route (/api/data/dts, keyless FiscalData) with no client view —
// same "shipped-data-no-UI" gap class as treasuryAuctions.tsx/tff.tsx
// (2026-08-16/17). RAW display of the day's deposit/withdrawal line items
// only — NOT a signal; the module's own hypothesis note (withheld-tax
// deposits form a daily payroll nowcast weeks ahead of BLS releases;
// corporate-tax/FUTA deltas nowcast macro turns) stays gate-locked. Reuses
// the generic .vt-filings-*/.vt-shortvol-* CSS — no new styles.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Banknote } from "lucide-react";

interface DtsLine {
  record_date: string;
  account_type: string;
  transaction_type: string;
  category: string;
  today_amt: number | null;
  mtd_amt: number | null;
  fytd_amt: number | null;
  src_line: number | null;
  rt: string;
}
interface DtsResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: string;
  record_date?: string;
  count?: number;
  note?: string;
  warming_up?: boolean;
  lines?: DtsLine[];
}

const fmtM = (n: number | null | undefined) =>
  n == null ? "—" : `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}M`;

/** Row order the module's own hypothesis names first (never re-sorted
 *  above other rows by amount — that would look like a ranked claim). */
const isWithheldTax = (category: string) => /withheld/i.test(category);

export default function TreasuryDtsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<DtsResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/dts");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const { deposits, withdrawals } = useMemo(() => {
    const lines = data?.lines || [];
    const bySrc = (a: DtsLine, b: DtsLine) => (a.src_line ?? 0) - (b.src_line ?? 0);
    return {
      deposits: lines.filter((l) => l.transaction_type === "Deposits").slice().sort(bySrc),
      withdrawals: lines.filter((l) => l.transaction_type === "Withdrawals").slice().sort(bySrc),
    };
  }, [data]);

  const renderTable = (label: string, rows: DtsLine[]) => (
    <>
      <div className="vt-filings-sub vt-shortvol-topheader">{label} — {rows.length} lines, as published</div>
      {rows.length === 0 && <div className="vt-filings-state">No {label.toLowerCase()} lines in today's statement.</div>}
      {rows.length > 0 && (
        <div className="vt-filings-tablewrap">
          <table className="vt-filings-table">
            <thead>
              <tr>
                <th>Category</th><th>Account</th>
                <th className="num">Today</th><th className="num">MTD</th><th className="num">FYTD</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((l) => (
                <tr key={`${l.transaction_type}-${l.src_line}-${l.category}`}
                    style={isWithheldTax(l.category) ? { background: "var(--bg-card-hover)" } : undefined}>
                  <td data-l="Category"><span className="vt-filings-ticker">{l.category}</span></td>
                  <td data-l="Account">{l.account_type}</td>
                  <td data-l="Today" className="num">{fmtM(l.today_amt)}</td>
                  <td data-l="MTD" className="num">{fmtM(l.mtd_amt)}</td>
                  <td data-l="FYTD" className="num">{fmtM(l.fytd_amt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );

  return (
    <div className="vt-filings-page" role="region" aria-label="Treasury Daily Statement">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Banknote size={16} />
        <div>
          <div className="vt-filings-title">Treasury Daily Statement</div>
          <div className="vt-filings-sub">
            operating-cash deposits &amp; withdrawals (cash-flow PROXY, not a signal) ·{" "}
            {data?.record_date ? `${data.count ?? 0} lines, ${data.record_date}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/deposits-and-withdrawals-of-operating-cash" target="_blank" rel="noreferrer">Treasury Fiscal Data <ExternalLink size={11} /></a>
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
            all deposit/withdrawal line items from the most recent published Daily Treasury Statement
            (~1-2 business-day lag), $ millions as published. Category order matches the statement's own
            source-line ordering, never re-ranked by amount.
          </div>
          <div className="vt-filings-sub">
            hypothesis under research (gate-locked, no trailing-history determination yet): withheld
            income/employment tax deposits (highlighted below) form a daily payroll nowcast weeks ahead
            of BLS releases; corporate-tax and FUTA category deltas nowcast macro turns. Nothing below
            reflects that hypothesis yet — it is a raw daily table, no model applied.
          </div>
          {!data && <div className="vt-filings-state">Loading latest statement…</div>}
          {data && renderTable("Deposits", deposits)}
          {data && renderTable("Withdrawals", withdrawals)}
        </div>
      )}
    </div>
  );
}
