// FilingsView — the full Form 4 insider-transactions view (#/data/filings).
// DESIGN.md-governed: readable at 390/768/1440 (table on desktop, stacked
// cards on phone via CSS), designed empty/loading/error states, theme tokens
// only. Data: /api/data/insider/history (accumulated archive, began
// 2026-07-04) merged with the live poll cache. RAW as-filed display — every
// row links to the actual SEC filing index.
// The sub-line states the archive days actually loaded (the route pages by
// whole day at a filing cap; 2026-11-12) and "Load older days" walks the
// route's own `next` — see client/src/lib/filingHistory.ts.
import { useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, FileText } from "lucide-react";
import { useFilingHistory } from "../lib/useFilingHistory";

interface Form4Filing {
  accession: string;
  filedAt?: string | null;
  indexUrl?: string | null;
  issuerName?: string | null;
  issuerTradingSymbol?: string | null;
  owners?: Array<{ name?: string | null; isOfficer?: boolean; isDirector?: boolean; isTenPercentOwner?: boolean; officerTitle?: string | null }> | null;
  transactions?: Array<{ kind: string; table: string; shares: number | null; pricePerShare: number | null; transactionDate: string | null }> | null;
}

interface FilingRow {
  issuer: string;
  symbol: string | null;
  owner: string;
  role: string;
  kind: string;
  table: string;
  shares: number | null;
  price: number | null;
  value: number | null;
  date: string | null;
  filedAt: string | null;
  secUrl: string | null;
  accession: string;
}

const KIND_LABEL: Record<string, string> = {
  open_market_buy: "BUY", open_market_sale: "SELL", award_grant: "GRANT",
  option_exercise: "EXERCISE", gift: "GIFT", tax_withholding: "TAX WH", other: "OTHER",
};
const KIND_COLOR: Record<string, string> = {
  open_market_buy: "var(--accent-green)", open_market_sale: "var(--accent-red)",
  award_grant: "var(--accent-orange)", option_exercise: "var(--accent)",
  gift: "var(--accent-purple)", tax_withholding: "var(--text-tertiary)", other: "var(--text-tertiary)",
};

type Filter = "all" | "open_market" | "buys" | "sells";

function roleOf(o: any): string {
  if (!o) return "—";
  const parts: string[] = [];
  if (o.isOfficer) parts.push(o.officerTitle || "Officer");
  if (o.isDirector) parts.push("Director");
  if (o.isTenPercentOwner) parts.push("10% owner");
  return parts.join(" · ") || "—";
}

const HISTORY_URL = "/api/data/insider/history?days=30";

export default function FilingsView({ onBack }: { onBack: () => void }) {
  const hist = useFilingHistory<Form4Filing>(HISTORY_URL, 30, "filings");
  const [filter, setFilter] = useState<Filter>("all");

  const rows = useMemo<FilingRow[] | null>(() => {
    if (!hist.filings) return null;
    return hist.filings.flatMap((f) =>
      (f.transactions || []).map((t) => ({
        issuer: f.issuerName || "—",
        symbol: f.issuerTradingSymbol || null,
        owner: f.owners?.[0]?.name || "—",
        role: roleOf(f.owners?.[0]),
        kind: t.kind, table: t.table,
        shares: t.shares,
        price: t.pricePerShare,
        value: t.shares != null && t.pricePerShare != null ? Math.round(t.shares * t.pricePerShare) : null,
        date: t.transactionDate,
        filedAt: f.filedAt ?? null,
        secUrl: f.indexUrl || null,
        accession: f.accession,
      })),
    );
  }, [hist.filings]);

  const visible = useMemo(() => {
    if (!rows) return [];
    switch (filter) {
      case "open_market": return rows.filter((r) => r.kind === "open_market_buy" || r.kind === "open_market_sale");
      case "buys": return rows.filter((r) => r.kind === "open_market_buy");
      case "sells": return rows.filter((r) => r.kind === "open_market_sale");
      default: return rows;
    }
  }, [rows, filter]);

  const error = hist.error;

  return (
    <div className="vt-filings-page" role="region" aria-label="Insider transactions (Form 4)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <FileText size={16} />
        <div>
          <div className="vt-filings-title">Insider transactions — SEC Form 4</div>
          <div className="vt-filings-sub" data-vt-coverage>
            as filed · {hist.label} (archive began 2026-07-04) ·{" "}
            <a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4" target="_blank" rel="noreferrer">SEC EDGAR</a>
          </div>
        </div>
      </div>
      <div className="vt-filings-filters" role="tablist" aria-label="Filter transactions">
        {([["all", "All"], ["open_market", "Open market"], ["buys", "Buys"], ["sells", "Sells"]] as [Filter, string][]).map(([k, label]) => (
          <button key={k} role="tab" aria-selected={filter === k}
                  className={`vt-filings-filter${filter === k ? " active" : ""}`}
                  onClick={() => setFilter(k)}>
            {label}
          </button>
        ))}
        {hist.hasOlder && (
          <button className="vt-filings-filter" data-vt-load-older onClick={hist.loadOlder} disabled={hist.loadingOlder}
                  aria-label="Load older archive days">
            {hist.loadingOlder ? "Loading older days…" : "Load older days"}
          </button>
        )}
        <span className="vt-filings-count">{visible.length.toLocaleString()} rows</span>
      </div>
      {hist.olderError && <div className="vt-filings-state">The older page did not load — the days above are still what is shown; try again.</div>}
      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && rows === null && <div className="vt-filings-state">Loading filings…</div>}
      {!error && rows !== null && visible.length === 0 && (
        <div className="vt-filings-state">
          {filter === "all"
            ? "No filings archived yet — the poll runs every ~15 minutes and history accumulates from 2026-07-04."
            : "No rows match this filter in the loaded window."}
        </div>
      )}
      {visible.length > 0 && (
        <div className="vt-filings-tablewrap om-sb">
          <table className="vt-filings-table">
            <thead>
              <tr>
                <th>Ticker</th><th>Insider</th><th>Type</th>
                <th className="num">Shares</th><th className="num">Price</th><th className="num">Value</th>
                <th>Date</th><th aria-label="SEC link" />
              </tr>
            </thead>
            <tbody>
              {visible.slice(0, 500).map((r, i) => (
                <tr key={`${r.accession}-${i}`}>
                  <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol || "—"}</span>
                    <span className="vt-filings-issuername">{r.issuer}</span></td>
                  <td data-l="Insider"><span>{r.owner}</span><span className="vt-filings-role">{r.role}</span></td>
                  <td data-l="Type"><span className="vt-filings-kindtag" style={{ color: KIND_COLOR[r.kind] }}>
                    {KIND_LABEL[r.kind] || r.kind}</span></td>
                  <td data-l="Shares" className="num">{r.shares != null ? r.shares.toLocaleString() : "—"}</td>
                  <td data-l="Price" className="num">{r.price != null ? `$${r.price.toLocaleString()}` : "—"}</td>
                  <td data-l="Value" className="num">{r.value != null ? `$${r.value.toLocaleString()}` : "—"}</td>
                  <td data-l="Date">{r.date || r.filedAt || "—"}</td>
                  <td>{r.secUrl && (
                    <a href={r.secUrl} target="_blank" rel="noreferrer" aria-label="Open SEC filing" className="vt-filings-seclink">
                      <ExternalLink size={13} />
                    </a>
                  )}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {visible.length > 500 && <div className="vt-filings-state">showing first 500 of {visible.length.toLocaleString()} rows</div>}
        </div>
      )}
    </div>
  );
}
