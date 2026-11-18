// FilingsView — the full Form 4 insider-transactions view (#/data/filings).
// DESIGN.md-governed: readable at 390/768/1440 (table on desktop, stacked
// cards on phone via CSS), designed empty/loading/error states, theme tokens
// only. Data: /api/data/insider/history (accumulated archive, began
// 2026-07-04). RAW as-filed display — every row links to the actual SEC
// filing index. The header states the days actually loaded (the route pages
// by whole days under a filing cap — lib/filingCoverage.ts) and the footer
// walks to the older days.
//
// BY FILED DAY (2026-11-18). The page asks for the last 30 FILED days
// (`?filed_from=`), not archive days: since 2026-11-17 a nightly pass recovers
// the filings the 15-minute poll missed, back to July, and writes them into
// the archive day of the night it ran, so the newest archive day is mostly
// months-old filings. Each filed day is a group with its own line: how many
// filings the archive holds for it and whether that is every Form 4 on
// EDGAR's daily index ("complete"), some of them ("partial"), or the poll's
// unchecked sample. A server that answers by archive day (an older build) is
// labelled as that and gets no day groups.
import { Fragment, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, FileText } from "lucide-react";
import { coverageWords, filedDayFirstQuery, filedDayState, filedDayWords, filedDaysSummary, loadOlderLabel, type FiledDayStatus } from "@/lib/filingCoverage";
import { useFilingHistory } from "@/lib/useFilingHistory";

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

const WINDOW_DAYS = 30;
const MAX_ROWS = 500;
const filedDayOf = (r: FilingRow): string => (typeof r.filedAt === "string" ? r.filedAt.slice(0, 10) : "");

/** Consecutive rows filed on the same day, in the order given (the server
 *  returns whole filed days, newest first). */
function groupByFiledDay(rows: FilingRow[]): { day: string; rows: FilingRow[] }[] {
  const out: { day: string; rows: FilingRow[] }[] = [];
  for (const r of rows) {
    const day = filedDayOf(r);
    const last = out[out.length - 1];
    if (last && last.day === day) last.rows.push(r); else out.push({ day, rows: [r] });
  }
  return out;
}

function roleOf(o: any): string {
  if (!o) return "—";
  const parts: string[] = [];
  if (o.isOfficer) parts.push(o.officerTitle || "Officer");
  if (o.isDirector) parts.push("Director");
  if (o.isTenPercentOwner) parts.push("10% owner");
  return parts.join(" · ") || "—";
}

export default function FilingsView({ onBack }: { onBack: () => void }) {
  const [filter, setFilter] = useState<Filter>("all");
  // fixed for the life of the view: a re-render after UTC midnight must not refetch
  const [firstQuery] = useState(() => filedDayFirstQuery(Date.now(), WINDOW_DAYS));
  const { first, filings, coverage, filedDays, error, loadingOlder, olderError, loadOlder } =
    useFilingHistory<any, any>("/api/data/insider/history", firstQuery);

  const rows = useMemo<FilingRow[] | null>(() => {
    if (!first) return null;
    return filings.flatMap((f: any) =>
      (f.transactions || []).map((t: any) => ({
        issuer: f.issuerName || "—",
        symbol: f.issuerTradingSymbol || null,
        owner: f.owners?.[0]?.name || "—",
        role: roleOf(f.owners?.[0]),
        kind: t.kind, table: t.table,
        shares: t.shares,
        price: t.pricePerShare,
        value: t.shares != null && t.pricePerShare != null ? Math.round(t.shares * t.pricePerShare) : null,
        date: t.transactionDate,
        filedAt: f.filedAt,
        secUrl: f.indexUrl || null,
        accession: f.accession,
      }))
    );
  }, [first, filings]);
  const covered = coverageWords(coverage);
  const olderLabel = loadOlderLabel(coverage, loadingOlder);
  const archiveStart = first?.archive_start || "2026-07-04";
  const byFiledDay = coverage?.by === "filed_day";
  const daysSummary = byFiledDay ? filedDaysSummary(filedDays) : "";
  const dayStatus = useMemo(() => new Map<string, FiledDayStatus>(filedDays.map((d) => [d.day, d])), [filedDays]);

  const visible = useMemo(() => {
    if (!rows) return [];
    switch (filter) {
      case "open_market": return rows.filter((r) => r.kind === "open_market_buy" || r.kind === "open_market_sale");
      case "buys": return rows.filter((r) => r.kind === "open_market_buy");
      case "sells": return rows.filter((r) => r.kind === "open_market_sale");
      default: return rows;
    }
  }, [rows, filter]);
  const shown = useMemo(() => visible.slice(0, MAX_ROWS), [visible]);
  const groups = useMemo(() => (byFiledDay ? groupByFiledDay(shown) : [{ day: "", rows: shown }]), [byFiledDay, shown]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Insider transactions (Form 4)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <FileText size={16} />
        <div>
          <div className="vt-filings-title">Insider transactions — SEC Form 4</div>
          <div className="vt-filings-sub">
            as filed · {filings.length.toLocaleString()} filings
            {covered ? <> · <span data-vt-coverage>{covered}</span> ({byFiledDay ? "days filed" : "UTC archive days"})</> : null}
            {daysSummary ? <> · <span data-vt-filed-summary>{daysSummary}</span></> : null}
            {" "}· archive began {archiveStart} ·{" "}
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
        <span className="vt-filings-count">{visible.length.toLocaleString()} rows</span>
      </div>
      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && rows === null && <div className="vt-filings-state">Loading filings…</div>}
      {!error && rows !== null && visible.length === 0 && (
        <div className="vt-filings-state">
          {filter === "all"
            ? "No filings archived yet — the poll runs every ~15 minutes and history accumulates from 2026-07-04."
            : "No rows match this filter in the loaded days."}
        </div>
      )}
      {visible.length > 0 && (
        <div className="vt-filings-tablewrap om-sb">
          <table className="vt-filings-table">
            <thead>
              <tr>
                <th>Ticker</th><th>Insider</th><th>Type</th>
                <th className="num">Shares</th><th className="num">Price</th><th className="num">Value</th>
                <th>Trade date</th><th aria-label="SEC link" />
              </tr>
            </thead>
            <tbody>
              {groups.map((g, gi) => {
                const st = dayStatus.get(g.day);
                return (
                  <Fragment key={g.day || `group-${gi}`}>
                    {byFiledDay && (
                      <tr className="vt-filings-dayrow" data-vt-filed-day={g.day} data-state={st ? filedDayState(st) : "sample"}>
                        <th colSpan={8} scope="colgroup">
                          <span className="vt-filings-dayrow-day">Filed {g.day || "on an unstated day"}</span>
                          {st && <span> · {st.filings.toLocaleString()} {st.filings === 1 ? "filing" : "filings"}</span>}
                          <span className="vt-filings-dayrow-state"> · {st ? filedDayWords(st) : "sample · not yet checked against EDGAR's index"}</span>
                        </th>
                      </tr>
                    )}
                    {g.rows.map((r, i) => (
                      <tr key={`${r.accession}-${i}`}>
                        <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol || "—"}</span>
                          <span className="vt-filings-issuername">{r.issuer}</span></td>
                        <td data-l="Insider"><span>{r.owner}</span><span className="vt-filings-role">{r.role}</span></td>
                        <td data-l="Type"><span className="vt-filings-kindtag" style={{ color: KIND_COLOR[r.kind] }}>
                          {KIND_LABEL[r.kind] || r.kind}</span></td>
                        <td data-l="Shares" className="num">{r.shares != null ? r.shares.toLocaleString() : "—"}</td>
                        <td data-l="Price" className="num">{r.price != null ? `$${r.price.toLocaleString()}` : "—"}</td>
                        <td data-l="Value" className="num">{r.value != null ? `$${r.value.toLocaleString()}` : "—"}</td>
                        <td data-l="Traded">{r.date || "—"}</td>
                        <td>{r.secUrl && (
                          <a href={r.secUrl} target="_blank" rel="noreferrer" aria-label="Open SEC filing" className="vt-filings-seclink">
                            <ExternalLink size={13} />
                          </a>
                        )}</td>
                      </tr>
                    ))}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          {visible.length > MAX_ROWS && <div className="vt-filings-state">showing first {MAX_ROWS} of {visible.length.toLocaleString()} rows — narrow with a filter to see the rest</div>}
        </div>
      )}
      {!error && olderLabel && (
        <div className="vt-filings-older">
          <button className="vt-filings-filter" onClick={loadOlder} disabled={loadingOlder} data-vt-load-older>
            {olderLabel}
          </button>
          {olderError && <span role="status">Could not load the older days — the loaded rows are unchanged. Try again.</span>}
        </div>
      )}
    </div>
  );
}
