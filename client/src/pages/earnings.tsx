// EarningsView — the full 8-K earnings-language view (#/data/earnings).
// DESIGN.md-governed: readable at 390/768/1440, designed empty/loading/error
// states, theme tokens only. Data: /api/data/earnings-language/history
// (accumulated archive, began 2026-07-04) merged with the live poll cache.
// RAW as-filed display — every card links to the actual SEC filing + exhibit.
// The header states the UTC archive days actually loaded (the route pages by
// whole days under a 500-release cap: in an earnings week 30 days asked for
// is about a week loaded) and the footer walks to the older days.
import { useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, FileText } from "lucide-react";
import { coverageWords, loadOlderLabel, type FilingHistoryMeta } from "@/lib/filingCoverage";
import { useFilingHistory } from "@/lib/useFilingHistory";

interface EarningsRow {
  accession: string;
  cik: string;
  companyName: string;
  filedAt: string | null;
  itemCodes: string[];
  indexUrl: string;
  exhibitUrl: string;
  text: string;
  textLength: number;
  truncated: boolean;
}

interface HistoryPayload extends FilingHistoryMeta {
  kind?: string;
  source?: string;
  filings?: EarningsRow[];
}

const EXCERPT_LEN = 480;
/** cards rendered at once; the company filter reaches the rest of what is loaded */
const VISIBLE_CAP = 200;

export default function EarningsView({ onBack }: { onBack: () => void }) {
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const { first, filings, coverage, error, loadingOlder, olderError, loadOlder } =
    useFilingHistory<EarningsRow, HistoryPayload>("/api/data/earnings-language/history", "?days=30");
  const rows = first ? filings : null;
  const covered = coverageWords(coverage);
  const olderLabel = loadOlderLabel(coverage, loadingOlder);

  const visible = useMemo(() => {
    if (!rows) return [];
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) => (r.companyName || "").toLowerCase().includes(q));
  }, [rows, query]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Earnings language (SEC 8-K)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <FileText size={16} />
        <div>
          <div className="vt-filings-title">Earnings language — SEC 8-K Item 2.02</div>
          <div className="vt-filings-sub">
            as filed · {filings.length.toLocaleString()} releases
            {covered ? <> · <span data-vt-coverage>{covered}</span> (UTC archive days)</> : null}
            {" "}· archive began {first?.archive_start || "2026-07-04"} ·
            prepared remarks/guidance only, Q&A is almost never filed ·{" "}
            <a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K" target="_blank" rel="noreferrer">SEC EDGAR</a>
          </div>
        </div>
      </div>
      <div className="vt-filings-filters">
        <input
          type="search"
          className="vt-earnings-search"
          placeholder="Filter by company name…"
          aria-label="Filter by company name"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="vt-filings-count">{visible.length.toLocaleString()} rows</span>
      </div>
      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && rows === null && <div className="vt-filings-state">Loading earnings-language releases…</div>}
      {!error && rows !== null && visible.length === 0 && (
        <div className="vt-filings-state">
          {query
            ? "No releases match that company name in the loaded days."
            : "No releases archived yet — the poll runs every ~15 minutes and history accumulates from 2026-07-04."}
        </div>
      )}
      {visible.length > 0 && (
        <div className="vt-earnings-listwrap">
          <ul className="vt-earnings-list">
            {visible.slice(0, VISIBLE_CAP).map((r) => {
              const isOpen = !!expanded[r.accession];
              const showFull = isOpen || r.text.length <= EXCERPT_LEN;
              const body = showFull ? r.text : r.text.slice(0, EXCERPT_LEN).trimEnd() + "…";
              return (
                <li key={r.accession} className="vt-earnings-card">
                  <div className="vt-earnings-card-head">
                    <span className="vt-earnings-company">{r.companyName}</span>
                    <span className="vt-earnings-date">{r.filedAt || "—"}</span>
                  </div>
                  <div className="vt-earnings-tags">
                    {r.itemCodes.map((c) => <span key={c} className="vt-earnings-itemtag">Item {c}</span>)}
                    {r.truncated && <span className="vt-earnings-itemtag vt-earnings-trunctag">exhibit truncated in archive</span>}
                  </div>
                  <p className="vt-earnings-text">{body}</p>
                  <div className="vt-earnings-actions">
                    {r.text.length > EXCERPT_LEN && (
                      <button className="vt-earnings-expandbtn"
                              onClick={() => setExpanded((s) => ({ ...s, [r.accession]: !s[r.accession] }))}>
                        {isOpen ? "Show less" : "Read full release"}
                      </button>
                    )}
                    <a href={r.exhibitUrl} target="_blank" rel="noreferrer" className="vt-earnings-linkbtn" aria-label="Open exhibit on SEC.gov">
                      Exhibit <ExternalLink size={13} />
                    </a>
                    <a href={r.indexUrl} target="_blank" rel="noreferrer" className="vt-earnings-linkbtn" aria-label="Open filing index on SEC.gov">
                      Filing <ExternalLink size={13} />
                    </a>
                  </div>
                </li>
              );
            })}
          </ul>
          {visible.length > VISIBLE_CAP && (
            <div className="vt-filings-state">
              showing first {VISIBLE_CAP} of {visible.length.toLocaleString()} rows — narrow with the company filter to see the rest
            </div>
          )}
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
