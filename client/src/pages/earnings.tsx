// EarningsView — the full 8-K earnings-language view (#/data/earnings).
// DESIGN.md-governed: readable at 390/768/1440, designed empty/loading/error
// states, theme tokens only. Data: /api/data/earnings-language/history
// (accumulated archive, began 2026-07-04) merged with the live poll cache.
// RAW as-filed display — every card links to the actual SEC filing + exhibit.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, FileText } from "lucide-react";

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

const EXCERPT_LEN = 480;

export default function EarningsView({ onBack }: { onBack: () => void }) {
  const [rows, setRows] = useState<EarningsRow[] | null>(null);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [meta, setMeta] = useState<{ days: number; count: number }>({ days: 30, count: 0 });

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/earnings-language/history?days=30");
        const d = await r.json();
        if (stop) return;
        setRows(d.filings || []);
        setMeta({ days: d.days ?? 30, count: d.count ?? (d.filings || []).length });
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const visible = useMemo(() => {
    if (!rows) return [];
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) => r.companyName.toLowerCase().includes(q));
  }, [rows, query]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Earnings language (SEC 8-K)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <FileText size={16} />
        <div>
          <div className="vt-filings-title">Earnings language — SEC 8-K Item 2.02</div>
          <div className="vt-filings-sub">
            as filed · {meta.count.toLocaleString()} releases, last {meta.days} days (archive began 2026-07-04) ·
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
            ? "No releases match that company name in the loaded window."
            : "No releases archived yet — the poll runs every ~15 minutes and history accumulates from 2026-07-04."}
        </div>
      )}
      {visible.length > 0 && (
        <div className="vt-earnings-listwrap">
          <ul className="vt-earnings-list">
            {visible.slice(0, 200).map((r) => {
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
          {visible.length > 200 && <div className="vt-filings-state">showing first 200 of {visible.length.toLocaleString()} rows</div>}
        </div>
      )}
    </div>
  );
}
