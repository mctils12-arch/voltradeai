// XbrlFundamentalsView — Company fundamentals as filed with the SEC,
// #/data/xbrl-fundamentals. The nightly record server/xbrlFrames.ts keeps
// (RAW per CLAUDE.md's RAW OVERLAYS vs SIGNALS rule: company-reported
// numbers, no predictive claim). Every word and number comes from
// client/src/lib/fundamentalsView.ts; this file lays them out with the
// .vt-filings-* / .vt-shortvol-* styles the other RAW table pages use.
//
// Not a map layer: the record has no coordinates, so it opens as a page
// from the streams panel, like the FMCSA and DTCC pages.
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { ArrowLeft, ExternalLink, Building2 } from "lucide-react";
import {
  changeLine, companyTitle, conceptLabel, conceptRows, factRows, fmtAmount, fmtPct, periodLabel, revisionPct, runLine,
  type CompanyPayload, type FundamentalsPayload,
} from "../lib/fundamentalsView";

const FIELD_STYLE = { background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text-primary)", padding: "5px 8px" } as const;
const REVISIONS_SHOWN = 25;

export default function XbrlFundamentalsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FundamentalsPayload | null>(null);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [company, setCompany] = useState<CompanyPayload | null>(null);
  const [lookup, setLookup] = useState<"idle" | "loading" | "none" | "error">("idle");
  const [allRevisions, setAllRevisions] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/xbrl-fundamentals");
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const concepts = data?.concepts ?? [];
  const rows = useMemo(() => (data ? conceptRows(data) : []), [data]);
  const revisions = data?.revisions ?? [];
  const shownRevisions = allRevisions ? revisions : revisions.slice(0, REVISIONS_SHOWN);
  const facts = useMemo(() => (company ? factRows(company, concepts) : []), [company, concepts]);

  const onLookup = async (e: FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLookup("loading");
    setCompany(null);
    try {
      const r = await fetch(`/api/data/xbrl-fundamentals/company?q=${encodeURIComponent(q)}`);
      if (r.status === 404) { setLookup("none"); return; }
      if (!r.ok) throw new Error(String(r.status));
      setCompany(await r.json());
      setLookup("idle");
    } catch {
      setLookup("error");
    }
  };

  return (
    <div className="vt-filings-page" role="region" aria-label="Company fundamentals as filed with the SEC">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Building2 size={16} />
        <div>
          <div className="vt-filings-title">Company fundamentals, as filed with the SEC</div>
          <div className="vt-filings-sub">
            numbers companies reported in their own quarterly and annual filings, read every night · RAW, not a signal ·{" "}
            <a href="https://www.sec.gov/search-filings/edgar-application-programming-interfaces" target="_blank" rel="noreferrer">SEC EDGAR <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Could not load the record. It may answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}

      {!error && data && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub" data-vt-fundamentals-run>{runLine(data, Date.now())}</div>
          {changeLine(data) && <div className="vt-filings-sub">{changeLine(data)}</div>}

          <div className="vt-filings-title" style={{ fontSize: 14 }}>Look up a company</div>
          <form className="vt-filings-sub" onSubmit={onLookup} style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <input
              type="text" aria-label="Ticker or CIK" placeholder="Ticker or CIK" value={query}
              onChange={(e) => setQuery(e.target.value)} style={{ ...FIELD_STYLE, flex: "1 1 200px", minWidth: 0 }}
            />
            <button type="submit" className="vt-icon-btn" style={{ width: "auto", padding: "5px 12px" }} disabled={lookup === "loading" || data.warming_up}>
              {lookup === "loading" ? "Looking…" : "Show numbers"}
            </button>
          </form>
          {data.warming_up && <div className="vt-filings-state">Nothing recorded yet, so there is nothing to look up.</div>}
          {lookup === "none" && <div className="vt-filings-state">No company matches that ticker or CIK.</div>}
          {lookup === "error" && <div className="vt-filings-state">The lookup failed. Try again.</div>}
          {company && facts.length === 0 && (
            <div className="vt-filings-state">{companyTitle(company)}: none of the twelve recorded items is on file for the last seven quarters.</div>
          )}
          {company && facts.length > 0 && (
            <>
              <div className="vt-filings-title" data-vt-fundamentals-company style={{ fontSize: 14 }}>{companyTitle(company)}</div>
              <div className="vt-filings-sub">the value each table held at the last read; the link opens the filing SEC names for it</div>
              <div className="vt-filings-tablewrap vt-tablewrap-flow">
                <table className="vt-filings-table">
                  <thead><tr><th>Item</th><th>Period</th><th>Period end</th><th className="num">Reported</th><th>Filing</th></tr></thead>
                  <tbody>
                    {facts.map((f) => (
                      <tr key={f.key}>
                        <td data-l="Item">{f.label}</td>
                        <td data-l="Period">{f.period}</td>
                        <td data-l="Period end">{f.end}</td>
                        <td data-l="Reported" className="num">{f.amount}</td>
                        <td data-l="Filing"><a href={f.filing} target="_blank" rel="noreferrer">open <ExternalLink size={11} /></a></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <div className="vt-filings-title" style={{ fontSize: 14 }}>Numbers that changed after we recorded them</div>
          {revisions.length === 0 && (
            <div className="vt-filings-state">
              None yet. A change appears here when a later filing reports a different number for a period already on record.
            </div>
          )}
          {revisions.length > 0 && (
            <div className="vt-filings-tablewrap vt-tablewrap-flow">
              <table className="vt-filings-table">
                <thead><tr><th>Seen</th><th>Company</th><th>Item</th><th>Period</th><th className="num">Was</th><th className="num">Now</th><th className="num">Change</th><th>Filings</th></tr></thead>
                <tbody>
                  {shownRevisions.map((r) => (
                    <tr key={`${r.seen}-${r.cik}-${r.tag}-${r.period}`}>
                      <td data-l="Seen">{r.seen.slice(0, 10)}</td>
                      <td data-l="Company">{r.name ?? `CIK ${r.cik}`}{r.ticker ? ` (${r.ticker})` : ""}</td>
                      <td data-l="Item">{conceptLabel(r.tag, concepts)}</td>
                      <td data-l="Period">{periodLabel(r.period)}</td>
                      <td data-l="Was" className="num">{fmtAmount(r.prev_val, r.unit)}</td>
                      <td data-l="Now" className="num">{fmtAmount(r.val, r.unit)}</td>
                      <td data-l="Change" className="num">{fmtPct(revisionPct(r.prev_val, r.val))}</td>
                      <td data-l="Filings">
                        <span>
                          {r.prev_filing && <><a href={r.prev_filing} target="_blank" rel="noreferrer">before</a>{" · "}</>}
                          <a href={r.filing} target="_blank" rel="noreferrer">after</a>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {!allRevisions && revisions.length > REVISIONS_SHOWN && (
            <button type="button" className="vt-icon-btn" style={{ width: "auto", padding: "4px 10px" }} onClick={() => setAllRevisions(true)}>
              Show all {revisions.length} recorded changes
            </button>
          )}

          <div className="vt-filings-title" style={{ fontSize: 14 }}>What is recorded</div>
          <div className="vt-filings-tablewrap vt-tablewrap-flow">
            <table className="vt-filings-table">
              <thead><tr><th>Item</th><th>Newest quarter</th><th className="num">Companies</th><th className="num">New last read</th><th className="num">Changed last read</th></tr></thead>
              <tbody>
                {rows.map((c) => (
                  <tr key={c.tag}>
                    <td data-l="Item">{c.label}</td>
                    <td data-l="Newest quarter">{c.newestPeriod ? periodLabel(c.newestPeriod) : "—"}</td>
                    <td data-l="Companies" className="num">{c.companies.toLocaleString("en-US")}</td>
                    <td data-l="New last read" className="num">{c.added.toLocaleString("en-US")}</td>
                    <td data-l="Changed last read" className="num">{c.revised.toLocaleString("en-US")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="vt-filings-sub vt-streams-foot">
            {data.note} · {data.nights ?? 0} night{data.nights === 1 ? "" : "s"} recorded · source: {data.attribution ?? "SEC EDGAR"}
          </div>
        </div>
      )}
    </div>
  );
}
