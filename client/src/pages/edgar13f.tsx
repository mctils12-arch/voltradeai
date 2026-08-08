// Institutional13FView — SEC 13F-HR institutional holdings (#/data/filings13f).
// server/edgar13f.ts (GATE 1 DATA passed — parser hand-checked against two
// real filed filings, see edgar13f.test.ts) has archived 13F-HR filings
// since 2026-07-05 with /api/data/filings13f + /api/data/filings13f/history
// live and no client view until now — this closes that gap, the same shape
// as the Form-4 filings precedent (both are SEC EDGAR as-filed feeds). RAW
// display only (kind:"raw"): every value is the filing's own reported
// number; GATE 2 (new-position clustering vs forward returns) has not been
// attempted, no predictive claim is made here. Reuses vt-filings-* CSS only
// — no new styles needed.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Landmark } from "lucide-react";

interface Holding13F {
  issuer: string | null;
  titleOfClass: string | null;
  cusip: string | null;
  value: number | null;
  shares: number | null;
  sharesType: string | null;
  putCall: string | null;
  discretion: string | null;
}
interface Filing13F {
  accession: string;
  filedAt: string | null;
  cik: string;
  indexUrl: string;
  managerCik: string | null;
  managerName: string | null;
  periodOfReport: string | null;
  submissionType: string | null;
  entryTotal: number | null;
  valueTotal: number | null;
  otherManagersCount: number | null;
  holdings: Holding13F[] | null;
  holdingsOmitted: boolean;
}
interface HistoryPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  days?: number;
  count?: number;
  focused_cap?: number;
  filings: Filing13F[];
}

function fmtUSD(v: number | null): string {
  if (v == null) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}b`;
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(1)}m`;
  if (abs >= 1e3) return `$${(v / 1e3).toFixed(0)}k`;
  return `$${v.toLocaleString()}`;
}

export default function Institutional13FView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<HistoryPayload | null>(null);
  const [error, setError] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/filings13f/history?days=45");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const filings = data?.filings ?? [];
  const active = useMemo(
    () => filings.find((f) => f.accession === selected) ?? null,
    [filings, selected],
  );
  const holdings = useMemo(
    () => (active?.holdings ?? []).slice().sort((a, b) => (b.value ?? 0) - (a.value ?? 0)),
    [active],
  );

  return (
    <div className="vt-filings-page" role="region" aria-label="Institutional holdings (13F-HR)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Landmark size={16} />
        <div>
          <div className="vt-filings-title">Institutional holdings — SEC 13F-HR</div>
          <div className="vt-filings-sub">
            as filed · {(data?.count ?? filings.length).toLocaleString()} filings, last {data?.days ?? 45} days
            (archive began 2026-07-05) · gate 1 (data) passed, gate 2 (new-position clustering) not attempted ·
            filers over {(data?.focused_cap ?? 250).toLocaleString()} positions archive as summary-only ·{" "}
            <a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=13F-HR" target="_blank" rel="noreferrer">
              SEC EDGAR <ExternalLink size={11} />
            </a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && !data?.warming_up && data && filings.length === 0 && (
        <div className="vt-filings-state">
          No filings archived yet — the poll runs every 15 minutes; 13F-HR filings cluster around the quarterly
          deadline (Feb/May/Aug/Nov 14).
        </div>
      )}

      {filings.length > 0 && (
        <>
          <div className="vt-filings-tablewrap om-sb">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Manager</th><th>Period</th><th>Filed</th>
                  <th className="num">Positions</th><th className="num">Value</th>
                  <th aria-label="SEC link" />
                </tr>
              </thead>
              <tbody>
                {filings.slice(0, 300).map((f) => (
                  <tr
                    key={f.accession}
                    onClick={() => setSelected(f.accession === selected ? null : f.accession)}
                    style={{ cursor: "pointer" }}
                    aria-selected={f.accession === selected}
                  >
                    <td data-l="Manager">
                      <span className="vt-filings-ticker">{f.managerName || "—"}</span>
                      {f.otherManagersCount ? (
                        <span className="vt-filings-issuername">+{f.otherManagersCount} other filer(s)</span>
                      ) : null}
                    </td>
                    <td data-l="Period">{f.periodOfReport || "—"}</td>
                    <td data-l="Filed">{f.filedAt ? f.filedAt.slice(0, 10) : "—"}</td>
                    <td data-l="Positions" className="num">
                      {f.entryTotal != null ? f.entryTotal.toLocaleString() : "—"}
                      {f.holdingsOmitted ? " (summary only)" : ""}
                    </td>
                    <td data-l="Value" className="num">{fmtUSD(f.valueTotal)}</td>
                    <td>{f.indexUrl && (
                      <a href={f.indexUrl} target="_blank" rel="noreferrer" aria-label="Open SEC filing"
                         className="vt-filings-seclink" onClick={(e) => e.stopPropagation()}>
                        <ExternalLink size={13} />
                      </a>
                    )}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filings.length > 300 && <div className="vt-filings-state">showing first 300 of {filings.length.toLocaleString()} filings</div>}
          </div>

          <div className="vt-filings-sub">
            {active
              ? `Top holdings by value — ${active.managerName || "selected manager"}, period ${active.periodOfReport || "—"}`
              : "Click a filing above to see its top holdings by value."}
          </div>

          {active && active.holdingsOmitted && (
            <div className="vt-filings-state">
              This manager reported more than {(data?.focused_cap ?? 250).toLocaleString()} positions — archived as
              summary-only (holdings table omitted) per the focused-manager cap; mega-manager tables are
              index-hugging noise for a capacity-constrained-manager signal (EDGE DOCTRINE #2).
            </div>
          )}
          {active && !active.holdingsOmitted && holdings.length === 0 && (
            <div className="vt-filings-state">No holdings parsed for this filing.</div>
          )}
          {active && !active.holdingsOmitted && holdings.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Issuer</th><th>Class</th>
                    <th className="num">Value</th><th className="num">Shares</th><th>Discretion</th>
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((h, i) => (
                    <tr key={`${h.cusip}-${i}`}>
                      <td data-l="Issuer">{h.issuer || "—"}{h.putCall ? ` (${h.putCall})` : ""}</td>
                      <td data-l="Class">{h.titleOfClass || "—"}</td>
                      <td data-l="Value" className="num">{fmtUSD(h.value)}</td>
                      <td data-l="Shares" className="num">{h.shares != null ? h.shares.toLocaleString() : "—"}{h.sharesType ? ` ${h.sharesType}` : ""}</td>
                      <td data-l="Discretion">{h.discretion || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
