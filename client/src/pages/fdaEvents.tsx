// FdaEventsView — FDA binary events: openFDA approvals (backward-looking
// confirmations) + Federal Register advisory-committee meeting notices
// (forward-looking catalysts), #/data/fda-events.
// server/fdaEvents.ts (DATA STREAM EXPANSION stream #5, 2026-07-05) shipped
// the API-only route (/api/data/fda-events, keyless openFDA + Federal
// Register) with no client view — same "shipped-data-no-UI" gap class as
// treasuryDts.tsx/treasuryAuctions.tsx/tff.tsx. RAW display only; the
// module's own hypothesis (binary-event timing for biotech options — IV
// ramps into scheduled FDA catalysts, a theta-side input, not directional)
// stays gate-locked at gate 2 (not attempted). Reuses the generic
// .vt-filings-*/.vt-ladder-badge-* CSS — no new styles.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Pill } from "lucide-react";

interface FdaEvent {
  t: "approval" | "adcom";
  key: string;
  appl: string | null;
  date: string | null;
  sponsor: string | null;
  brand: string | null;
  committee: string | null;
  title: string | null;
  url: string | null;
  pub: string | null;
  dateConf: "parsed" | "unparsed" | null;
  rt: string;
}
interface FdaResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: string;
  count?: number;
  note?: string;
  warming_up?: boolean;
  events?: FdaEvent[];
}

const fmtDate = (d: string | null) => d || "—";

export default function FdaEventsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FdaResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/fda-events");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const { approvals, adcoms } = useMemo(() => {
    const events = data?.events || [];
    // Approvals are backward-looking confirmations — newest first.
    const ap = events.filter((e) => e.t === "approval").slice()
      .sort((a, b) => (b.date || "").localeCompare(a.date || ""));
    // AdCom notices are forward-looking catalysts — parsed dates soonest
    // first; unparsed dates (never guessed) sort after, newest-published
    // first so the most recent unconfirmed notice is still easy to find.
    const ac = events.filter((e) => e.t === "adcom").slice().sort((a, b) => {
      if (a.date && b.date) return a.date.localeCompare(b.date);
      if (a.date) return -1;
      if (b.date) return 1;
      return (b.pub || "").localeCompare(a.pub || "");
    });
    return { approvals: ap, adcoms: ac };
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="FDA binary events">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Pill size={16} />
        <div>
          <div className="vt-filings-title">FDA Binary Events</div>
          <div className="vt-filings-sub">
            drug approvals &amp; advisory-committee meeting notices (RAW, not a signal) ·{" "}
            {data && !data.warming_up ? `${data.count ?? 0} events` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://open.fda.gov/apis/drug/drugsfda/" target="_blank" rel="noreferrer">openFDA <ExternalLink size={11} /></a>
            {" + "}
            <a href="https://www.federalregister.gov/" target="_blank" rel="noreferrer">Federal Register <ExternalLink size={11} /></a>
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
            hypothesis under research (gate-locked, gate 2 not attempted): binary-event timing for
            biotech options — implied volatility ramps into a scheduled FDA catalyst, a theta-side
            input, not a directional call. Nothing below reflects that hypothesis yet — these are raw
            event listings, no model applied.
          </div>
          <div className="vt-filings-sub">
            no PDUFA target dates — legally unavailable free (21&nbsp;CFR&nbsp;314.430; ToS-protected
            aggregator scrapes are not used here). Advisory-committee meeting dates are parsed from the
            notice text where stated; a miss is labeled "date unconfirmed", never guessed.
          </div>

          {!data && <div className="vt-filings-state">Loading latest events…</div>}

          {data && (
            <>
              <div className="vt-filings-sub vt-shortvol-topheader">
                Approvals (backward-looking) — {approvals.length} in the last 30 days
              </div>
              {approvals.length === 0 && <div className="vt-filings-state">No approvals in the current window.</div>}
              {approvals.length > 0 && (
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Date</th><th>Application</th><th>Sponsor</th><th>Brand</th></tr>
                    </thead>
                    <tbody>
                      {approvals.map((e) => (
                        <tr key={e.key}>
                          <td data-l="Date">{fmtDate(e.date)}</td>
                          <td data-l="Application"><span className="vt-filings-ticker">{e.appl || "—"}</span></td>
                          <td data-l="Sponsor">{e.sponsor || "—"}</td>
                          <td data-l="Brand">{e.brand || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <div className="vt-filings-sub vt-shortvol-topheader">
                Advisory-committee meetings (forward-looking catalysts) — {adcoms.length} notices
              </div>
              {adcoms.length === 0 && <div className="vt-filings-state">No advisory-committee notices in the current window.</div>}
              {adcoms.length > 0 && (
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Meeting date</th><th>Committee</th><th>Sponsor</th><th>Notice</th></tr>
                    </thead>
                    <tbody>
                      {adcoms.map((e) => (
                        <tr key={e.key}>
                          <td data-l="Meeting date">
                            {e.date ? fmtDate(e.date) : "date unconfirmed"}{" "}
                            <span className={`vt-ladder-badge vt-ladder-badge-${e.dateConf === "parsed" ? "pass" : "pending"}`}>
                              {e.dateConf === "parsed" ? "parsed" : "unparsed"}
                            </span>
                          </td>
                          <td data-l="Committee">{e.committee || "—"}</td>
                          <td data-l="Sponsor">{e.sponsor || "—"}</td>
                          <td data-l="Notice">
                            {e.url ? (
                              <a href={e.url} target="_blank" rel="noreferrer">view <ExternalLink size={11} /></a>
                            ) : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
