// DtccSwapsView — DTCC SBSDR equity total-return-swap dissemination
// (#/data/dtcc-swaps). server/dtccSwaps.ts (GATE 1 DATA passed 2026-08-22,
// datacore/signal_ladder.json id "dtcc_sbsdr_equity_swaps") shipped a
// status/count-only surface with no client view until now — same "next
// PRODUCT session" sequencing as the JODI/VIX precedents this page reuses
// the shell from. RAW display only (kind:"raw"): these are individual
// swap-dissemination events DTCC itself published (SEC Reg SBSR real-time
// reporting), not a derived reading — no predictive claim is made, and the
// route's own gate-2-locked note is restated verbatim rather than implied.
// top_rows is explicitly the largest-notional events from the source
// file's most recently published day, not a running archive-wide
// ranking (see server/dtccSwaps.ts's topNotionalRows doc comment) — the
// UI says so rather than reading like a full-archive browser.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Repeat } from "lucide-react";

interface DtccRow {
  dissemination_id: string;
  action_type: string;
  event_timestamp: string;
  effective_date: string;
  notional_amount: number | null;
  notional_currency: string;
  underlier_id: string;
  underlier_id_source: string;
  underlier_name: string;
}
interface DtccPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  file_date?: string;
  source_date?: string;
  us_underlier_rows_today?: number;
  new_rows_archived?: number;
  total_archived?: number;
  top_rows: DtccRow[];
  note?: string;
}

const notional = (n: number | null, ccy: string) =>
  n == null ? "— (masked)" : `${ccy} ${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export default function DtccSwapsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<DtccPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/dtcc-swaps");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = data?.top_rows ?? [];

  return (
    <div className="vt-filings-page" role="region" aria-label="DTCC SBSDR equity swap dissemination">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Repeat size={16} />
        <div>
          <div className="vt-filings-title">Equity swap dissemination — DTCC SBSDR</div>
          <div className="vt-filings-sub">
            US-underlier total-return-swap events, largest notional first · RAW ·{" "}
            <a href="https://pddata.dtcc.com/gtr/sbsdr" target="_blank" rel="noreferrer">DTCC SBSDR <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data?.warming_up && <div className="vt-filings-state">Warming up — first poll not yet complete.</div>}
      {!error && data && !data.warming_up && rows.length === 0 && (
        <div className="vt-filings-state">No US-underlier events in the most recently published file.</div>
      )}

      {!error && data && !data.warming_up && rows.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            source file dated {data.source_date} · {data.us_underlier_rows_today?.toLocaleString()} US-underlier
            rows that day · {data.new_rows_archived?.toLocaleString()} newly archived ·{" "}
            {data.total_archived?.toLocaleString()} total events archived to date
          </div>

          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Underlier</th>
                  <th>ID</th>
                  <th>Action</th>
                  <th>Event time</th>
                  <th className="num">Notional</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.dissemination_id}>
                    <td data-l="Underlier">{row.underlier_name || "—"}</td>
                    <td data-l="ID">{row.underlier_id} <span className="vt-filings-sub">({row.underlier_id_source})</span></td>
                    <td data-l="Action">{row.action_type}</td>
                    <td data-l="Event time">{row.event_timestamp}</td>
                    <td data-l="Notional" className="num">{notional(row.notional_amount, row.notional_currency)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="vt-filings-sub vt-streams-foot">
            {data.attribution} · largest-notional events from the source's most recently published day only, not a
            running archive-wide ranking · {data.note}
          </div>
        </div>
      )}
    </div>
  );
}
