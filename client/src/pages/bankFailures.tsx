// BankFailuresView — FDIC bank failures event stream, #/data/bank-failures.
// server/fdicBanks.ts (BUILD ORDER 6 #3 v1, 2026-07-06) shipped the
// API-only route (/api/data/bank-failures, keyless FDIC Bank Data API)
// with a server-side event poll but no client view and no
// datacore/layers.json entry at all — the same "double zero" gap class
// vehicleComplaints.tsx (2026-08-18) closed for /api/data/vehicle-
// complaints, the last other zero-wiring candidate the 2026-08-16 TFF
// sweep found. v1 SCOPE — FAILURES ONLY (event-driven; the quarterly
// financials snapshot is a documented, separate follow-up in
// fdicBanks.ts, not this route). RAW display only; the module's own
// hypothesis (deposit flight + failures at SMALL regional banks lead
// KRE and listed small-cap banks — EDGE DOCTRINE #2 whales-can't-fish
// territory) stays gate-locked at gate 2 (not attempted — no ticker
// join exists here; most failed institutions are small, non-public
// regional banks, unlike the NHTSA feed's pre-joined watchlist). Gate 1
// (DATA — QBFASSET/QBFDEP cross-checked against the bank's own FDIC
// Call Reports) ran 2026-08-18: PASS with a documented caveat — see
// research/open_questions.md's FDIC BANK DATA entry and
// datacore/signal_ladder.json's fdic_bank_failures entry.
// Reuses the generic .vt-filings-*/.vt-shortvol-* CSS — no new styles.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Building2 } from "lucide-react";

interface BankFailure {
  cert: number | null;
  name: string;
  fail_date: string;
  city_state: string | null;
  state: string | null;
  charter_class: string | null;
  restype: string | null;
  assets_k: number | null;
  deposits_k: number | null;
  cost_k: number | null;
  fund: string | null;
  rt: string;
}
interface FailuresResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: string;
  count?: number;
  note?: string;
  warming_up?: boolean;
  failures?: BankFailure[];
}

// Amounts arrive in $ thousands, as published (assets_k/deposits_k/cost_k) —
// same "state the published unit" convention as treasuryDts.tsx's $M lines.
const fmtK = (n: number | null | undefined) =>
  n == null ? "—" : `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}K`;

export default function BankFailuresView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FailuresResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/bank-failures");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  // Newest failure first — a Friday-evening event feed, most-recent-first
  // is the natural read order (same convention as fdaEvents.tsx's approvals).
  const failures = useMemo(() => {
    return (data?.failures || []).slice().sort((a, b) => b.fail_date.localeCompare(a.fail_date));
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="FDIC bank failures">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Building2 size={16} />
        <div>
          <div className="vt-filings-title">Bank Failures</div>
          <div className="vt-filings-sub">
            FDIC bank failure &amp; assistance events (RAW, not a signal) ·{" "}
            {data && !data.warming_up ? `${data.count ?? 0} events` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://banks.data.fdic.gov/docs/" target="_blank" rel="noreferrer">FDIC Bank Data API <ExternalLink size={11} /></a>
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
            v1 scope is FAILURES ONLY — the event-driven live kicker, back to 1934. The quarterly
            financials snapshot (~4.6k banks, deposit/asset-quality trends ahead of failure) is a
            documented follow-up, not this feed. Most failed institutions here are small, non-public
            regional banks — no ticker join exists in this route, unlike the pre-joined NHTSA
            watchlist elsewhere on this map.
          </div>
          <div className="vt-filings-sub">
            hypothesis under research (gate-locked, gate 2 not attempted): deposit flight and
            failures at small regional banks lead KRE and listed small-cap bank returns — a
            whales-can't-fish edge (capacity-constrained funds can't size into names this small).
            Nothing below reflects that hypothesis yet — this is a raw event listing, no model
            applied. Amounts are $ thousands as published; estimated DIF loss ("Cost") stays blank
            until FDIC publishes an estimate — never shown as zero.
          </div>
          <div className="vt-filings-sub">
            gate 1 (data ground-truth check, 2026-08-18): Assets/Deposits sampled across 4 failures
            matched the institution's own last FDIC Call Report on file exactly in 3 of 4 cases
            (confirmed against the FDIC's own published press releases). The one mismatch was the
            single most-recent failure at check time, where the figures ran ahead of the newest
            Call Report FDIC's public financials index had on file — most likely that index lagging
            a final pre-failure filing, not a defect in this feed. Treat Assets/Deposits as FDIC's
            own reported figures for the institution, which can occasionally be more current than
            its last published quarterly Call Report for a very recent failure.
          </div>

          {!data && <div className="vt-filings-state">Loading latest failures…</div>}
          {data && failures.length === 0 && <div className="vt-filings-state">No failures in the current polled window.</div>}

          {failures.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Fail date</th><th>Institution</th><th>State</th><th>Type</th>
                    <th className="num">Assets</th><th className="num">Deposits</th><th className="num">Est. DIF cost</th>
                  </tr>
                </thead>
                <tbody>
                  {failures.map((f) => (
                    <tr key={`${f.cert ?? f.name}-${f.fail_date}`}>
                      <td data-l="Fail date">{f.fail_date}</td>
                      <td data-l="Institution">{f.name}</td>
                      <td data-l="State">{f.state || "—"}</td>
                      <td data-l="Type">{f.restype || "—"}</td>
                      <td data-l="Assets" className="num">{fmtK(f.assets_k)}</td>
                      <td data-l="Deposits" className="num">{fmtK(f.deposits_k)}</td>
                      <td data-l="Est. DIF cost" className="num">{fmtK(f.cost_k)}</td>
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
