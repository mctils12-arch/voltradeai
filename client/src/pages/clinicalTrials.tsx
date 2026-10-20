// ClinicalTrialsView — ClinicalTrials.gov catalyst-drift archive,
// #/data/clinical-trials. server/clinicalTrials.ts (v1.0.1102) records CHANGES
// in industry Phase 2/3 primary completion dates / statuses by diffing daily
// snapshots; v1.0.1112 added the identity-only sponsor->ticker annotation.
// The API shipped with no client view — same "shipped-data-no-UI" gap class
// as fdaEvents.tsx. RAW display only: the module's hypothesis (a PCD slip of
// >= 90 days leads operational trouble for small-cap sponsors) is gate-locked
// at gate 2 (needs >= 52 weeks of accumulation) and nothing here applies it.
// A ticker shown is a current-day name+SIC-lock identity match, never a call.
// Reuses the generic .vt-filings-*/.vt-ladder-badge-* CSS — no new styles.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, FlaskConical } from "lucide-react";

interface TrialEvent {
  t: "bootstrap" | "new" | "entered_window" | "pcd_shift" | "status" | "left_window";
  key: string;
  nct: string | null;
  sponsor: string | null;
  phase: string | null;
  status: string | null;
  prevStatus: string | null;
  pcd: string | null;
  prevPcd: string | null;
  shiftDays: number | null;
  coarse: boolean;
  title: string | null;
  cond: string | null;
  enroll: number | null;
  rt: string;
  ticker?: string | null;
  cik?: string | null;
}
interface TrialsResponse {
  kind: string;
  source?: string;
  attribution?: string;
  time?: number;
  tracked?: number;
  count?: number;
  note?: string;
  warming_up?: boolean;
  events?: TrialEvent[];
}

const KIND_LABEL: Record<TrialEvent["t"], string> = {
  bootstrap: "baseline",
  new: "new trial",
  entered_window: "entered window",
  pcd_shift: "date moved",
  status: "status change",
  left_window: "left window",
};

const fmt = (d: string | null) => d || "—";
const humanStatus = (s: string | null) => (s ? s.replace(/_/g, " ").toLowerCase() : "—");
const shiftLabel = (n: number | null) => (n == null ? "—" : `${n > 0 ? "+" : ""}${n} d`);

function ageLabel(ms: number | undefined, now: number): string {
  if (!ms) return "age unknown";
  const h = Math.max(0, Math.round((now - ms) / 3_600_000));
  return h < 1 ? "updated <1 h ago" : h < 48 ? `updated ${h} h ago` : `updated ${Math.round(h / 24)} d ago`;
}

export default function ClinicalTrialsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<TrialsResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/clinical-trials");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const { shifts, other, tickered } = useMemo(() => {
    const events = (data?.events || []).filter((e) => e.t !== "bootstrap");
    // Date moves are the archive's headline (the gate-2 hypothesis input) —
    // largest absolute shift first. Everything else newest-seen first.
    const sh = events.filter((e) => e.t === "pcd_shift").slice()
      .sort((a, b) => Math.abs(b.shiftDays ?? 0) - Math.abs(a.shiftDays ?? 0));
    const ot = events.filter((e) => e.t !== "pcd_shift").slice()
      .sort((a, b) => (b.rt || "").localeCompare(a.rt || ""));
    return { shifts: sh, other: ot, tickered: events.filter((e) => e.ticker).length };
  }, [data]);

  const loaded = !!data && !data.warming_up;
  const now = Date.now();

  const trialCells = (e: TrialEvent) => (
    <>
      <td data-l="Sponsor">
        {e.sponsor || "—"}{" "}
        {e.ticker && <span className="vt-filings-ticker" title="Identity match only (name + SIC lock), not a signal">{e.ticker}</span>}
      </td>
      <td data-l="Trial">
        {e.nct ? (
          <a href={`https://clinicaltrials.gov/study/${e.nct}`} target="_blank" rel="noreferrer">
            {e.nct} <ExternalLink size={11} />
          </a>
        ) : "—"}
      </td>
    </>
  );

  return (
    <div className="vt-filings-page" role="region" aria-label="ClinicalTrials.gov catalyst drift">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <FlaskConical size={16} />
        <div>
          <div className="vt-filings-title">Clinical-Trial Date Drift</div>
          <div className="vt-filings-sub">
            changes to sponsor-reported primary completion dates &amp; status (RAW, not a signal) ·{" "}
            {loaded ? `${data?.count ?? 0} changes, ${data?.tracked ?? 0} trials tracked · ${ageLabel(data?.time, now)}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://clinicaltrials.gov/" target="_blank" rel="noreferrer">ClinicalTrials.gov <ExternalLink size={11} /></a>
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
            hypothesis under research (gate-locked, gate 2 not attempted): a sponsor pushing a primary
            completion date out ≥&nbsp;90 days leads operational trouble, concentrated in small-cap
            sponsors. Testing needs ≥&nbsp;52 weeks of recorded history; nothing below applies it —
            these are raw change records, last 14 days.
          </div>
          <div className="vt-filings-sub">
            dates are what the sponsor self-reports (estimated until completion); "~" marks a
            month-only date. A ticker appears only where the sponsor's exact name matches an SEC filer
            with a pharma/biotech SIC code — a current-day identity match
            ({tickered} of {shifts.length + other.length} here), not point-in-time and not a call.
          </div>

          {!data && <div className="vt-filings-state">Loading latest changes…</div>}

          {data && (
            <>
              <div className="vt-filings-sub vt-shortvol-topheader">
                Primary completion date moved — {shifts.length}
              </div>
              {shifts.length === 0 && <div className="vt-filings-state">No date moves in the current window.</div>}
              {shifts.length > 0 && (
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Seen</th><th>Sponsor</th><th>Trial</th><th>Previous</th><th>Now</th><th>Shift</th></tr>
                    </thead>
                    <tbody>
                      {shifts.map((e) => (
                        <tr key={e.key}>
                          <td data-l="Seen">{fmt(e.rt)}</td>
                          {trialCells(e)}
                          <td data-l="Previous">{fmt(e.prevPcd)}</td>
                          <td data-l="Now">{e.coarse ? "~" : ""}{fmt(e.pcd)}</td>
                          <td data-l="Shift">
                            <span className={`vt-ladder-badge vt-ladder-badge-${(e.shiftDays ?? 0) > 0 ? "pending" : "pass"}`}>
                              {shiftLabel(e.shiftDays)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <div className="vt-filings-sub vt-shortvol-topheader">
                Other changes (new, status, window entry/exit) — {other.length}
              </div>
              {other.length === 0 && <div className="vt-filings-state">No other changes in the current window.</div>}
              {other.length > 0 && (
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Seen</th><th>Sponsor</th><th>Trial</th><th>Change</th><th>Status</th><th>Primary completion</th></tr>
                    </thead>
                    <tbody>
                      {other.map((e) => (
                        <tr key={e.key}>
                          <td data-l="Seen">{fmt(e.rt)}</td>
                          {trialCells(e)}
                          <td data-l="Change">{KIND_LABEL[e.t]}</td>
                          <td data-l="Status">
                            {e.t === "status" && e.prevStatus
                              ? `${humanStatus(e.prevStatus)} → ${humanStatus(e.status)}`
                              : humanStatus(e.status)}
                          </td>
                          <td data-l="Primary completion">{e.coarse ? "~" : ""}{fmt(e.pcd)}</td>
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
