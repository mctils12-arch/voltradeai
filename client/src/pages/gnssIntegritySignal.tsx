// GnssIntegritySignalView — the FIRST live SIGNAL detail page for a root
// past ladder gate 2 (#/data/gnss-integrity). datacore/signal_ladder.json's
// gnss_integrity_adsb entry: GATE 2 (SIGNAL) PASS, re-confirmed and
// strengthened 2026-08-15 at 4 accumulated archive days; GATE 1 is
// PARTIAL — DTU Space's Bornholm RF station independently corroborates the
// phenomenon/region, not the exact sample days (research/open_questions.md).
// Reads /api/data/gnss-integrity-signal (server/gnssIntegritySignal.ts) —
// a live, no-token aggregate computed over a rolling window of the
// aircraft archive. PREMIUM EXPERIENCE STANDARD (c): every number here
// carries freshness/provenance, and the honest caveats (gate-1-partial,
// small sample) are surfaced as prominently as the pass verdict itself —
// this is a SIGNAL, not tradeable, and says so.
// Reuses .vt-filings-*/.vt-quality-*/.vt-ladder-badge-* — no new CSS.
import { useEffect, useState } from "react";
import { ArrowLeft, Radio } from "lucide-react";

interface BandRow {
  band: string;
  candidate_k: number;
  candidate_n: number;
  control_rate: number;
  expected_under_null: number;
  p_value: number;
  elevated: boolean;
  expected_to_elevate: boolean;
}

interface SignalPayload {
  kind: "signal";
  warming_up?: boolean;
  root_id?: string;
  generated_at?: string;
  gate?: { current_gate: number; status: string };
  verdict?: "PASS" | "FAIL" | "INCONCLUSIVE";
  bands?: BandRow[];
  region?: {
    candidate_label: string; candidate_bbox: number[];
    control_label: string; control_bbox: number[];
  };
  freshness?: {
    writer_live_since: string;
    candidate: { days_read: string[]; days_missing: string[]; rows_scanned: number; truncated: boolean };
    control: { days_read: string[]; days_missing: string[]; rows_scanned: number; truncated: boolean };
  };
  methodology_note?: string;
  caveats?: string[];
  license?: { source: string; note: string };
  note?: string;
}

const BAND_LABEL: Record<string, string> = {
  cruise: "Cruise (≥25,000 ft)", mid: "Mid (10–25,000 ft)", low: "Low (<10,000 ft)", ground: "Ground",
};

function pct(n: number): string {
  return `${(n * 100).toFixed(n < 0.01 ? 3 : 2)}%`;
}
function pval(p: number): string {
  return p < 1e-6 ? "<0.000001" : p.toFixed(6);
}

export default function GnssIntegritySignalView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<SignalPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/gnss-integrity-signal");
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const bands = data?.bands ?? [];

  return (
    <div className="vt-filings-page" role="region" aria-label="GPS/GNSS integrity anomaly signal">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Radio size={16} />
        <div>
          <div className="vt-filings-title">GNSS integrity anomaly — Baltic corridor</div>
          <div className="vt-filings-sub">
            {data && !data.warming_up
              ? <>gate 2 (SIGNAL) {(data.gate?.status || "").replace("gate2_", "")} · derived from our own ADS-B position archive</>
              : error ? "feed error — retry on refresh" : "loading…"}
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Could not load the signal — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && <div className="vt-filings-state">First archive scan in progress — retry shortly.</div>}

      {!error && data && !data.warming_up && (
        <div className="vt-streams-body om-sb">
          <section className="vt-quality-section">
            <div className="vt-quality-section-head">
              <span>Verdict</span>
              <span className="vt-quality-section-sub">
                {data.region?.candidate_label} vs. {data.region?.control_label}
              </span>
            </div>
            <div className="vt-ladder-row">
              <div className="vt-ladder-row-top">
                <span className="vt-ladder-row-name">GPS position-integrity degradation (nic==0), broadcast-origin only</span>
                <span className={`vt-ladder-badge vt-ladder-badge-${data.verdict === "PASS" ? "pass" : data.verdict === "FAIL" ? "fail" : "pending"}`}>
                  {data.verdict === "PASS" ? "gate 2 pass" : data.verdict || "—"}
                </span>
              </div>
            </div>
          </section>

          <section className="vt-quality-section">
            <div className="vt-quality-section-head">
              <span>Per-altitude-band statistics</span>
              <span className="vt-quality-section-sub">one-tailed exact binomial test, p&lt;0.01 significance bar</span>
            </div>
            {bands.length === 0 && (
              <div className="vt-filings-state">No broadcast-origin rows in both regions for the current window yet.</div>
            )}
            {bands.length > 0 && (
              <div className="vt-filings-tablewrap">
                <table className="vt-filings-table">
                  <thead>
                    <tr>
                      <th>Band</th>
                      <th className="num">Candidate</th>
                      <th className="num">Candidate rate</th>
                      <th className="num">Control rate</th>
                      <th className="num">p-value</th>
                      <th>Elevated?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bands.map((b) => (
                      <tr key={b.band}>
                        <td data-l="Band">{BAND_LABEL[b.band] || b.band}</td>
                        <td data-l="Candidate" className="num">{b.candidate_k}/{b.candidate_n}</td>
                        <td data-l="Candidate rate" className="num">{pct(b.candidate_n ? b.candidate_k / b.candidate_n : 0)}</td>
                        <td data-l="Control rate" className="num">{pct(b.control_rate)}</td>
                        <td data-l="p-value" className="num">{pval(b.p_value)}</td>
                        <td data-l="Elevated?">
                          <span className={`vt-ladder-badge vt-ladder-badge-${b.elevated ? "pass" : "raw"}`}>
                            {b.elevated ? "elevated" : "not elevated"}
                          </span>
                          {b.expected_to_elevate && <span className="vt-filings-sub" style={{ marginLeft: 6 }}>(expected)</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="vt-quality-section">
            <div className="vt-quality-section-head"><span>Freshness</span></div>
            <div className="vt-filings-sub">
              candidate region: {data.freshness?.candidate.days_read.length ?? 0} day(s) read
              {data.freshness?.candidate.days_missing.length ? `, ${data.freshness.candidate.days_missing.length} missing` : ""}
              {" "}({data.freshness?.candidate.rows_scanned.toLocaleString()} rows scanned) ·
              archive carries this field only since {data.freshness?.writer_live_since} · generated {data.generated_at}
            </div>
          </section>

          <section className="vt-quality-section">
            <div className="vt-quality-section-head"><span>Method &amp; honest caveats</span></div>
            <div className="vt-filings-sub">{data.methodology_note}</div>
            {(data.caveats || []).map((c, i) => (
              <div key={i} className="vt-filings-sub" style={{ color: "var(--accent-orange)" }}>⚠ {c}</div>
            ))}
          </section>

          <section className="vt-quality-section">
            <div className="vt-quality-section-head"><span>Source &amp; license</span></div>
            <div className="vt-filings-sub">{data.license?.note}</div>
            <div className="vt-filings-sub">
              Gate-1 phenomenon corroboration: DTU Space's Tein RF monitoring station on Bornholm, as
              reported by Danish public broadcaster DR (2026-08-15) — independent of this signal's own
              ADS-B source, not derived from it.
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
