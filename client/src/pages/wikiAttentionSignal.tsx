// WikiAttentionSignalView — the SECOND live SIGNAL detail page for a root
// past ladder gate 2 (#/data/wiki-attention-signal), after
// gnssIntegritySignal.tsx. datacore/signal_ladder.json's
// wikimedia_pageviews_attention entry: GATE 2 (SIGNAL) PASS 2026-09-04 for
// the VOLUME channel — a pageview attention spike on a small/mid-cap seed
// ticker is followed by elevated forward trading volume, net of a
// same-day-or-prior-day SEC 8-K (the news-free control). Reads
// /api/data/wiki-attention-signal (server/wikiAttentionSignal.ts) — a
// live, no-token board computed over this repo's own rolling pageviews
// archive. PREMIUM EXPERIENCE STANDARD (c): the validated result and every
// honest caveat (news check is NOT live, no volatility/price claim, GATE 3
// not attempted) are surfaced as prominently as the live spike board
// itself — this is a SIGNAL, not tradeable, and says so.
// Reuses .vt-filings-*/.vt-shortvol-*/.vt-ladder-badge-* — no new CSS.
import { useEffect, useState } from "react";
import { ArrowLeft, TrendingUp } from "lucide-react";

interface TickerRow {
  ticker: string; article: string; cap_tier: "small_mid" | "mega";
  latest_date: string | null; current_views: number | null; baseline_mean: number | null;
  baseline_days: number; baseline_complete: boolean; z_score: number | null; spike: boolean;
}
interface EffectRow { horizon_days: number; mean_ratio: number; baseline_ratio: number; p_value: number; }

interface SignalPayload {
  kind: "signal";
  generated_at?: string;
  gate?: { current_gate: number; status: string; channel: string };
  z_threshold?: number;
  trailing_window_days?: number;
  min_baseline_days?: number;
  tickers?: TickerRow[];
  spike_count?: number;
  validated_effect?: { study_date: string; bonferroni_alpha: number; small_mid: EffectRow[]; mega: EffectRow[] };
  methodology_note?: string;
  caveats?: string[];
  license?: { source: string; note: string };
}

const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString());
// "Excess" = mean_ratio - baseline_ratio, in percentage points of the same
// ratio scale the research/open_questions.md write-up itself reports
// ("diff 0.240" == +24.0pp) — not a percent-of-percent recomputation.
const excessPp = (mean: number, baseline: number) => `+${((mean - baseline) * 100).toFixed(1)}pp`;
const pval = (p: number) => (p < 1e-4 ? "<0.0001" : p.toFixed(4));

export default function WikiAttentionSignalView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<SignalPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/wiki-attention-signal");
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const tickers = data?.tickers ?? [];

  return (
    <div className="vt-filings-page" role="region" aria-label="Wikipedia attention spike-to-volume signal">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <TrendingUp size={16} />
        <div>
          <div className="vt-filings-title">Attention spike → forward volume signal</div>
          <div className="vt-filings-sub">
            {data
              ? <>gate 2 (SIGNAL) {(data.gate?.status || "").replace("gate2_", "")} · {data.spike_count ?? 0} of {tickers.length} tickers currently spiking</>
              : error ? "feed error — retry on refresh" : "loading…"}
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Could not load the signal — the archive may still answer on refresh.</div>}

      {!error && data && (
        <div className="vt-streams-body om-sb">
          <section className="vt-quality-section">
            <div className="vt-quality-section-head">
              <span>Live attention board</span>
              <span className="vt-quality-section-sub">
                z-score of latest pageviews vs. each ticker's own trailing baseline · spike at z≥{data.z_threshold ?? 2}
              </span>
            </div>
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Ticker</th><th>Cap tier</th><th>Latest day</th>
                    <th className="num">Views</th><th className="num">Baseline</th>
                    <th className="num">z-score</th><th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {tickers.map((t) => (
                    <tr key={t.ticker}>
                      <td data-l="Ticker"><span className="vt-filings-ticker">{t.ticker}</span></td>
                      <td data-l="Cap tier">{t.cap_tier === "mega" ? "mega" : "small/mid"}</td>
                      <td data-l="Latest day">{t.latest_date || "—"}</td>
                      <td data-l="Views" className="num">{fmt(t.current_views)}</td>
                      <td data-l="Baseline" className="num">
                        {fmt(t.baseline_mean)}
                        {t.baseline_days > 0 && !t.baseline_complete && (
                          <span className="vt-filings-sub" style={{ marginLeft: 4 }}>({t.baseline_days}d)</span>
                        )}
                      </td>
                      <td data-l="z-score" className="num">{t.z_score == null ? "—" : t.z_score.toFixed(2)}</td>
                      <td data-l="Status">
                        <span className={`vt-ladder-badge vt-ladder-badge-${t.spike ? "pass" : t.z_score == null ? "pending" : "raw"}`}>
                          {t.spike ? "spiking" : t.z_score == null ? "no data yet" : "normal"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {tickers.length === 0 && <div className="vt-filings-state">No archived rows yet — the panel fills in as the poller runs.</div>}
          </section>

          <section className="vt-quality-section">
            <div className="vt-quality-section-head">
              <span>Validated effect (pooled, historical)</span>
              <span className="vt-quality-section-sub">
                news-free control, {data.validated_effect?.study_date} · Bonferroni bar p&lt;{data.validated_effect?.bonferroni_alpha}
              </span>
            </div>
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr><th>Group</th><th className="num">Horizon</th><th className="num">Mean ratio</th><th className="num">Baseline ratio</th><th className="num">Excess</th><th className="num">p-value</th></tr>
                </thead>
                <tbody>
                  {(data.validated_effect?.small_mid || []).map((r) => (
                    <tr key={`sm-${r.horizon_days}`}>
                      <td data-l="Group">small/mid</td>
                      <td data-l="Horizon" className="num">{r.horizon_days}d</td>
                      <td data-l="Mean ratio" className="num">{r.mean_ratio.toFixed(3)}x</td>
                      <td data-l="Baseline ratio" className="num">{r.baseline_ratio.toFixed(3)}x</td>
                      <td data-l="Excess" className="num">{excessPp(r.mean_ratio, r.baseline_ratio)}</td>
                      <td data-l="p-value" className="num">
                        <span className={`vt-ladder-badge vt-ladder-badge-${r.p_value < (data.validated_effect?.bonferroni_alpha ?? 0.005) ? "pass" : "fail"}`}>
                          {pval(r.p_value)}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {(data.validated_effect?.mega || []).map((r) => (
                    <tr key={`mega-${r.horizon_days}`}>
                      <td data-l="Group">mega (comparison)</td>
                      <td data-l="Horizon" className="num">{r.horizon_days}d</td>
                      <td data-l="Mean ratio" className="num">{r.mean_ratio.toFixed(3)}x</td>
                      <td data-l="Baseline ratio" className="num">{r.baseline_ratio.toFixed(3)}x</td>
                      <td data-l="Excess" className="num">{excessPp(r.mean_ratio, r.baseline_ratio)}</td>
                      <td data-l="p-value" className="num">
                        <span className={`vt-ladder-badge vt-ladder-badge-${r.p_value < (data.validated_effect?.bonferroni_alpha ?? 0.005) ? "pass" : "fail"}`}>
                          {pval(r.p_value)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
            <div className="vt-filings-sub">generated {data.generated_at}</div>
          </section>
        </div>
      )}
    </div>
  );
}
