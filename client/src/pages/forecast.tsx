// Weekly forecast screen (2026-10-27, human's Phase 7; SHADOW). One call per
// name for the next 5 trading sessions — SPY, every open position, the last
// scan's top candidates — with its direction, its probability and a 68% price
// range, plus how past calls scored and how well their probabilities are
// calibrated. Every number carries its batch time, base close and model
// version; the model is labeled experimental and nothing trades on it.
// Data: GET /api/forecast (forecast.py report(); owner only).
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowDownRight, ArrowUpRight, FlaskConical, RefreshCw, Target } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import {
  type ForecastReport, type ForecastCall, type CalBin, type Accuracy,
  GROUP_LABEL, VOL_LABEL, pct, usd, bandPct, ageText, callState, skillText, calChart, dotRadius,
} from "@/lib/forecast";

const POLL_MS = 5 * 60_000;
type Filter = "all" | "index" | "position" | "candidate";
const FILTER_LABEL: Record<Filter, string> = { all: "All", index: "Index", position: "Positions", candidate: "Candidates" };

async function fetchForecast(): Promise<ForecastReport> {
  const r = await apiRequest("GET", "/api/forecast", undefined, { timeoutMs: 30_000 });
  return (await r.json()) as ForecastReport;
}

export default function ForecastPage() {
  const q = useQuery({ queryKey: ["/api/forecast"], queryFn: fetchForecast, refetchInterval: POLL_MS, staleTime: 60_000 });
  const [filter, setFilter] = useState<Filter>("all");
  const now = Date.now();
  const rep = q.data;
  const latest = rep?.latest ?? null;
  const calls = useMemo(() => (latest?.forecasts ?? []).filter((c) => filter === "all" || c.group === filter), [latest, filter]);
  const counts = useMemo(() => {
    const out: Record<Filter, number> = { all: 0, index: 0, position: 0, candidate: 0 };
    for (const c of latest?.forecasts ?? []) { out.all += 1; out[c.group as Filter] = (out[c.group as Filter] ?? 0) + 1; }
    return out;
  }, [latest]);

  return (
    <div className="vt-fc" data-vt-forecast>
      <header className="vt-fc-head">
        <div className="vt-fc-title">
          <Target size={18} aria-hidden="true" />
          <h1>Weekly forecast</h1>
          <span className="vt-fc-chip" data-tone="orange" title="A simple statistical model under evaluation — read its skill numbers before trusting a call"><FlaskConical size={12} aria-hidden="true" /> Experimental model</span>
          <span className="vt-fc-chip" title="Display and tracking only: no order, size or threshold reads these calls">Shadow · nothing trades on these</span>
        </div>
        <p className="vt-fc-sub">
          Direction, probability and a 68% price range for the next {rep?.horizon_days ?? 5} trading sessions — SPY, every open position and the
          last scan's top candidates. A call is always made (up when P(up) ≥ 50%), and its probability is the model's, never rounded up.
        </p>
        <div className="vt-fc-fresh" aria-live="polite">
          {latest ? (
            <>
              <span>made <b>{ageText(latest.made_at, now)}</b> ago</span>
              <span>base close <b>{latest.base_date}</b></span>
              <span>horizon → <b>{latest.horizon_end}</b></span>
              <span>model <b className="vt-fc-m">{latest.model_version}</b></span>
              {!latest.on_time ? <span className="vt-fc-warn">late batch — made after the first horizon session opened</span> : null}
            </>
          ) : null}
          {rep ? <span>page data {ageText(rep.generated_at, now)} old</span> : null}
          <button type="button" className="vt-fc-btn" onClick={() => { void q.refetch(); }} disabled={q.isFetching} aria-label="Refresh forecast">
            <RefreshCw size={13} aria-hidden="true" className={q.isFetching ? "vt-fc-spin" : undefined} /> {q.isFetching ? "refreshing" : "refresh"}
          </button>
        </div>
      </header>

      {q.isLoading ? <Skeleton /> : q.isError && !rep ? (
        <div className="vt-fc-state" data-tone="red" role="alert">
          <b>Forecast service unavailable.</b> {String(q.error instanceof Error ? q.error.message : q.error).slice(0, 160)} — the page retries every 5 minutes.
        </div>
      ) : rep && !latest ? (
        <div className="vt-fc-state">
          <b>No weekly batch yet.</b> The first batch is made at the first check after a session closes (normally Friday after 16:15 ET) and lands here on its own.
          {rep.open ? ` ${rep.open} open calls are waiting for their horizon.` : ""}
        </div>
      ) : rep && latest ? (
        <>
          <section className="vt-fc-sec" aria-labelledby="vt-fc-calls">
            <div className="vt-fc-sechead">
              <h2 id="vt-fc-calls">This week's calls</h2>
              <div className="vt-fc-filters" role="group" aria-label="Filter calls">
                {(["all", "index", "position", "candidate"] as Filter[]).map((f) => (
                  <button key={f} type="button" className="vt-fc-btn" aria-pressed={filter === f} onClick={() => setFilter(f)} disabled={f !== "all" && !counts[f]}>
                    {FILTER_LABEL[f]} <span className="vt-fc-m">{counts[f] ?? 0}</span>
                  </button>
                ))}
              </div>
            </div>
            {!latest.candidates_available ? <p className="vt-fc-note">No scan candidates were available when this batch was made (the scan had not run since the last restart) — positions and SPY only.</p> : null}
            <div className="vt-fc-grid">
              {calls.map((c) => <CallCard key={c.id} c={c} />)}
            </div>
            {latest.skipped.length ? <p className="vt-fc-note">Not forecast: {latest.skipped.map((s) => `${s.symbol} (${s.reason.replace(/_/g, " ")})`).join(", ")}.</p> : null}
            <p className="vt-fc-note">
              Range = 68% band of a zero-drift lognormal; {latest.iv_names} of {latest.n_forecasts} use chain IV30{latest.iv_day ? ` (options archive, ${latest.iv_day})` : ""}, the rest 20-day realized vol.
              {latest.candidates_scan_at ? ` Candidates from the scan at ${new Date(latest.candidates_scan_at).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })} ET.` : ""}
            </p>
          </section>

          <section className="vt-fc-sec" aria-labelledby="vt-fc-acc">
            <div className="vt-fc-sechead">
              <h2 id="vt-fc-acc">Track record</h2>
              <span className="vt-fc-meta">{rep.accuracy.n} resolved · {rep.open} open · as of {ageText(rep.generated_at, now)} ago</span>
            </div>
            <AccuracyTiles a={rep.accuracy} />
            {rep.accuracy.n > 0 && rep.accuracy.on_time.n !== rep.accuracy.n ? <p className="vt-fc-note">On-time batches only: {rep.accuracy.on_time.n} calls, {pct(rep.accuracy.on_time.hit_rate)} direction hits.</p> : null}
          </section>

          <section className="vt-fc-sec" aria-labelledby="vt-fc-cal">
            <div className="vt-fc-sechead">
              <h2 id="vt-fc-cal">Calibration</h2>
              <span className="vt-fc-meta">predicted P(up) vs how often the price actually rose</span>
            </div>
            <CalibrationChart live={rep.calibration.bins} holdout={latest.model.validation.calibration} />
            <ModelCard latest={latest} />
          </section>

          {rep.weeks.length ? (
            <section className="vt-fc-sec" aria-labelledby="vt-fc-weeks">
              <div className="vt-fc-sechead"><h2 id="vt-fc-weeks">By week</h2><span className="vt-fc-meta">newest first</span></div>
              <div className="vt-fc-tablewrap">
                <table className="vt-fc-table">
                  <thead><tr><th>Week</th><th>Base close</th><th>Calls</th><th>Resolved</th><th>Hits</th><th>Brier</th><th>In range</th><th>Model</th></tr></thead>
                  <tbody>
                    {rep.weeks.map((w) => (
                      <tr key={w.week_key}>
                        <td className="vt-fc-m">{w.week_key}{w.on_time ? "" : " · late"}</td><td className="vt-fc-m">{w.base_date}</td><td className="vt-fc-m">{w.n}</td>
                        <td className="vt-fc-m">{w.resolved}</td><td className="vt-fc-m">{w.resolved ? `${w.hits} (${pct(w.hit_rate)})` : "—"}</td>
                        <td className="vt-fc-m">{w.brier ?? "—"}</td><td className="vt-fc-m">{pct(w.in_range_rate)}</td><td className="vt-fc-m">{w.model_version}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

          <footer className="vt-fc-foot">
            <p>{rep.method}</p>
            {rep.caveats.map((c) => <p key={c}>{c}</p>)}
          </footer>
        </>
      ) : null}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="vt-fc-grid" aria-busy="true" aria-label="Loading forecast">
      {Array.from({ length: 6 }, (_, k) => <div key={k} className="vt-fc-card vt-fc-skel" />)}
    </div>
  );
}

function CallCard({ c }: { c: ForecastCall }) {
  const up = c.direction === "up";
  const b = bandPct(c);
  const st = callState(c);
  return (
    <article className="vt-fc-card" data-dir={c.direction} aria-label={`${c.symbol}: ${c.direction}, ${pct(c.p_call)}`}>
      <div className="vt-fc-cardtop">
        <span className="vt-fc-sym">{c.symbol}</span>
        <span className="vt-fc-chip" data-small="true">{GROUP_LABEL[c.group] ?? c.group}</span>
        <span className="vt-fc-state-badge" data-state={st}>{st === "open" ? `open → ${c.horizon_end}` : st === "no-data" ? "no price data" : st}</span>
      </div>
      <div className="vt-fc-call">
        <span className="vt-fc-dir" data-dir={c.direction}>{up ? <ArrowUpRight size={18} aria-hidden="true" /> : <ArrowDownRight size={18} aria-hidden="true" />}{up ? "Up" : "Down"}</span>
        <span className="vt-fc-prob vt-fc-m" title={`P(up) = ${pct(c.p_up, 1)}`}>{pct(c.p_call, 1)}</span>
        <span className="vt-fc-meta">P(up) {pct(c.p_up, 1)}</span>
      </div>
      <div className="vt-fc-range" title={`${pct(c.band_coverage, 0)} band, ${VOL_LABEL[c.vol_source] ?? c.vol_source}`}>
        <span className="vt-fc-m">{usd(c.range_lo)}</span>
        <span className="vt-fc-bar" aria-hidden="true"><span /></span>
        <span className="vt-fc-m">{usd(c.range_hi)}</span>
      </div>
      <div className="vt-fc-meta vt-fc-m">spot {usd(c.spot)} · {b.down.toFixed(1)}% / +{b.up.toFixed(1)}% · {VOL_LABEL[c.vol_source] ?? c.vol_source}</div>
      {c.score?.status === "resolved" ? (
        <div className="vt-fc-meta vt-fc-m">closed {usd(c.score.realized_close)} on {c.score.resolved_date} ({(c.score.realized_ret_pct ?? 0) >= 0 ? "+" : ""}{(c.score.realized_ret_pct ?? 0).toFixed(2)}%) · {c.score.in_range ? "inside" : "outside"} range · Brier {c.score.brier?.toFixed(3)}</div>
      ) : null}
      <div className="vt-fc-meta">base {c.base_date} · {c.model_version}</div>
    </article>
  );
}

function AccuracyTiles({ a }: { a: Accuracy }) {
  const thin = a.thin ? <span className="vt-fc-warn" title="Too few resolved calls for a stable rate"> · thin</span> : null;
  return (
    <div className="vt-fc-tiles">
      <div className="vt-fc-tile"><span className="vt-fc-tl">Direction hits</span><span className="vt-fc-tv vt-fc-m">{pct(a.hit_rate, 1)}</span><span className="vt-fc-meta">{a.hits} of {a.n}{thin}</span></div>
      <div className="vt-fc-tile"><span className="vt-fc-tl">Brier score</span><span className="vt-fc-tv vt-fc-m">{a.brier ?? "—"}</span><span className="vt-fc-meta">lower is better · coin-flip 0.25 · {skillText(a.brier_skill_vs_coinflip)}</span></div>
      <div className="vt-fc-tile"><span className="vt-fc-tl">Inside the range</span><span className="vt-fc-tv vt-fc-m">{pct(a.in_range_rate, 1)}</span><span className="vt-fc-meta">target {pct(a.in_range_target, 0)} · {a.in_range} of {a.n}{thin}</span></div>
      <div className="vt-fc-tile"><span className="vt-fc-tl">Rose over the week</span><span className="vt-fc-tv vt-fc-m">{pct(a.up_rate, 1)}</span><span className="vt-fc-meta">base rate the calls are judged against</span></div>
    </div>
  );
}

function CalibrationChart({ live, holdout }: { live: CalBin[]; holdout: CalBin[] }) {
  // phone: a narrower viewBox so the axis text stays legible at 390px (the svg scales to its column)
  const narrow = useNarrow();
  const W = narrow ? 340 : 560, H = narrow ? 260 : 300;
  const g = useMemo(() => calChart(W, H, [live, holdout]), [live, holdout]);
  const maxN = Math.max(1, ...live.map((b) => b.n), ...holdout.map((b) => b.n));
  const [hover, setHover] = useState<{ s: "live" | "holdout"; b: CalBin } | null>(null);
  const series: Array<{ key: "live" | "holdout"; label: string; bins: CalBin[]; color: string }> = [
    { key: "holdout", label: "Model holdout (fit time)", bins: holdout, color: "var(--accent-purple)" },
    { key: "live", label: "Live calls", bins: live, color: "var(--accent)" },
  ];
  const liveN = live.reduce((s, b) => s + b.n, 0);
  return (
    <div className="vt-fc-chart">
      <div className="vt-fc-legend">
        {series.map((s) => <span key={s.key}><i style={{ background: s.color }} aria-hidden="true" />{s.label} <span className="vt-fc-m">n={s.bins.reduce((t, b) => t + b.n, 0)}</span></span>)}
        <span><i data-diag="true" aria-hidden="true" />perfect calibration</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Calibration: predicted probability of a rise against the realized frequency, per probability bin" onMouseLeave={() => setHover(null)}>
        {g.yTicks.map((t) => <g key={`y${t}`}><line x1={g.pad.l} x2={W - g.pad.r} y1={g.y(t)} y2={g.y(t)} className="vt-fc-grid-l" /><text x={g.pad.l - 6} y={g.y(t) + (t === 0 ? 0 : 4)} textAnchor="end" className="vt-fc-axis">{Math.round(t * 100)}%</text></g>)}
        {g.xTicks.map((t) => <text key={`x${t}`} x={g.x(t)} y={H - g.pad.b + 16} textAnchor="middle" className="vt-fc-axis">{Math.round(t * 100)}%</text>)}
        <text x={(g.pad.l + W - g.pad.r) / 2} y={H - 2} textAnchor="middle" className="vt-fc-axis">predicted P(up)</text>
        <line x1={g.diagonal.x1} y1={g.diagonal.y1} x2={g.diagonal.x2} y2={g.diagonal.y2} className="vt-fc-diag" />
        {series.map((s) => s.bins.filter((b) => b.n > 0 && b.mean_p !== null && b.freq !== null).map((b) => {
          const cx = g.x(b.mean_p as number), cy = g.y(b.freq as number), r = dotRadius(b.n, maxN);
          const on = hover?.s === s.key && hover.b === b;
          return (
            <g key={`${s.key}${b.lo}`} onMouseEnter={() => setHover({ s: s.key, b })} onClick={() => setHover({ s: s.key, b })}>
              <circle cx={cx} cy={cy} r={Math.max(r + 8, 14)} fill="transparent" />
              <circle cx={cx} cy={cy} r={on ? r + 2 : r} fill={s.color} fillOpacity={s.key === "holdout" ? 0.55 : 0.9} stroke="var(--bg-primary)" strokeWidth={2} />
              <text x={cx + r + 4} y={cy - r - 2} className="vt-fc-axis">{b.n}</text>
              <title>{`${s.label}: P(up) ${pct(b.lo)}–${pct(b.hi)} · ${b.n} calls · predicted ${pct(b.mean_p, 1)} · rose ${pct(b.freq, 1)}`}</title>
            </g>
          );
        }))}
      </svg>
      <div className="vt-fc-readout" aria-live="polite">
        {hover ? `${hover.s === "live" ? "Live calls" : "Model holdout"} · P(up) ${pct(hover.b.lo)}–${pct(hover.b.hi)}: ${hover.b.n} calls, predicted ${pct(hover.b.mean_p, 1)}, rose ${pct(hover.b.freq, 1)}`
          : liveN ? "Hover or tap a dot. Dots on the dashed line are calibrated; above it the model was too pessimistic, below it too optimistic." : "No live call has resolved yet — the purple dots are the model's own holdout test from the week it was fit."}
      </div>
      <details className="vt-fc-details">
        <summary>Table view</summary>
        <div className="vt-fc-tablewrap">
          <table className="vt-fc-table">
            <thead><tr><th>P(up) bin</th><th>Live n</th><th>Live predicted</th><th>Live rose</th><th>Holdout n</th><th>Holdout predicted</th><th>Holdout rose</th></tr></thead>
            <tbody>
              {live.map((b, k) => (
                <tr key={b.lo}><td className="vt-fc-m">{pct(b.lo)}–{pct(b.hi)}</td><td className="vt-fc-m">{b.n}</td><td className="vt-fc-m">{pct(b.mean_p, 1)}</td><td className="vt-fc-m">{pct(b.freq, 1)}</td>
                  <td className="vt-fc-m">{holdout[k]?.n ?? 0}</td><td className="vt-fc-m">{pct(holdout[k]?.mean_p, 1)}</td><td className="vt-fc-m">{pct(holdout[k]?.freq, 1)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function useNarrow(): boolean {
  const read = () => typeof window !== "undefined" && window.innerWidth < 640;
  const [narrow, setNarrow] = useState(read);
  useEffect(() => {
    const on = () => setNarrow(read());
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return narrow;
}

function ModelCard({ latest }: { latest: NonNullable<ForecastReport["latest"]> }) {
  const v = latest.model.validation;
  return (
    <div className="vt-fc-model">
      <div className="vt-fc-tl">Model check at fit time ({latest.model_version}, fit {latest.made_at.slice(0, 10)})</div>
      <div className="vt-fc-meta vt-fc-m">
        holdout {v.n_test} samples from {v.split_date} (trained on {v.n_train} earlier) · Brier {v.brier} vs {v.brier_climatology} always-base-rate · skill {skillText(v.brier_skill)} · hits {pct(v.hit_rate, 1)} · rose {pct(v.test_base_rate, 1)}
      </div>
      <div className="vt-fc-meta">fit on {latest.model.n_samples} weekly samples, {latest.model.first_date} → {latest.model.last_date}, {latest.train_tickers.length} names; features: {latest.model.features.join(", ")}</div>
    </div>
  );
}
