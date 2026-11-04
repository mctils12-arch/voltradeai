// Forecast screen (SHADOW). Two horizons for the same universe — SPY, every
// open position, the last scan's top candidates:
//   * DAILY (Part C, 2026-11-04; the default view): the next session's close —
//     direction = the base rate (no feature beat it in the pre-registered
//     study), a 68% range from fc-range-har-v1 (HAR-RV + implied vol), a live
//     band re-centred hourly through the session (display only), and every
//     close's score: Brier vs the base rate AND vs a coin flip, hit rate,
//     inside-the-range rate, interval-score skill vs a constant band.
//   * WEEKLY (2026-10-27, human's Phase 7): the next 5 trading sessions,
//     fc-logit-v1 — unchanged.
// Every number carries its batch time, base close and model version; the
// model is labeled experimental and nothing trades on it.
// Data: GET /api/forecast (forecast.py report(); owner only).
// Layout (2026-10-29, human-asked): the AI Engine's cockpit — a KPI strip,
// then the calls, track record and calibration as panels in one full-width
// grid (pages/cockpit.tsx); phone and tablet stack the same panels.
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowDownRight, ArrowUpRight, FlaskConical, LogIn, RefreshCw, Target } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { authErrorKind, notifySessionExpired, SESSION_EXPIRED_NOTICE } from "@/lib/authExpiry";
import {
  type ForecastReport, type ForecastCall, type CalBin, type Accuracy, type DailyCall, type DailyAccuracy, type DailyReport, type Horizon,
  GROUP_LABEL, VOL_LABEL, RANGE_SOURCE_LABEL, pct, usd, bandPct, ageText, callState, skillText, calChart, dotRadius, forecastKpis,
  dailyKpis, signedPct, parseHorizon, liveText, dailyRangeNote, dailyLiveNote, timingNote, shortDate, fixed,
} from "@/lib/forecast";
import { Bar, KpiStrip, KpiTile, Panel } from "./cockpit";

const POLL_MS = 5 * 60_000;
type Filter = "all" | "index" | "position" | "candidate";
const FILTER_LABEL: Record<Filter, string> = { all: "All", index: "Index", position: "Positions", candidate: "Candidates" };
/** SPY and the positions first, then the scan's candidates (stable within a group). */
const GROUP_ORDER: Record<string, number> = { index: 0, position: 1, candidate: 2 };
const HORIZON_KEY = "vt-fc-horizon";   // a per-viewer convenience only (never state the page depends on)

function readHorizon(): Horizon {
  try {
    return parseHorizon(window.localStorage.getItem(HORIZON_KEY));
  } catch {
    return "daily";   // storage blocked (private mode, previews): the default view
  }
}

function storeHorizon(h: Horizon): void {
  try {
    window.localStorage.setItem(HORIZON_KEY, h);
  } catch (e) {
    console.debug("[forecast] horizon not remembered:", e instanceof Error ? e.message : e);
  }
}

async function fetchForecast(): Promise<ForecastReport> {
  const r = await apiRequest("GET", "/api/forecast", undefined, { timeoutMs: 30_000 });
  return (await r.json()) as ForecastReport;
}

export default function ForecastPage() {
  // A 401/403 is an answer about the viewer, not an outage: stop polling on it
  // (field bug 2026-10-28 — an expired session read as "service unavailable").
  const q = useQuery({
    queryKey: ["/api/forecast"], queryFn: fetchForecast, staleTime: 60_000,
    refetchInterval: (query) => (authErrorKind(query.state.error) ? false : POLL_MS),
  });
  const authErr = q.isError && !q.data ? authErrorKind(q.error) : null;
  const [horizon, setHorizonState] = useState<Horizon>(readHorizon);
  const setHorizon = (h: Horizon) => { setHorizonState(h); storeHorizon(h); };
  const now = Date.now();
  const rep = q.data;
  const daily = horizon === "daily";
  const dl = rep?.daily?.latest ?? null;
  const wl = rep?.latest ?? null;

  return (
    <div className="vt-ck vt-fc" data-vt-forecast data-horizon={horizon}>
      <header className="vt-ck-head">
        <h1><Target size={16} aria-hidden="true" /> Forecast</h1>
        <div className="vt-fc-horizon" role="group" aria-label="Forecast horizon">
          <button type="button" className="vt-fc-btn" aria-pressed={daily} data-vt-fc-horizon="daily" onClick={() => setHorizon("daily")}>Daily · next close</button>
          <button type="button" className="vt-fc-btn" aria-pressed={!daily} data-vt-fc-horizon="weekly" onClick={() => setHorizon("weekly")}>Weekly · 5 sessions</button>
        </div>
        <span className="vt-fc-chip" data-tone="orange" title="Simple statistical models under evaluation — read their skill numbers before trusting a call"><FlaskConical size={12} aria-hidden="true" /> Experimental model</span>
        <span className="vt-fc-chip" title="Display and tracking only: no order, size or threshold reads these calls">Shadow · nothing trades on these</span>
        <div className="vt-ck-fresh" aria-live="polite">
          {daily && dl ? (
            <>
              <span>made <b>{ageText(dl.made_at, now)}</b> ago</span>
              <span>base close <b>{dl.base_date}</b></span>
              <span>target close <b>{dl.session}</b></span>
              <span>model <b>{dl.model_version}</b> · range <b>{dl.range_model_version}</b></span>
              {!dl.on_time ? <span className="vt-fc-warn">late batch — made after the session opened (rest-of-day calls)</span> : null}
            </>
          ) : null}
          {!daily && wl ? (
            <>
              <span>made <b>{ageText(wl.made_at, now)}</b> ago</span>
              <span>base close <b>{wl.base_date}</b></span>
              <span>horizon → <b>{wl.horizon_end}</b></span>
              <span>model <b>{wl.model_version}</b></span>
              {!wl.on_time ? <span className="vt-fc-warn">late batch — made after the first horizon session opened</span> : null}
            </>
          ) : null}
          {rep ? <span>page data {ageText(rep.generated_at, now)} old</span> : null}
          <button type="button" className="vt-fc-btn" onClick={() => { void q.refetch(); }} disabled={q.isFetching} aria-label="Refresh forecast">
            <RefreshCw size={13} aria-hidden="true" className={q.isFetching ? "vt-fc-spin" : undefined} /> {q.isFetching ? "refreshing" : "refresh"}
          </button>
        </div>
        <p className="vt-ck-sub">
          {daily
            ? "Direction, probability and a 68% price range for the next session's close — SPY, every open position and the last scan's top candidates. The direction is the base rate (no tested feature beat it), shown with its skill against a coin flip; the range is the part with measured skill. During the session the band re-centres on the live price each hour."
            : `Direction, probability and a 68% price range for the next ${rep?.horizon_days ?? 5} trading sessions — SPY, every open position and the last scan's top candidates. A call is always made (up when P(up) ≥ 50%), and its probability is the model's, never rounded up.`}
        </p>
      </header>

      {q.isLoading ? <Skeleton /> : authErr === "unauthorized" ? (
        <div className="vt-fc-state" data-vt-fc-auth="signin" role="status">
          <b>{SESSION_EXPIRED_NOTICE}.</b> The forecast needs a signed-in owner session; sessions last 24 hours.
          <div><button type="button" className="vt-fc-btn" style={{ marginTop: 10 }} onClick={() => notifySessionExpired("/api/forecast")}>
            <LogIn size={13} aria-hidden="true" /> Sign in
          </button></div>
        </div>
      ) : authErr === "forbidden" ? (
        <div className="vt-fc-state" data-vt-fc-auth="owner" role="status">
          <b>Owner only.</b> The forecast names the account's open positions, so it is restricted to the account owner.
        </div>
      ) : q.isError && !rep ? (
        <div className="vt-fc-state" data-tone="red" role="alert">
          <b>Forecast service unavailable.</b> {String(q.error instanceof Error ? q.error.message : q.error).slice(0, 160)} — the page retries every 5 minutes.
        </div>
      ) : rep ? (daily ? <DailyView rep={rep} now={now} /> : <WeeklyView rep={rep} now={now} />) : null}
    </div>
  );
}

function WeeklyView({ rep, now }: { rep: ForecastReport; now: number }) {
  const latest = rep.latest;
  const [filter, setFilter] = useState<Filter>("all");
  const calls = useMemo(() => (latest?.forecasts ?? []).filter((c) => filter === "all" || c.group === filter)
    .sort((a, b) => (GROUP_ORDER[a.group] ?? 9) - (GROUP_ORDER[b.group] ?? 9)), [latest, filter]);
  const counts = useMemo(() => {
    const out: Record<Filter, number> = { all: 0, index: 0, position: 0, candidate: 0 };
    for (const c of latest?.forecasts ?? []) { out.all += 1; out[c.group as Filter] = (out[c.group as Filter] ?? 0) + 1; }
    return out;
  }, [latest]);
  const kpis = useMemo(() => forecastKpis(rep), [rep]);
  if (!latest) {
    return (
      <div className="vt-fc-state">
        <b>No weekly batch yet.</b> The first batch is made at the first check after a session closes (normally Friday after 16:15 ET) and lands here on its own.
        {rep.open ? ` ${rep.open} open calls are waiting for their horizon.` : ""}
      </div>
    );
  }
  return (
    <>
      <KpiStrip n={5} label="Weekly forecast summary">
        {kpis.map((k) => <KpiTile key={k.label} label={k.label} value={k.value} line={k.line} tone={k.tone} />)}
      </KpiStrip>
      <div className="vt-ck-grid">
        <Bar id="vt-fc-calls" title="This week's calls">
          <div className="vt-fc-filters" role="group" aria-label="Filter calls">
            {(["all", "index", "position", "candidate"] as Filter[]).map((f) => (
              <button key={f} type="button" className="vt-fc-btn" aria-pressed={filter === f} onClick={() => setFilter(f)} disabled={f !== "all" && !counts[f]}>
                {FILTER_LABEL[f]} <span className="vt-fc-m">{counts[f] ?? 0}</span>
              </button>
            ))}
          </div>
          {!latest.candidates_available ? <p className="vt-fc-note">No scan candidates were available when this batch was made (the scan had not run since the last restart) — positions and SPY only.</p> : null}
          {latest.skipped.length ? <p className="vt-fc-note">Not forecast: {latest.skipped.map((s) => `${s.symbol} (${s.reason.replace(/_/g, " ")})`).join(", ")}.</p> : null}
          <p className="vt-fc-note">
            Range = 68% band of a zero-drift lognormal; {latest.iv_names} of {latest.n_forecasts} use chain IV30{latest.iv_day ? ` (options archive, ${latest.iv_day})` : ""}, the rest 20-day realized vol.
            {latest.candidates_scan_at ? ` Candidates from the scan at ${new Date(latest.candidates_scan_at).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })} ET.` : ""}
          </p>
        </Bar>
        {calls.map((c) => <CallCard key={c.id} c={c} />)}

        <Panel id="vt-fc-acc" title="Track record" meta={`${rep.accuracy.n} resolved · ${rep.open} open · as of ${ageText(rep.generated_at, now)} ago`}>
          <AccuracyTiles a={rep.accuracy} />
          {rep.accuracy.n > 0 && rep.accuracy.on_time.n !== rep.accuracy.n ? <p className="vt-fc-note">On-time batches only: {rep.accuracy.on_time.n} calls, {pct(rep.accuracy.on_time.hit_rate)} direction hits.</p> : null}
        </Panel>

        <Panel id="vt-fc-cal" title="Calibration" meta="predicted P(up) vs how often the price actually rose">
          <CalibrationChart live={rep.calibration.bins} holdout={latest.model.validation.calibration} />
          <ModelCard latest={latest} />
        </Panel>

        {rep.weeks.length ? (
          <Panel id="vt-fc-weeks" title="By week" meta="newest first" span={2}>
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
          </Panel>
        ) : null}
      </div>

      <footer className="vt-ck-foot">
        <p>{rep.method}</p>
        {rep.caveats.map((c) => <p key={c}>{c}</p>)}
      </footer>
    </>
  );
}

// ── the daily view (Part C, 2026-11-04) ─────────────────────────────────────
function DailyView({ rep, now }: { rep: ForecastReport; now: number }) {
  const d = rep.daily ?? null;
  const latest = d?.latest ?? null;
  const [filter, setFilter] = useState<Filter>("all");
  const calls = useMemo(() => (latest?.forecasts ?? []).filter((c) => filter === "all" || c.group === filter)
    .sort((a, b) => (GROUP_ORDER[a.group] ?? 9) - (GROUP_ORDER[b.group] ?? 9)), [latest, filter]);
  const counts = useMemo(() => {
    const out: Record<Filter, number> = { all: 0, index: 0, position: 0, candidate: 0 };
    for (const c of latest?.forecasts ?? []) { out.all += 1; out[c.group as Filter] = (out[c.group as Filter] ?? 0) + 1; }
    return out;
  }, [latest]);
  const kpis = useMemo(() => dailyKpis(d), [d]);
  if (!d) {
    return <div className="vt-fc-state" data-vt-fc-daily="absent"><b>No daily forecast on this server yet.</b> The daily horizon arrives with the next deploy; the weekly view is unchanged.</div>;
  }
  return (
    <>
      <KpiStrip n={5} label="Daily forecast summary">
        {kpis.map((k) => <KpiTile key={k.label} label={k.label} value={k.value} line={k.line} tone={k.tone} />)}
      </KpiStrip>
      {!latest ? (
        <div className="vt-fc-state">
          <b>No daily calls yet.</b> The first batch is made after a session closes — once the options archive's 16:30–18:30 ET capture is done and the day's VIX close is in (Cboe publishes it in the evening) — and lands here on its own. It is scored after the next close.
          {d.open ? ` ${d.open} open calls are waiting for their close.` : ""}
        </div>
      ) : (
        <div className="vt-ck-grid">
          <Bar id="vt-fc-dcalls" title={`Calls for the ${shortDate(latest.session)} close`}>
            <div className="vt-fc-filters" role="group" aria-label="Filter daily calls">
              {(["all", "index", "position", "candidate"] as Filter[]).map((f) => (
                <button key={f} type="button" className="vt-fc-btn" aria-pressed={filter === f} onClick={() => setFilter(f)} disabled={f !== "all" && !counts[f]}>
                  {FILTER_LABEL[f]} <span className="vt-fc-m">{counts[f] ?? 0}</span>
                </button>
              ))}
            </div>
            <p className="vt-fc-note">{dailyRangeNote(latest)}</p>
            <p className="vt-fc-note">{dailyLiveNote(d, now)}</p>
            {latest.skipped?.length ? <p className="vt-fc-note">Not forecast: {latest.skipped.map((s) => `${s.symbol} (${s.reason.replace(/_/g, " ")})`).join(", ")}.</p> : null}
          </Bar>
          {calls.map((c) => <DailyCallCard key={c.id} c={c} now={now} />)}

          <Panel id="vt-fc-dacc" title="Daily track record" meta={`${d.accuracy.n} resolved · ${d.open} open`}>
            <DailyAccuracyTiles a={d.accuracy} />
            {d.accuracy.n > 0 ? <p className="vt-fc-note">{timingNote(d.accuracy.by_timing)}</p> : null}
          </Panel>

          <Panel id="vt-fc-dcal" title="Daily calibration" meta="predicted P(up) vs how often the close actually rose">
            <CalibrationChart live={d.calibration.bins} holdout={[]} emptyText="No daily call has resolved yet. Every call sits at the base rate, so the resolved ones will fall in one probability bin — the dot shows how often the close actually rose." />
          </Panel>

          <Panel id="vt-fc-dstudy" title="Research check" meta={`out of sample, ${d.study.range.holdout}`}>
            <StudyCard d={d} />
          </Panel>

          {d.sessions.length ? (
            <Panel id="vt-fc-dsess" title="By session" meta="newest first" span="full">
              <div className="vt-fc-tablewrap">
                <table className="vt-fc-table">
                  <thead><tr><th>Close</th><th>Calls</th><th>Resolved</th><th>Hits</th><th>Brier</th><th>In range</th><th>Interval score</th><th>Range skill</th></tr></thead>
                  <tbody>
                    {d.sessions.map((s) => (
                      <tr key={s.session}>
                        <td className="vt-fc-m">{s.session}</td><td className="vt-fc-m">{s.n}</td><td className="vt-fc-m">{s.resolved}</td>
                        <td className="vt-fc-m">{s.resolved ? `${s.hits} (${pct(s.hit_rate)})` : "—"}</td><td className="vt-fc-m">{s.brier ?? "—"}</td>
                        <td className="vt-fc-m">{pct(s.in_range_rate)}</td><td className="vt-fc-m">{s.interval_score_pct === null ? "—" : `${fixed(s.interval_score_pct)}%`}</td>
                        <td className="vt-fc-m">{signedPct(s.range_skill_vs_constant)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          ) : null}
        </div>
      )}
      <footer className="vt-ck-foot">
        <p>{d.method}</p>
        {d.caveats.map((c) => <p key={c}>{c}</p>)}
      </footer>
    </>
  );
}

function DailyCallCard({ c, now }: { c: DailyCall; now: number }) {
  const up = c.direction === "up";
  const b = bandPct(c);
  const st = callState(c);
  const lv = c.live;
  return (
    <article className="vt-ck-panel vt-fc-call" data-dir={c.direction} data-timing={c.timing} aria-label={`${c.symbol}: ${c.direction}, ${pct(c.p_call)} for the ${c.session} close`}>
      <div className="vt-ck-ph">
        <span className="vt-fc-sym">{c.symbol}</span>
        <span className="vt-fc-chip" data-small="true">{GROUP_LABEL[c.group] ?? c.group}</span>
        {c.timing === "intraday" ? <span className="vt-fc-chip" data-small="true" data-tone="orange" title="Made during the session at the live price — a rest-of-day call, scored in its own bucket">intraday</span> : null}
        <span className="vt-fc-state-badge" data-state={st}>{st === "open" ? `open → ${shortDate(c.session)} close` : st === "no-data" ? "no price data" : st}</span>
      </div>
      <div className="vt-ck-pb">
        <div className="vt-fc-call">
          <span className="vt-fc-dir" data-dir={c.direction}>{up ? <ArrowUpRight size={18} aria-hidden="true" /> : <ArrowDownRight size={18} aria-hidden="true" />}{up ? "Up" : "Down"}</span>
          <span className="vt-fc-prob vt-fc-m" title={`P(up) = ${pct(c.p_up, 1)} — the base rate`}>{pct(c.p_call, 1)}</span>
          <span className="vt-fc-meta">P(up) {pct(c.p_up, 1)} · base rate, no edge</span>
        </div>
        <div className="vt-fc-range" title={`${pct(c.band_coverage, 0)} band for the ${c.session} close · ${RANGE_SOURCE_LABEL[c.range_source] ?? c.range_source}`}>
          <span className="vt-fc-m">{usd(c.range_lo)}</span>
          <span className="vt-fc-bar" aria-hidden="true"><span /></span>
          <span className="vt-fc-m">{usd(c.range_hi)}</span>
        </div>
        <div className="vt-fc-meta vt-fc-m">{c.timing === "pre_open" ? `${c.base_date} close` : "price when made"} {usd(c.spot)} · {b.down.toFixed(2)}% / +{b.up.toFixed(2)}% · {RANGE_SOURCE_LABEL[c.range_source] ?? c.range_source}</div>
        {lv && !c.score ? <div className="vt-fc-live vt-fc-m" title="Display only, never scored: the band re-centred on the latest price with the variance still to come">{liveText(c, lv, now)}</div> : null}
        {c.score?.status === "resolved" ? (
          <div className="vt-fc-meta vt-fc-m">closed {usd(c.score.realized_close)} ({(c.score.realized_ret_pct ?? 0) >= 0 ? "+" : ""}{(c.score.realized_ret_pct ?? 0).toFixed(2)}%) · {c.score.in_range ? "inside" : "outside"} range · Brier {c.score.brier?.toFixed(3)}</div>
        ) : null}
        <div className="vt-fc-meta">made {ageText(c.made_at, now)} ago · {c.model_version} · range {c.range_model_version}</div>
      </div>
    </article>
  );
}

function DailyAccuracyTiles({ a }: { a: DailyAccuracy }) {
  const thin = a.thin ? <span className="vt-fc-warn" title="Too few resolved calls for a stable rate"> · thin</span> : null;
  return (
    <div className="vt-fc-tiles">
      <div className="vt-fc-tile"><span className="vt-fc-tl">Direction hits</span><span className="vt-fc-tv vt-fc-m">{pct(a.hit_rate, 1)}</span><span className="vt-fc-meta">{a.hits} of {a.n} · rose {pct(a.up_rate, 1)}{thin}</span></div>
      <div className="vt-fc-tile"><span className="vt-fc-tl">Brier skill</span><span className="vt-fc-tv vt-fc-m">{signedPct(a.bss_vs_coinflip)}</span><span className="vt-fc-meta">vs coin flip 0.25 · {skillText(a.bss_vs_base_rate)} vs the base rate · Brier {a.brier ?? "—"}{thin}</span></div>
      <div className="vt-fc-tile"><span className="vt-fc-tl">Inside the range</span><span className="vt-fc-tv vt-fc-m">{pct(a.in_range_rate, 1)}</span><span className="vt-fc-meta">target {pct(a.in_range_target, 0)} · {a.in_range} of {a.n}{thin}</span></div>
      <div className="vt-fc-tile"><span className="vt-fc-tl">Range skill</span><span className="vt-fc-tv vt-fc-m">{signedPct(a.range_skill_vs_constant)}</span><span className="vt-fc-meta">interval score vs a constant-width band · {a.range_skill_n} paired{thin}</span></div>
    </div>
  );
}

function StudyCard({ d }: { d: DailyReport }) {
  const r = d.study.range, x = d.study.direction;
  return (
    <div className="vt-fc-model" data-vt-fc-study>
      <div className="vt-fc-tl">{d.range_model_version} range · {d.direction_model} direction</div>
      <div className="vt-fc-meta vt-fc-m">
        range — SPY: {pct(r.spy_coverage, 1)} inside the 68% band, interval-score skill {signedPct(r.spy_skill_vs_constant)} vs a constant band and {signedPct(r.spy_skill_vs_current)} vs the weekly page's IV/20-day-vol method · names with IV: {pct(r.iv_names_coverage, 1)} inside, {signedPct(r.iv_names_skill_vs_constant)} · names without IV: {pct(r.names_coverage, 1)} inside, {signedPct(r.names_skill_vs_constant)}
      </div>
      <div className="vt-fc-meta vt-fc-m">
        direction: {x.passed} of {x.tests} pre-registered features beat the base rate · SPY base rate {pct(x.spy_base_rate, 1)} up, Brier {x.spy_brier_base_rate.toFixed(4)} vs coin flip 0.25 ({signedPct(x.spy_skill_vs_coinflip)}) — the drift, not a signal
      </div>
      <div className="vt-fc-meta">{r.doc} · {x.doc}</div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="vt-ck-grid" aria-busy="true" aria-label="Loading forecast">
      {Array.from({ length: 6 }, (_, k) => <div key={k} className="vt-ck-panel vt-fc-skel" />)}
    </div>
  );
}

function CallCard({ c }: { c: ForecastCall }) {
  const up = c.direction === "up";
  const b = bandPct(c);
  const st = callState(c);
  return (
    <article className="vt-ck-panel vt-fc-call" data-dir={c.direction} aria-label={`${c.symbol}: ${c.direction}, ${pct(c.p_call)}`}>
      <div className="vt-ck-ph">
        <span className="vt-fc-sym">{c.symbol}</span>
        <span className="vt-fc-chip" data-small="true">{GROUP_LABEL[c.group] ?? c.group}</span>
        <span className="vt-fc-state-badge" data-state={st}>{st === "open" ? `open → ${c.horizon_end}` : st === "no-data" ? "no price data" : st}</span>
      </div>
      <div className="vt-ck-pb">
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
      </div>
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

function CalibrationChart({ live, holdout, emptyText }: { live: CalBin[]; holdout: CalBin[]; emptyText?: string }) {
  // phone: a narrower viewBox so the axis text stays legible at 390px (the svg scales to its column)
  const narrow = useNarrow();
  const W = narrow ? 340 : 560, H = narrow ? 260 : 300;
  const g = useMemo(() => calChart(W, H, [live, holdout]), [live, holdout]);
  const maxN = Math.max(1, ...live.map((b) => b.n), ...holdout.map((b) => b.n));
  const [hover, setHover] = useState<{ s: "live" | "holdout"; b: CalBin } | null>(null);
  const series: Array<{ key: "live" | "holdout"; label: string; bins: CalBin[]; color: string }> = [
    { key: "holdout" as const, label: "Model holdout (fit time)", bins: holdout, color: "var(--accent-purple)" },
    { key: "live" as const, label: "Live calls", bins: live, color: "var(--accent)" },
  ].filter((x) => x.key === "live" || x.bins.length > 0);
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
          : liveN ? "Hover or tap a dot. Dots on the dashed line are calibrated; above it the model was too pessimistic, below it too optimistic." : (emptyText ?? "No live call has resolved yet — the purple dots are the model's own holdout test from the week it was fit.")}
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
