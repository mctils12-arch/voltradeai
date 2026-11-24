// PipelineHealthDashboardView — MAP V2 ROADMAP R6(c) PIPELINE-HEALTH
// dashboard (#/data/pipeline-health), closing the R6 dashboard trio
// (R6(a) signal strength + R6(b) data quality shipped 2026-07-30). Unlike
// those two pure joins, this one reads an actual time series: /api/health
// checks history, provider backoff state (scanner), and compliance status
// over trailing 24h/7d windows. Mobile-first (390px primary). DESIGN.md-
// governed: reuses the .vt-filings-* / .vt-quality-* shell + theme tokens.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Activity, RefreshCw } from "lucide-react";
import { ago, jobRows, jobsSummary, type JobTone, type PublicJobsView } from "@/lib/jobsView";
import { edgarPollRows, edgarPollSummary, type PublicEdgarPollView } from "@/lib/edgarPollView";

interface HealthSnapshot {
  t: number;
  status: "ok" | "degraded";
  database_ok: boolean;
  alpaca_ok: boolean;
  python_ok: boolean;
  scanner_ok: boolean;
  scanner_consecutive_failures: number;
  licensing_ok: boolean;
  bot_status: string;
  bot_liveness_dark: boolean;
}
interface HealthSummary {
  window_hours: number;
  sample_count: number;
  uptime_pct: number | null;
  current: HealthSnapshot | null;
  degraded_counts: {
    database: number; alpaca: number; python: number;
    scanner: number; licensing: number; bot_liveness_dark: number;
  };
  timeline: Array<{ t: number; status: "ok" | "degraded" }>;
}
interface PipelineHealthPayload {
  generated_at: string;
  warming_up: boolean;
  window_24h: HealthSummary;
  window_7d: HealthSummary;
  /** every scheduled job against its own schedule; null = the server could not read it; absent = an older server */
  jobs?: PublicJobsView | null;
  /** the three SEC filing pollers' latest poll; null = the server could not read it; absent = an older server */
  edgar?: PublicEdgarPollView | null;
}

const JOB_COLOR: Record<JobTone, string> = {
  bad: "var(--accent-red, #e5484d)", ok: "var(--accent-green)", idle: "var(--text-tertiary)",
};

const CHECK_LABEL: Record<keyof HealthSummary["degraded_counts"], string> = {
  database: "Database", alpaca: "Alpaca API", python: "Python engine",
  scanner: "Scanner (Tier-2 scans)", licensing: "Data-provider licensing",
  bot_liveness_dark: "Trading-loop liveness",
};

function timeAgo(iso: number): string {
  const s = Math.max(0, Math.round(Date.now() / 1000 - iso));
  if (s < 90) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m}m ago`;
  const h = Math.round(m / 60);
  return `${h}h ago`;
}

export default function PipelineHealthDashboardView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<PipelineHealthPayload | null>(null);
  const [error, setError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const load = async () => {
    setRefreshing(true);
    try {
      const r = await fetch("/api/data/pipeline-health-dashboard");
      // crash-guard (repair 2026-08-06): a JSON-bodied 5xx parsed fine and
      // crashed downstream on missing fields — route errors to the error state.
      if (!r.ok) throw new Error(String(r.status));
      setData(await r.json());
      setError(false);
    } catch {
      setError(true);
    } finally {
      setRefreshing(false);
    }
  };
  useEffect(() => { void load(); }, []);

  const worstChecks = useMemo(() => {
    if (!data) return [];
    const c = data.window_24h.degraded_counts;
    return (Object.keys(CHECK_LABEL) as Array<keyof typeof CHECK_LABEL>)
      .map((k) => ({ key: k, label: CHECK_LABEL[k], count: c[k] }))
      .filter((r) => r.count > 0)
      .sort((a, b) => b.count - a.count);
  }, [data]);

  // Scheduled jobs: the words are built once per payload, against the time it was loaded.
  const jobs = useMemo(() => {
    if (!data || data.jobs === undefined) return null;
    const now = Date.now();
    return { rows: jobRows(data.jobs, now), summary: jobsSummary(data.jobs), read: ago(data.jobs?.evaluated_at, now), failed: data.jobs === null };
  }, [data]);

  // SEC filing pollers: same rule, words built once per payload.
  const edgar = useMemo(() => {
    if (!data || data.edgar === undefined) return null;
    const now = Date.now();
    return { rows: edgarPollRows(data.edgar, now), summary: edgarPollSummary(data.edgar, now), read: ago(data.edgar?.read_at, now), failed: data.edgar === null };
  }, [data]);

  const uptimeColor = (pct: number | null) => {
    if (pct == null) return "var(--text-tertiary)";
    if (pct >= 99.5) return "var(--accent-green)";
    if (pct >= 97) return "var(--accent-orange)";
    return "var(--accent-red, #e5484d)";
  };

  return (
    <div className="vt-filings-page" role="region" aria-label="Pipeline health dashboard">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Activity size={16} />
        <div>
          <div className="vt-filings-title">Pipeline health</div>
          <div className="vt-filings-sub">
            {data ? "/api/health checks over time — uptime, provider backoff, and compliance status"
              : error ? "dashboard unavailable — retry below" : "loading…"}
          </div>
        </div>
        <button className="vt-icon-btn" aria-label="Refresh dashboard" onClick={() => void load()}
                style={{ marginLeft: "auto" }} disabled={refreshing}>
          <RefreshCw size={15} className={refreshing ? "vt-spin" : undefined} />
        </button>
      </div>

      <div className="vt-streams-body om-sb">
        {error && !data && (
          <div className="vt-filings-state">
            Could not load the dashboard. <button type="button" className="vt-graph-example" onClick={() => void load()}>Retry</button>
          </div>
        )}

        {data && data.warming_up && (
          <div className="vt-filings-state">
            No history recorded yet — this panel started capturing on this
            deploy; it fills in over the next few hours as /api/health is
            polled. Nothing is fabricated in the meantime.
          </div>
        )}

        {data && !data.warming_up && (
          <>
            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>Uptime</span>
                <span className="vt-quality-section-sub">
                  share of recorded checks reading fully healthy
                </span>
              </div>
              <div className="vt-quality-grid">
                {([["24h", data.window_24h], ["7d", data.window_7d]] as const).map(([label, w]) => (
                  <div key={label} className="vt-quality-tile">
                    <div className="vt-quality-tile-value" style={{ color: uptimeColor(w.uptime_pct) }}>
                      {w.uptime_pct == null ? "—" : `${w.uptime_pct}%`}
                    </div>
                    <div className="vt-quality-tile-label">last {label}</div>
                    <div className="vt-filings-sub" style={{ marginTop: 2 }}>
                      {w.sample_count.toLocaleString()} checks recorded
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>24h timeline</span>
                <span className="vt-quality-section-sub">
                  each bar is one recorded check, oldest to newest
                </span>
              </div>
              <div style={{ display: "flex", flexDirection: "row", gap: 1, alignItems: "flex-end", height: 28 }}>
                {data.window_24h.timeline.length === 0 && (
                  <div className="vt-filings-state">No checks recorded in this window yet.</div>
                )}
                {data.window_24h.timeline.map((pt, i) => (
                  <div key={i} title={new Date(pt.t * 1000).toLocaleString()}
                       style={{
                         flex: "1 1 0", minWidth: 2, height: "100%",
                         background: pt.status === "ok" ? "var(--accent-green)" : "var(--accent-red, #e5484d)",
                         borderRadius: 1,
                       }} />
                ))}
              </div>
              {data.window_24h.current && (
                <div className="vt-filings-sub vt-streams-foot">
                  Current: <strong style={{ color: data.window_24h.current.status === "ok" ? "var(--accent-green)" : "var(--accent-red, #e5484d)" }}>
                    {data.window_24h.current.status}
                  </strong>{" "}
                  ({timeAgo(data.window_24h.current.t)}) · bot {data.window_24h.current.bot_status}
                  {data.window_24h.current.scanner_consecutive_failures > 0 &&
                    ` · scanner backoff: ${data.window_24h.current.scanner_consecutive_failures} consecutive failures`}
                </div>
              )}
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>What's degrading (last 24h)</span>
                <span className="vt-quality-section-sub">
                  {worstChecks.length === 0 ? "nothing — every check has been clean" : "count of checks that failed, by subsystem"}
                </span>
              </div>
              {worstChecks.length > 0 && (
                <div className="vt-quality-kinds">
                  {worstChecks.map((r) => {
                    const maxCount = worstChecks[0].count || 1;
                    return (
                      <div key={r.key} className="vt-quality-kind-row">
                        <span className="vt-quality-kind-name">{r.label}</span>
                        <span className="vt-quality-kind-bar-track">
                          <span className="vt-quality-kind-bar-fill"
                                style={{ width: `${Math.max(2, (r.count / maxCount) * 100)}%`, background: "var(--accent-orange)" }} />
                        </span>
                        <span className="vt-quality-kind-bytes">{r.count}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          </>
        )}

        {jobs && (
          <section className="vt-quality-section" data-vt-jobs={jobs.failed ? "unread" : jobs.rows.length}>
            <div className="vt-quality-section-head">
              <span>Scheduled jobs</span>
              <span className="vt-quality-section-sub">
                {jobs.failed ? "not read" : `${jobs.summary}${jobs.read ? ` · read ${jobs.read}` : ""}`}
              </span>
            </div>
            {jobs.failed ? (
              <div className="vt-filings-state">
                The server could not read its job schedule on this request. Nothing is shown in its place.
              </div>
            ) : (
              <>
                <div className="vt-filings-sub">
                  Every job the server runs on a timer, judged against its own schedule: the trading
                  loop's cycles, the archive recordings and their completeness checks, and the weekly
                  signal tests.
                </div>
                <div className="vt-jobs-list">
                  {jobs.rows.map((r) => (
                    <div key={r.id} className="vt-job" data-tone={r.tone}>
                      <div className="vt-job-head">
                        <span className="vt-job-name">{r.label}</span>
                        <span className="vt-streams-chip" style={{ color: JOB_COLOR[r.tone], borderColor: JOB_COLOR[r.tone] }}>{r.state}</span>
                      </div>
                      {r.cause && <div className="vt-job-cause">{r.cause}</div>}
                      <div className="vt-job-meta">{r.last} Schedule: {r.schedule}.</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </section>
        )}

        {edgar && (
          <section className="vt-quality-section" data-vt-edgar={edgar.failed ? "unread" : edgar.rows.length}>
            <div className="vt-quality-section-head">
              <span>SEC filing pollers</span>
              <span className="vt-quality-section-sub">
                {edgar.failed ? "not read" : `${edgar.summary}${edgar.read ? ` · read ${edgar.read}` : ""}`}
              </span>
            </div>
            {edgar.failed ? (
              <div className="vt-filings-state">
                The server could not read its filing pollers on this request. Nothing is shown in its place.
              </div>
            ) : (
              <>
                <div className="vt-filings-sub">
                  The three feeds read from SEC EDGAR as filings are accepted, and what each could
                  and could not read on its latest poll. A filing a poll misses is fetched again
                  by the nightly completeness pass where the feed has one (Form 4 and 13F).
                </div>
                <div className="vt-jobs-list">
                  {edgar.rows.map((r) => (
                    <div key={r.id} className="vt-job" data-tone={r.tone}>
                      <div className="vt-job-head">
                        <span className="vt-job-name">{r.label}</span>
                        <span className="vt-streams-chip" style={{ color: JOB_COLOR[r.tone], borderColor: JOB_COLOR[r.tone] }}>{r.state}</span>
                      </div>
                      {r.cause && <div className="vt-job-cause">{r.cause}</div>}
                      <div className="vt-job-meta">{r.meta}</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
