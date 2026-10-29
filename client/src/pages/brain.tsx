// Brain page (2026-10-29, human-asked: "how would I know what is being used at
// the moment and what combinations could have produced better results ... the
// short term is one thing but the long term is different"). Every input, rule,
// model and sleeve the bot has, its status RIGHT NOW (live / shadow / off,
// derived from the real switches), the short-term evidence (live shadow
// outcomes) beside the long-term evidence (multi-year tests that exist), the
// agreement flag, and the best combinations with their multiple-testing
// discount. Every number carries its window, source and age; gaps say so.
// Data: GET /api/bot/brain (server/brain.ts; owner only).
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Brain, RefreshCw } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { ageText } from "@/lib/forecast";
import {
  type BrainReport, type BrainComponent, type ShortTerm, type LongTerm, type SelectionArm, type StatusFilter, type BrainStatus,
  BRAIN_KIND_LABEL, BRAIN_STATUS_LABEL, STATUS_HINT, AGREE_LABEL, SIGN_LABEL,
  signedPct, frac, ciText, icText, windowText, groupByKind, filterComponents, headlineArm,
} from "@/lib/brain";

const POLL_MS = 5 * 60_000;

async function fetchBrain(): Promise<BrainReport> {
  const r = await apiRequest("GET", "/api/bot/brain", undefined, { timeoutMs: 100_000 });
  return (await r.json()) as BrainReport;
}

export default function BrainPage() {
  const q = useQuery({ queryKey: ["/api/bot/brain"], queryFn: fetchBrain, refetchInterval: POLL_MS, staleTime: 60_000 });
  const [filter, setFilter] = useState<StatusFilter>("all");
  const rep = q.data;
  const now = Date.now();
  const shown = useMemo(() => groupByKind(filterComponents(rep?.components ?? [], filter)), [rep, filter]);
  const disagree = rep?.components.filter((c) => c.agreement.flag === "disagree").length ?? 0;

  return (
    <div className="vt-fc vt-br" data-vt-brain>
      <header className="vt-fc-head">
        <div className="vt-fc-title">
          <Brain size={18} aria-hidden="true" />
          <h1>Brain</h1>
          <span className="vt-fc-chip" title="Read-only: this page changes nothing">What the bot uses, and the evidence</span>
        </div>
        <p className="vt-br-explainer">
          {rep?.explainer ?? "Short-term = live shadow outcomes, long-term = multi-year tests; a component is promoted only when both agree."}
        </p>
        <p className="vt-fc-sub">{rep?.rule ?? "Live beats backtest when they disagree."}</p>
        <div className="vt-fc-fresh" aria-live="polite">
          {rep ? (
            <>
              <span>status computed <b>{ageText(rep.generated_at, now)}</b> ago</span>
              <span>evidence <b>{rep.evidence_generated_at ? `${ageText(rep.evidence_generated_at, now)} old` : "unavailable"}</b>{rep.evidence_via ? ` via ${rep.evidence_via}` : ""}</span>
              <span>shadow log <b>{windowText(rep.short_term.window)}</b>{rep.short_term.records !== null ? ` · ${rep.short_term.records.toLocaleString("en-US")} records` : ""}</span>
              <span>SPY benchmark <b>{rep.short_term.spy?.ok ? "on" : "unavailable"}</b></span>
              <span>switches from <b>{rep.switches.config_source === "python" ? "system_config + env" : "Node mirror + env"}</b></span>
            </>
          ) : null}
          <button type="button" className="vt-fc-btn" onClick={() => { void q.refetch(); }} disabled={q.isFetching} aria-label="Refresh the Brain page">
            <RefreshCw size={13} aria-hidden="true" className={q.isFetching ? "vt-fc-spin" : undefined} /> {q.isFetching ? "refreshing" : "refresh"}
          </button>
        </div>
        {rep?.evidence_error ? (
          <div className="vt-fc-state" data-tone="orange" role="status">
            <b>Evidence call failed:</b> {rep.evidence_error}. Status below is still live (env + config); {rep.evidence_generated_at ? "evidence shown is the last good copy." : "short-term numbers are missing until the next refresh."}
          </div>
        ) : null}
      </header>

      {q.isLoading ? <Skeleton /> : q.isError && !rep ? (
        <div className="vt-fc-state" data-tone="red" role="alert">
          <b>Brain service unavailable.</b> {String(q.error instanceof Error ? q.error.message : q.error).slice(0, 160)} — the page retries every 5 minutes.
        </div>
      ) : rep ? (
        <>
          <section className="vt-fc-sec" aria-labelledby="vt-br-now">
            <div className="vt-fc-sechead">
              <h2 id="vt-br-now">What is running right now</h2>
              <span className="vt-fc-meta">entries {rep.switches.entry_shadow.value ? `VOLTRADE_SHADOW_ENTRIES=${rep.switches.entry_shadow.value}` : "VOLTRADE_SHADOW_ENTRIES off"} ({rep.switches.entry_shadow.source})</span>
            </div>
            <div className="vt-br-counts" role="group" aria-label="Filter by status">
              {(["all", "live", "shadow", "off", "disagree"] as StatusFilter[]).map((f) => {
                const n = f === "all" ? rep.components.length : f === "disagree" ? disagree : rep.status_counts[f as BrainStatus];
                return (
                  <button key={f} type="button" className="vt-br-count" data-status={f} aria-pressed={filter === f} onClick={() => setFilter(f)}>
                    <span className="vt-br-count-n vt-fc-m">{n}</span>
                    <span className="vt-br-count-l">{f === "all" ? "All parts" : f === "disagree" ? "Short vs long disagree" : BRAIN_STATUS_LABEL[f as BrainStatus]}</span>
                    {f !== "all" && f !== "disagree" ? <span className="vt-fc-meta">{STATUS_HINT[f as BrainStatus]}</span> : null}
                  </button>
                );
              })}
            </div>
            <p className="vt-fc-note">{rep.short_term.note} {rep.short_term.primary_outcome}.</p>
          </section>

          {shown.length === 0 ? <div className="vt-fc-state">No component matches this filter.</div> : shown.map((g) => (
            <section key={g.kind} className="vt-fc-sec" aria-labelledby={`vt-br-k-${g.kind}`}>
              <div className="vt-fc-sechead"><h2 id={`vt-br-k-${g.kind}`}>{BRAIN_KIND_LABEL[g.kind]}</h2><span className="vt-fc-meta">{g.items.length}</span></div>
              <div className="vt-br-list">
                {g.items.map((c) => <ComponentCard key={c.id} c={c} now={now} />)}
              </div>
            </section>
          ))}

          <Combinations rep={rep} now={now} />

          <footer className="vt-fc-foot vt-br-foot">
            <p><b>Known gaps.</b></p>
            {rep.long_term.gaps.map((g) => <p key={g}>{g}</p>)}
            <p>Long-term sources: {rep.long_term.sources.map((s) => `${s.file} (${s.date})`).join(" · ")}.</p>
          </footer>
        </>
      ) : null}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="vt-br-list" aria-busy="true" aria-label="Loading the Brain page">
      {Array.from({ length: 5 }, (_, k) => <div key={k} className="vt-fc-card vt-fc-skel" />)}
    </div>
  );
}

function ComponentCard({ c, now }: { c: BrainComponent; now: number }) {
  return (
    <article className="vt-br-card" data-status={c.status} aria-label={`${c.name}: ${BRAIN_STATUS_LABEL[c.status]}`}>
      <div className="vt-br-cardtop">
        <span className="vt-br-name">{c.name}</span>
        <span className="vt-br-status" data-status={c.status} title={STATUS_HINT[c.status]}>{BRAIN_STATUS_LABEL[c.status]}</span>
        <span className="vt-fc-chip" data-small="true">{c.group}</span>
      </div>
      <p className="vt-br-does">{c.does}</p>
      <p className="vt-fc-meta">why {BRAIN_STATUS_LABEL[c.status].toLowerCase()}: {c.detail} <span className="vt-br-src">[{c.status_source}]</span></p>
      <div className="vt-br-sides">
        <ShortPanel s={c.short_term} now={now} />
        <LongPanel l={c.long_term} />
      </div>
      <div className="vt-br-agree" data-flag={c.agreement.flag}>
        <span className="vt-br-agree-l">{AGREE_LABEL[c.agreement.flag]}</span>
        <span className="vt-fc-meta">{c.agreement.note}</span>
      </div>
    </article>
  );
}

function SignPill({ sign }: { sign: ShortTerm["sign"] }) {
  return <span className="vt-br-sign" data-sign={sign}>{SIGN_LABEL[sign]}</span>;
}

function ShortPanel({ s, now }: { s: ShortTerm; now: number }) {
  return (
    <div className="vt-br-side" data-side="short">
      <div className="vt-br-sidehead"><span>Short-term · live shadow</span><SignPill sign={s.sign} /></div>
      {!s.measured ? (
        <p className="vt-br-gap">{s.reason ?? "Not yet measured."}</p>
      ) : s.metric === "ic" ? (
        <>
          <div className="vt-br-big vt-fc-m">IC {icText(s.ic)} <span className="vt-fc-meta">n={s.n}{s.n_days ? ` · ${s.n_days} days` : ""}</span></div>
          <div className="vt-fc-meta vt-fc-m">{ciText(s.ci95, "", 3)}{s.q_bh !== undefined ? ` · BH q ${s.q_bh}` : ""}</div>
          {s.verdict ? <div className="vt-fc-meta">verdict <b className="vt-br-verdict" data-verdict={s.verdict}>{s.verdict.replace(/_/g, " ")}</b></div> : null}
        </>
      ) : s.metric === "forecast" ? (
        <div className="vt-br-big vt-fc-m">{frac(s.hit_rate, 1)} hits <span className="vt-fc-meta">n={s.n}</span></div>
      ) : (
        <>
          <div className="vt-br-big vt-fc-m">{signedPct(s.mean_pct)} <span className="vt-fc-meta">mean · n={s.n} · {frac(s.hit_rate)} positive</span></div>
          <div className="vt-fc-meta vt-fc-m">{ciText(s.ci95)}</div>
          <div className="vt-fc-meta vt-fc-m">{s.vs_spy ? `vs SPY same window ${signedPct(s.vs_spy.mean_excess_pct)} · ${ciText(s.vs_spy.ci95)}` : "vs SPY: benchmark unavailable"}</div>
          {s.horizons ? (
            <div className="vt-fc-meta vt-fc-m">
              {Object.entries(s.horizons).filter(([k]) => k !== "xa").map(([k, h]) => `${k} ${signedPct(h.mean_pct)} (n=${h.n})`).join(" · ")}
            </div>
          ) : null}
        </>
      )}
      {s.measured ? (
        <>
          <p className="vt-fc-meta">{s.what}{s.bucket ? ` · bucket ${s.bucket}` : ""}</p>
          <p className="vt-br-prov">{s.sign_note ? `${s.sign_note} · ` : ""}{windowText(s.window)} · {s.source}{s.as_of ? ` · ${ageText(s.as_of, now)} old` : ""}</p>
        </>
      ) : null}
    </div>
  );
}

function LongPanel({ l }: { l: LongTerm }) {
  return (
    <div className="vt-br-side" data-side="long">
      <div className="vt-br-sidehead"><span>Long-term · multi-year tests</span><SignPill sign={l.sign} /></div>
      {l.items.length === 0 ? <p className="vt-br-gap">{l.reason ?? "Not yet measured."}</p> : (
        <>
          {l.items.map((it) => (
            <div key={it.id} className="vt-br-item">
              <div className="vt-br-itemtitle">{it.title}{it.live ? <span className="vt-fc-chip" data-small="true">live file</span> : null}</div>
              <p className="vt-br-finding">{it.finding}</p>
              <p className="vt-br-prov">{longProv(it)}</p>
            </div>
          ))}
          {l.reason ? <p className="vt-br-gap">{l.reason}</p> : null}
        </>
      )}
    </div>
  );
}

function longProv(it: LongTerm["items"][number]): string {
  const parts = [
    it.horizon && it.horizon !== "—" ? it.horizon : "",
    it.compared_to && it.compared_to !== "—" ? `vs ${it.compared_to}` : "",
    it.variants_tried ? `${it.variants_tried} variant${it.variants_tried === 1 ? "" : "s"} tried` : "",
    `${it.source_file} (${it.date})`,
  ];
  return parts.filter(Boolean).join(" · ");
}

function Combinations({ rep, now }: { rep: BrainReport; now: number }) {
  const st = rep.combinations.short_term;
  const tried = st.arms.reduce((t, a) => t + a.variants_tried, 0);
  const head = headlineArm(st.arms);
  return (
    <section className="vt-fc-sec" aria-labelledby="vt-br-combos">
      <div className="vt-fc-sechead">
        <h2 id="vt-br-combos">Best combinations (discounted for {tried} tried)</h2>
        <span className="vt-fc-meta">{st.as_of ? `evidence ${ageText(st.as_of, now)} old` : "no evidence"}</span>
      </div>
      <p className="vt-fc-note">{st.source}. {st.family ? `Single sources: ${st.family}.` : ""}</p>
      {!st.available ? <div className="vt-fc-state">{st.reason ?? "Forward selection has not run."}</div> : (
        <>
          {head ? <p className="vt-br-headline">{armHeadline(head)}</p> : null}
          <div className="vt-br-arms">
            {st.arms.map((a) => <ArmCard key={a.arm} a={a} />)}
          </div>
          {st.ablation ? (
            <div className="vt-br-sub">
              <h3>Ablation — the live score with each group removed</h3>
              {st.ablation.status !== "OK" ? <p className="vt-fc-note">{st.ablation.status ?? "—"}: {st.ablation.reason ?? ""}</p> : (
                <>
                  <p className="vt-fc-note">
                    {st.ablation.n_rows ?? 0} rows with logged components. Baseline score {st.ablation.baseline_inverted ? "ranks backwards (negative IC)" : "ranks forwards"};
                    removing the unvalidated adjustments {st.ablation.un_inverts ? "un-inverts it" : "does not un-invert it"}
                    {st.ablation.improvement_significant ? " (significant)." : " (not significant at 1.96 SE)."}
                  </p>
                  <div className="vt-fc-tablewrap">
                    <table className="vt-fc-table">
                      <thead><tr><th>Variant</th><th>Removed</th><th>n</th><th>IC</th><th>95% CI</th><th>Δ IC vs live</th><th>Q5−Q1</th></tr></thead>
                      <tbody>
                        {st.ablation.rows.map((r) => (
                          <tr key={r.name}>
                            <td className="vt-fc-m">{r.name}</td><td>{r.removed.length ? r.removed.join(", ") : "—"}</td><td className="vt-fc-m">{r.n ?? "—"}</td>
                            <td className="vt-fc-m">{icText(r.ic)}</td><td className="vt-fc-m">{r.ci95 ? `${icText(r.ci95[0])} … ${icText(r.ci95[1])}` : "—"}</td>
                            <td className="vt-fc-m">{r.ic_change_vs_baseline === null ? "—" : `${icText(r.ic_change_vs_baseline)}${r.ic_change_se ? ` ± ${r.ic_change_se.toFixed(3)}` : ""}`}</td>
                            <td className="vt-fc-m">{signedPct(r.q5_minus_q1_pct)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          ) : null}
        </>
      )}
      <div className="vt-br-sub">
        <h3>Long-term combinations (multi-year)</h3>
        {rep.combinations.long_term.map((lc) => (
          <div key={lc.id} className="vt-br-item">
            <div className="vt-br-itemtitle">{lc.title} <span className="vt-fc-meta">· {lc.variants_tried} arms tried</span></div>
            <div className="vt-fc-tablewrap">
              <table className="vt-fc-table">
                <thead><tr>{lc.columns.map((h) => <th key={h}>{h}</th>)}</tr></thead>
                <tbody>{lc.rows.map((r) => <tr key={r[0]}>{r.map((v, k) => <td key={k} className="vt-fc-m">{v}</td>)}</tr>)}</tbody>
              </table>
            </div>
            <p className="vt-br-finding">{lc.discount}</p>
            <p className="vt-br-prov">{lc.horizon} · {lc.source_file} ({lc.date})</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function armHeadline(a: SelectionArm): string {
  if (a.status === "SELECTED") {
    const g = a.holdout_vs_live_score;
    return `Best combination found: ${a.selected.join(" + ")} — holdout IC ${icText(a.holdout_selected?.ic)} vs the live score's ${icText(a.holdout_live_score?.ic)}${g ? ` (gain ${icText(g.ic_gain)}${g.se ? ` ± ${g.se.toFixed(3)}` : ""})` : ""}, after a Bonferroni discount for ${a.variants_tried} candidates tried.`;
  }
  if (a.status === "NOTHING_SELECTED") return `No combination beat chance after the discount: ${a.variants_tried} candidates were tried and none added significant out-of-sample IC. Nothing is better than the live score on this evidence${a.holdout_live_score?.ic !== undefined ? ` (live score holdout IC ${icText(a.holdout_live_score.ic)})` : ""}.`;
  return `Not enough data for a combination search yet: ${a.reason ?? a.status}.`;
}

function ArmCard({ a }: { a: SelectionArm }) {
  return (
    <div className="vt-br-arm">
      <div className="vt-br-itemtitle">{a.label} <span className="vt-br-verdict" data-verdict={a.status}>{a.status.replace(/_/g, " ")}</span></div>
      <p className="vt-fc-meta vt-fc-m">{a.n_rows ?? 0} rows · {a.candidates ?? 0} candidates · {a.variants_tried} tried{a.holdout_window ? ` · holdout ${windowText(a.holdout_window)}` : ""}</p>
      {a.reason ? <p className="vt-fc-note">{a.reason}</p> : null}
      {a.steps.length ? (
        <div className="vt-fc-tablewrap">
          <table className="vt-fc-table">
            <thead><tr><th>Step</th><th>Candidate</th><th>Select IC</th><th>Gain</th><th>p (Bonferroni)</th><th>Tried</th><th>Kept</th></tr></thead>
            <tbody>
              {a.steps.map((s, k) => (
                <tr key={`${s.candidate}${k}`}>
                  <td className="vt-fc-m">{k + 1}</td><td className="vt-fc-m">{s.candidate}</td><td className="vt-fc-m">{icText(s.select_ic)}</td>
                  <td className="vt-fc-m">{icText(s.gain)}</td><td className="vt-fc-m">{s.p_bonferroni}</td><td className="vt-fc-m">{s.tried}</td><td>{s.accepted ? "yes" : "no"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
