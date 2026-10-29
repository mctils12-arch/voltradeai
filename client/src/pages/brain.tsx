// Brain page (2026-10-29, human-asked: "how would I know what is being used at
// the moment and what combinations could have produced better results ... the
// short term is one thing but the long term is different"). Every input, rule,
// model and sleeve the bot has, its status RIGHT NOW (live / shadow / off,
// derived from the real switches), the short-term evidence (live shadow
// outcomes) beside the long-term evidence (multi-year tests that exist), the
// agreement flag, and the best combinations with their multiple-testing
// discount. Every number carries its window, source and age; gaps say so.
// Data: GET /api/bot/brain (server/brain.ts; owner only).
// Layout (2026-10-29, human-asked): the AI Engine's cockpit — the status
// counts are the KPI strip (and still the filters), the component cards flow
// into a full-width grid (short/long evidence side by side inside a card where
// it is wide enough), combinations and known gaps are wide panels.
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Brain, RefreshCw } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { ageText } from "@/lib/forecast";
import {
  type BrainReport, type BrainComponent, type ShortTerm, type LongTerm, type SelectionArm, type StatusFilter,
  BRAIN_KIND_LABEL, BRAIN_STATUS_LABEL, STATUS_HINT, AGREE_LABEL, SIGN_LABEL,
  signedPct, frac, ciText, icText, windowText, groupByKind, filterComponents, headlineArm, brainKpis,
} from "@/lib/brain";
import { Bar, KpiStrip, KpiTile, Panel } from "./cockpit";

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
  const kpis = useMemo(() => (rep ? brainKpis(rep, now) : []), [rep, now]);

  return (
    <div className="vt-ck vt-br" data-vt-brain>
      <header className="vt-ck-head">
        <h1><Brain size={16} aria-hidden="true" /> Brain</h1>
        <span className="vt-fc-chip" title="Read-only: this page changes nothing">What the bot uses, and the evidence</span>
        <div className="vt-ck-fresh" aria-live="polite">
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
        <p className="vt-br-explainer">
          {rep?.explainer ?? "Short-term = live shadow outcomes, long-term = multi-year tests; a component is promoted only when both agree."}
        </p>
        <p className="vt-ck-sub">{rep?.rule ?? "Live beats backtest when they disagree."}</p>
        {rep?.evidence_error ? (
          <div className="vt-fc-state" data-tone="orange" role="status" style={{ flex: "1 1 100%" }}>
            <b>Evidence call failed:</b> {rep.evidence_error}. Status below is still live (env + config); {rep.evidence_generated_at ? "evidence shown is the last good copy." : "short-term numbers are missing until the next refresh."}
          </div>
        ) : null}
      </header>

      {rep ? (
        <KpiStrip n={6} label="Filter by status">
          {kpis.map((k) => k.key === "evidence"
            ? <KpiTile key={k.key} label={k.label} value={k.value} line={k.line} tone={k.tone} status={k.key} />
            : <KpiTile key={k.key} label={k.label} value={k.value} line={k.line} tone={k.tone} status={k.key} onClick={() => setFilter(k.key as StatusFilter)} pressed={filter === k.key} />)}
        </KpiStrip>
      ) : null}

      {q.isLoading ? <Skeleton /> : q.isError && !rep ? (
        <div className="vt-fc-state" data-tone="red" role="alert">
          <b>Brain service unavailable.</b> {String(q.error instanceof Error ? q.error.message : q.error).slice(0, 160)} — the page retries every 5 minutes.
        </div>
      ) : rep ? (
        <div className="vt-ck-grid" data-basis="wide">
          <Bar id="vt-br-now" title="What is running right now" meta={`entries ${rep.switches.entry_shadow.value ? `VOLTRADE_SHADOW_ENTRIES=${rep.switches.entry_shadow.value}` : "VOLTRADE_SHADOW_ENTRIES off"} (${rep.switches.entry_shadow.source})`}>
            <p className="vt-fc-note">{rep.short_term.note} {rep.short_term.primary_outcome}.</p>
          </Bar>
          {shown.length === 0 ? <div className="vt-fc-state vt-ck-full">No component matches this filter.</div> : shown.map((g) => g.items.map((c) => <ComponentCard key={c.id} c={c} now={now} />))}

          <Combinations rep={rep} now={now} />

          <Panel id="vt-br-gaps" title="Known gaps" span="full" meta={`${rep.long_term.gaps.length} listed`}>
            <div className="vt-br-gaps">
              {rep.long_term.gaps.map((g) => <p key={g}>{g}</p>)}
            </div>
            <p className="vt-br-prov">Long-term sources: {rep.long_term.sources.map((s) => `${s.file} (${s.date})`).join(" · ")}.</p>
          </Panel>
        </div>
      ) : null}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="vt-ck-grid" data-basis="wide" aria-busy="true" aria-label="Loading the Brain page">
      {Array.from({ length: 4 }, (_, k) => <div key={k} className="vt-ck-panel vt-fc-skel" />)}
    </div>
  );
}

function ComponentCard({ c, now }: { c: BrainComponent; now: number }) {
  return (
    <article className="vt-ck-panel vt-br-card" data-status={c.status} aria-label={`${c.name}: ${BRAIN_STATUS_LABEL[c.status]}`}>
      <div className="vt-ck-ph">
        <span className="vt-bot-eb">{BRAIN_KIND_LABEL[c.kind]}{c.group !== BRAIN_KIND_LABEL[c.kind] ? ` · ${c.group}` : ""}</span>
        <h3 className="vt-ck-ti">{c.name}</h3>
        <span className="vt-br-status" data-status={c.status} title={STATUS_HINT[c.status]}>{BRAIN_STATUS_LABEL[c.status]}</span>
      </div>
      <div className="vt-ck-pb">
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
    <Panel id="vt-br-combos" title={`Best combinations (discounted for ${tried} tried)`} span="full" meta={st.as_of ? `evidence ${ageText(st.as_of, now)} old` : "no evidence"}>
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
    </Panel>
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
