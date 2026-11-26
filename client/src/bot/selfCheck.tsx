// Bot page — the self-check and alerts blocks under the Issues panel
// ([PRODUCT] 2026-11-20, UI coverage batch A, second part). The Issues list is
// what the PAGE derives from the audit log; these two are what the BOT says
// about itself: its own diagnostics run (and the size multiplier and score bar
// that run sets) and the alerts it raised since the server last restarted.
// Every sentence is built in selfCheckView.ts (pure, tested) from a route
// field; each block is one line until opened, says where it came from and how
// old it is, and a route that is not there is a sentence, never a blank.
import { useCallback, useMemo, useState, type ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Help } from "./help";
import type { GlossaryKey } from "./glossary";
import { fmtAge, fmtClockET, fmtMonDay, fmtMoney } from "./format";
import { unavailableWords } from "./whyView";
import { parseSelfCheck, selfCheckSentence, checkItemWords, appliedNote, parseAlerts, alertsSentence, alertTone, ALERT_KIND_WORDS } from "./selfCheckView";
import { parseLimits, limitsSentence, usedWords } from "./limitsView";

const TONE: Readonly<Record<string, string>> = { ok: "vt-bot-t2", warn: "vt-bot-o", bad: "vt-bot-r", none: "vt-bot-t3" };

export function Fold({ id, title, help, summary, cls, children, onOpen }: { id: string; title: string; help: GlossaryKey; summary: string; cls: string; children?: ReactNode; onOpen?: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="vt-sc" data-sc={id} data-open={open ? "true" : undefined}>
      <div className="vt-sc-h">
        <button type="button" aria-expanded={open} disabled={!children} onClick={() => setOpen((o) => { if (!o) onOpen?.(); return !o; })}><span className="t">{title}</span><span className="chev" aria-hidden="true">{children ? (open ? "▾" : "▸") : ""}</span></button>
        <Help k={help} />
      </div>
      <p className={`vt-sc-s ${cls}`}>{summary}</p>
      {open && children ? <div className="vt-sc-b">{children}</div> : null}
    </section>
  );
}

export function SelfCheck({ q, appliedScoreBar, nowMs }: { q: UseQueryResult<unknown>; appliedScoreBar: number | null; nowMs: number }) {
  const v = useMemo(() => parseSelfCheck(q.data), [q.data]);
  if (!v) {
    return <Fold id="check" title="Self-check" help="selfCheck" cls="vt-bot-t3" summary={q.isLoading ? "Running the bot's self-check…" : unavailableWords(q.isError ? q.error : null, "the bot's self-check")} />;
  }
  const say = selfCheckSentence(v);
  const note = appliedNote(v, appliedScoreBar);
  const has = v.problems.length || v.warnings.length || v.housekeeping.length || v.recommendations.length || note;
  return (
    <Fold id="check" title="Self-check" help="selfCheck" cls={TONE[say.tone]} summary={say.text}>
      {has || v.ok ? (
        <>
          {v.problems.length ? <ul className="vt-sc-l" data-list="problems">{v.problems.map((p, i) => { const w = checkItemWords(p); return <li key={i} data-sev={p.severity ?? undefined}><b>{w.lead}</b>{p.severity ? <i className="sev">{p.severity}</i> : null} {w.text}{w.fix ? <span className="fix"> → {w.fix}</span> : null}</li>; })}</ul> : null}
          {v.warnings.length ? <ul className="vt-sc-l" data-list="warnings">{v.warnings.map((p, i) => { const w = checkItemWords(p); return <li key={i}><b>{w.lead}</b> {w.text}</li>; })}</ul> : null}
          {!v.problems.length && !v.warnings.length && v.ok ? <div className="vt-bot-t3">Nothing found: data freshness, the model, disk and data sources all passed.</div> : null}
          {note ? <div className="vt-bot-o">{note}</div> : null}
          {v.recommendations.length ? <div><span className="vt-bot-t3">It suggests:</span> {v.recommendations.join(" · ")}</div> : null}
          {v.housekeeping.length ? <div className="vt-bot-t3">Housekeeping it did: {v.housekeeping.join(" · ")}</div> : null}
          <div className="vt-sc-src">the bot's own diagnostics (diagnostics.py), run for this page{q.dataUpdatedAt ? ` ${fmtAge(nowMs - q.dataUpdatedAt)} ago` : ""} · the loop runs the same check itself every fifth scan and applies its size and score settings; this is a fresh run, not the loop's last one · the win-rate lines read the bot's own trade log, not the broker's lots</div>
        </>
      ) : null}
    </Fold>
  );
}

// SAFETY LIMITS ([PRODUCT] 2026-11-20, limitsView.ts over GET /api/bot/safety-limits): every guard that can block
// entries or stop the bot, with the reading the guard itself keeps, its limit, how far along it is (the bar) and what
// it does at the limit. One line until opened: what is tripped or near, then the three headline readings.
export function SafetyLimits({ q, nowMs }: { q: UseQueryResult<unknown>; nowMs: number }) {
  const v = useMemo(() => parseLimits(q.data), [q.data]);
  if (!v) return <Fold id="limits" title="Safety limits" help="safetyLimits" cls="vt-bot-t3" summary={q.isLoading ? "Reading the safety limits…" : unavailableWords(q.isError ? q.error : null, "the safety-limits read")} />;
  const say = limitsSentence(v);
  return (
    <Fold id="limits" title="Safety limits" help="safetyLimits" cls={TONE[say.tone]} summary={say.text}>
      <div className="vt-sc-lims" role="list">
        {v.rows.map((r) => (
          <div key={r.id} className="vt-sc-lim" role="listitem" data-limit={r.id} data-state={r.state}>
            <span className="n">{r.name}</span>
            <span className="bar" aria-hidden="true"><i style={{ width: `${Math.min(100, Math.round((r.used ?? 0) * 100))}%` }} /></span>
            <span className="u">{usedWords(r)}</span>
            <span className="r">{r.reading}{r.limit ? ` · limit ${r.limit}` : ""}{r.tripped ? <span className="t"> · {r.tripped}</span> : null}</span>
            <span className="e">at the limit: {r.effect}</span>
          </div>
        ))}
      </div>
      {v.peaksNote ? <div className="vt-bot-o" data-lim="peaks">{v.peaksNote}</div> : null}
      {v.engineError ? <div className="vt-bot-o" data-lim="engine-error">The engine's own guards did not answer ({v.engineError}): only this loop's drawdown kill is shown.</div> : null}
      {v.others.length ? <ul className="vt-sc-l" data-list="others">{v.others.map((t, i) => <li key={i}>{t}</li>)}</ul> : null}
      {v.events.length ? <div className="vt-bot-t3" data-lim="events">Kill switch history: {v.events.join(" · ")}</div> : null}
      <div className="vt-sc-src">each guard's own record{v.equity !== null ? `, measured against one account value (${fmtMoney(v.equity)})` : ""}{q.dataUpdatedAt ? ` · read ${fmtAge(nowMs - q.dataUpdatedAt)} ago` : ""} · the limits are settings: shown here, never changed from this page</div>
    </Fold>
  );
}

export const ALERTS_SHOWN = 20;
export function Alerts({ q, nowMs }: { q: UseQueryResult<unknown>; nowMs: number }) {
  const rows = useMemo(() => parseAlerts(q.data), [q.data]);
  const [err, setErr] = useState<string | null>(null);
  // opening the list is reading it: the server marks them read, the next poll shows no "new"
  const markRead = useCallback(() => {
    if (!rows || !rows.some((r) => !r.read)) return;
    apiRequest("POST", "/api/bot/notifications/read")
      .then(() => queryClient.invalidateQueries({ queryKey: ["/api/bot/notifications"] }))
      .catch((e: unknown) => { setErr(String(e instanceof Error ? e.message : e).slice(0, 80)); });
  }, [rows]);
  if (!rows) return <Fold id="alerts" title="Alerts" help="botAlerts" cls="vt-bot-t3" summary={q.isLoading ? "Reading the bot's alerts…" : unavailableWords(q.isError ? q.error : null, "the bot's alerts")} />;
  const unread = rows.filter((r) => !r.read).length;
  const worst = rows.some((r) => !r.read && r.kind === "critical") ? "bad" : rows.some((r) => !r.read && r.kind === "alert") ? "warn" : rows.length ? "ok" : "none";
  return (
    <Fold id="alerts" title={`Alerts${unread ? ` · ${unread} new` : ""}`} help="botAlerts" cls={TONE[worst]} summary={alertsSentence(rows, nowMs)} onOpen={markRead}>
      {rows.length ? (
        <>
          <div className="vt-sc-al" role="list">
            {rows.slice(0, ALERTS_SHOWN).map((r) => (
              <div key={r.key} className="vt-sc-ar" role="listitem" data-kind={r.kind} data-unread={r.read ? undefined : "true"}>
                <span className="w">{fmtMonDay(r.time)} {fmtClockET(r.time).slice(0, 5)}</span>
                <span className={`k ${TONE[alertTone(r.kind)]}`}>{ALERT_KIND_WORDS[r.kind]}</span>
                <span className="m">{r.message}</span>
              </div>
            ))}
          </div>
          {err ? <div className="vt-bot-o" role="status">could not mark them read · {err}</div> : null}
          <div className="vt-sc-src">{rows.length > ALERTS_SHOWN ? `the newest ${ALERTS_SHOWN} of ${rows.length} · ` : ""}kept in the server's memory (the newest 100) and cleared by a restart or a deploy · times ET · the full record is the audit log on the Cockpit's timeline · fetched {q.dataUpdatedAt ? `${fmtAge(nowMs - q.dataUpdatedAt)} ago` : "—"}</div>
        </>
      ) : null}
    </Fold>
  );
}
