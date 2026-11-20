// Bot page — the evidence blocks ([PRODUCT] 2026-11-20, UI coverage batch A):
// the closed-lot table and its CSV under the Book, the order history (each
// price-ladder step) in the Orders panel, the base holding against its own
// rule and what each blocking rule prevented as sections of the Why panel, and
// the option-quote-feed line under the Book's options. Every sentence and
// number is built in evidenceView.ts (pure, pinned by its test) from a server
// field; every block says where it came from and how old it is; absence is a
// designed sentence. Tables reuse the Book's mono table (.vt-exp-t: the first
// column pinned, the wrapper scrolls sideways on a phone — no cell is cut and
// the page never overflows).
import { useCallback, useMemo, useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Help } from "./help";
import { St, NotRecorded } from "./panels";
import { fmtAge, fmtClockET, fmtMoney, fmtMonDay, fmtPct, fmtSigned } from "./format";
import { unavailableWords, dayWords } from "./whyView";
import {
  parseLots, lotsCsv, lotsCaption, lotSideWords, heldWords, parseOrderHistory, orderOutcome, orderAsk, orderStepWords, orderHistorySummary,
  floorTriggerWords, ruleGroupWords, signedUsd, RULE_VERDICT_WORDS, type FloorView, type RulesView, type RegimeCfView, type OptionsFeedWords, type LotView,
} from "./evidenceView";

const tone = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "t3" : v < 0 ? "r" : v > 0 ? "g" : "t2");
const px = (v: number | null) => (v === null ? "—" : `$${v.toFixed(2)}`);
/** a lot's P&L to the cent, signed: a $0.40 option lot is not "$0" */
const usdCents = (v: number) => `${v > 0 ? "+" : ""}${fmtMoney(Math.round(v * 100) / 100, true)}`;
const fetchedWords = (iso: string | null, updatedAt: number, nowMs: number) => (iso && Number.isFinite(Date.parse(iso)) ? `ledger read ${fmtMonDay(iso)} ${fmtClockET(iso).slice(0, 5)} ET` : updatedAt ? `fetched ${fmtAge(nowMs - updatedAt)} ago` : "not fetched yet");

/** Hand the browser a text file to save. Returns false when the page cannot (no DOM, a blocked blob URL). */
function saveText(name: string, text: string): boolean {
  try {
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url; a.download = name; a.rel = "noopener";
    document.body.appendChild(a); a.click(); a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 5_000);
    return true;
  } catch (err) { console.warn("[bot page] the CSV could not be handed to the browser:", err); return false; }
}

// ── closed trades (the Book) ────────────────────────────────────────────────
export const LOTS_FIRST = 8;
export const LOTS_MORE = 50;
export function ClosedLots({ q, nowMs }: { q: UseQueryResult<unknown>; nowMs: number }) {
  const v = useMemo(() => parseLots(q.data), [q.data]);
  const [more, setMore] = useState(false);
  const [csv, setCsv] = useState<{ busy: boolean; note: string | null }>({ busy: false, note: null });
  // the CSV is the WHOLE ledger (the page itself holds only the newest lots): one read on demand
  const download = useCallback(async () => {
    setCsv({ busy: true, note: null });
    try {
      const r = await apiRequest("GET", "/api/bot/lots?limit=5000", undefined, { timeoutMs: 30_000 });
      const all = parseLots(await r.json());
      if (!all || !all.rows.length) { setCsv({ busy: false, note: "nothing to export yet" }); return; }
      const ok = saveText(`voltrade_closed_lots_${new Date().toISOString().slice(0, 10)}.csv`, lotsCsv(all.rows));
      setCsv({ busy: false, note: ok ? `${all.rows.length.toLocaleString("en-US")} lots saved${all.truncated ? ` (the newest of ${all.count.toLocaleString("en-US")})` : ""}` : "the browser refused the download" });
    } catch (e: unknown) { setCsv({ busy: false, note: `export failed · ${String(e instanceof Error ? e.message : e).slice(0, 80)}` }); }
  }, []);
  const head = <div className="vt-bot-eb vt-exp-eb">Closed trades<Help k="closedLots" /> · newest first · broker</div>;
  if (q.isLoading && !v) return <div className="vt-ev" data-ev="lots" data-state="loading" aria-busy="true">{head}<i className="vt-score-skel" style={{ width: "85%", height: 14 }} /><i className="vt-score-skel" style={{ width: "70%", height: 14 }} /></div>;
  if (!v) return <div className="vt-ev" data-ev="lots" data-state="absent">{head}<div className="vt-exp-src"><NotRecorded what={unavailableWords(q.isError ? q.error : null, "the closed-lot ledger")} /></div></div>;
  if (v.state !== "ready") return <div className="vt-ev" data-ev="lots" data-state={v.state}>{head}<div className="vt-exp-src"><St word={v.state === "pending" ? "the fill ledger is still computing · the page asks again shortly" : "no lot has closed in the fetched history"} /></div></div>;
  const rows = v.rows.slice(0, more ? LOTS_MORE : LOTS_FIRST);
  return (
    <div className="vt-ev" data-ev="lots" data-state="ready">
      {head}
      <div className="vt-exp-wrap">
        <table className="vt-exp-t vt-ev-t" aria-label={`Closed trades, the newest ${rows.length}`}>
          <thead><tr><th className="s" scope="col">position</th><th scope="col">P&amp;L</th><th scope="col">return</th><th scope="col">closed</th><th scope="col">held</th><th scope="col">entry</th><th scope="col">exit</th><th scope="col" className="l">what</th></tr></thead>
          <tbody>
            {rows.map((l: LotView) => (
              <tr key={l.key} data-side={l.side} title={`${l.symbol} · opened ${l.openedAt ? `${fmtMonDay(l.openedAt)} ${fmtClockET(l.openedAt).slice(0, 5)} ET` : "before the fetched history"} · closed ${fmtMonDay(l.closedAt)} ${fmtClockET(l.closedAt).slice(0, 5)} ET · cost ${l.basis === null ? "—" : fmtMoney(l.basis, true)}`}>
                <td className="s">{l.label}</td>
                <td className={tone(l.pnl)}>{l.pnl === null ? "—" : usdCents(l.pnl)}</td>
                <td className={tone(l.returnPct)}>{fmtPct(l.returnPct, 1)}</td>
                <td className="t2">{fmtMonDay(l.closedAt)}</td>
                <td className="t3">{heldWords(l.heldMs)}</td>
                <td>{px(l.entryPrice)}</td>
                <td>{px(l.exitPrice)}</td>
                <td className="l t2">{lotSideWords(l)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="vt-ev-act">
        {v.rows.length > LOTS_FIRST ? <button type="button" className="vt-bot-wbtn" aria-expanded={more} onClick={() => setMore((m) => !m)}>{more ? `show ${LOTS_FIRST}` : `show ${Math.min(LOTS_MORE, v.rows.length)}`}</button> : null}
        <button type="button" className="vt-bot-wbtn" disabled={csv.busy} onClick={() => void download()}>{csv.busy ? "preparing…" : "download CSV · every lot"}</button>
        <a className="vt-bot-wbtn" href="/api/bot/export-trades" download title="the bot's own trade log (score, strategy, P&L % at the trigger price): a different record from the broker's lots above">bot log CSV</a>
        {csv.note ? <span className="vt-bot-t3" role="status">{csv.note}</span> : null}
      </div>
      <div className="vt-exp-src">{lotsCaption(v, rows.length)} · {fetchedWords(v.fetchedAt, q.dataUpdatedAt, nowMs)}{v.lastError ? <span className="vt-bot-o"> · last refresh failed: {v.lastError.slice(0, 100)}</span> : null}</div>
    </div>
  );
}

// ── order history (the Orders panel) ────────────────────────────────────────
export const ORDERS_SHOWN = 30;
export function OrderHistory({ q, nowMs }: { q: UseQueryResult<unknown>; nowMs: number }) {
  const rows = useMemo(() => parseOrderHistory(q.data), [q.data]);
  const head = <div className="vt-bot-eb vt-exp-eb">Order history<Help k="orderHistory" /> · every order sent, newest first</div>;
  if (q.isLoading && !rows) return <div className="vt-ev" data-ev="orders" data-state="loading" aria-busy="true">{head}<i className="vt-score-skel" style={{ width: "80%", height: 14 }} /></div>;
  if (!rows) return <div className="vt-ev" data-ev="orders" data-state="absent">{head}<div className="vt-exp-src"><NotRecorded what={unavailableWords(q.isError ? q.error : null, "the order history")} /></div></div>;
  if (!rows.length) return <div className="vt-ev" data-ev="orders" data-state="empty">{head}<div className="vt-exp-src"><St word="no orders in the broker's recent history" /></div></div>;
  return (
    <div className="vt-ev" data-ev="orders" data-state="ready">
      {head}
      <div className="vt-ev-ol" role="list">
        {rows.slice(0, ORDERS_SHOWN).map((o) => { const out = orderOutcome(o); const step = orderStepWords(o); return (
          <div key={o.id} className="vt-ev-or" role="listitem" data-outcome={out.tone} title={`${o.symbol} · sent ${fmtMonDay(o.submittedAt)} ${fmtClockET(o.submittedAt)} ET`}>
            <span className="w">{fmtMonDay(o.submittedAt)} {fmtClockET(o.submittedAt).slice(0, 5)}</span>
            <span className="sym">{o.label}</span>
            <span className="ask">{orderAsk(o)}</span>
            <span className={`out ${out.tone === "ok" ? "vt-bot-g" : out.tone === "wait" ? "vt-bot-b" : "vt-bot-t3"}`}>{out.text}</span>
            {step ? <span className="step">{step}</span> : null}
          </div>); })}
      </div>
      <div className="vt-exp-src">{orderHistorySummary(rows)}{rows.length > ORDERS_SHOWN ? ` · the newest ${ORDERS_SHOWN} shown` : ""} · Alpaca orders, every status · times ET · fetched {q.dataUpdatedAt ? `${fmtAge(nowMs - q.dataUpdatedAt)} ago` : "—"} · a step is one of several orders for the same contract and side that day (the price ladder re-prices by cancelling and sending again)</div>
    </div>
  );
}

// ── the base holding (a Why section's body) ─────────────────────────────────
export function BaseBody({ f, updatedAt, nowMs }: { f: FloorView; updatedAt: number; nowMs: number }) {
  const trig = floorTriggerWords(f);
  const t = f.turnover;
  return (
    <>
      {f.sessions.length ? (
        <div className="vt-exp-wrap">
          <table className="vt-exp-t vt-ev-t" aria-label={`The base holding against its rule, the last ${f.sessions.length} sessions`}>
            <thead><tr><th className="s" scope="col">session</th><th scope="col">actual</th><th scope="col">rule</th><th scope="col">gap</th><th scope="col">{f.base ?? "base"}</th><th scope="col">target</th><th scope="col">held</th></tr></thead>
            <tbody>
              {f.sessions.map((s) => (
                <tr key={s.day} data-entry={s.entry ? "true" : undefined} title={s.reason ?? (s.entry ? "the session the holding was bought: measured from its fills, in no window" : undefined)}>
                  <td className="s">{dayWords(s.day)}{s.entry ? <i>entry</i> : s.rebalance ? <i>rebalanced</i> : null}</td>
                  <td className={tone(s.floorPct)}>{fmtPct(s.floorPct)}</td>
                  <td className={tone(s.rulePct)}>{fmtPct(s.rulePct)}</td>
                  <td className={tone(s.trackingPp)}>{fmtSigned(s.trackingPp, 2, " pt")}</td>
                  <td className={tone(s.spyPct)}>{fmtPct(s.spyPct)}</td>
                  <td>{s.target === null ? "—" : `${s.target.toFixed(2)}×`}</td>
                  <td>{s.held === null ? "—" : `${s.held.toFixed(2)}×`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {trig ? <div className={trig.tone === "warn" ? "vt-bot-r" : trig.tone === "ok" ? undefined : "vt-bot-t3"} data-trigger={trig.tone}>{trig.text}</div> : null}
      {f.entry ? <div>Bought {dayWords(f.entry.day)}{f.entry.shortfallPp !== null ? `: ${fmtSigned(f.entry.shortfallPp, 2, " pt")} against the rule that day, from entering after the open` : ""}. That day is in no window.</div> : null}
      {t && t.sessions !== null ? <div>Trading, last {t.sessions} sessions: {t.orders ?? 0} order{t.orders === 1 ? "" : "s"} sent, {t.filled ?? 0} filled{t.dollars ? `, ${fmtMoney(t.dollars)} traded` : ""}{t.rebalanceSessions !== null ? ` · ${t.rebalanceSessions} rebalance session${t.rebalanceSessions === 1 ? "" : "s"}` : ""}{t.outsideBandNoTrade ? ` · ${t.outsideBandNoTrade} session${t.outsideBandNoTrade === 1 ? "" : "s"} began outside the band with no trade` : ""}.</div> : null}
      <div className="vt-why-src">
        <span>floor tracking report · {f.mode ?? "mode not recorded"}{f.capitalPct !== null ? ` · on ${Math.round(f.capitalPct * 100)} % of equity` : ""}{f.levered ? ` · ${f.base} + ${f.levered}${f.cashFund ? `, idle cash in ${f.cashFund}` : ""}` : ""} · {f.generatedAt ? `computed ${fmtMonDay(f.generatedAt)} ${fmtClockET(f.generatedAt).slice(0, 5)} ET` : updatedAt ? `fetched ${fmtAge(nowMs - updatedAt)} ago` : "—"}</span>
        <span>rule = target × {f.base ?? "base"}'s close-to-close move, price only · actual = the holding's realized P&amp;L on the same capital · gap = actual − rule, in points of that capital</span>
        {f.trigger?.text ? <span>{f.trigger.text}</span> : null}
      </div>
    </>
  );
}

// ── what each blocking rule prevented (a Why section's body) ────────────────
export function RulesBody({ v, updatedAt, nowMs }: { v: RulesView; updatedAt: number; nowMs: number }) {
  return (
    <>
      {v.groups.length ? (
        <div className="vt-exp-wrap">
          <table className="vt-exp-t vt-ev-t vt-ev-t--stack" aria-label="What each rule blocked and what those names went on to do">
            <thead><tr><th className="s" scope="col">rule</th><th scope="col">their return</th><th scope="col">all scanned</th><th scope="col" className="l">blocked names did</th><th scope="col">judged</th><th scope="col">blocked</th><th scope="col">est. $ saved</th></tr></thead>
            <tbody>
              {v.groups.map((g) => (
                <tr key={g.key} data-verdict={g.verdict} title={`${g.name} · ${ruleGroupWords(g)} · ${dayWords(g.first)} to ${dayWords(g.last)} · ${g.pending} still waiting for their ${v.horizon}`}>
                  <td className="s">{g.name}<i>{ruleGroupWords(g)}</i></td>
                  <td className={tone(g.meanPct)}>{fmtPct(g.meanPct, 1)}</td>
                  <td className={tone(g.baseMeanPct)}>{fmtPct(g.baseMeanPct, 1)}{g.baseN !== null && g.baseMeanPct !== null ? <i className="n"> n {g.baseN}</i> : null}</td>
                  <td className={`l ${g.verdict === "thin" ? "t3" : "t2"}`}>{RULE_VERDICT_WORDS[g.verdict]}</td>
                  <td className={g.labeled < v.minN ? "t3" : undefined}>{g.labeled}</td>
                  <td>{g.nameDays}</td>
                  <td className={tone(g.preventionUsd)}>{g.preventionUsd === null ? "not sized" : signedUsd(Math.round(g.preventionUsd))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {v.rules.filter((r) => !r.nameDays).length ? <div className="vt-bot-t3">Never fired in the logged period: {v.rules.filter((r) => !r.nameDays).map((r) => r.name.toLowerCase()).join(", ")}.</div> : null}
      <div className="vt-why-src">
        <span>rule counterfactuals · shadow log {dayWords(v.first)} to {dayWords(v.last)} · {v.generatedAt ? `computed ${fmtMonDay(v.generatedAt)} ${fmtClockET(v.generatedAt).slice(0, 5)} ET` : updatedAt ? `fetched ${fmtAge(nowMs - updatedAt)} ago` : "—"}<Help k="ruleCounterfactuals" /></span>
        <span>blocked = name-days (one per name per day) · judged = those whose {v.horizon} have passed · their return = the stock held long with a +2 % target and −4 % stop, an ESTIMATE of direction, not option P&amp;L · a reading needs {v.minN} judged</span>
        <span>The reading compares two averages. It is not a statistical test: on this few names a gap of a point can be chance, so no rule is changed on it alone.</span>
        {v.basis ? <span>{v.basis}</span> : null}
        {v.limits.slice(0, 2).map((t, i) => <span key={i}>{t}</span>)}
      </div>
    </>
  );
}

// ── the regime engine against the old rule (a line in the regime section) ───
export function RegimeCfLine({ v, sentence }: { v: RegimeCfView; sentence: string }) {
  return <div className="vt-ev-cf" data-ev="regime-cf" title={v.note ?? undefined}>{sentence}</div>;
}

// ── which option quote feed (under the Book's options) ──────────────────────
export function OptionsFeedLine({ w }: { w: OptionsFeedWords | null }) {
  if (!w) return null;
  return <div className={`vt-ev-feed ${w.tone === "warn" ? "vt-bot-o" : "vt-bot-t3"}`} data-ev="options-feed" data-feed={w.feed} role="note">{w.text}{w.checkedAt ? ` · checked ${fmtMonDay(w.checkedAt)}` : ""}<Help k="optionsFeed" /></div>;
}
