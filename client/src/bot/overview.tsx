// Bot page OVERVIEW (human refocus spec 2026-11-05 §8: "simplify; larger main
// chart: portfolio vs SPY, after tax; per pick a plain-English card …; a clear
// 'why are we losing money' panel"). The three panel bodies and the view switch;
// every number is a server field (perfView / picks / whyView turn them into
// lines and sentences, pure and pinned by their tests), every block says where
// it came from and how old it is, and absence is a designed sentence, never a
// zero. Theme tokens only (DESIGN.md); the chart is SVG, no canvas.
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import type { Scoreboard, Benchmark, BotPosition, Sleeves } from "./useBotData";
import { ATTRIBUTION_WINDOWS } from "./useBotData";
import { perfModel, perfHeadline, yDomain, niceTicks, tickLabel, monthTicks, linePath, gapBands, nearestIndex, type PerfModel } from "./perfView";
import { parseDecisions, pickCards, closedPicks, passedOn, CLOSED_PICK_DAYS, PASSED_ON_HOURS, type PickCard, type MiniPick } from "./picks";
import { parseAttribution, attributionFromSleeves, whyHeadline, parseFlows, flowsSentence, parseTax, taxSentence, parseSignals, signalsSentence, parseRiskBudget, riskSentence, unavailableWords, signedMoney, dayWords, type AttrGroup, type AttributionView } from "./whyView";
import { fmtPct, fmtSigned, fmtMoney, fmtClockET, fmtMonDay, fmtAge, fmtOcc } from "./format";
import { humanError } from "./chart/BotChart";
import { Help } from "./help";
import type { GlossaryKey } from "./glossary";
import { St, NotRecorded } from "./panels";
import type { BotMode } from "./zoneModel";

const tone = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "vt-bot-t3" : v < 0 ? "vt-bot-r" : v > 0 ? "vt-bot-g" : "vt-bot-t2");
const asOfWords = (iso: string | null | undefined, updatedAt: number, nowMs: number) => (iso && Number.isFinite(Date.parse(iso)) ? `as of ${fmtMonDay(iso)} ${fmtClockET(iso).slice(0, 5)} ET` : updatedAt ? `fetched ${fmtAge(nowMs - updatedAt)} ago` : "not fetched yet");

// ── the view switch ─────────────────────────────────────────────────────────
export function ModeBar({ mode, onMode, inNav = false }: { mode: BotMode; onMode: (m: BotMode) => void; inNav?: boolean }) {
  return (
    <div className="vt-bot-modebar" data-mode={mode} data-nav={inNav ? "true" : undefined}>
      <div className="vt-bot-seg" role="group" aria-label="Bot page view">
        <button type="button" data-view="overview" aria-pressed={mode === "overview"} onClick={() => onMode("overview")}>Overview</button>
        <button type="button" data-view="cockpit" aria-pressed={mode === "cockpit"} onClick={() => onMode("cockpit")}>Cockpit</button>
      </div>
      {inNav ? null : <span className="vt-bot-modebar-say">{mode === "overview" ? "after tax vs SPY, why money is lost, and every pick in plain English" : "every engine panel — stages, scan, model, candles, timeline"}</span>}
      <Help k="viewMode" />
    </div>
  );
}

// ── the main chart: portfolio vs SPY, after tax ─────────────────────────────
function useBox<T extends HTMLElement>(): [React.RefObject<T>, { w: number; h: number }] {
  const ref = useRef<T>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => { const r = el.getBoundingClientRect(); setBox((b) => (Math.abs(b.w - r.width) < 1 && Math.abs(b.h - r.height) < 1 ? b : { w: r.width, h: r.height })); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, box];
}

const DD_H = 46, AX_H = 18, PAD_L = 46, PAD_R = 14, PAD_T = 10;
function PerfSvg({ m, fill }: { m: PerfModel; fill: boolean }) {
  const [ref, box] = useBox<HTMLDivElement>();
  const [idx, setIdx] = useState<number | null>(null);
  const pts = m.points;
  const n = pts.length;
  const w = Math.max(0, Math.round(box.w));
  const h = Math.max(0, Math.round(box.h));
  const x1 = Math.max(PAD_L + 10, w - PAD_R);
  const mainBottom = Math.max(PAD_T + 40, h - DD_H - AX_H - 6);
  const vals = [...pts.flatMap((p) => [p.bot, p.spy, p.botPre, p.spyPre]), ...(m.soldNow ? [m.soldNow.bot, m.soldNow.spy] : [])];
  const [lo, hi] = yDomain(vals);
  const x = (i: number) => PAD_L + (n <= 1 ? 0 : (i / (n - 1)) * (x1 - PAD_L));
  const y = (v: number) => PAD_T + ((hi - v) / (hi - lo)) * (mainBottom - PAD_T);
  const ddTop = mainBottom + AX_H + 6, ddBottom = h - 2;
  const ddLo = Math.min(-1, ...pts.map((p) => Math.min(p.botDd, p.spyDd))) * 1.08;
  const yd = (v: number) => ddTop + (v / ddLo) * (ddBottom - ddTop);
  const ticks = niceTicks(lo, hi, h < 260 ? 4 : 6);
  const months = monthTicks(pts.map((p) => p.date));
  const step = Math.max(1, Math.ceil(months.length / Math.max(1, Math.floor((x1 - PAD_L) / 46))));
  // ~130 points: recomputed per render (a resize or a new day), no memo needed
  const bands = w ? gapBands(pts.map((p) => p.bot), pts.map((p) => p.spy), x, y) : [];
  const sel = idx !== null && idx < n ? pts[idx] : null;
  const pick = (clientX: number, el: SVGSVGElement) => { const r = el.getBoundingClientRect(); setIdx(nearestIndex(clientX - r.left, n, PAD_L, x1)); };
  const label = m.last ? `${m.afterTax ? "After tax" : "Before tax"}: portfolio ${fmtPct(m.last.bot)}, SPY ${fmtPct(m.last.spy)} since ${dayWords(m.from)}, ${n} trading days` : "no data";
  return (
    <div className="vt-perf-plot" ref={ref} data-fill={fill ? "true" : undefined}>
      {w > 0 && h > 0 ? (
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label} tabIndex={0}
          onPointerMove={(e) => { if (e.pointerType === "mouse") pick(e.clientX, e.currentTarget); }}
          onPointerDown={(e) => pick(e.clientX, e.currentTarget)}
          onPointerLeave={(e) => { if (e.pointerType === "mouse") setIdx(null); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); setIdx((i) => Math.max(0, Math.min(n - 1, (i ?? n - 1) + (e.key === "ArrowLeft" ? -1 : 1)))); }
            else if (e.key === "Escape") setIdx(null);
          }}>
          {ticks.map((t) => <g key={`t${t}`}><line className={t === 0 ? "vt-perf-zero" : "vt-perf-grid"} x1={PAD_L} x2={x1} y1={y(t)} y2={y(t)} /><text className="vt-perf-ylab" x={PAD_L - 6} y={y(t) + 3.5} textAnchor="end">{tickLabel(t)}</text></g>)}
          {months.filter((_, k) => k % step === 0).map((t) => <text key={`m${t.i}`} className="vt-perf-xlab" x={x(t.i)} y={mainBottom + 13} textAnchor="middle">{t.label}</text>)}
          {bands.map((b, k) => <path key={`b${k}`} className="vt-perf-band" data-sign={b.sign > 0 ? "ahead" : "behind"} d={b.d} />)}
          {m.afterTax ? <><path className="vt-perf-line vt-perf-pre" data-line="spy-pre" d={linePath(pts.map((p) => p.spyPre), x, y)} /><path className="vt-perf-line vt-perf-pre vt-perf-botc" data-line="bot-pre" d={linePath(pts.map((p) => p.botPre), x, y)} /></> : null}
          <path className="vt-perf-line vt-perf-spyc" data-line="spy" d={linePath(pts.map((p) => p.spy), x, y)} />
          <path className="vt-perf-line vt-perf-botc vt-perf-main" data-line="bot" d={linePath(pts.map((p) => p.bot), x, y)} />
          {m.soldNow ? <><circle className="vt-perf-sold vt-perf-spyc" data-sold="spy" cx={x(n - 1)} cy={y(m.soldNow.spy)} r={4} /><circle className="vt-perf-sold vt-perf-botc" data-sold="bot" cx={x(n - 1)} cy={y(m.soldNow.bot)} r={4} /></> : null}
          <line className="vt-perf-zero" x1={PAD_L} x2={x1} y1={ddTop} y2={ddTop} />
          <path className="vt-perf-dd vt-perf-botc" data-line="bot-dd" d={`${linePath(pts.map((p) => p.botDd), x, yd)} L${x(n - 1).toFixed(1)} ${ddTop} L${x(0).toFixed(1)} ${ddTop} Z`} />
          <path className="vt-perf-line vt-perf-spyc vt-perf-ddl" data-line="spy-dd" d={linePath(pts.map((p) => p.spyDd), x, yd)} />
          <text className="vt-perf-ylab" x={PAD_L - 6} y={ddTop + 10} textAnchor="end">dd</text>
          {sel && idx !== null ? (
            <g className="vt-perf-cross">
              <line x1={x(idx)} x2={x(idx)} y1={PAD_T} y2={ddBottom} />
              <circle className="vt-perf-botc" cx={x(idx)} cy={y(sel.bot)} r={3.5} />
              <circle className="vt-perf-spyc" cx={x(idx)} cy={y(sel.spy)} r={3.5} />
            </g>
          ) : null}
        </svg>
      ) : null}
      {sel && idx !== null ? (
        <div className="vt-perf-read" data-side={x(idx) > w / 2 ? "left" : "right"} role="status">
          <b>{dayWords(sel.date)}</b>
          <span><i className="vt-perf-sw vt-perf-botc" />portfolio {fmtPct(sel.bot)}{m.afterTax ? <em> · before tax {fmtPct(sel.botPre)}</em> : null}</span>
          <span><i className="vt-perf-sw vt-perf-spyc" />SPY {fmtPct(sel.spy)}{m.afterTax ? <em> · before tax {fmtPct(sel.spyPre)}</em> : null}</span>
          <span className={tone(sel.bot - sel.spy)}>gap {fmtSigned(sel.bot - sel.spy, 2, " pt")}</span>
          <span className="vt-bot-t3">drawdown {fmtPct(sel.botDd, 1)} · SPY {fmtPct(sel.spyDd, 1)}</span>
        </div>
      ) : null}
    </div>
  );
}

export function PerfBody({ q, bench, nowMs, fill }: { q: UseQueryResult<Scoreboard>; bench: Benchmark | undefined; nowMs: number; fill: boolean }) {
  const m = useMemo(() => perfModel(q.data, bench, { loading: q.isLoading, error: q.isError ? humanError(q.error) : null }), [q.data, bench, q.isLoading, q.isError, q.error]);
  if (m.state === "loading") return <div className="vt-perf" data-state="loading" aria-busy="true" aria-label="Loading the after-tax chart"><i className="vt-score-skel" style={{ width: "55%", height: 24 }} /><i className="vt-score-skel" style={{ width: "80%", height: 12 }} /><i className="vt-score-skel vt-perf-skelplot" /></div>;
  if (m.state === "error" || m.state === "empty" || !m.last) {
    return (
      <div className="vt-perf vt-bot-empty" data-state={m.state} role="status">
        {m.state === "error" ? <St word="chart unavailable" cls="vt-bot-o" /> : <NotRecorded what="not enough history yet" />}
        <span>{m.note ?? "nothing to draw yet"}{m.state === "error" ? " · the page asks again every 10 min" : ""}</span>
      </div>
    );
  }
  const last = m.last;
  const gap = last.bot - last.spy;
  const board = q.data;
  const fetched = board?.fetchedAt ? `${fmtMonDay(board.fetchedAt)} ${fmtClockET(board.fetchedAt).slice(0, 5)} ET` : "—";
  const rates = m.rates ? `${Math.round(m.rates.shortTerm * 100)} % short-term / ${Math.round(m.rates.longTerm * 100)} % long-term` : null;
  return (
    <div className="vt-perf" data-state="ready" data-basis={m.afterTax ? "after-tax" : "pre-tax"}>
      <div className="vt-perf-figs">
        <div className="vt-perf-fig" data-fig="bot"><span className="k" title={m.afterTax ? "portfolio, after tax" : "portfolio, before tax"}><i className="vt-perf-sw vt-perf-botc" />Portfolio</span><span className={`v ${tone(last.bot)}`}>{fmtPct(last.bot)}</span><span className="s">{m.afterTax ? `before tax ${fmtPct(last.botPre)}` : `since ${dayWords(m.from)}`}</span></div>
        <div className="vt-perf-fig" data-fig="spy"><span className="k" title={m.afterTax ? "SPY, after tax" : "SPY, before tax"}><i className="vt-perf-sw vt-perf-spyc" />SPY</span><span className={`v ${tone(last.spy)}`}>{fmtPct(last.spy)}</span><span className="s">{m.afterTax ? `before tax ${fmtPct(last.spyPre)}` : "buy and hold"}</span></div>
        <div className="vt-perf-fig" data-fig="gap"><span className="k">vs SPY</span><span className={`v ${tone(gap)}`}>{fmtSigned(gap, 2, " pt")}</span><span className="s">{m.soldNow ? <>sold now {fmtSigned(m.soldNow.gap, 2, " pt")}<Help k="soldNow" /></> : `max drawdown ${fmtPct(m.maxDd?.bot ?? null, 1)}`}</span></div>
      </div>
      <p className="vt-perf-say">{perfHeadline(m)}</p>
      {m.note ? <div className="vt-perf-note vt-bot-o" role="status">{m.note}</div> : null}
      <PerfSvg m={m} fill={fill} />
      <div className="vt-perf-legend">
        <span><i className="vt-perf-sw vt-perf-botc" />portfolio</span>
        <span><i className="vt-perf-sw vt-perf-spyc" />SPY</span>
        {m.afterTax ? <span><i className="vt-perf-sw vt-perf-dash" />before tax</span> : null}
        {m.soldNow ? <span><i className="vt-perf-sw vt-perf-ring" />sold now<Help k="soldNow" /></span> : null}
        <span><i className="vt-perf-sw vt-perf-ddsw" />drawdown {fmtPct(m.maxDd?.bot ?? null, 1)} · SPY {fmtPct(m.maxDd?.spy ?? null, 1)}<Help k="drawdown" /></span>
      </div>
      <div className="vt-perf-src">
        <span>Portfolio: {board?.sources?.equity ?? "the bot's daily equity snapshot"} · SPY: {board?.sources?.spy ?? "SPY daily closes"} · {sessionCount(m)} trading days since {dayWords(m.from)} through {dayWords(m.to)} · fetched {fetched}</span>
        {m.afterTax ? <span>After tax: an estimate{rates ? ` at ${rates}` : ""}, both sold at each close; the portfolio's open positions are taxed only in the "sold now" ring — the paper account pays no tax<Help k="perfChart" /></span> : null}
        {board?.cashFlows && (board.cashFlows.journals || board.cashFlows.transfers) ? <span className="vt-bot-o">deposits / transfers since the start are inside the portfolio's line (not adjusted)</span> : null}
      </div>
    </div>
  );
}
/** trading days drawn after the start point */
const sessionCount = (m: PerfModel) => Math.max(0, m.points.length - 1);

// ── the pick cards ──────────────────────────────────────────────────────────
function Line({ k, help, children }: { k: string; help?: GlossaryKey; children: ReactNode }) {
  return <div className="vt-pick-ln"><span className="k">{k}{help ? <Help k={help} /> : null}</span><div className="v">{children}</div></div>;
}
function PickCardView({ c, open, onToggle, nowMs }: { c: PickCard; open: boolean; onToggle: () => void; nowMs: number }) {
  const one = c.legs.length === 1 ? c.legs[0] : null;
  const isOptLeg = (p: BotPosition) => !!p.isOption || (p.asset_class ?? "").toLowerCase().includes("option");
  return (
    <article className="vt-pick" data-source={c.source} data-open={open ? "true" : undefined} data-key={c.key}>
      <button type="button" className="vt-pick-hd" aria-expanded={open} onClick={onToggle}>
        <span className="sym">{c.title}</span>
        <span className="sub">{c.sub}</span>
        <span className={`pl ${tone(c.pnlPct ?? c.pnlUsd)}`}>{c.pnlPct !== null ? fmtPct(c.pnlPct, 1) : "net"}<span className="usd"> {c.pnlUsd === null ? "—" : signedMoney(c.pnlUsd)}</span></span>
        <span className="chev" aria-hidden="true">{open ? "▾" : "▸"}</span>
      </button>
      <p className="vt-pick-why" data-source={c.source}>{c.why}</p>
      {open ? (
        <div className="vt-pick-more">
          <Line k="What it is" help="pickWhat">{c.what}</Line>
          {c.chosen ? <Line k="Chose">{c.chosen}</Line> : null}
          <Line k="Passed over" help="pickPassedOver">{c.rejected.length ? <ul>{c.rejected.map((r, i) => <li key={i}>{r.instrument ? <b>{r.instrument}</b> : null}{r.instrument && r.reason ? " — " : ""}{r.reason}</li>)}</ul> : <span className="vt-bot-t3">{c.source === "journal" ? "no alternative recorded for this decision" : "not recorded"}</span>}</Line>
          {c.stages.length ? <Line k="How it decided" help="pickStages"><ul>{c.stages.map((s) => <li key={s.key}><b>{s.label}</b> — {s.text}</li>)}</ul></Line> : null}
          {c.sizing ? <Line k="Size">{c.sizing}</Line> : null}
          {c.later.length ? <Line k="Since then"><ul>{c.later.map((l, i) => <li key={i}><b>{l.action.replace(/_/g, " ")}</b> {fmtMonDay(l.ts)} — {l.text}</li>)}</ul></Line> : null}
          <Line k="Now">{c.legs.map((p) => <div key={p.ticker} className="vt-pick-leg"><span>{c.legs.length > 1 ? `${fmtOcc(p.ticker)}: ` : ""}{isOptLeg(p) ? `${Number(p.qty) < 0 ? "sold" : "bought"} at $${Number.isFinite(p.entryPrice) ? p.entryPrice.toFixed(2) : "—"}, now $${Number.isFinite(p.currentPrice) ? p.currentPrice.toFixed(2) : "—"}` : `entry $${Number.isFinite(p.entryPrice) ? p.entryPrice.toFixed(2) : "—"}, now $${Number.isFinite(p.currentPrice) ? p.currentPrice.toFixed(2) : "—"} · value ${fmtMoney(p.marketValue)}`}</span><span className={tone(p.pnl)}> · {fmtPct(p.pnlPct, 1)} · {signedMoney(Number(p.pnl) || 0)}</span></div>)}</Line>
          <Line k="Exit plan">{c.exit.map((s, i) => <span key={i}>{i ? " · " : ""}{s.hit ? <b className="hit">{s.text}</b> : s.text}</span>)}</Line>
          <div className="vt-pick-src">{c.source === "journal" && c.decidedAt ? `decision journal${c.backfilled ? " (reconstructed from the broker's fills — no reason was recorded at the time)" : ""} · decided ${fmtMonDay(c.decidedAt)} ${fmtClockET(c.decidedAt).slice(0, 5)} ET (${fmtAge(nowMs - Date.parse(c.decidedAt))} ago)` : "no decision-journal row for this position · the lines above are its own fields and the engine's exit rules"}{one ? ` · ${one.ticker}` : ""}</div>
        </div>
      ) : null}
    </article>
  );
}
function MiniList({ title, rows, nowMs, kind }: { title: string; rows: MiniPick[]; nowMs: number; kind: "closed" | "passed" }) {
  if (!rows.length) return null;
  return (
    <div className="vt-pick-mini" data-kind={kind}>
      <div className="vt-pick-sec">{title}</div>
      {rows.map((r) => <div key={r.key} className="vt-pick-mrow"><b>{r.ticker}{r.pnlPct !== null ? <span className={tone(r.pnlPct)}> {fmtPct(r.pnlPct, 1)}</span> : null}</b><span className="t">{fmtAge(nowMs - Date.parse(r.ts))} ago</span><span className="x">{r.text}</span></div>)}
    </div>
  );
}
export function PicksBody({ positions, decisionsQ, nowMs }: { positions: BotPosition[] | undefined; decisionsQ: UseQueryResult<unknown>; nowMs: number }) {
  const parsed = useMemo(() => parseDecisions(decisionsQ.data, decisionsQ.isError ? decisionsQ.error : undefined), [decisionsQ.data, decisionsQ.isError, decisionsQ.error]);
  const minute = Math.floor(nowMs / 60_000);
  const cards = useMemo(() => pickCards(positions, parsed.rows, minute * 60_000), [positions, parsed.rows, minute]);
  const closed = useMemo(() => closedPicks(parsed.rows, positions, minute * 60_000), [parsed.rows, positions, minute]);
  const passed = useMemo(() => passedOn(parsed.rows, positions, minute * 60_000), [parsed.rows, positions, minute]);
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const toggle = (k: string) => setOpen((s) => { const n = new Set(s); if (n.has(k)) n.delete(k); else n.add(k); return n; });
  if (!positions) return <div className="vt-bot-empty"><NotRecorded what="positions not loaded" /></div>;
  const journal = decisionsQ.isLoading ? "decision journal loading…" : parsed.ok ? `decision journal · ${parsed.rows.length} decision${parsed.rows.length === 1 ? "" : "s"} · ${asOfWords(null, decisionsQ.dataUpdatedAt, nowMs)}` : parsed.reason ?? "decision journal not loaded";
  const explained = cards.filter((c) => c.source === "journal").length;
  return (
    <div className="vt-picks" data-journal={parsed.ok ? "ok" : "missing"}>
      <div className="vt-picks-hd">
        <span className={parsed.ok || decisionsQ.isLoading ? "vt-bot-t3" : "vt-bot-o"}>{journal}</span>
        {cards.length ? <span className="vt-bot-t3">{explained} of {cards.length} with a recorded reason</span> : null}
        {cards.length ? <button type="button" className="vt-bot-wbtn" onClick={() => setOpen((s) => (s.size >= cards.length ? new Set() : new Set(cards.map((c) => c.key))))}>{open.size >= cards.length ? "collapse all" : "expand all"}</button> : null}
      </div>
      {cards.length ? cards.map((c) => <PickCardView key={c.key} c={c} open={open.has(c.key)} onToggle={() => toggle(c.key)} nowMs={nowMs} />) : <div className="vt-bot-empty"><St word="flat · no positions" /></div>}
      <MiniList title={`Recently closed · last ${CLOSED_PICK_DAYS} days`} rows={closed} nowMs={nowMs} kind="closed" />
      <MiniList title={`Passed on · last ${PASSED_ON_HOURS} h`} rows={passed} nowMs={nowMs} kind="passed" />
    </div>
  );
}

// ── why are we losing money ─────────────────────────────────────────────────
function AttrGroupView({ g, max }: { g: AttrGroup; max: number }) {
  return (
    <div className="vt-why-g" data-group={g.key}>
      <div className="vt-why-gt">{g.title}</div>
      {g.rows.slice(0, 6).map((r, i) => (
        <div key={`${r.label}-${i}`} className="vt-why-row" data-sign={r.pnl < 0 ? "neg" : r.pnl > 0 ? "pos" : "zero"}>
          <span className="l">{r.label}</span>
          <span className="b" aria-hidden="true"><i style={{ width: `${Math.max(2, Math.min(50, (Math.abs(r.pnl) / max) * 50))}%`, ...(r.pnl < 0 ? { right: "50%" } : { left: "50%" }) }} /></span>
          <span className={`p ${tone(r.pnl)}`}>{signedMoney(r.pnl)}</span>
          <span className="n">{r.n !== null ? `${r.n}×` : ""}</span>
          {r.reason ? <span className="r">{r.reason}</span> : null}
        </div>
      ))}
      {g.rows.length > 6 ? <div className="vt-bot-t3 vt-why-more">+{g.rows.length - 6} more on the route</div> : null}
    </div>
  );
}
function Sect({ id, title, help, summary, cls, children, defaultOpen = false }: { id: string; title: string; help: GlossaryKey; summary: string; cls?: string; children?: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="vt-why-sec" data-sec={id} data-open={open ? "true" : undefined}>
      <div className="vt-why-sh">
        <button type="button" aria-expanded={open} disabled={!children} onClick={() => setOpen((o) => !o)}><span className="t">{title}</span><span className="chev" aria-hidden="true">{children ? (open ? "▾" : "▸") : ""}</span></button>
        <Help k={help} />
      </div>
      <p className={`vt-why-ss ${cls ?? ""}`}>{summary}</p>
      {open && children ? <div className="vt-why-sb">{children}</div> : null}
    </section>
  );
}
/** A row of small bars for a short series (the flow path per session, the risk budget per day): height = |value|
 *  against the largest, colour = sign or band; each bar carries its date and value as its title. */
function MiniBars({ rows, kind }: { rows: Array<{ label: string; value: number; tone: "pos" | "neg" | "zero" }>; kind: string }) {
  const max = Math.max(1e-9, ...rows.map((r) => Math.abs(r.value)));
  return <div className="vt-why-bars" data-kind={kind} role="img" aria-label={rows.map((r) => `${r.label} ${r.value.toFixed(2)}`).join(", ")}>{rows.map((r, i) => <i key={i} data-tone={r.tone} title={`${r.label} · ${r.value.toFixed(2)}`} style={{ height: `${Math.max(8, (Math.abs(r.value) / max) * 100)}%` }} />)}</div>;
}
export function WhyBody({ attrQ, days, onDays, sleeves, flowsQ, taxQ, signalsQ, regimeQ, nowMs }: {
  attrQ: UseQueryResult<unknown>; days: number; onDays: (d: number) => void; sleeves: Sleeves | undefined;
  flowsQ: UseQueryResult<unknown>; taxQ: UseQueryResult<unknown>; signalsQ: UseQueryResult<unknown>;
  /** /api/bot/regime: its risk_budget block (the pipeline's first stage), when the server has it */
  regimeQ: UseQueryResult<unknown>; nowMs: number;
}) {
  const journal = useMemo(() => parseAttribution(attrQ.data, days), [attrQ.data, days]);
  const fallback = useMemo(() => (journal ? null : attributionFromSleeves(sleeves)), [journal, sleeves]);
  const view: AttributionView | null = journal ?? (attrQ.isError || attrQ.isFetched ? fallback : null);
  const flows = useMemo(() => parseFlows(flowsQ.data), [flowsQ.data]);
  const tax = useMemo(() => parseTax(taxQ.data), [taxQ.data]);
  const sig = useMemo(() => parseSignals(signalsQ.data), [signalsQ.data]);
  const risk = useMemo(() => parseRiskBudget(regimeQ.data), [regimeQ.data]);
  const max = view ? Math.max(1, ...view.groups.flatMap((g) => g.rows.map((r) => Math.abs(r.pnl)))) : 1;
  const why = attrQ.isLoading && !view ? (
    <div className="vt-why-top" aria-busy="true"><i className="vt-score-skel" style={{ width: "90%", height: 14 }} /><i className="vt-score-skel" style={{ width: "70%", height: 14 }} /></div>
  ) : !view ? (
    <div className="vt-bot-empty" role="status"><St word="no attribution yet" cls="vt-bot-o" /><span>{unavailableWords(attrQ.isError ? attrQ.error : null, "loss attribution (the decision journal)")}{sleeves?.pending ? " · the fill ledger is still computing — the page asks again shortly" : ""}</span></div>
  ) : (
    <>
      <div className="vt-why-top">
        <p className="vt-why-say">{whyHeadline(view)}</p>
        {view.source === "journal" ? (
          <div className="vt-bot-seg vt-why-win" role="group" aria-label="Attribution window">{ATTRIBUTION_WINDOWS.map((d) => <button key={d} type="button" aria-pressed={d === days} onClick={() => onDays(d)}>{d}d</button>)}</div>
        ) : null}
      </div>
      <div className="vt-why-groups">{view.groups.map((g) => <AttrGroupView key={g.key} g={g} max={max} />)}</div>
      <div className="vt-why-src">
        <span>{view.source === "journal" ? `decision journal attribution · ${view.windowLabel} · ${asOfWords(view.asOf, attrQ.dataUpdatedAt, nowMs)}` : `fill ledger · ${view.windowLabel} · ${asOfWords(view.asOf, 0, nowMs)} · ${unavailableWords(attrQ.isError ? attrQ.error : null, "the journal's attribution")}`}<Help k="why" /></span>
        {view.notes.map((t, i) => <span key={i}>{t}</span>)}
      </div>
    </>
  );
  return (
    <div className="vt-why" data-source={view?.source ?? "none"}>
      {why}
      <Sect id="regime" title="Market regime · risk budget" help="regimeBudget" summary={risk ? riskSentence(risk) : unavailableWords(regimeQ.isError ? regimeQ.error : null, "the regime risk budget")} cls={risk ? (risk.stale ? "vt-bot-o" : undefined) : "vt-bot-t3"}>
        {risk ? (
          <>
            {risk.band ? <div className="vt-why-chips"><span data-band={risk.band}>{risk.band}{risk.budget !== null ? ` · ${risk.budget.toFixed(2)}` : ""}</span>{risk.enabled === false ? <span>switched off</span> : risk.applied === false ? <span>not applied</span> : null}</div> : null}
            {risk.drivers.length ? <ul>{risk.drivers.map((d, i) => <li key={i}><b>{d.label}</b>{d.cut !== null && d.cut !== 0 ? ` ${fmtSigned(d.cut, 2)}` : ""}{d.reason ? ` — ${d.reason}` : ""}</li>)}</ul> : null}
            {risk.missing.length ? <div className="vt-bot-o">inputs missing: {risk.missing.join(", ")}</div> : null}
            {risk.history.length > 1 ? <MiniBars kind="budget" rows={risk.history.slice(-30).map((h) => ({ label: dayWords(h.asOf), value: h.budget, tone: h.band === "full" ? "pos" : h.band === "reduced" ? "zero" : "neg" }))} /> : null}
            <div className="vt-why-src"><span>regime engine · as of {dayWords(risk.asOf)}{risk.ageMarketDays !== null ? ` (${risk.ageMarketDays} market day${risk.ageMarketDays === 1 ? "" : "s"} old)` : ""} · {asOfWords(null, regimeQ.dataUpdatedAt, nowMs)}</span>{risk.note ? <span>{risk.note}</span> : null}</div>
          </>
        ) : null}
      </Sect>
      <Sect id="flows" title="Money flows ahead" help="flows" summary={flows ? flowsSentence(flows) : unavailableWords(flowsQ.isError ? flowsQ.error : null, "the institutional flow model")} cls={flows ? (flows.stale || !flows.computed ? "vt-bot-o" : undefined) : "vt-bot-t3"}>
        {flows && flows.computed ? (
          <>
            {flows.floor ? <div><b>Floor ×{flows.floor.multiplier.toFixed(2)}</b> {flows.floor.applied ? "applied" : "computed, not applied"}{flows.floor.reason ? ` — ${flows.floor.reason}` : ""}</div> : null}
            {flows.sessions.length > 1 ? <MiniBars kind="flow" rows={flows.sessions.map((x) => ({ label: dayWords(x.date), value: x.pressure, tone: x.pressure > 0 ? "pos" : x.pressure < 0 ? "neg" : "zero" }))} /> : null}
            {flows.equity?.drivers.length ? <ul>{flows.equity.drivers.map((d, i) => <li key={i}><b>{d.name}</b>{d.value ? ` ${d.value}` : ""}{d.reason ? ` — ${d.reason}` : ""}</li>)}</ul> : null}
            {flows.sectors.length ? <div className="vt-why-chips">{flows.sectors.map((s) => <span key={s.sector} data-dir={s.direction} title={s.reason ?? undefined}>{s.etf ?? s.sector} {s.direction === "buy" ? "in" : s.direction === "sell" ? "out" : "flat"}{s.pressure !== null ? ` ${fmtSigned(s.pressure, 2)}` : ""}</span>)}</div> : null}
            {flows.calendar.length ? <ul>{flows.calendar.slice(0, 4).map((c, i) => <li key={i}><b>{dayWords(c.date)}</b> {c.event}{c.effect ? ` — ${c.effect}` : ""}</li>)}</ul> : null}
            <div className="vt-why-src"><span>{flows.label ? `${flows.label} · ` : ""}flow model (an estimate, not a measured flow) · {asOfWords(flows.asOf, flowsQ.dataUpdatedAt, nowMs)}{flows.ageDays !== null ? ` · ${flows.ageDays} day${flows.ageDays === 1 ? "" : "s"} old` : ""}</span>{flows.notes.map((t, i) => <span key={i} className={/^Stale/.test(t) ? "vt-bot-o" : undefined}>{t}</span>)}</div>
          </>
        ) : null}
      </Sect>
      <Sect id="tax" title="Tax" help="taxFilter" summary={tax ? taxSentence(tax, nowMs) : unavailableWords(taxQ.isError ? taxQ.error : null, "the tax filter")} cls={tax ? undefined : "vt-bot-t3"}>
        {tax ? (
          <>
            {tax.decisions.length ? <ul>{tax.decisions.slice(0, 5).map((d, i) => <li key={i}><b>{d.ticker}</b> {tax.enforced === false ? (d.wouldAllow === false ? "would skip" : "would allow") : d.allowed === false ? "skipped" : d.allowed ? "allowed" : "checked"}{d.afterTaxPct !== null && d.spyAfterTaxPct !== null ? ` · ${fmtPct(d.afterTaxPct, 2)} after tax vs SPY ${fmtPct(d.spyAfterTaxPct, 2)}` : ""}{d.reason ? ` — ${d.reason}` : ""}</li>)}</ul> : null}
            {tax.harvest.length ? <><div className="vt-pick-sec">Losses to harvest</div><ul>{tax.harvest.map((hv, i) => <li key={i}><b>{hv.ticker}</b>{hv.unrealized !== null ? ` ${signedMoney(hv.unrealized)}` : ""}{hv.replacement ? ` → ${hv.replacement}` : ""}{hv.washSaleUntil ? ` · wash-sale window to ${dayWords(hv.washSaleUntil)}` : ""}{hv.reason ? ` — ${hv.reason}` : ""}</li>)}</ul></> : null}
            {tax.ltApproaching.length ? <><div className="vt-pick-sec">Nearing long-term</div><ul>{tax.ltApproaching.map((l, i) => <li key={i}><b>{l.ticker}</b>{l.daysToLongTerm !== null ? ` in ${l.daysToLongTerm} days` : ""}{l.gain !== null ? ` · gain ${signedMoney(l.gain)}` : ""}{l.reason ? ` — ${l.reason}` : ""}</li>)}</ul></> : null}
            <div className="vt-why-src"><span>tax filter · an estimate at {tax.rates ? `${Math.round(tax.rates.shortTerm * 100)} / ${Math.round(tax.rates.longTerm * 100)} %` : "the server's rates"} · {asOfWords(null, taxQ.dataUpdatedAt, nowMs)} · the paper account pays no tax</span>{tax.notes.map((t, i) => <span key={i}>{t}</span>)}</div>
          </>
        ) : null}
      </Sect>
      <Sect id="signals" title="Which signals work" help="signalsWork" summary={sig ? signalsSentence(sig) : unavailableWords(signalsQ.isError ? signalsQ.error : null, "the signal audit")} cls={sig && sig.kind === "report" ? undefined : "vt-bot-t3"}>
        {sig && sig.kind === "report" ? (
          <>
            <div className="vt-why-sig">{sig.rows.map((r) => <div key={r.name} className="vt-why-sigr" data-verdict={r.verdict}><b>{r.name}</b><span className="vd">{r.verdict === "works" ? "adds return" : r.verdict === "noise" ? "noise" : r.verdict === "hurts" ? "hurts" : "unclear"}</span><span className="mt">{r.metric !== null ? `${r.metricLabel} ${Math.abs(r.metric) < 10 ? r.metric.toFixed(3) : Math.round(r.metric).toLocaleString("en-US")}` : ""}{r.n !== null ? ` · n ${r.n.toLocaleString("en-US")}` : ""}</span>{r.reason ? <span className="r">{r.reason}</span> : null}</div>)}</div>
            <div className="vt-why-src"><span>signal audit · {asOfWords(sig.asOf, signalsQ.dataUpdatedAt, nowMs)}</span>{sig.notes.map((t, i) => <span key={i}>{t}</span>)}</div>
          </>
        ) : null}
      </Sect>
    </div>
  );
}
