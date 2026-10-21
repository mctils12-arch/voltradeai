// Bot page v2 — the panels. Every number comes from a real field; absence is
// drawn as an outlined dot + "not recorded", never as a zero.
import { useCallback, useMemo, useState, type ReactNode } from "react";
import { sanitizePosition } from "@/lib/tradeChartGuards";
import { BotChart } from "./chart/BotChart";
import { DEFAULT_TIMEFRAME, timeframeSpec, type Timeframe } from "./chart/scales";
import { underlyingOf, overlayFor, candidateOverlay, EMPTY_OVERLAY, type Overlay } from "./chart/levels";
import type { Theme } from "./chart/draw";
import { candidateKey, FACTOR_KEYS, FACTOR_SHORT, type WatchModel, type WatchCandidate } from "./watch";
import { Help } from "./help";
import type { GlossaryKey } from "./glossary";
import { fmtAge, fmtPct, fmtMoney, fmtCompact, fmtCountdown, fmtClockET, fmtOcc, fmtSigned, spyTodayPct, edgeStats, MINUS } from "./format";
import type { StageRow, Issue, DayReview } from "./derive";
import type { DerivedState, AuditRow } from "./pageState";
import type { Insight, BotPosition, OpenOrder, LastScan, MarketStatus, BotAccount, BotStatus, Performance, Calendar, Benchmark, Sleeves } from "./useBotData";
import { num } from "./useBotData";
import type { PanelId } from "./zoneModel";

// ── chrome ──────────────────────────────────────────────────────────────────
const I = {
  hold: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true"><path d="M8 5v14M16 5v14" /></svg>,
  play: <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true"><path d="M7 4l12 8-12 8z" /></svg>,
  pin: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M12 17v5M8 3h8l-1 7 3 3H6l3-3z" /></svg>,
  win: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><rect x="3" y="8" width="13" height="13" rx="1.5" /><path d="M8 8V3h13v13h-5" /></svg>,
  zoom: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M4 10V4h6M20 14v6h-6M20 4l-7 7M4 20l7-7" /></svg>,
  dock: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M4 14v6h6M20 10V4h-6M4 20l7-7M20 4l-7 7" /></svg>,
  close: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>,
  pop: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M11 5H4v15h15v-7" /></svg>,
  arrow: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>,
};
export const ICONS = I;

export function St({ word, cls = "vt-bot-t3", out = false }: { word: string; cls?: string; out?: boolean }) {
  return <span className={`vt-bot-st ${cls}`}><span className="vt-bot-dot" data-out={out ? "true" : undefined} />{word}</span>;
}
export const NotRecorded = ({ what = "not recorded" }: { what?: string }) => <St word={what} out />;

export interface Chrome {
  id: PanelId; held: boolean; pinned: boolean; heldSince: number | null; ticksSinceHold: number; nowMs: number;
  onHold: (id: PanelId) => void; onPin: (id: PanelId) => void; onFloat: (id: PanelId) => void;
  /** open this panel in its own browser window (popout.ts) */
  onPop: (id: PanelId) => void;
  /** this panel IS its own browser window: nothing to pin to or float over */
  pane: boolean;
  /** the pop-out control is offered (not on touch / the phone tier) */
  canPop: boolean;
}
export function Panel({ chrome, title, help, status, meta, ageMs, children, scroll, style }: {
  chrome: Chrome; title: string; help?: GlossaryKey; status?: ReactNode; meta?: ReactNode; ageMs: number | null; children: ReactNode; scroll?: boolean; style?: React.CSSProperties;
}) {
  return (
    <section className="vt-bot-panel" data-panel={chrome.id} data-held={chrome.held ? "true" : undefined} data-pinned={chrome.pinned ? "true" : undefined} style={style}>
      <div className="vt-bot-ph">
        <span className="vt-bot-ti">{title}{help ? <Help k={help} /> : null}</span>
        {status}
        {meta ? <span className="vt-bot-meta">{meta}</span> : <span className="vt-bot-sp" />}
        <span className="vt-bot-age">{fmtAge(ageMs)}</span>
        <div className="vt-bot-ctl">
          <button type="button" aria-label={chrome.held ? "Resume this panel" : "Hold this panel"} aria-pressed={chrome.held} onClick={() => chrome.onHold(chrome.id)}>{chrome.held ? I.play : I.hold}</button>
          {chrome.pane ? (
            // a pane's one way back is closing its window; the page restores the slot
            <button type="button" aria-label="Back to the page" title="Close this window — the panel returns to its slot on the page" onClick={() => window.close()}>{I.dock}</button>
          ) : (
            <>
              <button type="button" aria-label="Pin this panel" aria-pressed={chrome.pinned} onClick={() => chrome.onPin(chrome.id)}>{I.pin}</button>
              <button type="button" aria-label="Open as a window" onClick={() => chrome.onFloat(chrome.id)}>{I.win}</button>
              {chrome.canPop ? <button type="button" aria-label="Open in its own browser window" title="Its own browser window — drag it to another monitor" onClick={() => chrome.onPop(chrome.id)}>{I.pop}</button> : null}
            </>
          )}
        </div>
      </div>
      <div className="vt-bot-pb" data-scroll={scroll ? "true" : undefined}>{children}</div>
      {chrome.held ? <div className="vt-bot-held"><span>held {fmtClockET(chrome.heldSince)} ET · live continues · +{chrome.ticksSinceHold}</span></div> : null}
    </section>
  );
}

// ── KPI strip ───────────────────────────────────────────────────────────────
export function KpiStrip({ perf, account, status, market, calendar, bench, now, state, hold, onHold }: {
  perf: Performance | undefined; account: BotAccount | undefined; status: BotStatus | undefined; market: MarketStatus | undefined; calendar: Calendar | undefined; bench: Benchmark | undefined;
  now: number; state: DerivedState; hold: boolean; onHold: () => void;
}) {
  const edge = edgeStats(perf?.winRate, perf?.avgGain, perf?.avgLoss);
  const n = num(perf?.totalTrades);
  const equity = num(account?.equity);
  const peak = num(status?.equityPeak) ?? num(account?.equityPeak);
  const maxDd = num(status?.maxDrawdownPct);
  const dd = equity !== null && peak ? (equity / peak - 1) * 100 : null;
  const killLine = peak !== null && maxDd !== null ? peak * (1 - maxDd / 100) : null;
  const toLine = dd !== null && maxDd !== null ? dd + maxDd : null;
  const curve = [...(perf?.equityCurve ?? [])].filter((p) => Number.isFinite(num(p.value) ?? NaN)).sort((a, b) => a.date.localeCompare(b.date));
  const vals = curve.map((p) => Number(p.value));
  const lo = vals.length ? Math.min(...vals, killLine ?? Infinity) : 0;
  const hi = vals.length ? Math.max(...vals) : 1;
  const pts = vals.map((v, k) => `${(k / Math.max(1, vals.length - 1)) * 110},${34 - ((v - lo) / Math.max(1e-9, hi - lo)) * 30}`).join(" ");
  const ky = killLine !== null && vals.length ? 34 - ((killLine - lo) / Math.max(1e-9, hi - lo)) * 30 : null;
  const open = market?.isOpen ?? null;
  const nextMs = market ? Date.parse(open ? market.nextClose : market.nextOpen) : NaN;
  const todayKey = new Date(now).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const early = calendar?.earlyCloses?.find((e) => e.date === todayKey);
  const stateWord = (() => {
    switch (state.state) {
      case "scanning": return `scanning · ${state.scan.elapsedMs !== null ? Math.round(state.scan.elapsedMs / 1000) : "?"} s`;
      case "candidate": return `${state.scan.candidates ?? ""} candidates`.trim();
      case "placed": return "order working";
      case "exit": return "exiting";
      case "incident": return `halted · ${state.incident?.kind ?? ""}`;
      case "closed": return "closed · review";
      default: return state.scan.nextAtMs ? `idle · next scan ${fmtCountdown(state.scan.nextAtMs - now)}` : "idle";
    }
  })();
  const stateCls = state.state === "incident" ? "vt-bot-r" : state.state === "scanning" ? "vt-bot-b" : state.state === "candidate" || state.state === "placed" || state.state === "exit" ? "vt-bot-t1" : "vt-bot-t2";
  return (
    <div className="vt-bot-kpi vt-bot-area-kpi">
      <div className="vt-bot-panel">
        <div style={{ minWidth: 0 }}>
          <div className="vt-bot-eb">Expectancy / trade · net<Help k="expectancy" /></div>
          {edge ? <div className={`vt-bot-hero ${edge.expectancy < 0 ? "vt-bot-r" : edge.expectancy > 0 ? "vt-bot-g" : ""}`}>{fmtPct(edge.expectancy)}</div> : <div className="vt-bot-hero vt-bot-t3">—</div>}
        </div>
        <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11, lineHeight: 1.35 }}>n {n ?? "?"}<br />{n !== null && n < 30 ? "wide band" : "trigger-price pnl"}</div>
      </div>
      <div className="vt-bot-panel">
        <div style={{ minWidth: 0 }}>
          <div className="vt-bot-eb" style={{ whiteSpace: "nowrap" }}>Win rate · breakeven<Help k="winRate" /></div>
          {edge ? <div className="vt-bot-kpiv"><span className={edge.winRate < edge.breakevenWr ? "vt-bot-r" : "vt-bot-g"}>{(edge.winRate * 100).toFixed(1)}%</span> <span className="vt-bot-t3">/ {(edge.breakevenWr * 100).toFixed(1)}%</span></div> : <div className="vt-bot-kpiv vt-bot-t3">—</div>}
          <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>{edge?.payoff ? `payoff ${edge.payoff.toFixed(2)}` : "payoff —"}{num(perf?.profitFactor) !== null ? ` · pf ${Number(perf?.profitFactor).toFixed(2)}` : ""}</div>
        </div>
        {edge ? <div className="vt-bot-wrbar"><div className="vt-bot-bar"><i style={{ width: `${Math.min(100, edge.winRate * 100)}%`, background: "var(--accent-red)" }} /><span className="vt-bot-tick" style={{ left: `${Math.min(100, edge.breakevenWr * 100)}%` }} /></div></div> : null}
      </div>
      <div className="vt-bot-panel">
        <div style={{ minWidth: 0 }}>
          <div className="vt-bot-eb">vs holding SPY<Help k="vsSpy" /></div>
          {bench?.since ? (
            <>
              <div className="vt-bot-kpiv" title={`account return minus SPY total return (dividend-adjusted closes through ${bench.barsThrough ?? "?"}), ${bench.since.from} to ${bench.since.to}`}>
                <span className={bench.since.gapPp < 0 ? "vt-bot-r" : "vt-bot-g"}>{fmtSigned(bench.since.gapPp, 1, " pt")}</span>
              </div>
              <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11, lineHeight: 1.35 }}>
                SPY {fmtPct(bench.since.spyPct, 1)} · acct {fmtPct(bench.since.accountPct, 1)}<br />
                since {bench.since.from.slice(5)} · {bench.d30 ? `${bench.d30.days}d ${fmtSigned(bench.d30.gapPp, 1)}` : "30d —"}
              </div>
            </>
          ) : (
            <>
              <div className="vt-bot-kpiv vt-bot-t3">—</div>
              <div style={{ fontSize: 11 }}><NotRecorded what={bench?.error ? "SPY series unavailable" : "not recorded · no SPY series"} /></div>
            </>
          )}
        </div>
      </div>
      <div className="vt-bot-panel">
        <div style={{ minWidth: 0 }}>
          <div className="vt-bot-eb">Equity · from peak<Help k="equityPeak" /></div>
          <div className="vt-bot-kpiv">{fmtMoney(equity)} {dd !== null ? <span className={`${dd < 0 ? "vt-bot-r" : "vt-bot-g"}`} style={{ fontSize: 14 }}>{fmtPct(dd)}</span> : null}</div>
          <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>peak {fmtMoney(peak)} · kill {maxDd !== null ? `${MINUS}${maxDd}%` : "—"}{toLine !== null ? ` · ${toLine.toFixed(2)} pt to line` : ""}</div>
        </div>
        {vals.length > 1 ? (
          <svg width="110" height="36" viewBox="0 0 110 36" role="img" aria-label={`Equity curve, ${vals.length} days, kill line at ${fmtMoney(killLine)}`}>
            {ky !== null ? <line x1="0" y1={ky} x2="110" y2={ky} stroke="var(--text-tertiary)" strokeDasharray="2 2" strokeWidth="1" /> : null}
            <polyline fill="none" stroke="var(--text-secondary)" strokeWidth="1.5" points={pts} />
          </svg>
        ) : <span className="vt-bot-t3" style={{ fontSize: 11 }}><NotRecorded what="no curve" /></span>}
      </div>
      <div className="vt-bot-panel">
        <div style={{ minWidth: 0 }}>
          <div className="vt-bot-eb">Session<Help k="session" /></div>
          <div className="vt-bot-m vt-bot-t1" style={{ fontSize: 14, fontWeight: 500, lineHeight: 1.2 }}>
            {open === null ? "—" : open ? `RTH · closes in ${fmtCountdown(nextMs - now)}` : `closed · opens in ${fmtCountdown(nextMs - now)}`}
          </div>
          <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>{fmtClockET(now)} ET{early ? ` · early close ${early.close}` : ""}</div>
        </div>
      </div>
      <div className="vt-bot-panel">
        <div style={{ minWidth: 0 }}>
          <div className="vt-bot-eb">Now<Help k="now" /></div>
          <div className="vt-bot-row" style={{ gap: 8 }}>
            <span className={`vt-bot-st ${stateCls}`} style={{ fontSize: 15, fontWeight: 600 }}><span className="vt-bot-dot" />{stateWord}</span>
            {hold ? <span className="vt-bot-st vt-bot-t1"><span className="vt-bot-dot" />layout held · bot live</span> : null}
          </div>
        </div>
        <span className="vt-bot-row" style={{ marginLeft: "auto", gap: 2 }}><Help k="hold" /><button type="button" className="vt-bot-wbtn" aria-pressed={hold} onClick={onHold}>hold</button></span>
      </div>
    </div>
  );
}

// ── stages ──────────────────────────────────────────────────────────────────
const LEVEL_CLS: Record<string, string> = { ok: "vt-bot-t3", warn: "vt-bot-o", bad: "vt-bot-r", unk: "vt-bot-t3" };
export function StagesBody({ rows }: { rows: StageRow[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      {rows.map((s) => (
        <div key={s.id} className="vt-bot-row" style={{ height: 23 }}>
          <span className={`vt-bot-dot ${LEVEL_CLS[s.level]}`} data-out={s.level === "unk" ? "true" : undefined} />
          <span className="vt-bot-t1" style={{ width: 60, fontSize: 12, fontWeight: 500 }}>{s.name}</span>
          {s.boxes.length ? <span className="vt-bot-sh" aria-label={`${s.boxes.filter((b) => b === "ok").length} of ${s.boxes.length} expected ticks`}>{s.boxes.map((b, k) => <i key={k} data-s={b} />)}</span> : null}
          <span className="vt-bot-t2 vt-bot-m" style={{ flex: 1, fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", textAlign: "right" }}>{s.value}</span>
        </div>
      ))}
    </div>
  );
}

// ── issues ──────────────────────────────────────────────────────────────────
export type ActionKind = "resume" | "stop" | "kill";
export function IssuesBody({ issues, nowMs, onFocus, onAct, busy }: { issues: Issue[]; nowMs: number; onFocus: (p: PanelId) => void; onAct: (k: ActionKind) => Promise<void>; busy: boolean }) {
  const [confirm, setConfirm] = useState<ActionKind | null>(null);
  const consequence: Record<ActionKind, string> = {
    resume: "clears the kill latch and re-activates the loop; the −10% rule re-latches at the same line",
    stop: "pauses scans and entries; stops keep firing; cleared by the next deploy",
    kill: "cancels every open order and stops all exits until resumed",
  };
  if (!issues.length) return <div className="vt-bot-row" style={{ paddingTop: 8 }}><St word={`normal · checked ${fmtClockET(nowMs)} ET`} /></div>;
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {issues.map((i) => (
        <div key={i.id} className="vt-bot-issue">
          <div className="vt-bot-row"><span className={`vt-bot-dot ${i.level === "L3" ? "vt-bot-r" : i.level === "L2" ? "vt-bot-o" : "vt-bot-t3"}`} /><span className="t">{i.title}</span><span className="vt-bot-age">{i.since ? fmtAge(nowMs - Date.parse(i.since)) : ""}</span></div>
          <div className="l">{i.condition}</div>
          <div className="l"><span className="vt-bot-t3">→</span> {i.consequence}</div>
          <div className="l"><span className="vt-bot-t3">clears</span> {i.clears}</div>
          {i.action ? (
            <div className="l">
              {i.action.kind === "focus" ? <button type="button" onClick={() => i.action?.target && onFocus(i.action.target)}>{i.action.label}</button>
                : confirm === i.action.kind ? (
                  <div className="vt-bot-confirm">
                    <div className="vt-bot-t2">{consequence[i.action.kind]}</div>
                    <div className="vt-bot-row"><button type="button" data-primary="true" disabled={busy} onClick={() => { const k = confirm; setConfirm(null); if (k) void onAct(k); }}>{i.action.label}</button><button type="button" onClick={() => setConfirm(null)}>cancel</button></div>
                  </div>
                ) : <button type="button" onClick={() => setConfirm(i.action!.kind as ActionKind)}>{i.action.label}</button>}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}

// ── state element: scanning ─────────────────────────────────────────────────
const PHASES: Array<[string, number]> = [["universe", 0.9], ["snapshot", 3.8], ["quick", 0.1], ["deep score", 41], ["trade loop", 0.4], ["manage", 0.1], ["options", 0.3], ["3rd leg", 0.1], ["convexity", 0.3], ["floors", 0.3], ["tiers", 0.1]];
export function ScanStripBody({ elapsedMs, lastScan }: { elapsedMs: number | null; lastScan: LastScan | undefined }) {
  const times = Array.isArray(lastScan?.scan_phase_times) ? (lastScan!.scan_phase_times as Array<{ phase?: string; duration_sec?: number }>) : [];
  const phases = times.length ? times.filter((p) => typeof p.phase === "string" && typeof p.duration_sec === "number").map((p) => [String(p.phase).replace(/^step\d+[a-z]?_/, "").replace(/_/g, " ").slice(0, 14), Number(p.duration_sec)] as [string, number]) : PHASES;
  const total = phases.reduce((a, [, s]) => a + s, 0) || 1;
  const el = elapsedMs !== null ? elapsedMs / 1000 : 0;
  let acc = 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div className="vt-bot-row"><span className="vt-bot-m vt-bot-t1" style={{ fontSize: 12 }}>{Math.round(el)} s / ~{Math.round(total)} s</span><span className="vt-bot-sp" /><span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>{times.length ? "phases from the last scan" : "typical phases"}</span></div>
      <div className="vt-bot-scan">{phases.map(([n, s]) => { const start = acc; acc += s; const st = acc <= el ? "done" : start <= el ? "active" : "todo"; return <i key={n} data-s={st} title={`${n} ${s.toFixed(1)} s`} style={{ width: `${Math.max(1.4, (s / total) * 100)}%` }} />; })}</div>
      <div className="vt-bot-row" style={{ gap: 12, flexWrap: "wrap" }}>{phases.filter(([, s]) => s >= 0.3).map(([n, s]) => <span key={n} className="vt-bot-m vt-bot-t3" style={{ fontSize: 10.5 }}>{n} {s.toFixed(1)}s</span>)}</div>
    </div>
  );
}

// ── state element: decision chain ───────────────────────────────────────────
interface TradeLike { ticker?: string; score?: number; side?: string; action_label?: string; reasons?: string[]; rules_score?: number | null; ml_score_raw?: number | null; price?: number; trade_type?: string; instrument_strategy?: string; timestamp?: string }
function asTrade(x: unknown): TradeLike | null {
  if (!x || typeof x !== "object") return null;
  const o = x as Record<string, unknown>;
  return { ticker: typeof o.ticker === "string" ? o.ticker : undefined, score: num(o.score) ?? undefined, side: typeof o.side === "string" ? o.side : undefined, action_label: typeof o.action_label === "string" ? o.action_label : undefined, reasons: Array.isArray(o.reasons) ? o.reasons.map(String) : undefined, rules_score: num(o.rules_score), ml_score_raw: num(o.ml_score_raw), price: num(o.price) ?? undefined, trade_type: typeof o.trade_type === "string" ? o.trade_type : undefined, instrument_strategy: typeof o.instrument_strategy === "string" ? o.instrument_strategy : undefined };
}
function Node({ title, cls, help, children }: { title: string; cls: string; help?: GlossaryKey; children: ReactNode }) {
  return <div className="vt-bot-node"><div className="vt-bot-row" style={{ gap: 6 }}><span className={`vt-bot-dot ${cls}`} /><span className="vt-bot-eb">{title}{help ? <Help k={help} /> : null}</span></div>{children}</div>;
}
export function DecisionBody({ lastScan, audit, nowMs }: { lastScan: LastScan | undefined; audit: AuditRow[]; nowMs: number }) {
  const trades = (lastScan?.new_trades ?? []).map(asTrade).filter((x): x is TradeLike => !!x && !!x.ticker);
  const top = trades[0] ?? (lastScan?.top_10?.[0] ? asTrade(lastScan.top_10[0]) : null);
  if (!top?.ticker) return <div className="vt-bot-empty"><NotRecorded what="no candidate in the last scan result" /></div>;
  const tk = top.ticker.toUpperCase();
  const rowsFor = audit.filter((r) => nowMs - Date.parse(r.time) < 45 * 60_000 && r.message.toUpperCase().includes(tk));
  const gate = rowsFor.filter((r) => /^(SKIP|SIZE-BLOCK|SIZE-CAP|CORR-CAP|HEAT-CAP|HALT-SKIP|DUP-SKIP|OPTIONS-SLOT-FULL|OPTIONS-SKIP|MORNING-|TIER-KILL|TIER-DEFER)/.test(r.type));
  const act = rowsFor.find((r) => /^(QUEUE|TRADE|ETF-TRADE|OPTIONS-TRADE|MORNING-TRADE)$/.test(r.type));
  const rules = top.rules_score, ml = top.ml_score_raw, final = top.score ?? null;
  return (
    <div className="vt-bot-chain">
      <Node title="trigger" cls="vt-bot-t3"><div className="vt-bot-kv"><span className="k">scan</span><span className="v">{lastScan?.timestamp ? fmtClockET(lastScan.timestamp) + " ET" : "—"}</span><span className="k">action</span><span className="v">{top.action_label ?? top.side ?? "—"}</span><span className="k">kind</span><span className="v">{top.trade_type ?? top.instrument_strategy ?? "—"}</span></div></Node>
      <div className="vt-bot-arrow">{I.arrow}</div>
      <Node title="features" cls="vt-bot-t3"><div className="vt-bot-t2" style={{ fontSize: 11.5 }}>{(top.reasons ?? []).slice(0, 3).map((r, k) => <div key={k} style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r}</div>)}</div><div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 10.5 }}>34 features on the shadow record</div></Node>
      <div className="vt-bot-arrow">{I.arrow}</div>
      <Node title="rules · fired / vetoed" cls={gate.length ? "vt-bot-r" : "vt-bot-t3"}>
        {gate.length ? gate.slice(0, 4).map((r, k) => <div key={k} className="vt-bot-row" style={{ gap: 8, height: 15 }}><span className="vt-bot-dot vt-bot-r" /><span className="vt-bot-t2" style={{ width: 70, fontSize: 11 }}>{r.type.toLowerCase()}</span><span className="vt-bot-m vt-bot-t1" style={{ fontSize: 11, flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.message.slice(0, 80)}</span></div>) : <div className="vt-bot-t3" style={{ fontSize: 11.5 }}>no veto row for {tk}</div>}
        <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 10.5 }}>only gates that write a row are shown · trace incomplete</div>
      </Node>
      <div className="vt-bot-arrow">{I.arrow}</div>
      <Node title="conviction" cls="vt-bot-t3" help="conviction">
        <div className="vt-bot-row" style={{ gap: 14 }}>
          <div><div className="vt-bot-eb" style={{ fontSize: 10 }}>rules</div><div className="vt-bot-kpiv">{rules ?? "—"}</div></div>
          <div><div className="vt-bot-eb" style={{ fontSize: 10 }}>ml</div><div className={`vt-bot-kpiv ${ml !== null ? "vt-bot-p" : "vt-bot-t3"}`}>{ml ?? "—"}</div></div>
          <div><div className="vt-bot-eb" style={{ fontSize: 10 }}>final</div><div className="vt-bot-kpiv">{final !== null ? final.toFixed(0) : "—"}</div></div>
        </div>
        <div style={{ fontSize: 10.5 }}><NotRecorded what="p · model_type · label bet" /></div>
      </Node>
      <div className="vt-bot-arrow">{I.arrow}</div>
      <Node title="action" cls={act ? "vt-bot-t1" : "vt-bot-o"}><div className="vt-bot-kv"><span className="k">row</span><span className="v">{act ? act.type.toLowerCase() : "none yet"}</span><span className="k">at</span><span className="v">{act ? fmtClockET(act.time) + " ET" : "—"}</span></div><div className="vt-bot-t2" style={{ fontSize: 11.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{act ? act.message.slice(0, 90) : "waiting for Node admission"}</div></Node>
    </div>
  );
}

// ── state element: incident ─────────────────────────────────────────────────
function autoResumeLines(ar: unknown): string[] {
  if (!ar || typeof ar !== "object") return [];
  return Object.entries(ar as Record<string, unknown>).filter(([, v]) => ["string", "number", "boolean"].includes(typeof v)).slice(0, 6).map(([k, v]) => `${k.replace(/([A-Z])/g, " $1").toLowerCase()} ${String(v)}`);
}
export function IncidentBody({ state, status, issues, onAct, busy }: { state: DerivedState; status: BotStatus | undefined; issues: Issue[]; onAct: (k: ActionKind) => Promise<void>; busy: boolean }) {
  const [confirm, setConfirm] = useState(false);
  const inc = state.incident;
  const lead = issues.find((i) => i.id === "kill") ?? issues[0];
  const lines = autoResumeLines(status?.autoResume);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10 }}>
      <div><div className="vt-bot-eb">Condition</div><div className="vt-bot-t1" style={{ fontSize: 12.5 }}>{inc?.detail ?? lead?.condition ?? "—"}</div></div>
      <div><div className="vt-bot-eb">Consequence</div><div className="vt-bot-t1" style={{ fontSize: 12.5 }}>{lead?.consequence ?? (inc?.kind === "model" ? "scores are rules-only" : "—")}</div></div>
      <div><div className="vt-bot-eb">Clears when</div><div className="vt-bot-t2" style={{ fontSize: 12 }}>{lead?.clears ?? "—"}</div>{lines.length ? <div className="vt-bot-kv" style={{ marginTop: 4 }}>{lines.map((l, k) => { const sp = l.lastIndexOf(" "); return <span key={k} style={{ display: "contents" }}><span className="k">{l.slice(0, sp)}</span><span className="v">{l.slice(sp + 1)}</span></span>; })}</div> : null}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}><div className="vt-bot-eb">Action</div>
        {inc?.kind === "kill" ? (confirm ? (
          <div className="vt-bot-confirm"><div className="vt-bot-t2">clears the latch and re-activates the loop; the rule re-latches at the same line</div><div className="vt-bot-row"><button type="button" data-primary="true" disabled={busy} onClick={() => { setConfirm(false); void onAct("resume"); }}>Resume</button><button type="button" onClick={() => setConfirm(false)}>cancel</button></div></div>
        ) : <button type="button" className="vt-bot-act" onClick={() => setConfirm(true)}>Resume · keep rule<small>re-latches at the same line</small></button>) : <div className="vt-bot-t3" style={{ fontSize: 12 }}>clears on its own condition</div>}
      </div>
    </div>
  );
}

// ── state element: day review ───────────────────────────────────────────────
export function DayReviewBody({ day, account, market, insight, bench }: { day: DayReview; account: BotAccount | undefined; market: MarketStatus | undefined; insight: Insight | undefined; bench?: Benchmark | undefined }) {
  const pnl = num(account?.dailyPnL), pnlPct = num(account?.dailyPnLPct);
  const spyToday = spyTodayPct(bench?.spyLast, Date.now());
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 10 }}>
      <div><div className="vt-bot-eb">Activity</div><div className="vt-bot-kv"><span className="k">scans</span><span className="v">{day.scans}</span><span className="k">most candidates</span><span className="v">{day.candidatesMax}</span><span className="k">orders placed</span><span className="v">{day.trades}</span><span className="k">refused by sizing</span><span className="v">{day.sizingBlocks}</span><span className="k">cancelled by deploy</span><span className="v">{day.deploys}</span><span className="k">event-loop stalls</span><span className="v">{day.stalls}{day.stallMaxMs ? ` · max ${(day.stallMaxMs / 1000).toFixed(1)} s` : ""}</span></div></div>
      <div><div className="vt-bot-eb">Result</div><div className="vt-bot-kpiv">{fmtMoney(num(account?.equity))} {pnl !== null ? <span className={pnl < 0 ? "vt-bot-r" : "vt-bot-g"} style={{ fontSize: 13 }}>{fmtCompact(pnl)}{pnlPct !== null ? ` · ${fmtPct(pnlPct)}` : ""}</span> : null}</div><div className="vt-bot-kv"><span className="k">SPY today</span>{spyToday !== null ? <span className={`v ${spyToday < 0 ? "vt-bot-r" : "vt-bot-g"}`}>{fmtPct(spyToday)}</span> : <span className="v vt-bot-t3">not recorded</span>}<span className="k">kill latched</span><span className="v">{day.killAt ? fmtClockET(day.killAt) + " ET" : "no"}</span><span className="k">emails</span><span className="v">{day.emails}</span></div></div>
      <div><div className="vt-bot-eb">Model today</div><div className="vt-bot-kv"><span className="k">retrains</span><span className="v">{day.retrains} · fresh ×{day.freshChecks}</span><span className="k">age</span><span className="v">{insight?.model?.age_hours != null ? `${insight.model.age_hours.toFixed(1)} h` : "—"}</span><span className="k">constant inputs</span><span className="v">{insight?.recent?.constant_features ? `${insight.recent.constant_features.length} / ${insight.recent.feature_count}` : "—"}</span><span className="k">next open</span><span className="v">{market?.nextOpen ? new Date(market.nextOpen).toLocaleString("en-US", { weekday: "short", hour: "2-digit", minute: "2-digit", timeZone: "America/New_York" }) : "—"}</span></div></div>
    </div>
  );
}

// ── model zone ──────────────────────────────────────────────────────────────
function Hbar({ label, w, n, cls, lw = 104 }: { label: string; w: number | null | undefined; n?: number; cls: string; lw?: number }) {
  return <div className="vt-bot-row" style={{ gap: 8, height: 16 }}><span className="vt-bot-t2" style={{ width: lw, fontSize: 10.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{label}</span><div className="vt-bot-bar"><i className={cls} style={{ width: `${Math.max(0, Math.min(100, w ?? 0))}%`, background: "currentColor" }} /></div><span className="vt-bot-m vt-bot-t1" style={{ width: 40, fontSize: 11, textAlign: "right" }}>{w != null ? `${w.toFixed(1)}%` : "—"}</span><span className="vt-bot-m vt-bot-t3" style={{ width: 44, fontSize: 10, textAlign: "right" }}>{n != null ? n.toLocaleString("en-US") : ""}</span></div>;
}
export function ModelBody({ insight, audit, compact }: { insight: Insight | undefined; audit: AuditRow[]; compact: boolean }) {
  if (!insight || insight.error) return <div className="vt-bot-empty"><NotRecorded what={insight?.error ? `insight unavailable · ${insight.error}` : "insight not loaded"} /></div>;
  const m = insight.model, r = insight.recent, b = insight.bands, bl = insight.blend;
  const age = m?.age_hours ?? null, fb = m?.fallback_at_hours ?? null;
  const vxxPinned = r?.vxx_ratio && r.vxx_ratio.min !== null && r.vxx_ratio.min === r.vxx_ratio.max;
  const tier3 = audit.filter((x) => x.type === "TIER3" && /retrain|fresh/i.test(x.message) || x.type === "TIER3-ML-ERROR").slice(0, 4);
  const q = b?.score_quintile ?? {};
  const edges = b?.score_quintile_edges ?? [];
  const be = b?.labeling?.win_pct && b?.labeling?.stop_pct ? Math.abs(b.labeling.stop_pct) / (Math.abs(b.labeling.win_pct) + Math.abs(b.labeling.stop_pct)) * 100 : null;
  const H = 64;
  const feats = [...(r?.features ?? [])].sort((a, c) => Number(c.constant) - Number(a.constant) || a.name.localeCompare(c.name));
  const mt = Object.entries(r?.model_types ?? {}).sort((a, c) => c[1] - a[1]);
  const typeWord = mt[0]?.[0] ?? "not recorded";
  return (
    <div style={{ display: "grid", gridTemplateColumns: compact ? "repeat(auto-fit, minmax(240px, 1fr))" : "0.9fr 1.3fr 1.15fr", gap: 12, flex: "1 1 auto", minHeight: 0 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 7, minWidth: 0 }}>
        <div className="vt-bot-row"><St word={typeWord} cls={typeWord.startsWith("fallback") ? "vt-bot-o" : typeWord === "not recorded" ? "vt-bot-t3" : "vt-bot-t1"} out={typeWord === "not recorded"} /><span className="vt-bot-sp" /><span className="vt-bot-st vt-bot-p"><span className="vt-bot-dot" />ml values</span></div>
        <div className="vt-bot-kv">
          <span className="k">age · fallback at<Help k="modelAge" /></span><span className="v">{age != null ? `${age.toFixed(1)} h` : "—"} · {fb ?? "—"} h</span>
          <span className="k">regime route<Help k="regimeRoute" /></span><span className="v">{Object.keys(r?.regime_used ?? {}).length ? Object.entries(r!.regime_used).map(([k, v]) => `${k} ${v}`).join(" · ") : "not recorded"}</span>
          <span className="k">vxx ratio<Help k="vxxRatio" /></span><span className={`v ${vxxPinned ? "vt-bot-r" : ""}`}>{r?.vxx_ratio?.last ?? "—"}{vxxPinned ? " · pinned" : r?.vxx_ratio?.min != null ? ` · ${r.vxx_ratio.min}–${r.vxx_ratio.max}` : ""}</span>
          <span className="k">blend r / ml<Help k="blend" /></span><span className="v">{bl?.rule_weight != null ? `${bl.rule_weight.toFixed(2)} / ${bl.ml_weight?.toFixed(2)}` : "—"}</span>
          <span className="k">gated · blended<Help k="gated" /></span><span className="v">{r ? `${r.gated} · ${r.blended} of ${r.traced}` : "—"}</span>
          <span className="k">errors</span><span className={`v ${r?.errors ? "vt-bot-r" : ""}`}>{r ? r.errors : "—"}</span>
          <span className="k">calibrator<Help k="calibrator" /></span><span className="v">{m?.calibrator_newer_than_model === null || m?.calibrator_newer_than_model === undefined ? "unknown" : m.calibrator_newer_than_model ? "newer than model" : "paired"}</span>
        </div>
        {age != null && fb ? <div className="vt-bot-row" style={{ gap: 8 }}><span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 10.5, width: 28 }}>age</span><div className="vt-bot-bar" style={{ height: 5 }}><i style={{ width: `${Math.min(100, (age / fb) * 100)}%`, background: "var(--text-secondary)" }} /><span className="vt-bot-tick" style={{ left: `${Math.min(100, ((fb - 4) / fb) * 100)}%`, background: "var(--accent-orange)", width: 1 }} /></div><span className="vt-bot-m vt-bot-t3" style={{ fontSize: 10.5 }}>{fb - 4} · {fb} h</span></div> : null}
        <div className="vt-bot-hr" />
        <div className="vt-bot-eb">Retrain checks<Help k="retrainChecks" /></div>
        {tier3.length ? tier3.map((x, k) => <div key={k} className="vt-bot-ev"><span className="ts">{fmtClockET(x.time)}</span><span className={`tx ${x.type === "TIER3-ML-ERROR" ? "vt-bot-r" : ""}`}>{x.message.replace(/^ML (model|retrain)\s*/i, "").slice(0, 70)}</span></div>) : <NotRecorded what="no Tier 3 row in the log window" />}
        {r?.win_probability?.n ? <><div className="vt-bot-hr" /><div className="vt-bot-eb">Win probability · p10 / p50 / p90<Help k="winProbability" /></div><div className="vt-bot-m vt-bot-p" style={{ fontSize: 12 }}>{r.win_probability.p10} / {r.win_probability.p50} / {r.win_probability.p90} <span className="vt-bot-t3">n {r.win_probability.n}</span></div></> : null}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0, minHeight: 0 }}>
        <div className="vt-bot-row"><span className="vt-bot-eb">What it saw · last {r?.window ?? 0} candidates<Help k="whatItSaw" /></span><span className="vt-bot-sp" /><span className="vt-bot-m" style={{ fontSize: 10.5 }}>{r ? <><span className="vt-bot-r">{r.constant_features.length} constant</span> · <span className="vt-bot-t2">{r.feature_count - r.constant_features.length} move</span></> : "—"}</span></div>
        {feats.length ? <div className="vt-bot-feat" style={{ overflow: "hidden" }}>{feats.map((f) => <div key={f.name} className="f" title={`mode ${f.mode} · ${(f.mode_fraction * 100).toFixed(0)}% · variance ${f.variance}`}><span className={`nm ${f.constant ? "vt-bot-r" : "vt-bot-t1"}`}>{f.name}</span><div className="vt-bot-bar" style={{ height: 4 }}><i style={{ width: `${Math.max(2, Math.min(100, f.constant ? 2 : 100 - f.mode_fraction * 100))}%`, background: f.constant ? "var(--accent-red)" : "var(--text-secondary)" }} /></div><span className="vv">{f.constant ? "constant" : f.last}</span></div>)}</div> : <NotRecorded what="no feature vectors in the window" />}
        <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 10.5 }}>bar = share of candidates where the value moved · constant = modal value on ≥ 98 %</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0, minHeight: 0 }}>
        <div className="vt-bot-row"><span className="vt-bot-eb">Score → forward win rate<Help k="scoreQuintile" /></span><span className="vt-bot-sp" /><span className="vt-bot-m vt-bot-t3" style={{ fontSize: 10.5 }}>+5d · +20d{b?.labeling ? ` · label +${b.labeling.win_pct} / ${b.labeling.stop_pct}` : ""}</span></div>
        {Object.keys(q).length ? (
          <div style={{ position: "relative" }}>
            {be !== null ? <><div style={{ position: "absolute", left: 0, right: 0, top: `${(1 - be / 100) * H}px`, borderTop: "1px dashed var(--text-secondary)", opacity: .6 }} /><span className="vt-bot-m vt-bot-t3" style={{ position: "absolute", right: 0, top: `${(1 - be / 100) * H - 13}px`, fontSize: 9.5 }}>breakeven {be.toFixed(1)}%</span></> : null}
            <div className="vt-bot-quint">{["Q1", "Q2", "Q3", "Q4", "Q5"].map((k, idx) => { const a = q[k]?.["+5d"]?.win_rate, c = q[k]?.["+20d"]?.win_rate; return <div key={k}><div className="bars"><i title={`+5d ${a ?? "—"}% n ${q[k]?.["+5d"]?.n ?? 0}`} style={{ height: `${((a ?? 0) / 100) * H}px` }} /><i data-h="2" title={`+20d ${c ?? "—"}% n ${q[k]?.["+20d"]?.n ?? 0}`} style={{ height: `${((c ?? 0) / 100) * H}px` }} /></div><span className="vt-bot-m vt-bot-t1" style={{ fontSize: 10.5 }}>{k}</span><span className="vt-bot-m vt-bot-t3" style={{ fontSize: 9.5 }}>{idx === 0 ? `≤${edges[0]?.toFixed(0) ?? ""}` : idx === 4 ? `≥${edges[3]?.toFixed(0) ?? ""}` : `${edges[idx - 1]?.toFixed(0) ?? ""}–${edges[idx]?.toFixed(0) ?? ""}`}</span></div>; })}</div>
          </div>
        ) : <NotRecorded what={b?.error ? `bands unavailable · ${b.error}` : "band report not loaded"} />}
        {b?.decision ? <><div className="vt-bot-hr" /><div className="vt-bot-eb" style={{ fontSize: 10 }}>By decision · +5d<Help k="byDecision" /></div>{Object.entries(b.decision).slice(0, 5).map(([k, v]) => <Hbar key={k} label={k.replace(/_/g, " · ")} w={v["+5d"]?.win_rate} n={v["+5d"]?.n} cls={k === "taken" ? "vt-bot-p" : "vt-bot-t3"} />)}</> : null}
        {b?.decision_x_band ? <><div className="vt-bot-hr" /><div className="vt-bot-eb" style={{ fontSize: 10 }}>Taken × day move · +5d</div>{Object.entries(b.decision_x_band).filter(([k]) => k.startsWith("taken|")).slice(0, 4).map(([k, v]) => <Hbar key={k} label={k.replace("taken|", "")} w={v["+5d"]?.win_rate} n={v["+5d"]?.n} cls={(v["+5d"]?.win_rate ?? 0) >= (be ?? 50) ? "vt-bot-g" : "vt-bot-r"} lw={120} />)}</> : null}
      </div>
    </div>
  );
}

// ── chart / book / orders ───────────────────────────────────────────────────
// ── chart ───────────────────────────────────────────────────────────────────
const TF_KEY = "vt-bot-chart-tf";
function readTf(): Timeframe {
  try { return timeframeSpec(sessionStorage.getItem(TF_KEY)).tf; } catch { return DEFAULT_TIMEFRAME; }
}
export interface ChartSubjectRow { key: string; label: string; symbol: string | null; kind: "position" | "candidate"; position?: BotPosition; candidate?: WatchCandidate; pnlPct: number | null; score: number | null }
/** One chart subject per open position (an option leg charts its underlying),
 *  then the scan's candidates that are not already held. */
export function chartSubjects(positions: BotPosition[] | undefined, candidates: readonly WatchCandidate[] = []): ChartSubjectRow[] {
  const rows: ChartSubjectRow[] = (positions ?? []).map((p) => {
    const isOption = !!p.isOption || (p.asset_class ?? "").toLowerCase().includes("option");
    const position = { ...p, isOption };
    // chips stay short ("CRWV 55P"); the expiry lives in the chart legend and the chip's title
    return { key: p.ticker, label: isOption ? fmtOcc(p.ticker).split(" ").slice(0, 2).join(" ") : p.ticker, symbol: underlyingOf(position), kind: "position" as const, position, pnlPct: num(p.pnlPct), score: null };
  });
  const held = new Set(rows.map((r) => r.symbol).filter((x): x is string => !!x));
  for (const c of candidates) {
    if (held.has(c.ticker)) continue;
    rows.push({ key: candidateKey(c.ticker), label: c.ticker, symbol: c.ticker, kind: "candidate", candidate: c, pnlPct: null, score: c.score });
  }
  return rows;
}
export function ChartBody({ positions, candidates, focus, onFocus, paused }: { positions: BotPosition[] | undefined; candidates?: readonly WatchCandidate[]; focus: string | null; onFocus: (key: string) => void; paused?: boolean }) {
  const [tf, setTfState] = useState<Timeframe>(readTf);
  const setTf = (t: Timeframe) => {
    setTfState(t);
    try { sessionStorage.setItem(TF_KEY, t); } catch (err) { console.warn("[bot chart] timeframe not persisted (storage unavailable):", err); }
  };
  const subjects = useMemo(() => chartSubjects(positions, candidates ?? []), [positions, candidates]);
  const sel = subjects.find((s) => s.key === focus) ?? subjects[0] ?? null;
  const overlayFn = useCallback((theme: Theme): Overlay => {
    if (!sel) return EMPTY_OVERLAY;
    if (sel.kind === "candidate" && sel.candidate) return candidateOverlay(sel.candidate, theme);
    return sel.position ? overlayFor(sanitizePosition(sel.position), theme, Date.now()) : EMPTY_OVERLAY;
  }, [sel]);
  if (!positions) return <div className="vt-bot-empty"><NotRecorded what="positions not loaded" /></div>;
  if (!subjects.length) return <div className="vt-bot-empty"><St word="flat · nothing to chart" out /></div>;
  const strip = (
    <div className="vt-chart-strip" aria-label="Chart symbol">
      {subjects.map((s) => (
        <button key={s.key} type="button" className="vt-chart-chip" data-kind={s.kind} aria-pressed={sel?.key === s.key} onClick={() => onFocus(s.key)} title={s.kind === "candidate" ? `${s.label} · scan candidate · score ${s.score?.toFixed(1) ?? "—"}` : s.position?.isOption ? `${fmtOcc(s.key)} · charts ${s.symbol ?? "?"}` : s.key}>
          {s.label}
          {s.kind === "candidate"
            ? <span className="pl vt-bot-t2" title="scan score">{s.score === null ? "·" : s.score.toFixed(0)}</span>
            : <span className={`pl ${s.pnlPct === null ? "vt-bot-t3" : s.pnlPct < 0 ? "vt-bot-r" : "vt-bot-g"}`}>{s.pnlPct === null ? "·" : fmtPct(s.pnlPct, 1)}</span>}
        </button>
      ))}
    </div>
  );
  return <BotChart symbol={sel?.symbol ?? null} overlayFor={overlayFn} tf={tf} onTf={setTf} top={strip} paused={paused} />;
}

// ── watch: what the scan looked at, decided and refused ─────────────────────
const OUTCOME_CLS: Record<string, string> = { queued: "vt-bot-b", refused: "vt-bot-o", blocked: "vt-bot-o" };
function CandidateRow({ c, threshold, focus, onFocus }: { c: WatchCandidate; threshold: number | null; focus: string | null; onFocus: (key: string) => void }) {
  const key = candidateKey(c.ticker);
  const pct = c.score === null ? 0 : Math.max(0, Math.min(100, c.score));
  const out = c.outcome;
  const word = out ? `${out.kind === "queued" ? "queued" : out.type.toLowerCase()}${out.kind !== "queued" ? ` · ${out.reason}` : ""}` : c.decided ? "handed to Node" : c.admitted === false ? "below threshold" : "no decision row";
  const cls = out ? OUTCOME_CLS[out.kind] : c.decided ? "vt-bot-t1" : "vt-bot-t3";
  return (
    <div className="vt-watch-row" data-admitted={c.admitted === null ? undefined : String(c.admitted)}>
      <span className="rk">{c.rank !== null ? `#${c.rank}` : "·"}</span>
      <button type="button" className="tk" aria-pressed={focus === key} onClick={() => onFocus(key)} title={`${c.ticker} · chart it`}>{c.ticker}</button>
      <span className="ac" title={c.side ?? undefined}>{c.action.toLowerCase()}{c.sectorClash ? " · sector clash" : ""}</span>
      <span className="sc"><span className={c.admitted === false ? "vt-bot-t3" : "vt-bot-t1"}>{c.score === null ? "—" : c.score.toFixed(1)}</span><div className="vt-bot-bar" title={threshold !== null ? `admission ≥ ${threshold}` : "threshold not recorded"}><i style={{ width: `${pct}%`, background: c.admitted === false ? "var(--text-tertiary)" : "var(--accent)" }} />{threshold !== null ? <span className="vt-bot-tick" style={{ left: `${Math.max(0, Math.min(100, threshold))}%` }} /> : null}</div></span>
      <span className={`ch ${c.changePct === null ? "vt-bot-t3" : c.changePct < 0 ? "vt-bot-r" : "vt-bot-g"}`}>{c.changePct === null ? "·" : fmtPct(c.changePct, 1)}</span>
      <span className={`oc ${cls}`} title={out ? `${out.type} · ${fmtClockET(out.time)} ET · ${out.reason}` : undefined}>{word}</span>
      <div className="vt-watch-sub">
        <span className="why" title={c.reasons.join(" · ")}>{c.reasons.length ? c.reasons.join(" · ") : c.sector ?? "no reasons recorded"}</span>
        {c.factors ? <span className="vt-watch-f" aria-label="factor scores">{FACTOR_KEYS.map((k) => { const v = c.factors![k]; return <span key={k} title={`${k} ${v === null ? "—" : v.toFixed(0)}`}>{FACTOR_SHORT[k]}<i><b style={{ width: `${v === null ? 0 : Math.max(0, Math.min(100, v))}%` }} /></i></span>; })}</span> : null}
      </div>
    </div>
  );
}
export function WatchBody({ model, focus, onFocus, nowMs }: { model: WatchModel; focus: string | null; onFocus: (key: string) => void; nowMs: number }) {
  if (!model.scanAt && !model.candidates.length) return <div className="vt-bot-empty"><NotRecorded what="no scan result yet" /><span className="vt-chart-note">the scan's top-10, its decisions and Node's refusals appear here after the first tier-2 scan</span></div>;
  return (
    <div className="vt-watch">
      <div className="vt-watch-hd">
        <span>scan {model.scanAt ? `${fmtClockET(model.scanAt)} ET` : "—"}{model.partial ? " · partial" : ""}</span>
        {model.scanned !== null ? <span>{model.scanned.toLocaleString("en-US")} scanned → {model.filtered ?? "?"} → {model.deep ?? "?"} deep</span> : null}
        <span>slots {model.slots ?? "?"} · options {model.optionsSlots ?? "?"}</span>
        <span>admission ≥ {model.threshold ?? "?"}<Help k="admission" /></span>
        <span>outcome<Help k="outcome" /></span>
      </div>
      {model.candidates.length ? model.candidates.map((c) => <CandidateRow key={c.ticker} c={c} threshold={model.threshold} focus={focus} onFocus={onFocus} />) : <div className="vt-bot-row"><St word="the last scan surfaced no candidates" out /></div>}
      {model.decided.length ? (
        <>
          <div className="vt-watch-sec">decided · handed to Node<Help k="decided" /></div>
          {model.decided.map((d) => <div key={`d-${d.ticker}`} className="vt-watch-ref"><button type="button" className="tk" aria-pressed={focus === candidateKey(d.ticker)} onClick={() => onFocus(candidateKey(d.ticker))}>{d.ticker}</button><span className="ty vt-bot-t2">{d.action.toLowerCase()}{d.kind ? ` · ${d.kind}` : ""}</span><span className="rs">{d.reasons[0] ?? ""}</span><span className="tm">{d.score === null ? "" : d.score.toFixed(0)}</span></div>)}
        </>
      ) : null}
      {model.refused.length ? (
        <>
          <div className="vt-watch-sec">refused · why<Help k="refused" /></div>
          {model.refused.slice(0, 8).map((r) => <div key={`r-${r.ticker}-${r.type}`} className="vt-watch-ref"><button type="button" className="tk" aria-pressed={focus === candidateKey(r.ticker)} onClick={() => onFocus(candidateKey(r.ticker))}>{r.ticker}</button><span className="ty">{r.type.toLowerCase()}</span><span className="rs" title={r.reason}>{r.reason}</span><span className="tm">{r.time ? fmtAge(nowMs - Date.parse(r.time)) : ""}</span></div>)}
        </>
      ) : null}
    </div>
  );
}
const SLEEVE_LABEL: Record<string, string> = { options: "options", floor: "ETF floor", stocks: "stocks" };
const ACTIVITY_LABEL: Array<[keyof NonNullable<Sleeves["nonTrade"]>["groups"], string]> = [["dividends", "div"], ["interest", "int"], ["fees", "fees"], ["journals", "journals"], ["optionEvents", "option events"], ["corporate", "corp"], ["transfers", "transfers"], ["other", "other"]];
/** Where the P&L comes from — realized since the first fetched fill + open. */
export function SleeveSummary({ sleeves }: { sleeves: Sleeves | undefined }) {
  if (!sleeves) return null;
  if (sleeves.pending) return <div className="vt-bot-t3" style={{ fontSize: 11, padding: "4px 0" }}><St word="P&L by sleeve · computing from fills" /></div>;
  if (!sleeves.rows) return <div style={{ fontSize: 11, padding: "4px 0" }}><NotRecorded what={`P&L by sleeve unavailable${sleeves.error ? ` · ${sleeves.error}` : ""}`} /></div>;
  const since = sleeves.since ? sleeves.since.slice(5, 10) : "?";
  const cls = (v: number | null | undefined) => (v === null || v === undefined ? "vt-bot-t3" : v < 0 ? "vt-bot-r" : v > 0 ? "vt-bot-g" : "");
  // the book's P&L column is sized for "−2.4%": thousands as k
  const k = (v: number) => (Math.abs(v) >= 1000 ? `${v < 0 ? MINUS : ""}$${(Math.abs(v) / 1000).toFixed(1)}k` : fmtCompact(v));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingBottom: 6 }} title={`realized from Alpaca fills since ${sleeves.since ?? "?"} (average cost; every lot's basis from the account's full fill history${sleeves.preWindowFills ? `, ${sleeves.preWindowFills} earlier fills` : ""}); open = current unrealized${sleeves.truncated ? " · oldest fill history TRUNCATED (basis approximate for the oldest lots)" : ""}`}>
      <div className="vt-bot-eb" style={{ padding: "6px 0 2px", fontSize: 10 }}>P&L by sleeve<Help k="sleeve" /> · realized since {since}<Help k="realized" /> · open</div>
      {sleeves.rows.map((r) => (
        <div key={r.sleeve} className="vt-bot-trow">
          <span className="sym">{SLEEVE_LABEL[r.sleeve] ?? r.sleeve}</span><span className="q vt-bot-t3">{r.fills}f</span>
          <span className={`px ${cls(r.realized)}`}>{k(r.realized)}</span><span className={`pl ${cls(r.unrealized)}`}>{r.unrealized === null ? "·" : k(r.unrealized)}</span>
        </div>
      ))}
      {sleeves.optionsByUnderlying?.length ? (() => {
        const worst = sleeves.optionsByUnderlying.filter((u) => u.realized < 0).slice(0, 5);
        return worst.length ? (
          <div className="vt-bot-t3" style={{ fontSize: 10.5, lineHeight: 1.4 }} title="options realized P&L by underlying, worst first (OCC root); the full list is on /api/bot/sleeves">
            worst options<Help k="worstOptions" />: {worst.map((u) => `${u.root} ${k(u.realized)}`).join(" · ")}{(sleeves.optionsUnderlyings ?? 0) > worst.length ? ` · of ${sleeves.optionsUnderlyings}` : ""}
          </div>
        ) : null;
      })() : null}
      {sleeves.optionsByHold?.some((h) => h.events > 0) ? (
        <div className="vt-bot-t3" style={{ fontSize: 10.5, lineHeight: 1.4 }} title="options realized P&L by how long the closed lot was held (close − open of the lot); 'unknown' = the lot predates the fetched fill history">
          options by hold<Help k="optionsByHold" />: {sleeves.optionsByHold.filter((h) => h.events > 0).map((h) => `${h.bucket} ${k(h.realized)} (${h.events})`).join(" · ")}
        </div>
      ) : null}
      {sleeves.nonTrade ? (() => {
        const g = sleeves.nonTrade.groups;
        const parts = ACTIVITY_LABEL.filter(([key]) => (g[key] ?? 0) !== 0).map(([key, label]) => `${label} ${k(g[key])}`);
        const r = sleeves.reconciliation;
        return (
          <div className="vt-bot-t3" style={{ fontSize: 10.5, lineHeight: 1.4 }} title={`cash the fill ledger cannot see, from Alpaca's activity ledger since ${since} (dividends, interest, fees, journals, option expiration / assignment); unexplained = equity change since ${r?.equityFrom ?? "?"} minus realized, open and other cash — what no ledger accounts for${sleeves.nonTrade.nonCashRows ? ` · ${sleeves.nonTrade.nonCashRows} share-only event(s) counted, not valued` : ""}`}>
            other cash<Help k="otherCash" /> {k(sleeves.nonTrade.total)}{parts.length ? ` · ${parts.join(" · ")}` : ""}{r && r.residual !== null ? ` · unexplained ${k(r.residual)}` : ""}
          </div>
        );
      })() : null}
      {sleeves.truncated || sleeves.rows.some((r) => r.unpricedCloses > 0) ? (
        <div className="vt-bot-t3" style={{ fontSize: 10 }}>{sleeves.truncated ? "fill history truncated · " : ""}{sleeves.rows.some((r) => r.unpricedCloses > 0) ? "some closes predate the window (not counted)" : ""}</div>
      ) : null}
    </div>
  );
}

export function BookBody({ positions, focus, onSelect, sleeves }: { positions: BotPosition[] | undefined; focus: string | null; onSelect: (ticker: string) => void; sleeves?: Sleeves | undefined }) {
  if (!positions) return <div className="vt-bot-empty"><NotRecorded what="positions not loaded" /></div>;
  if (!positions.length) return <div className="vt-bot-list"><SleeveSummary sleeves={sleeves} /><div className="vt-bot-empty"><St word="flat · no positions" /></div></div>;
  const isOpt = (p: BotPosition) => !!p.isOption || (p.asset_class ?? "").toLowerCase().includes("option");
  const groups: Array<[string, BotPosition[]]> = [["equities", positions.filter((p) => !isOpt(p))], ["options", positions.filter(isOpt)]];
  return (
    <div className="vt-bot-list">
      <SleeveSummary sleeves={sleeves} />
      {groups.filter(([, g]) => g.length).map(([name, g]) => (
        <div key={name} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <div className="vt-bot-eb" style={{ padding: "6px 0 2px", fontSize: 10 }}>{name} · {g.length}</div>
          {g.map((p) => { const pl = num(p.pnlPct); return (
            <div key={p.ticker} className="vt-bot-trow" role="button" tabIndex={0} aria-pressed={focus === p.ticker} title={`${p.ticker} · chart it`} onClick={() => onSelect(p.ticker)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(p.ticker); } }} style={{ cursor: "pointer", ...(focus === p.ticker ? { background: "var(--elevate-2, rgba(77,159,255,.08))", margin: "0 -6px", padding: "0 6px", borderRadius: 4 } : {}) }}>
              <span className="sym" title={p.ticker}>{fmtOcc(p.ticker)}</span><span className="q">{num(p.qty) ?? "·"}</span><span className="px">{num(p.currentPrice)?.toFixed(2) ?? "·"}</span><span className={`pl ${pl === null ? "vt-bot-t3" : pl < 0 ? "vt-bot-r" : "vt-bot-g"}`}>{pl === null ? "·" : fmtPct(pl, 1)}</span>
            </div>); })}
        </div>
      ))}
    </div>
  );
}
export function OrdersBody({ orders, deploysToday }: { orders: OpenOrder[] | undefined; deploysToday: number }) {
  if (!orders) return <div className="vt-bot-empty"><NotRecorded what="orders not loaded" /></div>;
  if (!orders.length) return <div className="vt-bot-row"><St word="none working" out /><span className="vt-bot-sp" /><span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 10.5 }}>{deploysToday ? `${deploysToday} cancelled by deploy today` : ""}</span></div>;
  return <div className="vt-bot-list">{orders.slice(0, 8).map((o, k) => <div key={o.id ?? k} className="vt-bot-trow"><span className="sym">{o.symbol}</span><span className="vt-bot-t2 vt-bot-m" style={{ fontSize: 11.5, flex: 1 }}>{o.side} {num(o.qty) ?? ""} · {o.type ?? ""}{num(o.limit_price) !== null ? ` ${num(o.limit_price)}` : ""}</span><St word={`${o.status ?? "working"}${o.submitted_at ? ` · ${fmtAge(Date.now() - Date.parse(o.submitted_at))}` : ""}`} cls="vt-bot-b" /></div>)}</div>;
}

// ── timeline ────────────────────────────────────────────────────────────────
const MARK: Record<string, [string, number]> = { TIER2: ["var(--text-tertiary)", 8], TIER3: ["var(--accent-purple)", 12], "TIER3-ML-ERROR": ["var(--accent-red)", 16], SHUTDOWN: ["var(--text-secondary)", 20], STARTUP: ["var(--text-secondary)", 12], QUEUE: ["var(--text-secondary)", 14], TRADE: ["var(--accent)", 22], "ETF-TRADE": ["var(--accent)", 22], "OPTIONS-TRADE": ["var(--accent)", 22], "DRAWDOWN-KILL": ["var(--accent-red)", 28], "DD-HALT": ["var(--accent-red)", 28], EMAIL: ["var(--accent-orange)", 16], "KILL-AUTO-RESUME-EVAL": ["var(--text-secondary)", 14], "POS-WARN": ["var(--accent-orange)", 12], "SIZE-BLOCK": ["var(--accent-orange)", 14], "WS-EXIT": ["var(--accent)", 20], "STREAM-DISCONNECT": ["var(--accent-red)", 18], "TIER2-FAIL": ["var(--accent-red)", 18], "TIER2-ERROR": ["var(--accent-red)", 14] };
function etHour(ms: number): number {
  const p = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).formatToParts(new Date(ms));
  const g = (t: string) => Number(p.find((x) => x.type === t)?.value ?? 0);
  return (g("hour") % 24) + g("minute") / 60 + g("second") / 3600;
}
export function TimelineBody({ audit, nowMs, killSince, replayAt, onReplay, closed }: { audit: AuditRow[]; nowMs: number; killSince: number | null; replayAt: number | null; onReplay: (ms: number | null) => void; closed: boolean }) {
  const x = (h: number) => ((h - 4) / 16) * 100;
  const nowH = etHour(nowMs);
  const dayKey = new Date(nowMs).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const rows = audit.filter((r) => new Date(Date.parse(r.time)).toLocaleDateString("en-CA", { timeZone: "America/New_York" }) === dayKey && MARK[r.type] && !/^(SCHEDULE|RULES|EXECUTION)$/.test(r.type));
  const dayStartMs = nowMs - (nowH - 4) * 3600_000;
  return (
    <div className="vt-bot-timeline">
      <div className="rth" style={{ left: `${x(9.5)}%`, width: `${x(16) - x(9.5)}%` }} />
      {killSince !== null ? <div className="halt" style={{ left: `${x(Math.max(4, etHour(killSince)))}%`, width: `${Math.max(0, x(Math.min(nowH, 20)) - x(Math.max(4, etHour(killSince))))}%` }} /> : null}
      {rows.map((r, k) => { const [c, h] = MARK[r.type]; const hh = etHour(Date.parse(r.time)); if (hh < 4 || hh > 20) return null; return <div key={k} className="mk" title={`${fmtClockET(r.time)} ET · ${r.type} · ${r.message.slice(0, 120)}`} style={{ left: `${x(hh)}%`, height: h, background: c, opacity: r.type === "TIER2" ? .55 : 1 }} />; })}
      {Array.from({ length: 9 }, (_, i) => 4 + i * 2).map((h) => <span key={h} className="hr" style={{ left: `${x(h)}%` }}>{String(h).padStart(2, "0")}:00</span>)}
      <div className="now" style={{ left: `${x(Math.min(20, Math.max(4, replayAt !== null ? etHour(replayAt) : nowH)))}%`, background: replayAt !== null ? "var(--accent-orange)" : undefined }} />
      <span className="vt-bot-m" style={{ position: "absolute", left: `calc(${x(Math.min(20, Math.max(4, replayAt !== null ? etHour(replayAt) : nowH)))}% + 4px)`, top: 2, fontSize: 10, color: replayAt !== null ? "var(--accent-orange)" : "var(--text-primary)" }}>{replayAt !== null ? `replay ${fmtClockET(replayAt)} ET` : `${closed ? "close" : "now"} ${fmtClockET(nowMs).slice(0, 5)} ET`}</span>
      <input type="range" min={0} max={960} value={replayAt !== null ? Math.round((etHour(replayAt) - 4) * 60) : Math.round((Math.min(20, nowH) - 4) * 60)} aria-label="Replay the day" onChange={(e) => { const m = Number(e.target.value); const ms = dayStartMs + m * 60_000; onReplay(ms >= nowMs - 30_000 ? null : ms); }} />
    </div>
  );
}
