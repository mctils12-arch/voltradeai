// Bot page v2 — one full-bleed page under the site nav. The grid is the
// resting layout; the centre column holds whatever the bot is doing right
// now (pageState → director → zones); every panel can be held, pinned,
// opened as a window over the page (ghost slot kept) or zoomed; the view
// freezes, the bot never does. Layout motion runs on frameCore springs
// (Law I: one rAF loop; transforms only during motion).
// Brief: research/bot_cockpit_design_brief_v2.md.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { frameCore, PRIORITY, type Spring } from "@/render/frameCore";
import { useBotData, num, type BotData } from "./useBotData";
import { derivePageState, type AuditRow } from "./pageState";
import { deriveStages, deriveIssues, deriveDayReview, rowsToday, type DeriveInput } from "./derive";
import { Director } from "./director";
import { tierFor, gridStyle, TEMPLATES, EMPTY_FLAGS, toggleIn, isPinned, initialFloatRect, clampRect, snapEdges, type LayoutFlags, type PanelId, type Rect, type Tier } from "./zoneModel";
import { Panel, KpiStrip, StagesBody, IssuesBody, ScanStripBody, DecisionBody, IncidentBody, DayReviewBody, ModelBody, ChartBody, BookBody, OrdersBody, TimelineBody, ICONS, type Chrome, type ActionKind } from "./panels";
import { fmtAge } from "./format";

function useNow(periodMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), periodMs); return () => clearInterval(id); }, [periodMs]);
  return now;
}
function useTier(): { tier: Tier; w: number; h: number; touch: boolean } {
  const read = () => ({ w: typeof window === "undefined" ? 1440 : window.innerWidth, h: typeof window === "undefined" ? 900 : window.innerHeight });
  const [vp, setVp] = useState(read);
  useEffect(() => { const on = () => setVp(read()); window.addEventListener("resize", on); window.addEventListener("orientationchange", on); return () => { window.removeEventListener("resize", on); window.removeEventListener("orientationchange", on); }; }, []);
  const touch = typeof window !== "undefined" && typeof window.matchMedia === "function" ? window.matchMedia("(pointer: coarse)").matches : false;
  return { tier: tierFor(vp.w), w: vp.w, h: vp.h, touch };
}
function reducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const TITLES: Record<PanelId, string> = { kpi: "Edge", stages: "Stages", issues: "Issues", state: "Now", model: "Model", chart: "Chart", book: "Book", orders: "Orders", timeline: "Today" };

/** FLIP on frameCore springs: when the layout key changes, every slot that
 *  moved springs from its old box to its new one with a transform; on
 *  settle the transform is dropped and CSS grid owns the final box. */
function useFlip(layoutKey: string, slots: React.MutableRefObject<Map<PanelId, HTMLElement>>) {
  const prev = useRef<Map<PanelId, DOMRect>>(new Map());
  const anim = useRef<Map<PanelId, { x: Spring; y: Spring; detach: () => void }>>(new Map());
  const loop = useMemo(() => frameCore(), []);
  useEffect(() => {
    loop.start();
    const un = loop.register(() => {
      for (const [id, a] of anim.current) {
        const el = slots.current.get(id);
        if (!el) continue;
        if (a.x.asleep && a.y.asleep) { el.style.transform = ""; a.detach(); anim.current.delete(id); continue; }
        el.style.transform = `translate(${a.x.value.toFixed(2)}px, ${a.y.value.toFixed(2)}px)`;
      }
    }, PRIORITY.RENDER, { label: "bot-page-flip" });
    return () => { un(); for (const a of anim.current.values()) a.detach(); anim.current.clear(); };
  }, [loop, slots]);
  useLayoutEffect(() => {
    const next = new Map<PanelId, DOMRect>();
    for (const [id, el] of slots.current) next.set(id, el.getBoundingClientRect());
    if (!reducedMotion()) {
      for (const [id, rect] of next) {
        const was = prev.current.get(id);
        if (!was) continue;
        const dx = was.left - rect.left, dy = was.top - rect.top;
        if (Math.abs(dx) < 1 && Math.abs(dy) < 1) continue;
        const cur = anim.current.get(id);
        if (cur) { cur.x.snap(dx + cur.x.value); cur.y.snap(dy + cur.y.value); cur.x.setTarget(0); cur.y.setTarget(0); continue; }
        const sx = loop.createSpring(dx, { stiffness: 170, damping: 26 }); const sy = loop.createSpring(dy, { stiffness: 170, damping: 26 });
        sx.spring.setTarget(0); sy.spring.setTarget(0);
        anim.current.set(id, { x: sx.spring, y: sy.spring, detach: () => { sx.detach(); sy.detach(); } });
      }
    }
    prev.current = next;
  }, [layoutKey, loop, slots]);
}

export default function BotPage() {
  const d = useBotData();
  const now = useNow(1000);
  const { tier, w: pageW, h: pageH, touch } = useTier();
  const [flags, setFlags] = useState<LayoutFlags>(EMPTY_FLAGS);
  const [rects, setRects] = useState<Partial<Record<PanelId, Rect>>>({});
  const [replayAt, setReplayAt] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [actErr, setActErr] = useState<string | null>(null);
  const slots = useRef<Map<PanelId, HTMLElement>>(new Map());
  const held = useRef<Map<PanelId, { since: number; body: ReactNode; ticks: number }>>(new Map());
  const director = useRef(new Director());
  const rootRef = useRef<HTMLDivElement>(null);

  const auditAll: AuditRow[] = useMemo(() => (d.audit.data ?? []).map((r) => ({ time: r.time, type: r.type ?? r.action ?? "", message: r.message ?? r.detail ?? "" })), [d.audit.data]);
  const effNow = replayAt ?? now;
  const audit = useMemo(() => (replayAt === null ? auditAll : auditAll.filter((r) => Date.parse(r.time) <= replayAt)), [auditAll, replayAt]);
  const status = d.status.data, account = d.account.data, market = d.market.data, insight = d.insight.data, perf = d.performance.data, lastScan = d.lastScan.data;
  const modelAge = num(insight?.model?.age_hours) ?? num(d.ml.data?.model_age_hours);
  const fallbackAt = num(insight?.model?.fallback_at_hours);

  const derived = useMemo(() => derivePageState({ audit, killSwitch: !!status?.killSwitch, active: !!status?.active, marketOpen: replayAt !== null ? true : market ? market.isOpen : null, nowMs: effNow, modelAgeHours: modelAge, fallbackAtHours: fallbackAt }), [audit, status?.killSwitch, status?.active, market, effNow, modelAge, fallbackAt, replayAt]);
  // The director only sees states built from fetched data: before the
  // first audit page lands the page is provisional, never "idle".
  const ready = d.audit.isFetched && d.status.isFetched;
  const shown = ready ? director.current.propose(derived.state, now, flags.hold) : "idle";

  const dIn: DeriveInput = useMemo(() => ({
    audit, nowMs: effNow, killSwitch: !!status?.killSwitch, active: !!status?.active, marketOpen: market ? market.isOpen : null,
    equity: num(account?.equity), equityPeak: num(status?.equityPeak) ?? num(account?.equityPeak), maxDrawdownPct: num(status?.maxDrawdownPct),
    modelAgeHours: modelAge, fallbackAtHours: fallbackAt, constantFeatures: insight?.recent?.constant_features ? insight.recent.constant_features.length : null, featureCount: num(insight?.recent?.feature_count),
    openOrders: d.orders.data ? d.orders.data.length : null, totalTrades: num(perf?.totalTrades), slotsAvailable: num((lastScan as { slots_available?: unknown } | undefined)?.slots_available),
    modelTypes: insight?.recent?.model_types ?? null,
  }), [audit, effNow, status, market, account, modelAge, fallbackAt, insight, d.orders.data, perf, lastScan]);
  const stages = useMemo(() => deriveStages(dIn), [dIn]);
  const issues = useMemo(() => deriveIssues(dIn), [dIn]);
  const day = useMemo(() => deriveDayReview(audit, effNow), [audit, effNow]);
  const deploysToday = useMemo(() => rowsToday(audit, effNow).filter((r) => r.type === "SHUTDOWN" && /orders cancelled/i.test(r.message)).length, [audit, effNow]);
  const killSince = useMemo(() => { const k = audit.find((r) => r.type === "DRAWDOWN-KILL"); return status?.killSwitch && k ? Date.parse(k.time) : null; }, [audit, status?.killSwitch]);

  // ticks since hold: count data updates
  const updates = [d.status, d.account, d.positions, d.audit, d.performance, d.market, d.insight, d.lastScan, d.orders].reduce((a, q) => a + (q.dataUpdatedAt || 0), 0);
  const tickCount = useRef(0); const lastUpdates = useRef(updates);
  if (updates !== lastUpdates.current) { lastUpdates.current = updates; tickCount.current += 1; }

  const act = useCallback(async (k: ActionKind) => {
    setBusy(true); setActErr(null);
    try {
      if (k === "resume") { if (status?.killSwitch) await apiRequest("POST", "/api/bot/kill"); await apiRequest("POST", "/api/bot/start"); }
      else if (k === "stop") await apiRequest("POST", "/api/bot/stop");
      else if (k === "kill") await apiRequest("POST", "/api/bot/kill");
      await queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] });
      await queryClient.invalidateQueries({ queryKey: ["/api/bot/audit"] });
    } catch (e: unknown) { setActErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }, [status?.killSwitch]);

  const onHold = useCallback((id: PanelId) => setFlags((f) => ({ ...f, held: toggleIn(f.held, id) })), []);
  const onPin = useCallback((id: PanelId) => setFlags((f) => ({ ...f, pinned: toggleIn(f.pinned, id) })), []);
  const onFloat = useCallback((id: PanelId) => setFlags((f) => {
    if (f.floating.includes(id)) return f;
    const el = slots.current.get(id); const r = el ? el.getBoundingClientRect() : null;
    const page = rootRef.current?.getBoundingClientRect();
    const px = page?.left ?? 0, py = page?.top ?? 0;
    const slot: Rect = r ? { x: r.left - px, y: r.top - py, w: r.width, h: r.height } : { x: 100, y: 100, w: 600, h: 400 };
    setRects((rs) => ({ ...rs, [id]: initialFloatRect(slot, { w: page?.width ?? pageW, h: page?.height ?? pageH }) }));
    return { ...f, floating: [...f.floating, id] };
  }), [pageW, pageH]);
  const onDock = useCallback((id: PanelId) => setFlags((f) => ({ ...f, floating: f.floating.filter((x) => x !== id), zoomed: f.zoomed === id ? null : f.zoomed })), []);
  const onZoom = useCallback((id: PanelId) => setFlags((f) => ({ ...f, zoomed: f.zoomed === id ? null : id, floating: f.floating.filter((x) => x !== id) })), []);
  const onFocus = useCallback((id: PanelId) => { onZoom(id); }, [onZoom]);
  // Esc docks every window and leaves the zoom. Listened on the document
  // while one is open: the page root only received keys while focus was
  // inside it, and a click on the page body left focus outside (2026-10-20).
  const windowsOpen = flags.floating.length > 0 || flags.zoomed !== null;
  useEffect(() => {
    if (!windowsOpen) return;
    const onEsc = (e: KeyboardEvent) => { if (e.key === "Escape") setFlags((f) => ({ ...f, zoomed: null, floating: [] })); };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [windowsOpen]);

  const chrome = useCallback((id: PanelId): Chrome => {
    const h = held.current.get(id);
    return { id, held: flags.held.includes(id), pinned: isPinned(flags, id), heldSince: h?.since ?? null, ticksSinceHold: h ? tickCount.current - h.ticks : 0, nowMs: now, onHold, onPin, onFloat };
  }, [flags, now, onHold, onPin, onFloat]);

  // bodies (fresh), then hold snapshots
  const ages = { status: d.status.dataUpdatedAt ? now - d.status.dataUpdatedAt : null, audit: d.audit.dataUpdatedAt ? now - d.audit.dataUpdatedAt : null, positions: d.positions.dataUpdatedAt ? now - d.positions.dataUpdatedAt : null, insight: d.insight.dataUpdatedAt ? now - d.insight.dataUpdatedAt : null, orders: d.orders.dataUpdatedAt ? now - d.orders.dataUpdatedAt : null, perf: d.performance.dataUpdatedAt ? now - d.performance.dataUpdatedAt : null };
  const freshBody: Record<PanelId, ReactNode> = {
    kpi: null,
    stages: <StagesBody rows={stages} />,
    issues: <IssuesBody issues={issues} nowMs={effNow} onFocus={onFocus} onAct={act} busy={busy} />,
    state: shown === "scanning" ? <ScanStripBody elapsedMs={derived.scan.elapsedMs} lastScan={lastScan} /> : shown === "candidate" || shown === "placed" || shown === "exit" ? <DecisionBody lastScan={lastScan} audit={audit} nowMs={effNow} /> : shown === "incident" ? <IncidentBody state={derived} status={status} issues={issues} onAct={act} busy={busy} /> : shown === "closed" ? <DayReviewBody day={day} account={account} market={market} insight={insight} bench={d.benchmark.data} /> : null,
    model: <ModelBody insight={insight} audit={audit} compact={shown !== "idle" || tier === "s"} />,
    chart: <ChartBody positions={d.positions.data} />,
    book: <BookBody positions={d.positions.data} spot={null} sleeves={d.sleeves.data} />,
    orders: <OrdersBody orders={d.orders.data} deploysToday={deploysToday} />,
    timeline: <TimelineBody audit={auditAll} nowMs={now} killSince={killSince} replayAt={replayAt} onReplay={setReplayAt} closed={market ? !market.isOpen : false} />,
  };
  for (const id of flags.held) { if (!held.current.has(id)) held.current.set(id, { since: now, body: freshBody[id], ticks: tickCount.current }); }
  for (const id of [...held.current.keys()]) { if (!flags.held.includes(id)) held.current.delete(id); }
  const body = (id: PanelId): ReactNode => (held.current.get(id)?.body ?? freshBody[id]);

  const stateTitle = shown === "scanning" ? "Scan · tier 2" : shown === "candidate" ? `Decision · ${((lastScan?.new_trades?.[0] as { ticker?: string } | undefined)?.ticker ?? lastScan?.top_10?.[0]?.ticker ?? "").toUpperCase()}` : shown === "placed" ? "Order working" : shown === "exit" ? "Exit" : shown === "incident" ? `Halted · ${derived.incident?.kind ?? ""} · ${derived.since ? fmtAge(effNow - Date.parse(derived.since)) : ""}` : shown === "closed" ? "Day review" : "Now";
  const hasState = shown !== "idle";
  const panelEl = (id: PanelId, extraStyle?: React.CSSProperties): ReactNode => {
    if (id === "kpi") return null;
    const c = chrome(id);
    const common = { chrome: c, ageMs: id === "model" ? ages.insight : id === "chart" || id === "book" ? ages.positions : id === "orders" ? ages.orders : id === "stages" || id === "issues" || id === "state" || id === "timeline" ? ages.audit : ages.status } as const;
    switch (id) {
      case "stages": return <Panel {...common} title="Stages" meta="last 10 expected ticks" style={{ flex: "0 0 auto", ...extraStyle }}>{body(id)}</Panel>;
      case "issues": return <Panel {...common} title="Issues" meta={issues.length ? `${issues.length} standing` : undefined} scroll style={{ flex: "1 1 auto", ...extraStyle }}>{body(id)}{actErr ? <div className="vt-bot-r" style={{ fontSize: 11.5 }}>action failed · {actErr}</div> : null}</Panel>;
      case "state": return <Panel {...common} title={stateTitle} status={shown === "incident" ? <span className="vt-bot-st vt-bot-r"><span className="vt-bot-dot" />first-out</span> : replayAt !== null ? <span className="vt-bot-st vt-bot-o"><span className="vt-bot-dot" />replay</span> : undefined} style={{ flex: shown === "closed" ? "1 1 auto" : "0 0 auto", ...extraStyle }}>{body(id)}</Panel>;
      case "model": return <Panel {...common} title="Model" meta={insight?.recent?.window ? `shadow · last ${insight.recent.window}` : undefined} style={{ flex: "1 1 auto", ...extraStyle }} scroll>{body(id)}</Panel>;
      case "chart": return <Panel {...common} title="Chart" meta={d.positions.data?.length ? `${d.positions.data.length} positions` : undefined} style={{ flex: "1.05 1 0", ...extraStyle }}>{body(id)}</Panel>;
      case "book": return <Panel {...common} title={`Book${d.positions.data ? ` · ${d.positions.data.length}` : ""}`} meta={account ? `cash ${account.cash != null ? `$${Math.round(Number(account.cash)).toLocaleString("en-US")}` : "—"}` : undefined} scroll style={{ flex: "1.35 1 0", ...extraStyle }}>{body(id)}</Panel>;
      case "orders": return <Panel {...common} title="Orders" style={{ flex: "0.6 1 0", ...extraStyle }}>{body(id)}</Panel>;
      case "timeline": return <Panel {...common} title={new Date(effNow).toLocaleDateString("en-US", { weekday: "short", day: "numeric", month: "short", timeZone: "America/New_York" })} meta="RTH 09:30–16:00 ET · hatched = latched · drag to replay" style={{ height: tier === "s" ? 120 : "100%", ...extraStyle }}>{body(id)}</Panel>;
      default: return null;
    }
  };
  const slotRef = (id: PanelId) => (el: HTMLElement | null) => { if (el) slots.current.set(id, el); else slots.current.delete(id); };
  const slot = (id: PanelId, extraStyle?: React.CSSProperties): ReactNode => flags.floating.includes(id)
    ? <div key={id} ref={slotRef(id)} className="vt-bot-ghost" role="button" tabIndex={0} onClick={() => onDock(id)} onKeyDown={(e) => { if (e.key === "Enter") onDock(id); }} style={extraStyle}>{TITLES[id]} · open as a window · click to dock</div>
    // the slot's flex reaches the panel too: each panel carries its desktop
    // grid flex ("1.35 1 0" for the book), and in the phone/tablet stack —
    // natural heights, nothing to grow into — a 0 basis collapsed Chart,
    // Book and Orders to 0 px (bot-ui harness battery, 2026-10-20)
    : <div key={id} ref={slotRef(id)} style={{ display: "flex", flexDirection: "column", minHeight: 0, minWidth: 0, ...extraStyle }}>{panelEl(id, extraStyle)}</div>;

  const layoutKey = `${tier}|${shown}|${flags.zoomed ?? ""}|${flags.floating.join(",")}`;
  useFlip(layoutKey, slots);
  const tpl = TEMPLATES[tier];
  const kpi = <KpiStrip perf={perf} account={account} status={status} market={market} calendar={d.calendar.data} bench={d.benchmark.data} now={effNow} state={derived} hold={flags.hold} onHold={() => setFlags((f) => ({ ...f, hold: !f.hold }))} />;

  return (
    <div ref={rootRef} className="vt-bot-page" data-touch={touch ? "true" : undefined} data-state={shown} data-tier={tier} tabIndex={-1}>
      {flags.zoomed ? (
        <div className="vt-bot-grid" style={{ gridTemplateColumns: "minmax(0, 1fr)", gridTemplateRows: `${tier === "s" ? "auto" : "76px"} minmax(0, 1fr)`, gridTemplateAreas: '"kpi" "centre"' }} data-scrolls={tpl.scrolls ? "true" : undefined}>
          {kpi}
          <div className="vt-bot-col vt-bot-area-centre">
            <div className="vt-bot-row" style={{ padding: "0 2px" }}><button type="button" className="vt-bot-wbtn" onClick={() => onZoom(flags.zoomed!)}>{ICONS.dock} back to the page</button><span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>Esc</span></div>
            {slot(flags.zoomed, { flex: "1 1 auto" })}
          </div>
        </div>
      ) : tpl.scrolls ? (
        // Tablet / phone: a prioritised stack that scrolls — the few things
        // that matter first, natural heights, never the desktop squeezed.
        <div className="vt-bot-stack" data-tier={tier}>
          {kpi}
          {hasState ? slot("state", { flex: "0 0 auto" }) : null}
          {slot("issues", { flex: "0 0 auto" })}
          {slot("model", { flex: "0 0 auto" })}
          {tier === "m" ? <div className="vt-bot-pair">{slot("stages", { flex: "0 0 auto" })}<div className="vt-bot-col">{slot("book", { flex: "0 0 auto" })}{slot("orders", { flex: "0 0 auto" })}</div></div> : <>{slot("book", { flex: "0 0 auto" })}{slot("stages", { flex: "0 0 auto" })}{slot("orders", { flex: "0 0 auto" })}</>}
          {slot("chart", { flex: "0 0 auto" })}
          {slot("timeline", { flex: "0 0 auto" })}
        </div>
      ) : (
        <div className="vt-bot-grid" style={gridStyle(tier)}>
          {kpi}
          <div className="vt-bot-col vt-bot-area-left">{slot("stages", { flex: "0 0 auto" })}{slot("issues", { flex: "1 1 auto" })}</div>
          <div className="vt-bot-col vt-bot-area-centre">{hasState ? slot("state", { flex: "0 0 auto" }) : null}{slot("model", { flex: "1 1 auto" })}</div>
          <div className="vt-bot-col vt-bot-area-right">{slot("chart", { flex: "1.05 1 0" })}{slot("book", { flex: "1.35 1 0" })}{slot("orders", { flex: "0.6 1 0" })}</div>
          <div className="vt-bot-col vt-bot-area-timeline">{slot("timeline", { flex: "1 1 auto" })}</div>
        </div>
      )}
      {flags.floating.map((id) => <FloatWin key={id} id={id} rect={rects[id] ?? { x: 80, y: 80, w: 760, h: 480 }} page={{ w: pageW, h: pageH }} onRect={(r) => setRects((rs) => ({ ...rs, [id]: r }))} onDock={() => onDock(id)} onZoom={() => onZoom(id)}>{panelEl(id, { flex: "1 1 auto" })}</FloatWin>)}
      {flags.hold ? <div className="vt-bot-hold-outline" aria-hidden="true" /> : null}
    </div>
  );
}

function FloatWin({ id, rect, page, onRect, onDock, onZoom, children }: { id: PanelId; rect: Rect; page: { w: number; h: number }; onRect: (r: Rect) => void; onDock: () => void; onZoom: () => void; children: ReactNode }) {
  const drag = useRef<{ mode: "move" | "size"; sx: number; sy: number; start: Rect } | null>(null);
  // A press on a title-bar BUTTON is not a drag: capturing the pointer to the
  // bar sent the button's click to the bar instead, so dock / zoom / close
  // never fired (field report 2026-10-20; bot-ui harness battery).
  const onDown = (mode: "move" | "size") => (e: React.PointerEvent) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { mode, sx: e.clientX, sy: e.clientY, start: rect };
  };
  const onMove = (e: React.PointerEvent) => {
    const g = drag.current; if (!g) return;
    const dx = e.clientX - g.sx, dy = e.clientY - g.sy;
    const next = g.mode === "move" ? { ...g.start, x: g.start.x + dx, y: g.start.y + dy } : { ...g.start, w: g.start.w + dx, h: g.start.h + dy };
    onRect(snapEdges(clampRect(next, page), page));
  };
  const onUp = () => { drag.current = null; };
  return (
    <div className="vt-bot-win" role="dialog" aria-label={`${TITLES[id]} window`} style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}>
      <div className="vt-bot-wh" onPointerDown={onDown("move")} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        <span className="vt-bot-ti">{TITLES[id]}</span>
        <span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>floating · ghost slot kept · Esc docks</span>
        <button type="button" className="vt-bot-wbtn" onClick={onDock}>{ICONS.dock} dock</button>
        <button type="button" className="vt-bot-wbtn" onClick={onZoom}>{ICONS.zoom} zoom</button>
        <button type="button" className="vt-bot-wbtn" aria-label="Close window" onClick={onDock}>{ICONS.close}</button>
      </div>
      <div className="vt-bot-wb">{children}</div>
      <div className="vt-bot-resize" aria-hidden="true" onPointerDown={onDown("size")} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} />
    </div>
  );
}
