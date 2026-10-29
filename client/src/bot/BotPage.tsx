// Bot page v2 — one full-bleed page under the site nav. The grid is the
// resting layout; the centre column holds whatever the bot is doing right
// now (pageState → director → zones); every panel can be held, pinned,
// opened as a window over the page (ghost slot kept) or zoomed; the view
// freezes, the bot never does. Layout motion runs on frameCore springs
// (Law I: one rAF loop; transforms only during motion).
// Brief: research/bot_cockpit_design_brief_v2.md.
// 2026-10-29: the panel engine (tiers, flags, float windows, pop-outs and
// their channel, hold snapshots, FLIP) is the shared client/src/cockpit/ —
// the Trade page runs the same one. This file keeps what is the bot's: the
// director, the derivations, live events, the focus symbol, the kill banner
// and the actions.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useCockpitLayout, useTier, useNow, useFlip, FloatWin, type ChannelMsg } from "@/cockpit";
import { useBotData, num, type BotData } from "./useBotData";
import { derivePageState, type AuditRow } from "./pageState";
import { deriveStages, deriveIssues, deriveDayReview, rowsToday, type DeriveInput } from "./derive";
import { Director } from "./director";
import { gridStyle, TEMPLATES, PANEL_IDS, type PanelId } from "./zoneModel";
import { Panel, KpiStrip, StagesBody, IssuesBody, ScanStripBody, DecisionBody, IncidentBody, DayReviewBody, ModelBody, ChartBody, BookBody, OrdersBody, TimelineBody, WatchBody, LookupBox, ICONS, type ActionKind } from "./panels";
import { watchModel } from "./watch";
import { Help } from "./help";
import { paneIdFromHash, parsePaneMsg, PANE_CHANNEL, PANE_PREFIX, PANE_WINDOW_PREFIX } from "./popout";
import { liveEvents, flashes, FRESH_MS, type Flash, type LiveEvent, type EventKind } from "./live";
import { parseLookupInput } from "./lookup";
import { fmtAge } from "./format";
import { ScoreBody } from "./scoreboard";
import { fmtDay } from "./scoreView";
import { ShadowBlock, ShadowChip } from "./shadow";

const TITLES: Record<PanelId, string> = { kpi: "Edge", stages: "Stages", issues: "Issues", state: "Now", watch: "Watch", model: "Model", chart: "Chart", book: "Book", orders: "Orders", timeline: "Today", score: "Daily vs SPY" };

export default function BotPage() {
  const d = useBotData();
  const now = useNow(1000);
  const { tier, w: pageW, h: pageH, touch } = useTier();
  const [replayAt, setReplayAt] = useState<number | null>(null);
  // the focus context (brief §6): the position the chart shows; a Book row
  // or a chart chip sets it
  const [focusSym, setFocusSym] = useState<string | null>(null);
  // pop-out windows (popout.ts): `pane` when THIS window is one panel popped
  // out of the page; the panels this page has sent to their own windows are
  // the engine's `popped` (their slots show a ghost until they come back)
  const pane = useMemo(() => paneIdFromHash(window.location.hash), []);
  const [busy, setBusy] = useState(false);
  const [actErr, setActErr] = useState<string | null>(null);
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
  // what the scan looked at, decided and refused (watch.ts) — replayed audit rows when replaying
  const watch = useMemo(() => watchModel({ lastScan, audit, threshold: num(status?.minScoreThreshold), nowMs: effNow }), [lastScan, audit, status?.minScoreThreshold, effNow]);
  // LIVE CHANGE (live.ts): the decisions that landed since the page's first
  // look (minus 15 min, so a fresh page shows what just happened), newest
  // first — the Now tile's latest line; a fresh LOUD one pulses the panel it
  // concerns. Replay shows the replayed moment's events.
  const firstLook = useRef(Date.now() - 15 * 60_000);
  const events: LiveEvent[] = useMemo(() => liveEvents(audit, replayAt !== null ? replayAt - 15 * 60_000 : firstLook.current), [audit, replayAt]);
  const pulses = useMemo(() => {
    const out: Partial<Record<PanelId, EventKind>> = {};
    for (const e of events) {
      if (!e.loud || now - e.t >= FRESH_MS) continue;
      const targets: PanelId[] = e.kind === "trade" || e.kind === "exit" ? ["book", "chart"] : e.kind === "queue" || e.kind === "refusal" ? ["watch"] : e.kind === "halt" || e.kind === "resume" || e.kind === "error" ? ["state", "stages"] : ["model"];
      for (const id of targets) if (!out[id]) out[id] = e.kind;
    }
    return out;
  }, [events, now]);
  // values that changed since the previous tick flash with their direction
  // the Edge tiles show the BROKER's expectancy and win rate per closed lot ([RULE-REVIEW] 2026-10-29, panels.tsx KpiStrip)
  const kpiVals = { expectancy: num(perf?.broker?.active?.expectancyPct), winRate: num(perf?.broker?.active?.winRate), vsSpy: num(d.benchmark.data?.since?.gapPp), equity: num(account?.equity) };
  const prevKpi = useRef<typeof kpiVals | null>(null);
  const kpiFlash = useRef<Partial<Record<keyof typeof kpiVals, Flash>>>({});
  kpiFlash.current = flashes(prevKpi.current, kpiVals, kpiFlash.current, now);
  prevKpi.current = kpiVals;
  const rowVals: Record<string, number | null> = {};
  for (const p of d.positions.data ?? []) rowVals[p.ticker] = num(p.pnlPct);
  const prevRows = useRef<Record<string, number | null> | null>(null);
  const rowFlash = useRef<Partial<Record<string, Flash>>>({});
  rowFlash.current = flashes(prevRows.current, rowVals, rowFlash.current, now);
  prevRows.current = rowVals;

  // ticks since hold: count data updates
  const updates = [d.status, d.account, d.positions, d.audit, d.performance, d.market, d.insight, d.lastScan, d.orders].reduce((a, q) => a + (q.dataUpdatedAt || 0), 0);
  const tickCount = useRef(0); const lastUpdates = useRef(updates);
  if (updates !== lastUpdates.current) { lastUpdates.current = updates; tickCount.current += 1; }

  // look-up (2026-10-22, human-asked) -> ticker dossier (2026-10-29, human-asked): the same
  // box now opens the dossier page (#/dossier/<SYM>), which folds the look-up card into its
  // stock layer alongside the market, sector, range, options and events layers; "/" still
  // focuses the box like a search field
  const [lookupInput, setLookupInput] = useState("");
  const lookupRef = useRef<HTMLInputElement | null>(null);
  const runLookup = useCallback((raw: string) => {
    const parsed = parseLookupInput(raw);
    if (!parsed) return;
    setLookupInput(parsed.symbol);
    window.location.hash = `#/dossier/${parsed.underlying}`;
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey && !(t instanceof HTMLInputElement) && !(t instanceof HTMLTextAreaElement) && !t?.isContentEditable) { e.preventDefault(); lookupRef.current?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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

  // the layout engine (client/src/cockpit): flags, float rects, pop-out
  // windows and their channel, slots, hold snapshots, the chrome handlers,
  // Esc. The focus symbol is the bot's message on that channel — shared with
  // every window of this page: a Book click in a pop-out charts the ticker on
  // the page, and the other way round
  const onPaneMsg = useCallback((m: ChannelMsg) => { const p = parsePaneMsg(m); if (p?.t === "focus") setFocusSym(p.sym); }, []);
  const layout = useCockpitLayout<PanelId>({ ids: PANEL_IDS, titles: TITLES, tier, touch, page: { w: pageW, h: pageH }, rootRef, paneId: pane, tickCount: tickCount.current, paneTitleSuffix: " · VolTrade bot", channelName: PANE_CHANNEL, paneHashPrefix: PANE_PREFIX, paneWindowPrefix: PANE_WINDOW_PREFIX, onMessage: onPaneMsg });
  const { flags, rects, popped, popNote, setPopNote, slotRef, onDock, onZoom, onPop, onRaise, onBringBack, onHoldAll, post } = layout;
  const shown = ready ? director.current.propose(derived.state, now, flags.hold) : "idle";
  const onFocus = useCallback((id: PanelId) => { onZoom(id); }, [onZoom]);
  const focus = useCallback((sym: string | null) => { setFocusSym(sym); post({ t: "focus", sym }); }, [post]);

  // bodies (fresh), then hold snapshots
  const tpl = TEMPLATES[tier];
  const ordersRoomy = flags.zoomed === "orders" || flags.floating.includes("orders") || pane === "orders";
  const ages = { status: d.status.dataUpdatedAt ? now - d.status.dataUpdatedAt : null, audit: d.audit.dataUpdatedAt ? now - d.audit.dataUpdatedAt : null, positions: d.positions.dataUpdatedAt ? now - d.positions.dataUpdatedAt : null, insight: d.insight.dataUpdatedAt ? now - d.insight.dataUpdatedAt : null, orders: d.orders.dataUpdatedAt ? now - d.orders.dataUpdatedAt : null, perf: d.performance.dataUpdatedAt ? now - d.performance.dataUpdatedAt : null, lastScan: d.lastScan.dataUpdatedAt ? now - d.lastScan.dataUpdatedAt : null, score: d.scoreboard.dataUpdatedAt ? now - d.scoreboard.dataUpdatedAt : null };
  const freshBody: Record<PanelId, ReactNode> = {
    kpi: null,
    stages: <StagesBody rows={stages} />,
    issues: <IssuesBody issues={issues} nowMs={effNow} onFocus={onFocus} onAct={act} busy={busy} />,
    state: shown === "scanning" ? <ScanStripBody elapsedMs={derived.scan.elapsedMs} lastScan={lastScan} /> : shown === "candidate" || shown === "placed" || shown === "exit" ? <DecisionBody lastScan={lastScan} audit={audit} nowMs={effNow} /> : shown === "incident" ? <IncidentBody state={derived} status={status} issues={issues} onAct={act} busy={busy} /> : shown === "closed" ? <DayReviewBody day={day} account={account} market={market} insight={insight} bench={d.benchmark.data} /> : null,
    watch: <WatchBody model={watch} focus={focusSym} onFocus={focus} nowMs={effNow} />,
    model: <ModelBody insight={insight} audit={audit} compact={shown !== "idle" || tier === "s"} />,
    chart: <ChartBody positions={d.positions.data} candidates={watch.candidates} focus={focusSym} onFocus={focus} paused={flags.held.includes("chart")} />,
    book: <BookBody positions={d.positions.data} focus={focusSym} onSelect={focus} sleeves={d.sleeves.data} rowFlash={rowFlash.current} nowMs={now} />,
    // entry shadow mode (shadow.tsx): the full block where Orders has room (zoomed, a window, its own
    // window) or folded in the phone/tablet stack; in the desktop grid only the header chip, so the
    // one-line Orders slot never grows and squeezes the chart
    orders: <>{ordersRoomy || tpl.scrolls ? <ShadowBlock status={status} mode={ordersRoomy ? "full" : "fold"} /> : null}<OrdersBody orders={d.orders.data} deploysToday={deploysToday} /></>,
    timeline: <TimelineBody audit={auditAll} nowMs={now} killSince={killSince} replayAt={replayAt} onReplay={setReplayAt} closed={market ? !market.isOpen : false} />,
    score: <ScoreBody q={d.scoreboard} nowMs={now} />,
  };
  // the hold ledger (engine): a newly held panel keeps this render's body
  const body = layout.heldBodies(freshBody, now);
  // a popped-out "Now" window between events: say so rather than show a bare frame
  if (pane === "state" && !freshBody.state) freshBody.state = <div className="vt-bot-empty"><span className="vt-bot-t3">idle · nothing in progress — this window shows the scan, the decision chain, a working order, an exit or an incident while one is running</span></div>;

  const stateTitle = shown === "scanning" ? "Scan · tier 2" : shown === "candidate" ? `Decision · ${((lastScan?.new_trades?.[0] as { ticker?: string } | undefined)?.ticker ?? lastScan?.top_10?.[0]?.ticker ?? "").toUpperCase()}` : shown === "placed" ? "Order working" : shown === "exit" ? "Exit" : shown === "incident" ? `Halted · ${derived.incident?.kind ?? ""} · ${derived.since ? fmtAge(effNow - Date.parse(derived.since)) : ""}` : shown === "closed" ? "Day review" : "Now";
  const hasState = shown !== "idle";
  // chart room (2026-10-21, "I'd rather have a larger chart"): while nothing is
  // working the Orders panel is one line at its natural height and the chart
  // takes the room; with real orders it grows to its rows' natural height
  // (OrdersBody shows at most 8). 2026-10-27 (audit D4): the old "0.65 1 0"
  // share had a 0 basis against the chart's auto basis, so with orders the
  // panel shrank to its header and clipped its rows (and, grow 0.65 inside its
  // own slot, left a strip of the slot blank). Unseen until /api/orders/open
  // was unwrapped and real orders reached this panel (harness bot @1440).
  const ordersFlex = "0 0 auto";
  const panelEl = (id: PanelId, extraStyle?: React.CSSProperties): ReactNode => {
    if (id === "kpi") return null;
    const c = layout.chrome(id, now);
    const common = { chrome: c, pulse: pulses[id], ageMs: id === "model" ? ages.insight : id === "chart" || id === "book" ? ages.positions : id === "orders" ? ages.orders : id === "watch" ? ages.lastScan : id === "score" ? ages.score : id === "stages" || id === "issues" || id === "state" || id === "timeline" ? ages.audit : ages.status } as const;
    switch (id) {
      case "stages": return <Panel {...common} title="Stages" help="stages" meta="last 10 expected ticks" style={{ flex: "0 0 auto", ...extraStyle }}>{body(id)}</Panel>;
      // the slot carries the column share and the cap (maxHeight); the panel
      // fills its slot — a cap on both sized the panel to 40 % of its own
      // wrapper and left a hole under it (harness 2026-10-22, same as watch)
      case "issues": return <Panel {...common} title="Issues" help="issues" meta={issues.length ? `${issues.length} standing` : undefined} scroll style={{ ...extraStyle, flex: "1 1 auto", minHeight: 0, maxHeight: undefined }}>{body(id)}{actErr ? <div className="vt-bot-r" style={{ fontSize: 11.5 }}>action failed · {actErr}</div> : null}</Panel>;
      case "state": return <Panel {...common} title={stateTitle} help={shown === "candidate" || shown === "placed" || shown === "exit" ? "decision" : "now"} status={shown === "incident" ? <span className="vt-bot-st vt-bot-r"><span className="vt-bot-dot" />first-out</span> : replayAt !== null ? <span className="vt-bot-st vt-bot-o"><span className="vt-bot-dot" />replay<Help k="replay" /></span> : undefined} style={{ flex: shown === "closed" ? "1 1 auto" : "0 0 auto", ...extraStyle }}>{body(id)}</Panel>;
      // the slot carries the column share (maxHeight); the panel fills its
      // slot — applying the cap to both halved the body to a strip (harness
      // 2026-10-21)
      case "watch": return <Panel {...common} title="Watch" help="watch" status={<LookupBox value={lookupInput} onChange={setLookupInput} onSubmit={() => runLookup(lookupInput)} busy={false} inputRef={lookupRef} />} meta={watch.candidates.length ? `${watch.candidates.length} candidates · ${watch.decided.length} decided · ${watch.refused.length} refused` : "what the last scan looked at"} scroll style={{ ...extraStyle, flex: "1 1 auto", minHeight: 0, maxHeight: undefined }}>{body(id)}</Panel>;
      case "model": return <Panel {...common} title="Model" help="model" meta={insight?.recent?.window ? `shadow · last ${insight.recent.window}` : undefined} style={{ flex: "1 1 auto", ...extraStyle }} scroll>{body(id)}</Panel>;
      case "chart": return <Panel {...common} title="Chart" help="chart" meta={d.positions.data?.length ? `${d.positions.data.length} positions · entry · stop · thirds · target` : undefined} style={{ flex: "2.4 1 0", ...extraStyle }}>{body(id)}</Panel>;
      case "book": return <Panel {...common} title={`Book${d.positions.data ? ` · ${d.positions.data.length}` : ""}`} help="book" meta={account ? `cash ${account.cash != null ? `$${Math.round(Number(account.cash)).toLocaleString("en-US")}` : "—"}` : undefined} scroll style={{ flex: "1.1 1 0", ...extraStyle }}>{body(id)}</Panel>;
      case "orders": return <Panel {...common} title="Orders" help="orders" status={ordersRoomy || tpl.scrolls ? undefined : <ShadowChip status={status} onOpen={() => onZoom("orders")} label="count" />} scroll={ordersRoomy} style={{ flex: ordersFlex, ...extraStyle }}>{body(id)}</Panel>;
      // the daily bot-vs-SPY scoreboard (2026-10-28, human-ordered): the slot carries the column share; the table scrolls inside
      case "score": return <Panel {...common} title="Daily vs SPY" help="scoreboard" meta={d.scoreboard.data?.summary ? `${d.scoreboard.data.rows.length} days · through ${fmtDay(d.scoreboard.data.summary.to)}` : "bot vs buy-and-hold SPY, per trading day"} scroll style={{ ...extraStyle, flex: "1 1 auto", minHeight: 0, maxHeight: undefined }}>{body(id)}</Panel>;
      case "timeline": return <Panel {...common} title={new Date(effNow).toLocaleDateString("en-US", { weekday: "short", day: "numeric", month: "short", timeZone: "America/New_York" })} help="timeline" status={replayAt !== null ? <span className="vt-bot-st vt-bot-o"><span className="vt-bot-dot" />replay<Help k="replay" /></span> : undefined} meta="RTH 09:30–16:00 ET · hatched = latched · drag to replay" style={{ height: tier === "s" ? 120 : "100%", ...extraStyle }}>{body(id)}</Panel>;
      default: return null;
    }
  };
  const slot = (id: PanelId, extraStyle?: React.CSSProperties): ReactNode => popped.includes(id)
    // in its own browser window: the ghost brings it forward, or back
    ? <div key={id} ref={slotRef(id)} className="vt-bot-ghost" data-popped="true" role="button" tabIndex={0} onClick={() => onRaise(id)} onKeyDown={(e) => { if (e.key === "Enter") onRaise(id); }} style={extraStyle}><span>{TITLES[id]} · in its own window · click to bring it forward</span><button type="button" className="vt-bot-wbtn" onClick={(e) => { e.stopPropagation(); onBringBack(id); }}>{ICONS.dock} bring back</button></div>
    : flags.floating.includes(id)
    ? <div key={id} ref={slotRef(id)} className="vt-bot-ghost" role="button" tabIndex={0} onClick={() => onDock(id)} onKeyDown={(e) => { if (e.key === "Enter") onDock(id); }} style={extraStyle}>{TITLES[id]} · open as a window · click to dock</div>
    // the slot's flex reaches the panel too: each panel carries its desktop
    // grid flex ("1.35 1 0" for the book), and in the phone/tablet stack —
    // natural heights, nothing to grow into — a 0 basis collapsed Chart,
    // Book and Orders to 0 px (bot-ui harness battery, 2026-10-20)
    : <div key={id} ref={slotRef(id)} style={{ display: "flex", flexDirection: "column", minHeight: 0, minWidth: 0, ...extraStyle }}>{panelEl(id, extraStyle)}</div>;

  const layoutKey = `${tier}|${shown}|${flags.zoomed ?? ""}|${flags.floating.join(",")}`;
  useFlip(layoutKey, layout.slots, { label: "bot-page-flip" });
  const kpi = <KpiStrip perf={perf} account={account} status={status} market={market} calendar={d.calendar.data} bench={d.benchmark.data} now={effNow} state={derived} hold={flags.hold} onHold={onHoldAll} flash={kpiFlash.current} events={events} onScoreboard={tpl.scrolls ? undefined : () => onZoom("score")} onShadow={() => onZoom("orders")} />;

  // a panel popped out into its own browser window (popout.ts): this window
  // IS the panel — no grid, and the KPI strip only when the strip itself was popped
  if (pane) {
    return (
      <div ref={rootRef} className="vt-bot-page" data-pane={pane} data-touch={touch ? "true" : undefined} data-state={shown} data-tier={tier} tabIndex={-1}>
        {pane === "kpi" ? kpi : panelEl(pane, { flex: "1 1 auto", maxHeight: "none" })}
      </div>
    );
  }
  return (
    <div ref={rootRef} className="vt-bot-page" data-touch={touch ? "true" : undefined} data-state={shown} data-tier={tier} tabIndex={-1}>
      {popNote ? <div className="vt-bot-popnote" role="status"><span>{popNote}</span><button type="button" className="vt-bot-wbtn" onClick={() => setPopNote(null)}>dismiss</button></div> : null}
      {status?.killSwitch ? <KillBanner days={killSince ? Math.floor((now - killSince) / 86_400_000) : null} onResume={() => void act("resume")} busy={busy} err={actErr} /> : null}
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
          {slot("chart", { flex: "0 0 auto" })}
          {slot("score", { flex: "0 0 auto" })}
          {slot("watch", { flex: "0 0 auto" })}
          {slot("model", { flex: "0 0 auto" })}
          {tier === "m" ? <div className="vt-bot-pair">{slot("stages", { flex: "0 0 auto" })}<div className="vt-bot-col">{slot("book", { flex: "0 0 auto" })}{slot("orders", { flex: "0 0 auto" })}</div></div> : <>{slot("book", { flex: "0 0 auto" })}{slot("stages", { flex: "0 0 auto" })}{slot("orders", { flex: "0 0 auto" })}</>}
          {slot("timeline", { flex: "0 0 auto" })}
        </div>
      ) : (
        <div className="vt-bot-grid" style={gridStyle(tier)}>
          {kpi}
          {/* the daily scoreboard has no desktop slot (the three columns are full): the vs-SPY tile's "daily" control zooms it */}
          {/* DEAD SPACE (2026-10-22, human-approved): no column leaves a blank — Issues takes only its content
              (capped, scrolls when long), the Book takes the rest of the left column, the chart the whole right
              column; Orders is one line while nothing works and grows back with real orders */}
          <div className="vt-bot-col vt-bot-area-left">{slot("stages", { flex: "0 0 auto" })}{slot("issues", { flex: "0 1 auto", maxHeight: "40%", minHeight: 0 })}{slot("book", { flex: "1 1 auto", minHeight: 120 })}</div>
          <div className="vt-bot-col vt-bot-area-centre">{hasState ? slot("state", { flex: "0 0 auto" }) : null}{slot("watch", { flex: "0 1 auto", maxHeight: hasState ? "36%" : "44%", minHeight: 136 })}{slot("model", { flex: "1 1 auto" })}</div>
          <div className="vt-bot-col vt-bot-area-right">{slot("chart", { flex: "1 1 auto" })}{slot("orders", { flex: ordersFlex })}</div>
          <div className="vt-bot-col vt-bot-area-timeline">{slot("timeline", { flex: "1 1 auto" })}</div>
        </div>
      )}
      {flags.floating.map((id) => <FloatWin key={id} title={TITLES[id]} rect={rects[id] ?? { x: 80, y: 80, w: 760, h: 480 }} page={{ w: pageW, h: pageH }} onRect={(r) => layout.setRect(id, r)} onDock={() => onDock(id)} onZoom={() => onZoom(id)} onPop={() => onPop(id)}>{panelEl(id, { flex: "1 1 auto" })}</FloatWin>)}
      {flags.hold ? <div className="vt-bot-hold-outline" aria-hidden="true" /> : null}
    </div>
  );
}

function KillBanner({ days, onResume, busy, err }: { days: number | null; onResume: () => void; busy: boolean; err: string | null }) {
  const [confirm, setConfirm] = useState(false);
  return (
    <div className="vt-bot-killbanner" role="alert">
      <div className="vt-bot-killbanner-body">
        <span className="vt-bot-killbanner-icon">&#9632;</span>
        <div className="vt-bot-killbanner-text">
          <strong>Kill switch latched{days !== null && days > 0 ? ` for ${days} day${days !== 1 ? "s" : ""}` : ""}</strong>
          <span>The bot is not scanning, trading, or managing exits. Click Resume to clear the latch and re-activate.</span>
        </div>
        <div className="vt-bot-killbanner-act">
          {confirm ? (
            <>
              <span style={{ fontSize: 11.5 }}>Clears the latch and re-starts the loop. The rule re-latches at the same drawdown line.</span>
              <button type="button" data-primary="true" disabled={busy} onClick={() => { setConfirm(false); onResume(); }}>Resume now</button>
              <button type="button" onClick={() => setConfirm(false)}>cancel</button>
            </>
          ) : (
            <button type="button" data-primary="true" onClick={() => setConfirm(true)}>Resume</button>
          )}
        </div>
      </div>
      {err ? <div className="vt-bot-killbanner-err">{err}</div> : null}
    </div>
  );
}
