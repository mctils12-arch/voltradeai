// Trade cockpit (2026-10-29, human: "it needs to look like the bot UI but have
// the functionality of what's there and no duplicates"). One full-bleed page
// under the site nav on the SAME engine as the bot page (client/src/cockpit/:
// grid per viewport tier, panel chrome, hold / pin / float / zoom, FLIP on
// frameCore) and the same moving world map behind it (TradeBackdrop).
//
// One look-up box owns the symbol (lib/activeTicker); every panel reads it.
// The panels are the existing features, each exactly once:
//   kpi        the symbol's numbers (price, vol, VRP, flow, verdict, range)
//   options    Options & Volatility (analyze.tsx's pieces, no page of its own)
//   stock      Research verdict · fundamentals · earnings & sentiment
//   smart      Smart Money (InsightsView)        structure  Structure (InsightsView)
//   screener   the one market scanner           watchlist  the tracked names
//   news       headlines, filtered to the symbol plan       Trade plan · Taxes
//   etf        ETF Builder, loaded on demand (its fetch costs seconds)
// The old Trade tabs stay as hash routes and phone tabs: a tool tab zooms its
// panel (zones.ts panelForTab); the analyze sections pick a panel.
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { gridStyle } from "@/cockpit/zoneModel";
import { useCockpitLayout } from "@/cockpit/useCockpitLayout";
import { useTier } from "@/cockpit/useTier";
import { useNow } from "@/cockpit/useNow";
import { useFlip } from "@/cockpit/useFlip";
import { FloatWin } from "@/cockpit/FloatWin";
import { Panel, LookupBox, NotRecorded, ICONS } from "@/bot/panels";
import { getActiveTicker, normalizeTicker, setActiveTicker, subscribeActiveTicker } from "@/lib/activeTicker";
import { resetTradeBackdrop, setTradeBackdrop } from "@/lib/tradeBackdrop";
import {
  useAnalyze, AnalyzeEmptyState, RecommendationCard, VolSurfaceChart, VolMetricsPanel, EdgeFactorsPanel,
  SentimentPanel, EarningsIntelPanel, FundamentalsPanel, ValuationPanel, SpreadCard,
} from "@/pages/analyze";
import InsightsView from "@/pages/InsightsView";
import ETFBuilderView from "@/pages/ETFBuilderView";
import { ScannerBody } from "@/pages/scanner";
import { NewsBody } from "@/pages/news";
import { WatchlistBody } from "@/pages/watchlist";
import { ResearchBody } from "@/pages/research";
import { PlannerBody } from "@/pages/planner";
import { TaxesBody } from "@/pages/taxes";
import { TRADE_PANEL_IDS, TRADE_TEMPLATES, TRADE_TITLES, panelForTab, type AnalyzeSectionId, type TradePanelId, type TradeTab } from "./zones";

const MINUS = "−";
const fmtPx = (n: number | null | undefined, d = 2) => (n === null || n === undefined || !Number.isFinite(n) ? "—" : n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const fmtSignedPct = (n: number | null | undefined, d = 2) => (n === null || n === undefined || !Number.isFinite(n) ? "—" : `${n < 0 ? MINUS : "+"}${Math.abs(n).toFixed(d)}%`);

/** Small in-panel tab row (the cockpit vocabulary has none; .vt-bot-tabs in index.css). */
function Tabs<T extends string>({ value, onChange, items }: { value: T; onChange: (v: T) => void; items: readonly { id: T; label: string }[] }) {
  return (
    <div className="vt-bot-tabs" role="tablist">
      {items.map((it) => (
        <button key={it.id} type="button" role="tab" className="vt-bot-tab" aria-selected={value === it.id} data-active={value === it.id ? "true" : undefined} onClick={() => onChange(it.id)}>{it.label}</button>
      ))}
    </div>
  );
}

function Tile({ label, value, sub, hero, cls }: { label: ReactNode; value: ReactNode; sub?: ReactNode; hero?: boolean; cls?: string }) {
  return (
    <div className="vt-bot-panel">
      <div style={{ minWidth: 0, flex: "1 1 auto" }}>
        <div className="vt-bot-eb">{label}</div>
        <div className={`${hero ? "vt-bot-hero" : "vt-bot-kpiv"} ${cls ?? ""}`}>{value}</div>
        {sub ? <div className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11, lineHeight: 1.35 }}>{sub}</div> : null}
      </div>
    </div>
  );
}

export default function TradePage({ tab, section, authenticated, isOwner }: { tab: TradeTab; section: AnalyzeSectionId; authenticated: boolean; isOwner: boolean }) {
  const sym = useSyncExternalStore(subscribeActiveTicker, getActiveTicker, getActiveTicker);
  const q = useAnalyze(sym);
  const data = q.data && !q.data.error ? q.data : undefined;
  const now = useNow(1000);
  const { tier, w: pageW, h: pageH, touch } = useTier();
  const rootRef = useRef<HTMLDivElement>(null);

  // the moving map behind the page (TradeBackdrop, mounted by the shell):
  // burst while fetching, beacon on the symbol once loaded, idle on leave
  useEffect(() => { setTradeBackdrop({ isLoading: q.isLoading, hasData: !!data, ticker: sym }); }, [q.isLoading, data, sym]);
  useEffect(() => () => { resetTradeBackdrop(); }, []);

  // one look-up box; "/" focuses it like the bot page
  const [lookup, setLookup] = useState(sym ?? "");
  const lookupRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => { setLookup(sym ?? ""); }, [sym]);
  const runLookup = useCallback((raw: string) => { const t = normalizeTicker(raw); if (t) setActiveTicker(t); }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey && !(t instanceof HTMLInputElement) && !(t instanceof HTMLTextAreaElement) && !t?.isContentEditable) { e.preventDefault(); lookupRef.current?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // data ticks: the analyze query is the page's pulse (bodies fetch on their own)
  const tickCount = useRef(0); const lastUpdated = useRef(q.dataUpdatedAt);
  if (q.dataUpdatedAt !== lastUpdated.current) { lastUpdated.current = q.dataUpdatedAt; tickCount.current += 1; }

  const L = useCockpitLayout<TradePanelId>({
    ids: TRADE_PANEL_IDS, titles: TRADE_TITLES, tier, touch, page: { w: pageW, h: pageH }, rootRef,
    paneId: null, tickCount: tickCount.current,
  });

  // the old tabs: a tool tab zooms its panel, an analyze section picks one
  useEffect(() => {
    const { panel, zoom } = panelForTab(tab, section);
    L.setFlags((f) => ({ ...f, zoomed: zoom && panel ? panel : null, floating: zoom && panel ? f.floating.filter((x) => x !== panel) : f.floating }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, section]);

  // in-panel tabs
  const [stockTab, setStockTab] = useState<"research" | "fundamentals" | "events">("research");
  const [planTab, setPlanTab] = useState<"plan" | "taxes">(tab === "taxes" ? "taxes" : "plan");
  useEffect(() => { if (tab === "taxes") setPlanTab("taxes"); if (tab === "planner") setPlanTab("plan"); }, [tab]);
  const [etfOpen, setEtfOpen] = useState(section === "etf-builder");
  useEffect(() => { if (section === "etf-builder") setEtfOpen(true); }, [section]);
  const pick = useCallback((t: string) => { const s = normalizeTicker(t); if (s) setActiveTicker(s); }, []);

  const ageMs = q.dataUpdatedAt ? now - q.dataUpdatedAt : null;
  const rec = data?.recommendation;
  const ef = data?.edge_factors;
  const atmIv = data?.atm_iv ?? null;
  const rv20 = data?.rv20 ?? null;
  const vrp = data?.vrp ?? null;
  const vrpWord = data?.vrp_regime ?? null;
  const vrpCls = vrpWord === "high" ? "vt-bot-g" : vrpWord === "low" ? "vt-bot-r" : "vt-bot-o";
  const change = data?.price_change_pct ?? null;
  const hi52 = data?.high_52 ?? null, lo52 = data?.low_52 ?? null;
  const range = data && hi52 !== null && lo52 !== null && hi52 > lo52 ? Math.max(0, Math.min(100, ((data.spot - lo52) / (hi52 - lo52)) * 100)) : null;

  const needsSym = (what: string) => <div className="vt-bot-empty"><span className="vt-bot-t3">{what} shows once a symbol is open · type one in the look-up box or press "/"</span></div>;
  const loading = <div className="vt-bot-empty"><span className="vt-bot-t3">loading {sym} …</span></div>;
  const failed = q.isError || q.data?.error ? <div className="vt-bot-empty"><span className="vt-bot-r">{q.data?.error ?? (q.error instanceof Error ? q.error.message : "could not load")}</span></div> : null;

  const freshBody: Record<TradePanelId, ReactNode> = {
    kpi: null,
    options: !sym ? <AnalyzeEmptyState section="options" /> : failed ?? (!data ? loading : (
      <>
        {data.recommendation ? <RecommendationCard rec={data.recommendation} ticker={data.ticker} /> : <NotRecorded what="no recommendation in this read" />}
        <div className="vt-bot-hr" />
        <div className="vt-bot-row" style={{ alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 260px", minWidth: 0 }}>
            <div className="vt-bot-eb">Volatility cone · realized vs implied</div>
            <div className="vt-bot-kv">
              <span className="k">10-day RV</span><span className="v">{fmtPx(data.rv10, 1)}%</span>
              <span className="k">20-day RV</span><span className="v">{fmtPx(data.rv20, 1)}%</span>
              <span className="k">30-day RV</span><span className="v">{fmtPx(data.rv30, 1)}%</span>
              <span className="k">ATM IV</span><span className={`v ${vrpCls}`}>{fmtPx(data.atm_iv, 1)}%</span>
            </div>
            {data.vol_surface?.length ? <VolSurfaceChart data={data.vol_surface} /> : null}
          </div>
          <div style={{ flex: "1 1 260px", minWidth: 0 }}>
            <div className="vt-bot-eb">Best opportunity · {data.spread_count ?? data.top_spreads?.length ?? 0} spreads scored</div>
            {data.top_spreads?.[0] ? <SpreadCard spread={data.top_spreads[0]} rank={1} /> : <NotRecorded what="no spread cleared the bar" />}
          </div>
        </div>
        {data.vrp_signal ? <div className="vt-bot-t2" style={{ fontSize: 12 }}>{data.vrp_signal}</div> : null}
        <div className="vt-bot-hr" />
        {ef ? <EdgeFactorsPanel ef={ef} spot={data.spot} /> : null}
        {data.vol_metrics ? <VolMetricsPanel metrics={data.vol_metrics} skew={data.skew} atm_iv={data.atm_iv} rv20={data.rv20} /> : null}
        {(data.top_spreads?.length ?? 0) > 1 ? (
          <>
            <div className="vt-bot-eb">All ranked spreads</div>
            {data.top_spreads.slice(1).map((s, i) => <SpreadCard key={i + 1} spread={s} rank={i + 2} />)}
          </>
        ) : null}
      </>
    )),
    stock: !sym ? needsSym("The stock") : (
      <>
        <Tabs value={stockTab} onChange={setStockTab} items={[{ id: "research", label: "Research" }, { id: "fundamentals", label: "Fundamentals" }, { id: "events", label: "Earnings & sentiment" }]} />
        {stockTab === "research" ? <ResearchBody ticker={sym} /> : null}
        {stockTab === "fundamentals" ? (data ? <>{data.fundamentals ? <FundamentalsPanel f={data.fundamentals} /> : <NotRecorded what="no fundamentals" />}{data.valuation ? <ValuationPanel valuation={data.valuation} /> : null}</> : failed ?? loading) : null}
        {stockTab === "events" ? (data ? <>{data.earnings_intel ? <EarningsIntelPanel intel={data.earnings_intel} /> : <NotRecorded what="no earnings history" />}{data.sentiment ? <SentimentPanel sentiment={data.sentiment} /> : <NotRecorded what="no sentiment read" />}</> : failed ?? loading) : null}
      </>
    ),
    smart: !sym ? needsSym("Smart money") : <InsightsView ticker={sym} filter="smart-money" />,
    structure: !sym ? needsSym("Structure") : <InsightsView ticker={sym} filter="structure" />,
    screener: <ScannerBody onSelectTicker={pick} compact />,
    watchlist: <WatchlistBody onSelectTicker={pick} authenticated={authenticated} />,
    news: <NewsBody ticker={sym} onSelectTicker={pick} />,
    plan: (
      <>
        <Tabs value={planTab} onChange={setPlanTab} items={[{ id: "plan", label: "Trade plan" }, { id: "taxes", label: "Taxes" }]} />
        {planTab === "plan" ? <PlannerBody ticker={sym} /> : <TaxesBody />}
      </>
    ),
    etf: !sym ? needsSym("The ETF builder") : etfOpen ? <ETFBuilderView ticker={sym} /> : (
      <div className="vt-bot-empty">
        <span className="vt-bot-t3">Holdings, drag and replication for {sym} if it is an ETF. Loads on request: the fetch takes seconds.</span>
        <button type="button" className="vt-bot-wbtn" onClick={() => setEtfOpen(true)}>{ICONS.arrow} load {sym}</button>
      </div>
    ),
  };
  const body = L.heldBodies(freshBody, now);

  const metaFor = (id: TradePanelId): ReactNode => {
    if (!data) return undefined;
    if (id === "options") return `${data.data_source ?? "chain"} · ${data.spread_count ?? data.top_spreads?.length ?? 0} spreads`;
    if (id === "stock") return data.company_name && data.company_name !== data.ticker ? data.company_name : undefined;
    return undefined;
  };
  const panelEl = (id: TradePanelId, extraStyle?: React.CSSProperties): ReactNode => {
    if (id === "kpi") return null;
    // no pane route for the Trade world yet (the shell renders #/bot/pane/* only), so no pop-out control
    const c = { ...L.chrome(id, now), canPop: false };
    const common = { chrome: c, ageMs: id === "options" || id === "stock" || id === "smart" || id === "structure" ? ageMs : null, style: { flex: "1 1 auto", ...extraStyle } };
    const title = id === "options" && sym ? `${TRADE_TITLES[id]} · ${sym}` : TRADE_TITLES[id];
    const status = id === "options" ? <LookupBox value={lookup} onChange={setLookup} onSubmit={() => runLookup(lookup)} busy={q.isFetching} inputRef={lookupRef} /> : undefined;
    return <Panel {...common} title={title} status={status} meta={metaFor(id)} scroll>{body(id)}</Panel>;
  };
  const slot = (id: TradePanelId, extraStyle?: React.CSSProperties): ReactNode => L.flags.floating.includes(id)
    ? <div key={id} ref={L.slotRef(id)} className="vt-bot-ghost" role="button" tabIndex={0} onClick={() => L.onDock(id)} onKeyDown={(e) => { if (e.key === "Enter") L.onDock(id); }} style={extraStyle}>{TRADE_TITLES[id]} · open as a window · click to dock</div>
    : <div key={id} ref={L.slotRef(id)} style={{ display: "flex", flexDirection: "column", minHeight: 0, minWidth: 0, ...extraStyle }}>{panelEl(id, extraStyle)}</div>;

  const tpl = TRADE_TEMPLATES[tier];
  const layoutKey = `${tier}|${L.flags.zoomed ?? ""}|${L.flags.floating.join(",")}`;
  useFlip(layoutKey, L.slots);

  const kpi = (
    <div className="vt-bot-kpi vt-bot-area-kpi" data-testid="trade-kpi">
      <Tile label={sym ? <>{sym}{data?.company_name && data.company_name !== sym ? <span style={{ textTransform: "none", letterSpacing: 0 }}> · {data.company_name}</span> : null}</> : "Symbol"} hero
        value={sym ? (data ? <>${fmtPx(data.spot)} <span className={`${(change ?? 0) < 0 ? "vt-bot-r" : "vt-bot-g"}`} style={{ fontSize: 14 }}>{fmtSignedPct(change)}</span></> : q.isLoading ? "…" : "—") : "—"}
        sub={sym ? (data ? `${data.data_source ?? "chain"} · ${ageMs !== null ? `${Math.round(ageMs / 1000)} s` : ""}` : "type a symbol or press /") : "one symbol, every panel"} />
      <Tile label="ATM IV · 20-day RV" value={data ? <>{fmtPx(atmIv, 1)}% <span className="vt-bot-t3" style={{ fontSize: 14 }}>/ {fmtPx(rv20, 1)}%</span></> : "—"} sub={data ? "implied vs realized, 30-day basis" : undefined} />
      <Tile label="Vol risk premium" value={data ? <span className={vrpCls}>{vrp !== null ? `${vrp < 0 ? MINUS : "+"}${Math.abs(vrp).toFixed(1)}` : "—"}</span> : "—"} sub={data ? `${vrpWord ?? "—"} · skew ${fmtPx(data.skew, 1)}` : undefined} />
      <Tile label="Put / call" value={ef ? fmtPx(ef.put_call_ratio) : "—"} sub={ef ? `${ef.put_call_signal ?? ""}${ef.gamma_pin ? ` · pin ${ef.gamma_pin}` : ""}` : undefined} />
      <Tile label="Verdict" value={rec ? <span className={rec.action === "WAIT" ? "vt-bot-o" : "vt-bot-g"}>{rec.action}</span> : "—"} sub={rec ? `${rec.signal ?? ""} · ${rec.horizon ?? ""}` : undefined} />
      <Tile label="52-week range" value={data ? <>{fmtPx(data.low_52, 0)} <span className="vt-bot-t3" style={{ fontSize: 13 }}>→</span> {fmtPx(data.high_52, 0)}</> : "—"}
        sub={range !== null ? <span className="vt-bot-bar" style={{ display: "block", marginTop: 6 }} title={`${range.toFixed(0)}% of the way from the low to the high`}><i style={{ width: `${range}%`, background: "var(--accent)" }} /></span> : undefined} />
    </div>
  );

  return (
    <div ref={rootRef} className="vt-bot-page" data-world="trade" data-touch={touch ? "true" : undefined} data-tier={tier} tabIndex={-1} data-testid="trade-cockpit">
      {L.flags.zoomed ? (
        <div className="vt-bot-grid" style={{ gridTemplateColumns: "minmax(0, 1fr)", gridTemplateRows: `${tier === "s" ? "auto" : "76px"} minmax(0, 1fr)`, gridTemplateAreas: '"kpi" "centre"' }} data-scrolls={tpl.scrolls ? "true" : undefined}>
          {kpi}
          <div className="vt-bot-col vt-bot-area-centre">
            <div className="vt-bot-row" style={{ padding: "0 2px" }}><button type="button" className="vt-bot-wbtn" onClick={() => L.onZoom(L.flags.zoomed!)}>{ICONS.dock} back to the page</button><span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>{TRADE_TITLES[L.flags.zoomed]} · Esc returns</span></div>
            {slot(L.flags.zoomed, { flex: "1 1 auto" })}
          </div>
        </div>
      ) : tpl.scrolls ? (
        <div className="vt-bot-stack" data-tier={tier}>
          {kpi}
          {slot("options", { flex: "0 0 auto" })}
          {slot("stock", { flex: "0 0 auto" })}
          {tier === "m" ? <div className="vt-bot-pair">{slot("smart", { flex: "0 0 auto" })}{slot("structure", { flex: "0 0 auto" })}</div> : <>{slot("smart", { flex: "0 0 auto" })}{slot("structure", { flex: "0 0 auto" })}</>}
          {slot("screener", { flex: "0 0 auto" })}
          {slot("watchlist", { flex: "0 0 auto" })}
          {slot("news", { flex: "0 0 auto" })}
          {slot("plan", { flex: "0 0 auto" })}
          {slot("etf", { flex: "0 0 auto" })}
        </div>
      ) : (
        <div className="vt-bot-grid" style={gridStyle(tpl)}>
          {kpi}
          <div className="vt-bot-col vt-bot-area-left">{slot("screener", { flex: "1 1 auto", minHeight: 160 })}{slot("watchlist", { flex: "0 1 auto", maxHeight: "30%", minHeight: 120 })}{slot("news", { flex: "0 1 auto", maxHeight: "32%", minHeight: 120 })}</div>
          <div className="vt-bot-col vt-bot-area-centre">{slot("options", { flex: "1.5 1 0", minHeight: 240 })}{slot("stock", { flex: "1 1 0", minHeight: 180 })}</div>
          <div className="vt-bot-col vt-bot-area-right">{slot("smart", { flex: "1 1 0", minHeight: 140 })}{slot("structure", { flex: "1 1 0", minHeight: 140 })}{slot("plan", { flex: "0 1 auto", maxHeight: "34%", minHeight: 120 })}{slot("etf", { flex: "0 0 auto" })}</div>
        </div>
      )}
      {L.flags.floating.map((id) => <FloatWin key={id} title={TRADE_TITLES[id]} rect={L.rects[id] ?? { x: 80, y: 80, w: 760, h: 480 }} page={{ w: pageW, h: pageH }} onRect={(r) => L.setRect(id, r)} onDock={() => L.onDock(id)} onZoom={() => L.onZoom(id)} onPop={() => L.onDock(id)}>{panelEl(id, { flex: "1 1 auto", maxHeight: "none" })}</FloatWin>)}
      {isOwner && sym ? <a className="vt-bot-wbtn vt-trade-dossier" href={`#/dossier/${sym}`} title="The owner's dossier: market → sector → stock → options → events, with the bot's own score">{ICONS.arrow} dossier · {sym}</a> : null}
    </div>
  );
}
