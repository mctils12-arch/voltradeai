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
// 2026-11-06 (the port onto tonight's tip): what the old Analyze and Scanner
// pages showed and the first cut of this page had dropped is back — the live
// price bar (day range, volume, market cap, beta) in Stock · Fundamentals, the
// reason no spread is ranked (spreadsEmpty.ts, never "none cleared the bar"
// when the chain had no live quotes), the options disclaimer, the sector
// heatmap whenever the screener has room (zoomed or a window), and the look-up
// box in the zoomed view (a tool tab like Research had no way to enter a symbol).
// HEAVY LOADS (2026-11-06, keep the system alive: the server's heap is ~2.7 GB
// and it OOM-restarted tonight): a look-up runs ONE Python process, the shared
// /api/analyze. Research (python -m alphadesk), Smart Money + Structure
// (insights.py, one run) and the ETF builder each wait for their own
// "Load … for <SYM>" press, and a new look-up clears every press
// (trade/heavyLoads.ts). The Stock panel opens on Fundamentals (the analyze
// data already loaded), on Research only from the Research tab.
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { gridStyle } from "@/cockpit/zoneModel";
import { useCockpitLayout } from "@/cockpit/useCockpitLayout";
import { useTier } from "@/cockpit/useTier";
import { useNow } from "@/cockpit/useNow";
import { useFlip } from "@/cockpit/useFlip";
import { FloatWin } from "@/cockpit/FloatWin";
import { PanelBoundary } from "@/cockpit/PanelBoundary";
import { Panel, LookupBox, NotRecorded, ICONS } from "@/bot/panels";
import { spreadsEmptyMessage } from "@/lib/spreadsEmpty";
import { getActiveTicker, normalizeTicker, setActiveTicker, subscribeActiveTicker } from "@/lib/activeTicker";
import { resetTradeBackdrop, setTradeBackdrop } from "@/lib/tradeBackdrop";
// the hero tile states its PRICE's own as-of (2026-11-11): "alpaca · 2 s" read as fresh while the price was the previous close
import { priceAsOfLabel } from "@/lib/liveRefresh";
import {
  useAnalyze, AnalyzeEmptyState, RecommendationCard, VolSurfaceChart, VolMetricsPanel, EdgeFactorsPanel,
  SentimentPanel, EarningsIntelPanel, FundamentalsPanel, ValuationPanel, SpreadCard, LivePriceBar,
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
import { NO_REQUESTS, isRequested, requestLoad, type HeavyLoad, type HeavyRequests } from "./heavyLoads";

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

  const tpl = TRADE_TEMPLATES[tier];
  const lookupBox = <LookupBox value={lookup} onChange={setLookup} onSubmit={() => runLookup(lookup)} busy={q.isFetching} inputRef={lookupRef} />;

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

  // in-panel tabs (the Stock panel opens on Fundamentals, which reads the analyze data already loaded; on
  // Research from the Research tab)
  const [stockTab, setStockTab] = useState<"research" | "fundamentals" | "events">(tab === "research" ? "research" : "fundamentals");
  useEffect(() => { if (tab === "research") setStockTab("research"); }, [tab]);
  const [planTab, setPlanTab] = useState<"plan" | "taxes">(tab === "taxes" ? "taxes" : "plan");
  useEffect(() => { if (tab === "taxes") setPlanTab("taxes"); if (tab === "planner") setPlanTab("plan"); }, [tab]);
  // HEAVY LOADS (trade/heavyLoads.ts): research, insights and the ETF builder each run a Python process on the
  // server, so none starts on a look-up — each waits for its panel's press, for the symbol open at the press; a
  // new look-up clears them all
  const [heavy, setHeavy] = useState<HeavyRequests>(NO_REQUESTS);
  useEffect(() => { setHeavy(NO_REQUESTS); }, [sym]);
  const loadNow = useCallback((what: HeavyLoad) => { if (sym) setHeavy((h) => requestLoad(h, what, sym)); }, [sym]);
  /** a heavy panel before its press: what it loads, and the one control that loads it */
  const onRequest = (what: HeavyLoad, label: string, note: string) => (
    <div className="vt-bot-empty" data-load={what}>
      <span className="vt-bot-t3">{note}</span>
      <button type="button" className="vt-bot-wbtn" onClick={() => loadNow(what)}>{ICONS.arrow} {label}</button>
    </div>
  );
  const pick = useCallback((t: string) => { const s = normalizeTicker(t); if (s) setActiveTicker(s); }, []);
  /** the panel has room: it fills the page (zoomed) or is a window over it */
  const roomy = (id: TradePanelId) => L.flags.zoomed === id || L.flags.floating.includes(id);

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
    options: !sym ? <AnalyzeEmptyState /> : failed ?? (!data ? loading : (
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
            {data.top_spreads?.[0] ? <SpreadCard spread={data.top_spreads[0]} rank={1} /> : (() => {
              // why nothing is ranked (spreadsEmpty.ts): no live two-sided quotes, or no buildable spread
              const m = spreadsEmptyMessage(data.ticker, !!data.vol_surface?.length);
              return <div className="vt-bot-empty" data-testid="spreads-empty"><span className="vt-bot-t2">{m.title}</span><span className="vt-bot-t3">{m.body}</span></div>;
            })()}
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
        <p className="vt-bot-t3" style={{ fontSize: 11, margin: 0 }}>For educational purposes only. Not financial advice. Options trading involves substantial risk of loss. Always consult a licensed financial professional before trading.</p>
      </>
    )),
    stock: !sym ? needsSym("The stock") : (
      <>
        <Tabs value={stockTab} onChange={setStockTab} items={[{ id: "research", label: "Research" }, { id: "fundamentals", label: "Fundamentals" }, { id: "events", label: "Earnings & sentiment" }]} />
        {stockTab === "research" ? (isRequested(heavy, "research", sym) ? <ResearchBody ticker={sym} /> : onRequest("research", `Load research for ${sym}`, `The research verdict for ${sym} (six weighted pillars: fundamentals, valuation, supply and demand, market context, SEC filings, options) runs on the server on request; it takes several seconds.`)) : null}
        {stockTab === "fundamentals" ? (data ? <>{data.fundamentals ? <><LivePriceBar f={data.fundamentals} spot={data.spot} high52={data.high_52} low52={data.low_52} /><FundamentalsPanel f={data.fundamentals} /></> : <NotRecorded what="no fundamentals" />}{data.valuation ? <ValuationPanel valuation={data.valuation} /> : null}</> : failed ?? loading) : null}
        {stockTab === "events" ? (data ? <>{data.earnings_intel ? <EarningsIntelPanel intel={data.earnings_intel} /> : <NotRecorded what="no earnings history" />}{data.sentiment ? <SentimentPanel sentiment={data.sentiment} /> : <NotRecorded what="no sentiment read" />}</> : failed ?? loading) : null}
      </>
    ),
    smart: !sym ? needsSym("Smart money") : isRequested(heavy, "insights", sym) ? <InsightsView ticker={sym} filter="smart-money" /> : onRequest("insights", `Load smart money for ${sym}`, `Insiders, institutions and options flow for ${sym} load on request: one server run fills Smart Money and Structure.`),
    structure: !sym ? needsSym("Structure") : isRequested(heavy, "insights", sym) ? <InsightsView ticker={sym} filter="structure" /> : onRequest("insights", `Load structure for ${sym}`, `Float, short interest and support / resistance for ${sym} load on request: one server run fills Structure and Smart Money.`),
    // the sector heatmap needs room: dropped in the column slot, back when the screener is zoomed (the Scanner tab) or a window
    screener: <ScannerBody onSelectTicker={pick} compact={!roomy("screener")} />,
    watchlist: <WatchlistBody onSelectTicker={pick} authenticated={authenticated} />,
    news: <NewsBody ticker={sym} onSelectTicker={pick} />,
    plan: (
      <>
        <Tabs value={planTab} onChange={setPlanTab} items={[{ id: "plan", label: "Trade plan" }, { id: "taxes", label: "Taxes" }]} />
        {planTab === "plan" ? <PlannerBody ticker={sym} /> : <TaxesBody />}
      </>
    ),
    etf: !sym ? needsSym("The ETF builder") : isRequested(heavy, "etf", sym) ? <ETFBuilderView ticker={sym} /> : onRequest("etf", `Load ETF holdings for ${sym}`, `Holdings, drag and replication for ${sym} if it is an ETF. Loads on request: the fetch takes seconds.`),
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
    // the SLOT carries the column share and any cap (maxHeight); the panel fills its slot — a cap on
    // both left a hole under the panel (the bot page found the same, harness 2026-10-21/22)
    // the age is the analyze query's, so only the panels that read it show it (Smart Money and Structure read
    // insights, loaded on request by their own body: they claim no age rather than borrow this one)
    const common = { chrome: c, ageMs: id === "options" || id === "stock" ? ageMs : null, style: { ...extraStyle, flex: "1 1 auto", minHeight: 0, maxHeight: undefined } };
    const title = id === "options" && sym ? `${TRADE_TITLES[id]} · ${sym}` : TRADE_TITLES[id];
    // the look-up box lives in the hero panel's header on the desktop grid; the phone stack gives it its
    // own row above the panels (the header there has the touch-size controls and no room), and the zoomed
    // view carries it in its bar — a tool tab zooms a panel that needs a symbol (Research's Stock, the ETF
    // Builder), so the box must stay reachable there. One box in every layout.
    const status = id === "options" && !tpl.scrolls && !L.flags.zoomed ? lookupBox : undefined;
    // one body's render error stays in its panel (PanelBoundary); a new symbol retries
    return <Panel {...common} title={title} status={status} meta={metaFor(id)} scroll><PanelBoundary resetKey={sym ?? ""}>{body(id)}</PanelBoundary></Panel>;
  };
  const slot = (id: TradePanelId, extraStyle?: React.CSSProperties): ReactNode => L.flags.floating.includes(id)
    ? <div key={id} ref={L.slotRef(id)} className="vt-bot-ghost" role="button" tabIndex={0} onClick={() => L.onDock(id)} onKeyDown={(e) => { if (e.key === "Enter") L.onDock(id); }} style={extraStyle}>{TRADE_TITLES[id]} · open as a window · click to dock</div>
    : <div key={id} ref={L.slotRef(id)} style={{ display: "flex", flexDirection: "column", minHeight: 0, minWidth: 0, ...extraStyle }}>{panelEl(id, extraStyle)}</div>;

  const layoutKey = `${tier}|${L.flags.zoomed ?? ""}|${L.flags.floating.join(",")}`;
  useFlip(layoutKey, L.slots);

  const kpi = (
    <div className="vt-bot-kpi vt-bot-area-kpi" data-testid="trade-kpi">
      <Tile label={sym ? <>{sym}{data?.company_name && data.company_name !== sym ? <span style={{ textTransform: "none", letterSpacing: 0 }}> · {data.company_name}</span> : null}</> : "Symbol"} hero
        value={sym ? (data ? <>${fmtPx(data.spot)} <span className={`${(change ?? 0) < 0 ? "vt-bot-r" : "vt-bot-g"}`} style={{ fontSize: 14 }}>{fmtSignedPct(change)}</span></> : q.isLoading ? "…" : "—") : "—"}
        sub={sym ? (data ? (priceAsOfLabel(data) || `${data.data_source ?? "chain"} · ${ageMs !== null ? `${Math.round(ageMs / 1000)} s` : ""}`) : "type a symbol or press /") : "one symbol, every panel"} />
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
        // the strip at its natural height, as on the bot page: its tiles carry an 86 px minimum from the l tier up, which a fixed 76 px track clipped
        <div className="vt-bot-grid" style={{ gridTemplateColumns: "minmax(0, 1fr)", gridTemplateRows: "auto minmax(0, 1fr)", gridTemplateAreas: '"kpi" "centre"' }} data-scrolls={tpl.scrolls ? "true" : undefined}>
          {kpi}
          <div className="vt-bot-col vt-bot-area-centre">
            <div className="vt-bot-row" style={{ padding: "0 2px", flexWrap: "wrap" }}><button type="button" className="vt-bot-wbtn" onClick={() => L.onZoom(L.flags.zoomed!)}>{ICONS.dock} back to the page</button><span className="vt-bot-t3 vt-bot-m" style={{ fontSize: 11 }}>{TRADE_TITLES[L.flags.zoomed]} · Esc returns</span><span style={{ marginLeft: "auto" }}>{lookupBox}</span></div>
            {slot(L.flags.zoomed, { flex: "1 1 auto" })}
          </div>
        </div>
      ) : tpl.scrolls ? (
        <div className="vt-bot-stack" data-tier={tier}>
          {kpi}
          <div className="vt-bot-panel vt-bot-row" style={{ padding: "8px 12px", gap: 10 }}><span className="vt-bot-eb">Look up</span>{lookupBox}</div>
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
