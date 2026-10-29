// Trade cockpit — the panel vocabulary and the zone templates per viewport
// tier (2026-10-29, human: "it needs to look like the bot UI but have the
// functionality of what's there and no duplicates"). Pure (tested). The
// engine (hold / pin / float / zoom / pop-out, FLIP) is the shared one in
// client/src/cockpit/; this file only names the Trade panels and where
// they sit, the way client/src/bot/zoneModel.ts does for the bot page.
import type { Template, Tier } from "@/cockpit/zoneModel";

/** One panel per Trade feature. Everything the old seven tabs showed is
 *  here once: no second scanner, no second ticker input, no second insider
 *  pipeline on the page. */
export type TradePanelId =
  | "kpi"        // the symbol's numbers strip (price, vol, VRP, flow, verdict, range)
  | "options"    // Options & Volatility: recommendation, cone, spreads, metrics, edge factors
  | "stock"      // the stock: Research verdict · fundamentals · earnings & sentiment
  | "smart"      // Smart Money: insiders, institutions, flow
  | "structure"  // Structure: float, short interest, support/resistance
  | "screener"   // the one market screener (scanner.tsx)
  | "watchlist"  // the user's tracked names
  | "news"       // headlines, filtered to the symbol when one is open
  | "plan"       // Trade plan · Taxes
  | "etf";       // ETF Builder, loaded on demand (its fetch is heavy)

export const TRADE_PANEL_IDS: readonly TradePanelId[] = ["kpi", "options", "stock", "smart", "structure", "screener", "watchlist", "news", "plan", "etf"];

export const TRADE_TITLES: Record<TradePanelId, string> = {
  kpi: "Symbol", options: "Options & Volatility", stock: "Stock", smart: "Smart Money", structure: "Structure",
  screener: "Screener", watchlist: "Watchlist", news: "News", plan: "Plan", etf: "ETF Builder",
};

/** grid-template-* per tier. Areas: kpi / left (screener, watchlist, news) /
 *  centre (options, stock) / right (smart money, structure, plan, ETF). The
 *  desktop tiers never scroll the page; panels scroll inside. */
export const TRADE_TEMPLATES: Record<Tier, Template> = {
  xl: { columns: "380px minmax(0, 1fr) 460px", rows: "88px minmax(0, 1fr)", areas: ["kpi kpi kpi", "left centre right"], scrolls: false },
  l:  { columns: "330px minmax(0, 1fr) 390px", rows: "86px minmax(0, 1fr)", areas: ["kpi kpi kpi", "left centre right"], scrolls: false },
  m:  { columns: "minmax(0, 1fr) 320px", rows: "auto minmax(480px, 1fr) auto", areas: ["kpi kpi", "centre right", "left left"], scrolls: true },
  s:  { columns: "minmax(0, 1fr)", rows: "auto", areas: ["kpi", "centre", "right", "left"], scrolls: true },
};

/** The old Trade tabs (still the hash routes and the phone More sheet) map
 *  onto the cockpit: a tool tab zooms its panel, the analyze sections pick a
 *  panel without zooming (the whole page is the analyze view). */
export type TradeTab = "analyze" | "research" | "scanner" | "news" | "watchlist" | "planner" | "taxes";
export type AnalyzeSectionId = "options" | "smart-money" | "structure" | "etf-builder";

export function panelForTab(tab: TradeTab, section: AnalyzeSectionId): { panel: TradePanelId | null; zoom: boolean } {
  switch (tab) {
    case "research":  return { panel: "stock", zoom: true };
    case "scanner":   return { panel: "screener", zoom: true };
    case "news":      return { panel: "news", zoom: true };
    case "watchlist": return { panel: "watchlist", zoom: true };
    case "planner":   return { panel: "plan", zoom: true };
    case "taxes":     return { panel: "plan", zoom: true };
    case "analyze":
      if (section === "smart-money") return { panel: "smart", zoom: false };
      if (section === "structure") return { panel: "structure", zoom: false };
      if (section === "etf-builder") return { panel: "etf", zoom: true };
      return { panel: null, zoom: false };
  }
}

/** Every panel but the strip must have a slot in every tier's areas. */
export function templateCoversAll(t: Template, columns: Record<string, readonly TradePanelId[]>): TradePanelId[] {
  const areas = new Set(t.areas.join(" ").split(/\s+/));
  const placed = new Set<TradePanelId>();
  for (const [area, ids] of Object.entries(columns)) if (areas.has(area)) for (const id of ids) placed.add(id);
  return TRADE_PANEL_IDS.filter((id) => id !== "kpi" && !placed.has(id));
}

/** Which panels each column holds, in order. */
export const TRADE_COLUMNS: Record<"left" | "centre" | "right", readonly TradePanelId[]> = {
  left: ["screener", "watchlist", "news"],
  centre: ["options", "stock"],
  right: ["smart", "structure", "plan", "etf"],
};
