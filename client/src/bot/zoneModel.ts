// Bot page v2 — the bot's binding of the cockpit zone vocabulary. Pure (tested).
// The grid is the resting layout; the self-arranging part is what the centre
// column holds (pageState) and the per-panel flags (held / pinned / floating /
// zoomed) the user sets. 2026-10-29 (ported 2026-11-06 onto the Overview ·
// Cockpit page): the generic engine — tiers, flags, float geometry, grid
// styles — is client/src/cockpit/zoneModel.ts so the Trade page runs the same
// one; what is the bot's stays here (its panel ids, both views' grid templates
// per tier, the typed flag set) and every export this module had is kept
// (pinned by zoneModel.test.ts), so no importer changes.
import { gridStyle as gridStyleOf, emptyFlags, type LayoutFlags as CockpitFlags, type Template, type Tier } from "../cockpit/zoneModel";
export { tierFor, toggleIn, isPinned, clampRect, initialFloatRect, snapEdges, type Tier, type Rect, type Template } from "../cockpit/zoneModel";

// perf / why / picks: the OVERVIEW panels (2026-11-06, human refocus spec §8 — the after-tax chart vs SPY, the
// "why are we losing money" attribution, a plain-English card per pick)
export type PanelId = "kpi" | "stages" | "issues" | "state" | "watch" | "model" | "chart" | "book" | "orders" | "timeline" | "score" | "perf" | "why" | "picks";
export const PANEL_IDS: readonly PanelId[] = ["kpi", "stages", "issues", "state", "watch", "model", "chart", "book", "orders", "timeline", "score", "perf", "why", "picks"];

/** KPI row: label + value + three 11 px lines (the vs-SPY tile carries four since the after-tax leg, 2026-10-22) need ~84 px uncut; 88/86 keep the strip from clipping its first and last line.
/** DEAD SPACE (2026-10-22, human-approved): the left column holds Stages, Issues (content height, capped) and the BOOK, which takes the rest of it — 40 px wider for the tape rows; the right column is the chart alone (Orders is one line while nothing works).
/** grid-template-* per tier. Areas: kpi / left (stages + issues) / centre
 *  (state element + model) / right (chart + book + orders) / timeline. */
export const TEMPLATES: Record<Tier, Template> = {
  // the right column carries the chart (2026-10-21: our own engine, the
  // column's largest share) — wider than the v2 brief's 380 / 316
  xl: { columns: "380px minmax(0, 1fr) 480px", rows: "88px minmax(0, 1fr) 116px", areas: ["kpi kpi kpi", "left centre right", "timeline timeline timeline"], scrolls: false },
  l:  { columns: "340px minmax(0, 1fr) 400px", rows: "86px minmax(0, 1fr) 108px", areas: ["kpi kpi kpi", "left centre right", "timeline timeline timeline"], scrolls: false },
  m:  { columns: "minmax(0, 1fr) 300px", rows: "auto minmax(420px, 1fr) auto auto", areas: ["kpi kpi", "centre right", "left left", "timeline timeline"], scrolls: true },
  s:  { columns: "minmax(0, 1fr)", rows: "auto", areas: ["kpi", "centre", "left", "right", "timeline"], scrolls: true },
};

export function gridStyle(tier: Tier): { gridTemplateColumns: string; gridTemplateRows: string; gridTemplateAreas: string } {
  return gridStyleOf(TEMPLATES[tier]);
}

export type LayoutFlags = CockpitFlags<PanelId>;

export const EMPTY_FLAGS: LayoutFlags = emptyFlags<PanelId>();

/** The page's two views (2026-11-06, human refocus spec §8 "simplify"): OVERVIEW — the after-tax chart vs SPY,
 *  why money is being lost, a plain-English card per pick, with the essentials (kill state, account, positions,
 *  orders) — and COCKPIT, every panel as before. Overview is the default; the choice is a per-viewer convenience
 *  (BotPage keeps it in localStorage), never state anyone else reads. */
export type BotMode = "overview" | "cockpit";
export const BOT_MODE_STORAGE_KEY = "vt-bot-mode";
export function botModeFrom(raw: string | null | undefined): BotMode {
  return raw === "cockpit" ? "cockpit" : "overview";
}
/** The panels each view shows (kpi is the strip, in both). */
export const MODE_PANELS: Record<BotMode, readonly PanelId[]> = {
  overview: ["perf", "why", "picks", "issues", "book", "orders"],
  cockpit: ["stages", "issues", "state", "watch", "model", "chart", "book", "orders", "timeline", "score"],
};
/** Overview grid at the desktop tiers: the chart spans the centre and right columns on top, the attribution and
 *  the pick cards share the row beneath, the essentials (issues, book, orders) hold the left column. The rows
 *  keep a floor (the chart never squeezes under 380 px); a viewport too short for both floors scrolls the grid
 *  instead of shrinking the chart. The phone and tablet stack the same panels, the chart first. */
export const OVERVIEW_TEMPLATES: Record<"xl" | "l", Omit<Template, "scrolls">> = {
  xl: { columns: "380px minmax(0, 1fr) 480px", rows: "88px minmax(460px, 1.3fr) minmax(320px, 1fr)", areas: ["kpi kpi kpi", "left perf perf", "left why picks"] },
  l: { columns: "340px minmax(0, 1fr) 400px", rows: "86px minmax(420px, 1.3fr) minmax(300px, 1fr)", areas: ["kpi kpi kpi", "left perf perf", "left why picks"] },
};
export function overviewGridStyle(tier: "xl" | "l"): { gridTemplateColumns: string; gridTemplateRows: string; gridTemplateAreas: string } {
  return gridStyleOf(OVERVIEW_TEMPLATES[tier]);
}
