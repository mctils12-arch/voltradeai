// Bot page v2 — the bot's binding of the cockpit zone vocabulary. 2026-10-29:
// the generic engine (tiers, flags, float geometry, grid styles) moved to
// client/src/cockpit/zoneModel.ts so the Trade page runs the same one; what is
// the bot's stays here — its panel ids, its grid templates per tier and the
// typed flag set — and every export this module had is kept (pinned by
// zoneModel.test.ts), so no importer changes.
import { gridStyle as gridStyleOf, emptyFlags, type LayoutFlags as CockpitFlags, type Template, type Tier } from "../cockpit/zoneModel";
export { tierFor, toggleIn, isPinned, clampRect, initialFloatRect, snapEdges, type Tier, type Rect, type Template } from "../cockpit/zoneModel";

export type PanelId = "kpi" | "stages" | "issues" | "state" | "watch" | "model" | "chart" | "book" | "orders" | "timeline" | "score";
export const PANEL_IDS: readonly PanelId[] = ["kpi", "stages", "issues", "state", "watch", "model", "chart", "book", "orders", "timeline", "score"];

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
