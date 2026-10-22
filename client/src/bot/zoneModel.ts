// Bot page v2 — zone templates per viewport tier and the panel vocabulary.
// Pure (tested). The grid is the resting layout; the self-arranging part is
// what the centre column holds (pageState) and the per-panel flags
// (held / pinned / floating / zoomed) the user sets.
export type Tier = "xl" | "l" | "m" | "s";
export type PanelId = "kpi" | "stages" | "issues" | "state" | "watch" | "model" | "chart" | "book" | "orders" | "timeline";
export const PANEL_IDS: readonly PanelId[] = ["kpi", "stages", "issues", "state", "watch", "model", "chart", "book", "orders", "timeline"];

export function tierFor(width: number): Tier {
  if (!Number.isFinite(width)) return "l";
  if (width >= 1920) return "xl";
  if (width >= 1180) return "l";
  if (width >= 768) return "m";
  return "s";
}

/** KPI row: label + value + three 11 px lines (the vs-SPY tile carries four since the after-tax leg, 2026-10-22) need ~84 px uncut; 88/86 keep the strip from clipping its first and last line.
/** grid-template-* per tier. Areas: kpi / left (stages + issues) / centre
 *  (state element + model) / right (chart + book + orders) / timeline. */
export const TEMPLATES: Record<Tier, { columns: string; rows: string; areas: string[]; scrolls: boolean }> = {
  // the right column carries the chart (2026-10-21: our own engine, the
  // column's largest share) — wider than the v2 brief's 380 / 316
  xl: { columns: "340px minmax(0, 1fr) 480px", rows: "88px minmax(0, 1fr) 116px", areas: ["kpi kpi kpi", "left centre right", "timeline timeline timeline"], scrolls: false },
  l:  { columns: "300px minmax(0, 1fr) 400px", rows: "86px minmax(0, 1fr) 108px", areas: ["kpi kpi kpi", "left centre right", "timeline timeline timeline"], scrolls: false },
  m:  { columns: "minmax(0, 1fr) 300px", rows: "auto minmax(420px, 1fr) auto auto", areas: ["kpi kpi", "centre right", "left left", "timeline timeline"], scrolls: true },
  s:  { columns: "minmax(0, 1fr)", rows: "auto", areas: ["kpi", "centre", "left", "right", "timeline"], scrolls: true },
};

export function gridStyle(tier: Tier): { gridTemplateColumns: string; gridTemplateRows: string; gridTemplateAreas: string } {
  const t = TEMPLATES[tier];
  return { gridTemplateColumns: t.columns, gridTemplateRows: t.rows, gridTemplateAreas: t.areas.map((r) => `"${r}"`).join(" ") };
}

export interface LayoutFlags {
  hold: boolean;                 // director paused; everything stays live
  zoomed: PanelId | null;        // one panel fills the page; others keep subscriptions
  floating: PanelId[];           // windows over the page; ghost slot kept
  held: PanelId[];               // display paused at T; bot continues
  pinned: PanelId[];             // the director may not move/resize/evict
}

export const EMPTY_FLAGS: LayoutFlags = { hold: false, zoomed: null, floating: [], held: [], pinned: [] };

export function toggleIn(list: readonly PanelId[], id: PanelId): PanelId[] {
  return list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
}

/** A held panel is implicitly pinned (the director may not take it away). */
export function isPinned(f: LayoutFlags, id: PanelId): boolean {
  return f.pinned.includes(id) || f.held.includes(id);
}

/** Float geometry: initial window rect for a panel popped out of its slot,
 *  clamped so at least 120 px stays on the page. */
export interface Rect { x: number; y: number; w: number; h: number }
export function clampRect(r: Rect, page: { w: number; h: number }, minVisible = 120): Rect {
  const w = Math.max(280, Math.min(r.w, page.w));
  const h = Math.max(160, Math.min(r.h, page.h));
  const x = Math.min(Math.max(r.x, minVisible - w), page.w - minVisible);
  const y = Math.min(Math.max(r.y, 0), page.h - minVisible);
  return { x, y, w, h };
}

export function initialFloatRect(slot: Rect, page: { w: number; h: number }): Rect {
  const w = Math.min(page.w * 0.78, Math.max(slot.w * 1.25, 640));
  const h = Math.min(page.h * 0.78, Math.max(slot.h * 1.25, 360));
  return clampRect({ x: (page.w - w) / 2, y: Math.max(8, (page.h - h) / 2 - 20), w, h }, page);
}

/** Edge snapping for a dragged window (12 px like the brief's spec). */
export function snapEdges(r: Rect, page: { w: number; h: number }, threshold = 12): Rect {
  let { x, y } = r;
  if (Math.abs(x) < threshold) x = 0;
  if (Math.abs(x + r.w - page.w) < threshold) x = page.w - r.w;
  if (Math.abs(y) < threshold) y = 0;
  if (Math.abs(y + r.h - page.h) < threshold) y = page.h - r.h;
  return { ...r, x, y };
}
