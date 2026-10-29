// Cockpit layout engine — the generic zone vocabulary. 2026-10-29: lifted out
// of client/src/bot/zoneModel.ts so a second cockpit (the Trade page) runs the
// SAME engine instead of a copy; the bot module binds these to its PanelId and
// keeps every export it had. Pure (tested). The grid is the resting layout;
// the per-panel flags (held / pinned / floating / zoomed) are what the user
// sets over it. `Id` is a cockpit's panel-id union — nothing here knows the
// bot's.
export type Tier = "xl" | "l" | "m" | "s";

export function tierFor(width: number): Tier {
  if (!Number.isFinite(width)) return "l";
  if (width >= 1920) return "xl";
  if (width >= 1180) return "l";
  if (width >= 768) return "m";
  return "s";
}

/** One tier's resting grid: the track lists, the named areas row by row, and
 *  whether the page scrolls (the phone/tablet stack) or fits (desktop). */
export interface Template { columns: string; rows: string; areas: string[]; scrolls: boolean }

export function gridStyle(t: Template): { gridTemplateColumns: string; gridTemplateRows: string; gridTemplateAreas: string } {
  return { gridTemplateColumns: t.columns, gridTemplateRows: t.rows, gridTemplateAreas: t.areas.map((r) => `"${r}"`).join(" ") };
}

export interface LayoutFlags<Id extends string> {
  hold: boolean;                 // director paused; everything stays live
  zoomed: Id | null;             // one panel fills the page; others keep subscriptions
  floating: Id[];                // windows over the page; ghost slot kept
  held: Id[];                    // display paused at T; the feed continues
  pinned: Id[];                  // the director may not move/resize/evict
}

/** A fresh empty flag set. A function, not a shared constant: each cockpit's
 *  typed EMPTY_FLAGS is one binding of it (the bot's in bot/zoneModel.ts). */
export function emptyFlags<Id extends string>(): LayoutFlags<Id> {
  return { hold: false, zoomed: null, floating: [], held: [], pinned: [] };
}

export function toggleIn<Id extends string>(list: readonly Id[], id: Id): Id[] {
  return list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
}

/** A held panel is implicitly pinned (the director may not take it away). */
export function isPinned<Id extends string>(f: LayoutFlags<Id>, id: Id): boolean {
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
