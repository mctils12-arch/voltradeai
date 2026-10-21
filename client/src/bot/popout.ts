// Bot page — a panel in its OWN browser window (slice D of
// research/bot_page_program.md; human directive 2026-10-21: "detach one of
// these blocks … move it to my other monitor"). The pop-out is the same app at
// /app#/bot/pane/<id>: BotPage renders that one panel full-window with its own
// data hooks — its own polling, nothing proxied through the main page, so it
// keeps working if the main page is closed or reloaded. The windows talk over
// one BroadcastChannel: the focus symbol (a Book click on either side charts
// the same ticker everywhere) and a pane's goodbye, so the main page restores
// the slot the moment the window closes (a 1 s `closed` poll is the fallback
// for a window killed without pagehide). Pure helpers; pinned by popout.test.ts.
import { PANEL_IDS, type PanelId } from "./zoneModel";

export const PANE_PREFIX = "#/bot/pane/";
export const PANE_CHANNEL = "vt-bot-panes";
/** the smallest useful pane; a slot smaller than this opens at this size */
export const PANE_MIN = { w: 480, h: 360 };

export type PaneMsg =
  | { t: "focus"; sym: string | null; from: string }
  | { t: "pane-closed"; id: PanelId; from: string }
  | { t: "pane-open"; id: PanelId; from: string };

/** "#/bot/pane/chart" → "chart"; anything else (other hashes, unknown ids) → null */
export function paneIdFromHash(hash: string): PanelId | null {
  if (!hash.startsWith(PANE_PREFIX)) return null;
  const id = hash.slice(PANE_PREFIX.length).split(/[/?&#]/)[0];
  return (PANEL_IDS as readonly string[]).includes(id) ? (id as PanelId) : null;
}

export function paneUrl(id: PanelId, loc: { origin: string; pathname: string }): string {
  return `${loc.origin}${loc.pathname}${PANE_PREFIX}${id}`;
}

/** one named window per panel: opening it twice focuses the existing pane */
export function paneName(id: PanelId): string { return `vt-pane-${id}`; }

export interface SlotRect { x: number; y: number; w: number; h: number }

/** window.open features: a popup sized from the slot (a little larger — the
 *  panel gains the window's whole height), never smaller than PANE_MIN nor
 *  larger than the screen, placed over the slot so it appears where the
 *  panel was and can be dragged to another monitor from there. */
export function paneFeatures(slot: SlotRect | null, screen: { w: number; h: number }): string {
  const w = Math.round(Math.min(Math.max(PANE_MIN.w, screen.w > 0 ? Math.min(screen.w, slot ? slot.w * 1.15 : 760) : 760), Math.max(screen.w, PANE_MIN.w)));
  const h = Math.round(Math.min(Math.max(PANE_MIN.h, screen.h > 0 ? Math.min(screen.h, slot ? slot.h * 1.3 : 520) : 520), Math.max(screen.h, PANE_MIN.h)));
  const left = Math.round(Math.max(0, Math.min(Math.max(0, screen.w - w), slot ? slot.x : (screen.w - w) / 2)));
  const top = Math.round(Math.max(0, Math.min(Math.max(0, screen.h - h), slot ? slot.y : (screen.h - h) / 2)));
  return `popup=yes,width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`;
}

/** a message from the channel, or null for anything that is not ours */
export function parsePaneMsg(data: unknown): PaneMsg | null {
  if (!data || typeof data !== "object") return null;
  const m = data as Record<string, unknown>;
  if (typeof m.from !== "string") return null;
  if (m.t === "focus" && (m.sym === null || typeof m.sym === "string")) return { t: "focus", sym: m.sym as string | null, from: m.from };
  if ((m.t === "pane-closed" || m.t === "pane-open") && typeof m.id === "string" && (PANEL_IDS as readonly string[]).includes(m.id)) return { t: m.t, id: m.id as PanelId, from: m.from };
  return null;
}

/** this window's id on the channel (a sender ignores its own echoes) */
export function windowId(): string { return Math.random().toString(36).slice(2, 10); }
