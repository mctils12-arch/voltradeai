// Cockpit pop-out — a panel in its OWN browser window (slice D of
// research/bot_page_program.md; human directive 2026-10-21: "detach one of
// these blocks … move it to my other monitor"). The pop-out is the same app at
// <hashPrefix><id>: the page renders that one panel full-window with its own
// data hooks — its own polling, nothing proxied through the main page, so it
// keeps working if the main page is closed or reloaded. The windows of one
// cockpit talk over one BroadcastChannel: the pane lifecycle here (a pane's
// goodbye, so the page restores the slot the moment the window closes; a 1 s
// `closed` poll in useCockpitLayout is the fallback for a window killed
// without pagehide), the cockpit's own messages (the bot's focus symbol)
// relayed to the page by useCockpitLayout's onMessage.
// 2026-10-29: generic over the panel id — client/src/bot/popout.ts binds the
// bot's prefix, channel, window names and ids and keeps its exports (pinned by
// bot/popout.test.ts); the mechanics here are pinned by cockpit/popout.test.ts.

/** the smallest useful pane; a slot smaller than this opens at this size */
export const PANE_MIN = { w: 480, h: 360 };

export interface SlotRect { x: number; y: number; w: number; h: number }

/** "<hashPrefix><id>" → id when `ids` has it; anything else (other hashes, unknown ids) → null */
export function paneIdFromHash<Id extends string>(hash: string, hashPrefix: string, ids: readonly Id[]): Id | null {
  if (!hash.startsWith(hashPrefix)) return null;
  const id = hash.slice(hashPrefix.length).split(/[/?&#]/)[0];
  return (ids as readonly string[]).includes(id) ? (id as Id) : null;
}

export function paneUrl(id: string, hashPrefix: string, loc: { origin: string; pathname: string }): string {
  return `${loc.origin}${loc.pathname}${hashPrefix}${id}`;
}

/** one named window per panel (`<windowPrefix>-<id>`): opening it twice focuses
 *  the existing pane. Window names are global to the browser, so each cockpit
 *  passes its own prefix. */
export function paneName(id: string, windowPrefix: string): string { return `${windowPrefix}-${id}`; }

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

/** anything on the channel that is ours: a kind and the sender's window id;
 *  the rest of the fields are the message's own */
export interface ChannelMsg { t: string; from: string; [k: string]: unknown }

/** a message from the channel, or null for anything that is not ours */
export function parseChannelMsg(data: unknown): ChannelMsg | null {
  if (!data || typeof data !== "object") return null;
  const m = data as Record<string, unknown>;
  if (typeof m.t !== "string" || typeof m.from !== "string") return null;
  return m as ChannelMsg;
}

/** the pane lifecycle every cockpit shares; `id` must be one of the cockpit's */
export type PaneLifecycleMsg<Id extends string> = { t: "pane-closed" | "pane-open"; id: Id; from: string };

export function paneLifecycleMsg<Id extends string>(m: ChannelMsg, ids: readonly Id[]): PaneLifecycleMsg<Id> | null {
  if ((m.t === "pane-closed" || m.t === "pane-open") && typeof m.id === "string" && (ids as readonly string[]).includes(m.id)) return { t: m.t, id: m.id as Id, from: m.from };
  return null;
}

/** this window's id on the channel (a sender ignores its own echoes) */
export function windowId(): string { return Math.random().toString(36).slice(2, 10); }
