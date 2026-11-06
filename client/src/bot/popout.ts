// Bot page — a panel in its OWN browser window (slice D of
// research/bot_page_program.md; human directive 2026-10-21: "detach one of
// these blocks … move it to my other monitor"). The pop-out is the same app at
// /app#/bot/pane/<id>: BotPage renders that one panel full-window with its own
// data hooks — its own polling, nothing proxied through the main page, so it
// keeps working if the main page is closed or reloaded. The windows talk over
// one BroadcastChannel: the focus symbol (a Book click on either side charts
// the same ticker everywhere) and a pane's goodbye, so the main page restores
// the slot the moment the window closes (a 1 s `closed` poll is the fallback
// for a window killed without pagehide). 2026-10-29: the mechanics are the
// shared client/src/cockpit/popout.ts; this module binds the bot's prefix,
// channel, window names and ids and keeps its exports (pinned by popout.test.ts).
import { PANEL_IDS, type PanelId } from "./zoneModel";
import { paneIdFromHash as paneIdFromHashOf, paneUrl as paneUrlOf, paneName as paneNameOf, parseChannelMsg, paneLifecycleMsg } from "../cockpit/popout";
export { PANE_MIN, paneFeatures, windowId, type SlotRect } from "../cockpit/popout";

export const PANE_PREFIX = "#/bot/pane/";
export const PANE_CHANNEL = "vt-bot-panes";
/** one named window per panel (`vt-pane-<id>`): opening it twice focuses the existing pane */
export const PANE_WINDOW_PREFIX = "vt-pane";

export type PaneMsg =
  | { t: "focus"; sym: string | null; from: string }
  | { t: "pane-closed"; id: PanelId; from: string }
  | { t: "pane-open"; id: PanelId; from: string };

/** "#/bot/pane/chart" → "chart"; anything else (other hashes, unknown ids) → null */
export function paneIdFromHash(hash: string): PanelId | null {
  return paneIdFromHashOf(hash, PANE_PREFIX, PANEL_IDS);
}

export function paneUrl(id: PanelId, loc: { origin: string; pathname: string }): string {
  return paneUrlOf(id, PANE_PREFIX, loc);
}

export function paneName(id: PanelId): string { return paneNameOf(id, PANE_WINDOW_PREFIX); }

/** a message from the channel, or null for anything that is not ours */
export function parsePaneMsg(data: unknown): PaneMsg | null {
  const m = parseChannelMsg(data);
  if (!m) return null;
  if (m.t === "focus" && (m.sym === null || typeof m.sym === "string")) return { t: "focus", sym: m.sym as string | null, from: m.from };
  return paneLifecycleMsg(m, PANEL_IDS);
}
