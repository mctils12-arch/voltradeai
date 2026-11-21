// FAA SWIM FEED STATUS — the Streams inventory's read of GET
// /api/data/swim/products (server/swimCapture.ts): the five FAA live feeds
// the server is set up for, whether each is connected, how many messages it
// has received since the server started and how old the newest one is
// ([PRODUCT] 2026-11-20, UI coverage batch B).
//
// These are LIVE FEEDS, not archived streams: no message is stored, and the
// counters start again at every server restart. Since 2026-11-21 the rows
// still in force (traffic programs, airport runway reports, flight
// restrictions) are saved each minute and read back at a start
// (server/swimStoreKeep.ts). The page says all three. Pure; pinned by swimFeeds.test.ts. Read tolerantly: another shape is
// null and the block says the status could not be read. The server reports a
// missing setting by NAME; the page prints only how many are missing.

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);

export const FEED_NAMES: Readonly<Record<string, string>> = {
  SFDPS: "Flight plans", TFMS: "Traffic management", NOTAM: "Notices to airmen", STDDS: "Terminal data", ITWS: "Terminal weather",
};
/** what the server does with a feed, in the page's words */
export const FEED_USE: Readonly<Record<string, string>> = {
  SFDPS: "filed flight plans and their amendments, on the flight card",
  TFMS: "ground stops, delay programs, restrictions and runway rates, on the FAA airport card; flow areas and national advisories are read and not yet shown",
  NOTAM: "flight restrictions, on the map's Flight restrictions layer; other notices are counted and not shown",
};
/** a feed with no message for this long while connected is "quiet" */
export const FEED_QUIET_MS = 5 * 60_000;

export type FeedState = "receiving" | "quiet" | "disconnected" | "not_consumed" | "not_set_up";
export interface FeedView {
  id: string; name: string; state: FeedState; use: string | null;
  received: number; processed: number; dropped: number; errors: number; reconnects: number;
  lastMessageAt: number | null; lastError: string | null; missingSettings: number;
}
export interface FeedsView { at: number | null; feeds: FeedView[] }

/** GET /api/data/swim/products → one row per feed, in the server's order; null on another shape. */
export function parseSwimFeeds(raw: unknown, nowMs: number): FeedsView | null {
  const o = obj(raw);
  const products = o ? obj(o.products) : null;
  if (!o || !products) return null;
  const feeds: FeedView[] = [];
  for (const [id, pv] of Object.entries(products)) {
    const p = obj(pv);
    if (!p) continue;
    const env = obj(p.env);
    const configured = env?.configured === true;
    const connected = p.connected === true;
    const consumer = str(p.consumer) ?? "";
    const last = fin(p.lastMessageAt) ? p.lastMessageAt : null;
    const state: FeedState = !configured ? "not_set_up"
      : /^configured-only/i.test(consumer) ? "not_consumed"
      : !connected ? "disconnected"
      : last !== null && nowMs - last <= FEED_QUIET_MS ? "receiving" : "quiet";
    feeds.push({
      id, name: FEED_NAMES[id] ?? id, state, use: FEED_USE[id] ?? null,
      received: fin(p.messagesReceived) ? p.messagesReceived : 0, processed: fin(p.messagesProcessed) ? p.messagesProcessed : 0,
      dropped: fin(p.droppedOverflow) ? p.droppedOverflow : 0, errors: fin(p.processErrors) ? p.processErrors : 0, reconnects: fin(p.reconnects) ? p.reconnects : 0,
      lastMessageAt: last, lastError: str(p.lastError), missingSettings: Array.isArray(env?.missing) ? env.missing.length : 0,
    });
  }
  return { at: fin(o.at) ? o.at : null, feeds };
}

export const FEED_STATE_WORDS: Readonly<Record<FeedState, string>> = {
  receiving: "receiving", quiet: "quiet", disconnected: "disconnected", not_consumed: "not read", not_set_up: "not set up",
};
/** "2 s ago" · "4 min ago" · "3.2 h ago" · "never" */
export function feedAge(lastMessageAt: number | null, nowMs: number): string {
  if (lastMessageAt === null) return "never";
  const s = Math.max(0, Math.round((nowMs - lastMessageAt) / 1000));
  if (s < 90) return `${s} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  return `${(s / 3600).toFixed(1)} h ago`;
}
const n = (v: number) => v.toLocaleString("en-US");
/** The card's counts line: what arrived, what was handled, and anything lost — never a lost message left unsaid. */
export function feedCounts(f: FeedView): string {
  if (f.state === "not_set_up") return `${f.missingSettings} setting${f.missingSettings === 1 ? "" : "s"} missing on the server`;
  if (f.state === "not_consumed") return "set up on the server, deliberately not read yet";
  const parts = [`${n(f.received)} message${f.received === 1 ? "" : "s"} since the server started`];
  if (f.processed !== f.received) parts.push(`${n(f.processed)} handled`);
  if (f.dropped) parts.push(`${n(f.dropped)} dropped under load`);
  if (f.errors) parts.push(`${n(f.errors)} failed to parse`);
  if (f.reconnects) parts.push(`${n(f.reconnects)} reconnect${f.reconnects === 1 ? "" : "s"}`);
  return parts.join(" · ");
}
/** The block's one line. */
export function feedsSentence(v: FeedsView): string {
  const live = v.feeds.filter((f) => f.state === "receiving").length;
  const read = v.feeds.filter((f) => f.state === "receiving" || f.state === "quiet" || f.state === "disconnected").length;
  const down = v.feeds.filter((f) => f.state === "disconnected" || f.state === "quiet");
  return `${live} of ${read} FAA feeds the server reads are receiving${down.length ? ` · ${down.map((f) => `${f.name.toLowerCase()} ${FEED_STATE_WORDS[f.state]}`).join(", ")}` : ""} · ${v.feeds.length - read} more not read`;
}
