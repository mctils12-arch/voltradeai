// liveRefresh.ts — how often a query that carries a PRICE re-reads while its
// page is open (2026-11-11 [REPAIR]; human: "on some of the new pages we have
// issues with ticker being just what they were at the start of the day not
// what they actually are").
//
// The app's query defaults never refetch (queryClient.ts: refetchInterval
// false, refetchOnWindowFocus false), and the Trade page keeps its panels
// mounted all day, so a price fetched when the page opened stayed on screen
// until a reload. These policies re-read while the page is visible — React
// Query pauses an interval in a background tab (refetchIntervalInBackground
// stays false) — and on coming back to the tab. Heavier server work gets the
// longer interval; nothing here refreshes faster than its server can answer.
export const LIVE_QUOTES = { refetchInterval: 60_000, refetchOnWindowFocus: true } as const;
export const LIVE_SCANNER = { refetchInterval: 2 * 60_000, refetchOnWindowFocus: true } as const;
export const LIVE_ANALYSIS = { refetchInterval: 5 * 60_000, refetchOnWindowFocus: true } as const;
/** heavy multi-source reports: re-read on returning to the tab, no interval */
export const ON_FOCUS = { refetchOnWindowFocus: true } as const;

/** "live · 3:58 PM ET" for a price from the session in progress, "close ·
 *  Nov 10" for a completed session's close; "" when the server said nothing
 *  about its price (an older server). */
export function priceAsOfLabel(p: { price_asof?: string | null; price_is_partial_session?: boolean | null } | null | undefined): string {
  if (!p || !p.price_asof) return "";
  const t = Date.parse(p.price_asof.length === 10 ? `${p.price_asof}T12:00:00Z` : p.price_asof);
  if (!Number.isFinite(t)) return "";
  if (p.price_is_partial_session) {
    const hm = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }).format(t);
    return `live · ${hm} ET`;
  }
  const md = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "short", day: "numeric" }).format(t);
  return `close · ${md}`;
}
