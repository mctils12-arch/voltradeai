// Alpaca market-data feed selection + upstream error detection.
// 2026-07-20 [REPAIR]: the data proxy hardcoded the sip feed param, which this
// account's subscription rejects ("subscription does not permit querying
// recent SIP data"). The error body ({message}) parsed as an empty snapshot
// map, so the scanner, sector heatmap, and watchlist prices all rendered
// empty with HTTP 200 — dead for every visitor, invisible in monitoring.
// delayed_sip keeps full consolidated-tape volumes (iex would report
// exchange-only volume and break volume ranking/floors) at a 15-minute
// delay. Override with ALPACA_DATA_FEED for accounts with a SIP plan.

export function resolveAlpacaFeed(env: string | undefined): string {
  return env && env.trim() ? env.trim() : "delayed_sip";
}

// The feed for HISTORICAL daily bars (/v2/stocks/bars and /v2/stocks/{sym}/bars)
// whose window ends on or before the day before the current ET date.
// resolveAlpacaFeed's default is WRONG for that endpoint: "delayed_sip" is a
// latest-data concept and is not in the bars endpoint's feed enum, so Alpaca
// answers HTTP 400 "invalid feed: delayed_sip" on any plan. Learned live three
// times before this function existed — alpaca_feed.py bars_feed() (2026-07-18),
// bot.ts /api/bot/bars (2026-07-20/27), and the github_gate2_weekly job's first
// registered run (2026-11-16, "alpaca HTTP 400" in 83 ms) — each time as a
// comment beside a literal, which the next author did not read.
// Default "sip": the consolidated tape, which the free plan serves once the
// window's end is the ET day before today (pricePanel.ts panelAsOf; its
// 27M-bar build used exactly this). A window that reaches today must use
// BARS_FALLBACK_FEED instead ("recent SIP data" is refused with 403).
// An explicit ALPACA_DATA_FEED wins unless it is delayed_sip.
export const BARS_FALLBACK_FEED = "iex";
export function resolveHistoricalBarsFeed(env: string | undefined): string {
  const v = (env ?? "").trim();
  return v && v !== "delayed_sip" ? v : "sip";
}

// Alpaca signals failure as {message: "..."} (usually with non-2xx, but the
// snapshots endpoint has returned 200 error bodies). Returns the error
// message when the response is an error, null when it's real data.
export function alpacaErrorBody(status: number, body: unknown): string | null {
  const isErrorShape =
    body !== null && typeof body === "object" && !Array.isArray(body) &&
    "message" in (body as Record<string, unknown>) &&
    Object.keys(body as Record<string, unknown>).length === 1;
  if (status >= 200 && status < 300 && !isErrorShape) return null;
  if (isErrorShape) return String((body as Record<string, unknown>).message);
  return `alpaca upstream status ${status}`;
}

// ── REAL-TIME DISPLAY PRICES (2026-11-12) ───────────────────────────────────
// Under "delayed_sip" every snapshot is 15 minutes old (post-open check: the
// watchlist's SPY trade at 10:09 ET was stamped 09:54). The plan does carry
// the IEX tape in real time. For DISPLAY routes only, the delayed consolidated
// snapshot stays the base (its daily bar's volume is the whole tape's) and its
// price moves to the IEX trade when that trade is a NEWER REGULAR-HOURS print.
// The scanner keeps delayed_sip alone: its volume ranking needs the full tape.
export const REALTIME_FEED = "iex";

/** Epoch ms of an Alpaca trade time (RFC 3339, nanoseconds allowed); NaN when unparseable. */
export function tradeMs(t: unknown): number {
  if (typeof t !== "string" || !t) return NaN;
  return Date.parse(t.replace(/(\.\d{3})\d+/, "$1"));
}

/** 09:30-16:00 ET on a weekday. */
export function inRegularHoursMs(ms: number): boolean {
  if (!Number.isFinite(ms)) return false;
  const parts = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hourCycle: "h23", weekday: "short",
    hour: "2-digit", minute: "2-digit" }).formatToParts(ms);
  const get = (k: string) => parts.find((p) => p.type === k)?.value || "";
  if (get("weekday") === "Sat" || get("weekday") === "Sun") return false;
  const minutes = (+get("hour") % 24) * 60 + +get("minute");
  return minutes >= 570 && minutes < 960;
}

type Snap = Record<string, any>;

/** Delayed consolidated snapshots with each symbol's price taken from the IEX
 *  tape when its latest trade is a newer regular-hours print. Pure. Same
 *  session: latestTrade from IEX, dailyBar.c = its price (range widened, the
 *  whole tape's volume kept). IEX already on a newer session (the open's first
 *  ~15 minutes): the delayed tape's last bar becomes prevDailyBar and today's
 *  bar is IEX's. Every symbol is tagged priceFeed "iex" or "delayed_sip". */
export function mergeRealtimeTrades(delayed: Record<string, Snap>, realtime: Record<string, Snap> | null | undefined): Record<string, Snap> {
  const out: Record<string, Snap> = {};
  for (const [sym, base] of Object.entries(delayed || {})) {
    const tagged: Snap = { ...base, priceFeed: "delayed_sip" };
    out[sym] = tagged;
    const rt = realtime?.[sym];
    if (!rt) continue;
    const p = Number(rt.latestTrade?.p);
    const rtMs = tradeMs(rt.latestTrade?.t);
    if (!(p > 0) || !inRegularHoursMs(rtMs)) continue;
    const baseMs = tradeMs(base?.latestTrade?.t);
    if (Number.isFinite(baseMs) && rtMs <= baseMs) continue;
    const baseDay = String(base?.dailyBar?.t || "").slice(0, 10);
    const rtDay = String(rt.dailyBar?.t || "").slice(0, 10);
    if (rtDay && baseDay && rtDay === baseDay) {
      const bar = base.dailyBar || {};
      out[sym] = { ...tagged, latestTrade: rt.latestTrade, priceFeed: "iex",
        dailyBar: { ...bar, c: p, h: Math.max(Number(bar.h) || p, p), l: Math.min(Number(bar.l) || p, p) } };
    } else if (rtDay && rtDay > baseDay) {
      out[sym] = { ...tagged, latestTrade: rt.latestTrade, priceFeed: "iex",
        prevDailyBar: base.dailyBar ?? base.prevDailyBar, dailyBar: { ...rt.dailyBar, c: p } };
    }
  }
  return out;
}
