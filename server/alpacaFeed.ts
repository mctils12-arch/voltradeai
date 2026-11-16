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
