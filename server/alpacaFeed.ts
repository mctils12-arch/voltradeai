// Alpaca market-data feed selection + upstream error detection.
// 2026-07-20 [REPAIR]: the data proxy hardcoded feed=sip, which this
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
