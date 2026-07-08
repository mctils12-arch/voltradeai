// tiles3dProxy — server-side proxy helpers for Google Photorealistic 3D Tiles
// (worldview-globe G3). Keeps GOOGLE_MAPS_API_KEY server-side: the browser never
// sees it. The deck.gl client routes EVERY 3D-tiles fetch (root + child glTF/
// json) through our proxy, which appends the key. The ROOT request is the
// billable unit and is gated by tiles3dBudget.authorizeRoot(); child fetches are
// free within the session and only allowlist-validated.
//
// SECURITY: is3dTilesUrl() is a TIGHT allowlist — https + exact host
// tile.googleapis.com + path under /v1/3dtiles/ only — so the /proxy endpoint
// can NEVER be turned into an open proxy / SSRF vector. Pure + tested.

export const ROOT_URL = "https://tile.googleapis.com/v1/3dtiles/root.json";

/** True only for a real Google 3D Tiles URL. Rejects any other host/scheme/path
 * (open-proxy / SSRF guard). Pure. */
export function is3dTilesUrl(u: string): boolean {
  let url: URL;
  try { url = new URL(u); } catch { return false; }
  return (
    url.protocol === "https:" &&
    url.hostname === "tile.googleapis.com" &&
    url.pathname.startsWith("/v1/3dtiles/")
  );
}

/** Append the API key to a URL's query (replacing any existing key). Pure. */
export function withKey(u: string, key: string): string {
  const url = new URL(u);
  url.searchParams.set("key", key);
  return url.toString();
}

/** The server-relative proxy URL the client should fetch instead of a raw
 * googleapis URL (so the key stays server-side). Pure. */
export function proxyPathFor(googleUrl: string): string {
  return `/api/data/3dtiles/proxy?u=${encodeURIComponent(googleUrl)}`;
}
