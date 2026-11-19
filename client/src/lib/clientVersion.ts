// clientVersion — which release this tab is running.
//
// The build writes the release version into index.html as
// <meta name="vt-client-version"> (vite.config.ts clientVersionMeta). It
// used to be `import { version } from package.json`, which baked the string
// into the entry chunk and renamed that chunk — 607 KB on the wire — on
// every deploy, including the 86% of deploys that change no client file.
// index.html is never cached and ships with the bundle it names, so the
// meta tag is the same fact without the cache cost.
//
// Compared against the layer registry's server_version to detect an open
// tab that is older than the server (rows the bundle has no wiring for:
// the 2026-07-04 toggle desync).

type MetaDoc = { querySelector(sel: string): { getAttribute(name: string): string | null } | null };

/** The version stated by the page that booted this tab; "" when it states
 *  none (a page not produced by our build). */
export function readClientVersion(doc?: MetaDoc | null): string {
  const d = doc ?? (typeof document !== "undefined" ? (document as unknown as MetaDoc) : null);
  if (!d) return "";
  try {
    const v = d.querySelector('meta[name="vt-client-version"]')?.getAttribute("content");
    return typeof v === "string" ? v.trim() : "";
  } catch {
    return "";
  }
}

/** Read once at module load: the tab keeps the version it booted with even
 *  if the document is later altered. */
export const CLIENT_VERSION: string = readClientVersion();

/** True only when BOTH versions are known and differ. An unknown client
 *  version is not evidence of an update — no banner is shown for it. */
export function isVersionSkew(serverVersion: string | null | undefined, clientVersion: string = CLIENT_VERSION): boolean {
  return !!serverVersion && !!clientVersion && serverVersion !== clientVersion;
}
