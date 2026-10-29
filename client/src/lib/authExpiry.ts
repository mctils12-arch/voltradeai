// Expired-session detection (field bug 2026-10-28: the owner, still showing
// the signed-in header, opened #/forecast and read "Forecast service
// unavailable. 401: {"error":"Not authenticated"} — the page retries every 5
// minutes"). server/auth.ts sessions live 24h; after that every app API call
// answers 401 while the client keeps its cached /api/auth/me = signed-in.
//
// One central place: queryClient.ts hands every non-OK answer to
// checkSessionExpiry(); a session-shaped 401 from an app endpoint notifies the
// listeners (App.tsx clears the cached auth state, Home opens the sign-in view
// with SESSION_EXPIRED_NOTICE and returns to the same tab afterwards).
// The /api/auth/* endpoints are excluded: a failed login is a 401 too, and the
// sign-in flow reporting itself would loop. Pure module — no React, no
// queryClient import — so node --test covers it.

export const SESSION_EXPIRED_NOTICE = "Your session expired — sign in again";

/** Bodies the server's session guards send with a 401 (server/auth.ts,
 *  server/routes.ts). A 401 for a bad API key or diagnostics token is a
 *  different thing and must not bounce the owner to the sign-in view. */
const SESSION_401 = /not authenticated|session expired|authentication required/i;

/** Path of a request URL, without origin or query string. */
export function apiPath(url: string): string {
  let p = url;
  const m = /^[a-z][a-z0-9+.-]*:\/\/[^/]+/i.exec(p);
  if (m) p = p.slice(m[0].length);
  const cut = p.search(/[?#]/);
  return cut >= 0 ? p.slice(0, cut) : p;
}

/** The sign-in machinery's own endpoints (login, me, logout, signup, reset):
 *  never reported, so the sign-in flow cannot trigger itself. */
export function isAuthEndpoint(url: string): boolean {
  const p = apiPath(url);
  return p === "/api/auth" || p.startsWith("/api/auth/");
}

/** True for a 401 from an app API call whose body says the session is gone. */
export function isSessionExpiry(url: string, status: number, bodyText: string): boolean {
  if (status !== 401) return false;
  const p = apiPath(url);
  if (!p.startsWith("/api/") || isAuthEndpoint(p)) return false;
  return SESSION_401.test(bodyText || "");
}

/** Classifies an error thrown by apiRequest/getQueryFn ("<status>: <body>"). */
export function authErrorKind(err: unknown): "unauthorized" | "forbidden" | null {
  const msg = err instanceof Error ? err.message : String(err ?? "");
  if (/^401:/.test(msg)) return "unauthorized";
  if (/^403:/.test(msg)) return "forbidden";
  return null;
}

type Listener = (url: string) => void;
const listeners = new Set<Listener>();

/** Subscribe to session-expiry reports; returns the unsubscribe. */
export function onSessionExpired(fn: Listener): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

/** Tell every listener the session is gone (also used by a page's own
 *  "Sign in" button on a 401 state). A throwing listener never stops the rest. */
export function notifySessionExpired(url: string): void {
  for (const fn of Array.from(listeners)) {
    try { fn(url); } catch (e) { console.warn("[authExpiry] listener failed", e); }
  }
}

/** Detect + dispatch. Returns whether this answer was reported as an expiry. */
export function checkSessionExpiry(url: string, status: number, bodyText: string): boolean {
  if (!isSessionExpiry(url, status, bodyText)) return false;
  notifySessionExpired(apiPath(url));
  return true;
}
