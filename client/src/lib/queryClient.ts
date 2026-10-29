import { QueryClient, QueryFunction } from "@tanstack/react-query";
import { checkSessionExpiry } from "./authExpiry";

const API_BASE = "__PORT_5000__".startsWith("__") ? "" : "__PORT_5000__";

/** Non-OK answer → "<status>: <body>" error. A session-shaped 401 from an
 *  app endpoint is also reported to authExpiry (field bug 2026-10-28: an
 *  expired 24h session read as a service outage) — the one central place. */
async function throwIfResNotOk(res: Response, url: string) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    checkSessionExpiry(url, res.status, text);
    throw new Error(`${res.status}: ${text}`);
  }
}

/** Field bug 2026-10-20 (Bot page empty until a manual reload): a request
 *  that never answers — a connection held open across a Railway deploy —
 *  blocks react-query polling for good, because an interval refetch JOINS
 *  the in-flight request instead of starting a new one. Polled reads pass
 *  `timeoutMs`; a request with no answer by then is aborted and surfaces as
 *  a "timeout:" error, which retryTransient treats like any dropped
 *  connection, so the next poll starts fresh. Opt-in, never a global
 *  default: some GETs are legitimately slow (a full /api/scan, analysis),
 *  and mutations must never be cut off mid-flight (a retried POST could
 *  act twice). */
export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
  opts: { timeoutMs?: number } = {},
): Promise<Response> {
  const ms = opts.timeoutMs && opts.timeoutMs > 0 ? opts.timeoutMs : 0;
  const ac = ms ? new AbortController() : null;
  let timedOut = false;
  const timer = ac ? setTimeout(() => { timedOut = true; ac.abort(); }, ms) : null;
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${url}`, {
      method,
      headers: data ? { "Content-Type": "application/json" } : {},
      body: data ? JSON.stringify(data) : undefined,
      credentials: "include",
      ...(ac ? { signal: ac.signal } : {}),
    });
  } catch (e) {
    if (timedOut) throw new Error(`timeout: no answer from ${url} in ${ms} ms`);
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }

  await throwIfResNotOk(res, url);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const url = queryKey.join("/");
    const res = await fetch(`${API_BASE}${url}`, {
      credentials: "include",
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      checkSessionExpiry(url, res.status, await res.text().catch(() => ""));
      return null;
    }

    await throwIfResNotOk(res, url);
    return await res.json();
  };

/** A failure worth retrying: the server was briefly unreachable (a deploy /
 *  restart window — 502/503/504 from the proxy, or a network error before
 *  any response). Real answers (400/401/404/500 with a body, "no options",
 *  bad ticker) are NOT retried: repeating them would only delay the error. */
export function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err ?? "");
  if (/^(502|503|504):/.test(msg)) return true;
  if (/^timeout:/.test(msg)) return true; // apiRequest's own timeout (a hung connection)
  // fetch() rejects with a TypeError before any response on a dropped connection
  return err instanceof TypeError || /failed to fetch|networkerror|load failed|network request failed/i.test(msg);
}

export const TRANSIENT_MAX_RETRIES = 4;

/** react-query `retry`: up to 4 retries, transient failures only. */
export function retryTransient(failureCount: number, err: unknown): boolean {
  return failureCount < TRANSIENT_MAX_RETRIES && isTransientError(err);
}

/** 3 s, 6 s, 12 s, 24 s (~45 s total) — spans a typical Railway restart. */
export function transientRetryDelay(attempt: number): number {
  return Math.min(3000 * 2 ** attempt, 24_000);
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "returnNull" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: 60_000,
      retry: retryTransient,
      retryDelay: transientRetryDelay,
    },
    mutations: {
      retry: false,
    },
  },
});
