import { QueryClient, QueryFunction } from "@tanstack/react-query";

const API_BASE = "__PORT_5000__".startsWith("__") ? "" : "__PORT_5000__";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  const res = await fetch(`${API_BASE}${url}`, {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  await throwIfResNotOk(res);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const res = await fetch(`${API_BASE}${queryKey.join("/")}`, {
      credentials: "include",
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

/** A failure worth retrying: the server was briefly unreachable (a deploy /
 *  restart window — 502/503/504 from the proxy, or a network error before
 *  any response). Real answers (400/401/404/500 with a body, "no options",
 *  bad ticker) are NOT retried: repeating them would only delay the error. */
export function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err ?? "");
  if (/^(502|503|504):/.test(msg)) return true;
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
