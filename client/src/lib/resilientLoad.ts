// resilientLoad — reliability helper for /data map layers that fetch ONCE with
// no polling interval (sites, powerplants, boundaries, orbital_sats). Without it,
// a single stalled request (server accepts the connection but never responds —
// the exact failure the trains route was hardened against) leaves the layer
// spinning on "loading" FOREVER, and a single error leaves it dead until the
// user manually toggles it off and on. On flaky mobile one blip at enable-time
// kills the layer for the whole session.
//
// This wraps a layer's load with:
//   - a hard per-attempt TIMEOUT (AbortController) so a hung request rejects
//     instead of hanging forever — no infinite spinner;
//   - AUTO-RETRY with backoff on any failure/timeout;
//   - an optional slow RE-POLL after success, so even static sets self-heal and
//     stay fresh over a long session.
//
// It is DOM/React-free and takes injectable timer/AbortController deps, so the
// scheduling logic is unit-testable under `tsx --test` with no real timers.

export interface ResilientOpts {
  /** hard per-attempt timeout in ms (default 15000). */
  timeoutMs?: number;
  /** if set, re-run this long after each SUCCESS (slow refresh / self-heal). */
  repollMs?: number;
  /** retry delays (ms) after consecutive failures; the last value repeats. */
  backoff?: number[];
}

export interface ResilientDeps {
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (handle: unknown) => void;
  AbortController?: typeof AbortController;
}

/** Default retry backoff: quick first retries, then settle to ~45s. */
export const DEFAULT_BACKOFF = [3000, 8000, 20000, 45000];

/** Pure: the delay (ms) before the retry that follows `failures` failures. */
export function backoffDelay(failures: number, backoff: number[] = DEFAULT_BACKOFF): number {
  const i = failures <= 0 ? 0 : Math.min(failures, backoff.length - 1);
  return backoff[i];
}

/**
 * Run `attempt(signal)` now; retry on rejection/timeout with backoff; and — if
 * `opts.repollMs` is set — re-run after each success. Returns a cleanup fn that
 * cancels the loop, clears any pending timer, and ABORTS the in-flight request
 * (so a load that resolves after teardown never mutates the map).
 *
 * `attempt` must do its own fetch with the passed AbortSignal, render, and
 * success side-effects, and must THROW on any failure (incl. !res.ok) so the
 * loop retries. It should check `signal.aborted` before mutating shared state.
 * `onError(failures)` reports the retrying state (failures = count BEFORE this
 * one, so 0 on the first failure).
 */
export function runResilientLoad(
  attempt: (signal: AbortSignal) => Promise<void>,
  onError: (failures: number) => void,
  opts: ResilientOpts = {},
  deps: ResilientDeps = {},
): () => void {
  const setT = deps.setTimeout ?? ((fn: () => void, ms: number) => setTimeout(fn, ms));
  const clearT = deps.clearTimeout ?? ((h: unknown) => clearTimeout(h as ReturnType<typeof setTimeout>));
  const AC = deps.AbortController ?? AbortController;
  const timeoutMs = opts.timeoutMs ?? 15000;
  const backoff = opts.backoff ?? DEFAULT_BACKOFF;

  let cancelled = false;
  let timer: unknown = null;
  let live: AbortController | null = null;
  let failures = 0;

  const run = async (): Promise<void> => {
    if (cancelled) return;
    const ctrl = new AC();
    live = ctrl;
    const to = setT(() => ctrl.abort(), timeoutMs);
    try {
      await attempt(ctrl.signal);
      clearT(to);
      if (cancelled) return;
      failures = 0;
      if (opts.repollMs != null) timer = setT(run, opts.repollMs);
    } catch {
      clearT(to);
      if (cancelled) return;
      onError(failures);
      const delay = backoffDelay(failures, backoff);
      failures += 1;
      timer = setT(run, delay);
    }
  };
  void run();

  return () => {
    cancelled = true;
    if (timer) clearT(timer);
    if (live) {
      try { live.abort(); } catch { /* already aborted */ }
    }
  };
}
