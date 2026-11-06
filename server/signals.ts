/**
 * signals.ts — the signal attribution table (2026-11-06, [RULE-REVIEW]
 * measurement; human refocus spec section 7: "which signals add return and
 * which are noise — cut what does not work").
 *
 * GET /api/bot/signals (owner) and the diag probe "signals" serve the SAME
 * body: signal_attribution.get_signal_attribution() — one row per stock-score
 * signal with its long-term walk-forward test (2017+, point-in-time), its
 * forward test on the shadow log, a pre-registered verdict (anti_predictive /
 * predictive / conflicting / dead / noise / untested), the action the served
 * score takes with it and a plain-English reason; plus the ML evidence (v3
 * honest OOS, the v2 fit's training-set duplication, the B3 serve_fix
 * comparison) and the legacy-vs-v2 composite comparison.
 *
 * Read-only. Daemon first ("signal_attribution" in voltrade_daemon.py), the
 * python3 -c command below as the fallback. The Python loads the ~45 MB
 * shadow log and bootstraps, so the body is cached SIGNALS_CACHE_MS; one
 * refresh in flight at a time; a failed refresh keeps the last good body
 * (stale beats empty) and says so. Pure module apart from the injected call.
 */

export const SIGNALS_CACHE_MS = 10 * 60_000;
export const SIGNALS_CMD = `python3 -c "import json; from signal_attribution import get_signal_attribution; print(json.dumps(get_signal_attribution(), default=str))"`;
/** pythonCall subprocess options and daemon timeout (the Brain route's lane). */
export const SIGNALS_EXEC_OPTS = { timeout: 85_000, maxBuffer: 8 * 1024 * 1024 } as const;
export const SIGNALS_DAEMON_TIMEOUT_MS = 60_000;

export interface PythonCallResult {
  success: boolean;
  result: unknown;
  via: string;
  error?: unknown;
}

export type SignalsCall = () => Promise<PythonCallResult>;

interface CacheEntry {
  at: number;                 // last refresh attempt (drives the TTL)
  bodyAt: number | null;      // when the body served was computed
  body: Record<string, unknown> | null;
  via: string | null;
  error: string | null;
}

/** A cached, de-duplicated reader of the attribution body. */
export function makeSignalsReader(call: SignalsCall, now: () => number = Date.now, ttlMs: number = SIGNALS_CACHE_MS) {
  let cache: CacheEntry | null = null;
  let inflight: Promise<void> | null = null;

  async function refresh(): Promise<void> {
    let r: PythonCallResult;
    try {
      r = await call();
    } catch (e: unknown) {
      r = { success: false, result: null, via: "none", error: e };
    }
    const res = r.result && typeof r.result === "object" ? (r.result as Record<string, unknown>) : null;
    if (r.success && res && !res.error) {
      const t = now();
      cache = { at: t, bodyAt: t, body: res, via: r.via, error: null };
      return;
    }
    const err = String(res?.error ?? (r.error instanceof Error ? r.error.message : r.error) ?? "signal_attribution failed").slice(0, 200);
    console.warn(`[signals] attribution call failed via ${r.via}: ${err}`);
    cache = { at: now(), bodyAt: cache?.bodyAt ?? null, body: cache?.body ?? null, via: cache?.via ?? r.via, error: err };
  }

  return async function read(): Promise<Record<string, unknown>> {
    if (!cache || now() - cache.at > ttlMs) {
      if (!inflight) inflight = refresh().finally(() => { inflight = null; });
      await inflight;
    }
    const c = cache;
    const ageS = c?.bodyAt != null ? Math.round((now() - c.bodyAt) / 1000) : null;
    if (!c?.body) {
      return { error: c?.error ?? "signal attribution unavailable", signals: [], cache_age_s: ageS };
    }
    return { ...c.body, via: c.via, cache_age_s: ageS, ...(c.error ? { stale: true, refresh_error: c.error } : {}) };
  };
}
