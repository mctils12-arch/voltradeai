/**
 * healthGate.ts — the ONE place that decides which HTTP code /api/health
 * returns. Pure (node:test safe — no sqlite, no fs, no network).
 *
 * WHY THIS MODULE EXISTS (KNOWN BROKEN #41, the 2026-09-10 -> 2026-09-16
 * five-day outage — full account in research/open_questions.md):
 *
 * /api/health answers TWO different questions with one payload:
 *   - to a human / the DAILY routine: "is anything wrong?" -> `status`
 *   - to Railway: "may this container serve traffic?" -> the HTTP code
 *     (railway.json, FROZEN: healthcheckPath=/api/health, timeout 60s —
 *     a non-2xx answer FAILS the deploy and the new container never
 *     takes over).
 *
 * Until 2026-09-16 the handler mapped ANY `status !== "ok"` to HTTP 503,
 * so every alarm that is genuinely worth raising (liveness dark, Alpaca
 * unreachable, scanner back-off, a licensing warning) was also a veto on
 * every future deploy. The 2026-08-12 feed dead-air alarm took the site
 * down for ~15 minutes that way (OPS GOTCHAS, open_questions.md) and was
 * exempted by hand; the LIVENESS ALARM was not. On 2026-09-10 the
 * drawdown kill switch latched (persisted on the volume, restored on
 * every boot, by design), the loop went dark, the liveness alarm flipped
 * `status` to "degraded" after 2 market hours, and from that moment every
 * fresh container — 30 merged PRs' worth over 5 days — answered Railway's
 * probe with 503 and was rejected. The site stayed at Railway's 502 edge
 * fallback the whole time while every session re-confirmed "still down".
 * A monitoring signal must never be able to veto the deploy that would
 * fix it.
 *
 * THE CONTRACT: the HTTP code reflects ONLY whether this container can
 * serve HTTP — the process is up and its database opens. Everything else
 * (broker, python engine, trading-loop liveness, scanner, feeds,
 * licensing) reports INSIDE the payload, keeps flipping `status` to
 * "degraded" exactly as before (CLAUDE.md Amendment 1: the loop going
 * dark IS a degraded state on /api/health — that is unchanged), and never
 * touches the HTTP code. `serving` in the payload makes the split visible
 * to anyone reading it.
 *
 * Adding a check to SERVING_CHECKS is a deploy-gate change: it must be a
 * condition under which the container genuinely cannot answer requests
 * AND under which a restart is a plausible remedy. A provider outage
 * satisfies neither.
 */

export const SERVING_CHECKS = ["server", "database"] as const;

export interface HealthCheckBlock {
  status?: string;
  [k: string]: unknown;
}

export interface HealthPayload {
  status: string;
  checks: Record<string, HealthCheckBlock | undefined>;
  [k: string]: unknown;
}

export interface ServingVerdict {
  /** true -> HTTP 200; false -> HTTP 503 (Railway rejects the container). */
  ok: boolean;
  /** the checks that decide the HTTP code — nothing else does */
  gates: string[];
  /** which of those gates is currently not "ok" (empty when serving) */
  failing: string[];
  /** printed so a reader of the payload learns the contract without the source */
  note: string;
}

export const SERVING_NOTE =
  "HTTP code reflects only whether this container can serve traffic (the " +
  "checks in `gates`); `status` carries every alarm and never gates the deploy";

export function servingVerdict(payload: HealthPayload): ServingVerdict {
  const failing = SERVING_CHECKS.filter((name) => payload.checks?.[name]?.status !== "ok");
  return {
    ok: failing.length === 0,
    gates: [...SERVING_CHECKS],
    failing: [...failing],
    note: SERVING_NOTE,
  };
}

export function healthHttpCode(payload: HealthPayload): 200 | 503 {
  return servingVerdict(payload).ok ? 200 : 503;
}
