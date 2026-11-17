// aircraftFastLane.ts — the SELECTED-AIRCRAFT fast lane (human 2026-09-30:
// "if it's hard to have fast real-time ADS-B on all planes, optimize for the
// plane you have clicked on and are watching — I don't want it to lag if
// there is data"). The worldwide snapshot refreshes most of the world every
// ~60 s; a selected plane zoomed out showed "last position 89s ago".
//
// GET /api/data/aircraft/live/:hex (globalAircraft.ts) asks this lane:
//   1. the snapshot row if it is younger than FAST_FRESH_MS — no upstream;
//   2. else the lane's own 2 s cache for that hex;
//   3. else ONE adsb.lol /v2/hex/<hex> request (monetization-lawful provider,
//      providerCompliance.ts), COALESCED per hex (concurrent viewers share the
//      in-flight promise), accounted to the shared adsb.lol governor on the
//      FOREGROUND (viewer) lane — never delayed by the background sweep; the
//      fix is published to aircraftFixBus so the snapshot, the plan's
//      callsign fill and every other viewer benefit.
// POLITENESS: at most FAST_MAX_HEXES_PER_MIN distinct hexes may hit upstream
// per minute and at most FAST_MAX_RPS requests per second in total (token
// bucket); over either bound the lane answers from the snapshot, saying so.
// HONESTY: the row's seenAt is the fix's OWN time (upstream `now` − seen_pos);
// when upstream has nothing newer the client sees the older time, never a
// fabricated fresh one.
//
// SECOND PROVIDER ON FAILURE (2026-11-16, owner report "tracking comes and
// goes, the model freezes"). Measured live that day: 25 of 57 lookups of one
// airborne aircraft ended in an adsb.lol timeout (6 s each) or refusal, and
// its fix aged from 46 s to 142 s. Until then the second provider (adsb.fi,
// coverageGapFill.ts) was asked only when adsb.lol ANSWERED with nothing.
//   - a failed adsb.lol request (429 / 5xx / timeout) now asks adsb.fi for
//     the same hex; a row it returns is published with "adsbfi" provenance
//     and the answer says source "fallback";
//   - after FAST_BREAKER_FAILURES consecutive adsb.lol failures the lane
//     stops sending it every lookup: adsb.fi answers, and adsb.lol gets one
//     probe per FAST_BREAKER_PROBE_MS until a probe succeeds. A viewer no
//     longer waits out a 6 s timeout on each poll, and a struggling provider
//     gets one request per 15 s from this lane, not one per poll.
// ADSB.LOL'S DAILY QUOTA (2026-11-17, adsbGovernor.ts). adsb.lol counts
// requests per day (500 without a key). Each lookup carries the API key when
// one is configured and reports the response's quota headers to the
// governor. While the quota is spent, or kept for the watched planes
// ("watched_only"), a lookup does not ask adsb.lol: it goes straight to the
// second provider (counted in quota_skips).
// adsb.fi is NON-COMMERCIAL (providerCompliance.ts already lists it; the
// monetization tripwire sees these rows as it sees the gap-fill's). Its own
// 1 request/second bucket (coverageGapFill.ts) bounds the extra requests; an
// empty bucket skips the fill and the answer is the honest "error".

import type { FixBatch } from "./aircraftFixBus";
import { ADSB_LOL_UA, adsbLolGovernor, adsbLolRequestHeaders, parseRetryAfter, type UpstreamGovernor } from "./adsbGovernor";
import { mapPointAircraft } from "./aircraftTiling";
import type { SnapRow } from "./globalSnapshot";

/** a snapshot row younger than this is served without an upstream request */
export const FAST_FRESH_MS = 3_000;
/** per-hex result cache */
export const FAST_CACHE_MS = 2_000;
/** distinct hexes allowed upstream per rolling minute */
export const FAST_MAX_HEXES_PER_MIN = 20;
/** total upstream request rate for the lane (token bucket, burst FAST_BURST) */
export const FAST_MAX_RPS = 1;
export const FAST_BURST = 2;
export const FAST_TIMEOUT_MS = 6_000;
/** consecutive adsb.lol failures on this lane before it is only probed ... */
export const FAST_BREAKER_FAILURES = 2;
/** ... once per this interval, until a probe answers */
export const FAST_BREAKER_PROBE_MS = 15_000;
export const FAST_UA = ADSB_LOL_UA;

export const ICAO_HEX_LC_RE = /^[0-9a-f]{6}$/;
export const fastLaneUrl = (hex: string): string => `https://api.adsb.lol/v2/hex/${hex}`;

/** "fallback" = adsb.lol failed (or is being probed only) and the second provider answered */
export type FastSource = "snapshot" | "cache" | "upstream" | "fallback" | "limited" | "error" | "absent";

export interface FastResult {
  row: SnapRow | null;
  /** broadcast barometric vertical rate (ft/min) for THIS row's fix, when upstream sent one */
  baroRateFpm: number | null;
  source: FastSource;
  /** upstream HTTP status of the request that produced this answer (null = none made) */
  upstreamStatus: number | null;
}

export interface FastLaneDeps {
  snapshotGet: (hex: string) => SnapRow | undefined;
  publish: (b: FixBatch) => void;
  fetchImpl?: typeof fetch;
  governor?: UpstreamGovernor;
  now?: () => number;
  headers?: Record<string, string>;
  /** where ADSBLOL_API_KEY is read when `headers` is not given (default process.env) */
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
  /** COVERAGE GAP-FILL (2026-10-14): rows for a hex adsb.lol returned NOTHING
   *  for, from adsb.fi (coverageGapFill.ts) — e.g. a plane taxiing on a
   *  surface adsb.lol's receivers cannot see. Rows keep "adsbfi" provenance. */
  fallbackHex?: (hex: string) => Promise<any[]>;
}

export interface FastLane {
  lookup(hex: string): Promise<FastResult>;
  stats(): {
    upstream_requests: number; coalesced: number; cache_hits: number; snapshot_hits: number; limited: number; errors: number; fallback_rows: number; hexes_last_min: number;
    /** lookups answered by the second provider after an adsb.lol failure, or while it is only probed */
    fallback_answers: number;
    /** lookups that did not ask adsb.lol because it is only being probed */
    breaker_skips: number;
    primary_consecutive_failures: number;
    primary_probe_only: boolean;
    /** lookups that did not ask adsb.lol because its daily quota is spent or kept for the watched planes */
    quota_skips: number;
  };
}

export function createFastLane(deps: FastLaneDeps): FastLane {
  const now = deps.now ?? Date.now;
  const fetchImpl = deps.fetchImpl ?? fetch;
  const gov = deps.governor ?? adsbLolGovernor;
  const headers = deps.headers ?? adsbLolRequestHeaders(deps.env);
  const timeoutMs = deps.timeoutMs ?? FAST_TIMEOUT_MS;
  const cache = new Map<string, { at: number; res: FastResult }>();
  const inflight = new Map<string, Promise<FastResult>>();
  /** hex -> last time it went upstream (distinct-hex window) */
  const hexSeen = new Map<string, number>();
  /** hex -> broadcast vertical rate keyed to the fix it came with */
  const baro = new Map<string, { seenAt: number; fpm: number }>();
  let tokens = FAST_BURST;
  let lastRefill = now();
  const counts = { upstream_requests: 0, coalesced: 0, cache_hits: 0, snapshot_hits: 0, limited: 0, errors: 0, fallback_rows: 0, fallback_answers: 0, breaker_skips: 0, quota_skips: 0 };
  /** consecutive adsb.lol failures on this lane, and when it was last asked */
  let primaryFails = 0;
  let primaryLastTryAt = -Infinity;
  const probeOnly = (t: number): boolean =>
    !!deps.fallbackHex && primaryFails >= FAST_BREAKER_FAILURES && t - primaryLastTryAt < FAST_BREAKER_PROBE_MS;

  const baroFor = (r: SnapRow | null | undefined): number | null => {
    if (!r) return null;
    const b = baro.get(r.hex);
    return b && b.seenAt === r.seenAt ? b.fpm : null;
  };
  const pruneHexes = (t: number) => {
    for (const [h, at] of hexSeen) if (t - at > 60_000) hexSeen.delete(h);
    for (const [h, c] of cache) if (t - c.at > 60_000) cache.delete(h);
    if (baro.size > 500) baro.clear();
  };
  const takeToken = (t: number): boolean => {
    tokens = Math.min(FAST_BURST, tokens + (Math.max(0, t - lastRefill) / 1000) * FAST_MAX_RPS);
    lastRefill = t;
    if (tokens < 1) return false;
    tokens -= 1;
    return true;
  };

  /** adsb.lol failed or is only being probed: the second provider's row for
   *  this hex when it has one (published with its own provenance), else the
   *  held row with its real age and source "error". */
  const viaFallback = async (hex: string, upstreamStatus: number | null): Promise<FastResult> => {
    const alt = deps.fallbackHex ? await deps.fallbackHex(hex).catch(() => []) : [];
    if (alt.length) {
      counts.fallback_rows += alt.length;
      counts.fallback_answers++;
      deps.publish({ provider: "adsbfi", origin: "fastlane", aircraft: alt, fetchedAt: now(), upstreamNowMs: null });
    } else {
      counts.errors++;
    }
    const row = deps.snapshotGet(hex) ?? null;
    return { row, baroRateFpm: baroFor(row), source: alt.length ? "fallback" : "error", upstreamStatus };
  };

  const goUpstream = async (hex: string): Promise<FastResult> => {
    const t0 = now();
    counts.upstream_requests++;
    hexSeen.set(hex, t0);
    primaryLastTryAt = t0;
    gov.noteRequest("fg", t0);
    let status: number | null = null;
    try {
      const r = await fetchImpl(fastLaneUrl(hex), { headers, signal: AbortSignal.timeout(timeoutMs) });
      status = r.status;
      gov.noteQuota(r.headers, now(), r.status);
      if (!r.ok) {
        gov.noteResult("fg", r.status, now(), parseRetryAfter(r.headers?.get?.("retry-after") ?? null));
        throw new Error(`adsb.lol hex ${r.status}`);
      }
      const raw = (await r.json()) as { ac?: Array<Record<string, unknown>>; now?: unknown };
      gov.noteResult("fg", 200, now());
      primaryFails = 0;
      const mapped = mapPointAircraft(raw, "ac", "adsblol")
        .filter((a: { icao24?: unknown }) => String(a.icao24 || "").toLowerCase() === hex);
      const upstreamNowMs = typeof raw?.now === "number" && Number.isFinite(raw.now) ? raw.now : null;
      if (mapped.length) {
        deps.publish({ provider: "adsblol", origin: "fastlane", aircraft: mapped, fetchedAt: now(), upstreamNowMs });
      } else if (deps.fallbackHex) {
        const alt = await deps.fallbackHex(hex).catch(() => []);
        if (alt.length) {
          counts.fallback_rows += alt.length;
          deps.publish({ provider: "adsbfi", origin: "fastlane", aircraft: alt, fetchedAt: now(), upstreamNowMs: null });
        }
      }
      const row = deps.snapshotGet(hex) ?? null;
      const ac = (raw?.ac || []).find((a) => String(a?.hex || "").toLowerCase() === hex);
      const br = ac && typeof ac.baro_rate === "number" && Number.isFinite(ac.baro_rate) ? ac.baro_rate
        : ac && typeof ac.geom_rate === "number" && Number.isFinite(ac.geom_rate) ? ac.geom_rate : null;
      if (row && br != null && mapped.length) {
        // key the rate to the fix the snapshot now holds for this hex, only
        // when that fix IS the one just fetched (freshest-wins merge)
        const base = upstreamNowMs ?? now();
        const sp = typeof ac?.seen_pos === "number" ? ac.seen_pos : 0;
        if (Math.abs(row.seenAt - (base - sp * 1000)) <= 1_500) baro.set(hex, { seenAt: row.seenAt, fpm: br });
      }
      return { row, baroRateFpm: baroFor(row), source: "upstream", upstreamStatus: status };
    } catch (e) {
      if (status == null) gov.noteResult("fg", 0, now()); // network error / timeout
      primaryFails++;
      return viaFallback(hex, status);
    }
  };

  return {
    async lookup(hexIn: string): Promise<FastResult> {
      const hex = String(hexIn || "").toLowerCase();
      if (!ICAO_HEX_LC_RE.test(hex)) throw new Error("icao24 hex required");
      const t = now();
      pruneHexes(t);
      const snap = deps.snapshotGet(hex);
      if (snap && t - snap.seenAt <= FAST_FRESH_MS) {
        counts.snapshot_hits++;
        return { row: snap, baroRateFpm: baroFor(snap), source: "snapshot", upstreamStatus: null };
      }
      const c = cache.get(hex);
      if (c && t - c.at <= FAST_CACHE_MS) {
        counts.cache_hits++;
        // the snapshot may have merged something newer since
        const row = deps.snapshotGet(hex) ?? c.res.row;
        return { ...c.res, row, baroRateFpm: baroFor(row), source: "cache" };
      }
      const pending = inflight.get(hex);
      if (pending) { counts.coalesced++; return pending; }
      const newHex = !hexSeen.has(hex);
      if ((newHex && hexSeen.size >= FAST_MAX_HEXES_PER_MIN) || !takeToken(t)) {
        counts.limited++;
        return { row: snap ?? null, baroRateFpm: baroFor(snap), source: snap ? "limited" : "absent", upstreamStatus: null };
      }
      // the daily quota first (spent, or kept for the watched planes), then the breaker
      const quotaHeld = !gov.admit("selected", t);
      const skipPrimary = quotaHeld || probeOnly(t);
      if (quotaHeld) { counts.quota_skips++; hexSeen.set(hex, t); }
      else if (skipPrimary) { counts.breaker_skips++; hexSeen.set(hex, t); }
      const p = (skipPrimary ? viaFallback(hex, null) : goUpstream(hex)).then((res) => {
        cache.set(hex, { at: now(), res });
        return res;
      }).finally(() => { inflight.delete(hex); });
      inflight.set(hex, p);
      return p;
    },
    stats() {
      pruneHexes(now());
      return { ...counts, hexes_last_min: hexSeen.size, primary_consecutive_failures: primaryFails, primary_probe_only: probeOnly(now()) };
    },
  };
}
