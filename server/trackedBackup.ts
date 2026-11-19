// trackedBackup.ts — the tracked planes are also asked of adsb.fi, by ICAO
// hex, every few seconds (2026-11-19, owner: a watched plane must be seen
// within seconds of broadcasting, with no API key and at no cost).
//
// WHY. The tracked poller (trackedPlanes.ts) asks adsb.lol only. adsb.lol
// counts requests per day and gives 500 to an address without a key
// (adsbGovernor.ts); the poller is then paced to one request per 432 s, and a
// cycle makes two, so each watched plane was checked about every 14 minutes.
//
// WHAT. One request to adsb.fi per tick for ALL tracked planes whose hex is
// known (https://opendata.adsb.fi/api/v2/hex/<hex1>,<hex2>,... — read live
// 2026-11-19 with two airliners: keys ac, ctime, msg, now, ptime, total; both
// aircraft in `ac`, the readsb row schema adsb.lol uses; `now` came in
// milliseconds on this endpoint, the disc endpoint sends seconds, and
// aircraftFixBus.providerClockMs reads either). Every 10 s while any tracked
// plane was heard in the last 5 minutes, every 30 s otherwise: at most 0.1
// request a second, against adsb.fi's published limit of 1.
//
// The rows go through the SAME fold as adsb.lol's (trackedPlanes
// .foldTrackedRows): a row with a position is a fix, a row without one is
// `heard` only, and a row that names another registration is not this plane
// (identity is never inferred). A row for a hex that was not asked for is
// dropped before the fold. Every fix is mapped with provider "adsbfi", so the
// worldwide snapshot's `src`, the archive's `pv` and the tracked planes' log
// say which provider served it, and adsb.lol's ODbL subset stays one filter.
//
// WHAT IT IS NOT. It does not find a plane whose hex is not known yet: that
// is still adsb.lol's registration query (status().planes_without_hex counts
// them). It does not replace adsb.lol, whose pacing is unchanged.
//
// LICENSING. adsb.fi is NON-COMMERCIAL (providerCompliance.ts, where the
// monetization tripwire already counts it). This poller sends nothing while
// billing is active.
//
// HEADERS. A plain User-Agent. The adsb.lol API key (adsbLolKeyHeader /
// adsbLolRequestHeaders) is for api.adsb.lol only and is never built here.
//
// KILL SWITCH: TRACKED_BACKUP_ENABLED=0 (default on).
//
// IMPORT NOTE. This module and trackedPlanes.ts import each other
// (startTrackedPoller builds the backup; the backup folds with
// foldTrackedRows). Every use of the other module is inside a function, so
// the load order does not matter. Keep it that way: no top-level statement
// here may read a value imported from trackedPlanes.ts.

import { mapPointAircraft } from "./aircraftTiling";
import { parseRetryAfter } from "./adsbGovernor";
import { UA } from "./globalSweep";
import { billingActive } from "./providerCompliance";
import { failoverBackoffMs } from "./sweepFailover";
import type { AircraftPoint } from "./datacoreArchive";
import { activeHexes, foldTrackedRows, rawAircraftRows, upstreamBaseMs, type TrackedRegistry } from "./trackedPlanes";

/** the provider key every fix of this poller carries */
export const TRACKED_BACKUP_PROVIDER = "adsbfi";
export const TRACKED_BACKUP_LABEL = "adsb.fi (ADS-B, community; non-commercial)";
/** cadence while any tracked plane is active (trackedPlanes.isTrackedActive) */
export const TRACKED_BACKUP_POLL_MS = 10_000;
/** cadence while none is */
export const TRACKED_BACKUP_IDLE_POLL_MS = 30_000;
/** a request that hangs must not stall the self-rescheduling loop */
export const TRACKED_BACKUP_TIMEOUT_MS = 8_000;
/** the loop's first request, after boot (adsb.lol's first cycle goes at once) */
export const TRACKED_BACKUP_START_DELAY_MS = 2_000;
/** failures are logged on the 1st and then every this many: loud, not a flood */
export const TRACKED_BACKUP_LOG_EVERY = 20;

const HEX_RE = /^[0-9a-f]{6}$/;
const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** The tracked planes' known ICAO hexes: lower case, valid only, each once. */
export function trackedBackupHexes(registry: TrackedRegistry): string[] {
  const out = new Set<string>();
  for (const tp of registry.planes) {
    const h = String(tp.hex ?? "").trim().toLowerCase();
    if (HEX_RE.test(h)) out.add(h);
  }
  return Array.from(out);
}

/** ONE adsb.fi request for several aircraft by hex; null when no hex is valid. */
export function trackedBackupUrl(hexes: string[]): string | null {
  const ok = Array.from(new Set(hexes.map((h) => String(h).trim().toLowerCase()).filter((h) => HEX_RE.test(h))));
  return ok.length ? `https://opendata.adsb.fi/api/v2/hex/${ok.join(",")}` : null;
}

/** Why the backup is off, or null when it may run. */
export function trackedBackupOffReason(env: NodeJS.ProcessEnv = process.env): string | null {
  const v = String(env.TRACKED_BACKUP_ENABLED ?? "1").trim().toLowerCase();
  if (v === "0" || v === "false" || v === "off" || v === "no") return "TRACKED_BACKUP_ENABLED=0";
  if (billingActive(env)) return "billing is active: adsb.fi is non-commercial";
  return null;
}

/** ms between two requests: 10 s while a tracked plane is active, else 30 s. */
export function trackedBackupPollMs(registry: TrackedRegistry, nowMs: number): number {
  return activeHexes(registry, nowMs).length ? TRACKED_BACKUP_POLL_MS : TRACKED_BACKUP_IDLE_POLL_MS;
}

export const TRACKED_BACKUP_RULE =
  "One request to adsb.fi (non-commercial) per tick for every tracked plane whose ICAO hex is known: every " +
  `${TRACKED_BACKUP_POLL_MS / 1000} s while a tracked plane was heard in the last 5 min, every ` +
  `${TRACKED_BACKUP_IDLE_POLL_MS / 1000} s otherwise. A plane whose hex is not known yet is first found by ` +
  "adsb.lol's registration query (planes_without_hex). A row with a position is a fix and keeps provider " +
  "adsbfi; a row without one is `heard` only; a row naming another registration is ignored. A failed request " +
  "backs off 30 s doubling to 15 min (Retry-After honoured, 401/403 30 min): see backoff_until. Sends nothing " +
  "while billing is active or with TRACKED_BACKUP_ENABLED=0.";

export interface TrackedBackupStatus {
  enabled: boolean;
  /** why it is off (null when enabled) */
  off_reason: string | null;
  provider: string;
  label: string;
  /** seconds between two requests as they run now; null while off */
  poll_s: number | null;
  active_poll_s: number;
  idle_poll_s: number;
  /** tracked planes one request asks for (hex known), and those it cannot ask for yet */
  planes_asked: number;
  planes_without_hex: number;
  requests_total: number;
  ok_total: number;
  failed_total: number;
  consecutive_failures: number;
  /** when the last request was sent / last answered 200 (unix ms) */
  last_at: number | null;
  last_ok_at: number | null;
  /** HTTP status of the last request; 0 = network error or timeout */
  last_status: number | null;
  last_error: string | null;
  /** rows for the asked hexes in the last 200 answer */
  rows_last: number | null;
  /** position fixes / heard-without-position readings since boot */
  fixes_total: number;
  heard_total: number;
  /** no request is sent before this time (unix ms); null = not backing off */
  backoff_until: number | null;
  rule: string;
}

export type TrackedBackupSkip = "off" | "backoff" | "no-hex";

export interface TrackedBackupOutcome {
  /** POSITION fixes only, each tagged provider "adsbfi" */
  points: AircraftPoint[];
  /** the registry this poll started from, with its readings folded in */
  registry: TrackedRegistry;
  /** tracked planes heard WITHOUT a position */
  heard: number;
  /** upstream requests made: 0 or 1 */
  requests: number;
  /** true when the request was answered 200 with an aircraft list */
  ok: boolean;
  /** why no request was sent; null when one was */
  skipped: TrackedBackupSkip | null;
  /** HTTP status (0 = network error or timeout); null when nothing was sent */
  status: number | null;
  error: string | null;
  /** adsb.fi's own clock for the answer, ms, never ahead of ours */
  upstreamNowMs: number | null;
}

export interface TrackedBackup {
  /** one tick: at most ONE request, for every tracked plane with a known hex.
   *  Never throws: a failure is counted, backed off and returned. */
  poll(registry: TrackedRegistry, nowMs?: number): Promise<TrackedBackupOutcome>;
  /** ms until the next tick */
  nextDelayMs(registry: TrackedRegistry, nowMs?: number): number;
  status(registry: TrackedRegistry, nowMs?: number): TrackedBackupStatus;
}

export function createTrackedBackup(deps: {
  fetchImpl?: typeof fetch;
  env?: NodeJS.ProcessEnv;
  now?: () => number;
  timeoutMs?: number;
} = {}): TrackedBackup {
  const env = deps.env ?? process.env;
  const fetchImpl = deps.fetchImpl ?? fetch;
  const now = deps.now ?? Date.now;
  const limitMs = deps.timeoutMs ?? TRACKED_BACKUP_TIMEOUT_MS;
  const c = { requests: 0, ok: 0, failed: 0, fixes: 0, heard: 0 };
  let failures = 0; // in a row
  let backoffUntil = 0;
  let lastAt: number | null = null;
  let lastOkAt: number | null = null;
  let lastStatus: number | null = null;
  let lastError: string | null = null;
  let rowsLast: number | null = null;

  const get = async (url: string): Promise<{ status: number; raw: unknown; retryAfterMs: number | null }> => {
    // a held deadline, as in sweepFailover.ts: an abort signal that is not
    // honoured must not be able to freeze a self-rescheduling loop
    const ac = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<never>((_, reject) => {
      timer = setTimeout(() => { ac.abort(); reject(new Error(`request exceeded ${limitMs}ms hard deadline`)); }, limitMs);
    });
    // the race below reports the deadline; this only keeps a deadline that
    // fires after the race is decided from being an unhandled rejection
    deadline.catch(() => undefined);
    try {
      // UA only: the adsb.lol key header is for api.adsb.lol and is never sent here
      const r = await Promise.race([fetchImpl(url, { headers: { ...UA }, signal: ac.signal }), deadline]);
      const status = r.status ?? (r.ok ? 200 : 0);
      if (!r.ok) {
        // parseRetryAfter answers in SECONDS
        const ra = parseRetryAfter(r.headers?.get?.("retry-after") ?? null, now());
        return { status, raw: null, retryAfterMs: ra == null ? null : ra * 1000 };
      }
      return { status: 200, raw: await Promise.race([r.json(), deadline]), retryAfterMs: null };
    } finally {
      clearTimeout(timer);
    }
  };

  const failed = (status: number, error: string, retryAfterMs: number | null): void => {
    c.failed++;
    failures++;
    lastStatus = status;
    lastError = error;
    const wait = failoverBackoffMs(status, failures, retryAfterMs);
    backoffUntil = now() + wait;
    if (failures === 1 || failures % TRACKED_BACKUP_LOG_EVERY === 0) {
      console.error(`[tracked-backup] adsb.fi request failed (${error}, ${failures} in a row): waiting ${Math.ceil(wait / 1000)}s`);
    }
  };

  return {
    async poll(registry, nowMs = now()) {
      const none = (skipped: TrackedBackupSkip | null, status: number | null, error: string | null, requests: number): TrackedBackupOutcome =>
        ({ points: [], registry, heard: 0, requests, ok: false, skipped, status, error, upstreamNowMs: null });
      if (trackedBackupOffReason(env)) return none("off", null, null, 0);
      if (nowMs < backoffUntil) return none("backoff", null, null, 0);
      const hexes = trackedBackupHexes(registry);
      const url = trackedBackupUrl(hexes);
      if (!url) return none("no-hex", null, null, 0);
      c.requests++;
      lastAt = nowMs;
      let status = 0;
      let raw: unknown = null;
      try {
        const res = await get(url);
        status = res.status;
        raw = res.raw;
        if (status !== 200) {
          failed(status, `adsb.fi ${status}`, res.retryAfterMs);
          return none(null, status, lastError, 1);
        }
      } catch (e) {
        failed(0, errMsg(e), null);
        return none(null, 0, lastError, 1);
      }
      if (!Array.isArray((raw as { ac?: unknown } | null)?.ac)) {
        // a 200 that is not an aircraft list is not an answer
        failed(0, "adsb.fi answered 200 without an `ac` list", null);
        lastStatus = 200;
        return none(null, 200, lastError, 1);
      }
      const asked = new Set(hexes);
      // only the aircraft that were asked for
      const rows = rawAircraftRows(raw).filter((a) => typeof a.hex === "string" && asked.has(a.hex.toLowerCase()));
      // adsb.fi's clock for this answer, never ahead of ours AS WE RECEIVED IT
      const baseMs = upstreamBaseMs(raw, Math.max(nowMs, now()));
      const f = foldTrackedRows(registry, rows, baseMs, nowMs, TRACKED_BACKUP_PROVIDER);
      const points = mapPointAircraft({ ac: f.positioned }, "ac", TRACKED_BACKUP_PROVIDER) as AircraftPoint[];
      c.ok++;
      failures = 0;
      backoffUntil = 0;
      lastOkAt = now();
      lastStatus = 200;
      lastError = null;
      rowsLast = rows.length;
      c.fixes += points.length;
      c.heard += f.heard;
      return { points, registry: f.registry, heard: f.heard, requests: 1, ok: true, skipped: null, status: 200, error: null, upstreamNowMs: baseMs };
    },
    nextDelayMs(registry, nowMs = now()) {
      return trackedBackupPollMs(registry, nowMs);
    },
    status(registry, nowMs = now()) {
      const off = trackedBackupOffReason(env);
      const asked = trackedBackupHexes(registry).length;
      return {
        enabled: !off,
        off_reason: off,
        provider: TRACKED_BACKUP_PROVIDER,
        label: TRACKED_BACKUP_LABEL,
        poll_s: off ? null : trackedBackupPollMs(registry, nowMs) / 1000,
        active_poll_s: TRACKED_BACKUP_POLL_MS / 1000,
        idle_poll_s: TRACKED_BACKUP_IDLE_POLL_MS / 1000,
        planes_asked: asked,
        planes_without_hex: registry.planes.filter((tp) => !HEX_RE.test(String(tp.hex ?? "").trim().toLowerCase())).length,
        requests_total: c.requests,
        ok_total: c.ok,
        failed_total: c.failed,
        consecutive_failures: failures,
        last_at: lastAt,
        last_ok_at: lastOkAt,
        last_status: lastStatus,
        last_error: lastError,
        rows_last: rowsLast,
        fixes_total: c.fixes,
        heard_total: c.heard,
        backoff_until: backoffUntil > nowMs ? backoffUntil : null,
        rule: TRACKED_BACKUP_RULE,
      };
    },
  };
}

/** `source` of GET /api/data/aircraft/tracked: the providers in use NOW. */
export function trackedSourceText(backup: Pick<TrackedBackupStatus, "enabled" | "off_reason" | "active_poll_s" | "idle_poll_s">): string {
  const lol = "adsb.lol (ODbL): one batched registration query per cycle, plus one batched hex query for the tracked planes it returned no position for, paced by adsb.lol's daily quota (pacing)";
  return backup.enabled
    ? `two sources. (1) ${lol}. (2) adsb.fi (non-commercial): one batched hex query for every tracked plane whose hex is known, every ` +
      `${backup.active_poll_s} s while one is being heard and every ${backup.idle_poll_s} s otherwise (backup)`
    : `one source. ${lol}. The adsb.fi backup is off: ${backup.off_reason}`;
}

/** Appended to the `note` of GET /api/data/aircraft/tracked: which provider
 *  a plane row's readings came from. */
export const TRACKED_PROVIDER_NOTE =
  ". Providers: adsblol always, adsbfi while backup.enabled. last_pos_provider / heard_provider name the provider that served each " +
  "reading (absent on a reading taken before 2026-11-19); a newer reading is never replaced by an older one from the other provider; " +
  "every archived fix carries its provider (pv in the archive and the log, src on the live endpoints)";
