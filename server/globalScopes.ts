// GLOBAL SCOPES ARCHIVER (human directive 2026-08-08: "we have the data all
// over the world"). adsb.lol's public API has NO all-aircraft endpoint
// (verified against its OpenAPI spec during design) — but three scopes ARE
// served globally in one request each: /v2/mil (military), /v2/ladd (FAA
// privacy-program), /v2/pia (privacy ICAO addresses). Together ~1,000+
// aircraft WORLDWIDE per cycle, archived continuously. All-civil global
// live exists only as a paid product (ADSBExchange) — filed in wishlist.md
// with the build-first analysis; this module archives everything the free
// tier lawfully offers.
//
// VOLUME GUARD: global-scope archiving grows the archive ~10x faster than
// viewport recording. The bot's state writes on the same volume outrank
// archive growth (GOAL priority 1) — under MIN_FREE_BYTES of headroom the
// archiver SKIPS writes and says so, rather than filling the disk.
//
// ADSB.LOL'S DAILY QUOTA (2026-11-17, adsbGovernor.ts). adsb.lol counts
// requests per day (500 without a key); these three scopes every 120 s are
// 2,160 a day. Each request carries the API key when one is configured and
// reports the response's quota headers to the governor, and the poller
// paces itself by the quota mode (createScopesPacer):
//   - no limit read yet: all three every 120 s, as before;
//   - "full": all three, the cycle spaced so a day of it uses at most
//     SCOPES_QUOTA_SHARE of the limit (7,200/day -> every 360 s);
//   - "watched_only": mil and pia are NOT asked. The LADD scope still is,
//     once per LADD_SCOPE_WATCHED_GAP_MS (15 min = 96 requests a day): the
//     LADD registry (laddRegistry.ts) decides whether FAA-sourced flight
//     data may be shown for an aircraft, and it must keep learning;
//   - quota spent: nothing is asked until the reset. The registry then
//     learns only from the LADD bit on other providers' rows (laddRegistry.ts).
// A scope that was not asked has no per_scope entry; the cycle says
// skipped_quota (held by the quota) or paced_wait (not due yet).

import fs from "fs";
import { archiveAircraft, archiveBaseDir, type AircraftPoint, type SitePoint } from "./datacoreArchive";
import { mapPointAircraft } from "./aircraftTiling";
import { publishFixes } from "./aircraftFixBus";
import { laddRegistry } from "./laddRegistry";
import { adsbLolGovernor, adsbLolKeyHeader, type UpstreamGovernor } from "./adsbGovernor";

/** "watched_only": the LADD scope alone is asked, once per this long
 *  (96 requests a day, counted against adsb.lol's daily quota) */
export const LADD_SCOPE_WATCHED_GAP_MS = 15 * 60_000;

/** What the daily quota allows the scopes poller, tick by tick. */
export interface ScopesPacer {
  /** start of a tick: false = "full" mode and the last cycle's share of the
   *  day has not passed yet (nothing is asked this tick) */
  cycleDue(now: number): boolean;
  /** one scope: "ask"; "quota" = held by the quota (spent, or kept for the
   *  watched planes); "wait" = the LADD scope's 15 minutes have not passed */
  decide(scope: GlobalScope, now: number): "ask" | "quota" | "wait";
  /** a request for `scope` is being sent */
  asked(scope: GlobalScope, now: number): void;
}

export function createScopesPacer(governor: UpstreamGovernor): ScopesPacer {
  let lastLaddAt: number | null = null;
  /** the cycle in progress or last run: when its first request went, and how many */
  let cycleAt: number | null = null;
  let cycleRequests = 0;
  let cycleOpen = false;
  return {
    cycleDue(now) {
      cycleOpen = false;
      const gap = governor.scopesMinGapMs(now);
      // 1 s of slack: the 120 s timer never has to wait a whole extra tick for a few ms
      return !(gap > 0 && cycleAt != null && now - cycleAt < gap * Math.max(1, cycleRequests) - 1_000);
    },
    decide(scope, now) {
      if (scope !== "ladd") return governor.admit("scopes", now) ? "ask" : "quota";
      if (!governor.admit("ladd", now)) return "quota";
      if (governor.quotaMode(now) !== "watched_only") return "ask";
      return lastLaddAt == null || now - lastLaddAt >= LADD_SCOPE_WATCHED_GAP_MS ? "ask" : "wait";
    },
    asked(scope, now) {
      if (!cycleOpen) { cycleOpen = true; cycleAt = now; cycleRequests = 0; }
      cycleRequests++;
      if (scope === "ladd") lastLaddAt = now;
    },
  };
}

export const GLOBAL_SCOPES = ["mil", "ladd", "pia"] as const;
export type GlobalScope = (typeof GLOBAL_SCOPES)[number];
// 60s -> 120s (2026-08-11): with the viewport discs + tracked poller +
// these scopes all leaving one Railway IP, prod showed adsb.lol partially
// failing over to airplanes.live and pia fetches erroring — provider
// pressure is real. Half the global cadence costs little (mil/ladd/pia
// membership changes slowly); the viewport chain keeps its own backoff.
export const GLOBAL_POLL_MS = 120_000;
export const MIN_FREE_BYTES = 1_073_741_824; // 1 GiB headroom for the bot

export function scopeUrl(scope: GlobalScope): string {
  return `https://api.adsb.lol/v2/${scope}`;
}

/** Pure guard decision: free-space reading -> write allowed? */
export function volumeAllowsWrite(freeBytes: number | null, minFree = MIN_FREE_BYTES): boolean {
  // an UNREADABLE reading fails open (viewport archiving has never been
  // guarded; blocking on a stat error would silently kill the archive) —
  // the guard exists for the measured-low case
  return freeBytes == null || freeBytes >= minFree;
}

export function readFreeBytes(dir: string): number | null {
  try {
    const s = (fs as any).statfsSync?.(dir);
    return s ? Number(s.bavail) * Number(s.bsize) : null;
  } catch { return null; }
}

export interface GlobalCycleStatus {
  at: number;
  per_scope: Record<string, number>; // aircraft returned per scope
  archived: number;                  // lines actually written (post-thinning)
  skipped_low_disk: boolean;
  free_bytes: number | null;
  /** true when at least one scope was NOT asked because adsb.lol's daily
   *  quota is spent or kept for the watched planes (no per_scope entry) */
  skipped_quota: boolean;
  /** true when a scope was not asked because its paced turn has not come
   *  (the LADD scope's 15 minutes, or the "full" mode share) */
  paced_wait: boolean;
  /** requests this cycle sent */
  asked: number;
}

/** One cycle, pure-injectable: fetch each scope, map through the SAME
 *  normalizer as every other aircraft path, archive with standard adaptive
 *  thinning (at a 60s cycle the cruise-75s rule keeps nearly every fix). */
export async function pollGlobalScopesOnce(
  fetchImpl: typeof fetch,
  headers: Record<string, string>,
  sites: SitePoint[],
  baseDir?: string,
  // LADD membership feeds server/laddRegistry.ts so FAA-sourced data
  // (filed plans) is withheld for these aircraft
  noteLadd: (rows: AircraftPoint[]) => void = (rows) => laddRegistry().note(rows),
  // the shared adsb.lol governor: reads each response's daily-quota headers
  // and says whether a scope may be asked at all (null = no quota handling)
  governor: UpstreamGovernor | null = null,
  // the poller's pacing state across ticks (default: a fresh one, so a
  // single call asks whatever the quota mode allows)
  pacer: ScopesPacer | null = governor ? createScopesPacer(governor) : null,
): Promise<GlobalCycleStatus> {
  const base = baseDir || archiveBaseDir();
  const free = readFreeBytes(base);
  const status: GlobalCycleStatus = {
    at: Date.now(), per_scope: {}, archived: 0,
    skipped_low_disk: !volumeAllowsWrite(free), free_bytes: free, skipped_quota: false, paced_wait: false, asked: 0,
  };
  // "full" mode: this cycle's share of the day has not come round yet
  if (pacer && !pacer.cycleDue(Date.now())) { status.paced_wait = true; return status; }
  const all: AircraftPoint[] = [];
  const seen = new Set<string>();
  for (const scope of GLOBAL_SCOPES) {
    // the daily quota: spent, or kept for the watched planes (mil/pia), or
    // the LADD scope's 15 minutes have not passed -> not asked
    const turn = pacer ? pacer.decide(scope, Date.now()) : "ask";
    if (turn === "quota") { status.skipped_quota = true; continue; }
    if (turn === "wait") { status.paced_wait = true; continue; }
    pacer?.asked(scope, Date.now());
    status.asked++;
    try {
      const res = await fetchImpl(scopeUrl(scope), { headers });
      governor?.noteQuota(res.headers, Date.now(), res.status);
      if (!res.ok) { status.per_scope[scope] = -1; continue; } // -1 = fetch failed this cycle
      // scopes are always adsb.lol (scopeUrl → api.adsb.lol) — tag provenance
      const pts = mapPointAircraft(await res.json(), "ac", "adsblol") as AircraftPoint[];
      status.per_scope[scope] = pts.length;
      if (scope === "ladd") {
        try { noteLadd(pts); } catch (e) { console.error("[global-scopes] ladd registry:", e instanceof Error ? e.message : String(e)); }
      }
      for (const p of pts) {
        if (seen.has(p.icao24)) continue; // scopes overlap (mil ∩ ladd)
        seen.add(p.icao24);
        all.push(p);
      }
    } catch {
      status.per_scope[scope] = -1;
    }
  }
  if (!status.skipped_low_disk && all.length) {
    status.archived = archiveAircraft(all, sites, baseDir);
  }
  // FLIGHT PROGRAM B1: the global live snapshot reuses these fixes (fix
  // bus; archiving above stays this path's own job — never written twice)
  if (all.length) {
    publishFixes({ provider: "adsblol", origin: "scopes", aircraft: all, fetchedAt: Date.now() });
  }
  return status;
}

/** Boot the archiver; unref'd interval, last-cycle status readable. */
export function startGlobalScopes(deps: {
  fetchImpl: typeof fetch;
  headers: Record<string, string>;
  sites: SitePoint[];
  baseDir?: string;
  /** the shared adsb.lol governor (default the process singleton; null = no quota handling) */
  governor?: UpstreamGovernor | null;
  /** where ADSBLOL_API_KEY is read (default process.env) */
  env?: NodeJS.ProcessEnv;
}): { getStatus: () => GlobalCycleStatus | null; stop: () => void } {
  let last: GlobalCycleStatus | null = null;
  let stopped = false;
  let lowDiskWarned = false;
  let quotaSkipLogged = false;
  const governor = deps.governor === undefined ? adsbLolGovernor : deps.governor;
  const pacer = governor ? createScopesPacer(governor) : null;
  // all three scopes are api.adsb.lol: the key rides on them
  const headers = { ...deps.headers, ...adsbLolKeyHeader(deps.env) };
  const tick = async () => {
    if (stopped) return;
    try {
      const st = await pollGlobalScopesOnce(deps.fetchImpl, headers, deps.sites, deps.baseDir, undefined, governor, pacer);
      // the status shown is the newest cycle that ASKED something; a tick
      // that only waited updates its quota flag
      last = st.asked > 0 || !last || last.asked === 0 ? st : { ...last, skipped_quota: st.skipped_quota, paced_wait: st.paced_wait };
      // said once per stretch, never every 120 s
      if (st.skipped_quota && !quotaSkipLogged) {
        quotaSkipLogged = true;
        console.error("[global-scopes] adsb.lol's daily quota: mil and pia are not asked; the LADD scope is asked once per 15 min while the quota is kept for the watched planes, and not at all while it is spent");
      } else if (!st.skipped_quota && !st.paced_wait && quotaSkipLogged) {
        quotaSkipLogged = false;
        console.error("[global-scopes] asking adsb.lol for mil/ladd/pia again");
      }
      if (st.skipped_low_disk && !lowDiskWarned) {
        lowDiskWarned = true;
        console.error(`[global-scopes] LOW DISK (${st.free_bytes} free) — global archiving paused; bot state writes take priority`);
      } else if (!st.skipped_low_disk && lowDiskWarned) {
        lowDiskWarned = false;
        console.error("[global-scopes] disk headroom recovered — global archiving resumed");
      }
    } catch (e: any) {
      console.error("[global-scopes] cycle:", e?.message || e);
    }
  };
  const timer = setInterval(() => { void tick(); }, GLOBAL_POLL_MS);
  (timer as any).unref?.();
  void tick();
  return { getStatus: () => last, stop: () => { stopped = true; clearInterval(timer); } };
}
