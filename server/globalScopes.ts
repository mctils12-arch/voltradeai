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

import fs from "fs";
import { archiveAircraft, archiveBaseDir, type AircraftPoint, type SitePoint } from "./datacoreArchive";
import { mapPointAircraft } from "./aircraftTiling";

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
}

/** One cycle, pure-injectable: fetch each scope, map through the SAME
 *  normalizer as every other aircraft path, archive with standard adaptive
 *  thinning (at a 60s cycle the cruise-75s rule keeps nearly every fix). */
export async function pollGlobalScopesOnce(
  fetchImpl: typeof fetch,
  headers: Record<string, string>,
  sites: SitePoint[],
  baseDir?: string,
): Promise<GlobalCycleStatus> {
  const base = baseDir || archiveBaseDir();
  const free = readFreeBytes(base);
  const status: GlobalCycleStatus = {
    at: Date.now(), per_scope: {}, archived: 0,
    skipped_low_disk: !volumeAllowsWrite(free), free_bytes: free,
  };
  const all: AircraftPoint[] = [];
  const seen = new Set<string>();
  for (const scope of GLOBAL_SCOPES) {
    try {
      const res = await fetchImpl(scopeUrl(scope), { headers });
      if (!res.ok) { status.per_scope[scope] = -1; continue; } // -1 = fetch failed this cycle
      const pts = mapPointAircraft(await res.json(), "ac") as AircraftPoint[];
      status.per_scope[scope] = pts.length;
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
  return status;
}

/** Boot the archiver; unref'd interval, last-cycle status readable. */
export function startGlobalScopes(deps: {
  fetchImpl: typeof fetch;
  headers: Record<string, string>;
  sites: SitePoint[];
  baseDir?: string;
}): { getStatus: () => GlobalCycleStatus | null; stop: () => void } {
  let last: GlobalCycleStatus | null = null;
  let stopped = false;
  let lowDiskWarned = false;
  const tick = async () => {
    if (stopped) return;
    try {
      last = await pollGlobalScopesOnce(deps.fetchImpl, deps.headers, deps.sites, deps.baseDir);
      if (last.skipped_low_disk && !lowDiskWarned) {
        lowDiskWarned = true;
        console.error(`[global-scopes] LOW DISK (${last.free_bytes} free) — global archiving paused; bot state writes take priority`);
      } else if (!last.skipped_low_disk && lowDiskWarned) {
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
