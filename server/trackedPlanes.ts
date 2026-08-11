// NONSTOP PLANE TRACKER (human directive 2026-08-08: "track all the data
// the adsb data nonstop when there signal pops up... use this plane n843s").
//
// The viewport archive only records planes inside discs some viewer's map
// requested — nobody looking = no data. This module tracks SPECIFIC tail
// numbers server-side, around the clock: one batched adsb.lol registration
// query (/v2/reg/A,B,C — a single request regardless of count) every 30s,
// every returned fix archived with the thinning OVERRIDDEN so nothing is
// dropped (ground taxi included). When a tracked plane has no signal the
// poll simply returns nothing for it — last_seen says how long it's been
// dark, honestly, and the moment its signal pops up the fixes flow again.
//
// Provider politeness: ONE request per cycle at a fixed 30s cadence,
// registry capped at TRACKED_CAP tails. adsb.lol is the monetization-lawful
// primary (ODbL) — same provider, same UA as the viewport path.

import fs from "fs";
import path from "path";
import { archiveAircraft, archiveBaseDir, type AircraftPoint, type SitePoint } from "./datacoreArchive";
import { mapPointAircraft } from "./aircraftTiling";

export const TRACKED_CAP = 20;
export const TRACKED_POLL_MS = 30_000;
/** archive EVERY polled fix — the poll cadence IS the sampling cadence */
export const TRACKED_ARCHIVE_INTERVAL_MS = 15_000;

export interface TrackedPlane {
  reg: string;                 // normalized tail number — the registry key
  added_at: number;            // unix ms
  hex?: string | null;         // learned from the first live fix
  last_seen?: number | null;   // unix ms of the newest fix we archived
  last_pos?: { la: number; lo: number; al: number | null } | null;
  note?: string;
}

export interface TrackedRegistry { planes: TrackedPlane[] }

/** Tail numbers are ICAO nationality mark + alphanumerics (N843S, G-EZTL,
 *  VH-OQA…). Uppercased; junk rejected rather than stored. */
export function normalizeReg(reg: string | null | undefined): string | null {
  const r = String(reg || "").trim().toUpperCase();
  return /^[A-Z0-9][A-Z0-9-]{1,9}$/.test(r) ? r : null;
}

export function registryPath(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "aircraft_tracked.json");
}

/** Load (or seed) the registry. Seeded with the human's named example so
 *  tracking starts the moment this ships — no empty-state dead end. */
export function loadRegistry(baseDir?: string): TrackedRegistry {
  const p = registryPath(baseDir);
  try {
    const d = JSON.parse(fs.readFileSync(p, "utf-8"));
    if (Array.isArray(d?.planes)) {
      return { planes: d.planes.filter((x: any) => normalizeReg(x?.reg)) };
    }
  } catch { /* absent or corrupt -> seed */ }
  return {
    planes: [{
      reg: "N843S", added_at: Date.now(),
      note: "seed — human directive 2026-08-08 (Falcon 7X, hex ab8c8e at seed time)",
    }],
  };
}

export function saveRegistry(r: TrackedRegistry, baseDir?: string): void {
  const p = registryPath(baseDir);
  try {
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, JSON.stringify(r, null, 1));
  } catch (e: any) {
    console.error("[tracked-planes] save:", e?.message || e);
  }
}

/** Pure: add a tail (idempotent, capped). Returns null when full/invalid. */
export function addTracked(r: TrackedRegistry, reg: string): TrackedRegistry | null {
  const n = normalizeReg(reg);
  if (!n) return null;
  if (r.planes.some((p) => p.reg === n)) return r; // idempotent
  if (r.planes.length >= TRACKED_CAP) return null;
  return { planes: [...r.planes, { reg: n, added_at: Date.now() }] };
}

export function removeTracked(r: TrackedRegistry, reg: string): TrackedRegistry {
  const n = normalizeReg(reg);
  return n ? { planes: r.planes.filter((p) => p.reg !== n) } : r;
}

export function batchUrl(regs: string[]): string {
  return `https://api.adsb.lol/v2/reg/${regs.map((r) => encodeURIComponent(r)).join(",")}`;
}

/** One poll cycle, pure-injectable: fetch the batch, map through the SAME
 *  normalizer as the viewport path, and fold updates into the registry.
 *  Planes with no signal are untouched (last_seen keeps aging — honest). */
export async function pollTrackedOnce(
  registry: TrackedRegistry,
  fetchImpl: typeof fetch,
  headers: Record<string, string>,
  nowMs: number,
): Promise<{ points: AircraftPoint[]; registry: TrackedRegistry }> {
  if (!registry.planes.length) return { points: [], registry };
  const res = await fetchImpl(batchUrl(registry.planes.map((p) => p.reg)), { headers });
  if (!res.ok) throw new Error(`adsb.lol reg batch ${res.status}`);
  const raw = await res.json();
  const points = mapPointAircraft(raw, "ac") as AircraftPoint[];
  const byReg = new Map<string, AircraftPoint>();
  for (const p of points) {
    const reg = normalizeReg((p as any).registration);
    if (reg) byReg.set(reg, p);
  }
  const planes = registry.planes.map((tp) => {
    const hit = byReg.get(tp.reg);
    if (!hit) return tp;
    return {
      ...tp,
      hex: hit.icao24 || tp.hex,
      last_seen: nowMs,
      last_pos: { la: hit.lat, lo: hit.lon, al: hit.altitude_m ?? null },
    };
  });
  return { points, registry: { planes } };
}

/** Boot the poller. Deps injectable for tests; the interval is unref'd so
 *  it never holds a shutdown. Registry mutations from the API layer land
 *  via getRegistry/setRegistry closures (single in-process owner). */
export function startTrackedPoller(deps: {
  fetchImpl: typeof fetch;
  headers: Record<string, string>;
  sites: SitePoint[];
  baseDir?: string;
  onCycle?: (archived: number, live: number) => void;
}): {
  getRegistry: () => TrackedRegistry;
  mutate: (fn: (r: TrackedRegistry) => TrackedRegistry | null) => boolean;
  stop: () => void;
} {
  let registry = loadRegistry(deps.baseDir);
  saveRegistry(registry, deps.baseDir); // persist the seed on first boot
  let stopped = false;
  const tick = async () => {
    if (stopped || !registry.planes.length) return;
    try {
      const now = Date.now();
      const out = await pollTrackedOnce(registry, deps.fetchImpl, deps.headers, now);
      registry = out.registry;
      const archived = out.points.length
        ? archiveAircraft(out.points, deps.sites, deps.baseDir, now, TRACKED_ARCHIVE_INTERVAL_MS)
        : 0;
      if (out.points.length) saveRegistry(registry, deps.baseDir);
      deps.onCycle?.(archived, out.points.length);
    } catch (e: any) {
      // a failed cycle is just a missed sample — next tick retries; the
      // viewport chain's backoff machinery is not needed at 2 req/min
      console.error("[tracked-planes] poll:", e?.message || e);
    }
  };
  const timer = setInterval(() => { void tick(); }, TRACKED_POLL_MS);
  (timer as any).unref?.();
  void tick(); // first cycle immediately — "when their signal pops up"
  return {
    getRegistry: () => registry,
    mutate: (fn) => {
      const next = fn(registry);
      if (!next) return false;
      registry = next;
      saveRegistry(registry, deps.baseDir);
      return true;
    },
    stop: () => { stopped = true; clearInterval(timer); },
  };
}
