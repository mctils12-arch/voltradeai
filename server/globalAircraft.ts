// globalAircraft.ts — GET /api/data/aircraft/global: every live aircraft
// we can lawfully see, worldwide, in one compact payload (FLIGHT PROGRAM
// B1, 2026-09-28). Registered from routes.ts with ONE line.
//
// FEEDS (all through the in-process fix bus, aircraftFixBus.ts):
//   - globalSweep.ts   — adsb.lol type lane (worldwide per ICAO type) +
//                        disc lane (static 250nm world plan), governor-paced
//   - viewport chain   — every disc a viewer's map already fetched (reused,
//                        never refetched; also credits the sweep's plan)
//   - globalScopes.ts  — adsb.lol mil / ladd / pia
//   - openskyGlobal.ts — OPTIONAL, only with OPENSKY_CLIENT_ID+SECRET
//                        (non-commercial; providerCompliance.ts)
// RECORDING: sweep + OpenSky fixes go through the EXISTING archiveAircraft
// path (standard adaptive thinning + record-on-change), behind a STRICTER,
// SEPARATE volume gate than every other aircraft writer: they are archived
// only while the volume has ≥ SWEEP_ARCHIVE_MIN_FREE_BYTES (2 GiB) free —
// twice the 1 GiB floor at which globalScopes' guard pauses. The sweep is a
// ~10x-volume writer (see the growth estimate in the B1 report); without
// its own higher floor it would drain the volume down to the shared 1 GiB
// floor within days and pause ALL aircraft archiving, viewport and tracked
// planes included. This gate FAILS CLOSED (unreadable free space → skip),
// unlike the shared guard's fail-open: losing optional sweep history is
// cheap, filling the bot's volume is not. Skipped fixes are still served
// live; archived-vs-skipped counters are in the endpoint's coverage block.
// Viewport and scope fixes are archived by their own paths (never twice).
// GLOBAL_SWEEP_ARCHIVE=0 keeps the snapshot live but stops these writes.
//
// RESPONSE CONTRACT (FLIGHT_PROGRAM_BRIEF): { at, count, fields, rows,
// coverage, honesty } — rows are positional arrays in `fields` order.
// Optional query: lamin/lamax/lomin/lomax (all four) filters rows to a
// bbox AND tells the sweep where viewers are looking (interest boost);
// since=<ms> returns only rows with seenAt > since (full:false);
// lawful=1 returns only adsb.lol (ODbL) rows. Short shared cache
// (max-age=10) + Express's weak ETag; gzip via the global compression
// middleware (server/index.ts).

import type { Express } from "express";
import datacoreSites from "../datacore/sites/strategic_sites.json";
import { archiveAircraft, archiveBaseDir, type SitePoint } from "./datacoreArchive";
import { volumeAllowsWrite, readFreeBytes } from "./globalScopes";
import { subscribeFixes, type FixBatch } from "./aircraftFixBus";
import { GlobalSnapshot, ROW_FIELDS, encodeRow, type BBox } from "./globalSnapshot";
import { startGlobalSweep, unrefTimer, type SweepHandle } from "./globalSweep";
import { startOpenSkyGlobal, type OpenSkyHandle } from "./openskyGlobal";
import { complianceAuditTick } from "./providerCompliance";
import { PLAN_RADIUS_NM, TRAFFIC_MASK } from "./globalDiscPlan";
import { setPlanLiveLookup } from "./flightPlans";

export const GLOBAL_EVICT_TICK_MS = 30_000;
/** sweep/OpenSky archive floor — 2x the shared MIN_FREE_BYTES (1 GiB) */
export const SWEEP_ARCHIVE_MIN_FREE_BYTES = 2 * 1_073_741_824;
/** Pure gate for sweep-sourced archive writes (fails CLOSED). */
export function sweepArchiveAllowed(freeBytes: number | null): boolean {
  return freeBytes != null && volumeAllowsWrite(freeBytes, SWEEP_ARCHIVE_MIN_FREE_BYTES);
}
export const GLOBAL_RESPONSE_CACHE_MS = 5_000;
/** origins whose fixes THIS module archives (others archive themselves) */
export const ARCHIVED_ORIGINS = new Set(["sweep-type", "sweep-disc", "opensky"]);

export const GLOBAL_HONESTY =
  "Worldwide live aircraft assembled from community ADS-B ground receivers: adsb.lol (ODbL) via a " +
  "type-list lane (all aircraft of each listed ICAO type, worldwide) and a 250nm disc lane over a " +
  "hand-authored traffic mask, plus the discs viewers' maps already fetched and adsb.lol's military/" +
  "LADD/PIA scopes" +
  " — and OpenSky Network (non-commercial) only when credentials are configured. One row per aircraft: " +
  "its FRESHEST fix; seenAt is that fix's time, so a row older than ~2 min is a stale position, not a " +
  "live one. Not every plane on Earth: remote oceans and polar regions beyond receiver range, and " +
  "aircraft with no ICAO type outside recently swept discs, can be missing. src names each row's " +
  "provider (lawful=1 → adsb.lol-only ODbL subset). altFt/gsKt carry ±2 ft / ±1 kt rounding.";

const SOURCE_META: Record<string, { label: string; license: string; commercial_ok: boolean }> = {
  adsblol: { label: "adsb.lol (ADS-B, community)", license: "ODbL 1.0", commercial_ok: true },
  airplaneslive: { label: "airplanes.live (viewport fallback only)", license: "non-commercial", commercial_ok: false },
  adsbfi: { label: "adsb.fi (viewport fallback only)", license: "non-commercial", commercial_ok: false },
  opensky: { label: "OpenSky Network states/all", license: "non-commercial (research/education; written agreement for operational use)", commercial_ok: false },
};

export function parseBBoxQuery(q: Record<string, unknown>): BBox | null {
  const keys = ["lamin", "lamax", "lomin", "lomax"] as const;
  const v = keys.map((k) => parseFloat(String(q?.[k] ?? "")));
  if (!v.every(Number.isFinite)) return null;
  const [lamin, lamax, lomin, lomax] = v;
  const b: BBox = {
    lamin: Math.max(-90, Math.min(lamin, lamax)), lamax: Math.min(90, Math.max(lamin, lamax)),
    lomin: Math.max(-180, Math.min(180, lomin)), lomax: Math.max(-180, Math.min(180, lomax)),
  };
  // a full-width request is the whole world — no filter, no interest
  if (b.lamax - b.lamin >= 170 && (b.lomax - b.lomin >= 359 || b.lomin === -180 && b.lomax === 180)) return null;
  return b;
}

export interface GlobalAircraftHandle {
  snapshot: GlobalSnapshot;
  sweep: SweepHandle;
  opensky: OpenSkyHandle;
  /** the bus subscriber (exposed for tests) */
  onBatch: (b: FixBatch) => void;
  stop: () => void;
}

export function registerGlobalAircraftRoutes(app: Express, deps: {
  env?: NodeJS.ProcessEnv;
  fetchImpl?: typeof fetch;
  sites?: SitePoint[];
  baseDir?: string;
  now?: () => number;
  /** free-space probe (tests inject; default statfs on the archive dir) */
  readFree?: (dir: string) => number | null;
} = {}): GlobalAircraftHandle {
  const env = deps.env ?? process.env;
  const now = deps.now ?? Date.now;
  const sites: SitePoint[] = deps.sites
    ?? ((datacoreSites as { sites?: SitePoint[] }).sites || []).map((s) => ({ lat: s.lat, lon: s.lon }));
  const archiveOn = String(env.GLOBAL_SWEEP_ARCHIVE ?? "1").trim() !== "0";
  const snapshot = new GlobalSnapshot();
  // flight plans fill a request that arrived without the aircraft's
  // callsign/position (a watched plane opened off-viewport) from this snapshot
  setPlanLiveLookup((hex) => {
    const r = snapshot.get(hex);
    return r ? { callsign: r.callsign, lat: r.lat, lon: r.lon, altFt: r.altFt, trk: r.trk, seenAt: r.seenAt } : null;
  });

  const sweep = startGlobalSweep({
    env, fetchImpl: deps.fetchImpl, typeCounts: () => snapshot.typeCounts(),
  });
  const opensky = startOpenSkyGlobal({ env, fetchImpl: deps.fetchImpl });

  // volume guard: statfs at most once a minute
  let freeAt = 0;
  let freeBytes: number | null = null;
  let lowDiskWarned = false;
  const counters = {
    fixes_offered_total: 0,          // sweep/OpenSky fixes that reached the archive decision
    lines_archived_total: 0,         // lines actually written (post adaptive thinning)
    fixes_skipped_low_disk_total: 0, // skipped: free < SWEEP_ARCHIVE_MIN_FREE_BYTES or unreadable
    fixes_skipped_disabled_total: 0, // skipped: GLOBAL_SWEEP_ARCHIVE=0
  };
  const readFree = deps.readFree ?? readFreeBytes;
  // internal failures are COUNTED (coverage.internal_errors) and logged on
  // the 1st and every 100th occurrence — never silently swallowed
  const internalErrors: Record<string, number> = {};
  const noteError = (where: string, e: unknown) => {
    const n = (internalErrors[where] = (internalErrors[where] || 0) + 1);
    if (n === 1 || n % 100 === 0) {
      console.error(`[global-aircraft] ${where} (x${n}):`, e instanceof Error ? e.message : String(e));
    }
  };
  const guardAllows = (): boolean => {
    const t = now();
    if (t - freeAt > 60_000 || freeAt === 0) { freeAt = t; freeBytes = readFree(deps.baseDir || archiveBaseDir()); }
    const ok = sweepArchiveAllowed(freeBytes);
    if (!ok && !lowDiskWarned) {
      lowDiskWarned = true;
      console.error(`[global-aircraft] sweep/OpenSky archiving PAUSED — ${freeBytes == null ? "free space unreadable (fails closed)" : `${freeBytes} bytes free`} < ${SWEEP_ARCHIVE_MIN_FREE_BYTES} floor; live snapshot unaffected, viewport/tracked archiving keeps priority`);
    } else if (ok && lowDiskWarned) {
      lowDiskWarned = false;
      console.error("[global-aircraft] disk headroom above the sweep floor — sweep/OpenSky archiving resumed");
    }
    return ok;
  };

  const onBatch = (b: FixBatch) => {
    snapshot.ingest(b, now());
    if (b.origin === "viewport" && b.disc) {
      try { sweep.scheduler.creditViewportDisc(b.disc, now()); } catch (e) { noteError("viewport-credit", e); }
    }
    if (ARCHIVED_ORIGINS.has(b.origin) && b.aircraft.length) {
      counters.fixes_offered_total += b.aircraft.length;
      if (!archiveOn) counters.fixes_skipped_disabled_total += b.aircraft.length;
      else if (guardAllows()) {
        try { counters.lines_archived_total += archiveAircraft(b.aircraft, sites, deps.baseDir, now()); } catch (e) { noteError("archive", e); }
      } else counters.fixes_skipped_low_disk_total += b.aircraft.length;
    }
  };
  const unsubscribe = subscribeFixes(onBatch);

  const evictTimer = setInterval(() => {
    try { snapshot.evict(now()); } catch (e) { noteError("evict", e); }
  }, GLOBAL_EVICT_TICK_MS);
  unrefTimer(evictTimer);

  const bodyCache = new Map<string, { at: number; body: string }>();

  app.get("/api/data/aircraft/global", (req, res) => {
    complianceAuditTick();
    const t = now();
    const q = req.query as Record<string, unknown>;
    const bbox = parseBBoxQuery(q);
    const lawfulOnly = String(q.lawful ?? "") === "1";
    const sinceN = parseFloat(String(q.since ?? ""));
    const sinceMs = Number.isFinite(sinceN) ? sinceN : null;
    if (bbox) { try { sweep.scheduler.noteInterest(bbox, t); } catch (e) { noteError("interest", e); } }
    const key = `${snapshot.version}|${bbox ? `${bbox.lamin},${bbox.lamax},${bbox.lomin},${bbox.lomax}` : "world"}|${lawfulOnly ? 1 : 0}|${sinceMs ?? ""}`;
    res.set("Cache-Control", "public, max-age=10");
    const hit = bodyCache.get(key);
    if (hit && t - hit.at <= GLOBAL_RESPONSE_CACHE_MS) {
      res.type("application/json");
      return res.send(hit.body);
    }
    snapshot.evict(t);
    const rows = snapshot.rows({ bbox, lawfulOnly, sinceMs });
    const sweepStatus = sweep.status();
    const osStatus = opensky.status();
    const summary = snapshot.summary(t);
    const sources = Object.entries(SOURCE_META).map(([k, m]) => ({
      key: k, ...m,
      active: k === "opensky" ? !!osStatus.enabled
        : k === "adsblol" ? (!!sweepStatus.enabled || (summary.rows_by_source[k] || 0) > 0)
        : (summary.rows_by_source[k] || 0) > 0,
      rows: summary.rows_by_source[k] || 0,
      last_ingest_at: summary.source_last_ingest_at[k] ?? null,
    }));
    const payload = {
      at: t,
      count: rows.length,
      full: sinceMs == null,
      scope: bbox ? "bbox" : "world",
      fields: ROW_FIELDS,
      rows: rows.map(encodeRow),
      coverage: {
        plan: {
          discs: sweepStatus.discs_in_plan, radius_nm: PLAN_RADIUS_NM, mask_boxes: TRAFFIC_MASK.length,
          basis: "hand-authored traffic mask (land with receiver coverage + island airports + North Atlantic / North Pacific / US–Hawaii / Tasman corridors); remote open ocean, >72°N and Antarctica are NOT in the plan",
        },
        discs_in_plan: sweepStatus.discs_in_plan,
        discs_refreshed_last_cycle: sweepStatus.discs_refreshed_last_cycle,
        cycle_window_s: sweepStatus.cycle_window_s,
        oldest_disc_age_s: sweepStatus.oldest_disc_age_s,
        median_disc_age_s: sweepStatus.median_disc_age_s,
        discs_visited_ever: sweepStatus.discs_visited_ever,
        viewport_credited_total: sweepStatus.viewport_credited_total,
        type_lane: sweepStatus.type_lane,
        sweep: {
          enabled: sweepStatus.enabled, steps: sweepStatus.steps, errors: sweepStatus.errors,
          last: sweepStatus.last, last_at: sweepStatus.last_at, governor: sweepStatus.governor,
        },
        opensky: osStatus,
        sources,
        snapshot: { ...summary, evict_after_s: snapshot.evictMs / 1000, cap: snapshot.cap },
        archive: {
          enabled: archiveOn,
          gate: `sweep/OpenSky fixes archived only while ≥ ${SWEEP_ARCHIVE_MIN_FREE_BYTES} bytes (2 GiB) are free — stricter than the shared 1 GiB floor so viewport/tracked archiving keeps priority; fails closed when free space is unreadable`,
          min_free_bytes: SWEEP_ARCHIVE_MIN_FREE_BYTES,
          free_bytes: freeBytes,
          paused: archiveOn && !sweepArchiveAllowed(freeBytes) && freeAt > 0,
          ...counters,
        },
        internal_errors: { ...internalErrors },
        note: "remote oceans and polar regions may be uncovered (ground-receiver ADS-B only); aircraft without an ICAO type appear only where a disc (sweep or viewer) was fetched in the last 10 min",
      },
      honesty: GLOBAL_HONESTY,
    };
    const body = JSON.stringify(payload);
    bodyCache.set(key, { at: t, body });
    if (bodyCache.size > 24) {
      const oldest = Array.from(bodyCache.entries()).sort((a, b) => a[1].at - b[1].at)[0];
      if (oldest) bodyCache.delete(oldest[0]);
    }
    res.type("application/json");
    res.send(body);
  });

  return {
    snapshot, sweep, opensky, onBatch,
    stop: () => {
      unsubscribe();
      clearInterval(evictTimer);
      sweep.stop();
      opensky.stop();
    },
  };
}
