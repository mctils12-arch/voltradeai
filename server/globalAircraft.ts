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
// SELECTED-AIRCRAFT LANE (2026-11-16, owner: "archive every received
// position"): fixes with origin "fastlane" (the one aircraft a viewer has
// selected, asked of the provider about every 2.5 s) were served live and
// then dropped. They are now archived too, by archiveFastlaneBatch below:
// at the fix's OWN time (never our receive time), each distinct fix once,
// under the shared 1 GiB floor (not the sweep's 2 GiB gate) and bounded per
// hour. AIRCRAFT_FASTLANE_ARCHIVE=0 stops these writes. Counters are in the
// coverage block under archive.fastlane.
//
// RESPONSE CONTRACT (FLIGHT_PROGRAM_BRIEF): { at, count, fields, rows,
// coverage, honesty } — rows are positional arrays in `fields` order.
// Optional query: lamin/lamax/lomin/lomax (all four) filters rows to a
// bbox AND tells the sweep where viewers are looking (interest boost);
// since=<ms> returns only rows with seenAt > since (full:false);
// changed=<ms> (the previous response's `at`) returns only rows that
// changed in the snapshot at/after that SERVER time (full:false) — the
// viewers' delta cursor (browser-side traffic only; upstream unaffected);
// lawful=1 returns only adsb.lol (ODbL) rows. Short shared cache
// (max-age=10) + Express's weak ETag; gzip via the global compression
// middleware (server/index.ts).

import type { Express } from "express";
import datacoreSites from "../datacore/sites/strategic_sites.json";
import { archiveAircraft, archiveAircraftAt, archiveBaseDir, intentArchiveStats, type SitePoint } from "./datacoreArchive";
import { volumeAllowsWrite, readFreeBytes, MIN_FREE_BYTES } from "./globalScopes";
import { createGapFiller, gapFiller, gapFillEnabled } from "./coverageGapFill";
import { publishFixes, subscribeFixes, type FixBatch } from "./aircraftFixBus";
import { createFastLane, ICAO_HEX_LC_RE } from "./aircraftFastLane";
import { GlobalSnapshot, ROW_FIELDS, SNAPSHOT_CAP, encodeRow, toSnapRow, type BBox, type SnapRow } from "./globalSnapshot";
import { startGlobalSweep, unrefTimer, type SweepHandle } from "./globalSweep";
import { createSweepFailover, FAILOVER_ORIGIN, type SweepFailover } from "./sweepFailover";
import { startOpenSkyGlobal, type OpenSkyHandle } from "./openskyGlobal";
import { complianceAuditTick } from "./providerCompliance";
import { PLAN_RADIUS_NM, TRAFFIC_MASK } from "./globalDiscPlan";
import { setPlanLiveLookup } from "./flightPlans";
import { IntentStore, type IntentView } from "./aircraftIntent";
import { foldIntentDays, intentDailyInventory } from "./aircraftIntentDaily";
import { registerHeapGauge, trackMap } from "./heapDiag";
import { idleTtlMap } from "./idleTtl"; // memory audit 2026-11-03 A3

export const GLOBAL_EVICT_TICK_MS = 30_000;
/** sweep/OpenSky archive floor — 2x the shared MIN_FREE_BYTES (1 GiB) */
export const SWEEP_ARCHIVE_MIN_FREE_BYTES = 2 * 1_073_741_824;
/** Pure gate for sweep-sourced archive writes (fails CLOSED). */
export function sweepArchiveAllowed(freeBytes: number | null): boolean {
  return freeBytes != null && volumeAllowsWrite(freeBytes, SWEEP_ARCHIVE_MIN_FREE_BYTES);
}
/** Selected-aircraft lane archive: a fix is written only when it is at least
 *  this much newer than the last one written for that aircraft. The lane
 *  re-serves the same upstream fix on consecutive polls, and its computed
 *  time (provider clock minus position age, 0.1 s steps) can wobble by a
 *  fraction of a second, so "newer" needs a margin — the snapshot's own tie
 *  window (globalSnapshot TIE_MS) is the same 1 s. */
export const FASTLANE_NEW_FIX_MIN_MS = 1_000;
/** A lane fix older than this is not archived. Two reasons: the snapshot
 *  itself refuses a position older than 10 min, and an own-time write must
 *  never land in an hour file that has already been gzipped (hour files are
 *  compressed 2 h after their hour STARTS; a fix under 10 min old always
 *  lands in a file under 70 min old). */
export const FASTLANE_MAX_FIX_AGE_MS = 10 * 60_000;
/** ... and one dated more than this AHEAD of our clock is not archived either:
 *  its time comes from the provider's clock, and a wrong clock must not write
 *  lines into a future hour. */
export const FASTLANE_MAX_FIX_AHEAD_MS = 60_000;
/** Hard bound on lane lines per clock hour (1 per second — the lane's own
 *  upstream ceiling, held here too so a change there cannot grow the disk). */
export const FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR = 3_600;
/** Aircraft whose last written fix time is remembered (oldest dropped first;
 *  a dropped aircraft can at worst write one repeated line). */
export const FASTLANE_DEDUPE_MAX_HEXES = 2_000;
/** Pure gate for selected-aircraft-lane archive writes: the SHARED 1 GiB
 *  floor, failing CLOSED when free space is unreadable (a new optional
 *  writer must never be the one that fills the bot's volume). */
export function fastlaneArchiveAllowed(freeBytes: number | null): boolean {
  return freeBytes != null && freeBytes >= MIN_FREE_BYTES;
}

/** What the worldwide snapshot and the intent store hold for one aircraft
 *  right now (aircraftTrackData.ts reads this for the "live on top" part). */
export interface AircraftLiveState { row: SnapRow; intent: IntentView | null }
export interface AircraftLiveLookup {
  byHex(hex: string, nowMs: number): AircraftLiveState | null;
  /** the freshest live row whose registration field equals `reg` (upper case) */
  byReg(reg: string): SnapRow | null;
}
let liveLookup: AircraftLiveLookup | null = null;
/** The registered worldwide snapshot's lookup; null before the routes are
 *  registered (tests, a boot in progress). */
export function aircraftLiveLookup(): AircraftLiveLookup | null { return liveLookup; }

export const GLOBAL_RESPONSE_CACHE_MS = 5_000;
/** origins whose fixes THIS module archives (others archive themselves) */
export const ARCHIVED_ORIGINS = new Set(["sweep-type", "sweep-disc", "opensky", FAILOVER_ORIGIN]);

export const GLOBAL_HONESTY =
  "Worldwide live aircraft assembled from community ADS-B ground receivers: adsb.lol (ODbL) via a " +
  "type-list lane (all aircraft of each listed ICAO type, worldwide) and a 250nm disc lane over a " +
  "hand-authored traffic mask, plus the discs viewers' maps already fetched and adsb.lol's military/" +
  "LADD/PIA scopes; while adsb.lol is not answering the sweep, or its daily request quota is spent or too " +
  "small for a world sweep (coverage.sweep.governor.quota — adsb.lol is then asked only for the tracked " +
  "tails and Find lookups, and its type lane and scopes are not running), the same discs are asked of " +
  "adsb.fi and airplanes.live (non-commercial) instead, a part of the plan at a time" +
  " — and OpenSky Network (non-commercial) only when credentials are configured. One row per aircraft: " +
  "its FRESHEST fix; seenAt is that fix's time, so a row older than ~2 min is a stale position, not a " +
  "live one. Not every plane on Earth: remote oceans and polar regions beyond receiver range, and " +
  "aircraft with no ICAO type outside recently swept discs, can be missing. src names each row's " +
  "provider (lawful=1 → adsb.lol-only ODbL subset). altFt/gsKt carry ±2 ft / ±1 kt rounding.";

const SOURCE_META: Record<string, { label: string; license: string; commercial_ok: boolean }> = {
  adsblol: { label: "adsb.lol (ADS-B, community)", license: "ODbL 1.0", commercial_ok: true },
  airplaneslive: { label: "airplanes.live (viewport fallback + sweep failover)", license: "non-commercial", commercial_ok: false },
  adsbfi: { label: "adsb.fi (viewport fallback + coverage gap-fill + sweep failover)", license: "non-commercial", commercial_ok: false },
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
  /** the other providers' disc sweep while adsb.lol's is failing */
  sweepFailover: SweepFailover;
  opensky: OpenSkyHandle;
  /** per-aircraft intent/derived fields (AIRCRAFT INTENT A) */
  intent: IntentStore;
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
  registerHeapGauge("globalAircraft.snapshot", () => ({ entries: snapshot.size, cap: SNAPSHOT_CAP })); // memory audit 2026-11-03 A0
  // AIRCRAFT INTENT A (2026-10-14): mode bits / selected altitude / derived
  // state per aircraft, fed by the same bus — no extra provider calls
  const intent = new IntentStore();
  // flight plans fill a request that arrived without the aircraft's
  // callsign/position (a watched plane opened off-viewport) from this snapshot
  setPlanLiveLookup((hex) => {
    const r = snapshot.get(hex);
    return r ? { callsign: r.callsign, lat: r.lat, lon: r.lon, altFt: r.altFt, trk: r.trk, seenAt: r.seenAt } : null;
  });
  const lookup: AircraftLiveLookup = {
    byHex: (hex, nowMs) => {
      const r = snapshot.get(hex);
      return r ? { row: r, intent: intent.view(hex, nowMs) } : null;
    },
    byReg: (reg) => {
      let best: SnapRow | null = null;
      for (const r of snapshot.rows({})) {
        if (r.reg && r.reg.toUpperCase() === reg && (!best || r.seenAt > best.seenAt)) best = r;
      }
      return best;
    },
  };
  liveLookup = lookup;

  const sweep = startGlobalSweep({
    env, fetchImpl: deps.fetchImpl, typeCounts: () => snapshot.typeCounts(),
  });
  // while that sweep is failing, or held off adsb.lol by its daily quota, the plan's discs are asked of the other
  // community providers (sweepFailover.ts; idle otherwise)
  const sweepFailover = createSweepFailover({
    env, fetchImpl: deps.fetchImpl, now, livePositions: () => snapshot.rows({}),
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
  const refreshFree = (): number | null => {
    const t = now();
    if (t - freeAt > 60_000 || freeAt === 0) { freeAt = t; freeBytes = readFree(deps.baseDir || archiveBaseDir()); }
    return freeBytes;
  };
  const guardAllows = (): boolean => {
    const ok = sweepArchiveAllowed(refreshFree());
    if (!ok && !lowDiskWarned) {
      lowDiskWarned = true;
      console.error(`[global-aircraft] sweep/OpenSky archiving PAUSED — ${freeBytes == null ? "free space unreadable (fails closed)" : `${freeBytes} bytes free`} < ${SWEEP_ARCHIVE_MIN_FREE_BYTES} floor; live snapshot unaffected, viewport/tracked archiving keeps priority`);
    } else if (ok && lowDiskWarned) {
      lowDiskWarned = false;
      console.error("[global-aircraft] disk headroom above the sweep floor — sweep/OpenSky archiving resumed");
    }
    return ok;
  };

  // ── selected-aircraft lane archive (see the header) ─────────────────────
  const fastArchiveOn = String(env.AIRCRAFT_FASTLANE_ARCHIVE ?? "1").trim() !== "0";
  const fast = {
    fixes_offered_total: 0,          // lane fixes that reached the archive decision
    lines_archived_total: 0,         // written, each at its fix's own time
    skipped_same_fix_total: 0,       // not newer than the last written fix of that aircraft (an unchanged upstream fix)
    skipped_no_fix_time_total: 0,    // the row carried no position age, so its own time is unknown
    skipped_stale_total: 0,          // fix older than FASTLANE_MAX_FIX_AGE_MS
    skipped_future_total: 0,         // fix dated more than FASTLANE_MAX_FIX_AHEAD_MS ahead of our clock
    skipped_hour_cap_total: 0,       // over FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR
    skipped_low_disk_total: 0,       // free < the shared 1 GiB floor, or unreadable
    skipped_disabled_total: 0,       // AIRCRAFT_FASTLANE_ARCHIVE=0
  };
  const fastLastFixMs = new Map<string, number>();
  let fastHour = -1;
  let fastHourLines = 0;
  const archiveFastlaneBatch = (b: FixBatch): void => {
    const t = now();
    for (const p of b.aircraft) {
      const row = toSnapRow(p, b);
      if (!row) continue;
      fast.fixes_offered_total++;
      if (!fastArchiveOn) { fast.skipped_disabled_total++; continue; }
      // the fix's own time needs the row's position age (or its own clock)
      if (!Number.isFinite(p.seen_at_ms as number) && !Number.isFinite(p.seen_pos as number)) { fast.skipped_no_fix_time_total++; continue; }
      if (t - row.seenAt > FASTLANE_MAX_FIX_AGE_MS) { fast.skipped_stale_total++; continue; }
      if (row.seenAt - t > FASTLANE_MAX_FIX_AHEAD_MS) { fast.skipped_future_total++; continue; }
      const last = fastLastFixMs.get(row.hex);
      if (last !== undefined && row.seenAt < last + FASTLANE_NEW_FIX_MIN_MS) { fast.skipped_same_fix_total++; continue; }
      if (!fastlaneArchiveAllowed(refreshFree())) { fast.skipped_low_disk_total++; continue; }
      const hour = Math.floor(t / 3_600_000);
      if (hour !== fastHour) { fastHour = hour; fastHourLines = 0; }
      if (fastHourLines >= FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR) { fast.skipped_hour_cap_total++; continue; }
      const n = archiveAircraftAt([{ ...p, provider: p.provider ?? b.provider, tSec: row.seenAt / 1000 }], deps.baseDir);
      if (n > 0) {
        fast.lines_archived_total += n;
        fastHourLines += n;
        fastLastFixMs.delete(row.hex); // re-insert: Map order = least recently written first
        fastLastFixMs.set(row.hex, row.seenAt);
        if (fastLastFixMs.size > FASTLANE_DEDUPE_MAX_HEXES) {
          const oldest = fastLastFixMs.keys().next().value;
          if (oldest !== undefined) fastLastFixMs.delete(oldest);
        }
      }
    }
  };

  const onBatch = (b: FixBatch) => {
    snapshot.ingest(b, now());
    try { intent.ingest(b, now()); } catch (e) { noteError("intent", e); }
    if (b.origin === "viewport" && b.disc) {
      try { sweep.scheduler.creditViewportDisc(b.disc, now()); } catch (e) { noteError("viewport-credit", e); }
      // a disc a viewer's map just fetched is not asked again by the failover
      try { sweepFailover.scheduler.creditViewportDisc(b.disc, now()); } catch (e) { noteError("failover-credit", e); }
    }
    if (ARCHIVED_ORIGINS.has(b.origin) && b.aircraft.length) {
      counters.fixes_offered_total += b.aircraft.length;
      if (!archiveOn) counters.fixes_skipped_disabled_total += b.aircraft.length;
      else if (guardAllows()) {
        try { counters.lines_archived_total += archiveAircraft(b.aircraft, sites, deps.baseDir, now()); } catch (e) { noteError("archive", e); }
      } else counters.fixes_skipped_low_disk_total += b.aircraft.length;
    }
    if (b.origin === "fastlane" && b.aircraft.length) {
      try { archiveFastlaneBatch(b); } catch (e) { noteError("fastlane-archive", e); }
    }
  };
  const unsubscribe = subscribeFixes(onBatch);

  // AIRCRAFT INTENT B: fold each complete UTC day of archived intent into
  // its permanent daily rollup (idempotent; streamed; 5 min after boot, then
  // every 30 min — deploys are more frequent than the 6-hourly chain)
  const foldIntent = () => {
    foldIntentDays(deps.baseDir || archiveBaseDir(), now())
      .then((r) => { if (r.daysFolded.length) console.log(`[intent-daily] folded ${r.daysFolded.join(",")} (${r.bytes} B)`); })
      .catch((e) => noteError("intent-daily", e));
  };
  const foldBoot = setTimeout(foldIntent, 5 * 60_000);
  const foldTimer = setInterval(foldIntent, 30 * 60_000);
  unrefTimer(foldBoot);
  unrefTimer(foldTimer);

  const evictTimer = setInterval(() => {
    try { snapshot.evict(now()); } catch (e) { noteError("evict", e); }
    try { intent.evict(now()); } catch (e) { noteError("intent-evict", e); }
  }, GLOBAL_EVICT_TICK_MS);
  unrefTimer(evictTimer);

  // memory audit 2026-11-03 A3: a 12k-row world body is ~1.1 MB and is never
  // served once older than GLOBAL_RESPONSE_CACHE_MS, yet 24 of them sat here
  // until the cap evicted them — entries unused for VOLTRADE_IDLE_TTL_MIN are
  // dropped (idleTtl.ts); the 24-entry cap eviction below is unchanged.
  const bodyCache = idleTtlMap<string, { at: number; body: string }>("globalAircraft.bodyCache");
  trackMap("globalAircraft.bodyCache", bodyCache, 24); // memory audit 2026-11-03 A0

  app.get("/api/data/aircraft/global", (req, res) => {
    complianceAuditTick();
    const t = now();
    const q = req.query as Record<string, unknown>;
    const bbox = parseBBoxQuery(q);
    const lawfulOnly = String(q.lawful ?? "") === "1";
    const sinceN = parseFloat(String(q.since ?? ""));
    const sinceMs = Number.isFinite(sinceN) ? sinceN : null;
    // changed=<server ms>: rows that CHANGED in the snapshot since then (the
    // previous response's `at`) — the client delta cursor (server clock,
    // exact; see SnapRow.ingestAt). A delta never includes evictions: the
    // client applies the same evict_after_s rule and resyncs fully.
    const changedN = parseFloat(String(q.changed ?? ""));
    const changedSinceMs = Number.isFinite(changedN) ? changedN : null;
    if (bbox) {
      try { sweep.scheduler.noteInterest(bbox, t); } catch (e) { noteError("interest", e); }
      try { sweepFailover.scheduler.noteInterest(bbox, t); } catch (e) { noteError("failover-interest", e); }
    }
    const key = `${snapshot.version}|${bbox ? `${bbox.lamin},${bbox.lamax},${bbox.lomin},${bbox.lomax}` : "world"}|${lawfulOnly ? 1 : 0}|${sinceMs ?? ""}|${changedSinceMs ?? ""}`;
    res.set("Cache-Control", "public, max-age=10");
    const hit = bodyCache.get(key);
    if (hit && t - hit.at <= GLOBAL_RESPONSE_CACHE_MS) {
      res.type("application/json");
      return res.send(hit.body);
    }
    snapshot.evict(t);
    const rows = snapshot.rows({ bbox, lawfulOnly, sinceMs, changedSinceMs });
    const sweepStatus = sweep.status();
    const osStatus = opensky.status();
    const summary = snapshot.summary(t);
    const sources = Object.entries(SOURCE_META).map(([k, m]) => ({
      key: k, ...m,
      active: k === "opensky" ? !!osStatus.enabled
        // the sweep counts only while the daily quota lets it ask adsb.lol
        : k === "adsblol" ? ((!!sweepStatus.enabled && !sweepStatus.governor.quota.exhausted && sweepStatus.governor.quota.mode !== "watched_only")
          || (summary.rows_by_source[k] || 0) > 0)
        : (summary.rows_by_source[k] || 0) > 0,
      rows: summary.rows_by_source[k] || 0,
      last_ingest_at: summary.source_last_ingest_at[k] ?? null,
    }));
    const payload = {
      at: t,
      count: rows.length,
      full: sinceMs == null && changedSinceMs == null,
      ...(changedSinceMs != null ? { changed_since: changedSinceMs } : {}),
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
        sweep_failover: sweepFailover.status(t),
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
          // the selected-aircraft lane's own writer (separate floor + bounds)
          fastlane: {
            enabled: fastArchiveOn,
            rule: `each distinct fix of a selected aircraft is written once, at the fix's own time; skipped when it is not at least ${FASTLANE_NEW_FIX_MIN_MS / 1000} s newer than the last one written, older than ${FASTLANE_MAX_FIX_AGE_MS / 60_000} min or dated more than ${FASTLANE_MAX_FIX_AHEAD_MS / 1000} s ahead of our clock, over ${FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR} lines in the clock hour, or while free space is under the shared floor (fails closed when unreadable)`,
            min_free_bytes: MIN_FREE_BYTES,
            paused: fastArchiveOn && !fastlaneArchiveAllowed(freeBytes) && freeAt > 0,
            max_lines_per_hour: FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR,
            ...fast,
          },
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

  // ONE aircraft's live row — the SELECTED/WATCHED aircraft's lookup + fast
  // lane (field bugs 2026-09-30: a watched plane opened off-viewport got a
  // false "not currently broadcasting" card; a selected plane zoomed out
  // read "last position 89s ago"). The snapshot row is served when it is
  // < FAST_FRESH_MS old; otherwise aircraftFastLane asks adsb.lol for this
  // one hex (viewer-lane governed, coalesced, 2 s cache, bounded) and the
  // fix lands in this snapshot for everyone. Same positional wire shape as
  // /global (+ baroRate, the broadcast vertical rate of THIS fix when
  // upstream sent one) so the client decodes it with the same adapter;
  // rows is empty when nothing holds the hex. AIRCRAFT_FAST_LANE=0 → the
  // snapshot only (no upstream requests).
  const fastOn = String(env.AIRCRAFT_FAST_LANE ?? "1").trim() !== "0";
  // coverage gap-fill (coverageGapFill.ts): a hex adsb.lol returns nothing
  // for is asked of adsb.fi. An injected fetch (tests) gets its own filler
  // so no test ever reaches the real network through the shared one.
  const filler = deps.fetchImpl
    ? createGapFiller({ fetchImpl: deps.fetchImpl, now, enabled: gapFillEnabled(env) })
    : gapFiller();
  const fastLane = createFastLane({
    snapshotGet: (hex) => snapshot.get(hex),
    publish: (b) => publishFixes(b),
    fetchImpl: deps.fetchImpl,
    now,
    env, // ADSBLOL_API_KEY rides on the adsb.lol lookup
    fallbackHex: (hex) => filler.fetchHex(hex),
  });
  app.get("/api/data/aircraft/live/:hex", async (req, res) => {
    const hex = String(req.params.hex || "").toLowerCase();
    if (!ICAO_HEX_LC_RE.test(hex)) return res.status(400).json({ error: "icao24 hex required" });
    let r: SnapRow | null | undefined;
    let baroRate: number | null = null;
    let source: string = "snapshot";
    if (fastOn) {
      try {
        const fr = await fastLane.lookup(hex);
        r = fr.row; baroRate = fr.baroRateFpm; source = fr.source;
      } catch (e) {
        noteError("fast-lane", e);
        r = snapshot.get(hex);
      }
    } else {
      r = snapshot.get(hex);
    }
    const t = now();
    const fresh = r && r.seenAt >= t - snapshot.evictMs ? r : null;
    res.set("Cache-Control", "no-store");
    res.json({
      at: t, hex, source,
      fields: [...ROW_FIELDS, "baroRate"],
      rows: fresh ? [[...encodeRow(fresh), baroRate]] : [],
      fast_lane: fastOn ? fastLane.stats() : null,
      // AIRCRAFT INTENT A: fresh values only (stale ones listed with age),
      // per-aircraft capability, units — null when never seen
      intent: intent.view(hex, t),
      honesty: GLOBAL_HONESTY,
    });
  });

  // AIRCRAFT INTENT A: which intent/derived fields the feeds actually send,
  // across airborne aircraft seen in the last 5 min (+ per provider)
  let coverageCache: { at: number; body: ReturnType<IntentStore["coverage"]> & { archive: unknown } } | null = null;
  app.get("/api/data/aircraft/intent-coverage", (_req, res) => {
    const t = now();
    if (!coverageCache || t - coverageCache.at > 15_000) {
      coverageCache = { at: t, body: {
        ...intent.coverage(t),
        // AIRCRAFT INTENT B: delta-encoded archive writes since boot + the
        // permanent daily rollups on disk
        archive: { writer: intentArchiveStats(), daily: intentDailyInventory(deps.baseDir || archiveBaseDir()) },
      } };
    }
    res.set("Cache-Control", "public, max-age=15");
    res.json(coverageCache.body);
  });

  return {
    snapshot, sweep, sweepFailover, opensky, intent, onBatch,
    stop: () => {
      if (liveLookup === lookup) liveLookup = null;
      unsubscribe();
      clearInterval(evictTimer);
      clearTimeout(foldBoot);
      clearInterval(foldTimer);
      sweep.stop();
      sweepFailover.stop();
      opensky.stop();
    },
  };
}
