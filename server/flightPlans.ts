// FLIGHT PROGRAM (2026-09-28) — server side of flight plans.
//
// GET /api/data/aircraft/plan/:hex?callsign=UAL1&lat=..&lon=..&alt=..&trk=..
// returns the plan an aircraft is flying (or is predicted to fly) from its
// origin to its destination, for the client's gray "planned" curtain, plus a
// deviation state machine that re-plans from the aircraft's REAL position when
// it leaves the plan. Sources, best first:
//
//   FILED_FAA          FAA SWIM SFDPS filed plan (server/swimSfdps.ts; inert
//                      until the human's SWIM_SFDPS_* credentials exist)
//   HISTORY_PREDICTED  the path this callsign ACTUALLY flew last time between
//                      the same airports, from our own ADS-B archive
//   ROUTE_DB_PREDICTED this callsign's usual origin/destination from the
//                      adsb.lol route DB (VRS standing data, CC0), checked
//                      against the aircraft's position, drawn great-circle
//   NONE               nothing honest to draw — nothing is drawn
//
// HONESTY (repo constitution): FILED vs PREDICTED is on every response
// (`source`, `label`, `honesty`); every altitude not filed/observed carries
// altEstimated=true; a route DB answer that does not fit the aircraft's
// position is REJECTED (NONE/HISTORY), never drawn wrong. The archive stays
// truth: deviation events are appended to <archive>/flight_events/ so a replay
// can show when an aircraft left and rejoined its plan.
//
// MONETIZATION TRIPWIRE: route data comes from adsb.lol (the one ADS-B
// provider lawful under monetization; the route files themselves are the
// Virtual Radar Server standing-data set, CC0 1.0). No non-commercial source
// is added, so server/providerCompliance.ts needs no new entry.
//
// Cost discipline: no timers, no work at import. Everything is on-demand,
// cached (route DB 12h LRU, history 30 min) and bounded (tracked deviations
// capped + evicted after 2h idle).

import fs from "fs";
import path from "path";
import type { Express } from "express";
import { archiveBaseDir, streamJsonlLines, recentTrackCached } from "./datacoreArchive";
import { splitTrips, TRIP_GAP_SEC, type ArchivedFix, type Trip } from "./aircraftTrips";
import { nearestAirport, airportByIdent } from "./airportsIndex";
import { SwimPlanStore, startSfdps, sfdpsStatus, routePointsOf, SFDPS_ENV_PREFIX, type StoredSwimPlan } from "./swimSfdps";
import { swimEnvVarNames, swimProductsEnvStatus } from "./swimConnector";
import {
  crossTrackNm, densifyPlan, estimateVerticalProfile, haversineNm, initialBearingDeg, angleDiffDeg,
  polylineLengthNm, replanFromPosition, splitPlanAt, typicalCruiseFt, cumulativeNm, FT_PER_M,
  type LatLon, type PlanPoint, type ReplanResult,
} from "../shared/flightPlanGeometry";

// ── contract types ──────────────────────────────────────────────────────────
export type PlanSource = "FILED_FAA" | "ROUTE_DB_PREDICTED" | "HISTORY_PREDICTED" | "NONE";
export type DeviationState = "ON_PLAN" | "OFF_PLAN" | "UNKNOWN";
export type PlanEventType = "DEVIATION_START" | "DEVIATION_END" | "REPLANNED" | "PLAN_AMENDED";

export interface AirportRef {
  icao: string; iata: string | null; name: string | null;
  lat: number; lon: number; elevM: number | null;
}
export interface PlanEvent { t: number; type: PlanEventType; detail: string }

export interface FlightPlanResponse {
  hex: string;
  callsign: string | null;
  source: PlanSource;
  label: string;
  origin: AirportRef | null;
  destination: AirportRef | null;
  cruiseAltFt: number | null;
  cruiseAltEstimated: boolean;
  /** CURRENT plan, origin -> destination order. When OFF_PLAN it is the
   *  re-plan, which starts at the aircraft's present position. */
  points: PlanPoint[];
  /** the plan before re-planning (only when re-planned) */
  originalPoints: PlanPoint[] | null;
  deviation: { state: DeviationState; crossTrackNm: number | null; since: number | null };
  events: PlanEvent[];
  fetchedAt: number;
  ageSec: number;
  honesty: string;
  /** additive: true when the lateral path itself is an estimate (great circle
   *  or last-flight path), false only for filed expanded-route geometry */
  pathEstimated: boolean;
}

// ── constants ───────────────────────────────────────────────────────────────
export const ROUTE_DB_TTL_MS = 12 * 3600_000;
export const ROUTE_DB_NEGATIVE_TTL_MS = 6 * 3600_000;
export const ROUTE_DB_CACHE_MAX = 5000;
export const UPSTREAM_TIMEOUT_MS = 5000;
export const ROUTESET_URL = "https://api.adsb.lol/api/0/routeset";
export const routeFileUrl = (cs: string) => `https://vrs-standing-data.adsb.lol/routes/${cs.slice(0, 2)}/${cs}.json`;
/** a route DB leg fits the aircraft when it is within max(150nm, 15% of the
 *  leg) of the leg's great circle. Real long-haul tracks (NAT/PACOTS, wind
 *  routing) sit hundreds of nm off the great circle, so the bound scales;
 *  its job is rejecting a stale/reused callsign, not grading the path. */
export const ROUTE_PLAUSIBLE_MIN_NM = 150;
export const ROUTE_PLAUSIBLE_FRACTION = 0.15;
export const legPlausibleNm = (legNm: number) => Math.max(ROUTE_PLAUSIBLE_MIN_NM, ROUTE_PLAUSIBLE_FRACTION * legNm);

export const HISTORY_LOOKBACK_HOURS = 48;   // same window recentTrackAsync already scans for detail cards
export const HISTORY_CACHE_TTL_MS = 30 * 60_000;
export const HISTORY_CACHE_MAX = 256;
export const HISTORY_MAX_CONCURRENT_SCANS = 2;
export const HISTORY_BUDGET_MS = 2500;
export const HISTORY_MAX_FIXES = 60_000;
export const HISTORY_PATH_MAX_POINTS = 300;

/** Deviation thresholds. Fix spacing in our feed/archive is ~30s to 5 min
 *  (adaptive thinning), so tighter cross-track thresholds are noise: a
 *  procedural offset, a weather deviation around one cell, or a great-circle
 *  vs. airway difference routinely reaches 3-5nm. 8nm for 3 consecutive
 *  en-route fixes = a sustained departure from the plan; < 4nm = back on it
 *  (hysteresis stops flapping). Terminal areas (40nm around the plan's ends)
 *  are excluded: radar vectors and SIDs/STARs are not deviations. */
export const DEVIATION_OFF_NM = 8;
export const DEVIATION_ON_NM = 4;
export const DEVIATION_CONSECUTIVE = 3;
export const DEVIATION_TERMINAL_NM = 40;
export const DEVIATION_MIN_FIX_SPACING_SEC = 30;
export const TRACK_IDLE_MS = 2 * 3600_000;
export const TRACK_MAX = 500;
export const TRACK_EVENTS_MAX = 50;
export const ARCHIVE_RECHECK_MS = 120_000;
/** a re-plan whose rejoin point moved > this beyond the aircraft's own
 *  progress is a new REPLANNED event (smaller drift is the rejoin point
 *  naturally sliding ahead) */
export const REPLAN_EVENT_SHIFT_NM = 50;
export const MAX_PLAN_POINTS = 800;

const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
const round1 = (x: number) => Math.round(x * 10) / 10;

export function sanitizeCallsign(v: unknown): string | null {
  const s = String(v ?? "").trim().toUpperCase();
  return /^[A-Z0-9]{2,8}$/.test(s) ? s : null;
}

// ── route DB: parse + leg selection (pure) ──────────────────────────────────
export interface RouteDbRoute {
  callsign: string;
  /** airports in route order (A-B or A-B-C...) */
  airports: AirportRef[];
  /** upstream's own plausibility flag (routeset only; null = not provided) */
  plausible: boolean | null;
  /** where upstream evaluated `plausible` */
  plausibleAt: LatLon | null;
}

/** the adsb.lol route record shape (route files and routeset entries) */
interface RouteDbJson {
  callsign?: unknown; airport_codes?: unknown; _airports?: unknown; plausible?: unknown;
}
interface RouteDbAirportJson {
  icao?: unknown; iata?: unknown; name?: unknown; lat?: unknown; lon?: unknown;
  alt_meters?: unknown; alt_feet?: unknown;
}
const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));
/** finite number or null — Number(null) is 0, which must not pass for "known" */
const finiteOrNull = (v: unknown): number | null => {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

export function parseRouteDbEntry(input: unknown, evaluatedAt: LatLon | null = null): RouteDbRoute | null {
  if (!input || typeof input !== "object") return null;
  const j = input as RouteDbJson;
  const cs = sanitizeCallsign(j.callsign);
  const codes = typeof j.airport_codes === "string" ? j.airport_codes.trim() : "";
  if (!cs || !codes || /^unknown$/i.test(codes)) return null;
  const aps: AirportRef[] = [];
  const list: unknown[] = Array.isArray(j._airports) ? j._airports : [];
  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    const a = item as RouteDbAirportJson;
    const lat = finiteOrNull(a.lat), lon = finiteOrNull(a.lon);
    const icao = String(a.icao || "").trim().toUpperCase();
    if (lat == null || lon == null || !icao) continue;
    const m = finiteOrNull(a.alt_meters), ft = finiteOrNull(a.alt_feet);
    const elev = m != null ? m : ft != null ? ft / FT_PER_M : null;
    aps.push({
      icao, iata: a.iata ? String(a.iata) : null, name: a.name ? String(a.name) : null,
      lat, lon, elevM: elev == null ? null : Math.round(elev * 10) / 10,
    });
  }
  // order by airport_codes when every code resolves (the "_airports" array
  // has matched it in every observed response, but the codes are canonical)
  const order = codes.split("-").map((c: string) => c.trim().toUpperCase());
  const byIcao = new Map(aps.map((a) => [a.icao, a]));
  const ordered = order.every((c: string) => byIcao.has(c)) ? order.map((c: string) => byIcao.get(c)!) : aps;
  if (ordered.length < 2) return null;
  const pl = j.plausible;
  const plausible = pl == null ? null : !(pl === false || pl === 0 || pl === "0" || pl === "false");
  return { callsign: cs, airports: ordered, plausible, plausibleAt: plausible == null ? null : evaluatedAt };
}

export interface LegChoice {
  origin: AirportRef;
  destination: AirportRef;
  legIndex: number;
  legCount: number;
  crossTrackNm: number | null;
  plausible: boolean;
  positionChecked: boolean;
  reason: string;
}

/** Pick the leg of a (possibly multi-leg) route consistent with the aircraft's
 *  position: nearest leg great circle, with a track-agreement tie-break, then
 *  a plausibility bound. Without a position a single-leg route is accepted
 *  unchecked; a multi-leg route cannot be resolved (null). */
export function selectLeg(route: RouteDbRoute, pos: LatLon | null, trkDeg: number | null = null): LegChoice | null {
  const aps = route.airports;
  const legCount = aps.length - 1;
  if (legCount < 1) return null;
  if (!pos) {
    if (legCount !== 1) return null;
    return {
      origin: aps[0], destination: aps[1], legIndex: 0, legCount, crossTrackNm: null,
      plausible: route.plausible !== false, positionChecked: false,
      reason: "no position supplied — route not checked against the aircraft",
    };
  }
  let best: { i: number; nm: number; score: number } | null = null;
  for (let i = 0; i < legCount; i++) {
    const a = aps[i], b = aps[i + 1];
    const xt = crossTrackNm(pos, [a, b]);
    if (!xt) continue;
    let score = xt.nm;
    if (trkDeg != null && Number.isFinite(trkDeg)) {
      const legBrg = haversineNm(xt.proj, b) > 1
        ? initialBearingDeg(xt.proj, b)
        : (initialBearingDeg(b, a) + 180) % 360; // at the leg's end: its final course
      if (angleDiffDeg(legBrg, trkDeg) > 90) score += 100; // flying the other way
    }
    if (!best || score < best.score) best = { i, nm: xt.nm, score };
  }
  if (!best) return null;
  const a = aps[best.i], b = aps[best.i + 1];
  const legNm = haversineNm(a, b);
  const bound = legPlausibleNm(legNm);
  let plausible = best.nm <= bound;
  let reason = plausible
    ? `aircraft ${best.nm.toFixed(0)} nm from the ${a.icao}→${b.icao} great circle (bound ${bound.toFixed(0)} nm)`
    : `aircraft ${best.nm.toFixed(0)} nm from the ${a.icao}→${b.icao} great circle — beyond the ${bound.toFixed(0)} nm plausibility bound`;
  // upstream veto only applies near where upstream evaluated it
  if (plausible && route.plausible === false && route.plausibleAt && haversineNm(route.plausibleAt, pos) < 100) {
    plausible = false;
    reason = "adsb.lol route DB flags this route implausible for the aircraft's position";
  }
  return {
    origin: a, destination: b, legIndex: best.i, legCount, crossTrackNm: round1(best.nm),
    plausible, positionChecked: true, reason,
  };
}

// ── route DB client (cache + batching + fallback) ───────────────────────────
export interface RouteLookup {
  route: RouteDbRoute | null;
  /** when the answer was fetched (null = never fetched) */
  fetchedAt: number | null;
  stale: boolean;
  error: string | null;
}

interface RouteCacheEntry { at: number; route: RouteDbRoute | null }

export interface RouteDbDeps {
  fetchImpl?: typeof fetch;
  now?: () => number;
  timeoutMs?: number;
  batchWindowMs?: number;
}

/** adsb.lol route lookups: POST /api/0/routeset (batched, position-aware,
 *  carries a plausibility flag) with per-callsign fallback to the static
 *  route files. routeset answered 201-with-empty-body when probed on
 *  2026-09-28 (the per-callsign GET has been redirected to the static files
 *  marked '#deprecated'), so an empty/non-JSON routeset reply parks routeset
 *  for an hour and the static files serve alone. */
export class RouteDbClient {
  private cache = new Map<string, RouteCacheEntry>();
  private inflight = new Map<string, Promise<RouteLookup>>();
  private queue: Array<{ cs: string; pos: LatLon; resolve: (r: RouteDbRoute | null | undefined) => void }> = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private fileSlots = 0;
  private fileWaiters: Array<() => void> = [];
  routesetDownUntil = 0;
  private backoffUntil = 0;
  private failures = 0;
  lastError: string | null = null;
  lastErrorAt: number | null = null;

  constructor(private deps: RouteDbDeps = {}) {}

  private now() { return (this.deps.now ?? Date.now)(); }
  private get fetchImpl(): typeof fetch { return this.deps.fetchImpl ?? fetch; }
  get size(): number { return this.cache.size; }

  private setCache(cs: string, route: RouteDbRoute | null, at: number) {
    this.cache.delete(cs);
    this.cache.set(cs, { at, route });
    while (this.cache.size > ROUTE_DB_CACHE_MAX) {
      const oldest = this.cache.keys().next().value;
      if (oldest === undefined) break;
      this.cache.delete(oldest);
    }
  }

  async get(csRaw: string, pos: LatLon | null = null): Promise<RouteLookup> {
    const cs = sanitizeCallsign(csRaw);
    if (!cs) return { route: null, fetchedAt: null, stale: false, error: "invalid callsign" };
    const now = this.now();
    const hit = this.cache.get(cs);
    if (hit) {
      const ttl = hit.route ? ROUTE_DB_TTL_MS : ROUTE_DB_NEGATIVE_TTL_MS;
      if (now - hit.at < ttl) {
        this.cache.delete(cs); this.cache.set(cs, hit); // LRU touch
        return { route: hit.route, fetchedAt: hit.at, stale: false, error: null };
      }
    }
    if (now < this.backoffUntil) {
      return { route: hit?.route ?? null, fetchedAt: hit?.at ?? null, stale: !!hit, error: `route DB backoff (${this.lastError || "upstream error"})` };
    }
    const running = this.inflight.get(cs);
    if (running) return running;
    const p = this.fetchRoute(cs, pos, hit ?? null).finally(() => this.inflight.delete(cs));
    this.inflight.set(cs, p);
    return p;
  }

  private async fetchRoute(cs: string, pos: LatLon | null, hit: RouteCacheEntry | null): Promise<RouteLookup> {
    try {
      let route: RouteDbRoute | null | undefined;
      if (pos && this.now() >= this.routesetDownUntil) route = await this.enqueueRouteset(cs, pos);
      if (route === undefined) route = await this.fetchRouteFile(cs);
      const at = this.now();
      this.setCache(cs, route, at);
      this.failures = 0;
      return { route, fetchedAt: at, stale: false, error: null };
    } catch (e) {
      this.failures++;
      this.lastError = errMsg(e).slice(0, 200);
      this.lastErrorAt = this.now();
      this.backoffUntil = this.now() + Math.min(15 * 60_000, 30_000 * 2 ** Math.min(5, this.failures - 1));
      return { route: hit?.route ?? null, fetchedAt: hit?.at ?? null, stale: !!hit, error: this.lastError };
    }
  }

  /** Coalesce concurrent lookups into one routeset POST (<= 50 planes).
   *  Resolves the route, null (upstream says unknown), or undefined (routeset
   *  gave no usable answer -> caller falls back to the route file). */
  private enqueueRouteset(cs: string, pos: LatLon): Promise<RouteDbRoute | null | undefined> {
    return new Promise((resolve) => {
      this.queue.push({ cs, pos, resolve });
      if (this.queue.length >= 50) { this.flushNow(); return; }
      if (!this.flushTimer) {
        // not unref'd: a caller is awaiting this flush (25ms)
        this.flushTimer = setTimeout(() => this.flushNow(), this.deps.batchWindowMs ?? 25);
      }
    });
  }

  private flushNow() {
    if (this.flushTimer) { clearTimeout(this.flushTimer); this.flushTimer = null; }
    const batch = this.queue.splice(0, 50);
    if (!batch.length) return;
    if (this.queue.length) this.flushNow();
    void this.postRouteset(batch.map((b) => ({ callsign: b.cs, lat: b.pos.lat, lng: b.pos.lon })))
      .then((answers) => {
        for (const b of batch) {
          if (!answers) { b.resolve(undefined); continue; }
          const raw = answers.find((a) => !!a && typeof a === "object" && sanitizeCallsign((a as RouteDbJson).callsign) === b.cs);
          if (!raw) { b.resolve(undefined); continue; }
          b.resolve(parseRouteDbEntry(raw, b.pos));
        }
      })
      .catch((e: unknown) => {
        this.lastError = `routeset: ${errMsg(e)}`.slice(0, 200);
        for (const b of batch) b.resolve(undefined);
      });
  }

  /** null when routeset gave no usable answer (and parks it for an hour) */
  private async postRouteset(planes: Array<{ callsign: string; lat: number; lng: number }>): Promise<unknown[] | null> {
    try {
      const r = await this.fetchImpl(ROUTESET_URL, {
        method: "POST",
        headers: { ...UA, "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ planes }),
        signal: AbortSignal.timeout(this.deps.timeoutMs ?? UPSTREAM_TIMEOUT_MS),
      });
      const text = r.ok ? await r.text() : "";
      let j: unknown = null;
      try { j = text ? JSON.parse(text) : null; } catch { j = null; }
      if (!Array.isArray(j)) { this.routesetDownUntil = this.now() + 3600_000; return null; }
      return j;
    } catch {
      this.routesetDownUntil = this.now() + 3600_000;
      return null;
    }
  }

  private async fetchRouteFile(cs: string): Promise<RouteDbRoute | null> {
    // politeness: at most 4 concurrent static-file fetches
    if (this.fileSlots >= 4) await new Promise<void>((res) => this.fileWaiters.push(res));
    this.fileSlots++;
    try {
      const r = await this.fetchImpl(routeFileUrl(cs), {
        headers: { ...UA, Accept: "application/json" },
        signal: AbortSignal.timeout(this.deps.timeoutMs ?? UPSTREAM_TIMEOUT_MS),
      });
      if (r.status === 404) return null; // not in the route DB: honest negative
      if (!r.ok) throw new Error(`route file HTTP ${r.status}`);
      return parseRouteDbEntry(await r.json());
    } finally {
      this.fileSlots--;
      this.fileWaiters.shift()?.();
    }
  }
}

// ── history: what this callsign actually flew last time ─────────────────────
export interface HistoryTrip {
  hex: string;
  trip: Trip;
  /** recorded path, thinned; altitudes are the recorded ones (flagged
   *  estimated: they are last flight's, not this flight's) */
  points: PlanPoint[];
}

/** Stream the aircraft archive newest-hour-first collecting fixes whose
 *  callsign is exactly `callsign`, grouped by hex. Bounded by lookback, a fix
 *  cap, and an optional early stop checked every few files. */
/** archive lines that failed to parse during history scans (torn writes) */
let archiveBadLines = 0;
export const historyArchiveBadLines = () => archiveBadLines;

export async function scanCallsignFixes(
  callsign: string, baseDir: string,
  opts: { nowSec: number; lookbackHours?: number; maxFixes?: number; stopWhen?: (byHex: Map<string, ArchivedFix[]>) => boolean },
): Promise<Map<string, ArchivedFix[]>> {
  const dir = path.join(baseDir, "aircraft");
  const byHex = new Map<string, ArchivedFix[]>();
  let files: string[] = [];
  try { files = (await fs.promises.readdir(dir)).sort().reverse(); } catch { return byHex; }
  const fromSec = opts.nowSec - (opts.lookbackHours ?? HISTORY_LOOKBACK_HOURS) * 3600;
  const needle = `"c":"${callsign}"`;
  const cap = opts.maxFixes ?? HISTORY_MAX_FIXES;
  let n = 0, filesRead = 0;
  for (const f of files) {
    const m = f.match(/^(\d{4}-\d{2}-\d{2})-(\d{2})\.jsonl(\.gz)?$/);
    if (!m) continue;
    const h0 = Date.parse(`${m[1]}T${m[2]}:00:00Z`) / 1000;
    if (!Number.isFinite(h0) || h0 + 3600 < fromSec || h0 > opts.nowSec) continue;
    await streamJsonlLines(path.join(dir, f), !!m[3], (line) => {
      if (n >= cap || !line.includes(needle)) return;
      try {
        const r = JSON.parse(line);
        if (r.c !== callsign || typeof r.i !== "string") return;
        const arr = byHex.get(r.i) || [];
        arr.push({ t: r.t, la: r.la, lo: r.lo, al: r.al ?? null, c: r.c, g: !!r.g });
        byHex.set(r.i, arr);
        n++;
      } catch {
        archiveBadLines++; // a torn/partial line (write in progress) — counted on plan-status
      }
    });
    filesRead++;
    if (n >= cap) break;
    if (opts.stopWhen && filesRead % 6 === 0 && opts.stopWhen(byHex)) break;
  }
  for (const arr of Array.from(byHex.values())) arr.sort((a, b) => a.t - b.t);
  return byHex;
}

const fixedWing = (la: number, lo: number) => nearestAirport(la, lo, 6, ["L", "M", "S", "W"]);

/** thin a recorded path to <= maxPoints by along-path distance */
export function thinPath<T extends LatLon>(pts: T[], maxPoints: number): T[] {
  if (pts.length <= maxPoints || maxPoints < 2) return pts.slice();
  const cum = cumulativeNm(pts);
  const step = cum[cum.length - 1] / (maxPoints - 1);
  const out: T[] = [pts[0]];
  let next = step;
  for (let i = 1; i < pts.length - 1; i++) {
    if (cum[i] >= next) { out.push(pts[i]); next = cum[i] + step; }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

/** Completed, airport-to-airport trips from grouped fixes, newest first,
 *  excluding anything that ended in the last 10 minutes (the current flight
 *  may still be landing). Pure given the airport resolver. */
export function completedTrips(
  byHex: Map<string, ArchivedFix[]>, nowSec: number,
  nearest: (la: number, lo: number) => ReturnType<typeof fixedWing> = fixedWing,
): HistoryTrip[] {
  const out: HistoryTrip[] = [];
  for (const [hex, fixes] of Array.from(byHex.entries())) {
    for (const trip of splitTrips(fixes, undefined, undefined, undefined, nearest)) {
      if (trip.quality !== "complete" || !trip.is_flight) continue;
      if (!trip.from_airport || !trip.to_airport || trip.from_airport.id === trip.to_airport.id) continue;
      if (trip.end_t > nowSec - 600) continue;
      const seg = fixes.filter((f) => f.t >= trip.start_t && f.t <= trip.end_t);
      const pts: PlanPoint[] = thinPath(seg.map((f) => ({
        lat: f.la, lon: f.lo,
        // on-ground fixes carry no altitude; the profile estimate fills them
        // (it runs from the field elevations at the ends)
        altM: f.g ? null : (f.al ?? null),
        altEstimated: true,
      })), HISTORY_PATH_MAX_POINTS);
      out.push({ hex, trip, points: pts });
    }
  }
  return out.sort((a, b) => b.trip.end_t - a.trip.end_t);
}

const endpointMatches = (tripEnd: { la: number; lo: number }, apId: string | undefined, want: AirportRef) =>
  apId === want.icao || haversineNm({ lat: tripEnd.la, lon: tripEnd.lo }, want) <= 6;

/** Most recent completed trip matching the wanted endpoints (when given). */
export function pickHistoryTrip(
  trips: HistoryTrip[], want: { origin: AirportRef | null; destination: AirportRef | null },
): HistoryTrip | null {
  for (const h of trips) {
    if (want.origin && !endpointMatches(h.trip.from, h.trip.from_airport?.id, want.origin)) continue;
    if (want.destination && !endpointMatches(h.trip.to, h.trip.to_airport?.id, want.destination)) continue;
    return h;
  }
  return null;
}

export interface HistoryDeps {
  baseDir?: () => string;
  now?: () => number;
  scan?: typeof scanCallsignFixes;
}

/** Per-callsign cache of completed trips (30 min), in-flight dedup, and a
 *  global cap on concurrent archive scans. */
export class HistoryIndex {
  private cache = new Map<string, { at: number; trips: HistoryTrip[] }>();
  private inflight = new Map<string, Promise<HistoryTrip[]>>();
  private running = 0;
  lastScanMs: number | null = null;
  constructor(private deps: HistoryDeps = {}) {}
  get size(): number { return this.cache.size; }

  /** trips for a callsign, or null when not available within budgetMs (the
   *  scan keeps running and fills the cache for the next poll) */
  async trips(callsign: string, budgetMs = HISTORY_BUDGET_MS): Promise<HistoryTrip[] | null> {
    const now = (this.deps.now ?? Date.now)();
    const hit = this.cache.get(callsign);
    if (hit && now - hit.at < HISTORY_CACHE_TTL_MS) return hit.trips;
    let p = this.inflight.get(callsign);
    if (!p) {
      if (this.running >= HISTORY_MAX_CONCURRENT_SCANS) return hit?.trips ?? null;
      p = this.runScan(callsign);
      this.inflight.set(callsign, p);
    }
    let timer: ReturnType<typeof setTimeout> | null = null;
    const timeout = new Promise<null>((res) => { timer = setTimeout(() => res(null), budgetMs); });
    try {
      return await Promise.race([p, timeout]);
    } finally { if (timer) clearTimeout(timer); }
  }

  private async runScan(callsign: string): Promise<HistoryTrip[]> {
    this.running++;
    const t0 = Date.now();
    try {
      const nowMs = (this.deps.now ?? Date.now)();
      const nowSec = Math.floor(nowMs / 1000);
      const byHex = await (this.deps.scan ?? scanCallsignFixes)(callsign, (this.deps.baseDir ?? archiveBaseDir)(), {
        nowSec,
        stopWhen: (m) => completedTrips(m, nowSec).length >= 2,
      });
      const trips = completedTrips(byHex, nowSec).slice(0, 5);
      this.cache.delete(callsign);
      this.cache.set(callsign, { at: nowMs, trips });
      while (this.cache.size > HISTORY_CACHE_MAX) {
        const oldest = this.cache.keys().next().value;
        if (oldest === undefined) break;
        this.cache.delete(oldest);
      }
      return trips;
    } catch {
      return [];
    } finally {
      this.running--;
      this.lastScanMs = Date.now() - t0;
      this.inflight.delete(callsign);
    }
  }
}

// ── flight events (append-only, part of the recorded history) ───────────────
export interface FlightEventRecord {
  t: number; hex: string; cs: string | null; type: PlanEventType; detail: string;
  src: PlanSource; xt: number | null; la: number | null; lo: number | null;
}

export function flightEventsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "flight_events");
}

/** fire-and-forget append to <archive>/flight_events/YYYY-MM-DD.jsonl (UTC
 *  day of the event) */
export function appendFlightEvent(rec: FlightEventRecord, baseDir?: string): Promise<void> {
  const dir = flightEventsDir(baseDir);
  const fp = path.join(dir, `${new Date(rec.t).toISOString().slice(0, 10)}.jsonl`);
  return fs.promises.mkdir(dir, { recursive: true })
    .then(() => fs.promises.appendFile(fp, JSON.stringify(rec) + "\n"))
    .catch((e: unknown) => { console.error("[flightPlans] event append:", errMsg(e)); });
}

// ── deviation state machine ─────────────────────────────────────────────────
export interface PlanFix { t: number; lat: number; lon: number; altM?: number | null; trk?: number | null }

export interface TrackState {
  hex: string;
  callsign: string | null;
  source: PlanSource;
  planKey: string;
  plan: PlanPoint[];
  state: DeviationState;
  offCount: number;
  crossTrackNm: number | null;
  since: number | null;
  lastFixT: number;
  lastReplanT: number;
  lastSeen: number;
  archiveCheckedAt: number;
  replan: ReplanResult | null;
  /** the last LOGGED re-plan (REPLANNED events are emitted on real changes) */
  replanLog: { mode: ReplanResult["mode"]; rejoinAlongNm: number | null; projAlongNm: number } | null;
  /** cumulative along-plan distance per vertex */
  planCum: number[];
  events: PlanEvent[];
}

export class DeviationTracker {
  private tracks = new Map<string, TrackState>();
  constructor(private opts: { max?: number; idleMs?: number; append?: (rec: FlightEventRecord) => void } = {}) {}
  get size(): number { return this.tracks.size; }
  get(hex: string): TrackState | undefined { return this.tracks.get(hex); }

  /** get-or-create the track for hex; a changed plan resets the state machine
   *  (events are kept — they are history) */
  touch(hex: string, callsign: string | null, source: PlanSource, planKey: string, plan: PlanPoint[], now: number): TrackState {
    let t = this.tracks.get(hex);
    if (!t) {
      t = {
        hex, callsign, source, planKey, plan, state: "UNKNOWN", offCount: 0, crossTrackNm: null, since: null,
        lastFixT: 0, lastReplanT: 0, lastSeen: now, archiveCheckedAt: 0, replan: null, replanLog: null, events: [],
        planCum: cumulativeNm(plan),
      };
    } else if (t.planKey !== planKey) {
      Object.assign(t, {
        callsign, source, planKey, plan, state: "UNKNOWN", offCount: 0, crossTrackNm: null, since: null,
        lastFixT: 0, lastReplanT: 0, replan: null, replanLog: null, planCum: cumulativeNm(plan),
      });
    }
    t.lastSeen = now;
    this.tracks.delete(hex);
    this.tracks.set(hex, t); // LRU by last request
    this.evict(now);
    return t;
  }

  forCallsign(cs: string): TrackState[] {
    return Array.from(this.tracks.values()).filter((t) => t.callsign === cs);
  }

  addEvent(t: TrackState, type: PlanEventType, detail: string, at: number, fix?: { lat: number; lon: number } | null): void {
    t.events.push({ t: at, type, detail });
    if (t.events.length > TRACK_EVENTS_MAX) t.events.splice(0, t.events.length - TRACK_EVENTS_MAX);
    try {
      this.opts.append?.({
        t: at, hex: t.hex, cs: t.callsign, type, detail, src: t.source,
        xt: t.crossTrackNm, la: fix?.lat ?? null, lo: fix?.lon ?? null,
      });
    } catch (e) {
      // the in-memory event above is kept; only the archive copy failed
      console.error("[flightPlans] event sink:", errMsg(e));
    }
  }

  /** one state-machine step for a fix (ms timestamp) */
  observe(t: TrackState, fix: PlanFix): void {
    if (!t.plan.length || !Number.isFinite(fix.lat) || !Number.isFinite(fix.lon)) return;
    const xt = crossTrackNm(fix, t.plan);
    if (!xt) return;
    const newer = fix.t > t.lastFixT;
    if (fix.t >= t.lastFixT) t.crossTrackNm = round1(xt.nm); // never let an older fix overwrite the display
    const counted = newer && (t.lastFixT === 0 || fix.t - t.lastFixT >= DEVIATION_MIN_FIX_SPACING_SEC * 1000);
    if (counted) {
      t.lastFixT = fix.t;
      const first = t.plan[0], last = t.plan[t.plan.length - 1];
      const enRoute = haversineNm(fix, first) >= DEVIATION_TERMINAL_NM && haversineNm(fix, last) >= DEVIATION_TERMINAL_NM;
      if (!enRoute) {
        t.offCount = 0; // terminal-area fixes neither start nor end a deviation
      } else if (xt.nm > DEVIATION_OFF_NM) {
        t.offCount++;
        if (t.state !== "OFF_PLAN" && t.offCount >= DEVIATION_CONSECUTIVE) {
          t.state = "OFF_PLAN";
          t.since = fix.t;
          this.addEvent(t, "DEVIATION_START",
            `${xt.nm.toFixed(1)} nm off the plan for ${t.offCount} consecutive en-route fixes (threshold ${DEVIATION_OFF_NM} nm)`, fix.t, fix);
        }
      } else {
        t.offCount = 0;
        if (xt.nm < DEVIATION_ON_NM) {
          if (t.state === "OFF_PLAN") {
            t.state = "ON_PLAN";
            t.since = fix.t;
            t.replan = null; t.replanLog = null;
            this.addEvent(t, "DEVIATION_END", `back within ${DEVIATION_ON_NM} nm of the plan (${xt.nm.toFixed(1)} nm)`, fix.t, fix);
          } else if (t.state === "UNKNOWN") { t.state = "ON_PLAN"; t.since = fix.t; }
        } else if (t.state === "UNKNOWN") { t.state = "ON_PLAN"; t.since = fix.t; } // 4-8 nm: within tolerance
      }
    }
    // re-plan from the newest known position while off plan (every fix, not
    // only counted ones — the gray curtain must start where the aircraft is)
    if (t.state === "OFF_PLAN" && fix.t >= t.lastReplanT) {
      const r = replanFromPosition(t.plan, { lat: fix.lat, lon: fix.lon, altM: fix.altM ?? null }, fix.trk ?? null);
      const rejoinAlong = r.rejoinIndex != null ? (t.planCum[r.rejoinIndex] ?? null) : null;
      const prev = t.replanLog;
      let changed = !prev || prev.mode !== r.mode;
      if (!changed && prev && r.mode === "REJOIN" && prev.rejoinAlongNm != null && rejoinAlong != null) {
        // the rejoin point slides ahead as the aircraft progresses (first
        // vertex >= 20nm ahead); only a shift BEYOND that progress is a real
        // change of re-plan worth an event — otherwise every fix would log one
        const drift = (rejoinAlong - prev.rejoinAlongNm) - (xt.alongNm - prev.projAlongNm);
        changed = Math.abs(drift) > REPLAN_EVENT_SHIFT_NM;
      }
      if (changed) {
        const dest = t.plan[t.plan.length - 1];
        const at = r.rejoinIndex != null ? t.plan[r.rejoinIndex] : null;
        this.addEvent(t, "REPLANNED", r.mode === "REJOIN" && at
          ? `re-planned from present position to rejoin the plan at ${at.name || `${at.lat.toFixed(2)},${at.lon.toFixed(2)}`}`
          : `re-planned from present position direct to ${dest?.name || "destination"}`, fix.t, fix);
        t.replanLog = { mode: r.mode, rejoinAlongNm: rejoinAlong, projAlongNm: xt.alongNm };
      }
      t.replan = r; t.lastReplanT = fix.t;
    }
  }

  evict(now: number): void {
    const idle = this.opts.idleMs ?? TRACK_IDLE_MS;
    for (const [k, t] of Array.from(this.tracks.entries())) {
      if (now - t.lastSeen > idle) this.tracks.delete(k);
      else break; // LRU order
    }
    const max = this.opts.max ?? TRACK_MAX;
    while (this.tracks.size > max) {
      const oldest = this.tracks.keys().next().value;
      if (oldest === undefined) break;
      this.tracks.delete(oldest);
    }
  }
}

/** fixes of the CURRENT flight only (after the last > 45-min gap), newest 200 */
export function currentFlightFixes<T extends { t: number }>(fixes: T[]): T[] {
  let start = 0;
  for (let i = fixes.length - 1; i > 0; i--) {
    if (fixes[i].t - fixes[i - 1].t > TRIP_GAP_SEC) { start = i; break; }
  }
  return fixes.slice(Math.max(start, fixes.length - 200));
}

// ── plan assembly ───────────────────────────────────────────────────────────
export interface PlanRequest {
  hex: string;
  callsign: string | null;
  lat: number | null;
  lon: number | null;
  altFt: number | null;
  trkDeg: number | null;
  /** fix time (ms); default = now */
  fixT: number | null;
}

export interface PlanContext {
  routeDb: RouteDbClient;
  history: HistoryIndex;
  swim: SwimPlanStore;
  tracker: DeviationTracker;
  now: () => number;
  recentFixes: (hex: string) => Promise<Array<{ t: number; la: number; lo: number; al?: number | null }>>;
  historyBudgetMs?: number;
}

interface Candidate {
  source: Exclude<PlanSource, "NONE">;
  label: string;
  origin: AirportRef | null;
  destination: AirportRef | null;
  cruiseAltFt: number | null;
  cruiseAltEstimated: boolean;
  /** raw plan (altM null where unknown) */
  points: PlanPoint[];
  fetchedAt: number;
  pathEstimated: boolean;
  honesty: string;
}

const apPoint = (a: AirportRef): PlanPoint =>
  ({ lat: a.lat, lon: a.lon, altM: a.elevM, altEstimated: a.elevM == null, name: a.iata || a.icao });
const stepFor = (nm: number) => Math.min(100, Math.max(10, nm / 200));

export function airportRefFromIdent(id: string | null): AirportRef | null {
  if (!id) return null;
  const a = airportByIdent(id);
  if (!a) return null;
  return { icao: a.id, iata: null, name: a.n, lat: a.la, lon: a.lo, elevM: a.el };
}

export function filedCandidate(p: StoredSwimPlan, resolve: (id: string | null) => AirportRef | null = airportRefFromIdent): Candidate | null {
  const origin = resolve(p.departure);
  const destination = resolve(p.arrival);
  const cruise = p.cruiseAltFt;
  let raw: PlanPoint[];
  let pathEstimated: boolean;
  const routePoints = routePointsOf(p);
  // SFDPS expandedRoute often carries only the two endpoint fixes (resolved via
  // the NASR gazetteer they sit AT the airports); placed alone they are a
  // great-circle, not a filed path — so require >=1 point that is not an
  // airport endpoint, else stay labelled estimated.
  const interior = routePoints.filter((r) => !(origin && haversineNm(origin, r) <= 2) && !(destination && haversineNm(destination, r) <= 2));
  if (interior.length >= 1) {
    raw = routePoints.map((r) => ({
      lat: r.lat, lon: r.lon, altM: r.altFt != null ? Math.round(r.altFt / FT_PER_M) : null,
      altEstimated: r.altFt == null, ...(r.name ? { name: r.name } : {}),
    }));
    if (origin && haversineNm(origin, raw[0]) > 2) raw.unshift(apPoint(origin));
    if (destination && haversineNm(destination, raw[raw.length - 1]) > 2) raw.push(apPoint(destination));
    pathEstimated = false;
  } else if (origin && destination) {
    raw = [apPoint(origin), apPoint(destination)];
    pathEstimated = true;
  } else {
    return null; // a filed plan we cannot place: fall through to predictions
  }
  if (raw.length < 2) return null;
  const len = polylineLengthNm(raw);
  const pair = `${p.departure || "?"}→${p.arrival || "?"}`;
  return {
    source: "FILED_FAA",
    label: pathEstimated
      ? `FILED — FAA SWIM flight plan ${pair} (route text only; path drawn great-circle, estimated)`
      : `FILED — FAA SWIM flight plan ${pair}`,
    origin, destination,
    cruiseAltFt: cruise ?? typicalCruiseFt(len),
    cruiseAltEstimated: cruise == null,
    points: densifyPlan(raw, stepFor(len)),
    fetchedAt: p.updatedAt,
    pathEstimated,
    honesty: `Route FILED with the FAA (SWIM SFDPS${p.amendments ? `, amended ${p.amendments}×` : ""})` +
      (pathEstimated ? "; the message carried route text only, so the path between the filed airports is a great-circle estimate" : "") +
      "; altitudes flagged altEstimated are a typical-jet profile estimate, not filed.",
  };
}

export function routeDbCandidate(leg: LegChoice, callsign: string, lookup: RouteLookup, now: number): Candidate {
  const raw = [apPoint(leg.origin), apPoint(leg.destination)];
  const len = polylineLengthNm(raw);
  const legNote = leg.legCount > 1 ? `, leg ${leg.legIndex + 1} of ${leg.legCount}` : "";
  const staleNote = lookup.stale && lookup.fetchedAt
    ? ` Route lookup is ${Math.round((now - lookup.fetchedAt) / 60000)} min old (provider unreachable; last good answer served).` : "";
  return {
    source: "ROUTE_DB_PREDICTED",
    label: `PREDICTED — usual route for ${callsign} (adsb.lol route DB${legNote}), great-circle path`,
    origin: leg.origin, destination: leg.destination,
    cruiseAltFt: typicalCruiseFt(len), cruiseAltEstimated: true,
    points: densifyPlan(raw, stepFor(len)),
    fetchedAt: lookup.fetchedAt ?? now,
    pathEstimated: true,
    honesty: `PREDICTED, not filed: ${leg.origin.icao}→${leg.destination.icao} is ${callsign}'s usual route in the ` +
      `community adsb.lol route DB (Virtual Radar Server standing data, CC0)` +
      (leg.positionChecked ? ", checked against the aircraft's position" : ", NOT checked against a position") +
      `; the path is a great circle and altitudes are a typical-jet estimate (250 ft/nm climb, 3° descent).${staleNote}`,
  };
}

export function historyCandidate(h: HistoryTrip, callsign: string, fetchedAt: number,
                                 origin: AirportRef | null, destination: AirportRef | null): Candidate {
  const fromId = h.trip.from_airport?.id || "?";
  const toId = h.trip.to_airport?.id || "?";
  const o: AirportRef | null = origin ?? (h.trip.from_airport ? {
    icao: fromId, iata: null, name: h.trip.from_airport.n, lat: h.trip.from.la, lon: h.trip.from.lo, elevM: h.trip.from_airport.el,
  } : null);
  const d: AirportRef | null = destination ?? (h.trip.to_airport ? {
    icao: toId, iata: null, name: h.trip.to_airport.n, lat: h.trip.to.la, lon: h.trip.to.lo, elevM: h.trip.to_airport.el,
  } : null);
  const pts = h.points.map((p) => ({ ...p }));
  if (o && haversineNm(o, pts[0]) > 2) pts.unshift(apPoint(o));
  if (d && haversineNm(d, pts[pts.length - 1]) > 2) pts.push(apPoint(d));
  if (pts.length) { pts[0].name = pts[0].name || o?.iata || o?.icao || fromId; pts[pts.length - 1].name = pts[pts.length - 1].name || d?.iata || d?.icao || toId; }
  const day = new Date(h.trip.start_t * 1000).toISOString().slice(0, 10);
  const maxFt = h.trip.max_alt_m != null ? Math.round((h.trip.max_alt_m * FT_PER_M) / 100) * 100 : null;
  return {
    source: "HISTORY_PREDICTED",
    label: `PREDICTED — the path ${callsign} actually flew on its last recorded ${fromId}→${toId} trip (${day}, our ADS-B archive)`,
    origin: o, destination: d,
    cruiseAltFt: maxFt, cruiseAltEstimated: true,
    points: pts,
    fetchedAt,
    pathEstimated: true,
    honesty: `PREDICTED, not filed: the path is what ${callsign} (hex ${h.hex}) actually flew ${fromId}→${toId} on ${day} per our own ` +
      "ADS-B archive — gaps in that recording are bridged straight; altitudes are that flight's recorded values, flagged estimated for this one.",
  };
}

function planKeyOf(c: Candidate): string {
  const pts = c.points;
  const sig = (p?: PlanPoint) => (p ? `${p.lat.toFixed(3)},${p.lon.toFixed(3)}` : "");
  return `${c.source}|${c.origin?.icao ?? ""}|${c.destination?.icao ?? ""}|${pts.length}|${sig(pts[0])}|${sig(pts[pts.length >> 1])}|${sig(pts[pts.length - 1])}`;
}

function capPoints(points: PlanPoint[]): PlanPoint[] {
  if (points.length <= MAX_PLAN_POINTS) return points;
  const named = points.filter((p, i) => p.name || i === 0 || i === points.length - 1);
  if (named.length >= MAX_PLAN_POINTS) return thinPath(points, MAX_PLAN_POINTS);
  const thinned = thinPath(points, MAX_PLAN_POINTS - named.length);
  const keep = new Set<PlanPoint>([...named, ...thinned]);
  return points.filter((p) => keep.has(p));
}

function noneResponse(q: PlanRequest, reason: string, now: number): FlightPlanResponse {
  return {
    hex: q.hex, callsign: q.callsign, source: "NONE",
    label: `NONE — no filed or predicted route${q.callsign ? ` for ${q.callsign}` : ""}`,
    origin: null, destination: null, cruiseAltFt: null, cruiseAltEstimated: true,
    points: [], originalPoints: null,
    deviation: { state: "UNKNOWN", crossTrackNm: null, since: null },
    events: [], fetchedAt: now, ageSec: 0,
    honesty: `No route is drawn: ${reason}. Nothing is shown rather than a guess; the real flown path is always the recorded ADS-B track.`,
    pathEstimated: false,
  };
}

/** Build the plan response. Never throws for upstream failures: a provider
 *  hiccup yields the last cached answer (with its age) or source NONE. */
export async function resolveFlightPlan(q: PlanRequest, ctx: PlanContext): Promise<FlightPlanResponse> {
  const now = ctx.now();
  const cs = q.callsign;
  if (!cs) return noneResponse(q, "the aircraft broadcasts no callsign, so there is no route to look up", now);
  const pos: LatLon | null = q.lat != null && q.lon != null ? { lat: q.lat, lon: q.lon } : null;
  const obsAltM = q.altFt != null ? q.altFt / FT_PER_M : null;

  let cand: Candidate | null = null;
  let noneReason = "";
  const filed = ctx.swim.lookup(cs, now);
  if (filed) cand = filedCandidate(filed);
  if (!cand) {
    // route DB and the (bounded, cached) history scan run concurrently
    const histP = ctx.history.trips(cs, ctx.historyBudgetMs ?? HISTORY_BUDGET_MS).catch(() => null);
    const lookup = await ctx.routeDb.get(cs, pos);
    const leg = lookup.route ? selectLeg(lookup.route, pos, q.trkDeg) : null;
    const legOk = !!leg && leg.plausible;
    const trips = await histP;
    const hist = trips ? pickHistoryTrip(trips, legOk ? { origin: leg!.origin, destination: leg!.destination } : { origin: null, destination: null }) : null;
    // standalone history (no plausible route DB leg) must itself fit the aircraft
    const histOk = hist && (legOk || !pos || (() => {
      const xt = crossTrackNm(pos, hist.points);
      return !!xt && xt.nm <= legPlausibleNm(polylineLengthNm(hist.points));
    })());
    if (hist && histOk) cand = historyCandidate(hist, cs, now, legOk ? leg!.origin : null, legOk ? leg!.destination : null);
    else if (legOk) cand = routeDbCandidate(leg!, cs, lookup, now);
    else if (leg && !leg.plausible) noneReason = `${cs}'s route in the adsb.lol route DB (${leg.origin.icao}→${leg.destination.icao}) does not fit the aircraft's position (${leg.reason})`;
    else if (lookup.route && !leg) noneReason = `${cs} has a multi-leg route in the adsb.lol route DB and no position was supplied to choose the leg`;
    else if (lookup.error) noneReason = `the route lookup failed (${lookup.error}) and nothing is cached`;
    else noneReason = `${cs} is not in the adsb.lol route DB and has no recent completed trip in our archive`;
  }
  if (!cand) return noneResponse(q, noneReason, now);

  const originElev = cand.origin?.elevM ?? null;
  const destElev = cand.destination?.elevM ?? null;
  // an ESTIMATED cruise can never sit below an altitude the aircraft is
  // observed at (it would draw the curtain dropping ahead of it); a FILED
  // cruise stays as filed
  const priorCruise = cand.cruiseAltFt ?? typicalCruiseFt(polylineLengthNm(cand.points));
  const cruiseFt = cand.cruiseAltEstimated && q.altFt != null && q.altFt > priorCruise
    ? Math.round(q.altFt / 1000) * 1000 : priorCruise;
  if (cand.cruiseAltEstimated) cand.cruiseAltFt = cruiseFt;
  const base = estimateVerticalProfile(cand.points, originElev, destElev, cruiseFt);

  // ── deviation tracking (plan requested = tracked) ──
  const track = ctx.tracker.touch(q.hex, cs, cand.source, planKeyOf(cand), base, now);
  // A FILED plan whose message carried route TEXT only has a great-circle
  // stand-in for its path (no parsed route fixes). Real flights follow
  // airways, not the great circle (live 2026-09-29: median 61 nm cross-track,
  // 44/59 false OFF_PLAN) — so deviation is not measurable against it and
  // stays UNKNOWN rather than claiming the aircraft left a route we never had.
  const deviationMeasurable = !(cand.source === "FILED_FAA" && cand.pathEstimated);
  if (deviationMeasurable && now - track.archiveCheckedAt > ARCHIVE_RECHECK_MS) {
    track.archiveCheckedAt = now;
    let fixes: Array<{ t: number; la: number; lo: number; al?: number | null }> = [];
    let timer: ReturnType<typeof setTimeout> | null = null;
    try {
      fixes = await Promise.race([
        ctx.recentFixes(q.hex),
        new Promise<never[]>((res) => { timer = setTimeout(() => res([]), 1500); }),
      ]);
    } catch { fixes = []; } finally { if (timer) clearTimeout(timer); }
    for (const f of currentFlightFixes(fixes)) {
      if (f.t * 1000 > track.lastFixT) ctx.tracker.observe(track, { t: f.t * 1000, lat: f.la, lon: f.lo, altM: f.al ?? null, trk: null });
    }
  }
  if (pos && deviationMeasurable) ctx.tracker.observe(track, { t: q.fixT ?? now, lat: pos.lat, lon: pos.lon, altM: obsAltM, trk: q.trkDeg });

  let points = base;
  let originalPoints: PlanPoint[] | null = null;
  if (track.state === "OFF_PLAN" && track.replan && track.replan.points.length) {
    const startAlt = track.replan.points[0].altM;
    points = estimateVerticalProfile(track.replan.points, startAlt ?? obsAltM, destElev, cruiseFt);
    originalPoints = base;
  } else if (pos && obsAltM != null && cand.points.length >= 2) {
    // re-anchor the estimated profile on the aircraft's OBSERVED altitude at
    // its projection onto the plan, so the gray curtain starts where it is
    const { behind, ahead } = splitPlanAt(cand.points, pos);
    const here: PlanPoint = { ...ahead[0], altM: Math.round(obsAltM), altEstimated: false, name: "present position (on plan)" };
    const b = estimateVerticalProfile([...behind.slice(0, -1), here], originElev, obsAltM, cruiseFt);
    const a = estimateVerticalProfile([here, ...ahead.slice(1)], obsAltM, destElev, cruiseFt);
    points = [...b.slice(0, -1), ...a];
  }

  let honesty = cand.honesty;
  if (track.state === "OFF_PLAN") {
    honesty += ` The aircraft is OFF its plan (> ${DEVIATION_OFF_NM} nm cross-track for ${DEVIATION_CONSECUTIVE} consecutive en-route fixes): ` +
      "the path ahead is re-planned from its real position (estimate); the plan it left is in originalPoints.";
    if (cand.source !== "FILED_FAA") {
      honesty += " Against a PREDICTED plan, 'off plan' can mean the prediction was wrong rather than the aircraft.";
    }
  }
  if (!deviationMeasurable) {
    honesty += " Deviation from the filed route is not assessed: only the route text was received, so there is no filed path to measure against.";
  }
  honesty += " The real flown path is always the recorded ADS-B track.";

  return {
    hex: q.hex, callsign: cs,
    source: cand.source, label: cand.label,
    origin: cand.origin, destination: cand.destination,
    cruiseAltFt: cand.cruiseAltFt ?? Math.round(cruiseFt), cruiseAltEstimated: cand.cruiseAltEstimated,
    points: capPoints(points),
    originalPoints: originalPoints ? capPoints(originalPoints) : null,
    deviation: { state: track.state, crossTrackNm: track.crossTrackNm, since: track.since },
    events: track.events.slice(),
    fetchedAt: cand.fetchedAt,
    ageSec: Math.max(0, Math.round((now - cand.fetchedAt) / 1000)),
    honesty,
    pathEstimated: cand.pathEstimated,
  };
}

// ── request parsing ─────────────────────────────────────────────────────────
const num = (v: unknown): number | null => {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
};

/** Validate the query. `alt` is FEET (aviation convention, matches the
 *  global snapshot's altFt); `altM` (meters) is accepted instead. `t` is the
 *  fix time in ms or s. Returns {error} for malformed explicit values. */
export function parsePlanQuery(hexRaw: string, query: Record<string, unknown>): PlanRequest | { error: string } {
  const hex = String(hexRaw || "").trim().toLowerCase();
  if (!/^[0-9a-f]{6}$/.test(hex)) return { error: "icao24 hex required (6 hex characters)" };
  const lat = num(query.lat), lon = num(query.lon);
  if ((lat == null) !== (lon == null)) return { error: "lat and lon must be given together" };
  if (lat != null && (Number.isNaN(lat) || Math.abs(lat) > 90)) return { error: "lat out of range" };
  if (lon != null && (Number.isNaN(lon as number) || Math.abs(lon as number) > 180)) return { error: "lon out of range" };
  let altFt = num(query.alt);
  const altM = num(query.altM);
  if (altFt == null && altM != null && !Number.isNaN(altM)) altFt = altM * FT_PER_M;
  if (altFt != null && (Number.isNaN(altFt) || altFt < -2000 || altFt > 80000)) return { error: "alt (feet) out of range" };
  const trk = num(query.trk);
  if (trk != null && (Number.isNaN(trk) || trk < 0 || trk > 360)) return { error: "trk out of range (0-360)" };
  let fixT = num(query.t);
  if (fixT != null && Number.isNaN(fixT)) return { error: "t must be a timestamp" };
  if (fixT != null && fixT < 1e12) fixT *= 1000;
  return {
    hex, callsign: sanitizeCallsign(query.callsign),
    lat: lat as number | null, lon: lon as number | null,
    altFt: altFt as number | null, trkDeg: trk as number | null, fixT: fixT as number | null,
  };
}

// ── wiring ──────────────────────────────────────────────────────────────────
let defaultCtx: PlanContext | null = null;

export function getFlightPlanContext(): PlanContext {
  if (defaultCtx) return defaultCtx;
  const tracker = new DeviationTracker({ append: (rec) => { void appendFlightEvent(rec); } });
  const swim = new SwimPlanStore({
    // SWIM amendments land on every tracked aircraft flying that callsign
    onAmended: (plan, detail) => {
      if (!plan.callsign) return;
      for (const t of tracker.forCallsign(plan.callsign)) tracker.addEvent(t, "PLAN_AMENDED", detail, Date.now());
    },
  });
  defaultCtx = {
    routeDb: new RouteDbClient(),
    history: new HistoryIndex(),
    swim,
    tracker,
    now: () => Date.now(),
    recentFixes: (hex) => recentTrackCached("aircraft", hex),
  };
  return defaultCtx;
}

export function planStatus(ctx: PlanContext) {
  const s = sfdpsStatus(ctx.swim);
  return {
    swimConfigured: s.configured,
    swimConnected: s.connected,
    solclientAvailable: s.solclientAvailable,
    routeDbCacheSize: ctx.routeDb.size,
    lastRouteDbError: ctx.routeDb.lastError,
    lastRouteDbErrorAt: ctx.routeDb.lastErrorAt,
    routesetParkedUntil: ctx.routeDb.routesetDownUntil > ctx.now() ? ctx.routeDb.routesetDownUntil : null,
    trackedDeviations: ctx.tracker.size,
    historyCacheSize: ctx.history.size,
    historyArchiveBadLines: historyArchiveBadLines(),
    lastHistoryScanMs: ctx.history.lastScanMs,
    // SFDPS consumer detail: transport counters + per-service counters (the
    // one queue mixes Flight FIXM, General Message, Airspace AIXM and Status;
    // only Flight FIXM is parsed, the rest are counted and acked)
    swim: {
      product: s.product,
      messagesReceived: s.messagesReceived, messagesProcessed: s.messagesProcessed,
      processErrors: s.processErrors, droppedOverflow: s.droppedOverflow, queueDepth: s.queueDepth,
      reconnects: s.reconnects, lastMessageAt: s.lastMessageAt, lastError: s.lastError,
      byService: s.counters.byService, flightByType: s.counters.flightByType,
      fullParses: s.counters.fullParses, lightParses: s.counters.lightParses, parseErrors: s.counters.parseErrors,
      planStoreSize: s.storeSize,
      routeShape: s.routeShape,
      envVars: swimEnvVarNames(SFDPS_ENV_PREFIX),
    },
    // env readiness of every SCDS product (names only, never values). Only
    // SFDPS has a consumer in this build; the others are reported, not
    // connected (volume), so ops can verify the Railway env ahead of time.
    swimProducts: swimProductsEnvStatus(),
    generatedAt: ctx.now(),
  };
}

export function registerFlightPlanRoutes(app: Express, ctx: PlanContext = getFlightPlanContext()): void {
  // no-op (no import, no socket) unless every SWIM_SFDPS_* env var is set
  void startSfdps(ctx.swim).catch((e: unknown) => console.error("[flightPlans] SFDPS start:", errMsg(e)));

  app.get("/api/data/aircraft/plan-status", (_req, res) => {
    res.json(planStatus(ctx));
  });

  app.get("/api/data/aircraft/plan/:hex", async (req, res) => {
    const q = parsePlanQuery(String(req.params.hex || ""), req.query as Record<string, unknown>);
    if ("error" in q) return res.status(400).json({ error: q.error });
    try {
      res.json(await resolveFlightPlan(q, ctx));
    } catch (e) {
      // a plan is decoration on a live aircraft: an internal failure degrades
      // to an honest NONE, never a 500 the client must special-case
      console.error("[flightPlans] resolve:", errMsg(e));
      res.json(noneResponse(q, "the plan could not be built (internal error, logged)", ctx.now()));
    }
  });
}
