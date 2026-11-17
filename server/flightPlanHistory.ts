// FLIGHT PLAN VERSIONS (2026-11-16): the serving half.
//
// GET /api/data/aircraft/plan-history/:hex[?callsign=][&geometry=all][&archive=1]
//
// For one aircraft: the route as first received (`original`, drawn gray by the
// app), the route now in force (`current`), every version held in between,
// each amendment classified with the time it took effect and where the
// aircraft was then, and the stretches where the flown ADS-B track left the
// route in force. The flown track itself is the existing track endpoint; this
// answer points at it and does not copy it.
//
// WHAT IS RAW AND WHAT IS DERIVED
//   raw      each version's fields: FAA SWIM SFDPS flight-plan messages as we
//            received them (route text, expanded route points, airports,
//            cruise altitude, message type and time).
//   derived  the drawable path of a route-TEXT plan (our parser, FAA NASR and
//            CIFP data), the classified diff, the position at an amendment
//            (the aircraft's own ADS-B fixes; `interpolated` when it lies
//            between two fixes), and the off-route episodes (the deviation
//            state machine of server/flightPlans.ts replayed over the track,
//            each fix judged against the version in force at its time).
//   never    a cause. An off-route stretch is followed by an amendment or it
//            is not; anything else is "cause not determined".
//
// LADD: no FAA-sourced data is served for an aircraft in the FAA LADD privacy
// program, exactly as /api/data/aircraft/plan withholds its filed plan. Here
// that is everything: versions, amendments, and the off-route episodes, which
// are judged against the FAA route. The answer does not say whether a plan
// exists.

import type { Express } from "express";
import {
  airportRefFromIdent, DeviationTracker, deviationAssessable, filedCandidate, fillFromLive, getFlightPlanContext,
  parsePlanQuery, planLiveLookup, DEVIATION_OFF_NM, DEVIATION_ON_NM, DEVIATION_CONSECUTIVE,
  type PlanContext, type PlanRequest,
} from "./flightPlans";
import { laddRegistry, LADD_WITHHELD_NOTE } from "./laddRegistry";
import {
  readArchivedVersions, FPV_READBACK_MAX_FILES, type ArchivedFlight, type FlightVersions, type PlanVersion,
} from "./flightPlanVersionStore";
import { TRIP_GAP_SEC } from "./aircraftTrips";
import { registerHeapGauge } from "./heapDiag";
import { planDiff, type StoredSwimPlan } from "./swimSfdps";
import {
  buildOffRouteEpisodes, classifyPlanChange, positionAtTime, AMENDMENT_LINK_MS, POSITION_INTERP_MAX_GAP_MS,
  POSITION_NEAREST_MAX_MS, type JudgedFix, type OffRouteEpisode, type PlanChange, type PlanShape,
  type PositionAtTime, type TrackFix,
} from "../shared/flightPlanDiff";
import { haversineNm, FT_PER_M, TERMINAL_AREA_NM, type PlanPoint } from "../shared/flightPlanGeometry";

// ── contract ────────────────────────────────────────────────────────────────
export interface RoutePointOut {
  lat: number; lon: number; name?: string;
  /** only when the message carried an altitude for the point */
  altFt?: number;
  /** the leg FROM this point spans route elements we could not place */
  unresolvedNext?: true;
}

export interface VersionOut {
  /** 1-based position in this answer's `versions` */
  version: number;
  /** the stored sequence number (restarts at 1 when the server restarts) */
  seq: number;
  heldIn: "memory" | "archive";
  /** versions known to be missing between this one and the one before it */
  versionsMissingBefore: number;
  receivedAtMs: number;
  swimMessageMs: number | null;
  messageType: string | null;
  departure: string | null;
  arrival: string | null;
  cruiseAltFt: number | null;
  routeText: string | null;
  routeKind: string | null;
  /** true when the path is a great-circle stand-in between the filed airports */
  pathEstimated: boolean | null;
  /** how the path was obtained, and what could not be placed */
  routeNote: string | null;
  /** named route points in order */
  fixes: string[];
  /** the drawable path; present on `original` and `current`, and on every
   *  version with ?geometry=all */
  points?: RoutePointOut[];
}

export interface AmendmentOut {
  version: number;
  fromVersion: number;
  /** when it took effect, as well as we know it */
  at: number;
  atBasis: "SWIM_MESSAGE_TIME" | "OUR_RECEIVE_TIME";
  atNote: string | null;
  receivedAtMs: number;
  swimMessageMs: number | null;
  messageType: string | null;
  /** the SWIM store's own one-line description of the change */
  detail: string | null;
  kinds: PlanChange["kinds"];
  change: PlanChange;
  /** true when versions between the two compared ones are no longer held */
  spansMissingVersions: boolean;
  /** the aircraft's ADS-B position at `at`; null with positionReason */
  position: PositionAtTime | null;
  positionReason: string | null;
}

/** A version that changed nothing about the route ahead: the FAA restated
 *  the route from where the aircraft was (or sent the full text again), or
 *  only fixes already flown were dropped. Kept out of `amendments` so that
 *  nothing is drawn or flagged for it. */
export interface RestatementOut {
  version: number;
  fromVersion: number;
  at: number;
  messageType: string | null;
  /** fixes dropped from the start that lay behind the aircraft */
  fixesFlown: string[];
  /** fixes dropped from the start with no position known for that moment:
   *  flown or skipped is not determined, and nothing is flagged for them */
  fixesNotJudged: string[];
  note: string;
}

export interface PlanHistoryResponse {
  hex: string;
  callsign: string | null;
  kind: "raw";
  source: "FILED_FAA" | "NONE";
  label: string;
  honesty: string;
  privacy?: "LADD";
  provenance: { plan: string; positions: string; derived: string[] };
  plan: null | {
    planFirstSeenMs: number;
    /** false = version 1 is the plan as we first SAW it; earlier versions may exist */
    originalAsFiled: boolean;
    originalIsFirstSeen: boolean;
    originalNote: string;
    versionsReturned: number;
    /** versions dropped from memory by the per-flight cap (not returned) */
    versionsMissing: number;
    ended: boolean;
    archive: { searched: boolean; filesScanned: number; complete: boolean; versionsFound: number; note: string };
  };
  original: VersionOut | null;
  current: VersionOut | null;
  versions: VersionOut[];
  amendments: AmendmentOut[];
  /** versions that are not amendments (see RestatementOut) */
  restatements: RestatementOut[];
  offRoute: {
    assessed: boolean;
    reason: string | null;
    thresholds: { offNm: number; onNm: number; consecutiveFixes: number; terminalAreaNm: number; amendmentLinkSec: number };
    fixesJudged: number;
    /** fixes of the current flight that could not be judged, and why */
    fixesNotJudged: { beforeFirstVersion: number; pathNotAssessable: number };
    track: { fromMs: number | null; toMs: number | null; fixes: number };
    episodes: OffRouteEpisode[];
    note: string;
  };
  flownTrack: { url: string; note: string };
  generatedAt: number;
}

export const PLAN_HISTORY_HONESTY_NONE =
  "The real flown path is always the recorded ADS-B track.";
const OFF_ROUTE_NOTE =
  "Episodes are the stretches where the deviation state machine calls the aircraft OFF its route " +
  "(the same thresholds as /api/data/aircraft/plan), with each fix judged against the plan version in force at its time. " +
  "Only the fixes we hold for the current flight are judged, so a missing episode is not proof the aircraft stayed on the route. " +
  "A cause is never guessed: an episode is linked to an amendment when one followed, otherwise the cause is not determined.";
const THRESHOLDS = {
  offNm: DEVIATION_OFF_NM, onNm: DEVIATION_ON_NM, consecutiveFixes: DEVIATION_CONSECUTIVE,
  terminalAreaNm: TERMINAL_AREA_NM, amendmentLinkSec: AMENDMENT_LINK_MS / 1000,
};
const PROVENANCE = {
  plan: "FAA SWIM SFDPS flight-plan messages, as received (US airspace flights only)",
  positions: "ADS-B positions from community receivers (adsb.lol), from our own archive and live snapshot",
  derived: [
    "the drawable path of a plan sent as route text (our parser over FAA NASR/CIFP data)",
    "amendment classification and lateral distance",
    `position at an amendment (nearest fix within ${POSITION_NEAREST_MAX_MS / 1000} s, else interpolated and flagged between fixes up to ${POSITION_INTERP_MAX_GAP_MS / 1000} s apart, else none)`,
    "off-route episodes",
  ],
};

/** a SWIM message time further than this from our receive time is not used as
 *  the amendment time (a clock or parse fault, not a real delay) */
export const SWIM_TIME_MAX_LAG_MS = 15 * 60_000;
export const SWIM_TIME_MAX_LEAD_MS = 60_000;
/** how far back ?archive=1 searches when nothing is in memory */
export const ARCHIVE_SEARCH_MS = 12 * 3600_000;
export const ARCHIVE_CACHE_TTL_MS = 5 * 60_000;
export const ARCHIVE_CACHE_MAX = 64;

// ── rendering one version ───────────────────────────────────────────────────
type Cifp = ReturnType<NonNullable<PlanContext["cifp"]>>;

interface Rendered {
  v: PlanVersion;
  atMs: number;
  atBasis: AmendmentOut["atBasis"];
  atNote: string | null;
  /** the plan path (lat/lon), null when the version cannot be placed */
  path: PlanPoint[] | null;
  measurable: boolean;
  shape: PlanShape;
  out: Omit<VersionOut, "version" | "versionsMissingBefore">;
  points: RoutePointOut[];
}

const asPlan = (v: PlanVersion, key: string, callsign: string | null): StoredSwimPlan => ({
  key, gufi: null, callsign, departure: v.departure, arrival: v.arrival, cruiseAltFt: v.cruiseAltFt,
  routeText: v.routeText, route: v.route, routeNames: v.routeNames,
  ...(v.routeKind ? { routeKind: v.routeKind } : {}),
  messageType: v.messageType, flightStatus: null, timestamp: v.swimMessageMs, departureTime: null,
  firstSeen: v.receivedAtMs, updatedAt: v.receivedAtMs, amendments: 0,
});

function effectiveTime(v: PlanVersion): Pick<Rendered, "atMs" | "atBasis" | "atNote"> {
  const m = v.swimMessageMs;
  if (m == null) return { atMs: v.receivedAtMs, atBasis: "OUR_RECEIVE_TIME", atNote: "the message carried no timestamp, so this is when we received it" };
  if (m > v.receivedAtMs + SWIM_TIME_MAX_LEAD_MS || m < v.receivedAtMs - SWIM_TIME_MAX_LAG_MS) {
    return { atMs: v.receivedAtMs, atBasis: "OUR_RECEIVE_TIME", atNote: `the message's own timestamp is ${Math.round((v.receivedAtMs - m) / 1000)} s from when we received it, which is not a plausible delay, so our receive time is used` };
  }
  return { atMs: m, atBasis: "SWIM_MESSAGE_TIME", atNote: null };
}

export function renderVersion(v: PlanVersion, key: string, callsign: string | null, cifp: Cifp): Rendered {
  const cand = filedCandidate(asPlan(v, key, callsign), airportRefFromIdent, cifp);
  const path = cand ? cand.points : null;
  const o = cand?.origin ?? null, d = cand?.destination ?? null;
  const points: RoutePointOut[] = [];
  const fixes: PlanShape["fixes"] = [];
  (path ?? []).forEach((p, i, all) => {
    // the two airport vertices carry the plan's own idents, so a diff never lists them as fixes
    const isOrigin = i === 0 && !!o && haversineNm(o, p) <= 2;
    const isDest = i === all.length - 1 && !!d && haversineNm(d, p) <= 2;
    const name = isOrigin ? (v.departure ?? p.name) : isDest ? (v.arrival ?? p.name) : p.name;
    const filedAlt = !p.altEstimated && p.altM != null && !isOrigin && !isDest ? Math.round(p.altM * FT_PER_M) : null;
    points.push({
      lat: p.lat, lon: p.lon, ...(name ? { name } : {}),
      ...(filedAlt != null ? { altFt: filedAlt } : {}),
      ...(p.unresolvedNext ? { unresolvedNext: true as const } : {}),
    });
    if (name) fixes.push({ lat: p.lat, lon: p.lon, name });
  });
  const presentPosition = cand?.filedRoute ? cand.filedRoute.amendedFromPresentPosition : /\.\/\./.test(v.routeText ?? "");
  const routeNote = !cand
    ? "this version could not be placed: its airports are not in our airport index and no route point was placed"
    : cand.honesty;
  return {
    v, ...effectiveTime(v), path,
    measurable: !!cand && deviationAssessable(cand).measurable,
    shape: {
      departure: v.departure, arrival: v.arrival, cruiseAltFt: v.cruiseAltFt, routeText: v.routeText,
      fixes, path: path ?? [], pathEstimated: cand ? cand.pathEstimated : true,
      amendedFromPresentPosition: presentPosition,
    },
    out: {
      seq: v.n, heldIn: v.heldIn, receivedAtMs: v.receivedAtMs, swimMessageMs: v.swimMessageMs, messageType: v.messageType,
      departure: v.departure, arrival: v.arrival, cruiseAltFt: v.cruiseAltFt, routeText: v.routeText,
      routeKind: cand?.routeKind ?? null, pathEstimated: cand ? cand.pathEstimated : null, routeNote,
      fixes: fixes.map((f) => f.name).filter((n): n is string => !!n && n !== v.departure && n !== v.arrival),
    },
    points,
  };
}

// ── the flown track ─────────────────────────────────────────────────────────
type ArchiveFix = { t: number; la: number; lo: number; al?: number | null };

export function toTrackFixes(fixes: ArchiveFix[], live: TrackFix | null): TrackFix[] {
  const out: TrackFix[] = fixes
    .filter((f) => Number.isFinite(f.t) && Number.isFinite(f.la) && Number.isFinite(f.lo))
    .map((f) => ({ tMs: f.t * 1000, lat: f.la, lon: f.lo, altFt: f.al != null ? Math.round(f.al * FT_PER_M) : null }))
    .sort((a, b) => a.tMs - b.tMs);
  if (live && (!out.length || live.tMs > out[out.length - 1].tMs)) out.push(live);
  return out;
}

/** the fixes after the last gap longer than a trip gap (the current flight) */
export function currentFlight(fixes: TrackFix[]): TrackFix[] {
  let start = 0;
  for (let i = fixes.length - 1; i > 0; i--) {
    if (fixes[i].tMs - fixes[i - 1].tMs > TRIP_GAP_SEC * 1000) { start = i; break; }
  }
  return fixes.slice(start);
}

/** Replay the deviation state machine over the flight, each fix against the
 *  version in force at its time. A private tracker: nothing is archived and
 *  the live tracker behind /api/data/aircraft/plan is not touched. */
export function judgeTrack(hex: string, callsign: string | null, fixes: TrackFix[], rendered: Rendered[]): {
  judged: JudgedFix[]; beforeFirstVersion: number; pathNotAssessable: number;
} {
  const tracker = new DeviationTracker({ max: 4 });
  const judged: JudgedFix[] = [];
  let beforeFirstVersion = 0, pathNotAssessable = 0;
  let idx = -1;
  for (const f of fixes) {
    while (idx + 1 < rendered.length && rendered[idx + 1].atMs <= f.tMs) idx++;
    if (idx < 0) { beforeFirstVersion++; continue; }
    const r = rendered[idx];
    if (!r.path || !r.measurable) { pathNotAssessable++; continue; }
    const track = tracker.touch(hex, callsign, "FILED_FAA", `version:${idx + 1}`, r.path, f.tMs);
    tracker.observe(track, { t: f.tMs, lat: f.lat, lon: f.lon, altM: f.altFt != null ? f.altFt / FT_PER_M : null, trk: null });
    judged.push({
      tMs: f.tMs, lat: f.lat, lon: f.lon, version: idx + 1, state: track.state, offCount: track.offCount,
      crossTrackNm: track.crossTrackNm, distToDestNm: haversineNm(f, r.path[r.path.length - 1]),
    });
  }
  return { judged, beforeFirstVersion, pathNotAssessable };
}

// ── archive read-back (bounded, cached) ─────────────────────────────────────
type ArchiveRead = Awaited<ReturnType<typeof readArchivedVersions>>;
const archiveCache = new Map<string, { at: number; read: ArchiveRead }>();
registerHeapGauge("flightPlanHistory.archiveCache", () => ({ entries: archiveCache.size, cap: ARCHIVE_CACHE_MAX }));
export function _resetPlanHistoryCache(): void { archiveCache.clear(); }

async function cachedArchiveRead(callsign: string, fromMs: number, toMs: number, now: number, baseDir?: string): Promise<ArchiveRead> {
  const key = `${callsign}|${Math.floor(fromMs / 3600_000)}|${baseDir ?? ""}`;
  const hit = archiveCache.get(key);
  if (hit && now - hit.at < ARCHIVE_CACHE_TTL_MS) return hit.read;
  const read = await readArchivedVersions({ callsign, fromMs, toMs, baseDir });
  archiveCache.set(key, { at: now, read });
  while (archiveCache.size > ARCHIVE_CACHE_MAX) {
    const oldest = archiveCache.keys().next().value;
    if (oldest === undefined) break;
    archiveCache.delete(oldest);
  }
  return read;
}

// ── assembly ────────────────────────────────────────────────────────────────
export interface PlanHistoryOptions {
  /** every version carries its path (default: only `original` and `current`) */
  allGeometry?: boolean;
  /** search the on-disk versions when nothing is held in memory */
  searchArchive?: boolean;
  /** archive base dir (tests) */
  baseDir?: string;
  /** when this process started; plans first seen before it may have versions on disk */
  bootMs?: number;
}

const BOOT_MS = Date.now();
const TRACK_URL = (hex: string) => `/api/data/track/aircraft/${hex}`;
const TRACK_NOTE = "the flown ADS-B track (t in seconds, la/lo degrees, al metres); add ?from=&to= (unix seconds) for a specific flight";

function emptyResponse(q: PlanRequest, now: number, reason: string, extra: Partial<PlanHistoryResponse> = {}): PlanHistoryResponse {
  return {
    hex: q.hex, callsign: q.callsign, kind: "raw", source: "NONE",
    label: `NONE — no flight plan versions${q.callsign ? ` for ${q.callsign}` : ""}`,
    honesty: `No plan versions are shown: ${reason}. ${PLAN_HISTORY_HONESTY_NONE}`,
    provenance: PROVENANCE, plan: null, original: null, current: null, versions: [], amendments: [], restatements: [],
    offRoute: {
      assessed: false, reason: "there is no filed route to judge the track against", thresholds: THRESHOLDS,
      fixesJudged: 0, fixesNotJudged: { beforeFirstVersion: 0, pathNotAssessable: 0 },
      track: { fromMs: null, toMs: null, fixes: 0 }, episodes: [], note: OFF_ROUTE_NOTE,
    },
    flownTrack: { url: TRACK_URL(q.hex), note: TRACK_NOTE },
    generatedAt: now,
    ...extra,
  };
}

export async function resolvePlanHistory(qIn: PlanRequest, ctx: PlanContext, opts: PlanHistoryOptions = {}): Promise<PlanHistoryResponse> {
  const now = ctx.now();
  const lookup = planLiveLookup(ctx);
  const q = fillFromLive(qIn, lookup, now);
  const cs = q.callsign;
  // FAA LADD: checked before anything is looked up, and the answer is the same
  // whether or not versions exist (saying which would leak flight activity)
  const ladd = (ctx.isLadd ?? ((h, c) => laddRegistry().isLadd(h, c)))(q.hex, cs);
  if (ladd) {
    const r = emptyResponse(q, now, "withheld");
    return {
      ...r, privacy: "LADD", honesty: `${LADD_WITHHELD_NOTE} ${PLAN_HISTORY_HONESTY_NONE}`,
      label: "WITHHELD — FAA-sourced flight plan data is not shown for this aircraft",
      offRoute: { ...r.offRoute, reason: "not assessed: the route it would be judged against is FAA-sourced and withheld" },
    };
  }
  if (!cs) return emptyResponse(q, now, "the aircraft broadcasts no callsign, so there is no flight plan to look up");
  const store = ctx.versions ?? null;
  if (!store) return emptyResponse(q, now, "flight plan versions are not being recorded on this server");

  const live = ctx.swim.lookup(cs, now);
  const mem: FlightVersions | null = (live ? store.forKey(live.key) : null) ?? store.forCallsign(cs);

  // ── versions: memory, plus the disk tier where it can add something ──
  let versions: PlanVersion[] = mem ? mem.versions.slice() : [];
  let originalAsFiled = mem ? mem.originalAsFiled : false;
  let planFirstSeenMs = mem ? mem.planFirstSeenMs : now;
  let key = mem ? mem.key : `cs:${cs}`;
  const archive = { searched: false, filesScanned: 0, complete: true, versionsFound: 0, note: "" };
  const bootMs = opts.bootMs ?? BOOT_MS;
  const heldBeforeRestart = !!mem && !mem.originalAsFiled && mem.planFirstSeenMs < bootMs;
  if (heldBeforeRestart || (!mem && opts.searchArchive)) {
    const fromMs = mem ? mem.planFirstSeenMs : now - ARCHIVE_SEARCH_MS;
    const toMs = mem ? mem.versions[0].receivedAtMs : now;
    let read: ArchiveRead | null = null;
    try { read = await cachedArchiveRead(cs, fromMs, toMs, now, opts.baseDir); } catch (e) {
      archive.note = `the on-disk versions could not be read (${e instanceof Error ? e.message : String(e)})`;
      archive.complete = false;
    }
    if (read) {
      archive.searched = true; archive.filesScanned = read.filesScanned; archive.complete = read.complete;
      let fl: ArchivedFlight | null = null;
      if (mem) fl = read.flights.find((x) => x.key === mem.key) ?? null;
      else for (const x of read.flights) {
        const lastAt = x.versions[x.versions.length - 1]?.receivedAtMs ?? 0;
        if (!fl || lastAt > (fl.versions[fl.versions.length - 1]?.receivedAtMs ?? 0)) fl = x;
      }
      if (fl) {
        const firstMem = mem ? mem.versions[0].receivedAtMs : Infinity;
        const earlier = fl.versions.filter((x) => x.receivedAtMs < firstMem);
        archive.versionsFound = earlier.length;
        if (earlier.length) {
          // the first version held since the restart is normally the last archived one seen again
          const lastEarlier = earlier[earlier.length - 1];
          const known = lastEarlier.routeText != null || lastEarlier.route.length > 0;
          if (versions.length && known && !planDiff(lastEarlier, versions[0])) versions = versions.slice(1);
          versions = [...earlier, ...versions];
          key = fl.key;
          if (fl.originalAsFiled != null && earlier[0].n === 1) originalAsFiled = fl.originalAsFiled;
          else originalAsFiled = false;
          if (fl.planFirstSeenMs != null) planFirstSeenMs = Math.min(planFirstSeenMs, fl.planFirstSeenMs);
        }
        if (fl.routeUnknown) archive.note = `${fl.routeUnknown} archived version(s) refer to an earlier line that was not in the files read, so their route is not known`;
      }
      if (!archive.note) {
        archive.note = read.complete
          ? `searched ${read.filesScanned} hour file(s) on disk; the disk holds a flight's versions only once it has been amended`
          : `the on-disk search stopped early (${read.filesScanned} of at most ${FPV_READBACK_MAX_FILES} hour files read): older versions may exist`;
      }
    }
  } else {
    archive.note = mem
      ? "not searched: every version of this plan held since we first saw it is in memory"
      : "not searched: add archive=1 to look for a flight that has left memory (the last 12 hours on disk; only amended flights are written)";
  }

  if (!versions.length) {
    const reason = live
      ? `an FAA plan is held for ${cs} but no message carrying its route has been received yet`
      : `no FAA SWIM flight plan is held for ${cs} (flights outside US airspace have none; a plan leaves memory 6 hours after its last message)`;
    const r = emptyResponse(q, now, reason);
    return { ...r, plan: null, honesty: `${r.honesty} ${archive.note ? `On-disk versions: ${archive.note}.` : ""}`.trim() };
  }

  // ── render, order, number ──
  const cifp = ctx.cifp ? ctx.cifp() : null;
  const rendered = versions.map((v) => renderVersion(v, key, cs, cifp));
  const outs: VersionOut[] = rendered.map((r, i) => {
    const prev = i > 0 ? rendered[i - 1].v : null;
    const sameRun = !!prev && prev.heldIn === r.v.heldIn;
    const missing = sameRun ? Math.max(0, r.v.n - prev!.n - 1) : 0;
    const withPoints = opts.allGeometry || i === 0 || i === rendered.length - 1;
    return { version: i + 1, versionsMissingBefore: missing, ...r.out, ...(withPoints ? { points: r.points } : {}) };
  });

  // ── the track ──
  let archiveFixes: ArchiveFix[] = [];
  let trackError: string | null = null;
  try { archiveFixes = await ctx.recentFixes(q.hex); } catch (e) {
    trackError = e instanceof Error ? e.message : String(e);
  }
  const liveRow = lookup ? lookup(q.hex) : null;
  const liveFix: TrackFix | null = q.lat != null && q.lon != null
    ? { tMs: q.fixT ?? now, lat: q.lat, lon: q.lon, altFt: q.altFt }
    : liveRow ? { tMs: liveRow.seenAt, lat: liveRow.lat, lon: liveRow.lon, altFt: liveRow.altFt } : null;
  const allFixes = toTrackFixes(archiveFixes, liveFix);
  const flight = currentFlight(allFixes);

  // ── amendments ──
  const amendments: AmendmentOut[] = [];
  const restatements: RestatementOut[] = [];
  for (let i = 1; i < rendered.length; i++) {
    const r = rendered[i];
    const pos = trackError
      ? { position: null, reason: `the ADS-B track could not be read (${trackError})` }
      : positionAtTime(allFixes, r.atMs);
    // the aircraft's own position decides whether fixes dropped from the
    // start of a present-position route were flown or skipped
    const change = classifyPlanChange(rendered[i - 1].shape, r.shape, {
      at: pos.position ? { lat: pos.position.lat, lon: pos.position.lon } : null,
    });
    if (!change.kinds.length) {
      const head = change.fromPresentPosition;
      const notJudged = head && !head.judged ? change.fixesRemovedFromStart : [];
      restatements.push({
        version: i + 1, fromVersion: i, at: r.atMs, messageType: r.v.messageType,
        fixesFlown: head?.fixesFlown ?? [], fixesNotJudged: notJudged,
        note: notJudged.length
          ? "the route was re-issued from the aircraft's position and no position is known for that moment: whether the fixes dropped from the start were flown or skipped is not determined"
          : head?.fixesFlown.length
            ? "the route was re-issued from the aircraft's position; the fixes dropped from the start lay behind it"
            : change.routeTextChanged
              ? "the same route ahead, stated again (from the aircraft's position, or the full text once more)"
              : "nothing we compare changed in this version",
      });
      continue;
    }
    amendments.push({
      version: i + 1, fromVersion: i, at: r.atMs, atBasis: r.atBasis, atNote: r.atNote,
      receivedAtMs: r.v.receivedAtMs, swimMessageMs: r.v.swimMessageMs, messageType: r.v.messageType,
      detail: r.v.detail, kinds: change.kinds, change,
      spansMissingVersions: outs[i].versionsMissingBefore > 0,
      position: pos.position, positionReason: pos.reason,
    });
  }

  // ── off-route episodes ──
  const { judged, beforeFirstVersion, pathNotAssessable } = judgeTrack(q.hex, cs, flight, rendered);
  const episodes = buildOffRouteEpisodes(judged, amendments.map((a) => ({ version: a.version, atMs: a.at })), now);
  const assessed = judged.length > 0;
  const offReason = assessed ? null
    : trackError ? `the ADS-B track could not be read (${trackError})`
    : !flight.length ? "no ADS-B fixes are held for this aircraft's current flight"
    : pathNotAssessable ? "the filed route could not be placed well enough to judge the track against (route text only, or one or two fixes)"
    : "every fix we hold is from before we first held the plan, so the route in force then is not known";

  const original = outs[0], current = outs[outs.length - 1];
  const versionsMissing = mem ? mem.droppedFromMemory : 0;
  const originalNote = originalAsFiled
    ? "version 1 came from a flight-plan (FH) message that was the first message we received for this flight, with no amendment seen before it. An FH can be sent again later in a flight, so a plan amended before we first saw the flight would look the same."
    : "version 1 is the plan as we FIRST SAW it, not necessarily as filed: we began receiving this flight after its plan existed (or the server restarted), so earlier versions may exist that we never held.";
  const honesty =
    `Flight plan versions for ${cs} from FAA SWIM SFDPS, matched to this aircraft by callsign. ` +
    `${outs.length} version${outs.length === 1 ? "" : "s"} returned` +
    (versionsMissing ? `, ${versionsMissing} more no longer held (per-flight memory cap; see versionsMissingBefore)` : "") + ". " +
    `${originalNote} ` +
    "Each version's fields are what the message carried; a path drawn from route text, the amendment classification, the position at an amendment and the off-route episodes are derived by us and flagged where estimated or interpolated. " +
    `No cause is given for leaving the route. ${PLAN_HISTORY_HONESTY_NONE}`;

  return {
    hex: q.hex, callsign: cs, kind: "raw", source: "FILED_FAA",
    label: `FILED — ${outs.length} version${outs.length === 1 ? "" : "s"} of the FAA SWIM flight plan ${current.departure ?? "?"}→${current.arrival ?? "?"}`,
    honesty, provenance: PROVENANCE,
    plan: {
      planFirstSeenMs, originalAsFiled, originalIsFirstSeen: !originalAsFiled, originalNote,
      versionsReturned: outs.length, versionsMissing,
      ended: mem ? mem.endedAtMs != null : !live,
      archive,
    },
    original, current, versions: outs, amendments, restatements,
    offRoute: {
      assessed, reason: offReason, thresholds: THRESHOLDS, fixesJudged: judged.length,
      fixesNotJudged: { beforeFirstVersion, pathNotAssessable },
      track: { fromMs: flight[0]?.tMs ?? null, toMs: flight[flight.length - 1]?.tMs ?? null, fixes: flight.length },
      episodes, note: OFF_ROUTE_NOTE,
    },
    flownTrack: { url: TRACK_URL(q.hex), note: TRACK_NOTE },
    generatedAt: now,
  };
}

// ── route ───────────────────────────────────────────────────────────────────
export function registerFlightPlanHistoryRoutes(app: Express, ctx: PlanContext = getFlightPlanContext()): void {
  app.get("/api/data/aircraft/plan-history/:hex", async (req, res) => {
    const query = req.query as Record<string, unknown>;
    const q = parsePlanQuery(String(req.params.hex || ""), query);
    if ("error" in q) return res.status(400).json({ error: q.error });
    try {
      res.json(await resolvePlanHistory(q, ctx, {
        allGeometry: String(query.geometry ?? "") === "all",
        searchArchive: String(query.archive ?? "") === "1",
      }));
    } catch (e) {
      // like /plan: an internal failure degrades to an honest NONE, never a 500
      console.error("[flightPlanHistory] resolve:", e instanceof Error ? e.message : String(e));
      res.json(emptyResponse(q, ctx.now(), "the plan history could not be built (internal error, logged)"));
    }
  });
}
