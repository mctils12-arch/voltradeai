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
// Provider politeness: ONE registration request per 30s cycle, registry
// capped at TRACKED_CAP tails. adsb.lol is the monetization-lawful primary
// (ODbL) — same provider, same UA as the viewport path.
//
// HEARD BEFORE POSITIONED (owner report 2026-11-12: "not tracking or picking
// up on adsb ... the planes right away i am looking for n843s"). Measured
// live as N843S left Chicago Midway: 19:48:28Z /v2/reg/N843S -> 0 aircraft;
// 19:49:43Z /v2/hex/ab8c8e -> 1 aircraft, type "mode_s", 850 ft, 144 kt, NO
// lat/lon; 19:50:29Z a real position; 19:50:52Z the registration query
// finally returned it. The registration query omits an aircraft that is
// heard without a position; the hex query returns it. So each cycle also
// asks by hex for the tracked planes the registration batch gave no position
// for (ONE request: /v2/hex accepts a comma-separated list — verified live
// 2026-11-12, /v2/hex/ab8c8e,a42dee -> both aircraft), records what it hears
// as `heard` (altitude + ground speed, never a position), and while a plane
// was heard in the last TRACKED_ACTIVE_WINDOW_MS polls its hex every
// TRACKED_FAST_POLL_MS so its first position lands within seconds.
//
// ADSB.LOL'S DAILY QUOTA (2026-11-17, adsbGovernor.ts). adsb.lol counts
// requests per day (500 without a key); this poller alone sent 2,880. Every
// request here carries the API key when one is configured (ADSBLOL_API_KEY,
// to api.adsb.lol only — not to the globe.adsb.lol trace backfill) and
// reports the response's quota headers to the governor. The tracked planes
// are what a small quota is spent on:
//   - limit under QUOTA_FULL_MIN_DAILY ("watched_only"): the other lanes stop
//     asking adsb.lol and this poller paces itself to TRACKED_QUOTA_SHARE of
//     the limit — one REQUEST per 86,400 s / (limit x 0.4), i.e. 432 s at
//     500/day. A cycle that needed two requests (the registration batch and
//     the hex lookup) is followed by two gaps. No 5 s fast ticks: there is no
//     quota for them. Never faster than TRACKED_POLL_MS.
//   - quota spent: nothing is sent until its reset (the tick still runs,
//     every 30 s, and asks the governor).
//   - a larger limit, or none read yet: the cadence above, unchanged.

import fs from "fs";
import path from "path";
import { archiveAircraft, archiveAircraftAt, archiveBaseDir, type AircraftPoint, type SitePoint } from "./datacoreArchive";
import { mapPointAircraft } from "./aircraftTiling";
import { providerClockMs, publishFixes } from "./aircraftFixBus";
import { TRACKED_QUOTA_SHARE, adsbLolGovernor, adsbLolKeyHeader, parseRetryAfter, type HeaderSource, type QuotaMode, type UpstreamGovernor } from "./adsbGovernor";

// ── TRACE BACKFILL (phase 2, same 2026-08-08 directive: "every time a plane
// turn on we have the data all over the world not just the 250nm") ─────────
// globe.adsb.lol serves each aircraft's COMPLETE current-day track as heard
// by the whole worldwide receiver network (tar1090 trace_full — verified
// live: N843S = 1,252 points vs our 30s poll samples). Every backfill cycle
// merges that global-fidelity trace into our archive, so a tracked plane's
// day is complete from its first broadcast ANYWHERE — not just while inside
// our polls. Read-side t-dedupe (fullTrackAsync) makes overlap harmless.
export const TRACE_BACKFILL_MS = 15 * 60_000;

export function traceUrl(hex: string): string {
  const h = hex.toLowerCase();
  return `https://globe.adsb.lol/data/traces/${h.slice(-2)}/trace_full_${h}.json`;
}

/** Parse a tar1090 trace payload into timestamped fixes. Trace point layout
 *  (tar1090 docs): [offset_s, lat, lon, alt_ft|"ground"|null, gs_kt, track,
 *  flags, vert_rate, aircraft?, source, geom_alt, geom_rate, roll, ...] —
 *  only the fields our archive line carries are read; unknowns stay null. */
export function parseTrace(d: any): Array<AircraftPoint & { tSec: number }> {
  const base = Number(d?.timestamp);
  const trace: any[] = Array.isArray(d?.trace) ? d.trace : [];
  if (!Number.isFinite(base) || !trace.length) return [];
  const hex = String(d?.icao || "").toLowerCase();
  const reg = d?.r ? String(d.r).trim() : null;
  const type = d?.t ? String(d.t).trim() : null;
  const out: Array<AircraftPoint & { tSec: number }> = [];
  for (const p of trace) {
    if (!Array.isArray(p) || p.length < 4) continue;
    const [off, lat, lon, alt] = p;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const ground = alt === "ground";
    out.push({
      tSec: base + Number(off || 0),
      icao24: hex, registration: reg, type,
      callsign: typeof p[8]?.flight === "string" ? p[8].flight.trim() : undefined,
      lat, lon,
      altitude_m: ground || alt == null || !Number.isFinite(alt) ? null : Math.round(alt * 0.3048),
      on_ground: ground,
      velocity_ms: Number.isFinite(p[4]) ? Math.round(p[4] * 0.5144) : null,
      heading: Number.isFinite(p[5]) ? p[5] : null,
      category: null,
    });
  }
  return out;
}

/** Fetch + parse one hex's current-day global trace. Returns [] on any
 *  failure (a missed backfill is just retried next cycle). */
export async function fetchDayTrace(
  hex: string, fetchImpl: typeof fetch, headers: Record<string, string>,
): Promise<Array<AircraftPoint & { tSec: number }>> {
  try {
    const res = await fetchImpl(traceUrl(hex), {
      headers: { ...headers, Referer: "https://globe.adsb.lol/" },
    });
    if (!res.ok) return [];
    return parseTrace(await res.json());
  } catch { return []; }
}

export const TRACKED_CAP = 20;
export const TRACKED_POLL_MS = 30_000;
/** archive EVERY polled fix — the poll cadence IS the sampling cadence */
export const TRACKED_ARCHIVE_INTERVAL_MS = 15_000;
/** Cadence of the hex poll while a tracked plane is ACTIVE (heard, with or
 *  without a position, in the last TRACKED_ACTIVE_WINDOW_MS). 5 s = 0.2
 *  request/s for the WHOLE registry (one batched /v2/hex request, however
 *  many planes are active): a fifth of what the selected-aircraft fast lane
 *  may send (FAST_MAX_RPS = 1, aircraftFastLane.ts) and, added to the world
 *  sweep's 0.4 request/s, still under the governor's 1 request/s ceiling for
 *  all of adsb.lol (adsbGovernor.ts). */
export const TRACKED_FAST_POLL_MS = 5_000;
/** a plane silent this long drops back to the 30 s cadence */
export const TRACKED_ACTIVE_WINDOW_MS = 5 * 60_000;
/** Hard ceiling on fast-tick requests per rolling minute (60 s / 5 s). With
 *  the base cycle's at most 2 requests every 30 s the poller's worst case is
 *  TRACKED_MAX_REQ_PER_MIN, whatever the registry holds (cap 20 planes). */
export const TRACKED_FAST_MAX_PER_MIN = 12;
export const TRACKED_MAX_REQ_PER_MIN = TRACKED_FAST_MAX_PER_MIN + 4;
/** a request that hangs must not stall the self-rescheduling poller */
export const TRACKED_TIMEOUT_MS = 10_000;
/** registry writes are coalesced to one per this interval on fast ticks */
export const TRACKED_SAVE_MIN_MS = 15_000;
/** a `heard` reading younger than this is "being heard now" (the same 2 min
 *  line the map uses for "broadcasting now") */
export const HEARD_FRESH_MS = 120_000;

/** The provider HEARD this aircraft's transponder. `has_position` false =
 *  heard without a decoded position (e.g. source "mode_s" in the first
 *  minute of a flight): altitude and ground speed only. NEVER a position. */
export interface HeardReading {
  /** unix ms of the newest message the provider reported (its clock − `seen`) */
  at: number;
  /** barometric (else geometric) altitude, ft; null on the ground / not sent */
  alt_ft: number | null;
  /** ground speed, knots; null when not sent */
  gs_kt: number | null;
  /** the provider's source type for the row (mode_s, adsb_icao, mlat, …) */
  source: string | null;
  has_position: boolean;
}

export interface TrackedPlane {
  reg: string;                 // normalized tail number — the registry key
  added_at: number;            // unix ms
  hex?: string | null;         // learned from the first live fix
  last_seen?: number | null;   // unix ms of the newest POSITION fix we archived
  last_pos?: { la: number; lo: number; al: number | null } | null;
  /** newest reading of the transponder being heard at all (with or without
   *  a position). last_seen/last_pos keep their meaning: a real position. */
  heard?: HeardReading | null;
  /** unix ms high-water mark of `heard.at` */
  last_heard?: number | null;
  /** unix SEC high-water mark of the day-trace backfill — each cycle only
   *  archives trace points newer than this (small overlap is collapsed by
   *  the read-side t-dedupe) */
  backfilled_to?: number | null;
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

const HEX_RE = /^[0-9a-f]{6}$/;
/** ONE request for several aircraft by ICAO hex (comma-separated list). */
export function hexBatchUrl(hexes: string[]): string {
  return `https://api.adsb.lol/v2/hex/${hexes.map((h) => h.toLowerCase()).filter((h) => HEX_RE.test(h)).join(",")}`;
}

/** one aircraft row exactly as the provider sent it */
export type RawAircraft = Record<string, unknown>;

/** never let a poller timer hold the process open */
const unrefTimer = (t: unknown): void => { (t as { unref?: () => void } | null)?.unref?.(); };
const finiteNum = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export function rawAircraftRows(raw: unknown): RawAircraft[] {
  const ac = (raw as { ac?: unknown } | null)?.ac;
  return Array.isArray(ac) ? ac.filter((a): a is RawAircraft => !!a && typeof a === "object") : [];
}

/** a row carries a decoded position only when BOTH lat and lon are numbers */
export function rawHasPosition(a: RawAircraft): boolean {
  return finiteNum(a.lat) != null && finiteNum(a.lon) != null;
}

/** The provider's own clock for a response (readsb `now`, ms), bounded by
 *  ours so a skewed upstream clock can never date a reading in the future. */
export function upstreamBaseMs(raw: unknown, nowMs: number): number {
  // one unit rule for every publisher: aircraftFixBus.providerClockMs
  return providerClockMs((raw as { now?: unknown } | null)?.now, nowMs) ?? nowMs;
}

/** Pure: what one provider row says about the transponder being heard.
 *  `at` = the provider's clock minus the row's own `seen` (seconds since its
 *  last message). Altitude: barometric, else geometric; "ground" -> null. */
export function heardFromRaw(a: RawAircraft, baseMs: number): HeardReading {
  const seen = finiteNum(a.seen);
  return {
    at: Math.round(baseMs - (seen != null && seen > 0 ? seen * 1000 : 0)),
    alt_ft: a.alt_baro === "ground" ? null : finiteNum(a.alt_baro) ?? finiteNum(a.alt_geom),
    gs_kt: finiteNum(a.gs),
    source: typeof a.type === "string" && a.type ? a.type : null,
    has_position: rawHasPosition(a),
  };
}

/** Pure: is this plane inside the fast-poll window? (hex known, and heard —
 *  with or without a position — within TRACKED_ACTIVE_WINDOW_MS) */
export function isTrackedActive(tp: TrackedPlane, nowMs: number): boolean {
  if (!tp.hex || !HEX_RE.test(tp.hex.toLowerCase())) return false;
  const newest = Math.max(tp.last_heard ?? 0, tp.last_seen ?? 0);
  return newest > 0 && nowMs - newest <= TRACKED_ACTIVE_WINDOW_MS;
}

export function activeHexes(registry: TrackedRegistry, nowMs: number): string[] {
  return registry.planes.filter((tp) => isTrackedActive(tp, nowMs)).map((tp) => String(tp.hex).toLowerCase());
}

export interface PollOutcome {
  /** POSITION fixes only (the archive + fix-bus input) */
  points: AircraftPoint[];
  registry: TrackedRegistry;
  /** tracked planes heard WITHOUT a position this cycle */
  heard: number;
  /** upstream requests this cycle made */
  requests: number;
  /** the hex lookup's failure, when it failed (the registration batch's
   *  fixes are still returned); null = ran clean, not needed, or held back */
  hexNote: string | null;
}

export interface PollHooks {
  /** may the extra hex request be sent now? (governor backoff / own cap) */
  allowHex?: () => boolean;
  /** called when a hex request is sent, and with its outcome (0 = network) */
  noteHexRequest?: () => void;
  noteHexResult?: (status: number, retryAfterSec: number | null) => void;
  /** EVERY adsb.lol response of the cycle (registration batch and hex), as
   *  soon as it arrives and before its outcome is recorded: the daily quota
   *  is read off its headers */
  noteResponse?: (status: number, headers: HeaderSource | null | undefined) => void;
}

/** Pure: fold provider rows into the registry. A row belongs to a tracked
 *  plane when its registration matches, or when its hex is the plane's known
 *  hex and the row names no OTHER registration (identity is never inferred).
 *  A row with a position updates last_seen/last_pos exactly as before; a row
 *  WITHOUT one only records `heard` — last_seen/last_pos are left alone. */
export function foldTrackedRows(
  registry: TrackedRegistry, rows: RawAircraft[], baseMs: number, nowMs: number,
): { registry: TrackedRegistry; positioned: RawAircraft[]; heard: number } {
  const byReg = new Map<string, RawAircraft>();
  const byHex = new Map<string, RawAircraft>();
  const better = (cur: RawAircraft | undefined, next: RawAircraft) => !cur || (!rawHasPosition(cur) && rawHasPosition(next));
  for (const a of rows) {
    const reg = normalizeReg(typeof a.r === "string" ? a.r : null);
    const hex = typeof a.hex === "string" ? a.hex.toLowerCase() : "";
    if (reg && better(byReg.get(reg), a)) byReg.set(reg, a);
    if (HEX_RE.test(hex) && better(byHex.get(hex), a)) byHex.set(hex, a);
  }
  const positioned: RawAircraft[] = [];
  let heard = 0;
  const planes = registry.planes.map((tp) => {
    let hit = byReg.get(tp.reg);
    if (!hit && tp.hex) {
      const h = byHex.get(tp.hex.toLowerCase());
      const named = h ? normalizeReg(typeof h.r === "string" ? h.r : null) : null;
      if (h && (named == null || named === tp.reg)) hit = h;
    }
    if (!hit) return tp;
    const reading = heardFromRaw(hit, baseMs);
    const lastHeard = Math.max(tp.last_heard ?? 0, reading.at);
    const hex = typeof hit.hex === "string" && HEX_RE.test(hit.hex.toLowerCase()) ? hit.hex : tp.hex;
    if (!reading.has_position) {
      heard++;
      // an older reading never replaces a newer one
      const keep = tp.heard && tp.heard.at > reading.at ? tp.heard : reading;
      return { ...tp, hex, heard: keep, last_heard: lastHeard };
    }
    positioned.push(hit);
    return {
      ...tp,
      hex,
      last_seen: nowMs,
      last_pos: { la: hit.lat as number, lo: hit.lon as number, al: reading.alt_ft == null ? null : Math.round(reading.alt_ft * 0.3048) },
      heard: reading,
      last_heard: lastHeard,
    };
  });
  return { registry: { planes }, positioned, heard };
}

async function fetchHexRows(
  hexes: string[], fetchImpl: typeof fetch, headers: Record<string, string>, hooks: PollHooks,
): Promise<unknown> {
  hooks.noteHexRequest?.();
  let status = 0;
  try {
    const res = await fetchImpl(hexBatchUrl(hexes), { headers, signal: AbortSignal.timeout(TRACKED_TIMEOUT_MS) });
    status = res.status;
    hooks.noteResponse?.(res.status, res.headers);
    hooks.noteHexResult?.(res.status, res.ok ? null : parseRetryAfter(res.headers?.get?.("retry-after") ?? null));
    if (!res.ok) throw new Error(`adsb.lol hex batch ${res.status}`);
    return await res.json();
  } catch (e) {
    if (status === 0) hooks.noteHexResult?.(0, null); // network error / timeout
    throw e;
  }
}

/** One poll cycle, pure-injectable: fetch the registration batch, map
 *  through the SAME normalizer as the viewport path, and fold updates into
 *  the registry; then ONE hex request for the known-hex planes the batch
 *  gave no position for (heard-without-position lives only there). Planes
 *  with no signal are untouched (last_seen keeps aging — honest). A failed
 *  hex lookup never loses the registration batch's fixes. */
export async function pollTrackedOnce(
  registry: TrackedRegistry,
  fetchImpl: typeof fetch,
  headers: Record<string, string>,
  nowMs: number,
  hooks: PollHooks = {},
): Promise<PollOutcome> {
  if (!registry.planes.length) return { points: [], registry, heard: 0, requests: 0, hexNote: null };
  const res = await fetchImpl(batchUrl(registry.planes.map((p) => p.reg)), {
    headers, signal: AbortSignal.timeout(TRACKED_TIMEOUT_MS),
  });
  hooks.noteResponse?.(res.status, res.headers);
  if (!res.ok) throw new Error(`adsb.lol reg batch ${res.status}`);
  const raw = await res.json();
  const regRows = rawAircraftRows(raw);
  const first = foldTrackedRows(registry, regRows, upstreamBaseMs(raw, nowMs), nowMs);
  // tracked poller uses the adsb.lol reg-batch endpoint (batchUrl) — tag it.
  // Every row the batch returned WITH a position, as before.
  const points = mapPointAircraft({ ac: regRows.filter(rawHasPosition) }, "ac", "adsblol") as AircraftPoint[];
  let out = first.registry;
  let heard = first.heard;
  let requests = 1;
  let hexNote: string | null = null;
  const gotPosition = new Set(first.positioned.map((a) => String(a.hex || "").toLowerCase()));
  const missing = out.planes
    .map((tp) => String(tp.hex || "").toLowerCase())
    .filter((h) => HEX_RE.test(h) && !gotPosition.has(h));
  if (missing.length) {
    if (!hooks.allowHex || hooks.allowHex()) {
      try {
        requests++;
        const hexRaw = await fetchHexRows(missing, fetchImpl, headers, hooks);
        const second = foldTrackedRows(out, rawAircraftRows(hexRaw), upstreamBaseMs(hexRaw, nowMs), nowMs);
        out = second.registry;
        heard += second.heard;
        points.push(...(mapPointAircraft({ ac: second.positioned }, "ac", "adsblol") as AircraftPoint[]));
      } catch (e) {
        hexNote = errText(e);
      }
    }
  }
  return { points, registry: out, heard, requests, hexNote };
}

/** A FAST tick: ONE hex request for the active planes only. Throws on an
 *  upstream failure (the caller treats it as a missed sample). */
export async function pollHexesOnce(
  registry: TrackedRegistry,
  hexes: string[],
  fetchImpl: typeof fetch,
  headers: Record<string, string>,
  nowMs: number,
  hooks: PollHooks = {},
): Promise<PollOutcome> {
  if (!hexes.length) return { points: [], registry, heard: 0, requests: 0, hexNote: null };
  const raw = await fetchHexRows(hexes, fetchImpl, headers, hooks);
  const f = foldTrackedRows(registry, rawAircraftRows(raw), upstreamBaseMs(raw, nowMs), nowMs);
  return {
    points: mapPointAircraft({ ac: f.positioned }, "ac", "adsblol") as AircraftPoint[],
    registry: f.registry, heard: f.heard, requests: 1, hexNote: null,
  };
}

/** "skipped-quota" = adsb.lol's daily quota is spent, nothing was sent;
 *  "quota-wait" = paced by the quota and the next request is not due yet */
export type CycleKind = "base" | "fast" | "skipped-backoff" | "skipped-cap" | "idle" | "skipped-quota" | "quota-wait";

/** How the poller is paced right now, and why (GET /api/data/aircraft/tracked). */
export interface TrackedPacing {
  /** adsb.lol's quota mode as the governor reads it */
  quota_mode: QuotaMode;
  /** true while the daily quota sets the interval */
  paced: boolean;
  /** true while the quota is spent: no request is sent */
  blocked: boolean;
  /** seconds between two cycles as they run now; null while blocked */
  effective_poll_s: number | null;
  /** the quota's gap between two REQUESTS (s); null when not paced */
  request_gap_s: number | null;
  /** requests the last cycle made (a paced cycle waits that many gaps) */
  last_cycle_requests: number;
  reason: string;
}

export interface TrackedCycle {
  /** run whichever cycle is due at `nowMs` */
  run(registry: TrackedRegistry, nowMs: number): Promise<PollOutcome & { kind: CycleKind }>;
  /** ms until the next tick: fast while any plane is active, else 30 s; the
   *  time to the next request the daily quota allows when it sets the pace */
  nextDelayMs(registry: TrackedRegistry, nowMs: number): number;
  /** the interval in force and why */
  pacing(registry: TrackedRegistry, nowMs: number): TrackedPacing;
  stats(): Record<string, number>;
}

/** The cadence decision, clock-injectable so tests never sleep.
 *  - every TRACKED_POLL_MS: the BASE cycle (registration batch + hex lookup);
 *  - between base cycles, while a plane is active: a FAST tick (hex only).
 *  Hex requests are accounted to the shared adsb.lol governor on its
 *  foreground lane (recorded, counted toward the total ceiling so the world
 *  sweep yields — the same lane the selected-aircraft fast lane uses) and
 *  are NOT sent while the governor is backing off after an adsb.lol
 *  429/5xx/network failure: then only the base registration request runs,
 *  as it always has. */
export function createTrackedCycle(deps: {
  fetchImpl: typeof fetch;
  headers: Record<string, string>;
  governor?: UpstreamGovernor;
  /** where ADSBLOL_API_KEY is read (default process.env) */
  env?: NodeJS.ProcessEnv;
}): TrackedCycle {
  const gov = deps.governor ?? adsbLolGovernor;
  // both request kinds go to api.adsb.lol: the key rides on them, and only them
  const headers = { ...deps.headers, ...adsbLolKeyHeader(deps.env) };
  let lastBaseAt: number | null = null;
  /** the last cycle that sent anything: when, and how many requests */
  let lastSentAt: number | null = null;
  let lastSentRequests = 1;
  const fastSent: number[] = [];
  const counts = { base_cycles: 0, fast_cycles: 0, skipped_backoff: 0, skipped_cap: 0, requests: 0, hex_requests: 0, hex_failures: 0, skipped_quota: 0, quota_paced_cycles: 0 };
  /** the quota's gap between two requests (0 = the quota does not set the pace) */
  const quotaGapMs = (nowMs: number): number => {
    const g = gov.trackedMinGapMs(nowMs);
    return g > 0 ? Math.max(TRACKED_POLL_MS, g) : 0;
  };
  /** when the quota allows the next cycle: one gap per request the last one made */
  const quotaDueAt = (gapMs: number): number => (lastSentAt == null ? 0 : lastSentAt + gapMs * lastSentRequests);
  const sent = (nowMs: number, requests: number) => {
    if (requests > 0) { lastSentAt = nowMs; lastSentRequests = requests; }
  };
  const hooksAt = (nowMs: number): PollHooks => ({
    noteResponse: (status, h) => { gov.noteQuota(h, nowMs, status); },
    allowHex: () => {
      // the registration batch may just have found the quota spent
      if (!gov.admit("tracked", nowMs)) { counts.skipped_quota++; return false; }
      if (gov.backoffMs(nowMs) <= 0) return true;
      counts.skipped_backoff++;
      return false;
    },
    noteHexRequest: () => { counts.hex_requests++; gov.noteRequest("fg", nowMs); },
    noteHexResult: (status, retryAfterSec) => {
      if (status < 200 || status >= 400) counts.hex_failures++;
      gov.noteResult("fg", status, nowMs, retryAfterSec);
    },
  });
  return {
    async run(registry, nowMs) {
      const none = { points: [], registry, heard: 0, requests: 0, hexNote: null };
      // adsb.lol's daily quota is spent: nothing is sent until its reset
      if (!gov.admit("tracked", nowMs)) { counts.skipped_quota++; return { ...none, kind: "skipped-quota" }; }
      const gap = quotaGapMs(nowMs);
      if (gap > 0) {
        // the quota sets the pace: one full cycle per allowed slot, no fast ticks
        if (nowMs < quotaDueAt(gap)) return { ...none, kind: "quota-wait" };
        lastBaseAt = nowMs;
        counts.base_cycles++;
        counts.quota_paced_cycles++;
        let requests = 1; // a cycle that throws sent its registration batch
        try {
          const out = await pollTrackedOnce(registry, deps.fetchImpl, headers, nowMs, hooksAt(nowMs));
          requests = out.requests;
          counts.requests += out.requests;
          return { ...out, kind: "base" };
        } finally {
          sent(nowMs, requests);
        }
      }
      const baseDue = lastBaseAt == null || nowMs - lastBaseAt >= TRACKED_POLL_MS - 100;
      if (baseDue) {
        lastBaseAt = nowMs;
        counts.base_cycles++;
        sent(nowMs, 1);
        const out = await pollTrackedOnce(registry, deps.fetchImpl, headers, nowMs, hooksAt(nowMs));
        counts.requests += out.requests;
        sent(nowMs, out.requests);
        return { ...out, kind: "base" };
      }
      const hexes = activeHexes(registry, nowMs);
      if (!hexes.length) return { ...none, kind: "idle" };
      while (fastSent.length && nowMs - fastSent[0] >= 60_000) fastSent.shift();
      if (fastSent.length >= TRACKED_FAST_MAX_PER_MIN) { counts.skipped_cap++; return { ...none, kind: "skipped-cap" }; }
      if (gov.backoffMs(nowMs) > 0) { counts.skipped_backoff++; return { ...none, kind: "skipped-backoff" }; }
      fastSent.push(nowMs);
      counts.fast_cycles++;
      sent(nowMs, 1);
      const out = await pollHexesOnce(registry, hexes, deps.fetchImpl, headers, nowMs, hooksAt(nowMs));
      counts.requests += out.requests;
      return { ...out, kind: "fast" };
    },
    nextDelayMs(registry, nowMs) {
      const until = gov.quotaExhaustedUntil(nowMs);
      // spent: look again at the reset, or in 30 s, whichever is sooner
      if (until != null) return Math.max(1_000, Math.min(TRACKED_POLL_MS, until - nowMs));
      const gap = quotaGapMs(nowMs);
      if (gap > 0) return Math.max(TRACKED_POLL_MS, quotaDueAt(gap) - nowMs);
      return activeHexes(registry, nowMs).length ? TRACKED_FAST_POLL_MS : TRACKED_POLL_MS;
    },
    pacing(registry, nowMs) {
      const quota_mode = gov.quotaMode(nowMs);
      const q = gov.stats(nowMs).quota;
      const until = gov.quotaExhaustedUntil(nowMs);
      if (until != null) {
        return {
          quota_mode, paced: false, blocked: true, effective_poll_s: null, request_gap_s: null, last_cycle_requests: lastSentRequests,
          reason: `adsb.lol daily quota spent until ${new Date(until).toISOString()}: no request is sent until then`,
        };
      }
      const gap = quotaGapMs(nowMs);
      if (gap > 0) {
        const share = Math.round(TRACKED_QUOTA_SHARE * 100);
        return {
          quota_mode, paced: true, blocked: false,
          effective_poll_s: Math.round((gap * lastSentRequests) / 1000), request_gap_s: Math.round(gap / 1000), last_cycle_requests: lastSentRequests,
          reason: `adsb.lol daily quota ${q.limit}/day: the tracked planes get ${share}% of it, one request per ${Math.round(gap / 1000)} s; ` +
            `the last cycle made ${lastSentRequests} request${lastSentRequests === 1 ? "" : "s"}, so the next follows after ${Math.round((gap * lastSentRequests) / 1000)} s`,
        };
      }
      const fast = activeHexes(registry, nowMs).length > 0;
      return {
        quota_mode, paced: false, blocked: false,
        effective_poll_s: (fast ? TRACKED_FAST_POLL_MS : TRACKED_POLL_MS) / 1000, request_gap_s: null, last_cycle_requests: lastSentRequests,
        reason: quota_mode === "unknown"
          ? "adsb.lol has not stated its daily quota yet: the standard cadence"
          : `adsb.lol daily quota ${q.limit ?? "unknown"}/day (mode ${quota_mode}): the standard cadence, not paced by the quota`,
      };
    },
    stats() { return { ...counts }; },
  };
}

/** Pure: land one cycle's readings on the CURRENT registry by tail number.
 *  The registry can change while a cycle is in flight (a tail added or
 *  removed through the API, the backfill's high-water mark) — only the
 *  fields a poll owns are taken from the polled copy. */
export function mergePolled(current: TrackedRegistry, polled: TrackedRegistry): TrackedRegistry {
  const byReg = new Map(polled.planes.map((p) => [p.reg, p] as const));
  return {
    planes: current.planes.map((cur) => {
      const p = byReg.get(cur.reg);
      if (!p) return cur;
      return { ...cur, hex: p.hex, last_seen: p.last_seen, last_pos: p.last_pos, heard: p.heard, last_heard: p.last_heard };
    }),
  };
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
  /** adsb.lol governor the hex requests are accounted to (tests inject) */
  governor?: UpstreamGovernor;
  /** where ADSBLOL_API_KEY is read (default process.env) */
  env?: NodeJS.ProcessEnv;
}): {
  getRegistry: () => TrackedRegistry;
  /** cadence counters (base/fast cycles, skips, requests) */
  cycleStats: () => Record<string, number>;
  /** the interval in force and why (the daily quota may set it) */
  pacing: () => TrackedPacing;
  mutate: (fn: (r: TrackedRegistry) => TrackedRegistry | null) => boolean;
  stop: () => void;
} {
  let registry = loadRegistry(deps.baseDir);
  saveRegistry(registry, deps.baseDir); // persist the seed on first boot
  let stopped = false;
  const cycle = createTrackedCycle({ fetchImpl: deps.fetchImpl, headers: deps.headers, governor: deps.governor, env: deps.env });
  let timer: ReturnType<typeof setTimeout> | null = null;
  let dirty = false;
  let lastSaveAt = 0;
  const tick = async () => {
    if (stopped) return;
    const started = Date.now();
    try {
      if (registry.planes.length) {
        const out = await cycle.run(registry, started);
        registry = mergePolled(registry, out.registry);
        // WATCHED-PLANE LOOKUP (2026-09-30): tracked fixes also feed the
        // worldwide snapshot, so GET /api/data/aircraft/live/:hex (and the
        // plan's missing-callsign fill) always hold a watched plane's newest
        // fix even where no sweep disc or viewer covered it. Origin "tracked"
        // is not in the global archiver's origins — these archive below.
        if (out.points.length) {
          publishFixes({ provider: "adsblol", origin: "tracked", aircraft: out.points, fetchedAt: started });
        }
        const archived = out.points.length
          ? archiveAircraft(out.points, deps.sites, deps.baseDir, started, TRACKED_ARCHIVE_INTERVAL_MS)
          : 0;
        if (out.points.length || out.heard) dirty = true;
        // a base cycle writes as before (a plane is always FIRST heard by
        // one); fast ticks coalesce their writes — the API serves the
        // in-memory registry, so nothing a viewer sees waits on the file
        if (dirty && (out.kind === "base" || started - lastSaveAt >= TRACKED_SAVE_MIN_MS)) {
          saveRegistry(registry, deps.baseDir);
          dirty = false;
          lastSaveAt = started;
        }
        if (out.hexNote) console.error("[tracked-planes] hex lookup failed:", out.hexNote);
        if (out.kind === "base" || out.kind === "fast") deps.onCycle?.(archived, out.points.length);
      }
    } catch (e: any) {
      // a failed cycle is just a missed sample — the next tick retries
      console.error("[tracked-planes] poll:", e?.message || e);
    } finally {
      if (!stopped) {
        const t = Date.now();
        const p = cycle.pacing(registry, t);
        // a fixed interval is measured from the tick's start; the quota's
        // wait is already the time left to the next allowed request
        const wait = cycle.nextDelayMs(registry, t) - (p.paced || p.blocked ? 0 : t - started);
        timer = setTimeout(() => { void tick(); }, Math.max(250, wait));
        unrefTimer(timer);
      }
    }
  };
  void tick(); // first cycle immediately — "when their signal pops up"
  // day-trace backfill: global-network fidelity for every tracked plane
  const backfill = async () => {
    if (stopped) return;
    for (const tp of registry.planes) {
      if (!tp.hex) continue; // no hex learned yet — nothing to trace
      try {
        const fixes = await fetchDayTrace(tp.hex, deps.fetchImpl, deps.headers);
        const mark = tp.backfilled_to ?? 0;
        const fresh = fixes.filter((f) => f.tSec > mark - 60); // 60s overlap; read-dedupe collapses
        if (!fresh.length) continue;
        const n = archiveAircraftAt(fresh, deps.baseDir);
        const newest = fresh[fresh.length - 1].tSec;
        registry = {
          planes: registry.planes.map((p) => (p.reg === tp.reg ? { ...p, backfilled_to: newest } : p)),
        };
        saveRegistry(registry, deps.baseDir);
        deps.onCycle?.(n, fresh.length);
      } catch (e: any) {
        console.error("[tracked-planes] backfill:", tp.reg, e?.message || e);
      }
    }
  };
  const backfillTimer = setInterval(() => { void backfill(); }, TRACE_BACKFILL_MS);
  unrefTimer(backfillTimer);
  const firstBackfill = setTimeout(() => { void backfill(); }, 60_000); // after the poller learns hexes
  unrefTimer(firstBackfill);
  return {
    getRegistry: () => registry,
    cycleStats: () => cycle.stats(),
    pacing: () => cycle.pacing(registry, Date.now()),
    mutate: (fn) => {
      const next = fn(registry);
      if (!next) return false;
      registry = next;
      saveRegistry(registry, deps.baseDir);
      return true;
    },
    stop: () => { stopped = true; if (timer) clearTimeout(timer); clearInterval(backfillTimer); clearTimeout(firstBackfill); },
  };
}
