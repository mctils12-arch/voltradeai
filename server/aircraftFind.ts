// aircraftFind.ts — GET /api/data/aircraft/find?q=<tail | callsign | hex>
// (owner report 2026-11-12: "it seem to be not tracking or picking up on adsb
// ... the planes right away i am looking for n843s").
//
// The map's "Find plane" box searched only the rows inside the current
// viewport, so a tail that was airborne elsewhere — or heard by the provider
// without a position yet — answered "no N843S in the current snapshot". This
// route answers for the whole world, from what the server already holds
// first, and says honestly which of four things is true:
//
//   live       a real position younger than FIND_LIVE_MAX_AGE_MS (the row, in
//              the /global wire shape, and its age);
//   heard      the provider hears the transponder NOW but has no current
//              position for it: altitude and ground speed only — a position
//              is never written or guessed from this;
//   last_seen  no signal now; the newest position we hold (worldwide
//              snapshot, 24/7 tracked registry, or our archive) and its time;
//   unknown    nothing anywhere we can look.
//
// LOOKUP ORDER: (1) the worldwide snapshot — which also holds every tracked
// fix (aircraftFixBus) — by hex, then by exact registration, then by exact
// callsign; (2) the tracked registry (its `heard` reading, its last
// position, and the tail's hex); (3) ONE adsb.lol request, by hex when one is
// known (the hex query returns an aircraft heard without a position — the
// registration query does not), else by registration; (4) our own archive's
// newest fix for the hex.
//
// POLITENESS: answers are cached FIND_CACHE_MS per query and concurrent
// requests for one query share the in-flight lookup; at most
// FIND_MAX_UPSTREAM_PER_MIN upstream requests leave per rolling minute (over
// that the route answers from what the server holds and says so); every
// upstream request is accounted to the shared adsb.lol governor on the
// viewer lane, like the selected-aircraft fast lane. The process-wide per-IP
// limiter (rateLimit.ts) covers the route like every other.
// DAILY QUOTA (2026-11-17, adsbGovernor.ts): the lookup carries the adsb.lol
// API key when one is configured and reports the response's quota headers.
// A Find lookup may spend the quota even when it is kept for the watched
// planes ("watched_only": someone is asking for one aircraft by name); once
// the quota is SPENT no lookup leaves and the answer comes from what the
// server holds, with upstream "limited".
// PRIVACY: community ADS-B only (adsb.lol, ODbL) — nothing FAA/SWIM-sourced.
// IDENTITY: exact matches only; a row is never attributed to a tail it does
// not itself name.

import type { Express } from "express";
import { adsbLolGovernor, adsbLolRequestHeaders, parseRetryAfter, type UpstreamGovernor } from "./adsbGovernor";
import type { FixBatch } from "./aircraftFixBus";
import { mapPointAircraft } from "./aircraftTiling";
import { ROW_FIELDS, encodeRow, toSnapRow, type SnapRow } from "./globalSnapshot";
import {
  HEARD_FRESH_MS, batchUrl, heardFromRaw, hexBatchUrl, normalizeReg, rawAircraftRows, rawHasPosition,
  upstreamBaseMs, type HeardReading, type RawAircraft, type TrackedPlane,
} from "./trackedPlanes";

/** a position younger than this is "live" (the map's own 2 min line) */
export const FIND_LIVE_MAX_AGE_MS = 120_000;
export const FIND_CACHE_MS = 5_000;
export const FIND_CACHE_MAX = 200;
/** upstream lookups allowed per rolling minute, all queries together */
export const FIND_MAX_UPSTREAM_PER_MIN = 12;
export const FIND_TIMEOUT_MS = 6_000;
/** The archive's recent-track read (two days of hour files) may take this
 *  long inside one answer, no longer. Measured live 2026-11-16 on a freshly
 *  booted server with adsb.lol down: the provider lookup timed out (6 s) and
 *  the archive read then took about 45 s; GET /find?q=N843S answered after
 *  53 s. */
export const FIND_ARCHIVE_BUDGET_MS = 2_500;

/** `p`'s value, or `fallback` once `ms` have passed. `p` keeps running (the
 *  archive read fills its own 30 s cache, so the next lookup is fast) and a
 *  late rejection is swallowed here, never unhandled. */
export function withinBudget<T>(p: Promise<T>, ms: number, fallback: T): Promise<{ value: T; timedOut: boolean }> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => resolve({ value: fallback, timedOut: true }), ms);
    (timer as { unref?: () => void }).unref?.();
    p.then(
      (value) => { clearTimeout(timer); resolve({ value, timedOut: false }); },
      // after the budget this reject is a no-op on a settled promise: the late failure ends here
      (e: unknown) => { clearTimeout(timer); reject(e); },
    );
  });
}

export const FIND_SOURCE =
  "adsb.lol (ODbL) community ADS-B — the server's worldwide snapshot and 24/7 tracked-tails registry, " +
  "one adsb.lol lookup when neither holds the aircraft, and our own ADS-B archive for the last position. " +
  // 2026-11-19: the tracked registry has a second source (trackedBackup.ts); the snapshot already had fallbacks
  "The snapshot and the tracked registry can also hold a reading served by adsb.fi (non-commercial): a live row " +
  "names its provider in src, a tracked plane's row in last_pos_provider / heard_provider (GET /api/data/aircraft/tracked)";
export const FIND_HONESTY =
  "live = a broadcast position under 2 min old. heard = the provider's receivers hear this transponder now " +
  "but hold no current position for it (altitude and ground speed only; no position is inferred). " +
  "last_seen = the newest position we hold and its time — the aircraft may have flown since, outside " +
  "receiver coverage or with its transponder off. unknown = not in our snapshot, tracked list, the " +
  "provider's lookup or our archive; that is not proof the aircraft is on the ground. Exact tail number, " +
  "callsign or ICAO hex matches only.";

export type FindState = "live" | "heard" | "last_seen" | "unknown";
export type FindUpstream = "not-needed" | "found" | "empty" | "limited" | "error";

export interface FindQuery {
  /** the query as matched: uppercase tail/callsign */
  ident: string;
  /** set when the query is also a valid 6-digit ICAO hex (lowercase) */
  hex: string | null;
}

/** Strict: a tail number / callsign (2-10 chars: A-Z 0-9 and "-", not
 *  starting with "-") or a 6-digit ICAO hex. Anything else is refused. */
export function parseFindQuery(q: unknown): FindQuery | null {
  if (typeof q !== "string") return null;
  const ident = normalizeReg(q);
  if (!ident) return null;
  return { ident, hex: /^[0-9A-F]{6}$/.test(ident) ? ident.toLowerCase() : null };
}

export interface FindLastSeen {
  /** unix ms of that position */
  at: number;
  lat: number;
  lon: number;
  alt_ft: number | null;
  via: "snapshot" | "tracked" | "archive" | "upstream";
}

export interface FindResult {
  kind: "raw";
  q: string;
  state: FindState;
  hex: string | null;
  reg: string | null;
  /** server clock of this answer (ms) */
  at: number;
  /** age of what the state rests on: the fix (live), the last message
   *  (heard), the last position (last_seen); null for unknown */
  age_s: number | null;
  fields: readonly string[];
  /** one /global-shaped row when state is live, else empty */
  rows: (string | number | null)[][];
  heard: HeardReading | null;
  last_seen: FindLastSeen | null;
  /** is this tail on the 24/7 tracked list? */
  tracked: boolean;
  answered_from: "snapshot" | "tracked" | "upstream" | "archive" | null;
  upstream: FindUpstream;
  /** what happened to the archive's recent-track read for this answer:
   *  "read"; "not_needed" (a live or heard answer); "timed_out"
   *  (not finished within FIND_ARCHIVE_BUDGET_MS: this answer does not include
   *  it, and asking again in a few seconds will); "failed"; "no_hex" */
  archive: "read" | "not_needed" | "timed_out" | "failed" | "no_hex";
  source: string;
  honesty: string;
}

export interface FindDeps {
  snapshot: { get(hex: string): SnapRow | undefined; rows(): SnapRow[] };
  tracked: () => TrackedPlane[];
  /** newest archived fix for a hex (null = none) */
  lastArchived?: (hex: string) => Promise<{ t: number; la: number; lo: number; al?: number } | null>;
  publish?: (b: FixBatch) => void;
  fetchImpl?: typeof fetch;
  governor?: UpstreamGovernor;
  now?: () => number;
  headers?: Record<string, string>;
  /** where ADSBLOL_API_KEY is read when `headers` is not given (default process.env) */
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
  /** time allowed for lastArchived inside one answer (default FIND_ARCHIVE_BUDGET_MS) */
  archiveBudgetMs?: number;
}

export interface AircraftFinder {
  find(q: FindQuery): Promise<FindResult>;
  stats(): Record<string, number>;
}

const M_TO_FT = 1 / 0.3048;
const errText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export function createAircraftFinder(deps: FindDeps): AircraftFinder {
  const now = deps.now ?? Date.now;
  const fetchImpl = deps.fetchImpl ?? fetch;
  const gov = deps.governor ?? adsbLolGovernor;
  const headers = deps.headers ?? adsbLolRequestHeaders(deps.env);
  const timeoutMs = deps.timeoutMs ?? FIND_TIMEOUT_MS;
  const cache = new Map<string, { at: number; res: FindResult }>();
  const inflight = new Map<string, Promise<FindResult>>();
  const upstreamSent: number[] = [];
  const counts = { requests: 0, cache_hits: 0, coalesced: 0, upstream_requests: 0, upstream_limited: 0, upstream_quota_skips: 0, upstream_errors: 0, archive_reads: 0, archive_timeouts: 0 };
  let lastError: string | null = null;

  const allowUpstream = (t: number): boolean => {
    while (upstreamSent.length && t - upstreamSent[0] >= 60_000) upstreamSent.shift();
    if (upstreamSent.length >= FIND_MAX_UPSTREAM_PER_MIN) return false;
    upstreamSent.push(t);
    return true;
  };

  /** ONE adsb.lol request: by hex when known, else by registration. */
  const askUpstream = async (q: { hex: string | null; ident: string }): Promise<{ row: RawAircraft | null; raw: unknown } | "error"> => {
    const t0 = now();
    counts.upstream_requests++;
    gov.noteRequest("fg", t0);
    let status = 0;
    try {
      const url = q.hex ? hexBatchUrl([q.hex]) : batchUrl([q.ident]);
      const r = await fetchImpl(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      status = r.status;
      gov.noteQuota(r.headers, now(), r.status);
      if (!r.ok) {
        gov.noteResult("fg", r.status, now(), parseRetryAfter(r.headers?.get?.("retry-after") ?? null));
        throw new Error(`adsb.lol find ${r.status}`);
      }
      const raw: unknown = await r.json();
      gov.noteResult("fg", 200, now());
      const rows = rawAircraftRows(raw);
      const row = q.hex
        ? rows.find((a) => String(a.hex || "").toLowerCase() === q.hex)
        : rows.find((a) => normalizeReg(typeof a.r === "string" ? a.r : null) === q.ident);
      return { row: row ?? null, raw };
    } catch (e) {
      if (status === 0) gov.noteResult("fg", 0, now()); // network error / timeout
      counts.upstream_errors++;
      lastError = errText(e);
      return "error";
    }
  };

  const lookup = async (q: FindQuery): Promise<FindResult> => {
    const t = now();
    // `at` and every age are read off the clock when the answer is BUILT
    // (an upstream lookup can take seconds)
    const base = (o: Partial<FindResult> & { state: FindState }): FindResult => ({
      kind: "raw", q: q.ident, hex: null, reg: null, at: now(), age_s: null,
      fields: ROW_FIELDS, rows: [], heard: null, last_seen: null, tracked: false,
      answered_from: null, upstream: "not-needed", archive: "not_needed", source: FIND_SOURCE, honesty: FIND_HONESTY,
      ...o,
    });
    const ageS = (ms: number) => Math.max(0, Math.round((now() - ms) / 100) / 10);

    // 1. the worldwide snapshot (which also holds the tracked fixes)
    let snap: SnapRow | undefined = q.hex ? deps.snapshot.get(q.hex) : undefined;
    if (!snap) {
      const rows = deps.snapshot.rows();
      snap = rows.find((r) => String(r.reg || "").toUpperCase() === q.ident)
        ?? rows.find((r) => String(r.callsign || "").toUpperCase() === q.ident);
    }
    // 2. the tracked registry: this tail, or the hex the snapshot row names
    const planes = deps.tracked();
    const hexHint = q.hex ?? snap?.hex ?? null;
    const tp = planes.find((p) => p.reg === q.ident)
      ?? (hexHint ? planes.find((p) => String(p.hex || "").toLowerCase() === hexHint) : undefined);
    const hex = hexHint ?? (tp?.hex ? String(tp.hex).toLowerCase() : null);
    const reg = tp?.reg ?? (snap?.reg ? String(snap.reg).toUpperCase() : null);
    const ident = { hex, reg, tracked: !!tp };

    if (snap && t - snap.seenAt <= FIND_LIVE_MAX_AGE_MS) {
      return base({ ...ident, hex: snap.hex, state: "live", age_s: ageS(snap.seenAt), rows: [encodeRow(snap)], answered_from: "snapshot" });
    }
    const lastSeen: FindLastSeen[] = [];
    if (snap) lastSeen.push({ at: snap.seenAt, lat: snap.lat, lon: snap.lon, alt_ft: snap.altFt, via: "snapshot" });
    if (tp?.last_seen != null && tp.last_pos && Number.isFinite(tp.last_pos.la) && Number.isFinite(tp.last_pos.lo)) {
      lastSeen.push({
        at: tp.last_seen, lat: tp.last_pos.la, lon: tp.last_pos.lo,
        alt_ft: tp.last_pos.al == null ? null : Math.round(tp.last_pos.al * M_TO_FT), via: "tracked",
      });
    }
    const newest = (): FindLastSeen | null => (lastSeen.length ? lastSeen.reduce((a, b) => (b.at > a.at ? b : a)) : null);
    const heardFresh = (h: HeardReading | null | undefined): h is HeardReading => !!h && t - h.at <= HEARD_FRESH_MS;
    if (heardFresh(tp?.heard) && !tp?.heard?.has_position) {
      return base({ ...ident, state: "heard", age_s: ageS(tp.heard.at), heard: tp.heard, last_seen: newest(), answered_from: "tracked" });
    }

    // 3. ONE upstream lookup
    let upstream: FindUpstream = "limited";
    let heard: HeardReading | null = null;
    // adsb.lol's daily quota is spent: nothing leaves until its reset
    if (!gov.admit("find", t)) counts.upstream_quota_skips++;
    else if (!allowUpstream(t)) counts.upstream_limited++;
    else {
      const up = await askUpstream({ hex, ident: q.ident });
      if (up === "error") upstream = "error";
      else if (!up.row) upstream = "empty";
      else {
        upstream = "found";
        const upBase = upstreamBaseMs(up.raw, now());
        const upHex = String(up.row.hex || "").toLowerCase() || hex;
        const upReg = normalizeReg(typeof up.row.r === "string" ? up.row.r : null) ?? reg;
        heard = heardFromRaw(up.row, upBase);
        if (rawHasPosition(up.row)) {
          const batch: FixBatch = {
            provider: "adsblol", origin: "find", fetchedAt: now(), upstreamNowMs: upBase,
            aircraft: mapPointAircraft({ ac: [up.row] }, "ac", "adsblol"),
          };
          deps.publish?.(batch); // the snapshot (and every other viewer) gets the fix
          const row = batch.aircraft.length ? toSnapRow(batch.aircraft[0], batch) : null;
          if (row && now() - row.seenAt <= FIND_LIVE_MAX_AGE_MS) {
            return base({ hex: row.hex, reg: upReg, tracked: !!tp, state: "live", age_s: ageS(row.seenAt), rows: [encodeRow(row)], answered_from: "upstream", upstream });
          }
          if (row) lastSeen.push({ at: row.seenAt, lat: row.lat, lon: row.lon, alt_ft: row.altFt, via: "upstream" });
        }
        if (heardFresh(heard)) {
          return base({ hex: upHex, reg: upReg, tracked: !!tp, state: "heard", age_s: ageS(heard.at), heard, last_seen: newest(), answered_from: "upstream", upstream });
        }
      }
    }

    // 4. the newest position we hold: snapshot / registry / our archive.
    // The archive read gets FIND_ARCHIVE_BUDGET_MS. Past that the answer is
    // built from what is already held and says the archive was not read in
    // time; the read keeps running and fills its cache for the next lookup.
    let archive: FindResult["archive"] = hex ? "not_needed" : "no_hex";
    if (hex && deps.lastArchived) {
      counts.archive_reads++;
      try {
        const got = await withinBudget(deps.lastArchived(hex), deps.archiveBudgetMs ?? FIND_ARCHIVE_BUDGET_MS, null);
        if (got.timedOut) { counts.archive_timeouts++; archive = "timed_out"; }
        else {
          archive = "read";
          const a = got.value;
          if (a && Number.isFinite(a.la) && Number.isFinite(a.lo)) {
            lastSeen.push({ at: a.t * 1000, lat: a.la, lon: a.lo, alt_ft: a.al == null ? null : Math.round(a.al * M_TO_FT), via: "archive" });
          }
        }
      } catch (e) {
        archive = "failed";
        lastError = errText(e);
      }
    }
    const last = newest();
    if (last) {
      return base({ ...ident, state: "last_seen", age_s: ageS(last.at), last_seen: last, answered_from: last.via === "upstream" ? "upstream" : last.via, upstream, archive });
    }
    return base({ ...ident, state: "unknown", upstream, archive });
  };

  return {
    find(q: FindQuery): Promise<FindResult> {
      counts.requests++;
      const t = now();
      const key = q.ident;
      const c = cache.get(key);
      if (c && t - c.at <= FIND_CACHE_MS) {
        counts.cache_hits++;
        // a cached answer is re-aged, never served with its old clock
        const older = (t - c.res.at) / 1000;
        return Promise.resolve({ ...c.res, at: t, age_s: c.res.age_s == null ? null : Math.round((c.res.age_s + older) * 10) / 10 });
      }
      const pending = inflight.get(key);
      if (pending) { counts.coalesced++; return pending; }
      const p = lookup(q).then((res) => {
        cache.set(key, { at: now(), res });
        if (cache.size > FIND_CACHE_MAX) {
          const oldest = cache.keys().next().value;
          if (oldest !== undefined) cache.delete(oldest);
        }
        return res;
      }).finally(() => { inflight.delete(key); });
      inflight.set(key, p);
      return p;
    },
    stats() {
      return { ...counts, cached: cache.size, last_error_set: lastError ? 1 : 0 };
    },
  };
}

/** Registered from routes.ts with ONE line (needs the snapshot the global
 *  route owns and the tracked poller's registry). */
export function registerAircraftFindRoute(app: Express, deps: FindDeps): AircraftFinder {
  const finder = createAircraftFinder(deps);
  app.get("/api/data/aircraft/find", async (req, res) => {
    const q = parseFindQuery(req.query.q);
    if (!q) {
      return res.status(400).json({ error: "q must be a tail number, a callsign or a 6-digit ICAO hex (2-10 characters: letters, digits, '-')" });
    }
    try {
      const out = await finder.find(q);
      res.set("Cache-Control", "no-store");
      res.json(out);
    } catch (e) {
      res.status(500).json({ error: errText(e) || "find failed" });
    }
  });
  return finder;
}
