// sweepFailover.ts — while adsb.lol is not answering the world sweep, the
// same disc plan is swept through the other community providers
// (2026-11-16, human task "merge several providers to cover gaps").
//
// WHY. The world sweep (globalSweep.ts) asks adsb.lol only. When adsb.lol
// rate-limits or stops answering, the shared governor makes the sweep wait
// (30 s doubling to 15 min) and the worldwide feed simply ages: read live
// 2026-11-17 01:47Z, 426 rows of a usual 8,000 to 9,000 and none fresher
// than 2 min, while a viewport request at the same moment was answered by
// adsb.fi in 0.4 s with 522 aircraft. The viewport chain and the selected-
// aircraft lane already fall back; the sweep did not.
//
// WHAT. One sequential loop that is idle unless the governor reports the
// sweep failing (stats().sweep_consecutive_failures > 0 — raised only by a
// failed sweep request, cleared only by a successful one, so it stays up
// across the sweep's own retry probes) OR kept off adsb.lol by its DAILY
// QUOTA (2026-11-17, adsbGovernor.ts: the quota is spent, or the limit is too
// small for a world sweep and is kept for the watched planes — then the
// sweep sends nothing, so it has no failure to report, and without this the
// world feed would simply age). While either holds, the loop takes
// the plan's discs in value order (the sweep's own scheduler with no type
// lane: age x sqrt(aircraft last seen there), viewer interest boosted,
// nothing older than 45 min) and asks the fallback providers for each disc.
// The order it starts in comes from where aircraft are: the positions the
// snapshot still holds when the sweep starts failing, or — on a server that
// booted with adsb.lol already down — where the large and medium airports
// are. The mask's own prior is one number per region and would spend the
// first minutes on that region's empty edge (measured 2026-11-17: the first
// three discs by prior lie over the Pacific off Baja and held 0 aircraft).
// Every answer goes on the fix bus with the provider that served it, so the
// snapshot row's `src` is that provider and `?lawful=1` (adsb.lol only)
// stays a provable subset.
//
// ON DEMAND (2026-11-17). With the quota keeping the sweep off adsb.lol for
// good ("watched planes only"), a failover that ran whenever it had a reason
// would ask each provider 43,200 times a day for a world feed nobody may be
// looking at. So it also needs DEMAND: the caller's `demand()` (the
// worldwide endpoint was requested in the last FAILOVER_DEMAND_WINDOW_MS).
// With a reason but no demand it sends nothing and its status says so; the
// feed takes a minute or two to fill after the first request. Without a
// `demand` dep it behaves as before (always wanted).
//
// WHAT IT IS NOT. Not a replacement for the type lane: neither fallback
// provider has a worldwide by-type query (adsb.fi answers /v2/type with 400,
// probed 2026-11-17), so coverage is one 250 nm disc per request. At the
// default pace one provider refreshes about 300 of the plan's ~780 discs
// inside the snapshot's 10 min age limit, busiest first. The endpoint
// reports the discs actually refreshed; it never claims the whole plan.
//
// POLITENESS. Each provider gets at most SWEEP_FAILOVER_RPS requests per
// second (default 0.5, hard maximum 1 — both providers publish 1 request per
// second as their limit, and the viewport chain may be asking them too).
// Requests are strictly sequential. A failure backs that provider off 30 s
// doubling to 15 min, Retry-After honored in full; 401/403 (the provider is
// refusing this server) backs off 30 min at once.
//
// LICENSING. adsb.fi and airplanes.live are NON-COMMERCIAL
// (providerCompliance.ts). The failover does not run while billing is
// active — a new use of a non-commercial provider must not start under
// monetization — and the existing tripwire already counts both providers.
//
// KILL SWITCH: SWEEP_FAILOVER_ENABLED=0 (default on), or SWEEP_FAILOVER_RPS=0.

import { mapPointAircraft, type DiscProvider } from "./aircraftTiling";
import { adsbLolGovernor, parseRetryAfter, type UpstreamGovernor } from "./adsbGovernor";
import { providerClockMs, publishFixes, type FixBatch } from "./aircraftFixBus";
import { adsbFiDiscUrl } from "./coverageGapFill";
import { haversineNm, worldPlan, type PlanDisc } from "./globalDiscPlan";
import { MAX_DISC_AGE_MS, SweepScheduler, UA, sweepEnabled, unrefTimer } from "./globalSweep";
import { majorAirportPoints } from "./airportsIndex";
import { billingActive } from "./providerCompliance";
import type { AircraftPoint } from "./datacoreArchive";

/** The bus origin of a failover disc (snapshot `via`, archive decision). */
export const FAILOVER_ORIGIN = "sweep-failover";
export const DEFAULT_FAILOVER_RPS = 0.5;
export const MAX_FAILOVER_RPS = 1;
export const FAILOVER_TIMEOUT_MS = 12_000;
export const FAILOVER_BACKOFF_BASE_MS = 30_000;
export const FAILOVER_BACKOFF_MAX_MS = 15 * 60_000;
/** 401/403: the provider is refusing this server, not having a bad minute */
export const FAILOVER_REFUSED_BACKOFF_MS = 30 * 60_000;
/** how long the idle loop sleeps between looks at the governor */
export const FAILOVER_IDLE_POLL_MS = 2_000;
/** the loop's first look, after boot (the sweep's own start delay) */
export const FAILOVER_START_DELAY_MS = 20_000;
/** a disc that failed this many times in a row is passed over, so one disc
 *  a provider cannot answer never holds the queue */
export const FAILOVER_DISC_MAX_FAILS = 2;
/** state-change log lines: at most one per this long */
export const FAILOVER_LOG_EVERY_MS = 5 * 60_000;
/** the window `discs_refreshed_last_cycle` counts over (the snapshot's age limit) */
export const FAILOVER_CYCLE_WINDOW_MS = 10 * 60_000;

/** the worldwide feed counts as wanted for this long after its endpoint was last requested */
export const FAILOVER_DEMAND_WINDOW_MS = 10 * 60_000;
/** what the status says while there is a reason to run and nobody is asking */
export const FAILOVER_IDLE_NO_DEMAND =
  `idle: nobody has asked for the worldwide feed in the last ${FAILOVER_DEMAND_WINDOW_MS / 60_000} min`;

/** the snapshot must hold at least this many positions to order the discs
 *  by them; under it the airports are used instead */
export const FAILOVER_LIVE_SEED_MIN_ROWS = 1_000;
/** a large airport counts this many times a medium one in the cold order */
export const FAILOVER_LARGE_AIRPORT_WEIGHT = 3;

export interface WeightedPoint { lat: number; lon: number; w?: number }

/** Sum of point weights inside each plan disc (disc id -> weight; discs
 *  holding nothing are absent). Discs are bucketed in 5 degree latitude
 *  bands: a 250 nm radius is 4.2 degrees, so a point is only ever compared
 *  with the discs of its own band and the two next to it. */
export function weightPerDisc(plan: PlanDisc[], points: Iterable<WeightedPoint>): Map<number, number> {
  const BAND = 5;
  const bands = new Map<number, PlanDisc[]>();
  for (const d of plan) {
    const k = Math.floor(d.lat / BAND);
    const arr = bands.get(k);
    if (arr) arr.push(d); else bands.set(k, [d]);
  }
  const out = new Map<number, number>();
  for (const p of points) {
    if (!Number.isFinite(p.lat) || !Number.isFinite(p.lon)) continue;
    const k = Math.floor(p.lat / BAND);
    for (let b = k - 1; b <= k + 1; b++) {
      for (const d of bands.get(b) ?? []) {
        if (Math.abs(d.lat - p.lat) * 60 > d.radiusNm) continue;
        if (haversineNm(d.lat, d.lon, p.lat, p.lon) <= d.radiusNm) out.set(d.id, (out.get(d.id) ?? 0) + (p.w ?? 1));
      }
    }
  }
  return out;
}

/** adsb.fi first: it is the one measured answering this server
 *  (2026-11-17). Both share the readsb point+radius row shape. */
export const FAILOVER_PROVIDERS: DiscProvider[] = [
  { key: "adsbfi", label: "adsb.fi (ADS-B, community)", arr: "aircraft", url: (la, lo, r) => adsbFiDiscUrl(la, lo, r) },
  { key: "airplaneslive", label: "airplanes.live (ADS-B, community)", arr: "ac", url: (la, lo, r) => `https://api.airplanes.live/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${Math.round(r)}` },
];

const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export function failoverRpsFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const n = parseFloat(String(env.SWEEP_FAILOVER_RPS ?? ""));
  return Number.isFinite(n) ? Math.min(MAX_FAILOVER_RPS, Math.max(0, n)) : DEFAULT_FAILOVER_RPS;
}

/** Why the failover is off, or null when it may run. */
export function failoverOffReason(env: NodeJS.ProcessEnv = process.env): string | null {
  const v = String(env.SWEEP_FAILOVER_ENABLED ?? "1").trim().toLowerCase();
  if (v === "0" || v === "false" || v === "off" || v === "no") return "SWEEP_FAILOVER_ENABLED=0";
  if (failoverRpsFromEnv(env) <= 0) return "SWEEP_FAILOVER_RPS=0";
  if (!sweepEnabled(env)) return "the world sweep itself is off (GLOBAL_SWEEP_ENABLED=0)";
  if (billingActive(env)) return "billing is active: adsb.fi and airplanes.live are non-commercial";
  return null;
}

/** Backoff after the n-th failure in a row (n >= 1). */
export function failoverBackoffMs(status: number, failures: number, retryAfterMs: number | null): number {
  if (status === 401 || status === 403) return Math.max(FAILOVER_REFUSED_BACKOFF_MS, retryAfterMs ?? 0);
  const own = Math.min(FAILOVER_BACKOFF_MAX_MS, FAILOVER_BACKOFF_BASE_MS * 2 ** Math.max(0, failures - 1));
  return Math.max(own, retryAfterMs ?? 0);
}

interface ProviderState {
  p: DiscProvider;
  nextAt: number;       // earliest next request (pace)
  backoffUntil: number; // earliest next request (failure)
  failures: number;     // in a row
  requests: number; ok: number; failed: number; rate_limited: number; refused: number;
  aircraft_total: number;
  last_ok_at: number | null;
  last_status: number | null;
  last_error: string | null;
}

export interface FailoverStepResult {
  provider: string;
  ok: boolean;
  status: number;
  count: number;
  error?: string;
}

export interface FailoverStatus {
  enabled: boolean;
  /** why it is off (null when enabled) */
  off_reason: string | null;
  /** true while the adsb.lol sweep is failing, or held off adsb.lol by its
   *  daily quota, AND the worldwide feed is wanted: the failover is sweeping */
  active: boolean;
  /** why it has cause to run: "adsb.lol sweep failing", "adsb.lol daily
   *  quota: watched planes only", "adsb.lol daily quota spent until <ISO
   *  time>" — followed by "; idle: nobody has asked for the worldwide feed in
   *  the last 10 min" when that is all that keeps it from running. null when
   *  the adsb.lol sweep is doing its job. */
  reason: string | null;
  /** is the worldwide feed wanted right now (its endpoint was requested recently)? */
  demand: boolean;
  /** when the worldwide endpoint was last requested (ms); null = never, or not tracked */
  last_demand_at: number | null;
  active_since: number | null;
  activations_total: number;
  /** seconds spent active since boot (the current stretch included) */
  active_s_total: number;
  rps_per_provider: number;
  requests_total: number;
  ok_total: number;
  failed_total: number;
  aircraft_last_answer: number | null;
  /** what the current disc order started from: the snapshot's own positions
   *  ("live"), the large and medium airports ("airports"), or only the
   *  plan's regional prior ("plan") */
  order_from: "live" | "airports" | "plan";
  order_set_at: number | null;
  last: FailoverStepResult | null;
  last_at: number | null;
  /** the failover's OWN answered discs — not the adsb.lol sweep's, and not
   *  discs a viewer's map fetched */
  discs_in_plan: number;
  discs_answered_ever: number;
  discs_refreshed_last_cycle: number;
  cycle_window_s: number;
  providers: Array<{
    key: string; label: string; requests: number; ok: number; failed: number; rate_limited: number;
    refused: number; aircraft_total: number; consecutive_failures: number; backoff_s: number;
    last_ok_at: number | null; last_status: number | null; last_error: string | null;
  }>;
  rule: string;
}

export const FAILOVER_RULE =
  "Runs only while the adsb.lol world sweep is failing (a sweep request failed and no later one has " +
  "succeeded) or is kept off adsb.lol by its daily request quota (spent, or too small for a world sweep " +
  "and kept for the watched planes; see reason). It then asks adsb.fi and airplanes.live (both non-commercial) for the sweep plan's 250 nm " +
  "discs, busiest and most-watched first, one request at a time, at most the stated rate per provider. Neither " +
  "has a worldwide by-type query, so this covers part of the plan (see discs_refreshed_last_cycle), not all of " +
  "it. Each row keeps the provider that served it (src). Off while billing is active. It runs on demand: only " +
  "while the worldwide endpoint has been requested in the last 10 min, so the feed takes a minute or two to fill " +
  "after the first request.";

export const FAILOVER_REASON_FAILING = "adsb.lol sweep failing";
export const FAILOVER_REASON_WATCHED_ONLY = "adsb.lol daily quota: watched planes only";

/** Why the failover should be sweeping now, or null when the adsb.lol sweep
 *  is doing its job. The quota comes first: while it holds the sweep off,
 *  that is the reason whatever the last failure count was. */
export function failoverReason(governor: UpstreamGovernor, now: number): string | null {
  const held = governor.sweepSuspendedByQuota(now);
  if (held === "exhausted") {
    return `adsb.lol daily quota spent until ${new Date(governor.quotaExhaustedUntil(now) ?? now).toISOString()}`;
  }
  if (held === "watched_only") return FAILOVER_REASON_WATCHED_ONLY;
  return governor.stats(now).sweep_consecutive_failures > 0 ? FAILOVER_REASON_FAILING : null;
}

export interface SweepFailover {
  scheduler: SweepScheduler;
  /** one pass of the loop body: returns the request made, or null when it made none */
  tick: (now: number) => Promise<FailoverStepResult | null>;
  status: (now?: number) => FailoverStatus;
  stop: () => void;
}

export function createSweepFailover(deps: {
  env?: NodeJS.ProcessEnv;
  fetchImpl?: typeof fetch;
  governor?: UpstreamGovernor;
  plan?: PlanDisc[];
  providers?: DiscProvider[];
  publish?: (b: FixBatch) => void;
  /** positions the worldwide snapshot holds right now (orders the discs) */
  livePositions?: () => Array<{ lat: number; lon: number }>;
  /** large/medium airports (the cold order); default the repo's dataset */
  airportPoints?: () => Array<{ lat: number; lon: number; large: boolean }>;
  now?: () => number;
  timeoutMs?: number;
  /** is the worldwide feed wanted right now? No request leaves without it.
   *  Default: always (the failover runs whenever it has a reason). */
  demand?: () => boolean;
  /** when the feed was last asked for (status only) */
  lastDemandAt?: () => number | null;
  /** false = build it without starting the timer loop (tests drive tick()) */
  start?: boolean;
} = {}): SweepFailover {
  const env = deps.env ?? process.env;
  const governor = deps.governor ?? adsbLolGovernor;
  const fetchImpl = deps.fetchImpl ?? fetch;
  const publish = deps.publish ?? publishFixes;
  const now = deps.now ?? Date.now;
  const demand = deps.demand ?? (() => true);
  const offReason = failoverOffReason(env);
  const rps = failoverRpsFromEnv(env);
  const gapMs = rps > 0 ? Math.ceil(1000 / rps) : Number.POSITIVE_INFINITY;
  const limitMs = deps.timeoutMs ?? FAILOVER_TIMEOUT_MS;
  // the sweep's own value-weighted disc scheduler, with no type lane: a
  // disc's weight is every aircraft its last answer held
  const scheduler = new SweepScheduler(deps.plan ?? worldPlan(), { now: now(), types: [] });
  const provs: ProviderState[] = (deps.providers ?? FAILOVER_PROVIDERS).map((p) => ({
    p, nextAt: 0, backoffUntil: 0, failures: 0,
    requests: 0, ok: 0, failed: 0, rate_limited: 0, refused: 0, aircraft_total: 0,
    last_ok_at: null, last_status: null, last_error: null,
  }));
  let active = false;
  let activeSince: number | null = null;
  let activations = 0;
  let activeMsDone = 0;
  let last: FailoverStepResult | null = null;
  let lastAt: number | null = null;
  let lastCount: number | null = null;
  let stopped = false;
  let lastErrLog = 0;
  /** disc id -> when THIS failover last got an answer for it */
  const answeredAt = new Map<number, number>();
  /** disc id -> failures in a row */
  const discFails = new Map<number, number>();
  let lastStateLog = -Infinity;

  let orderFrom: "live" | "airports" | "plan" = "plan";
  let orderSetAt: number | null = null;
  let airportWeights: Map<number, number> | null = null; // computed once
  let seedErrLogged = false;
  /** Order the queue from where aircraft are. Discs this failover answered
   *  inside the cycle window keep their real count and age; every other disc
   *  gets its weight and the constructor's "half overdue" age, so the busiest
   *  come first and an hours-old age cannot force one full pass of the whole
   *  plan before anything is refreshed twice. */
  const seedOrder = (t: number) => {
    try {
      let weights: Map<number, number> | null = null;
      let from: "live" | "airports" | "plan" = "plan";
      const live = deps.livePositions?.() ?? [];
      if (live.length >= FAILOVER_LIVE_SEED_MIN_ROWS) {
        weights = weightPerDisc(scheduler.discs.map((s) => s.disc), live);
        from = "live";
      } else {
        if (!airportWeights) {
          const pts = (deps.airportPoints ?? majorAirportPoints)();
          airportWeights = weightPerDisc(scheduler.discs.map((s) => s.disc),
            pts.map((a) => ({ lat: a.lat, lon: a.lon, w: a.large ? FAILOVER_LARGE_AIRPORT_WEIGHT : 1 })));
        }
        if (airportWeights.size) { weights = airportWeights; from = "airports"; }
      }
      for (const st of scheduler.discs) {
        const at = answeredAt.get(st.disc.id);
        if (at != null && t - at <= FAILOVER_CYCLE_WINDOW_MS) continue;
        st.lastAt = t - MAX_DISC_AGE_MS / 2;
        if (weights) st.residual = weights.get(st.disc.id) ?? 0;
      }
      orderFrom = from;
      orderSetAt = t;
    } catch (e) {
      // the plan's own order still works; said once
      if (!seedErrLogged) { seedErrLogged = true; console.error("[sweep-failover] disc order:", errMsg(e)); }
    }
  };

  const setActive = (why: string | null, t: number) => {
    const on = why != null;
    if (on === active) return;
    active = on;
    if (on) {
      activations++;
      activeSince = t;
      seedOrder(t);
    } else {
      activeMsDone += t - (activeSince ?? t);
      activeSince = null;
    }
    if (t - lastStateLog >= FAILOVER_LOG_EVERY_MS) {
      lastStateLog = t;
      console.error(on
        ? `[sweep-failover] ${why}${why === FAILOVER_REASON_FAILING ? ` (${governor.stats(t).sweep_consecutive_failures} in a row)` : ""} — sweeping the disc plan through ${provs.map((s) => s.p.key).join(", ")} (activation ${activations})`
        : failoverReason(governor, t) == null
          ? "[sweep-failover] the adsb.lol sweep is running again — failover idle"
          : `[sweep-failover] ${FAILOVER_IDLE_NO_DEMAND}`);
    }
  };

  /** the provider that may be asked now: not backing off, pace elapsed, the
   *  one that has waited longest first */
  const pickProvider = (t: number): ProviderState | null => {
    let best: ProviderState | null = null;
    for (const s of provs) {
      if (t < s.backoffUntil || t < s.nextAt) continue;
      if (!best || s.nextAt < best.nextAt) best = s;
    }
    return best;
  };

  const fetchDisc = async (s: ProviderState, d: PlanDisc): Promise<{ status: number; raw: unknown; retryAfterMs: number | null }> => {
    // strongly-held deadline: an abort signal that is not honoured must not
    // be able to freeze a sequential loop (the 2026-10-12 sweep wedge)
    const ac = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<never>((_, reject) => {
      timer = setTimeout(() => { ac.abort(); reject(new Error(`request exceeded ${limitMs}ms hard deadline`)); }, limitMs);
    });
    deadline.catch(() => {});
    try {
      const r = await Promise.race([fetchImpl(s.p.url(d.lat, d.lon, d.radiusNm), { headers: UA, signal: ac.signal }), deadline]);
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

  const tick = async (t: number): Promise<FailoverStepResult | null> => {
    if (offReason) return null;
    // a reason to run AND someone asking for the worldwide feed
    const why = failoverReason(governor, t);
    setActive(why != null && demand() ? why : null, t);
    if (!active) return null;
    const s = pickProvider(t);
    if (!s) return null;
    const job = scheduler.next(t);
    if (!job || job.kind !== "disc") return null;
    const d = job.state.disc;
    s.nextAt = t + gapMs;
    s.requests++;
    let res: FailoverStepResult;
    // a failed disc stays first in the queue for the other provider, unless
    // the provider rejected the request itself (4xx other than a rate limit
    // or a refusal) or the disc has now failed FAILOVER_DISC_MAX_FAILS times
    const discFailed = (status: number, at: number) => {
      const n = (discFails.get(d.id) ?? 0) + 1;
      const rejected = status >= 400 && status < 500 && status !== 429 && status !== 401 && status !== 403;
      if (rejected || n >= FAILOVER_DISC_MAX_FAILS) {
        discFails.delete(d.id);
        scheduler.markDisc(job.state, at, job.state.residual, false);
      } else discFails.set(d.id, n);
    };
    try {
      const { status, raw, retryAfterMs } = await fetchDisc(s, d);
      const at = now();
      s.last_status = status;
      if (status !== 200) {
        s.failed++; s.failures++;
        if (status === 429) s.rate_limited++;
        if (status === 401 || status === 403) s.refused++;
        s.last_error = `${s.p.key} ${status}`;
        s.backoffUntil = at + failoverBackoffMs(status, s.failures, retryAfterMs);
        discFailed(status, at);
        res = { provider: s.p.key, ok: false, status, count: 0, error: s.last_error };
      } else {
        const aircraft = mapPointAircraft(raw, s.p.arr, s.p.key) as AircraftPoint[];
        s.ok++; s.failures = 0; s.backoffUntil = 0; s.last_ok_at = at; s.last_error = null;
        s.aircraft_total += aircraft.length;
        publish({
          provider: s.p.key, origin: FAILOVER_ORIGIN, aircraft, fetchedAt: at,
          upstreamNowMs: providerClockMs((raw as { now?: unknown } | null)?.now),
          disc: { lat: d.lat, lon: d.lon, radiusNm: d.radiusNm },
        });
        scheduler.markDisc(job.state, at, aircraft.length, true);
        answeredAt.set(d.id, at);
        discFails.delete(d.id);
        lastCount = aircraft.length;
        res = { provider: s.p.key, ok: true, status: 200, count: aircraft.length };
      }
    } catch (e) {
      const at = now();
      s.failed++; s.failures++;
      s.last_status = 0;
      s.last_error = errMsg(e);
      s.backoffUntil = at + failoverBackoffMs(0, s.failures, null);
      discFailed(0, at);
      res = { provider: s.p.key, ok: false, status: 0, count: 0, error: s.last_error };
    }
    last = res;
    lastAt = now();
    if (!res.ok && lastAt - lastErrLog > 60_000) {
      lastErrLog = lastAt;
      console.error(`[sweep-failover] ${res.provider} request failed (${res.error}) — that provider waits ${Math.ceil((s.backoffUntil - lastAt) / 1000)}s`);
    }
    return res;
  };

  /** read live, so the answer to the request that IS the demand never says "nobody has asked" */
  const statusReason = (t: number): string | null => {
    if (offReason) return null;
    const why = failoverReason(governor, t);
    if (why == null) return null;
    return demand() ? why : `${why}; ${FAILOVER_IDLE_NO_DEMAND}`;
  };

  const timers = new Set<ReturnType<typeof setTimeout>>();
  const sleep = (ms: number) => new Promise<void>((resolve) => {
    const tm = setTimeout(() => { timers.delete(tm); resolve(); }, ms);
    unrefTimer(tm);
    timers.add(tm);
  });
  const loop = async () => {
    await sleep(FAILOVER_START_DELAY_MS);
    while (!stopped) {
      let made: FailoverStepResult | null = null;
      try { made = await tick(now()); } catch (e) { console.error("[sweep-failover] tick:", errMsg(e)); }
      if (stopped) break;
      if (!active) { await sleep(FAILOVER_IDLE_POLL_MS); continue; }
      if (made) continue; // the next provider may already be due
      // active but nothing was due: sleep until the earliest provider is
      const t = now();
      const due = provs.reduce((m, s) => Math.min(m, Math.max(s.nextAt, s.backoffUntil)), Number.POSITIVE_INFINITY);
      await sleep(Math.min(FAILOVER_IDLE_POLL_MS, Math.max(50, due - t)));
    }
  };
  if (!offReason && deps.start !== false) void loop();

  return {
    scheduler,
    tick,
    status: (t: number = now()) => {
      let refreshed = 0;
      for (const at of Array.from(answeredAt.values())) if (t - at <= FAILOVER_CYCLE_WINDOW_MS) refreshed++;
      return {
        enabled: !offReason,
        off_reason: offReason,
        active,
        reason: statusReason(t),
        demand: demand(),
        last_demand_at: deps.lastDemandAt?.() ?? null,
        active_since: activeSince,
        activations_total: activations,
        active_s_total: Math.round((activeMsDone + (activeSince != null ? t - activeSince : 0)) / 1000),
        rps_per_provider: rps,
        requests_total: provs.reduce((n, s) => n + s.requests, 0),
        ok_total: provs.reduce((n, s) => n + s.ok, 0),
        failed_total: provs.reduce((n, s) => n + s.failed, 0),
        aircraft_last_answer: lastCount,
        order_from: orderFrom,
        order_set_at: orderSetAt,
        last,
        last_at: lastAt,
        discs_in_plan: scheduler.discs.length,
        discs_answered_ever: answeredAt.size,
        discs_refreshed_last_cycle: refreshed,
        cycle_window_s: FAILOVER_CYCLE_WINDOW_MS / 1000,
        providers: provs.map((s) => ({
          key: s.p.key, label: s.p.label, requests: s.requests, ok: s.ok, failed: s.failed,
          rate_limited: s.rate_limited, refused: s.refused, aircraft_total: s.aircraft_total,
          consecutive_failures: s.failures, backoff_s: Math.max(0, Math.ceil((s.backoffUntil - t) / 1000)),
          last_ok_at: s.last_ok_at, last_status: s.last_status, last_error: s.last_error,
        })),
        rule: FAILOVER_RULE,
      };
    },
    stop: () => {
      stopped = true;
      for (const tm of Array.from(timers)) clearTimeout(tm);
      timers.clear();
    },
  };
}
