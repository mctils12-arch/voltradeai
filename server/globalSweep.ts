// globalSweep.ts — a rotating, POLITE world sweep of live aircraft through
// adsb.lol (ODbL — the only monetization-lawful free provider), feeding the
// global snapshot (FLIGHT PROGRAM B1, 2026-09-28).
//
// adsb.lol has NO all-aircraft endpoint. Two documented endpoints together
// approximate one, and the sweep runs both as LANES sharing one budget:
//
//  TYPE LANE (primary). /v2/type/{A,B,C,...} accepts a comma list and
//   answers WORLDWIDE for every aircraft of those ICAO types (probed live
//   2026-09-28: one request for 50 common types = 8,117 aircraft, 700 KB
//   compressed, 1.3 s). A seed list of common types (SEED_TYPES), batched
//   TYPES_PER_BATCH per request, each batch refreshed every
//   GLOBAL_SWEEP_TYPE_REFRESH_S (default 60s ≈ 0.08 req/s). Types observed
//   in the snapshot but missing from the list are PROMOTED into it every
//   PROMOTE_EVERY_MS (count ≥ PROMOTE_MIN, capped MAX_TYPES) — the list
//   learns the long tail from real traffic. Blind spot, stated in the
//   endpoint: aircraft with NO type in adsb.lol's database (~5% of a
//   central-US sample) never match a type query.
//
//  DISC LANE (gap-filler). The static world plan (globalDiscPlan.ts: ~780
//   250nm discs over the traffic mask) via /v2/point — catches untyped and
//   not-yet-promoted aircraft. Scheduling is value-weighted staleness:
//   score = age × sqrt(residual + K) × (viewer-interest boost), where
//   residual = aircraft in that disc's last answer the TYPE lane does not
//   cover (the only aircraft the disc adds). The square-root rule is the
//   classic optimum for minimizing weighted mean age under a fixed request
//   budget. Discs a viewport request covered recently are CREDITED as fresh
//   (the viewport chain already fetched them — reused through the fix bus,
//   never refetched); any disc older than MAX_DISC_AGE_MS jumps the queue so
//   nothing goes unvisited forever.
//
// PACING: every request takes a token from the shared adsb.lol governor
// (adsbGovernor.ts) — GLOBAL_SWEEP_RPS (default 0.4 req/s, burst 1), a
// total ceiling across viewers + sweep (viewers are never delayed; the sweep
// yields), and exponential backoff on 429/5xx/network errors from EITHER
// lane (Retry-After honored). Requests are strictly sequential. The disc
// lane additionally keeps its own slower pace, GLOBAL_SWEEP_DISC_RPS
// (default 0.15 req/s): it only fills the ~5% the type lane can't see, so
// it never gets the whole budget. Default steady state ≈ 0.08 (type) +
// 0.15 (disc) ≈ 0.23 req/s, under the 0.4 ceiling with headroom to spare.
//
// KILL SWITCH: GLOBAL_SWEEP_ENABLED=0 stops the sweep (default ON at the
// conservative rate). The snapshot keeps serving viewport + scope + OpenSky
// fixes either way.
//
// BANDWIDTH (politeness, measured sizes): type lane ≈ one full world pass
// per minute ≈ 1 MB compressed ≈ 1.4 GB/day; disc lane at 0.15 req/s ×
// ~25 KB ≈ 0.3 GB/day. Comparable to one browser left open on
// globe.adsb.lol; halve the type lane with GLOBAL_SWEEP_TYPE_REFRESH_S=120.

import { mapPointAircraft } from "./aircraftTiling";
import { adsbLolGovernor, parseRetryAfter, type UpstreamGovernor } from "./adsbGovernor";
import { publishFixes, type FixBatch } from "./aircraftFixBus";
import { haversineNm, worldPlan, type PlanDisc } from "./globalDiscPlan";
import type { BBox } from "./globalSnapshot";
import type { AircraftPoint } from "./datacoreArchive";

/** Node timers expose unref(); typed loosely because the dom lib is also loaded. */
export function unrefTimer(t: unknown): void {
  (t as { unref?: () => void } | null)?.unref?.();
}

const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

export const TYPES_PER_BATCH = 50;
export const MAX_TYPES = 400;
export const PROMOTE_MIN = 2;
export const PROMOTE_EVERY_MS = 10 * 60_000;
export const DEFAULT_TYPE_REFRESH_S = 60;
export const MAX_DISC_AGE_MS = 45 * 60_000;
/** residual prior so an empty disc still gets (rare) revisits */
export const DISC_WEIGHT_K = 4;
/** viewer-interest multiplier on a disc's score */
export const INTEREST_BOOST = 4;
/** how long a viewer bbox (from the global endpoint) stays "interesting" */
export const INTEREST_TTL_MS = 2 * 60_000;
/** how long a viewport disc credits overlapping plan discs */
export const VIEWPORT_CREDIT_WINDOW_MS = 60_000;
export const SWEEP_START_DELAY_MS = 20_000;
export const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

/** Seed ICAO type designators — the most common airframes in ADS-B traffic
 *  (airliners, regional, cargo, business jets, GA, helicopters). Order is
 *  roughly by expected global count so batch 1 carries the bulk. Unknown or
 *  rare designators cost nothing (they simply match no aircraft). */
export const SEED_TYPES: string[] = [
  // batch-1 core (probed: 8,117 aircraft worldwide in one request)
  "B738", "A320", "A321", "A20N", "A21N", "B38M", "A319", "B739", "B737", "E75L",
  "BCS3", "B77W", "B789", "B788", "A333", "A359", "B763", "A332", "B772", "E190",
  "CRJ9", "CRJ7", "DH8D", "AT76", "AT75", "AT72", "E195", "E170", "E75S", "B752",
  "B744", "B748", "A388", "A339", "A35K", "B78X", "B39M", "BCS1", "E290", "E295",
  "CRJ2", "B712", "MD11", "B77L", "A306", "C172", "PC12", "B350", "C208", "C56X",
  // narrow/regional long tail
  "A318", "A19N", "B733", "B734", "B735", "B736", "B753", "B762", "B764", "B773",
  "B778", "B779", "A310", "A338", "A342", "A343", "A345", "A346", "A3ST", "A337",
  "E135", "E145", "E45X", "E35L", "E120", "CRJ1", "CRJX", "F70", "F100", "RJ85",
  "RJ1H", "SU95", "AJ27", "C919", "MD82", "MD83", "MD88", "MD90", "B732", "DC10",
  "IL96", "T204", "A148", "AT43", "AT45", "DH8A", "DH8B", "DH8C", "SF34", "SB20",
  "JS31", "JS32", "JS41", "D328", "J328", "B190", "L410", "DHC6", "DHC7", "AN24",
  "AN26", "AN12", "IL76", "A124", "MA60", "F50", "C212", "CN35", "C295", "Y12",
  // cargo / state
  "C130", "C30J", "C17", "K35R", "A400", "C5M", "E3CF", "E3TF", "P8", "C2",
  // business jets
  "C25A", "C25B", "C25C", "C25M", "C500", "C501", "C510", "C525", "C550", "C560",
  "C650", "C680", "C68A", "C700", "C750", "CL30", "CL35", "CL60", "GL5T", "GL7T",
  "GLEX", "GLF4", "GLF5", "GLF6", "G150", "G280", "GALX", "ASTR", "FA50", "FA7X",
  "FA8X", "F900", "F2TH", "FA6X", "LJ31", "LJ35", "LJ40", "LJ45", "LJ60", "LJ75",
  "H25B", "H25C", "HA4T", "HDJT", "PC24", "E50P", "E55P", "E545", "E550", "PRM1",
  "BE40", "SF50", "EA50",
  // GA pistons / turboprops
  "P28A", "P28B", "P28R", "P28T", "P32R", "PA18", "PA24", "PA27", "PA31", "PA32",
  "PA34", "PA44", "PA46", "P46T", "C150", "C152", "C162", "C170", "C175", "C177",
  "C180", "C182", "C185", "C206", "C207", "C210", "C310", "C337", "C340", "C402",
  "C404", "C414", "C421", "C425", "C441", "C77R", "C82R", "C82S", "C82T", "M20P",
  "M20T", "SR20", "SR22", "S22T", "DA20", "DA40", "DA42", "DA62", "DV20", "BE19",
  "BE23", "BE33", "BE35", "BE36", "BE55", "BE58", "BE60", "BE76", "BE95", "BE9L",
  "BE20", "BE30", "TBM7", "TBM8", "TBM9", "TEX2", "PC6T", "PC7", "PC9", "KODI",
  "AT8T", "AT5T", "AT6T", "RV7", "RV8", "RV10", "RV14", "AA5", "TB20", "G115",
  "J3", "CH7A", "AC50", "AC11",
  // helicopters
  "EC20", "EC30", "EC35", "EC45", "EC55", "EC75", "EC25", "AS32", "AS50", "AS55",
  "AS65", "A109", "A119", "A139", "A169", "A189", "B06", "B06T", "B212", "B412",
  "B429", "B407", "B505", "BK17", "R22", "R44", "R66", "S76", "S92", "H60", "H47",
  "UH1", "H500",
];

export function chunk<T>(xs: T[], n: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < xs.length; i += n) out.push(xs.slice(i, i + n));
  return out;
}

export function typeUrl(types: string[]): string {
  return `https://api.adsb.lol/v2/type/${types.map((t) => encodeURIComponent(t)).join(",")}`;
}

export function discUrl(d: { lat: number; lon: number; radiusNm: number }): string {
  return `https://api.adsb.lol/v2/point/${d.lat.toFixed(3)}/${d.lon.toFixed(3)}/${Math.round(d.radiusNm)}`;
}

/** Type promotion: seed + observed-but-unlisted types (count ≥ min),
 *  most common first, name as tiebreak — deterministic, capped. */
export function promoteTypes(seed: string[], counts: Map<string, number>, current: string[],
                             min = PROMOTE_MIN, maxTypes = MAX_TYPES): string[] {
  const have = new Set(current);
  const extra = Array.from(counts.entries())
    .filter(([t, n]) => n >= min && !have.has(t) && /^[A-Z0-9]{2,4}$/.test(t))
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .map(([t]) => t);
  const base = current.length ? current : seed;
  return base.concat(extra).slice(0, maxTypes);
}

interface DiscState { disc: PlanDisc; lastAt: number; residual: number; fails: number; visits: number }
interface TypeBatchState { id: number; types: string[]; lastAt: number; lastCount: number; fails: number }
export type SweepJob =
  | { kind: "type"; batch: TypeBatchState }
  | { kind: "disc"; state: DiscState };

/** Is the plan disc's center within `pad` of the bbox? (interest test) */
function discTouchesBBox(d: PlanDisc, b: BBox): boolean {
  const padLat = d.radiusNm / 60;
  if (d.lat < b.lamin - padLat || d.lat > b.lamax + padLat) return false;
  const padLon = d.radiusNm / (60 * Math.max(0.1, Math.cos((d.lat * Math.PI) / 180)));
  const lo = d.lon;
  if (b.lomin <= b.lomax) return lo >= b.lomin - padLon && lo <= b.lomax + padLon;
  return lo >= b.lomin - padLon || lo <= b.lomax + padLon;
}

/** Sample points of a disc: center + 8 at 0.5r + 12 at 0.98r. */
export function discSamples(lat: number, lon: number, radiusNm: number): [number, number][] {
  const pts: [number, number][] = [[lat, lon]];
  const ring = (frac: number, n: number) => {
    for (let i = 0; i < n; i++) {
      const a = (2 * Math.PI * i) / n;
      const dLat = (radiusNm * frac * Math.cos(a)) / 60;
      const dLon = (radiusNm * frac * Math.sin(a)) / (60 * Math.max(0.05, Math.cos((lat * Math.PI) / 180)));
      pts.push([lat + dLat, lon + dLon]);
    }
  };
  ring(0.5, 8);
  ring(0.98, 12);
  return pts;
}

export class SweepScheduler {
  readonly discs: DiscState[];
  batches: TypeBatchState[] = [];
  types: string[];
  readonly typeRefreshMs: number;
  private interest: { bbox: BBox; at: number }[] = [];
  private viewportDiscs: { lat: number; lon: number; radiusNm: number; at: number }[] = [];
  creditedTotal = 0;

  constructor(plan: PlanDisc[], opts: { now: number; typeRefreshMs?: number; types?: string[] }) {
    this.typeRefreshMs = opts.typeRefreshMs ?? DEFAULT_TYPE_REFRESH_S * 1000;
    // first pass: pretend every disc was last seen long ago, weighted by
    // its mask prior — dense regions are visited first
    this.discs = plan.map((disc) => ({
      disc, lastAt: opts.now - MAX_DISC_AGE_MS / 2, residual: disc.prior, fails: 0, visits: 0,
    }));
    this.types = (opts.types ?? SEED_TYPES).slice(0, MAX_TYPES);
    this.setTypes(this.types);
  }

  setTypes(types: string[]): void {
    const prev = new Map(this.batches.map((b) => [b.types.join(","), b]));
    this.types = types.slice(0, MAX_TYPES);
    this.batches = chunk(this.types, TYPES_PER_BATCH).map((t, id) => {
      const old = prev.get(t.join(","));
      return { id, types: t, lastAt: old?.lastAt ?? 0, lastCount: old?.lastCount ?? 0, fails: old?.fails ?? 0 };
    });
  }

  noteInterest(bbox: BBox, now: number): void {
    this.interest.push({ bbox, at: now });
    this.interest = this.interest.filter((i) => now - i.at <= INTEREST_TTL_MS).slice(-32);
  }

  private inInterest(d: PlanDisc, now: number): boolean {
    for (const i of this.interest) {
      if (now - i.at <= INTEREST_TTL_MS && discTouchesBBox(d, i.bbox)) return true;
    }
    return false;
  }

  /** A viewport disc was just fetched: credit plan discs the recent
   *  viewport discs fully cover (every sample point inside one). */
  creditViewportDisc(v: { lat: number; lon: number; radiusNm: number }, now: number): number {
    this.viewportDiscs.push({ ...v, at: now });
    this.viewportDiscs = this.viewportDiscs.filter((x) => now - x.at <= VIEWPORT_CREDIT_WINDOW_MS).slice(-64);
    let credited = 0;
    for (const s of this.discs) {
      const d = s.disc;
      if (haversineNm(d.lat, d.lon, v.lat, v.lon) > d.radiusNm + v.radiusNm) continue;
      const covered = discSamples(d.lat, d.lon, d.radiusNm).every(([la, lo]) =>
        this.viewportDiscs.some((x) => haversineNm(la, lo, x.lat, x.lon) <= x.radiusNm));
      if (covered && s.lastAt < now) { s.lastAt = now; credited++; }
    }
    this.creditedTotal += credited;
    return credited;
  }

  discScore(s: DiscState, now: number): number {
    const age = Math.max(0, now - s.lastAt);
    let score = age * Math.sqrt(Math.max(0, s.residual) + DISC_WEIGHT_K);
    if (this.inInterest(s.disc, now)) score *= INTEREST_BOOST;
    if (age > MAX_DISC_AGE_MS) score *= 1000; // nothing goes unvisited forever
    return score;
  }

  /** The next job: an overdue type batch first (cheap, global), else the
   *  highest-scoring disc. */
  next(now: number): SweepJob | null {
    let due: TypeBatchState | null = null;
    for (const b of this.batches) {
      if (now - b.lastAt >= this.typeRefreshMs && (!due || b.lastAt < due.lastAt)) due = b;
    }
    if (due) return { kind: "type", batch: due };
    let best: DiscState | null = null;
    let bestScore = -1;
    for (const s of this.discs) {
      const sc = this.discScore(s, now);
      if (sc > bestScore) { bestScore = sc; best = s; }
    }
    return best ? { kind: "disc", state: best } : null;
  }

  markType(b: TypeBatchState, now: number, count: number, ok: boolean): void {
    b.lastAt = now;
    if (ok) { b.lastCount = count; b.fails = 0; } else b.fails++;
  }

  markDisc(s: DiscState, now: number, residual: number, ok: boolean): void {
    s.lastAt = now;
    s.visits++;
    if (ok) { s.residual = residual; s.fails = 0; } else s.fails++;
  }

  /** honest coverage numbers for the endpoint */
  coverage(now: number, cycleWindowMs = 10 * 60_000) {
    const ages = this.discs.map((s) => now - s.lastAt).sort((a, b) => a - b);
    const visited = this.discs.filter((s) => s.visits > 0).length;
    const oldestBatch = this.batches.reduce((m, b) => Math.max(m, b.lastAt ? now - b.lastAt : Infinity), 0);
    return {
      discs_in_plan: this.discs.length,
      discs_visited_ever: visited,
      discs_refreshed_last_cycle: ages.filter((a) => a <= cycleWindowMs).length,
      cycle_window_s: cycleWindowMs / 1000,
      oldest_disc_age_s: ages.length ? Math.round(ages[ages.length - 1] / 1000) : null,
      median_disc_age_s: ages.length ? Math.round(ages[Math.floor(ages.length / 2)] / 1000) : null,
      viewport_credited_total: this.creditedTotal,
      type_lane: {
        types: this.types.length,
        batches: this.batches.length,
        refresh_s: this.typeRefreshMs / 1000,
        oldest_batch_age_s: Number.isFinite(oldestBatch) ? Math.round(oldestBatch / 1000) : null,
        aircraft_last_pass: this.batches.reduce((n, b) => n + b.lastCount, 0),
      },
    };
  }
}

export interface SweepDeps {
  fetchImpl: typeof fetch;
  governor: UpstreamGovernor;
  publish?: (b: FixBatch) => void;
  now?: () => number;
  timeoutMs?: number;
}

export interface StepResult {
  kind: "type" | "disc";
  ok: boolean;
  status: number;
  count: number;
  residual?: number;
  error?: string;
}

/**
 * Execute ONE sweep job (the caller already holds a governor token):
 * fetch → map through the SAME normalizer as every aircraft path → publish
 * on the fix bus (snapshot + archive subscribe) → mark the scheduler.
 */
export async function runSweepJob(job: SweepJob, sched: SweepScheduler, deps: SweepDeps): Promise<StepResult> {
  const now = deps.now ?? Date.now;
  const publish = deps.publish ?? publishFixes;
  const url = job.kind === "type" ? typeUrl(job.batch.types) : discUrl(job.state.disc);
  let status = 0;
  try {
    const r = await deps.fetchImpl(url, { headers: UA, signal: AbortSignal.timeout(deps.timeoutMs ?? 20_000) });
    status = r.status ?? (r.ok ? 200 : 0);
    if (!r.ok) {
      deps.governor.noteResult("bg", status, now(), parseRetryAfter(r.headers?.get?.("retry-after") ?? null));
      if (job.kind === "type") sched.markType(job.batch, now(), 0, false);
      else sched.markDisc(job.state, now(), job.state.residual, false);
      return { kind: job.kind, ok: false, status, count: 0, error: `adsblol ${status}` };
    }
    const raw: unknown = await r.json();
    deps.governor.noteResult("bg", 200, now());
    const aircraft = mapPointAircraft(raw, "ac", "adsblol") as AircraftPoint[];
    const fetchedAt = now();
    const nowField = (raw as { now?: unknown } | null)?.now;
    const upstreamNowMs = typeof nowField === "number" && Number.isFinite(nowField) ? nowField : null;
    if (job.kind === "type") {
      publish({ provider: "adsblol", origin: "sweep-type", aircraft, fetchedAt, upstreamNowMs });
      sched.markType(job.batch, fetchedAt, aircraft.length, true);
      return { kind: "type", ok: true, status: 200, count: aircraft.length };
    }
    const typeSet = new Set(sched.types);
    const residual = aircraft.filter((a) => !a.type || !typeSet.has(a.type)).length;
    publish({
      provider: "adsblol", origin: "sweep-disc", aircraft, fetchedAt, upstreamNowMs,
      disc: { lat: job.state.disc.lat, lon: job.state.disc.lon, radiusNm: job.state.disc.radiusNm },
    });
    sched.markDisc(job.state, fetchedAt, residual, true);
    return { kind: "disc", ok: true, status: 200, count: aircraft.length, residual };
  } catch (e) {
    if (status === 0) deps.governor.noteResult("bg", 0, now());
    if (job.kind === "type") sched.markType(job.batch, now(), 0, false);
    else sched.markDisc(job.state, now(), job.state.residual, false);
    return { kind: job.kind, ok: false, status, count: 0, error: errMsg(e) };
  }
}

export function sweepEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = String(env.GLOBAL_SWEEP_ENABLED ?? "1").trim().toLowerCase();
  return !(v === "0" || v === "false" || v === "off" || v === "no");
}

/** Disc-lane pace (req/s) — its own, lower share of the governor budget:
 *  with the type lane covering ~95% of traffic the disc lane only fills the
 *  untyped/unlisted gap, so spending the full governor rate on it would be
 *  impolite for little value. Env GLOBAL_SWEEP_DISC_RPS (0 = disc lane off). */
export const DEFAULT_DISC_RPS = 0.15;
export function discMinGapMsFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const n = parseFloat(String(env.GLOBAL_SWEEP_DISC_RPS ?? ""));
  const rps = Number.isFinite(n) ? Math.min(2, Math.max(0, n)) : DEFAULT_DISC_RPS;
  return rps > 0 ? Math.ceil(1000 / rps) : Number.POSITIVE_INFINITY;
}

export function typeRefreshMsFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const n = parseFloat(String(env.GLOBAL_SWEEP_TYPE_REFRESH_S ?? ""));
  return (Number.isFinite(n) ? Math.min(3600, Math.max(20, n)) : DEFAULT_TYPE_REFRESH_S) * 1000;
}

export type SweepStatus = {
  enabled: boolean;
  disc_rps: number;
  steps: number;
  errors: number;
  promote_errors: number;
  last: StepResult | null;
  last_at: number | null;
  governor: ReturnType<UpstreamGovernor["stats"]>;
} & ReturnType<SweepScheduler["coverage"]>;

export interface SweepHandle {
  scheduler: SweepScheduler;
  status: () => SweepStatus;
  stop: () => void;
}

/** Boot the sweep loop (sequential, governor-paced, unref'd timers). */
export function startGlobalSweep(deps: {
  fetchImpl?: typeof fetch;
  governor?: UpstreamGovernor;
  env?: NodeJS.ProcessEnv;
  plan?: PlanDisc[];
  typeCounts?: () => Map<string, number>;
  startDelayMs?: number;
  /** initial type list (default SEED_TYPES) */
  types?: string[];
}): SweepHandle {
  const env = deps.env ?? process.env;
  const governor = deps.governor ?? adsbLolGovernor;
  const fetchImpl = deps.fetchImpl ?? fetch;
  const enabled = sweepEnabled(env);
  const scheduler = new SweepScheduler(deps.plan ?? worldPlan(), {
    now: Date.now(), typeRefreshMs: typeRefreshMsFromEnv(env), types: deps.types,
  });
  let stopped = false;
  let last: StepResult | null = null;
  let lastAt = 0;
  let steps = 0, errors = 0, promoteErrors = 0;
  let lastPromote = Date.now();
  let lastDiscAt = 0;
  const discMinGapMs = discMinGapMsFromEnv(env);
  let lastErrLog = 0;
  const timers = new Set<ReturnType<typeof setTimeout>>();
  const sleep = (ms: number) => new Promise<void>((res) => {
    const t = setTimeout(() => { timers.delete(t); res(); }, ms);
    unrefTimer(t);
    timers.add(t);
  });

  const loop = async () => {
    await sleep(deps.startDelayMs ?? SWEEP_START_DELAY_MS);
    while (!stopped) {
      const now = Date.now();
      if (deps.typeCounts && now - lastPromote >= PROMOTE_EVERY_MS) {
        lastPromote = now;
        try {
          const next = promoteTypes(SEED_TYPES, deps.typeCounts(), scheduler.types);
          if (next.length !== scheduler.types.length) scheduler.setTypes(next);
        } catch (e) {
          promoteErrors++; // the seed list keeps working; surfaced in status()
          console.error("[global-sweep] type promotion failed:", errMsg(e));
        }
      }
      const job = scheduler.next(now);
      if (!job) { await sleep(5_000); continue; }
      if (job.kind === "disc") {
        // the disc lane keeps its own slower pace (type batches never wait on it)
        const gap = discMinGapMs - (now - lastDiscAt);
        if (gap > 0) { await sleep(Math.min(Math.max(gap, 50), 5_000)); continue; }
      }
      const wait = governor.backgroundWaitMs(now);
      if (wait > 0) { await sleep(Math.min(Math.max(wait, 50), 10_000)); continue; }
      if (!governor.tryAcquireBackground(now)) { await sleep(250); continue; }
      if (job.kind === "disc") lastDiscAt = now;
      try {
        last = await runSweepJob(job, scheduler, { fetchImpl, governor });
      } catch (e) {
        last = { kind: job.kind, ok: false, status: 0, count: 0, error: errMsg(e) };
      }
      lastAt = Date.now();
      steps++;
      if (!last.ok) {
        errors++;
        // loud but throttled (Law V: silent degradation is how staleness recurs)
        if (lastAt - lastErrLog > 60_000) {
          lastErrLog = lastAt;
          console.error(`[global-sweep] ${last.kind} request failed (${last.error}) — governor backoff ${governor.stats(lastAt).backoff_s}s`);
        }
      }
    }
  };
  if (enabled) void loop();
  return {
    scheduler,
    status: () => ({
      enabled,
      disc_rps: Number.isFinite(discMinGapMs) ? Math.round((1000 / discMinGapMs) * 1000) / 1000 : 0,
      steps, errors, promote_errors: promoteErrors,
      last, last_at: lastAt || null,
      governor: governor.stats(Date.now()),
      ...scheduler.coverage(Date.now()),
    }),
    stop: () => {
      stopped = true;
      for (const t of Array.from(timers)) clearTimeout(t);
      timers.clear();
    },
  };
}
