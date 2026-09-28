// adsbGovernor.ts — ONE module-level rate governor for every request this
// process sends to adsb.lol (FLIGHT PROGRAM B1, 2026-09-28).
//
// WHY: the viewport tiling chain (aircraftTiling.fetchDiscs), the tracked
// poller, the global-scope archiver and now the GLOBAL SWEEP all leave one
// Railway IP toward the same free community API. Prod has already shown
// adsb.lol partially failing over under that pressure (globalScopes.ts
// 2026-08-11 note). Before this module nothing shared a budget: each caller
// paced itself blind to the others.
//
// TWO LANES, ONE BUDGET:
//   - FOREGROUND (viewer-driven viewport discs) is NEVER delayed or denied
//     here. Its requests and failures are only RECORDED — a viewer waiting on
//     the map always outranks the background world sweep ("never starve the
//     viewport chain").
//   - BACKGROUND (the global sweep) must acquire a token first. Three gates:
//       1. token bucket at `bgRps` (burst 1 — no bursts toward a free API);
//       2. a total ceiling: if foreground+background requests over the last
//          `windowMs` already reach `ceilingRps`, the background yields;
//       3. exponential backoff after any adsb.lol 429/5xx/network failure —
//          from EITHER lane (a viewport disc seeing adsb.lol struggle pauses
//          the sweep too), honoring Retry-After when the upstream sends it.
//     Only a BACKGROUND success clears the backoff: a lucky viewport 200 a
//     second after a sweep 429 must not re-open the floodgate.
//
// Pure + clock-injectable (every method takes `now`), so tests never sleep.

export interface GovernorOptions {
  /** background token refill rate (requests / second) */
  bgRps: number;
  /** total adsb.lol ceiling across BOTH lanes (requests / second) */
  ceilingRps: number;
  /** sliding window the ceiling is measured over (ms) */
  windowMs?: number;
  /** first backoff step (ms); doubles per consecutive failure */
  baseBackoffMs?: number;
  /** backoff ceiling (ms) */
  maxBackoffMs?: number;
}

export type Lane = "fg" | "bg";

/** Status codes that mean "the upstream is struggling — back off". 0 is our
 *  convention for a network error / timeout (no HTTP status at all). */
export function isBackoffStatus(status: number): boolean {
  return status === 0 || status === 429 || status >= 500;
}

export class UpstreamGovernor {
  readonly bgRps: number;
  readonly ceilingRps: number;
  readonly windowMs: number;
  readonly baseBackoffMs: number;
  readonly maxBackoffMs: number;
  private tokens = 1;
  private lastRefill: number | null = null;
  /** request timestamps (both lanes) inside the sliding window */
  private recent: { t: number; lane: Lane }[] = [];
  private failures = 0;
  private backoffUntil = 0;
  private lastStatus: { lane: Lane; status: number; at: number } | null = null;
  private counts = { fg: 0, bg: 0, fgFail: 0, bgFail: 0 };

  constructor(o: GovernorOptions) {
    this.bgRps = Math.max(0, o.bgRps);
    this.ceilingRps = Math.max(0.01, o.ceilingRps);
    this.windowMs = o.windowMs ?? 60_000;
    this.baseBackoffMs = o.baseBackoffMs ?? 30_000;
    this.maxBackoffMs = o.maxBackoffMs ?? 15 * 60_000;
  }

  private prune(now: number) {
    const cut = now - this.windowMs;
    let i = 0;
    while (i < this.recent.length && this.recent[i].t <= cut) i++;
    if (i) this.recent.splice(0, i);
    // hard bound (a pathological foreground storm must not grow memory)
    if (this.recent.length > 5000) this.recent.splice(0, this.recent.length - 5000);
  }

  private refill(now: number) {
    if (this.lastRefill == null) { this.lastRefill = now; return; }
    const dt = Math.max(0, now - this.lastRefill) / 1000;
    this.tokens = Math.min(1, this.tokens + dt * this.bgRps);
    this.lastRefill = now;
  }

  /** Record that a request was SENT on `lane` (foreground calls this; the
   *  background path records inside tryAcquireBackground). */
  noteRequest(lane: Lane, now: number): void {
    this.prune(now);
    this.recent.push({ t: now, lane });
    this.counts[lane]++;
  }

  /** Record a request's OUTCOME. `retryAfterSec` from the upstream's
   *  Retry-After header (when present) floors the backoff. */
  noteResult(lane: Lane, status: number, now: number, retryAfterSec?: number | null): void {
    this.lastStatus = { lane, status, at: now };
    if (isBackoffStatus(status)) {
      this.failures++;
      if (lane === "fg") this.counts.fgFail++; else this.counts.bgFail++;
      const step = Math.min(this.maxBackoffMs, this.baseBackoffMs * 2 ** (this.failures - 1));
      const ra = retryAfterSec != null && Number.isFinite(retryAfterSec) && retryAfterSec > 0
        ? Math.min(this.maxBackoffMs, retryAfterSec * 1000) : 0;
      this.backoffUntil = Math.max(this.backoffUntil, now + Math.max(step, ra));
    } else if (lane === "bg" && status >= 200 && status < 400) {
      this.failures = 0;
      this.backoffUntil = 0;
    }
  }

  /** Requests (both lanes) inside the sliding window. */
  windowCount(now: number): number {
    this.prune(now);
    return this.recent.length;
  }

  /** ms until the background lane could next acquire (0 = now). */
  backgroundWaitMs(now: number): number {
    if (this.bgRps <= 0) return Number.POSITIVE_INFINITY;
    this.refill(now);
    this.prune(now);
    let wait = Math.max(0, this.backoffUntil - now);
    if (this.tokens < 1) wait = Math.max(wait, ((1 - this.tokens) / this.bgRps) * 1000);
    const cap = this.ceilingRps * (this.windowMs / 1000);
    if (this.recent.length >= cap && this.recent.length > 0) {
      // the window frees up when its oldest entries age out
      const k = this.recent.length - Math.floor(cap) ; // entries that must expire
      const idx = Math.min(this.recent.length - 1, Math.max(0, k));
      wait = Math.max(wait, this.recent[idx].t + this.windowMs - now + 1);
    }
    return Math.ceil(wait);
  }

  /** Background lane: take a token if every gate allows, recording the
   *  request. Returns false (and consumes nothing) otherwise. */
  tryAcquireBackground(now: number): boolean {
    if (this.backgroundWaitMs(now) > 0) return false;
    this.tokens -= 1;
    this.noteRequest("bg", now);
    return true;
  }

  stats(now: number) {
    this.prune(now);
    const fgWin = this.recent.filter((r) => r.lane === "fg").length;
    return {
      bg_rps: this.bgRps,
      ceiling_rps: this.ceilingRps,
      window_s: this.windowMs / 1000,
      window_fg: fgWin,
      window_bg: this.recent.length - fgWin,
      backoff_s: Math.max(0, Math.ceil((this.backoffUntil - now) / 1000)),
      consecutive_failures: this.failures,
      totals: { ...this.counts },
      last_status: this.lastStatus,
    };
  }
}

const envNum = (v: string | undefined, dflt: number, lo: number, hi: number): number => {
  const n = parseFloat(String(v ?? ""));
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
};

/** Default background rate (req/s): conservative — see globalSweep.ts for
 *  the bandwidth math. Env GLOBAL_SWEEP_RPS (clamped 0..2). */
export const DEFAULT_GLOBAL_SWEEP_RPS = 0.4;
/** Default total ceiling across lanes (req/s). Env ADSB_TOTAL_RPS_CEILING. */
export const DEFAULT_ADSB_TOTAL_RPS_CEILING = 1.0;

export function governorFromEnv(env: NodeJS.ProcessEnv = process.env): UpstreamGovernor {
  return new UpstreamGovernor({
    bgRps: envNum(env.GLOBAL_SWEEP_RPS, DEFAULT_GLOBAL_SWEEP_RPS, 0, 2),
    ceilingRps: envNum(env.ADSB_TOTAL_RPS_CEILING, DEFAULT_ADSB_TOTAL_RPS_CEILING, 0.05, 5),
  });
}

/** THE process-wide adsb.lol governor (viewport chain + global sweep). */
export const adsbLolGovernor: UpstreamGovernor = governorFromEnv();

/** Provider key the governor accounts for (the other chain members are
 *  separate upstreams with their own limits). */
export const GOVERNED_PROVIDER_KEY = "adsblol";

/** Parse a Retry-After header value (seconds form; HTTP-date form too). */
export function parseRetryAfter(v: string | null | undefined, now: number = Date.now()): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  if (Number.isFinite(n)) return n >= 0 ? n : null;
  const t = Date.parse(v);
  return Number.isFinite(t) ? Math.max(0, (t - now) / 1000) : null;
}
