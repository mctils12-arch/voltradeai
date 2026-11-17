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
//       3. backoff after an adsb.lol 429/5xx/network failure, honoring
//          Retry-After when the upstream sends it:
//          - a BACKGROUND failure starts the sweep's own exponential backoff
//            (30 s doubling to 15 min), which only a BACKGROUND success
//            clears: a lucky viewport 200 a second after a sweep 429 must not
//            re-open the floodgate;
//          - a FOREGROUND failure (a viewport disc, the selected-aircraft
//            lane) pauses the sweep too, but only briefly (FG_FAILURE_PAUSE_MS)
//            and never more than FG_PAUSE_MAX_MS past the sweep's own last
//            request. Then ONE sweep request goes out and its own outcome
//            decides.
//          Until 2026-11-16 a foreground failure fed the same exponential
//          counter. Live that day: 127 consecutive failures, all but a few
//          from the foreground lane, held the pause at its 15-minute ceiling
//          while adsb.lol was answering again. Only a sweep success could
//          clear it and the sweep was not allowed to send, so it sent nothing
//          for 24+ minutes and the worldwide snapshot aged out (1 row of
//          ~1,100 under 2 minutes old; rows are evicted at 10 minutes).
//
// THE DAILY QUOTA (2026-11-17). api.adsb.lol counts requests per day and per
// IP: without a key 500 a day. Every response says so:
//   x-ratelimit-limit: 500, 500;w=86400
//   x-ratelimit-remaining: <n>
//   x-ratelimit-reset: <seconds until 00:00 UTC>
// Over the quota it answers 429 with `x-envoy-ratelimited: true`, an empty
// body and NO Retry-After. Measured that day: this server sent about 29,500
// requests a day (type lane 11,520, disc lane 12,960, tracked poller 2,880,
// mil/ladd/pia 2,160, plus viewers), so the quota was gone about 25 minutes
// after midnight UTC and every lane then drew 429 until the next midnight,
// with the backoff above re-probing every 15 minutes for a Retry-After that
// never comes. A key (header x-api-key, env ADSBLOL_API_KEY) raises the
// limit: 1,000/day for the free "adhoc" key, 7,200/day for a feeder key,
// more for paid tiers.
//
// So the governor now READS the quota (noteQuota, on every response) and
// decides who may ask (laneAllowed / admit):
//   - EXHAUSTED (remaining 0, or the quota 429): nobody asks until the reset.
//     A quota 429 is not an outage: it grows no failure backoff, cuts no
//     AIMD rate and does not count as a sweep failure.
//     Only remaining 0 or the quota's own 429 (x-envoy-ratelimited) says so:
//     a 429 with no quota information is an ordinary rate limit and goes
//     through the failure backoff as before (a stray 429 must not silence
//     adsb.lol until midnight).
//   - WATCHED_ONLY (limit under QUOTA_FULL_MIN_DAILY): only the tracked-planes
//     poller, a Find lookup and the LADD scope ask adsb.lol. The tracked
//     poller paces itself to TRACKED_QUOTA_SHARE of the limit; the LADD scope
//     is asked once per 15 minutes (globalScopes.ts: the privacy registry
//     must keep learning). The world sweep, the mil/pia scopes, the viewport
//     discs and the selected-plane lane do not ask at all (the other
//     providers carry them: sweepFailover.ts, the viewport chain, the fast
//     lane's second provider).
//   - FULL (limit at or over QUOTA_FULL_MIN_DAILY): the paced lanes share the
//     limit — world sweep SWEEP_QUOTA_SHARE (0.60), tracked poller
//     TRACKED_QUOTA_SHARE_FULL (0.25), scopes SCOPES_QUOTA_SHARE (0.10) — and
//     the rest is left for viewers, the selected plane and Find. The disc
//     lane asks only from DISC_LANE_MIN_DAILY.
//   - UNKNOWN (no quota header seen yet): as before.
//
// Pure + clock-injectable (every method takes `now`), so tests never sleep.

/** adsb.lol's daily limit at or over which the world sweep may use it */
export const QUOTA_FULL_MIN_DAILY = 5000;
/** share of the daily limit the world sweep may spend ("full" mode) */
export const SWEEP_QUOTA_SHARE = 0.6;
/** share of the daily limit the tracked-planes poller may spend ("watched_only" mode) */
export const TRACKED_QUOTA_SHARE = 0.4;
/** ... and in "full" mode, where the sweep and the scopes share the limit with it */
export const TRACKED_QUOTA_SHARE_FULL = 0.25;
/** share of the daily limit the mil/ladd/pia scopes may spend ("full" mode) */
export const SCOPES_QUOTA_SHARE = 0.1;
/** the sweep's disc lane asks adsb.lol only from this daily limit */
export const DISC_LANE_MIN_DAILY = 20000;
/** how long the suspended sweep sleeps between looks at the quota mode */
export const QUOTA_RECHECK_MS = 60_000;
const DAY_MS = 86_400_000;

export type QuotaMode = "unknown" | "watched_only" | "full";
/** who is asking adsb.lol (see laneAllowed) */
export type QuotaLane = "sweep" | "scopes" | "ladd" | "viewport" | "selected" | "tracked" | "find";
/** anything with a Headers-like get (a fetch Response's headers) */
export interface HeaderSource { get(name: string): string | null }

export interface QuotaHeaders {
  /** requests per day (the first integer of x-ratelimit-limit) */
  limit: number | null;
  remaining: number | null;
  /** seconds until the quota resets (00:00 UTC) */
  resetSec: number | null;
  /** x-envoy-ratelimited: true — the 429 is the quota's */
  ratelimited: boolean;
}

/** Pure: read adsb.lol's quota headers. A header that is absent or not a
 *  number reads null; nothing is guessed. */
export function parseQuotaHeaders(get: (name: string) => string | null): QuotaHeaders {
  const read = (name: string): string => {
    try { return String(get(name) ?? "").trim(); } catch { return ""; }
  };
  const lim = /^(\d{1,9})(?!\d)/.exec(read("x-ratelimit-limit"));
  const limit = lim && Number(lim[1]) > 0 ? Number(lim[1]) : null;
  const rem = /^-?\d{1,9}$/.exec(read("x-ratelimit-remaining"));
  const remaining = rem ? Number(rem[0]) : null;
  const rs = /^\d{1,6}(\.\d+)?$/.exec(read("x-ratelimit-reset"));
  // the window is one day: a figure past two days is not a reset time
  const resetSec = rs && Number(rs[0]) <= 2 * 86_400 ? Number(rs[0]) : null;
  return { limit, remaining, resetSec, ratelimited: read("x-envoy-ratelimited").toLowerCase() === "true" };
}

/** The User-Agent every request to adsb.lol carries. */
export const ADSB_LOL_UA = "voltradeai-datacore/1.0 (+https://voltradeai.com)";

/** `{ "x-api-key": <key> }` when ADSBLOL_API_KEY is set, else `{}`. For
 *  api.adsb.lol ONLY: never merge this into a request to another host. The
 *  key is never logged and never put in a status. */
export function adsbLolKeyHeader(env: NodeJS.ProcessEnv = process.env): Record<string, string> {
  const key = String(env.ADSBLOL_API_KEY ?? "").trim();
  return key ? { "x-api-key": key } : {};
}

/** The headers of a request to api.adsb.lol. */
export function adsbLolRequestHeaders(env: NodeJS.ProcessEnv = process.env): Record<string, string> {
  return { "User-Agent": ADSB_LOL_UA, ...adsbLolKeyHeader(env) };
}

/** ADSBLOL_QUOTA_MODE=full|watched_only (tests / ops); anything else = read the header. */
export function quotaModeOverrideFromEnv(env: NodeJS.ProcessEnv = process.env): "full" | "watched_only" | null {
  const v = String(env.ADSBLOL_QUOTA_MODE ?? "").trim().toLowerCase();
  return v === "full" || v === "watched_only" ? v : null;
}

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
  /** fixes the quota mode whatever the limit header says (env ADSBLOL_QUOTA_MODE) */
  quotaModeOverride?: "full" | "watched_only" | null;
  /** is an API key sent with each request? (a boolean for the status; the key itself never comes here) */
  keyConfigured?: boolean;
  /** where state changes are said (the process singleton logs; tests stay quiet) */
  log?: (msg: string) => void;
}

export type Lane = "fg" | "bg";

/** How an outcome is counted for the ops check (429 = adsb.lol rate-limited
 *  us; 5xx = it struggled; net = timeout / no response; other = any other
 *  non-2xx/3xx such as 403/404). The DAILY QUOTA's 429 is counted as a 429
 *  here too; what sets it apart is in noteResult. */
export type OutcomeKind = "ok" | "429" | "5xx" | "net" | "other";
export function outcomeKind(status: number): OutcomeKind {
  if (status === 0) return "net";
  if (status === 429) return "429";
  if (status >= 500) return "5xx";
  if (status >= 200 && status < 400) return "ok";
  return "other";
}
/** AIMD on the BACKGROUND lane (ops auto-fix 2026-10-14): live, the sweep
 *  saw 38 failures in 93 requests, mostly 429 — the backoff paused it after
 *  each 429, but the first success snapped straight back to the full rate
 *  and drew the next 429. Now each background 429 halves the effective rate
 *  (floor AIMD_MIN_FRACTION of bgRps) and each background success adds
 *  AIMD_STEP_FRACTION back (cap: bgRps). Only 429 — "you are too fast" —
 *  cuts the rate; 5xx/network are the upstream's trouble and stay with the
 *  backoff. The foreground lane is never paced (viewers first). */
export const AIMD_MIN_FRACTION = 0.1;
export const AIMD_STEP_FRACTION = 0.05;

/** A FOREGROUND failure pauses the sweep this long (the sweep's own first
 *  backoff step) ... */
export const FG_FAILURE_PAUSE_MS = 30_000;
/** ... and foreground failures alone never hold the sweep longer than this
 *  past its own last request (or past the first foreground failure since).
 *  After it, one sweep request goes out as the probe. During an outage that
 *  probe fails and the sweep's exponential backoff takes over, so the upstream
 *  sees one probe, not a resumed sweep. */
export const FG_PAUSE_MAX_MS = 120_000;

/** the ops window: outcomes kept for one hour (bounded) */
export const OUTCOME_WINDOW_MS = 3600_000;
const OUTCOME_LOG_MAX = 20_000;

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
  /** AIMD multiplier on bgRps (1 = full configured rate) */
  private rateFrac = 1;
  private lastRefill: number | null = null;
  /** request timestamps (both lanes) inside the sliding window */
  private recent: { t: number; lane: Lane }[] = [];
  /** consecutive failures from EITHER lane since the last sweep success (the ops figure) */
  private failures = 0;
  /** consecutive SWEEP failures: the exponent of the sweep's own backoff */
  private bgFailures = 0;
  /** the sweep's own backoff (set by background failures only) */
  private backoffUntil = 0;
  /** the bounded pause a foreground failure asks for */
  private fgPauseUntil = 0;
  /** first foreground failure since the sweep last sent a request */
  private fgRunStart: number | null = null;
  private lastBgRequestAt: number | null = null;
  private lastStatus: { lane: Lane; status: number; at: number } | null = null;
  private counts = { fg: 0, bg: 0, fgFail: 0, bgFail: 0 };
  /** outcomes of the last OUTCOME_WINDOW_MS, oldest first (ops check: the
   *  429 rate and the viewport failover rate) */
  private outcomes: { t: number; lane: Lane; kind: OutcomeKind }[] = [];
  // ── the daily quota, as adsb.lol's own response headers state it ──
  private quotaLimit: number | null = null;
  private quotaRemaining: number | null = null;
  /** when the quota resets (ms), from the newest x-ratelimit-reset */
  private quotaResetAt: number | null = null;
  /** nobody asks adsb.lol before this (0 = not exhausted) */
  private quotaExhaustedUntilMs = 0;
  private quotaExhaustedTotal = 0;
  private requestsSkippedQuota = 0;
  private readonly quotaModeOverride: "full" | "watched_only" | null;
  private readonly keyConfigured: boolean;
  private readonly log: ((msg: string) => void) | null;

  constructor(o: GovernorOptions) {
    this.bgRps = Math.max(0, o.bgRps);
    this.ceilingRps = Math.max(0.01, o.ceilingRps);
    this.windowMs = o.windowMs ?? 60_000;
    this.baseBackoffMs = o.baseBackoffMs ?? 30_000;
    this.maxBackoffMs = o.maxBackoffMs ?? 15 * 60_000;
    this.quotaModeOverride = o.quotaModeOverride ?? null;
    this.keyConfigured = !!o.keyConfigured;
    this.log = o.log ?? null;
  }

  // ── daily quota ────────────────────────────────────────────────────────

  /** Read one adsb.lol response's quota headers (call it for EVERY response,
   *  BEFORE noteResult). Absent or unreadable headers leave the state as it
   *  was. Returns true when this response says the quota is spent:
   *    - x-ratelimit-remaining <= 0 (on any status), or
   *    - a 429 with x-envoy-ratelimited: true, or
   *    - a 429 while the last remaining figure read (in its window) was 0.
   *  Then nobody asks until the reset: now + x-ratelimit-reset, or the next
   *  00:00 UTC when the response gave none. A 429 with NO quota information
   *  is not the quota: it returns false and the caller's noteResult treats it
   *  as an ordinary failure. */
  noteQuota(headers: HeaderSource | null | undefined, now: number, status?: number): boolean {
    if (!headers || typeof headers.get !== "function") return false;
    const q = parseQuotaHeaders((name) => headers.get(name));
    if (q.limit != null) this.quotaLimit = q.limit;
    if (q.remaining != null) this.quotaRemaining = q.remaining;
    if (q.resetSec != null) this.quotaResetAt = now + q.resetSec * 1000;
    let spent = q.remaining != null && q.remaining <= 0;
    if (!spent && status === 429) {
      const lastRead = q.remaining ?? this.currentRemaining(now);
      spent = q.ratelimited || (lastRead != null && lastRead <= 0);
    }
    if (!spent) return false;
    const nextUtcMidnight = (Math.floor(now / DAY_MS) + 1) * DAY_MS;
    // a reset "0 s away" still means this request was refused: hold one second
    const until = q.resetSec != null ? now + Math.max(1_000, q.resetSec * 1000) : nextUtcMidnight;
    const was = this.quotaBlocked(now);
    this.quotaExhaustedUntilMs = Math.max(this.quotaExhaustedUntilMs, until);
    this.quotaRemaining = 0;
    if (q.resetSec == null) this.quotaResetAt = this.quotaExhaustedUntilMs;
    if (!was) {
      this.quotaExhaustedTotal++;
      this.log?.(`[adsb-governor] adsb.lol daily quota spent (limit ${this.quotaLimit ?? "unknown"}/day) — no request to adsb.lol until ${new Date(this.quotaExhaustedUntilMs).toISOString()}`);
    }
    return true;
  }

  /** the last remaining figure read, while its window still runs */
  private currentRemaining(now: number): number | null {
    if (this.quotaResetAt != null && now >= this.quotaResetAt) return null;
    return this.quotaRemaining;
  }

  /** Is the daily quota spent? While it is, NO lane asks adsb.lol. */
  quotaBlocked(now: number): boolean {
    return now < this.quotaExhaustedUntilMs;
  }

  /** when the spent quota resets (ms), null while it is not spent */
  quotaExhaustedUntil(now: number): number | null {
    return this.quotaBlocked(now) ? this.quotaExhaustedUntilMs : null;
  }

  /** "unknown" until a response has stated the limit; then by the limit
   *  (under QUOTA_FULL_MIN_DAILY = "watched_only"). ADSBLOL_QUOTA_MODE fixes it. */
  quotaMode(_now: number): QuotaMode {
    if (this.quotaModeOverride) return this.quotaModeOverride;
    if (this.quotaLimit == null) return "unknown";
    return this.quotaLimit >= QUOTA_FULL_MIN_DAILY ? "full" : "watched_only";
  }

  /** Why the world sweep (the background lane) is kept off adsb.lol by the
   *  quota, or null when the quota does not stop it. */
  sweepSuspendedByQuota(now: number): "exhausted" | "watched_only" | null {
    if (this.quotaBlocked(now)) return "exhausted";
    return this.quotaMode(now) === "watched_only" ? "watched_only" : null;
  }

  /** Pure: may `lane` send a request to adsb.lol now, as far as the daily
   *  quota goes? Spent: nobody. "watched_only": the tracked poller, a Find
   *  lookup and the LADD scope (which paces itself, globalScopes.ts).
   *  Otherwise everyone. */
  laneAllowed(lane: QuotaLane, now: number): boolean {
    if (this.quotaBlocked(now)) return false;
    if (this.quotaMode(now) === "watched_only") return lane === "tracked" || lane === "find" || lane === "ladd";
    return true;
  }

  /** laneAllowed, and a refusal is counted as `requests` requests NOT sent
   *  (requests_skipped_quota). Call it where a request would otherwise go out. */
  admit(lane: QuotaLane, now: number, requests = 1): boolean {
    if (this.laneAllowed(lane, now)) return true;
    this.requestsSkippedQuota += Math.max(0, requests);
    return false;
  }

  /** The world sweep's smallest gap between two requests (ms) so a day of it
   *  uses at most SWEEP_QUOTA_SHARE of the limit; 0 while the limit is unknown. */
  sweepMinGapMs(_now: number): number {
    return this.quotaLimit == null ? 0 : DAY_MS / (this.quotaLimit * SWEEP_QUOTA_SHARE);
  }

  /** May the sweep's disc lane ask adsb.lol? Unknown limit: yes, as before. */
  discLaneAllowed(now: number): boolean {
    if (this.sweepSuspendedByQuota(now)) return false;
    return this.quotaLimit == null || this.quotaLimit >= DISC_LANE_MIN_DAILY;
  }

  /** The tracked poller's smallest gap between two requests (ms) so a day of
   *  it uses at most its share of the limit: TRACKED_QUOTA_SHARE in
   *  "watched_only" (500/day -> one request per 432 s), TRACKED_QUOTA_SHARE_FULL
   *  in "full" (7,200/day -> 48 s). 0 while the limit is unknown. */
  trackedMinGapMs(now: number): number {
    const mode = this.quotaMode(now);
    if (this.quotaLimit == null || mode === "unknown") return 0;
    return Math.ceil(DAY_MS / (this.quotaLimit * (mode === "watched_only" ? TRACKED_QUOTA_SHARE : TRACKED_QUOTA_SHARE_FULL)));
  }

  /** "full": the scopes poller's smallest gap between two requests (ms) so a
   *  day of it uses at most SCOPES_QUOTA_SHARE of the limit (7,200/day -> one
   *  request per 120 s, a three-scope cycle every 360 s). 0 = not paced by a
   *  share (unknown limit; "watched_only" has its own rule, globalScopes.ts). */
  scopesMinGapMs(now: number): number {
    if (this.quotaLimit == null || this.quotaMode(now) !== "full") return 0;
    return Math.ceil(DAY_MS / (this.quotaLimit * SCOPES_QUOTA_SHARE));
  }

  /** The quota block of stats(). `key_configured` is a boolean: the key
   *  itself is never held here. */
  quotaStats(now: number) {
    const exhausted = this.quotaBlocked(now);
    return {
      limit: this.quotaLimit,
      remaining: exhausted ? 0 : this.currentRemaining(now),
      reset_at: this.quotaResetAt != null && now < this.quotaResetAt ? this.quotaResetAt : null,
      exhausted,
      exhausted_until: exhausted ? this.quotaExhaustedUntilMs : null,
      mode: this.quotaMode(now),
      key_configured: this.keyConfigured,
      sweep_min_gap_ms: Math.round(this.sweepMinGapMs(now)),
      // times the quota was found spent since boot
      quota_exhausted_total: this.quotaExhaustedTotal,
      // requests a lane did NOT send because of the quota (viewport discs,
      // selected-plane lookups, Find lookups, tracked cycles, scope requests;
      // the suspended sweep has no fixed request count and is not counted)
      requests_skipped_quota: this.requestsSkippedQuota,
    };
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
    this.tokens = Math.min(1, this.tokens + dt * this.effectiveBgRps());
    this.lastRefill = now;
  }

  /** Record that a request was SENT on `lane` (foreground calls this; the
   *  background path records inside tryAcquireBackground). */
  noteRequest(lane: Lane, now: number): void {
    this.prune(now);
    this.recent.push({ t: now, lane });
    this.counts[lane]++;
    if (lane === "bg") this.lastBgRequestAt = now;
  }

  /** Record a request's OUTCOME. `retryAfterSec` from the upstream's
   *  Retry-After header (when present) floors the backoff. */
  noteResult(lane: Lane, status: number, now: number, retryAfterSec?: number | null): void {
    this.lastStatus = { lane, status, at: now };
    if (status === 429 && this.quotaBlocked(now)) {
      // the DAILY QUOTA's 429 (noteQuota read it just before): a failed
      // request, counted, but not an outage — no backoff grows, the AIMD rate
      // is not cut and the sweep's failure run does not move. The hold is the
      // quota's own, until its reset. (In the last-hour window it is one more
      // rate_limited result.)
      this.outcomes.push({ t: now, lane, kind: "429" });
      this.pruneOutcomes(now);
      if (lane === "fg") this.counts.fgFail++; else this.counts.bgFail++;
      return;
    }
    this.outcomes.push({ t: now, lane, kind: outcomeKind(status) });
    this.pruneOutcomes(now);
    if (lane === "bg" && status === 429) this.rateFrac = Math.max(AIMD_MIN_FRACTION, this.rateFrac / 2);
    else if (lane === "bg" && status >= 200 && status < 400) this.rateFrac = Math.min(1, this.rateFrac + AIMD_STEP_FRACTION);
    if (isBackoffStatus(status)) {
      this.failures++;
      const ra = retryAfterSec != null && Number.isFinite(retryAfterSec) && retryAfterSec > 0
        ? Math.min(this.maxBackoffMs, retryAfterSec * 1000) : 0;
      if (lane === "fg") {
        this.counts.fgFail++;
        // a new run of foreground failures starts once the sweep has sent since the last one began
        if (this.fgRunStart == null || this.fgRunStart < (this.lastBgRequestAt ?? 0)) this.fgRunStart = now;
        const bounded = Math.min(now + FG_FAILURE_PAUSE_MS, this.fgRunStart + FG_PAUSE_MAX_MS);
        // an explicit Retry-After is the upstream's own instruction: honored in full
        this.fgPauseUntil = Math.max(this.fgPauseUntil, bounded, ra > 0 ? now + ra : 0);
      } else {
        this.counts.bgFail++;
        this.bgFailures++;
        const step = Math.min(this.maxBackoffMs, this.baseBackoffMs * 2 ** (this.bgFailures - 1));
        this.backoffUntil = Math.max(this.backoffUntil, now + Math.max(step, ra));
      }
    } else if (lane === "bg" && status >= 200 && status < 400) {
      this.failures = 0;
      this.bgFailures = 0;
      this.backoffUntil = 0;
      this.fgPauseUntil = 0;
    }
  }

  /** the background rate in force after AIMD (req/s) */
  effectiveBgRps(): number {
    return this.bgRps * this.rateFrac;
  }

  private pruneOutcomes(now: number) {
    const cut = now - OUTCOME_WINDOW_MS;
    let i = 0;
    while (i < this.outcomes.length && this.outcomes[i].t <= cut) i++;
    if (i) this.outcomes.splice(0, i);
    if (this.outcomes.length > OUTCOME_LOG_MAX) this.outcomes.splice(0, this.outcomes.length - OUTCOME_LOG_MAX);
  }

  /** Per-lane outcome counts over the last hour. A foreground failure is a
   *  viewport disc that FAILED OVER to the next provider in the chain. */
  lastHour(now: number) {
    this.pruneOutcomes(now);
    const blank = () => ({ results: 0, ok: 0, rate_limited: 0, server_error: 0, network_error: 0, other: 0 });
    const out = { fg: blank(), bg: blank() };
    const key: Record<OutcomeKind, keyof ReturnType<typeof blank>> = {
      ok: "ok", "429": "rate_limited", "5xx": "server_error", net: "network_error", other: "other",
    };
    for (const o of this.outcomes) { out[o.lane].results++; out[o.lane][key[o.kind]]++; }
    return out;
  }

  /** ms the sweep is held for (0 = not held): its own backoff or the bounded
   *  pause a foreground failure asked for, whichever ends later. Read-only: a
   *  caller that records on the foreground lane but can afford to wait (the
   *  tracked planes' hex poll) holds its request for the same time, so its
   *  own 429 pauses it for FG_FAILURE_PAUSE_MS and it never outruns the sweep
   *  while adsb.lol is struggling. A spent daily quota holds it too, until
   *  the reset. */
  backoffMs(now: number): number {
    return Math.max(0, this.backoffUntil - now, this.fgPauseUntil - now, this.quotaExhaustedUntilMs - now);
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
    let wait = Math.max(0, this.backoffUntil - now, this.fgPauseUntil - now);
    // the daily quota: spent -> until its reset; "watched_only" -> the sweep
    // does not ask at all (looked at again every QUOTA_RECHECK_MS)
    const held = this.sweepSuspendedByQuota(now);
    if (held === "exhausted") wait = Math.max(wait, this.quotaExhaustedUntilMs - now);
    else if (held === "watched_only") wait = Math.max(wait, QUOTA_RECHECK_MS);
    // ... and its share of the limit: one request per sweepMinGapMs
    const gap = this.sweepMinGapMs(now);
    if (gap > 0 && this.lastBgRequestAt != null) wait = Math.max(wait, this.lastBgRequestAt + gap - now);
    if (this.tokens < 1) wait = Math.max(wait, ((1 - this.tokens) / this.effectiveBgRps()) * 1000);
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
      effective_bg_rps: Math.round(this.effectiveBgRps() * 1000) / 1000,
      ceiling_rps: this.ceilingRps,
      window_s: this.windowMs / 1000,
      window_fg: fgWin,
      window_bg: this.recent.length - fgWin,
      // how long the sweep is held in total, then its two parts: its own
      // backoff, and the bounded pause foreground failures asked for
      backoff_s: Math.max(0, Math.ceil((Math.max(this.backoffUntil, this.fgPauseUntil) - now) / 1000)),
      sweep_backoff_s: Math.max(0, Math.ceil((this.backoffUntil - now) / 1000)),
      viewer_pause_s: Math.max(0, Math.ceil((this.fgPauseUntil - now) / 1000)),
      consecutive_failures: this.failures,
      sweep_consecutive_failures: this.bgFailures,
      totals: { ...this.counts },
      last_hour: this.lastHour(now),
      last_status: this.lastStatus,
      quota: this.quotaStats(now),
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

export function governorFromEnv(env: NodeJS.ProcessEnv = process.env, log?: (msg: string) => void): UpstreamGovernor {
  return new UpstreamGovernor({
    bgRps: envNum(env.GLOBAL_SWEEP_RPS, DEFAULT_GLOBAL_SWEEP_RPS, 0, 2),
    ceilingRps: envNum(env.ADSB_TOTAL_RPS_CEILING, DEFAULT_ADSB_TOTAL_RPS_CEILING, 0.05, 5),
    quotaModeOverride: quotaModeOverrideFromEnv(env),
    keyConfigured: Object.keys(adsbLolKeyHeader(env)).length > 0,
    log,
  });
}

/** THE process-wide adsb.lol governor (viewport chain + global sweep). It is
 *  the one that says quota changes in the log. */
export const adsbLolGovernor: UpstreamGovernor = governorFromEnv(process.env, (msg) => console.error(msg));

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
