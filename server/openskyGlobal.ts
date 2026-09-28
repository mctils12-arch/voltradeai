// openskyGlobal.ts — OPTIONAL OpenSky Network global snapshot feeding the
// global aircraft snapshot (FLIGHT PROGRAM B1, 2026-09-28).
//
// ACTIVE ONLY when BOTH OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET are set
// (OAuth2 client-credentials — OpenSky retired basic auth). Absent either,
// this module makes ZERO network calls and reports enabled:false.
//
// LICENSING (read before relying on it): OpenSky's terms grant use "solely
// for the purpose of non-profit research and non-profit education", and its
// §3 requires a WRITTEN AGREEMENT for any operational use of the REST API
// (research/wishlist.md MONETIZATION TRIPWIRE, 2026-07-03). It is therefore
// registered in providerCompliance.ts as NON-COMMERCIAL the moment
// credentials exist: a billing flip with OpenSky active degrades
// /api/health and writes a COMPLIANCE-WARNING, exactly like airplanes.live.
// Every OpenSky row is tagged src/provider "opensky" in the snapshot AND the
// archive (pv), so the adsb.lol-only (ODbL) subset stays a provable filter.
// HISTORY: OpenSky was removed from the viewport chain 2026-07-03 because
// Railway egress was rejected even with OAuth creds — failures here are
// logged loudly and surfaced in the endpoint's coverage block, never silent.
//
// CREDIT BUDGET: /api/states/all without a bbox costs 4 credits; a standard
// account gets ~4,000/day. CreditBudget spends at most
// OPENSKY_DAILY_CREDITS − reserve per UTC day, derives the minimum interval
// that lasts the whole day (≥ OPENSKY_INTERVAL_S, default 100s ≈ 864
// calls ≈ 3,456 credits), stops early when the X-Rate-Limit-Remaining
// header runs low, and on 429 pauses for X-Rate-Limit-Retry-After-Seconds.
//
// UNITS: OpenSky states are SI — baro_altitude/geo_altitude in METERS,
// velocity in M/S, true_track in degrees. The pipeline's AircraftPoint is
// also metric (altitude_m, velocity_ms), so the values pass through with
// the same integer rounding mapPointAircraft applies — never double-
// converted. Category is decoded from OpenSky's numeric emitter category
// (extended=1, index 17) to the DO-260 A1..C5 codes the rest of the map
// reads, via the documented table below (no inference beyond it).

import type { AircraftPoint } from "./datacoreArchive";
import { publishFixes, type FixBatch } from "./aircraftFixBus";
import { unrefTimer } from "./globalSweep";

export const OPENSKY_PROVIDER_KEY = "opensky";
export const OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all?extended=1";
export const OPENSKY_TOKEN_URL =
  "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token";
export const STATES_ALL_CREDIT_COST = 4;
export const DEFAULT_DAILY_CREDITS = 4000;
export const DEFAULT_INTERVAL_S = 100;
/** fraction of the daily allowance never spent (headroom for manual use) */
export const DEFAULT_RESERVE_FRACTION = 0.1;

export function openskyConfigured(env: NodeJS.ProcessEnv = process.env): boolean {
  return !!(env.OPENSKY_CLIENT_ID && String(env.OPENSKY_CLIENT_ID).trim()
    && env.OPENSKY_CLIENT_SECRET && String(env.OPENSKY_CLIENT_SECRET).trim());
}

/** OpenSky numeric emitter category (states/all extended=1, index 17) →
 *  DO-260 ADS-B emitter category. 0/1 = no information → null. 13 reserved. */
export const OPENSKY_CATEGORY_TO_ADSB: Record<number, string> = {
  2: "A1", 3: "A2", 4: "A3", 5: "A4", 6: "A5", 7: "A6", 8: "A7",
  9: "B1", 10: "B2", 11: "B3", 12: "B4", 14: "B6", 15: "B7",
  16: "C1", 17: "C2", 18: "C3", 19: "C4", 20: "C5",
};

const fin = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);

/**
 * states/all payload → pipeline AircraftPoints (provider "opensky") with a
 * per-row fix time (seen_at_ms = time_position, else last_contact). Rows
 * without a position or a position time are skipped — nothing is shown
 * from a guessed time.
 * State vector layout: [0 icao24, 1 callsign, 2 origin_country,
 * 3 time_position, 4 last_contact, 5 longitude, 6 latitude,
 * 7 baro_altitude(m), 8 on_ground, 9 velocity(m/s), 10 true_track,
 * 11 vertical_rate, 12 sensors, 13 geo_altitude(m), 14 squawk, 15 spi,
 * 16 position_source, 17 category].
 */
export function mapOpenSkyStates(raw: unknown): Array<AircraftPoint & { seen_at_ms: number }> {
  const st = (raw as { states?: unknown } | null)?.states;
  const states: unknown[] = Array.isArray(st) ? st : [];
  const out: Array<AircraftPoint & { seen_at_ms: number }> = [];
  for (const s of states) {
    if (!Array.isArray(s)) continue;
    const hex = typeof s[0] === "string" ? s[0].trim().toLowerCase() : "";
    const lon = fin(s[5]), lat = fin(s[6]);
    if (!hex || lon == null || lat == null) continue;
    const tPos = fin(s[3]) ?? fin(s[4]);
    if (tPos == null) continue;
    const onGround = s[8] === true;
    const altM = onGround ? null : (fin(s[7]) ?? fin(s[13]));
    const vel = fin(s[9]);
    const trk = fin(s[10]);
    const catNum = fin(s[17]);
    const posSrc = fin(s[16]);
    out.push({
      icao24: hex,
      callsign: typeof s[1] === "string" ? s[1].trim() : "",
      registration: null,
      lon, lat,
      altitude_m: altM == null ? null : Math.round(altM),
      on_ground: onGround,
      velocity_ms: vel == null ? null : Math.round(vel),
      heading: trk,
      type: null,
      category: catNum == null ? null : (OPENSKY_CATEGORY_TO_ADSB[catNum] ?? null),
      // OpenSky position_source 2 = MLAT (ground-computed) — the SAME
      // meaning readsb's "mlat" carries; 0 ADS-B / 1 ASTERIX / 3 FLARM have
      // no exact readsb equivalent and stay null (never a guessed subtype)
      pos_type: posSrc === 2 ? "mlat" : null,
      provider: OPENSKY_PROVIDER_KEY,
      seen_at_ms: Math.round(tPos * 1000),
    });
  }
  return out;
}

const utcDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);
const nextUtcMidnight = (ms: number) => {
  const d = new Date(ms);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1);
};

/** Daily OpenSky credit budgeter (pure; clock injected). */
export class CreditBudget {
  readonly dailyCredits: number;
  readonly reserve: number;
  readonly cost: number;
  private day = "";
  used = 0;
  headerRemaining: number | null = null;
  pausedUntil = 0;
  pauseReason: string | null = null;

  constructor(o: { dailyCredits?: number; reserve?: number; cost?: number } = {}) {
    this.dailyCredits = o.dailyCredits ?? DEFAULT_DAILY_CREDITS;
    this.reserve = o.reserve ?? Math.round(this.dailyCredits * DEFAULT_RESERVE_FRACTION);
    this.cost = o.cost ?? STATES_ALL_CREDIT_COST;
  }

  private roll(now: number) {
    const d = utcDay(now);
    if (d !== this.day) { this.day = d; this.used = 0; this.headerRemaining = null; }
  }

  /** Spendable credits left today under our own ceiling. */
  remaining(now: number): number {
    this.roll(now);
    return Math.max(0, this.dailyCredits - this.reserve - this.used);
  }

  /** Minimum call spacing that makes the budget last a whole day. */
  minIntervalMs(): number {
    const calls = Math.max(1, Math.floor((this.dailyCredits - this.reserve) / this.cost));
    return Math.ceil(86_400_000 / calls);
  }

  canSpend(now: number): boolean {
    this.roll(now);
    if (now < this.pausedUntil) return false;
    if (this.remaining(now) < this.cost) return false;
    // upstream's own counter: stop while it still has 2 calls of headroom
    if (this.headerRemaining != null && this.headerRemaining < this.cost * 2) return false;
    return true;
  }

  spend(now: number): void {
    this.roll(now);
    this.used += this.cost;
  }

  /** Feed response headers/status back (X-Rate-Limit-Remaining, 429 Retry-After). */
  noteResponse(now: number, status: number, remainingHeader: string | null, retryAfterSec: string | null): void {
    this.roll(now);
    const rem = remainingHeader != null && remainingHeader !== "" ? Number(remainingHeader) : NaN;
    if (Number.isFinite(rem)) {
      this.headerRemaining = rem;
      if (rem < this.cost * 2) {
        this.pausedUntil = Math.max(this.pausedUntil, nextUtcMidnight(now));
        this.pauseReason = `X-Rate-Limit-Remaining ${rem} — paused until next UTC day`;
      }
    }
    if (status === 429) {
      const ra = retryAfterSec != null && retryAfterSec !== "" ? Number(retryAfterSec) : NaN;
      this.pausedUntil = Math.max(this.pausedUntil,
        Number.isFinite(ra) && ra > 0 ? now + ra * 1000 : nextUtcMidnight(now));
      this.pauseReason = `429 from OpenSky${Number.isFinite(ra) ? ` (retry after ${ra}s)` : ""}`;
    }
  }

  status(now: number) {
    this.roll(now);
    return {
      day_utc: this.day,
      credits_used_today: this.used,
      daily_credits: this.dailyCredits,
      reserve: this.reserve,
      cost_per_call: this.cost,
      upstream_remaining: this.headerRemaining,
      paused_until: this.pausedUntil > now ? this.pausedUntil : null,
      pause_reason: this.pausedUntil > now ? this.pauseReason : null,
    };
  }
}

export function intervalMsFromEnv(env: NodeJS.ProcessEnv, budget: CreditBudget): number {
  const n = parseFloat(String(env.OPENSKY_INTERVAL_S ?? ""));
  const want = (Number.isFinite(n) && n > 0 ? n : DEFAULT_INTERVAL_S) * 1000;
  return Math.max(want, budget.minIntervalMs());
}

export function budgetFromEnv(env: NodeJS.ProcessEnv): CreditBudget {
  const d = parseFloat(String(env.OPENSKY_DAILY_CREDITS ?? ""));
  return new CreditBudget({ dailyCredits: Number.isFinite(d) && d > 0 ? d : DEFAULT_DAILY_CREDITS });
}

/** OAuth2 client-credentials token cache. */
export class TokenCache {
  private token: string | null = null;
  private expiresAt = 0;
  constructor(private readonly id: string, private readonly secret: string, private readonly fetchImpl: typeof fetch) {}
  invalidate() { this.token = null; this.expiresAt = 0; }
  async get(now: number): Promise<string> {
    if (this.token && now < this.expiresAt) return this.token;
    const body = new URLSearchParams({
      grant_type: "client_credentials", client_id: this.id, client_secret: this.secret,
    });
    const r = await this.fetchImpl(OPENSKY_TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: body.toString(),
      signal: AbortSignal.timeout(15_000),
    });
    if (!r.ok) throw new Error(`opensky token ${r.status}`);
    const j = (await r.json()) as { access_token?: unknown; expires_in?: unknown } | null;
    if (!j?.access_token) throw new Error("opensky token: no access_token");
    const ttl = Number(j.expires_in);
    this.token = String(j.access_token);
    this.expiresAt = now + Math.max(60, (Number.isFinite(ttl) ? ttl : 1800) - 60) * 1000;
    return this.token;
  }
}

export interface OpenSkyStepResult { ok: boolean; status: number; count: number; error?: string; skipped?: string }

/** One budgeted poll (pure-injectable). */
export async function pollOpenSkyOnce(deps: {
  fetchImpl: typeof fetch;
  tokens: TokenCache;
  budget: CreditBudget;
  now: () => number;
  publish?: (b: FixBatch) => void;
}): Promise<OpenSkyStepResult> {
  const { budget, now } = deps;
  if (!budget.canSpend(now())) return { ok: false, status: 0, count: 0, skipped: "credit budget" };
  let status = 0;
  try {
    const token = await deps.tokens.get(now());
    budget.spend(now()); // a sent request counts even if it fails
    const r = await deps.fetchImpl(OPENSKY_STATES_URL, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(30_000),
    });
    status = r.status ?? (r.ok ? 200 : 0);
    const h = (k: string): string | null => r.headers?.get?.(k) ?? null;
    budget.noteResponse(now(), status, h("x-rate-limit-remaining"), h("x-rate-limit-retry-after-seconds") ?? h("retry-after"));
    if (status === 401) deps.tokens.invalidate();
    if (!r.ok) return { ok: false, status, count: 0, error: `opensky states ${status}` };
    const raw: unknown = await r.json();
    const aircraft = mapOpenSkyStates(raw);
    (deps.publish ?? publishFixes)({
      provider: OPENSKY_PROVIDER_KEY, origin: "opensky", aircraft, fetchedAt: now(),
      upstreamNowMs: (() => { const t = fin((raw as { time?: unknown } | null)?.time); return t == null ? null : t * 1000; })(),
    });
    return { ok: true, status: 200, count: aircraft.length };
  } catch (e) {
    return { ok: false, status, count: 0, error: e instanceof Error ? e.message : String(e) };
  }
}

export interface OpenSkyStatus {
  enabled: boolean;
  /** why it is inactive (no credentials) */
  reason?: string;
  interval_s?: number;
  last?: OpenSkyStepResult | null;
  last_at?: number | null;
  last_ok_at?: number | null;
  budget?: ReturnType<CreditBudget["status"]>;
  license?: string;
}

export interface OpenSkyHandle {
  status: () => OpenSkyStatus;
  stop: () => void;
}

/** Boot the poller iff credentials exist; otherwise an inert handle. */
export function startOpenSkyGlobal(deps: { env?: NodeJS.ProcessEnv; fetchImpl?: typeof fetch } = {}): OpenSkyHandle {
  const env = deps.env ?? process.env;
  if (!openskyConfigured(env)) {
    return {
      status: () => ({ enabled: false, reason: "OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET not set — OpenSky inactive (adsb.lol only)" }),
      stop: () => {},
    };
  }
  const fetchImpl = deps.fetchImpl ?? fetch;
  const budget = budgetFromEnv(env);
  const intervalMs = intervalMsFromEnv(env, budget);
  const tokens = new TokenCache(String(env.OPENSKY_CLIENT_ID).trim(), String(env.OPENSKY_CLIENT_SECRET).trim(), fetchImpl);
  let last: OpenSkyStepResult | null = null;
  let lastAt = 0;
  let lastOkAt = 0;
  let lastErrLog = 0;
  let stopped = false;
  const tick = async () => {
    if (stopped) return;
    last = await pollOpenSkyOnce({ fetchImpl, tokens, budget, now: Date.now });
    lastAt = Date.now();
    if (last.ok) lastOkAt = lastAt;
    else if (!last.skipped && lastAt - lastErrLog > 10 * 60_000) {
      lastErrLog = lastAt;
      console.error(`[opensky-global] poll failed: ${last.error} — global snapshot continues on adsb.lol`);
    }
  };
  const timer = setInterval(() => { void tick(); }, intervalMs);
  unrefTimer(timer);
  const first = setTimeout(() => { void tick(); }, 30_000);
  unrefTimer(first);
  return {
    status: () => ({
      enabled: true, interval_s: intervalMs / 1000, last, last_at: lastAt || null,
      last_ok_at: lastOkAt || null, budget: budget.status(Date.now()),
      license: "NON-COMMERCIAL (research/education; operational use needs a written OpenSky agreement) — registered in providerCompliance.ts",
    }),
    stop: () => { stopped = true; clearInterval(timer); clearTimeout(first); },
  };
}
