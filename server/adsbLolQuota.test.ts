// adsb.lol's DAILY QUOTA (2026-11-17). Measured that day: api.adsb.lol allows
// 500 requests a day per IP without a key and says so on every response
//   x-ratelimit-limit: 500, 500;w=86400
//   x-ratelimit-remaining: <n>
//   x-ratelimit-reset: <seconds until 00:00 UTC>
// and, over the quota, answers 429 with `x-envoy-ratelimited: true`, an empty
// body and no Retry-After. This server sent about 29,500 requests a day, so
// the quota was gone ~25 minutes after midnight UTC and every lane drew 429
// for the rest of the day.
//
// These tests pin: the headers are read; a spent quota stops EVERY lane
// until its reset and is not treated as an outage; a small limit is spent on
// the tracked planes (and Find) only; a larger one paces the sweep; the API
// key is sent to adsb.lol only and never shows in a status.
// Run: npx tsx --test server/adsbLolQuota.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import express from "express";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { AddressInfo } from "node:net";
import {
  ADSB_LOL_UA, DISC_LANE_MIN_DAILY, QUOTA_FULL_MIN_DAILY, QUOTA_RECHECK_MS, SCOPES_QUOTA_SHARE, SWEEP_QUOTA_SHARE,
  TRACKED_QUOTA_SHARE, TRACKED_QUOTA_SHARE_FULL,
  UpstreamGovernor, adsbLolKeyHeader, adsbLolRequestHeaders, governorFromEnv, parseQuotaHeaders, quotaModeOverrideFromEnv,
  type QuotaLane,
} from "./adsbGovernor";
import { fetchDiscs, mapPointAircraft, planDiscs, type DiscProvider } from "./aircraftTiling";
import { createFastLane, fastLaneUrl } from "./aircraftFastLane";
import { createAircraftFinder, parseFindQuery } from "./aircraftFind";
import { GlobalSnapshot } from "./globalSnapshot";
import { SweepScheduler, runSweepJob, startGlobalSweep } from "./globalSweep";
import {
  GLOBAL_POLL_MS, GLOBAL_SCOPES, LADD_SCOPE_WATCHED_GAP_MS, createScopesPacer, pollGlobalScopesOnce, startGlobalScopes,
} from "./globalScopes";
import {
  FAILOVER_DEMAND_WINDOW_MS, FAILOVER_REASON_FAILING, FAILOVER_REASON_WATCHED_ONLY, createSweepFailover, failoverReason,
} from "./sweepFailover";
import {
  DB_FLAG_LADD, LADD_FLAG_SAVE_MIN_MS, LaddRegistry, learnLaddFromRows, rawRowIsLadd, rowIsLadd,
} from "./laddRegistry";
import { FEED_DEAD_AIR_HOURS, observeFeedDeadAir } from "./feedDeadAir";
import { publishFixes } from "./aircraftFixBus";
import { fileURLToPath } from "node:url";
import { TRACKED_POLL_MS, createTrackedCycle, type TrackedRegistry } from "./trackedPlanes";
import { registerGlobalAircraftRoutes } from "./globalAircraft";
import type { PlanDisc } from "./globalDiscPlan";
import type { FixBatch } from "./aircraftFixBus";

const here = path.dirname(fileURLToPath(import.meta.url));
/** 2026-11-17 20:00:00Z: four hours before the quota's reset at 00:00 UTC */
const T0 = Date.UTC(2026, 10, 17, 20, 0, 0);
const MIDNIGHT = Date.UTC(2026, 10, 18, 0, 0, 0);
const RESET_S = (MIDNIGHT - T0) / 1000; // 14,400
const DAY_MS = 86_400_000;

const hdr = (h: Record<string, string> = {}) => ({ get: (k: string) => h[k.toLowerCase()] ?? null });
/** the headers adsb.lol sends with every answer */
const quotaHdr = (limit: number, remaining: number, resetS: number = RESET_S, extra: Record<string, string> = {}) =>
  hdr({ "x-ratelimit-limit": `${limit}, ${limit};w=86400`, "x-ratelimit-remaining": String(remaining), "x-ratelimit-reset": String(resetS), ...extra });
/** the over-quota answer, as measured */
const SPENT_429 = (limit = 500, resetS: number = RESET_S) => quotaHdr(limit, 0, resetS, { "x-envoy-ratelimited": "true" });
const gov = () => new UpstreamGovernor({ bgRps: 10, ceilingRps: 100 });
/** a governor that has read `limit` requests a day with plenty left */
const govWithLimit = (limit: number, at = T0) => {
  const g = gov();
  g.noteQuota(quotaHdr(limit, Math.floor(limit / 2), (MIDNIGHT - at) / 1000), at, 200);
  return g;
};
const ALL_LANES: QuotaLane[] = ["sweep", "scopes", "ladd", "viewport", "selected", "tracked", "find"];
const quiet = async <T>(fn: () => Promise<T>): Promise<T> => {
  const orig = console.error;
  console.error = () => {};
  try { return await fn(); } finally { console.error = orig; }
};
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// ── reading the headers ──────────────────────────────────────────────────

test("parseQuotaHeaders: the three headers as adsb.lol sends them; the limit is the first integer", () => {
  const h = quotaHdr(500, 137, 11_009);
  assert.deepEqual(parseQuotaHeaders((k) => h.get(k)), { limit: 500, remaining: 137, resetSec: 11_009, ratelimited: false });
  const spent = SPENT_429(500, 11_009);
  assert.deepEqual(parseQuotaHeaders((k) => spent.get(k)), { limit: 500, remaining: 0, resetSec: 11_009, ratelimited: true });
  const keyed = hdr({ "x-ratelimit-limit": "7200, 7200;w=86400", "x-ratelimit-remaining": "7199", "x-ratelimit-reset": "86399" });
  assert.equal(parseQuotaHeaders((k) => keyed.get(k)).limit, 7200);
});

test("parseQuotaHeaders: absent or garbage headers read null, never a guess; a throwing reader is survived", () => {
  const none = { limit: null, remaining: null, resetSec: null, ratelimited: false };
  assert.deepEqual(parseQuotaHeaders(() => null), none);
  const junk = hdr({ "x-ratelimit-limit": "lots", "x-ratelimit-remaining": "NaN", "x-ratelimit-reset": "soon", "x-envoy-ratelimited": "maybe" });
  assert.deepEqual(parseQuotaHeaders((k) => junk.get(k)), none);
  const odd = hdr({ "x-ratelimit-limit": "0", "x-ratelimit-remaining": "1e3", "x-ratelimit-reset": "999999999", "x-envoy-ratelimited": "TRUE" });
  assert.deepEqual(parseQuotaHeaders((k) => odd.get(k)), { limit: null, remaining: null, resetSec: null, ratelimited: true },
    "a zero limit, an exponent and a reset years away are not figures; the flag is read case-blind");
  assert.deepEqual(parseQuotaHeaders(() => { throw new Error("no headers"); }), none);
});

test("noteQuota: a response without readable quota headers leaves the state exactly as it was", () => {
  const g = govWithLimit(500);
  const before = JSON.stringify(g.stats(T0).quota);
  assert.equal(g.noteQuota(null, T0, 200), false);
  assert.equal(g.noteQuota(undefined, T0, 503), false);
  assert.equal(g.noteQuota(hdr({ "x-ratelimit-limit": "garbage" }), T0, 200), false);
  assert.equal(g.noteQuota({} as never, T0, 200), false, "an object with no get()");
  assert.equal(JSON.stringify(g.stats(T0).quota), before);
  // and a governor that has read nothing says so
  const fresh = gov();
  fresh.noteQuota(hdr({}), T0, 200);
  assert.deepEqual(
    { limit: fresh.stats(T0).quota.limit, remaining: fresh.stats(T0).quota.remaining, mode: fresh.stats(T0).quota.mode, exhausted: fresh.stats(T0).quota.exhausted },
    { limit: null, remaining: null, mode: "unknown", exhausted: false });
});

// ── a spent quota ────────────────────────────────────────────────────────

test("spent: remaining 0 on an answered request stops every lane until now + x-ratelimit-reset", () => {
  const g = gov();
  assert.equal(g.noteQuota(quotaHdr(500, 1), T0, 200), false, "one left: not spent");
  assert.equal(g.quotaBlocked(T0), false);
  assert.equal(g.noteQuota(quotaHdr(500, 0), T0 + 1_000, 200), true, "that was the last one");
  const q = g.stats(T0 + 1_000).quota;
  assert.equal(q.exhausted, true);
  assert.equal(q.exhausted_until, T0 + 1_000 + RESET_S * 1000);
  assert.equal(q.remaining, 0);
  assert.equal(q.quota_exhausted_total, 1);
});

test("spent: the measured 429 (x-envoy-ratelimited, no Retry-After) stops every lane; without a reset header, until the next 00:00 UTC", () => {
  const g = gov();
  assert.equal(g.noteQuota(SPENT_429(), T0, 429), true);
  assert.equal(g.quotaExhaustedUntil(T0), MIDNIGHT);
  // the flag alone (no remaining, no reset): the next 00:00 UTC
  const bare = gov();
  assert.equal(bare.noteQuota(hdr({ "x-envoy-ratelimited": "true" }), T0, 429), true);
  assert.equal(bare.quotaExhaustedUntil(T0), MIDNIGHT);
  // a 429 with no headers of its own, after an answer that said 0 left... is already held:
  // remaining 0 stopped it, and the 429 changes nothing
  const zero = gov();
  assert.equal(zero.noteQuota(quotaHdr(500, 0), T0, 200), true);
  assert.equal(zero.noteQuota(hdr({}), T0 + 1_000, 429), true, "the last figure read was 0");
  assert.equal(zero.stats(T0 + 1_000).quota.quota_exhausted_total, 1, "one stretch, counted once");
});

test("a bare 429 is NOT the quota: with no quota information it is an ordinary rate limit and backs off as before", () => {
  // nothing read yet, a 429 with no headers worth the name: not the quota
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 100 });
  assert.equal(g.noteQuota(hdr({}), T0, 429), false);
  assert.equal(g.quotaBlocked(T0), false, "a stray 429 must not silence adsb.lol until midnight");
  g.noteResult("bg", 429, T0);
  const st = g.stats(T0);
  assert.equal(st.sweep_consecutive_failures, 1, "the ordinary failure path");
  assert.equal(st.sweep_backoff_s, 30);
  assert.equal(st.effective_bg_rps, 0.2, "and the ordinary AIMD cut");
  assert.equal(st.quota.quota_exhausted_total, 0);
  assert.equal(g.laneAllowed("tracked", T0), true);
  // with a Retry-After, or with requests still left: the same
  assert.equal(g.noteQuota(hdr({ "retry-after": "90" }), T0, 429), false);
  assert.equal(g.noteQuota(quotaHdr(7200, 6000), T0, 429), false, "6,000 left: not the daily quota");
  assert.equal(g.noteQuota(hdr({}), T0 + 1_000, 429), false, "no figure on this response; the last one read said 6,000 left");
  // once the window of a remembered 0 has passed, a bare 429 is ordinary again
  const old = gov();
  old.noteQuota(quotaHdr(500, 0, 60), T0, 200);
  assert.equal(old.noteQuota(hdr({}), T0 + 61_000, 429), false, "that figure belonged to yesterday's window");
  // and a 5xx or a 200 with nothing readable never is
  assert.equal(g.noteQuota(hdr({}), T0, 503), false);
  assert.equal(g.quotaBlocked(T0 + 1_000), false);
});

test("while spent NO lane asks adsb.lol, the sweep waits exactly to the reset, and everything resumes there", () => {
  const g = gov();
  g.noteQuota(SPENT_429(), T0, 429);
  for (const lane of ALL_LANES) assert.equal(g.laneAllowed(lane, T0 + 60_000), false, `${lane} must not ask`);
  assert.equal(g.backgroundWaitMs(T0), RESET_S * 1000, "the sweep's wait is the time to the reset");
  assert.equal(g.tryAcquireBackground(T0 + 3_600_000), false);
  assert.equal(g.backoffMs(T0), RESET_S * 1000, "callers that hold on backoffMs hold too");
  assert.equal(g.sweepSuspendedByQuota(T0), "exhausted");
  // refusals are counted as requests not sent
  assert.equal(g.admit("viewport", T0), false);
  assert.equal(g.admit("scopes", T0, 3), false);
  assert.equal(g.stats(T0).quota.requests_skipped_quota, 4);
  // one millisecond before the reset: still held. At the reset: open again.
  assert.equal(g.quotaBlocked(MIDNIGHT - 1), true);
  assert.equal(g.quotaBlocked(MIDNIGHT), false);
  assert.equal(g.laneAllowed("tracked", MIDNIGHT), true);
  const q = g.stats(MIDNIGHT).quota;
  assert.deepEqual({ exhausted: q.exhausted, exhausted_until: q.exhausted_until, remaining: q.remaining }, { exhausted: false, exhausted_until: null, remaining: null },
    "after the reset the old remaining figure is not repeated");
});

test("a quota 429 is not an outage: no failure backoff, no AIMD cut, no sweep failure — an ordinary 429 still is one", () => {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 100 });
  g.noteQuota(SPENT_429(7200), T0, 429);
  g.noteResult("bg", 429, T0);
  g.noteResult("fg", 429, T0);
  const st = g.stats(T0);
  assert.equal(st.sweep_consecutive_failures, 0, "the failover's 'sweep failing' signal does not move");
  assert.equal(st.consecutive_failures, 0);
  assert.equal(st.sweep_backoff_s, 0);
  assert.equal(st.viewer_pause_s, 0);
  assert.equal(st.effective_bg_rps, 0.4, "the sweep's rate is not halved");
  assert.deepEqual(st.totals, { fg: 0, bg: 0, fgFail: 1, bgFail: 1 }, "the refused requests are still counted");
  assert.equal(st.last_hour.bg.rate_limited, 1);
  // at the reset the sweep goes straight back out: nothing is left to wait for
  assert.equal(g.backgroundWaitMs(MIDNIGHT), 0);
  assert.equal(g.tryAcquireBackground(MIDNIGHT), true);
  // control: the same 429 with a Retry-After is a failure, as it always was
  const c = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 100 });
  c.noteQuota(hdr({ "retry-after": "60" }), T0, 429);
  c.noteResult("bg", 429, T0, 60);
  assert.equal(c.stats(T0).sweep_consecutive_failures, 1);
  assert.ok(c.stats(T0).sweep_backoff_s >= 60);
  assert.equal(c.stats(T0).effective_bg_rps, 0.2);
});

// ── the mode, from the limit ─────────────────────────────────────────────

test("mode: unknown until a header is read; under 5,000 a day = watched_only; from 5,000 = full", () => {
  assert.equal(QUOTA_FULL_MIN_DAILY, 5000);
  assert.equal(gov().quotaMode(T0), "unknown");
  assert.equal(govWithLimit(500).quotaMode(T0), "watched_only", "no key");
  assert.equal(govWithLimit(1000).quotaMode(T0), "watched_only", "the free adhoc key");
  assert.equal(govWithLimit(4999).quotaMode(T0), "watched_only");
  assert.equal(govWithLimit(5000).quotaMode(T0), "full");
  assert.equal(govWithLimit(7200).quotaMode(T0), "full", "a feeder key");
});

test("mode: who may ask adsb.lol — watched_only keeps it for the tracked planes, Find and the LADD scope; full and unknown let everyone", () => {
  const w = govWithLimit(500);
  const allowed = (g: UpstreamGovernor) => ALL_LANES.filter((l) => g.laneAllowed(l, T0));
  assert.deepEqual(allowed(w), ["ladd", "tracked", "find"], "the LADD scope keeps teaching the privacy registry (paced in globalScopes.ts)");
  assert.deepEqual(allowed(govWithLimit(7200)), ALL_LANES);
  assert.deepEqual(allowed(gov()), ALL_LANES, "no header yet: as before");
  // the sweep's lane is held, and looked at again every QUOTA_RECHECK_MS
  assert.equal(w.sweepSuspendedByQuota(T0), "watched_only");
  assert.equal(w.backgroundWaitMs(T0), QUOTA_RECHECK_MS);
  assert.equal(w.tryAcquireBackground(T0 + 10 * QUOTA_RECHECK_MS), false);
  assert.equal(govWithLimit(7200).sweepSuspendedByQuota(T0), null);
});

test("mode: ADSBLOL_QUOTA_MODE=full|watched_only fixes it whatever the header says; anything else is ignored", () => {
  assert.equal(quotaModeOverrideFromEnv({ ADSBLOL_QUOTA_MODE: " Full " } as NodeJS.ProcessEnv), "full");
  assert.equal(quotaModeOverrideFromEnv({ ADSBLOL_QUOTA_MODE: "watched_only" } as NodeJS.ProcessEnv), "watched_only");
  assert.equal(quotaModeOverrideFromEnv({ ADSBLOL_QUOTA_MODE: "auto" } as NodeJS.ProcessEnv), null);
  assert.equal(quotaModeOverrideFromEnv({} as NodeJS.ProcessEnv), null);
  const full = governorFromEnv({ ADSBLOL_QUOTA_MODE: "full" } as NodeJS.ProcessEnv);
  full.noteQuota(quotaHdr(500, 400), T0, 200);
  assert.equal(full.quotaMode(T0), "full");
  assert.equal(full.laneAllowed("sweep", T0), true);
  const watched = governorFromEnv({ ADSBLOL_QUOTA_MODE: "watched_only" } as NodeJS.ProcessEnv);
  assert.equal(watched.quotaMode(T0), "watched_only", "before any header");
  watched.noteQuota(quotaHdr(50_000, 40_000), T0, 200);
  assert.equal(watched.quotaMode(T0), "watched_only");
  assert.equal(watched.laneAllowed("viewport", T0), false);
  // the override never lifts a spent quota
  full.noteQuota(SPENT_429(), T0, 429);
  assert.equal(full.laneAllowed("sweep", T0), false);
});

test("full mode: the sweep is spaced to 60% of the limit — 20 s a request at 7,200 a day, 8 s at 18,000", () => {
  assert.equal(SWEEP_QUOTA_SHARE, 0.6);
  assert.equal(gov().sweepMinGapMs(T0), 0, "unknown limit: no gap");
  const feeder = govWithLimit(7200);
  assert.equal(feeder.sweepMinGapMs(T0), 20_000);
  assert.equal(feeder.stats(T0).quota.sweep_min_gap_ms, 20_000);
  assert.equal(govWithLimit(18_000).sweepMinGapMs(T0), 8_000);
  // and the governor enforces it (bgRps 10 would otherwise allow one every 100 ms)
  assert.equal(feeder.tryAcquireBackground(T0), true);
  assert.equal(feeder.tryAcquireBackground(T0 + 5_000), false);
  assert.equal(feeder.backgroundWaitMs(T0 + 5_000), 15_000);
  assert.equal(feeder.tryAcquireBackground(T0 + 19_999), false);
  assert.equal(feeder.tryAcquireBackground(T0 + 20_000), true);
  // a whole day of it stays inside the share
  let n = 2;
  for (let t = T0 + 20_500; t < T0 + DAY_MS; t += 500) if (feeder.tryAcquireBackground(t)) n++;
  assert.ok(n <= 7200 * SWEEP_QUOTA_SHARE, `${n} sweep requests in a day at 7,200/day`);
  assert.ok(n >= 7200 * SWEEP_QUOTA_SHARE - 5, `the share is used, not wasted (${n})`);
});

test("full mode: the disc lane asks adsb.lol only from 20,000 a day; with no limit read it runs as before", () => {
  assert.equal(DISC_LANE_MIN_DAILY, 20_000);
  assert.equal(gov().discLaneAllowed(T0), true);
  assert.equal(govWithLimit(7200).discLaneAllowed(T0), false);
  assert.equal(govWithLimit(19_999).discLaneAllowed(T0), false);
  assert.equal(govWithLimit(20_000).discLaneAllowed(T0), true);
  assert.equal(govWithLimit(500).discLaneAllowed(T0), false, "watched_only: the sweep does not run at all");
});

// ── the key ──────────────────────────────────────────────────────────────

test("the API key: sent as x-api-key when ADSBLOL_API_KEY is set, absent otherwise", () => {
  assert.deepEqual(adsbLolRequestHeaders({} as NodeJS.ProcessEnv), { "User-Agent": ADSB_LOL_UA });
  assert.deepEqual(adsbLolRequestHeaders({ ADSBLOL_API_KEY: "  " } as NodeJS.ProcessEnv), { "User-Agent": ADSB_LOL_UA }, "blank is no key");
  assert.deepEqual(adsbLolRequestHeaders({ ADSBLOL_API_KEY: " k-123 " } as NodeJS.ProcessEnv), { "User-Agent": ADSB_LOL_UA, "x-api-key": "k-123" });
  assert.deepEqual(adsbLolKeyHeader({ ADSBLOL_API_KEY: "k-123" } as NodeJS.ProcessEnv), { "x-api-key": "k-123" });
  assert.match(ADSB_LOL_UA, /^voltradeai-datacore\/1\.0/);
});

test("the API key never appears in the governor's status or its log line; key_configured is a boolean", () => {
  const KEY = "sekrit-ADSBLOL-key-9f3a";
  const said: string[] = [];
  const g = governorFromEnv({ ADSBLOL_API_KEY: KEY } as NodeJS.ProcessEnv, (m) => said.push(m));
  g.noteQuota(quotaHdr(7200, 10), T0, 200);
  g.noteQuota(SPENT_429(7200, RESET_S - 1), T0 + 1_000, 429);
  const st = g.stats(T0 + 1_000);
  assert.equal(st.quota.key_configured, true);
  assert.equal(JSON.stringify(st).includes(KEY), false);
  assert.equal(said.length, 1, "the spent quota is said once");
  assert.match(said[0], /daily quota spent \(limit 7200\/day\).*until 2026-11-18T00:00:00\.000Z/);
  assert.equal(said.join("\n").includes(KEY), false);
  assert.equal(governorFromEnv({} as NodeJS.ProcessEnv).stats(T0).quota.key_configured, false);
  assert.deepEqual(Object.keys(st.quota).sort(), [
    "exhausted", "exhausted_until", "key_configured", "limit", "mode", "quota_exhausted_total",
    "remaining", "requests_skipped_quota", "reset_at", "sweep_min_gap_ms",
  ]);
  assert.equal(st.quota.reset_at, MIDNIGHT);
});

// ── the world sweep ──────────────────────────────────────────────────────

const D = (id: number, lat: number, lon: number): PlanDisc => ({ id, lat, lon, radiusNm: 250, prior: 5 });

test("runSweepJob sends the key to adsb.lol and reports the answer's quota headers to the governor", async () => {
  const g = gov();
  const s = new SweepScheduler([D(0, 40, -95)], { now: T0, types: ["B738"] });
  const seen: Array<Record<string, string>> = [];
  const r = await runSweepJob(s.next(T0)!, s, {
    governor: g, now: () => T0, publish: () => {}, headers: adsbLolRequestHeaders({ ADSBLOL_API_KEY: "k-123" } as NodeJS.ProcessEnv),
    fetchImpl: (async (_u: unknown, init: { headers: Record<string, string> }) => {
      seen.push(init.headers);
      return { ok: true, status: 200, headers: quotaHdr(7200, 7100), json: async () => ({ ac: [] }) };
    }) as never,
  });
  assert.equal(r.ok, true);
  assert.equal(seen[0]["x-api-key"], "k-123");
  assert.equal(seen[0]["User-Agent"], ADSB_LOL_UA);
  assert.equal(g.stats(T0).quota.limit, 7200);
  assert.equal(g.stats(T0).quota.remaining, 7100);
  assert.equal(g.quotaMode(T0), "full");
});

test("runSweepJob: the quota's 429 is a failed job but not a sweep failure; the sweep then waits for the reset", async () => {
  const g = gov();
  const s = new SweepScheduler([D(0, 40, -95)], { now: T0, types: ["B738"] });
  const r = await runSweepJob(s.next(T0)!, s, {
    governor: g, now: () => T0, publish: () => { throw new Error("must not publish"); },
    fetchImpl: (async () => ({ ok: false, status: 429, headers: SPENT_429() })) as never,
  });
  assert.equal(r.ok, false);
  assert.equal(r.status, 429);
  assert.equal(g.stats(T0).sweep_consecutive_failures, 0);
  assert.equal(g.backgroundWaitMs(T0), RESET_S * 1000);
  assert.equal(s.batches[0].fails, 1);
});

/** run the real sweep loop for a moment and count what it sent */
async function sweepRequests(g: UpstreamGovernor, ms = 250): Promise<number> {
  let calls = 0;
  const h = startGlobalSweep({
    env: {} as NodeJS.ProcessEnv, governor: g, plan: [D(0, 40, -95), D(1, 51, 0)], types: ["B738", "A320"], startDelayMs: 0,
    fetchImpl: (async () => { calls++; return { ok: true, status: 200, headers: hdr({}), json: async () => ({ ac: [] }) }; }) as never,
  });
  await sleep(ms);
  h.stop();
  return calls;
}

test("the sweep loop sends NOTHING in watched_only mode or while the quota is spent (and does send when neither holds)", async () => {
  assert.ok(await sweepRequests(gov()) >= 1, "control: with no limit read the loop does ask");
  assert.equal(await sweepRequests(govWithLimit(500, Date.now())), 0, "watched_only: both lanes silent");
  const spent = gov();
  spent.noteQuota(SPENT_429(7200, 3600), Date.now(), 429);
  assert.equal(await sweepRequests(spent), 0, "spent: both lanes silent");
});

test("the sweep loop under a 7,200/day limit: the type lane only, one request per gap — the disc lane does not ask", async () => {
  const urls: string[] = [];
  const g = govWithLimit(7200, Date.now());
  const h = startGlobalSweep({
    env: {} as NodeJS.ProcessEnv, governor: g, plan: [D(0, 40, -95)], types: ["B738"], startDelayMs: 0,
    fetchImpl: (async (u: unknown) => { urls.push(String(u)); return { ok: true, status: 200, headers: hdr({}), json: async () => ({ ac: [] }) }; }) as never,
  });
  await sleep(300);
  h.stop();
  assert.deepEqual(urls, ["https://api.adsb.lol/v2/type/B738"], "one type batch, then the 20 s gap; never a /v2/point disc");
});

// ── the scopes ───────────────────────────────────────────────────────────

test("scopes in watched_only mode: mil and pia are not asked; the LADD scope alone is, at most once per 15 minutes (96 a day)", async () => {
  assert.equal(LADD_SCOPE_WATCHED_GAP_MS, 15 * 60_000);
  // the pacer, on a fake clock, one poller tick every 120 s for a day
  const g = govWithLimit(500);
  const pacer = createScopesPacer(g);
  const asked: Array<{ t: number; scope: string }> = [];
  for (let t = T0; t < T0 + DAY_MS; t += GLOBAL_POLL_MS) {
    assert.equal(pacer.cycleDue(t), true);
    for (const scope of GLOBAL_SCOPES) {
      const turn = pacer.decide(scope, t);
      if (scope !== "ladd") assert.equal(turn, "quota");
      if (turn === "ask") { pacer.asked(scope, t); asked.push({ t, scope }); }
    }
  }
  assert.ok(asked.every((a) => a.scope === "ladd"));
  assert.equal(asked.length, 90, "at most one per 15 minutes; on the poller's 120 s tick that is every 16 minutes");
  assert.ok(asked.length <= DAY_MS / LADD_SCOPE_WATCHED_GAP_MS, "never more than 96 a day");
  const gaps = asked.slice(1).map((a, i) => a.t - asked[i].t);
  assert.ok(gaps.every((d) => d >= LADD_SCOPE_WATCHED_GAP_MS), "never sooner");
  assert.ok(DAY_MS / LADD_SCOPE_WATCHED_GAP_MS + 500 * TRACKED_QUOTA_SHARE <= 500, "LADD (at most 96) + the tracked planes (200) fit 500 a day");

  // the real cycle: one request, to /v2/ladd, and its rows reach the registry
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-scopes-"));
  try {
    const now = Date.now();
    const live = govWithLimit(500, now);
    const urls: string[] = [];
    const taught: string[] = [];
    const cyclePacer = createScopesPacer(live);
    const fetchImpl = (async (u: unknown) => {
      urls.push(String(u));
      return { ok: true, status: 200, headers: quotaHdr(500, 400, 3600), json: async () => ({ ac: [{ hex: "ab8c8e", flight: "N843S", lat: 41, lon: -87 }] }) };
    }) as never;
    const st = await pollGlobalScopesOnce(fetchImpl, {}, [], dir, (rows) => { for (const r of rows) taught.push(r.icao24); }, live, cyclePacer);
    assert.deepEqual(urls, ["https://api.adsb.lol/v2/ladd"]);
    assert.deepEqual(taught, ["ab8c8e"], "the LADD registry is taught");
    assert.deepEqual(st.per_scope, { ladd: 1 }, "a scope that was not asked has no entry (-1 stays 'the fetch failed')");
    assert.deepEqual({ skipped_quota: st.skipped_quota, paced_wait: st.paced_wait, asked: st.asked }, { skipped_quota: true, paced_wait: false, asked: 1 });
    assert.equal(live.stats(now).quota.requests_skipped_quota, 2, "mil and pia: two requests not sent");
    // the next tick, two minutes later in the poller, is inside the 15 minutes: nothing at all
    const again = await pollGlobalScopesOnce(fetchImpl, {}, [], dir, () => {}, live, cyclePacer);
    assert.equal(urls.length, 1);
    assert.deepEqual({ skipped_quota: again.skipped_quota, paced_wait: again.paced_wait, asked: again.asked }, { skipped_quota: true, paced_wait: true, asked: 0 });
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("scopes while the quota is spent: nothing is asked, not even LADD; the cycle says skipped_quota", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-scopes-"));
  try {
    const now = Date.now(); // the scopes poller reads the wall clock
    const spent = gov();
    spent.noteQuota(SPENT_429(500, 3600), now, 429);
    let calls = 0;
    const st = await pollGlobalScopesOnce((async () => { calls++; return { ok: true, json: async () => ({ ac: [] }) }; }) as never, {}, [], dir, () => {}, spent);
    assert.equal(calls, 0);
    assert.equal(st.skipped_quota, true);
    assert.deepEqual(st.per_scope, {});
    assert.equal(spent.stats(now).quota.requests_skipped_quota, 3, "three requests not sent");
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("scopes: the first answer states 500 a day, so pia is not asked in that cycle — LADD still is", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-scopes-"));
  try {
    const g = gov();
    const urls: string[] = [];
    const st = await pollGlobalScopesOnce((async (u: unknown) => {
      urls.push(String(u));
      return { ok: true, status: 200, headers: quotaHdr(500, 480, 3600), json: async () => ({ ac: [] }) };
    }) as never, {}, [], dir, () => {}, g);
    assert.deepEqual(urls, ["https://api.adsb.lol/v2/mil", "https://api.adsb.lol/v2/ladd"]);
    assert.deepEqual(st.per_scope, { mil: 0, ladd: 0 });
    assert.equal(st.skipped_quota, true);
    // without a governor the cycle is exactly what it was
    const plain = await pollGlobalScopesOnce((async () => ({ ok: true, json: async () => ({ ac: [] }) })) as never, {}, [], dir, () => {});
    assert.deepEqual(plain.per_scope, { mil: 0, ladd: 0, pia: 0 });
    assert.deepEqual({ skipped_quota: plain.skipped_quota, paced_wait: plain.paced_wait, asked: plain.asked }, { skipped_quota: false, paced_wait: false, asked: 3 });
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("scopes: the poller sends the key with each scope request", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-scopes-"));
  const seen: Array<Record<string, string>> = [];
  const h = startGlobalScopes({
    fetchImpl: (async (_u: unknown, init: { headers: Record<string, string> }) => { seen.push(init.headers); return { ok: true, status: 200, headers: hdr({}), json: async () => ({ ac: [] }) }; }) as never,
    headers: { "User-Agent": "ua" }, sites: [], baseDir: dir, governor: gov(), env: { ADSBLOL_API_KEY: "k-123" } as NodeJS.ProcessEnv,
  });
  await sleep(50);
  h.stop();
  fs.rmSync(dir, { recursive: true, force: true });
  assert.equal(seen.length, 3);
  for (const s of seen) assert.deepEqual(s, { "User-Agent": "ua", "x-api-key": "k-123" });
});

// ── the viewport chain ───────────────────────────────────────────────────

const CHAIN: DiscProvider[] = [
  { key: "adsblol", label: "adsb.lol", arr: "ac", url: (la, lo, r) => `https://api.adsb.lol/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${r}` },
  { key: "airplaneslive", label: "airplanes.live", arr: "ac", url: (la, lo, r) => `https://api.airplanes.live/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${r}` },
  { key: "adsbfi", label: "adsb.fi", arr: "aircraft", url: (la, lo, r) => `https://opendata.adsb.fi/api/v2/lat/${la.toFixed(3)}/lon/${lo.toFixed(3)}/dist/${r}` },
];

test("viewport discs skip adsb.lol in watched_only mode: no request spent, no backoff bumped, the next provider answers at once", async () => {
  const now = Date.now();
  for (const g of [govWithLimit(500, now), (() => { const x = gov(); x.noteQuota(SPENT_429(7200, 3600), now, 429); return x; })()]) {
    const urls: string[] = [];
    const bumped: string[] = [];
    const out = await fetchDiscs(planDiscs(-1.5, 1.5, -4, 4), {
      providers: CHAIN, governor: g, sleep: async () => {}, backoffBump: (k) => bumped.push(k),
      fetchImpl: (async (u: unknown) => { urls.push(String(u)); return { ok: true, status: 200, json: async () => ({ ac: [{ hex: "abc123", lat: 1, lon: 1 }] }) }; }) as never,
    });
    assert.equal(urls.length, 2, "two discs");
    assert.ok(urls.every((u) => u.includes("airplanes.live")), "adsb.lol was never asked");
    assert.deepEqual(bumped, [], "being held by the quota is not a provider failure");
    assert.deepEqual(out.sources, ["airplanes.live"]);
    assert.equal(out.discsOk, 2);
    assert.ok(out.errs.some((e) => /adsblol held by its daily quota/.test(e)));
    assert.equal(g.stats(now).totals.fg, 0, "nothing recorded as sent");
    assert.equal(g.stats(now).quota.requests_skipped_quota, 2);
  }
});

test("viewport discs: the key goes to adsb.lol and to no other provider; the answer's quota headers reach the governor", async () => {
  const g = gov();
  const seen: Array<{ url: string; headers: Record<string, string> }> = [];
  await fetchDiscs(planDiscs(39.5, 40.5, -75.5, -74.5), {
    providers: CHAIN, governor: g, sleep: async () => {}, headers: { "User-Agent": "ua" }, env: { ADSBLOL_API_KEY: "k-123" } as NodeJS.ProcessEnv,
    fetchImpl: (async (u: unknown, init: { headers: Record<string, string> }) => {
      seen.push({ url: String(u), headers: init.headers });
      if (String(u).includes("adsb.lol")) return { ok: false, status: 503, headers: quotaHdr(7200, 7000) };
      if (String(u).includes("airplanes.live")) return { ok: false, status: 503, headers: hdr({}) };
      return { ok: true, status: 200, headers: hdr({}), json: async () => ({ aircraft: [{ hex: "abc123", lat: 40, lon: -75 }] }) };
    }) as never,
  });
  assert.equal(seen.length, 3);
  assert.deepEqual(seen[0].headers, { "User-Agent": "ua", "x-api-key": "k-123" });
  assert.deepEqual(seen[1].headers, { "User-Agent": "ua" }, "airplanes.live never sees the adsb.lol key");
  assert.deepEqual(seen[2].headers, { "User-Agent": "ua" }, "adsb.fi never sees the adsb.lol key");
  assert.equal(g.stats(Date.now()).quota.limit, 7200, "read off a failed answer too");
});

test("viewport discs: the quota's 429 on the first disc sends that disc and every later one to the next provider", async () => {
  const g = gov();
  const urls: string[] = [];
  const out = await fetchDiscs(planDiscs(-1.5, 1.5, -4, 4), {
    providers: CHAIN, governor: g, sleep: async () => {},
    fetchImpl: (async (u: unknown) => {
      urls.push(String(u));
      if (String(u).includes("adsb.lol")) return { ok: false, status: 429, headers: SPENT_429(500, 3600) };
      return { ok: true, status: 200, json: async () => ({ ac: [{ hex: "abc123", lat: 1, lon: 1 }] }) };
    }) as never,
  });
  assert.equal(urls.filter((u) => u.includes("adsb.lol")).length, 1, "one request found the quota spent; the second disc did not ask");
  assert.equal(out.discsOk, 2);
  assert.equal(g.stats(Date.now()).consecutive_failures, 0);
});

// ── the selected-plane lane ──────────────────────────────────────────────

function fastRig(g: UpstreamGovernor, env: NodeJS.ProcessEnv = {} as NodeJS.ProcessEnv) {
  let clock = T0;
  const snap = new GlobalSnapshot();
  const primary: Array<{ url: string; headers: Record<string, string> }> = [];
  const fallback: string[] = [];
  const lane = createFastLane({
    snapshotGet: (h) => snap.get(h),
    publish: (b: FixBatch) => { snap.ingest(b, clock); },
    governor: g, now: () => clock, env,
    fetchImpl: (async (url: string, init: { headers: Record<string, string> }) => {
      primary.push({ url, headers: init.headers });
      return {
        ok: true, status: 200, headers: quotaHdr(7200, 7000, 3600),
        json: async () => ({ now: clock, ac: [{ hex: "ab8c8e", r: "N843S", lat: 36.2, lon: -83.95, alt_baro: 43000, gs: 466, track: 186, seen_pos: 0.4 }] }),
      };
    }) as never,
    fallbackHex: async (hex) => {
      fallback.push(hex);
      return [{ icao24: hex, callsign: "N843S", lat: 36.4, lon: -83.9, altitude_m: 13100, on_ground: false, velocity_ms: 240, heading: 186, seen_pos: 1, provider: "adsbfi" }];
    },
  });
  return { lane, primary, fallback, tick: (ms: number) => { clock += ms; } };
}

test("selected-plane lane: in watched_only mode and while spent it goes straight to the second provider, never to adsb.lol", async () => {
  for (const g of [govWithLimit(500), (() => { const x = gov(); x.noteQuota(SPENT_429(7200), T0, 429); return x; })()]) {
    const r = fastRig(g);
    const res = await r.lane.lookup("ab8c8e");
    assert.equal(res.source, "fallback");
    assert.equal(res.row?.src, "adsbfi", "the row says which provider it came from");
    assert.equal(res.upstreamStatus, null, "no adsb.lol request was made");
    assert.deepEqual(r.primary, []);
    assert.deepEqual(r.fallback, ["ab8c8e"]);
    r.tick(5_000);
    await r.lane.lookup("ab8c8e");
    assert.deepEqual(r.primary, []);
    const st = r.lane.stats();
    assert.equal(st.quota_skips, 2);
    assert.equal(st.breaker_skips, 0, "the quota, not the failure breaker");
    assert.equal(st.upstream_requests, 0);
    assert.equal(g.stats(T0).totals.fg, 0);
  }
});

test("selected-plane lane: with quota to spend it asks adsb.lol with the key and reports the quota headers", async () => {
  const g = gov();
  const r = fastRig(g, { ADSBLOL_API_KEY: "k-123" } as NodeJS.ProcessEnv);
  const res = await r.lane.lookup("ab8c8e");
  assert.equal(res.source, "upstream");
  assert.equal(r.primary[0].url, fastLaneUrl("ab8c8e"));
  assert.deepEqual(r.primary[0].headers, { "User-Agent": ADSB_LOL_UA, "x-api-key": "k-123" });
  assert.equal(g.stats(T0).quota.limit, 7200);
  assert.deepEqual(r.fallback, []);
});

// ── Find ─────────────────────────────────────────────────────────────────

function findRig(g: UpstreamGovernor, env: NodeJS.ProcessEnv = {} as NodeJS.ProcessEnv) {
  const calls: Array<{ url: string; headers: Record<string, string> }> = [];
  const finder = createAircraftFinder({
    snapshot: new GlobalSnapshot(), tracked: () => [], governor: g, now: () => T0, env,
    fetchImpl: (async (url: string, init: { headers: Record<string, string> }) => {
      calls.push({ url, headers: init.headers });
      return { ok: true, status: 200, headers: quotaHdr(500, 300), json: async () => ({ now: T0, ac: [{ hex: "ab8c8e", r: "N843S", lat: 41.7, lon: -87.8, alt_baro: 2825, gs: 200, seen: 0.1, seen_pos: 0.3 }] }) };
    }) as never,
  });
  return { finder, calls };
}

test("Find: a lookup may spend the quota in watched_only mode (with the key); once the quota is spent it asks nothing and says so", async () => {
  const w = findRig(govWithLimit(500), { ADSBLOL_API_KEY: "k-123" } as NodeJS.ProcessEnv);
  const live = await w.finder.find(parseFindQuery("N843S")!);
  assert.equal(live.state, "live");
  assert.equal(live.upstream, "found");
  assert.deepEqual(w.calls.map((c) => c.url), ["https://api.adsb.lol/v2/reg/N843S"]);
  assert.deepEqual(w.calls[0].headers, { "User-Agent": ADSB_LOL_UA, "x-api-key": "k-123" });

  const spentGov = gov();
  spentGov.noteQuota(SPENT_429(), T0, 429);
  const s = findRig(spentGov);
  const held = await s.finder.find(parseFindQuery("N843S")!);
  assert.deepEqual(s.calls, []);
  assert.equal(held.state, "unknown");
  assert.equal(held.upstream, "limited");
  assert.equal(s.finder.stats().upstream_quota_skips, 1);
  assert.equal(s.finder.stats().upstream_requests, 0);
});

// ── the tracked planes ───────────────────────────────────────────────────

interface TrackedCall { t: number; url: string; headers: Record<string, string> }
/** adsb.lol for the tracked poller: `reg` and `hex` answer by URL kind, every answer carries `headers()` */
function trackedProvider(o: {
  clock: () => number;
  headers: () => { get: (k: string) => string | null };
  status?: () => number;
  reg?: (t: number) => unknown;
  hex?: (t: number) => unknown;
}) {
  const calls: TrackedCall[] = [];
  const fetchImpl = (async (url: string, init: { headers: Record<string, string> }) => {
    const t = o.clock();
    calls.push({ t, url, headers: init.headers });
    const status = o.status?.() ?? 200;
    const body = url.includes("/v2/hex/") ? (o.hex?.(t) ?? { ac: [], now: t }) : (o.reg?.(t) ?? { ac: [], now: t });
    return { ok: status === 200, status, headers: o.headers(), json: async () => body };
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}
const positioned = (t: number) => ({ now: t, ac: [{ hex: "ab8c8e", r: "N843S", t: "FA7X", alt_baro: 41000, gs: 470, lat: 39.1, lon: -94.6, seen: 0.2, seen_pos: 0.4 }] });

test("tracked planes at 500 a day: one request per 432 s (40% of the limit), no 5 s fast ticks, 200 requests in a day", async () => {
  assert.equal(TRACKED_QUOTA_SHARE, 0.4);
  assert.equal(govWithLimit(500).trackedMinGapMs(T0), 432_000);
  assert.equal(govWithLimit(1000).trackedMinGapMs(T0), 216_000, "the free adhoc key");
  assert.equal(govWithLimit(7200).trackedMinGapMs(T0), 48_000, "full mode: its 25% share (see the full-mode test)");
  assert.equal(gov().trackedMinGapMs(T0), 0, "unknown: as before");

  const g = gov();
  let t = T0;
  let reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e" }] };
  const p = trackedProvider({ clock: () => t, headers: () => quotaHdr(500, 400, (MIDNIGHT + DAY_MS - t) / 1000), reg: positioned });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: g, env: {} as NodeJS.ProcessEnv });
  const first = await cycle.run(reg, t); // the mode is still unknown: an ordinary cycle; its answer states the limit
  reg = first.registry;
  assert.equal(first.kind, "base");
  assert.equal(g.quotaMode(t), "watched_only");
  assert.equal(cycle.nextDelayMs(reg, t), 432_000, "the next tick is one gap away, not 5 s (the plane is airborne and 'active')");
  // tick every 30 s for a day, as the poller's timer would at worst
  const kinds = new Map<string, number>();
  for (t = T0 + TRACKED_POLL_MS; t < T0 + DAY_MS; t += TRACKED_POLL_MS) {
    const out = await cycle.run(reg, t);
    reg = out.registry;
    kinds.set(out.kind, (kinds.get(out.kind) ?? 0) + 1);
  }
  const gaps = p.calls.slice(1).map((c, i) => c.t - p.calls[i].t);
  assert.ok(gaps.every((d) => d >= 432_000), `every gap is at least 432 s (smallest ${Math.min(...gaps) / 1000} s)`);
  assert.ok(p.calls.length <= 500 * TRACKED_QUOTA_SHARE, `${p.calls.length} requests in a day, within 40% of 500`);
  assert.ok(p.calls.length >= 190, `the share is used (${p.calls.length})`);
  assert.equal(kinds.get("fast") ?? 0, 0, "no fast ticks while the quota sets the pace");
  assert.ok((kinds.get("quota-wait") ?? 0) > 2000, "the ticks in between send nothing");
  assert.ok(p.calls.every((c) => c.url === "https://api.adsb.lol/v2/reg/N843S"));
  assert.equal(cycle.stats().quota_paced_cycles, p.calls.length - 1);
});

test("tracked planes: a cycle that needs two requests (registration batch + hex lookup) is followed by two gaps", async () => {
  const g = govWithLimit(500);
  let t = T0;
  const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e" }] }; // parked: the batch returns nothing, so the hex is asked too
  const p = trackedProvider({ clock: () => t, headers: () => quotaHdr(500, 400) });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: g, env: {} as NodeJS.ProcessEnv });
  const out = await cycle.run(reg, t);
  assert.equal(out.requests, 2);
  assert.equal(cycle.nextDelayMs(reg, t), 864_000);
  t = T0 + 863_000;
  assert.equal((await cycle.run(reg, t)).kind, "quota-wait");
  assert.equal(p.calls.length, 2);
  t = T0 + 864_000;
  assert.equal((await cycle.run(reg, t)).kind, "base");
  assert.equal(p.calls.length, 4);
  const pace = cycle.pacing(reg, t);
  assert.deepEqual(
    { quota_mode: pace.quota_mode, paced: pace.paced, blocked: pace.blocked, effective_poll_s: pace.effective_poll_s, request_gap_s: pace.request_gap_s, last_cycle_requests: pace.last_cycle_requests },
    { quota_mode: "watched_only", paced: true, blocked: false, effective_poll_s: 864, request_gap_s: 432, last_cycle_requests: 2 });
  assert.match(pace.reason, /adsb\.lol daily quota 500\/day: the tracked planes get 40% of it, one request per 432 s/);
});

test("tracked planes: the key rides on both request kinds; with no limit read, or one large enough for the standard cadence, nothing is paced", async () => {
  // 500,000/day: 25% of it is a request every 0.69 s, more than the poller ever sends (16 a minute)
  for (const g of [gov(), govWithLimit(500_000)]) {
    let t = T0;
    const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e", last_heard: T0 - 1000 }] };
    const p = trackedProvider({ clock: () => t, headers: () => hdr({}) });
    const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: { "User-Agent": "ua" }, governor: g, env: { ADSBLOL_API_KEY: "k-123" } as NodeJS.ProcessEnv });
    await cycle.run(reg, t);
    assert.deepEqual(p.calls.map((c) => c.url), ["https://api.adsb.lol/v2/reg/N843S", "https://api.adsb.lol/v2/hex/ab8c8e"]);
    for (const c of p.calls) assert.deepEqual(c.headers, { "User-Agent": "ua", "x-api-key": "k-123" });
    assert.equal(cycle.nextDelayMs(reg, t), 5_000, "an active plane is still polled every 5 s");
    t += 5_000;
    assert.equal((await cycle.run(reg, t)).kind, "fast");
    const pace = cycle.pacing(reg, t);
    assert.equal(pace.paced, false);
    assert.equal(pace.effective_poll_s, 5);
  }
});

test("tracked planes: the quota's 429 stops the poller until the reset, where it resumes on its own", async () => {
  const g = gov();
  let t = T0;
  let spent = true;
  const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e" }] };
  const p = trackedProvider({
    clock: () => t, status: () => (spent ? 429 : 200), reg: positioned,
    headers: () => (spent ? SPENT_429(500, (MIDNIGHT - t) / 1000) : quotaHdr(500, 499, (MIDNIGHT + DAY_MS - t) / 1000)),
  });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: g, env: {} as NodeJS.ProcessEnv });
  await assert.rejects(() => cycle.run(reg, t), /reg batch 429/);
  assert.equal(p.calls.length, 1, "the registration batch found the quota spent; the hex lookup was not sent");
  assert.equal(g.quotaExhaustedUntil(t), MIDNIGHT);
  for (t = T0 + TRACKED_POLL_MS; t < MIDNIGHT; t += 20 * 60_000) {
    assert.equal((await cycle.run(reg, t)).kind, "skipped-quota");
    assert.equal(cycle.nextDelayMs(reg, t), Math.min(TRACKED_POLL_MS, MIDNIGHT - t));
  }
  assert.equal(p.calls.length, 1, "not one request in the four hours to the reset");
  const blocked = cycle.pacing(reg, MIDNIGHT - 60_000);
  assert.equal(blocked.blocked, true);
  assert.equal(blocked.effective_poll_s, null);
  assert.match(blocked.reason, /daily quota spent until 2026-11-18T00:00:00\.000Z/);
  assert.equal(cycle.nextDelayMs(reg, MIDNIGHT - 4_000), 4_000, "the tick lands on the reset");
  spent = false;
  t = MIDNIGHT;
  const back = await cycle.run(reg, t);
  assert.equal(back.kind, "base");
  assert.equal(back.points.length, 1);
  assert.equal(g.stats(t).consecutive_failures, 0, "the quota's 429 left no failure behind");
});

// ── the failover ─────────────────────────────────────────────────────────

const FPLAN: PlanDisc[] = [{ id: 0, lat: 40, lon: -74, radiusNm: 250, prior: 9 }, { id: 1, lat: 51, lon: 0, radiusNm: 250, prior: 5 }];
function failoverRig(g: UpstreamGovernor) {
  let clock = T0;
  const calls: Array<{ url: string; headers: Record<string, string> }> = [];
  const f = createSweepFailover({
    env: {} as NodeJS.ProcessEnv, governor: g, plan: FPLAN, start: false, now: () => clock, publish: () => {}, airportPoints: () => [],
    fetchImpl: (async (u: unknown, init: { headers: Record<string, string> }) => {
      calls.push({ url: String(u), headers: init.headers });
      return { ok: true, status: 200, headers: hdr({}), json: async () => ({ now: clock / 1000, aircraft: [{ hex: "a00001", lat: 40, lon: -74, seen_pos: 1 }], ac: [] }) };
    }) as never,
  });
  return { f, calls, at: (ms: number) => { clock = T0 + ms; return clock; } };
}

test("failover: runs while the quota keeps the sweep off adsb.lol, with the reason; idle when the sweep can run", async () => {
  await quiet(async () => {
    // watched_only: the sweep has not failed (it sends nothing), yet the world feed must be carried
    const w = failoverRig(govWithLimit(500));
    const made = await w.f.tick(T0);
    assert.equal(made?.provider, "adsbfi");
    assert.equal(w.f.status(T0).active, true);
    assert.equal(w.f.status(T0).reason, FAILOVER_REASON_WATCHED_ONLY);
    assert.equal(FAILOVER_REASON_WATCHED_ONLY, "adsb.lol daily quota: watched planes only");
    assert.equal("x-api-key" in w.calls[0].headers, false, "the adsb.lol key is never sent to another provider");

    // spent, on a limit that would otherwise allow the sweep
    const g = govWithLimit(7200);
    const s = failoverRig(g);
    assert.equal(await s.f.tick(T0), null, "full mode, sweep healthy: idle");
    assert.equal(s.f.status(T0).active, false);
    assert.equal(s.f.status(T0).reason, null);
    g.noteQuota(SPENT_429(7200, RESET_S - 1), T0 + 1_000, 429);
    g.noteResult("bg", 429, T0 + 1_000);
    assert.ok(await s.f.tick(s.at(2_000)));
    assert.equal(s.f.status(T0 + 2_000).reason, "adsb.lol daily quota spent until 2026-11-18T00:00:00.000Z");
    // the reset: the sweep may ask again, the failover stops
    assert.equal(await s.f.tick(s.at(MIDNIGHT - T0)), null);
    assert.equal(s.f.status(MIDNIGHT).active, false);
    assert.equal(s.f.status(MIDNIGHT).reason, null);
    assert.equal(s.f.status(MIDNIGHT).activations_total, 1);
  });
});

test("failover: the reason names what holds — failing sweep, watched planes only, or the spent quota (which wins)", () => {
  assert.equal(failoverReason(gov(), T0), null);
  assert.equal(failoverReason(govWithLimit(7200), T0), null);
  const failing = gov();
  failing.noteResult("bg", 503, T0);
  assert.equal(failoverReason(failing, T0), FAILOVER_REASON_FAILING);
  assert.equal(FAILOVER_REASON_FAILING, "adsb.lol sweep failing");
  const both = govWithLimit(500);
  both.noteResult("bg", 503, T0);
  assert.equal(failoverReason(both, T0), FAILOVER_REASON_WATCHED_ONLY, "the quota holds the sweep whatever its last failure count");
  both.noteQuota(SPENT_429(), T0, 429);
  assert.match(String(failoverReason(both, T0)), /^adsb\.lol daily quota spent until 2026-11-18T00:00:00\.000Z$/);
});

// ── the failover runs on demand ──────────────────────────────────────────

function demandRig(g: UpstreamGovernor) {
  let clock = T0;
  let lastAsked: number | null = null;
  const calls: string[] = [];
  const f = createSweepFailover({
    env: {} as NodeJS.ProcessEnv, governor: g, plan: FPLAN, start: false, now: () => clock, publish: () => {}, airportPoints: () => [],
    demand: () => lastAsked != null && clock - lastAsked <= FAILOVER_DEMAND_WINDOW_MS, lastDemandAt: () => lastAsked,
    fetchImpl: (async (u: unknown) => {
      calls.push(String(u));
      return { ok: true, status: 200, headers: hdr({}), json: async () => ({ now: clock / 1000, aircraft: [{ hex: "a00001", lat: 40, lon: -74, seen_pos: 1 }], ac: [] }) };
    }) as never,
  });
  return { f, calls, at: (ms: number) => { clock = T0 + ms; return clock; }, ask: () => { lastAsked = clock; } };
}

test("failover on demand: a reason but nobody asking -> no request to any provider, and the status says why it is idle", async () => {
  await quiet(async () => {
    assert.equal(FAILOVER_DEMAND_WINDOW_MS, 10 * 60_000);
    const r = demandRig(govWithLimit(500)); // watched planes only: a standing reason to run
    for (let ms = 0; ms <= 3_600_000; ms += 60_000) assert.equal(await r.f.tick(r.at(ms)), null);
    assert.deepEqual(r.calls, [], "an hour with a reason and no viewer: not one request");
    const st = r.f.status(r.at(3_600_000));
    assert.equal(st.active, false);
    assert.equal(st.demand, false);
    assert.equal(st.last_demand_at, null);
    assert.equal(st.reason, "adsb.lol daily quota: watched planes only; idle: nobody has asked for the worldwide feed in the last 10 min");
    assert.equal(st.requests_total, 0);
    assert.equal(st.activations_total, 0);
  });
});

test("failover on demand: the first request starts it, it runs while requests keep coming, and stops 10 minutes after the last", async () => {
  await quiet(async () => {
    const r = demandRig(govWithLimit(500));
    assert.equal(await r.f.tick(r.at(0)), null);
    r.at(60_000);
    r.ask(); // GET /api/data/aircraft/global
    assert.equal(r.f.status(r.at(60_000)).reason, FAILOVER_REASON_WATCHED_ONLY, "the answer to the request that IS the demand never says 'nobody has asked'");
    assert.ok(await r.f.tick(r.at(60_000)), "runs at once");
    assert.equal(r.f.status(r.at(60_000)).active, true);
    assert.equal(r.f.status(r.at(60_000)).last_demand_at, T0 + 60_000);
    // nine minutes later, nobody has asked again: still inside the window
    assert.ok(await r.f.tick(r.at(60_000 + 9 * 60_000)));
    // 10 min + 1 ms after the last request: it stops
    const n = r.calls.length;
    assert.equal(await r.f.tick(r.at(60_000 + FAILOVER_DEMAND_WINDOW_MS + 1)), null);
    assert.equal(await r.f.tick(r.at(60_000 + 30 * 60_000)), null);
    assert.equal(r.calls.length, n, "nothing after the demand lapsed");
    const st = r.f.status(r.at(60_000 + 30 * 60_000));
    assert.deepEqual({ active: st.active, demand: st.demand, activations_total: st.activations_total }, { active: false, demand: false, activations_total: 1 });
    assert.match(String(st.reason), /; idle: nobody has asked for the worldwide feed in the last 10 min$/);
    // a new request starts a second stretch
    r.ask();
    assert.ok(await r.f.tick(r.at(60_000 + 30 * 60_000)));
    assert.equal(r.f.status(r.at(60_000 + 30 * 60_000)).activations_total, 2);
  });
});

test("failover on demand: demand without a reason sends nothing (the adsb.lol sweep is doing its job)", async () => {
  const r = demandRig(govWithLimit(7200));
  r.ask();
  assert.equal(await r.f.tick(r.at(0)), null);
  assert.deepEqual(r.calls, []);
  assert.deepEqual({ reason: r.f.status(T0).reason, demand: r.f.status(T0).demand, active: r.f.status(T0).active }, { reason: null, demand: true, active: false });
});

test("failover on demand, wired: only GET /api/data/aircraft/global counts as demand, for 10 minutes", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-demand-"));
  let clock = Date.now();
  const app = express();
  const h = registerGlobalAircraftRoutes(app, {
    env: { GLOBAL_SWEEP_ENABLED: "0", AIRCRAFT_FAST_LANE: "0" } as NodeJS.ProcessEnv, now: () => clock,
    fetchImpl: (async () => { throw new Error("no network in tests"); }) as unknown as typeof fetch,
    sites: [], baseDir: dir, readFree: () => 50 * 1_073_741_824, ladd: new LaddRegistry({ file: null }),
  });
  const srv = app.listen(0);
  try {
    const base = `http://127.0.0.1:${(srv.address() as AddressInfo).port}`;
    assert.deepEqual({ demand: h.sweepFailover.status().demand, last: h.sweepFailover.status().last_demand_at }, { demand: false, last: null }, "nobody has asked since boot");
    await fetch(`${base}/api/data/aircraft/live/ab8c8e`);
    assert.equal(h.sweepFailover.status().demand, false, "one selected plane is not a request for the worldwide feed");
    const asked = clock;
    const d = await (await fetch(`${base}/api/data/aircraft/global`)).json() as { honesty: string; coverage: { sweep_failover: { demand: boolean; last_demand_at: number | null } } };
    assert.deepEqual(d.coverage.sweep_failover, { ...d.coverage.sweep_failover, demand: true, last_demand_at: asked });
    assert.match(d.honesty, /ON DEMAND: only while this endpoint has been requested in the last 10 min/);
    assert.match(d.honesty, /takes a minute or two to populate after the first request/);
    clock += FAILOVER_DEMAND_WINDOW_MS;
    assert.equal(h.sweepFailover.status().demand, true, "still wanted at exactly 10 minutes");
    clock += 1;
    assert.deepEqual({ demand: h.sweepFailover.status().demand, last: h.sweepFailover.status().last_demand_at }, { demand: false, last: asked });
  } finally {
    h.stop();
    srv.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── full mode fits the limit ─────────────────────────────────────────────

test("full mode at 7,200 a day: sweep 60% + tracked planes 25% + scopes 10% = 6,840 requests, 95% of the limit, the rest left for viewers", async () => {
  assert.equal(TRACKED_QUOTA_SHARE_FULL, 0.25);
  assert.equal(SCOPES_QUOTA_SHARE, 0.1);
  assert.ok(SWEEP_QUOTA_SHARE + TRACKED_QUOTA_SHARE_FULL + SCOPES_QUOTA_SHARE <= 0.95 + 1e-9);
  const LIMIT = 7200;
  const g = govWithLimit(LIMIT);
  assert.equal(g.sweepMinGapMs(T0), 20_000);
  assert.equal(g.trackedMinGapMs(T0), 48_000);
  assert.equal(g.scopesMinGapMs(T0), 120_000);
  assert.equal(gov().scopesMinGapMs(T0), 0, "no limit read: not paced");
  assert.equal(govWithLimit(500).scopesMinGapMs(T0), 0, "watched_only has its own rule (LADD alone)");

  // 1. the sweep, asked for a token every 500 ms for a day
  let sweep = 0;
  for (let t = T0; t < T0 + DAY_MS; t += 500) if (g.tryAcquireBackground(t)) sweep++;

  // 2. the tracked poller: one airborne (so "active") plane, ticked every 5 s for a day
  let t = T0;
  let reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e" }] };
  const p = trackedProvider({ clock: () => t, headers: () => quotaHdr(LIMIT, 5000, 40_000), reg: positioned });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: g, env: {} as NodeJS.ProcessEnv });
  for (t = T0; t < T0 + DAY_MS; t += 5_000) reg = (await cycle.run(reg, t)).registry;
  const tracked = p.calls.length;
  assert.equal(cycle.pacing(reg, T0 + DAY_MS).request_gap_s, 48);
  assert.match(cycle.pacing(reg, T0 + DAY_MS).reason, /7200\/day: the tracked planes get 25% of it, one request per 48 s/);

  // 3. the scopes poller, on its 120 s tick for a day
  const pacer = createScopesPacer(g);
  let scopes = 0;
  const cycleTimes: number[] = [];
  for (let u = T0; u < T0 + DAY_MS; u += GLOBAL_POLL_MS) {
    if (!pacer.cycleDue(u)) continue;
    cycleTimes.push(u);
    for (const scope of GLOBAL_SCOPES) if (pacer.decide(scope, u) === "ask") { pacer.asked(scope, u); scopes++; }
  }
  assert.ok(cycleTimes.slice(1).every((c, i) => c - cycleTimes[i] === 360_000), "all three scopes every 360 s");

  assert.ok(sweep <= LIMIT * SWEEP_QUOTA_SHARE, `sweep ${sweep} <= 4,320`);
  assert.ok(tracked <= LIMIT * TRACKED_QUOTA_SHARE_FULL, `tracked ${tracked} <= 1,800`);
  assert.ok(scopes <= LIMIT * SCOPES_QUOTA_SHARE, `scopes ${scopes} <= 720`);
  assert.ok(sweep + tracked + scopes <= 0.95 * LIMIT, `${sweep} + ${tracked} + ${scopes} = ${sweep + tracked + scopes} <= 6,840`);
  assert.ok(sweep + tracked + scopes >= 0.93 * LIMIT, `and the shares are used, not wasted (${sweep + tracked + scopes})`);
});

test("full mode, a limit between the shares: hex ticks still run between base cycles, each paid for by its gap", async () => {
  // 50,000/day: 25% = 12,500 requests, one per 6.912 s
  const g = govWithLimit(50_000);
  assert.equal(g.trackedMinGapMs(T0), 6_912);
  let t = T0;
  let reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e" }] };
  const p = trackedProvider({ clock: () => t, headers: () => quotaHdr(50_000, 40_000, 40_000), reg: positioned, hex: positioned });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: g, env: {} as NodeJS.ProcessEnv });
  const kinds: string[] = [];
  for (t = T0; t <= T0 + 60_000; t += 1_000) { const out = await cycle.run(reg, t); reg = out.registry; if (out.requests) kinds.push(out.kind); }
  // requests at 0, 7, 14, 21, 28, 35 (the base cycle is due again), 42, 49, 56 s
  assert.deepEqual(kinds, ["base", "fast", "fast", "fast", "fast", "base", "fast", "fast", "fast"]);
  const gaps = p.calls.slice(1).map((c, i) => c.t - p.calls[i].t);
  assert.ok(gaps.every((d) => d >= 6_912), `every request waits its gap (smallest ${Math.min(...gaps)} ms)`);
});

// ── the LADD privacy registry keeps learning ─────────────────────────────

test("LADD bit: read off the raw row (dbFlags & 8), kept on the mapped row without changing its JSON", () => {
  assert.equal(DB_FLAG_LADD, 8);
  assert.equal(rawRowIsLadd({ dbFlags: 8 }), true);
  assert.equal(rawRowIsLadd({ dbFlags: 9 }), true, "military + LADD");
  assert.equal(rawRowIsLadd({ dbFlags: 1 }), false, "military only");
  assert.equal(rawRowIsLadd({ dbFlags: 4 }), false, "PIA only");
  for (const junk of [{}, { dbFlags: "8" }, { dbFlags: 8.5 }, { dbFlags: null }, null, undefined, "x"]) assert.equal(rawRowIsLadd(junk), false);
  // the adsb.fi answer shape (measured 2026-11-17: 38 of 513 rows in one disc carried the bit)
  const fi = mapPointAircraft({ aircraft: [
    { hex: "a5ed66", r: "N481MC", t: "B744", flight: "CMB522  ", lat: 39.1, lon: -94.6, dbFlags: 8 },
    { hex: "ae1234", r: "02-1111", t: "C17", lat: 39.2, lon: -94.7, dbFlags: 1 },
    { hex: "a00001", lat: 39.3, lon: -94.8 },
  ] }, "aircraft", "adsbfi");
  assert.deepEqual(fi.map(rowIsLadd), [true, false, false]);
  const plain = mapPointAircraft({ ac: [{ hex: "a5ed66", r: "N481MC", t: "B744", flight: "CMB522  ", lat: 39.1, lon: -94.6 }] }, "ac", "adsbfi");
  assert.equal(JSON.stringify(fi[0]), JSON.stringify(plain[0]), "the serialized row is the same with or without the bit");
  assert.equal(rowIsLadd({ ...fi[0] }), true, "a copied row keeps it");
});

test("LADD registry learns from flagged rows of any provider, and only from them; an unflagged row proves nothing", () => {
  let clock = T0;
  const reg = new LaddRegistry({ file: null, now: () => clock });
  for (const [i, provider] of ["adsblol", "airplaneslive", "adsbfi"].entries()) {
    const rows = mapPointAircraft({ ac: [
      { hex: `aa000${i}`, flight: `LADD${i}   `, lat: 1, lon: 1, dbFlags: 8 },
      { hex: `bb000${i}`, flight: `OPEN${i}   `, lat: 1, lon: 1, dbFlags: 2 },
    ] }, "ac", provider);
    assert.equal(learnLaddFromRows(rows, reg), 1, `${provider}: one flagged row`);
    assert.equal(reg.isLadd(`aa000${i}`), true);
    assert.equal(reg.isLadd(null, `LADD${i}`), true, "by callsign too, as for the scope");
    assert.equal(reg.isLadd(`bb000${i}`, `OPEN${i}`), false);
  }
  assert.deepEqual(reg.size, { hexes: 3, callsigns: 3 });
  // a member already known stays known when a later row of it carries no bit
  clock += 3_600_000;
  assert.equal(learnLaddFromRows(mapPointAircraft({ ac: [{ hex: "aa0000", flight: "LADD0", lat: 1, lon: 1 }] }, "ac", "adsblol"), reg), 0);
  assert.equal(reg.isLadd("aa0000"), true);
});

test("LADD registry: a member learned from the bit is written to disk within 30 s, not once per fix batch; a scope sighting still saves at once", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-ladd-"));
  try {
    const file = path.join(dir, "ladd_registry.json");
    let clock = T0;
    const reg = new LaddRegistry({ file, now: () => clock });
    const saved = () => (fs.existsSync(file) ? Object.keys((JSON.parse(fs.readFileSync(file, "utf8")) as { hexes: Record<string, number> }).hexes).sort() : null);
    const flagged = (hex: string) => mapPointAircraft({ ac: [{ hex, lat: 1, lon: 1, dbFlags: 8 }] }, "ac", "adsbfi");
    learnLaddFromRows(flagged("aa0001"), reg);
    assert.deepEqual(saved(), ["aa0001"], "the first write is immediate (nothing saved in the last 30 s)");
    clock += 1_000;
    learnLaddFromRows(flagged("aa0002"), reg);
    clock += 1_000;
    learnLaddFromRows(flagged("aa0003"), reg);
    assert.deepEqual(saved(), ["aa0001"], "two batches a second apart: no rewrite each time");
    assert.equal(reg.isLadd("aa0003"), true, "but known in memory at once: FAA data is withheld from the first row");
    clock += LADD_FLAG_SAVE_MIN_MS;
    learnLaddFromRows(flagged("aa0001"), reg); // any later batch, even of a known member
    assert.deepEqual(saved(), ["aa0001", "aa0002", "aa0003"]);
    // the scope's own sighting of a new member: saved at once, as before
    clock += 1_000;
    reg.note([{ icao24: "aa0004" }]);
    assert.deepEqual(saved(), ["aa0001", "aa0002", "aa0003", "aa0004"]);
    assert.equal(new LaddRegistry({ file, now: () => clock }).isLadd("aa0002"), true, "survives a restart");
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("LADD registry is taught through the fix bus: a failover disc from adsb.fi, a viewport disc, the selected plane", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-laddbus-"));
  const ladd = new LaddRegistry({ file: null });
  const app = express();
  const h = registerGlobalAircraftRoutes(app, {
    env: { GLOBAL_SWEEP_ENABLED: "0", GLOBAL_SWEEP_ARCHIVE: "0", AIRCRAFT_FASTLANE_ARCHIVE: "0" } as NodeJS.ProcessEnv,
    fetchImpl: (async () => { throw new Error("no network in tests"); }) as unknown as typeof fetch,
    sites: [], baseDir: dir, readFree: () => 50 * 1_073_741_824, ladd,
  });
  const srv = app.listen(0);
  try {
    const now = Date.now();
    publishFixes({ provider: "adsbfi", origin: "sweep-failover", fetchedAt: now,
      aircraft: mapPointAircraft({ aircraft: [{ hex: "a5ed66", flight: "CMB522  ", lat: 39.1, lon: -94.6, dbFlags: 8 }, { hex: "a00001", lat: 39.3, lon: -94.8 }] }, "aircraft", "adsbfi") });
    publishFixes({ provider: "airplaneslive", origin: "viewport", fetchedAt: now, disc: { lat: 39, lon: -94, radiusNm: 250 },
      aircraft: mapPointAircraft({ ac: [{ hex: "ab8c8e", flight: "N843S   ", lat: 41.7, lon: -87.8, dbFlags: 8 }] }, "ac", "airplaneslive") });
    publishFixes({ provider: "adsblol", origin: "fastlane", fetchedAt: now,
      aircraft: mapPointAircraft({ ac: [{ hex: "a42dee", flight: "N369CA  ", lat: 41.7, lon: -87.8, dbFlags: 12, seen_pos: 0.5 }] }, "ac", "adsblol") });
    assert.equal(ladd.isLadd("a5ed66"), true, "adsb.fi, sweep failover");
    assert.equal(ladd.isLadd("ab8c8e"), true, "airplanes.live, a viewer's disc");
    assert.equal(ladd.isLadd("a42dee"), true, "adsb.lol, the selected plane's own row");
    assert.equal(ladd.isLadd("a00001"), false);
    const d = await (await fetch(`http://127.0.0.1:${(srv.address() as AddressInfo).port}/api/data/aircraft/global`)).json() as {
      coverage: { ladd: { hexes: number; callsigns: number; flagged_rows_total: number; learns_from: string } };
    };
    assert.deepEqual({ hexes: d.coverage.ladd.hexes, flagged_rows_total: d.coverage.ladd.flagged_rows_total }, { hexes: 3, flagged_rows_total: 3 });
    assert.match(d.coverage.ladd.learns_from, /\/v2\/ladd .* LADD bit \(dbFlags 8\) on any provider's rows/);
  } finally {
    h.stop();
    srv.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── health: the aircraft feed check does not depend on viewers ───────────

test("health: checks.feeds.feeds.aircraft reads the archive's newest hour file, which the 10-minute archive tick keeps writing with no viewer and no adsb.lol", async () => {
  // 1. what the check keys on: the newest hour file under <archive>/aircraft, dead past 3 h
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-health-"));
  try {
    fs.mkdirSync(path.join(dir, "aircraft"), { recursive: true });
    fs.writeFileSync(path.join(dir, "aircraft", "2026-11-17-19.jsonl"), "");
    assert.deepEqual(observeFeedDeadAir(T0, dir).feeds.aircraft, { silent_hours: 1, dead: false });
    assert.equal(observeFeedDeadAir(T0 + FEED_DEAD_AIR_HOURS * 3_600_000, dir).feeds.aircraft.dead, true);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  // 2. what keeps writing it on a day nobody looks: routes.ts's eager archive tick — every 10 minutes one
  //    region through the viewport provider chain, archived by fetchAircraft itself (not by the world sweep,
  //    its failover or the tracked poller)
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.match(routes, /const archiveTick = \(\) => \{\s+const r = ARCHIVE_TICK_REGIONS\[archiveTickIdx\+\+ % ARCHIVE_TICK_REGIONS\.length\];\s+fetchAircraft\(r\.lamin, r\.lamax, r\.lomin, r\.lomax\)/);
  assert.match(routes, /archiveTick\(\);\s+setInterval\(archiveTick, 10 \* 60_000\)/);
  const fetchAircraftSrc = routes.slice(routes.indexOf("async function fetchAircraft("), routes.indexOf("function getOrStartAircraftFetch("));
  assert.match(fetchAircraftSrc, /await fetchDiscs\(plan, \{\s+providers: PROVIDERS/);
  for (const key of ["adsblol", "airplaneslive", "adsbfi"]) assert.ok(fetchAircraftSrc.includes(`key: "${key}"`), `${key} is in the tick's chain`);
  assert.match(fetchAircraftSrc, /archiveAircraft\(aircraft, ARCHIVE_SITES\)/);
  // 3. and that chain answers while adsb.lol is kept for the watched planes (no request to it)
  const now = Date.now();
  const urls: string[] = [];
  const out = await fetchDiscs(planDiscs(33, 39, -100, -93, 1), {
    providers: CHAIN, governor: govWithLimit(500, now), sleep: async () => {},
    fetchImpl: (async (u: unknown) => { urls.push(String(u)); return { ok: true, status: 200, json: async () => ({ ac: [{ hex: "abc123", lat: 36, lon: -96 }] }) }; }) as never,
  });
  assert.equal(out.aircraft.length, 1);
  assert.ok(urls.length === 1 && urls[0].includes("airplanes.live"));
});

// ── the endpoint ─────────────────────────────────────────────────────────

test("GET /api/data/aircraft/global: coverage.sweep.governor.quota states the quota; the honesty text says what a small quota changes", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-quota-global-"));
  const app = express();
  const h = registerGlobalAircraftRoutes(app, {
    env: { GLOBAL_SWEEP_ENABLED: "0", SWEEP_FAILOVER_ENABLED: "0" } as NodeJS.ProcessEnv,
    fetchImpl: (async () => { throw new Error("no network in tests"); }) as unknown as typeof fetch,
    sites: [], baseDir: dir, readFree: () => 50 * 1_073_741_824,
  });
  const srv = app.listen(0);
  try {
    const d = await (await fetch(`http://127.0.0.1:${(srv.address() as AddressInfo).port}/api/data/aircraft/global`)).json() as {
      honesty: string; coverage: { sweep: { governor: { quota: Record<string, unknown> } }; sweep_failover: Record<string, unknown> };
    };
    const q = d.coverage.sweep.governor.quota;
    for (const k of ["limit", "remaining", "reset_at", "exhausted", "exhausted_until", "mode", "key_configured", "sweep_min_gap_ms", "quota_exhausted_total", "requests_skipped_quota"]) {
      assert.ok(k in q, `quota.${k} missing`);
    }
    assert.equal(typeof q.key_configured, "boolean");
    assert.ok("reason" in d.coverage.sweep_failover, "the failover's status carries its reason");
    assert.match(d.honesty, /daily request quota is spent or too small for a world sweep/);
    assert.match(d.honesty, /asked only for the tracked tails, Find lookups and its LADD scope/);
  } finally {
    h.stop();
    srv.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
