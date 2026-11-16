// FLIGHT PROGRAM B1 — the shared adsb.lol governor: background pacing,
// the total ceiling that makes the sweep yield to viewers, and backoff
// semantics. Clock-injected; no timers, no network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { AIMD_MIN_FRACTION, AIMD_STEP_FRACTION, FG_FAILURE_PAUSE_MS, FG_PAUSE_MAX_MS, outcomeKind, UpstreamGovernor, isBackoffStatus, parseRetryAfter, governorFromEnv, DEFAULT_GLOBAL_SWEEP_RPS } from "./adsbGovernor";
import { fetchDiscs, planDiscs, type DiscProvider } from "./aircraftTiling";
import { subscribeFixes, type FixBatch } from "./aircraftFixBus";

const T0 = 1_800_000_000_000;

test("background pacing: token bucket at bgRps, burst 1 — never a burst toward a free API", () => {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 100 });
  assert.equal(g.tryAcquireBackground(T0), true, "first token available");
  assert.equal(g.tryAcquireBackground(T0), false, "no burst");
  assert.equal(g.tryAcquireBackground(T0 + 2_000), false, "0.8 token after 2s");
  assert.equal(g.backgroundWaitMs(T0 + 2_000), 500, "exact wait to the next whole token");
  assert.equal(g.tryAcquireBackground(T0 + 2_500), true, "one request per 2.5s at 0.4 rps");
  // over 60s of continuous polling the lane gets ~24 requests, not more
  let n = 0;
  for (let t = T0 + 2_600; t <= T0 + 62_500; t += 100) if (g.tryAcquireBackground(t)) n++;
  assert.ok(n >= 23 && n <= 25, `~24 requests/minute at 0.4 rps (got ${n})`);
});

test("ceiling: viewers are NEVER delayed; the sweep yields when the shared window is full", () => {
  const g = new UpstreamGovernor({ bgRps: 10, ceilingRps: 0.5, windowMs: 60_000 }); // 30 req/min total
  for (let i = 0; i < 30; i++) g.noteRequest("fg", T0 + i * 100); // viewer burst — always recorded
  assert.equal(g.windowCount(T0 + 3_000), 30);
  assert.equal(g.tryAcquireBackground(T0 + 3_000), false, "window full -> background yields");
  const wait = g.backgroundWaitMs(T0 + 3_000);
  assert.ok(wait > 56_000 && wait <= 57_001, `waits until the oldest viewer request ages out (${wait})`);
  assert.equal(g.tryAcquireBackground(T0 + 60_001), true, "frees up as the window slides");
});

test("backoff: 429/5xx/network from EITHER lane pauses the background; Retry-After floors it", () => {
  const g = new UpstreamGovernor({ bgRps: 10, ceilingRps: 100, baseBackoffMs: 30_000 });
  g.noteResult("fg", 503, T0);                        // a viewport disc saw adsb.lol struggle
  assert.equal(g.tryAcquireBackground(T0 + 1_000), false);
  assert.equal(g.backgroundWaitMs(T0), 30_000);
  g.noteResult("bg", 429, T0 + 1_000, 120);           // sweep 429 with Retry-After: 120
  assert.ok(g.backgroundWaitMs(T0 + 1_000) >= 120_000, "Retry-After honored");
  g.noteResult("bg", 0, T0 + 2_000);                  // network error: 3rd failure -> 120s step
  assert.equal(g.stats(T0 + 2_000).consecutive_failures, 3);
  // a lucky viewport 200 must NOT reopen the floodgate
  g.noteResult("fg", 200, T0 + 3_000);
  assert.ok(g.backgroundWaitMs(T0 + 3_000) > 0, "foreground success does not clear sweep backoff");
  // only a background success clears it
  g.noteResult("bg", 200, T0 + 200_000);
  assert.equal(g.stats(T0 + 200_000).consecutive_failures, 0);
  assert.equal(g.stats(T0 + 200_000).backoff_s, 0);
});

test("backoff is exponential and capped", () => {
  const g = new UpstreamGovernor({ bgRps: 1, ceilingRps: 100, baseBackoffMs: 1_000, maxBackoffMs: 8_000 });
  const waits: number[] = [];
  for (let i = 0; i < 6; i++) { g.noteResult("bg", 500, T0); waits.push(g.backgroundWaitMs(T0)); }
  // backoffUntil is a max() so monotone; steps 1,2,4,8,8,8 s
  assert.deepEqual(waits, [1_000, 2_000, 4_000, 8_000, 8_000, 8_000]);
});

test("isBackoffStatus / parseRetryAfter", () => {
  assert.equal(isBackoffStatus(429), true);
  assert.equal(isBackoffStatus(502), true);
  assert.equal(isBackoffStatus(0), true, "network error");
  assert.equal(isBackoffStatus(404), false, "a 404 is not upstream pressure");
  assert.equal(parseRetryAfter("30"), 30);
  assert.equal(parseRetryAfter(null), null);
  assert.equal(parseRetryAfter("garbage"), null);
  const d = new Date(T0 + 45_000).toUTCString();
  assert.equal(Math.round(parseRetryAfter(d, T0)!), 45);
});

test("governorFromEnv: env-tunable, clamped, default conservative", () => {
  assert.equal(governorFromEnv({} as any).bgRps, DEFAULT_GLOBAL_SWEEP_RPS);
  assert.equal(DEFAULT_GLOBAL_SWEEP_RPS, 0.4);
  assert.equal(governorFromEnv({ GLOBAL_SWEEP_RPS: "0.1" } as any).bgRps, 0.1);
  assert.equal(governorFromEnv({ GLOBAL_SWEEP_RPS: "50" } as any).bgRps, 2, "clamped");
  assert.equal(governorFromEnv({ GLOBAL_SWEEP_RPS: "0" } as any).backgroundWaitMs(T0), Number.POSITIVE_INFINITY, "0 = sweep off");
});

// ── the viewport chain RECORDS on the governor and PUBLISHES on the bus ──
const CHAIN: DiscProvider[] = [
  { key: "adsblol", label: "adsb.lol", arr: "ac", url: (la, lo, r) => `https://api.adsb.lol/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${r}` },
  { key: "airplaneslive", label: "airplanes.live", arr: "ac", url: (la, lo, r) => `https://api.airplanes.live/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${r}` },
];

test("fetchDiscs: adsb.lol requests + a 429 land on the governor; other providers do not", async () => {
  const g = new UpstreamGovernor({ bgRps: 1, ceilingRps: 100 });
  const plan = planDiscs(-1.5, 1.5, -4, 4); // 2 discs
  await fetchDiscs(plan, {
    providers: CHAIN, governor: g, sleep: async () => {},
    fetchImpl: (async (u: unknown) => {
      const s = String(u);
      if (s.includes("adsb.lol") && s.includes("/2.000/")) {
        return { ok: false, status: 429, headers: { get: (k: string) => (k === "retry-after" ? "90" : null) } } as any;
      }
      return { ok: true, status: 200, json: async () => ({ ac: [{ hex: "x", lat: 1, lon: 1 }] }) } as any;
    }) as any,
  });
  const st = g.stats(Date.now());
  assert.equal(st.totals.fg, 2, "both adsb.lol disc attempts recorded (the fallback's is not)");
  assert.equal(st.totals.fgFail, 1);
  assert.ok(st.backoff_s >= 89, "viewport 429 + Retry-After pauses the background sweep");
});

test("fetchDiscs publishes each successful disc on the fix bus (the global snapshot reuses it)", async () => {
  const got: FixBatch[] = [];
  const unsub = subscribeFixes((b) => got.push(b));
  try {
    const plan = planDiscs(39.5, 40.5, -75.5, -74.5);
    await fetchDiscs(plan, {
      providers: CHAIN, governor: null, sleep: async () => {},
      fetchImpl: (async () => ({ ok: true, status: 200, json: async () => ({ now: 1790000000000, ac: [{ hex: "abc", lat: 40, lon: -75, seen_pos: 2 }] }) })) as any,
    });
  } finally { unsub(); }
  assert.equal(got.length, 1);
  assert.equal(got[0].origin, "viewport");
  assert.equal(got[0].provider, "adsblol");
  assert.equal(got[0].upstreamNowMs, 1790000000000);
  assert.deepEqual(got[0].disc, { lat: 40, lon: -75, radiusNm: 50 });
  assert.equal(got[0].aircraft[0].icao24, "abc");
});

test("ops window: last-hour outcomes per lane split 429 / 5xx / network / other; older outcomes age out", () => {
  const g = new UpstreamGovernor({ bgRps: 1, ceilingRps: 5 });
  const T = 1_800_000_000_000;
  g.noteResult("bg", 200, T);
  g.noteResult("bg", 429, T + 1);
  g.noteResult("bg", 429, T + 2);
  g.noteResult("bg", 503, T + 3);
  g.noteResult("fg", 0, T + 4);
  g.noteResult("fg", 404, T + 5);
  g.noteResult("fg", 200, T + 6);
  const h = g.stats(T + 10).last_hour;
  assert.deepEqual(h.bg, { results: 4, ok: 1, rate_limited: 2, server_error: 1, network_error: 0, other: 0 });
  assert.deepEqual(h.fg, { results: 3, ok: 1, rate_limited: 0, server_error: 0, network_error: 1, other: 1 });
  assert.equal(g.stats(T + 3600_000 + 4).last_hour.bg.results, 0, "an hour later the bg outcomes are gone");
  assert.equal(g.stats(T + 3600_000 + 4).last_hour.fg.results, 2);
  assert.equal(outcomeKind(302), "ok");
});

test("AIMD (ops auto-fix): background 429s halve the sweep rate to a floor; successes add it back; 5xx/network and foreground 429s never cut it", () => {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 5 });
  const T = 1_800_000_000_000;
  assert.equal(g.effectiveBgRps(), 0.4);
  g.noteResult("bg", 429, T);
  assert.equal(g.effectiveBgRps(), 0.2);
  for (let i = 0; i < 10; i++) g.noteResult("bg", 429, T + i);
  assert.ok(Math.abs(g.effectiveBgRps() - 0.4 * AIMD_MIN_FRACTION) < 1e-9, "floored, never zero");
  g.noteResult("bg", 503, T + 20); g.noteResult("bg", 0, T + 21); g.noteResult("fg", 429, T + 22);
  assert.ok(Math.abs(g.effectiveBgRps() - 0.04) < 1e-9, "only a background 429 cuts the rate");
  for (let i = 0; i < 30; i++) g.noteResult("bg", 200, T + 100 + i);
  assert.equal(g.effectiveBgRps(), 0.4, "recovers to the configured rate, never above");
  assert.equal(g.stats(T + 200).effective_bg_rps, 0.4);
  // the token wait follows the effective rate once the backoff has expired
  const h = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 5, baseBackoffMs: 1 });
  assert.equal(h.tryAcquireBackground(T), true);
  h.noteResult("bg", 429, T + 1);
  const w = h.backgroundWaitMs(T + 2);
  assert.ok(w > 4_000 && w <= 5_000, `0.2 req/s -> ~5 s per token (got ${w})`);
  assert.ok(AIMD_STEP_FRACTION > 0);
});

// ── 2026-11-16 [REPAIR]: foreground failures pinned the sweep at its 15-minute ceiling ──
// Read live from /api/data/aircraft/global that day: consecutive_failures 105 -> 127 in four minutes, every one
// from the foreground lane; backoff_s pinned at ~900; window_bg 0 (the sweep sent nothing, its last request 24
// minutes old); 1 of ~1,100 snapshot rows under 2 minutes old. The selected-aircraft lane was failing in bursts
// (25 of 57 polls), each failure re-arming a 15-minute pause that only a sweep success could clear.

/** Drive a governor through a foreground outcome trace while a sweep loop polls it every 250 ms (globalSweep's
 *  own cadence), every sweep request succeeding. Returns the times the sweep sent a request. */
function runSweep(g: UpstreamGovernor, fromMs: number, toMs: number, fgFailureAt: (t: number) => boolean, fgEveryMs: number): number[] {
  const sent: number[] = [];
  let nextFg = fromMs;
  for (let t = fromMs; t <= toMs; t += 250) {
    if (t >= nextFg) {
      g.noteRequest("fg", t);
      g.noteResult("fg", fgFailureAt(t) ? 0 : 200, t);
      nextFg = t + fgEveryMs;
    }
    if (g.tryAcquireBackground(t)) { sent.push(t); g.noteResult("bg", 200, t); }
  }
  return sent;
}

test("the 11-16 trace: a burst of foreground failures no longer stops the sweep for 15 minutes", () => {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 });
  // the selected-aircraft lane polls every 3 s; it fails for the first 100 s, then answers for 10 minutes
  const sent = runSweep(g, T0, T0 + 700_000, (t) => t < T0 + 100_000, 3_000);
  const during = sent.filter((t) => t < T0 + 100_000);
  const after = sent.filter((t) => t >= T0 + 100_000);
  assert.ok(during.length <= 2, `during the burst the sweep at most probes (${during.length})`);
  const firstAfter = after[0];
  assert.ok(firstAfter - (T0 + 99_000) <= FG_FAILURE_PAUSE_MS + 250, `the sweep is back within 30 s of the last foreground failure (${(firstAfter - T0) / 1000}s)`);
  // 10 clean minutes at 0.4 req/s, shared with the lane's 20 req/min under the 60/min ceiling: ~240 requests
  assert.ok(after.length >= 200, `the sweep runs at its rate once the burst is over (${after.length} requests in 10 min)`);
  assert.equal(g.stats(T0 + 700_000).backoff_s, 0);
});

test("foreground failures that never stop: the sweep still gets through every two minutes, and its own failure backs off on its own count", () => {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 });
  // 20 minutes of a foreground failure every 3 s (400 of them), every sweep request answered
  const sent = runSweep(g, T0, T0 + 20 * 60_000, () => true, 3_000);
  // each two-minute bound lets one request out, plus the one more that fits before the next foreground failure
  assert.ok(sent.length >= 9 && sent.length <= 22, `a trickle, about ${20 * 60_000 / FG_PAUSE_MAX_MS} openings in 20 min (${sent.length} requests; an unpaused sweep sends ~480)`);
  for (let i = 1; i < sent.length; i++) {
    assert.ok(sent[i] - sent[i - 1] <= FG_PAUSE_MAX_MS + 3_500, `never held longer than the bound (${(sent[i] - sent[i - 1]) / 1000}s)`);
  }
  const st = g.stats(T0 + 20 * 60_000);
  assert.ok(st.backoff_s <= FG_PAUSE_MAX_MS / 1000, `the hold is the bounded pause, never the 15-minute ceiling (${st.backoff_s}s)`);
  assert.equal(st.sweep_backoff_s, 0, "the sweep's own backoff was never started: its requests all succeeded");
  assert.equal(st.sweep_consecutive_failures, 0);
  // 400 foreground failures behind it, the sweep's first own failure is step one (30 s), not the ceiling
  const t = T0 + 21 * 60_000;
  g.noteResult("bg", 503, t);
  assert.equal(g.stats(t).sweep_backoff_s, 30);
  g.noteResult("bg", 0, t + 1);
  assert.equal(g.stats(t + 1).sweep_backoff_s, 60, "and doubles on its own count");
  assert.equal(g.stats(t + 1).sweep_consecutive_failures, 2);
});

test("a real outage: both lanes fail, the sweep's probes thin out to the 15-minute ceiling and a foreground success never clears it", () => {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 });
  const sent: number[] = [];
  for (let t = T0; t <= T0 + 60 * 60_000; t += 250) {
    if ((t - T0) % 3_000 === 0) { g.noteRequest("fg", t); g.noteResult("fg", 0, t); }
    if (g.tryAcquireBackground(t)) { sent.push(t); g.noteResult("bg", 0, t); }
  }
  // 30 s, 60 s, 2, 4, 8, then 15 min steps: 8 probes in the hour, not a sweep
  assert.ok(sent.length >= 6 && sent.length <= 9, `probes only (${sent.length} in an hour)`);
  const gaps = sent.slice(1).map((t, i) => t - sent[i]);
  assert.ok(gaps[gaps.length - 1] >= 14 * 60_000, "the last gap is at the ceiling");
  const end = T0 + 60 * 60_000;
  g.noteResult("fg", 200, end);
  assert.ok(g.stats(end).sweep_backoff_s > 0, "a foreground success does not clear the sweep's own backoff");
  g.noteResult("bg", 200, end + 1);
  assert.equal(g.stats(end + 1).backoff_s, 0, "a sweep success does");
  assert.equal(g.stats(end + 1).consecutive_failures, 0);
});

test("a foreground Retry-After is honored in full, past the two-minute bound; the stats name both holds", () => {
  const g = new UpstreamGovernor({ bgRps: 10, ceilingRps: 100 });
  assert.equal(g.tryAcquireBackground(T0), true);
  g.noteResult("bg", 200, T0);
  g.noteResult("fg", 429, T0 + 1_000, 300); // "retry after 5 minutes", said to a viewer's request
  const st = g.stats(T0 + 1_000);
  assert.equal(st.viewer_pause_s, 300);
  assert.equal(st.sweep_backoff_s, 0);
  assert.equal(st.backoff_s, 300, "backoff_s is the whole hold");
  assert.equal(g.tryAcquireBackground(T0 + 200_000), false, "still held after the two-minute bound");
  assert.equal(g.tryAcquireBackground(T0 + 301_001), true);
  // a plain foreground failure: 30 s, and the constants are the documented ones
  const h = new UpstreamGovernor({ bgRps: 10, ceilingRps: 100 });
  h.noteResult("fg", 0, T0);
  assert.equal(h.stats(T0).viewer_pause_s, 30);
  assert.equal(FG_FAILURE_PAUSE_MS, 30_000);
  assert.equal(FG_PAUSE_MAX_MS, 120_000);
});
