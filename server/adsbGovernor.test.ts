// FLIGHT PROGRAM B1 — the shared adsb.lol governor: background pacing,
// the total ceiling that makes the sweep yield to viewers, and backoff
// semantics. Clock-injected; no timers, no network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { UpstreamGovernor, isBackoffStatus, parseRetryAfter, governorFromEnv, DEFAULT_GLOBAL_SWEEP_RPS } from "./adsbGovernor";
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
