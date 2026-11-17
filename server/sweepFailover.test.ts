// Sweep failover (2026-11-16): while adsb.lol is not answering the world
// sweep, the plan's discs are asked of adsb.fi and airplanes.live. These
// tests pin when it runs, what it publishes, how it paces and backs off, and
// that the worldwide snapshot keeps the rows with their own provider.
// Run: npx tsx --test server/sweepFailover.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  createSweepFailover, failoverBackoffMs, failoverOffReason, failoverRpsFromEnv,
  FAILOVER_ORIGIN, FAILOVER_PROVIDERS, FAILOVER_BACKOFF_BASE_MS, FAILOVER_BACKOFF_MAX_MS,
  FAILOVER_REFUSED_BACKOFF_MS, FAILOVER_DISC_MAX_FAILS, DEFAULT_FAILOVER_RPS, MAX_FAILOVER_RPS,
  FAILOVER_LIVE_SEED_MIN_ROWS, FAILOVER_LARGE_AIRPORT_WEIGHT, weightPerDisc,
} from "./sweepFailover";
import { majorAirportPoints } from "./airportsIndex";
import { worldPlan } from "./globalDiscPlan";
import { UpstreamGovernor } from "./adsbGovernor";
import { GlobalSnapshot } from "./globalSnapshot";
import { ARCHIVED_ORIGINS } from "./globalAircraft";
import { NON_COMMERCIAL_AIRCRAFT_PROVIDERS } from "./providerCompliance";
import type { FixBatch } from "./aircraftFixBus";
import type { PlanDisc } from "./globalDiscPlan";

const here = path.dirname(fileURLToPath(import.meta.url));
const T0 = 1_794_880_000_000;
const PLAN: PlanDisc[] = [
  { id: 0, lat: 40, lon: -74, radiusNm: 250, prior: 9 },
  { id: 1, lat: 51, lon: 0, radiusNm: 250, prior: 5 },
  { id: 2, lat: -33, lon: 151, radiusNm: 250, prior: 1 },
];
const ENV = {} as NodeJS.ProcessEnv;
const hdr = (h: Record<string, string> = {}) => ({ get: (k: string) => h[k.toLowerCase()] ?? null });
const res = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  ({ ok: status >= 200 && status < 300, status, headers: hdr(headers), json: async () => body }) as unknown as Response;
/** adsb.fi answers in its own shape: `aircraft` + `now` in SECONDS */
const fiBody = (n: number, nowMs: number) => ({
  now: nowMs / 1000,
  aircraft: Array.from({ length: n }, (_, i) => ({ hex: `a${i.toString(16).padStart(5, "0")}`, lat: 40 + i * 0.01, lon: -74, alt_baro: 30000, gs: 400, track: 90, seen_pos: 1, t: "B738" })),
});
/** airplanes.live answers in the readsb shape: `ac` + `now` in ms */
const liveBody = (n: number, nowMs: number) => ({
  now: nowMs,
  ac: Array.from({ length: n }, (_, i) => ({ hex: `b${i.toString(16).padStart(5, "0")}`, lat: 51 + i * 0.01, lon: 0, alt_baro: 20000, gs: 300, track: 180, seen_pos: 2 })),
});

/** a governor whose sweep has just failed `n` times in a row */
function failingGovernor(n = 1, at = T0 - 1): UpstreamGovernor {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 100 });
  for (let i = 0; i < n; i++) g.noteResult("bg", 0, at);
  return g;
}

function harness(o: {
  governor?: UpstreamGovernor;
  env?: NodeJS.ProcessEnv;
  answer?: (url: string) => Response | Promise<Response>;
  plan?: PlanDisc[];
  timeoutMs?: number;
  livePositions?: () => Array<{ lat: number; lon: number }>;
  airportPoints?: () => Array<{ lat: number; lon: number; large: boolean }>;
} = {}) {
  let clock = T0;
  const calls: string[] = [];
  const batches: FixBatch[] = [];
  const f = createSweepFailover({
    env: o.env ?? ENV, governor: o.governor ?? failingGovernor(), plan: o.plan ?? PLAN, start: false,
    now: () => clock, publish: (b) => { batches.push(b); }, timeoutMs: o.timeoutMs,
    livePositions: o.livePositions, airportPoints: o.airportPoints ?? (() => []),
    fetchImpl: (async (u: unknown) => {
      calls.push(String(u));
      if (o.answer) return o.answer(String(u));
      return String(u).includes("adsb.fi") ? res(fiBody(3, clock)) : res(liveBody(2, clock));
    }) as unknown as typeof fetch,
  });
  return { f, calls, batches, at: (ms: number) => { clock = T0 + ms; return clock; } };
}
const quiet = async <T>(fn: () => Promise<T>): Promise<T> => {
  const orig = console.error;
  console.error = () => {};
  try { return await fn(); } finally { console.error = orig; }
};

test("idle while the adsb.lol sweep is healthy: no request to another provider", async () => {
  const healthy = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 100 });
  healthy.noteResult("bg", 200, T0 - 1);
  const h = harness({ governor: healthy });
  assert.equal(await h.f.tick(h.at(0)), null);
  assert.equal(h.calls.length, 0);
  const s = h.f.status(T0);
  assert.equal(s.enabled, true);
  assert.equal(s.active, false);
  assert.equal(s.activations_total, 0);
});

test("a failed sweep request starts it: the busiest disc is asked of adsb.fi and published with that provider", async () => {
  const h = harness();
  const r = await quiet(() => h.f.tick(h.at(0)));
  assert.deepEqual(r, { provider: "adsbfi", ok: true, status: 200, count: 3 });
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0], "https://opendata.adsb.fi/api/v2/lat/40.000/lon/-74.000/dist/250", "prior 9 goes first");
  assert.equal(h.batches.length, 1);
  const b = h.batches[0];
  assert.equal(b.provider, "adsbfi");
  assert.equal(b.origin, FAILOVER_ORIGIN);
  assert.deepEqual(b.disc, { lat: 40, lon: -74, radiusNm: 250 });
  assert.equal(b.upstreamNowMs, T0, "adsb.fi's seconds clock arrives in milliseconds");
  assert.ok(b.aircraft.every((a) => a.provider === "adsbfi"), "every row carries the provider that served it");
  // the worldwide snapshot keeps them, as adsb.fi rows (never in the lawful subset)
  const snap = new GlobalSnapshot();
  assert.equal(snap.ingest(b, T0), 3);
  assert.equal(snap.rows({ lawfulOnly: true }).length, 0);
  assert.deepEqual(snap.summary(T0).rows_by_path, { [FAILOVER_ORIGIN]: 3 });
  const s = h.f.status(T0);
  assert.equal(s.active, true);
  assert.equal(s.activations_total, 1);
  assert.equal(s.discs_answered_ever, 1);
  assert.equal(s.discs_refreshed_last_cycle, 1);
  assert.equal(s.discs_in_plan, 3);
});

test("pace: each provider waits its own gap, the two alternate, and discs are taken in value order", async () => {
  const h = harness();
  await quiet(async () => {
    assert.equal((await h.f.tick(h.at(0)))?.provider, "adsbfi");
    // adsb.fi's 2 s gap has not elapsed: the other provider takes the next disc
    assert.equal((await h.f.tick(h.at(100)))?.provider, "airplaneslive");
    // neither is due
    assert.equal(await h.f.tick(h.at(1_000)), null);
    assert.equal((await h.f.tick(h.at(2_000)))?.provider, "adsbfi");
    assert.equal(await h.f.tick(h.at(2_050)), null, "airplanes.live is due only at 2,100");
  });
  assert.equal(h.calls.length, 3);
  assert.match(h.calls[1], /^https:\/\/api\.airplanes\.live\/v2\/point\/51\.000\/0\.000\/250$/);
  assert.match(h.calls[2], /lat\/-33\.000\/lon\/151\.000/, "the third disc, not a repeat");
  assert.equal(h.batches[1].provider, "airplaneslive");
  assert.equal(h.batches[1].upstreamNowMs, T0 + 100, "a milliseconds clock passes through");
  const s = h.f.status(T0 + 2_050);
  assert.equal(s.requests_total, 3);
  assert.equal(s.ok_total, 3);
  assert.equal(s.discs_refreshed_last_cycle, 3);
  assert.equal(s.rps_per_provider, DEFAULT_FAILOVER_RPS);
});

test("order: discs start from where the snapshot's aircraft are, not from the plan's regional prior", async () => {
  // 1,200 live positions over Sydney (plan prior 1, the LAST disc by prior), a handful over London
  const live = [
    ...Array.from({ length: FAILOVER_LIVE_SEED_MIN_ROWS + 200 }, (_, i) => ({ lat: -33 + (i % 10) * 0.01, lon: 151 })),
    ...Array.from({ length: 5 }, () => ({ lat: 51, lon: 0 })),
  ];
  const h = harness({ livePositions: () => live });
  await quiet(async () => {
    await h.f.tick(h.at(0));
    await h.f.tick(h.at(100));
    await h.f.tick(h.at(2_000));
  });
  assert.match(h.calls[0], /lat\/-33\.000\/lon\/151\.000/, "the disc holding the aircraft goes first");
  assert.match(h.calls[1], /point\/51\.000\/0\.000/);
  assert.match(h.calls[2], /lat\/40\.000\/lon\/-74\.000/, "prior 9 but no aircraft seen there: last");
  const s = h.f.status(T0 + 2_000);
  assert.equal(s.order_from, "live");
  assert.equal(s.order_set_at, T0);
});

test("order: a server that booted with adsb.lol already down starts from the large and medium airports", async () => {
  const airports = [
    { lat: 51.47, lon: -0.45, large: true }, { lat: 51.15, lon: -0.18, large: true }, // two large near London
    { lat: -33.95, lon: 151.18, large: true },                                        // one large near Sydney
    { lat: 40.6, lon: -73.8, large: false },                                          // one medium near New York
  ];
  // the snapshot is nearly empty (under the floor), so it is not used
  const h = harness({ livePositions: () => [{ lat: 40, lon: -74 }], airportPoints: () => airports });
  await quiet(async () => {
    await h.f.tick(h.at(0));
    await h.f.tick(h.at(100));
    await h.f.tick(h.at(2_000));
  });
  assert.match(h.calls[0], /lat\/51\.000\/lon\/0\.000/, `2 large = ${2 * FAILOVER_LARGE_AIRPORT_WEIGHT}`);
  assert.match(h.calls[1], /point\/-33\.000\/151\.000/);
  assert.match(h.calls[2], /lat\/40\.000\/lon\/-74\.000/);
  assert.equal(h.f.status(T0 + 2_000).order_from, "airports");
  // with neither, the plan's own prior stands
  const bare = harness();
  await quiet(() => bare.f.tick(bare.at(0)));
  assert.equal(bare.f.status(T0).order_from, "plan");
});

test("weightPerDisc: a point counts for every disc that holds it, and only those", () => {
  const w = weightPerDisc(PLAN, [
    { lat: 40, lon: -74 }, { lat: 43.9, lon: -74 }, { lat: 44.5, lon: -74 }, // 0, 234 and 270 nm north of disc 0
    { lat: 51, lon: 0, w: 3 }, { lat: NaN, lon: 0 }, { lat: 0, lon: 0 },
  ]);
  assert.deepEqual(Array.from(w.entries()).sort(), [[0, 2], [1, 3]]);
  // the repo's own plan and airports: the busiest discs by this measure are over land with airports
  const real = weightPerDisc(worldPlan(), majorAirportPoints().map((a) => ({ lat: a.lat, lon: a.lon, w: a.large ? 3 : 1 })));
  assert.ok(real.size > 300, `discs holding a large or medium airport: ${real.size}`);
});

test("it stops the moment the adsb.lol sweep answers again, and counts the stretch", async () => {
  const g = failingGovernor(3);
  const h = harness({ governor: g });
  await quiet(async () => {
    assert.ok(await h.f.tick(h.at(0)));
    g.noteResult("bg", 200, h.at(5_000)); // the sweep's own probe succeeded
    assert.equal(await h.f.tick(h.at(5_000)), null);
  });
  assert.equal(h.calls.length, 1);
  const s = h.f.status(T0 + 60_000);
  assert.equal(s.active, false);
  assert.equal(s.active_since, null);
  assert.equal(s.active_s_total, 5);
  assert.equal(s.activations_total, 1);
});

test("a viewer-lane failure alone does not start it (the sweep itself has not failed)", async () => {
  const g = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 100 });
  g.noteResult("fg", 0, T0 - 1);
  const h = harness({ governor: g });
  assert.equal(await h.f.tick(h.at(0)), null);
  assert.equal(h.calls.length, 0);
});

test("a provider that fails backs off alone and its disc goes to the other provider at once", async () => {
  const h = harness({ answer: (u) => (u.includes("adsb.fi") ? res(null, 503) : res(liveBody(4, T0))) });
  await quiet(async () => {
    const r1 = await h.f.tick(h.at(0));
    assert.deepEqual(r1, { provider: "adsbfi", ok: false, status: 503, count: 0, error: "adsbfi 503" });
    const r2 = await h.f.tick(h.at(10));
    assert.equal(r2?.provider, "airplaneslive");
    assert.equal(r2?.ok, true);
    // adsb.fi stays out for its 30 s; airplanes.live carries on at its own pace
    assert.equal((await h.f.tick(h.at(2_100)))?.provider, "airplaneslive");
    assert.equal((await h.f.tick(h.at(29_000)))?.provider, "airplaneslive");
    assert.equal((await h.f.tick(h.at(30_500)))?.provider, "adsbfi", "asked again after its backoff");
  });
  assert.match(h.calls[1], /point\/40\.000\/-74\.000\/250/, "the SAME disc, from the other provider");
  const fi = h.f.status(T0 + 31_000).providers.find((p) => p.key === "adsbfi");
  assert.ok(fi);
  assert.equal(fi.failed, 2);
  assert.equal(fi.consecutive_failures, 2);
  assert.equal(fi.last_status, 503);
  assert.ok(fi.backoff_s > 55 && fi.backoff_s <= 60, "second failure in a row: 60 s");
});

test("backoff: 30 s doubling to 15 min, Retry-After honored in full, a refusal waits 30 min at once", async () => {
  assert.equal(failoverBackoffMs(503, 1, null), FAILOVER_BACKOFF_BASE_MS);
  assert.equal(failoverBackoffMs(0, 2, null), 60_000);
  assert.equal(failoverBackoffMs(429, 3, null), 120_000);
  assert.equal(failoverBackoffMs(500, 20, null), FAILOVER_BACKOFF_MAX_MS);
  assert.equal(failoverBackoffMs(429, 1, 300_000), 300_000, "the provider's own figure wins when longer");
  assert.equal(failoverBackoffMs(429, 4, 5_000), 240_000, "and never shortens ours");
  assert.equal(failoverBackoffMs(403, 1, null), FAILOVER_REFUSED_BACKOFF_MS);
  assert.equal(failoverBackoffMs(401, 1, null), FAILOVER_REFUSED_BACKOFF_MS);

  // live: airplanes.live answered this session's network 403 "Please contact us" (2026-11-17)
  const h = harness({
    answer: (u) => (u.includes("airplanes.live") ? res({ error: "Please contact us" }, 403)
      : res(null, 429, { "retry-after": "120" })),
  });
  await quiet(async () => {
    assert.equal((await h.f.tick(h.at(0)))?.status, 429);
    assert.equal((await h.f.tick(h.at(10)))?.status, 403);
    assert.equal(await h.f.tick(h.at(60_000)), null, "both are waiting");
    assert.equal((await h.f.tick(h.at(120_500)))?.provider, "adsbfi", "Retry-After: 120 s, not our 30 s");
  });
  const s = h.f.status(T0 + 121_000);
  const live = s.providers.find((p) => p.key === "airplaneslive");
  const fi = s.providers.find((p) => p.key === "adsbfi");
  assert.ok(live && fi);
  assert.equal(live.refused, 1);
  assert.ok(live.backoff_s > 27 * 60, "refused: half an hour");
  assert.equal(fi.rate_limited, 2);
  assert.equal(h.batches.length, 0, "a failed answer publishes nothing");
  assert.equal(s.discs_refreshed_last_cycle, 0, "and is never counted as a refreshed disc");
});

test("one disc a provider cannot answer never holds the queue", async () => {
  // only adsb.fi, and it rejects the first disc itself (400)
  const calls: string[] = [];
  let clock = T0;
  const only = createSweepFailover({
    env: ENV, governor: failingGovernor(), plan: PLAN, start: false, providers: [FAILOVER_PROVIDERS[0]],
    now: () => clock, publish: () => {}, airportPoints: () => [],
    fetchImpl: (async (u: unknown) => { calls.push(String(u)); return String(u).includes("lat/40.000") ? res(null, 400) : res(fiBody(1, clock)); }) as unknown as typeof fetch,
  });
  await quiet(async () => {
    assert.equal((await only.tick(clock))?.status, 400);
    clock = T0 + 31_000;
    assert.equal((await only.tick(clock))?.ok, true);
  });
  assert.match(calls[1], /lat\/51\.000/, "a rejected disc is passed over, not asked again first");

  // a disc that merely failed (5xx, no answer) is retried, but only FAILOVER_DISC_MAX_FAILS times in a row
  const calls2: string[] = [];
  clock = T0;
  const flaky = createSweepFailover({
    env: ENV, governor: failingGovernor(), plan: PLAN, start: false, providers: [FAILOVER_PROVIDERS[0]],
    now: () => clock, publish: () => {}, airportPoints: () => [],
    fetchImpl: (async (u: unknown) => { calls2.push(String(u)); return String(u).includes("lat/40.000") ? res(null, 503) : res(fiBody(1, clock)); }) as unknown as typeof fetch,
  });
  await quiet(async () => {
    assert.equal((await flaky.tick(clock))?.status, 503);
    clock = T0 + 31_000;
    assert.equal((await flaky.tick(clock))?.status, 503, "the same disc once more");
    clock = T0 + 31_000 + 61_000;
    assert.equal((await flaky.tick(clock))?.ok, true);
  });
  assert.equal(FAILOVER_DISC_MAX_FAILS, 2);
  assert.match(calls2[1], /lat\/40\.000/);
  assert.match(calls2[2], /lat\/51\.000/, "then the queue moves on");
});

test("a request that never answers is cut at the hard deadline and counted as a failure", async () => {
  const h = harness({ timeoutMs: 40, answer: () => new Promise<Response>(() => {}) });
  const r = await quiet(() => h.f.tick(h.at(0)));
  assert.equal(r?.ok, false);
  assert.equal(r?.status, 0);
  assert.match(String(r?.error), /hard deadline/);
});

test("off switches: the env flag, a zero rate, the sweep itself off, and billing", async () => {
  assert.equal(failoverOffReason({} as NodeJS.ProcessEnv), null);
  assert.match(String(failoverOffReason({ SWEEP_FAILOVER_ENABLED: "0" } as NodeJS.ProcessEnv)), /SWEEP_FAILOVER_ENABLED=0/);
  assert.match(String(failoverOffReason({ SWEEP_FAILOVER_RPS: "0" } as NodeJS.ProcessEnv)), /SWEEP_FAILOVER_RPS=0/);
  assert.match(String(failoverOffReason({ GLOBAL_SWEEP_ENABLED: "0" } as NodeJS.ProcessEnv)), /world sweep itself is off/);
  assert.match(String(failoverOffReason({ STRIPE_SECRET_KEY: "sk_live_x" } as NodeJS.ProcessEnv)), /billing is active/);
  assert.match(String(failoverOffReason({ BILLING_ENABLED: "true" } as NodeJS.ProcessEnv)), /non-commercial/);
  assert.equal(failoverRpsFromEnv({} as NodeJS.ProcessEnv), DEFAULT_FAILOVER_RPS);
  assert.equal(failoverRpsFromEnv({ SWEEP_FAILOVER_RPS: "5" } as NodeJS.ProcessEnv), MAX_FAILOVER_RPS, "never above the providers' published 1 request per second");
  assert.equal(failoverRpsFromEnv({ SWEEP_FAILOVER_RPS: "junk" } as NodeJS.ProcessEnv), DEFAULT_FAILOVER_RPS);

  const h = harness({ env: { BILLING_ENABLED: "true" } as NodeJS.ProcessEnv });
  assert.equal(await h.f.tick(h.at(0)), null);
  assert.equal(h.calls.length, 0, "no request to a non-commercial provider once billing is on");
  const s = h.f.status(T0);
  assert.equal(s.enabled, false);
  assert.match(String(s.off_reason), /billing is active/);
});

test("licensing: both failover providers are on the monetization tripwire's list", () => {
  for (const p of FAILOVER_PROVIDERS) {
    assert.ok(NON_COMMERCIAL_AIRCRAFT_PROVIDERS.includes(p.key), `${p.key} must be counted by providerCompliance`);
  }
  assert.ok(!FAILOVER_PROVIDERS.some((p) => p.key === "adsblol"));
});

test("wiring: the worldwide service builds it, archives its fixes under the sweep's disk gate, reports it and stops it", () => {
  assert.ok(ARCHIVED_ORIGINS.has(FAILOVER_ORIGIN));
  const src = fs.readFileSync(path.join(here, "globalAircraft.ts"), "utf8");
  // CHANGED ASSERTION 2026-11-17: before the worldwide service built the failover with env, fetchImpl, now and
  // livePositions only / after it also hands it the worldwide feed's demand (the failover runs on demand: only
  // while GET /api/data/aircraft/global was requested in the last 10 min). The four earlier arguments stay pinned.
  assert.match(src, /createSweepFailover\(\{\s+env, fetchImpl: deps\.fetchImpl, now, livePositions: \(\) => snapshot\.rows\(\{\}\),\s+demand: worldFeedDemand, lastDemandAt: \(\) => lastGlobalRequestAt,\s+\}\)/);
  assert.match(src, /sweep_failover: sweepFailover\.status\(t\)/);
  assert.match(src, /sweepFailover\.stop\(\)/);
  assert.match(src, /sweepFailover\.scheduler\.noteInterest\(bbox, t\)/);
  assert.match(src, /sweepFailover\.scheduler\.creditViewportDisc\(b\.disc, now\(\)\)/);
  // the same disc URL shapes as the viewport chain in routes.ts
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("https://api.airplanes.live/v2/point/${la.toFixed(3)}/${lo.toFixed(3)}/${r}"));
  assert.ok(routes.includes("https://opendata.adsb.fi/api/v2/lat/${la.toFixed(3)}/lon/${lo.toFixed(3)}/dist/${r}"));
  assert.equal(FAILOVER_PROVIDERS[0].url(40, -74, 250), "https://opendata.adsb.fi/api/v2/lat/40.000/lon/-74.000/dist/250");
  assert.equal(FAILOVER_PROVIDERS[1].url(40, -74, 250), "https://api.airplanes.live/v2/point/40.000/-74.000/250");
});
