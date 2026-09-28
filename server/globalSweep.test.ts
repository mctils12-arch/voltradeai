// FLIGHT PROGRAM B1 — the global sweep: type lane first, value-weighted
// disc scheduling, viewer interest, viewport credit, type promotion, one
// job end-to-end with an injected fetch (no network), and the kill switch.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SweepScheduler, runSweepJob, promoteTypes, chunk, typeUrl, discUrl, discSamples,
  sweepEnabled, typeRefreshMsFromEnv, discMinGapMsFromEnv, startGlobalSweep, DEFAULT_DISC_RPS,
  SEED_TYPES, TYPES_PER_BATCH, MAX_TYPES, MAX_DISC_AGE_MS, INTEREST_BOOST,
} from "./globalSweep";
import { UpstreamGovernor } from "./adsbGovernor";
import type { PlanDisc } from "./globalDiscPlan";
import type { FixBatch } from "./aircraftFixBus";

const T = 1_800_000_000_000;
const D = (id: number, lat: number, lon: number, prior = 10): PlanDisc => ({ id, lat, lon, radiusNm: 250, prior });

test("seed types: unique, valid ICAO designators, batched TYPES_PER_BATCH per request", () => {
  assert.equal(new Set(SEED_TYPES).size, SEED_TYPES.length, "no duplicate designators");
  for (const t of SEED_TYPES) assert.match(t, /^[A-Z0-9]{2,4}$/, t);
  assert.ok(SEED_TYPES.length >= 200 && SEED_TYPES.length <= MAX_TYPES);
  assert.equal(TYPES_PER_BATCH, 50, "50 per request — the size probed live");
  assert.deepEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.equal(typeUrl(["B738", "A320"]), "https://api.adsb.lol/v2/type/B738,A320");
  assert.equal(discUrl({ lat: 40, lon: -95.5, radiusNm: 250 }), "https://api.adsb.lol/v2/point/40.000/-95.500/250");
});

test("scheduler: an overdue type batch always goes before any disc", () => {
  const s = new SweepScheduler([D(0, 40, -95, 300)], { now: T, types: ["B738", "A320"], typeRefreshMs: 60_000 });
  const j1 = s.next(T);
  assert.equal(j1!.kind, "type", "never-fetched batch first");
  if (j1!.kind === "type") s.markType(j1!.batch, T, 1000, true);
  assert.equal(s.next(T + 1_000)!.kind, "disc", "batch fresh -> disc lane");
  assert.equal(s.next(T + 60_000)!.kind, "type", "batch due again after the refresh interval");
});

test("scheduler: value-weighted staleness — busy (high-residual) discs revisit more often", () => {
  const s = new SweepScheduler([D(0, 40, -95), D(1, -20, 20)], { now: T, types: [] });
  const [busy, empty] = s.discs;
  s.markDisc(busy, T, 400, true);
  s.markDisc(empty, T, 0, true);
  // equal age -> busy wins
  const j = s.next(T + 60_000)!;
  assert.equal(j.kind === "disc" && j.state.disc.id, 0);
  // empty disc must wait ~sqrt(404/4) ≈ 10x longer before it outranks the busy one
  s.markDisc(busy, T + 600_000, 400, true);
  const later = s.next(T + 600_000 + 50_000)!; // busy age 50s vs empty age 650s: 50*20.1 < 650*2
  assert.equal(later.kind === "disc" && later.state.disc.id, 1, "an empty disc still gets its turn");
});

test("scheduler: viewer interest boosts discs in view; MAX_DISC_AGE guarantees a visit", () => {
  const s = new SweepScheduler([D(0, 40, -95), D(1, 50, 10)], { now: T, types: [] });
  for (const d of s.discs) s.markDisc(d, T, 10, true);
  s.noteInterest({ lamin: 45, lamax: 55, lomin: 0, lomax: 20 }, T);
  const sEu = s.discScore(s.discs[1], T + 10_000);
  const sUs = s.discScore(s.discs[0], T + 10_000);
  assert.ok(Math.abs(sEu / sUs - INTEREST_BOOST) < 1e-9, "Europe (in view) boosted x4");
  s.markDisc(s.discs[1], T + 10_000, 5000, true); // hugely busy + interesting, just visited
  const starved = s.discScore(s.discs[0], T + MAX_DISC_AGE_MS + 1);
  const hot = s.discScore(s.discs[1], T + MAX_DISC_AGE_MS + 1);
  assert.ok(starved > hot, "a disc past MAX_DISC_AGE jumps the queue");
});

test("scheduler: viewport discs CREDIT the plan discs they fully cover (reuse, never refetch)", () => {
  const s = new SweepScheduler([D(0, 40, -95), D(1, 10, 10)], { now: T, types: [] });
  const before = s.discs[0].lastAt;
  // one viewport disc can't cover a 250nm plan disc unless it contains it
  assert.equal(s.creditViewportDisc({ lat: 40, lon: -95, radiusNm: 100 }, T), 0);
  assert.equal(s.discs[0].lastAt, before);
  assert.equal(s.creditViewportDisc({ lat: 40, lon: -95, radiusNm: 260 }, T), 1);
  assert.equal(s.discs[0].lastAt, T, "credited as freshly fetched");
  assert.equal(s.discs[1].lastAt, before, "far disc untouched");
  assert.equal(discSamples(0, 0, 100).length, 21);
});

test("promoteTypes: observed long-tail types join the list, deterministic, capped", () => {
  const counts = new Map([["P28A", 29], ["XYZ9", 5], ["C172", 99], ["ONE1", 1], ["bad!", 50]]);
  const out = promoteTypes(["C172"], counts, ["C172"], 2, 10);
  assert.deepEqual(out, ["C172", "P28A", "XYZ9"], "count>=2, not already listed, valid designator, most common first");
  assert.equal(promoteTypes(["AA"], new Map([["BB", 9], ["CC", 9]]), ["AA"], 2, 2).length, 2, "cap");
});

const govFree = () => new UpstreamGovernor({ bgRps: 10, ceilingRps: 100 });

test("runSweepJob (type lane): one worldwide request, mapped + published as adsb.lol, batch marked", async () => {
  const s = new SweepScheduler([D(0, 40, -95)], { now: T, types: ["B738", "A320"] });
  const urls: string[] = [];
  const got: FixBatch[] = [];
  const job = s.next(T)!;
  const r = await runSweepJob(job, s, {
    governor: govFree(), now: () => T + 5,
    publish: (b) => got.push(b),
    fetchImpl: (async (u: unknown) => {
      urls.push(String(u));
      return { ok: true, status: 200, json: async () => ({ now: T, ac: [
        { hex: "a1", t: "B738", lat: 51, lon: 0, alt_baro: 30000, gs: 450, track: 270, seen_pos: 1 },
        { hex: "a2", t: "A320", lat: -33, lon: 151, alt_baro: "ground", gs: 5 },
        { hex: "nopos", t: "A320" },
      ] }) };
    }) as any,
  });
  assert.deepEqual(urls, ["https://api.adsb.lol/v2/type/B738,A320"]);
  assert.equal(r.ok, true);
  assert.equal(r.count, 2, "rows without a position are dropped by the shared normalizer");
  assert.equal(got[0].origin, "sweep-type");
  assert.equal(got[0].provider, "adsblol");
  assert.equal(got[0].upstreamNowMs, T);
  assert.equal(got[0].aircraft[0].provider, "adsblol", "row-level provenance");
  assert.equal(s.batches[0].lastCount, 2);
});

test("runSweepJob (disc lane): residual = aircraft the type lane can't see", async () => {
  const s = new SweepScheduler([D(0, 40, -95)], { now: T, types: ["B738"] });
  s.markType(s.batches[0], T, 0, true);
  const job = s.next(T + 1)!;
  assert.equal(job.kind, "disc");
  const got: FixBatch[] = [];
  const r = await runSweepJob(job, s, {
    governor: govFree(), now: () => T + 10, publish: (b) => got.push(b),
    fetchImpl: (async () => ({ ok: true, status: 200, json: async () => ({ ac: [
      { hex: "b1", t: "B738", lat: 40, lon: -95 }, { hex: "b2", t: "P28A", lat: 40, lon: -95 }, { hex: "b3", lat: 40, lon: -95 },
    ] }) })) as any,
  });
  assert.equal(r.residual, 2, "P28A (unlisted) + untyped");
  assert.equal(s.discs[0].residual, 2);
  assert.equal(got[0].origin, "sweep-disc");
  assert.deepEqual(got[0].disc, { lat: 40, lon: -95, radiusNm: 250 });
});

test("runSweepJob: a 429 with Retry-After backs the governor off; the job is marked failed, not retried hot", async () => {
  const g = govFree();
  const s = new SweepScheduler([D(0, 40, -95)], { now: T, types: ["B738"] });
  const job = s.next(T)!;
  const r = await runSweepJob(job, s, {
    governor: g, now: () => T, publish: () => { throw new Error("must not publish"); },
    fetchImpl: (async () => ({ ok: false, status: 429, headers: { get: (k: string) => (k === "retry-after" ? "300" : null) } })) as any,
  });
  assert.equal(r.ok, false);
  assert.equal(r.status, 429);
  assert.ok(g.backgroundWaitMs(T) >= 300_000, "Retry-After floors the sweep pause");
  assert.equal(s.batches[0].fails, 1);
  assert.equal(s.batches[0].lastAt, T, "not immediately due again");
});

test("kill switch + env parsing: GLOBAL_SWEEP_ENABLED=0 makes ZERO upstream calls", async () => {
  assert.equal(sweepEnabled({} as any), true, "default ON (conservative rate)");
  for (const v of ["0", "false", "off", "NO"]) assert.equal(sweepEnabled({ GLOBAL_SWEEP_ENABLED: v } as any), false, v);
  assert.equal(typeRefreshMsFromEnv({} as any), 60_000);
  assert.equal(typeRefreshMsFromEnv({ GLOBAL_SWEEP_TYPE_REFRESH_S: "5" } as any), 20_000, "floored at 20s");
  let calls = 0;
  const h = startGlobalSweep({
    env: { GLOBAL_SWEEP_ENABLED: "0" } as any, governor: govFree(), startDelayMs: 0,
    fetchImpl: (async () => { calls++; return { ok: true, json: async () => ({ ac: [] }) }; }) as any,
    plan: [D(0, 40, -95)],
  });
  await new Promise((r) => setTimeout(r, 30));
  h.stop();
  assert.equal(calls, 0);
  assert.equal((h.status() as any).enabled, false);
});

test("startGlobalSweep: sequential, governor-paced loop (injected fetch, fast governor)", async () => {
  const urls: string[] = [];
  const g = new UpstreamGovernor({ bgRps: 50, ceilingRps: 1000 });
  const h = startGlobalSweep({
    env: {} as NodeJS.ProcessEnv, governor: g, startDelayMs: 0,
    // 120 types = 3 type batches, then the disc lane
    plan: [D(0, 40, -95), D(1, 50, 10)], types: Array.from({ length: 120 }, (_, i) => `T${i}`),
    fetchImpl: (async (u: unknown) => { urls.push(String(u)); return { ok: true, status: 200, json: async () => ({ ac: [] }) }; }) as any,
  });
  await new Promise((r) => setTimeout(r, 400));
  h.stop();
  assert.ok(urls.length >= 4, `loop ran (${urls.length} requests)`);
  assert.deepEqual(urls.slice(0, 3).map((u) => u.includes("/v2/type/")), [true, true, true], "all 3 type batches first");
  assert.ok(urls[0].includes("/v2/type/"), "type lane first");
  assert.ok(urls.some((u) => u.includes("/v2/point/")), "then the disc lane");
  assert.ok(urls.length <= 21, "paced by the governor (≤ 50 rps × 0.4s + 1)");
  assert.ok(urls.every((u) => u.startsWith("https://api.adsb.lol/")), "sweep uses ONLY the lawful provider");
});

test("disc lane keeps its own slower pace (default 0.15 req/s) — type batches never wait on it", async () => {
  assert.equal(DEFAULT_DISC_RPS, 0.15);
  assert.equal(discMinGapMsFromEnv({} as NodeJS.ProcessEnv), 6667);
  assert.equal(discMinGapMsFromEnv({ GLOBAL_SWEEP_DISC_RPS: "0" } as NodeJS.ProcessEnv), Number.POSITIVE_INFINITY, "0 = disc lane off");
  const urls: string[] = [];
  const g = new UpstreamGovernor({ bgRps: 50, ceilingRps: 1000 });
  const h = startGlobalSweep({
    env: {} as NodeJS.ProcessEnv, governor: g, startDelayMs: 0,
    plan: [D(0, 40, -95), D(1, 50, 10), D(2, -30, 20)], types: ["B738", "A320"],
    fetchImpl: (async (u: unknown) => { urls.push(String(u)); return { ok: true, status: 200, json: async () => ({ ac: [] }) }; }) as unknown as typeof fetch,
  });
  await new Promise((r) => setTimeout(r, 200));
  h.stop();
  assert.equal(urls.filter((u) => u.includes("/v2/type/")).length, 1, "one type batch (fresh for 60s)");
  assert.equal(urls.filter((u) => u.includes("/v2/point/")).length, 1, "ONE disc in 200ms despite a 50 rps governor");
  assert.equal(h.status().disc_rps, 0.15);
});
