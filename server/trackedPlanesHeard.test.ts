// A TRACKED PLANE IS PICKED UP WHEN ITS TRANSPONDER IS FIRST HEARD
// (owner report 2026-11-12: "it seem to be not tracking or picking up on adsb
// or other background ssytems the planes right away i am looking ofr n843s").
//
// Measured live that day as N843S (hex ab8c8e) left Chicago Midway:
//   19:48:28Z  adsb.lol /v2/reg/N843S  -> 0 aircraft
//   19:49:43Z  adsb.lol /v2/hex/ab8c8e -> 1 aircraft: type "mode_s",
//              alt_baro 850, gs 144.3, seen 0.0, NO lat/lon
//   19:50:29Z  a real position (41.787, -87.800), 2,825 ft
//   19:50:52Z  /v2/reg/N843S finally returns it
// The fixtures below are those answers. Run:
//   npx tsx --test server/trackedPlanesHeard.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  TRACKED_ACTIVE_WINDOW_MS, TRACKED_CAP, TRACKED_FAST_MAX_PER_MIN, TRACKED_FAST_POLL_MS, TRACKED_MAX_REQ_PER_MIN,
  TRACKED_POLL_MS, activeHexes, batchUrl, createTrackedCycle, foldTrackedRows, heardFromRaw, hexBatchUrl,
  isTrackedActive, mergePolled, pollTrackedOnce, registryPath, saveRegistry, startTrackedPoller,
  type TrackedRegistry,
} from "./trackedPlanes";
import { UpstreamGovernor } from "./adsbGovernor";
import { subscribeFixes, type FixBatch } from "./aircraftFixBus";

const T0 = Date.UTC(2026, 10, 12, 19, 49, 43); // 19:49:43Z, the hex answer
const H3 = 3 * 3600_000;
/** where the registry last saw N843S before this flight (a real earlier fix) */
const EARLIER = { last_seen: T0 - H3, last_pos: { la: 41.9221, lo: -72.7071, al: null } };

const regEmpty = (now: number) => ({ ac: [], msg: "No error", now, total: 0 });
/** the 19:49:43Z hex answer: heard, no position (the keys are simply absent) */
const hexHeard = (now: number) => ({
  ac: [{ hex: "ab8c8e", type: "mode_s", r: "N843S", t: "FA7X", alt_baro: 850, gs: 144.3, track: 221.4, seen: 0.0, messages: 41, rssi: -24.1 }],
  msg: "No error", now, total: 1,
});
/** the 19:50:29Z answer: a real position */
const hexPositioned = (now: number) => ({
  ac: [{ hex: "ab8c8e", type: "adsb_icao", flight: "N843S   ", r: "N843S", t: "FA7X", alt_baro: 2825, gs: 201.6, track: 148.2,
         category: "A2", lat: 41.787, lon: -87.8, seen_pos: 0.3, seen: 0.1 }],
  msg: "No error", now, total: 1,
});

interface Call { t: number; url: string }
/** a provider that answers by URL kind; `clock` is the test's fake time */
function provider(clock: () => number, answer: { reg: (t: number) => unknown; hex: (t: number, hexes: string[]) => unknown; hexStatus?: () => number }) {
  const calls: Call[] = [];
  const fetchImpl = (async (url: string) => {
    const t = clock();
    calls.push({ t, url });
    const isHex = url.includes("/v2/hex/");
    const status = isHex ? (answer.hexStatus?.() ?? 200) : 200;
    const body = isHex ? answer.hex(t, url.split("/v2/hex/")[1].split(",")) : answer.reg(t);
    return { ok: status === 200, status, headers: { get: () => null }, json: async () => body };
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}
const quietGovernor = () => new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 });

test("the live case: registration batch empty + hex heard as mode_s without lat/lon -> heard recorded, NO position written", async () => {
  const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e", ...EARLIER }] };
  const p = provider(() => T0, { reg: regEmpty, hex: hexHeard });
  const out = await pollTrackedOnce(reg, p.fetchImpl, {}, T0);
  assert.deepEqual(p.calls.map((c) => c.url), [
    "https://api.adsb.lol/v2/reg/N843S",
    "https://api.adsb.lol/v2/hex/ab8c8e",
  ], "the registration batch, then ONE hex request for the plane it gave no position for");
  const tp = out.registry.planes[0];
  assert.deepEqual(tp.heard, { at: T0, alt_ft: 850, gs_kt: 144.3, source: "mode_s", has_position: false });
  assert.equal(tp.last_heard, T0);
  assert.equal(tp.last_seen, EARLIER.last_seen, "last_seen means a POSITION fix — untouched by a heard-only reading");
  assert.deepEqual(tp.last_pos, EARLIER.last_pos, "no position is written or invented from a heard-only reading");
  assert.deepEqual(out.points, [], "nothing goes to the fix bus or the archive without a position");
  assert.equal(out.heard, 1);
  assert.equal(out.requests, 2);
});

test("explicit null lat/lon (how a reader may show the same answer) is still 'no position'", async () => {
  const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e" }] };
  const p = provider(() => T0, {
    reg: regEmpty,
    hex: (now) => ({ ac: [{ hex: "ab8c8e", type: "mode_s", alt_baro: 850, gs: 144.3, seen: 0.0, lat: null, lon: null, seen_pos: null }], now }),
  });
  const out = await pollTrackedOnce(reg, p.fetchImpl, {}, T0);
  assert.equal(out.registry.planes[0].heard?.has_position, false);
  assert.equal(out.registry.planes[0].last_seen ?? null, null);
  assert.equal(out.registry.planes[0].last_pos ?? null, null);
  assert.deepEqual(out.points, []);
});

test("next cycle with a position: last_seen / last_pos update and heard is marked has_position", async () => {
  let reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e", ...EARLIER }] };
  let t = T0;
  const p = provider(() => t, { reg: regEmpty, hex: (now) => (now >= T0 + 46_000 ? hexPositioned(now) : hexHeard(now)) });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: quietGovernor() });
  const first = await cycle.run(reg, t);
  assert.equal(first.kind, "base");
  reg = mergePolled(reg, first.registry);
  assert.equal(cycle.nextDelayMs(reg, t), TRACKED_FAST_POLL_MS, "heard -> the fast cadence starts at once");
  // fast ticks until the provider decodes a position (19:50:29Z = T0 + 46 s)
  let firstFixAt: number | null = null;
  while (t < T0 + 60_000 && firstFixAt == null) {
    t += cycle.nextDelayMs(reg, t);
    const out = await cycle.run(reg, t);
    reg = mergePolled(reg, out.registry);
    if (out.points.length) {
      firstFixAt = t;
      assert.equal(out.points.length, 1);
      assert.equal(out.points[0].icao24, "ab8c8e");
      assert.equal(out.points[0].lat, 41.787);
      assert.equal(out.points[0].lon, -87.8);
      assert.equal(out.points[0].altitude_m, Math.round(2825 * 0.3048));
    }
  }
  assert.ok(firstFixAt != null, "the position was picked up");
  assert.ok(firstFixAt! - (T0 + 46_000) <= TRACKED_FAST_POLL_MS, `first position within one fast tick of the provider having it (was ${firstFixAt! - (T0 + 46_000)} ms)`);
  const tp = reg.planes[0];
  assert.equal(tp.last_seen, firstFixAt);
  assert.deepEqual(tp.last_pos, { la: 41.787, lo: -87.8, al: Math.round(2825 * 0.3048) });
  assert.equal(tp.heard?.has_position, true);
  assert.equal(tp.heard?.source, "adsb_icao");
  assert.equal(tp.heard?.alt_ft, 2825);
});

test("cadence: 30 s while silent, 5 s while active, back to 30 s after TRACKED_ACTIVE_WINDOW_MS of silence; never over the stated request cap", async () => {
  let reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e", ...EARLIER }] };
  let t = T0;
  // silent for 60 s, heard without position 60-100 s, positions 100-400 s, silent after
  const HEARD_FROM = T0 + 60_000, POS_FROM = T0 + 100_000, GONE_FROM = T0 + 400_000;
  const p = provider(() => t, {
    reg: regEmpty,
    hex: (now) => (now < HEARD_FROM || now >= GONE_FROM ? regEmpty(now) : now < POS_FROM ? hexHeard(now) : hexPositioned(now)),
  });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: quietGovernor() });
  const delays: { t: number; delay: number }[] = [];
  const END = T0 + 20 * 60_000;
  while (t < END) {
    const out = await cycle.run(reg, t);
    reg = mergePolled(reg, out.registry);
    const delay = cycle.nextDelayMs(reg, t);
    delays.push({ t, delay });
    t += delay;
  }
  const delayAt = (ms: number) => delays.filter((d) => d.t <= ms).slice(-1)[0].delay;
  assert.equal(delayAt(T0 + 30_000), TRACKED_POLL_MS, "silent: the 30 s cadence");
  assert.equal(delayAt(HEARD_FROM + 1), TRACKED_FAST_POLL_MS, "heard: fast");
  assert.equal(delayAt(POS_FROM + 60_000), TRACKED_FAST_POLL_MS, "positions flowing: still fast");
  assert.equal(delayAt(GONE_FROM + TRACKED_ACTIVE_WINDOW_MS - 10_000), TRACKED_FAST_POLL_MS, "inside the active window after the last signal: still fast");
  assert.equal(delayAt(GONE_FROM + TRACKED_ACTIVE_WINDOW_MS + 35_000), TRACKED_POLL_MS, "silent past the active window: back to 30 s");
  // request budget: every rolling minute of the whole 20 min run
  let worst = 0;
  for (const c of p.calls) {
    const n = p.calls.filter((o) => o.t > c.t - 60_000 && o.t <= c.t).length;
    worst = Math.max(worst, n);
  }
  assert.ok(worst <= TRACKED_MAX_REQ_PER_MIN, `worst rolling minute ${worst} requests > cap ${TRACKED_MAX_REQ_PER_MIN}`);
  assert.equal(TRACKED_MAX_REQ_PER_MIN, 16);
  const idle = p.calls.filter((c) => c.t >= END - 60_000).length;
  assert.ok(idle <= 4, `silent minute: ${idle} requests (registration batch + hex lookup per 30 s cycle)`);
  const stats = cycle.stats();
  assert.ok(stats.fast_cycles > 0 && stats.base_cycles >= 40);
});

test("the request cap holds even if the timer misfires every second, and for a FULL registry (one batched hex request)", async () => {
  const planes = Array.from({ length: TRACKED_CAP }, (_, i) => ({
    reg: `N${100 + i}AB`, added_at: 1, hex: (0xa00000 + i).toString(16), last_seen: T0 - 1000, last_heard: T0 - 1000,
  }));
  let reg: TrackedRegistry = { planes };
  let t = T0;
  const p = provider(() => t, {
    reg: regEmpty,
    hex: (now, hexes) => ({ now, ac: hexes.map((h) => ({ hex: h, type: "mode_s", alt_baro: 1200, gs: 150, seen: 0.2 })) }),
  });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: quietGovernor() });
  const kinds: string[] = [];
  for (let i = 0; i < 180; i++) { // 3 minutes, one tick per SECOND
    const out = await cycle.run(reg, t);
    kinds.push(out.kind);
    reg = mergePolled(reg, out.registry);
    t += 1_000;
  }
  for (const c of p.calls) {
    const n = p.calls.filter((o) => o.t > c.t - 60_000 && o.t <= c.t).length;
    assert.ok(n <= TRACKED_MAX_REQ_PER_MIN, `${n} requests in a rolling minute > ${TRACKED_MAX_REQ_PER_MIN}`);
  }
  assert.ok(kinds.includes("skipped-cap"), "fast ticks beyond TRACKED_FAST_MAX_PER_MIN are refused");
  assert.equal(TRACKED_FAST_MAX_PER_MIN, 60_000 / TRACKED_FAST_POLL_MS);
  const hexCalls = p.calls.filter((c) => c.url.includes("/v2/hex/"));
  assert.ok(hexCalls.length > 0);
  for (const c of hexCalls) {
    assert.equal(c.url.split("/v2/hex/")[1].split(",").length, TRACKED_CAP, "20 active planes = ONE request with 20 hexes, not 20 requests");
  }
});

test("hexBatchUrl: one comma-separated request (verified live 2026-11-12: /v2/hex/ab8c8e,a42dee returned both)", () => {
  assert.equal(hexBatchUrl(["AB8C8E", "a42dee"]), "https://api.adsb.lol/v2/hex/ab8c8e,a42dee");
  assert.equal(hexBatchUrl(["ab8c8e", "not-a-hex", "../x"]), "https://api.adsb.lol/v2/hex/ab8c8e", "only 6-digit hexes ever reach the URL");
  assert.equal(batchUrl(["N843S"]), "https://api.adsb.lol/v2/reg/N843S");
});

test("governor: hex requests are accounted on the shared adsb.lol governor and held back while it is backing off", async () => {
  const gov = quietGovernor();
  let reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e", last_heard: T0 - 1000 }] };
  let t = T0;
  const p = provider(() => t, { reg: regEmpty, hex: hexHeard });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: gov });
  const before = gov.windowCount(t);
  await cycle.run(reg, t); // base: reg (as before, not accounted) + hex (accounted)
  assert.equal(gov.windowCount(t) - before, 1, "exactly the NEW hex request is recorded on the governor");
  // adsb.lol rate-limits the sweep: every lane backs off
  gov.noteResult("bg", 429, t);
  assert.ok(gov.backoffMs(t) > 0);
  t += TRACKED_FAST_POLL_MS;
  const n0 = p.calls.length;
  const fast = await cycle.run(reg, t);
  assert.equal(fast.kind, "skipped-backoff");
  assert.equal(p.calls.length, n0, "no fast request while adsb.lol is struggling");
  t = T0 + TRACKED_POLL_MS; // still inside the 30 s backoff window's edge? the base cycle is due
  gov.noteResult("bg", 429, t); // keep the backoff alive for the base cycle
  const base = await cycle.run(reg, t);
  assert.equal(base.kind, "base");
  assert.equal(base.requests, 1, "during a backoff only the registration batch runs, as it always has");
  assert.deepEqual(p.calls.slice(n0).map((c) => c.url), ["https://api.adsb.lol/v2/reg/N843S"]);
  reg = mergePolled(reg, base.registry);
});

test("a hex 429 feeds the governor's backoff; the registration batch's fixes are still returned", async () => {
  const gov = quietGovernor();
  const reg: TrackedRegistry = { planes: [
    { reg: "N843S", added_at: 1, hex: "ab8c8e" },
    { reg: "N369CA", added_at: 1 },
  ] };
  const p = provider(() => T0, {
    reg: (now) => ({ now, ac: [{ hex: "a42dee", r: "N369CA", t: "FA7X", alt_baro: 38000, gs: 470, lat: 39.1, lon: -94.6, seen: 0.2, seen_pos: 0.4 }] }),
    hex: regEmpty, hexStatus: () => 429,
  });
  const cycle = createTrackedCycle({ fetchImpl: p.fetchImpl, headers: {}, governor: gov });
  const out = await cycle.run(reg, T0);
  assert.equal(out.points.length, 1, "the registration batch's fix survives a failed hex lookup");
  assert.equal(out.points[0].icao24, "a42dee");
  assert.match(String(out.hexNote), /429/);
  assert.ok(gov.backoffMs(T0) > 0, "the 429 is reported to the shared governor");
  assert.equal(out.registry.planes.find((x) => x.reg === "N369CA")?.hex, "a42dee");
});

test("identity is never inferred: a hex row naming ANOTHER registration is not attributed to the tracked tail", () => {
  const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e" }] };
  const other = foldTrackedRows(reg, [{ hex: "ab8c8e", r: "N111ZZ", type: "adsb_icao", alt_baro: 5000, gs: 200, lat: 40, lon: -80, seen: 0 }], T0, T0);
  assert.deepEqual(other.registry.planes[0], reg.planes[0], "untouched");
  assert.equal(other.positioned.length, 0);
  // a row with NO registration on the tracked plane's own hex is that plane
  const bare = foldTrackedRows(reg, [{ hex: "ab8c8e", type: "mode_s", alt_baro: 850, gs: 144.3, seen: 0 }], T0, T0);
  assert.equal(bare.registry.planes[0].heard?.source, "mode_s");
});

test("a registration-batch row WITHOUT lat/lon is a heard reading, never a position (was: last_seen + an undefined last_pos)", async () => {
  const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1 }] };
  const p = provider(() => T0, {
    reg: (now) => ({ now, ac: [{ hex: "ab8c8e", r: "N843S", type: "mode_s", alt_baro: 850, gs: 144.3, seen: 1.5 }] }),
    hex: (now) => ({ now, ac: [{ hex: "ab8c8e", r: "N843S", type: "mode_s", alt_baro: 850, gs: 144.3, seen: 1.5 }] }),
  });
  const out = await pollTrackedOnce(reg, p.fetchImpl, {}, T0);
  const tp = out.registry.planes[0];
  assert.equal(tp.hex, "ab8c8e", "the hex is still learned");
  assert.equal(tp.last_seen ?? null, null);
  assert.equal(tp.last_pos ?? null, null);
  assert.equal(tp.heard?.at, T0 - 1500, "at = the provider's clock minus the row's own `seen`");
  assert.deepEqual(out.points, []);
});

test("heardFromRaw: ground -> no altitude; missing fields stay null; an older reading never replaces a newer one", () => {
  assert.deepEqual(heardFromRaw({ hex: "ab8c8e", type: "mode_s", alt_baro: "ground", alt_geom: 620, seen: 2 }, T0),
    { at: T0 - 2000, alt_ft: null, gs_kt: null, source: "mode_s", has_position: false });
  assert.deepEqual(heardFromRaw({ hex: "ab8c8e", alt_geom: 900, gs: 130 }, T0),
    { at: T0, alt_ft: 900, gs_kt: 130, source: null, has_position: false });
  const reg: TrackedRegistry = { planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e",
    heard: { at: T0, alt_ft: 850, gs_kt: 144.3, source: "mode_s", has_position: false }, last_heard: T0 }] };
  const stale = foldTrackedRows(reg, [{ hex: "ab8c8e", type: "mode_s", alt_baro: 700, seen: 40 }], T0 + 5_000, T0 + 5_000);
  assert.equal(stale.registry.planes[0].heard?.alt_ft, 850, "a 40 s old message does not overwrite a newer reading");
  assert.equal(stale.registry.planes[0].last_heard, T0);
});

test("isTrackedActive / activeHexes: heard or positioned inside the window, hex known", () => {
  const now = T0;
  assert.equal(isTrackedActive({ reg: "N1", added_at: 1, hex: "ab8c8e", last_heard: now - 10_000 }, now), true);
  assert.equal(isTrackedActive({ reg: "N1", added_at: 1, hex: "ab8c8e", last_seen: now - TRACKED_ACTIVE_WINDOW_MS + 1 }, now), true);
  assert.equal(isTrackedActive({ reg: "N1", added_at: 1, hex: "ab8c8e", last_seen: now - TRACKED_ACTIVE_WINDOW_MS - 1 }, now), false);
  assert.equal(isTrackedActive({ reg: "N1", added_at: 1, last_heard: now }, now), false, "no hex yet -> nothing to ask by hex");
  assert.equal(isTrackedActive({ reg: "N1", added_at: 1, hex: "ab8c8e" }, now), false, "never seen");
  assert.deepEqual(activeHexes({ planes: [
    { reg: "N1", added_at: 1, hex: "AB8C8E", last_heard: now },
    { reg: "N2", added_at: 1, hex: "a42dee", last_seen: now - H3 },
  ] }, now), ["ab8c8e"]);
});

test("mergePolled: a tail added or removed while a cycle was in flight, and the backfill mark, are not clobbered", () => {
  const polled: TrackedRegistry = { planes: [
    { reg: "N843S", added_at: 1, hex: "ab8c8e", last_seen: T0, last_pos: { la: 41.787, lo: -87.8, al: 861 }, backfilled_to: 100 },
    { reg: "N999XX", added_at: 1, hex: "a00001", last_heard: T0 },
  ] };
  const current: TrackedRegistry = { planes: [
    { reg: "N843S", added_at: 1, backfilled_to: 900 },   // backfill advanced meanwhile
    { reg: "N555NEW", added_at: 2 },                     // added through the API meanwhile
    // N999XX was removed through the API meanwhile
  ] };
  const m = mergePolled(current, polled);
  assert.deepEqual(m.planes.map((p) => p.reg), ["N843S", "N555NEW"]);
  assert.equal(m.planes[0].last_seen, T0);
  assert.equal(m.planes[0].hex, "ab8c8e");
  assert.equal(m.planes[0].backfilled_to, 900, "the backfill's high-water mark is the backfill's to write");
});

test("the poller end to end: a heard-only cycle writes `heard` to the registry file and NOTHING to the fix bus or the archive", async () => {
  const base = mkdtempSync(path.join(tmpdir(), "tracked-heard-"));
  saveRegistry({ planes: [{ reg: "N843S", added_at: 1, hex: "ab8c8e", ...EARLIER }] }, base);
  const batches: FixBatch[] = [];
  const off = subscribeFixes((b) => { batches.push(b); });
  const p = provider(() => Date.now(), { reg: regEmpty, hex: hexHeard });
  const poller = startTrackedPoller({ fetchImpl: p.fetchImpl, headers: {}, sites: [], baseDir: base, governor: quietGovernor() });
  try {
    for (let i = 0; i < 100 && !poller.getRegistry().planes[0].heard; i++) await new Promise((r) => setTimeout(r, 10));
    const tp = poller.getRegistry().planes[0];
    assert.equal(tp.heard?.has_position, false);
    assert.equal(tp.heard?.alt_ft, 850);
    assert.equal(tp.last_seen, EARLIER.last_seen);
    assert.deepEqual(tp.last_pos, EARLIER.last_pos);
    assert.deepEqual(batches, [], "no fix published");
    const dir = path.join(base, "aircraft");
    assert.ok(!existsSync(dir) || readdirSync(dir).length === 0, "no archive line written");
    const onDisk = JSON.parse(readFileSync(registryPath(base), "utf-8")) as TrackedRegistry;
    assert.equal(onDisk.planes[0].heard?.source, "mode_s", "the heard reading is persisted");
    assert.equal(onDisk.planes[0].last_seen, EARLIER.last_seen);
    assert.ok(poller.cycleStats().base_cycles >= 1);
  } finally {
    poller.stop();
    off();
    rmSync(base, { recursive: true, force: true });
  }
});
