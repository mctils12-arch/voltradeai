// Selected-aircraft fast lane (human 2026-09-30: "optimize for the plane you
// have clicked on … I don't want it to lag if there is data"): fresh snapshot
// short-circuits, per-hex coalescing, 2 s cache, viewer-lane governor
// accounting (never gated by the background sweep), publish to the fix bus,
// politeness bounds, honest fix times.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  createFastLane, fastLaneUrl, FAST_CACHE_MS, FAST_FRESH_MS, FAST_MAX_HEXES_PER_MIN, FAST_MAX_RPS, FAST_BURST,
  FAST_BREAKER_FAILURES, FAST_BREAKER_PROBE_MS,
} from "./aircraftFastLane";
import { UpstreamGovernor } from "./adsbGovernor";
import { GlobalSnapshot } from "./globalSnapshot";
import type { FixBatch } from "./aircraftFixBus";

const NOW0 = 1_790_799_400_000;

function rig(opts: { status?: number; seenPos?: number; baro?: number; hold?: boolean } = {}) {
  let clock = NOW0;
  const snap = new GlobalSnapshot();
  const published: FixBatch[] = [];
  const urls: string[] = [];
  const release: (() => void)[] = [];
  const gov = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 });
  const fetchImpl = (async (url: string) => {
    urls.push(url);
    if (opts.hold) await new Promise<void>((r) => release.push(r));
    const status = opts.status ?? 200;
    const hex = url.split("/").pop() as string;
    return {
      ok: status >= 200 && status < 300, status, headers: { get: () => null },
      json: async () => ({
        now: clock, total: 1,
        ac: [{ hex, flight: "N843S   ", r: "N843S", t: "FA7X", lat: 36.2, lon: -83.95, alt_baro: 43000, gs: 466, track: 186,
          baro_rate: opts.baro ?? -64, seen_pos: opts.seenPos ?? 0.4, category: "A2" }],
      }),
    } as unknown as Response;
  }) as unknown as typeof fetch;
  const lane = createFastLane({
    snapshotGet: (h) => snap.get(h),
    publish: (b) => { published.push(b); snap.ingest(b, clock); },
    fetchImpl, governor: gov, now: () => clock,
  });
  return { lane, snap, published, urls, gov, release, tick: (ms: number) => { clock += ms; }, now: () => clock };
}

test("fresh snapshot row (< FAST_FRESH_MS) is served with NO upstream request", async () => {
  const r = rig();
  r.snap.ingest({ provider: "adsblol", origin: "sweep-disc", fetchedAt: NOW0, upstreamNowMs: NOW0,
    aircraft: [{ icao24: "ab8c8e", callsign: "N843S", lat: 36.1, lon: -83.9, altitude_m: 13106, on_ground: false, velocity_ms: 240, heading: 185, seen_pos: 1 } as any] }, NOW0);
  const res = await r.lane.lookup("AB8C8E");
  assert.equal(res.source, "snapshot");
  assert.equal(r.urls.length, 0);
  assert.ok(NOW0 - res.row!.seenAt <= FAST_FRESH_MS);
});

test("stale/absent -> ONE adsb.lol /v2/hex request, governed on the VIEWER lane, published to the fix bus, honest fix time", async () => {
  const r = rig({ seenPos: 0.4, baro: -64 });
  const res = await r.lane.lookup("ab8c8e");
  assert.deepEqual(r.urls, [fastLaneUrl("ab8c8e")]);
  assert.equal(fastLaneUrl("ab8c8e"), "https://api.adsb.lol/v2/hex/ab8c8e");
  assert.equal(res.source, "upstream");
  assert.equal(res.row?.callsign, "N843S");
  assert.equal(res.row?.seenAt, NOW0 - 400, "fix time = upstream now − seen_pos (never 'now')");
  assert.equal(res.baroRateFpm, -64, "broadcast vertical rate keyed to this fix");
  assert.equal(r.published.length, 1);
  assert.equal(r.published[0].origin, "fastlane");
  assert.equal(r.published[0].provider, "adsblol", "monetization-lawful provider");
  const st = r.gov.stats(r.now());
  assert.equal(st.window_fg, 1, "accounted as a FOREGROUND (viewer) request");
  assert.equal(st.window_bg, 0);
});

test("viewer lane is never gated by the background sweep: a saturated background still gets its answer", async () => {
  const r = rig();
  for (let i = 0; i < 80; i++) r.gov.noteRequest("bg", r.now()); // sweep filled the window
  assert.ok(r.gov.backgroundWaitMs(r.now()) > 0, "background would wait");
  const res = await r.lane.lookup("ab8c8e");
  assert.equal(res.source, "upstream");
});

test("concurrent requests for one hex COALESCE into one upstream call; the 2 s cache answers repeats", async () => {
  const r = rig({ hold: true });
  const a = r.lane.lookup("ab8c8e");
  const b = r.lane.lookup("ab8c8e");
  await new Promise((res) => setTimeout(res, 0));
  assert.equal(r.urls.length, 1, "one in flight for both viewers");
  r.release.forEach((f) => f());
  const [ra, rb] = await Promise.all([a, b]);
  assert.equal(ra.row?.hex, "ab8c8e");
  assert.equal(rb.row?.hex, "ab8c8e");
  r.tick(FAST_CACHE_MS - 500); // inside the 2 s cache window
  const c = await r.lane.lookup("ab8c8e");
  assert.ok(c.source === "snapshot" || c.source === "cache", `served locally (${c.source})`);
  assert.equal(r.urls.length, 1);
  r.tick(FAST_CACHE_MS + FAST_FRESH_MS + 10);
  const d = r.lane.lookup("ab8c8e");
  await new Promise((res) => setTimeout(res, 0));
  r.release.forEach((f) => f());
  await d;
  assert.equal(r.urls.length, 2, "after the cache window a new request goes out");
});

test("politeness: distinct-hex cap per minute and total rate bucket; over either bound the lane serves the snapshot", async () => {
  const r = rig();
  let upstream = 0;
  for (let i = 0; i < FAST_MAX_HEXES_PER_MIN + 5; i++) {
    r.tick(1_000 / FAST_MAX_RPS + 1); // keep the rate bucket full so only the hex cap binds
    const hex = (0xa00000 + i).toString(16);
    const res = await r.lane.lookup(hex);
    if (res.source === "upstream") upstream++;
    else assert.equal(res.source, "absent", "over the distinct-hex cap and not held by the snapshot -> honest absent, no request");
  }
  assert.equal(upstream, FAST_MAX_HEXES_PER_MIN);
  // total rate: a burst of distinct hexes after the window resets
  const r2 = rig();
  let up2 = 0;
  for (let i = 0; i < 6; i++) {
    const res = await r2.lane.lookup((0xb00000 + i).toString(16));
    if (res.source === "upstream") up2++;
  }
  assert.equal(up2, FAST_BURST, "burst bounded, no refill without time passing");
  assert.equal((await r2.lane.lookup("b00010")).source, "absent", "nothing held + rate-limited = honest absent");
});

test("upstream failure: counted on the governor (backoff) and the older snapshot row is returned with its REAL age", async () => {
  const r = rig({ status: 503 });
  r.snap.ingest({ provider: "adsblol", origin: "sweep-disc", fetchedAt: NOW0 - 60_000, upstreamNowMs: NOW0 - 60_000,
    aircraft: [{ icao24: "ab8c8e", callsign: "N843S", lat: 36.1, lon: -83.9, altitude_m: 13106, on_ground: false, velocity_ms: 240, heading: 185, seen_pos: 0 } as any] }, NOW0);
  const res = await r.lane.lookup("ab8c8e");
  assert.equal(res.source, "error");
  assert.equal(res.upstreamStatus, 503);
  assert.equal(res.row?.seenAt, NOW0 - 60_000, "never re-stamped as fresh");
  assert.ok(r.gov.stats(r.now()).backoff_s > 0, "the shared governor backs the sweep off");
  await assert.rejects(r.lane.lookup("zz"), /hex required/);
});

// ── 2026-11-16 [REPAIR]: the second provider answers when adsb.lol FAILS ──
// Measured live that day: 25 of 57 lookups of one airborne aircraft ended in an adsb.lol timeout (6 s each) or a
// refusal, and its fix aged from 46 s to 142 s. adsb.fi was asked only when adsb.lol answered with nothing.

/** A lane whose adsb.lol answers with `primary()` (an HTTP status, or "timeout") and whose second provider
 *  returns one row for the hex, `fallbackAgeS` seconds old, or nothing when `fallbackAgeS` is null. */
function failoverRig(primary: () => number | "timeout", fallbackAgeS: number | null = 1) {
  let clock = NOW0;
  const snap = new GlobalSnapshot();
  const published: FixBatch[] = [];
  const primaryUrls: string[] = [];
  const fallbackHexes: string[] = [];
  const gov = new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 });
  const fetchImpl = (async (url: string) => {
    primaryUrls.push(url);
    const st = primary();
    if (st === "timeout") throw Object.assign(new Error("The operation was aborted due to timeout"), { name: "TimeoutError" });
    const hex = url.split("/").pop() as string;
    return {
      ok: st >= 200 && st < 300, status: st, headers: { get: () => null },
      json: async () => ({ now: clock, ac: [{ hex, flight: "N843S   ", r: "N843S", t: "FA7X", lat: 36.2, lon: -83.95, alt_baro: 43000, gs: 466, track: 186, seen_pos: 0.4 }] }),
    } as unknown as Response;
  }) as unknown as typeof fetch;
  const lane = createFastLane({
    snapshotGet: (h) => snap.get(h),
    publish: (b) => { published.push(b); snap.ingest(b, clock); },
    fetchImpl, governor: gov, now: () => clock,
    fallbackHex: async (hex) => {
      fallbackHexes.push(hex);
      if (fallbackAgeS == null) return [];
      return [{ icao24: hex, callsign: "N843S", lat: 36.4, lon: -83.9, altitude_m: 13100, on_ground: false, velocity_ms: 240, heading: 186, seen_pos: fallbackAgeS, provider: "adsbfi" }];
    },
  });
  const heldRow = (ageMs: number) => snap.ingest({ provider: "adsblol", origin: "sweep-disc", fetchedAt: clock - ageMs, upstreamNowMs: clock - ageMs,
    aircraft: [{ icao24: "ab8c8e", callsign: "N843S", lat: 36.1, lon: -83.9, altitude_m: 13106, on_ground: false, velocity_ms: 240, heading: 185, seen_pos: 0 } as any] }, clock);
  return { lane, snap, published, primaryUrls, fallbackHexes, gov, heldRow, tick: (ms: number) => { clock += ms; }, now: () => clock };
}

for (const failure of [429, 503, "timeout"] as const) {
  test(`adsb.lol ${failure}: the second provider's fix is served, with its own provenance and its own fix time`, async () => {
    const r = failoverRig(() => failure, 1);
    r.heldRow(46_000); // the 11-16 case: the held fix is 46 s old
    const res = await r.lane.lookup("ab8c8e");
    assert.equal(res.source, "fallback");
    assert.equal(res.upstreamStatus, failure === "timeout" ? null : failure);
    assert.equal(res.row?.seenAt, NOW0 - 1_000, "the second provider's own fix time, 1 s old");
    assert.equal(res.row?.src, "adsbfi", "the row says which provider it came from");
    assert.deepEqual(r.fallbackHexes, ["ab8c8e"]);
    assert.equal(r.published.length, 1);
    assert.equal(r.published[0].provider, "adsbfi");
    assert.equal(r.published[0].origin, "fastlane");
    const st = r.gov.stats(r.now());
    assert.equal(st.totals.fg, 1);
    assert.equal(st.totals.fgFail, 1, "the adsb.lol failure is still counted on the governor");
    assert.equal(r.lane.stats().fallback_answers, 1);
  });
}

test("both providers have nothing newer: the held row comes back as 'error' with its REAL age, never re-stamped", async () => {
  const r = failoverRig(() => 503, null);
  r.heldRow(46_000);
  const res = await r.lane.lookup("ab8c8e");
  assert.equal(res.source, "error");
  assert.equal(res.row?.seenAt, NOW0 - 46_000);
  assert.equal(r.published.length, 0);
  assert.equal(r.lane.stats().errors, 1);
  // a second-provider fix OLDER than the held one never replaces it
  const o = failoverRig(() => 503, 120);
  o.heldRow(46_000);
  const older = await o.lane.lookup("ab8c8e");
  assert.equal(older.row?.seenAt, NOW0 - 46_000, "freshest fix wins");
  assert.equal(older.row?.src, "adsblol");
});

test("after two failures adsb.lol is only probed: lookups skip its 6 s timeout, one probe per 15 s, and a probe that answers ends it", async () => {
  let primaryUp = false;
  const r = failoverRig(() => (primaryUp ? 200 : "timeout"), 1);
  assert.equal(FAST_BREAKER_FAILURES, 2);
  assert.equal(FAST_BREAKER_PROBE_MS, 15_000);
  // a viewer polling every 2.5 s for 60 s while adsb.lol times out
  const sources: string[] = [];
  for (let i = 0; i < 24; i++) {
    sources.push((await r.lane.lookup("ab8c8e")).source);
    r.tick(2_500);
  }
  assert.ok(sources.every((s) => s === "fallback"), `every poll was answered by the second provider (${[...new Set(sources)].join(",")})`);
  // 2 failures open it, then one probe per 15 s over the remaining ~55 s: 5 or 6 requests, not 24
  assert.ok(r.primaryUrls.length >= 5 && r.primaryUrls.length <= 6, `adsb.lol was asked ${r.primaryUrls.length} times in 60 s (24 polls)`);
  assert.equal(r.fallbackHexes.length, 24);
  assert.equal(r.gov.stats(r.now()).totals.fgFail, r.primaryUrls.length, "the governor hears only the requests actually sent");
  const st = r.lane.stats();
  assert.equal(st.breaker_skips, 24 - r.primaryUrls.length);
  assert.ok(st.primary_consecutive_failures >= 2);
  // adsb.lol recovers: the next probe answers and it is asked on every lookup again
  primaryUp = true;
  r.tick(FAST_BREAKER_PROBE_MS);
  const back = await r.lane.lookup("ab8c8e");
  assert.equal(back.source, "upstream");
  assert.equal(back.row?.src, "adsblol");
  assert.equal(r.lane.stats().primary_consecutive_failures, 0);
  assert.equal(r.lane.stats().primary_probe_only, false);
  const asked = r.primaryUrls.length;
  r.tick(5_000); // past the 3 s in which the snapshot row itself is fresh enough
  assert.equal((await r.lane.lookup("ab8c8e")).source, "upstream");
  assert.equal(r.primaryUrls.length, asked + 1);
});

test("without a second provider nothing changes: adsb.lol is asked on every lookup and a failure is 'error'", async () => {
  const r = rig({ status: 503 });
  for (let i = 0; i < 5; i++) {
    assert.equal((await r.lane.lookup("ab8c8e")).source, "error");
    r.tick(FAST_CACHE_MS + 1_000);
  }
  assert.equal(r.urls.length, 5, "no probe-only mode when there is nothing to answer in its place");
  assert.equal(r.lane.stats().breaker_skips, 0);
});
