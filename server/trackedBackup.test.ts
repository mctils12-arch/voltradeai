// THE TRACKED PLANES' SECOND SOURCE (2026-11-19, server/trackedBackup.ts).
//
// adsb.lol gives 500 requests a day to an address without a key, which paces
// the tracked poller to one check of each watched plane about every 14
// minutes. The same planes are also asked of adsb.fi by hex, every 10 s while
// one is active and every 30 s otherwise. What is pinned here:
//   - ONE request for every tracked plane with a valid hex; none = no request;
//   - the kill switch and the billing rule (adsb.fi is non-commercial);
//   - a row with a position is a fix tagged "adsbfi"; a row without one is
//     `heard` only; a row naming another registration, or a hex that was not
//     asked for, is ignored;
//   - adsb.fi's `now` is read in seconds or milliseconds, whichever it sends;
//   - 429 / 403 / a network failure back off, and nothing is sent meanwhile;
//   - the two loops' readings merge newest-wins, under both interleavings;
//   - the adsb.lol API key is never sent to adsb.fi;
//   - provenance reaches the fix bus, the snapshot row, the archive line and
//     the tracked planes' permanent log.
// The answer shape is the one read live 2026-11-19 from
// /api/v2/hex/<two airliners>: keys ac, ctime, msg, now, ptime, total.
// Run:  npx tsx --test server/trackedBackup.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  TRACKED_BACKUP_IDLE_POLL_MS, TRACKED_BACKUP_POLL_MS, TRACKED_BACKUP_PROVIDER, TRACKED_BACKUP_TIMEOUT_MS, TRACKED_PROVIDER_NOTE,
  createTrackedBackup, trackedBackupHexes, trackedBackupOffReason, trackedBackupPollMs, trackedBackupUrl, trackedSourceText,
} from "./trackedBackup";
import {
  TRACKED_ACTIVE_WINDOW_MS, TRACKED_PRIMARY_PROVIDER, mergePolled, pollHexesOnce, registryPath, saveRegistry, startTrackedPoller,
  type TrackedPlane, type TrackedRegistry,
} from "./trackedPlanes";
import { FAILOVER_BACKOFF_BASE_MS, FAILOVER_REFUSED_BACKOFF_MS } from "./sweepFailover";
import { UpstreamGovernor } from "./adsbGovernor";
import { subscribeFixes, type FixBatch } from "./aircraftFixBus";
import { GlobalSnapshot, ROW_FIELDS, encodeRow } from "./globalSnapshot";
import { TrackedPlaneLog } from "./trackedPlaneLog";

const here = path.dirname(fileURLToPath(import.meta.url));
const T0 = Date.UTC(2026, 10, 19, 19, 17, 16);
const H3 = 3 * 3600_000;
const A = "ab8c8e"; // N843S, the registry's seed
const B = "a42dee";
/** where the registry last saw the plane before this flight */
const EARLIER = { last_seen: T0 - H3, last_pos: { la: 41.9221, lo: -72.7071, al: null }, last_pos_provider: TRACKED_PRIMARY_PROVIDER };

type Raw = Record<string, unknown>;
/** adsb.fi's answer envelope as read live (`now` in ms on the hex endpoint) */
const answer = (now: number, ...ac: Raw[]): Raw => ({ ac, msg: "No error", now, total: ac.length, ctime: now, ptime: 0 });
const positioned = (hex: string, o: Raw = {}): Raw => ({
  hex, type: "adsb_icao", flight: "N843S   ", r: "N843S", t: "FA7X", alt_baro: 2825, gs: 201.6, track: 148.2,
  category: "A2", lat: 41.787, lon: -87.8, seen_pos: 0.4, seen: 0.2, ...o,
});
const heardOnly = (hex: string, o: Raw = {}): Raw => ({ hex, type: "mode_s", r: "N843S", t: "FA7X", alt_baro: 850, gs: 144.3, seen: 0.5, ...o });

interface Call { url: string; headers: Record<string, string> }
interface Scripted { status?: number; body?: unknown; retryAfter?: string; throws?: string }
/** a fetch whose answer comes from `next(url)`; every call's URL and headers are kept */
function mockFetch(next: (url: string) => Scripted) {
  const calls: Call[] = [];
  const fetchImpl = (async (url: string, init?: { headers?: Record<string, string> }) => {
    calls.push({ url, headers: { ...(init?.headers ?? {}) } });
    const s = next(url);
    if (s.throws) throw new Error(s.throws);
    const status = s.status ?? 200;
    return {
      ok: status === 200, status,
      headers: { get: (k: string) => (k.toLowerCase() === "retry-after" ? s.retryAfter ?? null : null) },
      json: async () => s.body,
    };
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}
const plane = (o: Partial<TrackedPlane> = {}): TrackedPlane => ({ reg: "N843S", added_at: 1, hex: A, ...o });
const reg1 = (o: Partial<TrackedPlane> = {}): TrackedRegistry => ({ planes: [plane(o)] });
const quietGovernor = () => new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 });

test("ONE adsb.fi request for every tracked plane with a valid hex; invalid hexes are dropped; none = no request", async () => {
  assert.equal(trackedBackupUrl([A, B]), `https://opendata.adsb.fi/api/v2/hex/${A},${B}`);
  assert.equal(trackedBackupUrl(["AB8C8E", " a42dee ", A]), `https://opendata.adsb.fi/api/v2/hex/${A},${B}`, "lower case, trimmed, each once");
  assert.equal(trackedBackupUrl(["xyz123", "ab8c8", "ab8c8e0", "", "ab8c8e/../x", A]), `https://opendata.adsb.fi/api/v2/hex/${A}`, "only 6 hex digits pass");
  assert.equal(trackedBackupUrl(["nothex", ""]), null);
  assert.equal(trackedBackupUrl([]), null);

  const registry: TrackedRegistry = { planes: [
    plane(), plane({ reg: "N786QS", hex: "A42DEE" }), plane({ reg: "N1", hex: null }), plane({ reg: "N2", hex: "junk" }), plane({ reg: "N3", hex: undefined }),
  ] };
  assert.deepEqual(trackedBackupHexes(registry), [A, B]);
  const m = mockFetch(() => ({ body: answer(T0) }));
  const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env: {}, now: () => T0 });
  const out = await backup.poll(registry, T0);
  assert.deepEqual(m.calls.map((c) => c.url), [`https://opendata.adsb.fi/api/v2/hex/${A},${B}`], "one request, both planes");
  assert.equal(out.requests, 1);
  assert.equal(out.ok, true);
  const st = backup.status(registry, T0);
  assert.equal(st.planes_asked, 2);
  assert.equal(st.planes_without_hex, 3, "the tails it cannot ask for yet are counted, not hidden");

  // no tracked plane has a hex yet: nothing is sent
  const none = mockFetch(() => ({ body: answer(T0) }));
  const b2 = createTrackedBackup({ fetchImpl: none.fetchImpl, env: {}, now: () => T0 });
  const skipped = await b2.poll({ planes: [plane({ hex: null }), plane({ reg: "N2", hex: "junk" })] }, T0);
  assert.equal(skipped.skipped, "no-hex");
  assert.equal(skipped.requests, 0);
  assert.equal(none.calls.length, 0);
  assert.equal((await b2.poll({ planes: [] }, T0)).skipped, "no-hex");
  assert.equal(none.calls.length, 0);
  assert.equal(b2.status({ planes: [] }, T0).requests_total, 0);
});

test("the kill switch and the billing rule: off = no request, and the status says why", async () => {
  assert.equal(trackedBackupOffReason({}), null, "on by default");
  assert.equal(trackedBackupOffReason({ TRACKED_BACKUP_ENABLED: "1" }), null);
  for (const v of ["0", "false", "off", "no", " OFF "]) assert.equal(trackedBackupOffReason({ TRACKED_BACKUP_ENABLED: v }), "TRACKED_BACKUP_ENABLED=0", v);
  // adsb.fi is non-commercial: the same billing signals as the monetization tripwire
  assert.equal(trackedBackupOffReason({ STRIPE_SECRET_KEY: "sk_test_x" }), "billing is active: adsb.fi is non-commercial");
  assert.equal(trackedBackupOffReason({ BILLING_ENABLED: "true" }), "billing is active: adsb.fi is non-commercial");
  assert.equal(trackedBackupOffReason({ BILLING_ENABLED: "true", TRACKED_BACKUP_ENABLED: "0" }), "TRACKED_BACKUP_ENABLED=0");

  for (const env of [{ TRACKED_BACKUP_ENABLED: "0" }, { STRIPE_SECRET_KEY: "sk_test_x" }, { BILLING_ENABLED: "true" }]) {
    const m = mockFetch(() => ({ body: answer(T0, positioned(A)) }));
    const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env, now: () => T0 });
    const out = await backup.poll(reg1(), T0);
    assert.equal(out.skipped, "off");
    assert.equal(out.requests, 0);
    assert.deepEqual(out.points, []);
    assert.equal(m.calls.length, 0, "nothing leaves while off");
    const st = backup.status(reg1(), T0);
    assert.equal(st.enabled, false);
    assert.equal(st.off_reason, trackedBackupOffReason(env));
    assert.equal(st.poll_s, null);
    assert.equal(st.requests_total, 0);
    assert.match(trackedSourceText(st), /^one source\. adsb\.lol \(ODbL\).*The adsb\.fi backup is off: /);
    assert.ok(trackedSourceText(st).endsWith(String(st.off_reason)));
  }
  const on = createTrackedBackup({ env: {}, now: () => T0 }).status(reg1(), T0);
  assert.equal(on.enabled, true);
  assert.equal(on.off_reason, null);
  assert.match(trackedSourceText(on), /^two sources\. \(1\) adsb\.lol \(ODbL\).*\(2\) adsb\.fi \(non-commercial\).*every 10 s while one is being heard and every 30 s otherwise/);
  assert.match(TRACKED_PROVIDER_NOTE, /adsblol always, adsbfi while backup\.enabled/);
  assert.match(TRACKED_PROVIDER_NOTE, /last_pos_provider \/ heard_provider name the provider that served each reading/);
  assert.match(TRACKED_PROVIDER_NOTE, /never replaced by an older one/);
});

test("a row with a position is a fix tagged adsbfi; the registry row says which provider served it", async () => {
  const m = mockFetch(() => ({ body: answer(T0, positioned(A)) }));
  const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env: {}, now: () => T0 });
  const out = await backup.poll(reg1(EARLIER), T0);
  assert.equal(out.ok, true);
  assert.equal(out.points.length, 1);
  const fix = out.points[0];
  assert.equal(fix.provider, "adsbfi", "the fix carries its real source");
  assert.equal(TRACKED_BACKUP_PROVIDER, "adsbfi");
  assert.equal(fix.icao24, A);
  assert.equal(fix.lat, 41.787);
  assert.equal(fix.lon, -87.8);
  assert.equal(fix.altitude_m, Math.round(2825 * 0.3048));
  assert.equal(fix.seen_pos, 0.4);
  const tp = out.registry.planes[0];
  assert.equal(tp.last_seen, T0);
  assert.deepEqual(tp.last_pos, { la: 41.787, lo: -87.8, al: Math.round(2825 * 0.3048) });
  assert.equal(tp.last_pos_provider, "adsbfi");
  assert.equal(tp.heard?.has_position, true);
  assert.equal(tp.heard_provider, "adsbfi");
  assert.equal(out.heard, 0);
  assert.equal(out.upstreamNowMs, T0);
  const st = backup.status(out.registry, T0);
  assert.deepEqual(
    { requests_total: st.requests_total, ok_total: st.ok_total, failed_total: st.failed_total, last_at: st.last_at, last_status: st.last_status, rows_last: st.rows_last, fixes_total: st.fixes_total, heard_total: st.heard_total, backoff_until: st.backoff_until, provider: st.provider },
    { requests_total: 1, ok_total: 1, failed_total: 0, last_at: T0, last_status: 200, rows_last: 1, fixes_total: 1, heard_total: 0, backoff_until: null, provider: "adsbfi" });
});

test("a row without a position is `heard` only: no fix, last_seen / last_pos untouched", async () => {
  // both spellings of "no position": the keys absent, and explicit nulls
  for (const row of [heardOnly(A), heardOnly(A, { lat: null, lon: null })]) {
    const m = mockFetch(() => ({ body: answer(T0, row) }));
    const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env: {}, now: () => T0 });
    const out = await backup.poll(reg1(EARLIER), T0);
    assert.equal(out.ok, true);
    assert.deepEqual(out.points, [], "nothing for the fix bus or the archive");
    assert.equal(out.heard, 1);
    const tp = out.registry.planes[0];
    assert.deepEqual(tp.heard, { at: T0 - 500, alt_ft: 850, gs_kt: 144.3, source: "mode_s", has_position: false });
    assert.equal(tp.heard_provider, "adsbfi");
    assert.equal(tp.last_heard, T0 - 500);
    assert.equal(tp.last_seen, EARLIER.last_seen, "a heard reading is never a position");
    assert.deepEqual(tp.last_pos, EARLIER.last_pos);
    assert.equal(tp.last_pos_provider, TRACKED_PRIMARY_PROVIDER, "the position still names the provider that served IT");
    assert.equal(backup.status(out.registry, T0).heard_total, 1);
    assert.equal(backup.status(out.registry, T0).fixes_total, 0);
  }
});

test("identity is never inferred: a row naming ANOTHER registration, or a hex that was not asked for, is ignored", async () => {
  const start = reg1(EARLIER);
  // adsb.fi's database names a different tail on the tracked plane's hex
  const other = mockFetch(() => ({ body: answer(T0, positioned(A, { r: "N111ZZ", flight: "N111ZZ  " })) }));
  const b1 = createTrackedBackup({ fetchImpl: other.fetchImpl, env: {}, now: () => T0 });
  const o1 = await b1.poll(start, T0);
  assert.equal(o1.ok, true);
  assert.deepEqual(o1.points, [], "not this plane: no fix");
  assert.equal(o1.heard, 0);
  assert.deepEqual(o1.registry.planes[0], start.planes[0], "the registry row is untouched");
  assert.equal(b1.status(o1.registry, T0).rows_last, 1, "the row was received and counted");
  assert.equal(b1.status(o1.registry, T0).fixes_total, 0);

  // a row for a hex nobody asked about, carrying the tracked tail's registration
  const stray = mockFetch(() => ({ body: answer(T0, positioned("c0ffee")) }));
  const b2 = createTrackedBackup({ fetchImpl: stray.fetchImpl, env: {}, now: () => T0 });
  const o2 = await b2.poll(start, T0);
  assert.deepEqual(o2.points, []);
  assert.equal(o2.registry.planes[0].hex, A, "the hex is not relearned from an unasked row");
  assert.equal(o2.registry.planes[0].last_seen, EARLIER.last_seen);
  assert.equal(b2.status(o2.registry, T0).rows_last, 0);

  // a row with NO registration on the tracked plane's own hex is that plane
  const bare = mockFetch(() => ({ body: answer(T0, positioned(A, { r: undefined, flight: undefined })) }));
  const o3 = await createTrackedBackup({ fetchImpl: bare.fetchImpl, env: {}, now: () => T0 }).poll(start, T0);
  assert.equal(o3.points.length, 1);
  assert.equal(o3.points[0].provider, "adsbfi");
});

test("adsb.fi's clock: `now` in SECONDS and in milliseconds both date the reading correctly, never in 1970 and never in the future", async () => {
  const ourNow = T0 + 400;
  const cases: Array<{ name: string; now: number; base: number }> = [
    { name: "seconds with a fraction (the disc endpoint's unit)", now: (T0 - 2_000) / 1000 + 0.001, base: T0 - 1_999 },
    { name: "milliseconds (the hex endpoint, read live 2026-11-19)", now: T0 - 2_000, base: T0 - 2_000 },
    { name: "a provider clock ahead of ours", now: (T0 + 60_000) / 1000, base: ourNow },
  ];
  for (const c of cases) {
    const m = mockFetch(() => ({ body: answer(c.now, positioned(A, { seen: 1.5, seen_pos: 2 })) }));
    const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env: {}, now: () => ourNow });
    const out = await backup.poll(reg1(), ourNow);
    assert.equal(out.upstreamNowMs, c.base, c.name);
    assert.equal(out.registry.planes[0].heard?.at, c.base - 1_500, `${c.name}: provider clock minus the row's own age`);
    // the worldwide snapshot dates the fix by the same clock and keeps it
    const snap = new GlobalSnapshot();
    const changed = snap.ingest({ provider: TRACKED_BACKUP_PROVIDER, origin: "tracked", aircraft: out.points, fetchedAt: ourNow, upstreamNowMs: out.upstreamNowMs }, ourNow);
    assert.equal(changed, 1, `${c.name}: not refused as too old`);
    assert.equal(snap.get(A)?.seenAt, c.base - 2_000);
  }
  // no usable clock in the answer: our own receive time
  const m = mockFetch(() => ({ body: { ac: [heardOnly(A)], msg: "No error" } }));
  const out = await createTrackedBackup({ fetchImpl: m.fetchImpl, env: {}, now: () => T0 }).poll(reg1(), T0);
  assert.equal(out.registry.planes[0].heard?.at, T0 - 500);
});

test("429 honours Retry-After (seconds), 403 waits 30 min, a network failure 30 s doubling; NO request is sent during a backoff", async () => {
  let t = T0;
  let script: Scripted = { status: 429, retryAfter: "120" };
  const m = mockFetch(() => script);
  const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env: {}, now: () => t });
  const registry = reg1(EARLIER);

  const limited = await backup.poll(registry, t);
  assert.equal(limited.ok, false);
  assert.equal(limited.status, 429);
  assert.equal(limited.requests, 1);
  assert.deepEqual(limited.points, []);
  assert.equal(limited.registry, registry, "a failed request changes nothing");
  assert.equal(backup.status(registry, t).backoff_until, T0 + 120_000, "Retry-After: 120 is 120 SECONDS");
  assert.equal(backup.status(registry, t).last_status, 429);
  assert.equal(backup.status(registry, t).failed_total, 1);
  for (const dt of [1_000, 10_000, 60_000, 119_999]) {
    t = T0 + dt;
    const held = await backup.poll(registry, t);
    assert.equal(held.skipped, "backoff", `+${dt} ms`);
    assert.equal(held.requests, 0);
  }
  assert.equal(m.calls.length, 1, "nothing was sent during the backoff");
  // the backoff is over: the next tick asks again, and a good answer clears it
  t = T0 + 120_000;
  script = { body: answer(t, positioned(A)) };
  const back = await backup.poll(registry, t);
  assert.equal(back.ok, true);
  assert.equal(m.calls.length, 2);
  assert.equal(backup.status(registry, t).backoff_until, null);
  assert.equal(backup.status(registry, t).consecutive_failures, 0);

  // a Retry-After shorter than the base backoff never shortens it
  script = { status: 429, retryAfter: "5" };
  await backup.poll(registry, t);
  assert.equal(backup.status(registry, t).backoff_until, t + FAILOVER_BACKOFF_BASE_MS);

  // 403: adsb.fi is refusing this server
  t += FAILOVER_BACKOFF_BASE_MS;
  script = { status: 403 };
  const refused = await backup.poll(registry, t);
  assert.equal(refused.status, 403);
  const refusedAt = t;
  assert.equal(backup.status(registry, t).backoff_until, refusedAt + FAILOVER_REFUSED_BACKOFF_MS);
  assert.equal(FAILOVER_REFUSED_BACKOFF_MS, 30 * 60_000);
  const sentBefore = m.calls.length;
  t = refusedAt + FAILOVER_REFUSED_BACKOFF_MS - 1;
  assert.equal((await backup.poll(registry, t)).skipped, "backoff");
  assert.equal(m.calls.length, sentBefore);
  t = refusedAt + FAILOVER_REFUSED_BACKOFF_MS;
  script = { body: answer(t) };
  assert.equal((await backup.poll(registry, t)).ok, true);

  // network failures: 30 s, then 60 s (in a row), and poll never throws
  script = { throws: "connect ECONNREFUSED" };
  const down = await backup.poll(registry, t);
  assert.equal(down.status, 0);
  assert.equal(down.error, "connect ECONNREFUSED");
  assert.equal(backup.status(registry, t).backoff_until, t + 30_000);
  t += 30_000;
  await backup.poll(registry, t);
  assert.equal(backup.status(registry, t).backoff_until, t + 60_000);
  assert.equal(backup.status(registry, t).consecutive_failures, 2);
  assert.equal(backup.status(registry, t).last_error, "connect ECONNREFUSED");

  // a 200 that is not an aircraft list is a failure, not "no aircraft"
  t += 60_000;
  script = { body: { error: "bad request" } };
  const odd = await backup.poll(registry, t);
  assert.equal(odd.ok, false);
  assert.equal(backup.status(registry, t).last_status, 200);
  assert.match(String(backup.status(registry, t).last_error), /without an `ac` list/);
  assert.ok((backup.status(registry, t).backoff_until ?? 0) > t);
});

test("a request that never answers is cut at the deadline and counted as a failure", async () => {
  const fetchImpl = (() => new Promise(() => undefined)) as unknown as typeof fetch;
  const backup = createTrackedBackup({ fetchImpl, env: {}, now: () => T0, timeoutMs: 30 });
  const out = await backup.poll(reg1(), T0);
  assert.equal(out.ok, false);
  assert.equal(out.status, 0);
  assert.match(String(out.error), /hard deadline/);
  assert.ok(TRACKED_BACKUP_TIMEOUT_MS < TRACKED_BACKUP_POLL_MS, "a hung request ends before the next tick is due");
});

test("cadence: every 10 s while a tracked plane is active, every 30 s otherwise", async () => {
  assert.equal(TRACKED_BACKUP_POLL_MS, 10_000);
  assert.equal(TRACKED_BACKUP_IDLE_POLL_MS, 30_000);
  const silent = reg1(EARLIER);
  assert.equal(trackedBackupPollMs(silent, T0), 30_000, "last seen 3 h ago: idle");
  assert.equal(trackedBackupPollMs(reg1({ last_heard: T0 - 10_000 }), T0), 10_000, "heard 10 s ago: active");
  assert.equal(trackedBackupPollMs(reg1({ last_seen: T0 - TRACKED_ACTIVE_WINDOW_MS + 1 }), T0), 10_000);
  assert.equal(trackedBackupPollMs(reg1({ last_seen: T0 - TRACKED_ACTIVE_WINDOW_MS - 1 }), T0), 30_000);
  assert.equal(trackedBackupPollMs({ planes: [plane({ hex: null, last_heard: T0 })] }, T0), 30_000, "no hex: nothing to ask by hex");
  assert.equal(trackedBackupPollMs({ planes: [plane(EARLIER), plane({ reg: "N786QS", hex: B, last_heard: T0 - 1_000 })] }, T0), 10_000, "ANY active plane");

  // a silent plane starts broadcasting: the loop asks every 30 s, then every 10 s, then every 30 s again
  let t = T0;
  let live = false;
  const m = mockFetch(() => ({ body: live ? answer(t, heardOnly(A, { seen: 0 })) : answer(t) }));
  const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env: {}, now: () => t });
  let registry = silent;
  const sentAt: number[] = [];
  const step = async () => {
    const out = await backup.poll(registry, t);
    if (out.requests) sentAt.push(t - T0);
    if (out.ok) registry = mergePolled(registry, out.registry);
    const wait = backup.nextDelayMs(registry, t);
    assert.equal(backup.status(registry, t).poll_s, wait / 1000, "the status states the cadence in force");
    t += wait;
  };
  await step(); await step();           // silent: 0, 30 s
  live = true;
  await step(); await step(); await step(); // heard at 60 s: 70 s, 80 s follow
  live = false;
  while (t - T0 <= 80_000 + TRACKED_ACTIVE_WINDOW_MS + 60_000) await step();
  assert.deepEqual(sentAt.slice(0, 5), [0, 30_000, 60_000, 70_000, 80_000]);
  const gaps = sentAt.slice(1).map((v, i) => v - sentAt[i]);
  assert.ok(gaps.every((g) => g === 10_000 || g === 30_000));
  // last heard at 80 s: 10 s ticks while inside the 5 min window, 30 s after it
  const lastFast = sentAt.filter((v, i) => i > 0 && v - sentAt[i - 1] === 10_000).pop()!;
  assert.ok(lastFast <= 80_000 + TRACKED_ACTIVE_WINDOW_MS + 10_000, `fast ticks stop once the window has passed (last at ${lastFast})`);
  assert.equal(gaps[gaps.length - 1], 30_000, "back to the idle cadence");
  // at most one request per 10 s: 0.1 request a second against adsb.fi's limit of 1
  assert.ok(Math.min(...gaps) >= TRACKED_BACKUP_POLL_MS);
});

test("mergePolled is newest-wins, interleaving 1: a slow adsb.lol cycle returns AFTER the backup wrote a newer position", async () => {
  // the adsb.lol hex request leaves at T0 and is held open
  let release: (v: unknown) => void = () => undefined;
  const slow = new Promise<unknown>((r) => { release = r; });
  const lol = mockFetch(() => ({ body: null }));
  const slowFetch = (async (url: string, init?: { headers?: Record<string, string> }) => {
    lol.calls.push({ url, headers: { ...(init?.headers ?? {}) } });
    const body = await slow;
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => body };
  }) as unknown as typeof fetch;
  let registry = reg1(EARLIER);
  const inFlight = pollHexesOnce(registry, [A], slowFetch, {}, T0); // folded into the registry AS IT IS NOW

  // meanwhile, 4 s later, adsb.fi has a position
  const fi = mockFetch(() => ({ body: answer(T0 + 4_000, positioned(A, { lat: 41.8, lon: -87.7 })) }));
  const backup = createTrackedBackup({ fetchImpl: fi.fetchImpl, env: {}, now: () => T0 + 4_000 });
  const fromFi = await backup.poll(registry, T0 + 4_000);
  registry = mergePolled(registry, fromFi.registry);
  assert.equal(registry.planes[0].last_seen, T0 + 4_000);

  // (a) adsb.lol answers "no such aircraft": its copy still holds the 3-hour-old position
  release({ ac: [], msg: "No error", now: T0 + 6_000, total: 0 });
  const fromLol = await inFlight;
  assert.equal(fromLol.registry.planes[0].last_seen, EARLIER.last_seen, "the slow cycle's copy is the OLD reading");
  const merged = mergePolled(registry, fromLol.registry);
  assert.equal(merged.planes[0].last_seen, T0 + 4_000, "the newer position is NOT replaced by the older one (it was, before 2026-11-19)");
  assert.deepEqual(merged.planes[0].last_pos, { la: 41.8, lo: -87.7, al: Math.round(2825 * 0.3048) });
  assert.equal(merged.planes[0].last_pos_provider, "adsbfi");
  assert.equal(merged.planes[0].heard?.has_position, true);
  assert.equal(merged.planes[0].last_heard, T0 + 4_000 - 200);

  // (b) the slow cycle instead comes back with a POSITION it stamped at its start (T0): still older
  const lolPos: TrackedRegistry = { planes: [{ ...plane(EARLIER), last_seen: T0, last_pos: { la: 41.7, lo: -87.9, al: 500 }, last_pos_provider: "adsblol",
    heard: { at: T0 - 300, alt_ft: 1640, gs_kt: 180, source: "adsb_icao", has_position: true }, heard_provider: "adsblol", last_heard: T0 - 300 }] };
  const b = mergePolled(registry, lolPos);
  assert.equal(b.planes[0].last_seen, T0 + 4_000);
  assert.equal(b.planes[0].last_pos_provider, "adsbfi");
  assert.equal(b.planes[0].heard_provider, "adsbfi");
  assert.equal(b.planes[0].last_heard, T0 + 4_000 - 200, "the high-water mark never goes back");

  // (c) the slow cycle comes back heard-only, with a reading NEWER than the backup's: each reading keeps its newest
  const lolHeard: TrackedRegistry = { planes: [{ ...plane(EARLIER), heard: { at: T0 + 5_500, alt_ft: 3100, gs_kt: 210, source: "mode_s", has_position: false },
    heard_provider: "adsblol", last_heard: T0 + 5_500 }] };
  const c = mergePolled(registry, lolHeard);
  assert.equal(c.planes[0].last_seen, T0 + 4_000, "the position stays the backup's: the heard-only copy holds an older one");
  assert.equal(c.planes[0].last_pos_provider, "adsbfi");
  assert.equal(c.planes[0].heard?.at, T0 + 5_500, "the transponder reading is adsb.lol's newer one");
  assert.equal(c.planes[0].heard_provider, "adsblol");
  assert.equal(c.planes[0].last_heard, T0 + 5_500);
});

test("mergePolled is newest-wins, interleaving 2: a slow adsb.fi answer returns AFTER the adsb.lol cycle wrote a newer position", async () => {
  let release: (v: Scripted) => void = () => undefined;
  const gate = new Promise<Scripted>((r) => { release = r; });
  const calls: string[] = [];
  const slowFi = (async (url: string) => {
    calls.push(url);
    const s = await gate;
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => s.body };
  }) as unknown as typeof fetch;
  let registry = reg1(EARLIER);
  const backup = createTrackedBackup({ fetchImpl: slowFi, env: {}, now: () => T0 });
  const inFlight = backup.poll(registry, T0); // folded into the registry AS IT IS NOW

  // meanwhile, 3 s later, adsb.lol's hex tick lands a position
  const lol = mockFetch(() => ({ body: { ac: [positioned(A, { lat: 41.75, lon: -87.85 })], msg: "No error", now: T0 + 3_000, total: 1 } }));
  const fromLol = await pollHexesOnce(registry, [A], lol.fetchImpl, {}, T0 + 3_000);
  registry = mergePolled(registry, fromLol.registry);
  assert.equal(registry.planes[0].last_seen, T0 + 3_000);
  assert.equal(registry.planes[0].last_pos_provider, "adsblol");

  // adsb.fi's answer arrives late: heard without a position, 1 s BEFORE adsb.lol's reading
  release({ body: answer(T0 + 2_000, heardOnly(A, { seen: 0 })) });
  const fromFi = await inFlight;
  assert.equal(fromFi.registry.planes[0].last_seen, EARLIER.last_seen, "the backup's copy predates adsb.lol's position");
  assert.equal(fromFi.registry.planes[0].heard?.has_position, false);
  const merged = mergePolled(registry, fromFi.registry);
  assert.equal(merged.planes[0].last_seen, T0 + 3_000, "adsb.lol's newer position stays");
  assert.deepEqual(merged.planes[0].last_pos, { la: 41.75, lo: -87.85, al: Math.round(2825 * 0.3048) });
  assert.equal(merged.planes[0].last_pos_provider, "adsblol");
  assert.equal(merged.planes[0].heard?.has_position, true, "and its newer transponder reading too");
  assert.equal(merged.planes[0].heard_provider, "adsblol");
  assert.equal(merged.planes[0].last_heard, T0 + 3_000 - 200);
  assert.equal(calls.length, 1);
});

test("mergePolled: with ONE loop the polled copy still lands whole; a hex is learned from either side; API edits made meanwhile survive", () => {
  // one loop: the polled copy is derived from the current registry and is never older
  const current = reg1({ ...EARLIER, heard: { at: T0 - H3, alt_ft: 0, gs_kt: 0, source: "adsb_icao", has_position: true }, last_heard: T0 - H3, backfilled_to: 900 });
  const polled = reg1({ last_seen: T0, last_pos: { la: 1, lo: 2, al: 3 }, last_pos_provider: "adsblol",
    heard: { at: T0 - 100, alt_ft: 10, gs_kt: 20, source: "adsb_icao", has_position: true }, heard_provider: "adsblol", last_heard: T0 - 100, backfilled_to: 100 });
  const one = mergePolled(current, polled).planes[0];
  assert.equal(one.last_seen, T0);
  assert.deepEqual(one.last_pos, { la: 1, lo: 2, al: 3 });
  assert.equal(one.heard?.at, T0 - 100);
  assert.equal(one.last_heard, T0 - 100);
  assert.equal(one.backfilled_to, 900, "the backfill's mark is not a poll's to write");
  // the same reading twice changes nothing; an untouched plane is returned as it is
  assert.deepEqual(mergePolled({ planes: [one] }, { planes: [one] }).planes[0], one);
  assert.equal(mergePolled(current, current).planes[0], current.planes[0]);

  // a hex learned by one loop is kept when the other loop's (hexless) copy lands
  const learned = mergePolled(reg1({ hex: null }), reg1({ hex: A, last_seen: T0, last_pos: { la: 1, lo: 2, al: null } }));
  assert.equal(learned.planes[0].hex, A);
  const kept = mergePolled(learned, reg1({ hex: null }));
  assert.equal(kept.planes[0].hex, A, "an older hexless copy does not unlearn it");
  assert.equal(kept.planes[0].last_seen, T0);
  // a plane never heard on either side keeps empty fields, not zeros
  const never = mergePolled(reg1({ hex: null }), { planes: [{ ...plane({ hex: null }) }] }).planes[0];
  assert.equal(never.last_seen, undefined);
  assert.equal(never.last_heard, undefined);
  assert.equal(never.heard, undefined);
});

test("the adsb.lol API key is NEVER sent to adsb.fi: a plain User-Agent only", async () => {
  const SECRET = "lol-key-7f3a-do-not-leak";
  const env = { ADSBLOL_API_KEY: SECRET };
  const m = mockFetch(() => ({ body: answer(T0, positioned(A)) }));
  const backup = createTrackedBackup({ fetchImpl: m.fetchImpl, env, now: () => T0 });
  await backup.poll(reg1(), T0);
  assert.equal(m.calls.length, 1);
  assert.deepEqual(Object.keys(m.calls[0].headers), ["User-Agent"], "one header");
  assert.match(m.calls[0].headers["User-Agent"], /^voltradeai-datacore\//);
  assert.ok(!Object.keys(m.calls[0].headers).some((k) => k.toLowerCase() === "x-api-key"));
  assert.ok(!JSON.stringify(m.calls[0]).includes(SECRET), "not in the URL, not in a header");
  assert.ok(!JSON.stringify(backup.status(reg1(), T0)).includes(SECRET), "and not in the status");
  // the module never builds adsb.lol's headers at all (a comment may name them; an import or a call may not)
  const code = fs.readFileSync(path.join(here, "trackedBackup.ts"), "utf8").split("\n").filter((l) => !l.trim().startsWith("//")).join("\n");
  assert.ok(!/adsbLolRequestHeaders|adsbLolKeyHeader|ADSBLOL_API_KEY|x-api-key/.test(code));
  assert.ok(!code.includes("api.adsb.lol"), "and asks adsb.lol nothing");
});

test("the poller end to end: an adsb.fi fix reaches the registry, the fix bus, the snapshot row, the archive and the plane's log with provider adsbfi; adsb.lol alone gets the key", async () => {
  const SECRET = "lol-key-e2e-do-not-leak";
  const HEX = "a0f1e2"; // its own archive-thinning key
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tracked-backup-"));
  saveRegistry({ planes: [{ reg: "N843S", added_at: 1, hex: HEX, ...EARLIER }] }, base);
  const batches: FixBatch[] = [];
  const snap = new GlobalSnapshot();
  const offBus = subscribeFixes((b) => { batches.push(b); snap.ingest(b); });
  // adsb.lol knows nothing about the plane; adsb.fi has its position
  const m = mockFetch((url) => (url.startsWith("https://opendata.adsb.fi/")
    ? { body: answer(Date.now(), positioned(HEX)) }
    : { body: { ac: [], msg: "No error", now: Date.now(), total: 0 } }));
  const poller = startTrackedPoller({
    fetchImpl: m.fetchImpl, headers: { "User-Agent": "test" }, sites: [], baseDir: base, governor: quietGovernor(),
    env: { ADSBLOL_API_KEY: SECRET }, backupStartDelayMs: 0,
  });
  const log = new TrackedPlaneLog({ baseDir: base, env: {}, readFree: () => 50 * 1_073_741_824, tracked: () => poller.getRegistry().planes });
  const offLog = log.start();
  try {
    for (let i = 0; i < 300 && !batches.length; i++) await new Promise((r) => setTimeout(r, 10));
    assert.equal(batches.length, 1, "one batch: adsb.lol had nothing to publish");
    const b = batches[0];
    assert.equal(b.provider, "adsbfi");
    assert.equal(b.origin, "tracked", "the same origin as the adsb.lol cycle: the same consumers");
    assert.equal(b.aircraft.length, 1);
    assert.equal(b.aircraft[0].provider, "adsbfi");
    assert.ok(typeof b.upstreamNowMs === "number" && b.upstreamNowMs <= Date.now(), "dated by adsb.fi's own clock");

    // the registry the API serves
    const tp = poller.getRegistry().planes[0];
    assert.deepEqual(tp.last_pos, { la: 41.787, lo: -87.8, al: Math.round(2825 * 0.3048) });
    assert.equal(tp.last_pos_provider, "adsbfi");
    assert.equal(tp.heard_provider, "adsbfi");
    assert.ok((tp.last_seen ?? 0) > EARLIER.last_seen);
    const onDisk = JSON.parse(fs.readFileSync(registryPath(base), "utf8")) as TrackedRegistry;
    assert.equal(onDisk.planes[0].last_pos_provider, "adsbfi", "and the registry file");

    // GET /api/data/aircraft/live/:hex serves encodeRow(snapshot row): `src` is the provider
    const row = snap.get(HEX);
    assert.equal(row?.src, "adsbfi");
    assert.equal(row?.via, "tracked");
    assert.equal(encodeRow(row!)[ROW_FIELDS.indexOf("src")], "adsbfi");

    // the hour archive line and the tracked plane's permanent log line: pv
    const files = (dir: string): string[] => (fs.existsSync(dir) ? fs.readdirSync(dir, { withFileTypes: true }) : [])
      .flatMap((d) => (d.isDirectory() ? files(path.join(dir, d.name)) : d.name.endsWith(".jsonl") ? [path.join(dir, d.name)] : []));
    const lines = (dir: string): Raw[] => files(dir)
      .flatMap((fp) => fs.readFileSync(fp, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as Raw));
    const archived = lines(path.join(base, "aircraft"));
    assert.equal(archived.length, 1);
    assert.equal(archived[0].i, HEX);
    assert.equal(archived[0].pv, "adsbfi");
    const logged = lines(path.join(base, "aircraft_tracked"));
    assert.equal(logged.length, 1);
    assert.equal(logged[0].pv, "adsbfi");
    assert.equal(logged[0].rg, "N843S");

    // the status block the endpoint serves
    const st = poller.backupStatus();
    assert.equal(st.enabled, true);
    assert.equal(st.provider, "adsbfi");
    assert.equal(st.ok_total, 1);
    assert.equal(st.fixes_total, 1);
    assert.equal(st.poll_s, 10, "the plane is active now");
    assert.ok(!JSON.stringify(st).includes(SECRET));

    // the key rides on adsb.lol's requests and on nothing else
    const fi = m.calls.filter((c) => c.url.startsWith("https://opendata.adsb.fi/"));
    const lol = m.calls.filter((c) => c.url.startsWith("https://api.adsb.lol/"));
    assert.equal(fi.length, 1);
    assert.equal(fi[0].url, `https://opendata.adsb.fi/api/v2/hex/${HEX}`);
    assert.ok(!JSON.stringify(fi).includes(SECRET), "adsb.fi never sees the adsb.lol key");
    assert.deepEqual(Object.keys(fi[0].headers), ["User-Agent"]);
    assert.ok(lol.length >= 1 && lol.every((c) => c.headers["x-api-key"] === SECRET), "adsb.lol's own requests are unchanged");
  } finally {
    poller.stop();
    offLog();
    offBus();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("the poller with the backup switched off sends adsb.fi nothing and says why", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tracked-backup-off-"));
  saveRegistry({ planes: [{ reg: "N843S", added_at: 1, hex: A, ...EARLIER }] }, base);
  const m = mockFetch(() => ({ body: { ac: [], msg: "No error", now: Date.now(), total: 0 } }));
  const poller = startTrackedPoller({
    fetchImpl: m.fetchImpl, headers: {}, sites: [], baseDir: base, governor: quietGovernor(),
    env: { STRIPE_SECRET_KEY: "sk_test_x" }, backupStartDelayMs: 0,
  });
  try {
    await new Promise((r) => setTimeout(r, 80));
    assert.ok(m.calls.length >= 1, "the adsb.lol cycle ran");
    assert.ok(m.calls.every((c) => c.url.startsWith("https://api.adsb.lol/")), "no request to adsb.fi while billing is active");
    const st = poller.backupStatus();
    assert.equal(st.enabled, false);
    assert.equal(st.off_reason, "billing is active: adsb.fi is non-commercial");
    assert.equal(st.requests_total, 0);
  } finally {
    poller.stop();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("wiring: startTrackedPoller runs the backup as its own unref'd loop through ONE application path; routes.ts serves its status", () => {
  const src = fs.readFileSync(path.join(here, "trackedPlanes.ts"), "utf8");
  const start = src.slice(src.indexOf("export function startTrackedPoller("));
  // built once, from the poller's own fetch and env
  assert.equal(src.split("createTrackedBackup(").length - 1, 1);
  assert.match(start, /const backup = createTrackedBackup\(\{ fetchImpl: deps\.fetchImpl, env: deps\.env \}\);/);
  // both loops land their readings through applyOutcome: merge, fix bus, archive, coalesced save
  const apply = start.slice(start.indexOf("const applyOutcome = ("), start.indexOf("const tick = async () => {"));
  assert.match(apply, /registry = mergePolled\(registry, out\.registry\);/);
  assert.match(apply, /publishFixes\(\{ provider, origin: "tracked", aircraft: out\.points, fetchedAt: at/);
  assert.match(apply, /archiveAircraft\(out\.points, deps\.sites, deps\.baseDir, at, TRACKED_ARCHIVE_INTERVAL_MS\)/);
  assert.match(apply, /saveRegistry\(registry, deps\.baseDir\);/);
  assert.equal(start.split("publishFixes(").length - 1, 1, "ONE publish site: neither loop has a path of its own");
  assert.equal(start.split("mergePolled(").length - 1, 1);
  assert.match(start, /applyOutcome\(out, started, TRACKED_PRIMARY_PROVIDER, out\.kind === "base"\)/);
  assert.match(start, /const out = await backup\.poll\(registry, started\);/);
  assert.match(start, /applyOutcome\(out, started, TRACKED_BACKUP_PROVIDER, false, out\.upstreamNowMs\)/);
  // a timer of its own, never holding the process open, cleared by stop()
  assert.match(start, /backupTimer = setTimeout\(\(\) => \{ void backupTick\(\); \}, Math\.max\(1_000, backup\.nextDelayMs\(registry, t\) - \(t - started\)\)\);\s+unrefTimer\(backupTimer\);/);
  assert.match(start, /backupTimer = setTimeout\(\(\) => \{ void backupTick\(\); \}, Math\.max\(0, deps\.backupStartDelayMs \?\? TRACKED_BACKUP_START_DELAY_MS\)\);\s+unrefTimer\(backupTimer\);/);
  assert.match(start, /if \(backupTimer\) clearTimeout\(backupTimer\);/);
  assert.match(start, /backupStatus: \(\) => backup\.status\(registry, Date\.now\(\)\)/);
  // adsb.lol's own pacing is not touched by the backup: the cycle is built and scheduled as before
  assert.match(start, /const cycle = createTrackedCycle\(\{ fetchImpl: deps\.fetchImpl, headers: deps\.headers, governor: deps\.governor, env: deps\.env \}\);/);
  assert.match(start, /const wait = cycle\.nextDelayMs\(registry, t\) - \(p\.paced \|\| p\.blocked \? 0 : t - started\);/);

  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('import { TRACKED_PROVIDER_NOTE, trackedSourceText } from "./trackedBackup";'));
  const handler = routes.slice(routes.indexOf('app.get("/api/data/aircraft/tracked"'), routes.indexOf('app.post("/api/data/aircraft/tracked/:reg"'));
  assert.match(handler, /const trackedBackup = trackedPoller\.backupStatus\(\);/);
  assert.match(handler, /backup: trackedBackup,/);
  assert.match(handler, /source: trackedSourceText\(trackedBackup\),/);
  assert.match(handler, /note: "tracked tails are archived continuously[^"]*" \+ TRACKED_PROVIDER_NOTE,/, "the row-field note keeps its text and gains the provider fields");
  // the adsb.lol blocks of the answer are still there
  for (const key of ["poll_s: TRACKED_POLL_MS / 1000", "effective_poll_s: trackedPacing.effective_poll_s", "pacing: trackedPacing", "cycles: trackedPoller.cycleStats()"]) {
    assert.ok(handler.includes(key), key);
  }
  // adsb.fi stays on the monetization tripwire's list
  const compliance = fs.readFileSync(path.join(here, "providerCompliance.ts"), "utf8");
  assert.match(compliance, /NON_COMMERCIAL_AIRCRAFT_PROVIDERS = \[[^\]]*"adsbfi"/);
});
