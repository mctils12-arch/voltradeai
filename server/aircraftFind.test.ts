// GET /api/data/aircraft/find?q= — one plane, anywhere, in one of four honest
// states (owner report 2026-11-12, N843S: the Find box answered `no "N843S"
// in the current snapshot` for a plane that was airborne outside the
// viewport / heard without a position).
// Run: npx tsx --test server/aircraftFind.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { AddressInfo } from "node:net";
import {
  FIND_CACHE_MS, FIND_LIVE_MAX_AGE_MS, FIND_MAX_UPSTREAM_PER_MIN, createAircraftFinder, parseFindQuery,
  registerAircraftFindRoute, type FindDeps, type FindResult,
} from "./aircraftFind";
import { GlobalSnapshot, ROW_FIELDS } from "./globalSnapshot";
import { UpstreamGovernor } from "./adsbGovernor";
import type { FixBatch } from "./aircraftFixBus";
import type { AircraftPoint } from "./datacoreArchive";
import type { TrackedPlane } from "./trackedPlanes";

const here = path.dirname(fileURLToPath(import.meta.url));
const T0 = Date.UTC(2026, 10, 12, 19, 49, 43);

const fix = (o: Partial<AircraftPoint> = {}): AircraftPoint => ({
  icao24: "ab8c8e", callsign: "N843S", registration: "N843S", lat: 41.787, lon: -87.8, altitude_m: 861,
  on_ground: false, velocity_ms: 104, heading: 148, type: "FA7X", category: "A2", ...o,
});

interface Rig {
  snapshot: GlobalSnapshot;
  tracked: TrackedPlane[];
  calls: string[];
  published: FixBatch[];
  clock: { t: number };
  deps: FindDeps;
}
function rig(o: {
  upstream?: (url: string) => { status?: number; body?: unknown };
  archived?: { t: number; la: number; lo: number; al?: number } | null;
} = {}): Rig {
  const snapshot = new GlobalSnapshot();
  const tracked: TrackedPlane[] = [];
  const calls: string[] = [];
  const published: FixBatch[] = [];
  const clock = { t: T0 };
  const fetchImpl = (async (url: string) => {
    calls.push(url);
    const a = o.upstream?.(url) ?? { body: { ac: [], now: clock.t } };
    const status = a.status ?? 200;
    return { ok: status === 200, status, headers: { get: () => null }, json: async () => a.body };
  }) as unknown as typeof fetch;
  const deps: FindDeps = {
    snapshot, tracked: () => tracked, fetchImpl, now: () => clock.t,
    governor: new UpstreamGovernor({ bgRps: 0.4, ceilingRps: 1 }),
    publish: (b) => { published.push(b); snapshot.ingest(b, clock.t); },
    lastArchived: async () => o.archived ?? null,
  };
  return { snapshot, tracked, calls, published, clock, deps };
}
const q = (s: string) => {
  const p = parseFindQuery(s);
  assert.ok(p, `query ${s} should parse`);
  return p!;
};
const col = (r: FindResult, name: string) => r.rows[0][r.fields.indexOf(name)];

test("input validation: a tail number, a callsign or a 6-digit hex — nothing else", () => {
  assert.deepEqual(parseFindQuery("n843s"), { ident: "N843S", hex: null });
  assert.deepEqual(parseFindQuery(" G-EZTL "), { ident: "G-EZTL", hex: null });
  assert.deepEqual(parseFindQuery("ab8c8e"), { ident: "AB8C8E", hex: "ab8c8e" }, "6 hex digits is both a hex and a possible callsign");
  assert.deepEqual(parseFindQuery("UAL123"), { ident: "UAL123", hex: null });
  for (const bad of ["", " ", "N", "N843S; DROP", "../etc/passwd", "N843S,N1", "-N843S", "N 843S", "WAY-TOO-LONG-REG", "N843S\nN1", "%2e%2e"]) {
    assert.equal(parseFindQuery(bad), null, `refused: ${JSON.stringify(bad)}`);
  }
  for (const notString of [undefined, null, 843, ["N843S", "N1"], { q: "N843S" }]) {
    assert.equal(parseFindQuery(notString), null);
  }
});

test("LIVE from the snapshot (which holds the tracked fixes): the row and its age, no upstream request", async () => {
  const r = rig();
  r.snapshot.ingest({ provider: "adsblol", origin: "tracked", aircraft: [fix()], fetchedAt: T0 - 4_000 }, T0 - 4_000);
  const finder = createAircraftFinder(r.deps);
  for (const query of ["N843S", "ab8c8e"]) {
    const out = await finder.find(q(query));
    assert.equal(out.state, "live", query);
    assert.equal(out.kind, "raw");
    assert.equal(out.hex, "ab8c8e");
    assert.equal(out.age_s, 4);
    assert.deepEqual(out.fields, ROW_FIELDS, "the /global wire shape");
    assert.equal(col(out, "lat"), 41.787);
    assert.equal(col(out, "lon"), -87.8);
    assert.equal(col(out, "src"), "adsblol");
    assert.equal(out.answered_from, "snapshot");
    assert.equal(out.upstream, "not-needed");
    assert.match(out.source, /adsb\.lol \(ODbL\)/);
    assert.ok(out.honesty.length > 50);
  }
  assert.deepEqual(r.calls, []);
});

test("LIVE by callsign: exact match only (a prefix never matches another aircraft)", async () => {
  const r = rig();
  r.snapshot.ingest({ provider: "adsblol", origin: "sweep-disc", fetchedAt: T0, aircraft: [
    fix({ icao24: "a11111", callsign: "UAL1234", registration: "N12345" }),
  ] }, T0);
  const finder = createAircraftFinder(r.deps);
  assert.equal((await finder.find(q("UAL1234"))).state, "live");
  const near = await finder.find(q("UAL123"));
  assert.equal(near.state, "unknown", "UAL123 is not UAL1234");
  assert.deepEqual(near.rows, []);
});

test("HEARD from the tracked registry: transponder heard now, no position — altitude and speed only, no upstream request", async () => {
  const r = rig();
  r.tracked.push({ reg: "N843S", added_at: 1, hex: "ab8c8e", last_seen: T0 - 3 * 3600_000, last_pos: { la: 41.9221, lo: -72.7071, al: null },
    heard: { at: T0 - 2_000, alt_ft: 850, gs_kt: 144.3, source: "mode_s", has_position: false }, last_heard: T0 - 2_000 });
  const out = await createAircraftFinder(r.deps).find(q("n843s"));
  assert.equal(out.state, "heard");
  assert.deepEqual(out.heard, { at: T0 - 2_000, alt_ft: 850, gs_kt: 144.3, source: "mode_s", has_position: false });
  assert.equal(out.age_s, 2);
  assert.deepEqual(out.rows, [], "a heard plane has NO position row — none is invented");
  assert.equal(out.last_seen?.via, "tracked", "its last real position is offered separately, with its own time");
  assert.equal(out.last_seen?.at, T0 - 3 * 3600_000);
  assert.equal(out.tracked, true);
  assert.equal(out.answered_from, "tracked");
  assert.deepEqual(r.calls, []);
});

test("HEARD from ONE upstream lookup by hex (the 19:49:43Z answer): mode_s, 850 ft, 144.3 kt, no lat/lon", async () => {
  const r = rig({ upstream: () => ({ body: { now: T0, ac: [{ hex: "ab8c8e", type: "mode_s", alt_baro: 850, gs: 144.3, seen: 0.0 }] } }) });
  r.tracked.push({ reg: "N843S", added_at: 1, hex: "ab8c8e" }); // the registry knows the hex, has heard nothing yet
  const out = await createAircraftFinder(r.deps).find(q("N843S"));
  assert.deepEqual(r.calls, ["https://api.adsb.lol/v2/hex/ab8c8e"], "by HEX: the registration query does not return a plane heard without a position");
  assert.equal(out.state, "heard");
  assert.equal(out.heard?.alt_ft, 850);
  assert.equal(out.heard?.gs_kt, 144.3);
  assert.equal(out.heard?.has_position, false);
  assert.deepEqual(out.rows, []);
  assert.equal(out.last_seen, null);
  assert.equal(out.upstream, "found");
  assert.deepEqual(r.published, [], "nothing is published to the fix bus without a position");
});

test("LIVE from ONE upstream lookup by registration: the fix is published to the fix bus and returned", async () => {
  const r = rig({ upstream: () => ({ body: { now: T0, ac: [
    { hex: "a42dee", r: "N369CA", t: "FA7X", flight: "N369CA  ", alt_baro: 38000, gs: 470, track: 90, lat: 39.1, lon: -94.6, seen: 0.2, seen_pos: 0.4 },
  ] } }) });
  const out = await createAircraftFinder(r.deps).find(q("N369CA"));
  assert.deepEqual(r.calls, ["https://api.adsb.lol/v2/reg/N369CA"]);
  assert.equal(out.state, "live");
  assert.equal(out.hex, "a42dee");
  assert.equal(col(out, "reg"), "N369CA");
  assert.equal(col(out, "lat"), 39.1);
  assert.equal(out.answered_from, "upstream");
  assert.equal(r.published.length, 1);
  assert.equal(r.published[0].origin, "find");
  assert.equal(r.published[0].provider, "adsblol");
  assert.ok(r.snapshot.get("a42dee"), "the snapshot now holds it for every other viewer");
});

test("upstream row for a DIFFERENT tail is never returned (reg query answers are matched exactly)", async () => {
  const r = rig({ upstream: () => ({ body: { now: T0, ac: [{ hex: "a00001", r: "N1", lat: 1, lon: 1, alt_baro: 1000, seen_pos: 0 }] } }) });
  const out = await createAircraftFinder(r.deps).find(q("N843S"));
  assert.equal(out.state, "unknown");
  assert.equal(out.upstream, "empty");
  assert.deepEqual(r.published, []);
});

test("LAST_SEEN: no signal now — the newest position held (registry vs archive vs stale snapshot row) with its time", async () => {
  const r = rig({ archived: { t: Math.floor((T0 - 40 * 60_000) / 1000), la: 41.7868, lo: -87.7522, al: 189 } });
  r.tracked.push({ reg: "N843S", added_at: 1, hex: "ab8c8e", last_seen: T0 - 3 * 3600_000, last_pos: { la: 41.9221, lo: -72.7071, al: 53 } });
  const out = await createAircraftFinder(r.deps).find(q("N843S"));
  assert.equal(out.state, "last_seen");
  assert.deepEqual(r.calls, ["https://api.adsb.lol/v2/hex/ab8c8e"], "one upstream lookup was made first");
  assert.equal(out.upstream, "empty");
  assert.equal(out.last_seen?.via, "archive", "the archive's fix is newer than the registry's");
  assert.equal(out.last_seen?.at, Math.floor((T0 - 40 * 60_000) / 1000) * 1000);
  assert.equal(out.last_seen?.lat, 41.7868);
  assert.equal(out.last_seen?.alt_ft, Math.round(189 / 0.3048));
  assert.equal(out.age_s, 40 * 60);
  assert.deepEqual(out.rows, [], "an old position is never served as a live row");
  assert.equal(out.hex, "ab8c8e");
  assert.equal(out.tracked, true);
});

test("a snapshot row older than the live line is last_seen, not live", async () => {
  const r = rig();
  r.snapshot.ingest({ provider: "adsblol", origin: "sweep-disc", aircraft: [fix()], fetchedAt: T0 - FIND_LIVE_MAX_AGE_MS - 30_000 }, T0 - FIND_LIVE_MAX_AGE_MS - 30_000);
  const out = await createAircraftFinder(r.deps).find(q("ab8c8e"));
  assert.equal(out.state, "last_seen");
  assert.equal(out.last_seen?.via, "snapshot");
  assert.equal(out.age_s, (FIND_LIVE_MAX_AGE_MS + 30_000) / 1000);
});

test("UNKNOWN: nothing in the snapshot, the registry, the provider or the archive", async () => {
  const r = rig();
  const out = await createAircraftFinder(r.deps).find(q("N000ZZ"));
  assert.equal(out.state, "unknown");
  assert.equal(out.hex, null);
  assert.equal(out.age_s, null);
  assert.deepEqual(out.rows, []);
  assert.equal(out.heard, null);
  assert.equal(out.last_seen, null);
  assert.equal(out.upstream, "empty");
  assert.deepEqual(r.calls, ["https://api.adsb.lol/v2/reg/N000ZZ"]);
});

test("politeness: FIND_CACHE_MS cache (re-aged), in-flight coalescing, and an upstream ceiling per minute", async () => {
  const r = rig();
  const finder = createAircraftFinder(r.deps);
  const [a, b] = await Promise.all([finder.find(q("N000ZZ")), finder.find(q("n000zz"))]);
  assert.equal(r.calls.length, 1, "two concurrent finds of one tail share ONE upstream request");
  assert.equal(a.state, b.state);
  r.clock.t += FIND_CACHE_MS - 1;
  await finder.find(q("N000ZZ"));
  assert.equal(r.calls.length, 1, "inside the cache window: no new request");
  r.clock.t += 2;
  await finder.find(q("N000ZZ"));
  assert.equal(r.calls.length, 2, "past it: asked again");
  // a burst of distinct tails: only FIND_MAX_UPSTREAM_PER_MIN leave per minute
  for (let i = 0; i < 30; i++) await finder.find(q(`N${100 + i}XY`));
  assert.equal(r.calls.length, FIND_MAX_UPSTREAM_PER_MIN, "the ceiling counts every upstream request in the rolling minute");
  const limited = await finder.find(q("N777LM"));
  assert.equal(limited.upstream, "limited", "over the ceiling the route says so instead of asking");
  assert.equal(limited.state, "unknown");
  r.clock.t += 61_000;
  const again = await finder.find(q("N777LM"));
  assert.equal(again.upstream, "empty", "a minute later the lookup runs");
  // a cached live answer is re-aged, never served with its old clock
  r.snapshot.ingest({ provider: "adsblol", origin: "tracked", aircraft: [fix()], fetchedAt: r.clock.t }, r.clock.t);
  const first = await finder.find(q("N843S"));
  r.clock.t += 3_000;
  const cached = await finder.find(q("N843S"));
  assert.equal(first.age_s, 0);
  assert.equal(cached.age_s, 3);
  assert.equal(cached.at, r.clock.t);
});

test("an upstream failure is reported, never thrown: the answer falls back to what the server holds", async () => {
  const r = rig({ upstream: () => ({ status: 503 }) });
  r.tracked.push({ reg: "N843S", added_at: 1, hex: "ab8c8e", last_seen: T0 - 3600_000, last_pos: { la: 41.9, lo: -72.7, al: null } });
  const out = await createAircraftFinder(r.deps).find(q("N843S"));
  assert.equal(out.upstream, "error");
  assert.equal(out.state, "last_seen");
  assert.equal(out.last_seen?.via, "tracked");
});

test("the route on a real Express app: 400 on a bad q, the four-state body, no-store", async () => {
  const r = rig();
  r.tracked.push({ reg: "N843S", added_at: 1, hex: "ab8c8e",
    heard: { at: T0 - 1_000, alt_ft: 850, gs_kt: 144.3, source: "mode_s", has_position: false }, last_heard: T0 - 1_000 });
  const app = express();
  registerAircraftFindRoute(app, r.deps);
  const server = app.listen(0);
  try {
    const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    for (const bad of ["", "?q=", "?q=N843S%20OR%201", "?q=N1&q=N2", "?q=..%2F..", "?q=THIS-IS-TOO-LONG"]) {
      const res = await fetch(`${base}/api/data/aircraft/find${bad}`);
      assert.equal(res.status, 400, `400 for ${bad || "(no q)"}`);
      assert.match(String(((await res.json()) as { error?: string }).error), /tail number/);
    }
    const res = await fetch(`${base}/api/data/aircraft/find?q=n843s`);
    assert.equal(res.status, 200);
    assert.equal(res.headers.get("cache-control"), "no-store");
    const d = (await res.json()) as FindResult;
    assert.equal(d.kind, "raw");
    assert.equal(d.q, "N843S");
    assert.equal(d.state, "heard");
    assert.equal(d.heard?.alt_ft, 850);
    assert.deepEqual(d.rows, []);
    assert.ok(d.source && d.honesty);
  } finally {
    server.close();
  }
});

test("routes.ts wiring: one registration, fed by the global snapshot and the tracked registry; /tracked exposes heard", () => {
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.equal(src.split("registerAircraftFindRoute(app").length - 1, 1);
  assert.ok(src.includes('import { registerAircraftFindRoute } from "./aircraftFind"'));
  assert.match(src, /snapshot: globalAircraft\.snapshot/);
  assert.match(src, /tracked: \(\) => trackedPoller\.getRegistry\(\)\.planes/);
  assert.match(src, /fast_poll_s: TRACKED_FAST_POLL_MS \/ 1000/);
  assert.match(src, /heard \/ last_heard say the provider heard the transponder/);
  // community ADS-B only: this feature adds no FAA/SWIM-sourced data
  const find = fs.readFileSync(path.join(here, "aircraftFind.ts"), "utf8");
  assert.ok(!/from "\.\/flightPlans"|swim|faa\.gov/i.test(find.replace(/\/\/.*$/gm, "")), "no FAA/SWIM source in the find path");
});
