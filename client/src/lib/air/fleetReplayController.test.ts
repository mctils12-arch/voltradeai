import { test } from "node:test";
import assert from "node:assert/strict";
import { FrameLoop, type FrameHost } from "../../render/frameCore.ts";
import { liveLayers } from "../../render/layerContract.ts";
import { FleetReplayController, framingOffset, playRateSecPerSec, springEase, type ReplayState } from "./fleetReplayController.ts";
import type { FleetReplayLayer } from "./fleetReplayLayer.ts";
import type { CloseApproachEntry } from "./closeApproachView.ts";
import type maplibregl from "maplibre-gl";

const T0 = 1_759_000_000;

interface EaseCall { center: [number, number]; zoom: number; pitch: number; bearing: number; easing?: (x: number) => number }
interface FakeMap {
  pose: { lng: number; lat: number; zoom: number; pitch: number; bearing: number };
  layers: Map<string, FleetReplayLayer>;
  calls: { easeTo: EaseCall[]; repaint: number };
  asMap(): maplibregl.Map;
}

/** minimal MapLibre stand-in: only what the controller touches */
function fakeMap(pose = { lng: -100, lat: 40, zoom: 8, pitch: 0, bearing: 0 }): FakeMap {
  const layers = new Map<string, FleetReplayLayer>();
  const calls = { easeTo: [] as EaseCall[], repaint: 0 };
  const fm: FakeMap = { pose, layers, calls, asMap: () => impl as unknown as maplibregl.Map };
  const impl = {
    getCenter: () => ({ lng: fm.pose.lng, lat: fm.pose.lat }),
    getZoom: () => fm.pose.zoom,
    getPitch: () => fm.pose.pitch,
    getBearing: () => fm.pose.bearing,
    getContainer: () => ({ clientWidth: 1000, clientHeight: 800, appendChild() { return null; } }),
    getLayer: (id: string) => layers.get(id),
    addLayer: (l: FleetReplayLayer) => { layers.set(l.id, l); },
    removeLayer: (id: string) => { layers.delete(id); },
    isStyleLoaded: () => true,
    triggerRepaint: () => { calls.repaint++; },
    easeTo: (o: EaseCall) => { calls.easeTo.push(o); },
    getTerrain: () => null,
  };
  return fm;
}

function manualLoop(): { loop: FrameLoop; step: (ms: number, n?: number) => void; now: () => number } {
  let clock = 1000;
  const host: FrameHost = {
    now: () => clock,
    scheduler: { request: () => 0, cancel: () => {} },
    visibility: { isHidden: () => false, subscribe: () => () => {} },
  };
  const loop = new FrameLoop(() => host);
  return {
    loop,
    now: () => clock,
    step: (ms: number, n = 1) => { for (let i = 0; i < n; i++) { clock += ms; loop.tick(clock); } },
  };
}

function windowPayload(opts: { hexes?: number; from: number; to: number; capped?: boolean; ca?: CloseApproachEntry[] }) {
  const hexes = [];
  const n = opts.hexes ?? 3;
  for (let i = 0; i < n; i++) {
    const points: Array<[number, number, number, number | null]> = [];
    for (let k = 0; k <= 12; k++) points.push([opts.from + k * 300, 40 + i * 0.05 + k * 0.01, -100 + k * 0.02, 9000 + i * 100]);
    hexes.push({ i: `h${i.toString(16).padStart(5, "0")}`, c: `CS${i}`, points, raw_count: points.length, truncated: false });
  }
  return {
    kind: "aircraft", from: opts.from, to: opts.to, zoom: 8, step_sec: 300,
    hexes, hexes_seen: opts.capped ? n + 50 : n, total_points: hexes.length * 13,
    coverage: { requested_from: opts.from, scanned_from: opts.from, complete: true, files_scanned: 1 },
    closeApproaches: opts.ca ?? [],
    closeApproachesMeta: { evaluated_hexes: n, found: (opts.ca ?? []).length, returned: (opts.ca ?? []).length, capped: false, partial_scan: false },
  };
}

type Pending = { url: string; resolve: (v: unknown) => void; signal?: AbortSignal };
function fakeFetch() {
  const pending: Pending[] = [];
  const impl = ((url: string, init?: RequestInit) => new Promise((resolve, reject) => {
    const p: Pending = { url, resolve, signal: init?.signal ?? undefined };
    init?.signal?.addEventListener("abort", () => {
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    });
    pending.push(p);
  })) as unknown as typeof fetch;
  const respond = (p: Pending, body: unknown, ok = true) =>
    p.resolve({ ok, status: ok ? 200 : 500, json: async () => body });
  return { impl, pending, respond };
}

const flush = () => new Promise((r) => setTimeout(r, 170));

test("the window read follows the camera TARGET: settled pose, not mid-gesture frames", async () => {
  const map = fakeMap();
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  const states: ReplayState[] = [];
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "full", onState: (s) => states.push(s), onPlayhead: () => {} });
  const from = T0, to = T0 + 3600;
  c.setQuery({ kind: "aircraft", fromSec: from, toSec: to, stepSec: 300 });
  step(16);
  assert.equal(f.pending.length, 1, "opening fetches immediately from the current pose");
  assert.match(f.pending[0].url, /\/api\/data\/aircraft\/window\?bbox=/);
  assert.match(f.pending[0].url, /&max=2000/, "full tier asks for the full fleet");
  f.respond(f.pending[0], windowPayload({ from, to }));
  await flush();
  step(16, 3);
  // a pan in progress: poses change every frame → no request
  for (let k = 0; k < 20; k++) { map.pose.lng += 0.2; step(16); }
  assert.equal(f.pending.length, 1, "no request while the destination is unknown");
  // the gesture stops; 250 ms of stillness → the settled pose is the target
  step(16, 20);
  assert.equal(f.pending.length, 2, "one re-read once the view settled outside the fetched box");
  c.dispose();
});

test("a newer target aborts the older read; the old fleet stays drawn until the new lands", async () => {
  const map = fakeMap();
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  let last: ReplayState | null = null;
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "reduced", onState: (s) => { last = s; }, onPlayhead: () => {} });
  const from = T0, to = T0 + 3600;
  c.setQuery({ kind: "aircraft", fromSec: from, toSec: to, stepSec: 300 });
  step(16);
  f.respond(f.pending[0], windowPayload({ from, to, hexes: 4 }));
  await flush();
  step(16, 2);
  const layer = map.layers.get("fleet-replay-3d");
  assert.ok(layer, "the fleet layer is on the map");
  assert.equal(last!.data!.hexes.length, 4);
  // jump far, settle → read #2 in flight (settle 250 ms + the 400 ms
  // minimum spacing between reads)
  map.pose.lng = -80; step(16, 40);
  assert.equal(f.pending.length, 2);
  const second = f.pending[1];
  // jump again before #2 lands → #2 aborted, #3 issued
  map.pose.lng = -60; step(500); step(16, 20);
  assert.equal(f.pending.length, 3);
  assert.equal(second.signal?.aborted, true, "superseded read aborted");
  assert.equal(last!.data!.hexes.length, 4, "previous fleet still the drawn data");
  f.respond(f.pending[2], windowPayload({ from, to, hexes: 2 }));
  await flush();
  assert.equal(last!.data!.hexes.length, 2);
  assert.match(f.pending[2].url, /&max=1000/, "reduced tier asks for its own cap");
  c.dispose();
});

test("LOD: batched geometry per class, heads instanced; tier budgets bound the counts", async () => {
  const map = fakeMap({ lng: -99.9, lat: 40.2, zoom: 8, pitch: 0, bearing: 0 });
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  let last: ReplayState | null = null;
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "minimal", onState: (s) => { last = s; }, onPlayhead: () => {} });
  const from = T0, to = T0 + 3600;
  c.setQuery({ kind: "aircraft", fromSec: from, toSec: to, stepSec: 300 });
  step(16);
  f.respond(f.pending[0], windowPayload({ from, to, hexes: 200 }));
  await flush();
  c.setPlayheadTarget(T0 + 1800);
  step(16, 40);
  await flush();
  const lod = last!.lod;
  assert.equal(lod.tracks, 200);
  assert.ok(lod.full <= 12, `minimal tier: ≤ 12 curtains (got ${lod.full})`);
  assert.ok(lod.full + lod.thin <= 12 + 120);
  assert.ok(lod.full > 0 && lod.thin > 0 && lod.headOnly > 0, JSON.stringify(lod));
  const counts = map.layers.get("fleet-replay-3d")!.getCounts();
  assert.ok(counts.full > 0 && counts.thin > 0, "one batched buffer per class");
  assert.ok(counts.heads > 0 && counts.heads <= 400);
  c.dispose();
});

test("close approach focus: playhead to t − 60 s, camera to the midpoint, full-res tracks, highlight", async () => {
  const map = fakeMap();
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  let last: ReplayState | null = null;
  let ph = { value: 0, target: 0 };
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "full", onState: (s) => { last = s; }, onPlayhead: (value, target) => { ph = { value, target }; } });
  const from = T0, to = T0 + 3600;
  const ca: CloseApproachEntry = { a: "h00000", b: "h00001", ca: "CS0", cb: "CS1", t: (T0 + 1500) * 1000, horizNm: 2.9, vertFt: 100, confidence: "high", basis: "x", lat: 40.1, lon: -99.8, altAFt: 29528, altBFt: 29856 };
  c.setQuery({ kind: "aircraft", fromSec: from, toSec: to, stepSec: 300 });
  step(16);
  f.respond(f.pending[0], windowPayload({ from, to, ca: [ca] }));
  await flush();
  step(16, 2);
  c.focusApproach(ca);
  assert.equal(map.calls.easeTo.length, 1);
  const e = map.calls.easeTo[0];
  assert.deepEqual(e.center, [-99.8, 40.1]);
  assert.equal(e.pitch, 55);
  assert.equal(typeof e.easing, "function");
  step(16, 60);
  assert.equal(Math.round(ph.target), T0 + 1440, "t − 60 s");
  assert.ok(Math.abs(ph.value - (T0 + 1440)) < 1, "the drawn playhead lerped there");
  // the pair's full-fidelity tracks are requested from the track endpoint
  const trackReqs = f.pending.filter((p) => p.url.startsWith("/api/data/track/aircraft/"));
  assert.equal(trackReqs.length, 2);
  for (const p of trackReqs) {
    f.respond(p, { points: [
      { t: T0 + 1400, la: 40.1, lo: -99.9, al: 9000 }, { t: T0 + 1430, la: 40.11, lo: -99.88, al: 9000 },
      { t: T0 + 1460, la: 40.12, lo: -99.86, al: 9000 },
    ] });
  }
  await flush();
  assert.equal(last!.focusFullRes, "ok");
  assert.match(last!.focusKey!, /^h00000\|h00001\|/);
  // the camera destination is the query target while the flight is in progress
  const before = f.pending.length;
  map.pose.lng = -99.85; step(16);
  assert.ok(f.pending.length >= before);
  c.focusApproach(null);
  await flush();
  assert.equal(last!.focusKey, null);
  c.dispose();
});

test("focus: a full-res response that does not cover the encounter is rejected (window tracks kept, honest 'failed')", async () => {
  const map = fakeMap();
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  let last: ReplayState | null = null;
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "full", onState: (s) => { last = s; }, onPlayhead: () => {} });
  const from = T0, to = T0 + 3600;
  const ca: CloseApproachEntry = { a: "h00000", b: "h00001", t: (T0 + 1500) * 1000, horizNm: 2.9, vertFt: 100, confidence: "high", basis: "x", lat: 40.1, lon: -99.8 };
  c.setQuery({ kind: "aircraft", fromSec: from, toSec: to, stepSec: 300 });
  step(16);
  f.respond(f.pending[0], windowPayload({ from, to, ca: [ca] }));
  await flush();
  step(16, 2);
  c.focusApproach(ca);
  for (const p of f.pending.filter((x) => x.url.startsWith("/api/data/track/"))) {
    // a track from somewhere else entirely (another time, another place)
    f.respond(p, { points: [{ t: T0 + 80_000, la: 36, lo: -97, al: 500 }, { t: T0 + 80_060, la: 36.1, lo: -97, al: 900 }] });
  }
  await flush();
  assert.equal(last!.focusFullRes, "failed");
  step(16, 30);
  await flush();
  const counts = map.layers.get("fleet-replay-3d")!.getCounts();
  assert.ok(counts.full > 0, "the focused pair still draws from its window tracks");
  c.dispose();
});

test("playback advances the target continuously and stops at the window end", async () => {
  const map = fakeMap();
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  let ph = { value: 0, target: 0, playing: false };
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "full", onState: () => {}, onPlayhead: (value, target, playing) => { ph = { value, target, playing }; } });
  const from = T0, to = T0 + 3600;
  c.setQuery({ kind: "aircraft", fromSec: from, toSec: to, stepSec: 300 });
  step(16);
  f.respond(f.pending[0], windowPayload({ from, to }));
  await flush();
  c.setPlaying(true); // at the end → restarts from the window start
  assert.equal(Math.round(ph.target), from);
  step(16, 1);
  const rate = playRateSecPerSec(3600, 300);
  step(100, 10);
  assert.ok(ph.target > from + rate * 0.9 && ph.target < from + rate * 1.2, `≈1 s of replay (${ph.target - from} vs ${rate})`);
  step(100, 200);
  assert.equal(ph.target, to);
  assert.equal(c.isPlaying(), false);
  c.dispose();
});

test("dispose tears everything down (Law IV): callbacks, layer, registry, in-flight reads", async () => {
  const map = fakeMap();
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  const before = loop.registrationCount;
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "full", onState: () => {}, onPlayhead: () => {} });
  assert.equal(loop.registrationCount, before + 3);
  assert.ok(liveLayers().some((l) => l.id === "fleet-replay-3d"));
  c.setQuery({ kind: "aircraft", fromSec: T0, toSec: T0 + 3600, stepSec: 300 });
  step(16);
  const inflight = f.pending[0];
  c.dispose();
  assert.equal(loop.registrationCount, before);
  assert.equal(map.layers.has("fleet-replay-3d"), false);
  assert.equal(liveLayers().some((l) => l.id === "fleet-replay-3d"), false);
  assert.equal(inflight.signal?.aborted, true);
  c.dispose(); // idempotent
});

test("a failing read keeps the old fleet, reports the error, and backs off (no 400 ms hammering)", async () => {
  const map = fakeMap();
  const { loop, step } = manualLoop();
  const f = fakeFetch();
  let last: ReplayState | null = null;
  const c = new FleetReplayController({ map: map.asMap(), loop, fetchImpl: f.impl, tier: "full", onState: (s) => { last = s; }, onPlayhead: () => {} });
  const from = T0, to = T0 + 3600;
  c.setQuery({ kind: "aircraft", fromSec: from, toSec: to, stepSec: 300 });
  step(16);
  f.respond(f.pending[0], windowPayload({ from, to, hexes: 5 }));
  await flush();
  map.pose.lng = -80; step(16, 40);
  assert.equal(f.pending.length, 2);
  f.respond(f.pending[1], { error: "window read failed" }, false);
  await flush();
  assert.equal(last!.error, "window read failed");
  assert.equal(last!.data!.hexes.length, 5, "previous fleet still drawn");
  step(16, 40); // 640 ms: inside the 1 s backoff
  assert.equal(f.pending.length, 2, "no retry inside the backoff");
  step(16, 40);
  assert.equal(f.pending.length, 3, "retried after the backoff");
  c.dispose();
});

test("framingOffset: the pair is framed in the map area the panel does not cover", () => {
  const container = { getBoundingClientRect: () => ({ left: 0, top: 48, right: 390, bottom: 780 }) };
  // phone bottom sheet (spans the width, covers the lower half) → shift up
  const [ox, oy] = framingOffset(container, { left: 8, top: 345, right: 382, bottom: 770 });
  assert.equal(ox, 0);
  const cy = (48 + 780) / 2;
  assert.equal(oy, Math.round((48 + (345 - 48) / 2) - cy));
  assert.ok(oy < 0);
  // desktop side panel on the left → shift right into the free band
  const desk = { getBoundingClientRect: () => ({ left: 0, top: 56, right: 1440, bottom: 900 }) };
  const [dx, dy] = framingOffset(desk, { left: 64, top: 218, right: 364, bottom: 700 });
  assert.equal(dy, 0);
  assert.equal(dx, Math.round((364 + (1440 - 364) / 2) - 720));
  // no overlap / no rect → no offset
  assert.deepEqual(framingOffset(desk, { left: 2000, top: 0, right: 2100, bottom: 10 }), [0, 0]);
  assert.deepEqual(framingOffset(null, null), [0, 0]);
});

test("helpers: play rate matches the old tick cadence; spring easing is monotone 0→1", () => {
  assert.equal(playRateSecPerSec(24 * 3600, 300), (24 * 3600 / 60) / 0.9);
  assert.equal(playRateSecPerSec(3600, 300), 300 / 0.9);
  assert.equal(springEase(0), 0);
  assert.equal(springEase(1), 1);
  let prev = 0;
  for (let x = 0.05; x <= 1; x += 0.05) { const v = springEase(x); assert.ok(v >= prev); prev = v; }
});
