import { test } from "node:test";
import assert from "node:assert/strict";
import { vesselStreamEnabled, bootVesselStream, scheduleVesselHeartbeat } from "./vesselStream";

// KNOWN BROKEN #9: the aisstream websocket connected lazily, only on the
// first /api/data/vessels request, so every deploy left the vessels layer
// (and its archive recording) cold until a visitor opened the map.

test("vesselStreamEnabled is true when AISSTREAM_KEY is set", () => {
  assert.equal(vesselStreamEnabled({ AISSTREAM_KEY: "abc123" } as NodeJS.ProcessEnv), true);
});

test("vesselStreamEnabled is false without a key", () => {
  assert.equal(vesselStreamEnabled({} as NodeJS.ProcessEnv), false);
  assert.equal(vesselStreamEnabled({ AISSTREAM_KEY: "" } as NodeJS.ProcessEnv), false);
});

test("bootVesselStream connects eagerly when a key is configured", () => {
  let called = false;
  bootVesselStream({ AISSTREAM_KEY: "abc123" } as NodeJS.ProcessEnv, () => { called = true; });
  assert.equal(called, true);
});

test("bootVesselStream does not attempt a connection without a key", () => {
  let called = false;
  bootVesselStream({} as NodeJS.ProcessEnv, () => { called = true; });
  assert.equal(called, false);
});

// Gap found 2026-07-20 (live): boot-only reconnect leaves a mid-session
// websocket drop dead forever unless a client happens to hit the route.
test("scheduleVesselHeartbeat schedules a periodic reconnect attempt when a key is configured", () => {
  let scheduledFn: (() => void) | null = null;
  let scheduledMs: number | null = null;
  let unrefCalled = false;
  const fakeSetInterval = (fn: () => void, ms: number) => {
    scheduledFn = fn;
    scheduledMs = ms;
    return { unref: () => { unrefCalled = true; } };
  };
  scheduleVesselHeartbeat({ AISSTREAM_KEY: "abc123" } as NodeJS.ProcessEnv, () => {}, 60_000, fakeSetInterval);
  assert.equal(scheduledMs, 60_000);
  assert.ok(scheduledFn, "expected a timer callback to be scheduled");
  assert.equal(unrefCalled, true);
});

test("scheduleVesselHeartbeat calls the injected connect function on each tick", () => {
  let ticks = 0;
  const fakeSetInterval = (fn: () => void) => { fn(); fn(); return {}; };
  scheduleVesselHeartbeat({ AISSTREAM_KEY: "abc123" } as NodeJS.ProcessEnv, () => { ticks++; }, 60_000, fakeSetInterval);
  assert.equal(ticks, 2);
});

test("scheduleVesselHeartbeat does not schedule anything without a key", () => {
  let scheduled = false;
  const fakeSetInterval = () => { scheduled = true; return {}; };
  scheduleVesselHeartbeat({} as NodeJS.ProcessEnv, () => {}, 60_000, fakeSetInterval);
  assert.equal(scheduled, false);
});
