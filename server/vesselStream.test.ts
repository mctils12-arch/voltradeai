import { test } from "node:test";
import assert from "node:assert/strict";
import { vesselStreamEnabled, bootVesselStream } from "./vesselStream";

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
