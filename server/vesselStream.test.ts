import { test } from "node:test";
import assert from "node:assert/strict";
import { vesselStreamEnabled, bootVesselStream, vesselFeedHealth, vesselLayerStatus, vesselNeverFramed, VESSEL_SILENT_THRESHOLD_MS } from "./vesselStream";

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

// ── feed-health verdict (repair 2026-08-06) ─────────────────────────────────
// Field defect pair: (1) no reconnect path without visitor traffic ->
// permanent archive gaps; (2) key-presence claimed "live" through outages.
// The one verdict below drives the watchdog, the registry override, and the
// route's honesty fields.
test("vesselFeedHealth: disconnected is down; connected+flowing is live", () => {
  const now = 1_000_000_000;
  const dead = vesselFeedHealth(3 /* CLOSED */, now - 10_000, now);
  assert.equal(dead.down, true);
  assert.equal(dead.zombie, false, "a closed socket is not a zombie — just down");
  const live = vesselFeedHealth(1, now - 5_000, now);
  assert.deepEqual({ down: live.down, zombie: live.zombie, connected: live.connected },
                   { down: false, zombie: false, connected: true });
  assert.equal(live.silentMs, 5_000);
});

test("vesselFeedHealth: half-open zombie — connected but silent past threshold — is down and flagged for redial", () => {
  const now = 1_000_000_000;
  const z = vesselFeedHealth(1, now - VESSEL_SILENT_THRESHOLD_MS - 1, now);
  assert.equal(z.zombie, true, "readyState 1 with minutes of silence on a global firehose IS an outage");
  assert.equal(z.down, true);
  const fine = vesselFeedHealth(1, now - VESSEL_SILENT_THRESHOLD_MS + 1_000, now);
  assert.equal(fine.zombie, false, "inside the threshold stays live");
});

test("vesselFeedHealth: never-received-a-frame -> silentMs null; only disconnection is provable", () => {
  const now = 1_000_000_000;
  const fresh = vesselFeedHealth(1, 0, now);
  assert.equal(fresh.silentMs, null);
  assert.equal(fresh.zombie, false, "no frame ever + no timestamp: zombie unprovable");
  assert.equal(fresh.down, false);
  const never = vesselFeedHealth(null, 0, now);
  assert.equal(never.down, true, "no socket at all is down");
});

// ── repair 2026-08-11: never-a-frame blind spot + layer-status honesty ──
test("vesselFeedHealth: connected socket that NEVER delivered a frame goes zombie once the connect-time silence passes the threshold", () => {
  const t0 = 1_000_000;
  // 08-06 behavior (no connectedAt): permanently healthy — the prod outage
  const legacy = vesselFeedHealth(1, 0, t0 + 10 * 60_000);
  assert.equal(legacy.down, false, "without connect-time the blind spot is unfixable — documents the old hole");
  // with connect-time: silent since connect > threshold => zombie + down
  const vh = vesselFeedHealth(1, 0, t0 + 10 * 60_000, undefined, t0);
  assert.equal(vh.zombie, true);
  assert.equal(vh.down, true);
  // inside the threshold it is still warming up, not down
  const young = vesselFeedHealth(1, 0, t0 + 60_000, undefined, t0);
  assert.equal(young.down, false);
  // a real frame timestamp always wins over connect-time
  const framed = vesselFeedHealth(1, t0 + 9 * 60_000, t0 + 10 * 60_000, undefined, t0);
  assert.equal(framed.down, false);
});

test("vesselLayerStatus: healthy keyed feed reports LIVE — never the static registry value", () => {
  const healthy = { connected: true, silentMs: 5_000, zombie: false, down: false };
  assert.deepEqual(vesselLayerStatus(true, healthy), { status: "live" });
  assert.equal(vesselLayerStatus(false, null).status, "awaiting_key");
  const dead = { connected: false, silentMs: null, zombie: false, down: true };
  assert.equal(vesselLayerStatus(true, dead).status, "down");
});

// ── round 2 (2026-08-11): the redial cycle was masking a never-alive feed ──
test("vesselNeverFramed: a feed that has never delivered a frame stays flagged ACROSS redials", () => {
  const t0 = 5_000_000;
  const past = t0 + VESSEL_SILENT_THRESHOLD_MS + 1_000;
  assert.equal(vesselNeverFramed(0, t0, past), true, "no frames ever, past the threshold since FIRST connect");
  // the watchdog redialing 10s ago must NOT reset this verdict — that reset
  // is exactly what made the panel read "live" 179s out of every 180s
  assert.equal(vesselNeverFramed(0, t0, past + 10 * 60_000), true);
  // one real frame ever clears it permanently (health takes over from there)
  assert.equal(vesselNeverFramed(1, t0, past), false);
  // inside the threshold it is warming up, not a verdict
  assert.equal(vesselNeverFramed(0, t0, t0 + 1_000), false);
  // never connected at all: no verdict to give
  assert.equal(vesselNeverFramed(0, 0, past), false);
});

test("vesselLayerStatus: never-framed outranks a freshly-redialed healthy socket", () => {
  const freshlyConnected = { connected: true, silentMs: 5_000, zombie: false, down: false };
  assert.equal(vesselLayerStatus(true, freshlyConnected, true).status, "down",
    "zero frames since startup can never be reported as live");
  assert.match(vesselLayerStatus(true, freshlyConnected, true).status_note || "", /ZERO frames/);
  assert.equal(vesselLayerStatus(true, freshlyConnected, false).status, "live");
});
