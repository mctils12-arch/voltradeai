// SCALE S1(b) vessels delta — pins the snapshot builder (bbox/freshness/
// cap/disclosure semantics extracted from the route), the outward-only
// bbox expansion, and the since→unchanged protocol.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  VESSEL_FRESH_MS,
  VESSEL_SNAPSHOT_TTL_MS,
  VESSEL_CAP,
  expandBbox1dp,
  buildVesselSnapshot,
  sinceUnchanged,
  shouldRebuildSnapshot,
  type LiveVesselPos,
  type LiveVesselStatic,
} from "./liveDelta";

// Client vessels poll cadence (datamap.tsx wireLivePoints id:"vessels"
// intervalMs) — kept as a local literal, same cross-package convention
// AIRCRAFT_TTL_MS already uses vs. its 15s client poll in server/routes.ts.
const VESSEL_CLIENT_POLL_MS = 20_000;

const NOW = 1_750_000_000_000;

function pos(over: Partial<LiveVesselPos> = {}): LiveVesselPos {
  return { lat: 30, lon: -80, sog: 10, cog: 90, name: "raw-name", at: NOW - 1000, ...over };
}

test("buildVesselSnapshot: bbox + freshness filter, statics join, time stamped", () => {
  const positions = new Map<string, LiveVesselPos>([
    ["a", pos()],
    ["b", pos({ lat: 60 })],                          // outside bbox
    ["c", pos({ at: NOW - VESSEL_FRESH_MS - 1 })],    // stale
    ["d", pos({ lon: -79.5, name: "fallback" })],
  ]);
  const statics = new Map<string, LiveVesselStatic>([
    ["a", { shiptype: 70, destination: "MIAMI", name: "EVER GIVEN" }],
  ]);
  const snap = buildVesselSnapshot(positions, statics, { lamin: 25, lamax: 35, lomin: -85, lomax: -75 }, NOW);
  assert.equal(snap.count, 2);
  assert.equal(snap.total_in_view, 2);
  assert.equal(snap.time, String(NOW), "time is the snapshot instant — the delta token");
  const a = snap.vessels.find((v) => v.mmsi === "a")!;
  assert.equal(a.name, "EVER GIVEN", "statics name outranks the raw report");
  assert.equal(a.shiptype, 70);
  assert.equal(a.destination, "MIAMI");
  const d = snap.vessels.find((v) => v.mmsi === "d")!;
  assert.equal(d.name, "fallback");
  assert.equal(d.shiptype, null, "no statics row = honest nulls, never guessed");
  assert.equal(snap.coverage, undefined, "under the cap: no partial-coverage claim");
});

test("buildVesselSnapshot: cap bites → count < total_in_view + honest disclosure", () => {
  const positions = new Map<string, LiveVesselPos>();
  for (let i = 0; i < 7; i++) positions.set(`m${i}`, pos({ lon: -80 + i * 0.01 }));
  const snap = buildVesselSnapshot(positions, new Map(), { lamin: 25, lamax: 35, lomin: -85, lomax: -75 }, NOW, 5);
  assert.equal(snap.count, 5);
  assert.equal(snap.total_in_view, 7);
  assert.equal(snap.coverage, "partial");
  assert.ok(/showing 5 of 7/.test(snap.coverage_note!), "disclosure names both numbers");
});

test("expandBbox1dp: outward only (never crops the viewport), clamped to world", () => {
  const b = expandBbox1dp(24.96, 25.01, -80.04, -79.99);
  assert.equal(b.lamin, 24.9);
  assert.equal(b.lamax, 25.1);
  assert.equal(b.lomin, -80.1);
  assert.equal(b.lomax, -79.9);
  // outward property on arbitrary values
  assert.ok(b.lamin <= 24.96 && b.lamax >= 25.01 && b.lomin <= -80.04 && b.lomax >= -79.99);
  const w = expandBbox1dp(-90, 90, -200, 200);
  assert.deepEqual(w, { lamin: -85, lamax: 85, lomin: -180, lomax: 180 });
});

test("sinceUnchanged: exact time match only; missing time or empty since fail OPEN to full payload", () => {
  assert.equal(sinceUnchanged({ time: "123" }, "123"), true);
  assert.equal(sinceUnchanged({ time: "123" }, "122"), false);
  assert.equal(sinceUnchanged({ time: "123" }, undefined), false);
  assert.equal(sinceUnchanged({ time: "123" }, ""), false);
  assert.equal(sinceUnchanged({}, "123"), false, "no time = never unchanged");
  assert.equal(sinceUnchanged({ time: 123 }, "123"), true, "number/string time compare via String()");
});

test("constants stay in the sane band the routes were built around", () => {
  assert.equal(VESSEL_FRESH_MS, 20 * 60_000, "freshness rule unchanged from the pre-delta handler");
  assert.equal(VESSEL_CAP, 15000, "cap unchanged from the pre-delta handler");
  // S-A2 (scale_program.md): TTL must EXCEED the client poll interval, not
  // undercut it — a shorter TTL means every poll misses the cache and
  // rebuilds a fresh snapshot with a new `time`, so `unchanged` can never
  // match. This is the inverse of the assertion this test used to pin
  // (VESSEL_SNAPSHOT_TTL_MS < 20_000), which was pinning the dead-code bug.
  assert.ok(
    VESSEL_SNAPSHOT_TTL_MS > VESSEL_CLIENT_POLL_MS,
    "snapshot TTL must outlive one client poll interval, or `unchanged` never fires",
  );
});

test("shouldRebuildSnapshot: a snapshot survives one real client poll interval (S-A2 regression — was dead code at the old 15s TTL)", () => {
  const at = 1_000_000;
  const hit = { at };
  // First poll: nothing cached yet.
  assert.equal(shouldRebuildSnapshot(undefined, at, VESSEL_SNAPSHOT_TTL_MS), true);
  // Second poll lands one real client-poll-interval later — must reuse the
  // cached entry (same `time`) so sinceUnchanged can match downstream.
  assert.equal(
    shouldRebuildSnapshot(hit, at + VESSEL_CLIENT_POLL_MS, VESSEL_SNAPSHOT_TTL_MS),
    false,
    "at the real 20s poll cadence, the snapshot must still be fresh",
  );
  // Once the TTL genuinely elapses, it does rebuild.
  assert.equal(shouldRebuildSnapshot(hit, at + VESSEL_SNAPSHOT_TTL_MS, VESSEL_SNAPSHOT_TTL_MS), true);
});

test("shouldRebuildSnapshot + sinceUnchanged compose into a real unchanged response at poll cadence", () => {
  const bbox = { lamin: 25, lamax: 35, lomin: -85, lomax: -75 };
  const t0 = 1_000_000;
  const snap0 = buildVesselSnapshot(new Map([["a", pos({ at: t0 - 1000 })]]), new Map(), bbox, t0);
  const hit = { at: t0, data: snap0 };
  const t1 = t0 + VESSEL_CLIENT_POLL_MS; // next poll, real cadence
  assert.equal(shouldRebuildSnapshot(hit, t1, VESSEL_SNAPSHOT_TTL_MS), false, "cache still fresh at t1");
  assert.equal(sinceUnchanged(hit.data, snap0.time), true, "client's since= (from the first response) now matches");
});
