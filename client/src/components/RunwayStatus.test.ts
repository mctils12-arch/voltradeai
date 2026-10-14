// Runway status row (2026-10-14). Run: npx tsx --test client/src/components/RunwayStatus.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { runwayStatusUrl, runwayLine, RUNWAY_POLL_MS, type RunwayStatusData } from "./RunwayStatus";

test("runwayStatusUrl: live fix -> query; unknown fields omitted, never sent as NaN/null", () => {
  const u = runwayStatusUrl("ab8c8e", { lat: 35.2144, lon: -80.9353, altFt: 650.4, gsKt: 160.2, trk: 354.04, onGround: false, vsFpm: null });
  const q = new URL(u, "http://x").searchParams;
  assert.equal(new URL(u, "http://x").pathname, "/api/data/procedures/runway");
  assert.equal(q.get("hex"), "ab8c8e");
  assert.equal(q.get("altFt"), "650");
  assert.equal(q.get("gsKt"), "160");
  assert.equal(q.get("trk"), "354.0");
  assert.equal(q.get("gnd"), "0");
  assert.equal(q.has("vs"), false);
  const g = new URL(runwayStatusUrl("ab8c8e", { lat: 1, lon: 2, altFt: null, gsKt: null, trk: null, onGround: true, vsFpm: NaN }), "http://x").searchParams;
  assert.equal(g.get("gnd"), "1");
  for (const k of ["altFt", "gsKt", "trk", "vs"]) assert.equal(g.has(k), false, k);
});

test("runwayLine: current state (with airport) first, else the trip's departure runway, else nothing", () => {
  const base: RunwayStatusData = { airport: null, current: null, departure: null, honesty: "x" };
  assert.equal(runwayLine(null), null);
  assert.equal(runwayLine(base), null);
  assert.equal(runwayLine({ ...base, departure: { airport: "KCLT", runway: "36", label: "Departed KCLT runway 36" } }), "Departed KCLT runway 36");
  assert.equal(runwayLine({
    ...base, airport: { icao: "KCLT", name: "Charlotte" },
    current: { state: "LINED_UP", runway: "36", label: "Lined up on runway 36" },
    departure: { airport: "KCLT", runway: "36", label: "Departed KCLT runway 36" },
  }), "Lined up on runway 36 · KCLT");
  assert.ok(RUNWAY_POLL_MS <= 10_000, "runway states change in seconds");
});
