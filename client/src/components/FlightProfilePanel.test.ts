// Hermetic tests for the altitude/time panel's pure rules (no DOM/React).
// Run: npx tsx --test client/src/components/FlightProfilePanel.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { replaySpeed, profileEdgeSec } from "./FlightProfilePanel.tsx";

test("profileEdgeSec: a live track's axis grows to now; a closed trip ends at its last fix", () => {
  const t1 = 1_000_000;
  const now = t1 + 10 * 3600; // the report's shape: trip flown 10h ago
  // live: the axis follows the wall clock so new fixes have somewhere to land
  assert.equal(profileEdgeSec(false, t1, now), now);
  assert.equal(profileEdgeSec(undefined, t1, now), now);
  // historical: the axis is the trip's own end — the whole 90 minutes of a
  // 10h-old flight must fill the chart, not 13% of it
  assert.equal(profileEdgeSec(true, t1, now), t1);
  // a live track whose newest fix is somehow ahead of the clock never shrinks
  assert.equal(profileEdgeSec(false, now + 5, now), now + 5);
});

test("replaySpeed: a full pass takes ~80s regardless of track length, never below 1x", () => {
  assert.equal(replaySpeed(8000), 100);
  assert.equal(replaySpeed(30), 1, "very short tracks still play at real time or faster");
});
