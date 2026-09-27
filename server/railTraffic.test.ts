// railTraffic.ts — pure-function view over the static STB EP724 archive.
// Mirrors unComtrade.test.ts's synthetic-fixture style (no network, no
// live archive dependency for the logic tests) plus a real-file
// coherence check.
import { test } from "node:test";
import assert from "node:assert/strict";
import { railTrafficView, RAIL_TRAFFIC_GATE_NOTE } from "./railTraffic";
import railArchive from "../datacore/rail/ep724_carloads.json";

function fixture(overrides: Record<string, (number | null)[]> = {}) {
  return {
    built: "2026-09-27T00:00:00Z",
    source: "STB EP724 consolidated rail service data (public domain)",
    attribution: "Surface Transportation Board (EP 724 rail service data)",
    selection: "cat 11 weekly carloads (22 commodities, all railroads) + cat 1 System train speed + cat 3 cars-on-line",
    n_weeks: 3,
    n_series: 5,
    weeks: ["2026-09-06", "2026-09-13", "2026-09-20"],
    series: {
      "BNSF|Weekly Carloads By 22 Commodity Categories|Grain": [1000, 1100, 1200],
      "UP|Weekly Carloads By 22 Commodity Categories|Grain": [500, null, 600], // UP missed the middle week
      "BNSF|Weekly Carloads By 22 Commodity Categories|Containers": [2000, 2100, 2200],
      "UP|Weekly Carloads By 22 Commodity Categories|Trailers": [50, 60, 70],
      "BNSF|Average Train Speed  (MPH)|System": [22.1, 22.3, 22.0], // different measure, never summed in
      ...overrides,
    },
  } as any;
}

test("sums carloads across railroads for the latest archived week only", () => {
  const v = railTrafficView(fixture());
  assert.equal(v.latestWeek, "2026-09-20");
  assert.equal(v.priorWeek, "2026-09-13");
  const grain = v.commodities.find((c) => c.variable === "Grain")!;
  assert.equal(grain.latestWeekCarloads, 1200 + 600);
});

test("a railroad missing the PRIOR week (not the latest) still contributes to latest-week totals", () => {
  const v = railTrafficView(fixture());
  const grain = v.commodities.find((c) => c.variable === "Grain")!;
  // prior week (2026-09-13): UP reported null, only BNSF's 1100 counts
  assert.equal(grain.priorWeekCarloads, 1100);
  assert.equal(grain.weekOverWeekDeltaPct, Math.round(((1800 - 1100) / 1100) * 1000) / 10);
});

test("only the CARLOAD_MEASURE series is summed -- train speed is never mixed in", () => {
  const v = railTrafficView(fixture());
  const variables = v.commodities.map((c) => c.variable);
  assert.ok(!variables.includes("System"), "Average Train Speed's variable name must not leak into commodity rows");
});

test("Containers and Trailers are flagged intermodal, everything else is not", () => {
  const v = railTrafficView(fixture());
  const containers = v.commodities.find((c) => c.variable === "Containers")!;
  const trailers = v.commodities.find((c) => c.variable === "Trailers")!;
  const grain = v.commodities.find((c) => c.variable === "Grain")!;
  assert.equal(containers.isIntermodal, true);
  assert.equal(trailers.isIntermodal, true);
  assert.equal(grain.isIntermodal, false);
});

test("system totals split cleanly into intermodal vs non-intermodal, summing back to the whole", () => {
  const v = railTrafficView(fixture());
  assert.equal(
    v.systemIntermodalLatestWeekCarloads + v.systemNonIntermodalLatestWeekCarloads,
    v.systemLatestWeekCarloads,
  );
  // latest week (2026-09-20): Grain 1800, Containers 2200, Trailers 70
  assert.equal(v.systemIntermodalLatestWeekCarloads, 2200 + 70);
  assert.equal(v.systemNonIntermodalLatestWeekCarloads, 1800);
});

test("per-railroad rows sum only that railroad's own reported carloads for the latest week", () => {
  const v = railTrafficView(fixture());
  const bnsf = v.railroads.find((r) => r.railroad === "BNSF")!;
  const up = v.railroads.find((r) => r.railroad === "UP")!;
  assert.equal(bnsf.latestWeekCarloads, 1200 + 2200);
  assert.equal(up.latestWeekCarloads, 600 + 70);
});

test("a commodity absent from the latest week is skipped entirely, never zero-filled", () => {
  const v = railTrafficView(fixture({
    "BNSF|Weekly Carloads By 22 Commodity Categories|Coal": [300, 310, null],
  }));
  assert.ok(!v.commodities.some((c) => c.variable === "Coal"), "a null-latest-week commodity must not appear as a zero row");
});

test("view is marked raw/non-predictive with the gate-1 note carried verbatim", () => {
  const v = railTrafficView(fixture());
  assert.equal(v.kind, "raw");
  assert.equal(v.predictive, false);
  assert.equal(v.note, RAIL_TRAFFIC_GATE_NOTE);
  assert.match(v.note, /GATE 1/);
});

test("commodity rows sort by latest-week carloads descending", () => {
  const v = railTrafficView(fixture());
  for (let i = 1; i < v.commodities.length; i++) {
    assert.ok(v.commodities[i - 1].latestWeekCarloads >= v.commodities[i].latestWeekCarloads);
  }
});

test("committed archive produces a coherent live view", () => {
  const v = railTrafficView(railArchive as any);
  assert.ok(v.commodities.length >= 15, "expects most of the 22 commodity categories to have reported the latest week");
  assert.ok(v.railroads.length >= 5, "expects most of the 7-8 Class I railroad codes to have reported the latest week");
  assert.ok(v.systemLatestWeekCarloads > 0);
  assert.ok(v.systemIntermodalLatestWeekCarloads > 0);
  assert.ok(v.systemNonIntermodalLatestWeekCarloads > 0);
  for (const c of v.commodities) {
    assert.ok(c.latestWeekCarloads > 0, `${c.variable} should be a positive real carload count`);
  }
});
