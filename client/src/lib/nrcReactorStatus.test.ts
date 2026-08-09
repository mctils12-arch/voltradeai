import { test } from "node:test";
import assert from "node:assert/strict";
import { statusRank, sortPlantsByStatus, statusCounts } from "./nrcReactorStatus";

test("statusRank orders worst-first: outage < reduced < unknown < full", () => {
  assert.ok(statusRank("outage") < statusRank("reduced"));
  assert.ok(statusRank("reduced") < statusRank("unknown"));
  assert.ok(statusRank("unknown") < statusRank("full"));
});

test("statusRank treats an unrecognized/missing status as worse than none of the known tiers, but stable", () => {
  assert.equal(statusRank(undefined), statusRank(null));
  assert.ok(statusRank("bogus") > statusRank("full"));
});

test("sortPlantsByStatus puts outage plants before reduced/unknown/full regardless of input order", () => {
  const plants = [
    { name: "Full Plant", status: "full", avgPower: 100 },
    { name: "Outage Plant", status: "outage", avgPower: 0 },
    { name: "Unknown Plant", status: "unknown", avgPower: null },
    { name: "Reduced Plant", status: "reduced", avgPower: 60 },
  ];
  const sorted = sortPlantsByStatus(plants);
  assert.deepEqual(sorted.map((p) => p.status), ["outage", "reduced", "unknown", "full"]);
});

test("sortPlantsByStatus breaks ties within a status tier by ascending avgPower, then name", () => {
  const plants = [
    { name: "B Reduced", status: "reduced", avgPower: 80 },
    { name: "A Reduced", status: "reduced", avgPower: 40 },
    { name: "C Reduced", status: "reduced", avgPower: 40 },
  ];
  const sorted = sortPlantsByStatus(plants);
  assert.deepEqual(sorted.map((p) => p.name), ["A Reduced", "C Reduced", "B Reduced"]);
});

test("sortPlantsByStatus never mutates the input array", () => {
  const plants = [
    { name: "Full Plant", status: "full", avgPower: 100 },
    { name: "Outage Plant", status: "outage", avgPower: 0 },
  ];
  const original = plants.slice();
  sortPlantsByStatus(plants);
  assert.deepEqual(plants, original);
});

test("sortPlantsByStatus places null avgPower (no unit reported) last within its status tier", () => {
  const plants = [
    { name: "No Reading A", status: "unknown", avgPower: null },
    { name: "Zero Power", status: "unknown", avgPower: 0 },
  ];
  const sorted = sortPlantsByStatus(plants);
  assert.deepEqual(sorted.map((p) => p.name), ["Zero Power", "No Reading A"]);
});

test("statusCounts tallies each status bucket and folds any unrecognized status into unknown", () => {
  const plants = [
    { status: "full" }, { status: "full" },
    { status: "reduced" },
    { status: "outage" },
    { status: "something-new" },
  ];
  assert.deepEqual(statusCounts(plants), { outage: 1, reduced: 1, unknown: 1, full: 2 });
});

test("statusCounts on an empty list returns all-zero buckets", () => {
  assert.deepEqual(statusCounts([]), { outage: 0, reduced: 0, unknown: 0, full: 0 });
});
