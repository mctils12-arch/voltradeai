// dataQuality battery — the DATA QUALITY GATE (research/location_context_engine.md).
// Proves inaccurate inputs are caught + quarantined, never passed as clean.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  validateRecord, coordValid, partitionValid, freshnessStatus, isOutlier, layerHealth,
} from "./dataQuality";

test("validateRecord: required, numeric bounds, enum, pattern", () => {
  const schema = {
    name: { required: true },
    mw: { numeric: true, min: 0, max: 100000 },
    status: { oneOf: ["OP", "SB", "RE"] },
    period: { pattern: /^\d{4}-\d{2}-\d{2}$/ },
  };
  assert.deepEqual(validateRecord({ name: "A", mw: 500, status: "OP", period: "2026-07-11" }, schema), []);
  // missing required
  assert.ok(validateRecord({ mw: 5 }, schema).some((i) => i.field === "name" && i.rule === "required"));
  // negative capacity — physically impossible, must be caught
  assert.ok(validateRecord({ name: "A", mw: -3 }, schema).some((i) => i.field === "mw" && i.rule === "min"));
  // over the upper bound
  assert.ok(validateRecord({ name: "A", mw: 999999 }, schema).some((i) => i.rule === "max"));
  // non-numeric where numeric expected
  assert.ok(validateRecord({ name: "A", mw: "n/a" }, schema).some((i) => i.rule === "numeric"));
  // bad enum + bad pattern
  assert.ok(validateRecord({ name: "A", status: "ZZ" }, schema).some((i) => i.rule === "oneOf"));
  assert.ok(validateRecord({ name: "A", period: "07/11/26" }, schema).some((i) => i.rule === "pattern"));
});

test("validateRecord: numeric strings pass min/max (EIA/registry values arrive as strings)", () => {
  const schema = { v: { numeric: true, min: 0, max: 10 } };
  assert.deepEqual(validateRecord({ v: "7" }, schema), []);
  assert.ok(validateRecord({ v: "-1" }, schema).length > 0);
});

test("coordValid: range + null-island guard", () => {
  assert.equal(coordValid(40.7, -74.0), true);
  assert.equal(coordValid(91, 0), false);      // lat out of range
  assert.equal(coordValid(0, 200), false);     // lon out of range
  assert.equal(coordValid(0, 0), false);       // null island = failed geocode
  assert.equal(coordValid(null, 5), false);
  assert.equal(coordValid("42.5", "-114.4"), true); // string coords ok
});

test("partitionValid: clean vs quarantined-suspect, nothing silently dropped", () => {
  const recs = [
    { name: "good", mw: 10 },
    { name: "", mw: 10 },        // missing name
    { name: "bad", mw: -5 },     // impossible value
  ];
  const schema = { name: { required: true }, mw: { numeric: true, min: 0 } };
  const p = partitionValid(recs, (r) => validateRecord(r, schema));
  assert.equal(p.clean.length, 1);
  assert.equal(p.suspect.length, 2);
  // every input is accounted for — clean + suspect === input (no silent loss)
  assert.equal(p.clean.length + p.suspect.length, recs.length);
});

test("freshnessStatus: within budget fresh, past budget stale, missing unknown", () => {
  const now = 1_000_000_000_000;
  const day = 86400_000;
  assert.equal(freshnessStatus(now - day, 2 * day, now), "fresh");
  assert.equal(freshnessStatus(now - 3 * day, 2 * day, now), "stale");
  assert.equal(freshnessStatus(null, day, now), "unknown");
});

test("isOutlier: catches an impossible step-change, withholds on thin history", () => {
  const recent = [100, 102, 98, 101, 99, 100];
  assert.equal(isOutlier(101, recent), false);      // normal
  assert.equal(isOutlier(5000, recent), true);      // corrupt spike
  assert.equal(isOutlier(9999, [100, 101]), false); // too few peers -> no guess
  // a flatline that suddenly moves is suspect
  assert.equal(isOutlier(7, [5, 5, 5, 5, 5, 5]), true);
  assert.equal(isOutlier(5, [5, 5, 5, 5, 5, 5]), false);
});

test("layerHealth: reports clean/suspect/freshness for the UI trust badge", () => {
  const part = { clean: [1, 2, 3], suspect: [{ record: 4, issues: [] }] } as any;
  const h = layerHealth("EPA SEMS", part, "fresh", "2026-07-01");
  assert.equal(h.clean, 3);
  assert.equal(h.suspect, 1);
  assert.equal(h.freshness, "fresh");
  assert.equal(h.source, "EPA SEMS");
});
