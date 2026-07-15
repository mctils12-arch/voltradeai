// pfas.ts — EPA UCMR 5 PFAS drinking-water detections (Location Context
// Engine hazard layer). systemFromRecord/parsePfasArtifact are pure
// functions tested against synthetic fixtures; latestPfas() (the module's
// one side-effecting export, static-imports the committed artifact) is
// checked for coherence against the real committed file.
import { test } from "node:test";
import assert from "node:assert/strict";
import { systemFromRecord, parsePfasArtifact, latestPfas } from "./pfas";

const validRecord = () => ({
  pwsid: "OK0000001",
  name: "Test Water System",
  state: "OK",
  lat: 35.94,
  lon: -96.75,
  population_served: 5000,
  n_analytes_detected: 1,
  detections: [
    { contaminant: "PFOA", units: "µg/L", mrl: 0.004, max_value: 0.01, first_detected: "1/1/2024", last_detected: "6/1/2024", n_detections: 2 },
  ],
});

test("systemFromRecord accepts a well-formed record", () => {
  const { system, issues } = systemFromRecord(validRecord());
  assert.equal(issues.length, 0);
  assert.equal(system?.pwsid, "OK0000001");
  assert.equal(system?.detections.length, 1);
});

test("systemFromRecord quarantines missing required fields", () => {
  const r = validRecord() as any;
  delete r.pwsid;
  const { system, issues } = systemFromRecord(r);
  assert.equal(system, undefined);
  assert.ok(issues.some((i) => i.field === "pwsid"));
});

test("systemFromRecord quarantines invalid coordinates (out of range and null island)", () => {
  const outOfRange = systemFromRecord({ ...validRecord(), lat: 999 });
  assert.equal(outOfRange.system, undefined);
  assert.ok(outOfRange.issues.some((i) => i.field === "coordinates"));

  const nullIsland = systemFromRecord({ ...validRecord(), lat: 0, lon: 0 });
  assert.equal(nullIsland.system, undefined);
  assert.ok(nullIsland.issues.some((i) => i.field === "coordinates"));
});

test("systemFromRecord quarantines a record with no detections (should never occur, but never trusted blindly)", () => {
  const { system, issues } = systemFromRecord({ ...validRecord(), detections: [] });
  assert.equal(system, undefined);
  assert.ok(issues.some((i) => i.field === "detections"));
});

test("parsePfasArtifact partitions clean vs suspect and never throws on a mixed batch", () => {
  const good = validRecord();
  const bad = { ...validRecord(), pwsid: undefined };
  const { systems, suspect } = parsePfasArtifact({ systems: [good, bad] });
  assert.equal(systems.length, 1);
  assert.equal(suspect, 1);
});

test("parsePfasArtifact handles an empty/missing systems array without crashing", () => {
  assert.deepEqual(parsePfasArtifact({ systems: [] }), { systems: [], suspect: 0 });
  assert.deepEqual(parsePfasArtifact({} as any), { systems: [], suspect: 0 });
});

test("the committed artifact is coherent: real data, positive count, every system has >=1 detection", () => {
  const { systems, health, meta } = latestPfas();
  assert.ok(systems.length > 1000, `expected thousands of systems, got ${systems.length}`);
  assert.equal(health.suspect, 0, "committed artifact should already be clean (built by the script's own ingest gate)");
  assert.equal(meta.predictive, false);
  assert.ok(meta.caveat.includes("NOT an MCL-exceedance"));
  for (const s of systems.slice(0, 50)) {
    assert.ok(s.detections.length > 0);
    assert.ok(Number.isFinite(s.lat) && Number.isFinite(s.lon));
  }
});
