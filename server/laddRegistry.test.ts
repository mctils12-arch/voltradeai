// FAA LADD registry (2026-10-14 repair): the FAA-filed plan of N843S, a LADD
// aircraft, was being served publicly. These tests pin the registry that now
// gates FAA-sourced data. Run: npx tsx --test server/laddRegistry.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { LaddRegistry, LADD_MEMORY_MS, LADD_WITHHELD_NOTE } from "./laddRegistry";
import { filedPlanUnlessLadd } from "./procedures";

test("isLadd matches by hex OR callsign, case-insensitively", () => {
  const r = new LaddRegistry({ file: null, now: () => 1_000 });
  r.note([{ icao24: "AB8C8E", callsign: "n843s " }]);
  assert.equal(r.isLadd("ab8c8e"), true);
  assert.equal(r.isLadd(null, "N843S"), true, "callsign alone is enough (procedures route is callsign-keyed)");
  assert.equal(r.isLadd("ab8c8e", "OTHER1"), true);
  assert.equal(r.isLadd("a00001", "UAL1"), false);
  assert.equal(r.isLadd(null, null), false);
});

test("a sighting is remembered for LADD_MEMORY_MS, then forgotten", () => {
  let now = 0;
  const r = new LaddRegistry({ file: null, now: () => now });
  r.note([{ icao24: "ab8c8e", callsign: "N843S" }]);
  now = LADD_MEMORY_MS - 1;
  assert.equal(r.isLadd("ab8c8e"), true, "parked at the gate days later is still known");
  now = LADD_MEMORY_MS + 1;
  assert.equal(r.isLadd("ab8c8e"), false);
});

test("membership persists across a restart", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "ladd-"));
  const file = path.join(dir, "sub", "ladd_registry.json");
  const a = new LaddRegistry({ file, now: () => 5_000 });
  a.note([{ icao24: "ab8c8e", callsign: "N843S" }]);
  assert.ok(existsSync(file), "first sighting is saved");
  const b = new LaddRegistry({ file, now: () => 6_000 });
  assert.equal(b.isLadd("ab8c8e", null), true);
  assert.equal(b.isLadd(null, "N843S"), true);
  assert.deepEqual(Object.keys(JSON.parse(readFileSync(file, "utf8"))).sort(), ["callsigns", "hexes"]);
  rmSync(dir, { recursive: true, force: true });
});

test("a corrupt registry file does not crash boot", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "ladd-bad-"));
  const file = path.join(dir, "ladd_registry.json");
  writeFileSync(file, "{not json");
  const logged: unknown[] = [];
  const real = console.error;
  console.error = (...a: unknown[]) => { logged.push(a); };
  try {
    const r = new LaddRegistry({ file });
    assert.equal(r.isLadd("ab8c8e"), false);
  } finally { console.error = real; }
  assert.equal(logged.length, 1, "the failure is logged, not swallowed");
  rmSync(dir, { recursive: true, force: true });
});

test("procedures: a LADD callsign never gets its FAA-filed plan", () => {
  const plan = { departure: "KCLT", arrival: "KBWI", routeText: "KCLT..ALDAN..KBWI", updatedAt: 1 };
  assert.equal(filedPlanUnlessLadd("N843S", () => plan, (cs) => cs === "N843S"), null);
  assert.deepEqual(filedPlanUnlessLadd("UAL1", () => plan, () => false), plan);
  assert.equal(filedPlanUnlessLadd("UAL1", () => null, () => false), null);
});

test("the withheld note never says whether a plan exists", () => {
  assert.doesNotMatch(LADD_WITHHELD_NOTE, /KBWI|filed to|destination/i);
  assert.match(LADD_WITHHELD_NOTE, /LADD/);
});
