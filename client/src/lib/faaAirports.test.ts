// FAA airport-status map layer helpers (datamap.tsx "faa_airports" layer,
// server/faaStatus.ts's own docstring naming this the deliberate follow-up
// once a coordinate table existed). Run:
// npx tsx --test client/src/lib/faaAirports.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { AIRPORT_COORDS, FAA_EVENT_COLOR, FAA_EVENT_LABEL, faaEventColor, faaEventLabel } from "./faaAirports.ts";

test("every airport entry carries a plausible US-region lat/lon (no fabricated/zero coordinates)", () => {
  const codes = Object.keys(AIRPORT_COORDS);
  assert.ok(codes.length >= 100, `expected a substantial curated table, found ${codes.length}`);
  for (const code of codes) {
    const c = AIRPORT_COORDS[code];
    assert.ok(/^[A-Z0-9]{3,4}$/.test(code), `${code}: key must look like an FAA ARPT code`);
    assert.ok(Number.isFinite(c.lat) && c.lat > 5 && c.lat < 72, `${code}: lat out of US/territories range`);
    assert.ok(Number.isFinite(c.lon) && c.lon > -180 && c.lon < 145, `${code}: lon out of US/territories range`);
    assert.ok(c.name && c.name.length > 2, `${code}: missing name`);
    assert.ok(c.city && c.state, `${code}: missing city/state`);
  }
});

test("no duplicate airport codes (a duplicate key would silently shadow an earlier entry)", () => {
  const codes = Object.keys(AIRPORT_COORDS);
  assert.equal(new Set(codes).size, codes.length);
});

test("the server test-fixture airports (server/faaStatus.test.ts) all resolve — the exact live-shape sample this layer must render", () => {
  for (const code of ["DCA", "JFK", "LGA", "ASE"]) {
    assert.ok(AIRPORT_COORDS[code], `${code} missing from the coordinate table`);
  }
});

test("faaEventColor/faaEventLabel: every FaaEvent type from server/faaStatus.ts is covered", () => {
  for (const type of ["ground_stop", "closure", "ground_delay", "delay"] as const) {
    assert.ok(FAA_EVENT_COLOR[type], `${type} missing a color`);
    assert.ok(FAA_EVENT_LABEL[type], `${type} missing a label`);
    assert.equal(faaEventColor(type), FAA_EVENT_COLOR[type]);
    assert.equal(faaEventLabel(type), FAA_EVENT_LABEL[type]);
  }
  // all four colors distinct — never let two severities silently collapse to the same tint
  const colors = new Set(Object.values(FAA_EVENT_COLOR));
  assert.equal(colors.size, 4);
});

test("faaEventColor/faaEventLabel: unknown/missing type degrades honestly, never throws or fabricates a known category", () => {
  assert.equal(faaEventColor(null), "#94a3b8");
  assert.equal(faaEventColor(undefined), "#94a3b8");
  assert.equal(faaEventColor("something_new"), "#94a3b8");
  assert.equal(faaEventLabel(null), "unknown");
  assert.equal(faaEventLabel("something_new"), "something_new");
});
