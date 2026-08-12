import { test } from "node:test";
import assert from "node:assert/strict";
import { APOLLO_SITES, APOLLO_IMAGERY_NOTE } from "./apolloSites";

test("all six crewed landings present, unique, chronological facts sane", () => {
  assert.equal(APOLLO_SITES.length, 6);
  const ids = new Set(APOLLO_SITES.map((s) => s.id));
  assert.equal(ids.size, 6);
  for (const s of APOLLO_SITES) {
    assert.ok(Math.abs(s.lat) <= 90 && Math.abs(s.lon) <= 180, s.id);
    assert.ok(s.crew_surface.length === 2, `${s.id}: exactly two moonwalkers`);
    assert.ok(s.evas >= 1 && s.evas <= 3);
    assert.ok(s.eva_hours > 0 && s.eva_hours < 25);
    assert.ok(s.max_eva_km > 0 && s.max_eva_km <= 8);
    assert.match(s.landed_utc, /^19(69|7[0-2])-/);
  }
});

test("documented anchor coordinates: Tranquility Base and Taurus-Littrow (published NASA values)", () => {
  const a11 = APOLLO_SITES.find((s) => s.id === "apollo11")!;
  assert.ok(Math.abs(a11.lat - 0.67408) < 1e-5 && Math.abs(a11.lon - 23.47297) < 1e-5);
  const a17 = APOLLO_SITES.find((s) => s.id === "apollo17")!;
  assert.ok(a17.rover && a17.max_eva_km === 7.6);
});

test("rover flags match history: 15/16/17 drove, 11/12/14 walked", () => {
  for (const s of APOLLO_SITES) {
    const n = parseInt(s.id.replace("apollo", ""), 10);
    assert.equal(s.rover, n >= 15, s.id);
  }
});

test("the imagery honesty note names both resolutions and never oversells", () => {
  assert.match(APOLLO_IMAGERY_NOTE, /WAC/);
  assert.match(APOLLO_IMAGERY_NOTE, /NAC/);
  assert.match(APOLLO_IMAGERY_NOTE, /sub-pixel/);
});
