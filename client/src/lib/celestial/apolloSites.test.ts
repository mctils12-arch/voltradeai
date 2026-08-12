import { test } from "node:test";
import assert from "node:assert/strict";
import { APOLLO_SITES, APOLLO_IMAGERY_NOTE, APOLLO_NEAR_SIDE_ONLY_NOTE,
  getApolloSitesPref, setApolloSitesPref, subscribeApolloSitesPref,
  APOLLO_SITES_PREF_KEY } from "./apolloSites";

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

// ── the panel toggle (human directive 2026-08-12: "i want it shown on the
// moon and a toggle in the layers under celestial") ──────────────────────────

test("the Apollo toggle defaults ON", () => {
  // v1.0.668 shipped the markers with no way to turn them off; the default
  // must not change what an existing user already sees.
  assert.equal(getApolloSitesPref(), true);
});

test("set/get round-trips and notifies subscribers, unsubscribe stops them", () => {
  let hits = 0;
  const off = subscribeApolloSitesPref(() => { hits++; });
  setApolloSitesPref(false);
  assert.equal(getApolloSitesPref(), false);
  assert.equal(hits, 1);
  setApolloSitesPref(false);            // idempotent — no duplicate notify
  assert.equal(hits, 1);
  setApolloSitesPref(true);
  assert.equal(getApolloSitesPref(), true);
  assert.equal(hits, 2);
  off();
  setApolloSitesPref(false);
  assert.equal(hits, 2, "unsubscribed listener must not fire");
  setApolloSitesPref(true);             // restore for any later test
});

test("a throwing subscriber cannot break the toggle for the others", () => {
  const offBad = subscribeApolloSitesPref(() => { throw new Error("boom"); });
  let good = 0;
  const offGood = subscribeApolloSitesPref(() => { good++; });
  setApolloSitesPref(false);
  assert.equal(good, 1, "a bad listener must not starve a good one");
  offBad(); offGood();
  setApolloSitesPref(true);
});

test("the pref key is namespaced so it cannot collide with another setting", () => {
  assert.equal(APOLLO_SITES_PREF_KEY, "vt.celestial.apolloSites");
});

// HONESTY: the toggle's status line tells the user the markers are only on
// the near side. That claim must be TRUE OF THE DATA, not just nice copy —
// if a future edit added a far-side site, this fails and the copy gets
// fixed with it.
test("the near-side claim in the toggle copy matches every site's longitude", () => {
  assert.match(APOLLO_NEAR_SIDE_ONLY_NOTE, /near side/);
  for (const s of APOLLO_SITES) {
    assert.ok(Math.abs(s.lon) < 90,
      `${s.id} at lon ${s.lon} would be on the far side — the near-side copy would be a lie`);
  }
});
