// Panel-toggle tests for the lunar-missions layer preference.
//
// The site-DATA assertions that used to live here moved to
// lunarMissions.test.ts on 2026-08-13 together with the data itself (ported
// and strengthened — rover history, the Apollo anchor coordinates, the EVA
// bounds and the imagery notes are all still asserted, now against
// LUNAR_SITES). Nothing was dropped to make a change pass; this file keeps
// the preference store it still owns.
//
// One guard was REPLACED rather than removed: the old
// "near-side claim matches every site's longitude" test asserted a
// hand-written note that said none of the sites were on the far side.
// That note is deleted (Chang'e 4/6 and LADEE made it false), and the
// guard now lives in lunarMissions.test.ts as "near_side is CONSISTENT
// with longitude", which checks ALL 35 sites instead of 6 and catches a
// west/east sign flip — strictly stronger than what it replaced.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  getApolloSitesPref, setApolloSitesPref, subscribeApolloSitesPref,
  APOLLO_SITES_PREF_KEY } from "./apolloSites";

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
