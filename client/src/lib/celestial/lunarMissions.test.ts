// Data-integrity tests for the lunar surface missions layer.
//
// These coordinates ship to users as fact, so the tests are about the data
// being WRONG, not about the code running. The single most likely defect in a
// dataset like this is a west/east longitude sign flip — it silently teleports
// a site to the opposite hemisphere, and for the far-side missions (Chang'e 4,
// Chang'e 6, LADEE) it would put them on the near side where they would draw
// in the wrong place and never be culled. near_side is therefore asserted
// against the longitude rather than trusted as typed.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LUNAR_SITES, LUNAR_FAR_SIDE_SITES, LUNAR_SIDE_NOTE, LUNAR_COVERAGE_NOTE,
  APOLLO_SITE_IDS, APOLLO_IMAGERY_NOTE, NON_APOLLO_IMAGERY_NOTE,
  getLunarSite, lrocFeaturedUrl,
} from "./lunarMissions";
import { APOLLO_NAC_SITES } from "./lroc";

test("ids are unique and stable", () => {
  const ids = LUNAR_SITES.map((s) => s.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate site id");
  assert.ok(LUNAR_SITES.length >= 30, `expected the full catalogue, got ${LUNAR_SITES.length}`);
});

test("the six Apollo ids keep their historical strings (join key into the NAC tier)", () => {
  for (const id of APOLLO_SITE_IDS) {
    assert.ok(getLunarSite(id), `${id} missing — renaming an Apollo id breaks the NAC site strips and the site card`);
  }
  // every NAC strip must have a site behind it, or the strip can never load
  for (const nac of APOLLO_NAC_SITES) {
    assert.ok(getLunarSite(nac.id),
      `lroc.ts APOLLO_NAC_SITES has "${nac.id}" with no matching LUNAR_SITES entry — the NAC tier would never trigger`);
  }
});

test("coordinates are in range and longitude is POSITIVE EAST", () => {
  for (const s of LUNAR_SITES) {
    assert.ok(s.lat >= -90 && s.lat <= 90, `${s.id}: lat ${s.lat} out of range`);
    assert.ok(s.lon > -180.0001 && s.lon <= 180.0001, `${s.id}: lon ${s.lon} out of range`);
    assert.ok(Number.isFinite(s.lat) && Number.isFinite(s.lon), `${s.id}: non-finite coordinate`);
  }
});

test("near_side is CONSISTENT with longitude (catches a west/east sign flip)", () => {
  for (const s of LUNAR_SITES) {
    assert.equal(s.near_side, Math.abs(s.lon) < 90,
      `${s.id} (${s.mission}): near_side=${s.near_side} but lon=${s.lon} — a sign flip would put this site on the wrong hemisphere`);
  }
});

test("the known far-side missions really are far side", () => {
  const far = new Set(LUNAR_FAR_SIDE_SITES.map((s) => s.id));
  for (const id of ["change4", "change6", "ladee"]) {
    assert.ok(far.has(id), `${id} must be FAR side — Chang'e 4/6 landed on the far side and LADEE impacted there`);
  }
  // and nothing near-side leaked into the far list
  for (const s of LUNAR_FAR_SIDE_SITES) {
    assert.ok(Math.abs(s.lon) >= 90, `${s.id} is listed far side with lon ${s.lon}`);
  }
});

test("every site carries a real https source_url", () => {
  for (const s of LUNAR_SITES) {
    assert.match(s.source_url, /^https:\/\//, `${s.id}: source_url must be a real https link — coordinates ship as fact`);
    assert.ok(s.note.length > 20, `${s.id}: note too thin to be useful`);
    assert.ok(s.region.length > 2, `${s.id}: missing region`);
  }
});

test("crewed sites have crew facts; non-crewed sites do NOT (the card branches on this)", () => {
  for (const s of LUNAR_SITES) {
    if (s.kind === "crewed") {
      assert.ok(s.crew, `${s.id}: crewed site without crew facts renders blank fields`);
      assert.equal(s.crew!.surface.length, 2, `${s.id}: two moonwalkers per Apollo landing`);
      assert.ok(s.crew!.eva_hours > 0 && s.crew!.max_eva_km > 0, `${s.id}: EVA facts must be real`);
    } else {
      assert.equal(s.crew, undefined, `${s.id}: only crewed landings carry crew facts`);
    }
  }
  // exactly the six Apollo landings are crewed — nobody else has landed people
  const crewed = LUNAR_SITES.filter((s) => s.kind === "crewed").map((s) => s.id).sort();
  assert.deepEqual(crewed, [...APOLLO_SITE_IDS].sort());
});

test("unsurveyed positions are marked as such and never claim to be located", () => {
  for (const s of LUNAR_SITES) {
    if (s.coord_confidence === "catalogued" || s.coord_confidence === "estimated") {
      assert.match(s.note, /reported|never been (found|located)|not (been )?confirmed|tracking-derived/i,
        `${s.id}: an unsurveyed position must say so in its note — the marker is a reported position, not a found site`);
    }
  }
  // Luna 2's crater has never been found; it must carry a degree-scale bound
  const luna2 = getLunarSite("luna2")!;
  assert.equal(luna2.coord_confidence, "estimated");
  assert.ok(luna2.uncertainty_deg && luna2.uncertainty_deg >= 1,
    "Luna 2 is an impact REGION (~1°, tens of km) — shipping it as a point without that bound would be a false precision claim");
});

test("provisional identifications are flagged (Luna 25 is 'likely', not proven)", () => {
  const l25 = getLunarSite("luna25")!;
  assert.equal(l25.attribution_certain, false,
    "NASA says the new crater is LIKELY Luna 25's — the card must not state it as proven");
  assert.match(l25.note, /likely|probable/i);
});

test("date_is_day_only sites carry no fake timestamp", () => {
  for (const s of LUNAR_SITES) {
    if (s.date_is_day_only) {
      assert.match(s.date_utc, /^\d{4}-\d{2}-\d{2}$/,
        `${s.id}: a day-only date must not carry an invented time`);
    } else {
      assert.match(s.date_utc, /^\d{4}-\d{2}-\d{2}T/, `${s.id}: expected a full instant`);
    }
  }
  // LADEE's impact time was never published
  assert.equal(getLunarSite("ladee")!.date_is_day_only, true);
});

test("China's missions are present, including the far-side sample return", () => {
  const cn = LUNAR_SITES.filter((s) => s.country === "China").map((s) => s.id);
  for (const id of ["change3", "change4", "change5", "change6"]) {
    assert.ok(cn.includes(id), `${id} missing — the human asked specifically for China's missions "even the new one"`);
  }
  const ce6 = getLunarSite("change6")!;
  assert.equal(ce6.kind, "sample_return");
  assert.equal(ce6.near_side, false);
});

test("the side note is COMPUTED, so it cannot go stale like the old hand-written one", () => {
  assert.ok(LUNAR_SIDE_NOTE.includes(String(LUNAR_SITES.length)));
  for (const s of LUNAR_FAR_SIDE_SITES) {
    assert.ok(LUNAR_SIDE_NOTE.includes(s.mission), `${s.mission} must be named in the far-side note`);
  }
  assert.match(LUNAR_SIDE_NOTE, /FAR side/);
});

test("the coverage note refuses to imply a complete catalogue", () => {
  assert.match(LUNAR_COVERAGE_NOTE, /NOT a complete catalogue/i,
    "this is a verified subset — claiming completeness would be the dishonest reading");
  assert.ok(LUNAR_COVERAGE_NOTE.includes(String(LUNAR_SITES.length)));
});

test("hardware points are real extra coordinates, never a route", () => {
  for (const s of LUNAR_SITES) {
    for (const hw of s.hardware ?? []) {
      assert.ok(Math.abs(hw.lat) <= 90 && Math.abs(hw.lon) <= 180, `${s.id}/${hw.label}: bad coordinate`);
      assert.ok(hw.label.length > 2);
    }
  }
  // A15/A16 ship LRV parking points precisely BECAUSE their traverses are not
  // available as data — the points must exist so the card has something true
  for (const id of ["apollo15", "apollo16"]) {
    const s = getLunarSite(id)!;
    assert.ok(s.hardware?.some((h) => /LRV/.test(h.label)),
      `${id}: LROC never released this traverse as data, so the surveyed LRV point is what we can honestly show`);
    assert.match(s.note, /no route line is drawn|unreleased as data/i,
      `${id}: the card must say why no line is drawn rather than leave the gap unexplained`);
  }
});

test("rover missions whose routes are not obtainable say so instead of drawing one", () => {
  for (const id of ["lunokhod1", "lunokhod2", "change3", "change4"]) {
    const s = getLunarSite(id)!;
    assert.match(s.note, /no (route|rover) line is drawn|not available as licensed data|paywalled/i,
      `${id}: a rover with no citable route data must state that, never imply a drawn path`);
  }
});

// ── assertions carried over from apolloSites.test.ts when APOLLO_SITES was
//    superseded by LUNAR_SITES (ported, not dropped — the old array is gone
//    but every fact it guarded is still guarded here) ─────────────────────────

test("Apollo rover flags match history: 15/16/17 drove, 11/12/14 walked", () => {
  for (const id of APOLLO_SITE_IDS) {
    const s = getLunarSite(id)!;
    const n = parseInt(id.replace("apollo", ""), 10);
    assert.equal(s.crew!.rover, n >= 15, `${id}: only Apollo 15, 16 and 17 carried the LRV`);
  }
});

test("Apollo anchor coordinates and EVA facts stay at their published values", () => {
  const a11 = getLunarSite("apollo11")!;
  assert.ok(Math.abs(a11.lat - 0.67416) < 1e-4 && Math.abs(a11.lon - 23.47314) < 1e-4,
    "Tranquility Base drifted from the LRO-surveyed value");
  const a17 = getLunarSite("apollo17")!;
  assert.ok(a17.crew!.rover && a17.crew!.max_eva_km === 7.6,
    "Apollo 17's farthest documented point is 7.6 km");
  for (const id of APOLLO_SITE_IDS) {
    const s = getLunarSite(id)!;
    assert.ok(s.crew!.evas >= 1 && s.crew!.evas <= 3, `${id}: EVA count`);
    assert.ok(s.crew!.eva_hours > 0 && s.crew!.eva_hours < 25, `${id}: EVA hours`);
    assert.ok(s.crew!.max_eva_km > 0 && s.crew!.max_eva_km <= 8, `${id}: max EVA distance`);
    assert.match(s.date_utc, /^19(69|7[0-2])-/, `${id}: Apollo landings are 1969-1972`);
  }
});

test("the imagery honesty notes name both resolutions and never oversell", () => {
  assert.match(APOLLO_IMAGERY_NOTE, /WAC/);
  assert.match(APOLLO_IMAGERY_NOTE, /NAC/);
  assert.match(APOLLO_IMAGERY_NOTE, /sub-pixel/);
  // the non-Apollo note must NOT promise NAC detail — no other site has a strip
  assert.match(NON_APOLLO_IMAGERY_NOTE, /below one pixel|sub-pixel/i);
  assert.ok(!/0\.5 m\/px/.test(NON_APOLLO_IMAGERY_NOTE),
    "only the six Apollo sites stream NAC imagery — no other card may imply that resolution");
});

test("lrocFeaturedUrl builds a real search link", () => {
  assert.match(lrocFeaturedUrl("apollo11"), /^https:\/\/www\.lroc\.asu\.edu\/search\?q=/);
});
