// spaceAssets tests — manifest integrity (assets actually committed), the
// Milky Way fade curve, the Moon texture-tier chooser, required license
// credits, and the pref stores. Hermetic node:test via tsx (no DOM).
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import {
  SPACE_TEXTURE_FILES,
  MOON_TIER_FILES,
  MOON_BUMP_FILE,
  SATURN_RING_FILES,
  MILKY_WAY_FILE,
  MILKY_WAY_CREDIT,
  SPACE_IMAGERY_CREDIT,
  MILKY_WAY_FADE_START_AU,
  milkyWayOpacity,
  moonTextureTier,
  MOON_TIER_2K_MIN_DISC_PX,
  MOON_TIER_8K_MIN_DISC_PX,
  getMilkyWayPref,
  setMilkyWayPref,
  subscribeMilkyWayPref,
  getEclipticGridPref,
  getMotionTrailsPref,
  getBodyLabelsPref,
} from "./spaceAssets.js";

const SPACE_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../public/space",
);

test("manifest: every declared texture asset is committed under client/public/space", () => {
  const files = new Set<string>([
    ...Object.values(SPACE_TEXTURE_FILES),
    ...Object.values(MOON_TIER_FILES),
    MOON_BUMP_FILE,
    SATURN_RING_FILES.color,
    SATURN_RING_FILES.alpha,
    MILKY_WAY_FILE,
  ]);
  assert.ok(files.size >= 13, `expected the full asset set, got ${files.size}`);
  for (const f of Array.from(files)) {
    assert.ok(existsSync(path.join(SPACE_DIR, f)), `missing asset: ${f}`);
  }
  // Earth is the LIVE MAP (handoff integration point 1) — never a texture
  assert.equal(SPACE_TEXTURE_FILES.earth, undefined, "Earth must not have a texture slot");
});

test("licenses: the visible credits carry the required attributions", () => {
  assert.match(MILKY_WAY_CREDIT, /solarsystemscope\.com/, "CC-BY attribution target");
  assert.match(MILKY_WAY_CREDIT, /CC-BY/, "CC-BY license named");
  assert.match(SPACE_IMAGERY_CREDIT, /NASA/, "NASA imagery credited");
  assert.match(SPACE_IMAGERY_CREDIT, /LRO/, "Moon LRO source credited");
});

test("milky way fade: reference curve clamp((camAU-8)/25,0,1) — invisible near Earth, saturated far out", () => {
  assert.equal(milkyWayOpacity(0), 0);
  assert.equal(milkyWayOpacity(1), 0, "invisible at Earth range (1 AU)");
  assert.equal(milkyWayOpacity(MILKY_WAY_FADE_START_AU), 0, "starts exactly at 8 AU");
  assert.ok(Math.abs(milkyWayOpacity(20.5) - 0.5) < 1e-12, "midpoint at 20.5 AU");
  assert.equal(milkyWayOpacity(33), 1, "saturated at 33 AU");
  assert.equal(milkyWayOpacity(4721), 1, "STAYS visible at thousands of AU (the confirmed far-zoom view)");
  assert.equal(milkyWayOpacity(Number.NaN), 0, "non-finite camera → hidden");
  let prev = -1;
  for (let au = 0; au <= 60; au += 0.5) {
    const o = milkyWayOpacity(au);
    assert.ok(o >= prev, "monotone non-decreasing");
    assert.ok(o >= 0 && o <= 1, "clamped");
    prev = o;
  }
});

test("moon texture tier: 1k default, 2k on a large disc, 8k ONLY focused/close, eviction side", () => {
  assert.equal(moonTextureTier(10, false), "1k");
  assert.equal(moonTextureTier(MOON_TIER_2K_MIN_DISC_PX - 0.1, false), "1k");
  assert.equal(moonTextureTier(MOON_TIER_2K_MIN_DISC_PX, false), "2k");
  assert.equal(moonTextureTier(500, false), "2k", "8k is NEVER unfocused — the directive's focused/close rule");
  assert.equal(moonTextureTier(MOON_TIER_8K_MIN_DISC_PX - 0.1, true), "2k");
  assert.equal(moonTextureTier(MOON_TIER_8K_MIN_DISC_PX, true), "8k");
  assert.equal(moonTextureTier(2000, true), "8k");
  assert.equal(moonTextureTier(50, true), "1k", "focused but small stays low");
});

test("view prefs: node-safe defaults + set/subscribe round trip", () => {
  assert.equal(getMilkyWayPref(), true, "milky way default ON");
  assert.equal(getEclipticGridPref(), false, "ecliptic grid default OFF (reference panel default)");
  assert.equal(getMotionTrailsPref(), true, "motion trails default ON");
  assert.equal(getBodyLabelsPref(), true, "labels default ON");
  let fired = 0;
  const off = subscribeMilkyWayPref(() => { fired++; });
  setMilkyWayPref(false);
  assert.equal(getMilkyWayPref(), false);
  assert.equal(fired, 1);
  setMilkyWayPref(false);
  assert.equal(fired, 1, "no-op set never notifies");
  setMilkyWayPref(true);
  assert.equal(getMilkyWayPref(), true);
  assert.equal(fired, 2);
  off();
  setMilkyWayPref(false);
  setMilkyWayPref(true);
  assert.equal(fired, 2, "unsubscribed");
});
