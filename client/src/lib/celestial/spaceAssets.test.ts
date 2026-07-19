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
  textureLonOffsetDeg,
  TEXTURE_LON_OFFSET_DEG,
  STAR_CATALOG_FILE,
  STAR_NAMES_FILE,
  STAR_FIELD_FADE_START_AU,
  starFieldOpacity,
  getStarsPref,
  getRealisticLightingPref,
  setRealisticLightingPref,
  subscribeRealisticLightingPref,
  loadStarCatalog,
} from "./spaceAssets.js";
import { decodeStarAsset } from "./starCatalog.js";
import { readFileSync } from "node:fs";

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

test("star field fade: bright stars appear EARLIER than the panorama, black near Earth", () => {
  assert.equal(starFieldOpacity(1), 0, "black near Earth (1 AU)");
  assert.equal(starFieldOpacity(STAR_FIELD_FADE_START_AU), 0, "starts at 2 AU");
  assert.ok(starFieldOpacity(4) > 0 && starFieldOpacity(4) < 1, "ramping mid inner-system");
  assert.equal(starFieldOpacity(6), 1, "full by 6 AU");
  assert.equal(starFieldOpacity(50), 1, "stays visible far out");
  assert.equal(starFieldOpacity(Number.NaN), 0, "non-finite → hidden");
  // the reconciliation: at 6 AU stars are already full while the panorama
  // (starts 8 AU) is still 0 — the real bright stars lead the faint glow
  assert.equal(milkyWayOpacity(6), 0, "panorama still absent where stars are full");
  let prev = -1;
  for (let au = 0; au <= 10; au += 0.25) {
    const o = starFieldOpacity(au);
    assert.ok(o >= prev && o >= 0 && o <= 1, "monotone clamped");
    prev = o;
  }
});

test("star catalog asset: committed, decodes to ~9096 real stars, compact vs the source", () => {
  const bin = readFileSync(path.join(SPACE_DIR, STAR_CATALOG_FILE));
  assert.ok(bin.byteLength < 200_000, `compact asset (${bin.byteLength} bytes) — far below the 932 KB source`);
  const ab = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const cat = decodeStarAsset(ab);
  assert.ok(cat.count >= 9000 && cat.count <= 9200, `star count ${cat.count} ≈ Yale BSC`);
  // every direction is a unit vector; magnitudes span the catalog range
  let minMag = Infinity;
  let maxMag = -Infinity;
  for (let i = 0; i < cat.count; i++) {
    const l = Math.hypot(cat.dirEq[i * 3], cat.dirEq[i * 3 + 1], cat.dirEq[i * 3 + 2]);
    assert.ok(Math.abs(l - 1) < 1e-3, `star ${i} unit direction`);
    if (cat.mag[i] < minMag) minMag = cat.mag[i];
    if (cat.mag[i] > maxMag) maxMag = cat.mag[i];
  }
  assert.ok(minMag < -1, `has a very bright star (Sirius −1.46): min ${minMag}`);
  assert.ok(maxMag > 6, `has faint naked-eye stars: max ${maxMag}`);
  // the brightest-names file exists and references real indices
  const names = JSON.parse(readFileSync(path.join(SPACE_DIR, STAR_NAMES_FILE), "utf8"));
  assert.ok(Array.isArray(names) && names.length >= 10, "bright-star names bundled");
  assert.ok(names.some((n: { name: string }) => n.name === "Sirius"), "Sirius named");
});

test("loadStarCatalog: fetches + decodes the bundled binary, attaches names (fixture fetch)", async () => {
  const bin = readFileSync(path.join(SPACE_DIR, STAR_CATALOG_FILE));
  const names = readFileSync(path.join(SPACE_DIR, STAR_NAMES_FILE));
  const orig = globalThis.fetch;
  (globalThis as unknown as { fetch: typeof fetch }).fetch = (async (url: string) => {
    const u = String(url);
    if (u.endsWith(STAR_CATALOG_FILE)) {
      return { ok: true, arrayBuffer: async () => bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) } as Response;
    }
    if (u.endsWith(STAR_NAMES_FILE)) {
      return { ok: true, json: async () => JSON.parse(names.toString()) } as Response;
    }
    return { ok: false } as Response;
  }) as typeof fetch;
  try {
    const cat = await loadStarCatalog("/space/");
    assert.ok(cat && cat.count > 9000, "catalog loaded");
    assert.ok(cat!.names.size >= 10, "names attached");
    // null on failure, never throw
    (globalThis as unknown as { fetch: typeof fetch }).fetch = (async () => ({ ok: false } as Response)) as typeof fetch;
    const none = await loadStarCatalog("/space/");
    assert.equal(none, null, "missing asset → null (degrade, never break)");
  } finally {
    (globalThis as unknown as { fetch: typeof fetch }).fetch = orig;
  }
});

test("B6 realistic-lighting pref: default ON + set/subscribe round trip", () => {
  assert.equal(getRealisticLightingPref(), true, "realistic lighting default ON");
  let fired = 0;
  const off = subscribeRealisticLightingPref(() => { fired++; });
  setRealisticLightingPref(false);
  assert.equal(getRealisticLightingPref(), false, "inspection mode after set(false)");
  assert.equal(fired, 1);
  setRealisticLightingPref(false);
  assert.equal(fired, 1, "no-op set never notifies");
  setRealisticLightingPref(true);
  assert.equal(getRealisticLightingPref(), true);
  off();
});

test("star prefs: default ON", () => {
  assert.equal(getStarsPref(), true, "real stars default ON");
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

test("B5 texture longitude offset: planet map family 180°, Moon/Sun 0°", () => {
  // the classic planet map family (marsmap1k/mercurymap/venusmap) places lon 0
  // at the texture's west edge — a 180° shift from equirectUV's centred lon 0.
  // Measured against each body's own Trek mosaic (Mars 0.63 / Venus 0.54 /
  // Mercury ~0.25 correlation, all peaking ~180°) on 2026-07-19.
  assert.equal(textureLonOffsetDeg("mars"), 180);
  assert.equal(textureLonOffsetDeg("mercury"), 180);
  assert.equal(textureLonOffsetDeg("venus"), 180);
  // the LRO moon_8k mosaic + everything unspecified keep the centred convention
  assert.equal(textureLonOffsetDeg("moon"), 0);
  assert.equal(textureLonOffsetDeg("sun"), 0);
  assert.equal(textureLonOffsetDeg("jupiter"), 0);
  assert.equal(textureLonOffsetDeg("earth"), 0);
  assert.equal(textureLonOffsetDeg("nonexistent"), 0);
  // the map only carries the offset bodies (no accidental 0-valued entries)
  assert.deepEqual(new Set(Object.keys(TEXTURE_LON_OFFSET_DEG)), new Set(["mars", "mercury", "venus"]));
});
