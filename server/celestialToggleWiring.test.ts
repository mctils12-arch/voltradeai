// celestialToggleWiring — ratchet: a space-frame preference can never ship
// with no way to reach it, and a live subscription can never leak.
//
// The defect class (live report 2026-08-13): B6 universal lighting shipped a
// persisted `realisticLighting` pref AND a `setRealisticLighting` handle on
// the space frame — but no CELESTIAL panel row ever drove them. The feature
// was therefore unreachable, and the user reasonably concluded the day/night
// shading "only works on the earth", because the only switch they could find
// was the unrelated Earth-map `daynight` geojson layer (which shades the flat
// map and nothing else). Every body path already honored the pref; the gap was
// pure wiring.
//
// Second defect class caught in the same pass: `offApollo` was created by
// subscribeApolloSitesPref at space-view mount and never called in the
// cleanup, so every enter/exit orphaned a listener that then called into a
// disposed handle.
//
// This test scrapes datamap.tsx and asserts, for each space-frame pref the
// panel is expected to drive: a toggle row exists, a live subscription exists,
// and that subscription is released in the cleanup.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const datamap = (): string =>
  fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");

/** pref setter -> the CELESTIAL row key that must drive it. Add a row here in
 *  the same PR that adds a space-frame toggle pref. */
const TOGGLES: Array<{ setter: string; subscribe: string; rowKey: string; off: string }> = [
  { setter: "setOrbitPathsPref", subscribe: "subscribeOrbitPathsPref", rowKey: "orbits", off: "offOrbits" },
  { setter: "setApolloSitesPref", subscribe: "subscribeApolloSitesPref", rowKey: "apollo", off: "offApollo" },
  { setter: "setRealisticLightingPref", subscribe: "subscribeRealisticLightingPref", rowKey: "lighting", off: "offRealistic" },
  { setter: "setMotionTrailsPref", subscribe: "subscribeMotionTrailsPref", rowKey: "trails", off: "offTrails" },
  { setter: "setMilkyWayPref", subscribe: "subscribeMilkyWayPref", rowKey: "galaxy", off: "offGalaxy" },
  { setter: "setEclipticGridPref", subscribe: "subscribeEclipticGridPref", rowKey: "grid", off: "offGrid" },
  { setter: "setBodyLabelsPref", subscribe: "subscribeBodyLabelsPref", rowKey: "labels", off: "offLabels" },
  { setter: "setLockHorizonPref", subscribe: "subscribeLockHorizonPref", rowKey: "lockhorizon", off: "offLock" },
];

test("RATCHET: every space-frame pref has a CELESTIAL panel row that drives it", () => {
  const src = datamap();
  for (const t of TOGGLES) {
    assert.ok(src.includes(`key: "${t.rowKey}"`),
      `CELESTIAL panel row key "${t.rowKey}" missing from datamap.tsx — a space-frame pref with no row is unreachable UI (the B6 realistic-lighting precedent)`);
    assert.ok(src.includes(`${t.setter}(!cel`),
      `panel row "${t.rowKey}" must call ${t.setter} — otherwise the switch renders but changes nothing`);
  }
});

test("RATCHET: every space-frame subscription is released in the space-view cleanup (no orphaned listeners)", () => {
  const src = datamap();
  const cleanup = src.match(/spaceCleanupRef\.current = \(\) => \{([\s\S]*?)\};/);
  assert.ok(cleanup, "spaceCleanupRef teardown not found — if it moved, update this scrape with it");
  const body = cleanup![1];
  for (const t of TOGGLES) {
    // the subscription is created at mount...
    assert.ok(src.includes(`const ${t.off} = ${t.subscribe}(`),
      `${t.subscribe} must be captured as ${t.off} at space-view mount`);
    // ...and MUST be released, or every enter/exit orphans a listener that
    // then fires into a disposed frame handle (the offApollo leak, 2026-08-13)
    assert.ok(new RegExp(`\\b${t.off}\\(\\)`).test(body),
      `${t.off}() is never called in the space-view cleanup — subscription leaks on every enter/exit (offApollo precedent)`);
  }
});

test("the realistic-lighting row states it covers the Moon AND planets, and says what OFF means", () => {
  const src = datamap();
  const row = src.slice(src.indexOf('key: "lighting"'), src.indexOf('key: "lighting"') + 1400);
  assert.ok(/moon/i.test(row) && /planet/i.test(row),
    "the lighting row's status must name the Moon and planets — the whole point of the report was that the user could not tell which surfaces a lighting switch governs");
  assert.ok(/full-bright|even-lit/i.test(row),
    "the OFF status must say bodies render evenly lit (inspection mode), so users know night-side features become visible");
  assert.ok(/hidden, not faked|not faked/i.test(row),
    "honesty: OFF hides real lighting geometry — it must not read as if the terminator were being invented or removed from the physics");
});

test("RATCHET: the realistic flag reaches the MOON patch AND the textured spheres (never Earth-only again)", () => {
  const frame = fs.readFileSync(
    path.join(here, "..", "client", "src", "lib", "celestial", "spaceFrame.ts"), "utf8");
  // the ONE flag the panel row drives
  assert.ok(/let realistic = opts\.realisticLighting \?\? getRealisticLightingPref\(\)/.test(frame),
    "spaceFrame must derive its `realistic` flag from the pref (opts override + pref fallback)");
  assert.ok(/setRealisticLighting\(/.test(frame),
    "spaceFrame must expose setRealisticLighting for the live subscription");
  // ...and it must reach BOTH render families, or the toggle silently covers
  // only some bodies — the exact confusion in the 2026-08-13 report.
  assert.ok(/fullBright:\s*!realistic/.test(frame),
    "the Moon's close-up surface patch must receive fullBright: !realistic (moonSurface consumes view.fullBright)");
  const sphereSites = frame.match(/fullBright:\s*!\w+(\.\w+)*\.realistic/g) || [];
  assert.ok(sphereSites.length >= 2,
    `the textured-sphere paths (planets + the Moon's disc) must receive the realistic flag; found ${sphereSites.length} call site(s)`);

  // and the consumers must actually honor it
  const moon = fs.readFileSync(
    path.join(here, "..", "client", "src", "lib", "celestial", "moonSurface.ts"), "utf8");
  assert.ok(/fullBright \? 1 :/.test(moon),
    "moonSurface must render full-bright when the flag is set — otherwise the Moon keeps its terminator with lighting OFF and night-side landing sites stay invisible");
  const sphere = fs.readFileSync(
    path.join(here, "..", "client", "src", "lib", "celestial", "textureSphere.ts"), "utf8");
  assert.ok(/lightCam && !fullBright/.test(sphere),
    "textureSphere must skip the lambert term when full-bright — this is what makes every PLANET honor the toggle");
});

test("the Earth-map daynight status line points at the space-view lighting toggle", () => {
  const src = datamap();
  const i = src.indexOf('setStatus("daynight", "active"');
  assert.ok(i > 0, "daynight active-status call not found");
  const status = src.slice(i, i + 900);
  assert.ok(/THIS MAP only|map only/i.test(status),
    "the daynight layer shades only the flat map — say so, or users read it as governing the Moon/planets too (the 2026-08-13 report)");
  assert.ok(/Realistic lighting/.test(status),
    "name the CELESTIAL › Realistic lighting toggle so the space-view control is discoverable from the layer the user actually found");
});

// ── SITE CLAIM (2026-08-13 report: "when i click on Moon Mission it thinks i
//    am clicking on the moon and pulls up that card"). A marker click flies to
//    the Moon to reach the site, and that flight used to open the Moon's BODY
//    card on top of the site card — two cards for one click. The map already
//    solves this with a feature claim (the more specific selection owns the
//    click); this is the same rule in the space frame. ──
test("RATCHET: a site fly-to claims the click and suppresses the body card", () => {
  const frame = fs.readFileSync(
    path.join(here, "..", "client", "src", "lib", "celestial", "spaceFrame.ts"), "utf8");
  assert.ok(/siteClaim\?:\s*boolean/.test(frame),
    "beginFlight must accept siteClaim — without it a marker click reopens the body card");
  assert.ok(/if \(o\?\.siteClaim\) opts\.onFocusBody\?\.\(null\)/.test(frame),
    "a site-claimed flight must CLOSE the body card, not open it");
  const flyTo = frame.slice(frame.indexOf("const flyToSiteImpl"), frame.indexOf("const flyToSiteImpl") + 900);
  assert.ok(/siteClaim:\s*true/.test(flyTo),
    "flyToSiteImpl must claim the click, or clicking a mission marker opens two cards");
  assert.ok(/onFocusSite\?\.\(siteId\)/.test(flyTo),
    "the site card is the one card a marker click opens");
});

test("RATCHET: space cards step aside for the open layers panel (nothing covers anything)", () => {
  const css = fs.readFileSync(path.join(here, "..", "client", "src", "index.css"), "utf8");
  const i = css.indexOf('.vt-map-page[data-vt-panel-open="true"] .vt-site-card.vt-space-card');
  assert.ok(i > 0,
    "the space cards are right-anchored on the same edge the layers panel owns — they must shift when it opens (the nav cluster and flight profile already do)");
  const rule = css.slice(i, i + 260);
  assert.ok(/right:\s*3\d\dpx/.test(rule),
    "shift must clear the panel width");
  assert.ok(/:not\(\[style\*="left:"\]\)/.test(css.slice(i - 120, i + 260)),
    "a card the user dragged carries an inline left/top — their placement must win over ours");
});
