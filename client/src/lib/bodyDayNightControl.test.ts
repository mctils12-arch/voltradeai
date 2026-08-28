// The "Day/Night & Moon" row must CARRY the body-lighting switch, not point
// at it.
//
// WHY THIS TEST EXISTS. The 2026-08-13 report was "day/night is on but the
// Moon has no dark side." That pass answered it by appending a sentence to the
// layer's status line naming where the real switch lives (CELESTIAL ›
// Realistic lighting). On 2026-08-14 the identical report came back.
//
// A POINTER IS NOT A CONTROL. The user was in the layers panel, reading a row
// called "Day/Night & Moon", while the switch that governs the Moon sat in a
// different panel that was collapsed. Offered a rename as the fix, the answer
// was explicit: "i want the feature not to rename." So the row keeps the name
// and earns it.
//
// This is a resurrection: the fix was originally shipped in PR #844
// (2026-08-14), which never merged — a main history rewrite orphaned its
// branch before CI could ever complete on its final commits (git reports
// "refusing to merge unrelated histories" against current main). Re-derived
// against current datamap.tsx this session (2026-08-28) rather than
// git-merged; the GPU moonSurfaceGL raycast port bundled in that same PR is
// NOT included here — it was never wired into the render path, is materially
// larger/riskier, and is re-queued separately in open_questions.md.
//
// These assertions pin the three properties that make the fix true, because
// each one is easy to lose in a later refactor and none of them fails loudly:
//   1. the control exists in the row at all;
//   2. it drives the SAME pref as the CELESTIAL panel — one source of truth,
//      so the two switches can never disagree;
//   3. it appears only where it does something (camera at a body).
//
// Source-scraped rather than DOM-rendered: datamap.tsx is ~14k lines and has
// no test harness that mounts it. Asserting on the CONSTRUCT with comments
// stripped first (L15/L18 — prose describing a control must not satisfy a
// check for the control).
// Run: npx tsx --test client/src/lib/bodyDayNightControl.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "..", "..", "..");
const SRC = path.join(repoRoot, "client/src/pages/datamap.tsx");

/** datamap.tsx with comment lines blanked, so prose cannot satisfy a match. */
function code(): string {
  return fs
    .readFileSync(SRC, "utf8")
    .split("\n")
    .filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l))
    .join("\n");
}

test("the daynight row renders a body-lighting control", () => {
  const src = code();
  assert.match(
    src,
    /l\.id === "daynight" && spaceActive &&/,
    'the "Day/Night & Moon" row must render a body-lighting block',
  );
  assert.match(src, /data-vt-body-daynight/, "the switch needs a stable test handle");
});

test("it drives the SAME pref as the CELESTIAL panel — not a second copy", () => {
  // Two switches over two prefs would drift, and the user would find one of
  // them lying. Both must call setRealisticLightingPref and read celRealistic.
  const src = code();
  const block = src.slice(src.indexOf('l.id === "daynight" && spaceActive'));
  const row = block.slice(0, block.indexOf("</div>"));
  assert.match(row, /setRealisticLightingPref\(!celRealistic\)/, "row toggles the shared pref");
  assert.match(row, /aria-checked=\{celRealistic\}/, "row reflects the shared pref");

  // And the CELESTIAL panel still drives the same one, so this is genuinely
  // one pref with two surfaces rather than a fork.
  assert.ok(
    (src.match(/setRealisticLightingPref\(!celRealistic\)/g) || []).length >= 2,
    "both the layers row and the CELESTIAL panel must drive the same pref",
  );
});

test("the control is gated on spaceActive, NOT on the row's own switch", () => {
  // Out at a body the map shade is invisible. Gating the body control behind
  // the map layer's switch would hide the lighting control exactly when it is
  // the only one of the two that does anything.
  const src = code();
  const i = src.indexOf('l.id === "daynight" && spaceActive');
  assert.ok(i > 0);
  const cond = src.slice(i, src.indexOf("(", i));
  assert.ok(
    !/\bon\b\s*&&/.test(cond),
    `the body control must not be gated on the row's own switch — got: ${cond.trim()}`,
  );
});

test("the status line no longer sends the user hunting for another panel", () => {
  // The 2026-08-13 fix. It was true and it did not work, because it described
  // a control instead of being one.
  const src = code();
  assert.ok(
    !/CELESTIAL › Realistic lighting/.test(src) ||
      /this row also carries that body's day\/night/.test(src),
    "the daynight status text must not read as a scavenger hunt",
  );
});
