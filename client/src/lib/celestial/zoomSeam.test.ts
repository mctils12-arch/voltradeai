// zoomSeam.test — the shared zoom curve, plus a DRIFT GUARD for every place
// that has to mirror it.
//
// Why the drift guard exists: /moon.html is a standalone static page with no
// bundler, so it cannot import this module — it mirrors ZOOM_STEP_PER_NOTCH as
// a literal. That copy silently drifted to 1.25 while this file said 1.18, so
// the moon zoomed ~38% further per wheel notch than Earth until a human noticed
// ("it should zoom at the same rate that the earth does", 2026-07-28). A
// mirrored constant with no test is a bug with a delay fuse; this is the fuse
// remover.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import {
  ZOOM_STEP_PER_NOTCH,
  zoomStepFactor,
  wheelDeltaForFactor,
  ZOOM_BUTTON_DELTAY,
  SEAM_ENTRY_DELTAY,
} from "./zoomSeam.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../.."); // client/src/lib/celestial -> repo root

test("zoomStepFactor: one notch is exactly ZOOM_STEP_PER_NOTCH, and it is exponential", () => {
  assert.equal(zoomStepFactor(0), 1, "no scroll = no change");
  assert.ok(Math.abs(zoomStepFactor(100) - ZOOM_STEP_PER_NOTCH) < 1e-12,
    "deltaY 100 = one notch out");
  // negative zooms IN, and is the exact reciprocal
  assert.ok(Math.abs(zoomStepFactor(-100) - 1 / ZOOM_STEP_PER_NOTCH) < 1e-12);
  // exponential: two notches compose into the square
  assert.ok(Math.abs(zoomStepFactor(200) - ZOOM_STEP_PER_NOTCH ** 2) < 1e-12);
  // a trackpad's fractional deltas ride the same curve
  assert.ok(zoomStepFactor(50) > 1 && zoomStepFactor(50) < ZOOM_STEP_PER_NOTCH);
});

test("wheelDeltaForFactor inverts zoomStepFactor, so discrete inputs ride the wheel curve", () => {
  for (const f of [0.25, 0.5, 1 / ZOOM_STEP_PER_NOTCH, 1, ZOOM_STEP_PER_NOTCH, 2, 4]) {
    const back = zoomStepFactor(wheelDeltaForFactor(f));
    assert.ok(Math.abs(back - f) < 1e-12, `round-trips factor ${f} (got ${back})`);
  }
  assert.ok(wheelDeltaForFactor(0.5) < 0, "zooming in is negative deltaY");
});

test("one +/- button click is exactly one MapLibre zoom level (x2)", () => {
  assert.ok(Math.abs(zoomStepFactor(ZOOM_BUTTON_DELTAY) - 2) < 1e-12);
  assert.ok(ZOOM_BUTTON_DELTAY > SEAM_ENTRY_DELTAY,
    "a deliberate button click must clear the seam-entry bar, or the buttons could not cross the seam");
});

test("DRIFT GUARD: /moon.html mirrors ZOOM_STEP_PER_NOTCH and the x2 button step", () => {
  const moon = readFileSync(resolve(REPO, "client/public/moon.html"), "utf8");

  // the mirrored literal must equal this module's value
  const m = moon.match(/const\s+ZOOM_STEP_PER_NOTCH\s*=\s*([0-9.]+)\s*;/);
  assert.ok(m, "moon.html must declare a mirrored ZOOM_STEP_PER_NOTCH constant");
  assert.equal(
    Number(m![1]),
    ZOOM_STEP_PER_NOTCH,
    `moon.html mirrors ZOOM_STEP_PER_NOTCH = ${m![1]} but zoomSeam.ts says ` +
      `${ZOOM_STEP_PER_NOTCH}. zoomSeam.ts is the source of truth — update the ` +
      `mirror in client/public/moon.html so the moon zooms at Earth's rate.`,
  );

  // ...and it must actually be USED for the wheel step, not just declared
  assert.match(
    moon,
    /WHEEL_LOG_PER_PX\s*=\s*Math\.log\(\s*ZOOM_STEP_PER_NOTCH\s*\)\s*\/\s*100/,
    "moon.html's wheel step must be derived from the mirrored constant, not a separate literal",
  );

  // a button tap is one zoom level there too (zoomBy takes a log delta)
  assert.match(moon, /zoomBy\(\s*sign\s*\*\s*Math\.log\(2\)\s*\)/,
    "a +/- tap on the moon must be exactly one MapLibre zoom level (x2)");

  // the page must name where the truth lives, so the next editor fixes the right file
  assert.match(moon, /zoomSeam\.ts/,
    "moon.html must cite zoomSeam.ts as the source of truth for the mirrored constant");
});
