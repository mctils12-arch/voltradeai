import { test } from "node:test";
import assert from "node:assert/strict";
import { indexColorCss, indexLuminance, indexTextColor, unitSymbol, readingAge } from "./aqIndex";

// ── indexColorCss — the omitted-channel rule is the whole point ──────────────

test("a protobuf color with omitted channels treats them as 0, not as absent", () => {
  // Live shape from /api/data/air-quality's usa_epa index: green only.
  assert.equal(indexColorCss({ green: 0.89411765 }), "rgb(0, 228, 0)");
});

test("a full three-channel color round-trips to 0-255", () => {
  assert.equal(indexColorCss({ red: 0.49803922, green: 0.8039216, blue: 0.2 }), "rgb(127, 205, 51)");
  assert.equal(indexColorCss({ red: 1, green: 1, blue: 1 }), "rgb(255, 255, 255)");
});

test("a color object with no recognized channel is null, NOT black", () => {
  // Rendering {} as rgb(0,0,0) would invent a severity the provider never sent.
  assert.equal(indexColorCss({}), null);
  assert.equal(indexColorCss(null), null);
  assert.equal(indexColorCss(undefined), null);
  assert.equal(indexColorCss({ alpha: 1 }), null);
});

test("non-numeric and out-of-range channels are clamped or ignored", () => {
  assert.equal(indexColorCss({ red: Number.NaN, green: 0.5 }), "rgb(0, 128, 0)");
  assert.equal(indexColorCss({ red: 5, green: -2, blue: 0.5 }), "rgb(255, 0, 128)");
  assert.equal(indexColorCss({ red: "1" as unknown as number }), null);
});

// ── contrast — Google's AQI palette spans both ends ──────────────────────────

test("luminance is 0 for black and 1 for white", () => {
  assert.equal(indexLuminance({ red: 0, green: 0, blue: 0 }), 0);
  assert.ok(Math.abs(indexLuminance({ red: 1, green: 1, blue: 1 }) - 1) < 1e-9);
  assert.equal(indexLuminance(null), 0);
});

test("chip text flips to dark on the light end of the palette and light on the dark end", () => {
  assert.equal(indexTextColor({ green: 0.89411765 }), "#0b0f14"); // EPA "Good" green
  assert.equal(indexTextColor({ red: 0.49803922, green: 0.8039216, blue: 0.2 }), "#0b0f14");
  assert.equal(indexTextColor({ red: 0.5, green: 0, blue: 0.1 }), "#ffffff"); // maroon "Hazardous" end
  assert.equal(indexTextColor(null), "#ffffff"); // no color → neutral dark chip
});

// ── unit enums — decode table only, never a guess ────────────────────────────

test("known unit enums decode to their conventional symbols", () => {
  assert.equal(unitSymbol("MICROGRAMS_PER_CUBIC_METER"), "µg/m³");
  assert.equal(unitSymbol("PARTS_PER_BILLION"), "ppb");
  assert.equal(unitSymbol("PARTS_PER_MILLION"), "ppm");
});

test("an unknown unit enum passes through verbatim rather than being guessed", () => {
  assert.equal(unitSymbol("SOMETHING_NEW_UPSTREAM"), "SOMETHING_NEW_UPSTREAM");
  assert.equal(unitSymbol(null), "");
  assert.equal(unitSymbol(undefined), "");
  assert.equal(unitSymbol(""), "");
});

// ── data age (Law V) ────────────────────────────────────────────────────────

test("reading age reports minutes, hours, then days", () => {
  const now = Date.parse("2026-08-20T16:00:00Z");
  assert.equal(readingAge("2026-08-20T15:30:00Z", now), "30m old");
  assert.equal(readingAge("2026-08-20T13:00:00Z", now), "3h old");
  assert.equal(readingAge("2026-08-18T13:00:00Z", now), "2d old");
});

test("a missing or unparseable timestamp is null, never a fabricated 0m", () => {
  const now = Date.parse("2026-08-20T16:00:00Z");
  assert.equal(readingAge(null, now), null);
  assert.equal(readingAge(undefined, now), null);
  assert.equal(readingAge("not a date", now), null);
});

test("a future timestamp reads 'just now' rather than a negative age", () => {
  const now = Date.parse("2026-08-20T16:00:00Z");
  assert.equal(readingAge("2026-08-20T16:30:00Z", now), "just now");
});
