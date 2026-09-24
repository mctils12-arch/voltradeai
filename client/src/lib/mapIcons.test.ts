// EPA CAMD utilization helpers (datamap.tsx "plant_operations" layer,
// server/epaCamd.ts's ground-truth stream). Run:
// npx tsx --test client/src/lib/mapIcons.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { camdQuarterHours, camdUtilizationPct, camdUtilizationColor, volcanoAlertColor, ironOreStatusColor, ironSteelTechColor, chemicalFeedstockColor } from "./mapIcons.ts";

test("camdQuarterHours: real calendar length per quarter, not a fixed 91-day assumption", () => {
  assert.equal(camdQuarterHours(2026, 1), 90 * 24); // Jan(31)+Feb(28, non-leap)+Mar(31)
  assert.equal(camdQuarterHours(2024, 1), 91 * 24); // 2024 is a leap year — Feb has 29
  assert.equal(camdQuarterHours(2026, 2), 91 * 24); // Apr(30)+May(31)+Jun(30)
  assert.equal(camdQuarterHours(2026, 3), 92 * 24); // Jul(31)+Aug(31)+Sep(30)
  assert.equal(camdQuarterHours(2026, 4), 92 * 24); // Oct(31)+Nov(30)+Dec(31)
});

test("camdUtilizationPct: sumOpTime / (unitCount x quarter hours), null on missing inputs", () => {
  const hours = camdQuarterHours(2026, 2); // 91 * 24 = 2184
  assert.ok(Math.abs((camdUtilizationPct(hours, 1, 2026, 2) ?? 0) - 1) < 1e-9, "one unit operating every possible hour = 100%");
  assert.ok(Math.abs((camdUtilizationPct(hours * 2, 2, 2026, 2) ?? 0) - 1) < 1e-9, "two units, same per-unit rate, still normalizes to 100%");
  assert.equal(camdUtilizationPct(null, 3, 2026, 2), null, "no sumOpTime — no fabricated pct");
  assert.equal(camdUtilizationPct(100, null, 2026, 2), null, "no unitCount — no fabricated pct");
  assert.equal(camdUtilizationPct(100, 0, 2026, 2), null, "zero units — never divide by zero");
});

test("camdUtilizationColor: low-to-high blue-to-red bands, null tints as the lowest band", () => {
  assert.equal(camdUtilizationColor(null), "#4d9fff");
  assert.equal(camdUtilizationColor(0.1), "#4d9fff");
  assert.equal(camdUtilizationColor(0.25), "#8bc34a");
  assert.equal(camdUtilizationColor(0.49), "#8bc34a");
  assert.equal(camdUtilizationColor(0.5), "#ffd23f");
  assert.equal(camdUtilizationColor(0.74), "#ffd23f");
  assert.equal(camdUtilizationColor(0.75), "#ff3b3b");
  assert.equal(camdUtilizationColor(1.4), "#ff3b3b", "readings above 100% (data quirks) still clamp to the top band, not crash");
});

test("volcanoAlertColor: USGS's own color_code vocabulary maps directly, unrecognized/missing falls back to neutral gray", () => {
  assert.equal(volcanoAlertColor("RED"), "#ff3b3b");
  assert.equal(volcanoAlertColor("ORANGE"), "#ff8c42");
  assert.equal(volcanoAlertColor("YELLOW"), "#ffd23f");
  assert.equal(volcanoAlertColor("GREEN"), "#8bc34a");
  assert.equal(volcanoAlertColor("orange"), "#ff8c42", "case-insensitive — feed values are uppercase but never assumed");
  assert.equal(volcanoAlertColor(null), "#9aa5b1");
  assert.equal(volcanoAlertColor(undefined), "#9aa5b1");
  assert.equal(volcanoAlertColor("UNKNOWN"), "#9aa5b1", "never guesses a severity color for an unrecognized code");
});

test("ironOreStatusColor: GEM's 7 lowercase Operating status buckets map directly, unrecognized/missing falls back to gray", () => {
  assert.equal(ironOreStatusColor("operating"), "#4ade80");
  assert.equal(ironOreStatusColor("proposed"), "#a78bfa");
  assert.equal(ironOreStatusColor("mothballed"), "#78716c");
  assert.equal(ironOreStatusColor("retired"), "#64748b");
  assert.equal(ironOreStatusColor("shelved"), "#fbbf24");
  assert.equal(ironOreStatusColor("cancelled"), "#f87171");
  assert.equal(ironOreStatusColor("unknown"), "#94a3b8");
  assert.equal(ironOreStatusColor(null), "#94a3b8");
  assert.equal(ironOreStatusColor(undefined), "#94a3b8");
  assert.equal(ironOreStatusColor("not-a-real-bucket"), "#94a3b8", "never guesses a lifecycle color for an unrecognized status");
});

test("ironSteelTechColor: GEM's 5 production-technology buckets map directly, unrecognized/missing falls back to the 'other' gray", () => {
  assert.equal(ironSteelTechColor("bf_bof"), "#f97316");
  assert.equal(ironSteelTechColor("dri"), "#22d3ee");
  assert.equal(ironSteelTechColor("eaf"), "#4ade80");
  assert.equal(ironSteelTechColor("if"), "#a78bfa");
  assert.equal(ironSteelTechColor("other"), "#94a3b8");
  assert.equal(ironSteelTechColor(null), "#94a3b8");
  assert.equal(ironSteelTechColor(undefined), "#94a3b8");
  assert.equal(ironSteelTechColor("not-a-real-bucket"), "#94a3b8", "never guesses a technology color for an unrecognized bucket");
});

test("chemicalFeedstockColor: GEM chemicals' 6 feedstock-family buckets map directly, unrecognized/missing falls back to the 'other' gray", () => {
  assert.equal(chemicalFeedstockColor("coal"), "#f87171");
  assert.equal(chemicalFeedstockColor("natural_gas"), "#fbbf24");
  assert.equal(chemicalFeedstockColor("petroleum"), "#f97316");
  assert.equal(chemicalFeedstockColor("ngl"), "#a78bfa");
  assert.equal(chemicalFeedstockColor("low_carbon"), "#4ade80");
  assert.equal(chemicalFeedstockColor("other"), "#94a3b8");
  assert.equal(chemicalFeedstockColor(null), "#94a3b8");
  assert.equal(chemicalFeedstockColor(undefined), "#94a3b8");
  assert.equal(chemicalFeedstockColor("not-a-real-bucket"), "#94a3b8", "never guesses a feedstock color for an unrecognized bucket");
});
