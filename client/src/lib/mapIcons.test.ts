// EPA CAMD utilization helpers (datamap.tsx "plant_operations" layer,
// server/epaCamd.ts's ground-truth stream). Run:
// npx tsx --test client/src/lib/mapIcons.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { camdQuarterHours, camdUtilizationPct, camdUtilizationColor } from "./mapIcons.ts";

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
