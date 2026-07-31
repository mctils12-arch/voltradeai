// EIA-930 gate-1 revision-stats battery: pure diff logic, no network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { computeRevisionStats, MATERIAL_REVISION_PCT } from "./gridDemandRevision";
import type { DemandObs } from "./gridDemand";

const O = (respondent: string, period: string, mwh: number | null, type: "D" | "DF" = "D"): DemandObs =>
  ({ respondent, period, type, mwh, rt: period.slice(0, 10) });

test("identical draws: zero revisions", () => {
  const draw = [O("US48", "2026-07-30T10", 500000), O("ERCO", "2026-07-30T10", 60000)];
  const at = Date.parse("2026-07-30T12:00:00Z");
  const r = computeRevisionStats(draw, at, draw, at + 60_000);
  assert.equal(r.compared, 2);
  assert.equal(r.revised, 0);
  assert.equal(r.revised_pct, 0);
  assert.equal(r.material_revised, 0);
  assert.equal(r.max_abs_diff_pct, 0);
  assert.equal(r.gap_minutes, 1);
});

test("small revision below MATERIAL_REVISION_PCT counted as revised but not material", () => {
  const at = Date.parse("2026-07-30T12:00:00Z");
  const draw1 = [O("US48", "2026-07-30T10", 500000)];
  const draw2 = [O("US48", "2026-07-30T10", 500500)]; // +0.1%
  const r = computeRevisionStats(draw1, at, draw2, at + 300_000);
  assert.equal(r.revised, 1);
  assert.equal(r.material_revised, 0, `0.1% < ${MATERIAL_REVISION_PCT}% threshold`);
  assert.ok(r.max_abs_diff_pct > 0 && r.max_abs_diff_pct < MATERIAL_REVISION_PCT);
});

test("large revision flagged material; sign preserved; worst-sorted by magnitude", () => {
  const at = Date.parse("2026-07-30T12:00:00Z");
  const draw1 = [O("US48", "2026-07-30T10", 500000), O("ERCO", "2026-07-30T10", 60000)];
  const draw2 = [O("US48", "2026-07-30T10", 520553), O("ERCO", "2026-07-30T10", 60006)]; // ~4.1% vs ~0.01%
  const r = computeRevisionStats(draw1, at, draw2, at + 600_000);
  assert.equal(r.compared, 2);
  assert.equal(r.revised, 2);
  assert.equal(r.material_revised, 1, "only the US48 cell clears the material threshold");
  assert.equal(r.worst[0].respondent, "US48", "biggest |diff_pct| sorts first");
  assert.ok(r.worst[0].diff_pct > 0, "revised UP");
});

test("cells present in only one draw are counted separately, never treated as a revision", () => {
  const at = Date.parse("2026-07-30T12:00:00Z");
  const draw1 = [O("US48", "2026-07-30T10", 500000), O("US48", "2026-07-30T09", 490000)];
  const draw2 = [O("US48", "2026-07-30T10", 500000), O("US48", "2026-07-30T11", 510000)];
  const r = computeRevisionStats(draw1, at, draw2, at + 60_000);
  assert.equal(r.compared, 1);
  assert.equal(r.revised, 0);
  assert.equal(r.only_in_draw1, 1, "T09 dropped out of draw2's window");
  assert.equal(r.only_in_draw2, 1, "T11 newly appeared in draw2's window");
});

test("null mwh cells are excluded from comparison entirely (never a false revision)", () => {
  const at = Date.parse("2026-07-30T12:00:00Z");
  const draw1 = [O("US48", "2026-07-30T10", null)];
  const draw2 = [O("US48", "2026-07-30T10", 500000)];
  const r = computeRevisionStats(draw1, at, draw2, at + 60_000);
  assert.equal(r.compared, 0);
  assert.equal(r.only_in_draw1, 0, "null-valued draw1 cell is never indexed into m1, so it can't show as only-in-draw1");
  assert.equal(r.only_in_draw2, 1, "draw2's non-null cell has no draw1 counterpart to match against");
});

test("D and DF for the same respondent+period are distinct cells", () => {
  const at = Date.parse("2026-07-30T12:00:00Z");
  const draw1 = [O("US48", "2026-07-30T10", 500000, "D"), O("US48", "2026-07-30T10", 480000, "DF")];
  const draw2 = [O("US48", "2026-07-30T10", 500000, "D"), O("US48", "2026-07-30T10", 495000, "DF")];
  const r = computeRevisionStats(draw1, at, draw2, at + 60_000);
  assert.equal(r.compared, 2);
  assert.equal(r.revised, 1);
  assert.equal(r.worst[0].type, "DF");
});

test("hours_old_at_draw1 reflects publication lag from the period timestamp", () => {
  const at = Date.parse("2026-07-30T12:00:00Z"); // 2h after the T10 period
  const draw1 = [O("US48", "2026-07-30T10", 500000)];
  const draw2 = [O("US48", "2026-07-30T10", 500500)];
  const r = computeRevisionStats(draw1, at, draw2, at + 60_000);
  assert.equal(r.worst[0].hours_old_at_draw1, 2);
});
