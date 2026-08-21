import { test } from "node:test";
import assert from "node:assert/strict";
import {
  gdeltDayToIso, addDaysIso, dayRange, nearbyDaysByFacility, hitWithinWindow,
  buildDayRows, verdictFromRows, computeGate2, GATE2_RADIUS_KM, GATE2_WINDOW_DAYS, GATE2_MIN_UNREST_DAYS,
  type Facility, type DayRow,
} from "./gdelt_fires_gate2";

test("gdeltDayToIso: normalizes GDELT's YYYYMMDD column to YYYY-MM-DD; rejects garbage", () => {
  assert.equal(gdeltDayToIso("20260815"), "2026-08-15");
  assert.equal(gdeltDayToIso("2026-08-15"), null);
  assert.equal(gdeltDayToIso(""), null);
  assert.equal(gdeltDayToIso("abcdefgh"), null);
});

test("addDaysIso / dayRange: basic date arithmetic, including a month boundary", () => {
  assert.equal(addDaysIso("2026-07-30", 3), "2026-08-02");
  assert.equal(addDaysIso("2026-07-05", 0), "2026-07-05");
  assert.deepEqual(dayRange("2026-07-30", "2026-08-01"), ["2026-07-30", "2026-07-31", "2026-08-01"]);
  assert.deepEqual(dayRange("2026-07-05", "2026-07-05"), ["2026-07-05"]);
});

const FAC: Facility[] = [
  { id: "cushing_hub", name: "Cushing", lat: 35.9487, lon: -96.7587, category: "tank_farm" },
  { id: "port_la", name: "Port of LA", lat: 33.7395, lon: -118.2610, category: "port" },
];

test("nearbyDaysByFacility: keeps a row within radiusKm, drops one just outside, ignores rows with no day/non-finite coords", () => {
  const rows = [
    { lat: 35.95, lon: -96.75, day: "2026-08-01" },   // ~1km from cushing_hub — in
    { lat: 36.5, lon: -96.7587, day: "2026-08-02" },  // ~61km north of cushing_hub — out at 25km
    { lat: 33.73, lon: -118.25, day: "2026-08-01" },  // near port_la — in
    { lat: NaN, lon: -96.76, day: "2026-08-03" },     // non-finite — dropped
    { lat: 35.95, lon: -96.76, day: "" },              // no day — dropped
  ];
  const byFac = nearbyDaysByFacility(rows, FAC, 25);
  assert.deepEqual([...byFac.get("cushing_hub")!].sort(), ["2026-08-01"]);
  assert.deepEqual([...byFac.get("port_la")!].sort(), ["2026-08-01"]);
});

test("hitWithinWindow: day itself and day+windowDays both count; day+windowDays+1 does not", () => {
  const fireDays = new Set(["2026-08-05"]);
  assert.equal(hitWithinWindow(fireDays, "2026-08-05", 3), true, "same-day hit");
  assert.equal(hitWithinWindow(fireDays, "2026-08-02", 3), true, "3 days before, at the edge of the window");
  assert.equal(hitWithinWindow(fireDays, "2026-08-01", 3), false, "4 days before — outside the window");
  assert.equal(hitWithinWindow(new Set(), "2026-08-05", 3), false, "empty fireDays never hits");
});

test("buildDayRows: marks unrest/hit per facility-day independently (one facility's fire doesn't leak into another's row)", () => {
  const unrest = new Map([["cushing_hub", new Set(["2026-08-01"])], ["port_la", new Set<string>()]]);
  const fires = new Map([["cushing_hub", new Set(["2026-08-02"])], ["port_la", new Set(["2026-08-01"])]]);
  const rows = buildDayRows(FAC, unrest, fires, ["2026-08-01", "2026-08-02"], 3);
  const byKey = Object.fromEntries(rows.map((r) => [`${r.facility}:${r.day}`, r]));
  assert.equal(byKey["cushing_hub:2026-08-01"].unrest, true);
  assert.equal(byKey["cushing_hub:2026-08-01"].hit, true, "2026-08-02 fire is within the 3-day window of the 08-01 unrest day");
  assert.equal(byKey["port_la:2026-08-01"].unrest, false);
  assert.equal(byKey["port_la:2026-08-01"].hit, true, "port_la's own fire is independent of cushing_hub's unrest flag");
});

function row(facility: string, category: string, day: string, unrest: boolean, hit: boolean): DayRow {
  return { facility, category, day, unrest, hit };
}

test("verdictFromRows: insufficient_n below GATE2_MIN_UNREST_DAYS reports honestly instead of fabricating a p-value", () => {
  const rows = [
    row("a", "port", "d1", true, true),
    row("a", "port", "d2", false, false),
    row("a", "port", "d3", false, false),
  ];
  const v = verdictFromRows(rows);
  assert.equal(v.n, 1);
  assert.ok(v.n < GATE2_MIN_UNREST_DAYS);
  assert.equal(v.insufficient_n, true);
  assert.ok(Number.isNaN(v.p_value));
  assert.equal(v.elevated, false);
});

test("verdictFromRows: a real elevation (unrest days hit far more than the control rate) clears the bar", () => {
  const rows: DayRow[] = [];
  // 6 unrest days, all hit
  for (let i = 0; i < 6; i++) rows.push(row("a", "port", `u${i}`, true, true));
  // 60 control days, low hit rate (5/60 ≈ 8.3%)
  for (let i = 0; i < 60; i++) rows.push(row("a", "port", `c${i}`, false, i < 5));
  const v = verdictFromRows(rows);
  assert.equal(v.n, 6);
  assert.equal(v.k, 6);
  assert.equal(v.insufficient_n, false);
  assert.ok(v.p_value < 0.05, `expected p<0.05, got ${v.p_value}`);
  assert.equal(v.elevated, true);
});

test("verdictFromRows: unrest-day rate equal to (not above) control rate is never reported elevated, however low p looks", () => {
  const rows: DayRow[] = [];
  for (let i = 0; i < 10; i++) rows.push(row("a", "port", `u${i}`, true, i < 1)); // 1/10 = 10%
  for (let i = 0; i < 100; i++) rows.push(row("a", "port", `c${i}`, false, i < 10)); // 10/100 = 10%, same rate
  const v = verdictFromRows(rows);
  assert.equal(v.k / v.n, v.control_rate);
  assert.equal(v.elevated, false, "matching the control rate exactly is not evidence of elevation regardless of p");
});

test("computeGate2: per-category breakdown can diverge from the pooled verdict (Simpson's-paradox guard)", () => {
  const rows: DayRow[] = [];
  // steel_mill: high baseline heat signature AND unrest days both near-always hit — real but likely a
  // pre-existing-heat confound, not a genuine unrest->fire link (exactly the case the docstring predicts).
  for (let i = 0; i < 6; i++) rows.push(row("mill1", "steel_mill", `mu${i}`, true, true));
  for (let i = 0; i < 60; i++) rows.push(row("mill1", "steel_mill", `mc${i}`, false, i < 54)); // 90% control rate
  // port: unrest days never hit, control rate is also 0 — clean, uninformative null.
  for (let i = 0; i < 6; i++) rows.push(row("port1", "port", `pu${i}`, true, false));
  for (let i = 0; i < 60; i++) rows.push(row("port1", "port", `pc${i}`, false, false));
  const { pooled, by_category } = computeGate2(rows);
  assert.equal(by_category.steel_mill.insufficient_n, false);
  assert.equal(by_category.port.insufficient_n, false);
  assert.equal(by_category.port.k, 0);
  // pooled masks that the port category contributes nothing — the per-category view is the one that
  // actually tells you which facility class, if any, is driving the pooled number.
  assert.ok(pooled.n === by_category.steel_mill.n + by_category.port.n);
});

test("GATE2_RADIUS_KM/GATE2_WINDOW_DAYS/GATE2_MIN_UNREST_DAYS stay the pre-registered values documented in the module header", () => {
  assert.equal(GATE2_RADIUS_KM, 25);
  assert.equal(GATE2_WINDOW_DAYS, 3);
  assert.equal(GATE2_MIN_UNREST_DAYS, 5);
});
