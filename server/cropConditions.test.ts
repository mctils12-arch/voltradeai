// NASS crop conditions battery (DATACORE MAXIMUS Phase 0b): key gate,
// documented-shape parse (Value/value, comma numbers), condition-class
// verbatim short_desc, event-identity dedup, key-never-logged url pin.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  cropConditionsEnabled, parseConditions, conditionsUrl, fetchConditions,
  archiveConditions, refreshConditions, latestConditions, COMMODITIES,
  readArchivedConditions, classFromItem, lookupCropConditionHistory,
  readConditionsAggregateHistory,
} from "./cropConditions";

// Documented QuickStats row shape (live verification pending — the key is
// Railway-only; the censusImports precedent applies).
const ROW = (week: string, item: string, value: string) => ({
  week_ending: week, short_desc: item, Value: value,
  commodity_desc: "CORN", statisticcat_desc: "CONDITION",
  agg_level_desc: "NATIONAL", unit_desc: "PCT",
});

test("key gate: disabled without NASS_API_KEY; zero network calls keyless", async () => {
  assert.equal(cropConditionsEnabled({} as any), false);
  assert.equal(cropConditionsEnabled({ NASS_API_KEY: "x" } as any), true);
  let called = 0;
  const spy = async () => { called++; return { ok: true, status: 200, text: async () => "{}" }; };
  assert.deepEqual(await fetchConditions(spy as any, {} as any, 0, 0), []);
  assert.equal(called, 0);
});

test("url: national weekly CONDITION query, key URL-encoded", () => {
  const u = conditionsUrl("CORN", 2026, "se cret");
  assert.ok(u.includes("commodity_desc=CORN"));
  assert.ok(u.includes("statisticcat_desc=CONDITION"));
  assert.ok(u.includes("agg_level_desc=NATIONAL"));
  assert.ok(u.includes("year=2026"));
  assert.ok(u.includes("key=se+cret") || u.includes("key=se%20cret"), "key encoded");
});

test("parse: comma numbers stripped, Value/value tolerated, bad rows dropped", () => {
  const obs = parseConditions({ data: [
    ROW("2026-06-29", "CORN - CONDITION, MEASURED IN PCT EXCELLENT", "14"),
    ROW("2026-06-29", "CORN - CONDITION, MEASURED IN PCT GOOD", "1,234"),  // comma-grouped
    { week_ending: "2026-06-29", short_desc: "CORN - CONDITION, MEASURED IN PCT FAIR", value: "22" },
    { week_ending: "bad", short_desc: "x", Value: "1" },
    { week_ending: "2026-06-29", Value: "1" },  // no short_desc
  ] }, "CORN", "2026-07-06");
  assert.equal(obs.length, 3);
  assert.equal(obs[0].pct, 14);
  assert.equal(obs[1].pct, 1234);
  assert.equal(obs[2].pct, 22, "lowercase value key tolerated");
  assert.ok(obs[0].item.includes("EXCELLENT"), "condition class travels verbatim in short_desc");
  assert.deepEqual(parseConditions(null, "CORN", "x"), []);
});

test("archive: commodity|week|item dedup across fetches and restarts", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "crop-"));
  const now = Date.parse("2026-07-06T12:00:00Z");
  const obs = parseConditions({ data: [
    ROW("2026-06-29", "CORN - CONDITION, MEASURED IN PCT EXCELLENT", "14"),
    ROW("2026-06-29", "CORN - CONDITION, MEASURED IN PCT GOOD", "45"),
  ] }, "CORN", "2026-07-06");
  assert.equal(archiveConditions(obs, base, now), 2);
  assert.equal(archiveConditions(obs, base, now), 0, "same week/class never re-archives");
  const plus = [...obs, ...parseConditions({ data: [ROW("2026-07-06", "CORN - CONDITION, MEASURED IN PCT EXCELLENT", "13")] }, "CORN", "2026-07-06")];
  assert.equal(archiveConditions(plus, base, now), 1, "new week's row lands");
});

test("refresh: one call per commodity; cache holds the NEWEST week only", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "crop-"));
  let calls = 0;
  const ok = async () => {
    calls++;
    return { ok: true, status: 200, text: async () => JSON.stringify({ data: [
      ROW("2026-06-29", "X - CONDITION, MEASURED IN PCT GOOD", "40"),
      ROW("2026-07-06", "X - CONDITION, MEASURED IN PCT GOOD", "42"),
    ] }) };
  };
  await refreshConditions(ok as any, { NASS_API_KEY: "k" } as any, Date.parse("2026-07-06T22:00:00Z"), base, 0);
  assert.equal(calls, COMMODITIES.length);
  const hit = latestConditions();
  assert.ok(hit);
  assert.equal(hit!.latest_week, "2026-07-06");
  assert.ok(hit!.rows.every((r) => r.week_ending === "2026-07-06"), "cache = newest week only");
});

// NOTE: archiveConditions's dedup Set is module-level (shared across every
// test in this process, regardless of tmpdir — same quirk documented in
// githubOrgActivity.test.ts), so the tests below use week_ending dates no
// other test in this file touches.

test("classFromItem: extracts the condition class verbatim from short_desc, null for anything that doesn't match", () => {
  assert.equal(classFromItem("CORN - CONDITION, MEASURED IN PCT EXCELLENT"), "EXCELLENT");
  assert.equal(classFromItem("SOYBEANS - CONDITION, MEASURED IN PCT VERY POOR"), "VERY POOR");
  assert.equal(classFromItem("soybeans - condition, measured in pct fair"), "FAIR", "case-insensitive");
  assert.equal(classFromItem("CORN - PLANTED, MEASURED IN PCT COMPLETE"), null, "a non-CONDITION item has no condition class");
});

test("readArchivedConditions: scans every day-file (not just the newest), dedups by commodity|week|item", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "crop-hist-"));
  const week1 = { commodity: "CORN", week_ending: "2027-01-04", item: "CORN - CONDITION, MEASURED IN PCT GOOD", pct: 40, rt: "2027-01-05" };
  const week2 = { commodity: "CORN", week_ending: "2027-01-11", item: "CORN - CONDITION, MEASURED IN PCT GOOD", pct: 42, rt: "2027-01-12" };
  // Written on two different days -> two different day-files, same dir.
  archiveConditions([week1], dir, Date.parse("2027-01-05T20:00:00Z"));
  archiveConditions([week2], dir, Date.parse("2027-01-12T20:00:00Z"));
  const all = readArchivedConditions(dir);
  assert.equal(all.length, 2, "must read across both day-files, not just the latest");
  assert.deepEqual(new Set(all.map((o) => o.week_ending)), new Set(["2027-01-04", "2027-01-11"]));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("lookupCropConditionHistory: one commodity's rows across weeks, ascending, the other commodity excluded, unfetched weeks never zero-filled", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "crop-hist-"));
  const t0 = Date.parse("2027-02-09T20:00:00Z");
  archiveConditions([
    { commodity: "CORN", week_ending: "2027-02-01", item: "CORN - CONDITION, MEASURED IN PCT GOOD", pct: 40, rt: "2027-02-09" },
    { commodity: "CORN", week_ending: "2027-02-08", item: "CORN - CONDITION, MEASURED IN PCT GOOD", pct: 44, rt: "2027-02-09" },
    { commodity: "SOYBEANS", week_ending: "2027-02-08", item: "SOYBEANS - CONDITION, MEASURED IN PCT GOOD", pct: 50, rt: "2027-02-09" },
  ], dir, t0);
  const series = lookupCropConditionHistory("corn", 10, dir);
  assert.equal(series.length, 2, "only CORN's own rows, SOYBEANS excluded");
  assert.deepEqual(series.map((o) => o.week_ending), ["2027-02-01", "2027-02-08"], "ascending by week_ending");
  assert.equal(series[1].pct, 44);
  const capped = lookupCropConditionHistory("CORN", 1, dir);
  assert.equal(capped.length, 1, "weeks param caps the series to the most recent N distinct weeks");
  assert.equal(capped[0].week_ending, "2027-02-08");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("readConditionsAggregateHistory: pivots both commodities to {class: pct} per week, ascending, an unmatched item is skipped rather than mis-keyed", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "crop-hist-"));
  const t0 = Date.parse("2027-03-02T20:00:00Z");
  archiveConditions([
    { commodity: "CORN", week_ending: "2027-03-01", item: "CORN - CONDITION, MEASURED IN PCT EXCELLENT", pct: 14, rt: "2027-03-02" },
    { commodity: "CORN", week_ending: "2027-03-01", item: "CORN - CONDITION, MEASURED IN PCT GOOD", pct: 45, rt: "2027-03-02" },
    { commodity: "SOYBEANS", week_ending: "2027-03-01", item: "SOYBEANS - CONDITION, MEASURED IN PCT FAIR", pct: 22, rt: "2027-03-02" },
    { commodity: "CORN", week_ending: "2027-03-01", item: "CORN - PLANTED, MEASURED IN PCT COMPLETE", pct: 99, rt: "2027-03-02" },
  ], dir, t0);
  const trend = readConditionsAggregateHistory(10, dir);
  assert.equal(trend.length, 1);
  assert.equal(trend[0].week_ending, "2027-03-01");
  assert.deepEqual(trend[0].corn, { EXCELLENT: 14, GOOD: 45 }, "the non-CONDITION PLANTED row must not appear under any class key");
  assert.deepEqual(trend[0].soybeans, { FAIR: 22 });
  fs.rmSync(dir, { recursive: true, force: true });
});
