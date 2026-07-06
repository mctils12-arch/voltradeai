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
