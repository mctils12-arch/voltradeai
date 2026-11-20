import { test } from "node:test";
import assert from "node:assert/strict";
import { mwLabel, fuelLine, endsLabel, ageLabel, signedPct, weeksBehind } from "./dataPanelsView";

test("capacity: the server's 0 means the database states none, so it never reads '0 MW'", () => {
  assert.equal(mwLabel(0), "not stated");
  assert.equal(mwLabel(null), "not stated");
  assert.equal(mwLabel(Number.NaN), "not stated");
  assert.equal(mwLabel(9.6), "9.6 MW");
  assert.equal(mwLabel(3241.2000000000003), "3,241 MW");
});

test("fuel line: largest first, unstated fuels dropped, empty says so", () => {
  // Live shape, 2026-11-20 (Brevard FL flood warning).
  assert.equal(fuelLine({ solar: 84.5, gas: 3241.2000000000003, other: 9.6 }), "gas 3,241 MW · solar 84.5 MW · other 9.6 MW");
  assert.equal(fuelLine({ wind: 0 }), "no capacity stated");
  assert.equal(fuelLine(null), "no capacity stated");
});

test("warning end: future, past, and absent are three different sentences", () => {
  const now = Date.parse("2026-11-20T12:00:00Z");
  assert.equal(endsLabel("2026-11-20T08:30:00-05:00", now), "ends in 1 h 30 min");
  assert.equal(endsLabel("2026-11-20T11:15:00Z", now), "end time passed 45 min ago");
  assert.equal(endsLabel(null, now), "no end time stated");
  assert.equal(endsLabel("not a date", now), "no end time stated");
  assert.equal(endsLabel("2026-11-24T12:00:00Z", now), "ends in 4 d");
});

test("age of the join is stated, or stated as unknown", () => {
  const now = 1795180000000;
  assert.equal(ageLabel(now - 20000, now), "under a minute ago");
  assert.equal(ageLabel(now - 9 * 60000, now), "9 min ago");
  assert.equal(ageLabel(undefined, now), "age unknown");
});

test("week-over-week: signed, and a missing prior week is a dash, not 0%", () => {
  assert.equal(signedPct(0.3), "+0.3%");
  assert.equal(signedPct(-14), "-14.0%");
  assert.equal(signedPct(0), "0.0%");
  assert.equal(signedPct(null), "—");
});

test("rail archive lag in whole weeks", () => {
  const now = Date.parse("2026-11-20T13:00:00Z");
  assert.equal(weeksBehind("2026-10-28", now), 3);
  assert.equal(weeksBehind("2026-11-18", now), 0);
  assert.equal(weeksBehind(null, now), null);
  assert.equal(weeksBehind("garbage", now), null);
});
