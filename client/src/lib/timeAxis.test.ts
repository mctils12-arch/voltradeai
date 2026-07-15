// Global time axis — EARTH TWIN E1 slice 1. Pins the store semantics and
// the honest date-clamping for dated (GIBS) followers.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  getTimeAxis,
  setTimeAxis,
  subscribeTimeAxis,
  gibsDateForAxis,
} from "./timeAxis.js";
import { gibsDefaultDate } from "./gibs.js";

const NOW = Date.UTC(2026, 6, 15, 12, 0, 0); // 2026-07-15T12:00Z

test("store: set notifies subscribers, identical state is a no-op, unsubscribe works", () => {
  setTimeAxis({ mode: "live" });
  let calls = 0;
  const un = subscribeTimeAxis(() => { calls++; });
  setTimeAxis({ mode: "historical", atMs: NOW - 86400_000 });
  assert.equal(calls, 1);
  assert.deepEqual(getTimeAxis(), { mode: "historical", atMs: NOW - 86400_000 });
  setTimeAxis({ mode: "historical", atMs: NOW - 86400_000 });
  assert.equal(calls, 1, "identical historical state must not re-notify");
  setTimeAxis({ mode: "live" });
  assert.equal(calls, 2);
  setTimeAxis({ mode: "live" });
  assert.equal(calls, 2, "identical live state must not re-notify");
  un();
  setTimeAxis({ mode: "historical", atMs: NOW });
  assert.equal(calls, 2, "unsubscribed listeners stay silent");
  setTimeAxis({ mode: "live" }); // reset for other tests
});

test("gibsDateForAxis: LIVE returns the layer's own latency-aware default", () => {
  const d1 = gibsDateForAxis({ mode: "live" }, NOW, 1);
  assert.equal(d1.dateISO, gibsDefaultDate(NOW, 1));
  assert.equal(d1.followingAxis, false);
  const d7 = gibsDateForAxis({ mode: "live" }, NOW, 7);
  assert.equal(d7.dateISO, gibsDefaultDate(NOW, 7), "high-latency layers keep their honest default");
});

test("gibsDateForAxis: HISTORICAL follows the axis date exactly when the layer has it", () => {
  const tenDaysAgo = NOW - 10 * 86400_000;
  const d = gibsDateForAxis({ mode: "historical", atMs: tenDaysAgo }, NOW, 1);
  assert.equal(d.dateISO, new Date(tenDaysAgo).toISOString().slice(0, 10));
  assert.equal(d.followingAxis, true);
  assert.equal(d.snapped, false);
});

test("gibsDateForAxis: HISTORICAL newer than the layer's ceiling snaps back honestly", () => {
  // axis at 2h ago — a daily product has no same-day data, a 7-day-latency
  // product is further behind; both must clamp to THEIR ceiling and say so
  const twoHoursAgo = NOW - 2 * 3600_000;
  const daily = gibsDateForAxis({ mode: "historical", atMs: twoHoursAgo }, NOW, 1);
  assert.equal(daily.dateISO, gibsDefaultDate(NOW, 1));
  assert.equal(daily.snapped, true, "the snap is reported — the panel must be able to say 'nearest available'");
  const smap = gibsDateForAxis({ mode: "historical", atMs: NOW - 3 * 86400_000 }, NOW, 7);
  assert.equal(smap.dateISO, gibsDefaultDate(NOW, 7), "3 days ago is still inside SMAP's ~7-day lag → its ceiling");
  assert.equal(smap.snapped, true);
});
