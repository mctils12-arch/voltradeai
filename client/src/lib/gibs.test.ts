import { test } from "node:test";
import assert from "node:assert/strict";
import { gibsTileUrl, gibsDefaultDate, gibsStepDate, gibsIsLatestAvailable } from "./gibs";

test("gibsTileUrl: builds the verified WMTS REST template with z/y/x tokens intact", () => {
  const url = gibsTileUrl(
    { layer: "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance", tileMatrixSet: "GoogleMapsCompatible_Level8", ext: "png" },
    "2026-07-07",
  );
  assert.equal(
    url,
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_DayNightBand_At_Sensor_Radiance/default/2026-07-07/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png",
  );
});

test("gibsDefaultDate: yesterday in UTC, not local time", () => {
  // 2026-07-08T00:30Z -> yesterday is 2026-07-07 regardless of the
  // container's local timezone (charter: default the scrubber to yesterday).
  const nowMs = Date.parse("2026-07-08T00:30:00Z");
  assert.equal(gibsDefaultDate(nowMs), "2026-07-07");
});

test("gibsDefaultDate: month/year boundary", () => {
  assert.equal(gibsDefaultDate(Date.parse("2026-01-01T12:00:00Z")), "2025-12-31");
});

test("gibsStepDate: steps forward and backward across a month boundary", () => {
  assert.equal(gibsStepDate("2026-07-01", -1), "2026-06-30");
  assert.equal(gibsStepDate("2026-06-30", 1), "2026-07-01");
  assert.equal(gibsStepDate("2026-07-15", 3), "2026-07-18");
});

test("gibsIsLatestAvailable: true only once the date reaches the honest 'yesterday' ceiling", () => {
  const nowMs = Date.parse("2026-07-08T15:00:00Z"); // "yesterday" = 2026-07-07
  assert.equal(gibsIsLatestAvailable("2026-07-06", nowMs), false);
  assert.equal(gibsIsLatestAvailable("2026-07-07", nowMs), true);
  // never claims "today" or a future date is behind the ceiling either
  assert.equal(gibsIsLatestAvailable("2026-07-08", nowMs), true);
});

test("gibsDefaultDate: latencyDays steps the default back by a layer's real lag (SMAP ~6d)", () => {
  const nowMs = Date.parse("2026-07-08T15:00:00Z");
  assert.equal(gibsDefaultDate(nowMs), "2026-07-07");        // default = yesterday
  assert.equal(gibsDefaultDate(nowMs, 1), "2026-07-07");     // explicit 1 = same
  assert.equal(gibsDefaultDate(nowMs, 7), "2026-07-01");     // 7-day-lag layer
  // never returns "today" even if given a nonsense sub-1 latency
  assert.equal(gibsDefaultDate(nowMs, 0), "2026-07-07");
});

test("gibsIsLatestAvailable: ceiling honors the same latencyDays as the default", () => {
  const nowMs = Date.parse("2026-07-08T15:00:00Z"); // 7-day ceiling = 2026-07-01
  assert.equal(gibsIsLatestAvailable("2026-06-30", nowMs, 7), false);
  assert.equal(gibsIsLatestAvailable("2026-07-01", nowMs, 7), true);
  assert.equal(gibsIsLatestAvailable("2026-07-05", nowMs, 7), true);
});
