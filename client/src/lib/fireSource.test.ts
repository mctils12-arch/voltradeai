import { test } from "node:test";
import assert from "node:assert/strict";
import { fireSatelliteName, fireSourceName, firesEmptyNote, firesSourceNote, type FirePoll } from "./fireSource";

const NOW = Date.parse("2026-11-20T19:00:00Z");
const poll = (over: Partial<FirePoll>): FirePoll => ({
  last_attempt_at: "2026-11-20T18:48:00Z", last_rows_at: null, source_used: null, sources: [], polls_without_rows: 0, ...over,
});

test("satellite codes are NASA's; an unknown one is shown as it is, a missing one is said to be missing", () => {
  assert.equal(fireSatelliteName("N"), "Suomi NPP");
  assert.equal(fireSatelliteName("N20"), "NOAA-20");
  assert.equal(fireSatelliteName("N21"), "NOAA-21");
  assert.equal(fireSatelliteName("Aqua"), "Aqua");
  assert.equal(fireSatelliteName(""), "satellite not stated");
  assert.equal(fireSatelliteName("unknown"), "satellite not stated");
  assert.equal(fireSourceName("VIIRS_NOAA20_NRT"), "NOAA-20");
  assert.equal(fireSourceName("VIIRS_FUTURE_NRT"), "VIIRS_FUTURE_NRT");
});

test("before the first poll the layer is warming up; that sentence is not used once a poll has run", () => {
  assert.equal(firesEmptyNote({ warming_up: true }, NOW), "warming up — first poll can take a few minutes");
  assert.equal(firesEmptyNote({ warming_up: true, poll: poll({ last_attempt_at: null }) }, NOW), "warming up — first poll can take a few minutes");
  const d = {
    no_detections: true,
    poll: poll({
      polls_without_rows: 3,
      sources: [
        { source: "VIIRS_NOAA20_NRT", rows: 0, error: null },
        { source: "VIIRS_NOAA21_NRT", rows: null, error: "FIRMS 500" },
        { source: "VIIRS_SNPP_NRT", rows: 0, error: null },
      ],
    }),
  };
  assert.equal(
    firesEmptyNote(d, NOW),
    "NASA FIRMS returned no detections from any satellite (asked 12 min ago): NOAA-20 no detections · NOAA-21 request failed · Suomi NPP no detections",
  );
});

test("every request failing reads as NASA not answering, not as a world without fires", () => {
  const d = { no_detections: true, poll: poll({ sources: [{ source: "VIIRS_NOAA20_NRT", rows: null, error: "FIRMS 400" }] }) };
  assert.equal(firesEmptyNote(d, NOW), "NASA FIRMS did not answer (asked 12 min ago): NOAA-20 request failed");
});

test("the live note names the satellite, and says when the lead one was passed over", () => {
  assert.equal(firesSourceNote(poll({ source_used: "VIIRS_NOAA20_NRT", sources: [{ source: "VIIRS_NOAA20_NRT", rows: 75878, error: null }] })), "from NOAA-20");
  assert.equal(
    firesSourceNote(poll({
      source_used: "VIIRS_NOAA21_NRT",
      sources: [{ source: "VIIRS_NOAA20_NRT", rows: 0, error: null }, { source: "VIIRS_NOAA21_NRT", rows: 69911, error: null }],
    })),
    "from NOAA-21 (NOAA-20 had none)",
  );
  assert.equal(firesSourceNote(null), "");
  assert.equal(firesSourceNote(poll({})), "", "an older server sends no source: nothing is claimed");
});
