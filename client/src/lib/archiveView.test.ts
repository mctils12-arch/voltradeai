import { test } from "node:test";
import assert from "node:assert/strict";
import { archiveRows, fileLabel, fmtBytes, newestAge, offsiteLines, type OffloadStatus } from "./archiveView";

const NOW = Date.parse("2026-11-20T18:10:00Z");

test("a file name's period: the day when it has one, the bare name otherwise, nothing for a marker file", () => {
  // every shape below is a live file name from /api/data/archive/stats, 2026-11-20
  assert.equal(fileLabel("2026-11-20-18.jsonl"), "2026-11-20");
  assert.equal(fileLabel("2026-10-30-00.jsonl.gz"), "2026-10-30");
  assert.equal(fileLabel("alerts-2026-07-29.jsonl.gz"), "2026-07-29");
  assert.equal(fileLabel("2026-06-08__T1.jsonl.gz"), "2026-06-08");
  assert.equal(fileLabel("A2605B/2026-11.jsonl"), "2026-11");
  assert.equal(fileLabel("TX_2024Q3.jsonl.gz"), "TX_2024Q3");
  assert.equal(fileLabel("202605a.jsonl.gz"), "202605a");
  assert.equal(fileLabel("2022.jsonl.gz"), "2022");
  assert.equal(fileLabel(".last_run_day"), null);
  assert.equal(fileLabel(null), null);
});

test("newest file's age is by the UTC day in its name; day-ahead files and names with no day are not given an age in days", () => {
  assert.equal(newestAge("2026-11-20-18.jsonl", NOW), "today");
  assert.equal(newestAge("2026-11-19.jsonl", NOW), "1 d ago");
  assert.equal(newestAge("2026-11-02.jsonl.gz", NOW), "18 d ago", "the fires archive on the day its stall was found");
  assert.equal(newestAge("2026-11-22.jsonl", NOW), "dated ahead", "day-ahead power prices");
  assert.equal(newestAge("2025q2.jsonl.gz", NOW), "");
  assert.equal(newestAge(null, NOW), "");
});

test("bytes: a missing figure is a dash, never 0 B", () => {
  assert.equal(fmtBytes(2012622084), "2.0 GB");
  assert.equal(fmtBytes(755590), "756 KB");
  assert.equal(fmtBytes(0), "0 B");
  assert.equal(fmtBytes(undefined), "—");
});

const OFFLOAD: OffloadStatus = {
  configured: true, retentionDays: 30, oldestReplayableDate: "2026-10-21", lastRun: "2026-11-20T18:01:14.017Z",
  lastRunSummary: { uploaded: 2, uploadFailed: 0, pending: 0, permanent: { uploaded: 0, uploadFailed: 0, pendingUploads: 0 } },
  lastError: null, lastErrorAt: null,
  tiers: { r2: { bytes: 1686760740, objects: 1386, byKind: { aircraft: { objects: 690, bytes: 318328602, oldestDate: "2026-10-21", newestDate: "2026-11-18" } } } },
  volume: { freeBytes: 1878728704, hotTierTargetFreeBytes: 2147483648, underPressure: true },
  permanentTier: {
    enabled: true, r2Objects: 1197, r2Bytes: 1541902410, conflicts: [], missing: [], lastReconcile: "2026-11-20T14:31:14.018Z",
    byStream: { aircraft_tracks: { r2Objects: 119, r2Bytes: 202020693, localFiles: 0, oldestDate: "2026-07-03", newestDate: "2026-10-29" } },
  },
};

test("rows: largest first, an archive with no file on the volume says what is off-site instead of looking lost", () => {
  const stats = { kinds: {
    aircraft: { files: 523, bytes: 272372657, oldest: "2026-10-30-00.jsonl.gz", newest: "2026-11-20-18.jsonl" },
    aircraft_tracks: { files: 0, bytes: 0, oldest: null, newest: null },
    fires: { files: 14, bytes: 15478920, oldest: "2026-10-20.jsonl.gz", newest: "2026-11-02.jsonl.gz" },
    optionchains: { files: 99, bytes: 96581068, oldest: ".last_run_day", newest: "2026-11-19.jsonl" },
  } };
  const rows = archiveRows(stats, OFFLOAD, NOW);
  assert.deepEqual(rows.map((r) => r.kind), ["aircraft", "optionchains", "fires", "aircraft_tracks"]);
  assert.deepEqual(rows[0], { kind: "aircraft", files: 523, bytes: 272372657, first: "2026-10-30", newest: "2026-11-20", age: "today", offsite: "690 files in the replay window, 2026-10-21 to 2026-11-18" });
  assert.equal(rows[1].first, "—", "a marker file is not a first day");
  assert.equal(rows[2].age, "18 d ago");
  assert.deepEqual(rows[3], { kind: "aircraft_tracks", files: 0, bytes: 0, first: "—", newest: "—", age: "", offsite: "119 day files kept for good, 2026-07-03 to 2026-10-29" });
  assert.deepEqual(archiveRows(null, null, NOW), []);
});

test("off-site copy: the live payload of 2026-11-20 in five sentences", () => {
  assert.deepEqual(offsiteLines(OFFLOAD, NOW), [
    "Last copy run 9 min ago: 2 files copied, 0 failed, 0 waiting.",
    "Held off-site: 3.2 GB in 2,583 files. 1.5 GB is day files that are never deleted; 1.7 GB is the last 30 days of hourly position logs, replayable from 2026-10-21.",
    "Compared with the off-site listing 3 h 39 min ago: 0 files differ from the copy, 0 missing off-site.",
    "The volume has 1.9 GB free; the server aims to keep 2.1 GB free. It is under that target, so days already verified off-site are removed from the volume first.",
  ]);
});

test("off-site copy: not set up, never run, and a failure are each said plainly", () => {
  assert.deepEqual(offsiteLines({ configured: false }, NOW), ["Off-site copy is not set up on this server: every archive file exists only on the volume."]);
  assert.deepEqual(offsiteLines(null, NOW), []);
  const cold = offsiteLines({ configured: true, lastRun: null, tiers: { r2: { bytes: 0, objects: 0 } } }, NOW);
  assert.deepEqual(cold, ["No copy run has finished since the server started.", "Nothing is held off-site yet."]);
  const bad = offsiteLines({ ...OFFLOAD, lastRunSummary: { uploaded: 0, uploadFailed: 3, pending: 5 }, lastError: "PUT 503", lastErrorAt: "2026-11-20T18:01:14Z", permanentTier: { ...OFFLOAD.permanentTier, conflicts: [{}], missing: [{}, {}] } }, NOW);
  assert.equal(bad[0], "Last copy run 9 min ago: 0 files copied, 3 failed, 5 waiting.");
  assert.equal(bad[2], "Compared with the off-site listing 3 h 39 min ago: 1 file differs from the copy, 2 missing off-site.");
  assert.equal(bad[4], "Last error 9 min ago: PUT 503");
});
