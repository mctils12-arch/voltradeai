// 2026-11-24 (KNOWN BROKEN #60 recurrence; open_questions (al3)): the day's
// record of full scans, read for the 300 s budget. See scanTimingHistory.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  HISTORY_ROWS_SERVED, SCAN_TIMEOUT_SEC, SCAN_TIMING_LOG_DIR, parseHistoryDay, parseScanTimingLines,
  readScanTimingHistory, scanTimingDataDirs, spread, summarizeScanDay, type ScanTimingRow,
} from "./scanTimingHistory";

function row(over: Partial<ScanTimingRow>): ScanTimingRow {
  return {
    scan_started: 1795543786, day: "2026-11-24", status: "completed", total_sec: 260, reached_sec: 260,
    last_phase: "scan_complete", options_stage_end_sec: 241, tier_engine_start_sec: 258, after_options_sec: 19,
    options: { deadline_sec: 240, candidates: 400, not_reached: 30, loop_sec: 160 }, options_outside_loop_sec: 38, tier: { total_sec: 2.5 },
    layer2: { cache_hit: true, elapsed_sec: 0 }, phase_sec: {}, ...over,
  };
}

test("spread is nearest-rank over the finite values, null when there are none", () => {
  assert.equal(spread([]), null);
  assert.equal(spread([NaN]), null);
  assert.deepEqual(spread([5]), { n: 1, min: 5, median: 5, p90: 5, max: 5 });
  assert.deepEqual(spread([10, 1, 2, 3, 4, 5, 6, 7, 8, 9]), { n: 10, min: 1, median: 5, p90: 9, max: 10 });
});

test("a day's summary: scans past the timeout are the late finishers plus every scan that never finished", () => {
  const rows = [
    row({ total_sec: 250, after_options_sec: 10 }),
    row({ total_sec: 262, after_options_sec: 21 }),
    row({ total_sec: 296, after_options_sec: 55, layer2: { cache_hit: false, elapsed_sec: 33 } }),
    row({ total_sec: 312.4, after_options_sec: 71, layer2: { cache_hit: false, elapsed_sec: 44 } }),
    row({ status: "unfinished", total_sec: null, reached_sec: 259, last_phase: "tier_engine_start", after_options_sec: 18, layer2: {}, tier: {} }),
    row({ total_sec: 120, after_options_sec: 9, options_stage_end_sec: 111, options: { deadline_sec: 240, candidates: 220, not_reached: 0 } }),
  ];
  const s = summarizeScanDay("2026-11-24", rows, 1);
  assert.equal(s.scans, 6);
  assert.equal(s.completed, 5);
  assert.equal(s.unfinished, 1);
  assert.equal(s.past_timeout, 2);
  assert.deepEqual(s.past_timeout_last_phases, { "finished late": 1, tier_engine_start: 1 });
  assert.equal(s.deadline_bound, 5);                     // the short scan read every candidate
  assert.equal(s.not_reached_when_bound?.median, 30);
  assert.deepEqual(s.total_sec, { n: 5, min: 120, median: 262, p90: 312.4, max: 312.4 });
  // what follows the options stage, completed scans only, split by whether the CSP scores were recomputed
  assert.equal(s.after_options_sec.all?.n, 5);
  assert.deepEqual(s.after_options_sec.scores_recomputed, { n: 2, min: 55, median: 55, p90: 71, max: 71 });
  assert.deepEqual(s.after_options_sec.scores_cached, { n: 3, min: 9, median: 10, p90: 21, max: 21 });
  assert.deepEqual(s.options_outside_loop_sec, { n: 6, min: 38, median: 38, p90: 38, max: 38 });
  assert.equal(s.malformed_lines, 1);
  assert.equal(SCAN_TIMEOUT_SEC, 300);
});

test("a completed scan at exactly the timeout counts as past it; an empty day is zeros and nulls", () => {
  assert.equal(summarizeScanDay("d", [row({ total_sec: 300 })]).past_timeout, 1);
  assert.equal(summarizeScanDay("d", [row({ total_sec: 299.99 })]).past_timeout, 0);
  const empty = summarizeScanDay("d", []);
  assert.equal(empty.scans, 0);
  assert.equal(empty.total_sec, null);
  assert.equal(empty.after_options_sec.all, null);
});

test("lines that are not a scan row are counted, never guessed at", () => {
  const text = [
    JSON.stringify(row({})),
    '{"scan_started": 17955',                                  // cut mid-write
    JSON.stringify({ status: "completed" }),                   // no start time
    JSON.stringify({ scan_started: 1, status: "running" }),    // a status the writer never writes
    "",
    JSON.stringify({ scan_started: 2, status: "unfinished", last_phase: "deep_score" }),
  ].join("\n");
  const { rows, malformed } = parseScanTimingLines(text);
  assert.equal(rows.length, 2);
  assert.equal(malformed, 3);
  assert.deepEqual(rows[1].options, {});                       // absent blocks read as empty, so the summary cannot throw
  assert.equal(summarizeScanDay("d", rows).unfinished, 1);
});

test("the day is a date or nothing: it becomes part of a file name", () => {
  assert.equal(parseHistoryDay("2026-11-24"), "2026-11-24");
  for (const bad of ["", "2026-11-2", "../../etc/passwd", "2026-11-24.jsonl", 20261124, undefined, "2026-11-24/x"]) {
    assert.equal(parseHistoryDay(bad), null);
  }
  assert.deepEqual(scanTimingDataDirs({ DATA_DIR: "/x" }), ["/x"]);
  assert.deepEqual(scanTimingDataDirs({}), ["/data/voltrade", "/tmp"]);
});

test("reading a day: newest day by default, the summary over every row, the newest rows with the total beside them", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "scan-timing-"));
  try {
    assert.deepEqual(readScanTimingHistory([dir], null), { day: "", found: false, days_held: [], summary: null, rows_total: 0, rows: [] });
    const logDir = path.join(dir, SCAN_TIMING_LOG_DIR);
    fs.mkdirSync(logDir);
    fs.writeFileSync(path.join(logDir, "2026-11-23.jsonl"), JSON.stringify(row({ day: "2026-11-23", total_sec: 301 })) + "\n");
    const many = Array.from({ length: HISTORY_ROWS_SERVED + 15 }, (_, i) => JSON.stringify(row({ scan_started: 1795543786 + i, total_sec: 200 + i })));
    fs.writeFileSync(path.join(logDir, "2026-11-24.jsonl"), many.join("\n") + "\n");
    fs.writeFileSync(path.join(logDir, "readme.txt"), "not a day file");

    const latest = readScanTimingHistory([path.join(dir, "absent"), dir], null);
    assert.equal(latest.day, "2026-11-24");
    assert.deepEqual(latest.days_held, ["2026-11-23", "2026-11-24"]);
    assert.equal(latest.rows_total, HISTORY_ROWS_SERVED + 15);
    assert.equal(latest.rows.length, HISTORY_ROWS_SERVED);          // cut, and rows_total says so
    assert.equal(latest.rows[latest.rows.length - 1].total_sec, 200 + HISTORY_ROWS_SERVED + 14);
    assert.equal(latest.summary?.scans, HISTORY_ROWS_SERVED + 15);  // the summary is over every row, not the rows served

    const monday = readScanTimingHistory([dir], "2026-11-23");
    assert.equal(monday.summary?.past_timeout, 1);
    const none = readScanTimingHistory([dir], "2026-11-20");
    assert.equal(none.found, false);
    assert.deepEqual(none.days_held, ["2026-11-23", "2026-11-24"]);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("the timings probe serves the history only when asked, and refuses a day that is not a date", () => {
  const src = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "bot.ts"), "utf8");
  const at = src.indexOf('case "timings": {');
  assert.ok(at > 0);
  const block = src.slice(at, at + 3000);
  assert.match(block, /req\.query\.history !== undefined/);
  assert.match(block, /readScanTimingHistory\(scanTimingDataDirs\(\), day\)/);
  assert.match(block, /status\(400\)\.json\(\{ probe: "timings", error: "day must be YYYY-MM-DD" \}\)/);
  assert.ok(block.indexOf("req.query.history") < block.indexOf("const timingPaths"), "without ?history the latest scan is served as before");
});
