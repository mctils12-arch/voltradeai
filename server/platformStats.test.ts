// Platform stats — the hero's numbers must be REAL and self-updating.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "path";
import zlib from "zlib";
import { fileURLToPath } from "node:url";
import {
  streamsRecording, countFileLines, countObservations, platformStats, listArchiveDirs, isObservationFile,
  _resetPlatformStatsCache, _setLineCounterForTests, DIR_LAYER_MAP, OPERATIONAL_DIRS, PLATFORM_STATS_CACHE_FILE,
} from "./platformStats";
import { PRIVATE_ARCHIVE_DIRS } from "./datacoreArchive";

const here = path.dirname(fileURLToPath(import.meta.url));

test("streams: live layers + unmapped archive dirs; operational and _tracks never counted", () => {
  const live = ["aircraft", "vessels", "weather", "weather_temp", "insider", "earnings"];
  const dirs = ["aircraft", "filings", "earnings8k", "optionchains", "waitlist", "apiusage", "aircraft_tracks"];
  // filings->insider and earnings8k->earnings are already live layers;
  // optionchains has no layer -> +1; waitlist/apiusage/_tracks excluded
  assert.equal(streamsRecording(live, dirs), live.length + 1);
  assert.equal(DIR_LAYER_MAP.filings, "insider", "dir-name translation must be pinned");
});

test("operational dirs ARE the archive's private user-data dirs (one set, no drift)", () => {
  assert.equal(OPERATIONAL_DIRS, PRIVATE_ARCHIVE_DIRS);
  assert.deepEqual([...OPERATIONAL_DIRS].sort(), ["apiusage", "waitlist"]);
});

test("observations: real line counts, gz-aware, operational dirs excluded", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "pstats-"));
  fs.mkdirSync(path.join(base, "aircraft"));
  fs.writeFileSync(path.join(base, "aircraft", "2026-07-05-01.jsonl"), '{"a":1}\n{"a":2}\n{"a":3}\n');
  fs.mkdirSync(path.join(base, "vessels"));
  fs.writeFileSync(path.join(base, "vessels", "2026-07-04-01.jsonl.gz"),
    zlib.gzipSync('{"v":1}\n{"v":2}\n'));
  fs.mkdirSync(path.join(base, "waitlist"));
  fs.writeFileSync(path.join(base, "waitlist", "2026-07-05.jsonl"), '{"email":"x"}\n');
  assert.equal(await countFileLines(path.join(base, "aircraft", "2026-07-05-01.jsonl")), 3);
  // REPAIR (found 2026-07-22, same root cause + fix as datacoreArchive.ts's
  // streamJsonlLines): readline.Interface re-emits a piped-in stream's error
  // on ITSELF too, independent of the stream.on("error", ...) guard here —
  // unlistened, a truncated/corrupt .gz crashed the WHOLE PROCESS. See
  // datacoreArchive.test.ts for the full writeup + minimal repro.
  const truncFp = path.join(base, "aircraft", "truncated.jsonl.gz");
  const good = zlib.gzipSync('{"a":1}\n{"a":2}\n');
  fs.writeFileSync(truncFp, good.subarray(0, good.length - 4));
  await countFileLines(truncFp); // must resolve, never crash the process
  assert.equal(await countObservations(base), 5, "3 plain + 2 gz; waitlist PII stream excluded");
  _resetPlatformStatsCache();
  const s = await platformStats(
    [{ id: "aircraft", status: "live" }, { id: "vessels", status: "live" }, { id: "tank_fill", status: "planned" }],
    base,
  );
  assert.equal(s.layers_live, 2);
  assert.equal(s.layers_total, 3);
  assert.equal(s.streams_recording, 2, "aircraft+vessels dirs map onto their live layers");
  assert.equal(s.observations, 5);
  assert.ok(s.observations_as_of, "count must carry its freshness timestamp");
  fs.rmSync(base, { recursive: true, force: true });
});

test("wiring pinned: route registered; hero consumes the endpoint, not the old samples field", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('"/api/data/platform/stats"'), "platform stats route missing");
  const landing = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "landing.tsx"), "utf8");
  assert.ok(landing.includes("/api/data/platform/stats"), "hero must read the real platform stats");
  assert.ok(!landing.includes("kinds[k]?.samples"), "the phantom samples field caused the prod dash — must be gone");
});

// [RULE-REVIEW 2026-11-06] measurement: a "_"-prefixed file is a stream's own
// bookkeeping in this archive (datacoreArchive.newestDataFile's rule), never
// observations. Before, every *.jsonl / *.jsonl.gz counted, so the DTCC
// seen-ID index (_seen_ids_index.jsonl.gz: one dedup ID per line, the SAME
// IDs as the day files' rows, ~6.94M lines on 2026-11-06) and FINRA's derived
// _summary_history.jsonl (one summary per day of rows already counted) were
// counted as observations on top of the rows they describe.

test("observation files: *.jsonl / *.jsonl.gz, never a '_'-prefixed bookkeeping file", () => {
  for (const f of ["2026-11-06.jsonl.gz", "2026-11-06-13.jsonl", "2026-11-06_spacex.jsonl", "readings.jsonl"]) {
    assert.equal(isObservationFile(f), true, f);
  }
  for (const f of ["_seen_ids_index.jsonl.gz", "_summary_history.jsonl", "_seen_ids.db", "_state.json.gz",
    "backfill_done.json", ".platform_stats_cache.json", "2026-11-06.jsonl.gz.tmp", "notes.txt"]) {
    assert.equal(isObservationFile(f), false, f);
  }
});

test("observations: the DTCC seen-ID index and FINRA's derived summary history are not counted; a dir holding only bookkeeping is not a recording stream", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "pstats-meta-"));
  try {
    fs.mkdirSync(path.join(base, "dtccswaps"));
    fs.writeFileSync(path.join(base, "dtccswaps", "2026-11-05.jsonl.gz"), zlib.gzipSync('{"disseminationId":"1"}\n{"disseminationId":"2"}\n'));
    fs.writeFileSync(path.join(base, "dtccswaps", "2026-11-06.jsonl.gz"), zlib.gzipSync('{"disseminationId":"3"}\n'));
    // the dedup index: the same three IDs, one per line — not three more observations
    fs.writeFileSync(path.join(base, "dtccswaps", "_seen_ids_index.jsonl.gz"), zlib.gzipSync("1\n2\n3\n"));
    fs.mkdirSync(path.join(base, "finrashortvol"));
    fs.writeFileSync(path.join(base, "finrashortvol", "2026-11-05.jsonl"), '{"symbol":"A"}\n{"symbol":"B"}\n');
    fs.writeFileSync(path.join(base, "finrashortvol", "_summary_history.jsonl"), '{"date":"2026-11-05","symbols":2,"agg_short_ratio":0.4}\n');
    fs.mkdirSync(path.join(base, "metaonly"));
    fs.writeFileSync(path.join(base, "metaonly", "_index.jsonl"), "x\ny\n");
    _resetPlatformStatsCache();
    assert.equal(await countObservations(base), 5, "3 DTCC rows + 2 FINRA rows (was 11: + 3 index lines + 1 summary line + 2 metaonly lines)");
    assert.deepEqual(listArchiveDirs(base).sort(), ["dtccswaps", "finrashortvol"]);
    _resetPlatformStatsCache();
    const s = await platformStats([], base);
    assert.equal(s.observations, 5);
    assert.equal(s.streams_recording, 2, "dtccswaps + finrashortvol; metaonly records no observations");
  } finally {
    _resetPlatformStatsCache();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("after the deploy: the persisted pre-change total is served first, then the recount drops the index's lines without reading any file again", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "pstats-meta-persisted-"));
  const dir = path.join(base, "dtccswaps");
  fs.mkdirSync(dir);
  fs.writeFileSync(path.join(dir, "2026-11-05.jsonl.gz"), zlib.gzipSync('{"disseminationId":"1"}\n{"disseminationId":"2"}\n'));
  fs.writeFileSync(path.join(dir, "2026-11-06.jsonl.gz"), zlib.gzipSync('{"disseminationId":"3"}\n'));
  fs.writeFileSync(path.join(dir, "_seen_ids_index.jsonl.gz"), zlib.gzipSync("1\n2\n3\n"));
  const entry = (f: string, lines: number) => {
    const st = fs.statSync(path.join(dir, f));
    return [st.size, st.mtimeMs, lines];
  };
  // the cache file as the old rule left it: the index counted as 3 observations
  const at = Date.now() - 60 * 60_000;
  fs.writeFileSync(path.join(base, PLATFORM_STATS_CACHE_FILE), JSON.stringify({ v: 1, at, observations: 6, files: {
    "dtccswaps/2026-11-05.jsonl.gz": entry("2026-11-05.jsonl.gz", 2),
    "dtccswaps/2026-11-06.jsonl.gz": entry("2026-11-06.jsonl.gz", 1),
    "dtccswaps/_seen_ids_index.jsonl.gz": entry("_seen_ids_index.jsonl.gz", 3),
  } }));
  const reads: string[] = [];
  _resetPlatformStatsCache();
  _setLineCounterForTests((fp) => { reads.push(path.basename(fp)); return countFileLines(fp); });
  try {
    const first = await platformStats([], base);
    assert.equal(first.observations, 6, "the persisted pre-change total, served at once with its own as_of");
    assert.equal(first.observations_as_of, new Date(at).toISOString());
    for (let i = 0; i < 200 && (await platformStats([], base)).observations_pending; i++) await new Promise((r) => setTimeout(r, 10));
    const after = await platformStats([], base);
    assert.equal(after.observations, 3, "the rows only");
    assert.deepEqual(reads, [], "the day files' persisted counts were trusted; the index was never read");
    const onDisk = JSON.parse(fs.readFileSync(path.join(base, PLATFORM_STATS_CACHE_FILE), "utf8"));
    assert.equal(onDisk.observations, 3);
    assert.deepEqual(Object.keys(onDisk.files).sort(), ["dtccswaps/2026-11-05.jsonl.gz", "dtccswaps/2026-11-06.jsonl.gz"],
      "the index's entry left the persisted counts");
  } finally {
    _setLineCounterForTests(null);
    _resetPlatformStatsCache();
    fs.rmSync(base, { recursive: true, force: true });
  }
});
