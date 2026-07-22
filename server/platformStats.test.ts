// Platform stats — the hero's numbers must be REAL and self-updating.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "path";
import zlib from "zlib";
import { fileURLToPath } from "node:url";
import {
  streamsRecording, countFileLines, countObservations, platformStats,
  _resetPlatformStatsCache, DIR_LAYER_MAP,
} from "./platformStats";

const here = path.dirname(fileURLToPath(import.meta.url));

test("streams: live layers + unmapped archive dirs; operational and _tracks never counted", () => {
  const live = ["aircraft", "vessels", "weather", "weather_temp", "insider", "earnings"];
  const dirs = ["aircraft", "filings", "earnings8k", "optionchains", "waitlist", "apiusage", "aircraft_tracks"];
  // filings->insider and earnings8k->earnings are already live layers;
  // optionchains has no layer -> +1; waitlist/apiusage/_tracks excluded
  assert.equal(streamsRecording(live, dirs), live.length + 1);
  assert.equal(DIR_LAYER_MAP.filings, "insider", "dir-name translation must be pinned");
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
