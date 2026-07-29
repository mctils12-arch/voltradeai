// SWPC space-weather battery (research/open_questions.md "GRID-ADJACENT
// FUTURE ROOTS" #1, picked up from the 2026-07-27 live-data gap sweep).
// Fixtures mirror the live shapes probed 2026-07-29: planetary-k-index.json
// (array of {time_tag,Kp,a_running,station_count}) and xrays-1-day.json
// (array of {time_tag,satellite,flux,energy,electron_contaminaton}, two
// energy bands interleaved, NOAA's own field misspelled upstream).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  parseKIndex, parseXray, classifyKp, classifyFlare,
  archiveKIndex, archiveXray, gzipOldDays, refreshSpaceWeatherCache, latestSpaceWeather,
} from "./spaceWeather";

test("parseKIndex: keeps well-formed rows, drops rows missing time_tag/Kp", () => {
  const recs = parseKIndex([
    { time_tag: "2026-07-29T00:00:00", Kp: 3.33, a_running: 18, station_count: 8 },
    { time_tag: "2026-07-29T03:00:00" }, // missing Kp
    { Kp: 2.0 }, // missing time_tag
  ]);
  assert.equal(recs.length, 1);
  assert.equal(recs[0].kp, 3.33);
  assert.equal(recs[0].a_running, 18);
});

test("parseXray: keeps both energy bands, normalizes NOAA's misspelled contamination field", () => {
  const recs = parseXray([
    { time_tag: "2026-07-29T13:06:00Z", satellite: 18, flux: 1.16e-6, energy: "0.1-0.8nm", electron_contaminaton: false },
    { time_tag: "2026-07-29T13:06:00Z", satellite: 18, flux: 1.5e-8, energy: "0.05-0.4nm", electron_contaminaton: true },
    { time_tag: "2026-07-29T13:07:00Z" }, // missing flux/energy
  ]);
  assert.equal(recs.length, 2);
  assert.equal(recs[0].electron_contamination, false);
  assert.equal(recs[1].electron_contamination, true);
});

test("classifyKp: official Kp -> G-scale boundaries", () => {
  assert.deepEqual(classifyKp(2), { gScale: null, label: "quiet" });
  assert.deepEqual(classifyKp(3), { gScale: null, label: "unsettled" });
  assert.deepEqual(classifyKp(4), { gScale: null, label: "active" });
  assert.equal(classifyKp(5).gScale, 1);
  assert.equal(classifyKp(6).gScale, 2);
  assert.equal(classifyKp(7).gScale, 3);
  assert.equal(classifyKp(8).gScale, 4);
  assert.equal(classifyKp(9).gScale, 5);
});

test("classifyFlare: official GOES class boundaries + magnitude", () => {
  assert.equal(classifyFlare(5e-9).cls, "A");
  assert.equal(classifyFlare(5e-8).cls, "A");
  assert.equal(classifyFlare(5e-7).cls, "B");
  assert.equal(classifyFlare(1.16e-6).cls, "C");
  assert.equal(classifyFlare(1.16e-6).magnitude, 1.16);
  assert.equal(classifyFlare(5e-5).cls, "M");
  assert.equal(classifyFlare(2e-4).cls, "X");
  assert.equal(classifyFlare(2e-4).magnitude, 2);
});

test("archiveKIndex: dedup by time_tag across polls, gz after 2 days", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "swpck-"));
  const now = Date.parse("2026-07-29T12:00:00Z");
  const recs = parseKIndex([
    { time_tag: "2026-07-29T00:00:00", Kp: 2.0 },
    { time_tag: "2026-07-29T03:00:00", Kp: 3.0 },
  ]);
  assert.equal(archiveKIndex(recs, base, now), 2);
  assert.equal(archiveKIndex(recs, base, now), 0, "same time_tags never re-archive");
  const recs2 = parseKIndex([{ time_tag: "2026-07-29T06:00:00", Kp: 1.5 }]);
  assert.equal(archiveKIndex(recs2, base, now), 1);
  const day = path.join(base, "swpckindex", "2026-07-29.jsonl");
  const lines = fs.readFileSync(day, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.deepEqual(lines.map((r: any) => r.time_tag), ["2026-07-29T00:00:00", "2026-07-29T03:00:00", "2026-07-29T06:00:00"]);
  assert.equal(gzipOldDays("swpckindex", base, now + 3 * 86400_000), 1);
  assert.ok(!fs.existsSync(day) && fs.existsSync(`${day}.gz`));
  const unz = zlib.gunzipSync(fs.readFileSync(`${day}.gz`)).toString("utf8");
  assert.ok(unz.includes("2026-07-29T00:00:00"));
});

test("archiveXray: dedup keys on time_tag+energy so both bands survive one time_tag", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "swpcx-"));
  const now = Date.parse("2026-07-29T13:06:00Z");
  const recs = parseXray([
    { time_tag: "2026-07-29T13:06:00Z", flux: 1.16e-6, energy: "0.1-0.8nm" },
    { time_tag: "2026-07-29T13:06:00Z", flux: 1.5e-8, energy: "0.05-0.4nm" },
  ]);
  assert.equal(archiveXray(recs, base, now), 2, "both bands of the same time_tag archive");
  assert.equal(archiveXray(recs, base, now), 0, "re-polling the same minute archives nothing new");
});

test("refreshSpaceWeatherCache: picks latest long-band X-ray reading, classifies both, one source failing keeps the other's data", async () => {
  const kJson = JSON.stringify([{ time_tag: "2026-07-29T09:00:00", Kp: 7.0, a_running: 39, station_count: 8 }]);
  const xJson = JSON.stringify([
    { time_tag: "2026-07-29T13:06:00Z", flux: 1.5e-8, energy: "0.05-0.4nm" },
    { time_tag: "2026-07-29T13:06:00Z", flux: 1.16e-6, energy: "0.1-0.8nm" },
    { time_tag: "2026-07-29T13:07:00Z", flux: 1.2e-6, energy: "0.1-0.8nm" },
  ]);
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "swpcboth-"));
  const fetchImpl = async (url: string) => ({
    ok: true, status: 200,
    text: async () => (url.includes("k-index") ? kJson : xJson),
  });
  await refreshSpaceWeatherCache(fetchImpl as any, Date.parse("2026-07-29T13:07:00Z"), base);
  const hit = latestSpaceWeather();
  assert.ok(hit);
  assert.equal(hit!.kindex.latest?.kp, 7.0);
  assert.equal(hit!.kindex.gScale?.gScale, 3);
  assert.equal(hit!.xray.latest?.flux, 1.2e-6, "picks the last long-band row, not the short-band interleave");
  assert.equal(hit!.xray.flare?.cls, "C");
  assert.equal(hit!.xray.history.length, 2, "long-band-only history, short band excluded");

  // Now simulate the X-ray endpoint failing: kindex data should persist.
  const failingXFetch = async (url: string) => {
    if (url.includes("k-index")) return { ok: true, status: 200, text: async () => kJson };
    return { ok: false, status: 500, text: async () => "" };
  };
  await refreshSpaceWeatherCache(failingXFetch as any, Date.parse("2026-07-29T13:12:00Z"), base);
  const hit2 = latestSpaceWeather();
  assert.equal(hit2!.xray.latest?.flux, 1.2e-6, "stale xray cache survives a failed poll rather than going blank");
});
