// A LATE raw file for an hour that is already compressed must be MERGED into the existing .gz, never written over
// it (2026-11-16 [REPAIR]). archiveAircraftAt writes a fix into the hour of the fix's OWN time; for an hour more
// than two hours old that creates YYYY-MM-DD-HH.jsonl beside the existing YYYY-MM-DD-HH.jsonl.gz, and both
// compressors then replaced the whole hour with the late lines (reproduced on main: 50 lines -> 1).
// Run: npx tsx --test server/archiveLateHourMerge.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  archiveAircraftAt, compressOldHours, compressOldHoursAsync, lateHourMergeStats, mergedHourBody, LATE_HOUR_MERGE_MAX_GZ_BYTES,
} from "./datacoreArchive";

const HOUR = "2026-11-16-08";                                  // the hour file under test
const HOUR_MS = Date.parse("2026-11-16T08:00:00Z");
const NOW = HOUR_MS + 6 * 3600_000;                            // six hours later: long past the two-hour line
const line = (i: number) => JSON.stringify({ t: HOUR_MS / 1000 + i, i: `a${String(i).padStart(5, "0")}`, la: 40 + i / 1000, lo: -100 });

function hourWith50(kind = "aircraft"): { base: string; dir: string; gz: string; raw: string } {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "latehour-"));
  const dir = path.join(base, kind);
  fs.mkdirSync(dir, { recursive: true });
  const gz = path.join(dir, `${HOUR}.jsonl.gz`);
  fs.writeFileSync(gz, zlib.gzipSync(Array.from({ length: 50 }, (_, i) => line(i)).join("\n") + "\n"));
  return { base, dir, gz, raw: path.join(dir, `${HOUR}.jsonl`) };
}
const linesOf = (gz: string): string[] => zlib.gunzipSync(fs.readFileSync(gz)).toString("utf8").split("\n").filter(Boolean);

for (const [name, run] of [
  ["sync", async (base: string, cap?: number) => compressOldHours(base, NOW, cap)],
  ["async", async (base: string, cap?: number) => compressOldHoursAsync(base, NOW, cap)],
] as const) {
  test(`${name}: a late line for a compressed hour is appended to the hour; the 50 lines already there survive`, async () => {
    const h = hourWith50();
    fs.writeFileSync(h.raw, line(999) + "\n");
    const before = lateHourMergeStats().merged;
    assert.equal(await run(h.base), 1);
    const lines = linesOf(h.gz);
    assert.equal(lines.length, 51, "50 existing + 1 late (main left 1)");
    assert.deepEqual(lines.slice(0, 50), Array.from({ length: 50 }, (_, i) => line(i)), "the existing lines, in order, first");
    assert.equal(lines[50], line(999));
    assert.equal(fs.existsSync(h.raw), false, "the raw file is gone once it is inside the .gz");
    assert.deepEqual(fs.readdirSync(h.dir), [`${HOUR}.jsonl.gz`], "no temp file left behind");
    assert.equal(lateHourMergeStats().merged, before + 1);
  });

  test(`${name}: an ordinary hour compresses as before, through a temp file`, async () => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "latehour-"));
    const dir = path.join(base, "vessels");
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${HOUR}.jsonl`), line(1) + "\n" + line(2) + "\n");
    const fresh = new Date(NOW - 3600_000).toISOString().slice(0, 13).replace("T", "-");
    fs.writeFileSync(path.join(dir, `${fresh}.jsonl`), line(3) + "\n"); // under two hours old: left alone
    assert.equal(await run(base), 1);
    assert.deepEqual(linesOf(path.join(dir, `${HOUR}.jsonl.gz`)), [line(1), line(2)]);
    assert.deepEqual(fs.readdirSync(dir).sort(), [`${HOUR}.jsonl.gz`, `${fresh}.jsonl`].sort());
  });

  test(`${name}: an existing .gz that does not gunzip is left exactly as it is, and so is the late file`, async () => {
    const h = hourWith50();
    fs.writeFileSync(h.gz, Buffer.from("not gzip at all"));
    fs.writeFileSync(h.raw, line(999) + "\n");
    const before = lateHourMergeStats().left_beside;
    assert.equal(await run(h.base), 0);
    assert.equal(fs.readFileSync(h.gz, "utf8"), "not gzip at all", "never deleted, never overwritten");
    assert.equal(fs.readFileSync(h.raw, "utf8"), line(999) + "\n", "the late lines stay readable beside it");
    assert.deepEqual(fs.readdirSync(h.dir).sort(), [`${HOUR}.jsonl`, `${HOUR}.jsonl.gz`]);
    assert.equal(lateHourMergeStats().left_beside, before + 1);
  });

  test(`${name}: an existing .gz over the merge size limit keeps both files`, async () => {
    const h = hourWith50();
    const size = fs.statSync(h.gz).size;
    fs.writeFileSync(h.raw, line(999) + "\n");
    assert.equal(await run(h.base, size - 1), 0);
    assert.equal(linesOf(h.gz).length, 50);
    assert.ok(fs.existsSync(h.raw));
    assert.equal(await run(h.base, size), 1, "at the limit it merges");
    assert.equal(linesOf(h.gz).length, 51);
  });
}

test("mergedHourBody: existing lines first; a missing final newline never joins two lines", () => {
  assert.equal(mergedHourBody(Buffer.from("a\n"), Buffer.from("b\n")).toString(), "a\nb\n");
  assert.equal(mergedHourBody(Buffer.from("a"), Buffer.from("b\n")).toString(), "a\nb\n");
  assert.equal(mergedHourBody(Buffer.alloc(0), Buffer.from("b\n")).toString(), "b\n");
  assert.equal(LATE_HOUR_MERGE_MAX_GZ_BYTES, 64 * 1024 * 1024);
});

test("end to end with the real writer: a backfilled fix for an old hour joins that hour's archive", async () => {
  const h = hourWith50();
  // the tracked tails' trace backfill: a fix whose own time is inside the already-compressed hour
  const n = archiveAircraftAt([{ icao24: "ab8c8e", lat: 41.5, lon: -87.7, altitude_m: 259, on_ground: false, velocity_ms: 74, heading: 180, tSec: HOUR_MS / 1000 + 1800 }], h.base);
  assert.equal(n, 1);
  assert.ok(fs.existsSync(h.raw), "the writer created a raw file beside the existing .gz");
  assert.equal(await compressOldHoursAsync(h.base, NOW), 1);
  const lines = linesOf(h.gz);
  assert.equal(lines.length, 51);
  assert.equal(JSON.parse(lines[50]).i, "ab8c8e");
  assert.equal(JSON.parse(lines[0]).i, "a00000", "every aircraft that was in the hour is still in it");
});
