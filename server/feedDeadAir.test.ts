import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  parseArchiveHourMs, feedDeadAir, feedDeadAirCheck, newestArchiveHourFile,
  observeFeedDeadAir, FEED_DEAD_AIR_HOURS, CONTINUOUS_FEEDS,
} from "./feedDeadAir";

// THE REGRESSION THIS FILE EXISTS FOR (CLAUDE.md "REPAIRS MUST RATCHET"):
// the AIS vessel feed went silent 2026-08-05 ~13:31 UTC and every liveness
// signal we had read "healthy" for seven days, because the reconnect loop
// kept succeeding. The archive is the one clock a redial cannot reset.

const HOUR = 3_600_000;
const AUG12_02Z = Date.parse("2026-08-12T02:38:00Z");

test("parseArchiveHourMs reads the UTC hour out of an archive filename", () => {
  assert.equal(parseArchiveHourMs("2026-08-05-13.jsonl"), Date.parse("2026-08-05T13:00:00Z"));
  assert.equal(parseArchiveHourMs("2026-08-05-13.jsonl.gz"), Date.parse("2026-08-05T13:00:00Z"));
  assert.equal(parseArchiveHourMs("2026-08-12-00.jsonl"), Date.parse("2026-08-12T00:00:00Z"));
});

test("parseArchiveHourMs refuses anything that is not an hour file", () => {
  // A name we cannot parse must never be mistaken for fresh data.
  for (const bad of [null, undefined, "", "2026-08-04.jsonl.gz", "notes.txt", "2026-08-05.day.jsonl.gz"]) {
    assert.equal(parseArchiveHourMs(bad as any), null, `expected null for ${String(bad)}`);
  }
});

test("a feed writing the current hour is alive", () => {
  const [v] = feedDeadAir([{ kind: "aircraft", newestHourFile: "2026-08-12-02.jsonl" }], AUG12_02Z);
  assert.equal(v.dead, false);
  assert.equal(v.detail, "");
  assert.ok(v.silentHours !== null && v.silentHours < 1);
});

test("hour-bucket granularity does not false-alarm a healthy feed", () => {
  // Sampled 59 minutes into the bucket, a live feed reads ~0.98h of
  // "silence". That is why the threshold is 3h, not 1h.
  const now = Date.parse("2026-08-12T02:59:00Z");
  const [v] = feedDeadAir([{ kind: "trains", newestHourFile: "2026-08-12-02.jsonl" }], now);
  assert.equal(v.dead, false);
});

test("THE AUG-5 CASE: vessels frozen while aircraft and trains stay current", () => {
  // Exactly the live production state on 2026-08-12T02:38Z, read from
  // /api/data/archive/stats: aircraft and trains at hour 2026-08-12-02,
  // vessels stuck on 2026-08-05-13. Before this module, /api/health had no
  // feed-freshness check at all and reported a clean bill for six days.
  const verdicts = feedDeadAir([
    { kind: "aircraft", newestHourFile: "2026-08-12-02.jsonl" },
    { kind: "vessels", newestHourFile: "2026-08-05-13.jsonl.gz" },
    { kind: "trains", newestHourFile: "2026-08-12-02.jsonl" },
  ], AUG12_02Z);

  const byKind = Object.fromEntries(verdicts.map((v) => [v.kind, v]));
  assert.equal(byKind.aircraft.dead, false);
  assert.equal(byKind.trains.dead, false);
  assert.equal(byKind.vessels.dead, true);
  assert.ok(byKind.vessels.silentHours! > 157, "vessels silent >6.5 days");

  const check = feedDeadAirCheck(verdicts);
  assert.equal(check.status, "degraded");
  assert.deepEqual(check.dead, ["vessels"]);
  assert.match(check.detail, /FEED DEAD-AIR ALARM/);
  assert.match(check.detail, /vessels/);
  // The detail must not assert a cause — this detector makes no causal claim.
  assert.doesNotMatch(check.detail, /key|aisstream|provider|replica/i);
});

test("the alarm would have fired the same morning, not on day seven", () => {
  // 4h after the last archived hour — the earliest a 3h threshold can fire.
  const fourHoursLater = Date.parse("2026-08-05T17:00:00Z");
  const [v] = feedDeadAir([{ kind: "vessels", newestHourFile: "2026-08-05-13.jsonl" }], fourHoursLater);
  assert.equal(v.dead, true);
});

test("threshold boundary: dead strictly past the threshold, not at it", () => {
  const base = Date.parse("2026-08-12T00:00:00Z");
  const at = feedDeadAir([{ kind: "vessels", newestHourFile: "2026-08-12-00.jsonl" }],
    base + FEED_DEAD_AIR_HOURS * HOUR);
  assert.equal(at[0].dead, false);
  const past = feedDeadAir([{ kind: "vessels", newestHourFile: "2026-08-12-00.jsonl" }],
    base + FEED_DEAD_AIR_HOURS * HOUR + 60_000);
  assert.equal(past[0].dead, true);
});

test("an empty archive directory is loud, not silent", () => {
  // The vessels dir REACHES this state on its own: raw hours roll up and
  // are deleted after ~7 days, so the frozen file disappears and staleness
  // would otherwise become invisible exactly as the outage got worse.
  const [v] = feedDeadAir([{ kind: "vessels", newestHourFile: null }], AUG12_02Z);
  assert.equal(v.dead, true);
  assert.equal(v.silentHours, null);
  assert.match(v.detail, /no archived hour files at all/);
});

test("all feeds healthy reports ok with no detail", () => {
  const check = feedDeadAirCheck(feedDeadAir(
    CONTINUOUS_FEEDS.map((kind) => ({ kind, newestHourFile: "2026-08-12-02.jsonl" })), AUG12_02Z));
  assert.equal(check.status, "ok");
  assert.deepEqual(check.dead, []);
  assert.equal(check.detail, "");
  assert.equal(Object.keys(check.feeds).length, 3);
  assert.equal(check.feeds.vessels.dead, false);
});

// ── the fs helper ────────────────────────────────────────────────────────────

function tmpArchive(files: Record<string, string[]>): string {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "feeddeadair-"));
  for (const [kind, names] of Object.entries(files)) {
    fs.mkdirSync(path.join(base, kind), { recursive: true });
    for (const n of names) fs.writeFileSync(path.join(base, kind, n), "{}\n");
  }
  return base;
}

test("newestArchiveHourFile picks the newest HOUR, ignoring junk filenames", () => {
  // A naive lexicographic "last file in the dir" picks `zz-scratch.txt`
  // and reads it as fresh data; a naive mtime sort picks whichever file
  // was compressed most recently, which is always an OLD hour.
  const base = tmpArchive({
    aircraft: ["2026-08-11-23.jsonl.gz", "2026-08-12-02.jsonl", "2026-08-12-01.jsonl.gz", "zz-scratch.txt"],
  });
  assert.equal(newestArchiveHourFile("aircraft", base), "2026-08-12-02.jsonl");
});

test("newestArchiveHourFile returns null for an absent or empty directory", () => {
  const base = tmpArchive({ vessels: [] });
  assert.equal(newestArchiveHourFile("vessels", base), null);
  assert.equal(newestArchiveHourFile("trains", base), null); // dir never created
});

test("observeFeedDeadAir end-to-end flags only the stale feed", () => {
  const base = tmpArchive({
    aircraft: ["2026-08-12-02.jsonl"],
    trains: ["2026-08-12-02.jsonl"],
    vessels: ["2026-08-05-13.jsonl.gz"],
  });
  const check = observeFeedDeadAir(AUG12_02Z, base);
  assert.equal(check.status, "degraded");
  assert.deepEqual(check.dead, ["vessels"]);
  assert.equal(check.feeds.aircraft.dead, false);
  assert.equal(check.feeds.trains.dead, false);
});

test("observeFeedDeadAir reports ok when every continuous feed is current", () => {
  const base = tmpArchive({
    aircraft: ["2026-08-12-02.jsonl"],
    trains: ["2026-08-12-02.jsonl"],
    vessels: ["2026-08-12-02.jsonl"],
  });
  assert.equal(observeFeedDeadAir(AUG12_02Z, base).status, "ok");
});
