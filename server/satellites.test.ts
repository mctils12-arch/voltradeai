// CelesTrak GP satellites battery (ANALYST CONSOLE W2): group list
// verified against the live elements index (charter's
// "active-geosynchronous" does not exist — `geo` is the documented
// Active Geosynchronous group), OMM JSON parse on the live-verified
// shape, NORAD_CAT_ID|EPOCH dedup across polls, day-file append +
// gz-merge, latest-per-satellite route data path, warming_up honesty,
// seeded-dedup-from-disk restart, unknown-group rejection, and the
// CelesTrak usage-policy stance (non-200 recorded, never retried).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  GROUPS, gpUrl, parseGp, fetchGroup, archiveGp, gzipOldSatDays,
  refreshSatellites, latestGroup, latestPerSatellite, satellitesResponse,
  satIssues, seedFileInWindow, SEED_WINDOW_DAYS, resetSatellitesForTests,
  ATTRIBUTION, POLL_INTERVAL_MS,
} from "./satellites";

// Mirrors the live-verified payload shape (2026-07-07, GROUP=stations):
// array of OMM objects. First record quoted from the live probe.
const ISS = {
  OBJECT_NAME: "ISS (ZARYA)", OBJECT_ID: "1998-067A",
  EPOCH: "2026-07-07T12:12:01.986912",
  MEAN_MOTION: 15.48933372, ECCENTRICITY: 0.0006687, INCLINATION: 51.6304,
  RA_OF_ASC_NODE: 199.5144, ARG_OF_PERICENTER: 267.6545, MEAN_ANOMALY: 92.3678,
  EPHEMERIS_TYPE: 0, CLASSIFICATION_TYPE: "U", NORAD_CAT_ID: 25544,
  ELEMENT_SET_NO: 999, REV_AT_EPOCH: 57490, BSTAR: 0.00011369,
  MEAN_MOTION_DOT: 5.806e-5, MEAN_MOTION_DDOT: 0,
};
const CSS = { ...ISS, OBJECT_NAME: "CSS (TIANHE)", OBJECT_ID: "2021-035A",
              NORAD_CAT_ID: 48274, EPOCH: "2026-07-07T10:00:00.000000" };
const advance = (r: any, epoch: string) => ({ ...r, EPOCH: epoch, REV_AT_EPOCH: r.REV_AT_EPOCH + 1 });

const okJson = (payload: unknown) =>
  ({ ok: true, status: 200, text: async () => JSON.stringify(payload) });

test("groups + url: verified group names, JSON format (TLE dies at catnr 69999)", () => {
  assert.deepEqual([...GROUPS], ["stations", "starlink", "gps-ops", "geo"],
    "geo (not the charter's 'active-geosynchronous') is CelesTrak's documented Active Geosynchronous group");
  assert.equal(gpUrl("gps-ops"),
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=json");
  assert.equal(POLL_INTERVAL_MS, 6 * 3600_000, "6h — 3x CelesTrak's 2h GP update cycle");
});

test("parseGp: live shape accepted; junk dropped, never guessed", () => {
  assert.equal(parseGp([ISS, CSS]).length, 2);
  assert.deepEqual(parseGp("not an array"), []);
  assert.deepEqual(parseGp([null, 7, { EPOCH: "2026-01-01" }, { NORAD_CAT_ID: 1 },
                            { NORAD_CAT_ID: "25544", EPOCH: "x" }]), [],
    "records need numeric NORAD_CAT_ID and string EPOCH");
});

test("fetchGroup: non-200 recorded as issue and NOT retried (CelesTrak usage policy)", async () => {
  resetSatellitesForTests();
  let calls = 0;
  const forbid = async () => { calls++; return { ok: false, status: 403, text: async () => "blocked" }; };
  assert.equal(await fetchGroup("stations", forbid as any), null);
  assert.equal(calls, 1, "exactly one attempt — a 403/404 never changes on retry");
  assert.match(satIssues().stations, /http 403/);
  // recovery clears the issue
  assert.equal((await fetchGroup("stations", (async () => okJson([ISS])) as any))!.length, 1);
  assert.ok(!("stations" in satIssues()));
});

test("archive: dedup by NORAD_CAT_ID|EPOCH across polls; epoch advance appends (orbit history)", () => {
  resetSatellitesForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "sats-"));
  const t0 = Date.parse("2026-07-07T12:30:00Z");
  assert.equal(archiveGp("stations", [ISS, CSS] as any, base, t0), 2);
  assert.equal(archiveGp("stations", [ISS, CSS] as any, base, t0 + 3600_000), 0,
    "same-epoch re-fetch appends nothing");
  const newEpoch = advance(ISS, "2026-07-07T18:04:33.120000");
  assert.equal(archiveGp("stations", [newEpoch, CSS] as any, base, t0 + 6 * 3600_000), 1,
    "EPOCH advance is a new element set — appends");
  const day = path.join(base, "satellites", "stations", "2026-07-07.jsonl");
  const rows = fs.readFileSync(day, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(rows.length, 3, "one UTC-fetch-day file per group, append-only");
  assert.equal(rows[0].rt, "2026-07-07T12:30:00.000Z", "every row carries the fetch timestamp");
  assert.deepEqual(rows.filter((r) => r.NORAD_CAT_ID === 25544).map((r) => r.EPOCH),
    ["2026-07-07T12:12:01.986912", "2026-07-07T18:04:33.120000"],
    "both epochs kept — the archive is the history");
});

test("seeded-dedup-from-disk: restart must not re-archive epochs already on disk", () => {
  resetSatellitesForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "sats-"));
  const t0 = Date.parse("2026-07-07T12:30:00Z");
  assert.equal(archiveGp("geo", [ISS] as any, base, t0), 1);
  resetSatellitesForTests(); // simulated process restart — in-memory Set gone
  assert.equal(archiveGp("geo", [ISS] as any, base, t0 + 3600_000), 0,
    "seed from disk caught the already-archived NORAD_CAT_ID|EPOCH");
  assert.equal(archiveGp("geo", [advance(ISS, "2026-07-08T01:00:00.000000")] as any, base, t0 + 3600_000), 1,
    "new epoch still appends after seeding");
  assert.ok(seedFileInWindow("2026-07-05.jsonl", Date.parse("2026-07-07")));
  assert.ok(!seedFileInWindow("2020-01-01.jsonl.gz", Date.parse("2026-07-07")),
    `seed window bounded to ${SEED_WINDOW_DAYS}d (heap protection)`);
});

test("gz merge: late append after the day gz'd must merge, never overwrite", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "sats-"));
  const dir = path.join(base, "satellites", "starlink");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-06-01.jsonl"), '{"a":1}\n');
  assert.equal(gzipOldSatDays(base, Date.parse("2026-07-07")), 1);
  fs.writeFileSync(path.join(dir, "2026-06-01.jsonl"), '{"a":2}\n'); // late append post-gz
  assert.equal(gzipOldSatDays(base, Date.parse("2026-07-07")), 1);
  const merged = zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-06-01.jsonl.gz"))).toString();
  assert.equal(merged, '{"a":1}\n{"a":2}\n', "prior gz rows survive the late re-gz");
  fs.writeFileSync(path.join(dir, "2026-07-06.jsonl"), '{"a":3}\n');
  assert.equal(gzipOldSatDays(base, Date.parse("2026-07-07")), 0, "recent days stay plain");
});

test("latestPerSatellite: highest EPOCH wins per NORAD_CAT_ID", () => {
  const stale = { ...ISS, EPOCH: "2026-07-06T00:00:00.000000" };
  const picked = latestPerSatellite([stale, ISS, CSS] as any);
  assert.equal(picked.length, 2);
  assert.equal(picked.find((r) => r.NORAD_CAT_ID === 25544)!.EPOCH, ISS.EPOCH);
});

test("warming_up honesty: before any successful poll, no records are invented", () => {
  resetSatellitesForTests();
  assert.equal(latestGroup("stations"), null);
  const r = satellitesResponse("stations");
  assert.equal(r.status, 200);
  assert.equal(r.body.warming_up, true);
  assert.deepEqual(r.body.satellites, []);
  assert.equal(r.body.count, 0);
  const all = satellitesResponse("all");
  assert.ok(all.body.summary.every((s: any) => s.warming_up && s.count === 0 && s.newest_epoch === null));
});

test("unknown group rejected with 400 + the valid group list", () => {
  const r = satellitesResponse("active-geosynchronous");
  assert.equal(r.status, 400);
  assert.match(r.body.error, /unknown group 'active-geosynchronous'/);
  assert.deepEqual(r.body.groups, [...GROUPS]);
});

test("refresh sweep + route envelope: cache-only serving, freshness, attribution, all-summary", async () => {
  resetSatellitesForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "sats-"));
  const t0 = Date.parse("2026-07-07T12:30:00Z");
  let calls = 0;
  const fake = async (url: string) => {
    calls++;
    if (url.includes("GROUP=stations")) return okJson([CSS, ISS, { ...ISS, EPOCH: "2026-07-06T00:00:00.000000" }]);
    if (url.includes("GROUP=geo")) return { ok: false, status: 403, text: async () => "blocked" };
    return okJson([{ ...CSS, NORAD_CAT_ID: 99999 + calls }]);
  };
  await refreshSatellites(fake as any, t0, base, 0);
  assert.equal(calls, GROUPS.length, "one request per group per sweep");
  const r = satellitesResponse("stations");
  assert.equal(r.status, 200);
  assert.equal(r.body.count, 2, "latest per NORAD_CAT_ID — the stale ISS duplicate collapsed");
  assert.equal(r.body.freshness.newest_epoch, ISS.EPOCH);
  assert.equal(r.body.freshness.fetched_at, t0);
  assert.equal(r.body.attribution, ATTRIBUTION);
  assert.equal(r.body.satellites.find((s: any) => s.NORAD_CAT_ID === 25544).EPOCH, ISS.EPOCH);
  // failed group: warming (no cache) + its issue surfaced, never zero-filled as data
  const geo = satellitesResponse("geo");
  assert.equal(geo.body.warming_up, true);
  assert.match(geo.body.issue, /http 403/);
  // all-mode summary: counts + freshness only, no record payloads
  const all = satellitesResponse("all");
  const st = all.body.summary.find((s: any) => s.group === "stations");
  assert.equal(st.count, 2);
  assert.equal(st.newest_epoch, ISS.EPOCH);
  assert.ok(!("satellites" in all.body));
  assert.match(all.body.issues.geo, /http 403/);
  // archive landed under satellites/<group>/<fetch-day>.jsonl
  assert.ok(fs.existsSync(path.join(base, "satellites", "stations", "2026-07-07.jsonl")));
});

// ---- R17 2026-07-07: transport-failure host fallback + cause surfacing ----

test("R17: transport failure on .org falls back to .com once; non-200 NEVER tries the alternate host", async () => {
  resetSatellitesForTests();
  const calls: string[] = [];
  const okBody = JSON.stringify([{ NORAD_CAT_ID: 25544, EPOCH: "2026-07-07T12:00:00", OBJECT_NAME: "ISS" }]);
  // transport failure on first host, success on second
  const recs = await fetchGroup("stations", (async (url: string) => {
    calls.push(url);
    if (url.includes("celestrak.org")) { const e: any = new Error("fetch failed"); e.cause = { code: "ETIMEDOUT" }; throw e; }
    return { ok: true, status: 200, text: async () => okBody };
  }) as any);
  assert.ok(recs && recs.length === 1, "alternate host result must be used");
  assert.equal(calls.length, 2);
  assert.ok(calls[0].includes("celestrak.org") && calls[1].includes("celestrak.com"));
  // non-200: answered by their server -> stop immediately, no second host
  resetSatellitesForTests();
  const calls2: string[] = [];
  const r2 = await fetchGroup("stations", (async (url: string) => {
    calls2.push(url);
    return { ok: false, status: 403, text: async () => "" };
  }) as any);
  assert.equal(r2, null);
  assert.equal(calls2.length, 1, "M2M rule: an HTTP response stops the sweep — no alternate-host retry");
  assert.ok(satIssues()["stations"].includes("http 403"));
});

test("R17: both hosts failing at transport level surfaces per-host REAL causes (undici error.cause), not 'fetch failed'", async () => {
  resetSatellitesForTests();
  const r = await fetchGroup("stations", (async (url: string) => {
    const e: any = new Error("fetch failed");
    e.cause = url.includes(".org") ? { code: "ETIMEDOUT" } : { code: "ENOTFOUND" };
    throw e;
  }) as any);
  assert.equal(r, null);
  const issue = satIssues()["stations"];
  assert.ok(issue.includes("celestrak.org: fetch failed | ETIMEDOUT"), `org cause missing: ${issue}`);
  assert.ok(issue.includes("celestrak.com: fetch failed | ENOTFOUND"), `com cause missing: ${issue}`);
});
