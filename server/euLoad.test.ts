// ENTSO-E actual-load battery (wishlist 9c, key landed 2026-07-07):
// key gate (either env name), url stamp format + token never in logs,
// GL document parse (PT60M + PT15M position math, published-points-only
// honesty), acknowledgement detection (live-verified shape), archive
// dedup across resolutions, poll stats.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  euLoadEnabled, ZONES, periodStamp, loadUrl, parseAck, parseLoad,
  fetchLoad, archiveLoad, refreshLoad, latestLoad, seedFileInWindow,
  SEED_WINDOW_DAYS, readRecentArchivedLoad, _resetEuLoadForTests,
  SEEN_RETAIN_DAYS, HOURS_PER_FETCH,
} from "./euLoad";
import { readHeapGauges } from "./heapDiag";

// Mirrors the live-verified document shapes (2026-07-07): GL_MarketDocument
// with Period > timeInterval/resolution/Point(position, quantity);
// Acknowledgement_MarketDocument with Reason(code, text).
const GL = (periods: string) => `<?xml version="1.0"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
 <TimeSeries><outBiddingZone_Domain.mRID>10YFR-RTE------C</outBiddingZone_Domain.mRID>
 ${periods}</TimeSeries></GL_MarketDocument>`;
const PERIOD = (start: string, res: string, points: [number, string][]) =>
  `<Period><timeInterval><start>${start}</start><end>x</end></timeInterval>
   <resolution>${res}</resolution>
   ${points.map(([p, q]) => `<Point><position>${p}</position><quantity>${q}</quantity></Point>`).join("")}
  </Period>`;
const ACK = `<?xml version="1.0"?><Acknowledgement_MarketDocument>
 <Reason><code>999</code><text>Authentication failed.</text></Reason>
</Acknowledgement_MarketDocument>`;

test("key gate: either env name activates; keyless = zero network calls", async () => {
  assert.equal(euLoadEnabled({} as any), false);
  assert.equal(euLoadEnabled({ ENTSOE_API_KEY: "x" } as any), true, "the name Mike set");
  assert.equal(euLoadEnabled({ ENTSOE_TOKEN: "x" } as any), true, "the name the wishlist proposed");
  let called = 0;
  const spy = async () => { called++; return { ok: true, status: 200, text: async () => "" }; };
  assert.deepEqual(await fetchLoad(spy as any, {} as any, 0, 0), []);
  assert.equal(called, 0);
});

test("url: verified param contract, UTC stamps, token encoded", () => {
  const start = Date.parse("2026-07-05T06:30:00Z");
  const end = Date.parse("2026-07-07T06:30:00Z");
  const u = loadUrl(ZONES.DE_LU, "se cret", start, end);
  assert.ok(u.includes("documentType=A65") && u.includes("processType=A16"), "realised total load");
  assert.ok(u.includes("outBiddingZone_Domain=10Y1001A1001A82H"));
  assert.ok(u.includes("periodStart=202607050630") && u.includes("periodEnd=202607070630"),
    "YYYYMMDDHHMM UTC stamps");
  assert.ok(u.includes("securityToken=se%20cret"), "token URL-encoded");
  assert.equal(periodStamp(Date.parse("2026-01-02T03:04:00Z")), "202601020304");
  assert.equal(Object.keys(ZONES).length, 8);
});

test("parseAck: live-verified acknowledgement shape; GL docs are not acks", () => {
  assert.equal(parseAck(ACK), "Authentication failed.");
  assert.equal(parseAck(GL(PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "50000"]]))), null);
});

test("parseLoad: PT60M and PT15M position math; published points only, never interpolated", () => {
  const xml = GL(
    PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "50000"], [2, "51000"], [4, "53000"]]) +
    PERIOD("2026-07-06T04:00Z", "PT15M", [[1, "49000"], [2, "49100"]])
  );
  const obs = parseLoad(xml, "FR", "2026-07-07");
  assert.equal(obs.length, 5);
  assert.deepEqual(obs[0], { zone: "FR", ts: "2026-07-06T00:00", mw: 50000, res: "PT60M", rt: "2026-07-07" });
  assert.equal(obs[2].ts, "2026-07-06T03:00", "position 4 lands 3h after start — the GAP (position 3) stays absent");
  assert.equal(obs[3].ts, "2026-07-06T04:00");
  assert.equal(obs[4].ts, "2026-07-06T04:15", "PT15M step");
  // junk resilience: unknown resolution and bad positions are skipped, never guessed
  assert.deepEqual(parseLoad(GL(PERIOD("2026-07-06T00:00Z", "PT7M", [[1, "1"]])), "FR", "x"), []);
  assert.equal(parseLoad(GL(PERIOD("2026-07-06T00:00Z", "PT60M", [[0, "1"], [1, ""]])), "FR", "x")[0].mw, null,
    "empty quantity stays null; position<1 dropped");
});

test("archive: value-aware dedup (vintage capture); day-file by observation day; revisions APPEND", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  const obs = parseLoad(GL(
    PERIOD("2026-07-05T23:00Z", "PT60M", [[1, "48000"], [2, "47000"]]) + // 23:00 + next-day 00:00
    PERIOD("2026-07-05T23:00Z", "PT15M", [[1, "48050"]])                  // same ts, finer res
  ), "DE_LU", "2026-07-07");
  assert.equal(archiveLoad(obs, base), 3);
  assert.equal(archiveLoad(obs, base), 0, "exact re-publication never re-archives");
  // the NL/IT leading-edge lesson (found live on day one): a REVISED value
  // for the same zone|ts|res is a new vintage and must append — a
  // value-free key would freeze the first partial number forever
  const revised = parseLoad(GL(PERIOD("2026-07-05T23:00Z", "PT15M", [[1, "51200"]])), "DE_LU", "2026-07-07");
  assert.equal(archiveLoad(revised, base), 1, "revision appends as a new vintage row");
  const dir = path.join(base, "euload");
  const lines = fs.readFileSync(path.join(dir, "2026-07-05.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const pt15 = lines.filter((r) => r.res === "PT15M" && r.ts === "2026-07-05T23:00");
  assert.deepEqual(pt15.map((r) => r.mw), [48050, 51200], "both vintages kept in arrival order — consumers take the last");
  assert.ok(fs.existsSync(path.join(dir, "2026-07-06.jsonl")), "hour 24 lands in the next UTC day");
  assert.ok(seedFileInWindow("2026-07-05.jsonl", Date.parse("2026-07-07")));
  assert.ok(!seedFileInWindow("2020-01-01.jsonl.gz", Date.parse("2026-07-07")),
    `seed window bounded to ${SEED_WINDOW_DAYS}d (heap protection, gridDemand precedent)`);
});

test("gz merge: a late revision after the day gz'd must merge, never overwrite", async () => {
  const { gzipOldLoadDays } = await import("./euLoad");
  const zlib = await import("node:zlib");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  const dir = path.join(base, "euload");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-06-01.jsonl"), '{"a":1}\n');
  assert.equal(gzipOldLoadDays(base, Date.parse("2026-07-07")), 1);
  // late vintage arrives after gz — appends a fresh plain file
  fs.writeFileSync(path.join(dir, "2026-06-01.jsonl"), '{"a":2}\n');
  assert.equal(gzipOldLoadDays(base, Date.parse("2026-07-07")), 1);
  const merged = zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-06-01.jsonl.gz"))).toString();
  assert.equal(merged, '{"a":1}\n{"a":2}\n', "prior gz rows survive the late-revision re-gz");
});

test("refresh sweep: one call per zone, per-zone stats + window shape, acked zone surfaced in issues", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  let calls = 0;
  const fake = async (url: string) => {
    calls++;
    if (url.includes(ZONES.SE)) return { ok: true, status: 200, text: async () => ACK }; // one zone acks
    return { ok: true, status: 200,
             text: async () => GL(PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "60000"], [2, "61000"]])) };
  };
  await refreshLoad(fake as any, { ENTSOE_API_KEY: "k" } as any,
                    Date.parse("2026-07-06T12:00:00Z"), base, 0);
  assert.equal(calls, Object.keys(ZONES).length);
  const hit = latestLoad();
  assert.ok(hit);
  assert.equal(hit!.stats.length, 7, "acked zone yields no rows — absent from stats, never zero-filled");
  assert.ok(hit!.stats.every((s) => s.latest_ts === "2026-07-06T11:00" && s.latest_mw === 61000
                                 && s.points_in_window === 2 && s.resolution === "PT60M"));
  // day-one lessons pinned: a missing zone must be self-diagnosing from
  // the route (DE_LU absent on the first prod cycle), and window shape
  // must expose series-level anomalies (NL 2.4GW latest point)
  assert.equal(hit!.issues.SE, "ack: Authentication failed.");
  assert.ok(!("FR" in hit!.issues), "healthy zones carry no issue entry");
  assert.ok(hit!.stats.every((s) => s.window_min_mw === 60000 && s.window_max_mw === 61000
                                 && s.window_mean_mw === 60500), "window min/max/mean computed");
});

test("refreshLoad: cold cache backfills from disk when every zone's live fetch fails — same class of fix as githubOrgActivity.ts's v1.0.881 cold-cache backfill (a transient ENTSO-E outage must not report warming_up over real archived load history)", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-coldcache-"));
  _resetEuLoadForTests();
  try {
    // Real archived history for FR from before the simulated restart.
    const priorObs = [{ zone: "FR", ts: "2026-07-05T10:00", mw: 55000, res: "PT60M", rt: "2026-07-05" }];
    assert.equal(archiveLoad(priorObs as any, base), 1);
    _resetEuLoadForTests(); // simulated restart — in-memory dedup/cache gone
    assert.equal(latestLoad(), null, "cache must still be cold going into this cycle");
    const failing = async () => { throw new Error("network unreachable"); };
    await refreshLoad(failing as any, { ENTSOE_API_KEY: "k" } as any,
                      Date.parse("2026-07-06T12:00:00Z"), base, 0);
    const hit = latestLoad();
    assert.ok(hit, "cache must be populated, not left null, despite every zone's live fetch failing");
    assert.equal(hit!.stats.length, 1, "acked/failed zones absent — never zero-filled — but FR's archived history survives");
    assert.equal(hit!.stats[0].zone, "FR");
    assert.equal(hit!.stats[0].latest_mw, 55000);
    const archived = readRecentArchivedLoad(base, Date.parse("2026-07-06T12:00:00Z"), 5);
    assert.equal(archived.length, 1, "readRecentArchivedLoad itself returns the raw archived rows the backfill used");
    assert.equal(archived[0].zone, "FR");
  } finally {
    _resetEuLoadForTests();
  }
});

// ── seen-set memory bound (2026-11-06, same mechanism as euGenerationMix) ───
// /api/health 2026-11-05: euLoad.seenObs held 279,670 keys because it kept the
// whole 120-day seed window plus everything since boot, while each sweep only
// fetches the last HOURS_PER_FETCH. These pin that the bound drops only keys no
// sweep can fetch again, and that dedup is unchanged inside the window.

const loadGauge = () => readHeapGauges()["euLoad.seenObs"] as Record<string, number>;
const hourLoad = (zone: string, startIso: string, hours: number, base = 40000) =>
  parseLoad(GL(PERIOD(startIso, "PT60M",
    Array.from({ length: hours }, (_, i) => [i + 1, String(base + i)] as [number, string]))), zone, startIso.slice(0, 10));

test("seen-set bound: the window covers the fetch window several times over, and the seed covers the window", () => {
  assert.ok(SEEN_RETAIN_DAYS * 24 >= 3 * HOURS_PER_FETCH,
    "a key is only dropped once it is >= 3 fetch windows older than the newest row");
  assert.ok(SEED_WINDOW_DAYS > SEEN_RETAIN_DAYS, "the boot seed must read every day the window can hold");
});

test("seen-set bound: keys older than the window are dropped; dedup inside the window is unchanged", () => {
  _resetEuLoadForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  const day1 = hourLoad("FR", "2026-07-01T00:00Z", 24);
  const day5 = hourLoad("FR", "2026-07-05T00:00Z", 24);
  assert.equal(archiveLoad(day1, base), 24);
  assert.equal(archiveLoad(day5, base), 24);
  assert.equal(loadGauge().entries, 48, "both days are inside the window of the newest row (07-05T23:00)");
  assert.equal(archiveLoad(day1, base), 0, "a re-publication inside the window still never re-archives");

  const day12 = hourLoad("FR", "2026-07-12T00:00Z", 24);
  assert.equal(archiveLoad(day12, base), 24);
  const g = loadGauge();
  assert.equal(g.entries, 25, "cutoff = newest 07-12T23:00 minus 7 days = 07-05T23:00: that one key + 07-12's 24 remain");
  assert.equal(g.pruned, 47, "all 24 keys of 07-01 and 07-05's 23 earlier hours are pruned");
  assert.equal(g.retain_days, SEEN_RETAIN_DAYS);
  assert.equal(archiveLoad(day12, base), 0, "the newest sweep re-published: still deduped");
  assert.equal(archiveLoad(day5.slice(-1), base), 0, "07-05T23:00 is exactly at the cutoff and still remembered");
  const lines = fs.readFileSync(path.join(base, "euload", "2026-07-12.jsonl"), "utf8").trim().split("\n");
  assert.equal(lines.length, 24, "no duplicate rows were written for the day the sweeps kept re-publishing");
});

test("seen-set bound: a row older than the window is still archived, and counted as late", () => {
  _resetEuLoadForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  assert.equal(archiveLoad(hourLoad("DE_LU", "2026-07-20T00:00Z", 2), base), 2);
  assert.equal(archiveLoad(hourLoad("DE_LU", "2026-07-01T00:00Z", 1), base), 1, "nothing is dropped from the archive");
  assert.ok(fs.existsSync(path.join(base, "euload", "2026-07-01.jsonl")));
  assert.equal(loadGauge().late_rows, 1, "a 19-day-old row is surfaced on /api/health, never silent");
  assert.equal(loadGauge().entries, 2, "and it does not stay in the set");
});

test("seen-set bound: one future-dated row cannot flush the window (reference capped at wall clock + 1 day)", () => {
  _resetEuLoadForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  const hourAgo = new Date(Math.floor((Date.now() - 3600_000) / 3600_000) * 3600_000).toISOString().slice(0, 16) + "Z";
  const recent = hourLoad("NL", hourAgo, 1);
  assert.equal(archiveLoad(recent, base), 1);
  assert.equal(archiveLoad(hourLoad("NL", "2099-01-01T00:00Z", 1), base), 1);
  assert.equal(archiveLoad(recent, base), 0,
    "the recent key survives: the cutoff is (now + 1 day) - 7 days, not 2099 - 7 days");
});

test("seen-set bound: the boot seed still dedups the days inside the window", () => {
  _resetEuLoadForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  const dir = path.join(base, "euload");
  fs.mkdirSync(dir, { recursive: true });
  const twoHoursAgo = new Date(Math.floor((Date.now() - 2 * 3600_000) / 3600_000) * 3600_000).toISOString().slice(0, 16) + "Z";
  const row = hourLoad("PL", twoHoursAgo, 1);
  fs.writeFileSync(path.join(dir, `${twoHoursAgo.slice(0, 10)}.jsonl`), row.map((r) => JSON.stringify(r)).join("\n") + "\n");
  assert.equal(archiveLoad(row, base), 0, "a row already on disk from today is seeded and never re-archived after a restart");
  assert.equal(loadGauge().entries, 1);
});
