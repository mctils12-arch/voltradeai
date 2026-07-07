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
  SEED_WINDOW_DAYS,
} from "./euLoad";

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

test("archive: zone|ts|res dedup; day-file by observation day; PT15M vs PT60M same ts are distinct", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euload-"));
  const obs = parseLoad(GL(
    PERIOD("2026-07-05T23:00Z", "PT60M", [[1, "48000"], [2, "47000"]]) + // 23:00 + next-day 00:00
    PERIOD("2026-07-05T23:00Z", "PT15M", [[1, "48050"]])                  // same ts, finer res
  ), "DE_LU", "2026-07-07");
  assert.equal(archiveLoad(obs, base), 3);
  assert.equal(archiveLoad(obs, base), 0, "never re-archives");
  const dir = path.join(base, "euload");
  assert.ok(fs.existsSync(path.join(dir, "2026-07-05.jsonl")));
  assert.ok(fs.existsSync(path.join(dir, "2026-07-06.jsonl")), "hour 24 lands in the next UTC day");
  assert.ok(seedFileInWindow("2026-07-05.jsonl", Date.parse("2026-07-07")));
  assert.ok(!seedFileInWindow("2020-01-01.jsonl.gz", Date.parse("2026-07-07")),
    `seed window bounded to ${SEED_WINDOW_DAYS}d (heap protection, gridDemand precedent)`);
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
