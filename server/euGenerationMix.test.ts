// ENTSO-E actual-generation-per-type battery (wishlist 9c follow-up,
// built this session): key gate shared with euLoad, in_Domain param
// (not outBiddingZone_Domain), MktPSRType/psrType extraction per
// TimeSeries, unrecognized-psrType skip, archive dedup keyed with psr,
// poll stats grouped by zone|psr.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  euGenerationMixEnabled, ZONES, PSRTYPE_MAPPINGS, periodStamp, genMixUrl,
  parseAck, parseGenMix, fetchGenMix, archiveGenMix, refreshGenMix,
  latestGenMix, seedFileInWindow, SEED_WINDOW_DAYS,
} from "./euGenerationMix";

const GL = (series: string) => `<?xml version="1.0"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
${series}</GL_MarketDocument>`;
const SERIES = (psr: string | null, periods: string) => `<TimeSeries>
 <inBiddingZone_Domain.mRID>10YFR-RTE------C</inBiddingZone_Domain.mRID>
 ${psr ? `<MktPSRType><psrType>${psr}</psrType></MktPSRType>` : ""}
 ${periods}</TimeSeries>`;
const PERIOD = (start: string, res: string, points: [number, string][]) =>
  `<Period><timeInterval><start>${start}</start><end>x</end></timeInterval>
   <resolution>${res}</resolution>
   ${points.map(([p, q]) => `<Point><position>${p}</position><quantity>${q}</quantity></Point>`).join("")}
  </Period>`;
const ACK = `<?xml version="1.0"?><Acknowledgement_MarketDocument>
 <Reason><code>999</code><text>Authentication failed.</text></Reason>
</Acknowledgement_MarketDocument>`;

test("key gate: shared with euLoad; keyless = zero network calls", async () => {
  assert.equal(euGenerationMixEnabled({} as any), false);
  assert.equal(euGenerationMixEnabled({ ENTSOE_API_KEY: "x" } as any), true);
  assert.equal(euGenerationMixEnabled({ ENTSOE_TOKEN: "x" } as any), true);
  let called = 0;
  const spy = async () => { called++; return { ok: true, status: 200, text: async () => "" }; };
  assert.deepEqual(await fetchGenMix(spy as any, {} as any, 0, 0), []);
  assert.equal(called, 0);
});

test("url: A75/A16, in_Domain (not outBiddingZone_Domain), UTC stamps, token encoded", () => {
  const start = Date.parse("2026-07-05T06:30:00Z");
  const end = Date.parse("2026-07-07T06:30:00Z");
  const u = genMixUrl(ZONES.DE_LU, "se cret", start, end);
  assert.ok(u.includes("documentType=A75") && u.includes("processType=A16"));
  assert.ok(u.includes("in_Domain=10Y1001A1001A82H"));
  assert.ok(!u.includes("outBiddingZone_Domain"), "generation uses in_Domain, load uses outBiddingZone_Domain");
  assert.ok(u.includes("periodStart=202607050630") && u.includes("periodEnd=202607070630"));
  assert.ok(u.includes("securityToken=se%20cret"));
  assert.equal(periodStamp(Date.parse("2026-01-02T03:04:00Z")), "202601020304");
  assert.equal(Object.keys(ZONES).length, 8);
});

test("PSRTYPE_MAPPINGS: entsoe-py canonical table, spot-checked", () => {
  assert.equal(PSRTYPE_MAPPINGS.B16, "Solar");
  assert.equal(PSRTYPE_MAPPINGS.B19, "Wind Onshore");
  assert.equal(PSRTYPE_MAPPINGS.B14, "Nuclear");
  assert.equal(PSRTYPE_MAPPINGS.B04, "Fossil Gas");
  assert.equal(Object.keys(PSRTYPE_MAPPINGS).length, 28);
});

test("parseAck: shared acknowledgement shape; GL docs are not acks", () => {
  assert.equal(parseAck(ACK), "Authentication failed.");
  assert.equal(parseAck(GL(SERIES("B16", PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "500"]])))), null);
});

test("parseGenMix: per-series psrType tagging; multiple fuel series in one document", () => {
  const xml = GL(
    SERIES("B16", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "1200"], [2, "1500"]])) +
    SERIES("B19", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "3400"]]))
  );
  const obs = parseGenMix(xml, "FR", "2026-07-07");
  assert.equal(obs.length, 3);
  assert.deepEqual(obs.filter((o) => o.psr === "B16").map((o) => o.mw), [1200, 1500]);
  assert.deepEqual(obs.filter((o) => o.psr === "B19").map((o) => o.mw), [3400]);
  assert.ok(obs.every((o) => o.zone === "FR" && o.res === "PT60M"));
});

test("parseGenMix: missing/unrecognized psrType is skipped, never mis-attributed", () => {
  const noPsr = parseGenMix(GL(SERIES(null, PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "1"]]))), "FR", "x");
  assert.deepEqual(noPsr, []);
  const badPsr = parseGenMix(GL(SERIES("Z99", PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "1"]]))), "FR", "x");
  assert.deepEqual(badPsr, []);
});

test("parseGenMix: position math + junk resilience mirrors load parsing", () => {
  const obs = parseGenMix(GL(SERIES("B14",
    PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "900"], [4, "950"]]))), "FR", "2026-07-07");
  assert.equal(obs[0].ts, "2026-07-06T00:00");
  assert.equal(obs[1].ts, "2026-07-06T03:00", "position 4 lands 3h after start — gap stays absent");
  assert.equal(parseGenMix(GL(SERIES("B14", PERIOD("2026-07-06T00:00Z", "PT7M", [[1, "1"]]))), "FR", "x").length, 0);
});

test("archive: zone|psr|ts|res|VALUE dedup; revisions append as new vintage", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const obs = parseGenMix(GL(
    SERIES("B16", PERIOD("2026-07-05T23:00Z", "PT60M", [[1, "100"], [2, "110"]])) +
    SERIES("B19", PERIOD("2026-07-05T23:00Z", "PT60M", [[1, "200"]]))
  ), "DE_LU", "2026-07-07");
  assert.equal(archiveGenMix(obs, base), 3);
  assert.equal(archiveGenMix(obs, base), 0, "exact re-publication never re-archives");
  const revised = parseGenMix(GL(SERIES("B16", PERIOD("2026-07-05T23:00Z", "PT60M", [[1, "150"]]))), "DE_LU", "2026-07-07");
  assert.equal(archiveGenMix(revised, base), 1, "revision appends as a new vintage row");
  const dir = path.join(base, "eugenmix");
  const lines = fs.readFileSync(path.join(dir, "2026-07-05.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const b16 = lines.filter((r) => r.psr === "B16" && r.ts === "2026-07-05T23:00");
  assert.deepEqual(b16.map((r) => r.mw), [100, 150], "same zone|ts, different psr never collide; both B16 vintages kept");
  assert.ok(seedFileInWindow("2026-07-05.jsonl", Date.parse("2026-07-07")));
  assert.ok(!seedFileInWindow("2020-01-01.jsonl.gz", Date.parse("2026-07-07")),
    `seed window bounded to ${SEED_WINDOW_DAYS}d`);
});

test("refresh sweep: stats grouped by zone|psr, acked zone surfaced in issues", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  let calls = 0;
  const fake = async (url: string) => {
    calls++;
    if (url.includes(ZONES.SE)) return { ok: true, status: 200, text: async () => ACK };
    return {
      ok: true, status: 200,
      text: async () => GL(
        SERIES("B16", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "500"], [2, "600"]])) +
        SERIES("B04", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "1000"]]))
      ),
    };
  };
  await refreshGenMix(fake as any, { ENTSOE_API_KEY: "k" } as any,
                      Date.parse("2026-07-06T12:00:00Z"), base, 0);
  assert.equal(calls, Object.keys(ZONES).length);
  const hit = latestGenMix();
  assert.ok(hit);
  assert.equal(hit!.stats.length, 14, "7 healthy zones x 2 psr series each — acked zone yields none");
  const frSolar = hit!.stats.find((s) => s.zone === "FR" && s.psr === "B16")!;
  assert.equal(frSolar.psr_name, "Solar");
  assert.equal(frSolar.latest_mw, 600);
  assert.equal(frSolar.window_min_mw, 500);
  assert.equal(frSolar.window_mean_mw, 550);
  assert.equal(hit!.issues.SE, "ack: Authentication failed.");
  assert.ok(!("FR" in hit!.issues), "healthy zones carry no issue entry");
});
