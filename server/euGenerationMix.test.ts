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
import zlib from "node:zlib";
import {
  euGenerationMixEnabled, ZONES, PSRTYPE_MAPPINGS, periodStamp, genMixUrl,
  parseAck, parseGenMix, fetchGenMix, archiveGenMix, refreshGenMix,
  latestGenMix, seedFileInWindow, SEED_WINDOW_DAYS,
  computeGenMixStats, readArchivedGenMix, _resetGenMixForTests,
  SEEN_RETAIN_DAYS, HOURS_PER_FETCH,
} from "./euGenerationMix";
import { readHeapGauges } from "./heapDiag";

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
  _resetGenMixForTests();
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

// ── cold-cache-no-disk-backfill fix thread (research/open_questions.md's
// module audit table) — a cold boot or a live ENTSO-E outage (every zone
// acking/erroring) left /api/data/eu-generation-mix warming_up forever
// despite a real per-day archive already on disk. A/B-verified against
// the pre-fix shape: the old refresh only ever wrote `cache` inside `if
// (obs.length)`, so an all-zones-empty sweep on a cold boot left `cache`
// permanently null even with archived days sitting right there.

test("computeGenMixStats: pure aggregation, reused for both a live sweep and a backfilled archive day", () => {
  const obs = parseGenMix(GL(
    SERIES("B16", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "500"], [2, "600"]])) +
    SERIES("B04", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "1000"]]))
  ), "FR", "2026-07-07");
  const stats = computeGenMixStats(obs);
  assert.equal(stats.length, 2);
  const solar = stats.find((s) => s.psr === "B16")!;
  assert.equal(solar.latest_mw, 600);
  assert.equal(solar.window_min_mw, 500);
  assert.equal(solar.window_mean_mw, 550);
});

test("readArchivedGenMix: returns the most recent archived day's raw obs, not a merged multi-day window", () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const day1 = parseGenMix(GL(SERIES("B16", PERIOD("2026-07-05T10:00Z", "PT60M", [[1, "100"]]))), "FR", "2026-07-05");
  const day2 = parseGenMix(GL(SERIES("B16", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "200"]]))), "FR", "2026-07-06");
  archiveGenMix(day1, base);
  archiveGenMix(day2, base);
  const obs = readArchivedGenMix(base, Date.parse("2026-07-07T00:00:00Z"));
  assert.ok(obs.length > 0);
  assert.ok(obs.every((o) => o.ts.startsWith("2026-07-06")), "walks back to the newest day with an archive file, not a blend of both days");
});

test("readArchivedGenMix: gzipped days are read too, and an empty archive returns []", () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  assert.deepEqual(readArchivedGenMix(base, Date.parse("2026-07-07T00:00:00Z")), [], "no archive on disk at all");
  const day = parseGenMix(GL(SERIES("B16", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "200"]]))), "FR", "2026-07-06");
  archiveGenMix(day, base);
  const dir = path.join(base, "eugenmix");
  fs.readdirSync(dir).forEach((f) => {
    if (!f.endsWith(".jsonl")) return;
    const fp = path.join(dir, f);
    fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
    fs.unlinkSync(fp);
  });
  const obs = readArchivedGenMix(base, Date.parse("2026-07-07T00:00:00Z"));
  assert.equal(obs.length, day.length);
});

test("refresh: a cold cache backfills from disk when every zone acks/errors, aggregated exactly like a live result", async () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const day = parseGenMix(GL(
    SERIES("B16", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "500"], [2, "600"]]))
  ), "FR", "2026-07-06");
  archiveGenMix(day, base);
  assert.equal(latestGenMix(), null, "pre-fix baseline: cache starts cold");

  const allAck = async () => ({ ok: true, status: 200, text: async () => ACK });
  await refreshGenMix(allAck as any, { ENTSOE_API_KEY: "k" } as any,
                      Date.parse("2026-07-07T12:00:00Z"), base, 0);
  const hit = latestGenMix();
  assert.ok(hit, "cold cache backfilled from the on-disk archive instead of staying warming_up forever");
  assert.equal(hit!.stats.length, 1);
  assert.equal(hit!.stats[0].zone, "FR");
  assert.equal(hit!.stats[0].latest_mw, 600, "backfilled rows flow through the same computeGenMixStats aggregation as a live result");
});

test("refresh: an already-good cache is never clobbered by a transient all-zones-empty sweep", async () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const ok = async (url: string) => {
    if (url.includes(ZONES.SE)) return { ok: true, status: 200, text: async () => "" };
    return { ok: true, status: 200, text: async () => GL(SERIES("B16", PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "500"]]))) };
  };
  await refreshGenMix(ok as any, { ENTSOE_API_KEY: "k" } as any,
                      Date.parse("2026-07-06T12:00:00Z"), base, 0);
  const first = latestGenMix();
  assert.ok(first);

  const allAck = async () => ({ ok: true, status: 200, text: async () => ACK });
  await refreshGenMix(allAck as any, { ENTSOE_API_KEY: "k" } as any,
                      Date.parse("2026-07-06T18:00:00Z"), base, 0);
  assert.deepEqual(latestGenMix(), first, "a transient all-acked sweep with a good cache already in hand leaves it untouched, not overwritten by an archive re-read");
});

test("refresh: cold cache with nothing archived either stays honestly null, never fabricates a result", async () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const allAck = async () => ({ ok: true, status: 200, text: async () => ACK });
  await refreshGenMix(allAck as any, { ENTSOE_API_KEY: "k" } as any,
                      Date.parse("2026-07-07T12:00:00Z"), base, 0);
  assert.equal(latestGenMix(), null, "no archive and no live result means warming_up is the honest state, not a fabricated one");
});

// ── seen-set memory bound (2026-11-06 dead-code/memory audit) ───────────────
// /api/health 2026-11-05: euGenerationMix.seenObs held 1,091,835 keys (~190 MB)
// because it kept the whole 120-day seed window plus everything since boot,
// while each sweep only fetches the last HOURS_PER_FETCH. These pin that the
// bound drops only keys no sweep can fetch again, and that dedup is unchanged
// inside the window.

const seenGauge = () => readHeapGauges()["euGenerationMix.seenObs"] as Record<string, number>;
const hourObs = (zone: string, startIso: string, hours: number, base = 100) =>
  parseGenMix(GL(SERIES("B16", PERIOD(startIso, "PT60M",
    Array.from({ length: hours }, (_, i) => [i + 1, String(base + i)] as [number, string])))), zone, startIso.slice(0, 10));

test("seen-set bound: the window covers the fetch window several times over, and the seed covers the window", () => {
  assert.ok(SEEN_RETAIN_DAYS * 24 >= 3 * HOURS_PER_FETCH,
    "a key is only dropped once it is >= 3 fetch windows older than the newest row");
  assert.ok(SEED_WINDOW_DAYS > SEEN_RETAIN_DAYS, "the boot seed must read every day the window can hold");
});

test("seen-set bound: keys older than the window are dropped; dedup inside the window is unchanged", () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const day1 = hourObs("FR", "2026-07-01T00:00Z", 24);
  const day5 = hourObs("FR", "2026-07-05T00:00Z", 24);
  assert.equal(archiveGenMix(day1, base), 24);
  assert.equal(archiveGenMix(day5, base), 24);
  assert.equal(seenGauge().entries, 48, "both days are inside the window of the newest row (07-05T23:00)");
  assert.equal(archiveGenMix(day1, base), 0, "a re-publication inside the window still never re-archives");

  const day12 = hourObs("FR", "2026-07-12T00:00Z", 24);
  assert.equal(archiveGenMix(day12, base), 24);
  const g = seenGauge();
  assert.equal(g.entries, 25, "cutoff = newest 07-12T23:00 minus 7 days = 07-05T23:00: that one key + 07-12's 24 remain");
  assert.equal(g.pruned, 47, "all 24 keys of 07-01 and 07-05's 23 earlier hours are pruned");
  assert.equal(g.retain_days, SEEN_RETAIN_DAYS);
  assert.equal(archiveGenMix(day12, base), 0, "the newest sweep re-published: still deduped");
  assert.equal(archiveGenMix(day5.slice(-1), base), 0, "07-05T23:00 is exactly at the cutoff and still remembered");
  const lines = fs.readFileSync(path.join(base, "eugenmix", "2026-07-12.jsonl"), "utf8").trim().split("\n");
  assert.equal(lines.length, 24, "no duplicate rows were written for the day the sweeps kept re-publishing");
});

test("seen-set bound: a row older than the window is still archived, and counted as late", () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  assert.equal(archiveGenMix(hourObs("DE_LU", "2026-07-20T00:00Z", 2), base), 2);
  const stale = hourObs("DE_LU", "2026-07-01T00:00Z", 1);
  assert.equal(archiveGenMix(stale, base), 1, "nothing is dropped from the archive");
  assert.ok(fs.existsSync(path.join(base, "eugenmix", "2026-07-01.jsonl")));
  assert.equal(seenGauge().late_rows, 1, "a 19-day-old row is surfaced on /api/health, never silent");
  assert.equal(seenGauge().entries, 2, "and it does not stay in the set");
});

test("seen-set bound: one future-dated row cannot flush the window (reference capped at wall clock + 1 day)", () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const hourAgo = new Date(Math.floor((Date.now() - 3600_000) / 3600_000) * 3600_000).toISOString().slice(0, 16) + "Z";
  const recent = hourObs("NL", hourAgo, 1);
  assert.equal(archiveGenMix(recent, base), 1);
  assert.equal(archiveGenMix(hourObs("NL", "2099-01-01T00:00Z", 1), base), 1);
  assert.equal(archiveGenMix(recent, base), 0,
    "the recent key survives: the cutoff is (now + 1 day) - 7 days, not 2099 - 7 days");
});

test("seen-set bound: the boot seed still dedups the days inside the window", () => {
  _resetGenMixForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eugenmix-"));
  const dir = path.join(base, "eugenmix");
  fs.mkdirSync(dir, { recursive: true });
  const todayHour = new Date(Math.floor((Date.now() - 2 * 3600_000) / 3600_000) * 3600_000).toISOString().slice(0, 16) + "Z";
  const row = hourObs("PL", todayHour, 1);
  fs.writeFileSync(path.join(dir, `${todayHour.slice(0, 10)}.jsonl`), row.map((r) => JSON.stringify(r)).join("\n") + "\n");
  assert.equal(archiveGenMix(row, base), 0, "a row already on disk from today is seeded and never re-archived after a restart");
  assert.equal(seenGauge().entries, 1);
});
