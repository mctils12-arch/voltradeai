// ENTSO-E day-ahead price battery (wishlist 9c's last open follow-up):
// key gate shared with euLoad/euGenerationMix, A44/in+out_Domain param
// contract (no processType), Publication_MarketDocument root (NOT
// GL_MarketDocument), price.amount + currency_Unit.name/
// price_Measure_Unit.name extraction, negative-price handling, forward
// window (now-24h .. now+48h), archive dedup, poll stats.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  euDayAheadPricesEnabled, ZONES, periodStamp, priceUrl, parseAck, parsePrices,
  fetchPrices, archivePrices, refreshPrices, latestPrices, seedFileInWindow,
  SEED_WINDOW_DAYS, HOURS_BEFORE, HOURS_AFTER,
} from "./euDayAheadPrices";

// Mirrors the ENTSO-E Publication_MarketDocument shape (A44 price
// document) per the ENTSO-E API guide + entsoe-py cross-check: distinct
// root element from GL_MarketDocument (load/generation), currency/unit
// carried per TimeSeries, point value tag is price.amount.
const PUB = (series: string) => `<?xml version="1.0"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
${series}</Publication_MarketDocument>`;
const SERIES = (currency: string | null, unit: string | null, periods: string) => `<TimeSeries>
 <in_Domain.mRID>10YFR-RTE------C</in_Domain.mRID>
 ${currency ? `<currency_Unit.name>${currency}</currency_Unit.name>` : ""}
 ${unit ? `<price_Measure_Unit.name>${unit}</price_Measure_Unit.name>` : ""}
 ${periods}</TimeSeries>`;
const PERIOD = (start: string, res: string, points: [number, string][]) =>
  `<Period><timeInterval><start>${start}</start><end>x</end></timeInterval>
   <resolution>${res}</resolution>
   ${points.map(([p, a]) => `<Point><position>${p}</position><price.amount>${a}</price.amount></Point>`).join("")}
  </Period>`;
const ACK = `<?xml version="1.0"?><Acknowledgement_MarketDocument>
 <Reason><code>999</code><text>Authentication failed.</text></Reason>
</Acknowledgement_MarketDocument>`;

test("key gate: shared with euLoad/euGenerationMix; keyless = zero network calls", async () => {
  assert.equal(euDayAheadPricesEnabled({} as any), false);
  assert.equal(euDayAheadPricesEnabled({ ENTSOE_API_KEY: "x" } as any), true);
  assert.equal(euDayAheadPricesEnabled({ ENTSOE_TOKEN: "x" } as any), true);
  let called = 0;
  const spy = async () => { called++; return { ok: true, status: 200, text: async () => "" }; };
  assert.deepEqual(await fetchPrices(spy as any, {} as any, 0, 0), []);
  assert.equal(called, 0);
});

test("url: A44, in_Domain AND out_Domain both set (no processType), forward window, token encoded", () => {
  const now = Date.parse("2026-07-06T12:00:00Z");
  const u = priceUrl(ZONES.DE_LU, "se cret", now - HOURS_BEFORE * 3600_000, now + HOURS_AFTER * 3600_000);
  assert.ok(u.includes("documentType=A44"));
  assert.ok(!u.includes("processType"), "price document takes no processType, unlike load/generation");
  assert.ok(u.includes("in_Domain=10Y1001A1001A82H") && u.includes("out_Domain=10Y1001A1001A82H"));
  assert.ok(u.includes("periodStart=202607051200") && u.includes("periodEnd=202607081200"),
    "24h before .. 48h after now, not trailing-only like realised load/generation");
  assert.ok(u.includes("securityToken=se%20cret"), "token URL-encoded");
  assert.equal(periodStamp(Date.parse("2026-01-02T03:04:00Z")), "202601020304");
  assert.equal(Object.keys(ZONES).length, 8);
});

test("parseAck: shared acknowledgement shape; Publication docs are not acks", () => {
  assert.equal(parseAck(ACK), "Authentication failed.");
  assert.equal(parseAck(PUB(SERIES("EUR", "MWH", PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "45.20"]])))), null);
});

test("parsePrices: Publication_MarketDocument root required — a GL_MarketDocument (wrong doc type) yields nothing", () => {
  const gl = `<?xml version="1.0"?><GL_MarketDocument>${SERIES("EUR", "MWH",
    PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "45.20"]]))}</GL_MarketDocument>`;
  assert.deepEqual(parsePrices(gl, "FR", "x"), []);
});

test("parsePrices: currency/unit read per-series, price.amount extracted, negative prices preserved (never clamped)", () => {
  const xml = PUB(SERIES("EUR", "MWH",
    PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "45.20"], [2, "-3.50"]])));
  const obs = parsePrices(xml, "FR", "2026-07-07");
  assert.equal(obs.length, 2);
  assert.deepEqual(obs.map((o) => o.price), [45.2, -3.5], "negative day-ahead prices are real (oversupply) and preserved");
  assert.ok(obs.every((o) => o.zone === "FR" && o.currency === "EUR" && o.unit === "MWH" && o.res === "PT60M"));
});

test("parsePrices: missing currency/unit stored as empty string, never guessed as EUR/MWH", () => {
  const obs = parsePrices(PUB(SERIES(null, null, PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "10"]]))), "SE", "x");
  assert.equal(obs.length, 1);
  assert.equal(obs[0].currency, "");
  assert.equal(obs[0].unit, "");
});

test("parsePrices: position math + junk resilience mirrors load/generation parsing", () => {
  const obs = parsePrices(PUB(SERIES("EUR", "MWH",
    PERIOD("2026-07-06T00:00Z", "PT60M", [[1, "50"], [4, "55"]]))), "FR", "2026-07-07");
  assert.equal(obs[0].ts, "2026-07-06T00:00");
  assert.equal(obs[1].ts, "2026-07-06T03:00", "position 4 lands 3h after start — gap stays absent");
  assert.equal(parsePrices(PUB(SERIES("EUR", "MWH", PERIOD("2026-07-06T00:00Z", "PT7M", [[1, "1"]]))), "FR", "x").length, 0);
  assert.equal(parsePrices(PUB(SERIES("EUR", "MWH", PERIOD("2026-07-06T00:00Z", "PT60M", [[1, ""]]))), "FR", "x")[0].price, null,
    "empty price.amount stays null, never zero");
});

test("archive: zone|ts|res|VALUE dedup; day-file keyed by observation day, including FUTURE (tomorrow's published auction)", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euprices-"));
  const obs = parsePrices(PUB(SERIES("EUR", "MWH",
    PERIOD("2026-07-08T23:00Z", "PT60M", [[1, "60"], [2, "65"]]))), "DE_LU", "2026-07-07");
  assert.equal(archivePrices(obs, base), 2, "tomorrow's already-published auction result archives today");
  assert.equal(archivePrices(obs, base), 0, "exact re-publication never re-archives");
  const revised = parsePrices(PUB(SERIES("EUR", "MWH", PERIOD("2026-07-08T23:00Z", "PT60M", [[1, "58"]]))), "DE_LU", "2026-07-07");
  assert.equal(archivePrices(revised, base), 1, "a corrected republication appends as a new vintage row rather than overwriting");
  const dir = path.join(base, "eudayaheadprices");
  const lines = fs.readFileSync(path.join(dir, "2026-07-08.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.filter((r) => r.ts === "2026-07-08T23:00").length, 2, "both vintages kept in arrival order");
  assert.ok(seedFileInWindow("2026-07-05.jsonl", Date.parse("2026-07-07")));
  assert.ok(!seedFileInWindow("2020-01-01.jsonl.gz", Date.parse("2026-07-07")),
    `seed window bounded to ${SEED_WINDOW_DAYS}d`);
});

test("gz merge: future/near-term day-files (including not-yet-arrived days) are never gz'd early", async () => {
  const { gzipOldPriceDays } = await import("./euDayAheadPrices");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euprices-"));
  const dir = path.join(base, "eudayaheadprices");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-07-08.jsonl"), '{"a":1}\n'); // tomorrow relative to "now" below
  fs.writeFileSync(path.join(dir, "2026-06-01.jsonl"), '{"a":2}\n'); // long past
  assert.equal(gzipOldPriceDays(base, Date.parse("2026-07-07T12:00:00Z")), 1, "only the >=3-day-old file gz's");
  assert.ok(fs.existsSync(path.join(dir, "2026-07-08.jsonl")), "tomorrow's file stays plain, not gz'd");
  assert.ok(fs.existsSync(path.join(dir, "2026-06-01.jsonl.gz")));
});

test("refresh sweep: forward window covers not-yet-published tomorrow honestly, acked zone surfaced in issues, negative-price count", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "euprices-"));
  let calls = 0;
  const fake = async (url: string) => {
    calls++;
    if (url.includes(ZONES.SE)) return { ok: true, status: 200, text: async () => ACK };
    return {
      ok: true, status: 200,
      text: async () => PUB(SERIES("EUR", "MWH",
        PERIOD("2026-07-06T10:00Z", "PT60M", [[1, "40"], [2, "-5"]]))),
    };
  };
  await refreshPrices(fake as any, { ENTSOE_API_KEY: "k" } as any,
                      Date.parse("2026-07-06T12:00:00Z"), base, 0);
  assert.equal(calls, Object.keys(ZONES).length);
  const hit = latestPrices();
  assert.ok(hit);
  assert.equal(hit!.stats.length, 7, "acked zone yields no rows — absent from stats, never zero-filled");
  const fr = hit!.stats.find((s) => s.zone === "FR")!;
  assert.equal(fr.latest_price, -5, "position 2 (later timestamp) is latest, even though it is the negative one");
  assert.equal(fr.window_min_price, -5);
  assert.equal(fr.window_max_price, 40);
  assert.equal(fr.window_mean_price, 17.5);
  assert.equal(fr.negative_price_points, 1);
  assert.equal(fr.currency, "EUR");
  assert.equal(fr.unit, "MWH");
  assert.equal(hit!.issues.SE, "ack: Authentication failed.");
  assert.ok(!("FR" in hit!.issues), "healthy zones carry no issue entry");
});
