import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DtppStore, PlateCache, chartsForProcedure, dtppMetafileUrl, dtppPdfUrl, isPdf, matchApproachChart, matchRouteCharts,
  parseDtppMetafile, parseRouteChartName,
} from "./dtpp";
import type { R2Client } from "./r2Client";

// Real d-TPP metafile excerpt (cycle 2609): KAUS (all 40 charts, record
// fields trimmed to the ones we read) + one airport with no ICAO ident.
const here = path.dirname(fileURLToPath(import.meta.url));
const XML = fs.readFileSync(path.join(here, "__fixtures__", "dtpp_metafile_sample.xml"), "utf-8");
const meta = parseDtppMetafile(XML);
const kaus = meta.airports.get("KAUS")!;
const nav = (id: string) => (id === "CWK" ? "CENTEX" : null);

test("metafile: cycle + dates, airports keyed by ICAO and FAA ident", () => {
  assert.equal(meta.cycle, "2609");
  assert.equal(meta.from, "0901Z  09/03/26");
  assert.equal(meta.to, "0901Z  10/01/26");
  assert.equal(meta.airports.get("AUS"), kaus);
  assert.equal(kaus.icao, "KAUS");
  assert.equal(kaus.name, "AUSTIN-BERGSTROM INTL");
  const small = meta.airports.get("79J")!;
  assert.equal(small.icao, null);
  assert.ok(small.charts.length > 0);
  const il = kaus.charts.find((c) => c.pdf === "00556IL18L.PDF")!;
  assert.deepEqual([il.code, il.name, il.amdt, il.amdtDate], ["IAP", "ILS OR LOC RWY 18L", "4B", "09/05/2024"]);
  assert.equal(kaus.charts.filter((c) => c.code === "IAP").length, 15);
  assert.equal(kaus.charts.find((c) => c.code === "APD")?.pdf, "00556AD.PDF");
});

test("route chart names: number words, (RNAV), continuation pages", () => {
  assert.deepEqual(parseRouteChartName("AUSTIN SEVEN"), { base: "AUSTIN", digit: "7", cont: false });
  assert.deepEqual(parseRouteChartName("DXEEE THREE (RNAV)"), { base: "DXEEE", digit: "3", cont: false });
  assert.deepEqual(parseRouteChartName("WLEEE SEVEN (RNAV), CONT.1"), { base: "WLEEE", digit: "7", cont: true });
});

test("SID/STAR ident -> chart: letters, abbreviation (AUS->AUSTIN), navaid name (CWK->CENTEX), continuation pages last", () => {
  assert.deepEqual(matchRouteCharts("SID", "AUS7", kaus.charts, nav).map((c) => c.pdf), ["00556AUSTIN.PDF", "00556AUSTIN_C.PDF"]);
  assert.deepEqual(matchRouteCharts("SID", "CWK8", kaus.charts, nav).map((c) => c.pdf), ["00556CENTEX.PDF", "00556CENTEX_C.PDF"]);
  assert.deepEqual(matchRouteCharts("STAR", "BLEWE5", kaus.charts, nav).map((c) => c.pdf), ["00556BLEWE.PDF"]);
  assert.deepEqual(matchRouteCharts("STAR", "DXEEE3", kaus.charts, nav).map((c) => c.pdf), ["00556DXEEE.PDF"]);
  assert.deepEqual(matchRouteCharts("STAR", "BLEWE4", kaus.charts, nav), [], "version digit must match");
  assert.deepEqual(matchRouteCharts("SID", "BLEWE5", kaus.charts, nav), [], "a STAR chart never answers a SID");
});

test("approach ident -> IAP chart: type, runway, variant; plain chart beats SA CAT / CAT II-III", () => {
  const pdf = (id: string) => matchApproachChart(id, kaus.charts)?.pdf ?? null;
  assert.equal(pdf("I18L"), "00556IL18L.PDF");
  assert.equal(pdf("L18L"), "00556IL18L.PDF");
  assert.equal(pdf("I36R"), "00556IL36R.PDF");
  assert.equal(pdf("R18LY"), "00556RY18L.PDF");
  assert.equal(pdf("H36RZ"), "00556RRZ36R.PDF");
  assert.equal(pdf("R18LZ"), null, "no RNAV (GPS) Z 18L chart exists");
  assert.equal(pdf("V18L"), null);
  assert.deepEqual(chartsForProcedure("IAP", "I18L", kaus.charts, nav).map((c) => c.name), ["ILS OR LOC RWY 18L"]);
});

test("urls", () => {
  assert.equal(dtppMetafileUrl("2609"), "https://aeronav.faa.gov/d-tpp/2609/xml_data/d-tpp_Metafile.xml");
  assert.equal(dtppPdfUrl("2609", "00556IL18L.PDF"), "https://aeronav.faa.gov/d-tpp/2609/00556IL18L.PDF");
});

test("DtppStore: current cycle 404 -> previous cycle, with backoff", async () => {
  const urls: string[] = [];
  const fetchImpl = (async (u: string | URL | Request) => {
    urls.push(String(u));
    return String(u).includes("/2609/") ? new Response("no", { status: 404 }) : new Response(XML.replace('cycle="2609"', 'cycle="2608"'), { status: 200 });
  }) as typeof fetch;
  const s = new DtppStore({ fetchImpl, now: () => Date.UTC(2026, 8, 30) });
  const got = await s.get();
  assert.equal(got.cycle.ident, "2608");
  assert.ok(got.idx.airports.has("KAUS"));
  await s.get();
  assert.equal(urls.length, 2, "backoff: the fallback is served without re-trying the current cycle each call");
});

const PDF_BODY = Buffer.from("%PDF-1.4\n% synthetic\n");
const fakePdfFetch = (calls: string[], body: Buffer = PDF_BODY) => (async (u: string | URL | Request) => {
  calls.push(String(u));
  return new Response(body, { status: 200 });
}) as typeof fetch;

test("PlateCache (tmp): FAA once, then /tmp; bounded LRU eviction; non-PDF refused", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "plates-test-"));
  const calls: string[] = [];
  const pc = new PlateCache({ fetchImpl: fakePdfFetch(calls), r2: null, dir, maxFiles: 2 });
  assert.equal(pc.backend, "tmp");
  const a = await pc.get("2609", "00556IL18L.PDF");
  assert.equal(a.source, "faa");
  assert.ok(isPdf(a.body));
  const b = await pc.get("2609", "00556il18l.pdf");
  assert.equal(b.source, "tmp");
  assert.equal(calls.length, 1);
  await pc.get("2609", "00556AD.PDF");
  await new Promise((r) => setTimeout(r, 20));
  await pc.get("2609", "00556RY18L.PDF");
  assert.ok(fs.readdirSync(dir).length <= 2, "evicted down to maxFiles");
  assert.ok(pc.counters.evicted >= 1);
  await assert.rejects(pc.get("2609", "../../etc/passwd"), /bad plate name/);
  const bad = new PlateCache({ fetchImpl: fakePdfFetch([], Buffer.from("<html>nope</html>")), r2: null, dir });
  await assert.rejects(bad.get("2609", "00556XX.PDF"), /non-PDF/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("PlateCache (R2): hit served from R2; miss fetched from FAA and put under plates/<cycle>/<pdf>", async () => {
  const store = new Map<string, Buffer>();
  const puts: string[] = [];
  const r2 = {
    configured: true,
    async getObject(key: string) { const b = store.get(key); return b ? { ok: true, body: b } : { ok: false, status: 404 }; },
    async putObject(key: string, body: Buffer | string) { puts.push(key); store.set(key, Buffer.from(body)); return { ok: true }; },
  } as unknown as R2Client;
  const calls: string[] = [];
  const pc = new PlateCache({ fetchImpl: fakePdfFetch(calls), r2 });
  assert.equal(pc.backend, "r2");
  assert.equal((await pc.get("2609", "00556IL18L.PDF")).source, "faa");
  assert.deepEqual(puts, ["plates/2609/00556IL18L.PDF"]);
  assert.equal((await pc.get("2609", "00556IL18L.PDF")).source, "r2");
  assert.equal(calls.length, 1);
});
