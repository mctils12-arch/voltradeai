import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import express from "express";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";
import { CifpIndex, CifpStore, cycleAt } from "./cifp";
import { DtppStore, PlateCache, parseDtppMetafile } from "./dtpp";
import {
  APPROACH_HONESTY, NOT_FOR_NAVIGATION, findFiledProcedures, parseMetarWind, registerProcedureRoutes, runwayHeadings,
  runwayWinds, suggestApproaches, summarize, type Wind,
} from "./procedures";

const here = path.dirname(fileURLToPath(import.meta.url));
const CIFP = fs.readFileSync(path.join(here, "__fixtures__", "cifp_kaus_sample.txt"));
const META = fs.readFileSync(path.join(here, "__fixtures__", "dtpp_metafile_sample.xml"), "utf-8");
const idx = CifpIndex.build(CIFP);
const kaus = idx.airport("KAUS")!;
const dtpp = parseDtppMetafile(META);
const nav = (id: string) => idx.navaidName(id);
const summaries = kaus.procedures.map((p) => summarize(kaus, p, dtpp.airports.get("KAUS")!, nav));

// ── filed DP / STAR ─────────────────────────────────────────────────────────
test("filed route text names the DP (+ enroute transition after it) and the STAR (+ transition before it)", () => {
  const m = findFiledProcedures("KAUS.AUS7.JCT J86 ELP..ACT.BLEWE5.KAUS", kaus, kaus);
  assert.deepEqual(m.dp, { id: "AUS7", transition: "JCT", filedAs: null, versionMismatch: false });
  assert.deepEqual(m.star, { id: "BLEWE5", transition: "ACT", filedAs: null, versionMismatch: false });
});

test("filed route: older/newer procedure version is reported as a mismatch, never silently promoted", () => {
  const m = findFiledProcedures("KDFW DCT CWK BLEWE4 KAUS", null, kaus);
  assert.equal(m.dp, null);
  assert.deepEqual(m.star, { id: "BLEWE5", transition: null, filedAs: "BLEWE4", versionMismatch: true });
  assert.deepEqual(findFiledProcedures(null, kaus, kaus), { dp: null, star: null });
  assert.deepEqual(findFiledProcedures("KAUS DCT KDFW", kaus, kaus), { dp: null, star: null });
});

// ── approach suggestions ────────────────────────────────────────────────────
test("runway true headings come from the coded thresholds (18L ~179°, 36R ~359°)", () => {
  const h = new Map(runwayHeadings(kaus).map((r) => [r.runway, r.headingTrue]));
  assert.ok(Math.abs(h.get("18L")! - 178.7) < 1, String(h.get("18L")));
  assert.ok(Math.abs(h.get("36R")! - 358.7) < 1, String(h.get("36R")));
});

test("METAR JSON -> wind; VRB direction is null", () => {
  const w = parseMetarWind([{ icaoId: "KAUS", wdir: 160, wspd: 9, obsTime: 1790747580, rawOb: "METAR KAUS 300553Z 16009KT" }])!;
  assert.deepEqual([w.dirDeg, w.speedKt, w.raw], [160, 9, "METAR KAUS 300553Z 16009KT"]);
  assert.equal(w.obsTime, new Date(1790747580 * 1000).toISOString());
  assert.equal(parseMetarWind([{ wdir: "VRB", wspd: 3 }])!.dirDeg, null);
  assert.equal(parseMetarWind([]), null);
});

test("suggestions: south wind favours RWY 18L; a north wind drops it (tailwind); calm keeps it, labelled", () => {
  const south: Wind = { dirDeg: 160, speedKt: 9, gustKt: null, obsTime: null, raw: null };
  const s = suggestApproaches(kaus, south, summaries);
  assert.equal(s[0].id, "I18L");
  assert.equal(s[0].name, "ILS OR LOC RWY 18L");
  assert.match(s[0].reason, /RWY 18L: 9 kt headwind, 3 kt crosswind/);
  assert.equal(s[0].charts[0].pdf, "00556IL18L.PDF");
  assert.deepEqual(suggestApproaches(kaus, { ...south, dirDeg: 340 }, summaries), []);
  const calm = suggestApproaches(kaus, { ...south, speedKt: 2 }, summaries);
  assert.match(calm[0].reason, /light\/variable/);
  const none = suggestApproaches(kaus, null, summaries);
  assert.match(none[0].reason, /no current wind/);
  const rw = runwayWinds(kaus, south).find((r) => r.runway === "18L")!;
  assert.ok(rw.headwindKt! > 8 && rw.crosswindKt! < 4);
});

// ── endpoints ───────────────────────────────────────────────────────────────
function synthPdf(): Buffer {
  const content = Buffer.from("BT /F 6 Tf 1 0 0 1 10 10 Tm (NOT A REAL PLATE) Tj ET", "latin1");
  return Buffer.from(`%PDF-1.4\n1 0 obj\n<< /Type /Page /MediaBox [0 0 400 600] /Contents 2 0 R >>\nendobj\n2 0 obj\n<< /Length ${content.length} >>\nstream\n${content.toString("latin1")}\nendstream\nendobj\n%%EOF\n`, "latin1");
}

async function withServer(fn: (base: string, calls: string[]) => Promise<void>) {
  const now = Date.UTC(2026, 8, 30, 12);
  const cycle = cycleAt(now);
  const cifp = new CifpStore({ now: () => now, fetchImpl: (async () => { throw new Error("no network in tests"); }) as typeof fetch });
  cifp._setIndex(idx, cycle);
  const d = new DtppStore({ now: () => now, fetchImpl: (async () => { throw new Error("no network in tests"); }) as typeof fetch });
  d._setIndex(dtpp, cycle);
  const calls: string[] = [];
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "proc-routes-"));
  const plates = new PlateCache({ r2: null, dir, fetchImpl: (async (u: string | URL | Request) => { calls.push(String(u)); return new Response(synthPdf(), { status: 200 }); }) as typeof fetch });
  const app = express();
  registerProcedureRoutes(app, {
    cifp, dtpp: d, plates, now: () => now,
    filedPlan: (cs) => (cs === "SWA1234" ? { departure: "KAUS", arrival: "KAUS", routeText: "KAUS.AUS7.JCT J86 ACT.BLEWE5.KAUS", updatedAt: now } : null),
    metar: async () => ({ dirDeg: 160, speedKt: 9, gustKt: null, obsTime: "2026-09-30T06:00:00.000Z", raw: "METAR KAUS 300553Z 16009KT" }),
  });
  const srv = app.listen(0);
  await new Promise<void>((r) => srv.once("listening", () => r()));
  try {
    await fn(`http://127.0.0.1:${(srv.address() as AddressInfo).port}`, calls);
  } finally {
    srv.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

test("GET /api/data/procedures/:airport lists SIDs/STARs/approaches with charts, cycles and honesty", async () => {
  await withServer(async (base) => {
    const r = await fetch(`${base}/api/data/procedures/AUS`);
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.equal(j.airport.icao, "KAUS");
    assert.equal(j.airport.faa, "AUS");
    assert.equal(j.cycle.ident, "2609");
    assert.equal(j.dtppCycle.ident, "2609");
    assert.deepEqual(j.sids.map((p: { id: string }) => p.id), ["AUS7"]);
    assert.equal(j.sids[0].name, "AUSTIN SEVEN");
    assert.equal(j.sids[0].charts.length, 2);
    assert.equal(j.stars[0].charts[0].url, "/api/data/plates/KAUS/00556BLEWE.PDF");
    assert.equal(j.approaches[0].id, "I18L");
    assert.equal(j.approaches[0].runway, "18L");
    assert.equal(j.charts.airportDiagram.pdf, "00556AD.PDF");
    assert.equal(j.notForNavigation, NOT_FOR_NAVIGATION);
    assert.equal(j.approachHonesty, APPROACH_HONESTY);
    assert.equal((await fetch(`${base}/api/data/procedures/KXYZ`)).status, 404);
    assert.equal((await fetch(`${base}/api/data/procedures/%3Cx%3E`)).status, 400);
  });
});

test("GET …/:procId/path returns GeoJSON legs + fixes with meta", async () => {
  await withServer(async (base) => {
    const j = await (await fetch(`${base}/api/data/procedures/KAUS/I18L/path?transition=HOUKM`)).json();
    assert.equal(j.type, "FeatureCollection");
    assert.equal(j.meta.procedure, "I18L");
    assert.equal(j.meta.selectedTransition, "HOUKM");
    assert.ok(j.features.some((f: { properties: { kind: string } }) => f.properties.kind === "leg"));
    assert.ok(j.features.every((f: { properties: { transition: string | null } }) => f.properties.transition !== "DOFFS"));
    assert.equal(j.source, "FAA CIFP (ARINC 424)");
    assert.equal((await fetch(`${base}/api/data/procedures/KAUS/NOPE1/path`)).status, 404);
  });
});

test("GET /api/data/plates/:airport/:pdf proxies only charts the d-TPP lists, then serves from cache", async () => {
  await withServer(async (base, calls) => {
    const r = await fetch(`${base}/api/data/plates/KAUS/00556IL18L.PDF`);
    assert.equal(r.status, 200);
    assert.equal(r.headers.get("content-type"), "application/pdf");
    assert.equal(r.headers.get("x-plate-cycle"), "2609");
    assert.equal(r.headers.get("x-plate-source"), "faa");
    assert.ok(Buffer.from(await r.arrayBuffer()).toString("latin1").startsWith("%PDF-"));
    const again = await fetch(`${base}/api/data/plates/KAUS/00556IL18L.PDF`);
    assert.equal(again.headers.get("x-plate-source"), "tmp");
    assert.deepEqual(calls, ["https://aeronav.faa.gov/d-tpp/2609/00556IL18L.PDF"]);
    assert.equal((await fetch(`${base}/api/data/plates/KAUS/00999XX.PDF`)).status, 404, "not a KAUS chart: never fetched");
    assert.equal(calls.length, 1);
  });
});

test("GET …/georef reports an honest non-georeference for a chart without procedure symbols", async () => {
  await withServer(async (base) => {
    const j = await (await fetch(`${base}/api/data/plates/KAUS/00556IL18L.PDF/georef?proc=I18L`)).json();
    assert.equal(j.georeferenced, false);
    assert.equal(typeof j.reason, "string");
    assert.equal(j.corners, null);
    assert.equal(j.proc, "I18L");
    assert.equal(j.cycle, "2609");
    assert.equal(j.pdfUrl, "/api/data/plates/KAUS/00556IL18L.PDF");
    assert.equal((await fetch(`${base}/api/data/plates/KAUS/00556IL18L.PDF/georef`)).status, 400);
  });
});

test("GET /api/data/procedures/flight/:callsign: filed DP/STAR auto-matched; approaches only suggested", async () => {
  await withServer(async (base) => {
    const j = await (await fetch(`${base}/api/data/procedures/flight/SWA1234`)).json();
    assert.equal(j.planSource, "FILED_FAA");
    assert.equal(j.filed.dp.id, "AUS7");
    assert.equal(j.filed.dp.transition, "JCT");
    assert.equal(j.filed.dp.name, "AUSTIN SEVEN");
    assert.equal(j.filed.star.id, "BLEWE5");
    assert.equal(j.filed.star.transition, "ACT");
    assert.equal(j.filed.star.charts[0].pdf, "00556BLEWE.PDF");
    assert.equal(j.suggestedApproaches[0].id, "I18L");
    assert.equal(j.wind.dirDeg, 160);
    assert.equal(j.approachHonesty, APPROACH_HONESTY);
    // no SWIM plan: airports from the client's predicted plan, nothing "filed"
    const k = await (await fetch(`${base}/api/data/procedures/flight/UAL9?arr=KAUS`)).json();
    assert.equal(k.planSource, "CLIENT_PLAN");
    assert.equal(k.filed.dp, null);
    assert.equal(k.filed.star, null);
    assert.match(k.filed.note, /No FAA-filed plan/);
    assert.equal(k.suggestedApproaches[0].id, "I18L");
    assert.equal((await fetch(`${base}/api/data/procedures/flight/%20`)).status, 400);
  });
});
