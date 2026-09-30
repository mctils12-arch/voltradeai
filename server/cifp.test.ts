import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  CifpIndex, CifpStore, arcPoints, buildLegs, cifpZipUrl, cycleAt, cycleFromEffective, destinationPoint, distNm,
  extractZipEntry, holdRacetrack, parseAltitude, parseApproachIdent, approachName, parseArincLatLon, parseMagVar,
  parseProcedureLeg, procedureFixes, procedurePath, segmentRole, type AirportData, type CifpLeg,
} from "./cifp";

// Real FAA CIFP records (cycle 2609, FAACIFP18) for KAUS: airport, runways,
// localizers, the terminal fixes they reference, ILS RWY 18L, BLEWE5, AUS7,
// plus the enroute waypoints / VHF navaids those procedures name.
const here = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = fs.readFileSync(path.join(here, "__fixtures__", "cifp_kaus_sample.txt"));
const idx = CifpIndex.build(FIXTURE);
const kaus = idx.airport("KAUS") as AirportData;
const proc = (id: string) => kaus.procedures.find((p) => p.id === id)!;

// ── cycle math ──────────────────────────────────────────────────────────────
test("AIRAC cycle: 2026-09-30 is 2609 (eff 09-03 09:01Z), CIFP zip named by effective date", () => {
  const c = cycleAt(Date.UTC(2026, 8, 30, 12));
  assert.equal(c.ident, "2609");
  assert.equal(c.effective, "2026-09-03T09:01:00.000Z");
  assert.equal(c.expires, "2026-10-01T09:01:00.000Z");
  assert.equal(c.yymmdd, "260903");
  assert.equal(cifpZipUrl(c), "https://aeronav.faa.gov/Upload_313-d/cifp/CIFP_260903.zip"); // verified live 2026-09-30
});

test("AIRAC cycle: boundaries, year rollover and offsets", () => {
  assert.equal(cycleAt(Date.UTC(2026, 9, 1, 9, 0)).ident, "2609");   // one minute before 2610
  assert.equal(cycleAt(Date.UTC(2026, 9, 1, 9, 1)).ident, "2610");
  assert.equal(cycleAt(Date.UTC(2026, 0, 22, 9, 1)).ident, "2601");
  assert.equal(cycleAt(Date.UTC(2025, 11, 31)).ident, "2513");        // 2025 had 13 cycles
  assert.equal(cycleAt(Date.UTC(2026, 8, 30), -1).ident, "2608");
  assert.equal(cycleAt(Date.UTC(2026, 8, 30), 1).ident, "2610");
  assert.equal(cycleFromEffective(Date.UTC(2026, 9, 1, 9, 1)).yymmdd, "261001");
});

// ── ARINC 424 fields ────────────────────────────────────────────────────────
test("ARINC lat/lon: degrees-minutes-seconds(hundredths)", () => {
  const p = parseArincLatLon("N30174456W097354238")!;
  assert.ok(Math.abs(p.lat - (30 + 17 / 60 + 44.56 / 3600)) < 1e-9);
  assert.ok(Math.abs(p.lon + (97 + 35 / 60 + 42.38 / 3600)) < 1e-9);
  assert.equal(parseArincLatLon("S01000000E010000000")!.lat, -1);
  assert.equal(parseArincLatLon("garbage"), null);
  assert.equal(parseMagVar("E0040"), 4);
  assert.equal(parseMagVar("W0110"), -11);
});

test("ARINC altitude constraints", () => {
  assert.deepEqual(parseAltitude("+", "05000", "     ")?.text, "5000A");
  assert.deepEqual(parseAltitude("-", "03000", "     ")?.text, "3000B");
  assert.deepEqual(parseAltitude("B", "14000", "10000")?.text, "10000–14000");
  assert.equal(parseAltitude("+", "FL200", "     ")?.alt1Ft, 20000);
  assert.equal(parseAltitude("+", "FL200", "     ")?.text, "FL200A");
  assert.equal(parseAltitude(" ", "     ", "     "), null);
});

test("parseProcedureLeg: real KAUS ILS 18L CF leg to RRTOO", () => {
  const line = FIXTURE.toString("latin1").split("\n").find((l) => l.startsWith("SUSAP KAUSK4FI18L  ADOFFS 030RRTOO"))!;
  const l = parseProcedureLeg(line)!;
  assert.equal(l.routeType, "A");
  assert.equal(l.transition, "DOFFS");
  assert.equal(l.seq, 30);
  assert.deepEqual(l.fix, { ident: "RRTOO", region: "K4", section: "P", sub: "C" });
  assert.equal(l.pathTerm, "CF");
  assert.equal(l.recNav?.ident, "IVNK");
  assert.equal(l.recNav?.section, "P");
  assert.equal(l.recNav?.sub, "I");
  assert.equal(l.thetaDeg, 354.7);
  assert.equal(l.rhoNm, 8);
  assert.equal(l.courseDeg, 174.7);
  assert.equal(l.courseTrue, false);
  assert.equal(l.distNm, 4.6);
  assert.equal(l.alt?.text, "2500A");
  assert.equal(l.desc[3], "B"); // intermediate fix
});

test("parseProcedureLeg: continuation records are skipped; hold time decoded", () => {
  const cont = "SUSAP KAUSK4FH18LZ H      020DDTOOK4PC2W                                                A031A242                      FS   276622104";
  assert.equal(parseProcedureLeg(cont), null);
  const hm = FIXTURE.toString("latin1").split("\n").find((l) => l.includes("070HOOKK") && l.includes(" HM "))!;
  const l = parseProcedureLeg(hm)!;
  assert.equal(l.pathTerm, "HM");
  assert.equal(l.turn, "R");
  assert.equal(l.timeMin, 1);
  assert.equal(l.distNm, null);
  assert.equal(l.courseDeg, 267.7);
});

// ── airport block + index ───────────────────────────────────────────────────
test("CifpIndex: KAUS airport, runways, localizers, procedures from the real file layout", () => {
  assert.equal(kaus.name, "AUSTIN-BERGSTROM INTL");
  assert.equal(kaus.magVar, 4);
  assert.equal(kaus.elevFt, 542);
  assert.deepEqual(Array.from(kaus.runways.keys()).sort(), ["RW18L", "RW18R", "RW36L", "RW36R"]);
  assert.equal(kaus.runways.get("RW18L")!.bearingMag, 175);
  assert.ok(kaus.localizers.has("IVNK"));
  assert.deepEqual(kaus.procedures.map((p) => `${p.kind}:${p.id}`).sort(), ["IAP:I18L", "SID:AUS7", "STAR:BLEWE5"]);
  assert.deepEqual(proc("I18L").transitions.map((t) => `${t.routeType}/${t.id}`), ["A/DOFFS", "A/HOUKM", "A/JEDYE", "I/"]);
  assert.equal(idx.airport("KXXX"), null);
  assert.equal(idx.navaidName("CWK"), "CENTEX");
});

test("CifpIndex.resolve: terminal fix, enroute waypoint, VHF navaid, runway; NASR fallback only for unresolved fixes", () => {
  const leg = (ident: string, section: string, sub: string) => ({ ident, region: "K4", section, sub });
  assert.ok(distNm(idx.resolve(leg("DOFFS", "P", "C"), kaus)!, { lat: 30.45266, lon: -97.66441 }) < 0.01);
  assert.ok(distNm(idx.resolve(leg("HOOKK", "E", "A"), kaus)!, { lat: 30.36, lon: -97.2025 }) < 0.1);
  assert.ok(distNm(idx.resolve(leg("CWK", "D", " "), kaus)!, { lat: 30.37855, lon: -97.52985 }) < 0.01);
  assert.ok(idx.resolve(leg("RW18L", "P", "G"), kaus));
  assert.equal(idx.resolve(leg("ZZZZZ", "E", "A"), kaus), null);
  const withExt = CifpIndex.build(FIXTURE);
  withExt.external = (id) => (id === "ZZZZZ" ? { lat: 1, lon: 2 } : null);
  const ap = withExt.airport("KAUS")!;
  assert.deepEqual(withExt.resolve(leg("ZZZZZ", "E", "A"), ap), { lat: 1, lon: 2 });
  assert.equal(withExt.resolve(leg("RW99", "P", "G"), ap), null); // never a runway from NASR
});

// ── naming ──────────────────────────────────────────────────────────────────
test("approach idents decode to chart-style names", () => {
  assert.deepEqual(parseApproachIdent("I18L"), { typeCode: "I", typeName: "ILS", runway: "18L", variant: null });
  assert.equal(approachName("R18LY"), "RNAV (GPS) Y RWY 18L");
  assert.equal(approachName("H36RZ"), "RNAV (RNP) Z RWY 36R");
  assert.equal(segmentRole("STAR", "1"), "enroute transition");
  assert.equal(segmentRole("STAR", "2"), "common");
  assert.equal(segmentRole("SID", "3"), "enroute transition");
  assert.equal(segmentRole("IAP", "A"), "approach transition");
  assert.equal(segmentRole("IAP", "I"), "final");
});

// ── geometry ────────────────────────────────────────────────────────────────
test("ILS 18L path: exact fixed legs, approx-flagged CA/VI/HM, missed approach tagged", () => {
  const p = procedurePath(idx, kaus, proc("I18L"));
  const legs = p.features.filter((f) => f.properties.kind === "leg");
  const fixes = p.features.filter((f) => f.properties.kind === "fix");
  assert.equal(p.meta.unresolvedLegs, 0);
  const byTerm = (t: string) => legs.filter((f) => f.properties.pathTerm === t);
  assert.ok(byTerm("TF").every((f) => f.properties.approx === false));
  assert.ok(byTerm("CF").every((f) => f.properties.approx === false));
  for (const t of ["CA", "VI", "HM"]) assert.ok(byTerm(t).every((f) => f.properties.approx === true && typeof f.properties.approxNote === "string"), t);
  assert.equal(p.meta.approxLegs, 3);
  // missed approach begins at the CA after the runway (MAP) and includes the hold
  assert.ok(byTerm("CA")[0].properties.missed === true);
  assert.ok(byTerm("HM")[0].properties.missed === true);
  assert.ok(legs.filter((f) => f.properties.fix === "DDTOO").every((f) => f.properties.missed === false));
  const fx = (id: string) => fixes.find((f) => f.properties.ident === id)!.properties;
  assert.equal(fx("DDTOO").role, "FAF");
  assert.equal(fx("RW18L").role, "MAP");
  assert.equal(fx("DOFFS").role, "IAF");
  assert.equal(fx("DOFFS").label, "DOFFS 5000A");
  assert.equal(fx("HOOKK").missed, true);
  assert.equal(fx("HOOKK").fixKind, "waypoint");
  assert.equal(fx("RW18L").fixKind, "runway");
  assert.equal(procedurePath(idx, kaus, proc("BLEWE5")).features.find((f) => f.properties.ident === "CWK")!.properties.fixKind, "navaid");
  // the final CF from DDTOO to the runway follows the localizer course (~178.7 true)
  const toRw = byTerm("CF").find((f) => f.properties.fix === "RW18L")!;
  const [a, b] = (toRw.geometry.coordinates as Array<[number, number]>);
  const brg = (Math.atan2((b[0] - a[0]) * Math.cos(30.2 * Math.PI / 180), b[1] - a[1]) * 180 / Math.PI + 360) % 360;
  assert.ok(Math.abs(brg - 178.7) < 1, `final course ${brg}`);
  // the hold closes on its fix
  const hm = byTerm("HM")[0].geometry.coordinates as Array<[number, number]>;
  assert.deepEqual(hm[hm.length - 1], fx("HOOKK") && (fixes.find((f) => f.properties.ident === "HOOKK")!.geometry.coordinates as [number, number]));
});

test("transition filter keeps the chosen transition + core route; STAR roles", () => {
  const p = procedurePath(idx, kaus, proc("BLEWE5"), "ACT");
  const trs = new Set(p.features.filter((f) => f.properties.kind === "leg").map((f) => f.properties.transition));
  assert.deepEqual(Array.from(trs).sort(), ["ACT", "ALL"]);
  assert.equal(p.meta.selectedTransition, "ACT");
  assert.deepEqual(p.meta.transitions.map((t) => t.role), ["enroute transition", "enroute transition", "common"]);
});

test("NASR cross-check: agreeing gazetteer reports 0; a moved fix is listed as disagreeing", () => {
  const other = CifpIndex.build(FIXTURE);
  other.external = (id) => (id === "SCALI" ? { lat: 30.5, lon: -97.66 } : id === "DOFFS" ? { lat: 30.45266, lon: -97.66441 } : null);
  const ap = other.airport("KAUS")!;
  const p = procedurePath(other, ap, ap.procedures.find((x) => x.id === "I18L")!);
  assert.ok(p.meta.nasrCrossCheck);
  assert.deepEqual(p.meta.nasrCrossCheck!.disagreeing, ["SCALI"]);
  assert.ok(p.meta.nasrCrossCheck!.checked >= 2);
  const doffs = p.features.find((f) => f.properties.ident === "DOFFS")!.properties;
  assert.equal(doffs.source, "CIFP");
  assert.ok((doffs.nasrDiffNm as number) < 0.01);
});

test("RF leg is a true arc about its centre fix", () => {
  const ap: AirportData = {
    icao: "KTST", name: "T", ll: { lat: 30, lon: -97 }, magVar: 0, elevFt: 0,
    terminalFixes: new Map([["AAAAA", destinationPoint({ lat: 30, lon: -97 }, 270, 3)], ["BBBBB", destinationPoint({ lat: 30, lon: -97 }, 0, 3)], ["CCCCC", { lat: 30, lon: -97 }]]),
    runways: new Map(), localizers: new Map(), procedures: [],
  };
  const base = { routeType: "R", transition: "", recNav: null, arcRadiusNm: 3, thetaDeg: null, rhoNm: null, courseDeg: null, courseTrue: false, distNm: null, timeMin: null, alt: null, speedKt: null, vertAngleDeg: null, missedStart: false, desc: "E   " };
  const legs: CifpLeg[] = [
    { ...base, seq: 10, fix: { ident: "AAAAA", region: "K4", section: "P", sub: "C" }, turn: null, pathTerm: "IF", center: null },
    { ...base, seq: 20, fix: { ident: "BBBBB", region: "K4", section: "P", sub: "C" }, turn: "R", pathTerm: "RF", center: { ident: "CCCCC", region: "K4", section: "P", sub: "C" } },
  ];
  const g = buildLegs(legs, ap, CifpIndex.build(""), null, null);
  const arc = g[1].coords;
  assert.equal(g[1].approx, false);
  assert.ok(arc.length > 10);
  for (const p of arc) assert.ok(Math.abs(distNm(p, { lat: 30, lon: -97 }) - 3) < 0.02);
  // right turn from west to north passes north-west (bearing ~315 from centre)
  const mid = arc[Math.floor(arc.length / 2)];
  assert.ok(mid.lat > 30 && mid.lon < -97);
});

test("arcPoints honours turn direction; holdRacetrack closes on the fix", () => {
  const c = { lat: 30, lon: -97 };
  const from = destinationPoint(c, 90, 2), to = destinationPoint(c, 180, 2);
  const right = arcPoints(c, from, to, "R");
  const left = arcPoints(c, from, to, "L");
  assert.ok(right.length < left.length, "right turn 90° is shorter than the left 270° way round");
  const hold = holdRacetrack(c, 270, 3.5, "R");
  assert.deepEqual(hold[0], c);
  assert.deepEqual(hold[hold.length - 1], c);
  const far = Math.max(...hold.map((p) => distNm(p, c)));
  assert.ok(far > 3 && far < 5, `hold extent ${far}`);
});

test("procedureFixes maps localizer ident to its plate spelling (I-VNK)", () => {
  const f = procedureFixes(idx, kaus, proc("I18L"));
  assert.ok(f.has("IVNK") && f.has("I-VNK") && f.has("DOFFS") && f.has("CWK") && f.has("HOOKK"));
});

// ── zip + store ─────────────────────────────────────────────────────────────
function makeZip(name: string, data: Buffer): Buffer {
  const comp = zlib.deflateRawSync(data);
  const nameB = Buffer.from(name);
  const local = Buffer.alloc(30);
  local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(8, 8);
  local.writeUInt32LE(comp.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(nameB.length, 26);
  const central = Buffer.alloc(46);
  central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(8, 10);
  central.writeUInt32LE(comp.length, 20); central.writeUInt32LE(data.length, 24); central.writeUInt16LE(nameB.length, 28);
  central.writeUInt32LE(0, 42);
  const cdOff = local.length + nameB.length + comp.length;
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0); eocd.writeUInt16LE(1, 8); eocd.writeUInt16LE(1, 10);
  eocd.writeUInt32LE(central.length + nameB.length, 12); eocd.writeUInt32LE(cdOff, 16);
  return Buffer.concat([local, nameB, comp, central, nameB, eocd]);
}

test("extractZipEntry inflates the named entry", () => {
  const z = makeZip("FAACIFP18", FIXTURE);
  const e = extractZipEntry(z, (n) => n === "FAACIFP18")!;
  assert.equal(e.data.length, FIXTURE.length);
  assert.equal(extractZipEntry(z, (n) => n === "nope"), null);
  assert.throws(() => extractZipEntry(Buffer.from("not a zip at all, definitely not"), () => true));
});

test("CifpStore: downloads the cycle zip to its dir, falls back to the previous cycle on 404, prunes old cycles", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cifp-test-"));
  fs.mkdirSync(path.join(dir, "2601"));
  const now = Date.UTC(2026, 8, 30);
  const urls: string[] = [];
  const fetchImpl = (async (u: string | URL | Request) => {
    const url = String(u);
    urls.push(url);
    if (url.endsWith("CIFP_260903.zip")) return new Response("missing", { status: 404 });
    return new Response(makeZip("FAACIFP18", FIXTURE), { status: 200 });
  }) as typeof fetch;
  const store = new CifpStore({ fetchImpl, now: () => now, dir, external: () => null });
  const { idx: got, cycle } = await store.get();
  assert.equal(cycle.ident, "2608");
  assert.deepEqual(urls, [
    "https://aeronav.faa.gov/Upload_313-d/cifp/CIFP_260903.zip",
    "https://aeronav.faa.gov/Upload_313-d/cifp/CIFP_260806.zip",
  ]);
  assert.ok(got.airport("KAUS"));
  assert.ok(got.external, "the NASR hook is attached to the built index");
  assert.ok(fs.existsSync(path.join(dir, "2608", "FAACIFP18")));
  assert.ok(!fs.existsSync(path.join(dir, "2601")), "stale cycle pruned");
  // second call is served from memory (no new fetch)
  await store.get();
  assert.equal(urls.length, 2);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("CifpStore: total failure throws an honest error and backs off", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cifp-test-"));
  let calls = 0;
  const store = new CifpStore({ fetchImpl: (async () => { calls++; return new Response("x", { status: 503 }); }) as typeof fetch, now: () => Date.UTC(2026, 8, 30), dir });
  await assert.rejects(store.get(), /CIFP unavailable/);
  await assert.rejects(store.get(), /CIFP unavailable/);
  assert.equal(calls, 2, "backoff: the second call does not refetch");
  assert.ok(store.status().lastError);
  fs.rmSync(dir, { recursive: true, force: true });
});
