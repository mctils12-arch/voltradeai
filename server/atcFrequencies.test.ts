// ATC frequencies (2026-10-14, human request: "is there a way to get the ATC
// frequencies, the ARTCC, or anything"). Fixture = the committed FAA NASR
// cycle 2026-10-01 table (datacore/aircraft/nasr_atc.json), so every expected
// frequency below is the real published one.
// Run: npx tsx --test server/atcFrequencies.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  loadAtcData, atcContext, artccAt, parseSector, sectorContains, pointInRing,
  ATC_HONESTY, CENTER_FLOOR_FT, MAX_CENTER_FREQS,
} from "./atcFrequencies";

const here = path.dirname(fileURLToPath(import.meta.url));
const data = loadAtcData();
const brief = (c: ReturnType<typeof atcContext>) =>
  ({ phase: c.phase, airport: c.airport, f: c.entries.map((e) => `${e.call} ${e.mhz}${e.likely ? "*" : ""}`) });

test("the shipped table: every US center boundary, CLT's published frequencies", () => {
  assert.ok(data.airports.length > 5000, `airports: ${data.airports.length}`);
  for (const id of ["ZTL", "ZDC", "ZJX", "ZNY", "ZAU", "ZDV", "ZLA", "ZOA", "ZSE", "ZBW"]) {
    assert.ok(data.artcc[id]?.L && data.artcc[id]?.H, `${id} LOW + HIGH rings`);
  }
  const clt = data.airports.find((a) => a.id === "KCLT")!;
  assert.equal(clt.lid, "CLT");
  assert.equal(clt.mv, -7);
  const has = (mhz: number, kind: string) => clt.f.some((r) => r[0] === mhz && r[1] === kind);
  assert.ok(has(118.1, "T") && has(126.4, "T") && has(133.35, "T"), "three CLT towers");
  assert.ok(has(121.8, "G") && has(121.9, "G") && has(127.15, "C") && has(132.1, "A"));
  assert.ok(!data.airports.some((a) => a.f.some((r) => r[0] === 121.5)), "the 121.5 guard frequency is never listed");
  assert.ok(data.airports.every((a) => a.f.every((r) => r[0] >= 118 && r[0] < 137)), "civil VHF airband only");
});

test("artccAt: airports sit in their own center; stratum by FL180; oceanic across the antimeridian", () => {
  assert.equal(artccAt(data, 35.2132, -80.9514, 3000)?.id, "ZTL");
  assert.equal(artccAt(data, 35.2132, -80.9514, CENTER_FLOOR_FT)?.stratum, "HIGH");
  assert.equal(artccAt(data, 41.9769, -87.9081, 30000)?.id, "ZAU");
  assert.equal(artccAt(data, 39.8617, -104.6732, 9000)?.id, "ZDV");
  assert.equal(artccAt(data, 35, -60, 37000)?.id, "ZWY", "mid-Atlantic = New York Oceanic, not Oakland");
  assert.equal(artccAt(data, 30, -170, 37000)?.stratum, "OCEANIC", "west of the antimeridian line, inside an oceanic FIR");
  const square: Array<[number, number]> = [[0, 170], [0, -170], [10, -170], [10, 170]];
  assert.equal(pointInRing(5, 179, square), true);
  assert.equal(pointInRing(5, -179, square), true);
  assert.equal(pointInRing(5, 0, square), false);
});

test("parseSector: the published forms we read, and the ones we refuse to guess at", () => {
  assert.deepEqual(
    (({ fromMag, toMag, maxFt, understood }) => ({ fromMag, toMag, maxFt, understood }))(parseSector("120-295 8000 FT & BLW")),
    { fromMag: 120, toMag: 295, maxFt: 8000, understood: true });
  assert.equal(parseSector("246-074 ABV 8000 FT").minFt, 8000);
  assert.equal(parseSector("296-360 8000 FT & BLW").toMag, 0);
  assert.deepEqual(parseSector("RWY 09L/27R, 09R/27L").runways, ["09L", "27R", "09R", "27L"]);
  assert.equal(parseSector("NORTH").side, "N");
  assert.equal(parseSector("150-219 SOUTH").understood, true, "a label word beside a bearing range");
  assert.deepEqual([parseSector("NW-E").fromMag, parseSector("NW-E").toMag], [315, 90]);
  for (const s of ["136-360 MRY", "WHEN NHK APCH CLSD", "0615-2200", "EAST OF VAD 360-VAD 180", "DC SFRA"]) {
    assert.equal(parseSector(s).understood, false, s);
  }
  const sec = parseSector("246-074 ABV 8000 FT");
  assert.equal(sectorContains(sec, 10, 9000), true, "wraps through north");
  assert.equal(sectorContains(sec, 10, 7000), false);
  assert.equal(sectorContains(sec, 120, 9000), false);
  assert.equal(sectorContains(sec, 10, null), null, "no altitude, no claim");
});

test("N843S at CLT, phase by phase: ground, runway 36 tower, departure sector, center", () => {
  assert.deepEqual(brief(atcContext(data, { lat: 35.2203, lon: -80.944, altFt: null, onGround: true })), {
    phase: "SURFACE", airport: "KCLT",
    f: ["Charlotte Ground 121.8", "Charlotte Ground 121.9", "Charlotte Clearance 127.15", "Charlotte ATIS 121.15", "Charlotte ATIS 132.1"],
  });
  assert.deepEqual(brief(atcContext(data, { lat: 35.205, lon: -80.934, altFt: 748, onGround: true },
    { state: "TAKEOFF_ROLL", runway: "36", airport: "KCLT" })),
  { phase: "RUNWAY", airport: "KCLT", f: ["Charlotte Tower 118.1*"] }, "runway 36 = the RWY 18/36 tower");
  assert.deepEqual(brief(atcContext(data, { lat: 35.26, lon: -80.935, altFt: 3000, onGround: false, vsFpm: 2000 },
    { state: "DEPARTED", runway: "36", airport: "KCLT" })),
  { phase: "RUNWAY", airport: "KCLT", f: ["Charlotte Tower 118.1*", "Charlotte Departure 128.325*"] });
  // 15 nm north climbing through 6000: CLT's own 001-119 sector beats the
  // nearer satellite field's general approach frequency
  assert.deepEqual(brief(atcContext(data, { lat: 35.46, lon: -80.95, altFt: 6000, onGround: false, vsFpm: 1500 })),
    { phase: "TERMINAL", airport: "KCLT", f: ["Charlotte Departure 128.325*"] });
  const cruise = atcContext(data, { lat: 33.9, lon: -83.9, altFt: 35000, onGround: false });
  assert.equal(cruise.phase, "ENROUTE");
  assert.deepEqual(cruise.artcc, { id: "ZTL", name: "ATLANTA", stratum: "HIGH" });
  assert.ok(cruise.entries.length > 0 && cruise.entries.length <= MAX_CENTER_FREQS);
  assert.ok(cruise.entries.every((e) => e.role === "Center" && !e.likely && /transmitter$/.test(e.note!)),
    "center sectors are not published: nearest transmitters, never 'the' frequency");
});

test("other airports: runway-split towers, non-towered CTAF, part-time tower hours", () => {
  const atl = atcContext(data, { lat: 33.64, lon: -84.43, altFt: 1026, onGround: true }, { state: "LINED_UP", runway: "27L", airport: "KATL" });
  assert.deepEqual(brief(atl).f, ["Atlanta Tower 119.3*"], "27L is on the 09R/27L tower");
  assert.deepEqual(brief(atcContext(data, { lat: 35.0174, lon: -80.6221, altFt: null, onGround: true })).f[0], "KEQY CTAF 122.7*");
  const jqf = atcContext(data, { lat: 35.3878, lon: -80.7091, altFt: null, onGround: true });
  assert.ok(jqf.entries.some((e) => e.role === "CTAF" && /tower is closed/.test(e.note!)), "part-time tower: CTAF when closed");
  assert.deepEqual(atcContext(data, { lat: 0, lon: -30, altFt: 37000, onGround: false }).entries, [], "nothing published, nothing claimed");
  assert.match(ATC_HONESTY, /not public/);
});

test("wiring: build.ts ships the table; refresh_nasr rebuilds it each cycle", () => {
  const build = fs.readFileSync(path.join(here, "..", "script", "build.ts"), "utf8");
  assert.ok(build.includes('cp("datacore/aircraft/nasr_atc.json", "dist/datacore/aircraft/nasr_atc.json"'));
  const refresh = fs.readFileSync(path.join(here, "..", "scripts", "refresh_nasr.py"), "utf8");
  assert.match(refresh, /_builder\("build_nasr_atc"\)\.main\(tmp\)/);
});
