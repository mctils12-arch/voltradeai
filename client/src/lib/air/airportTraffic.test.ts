// airportTraffic.test — what the FAA traffic-management feed says about one
// airport, parsed from the shape GET /api/data/swim/tfms/active answers with
// (rows below are the production answer of 2026-11-20 22:48Z, trimmed) and
// from the shapes the card must survive.
import test from "node:test";
import assert from "node:assert/strict";
import { parseAirportTraffic, trafficHeadline, configLine, programWords, restartWords, restrictionLine, reasonWords, zuluAt, periodWords } from "./airportTraffic.ts";

const NOW = Date.UTC(2026, 10, 20, 22, 48, 10);
const at = (d: number, h: number, m: number) => Date.UTC(2026, 10, d, h, m, 0);
const prog = (over: Record<string, unknown>) => ({ id: "x", kind: "RESTRICTION", status: null, airport: null, facility: null, start: null, end: null, title: null, text: null, reason: null, elements: [], geometry: null, floorFl: null, ceilingFl: null, updatedAt: at(20, 20, 0), ...over });
const LIVE = {
  at: NOW,
  programs: [
    prog({ id: "ADV:ATCSCC:0059:2026-11-20", kind: "ADVISORY", status: "GDP", airport: "SAN", facility: "DCCO", start: at(20, 17, 0), end: at(21, 4, 59), title: "ATCSCC ADVZY 059 SAN/ZLA 11/20/2026 CDM GROUND DELAY PROGRAM", text: "CTL ELEMENT: SAN" }),
    prog({ id: "GDP:SAN", kind: "GDP", status: "ACTUAL", airport: "SAN", facility: "ZLA", start: at(20, 17, 0), end: at(21, 4, 59), title: "gdpAdvisory", reason: "VOLUME" }),
    prog({ id: "RSTR:1", status: "4", airport: "SAN", facility: "ZLA", start: at(20, 20, 0), end: at(20, 23, 0), title: "MIT DEPARTURE", reason: "VOL:Compacted Demand", elements: ["PADRZ"] }),
    // the same restriction re-sent under a new id and action code: one line, not two
    prog({ id: "RSTR:2", status: "5", airport: "SAN", facility: "ZLA", start: at(20, 20, 0), end: at(20, 23, 0), title: "MIT DEPARTURE", reason: "VOL:COMPACTED DEMAND", elements: ["PADRZ"], updatedAt: at(20, 20, 30) }),
    prog({ id: "RSTR:3", status: "4", airport: "SAN", facility: "SCT", start: at(20, 19, 0), end: at(21, 6, 0), title: "APREQ DEPARTURE", reason: "NONE", elements: ["SAN"] }),
    prog({ id: "ADV:ATCSCC:0096:2026-11-20", kind: "ADVISORY", status: "GS", airport: "TEB", facility: "DCCO", start: at(20, 20, 51), end: at(20, 22, 45), title: "ATCSCC ADVZY 096 TEB/ZNY 11/20/2026 CDM GROUND STOP", updatedAt: at(20, 21, 26) }),
    prog({ id: "ADV:ATCSCC:0097:2026-11-20", kind: "ADVISORY", status: "GS", airport: "TEB", facility: "DCCO", start: at(20, 21, 26), end: at(20, 23, 45), title: "ATCSCC ADVZY 097 TEB/ZNY 11/20/2026 CDM GROUND STOP", updatedAt: at(20, 21, 36) }),
    prog({ id: "RSTR:4", status: "4", airport: "MIA/MCO/FLL", facility: "ZMA", start: at(20, 23, 10), end: at(21, 1, 0), title: "STOP ARRIVAL", reason: "WX:Thunderstorms", elements: ["OSPRY", "RDSOX"] }),
    prog({ id: "RSTR:5", status: "4", airport: "ALL", facility: "ZJX", start: at(20, 21, 0), end: at(20, 23, 30), title: "MIT DEPARTURE", reason: "WX:Thunderstorms", elements: ["OSPRY/RDSOX"] }),
    prog({ id: "RSTR:6", status: "4", airport: "ORD", facility: "C90", start: at(20, 21, 0), end: at(20, 23, 59), title: "CFR SWAP", reason: "TM Initiatives:MIT:VOL", elements: [] }),
    prog({ id: "ADV:ATCSCC:0055:2026-11-20", kind: "ADVISORY", status: null, airport: null, facility: "DCCO", title: "ATCSCC ADVZY 055 DCC 11/20/2026 US/MEXICO OUTLOOKFYI" }),
    prog({ id: "FCA:1", kind: "FCA", status: "UPDATED", airport: null, facility: "DCC", title: "FCALEW", geometry: [[45.32, -82.75], [46.13, -81.27]] }),
  ],
  airportConfigs: [
    { airport: "ATL", arrRunways: "08L/09R/10", depRunways: null, aar: 132, adr: 0, weather: "VMC", facility: "A80", at: at(20, 21, 58) },
    { airport: "BOS", arrRunways: "27/32", depRunways: "33L", aar: 44, adr: 36, weather: "VMC", facility: "A90", at: at(20, 17, 45) },
    { airport: "SAN", arrRunways: "27", depRunways: "27", aar: 24, adr: 24, weather: "IMC", facility: "SCT", at: at(20, 14, 56) },
  ],
  stats: { programs: 447 },
  source: "FAA SWIM TFMS (SCDS, NDRB pre-approved for public release; not for operational use)",
};

test("an airport with a delay program: the program message gives the times and reason, the advisory its title; re-sent restrictions count once", () => {
  const v = parseAirportTraffic(LIVE, "SAN")!;
  assert.deepEqual(v.programs, [{ key: "GDP", kind: "GDP", fromProgram: true, startMs: at(20, 17, 0), endMs: at(21, 4, 59), reason: "volume", advisory: "ATCSCC ADVZY 059 SAN/ZLA 11/20/2026 CDM GROUND DELAY PROGRAM" }]);
  assert.equal(v.restrictions.length, 2);
  assert.equal(trafficHeadline(v, NOW), "Ground delay program · volume · until Nov 21 04:59Z · 2 restrictions");
  assert.deepEqual(v.restrictions.map((r) => restrictionLine(r, NOW)), [
    "Call for release, departures · via SAN · set by SCT · until Nov 21 06:00Z",
    "Miles-in-trail, departures · via PADRZ · set by ZLA · volume: compacted demand · until 23:00Z",
  ]);
  assert.equal(configLine(v, NOW), "Arriving 27 · departing 27 · 24 arrivals / 24 departures an hour · instrument conditions (IMC) · reported 14:56Z");
  assert.equal(v.searched, 12);
  assert.equal(v.source, LIVE.source);
  // a four-letter ICAO code for a US airport reads the same rows
  assert.deepEqual(parseAirportTraffic(LIVE, "ksan")!.programs, v.programs);
});

test("a ground stop known only from advisories: the newest advisory's period, and it is marked as not from a program message", () => {
  const v = parseAirportTraffic(LIVE, "TEB")!;
  assert.deepEqual(v.programs.map((p) => [p.kind, p.fromProgram, p.advisory, programWords(p, NOW)]), [["GS", false, "ATCSCC ADVZY 097 TEB/ZNY 11/20/2026 CDM GROUND STOP", "Ground stop · until 23:45Z"]]);
  assert.equal(trafficHeadline(v, NOW), "Ground stop · until 23:45Z");
  assert.equal(configLine(v, NOW), null);
});

test("restrictions: a list of airports matches each of them, ALL matches none, a future one says when it starts, an unknown category is shown as sent", () => {
  for (const code of ["MIA", "MCO", "FLL"]) {
    const v = parseAirportTraffic(LIVE, code)!;
    assert.equal(trafficHeadline(v, NOW), "No ground stop or delay program · 1 restriction");
    assert.equal(restrictionLine(v.restrictions[0], NOW), "Stop, arrivals · via OSPRY, RDSOX · set by ZMA · weather: thunderstorms · from 23:10Z to Nov 21 01:00Z");
  }
  // "ALL" is the feed's word for every airport: it is not this airport's restriction and is not counted on its card
  assert.equal(parseAirportTraffic(LIVE, "ALL")!.restrictions.length, 1);
  assert.equal(parseAirportTraffic(LIVE, "JAX")!.restrictions.length, 0);
  const ord = parseAirportTraffic(LIVE, "ORD")!;
  assert.equal(restrictionLine(ord.restrictions[0], NOW), "CFR SWAP · set by C90 · tm initiatives: mit: vol · until 23:59Z");
});

test("nothing for the airport is a sentence; rates of 0 are not given; a tower report without programs still shows", () => {
  const none = parseAirportTraffic(LIVE, "XYZ")!;
  assert.equal(trafficHeadline(none, NOW), "Nothing for XYZ among the 12 programs and restrictions held from the FAA feed (the list restarts with the server)");
  // one row held reads in the singular; an empty list says so with its number
  assert.equal(trafficHeadline(parseAirportTraffic({ at: NOW, programs: [LIVE.programs[0]] }, "XYZ")!, NOW), "Nothing for XYZ among the 1 program and restriction held from the FAA feed (the list restarts with the server)");
  assert.equal(trafficHeadline(parseAirportTraffic({ at: NOW, programs: [] }, "XYZ")!, NOW), "Nothing for XYZ among the 0 programs and restrictions held from the FAA feed (the list restarts with the server)");
  assert.equal(configLine(none, NOW), null);
  const atl = parseAirportTraffic(LIVE, "ATL")!;
  assert.equal(configLine(atl, NOW), "Arriving 08L/09R/10 · 132 arrivals an hour · visual conditions (VMC) · reported 21:58Z");
  assert.equal(configLine(parseAirportTraffic(LIVE, "BOS")!, NOW), "Arriving 27/32 · departing 33L · 44 arrivals / 36 departures an hour · visual conditions (VMC) · reported 17:45Z");
  const bare = parseAirportTraffic({ at: NOW, programs: [], airportConfigs: [{ airport: "DAL", arrRunways: null, depRunways: "13R", aar: null, adr: null, weather: "MVFR", at: null }] }, "DAL")!;
  assert.equal(configLine(bare, NOW), "Departing 13R · MVFR");
});

test("words: reasons, times, periods", () => {
  assert.equal(reasonWords("VOL:COMPACTED DEMAND"), "volume: compacted demand");
  assert.equal(reasonWords("VOL:Volume"), "volume");
  assert.equal(reasonWords("WEATHER"), "weather");
  assert.equal(reasonWords("OTHER:SECTOR CONFIG"), "other: sector config");
  assert.equal(reasonWords("NONE"), null);
  assert.equal(reasonWords(null), null);
  assert.equal(zuluAt(at(20, 23, 5), NOW), "23:05Z");
  assert.equal(zuluAt(at(21, 4, 59), NOW), "Nov 21 04:59Z");
  assert.equal(periodWords(null, null, NOW), "");
  assert.equal(periodWords(at(20, 23, 10), null, NOW), "from 23:10Z");
});

test("another shape is null, never a guess", () => {
  assert.equal(parseAirportTraffic(null, "SAN"), null);
  assert.equal(parseAirportTraffic({ error: "x" }, "SAN"), null);
  assert.equal(parseAirportTraffic({ programs: {} }, "SAN"), null);
  assert.equal(parseAirportTraffic(LIVE, ""), null);
  assert.equal(parseAirportTraffic(LIVE, "not a code"), null);
  const odd = parseAirportTraffic({ programs: [null, "x", { kind: "GDP" }, { kind: "RESTRICTION", airport: "SAN" }], airportConfigs: "nope" }, "SAN")!;
  assert.deepEqual([odd.programs.length, odd.restrictions.length, odd.config, odd.atMs], [0, 1, null, null]);
  assert.equal(restrictionLine(odd.restrictions[0], NOW), "Restriction");
});

test("a server that keeps its list across a restart: the warning goes, and the source line says what was carried over and what a restart can still lose", () => {
  const OLD = "The list starts again when the server restarts: a program sent before then is missing until the FAA sends it again.";
  const GAP = "What the FAA sent while the server was restarting is missing until it is sent again, and a program cancelled then stays listed until its end.";
  // a server that says nothing about keeping (an older build, or keeping switched off: `kept: null`)
  for (const body of [{ at: NOW, programs: [] }, { at: NOW, programs: [], kept: null }, { at: NOW, programs: [], kept: "yes" }, { at: NOW, programs: [], kept: {} }]) {
    const v = parseAirportTraffic(body, "XYZ")!;
    assert.equal(v.kept, null);
    assert.equal(restartWords(v, NOW), OLD);
    assert.match(trafficHeadline(v, NOW), /\(the list restarts with the server\)$/);
  }
  // rows read back at the last start (the route's shape, server/swimCapture.ts keptFor)
  const kept = { saved_every_s: 60, last_saved_at: NOW - 30_000, last_save: "ok", restored: { at: at(20, 22, 2), saved_at: at(20, 22, 1), cause: "ok", programs: 351, airport_configs: 43 } };
  const v = parseAirportTraffic({ at: NOW, programs: [], kept }, "XYZ")!;
  assert.deepEqual(v.kept, { carried: 351, savedAtMs: at(20, 22, 1), restoredAtMs: at(20, 22, 2) });
  assert.equal(trafficHeadline(v, NOW), "Nothing for XYZ among the 0 programs and restrictions held from the FAA feed");
  assert.equal(restartWords(v, NOW), `The list is kept across a server restart: 351 programs and restrictions carried over the last one (saved 22:01Z, read back 22:02Z). ${GAP}`);
  // one row carried reads in the singular; a save from another UTC day carries its date
  const one = parseAirportTraffic({ at: NOW, programs: [], kept: { ...kept, restored: { ...kept.restored, programs: 1, saved_at: at(19, 23, 59) } } }, "XYZ")!;
  assert.equal(restartWords(one, NOW), `The list is kept across a server restart: 1 program and restriction carried over the last one (saved Nov 19 23:59Z, read back 22:02Z). ${GAP}`);
  // kept, but the last start read nothing back: first boot, a file too old, or one it could not read
  const NONE = "The list is saved each minute to be kept across a server restart, but nothing was read back at the last one: a program sent before it is missing until the FAA sends it again.";
  for (const restored of [null, { at: at(20, 22, 2), saved_at: null, cause: "no_file", programs: 0, airport_configs: 0 }, { at: at(20, 22, 2), saved_at: at(19, 20, 0), cause: "too_old", programs: 0, airport_configs: 0 }, { at: at(20, 22, 2), saved_at: null, cause: "unreadable", programs: 0 }]) {
    const n = parseAirportTraffic({ at: NOW, programs: [], kept: { ...kept, restored } }, "XYZ")!;
    assert.equal(n.kept?.carried, null);
    assert.equal(restartWords(n, NOW), NONE);
    assert.equal(trafficHeadline(n, NOW), "Nothing for XYZ among the 0 programs and restrictions held from the FAA feed");
  }
  // a restore that put back zero rows from a readable file is a count, not "nothing was read"
  const zero = parseAirportTraffic({ at: NOW, programs: [], kept: { ...kept, restored: { ...kept.restored, programs: 0 } } }, "XYZ")!;
  assert.match(restartWords(zero, NOW), /^The list is kept across a server restart: 0 programs and restrictions carried over/);
  assert.doesNotMatch(restartWords(v, NOW) + restartWords(zero, NOW) + NONE, /NaN|undefined|null/);
});
