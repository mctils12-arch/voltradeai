// Flight plan versions: the store (server/flightPlanVersionStore.ts), its
// disk tier, and the plan-history answer (server/flightPlanHistory.ts).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import express from "express";
import type { AddressInfo } from "node:net";
import {
  PlanVersionStore, PlanVersionArchive, readArchivedVersions, parseVersionLine, versionRoutePoints, registerPlanVersionGauge,
  FPV_ARCHIVE_ENV, FPV_MIN_FREE_BYTES, FPV_DAY_CAP_BYTES, FPV_RETENTION_DAYS, FPV_BUFFER_MAX_LINES, FPV_MAX_VERSIONS_PER_FLIGHT,
  FPV_MAX_VERSIONS_TOTAL, FPV_MAX_BYTES, FPV_ENDED_KEEP_MS, FPV_STREAM,
} from "./flightPlanVersionStore";
import { resolvePlanHistory, registerFlightPlanHistoryRoutes, _resetPlanHistoryCache, type PlanHistoryResponse } from "./flightPlanHistory";
import {
  DeviationTracker, HistoryIndex, RouteDbClient, parsePlanQuery, planStatus, resolveFlightPlan, registerFlightPlanRoutes,
  type PlanContext,
} from "./flightPlans";
import { SwimPlanStore, handleSfdpsPayload, lightFlight, type SwimFlightMessage } from "./swimSfdps";
import { MIN_FREE_BYTES } from "./globalScopes";
import { LADD_WITHHELD_NOTE } from "./laddRegistry";
import { readHeapGauges } from "./heapDiag";
import { destinationPoint, haversineNm, initialBearingDeg, interpolateGC, type LatLon } from "../shared/flightPlanGeometry";

// ── fixtures ────────────────────────────────────────────────────────────────
// FAA NASR positions (server/navFixes, read 2026-11-16)
const FIX: Record<string, LatLon> = {
  SNS: { lat: 36.66384, lon: -121.60318 }, AVE: { lat: 35.64698, lon: -119.97861 }, FIM: { lat: 34.35669, lon: -118.88129 },
  PXN: { lat: 36.71546, lon: -120.77868 }, EHF: { lat: 35.48455, lon: -119.09729 }, GMN: { lat: 34.80403, lon: -118.86136 },
  HEC: { lat: 34.79702, lon: -116.46292 },
};
const T0 = Date.parse("2026-11-16T17:00:00Z");
const MIN = 60_000;

/** one parsed SFDPS flight message */
function msg(p: Partial<SwimFlightMessage> & { fixes?: string[] }): SwimFlightMessage {
  const { fixes, ...rest } = p;
  return {
    gufi: "g-skw", callsign: "SKW5000", departure: "KSFO", arrival: "KLAX", cruiseAltFt: null, routeText: null,
    routePoints: (fixes ?? []).map((n) => ({ ...FIX[n], name: n })), ...(fixes?.length ? { routeKind: "expanded" as const } : {}),
    messageType: "TH", isAmendment: false, isCancellation: false, isCompleted: false, timestamp: null,
    departureTime: null, position: null, flightStatus: null, ...rest,
  };
}
const ROUTE1 = ["SNS", "AVE", "FIM"], ROUTE2 = ["SNS", "PXN", "EHF", "GMN"];
const text = (f: string[], dep = "KSFO", arr = "KLAX") => [dep, ...f, arr].join("..");
const FH1 = (ts: number | null = T0) => msg({ messageType: "FH", cruiseAltFt: 24000, routeText: text(ROUTE1), fixes: ROUTE1, timestamp: ts });
const AH_REROUTE = (ts: number | null) => msg({ messageType: "AH", isAmendment: true, routeText: text(ROUTE2), fixes: ROUTE2, timestamp: ts });
const AH_CRUISE = (ts: number | null, ft = 28000) => msg({ messageType: "AH", isAmendment: true, cruiseAltFt: ft, timestamp: ts });
const TRACK_MSG = () => msg({ messageType: "TH" });

function wired(opts: ConstructorParameters<typeof PlanVersionStore>[0] = {}) {
  const store = new PlanVersionStore(opts);
  let clock = T0;
  let calls = 0;
  const swim = new SwimPlanStore({
    onPlanContent: (plan, m, now) => { calls++; store.record(plan, m, now); },
    onRemoved: (key, reason) => { store.planRemoved(key, reason, clock); },
  });
  return { store, swim, hookCalls: () => calls, setClock: (t: number) => { clock = t; } };
}

// ── the store ───────────────────────────────────────────────────────────────

test("version 1 on first sight of a route; our clock and the message's own time are both kept", () => {
  const { store, swim } = wired();
  swim.upsert(FH1(T0 - 4000), T0);
  const rec = store.forCallsign("SKW5000")!;
  assert.equal(rec.versions.length, 1);
  const v = rec.versions[0];
  assert.equal(v.n, 1);
  assert.equal(v.receivedAtMs, T0);
  assert.equal(v.swimMessageMs, T0 - 4000);
  assert.equal(v.messageType, "FH");
  assert.deepEqual([v.departure, v.arrival, v.cruiseAltFt, v.routeText], ["KSFO", "KLAX", 24000, "KSFO..SNS..AVE..FIM..KLAX"]);
  assert.deepEqual(versionRoutePoints(v).map((p) => p.name), ROUTE1);
  assert.ok(Math.abs(versionRoutePoints(v)[1].lat - FIX.AVE.lat) < 1e-4);
  assert.equal(v.detail, null);
  assert.equal(rec.originalAsFiled, true, "an FH that is the first message of the flight, no amendment before it");
  assert.equal(store.counters.firstVersions, 1);
  // a message with no timestamp: swimMessageMs is null, never our clock dressed up
  const b = wired();
  b.swim.upsert(FH1(null), T0);
  assert.equal(b.store.forCallsign("SKW5000")!.versions[0].swimMessageMs, null);
});

test("a no-op message records nothing: repeated FH, an AH that changes none of route/destination/cruise, track messages", () => {
  const { store, swim, hookCalls } = wired();
  swim.upsert(FH1(), T0);
  swim.upsert(FH1(T0 + MIN), T0 + MIN);                                   // the same plan sent again
  swim.upsert(msg({ messageType: "AH", isAmendment: true }), T0 + 2 * MIN); // an amendment to something we do not version
  const before = hookCalls();
  for (let i = 0; i < 50; i++) swim.upsert(TRACK_MSG(), T0 + 3 * MIN + i);  // the 95% of the feed
  assert.equal(hookCalls(), before, "track messages never reach the recorder");
  assert.equal(store.versions, 1);
  assert.equal(store.forCallsign("SKW5000")!.versions.length, 1);
  assert.equal(store.counters.versionsRecorded, 1);
  assert.equal(store.counters.messagesNoChange, 2);
  assert.equal(swim.lookup("SKW5000", T0 + 4 * MIN)!.amendments, 1, "the SWIM store still counts the AH; only versions ignore it");
});

test("each kind of change is a new version, in order, with what changed", () => {
  const { store, swim } = wired();
  swim.upsert(FH1(), T0);
  swim.upsert(AH_REROUTE(T0 + 10 * MIN - 2000), T0 + 10 * MIN);
  swim.upsert(AH_CRUISE(T0 + 20 * MIN), T0 + 20 * MIN);
  swim.upsert(msg({ messageType: "AH", isAmendment: true, arrival: "KLAS", routeText: text(["SNS", "PXN", "HEC"], "KSFO", "KLAS"), fixes: ["SNS", "PXN", "HEC"] }), T0 + 30 * MIN);
  const rec = store.forCallsign("SKW5000")!;
  assert.deepEqual(rec.versions.map((v) => v.n), [1, 2, 3, 4]);
  assert.deepEqual(rec.versions.map((v) => v.receivedAtMs - T0), [0, 10 * MIN, 20 * MIN, 30 * MIN]);
  assert.equal(rec.versions[1].swimMessageMs, T0 + 10 * MIN - 2000);
  assert.match(rec.versions[1].detail!, /route text changed; expanded route changed \(3→4 points\)/);
  assert.deepEqual(versionRoutePoints(rec.versions[1]).map((p) => p.name), ROUTE2);
  assert.equal(rec.versions[2].detail, "cruise 24000→28000 ft");
  assert.equal(rec.versions[2].cruiseAltFt, 28000);
  assert.equal(rec.versions[2].routeText, text(ROUTE2), "a cruise-only version keeps the route in force");
  assert.match(rec.versions[3].detail!, /destination KLAX→KLAS/);
  assert.equal(rec.versions[3].arrival, "KLAS");
  // version 1 is untouched by what came after
  assert.deepEqual([rec.versions[0].routeText, rec.versions[0].cruiseAltFt, rec.versions[0].arrival], [text(ROUTE1), 24000, "KLAX"]);
  assert.deepEqual(versionRoutePoints(rec.versions[0]).map((p) => p.name), ROUTE1);
  assert.equal(store.counters.changeVersions, 3);
  assert.equal(rec.recorded, 4);
});

test("first sight that is NOT a filing is flagged: track first, an AH first, or an amendment already counted", () => {
  const late = wired();
  late.swim.upsert(TRACK_MSG(), T0 - 5 * MIN);  // we joined the flight in the air
  assert.equal(late.store.flights, 0, "no route yet: nothing to version");
  late.swim.upsert(FH1(), T0);
  assert.equal(late.store.forCallsign("SKW5000")!.originalAsFiled, false, "the FH was not the first message we saw");
  assert.equal(late.store.forCallsign("SKW5000")!.planFirstSeenMs, T0 - 5 * MIN);

  const ah = wired();
  ah.swim.upsert(AH_REROUTE(T0), T0);
  assert.equal(ah.store.forCallsign("SKW5000")!.originalAsFiled, false, "an amendment as first sight is never 'as filed'");
  assert.equal(ah.store.forCallsign("SKW5000")!.versions[0].messageType, "AH");

  // a plan restored after a restart: the SWIM store already knows it was first seen earlier
  const restarted = wired();
  restarted.swim.seedHistory([["gufi:g-skw", T0 - 60 * MIN, 0, T0 - MIN]], T0);
  restarted.swim.upsert(FH1(), T0);
  assert.equal(restarted.store.forCallsign("SKW5000")!.originalAsFiled, false);
});

test("a field the first route message lacked is filled into the same version, not counted as a change", () => {
  const { store, swim } = wired();
  swim.upsert(msg({ messageType: "FH", routeText: text(ROUTE1), timestamp: T0 }), T0); // text only, no cruise, no points
  swim.upsert(msg({ messageType: "FH", cruiseAltFt: 24000, routeText: text(ROUTE1), fixes: ROUTE1 }), T0 + MIN);
  const rec = store.forCallsign("SKW5000")!;
  assert.equal(rec.versions.length, 1);
  assert.equal(rec.versions[0].cruiseAltFt, 24000);
  assert.deepEqual(versionRoutePoints(rec.versions[0]).map((p) => p.name), ROUTE1);
  assert.equal(rec.versions[0].receivedAtMs, T0, "still the version first seen at T0");
  assert.equal(store.counters.fieldsFilled, 1);
});

test("caps: the per-flight cap keeps version 1 and the newest, and says how many it dropped", () => {
  const { store, swim } = wired({ maxPerFlight: 4 });
  swim.upsert(FH1(), T0);
  for (let i = 1; i <= 9; i++) swim.upsert(AH_CRUISE(null, 24000 + i * 1000), T0 + i * MIN);
  const rec = store.forCallsign("SKW5000")!;
  assert.deepEqual(rec.versions.map((v) => v.n), [1, 8, 9, 10]);
  assert.equal(rec.droppedFromMemory, 6);
  assert.equal(rec.recorded, 10);
  assert.equal(store.versions, 4);
  assert.equal(store.counters.versionsDropped.per_flight_cap, 6);
  assert.equal(FPV_MAX_VERSIONS_PER_FLIGHT, 12);
});

test("caps: total versions, bytes and flights evict the least recently changed flights, counted by reason", () => {
  const fill = (store: PlanVersionStore, swim: SwimPlanStore, n: number) => {
    for (let i = 0; i < n; i++) swim.upsert({ ...FH1(), gufi: `g-${i}`, callsign: `TST${i}` }, T0 + i);
  };
  const a = wired({ maxVersions: 5 });
  fill(a.store, a.swim, 8);
  assert.equal(a.store.versions, 5);
  assert.equal(a.store.forCallsign("TST0"), null, "the oldest went first");
  assert.ok(a.store.forCallsign("TST7"));
  assert.equal(a.store.counters.versionsDropped.total_cap, 3);
  assert.equal(a.store.counters.originalsNotArchived, 3, "they left memory without ever reaching the disk");

  const b = wired({ maxFlights: 3 });
  fill(b.store, b.swim, 5);
  assert.equal(b.store.flights, 3);
  assert.equal(b.store.counters.versionsDropped.flights_cap, 2);

  const c = wired({ maxBytes: 1200 });
  fill(c.store, c.swim, 6);
  assert.ok(c.store.estimatedBytes <= 1200, `bytes ${c.store.estimatedBytes}`);
  assert.ok(c.store.counters.versionsDropped.bytes_cap > 0);
  assert.ok(c.store.flights >= 1);
  // the estimate is arithmetic over what is held and returns to zero with it
  const one = wired();
  one.swim.upsert(FH1(), T0);
  const bytes = one.store.estimatedBytes;
  assert.ok(bytes > 300 && bytes < 600, `one 3-point version is about 0.4 KB (${bytes})`);
  assert.ok(FPV_MAX_VERSIONS_TOTAL * bytes < FPV_MAX_BYTES, "at the version cap a typical plan stays under the byte cap");
});

test("eviction follows the plan: expired with it, kept one hour after a flight ends", () => {
  const exp = wired();
  const swimTtl = new SwimPlanStore({
    ttlMs: 60 * MIN,
    onPlanContent: (p, m, now) => { exp.store.record(p, m, now); },
    onRemoved: (k, r) => { exp.store.planRemoved(k, r, T0 + 200 * MIN); },
  });
  swimTtl.upsert(FH1(), T0);
  assert.equal(exp.store.flights, 1);
  swimTtl.evict(T0 + 61 * MIN); // the plan's own TTL
  assert.equal(exp.store.flights, 0);
  assert.equal(exp.store.versions, 0);
  assert.equal(exp.store.estimatedBytes, 0);
  assert.equal(exp.store.counters.versionsDropped.plan_expired, 1);

  const end = wired();
  end.swim.upsert(FH1(), T0);
  end.swim.upsert(AH_CRUISE(null), T0 + MIN);
  end.setClock(T0 + 2 * MIN);
  end.swim.upsert(msg({ isCompleted: true, flightStatus: "COMPLETED" }), T0 + 2 * MIN);
  assert.equal(end.swim.lookup("SKW5000", T0 + 3 * MIN), null, "the plan left the SWIM store");
  const kept = end.store.forCallsign("SKW5000")!;
  assert.equal(kept.versions.length, 2, "its versions are still readable");
  assert.equal(kept.endedAtMs, T0 + 2 * MIN);
  end.store.evict(T0 + 2 * MIN + FPV_ENDED_KEEP_MS - 1);
  assert.ok(end.store.forCallsign("SKW5000"));
  end.store.evict(T0 + 2 * MIN + FPV_ENDED_KEEP_MS + 1);
  assert.equal(end.store.forCallsign("SKW5000"), null);
  assert.equal(end.store.counters.versionsDropped.ended_grace_over, 2);
});

test("heap gauge: versions, cap and the byte estimate are registered", () => {
  const { store, swim } = wired();
  swim.upsert(FH1(), T0);
  registerPlanVersionGauge(store);
  const g = readHeapGauges()["flightPlanVersions.store"] as { entries: number; cap: number; bytes: number; flights: number };
  assert.equal(g.entries, 1);
  assert.equal(g.cap, FPV_MAX_VERSIONS_TOTAL);
  assert.equal(g.flights, 1);
  assert.equal(g.bytes, store.estimatedBytes);
});

test("end to end from XML: an FH then an AH through handleSfdpsPayload give two versions; a light track message gives none", () => {
  const { store, swim } = wired();
  const flight = (source: string, ts: string, route: string, fixes: string[], alt: number) =>
    `<m:MessageCollection xmlns:m="urn:x"><message><flight source="${source}" timestamp="${ts}">
      <flightIdentification aircraftIdentification="SKW5000"/><gufi>g-xml</gufi>
      <departure departurePoint="KSFO"/><arrival arrivalPoint="KLAX"/>
      <requestedAltitude><simple uom="FEET">${alt}</simple></requestedAltitude>
      <agreed><route nasRouteText="${route}"><expandedRoute>${fixes.map((f) =>
        `<routePoint><point fix="${f}"><location><pos>${FIX[f].lat} ${FIX[f].lon}</pos></location></point></routePoint>`).join("")}
      </expandedRoute></route></agreed></flight></message></m:MessageCollection>`;
  handleSfdpsPayload(flight("FH", "2026-11-16T17:00:00Z", text(ROUTE1), ROUTE1, 24000), swim, T0 + 900);
  const th = lightFlight(`<flight source="TH" timestamp="2026-11-16T17:05:00Z"><flightIdentification aircraftIdentification="SKW5000"/><gufi>g-xml</gufi></flight>`, null)!;
  swim.upsert(th, T0 + 5 * MIN);
  handleSfdpsPayload(flight("AH", "2026-11-16T17:10:00Z", text(ROUTE2), ROUTE2, 24000), swim, T0 + 10 * MIN + 700);
  const rec = store.forCallsign("SKW5000")!;
  assert.deepEqual(rec.versions.map((v) => [v.n, v.messageType, v.swimMessageMs, v.receivedAtMs]), [
    [1, "FH", T0, T0 + 900],
    [2, "AH", T0 + 10 * MIN, T0 + 10 * MIN + 700],
  ]);
  assert.equal(rec.originalAsFiled, true);
});

// ── LADD at the store ───────────────────────────────────────────────────────

test("LADD: a callsign known to be in the FAA privacy program is neither held nor written, and that is counted", () => {
  const written: string[] = [];
  const archive = new PlanVersionArchive({ env: {}, readFree: () => 10 * FPV_MIN_FREE_BYTES, append: (ls) => { written.push(...ls); } });
  const { store, swim } = wired({ archive, isLadd: (cs) => cs === "SKW5000" });
  swim.upsert(FH1(), T0);
  swim.upsert(AH_REROUTE(null), T0 + MIN);
  swim.upsert({ ...FH1(), gufi: "g-open", callsign: "UAL9" }, T0 + MIN);
  swim.upsert({ ...AH_REROUTE(null), gufi: "g-open", callsign: "UAL9" }, T0 + 2 * MIN);
  archive.flush();
  assert.equal(store.forCallsign("SKW5000"), null);
  assert.ok(store.forCallsign("UAL9"));
  assert.ok(store.counters.skippedLadd >= 2);
  assert.ok(written.length === 2 && written.every((l) => !l.includes("SKW5000") && l.includes("UAL9")));
});

test("LADD: the generic archive readers (behind /api/diag/archive) refuse the version stream; only the LADD-checking reader serves it", async () => {
  const { readArchiveDay, readArchiveDayEvenSample, isRefusedByGenericReaders, LADD_GATED_ARCHIVE_DIRS } = await import("./datacoreArchive");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "planver-generic-"));
  try {
    const archive = new PlanVersionArchive({ baseDir: dir, env: {}, readFree: () => 10 * FPV_MIN_FREE_BYTES, now: () => T0 });
    const { swim } = wired({ archive });
    swim.upsert(FH1(), T0);
    swim.upsert(AH_REROUTE(null), T0 + MIN);
    assert.equal(archive.flush(), 2);
    assert.ok(LADD_GATED_ARCHIVE_DIRS.has(FPV_STREAM));
    for (const stream of [FPV_STREAM, `./${FPV_STREAM}`, `x/../${FPV_STREAM}`]) {
      assert.ok(isRefusedByGenericReaders(dir, stream));
      assert.equal(await readArchiveDay(stream, "2026-11-16", dir), null, "reads as a missing stream");
      assert.equal(await readArchiveDayEvenSample(stream, "2026-11-16", dir), null);
    }
    // an ordinary stream in the same base still reads
    fs.mkdirSync(path.join(dir, "usaspending"), { recursive: true });
    fs.writeFileSync(path.join(dir, "usaspending", "2026-11-16.jsonl"), '{"aid":"a1"}\n');
    assert.equal(isRefusedByGenericReaders(dir, "usaspending"), false);
    assert.equal((await readArchiveDay("usaspending", "2026-11-16", dir))!.rows.length, 1);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

// ── the disk tier ───────────────────────────────────────────────────────────
const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "planver-"));
const roomy = () => 10 * FPV_MIN_FREE_BYTES;

test("disk: changes only. Version 1 is written with the first change; unchanged route text and points are not repeated", () => {
  const lines: string[] = [];
  const archive = new PlanVersionArchive({ env: {}, readFree: roomy, now: () => T0 + 30 * MIN, append: (ls) => { lines.push(...ls); } });
  const { store, swim } = wired({ archive });
  swim.upsert(FH1(T0 - 1000), T0);
  archive.flush();
  assert.equal(lines.length, 0, "a plan that has not changed is not on disk");
  assert.equal(archive.pendingLines, 0);
  swim.upsert(AH_REROUTE(T0 + 10 * MIN), T0 + 10 * MIN);
  assert.equal(archive.pendingLines, 2);
  swim.upsert(AH_CRUISE(null), T0 + 20 * MIN);
  assert.equal(archive.flush(), 3);
  const rows = lines.map((l) => JSON.parse(l) as Record<string, unknown>);
  assert.deepEqual(rows.map((r) => r.n), [1, 2, 3]);
  assert.deepEqual(rows.map((r) => r.t), [T0, T0 + 10 * MIN, T0 + 20 * MIN], "version 1 carries its OWN time, not the time it was written");
  assert.deepEqual([rows[0].mt, rows[0].ty, rows[0].o, rows[0].fs], [T0 - 1000, "FH", 1, T0]);
  assert.equal(rows[0].rt, text(ROUTE1));
  assert.equal(rows[0].pn, "SNS|AVE|FIM");
  assert.deepEqual((rows[0].pc as number[]).slice(0, 2), [36.66384, -121.60318]);
  assert.equal(rows[1].rt, text(ROUTE2));
  assert.equal(rows[1].d, "route text changed; expanded route changed (3→4 points)");
  // the cruise-only change: no route text, no points, just the markers
  assert.deepEqual([rows[2].rs, rows[2].ps, rows[2].rt, rows[2].pc, rows[2].pn, rows[2].alt], [1, 1, undefined, undefined, undefined, 28000]);
  // LINE SIZES (the disk estimate in the module header rests on these)
  const sizes = lines.map((l) => Buffer.byteLength(l) + 1);
  console.log(`# line sizes: 3-point original ${sizes[0]} B, 4-point reroute ${sizes[1]} B, cruise-only ${sizes[2]} B`);
  assert.ok(sizes[0] > 150 && sizes[0] < 330, `3-point original ${sizes[0]} B`);
  assert.ok(sizes[1] > 150 && sizes[1] < 400, `4-point reroute ${sizes[1]} B`);
  assert.ok(sizes[2] < 180, `cruise-only change ${sizes[2]} B`);
  assert.deepEqual([archive.counters.linesWritten, archive.counters.bytesWritten], [3, sizes[0] + sizes[1] + sizes[2]]);
  assert.equal(store.forCallsign("SKW5000")!.disk.originalWritten, true);
  // a 10-point, 80-character route: the size the estimate uses
  const big = wired({ archive });
  const ten = Array.from({ length: 10 }, (_, i) => ({ lat: 36 - i * 0.31234, lon: -121 + i * 0.41234, name: `FIX${i}X` }));
  big.swim.upsert({ ...FH1(), gufi: "us.fdps.2026-11-16T17:00:00Z.000/17/200", callsign: "UAL1234", routeText: "KSFO.SSTIK5.EBAYE..FIX0X.J80.FIX4X..FIX7X.Q120.FIX9X.BASET5.KLAX/0125", routePoints: ten }, T0);
  big.swim.upsert({ ...AH_CRUISE(null), gufi: "us.fdps.2026-11-16T17:00:00Z.000/17/200", callsign: "UAL1234" }, T0 + MIN);
  const before = lines.length;
  archive.flush();
  const tenPt = Buffer.byteLength(lines[before]) + 1, cruiseOnly = Buffer.byteLength(lines[before + 1]) + 1;
  console.log(`# line sizes: 10-point original with a real GUFI ${tenPt} B, cruise-only ${cruiseOnly} B`);
  assert.ok(tenPt > 430 && tenPt < 620, `10-point original ${tenPt} B`);
  assert.ok(cruiseOnly > 110 && cruiseOnly < 200, `cruise-only with a real GUFI ${cruiseOnly} B`);
});

test("disk: the kill switch stops writes, counts them, and leaves memory working", () => {
  const lines: string[] = [];
  const archive = new PlanVersionArchive({ env: { [FPV_ARCHIVE_ENV]: "0" }, readFree: roomy, append: (ls) => { lines.push(...ls); } });
  const { store, swim } = wired({ archive });
  swim.upsert(FH1(), T0);
  swim.upsert(AH_REROUTE(null), T0 + MIN);
  archive.flush();
  assert.equal(lines.length, 0);
  assert.equal(archive.counters.skipped.disabled, 2, "both the original and the change are counted");
  assert.equal(archive.status().enabled, false);
  assert.equal(store.forCallsign("SKW5000")!.versions.length, 2);
  assert.equal(FPV_ARCHIVE_ENV, "FLIGHT_PLAN_VERSIONS_ARCHIVE");
});

test("disk: the low-disk guard fails closed, counts, and the next write after recovery stands on its own", () => {
  const lines: string[] = [];
  let free: number | null = FPV_MIN_FREE_BYTES - 1;
  let now = T0;
  const archive = new PlanVersionArchive({ env: {}, readFree: () => free, now: () => now, append: (ls) => { lines.push(...ls); } });
  const { store, swim } = wired({ archive });
  swim.upsert(FH1(), T0);
  swim.upsert(AH_REROUTE(null), T0 + MIN);            // below the floor
  assert.equal(archive.counters.skipped.low_disk, 2);
  free = null; now = T0 + 2 * MIN;                     // unreadable: still closed
  swim.upsert(AH_CRUISE(null, 26000), T0 + 2 * MIN);
  assert.equal(archive.counters.skipped.low_disk, 4, "version 1 is still owed, so it is counted again with the new change");
  assert.equal(archive.pendingLines, 0);
  free = FPV_MIN_FREE_BYTES; now = T0 + 4 * MIN;       // exactly at the floor: allowed
  swim.upsert(AH_CRUISE(null, 28000), T0 + 4 * MIN);
  archive.flush();
  const rows = lines.map((l) => JSON.parse(l) as Record<string, unknown>);
  assert.deepEqual(rows.map((r) => r.n), [1, 4], "the versions skipped in between are gone from disk, and counted");
  assert.equal(rows[1].rt, text(ROUTE2), "after a skip the line carries its full route, not a pointer to a line that was never written");
  assert.equal(rows[1].rs, undefined);
  assert.equal((rows[1].pc as number[]).length, 8);
  assert.equal(store.forCallsign("SKW5000")!.versions.length, 4, "memory kept all four");
  assert.ok(FPV_MIN_FREE_BYTES > MIN_FREE_BYTES, "stops above the 1 GiB floor that protects bot state");
  assert.equal(archive.status().freeBytes, FPV_MIN_FREE_BYTES);
});

test("disk: the daily byte cap, a full buffer and a failed append are each counted, never silent", () => {
  const cap = new PlanVersionArchive({ env: {}, readFree: roomy, now: () => T0, dayCapBytes: 800, append: () => undefined });
  const a = wired({ archive: cap });
  a.swim.upsert(FH1(), T0);
  a.swim.upsert(AH_REROUTE(null), T0 + MIN);  // about 600 B for the two lines: fits
  a.swim.upsert(AH_CRUISE(null), T0 + 2 * MIN);
  a.swim.upsert(msg({ messageType: "AH", isAmendment: true, routeText: text(ROUTE1), fixes: ROUTE1 }), T0 + 3 * MIN);
  assert.equal(cap.counters.skipped.day_cap, 1, "the line that would cross the cap is skipped");
  assert.ok(cap.status().dayBytes <= 800 && cap.status().dayBytes > 600);
  assert.equal(FPV_DAY_CAP_BYTES, 72 * 1024 * 1024);

  const full = new PlanVersionArchive({ env: {}, readFree: roomy, now: () => T0, append: () => undefined });
  const b = wired({ archive: full });
  for (let i = 0; i < FPV_BUFFER_MAX_LINES / 2 + 3; i++) {
    b.swim.upsert({ ...FH1(), gufi: `g-${i}`, callsign: `TST${i}` }, T0);
    b.swim.upsert({ ...AH_CRUISE(null), gufi: `g-${i}`, callsign: `TST${i}` }, T0 + 1);
  }
  assert.equal(full.pendingLines, FPV_BUFFER_MAX_LINES);
  assert.equal(full.counters.skipped.buffer_full, 6);

  const bad = new PlanVersionArchive({ env: {}, readFree: roomy, now: () => T0, append: () => { throw new Error("ENOSPC"); } });
  const c = wired({ archive: bad });
  c.swim.upsert(FH1(), T0);
  c.swim.upsert(AH_REROUTE(null), T0 + MIN);
  assert.equal(bad.flush(), 0);
  assert.equal(bad.counters.skipped.write_error, 2);
  assert.equal(bad.counters.linesWritten, 0);
  assert.equal(bad.status().lastError, "ENOSPC");
});

test("disk round trip: hour files through appendLines, read back with routes resolved; old hours are gzipped, expired ones deleted", async () => {
  const dir = tmp();
  try {
    let now = T0 + 10 * MIN;
    const archive = new PlanVersionArchive({ baseDir: dir, env: {}, readFree: roomy, now: () => now });
    const { store, swim } = wired({ archive });
    swim.upsert(FH1(T0 - 1000), T0);
    swim.upsert(AH_REROUTE(T0 + 10 * MIN), T0 + 10 * MIN);
    swim.upsert(AH_CRUISE(null), T0 + 20 * MIN);
    swim.upsert({ ...FH1(), gufi: "g-other", callsign: "SKW500" }, T0);          // a callsign that is a prefix
    swim.upsert({ ...AH_CRUISE(null), gufi: "g-other", callsign: "SKW500" }, T0 + MIN);
    assert.equal(archive.flush(), 5);
    const hourFile = path.join(dir, FPV_STREAM, "2026-11-16-17.jsonl");
    assert.ok(fs.existsSync(hourFile), "the hour-file layout of datacoreArchive.appendLines");
    const read = await readArchivedVersions({ callsign: "SKW5000", fromMs: T0 - 60 * MIN, toMs: T0 + 60 * MIN, baseDir: dir });
    assert.equal(read.complete, true);
    assert.equal(read.flights.length, 1, "SKW500 is another flight");
    const fl = read.flights[0];
    assert.equal(fl.key, "gufi:g-skw");
    assert.equal(fl.originalAsFiled, true);
    assert.equal(fl.planFirstSeenMs, T0);
    const mem = store.forCallsign("SKW5000")!.versions;
    assert.equal(fl.versions.length, 3);
    for (let i = 0; i < 3; i++) {
      const a = fl.versions[i], m = mem[i];
      assert.deepEqual(
        [a.n, a.receivedAtMs, a.swimMessageMs, a.messageType, a.departure, a.arrival, a.cruiseAltFt, a.routeText, a.routeKind, a.detail],
        [m.n, m.receivedAtMs, m.swimMessageMs, m.messageType, m.departure, m.arrival, m.cruiseAltFt, m.routeText, m.routeKind, m.detail]);
      assert.deepEqual(versionRoutePoints(a), versionRoutePoints(m), `version ${a.n} route points survive the disk`);
      assert.equal(a.heldIn, "archive");
    }
    assert.equal(fl.routeUnknown, 0);
    // maintenance: gzip after two hours, same answer from the .gz
    now = T0 + 4 * 3600_000;
    assert.deepEqual(await archive.maintain(now), { compressed: 1, expired: 0 });
    assert.ok(!fs.existsSync(hourFile) && fs.existsSync(`${hourFile}.gz`));
    const gzRead = await readArchivedVersions({ callsign: "SKW5000", fromMs: T0 - 60 * MIN, toMs: T0 + 60 * MIN, baseDir: dir });
    assert.equal(gzRead.flights[0].versions.length, 3);
    assert.equal(zlib.gunzipSync(fs.readFileSync(`${hourFile}.gz`)).toString().trim().split("\n").length, 5);
    // retention
    assert.deepEqual(await archive.maintain(T0 + (FPV_RETENTION_DAYS + 1) * 86_400_000), { compressed: 0, expired: 1 });
    assert.deepEqual(fs.readdirSync(path.join(dir, FPV_STREAM)), []);
    // a missing directory is an empty, complete answer
    const none = await readArchivedVersions({ callsign: "SKW5000", fromMs: T0, toMs: T0 + MIN, baseDir: path.join(dir, "nope") });
    assert.deepEqual([none.flights.length, none.complete], [0, true]);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("read-back: a line whose route points at an earlier line outside the files read says so; garbage lines are counted", async () => {
  const dir = tmp();
  try {
    const d = path.join(dir, FPV_STREAM);
    fs.mkdirSync(d, { recursive: true });
    fs.writeFileSync(path.join(d, "2026-11-16-17.jsonl"), [
      JSON.stringify({ k: "gufi:x", n: 3, t: T0, cs: "SKW5000", alt: 28000, rs: 1, ps: 1, d: "cruise 24000→28000 ft" }),
      `{"cs":"SKW5000", broken`,
    ].join("\n") + "\n");
    const read = await readArchivedVersions({ callsign: "SKW5000", fromMs: T0 - MIN, toMs: T0 + MIN, baseDir: dir });
    assert.equal(read.badLines, 1);
    assert.equal(read.flights[0].routeUnknown, 1);
    assert.equal(read.flights[0].versions[0].routeText, null, "an unknown route is null, not a guess");
    assert.equal(parseVersionLine("not json"), null);
    assert.equal(parseVersionLine(JSON.stringify({ n: 1, t: 1 })), null);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

// ── the plan-history answer ─────────────────────────────────────────────────
const KSFO: LatLon = { lat: 37.6198, lon: -122.3748 };
const HEX = "a1b2c3";
const q = (extra: Record<string, unknown> = {}) =>
  parsePlanQuery(HEX, { callsign: "SKW5000", ...extra }) as Exclude<ReturnType<typeof parsePlanQuery>, { error: string }>;

function ctxFor(w: ReturnType<typeof wired>, opts: { recent?: Array<{ t: number; la: number; lo: number; al?: number | null }>; now?: number; isLadd?: PlanContext["isLadd"] } = {}): PlanContext {
  const now = () => opts.now ?? T0 + 60 * MIN;
  const noNet = (async () => new Response("nf", { status: 404 })) as unknown as typeof fetch;
  return {
    routeDb: new RouteDbClient({ now, batchWindowMs: 1, fetchImpl: noNet }),
    history: new HistoryIndex({ now, scan: async () => new Map() }),
    swim: w.swim, tracker: new DeviationTracker(), now,
    recentFixes: async () => opts.recent ?? [],
    historyBudgetMs: 200, liveLookup: null, versions: w.store,
    ...(opts.isLadd ? { isLadd: opts.isLadd } : {}),
  };
}
/** archive-shaped fixes (t in seconds) every 60 s along a path of legs, optionally pushed sideways */
function fly(legs: Array<[LatLon, LatLon]>, startMs: number, stepNm: number, offset: (i: number) => number = () => 0) {
  const out: Array<{ t: number; la: number; lo: number; al: number }> = [];
  let i = 0;
  for (const [a, b] of legs) {
    const len = haversineNm(a, b), brg = initialBearingDeg(a, b);
    for (let dNm = 0; dNm < len; dNm += stepNm, i++) {
      const on = interpolateGC(a, b, dNm / len);
      const p = offset(i) ? destinationPoint(on, brg + 90, offset(i)) : on;
      out.push({ t: (startMs + i * MIN) / 1000, la: p.lat, lo: p.lon, al: 7300 });
    }
  }
  return out;
}

test("plan-history: original, current, every version, and each amendment classified with its time and the aircraft's position", async () => {
  const w = wired();
  w.swim.upsert(FH1(T0 - 30 * MIN), T0 - 30 * MIN);
  // the aircraft flies the filed route to SNS, then the amended one
  const fixes = fly([[KSFO, FIX.SNS], [FIX.SNS, FIX.PXN], [FIX.PXN, FIX.EHF]], T0, 7);
  const atSns = fixes.findIndex((f) => haversineNm({ lat: f.la, lon: f.lo }, FIX.SNS) < 4);
  const amendAt = fixes[atSns].t * 1000 + 10_000; // 10 s after a fix
  w.swim.upsert(AH_REROUTE(amendAt), amendAt + 3000);
  w.swim.upsert(AH_CRUISE(null), amendAt + 5 * MIN + 30_000); // no message time: halfway between two fixes
  const r = await resolvePlanHistory(q(), ctxFor(w, { recent: fixes, now: amendAt + 20 * MIN }));

  assert.equal(r.source, "FILED_FAA");
  assert.equal(r.kind, "raw");
  assert.equal(r.callsign, "SKW5000");
  assert.equal(r.plan!.originalAsFiled, true);
  assert.equal(r.plan!.originalIsFirstSeen, false);
  assert.equal(r.plan!.versionsReturned, 3);
  assert.equal(r.plan!.versionsMissing, 0);
  assert.deepEqual(r.versions.map((v) => [v.version, v.seq, v.messageType, v.cruiseAltFt]), [[1, 1, "FH", 24000], [2, 2, "AH", 24000], [3, 3, "AH", 28000]]);
  // the gray original and the current route, both drawable
  assert.equal(r.original!.version, 1);
  assert.deepEqual(r.original!.fixes, ROUTE1);
  assert.equal(r.original!.routeText, text(ROUTE1));
  assert.equal(r.original!.pathEstimated, false);
  assert.equal(r.original!.points![0].name, "KSFO");
  assert.equal(r.original!.points![r.original!.points!.length - 1].name, "KLAX");
  assert.ok(r.original!.points!.length > 10, "densified for drawing");
  assert.equal(r.current!.version, 3);
  assert.deepEqual(r.current!.fixes, ROUTE2);
  assert.ok(r.current!.points!.some((p) => p.name === "PXN"));
  assert.equal(r.versions[1].points, undefined, "middle versions carry no path unless asked");
  // amendments
  assert.deepEqual(r.amendments.map((a) => [a.version, a.fromVersion]), [[2, 1], [3, 2]]);
  const reroute = r.amendments[0];
  assert.deepEqual(reroute.kinds, ["REROUTE", "FIXES_ADDED", "FIXES_REMOVED"]);
  assert.deepEqual(reroute.change.fixesAdded, ["PXN", "EHF", "GMN"]);
  assert.deepEqual(reroute.change.fixesRemoved, ["AVE", "FIM"]);
  assert.equal(reroute.change.geometry.firstDifferingLeg!.from!.name, "SNS");
  assert.equal(reroute.change.geometry.firstDifferingLeg!.to!.name, "PXN");
  assert.ok(reroute.change.geometry.maxLateralNm! > 25 && reroute.change.geometry.maxLateralNm! < 35);
  assert.equal(reroute.at, amendAt);
  assert.equal(reroute.atBasis, "SWIM_MESSAGE_TIME");
  assert.equal(reroute.receivedAtMs, amendAt + 3000);
  assert.equal(reroute.detail, "route text changed; expanded route changed (3→4 points)");
  // the marker: the real fix 10 s earlier, said to be 10 s away, not interpolated
  assert.equal(reroute.position!.method, "NEAREST");
  assert.equal(reroute.position!.interpolated, false);
  assert.equal(reroute.position!.fixTimeMs, fixes[atSns].t * 1000);
  assert.equal(reroute.position!.gapSec, 10);
  assert.ok(haversineNm(reroute.position!, FIX.SNS) < 4);
  assert.equal(reroute.positionReason, null);
  const cruise = r.amendments[1];
  assert.deepEqual(cruise.kinds, ["ALTITUDE_CHANGE"]);
  assert.deepEqual(cruise.change.cruiseAltFt, { from: 24000, to: 28000, deltaFt: 4000 });
  assert.equal(cruise.atBasis, "OUR_RECEIVE_TIME");
  assert.match(cruise.atNote!, /no timestamp/);
  // received 5 min 40 s after the fix at SNS: the next fix is 20 s later and is used as it is
  assert.equal(cruise.position!.method, "NEAREST");
  assert.equal(cruise.position!.gapSec, 20);
  assert.equal(cruise.position!.fixTimeMs, fixes[atSns + 6].t * 1000);
  assert.equal(r.flownTrack.url, `/api/data/track/aircraft/${HEX}`);
  assert.match(r.honesty, /matched to this aircraft by callsign/);
  assert.match(r.honesty, /No cause is given/);
  // ?geometry=all puts the path on every version
  const all = await resolvePlanHistory(q(), ctxFor(w, { recent: fixes, now: amendAt + 20 * MIN }), { allGeometry: true });
  assert.ok(all.versions.every((v) => Array.isArray(v.points) && v.points.length > 10));
});

test("plan-history: amendment position is interpolated-and-flagged between sparse fixes, and null with a reason when nothing is near", async () => {
  const w = wired();
  w.swim.upsert(FH1(T0 - 30 * MIN), T0 - 30 * MIN);
  const a = interpolateGC(FIX.SNS, FIX.AVE, 0.2), b = interpolateGC(FIX.SNS, FIX.AVE, 0.6);
  const sparse = [{ t: T0 / 1000, la: a.lat, lo: a.lon, al: 7300 }, { t: (T0 + 4 * MIN) / 1000, la: b.lat, lo: b.lon, al: 7900 }];
  w.swim.upsert(AH_CRUISE(T0 + 2 * MIN), T0 + 2 * MIN);             // midway between two fixes 4 min apart
  w.swim.upsert(AH_CRUISE(T0 + 30 * MIN, 30000), T0 + 30 * MIN);    // 26 min after the last fix
  const r = await resolvePlanHistory(q(), ctxFor(w, { recent: sparse, now: T0 + 31 * MIN }));
  const [mid, late] = r.amendments;
  assert.equal(mid.position!.method, "INTERPOLATED");
  assert.equal(mid.position!.interpolated, true);
  assert.deepEqual(mid.position!.between, { beforeMs: T0, afterMs: T0 + 4 * MIN, spanSec: 240 });
  assert.ok(haversineNm(mid.position!, interpolateGC(a, b, 0.5)) < 0.05);
  assert.equal(late.position, null);
  assert.match(late.positionReason!, /nearest ADS-B fix is 1560 s from that time \(tolerance 30 s\) and the track ends before that time/);
  // no track at all
  const none = await resolvePlanHistory(q(), ctxFor(w, { recent: [], now: T0 + 31 * MIN }));
  assert.equal(none.amendments[0].position, null);
  assert.equal(none.amendments[0].positionReason, "no ADS-B fixes are held for this aircraft");
  assert.equal(none.offRoute.assessed, false);
  assert.match(none.offRoute.reason!, /no ADS-B fixes are held/);
});

test("off route WITHOUT an amendment: the episode has its start, end, positions and largest distance, and no cause", async () => {
  const w = wired();
  w.swim.upsert(FH1(T0 - 30 * MIN), T0 - 30 * MIN);
  // along SNS -> AVE (100 nm, well clear of both terminal areas), 15 nm off the route for fixes 5..9
  const fixes = fly([[FIX.SNS, FIX.AVE]], T0, 7, (i) => (i >= 5 && i <= 9 ? 15 : 0));
  const r = await resolvePlanHistory(q(), ctxFor(w, { recent: fixes, now: T0 + 16 * MIN }));
  assert.equal(r.offRoute.assessed, true);
  assert.equal(r.offRoute.fixesJudged, fixes.length);
  assert.deepEqual(r.offRoute.thresholds, { offNm: 8, onNm: 4, consecutiveFixes: 3, terminalAreaNm: 40, amendmentLinkSec: 600 });
  assert.equal(r.offRoute.episodes.length, 1);
  const e = r.offRoute.episodes[0];
  assert.equal(e.startMs, T0 + 5 * MIN, "the first fix off the route");
  assert.equal(e.detectedAtMs, T0 + 7 * MIN, "the state machine's third consecutive fix");
  assert.equal(e.endMs, T0 + 10 * MIN);
  assert.equal(e.endReason, "REJOINED");
  assert.ok(Math.abs(e.start.lat - fixes[5].la) < 1e-9 && Math.abs(e.end!.lon - fixes[10].lo) < 1e-9);
  assert.ok(e.maxCrossTrackNm! > 14.5 && e.maxCrossTrackNm! < 15.5, `max ${e.maxCrossTrackNm}`);
  assert.equal(e.classification, "NO_AMENDMENT_RECEIVED");
  assert.match(e.cause, /^cause not determined/);
  assert.equal(e.linkedAmendmentVersion, null);
  assert.equal(e.againstVersion, 1);
  assert.equal(r.amendments.length, 0);
  assert.ok(!JSON.stringify(r.offRoute.episodes).includes("weather deviation"));
});

test("off route FOLLOWED by an amendment: the episode is a reroute linked to that version, and the new route is judged clean", async () => {
  const w = wired();
  w.swim.upsert(FH1(T0 - 30 * MIN), T0 - 30 * MIN);
  // filed SNS -> AVE; the aircraft turns at SNS towards PXN (the route it is about to be given)
  const fixes = fly([[FIX.SNS, FIX.PXN], [FIX.PXN, FIX.EHF]], T0, 7);
  const amendAt = T0 + 5 * MIN + 20_000; // 5 min after the turn, between two fixes
  w.swim.upsert(AH_REROUTE(amendAt), amendAt + 2000);
  const r = await resolvePlanHistory(q(), ctxFor(w, { recent: fixes, now: T0 + 19 * MIN }));
  assert.equal(r.offRoute.episodes.length, 1, "after the amendment the aircraft is on its route");
  const e = r.offRoute.episodes[0];
  assert.equal(e.againstVersion, 1);
  assert.equal(e.endReason, "AMENDED");
  assert.equal(e.classification, "AMENDMENT_FOLLOWED");
  assert.equal(e.linkedAmendmentVersion, 2);
  assert.equal(e.startMs, T0 + 2 * MIN, "14 nm past SNS the two routes are 9 nm apart");
  assert.equal(e.endMs, T0 + 5 * MIN, "the last off-route fix before the amendment");
  assert.equal(e.amendmentLagSec, 200);
  assert.match(e.cause, /^reroute: an amendment to the filed route took effect 200 s after/);
  assert.equal(r.amendments[0].version, 2);
  assert.equal(r.amendments[0].position!.method, "NEAREST");
  // the SAME track judged only against the current route (what the live tracker does on a first request)
  // would have looked clean from the start; judged per version it shows the real sequence
  const after = r.offRoute.episodes.filter((x) => x.againstVersion === 2);
  assert.equal(after.length, 0);
});

test("off route is not assessed where it cannot be: fixes from before we held the plan, and a route we could not place", async () => {
  const early = wired();
  const fixes = fly([[FIX.SNS, FIX.AVE]], T0, 7, (i) => (i >= 2 && i <= 6 ? 15 : 0));
  early.swim.upsert(TRACK_MSG(), T0 + 20 * MIN);       // we joined the flight after these fixes
  early.swim.upsert(FH1(null), T0 + 21 * MIN);
  const r = await resolvePlanHistory(q(), ctxFor(early, { recent: fixes, now: T0 + 22 * MIN }));
  assert.equal(r.plan!.originalIsFirstSeen, true);
  assert.match(r.plan!.originalNote, /as we FIRST SAW it, not necessarily as filed/);
  assert.equal(r.offRoute.assessed, false);
  assert.equal(r.offRoute.fixesNotJudged.beforeFirstVersion, fixes.length);
  assert.match(r.offRoute.reason!, /before we first held the plan/);
  assert.deepEqual(r.offRoute.episodes, [], "not judged is not 'on route'");

  const textOnly = wired();
  textOnly.swim.upsert(msg({ messageType: "FH", cruiseAltFt: 24000, routeText: "KSFO..ZZZZZ..KLAX", timestamp: T0 - MIN }), T0 - MIN);
  const t = await resolvePlanHistory(q(), ctxFor(textOnly, { recent: fixes, now: T0 + 22 * MIN }));
  assert.equal(t.original!.pathEstimated, true);
  assert.equal(t.offRoute.assessed, false);
  assert.equal(t.offRoute.fixesNotJudged.pathNotAssessable, fixes.length);
  assert.match(t.offRoute.reason!, /could not be placed well enough/);
});

test("LADD: nothing FAA-sourced is served, and the answer is the same whether or not versions exist", async () => {
  const w = wired();
  w.swim.upsert(FH1(T0 - 30 * MIN), T0 - 30 * MIN);
  const fixes = fly([[FIX.SNS, FIX.PXN], [FIX.PXN, FIX.EHF]], T0, 7);
  w.swim.upsert(AH_REROUTE(T0 + 5 * MIN), T0 + 5 * MIN);
  const seen: Array<[string, string | null]> = [];
  const ladd = ctxFor(w, { recent: fixes, now: T0 + 19 * MIN, isLadd: (h, c) => { seen.push([h, c]); return true; } });
  const r = await resolvePlanHistory(q(), ladd, { allGeometry: true, searchArchive: true });
  assert.deepEqual(seen, [[HEX, "SKW5000"]], "checked by hex and callsign, like /plan");
  assert.equal(r.privacy, "LADD");
  assert.equal(r.source, "NONE");
  assert.ok(r.honesty.includes(LADD_WITHHELD_NOTE));
  assert.deepEqual([r.plan, r.original, r.current], [null, null, null]);
  assert.deepEqual([r.versions, r.amendments, r.offRoute.episodes], [[], [], []]);
  assert.equal(r.offRoute.assessed, false);
  const body = JSON.stringify(r);
  for (const leak of ["SNS", "PXN", "KLAX", "KSFO", "24000", "AH", "FH\""]) assert.ok(!body.includes(leak), `leaked ${leak}`);
  // an aircraft with no plan at all gets the identical answer
  const empty = wired();
  const none = await resolvePlanHistory(q(), ctxFor(empty, { recent: fixes, now: T0 + 19 * MIN, isLadd: () => true }));
  assert.deepEqual({ ...none, generatedAt: 0 }, { ...r, generatedAt: 0 });
  // not LADD: the same aircraft's versions are served, with no privacy field
  const open = await resolvePlanHistory(q(), ctxFor(w, { recent: fixes, now: T0 + 19 * MIN, isLadd: () => false }));
  assert.equal(open.source, "FILED_FAA");
  assert.equal(open.privacy, undefined);
  assert.equal(open.versions.length, 2);
  // and the existing /plan answer for the LADD aircraft is unchanged: FAA plan withheld there too
  const plan = await resolveFlightPlan(q(), ladd);
  assert.notEqual(plan.source, "FILED_FAA");
  assert.equal(plan.privacy, "LADD");
});

test("no versions: said plainly, with the reason (no callsign, no FAA plan, recording off)", async () => {
  const w = wired();
  const noPlan = await resolvePlanHistory(q(), ctxFor(w));
  assert.equal(noPlan.source, "NONE");
  assert.match(noPlan.honesty, /no FAA SWIM flight plan is held for SKW5000/);
  assert.match(noPlan.honesty, /add archive=1/);
  const noCs = await resolvePlanHistory(parsePlanQuery(HEX, {}) as ReturnType<typeof q>, ctxFor(w));
  assert.match(noCs.honesty, /broadcasts no callsign/);
  const off = await resolvePlanHistory(q(), { ...ctxFor(w), versions: null });
  assert.match(off.honesty, /not being recorded on this server/);
  for (const r of [noPlan, noCs, off]) assert.deepEqual([r.versions, r.amendments, r.offRoute.episodes, r.kind], [[], [], [], "raw"]);
});

test("after a restart: versions written before it are read back from disk and joined to what memory holds", async () => {
  const dir = tmp();
  _resetPlanHistoryCache();
  try {
    // the run before the restart: filed, then rerouted (both lines reach the disk)
    const before = wired({ archive: new PlanVersionArchive({ baseDir: dir, env: {}, readFree: roomy, now: () => T0 + 10 * MIN }) });
    before.swim.upsert(FH1(T0), T0);
    before.swim.upsert(AH_REROUTE(T0 + 10 * MIN), T0 + 10 * MIN);
    before.store.archive!.flush();
    // the restart: an empty store, the plan's first-seen time restored from the snapshot
    const boot = T0 + 30 * MIN;
    const after = wired();
    after.swim.seedHistory(before.swim.exportHistory(), boot);
    after.swim.upsert({ ...AH_REROUTE(boot + MIN), cruiseAltFt: 24000 }, boot + MIN); // the current plan seen again, unchanged
    after.swim.upsert(AH_CRUISE(boot + 5 * MIN), boot + 5 * MIN); // and then really changed
    const mem = after.store.forCallsign("SKW5000")!;
    assert.equal(mem.originalAsFiled, false);
    assert.deepEqual(mem.versions.map((v) => v.n), [1, 2], "numbering restarts with the process");
    const r = await resolvePlanHistory(q(), ctxFor(after, { now: boot + 6 * MIN }), { baseDir: dir, bootMs: boot });
    assert.equal(r.plan!.archive.searched, true);
    assert.equal(r.plan!.archive.versionsFound, 2);
    assert.equal(r.plan!.originalAsFiled, true, "the original on disk was a filing");
    assert.deepEqual(r.versions.map((v) => [v.version, v.heldIn, v.seq, v.receivedAtMs - T0]), [
      [1, "archive", 1, 0], [2, "archive", 2, 10 * MIN], [3, "memory", 2, 35 * MIN],
    ], "the plan seen again after the restart is not listed twice");
    assert.deepEqual(r.original!.fixes, ROUTE1);
    assert.deepEqual(r.amendments.map((a) => a.kinds), [["REROUTE", "FIXES_ADDED", "FIXES_REMOVED"], ["ALTITUDE_CHANGE"]]);
    // a flight that has left memory: found only when asked for
    const gone = wired();
    const silent = await resolvePlanHistory(q(), ctxFor(gone, { now: boot + 6 * MIN }), { baseDir: dir, bootMs: boot });
    assert.equal(silent.source, "NONE");
    _resetPlanHistoryCache();
    const asked = await resolvePlanHistory(q(), ctxFor(gone, { now: boot + 6 * MIN }), { baseDir: dir, bootMs: boot, searchArchive: true });
    assert.equal(asked.source, "FILED_FAA");
    assert.deepEqual(asked.versions.map((v) => v.heldIn), ["archive", "archive"]);
    assert.equal(asked.plan!.ended, true);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); _resetPlanHistoryCache(); }
});

test("the per-flight cap shows in the answer: missing versions are counted, and the diff across the gap is flagged", async () => {
  const w = wired({ maxPerFlight: 3 });
  w.swim.upsert(FH1(T0), T0);
  for (let i = 1; i <= 5; i++) w.swim.upsert(AH_CRUISE(null, 24000 + i * 1000), T0 + i * MIN);
  const r = await resolvePlanHistory(q(), ctxFor(w, { now: T0 + 10 * MIN }));
  assert.deepEqual(r.versions.map((v) => [v.seq, v.versionsMissingBefore]), [[1, 0], [5, 3], [6, 0]]);
  assert.equal(r.plan!.versionsMissing, 3);
  assert.equal(r.amendments[0].spansMissingVersions, true);
  assert.deepEqual(r.amendments[0].change.cruiseAltFt, { from: 24000, to: 28000, deltaFt: 4000 });
  assert.match(r.honesty, /3 more no longer held/);
});

// ── wiring ──────────────────────────────────────────────────────────────────

test("route + status: GET /api/data/aircraft/plan-history/:hex answers, bad hex is a 400, plan-status carries the counters, /plan is untouched", async () => {
  const lines: string[] = [];
  const archive = new PlanVersionArchive({ env: {}, readFree: roomy, now: () => T0, append: (ls) => { lines.push(...ls); } });
  const w = wired({ archive });
  w.swim.upsert(FH1(T0), T0);
  w.swim.upsert(AH_REROUTE(T0 + MIN), T0 + MIN);
  archive.flush();
  const ctx = ctxFor(w, { now: T0 + 2 * MIN, isLadd: () => false });
  const app = express();
  registerFlightPlanHistoryRoutes(app, ctx);
  const server = app.listen(0);
  try {
    const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    const ok = await fetch(`${base}/api/data/aircraft/plan-history/${HEX}?callsign=SKW5000`);
    assert.equal(ok.status, 200);
    const body = await ok.json() as PlanHistoryResponse;
    assert.deepEqual(Object.keys(body).sort(), [
      "amendments", "callsign", "current", "flownTrack", "generatedAt", "hex", "honesty", "kind", "label", "offRoute",
      "original", "plan", "provenance", "source", "versions",
    ]);
    assert.equal(body.versions.length, 2);
    assert.equal((await fetch(`${base}/api/data/aircraft/plan-history/nothex`)).status, 400);
  } finally { server.close(); }
  const st = planStatus(ctx).planVersions!;
  assert.deepEqual([st.flightsHeld, st.versionsHeld, st.versionsRecorded], [1, 2, 2]);
  assert.ok(st.estimatedBytes > 0);
  assert.deepEqual([st.archive!.linesWritten, st.archive!.bytesWritten], [2, lines.reduce((n, l) => n + Buffer.byteLength(l) + 1, 0)]);
  assert.deepEqual(Object.keys(st.archive!.skipped).sort(), ["buffer_full", "day_cap", "disabled", "low_disk", "write_error"]);
  assert.deepEqual(Object.keys(st.versionsDropped).sort(), ["bytes_cap", "ended_grace_over", "flights_cap", "per_flight_cap", "plan_expired", "total_cap"]);
  assert.equal(planStatus({ ...ctx, versions: null }).planVersions, null);
  // the existing endpoint's answer for the same aircraft: same contract as before
  const plan = await resolveFlightPlan(q(), ctx);
  assert.equal(plan.source, "FILED_FAA");
  assert.equal(plan.planHistory!.amendments, 1);
  assert.equal("versions" in plan, false);
});

test("routes.ts registers the endpoint next to the plan routes; the default context records versions", () => {
  const src = fs.readFileSync(path.join(path.dirname(new URL(import.meta.url).pathname), "routes.ts"), "utf8");
  assert.match(src, /import \{ registerFlightPlanHistoryRoutes \} from "\.\/flightPlanHistory";/);
  const at = src.indexOf("\n  registerFlightPlanRoutes(app);");
  assert.ok(at > 0);
  assert.match(src.slice(at, at + 400), /\n  registerFlightPlanRoutes\(app\);[^\n]*\n  registerFlightPlanHistoryRoutes\(app\);/);
  const plans = fs.readFileSync(path.join(path.dirname(new URL(import.meta.url).pathname), "flightPlans.ts"), "utf8");
  assert.match(plans, /onPlanContent: \(plan, msg, now\) => \{ versions\.record\(plan, msg, now\); \}/);
  assert.match(plans, /onRemoved: \(key, reason\) => \{ versions\.planRemoved\(key, reason, Date\.now\(\)\); \}/);
  assert.match(plans, /isLadd: \(cs\) => laddRegistry\(\)\.isLadd\(null, cs\)/);
  assert.equal(typeof registerFlightPlanRoutes, "function");
});
