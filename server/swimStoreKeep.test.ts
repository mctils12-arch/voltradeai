// The FAA traffic-management and flight-restriction lists kept across a
// restart ([PIPELINE] 2026-11-21, open_questions 2026-11-20 (t6)). The
// messages are built in the element structure of the real ones the store
// tests use (swimTfms.test.ts, swimNotam.test.ts).
// Run: npx tsx --test server/swimStoreKeep.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import express from "express";
import { fileURLToPath } from "node:url";
import { TfmsStore, TFMS_PROGRAM_DEFAULT_TTL_MS } from "./swimTfms";
import { NotamStore } from "./swimNotam";
import { createSwimKeeper, restoreSwimStores, swimKeepEnabled, swimKeepFile, SWIM_KEEP_MAX_AGE_MS, SWIM_KEEP_VERSION } from "./swimStoreKeep";
import { keptFor, registerSwimCaptureRoutes, sharedNotamStore, sharedTfmsStore, startSwimStoreKeep, swimKeepView, _resetSwimCaptureForTests } from "./swimCapture";

const NOW = Date.parse("2026-11-20T23:30:00Z");
const MIN = 60_000;
const fi = (msgType: string, body: string, src = "TFMS", ts = "2026-11-20T23:21:42Z") =>
  `<?xml version="1.0"?><ds:tfmDataService xmlns:ds="urn:a" xmlns:fi="urn:b" xmlns:fcm="urn:c" xmlns:fce="urn:d"><ds:fiOutput><fi:fiMessage sourceFacility="${src}" sourceTimeStamp="${ts}" msgType="${msgType}">${body}</fi:fiMessage></ds:fiOutput></ds:tfmDataService>`;
const rstr = (id: string, airport: string, stop: string, ts?: string) => fi("RSTR", `<fi:restrictionMessage><fcm:eventTime>2026-11-20T19:20:00Z</fcm:eventTime><fcm:facility>N90</fcm:facility><fcm:action>4</fcm:action><fcm:restrictionId>${id}</fcm:restrictionId><fcm:restrictedNasElements>COATE</fcm:restrictedNasElements><fcm:startTime>2026-11-20T20:45:00Z</fcm:startTime><fcm:stopTime>${stop}</fcm:stopTime><fcm:airports>${airport}</fcm:airports><fcm:restrictionType>DEPARTURE</fcm:restrictionType><fcm:restrictionCategory>APREQ</fcm:restrictionCategory><fcm:reasonText>VOL:Compacted Demand</fcm:reasonText></fi:restrictionMessage>`, "N90", ts);
const aptc = (airport: string, at: string) => fi("APTC", `<fi:airportConfigMessage><fcm:eventTime>${at}</fcm:eventTime><fcm:facility>M98</fcm:facility><fcm:airport>${airport}</fcm:airport><fcm:arrRunwayConf>12L/12R</fcm:arrRunwayConf><fcm:depRunwayConf>12L/12R</fcm:depRunwayConf><fcm:arrRate>62</fcm:arrRate><fcm:depRate>66</fcm:depRate><fcm:updateTime>${at}</fcm:updateTime><fcm:weather>LVMC</fcm:weather></fi:airportConfigMessage>`, "M98", at);
const tfr = (id: string, start: string, end: string, type = "N") =>
  `<?xml version="1.0"?><message:AIXMBasicMessage xmlns:message="m" xmlns:gml="g" xmlns:event="e" xmlns:aixm="a" xmlns:fnse="f"><message:hasMember><event:Event gml:id="E1"><event:timeSlice><event:EventTimeSlice gml:id="T1"><event:scenario>15</event:scenario><event:textNOTAM><event:NOTAM gml:id="N1"><event:series>A</event:series><event:number>1381</event:number><event:year>2026</event:year><event:type>${type}</event:type><event:issued>2026-11-20T20:00:00.000Z</event:issued><event:selectionCode></event:selectionCode><event:minimumFL>000</event:minimumFL><event:maximumFL>180</event:maximumFL><event:coordinates>3854N07702W</event:coordinates><event:radius>30</event:radius><event:location>DCA</event:location><event:effectiveStart>${start}</event:effectiveStart><event:effectiveEnd>${end}</event:effectiveEnd><event:text>VIP MOVEMENT ... 99.7 SPECIAL SECURITY INSTRUCTIONS</event:text></event:NOTAM></event:textNOTAM><event:extension><fnse:EventExtension gml:id="X1"><fnse:classification>FDC</fnse:classification><fnse:xovernotamID>${id}</fnse:xovernotamID><fnse:icaoLocation>KDCA</fnse:icaoLocation></fnse:EventExtension></event:extension></event:EventTimeSlice></event:timeSlice></event:Event></message:hasMember></message:AIXMBasicMessage>`;

function tmpFile(): string {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), "swimkeep-")), "swim_active_stores.json");
}
function filled(): { tfms: TfmsStore; notam: NotamStore } {
  const tfms = new TfmsStore({ flightStore: false });
  tfms.ingest(rstr("1001", "TEB", "2026-11-21T01:45:00Z"), NOW);
  tfms.ingest(rstr("1002", "ORD", "2026-11-21T03:00:00Z"), NOW);
  tfms.ingest(aptc("MSP", "2026-11-20T21:01:00Z"), NOW);
  const notam = new NotamStore();
  notam.ingest(tfr("6/2222", "202611202200", "202611210400"), NOW);
  return { tfms, notam };
}
const quiet = () => {};

test("a restart keeps the list: what one process saved, the next one serves", async () => {
  const file = tmpFile();
  const a = filled();
  const k1 = createSwimKeeper(a.tfms, a.notam, { file, enabled: true, now: () => NOW, log: quiet, intervalMs: 3_600_000 });
  assert.equal(k1.start()?.cause, "no_file", "the first boot has nothing to read and says so");
  // the messages were ingested before start(), so the mark already equals them: a new one makes the save due
  a.tfms.ingest(rstr("1003", "BOS", "2026-11-21T02:00:00Z"), NOW);
  assert.equal(await k1.saveNow(NOW), "ok");
  k1.stop();
  assert.equal(fs.existsSync(`${file}.${process.pid}.tmp`), false, "the file is renamed into place, nothing is left beside it");

  // the next process, two minutes later
  const b = { tfms: new TfmsStore({ flightStore: false }), notam: new NotamStore() };
  const k2 = createSwimKeeper(b.tfms, b.notam, { file, enabled: true, now: () => NOW + 2 * MIN, log: quiet, intervalMs: 3_600_000 });
  const r = k2.start()!;
  k2.stop();
  assert.deepEqual({ cause: r.cause, programs: r.programs, configs: r.configs, tfrs: r.tfrs, dropped: r.dropped }, { cause: "ok", programs: 3, configs: 1, tfrs: 1, dropped: 0 });
  assert.equal(r.saved_at, NOW);
  const act = b.tfms.active(NOW + 2 * MIN);
  assert.deepEqual(act.programs.map((p) => p.airport).sort(), ["BOS", "ORD", "TEB"]);
  assert.deepEqual(act.programs.find((p) => p.airport === "TEB"), a.tfms.active(NOW).programs.find((p) => p.airport === "TEB"), "a restored row is the saved row, field for field");
  assert.equal(b.tfms.airportConfig("KMSP")?.aar, 62);
  assert.equal(b.notam.active(NOW + 2 * MIN).length, 1);
  assert.equal(b.notam.active(NOW + 2 * MIN)[0].geometry?.kind, "circle");
});

test("nothing is written while no message arrived, and a second start does nothing", async () => {
  const file = tmpFile();
  const a = filled();
  const k = createSwimKeeper(a.tfms, a.notam, { file, enabled: true, now: () => NOW, log: quiet, intervalMs: 3_600_000 });
  k.start();
  assert.equal(await k.saveNow(NOW), "unchanged", "start() took the mark: what was read back is what the file holds");
  assert.equal(fs.existsSync(file), false);
  a.notam.ingest(tfr("6/3333", "202611202200", "202611210400"), NOW);
  assert.equal(await k.saveNow(NOW), "ok");
  const m1 = fs.statSync(file).mtimeMs;
  assert.equal(await k.saveNow(NOW + MIN), "unchanged");
  assert.equal(fs.statSync(file).mtimeMs, m1);
  assert.equal(k.start()?.cause, "no_file", "a second start returns the first restore and reads nothing again");
  const v = k.view();
  k.stop();
  assert.deepEqual({ saves: v.saves, failures: v.save_failures, last: v.last_save, at: v.last_saved_at }, { saves: 1, failures: 0, last: "unchanged", at: NOW });
});

test("a row past its end is not put back; a row held already is not overwritten by an older saved copy", async () => {
  const file = tmpFile();
  const a = filled();
  const k1 = createSwimKeeper(a.tfms, a.notam, { file, enabled: true, now: () => NOW, log: quiet, intervalMs: 3_600_000 });
  k1.start();
  a.tfms.ingest(aptc("DEN", "2026-11-20T22:00:00Z"), NOW);
  await k1.saveNow(NOW);
  k1.stop();

  // five hours later: the saved TEB and ORD restrictions ended 01:45Z and 03:00Z (both over an hour ago), the TFR ended 04:00Z
  const later = Date.parse("2026-11-21T04:30:00Z");
  const b = { tfms: new TfmsStore({ flightStore: false }), notam: new NotamStore() };
  // a newer ORD message reached this process before the restore
  b.tfms.ingest(rstr("1002", "ORD", "2026-11-21T06:00:00Z", "2026-11-21T04:29:00Z"), later);
  const r = restoreSwimStores(b.tfms, b.notam, { file, now: later });
  assert.equal(r.cause, "ok");
  assert.equal(r.programs, 0, "both saved restrictions are past their end");
  assert.equal(r.tfrs, 0);
  assert.equal(r.configs, 2);
  assert.equal(r.dropped, 3, "the two ended restrictions and the ended flight restriction are counted, not silently lost");
  assert.equal(b.tfms.active(later).programs.length, 1);
  assert.equal(b.tfms.active(later).programs[0].end, Date.parse("2026-11-21T06:00:00Z"), "the newer message's end stands");

  // a saved copy still in force but OLDER than the row held: the held row stands
  const c = new TfmsStore({ flightStore: false });
  c.ingest(rstr("1002", "ORD", "2026-11-21T06:00:00Z", "2026-11-20T23:29:00Z"), NOW);
  const held = c.active(NOW).programs[0];
  const older = { ...held, end: Date.parse("2026-11-21T03:00:00Z"), updatedAt: held.updatedAt - MIN };
  assert.deepEqual(c.restore({ programs: [older], configs: [] }, NOW), { programs: 0, configs: 0, dropped: 0 });
  assert.equal(c.active(NOW).programs[0].end, Date.parse("2026-11-21T06:00:00Z"));
  // and a saved copy NEWER than the row held replaces it
  assert.equal(c.restore({ programs: [{ ...held, end: Date.parse("2026-11-21T07:00:00Z"), updatedAt: held.updatedAt + MIN }], configs: [] }, NOW).programs, 1);
  assert.equal(c.active(NOW).programs[0].end, Date.parse("2026-11-21T07:00:00Z"));
});

test("an airport configuration is put back whatever its report time: the FAA re-sends a standing report with its original time", () => {
  // production 2026-11-21 14:20Z: LAX's held report was 22.8 h old on a server 11 h up
  const store = new TfmsStore({ flightStore: false });
  const r = store.restore({
    programs: [],
    configs: [
      { airport: "LAX", arrRunways: "24R", depRunways: "25R", aar: 60, adr: 60, weather: "VMC", facility: "SCT", at: NOW - 30 * 3_600_000 },
      { airport: "DEN", arrRunways: "16L", depRunways: "17R", aar: 96, adr: 90, weather: "VMC", facility: "D01", at: NOW - 3_600_000 },
      { airport: "ORD", arrRunways: "27L", depRunways: "28R", aar: 90, adr: 90, weather: "VMC", facility: "C90", at: NOW + 3_600_000 },
    ],
  }, NOW);
  assert.deepEqual(r, { programs: 0, configs: 2, dropped: 1 }, "only the report dated in the future is refused");
  assert.equal(store.airportConfig("LAX")?.at, NOW - 30 * 3_600_000, "it keeps its own report time, which the card prints");
  assert.equal(store.airportConfig("ORD"), null);
});

test("a program with no end is kept for its default lifetime from its last update, measured across the restart", () => {
  const store = new TfmsStore({ flightStore: false });
  const base = { kind: "ADVISORY", status: null, airport: "LGA", facility: null, start: null, end: null, title: "ADVZY", text: null, reason: null, elements: [], geometry: null, floorFl: null, ceilingFl: null };
  const r = store.restore({ programs: [
    { ...base, id: "fresh", updatedAt: NOW - 3_600_000 },
    { ...base, id: "stale", updatedAt: NOW - TFMS_PROGRAM_DEFAULT_TTL_MS - 3_600_000 - 1 },
  ], configs: [] }, NOW);
  assert.deepEqual(r, { programs: 1, configs: 0, dropped: 1 });
  assert.deepEqual(store.active(NOW).programs.map((p) => p.id), ["fresh"]);
});

test("a broken, foreign or old file restores nothing and names the cause; malformed rows are skipped and counted", () => {
  const stores = () => ({ tfms: new TfmsStore({ flightStore: false }), notam: new NotamStore() });
  const write = (body: string) => { const f = tmpFile(); fs.writeFileSync(f, body); return f; };
  const run = (f: string, now = NOW) => { const s = stores(); return { r: restoreSwimStores(s.tfms, s.notam, { file: f, now }), s }; };
  assert.equal(run(path.join(os.tmpdir(), "swimkeep-absent", "none.json")).r.cause, "no_file");
  assert.equal(run(write("{ not json")).r.cause, "unreadable");
  assert.equal(run(write("null")).r.cause, "unreadable");
  assert.equal(run(write(JSON.stringify({ v: SWIM_KEEP_VERSION + 1, saved_at: NOW, tfms: { programs: [], configs: [] }, tfrs: [] }))).r.cause, "wrong_version");
  const old = run(write(JSON.stringify({ v: SWIM_KEEP_VERSION, saved_at: NOW - SWIM_KEEP_MAX_AGE_MS - 1, tfms: { programs: [], configs: [] }, tfrs: [] })));
  assert.equal(old.r.cause, "too_old");
  assert.equal(old.r.saved_at, NOW - SWIM_KEEP_MAX_AGE_MS - 1, "the page can still say how old the refused file was");
  assert.equal(run(write(JSON.stringify({ v: SWIM_KEEP_VERSION, saved_at: NOW + 10 * MIN, tfms: { programs: [], configs: [] }, tfrs: [] }))).r.cause, "too_old", "a file from the future is not trusted");
  const junk = run(write(JSON.stringify({
    v: SWIM_KEEP_VERSION, saved_at: NOW,
    tfms: { programs: [null, 7, { id: "", kind: "GS", updatedAt: NOW }, { id: "x", kind: "GS", updatedAt: "soon" }, { id: "ok", kind: "GS", status: null, airport: "TEB", start: null, end: NOW + MIN, title: null, updatedAt: NOW }], configs: [{ airport: 5, at: NOW }, "x"] },
    tfrs: [{ id: "a" }, { id: "b", geometry: { kind: "blob" } }, null],
  })));
  assert.deepEqual({ programs: junk.r.programs, configs: junk.r.configs, tfrs: junk.r.tfrs, dropped: junk.r.dropped }, { programs: 1, configs: 0, tfrs: 0, dropped: 9 });
  assert.deepEqual(junk.s.tfms.active(NOW).programs[0].elements, [], "a row saved without its list fields is served with empty ones, never undefined");
});

test("switched off: nothing is read or written", async () => {
  assert.equal(swimKeepEnabled({}), true, "on by default");
  for (const v of ["off", "0", "false", "no", " OFF "]) assert.equal(swimKeepEnabled({ VOLTRADE_SWIM_STORE_KEEP: v }), false);
  const file = tmpFile();
  fs.writeFileSync(file, JSON.stringify({ v: SWIM_KEEP_VERSION, saved_at: NOW, tfms: filled().tfms.snapshot(NOW), tfrs: [] }));
  const before = fs.readFileSync(file, "utf8");
  const s = { tfms: new TfmsStore({ flightStore: false }), notam: new NotamStore() };
  const k = createSwimKeeper(s.tfms, s.notam, { file, enabled: false, now: () => NOW, log: quiet });
  assert.equal(k.start(), null);
  s.tfms.ingest(rstr("9", "TEB", "2026-11-21T01:45:00Z"), NOW);
  assert.equal(await k.saveNow(NOW), "unchanged");
  assert.equal(s.tfms.active(NOW).programs.length, 1, "only the row this process ingested");
  assert.equal(fs.readFileSync(file, "utf8"), before);
  assert.equal(k.view().enabled, false);
  assert.equal(keptFor("tfms", k.view()), null, "a route then says the list is not kept");
});

test("a failed write is counted and named, and the next minute tries again", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "swimkeep-"));
  // the target's parent is a FILE, so the write cannot succeed
  fs.writeFileSync(path.join(dir, "blocker"), "x");
  const file = path.join(dir, "blocker", "swim_active_stores.json");
  const a = filled();
  const logs: string[] = [];
  const k = createSwimKeeper(a.tfms, a.notam, { file, enabled: true, now: () => NOW, log: (m) => logs.push(m), intervalMs: 3_600_000 });
  k.start();
  a.tfms.ingest(rstr("1003", "BOS", "2026-11-21T02:00:00Z"), NOW);
  assert.equal(await k.saveNow(NOW), "write_failed");
  assert.equal(await k.saveNow(NOW), "write_failed", "the mark is not taken on a failure, so the save stays due");
  k.stop();
  assert.equal(k.view().save_failures, 2);
  assert.equal(logs.filter((l) => /save failed/.test(l)).length, 1, "one line for a run of failures, not one a minute");
});

test("the file sits on the volume beside the archive folder, not inside it", () => {
  assert.equal(swimKeepFile("/data/voltrade"), "/data/voltrade/swim_active_stores.json");
  assert.equal(path.basename(path.dirname(swimKeepFile())) === "datacore_archive", false);
});

test("wiring: no keeper without a configured feed; with one, both routes state what was carried over, in counts and times only", async () => {
  _resetSwimCaptureForTests();
  assert.equal(startSwimStoreKeep({ env: {}, log: quiet }), null, "a process with no FAA feed neither reads nor writes the file");
  assert.equal(swimKeepView(), null);

  const file = tmpFile();
  const prev = filled();
  fs.writeFileSync(file, JSON.stringify({ v: SWIM_KEEP_VERSION, saved_at: Date.now() - 2 * MIN, tfms: { programs: [], configs: prev.tfms.snapshot(NOW).configs.map((c) => ({ ...c, at: Date.now() - 5 * MIN })) }, tfrs: [] }));
  const env: NodeJS.ProcessEnv = {};
  for (const k of ["URL", "VPN", "QUEUE", "USERNAME", "PASSWORD", "HOST", "USER"]) env[`SWIM_TFMS_${k}`] = "x";
  const { swimMissingEnv } = await import("./swimConnector");
  assert.deepEqual(swimMissingEnv("SWIM_TFMS", env), [], "the stand-in environment names every variable the connector asks for");
  const keeper = startSwimStoreKeep({ env, file, intervalMs: 3_600_000, log: quiet });
  assert.ok(keeper);
  assert.equal(startSwimStoreKeep({ env, file, log: quiet }), keeper, "one keeper per process");
  assert.equal(sharedTfmsStore().airportConfig("MSP")?.aar, 62, "the shared store the routes read was filled before any message");

  const app = express();
  registerSwimCaptureRoutes(app, {});
  const srv = app.listen(0);
  try {
    const base = `http://127.0.0.1:${(srv.address() as { port: number }).port}`;
    const t = await (await fetch(`${base}/api/data/swim/tfms/active`)).json() as { kept: Record<string, any> | null; airportConfigs: unknown[] };
    assert.equal(t.airportConfigs.length, 1);
    assert.deepEqual(Object.keys(t.kept!).sort(), ["last_save", "last_saved_at", "restored", "saved_every_s"]);
    assert.deepEqual(Object.keys(t.kept!.restored).sort(), ["airport_configs", "at", "cause", "programs", "saved_at"]);
    assert.deepEqual({ cause: t.kept!.restored.cause, programs: t.kept!.restored.programs, configs: t.kept!.restored.airport_configs }, { cause: "ok", programs: 0, configs: 1 });
    const n = await (await fetch(`${base}/api/data/swim/notam/tfrs`)).json() as { kept: Record<string, any> | null };
    assert.deepEqual(Object.keys(n.kept!.restored).sort(), ["at", "cause", "saved_at", "tfrs"]);
    assert.equal(n.kept!.restored.tfrs, 0);
    assert.equal(sharedNotamStore().active().length, 0);
  } finally {
    srv.close();
    _resetSwimCaptureForTests();
  }
  assert.equal(swimKeepView(), null, "the test seam stops the keeper");
});

test("RATCHET: startup starts the keeper, and the keeper never serialises per-flight records", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const fp = fs.readFileSync(path.join(here, "flightPlans.ts"), "utf8");
  assert.match(fp, /startSwimCapture\(\)[^\n]*\n[^]*?startSwimStoreKeep\(\)/, "registerFlightPlanRoutes must start the keeper beside the capture consumers");
  const tfms = fs.readFileSync(path.join(here, "swimTfms.ts"), "utf8");
  const snap = /snapshot\(now: number = Date\.now\(\)\): TfmsSnapshot \{[^]*?\n  \}/.exec(tfms)?.[0] ?? "";
  assert.ok(snap, "TfmsStore.snapshot not found");
  assert.doesNotMatch(snap, /this\.flights/, "per-flight amendments are LADD-gated and are not written to the file");
});
