// Everything Graph R1 battery (BUILD ORDER 3 #5): cross-stream join over
// synthetic archive day-files shaped exactly like the real writers emit
// (nwsAlerts AlertRec, nasaFirms FireDetection, usgsWater GaugeObs,
// datacoreArchive position lines).
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { buildSiteTimelines, foldEventStreams, _resetSiteTimelineCache, RADIUS_KM, EVENTS_CAP, type SiteRef } from "./siteTimeline";

beforeEach(() => _resetSiteTimelineCache());

const CUSHING: SiteRef = { id: "cushing_hub", name: "Cushing", lat: 35.94, lon: -96.74 };
const PORT_LA: SiteRef = { id: "port_la", name: "Port of LA", lat: 33.73, lon: -118.26 };
const SITES = [CUSHING, PORT_LA];
const NOW = Date.parse("2026-07-05T12:00:00Z");
const DAY = "2026-07-05";

function writeDay(base: string, kind: string, day: string, recs: any[], gz = false) {
  const dir = path.join(base, kind);
  fs.mkdirSync(dir, { recursive: true });
  const body = recs.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(body));
  else fs.writeFileSync(path.join(dir, `${day}.jsonl`), body);
}

test("foldEventStreams: joins each stream to the site within radius only", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tl-"));
  writeDay(base, "nwsalerts", DAY, [
    { id: "a1", event: "Tornado Warning", severity: "Extreme", sent: "2026-07-05T10:00:00Z", lat: 35.9, lon: -96.7 },   // ~7 km from Cushing
    { id: "a2", event: "Flood Watch", severity: "Moderate", sent: "2026-07-05T09:00:00Z", lat: 40.0, lon: -90.0 },      // far from both
    { id: "a3", event: "Zone Alert", severity: "Minor", sent: "2026-07-05T08:00:00Z", lat: null, lon: null },           // zone-only -> excluded
  ]);
  writeDay(base, "fires", DAY, [
    { id: "f1", lat: 33.8, lon: -118.3, confidence: "high", frp: 12.5, acq_date: DAY },   // near Port LA
  ], true);
  writeDay(base, "usgswater", DAY, [
    { site: "07153000", name: "Cimarron River nr Cushing", param: "00065", d: "2026-07-05T06:00:00Z", v: 4.2, q: "P", lat: 36.0, lon: -96.77 },
  ]);
  const events = foldEventStreams(SITES, base, NOW);
  const cushing = events.get("cushing_hub")!;
  assert.deepEqual(cushing.map((e) => e.kind).sort(), ["alert", "gauge"]);
  assert.equal(cushing.find((e) => e.kind === "alert")!.label, "Tornado Warning");
  const la = events.get("port_la")!;
  assert.equal(la.length, 1);
  assert.equal(la[0].kind, "fire");
  assert.equal(la[0].value, 12.5, "fire events carry FRP as the value");
});

test("buildSiteTimelines: position density per day within radius; absent days absent", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tl-"));
  const t0 = Math.floor(NOW / 1000) - 3600;
  // hourly-file naming like the real writer: YYYY-MM-DD-HH.jsonl
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-07-05-11.jsonl"), [
    { t: t0, i: "aaa", la: 35.95, lo: -96.75 },      // near Cushing
    { t: t0 + 60, i: "bbb", la: 35.96, lo: -96.73 }, // near Cushing
    { t: t0 + 120, i: "ccc", la: 51.0, lo: 0.0 },    // far
  ].map((r) => JSON.stringify(r)).join("\n") + "\n");
  const timelines = await buildSiteTimelines(SITES, base, NOW);
  const c = timelines.get("cushing_hub")!;
  assert.equal(c.density[DAY].a, 2);
  assert.equal(c.density[DAY].v, 0);
  assert.equal(Object.keys(timelines.get("port_la")!.density).length, 0, "no traffic near LA -> absent, not zero");
});

// REPAIR (found 2026-07-22, same root cause + fix as datacoreArchive.ts's
// streamJsonlLines): readline.Interface re-emits a piped-in stream's error
// on ITSELF too, independent of the stream.on("error", ...) guard here —
// unlistened, a truncated/corrupt .gz crashed the WHOLE PROCESS. See
// datacoreArchive.test.ts for the full writeup + minimal repro.
test("buildSiteTimelines resolves (never crashes the process) on a truncated/corrupt gzip file", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tl-trunc-"));
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  const t0 = Math.floor(NOW / 1000) - 3600;
  const good = zlib.gzipSync(JSON.stringify({ t: t0, i: "aaa", la: 35.95, lo: -96.75 }) + "\n");
  fs.writeFileSync(path.join(dir, "2026-07-05-11.jsonl.gz"), good.subarray(0, good.length - 4));
  await buildSiteTimelines(SITES, base, NOW);
});

test("events sort newest-first and cap honestly", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tl-"));
  const many = Array.from({ length: 20 }, (_, i) => ({
    id: `a${i}`, event: `Alert ${i}`, severity: "Minor",
    sent: `2026-07-05T${String(i).padStart(2, "0")}:00:00Z`, lat: 35.94, lon: -96.74,
  }));
  writeDay(base, "nwsalerts", DAY, many);
  return buildSiteTimelines([CUSHING], base, NOW).then((timelines) => {
    const ev = timelines.get("cushing_hub")!.events;
    assert.equal(ev.length, EVENTS_CAP);
    assert.equal(ev[0].label, "Alert 19", "newest first");
  });
});

test("missing archive dirs degrade to empty timelines, never throw", async () => {
  const timelines = await buildSiteTimelines(SITES, "/nonexistent/base", NOW);
  assert.equal(timelines.get("cushing_hub")!.events.length, 0);
  assert.ok(RADIUS_KM > 0);
});
