import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { zipSync, strToU8 } from "fflate";
import {
  parseGdeltExport,
  parseLastUpdate,
  facilityBoxes,
  nearestFacility,
  archiveGdeltEvents,
  GDELT_ROOT_CODES,
  readRecentArchivedGdelt,
  refreshGdeltCache,
  latestGdeltEvents,
  _resetGdeltCacheForTests,
  type GdeltEvent,
} from "./gdeltEvents";

const here = path.dirname(fileURLToPath(import.meta.url));

// Builds a 61-column v2 Events row (indices verified against a real export
// 2026-07-05: id 0, day 1, code 26, root 28, gold 30, mentions 31, tone 34,
// lat 56, lon 57, url 60).
function row(over: Record<number, string>): string {
  const c = Array(61).fill("");
  c[0] = "1234567890"; c[1] = "20260705"; c[26] = "145"; c[28] = "14";
  c[30] = "-6.5"; c[31] = "12"; c[34] = "-3.2";
  c[56] = "35.95"; c[57] = "-96.76"; // inside the Cushing box
  c[60] = "https://example.com/article";
  for (const [k, v] of Object.entries(over)) c[Number(k)] = v;
  return c.join("\t");
}

test("facilityBoxes loads the strategic sites with coordinates", () => {
  const boxes = facilityBoxes();
  assert.ok(boxes.length >= 10, "strategic sites present");
  assert.ok(boxes.some((b) => b.id === "cushing_hub"));
  assert.equal(nearestFacility(35.95, -96.76, boxes), "cushing_hub");
  assert.equal(nearestFacility(0, 0, boxes), null);
});

test("parseGdeltExport: CAMEO + bbox filter — unrest near a facility kept, everything else dropped", () => {
  const boxes = facilityBoxes();
  const csv = [
    row({}),                                        // protest at Cushing -> kept
    row({ 0: "2", 26: "1431", 28: "14" }),          // strike code 143x -> kept
    row({ 0: "3", 26: "042", 28: "04" }),           // diplomatic consult -> dropped (code)
    row({ 0: "4", 56: "48.85", 57: "2.35" }),       // protest in Paris, no facility -> dropped (geo)
    row({ 0: "5", 56: "", 57: "" }),                // no geo -> dropped
    "short\trow",                                   // malformed -> dropped
  ].join("\n");
  const ev = parseGdeltExport(csv, boxes, "2026-07-05T06:00:00Z");
  assert.deepEqual(ev.map((e) => e.id), ["1234567890", "2"]);
  assert.equal(ev[0].site, "cushing_hub");
  assert.equal(ev[0].gold, -6.5);
  assert.equal(ev[0].tone, -3.2);
  assert.ok(GDELT_ROOT_CODES.has(ev[0].root));
});

test("parseLastUpdate finds the export zip and downgrades to http (invalid-cert host)", () => {
  const txt = [
    "24511 8f0d… http://data.gdeltproject.org/gdeltv2/20260705060000.export.CSV.zip",
    "171223 aa… http://data.gdeltproject.org/gdeltv2/20260705060000.mentions.CSV.zip",
    "1712230 bb… http://data.gdeltproject.org/gdeltv2/20260705060000.gkg.csv.zip",
  ].join("\n");
  assert.ok(parseLastUpdate(txt)!.endsWith("export.CSV.zip"));
  assert.ok(parseLastUpdate(txt.replace(/http:/g, "https:"))!.startsWith("http://"), "https downgraded");
  assert.equal(parseLastUpdate("garbage"), null);
});

test("archive round-trip with GlobalEventID dedup", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtgdelt-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const ev = parseGdeltExport(row({}), facilityBoxes(), "2026-07-05T06:00:00Z");
  assert.equal(archiveGdeltEvents(ev, dir, t0), 1);
  assert.equal(archiveGdeltEvents(ev, dir, t0), 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

// ── Cold-cache-no-disk-backfill (research/open_questions.md's module audit
// table — "gdeltEvents.ts | new reader needed") ────────────────────────────

const mkArchived = (id: string, rt: string): GdeltEvent => ({
  id, day: rt.slice(0, 10).replace(/-/g, ""), code: "145", root: "14",
  gold: -6.5, tone: -3.2, mentions: 12, lat: 35.95, lon: -96.76,
  site: "cushing_hub", url: null, rt,
});

// gdeltDir() (private to gdeltEvents.ts) joins baseDir + "gdelt" — mirrored
// here so day-files land where readRecentArchivedGdelt()/archiveGdeltEvents()
// actually look, same convention archiveGdeltEvents' own round-trip test uses.
function writeGdeltDay(baseDir: string, day: string, events: GdeltEvent[]): void {
  const dir = path.join(baseDir, "gdelt");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, `${day}.jsonl`), events.map((e) => JSON.stringify(e)).join("\n") + "\n");
}

test("readRecentArchivedGdelt: dedups across day-files, drops events outside the window, sorts newest-first", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtgdelt-backfill-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const today = new Date(t0).toISOString().slice(0, 10);
  const yesterday = new Date(t0 - 86400_000).toISOString().slice(0, 10);
  const stale = new Date(t0 - 5 * 86400_000).toISOString().slice(0, 10);
  writeGdeltDay(dir, today, [
    mkArchived("A", new Date(t0 - 3600_000).toISOString()),
    mkArchived("A", new Date(t0 - 3600_000).toISOString()), // duplicate id, same day
  ]);
  writeGdeltDay(dir, yesterday, [mkArchived("B", new Date(t0 - 30 * 3600_000).toISOString())]);
  writeGdeltDay(dir, stale, [mkArchived("C", new Date(t0 - 5 * 86400_000).toISOString())]);

  const out = readRecentArchivedGdelt(dir, t0, 48);
  assert.deepEqual(out.map((e) => e.id), ["A", "B"], "dedups A, keeps B (within 48h), drops C (outside window)");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("readRecentArchivedGdelt: empty archive returns []", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtgdelt-empty-"));
  assert.deepEqual(readRecentArchivedGdelt(dir, Date.now(), 48), []);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshGdeltCache: cold cache backfills from the on-disk archive when the live poll throws (network/GDELT outage on a fresh boot must not report warming_up over a real archive)", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtgdelt-cold-throw-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const today = new Date(t0).toISOString().slice(0, 10);
  writeGdeltDay(dir, today, [mkArchived("COLD-1", new Date(t0 - 3600_000).toISOString())]);
  _resetGdeltCacheForTests();
  try {
    assert.equal(latestGdeltEvents(), null, "cache must still be cold going into this cycle");
    const failing = async () => { throw new Error("GDELT unreachable"); };
    await refreshGdeltCache(failing as any, t0, dir);
    const cached = latestGdeltEvents();
    assert.ok(cached, "cache must be populated, not left null, despite the live poll throwing");
    assert.equal(cached!.events.length, 1);
    assert.equal(cached!.events[0].id, "COLD-1", "backfilled from the archive, not fabricated");
  } finally {
    _resetGdeltCacheForTests();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// Builds a mock fetch matching fetchGdeltEvents' two-call shape
// (lastupdate.txt, then the export zip it names), serving `csv` as the
// zip's sole entry under `exportName`.
function mockGdeltFetch(exportName: string, csv: string) {
  const zip = zipSync({ [exportName]: strToU8(csv) });
  const buf = zip.buffer.slice(zip.byteOffset, zip.byteOffset + zip.byteLength);
  return async (url: string) => url.includes("lastupdate")
    ? { ok: true, status: 200, text: async () => `24511 8f0d… http://data.gdeltproject.org/gdeltv2/${exportName}.zip`, arrayBuffer: async () => new ArrayBuffer(0) }
    : { ok: true, status: 200, text: async () => "", arrayBuffer: async () => buf };
}

test("refreshGdeltCache: cold cache also backfills on an empty-but-successful poll (a real export file with no facility-matching rows — the feed's normal quiet cycle, not an outage)", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtgdelt-cold-empty-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const today = new Date(t0).toISOString().slice(0, 10);
  writeGdeltDay(dir, today, [mkArchived("COLD-2", new Date(t0 - 3600_000).toISOString())]);
  _resetGdeltCacheForTests();
  try {
    const noMatchCsv = row({ 56: "48.85", 57: "2.35" }); // Paris — no tracked facility nearby
    const fetchImpl = mockGdeltFetch("20260705060000.export.CSV", noMatchCsv);
    await refreshGdeltCache(fetchImpl as any, t0, dir);
    const cached = latestGdeltEvents();
    assert.ok(cached, "cache must be populated from the archive on an empty-but-successful poll");
    assert.equal(cached!.events[0].id, "COLD-2");
  } finally {
    _resetGdeltCacheForTests();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("refreshGdeltCache: a transient throw never clobbers an already-warm cache with a stale archive read", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtgdelt-warm-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const today = new Date(t0).toISOString().slice(0, 10);
  writeGdeltDay(dir, today, [mkArchived("STALE-1", new Date(t0 - 3600_000).toISOString())]);
  _resetGdeltCacheForTests();
  try {
    const liveFetch = mockGdeltFetch("20260705070000.export.CSV", row({ 0: "LIVE-1" }));
    await refreshGdeltCache(liveFetch as any, t0, dir);
    const afterLive = latestGdeltEvents();
    assert.equal(afterLive?.events[0]?.id, "LIVE-1", "cache warmed from the real live filing, not the archive");

    const failing = async () => { throw new Error("transient network blip"); };
    await refreshGdeltCache(failing as any, t0 + 15 * 60_000, dir);
    const afterThrow = latestGdeltEvents();
    assert.equal(afterThrow?.events.map((e) => e.id).sort().join(","), "LIVE-1",
      "an already-warm cache must be left untouched by a transient throw, not clobbered with a stale archive read");
  } finally {
    _resetGdeltCacheForTests();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("routes.ts boots the GDELT poll and registers /api/data/facility-events; manifest carries attribution + approximation honesty", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootGdeltPoll()"), "GDELT poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/facility-events"'), "route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "gdelt.json"), "utf8"));
  assert.equal(manifest.stream, "gdelt");
  assert.ok(String(manifest.attribution).includes("GDELT"), "attribution required by license");
  assert.ok(String(manifest.field_map.lat).toUpperCase().includes("APPROXIMATE"), "geo approximation honesty");
  assert.ok(String(manifest.confidence_model).includes("NOT clean industrial accidents"), "CAMEO limitation stated");
});
