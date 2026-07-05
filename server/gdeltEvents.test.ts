import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  parseGdeltExport,
  parseLastUpdate,
  facilityBoxes,
  nearestFacility,
  archiveGdeltEvents,
  GDELT_ROOT_CODES,
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
