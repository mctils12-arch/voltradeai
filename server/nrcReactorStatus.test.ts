// NRC daily Power Reactor Status Reports battery: pipe-delimited parse,
// unit->registry-plant name reconciliation (gate 1), event-identity
// archive dedup, latest-day-only cache.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseReactorStatus, reactorStatusUrl, latestDay, normalizePlantName,
  matchToRegistry, EXPECTED_REGISTRY_ONLY, fetchReactorStatus,
  archiveReactorStatus, refreshReactorStatus, latestReactorStatus,
} from "./nrcReactorStatus";

const SAMPLE = [
  "ReportDt|Unit|Power",
  "8/3/2026 12:00:00 AM|Arkansas Nuclear 1|100",
  "8/3/2026 12:00:00 AM|Beaver Valley 1|97",
  "8/3/2026 12:00:00 AM|Palo Verde 2|0",
  "8/2/2026 12:00:00 AM|Arkansas Nuclear 1|100",
].join("\n");

test("url: one file per calendar year", () => {
  assert.equal(reactorStatusUrl(2026), "https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/2026/2026PowerStatus.txt");
});

test("parse: pipe-delimited, header dropped, dates normalized to YYYY-MM-DD, malformed lines skipped", () => {
  const rows = parseReactorStatus(SAMPLE);
  assert.equal(rows.length, 4);
  assert.deepEqual(rows[0], { date: "2026-08-03", unit: "Arkansas Nuclear 1", power: 100 });
  assert.equal(rows[2].power, 0, "0% (shutdown) is a real value, not treated as missing");
  assert.deepEqual(parseReactorStatus("garbage\n\nA|B"), []);
  assert.deepEqual(parseReactorStatus("1/1/2026|Unit X|"), [{ date: "2026-01-01", unit: "Unit X", power: null }]);
});

test("latestDay: max date string across rows", () => {
  assert.equal(latestDay(parseReactorStatus(SAMPLE)), "2026-08-03");
  assert.equal(latestDay([]), "");
});

test("normalizePlantName: strips punctuation, hyphens, and one suffix phrase", () => {
  assert.equal(normalizePlantName("Davis-Besse"), "davis besse");
  assert.equal(normalizePlantName("D.C. Cook"), "dc cook");
  assert.equal(normalizePlantName("Clinton Power Station"), "clinton");
  assert.equal(normalizePlantName("Wolf Creek Generating Station"), "wolf creek");
  assert.equal(normalizePlantName("Waterford 3"), "waterford 3", "no suffix word to strip; unit-strip is a separate step");
});

test("matchToRegistry: live-shaped panel matches 100%, with documented (not silent) retired-plant gaps", () => {
  const nrcUnits = [
    "Beaver Valley 1", "Beaver Valley 2", "Arkansas Nuclear 1", "D.C. Cook 1",
    "Davis-Besse", "Clinton", "River Bend Station 1", "Hope Creek 1", "Salem 1",
    "Susquehanna 1", "Watts Bar 1",
  ];
  const registry = [
    "Beaver Valley", "Arkansas Nuclear One", "Donald C Cook", "Davis Besse",
    "Clinton Power Station", "River Bend", "PSEG Hope Creek Generating Station",
    "PSEG Salem Generating Station", "TalenEnergy Susquehanna",
    "Watts Bar Nuclear Plant", "Indian Point 2",
  ];
  const r = matchToRegistry(nrcUnits, registry);
  assert.equal(r.matched, nrcUnits.length);
  assert.deepEqual(r.unmatched, []);
  assert.deepEqual(r.unexpectedRegistryGaps, [], "no silently-dropped registry entry");
  assert.ok(r.expectedGaps.includes("indian point 2"), "documented retired-plant gap surfaces, not silently dropped");
  assert.equal(r.matchRate, 1);
});

test("matchToRegistry: a genuinely unexplained registry gap is reported, never silently swallowed", () => {
  const r = matchToRegistry(["Beaver Valley 1"], ["Beaver Valley", "Some New Plant"]);
  assert.deepEqual(r.unexpectedRegistryGaps, ["Some New Plant"]);
  assert.deepEqual(r.expectedGaps, []);
});

test("matchToRegistry: an NRC unit with no registry counterpart is reported unmatched, not silently dropped", () => {
  const r = matchToRegistry(["Totally Unknown Unit 1"], ["Beaver Valley"]);
  assert.deepEqual(r.unmatched, ["Totally Unknown Unit 1"]);
  assert.equal(r.matched, 0);
});

test("EXPECTED_REGISTRY_ONLY carries only documented-retired plants", () => {
  assert.ok(EXPECTED_REGISTRY_ONLY.has("indian point 2"));
  assert.ok(EXPECTED_REGISTRY_ONLY.has("duane arnold"));
});

test("fetch: keeps only the newest day's rows, on non-ok logs and returns empty", async () => {
  const ok = async () => ({ ok: true, status: 200, text: async () => SAMPLE });
  const rows = await fetchReactorStatus(ok as any, Date.parse("2026-08-03T12:00:00Z"));
  assert.equal(rows.length, 3, "only 2026-08-03 rows, the 2026-08-02 Arkansas row excluded");
  assert.ok(rows.every((r) => r.date === "2026-08-03"));

  const bad = async () => ({ ok: false, status: 500, text: async () => "err" });
  assert.deepEqual(await fetchReactorStatus(bad as any), []);

  const throws = async () => { throw new Error("network down"); };
  assert.deepEqual(await fetchReactorStatus(throws as any), []);
});

test("archive: date|unit dedup across fetches and restarts", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nrc-"));
  const rows = parseReactorStatus(SAMPLE).filter((r) => r.date === "2026-08-03");
  const now = Date.parse("2026-08-03T12:00:00Z");
  assert.equal(archiveReactorStatus(rows, base, now), 3);
  assert.equal(archiveReactorStatus(rows, base, now), 0, "same date/unit never re-archives");
  const plus = [...rows, { date: "2026-08-04", unit: "Arkansas Nuclear 1", power: 100 }];
  assert.equal(archiveReactorStatus(plus, base, now), 1, "new day's row lands");
});

test("refresh: cache holds only the newest day's rows", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nrc-"));
  const ok = async () => ({ ok: true, status: 200, text: async () => SAMPLE });
  await refreshReactorStatus(ok as any, Date.parse("2026-08-03T12:00:00Z"), base);
  const hit = latestReactorStatus();
  assert.ok(hit);
  assert.equal(hit!.date, "2026-08-03");
  assert.ok(hit!.rows.every((r) => r.date === "2026-08-03"));
});
