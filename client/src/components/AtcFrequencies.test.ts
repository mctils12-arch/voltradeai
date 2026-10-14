// ATC frequency rows (2026-10-14). Run: npx tsx --test client/src/components/AtcFrequencies.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { fmtMhz, noteInUnits, artccLine, atcRows, ATC_MAX_ROWS, type AtcContextData } from "./AtcFrequencies";

const here = path.dirname(fileURLToPath(import.meta.url));
const base: AtcContextData = { phase: null, airport: null, artcc: null, entries: [], cycle: "2026-10-01", honesty: "x" };
const e = (call: string, mhz: number, likely: boolean, note: string | null = null) => ({ role: call.split(" ").pop()!, call, mhz, note, likely });

test("fmtMhz: as published — 118.1, 128.325, 120.05, 121.9", () => {
  assert.deepEqual([118.1, 128.325, 120.05, 121.9, 134.75].map(fmtMhz), ["118.1", "128.325", "120.05", "121.9", "134.75"]);
});

test("noteInUnits: published sector altitudes follow the units setting; other text verbatim", () => {
  assert.equal(noteInUnits("001-119 8000 FT & BLW", "imperial"), "001-119 8000 ft & BLW");
  assert.equal(noteInUnits("001-119 8000 FT & BLW", "metric"), "001-119 2438 m & BLW");
  assert.equal(noteInUnits("RWY 18/36", "metric"), "RWY 18/36");
});

test("artccLine: center + stratum; oceanic FIRs by name", () => {
  assert.equal(artccLine(null), null);
  assert.equal(artccLine({ ...base, artcc: { id: "ZTL", name: "ATLANTA", stratum: "HIGH" } }), "Atlanta Center (ZTL) · high");
  assert.equal(artccLine({ ...base, artcc: { id: "ZWY", name: "NEW YORK OCEANIC FIR", stratum: "OCEANIC" } }), "New York Oceanic Fir (ZWY)");
});

test("atcRows: geometry-singled-out entries first, capped, with notes", () => {
  const atc: AtcContextData = {
    ...base, phase: "SURFACE", airport: "KCLT",
    entries: [
      e("Charlotte Ground", 121.8, false, "WEST"), e("Charlotte Ground", 121.9, false, "EAST"),
      e("Charlotte Clearance", 127.15, false), e("Charlotte ATIS", 121.15, false, "arrivals"),
      e("Charlotte Tower", 118.1, true, "RWY 18/36"),
    ],
  };
  const rows = atcRows(atc, "imperial");
  assert.equal(rows.length, ATC_MAX_ROWS);
  assert.equal(rows[0].text, "Charlotte Tower 118.1 · RWY 18/36");
  assert.equal(rows[0].likely, true);
  assert.equal(rows[1].text, "Charlotte Ground 121.8 · WEST");
  assert.deepEqual(atcRows(base), []);
});

test("wiring: the runway hook renders the ATC rows from the same response; datamap shows them on the card", () => {
  const hook = fs.readFileSync(path.join(here, "RunwayStatus.tsx"), "utf8");
  assert.match(hook, /AtcFrequencyRows atc=\{data\.atc\}/);
  const page = fs.readFileSync(path.join(here, "..", "pages", "datamap.tsx"), "utf8");
  assert.match(page, /detail\.kind === "aircraft" && runwayStatus\.atcRow/);
});
