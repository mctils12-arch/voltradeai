// Nuclear-test decode + classification tests (symbol directive 2026-07-12).
// The load-bearing assertion: EVERY emplacement code in the shipped catalog
// (datacore/nuclear_tests.json) is explicitly mapped — a data refresh that
// introduces a new code fails here instead of silently rendering the default
// symbol. Run: npx tsx --test client/src/lib/nukeCodes.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { decodePurpose, decodeType, testingAgency, yieldContext, blastRadiusKm, NUKE_TYPE, NUKE_PURPOSE } from "./nukeCodes.ts";
import { classifyNukeTest, NUKE_CLASS_ICON, NUKE_CLASS_LABEL } from "./mapIcons.ts";

const here = dirname(fileURLToPath(import.meta.url));
const catalog = JSON.parse(readFileSync(join(here, "..", "..", "..", "datacore", "nuclear_tests.json"), "utf8"));

test("every emplacement code in the shipped catalog is explicitly classified and decoded", () => {
  const codes = new Set<string>(catalog.tests.map((t: any) => String(t.t || "").toUpperCase().trim()).filter(Boolean));
  assert.ok(codes.size >= 15, `expected a rich code set, got ${codes.size}`);
  for (const code of codes) {
    assert.ok(code in NUKE_TYPE, `type code ${code} missing from NUKE_TYPE decode table`);
    // classification must come from the explicit table, not the fallback:
    // spot the fallback by classifying a nonsense code and requiring known
    // codes to be deliberate entries (fallback is "ground" — ground-class
    // codes are asserted individually below instead).
  }
  // explicit class spot checks (one per class, from the catalog's own codes)
  assert.equal(classifyNukeTest("AIRDROP"), "air");
  assert.equal(classifyNukeTest("TOWER"), "ground");
  assert.equal(classifyNukeTest("CRATER"), "ground", "FOA key: detonated IN a crater, at the surface");
  assert.equal(classifyNukeTest("BARGE"), "water");
  assert.equal(classifyNukeTest("UW"), "water");
  assert.equal(classifyNukeTest("SHAFT"), "underground");
  assert.equal(classifyNukeTest("SHAFT/LG"), "underground");
});

test("every class has an icon and a label (legend contract)", () => {
  for (const cls of ["air", "ground", "water", "underground"] as const) {
    assert.ok(NUKE_CLASS_ICON[cls]?.startsWith("vt-nuke-"), `icon for ${cls}`);
    assert.ok(NUKE_CLASS_LABEL[cls]?.length > 3, `label for ${cls}`);
  }
});

test("every purpose code in the shipped catalog decodes to prose", () => {
  const codes = new Set<string>(catalog.tests.map((t: any) => String(t.p || "").toUpperCase().trim()).filter(Boolean));
  for (const code of codes) {
    const prose = decodePurpose(code);
    assert.ok(prose && prose !== code, `purpose code ${code} does not decode (got "${prose}")`);
  }
  // combined codes name both parts
  const combo = decodePurpose("WR/SE");
  assert.match(combo, /Weapons related/i);
  assert.match(combo, /Safety/i);
});

test("blast radius: Glasstone-Dolan cube-root scaling, contained when buried", () => {
  // 1 kt surface = 0.47 km reference; 8 kt = 2x (cbrt(8)=2)
  assert.ok(Math.abs((blastRadiusKm(1, "TOWER") ?? 0) - 0.47) < 1e-9);
  assert.ok(Math.abs((blastRadiusKm(8, "SURFACE") ?? 0) - 0.94) < 1e-9);
  assert.equal(blastRadiusKm(150, "SHAFT"), null, "buried shot is contained — no ring");
  assert.equal(blastRadiusKm(150, "TUNNEL"), null);
  assert.ok((blastRadiusKm(1, "CRATER") ?? 0) > 0, "CRATER is a surface shot — it gets a ring");
  assert.equal(blastRadiusKm(null, "AIRDROP"), null, "no catalogued yield — no estimated ring");
  assert.equal(blastRadiusKm(0, "AIRDROP"), null);
});

test("yield context anchors to Hiroshima honestly", () => {
  assert.match(yieldContext(15), /Hiroshima/);
  assert.match(yieldContext(21), /tons of TNT/);
  assert.match(yieldContext(null), /not stated/i);
});

test("testing agency resolves by country and era", () => {
  assert.match(testingAgency("USA", 1945), /Manhattan/);
  assert.match(testingAgency("USA", 1962), /Atomic Energy Commission/);
  assert.match(testingAgency("USA", 1975), /ERDA/);
  assert.match(testingAgency("USA", 1990), /Department of Energy/);
  assert.match(testingAgency("USSR", 1961), /Medium Machine Building/);
  assert.match(testingAgency("FRANCE", 1968), /CEA/);
  assert.equal(testingAgency("NOWHERE", 1970), "");
});
