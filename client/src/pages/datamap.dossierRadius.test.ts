// LOCATION DOSSIER radius client toggle (research/location_context_engine.md's
// "radius_km client toggle still not built" gap, filed across the
// 2026-07-12/13/15 session entries, closed this session). Source-inspection
// style, matching datamap.symbols.test.ts's established pattern — a real
// headless map render isn't available in this test environment, so this
// pins the wiring contract: the client presets stay within the server's own
// clamp, fetchDossier actually sends radius_km, and the radius button row is
// wired to both the anchor-store and the re-fetch, not just cosmetic.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const datamapSrc = fs.readFileSync(path.join(here, "datamap.tsx"), "utf8");
const dossierSrc = fs.readFileSync(path.join(here, "..", "..", "..", "server", "dossier.ts"), "utf8");

test("HAZARD_RADIUS_PRESETS_KM presets stay within server/dossier.ts's own HAZARD_RADIUS_KM_MAX clamp", () => {
  const maxMatch = dossierSrc.match(/HAZARD_RADIUS_KM_MAX\s*=\s*(\d+)/);
  const defaultMatch = dossierSrc.match(/HAZARD_RADIUS_KM_DEFAULT\s*=\s*(\d+)/);
  assert.ok(maxMatch, "server/dossier.ts HAZARD_RADIUS_KM_MAX not found");
  assert.ok(defaultMatch, "server/dossier.ts HAZARD_RADIUS_KM_DEFAULT not found");
  const serverMax = Number(maxMatch![1]);
  const serverDefault = Number(defaultMatch![1]);

  const presetsMatch = datamapSrc.match(/const HAZARD_RADIUS_PRESETS_KM = \[([^\]]+)\]/);
  assert.ok(presetsMatch, "HAZARD_RADIUS_PRESETS_KM not found in datamap.tsx");
  const presets = presetsMatch![1].split(",").map((s) => Number(s.trim()));
  assert.ok(presets.length >= 3, "expected at least 3 radius presets");
  for (const p of presets) {
    assert.ok(p >= 1 && p <= serverMax, `preset ${p} falls outside server's [1, ${serverMax}] clamp`);
  }
  assert.ok(presets.includes(serverDefault), `presets should include the server default (${serverDefault})`);

  const clientDefaultMatch = datamapSrc.match(/const HAZARD_RADIUS_KM_DEFAULT = (\d+)/);
  assert.ok(clientDefaultMatch, "client HAZARD_RADIUS_KM_DEFAULT not found");
  assert.equal(Number(clientDefaultMatch![1]), serverDefault, "client default radius must match the server default");
});

test("RATCHET: fetchDossier sends radius_km on every call, honoring an explicit override over the current state", () => {
  const fnStart = datamapSrc.indexOf("const fetchDossier = async");
  assert.ok(fnStart >= 0, "fetchDossier not found");
  const block = datamapSrc.slice(fnStart, fnStart + 900);
  assert.match(block, /radiusKm\?\s*:\s*number/, "fetchDossier should accept an optional radiusKm override");
  assert.match(block, /radiusKm\s*\?\?\s*dossierRadiusKm/, "fetchDossier should fall back to the dossierRadiusKm state when no override is given");
  assert.match(block, /radius_km:\s*String\(r_km\)/, "fetchDossier must send radius_km in the querystring");
});

test("RATCHET: fetchDossier stashes a dossierAnchor so the radius toggle can re-fetch the same anchor", () => {
  const fnStart = datamapSrc.indexOf("const fetchDossier = async");
  const block = datamapSrc.slice(fnStart, fnStart + 900);
  assert.match(block, /dossierAnchor:\s*\{\s*entityId,\s*lat,\s*lon\s*\}/, "fetchDossier must stash {entityId, lat, lon} as dossierAnchor");
});

test("RATCHET: the hazard section renders a radius button row wired to setDossierRadiusKm + a re-fetch using the stashed anchor", () => {
  const idx = datamapSrc.indexOf("vt-radius-row");
  assert.ok(idx >= 0, "vt-radius-row control not found");
  const block = datamapSrc.slice(idx, idx + 900);
  assert.match(block, /HAZARD_RADIUS_PRESETS_KM\.map/, "radius row must be built from HAZARD_RADIUS_PRESETS_KM");
  assert.match(block, /setDossierRadiusKm\(km\)/, "radius button click must update dossierRadiusKm");
  assert.match(block, /fetchDossier\(detail\.dossierKey!, a\.entityId, a\.lat, a\.lon, km\)/, "radius button click must re-fetch the dossier with the stashed anchor at the new radius");
  assert.match(block, /dossierRadiusKm === km/, "the active preset should reflect the current dossierRadiusKm state");
});

test("RATCHET: the radius row (and the empty-radius note) gate on hazardsReady, not just non-empty hazardCats, so widening the radius stays discoverable on a zero-hit result", () => {
  const idx = datamapSrc.indexOf("hazardsReady && (");
  assert.ok(idx >= 0, "hazard section should be gated on hazardsReady (not hazardCats.length > 0), so an empty result still shows the radius control");
  const block = datamapSrc.slice(idx, idx + 1600);
  assert.match(block, /None on file within this radius/, "an empty-at-this-radius result should tell the user to try a larger radius");
});
