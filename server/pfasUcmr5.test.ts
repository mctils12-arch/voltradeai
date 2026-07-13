// pfasUcmr5 battery — static UCMR5 PFAS artifact serving + coordinate gate.
import { test } from "node:test";
import assert from "node:assert/strict";
import { pfasSystems, pfasArtifactMeta } from "./pfasUcmr5";

test("pfasSystems: returns non-empty coordinate-valid records from the committed artifact", () => {
  const systems = pfasSystems();
  assert.ok(systems.length > 0);
  for (const s of systems) {
    assert.ok(Number.isFinite(s.lat) && s.lat >= -90 && s.lat <= 90);
    assert.ok(Number.isFinite(s.lon) && s.lon >= -180 && s.lon <= 180);
    assert.ok(s.n_detections === s.detections.length && s.detections.length > 0);
    assert.ok(s.pws_id && s.name);
  }
});

test("pfasSystems: every detection carries a real result at/above its MRL", () => {
  for (const s of pfasSystems().slice(0, 100)) {
    for (const d of s.detections) {
      assert.ok(d.result >= d.mrl, `${s.pws_id} ${d.contaminant} result ${d.result} < mrl ${d.mrl}`);
      assert.equal(d.units, "µg/L");
    }
  }
});

test("pfasSystems: lithium (non-PFAS UCMR5 analyte) never appears", () => {
  const contaminants = new Set(pfasSystems().flatMap((s) => s.detections.map((d) => d.contaminant)));
  assert.equal(contaminants.has("lithium"), false);
});

test("pfasArtifactMeta: honesty fields present — RAW caveat, counts reconcile", () => {
  const meta = pfasArtifactMeta();
  assert.match(meta.caveat, /not a regulatory violation/);
  assert.equal(
    meta.counts.pws_geocoded + meta.counts.pws_missing_geocode,
    meta.counts.pws_with_detection,
  );
  assert.ok(meta.counts.pws_monitored_for_pfas >= meta.counts.pws_with_detection);
  assert.equal(pfasSystems().length, meta.counts.pws_geocoded);
});
