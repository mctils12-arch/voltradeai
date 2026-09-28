import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fmtHorizNm, fmtVertFt, fmtSeparation, liveSeparation, withinMinima, sideLabel, approachKey,
  confidenceText, fmtUtcTime,
} from "./closeApproachView.ts";

test("separation renders nm (the standard's unit) plus the user's system via units.ts", () => {
  assert.equal(fmtHorizNm(2.1, "imperial"), "2.1 nm (2.4 mi)");
  assert.equal(fmtHorizNm(2.1, "metric"), "2.1 nm (3.9 km)");
  assert.equal(fmtVertFt(400, "imperial"), "400 ft");
  assert.equal(fmtVertFt(400, "metric"), "122 m");
  assert.equal(fmtVertFt(null, "metric"), "altitude unknown");
  assert.equal(fmtSeparation(0.5, 800, "imperial"), "0.5 nm (0.6 mi) · 800 ft");
});

test("live separation: haversine nm + absolute vertical ft; unknown altitude is never 'within'", () => {
  const s = liveSeparation({ lat: 40, lon: -100, altM: 10000 }, { lat: 40 + 3 / 60, lon: -100, altM: 10000 + 800 * 0.3048 });
  assert.ok(Math.abs(s.horizNm - 3) < 0.01);
  assert.ok(Math.abs((s.vertFt as number) - 800) < 0.01);
  assert.equal(withinMinima(s), true);
  assert.equal(withinMinima({ horizNm: 6, vertFt: 0 }), false);
  assert.equal(withinMinima({ horizNm: 1, vertFt: 1000 }), false);
  const unknown = liveSeparation({ lat: 40, lon: -100, altM: NaN }, { lat: 40, lon: -100.01, altM: 9000 });
  assert.equal(unknown.vertFt, null);
  assert.equal(withinMinima(unknown), false);
  // antimeridian
  const seam = liveSeparation({ lat: 0, lon: 179.99, altM: 0 }, { lat: 0, lon: -179.99, altM: 0 });
  assert.ok(seam.horizNm < 1.3);
});

test("labels: callsign when known, hex otherwise; keys are stable; honest confidence text", () => {
  assert.equal(sideLabel("a1b2c3", " UAL1 "), "UAL1");
  assert.equal(sideLabel("a1b2c3"), "A1B2C3");
  assert.equal(approachKey({ a: "a", b: "b", t: 5 }), "a|b|5");
  assert.match(confidenceText("high"), /20 s/);
  assert.match(confidenceText("low"), /90 s/);
  assert.equal(fmtUtcTime(Date.parse("2026-09-28T14:02:31Z")), "14:02:31 UTC");
});
