import { test } from "node:test";
import assert from "node:assert/strict";
import { metersPerPixel, scaleReading, zoomLabel } from "./mapScale";

test("metersPerPixel: equator z0 is the web-mercator constant; halves per zoom", () => {
  const z0 = metersPerPixel(0, 0);
  assert.ok(Math.abs(z0 - 156543.034) < 0.1, `${z0}`);
  assert.ok(Math.abs(metersPerPixel(1, 0) - z0 / 2) < 0.1);
  assert.ok(Math.abs(metersPerPixel(10, 0) - z0 / 1024) < 1e-6);
});

test("metersPerPixel: latitude compresses by cos(lat)", () => {
  const eq = metersPerPixel(5, 0);
  const at60 = metersPerPixel(5, 60);
  assert.ok(Math.abs(at60 - eq * Math.cos((60 * Math.PI) / 180)) < 1e-6);
  assert.ok(at60 < eq);
});

test("scaleReading: label is a nice round number, bar width is exact", () => {
  const r = scaleReading(9, 35, "imperial");
  assert.match(r.label, /^\d[\d,]* (ft|mi)$/);
  // parse the nice number back and confirm widthPx maps to it exactly
  const mPerPx = metersPerPixel(9, 35);
  const [numStr, unit] = r.label.split(" ");
  const num = Number(numStr.replace(/,/g, ""));
  const meters = unit === "mi" ? num * 1609.344 : num / 3.28084;
  assert.ok(Math.abs(r.widthPx - meters / mPerPx) < 0.01, "bar width matches the nice distance");
  assert.ok(r.widthPx > 20 && r.widthPx < 200, `bar targets ~90px, got ${r.widthPx}`);
});

test("scaleReading: metric switches m → km at the kilometre", () => {
  const far = scaleReading(4, 0, "metric");
  assert.match(far.label, /km$/);
  const near = scaleReading(16, 0, "metric");
  assert.match(near.label, /m$/);
});

test("scaleReading: imperial switches ft → mi at the mile", () => {
  const far = scaleReading(4, 0, "imperial");
  assert.match(far.label, /mi$/);
  const near = scaleReading(17, 0, "imperial");
  assert.match(near.label, /ft$/);
});

test("scaleReading: nice numbers are 1/2/3/5 × 10ⁿ", () => {
  for (let z = 2; z <= 18; z++) {
    for (const sys of ["metric", "imperial"] as const) {
      const num = Number(scaleReading(z, 30, sys).label.split(" ")[0].replace(/,/g, ""));
      const pow = Math.pow(10, Math.floor(Math.log10(num)));
      const f = Math.round(num / pow);
      assert.ok([1, 2, 3, 5].includes(f), `z${z} ${sys}: ${num} → lead ${f}`);
    }
  }
});

test("zoomLabel", () => {
  assert.equal(zoomLabel(9.42), "z9.4");
  assert.equal(zoomLabel(3), "z3.0");
  assert.equal(zoomLabel(NaN), "");
});

test("degenerate camera never throws", () => {
  assert.deepEqual(scaleReading(NaN, 0, "metric").label, "—");
});
