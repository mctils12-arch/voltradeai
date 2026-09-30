import { test } from "node:test";
import assert from "node:assert/strict";
import zlib from "node:zlib";
import {
  clusterSymbols, embeddedGeoModel, fitSimilarity, fixLabels, fromLocalNm, georeferencePlate, interpretContent,
  lccFromWkt, parsePdfObjects, firstPage, robustFit, simApply, simRotationDeg, simScale, toLocalNm, tokenize,
  GEOREF_MAX_RMS_NM, type ControlCandidate, type LL,
} from "./plateGeoref";

// ── a tiny synthetic "plate": a real PDF (Flate content stream, Type1 font)
//    whose plan view is drawn at a KNOWN scale, so the fit has ground truth.
const CENTER: LL = { lat: 30.2, lon: -97.67 };
const NM_PER_PT = 6.9 / 72;          // a typical IAP plan-view scale
const ORIGIN = [200, 330];            // page point of CENTER
const pageToLL = (x: number, y: number) => fromLocalNm((x - ORIGIN[0]) * NM_PER_PT, (y - ORIGIN[1]) * NM_PER_PT, CENTER);

function makePdf(content: string, pageExtra = ""): Buffer {
  const stream = zlib.deflateSync(Buffer.from(content, "latin1"));
  const parts: Buffer[] = [];
  const push = (s: string | Buffer) => parts.push(typeof s === "string" ? Buffer.from(s, "latin1") : s);
  push("%PDF-1.4\n");
  push("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
  push("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
  push(`3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 600] /Contents 4 0 R /Resources << /Font << /T1_0 5 0 R >> >>${pageExtra} >>\nendobj\n`);
  push(`4 0 obj\n<< /Length ${stream.length} /Filter /FlateDecode >>\nstream\n`);
  push(stream);
  push("\nendstream\nendobj\n");
  push("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /WinAnsiEncoding >>\nendobj\n");
  push("trailer\n<< /Root 1 0 R >>\n%%EOF\n");
  return Buffer.concat(parts);
}

/** waypoint star drawn the FAA way: two filled halves offset about the fix */
const star = (x: number, y: number) =>
  `q 1 0 0 1 0 0 cm ${x + 2.7 - 5} ${y - 2.7 - 5} 10 10 re f ${x - 2.7 - 5} ${y + 2.7 - 5} 10 10 re f Q\n`;
const label = (name: string, x: number, y: number) => `BT /T1_0 6 Tf 1 0 0 1 ${x} ${y} Tm (${name}) Tj ET\n`;

const GOOD: Array<[string, number, number]> = [["AAAAA", 120, 400], ["BBBBB", 260, 385], ["CCCCC", 200, 300], ["DDDDD", 150, 255], ["EEEEE", 320, 250]];

function plate(opts: { fixes?: Array<[string, number, number]>; pageExtra?: string } = {}): Buffer {
  const fx = opts.fixes ?? GOOD;
  let c = "0 G 1 w 20 200 360 250 re S\n";              // plan-view border
  c += "20 60 360 120 re S\n";                          // profile box below it
  for (const [n, x, y] of fx) c += star(x, y) + label(n, x + 12, y - 2);
  c += label("AAAAA", 60, 100) + star(90, 90);          // decoy: profile repeats a name
  c += "BT /T1_0 5 Tf 1 0 0 1 30 560 Tm [(ILS OR LOC R) -30 (WY 18L)] TJ ET\n";
  return makePdf(c, opts.pageExtra ?? "");
}
const fixMap = (fx: Array<[string, number, number]>) => new Map(fx.map(([n, x, y]) => [n, pageToLL(x, y)]));

// ── tokenizer / interpreter ─────────────────────────────────────────────────
test("tokenize: literal strings with escapes + nesting, hex strings, arrays, inline images skipped", () => {
  const t = tokenize("(a\\(b\\) (c)) <48 49> [(X) -250 (Y)] BI /W 1 ID \x00\x01EI\nQ 1.5 -2 Tm");
  assert.deepEqual(t[0], { k: "str", v: "a(b) (c)" });
  assert.deepEqual(t[1], { k: "str", v: "HI" });
  assert.equal(t[2].k, "arr");
  const ops = t.filter((x) => x.k === "op").map((x) => (x as { v: string }).v);
  assert.deepEqual(ops, ["BI", "Q", "Tm"]);
});

test("interpretContent: text positions through cm + Tm, TJ kerning gaps become spaces, octal ° decoded", () => {
  const c = interpretContent([Buffer.from("q 1 0 0 1 10 20 cm BT /T1_0 5 Tf 1 0 0 1 30 40 Tm (175\\260) Tj ET Q BT 1 0 0 1 5 5 Tm [(RRTOO) -300 (INT)] TJ ET", "latin1")]);
  assert.equal(c.texts[0].text, "175°");
  assert.equal(c.texts[0].x, 40);
  assert.equal(c.texts[0].y, 60);
  assert.equal(c.texts[1].text, "RRTOO INT");
});

test("PDF reader: page, media box, content stream; symbol clustering joins the two star halves", () => {
  const page = firstPage(parsePdfObjects(plate()))!;
  assert.equal(page.width, 400);
  assert.equal(page.height, 600);
  const content = interpretContent(page.contents);
  assert.ok(content.texts.some((t) => t.text === "ILS OR LOC RWY 18L"));
  const syms = clusterSymbols(content.marks);
  for (const [, x, y] of GOOD) {
    const s = syms.find((q) => Math.hypot(q.x - x, q.y - y) < 0.01);
    assert.ok(s, `star at ${x},${y} found as ONE symbol centred on the fix`);
    assert.equal(s!.parts, 2);
  }
  const labels = fixLabels(content.texts, fixMap(GOOD));
  assert.equal(labels.filter((l) => l.fix === "AAAAA").length, 2, "plan-view label + profile decoy");
});

// ── the fit ─────────────────────────────────────────────────────────────────
test("fitSimilarity recovers scale, rotation, translation exactly", () => {
  const th = 12 * Math.PI / 180, s = 0.1;
  const T0 = { a: s * Math.cos(th), b: s * Math.sin(th), tx: 3, ty: -2 };
  const src: Array<[number, number]> = [[0, 0], [100, 0], [30, 80], [-40, 20]];
  const dst = src.map(([x, y]) => simApply(T0, x, y));
  const T = fitSimilarity(src, dst)!;
  assert.ok(Math.abs(simScale(T) - s) < 1e-12);
  assert.ok(Math.abs(simRotationDeg(T) - 12) < 1e-9);
  assert.ok(Math.abs(T.tx - 3) < 1e-9 && Math.abs(T.ty + 2) < 1e-9);
});

test("robustFit: noisy inliers fit; a gross outlier and a duplicate-name decoy are rejected", () => {
  const truth = (x: number, y: number): [number, number] => [(x - 200) * NM_PER_PT, (y - 330) * NM_PER_PT];
  const cands: ControlCandidate[] = GOOD.map(([n, x, y], i) => ({ fix: n, page: [x + (i % 2 ? 0.8 : -0.8), y] as [number, number], ground: truth(x, y) }));
  cands.push({ fix: "EEEEE", page: [90, 90], ground: truth(320, 250) });   // decoy symbol for a real fix
  cands.push({ fix: "FFFFF", page: [300, 420], ground: truth(260, 420) }); // wrong place entirely (~3.8 nm)
  const fit = robustFit(cands)!;
  assert.deepEqual(fit.inliers.map((c) => c.fix).sort(), ["AAAAA", "BBBBB", "CCCCC", "DDDDD", "EEEEE"]);
  assert.ok(fit.inliers.every((c) => !(c.page[0] === 90 && c.page[1] === 90)));
  assert.ok(fit.rmsNm < 0.1, `rms ${fit.rmsNm}`);
  assert.ok(Math.abs(simScale(fit.T) - NM_PER_PT) / NM_PER_PT < 0.01);
  assert.equal(fit.rejected, 2);
});

test("georeferencePlate: synthetic to-scale plate georeferences blind (RANSAC) with an outlier rejected", () => {
  // EEEEE's CIFP position is 40 pt (~3.8 nm) away from where the chart draws it
  const fixes = fixMap(GOOD);
  fixes.set("EEEEE", pageToLL(320 + 40, 250));
  const g = georeferencePlate(plate(), fixes, CENTER);
  assert.equal(g.method, "ransac");
  assert.equal(g.georeferenced, true, g.reason);
  assert.deepEqual(g.controlPoints.map((c) => c.fix).sort(), ["AAAAA", "BBBBB", "CCCCC", "DDDDD"]);
  assert.ok(g.rmsNm! < 0.05, `rms ${g.rmsNm}`);
  assert.ok(g.rmsNm! < GEOREF_MAX_RMS_NM);
  assert.ok(Math.abs(g.scaleNmPerInch! - 6.9) < 0.05);
  assert.ok(Math.abs(g.rotationDeg!) < 0.2);
  assert.deepEqual(g.planView, { x0: 20, y0: 200, x1: 380, y1: 450 });
  // top-left crop corner lands where the known transform puts page (20, 450)
  const tl = pageToLL(20, 450);
  const [lon, lat] = g.corners![0];
  assert.ok(Math.hypot(...toLocalNm({ lat, lon }, tl)) < 0.1);
  assert.equal(g.embeddedGeoPdf, false);
  assert.equal(g.embeddedAgreementNm, null);
});

test("georeferencePlate: blind RANSAC needs 4 fixes — 3 are not enough without a GeoPDF seed", () => {
  const three = GOOD.slice(0, 3);
  const g = georeferencePlate(plate({ fixes: three }), fixMap(three), CENTER);
  assert.equal(g.georeferenced, false);
  assert.match(g.reason, /need 4/);
  assert.equal(g.corners, null);
});

test("georeferencePlate: a NOT-TO-SCALE chart (positions inconsistent with the ground) is refused", () => {
  const scrambled = new Map(GOOD.map(([n], i) => [n, pageToLL(60 + ((i * 137) % 300), 210 + ((i * 89) % 230))]));
  const g = georeferencePlate(plate(), scrambled, CENTER);
  assert.equal(g.georeferenced, false);
  assert.equal(g.corners, null);
});

function vp(shiftNm = 0): string {
  const b = [20, 200, 380, 450];
  const corners: Array<[number, number]> = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
  const g = corners.map(([x, y]) => pageToLL(x + shiftNm / NM_PER_PT, y)).flatMap((p) => [p.lat, p.lon]);
  return ` /VP [ << /Type /Viewport /BBox [${b.join(" ")}] /Measure << /Type /Measure /Subtype /GEO /GPTS [${g.join(" ")}] /LPTS [0 0 1 0 1 1 0 1] >> >> ]`;
}

test("georeferencePlate: embedded GeoPDF seeds the association (3 fixes suffice) and agrees", () => {
  const three = GOOD.slice(0, 3);
  const g = georeferencePlate(plate({ fixes: three, pageExtra: vp() }), fixMap(three), CENTER);
  assert.equal(g.method, "geopdf-seeded");
  assert.equal(g.embeddedGeoPdf, true);
  assert.equal(g.georeferenced, true, g.reason);
  assert.ok(g.embeddedAgreementNm! < 0.05);
  assert.match(g.reason, /embedded GeoPDF agrees/);
});

test("georeferencePlate: a GeoPDF that disagrees with the chart's own symbols is not trusted", () => {
  const g = georeferencePlate(plate({ pageExtra: vp(4) }), fixMap(GOOD), CENTER);
  assert.equal(g.georeferenced, false);
});

test("lccFromWkt + embeddedGeoModel on the REAL KAUS ILS 18L viewport (d-TPP 2609)", () => {
  const wkt = 'PROJCS["FAA LCC (0055613925)",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.000,298.25722210]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",0.000],PARAMETER["False_Northing",0.000],PARAMETER["Central_Meridian",-97.66047222222220],PARAMETER["Latitude_Of_Origin",30.30236111111110],PARAMETER["Standard_Parallel_1",30.66666666666670],PARAMETER["Standard_Parallel_2",25.33333333333330],UNIT["Inch",0.02540005080010]]';
  assert.ok(lccFromWkt(wkt));
  assert.equal(lccFromWkt("GEOGCS[...]"), null);
  const m = embeddedGeoModel({
    bbox: [9.18, 2.628, 378.18, 591.372],
    lpts: [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
    gpts: [29.89082498591, -97.93014344643, 29.89082494368, -97.3907790727, 30.64046326543, -97.38886317304, 30.64046330794, -97.93205919033],
    wkt,
  })!;
  assert.equal(m.kind, "lcc");
  assert.ok(m.cornerResidualPt < 0.01, `LCC corners fit an affine to ${m.cornerResidualPt} pt`);
  // HOUKM (CIFP 30°25'44.35"N 97°45'08.98"W) lands on its waypoint star (found at ~143.1,398.4 in the PDF)
  const [x, y] = m.toPage({ lat: 30.42898611, lon: -97.75249444 });
  assert.ok(Math.hypot(x - 143.1, y - 398.4) < 2, `${x},${y}`);
});
