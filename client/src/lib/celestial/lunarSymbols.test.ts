// SYMBOLS NOT DOTS, enforced mechanically.
//
// The human's test for the map is "readable at a glance without clicking".
// That fails silently if two mission kinds ever draw the same shape, so these
// tests record the drawing calls into a stub context and assert the glyphs
// actually DIFFER — a regression that made every site a dot again would go
// unnoticed by a snapshot or a render-without-throwing check.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  drawSiteGlyph, countryColor, COUNTRY_COLOR, KIND_LABEL, OUTCOME_LABEL,
  LEGEND_KINDS, GLYPH_HIT_R, DEFAULT_SITE_COLOR, type Ctx2D,
} from "./lunarSymbols";
import { LUNAR_SITES } from "./lunarMissions";
import type { LunarKind, LunarOutcome } from "./lunarMissions";

/** Records every path op so two glyphs can be compared structurally. */
function recorder(): Ctx2D & { ops: string[]; dashes: number[][] } {
  const ops: string[] = [];
  const dashes: number[][] = [];
  return {
    ops, dashes,
    fillStyle: "", strokeStyle: "", lineWidth: 0,
    save() { ops.push("save"); },
    restore() { ops.push("restore"); },
    beginPath() { ops.push("beginPath"); },
    closePath() { ops.push("closePath"); },
    moveTo(x, y) { ops.push(`moveTo(${x.toFixed(1)},${y.toFixed(1)})`); },
    lineTo(x, y) { ops.push(`lineTo(${x.toFixed(1)},${y.toFixed(1)})`); },
    arc(x, y, r) { ops.push(`arc(${x.toFixed(1)},${y.toFixed(1)},${r.toFixed(1)})`); },
    fill() { ops.push("fill"); },
    stroke() { ops.push("stroke"); },
    setLineDash(d) { dashes.push(d); ops.push(`dash(${d.join(",")})`); },
  };
}

const draw = (kind: LunarKind, outcome: LunarOutcome = "landed", country = "USA", conf?: any) => {
  const r = recorder();
  drawSiteGlyph(r, 100, 100, { kind, outcome, country, coord_confidence: conf });
  return r;
};

test("every mission kind draws a STRUCTURALLY DIFFERENT glyph (never the same dot)", () => {
  const kinds: LunarKind[] = ["crewed", "robotic_lander", "rover", "sample_return", "impactor"];
  const sigs = new Map<string, LunarKind>();
  for (const k of kinds) {
    const sig = draw(k).ops.join("|");
    for (const [other, ok] of sigs) {
      assert.notEqual(sig, other, `${k} draws the same shape as ${ok} — the map would be unreadable without clicking`);
    }
    sigs.set(sig, k);
  }
  assert.equal(sigs.size, kinds.length);
});

test("each glyph anchors ON the coordinate (the marker cannot drift off its site)", () => {
  for (const k of ["crewed", "robotic_lander", "rover", "sample_return", "impactor"] as LunarKind[]) {
    const ops = draw(k).ops;
    assert.ok(ops.some((o) => o.startsWith("arc(100.0,100.0,1.6")),
      `${k}: missing the anchor dot exactly at the site point`);
  }
});

test("outcome changes the treatment: crashed is struck through, partial gets its bar", () => {
  const landed = draw("robotic_lander", "landed").ops.join("|");
  const crashed = draw("robotic_lander", "crashed").ops.join("|");
  const partial = draw("robotic_lander", "partial").ops.join("|");
  assert.notEqual(landed, crashed, "a crashed lander must not look like a successful one");
  assert.notEqual(landed, partial, "a tipped-over lander must not look like a clean landing");
  assert.notEqual(crashed, partial);
  // crashed strokes the slash across the glyph
  assert.ok(crashed.includes("moveTo(93.0,88.0)"), "crashed glyph missing the strike-through");
});

test("an unsurveyed position draws a DASHED halo — a reported point never looks located", () => {
  const surveyed = draw("robotic_lander", "landed", "USSR", "surveyed_lro");
  const catalogued = draw("robotic_lander", "landed", "USSR", "catalogued");
  const estimated = draw("robotic_lander", "landed", "USSR", "estimated");
  assert.equal(surveyed.dashes.filter((d) => d.length > 0).length, 0,
    "a surveyed site must have no dashed halo");
  assert.ok(catalogued.dashes.some((d) => d.length > 0), "catalogued position must draw the dashed halo");
  assert.ok(estimated.dashes.some((d) => d.length > 0), "estimated position must draw the dashed halo");
});

test("colour encodes the operating nation, with a safe fallback", () => {
  assert.equal(countryColor("USA"), COUNTRY_COLOR.USA);
  assert.equal(countryColor("China"), COUNTRY_COLOR.China);
  assert.equal(countryColor("USSR"), countryColor("Russia"), "USSR and Russia share the lineage hue");
  assert.equal(countryColor("Atlantis"), DEFAULT_SITE_COLOR, "unknown country falls back, never throws");
  // every country actually present in the data must have a real colour
  for (const s of LUNAR_SITES) {
    assert.notEqual(countryColor(s.country), undefined);
    assert.match(countryColor(s.country), /^#[0-9a-f]{6}$/i, `${s.country}: colour must be a hex value`);
  }
});

test("every kind and outcome in the data has a human label (legend completeness)", () => {
  for (const s of LUNAR_SITES) {
    assert.ok(KIND_LABEL[s.kind], `${s.kind}: missing legend label`);
    assert.ok(OUTCOME_LABEL[s.outcome], `${s.outcome}: missing outcome label`);
  }
  // the legend must cover every kind the map can draw — a symbol with no key
  // is exactly the "readable at a glance" failure this rule exists to prevent
  const legendKinds = new Set(LEGEND_KINDS.map((l) => l.kind));
  for (const s of LUNAR_SITES) {
    assert.ok(legendKinds.has(s.kind), `${s.kind} is drawn on the map but absent from the legend`);
  }
});

test("drawing never throws on any real site in the catalogue", () => {
  for (const s of LUNAR_SITES) {
    assert.doesNotThrow(() => {
      drawSiteGlyph(recorder(), 10, 10, {
        kind: s.kind, outcome: s.outcome, country: s.country, coord_confidence: s.coord_confidence,
      });
    }, `${s.id} failed to draw`);
  }
});

test("the hit radius is big enough to click on a touch screen", () => {
  assert.ok(GLYPH_HIT_R >= 10, "a marker smaller than ~10px is unclickable on a phone");
});
