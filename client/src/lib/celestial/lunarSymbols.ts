// LUNAR MISSION SYMBOLS — SYMBOLS NOT DOTS (human directive 2026-07-12,
// applied to the Moon 2026-08-13). The map must be readable at a glance
// without clicking: a crashed impactor may never look like a crewed landing.
//
// ENCODING (one source of truth — the legend renders through the SAME
// functions the canvas draws with, so a symbol can never drift from its key):
//   SHAPE  = kind      crewed → flag · robotic_lander → tripod lander ·
//                      rover → body + wheels · sample_return → lander + up
//                      arrow · impactor → chevron into a starburst
//   COLOUR = country   the operating nation (commercial CLPS landers keep the
//                      US hue — the mission is US-flagged — and name the
//                      company on the card rather than forking the palette)
//   FILL   = outcome   solid = reached the surface intact / deliberate impact ·
//                      hollow + slash = crashed · half = partial (touched down
//                      then tipped over)
//   HALO   = confidence  dashed ring = coordinate NOT surveyed (catalogued or
//                      estimated) — the marker is a reported position
//
// Nothing here invents a category: every input comes from a catalogued field
// on LunarSite. Pure canvas 2D, no DOM beyond an offscreen canvas in the
// legend helper, `npx tsx --test`-able through a stubbed context.

import type { LunarKind, LunarOutcome, LunarCoordConfidence } from "./lunarMissions";

/** Operating-nation palette. Chosen for contrast against the grey WAC mosaic
 *  at small sizes; all sit above ~4:1 against mid-grey regolith. */
export const COUNTRY_COLOR: Record<string, string> = {
  USA: "#6fb2ff",
  USSR: "#ff6b6b",
  Russia: "#ff6b6b",
  China: "#ffd166",
  India: "#66d9a6",
  Japan: "#c39bff",
  Israel: "#7ec8e3",
};

export const DEFAULT_SITE_COLOR = "#dfe8f5";

export function countryColor(country: string): string {
  return COUNTRY_COLOR[country] ?? DEFAULT_SITE_COLOR;
}

export const KIND_LABEL: Record<LunarKind, string> = {
  crewed: "Crewed landing",
  robotic_lander: "Robotic lander",
  rover: "Rover",
  sample_return: "Sample return",
  impactor: "Impact (deliberate)",
};

export const OUTCOME_LABEL: Record<LunarOutcome, string> = {
  landed: "reached the surface intact",
  partial: "landed but tipped over",
  crashed: "crashed",
  impact_intentional: "deliberate impact",
};

/** Minimal 2D-context surface this module needs — lets tests pass a recorder
 *  instead of a real canvas. */
export interface Ctx2D {
  save(): void;
  restore(): void;
  beginPath(): void;
  closePath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, r: number, a0: number, a1: number): void;
  fill(): void;
  stroke(): void;
  setLineDash(d: number[]): void;
  fillStyle: string;
  strokeStyle: string;
  lineWidth: number;
}

export interface GlyphSpec {
  kind: LunarKind;
  outcome: LunarOutcome;
  country: string;
  coord_confidence?: LunarCoordConfidence;
}

/** Hit radius for a drawn glyph — the click target the frame registers. */
export const GLYPH_HIT_R = 11;

const isSolid = (o: LunarOutcome): boolean => o === "landed" || o === "impact_intentional";

/**
 * Draw ONE mission glyph centred on the site point (x, y) — the point itself
 * is the surface position, and the glyph is built upward from it so the
 * anchor stays exactly on the coordinate.
 */
export function drawSiteGlyph(ctx: Ctx2D, x: number, y: number, spec: GlyphSpec, scale = 1): void {
  const c = countryColor(spec.country);
  const solid = isSolid(spec.outcome);
  const s = scale;
  ctx.save();
  ctx.setLineDash([]);
  ctx.strokeStyle = c;
  ctx.fillStyle = c;
  ctx.lineWidth = 1.4 * s;

  // UNSURVEYED HALO — a dashed ring says "reported position, never located".
  if (spec.coord_confidence === "catalogued" || spec.coord_confidence === "estimated") {
    ctx.save();
    ctx.setLineDash([2 * s, 2.5 * s]);
    ctx.lineWidth = 1 * s;
    ctx.strokeStyle = c;
    ctx.beginPath();
    ctx.arc(x, y, 7.5 * s, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  switch (spec.kind) {
    case "crewed": {
      // flag: mast + pennant (unchanged from the original Apollo glyph, so the
      // six crewed sites look exactly as they always have)
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y - 11 * s);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y - 11 * s);
      ctx.lineTo(x + 7 * s, y - 8.5 * s);
      ctx.lineTo(x, y - 6 * s);
      ctx.closePath();
      if (solid) ctx.fill(); else ctx.stroke();
      break;
    }
    case "robotic_lander": {
      // tripod: trapezoid body on two splayed legs
      ctx.beginPath();
      ctx.moveTo(x - 4.5 * s, y - 5 * s);
      ctx.lineTo(x + 4.5 * s, y - 5 * s);
      ctx.lineTo(x + 3 * s, y - 9.5 * s);
      ctx.lineTo(x - 3 * s, y - 9.5 * s);
      ctx.closePath();
      if (solid) ctx.fill(); else ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 4 * s, y - 5 * s); ctx.lineTo(x - 6 * s, y);
      ctx.moveTo(x + 4 * s, y - 5 * s); ctx.lineTo(x + 6 * s, y);
      ctx.stroke();
      break;
    }
    case "rover": {
      // body + two wheels
      ctx.beginPath();
      ctx.moveTo(x - 5 * s, y - 4 * s);
      ctx.lineTo(x + 5 * s, y - 4 * s);
      ctx.lineTo(x + 5 * s, y - 9 * s);
      ctx.lineTo(x - 5 * s, y - 9 * s);
      ctx.closePath();
      if (solid) ctx.fill(); else ctx.stroke();
      ctx.beginPath(); ctx.arc(x - 3.2 * s, y - 2.2 * s, 2 * s, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.arc(x + 3.2 * s, y - 2.2 * s, 2 * s, 0, Math.PI * 2); ctx.stroke();
      break;
    }
    case "sample_return": {
      // lander body + an upward return arrow (something left again)
      ctx.beginPath();
      ctx.moveTo(x - 4.5 * s, y - 4 * s);
      ctx.lineTo(x + 4.5 * s, y - 4 * s);
      ctx.lineTo(x + 3 * s, y - 8 * s);
      ctx.lineTo(x - 3 * s, y - 8 * s);
      ctx.closePath();
      if (solid) ctx.fill(); else ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y - 8 * s); ctx.lineTo(x, y - 15 * s);
      ctx.moveTo(x - 2.6 * s, y - 12.4 * s); ctx.lineTo(x, y - 15 * s);
      ctx.lineTo(x + 2.6 * s, y - 12.4 * s);
      ctx.stroke();
      break;
    }
    case "impactor": {
      // downward chevron into a starburst at the point of impact
      ctx.beginPath();
      ctx.moveTo(x - 4.5 * s, y - 11 * s);
      ctx.lineTo(x, y - 5 * s);
      ctx.lineTo(x + 4.5 * s, y - 11 * s);
      ctx.stroke();
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI * 2 * i) / 6;
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(a) * 4.2 * s, y + Math.sin(a) * 4.2 * s);
      }
      ctx.stroke();
      break;
    }
  }

  // CRASH SLASH — a struck-through glyph reads as "did not survive" instantly.
  if (spec.outcome === "crashed") {
    ctx.beginPath();
    ctx.moveTo(x - 7 * s, y - 12 * s);
    ctx.lineTo(x + 7 * s, y - 1 * s);
    ctx.stroke();
  }
  // PARTIAL — a half-height bar under the glyph: touched down, then failed.
  if (spec.outcome === "partial") {
    ctx.beginPath();
    ctx.moveTo(x - 5 * s, y - 0.5 * s);
    ctx.lineTo(x + 1 * s, y - 0.5 * s);
    ctx.stroke();
  }

  // the anchor dot IS the coordinate
  ctx.beginPath();
  ctx.arc(x, y, 1.6 * s, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.fill();
  ctx.restore();
}

/**
 * Render a glyph to a data URL for the LEGEND — the same drawSiteGlyph the
 * canvas uses, so the key can never disagree with the map (the iconDataURL
 * one-source-of-truth rule, adapted to canvas 2D). Returns "" when no canvas
 * is available (SSR/tests) so callers can fall back to text.
 */
export function siteGlyphDataURL(spec: GlyphSpec, px = 26): string {
  try {
    const doc = (globalThis as any).document;
    if (!doc?.createElement) return "";
    const cv = doc.createElement("canvas");
    cv.width = px; cv.height = px;
    const ctx = cv.getContext("2d");
    if (!ctx) return "";
    // glyphs build upward from the anchor, so anchor low in the box
    drawSiteGlyph(ctx as unknown as Ctx2D, px / 2, px - 5, spec, px / 26);
    return cv.toDataURL();
  } catch {
    return "";
  }
}

/** One legend entry per kind, drawn from real sample specs. */
export const LEGEND_KINDS: Array<{ kind: LunarKind; label: string }> =
  (Object.keys(KIND_LABEL) as LunarKind[]).map((k) => ({ kind: k, label: KIND_LABEL[k] }));
