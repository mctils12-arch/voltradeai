/**
 * mapIcons.ts — runtime-generated SDF marker icons + type classification for
 * the /data map. All shapes are drawn once to small canvases and registered
 * with maplibre as SDF images, so per-feature `icon-color` and `icon-rotate`
 * work on the GPU (symbol layers = WebGL, viewport-culled — the DESIGN.md
 * performance budget's rendering path).
 *
 * Classification is honest-best-effort from free feed fields:
 *  - aircraft: ADS-B emitter category (A1..A7) + ICAO type designator
 *    prefixes (e.g. B738 -> jet, C172 -> piston, AT76 -> turboprop,
 *    EC35 -> helicopter). Unknown stays "unknown" with a generic mark.
 *  - vessels: AIS ship-type code ranges (60s passenger, 70s cargo,
 *    80s tanker, 30 fishing, 31/32/52 tug/pilot).
 */

export type AircraftClass = "jet" | "turboprop" | "piston" | "helicopter" | "unknown";
export type VesselClass = "tanker" | "cargo" | "passenger" | "fishing" | "tug" | "other";

const JET_PREFIX = /^(B7|A3|A2|A1[0-9]|E1|E2|CRJ|GLF|G[2-6]|C25|C5[0-9]|C68|FA|F2TH|F900|LJ|H25|BD|CL3|CL6|MD8|MD9|DC9|B73|B74|B75|B76|B77|B78)/;
const TURBOPROP_PREFIX = /^(AT4|AT7|DH8|PC12|PC24|TBM|B350|BE20|BE30|BE9|C208|D228|D328|E110|F50|SF34|SW[34]|C130|P180)/;
const PISTON_PREFIX = /^(C1[0-9]{2}|C2[0-9]{2}|P28|PA[0-9]|BE3[0-6]|BE5[0-9]|BE76|DA4|DA6|DV20|M20|SR2|RV[0-9]|GLST|LNC)/;
const HELI_PREFIX = /^(R22|R44|R66|EC[0-9]|H4[0-9]|H60|H64|H47|UH|AS[0-9]|AW[0-9]|B06|B407|B412|B429|S76|S92|MI[0-9]|KA[0-9])/;

export function classifyAircraft(type?: string | null, category?: string | null): AircraftClass {
  const cat = String(category || "").toUpperCase();
  if (cat === "A7") return "helicopter";
  const t = String(type || "").toUpperCase();
  if (t) {
    if (HELI_PREFIX.test(t)) return "helicopter";
    if (TURBOPROP_PREFIX.test(t)) return "turboprop";
    if (PISTON_PREFIX.test(t)) return "piston";
    if (JET_PREFIX.test(t)) return "jet";
  }
  if (cat === "A1") return "piston";       // light (<15.5k lb)
  if (cat === "A3" || cat === "A4" || cat === "A5") return "jet";
  if (cat === "A2") return "turboprop";    // small (~ regional)
  return "unknown";
}

export function classifyVessel(shiptype?: number | null): VesselClass {
  const s = shiptype ?? -1;
  if (s >= 80 && s <= 89) return "tanker";
  if (s >= 70 && s <= 79) return "cargo";
  if (s >= 60 && s <= 69) return "passenger";
  if (s === 30) return "fishing";
  if (s === 31 || s === 32 || s === 52) return "tug";
  return "other";
}

export const VESSEL_CLASS_LABEL: Record<VesselClass, string> = {
  tanker: "Tanker", cargo: "Cargo", passenger: "Passenger",
  fishing: "Fishing", tug: "Tug/Pilot", other: "Vessel",
};

export const AIRCRAFT_CLASS_LABEL: Record<AircraftClass, string> = {
  jet: "Jet", turboprop: "Turboprop", piston: "Light piston",
  helicopter: "Helicopter", unknown: "Aircraft",
};

// ── SDF shape drawing ────────────────────────────────────────────────────────
// White shapes on transparent, registered with {sdf:true} so maplibre tints
// them per-feature via icon-color. All point "up" (north); icon-rotate turns
// them to heading.
function draw(size: number, fn: (ctx: CanvasRenderingContext2D, s: number) => void): ImageData {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#fff";
  fn(ctx, size);
  return ctx.getImageData(0, 0, size, size);
}

const S = 40; // px canvas; rendered ~0.45 scale on map

const shapes: Record<string, () => ImageData> = {
  // swept-wing jet silhouette
  "vt-jet": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 3);                 // nose
    ctx.lineTo(m + 3, m - 2);
    ctx.lineTo(s - 5, m + 6);         // right wing (swept back)
    ctx.lineTo(m + 3, m + 4);
    ctx.lineTo(m + 6, s - 7);         // right tail
    ctx.lineTo(m, s - 10);
    ctx.lineTo(m - 6, s - 7);         // left tail
    ctx.lineTo(m - 3, m + 4);
    ctx.lineTo(5, m + 6);             // left wing
    ctx.lineTo(m - 3, m - 2);
    ctx.closePath(); ctx.fill();
  }),
  // straight-wing prop
  "vt-prop": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 4);
    ctx.lineTo(m + 2.5, m - 4);
    ctx.lineTo(s - 4, m - 3);         // straight right wing
    ctx.lineTo(s - 4, m + 2);
    ctx.lineTo(m + 2.5, m + 2);
    ctx.lineTo(m + 4.5, s - 8);
    ctx.lineTo(m, s - 10);
    ctx.lineTo(m - 4.5, s - 8);
    ctx.lineTo(m - 2.5, m + 2);
    ctx.lineTo(4, m + 2);
    ctx.lineTo(4, m - 3);
    ctx.lineTo(m - 2.5, m - 4);
    ctx.closePath(); ctx.fill();
  }),
  // helicopter: fuselage + rotor cross
  "vt-heli": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(6, 6); ctx.lineTo(s - 6, s - 6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(s - 6, 6); ctx.lineTo(6, s - 6); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(m, m + 2, 5, 9, 0, 0, Math.PI * 2); ctx.fill();
  }),
  // generic aircraft (unknown)
  "vt-plane": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 4);
    ctx.lineTo(m + 3, m);
    ctx.lineTo(s - 6, m + 4);
    ctx.lineTo(m + 3, m + 3);
    ctx.lineTo(m + 5, s - 8);
    ctx.lineTo(m, s - 11);
    ctx.lineTo(m - 5, s - 8);
    ctx.lineTo(m - 3, m + 3);
    ctx.lineTo(6, m + 4);
    ctx.lineTo(m - 3, m);
    ctx.closePath(); ctx.fill();
  }),
  // vessels: pointed-bow hull silhouettes at different aspect ratios
  "vt-tanker": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 4);                          // bow
    ctx.lineTo(m + 6, 12);
    ctx.lineTo(m + 6, s - 6);
    ctx.lineTo(m - 6, s - 6);
    ctx.lineTo(m - 6, 12);
    ctx.closePath(); ctx.fill();
  }),
  "vt-cargo": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 6);
    ctx.lineTo(m + 5, 14);
    ctx.lineTo(m + 5, s - 8);
    ctx.lineTo(m - 5, s - 8);
    ctx.lineTo(m - 5, 14);
    ctx.closePath(); ctx.fill();
    ctx.clearRect(m - 2.5, m - 2, 5, 5);       // hatch notch
  }),
  "vt-boat": () => draw(S, (ctx, s) => {       // small craft (fishing/tug/other)
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 10);
    ctx.lineTo(m + 5, s - 12);
    ctx.lineTo(m - 5, s - 12);
    ctx.closePath(); ctx.fill();
  }),
  // ── strategic-site category silhouettes (upright, never rotated) ──
  // port: anchor — ring, shank, stock, curved flukes
  "vt-port": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.arc(m, 9, 3.5, 0, Math.PI * 2); ctx.stroke();   // ring
    ctx.beginPath(); ctx.moveTo(m, 12.5); ctx.lineTo(m, s - 8); ctx.stroke(); // shank
    ctx.beginPath(); ctx.moveTo(m - 8, 18); ctx.lineTo(m + 8, 18); ctx.stroke(); // stock
    ctx.beginPath(); ctx.arc(m, s - 16, 9.5, Math.PI * 0.15, Math.PI * 0.85); ctx.stroke(); // flukes
  }),
  // tank farm: cylinder cluster — three tank tops seen from above
  "vt-tank": () => draw(S, (ctx, s) => {
    const m = s / 2;
    for (const [cx, cy, r] of [[m - 7, m + 6, 6.5], [m + 7, m + 6, 6.5], [m, m - 7, 6.5]] as const) {
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
      // punch a small hole so clusters read as rings at map scale
      ctx.save(); ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath(); ctx.arc(cx, cy, 2.4, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }
  }),
  // steel mill: factory — building block, sawtooth roof, chimney
  "vt-mill": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.fillRect(m - 12, m + 2, 24, 12);                    // main block
    ctx.beginPath();                                        // sawtooth roof
    ctx.moveTo(m - 12, m + 2);
    ctx.lineTo(m - 12, m - 5); ctx.lineTo(m - 4, m + 2);
    ctx.lineTo(m - 4, m - 5); ctx.lineTo(m + 4, m + 2);
    ctx.lineTo(m + 4, m - 5); ctx.lineTo(m + 12, m + 2);
    ctx.closePath(); ctx.fill();
    ctx.fillRect(m + 6, m - 14, 4.5, 12);                   // chimney
  }),
  // ── power-plant fuel silhouettes (upright) ──
  // nuclear: atom — nucleus + two crossed orbital ellipses
  "vt-nuclear": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(m, m, 3.8, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(m, m, 13, 5.5, Math.PI / 4, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(m, m, 13, 5.5, -Math.PI / 4, 0, Math.PI * 2); ctx.stroke();
  }),
  // coal: mound of lumps
  "vt-coal": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath(); ctx.arc(m - 6.5, m + 6, 6, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(m + 6.5, m + 6, 6, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(m, m - 4, 6.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillRect(m - 10, m + 6, 20, 6);
  }),
  // gas: flame
  "vt-gas": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 5);
    ctx.bezierCurveTo(m + 12, m - 4, m + 9, s - 8, m, s - 6);
    ctx.bezierCurveTo(m - 9, s - 8, m - 12, m - 4, m, 5);
    ctx.closePath(); ctx.fill();
    ctx.save(); ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath(); ctx.ellipse(m, s - 12, 3.5, 5, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }),
  // oil: derrick — narrow A-frame tower
  "vt-oil": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(m - 9, s - 6); ctx.lineTo(m, 6); ctx.lineTo(m + 9, s - 6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(m - 6, m + 2); ctx.lineTo(m + 6, m + 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(m - 9, s - 6); ctx.lineTo(m + 9, s - 6); ctx.stroke();
  }),
  // hydro: water drop
  "vt-hydro": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 5);
    ctx.bezierCurveTo(m + 11, m + 1, m + 9, s - 9, m, s - 7);
    ctx.bezierCurveTo(m - 9, s - 9, m - 11, m + 1, m, 5);
    ctx.closePath(); ctx.fill();
  }),
  // wind: turbine — mast + three blades
  "vt-wind": () => draw(S, (ctx, s) => {
    const m = s / 2;
    const hx = m, hy = m - 3;
    ctx.lineWidth = 3.2;
    ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx, s - 5); ctx.stroke(); // mast
    for (const a of [-Math.PI / 2, Math.PI / 6, (5 * Math.PI) / 6]) {         // blades
      ctx.beginPath(); ctx.moveTo(hx, hy);
      ctx.lineTo(hx + 12 * Math.cos(a), hy + 12 * Math.sin(a)); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(hx, hy, 3, 0, Math.PI * 2); ctx.fill();
  }),
  // solar: sun — disc + rays
  "vt-solar": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath(); ctx.arc(m, m, 6.5, 0, Math.PI * 2); ctx.fill();
    ctx.lineWidth = 3;
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      ctx.beginPath();
      ctx.moveTo(m + 10 * Math.cos(a), m + 10 * Math.sin(a));
      ctx.lineTo(m + 15 * Math.cos(a), m + 15 * Math.sin(a));
      ctx.stroke();
    }
  }),
  // other/unknown power: lightning bolt
  "vt-power": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m + 4, 5); ctx.lineTo(m - 8, m + 4); ctx.lineTo(m - 1, m + 4);
    ctx.lineTo(m - 4, s - 5); ctx.lineTo(m + 8, m - 2); ctx.lineTo(m + 1, m - 2);
    ctx.closePath(); ctx.fill();
  }),
  // substation: bordered square (electrical facility) enclosing a small bolt —
  // reads distinctly from a bare plant bolt, matching the map convention of a
  // boxed switchyard/transformer station.
  "vt-substation": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.lineWidth = 3;
    const r = 4, x0 = 8, y0 = 8, x1 = s - 8, y1 = s - 8;      // rounded square outline
    ctx.beginPath();
    ctx.moveTo(x0 + r, y0);
    ctx.arcTo(x1, y0, x1, y1, r); ctx.arcTo(x1, y1, x0, y1, r);
    ctx.arcTo(x0, y1, x0, y0, r); ctx.arcTo(x0, y0, x1, y0, r);
    ctx.closePath(); ctx.stroke();
    ctx.beginPath();                                          // enclosed bolt
    ctx.moveTo(m + 3, m - 7); ctx.lineTo(m - 5, m + 1); ctx.lineTo(m - 1, m + 1);
    ctx.lineTo(m - 3, m + 7); ctx.lineTo(m + 5, m - 1); ctx.lineTo(m + 1, m - 1);
    ctx.closePath(); ctx.fill();
  }),
  // biomass: leaf — two bezier lobes + midrib
  "vt-biomass": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 6);
    ctx.bezierCurveTo(m + 12, m - 6, m + 11, m + 8, m, s - 6);
    ctx.bezierCurveTo(m - 11, m + 8, m - 12, m - 6, m, 6);
    ctx.closePath(); ctx.fill();
    ctx.save(); ctx.globalCompositeOperation = "destination-out";
    ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(m, 10); ctx.lineTo(m, s - 9); ctx.stroke();   // midrib
    ctx.restore();
  }),
  // active-fire detection: teardrop flame, upright (never rotated) —
  // per-feature icon-color carries the confidence tint (see FIRE_COLOR)
  "vt-fire": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 3);
    ctx.bezierCurveTo(m + 10, m - 6, m + 8, m + 4, m + 4, m + 6);
    ctx.bezierCurveTo(m + 7, m - 2, m + 2, m - 4, m, m - 9);
    ctx.bezierCurveTo(m - 2, m - 4, m - 7, m - 2, m - 4, m + 6);
    ctx.bezierCurveTo(m - 8, m + 4, m - 10, m - 6, m, 3);
    ctx.closePath(); ctx.fill();
    ctx.beginPath();
    ctx.moveTo(m, m + 2);
    ctx.bezierCurveTo(m + 5, m + 5, m + 4, s - 6, m, s - 5);
    ctx.bezierCurveTo(m - 4, s - 6, m - 5, m + 5, m, m + 2);
    ctx.closePath(); ctx.fill();
  }),
  // wind arrow: slim shaft + head, points "up"/north — icon-rotate turns it
  // to the wind bearing (direction wind blows TOWARD, so rotate = deg+180
  // from OWM's from-direction; the caller owns that conversion)
  "vt-wind-arrow": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 4);                 // tip
    ctx.lineTo(m + 6, 14);
    ctx.lineTo(m + 2, 12.5);
    ctx.lineTo(m + 2, s - 6);
    ctx.lineTo(m - 2, s - 6);
    ctx.lineTo(m - 2, 12.5);
    ctx.lineTo(m - 6, 14);
    ctx.closePath(); ctx.fill();
  }),
  // river gauge: staff gauge — vertical post with level ticks over a
  // water wave (upright, never rotated)
  "vt-gauge": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.lineWidth = 3.2;
    ctx.beginPath(); ctx.moveTo(m, 5); ctx.lineTo(m, s - 12); ctx.stroke();   // staff
    for (const y of [9, 15, 21]) {                                            // level ticks
      ctx.beginPath(); ctx.moveTo(m, y); ctx.lineTo(m + 7, y); ctx.stroke();
    }
    ctx.lineWidth = 3.4;                                                      // water wave
    ctx.beginPath();
    ctx.moveTo(6, s - 9);
    ctx.quadraticCurveTo(m - 7, s - 15, m, s - 9);
    ctx.quadraticCurveTo(m + 7, s - 3, s - 6, s - 9);
    ctx.stroke();
  }),
  // train: locomotive nose-up — rounded body, cab window notch, nose taper
  // (points "up"/north so icon-rotate can turn it to bearing when known)
  "vt-train": () => draw(S, (ctx, s) => {
    const m = s / 2;
    ctx.beginPath();
    ctx.moveTo(m, 4);                          // nose tip
    ctx.lineTo(m + 7, 12);
    ctx.lineTo(m + 7, s - 8);
    ctx.lineTo(m - 7, s - 8);
    ctx.lineTo(m - 7, 12);
    ctx.closePath(); ctx.fill();
    ctx.save(); ctx.globalCompositeOperation = "destination-out";
    ctx.fillRect(m - 4.5, 14, 9, 4);           // cab window
    ctx.fillRect(m - 4.5, s - 16, 9, 3);       // body seam
    ctx.restore();
  }),
};

/** Register all SDF icons on a maplibre map (idempotent). */
export function registerIcons(map: any) {
  for (const [name, make] of Object.entries(shapes)) {
    if (!map.hasImage || map.hasImage(name)) continue;
    map.addImage(name, make(), { sdf: true });
  }
}

/** Render a registry shape to a tinted data-URL for the LEGEND (and any
 *  other UI). Draws the SAME ImageData the map registers, then tints it the
 *  way maplibre tints SDF icons (icon-color) — legend and map are the same
 *  source of truth by construction and can never diverge (DESIGN.md legend
 *  rule). Cached per (name,color): the legend re-renders freely. */
const iconUrlCache = new Map<string, string>();
export function iconDataURL(name: string, color: string, out = 18): string {
  const key = `${name}|${color}|${out}`;
  const hit = iconUrlCache.get(key);
  if (hit) return hit;
  const make = shapes[name];
  if (!make) return "";
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const ctx = c.getContext("2d")!;
  ctx.putImageData(make(), 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, S, S);
  const o = document.createElement("canvas");
  o.width = o.height = out;
  o.getContext("2d")!.drawImage(c, 0, 0, out, out);
  const url = o.toDataURL();
  iconUrlCache.set(key, url);
  return url;
}

export const AIRCRAFT_ICON: Record<AircraftClass, string> = {
  jet: "vt-jet", turboprop: "vt-prop", piston: "vt-prop",
  helicopter: "vt-heli", unknown: "vt-plane",
};

export const VESSEL_ICON: Record<VesselClass, string> = {
  tanker: "vt-tanker", cargo: "vt-cargo", passenger: "vt-cargo",
  fishing: "vt-boat", tug: "vt-boat", other: "vt-boat",
};

/** Strategic-site category (raw id from datacore/sites) -> silhouette. */
export const SITE_ICON: Record<string, string> = {
  port: "vt-port", tank_farm: "vt-tank", steel_mill: "vt-mill",
};

/** Power-plant fuel code (datacore/powerplants) -> silhouette + color. */
export const POWER_FUEL_ICON: Record<string, string> = {
  nuclear: "vt-nuclear", coal: "vt-coal", gas: "vt-gas", oil: "vt-oil",
  hydro: "vt-hydro", wind: "vt-wind", solar: "vt-solar", biomass: "vt-biomass",
  other: "vt-power",
};
export const POWER_FUEL_COLOR: Record<string, string> = {
  nuclear: "#c084fc", coal: "#94a3b8", gas: "#fbb24c", oil: "#ff8a5c",
  hydro: "#4d9fff", wind: "#7cc4ff", solar: "#fde047", biomass: "#4ade80",
  other: "#6680a0",
};
export const POWER_FUEL_LABEL: Record<string, string> = {
  nuclear: "Nuclear", coal: "Coal", gas: "Gas", oil: "Oil",
  hydro: "Hydro", wind: "Wind", solar: "Solar", biomass: "Biomass",
  other: "Other",
};

/** EIA-860 energy-source code (HIFLD PRIM_FUEL) -> canonical fuel key above.
 *  Anything not listed falls through to "other" (vt-power / gray). */
export const EIA_FUEL_TO_CANON: Record<string, string> = {
  SUN: "solar", WND: "wind", WAT: "hydro", NUC: "nuclear",
  NG: "gas", OG: "gas", BFG: "gas", RG: "gas", SGC: "gas",
  BIT: "coal", SUB: "coal", LIG: "coal", RC: "coal", WC: "coal", PC: "coal",
  DFO: "oil", RFO: "oil", KER: "oil", JF: "oil", WO: "oil",
  LFG: "biomass", WDS: "biomass", OBG: "biomass", MSW: "biomass", BLQ: "biomass",
  AB: "biomass", OBS: "biomass", OBL: "biomass", WDL: "biomass", SLW: "biomass", TDF: "biomass",
};
/** EIA-860 energy-source code -> precise human label (popup text). Covers the
 *  codes above plus the ones that fold to "other" for the icon but deserve an
 *  exact name in the detail card. */
export const EIA_FUEL_LABEL: Record<string, string> = {
  SUN: "Solar", WND: "Wind", WAT: "Hydro", NUC: "Nuclear",
  NG: "Natural gas", OG: "Other gas", BFG: "Blast-furnace gas", RG: "Refinery gas", SGC: "Coal-gas",
  BIT: "Bituminous coal", SUB: "Subbituminous coal", LIG: "Lignite coal", RC: "Refined coal",
  WC: "Waste coal", PC: "Petroleum coke",
  DFO: "Distillate fuel oil", RFO: "Residual fuel oil", KER: "Kerosene", JF: "Jet fuel", WO: "Waste oil",
  LFG: "Landfill gas", WDS: "Wood/wood waste", OBG: "Other biomass gas", MSW: "Municipal solid waste",
  BLQ: "Black liquor", AB: "Agricultural byproduct", OBS: "Other biomass solid", OBL: "Other biomass liquid",
  WDL: "Wood-waste liquid", SLW: "Sludge waste", TDF: "Tire-derived fuel",
  GEO: "Geothermal", MWH: "Battery storage", WH: "Waste heat", PUR: "Purchased steam",
};

/** FIRMS confidence (low/nominal/high) -> marker tint. */
export const FIRE_CONFIDENCE_COLOR: Record<string, string> = {
  low: "#fde047", nominal: "#fb923c", high: "#ff3b30",
};

/** Project a short velocity-vector endpoint from position/heading/speed.
 *  Length scales with speed (capped) — pure math, cheap for 10k features. */
export function velocityEndpoint(lat: number, lon: number, headingDeg: number,
                                 speedMs: number | null, scaleKmPerMs = 0.06): [number, number] {
  const speed = Math.min(speedMs || 0, 350);
  const km = Math.max(2, speed * scaleKmPerMs * 10);
  const rad = (headingDeg * Math.PI) / 180;
  const dLat = (km / 111.32) * Math.cos(rad);
  const dLon = (km / (111.32 * Math.max(0.1, Math.cos((lat * Math.PI) / 180)))) * Math.sin(rad);
  return [lon + dLon, lat + dLat];
}
