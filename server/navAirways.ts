// Airway expansion for SFDPS filed routes. ~52% of live plans carry only
// nasRouteText ("KBOS..HTO.J150.OOD..KATL"); the airway between two named fixes
// is published in FAA NASR AWY_BASE (public domain, scripts/build_nasr_airways.py).
// Only the unambiguous pattern FIX AIRWAY FIX is expanded, and only when both
// fixes lie on the airway; anything else (SID/STAR names, unknown idents,
// airway-to-airway joins, a fix not on the airway) is left unplaced — never guessed.
import fs from "fs";
import path from "path";
import { repoDataPath } from "./repoFiles";
import { lookupFix } from "./navFixes";
import { EARTH_RADIUS_NM } from "../shared/flightPlanGeometry";

let table: Map<string, string[][]> | null = null;
let cycle: string | null = null;

export function loadNavAirways(jsonPath?: string): number {
  if (table) return table.size;
  const p = jsonPath || repoDataPath(path.join("datacore", "aircraft", "nasr_airways.json"));
  try {
    const d = JSON.parse(fs.readFileSync(p, "utf-8"));
    table = new Map(Object.entries(d.airways || {}) as [string, string[][]][]);
    cycle = d.cycle ?? null;
  } catch (e: unknown) {
    console.error("[navAirways] load:", e instanceof Error ? e.message : e);
    table = new Map();
  }
  return table.size;
}

/** idents strictly along `airway` from `from` to `to` inclusive, in travel order;
 *  null when no variant holds both or the match is ambiguous (different variants
 *  give different segments). */
export function airwaySegment(airway: string, from: string, to: string): string[] | null {
  if (!table) loadNavAirways();
  const variants = table!.get(airway.toUpperCase());
  if (!variants) return null;
  const hits: string[][] = [];
  for (const v of variants) {
    const i = v.indexOf(from), j = v.indexOf(to);
    if (i < 0 || j < 0 || i === j) continue;
    hits.push(i < j ? v.slice(i, j + 1) : v.slice(j, i + 1).reverse());
  }
  if (hits.length === 0) return null;
  const key = hits[0].join(" ");
  return hits.every((h) => h.join(" ") === key) ? hits[0] : null;
}

const AIRWAY_RE = /^(?:[JVQTABGRLMNH]|UL|UA|UB|UG|UM|UN|UR|UT|UV|UW|UY|UZ)\d{1,3}$/;

/** Place the airway-expanded interior of a filed route: tokens `FIX AWY FIX`
 *  become the fixes along the airway. Returns ordered points (deduped at joins). */
export function expandRouteText(routeText: string | null | undefined): { lat: number; lon: number; name: string }[] {
  if (!routeText) return [];
  if (!table) loadNavAirways();
  // "&"-notes and speed/level groups are not route elements; dots separate tokens
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  const names: string[] = [];
  const push = (n: string) => { if (names[names.length - 1] !== n) names.push(n); };
  let expandedAny = false;
  for (let k = 0; k < toks.length; k++) {
    const t = toks[k];
    if (table!.has(t) && AIRWAY_RE.test(t) && names.length && k + 1 < toks.length) {
      const seg = airwaySegment(t, names[names.length - 1], toks[k + 1]);
      if (seg) { for (const n of seg) push(n); expandedAny = true; k++; continue; }
      continue; // airway we cannot bound: drop it, keep the surrounding fixes
    }
    if (lookupFix(t)) push(t);
  }
  // bare named fixes alone are a separate, untested step: emit only when an airway
  // was actually expanded (the ~52% text-only case this module exists for)
  if (!expandedAny) return [];
  const out: { lat: number; lon: number; name: string }[] = [];
  for (const n of names) { const p = lookupFix(n); if (p) out.push({ ...p, name: n }); }
  return out;
}

const DIRECT_MAX_LEG_NM = 1200; // a longer leg between two "fixes" means an ident collision, not a route

function legNm(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const r = Math.PI / 180;
  const h = Math.sin(((b.lat - a.lat) * r) / 2) ** 2 +
    Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(((b.lon - a.lon) * r) / 2) ** 2;
  return 2 * EARTH_RADIUS_NM * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** Place a PURE direct-fix filed route (`SID..FIX..FIX..FIX..STAR`, no airway
 *  anywhere). >=3 resolved fixes required, consecutive legs must be plausible,
 *  and any airway token refuses the whole route (skipping it would draw a
 *  straight line across a segment we could not expand). */
export function placeDirectFixes(routeText: string | null | undefined): { lat: number; lon: number; name: string }[] {
  if (!routeText) return [];
  if (!table) loadNavAirways();
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  if (toks.some((t) => AIRWAY_RE.test(t) && table!.has(t))) return [];
  const out: { lat: number; lon: number; name: string }[] = [];
  for (const t of toks) {
    const p = lookupFix(t);
    if (p && out[out.length - 1]?.name !== t) out.push({ ...p, name: t });
  }
  if (out.length < 3) return [];
  for (let i = 1; i < out.length; i++) if (legNm(out[i - 1], out[i]) > DIRECT_MAX_LEG_NM) return [];
  return out;
}

export type UnplacedReason = "noText" | "noFixResolved" | "fewFixes" | "airwayUnbounded" | "legTooLong" | "other";

/** Report-only: WHY a route text yields no placed points via expandRouteText /
 *  placeDirectFixes (diagnostic counter for the gate-1 funnel; never alters placement). */
export function classifyUnplacedRoute(routeText: string | null | undefined): UnplacedReason {
  if (!routeText || !routeText.trim()) return "noText";
  if (!table) loadNavAirways();
  const toks = routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter(Boolean);
  const hasAirway = toks.some((t) => AIRWAY_RE.test(t) && table!.has(t));
  const fixes: { lat: number; lon: number; name: string }[] = [];
  for (const t of toks) {
    const p = lookupFix(t);
    if (p && fixes[fixes.length - 1]?.name !== t) fixes.push({ ...p, name: t });
  }
  if (hasAirway) return "airwayUnbounded";
  if (fixes.length === 0) return "noFixResolved";
  if (fixes.length < 3) return "fewFixes";
  for (let i = 1; i < fixes.length; i++) if (legNm(fixes[i - 1], fixes[i]) > DIRECT_MAX_LEG_NM) return "legTooLong";
  return "other";
}

/** Report-only: route-text tokens that do NOT resolve in the NASR fix table (SID/STAR names,
 *  "DCT", airport idents, junk). Feeds the gate-1 unplaced-token diagnostic; never alters placement. */
export function unresolvedRouteTokens(routeText: string | null | undefined): string[] {
  if (!routeText) return [];
  if (!table) loadNavAirways();
  return routeText.toUpperCase().split("&")[0].split(/[\s.]+/).filter((t) => t && !lookupFix(t));
}

export const navAirwaysCycle = (): string | null => cycle;
export function resetNavAirways(): void { table = null; cycle = null; }
