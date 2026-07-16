// Auto-3D satellite selection — EARTH TWIN O6-2 (human directive: "zoom in
// and see them like this WITHOUT clicking … zoom past them, they go away").
//
// Pure selection: at close camera altitudes, the satellites nearest the view
// centre (and inside the viewport) get a 3D class-form mini instead of a
// glyph. HONESTY: only objects with a SATCAT row qualify — an unidentified
// object stays a dot at every zoom (never dressed as a spacecraft). The cap
// bounds draw calls; the count is surfaced by the caller, never silent.

import { SAT_STRIDE } from './satBuffer.js';
import { classFormNamed, type FormKind } from './model3d.js';
import type { GpRecord, SatcatRecord } from './tle.js';

/** Band ceiling: minis appear when the camera is below this altitude (and
 *  disappear with the whole layer below the registry LOD floor — "zoom past
 *  them and back to the earth" is the existing envelope). */
export const MINI_MAX_CAM_KM = 2500;
/** Max simultaneous minis (one small uniform-anchored draw each). */
export const MINI_CAP = 12;

export interface MiniSat {
  index: number;
  mercX: number;
  mercY: number;
  altMeters: number;
  form: FormKind;
}

/** Per-index form lookup (null = uncatalogued → stays a dot). Computed once
 *  when SATCAT lands, index-aligned with the gp array/worker buffer. */
export function formsFromSatcat(
  gp: GpRecord[],
  byNorad: Map<number, SatcatRecord>,
): (FormKind | null)[] {
  return gp.map((g) => {
    const sc = byNorad.get(g.noradId);
    return sc ? classFormNamed(sc.objectType, sc.rcsSize, g.name ?? sc.name) : null;
  });
}

const wrapDx = (dx: number): number => {
  const a = Math.abs(dx) % 1;
  return Math.min(a, 1 - a);
};

/** Inside-viewport test with antimeridian wrap on x. */
function inView(x: number, y: number, v: { minX: number; maxX: number; minY: number; maxY: number }): boolean {
  if (y < v.minY || y > v.maxY) return false;
  if (x >= v.minX && x <= v.maxX) return true;
  return x + 1 >= v.minX && x + 1 <= v.maxX || x - 1 >= v.minX && x - 1 <= v.maxX;
}

/**
 * The minis to draw this tick: valid + catalogued + in view, nearest to the
 * view centre first, capped. excludeIndex drops the FOLLOWED object (it
 * already has the big focused model).
 */
export function selectMiniSats(
  buffer: ArrayLike<number> | null,
  forms: (FormKind | null)[] | null,
  view: { minX: number; maxX: number; minY: number; maxY: number; cx: number; cy: number },
  cap: number = MINI_CAP,
  excludeIndex = -1,
  /** CO-LOCATION THINNING (live bug: Progress/Dragon/modules are catalogued
   *  separately but orbit AT the ISS — a mini rendered on top of the
   *  focused model). Two minis closer than this (mercator units) collapse
   *  to the nearest-to-centre one; anything this close to excludePos (the
   *  FOCUSED object) is dropped — the big model owns that spot. */
  minSepMerc = 0,
  excludePos: { x: number; y: number } | null = null,
): MiniSat[] {
  if (!buffer || !forms) return [];
  const n = Math.min(Math.floor(buffer.length / SAT_STRIDE), forms.length);
  const candidates: (MiniSat & { d2: number })[] = [];
  for (let i = 0; i < n; i++) {
    if (i === excludeIndex) continue;
    const form = forms[i];
    if (!form) continue; // uncatalogued: honest dot, never a guessed form
    const base = i * SAT_STRIDE;
    if (buffer[base + 3] < 0) continue; // sentinel slot (deep-space/invalid/filtered)
    const x = buffer[base], y = buffer[base + 1];
    if (!inView(x, y, view)) continue;
    const dx = wrapDx(x - view.cx), dy = y - view.cy;
    candidates.push({ index: i, mercX: x, mercY: y, altMeters: buffer[base + 2], form, d2: dx * dx + dy * dy });
  }
  candidates.sort((a, b) => a.d2 - b.d2);
  if (minSepMerc <= 0) return candidates.slice(0, cap).map(({ d2: _d2, ...m }) => m);
  const sep2 = minSepMerc * minSepMerc;
  const near = (ax: number, ay: number, bx: number, by: number) => {
    const dx = wrapDx(ax - bx), dy = ay - by;
    return dx * dx + dy * dy < sep2;
  };
  const out: MiniSat[] = [];
  for (const c of candidates) {
    if (out.length >= cap) break;
    if (excludePos && near(c.mercX, c.mercY, excludePos.x, excludePos.y)) continue;
    if (out.some((m) => near(c.mercX, c.mercY, m.mercX, m.mercY))) continue;
    const { d2: _d2, ...m } = c;
    out.push(m);
  }
  return out;
}
