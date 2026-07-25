// scaleModel — USER-CONTROLLED SCALE for the space frame (celestial v2 B2,
// directive §2, 2026-07-18: "True scale alone is useless for viewing ...
// Replace with USER-CONTROLLED scale, defaulting to a visible configuration").
// THE RULE THIS MODULE EXISTS TO ENFORCE: the LAYOUT may compress; the
// NUMBERS never lie. Everything here is RENDER LAYOUT ONLY — ephemeris
// positions, label distances, and the scale bar always come from the true
// ephemeris, never from these mappings.
//
// ── DISTANCE COMPRESSION (the c slider, [0,1]) ──────────────────────────────
// Chosen mapping: r' = AU · (r/AU)^p(c) with p(c) = 1 − 0.55·c, applied to a
// body's HELIOCENTRIC radial distance (direction always preserved).
// Equivalently ln(r'/AU) = p·ln(r/AU): a uniform contraction of log-distance
// about the 1 AU anchor. Why this form (vs. a hard log or piecewise blend):
//  · c = 0 is the EXACT identity (p = 1, special-cased so no pow rounding) —
//    the TRUE SCALE preset renders bit-identically to the B1 frame;
//  · strictly monotone in r for every c (p ≥ 0.45 > 0) — ordering of the
//    planets is preserved by construction;
//  · equal distance RATIOS map to equal compressed ratios — the "relative
//    sense of spacing" survives: Mars/Earth vs Saturn/Jupiter keep their
//    proportional relationship in log space;
//  · c = 1 (p = 0.45) puts Mercury→Neptune at ≈0.65→4.6 AU (span ratio ~7
//    instead of the true ~78), so the whole system fits one comfortable
//    perspective view from ~15 AU out — well inside the frame's existing
//    46.8 AU camera ceiling, which TRUE scale physically cannot satisfy
//    (framing 30 AU Neptune needs ~95 AU of standoff).
//
// PLANETOCENTRIC OFFSETS ARE NOT COMPRESSED — a deliberate, stated deviation
// from a literal reading of directive §2 ("mapping of heliocentric and
// planetocentric distances"): satellites ride their parent at the TRUE local
// offset (layout_moon = layout_earth + (true_moon − true_earth)). Reasons:
//  · local systems are already human-viewable at true scale (Moon 60 R⊕,
//    Io 6 R_J) — compression buys no visibility there;
//  · any nonidentity local mapping destroys the B1 physical payoff the human
//    approved ("standing near the Moon, Earth is right there — the live
//    one"): the 1 AU-anchored map would push the Moon 27× out; a local
//    power map would pull it 10× in — either way Earth stops subtending its
//    real ~1.9° from the Moon;
//  · the parentId channel in applyDistanceCompression is exactly where a
//    per-system local mapping plugs in later if B3's curated moons ever
//    need one.
//
// ── BODY SIZE (the s slider, [1, 5000]) ─────────────────────────────────────
// Multiplies RENDERED disc diameters so planets read at solar-system zoom.
//
// RESPONSE CURVE (SPACE VIEW VISUAL UPGRADE, human directive 2026-07-18 —
// reconciled with the confirmed reference scene research/directives/
// space_view_reference_2026-07-18.html): the reference renders a body at
// display radius ∝ (rel·m)^0.78 (rel = radius in Earth radii, m = the
// slider multiplier 1..5000, mid-slider ≈ ×648 — the value the human's
// screenshot confirmed). Its display/true enlargement factor is therefore
//   mEff = m^0.78 · rel^(−0.22)
// — big bodies get proportionally LESS enlargement, compressing the
// giant/terrestrial size ratio so both read at once. Adopted here, with
// OUR standing guards kept on top (the reconciliation rule: the reference
// wins on the LOOK, this system wins on the honesty guarantees):
//  · m = 1 is the EXACT identity (special-cased — the reference's own
//    "true scale" is rel^0.78, which is NOT 1:1; ours is), and mEff is
//    floored at 1 (the slider only ever enlarges);
//  · the apparent cap, anchor exemption, and Sun cap below all still
//    apply unchanged.
// Two guards keep it honest and keep every B1 contract intact:
//  · APPARENT CAP: enlargement never pushes a disc past SIZE_APPARENT_CAP_PX,
//    and a body whose TRUE disc is already at/above the cap renders at its
//    true size (rendered = trueDisc when trueDisc ≥ cap; else
//    min(s·trueDisc, cap)). Consequences, all pinned by tests: fly-to
//    arrival framing (true disc ~612 px) is exact at any s; apparent size
//    remains monotone non-increasing as the camera pulls away (no body ever
//    balloons while zooming out); s = 1 is the exact identity.
//  · THE MAP-ANCHOR BODY (Earth — the live map canvas) IS NEVER SIZE-
//    SCALED. The seam, the CSS pose, and the live-map↔impostor crossfade
//    are all keyed on Earth's true disc; multiplying it would fire the
//    seam exit while zooming OUT (cssScale > 1) and pin the map at
//    opacity 1 forever. Earth therefore stays honestly small at system
//    zoom — the marker + label carry it, exactly like B1.
//  · THE SUN is additionally capped at SUN_SIZE_MULT_CAP (×20) so maximum
//    exaggeration never swallows the inner system (tested against
//    Mercury's compressed orbit).
//
// PERF NOTE (directive asks worker precompute + GPU interpolate): with ten
// registry bodies the whole mapping is ~10 Math.pow calls ≈ 1–2 µs per
// (time, c) change, memoized in the frame — the worker/interpolation
// machinery becomes necessary only when the body count grows by orders of
// magnitude (B3+); building it now would be speculative complexity.
//
// Hermetic: pure math + a tiny persisted preference store (localStorage
// guarded, same pattern as lib/units.ts). No DOM. `npx tsx --test`-able.

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** 1 AU in meters (IAU 2012 exact) — duplicated from solarSystem.ts ON
 *  PURPOSE: importing it would drag the ephemeris into the eagerly-loaded
 *  bundle (this module is statically imported by the layers panel; the
 *  space frame stays a lazy chunk — the zoomSeam.ts precedent). */
export const AU_M = 149_597_870_700;

// ── scale state ─────────────────────────────────────────────────────────────

/** c: distance compression 0 (true 1:1) → 1 (fully compressed).
 *  s: body-size multiplier on rendered diameters, 1 (true) → 5000. */
export interface ScaleState {
  c: number;
  s: number;
}

export const SIZE_MULT_MIN = 1;
export const SIZE_MULT_MAX = 5000;

/** The reference response curve exponents: mEff = m^0.78 · rel^(−0.22)
 *  (see the BODY SIZE header note). */
export const SIZE_RESPONSE_EXP = 0.78;
export const SIZE_REL_EXP = 0.22;

/** rel is measured in Earth mean radii (the reference's convention). */
export const SIZE_REL_EARTH_KM = 6371;

/** The Sun's own multiplier ceiling (directive §2: "Sun capped separately
 *  so it never swallows the inner system"). 20 R☉ = 0.093 AU — under a
 *  third of Mercury's TRUE perihelion (0.307 AU), which is the smallest
 *  compressed perihelion over all c (compression only moves sub-AU
 *  distances outward). Tested. */
export const SUN_SIZE_MULT_CAP = 20;

/** Enlargement ceiling on any rendered disc, px. Bodies whose TRUE disc is
 *  already this big render true — which is what keeps fly-to framing and
 *  every close-range B1 guarantee exact under any s. */
export const SIZE_APPARENT_CAP_PX = 40;

/** p(c) exponent range: p(1) = 0.45 fits Mercury→Neptune in ~4.6 AU. */
export const COMPRESS_EXP_SPAN = 0.55;

export function clampScaleState(st: Partial<ScaleState> | null | undefined): ScaleState {
  const c = Number(st?.c);
  const s = Number(st?.s);
  return {
    c: Number.isFinite(c) ? Math.min(1, Math.max(0, c)) : SCALE_PRESET_VISIBLE.c,
    s: Number.isFinite(s) ? Math.min(SIZE_MULT_MAX, Math.max(SIZE_MULT_MIN, s)) : SCALE_PRESET_VISIBLE.s,
  };
}

// ── presets (directive §2: TRUE SCALE and VISIBLE, VISIBLE = startup) ───────

/** Both sliders at 1:1 — renders bit-identically to the B1 true-scale frame. */
export const SCALE_PRESET_TRUE: ScaleState = Object.freeze({ c: 0, s: 1 });

/** The startup default. c = 1 fits all eight planets in one view from
 *  ~15 AU; s = 647 is the reference's own default (slider 76 → ×647; the
 *  human's confirmed screenshot showed ×648 — one rounding hair off the
 *  same slider stop, and 647 round-trips the slider exactly) — through the response
 *  curve the giants render ~10–17 px at that framing while terrestrials
 *  sit at 1–3 px, so the B1 marker honesty machinery stays visibly
 *  exercised, and every fly-to arrival still frames the TRUE disc
 *  (apparent cap). */
export const SCALE_PRESET_VISIBLE: ScaleState = Object.freeze({ c: 1, s: 647 });

export function isTrueScale(st: ScaleState): boolean {
  return st.c === 0 && st.s === 1;
}

// ── distance compression ────────────────────────────────────────────────────

/** Compression exponent p(c) = 1 − COMPRESS_EXP_SPAN·c, clamped c. */
export function compressExponent(c: number): number {
  const cc = Math.min(1, Math.max(0, c));
  return 1 - COMPRESS_EXP_SPAN * cc;
}

/**
 * The radial map r → r' (meters). c = 0 returns r EXACTLY (identity is
 * special-cased so the TRUE preset carries zero pow rounding).
 */
export function compressDistance(rM: number, c: number): number {
  if (!(rM > 0)) return 0;
  const cc = Math.min(1, Math.max(0, c));
  if (cc === 0) return rM;
  return AU_M * Math.pow(rM / AU_M, compressExponent(cc));
}

export interface ScaleBodyIn {
  id: string;
  /** ride this body at the TRUE local offset (satellites — see header);
   *  null/undefined ⇒ compress the heliocentric (origin) radius. */
  parentId?: string | null;
  /** TRUE heliocentric position, meters. NEVER mutated by this module. */
  pos: Vec3;
}

/**
 * Compressed layout positions for a body set. Parentless bodies compress
 * their radial distance from the origin (the Sun) with direction preserved;
 * parented bodies attach to their parent's COMPRESSED position at the TRUE
 * local offset. `out` vectors are mutated in place when provided (the space
 * frame passes a preallocated record so slider drags allocate nothing);
 * input positions are read-only.
 */
export function applyDistanceCompression(
  bodies: ScaleBodyIn[],
  c: number,
  out?: Record<string, Vec3>,
): Record<string, Vec3> {
  const res: Record<string, Vec3> = out ?? {};
  const setV = (id: string, x: number, y: number, z: number): void => {
    const v = res[id];
    if (v) {
      v.x = x;
      v.y = y;
      v.z = z;
    } else {
      res[id] = { x, y, z };
    }
  };
  // pass 1: parentless (the heliocentric ladder)
  for (const b of bodies) {
    if (b.parentId) continue;
    const r = Math.hypot(b.pos.x, b.pos.y, b.pos.z);
    const k = r > 0 ? compressDistance(r, c) / r : 0;
    setV(b.id, b.pos.x * k, b.pos.y * k, b.pos.z * k);
  }
  // pass 2: satellites ride their (already-compressed) parent at true offset
  for (const b of bodies) {
    if (!b.parentId) continue;
    const parent = bodies.find((p) => p.id === b.parentId);
    const pOut = parent ? res[parent.id] : undefined;
    if (!parent || !pOut) {
      // orphan declaration: fall back to the heliocentric rule — never drop
      // a body (honesty: everything with a position renders somewhere real)
      const r = Math.hypot(b.pos.x, b.pos.y, b.pos.z);
      const k = r > 0 ? compressDistance(r, c) / r : 0;
      setV(b.id, b.pos.x * k, b.pos.y * k, b.pos.z * k);
      continue;
    }
    setV(
      b.id,
      pOut.x + (b.pos.x - parent.pos.x),
      pOut.y + (b.pos.y - parent.pos.y),
      pOut.z + (b.pos.z - parent.pos.z),
    );
  }
  return res;
}

// ── body size ───────────────────────────────────────────────────────────────

/**
 * Rendered disc diameter (px) from the TRUE disc diameter (px, computed by
 * the frame's projection at the body's LAYOUT distance). Pure screen-space:
 *  · map-anchor bodies: always the true disc (see header — seam integrity);
 *  · bodies that RIDE the anchor (parentIsMapAnchor — the Moon, whose
 *    parentId is Earth): also always the true disc. Two payoffs, one rule:
 *    (1) the one body pair with a fixed true-scale reference (Earth, which
 *    is never scaled) keeps its REAL size ratio — Moon = 0.27 R⊕, never the
 *    ~200× that the size boost would give a 1737 km body at s=647+, which is
 *    why the Moon rendered bigger than Earth; (2) the far impostor sprite
 *    (this function) and the close true-geometry sphere (drawBodyPatch, which
 *    is already decoupled from the slider) then agree at every distance — a
 *    boosted sprite pinned at SIZE_APPARENT_CAP_PX while the close sphere is
 *    true is exactly what made zooming the Moon feel stuck (apparent size
 *    frozen at the cap across a huge range) and then snap ("zooms all the way
 *    in") when the true disc finally overtook the cap. Consistent with the
 *    header's rule that local satellite systems are already human-viewable at
 *    true scale, so compression/exaggeration buys nothing there;
 *  · s = 1: the exact identity (TRUE preset renders bit-identically);
 *  · true disc ≥ SIZE_APPARENT_CAP_PX: true disc (close range stays real);
 *  · else: min(mEff · trueDisc, cap) with the REFERENCE response curve
 *    mEff = max(1, m^0.78 · rel^(−0.22)) (rel = body radius in Earth
 *    radii; the Sun's m additionally capped at SUN_SIZE_MULT_CAP).
 * Continuous and monotone in trueDiscPx; identity at s = 1.
 */
export function renderedDiscPx(
  trueDiscPx: number,
  sizeMult: number,
  relEarthRadii: number,
  isMapAnchor: boolean,
  isEmissive: boolean,
  parentIsMapAnchor = false,
): number {
  if (!(trueDiscPx > 0)) return 0;
  if (isMapAnchor || parentIsMapAnchor) return trueDiscPx;
  let m = Math.min(SIZE_MULT_MAX, Math.max(SIZE_MULT_MIN, sizeMult));
  if (isEmissive) m = Math.min(m, SUN_SIZE_MULT_CAP);
  if (m === 1) return trueDiscPx; // exact identity
  if (trueDiscPx >= SIZE_APPARENT_CAP_PX) return trueDiscPx;
  const rel = relEarthRadii > 0 && Number.isFinite(relEarthRadii) ? relEarthRadii : 1;
  const mEff = Math.max(1, Math.pow(m, SIZE_RESPONSE_EXP) * Math.pow(rel, -SIZE_REL_EXP));
  return Math.min(mEff * trueDiscPx, SIZE_APPARENT_CAP_PX);
}

// ── size-slider mapping (log scale: ×1 → ×5000 across 0..100 — the
// reference's own 10^((pct/100)·3.699) curve, since 3.699 = log10(5000)) ────

export function sizeSliderToMult(v: number): number {
  const vv = Math.min(100, Math.max(0, v));
  return Math.round(Math.pow(SIZE_MULT_MAX, vv / 100));
}

export function multToSizeSlider(s: number): number {
  const ss = Math.min(SIZE_MULT_MAX, Math.max(SIZE_MULT_MIN, s));
  return Math.round((100 * Math.log(ss)) / Math.log(SIZE_MULT_MAX));
}

// ── persisted preference store (the lib/units.ts pattern: localStorage +
// get/set/subscribe; layer state in this codebase persists via localStorage
// keys — vt-map-globe, vt-map-preset, vt-units — the URL hash is used for
// sub-view routing only, so scale state matches the localStorage pattern) ──

export const SCALE_PREF_KEY = "vt-celestial-scale";

export function readScalePref(): ScaleState {
  try {
    const raw = globalThis.localStorage?.getItem(SCALE_PREF_KEY);
    if (raw) return clampScaleState(JSON.parse(raw) as Partial<ScaleState>);
  } catch {
    /* no storage / bad JSON — fall through to the startup default */
  }
  return { ...SCALE_PRESET_VISIBLE };
}

let current: ScaleState = readScalePref();
const listeners = new Set<() => void>();

export function getCelestialScale(): ScaleState {
  return current;
}

export function setCelestialScale(st: Partial<ScaleState>): void {
  const next = clampScaleState({ ...current, ...st });
  if (next.c === current.c && next.s === current.s) return;
  current = next;
  try {
    globalThis.localStorage?.setItem(SCALE_PREF_KEY, JSON.stringify(current));
  } catch {
    /* best-effort */
  }
  listeners.forEach((fn) => {
    try {
      fn();
    } catch {
      /* listener owns its errors */
    }
  });
}

/** Subscribe to scale changes (useSyncExternalStore-compatible). */
export function subscribeCelestialScale(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
