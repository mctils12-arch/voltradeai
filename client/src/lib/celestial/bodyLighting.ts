// bodyLighting — PURE sun-driven lighting geometry for the CELESTIAL v2 space
// frame (B6 "UNIVERSAL LIGHTING", directive §8). ONE Sun lights everything at
// every zoom; every terminator, phase, eclipse and ring shadow here is computed
// from REAL Sun+body positions at the sim-clock instant — never faked, never
// labelled more precise than the ephemeris (~arcmin). No DOM, no deps,
// `npx tsx --test`-able.
//
// WHAT LIVES HERE (all pure vector math):
//  · terminatorLit — the day/night sign a surface point carries (n·ŝ);
//  · phaseFraction — the illuminated fraction of a body's disc from a
//    viewpoint ((1+cosα)/2) — the crescent/gibbous geometry (identical to
//    spaceFrame.apparentLitFraction, pinned here as the shared contract);
//  · shadowConeFactor — the umbra/penumbra cone of ANY occluder on ANY point
//    (lunar eclipse = Moon in Earth's cone; Jovian-moon eclipse = moon in
//    Jupiter's cone; craft in Earth's umbra — this generalises the ENU-frame
//    lib/orbital/followCamera.earthShadow to a 3-D body↔body frame);
//  · surfaceInRingShadow — a planet surface point shadowed by its ring
//    annulus (Saturn's rings darken the winter hemisphere);
//  · pointInSphereShadow — a ring point (or craft) shadowed BY the planet
//    (the planet's shadow cast onto its own rings).
//
// The realistic-lighting toggle lives at the call sites: realistic OFF ⇒ the
// shading paths skip the lambert term (even-lit inspection mode); realistic ON
// ⇒ lambert × the shadow factors below. Keeping this module geometry-only (no
// import of textureSphere's lambert curve) lets textureSphere import IT for the
// per-pixel ring test without a cycle.

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;

// ── terminator + phase ──────────────────────────────────────────────────────

/**
 * Signed illumination of a surface point: n·ŝ, with `normal` the outward unit
 * surface normal and `sunDir` the unit direction TO the Sun (same frame). > 0
 * ⇒ the lit (day) hemisphere, < 0 ⇒ the dark (night) hemisphere, 0 ⇒ the
 * terminator. This IS the terminator: the sign flips exactly across the great
 * circle ⊥ to the Sun direction. (The soft ±band and 5 % night floor are
 * display smoothing applied downstream — see textureSphere.lambertWeight.)
 */
export function terminatorLit(normal: Vec3, sunDir: Vec3): number {
  return dot(normal, sunDir);
}

/**
 * Illuminated fraction of a body's disc as seen from a viewpoint: (1 + cos α)/2
 * with α the Sun–body–camera angle. `toSun`/`toCam` are unit vectors FROM the
 * body toward the Sun and toward the camera. Full phase (Sun behind the
 * observer) → 1; new (body between observer and Sun, toSun ≈ −toCam) → 0, i.e.
 * a thin crescent; quadrature → 0.5. Identical to
 * spaceFrame.apparentLitFraction — the one phase contract, pinned here.
 */
export function phaseFraction(toSun: Vec3, toCam: Vec3): number {
  return (1 + dot(toSun, toCam)) / 2;
}

// ── umbra / penumbra shadow cone (eclipses + craft umbra) ────────────────────

export interface ShadowConeResult {
  /** direct-sunlight factor 0..1 — 1 sunlit, 0 in umbra, ramp across penumbra. */
  factor: number;
  region: "sunlit" | "penumbra" | "umbra";
}

/**
 * Direct-sunlight factor at a point `pRel` (relative to the OCCLUDER centre,
 * metres) inside/around the shadow the occluder casts away from the Sun. Real
 * cone geometry: with ŝ the unit occluder→Sun direction, the along-axis depth
 * is d = −pRel·ŝ (> 0 on the anti-sun side) and the off-axis miss distance is
 * r = |pRel − (pRel·ŝ)ŝ|. The umbra converges at tanα = (R☉ − Rocc)/D, the
 * penumbra diverges at tanβ = (R☉ + Rocc)/D (D = occluder→Sun distance). Inside
 * the umbra radius ⇒ 0; outside the penumbra radius ⇒ 1; a linear ramp across
 * the band (a stated display-grade approximation of the real partial-eclipse
 * falloff — the SAME model lib/orbital/followCamera.earthShadow uses for the
 * craft, generalised off the ENU axis to an arbitrary 3-D geometry).
 *
 * Lunar eclipse: pRel = Moon − Earth, ŝ = unit(Sun − Earth), Rocc = R⊕,
 * R☉ = Sun radius, D = |Sun − Earth|. Jovian-moon eclipse: occluder = Jupiter.
 */
export function shadowConeFactor(
  pRel: Vec3,
  sunDir: Vec3,
  occluderRadiusM: number,
  sunRadiusM: number,
  sunDistM: number,
): ShadowConeResult {
  const su = dot(pRel, sunDir); // signed along-axis coordinate
  const dAlong = -su; // > 0 on the anti-sun (shadow) side
  if (dAlong <= 0) return { factor: 1, region: "sunlit" }; // sun-ward hemisphere
  const off2 = dot(pRel, pRel) - su * su;
  const rOff = Math.sqrt(Math.max(0, off2));
  const tanUmbra = (sunRadiusM - occluderRadiusM) / sunDistM; // converging
  const tanPen = (sunRadiusM + occluderRadiusM) / sunDistM; // diverging
  const rUmbra = occluderRadiusM - dAlong * tanUmbra;
  const rPen = occluderRadiusM + dAlong * tanPen;
  if (rUmbra > 0 && rOff <= rUmbra) return { factor: 0, region: "umbra" };
  if (rOff >= rPen) return { factor: 1, region: "sunlit" };
  const span = Math.max(1, rPen - Math.max(0, rUmbra));
  const f = (rOff - Math.max(0, rUmbra)) / span;
  return { factor: Math.max(0, Math.min(1, f)), region: "penumbra" };
}

// ── ring shadows ────────────────────────────────────────────────────────────

/**
 * True when a planet surface point (outward unit normal `p`, in the BODY frame
 * whose z-axis is the spin axis so the equatorial ring plane is z = 0) is
 * shadowed by the ring annulus [rInner, rOuter] (planet radii). `sun` is the
 * unit Sun direction in the SAME body frame. The ray from the surface point
 * toward the Sun (p + t·sun, planet-radius units) crosses the ring plane at
 * t = −p_z / sun_z; if t > 0 (the ring lies between the point and the Sun) and
 * the crossing radius ρ = |(p + t·sun)_xy| falls inside the annulus, the point
 * is in ring shadow. Only the day hemisphere (n·sun > 0) can be ring-shadowed;
 * geometry puts the band on the hemisphere opposite the Sun's ring-plane
 * elevation (the "winter" hemisphere) — its width and latitude track the Sun's
 * elevation above the ring plane exactly.
 */
export function surfaceInRingShadow(p: Vec3, sun: Vec3, rInner: number, rOuter: number): boolean {
  if (dot(p, sun) <= 0) return false; // night side — already dark
  if (Math.abs(sun.z) < 1e-4) return false; // Sun in the ring plane: no cast band
  const t = -p.z / sun.z;
  if (t <= 0) return false; // ring plane is not between the point and the Sun
  const cx = p.x + t * sun.x;
  const cy = p.y + t * sun.y;
  const rho = Math.hypot(cx, cy);
  return rho >= rInner && rho <= rOuter;
}

/**
 * True when a point `qRel` (relative to the sphere centre, any consistent
 * units) is shadowed by the sphere of radius `sphereRadius` for a Sun in unit
 * direction `sunDir`: the ray qRel + t·ŝ (t > 0) intersects the sphere. Used
 * for the planet's shadow cast onto its own rings (ring point behind the
 * planet from the Sun) and equivalently for a craft in the planet's umbra.
 * The sphere must lie sun-ward of the point (qRel·ŝ < 0) and the line's miss
 * distance to the centre must be under the radius.
 */
export function pointInSphereShadow(qRel: Vec3, sunDir: Vec3, sphereRadius: number): boolean {
  const along = dot(qRel, sunDir);
  if (along >= 0) return false; // the point is sun-ward of the sphere
  const perp2 = dot(qRel, qRel) - along * along;
  return perp2 < sphereRadius * sphereRadius;
}
