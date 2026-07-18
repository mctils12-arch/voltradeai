// zoomSeam — the SHARED exponential-zoom math of the map↔space seam
// (celestial v2 B1, directive §1: "The existing zoom controls (buttons,
// scroll, pinch) extend continuously ... No mode switch").
//
// Why a separate module: datamap.tsx wires EVERY zoom input (wheel, the
// +/- NavigationControl buttons, pinch, keyboard) through the same seam,
// but the space frame itself stays a LAZY chunk (the map bundle grows by
// nothing until a user actually leaves Earth). These few constants are the
// only part both sides need at page load, so they live here — statically
// importable by datamap, re-exported by spaceFrame so the frame's public
// surface (and its tests) keep one name for each contract.
//
// Hermetic: pure math, no DOM — unit-testable with `npx tsx --test`.

/** One standard wheel notch (deltaY 100) multiplies camera distance by this
 *  (~×1.18/tick per the approved design); deltaY scales the exponent so
 *  trackpads glide smoothly through the same curve. */
export const ZOOM_STEP_PER_NOTCH = 1.18;

/** Exponential zoom: wheel deltaY → camera-distance multiplier. deltaY 100
 *  (one notch) = ×ZOOM_STEP_PER_NOTCH out; negative deltaY zooms in. */
export function zoomStepFactor(deltaY: number): number {
  return Math.pow(ZOOM_STEP_PER_NOTCH, deltaY / 100);
}

/** Inverse of zoomStepFactor: the wheel deltaY whose exponential step
 *  multiplies camera distance by `factor` (factor < 1 ⇒ negative deltaY =
 *  zoom in). This is how discrete inputs (buttons, keys, an accumulated
 *  pinch ratio) ride the exact same curve as the wheel. */
export function wheelDeltaForFactor(factor: number): number {
  return (100 * Math.log(factor)) / Math.log(ZOOM_STEP_PER_NOTCH);
}

/** One +/- zoom-BUTTON click (or keyboard +/-) in the space frame, as wheel
 *  deltaY. ×2 per click — exactly one MapLibre zoom level, so the buttons
 *  feel identical on both sides of the seam (on the map one click doubles
 *  the globe's apparent size; out here it doubles camera distance, which is
 *  the same visual step under the shared perspective camera). */
export const ZOOM_BUTTON_DELTAY = wheelDeltaForFactor(2);

/** The wheel-accumulator threshold at the map's zoom floor: ≥ this much
 *  deltaY inside one accumulation window ≈ one real notch of intent —
 *  trackpad micro-jitter never triggers the seam. (Shared so the pinch
 *  path can reuse the same "one real gesture" bar.) */
export const SEAM_ENTRY_DELTAY = 60;
