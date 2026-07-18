# SOLAR VIEW SPIKE — O6-7 tier 2 camera handoff design

2026-07-16 · EARTH TWIN program (research/earth_twin_program.md → "O6-7 —
CELESTIAL, TWO TIERS", tier 2: "when you zoom out far enough … they appear
at literally accurate scale"). This file is the integration recipe for the
parent session's datamap.tsx PR. The spike itself is NEW FILES ONLY; every
edit to an existing file that integration requires is listed in §6 and was
deliberately NOT made here.

## 1. What this spike shipped (all new files, no existing file touched)

- `client/src/lib/celestial/solarSystem.ts` — real ephemeris: Sun, Moon,
  Earth + the 8 planets. Theory: Paul Schlyter, "How to compute planetary
  positions" (stjarnhimlen.se/comp/ppcomp.html), derived from van Flandern
  & Pulkkinen, ApJS 41 (1979) — osculating elements + the major Moon/
  Jupiter/Saturn/Uranus perturbation terms, ecliptic-of-date frame, §8
  precession converter for J2000 comparisons. Pure function of the
  timestamp; no network, no DOM.
- `client/src/lib/celestial/solarSystem.test.ts` — anchor validation
  against JPL Horizons API vectors (DE441; queries quoted in the file) at
  J2000.0 and 2026-01-01, plus scale-math and physical-band invariants.
  MEASURED accuracy at the anchors: planets ≤ 1.25 arcmin angular error
  (worst: Saturn E2000), Moon ≤ 3.01 arcmin; heliocentric distance ≤
  0.013% relative (inner) / 0.24% (outer, Saturn E2026); Moon distance ≤
  0.064%. Asserted with ~2x headroom (planets 3', Moon 6') so the pin
  catches dropped perturbation terms without flaking on the theory's own
  noise floor. Honest tier: arcmin-class, visualization-grade — NOT
  navigation/occultation/pointing grade.
- `client/src/lib/celestial/solarView.ts` — the tier-2 prototype renderer.
  2D canvas overlay, no React, no map dependency. API:
  `mount(container, opts) → handle`, `handle.setTime(date)`,
  `handle.setScale(mPerPx)`, `handle.setAnnotate(on)`, `handle.render()`,
  `handle.getState()`, `handle.dispose()`. One number (metersPerPixel)
  maps world to screen; every body is drawn at its TRUE radius and TRUE
  position (top-down orthographic projection of the ecliptic plane — z
  discarded in projection only, never in the data). Sub-pixel bodies are
  drawn as a single dim pixel — at whole-system scale even the Sun is
  sub-pixel, and that emptiness is the honesty the human asked for.
  Annotation mode (default ON, labeled on-canvas "ANNOTATED — markers/
  labels only; sizes & distances never scaled"): ring markers OUTSIDE the
  true disc, name + true-distance labels (lib/units.ts formatters below
  0.01 AU, AU above — UNITS PREFERENCE directive honored), orbit paths
  sampled from the same ephemeris, edge pointers for off-screen bodies.
  Chrome: real scale bar (1/2/5 decades), provenance caption ("COMPUTED
  EPHEMERIS · TRUE SCALE"), time readout, live Earth-disc-px readout.
  No starfield: decorative random stars would violate "real position or
  absent" — a real bright-star catalog (e.g. Yale BSC) is future work.
- `client/src/lib/celestial/solarView.test.ts` — pure view-math pins:
  projection orientation, literal true-scale (Earth = 637.1 px radius at
  10 km/px; inner system sub-pixel at 30 AU/450 px), entry/exit
  hysteresis, scale clamp, scale-bar mantissas, unit-labeled distances.

Branch note: this worktree branch was cut just before #494 (tier 1,
`client/src/lib/celestial/ephemeris.ts`) merged to main. All spike files
are NEW, so merging is conflict-free; after merge the celestial/ dir holds
both tiers. ephemeris.ts (tier 1) and solarSystem.ts (tier 2) each carry a
small truncated solar-longitude series — deliberate spike duplication (two
different truncations for two different jobs); if it ever bothers a
staleness audit, tier 1's subsolar math could be re-derived from
solarSystem.ts, but nothing requires it now.

## 2. The handoff: globe → solar system

The map camera cannot leave the globe (MapLibre's transform is planet-
locked), so tier 2 is a SEPARATE renderer that takes over the viewport at
the zoom floor. The whole trick is making the seam invisible: Earth must
not jump in size, and input must never be owned by both surfaces at once.

### 2.1 Threshold detection (enter)

Facts (verified in maplibre-gl 5.24.0 dist + datamap.tsx on main):
- datamap.tsx creates the map with NO explicit minZoom → MapLibre default
  floor is zoom 0.
- In globe projection the globe's apparent radius is
  `getGlobeRadiusPixels(worldSize, centerLat) = worldSize / (2π) /
  cos(centerLat)` px, with `worldSize = 512 · 2^zoom`. At zoom 0 on the
  equator: radius ≈ 81.5 px → disc ≈ 163 px.

At the floor, further zoom-out gestures no longer change `map.getZoom()`,
so "zoom" events go silent — intent must be read from the raw gesture:

- WHEEL (desktop): a capture-phase `wheel` listener on the map container.
  When `map.getZoom() <= map.getMinZoom() + 0.05` and `e.deltaY > 0`,
  accumulate deltaY; reset the accumulator 400 ms after the last tick;
  trigger the handoff at ≥ 240 accumulated (≈ 2-3 notches — one stray
  notch never triggers). Do NOT preventDefault before triggering; the map
  is already at the floor so the events are inert.
- PINCH (touch): MapLibre exposes no "pinch past floor" signal without
  poking internals — do not try. Mobile path is the explicit affordance:
- BUTTON (all inputs, accessibility): when the camera sits at the zoom
  floor, show a small "Solar system" chip (same family as the existing
  zoom-out affordances). Click = same handoff. This is the guaranteed
  path; the wheel accumulator is the delightful one.

### 2.2 Scale continuity (the no-jump rule)

At handoff, measure the globe's CURRENT apparent disc and hand it to the
solar view so the drawn Earth is pixel-for-pixel the size the globe was:

```ts
const worldSize = 512 * Math.pow(2, map.getZoom());
const discPx = worldSize / Math.PI / Math.cos(map.getCenter().lat * Math.PI / 180);
const handle = mount(container, {
  metersPerPixel: entryScaleForEarthDisc(discPx),
  onZoomIntoEarth: exitToGlobe,
});
```

Perspective foreshortening makes the rendered limb a few px smaller than
the formula; the integration PR should verify the seam visually (harness
screenshot at the boundary) and, if the jump is visible, correct discPx
with a one-time measured factor — a constant, stated in a comment, not a
fudge of the solar view's scale math (the view stays literally true).

FLAT-PROJECTION NOTE: the handoff only makes sense from the globe. If the
user has the W1 toggle on mercator, either (a) suppress the handoff, or
(b) flip to globe first (`map.setProjection({type:"globe"})`), then hand
off. Recommend (b) — it reads as "zooming out of the world".

### 2.3 Transition animation

Enter: mount the solar view (it appends an absolutely-positioned canvas
over the map inside the same container), start at `opacity: 0`, fade to 1
over ~250 ms (CSS transition on the canvas). The map keeps rendering
beneath during the fade — Earth-on-globe dissolves into Earth-at-true-
scale at the same screen size. After the fade completes, the map can be
left untouched (it is fully covered; MapLibre renders on demand, so a
static covered map costs ~nothing).

Exit (reverse): fade the canvas to 0 over ~250 ms, `handle.dispose()`,
then `map.easeTo({ zoom: map.getMinZoom() + 0.75, duration: 400 })`. The
+0.75 gives the user zoom headroom so the next wheel notch doesn't
instantly re-trigger the handoff (enter-side hysteresis; the exit side is
inside the view: exitFraction 0.6 > entry disc fraction).

### 2.4 Input routing while the solar view owns the canvas

- The solar canvas sits on top (`position:absolute; inset:0`; give it
  `z-index` above the layer panel triggers — 30 works with the current
  stack) with default `pointer-events`, so pointer/wheel events physically
  cannot reach the map canvas. Its wheel handler preventDefaults (page
  never scrolls).
- Belt and braces: disable the map's handlers for the duration —
  `map.scrollZoom/dragPan/dragRotate/doubleClickZoom/touchZoomRotate/
  keyboard` `.disable()` on enter, `.enable()` on exit — so a focused-map
  keyboard zoom can't move the hidden camera.
- Hide the map controls (`.maplibregl-ctrl-bottom-left` etc.) via a
  container class (e.g. `vt-solar-active`) — zoom chevrons pointing at a
  hidden map are a lie.
- Keyboard: the solar view binds nothing global. The parent's EXISTING
  Escape handler (datamap.tsx ~line 1075 on main) gains one early case:
  if solar view active → exit to globe, consume the event.
- In-view interactions (already implemented in the spike): wheel = zoom
  about cursor, drag = pan, double-click = re-center on Earth. Zooming IN
  past `exitFraction` (Earth disc > 60% of the viewport's short side)
  fires `onZoomIntoEarth` exactly once → parent runs the exit sequence.

### 2.5 Returning to the globe — user paths

1. Zoom in on Earth (wheel/pinch-equivalent drag) → `onZoomIntoEarth`.
2. Escape (parent handler).
3. A visible "Back to Earth" chip the parent renders while active (mobile
   has no wheel; this is its guaranteed return path, mirroring the enter
   button).
All three converge on the same `exitToGlobe()`.

## 3. Time and units wiring

- TIME AXIS (lib/timeAxis.ts): on enter, seed `mount(…, { timeMs })` from
  `getTimeAxis()` (historical instant if scrubbing, else `Date.now()`).
  Subscribe while active: axis change → `handle.setTime(...)`. In LIVE
  mode, a 60 s interval calling `handle.setTime(Date.now())` is plenty —
  the fastest thing on screen (the Moon) moves ~0.5 arcmin of its orbit
  per minute; clear the interval on exit. The time scrubber becomes a
  planetarium for free.
- UNITS (lib/units.ts): the view renders distance labels through fmtKm
  already; subscribe `subscribeUnits(() => handle.render())` while active
  so an imperial/metric flip repaints labels. Unsubscribe on exit.

## 4. Performance

- Render-on-demand only: renders happen on mount/zoom/pan/setTime/resize/
  render(). No rAF loop, no timers inside the view.
- One frame costs ~10 `solarSystemState` evaluations' worth of math for
  bodies (sub-ms) + annotation orbit paths: 9 orbits × 193 samples of the
  ephemeris, cached and reused until setTime moves > 6 h — first
  annotated frame ~a few ms, subsequent frames sub-ms. No allocation
  churn beyond small per-frame arrays.
- DPR-aware canvas (crisp on retina), ResizeObserver for viewport
  changes. No WebGL, no workers, no network — zero requests.
- Zero cost when not mounted. Ship it behind a dynamic import
  (`await import("@/lib/celestial/solarView")`) triggered by the handoff
  so the map bundle grows by nothing until a user actually leaves Earth.
- Perf-harness note for the integration PR: assert the map's own frame
  budget is unaffected while the view is active (map is static beneath),
  and that enter+first-paint stays under ~50 ms on the harness box.

## 5. Exactly what the parent session must wire (datamap.tsx PR checklist)

All anchors refer to datamap.tsx on main (post-#494).

1. Lazy import `mount` + `entryScaleForEarthDisc` from
   `@/lib/celestial/solarView` (dynamic import inside the enter handler).
2. State: `solarActiveRef` (ref, not state, for the wheel handler) + a
   `solarHandleRef`; a `useState` mirror only if the button/chip UI needs
   re-render.
3. Enter detection (§2.1): capture-phase wheel listener on
   `mapContainer.current` (the `<div ref={mapContainer}
   className="vt-map-canvas" />`, ~line 6114) with the deltaY
   accumulator, plus the "Solar system" chip shown when
   `map.getZoom() <= map.getMinZoom() + 0.05`.
4. Enter sequence (§2.2-2.4): compute discPx → `mount` into
   `mapContainer.current` → fade in → disable the six map handlers →
   add `vt-solar-active` class (controls hidden via CSS).
5. Exit sequence (§2.3): fade out → `dispose()` → re-enable handlers →
   remove class → `easeTo(minZoom + 0.75)`. Wire it as the
   `onZoomIntoEarth` callback, the Escape early-case (~line 1075), and
   the "Back to Earth" chip.
6. Time axis + units subscriptions while active (§3).
7. CSS (index.css): `.vt-solar-active .maplibregl-ctrl { display: none }`
   plus an opacity transition class for `.vt-solar-view`. (The spike
   inlines its canvas styles so it runs without any CSS edit; moving them
   to index.css during integration is optional polish.)
8. Mercator case: flip to globe before handoff (§2.2 note).
9. PROMOTION RULES: own PR, real `npm run build`, visual harness at
   390/768/1440 including a screenshot at the handoff boundary and one
   inside the solar view (annotated + un-annotated); version bump at
   commit time; experiments.md entry tagged [PRODUCT].
10. HONESTY review before merge: every on-screen number in the view must
    trace to solarSystem.ts or the scale math — the view never invents a
    size, a distance, or a position. The annotation label text and the
    provenance caption ship as-is unless DESIGN.md dictates styling.

Existing files this spike deliberately did NOT touch (the integration PR
owns them): `client/src/pages/datamap.tsx`, `client/index.css` (optional),
`package.json` (version bump), `research/experiments.md` (session log).
No change to `ephemeris.ts` is required.

## 6. Follow-ups filed (not in this spike)

- GEO shell at the boundary: once satellites render at tier boundary
  (SDP4 shipped in #492), the 35,786 km GEO ring is ~5.6 Earth radii —
  visible and true at handoff scale; drawing it in the solar view needs
  the sat positions handed in (solarView stays dependency-free).
- Real star backdrop from a real catalog (Yale BSC / Hipparcos bright
  subset) — positions real, magnitudes real, or absent.
- 3D/tilted view: the spike is top-down orthographic (z discarded in
  projection). A tilted camera needs no new ephemeris — same vectors.
- Planet inclination honesty: at top-down scale the z-excursions are
  invisible anyway (Mercury's ±0.4 AU worst); if a future edge-on view
  ships, z is already in the data.

---
SUPERSEDED 2026-07-18: the separate solar-system view this spike led to
(solarView.ts, shipped v1.0.358/364) is retired — replaced by the
continuous true-scale space frame (client/src/lib/celestial/
spaceFrame.ts; experiments.md 2026-07-18 v1.0.396 entry). The spike's
ephemeris findings (Schlyter/van Flandern accuracy, body radii) live on
in solarSystem.ts, which spaceFrame consumes. Kept for the record.
