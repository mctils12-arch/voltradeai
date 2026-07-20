<!-- PROVENANCE: verbatim human-authored design handoff, uploaded 2026-07-20
as design_handoff_flight_track_3d.zip (README.md + flight-path-3d.html +
screenshots). Filed per the space_view_handoff precedent (the 2026-07-19
3D-terrain wave's handoff docs were never committed — experiments.md records
the resulting "same issues" recurrence; this filing closes that gap).
Reference prototype: flight_track_3d_reference_2026-07-20.html (three.js —
a DESIGN REFERENCE, not production code; the site implementation lives in
client/src/lib/air/flightTrackLayer.ts + trackModel.ts + cameraRig.ts +
components/MapNavCluster.tsx + FlightProfilePanel.tsx).
IMPLEMENTATION NOTE (2026-07-20 session, empirical probe): the prototype's
N/S pan buttons move opposite their own "Pan forward" tooltip and this
README's "up = away from camera" (E/W matches) — a prototype sign slip;
the site implements the documented intent. All other control signs were
probe-verified and match the prototype exactly. -->

# Handoff: 3D Flight Track — Terrain Curtain + Button Navigation

## Overview
Add two things to the existing VolTrade 3D map (Cesium-based terrain view), exactly as shown in the approved prototype `flight-path-3d.html`:

1. **Flight track visualization** — a 3D altitude line that follows the flight, a ground trace draped on the topography, and a translucent altitude "curtain" connecting them. The curtain must remain visible at ALL camera angles, including extreme near-horizontal tilt (this is the bug in the current site — the shaded part disappears at grazing angles).
2. **Button-driven navigation cluster** — compass dial + rotate/tilt/zoom/pan/reset buttons so the map can be moved 360° without mouse-dragging. This goes on the WHOLE site (every 3D map view).

**Do not redesign anything. Recreate exactly what the prototype shows.** The prototype uses simulated terrain and a simulated ADS-B track; on the site, wire the same visuals to the real terrain and the real ADS-B feed that already exist.

## About the Design Files
`flight-path-3d.html` is a **design reference built in HTML + three.js**. It is a working prototype, not production code to paste in. Recreate it inside the site's existing map stack (CesiumJS or equivalent) using its established patterns. The DOM/CSS for the UI panels (nav cluster, flight card, profile panel) CAN be lifted nearly verbatim — they are plain HTML/CSS overlays. The 3D geometry must be re-expressed in the site's engine (Cesium recipes below).

## Fidelity
**High-fidelity.** Colors, sizes, spacing, typography, control behavior, and animation rates are final. Match them exactly.

---

## THE CRITICAL FIX — why the curtain fails today and how the prototype fixes it

The curtain is one vertical ribbon of triangles: top edge = flight path at altitude, bottom edge = the ground. Three rules make it bulletproof (see the `THE CURTAIN` block in the prototype source):

1. **Drape the bottom edge.** Re-sample the terrain height under the track every ~120 m and place bottom vertices **40 m BELOW the terrain surface**. The strip then overlaps into the ground, so ridges never open gaps. Never use a constant bottom height (sea level / min altitude) — that is what breaks over mountains.
2. **Render both faces.** `side: THREE.DoubleSide` (three.js) / `cull: { enabled: false }` (Cesium renderState). A single-sided ribbon gets back-face-culled the moment the camera tilts past it — that is the "shade stops working at extreme angles" bug.
3. **Transparent geometry must not write depth.** `transparent: true, opacity 0.34, depthWrite: false, depthTest: true`, and draw AFTER the terrain (`renderOrder` in three.js; translucent pass handles this in Cesium with `depthMask: false`). This kills the z-fighting/vanishing at grazing angles.

The **ground trace** (line at the bottom following topography) is the same polyline draped at terrain + 16 m — in Cesium use a `GroundPolylinePrimitive` (clamp-to-ground), width ~3 px, color `#1e5fd6`.

The **altitude line** is the track at true altitude, colored by altitude (ramp below), rendered as a tube/thick polyline ~4 px.

### Cesium implementation sketch (site's engine)
```js
// positions: track lat/lon; alts: ADS-B altitudes (m MSL)
// 1) densify track to ~120 m spacing, then:
const ground = await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartos);
const minimumHeights = ground.map(c => c.height - 40);   // 40 m BELOW terrain
const maximumHeights = alts;

const curtain = new Cesium.Primitive({
  geometryInstances: new Cesium.GeometryInstance({
    geometry: new Cesium.WallGeometry({ positions, maximumHeights, minimumHeights })
  }),
  appearance: new Cesium.MaterialAppearance({
    translucent: true,
    closed: false,
    material: altitudeGradientMaterial,        // rgba, alpha 0.34
    renderState: {
      cull: { enabled: false },                // = DoubleSide  ← fix
      depthTest: { enabled: true },
      depthMask: false                         // = depthWrite:false ← fix
    }
  }),
  asynchronous: false
});
viewer.scene.primitives.add(curtain);

// ground trace — follows topography exactly:
new Cesium.GroundPolylinePrimitive({ /* clampToGround polyline, #1e5fd6, width 3 */ });
```
Rebuild `maximumHeights` as new ADS-B points arrive (append to the wall or rebuild the primitive; it is cheap at these vertex counts).

### Altitude color ramp (line + curtain top edge)
Map altitude min→max across the visible track:
- low `#38d1c1` (teal) → mid `#4da3ff` (blue) → high `#a06bff` (violet), linear in two halves.
- Curtain fill = same color at **34% opacity**, darkening toward the bottom edge (bottom vertex color = top × [0.25, 0.28, 0.45]).

---

## Screens / Views

### 1. Navigation cluster (site-wide, every 3D view)
Fixed to the right edge, vertically centered: `position: fixed; right: 16px; top: 16px; bottom: 16px; width: 144px;` column, `gap: 10px`, contents centered, above all other panels (z-index 12). Below 640 px viewport height: gap 6 px, hide the small labels, compass shrinks 92→72 px.

Panel container style (shared token, used by every card on the site):
- Background `rgba(9,15,27,.86)`, `backdrop-filter: blur(14px)`
- Border `1px solid rgba(130,170,230,.16)`, radius `14px`
- Shadow `0 8px 30px rgba(0,0,0,.45)`

Top to bottom:
1. **Compass dial** — 92×92 circle (panel style, fully round). Ring with 30° ticks (90° ticks longer), red **N**, dim S/E/W (`#8fa3bf`, 9 px), fixed white needle at top. The ring rotates to `-cameraHeading` so N always points to world north. **Drag anywhere = dial rotation** (grab angle → heading follows the delta, full 360°). **Click (no drag) = animate heading to north** (shortest way).
2. Label `ROTATE · TILT` — 9 px, letter-spacing 1.2 px, `#8fa3bf`.
3. **Button card** — 3 rows × 2 buttons: rotate-left / rotate-right (circular-arrow icons), tilt-up / tilt-down (chevron-to-line icons), zoom + / −. Buttons 38×38, radius 10, icon-only, background `rgba(120,160,220,.08)`, hover `rgba(77,163,255,.16)` + border `rgba(77,163,255,.4)`, pressed `rgba(77,163,255,.3)`.
4. Label `PAN`.
5. **Pan D-pad** — 3×3 grid of 38 px cells, ↑←→↓ arrows, center cell empty with dashed border `rgba(130,170,230,.25)`.
6. **RESET VIEW** — 82×38 pill-ish button, 11 px / 600 / letter-spacing 0.6, dim text brightening on hover.

**Behavior (exact rates — these are tuned):**
- All buttons are **hold-to-repeat** (pointerdown → act every frame until pointerup/cancel; use pointer capture).
- Rotate: heading ± **1.5 rad/s** (~86°/s). Tilt: pitch ± **0.9 rad/s**. Zoom: distance × **e^(∓1.6·dt)** (exponential). Pan: **0.9 × distance per second**, screen-relative (up = away from camera), and panning disables follow-aircraft.
- Camera is a **damped orbit rig**: state {target, heading, pitch, distance} each with a goal; every frame `value += (goal − value) × (1 − e^(−7·dt))`. In Cesium drive it with `camera.setView({ destination: target, orientation: { heading, pitch } })`-style per-frame updates or `lookAt` with `HeadingPitchRange`.
- Clamps: pitch **2°–88°** (2° = extreme tilt must work — the curtain must survive it), distance 900 m–70 km, target inside the data region; camera never below terrain + 120 m.
- Keyboard: **Q/E** rotate, **R/F** tilt, **arrows** pan, **+/−** zoom, **Space** play/pause. Mouse still works: left-drag rotate (0.005 rad/px, 0.004 rad/px), right- or shift-drag pan, wheel zoom (`× e^(0.0011·deltaY)`), double-click recenters target on the clicked terrain point.
- Hint bar top-center: `DRAG ROTATE · RIGHT-DRAG PAN · DBL-CLICK RECENTER · SPACE PLAY` — 9 px, in a `rgba(9,15,27,.5)` pill; hidden under 860 px width.

### 2. Flight card (top-left, per selected aircraft)
236 px wide panel: ✈ icon + callsign (17 px / 700), **ADS-B** live badge (10 px / 600, teal `#38d1c1`, teal border `rgba(56,209,193,.35)`, pulsing 6 px dot, radius 99, never wraps). 2×2 grid of stats: ALT MSL, ALT AGL, GND SPD, VERT SPD — label 10 px letter-spacing 1 px `#8fa3bf`, value 15 px / 600 mono with small unit. **Follow aircraft** toggle button full-width below: idle = dim outline; active = `rgba(77,163,255,.16)` bg, `#4da3ff` border/text. Follow keeps the camera target locked to the aircraft; user heading/tilt/zoom stay free.

### 3. Altitude / time profile (bottom bar)
Fixed bottom, `left: 16px; right: 176px` (clears the nav cluster). Header row: round play/pause (34 px, `#4da3ff` outline on `rgba(77,163,255,.16)`), title `ALTITUDE / TIME` + live UTC clock (mono), legend right-aligned (ALTITUDE blue line, TERRAIN green line, AGL BAND swatch).

Chart (~118 px tall, full width):
- **Terrain profile under the track**: filled `rgba(56,84,52,.55)`, stroke `#6b8f5e` 1.2 px.
- **Altitude line**: `#4da3ff`, 2 px.
- **AGL band** between them: `rgba(77,163,255,.16)` — this is the 2D twin of the 3D curtain.
- Gridlines every 5,000 ft, dashed `rgba(130,170,230,.12)`, mono labels `5k ft…`.
- **Playhead**: 1 px white line + white/blue dot riding the altitude line, synced both ways with the 3D aircraft position.
- **Scrub**: pointer down/drag anywhere on the chart sets time (pauses playback). Start/end UTC timestamps under the corners.

### 4. Aircraft marker + tag (in-scene)
- Blue aircraft glyph `#8fd0ff` at true position, nose along track tangent.
- Vertical **AGL drop line** from aircraft to ground (`#bfe0ff` at 65%) + soft ground dot.
- Floating tag above the aircraft (screen-projected DOM chip, panel style, radius 8): `✈ CALLSIGN` + current altitude in `#4da3ff` mono. Hide when behind the camera.

### 5. Info popover ("i" button, optional but included in prototype)
34 px round button under the flight card; toggles a 330 px panel listing the curtain rendering rules (content in prototype). Useful for QA; safe to omit in production.

## Interactions & Behavior (summary)
- Playback loops; Space toggles; scrubbing pauses. On the live site, "now" = latest ADS-B point; the scrubber replays history.
- Follow-aircraft: damped target tracking; auto-disabled by manual pan or double-click recenter.
- All transitions come from the damped rig (no CSS animation on the camera); button hover/active states are instant.

## State Management
- Camera rig: `{tx, tz, heading, pitch, dist}` + goals; damping factor 7/s; single rAF loop.
- Playback: `simT` (seconds into track), `playing`, `follow`.
- Track data: array of samples `{t, lat, lon, altMSL, terrainHeight, gs, vs}` — terrainHeight sampled once per point (`sampleTerrainMostDetailed`), reused by curtain, profile chart, and AGL readout.

## Design Tokens
- Panel bg `rgba(9,15,27,.86)` · border `rgba(130,170,230,.16)` · radius 14 · blur 14 · shadow `0 8px 30px rgba(0,0,0,.45)`
- Ink `#dfe8f5` · dim `#8fa3bf` · accent `#4da3ff` · accent-soft `rgba(77,163,255,.16)` · teal `#38d1c1` · violet `#a06bff`
- Trace blue `#1e5fd6` · aircraft `#8fd0ff` · terrain-profile green `#6b8f5e` / fill `rgba(56,84,52,.55)`
- Fonts: **Space Grotesk** (UI: 400–700), **JetBrains Mono** (all numeric readouts/timestamps). If the site already standardizes on other fonts, keep the site's fonts — everything else stays exact.
- Button: 38×38, radius 10; icons 14–16 px, stroke 2–2.4.

## Assets
None external. All icons are inline SVG strokes (copy from the prototype markup). Fonts from Google Fonts.

## Screenshots
`screenshots/` — captured from the prototype: default view, **extreme grazing tilt with the curtain still rendering** (02 — this is the acceptance test for the fix), top-down, mid-tilt, zoomed-in. UI panels shown are the compact (short-viewport) variant.

## Files
- `flight-path-3d.html` — the complete working prototype (single file). Key sections are commented: terrain heightfield (stand-in), flight track, **THE CURTAIN** (the fix), camera rig, hold-button wiring, compass dial, profile chart. UI markup/CSS is at the top of the file.
