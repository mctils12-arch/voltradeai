# CELESTIAL V2 — Integrated Solar System (multi-session program charter)

Installed 2026-07-18 from the verbatim human directive
research/directives/celestial_v2_prompt_2026-07-18.md (authoritative —
this charter sequences it; CLAUDE.md governs HOW everything ships).
Companion directive: satellite_ux_prompt_2026-07-18.md (same upload).

## VERDICT ON V1 (honest record)

spaceFrame v1 (#528) delivered the continuity principle (one camera,
live-map Earth anchor, true scale, markers) but is REJECTED in current
form: (1) the ☉ chip still reads as a separate mode — it must go;
(2) a "Page Unresponsive" freeze at /app#/data load is a RELEASE
BLOCKER (agent on it with profiler-evidence mandate, 2026-07-18);
(3) true scale ALONE is not enough — the human wants USER-CONTROLLED
scale (see B2) with honest labels, superseding the fixed-true-scale
decision recorded in the v1.0.396 experiments entry. Keep from v1:
real ephemeris, marker concept, moon phase, sun bloom.

## BUILD ORDER (each slice = own PR chain; directive §-refs)

- B0 FREEZE FIX (§6) — BLOCKER, in flight. No new celestial work
  ships until load is clean (<16ms tasks, interactive <2s, watchdog).
- B1 ONE CAMERA, NO BUTTON (§1) — delete the ☉ chip; zoom controls/
  scroll/pinch extend continuously street→solar system; floating-
  origin camera-relative rendering (f64 origin on CPU, f32 camera-
  relative GPU); scale bar mi→AU through the units toggle; existing
  layers fade by relevance, never pop.
- B2 SCALE SYSTEM (§2) — distance-compression slider (True 1:1 ↔
  Compressed, log/power mapping, worker precompute + GPU interpolate),
  body-size slider (1×–~2500×, Sun capped separately), presets TRUE
  SCALE / VISIBLE (default), labels ALWAYS show real ephemeris
  distances, state persists (URL hash/localStorage). The layout may
  compress; the numbers never lie.
- B3 ORBITS/ROTATION/MOONS/TIME (§3) — orbit ellipses Mercury–Neptune
  + Moon + curated moons (Io Europa Ganymede Callisto Titan Triton
  Phobos Deimos) as real ephemeris bodies; axial tilt + true rotation
  (Moon tidally locked); ONE simulation clock (1×/60×/3600×/1day-s/
  pause/⟲now) driving positions, rotations, terminator, phases, sat
  propagation epoch together.
- B4 MOON FOR REAL (§4) — LROC WAC global mosaic tile pyramid (public
  domain ~100m/px) as a mapAnchor (the v1 registry slot exists for
  exactly this), LOLA relief, sun-driven lighting + libration;
  optional mare/crater labels (low priority).
- B5 PLANET SURFACES (§5) — NASA/USGS PD textures, modest base res,
  proximity-streamed higher res; Saturn rings; Sun keeps v1 flare.
- B6 UNIVERSAL LIGHTING (§8) — one Sun lights everything at every
  zoom: terminators/phases from geometry on every body, eclipse and
  ring shadows at correct times, craft dark in Earth's umbra,
  "Realistic lighting" toggle (ON default; OFF = even-lit inspection
  mode); all under the B3 clock. Optional Black Marble night lights.
- B7 UI INTEGRATION (§7) — CELESTIAL section in the LAYERS panel,
  existing styling (RAW pill, amber mono), honest load indicator.
- SAT-UX §3 INSPECT FOLLOW-CAMERA (companion directive) — orbit view
  (camera attached to the live moving craft, drag-orbit, zoom close↔
  Earth-limb; retired-scene mechanics as reference, but time never
  stops and the world is the live one) + onboard view (free-look from
  the craft); lighting per B6 standard; Back-to-map restores exact
  prior camera. Sequenced after B0 + pulse fix land.

## ACCEPTANCE = the directive's own §Acceptance list, verbatim.

## PERF RULES (standing, from §6)
No main-thread task >16ms at load or interaction; workers +
transferable buffers for ephemeris/orbit/layout; progressive lazy
assets (nothing celestial beyond low-res sun/moon + positions at
startup); GPU memory budget + eviction; long-task watchdog in dev;
mid-range mobile verification with before/after profiler numbers in
every PR.

## RESUME STATE (newest first)
- 2026-07-18 installed. B0 agent out (freeze root cause). Companion
  sat-UX agents out (card system per design doc; motion pulse).
  Nothing of B1+ built. The satellite UI design doc lives at
  research/directives/satellite_ui_design_2026-07-18.dc.html
  (7 screens, from the human's Claude Design project via DesignSync).
