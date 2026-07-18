# VolTradeAI — Celestial v2: Integrated Solar System

The v1 celestial implementation is rejected in its current form. Two failures: (1) it shipped as a separate "Solar system" mode/button instead of being part of the main map, and (2) TRUE SCALE mode made the page unresponsive (Chrome "Page Unresponsive" kill dialog during load at `/app#/data`). Rebuild per this spec. Spawn subagents in parallel where dependencies allow; no run caps; batch into larger PRs within the existing PR-size cap; fix regressions before new work; follow CLAUDE.md governance.

Keep from v1: real ephemeris (Schlyter/van Flandern is acceptable; upgrade to `astronomy-engine` if accuracy or code clarity improves), the "markers flag bodies smaller than a pixel" concept, real moon phase, and the sun bloom/flare work.

---

## 1 — One scene, one camera. Remove the separate mode.

- Delete the "Solar system" button and any separate page/route/mode. There is exactly ONE map view and ONE camera.
- The existing zoom controls (buttons, scroll, pinch) extend continuously: street level → country → globe → Earth orbit → cislunar space (Moon visible in orbit) → inner planets → full solar system. Zooming back in reverses seamlessly. No mode switch, no transition screen, no camera teleport.
- Engineering requirement for this to work without jitter/precision artifacts: camera-relative (floating-origin) rendering with double-precision origin offsets on CPU, single-precision only for camera-relative GPU coords. Earth-surface layers (satellites, aircraft, vessels, all existing layers) keep rendering correctly at every zoom and fade by relevance (e.g., aircraft fade out beyond ~X000 mi camera distance) rather than popping.
- The on-screen scale bar (currently "10000 mi") extends through the full range: mi → thousands of mi → AU, respecting the existing mi/km units toggle.

## 2 — Scale system (the core of this rebuild)

True scale alone is useless for viewing (bodies are sub-pixel) and is what v1 shipped. Replace with USER-CONTROLLED scale, defaulting to a visible configuration:

- **Distance compression slider:** continuous from `True (1:1)` to `Compressed`. Compressed uses a logarithmic/power mapping of heliocentric and planetocentric distances so all planets fit in view while preserving ordering and relative sense of spacing. Slider updates layout live at 60fps (precompute in worker, interpolate on GPU).
- **Body size slider:** `1× (true)` up to `~2500×`, scaling rendered body diameters so planets are visible at solar-system zoom. Sun capped separately so it never swallows the inner system at high exaggeration.
- **Presets:** two buttons — `TRUE SCALE` (both sliders to 1:1; markers + labels carry the view since bodies go sub-pixel) and `VISIBLE` (sensible defaults, the startup state).
- **Honest labels, always:** every body label shows its REAL current distance (e.g., `Moon · 355,341 mi`, `Venus · 0.908 AU`) computed from ephemeris, regardless of visual compression. Panel description states plainly, in the existing amber-mono style: `distances/sizes compressed for visibility — labels always show real values`. The layout may compress; the numbers never lie.
- Scale state persists in the URL hash/localStorage like other layer state.

## 3 — Orbits, rotation, and moons

- **Orbital paths toggle:** full orbit ellipses for Mercury–Neptune (correct orbital elements, drawn consistently in whatever compression the slider is set to), the Moon's orbit around Earth, and orbits for a curated set of major moons: Io, Europa, Ganymede, Callisto (Jupiter), Titan (Saturn), Triton (Neptune), Phobos + Deimos (Mars). Paths are polylines sampled in a worker, cached, restyled — never resampled — on slider moves.
- **Moons rendered as bodies:** each curated moon is a real object at its ephemeris position orbiting its planet, subject to the same size/distance scaling, with marker + label when sub-pixel.
- **Axial rotation:** every planet + the Moon rendered with correct axial tilt and rotating at its true rate (Moon tidally locked — always same face to Earth). 
- **Time controls:** a compact time-rate control in the celestial panel: `1× (now)`, `60×`, `3600×`, `1 day/s`, `pause`, and `⟲ now` reset. Time rate drives EVERYTHING consistently from one simulation clock: orbital positions, rotations, Earth's terminator, moon phase, satellite propagation epoch. At 1× the view always tracks real time.

## 4 — Map the Moon for real

- Lunar surface from the NASA LRO **LROC WAC global mosaic** (public domain, ~100 m/px) served as a proper tile pyramid, so zooming to the Moon works exactly like zooming to Earth — maria and craters resolve progressively.
- **LOLA elevation** for relief shading (displacement or normal map at close zoom).
- Lighting is sun-driven: correct terminator, phase, and earthshine hint on the dark limb; approximate libration so the visible face wobbles realistically over the month.
- Optional at close zoom: labels for major maria and a handful of famous craters (Tycho, Copernicus, Kepler). Low priority.

## 5 — Planet surfaces

- NASA/USGS public-domain texture maps for each planet (and the curated moons) at modest base resolution; higher-res loaded ONLY when the camera approaches that body (see §6). Saturn gets rings (translucent, correctly tilted). Sun keeps v1's limb darkening + bloom/flare on zoom.

## 6 — Fix the freeze (root-cause, not band-aid)

The v1 hang is a release blocker. Requirements:

- **Diagnose first:** profile the current build, identify the exact long task(s) that trigger "Page Unresponsive" (likely synchronous texture decode, giant geometry build, or an unchunked ephemeris/orbit precompute on the main thread). State the root cause in the PR description with profiler evidence.
- **Rules after the fix:**
  - No main-thread task > 16 ms during load or interaction. All ephemeris, orbit sampling, and layout precompute in Web Workers with transferable buffers.
  - Progressive, lazy asset loading: at startup load NOTHING celestial beyond sun/moon low-res + positions. Planet textures, moon tiles, and orbit geometry stream on demand (camera proximity), decoded off-thread (`createImageBitmap` / KTX2 compressed textures), with strict GPU-memory budget and eviction.
  - App interactive < 2 s on desktop; celestial layer can never block other layers or input. Add a long-task watchdog (PerformanceObserver) that logs offenders in dev.
  - Verify on a mid-range Android (Galaxy S24 class): full zoom-out to solar system and back with satellites + 2 live layers on, no hitch > 100 ms, no crash, no unresponsive dialog. Record before/after profiler numbers in the PR.

## 7 — UI integration

- All celestial controls live in the existing LAYERS panel as a `CELESTIAL` section styled exactly like current sections (RAW pill, amber mono descriptions, same toggles/sliders): orbital paths toggle, distance slider, size slider, presets, time-rate control.
- The `moderate load` indicator in the panel header must account for celestial cost honestly.
- Existing zoom in/out buttons operate across the entire range; nothing about the current Earth-layer UX regresses.

## 8 — Universal realistic lighting & shadows (toggleable)

One physically consistent light source — the Sun — drives the ENTIRE scene at every zoom, exactly like the existing ISS "computed ephemeris view" already does for one craft. Extend that treatment universe-wide:

- Day/night terminator on every body: Earth (already present — keep), the Moon, every planet, every curated moon. Phases are not painted on; they emerge from the sun direction.
- Shadow casting where it matters: moons darken in their planet's shadow and vice versa (eclipses happen at the correct times), Saturn's rings shadow the planet and the planet shadows the rings, spacecraft go dark passing through Earth's umbra (with a dim penumbra transition).
- Spacecraft/3D models in inspect views: lit by the real sun direction with self-shadowing, earthshine fill only when over the sunlit Earth — the ISS view's existing behavior becomes the standard for all craft.
- Optional polish: city lights on Earth's night side (NASA Black Marble), low priority.
- **Toggle:** `Realistic lighting` in the CELESTIAL panel section, ON by default. OFF = uniform full-lit mode (every body and craft evenly illuminated for inspection/visibility). Same RAW-pill/mono styling as every other toggle; description states what it does in one line.
- All of it obeys the simulation clock from §3 — scrub time and eclipses, terminators, and phases move together.

## Acceptance

1. No "Solar system" button exists; one continuous camera from street level to Neptune and back, smooth, no jitter.
2. Default view on zoom-out shows all planets visible (compressed + enlarged), each labeled with its true current distance.
3. TRUE SCALE preset works without hanging: bodies collapse to markers, page stays responsive.
4. Orbit paths, planet rotation, Moon + curated moons orbiting, and time controls all function and stay mutually consistent (terminator, phases, positions all agree at any time rate).
5. Zooming to the Moon resolves real LROC surface detail with correct lighting.
6. Zero "Page Unresponsive" dialogs through a scripted stress pass (documented in PR); root cause of the v1 hang identified and fixed with profiler evidence.
7. All other layers unaffected at every zoom; CI green; before/after perf numbers in the PR.
8. Realistic lighting toggle works: ON shows correct terminators/phases/eclipse shadows everywhere and craft darkening in Earth's umbra; OFF renders everything evenly lit; both states hold at any time rate.
