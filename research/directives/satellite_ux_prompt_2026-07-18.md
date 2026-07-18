# VolTradeAI — Satellite UX & UI Fixes

Issues observed live at `/app#/data` (screen recording + screenshots). Fix all of them. Spawn subagents in parallel; no run caps; batch into larger PRs within the existing PR-size cap; CI green; fix any regressions you cause before new work; CLAUDE.md governance applies.

---

## 1 — Satellite info card: streamline it, never cover controls, never overflow

Current behavior: clicking a satellite opens a tall, dense card listing every orbital element. It covers the LAYERS panel and right-side controls, and its content runs off the bottom of the viewport.

- **Compact by default.** Header: name, NORAD ID, operator/country, status dot. Below it ONE row of four stat chips: `ALT · SPEED · INCLINATION · PERIOD`. That is the whole default card.
- Everything else (apogee, perigee, eccentricity, RAAN, arg of perigee, mean anomaly, epoch, TLE age, launch date, etc.) moves behind a `Details ▾` expander. Expanded content scrolls INSIDE the card — hard max-height ~60vh — and can never extend past the viewport.
- **Placement:** anchor the card on the LEFT side of the map (or dock it so it provably never overlaps the LAYERS panel, zoom controls, or map-style switcher at any viewport ≥ 1280px). On mobile widths it becomes a bottom sheet with a drag handle.
- Action buttons (Inspect etc.) always visible without scrolling. Dismiss via ✕, tap-away, and Esc.
- Same treatment pattern applies to every other clickable layer's popup (see §5).

## 2 — Satellite motion: kill the pulse

The recording clearly shows rhythmic pulsing/glitching in satellite motion — positions visibly snap at the propagation tick instead of flowing.

- Root-cause it with a profiler capture (most likely: keyframes applied at ~1 Hz with no per-frame interpolation, and/or GC hitches from per-tick allocation, and/or a blocking buffer swap).
- Fix per the main sprint prompt W2 architecture: worker-pool SGP4 producing keyframes into transferable buffers, per-frame interpolation on the render side, zero allocation in the render loop, reused buffers.
- Verification: motion is continuous and smooth at multiple zooms with the full constellation on; profiler shows no periodic main-thread spikes; state the root cause + before/after evidence in the PR.

## 3 — Inspect mode: real follow-camera with two views

Current behavior: clicking Inspect, the satellite leaves the frame entirely — you end up looking at empty space.

Rebuild Inspect as a camera ATTACHED to the live, moving satellite. Time does not stop; the sat keeps flying its real orbit over the rotating Earth in both views.

- **Orbit view (default):** camera orbits the craft itself — drag to maneuver all the way around it, scroll/pinch to zoom from close-up structural detail out to the craft small against the Earth limb. Exactly the camera mechanics of the existing ISS "COMPUTED EPHEMERIS VIEW" screen — that screen is the reference implementation. Use a real 3D model where one exists (ISS, Hubble, etc.); otherwise a quality generic bus-and-solar-panel model scaled to the craft's approximate size class.
- **Onboard view:** camera positioned AT the satellite looking out — free-look in every direction (drag), zoom toward whatever you're facing: down to Earth's surface detail, out to the Moon, the Sun, other satellites, the stars. Like standing on the station.
- Clean toggle between the two views inside the inspect UI; `Back to map` restores the exact prior map camera.
- **Lighting in both views:** real sun direction with self-shadowing on the craft, earthshine fill only when over the sunlit Earth, craft goes dark in Earth's umbra — the ISS view's existing lighting is the standard, and it honors the global `Realistic lighting` toggle from the celestial v2 prompt.
- The info card (§1) stays available in inspect without covering the view.

## 4 — Layers panel scrollbar: theme it, edge it

The panel currently shows the default white browser scrollbar, inset from the screen edge — it clashes with everything.

- Custom scrollbar matching the card: thin (~6px), track the same color as the panel background, thumb a subtle blue-gray that brightens on hover. Implement with `scrollbar-width`/`scrollbar-color` plus `::-webkit-scrollbar` rules for Chromium.
- Move the panel flush to the right edge of the viewport so the scrollbar rides the actual screen edge.
- Define it once as a shared style and apply to EVERY scrollable surface in the app — layers panel, sat card details, streams inventory, any list or drawer. No native-looking scrollbar anywhere.

## 5 — Full-view audit (same classes of issue, everywhere)

Sweep the entire `/data` view and fix everything in these classes in this same pass:

- Any popup/info card that can cover controls or overflow the viewport — click-test every interactive layer: aircraft, vessels, trains, fires, US power plants, nuclear facilities, military installations, strategic sites, ports, SEC filings, earthquakes/whatever else is clickable. Apply the §1 pattern (compact + expander + internal scroll + safe placement + mobile bottom sheet).
- Any remaining default/unthemed scrollbar or native-looking control.
- Any layer whose motion snaps/pulses instead of interpolating (check aircraft, vessels, trains against §2's standard).
- Overlapping labels or UI collisions at common zooms; anything clipped at 1920×1080 or on mobile widths.
- List everything found and fixed in the PR description with before/after screenshots.

## Acceptance

1. Sat card: default state is header + one chip row; details expand and scroll internally; card never overlaps LAYERS panel/zoom/style controls at ≥1280px; bottom sheet on mobile.
2. Satellite motion shows zero rhythmic pulsing at any zoom; root cause documented with profiler evidence.
3. Inspect: craft always in frame, orbitable and zoomable exactly like the ISS reference view; onboard view free-looks and zooms to Earth/Moon/anything; both track the live moving sat; lighting matches the ISS-view standard.
4. No native scrollbars anywhere; panel scrollbar sits at the true screen edge and matches the theme.
5. Audit results in the PR with before/after screenshots for every popup layer.
