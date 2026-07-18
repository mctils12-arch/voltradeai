# Space View — handoff notes (replaces current space feature on voltradeai)

Spec + working reference: `space-view.html` (open in browser; all logic in one inline module script). MIT/CC-BY assets in `images/` (NASA LRO 8k moon `moon_8k.jpg`, Solar System Scope 8k Milky Way `8k_stars_milky_way.jpg` — CC-BY 4.0 credit solarsystemscope.com, threex.planets 1k planet maps).

## What replaces what
Current site behavior: zoomed-out "gray blobs" space view → replace with this scene (three.js r184):
- Textured planets w/ real axial tilts, retrograde Venus/Uranus; shared sim clock: orbit + spin at true relative rates, scaled by `timeMult()` (slider "time ×").
- Positions: mean-element circular orbits, `L0 + 360·d/P` (d = days since J2000). Same accuracy class as your Schlyter/van Flandern note.
- Distance compression: `R(a)=60·a^exp`, exp lerps 1→0.42 via compression slider; `invR` maps back for HUD/labels (labels always show real AU — computed from real-AU positions, not display space).
- Orbit paths + 60° trailing motion arcs; moons draw theirs around their parent (line `position` follows planet each frame).
- Milky Way: 8k equirect on BackSide sphere + additive second pass (glow); fades in past 8 AU camera altitude (`clamp((camAU-8)/25,0,1)`).
- Click label → fly-to (1.1s ease) then **follow**: target tracks body each frame, camera translated by same delta; `controls.minDistance` + `camera.near` shrink to body size for full zoom-in.
- Body info card (#bodycard) on focus: day length, orbit period, distance, radius, tilt.
- Panel toggles/sliders persist (`localStorage['vt-space-proto']`).

## Integration points (your codebase)
1. **Earth slot = your live globe.** Replace the textured Earth mesh with your existing Earth model + whatever layers are active (sats, trains, plants render for free). The demo LEO/MEO/GEO shells (`earthSats`) are stand-ins — delete on integration.
2. **Camera/LOD handoff:** your map resumes on approach (your existing "live map resumes on approach" behavior) — trigger at ~0.02 AU camera altitude; hand camera lat/lon/alt to the map, reverse on zoom-out.
3. **HUD strings** already match your vocabulary ("at Earth · camera N AU out", LOD hints, RAW badges).

## Open todos
- Release tracking gesture (click empty space → back to Sun frame).
- Real sat catalog passthrough instead of demo shells.
- Optional: bloom postprocess + ESO/Gaia pano for richer galaxy; moons of Mars/Jupiter/Saturn.
