# ORBITAL — DESIGN-CLASS MODEL LIBRARY (research, workstream (c))

Status: RESEARCH ARTIFACT. No code, no assets produced. This doc is the
honest feasibility plan a future O5 (design-class glTF LOD) / O6 (splat
pipeline) build session executes. Scope owner: T-CLIENT. Charter:
`research/orbital_program.md` — "MASTER-BUILD EXTENSION: FIDELITY BY
DESIGN CLASS", "THE LOD REALITY", "GAUSSIAN SPLAT — HONEST SCOPE".

Discipline being applied (from the charter, verbatim intent): model each
satellite DESIGN / GENERATION **once** from real **published GROUND
imagery/specs** (photos taken on Earth before launch — display units,
press kits, spec sheets), then GPU-instance that one real model across
every constellation member at its true SGP4 position. NEVER photograph a
satellite in orbit; NEVER invent a model to fill a gap; where no usable
ground reference exists → an honest typed SYMBOLIC marker.

---

## 0. HEADLINE VERDICT (read this first)

- **Splat is essentially a non-starter from existing free imagery.**
  A real browser Gaussian splat needs dozens–hundreds of consistent
  multi-angle GROUND photographs of one physical article. Satellites do
  not have that publicly: marquee designs (Starlink, GPS III, OneWeb)
  have only a handful of press photos, and the two most-photographed
  unique craft (ISS, Hubble) are orbit-only objects whose multi-angle
  imagery is sparse, space-based, and inconsistently lit. **True splat
  candidates from existing public imagery: 0.** The only designs where a
  ground article even physically exists to photograph are a Hubble
  full-scale mockup and a handful of museum test articles — and capturing
  those is our own photogrammetry trip, which is out of scope and not
  worth it. **Recommendation: glTF everywhere; splats deferred
  indefinitely (drop O6, or gate it behind a museum-capture trip the
  human explicitly funds).**
- **glTF is the right call for every design worth a model,** and it gets
  ~90% of the "premium per-design detail" value (the charter already
  says this). ~20 real design models cover the DESIGNS behind the large
  majority of tracked *payloads* because a few mega-constellations
  dominate the count (Starlink alone is ~75% of active payloads).
- **The honest coverage boundary:** a large majority of tracked
  *satellites* resolve to a real design model; the remainder — the long
  tail of one-off/unique payloads, all rocket bodies, and all debris —
  render as honest typed markers. Verbatim sentence in §5.

Tier counts for the inventory below: **SPLAT-CANDIDATE: 0** ·
**GLTF-3D: 18** · **SYMBOLIC-ONLY / SYMBOLIC-GENERIC: 5 typed classes**
(covering thousands of long-tail payloads, all rocket bodies, all debris).

---

## 1. DESIGN INVENTORY (prioritized by member-count × imagery availability)

Counts are mid-2026 on-orbit figures (sources §7); they drift with every
launch, so the build reads live counts from SATCAT — these are for
prioritization only. "Members covered" is why we model the design once.

Priority tiers: **P0** = one model covers thousands; **P1** = hundreds;
**P2** = tens or a unique marquee craft; **P3** = generic fallbacks.

| # | Design / generation | ~On-orbit members | Ground-imagery source | Fidelity tier | Priority |
|---|---|---|---|---|---|
| 1 | **Starlink v2 Mini** (+ v2 Mini "Optimized") | bulk of ~10.7k Starlink | SpaceX press kit photos, flat-pack/deploy imagery, Gen2 spec PDF | GLTF-3D | **P0** |
| 2 | **Starlink v1.5** (legacy, deorbiting) | declining share of ~10.7k | SpaceX press photos, published dimensions | GLTF-3D | **P0** |
| 3 | **Starlink v3** (emerging, Gen2 large) | small but growing | SpaceX Gen2 spec sheet + early press imagery | GLTF-3D | P1 (as members grow) |
| 4 | **OneWeb** (Gen-1) | 648 | Airbus/OneWeb Satellites press photos + spec sheet | GLTF-3D | **P0** |
| 5 | **Planet SuperDove** (Dove / PlanetScope, 3U cubesat) | ~200 | Planet Labs press photos (simple 3U form) | GLTF-3D | **P0** |
| 6 | **Iridium NEXT** | 66 + ~9 spares | Thales Alenia Space press photos + spec | GLTF-3D | P1 |
| 7 | **GPS III / IIIF** | ~10 (III block completed 2026, SV10) | Lockheed Martin press renders + published spec | GLTF-3D | P1 |
| 8 | **GPS legacy (IIR / IIR-M / IIF)** | ~20 combined | Boeing / Lockheed press images | GLTF-3D | P1 |
| 9 | **Planet SkySat** | ~21 | Planet / Maxar (SSL) press photos | GLTF-3D | P2 |
| 10 | **Globalstar (Gen-2)** | ~48 | Thales Alenia press images | GLTF-3D | P2 |
| 11 | **ISS** (unique) | 1 | **NASA 3D Resources** glTF (authoritative) | GLTF-3D | P2 (marquee) |
| 12 | **Hubble Space Telescope** (unique) | 1 | **NASA 3D Resources** glTF (authoritative) | GLTF-3D | P2 (marquee) |
| 13 | **Sentinel-2** (2A / 2B / 2C) | 3 | ESA press imagery/renders, ESA SciFleet, Sketchfab CC | GLTF-3D | P2 (cross-tie (a)) |
| 14 | **Landsat 8 / 9** | 2 | NASA/USGS imagery; NASA 3D Resources (EO buses) | GLTF-3D | P2 (cross-tie (a)) |
| 15 | **GOES-R series** (16 / 18 / 19) | 3–4 | NOAA/NASA/Lockheed press imagery | GLTF-3D | P2 |
| 16 | **NASA EOS bus** (Terra / Aqua / Aura) | 3 | **NASA 3D Resources** glTF (each modeled) | GLTF-3D | P2 |
| 17 | **TDRS** (relay) | ~6 | **NASA 3D Resources** glTF | GLTF-3D | P3 |
| 18 | **Generic GEO comms bus** (one-off comsats) | hundreds (long tail) | representative bus from public GEO-comsat photos, **labeled generic** | GLTF-3D (generic) | P3 |
| 19 | **Generic CubeSat** (1U / 3U / 6U typed) | hundreds of smallsats | **NASA 3D Resources** CubeSat models (typed by U-size) | GLTF-3D (generic) | P3 |
| 20 | **Generic rocket body** | thousands (full catalog) | generic cylinder mesh, **explicitly labeled generic**, sized by RCS class | SYMBOLIC-GENERIC | P3 |
| 21 | **Generic debris fragment** | largest class (full catalog) | none — unknowable shape | SYMBOLIC-ONLY | P3 |
| 22 | **Unknown / classified payload** (USA-xxx, NROL) | hundreds | none published | SYMBOLIC-ONLY | P3 |
| 23 | **Long-tail unique payloads** (science/gov one-offs w/o a model) | thousands (aggregate) | none reliably reusable | SYMBOLIC-ONLY | P3 |

Notes on prioritization:
- Rows 1–5 are the whole game: Starlink generations + OneWeb + Planet
  SuperDove cover the overwhelming majority of tracked payloads with ~5
  models. Build these first.
- Rows 11–17 are marquee/EO craft: low member counts but high brand and
  cross-tie value (Sentinel-2 / Landsat feed the imagery-overpass tie
  (a)). NASA 3D Resources already ships glTF for many of them, so cost is
  near-zero — do them early precisely because they are cheap and real.
- Rows 20–23 are honest fallbacks, not designs. A generic rocket-body
  cylinder is allowed ONLY if labeled generic ("representative rocket
  body, not the specific stage"). Debris and unknown payloads never get a
  fabricated mesh — a typed marker only.

---

## 2. FIDELITY TIER RATIONALE (why each lands where it does)

**GLTF-3D** — a modeled mesh is honest because a real ground reference
exists (published photos + a spec sheet with true dimensions, or an
existing authoritative model like NASA's). The model is labeled
"representative design model, from [source]" — it is the design, not a
photograph of the specific on-orbit unit. This is the honest, correct
tier for essentially every design worth detailing:
- Starlink v1.5 / v2 Mini / v3, OneWeb, Iridium NEXT, GPS III/IIIF,
  Globalstar, Planet SuperDove/SkySat: manufacturers (SpaceX, Airbus,
  Thales Alenia, Lockheed Martin, Planet) publish press photos + spec
  sheets. Enough to model the design faithfully; **not** enough for a
  splat.
- ISS, Hubble, Terra/Aqua/Aura, TDRS, and several EO buses: **NASA 3D
  Resources already provides authoritative glTF** — no modeling needed,
  just license-clear reuse (§4).
- Sentinel-2, Landsat, GOES: agency press renders + community CC models
  exist; commission or license-clear an existing one.

**SYMBOLIC-ONLY** — no usable ground imagery of the specific object, so
any mesh would be fabrication. Debris fragments (shape genuinely
unknown), classified/unknown payloads, and the long tail of one-off
craft for which no reusable reference exists. Rendered as a typed marker
(color/glyph by SATCAT object type: payload / rocket-body / debris),
sized by RCS class where SATCAT gives one. This is the charter's
"UNPHOTOGRAPHED → honest symbolic marker" rule.

**SYMBOLIC-GENERIC** — a single generic mesh (e.g., a rocket-body
cylinder, a generic GEO-comsat bus) used as a typed representative,
**explicitly labeled generic**, never claimed to be the specific unit.
Borderline case: acceptable only with the label; when in doubt, drop to
SYMBOLIC-ONLY.

**SPLAT-CANDIDATE — none.** See §3.

---

## 3. GAUSSIAN-SPLAT FEASIBILITY — the honest reality check

### What a browser splat actually requires
1. **Capture:** dozens–hundreds of overlapping GROUND photos of ONE
   physical article, consistent lighting, full angular coverage (all
   sides + top/bottom). A handful of press shots from 2–3 angles is
   nowhere near enough — the reconstruction is holes and floaters.
2. **Offline processing:** structure-from-motion / camera-pose solve
   (COLMAP or equivalent) → Gaussian-splat training (gsplat / Nerfstudio
   / Postshot) → a `.ply` / `.splat` / `.spz` / `.sog` asset. Hours of
   GPU time, done offline. **We cannot do this** (no photogrammetry per
   task scope, and no source imagery to feed it anyway).
3. **In-browser renderer:** a real WebGL/WebGPU dependency, used ONLY in
   the focus/detail view — never for the 10k population.

### Which designs realistically have enough public multi-angle GROUND imagery
Being skeptical, as instructed:
- **Starlink** (the charter's own splat hope): publicly there are
  flat-pack stack photos and in-orbit deployment shots — a few angles of
  a folded panel, not a 360° ground survey of a deployed unit. **Not
  splat-viable.** glTF from the press photos + Gen2 spec is the honest
  call. This corrects the charter's speculation directly.
- **ISS / Hubble:** the most-imaged craft in history, but the imagery is
  (a) from spacecraft fly-arounds in orbit, not the ground, (b) sparse in
  angle, and (c) lit by raw sun with no consistent baseline — hostile to
  SfM. A splat would be partial and ugly, and NASA already ships a clean
  authoritative glTF. **glTF, not splat.**
- **Everything else** (OneWeb, GPS, Iridium, Planet, EO/GOES): a few
  manufacturer PR images each. **Not remotely splat-viable.**

### The only theoretical splat path (and why we still say no)
A genuine splat would require a **physical ground article we photograph
ourselves** — e.g., the full-scale Hubble structural mockup, a museum
test article at Udvar-Hazy, or an engineering model. That is a
capture-trip project (travel, a rig, offline training), explicitly out of
this workstream's scope and low-value versus a glTF that looks ~as good
in a small detail-view canvas. **Do not schedule O6.** If it is ever
revisited, it is exactly one design (Hubble mockup), human-funded, and
gated behind its own imagery-sourcing spike.

### The asset-size argument that also kills it
Even if imagery existed, a single splat scene is typically **20–150 MB**,
versus **~0.5–5 MB** for a glTF of the same object. For a
mobile-flawless-at-390px product (PREMIUM EXPERIENCE STANDARD), shipping
a 50 MB+ asset into the detail view is a non-starter on its own.

### In-browser splat renderer options (for the record, if ever needed)
- **Spark** (`@sparkjsdev/spark`) — MIT. three.js-based, fuses splats
  with meshes in one scene, targets 98%+ WebGL2, supports
  `.ply/.spz/.splat/.ksplat/.sog`. The actively-developed choice (World
  Labs + community). Renderer JS is a modest add (order 100s of KB); the
  **asset** is the real cost, not the lib.
- **@mkkellogg/GaussianSplats3D** — the original three.js splat renderer;
  author now points users to Spark and has wound down active development.
  Do not adopt for new work.
- **PlayCanvas SuperSplat / supersplat-viewer** — MIT; excellent
  editor/viewer but built on the PlayCanvas engine, i.e. a second engine
  alongside three.js — heavier integration cost for our stack. Use its
  CLI (`SplatTransform`) for asset prep only if we ever produce a splat.

**Splat verdict, stated plainly:** 0 designs are true splat candidates
today. glTF is the right call for 100% of the model library. Spark is the
renderer to reach for *only* if a human ever funds a Hubble-mockup
capture; otherwise this line item is closed.

---

## 4. THE MODELING PIPELINE PLAN (glTF tier — we NEVER fabricate)

Every model is sourced from a real reference and carries its provenance.
Three sourcing routes, in cost order:

### Route A — reuse authoritative free models (do this first)
**NASA 3D Resources** (`nasa3d.arc.nasa.gov`, mirrored at
`science.nasa.gov/3d-resources` and `github.com/nasa/NASA-3D-Resources`).
- **License (verified):** the repo README states verbatim, *"These assets
  are free and without copyright,"* and directs users to NASA's media
  usage guidelines (nasa.gov brand center). NASA-authored works are
  generally public domain (17 U.S.C. §105). **Honest caveats to enforce
  per asset:** (1) a few repository models are third-party contributions —
  check each model's own readme for any contributor restriction before
  use; (2) NASA media guidelines forbid using the NASA insignia/logo or
  implying NASA endorsement — fine for us, we're depicting the hardware,
  not badging our product with NASA marks. Net: usable in our monetized
  product with attribution + per-asset check.
- **Formats offered:** `3ds, blend, fbx, glb, jpg, lwo, max, maya,
  openvsp, stl, tif` — `.glb` (glTF) is available for the modern models;
  older ones may need a Blender → glTF export.
- **Confirmed relevant models present:** ISS, Hubble (A/B), Chandra,
  Fermi, Aqua/Aura/Terra (EOS), CloudSat, GRACE, ICESat, EO-1, TDRS-class
  relays, Cassini, Dawn, Galileo, DSN dishes, CubeSats, plus many science
  buses. Landsat/GOES/GPS/Sentinel/Starlink are **not** guaranteed there
  (commercial / non-NASA craft) → Route B/C for those.
- Action: enumerate the repo, pull `.glb` where present, verify each
  model's readme, log the source URL + license per asset in the build PR.

**ESA** — ESA SciFleet (`scifleet.esa.int`) offers ESA science-craft
models (largely `.stl` for 3D printing; usable as a modeling base) and
GSSC's Galileo-in-3D. ESA imagery/models carry ESA-specific terms — check
per asset; do not assume public domain the way NASA is.

### Route B — license a community model (verify commercial rights)
**Sketchfab / CGTrader / TurboSquid** have Sentinel-2, Iridium, OneWeb,
GPS, Starlink community models. **Hard licensing rule (ties to our
MONETIZATION TRIPWIRE — billing is enabled):** our product is monetized,
so any asset MUST be commercial-use-licensed — **public domain / CC0 /
CC-BY / a purchased commercial license only.** **Disqualified:** CC-BY-NC,
CC-BY-ND, and "editorial use only" assets (much of TurboSquid). CC-BY
requires visible attribution — surface it in the model's provenance chip.
Record the license + author + URL per asset in the PR.

### Route C — commission a model from published references
For designs with no clean free/licensed model (likely Starlink
generations, OneWeb, GPS III, Iridium NEXT, GOES): commission a modeler
to build a low-poly glTF **strictly from published manufacturer photos +
spec-sheet dimensions**, delivered with a source manifest (which photos /
which spec doc drove the geometry). This keeps "modeled from real
reference, never fabricated" auditable. Budget/vendor selection → file in
`wishlist.md` if it needs spend (build-first rule: reuse/license before
commissioning).

### Provenance & honesty on every model (PREMIUM EXPERIENCE STANDARD)
Each model's detail-view chip states: source (e.g., "Lockheed Martin GPS
III press renders + published spec sheet" / "NASA 3D Resources, public
domain"), license, and a fidelity caveat: *"Representative design model —
the design flown, not a photograph of this specific on-orbit unit."*
Attitude/rotation is driven by real orbital data; if true attitude is
unknown, the spin is labeled illustrative, not measured.

### Asset hygiene
Low-poly, Draco-compressed `.glb`, target ≤ ~2–5 MB per model; one model
per design, GPU-instanced across members at their SGP4 positions (far
view stays points; §6). Model library lives with the client territory
(T-CLIENT), no import from trading logic (SPINOUT-READY DATA LAYER rule).

---

## 5. COVERAGE BOUNDARY STATEMENT (verbatim-ready for the layer info)

> **Fidelity coverage.** A small number of real, ground-referenced design
> models — about twenty — cover the large majority of the satellites you
> see here, because a handful of mega-constellations (Starlink, OneWeb,
> Planet, Iridium, GPS) account for most tracked spacecraft, and every
> member of a constellation genuinely shares one design. Those models are
> built from published ground photos and spec sheets, or from public-domain
> agency models (e.g. NASA 3D Resources for the ISS and Hubble); each is
> labeled with its source and marked as the design flown, not a photograph
> of that specific unit in orbit. Everything else — the long tail of
> one-of-a-kind payloads, all spent rocket bodies, and all debris — is
> shown as an honest typed marker (payload / rocket body / debris),
> because no real ground imagery of those objects exists and we never
> invent a model to fill a gap. No object is hidden: it either resolves to
> a real design model on focus, or it is an honestly-labeled marker.

Quantified honestly (for internal use; live counts drive the real UI):
- By *object count*: a large majority of active **payloads** resolve to a
  real model (Starlink ~75% alone; Starlink + OneWeb + Planet + Iridium +
  GPS ≈ 90%+ of active payloads).
- By *distinct design*: ~20 modeled designs are a tiny fraction of the
  2,000+ designs ever flown — so most *kinds* of craft, and effectively
  all rocket bodies and debris, are symbolic. Both facts are true at once;
  the boundary statement above says so plainly.
- The full ~25k+ catalog (Space-Track, debris-heavy, gated) is
  overwhelmingly rocket bodies + debris → symbolic. The clean CelesTrak
  active set we ship first is payload-dominated, which is *why* model
  coverage of that set is high.

---

## 6. DEPENDENCY + SIZE BUDGET

**Detail models load ONLY in the focus/detail view — never for the 10k
population.** The globe/far view stays GPU points/billboards (the O1
render path). This is the charter's LOD reality and the hard performance
rule; the model library never touches the whole-sky frame budget.

- **glTF loader:** three.js `GLTFLoader` + `DRACOLoader`. If the globe is
  a MapLibre CustomLayer (O1 decision pending), the cleanest architecture
  is a **separate small three.js canvas for the detail view** — one model,
  lit, rotating on live orbital data — decoupled from the globe renderer.
  Budget: three.js core ~150 KB gzipped; Draco decoder ~200 KB WASM,
  **lazy-loaded on first focus** (never on landing). glTF assets ~0.5–5 MB
  each, **fetched one at a time on focus**, then cached. Nothing 3D-model
  loads until a user clicks a satellite → zero cost to the population view
  and to first paint.
- **Splat renderer (Spark, MIT):** NOT adopted (§3). If ever needed,
  three.js-compatible; renderer JS is a modest add but **splat assets are
  20–150 MB each** → disqualifying for a mobile-flawless-at-390px product.
- **Mobile (390px):** the population-as-points view is the mobile
  experience; the detail canvas renders a single ≤5 MB glTF on demand,
  which a phone handles fine. Guardrails: cap concurrent detail models at
  1 (the focused object), dispose GLTF scenes on defocus, keep Draco lazy,
  and never instance detailed meshes into the far view.
- **Net new dependency footprint:** three.js (if not already present) +
  Draco decoder, both lazy. No splat dependency. No per-model runtime cost
  until focus.

---

## 7. SOURCES (verified this session)

- NASA 3D Resources — model list, formats, and license ("These assets are
  free and without copyright"): https://github.com/nasa/NASA-3D-Resources
  · https://science.nasa.gov/3d-resources/ · https://www.nasa.gov/3d-resources/
- NASA 3D Models: International Space Station (data.gov catalog entry):
  https://catalog.data.gov/dataset/nasa-3d-models-international-space-station
- Spark splat renderer (MIT, three.js, formats, 98% WebGL2):
  https://github.com/sparkjsdev/spark
- @mkkellogg/GaussianSplats3D (wound down, points to Spark):
  https://github.com/mkkellogg/GaussianSplats3D
- PlayCanvas SuperSplat / viewer (MIT, PlayCanvas engine):
  https://github.com/playcanvas/supersplat ·
  https://github.com/playcanvas/supersplat-viewer
- Starlink on-orbit count (~10.7k active, mid-2026, McDowell):
  https://www.space.com/spacex-starlink-satellites.html ·
  https://planet4589.org/space/con/star/stats.html
- OneWeb (648): https://en.wikipedia.org/wiki/OneWeb_satellite_constellation
- Iridium NEXT (66 + 9 spares):
  https://en.wikipedia.org/wiki/Iridium_satellite_constellation ·
  https://www.eoportal.org/satellite-missions/iridium-next
- GPS III / IIIF (block completed 2026, SV10):
  https://en.wikipedia.org/wiki/GPS_Block_III ·
  https://news.lockheedmartin.com/2026-04-21-Lockheed-Martin-Launches-GPS-III-Satellite
- Planet Labs fleet (~200 SuperDove + ~21 SkySat):
  https://orbitalradar.com/satellites/operator/planet-labs
- Sentinel-2 (2A/2B/2C): https://space.skyrocket.de/doc_sdat/sentinel-2.htm ·
  https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2
- Landsat 8/9 + GOES-R (16/18/19 operational):
  https://www.nesdis.noaa.gov/our-satellites/currently-flying/goes-east-west/goes-r-series-transition-operations
- ESA SciFleet spacecraft models / Galileo-3D: https://scifleet.esa.int/ ·
  https://gssc.esa.int/education/galileo3d/

---

## 8. HANDOFF NOTES for the build session

- Build order within O5: rows 1–5 first (Starlink generations, OneWeb,
  Planet SuperDove) — max coverage per model — then the cheap NASA-3D
  marquee/EO craft (ISS, Hubble, Terra/Aqua/Aura, TDRS, Landsat proxy),
  then Iridium/GPS/GOES, then the generic fallbacks (rows 18–23).
- **Drop / do-not-schedule O6 (splat)** unless the human funds a
  museum-capture trip; if raised, it is one design (Hubble mockup) behind
  its own imagery-sourcing spike. Record the closure in experiments.md.
- Every model ships with a provenance manifest (source URL, license,
  reference photos/spec) and a UI provenance chip; CC-BY needs visible
  attribution; CC-BY-NC / editorial-only assets are disqualified because
  billing is enabled (MONETIZATION TRIPWIRE).
- Detail models are lazy, focus-only, ≤5 MB, Draco-compressed; the 10k
  population never leaves the GPU-points path.
