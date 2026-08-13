# RENDERING & MOTION OVERHAUL — program charter + resume state

Installed 2026-08-12 from the human's ten-PR work order and the companion
"Article: Rendering & Motion Law" (the article is now CLAUDE.md Amendment 6).
Multi-session program. This file is the RESUME STATE — read it before touching
any layer, and update it as PRs land.

TERRITORY: T-CLIENT throughout (`client/src/render/**`, `client/src/lib/**`
layer modules, `client/src/pages/datamap.tsx`). SHARED: package.json, research/*.

---

## THE DIAGNOSIS (the human's, and it is correct)

Moon tile fuzz, satellite pulsing, the glitching curtain behind the aircraft
and layer flicker on zoom are one bug appearing four times: **visual state is
driven by discrete events (`zoomend`, `moveend`, data ticks, tile arrival)
instead of by a continuous per-frame loop.** Each layer grew its own loading
and animation logic, so fixes overlap and re-break each other. The overhaul
builds one shared contract (`client/src/render/`) and migrates every layer
onto it.

---

## STATUS

| PR | Scope | State |
|----|-------|-------|
| 1 | frameCore + perfHud + zoomInput + the Law article | **SHIPPED** v1.0.676 |
| 2 | tileCore — the raster streamer | **SHIPPED** v1.0.677 |
| 3 | Moon surface migration | **RE-SCOPED into 3a/3b — see F17/F18** |
| 3a | Moon bake: our own CDN pyramid (Law II.8) | **PILOT SHIPPED** z0–z5, 2,730 tiles on R2 |
| 3b | Moon fuzz: cover-margin/LOD budget policy (Law II.3/II.4) | **SHIPPED** v1.0.690 — +0.90 levels avg, F19 |
| 4 | Earth base (MapLibre) config | **SHIPPED** v1.0.678 (3 of 7 items not applicable) |
| 5 | Satellites (Law I) | **PREMISE STALE — verify before changing anything** |
| 6 | Aircraft trail / curtain (Law I) | **OPEN — 6 causes eliminated, 2 candidates, see F16** |
| 7 | Radar (Law III) | not started |
| 8a | Law IV contract: budgets declared + `dispose()` on all 5 GL layers | **SHIPPED** v1.0.679 |
| 8b | Law IV runtime: caps armed + enforced | **SHIPPED** v1.0.680 (bounded archive queries still open → 8c) |
| 9 | Freshness (Law V) | **SHIPPED** v1.0.682 — OWM root cause found and fixed |
| 10a | Self-see STATIC assertions (Law I + Law IV + constants + hysteresis) | **SHIPPED** v1.0.681 — in CI |
| 10b | Self-see RUNTIME assertions (#2 unready draws, #3 p95, #4 heap) | blocked — see notes |

PRs 1 and 2 add NO runtime behavior. Nothing consumes `render/` yet except
PR4's `mapBaseConfig`. That is deliberate: the contract lands first, the
migrations follow one layer per PR so attribution survives.

---

## VERIFIED STACK FINDINGS — read these before believing the work order

The work order was written against assumptions that do not all match this
repo. Each of these was checked against the code, not inferred.

**F1. THERE IS NO THREE.JS.** The directive's raster sections use three.js
vocabulary throughout (`KTX2Loader`, `renderer.capabilities.getMaxAnisotropy()`,
`logarithmicDepthBuffer`, `generateMipmaps`, `LinearMipmapLinearFilter`). This
repo is MapLibre GL 5.24 plus hand-written WebGL2 custom layers
(`CustomLayerInterface`). The LAWS all still apply; the named APIs must be read
as "the equivalent capability" and implemented through raw GL
(`EXT_texture_filter_anisotropic`, `WEBGL_compressed_texture_*`). Recorded in
CLAUDE.md's STACK NOTE.

**F2. THE MOON IS NOT A GPU TILE PYRAMID — PR3 NEEDS RE-SCOPING.** The work
order says to "keep the existing `uTerminator` altitude-lighting shader" and
extend it with a two-sampler crossfade. `uTerminator` and `uFade` appear
NOWHERE in the repo. The actual moon path:
- `client/src/lib/celestial/moonTiles.ts` fetches LROC WAC tiles from NASA Trek
  **directly in the browser** and stitches them into ONE equirect RGBA mosaic
  capped at 2048² (`MOON_MOSAIC_MAX_PX`), evicted on zoom-out.
- `client/src/lib/celestial/moonSurface.ts` **CPU-ray-casts** the visible
  sphere patch in perspective and samples that mosaic per pixel.
So the moon has no per-tile GPU texture, no crossfade shader, and no LOD
quadtree — it has a single stitched raster and a software rasterizer. PR3 as
written cannot be applied. `tileCore` was deliberately built source- AND
sink-agnostic (`TileSink.upload` can mean "blit into the mosaic canvas" just as
well as "texImage2D") so the ready-gate, epoch guards, abort and eviction still
transplant — but the crossfade and the two-sampler material do not, and the
real question is whether the moon should become a GPU layer at all. Decide
that before writing PR3.
- Also note Law II.8: `moonTiles.ts` fetches from NASA Trek **at runtime**.
  That is an upstream WMTS in the live path. Fixing it requires the RunPod
  pyramid bake to exist first.

**F3. `setPrefetchZoomDelta` DOES NOT EXIST IN MAPLIBRE.** It is a Mapbox GL JS
API. Verified: zero occurrences of `prefetchZoomDelta` in the shipped
maplibre-gl 5.24.0 bundle. Calling it throws. Do not re-add it.

**F4. MAPLIBRE'S DEFAULT `wheelZoomRate` IS ALREADY 1/450.** Verified in the
bundle (`jr=1/450`). The work order's value matches the library default, so
`setWheelZoomRate(1/450)` is a no-op today. PR4 sets it anyway as a PIN, not a
fix, so the Earth base and the celestial views cannot drift apart silently.

**F5. `tileSize: 512` IS WRONG FOR THE CURRENT IMAGERY SOURCE.** The base is
Esri World_Imagery, which serves 256px tiles. Declaring 512 makes MapLibre
request one zoom level COARSER and upscale — a permanently blurrier base map,
the opposite of the intent. 512 becomes correct when the base moves to our own
baked 512px pyramid.

**F6. PR5's OCCLUSION CULLING IS ALREADY DONE.** The work order calls it "the
still-open occlusion culling". `client/src/lib/orbital/occlusion.ts` exists,
is unit-tested, exports `cameraFromClippingPlane`, is imported by
`satLayer.ts`, and the shader inlines its math with `satLayer.test.ts` pinning
the CPU mirror in sync. It is an analytic horizon test, exactly as the work
order specifies. Nothing to do.

**F7. PR5's PULSING ALREADY HAS A DOCUMENTED, TESTED FIX.** `satLayer.ts`
implements a GLIDE system: per-satellite velocity (`a_vel`) is uploaded and
the shader extrapolates position per frame from the last propagated state, so
positions are NOT snapped on arrival. `tickAnchorFromEpoch` /
`tickAnchorFromSimEpoch` are labelled "pulse fix" in-source and anchor the
glide clock to the worker's propagation epoch rather than message-arrival time
— the comment names the exact symptom ("its tick-to-tick JITTER rendered as
small backward snaps at close zooms"). Instance buffers already use
`bufferSubData` with `bufferData` only on reallocation.
  - The work order asks for a DIFFERENT architecture: render at
    `now - INTERP_DELAY_MS` and interpolate between two bracketing states.
    That trades zero-latency-with-tick-discontinuity for
    200ms-latency-with-no-discontinuity. For orbital motion over a ~2s tick,
    first-order velocity extrapolation is very accurate, so the existing
    choice is defensible.
  - **Therefore: do not rewrite this blind.** Per CLAUDE.md's RECURRENCE
    ESCALATES rule, re-patching an already-fixed subsystem is forbidden
    without a root cause. Next session must first CONFIRM ON LIVE whether
    pulsing still occurs at close zoom, and if so localize it, before choosing
    between "tune the glide" and "switch to delayed interpolation".

**F8. THE AIRCRAFT TRAIL IS ALREADY STATIC GEOMETRY.** `flightTrackLayer.ts`
builds its vertex/index buffers with `gl.bufferData(..., STATIC_DRAW)` and
registers NO map event handlers. The work order's stated root cause ("trail
geometry is rebuilt on camera events") is not true of that module. The curtain
glitch is real but is coming from somewhere else — candidates: the terrain
drape-order/RTT path (`installDrapeOrderGuard`, which re-floats layers on every
`styledata`), or one of datamap.tsx's own `move` handlers. **Localize before
writing PR6.**

**F9. NO LAYER EXPORTS THE PR8/PR10 CONTRACT.** *(RESOLVED by PR8a,
v1.0.679 — all five GL layers now export `maxFeatures`/`vramBudget` and
implement `dispose()`. PR10's static assertion #5 will now pass against
them. The finding text below is kept as the record of what was found.)* `dispose`, `maxFeatures` and
`vramBudget` appear in ZERO of satLayer / flightTrackLayer / airLayer /
arcLayer / modelLayer. They DO all implement MapLibre's `onRemove`, which is
the real teardown hook — so PR8 is largely about (a) declaring budgets that do
not exist at all and (b) naming a uniform contract over teardown that mostly
exists, not writing teardown from scratch. This is why **PR10's static
assertion #5 would fail the build against every layer today** and PR10 must
land last.

**F10. THE LAW I VIOLATIONS, ENUMERATED.** Ten handlers, all real:
`datamap.tsx` — `move` ×3 (4587, 6247, 6475), `moveend` ×3 (4633, 7534, 8083),
`zoomend` (8091), `pitch` (5829), `idle` (6477); `MapNavCluster.tsx` — `move`
(217). (Line numbers as of v1.0.675; they shift.) Each belongs to the layer PR
that owns it — bundling them into one sweep destroys attribution, which is
what CLAUDE.md promotion rule 5 forbids.

---

## THE ACCEPTANCE NUMBER IS NOWHERE NEAR MET

Visual harness on the /data page at v1.0.678: frame p95 **67ms @390, 183ms
@768, 217ms @1440**, against a 16.7ms target. That is headless SwiftShader
(software rasterizer), NOT a GPU, so it is not the Galaxy S24 number and must
not be reported as one. But it is the only continuous measurement available in
CI today, and it is 4-13× over budget. Establishing a real S24-class number is
itself work that PR10's assertion #3 depends on.

`?perf=1` (PR1) is the instrument for everything above. Use it before and
after every remaining PR — "the moon looks fuzzy" is not a measurement.

---

## RESUME ORDER (recommended, given the findings — see also F11 below,
## which upgrades the Law II.8 work from "blocked" to "tractable")

1. ~~**PR8 first, not last.**~~ **DONE (8a).** Budgets declared and derived,
   `dispose()` on every GL layer, the no-silent-caps downsampler built and
   tested. PR10's assertion #5 now has something to pass against.
   **NEXT IS 8b:** wire `applyFeatureCap()` into each layer's data path (the
   caps are declared and tested but NOT yet enforced at runtime), and convert
   the archive-backed layers (aircraft, vessels, trains, chokepoints) to
   viewport-bbox + time-window queries instead of unbounded selects. 8b is
   where the actual memory win lives — 8a made it declarable, not bounded.
2. **PR6 and PR5 need a localization session each** (F7, F8) before any code
   is written. Confirm the symptom on live with `?perf=1` on, then localize.
   Do not rewrite a documented, tested fix on the strength of a stale premise.
3. **PR3 needs a decision, not an implementation** (F2): does the moon become
   a GPU tile layer, or does tileCore's gate get transplanted onto the CPU
   mosaic path? File the decision in wishlist.md if it needs the human.
4. **PR7 and PR9** are independent of all of the above.
5. **PR10 last**, once 3/5/6/8 have removed the violations its static
   assertions would otherwise fail on.

## OUTSTANDING LAW VIOLATIONS ACCEPTED FOR NOW (tracked, not hidden)

- **Law II.8, twice.** The Earth base fetches Esri World_Imagery at runtime;
  the moon fetches NASA Trek at runtime.
- **Law IV**, every layer (F9).
- **Law I**, ten handlers (F10).

### CORRECTION 2026-08-12 (same session) — F11: THE CDN ALREADY EXISTS

My first pass through this file said Law II.8 "needs the RunPod bake + a CDN
before it can be fixed" and that pointing at a CDN "that does not exist" would
blank the surface. **That was wrong, and too pessimistic.** Verified:

- A **Cloudflare R2 bucket is already in production use.** `server/routes.ts`
  (~line 779) serves `/tiles-r2/:name`, streaming PMTiles range requests from
  `R2_PUBLIC_URL` (default `pub-4d65a892936747ada1c67a1f00e286c8.r2.dev`)
  through our own origin — added 2026-07-31 for the GRID VISION world rollout.
- That route already sets `cache-control: public, max-age=86400, immutable`
  and streams rather than buffers. The work order's CDN requirements
  (CORS-avoidance via same-origin, immutable cache headers, no domain
  sharding) are already satisfied by this path. Only `max-age` differs
  (86400 vs the work order's 31536000) and it is content-addressed enough to
  raise safely.
- The moon's base texture is ALREADY self-hosted: `client/public/space/
  moon_8k.jpg` (5.1MB) and `2k_moon.jpg`. So Law II.2's "pinned base level"
  already has its asset on our origin — it is the DETAIL tiles that go
  upstream to NASA Trek.
- `client/public/tiles/` and `client/public/imagery/sites/` exist as
  self-hosted asset roots.

**Consequence: Law II.8 is not blocked on infrastructure. It is blocked only
on BAKED CONTENT.** The bucket, the passthrough route, the cache headers and
the allowlist pattern all exist and are proven in production. Resolving Law
II.8 means (a) running the RunPod pyramid bake, (b) uploading to the existing
bucket, (c) widening the `/tiles-r2/` allowlist regex beyond `*.pmtiles`, and
(d) repointing the source URL. That is a tractable pipeline job, not a
prerequisite platform build.

This also revives the work order's `tileSize: 512` and `transformRequest`
items (F5, and PR4 item 3): both become correct the moment our own 512px
pyramid is in the bucket.

## BRANCH NOTE

All ten PRs are being developed as sequenced COMMITS on the single authorized
branch `claude/rendering-motion-overhaul-x0efmb` (only one branch was
authorized for this work). Each commit is one logical change with its own
message, so commit-level attribution is preserved even though they arrive as
one pull request rather than ten.

---

## SESSION 2 ADDENDUM (2026-08-12) — PRs 8a/8b/9/10a shipped

**F12. THE SATELLITE CAP WAS A LEVER NOBODY PULLED.** `satLayer.setRenderCap`
has existed since O1 with full honest accounting (`getCounts()` reports
capped/rendered/total). A grep for callers across the entire client returns
NOTHING — it defaulted to `null` and the layer rendered the whole worker
output. PR8b armed it with the declared `maxFeatures`. Reused rather than
rebuilt, per READ BEFORE WRITE.

**F13. THE OPENWEATHERMAP NON-RENDER, ROOT-CAUSED.** Not upstream, not the
key. `setProjection()` (globe/flat toggle) rebuilds the MapLibre style, which
wipes imperatively-added sources and layers. The weather effect's dep array
excludes anything a style rebuild changes, and nothing listened for
`styledata` — so the sole recovery path was a 10-minute `setInterval`. Toggle
the globe with weather on → the layer is gone for up to ten minutes while the
panel reads "active". **A retry cannot fix this because the retry IS the
ten-minute timer.** Fixed by reconciling presence on `styledata`.
  - Second bug found while fixing it: the guard was `if (!map.getSource(...))`
    — SOURCE only. A half-built pair (source present, layer missing) read as
    healthy forever, drew nothing, and was never repaired.

**F14. THE LAW I VIOLATIONS ARE NOT IN THE LAYERS.** All ten live in
`datamap.tsx` and `MapNavCluster.tsx`, which are pages/components, not layer
modules. `client/src/lib/{orbital,air,celestial}` has ZERO. This is what made
PR10a's static assertion shippable immediately rather than last.

**F15. CI DOES NOT RUN THE CLIENT TESTS OR THE VISUAL HARNESS.** `ci.yml`
runs the python pytest set, `tsc --noEmit || true`, `npm run build`, and the
docker build. So `client/src/**/*.test.ts` and `npm run visual` are
session-time gates, NOT merge gates. That is why PR10a's assertions were
placed in `test_audit_critical.py` (which CI does run) rather than in the
visual harness. Any future assertion intended as a MERGE GATE must go
somewhere CI executes.

### What 10b still needs
- **#2 (no unready node drawn)** — not yet meaningful: no layer consumes
  tileCore, so there are no nodes to audit. Lands with the first migration.
- **#3 (p95 < 16.7ms)** — cannot be honestly gated on headless SwiftShader,
  which runs 4-13× over budget for reasons unrelated to our code. Needs a
  real S24-class measurement path; that is its own piece of work.
- **#4 (heap/texture return to baseline)** — needs forced GC
  (`--js-flags=--expose-gc`) in the visual harness.

### Remaining queue, in recommended order
1. **8c** — bounded archive queries (aircraft/vessels/trains/chokepoints
   query by viewport bbox + time window instead of unbounded selects). The
   remaining half of the memory story; a client-fetch + server-route change.
2. **Trail decimation** — `MIN_POINT_SPACING_M` with `tzMarkIdx` remapped
   alongside, done where the track is ASSEMBLED. PR8b deliberately only
   REPORTS the over-cap trail because decimating inside `buildTrackVertices`
   would desync the v1.0.671 timezone marks.
3. **The moon bake → PR3** (F2 + F11) — unblocks the last Law II.8 violation
   and revives `tileSize: 512` + `transformRequest`.
4. **PR7 radar** — premise unverified.
5. **PRs 5/6** — need live confirmation first (F6/F7/F8).
6. **10b** once the above land.


---

## F16 — THE CURTAIN CUT-OFF: SIX ELIMINATIONS, NO FIX (2026-08-12)

The human supplied live evidence: an ARCHIVED flight (complete data), where
the altitude/time chart draws the FULL profile while the map trail cuts off
at a fixed distance. Same data, two renderers, one truncates — **this is a
render-side truncation, not missing data.** Do not re-investigate the data
path.

### Eliminated, with the evidence — do not re-derive these

| # | Hypothesis | Killed by |
|---|-----------|-----------|
| 1 | "Trail rebuilt on camera events" (the work order's stated cause) | `flightTrackLayer` builds `gl.bufferData(..., STATIC_DRAW)` once and registers ZERO map handlers |
| 2 | Array-length mismatch in `n = Math.min(merc>>1, altM, groundZ)` | `merc`, `altDisp` and `groundZ` are all allocated `Float32Array(n)` in datamap.tsx (~3988-4046) |
| 3 | 16-bit index overflow past ~16k segments | indices are `Uint32Array`, drawn with `gl.UNSIGNED_INT` |
| 4 | `TRACK_MAX_SAMPLES = 6000` truncating the route | it sets DENSIFICATION SPACING (`total / (MAX - fixes.length)`), so the whole route stays covered, just coarser |
| 5 | Vertex-buffer overflow silently dropping the tail (typed-array writes past `length` are silent no-ops) | counted exactly: the 4 emission groups write `((n-1)*3 + marks) * 4` vertices = exactly `maxSegs * FT_VERTS_PER_SEG`, and `out.slice(0, o)` trims correctly |
| 6 | Data truncation | an archived flight has complete samples, and the chart proves it |

### The two survivors

**(a) Segment-skip guards** in `buildTrackVertices` — `wrapOk` (skips any
segment whose normalized-mercator Δx > 0.5) and the three
`Number.isNaN(altM[i]) continue` statements. **Leading candidate:** the
human's test flight is labelled *SIGNAL LOST AIRBORNE*, which is exactly the
condition that produces a NaN altitude run. Note all three emission groups
(curtain, altitude line, marks) skip on NaN while the GROUND TRACE does not
("draws through altitude gaps") — so if this is it, the ground trace should
continue past the cut-off while the curtain and altitude line stop. **That
asymmetry is itself a diagnostic: look at whether the thin ground line
survives past where the curtain dies.**

**(b) Globe horizon / occlusion culling** in the vertex shader
(`mercatorToSphere` / `mercatorZFromAltitude` from `orbital/occlusion`).

### The discriminating test (cheap, needs a human at the map)

Rotate the globe and watch the cut-off point.
- **Moves with the camera** → (b), horizon culling.
- **Stays pinned to the same ground position** → (a), a segment skip.

Second, independent check: does the thin ground trace continue past the point
where the curtain stops? If yes → the NaN guard, conclusively.

### Separate real bug found while investigating (NOT the cut-off)

`altScale` is referenced but UNDEFINED at `datamap.tsx:4261` (×2) and `:4298`
— `tsc` TS2304, previously dismissed as pre-existing noise. It sits in the
**AGL readout**, not the trail geometry, so it does not cause the cut-off,
but the flight card's AGL number is computed from an undefined variable.
Deserves its own PR.

---

## F17 — THE MOON FUZZ IS A BUDGET-POLICY BUG, NOT A TILE-SOURCE BUG (2026-08-13)

**MEASURED, not reasoned.** Ran the real `planMoonTarget` (moonTiles.ts) via
`npx tsx` across realistic disc sizes and surface spans, with the real
`MOON_PATCH_COVER_MARGIN = 2.8` from spaceFrame.ts:634:

```
discPx  spanDeg  pxPerDeg  idealZ  chosenZ  lost  mosaic      visible%
 400      20        20        4       4       0   1536x1536     36%
 800      20        40        5       4       1   1536x1536     36%
1200      20        60        6       4       2   1536x1536     36%
1600      60        27        5       2       3   1280x1024     36%
2000      20       100        7       4       3   1536x1536     36%
MOON_MOSAIC_MAX_PX = 2048
```

The Moon renders **1–3 zoom levels below what the screen deserves — 2× to 8×
under-resolved** — on every viewport bigger than ~400px of disc. That is the
fuzz, and the mechanism is arithmetic, not mysterious:

1. `spaceFrame` asks for a span **2.8× wider than the visible disc**
   (`MOON_PATCH_COVER_MARGIN`), i.e. **7.8× the area**.
2. `planMoonTarget` then *backs off whole zoom levels* until that inflated
   mosaic fits `MOON_MOSAIC_MAX_PX = 2048`.
3. So the visible 36% of the span pays for the invisible 64%, in sharpness.

Two things make it worse than the numbers suggest:

- **The budget is never even spent.** The chosen mosaics land at 1536px and
  1280px against a 2048px cap: the planner drops a whole level the moment the
  next step would exceed the cap. That is a hysteresis-free cliff — exactly
  what Law II.3 forbids ("split above 2px, merge below 1px, hysteresis is
  mandatory"). A ~25% budget headroom is left unused while the image is 2×
  soft.
- **It is resolution-regressive.** Bigger screen ⇒ larger `pxPerDeg` ⇒ larger
  `idealZ` ⇒ *more* levels lost. The 2000px case loses 3. The Moon gets
  blurrier the better your display is.

**Law II.4 already prescribes the fix**: prefetch is a *bounded ring around
the target*, not a blanket span multiplier. Keeping the visible region at
native resolution and adding a one-tile ring costs ~1 tile per edge instead
of 6.8× the area.

### What this means for PR 3 — and an honest correction

The moon bake (below) does **NOT** fix this. The bake is a Law II.8 item
(stop hitting an upstream WMTS at runtime); the fuzz is a Law II.3/II.4
budget-policy item in `planMoonTarget` + `MOON_PATCH_COVER_MARGIN`. They are
independent, and shipping the bake alone would have left the fuzz exactly
where it is. Recording that explicitly because the pre-bake assumption in
this program was that better tiles would sharpen the Moon; measurement says
otherwise.

The margin change is a RULE REVIEW–class threshold change: one constant at a
time, prior value logged, rollback trigger stated. It is NOT bundled with the
bake.

---

## F18 — MOON BAKE: OUR OWN CDN PYRAMID, PILOT SHIPPED (2026-08-13)

Law II.8 ("Runtime never touches an upstream WMTS. All tiles come from our
CDN. Upstream is a bake-time input only."). `lroc.ts` points the browser at
`trek.nasa.gov` on every close Moon frame — a live third-party runtime
dependency we cannot cache, pre-warm, or survive an outage of.

**`scripts/moon_bake.py`** (+ `test_moon_bake.py`, 19 tests) bakes the same
imagery from the ORIGINAL USGS source into our R2 bucket.

### Source — probed live, nothing assumed

| field | value |
|---|---|
| url | `planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` |
| bytes | 5,959,263,751 (5.55 GiB), `Accept-Ranges: bytes` (302 → S3) |
| raster | 109164 × 54582, Byte, 1 band, **uncompressed** |
| blocks | **109164 × 1 — one-scanline strips, NO overviews** |
| licence | **Public Domain** (USGS/NASA) |

The strip layout is why the script downloads once and re-reads locally:
windowed reads cost full scanlines, so only a sequential pass is efficient.
Download measured 13.3 MB/s ⇒ ~7.5 min, one time.

**Licensing resolved by construction.** Baking from USGS PDS (public domain)
rather than re-hosting Trek tiles sidesteps the redistribution question
entirely — re-serving someone else's tiles from our CDN is a stronger claim
than the runtime hotlinking we do today, and we simply do not need to make it.

### Resolution honesty

Trek's product is "303 ppd" = 109,080 px ⇒ the SAME 100 m/px source. So
**Trek's deepest level (z8) is already an upsample**, not extra real detail.
Our native ceiling is **z7** (65536px wide, a true downsample). The manifest
flags any level above that `"upsampled": true`, and a test pins it — the
product can never advertise resolution the data does not contain.

### Scheme — deliberately identical to Trek EQ

`MatrixWidth = 2^(z+1)`, `MatrixHeight = 2^z`, 256px, plate carrée,
TopLeftCorner (−180,+90), key order **`{z}/{y}/{x}`** (row before column).
Matching exactly means `lroc.ts`'s already-tested math is reused verbatim and
the switch is a `baseUrl` change. `test_moon_bake.py` **parses lroc.ts** and
asserts the TypeScript still says what the Python assumes — this is a
cross-language contract that otherwise fails silently at runtime (404s / black
Moon), never in CI.

### Pilot result — $0, no RunPod needed

```
z0..z5   2,730 tiles   53 MB   build 2m   upload 42s
verified: tiles.json 200, 0/0/0 200, 5/0/0 200, 5/31/63 200, z6 404
visual:   8 baked z1 tiles restitched → coherent globe, no seams,
          near-side maria centred on lon 0, polar illumination gaps preserved
```

The full native set (z0–z7) is **43,690 tiles**; z0–z5 measured ~19 KB/tile,
so the full set extrapolates to roughly 0.8 GB and a few hours of local CPU.
**RunPod is not obviously justified** — the pilot ran entirely in-session on 4
cores, and the bottleneck is the one-time 5.5 GB download, which a pod does
not remove. Measure the z6/z7 pass locally before spending from the $43.12
ledger.

### BLOCKER for wiring it up: no CORS on r2.dev

Probed 2026-08-13: `pub-*.r2.dev` returns **no `access-control-allow-origin`
header**, which is why the pmtiles go through the `/tiles-r2/` same-origin
passthrough in `server/routes.ts:779`. Moon tiles are read into a canvas
(`crossOrigin="anonymous"` + `createImageBitmap`), so they need CORS. Two
paths: (a) extend the same-origin passthrough — works today, no human action,
costs a Railway hop; (b) set a bucket CORS policy — better, needs the human
in the Cloudflare dashboard. Filed in wishlist.md; (a) is the default.

---

## F19 — 3b SHIPPED: margin is now spent before resolution (2026-08-13)

Fix for F17. `planMoonTarget` now gives back **prefetch margin first** and a
**zoom level only when the minimum span still will not fit**. Rationale:
resolution is what the user sees; the cover margin is only an optimisation.

`spanLadder(want, must, z)` (new, exported, pure, tested) walks from the
desired span down to `must + degPerTile(z)` — the visible region plus a
**one-tile pan ring**, which is exactly Law II.4's prescription — and always
ends on that floor so the cheapest option is definitely tried.

### Measured, same matrix as F17

```
discPx  span  idealZ  beforeZ  afterZ  lostB  lostA   mosaicBefore  mosaicAfter
 400     20     4        4        4      0      0     1536x1536     1536x1536
 400     60     3        2        3      1      0     1280x1024     2048x2048
 800     20     5        4        5      1      0     1536x1536     2048x2048
 800     60     4        2        3      2      1     1280x1024     2048x2048
1200     20     6        4        5      2      1     1536x1536     2048x2048
1600     60     5        2        3      3      2     1280x1024     2048x2048
2000     20     7        4        5      3      2     1536x1536     2048x2048
```

**9 levels recovered across 10 cases (avg +0.90 ⇒ ~1.9× sharper).** Nine of
ten improved; the tenth was already optimal. No case regressed — pinned by a
test that asserts `after.z >= before.z` across the whole matrix.

### The VRAM trade, stated honestly

Mosaics move from 5.2–9.4 MB to 16.8 MB. That is **not new budget** — it is
the budget the code already declared and was failing to spend
(`MOON_MOSAIC_MAX_PX = 2048`, whose own comment reads "2048² RGBA ≈ 16 MB,
under the ~4096 mobile texture cap"). The old planner dropped a whole level
rather than approach its own cap; that unused ~25% headroom while the image
was 2× soft is precisely the hysteresis-free cliff Law II.3 forbids.

### Residual

Some cases are still 1–2 levels short of ideal. That remainder is bounded by
the 2048px mosaic cap itself and cannot be recovered by planning — closing it
needs per-tile GPU textures (a real tileCore migration) rather than one
stitched mosaic. Not claimed as fixed.

### Safety property (the only way this could be a bug)

If a shrunk span ever failed to cover the visible disc the user would see an
untextured edge — strictly worse than fuzz. Pinned by a test asserting the
returned window covers the visible lon/lat extent across disc sizes, spans,
and latitudes including near-polar clamping.

### Backward compatible

With `minHalfSpanDeg` absent the floor equals the desired span, the ladder
collapses to one rung, and behaviour is byte-identical. Only the one opted-in
call site in `spaceFrame` changes. Pinned by a test.

---

## F20 — THE VISUAL HARNESS HAS NO CELESTIAL COVERAGE (2026-08-13)

Found while trying to satisfy PROMOTION RULES item 6 for the 3b change.
`scripts/visual_check.mjs` `PAGES` covers `/data` (map + dashboards) and the
marketing pages — **there is no space/celestial/Moon scenario at all**. So the
harness cannot see the Moon, and a Moon rendering change cannot be visually
verified by it. The 3b change was therefore verified by unit tests (including
the coverage-safety property) plus the measured before/after matrix, NOT by
screenshots — stating that plainly rather than letting a green harness run
imply coverage it does not have.

Adding a celestial scenario is the obvious ratchet (the same "every new view
passes the harness" rule the PAGES comments cite for streams/quality/signals)
and is queued in open_questions.md.

**Also observed, pre-existing:** the harness is FLAKY — one run reported
1 hard failure, three consecutive re-runs reported 0, with no code change
between them. And `data` p95 frame time measures **250ms @1440, 183ms @768,
100ms @390**, against the Law's 16.7ms. Both are separate items; the p95 gap
is already recorded under "THE ACCEPTANCE NUMBER IS NOWHERE NEAR MET".

---

## F21 — THE CAMERA COULD NOT CROSS A POLE, AND WHY (2026-08-13, human report)

Human: *"when looking at the moon it will not let you go over the axis of
rotation like over the poles same thing on the earth ... I came across this
issue when looking at the bottom of the moon to see the missions"* — plus a
Google Earth screenshot as the quality bar.

### Root cause — one line of vector math

`camBasis(dir, upRef)` (spaceFrame.ts:910) derives the camera's up-vector from
a CONSTANT world axis every frame:

```ts
let r = cross(f, upRef);
if (len3(r) < 1e-6) r = cross(f, /* arbitrary fallback */);
const u = norm3(cross(r, f));
```

As the view direction approaches the axis, `cross(f, upRef) → 0`: the basis is
DEGENERATE at the pole and snaps to an arbitrary fallback orientation. Rather
than remove that singularity, the code fenced it off with a polar clamp — and
the fence, not the math, is what the user hit:

| state | polar range allowed | consequence |
|---|---|---|
| lock ON (**default**) | 0.12 … π/2+0.42 rad = **6.88° … 114.06°** | the entire under-side of every body is unreachable |
| lock OFF | 0.05 … π−0.05 rad = **2.86° … 177.14°** | can approach a pole, can never cross it |

So on the DEFAULT setting the camera stops 66° short of the Moon's south pole.
The landing sites the user was trying to inspect (Chang'e 4/6, LCROSS, IM-1,
Chandrayaan-3) are in exactly that unreachable cap. The doc comment claiming
unlock "only WIDENS" was accurate but beside the point — both states forbid the
crossing.

### Fix — carry the up-vector instead of deriving it

Stop rebuilding `up` from a world constant; PARALLEL-TRANSPORT it along the
view's own rotation and re-orthonormalise. That is continuous everywhere,
including at the pole, so the singularity is gone rather than fenced:

- `orthonormalUp(dir, up)` — Gram-Schmidt, never NaN
- `transportUp(dirPrev, dirNext, upPrev)` — minimal-rotation transport
- `levelBlend(dir, axis)` — smoothstep, 1 away from the axis → 0 within
  `LEVEL_FADE_RAD` (0.35 rad) of EITHER pole, because "level" is UNDEFINED at
  a pole; that is the honest statement of the singularity
- `relevelUp(dir, up, axis, t)` — ease the carried up back toward level
- `camBasisFromUp(dir, up)` — basis from an explicit up, no degeneracy

`setDir()` is now the ONLY place `dir` changes, so orbit drags AND discrete
re-anchors (flight arrival, focus release, re-focus) all carry the up through
the change — none of them can snap the horizon.

**The polar clamp is deleted outright**, along with `polarClampDots` and its
four constants. There is nothing left to clamp.

### `lockHorizon` re-scoped, not removed

It no longer bounds WHERE the camera may go. ON (default) = keep the horizon
level wherever level is defined, releasing only inside the small polar cap; OFF
= never re-level (free roll). Both allow crossing. This preserves the human's
2026-07-19 "level, not tilted/wonky" directive for normal viewing while
satisfying the new one.

### A test that pinned the bug was REMOVED — deliberately, and stated

`"polarClampDots: lock forbids under-ecliptic, unlock only widens"` asserted
`"poles excluded even unlocked"`. That assertion encoded the defect, so it
could not survive the fix. Per REPAIRS MUST RATCHET it was replaced by five
stronger tests, not merely dropped — notably a **400-step sweep that walks the
camera over the pole and asserts the up-vector never jumps by >0.05** (the old
basis snapped there), and orthonormality asserted AT `(0,0,±1)` exactly — the
input that used to hit the degenerate fallback.

285 celestial tests pass. Zero new tsc errors (83 before / 83 after).

### Still open — the Earth half of the same report

MapLibre's globe is a separate camera we do not own; its polar behaviour and
the pinwheel artifact are F22.

---

## F22 — THE ARCTIC PINWHEEL: MERCATOR DATA STRETCHED OVER THE GLOBE'S POLE CAP

The fan of triangular wedges radiating from the north pole in the user's
screenshot is the **seafloor depth tint**.

`seafloor-dem` (datamap.tsx:5038) is a **Web Mercator** terrarium raster-dem:

```
tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"]
```

Web Mercator tiles exist only to **±85.051129°**. Above that there is no data
at all. MapLibre's globe projection triangulates the residual polar cap as a
FAN of triangles meeting at the pole, and the `color-relief` ramp interpolates
the topmost tile row's values across those wedges — producing precisely the
pinwheel. The same limit applies to every Mercator raster source we stack
(terrain DEM, GIBS imagery, radar, OWM).

This is **not** a bug in the ramp or the palette: it is invented data being
painted in a region where none exists, which the honesty rails forbid
regardless of how it looks. Fix in the next commit: declare the sources' real
latitude `bounds` so nothing paints above 85.051129° — a gap shown as a gap.
Real polar coverage needs a polar-stereographic source (NASA GIBS publishes
EPSG:3413 Arctic / EPSG:3031 Antarctic matrix sets); filed as follow-up.
