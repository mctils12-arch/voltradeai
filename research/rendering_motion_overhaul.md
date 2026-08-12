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
| 3 | Moon surface migration | **BLOCKED — re-scope required, see below** |
| 4 | Earth base (MapLibre) config | **SHIPPED** v1.0.678 (3 of 7 items not applicable) |
| 5 | Satellites (Law I) | **PREMISE STALE — verify before changing anything** |
| 6 | Aircraft trail / curtain (Law I) | **PREMISE NOT LOCALIZED — see below** |
| 7 | Radar (Law III) | not started |
| 8a | Law IV contract: budgets declared + `dispose()` on all 5 GL layers | **SHIPPED** v1.0.679 |
| 8b | Law IV runtime: bounded archive queries + wire `applyFeatureCap` | not started — next |
| 9 | Freshness (Law V) | not started |
| 10 | Self-see harness assertions | **MUST BE LAST** — would fail the build today |

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
