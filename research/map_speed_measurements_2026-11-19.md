# /data map speed — measurements, 2026-11-19

Request: "On the map side, can we get it to render faster?" (desktop Chrome, about 1440 wide, satellite base, aircraft + vessels + facility markers on).

Branch `claude/mapperf-1119`, three commits on top of `origin/main` 25fd2493 (v1.0.1472, the version production was serving). File and line references are to that base commit unless a fix commit is named.

## How this was measured, and what the numbers can and cannot say

- **Build**: `npm run build` (the production client), served by a local static server with production's cache headers.
- **Data**: production payloads replayed. Each `/api/data/*` endpoint the page calls was fetched from production once (single GETs, 24 in total, sizes and timings in `prod/timings.txt`) and served back from disk with production's measured time-to-first-byte.
- **Browser**: headless Chromium via Playwright, CDP tracing plus the sampling CPU profiler. Viewports 1440×900 @1 and 412×915 @2. An unminified build (`dist-unmin`) was used only to name functions in stacks.
- **Tools** (all in `tools/`): `measure.mjs` (scenarios `load`, `panzoom`, `still`, `toggle`, `tiles`), `deploycache.mjs`, `canvasbench.mjs`, `skewcheck.mjs`, `battery.sh`, `loadset.sh`. Raw results are in `runs/*.json`.

Three limits apply to every number below.

1. **There is no GPU here.** WebGL runs in SwiftShader (software). The main thread spends most of its "busy" time waiting for that process: during the 1440 load it was busy 14.7 s by the clock but used 2.2 s of CPU. So each main-thread figure is given as **wall** (this machine, mostly GPU wait, not transferable) and **CPU** (thread time, transferable up to CPU speed; this is a 2.1 GHz Xeon).
2. **Frame times are not representative.** A frame with satellite imagery on the globe takes 0.3–1.6 s in software GL. They are recorded for completeness and should not be read as what the owner sees.
3. **The machine was shared.** Two other sessions were running their own Chromium harnesses on the same 4 cores, so wall-clock timelines moved by ±1–2 s between identical runs. Counts, bytes, request order and CPU time are stable; whole-page wall times are not.

One tracer artifact was found and removed: starting the CPU profiler on a warm page makes V8 log all existing code, a 0.4–0.9 s stall billed to whichever callback runs next. Mid-session traces now start, wait 2 s, and analyze only what follows a mark.

## Ranked list

### 1. Every deploy made every returning visitor download all the JavaScript again — FIXED (commit a910ef59)

- **Number**: 888 KB of JS over the wire after each deploy (entry chunk 607 KB + maplibre 281 KB), before the map can start. 194 merges to main in the 14 days to 2026-11-19, about 14 a day; 167 of them (86%) changed nothing the client bundle is built from.
- **Evidence**:
  - `client/src/pages/datamap.tsx:460` imported `version` from `package.json`, so the version string was inside the entry chunk and every version bump renamed it.
  - The built maplibre chunk began `import{g as B_}from"./index-<hash>.js"` (one 120-byte CommonJS helper), so it was renamed with the entry. Changing only `mapIcons.ts` renamed `maplibre-gl-SEKs8A_a.js` to `maplibre-gl-C1RMiKPj.js`; the two files are byte-identical apart from that import.
  - Production serves hashed assets `immutable` for a year; fetching them cold costs 0.58 s (entry, TTFB 0.48 s) and 0.40 s (maplibre).
- **Fix**: the version travels in `index.html` as `<meta name="vt-client-version">` (no-store, deployed with the bundle); the interop helper gets its own chunk.
- **Before / after** (`tools/deploycache.mjs`: same browser profile, production cache headers, 50 Mbit/s + 28 ms line, a deploy that only bumps the version):

  | | JS over the wire | entry chunk available | maplibre available |
  |---|---|---|---|
  | before | 888 KB | 352 ms | 1586 ms |
  | after | 0 KB | 44 ms | 225 ms |

  After a deploy that does change client code: 888 KB → 607 KB.
- **Risk**: low. The version-skew banner was checked in the browser (equal versions: no banner; different: "Site updated to v9.9.9 (you're on v1.0.1472)"). A page with no meta tag shows no banner rather than a false one.
- **Tests**: `client/src/lib/clientVersion.test.ts`, `client/src/lib/chunkStability.test.ts`, and a new `data-chunks` check in `scripts/visual_check.mjs`.

### 2. Icon shapes were drawn on GPU canvases and read back one by one, twice — FIXED (commit cbedd1bd)

- **Number**: `registerIcons`, one main-thread task in the frame after the map turns interactive: 1547 ms wall / 81 ms CPU at 1440, 1539 ms / 90 ms at 412. With the legend open (desktop default) its first render drew every shape again: a second task of 1677 ms wall.
- **Evidence**: `client/src/lib/mapIcons.ts:64` created a default 2D canvas per shape (56 shapes) and called `getImageData`; `iconDataURL` (`:946`) called the shape function again for the legend. `tools/canvasbench.mjs`: 150 such canvases cost 803 ms GPU-backed and 8 ms with `willReadFrequently`, same machine.
- **Fix**: CPU-backed canvases (`willReadFrequently`) and one cached raster per shape, shared by the map and the legend.
- **Before / after** (the `registerIcons` call itself, from the trace):

  | viewport | before | after |
  |---|---|---|
  | 1440×900 | 1547 ms wall / 81 ms CPU | 32 ms wall / 14 ms CPU |
  | 412×915 @2 | 1539 ms wall / 90 ms CPU | 63 ms wall / 23 ms CPU |
  | 1440 at quarter resolution | 681 ms wall / 54 ms CPU | 49 ms wall / 17 ms CPU |

  The legend's first render at 1440: 1677 ms wall → 122 ms wall for the whole commit it sits in.
- **What transfers**: the read-backs are synchronous round trips to the GPU process, issued while the map creates its context and compiles shaders. They are gone on any machine. How long each one took on the owner's GPU is not known from here; the 1.5 s is this machine's.
- **Risk**: low. Software and GPU rasterization can differ by an anti-aliasing level on shape edges; the harness legend-parity check runs on it.
- **Test**: `client/src/lib/mapIcons.test.ts` (every icon context CPU-backed, one canvas per shape, nothing redrawn for a second map or the legend).

### 3. The default desktop view downloaded the whole aircraft snapshot every 8 seconds — FIXED (commit 4bebea98)

- **Number**: at 1440×900, 4 of 4 polls in 36 s were full snapshots: 332 KB over the wire and 935 KB of JSON each. The 412 px view polled deltas (57 KB / 161 KB) as designed.
- **Evidence**: `client/src/lib/air/globalFeed.ts` `heldQueryCovers` returned "re-query" for any antimeridian-crossing box ("rare, always correct"). The desktop globe's default view reports bounds west of −180, so its padded query is `lomin=142&lomax=24`: a crossing box, every time. Production answers `changed=` for that box correctly (checked: `full:false`, 1,608 rows; 0.24 s against 0.55 s for the full answer).
- **Fix**: longitude is compared as arcs, so a wrapping view keeps its held set.
- **Before / after** (`runs/polls-before-1440.json`, `runs/polls-after-1440.json`):

  | | polls | per poll, wire | per poll, JSON |
  |---|---|---|---|
  | before | FULL, FULL, FULL, FULL | 332 KB | 935 KB |
  | after | FULL, delta, delta, delta, delta | 57 KB | 161 KB |

- **What it does not change**: the rebuild of the map source after each poll (item 5). A delta still rebuilds every held row.
- **Risk**: low. Pure function; non-wrapping cases, the latitude rule and the area rule are unchanged, and the 90 s full resync still runs.
- **Test**: `client/src/lib/air/globalFeed.test.ts` (the measured box, pans inside it, edges leaving it, the area rule; and the feed's second poll of an unmoved wrapped view is a delta).

### 4. The map's own JavaScript per frame while panning and zooming — NOT FIXED, needs design

- **Number**: main-thread CPU per rendered map frame during a scripted pan + zoom: average 34 ms, worst 84 ms at 1440 (default layers); 41 ms / 78 ms with vessels and military installations added; 15 ms / 37 ms at 412. A 60 fps frame has 16.7 ms.
- **Evidence**: MapLibre's symbol pipeline over large GeoJSON symbol layers: `_updatePlacement`, `CrossTileSymbolIndex.addLayer` → `findMatches`, `updateBucketOpacities`, `removeBucketCrossTileIDs`. Sources hold 9.8–11.3k aircraft rows, 9,833 power plants, 2.5k vessels; each integer zoom and each `setData` rebuilds every tile of a source.
- **Proposal**: this is the SCALE program's S2 (server aggregation or tiling for the dense point layers), so low zoom ships cluster counts or vector tiles instead of every point. Expected gain: per-frame placement cost falls with the symbol count; not estimated further here because it depends on the tiling chosen. Risk: medium to high, touches the densest layers and their harness guards (one pins ≥9,500 aircraft in the source at 1440).

### 5. Every aircraft poll rebuilds and re-serializes every row — NOT FIXED

- **Number**: one main-thread task of 55–146 ms CPU per poll, every 8 s at world zoom (measured 55, 65, 67, 83, 127, 146 ms across runs).
- **Evidence**: `datamap.tsx:9112-9116` (`toFeatures` over the whole held set, then `src.setData(fc)`), which MapLibre serializes for its worker (`sendAsync` → serialize). The delta only changed 1,608 of 9,766 rows.
- **Proposal**: send the changed rows with `GeoJSONSource.updateData` (needs a stable feature id per aircraft and explicit removals on eviction). Expected gain: serialization proportional to changed rows, roughly a sixth of today's. Risk: medium; it is the core of the live layer and its glide and 3D hand-off read the same payload. Zoomed in to z5.5–8 the 300 ms glide re-ships the visible rows the same way: 566 ms of main-thread CPU per 10 s parked over the east coast with 1,246 aircraft, no long tasks.

### 6. Default-on layers start fetching late — NOT FIXED, a design decision

- **Number**: `sites` and aircraft are requested only once the map is ready; power plants, trains and the panel feeds only after the map's first idle or a 4 s failsafe. Measured gap from "map interactive" to the power-plant request: 1.1 s, 2.3 s and 4.4 s in three runs. Their network time on production is then added: sites 0.39 s, power plants 0.27 s (330 KB), trains 0.33 s.
- **Proposal**: request the default-on feeds when the page mounts and hand the response to the layer when its gate opens. Expected gain: 0.25–0.45 s sooner per layer. Not built because DESIGN.md says heavy default-on layers mount after first idle so the base map and aircraft win the initial contention; fetching early spends bandwidth during that window. That trade is the owner's.

### 7. Sharp imagery after a zoom arrives one tile round-trip after the camera stops — NOT FIXED

- **Number**: with tiles answering in 120 ms, sharp tiles were resident 451 ms after a jump and 637 ms after a 2.5 s zoom from z11 to z15. With real Esri tiles through this session's proxy (p50 0.4–0.8 s, p95 up to 1.8 s per tile): 1.9 s after a jump, 0.8 s after the ease. Add the 300 ms crossfade on top.
- **Evidence**: the pipeline itself behaves: a jump puts 31 requests in flight at once (cap 32); a continuous zoom skips intermediate levels; superseded requests are aborted (with 1.5 s tiles: 24 of 48 and 56 of 80 aborted, `net::ERR_ABORTED`). But the destination's tiles are requested only when the camera reaches the destination zoom, never when the zoom starts.
- **Proposal**: when a zoom has a known target (button, fly-to, search), fetch the target view's tile URLs at the start so they are in the HTTP cache on arrival. Expected gain: up to the gesture's duration. Risk: low to medium; needs the tile cover for the target view. Real tile latency on the owner's connection could not be measured from here.

### 8. Two synchronous GPU calls on the load path — NOT FIXED

- **Number**: `readGpuInfo` creates a throwaway WebGL context while the entry chunk is evaluated, on every route: 234–420 ms wall here. The device-tier effect reads the renderer string from the map's context after it is ready: 257–1238 ms wall here. CPU for both is small; the wall time is this machine's GPU process.
- **Evidence**: `datamap.tsx:737-758` (`readGpuInfo`) and the device-tier effect at `datamap.tsx:4255-4263`.
- **Not touched** because the first is deliberate crash forensics ("read eagerly at module load: a dead GPU process reports no-webgl and the identity is lost"), and the cost on a real GPU cannot be measured here.

### 9. Bundle shape — NOT FIXED

- **Number**: one entry chunk for every route: 2,112 KB raw, 626 KB gzip, 614 KB as production serves it (brotli on the fly; TTFB 0.48 s against 0.16 s for index.html). `datamap.tsx` is 20% of it, third-party code 14%; the trade pages, bot dashboard and login ride along. Evaluating it costs 57–84 ms CPU; compile is streamed (3–4 ms on the main thread). maplibre (1,053 KB raw, 283 KB wire) is requested only after React mounts.
- **Proposal**: route-level code splitting for the non-map tabs, and brotli at build time. After item 1 this matters on first visits and on the 14% of deploys that change client code.

### 10. Payloads and server response times (production, single requests)

| endpoint | TTFB | wire (br) | raw | cache-control |
|---|---|---|---|---|
| `/api/data/layers` | 0.44 s | 28 KB | 147 KB | none |
| `aircraft/global` (full) | 0.43–0.45 s | 340–377 KB | 0.96–1.11 MB | max-age=10 |
| `aircraft/global` (delta) | 0.20 s | 58 KB | 165 KB | max-age=10 |
| `aircraft` (east-coast box, cold) | 1.39 s | 68 KB | 672 KB | max-age=10 |
| `powerplants` | 0.23 s | 330 KB | 1.17 MB | 1 day |
| `vessels` | 0.38 s | 549 KB | 2.15 MB | max-age=10 |
| `military_installations` | 1.29 s | 520 KB | 2.30 MB | 1 day |
| `earnings-language` (a panel count) | 0.13 s | 85 KB | 293 KB | none |
| `cot` (a panel count) | 0.13 s | 17 KB | 113 KB | 1 hour |
| `sites` | 0.39 s | 3 KB | 10 KB | none |
| `trains` | 0.33 s | 5 KB | 30 KB | max-age=10 |

First load: 15 API requests, 801 KB over the wire, 2.67 MB of JSON (1440, one aircraft fetch in the window). The regional aircraft feed is 477 bytes of JSON per aircraft.

### 11. React is not where the time goes

- Load: 17–22 commits; React's scheduler used 161–319 ms of CPU in total.
- Toggling a layer: 3–5 commits, 11–18 ms of scheduler CPU plus 6–12 ms in the click handler. The rest of a toggle (166–298 ms CPU) is MapLibre.
- Camera parked for 10 s: 0–3 commits. Pan + zoom: 2–9 commits per 5 s pass.
- So the open SCALE item S1(d), memo boundaries for LayersPanel and DetailCard, would save under 20 ms per toggle. Commit durations per component could not be read: production react-dom carries no profiling data, and a profiling build was not made.

## The measurements asked for

**(a) Navigation to first map frame, base tiles, first data layer** (baseline, wall ms, software GL):

| | 1440×900 | 412×915 @2 | 1440 at quarter resolution |
|---|---|---|---|
| boot splash painted | 1188 | 624 | 576 |
| /data page mounted | 1594 | 934 | 911 |
| map object created | 4737 | 1825 | 3468 |
| first map frame | 5244 | 2125 | 4072 |
| skeleton gone | 8013 | 2634 | 6915 |
| base tiles resident | 8055 | 3095 | 5291 |
| first data layer drawn | 14827 | 5975 | 10668 |
| aircraft drawn | 14827 | 6959 | 13882 |
| power plants drawn | 15127 | 9920 | 13799 |
| main thread busy, wall / CPU | 14.7 s / 2.2 s | 8.9 s / 1.8 s | 10.8 s / 1.5 s |

After the three fixes the same runs gave skeleton-gone 7119 / 2994 / 7346 ms: inside the run-to-run noise of this shared machine, which is why the fixes are reported by the specific task, bytes or request they removed rather than by this table.

**(b) What blocks.** Bundle and API sizes: items 9 and 10. Long tasks over 50 ms on the main thread at 1440: 32 by wall (12.9 s in total), 13 by CPU. Largest by CPU, with top frames:

| CPU | wall | what |
|---|---|---|
| 127 ms | 177 ms | aircraft payload: `toFeatures` + MapLibre `sendAsync`/serialize (`datamap.tsx:9112`) |
| 113 ms | 2678 ms | `new maplibregl.Map` → `_setupPainter` (WebGL context; `datamap.tsx:3670`) |
| 104 ms | 1597 ms | `registerIcons` (`datamap.tsx:3773` → `mapIcons.ts:933`) — fixed |
| 102 ms | 548 ms | first frame: `_render` → `updateGPUdependent` (globe) |
| 87 ms | 1063 ms | `_render` → symbol layer → `useProgram` (shader compile) |
| 83 ms | 169 ms | second aircraft poll: serialize |
| 71 ms | 160 ms | `_render` → `_updatePlacement` |
| 56 ms | 1677 ms | React commit drawing the legend's icons — fixed |

**(c) Frame time during pan + zoom** (rAF deltas, wall, software GL — not representative; CPU per rendered map frame is the transferable figure):

| scenario | p50 | p95 | CPU per map frame, avg / worst |
|---|---|---|---|
| 1440, default layers | 867 ms | 1600 ms | 34 / 84 ms |
| 1440 at quarter resolution, default | 650 ms | 1417 ms | 28 / 45 ms |
| same, + vessels + military installations | 567 ms | 1183 ms | 41 / 78 ms |
| 412×915 @2, default | 283 ms | 583 ms | 15 / 37 ms |

**(d) Base tiles during zoom-in**: item 7.

## Not measured honestly

- Anything on the GPU: context creation, shader compilation, draw time, the globe's read-back, texture upload. No GPU in this environment.
- Frame rate as the owner sees it.
- Real tile latency from Esri on the owner's connection (the figures above went through this session's proxy).
- Per-component React commit durations.
- The effect of fix 2 on the owner's machine in milliseconds: the mechanism is removed, its size there is unknown.
- A before/after for the page-level timeline: the noise on this shared 4-core machine is larger than the fixes' effect on it.

## Shipping the three commits

The commits touch disjoint files, and each was cherry-picked onto `origin/main` 25fd2493 alone and passed its own tests there. They can ship in any order.

| commit | files | guard | alone on main |
|---|---|---|---|
| cbedd1bd icon rasterization | `client/src/lib/mapIcons.ts`, `mapIcons.test.ts` | `mapIcons.test.ts` | 13 of 13 pass |
| a910ef59 asset names | `vite.config.ts`, `client/src/lib/clientVersion.ts` + test, `client/src/lib/chunkStability.test.ts`, `scripts/chunkStability.mjs`, `scripts/visual_check.mjs`, `client/src/pages/datamap.tsx` (3 lines) | `clientVersion.test.ts`, `chunkStability.test.ts`, harness check `data-chunks` | 35 of 35 pass with `server/layersRegistry.test.ts`; tsc 6 |
| 4bebea98 aircraft deltas | `client/src/lib/air/globalFeed.ts`, `globalFeed.test.ts` | `globalFeed.test.ts` | 13 of 13 pass |

a910ef59 holds two parts with one purpose (the version meta tag, and the interop helper chunk). They are in separate hunks of `vite.config.ts` and could be split into two PRs if wanted.

**What a returning visitor's browser does on the first deploy after a910ef59 ships.** That deploy changes client code, so it behaves like any deploy today: `index.html` is fetched fresh (it is `no-store`), it names a new entry chunk, and the browser downloads the entry (607 KB), the maplibre chunk under its new stable name (281 KB) and the 120-byte `cjs-interop` chunk. That is one last full download. The old files stay in the browser cache unused until evicted. From the next deploy on: if it changed no client file, `index.html` (fresh, with the new version in its meta tag) names the same files and nothing is downloaded; if it changed client code, only the entry chunk is. A tab left open across a deploy still shows the "Site updated" banner on resume, and a reload clears it.

**Server caching headers: no change needed.** `server/static.ts` already serves everything under `/assets/` as `public, max-age=31536000, immutable` and every `.html` as `no-cache, no-store, must-revalidate`. The new `cjs-interop-<hash>.js` is emitted under `/assets/` and gets the same header. The fix only stops file names from changing when contents have not; it relies on the headers exactly as they are. The build is deterministic: two builds of the final tree produced the same names (`index-CLYJq-zq.js`, `maplibre-gl-DXl5kUcc.js`, `cjs-interop-CqkleIqs.js`).

## Gates on the final tree (4bebea98)

| gate | result |
|---|---|
| `npx tsc --noEmit -p .` error count | 6 |
| `bash scripts/counter_ratchet.sh` | `OK: 28 counters at or better than baseline.` |
| `python3 -m pytest -q test_ts_code_only.py` | 11 passed |
| client suite (`npx tsx --test` over `client/src` and `shared`) | 1644 tests, 1644 pass, 0 fail |
| full Python suite (with the one deselect) | 4877 passed, 2 skipped, 1 deselected, exit 0 |
| full server suite (nothing under `server/` was touched; run once before the container restart on the same tree) | 4127 tests, 4127 pass, 0 fail |
| `node scripts/deploy_gate_smoke.mjs` (builds, boots the bundle) | PASS: `/api/health` answered 200 in 8.1 s |
| `npm run visual` | **not completed** — see below |

**Visual harness.** The full run never finished: it was stopped twice by the 60-minute background limit and once by the container restart, on a machine at load 10–22. One map-page run (`--page data`) did finish, at load 11–17 while this session's own Python and server suites were also running:

- PASS: `data-all-off` (zero-cost, TTI 2078 ms), `data-chunks` (the new vendor-chunk check), `data-scale`.
- FAIL, 51 hard failures on the data page:
  - 390: 44 — frame median 183 ms against a 120 ms gate; the time-scrubber's close control could not be clicked and its panel stayed open; then 41 "registered layer has no reachable panel row".
  - 768: 3 — TTI 3299 ms against 3000 ms; frame median 300 ms against 200 ms; p95 550 ms against 350 ms.
  - 1440: 4 — frame median 367 ms against 250 ms; p95 600 ms against 350 ms; the time-scrubber's slider and play button reported as covered by the floating legend.

These are not attributed. The frame-time and TTI gates measure wall time and the machine was oversubscribed. The time-scrubber and panel failures could be timing under load or could exist on main; the comparison run against the base build was cut off by the restart and was not repeated, to keep to one browser process at a time while another session's harness was running. None of the three commits touches the layers panel, the time scrubber or the legend's layout. The 390, 768 and 1440 screenshots of the data page from the final build were reviewed: map, marker symbols, legend icons and panel render as expected. This gate needs one run on a quiet machine before the PRs merge.
