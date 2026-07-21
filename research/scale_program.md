# SCALE PROGRAM — "everything on, more data, no loss of detail/latency/info"

INSTALLED 2026-07-07 by human directive (screenshot: /data LAYERS
panel showing a red "heavy load" badge with 6/6 BASE on): "how do we
get this to run faster with all information on — we will add way more
than what we have now and we don't want to lose info or data latency
or details." Multi-session program like GRID VISION / ANALYST CONSOLE;
RESUME STATE at the bottom is authoritative. CLAUDE.md governs HOW
everything ships; this names WHAT scale requires.

## THE REFRAME (the whole program in one idea)

You never lose data, because STORAGE fidelity and RENDER fidelity are
different problems:
- STORAGE: the archives keep EVERYTHING, forever, at full resolution.
  Adding 10x more streams changes storage, never the map's frame time.
- RENDER: the map only ever draws what is in the current VIEWPORT at a
  zoom-appropriate level of detail. The screen shows a few thousand
  features whether the platform holds 25 streams or 250.

"All layers on" must cost ~the same regardless of total data volume.
The "heavy load" badge is honest today (it sums the static cost of
active layers); the program makes that cost VIEWPORT-REAL and small,
so the badge rarely trips even with everything on. Detail is not
dropped — it is REVEALED on zoom. Anything that instead just turns
layers off, or decimates harder globally, trades away one of the
three constraints; this architecture satisfies all three at once.

## THE THREE LEVERS (priority order; each its own PR slice)

### S1 — VIEWPORT-BOUNDED SERVING (bbox + zoom). THE unlock.
Today several layers fetch broad/national data and the client renders
all of it. Change: every layer data request carries the map's current
bounding box + zoom; the server returns only features inside the view
at that zoom's density. Pan/zoom → debounced refetch (the client
already has fetch/poll discipline from P-PERF). Rendered volume
becomes ~constant, independent of total archive size. This is the
single biggest win and the precondition for the rest. Touches every
layer's fetch path — sequence it as: (a) add bbox/zoom params to the
shared datacore serving helper; (b) migrate the densest layers first
(aircraft, vessels), one PR each, harness perf gate proving flat
frame-time as data grows; (c) sweep the rest.

### S2 — SERVER-SIDE TILING / AGGREGATION for dense point layers.
Aircraft/vessels are tens of thousands of points (78k aircraft near
Cushing in a week). Pre-aggregate into a tile pyramid or cluster grid
keyed by zoom (the "2.7k" cluster bubbles are this idea done
client-side — move it server-side). Low zoom ships hundreds of cluster
counts; high zoom ships the individual features for that tile only.
Reuse the vector-tile/decimation tooling from the power-grid build
(DM-2). Full detail preserved in the archive AND revealed on zoom.

### S3 — SPATIAL INDEX on the archives. THE latency answer.
Archives are JSONL day-files; "everything near here" currently SCANS
files (the W4 query engine does this per request — fine now, the
latency risk at 10x). Add a spatial index so a viewport query is
O(log n), not O(scan): SQLite R*Tree (better-sqlite3 is already in the
stack) or a geohash/tile key stamped at ingest, updated append-only
alongside the JSONL. Query time stays FLAT as data grows — this is
what keeps "no added latency" true when the archive is 10x bigger.

## SUPPORTING MOVES (fold into the slices above)
- CLIENT GPU RENDERING: any marker-based layer → MapLibre GeoJSON
  source + data-driven style layers (GPU draws 100k+ features fine;
  chokes on thousands of DOM markers). Audit which layers still use
  markers.
- PROGRESSIVE / PRIORITY LOAD: base + highest-value layers paint
  first, the rest stream in with skeletons (perceived-performance rule
  already gated in the harness).
- OFF-SCREEN = FREE: a layer "on" with no features in view costs
  nothing — falls out of S1 automatically.
- VIEWPORT-AWARE "heavy load" BADGE: once serving is bbox-bounded, the
  real cost is "features currently rendered," so the advisory should
  reflect that, not the static per-layer cost sum. Update after S1.

## HONESTY / NON-NEGOTIABLES
- The archive is never thinned to make the map faster (data-loss ban).
- Decimation/LOD is a RENDER choice, always reversible by zoom, never
  a storage choice.
- Every slice ships with a harness perf assertion PROVING frame-time
  and query-time stay flat as feature count rises (the point of the
  program is measurable, not vibes).
- No silent caps: if a tile/zoom drops features, the count dropped is
  stated (existing silent-cap ratchet extends here).

## RESUME STATE (update every session that touches this program)
- 2026-07-16 (scheduled-routine session, T-CLIENT, PR #505): S1(d) React
  memo boundary — PARTIALLY shipped (v1.0.373). datamap.tsx's Legend
  section (~330 lines, 24 well-scoped dependencies, zero live-tick
  coupling) extracted into `LegendPanel`, wrapped in `React.memo`, so it
  stops re-rendering on the satellite/aircraft/vessel position ticks
  that dominate DataMapPage's render volume once any live layer is on.
  LayersPanel and DetailCard (the other two named in the original S1(d)
  item) remain unextracted — each carries materially more entangled
  interactive state (opacity sliders/date scrubbers/description toggles;
  ~15-way branch on entity kind with fetch/trail side effects) and needs
  its own scoped session. Full reasoning + verification in
  experiments.md and earth_twin_program.md's RESUME STATE. No perf-
  harness assertion added — this sandbox's visual harness crashes on the
  real /data page (pre-existing, documented 2026-07-14) before it can
  exercise the live-tick path; substituted an ad hoc Playwright
  reactivity check instead. REMAINING QUEUE (unchanged otherwise): (d)
  LayersPanel + DetailCard memo boundaries, (e) median lever (human
  input), (f) S2 server aggregation.
- 2026-07-07: charter installed. NOTHING BUILT YET. Next action: S1
  step (a) — add bbox+zoom params to the shared datacore serving
  helper + a harness perf fixture that scales feature count and
  asserts flat frame-time; then migrate aircraft (densest) as the
  first proof. T-DATACORE (server serving) + T-CLIENT (bbox on fetch)
  — one logical change per PR, serialized on the shared files.
- 2026-07-15 (EARTH TWIN session #2, human report "extremely laggy and
  freezes often"): full 3-agent perf PROFILE run (client render path,
  every datamap timer/poller + React churn, server payloads/event-loop
  blockers — evidence in experiments.md). SHIPPED THIS SESSION:
  v1.0.324 SATCAT parse off-thread (~300ms-1s enable freeze killed);
  v1.0.325 /api/data/track + /api/v1/tracks off the event loop
  (streamed reads + id prefilter + 30s cache — was up to 48 sync
  readFileSync+gunzipSync+parse per request, re-fired every 30s per
  open card, stalling ALL responses AND the trading loop);
  v1.0.326 live-points tick pipeline (vector build gated below
  visibility + zoomend lazy build, isMoving tick skip, jitter-pan
  refetch skip vs the served 250nm circle, display-quantized counts so
  the render bail engages — p95 frame 200→133ms @768, 267→183ms @1440,
  upload-hitch warning gone; medians unchanged as expected: they are
  steady-state draw volume, the harness stubs the tick path).
  SAME SESSION, QUEUE ITEMS SHIPPED: (1) maintenance-timer stalls →
  v1.0.327 (compressOldHoursAsync streamed pipeline w/ partial-gz
  rollback + rollupOldDaysAsync on shared accumulation helpers +
  in-flight latches; equivalence test-pinned vs the sync paths);
  (3) GP parse → v1.0.328 (gpWorker one-shot fetch+parse, abort →
  terminate, main-thread fallback; the satellite-enable 150-500ms
  freeze removed).
  REMAINING QUEUE (ranked; next continuous-build session picks up
  from here):
  (a) W3/W4 SYNC READS: querySnapshot/scanEventLayer readJsonlDay is
      still readFileSync+gunzipSync per uncached scrub position —
      convert to the shared streamJsonlLines pattern (v1.0.325/327
      precedent, straightforward).
  (b) VESSELS DELTA: handler emits no `time` and ignores `since` →
      full 2.37MB re-ship every 20s when enabled; give it the aircraft
      treatment + Cache-Control on the three live endpoints. (Lower
      urgency: vessels is default-off + awaiting_key today.)
  (c) 1Hz ORBITAL REPAINT: updatePositions → triggerRepaint every
      second means the map never idles while satellites are on —
      weak-GPU lag; consider skip-when-subpixel or lower hz + shader
      interpolation.
  (d) REACT MEMO BOUNDARIES: 45 useState in one 5.8k-line component,
      zero memo — extract LayersPanel/Legend/DetailCard as memoized
      children (remaining full-tree renders → subtree renders).
  (e) MEDIAN LEVER (visual tradeoff — human input welcome): low-zoom
      draw density keepFraction 0.35→~0.2 + globe cost. 2026-07-05
      precedent measured 10k→3.5k icons = median 117→83ms @1440;
      today's 133-167ms medians are steady-state draw volume under
      globe projection.
  (f) S2 SERVER AGGREGATION (structural): low zoom ships cluster
      counts, not 10-15k individual records — the charter's own plan;
      kills the payload+parse class entirely at continent zoom.
- 2026-07-15 (continuous-build session, T-CLIENT only): QUEUE CORRECTED
  before picking up work — (a) W3/W4 SYNC READS was already done: an
  earlier same-day commit ("OUTAGE-CLASS SWEEP 2/2", queryEngine.ts:
  111-130) converted querySnapshot's readJsonlDay call (BOTH position
  and event modes) to the async streamed reader; only scanEventLayer's
  small-file multi-day fold stays sync, already audited low-risk
  (KB-scale daily pulls). (b) VESSELS DELTA also already done
  (v1.0.340, logged in experiments.md, this file's own item (b) simply
  never got checked off). Neither is re-litigated here — pure
  record-keeping so the next session doesn't re-derive it.
  (c) 1Hz ORBITAL REPAINT SHIPPED (v1.0.343): SatLayer.updatePositions()
  gained an optional tickIntervalSec param — when the worker (hz=1)
  supplies it, the layer accumulates elapsed tick time since the last
  ACTUAL repaint and only forces map.triggerRepaint() once the
  worst-case ground-track displacement (conservative 8000 m/s bound,
  covers any LEO object) would exceed one screen pixel at the map's
  current center lat/zoom (reuses lib/lod.ts's existing metersPerPixel,
  no new math primitive) — then resets the accumulator. Accumulating
  (not just checking the latest tick) bounds staleness to <1px of
  drift at all times, so the layer can never silently freeze: any
  repaint from ANY source (camera move, another layer) still redraws
  fresh data via the pre-existing dataDirty flag, and the position
  buffer itself is always updated regardless of the repaint decision.
  Followed-satellite tracking is untouched — modelLayer.setAnchor()
  (called every tick from followTick()) already triggers its own
  repaint unconditionally, so follow stays fully live. 5 new tests in
  satLayer.test.ts (pure threshold math incl. fail-open on bad input,
  accumulation-then-forced-repaint at low zoom, immediate repaint at
  close zoom, backward-compatible no-tickIntervalSec path). GATES:
  server 699/699, client 199/199, tsc 66-line baseline, build clean;
  visual harness 0 hard failures at 390/768/1440 (this default battery
  does not enable orbital_sats, so it cannot exercise the tick path
  directly — same class of measurement gap the live-points fixes
  noted; medians unchanged-to-slightly-better: 33/83/100ms vs the
  prior 33/83/117ms baseline, within normal run-to-run noise). REMAINING
  QUEUE: (d) React memo boundaries, (e) median lever (human input),
  (f) S2 server aggregation.

## 2026-07-21 — FULL-STACK SPEED AUDIT (human: "the map as a whole with any
## layer on or all on is very slow load slow the data is slow … fix all over
## for speed of the ui") — measured backlog, execute top-down

4-agent audit (load path / data path / UI render / production measurement);
raw timings in the session scratchpad (perf_audit/speed/prod_timings.txt),
full findings in experiments.md round-12 entry's audit reference. Local
baseline is healthy (map visible ~760ms, first layer ~1.5s under software
GL) — the cost is network/payload/server-stall/React-render, not GL.

SPEED WAVE 1 (server, ~1.5d total — biggest measured wins):
- S-A1 aircraft STALE-WHILE-REVALIDATE (routes.ts ~876): cold-cache poll
  waits 1-15s on the upstream inline (measured 3.8s TTFB vs 0.21s warm,
  recurring every 10s TTL). Serve the expired cache immediately (marked
  cached:true — honesty chrome already shows staleness) + background
  refresh. THE top "data is slow" item.
- S-A2 vessels delta DEAD CODE (liveDelta.ts:39): snapshot TTL 15s <
  client poll 20s so `unchanged` NEVER fires — ~2MB raw/~400KB gz re-ships
  every poll. Raise TTL ≥ 25s (one constant + test); follow-up: zoom-capped
  payload (S2 alignment).
- S-A3 /api/data/layers: slowest median TTFB (569ms), no cache-control,
  near-static 69KB registry fetched on every open → memoize + max-age=60.
- S-A4 logging middleware double-stringify (index.ts:88): full-body
  re-serialize per /api response → cap capture for large bodies.
- S-A5 static datacore JSONs (powerplants et al) → pre-stringified,
  pre-gzipped buffers at module init + cache-control on layers/sites/
  streams/fires (currently none).
- S-A6 trains since=/unchanged delta (client already speaks it).

SPEED WAVE 2 (client load, ~2d):
- S-B1 maplibre import hoist to module scope (fetch starts during main-
  chunk eval instead of after React mount; today fully serialized after
  400KB gz + parse) + modulepreload injection at build.
- S-B2 Google Fonts render-blocking CSS → self-host woff2 (removes 2
  third-party origins from the first-paint path).
- S-B3 mapSettled fetch burst split: map-visible layers first; the 5
  panel-count-only feeds (insider/earnings/shortvol/attention/cot) +
  graph/shadowstats behind requestIdleCallback/panel-open.
- S-B4 brotli precompress at build (+123KB saved on the two critical
  chunks) via .br siblings + static middleware.
- S-B5 tab-level code splitting (home.tsx eager-imports all 9 tabs incl.
  bot+lightweight-charts; datamap eager-imports 9 overlay views) — the
  1.36MB index chunk is mostly non-/data code.
- S-B6 celestialSky chunk gated behind the globe branch (49KB gz wasted
  on flat map).

SPEED WAVE 3 (UI responsiveness, ~2d):
- S-C1 ticker colocation (1Hz sim-clock chip + space card + 10s freshness
  re-render the whole 10.5k-line DataMapPage; ~90 useState/102 useEffect)
  → leaf components (FpsChip precedent).
- S-C2 extract memoized <LayersPanel> + colocate its interaction state
  (123 registry rows re-diffed on every parent tick; LegendPanel
  precedent).
- S-C3 backdrop-filter blur off the always-visible panel surfaces (per-
  frame GPU re-blur over the animating map; keep on transient popovers).
- S-C4 applyMarkerLod early-out on unchanged quantized camera altitude
  (runs per move event, 60fps during follows).

Rules: each slice = own PR + tests per promotion ladder; measurement
changes (S-A4) separate from behavior changes; the harness perf section
gates regressions. Server slices touch routes.ts (SHARED territory —
serialize, smallest-last-commit).
