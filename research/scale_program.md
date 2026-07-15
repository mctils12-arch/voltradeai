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
  EVIDENCE-BACKED QUEUE (next unblocked slices, ranked):
  (1) MAINTENANCE-TIMER STALLS: compressOldHours (30min) +
      rollupOldDays (6h) gzipSync whole hours/days in-process —
      periodic multi-second freezes with zero user interaction; same
      class: querySnapshot/scanEventLayer sync readJsonlDay (W3/W4).
      Move to worker_threads or streamed async (the v1.0.325 pattern).
  (2) VESSELS DELTA: handler emits no `time` and ignores `since` →
      full 2.37MB re-ship every 20s when enabled; give it the aircraft
      treatment + Cache-Control on the three live endpoints.
  (3) GP PARSE: 6.6MB res.json()+parseGp on the main thread at
      satellite enable (~150-500ms) — move into satWorker (E4-2
      pattern); also reconsider the 1Hz triggerRepaint (map never
      idles while satellites are on — weak-GPU lag).
  (4) REACT MEMO BOUNDARIES: 45 useState in one 5.8k-line component,
      zero memo — extract LayersPanel/Legend/DetailCard as memoized
      children (remaining full-tree renders → subtree renders).
  (5) MEDIAN LEVER (visual tradeoff — human input welcome): low-zoom
      draw density keepFraction 0.35→~0.2 + globe cost. 2026-07-05
      precedent measured 10k→3.5k icons = median 117→83ms @1440;
      today's 133-167ms medians are steady-state draw volume under
      globe projection.
  (6) S2 SERVER AGGREGATION (structural): low zoom ships cluster
      counts, not 10-15k individual records — the charter's own plan;
      kills the payload+parse class entirely at continent zoom.
