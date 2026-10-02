# VolTradeAI performance and freshness audit

* **Date:** 2026-10-02, 16:35–22:45 UTC.
* **What was audited:** production `voltradeai.com`, v1.0.1020 (`main` @ 8a3f366).
* **Scope:** audit only. No code, config or data was changed and no PR was opened. Measurement scripts lived in the session scratchpad, outside the repository.

**Instruments used.** The brief named Chrome DevTools MCP and Railway logs/metrics; **neither was available in this session.** These substitutes were used and are named next to every number:

| Area | Instrument |
|---|---|
| Client | Headless Chromium driven over the Chrome DevTools Protocol: performance metrics, network events, forced GC, plus in-page wrappers on timers, rAF, workers, sockets, `fetch` and WebGL. **Rendering is SwiftShader (a CPU-emulated GPU, 4 cores),** so absolute frame rates are worst-case. Use them for ranking, and read the main-thread and network numbers at face value. |
| Backend (production) | The service's own surfaces: `/api/health`, `/api/data/*` coverage blocks, `/api/data/archive/*`, `/api/data/pipeline-health-dashboard`, and the token-gated `/api/diag/{audit,daemon,archive}` probes (read-only, via `DIAG_TOKEN`). A 60 s server monitor ran for about 4.5 h. |
| Backend (controlled) | The same production bundle, built locally from 8a3f366 and run with a preload hook. The hook logged every outbound HTTP call and event-loop delay and could inject upstream timeouts or 500s. It ran with no broker, R2 or GitHub credentials, so it could not trade or touch real storage. |

---

## 1. Executive summary: top 5 problems by user impact

**1. The worldwide aircraft layer is routinely stale or half-empty, and the UI says it is fresh.**
* **Server side:** the median server-side position age is **47 s** (p90 351 s).
* **Collapses:** the global snapshot repeatedly collapsed from ~13,000 aircraft to **1,900–3,300 for 10–20 min at a time** (17:47, 17:57, 18:18Z). In the 60-min soak, the share of positions updated in the last 2 min swung from **2 % to 98 %**, while the panel label read "data 0m old" the entire time.
* **Cause:**
  * adsb.lol rejects **36 %** of sweep calls with 429 (304 of 834);
  * the disc sweep is effectively dead (5 of 782 discs ever visited);
  * the type lane refreshes only every 60 s;
  * rows are evicted after 600 s.
* **Per-user path:** the viewport aircraft endpoint returns **502 to 7.5 % of requests at 10 concurrent users and 56 % at 50**. Overflow falls onto adsb.fi, which is **not licensed for commercial use**, at up to 82 req/min against its ~60/min limit.
* Sections D3, D4, D8, G and B3.

**2. Seven redeploys a day reset the whole data platform, and four of the seven change nothing.**
* **Frequency:** 7 SIGTERM restarts in 24 h, 4 of them redeploying the identical version 1.0.1020. `railway.json` has no `watchPatterns`, so `research/*`-only `[NO-ACTION]` log commits redeploy.
* **Every boot costs:**
  * an event-loop stall of **2.9–9.5 s** about 1–2 min after boot (all ~55 collectors fetch at once: 178 calls to 48 hosts in 19 s);
  * all in-memory feeds reset;
  * the **default-on `portdwell` and `shadowstats` layers stay blank for ~60 min**;
  * the first request to archive endpoints takes **50–90 s or times out** (`/api/data/aircraft/hexes`, `/api/data/fleet-utilization`, `/api/data/platform/stats`).
* **Crash-guard trap:** a SIGTERM during a long fold is recorded as a crash and suppresses that fold for 6 h (`crashSafeRefresh.ts`).
* Sections D1, D2, D7 and F2.

**3. The worldwide aircraft archive is not being written; the volume is under its safety floor.**
* **Volume:** 1.81 GB free of 5 GB, below the 2 GiB gate. The global sweep archive is **paused**: **854,251 fixes were dropped between 16:06 and 19:58Z, with 0 archived.**
* **Effect on replay:** it is thin. Austin over the last hour returned 15 aircraft with about 4 points each.
* **Runway:** the permanent tier grows about 9.4 MB/day, which hits the 1 GiB guard in about 85 days.
* Sections E1 and E3.

**4. Turning on more than a handful of layers makes the map unusable, and several layers fan out per user to third-party hosts.**
* **All layers on:**
  * 0.1 fps, with long tasks of **9.3 s idle and 35 s while interacting**;
  * **about 360 MB of GPU textures (estimate)**, 44 timers, 2.4 GB of browser memory;
  * main-thread JavaScript alone is 255 ms per frame, which a real GPU does not remove.
* **Heaviest single layers:** `seafloor` (3.3 fps idle, +165 MB textures, a 3.9 s long task), `orbital_sats` (9.7 fps, and it leaks about 2 MB of heap per toggle), `terrain` (+224 MB textures).
* **Per-user fan-out:** each raster layer makes **750–880 tile requests/min per user, straight from the browser** to NASA GIBS, NOAA nowCOAST, GEBCO, AWS and Esri (Law II: "runtime never touches an upstream WMTS").
* **Galaxy S24 emulation:** default layers under stress run at **4.4 fps with the main thread 92 % busy**; all layers could not even be enabled within 4 min.
* Sections A2–A5, A8 and B5.

**5. The freshness signals users see are frozen or wrong.**
* **Frozen panel:** `/api/data/layers` is fetched **once per page load** (`datamap.tsx:3547`). Ages never advance, and a feed reported "down" at load stays disabled for the session. This was observed: the `vessels` switch was disabled while AIS was live.
* **False stale label:** `shortvol` displays "**88.8 days old**" while its data is one day old (`streamsInventory.ts:158-164` picks `backfill_done.json` as the newest file).
* **Missing labels:** only **16 of 256** layers show any age at all.
* **Silent staleness during an outage:**
  * the global aircraft feed, the fast lane (`200 {}`), and earthquakes, alerts, space weather and fires all keep serving old data with no stale flag;
  * only the viewport aircraft endpoint flags `stale: true`.
* Sections G, D5, B3 and C3.

**Outside this audit's scope, but critical (CLAUDE.md Priority 1):**
* **Halt:** the bot's Tier-2 scan has been **blocked by a drawdown halt since at least 2026-10-01 19:21Z (≥ 27 h)**.
* **Stale figure:** the halt message cites "DD 18.39 % (cur=$91,185)", while the live equity is $104,876 (a 6.1 % drawdown).
* **Alarm silent:** `/api/health` reports the bot `active`, with liveness not dark.
* **Where:** `server/bot.ts:4372`, from `/api/diag/audit?type=DD-HALT`. Not investigated further, per the audit-only scope.

---

## 2. Every load source ranked by cost

| # | Source | CPU | Memory | GPU (estimate) | Network | $ |
|---|---|---|---|---|---|---|
| 1 | Client: all layers on | 1.1 s main thread per frame; long tasks up to 35 s | 108 MB JS; 2.4 GB browser total | 360 MB textures + 34 MB buffers | 39–101 req/min | — |
| 2 | Client: **seafloor** layer | 13 ms/frame idle, 1,030 ms/frame panning | +17.6 MB | **+165 MB** | 840 req/min panning (S3, GEBCO) | upstream-borne |
| 3 | Server: **boot storm** (×7/day) | 94–157 % CPU locally; 2.9–9.5 s loop stalls in prod | RSS +356 MB in 16 s | — | 178 calls / 48 hosts in 19 s | — |
| 4 | Client: **terrain** layer | 218 ms/frame panning | +6.3 MB | **+224 MB** | 338 req/min panning | — |
| 5 | Client: **orbital_sats** | 9.6 ms/frame idle (9.7 fps); 1 worker at 1 Hz | +15 MB; leaks ~2 MB per toggle | 0 | catalog 7 MB once | — |
| 6 | Client: GIBS / radar / seafloor rasters (each) | 48–60 ms/frame panning | +3–10 MB | +0.6–17 MB | **750–880 req/min per user, to third parties** | third-party quota |
| 7 | Server: cold archive scans after a restart (`hexes`, `fleet-utilization`, `platform/stats`, `trips`, `shadowstats`/`portdwell` folds) | 16 s – 60 min each; ~0.5 s loop stalls | not measured | — | 7.2 MB response (`hexes`) | — |
| 8 | Server: Node process, steady state | 16–27 % CPU (local, 1–50 users) | heap 0.83–1.21 GB, RSS 1.2–1.68 GB (prod) | — | 25 outbound/min idle | Railway (UNMEASURED) |
| 9 | Client: aircraft layer | 1.3 ms/frame idle; 3 timers; 3.3 Hz `setData` | **+32.5 MB** | +0.5 MB | 15 s poll + 8 s delta; 2 s while a card is open | — |
| 10 | Server: fast lane, per watched plane | — | — | — | **+28 adsb.lol calls/min per user** | provider budget |
| 11 | Server: viewport aircraft, per distinct viewport | — | — | — | 1 upstream call per bbox per 30 s (4 s with `fresh=1`) | provider budget |
| 12 | Client: weather temp/wind | 44–58 ms/frame panning | +3–7 MB | ≤ 1.2 MB | **~770 req/min per user to our `/api/data/weather/grid`** | — |
| 13 | Client: places / powergrid (pmtiles) | 52–86 ms/frame panning | +2.5–2.8 MB | ≤ 1.1 MB | 135–165 req/min panning | R2 egress $0 |
| 14 | Python daemon | not measured | 187 MB RSS (cap 1,024) | — | — | — |
| 15 | Client: every other point layer (~70) | 0.16–0.6 ms/frame idle | +1–26 MB each | ≤ 0.5 MB | 1 fetch, then a 5–60 min poll | — |
| 16 | Storage: R2 | — | 1.76 GB, 2.16 GB steady state | — | — | **$0** (free tier) |
| 17 | RunPod | — | — | — | — | **$0/month** ($6.88 lifetime, last job 07-10) |
| 18 | GCP / BigQuery | not used by the code | | | | $0 |

## 3. Every live layer: freshness target vs measured staleness

| Layer | Target (registry/description) | Measured | Verdict |
|---|---|---|---|
| Aircraft, global | "updates ~10 s" | server rows p50 **47 s**, p90 351 s, max 600 s; fresh share 2–98 % | **fails** |
| Aircraft, viewport | ~10 s | 30 s SWR cache + 15 s poll, up to ~45 s; frozen with `stale: true` during provider failure | fails the target, honest flag |
| Aircraft, selected (fast lane) | 2.5 s | 150–170 ms per request when the provider answers; silent `{}` when it doesn't | ok, but fails silently |
| Vessels | live | last AIS message < 2 s; 30 s cache; 20 s poll | **ok**, but the panel showed "down" all session (load-time snapshot) |
| Trains | live | 30 s poll; 0 rows for 2.5 min after an upstream blip | ok / brittle |
| Earthquakes | live | collector 15 min + client 2 min, up to 17 min | collector-bound |
| NWS alerts | live | up to 15 min (10 + 5) | collector-bound |
| Space weather | live | up to 15 min (10 + 5); 3.5 min at probe | ok |
| Fires (FIRMS) | NRT | registry 0.7 h; upstream ~3 h | ok (upstream) |
| Radar (nowCOAST) | ~5 min | browser-direct, not observable server-side; 10–31 aborted requests per 8 s pan | UNMEASURED |
| GIBS rasters | daily | browser-direct | UNMEASURED |
| portdwell / shadowstats (default on) | 10 min | **blank for ~60 min after every restart** | **fails** about 29 % of the day at 7 restarts |
| waterviolators | weekly | still `warming_up` at boot + 80 min | fails |
| shortvol | daily | data 1 day old; **label says 88.8 days** | label wrong |
| cot | weekly | 41 h | ok |
| earnings | daily | 3.4–4.1 h | ok |
| satellites, server path | 6 h | 0 objects (CelesTrak blocks Railway) | known (R17); the client path works |

## 4. Duplicated, orphaned or leaking loops and connections

| Issue | Kind | File:line | Evidence |
|---|---|---|---|
| The selected plane is fed by 3 loops: viewport poll at 2 s with `fresh=1`, fast lane at 2.5–4 s, global feed at 8 s | duplicate | `datamap.tsx:9128`, `lib/air/selectedFastLane.ts:21`, `lib/air/globalFeed.ts:40` | static, plus D3 (+28 upstream calls/min per watched plane) |
| Three per-frame samplers: frameCore + frame recorder + governor | duplicate | `render/frameCore.ts`, `datamap.tsx:757` (started :4131), `datamap.tsx:4215` | 150 rAF callbacks/s at an idle baseline |
| Followed satellite propagated twice (worker 1 Hz + per-frame SGP4) | duplicate | `lib/orbital/satWorker.ts:339`, `datamap.tsx:6805` | static |
| `smoothFollowFrame` re-arms rAF every frame whenever satellites are on, even with nothing followed | orphan-ish | `datamap.tsx:6805` | static |
| `orbital_sats` toggle leaks heap | leak | satellite effect `datamap.tsx:6699–7258` | +21 MB after 10 toggles, after GC |
| radar / terrain / seafloor textures not freed on toggle-off | leak (estimated GPU) | radar `datamap.tsx:8321`; DEM `:5516–5734` | +16.7 / +54 / +51 MB after one cycle |
| Global aircraft poll and viewport poll not gated on hidden tab | should be event-driven | `datamap.tsx:8954`, `globalFeed.ts` | static |
| fires, weather wind/temp, shadowstats, portdwell, vessels intervals not hidden-gated | should be gated | `datamap.tsx:12207, 8463, 12938, 13030, 9499` | static |
| Map-event recomputation (Law I) | event-driven visuals/fetches | `datamap.tsx:7081, 7309–7311, 9059, 9067, 8462, 5441` | static |
| Aircraft "glide" = 3.3 Hz GeoJSON re-tile instead of per-frame lerp | Law I | `datamap.tsx:8983` | static |
| `/api/data/layers` read once; never refreshed | missing loop | `datamap.tsx:3547` | vessels switch stuck "down" (A2) |
| Unix-socket RPC drops connections under concurrency | connection | `voltrade_daemon.py:734` (socketserver default backlog) ↔ `server/bot.ts:101` | 24/32 and 52/64 concurrent connects fail with EAGAIN, falling back to Python subprocesses |
| Every collector starts at boot with phase-locked timers | thundering herd | `server/*Poll` (e.g. `cftcCot.ts:349`) | identical `time` stamps 18:06:24; loop stalls at boot + 1–2 min |
| Pollers surviving layer toggle-off | none found | — | 93 single + 150 repeated toggles: 0 residual intervals, 0 fetches |
| WebSocket / SSE duplicates | none | — | the client opens none |

## 5. Root causes of stale data, ranked

1. **Upstream budget exhaustion on the primary aircraft provider.**
   * 36 % of sweep calls get 429.
   * The governor backs off the type lane; one sweep stalled for 20 min (steps 373 → 378).
   * Rows expire at 600 s, so the snapshot collapses to 15–25 %.
   * Evidence: D4 and the server monitor in F1/G.
2. **Restart-driven state loss, about 7 times a day.**
   * Every deploy discards the in-memory snapshots and caches.
   * Folds rerun from scratch, giving portdwell/shadowstats a 60 min blank, plus the 6 h cooldown trap.
   * Evidence: D7 and the warm-up poll (17:05 still warming, 17:10 ready).
3. **The client never re-reads freshness or status.**
   * `/api/data/layers` is read once per page load, so labels freeze at load time.
   * Evidence: B3, a constant "data 0m old" while the fresh share reached 2 %; the A2 vessels switch.
4. **Silent stale serving on outage.**
   * Global aircraft, the fast lane and the hazard feeds return 200 with old or empty data and no flag.
   * Evidence: D5.
5. **Collector cadence well behind the upstream.** The 15-minute earthquakes collector and 10-minute alerts and space-weather collectors set the floor, not the upstream. Evidence: G.
6. **Wrong freshness computation.** Newest-file-by-name instead of by date: `streamsInventory.ts:158-164` → the shortvol 88.8 d label.
7. **Archive write gate.** With the volume < 2 GiB free, the sweep archive is paused, so replay and history are stale or thin. Evidence: E1.

## 6. Scaling limits: tabs, users and layers before things break

| Dimension | Breaking point (measured) | What breaks first |
|---|---|---|
| Layers, laptop-class headless (software GL) | 3 heavy layers (seafloor / orbital / terrain) or "all on" | frame time (0.1–3 fps); 9–35 s long tasks; +165–360 MB of textures |
| Layers, Galaxy S24 emulation | default 14 layers under interaction: 4.4 fps, 92 % busy | main thread; enabling all layers didn't finish in 240 s |
| Tabs (default layers, one 16 GB / 4-core machine) | each tab costs ~0.85 GB browser RSS; **10 tabs = 10.1 GB**, 10th tab ready after 604 s | memory beyond ~8 tabs; CPU already saturated at 1 tab in software GL |
| WebGL contexts | 1 per tab; not reached at 10 tabs | — |
| Concurrent users, distinct viewports | **10 users: 7.5 % viewport-aircraft 502; 50 users: 56 %** | the upstream provider budget, not Node CPU (27 %) |
| Concurrent users sharing one IP | **20 users blocked within 90 s** (local; 403 for 15 min) | anti-scraping false positive |
| Upstream licence | adsb.fi reached 82 req/min at 50 users | non-commercial fallback over its limit; monetization tripwire |
| Python RPC concurrency | ≥ 8–32 simultaneous calls → most connects EAGAIN | daemon socket backlog, then subprocess storms |
| Disk | ~85 days to the 1 GiB guard at the current permanent-tier growth | sweep archive already paused (< 2 GiB) |

## 7. Unmeasured items and what's needed to measure them

| Item | Why it's unmeasured | What's needed |
|---|---|---|
| Real-GPU FPS, frame time, GPU memory (all A/B scenarios) | only SwiftShader in this container; GPU memory is estimated from WebGL allocation calls | Chrome DevTools MCP (or the `?perf=1` HUD) on the user's laptop and a real Galaxy S24 |
| Visual quality (label swimming, tile popping, glitches) during A5 | headless can't judge visuals reliably | screen recording on real hardware |
| A4: Moon + satellites together | satellites are a map layer; the space frame couldn't be driven back to the map headless | real browser |
| A6: return from space to the map | the "Fly home" control didn't hand back within 13 s headless | real browser |
| A7: plane card, replay scrub, satellite follow frame cost | aircraft render as 3D models at z ≥ 8 (not hit-testable with `queryRenderedFeatures`); the replay scrubber didn't mount headless; the ISS chip click didn't start a follow | real browser or test hooks |
| B2: real background throttling for 15 / 30 min | headless and Xvfb Chromium never background a tab; the CDP freeze didn't take effect | real Chrome with a tab actually backgrounded |
| B1/B3: per-tab native memory | `ps` RSS sums every page in the browser | one browser per tab, or Chrome's task manager |
| D1: per-collector CPU, memory, run duration in production | no Railway metrics; collectors don't time themselves | Railway metrics, or per-collector timing in an `/api/diag` probe |
| D5: retry rates in the warm 500-injection run | the hook log was lost (relative path) | rerun the injection |
| D6: production RPC queue depth over time | the daemon probe only gives the current instant | periodic `/api/diag/daemon` sampling |
| D9: why the production anti-scraping ladder didn't engage | egress through a multi-IP proxy, unverified | test from a single fixed IP |
| E1: SQLite DB size; per-hour row counts (content gaps) | not exposed; would need ~1 GB of reads | a DB size probe; offline archive scan |
| F1: Railway CPU / memory / egress over 30 days | no Railway token or CLI in the session | Railway dashboard or API token |
| F2: production downtime per deploy | no Railway deploy logs | Railway deploy logs |
| F3: Railway bill; Anthropic API spend for the analyst | billing not accessible | Railway billing page; Anthropic console |
| G: radar and GIBS upstream-to-render latency | fetched browser-direct; no server hop to time | client timestamps on tile responses |

---

The detailed sections follow (A–G), in the order of the brief.

## A. Client: the /data map

**How these were measured, and the caveat that applies to every frame number.** Headless Chromium (Playwright build 1194) ran against the live site (`https://voltradeai.com/app#/data`, v1.0.1020) through CDP: `Performance.getMetrics`, CDP network events, forced GC, and in-page wrappers installed before app code.

**Rendering here is SwiftShader**, i.e. a software GPU on 4 CPU cores. Absolute fps is therefore a worst case and far below what a laptop GPU achieves. Use the frame numbers to **rank** layers against each other and against the baseline, not as user-facing fps.

* **Main-thread ms/frame** is the CPU time the page itself spends, and is the most transferable number.
* **GPU memory** is an **estimate**: bytes passed to `texImage2D`/`texStorage2D`/`bufferData` minus deletions. Real GPU memory is UNMEASURED.
* **Viewport:** 1440×900, DPR 1, camera at (−95, 38), zoom 3.

### A1. Baseline: empty map (imagery only)

| Metric | Value |
|---|---|
| Time to interactive map | 6.8–9.3 s |
| Idle FPS / p95 frame | 59.4 / 16.8 ms |
| Pan+zoom FPS / p95 | 3.3 / 467 ms (software raster) |
| Main thread ms/frame, idle / pan | 0.21 / 25.1 |
| Long tasks > 50 ms | 0 idle |
| JS heap | 10.1 MB |
| GPU memory (estimate) | 22.6–34.6 MB textures, 69 live textures |
| WebGL contexts | 1 alive (2 created; the second is the app's capability probe, released at `datamap.tsx:733`) |
| Network | 0 req/min idle. Panning fires Esri `identify` ≈ 18–22 req/min, browser → Esri, on every `moveend` |
| rAF callbacks | 150/s: 3 independent loops at 50 fps |

### A2. Each layer alone

All 93 switchable top-level layers (plus 4 sample state grids) were each turned on alone for about 21 s and then off. Full per-layer data is in the audit run output; the costliest are below. A composite score weights idle main-thread cost, fps lost, pan cost, heap, texture and long tasks.

| Rank | Layer | Idle FPS | Idle main-thread ms/frame | Pan main-thread ms/frame | Heap Δ | GPU tex Δ (est) | Longest task | Network while panning (per user) |
|---|---|---|---|---|---|---|---|---|
| 1 | **seafloor** | **3.3** | 13.0 | **1,030** | +17.6 MB | **+165 MB** | **3.9 s** | 840/min → S3 + GEBCO WMS |
| 2 | **orbital_sats** | **9.7** | 9.6 | 75 | +15.1 MB | 0 | 0.94 s | worker, 1 Hz |
| 3 | **terrain** | 60 | 0.4 | **218** | +6.3 MB | **+224 MB** | 1.4 s | 338/min → Esri + Mapterhorn |
| 4 | forest | 16.5 | 1.1 | 53 | +1.4 MB | 0 | — | GIBS |
| 5 | aircraft | 56.8 | 1.3 | 43 | **+32.5 MB** | +0.5 | 67 ms | 6 API req per 21 s; 3 timers |
| 6 | places | 60 | 0.2 | 86 | +2.5 MB | +1.1 | 1.1 s | 165/min pmtiles |
| 7 | powerplants | 60 | 0.5 | 67 | +12.7 MB | +0.5 | 149 ms | 1 fetch |
| 8 | coal_mine_features | 60 | 0.5 | 46 | +25.8 MB | +0.2 | — | 1 fetch |
| 9 | nucleartests | 60 | 0.4 | 53 | +16.9 MB | +0.3 | 174 ms | 1 fetch |
| 10 | weather (radar) | 59.8 | 0.3 | 50 | +5.2 MB | +17.3 MB | — | **765/min → nowcoast.noaa.gov** |
| — | GIBS rasters (nightlights, aerosol, no2, floods, firetemp, biomass…) | 60 | 0.2–0.5 | 48–60 | +3–10 MB | +0.6 | — | **790–880/min each → gibs.earthdata.nasa.gov**, 11–37 aborted per 8 s pan |
| — | weather_temp / weather_wind | 60 | 0.2 | 44–58 | +3–7 MB | ≤ 1.2 | — | **~765–790/min to our `/api/data/weather/grid`** |
| — | powergrid_all | 60 | 0.2 | 52 | +2.8 MB | — | — | 62 pmtiles range requests, 5.7 MB |
| — | every other point/registry layer (~70) | 60 | 0.16–0.6 | 25–58 | +1–22 MB | ≤ 0.5 | ≤ 175 ms | 1 fetch, then a 5–60 min poll |

Two layers would not turn on:
* `vessels`: the panel had loaded "AIS socket down" at page load. It is never refreshed (see G).
* `tank_fill`: status "planned".

### A3. All layers on at once

88 of 89 top-level layers were enabled; the bulk enable took 146 s.

| Metric | Value |
|---|---|
| FPS | **0.1** (2 frames in 20 s); frame p95 **11.4 s** |
| Main thread ms/frame | 1,117 (JS 255) |
| Long tasks | 142 in about 4 min; **max 9.3 s idle, 35 s under interaction** |
| JS heap / DOM nodes / listeners | 108 MB / 8,718 / 1,138 |
| Live intervals / workers / WebGL contexts | **44** / 2 / 1 |
| GPU memory (estimate) | **360 MB textures (1,838 live) + 34 MB buffers** |
| Chromium process RSS | renderer 1.23 GB, GPU process 709 MB, total 2.43 GB |
| Network idle | 39 req/min (25 to nowcoast.noaa.gov) |

### A4. Worst-case combinations

| Combination | Idle FPS / main-thread ms | Pan FPS / main-thread ms | Heap | GPU tex (est) |
|---|---|---|---|---|
| satellites + aircraft + vessels | **3.0** / 26 | 2.3 / 106 | 64 MB | 35 MB |
| aircraft + trains + vessels | 31.7 / 2.2 | 2.7 / 48 | **123 MB** | 75 MB |
| radar + fires + nightlights + aerosol (stacked raster) | 60 / 0.3 | **1.5** / 89 | 44 MB | **109 MB** |
| Moon + satellites | UNMEASURED (see section 7) | | | |

### A5. Interaction stress (60 s pan/zoom/tilt/rotate, all layers on)

* **Frames:** 3 in 60 s, with frames up to 27 s long and one long task of 35 s.
* **Network:** 101 requests/min. Textures grew +49 MB.
* **Visual glitches:** headless can't judge visual quality reliably. In code, the pan/zoom handlers listed in C2 recompute label, satellite and marker level-of-detail and rebuild geometry on map events; those are the documented sources of swimming and popping.

### A6. Mode transitions (map → space → Moon)

* **Entering space mode:** 1.3 s the first time, 0.45 s afterwards.
* **Space view:** 59.8 fps idle at 0.19 ms/frame. **Moon:** 58.4 fps at 0.3 ms/frame.
* **Across 3 cycles:** WebGL contexts stayed at 1 alive / 2 created, heap 10–15 MB, and the estimated texture count was unchanged. **No context or memory leak** from entering space.
* **Return to the map:** UNMEASURED. The "Fly home to the live map" control did not hand back to the map within 13 s in headless; see section 7.

### A7. Inspect and follow

**UNMEASURED** (section 7). Static evidence of the cost:

* **Opening a plane card switches the viewport poll to 2 s with `fresh=1`**, on top of the fast lane (2.5–4 s) and the global feed (8 s); see C2.1.
* **Satellite follow** adds a per-frame SGP4 propagation.
* With satellites on, the page ran at 34–37 fps.

### A8. Toggling each layer on/off

* **Once each (93 layers):**
  * timers, workers and WebGL contexts returned to baseline for every layer;
  * 0 fetches in the 10 s after a layer went off;
  * leftovers: `terrain` +54 MB and `seafloor` +51 MB of texture, `weather` +16.6 MB texture, `orbital_sats` +16 MB heap, `aircraft` +3.7 MB heap.
* **10× each (15 stateful layers):** `orbital_sats` **+21 MB heap** (about 2 MB per cycle), `weather` +16.7 MB texture, `terrain` +6 MB texture. Everything else stayed within 0.5 MB, with the same intervals, workers and contexts.

## B. Multiple tabs and long sessions

Every all-layers configuration was undrivable in software GL: two all-layers tabs stopped answering CDP for 2 h. **B1–B4 therefore use the default layer set a normal visitor gets (14 layers on).**

### B1. Tabs

All tabs rendered at once; headless never backgrounds a tab, so this is the upper bound for windows side by side.

| Tabs | Per-tab FPS | Per-tab heap | WebGL contexts per tab | Chromium RSS total (renderer / GPU) | CPU cores used | Slowest tab ready |
|---|---|---|---|---|---|---|
| 1 | 2.5 | 76 MB | 1 | 1.39 GB (0.66 / 0.28) | 3.82 of 4 | 9 s |
| 3 | 0.8 | 30–57 MB | 1 | 2.73 GB (1.75 / 0.50) | 3.87 | 102 s |
| 5 | 0.4–0.5 | 29–59 MB | 1 | 4.40 GB (3.21 / 0.69) | 3.79 | 227 s |
| 10 | 0.2–0.3 | 32–58 MB | 1 | **10.1 GB** (8.25 / 1.29) | 3.8 | **604 s** |

* **WebGL context limit:** not reached. Each tab uses 1 context; Chrome's per-process limit is about 16.
* **Memory:** grows about 0.85 GB per tab and is the binding resource beyond about 8 tabs on a 16 GB laptop.

### B2. Background tab, then refocus

| Background time | Method | Fetches while hidden | First fresh aircraft data after refocus | Burst on refocus | Duplicate markers |
|---|---|---|---|---|---|
| 1 min | app hidden path (`document.hidden` forced) | 0 | 11.1 s | 3 requests | 0 |
| 5 min | same | 0 | 8.1 s | 3 requests | 0 |
| 15 / 30 min | CDP freeze | **did not take effect** (191 / 380 fetches continued) | — | — | 0 |

* **Hidden-tab gating works.** There's no backlog burst and no duplicates.
* **But refocus doesn't refresh immediately:** the aircraft layer waits for its next 8 s poll, so positions are 8–11 s behind after every tab switch.
* **Real browser background throttling over 15/30 min is UNMEASURED** (section 7).

### B3. Soak: one tab, default layers, 60 min, sampled every 2 min

| Minute | Heap | DOM nodes | Listeners | Intervals | Workers | GPU tex (est) | Long tasks (cumulative) | Aircraft "% of positions updated < 2 min" (UI) | UI age label |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 28.9 MB | 1,248 | 350 | 15 | 1 | 27.3 MB | 13 | 86 % | "data 0m old" |
| 10 | 32.6 | 1,248 | 350 | 15 | 1 | 27.3 | 84 | **7 %** | "data 0m old" |
| 12 | 26.7 | 1,248 | 350 | 15 | 1 | 27.3 | 88 | 98 % of **1,369** (from ~8,800) | "data 0m old" |
| 30 | 35.8 | 1,248 | 350 | 15 | 1 | 27.3 | 197 | 16 % | "data 0m old" |
| 42 | 36.1 | 1,248 | 350 | 15 | 1 | 27.3 | 275 | **2 %** | "data 0m old" |
| 58 | 35.0 | 1,248 | 367 | 15 | 1 | 27.3 | 382 | 73 % | "data 0m old" |

* **No JS-side leak:** heap is flat within ±4 MB, and nodes, intervals, workers, contexts and textures are constant.
* **Native memory grew:** renderer RSS went 0.97 → 1.80 GB over the hour. That is summed over the soak tab and the B2 tab, so per-tab attribution is UNMEASURED.
* **Long tasks:** about 6/min continuously.
* **Staleness is visible here:** the share of fresh aircraft swung from 2% to 98%, while the age label never changed.

### B4. Network drop: 2 min offline, then restore

* **While offline:** 23 attempts and 22 failures at the normal cadence, so no retry storm.
* **After restore:**
  * 15 intervals before and after, so no duplicate pollers;
  * the first aircraft fetch succeeded **3.0 s after reconnect**;
  * request rate 12/min before and after, with no reconnect burst (1 request in the first 10 s);
  * no duplicate markers.
* **During the outage** the aircraft row lost its "% fresh" line, but the "data 0m old" label stayed.

### B5. Galaxy S24 emulation

Settings: 412×915 at DPR 2, touch, mobile UA, **4× CPU throttle**, Chrome's "Regular 4G" (4 Mbps / 20 ms). Mobile GPU can't be emulated (still SwiftShader).

| Scenario | FPS | p95 frame | Main thread busy | Long tasks | Network |
|---|---|---|---|---|---|
| Time to map ready | 8.2 s (Regular 4G), 11.6 s (Slow 4G) | | | | |
| A1 idle, base map | 27.5 | 167 ms | 26 % | 13 in 10 s (max 198 ms) | 6 req/min |
| A1 pan/zoom | 5.7 | 317 ms | 64 % | — | 90 req/min |
| A3 default layers, idle | 9.8 | 283 ms | 39 % | 36 in 15 s | 16 req/min |
| A5 60 s stress (default layers) | **4.4** | 400 ms (max 967) | **92 %** | **235 (37.5 s of a 60 s window)** | **1,216 req/min (913 → Esri imagery tiles), 307 failed or aborted** |
| All layers on | not reachable: bulk enable **timed out at 240 s**; 0.3 fps | 4.8 s | 277 % | — | — |

## C. Connections and polling inventory

Measured at runtime by wrapping `setInterval`, `setTimeout`, rAF, `Worker`, `WebSocket`, `EventSource`, `fetch` and `getContext` before app code loaded. Mapped to source by static reading at `origin/main` 8a3f366, which is the deployed 1.0.1020.

### C1. Inventory (client, `/data`)

* **WebSocket:** none.
* **EventSource:** none.
* **Web workers:** 2.
* **Independent rAF loops:** 3 always on, more while flying or following.
* **`setInterval`:** 63 sites in `client/src`, 52 of them in `pages/datamap.tsx`.
* **Measured live intervals:** 3 with all layers off; **44 with all layers on**.

| Kind | Source | Interval | Owner | Gated on hidden tab? |
|---|---|---|---|---|
| rAF | `render/frameCore.ts` | per frame | the Law-I loop | rAF pauses |
| rAF | `datamap.tsx:757` frame recorder (started :4131) | per frame, always | WebGL-loss diagnostics | rAF pauses |
| rAF | `datamap.tsx:4215` device-tier governor sampler | per frame, always | perf governor | rAF pauses |
| rAF | `datamap.tsx:6805` `smoothFollowFrame` | per frame **whenever `orbital_sats` is on**, even with nothing followed (returns early, re-arms) | satellites | rAF pauses |
| rAF | `celestial/spaceFrame.ts:3826`, `celestialSky.ts:1034`, `MapNavCluster.tsx:295/347/361`, `FlightProfilePanel.tsx:390`, `DataWorldMap.tsx:709`, `datamap.tsx:2497` (warp) | per frame while active | space view, nav animation, profile playback | rAF pauses |
| Worker | `datamap.tsx:353/382` (blob worker, always) | — | map (1 live at baseline) | — |
| Worker | `datamap.tsx:7173` `satWorker` | 1 Hz SGP4 ticks (`satWorker.ts:339`) | `orbital_sats` | terminated on off ✓ |
| Poll (setTimeout chain) | `datamap.tsx:8954` aircraft viewport | 15 s; **2 s + `fresh=1` while a card is open** (:9128) | aircraft | **no** |
| Poll | `lib/air/globalFeed.ts` via `airFeed.pollMs()` | 8 s delta | aircraft | **no** |
| Poll | `lib/air/selectedFastLane.ts` | 2.5–4 s | selected aircraft | yes |
| Poll | `lib/air/planRouteController.ts:597` | 60 s | flight plan | yes |
| Interval | `datamap.tsx:8983` glide | ~300 ms (3.3 Hz `setData`) | aircraft | yes (skips) |
| Interval | `datamap.tsx:9448` glide repaint | — | aircraft | — |
| Interval | `datamap.tsx:1375` tracked-planes panel | 30 s | aircraft (mounted at :13729 only when on) | yes |
| Interval | `datamap.tsx:9499` vessels | 20 s | vessels | no |
| Interval | `datamap.tsx:12108` trains | 30 s | trains | yes |
| Interval | `datamap.tsx:12207` fires | 15 min | fires | **no** |
| Interval | `datamap.tsx:8463` weather wind/temp | 10 min, plus a `moveend` reload (:8462) | weather_wind/temp | **no** |
| Interval | `datamap.tsx:8341` radar | frame advance | weather | — |
| Interval | `datamap.tsx:12326/12407/12516/12610/12734/12826/12913` | 60/5/5/2/10/5/5 min | rivergauges, alerts, spaceweather, earthquakes, meteors, volcanoes, buoys | yes |
| Interval | `datamap.tsx:12938, 13030` | 10 min | shadowstats, portdwell | **no** |
| Interval | `datamap.tsx:13057 … 13551` (20 sites) | 5 min each | insider, earnings, shortvol, ats_summary, midas, secftd, fleet_utilization, grid_demand/generation, occ_volume, tff, treasury×2, fda, vehicle_complaints, bank_failures, contracts, attention, cot, graph | yes |
| Interval | `datamap.tsx:10075/10190` | 15/60 min | faa_airports, border_waits | yes |
| Interval | `datamap.tsx:3527` | 10 s | status-note sweeper | — |
| Interval | `datamap.tsx:2571` | heartbeat | boot store | — |
| Interval | `lib/layerKeeper.ts` | 250 ms | custom-layer re-add | — |
| Map-event fetch | `datamap.tsx:5441` | every `moveend` (1.2 s debounce) | Esri imagery-date `identify`, **browser → Esri** | — |
| Map-event fetch | `datamap.tsx:9059` | every `moveend` (400 ms debounce) | aircraft refetch | — |
| One-shot | `/api/data/layers` (`datamap.tsx:3547`) | **once per page load** | layer statuses + freshness | — |

### C2. Duplicates, orphans, and loops that should be event-driven

**Duplicates (the same data fetched by more than one loop):**

1. **Selected aircraft is fed by three concurrent loops while a card is open:**
   * the viewport poll at 2 s with `fresh=1` (`datamap.tsx:9128`), which re-downloads the whole viewport payload every 2 s and tightens the server's upstream TTL to 4 s;
   * the fast lane at 2.5–4 s (`selectedFastLane.ts`);
   * the global feed at 8 s.

   The fast lane (09-30) was added without retiring the July "2 s freshness" mode.
2. **Frame-interval sampling three times:** frameCore, the frame recorder (`datamap.tsx:757`) and the governor (`datamap.tsx:4215`). Measured: **150 rAF callbacks/s at the idle baseline**, i.e. 3 loops at 50 fps.
3. **Satellite propagation two ways for the followed object:** the worker's 1 Hz SGP4 plus a per-frame SGP4 in `smoothFollowFrame`.

**Orphans (running with no owning layer):**

1. `smoothFollowFrame` keeps a rAF loop alive for as long as `orbital_sats` is on, even with nothing followed.
2. No orphaned timers survived a layer turned off. All 93 layers were toggled once, and 15 heavy layers 10× each. The interval count returned to the baseline of 3, workers to 1, and there were 0 fetches in the 10 s after toggle-off.

**Leaks after toggling off (measured after forced garbage collection):**

1. **`orbital_sats`:** +21 MB of heap after 10 on/off cycles (17.5 → 38.5 MB), +15–16 MB after a single cycle.
2. **`weather` (radar):** +16.7 MB of estimated GPU texture.
3. **`terrain`:** +6 MB of texture after 10 cycles, and +54 MB not released after a single cycle.
4. **`seafloor`:** +51 MB of texture not released after a single cycle.

**Event-driven (Law I) violations — visual state or data fetched on map events:**

| Handler | What it does |
|---|---|
| `move` → `applyLod` (7081) | satellite level-of-detail |
| `move` / `idle` → `applyMarkerLod` → `setPaintProperty` (7309–7311) | marker LOD |
| `zoomend` → `applyVectors` (9067) | geometry build |
| `moveend` → aircraft reload (9059) | data fetch |
| `moveend` → weather reload (8462) | data fetch |
| `moveend` → Esri `identify` (5441) | data fetch |

The aircraft "glide" is a 3.3 Hz GeoJSON `setData` re-tile (8983), not a per-frame interpolation.

**No hidden-tab gate:** the aircraft viewport poll (8954), the global feed, vessels, fires, weather wind/temp, shadowstats and portdwell.

### C3. Overlapping fixes in history

Full history: 1,674 commits on main.

* **ADS-B freshness.** Each of these added a new path, and none removed an older one:
  * 07-08: listener stacking fix (#374);
  * 07-21: SWR "never block a poll" (#573);
  * 07-22: "ADS-B 2 s freshness" plus "stale-on-return refresh" (v1.0.471/473/478);
  * 08-11: 24/7 tracked-planes poller (#753);
  * 09-28: worldwide global feed with 8 s delta polling (#1207);
  * 09-30: fast lane plus watched lookup (#1222).

  The result is the triple feed in C2.1.
* **Satellites:**
  * 07-17: "every satellite glides on real velocity" (#513);
  * 07-18: "satellite pulse — glide dead band z3.3–8.2" (#531);
  * 07-19: smooth sat-lock with per-frame SGP4 (#540 wave);
  * 08-12: the Rendering & Motion Law names satellite pulsing again;
  * 07-31 and 08-01: far-side cull fixes (v1.0.563, #667).

  The result is two propagation paths (C2.3) and a still-event-driven LOD (`move` → `applyLod`).
* **Stale-data handling:**
  * `feedDeadAir.ts` documents two prior failed liveness fixes (08-06, 08-11) before the archive-clock design;
  * `spaceWeather.ts` "everSucceeded honesty flag" (09-20).

  Server-side flags exist, but the client never re-reads them after page load (`/api/data/layers` once).

## D. Backend — Node/Express, Python engine, RPC

### D1. Background collectors (inventory)

All collectors run **inside the single Node process** (`dist/index.cjs`, pid 1 on Railway). The Python daemon (`voltrade_daemon.py`, separate process, RSS 187 MB measured via `/api/diag/daemon`) only serves bot RPC; it runs no data collectors.

Static inventory: 81 `setInterval` sites in `server/*.ts`. Every collector uses the same *eager boot* pattern (`refreshX(); setInterval(refreshX, intervalMs)`, e.g. `server/cftcCot.ts:349-353`), so **all of them fire within the first seconds of every boot**.

| Cadence | Collectors (server/…) |
|---|---|
| 30 s | globalAircraft evict tick, trackedPlanes poll |
| 60 s | globalSweep type lane (adsb.lol `/v2/type`), bot.ts:7063, routes.ts:1224 |
| 2 min | globalScopes (adsb.lol mil/ladd/pia) |
| 5 min | streamsInventory, bot.ts:7224 |
| 10 min | nwsAlerts, spaceWeather, shadowstats/portdwell folds (routes.ts:4286/4514/4548), routes.ts:2061 |
| 15 min | edgarForm4, edgar13f, sec8kEarnings, usgsQuakes, faaStatus, gdeltEvents, entityGraph, trackedPlanes trace backfill |
| 30 min | nasaFirms, ndbcBuoys, usgsVolcanoes, archiveOffload, routes.ts:1332 |
| 1 h | cbpBorderWait, optionsChainArchive, usgsWater |
| 2 h | euDayAheadPrices, euGenerationMix, euLoad, gridDemand, gridGeneration |
| 3–4 h | airQuality, ambientRadiation, cboeVix, occVolume |
| 6 h | appStoreRankings, dtccSwaps, euMacro, fdaEvents, finraQuery, finraShortVolume, fmcsaOutOfService, fredMacro, gridStress, nrcReactorStatus, satellites, settlementStress, treasuryAuctions, treasuryDts, usaSpending, meteors, routes.ts:1340 |
| 12 h | cftcCot, cftcTff, cropConditions, epaCamd, fdicBanks, nhtsaComplaints, secFtd, wikiAttention, aeroCharts prefetch |
| 24 h – 7 d | censusImports, droughtMonitor, githubOrgActivity, secMidas, superfund, waterViolators |
| push | vesselStream (aisstream.io WebSocket), swimConnector (FAA SWIM over Solace) |

Per-collector CPU/memory/run duration in production: **UNMEASURED** (no Railway metrics or per-collector timing in this session). Measured in aggregate instead:

* **Local cold boot (same bundle, hook-instrumented, `nice 19`):** 178 outbound calls to 48 hosts in the first 19 s; RSS 404 → 760 MB in 16 s; CPU 94–157 %; event-loop stalls of 2.8 s (t+9 s), 3.7 s (t+14 s), 5.3 s (t+24 s).
* **Production (audit log `EVENTLOOP-LAG`, threshold 500 ms):** every one of the 7 boots in the last 24 h produced its worst stall **1.2–2.1 min after boot**: 3.2 s, 3.2 s, 9.5 s, 4.0 s, 3.4 s, 3.2 s, 2.9 s.
* **Collector alignment.** Because every collector starts at boot, their timers stay phase-locked. The `time` stamps of alerts, earthquakes, fires and space weather were all exactly `18:06:24`, two hours after the `16:06:23` boot.

### D2. Event-loop lag on Express

| Condition | Measured | Source |
|---|---|---|
| Idle steady state (local, 0 users) | p99 ≈ 18 ms, max 550 ms per 90 s window | hook `monitorEventLoopDelay` |
| 1 / 10 / 50 users (local) | max 689 / 882 / 691 ms | same |
| Boot storm (local) | 2.8–5.3 s | same |
| Boot storm (prod) | 2.9–9.5 s, every boot | `/api/diag/audit?type=EVENTLOOP-LAG` |
| Prod steady state | 0.5–0.9 s stalls at irregular times, about 1 every 80 min (18 entries over ~25 h outside the boot windows) | same |
| Cold archive scan (prod, my `/api/data/aircraft/hexes` hit) | 546 ms stall logged at 16:52:06, same minute as a 50 s cold scan | same |

### D3. Upstream fan-out (per server vs per client)

The hook counted every outbound call on a local server; 90 s windows, users with a selected plane (fast lane) on user 0.

| Clients | Outbound/min (all) | adsb.lol/min | adsb.fi/min | Notes |
|---|---|---|---|---|
| 0 | 25.3 | 2.0 | 0 | collectors only |
| 1 | 63.3 | **30.6** | 5.3 | one watched plane ≈ +28 adsb.lol calls/min (fast lane, 2.5 s) |
| 5 | 85.3 | 36.0 | 22.7 | viewport tiles shared, overflow spills to adsb.fi |
| 10 (distinct IPs) | 84.0 | 27.5 | 31.0 | governor caps adsb.lol; adsb.fi takes the rest |
| 50 (distinct IPs) | 154.9 | 29.0 | **81.9** | adsb.fi above its ~1 req/s limit |

* **Shared per server (cached):** all collectors, `/api/data/aircraft/global` (snapshot), vessels (one AIS socket), trains, `/api/data/*` registry endpoints.
* **Per client (scales with users):**
  * `/api/data/aircraft` viewport fetches, one upstream call per distinct 0.1° bbox per 30 s (or 4 s with `fresh=1`);
  * `/api/data/aircraft/live/:hex` (fast lane), one upstream call per watched hex per 2.5–4 s.
* **Per client, browser-direct, never touching our server** (Law II violation; the provider sees every user):
  * Esri `server.arcgisonline.com` (imagery tiles) and `services.arcgisonline.com/.../identify` (on every `moveend`, datamap.tsx:5413/5441);
  * NOAA `nowcoast.noaa.gov` radar WMS (datamap.tsx:8320, 25 req/min measured with all layers on);
  * NASA `gibs.earthdata.nasa.gov`;
  * `s3.amazonaws.com/elevation-tiles-prod` (elevation.ts:26, datamap.tsx:5550/5575);
  * `tiles.mapterhorn.com`;
  * `demotiles.maplibre.org` (glyphs);
  * `celestrak.org` (satcat link).

### D4. Rate limits and failover

| Provider | Published/observed limit | Actual usage measured | Status |
|---|---|---|---|
| adsb.lol (primary, ODbL, only commercial-lawful) | dynamic; returns 429 | prod sweep: 59/166 = **36 % of steps 429** (16:44); governor `ceiling_rps 1, bg_rps 0.4` | over budget; global disc sweep effectively dead (5 of 782 discs ever visited, oldest disc 3,642 s) |
| adsb.fi (non-commercial fallback) | ~1 req/s | 7.5 → 31 → 82 req/min at 1/10/50 users | exceeded at 50 users; **licence not commercial-lawful** — serving it to paying users trips the MONETIZATION TRIPWIRE |
| airplanes.live (non-commercial fallback) | ~1 req/s | 1–1.3/min | fine |
| OpenSky | 4,000 credits/day | 0 (not configured: `OPENSKY_CLIENT_ID` unset) | inactive |
| aisstream.io | one socket | 1 socket, 707k frames since boot | fine |
| CelesTrak | — | server path firewalled from Railway (`/api/data/satellites` count 0) | known (R17) |

**Failover staleness (measured, warm server, aircraft providers forced to time out for 6 min):**
* **Viewport `/api/data/aircraft`:** served the last cache with `stale: true` for the whole outage. The same 787 aircraft were frozen, so data was up to 6 min+ old, but it was **flagged**.
* **`/api/data/aircraft/global`:** kept serving 3,261 rows with **no top-level stale flag**. Rows age silently until the 600 s eviction, then the count collapses; 2.5 min after recovery the region held 466–511 rows against 3,200 before.
* **Fast lane `/live/:hex`:** returned `200 {}` after ~3.3 s every call. **Silent.**

### D5. Upstream outage behaviour

| Scenario (local, measured) | Blocks other feeds? | Retry storm? | Serves stale silently? |
|---|---|---|---|
| Aircraft providers timeout from cold boot | No (earthquakes 2–3 ms throughout) | No: 12.6 attempts/min to the dead hosts, backoff respected | Viewport: **no**; the first request blocks 15.08 s, then a 502 with an honest message. Global: **yes**, 200 with 0 rows. Fast lane: **yes**, 200 `{}`. |
| Warm server; aircraft timeout + SWPC/USGS/NWS/SEC/digitraffic/entur/OCC/FIRMS return 500 for 6 min | No | Retry counts UNMEASURED for this run (hook log lost to a relative-path error) | earthquakes / alerts / spaceweather / fires: kept serving the pre-outage payload with **no stale flag**. Trains: dropped to `count 0` during the outage and took **2.5 min after recovery** to refill. Viewport aircraft: still `stale: true` 2.5 min after recovery (providers in backoff). |

### D6. Unix-socket RPC to the Python daemon

* **Latency, 1 caller:** p50 5.5 ms for `health` (local daemon).
* **Under concurrency: connects fail rather than queue.** Python's `socketserver` default listen backlog (5) is the limit; failed connects return `EAGAIN`.
  * Python client: 25/32 failed at concurrency 8, 52/64 at 16, 119/128 at 32.
  * Node client written the same way as `server/bot.ts:101 pythonRpc`: 1/8, 0/16, **24/32 and 52/64** failed.
* **What a failed connect does:** `pythonCall` (bot.ts) falls back to spawning a Python subprocess per call. That costs seconds of CPU and hundreds of MB to import pandas/lightgbm. The heavy methods (`run_full_scan`, `scan_market`, `manage_positions`) fail outright instead.
* **Daemon down:** Node gets `ECONNREFUSED` within 2 ms, which also triggers the subprocess fallback. `run_with_daemon.sh` restarts the daemon after `sleep 2`; locally the socket was back about 2 s later.
* **In production:** `/api/diag/daemon` showed 1 active dispatch and RSS 187 MB of 1,024 MB. The production queue depth over time is **UNMEASURED**.

### D7. Cold start and restart

* **Restart frequency (prod audit log):** 7 SIGTERM restarts in 24 h (10-01 16:05, 20:19; 10-02 00:24, 02:43, 11:04, 13:17, 16:06).
  * 4 of the 7 redeployed the **same version** (1.0.1020).
  * `railway.json` / `railway.toml` have no `watchPatterns`, so every commit to `main` redeploys, including docs-only `research/*` log commits.
  * There were 237 main commits in 31 days, 7.6 per day.
* **Time to ready:** `/api/health` answered 200 about 5 s after spawn locally.

Time until each layer is fresh again:

| Layer / cache | Time to fresh after a restart (measured) |
|---|---|
| Global aircraft snapshot | 865 rows at boot+8 min vs 12–14 k steady; ~40 min to 13 k |
| `portdwell`, `shadowstats` (default ON) | `warming_up: true` until **~60–64 min after boot** (blank layer); measured 17:05 → 17:10 |
| `waterviolators` | still `warming_up` at boot+80 min |
| `/api/data/aircraft/hexes` (public) | first request 50.1 s (7.2 MB) |
| `/api/data/fleet-utilization` | first request timed out at 60 s; 0.14 s once warm |
| `/api/data/platform/stats` | first request timed out at 90 s; 0.16 s once warm |
| `/api/data/aircraft/trips/:hex` | 16.3 s cold, 0.13 s warm |

* **Collectors don't miss windows** (eager boot re-fetches).
* **In-memory state is lost on every restart.**
* **Crash-guard trap (`server/crashSafeRefresh.ts`).** The guard records an "attempt" marker before each long archive fold, so a normal SIGTERM redeploy during a fold looks like a crash. The next boot then **skips that fold for 6 h** (`REFRESH_CRASH_COOLDOWN_MS`, routes.ts:4261).

### D8. Concurrent users

Local server; each simulated user runs the default-on polling mix with its own viewport and distinct IP:

| Users | Requests/s | p50 | p95 | p99 | Max | `/api/data/aircraft` 502s | Server CPU | Max loop lag | RSS |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.77 | 143 ms | 760 ms | 1.1 s | 1.1 s | 0 | 16 % | 689 ms | 906 MB |
| 10 | 4.07 | 9 ms | 497 ms | 1.6 s | 2.3 s | 9 / 120 (7.5 %) | 21 % | 882 ms | 924 MB |
| 50 | 18.7 | 6 ms | 455 ms | 1.6 s | 5.9 s | **338 / 600 (56 %)** | 27 % | 691 ms | 951 MB |

* **What breaks first is the upstream budget, not Node CPU.** Distinct viewports each need their own adsb.lol call, the governor caps adsb.lol at ~1 req/s, and the overflow fails or spills onto adsb.fi. **The viewport aircraft layer goes stale or empty for the majority of users somewhere between 10 and 50 concurrent users.**
* **Same IP:** 20 users sharing one IP (office NAT, or one person with many tabs) were **blocked by anti-scraping within 90 s**: ~600 × 403 and 15 × 429, with the block lasting 15 min.

### D9. Bot / scraper traffic

* **Protection that exists:**
  * `server/antiScraping.ts` scores each IP: 429 at score ≥ 100, 403 for 15 min at ≥ 200;
  * velocity starts scoring above 150 req/10 s per IP and breadth above 80 distinct endpoints/10 s;
  * plus honeypots.
  * The client IP comes from the leftmost `X-Forwarded-For` (`trust proxy: true`), which a client can spoof.
* **What is unprotected:**
  * no global concurrency limit and no per-endpoint cost limit;
  * cold-scan endpoints are public: `/api/data/aircraft/hexes` (50 s, 7.2 MB) and `/api/data/fleet-utilization` / `/api/data/platform/stats` (60–90 s) can be hit once per IP after each restart.
  * A rotating-IP scraper never accumulates a score.
* **Production hammer test (measured 22:41Z):**

| Phase | Requests | Result | Latency (p50 / p95 / max) |
|---|---|---|---|
| Baseline, ~2 req/s | 20 | all 200 | 1,181 / 1,238 / 1,238 ms |
| **~20 req/s for 30 s from this session's IP** | **600** | **599 × 200, 0 × 429, 0 × 403**, 1 network error | 466 / 969 / 1,390 ms |
| Immediately after, same IP, `/api/health` | 20 | all 200; **no block engaged** | 1,215 / 1,429 / 1,429 ms |
| ~20 req/s with a different spoofed `X-Forwarded-For` per request | 400 | all 200 | 736 / 1,082 / 1,354 ms |

* **Production protection did not engage.** The anti-scraping ladder didn't trip at 200 req/10 s from one client, although the same ladder blocked the local 20-user test.
  * The likely reason is that this session's egress goes through a multi-IP agent proxy, so the server saw several source IPs. That was not verified (UNMEASURED).
  * Either way, spoofing `X-Forwarded-For` defeats the per-IP identity outright, so the ladder can't stop a determined scraper.
* **Impact on real users:** no latency degradation was visible during the 20 req/s burst against a cached endpoint. The real exposure is the cold-scan endpoints after a restart, where one request costs 50–90 s of server work.

## E. Storage and archives

### E1. Size and growth

Source: `/api/data/archive/stats` and `/api/data/archive/offload-status` (16:45Z).

* **Railway volume:** 5 GB, with **1.81 GB free**. That is below the 2 GiB hot-tier target, so it is flagged `underPressure: true`.
* **Archive directory:** 2.84 GB across 61 streams.

| Stream | Size now | Growth (measured span) | Retention | 30 d | 90 d | 365 d |
|---|---|---|---|---|---|---|
| vessels (raw, hourly) | 943 MB local (22 d) + 1,405 MB R2 | 42.9 MB/day | 30 d rolling (local ≥ 22 d, then R2) | plateau ≈ 1.3 GB | plateau | plateau |
| aircraft (raw, hourly) | 239 MB local + 351 MB R2 | 10.9 MB/day | 30 d rolling | plateau ≈ 330 MB | plateau | plateau |
| vessels_tracks (permanent rollup) | 259 MB / 70 d | 3.7 MB/day | permanent | +111 MB | +333 MB | +1.35 GB |
| aircraft_tracks (permanent) | 111 MB / 70 d | 1.6 MB/day | permanent | +48 MB | +143 MB | +580 MB |
| fires | 181 MB / 91 d | 2.0 MB/day | permanent | +60 MB | +179 MB | +727 MB |
| buoys | 103 MB / 87 d | 1.2 MB/day | permanent | +35 MB | +106 MB | +432 MB |
| finraweekly / earnings8k / filings13f / others | ~0.9 MB/day combined | — | permanent | +27 MB | +81 MB | +330 MB |
| One-time backfills (occvolume 391 MB, secmidas 150 MB, dtccswaps 125 MB, finrashortvol 97 MB) | 763 MB | ~0 | permanent | — | — | — |
| **Permanent tier total** | — | **≈ 9.4 MB/day** | — | **+0.28 GB** | **+0.85 GB** | **+3.4 GB** |

* **When the volume fills.** With 1.81 GB free and the permanent tier growing about 9.4 MB/day, the volume reaches the **1 GiB `globalScopesGuardBytes` floor in roughly 80–85 days** and fills completely in about 190 days. That assumes the 30-day raw window keeps plateauing; verified rolling eviction requires R2.
* **R2:** 1.76 GB / 1,258 objects. `estimatedSteadyStateBytes` 2.16 GB fits inside the free 10 GB, so cost is $0.
* **BigQuery / GCP Cloud Storage: not used.** No code reference was found in `server/` or the Python modules. The SQLite DB on the volume was not sized (UNMEASURED).
* **Measured side effect of the volume pressure.** The global aircraft sweep archive is gated on ≥ 2 GiB free (`globalAircraft` coverage block). It is **paused**: `fixes_skipped_low_disk_total` 197,171 at 16:44 and 374,787 at 17:50, with **0 lines archived**. Every worldwide aircraft position collected since this boot has been dropped from the archive. Only viewport/tracked fixes are written, which is why a 1-hour replay over Austin returned 15 aircraft with ~4 points each.

### E2. Archive read latency

Measured on production, cold then warm:

| Read | Cold | Warm | Notes |
|---|---|---|---|
| Replay window, Austin 2°×2°, 1 h, step 60 | 0.38 s | 0.18 s | 15 aircraft / 58 points |
| Replay window, CONUS, 1 h | 0.58 s | — | 2,000 of 3,579 aircraft (cap) |
| Replay window, Austin, 24 h, step 300 | 2.39 s | — | |
| Replay window, CONUS, 24 h, step 900 | 4.76 s | — | 2,000 of 18,647 (cap) |
| Replay window, vessels, Europe, 1 h | 0.79 s | — | 2,000 of 15,240 (cap) |
| `/api/data/aircraft/trips/:hex` | **16.3 s** | 0.13 s | per-hex full-archive scan; there is no per-hex index (earlier session measured 113.8 s) |
| `/api/data/aircraft/hexes` | **50.1 s** | 0.61 s | full archive scan |
| `/api/data/fleet-utilization` | **> 60 s (timeout)** | 0.14 s | |
| `/api/data/platform/stats` | **> 90 s (timeout)** | 0.16 s | |
| shadowstats / portdwell fold | **~60 min** to first result | 10 min refresh | 7-day AIS archive fold at boot |
| Space-weather history, chokepoint stats | `/api/data/spaceweather` 0.19 s; `/api/v1/stats/portdwell` needs auth | — | the authenticated endpoint is UNMEASURED |

The archive is flat JSONL/gz hour and day files with no index. Every by-entity read (`trips`, `hexes`, fleet utilisation) is a linear scan of the window, and its cache dies on every restart. **The missing index is a per-hex (entity) index on the aircraft/vessel archives.**

### E3. Gaps and duplicates (last 30 days)

Hour-file presence was checked per day via `/api/diag/archive` (local window, 2026-09-03/11 → 10-02):

| Stream | Expected hours | Missing | Where |
|---|---|---|---|
| aircraft | 521 | 77 (14.8 %) | **all** between 2026-09-10 17:00 and 09-16 01:00 |
| vessels | 521 | 84 (16.1 %) | same window |
| trains | 713 | 91 (12.8 %) | same window |

* **The window matches the 09-10 → 09-16 `/api/health` deploy-gate outage** (KNOWN STATE). **No hour is missing since 2026-09-16 02:00.**
* **Duplicates:** one hour (2026-10-02 15:00) existed as both `.jsonl` and `.jsonl.gz`, a transient state of the compress step.
* **Content gaps inside present hours.** The paused sweep archive and the per-boot refill of the global snapshot are not visible as missing files. Per-hour row counts were not measured (UNMEASURED: it needs a full read of ~1 GB).

## F. Infra and cost

### F1. Railway CPU / memory / egress, last 30 days

**UNMEASURED.** There is no Railway API token or CLI in this session. What was measurable from the service itself:

* **Node heap:** 862 MB at uptime 38 min; then a sawtooth of 0.94–1.19 GB from 55 min to 2 h of uptime. **RSS** 1.22 → 1.57 GB (servermon, 60 s samples). The heap limit is 6,192 MB and the cgroup limit 22.9 GB, so there's no memory pressure.
* **Python daemon:** 187 MB RSS.
* **`/api/data/pipeline-health-dashboard`:** 7-day "uptime" is 44.6 %. That figure is driven entirely by the `bot_liveness_dark` flag (62 of 112 samples), not by the process being down.

### F2. Impact of routines and GitHub Actions

* **Restarts:** 7 production restarts in 24 h, all graceful SIGTERM redeploys, and 4 of them redeployed an identical version.
* **Commit volume:** 237 commits to main in 31 days (7.6/day). On 10-01 and 10-02, 6 of the 11 main commits were `[NO-ACTION]` session logs that touch only `research/*`.
* **What each restart costs, measured above:**
  * a 3–9.5 s event-loop stall;
  * all in-memory aircraft/vessel state lost (~40 min to refill the global snapshot);
  * 60+ min of blank `portdwell`/`shadowstats` layers;
  * 50–90 s first-hit timeouts on archive endpoints;
  * risk of the 6 h crash-guard skip.
* **Downtime:** about 5 s of HTTP unavailability per deploy locally. Production downtime per deploy is **UNMEASURED**, because Railway deploy logs weren't available.

### F3. Monthly cost by source

| Source | Monthly cost | Basis |
|---|---|---|
| Railway | **UNMEASURED** | needs the Railway billing page |
| Cloudflare R2 | $0.00 | free tier; 2.16 GB steady state (`/api/data/archive/offload-status`) |
| GCP (BigQuery/GCS) | $0.00 | not used by the codebase |
| RunPod | $0.00 this month | ledger `datacore/runpod/ledger.jsonl`: $6.88 lifetime across 20 jobs; last job 2026-07-10 |
| API plans | $0.00 measured | adsb.lol, adsb.fi, airplanes.live, aisstream, FAA SWIM, NOAA/USGS/NASA/SEC are free; OpenSky not configured; ADSBx/paid plans not active. Whether the analyst's `ANTHROPIC_API_KEY` is set and what it spends is UNMEASURED |

## G. Data freshness end to end

Hop latencies per layer. "Upstream" is the provider's own publish cadence (from its documentation or payload); the other hops were measured or read from code.

| Layer | Upstream → collector fetch | Collector → storage/cache | Cache → API | API → render (client poll) | Measured staleness | Stale hop |
|---|---|---|---|---|---|---|
| Aircraft, global feed | receivers → adsb.lol ~1–5 s | **type lane every 60 s, 36 % of calls 429**; disc sweep dead | snapshot, rows evicted at 600 s | client delta poll 8 s, glide 3.3 Hz | server rows: p10 46 s, **p50 47 s**, p90 351 s, max 600 s; "fresh < 2 min" swings 11k ↔ 2.2k per minute | **collector hop** (60 s lane + 429s) |
| Aircraft, viewport | adsb.lol point query | SWR cache TTL 30 s (4 s with `fresh=1`) | — | 15 s poll (2 s with a card open) | up to ~45 s; under provider failure `stale: true` for minutes | cache TTL + provider budget |
| Aircraft, selected (fast lane) | adsb.lol `/v2/hex` | none (coalesced) | — | 2.5–4 s | p50 150–170 ms per request; fresh when the provider answers, silently `{}` when not | provider |
| Vessels | aisstream push, real time | snapshot TTL 30 s | — | client 20 s poll | last message −1.5 s (live) | — (healthy) |
| Trains | digitraffic / entur | background tick | — | 30 s | refills 2.5 min after an upstream blip | collector backoff |
| Earthquakes | USGS feed ~1 min | **15 min** | in-memory | 2 min | ≤ 17 min | collector cadence |
| NWS alerts | api.weather.gov ~1 min | **10 min** | — | 5 min | ≤ 15 min | collector |
| Space weather | SWPC 1–5 min | **10 min** | — | 5 min | collector 3.5 min old at probe | collector |
| FIRMS fires | NASA NRT ~3 h latency | 30 min | — | 15 min | registry 0.7 h | upstream |
| Radar (nowCOAST) | NOAA ~5 min | **none: browser direct** | — | client refresh | not measurable server-side | browser fetch, per client |
| GIBS rasters | NASA daily | **none: browser direct** | — | — | — | — |
| portdwell / shadowstats | own AIS archive | 7-day fold, 10 min | `warming_up` | 5–10 min | **blank 60+ min after every restart** | fold at boot |
| `shortvol` (panel label) | FINRA daily | 6 h | — | 5 min | **label says 88.8 days old; data is 1 day old** | label computation (`streamsInventory.ts:158-164`) |
| Layer panel status/ages (all) | — | — | `/api/data/layers` | **fetched once per page load** (`datamap.tsx:3547`) | ages never advance; a feed marked "down" at load stays disabled until reload | client |

Freshness labels are present for only 16 of the 256 registry layers (Law V: "a layer that cannot say how old it is may not claim to be live").
