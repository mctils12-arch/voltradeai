# Client + data/research audit — /home/user/voltradeai (2026-10-27)

Method: static import graph of `client/src` (script `scratchpad/importgraph.py`, output `client_graph.json`; resolves `@/`, `@shared/`, `@assets/`, relative, TS `./x.js`→`x.ts`, `import()`, `?raw`, `new URL(..., import.meta.url)`), reachability from `client/src/main.tsx`; cross-checked with `npx knip` (`client_knip_out.txt`) and `npx tsc --noEmit` (`client_tsc_out.txt`, 6 errors = exactly the pinned `ci/tsc_baseline.txt` TOTAL 6; 2 of them in `client/src/pages/datamap.tsx` lines 3778, 15845). Filename greps for assets/data across `client/src`, `server/`, `scripts/`, `shared/`, `datacore/layers.json`, root `*.py`. Read-only; no tracked file modified.

## 1. Entry points

| Entry | How | Notes |
|---|---|---|
| `client/index.html` → `client/src/main.tsx` → `client/src/App.tsx` | Vite root = `client/`, `base: "/"`, outDir `dist/public` | Inline boot splash; fonts from Google Fonts; favicons `/favicon*.{ico,png}` |
| `client/public/moon.html` (72 KB, standalone) | Static page, three.js r0.185.1 vendored at `client/public/vendor/three.module.min.js` (+ `three.core.min.js` which the module imports internally) | Opened from `client/src/components/MapNavCluster.tsx`; imagery from `/space/*`; streams LROC tiles from NASA Moon Trek at runtime (Law II §8 "runtime never touches upstream WMTS" is not met by this page — stated in its own header) |
| `client/src/pages/landing.tsx` | imports `_landing_body.html.txt`, `_landing_styles.css.txt`, `_landing_script.js.txt` via `?raw` and injects them | Landing is a raw-HTML island inside React (see Duplicates) |

No other HTML entries. No `React.lazy` pages: the only lazy chunks are `AnalystPane`, `TimeScrubber` (from datamap.tsx) and `DataWorldMap` (from analyze.tsx). Built bundle: `dist/public/assets/index-*.js` = 1.89 MB (datamap.tsx is inside it — not code-split), `maplibre-gl-*.js` 1.05 MB, CSS 282 KB, 12 asset files total. `dist/` (152 MB, gitignored) is a full copy of `client/public` (80 MB tiles etc.).

## 2. Routes table

### wouter routes (`client/src/App.tsx` → `AppShell`)

| Path | Component | File |
|---|---|---|
| `/` | LandingPage | `client/src/pages/landing.tsx` |
| `/login` | LoginRoute → LoginPage | `client/src/pages/login.tsx` |
| `/pricing` | PricingPage | `client/src/pages/pricing.tsx` |
| `/about` | AboutPage | `client/src/pages/about.tsx` (uses `_MarketingPage.tsx`) |
| `/newsletter/:slug` | NewsletterIssuePage | `client/src/pages/newsletter-issue.tsx` |
| `/newsletter` | NewsletterPage | `client/src/pages/newsletter.tsx` |
| `/careers` | CareersPage | `client/src/pages/careers.tsx` |
| `/contact` | ContactPage | `client/src/pages/contact.tsx` |
| `/developers` | DevelopersPage | `client/src/pages/developers.tsx` |
| `/apikeys` | ApiKeysPage | `client/src/pages/apikeys.tsx` |
| `/admin` | AdminPage | `client/src/pages/admin.tsx` |
| `/app`, `/app/:rest*` | AppDashboard → Home | `client/src/pages/home.tsx` |
| (fallback) | LandingPage | — note: `client/src/pages/not-found.tsx` is NOT used; a bare `https://voltradeai.com/data` hits the SPA fallback and renders the landing page, not the map (no server redirect found in `server/static.ts`/`server/index.ts`) |

### Hash tabs inside `/app` (`client/src/pages/home.tsx` TABS, `#/<tab>`)

| Hash | Page file |
|---|---|
| `#/analyze[/options|smart-money|structure|etf-builder]` | `pages/analyze.tsx` (+ `InsightsView.tsx`, `ETFBuilderView.tsx`, `components/DataWorldMap.tsx` lazy) |
| `#/research` | `pages/research.tsx` |
| `#/scanner` | `pages/scanner.tsx` (+ `components/SectorHeatmap.tsx`) |
| `#/news` | `pages/news.tsx` |
| `#/watchlist` | `pages/watchlist.tsx` |
| `#/data[/<overlay>]` | `pages/datamap.tsx` (16,628 lines) |
| `#/planner` | `pages/planner.tsx` |
| `#/taxes` | `pages/taxes.tsx` |
| `#/bot` (owner-only), `#/bot/pane/<id>` popout | `pages/bot.tsx` → `client/src/bot/BotPage.tsx` |

### Overlays inside `#/data/*` (datamap.tsx `hashchange` listener at line 3538; 50 hash keys)
`streams, graph, signals, quality, pipeline-health, filings, filings13f, earnings, contracts, insider(history via api), short-volume, short-interest, ftd, midas, ats-summary, occ-volume, dtcc-swaps, cot, tff, treasury-auctions, treasury-dts, fred-macro, eu-macro, eu-power, vix-term-structure, bank-failures, fda-events, clinical-trials, vehicle-complaints, fmcsa-oos, appstore-rankings, github-activity, attention, wiki-attention-signal, gnss-integrity, air-quality, drought, crop-conditions, methane-hotspots, fires-near-facilities, facility-events, plant-operations, nrc-reactor-status, grid-demand, grid-generation, grid-stress, fleet-utilization, port-dwell, imports, jodi-oil-stocks, un-comtrade` → one file each under `client/src/pages/` (all imported eagerly by datamap.tsx).

## 3. Reachability counts

| Bucket | Files | Lines |
|---|---|---|
| `client/src` total | 423 | ~111 k |
| non-test modules | 283 | 94,811 |
| **ACTIVE** (reachable from main.tsx) | **233** | **88,633** |
| **DEAD** (unreachable from main; knip agrees on all 50) | **50** | **6,178** |
| of which reachable only via tests | 2 (`render/tileCore.ts`, `render/perfHud.ts`) | 1,157 |
| test files | 140 | 22,274 |

Per-directory: pages 77 active / 1 dead; lib (flat) 44/0; lib/celestial 20/0; lib/air 26/0; lib/orbital 23/0; bot 19/0; render 6/2; components 14 active / **46 dead** (45 of them `components/ui/*` + `TradingActivity.tsx`); hooks 1/1.

## 4. ACTIVE (grouped by feature)

- **Shell**: `main.tsx`, `App.tsx`, `index.css` (4,855 lines), `pages/home.tsx`, `lib/queryClient.ts`, `components/ui/toast.tsx`, `components/ui/toaster.tsx`, `hooks/use-toast.ts`, `lib/utils.ts` (6 lines, `cn`).
- **Marketing/account**: `pages/landing.tsx` (+3 `_landing_*.txt`), `_MarketingPage.tsx`, `about/careers/contact/pricing/newsletter/newsletter-issue/developers/apikeys/admin/login.tsx`.
- **Trade world (/app tabs)**: `analyze.tsx` (2,602), `InsightsView.tsx`, `ETFBuilderView.tsx`, `components/DataWorldMap.tsx` (topojson canvas globe for Analyze), `scanner.tsx`, `components/SectorHeatmap.tsx`, `research.tsx`, `news.tsx`, `watchlist.tsx`, `planner.tsx`, `taxes.tsx`, `lib/tradeChartGuards.ts`, `lib/sectorHeatmapGuards.ts`, `lib/spreadsEmpty.ts`, `lib/tabResume.ts`.
- **Bot cockpit (`client/src/bot/`, 19 modules)**: `BotPage.tsx`, `panels.tsx`, `useBotData.ts`, `chart/{BotChart.tsx,draw,levels,scales}.ts`, `derive/director/flipGeometry/format/glossary/help/live/lookup/pageState/popout/watch/zoneModel`. Uses `render/frameCore.ts` + `render/zoomInput.ts` (the only non-map consumers of the render contract).
- **/data map core**: `pages/datamap.tsx` (95 useState, 154 useEffect, 98 addLayer, 80 addSource, 80 fetch calls, 146 import lines — a monolith), `components/{MapNavCluster,TimeScrubber,AnalystPane,SatFinder,FlightProcedures,FlightProfilePanel,PlannedRoute,RunwayStatus,AtcFrequencies,IntentStatus}.tsx`, `pages/streams.tsx`, `pages/graph.tsx`, `pages/signalLadder.tsx`, `pages/qualityDashboard.tsx`, `pages/pipelineHealthDashboard.tsx`, 42 stream-overlay pages.
- **Map libs (flat `lib/`)**: `mapIcons.ts` (1,463 — icon registry, SYMBOLS NOT DOTS), `mapBaseConfig`(render), `layerRegistry.ts`, `layerKeeper.ts`, `panelLayout.ts`, `units.ts` (UNITS PREFERENCE), `deviceTier.ts`, `lod.ts`, `glElev.ts`, `elevation.ts`, `bathymetry.ts`, `seafloorV2.ts`, `oceanBasemap.ts`, `globeAtmosphere.ts`, `drapeOrder.ts`, `terrainExag.ts`, `cameraRig.ts`, `gibs.ts`, `timeAxis.ts`, `gridMaster.ts`, `aeroCharts.ts`, `faaAirports.ts`, `cbpBorderCrossings.ts`, `cameoEvents.ts`, `meteors.ts`, `nukeCodes.ts`, `nrcReactorStatus.ts`, `mmsiFlag.ts`, `portDetail/portDwellView/portImports.ts`, `livePoints.ts`, `mapInteractions.ts`, `mapScale.ts`, `resilientLoad.ts`, `longTasks.ts`, `blackbox.ts`, `apiMetaGuard.ts`, `aqIndex.ts`.
- **Aircraft stack `lib/air/` (26)**: `airLayer`, `airGlide`, `airFilter`, `globalFeed`, `selectedFastLane`, `watchedLookup`, `watchlist`, `planeIdentity`, `intentStatus`, `trackModel`, `flightTrackLayer`, `breadcrumbs`, `flightPlan`, `planCurtainLayer`, `planRouteController`, `terrainFollow`, `groundDatum`, `fleetModel`, `fleetReplayLayer`, `fleetReplayController`, `replayCamera`, `procedures`, `procedureLayer`, `pdfPlate`, `tzCrossings`, `closeApproachView`.
- **Orbital stack `lib/orbital/` (23)**: `tle`, `propagate` (inline SGP4), `satDerived`, `geometry`, `occlusion`, `satBuffer`, `satWorker`, `gpWorker`, `satcatWorker`, `gpCache`, `satLayer`, `arcLayer`, `orbitArc`, `modelLayer`, `model3d`, `realMesh` (loads `/models/*.vtm`), `miniSelect`, `pick`, `follow`, `satFind`, `identity`, `entityJoin`, `siteQuery`. (CLAUDE.md says `operators.json` lives here; it is actually `datacore/orbital/operators.json`.)
- **Celestial stack `lib/celestial/` (20)**: `spaceFrame.ts` (200 KB source, largest lib file), `celestialSky`, `solarSystem`, `moons`, `rotation`, `ephemeris`, `orbitPath`, `scaleModel`, `simClock`, `spaceAssets`, `starCatalog`, `textureSphere`, `bodyLighting`, `moonSurface`, `moonTiles` (own LRU tile streamer), `lroc`, `lunarMissions`, `lunarSymbols`, `apolloSites`, `zoomSeam`.
- **Render contract `client/src/render/`**: ACTIVE = `frameCore.ts`, `zoomInput.ts`, `perfMetrics.ts`, `layerContract.ts`, `freshness.ts`, `mapBaseConfig.ts`. DEAD-from-main = `tileCore.ts`, `perfHud.ts` (see §5).

## 5. DEAD (file | lines | evidence)

| File | Lines | Evidence |
|---|---|---|
| `client/src/components/TradingActivity.tsx` | 392 | no importers in client/src; knip unused; last touched 2026-05-09 |
| `client/src/pages/not-found.tsx` | 17 | never routed (App.tsx fallback is LandingPage); last touched 2026-04-09 |
| `client/src/hooks/use-mobile.tsx` | 19 | only importer is dead `ui/sidebar.tsx`; App.tsx re-implements `useMobile` inline |
| `client/src/render/tileCore.ts` | 884 | importers: only `tileCore.test.ts`, `mapBaseConfig.test.ts`. `git log -S` finds NO non-test import ever since it shipped 2026-08-12. `mapBaseConfig.ts:4` states "Earth base is MapLibre's own tile pipeline, so tileCore does not" apply. The Law II streamer exists but nothing renders through it; `lib/celestial/moonTiles.ts` rolls its own fetch/createImageBitmap/LRU/AbortController streamer (lines 240–321) — exactly what RENDERING & MOTION LAW says is a violation. Classify: DEAD-in-production / contract-only. |
| `client/src/render/perfHud.ts` | 273 | importer: only its test. `?perf=1` HUD is referenced in comments (`frameCore.ts:21`, `perfMetrics.ts:3`) but never mounted. |
| `client/src/components/ui/*` — 45 files (all except `toast.tsx`, `toaster.tsx`) | 4,557 total | shadcn scaffold from the 2026-04-03 initial commit (`d1ba62fd`), 1 commit each, zero importers outside the group itself (`button/dialog/input/label/separator/sheet/skeleton/toggle/tooltip` are imported only by other dead ui files). Largest: `sidebar.tsx` 727, `chart.tsx` 365 (recharts), `carousel.tsx` 260, `menubar.tsx` 256, `context-menu.tsx` 198, `dropdown-menu.tsx` 198, `form.tsx` 178, `select.tsx` 160, `command.tsx` 151. Knip consequently flags ~30 unused deps: every `@radix-ui/react-*` except `react-toast`, plus `cmdk, date-fns, embla-carousel-react, input-otp, react-day-picker, react-hook-form, react-resizable-panels, recharts, vaul, zod` (zod may be used server-side — verify in the server audit before removal; `shared/schema.ts` also knip-unused). |

Dead static assets (see §7): `client/public/space/earthmap1k.jpg` (344 KB), `client/public/newsletter-hero.png`, `newsletter-mockup.png`, `newsletter-logo.png` (root copies; the `newsletter/images/welcome-to-voltradeai/hero.png` byte-identical copy IS referenced by `server/newsletter.ts`; `.../mockup.png` is not), `attached_assets/login_bg.png` (2.9 MB, `@assets` alias exists but nothing imports it).

## 6. UNCLEAR

- `client/public/imagery/sites/*.jpg` (17 files, 884 KB): referenced only by `scripts/cdse_site_chips.py` (writes `/imagery/sites/<id>.jpg` into `datacore/sentinel2/site_latest_index.json`) and served by `server/siteImagery.ts` through that index — ACTIVE via data indirection, but 14 of 17 are named nowhere in code. Keep; they are small.
- `client/public/models/{aqua-27424,hubble-20580,iss-25544}.json` (sidecar provenance; only `tdrs-boeing601.json` is named in `lib/orbital/realMesh.ts`): likely documentation sidecars — harmless.
- `client/src/lib/air/pdfPlate.ts` loads pdf.js from a CDN (`PDFJS_SRC` exported but knip-unused): runtime path exists only through `FlightProcedures.tsx`; verify CSP allows it.
- `pages/signalLadder.tsx` + `#/data/signals` reach the `tank_fill` signal (`/api/v1/signals/tankfill`) while the registry marks it `planned`/gated — UI presumably shows the gate state; worth a glance by whoever owns ladder honesty.
- knip "unused exports" list (175) is dominated by exported tuning constants (`PLAN_*`, `FLEET_*`, `MOON_TREK_*`, `SCALE_*`…) exported for tests — not dead code, just over-exported.

## 7. Static assets (client/)

`client/public` = 99 MB of 106 MB; **91.8 MB of committed binaries**. Top items:

| Size | Path | Referenced? | Note |
|---|---|---|---|
| 58.1 MB | `client/public/tiles/places.pmtiles` | datamap.tsx, visual_check.mjs | baked PMTiles — belongs on R2/CDN, not git |
| 21.4 MB | `client/public/tiles/seafloor_gebco_mariana.pmtiles` | lib/seafloorV2.ts, server/seafloorTiles.test.ts | baked; same |
| 4.9 MB | `client/public/space/moon_8k.jpg` | spaceAssets.ts, moon.html | |
| 3.6 MB | `client/public/cables/submarine_cables.json` | datamap.tsx, scripts/submarine_cables_build.py | generated by script; TeleGeography NC terms |
| 1.8 MB | `space/8k_stars_milky_way.jpg` | spaceAssets.ts | |
| 1.0 MB | `space/2k_moon.jpg` | spaceAssets.ts, moon.html | |
| 0.84 MB | `tz/timezone_lines.json` | datamap.tsx | generated |
| 0.80 MB | `models/iss-25544.vtm` | realMesh.ts | baked mesh (scripts/earthtwin_real_mesh.mjs) |
| 0.5/0.38/0.37/0.33 MB | `space/marsmap1k, moonmap1k, moonbump1k, earthmap1k.jpg` | first three yes; **earthmap1k.jpg NO** | |
| 0.37+0.35 MB | `vendor/three.core.min.js`, `three.module.min.js` | moon.html only | vendored on purpose (header explains) |
| 0.34 MB | `tiles/seafloor_tid_mariana.pmtiles` | seafloorV2.ts, layers.json | |
| 11 × 0.05–0.29 MB | other `space/*map.jpg` planet textures | spaceAssets.ts | |

Unreferenced: `space/earthmap1k.jpg`, `newsletter-hero.png`, `newsletter-mockup.png`, `newsletter-logo.png`, `newsletter/images/welcome-to-voltradeai/mockup.png` (≈0.6 MB total). Duplicates: `newsletter-hero.png` == `newsletter/images/.../hero.png` (157,399 B both); `newsletter-mockup.png` == `.../mockup.png`. `dist/public/` duplicates the whole tree at build time (152 MB, ignored). The two big PMTiles (80 MB) are the obvious "should live on R2/CDN" candidates; `server/seafloorTiles.ts` already exists, suggesting the serving path could be swapped.

## 8. datacore/ analysis

Structure: **pure data + docs, no code** — 143 `.json`, 7 `.jsonl`, 7 `.gz`, 1 `.geojson`, 5 `.md` (`README.md`, `EVERYTHING_GRAPH.md`, `LICENSING_AUDIT.md`, `API_TERMS_DRAFT.md`, `SENTINEL2_CHANGE_SPEC.md` — all referenced). 163 files, 79 MB, all tracked, nothing gitignored. Served/read by `server/*.ts` via `repoDataPath("datacore/...")`.

Largest: `gem/` 40 MB (Global Energy Monitor extracts, 22 files), `aircraft/` 14 MB (`airports_min.json` 6.2 MB, `entity_spine.json` 3.8 MB, `nasr_fixes.json` 2.0 MB), `sentinel2/` 5.3 MB (`readings_v2.jsonl` 3.8 MB — pipeline OUTPUT archive, grows), `powerplants/` 4.7 MB, `military_installations.json` 2.5 MB, `pfas/` 2.3 MB, `boundaries/` 2.3 MB, `cdc_cancer/` 2.2 MB, `quake_history.json` 1.6 MB, `jodi/` 1.4 MB.

Kinds: reference data (sites, boundaries, powerplants, gem, aircraft, nuclear_*, ofac, military…); registries (`layers.json` 156 KB, `signal_ladder.json` 213 KB, `entity_map.json`, `clinical_trials_sponsor_map.json`, `fleet_operator_tickers.json`); per-stream `manifests/*.json` (74 files, enumerated at runtime by `server/routes.ts:1704` — ACTIVE though no file is named individually); generated outputs (`sentinel2/*.jsonl`, `port_dwell_weekly.json`, `gridvision/gate2_result*.json`, `gridvision/leaderboard.json`); GPU job ledger `runpod/ledger.jsonl` (40 lines, 2026-07-08 → 07-10, jobs smoke + gv-div1..5, last actual_cost 0.22 USD); bake inputs `faa_charts/edges/{sectional,tac}.json` (read by `scripts/faa_charts/run.py` EDGES_DIR).

Referenced-nowhere (non-test) DEAD candidates (only 6 real ones after resolving dynamic loaders): `datacore/gem/cement.json` (3.9 MB), `gem/oil_ngl_pipelines.json` (2.0 MB), `gem/iron_units.json` (1.8 MB), `gem/lng_terminals.json` (1.3 MB), `gem/power_units.json.gz` (4.1 MB, test-only), `gem/coal_finance.json.gz` (0.8 MB, test-only) — ~14 MB ingested but never served. Full per-file table in `scratchpad/client_datacore_refs.txt`.

Layers registry `datacore/layers.json`: `_doc` + **257 layers**; kind 256 raw / 1 signal (`tank_fill`); status 254 live / 2 awaiting_key / 1 planned; groups grid 51, grid_eu 48, grid_as 38, facilities 25, environmental 21, filings 20, base 13, grid_ca 13, grid_sa 13, hazards 9, live 4, graph 1, signals 1; costTier low 166 / light 56 / moderate 29 / heavy 6. Cross-check: 164 ids have no quoted literal in `client/src` — 163 are `powergrid_<code>` generated by template literals in `datamap.tsx:7986-8323` (ACTIVE), leaving exactly **one true orphan: `tank_fill`** (status planned, gated, referenced only by scripts/tests). Server-side only 25/257 ids appear as literals (server serves the registry wholesale via `/api/data/layers`; the per-layer data routes are named by endpoint, not id). Orphan list file: `scratchpad/client_layers_xcheck.json`.

## 9. research/ + root files

research/: 78 files, 14 MB — 54 md, 21 json, 3 html. `experiments.md` 7.27 MB / 110,670 lines (append-only, newest-at-top; last 12 entries are [NO-ACTION]/[RULE-REVIEW]/[RESEARCH], 0/10 REPAIR); `open_questions.md` 1.50 MB / 23,207 lines; `wishlist.md` 298 KB / 4,727 lines.

Constitution-referenced charters (all present): root `VISION.md`, `GIP.md`, `DESIGN.md`; `research/{experiments,open_questions,wishlist,grid_vision,grid_vision_phaseb,scale_program,orbital_program,orbital_models,earth_twin_program,console_charter,usage_log,runpod_ledger}.md`. Other program docs referenced ≥4× from within research/scripts: `PROGRAM_STATE.md` (24), `location_context_engine.md`, `grid_vision_research/products/events_ercot.md`, `bot_logic_research_2026-10-20.md`, `data_census.md`, `tsc_baseline.md`, `position_audit_2026-07-18.md`, `bot_cockpit_audit.md` + `bot_cockpit_audit/*.md` (8 chapters). Referenced once or less (candidates to fold into experiments.md): `bot_page_program.md`, `celestial_v2_program.md`, `csp_manager_rule_sim_2026-10-26_uncut.md`, `grid_vision_data_modalities.md`, `floor_basket_vs_index_2026-10-20.md`, `ocean_quality_notes.md`, `solar_view_spike.md`, `vrp_single_name_vs_index_2026-10-20.md`, `directives/flight_track_3d_handoff_2026-07-20.md`, `directives/space_view_handoff_2026-07-18.md`.

Stray outputs in research/: `csp_manager_rule_sim_2026-10-26_entries.json` **2.07 MB** (referenced in experiments.md), `vrp_single_name_vs_index_2026-10-20.json` **1.16 MB** (referenced NOWHERE, not even its md twin), `csp_sim_earnings_cache.json` 75 KB + `csp_sim_asset_names.json` (read by `scripts/csp_manager_rule_sim.py` — a cache living in research/), `csp_manager_rule_sim_2026-10-26.json` 51 KB, `exit_model_reconciliation_2026-10-20.json`, `floor_basket_vs_index_2026-10-20.json` (both unreferenced), `flightplan_gate1_runs/run3..14.json` (12 files, referenced from experiments.md), `outage_state.json` / `liveness_notify_state.json` (runtime state written by `scripts/session_health_check.py` into a tracked docs folder), 3 html design references under `directives/` (96 KB `.dc.html` referenced from `index.css`; the other two from celestial code/comments).

Root-level non-code files:

| File | Class | Note |
|---|---|---|
| `CLAUDE.md`, `VISION.md`, `GIP.md`, `DESIGN.md` | constitution/charter | referenced |
| `README.md` | doc — **stale**: titled "ETF Builder — Production Data Source Edition", not a repo README | |
| `CHANGELOG.md` | doc (May-2026 fix pack; referenced by CLAUDE.md) | |
| `HANDOFF.md`, `BOOTSTRAP_PROMPT.md` | stray setup docs (2026-07-03), referenced nowhere | |
| `INTEGRATION_NOTES.md`, `side_effect_analysis.md`, `side_effect_analysis_v1033.md` | stray one-off analyses (Apr/May 2026), referenced nowhere; belong in research/ or deleted | |
| `BILLING_SETUP.md`, `BEEHIIV_NEWSLETTER_SETUP.md` | ops docs, referenced nowhere | |
| `package.json` (v1.0.1168), `package-lock.json` (**says 1.0.1131** — lock not re-synced since the version bumps), `tsconfig.json`, `components.json` (shadcn, points at absent `tailwind.config.ts`), `railway.json` | config | |
| `backtest_10yr_results.json` 1.98 MB | data — referenced by `backtest_v2.py`, `bot_engine.py`, `position_sizing.py`, `seed_feedback_from_backtest.py` (Python audit's call; generated 2026-04-07) | |
| `INSTALL.bat`, `VolTrade.bat`, `VolTrade-Launch.bat`, `VolTrade.ico` | stray Windows-desktop launcher from the 2026-04-03 initial commit; nothing references them | |
| `requirements.txt`, `requirements-dev.txt` | config | |
| `postcss.config.js` | config (tailwind/autoprefixer; no `tailwind.config.ts` in repo root) | |
| `attached_assets/login_bg.png` 2.9 MB | stray; `@assets` alias exists, zero imports | |
| `content/newsletter/{README.md, 2026-11-17-welcome-to-voltradeai.md}` | ACTIVE content (read by `server/newsletter.ts`, copied by Dockerfile) | |
| `ci/{quarantine.txt (empty list), quarantine_max.txt, tsc_baseline.txt (TOTAL 6), counter_baseline.txt}` | CI config, referenced by `.github/workflows/ci.yml` + `scripts/gated_tests.sh` | |

## 10. Client tests

140 `client/**/*.test.ts` files (CI: `scripts/gated_tests.sh` collects `client/**/*.test.ts`, runs `npx tsx --test`; quarantine list is empty). 132 tests target only ACTIVE modules. 5 are source-grep tests importing no client module (`pages/datamap.{symbols,sheetOverlap,dossierRadius}.test.ts`, `pages/developers.graphQuery.test.ts`, `lib/bodyDayNightControl.test.ts`). 3 tests target modules that production never loads — not broken, but they certify unused code: `render/tileCore.test.ts` (810 lines), `render/perfHud.test.ts` (228), `render/mapBaseConfig.test.ts` (imports tileCore). Cross-boundary imports: `lib/air/globalFeed.test.ts` → `server/aircraftTiling.ts`; `lib/air/planSeam.test.ts` → `shared/flightPlanGeometry.ts`; one → `shared/positionLevels.ts`.

Gap vs CLAUDE.md "Mechanical enforcement": no test or script greps for `on('zoomend'|'moveend'|'idle'|'move'|...)` (item 1) — grep across all `*.test.ts` and `scripts/` for `moveend` returns nothing. Live handlers exist: `datamap.tsx` lines 5552, 5598, 6820, 7238 (`applyLod`, drives sat LOD + `updateMinis()`), 7466/7468 (`applyMarkerLod` → `setRuntime`), 8619, 9222, 9230 (`onZoomEnd` → `applyVectors`), and `MapNavCluster.tsx:217` (`paint` writes SVG transform on `move`). Several of these are visual updates on map events.

## 11. Duplicate groups (exact files)

1. **Two mobile-detection hooks**: `client/src/App.tsx:66 useMobile` (inline) vs `client/src/hooks/use-mobile.tsx useIsMobile` (dead) — same matchMedia logic.
2. **Two `fmtAgeShort`** with different units: `client/src/lib/air/watchedLookup.ts:109` (ms) vs `client/src/lib/air/flightPlan.ts:299` (sec|null).
3. **Freshness/age formatting ×4**: `client/src/render/freshness.ts formatDataAge`, `client/src/bot/format.ts fmtAge`, local `fmtAge` in `pages/streams.tsx`, local age pills in `pages/news.tsx`, `pages/clinicalTrials.tsx`, `pages/pipelineHealthDashboard.tsx`.
4. **Percent/money/number formatters re-rolled per page** (should sit next to `lib/units.ts`): `fmtPct` in `pages/scanner.tsx`, `pages/gridstress.tsx`, `pages/InsightsView.tsx`, `pages/ETFBuilderView.tsx`; `fmtMoney/fmtNum` in `ETFBuilderView.tsx`, `fmtNum` in `plantOperations.tsx`, `fmtUsd` in `usaspendingContracts.tsx`, `fmtK` in `bankFailures.tsx`, `fmtDate` in `fdaEvents.tsx`/`apikeys.tsx`/`admin.tsx`; canonical `fmtPct/fmtMoney/fmtCompact/fmtSigned` already exist in `client/src/bot/format.ts`.
5. **Stream-overlay page scaffold ×42**: 42 files under `client/src/pages/` each re-implement the same header (ArrowLeft back button + ExternalLink source link) + own `fetch` + own loading/error states; zero pages import a shared shell component.
6. **Two chart stacks**: `client/src/bot/chart/BotChart.tsx` (hand-rolled canvas + frameCore) vs dead `client/src/components/ui/chart.tsx` (recharts wrapper) — recharts can go with it.
7. **Two globes**: `client/src/pages/landing.tsx:175` creates its own `maplibregl.Map` globe; `client/src/components/DataWorldMap.tsx` is a third (topojson canvas) world map for Analyze; `datamap.tsx:3634` is the real one.
8. **Two tile streamers**: `client/src/render/tileCore.ts` (contract, unused) vs `client/src/lib/celestial/moonTiles.ts` (own fetch/decode/LRU/abort) — plus `client/public/moon.html` has a third, in vanilla JS.
9. **Landing page in two technologies**: `pages/landing.tsx` + raw `_landing_body.html.txt`/`_landing_styles.css.txt` (1,017 lines of CSS outside `index.css`)/`_landing_script.js.txt`.
10. **Newsletter images committed twice** (root `newsletter-*.png` vs `newsletter/images/welcome-to-voltradeai/*`).
11. **Timing constants duplicated client/server**: `moon.html:255` mirrors a TS constant by hand ("cannot import that module").

## 12. API routes the client calls (193 distinct literals; full list `scratchpad/client_api_literals.txt`)

Transport: plain `fetch()` (213 call sites in 79 files) + `apiRequest()` from `lib/queryClient.ts` (42 sites); **no EventSource/WebSocket anywhere in client/src** — all "live" data is polling (`setInterval` 33 sites, react-query `refetchInterval` 20 sites). Only one `useQuery` (`/api/auth/me` in App.tsx).

By prefix: `/api/data/*` 129 · `/api/bot/*` 19 · `/api/v1/*` 6 · `/api/auth/*` 6 · `/api/market/*` 4 · `/api/watchlist*` 3 · `/api/account/api-keys*` 3 · `/api/scan*` 2 · `/api/research*` 2 · `/api/newsletter/issues*` 2 · `/api/insights*` 2 · `/api/etf*` 2 · `/api/billing/{checkout,portal}` 2 · `/api/analyze*` 2 · singletons `/api/waitlist`, `/api/trades/today`, `/api/tax/estimate`, `/api/plan`, `/api/orders/open`, `/api/ml/status`, `/api/health`, `/api/analyst`, `/api/admin/site-report`.

Bot: `/api/bot/{status,positions,audit,sleeves,performance,market-status,last-scan,kill,insight,calendar,benchmark,account,toggle,stop,start,lookup/:id,lookup-ledger,bars,bars/:sym}`. v1: `/api/v1/{meta,openapi.json,graph,agent-tools,stats/portdwell,signals/tankfill}`. Auth: `/api/auth/{me,login,logout,register,reset,reset-confirm}`.

`/api/data/*` (129): `layers, streams, snapshot, sites, dossier, graph, signal-ladder, quality-dashboard, pipeline-health-dashboard, platform/stats, macro, boundaries, boundaries_admin1, aircraft{,/global,/window,/tracked,/tracked/:hex,/live/:hex,/plan/:hex,/trips/:hex,/entity/:id}, track/{aircraft,:kind}/:id, procedures/{runway,flight/:cs,:apt/:proc/path,:x}, plates/KAUS/*.PDF{,/georef}, aero/status, airport-status, vessels, portdwell, trains, orbital/{catalog,satcat}, weather/{grid,global/status}, wxtile/:layer/{z}/{x}/{y}, spaceweather, fires, fires-near-facilities, earthquakes, quakehistory, volcanoes, meteors, buoys, rivergauges, plants-near-rivergauges, drought, crop-conditions, air-quality, alerts, radiation, pfas, superfund, waterviolators, cancer-rates, nukefacilities, nukeaccidents, nucleartests, military_installations, powerplants, plant-operations, nrc-reactor-status, grid-demand, grid-generation, grid-stress, eu-{load,generation-mix,day-ahead-prices,macro}, coal-terminals, coal-mine-features, iron-ore-mines, iron-steel-plants, steel-raw-materials, chemicals, lng-shipyards, methane-plumes{,/asset-stats}, jodi-oil-stocks, un-comtrade, imports, border-waits, contracts, filings13f/history, insider{,/history}, earnings-language{,/history}, short-volume{,/history}, short-interest, ftd, microstructure, ats-summary, occ-volume, dtcc-swaps, cot{,/history}, tff, treasury-auctions, dts, vix-term-structure, bank-failures, fda-events, clinical-trials, vehicle-complaints, fmcsa-oos, appstore-rankings, github-activity, attention{,/history}, wiki-attention-signal, gnss-integrity-signal, facility-events, fleet-utilization, site-timeline/:id, shadowstats, :param`.

## 13. Architecture notes

1. Shell: Vite + React 18, `wouter` for 12 flat top-level routes (comment in App.tsx says routes are kept single-segment because of an old `base:"./"` bug — the base is now `/`, so that constraint is obsolete).
2. `/app` is a hash-routed tabbed dashboard (`pages/home.tsx`): Trade menu (analyze/scanner/research/news/watchlist/planner/taxes), Data menu (`#/data`, `#/data/streams`, `#/data/graph`, routes to `/developers`, `/apikeys`), owner-only AI Engine (`#/bot`). A bare `/data` URL is NOT the map (lands on landing page).
3. `/data` = `pages/datamap.tsx`, a 16.6 k-line single component owning one `maplibregl.Map`, 257 registry layers (from `/api/data/layers`), the hashchange overlay router (50 overlays), 95 state hooks and 154 effects. All 42 overlay pages + components are eagerly bundled into the 1.9 MB index chunk.
4. Data path: REST JSON only, through `fetch`/`apiRequest` with cookie credentials; no SSE/WebSocket on the client; liveness = polling with `AbortController` timeouts; react-query is used for auth only. State management = React local state + `localStorage` (4 files) / `sessionStorage` (4 files) + custom pub/sub (`lib/units.ts`, `spaceAssets` prefs); no store library.
5. Render contract (`client/src/render/`): `frameCore` (one rAF loop, springs) and `zoomInput` are consumed by the bot chart and the map nav; `layerContract`/`perfMetrics` by the air/orbital layers; `tileCore` (Law II) and `perfHud` (`?perf=1`) are implemented and tested but wired to nothing.
6. Custom WebGL layers: `lib/air/*` (aircraft instancing, curtains, fleet replay), `lib/orbital/*` (SGP4 in a worker, satBuffer, pick/occlusion), `lib/celestial/*` (solar-system view, moon tiles); plus a separate three.js `moon.html`.
7. Honesty/units/symbol machinery is centralized where CLAUDE.md says: `lib/units.ts`, `lib/mapIcons.ts` (icon registry + legend), `render/freshness.ts`.
8. Tests are node:test via tsx, many of them source-grep contracts; tsc has 6 baseline errors (2 in datamap.tsx).
9. Design-system debt: shadcn scaffold (45 files + ~30 radix deps) is dead weight; `components.json`/`postcss.config.js` reference a tailwind config that does not exist at root; styling is mostly inline-style objects + a 4,855-line `index.css`.
10. Biggest structural risks: the datamap monolith (two tsc errors, nine map-event visual handlers, no code-splitting) and 80 MB of baked PMTiles in git.
