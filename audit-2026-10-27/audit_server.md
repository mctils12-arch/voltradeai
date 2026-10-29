# Server (Node/TypeScript) + scripts/ audit — voltradeai @ v1.0.1168 (2026-10-27)

Method: static import graph (regex `import … from` / `require()` / `import()`; aliases `@shared/*`, `@/*` from tsconfig paths + vite.config.ts) over server/, shared/, script/, scripts/ (+ client/src as resolution target only); reachability from `server/index.ts` (runtime) and `script/build.ts` (build). Cross-checked with `npx knip` (ran, exit 1 = findings) and `bash scripts/tsc_ratchet.sh` (6 errors == pin 6, TS2304 = 0, OK). Read-only; nothing tracked was modified. Scratch artifacts: graph.json, modules.tsv, routes_list.txt, scripts_final.json, deps_out.txt, env_out.txt, knip.txt (same directory as this file).

## 1. Entry points & boot chain

- Railway: `railway.json`/`railway.toml` → Dockerfile build (`npm run build` = `tsx script/build.ts`) → `startCommand ./run_with_daemon.sh` → `node --max-old-space-size=<60% cgroup> dist/index.cjs` + supervised `python3 voltrade_daemon.py` loop. Healthcheck `/api/health` (60s). Dockerfile `CMD` is `node dist/index.cjs` but railway start overrides it.
- `script/build.ts`: Vite client build → `dist/public`; esbuild bundles `server/index.ts` → `dist/index.cjs` (cjs, minified, all package.json deps external except an allowlist that still names drizzle/passport/openai/axios etc. that are NOT dependencies — harmless but stale); copies 19 runtime datacore files into `dist/datacore/` (manifests, sentinel2 readings, NASR airports/fixes/airways/atc, clinical-trials sponsor map, 13 GEM files).
- `server/index.ts` (150 lines) boot order: express app, `trust proxy` → securityHeaders → startVisitHistory + requestLog → globalRateLimit → antiScraping → strict auth limiter on `/api/auth/login|register` → registerRobots / registerHoneypots / registerAdminStats / registerTerms → compression → **registerBillingRoutes (before express.json, raw body for Stripe webhook)** → express.json(1mb)/urlencoded → registerNewsletterRoutes → inline `/api` response logger (duplicates requestLog, see §11) → `await registerRoutes(httpServer, app)` → error handler → production: `serveStatic` (dist/public + SPA fallback `/{*path}`), dev: dynamic `import("./vite")` → listen `PORT` (default 5000).
- `server/routes.ts` (7409 lines) `registerRoutes`: cookieParser, session helper on SQLite `db` (from auth.ts), `registerAuthRoutes`, `registerBotRoutes` (bot.ts, 8602 lines — the orchestrator), then ~221 inline routes (watchlist, market, /api/data/*, /api/v1/*), plus `registerGlobalAircraftRoutes`, `registerFlightPlanRoutes` (which itself mounts `registerSwimCaptureRoutes`), `registerProcedureRoutes`, `registerArchiveOffloadRoutes`, `registerAeroChartRoutes`. routes.ts also starts 10 module-level `setInterval` background jobs.
- Only other runtime roots: none. Nothing is reached solely via `script/build.ts` (build-only count = 0).

## 2. Reachability counts

| set | count |
|---|---:|
| server/*.ts files | 480 (+5 fixtures in server/__fixtures__) |
| server test files (`*.test.ts`) | 263 |
| server non-test modules | 217 |
| ACTIVE (reachable from server/index.ts) | 209 |
| reachable only from script/build.ts | 0 |
| DEAD / unreachable | 8 |
| shared/ reachable | 3 of 4 (`flightPlanGeometry`, `inverseEtfs`, `positionLevels`; `schema.ts` dead) |
| scripts/ files imported by the runtime bundle | 1 (`scripts/statsUtils.ts` ← server/gnssIntegritySignal.ts) |
| unresolved relative imports anywhere | 0 |
| scripts/ tracked files | 178 (134 .py, 39 .ts, 9 .mjs, 6 .sh, 1 .json) |

## 3. ACTIVE modules (grouped by feature; chain = importer path from server/index.ts)

#### Trading bot orchestrator + bot health/P&L helpers (35)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| accountActivities.ts | 153 | index→routes → bot.ts |
| afterTaxBenchmark.ts | 83 | index→routes → bot.ts |
| alpacaFeed.ts | 27 | index→routes |
| bisectionFlags.ts | 36 | index→routes → bot.ts |
| bot.ts | 8602 | index→routes |
| cacheBackfill.ts | 73 | index→routes → ndbcBuoys.ts |
| chainCspEntry.ts | 221 | index→routes → bot.ts |
| chainIv.ts | 211 | index→routes → bot.ts |
| crashSafeRefresh.ts | 114 | index→routes |
| crashVisibility.ts | 147 | index→routes → bot.ts |
| cspSweepCooldown.ts | 188 | index→routes → bot.ts |
| dbWriteTiming.ts | 68 | index→routes → bot.ts |
| derivedSnapshot.ts | 57 | index→routes |
| diag.ts | 448 | index→routes → bot.ts |
| drawdownGuard.ts | 108 | index→routes → bot.ts |
| entryFill.ts | 59 | index→routes → bot.ts |
| eventLoopLag.ts | 51 | index→routes → bot.ts |
| exitFill.ts | 75 | index→routes → bot.ts |
| feedDeadAir.ts | 203 | index→routes → bot.ts |
| healthGate.ts | 86 | index→routes → bot.ts |
| killSwitchAutoResume.ts | 563 | index→routes → bot.ts |
| liveness.ts | 105 | index→routes → bot.ts |
| memoryCeiling.ts | 114 | index→routes → bot.ts |
| mlRetrainStatus.ts | 29 | index→routes → bot.ts |
| optionsChainArchive.ts | 323 | index→routes |
| optionsMgmtAudit.ts | 54 | index→routes → bot.ts |
| orderParams.ts | 77 | index→routes → bot.ts |
| reconstructPnl.ts | 137 | index→routes → bot.ts |
| scannerHealth.ts | 26 | index→routes → bot.ts |
| sizeBlocks.ts | 24 | index→routes → bot.ts |
| sleeveLedger.ts | 233 | index→routes → bot.ts |
| spyBenchmark.ts | 79 | index→routes → bot.ts |
| taxLedger.ts | 161 | index→routes → bot.ts |
| tierDispatchGate.ts | 54 | index→routes → bot.ts |
| tmpCleanup.ts | 70 | index→routes → bot.ts |

#### Boot / HTTP plumbing / security middleware (15)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| adminStats.ts | 73 | index |
| antiScraping.ts | 360 | index |
| emailSender.ts | 12 | index → billing.ts → auth.ts |
| index.ts | 150 |  |
| rateLimit.ts | 296 | index |
| repoFiles.ts | 31 | index→routes → datacoreArchive.ts → aircraftIntent.ts → atcFrequencies.ts |
| requestLog.ts | 235 | index |
| robots.ts | 31 | index |
| routeGuards.ts | 88 | index→routes |
| routes.ts | 7409 | index |
| securityHeaders.ts | 123 | index |
| static.ts | 49 | index |
| terms.ts | 152 | index |
| visitHistory.ts | 185 | index |
| vite.ts | 59 | index |

#### Aircraft: live feed, sweep, tiling, entities, trips (18)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| adsbGovernor.ts | 258 | index→routes → aircraftTiling.ts |
| aircraftEntities.ts | 156 | index→routes |
| aircraftFastLane.ts | 192 | index→routes → globalAircraft.ts |
| aircraftTiling.ts | 431 | index→routes |
| aircraftTrips.ts | 325 | index→routes |
| aircraftWindow.ts | 447 | index→routes |
| airportsIndex.ts | 106 | index→routes |
| closeApproach.ts | 585 | index→routes → aircraftWindow.ts |
| faaStatus.ts | 286 | index→routes |
| fleetUtilization.ts | 308 | index→routes |
| globalAircraft.ts | 383 | index→routes |
| globalDiscPlan.ts | 209 | index→routes → globalAircraft.ts |
| globalSnapshot.ts | 238 | index→routes → globalAircraft.ts |
| globalSweep.ts | 516 | index→routes → globalAircraft.ts |
| laddRegistry.ts | 127 | index→routes → procedures.ts |
| openskyGlobal.ts | 344 | index→routes → globalAircraft.ts |
| operatorResolution.ts | 109 | index→routes → fleetUtilization.ts |
| trackedPlanes.ts | 277 | index→routes |

#### FLIGHT program: SWIM plans, procedures/plates, FAA charts (18)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| aeroBake.ts | 158 | index→routes → aeroCharts.ts |
| aeroCharts.ts | 1107 | index→routes |
| aeroOverview.ts | 125 | index→routes → aeroCharts.ts |
| atcFrequencies.ts | 379 | index→routes → datacoreArchive.ts → aircraftIntent.ts |
| cifp.ts | 995 | index→routes → procedures.ts |
| cifpShared.ts | 15 | index→routes → procedures.ts |
| dtpp.ts | 381 | index→routes → procedures.ts |
| flightPlans.ts | 1484 | index→routes |
| navAirways.ts | 564 | index→routes → flightPlans.ts |
| navFixes.ts | 48 | index→routes → procedures.ts → cifpShared.ts |
| plateGeoref.ts | 853 | index→routes → procedures.ts |
| procedures.ts | 485 | index→routes |
| runwayDetect.ts | 200 | index→routes → procedures.ts |
| swimCapture.ts | 239 | index→routes → flightPlans.ts |
| swimConnector.ts | 482 | index→routes → flightPlans.ts |
| swimNotam.ts | 175 | index→routes → flightPlans.ts → swimCapture.ts |
| swimSfdps.ts | 981 | index→routes → flightPlans.ts |
| swimTfms.ts | 335 | index→routes → flightPlans.ts → swimCapture.ts |

#### Datacore archive, R2 offload, quality/freshness dashboards (19)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| aircraftFixBus.ts | 52 | index→routes → datacoreArchive.ts → aircraftIntent.ts |
| aircraftIntent.ts | 754 | index→routes → datacoreArchive.ts |
| aircraftIntentDaily.ts | 248 | index→routes → archiveOffload.ts |
| archiveOffload.ts | 933 | index→routes |
| coverageGapFill.ts | 150 | index→routes |
| dataQuality.ts | 130 | index→routes → gridDemand.ts |
| datacoreArchive.ts | 1147 | index→routes |
| globalScopes.ts | 139 | index→routes |
| layerFreshness.ts | 80 | index→routes |
| liveDelta.ts | 141 | index→routes |
| pipelineHealthHistory.ts | 180 | index→routes |
| platformStats.ts | 154 | index→routes |
| qualityDashboard.ts | 113 | index→routes |
| r2Client.ts | 593 | index→routes → procedures.ts → dtpp.ts |
| signalLadder.ts | 131 | index→routes |
| siteReport.ts | 298 | index→routes |
| snapshotStreams.ts | 35 | index→routes → streamsInventory.ts |
| streamsInventory.ts | 283 | index→routes |
| viewport.ts | 116 | index→routes |

#### Earth/space/environment layers + tile proxies (27)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| airQuality.ts | 391 | index→routes |
| ambientRadiation.ts | 262 | index→routes |
| catalogMirror.ts | 49 | index→routes |
| cropConditions.ts | 342 | index→routes |
| droughtMonitor.ts | 295 | index→routes |
| femaFlood.ts | 182 | index→routes |
| gnssIntegrityDaily.ts | 201 | index→routes |
| gnssIntegrityQuery.ts | 211 | index→routes → gnssIntegrityDaily.ts |
| gnssIntegritySignal.ts | 277 | index→routes |
| meteors.ts | 187 | index→routes |
| nasaFirms.ts | 364 | index→routes |
| nwsAlerts.ts | 248 | index→routes |
| owmTiles.ts | 129 | index→routes |
| pfas.ts | 119 | index→routes |
| satellites.ts | 401 | index→routes |
| siteImagery.ts | 31 | index→routes |
| siteTimeline.ts | 190 | index→routes |
| spaceWeather.ts | 787 | index→routes |
| spaceweatherNormalizer.ts | 122 | index→routes → spaceWeather.ts |
| superfund.ts | 160 | index→routes |
| tiles3dBudget.ts | 132 | index→routes |
| tiles3dProxy.ts | 38 | index→routes |
| usgsQuakes.ts | 295 | index→routes |
| usgsVolcanoes.ts | 306 | index→routes |
| usgsWater.ts | 265 | index→routes |
| waterViolators.ts | 165 | index→routes |
| weatherGrid.ts | 88 | index→routes |

#### Vessels, ports, rail, road (12)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| aisFeed.ts | 129 | index→routes |
| cbpBorderWait.ts | 256 | index→routes |
| fmcsaOutOfService.ts | 253 | index→routes |
| ndbcBuoys.ts | 289 | index→routes |
| portDwell.ts | 446 | index→routes |
| portDwellCapture.ts | 226 | index→routes → bot.ts |
| portDwellWeekly.ts | 137 | index→routes → bot.ts → portDwellCapture.ts |
| railTraffic.ts | 172 | index→routes |
| shadowFleet.ts | 615 | index→routes |
| shadowFleetGate1.ts | 173 | index→routes → bot.ts |
| trainsFeed.ts | 91 | index→routes |
| vesselStream.ts | 104 | index→routes |

#### Entity graph, dossier, query engine, analyst (4)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| analyst.ts | 744 | index→routes |
| dossier.ts | 327 | index→routes |
| entityGraph.ts | 547 | index→routes |
| queryEngine.ts | 442 | index→routes |

#### Auth, billing, API product (/api/v1), compliance (7)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| apiKeyAccounts.ts | 190 | index→routes |
| apiProduct.ts | 1352 | index→routes |
| auth.ts | 419 | index → billing.ts |
| billing.ts | 335 | index |
| newsletter.ts | 164 | index |
| providerCompliance.ts | 91 | index→routes |
| waitlist.ts | 61 | index→routes |

#### Markets, filings, macro, alt-data roots (29)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| appStoreRankings.ts | 408 | index→routes |
| cboeVix.ts | 367 | index→routes |
| cdcCancer.ts | 288 | index→routes |
| cftcCot.ts | 354 | index→routes |
| cftcTff.ts | 325 | index→routes |
| clinicalTrials.ts | 380 | index→routes |
| clinicalTrialsTicker.ts | 132 | index→routes |
| dtccSwaps.ts | 679 | index→routes |
| edgar13f.ts | 373 | index→routes |
| edgarForm4.ts | 416 | index→routes |
| fdaEvents.ts | 323 | index→routes |
| fdicBanks.ts | 303 | index→routes |
| fdicGate2.ts | 163 | index→routes → bot.ts |
| finraQuery.ts | 770 | index→routes |
| finraShortVolume.ts | 706 | index→routes |
| fredMacro.ts | 356 | index→routes |
| gdeltEvents.ts | 296 | index→routes |
| githubOrgActivity.ts | 519 | index→routes |
| nhtsaComplaints.ts | 306 | index→routes |
| occVolume.ts | 353 | index→routes |
| sec8kEarnings.ts | 443 | index→routes |
| secFtd.ts | 364 | index→routes |
| secMidas.ts | 634 | index→routes |
| settlementStress.ts | 400 | index→routes |
| treasuryAuctions.ts | 281 | index→routes |
| treasuryDts.ts | 266 | index→routes |
| usaSpending.ts | 476 | index→routes |
| wikiAttention.ts | 383 | index→routes |
| wikiAttentionSignal.ts | 242 | index→routes |

#### GEM trackers, oil, trade (13)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| censusImports.ts | 302 | index→routes |
| gemChemicals.ts | 195 | index→routes |
| gemCoalMineFeatures.ts | 148 | index→routes |
| gemCoalTerminals.ts | 161 | index→routes |
| gemIronOreMines.ts | 186 | index→routes |
| gemIronSteelPlants.ts | 378 | index→routes |
| gemLngCarriers.ts | 212 | index→routes |
| gemMethane.ts | 139 | index→routes → gemMethaneProximity.ts |
| gemMethaneAssets.ts | 151 | index→routes → gemMethaneProximity.ts |
| gemMethaneProximity.ts | 272 | index→routes |
| gemSteelRawMaterials.ts | 255 | index→routes |
| jodiOil.ts | 151 | index→routes |
| unComtrade.ts | 129 | index→routes |

#### Power grid / energy (12)

| module | lines | importer chain (from server/index.ts) |
|---|---:|---|
| epaCamd.ts | 442 | index→routes |
| euDayAheadPrices.ts | 424 | index→routes |
| euGenerationMix.ts | 417 | index→routes |
| euLoad.ts | 400 | index→routes |
| euMacro.ts | 384 | index→routes |
| firesFacilities.ts | 72 | index→routes |
| gridDemand.ts | 466 | index→routes |
| gridGeneration.ts | 380 | index→routes |
| gridStress.ts | 331 | index→routes |
| nrcReactorStatus.ts | 489 | index→routes |
| plantsUnderAlerts.ts | 100 | index→routes |
| riverPlants.ts | 93 | index→routes |

Notes on ACTIVE: `vite.ts` is reached only through `await import("./vite")` in the non-production branch (dev-only, but esbuild bundles it; `vite` stays external). `swimConnector.ts` loads `solclientjs` through a variable-named dynamic `import()` only when `SWIM_*` env is complete — the dependency is optional at runtime by design.

## 4. DEAD modules (unreachable from server/index.ts)

| module | lines | last commit | evidence |
|---|---:|---|---|
| server/storage.ts | 5 | 2026-04-03 (template scaffold) | zero importers anywhere (grep `storage` hits are unrelated words); not in package.json scripts/CI/Dockerfile/run_with_daemon.sh; knip "unused file". Noop `IStorage` stub from the Replit template. |
| shared/schema.ts | 5 | template | zero importers; `export {}` placeholder; knip "unused file". |
| server/buoyPortProximity.ts | 140 | 2026-08-08 | importers: only its test + `scripts/buoy_port_proximity_probe.ts` (itself docs-only, §8); mentioned in research/*.md + datacore/signal_ladder.json only. Gate-2 join for a hypothesis whose probe recorded an honesty finding. |
| server/countryLookup.ts | 147 | 2026-09-02 | importers: its test + `server/gasFlareCandidates.ts` (dead) + `scripts/gasflare_gate1.ts` (docs-only). Mention in gemSteelRawMaterials.ts is a comment. |
| server/gasFlareCandidates.ts | 213 | 2026-09-01 | importers: its test + `scripts/gasflare_gate1.ts` (docs-only). Gate-1 "clean negative" per commit message — code left in tree. |
| server/cusipResolver.ts | 181 | 2026-08-22 | importers: its test + `scripts/dtcc_resolve_cusips.ts` (docs-only; reads `OPENFIGI_API_KEY`). Referenced in datacore/manifests/dtccswaps.json provenance text only. |
| server/fleetOperatorTickers.ts | 40 | 2026-09-03 | importers: its test only (0 non-test importers). Imports datacore/fleet_operator_tickers.json. |
| server/gridDemandRevision.ts | 116 | 2026-07-30 | importers: its test + `scripts/gate1_eia930_revision_probe.ts` (docs-only). |
| server/shadowFleetReference.ts | 102 | 2026-08-29 | importers: its test + `scripts/ofac_sdn_ingest.ts` (docs-only). Mention in shadowFleetGate1.ts is a comment. |

None of these is referenced by package.json scripts, .github/workflows/*, Dockerfile, run_with_daemon.sh, or any CI-run script. Seven of them (all but storage.ts and schema.ts; list in §7) still have tests that run in the gate, so CI keeps dead code compiling. **Big dead files (>300 lines): none** — the largest dead server module is 213 lines. Largest unreferenced scripts/ files >300 lines: `scripts/gdelt_fires_gate2.ts` (381), `scripts/treasury_dts_gate2.ts` (337), `scripts/finra_shortvol_gate2_retest.ts` (323), `scripts/settlement_stress_gate2.ts` (309), `scripts/github_activity_gate2.ts` (303), `scripts/labeler_before_after.py` (340).

Pattern: 7 of 8 dead server modules are ROOT-VALIDATION-LADDER gate-1/gate-2 helpers whose root died at a gate; the ladder entry was recorded in research/ but the code was never removed (CLAUDE.md STALENESS AUDIT rule: "the same PR removes its code from all active execution paths").

## 5. UNCLEAR

| item | why unclear |
|---|---|
| server/vite.ts (59) | reached only via dev-branch dynamic import; bundled into dist but never executed in production. Imports `nanoid`, which is NOT in package.json (transitive, see §9). |
| server/swimConnector.ts → `solclientjs` | variable-named dynamic import; active only when `SWIM_SFDPS_*`/`SWIM_SHARED_*` env is set. |
| scripts/*.test.ts / .mjs (13 files) | present but `scripts/gated_tests.sh` collects only `server/*.test.ts` and `client/**/*.test.ts`; `npm run test:node` likewise. These 13 never run in CI: demandProfile.test.mjs, dtcc_resolve_cusips, finra_shortvol_gate2, finra_shortvol_gate2_retest, gdelt_fires_gate2, github_activity_gate2, gnss_integrity_gate2, occ_volume_gate2_clustered_disjoint, portdwell_weekly_snapshot, settlement_stress_gate2, statsUtils, treasury_dts_gate2, wikiattention_gate1 (.test.ts). |
| 41 scripts/ files referenced only by research/*.md, datacore manifests/signal_ladder.json, or code comments | see §8 list; they are session tools, not product paths. |

## 6. Routes & unmounted route modules

304 registrations in 17 files (221 routes.ts, 42 bot.ts, 7 procedures.ts, 7 auth.ts, 4 swimCapture.ts, 4 billing.ts, 3 globalAircraft.ts, 2 each terms/newsletter/index/flightPlans/aeroCharts/adminStats, 1 each vite/static/robots/archiveOffload).

Prefix groups: `/api/data/*` 143 (aircraft 14, procedures 5, swim 4, 3dtiles 3, then ~95 one-route data layers), `/api/v1/*` 60 (data 44, stats 11, tracks, openapi.json, meta, graph, agent-tools), `/api/bot/*` 33 (+ `/api/health`, `/api/system/*` 3, `/api/monitoring/*` 4, `/api/diag/:probe`, `/api/daemon/health` — all in bot.ts), `/api/auth/*` 9, `/api/market*` 5, `/api/account` 4, `/api/watchlist` 3, `/api/ml` 3, `/api/billing` 3, `/api/admin` 2 + `/api/_admin` 2, `/api/newsletter` 2, `/api/trades` 2, `/api/scan` 2, `/api/debug` 2, singles: analyst, analyze, etf, insights, legal/terms, news, orders, plan, positions, research, shorts, tax, waitlist; non-API: `/tiles/aero/:chart/:z/:x/:y`, `/tiles-r2/:name`, `/terms`, `/robots.txt`, SPA `/{*path}`.

Unmounted route modules: **none**. Every exported `register*Routes` is called (index.ts: billing, newsletter, robots, honeypots, adminStats, terms; routes.ts: auth, bot, globalAircraft, flightPlan, procedure, archiveOffload, aeroChart; flightPlans.ts: swimCapture). No `express.Router()` instances exist — everything registers directly on `app`. Only duplicate method+path is `use /{*path}` in static.ts vs vite.ts, which are mutually exclusive by NODE_ENV (not a real conflict). `scripts/duplicate_route_registration.py` is already a CI counter for this.

## 7. Tests

- 263 `server/*.test.ts` (all gated by scripts/gated_tests.sh; quarantine list empty, pin file present). 13 scripts/ tests un-gated (§5). 140 client tests (other agent).
- Broken imports: **0** test imports an unresolvable local path (graph), and `tsc_ratchet` is at its pin (tests are excluded from tsconfig, so tsc does not cover them).
- Tests of DEAD modules (run in CI, keep dead code alive): `buoyPortProximity.test.ts`, `countryLookup.test.ts` (imports `./countryLookup.ts` with extension), `cusipResolver.test.ts`, `fleetOperatorTickers.test.ts`, `gasFlareCandidates.test.ts`, `gridDemandRevision.test.ts`, `shadowFleetReference.test.ts` (7 files).
- 67 test files have no same-named module; 56 of those import **no local module at all** — they are source-text/regex assertions (`fs.readFileSync` of bot.ts/routes.ts/ci.yml/etc.), e.g. tier1DrawdownKillWiring, posKill*, staleOrderSweep*, optionsSlot*, ciEnvHonesty, tscRatchet, gatedTests, layersWiring, visualPagesWiring. They are brittle to refactors (they pin line text, not behavior) but currently pass.
- Tests read dated archive fixtures (e.g. `airquality/2026-05-01.jsonl`, `fredmacro/2026-07-05.jsonl`) from temp dirs they create — not repo paths, so not stale.
- knip: 110 unused exports in server/shared/scripts (flightPlans 30, plateGeoref 26, aeroCharts 22, archiveOffload 19, cifp 15, analyst 10) and 107 unused exported types — exported for tests or leftover; not dead files.

## 8. scripts/ classification (178 tracked files)

Totals: ACTIVE-tooling 6 (named in package.json/CI) + 8 transitively CI-active; INVOKED-by-code 90 (spawned/imported by another script, server, or python non-test code); TESTED-only 43 (only a root `test_*.py` / `server/*.test.ts` names them — exercised by the pytest/node gate, never run as tooling); UNCLEAR-docs-only 41 (only research/*.md, datacore json, or code comments); DEAD 1.

- **ACTIVE-tooling**: `tsc_ratchet.sh`, `counter_ratchet.sh`, `gated_tests.sh`, `program_status.sh` (ci.yml), `grid_county_ba.py` (ci.yml), `visual_check.mjs` (npm `visual`, 4415 lines). Transitive: `deploy_gate_smoke.mjs` (gated_tests), `design_token_drift.py`, `duplicate_route_registration.py`, `hardcoded_palette_hex.py`, `param_shadows_outer_binding.py` (program_status → counter_ratchet), `demandProfile.mjs`, `site_verify.py`, `stb_rail.py` (visual_check).
- **DEAD**: `scripts/demandProfile.test.mjs` (41 lines; a test no runner collects). `scripts/faa_charts/common.py` was a false positive — imported by stem (`from common import …`) by bake/edges/run/verify.
- **Groups** (counts; [U]=docs-only, [T]=tested-only, [CI]=tooling):
  - datacore ingest + gate-1 validation (42): cdc_cancer_rates, cdse_*, cpc_degree_days, dtcc_swaps_gate1, eia860*/eia861/eia930*/eia_*/eia_weekly, gem_ingest, gem_suite_ingest [T], gem_methane_gate2c [T], jodi_*, nasa_gibs_nightlights_gate1, nightlights_archiver [T], nrc_gate1_registry_match, pfas_ucmr5 [T], rail_traffic_gate1/2, sentinel2_*, tankfill_*, un_comtrade_*; [U]: clinical_trials_sponsor_join.ts, dtcc_resolve_cusips(.test).ts, ofac_sdn_ingest.ts.
  - ladder gate tests (41, mostly one-off): [U] 21 — finra_gate1.ts, finra_shortvol_gate2(.test/_retest).ts, gasflare_gate1.ts, gate1_eia930_revision_probe.ts, gate1_usaspending_ticker_check.ts, gdelt_fires_gate2(.test).ts, github_activity_gate2.ts, gnss_integrity_gate2.test.ts, occ_volume_gate1/_clustered/_disjoint(.test).ts, portdwell_gate1_rollup.ts, settlement_stress_gate2.ts, treasury_dts_gate2.test.ts, wikiattention_gate1.test.ts, earnings_language_gate2_robustness.py.
  - GRID VISION + RunPod (27): grid_*, gridvision_*, gridvision_train/*, runpod_launch/reap/budget, labeler_before_after.py [U].
  - research probes (19): hurst/kuramoto/omori/contagion/critical_slowing/permutation_entropy/illiquid_universe_* (chain-invoked by each other), buoy_port_proximity_probe.ts [U].
  - static-dataset builders / bakes (18): build_* (powerplants, power_tiles.sh, nasr_*, entity_map [U], entity_spine [T], airports.mjs [U], tank_registry [U]), boundaries_admin1_build [U], military_installations_build, submarine_cables_build [T], earthtwin_real_mesh/glb_decompress.mjs [U], gebco_seafloor_tiles [T], refresh_nasr, nuclear_tests_site_check [T].
  - CI gates / repo health (17): listed above + research_state_check, session_health_check, snapshot_refresh, ladder_readiness_check, ladder_registry_coverage_check [T], data_stream_registry_check, github_branch_cleanup.sh [U], ts_code_only.py [U].
  - bot/ladder analyses (10): csp_manager_rule_sim [T], exit_model_reconciliation, floor_basket_vs_index [T], vrp_single_name_vs_index [T], microcap_cost_floor_check (imported by backtest_v2.py), reconstruct_position_pnl [T], portdwell_weekly_snapshot(.test) , statsUtils(.test) (statsUtils.ts is imported by the runtime server bundle).
  - visual/perf harness (8): visual_check.mjs [CI], demandProfile.mjs [CI], demandProfile.test.mjs [DEAD], perf_profile.mjs [U], verify_weather_prod.mjs [U], site_verify.py [CI], site_candidate_verify.py [U], geo_consistency.py [T].
  - FAA chart bake (7): faa_charts/{run,bake,edges,cycle,verify,common}.py, make_fixture.py [T].
- **Completed one-off migrations / probes that look finished** (candidates for deletion or `scripts/attic/`): every `*_gate1.ts` / `*_gate2*.ts` whose root is recorded KILLED or passed in datacore/signal_ladder.json (finra_gate1, occ_volume_gate1, gasflare_gate1, gate1_eia930_revision_probe, gate1_usaspending_ticker_check, portdwell_gate1_rollup, treasury_dts_gate1, nrc_gate1_registry_match, dtcc_swaps_gate1, crop_conditions_gate1, buoy_port_proximity_probe); `build_tank_registry.py`, `build_entity_map.py`, `boundaries_admin1_build.py`, `build_airports.mjs` (their outputs are committed under datacore/); `labeler_before_after.py`; `earthtwin_*.mjs`; `verify_weather_prod.mjs`, `perf_profile.mjs`.
- The server never spawns anything under scripts/: all 13 `gem_ingest.py` and other mentions in server/*.ts are comments or provenance strings in API metadata (apiProduct.ts, jodiOil.ts, railTraffic.ts, unComtrade.ts, qualityDashboard.ts, cdcCancer.ts).

## 9. npm dependencies (68 deps + 18 dev + 1 optional)

Client reachability was computed from the single Vite entry `client/index.html → /src/main.tsx` (222 of 279 non-test client files reachable). Only `components/ui/toast.tsx` and `toaster.tsx` of the 48 shadcn `components/ui/*` files are reachable, so every radix package except `@radix-ui/react-toast` is dead weight (knip agrees: "Unused files" lists the other 46 ui/*.tsx; the client auditor should confirm before removal).

**UNUSED dependencies (37)** — no reachable importer:
- 26 radix: `@radix-ui/react-{accordion, alert-dialog, aspect-ratio, avatar, checkbox, collapsible, context-menu, dialog, dropdown-menu, hover-card, label, menubar, navigation-menu, popover, progress, radio-group, scroll-area, select, separator, slider, slot, switch, tabs, toggle, toggle-group, tooltip}` (imported only by unreachable `client/src/components/ui/*.tsx`).
- shadcn companions imported only by unreachable ui files: `cmdk`, `embla-carousel-react`, `input-otp`, `react-day-picker`, `react-hook-form`, `react-resizable-panels`, `recharts`, `vaul`.
- `date-fns` and `zod`: **zero** imports anywhere in server/client/shared/scripts (only named in script/build.ts's esbuild allowlist).
- `@types/bcryptjs`, `@types/better-sqlite3`, `@types/cookie-parser` are in `dependencies` (should be devDependencies; they are used for typing).
**UNUSED devDependencies (2)**: `@tailwindcss/typography` (not in tailwind.config.ts plugins — only `tailwindcss-animate` is), `@tailwindcss/vite` (Tailwind v4 Vite plugin; project uses Tailwind 3 via postcss.config.js).
Used via config/CLI only (fine): `autoprefixer`, `postcss`, `tailwindcss`, `tailwindcss-animate` (tailwind/postcss config), `cross-env`/`tsx`/`typescript`/`esbuild`/`vite`/`@vitejs/plugin-react` (npm scripts, build.ts, vite.config.ts), `playwright` (scripts/visual_check.mjs, perf_profile.mjs), `pngjs` (visual_check.mjs, earthtwin_real_mesh.mjs), `bufferutil` (optional ws accelerator).
Runtime-server deps confirmed used: express, compression, cookie-parser, bcryptjs, better-sqlite3, stripe, ws, fflate, gray-matter, marked, sharp, pmtiles, topojson-client, solclientjs (optional dynamic).

**Imported but missing from package.json (4 real)**: `nanoid` (server/vite.ts; transitive, and in build.ts's bundle allowlist), `topojson-specification` (type-only import in client/src/components/DataWorldMap.tsx; not even in node_modules — compiles only because `import type` is erased), `@maplibre/maplibre-gl-style-spec` (client tests), `@tanstack/query-core` (client test) — the latter two are transitive. (`today`, `sustained` were regex false positives.)

build.ts allowlist names 20 packages that are not dependencies (`@google/generative-ai`, axios, cors, drizzle-orm, drizzle-zod, express-rate-limit, express-session, jsonwebtoken, memorystore, multer, nodemailer, openai, passport, passport-local, uuid, xlsx) — template residue; no effect.

## 10. Environment variables (process.env reads in server/, shared/, script/, scripts/)

87 distinct real env names (104 regex hits minus 17 shell-local variables in .sh files: APPLY, BATCH, BASELINE, BASELINE_FILE, CUR, DAYS, JSON, KEEP_FILE, LIST, MAX_AGE_DAYS, MAX_FILE, OUT, PBF, PIN, REGION, RUN_TSC, WORK). **No env var is read only by a DEAD module.** No `.env.example` exists; README.md documents only ALPACA_KEY/ALPACA_SECRET/POLYGON_API_KEY. Full table with files in `env_out.txt`; summary:

- Trading/broker (bot.ts, routes.ts, optionsChainArchive.ts): `ALPACA_KEY`, `ALPACA_SECRET` (README), `ALPACA_BASE_URL`, `ALPACA_DATA_FEED`, `POLYGON_API_KEY` (README), `VOLTRADE_DAEMON_ENABLED`, `VOLTRADE_DAEMON_SOCKET`, `VOLTRADE_LIQUIDATE_ON_KILL`, `VOLTRADE_DISABLE_TIER2/3` (wishlist), `ALERT_EMAIL`, `RESEND_KEY` — mostly UNDOCUMENTED.
- Auth/billing (frozen files): `DB_PATH`, `OWNER_EMAIL`, `OWNER_DEFAULT_PASSWORD`, `APP_URL`, `STRIPE_SECRET_KEY` (CLAUDE.md), `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID_MONTHLY/ANNUAL`, `BILLING_ENABLED` (CLAUDE.md), `API_PRODUCT_KEYS`, `ADMIN_STATS_TOKEN`, `DEBUG_ENABLED`, `DIAG_TOKEN` (wishlist).
- Security/perf: `REQUEST_LOG_CAP`, `REQUEST_LOG_IP_SALT`, `SECURITY_CSP_ENFORCE`, `PORT`, `NODE_ENV`, `DATA_DIR` (9 readers; wishlist/open_questions only).
- Data-provider keys (documented in research/wishlist.md): `AISSTREAM_KEY`, `ANTHROPIC_API_KEY`, `CENSUS_API_KEY`, `EIA_API_KEY`, `ENTSOE_API_KEY`/`ENTSOE_TOKEN` (both read — duplicate names), `EPA_CAMD_API_KEY`, `FIRMS_MAP_KEY`/`NASA_FIRMS_MAP_KEY` (duplicate names), `FRED_API_KEY`, `GOOGLE_MAPS_API_KEY`, `NASS_API_KEY`, `OPENSKY_CLIENT_ID/SECRET`, `OPENWEATHERMAP_KEY`; undocumented: `GITHUB_CATALOG_TOKEN`, `OPENFIGI_API_KEY` (scripts/dtcc_resolve_cusips.ts only), `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ARCHIVE_BUCKET`, `R2_PUBLIC_URL`.
- Feature/tuning knobs (all UNDOCUMENTED): `ADSB_TOTAL_RPS_CEILING`, `AERO_PREFETCH_RPS`, `AERO_R2_MAX_PUTS_PER_DAY`, `AERO_TILE_CACHE_MAX_BYTES`, `AIRCRAFT_FAST_LANE`, `AIRCRAFT_GAPFILL`, `ANALYST_DAILY_TOKENS`, `ANALYST_MAX_TOKENS`, `ANALYST_MODEL`, `GLOBAL_SWEEP_ARCHIVE`, `GLOBAL_SWEEP_DISC_RPS`, `GLOBAL_SWEEP_ENABLED`, `GLOBAL_SWEEP_RPS`, `GLOBAL_SWEEP_TYPE_REFRESH_S`, `GRID_DEMAND_BACKFILL`, `LOCAL_HOT_DAYS`, `R2_COLD_CACHE_MAX_BYTES`, `OPENSKY_DAILY_CREDITS`, `OPENSKY_INTERVAL_S`; documented in wishlist: `FINRA_DEEP_BACKFILL`, `FINRA_ORF_DEEP_BACKFILL`, `FINRA_QUERY_BACKFILL`, `OCC_DEEP_BACKFILL`, `REPLAY_RETENTION_DAYS`.
- SWIM (swimConnector/swimSfdps/swimCapture/flightPlans; undocumented): `SWIM_PRODUCTS`, `SWIM_SFDPS_URL|USER|PASSWORD|QUEUE|VPN`, `SWIM_SHARED_USER|PASSWORD`, `SWIM_USER|PASSWORD`, `SWIM_HOURLY_KEEP`, `SWIM_PLAN_TTL_MS`, `SWIM_ROUTE_MAX_POINTS`, `SWIM_ROUTE_TEXT_MAX`, `SWIM_STORE_MAX`, plus `${prefix}_USER/_PASSWORD` per product.
- CI/tooling scripts: `QUARANTINE_FILE`, `QUARANTINE_MAX_FILE`, `QUARANTINE_MAX_AGE_DAYS`, `DEPLOY_GATE_SMOKE`, `TSC_BASELINE_FILE`, `COUNTER_BASELINE_FILE`, `RAILWAY_ENVIRONMENT` (deploy_gate_smoke), `VOLTRADE_BASE_URL`, `VOLTRADE_PROD_URL`, `HTTPS_PROXY` (gate scripts).

## 11. Duplicate / overlapping implementations

1. **Request logging twice**: `server/requestLog.ts` middleware (structured, salted-IP, feeds visitHistory) AND the inline `/api` logger in `server/index.ts` lines 82–107 (console line + 500-char JSON capture, template residue).
2. **Two SQLite databases, two `new Database(` sites**: `server/auth.ts` (`/data/voltrade.db` on the volume; shared handle `db` imported by routes.ts and bot.ts for sessions, users, watchlists, audit_log) and `server/billing.ts` (`path.resolve(process.cwd(), "voltradeai.db")` — **not on the persistent volume**, lost on each deploy). Both frozen files.
3. **DATA_DIR resolution re-implemented** in analyst.ts, crashSafeRefresh.ts, datacoreArchive.ts (`env.DATA_DIR || (/data exists ? /data/voltrade : /tmp)`), auth.ts (`/data` vs cwd — a different rule), and ~14 inline Python `-c` snippets in bot.ts; `storage_config.py` is the Python source of truth but bot.ts hardcodes `/data/voltrade/voltrade_*.json` paths (equity curve, peak, kill switch, liveness, deploy timestamp, scan timings).
4. **Duplicate env names for one provider**: `ENTSOE_API_KEY` vs `ENTSOE_TOKEN` (euLoad/euDayAheadPrices/euGenerationMix read both); `FIRMS_MAP_KEY` vs `NASA_FIRMS_MAP_KEY` (nasaFirms.ts).
5. **Ladder-gate pairs where the gate helper outlived its use**: shadowFleet.ts (live) / shadowFleetGate1.ts (live, bot Tier-3) / shadowFleetReference.ts (dead); gridDemand.ts (live) / gridDemandRevision.ts (dead); gasFlareCandidates.ts + countryLookup.ts (dead pair).
6. **Signal-vs-raw sibling modules** (intentional per the RAW OVERLAYS vs SIGNALS rule, but each pair carries its own fetch/cache): wikiAttention / wikiAttentionSignal; gnssIntegrityDaily / gnssIntegrityQuery / gnssIntegritySignal; portDwell / portDwellCapture / portDwellWeekly; gemMethane / gemMethaneAssets / gemMethaneProximity.
7. **Seven GEM "RAW-DATA overlay" modules** (gemChemicals, gemCoalMineFeatures, gemCoalTerminals, gemIronOreMines, gemIronSteelPlants, gemLngCarriers, gemSteelRawMaterials, 148–378 lines each) each re-implement the same gz-JSON → GeoJSON loader + route — one generic loader with a table would replace ~1,500 lines.
8. **Rate limiting / throttling**: rateLimit.ts (inbound global + strict auth), antiScraping.ts (inbound scorer + honeypots), adsbGovernor.ts (outbound ADS-B RPS), tiles3dBudget.ts (outbound Google 3D tiles cost cap), aeroCharts `AERO_PREFETCH_RPS` — four different purposes, not true duplicates, but no shared token-bucket primitive.
9. **Tile proxies/bakes**: owmTiles.ts (OWM PNG proxy), tiles3dProxy.ts (Google 3D tiles), aeroCharts.ts + aeroBake.ts (FAA charts; live proxy + R2 bake), `/tiles-r2/:name` in routes.ts — three independent cache/abort/header implementations; the RENDERING & MOTION LAW's "one raster streamer" exists only client-side.
10. **Health surfaces**: healthGate.ts (deploy gate), liveness.ts, scannerHealth.ts, feedDeadAir.ts, crashVisibility.ts, memoryCeiling.ts, eventLoopLag.ts, dbWriteTiming.ts, tmpCleanup.ts — all consumed by /api/health in bot.ts; coherent but spread over 9 files plus `/api/system/health-check`, `/api/daemon/health`, `/api/monitoring/*` endpoints that overlap `/api/health`.
11. **SSE vs WS**: bot.ts `/api/bot/stream` (SSE audit feed) and a `ws` WebSocketServer in routes.ts/bot.ts (`wsPositionFeed.test.ts`) — two push channels.

## 12. Python invocations from server (for the Python audit)

- **Daemon RPC** (`pythonRpc`, Unix socket `VOLTRADE_DAEMON_SOCKET` default `/tmp/voltrade_daemon.sock`, JSON `{method,args}` newline-delimited, 30s default): methods `health` (6 call sites), and via `pythonCall(daemonMethod, args, subprocessFallbackCmd)`: `run_full_scan`, `evaluate_and_execute`, `track_fill` (4), `snapshot_python_state`, `shadow_update_decisions`, `risk_status`, `paper_resume_sync`, `ml_status`, `lookup_ticker`, `lookup_ledger_report`, `get_peak_equity`, `check_halt`, `check_correlation_pre_trade`, `cache_inventory`, `csp_gate_shadow`, `strategy_pnl_stats`. `run_full_scan`, `scan_market`, `manage_positions` are daemon-only (no subprocess fallback). These must all exist in `voltrade_daemon.py`'s method table.
- **Subprocess (`execPythonSerialized`, mutex-serialized)** `python3 -c` snippets import: `storage_config` (14), `ml_model_v2` (track_fill, …) (8), `shadow_portfolio` (backfill_outcomes, backfill_exit_aligned) (6), `options_execution` (5), `diagnostics` (5), `risk_kill_switch` (get_kill_switch_status, get_peak_equity) (4), `tiered_strategy` (2), `system_config` (2), `options_manager` (2), `lookup_ledger` (report, backfill) (2), `intraday_shorts` (close_open_shorts) (2), and once each: `stress_index`, `sec_form4_bulk`, `probability_engine`, `position_sizing`, `paper_resume_sync`, `ml_insight`, `manipulation_detect`, `macro_data`, `lookup_ticker`, `feedback_boot_cleanup`, `csp_universe`, `cftc_cot`, `bot_engine`, `analyze`, `insider_cusum_probe` (scripts/ module imported by name from cwd), plus `requests` (direct Alpaca snapshot).
- **Script entry points**: `python3 ml_status.py`, `python3 ml_toggle.py <action>`, `python3 ml_retrain_safe.py` (routes.ts /api/ml/*), `python3 -W ignore "<bot_engine.py>" full` (bot.ts Tier-2 fallback), `python3 -m alphadesk <TICKER> --json`, `python3 -m alphadesk taxes|plan --payload` (routes.ts /api/research, /api/tax, /api/plan).
- Literal .py names in server text (comments/provenance, not executed): bot_engine.py, system_config.py, voltrade_daemon.py, market_calendar.py, cftc_cot.py, analyze.py, diagnostics.py, shadow_portfolio.py, options_manager.py, ml_model_v2.py, ml_diagnostics.py, gem_ingest.py, alpaca_feed.py, system_status.py, storage_config.py, position_sizing.py, options_execution.py, un_comtrade_ingest.py, sec_form4_bulk.py, hazard_rate_probe.py, scripts/{gem_ingest, reconstruct_position_pnl, jodi_oil, jodi_eia_reconcile, flightplan_gate1, cdc_cancer_rates, wikiattention_gate2, un_comtrade_*, stb_rail, site_verify, rail_traffic_gate1, pfas_ucmr5, insider_cusum_probe, grid_stress_gate2, faa_charts/run, csp_manager_rule_sim, cdse_site_chips}.py.
- Dockerfile ships `*.py`, `strategies/`, `alphadesk/`, `backtest_10yr_results.json`, `run_with_daemon.sh` — scripts/ is NOT in the image, consistent with the server never spawning it.

## 13. Architecture notes

1. **Process model**: one Node process (`dist/index.cjs`, esbuild CJS bundle of server/index.ts; heap sized by run_with_daemon.sh) + one supervised Python daemon (`voltrade_daemon.py`, Unix-socket JSON-RPC, auto-restarted every 2s on exit). Node talks to Python via `pythonRpc` → daemon, falling back to mutex-serialized `python3 -c` subprocesses (`execPythonSerialized`, stale-lock sweeper every tick) for light methods only.
2. **bot.ts scheduler** (inside `registerBotRoutes`): Tier 1 `tier1Reflex()` every 45s during market hours (stops, stale orders, drawdown kill check; while killed it runs `killSwitchAutoResumeTick`); Tier 2 `tier2Intelligence()` on an adaptive timer `getTier2Interval()` (armed via `armTier2Timer`, window 04:00–20:00 ET; `run_full_scan` daemon-only; `VOLTRADE_DISABLE_TIER2` bisection flag); Tier 3 `tier3Strategic()` hourly (3,600,000ms) plus one run 30s after boot (ML retrain, diagnostics, port-dwell weekly fold; `VOLTRADE_DISABLE_TIER3`). Three more module-level intervals: python-lock watchdog, orphaned /tmp cleanup, liveness stamp.
3. **Audit log + SSE**: `audit(type, detail)` → console + `broadcastSSE` to `/api/bot/stream` clients + `persistAudit` → SQLite `audit_log` table (pruned to last 2000 rows). Notifications are in-memory (100). Crash handlers (crashVisibility.ts) also audit.
4. **State**: SQLite `/data/voltrade.db` via better-sqlite3 (users, sessions, watchlists, audit_log; opened in auth.ts, shared handle); billing's `voltradeai.db` in cwd (ephemeral); JSON state files at `/data/voltrade/voltrade_*.json` (equity curve, peak, kill switch, liveness, deploy timestamp, scan timings, analyst budget, refresh attempts) with `/tmp` fallback; Python side via storage_config.py.
5. **Datacore archive** (`datacoreArchive.ts`, 85 importers — the hub): append-only JSONL per UTC hour under `DATA_DIR/datacore_archive/<kind>/`, gzip after ~2h, daily rollups after RAW_RETENTION_DAYS; adaptive thinning near strategic sites; aircraft/vessel/train writers; `archiveOffload.ts` moves cold hours to Cloudflare R2 via `r2Client.ts` (also used by aeroBake and `/tiles-r2`). Intent parsing (`aircraftIntent.ts`), fix bus (`aircraftFixBus.ts`), SWIM plan capture (`swimCapture`), and per-root archives (gnssIntegrityDaily, globalScopes, optionsChainArchive) all write beside it.
6. **Datacore server modules**: ~120 one-file-per-root fetch→normalize→cache→route modules (EDGAR, FINRA, OCC, CFTC, EIA/ENTSO-E grid, GEM, USGS, NASA FIRMS, NOAA, FDA, CDC, etc.) each exposing `/api/data/<root>` and, when ladder-validated, `/api/v1/data|stats/<root>` through `apiProduct.ts` (key-gated via apiKeyAccounts.ts). `dataQuality.ts` gates every layer; `streamsInventory`/`platformStats`/`layerFreshness`/`signalLadder`/`qualityDashboard`/`pipelineHealthHistory` are the honesty dashboards.
7. **Client serving**: production `serveStatic` (hashed assets immutable, index.html no-cache, SPA fallback `/{*path}`); dev `vite.ts` middleware. Vite root = client/, single entry main.tsx, aliases `@`→client/src, `@shared`→shared, `@assets`→attached_assets.
8. **Security edge**: securityHeaders (CSP report/enforce via `SECURITY_CSP_ENFORCE`), requestLog (salted IPs, feeds visitHistory + adminStats `/api/_admin/*`), global + strict-auth rate limiters, antiScraping scorer + honeypots, robots.txt, terms; routeGuards for owner/diag tokens; providerCompliance degrades /api/health when billing is on with a non-commercial ADS-B provider.
9. **Health**: `/api/health` HTTP code decided solely by healthGate.ts (`server`, `database` checks); everything else only sets `degraded`. `scripts/deploy_gate_smoke.mjs` boots the real bundle in CI to assert 200 within 60s.
10. **CI**: path-filtered jobs; `tsc_ratchet.sh` (pin 6, TS2304=0), `npm run build`, `counter_ratchet.sh` (program_status.sh counters), `gated_tests.sh` (263 server + 140 client node:test files, pytest, deploy smoke), docker build, auto-merge for `claude/*` branches. A second workflow mirrors the CelesTrak catalog to a data branch every 6h (server relays it at `/api/data/orbital/*`).
