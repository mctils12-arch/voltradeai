# VolTradeAI repository audit — 2026-10-27

Phases 1–3 of the cleanup brief. Nothing was merged, deleted, or removed.
Evidence tables live next to this file:

| File | What it holds |
|---|---|
| `branches.tsv` | all 908 remote branches: PR state, last commit, author, behind/ahead main, subject |
| `pull_requests.tsv` | all 1,474 PRs: state, merged/closed date, head branch, title |
| `unmerged_branch_content_check.tsv` | for every unmerged branch: files touched and % of its added lines already present in main |
| `git_history_bloat.txt` | paths that account for the 1.2 GB pack |
| `audit_python.md`, `audit_server.md`, `audit_client_data.md` | module-by-module ACTIVE / DEAD / UNCLEAR tables with evidence |
| `ci_local_run.log` | the local run of the exact CI gate (typecheck ratchet, build, counters, all test suites, deploy smoke) |

---

## 1. Git state

### 1.1 Head of main and the working copy

| Item | Value |
|---|---|
| `origin/main` | `f9892473` — `[NO-ACTION] 2026-10-27 daily-midday … (#1474)` |
| `package.json` version on main | 1.0.1168 |
| Commits on main | 1,909 |
| This working copy | clean: no modified, untracked, ignored-but-real, or stashed files; no extra worktrees |
| Local branches | `main` (== origin/main) and the session branch `ccr-5f668b84-lz6g69` (now carries only this audit folder, pushed) |
| Tags | none, local or remote |
| Repo settings | private; squash/merge/rebase all allowed; **delete-branch-on-merge is OFF** (this is why 815 merged branches linger) |

Nothing real was at risk in the working copy, so item 3 of Phase 1 needed no rescue commit. The audit folder itself was committed to `ccr-5f668b84-lz6g69` and pushed.

### 1.2 Pull requests (1,474 total)

| State | Count |
|---|---|
| Merged | 1,427 (all squash-merged by `mctils12-arch` / the CI auto-merge job) |
| Closed, not merged | 46 |
| Open at audit start | 1 — **#1466** `[NO-ACTION] session log 2026-10-26 evening (docs only)` |
| Opened during the audit | **#1475** `[RULE-REVIEW] New probe shadow_bands_dedup …` (a concurrent autonomous session; CI was running on it) |

Merged per month: Apr 77 · Jun 7 · Jul 560 · Aug 298 · Sep 239 · Oct 246.

**#1466** adds 9 lines to `research/experiments.md`. GitHub shows **zero workflow runs** for its head commit, so the auto-merge job (which needs the `changes` job to have run) never fired. It does not conflict with main (docs append). Safe to merge; the fix is simply to merge it by hand or re-push the branch so CI triggers.

**#1475** should be left to the normal auto-merge path.

The 46 closed-unmerged PRs are listed in `pull_requests.tsv`. For almost all of them the *feature* was later re-done in a different PR (see 1.4); the branch, not the work, is what was abandoned.

### 1.3 Branches (908 on origin)

| Bucket | Count | Meaning |
|---|---|---|
| Branch of an already-**merged** PR | **815** | squash-merged; the branch tip is not an ancestor of main but its content is on main. Pure clutter. |
| Branch of a **closed-unmerged** PR | 32 | abandoned or superseded work |
| **No PR ever opened** | 60 | 25 docs-only session-log branches, ~13 code branches, 11 GPU-pod result branches, 1 workflow data branch, misc April-era branches |
| Open PR | 1 (+1 new) | |

Authors: 856 `Claude`, 16 `Claude Agent`, 9 `mctils12-arch`, 9 `gridvision-pod`, the rest are bot/variant names. Age: Apr 52 · May 1 · Jul 234 · Aug 220 · Sep 210 · Oct 191.

### 1.4 Pushed but never merged — what is actually unique

I diffed every unmerged branch against its merge-base and checked (a) whether each file it *adds* exists on main and (b) what share of its added lines already appear in main.

**A. Docs-only session logs (25 branches, no PR).** Each adds one `## 2026-xx-xx … [NO-ACTION|RESEARCH|PIPELINE]` entry to `research/experiments.md` (and sometimes `open_questions.md`/`wishlist.md`) that never reached main. Dates: 09-29, 09-30, 10-01, 10-03 (×2), 10-04, 10-05, 10-06, 10-08 (×2), 10-09, 10-13, 10-14, 10-16, 10-18, 10-20, 10-23 (×2), 10-24, 10-25, 10-26 (×3), 10-27 (×3). Five of the closed PRs (#1238, #1253, #1311, #1186, #604) are the same shape. Decision needed: salvage (append the entries to main in one docs PR, in date order) or accept the gap and delete.

**B. Code branches whose work is NOT on main in any form** (verified by grepping main for the feature, not just the file name):

| Branch | Date | What it is | Status on main |
|---|---|---|---|
| `claude/dazzling-planck-8h76kt` | 2026-07-16 | vessel identity via IMO checksum (`server/vesselIdentity.ts` + test + manifest) | absent |
| `claude/lucid-keller-eej0un` (PR #707) | 2026-08-06 | macro regime series `/data` UI (`client/src/pages/macro.tsx`) | absent |
| `claude/lucid-keller-slocuv` | 2026-07-16 | wind-farm position verification vs OSM (`scripts/verify_wind_positions.py` + test + json) | absent |
| `claude/eloquent-dijkstra-ngf8ct` (PR #343) | 2026-07-07 | manipulation-scan Tier-3 repair (KNOWN BROKEN #14) + `server/manipulationVisibility.test.ts` | test absent; whether the bug was fixed another way needs a look at open_questions #14 |
| `claude/lucid-keller-4o6t07` (PR #350) | 2026-07-07 | `client/src/lib/satelliteOrbits.ts` | superseded by `client/src/lib/orbital/` |
| `fix/tier2-full-scan-oom` (PR #77) | 2026-04-20 | env-gated inline ML retrain + `test_tier2_oom_fix.py` | April-era; bot.ts has moved far since |
| `backtest-pr71-10yr` | 2026-04-19 | 15 files: `backtest_aggressive.py`, sweep result JSONs | pre-dates the 2026-07-03 backtest rebuild; historical only |
| `data-backup` | 2026-04-05 | nightly backup of 8 state JSONs | historical only |

**C. Code branches whose feature WAS re-implemented on main later** (safe to delete; names of the main-side implementation in parentheses): PFAS (`pfas` layer, 17 files), FDIC bank financials (`bankFailures.tsx` + layer), GEBCO confidence (`lib/bathymetry.ts`), GEM methane asset rates (`methaneHotspots.tsx`), steel furnace units (`steel_units` manifest), submarine cables (`client/public/cables/` + build script), SWPC space weather (manifest + layer), MIDAS (29 files), CFTC COT (`cftc_cot.py` + gate tests), critical-slowing-down probe (`scripts/`), microcap cost floor (98 % of lines present), options sizer KNOWN BROKEN #27 (100 %), DD-halt visibility #1038 (98 %), satellite retry fix #420 (95 %), hazard-rate #999 (100 %).

**D. Artifact / data branches (not code):**

| Branch(es) | Keep? |
|---|---|
| `celestial-catalog-data` | **KEEP.** Orphan branch force-pushed every 6 h by `.github/workflows/celestial-catalog-mirror.yml` (CelesTrak GP + SATCAT mirror). Deleting it just makes the next run recreate it. |
| `gridvision-pod-result`, `gridvision-pod-result-gv-div4-ks`, `-gv-div5-ks`, `gv-res-{div,div2,eaug,g1a,g1d}-{az,ks,s}` (11 branches) | GPU training outputs from 2026-07-09/10: `gv_best.pt` (≈16 MB each, 147 MB across 9 copies), `gv_result.txt`, `gv_train_tail.log`, plus ledger/leaderboard JSON that IS on main (60–91 % match). Recommend: tag each as `archive/<branch>` then delete the branch, so the weights stay recoverable without 11 heads. |
| `data-backup`, `backtest-pr71-10yr` | tag + delete (historical) |
| `patch-12-strategy-key-alignment`, `fix/full-audit`, `claude/lucid-keller-v0p1j2`, `claude/lucid-keller-ob9hxo`, `claude/quirky-hopper-932s7z` | 0 commits ahead of main — content already landed; delete |
| `release-2026-04-20`, `revert-cleanup-cmd`, `revert/binary-rain-pr23`, `run-cleanup`, `landing-page`, `enable-ml`, `enhanced-2d-world-map`, `optimized-convexity-puts`, `data`, `ccr-d10290e6…`, `ccr-bafbe08e…`, `feature/*`, `feat/*`, `fix/*` (34) | April/July era; every one with a PR is merged or closed (see `branches.tsv`); delete with the merged bucket |

### 1.5 Repository weight

| Measure | Value |
|---|---|
| GitHub reported size | 1.14 GB |
| Local pack after full fetch | 1.24 GiB |
| `research/experiments.md` | 7.4 MB now; **1,302 historical versions ≈ 4.8 GB uncompressed** (the single biggest contributor) |
| `.pmtiles` committed then deleted | 132 files no longer on main, roughly 1 GB of history (`power_*.pmtiles` per country/state) |
| Binaries still on main | `client/public` 92 MB (`tiles/places.pmtiles` 58 MB, `tiles/seafloor_gebco_mariana.pmtiles` 21 MB, `space/moon_8k.jpg` 5 MB) |
| `gv_best.pt` on pod branches | 147 MB across 9 blobs |

Shrinking history means rewriting it, which touches all 900 branch refs and every PR link. That is a separate decision, not part of this cleanup; deleting the pod branches and running GC on GitHub's side recovers the 147 MB without a rewrite.

---

## 2. Deploy state

| Question | Answer | Evidence |
|---|---|---|
| How does deploy work? | GitHub `main` → Railway GitHub integration → Docker build from `Dockerfile` (multi-stage: `npm ci`, `npm run build`, prod-only deps + python3) → `./run_with_daemon.sh` (Node `dist/index.cjs` + supervised `voltrade_daemon.py`) → healthcheck `GET /api/health` within 60 s | `railway.json`, `railway.toml`, `Dockerfile`, `run_with_daemon.sh` |
| Which commit is live? | The build serving **`server_version` 1.0.1168**, which is `package.json` on main `f9892473`. Every PR bumps the version, so version == commit here. | `curl https://voltradeai.com/api/data/layers` |
| Is live == latest main? | **Yes.** | same |
| Live health | `status: ok`; checks server/database/alpaca/python/bot(active)/scanner/feeds/licensing all `ok`; serving gates `[server, database]`, none failing | `curl https://voltradeai.com/api/health` (19:33 Z) |
| Railway CLI / API | not available in this sandbox (no `railway` binary, no `RAILWAY_*` env, no token). The version probe above is the authoritative substitute and is the same method the project's own sessions use. | |
| On GitHub but NOT live | exactly the unmerged items in 1.4 (PR #1466's 9 doc lines; the 25 orphan session logs; the code branches in 1.4-B). Nothing merged is undeployed. | |
| CI as a deploy gate | CI runs on every PR and gates auto-merge. **Push-to-main runs do not fire for auto-merged PRs** (GitHub never triggers workflows from `GITHUB_TOKEN` pushes; the last push-event CI run on main is 2026-10-19 from a human merge). Railway's "wait for CI" would therefore see no check suite on most main commits; the PR-level gate is the real one. Not a defect today, but worth knowing. | `actions/runs?event=push` |

Local run of the full CI gate on main (`ci_local_run.log`): typecheck ratchet OK (6 pinned errors, 0 new), `npm run build` OK, counter ratchet OK, server suite OK (263 files), client suite OK (140 files), pytest **2,834 passed / 2 skipped**, deploy-gate smoke boots the real bundle and answers 200. Quarantine 0/1.

---

## 3. System overview

### 3.1 Architecture and data flow

```
                GitHub main ──► Railway (Dockerfile) ──► one container
                                                           │
   ┌──────────────────────────── run_with_daemon.sh ───────┴────────────────────────────┐
   │                                                                                      │
   │  Node  dist/index.cjs  (server/index.ts, esbuild CJS bundle, heap = 60% of cgroup)   │
   │   ├─ securityHeaders · requestLog · rateLimit · antiScraping                         │
   │   ├─ auth.ts  (SQLite /data/voltrade.db: users, sessions, watchlists, audit_log)     │
   │   ├─ routes.ts  (221 route registrations: /api/data/* 143, /api/v1/* 60, …)          │
   │   ├─ bot.ts  THE ORCHESTRATOR                                                        │
   │   │    Tier 1 every ~45 s (stops, stale orders, drawdown kill check)                 │
   │   │    Tier 2 scans on a time-of-day cadence (scan_market → deep_score → execute)    │
   │   │    Tier 3 hourly (ML retrain, macro, diagnostics, ladder gate jobs)              │
   │   │    audit() → console + SSE /api/bot/stream + SQLite audit_log                    │
   │   │    /api/health (HTTP code decided only by healthGate.ts: server + database)       │
   │   ├─ datacoreArchive.ts  (hub, 85 importers): append-only JSONL per UTC hour under   │
   │   │    DATA_DIR/datacore_archive/<kind>/, gzip, rollups, R2 offload                  │
   │   ├─ ~120 datacore modules: fetch → normalize → cache → /api/data/<root> (+ /api/v1  │
   │   │    when ladder-validated): EDGAR, FINRA, OCC, CFTC, EIA/ENTSO-E, GEM, USGS, FIRMS…│
   │   └─ serveStatic: hashed assets, SPA fallback → client/                              │
   │                                                                                      │
   │  Python  voltrade_daemon.py  (Unix-socket JSON-RPC, restarted 2 s after any exit,    │
   │   self-kills > 1 GB RSS; subprocess fallback `python3 -c` if the socket is down)      │
   │   methods used by bot.ts: health, run_full_scan, evaluate_and_execute, track_fill,    │
   │   snapshot_python_state, shadow_update_decisions, risk_status, paper_resume_sync,     │
   │   ml_status, lookup_ticker, lookup_ledger_report, get_peak_equity, check_halt,        │
   │   check_correlation_pre_trade, cache_inventory, csp_gate_shadow, strategy_pnl_stats   │
   │     ├─ bot_engine.py (scan_market, deep_score, manage_positions, floors, convexity)   │
   │     ├─ ml_model_v2.py (LightGBM, triple-barrier labels, purged CV, regime)            │
   │     ├─ system_config.py (ALL tunables, regime-adaptive)                              │
   │     ├─ risk_kill_switch.py (halts, correlation, liquidation — frozen mechanisms)      │
   │     ├─ options_scanner / options_manager / options_execution / csp_universe          │
   │     ├─ strategies/ (momentum, mean_reversion, squeeze)                               │
   │     ├─ data modules: alt_data, macro_data, social_data, institutional_data,          │
   │     │   intelligence, finnhub_data  (wired, but every call is inside try/except → {}) │
   │     └─ alphadesk/ (EDGAR filings / catalyst engine, `python3 -m alphadesk`)           │
   │                                                                                      │
   │  State: /data/voltrade/*.json (equity curve, fills, feedback, positions…),           │
   │         /data/voltrade.db (SQLite), datacore archive JSONL, R2 for tiles/offload      │
   └──────────────────────────────────────────────────────────────────────────────────────┘
          ▲ REST only (fetch/apiRequest, polling; SSE used by the bot audit stream)
   React client (Vite, wouter): / landing · /app hash-tabbed shell (analyze, research,
   scanner, news, watchlist, #/data map, planner, taxes, #/bot owner console) · /pricing
   /about /newsletter /careers /contact /developers /apikeys /admin /login
   #/data = pages/datamap.tsx (MapLibre GL globe, 257 layers from datacore/layers.json,
   custom WebGL layers for satellites/celestial, 50 #/data/<overlay> sub-pages)
```

### 3.2 How UI and bot interact

- The bot never talks to the browser directly. `bot.ts` writes state (audit rows, equity curve, positions, fills) to SQLite/JSON; `routes.ts`/`bot.ts` expose it as `/api/bot/*` (33 routes) and the owner-only `#/bot` console (`client/src/bot/BotPage.tsx`) polls those plus the SSE audit feed `/api/bot/stream`.
- `/app` analysis tabs call `/api/analyze`, `/api/etf/*`, `/api/scanner/*`, which in turn spawn or RPC Python (`analyze.py`, `etf_data_sources.py`, `lookup_ticker`).
- The data platform (`#/data`, `/api/data/*`, `/api/v1/*`) is served entirely from the Node datacore modules; the bot consumes the same `/api/v1` signals as a customer would (SPINOUT-READY rule).
- Python never schedules anything itself; all cadence lives in `bot.ts` tiers.

### 3.3 Folder map

| Path | Role | Size |
|---|---|---|
| `server/` | Node server: orchestrator, routes, ~120 datacore modules, 263 tests | 485 files |
| `bot_engine.py`, `ml_model_v2.py`, `system_config.py`, `risk_kill_switch.py`, `options_*.py`, `voltrade_daemon.py` + ~45 helpers | Python trading engine (root level) | 292 root `.py` of which ~220 are tests/CLIs |
| `strategies/`, `alphadesk/` | strategy modules; EDGAR research package | 4 / 25 files |
| `client/src/` | React app; `pages/datamap.tsx` is 16,628 lines | 486 files incl. `client/public` 99 MB |
| `client/src/render/`, `client/src/lib/orbital/`, `client/src/lib/celestial/` | rendering contract (frameCore, zoomInput, tileCore, perfMetrics), SGP4 satellite stack, celestial sky | |
| `datacore/` | pure data: 143 json / 7 jsonl / 7 gz + 5 md; `layers.json` (257 layers), `signal_ladder.json`, `manifests/` (74, enumerated at runtime) | 163 files, 79 MB |
| `scripts/` | 178 tooling files (gate probes, bakes, visual harness, runpod, migrations); none ship in the Docker image or run at runtime (one exception: `scripts/statsUtils.ts` is bundled via `server/gnssIntegritySignal.ts`) | |
| `script/build.ts` | the build (Vite client + esbuild server + copy of runtime datacore files) | |
| `research/` | constitution-adjacent logs: `experiments.md` 7.4 MB, `open_questions.md` 1.5 MB, `wishlist.md`, program charters | 78 files, 14 MB |
| `ci/` | ratchet pins (`tsc_baseline.txt`, `counter_baseline.txt`, `quarantine*.txt`) | |
| `.github/workflows/` | `ci.yml` (path-filtered gate + auto-merge of `claude/*` PRs), `celestial-catalog-mirror.yml` | |

---

## 4. ACTIVE / DEAD / UNCLEAR

Full per-module tables with importer chains are in the three `audit_*.md` files. Totals:

| Area | Modules examined | ACTIVE | DEAD | UNCLEAR |
|---|---|---|---|---|
| Python (non-test, non-scripts) | 89 | 75 (root 54, strategies 4, alphadesk 17) | 9 | 3 (+2 tooling-only) |
| Server TS (non-test) | 217 | 209 | 8 | 0 |
| Client TS/TSX (non-test) | 283 | 233 (88.6 k lines) | 50 (6.2 k lines) | a handful of assets |
| `scripts/` | 178 | 6 CI/npm + 8 transitive + 90 invoked by other code + 43 tested-only | 1 orphan | 41 referenced only in docs |
| `datacore/` data files | 163 | all manifests + everything a server/client file names | 6 (≈14 MB) | 1 orphan layer id |

### 4.1 DEAD — Python

| Module | Lines | Evidence | Superseded by |
|---|---|---|---|
| **`ml_model.py`** | **1,149** | zero importers in py/ts/sh; not in daemon table; not in CI; only string mentions ("retired") | `ml_model_v2.py` |
| `sim_trading_day_v2.py` | 334 | no importers; executes on import; 3 research mentions | `test_full_system.py`, `scripts/session_health_check.py` |
| `cleanup_feedback.py` | 90 | one-off; 1 test importer | `feedback_boot_cleanup.py` + ml_model_v2 startup cleanup |
| `bot_backtest_subperiods.py` | 121 | 1 test importer; result already filed in open_questions | — |
| `cftc_tff_tlt_disjoint_replication.py` | 99 | 1 test importer | — |
| `cot_gate2_oos.py` | 67 | 1 test importer | — |
| `regime_detector_compare.py` | 277 | 1 test importer; one-shot comparison | — |
| `sec_filings_classify.py` / `sec_filings_similarity.py` / `sec_filings_gate2.py` | 113 / 336 / 309 | tests only; Lazy Prices research chain, blocked on a paid price panel (wishlist) | — |

Also found: `voltrade_daemon.py` maps `ml_status` / `ml_toggle` to modules `ml_status_impl` / `ml_toggle_impl` that **do not exist**, so the inline fallback always runs and `routes.ts` still spawns `python3 ml_toggle.py` directly. Not dead code, but a latent repair item.

### 4.2 DEAD — server

`server/storage.ts` (5, template stub), `shared/schema.ts` (5, `export {}`), `server/buoyPortProximity.ts` (140), `server/countryLookup.ts` (147), `server/gasFlareCandidates.ts` (213), `server/cusipResolver.ts` (181), `server/fleetOperatorTickers.ts` (40), `server/gridDemandRevision.ts` (116), `server/shadowFleetReference.ts` (102). All have zero non-test importers; seven are ladder gate-1/2 helpers whose root died at a gate, and seven still have CI-run tests keeping them "alive". Largest unreferenced scripts (>300 lines, docs-only): `scripts/gdelt_fires_gate2.ts`, `treasury_dts_gate2.ts`, `finra_shortvol_gate2_retest.ts`, `settlement_stress_gate2.ts`, `github_activity_gate2.ts`, `labeler_before_after.py`; true orphan: `scripts/faa_charts/make_fixture.py`.

### 4.3 DEAD — client

- **45 shadcn `client/src/components/ui/*` files** (4,557 lines) from the 2026-04-03 scaffold; only `toast.tsx` + `toaster.tsx` are imported. They are the sole reason 34 npm packages are installed (see §5).
- `client/src/components/TradingActivity.tsx` (392), `client/src/pages/not-found.tsx` (17, never routed), `client/src/hooks/use-mobile.tsx` (19).
- `client/src/render/tileCore.ts` (884) and `client/src/render/perfHud.ts` (273): **no non-test importer since they shipped on 2026-08-12.** These are the Rendering & Motion Law's "contract" code (CLAUDE.md Amendment 6); `lib/celestial/moonTiles.ts` rolls its own streamer instead. I am classifying these **UNCLEAR, decision needed**, not DEAD: deleting them would quietly retire a constitutional article; the honest options are "wire them" or "amend the article".
- Unreferenced assets: `client/public/space/earthmap1k.jpg`, root `newsletter-{hero,mockup,logo}.png` (byte-identical copies live under `newsletter/images/`), `attached_assets/login_bg.png` (2.9 MB).

### 4.4 DEAD — data

`datacore/gem/{cement,oil_ngl_pipelines,iron_units,lng_terminals}.json`, `gem/coal_finance.json.gz` (ingested, never served; ≈13 MB). `gem/power_units.json.gz` has one reference and stays UNCLEAR. `datacore/layers.json` has exactly one orphan id, `tank_fill` (status `planned`, correctly gated). `research/` strays: `vrp_single_name_vs_index_2026-10-20.json` (1.2 MB, referenced nowhere), `csp_manager_rule_sim_2026-10-26_entries.json` (2.1 MB), and two runtime state files (`outage_state.json`, `liveness_notify_state.json`) that belong under `/data`, not in git.

### 4.5 Root-level strays

`README.md` is a stale "ETF Builder drop" note, not a project README. `HANDOFF.md`, `BOOTSTRAP_PROMPT.md`, `INTEGRATION_NOTES.md`, `side_effect_analysis*.md` (2), three `.bat` launchers and a `.ico` (Windows), `components.json` points at a tailwind config that does not exist, `package-lock.json` says 1.0.1131 while `package.json` says 1.0.1168 (npm ci still works; it is just drift). No `.dockerignore` and no `.env.example`.

---

## 5. Dependencies, env vars, duplicates, tests

### 5.1 npm (68 deps + 18 dev)

- **37 unused dependencies**: 26 `@radix-ui/react-*` (everything but `react-toast`), `cmdk`, `embla-carousel-react`, `input-otp`, `react-day-picker`, `react-hook-form`, `react-resizable-panels`, `recharts`, `vaul` (all pulled in only by the dead shadcn files), plus `date-fns` and `zod` (zero imports anywhere).
- **2 unused devDependencies**: `@tailwindcss/typography`, `@tailwindcss/vite`.
- `@types/bcryptjs`, `@types/better-sqlite3`, `@types/cookie-parser` sit in `dependencies` (should be dev).
- Missing from `package.json` but imported: `nanoid` (dev-only `server/vite.ts`, transitive today), `topojson-specification` (type-only), `@maplibre/maplibre-gl-style-spec` and `@tanstack/query-core` (client tests).
- `script/build.ts`'s esbuild allowlist names 20 packages that are not dependencies at all (`drizzle-orm`, `axios`, `cors`, `jsonwebtoken`, …) — harmless residue.

### 5.2 Python

All 10 `requirements.txt` entries are imported by ACTIVE code; nothing undeclared is on the runtime path. `laspy` and `ultralytics` (GPU tooling under `scripts/`) are missing from `requirements-dev.txt`.

### 5.3 Environment variables

87 distinct names on the Node side, ~40 on the Python side; **62 are documented nowhere** (there is no `.env.example`; README names 3). None is read only by dead code. Same credential under two names: `POLYGON_KEY` / `POLYGON_API_KEY`; `ENTSOE_API_KEY` / `ENTSOE_TOKEN`; `FIRMS_MAP_KEY` / `NASA_FIRMS_MAP_KEY`; `ALPACA_KEY/SECRET` vs alphadesk's `APCA_API_KEY_ID/SECRET_KEY`. The live Railway variable set cannot be read from here, so "unused in Railway" could not be checked; full lists are in `audit_server.md §10` and `audit_python.md §8`.

### 5.4 Duplicate implementations (highest-value groups)

1. **ML**: `ml_model.py` (dead v1) vs `ml_model_v2.py`.
2. **Backtest engines**: `backtest.py`→`backtest_v2.py` (live) vs `bot_backtest.py` (+`_subperiods`) vs `sim_trading_day_v2.py`.
3. **Regime detection ×3**: `markov_regime`, `regime_util`, `regime_inputs` (+ ml_model_v2's own).
4. **Alpaca HTTP**: `alpaca_feed.py` is the declared central feed, yet 25 modules still call `alpaca.markets` directly.
5. **Insider/EDGAR ×7** across `intelligence`, `finnhub_data`, `sec_form4_bulk`, `institutional_data`, `alt_data`, `alphadesk/edgar`, Lazy-Prices chain — plus `server/edgarForm4.ts`.
6. **Two SQLite databases**: `auth.ts` on the `/data` volume; **`billing.ts` opens `voltradeai.db` in `process.cwd()`, which is ephemeral on Railway** (frozen path — flagging, not touching).
7. **DATA_DIR resolution** re-implemented in 4 TS files and ~14 inline Python snippets, with two different rules.
8. **Request logging twice** (`requestLog.ts` + inline logger in `index.ts`); **two push channels** (SSE + `ws`).
9. **Seven GEM loader modules** and **three tile proxies** each with their own cache/abort/header code.
10. **Client**: 42 `#/data/*` overlay pages each re-roll the same header/fetch/loading scaffold; 4 age formatters; per-page `fmtPct/fmtMoney`; two mobile hooks; two chart stacks; three globes; three tile streamers; the landing page exists both as React and as raw `.html.txt/.css.txt/.js.txt`.

### 5.5 Tests

Everything required by CI is green (see §2). Stale or mis-wired:

- Python: 5 tests guard dead modules (`test_bot_backtest_subperiods`, `test_cftc_tff_tlt_disjoint`, `test_cot_gate2_oos`, `test_regime_detector_compare`, `test_sec_filings_gate2`) + 4 Lazy-Prices tests for research-only code; 4 CLIs (`cftc_tff_gate2_test.py`, `cot_gate2_test.py`, `form4_gate2_test.py`, `scripts/jodi_gate2_test.py`) are collected by pytest with 0 test functions.
- Server: 7 tests keep the 7 dead modules alive; 56 tests import no module (source-text regex pins); **13 `scripts/*.test.{ts,mjs}` are never run** (the gate collects only `server/` and `client/`).
- Client: 3 tests certify unwired code (`tileCore`, `perfHud`, `mapBaseConfig`).
- Typecheck: 6 pinned errors (`datamap.tsx` ×2, `billing.ts` ×3 — Stripe type drift, `owmTiles.ts` pngjs types).
- The constitution's Law-I static grep (`on('zoomend'|'moveend'…)` → fail) is implemented nowhere; `datamap.tsx` has 9 such handlers doing visual work.

---

## 6. Recommended actions, in order (this is the Phase 4 plan to approve)

Each code removal is its own small commit/PR, per CLAUDE.md's one-change rule and its STALENESS AUDIT rule ("findings become removal PRs"; tests of a removed module go with it). Concurrent autonomous sessions are live (PR #1475 opened mid-audit), so branch deletion must skip any branch with an open PR at execution time.

**Step 0 — safety.** Tag `pre-cleanup-2026-10-27` on `origin/main` (and `archive/<branch>` tags on the 11 GPU-pod branches, `data-backup`, `backtest-pr71-10yr` before they go). Ask you to flip the repo setting *Automatically delete head branches* (owner-only; it stops the 815-branch pile from regrowing).

**Step 1 — PRs.** Merge #1466 (docs, 9 lines). Leave #1475 to auto-merge.

**Step 2 — branches (needs your call on A).**
- 2a. Delete the **815** merged-PR branches (zero risk).
- 2b. The **25 docs-only session-log branches** + 5 docs-only closed PRs: *recommended* salvage — one docs PR that appends their `experiments.md` entries in date order, then delete. Alternative: delete without salvage.
- 2c. Delete the closed/no-PR code branches whose feature is on main (1.4-C) and the 0-ahead branches and April-era leftovers (1.4-D last two rows).
- 2d. The 8 branches with unique work (1.4-B): *recommended* tag `archive/…` + delete for all except `vesselIdentity`, `macro.tsx` and `verify_wind_positions`, which you may want revived as fresh PRs. Your decision.
- 2e. Keep `celestial-catalog-data`.

**Step 3 — dead code, in this order (small commits, test gate after each).**
1. `ml_model.py` + the two string references in `test_auto_discovery.py` / `test_silent_except_ratchet.py` baseline.
2. `sim_trading_day_v2.py`, `cleanup_feedback.py` (+test).
3. Research one-shots → either delete or move to `scripts/` with their tests: `bot_backtest_subperiods.py`, `cftc_tff_tlt_disjoint_replication.py`, `cot_gate2_oos.py`, `regime_detector_compare.py`. Rename the 4 `*_gate2_test.py` CLIs so pytest stops collecting them.
4. Server: 8 dead modules + their 7 tests; `scripts/faa_charts/make_fixture.py`.
5. Client: 45 shadcn `ui/*` files, `TradingActivity.tsx`, `not-found.tsx`, `use-mobile.tsx`, 4 unreferenced images + `attached_assets/login_bg.png`.
6. `npm uninstall` the 37 + 2 packages; move 3 `@types/*` to dev; add `nanoid` explicitly; regenerate `package-lock.json` (fixes the version drift too).
7. `datacore/gem/*` dead data (≈13 MB) and the two `research/*.json` dumps; move the two runtime state files out of `research/` (code change in whatever writes them — needs a quick look).
8. Root strays (`.bat`, `.ico`, `HANDOFF.md`, `BOOTSTRAP_PROMPT.md`, `INTEGRATION_NOTES.md`, `side_effect_analysis*.md`); fix `components.json`; add `.dockerignore` and `.env.example` (generated from the env-var tables).
9. Repair (with a regression test): daemon phantom `ml_status_impl`/`ml_toggle_impl` entries; `requirements-dev.txt` += `laspy`, `ultralytics`; add `scripts/**/*.test.*` to the gate or delete those 13 tests.

**Step 4 — verify.** Full local gate, push, confirm Railway serves the new `server_version`, confirm `/api/health` ok and bot `active`.

**Step 5 — docs (Phase 5).** Rewrite `README.md`; update CLAUDE.md's CODEBASE MAP / KNOWN STATE (factual sections) and add your three standing rules as a dated, human-approved section (CLAUDE.md's own text says rule additions need human approval — your instruction is that approval).

**Not in this cleanup — flagged for a separate decision:**
- `tileCore.ts` / `perfHud.ts`: wire or amend (constitutional).
- History rewrite to drop ~1 GB of deleted `.pmtiles` and the 4.8 GB of `experiments.md` versions; moving `client/public/tiles/*.pmtiles` (79 MB) to R2.
- `experiments.md` at 7.4 MB keeps growing by ~5 KB per session; split by month or move logs out of the code repo.
- `datamap.tsx` at 16,628 lines; the 42 copy-pasted overlay pages.
- `billing.ts` SQLite in cwd (frozen path; data loss on every deploy if billing is ever enabled).
- Consolidating the duplicate credential names (needs the Railway variable list).
