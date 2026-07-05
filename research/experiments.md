# Experiment Log

Append-only. Newest at top. Never rewrite history (CLAUDE.md — MEMORY PROTOCOL).
Each entry: date · change · version tag · backtest result · hypothesis · (later) live-vs-backtest.

## AUDIT REGISTER (maintained in place per the AUDIT CYCLE clause,
CLAUDE.md SESSION BUDGET — this block is updatable state, the only
exception to append-only; the log below it stays append-only)

| audit | cadence | last run |
|---|---|---|
| staleness audit (code/deps/config/expired adapters — DEAD CODE POLICY governs) | 30d | 2026-07-05 COMPLETE (both sides). Server side: all 23 env reads wired; adapters none expired, next review 2026-08-17. Python side: requirements.txt zero unused; 6 session-run deps undeclared → requirements-dev.txt; VOLTRADE_STATE_DIR dead env write removed; vacuous-pass test sweep 1 low fix, 3 judged acceptable (see log entry). Next full pass due 2026-08-04 |
| constitutional audit (rules — CONSTITUTIONAL HYGIENE governs) | 30d | 2026-07-04 (human-directed CONSTITUTIONAL REPAIR: 4 proposals filed in wishlist.md, awaiting approval) |
| market_calendar year-add (FROZEN PATHS exception governs) | December | 2026 dates present; add 2027 in Dec 2026 |

## 2026-07-05 — [PIPELINE] Wikimedia pageviews attention stream (BUILD ORDER 5 #3) — curated seed + /api/data/attention (v1.0.135)

- TERRITORY: T-DATACORE. The PYTRENDS REPLACEMENT (gate-1 FAIL #215):
  server/wikiAttention.ts polls the keyless Wikimedia pageviews REST
  API (en.wikipedia, all-access, agent=user — bot traffic excluded at
  the source) for a curated 23-ticker seed, 7-day window, 12h poll.
- CURATION HONESTY: datacore/wiki_articles.json — EVERY pair was
  hand-probed against the live API before inclusion (24 probed, 23
  kept). RIOT dropped at curation: the article was renamed and the
  pageviews API does not follow redirects (both candidate titles
  failed — one 404, one valid-but-no-data). Expansion rule embedded
  in the file: no new pair without a passing probe; true small caps
  often lack articles and that ABSENCE IS DATA. Observed live: quick
  bursts 429 — poller spaces requests >=600ms (test asserts the
  constant respects the observed limit).
- Docker image rule: the seed is a STATIC IMPORT (bundled) — the
  frozen Dockerfile never copies datacore/, so a runtime disk read
  would serve nothing in prod (entity-spine lesson #226).
- PANEL DISCIPLINE: the served day is the newest with a MAJORITY of
  the seed present, so an in-progress publish day (2 of 23 articles)
  never masquerades as the panel (test-pinned). v1 serves RAW daily
  views only — no z-scores or spike labels until the archive holds
  the trailing history to compute them honestly AND gate 1 passes.
- GATE 1 (pre-stated): on 10 hand-checked tickers, views series must
  spike on known event dates (earnings, major announcements) vs the
  surrounding baseline; article-identity errors (wrong company) are
  an automatic fail for that pair. Run after ~2 weeks of archive.
- HYPOTHESIS (gate-locked): attention spikes lead volume/vol 1-5d,
  most interesting on smaller names without same-day news
  (attention-without-news subset). Prior ~30% stated at filing.
- Tests 5/5 (seed bundling + RIOT-absent honesty, API-shape parse,
  one-request-per-article + 404-absence, dedup by view day + 4d gz
  with corrected fixture arithmetic, majority panel-day rule);
  manifest battery 3/3; tsc 64 baseline; pytest 397/1.

## 2026-07-05 — [PIPELINE] CFTC COT disaggregated stream (BUILD ORDER 5 #2) — keyless Socrata archiver + /api/data/cot (v1.0.134)

- TERRITORY: T-DATACORE. server/cftcCot.ts polls the CFTC Public
  Reporting Socrata dataset 72hh-3qpy (disaggregated futures-only,
  ~274 markets/week, Tuesday as-of / Friday ~15:30 ET publish).
- SOURCE CHOICE WORTH REMEMBERING: the build order named the
  f_disagg.txt flat file (probed 200/442KB), but inspection showed
  it is HEADERLESS positional CSV — parsing ~70 columns by position
  is exactly the guess the query-shape honesty rule forbids. The
  Socrata endpoint serves the SAME data keyless with NAMED fields;
  built against that instead. The names have real quirks, verified
  live and encoded in a FIELD constant with a comment each:
  swap__positions_short_all / swap__positions_spread_all carry a
  DOUBLE underscore; several fields drop the _all suffix
  (prod_merc_positions_long, m_money_positions_spread). A test
  fixture mirrors the quirky shape so a silent source rename fails
  loudly.
- WEEK DISCIPLINE: a DESC-ordered fetch can straddle two report
  weeks at the publish boundary — parseCot keeps ONLY the newest
  report_date so vintages never mix in one archive file
  (test-pinned). Week-level dedup; gz after 9 days (a report stays
  plain until superseded); restart rebuilds the cache from the
  newest archived week even with the fetch down (test-pinned).
- Route /api/data/cot serves the poller's cached week (274 rows,
  event-loop rule) with the futures-ONLY caveat and the honest note
  that positioning-extreme signals need trailing history the archive
  is only beginning to accumulate — accumulation substitutes for
  purchase (vendors sell exactly this series recorded over time).
- HYPOTHESIS (gate-locked): managed-money net-positioning extremes
  (percentile vs trailing history) mean-revert in commodity-linked
  ETFs; joins EIA petroleum/natgas + tank-fill work. Prior ~30%
  stated at filing. Gate 1 design when history depth allows:
  archived weeks vs CFTC's own historical annual files on a sampled
  quarter.
- Tests 5/5 (quirky-name fixture, ''->null, week-boundary keep-only-
  newest, week dedup + gz + gz readback, restart-rebuild with fetch
  down); manifest battery 3/3; tsc at the 64 baseline; pytest 397/1.

## 2026-07-05 — [PIPELINE] FINRA daily short-sale volume stream (BUILD ORDER 5 #1) — keyless CNMS archiver + /api/data/short-volume (v1.0.133)

- TERRITORY: T-DATACORE. server/finraShortVolume.ts: keyless daily
  CNMS file (~12.2K symbols/trading day, format verified live —
  pipe header, fractional share counts, bare row-count trailer;
  weekend/holiday URLs 403 = valid not-published). 6h poll with
  7-day lookback newest-first; eager boot (KNOWN BROKEN #9).
- TWO DESIGN DECISIONS WORTH REMEMBERING: (1) DEDUP IS DATE-LEVEL,
  not per-row — the file is atomic and final once published, and
  seeding 12K keys/day x 40d would waste ~50MB in the RSS-capped
  process; if FINRA reposts a corrected file we keep the first
  capture (stated in the manifest, not hidden). (2) TRAILER
  INTEGRITY GATE — the file's own row-count trailer must equal
  parsed rows or the whole file is refused, so a truncated CDN
  download can never poison the archive.
- RESTART HONESTY: on boot with the newest day already on disk, the
  summary cache rebuilds FROM the archive instead of serving
  warming_up until the next publish (test-pinned, no refetch).
- Route serves the poller's cached day summary ONLY (event-loop
  rule): aggregate short ratio + top-30 by ratio with a stated
  500K-share total-volume floor. LABEL HONESTY: this is short-marked
  EXECUTION volume (flow proxy), NOT short interest — route note +
  manifest confidence_model both say so explicitly.
- HYPOTHESIS (gate-locked): small-cap short-ratio extremes/deltas x
  13F+Form4 joins = squeeze-candidate screen. Prior ~35% (stated at
  filing). GATE 1 next: parsed ratios vs FINRA's own monthly
  aggregates on a sampled month; 1-2y session-side backfill after
  gate 1.
- Tests 6/6 (real-format fixture incl. trailer guard + truncation
  refusal, 403-vs-500 semantics, date dedup + gz + gz-readback,
  summary floor/cap honesty, restart-rebuild-no-refetch); manifest
  battery 3/3; pytest 397/1 skip. tsc NOTE: baseline is now 64 on
  main itself (was 63; the +1 is client/src/pages/datamap.tsx:2143
  from another session's merge, verified by stashing my changes) —
  this change adds zero new errors.

## 2026-07-05 — [RESEARCH] BUILD ORDER 5 filed — new roots at microstructure + attention + freight friction, all sources probed live first (docs)

- Standing directive: T-DATACORE queue emptied (BUILD ORDER 4
  resolved for this territory; remainders wait on external clocks or
  belong to T-BOT/T-CLIENT), so the session generated the next build
  order itself. Full text with per-item hypotheses, priors, and gate
  designs in open_questions.md BUILD ORDER 5.
- PROBED BEFORE FILING (all keyless, HTTP status + size recorded):
  FINRA daily short-sale volume CNMS file 200/540KB; CFTC COT
  disaggregated f_disagg.txt 200/442KB (legacy deacot.txt 404 — the
  disaggregated report is the build target); Wikimedia pageviews API
  200 with real daily counts (Nvidia article, June 2026); FAA
  airport-status API 200; CBP border-wait API 200. USPTO deliberately
  filed unprobed-last with a probe-first instruction (may be
  key-gated → Census/BLOCKED-FOR-MIKE precedent, never build blind).
- Order rationale: (1) FINRA short volume first — every-equity
  coverage including the small-cap tail (EDGE DOCTRINE #2), joins
  13F+Form4 for a free squeeze-candidate screen, and dated files
  allow 1-2y session-side backfill; (2) CFTC COT — accumulation
  substitutes for purchase (vendors sell exactly this file recorded
  over time), named in the EDGE DOCTRINE since day one; (3)
  Wikimedia pageviews — the pytrends replacement after its gate-1
  FAIL (#215); (4)/(5) FAA + CBP — cheap honest RAW layers that
  close the air-ops and road-freight gaps; (6) USPTO last.
- Priors stated per item (30-35% for the signal-bearing roots, 20%
  for the friction archives) BEFORE any data is scored, per
  Reasoning Standard #10; every hypothesis enters the ROOT
  VALIDATION LADDER before belief or surfacing.

## 2026-07-05 — [PIPELINE] Census imports VERIFIED LIVE on prod — 686 records, first query variant correct, key confirmed in Railway (docs)

- ~30 min after #249 merged, /api/data/imports served 686 records:
  April 2026 port-level import values with cnt_val/cnt_wgt POPULATED
  — the first QUERY_VARIANT (full containerized set) was correct;
  the anticipated fix-shape-from-prod-logs path was never needed.
  LIVE VERIFICATION PENDING → COMPLETE, ahead of the planned
  first-DAILY-session check.
- Key location resolved: prod answered warming_up (not
  enabled:false) on the very first probe → CENSUS_API_KEY IS in
  Railway. A transient 502 mid-verification was #250's redeploy
  restarting the app, not a stream defect (health 200 immediately
  after; data followed).
- Data note: Census publishes a national aggregate row (port "-",
  "TOTAL FOR ALL PORTS") alongside per-port rows; the header-driven
  parse keeps it as published — consumers filtering to real ports
  should exclude port "-". Kept raw per the as-published discipline.
- Wishlist BLOCKED-FOR-MIKE #6 → DONE-VERIFIED. Ladder status:
  archive accumulating; gate 1 (readings vs a second official
  source) still to be designed before any signal claim.

## 2026-07-05 — [RESEARCH] Staleness audit Python-side pass COMPLETE — deps clean, 6 undeclared session deps filed, 1 dead env write removed, vacuous-pass sweep judged (docs+tests, no runtime change)

- Closes the register's UNSCANNED half (Python deps/config +
  requirements-vs-imports), fanned out to two subagents with judgment
  retained in the parent per WORKSTREAM PARTITION.
- DEPS: requirements.txt has ZERO unused entries (every package
  imported; pytrends stays per its documented re-probe trigger;
  anthropic conditional behind ANTHROPIC_API_KEY in alphadesk).
  Reverse direction found 6 imported-but-undeclared packages — all
  session-run scripts/ or test tooling, none on any runtime path:
  pytest, xlrd, openpyxl, tifffile, rasterio, Pillow. Filed in NEW
  requirements-dev.txt (Dockerfile is frozen and installs
  requirements.txt only — runtime image unchanged by design).
- ENV VARS (Python side): full inventory taken; every var read at a
  live call site EXCEPT VOLTRADE_STATE_DIR — written once in
  test_patches_verification.py ("if storage_config respects it") and
  read NOWHERE. Worse than dead: the comment claimed a tempdir
  sandbox protected production state, but the write was a no-op, so
  peak-equity writes were hitting the real local state dir all along
  (harmless — the assertions are >= ratchets). Removed the dead
  write + false comment + now-unused tempfile import; honest NOTE
  left in place. Two parallel data-dir vars noted (DATA_DIR and
  VOLTRADE_DATA_DIR both live) — mild inconsistency, not debt; no
  action.
- VACUOUS-PASS SWEEP (queued by the R6 lesson): 26 test files
  swept. Read-before-write review DOWNGRADED the subagent's two
  loudest findings: t_vxx_panic_setup (test_full_system.py) asserts
  the hours-gate itself in its outside-hours branch (PASS only on
  None, WARN otherwise) — deliberate two-mode live harness, NOT the
  TestFix7 class; the two test_diagnostic_false_positives.py
  methods are negative-claim tests where an empty match set is the
  correct pass, with a sibling pinning the source-level invariant
  unconditionally. One real (low) fix shipped:
  test_voltrade_daemon.py::test_every_local_route_resolves gained
  the same checked>0 zero-iteration guard its sibling already had.
  No AssertionError-swallowing anywhere.
- No version bump: docs + test files + a dev-only requirements file;
  nothing in the runtime image changes.

## 2026-07-05 — [PIPELINE] Everything Graph R5 step 1 — datacore/entity_map.json (operator→ticker table), unblocks the flagship graph build + fusion (b) gate 1 (v1.0.131) [T-DATACORE]

- [T-DATACORE] Territory: datacore/entity_map.json (new),
  scripts/build_entity_map.py (new), server/entityMap.test.ts (new) —
  per WORKSTREAM PARTITION. SHARED files touched minimally, last:
  package.json (version bump), research/open_questions.md +
  EVERYTHING_GRAPH.md (status update), this entry.
- SESSION START per MEMORY PROTOCOL: read CLAUDE.md, this file (tail),
  open_questions.md (full), wishlist.md (full). Loop-health ratio over
  the last 10 entries: 3 PRODUCT/PIPELINE-class, 2 RULE-REVIEW, 2
  RESEARCH, 2 REPAIR-class among the DATACORE DEFECT QUEUE closure —
  well under the 7/10 REPAIR thrash threshold. `git`/GitHub state
  checked directly (list_commits on `main` via the GitHub API, not
  just local `git fetch`, after the prior session's own note about a
  stale local ref cache): branch head 99921a6 already equals origin
  main's head — this session starts clean, no reset needed. KNOWN
  BROKEN: no unresolved trading-loop-threatening item found in
  open_questions.md's KNOWN BROKEN section (items 1-2, 5-11 resolved;
  item 3 CSP cascade and item 4 general-health remain ACCESS-LIMITED
  verification gaps unchanged from prior sessions, not new breaks, and
  per the task framing a [PRODUCT] session does not preempt DAILY's
  repair duty for them — noted, not chased this session).
- PRIMARY ACTION SELECTION: surveyed BUILD ORDER 4 (items 1-2 shipped
  today already, item 3 international registries deferred pending a
  fresh per-country session, item 4 blocked until ~2026-09-27, item 5
  blocked until 2026-07-06 close, item 6 counterfactual logger already
  extended this same day in v1.0.130) — every BUILD ORDER 4 item was
  either done or genuinely time/access-blocked. Went to the GIP BUILD
  QUEUE / MAP V2 ROADMAP R5 (THE EVERYTHING GRAPH, explicitly named
  "flagship" by the 2026-07-04 charter directive) and found its own
  design doc (datacore/EVERYTHING_GRAPH.md) names an exact, unbuilt,
  fully-specified next step: build-plan item 1,
  `datacore/entity_map.json`. Confirmed via `ls`/`grep` that no such
  file existed yet and that today's earlier aircraft
  registrant→operator work (v1.0.127, `datacore/aircraft/entity_spine.json`)
  is a DIFFERENT table for a different roadmap item (BUILD ORDER 4 #1,
  aircraft tail→operating-airline resolution) — confusingly similar
  name, distinct scope; did not double-build.
- WHAT WAS BUILT: `scripts/build_entity_map.py` reads the exact
  `operator` strings from `datacore/sites/strategic_sites.json` (13,
  all sites) and the exact `owner` strings from the top-100-by-
  capacity_mw plants in `datacore/powerplants/us_power_plants.json`
  (56) — 69 total distinct registry strings, asserts every one has a
  researched entry (fails loudly if a source registry changes and
  introduces a new unresearched string), and emits
  `datacore/entity_map.json` with `{operator, ticker, confidence,
  parent, note}` per REASONING STANDARD #10 discipline stated in the
  design doc ("confidence: exact-name match (high) / alias match
  (medium) / manual research"). RESEARCH METHOD (REFERENCE DATA
  ACCURACY rule): every entry is backed by a live WebSearch this
  session against SEC filings, company investor-relations pages, or
  primary-sourced summaries — not recalled from training data alone.
  This mattered: regulated-utility subsidiary structures (PG&E→PCG,
  Duke Energy Carolinas/Progress/Florida/Indiana→DUK, Southern Company
  subsidiaries→SO, AEP/Entergy/Dominion/PPL/Ameren/Xcel/NiSource/
  FirstEnergy/CMS/DTE/Pinnacle West/IDACORP-class operating companies,
  Evergy Kansas Central→EVRG, AES Indiana→AES) are decades-stable and
  matched prior knowledge, but merchant/IPP generators churn fast and
  TWO SEARCHES CAUGHT STALE ASSUMPTIONS BEFORE THEY SHIPPED: (1)
  `NRG Homer City Services LLC` — the plant retired June 2023 and was
  physically demolished in 2025 (now a gas-fired/data-center
  redevelopment); NRG was the contracted OPERATOR, never the OWNER
  (ownership sat with a hedge-fund vehicle from the plant's bankruptcy
  era) — mapping this to NRG would have been a confidently-wrong
  ticker on a plant that no longer exists. (2) `Louisiana Generating
  LLC` — sold by NRG to Cleco in Feb 2019; Cleco itself has been
  privately held since a 2016 investor buyout, so there is NO current
  public-ticker mapping at all, not even Cleco. Also resolved
  correctly via search rather than guesswork: the Feb-2022 Exelon→
  Constellation spinoff (`Exelon Nuclear`/`Constellation Mystic Power
  LLC`→CEG, not EXC), the 2018 Dynegy→Vistra merger (`Dynegy Midwest
  Generation Inc`→VST), Talen Energy's 2022 bankruptcy-emergence
  relisting (NASDAQ: TLN), and the fragmented, no-single-public-parent
  ownership of South Texas Project (`STP Nuclear Operating Co` —
  Constellation 42% / CPS Energy 42% municipal / Austin Energy 16%
  municipal as of a 2024 transaction) and Keystone/Conemaugh (`KeyCon
  Operating LLC` — multiple PE holders + a ~12-16% Talen minority
  stake). RESULT: 44/69 mapped (34 "high" confidence wholly-owned
  regulated-utility or clean-merger cases, 1 "medium" —
  `PacifiCorp`→BRK.B, two levels removed via Berkshire Hathaway
  Energy, confidence deliberately held down), 25/69 honest unmapped
  gaps — federal agencies (TVA, USACE, Bureau of Reclamation), state/
  municipal port and power authorities (8 of the 13 site operators),
  and privately-held or fragmented-JV merchant generators (GenOn,
  Gavin Power, Helix Ravenswood, LaFrontera, Midland Cogeneration
  Venture, Cardinal Operating, KeyCon, plus the two stale entries
  above). Zero guessed tickers anywhere — every unmapped entry carries
  a one-line reason.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): this table -> unblocks
  EVERYTHING_GRAPH.md build-plan step 2 (`server/entityGraph.ts`, the
  `operates` edge type company→facility) for whichever session claims
  it next -> ALSO unblocks the independently-filed fusion hypothesis
  (b) "Generation shifts × utility tickers" (open_questions.md FUSION
  HYPOTHESES section), whose own gate-1 ground truth explicitly
  required "a registry-owner→ticker mapping table" — this is that
  table, built once and shared, per the design doc's stated intent
  ("removes the join labor, grants no evidential shortcut" — gate 1
  for that hypothesis still requires the separate EIA-930-vs-registry-
  capacity reconciliation, not done here). No trading behavior changed
  (RAW reference data only, no SIGNAL claim, nothing wired into
  deep_score or any live route yet — step 2 is what serves it).
- REGRESSION TESTS: `server/entityMap.test.ts` (new, 5 cases) — every
  entity has required fields + valid confidence tier + no duplicate
  keys; unmapped entities never carry a ticker; the `coverage` block
  in the JSON matches the actual entity list (catches stale
  bookkeeping); COVERAGE HONESTY — every operator string currently in
  `strategic_sites.json` and every top-100-plant owner in
  `us_power_plants.json` has an entry (fails if either source registry
  grows a new unresearched operator, forcing the next session to
  research it rather than silently under-covering); the doc text
  carries the "re-verify" honesty warning and is marked "no predictive
  claim" (RAW not SIGNAL); a fifth test spot-checks that named federal/
  municipal operators are specifically `unmapped`, not silently
  dropped.
- PROMOTION RULES: (1) `npm run test:node` — 214/217 passed; the 3
  failures (`compression.test.ts`, `gdeltEvents.test.ts`,
  `owmTiles.test.ts`) are PRE-EXISTING and unrelated, confirmed via
  `git stash` A/B on the exact same command (209/212 passed on the
  pre-PR commit, identical 3 failures) — this session's 5 new tests
  all pass, no regression. `python3 -m pytest -q` — this sandbox
  started with NO python deps installed at all (`voltrade_daemon.py`
  hard `sys.exit(2)`s if numpy/pandas/requests import fails, which
  crashed pytest's collector entirely); installed `requirements.txt`
  to get a real signal rather than skip the gate, then confirmed via
  the same `git stash` A/B that the result (392 passed, 2 skipped, 2
  pre-existing failures in
  `test_options_v134_fixes.py::TestFix7_EarningsAlwaysIronCondor` —
  the same `KeyError: 'opt_type'` in `options_scanner.py:490` a prior
  session already logged) is identical before and after this PR — this
  PR touches zero files that pytest's suite exercises. (2) new tests
  ship with the new file (rule 2). (3) not a strategy/parameter change
  — no backtest required (this is RAW reference data, not a trading
  rule). (4) version bumped 1.0.130 -> 1.0.131, read-and-increment,
  confirmed against the GitHub API's live `main` HEAD (99921a6,
  identical to this branch's parent) immediately before bumping — the
  prior session's OPS GOTCHA about a stale local `git fetch` cache
  reproduced again this session (a first `git log origin/main` showed
  main 50 commits "behind" after a plain `git fetch`; the GitHub API's
  `list_commits(sha=main)` immediately showed main's HEAD already
  matched local HEAD, i.e. no actual divergence) — worth a permanent
  note: prefer the GitHub API or `git ls-remote` over a bare
  `git fetch && git log origin/main` when this repo's mirror seems
  stale, since the cache artifact has now recurred twice. (5) one
  logical change (one new registry file + its builder + its test);
  research/doc updates and the version bump are the SHARED-file tail
  of the same PR, not a second logical change. (6) VISUAL VERIFICATION
  not applicable — no client/ files touched.
- LIVE HEALTH CHECK (task-mandated, done before the primary action):
  `GET https://voltradeai-production.up.railway.app/api/health`
  returned `status: ok` across the board — server, database, Alpaca
  ACTIVE, python bridge, bot `active`, equityPeak $108,151.39,
  drawdownPct 0.0, `liveness.dark: false`. No LIVENESS ALARM
  condition, nothing to surface top-of-report; unchanged from the
  prior session's reading a few hours earlier.
- NOT IN SCOPE, FLAGGED HONESTLY: step 2 (`server/entityGraph.ts` +
  `/api/data/graph` + tests) and step 3 (the `/data` graph panel) are
  the natural next PRs for whichever session (interactive or routine)
  claims R5 next — this entry's build-plan update in
  EVERYTHING_GRAPH.md and open_questions.md is the claim-before-
  building marker for step 1 only, step 2/3 remain open and unclaimed.

## 2026-07-05 — [RULE-REVIEW] Counterfactual logger: correlation/spread rejections now labeled truthfully in the shadow archive (BUILD ORDER 4 #6 continuation) (v1.0.130) [T-BOT]

- [T-BOT] Territory: bot_engine.py, shadow_portfolio.py (a data module,
  MUTABLE per CLAUDE.md; not one of the T-DATACORE-listed modules but
  squarely inside T-BOT's `server/bot.ts`-adjacent trading-loop scope
  since it's called exclusively from `bot_engine.py`'s scan path), new
  test_shadow_portfolio.py (WORKSTREAM PARTITION). SESSION START per
  MEMORY PROTOCOL: read CLAUDE.md, experiments.md, open_questions.md,
  wishlist.md. Loop-health ratio over the last 10 entries: PRODUCT 3,
  RULE-REVIEW 2, PIPELINE 1, RESEARCH 2, REPAIR 2 — no thrash (2/10
  REPAIR, well under the 7/10 trigger). /api/health checked first: all
  ok, bot active, equityPeak $108,151.39, drawdownPct 0.0, liveness not
  dark — no KNOWN BROKEN item blocked this session, nothing to surface
  top-of-report. `git fetch origin main` confirmed the branch starts
  clean at v1.0.129 (a stale local ref cache briefly showed origin/main
  20+ commits behind; a fresh fetch resolved it — no actual divergence,
  just a cache artifact worth noting for the next session).
- PRIMARY ACTION: BUILD ORDER 4 #6's own build plan (filed this same
  day, see the `be5125e` entry below) named the next concrete, unblocked
  step — "add a block_reasons[] tag ... at the rejection sites in
  bot_engine.py scan/deep_score ... LOGGING ONLY — no mechanism change."
  This was the highest-value fall-through item: BUILD ORDER 4 #4/#5
  wait on future calendar dates (~2026-09-27, 2026-07-06), #3
  (international registries) was already probed-and-deferred today,
  #2 (UI scalability) already shipped today (v1.0.129) — #6's gap was
  the only unblocked, fully-specified queued item left in the roadmap.
  SESSION-BUDGET TIER: "fix a bug seen in audit logs" was not directly
  actionable (no owner-auth audit-log access, per KNOWN BROKEN #4's
  ACCESS LIMITATION, unchanged), so this queued item — fall-through
  tier 1 — was the correct next action, not a downgrade to research.
- HYPOTHESIS STATED BEFORE MEASURING (REASONING STANDARD #10): read
  `bot_engine.py`'s `_scan_market_inner()` (the closure `scan_market()`
  actually delegates candidate-filtering to — NOT `scan_market()`
  itself, which is a thin timeout wrapper; caught this by running the
  new tests against the wrong function first, see BUG FOUND below) top
  to bottom this session. `deep_score()` calls `log_candidate()` with
  decision `"taken"` the moment `combined_score >= MIN_SCORE` — but
  `_scan_market_inner()`'s per-candidate loop applies MORE filters
  AFTER that: cooldown, regime block, correlation/sector
  (`check_sector_correlation`), $50M dollar-volume floor, blocked-
  ticker list, extreme-mover (>50% today) watchlist diversion, and a
  live bid/ask spread check — all of which can still `continue` (skip)
  a candidate `deep_score()` already logged as `"taken"`. PRIOR: ~70%
  chance at least one of these downstream filters was silently
  mislabeling real rejections as "taken" in the shadow archive, since
  nothing in the codebase or `open_questions.md`'s prior audits (KNOWN
  BROKEN #10, the 2026-07-04 `shadow_portfolio.py` audit) had traced
  candidates past `deep_score()`'s own logging call.
  CONFIRMED: correlation/sector blocks and the quote-time spread check
  are exactly this bug — real, frequent rejection paths whose
  candidates were being recorded as "taken" in the learning archive
  before this fix. This is a live HONESTY METRIC risk (GOAL doc): any
  future session running `get_shadow_stats()`'s `win_rate_by_decision`
  would have attributed correlation/spread REJECTIONS' outcomes to the
  "taken" bucket, corrupting exactly the MIN_SCORE RULE COST AUDIT
  question it's meant to answer (mixing "we actually traded this" with
  "we scored it but a downstream filter blocked it" is a different
  population with a different expected win rate).
- BUILT (one logical change, logging-only, mechanisms untouched):
  (1) `shadow_portfolio.update_last_decision(ticker, decision,
  decision_reason, max_age_seconds=120.0)` — new function, mirrors
  `log_candidate()`'s non-blocking contract (any failure swallowed,
  logging must never break the trading loop). Walks the shadow log in
  reverse, finds the MOST RECENT record for the ticker, and corrects
  its `decision`/`decision_reason` ONLY IF that record is still
  `"taken"` (not already resolved by something else) AND still fresh
  (<=120s old, so a stale same-ticker record from an earlier scan can
  never be mislabeled). (2) Two call sites added in
  `_scan_market_inner()`: the `check_sector_correlation()` rejection
  branch now calls `update_last_decision(ticker, "rejected_heat", ...)`
  before its `continue`; the `_spread_pct > 0.005` rejection branch now
  calls `update_last_decision(ticker, "rejected_other", ...)` before
  its `continue`. Both wrapped in their own `try/except Exception:
  pass` so a shadow-logging failure can never affect which candidates
  actually get skipped — the filters' actual trading behavior is
  byte-for-byte unchanged.
- NOT WIRED, deliberately, and why (per the build plan's own scope):
  `rejected_halt` — `check_kill_switches()` (risk_kill_switch.py, a
  FROZEN PATH for mechanisms) gates the separate `TieredStrategy`
  action list (`tiered_actions`), a DIFFERENT code path from the
  `deep_score()`-based `trades` loop this fix targets; wiring it would
  mean tracing kill-switch state into a structurally separate strategy
  engine, a bigger and riskier change than this session's scope.
  `rejected_earnings` — grepped the entire `trades`-loop path: no
  per-candidate stock-long earnings blackout exists today (earnings
  only enters `deep_score()` as a soft ML feature, and separately gates
  covered-call selection in `options_execution.py`'s
  `_check_earnings_guard`, a different candidate population). There is
  no REJECTION SITE to log yet — adding one would mean adding a new
  hard gate, which is a genuine RULE REVIEW-gated behavior change
  (evidence or ablation required), not a logging-only PR. Both left
  open in `open_questions.md`'s BUILD ORDER 4 #6 entry as the next
  slice, correctly scoped separately per PROMOTION RULES rule 5 (one
  logical change per PR).
- BUG FOUND DURING BUILD (the tests caught it, not review): the first
  version of the source-inspection test called
  `inspect.getsource(bot_engine.scan_market)` and got a suspiciously
  short 2,651-char result with neither call site in it — `scan_market()`
  is a thin `def` that sets up a timeout signal handler and delegates
  the real loop to a nested closure, `_scan_market_inner()` (confirmed
  via `grep -n "^def " bot_engine.py` around the relevant line range).
  Fixed by inspecting `_scan_market_inner` instead; both assertions then
  passed. Exactly the READ-BEFORE-WRITE risk CLAUDE.md warns about —
  caught here by a failing test, not by assumption.
- REGRESSION TESTS (new file, `test_shadow_portfolio.py`, written
  BEFORE considering the change complete, per loop-health rule 3):
  6 unit tests on `update_last_decision()` directly (updates a fresh
  "taken" record; no-ops when the most recent record is already
  resolved; no-ops when the record is stale even if max_age_seconds is
  explicit; no-ops on a ticker with no matching record; only ever
  touches the FRESHEST record for a ticker, proven with two seeded
  records for the same ticker at different ages; empty-ticker guard) +
  2 source-inspection tests pinning both `_scan_market_inner()` call
  sites to their exact decision-bucket string and to a defensive
  `except Exception` wrapper (mirrors `test_voltrade_daemon.py`'s
  established pattern of pinning wiring via source inspection). All 8
  pass. Full offline CI-gate subset re-run: `python3 -m unittest
  test_risk_controls test_audit_critical test_diagnostic_false_positives
  test_patches_verification test_voltrade_daemon test_shadow_portfolio`
  — 133 passed, 1 skipped (pre-existing skip, unrelated; identical
  baseline otherwise). `python3 -m py_compile bot_engine.py
  shadow_portfolio.py test_shadow_portfolio.py` clean. NOTE: this
  sandbox lacked numpy/pandas/lightgbm/scikit-learn/requests before
  `pip3 install`, which is why `bot_engine`-importing tests initially
  errored on `ModuleNotFoundError` unrelated to this change — resolved
  by installing the same packages `requirements.txt` already declares;
  not a repo issue.
- Downstream chain (REASONING STANDARD #1): a mislabeled shadow record
  -> `win_rate_by_decision["taken"]` silently pools real trades with
  candidates a downstream filter actually blocked -> the MIN_SCORE RULE
  COST AUDIT question (and any future one relying on the "taken" bucket
  meaning "we actually traded this") reads a biased number without any
  visible error -> fixing the label means `rejected_heat`/
  `rejected_other` become their OWN buckets in `get_shadow_stats()`
  automatically (it groups generically by whatever `decision` string
  appears, confirmed by reading `get_shadow_stats()`'s implementation —
  no code change needed there) -> once >=90d of shadow history
  accumulates (~2026-10-02, unchanged from the prior estimate — this
  PR doesn't touch WHEN backfill runs), the correlation-block and
  spread-filter RULE COST AUDIT questions in `open_questions.md` become
  answerable for the first time. Zero live-trading-behavior change:
  no filter's pass/skip decision changed, only which population its
  rejected candidates get correctly bucketed into for later analysis.
  Version bumped 1.0.129 -> 1.0.130 (read-and-increment, package.json
  is a SHARED file per WORKSTREAM PARTITION — this was the last, small,
  isolated edit in the PR). PROMOTION RULES rule 3 (backtest
  requirement) doesn't apply — no strategy/parameter/threshold changed,
  only which decision string an already-existing rejection logs under.
- MARKET-HOURS NOTE (this run occurs during market hours, per session
  instructions): PR prepared; recommend merge waits until after 4:00 PM
  ET today unless a critical live break is found (none was — this is a
  logging-only correction to the shadow/learning archive, not a
  trading-path fix, so there is no urgency argument for an immediate
  merge).
- STARVED: no — this session's scope (the queued B4-6 item, in full for
  its logging-only-appropriate slice) shipped completely. High-value
  work remains queued for future sessions: `rejected_halt`/
  `rejected_earnings` wiring (this entry, above), KNOWN BROKEN #3
  (CSP execution cascade — still needs owner-auth audit-log
  verification), KNOWN BROKEN #10 (SCORE_BAND_MAX/MAX_CHANGE_PCT
  evidence-or-retire decision, still waiting on shadow history), BUILD
  ORDER 4 #3/#4/#5 (all correctly deferred to their own trigger dates),
  the Python-side staleness-audit sweep the register above still marks
  UNSCANNED, and the CONSTITUTIONAL REPAIR proposals awaiting human
  review in wishlist.md.

## 2026-07-05 — [PRODUCT] UI SCALABILITY ARCHITECTURE — registry-native group/costTier + panel row-cap + 50/100/200-layer synthetic harness (BUILD ORDER 4 #2, GIP Part 4) (v1.0.129) [T-CLIENT, touching datacore/layers.json additively]

- [T-CLIENT] Territory: client/src/**, index.css, scripts/visual_check.mjs
  (WORKSTREAM PARTITION); this PR also touches datacore/layers.json but
  ADDITIVELY only (two new optional metadata fields per layer, no pipeline
  logic changed) — declared here per the partition's cross-territory rule
  (one session, one logical change, not split). SESSION START per MEMORY
  PROTOCOL: read CLAUDE.md, experiments.md, open_questions.md, wishlist.md.
  Loop-health ratio over the last 10 entries: RULE-REVIEW 2, PIPELINE 2,
  RESEARCH 2, REPAIR 2, PRODUCT 2 — no thrash (well under 7/10). /api/health
  checked first: all ok, bot active, equityPeak $108,151.39, drawdownPct
  0.0, liveness not dark — no KNOWN BROKEN item blocked this session, none
  required noting at top-of-report.
- PRIMARY ACTION: BUILD ORDER 4 #2 (self-proposed 2026-07-05, GIP Part 4
  UI SCALABILITY, "IN-PROGRESS" since 2026-07-04) was the next unblocked
  queued product item — #1 (operator resolution) shipped v1.0.127, #3
  (international registries) is blocked on per-country access-page
  discovery, #4 (natgas gate-2) and #5 (options-chain QA) both wait on
  future calendar dates, #6 (counterfactual logger) is T-BOT. This is a
  T-CLIENT item and the only one actually actionable today.
- HYPOTHESIS STATED BEFORE MEASURING (REASONING STANDARD #10): the panel's
  default-open groups (base, live) render ALL members unconditionally and
  PANEL_GROUPS/LAYER_GROUP are hardcoded per-id maps in datamap.tsx — as
  the registry grows toward "hundreds of layers" (GIP Part 4), (a) a large
  default-open group could dump unbounded DOM, and (b) a new layer added
  only to the registry (no client code change) would need a matching
  hardcoded LAYER_GROUP/groupCollapsed entry or it silently mis-groups or
  defaults OPEN. Prior: ~40% chance today's small registry (21 layers, max
  group size 6) already masks a real scaling defect that only shows at
  50-200 layers — worth measuring rather than assuming either way.
- BUILT (one logical change): (1) registry-native `group` + `costTier`
  fields added to every datacore/layers.json entry (schema documented in
  `_doc`) — `groupOf()` in datamap.tsx now prefers `l.group`, falling back
  to the old LAYER_GROUP map only for the visual-harness fixture / a
  registry response from an older deploy mid-rollout; (2) `groupCollapsed`
  init switched from a hardcoded collapsed-name list to a computed
  `!OPEN_GROUPS_BY_DEFAULT.has(id)` (OPEN_GROUPS_BY_DEFAULT = {base, live})
  — IDENTICAL result for today's 6 groups (verified: zero visual diff) but
  any group id introduced later defaults COLLAPSED automatically instead
  of needing a second hardcoded entry remembered; (3) GROUP_ROW_CAP = 12:
  an open group renders at most 12 rows behind a "+N more — show all"
  control — no-op today (max group size 6) but bounds DOM per group at any
  registry size; (4) a `costWeightOf`-summed active-cost-budget badge
  ("moderate load"/"heavy load", silent below weight 15) in the panel
  header — a genuine consumer of costTier, not decorative metadata (the
  STALENESS AUDIT would rightly flag an unused field); (5) an unknown-group
  catch-all ("_more") so a layer whose `group` isn't in PANEL_GROUPS still
  renders instead of silently vanishing from the panel.
- BUG FOUND AND FIXED DURING BUILD (the harness caught it, not review): the
  first version of the "_more" catch-all still showed only ~75% of
  synthetic layers reachable via "show all" — traced to exactly the defect
  the catch-all exists to prevent (layers whose `group` didn't match any
  PANEL_GROUPS id were filtered out of the render entirely, only labeled
  correctly). Fixed by extracting `renderPanelGroup()` and calling it for
  both the named PANEL_GROUPS and the orphan set; re-ran the harness to
  confirm 50/100/200 all reach 100%.
- SEPARATE FINDING, NOT A BUG (verified by direct debugging, see below):
  synthetic layers correctly render their toggle DISABLED ("unwired" guard,
  pre-existing 2026-07-04 open-tab-skew protection) because no real
  map-data fetch/render effect exists for a fabricated `synth_N` id — a
  registry edit alone (adding `group`/`costTier`) was never going to make
  a brand-new interactive layer functional without an actual client
  deploy, and the guard correctly refuses a toggle that would flip and
  paint nothing. Confirms the guard and my registry-native change are
  solving two different problems (grouping/visibility vs. real wiring) —
  worth recording so a future session doesn't re-litigate it.
- MEASURED (scripts/visual_check.mjs, new `--page scale` battery,
  synthetic 50/100/200-layer registries via a per-context Playwright route
  override — the shared FIXTURES/server untouched, no determinism cost to
  any other page): default-open panel rows stayed at 14/24/24 (well under
  the 30-row regression-guard budget) across n=50/100/200 — collapse-by-
  default + GROUP_ROW_CAP hold regardless of registry size, not just
  today's. "Show all" reached 50/50, 100/100, 200/200 layers (100% self-
  see at scale, after the fix above). TTI stayed 1.2-2.5s, under the
  existing 3000ms map-page gate this file already uses elsewhere — no
  regression at scale. The cost-budget badge was separately exercised on
  the REAL 21-layer fixture (toggling all 9 non-default heavy/moderate
  layers, weight 13->35) and correctly read "heavy load" — `.visual/
  results.json`'s `data`/1440 entry: `costBudgetBadge: "heavy load"`,
  `toggleConsistency: "16 layers toggled clean"`.
- CONCLUSION vs PRIOR: hypothesis partially confirmed — the architecture
  DID need the fix (the orphan-group defect was real, not hypothetical),
  but once fixed, today's collapse-by-default + 12-row cap combination
  already holds at 200 layers with real margin (24 rows vs 30 budget) —
  literal windowed DOM virtualization is NOT yet evidence-justified;
  filed as a precise trigger condition in open_questions.md BUILD ORDER 4
  #2 rather than built speculatively (CLAUDE.md: don't design for
  hypothetical requirements) — revisit if any single group's real member
  count approaches ~25 (row cap 12 + one showAll click still renders all
  25, which is the actual measured-safe ceiling per the n=50 case above:
  the largest synthetic group there held ~7 members and passed cleanly;
  extrapolating the n=200 case, up to ~25-member groups measured clean).
- DOWNSTREAM CHAIN (REASONING STANDARD #1): registry-native group/costTier
  -> a future pipeline session can add a datacore layer with correct panel
  placement and a cost estimate by editing layers.json alone (no
  datamap.tsx PR required for KNOWN groups) -> the human's weekly /data
  review keeps working as the registry grows -> BUT the layer stays
  non-interactive ("unwired"/"reload to enable") until a follow-up client
  PR adds its real fetch/render effect, by design — this PR does not
  change that constraint, only removes the panel-placement bottleneck.
- PROMOTION RULES: (1) full test:node suite 223/223 passed (unchanged
  count from before this PR — no test removed/weakened); (2) new tests are
  the scale-harness battery itself (mechanical, in scripts/visual_check.mjs,
  since this repo's client layer has no unit-test framework — DESIGN.md's
  visual harness is the established verification path for client/) plus
  the real-fixture cost-budget exercise; (3) not a strategy/parameter
  change — no backtest required; (4) version bumped 1.0.128 -> 1.0.129
  (read-and-increment at commit time per MERGE-ORDER PROTOCOL, confirmed
  against origin/main immediately before bumping — no new merges since
  this branch was cut); (5) one logical change, own PR; (6) VISUAL
  VERIFICATION: `npm run visual` (soft mode) run at 390/768/1440 for all
  three pages (data/developers/landing) plus the new scale/all-off
  batteries — 0 hard failures; screenshots reviewed
  (.visual/data-1440.png shows the unchanged default panel — no cost
  badge, matches the "silent below weight 15" design; .visual/
  data-scale-200.png shows the 200-layer synthetic registry rendering
  correctly with the "reload to enable" unwired state and a scrollable
  panel). Pre-existing warnings (nav touch-target sizes, "Filings & flows"
  clipped-control note) verified UNCHANGED from the pre-PR baseline via
  `git stash` A/B (not a regression — filed nowhere new, already
  pre-existing per the file's own history).
- NOT IN SCOPE, FLAGGED HONESTLY: `python3 -m pytest -q` was run as a
  sanity check (no Python files touched by this PR) and found 2
  PRE-EXISTING failures unrelated to this change — `test_options_v134_fixes
  .py::TestFix7_EarningsAlwaysIronCondor` (both cases), `KeyError:
  'opt_type'` in `options_scanner.py:490`'s `_find_by_delta` — confirmed
  pre-existing via `git stash` A/B (identical 2 failures on the pre-PR
  commit). T-BOT territory, out of scope for this T-CLIENT PR per
  one-logical-change-per-PR; noting here per the REPAIR MANDATE so a
  T-BOT session doesn't have to rediscover it. `npm run check` (tsc): 63
  pre-existing errors, unchanged count, none touch datamap.tsx (verified
  by grep) — consistent with the prior session's note that tsc is not a
  clean gate in this repo.
- BUILD ORDER 4 #2 STATUS: updated in open_questions.md with the measured
  numbers above — item stays open only for the "revisit if a group
  approaches ~25 members" trigger; not closed as "done forever" since
  that's a real future condition, not a today-problem.

## 2026-07-05 — [RULE-REVIEW] Performance/ml-status/diag slippage stats were reading a dead file — realistic-P&L honesty bug fixed (v1.0.128) [T-BOT]

- [T-BOT] Territory: bot_engine.py/ml_model_v2.py/server/bot.ts outside
  frozen paths, per WORKSTREAM PARTITION. SESSION START per MEMORY
  PROTOCOL: read CLAUDE.md, this file, open_questions.md, wishlist.md.
  Loop-health ratio over the last 10 entries at session start: 2
  [REPAIR], 3 [RESEARCH], 2 [PRODUCT], 3 [PIPELINE], 0 [RULE-REVIEW] —
  well under the 7/10 [REPAIR] thrash threshold, no meta-problem to
  address. DATACORE DEFECT QUEUE (all 10 items + #237/#238) confirmed
  fully closed by prior sessions. /api/health: all checks ok, bot
  active, equityPeak $108,151.39, drawdownPct 0.0, liveness not dark.
- PRIMARY ACTION (SESSION BUDGET tier 1: "fix a bug seen in audit logs"):
  used the human-approved /api/diag route (DIAG_TOKEN, wishlist option
  (d)) to probe live state. `/api/diag/ml` returned
  `{feedback_count: 500, fills_count: 0}` — 500 real trade_feedback
  records but zero "fills." Traced (READ BEFORE WRITE): FILLS_PATH
  (storage_config.py, voltrade_fills.json) has had exactly ONE writer
  ever, ml_model.py's legacy track_fill() — and nothing imports
  ml_model.py anymore (grep confirmed zero call sites repo-wide; it's
  fully orphaned). The LIVE track_fill (ml_model_v2.py, the one bot.ts
  actually calls on every order fill) writes entry-fill
  expected_price/fill_price/slippage_pct straight into
  TRADE_FEEDBACK_PATH instead, and has done so since v1.0.34. Result:
  every route reading FILLS_PATH for slippage/fill-count data
  (`/api/bot/performance`'s realistic-P&L calc, `/api/bot/ml-status`,
  the `/api/diag/ml` probe, plus an unused dead read in
  `/api/bot/export-trades`) has ALWAYS seen an empty list, so
  avgSlippagePct/totalSlippageCost/slippageGapPct/totalFills have been
  silently pinned at zero on every deploy regardless of real trading
  activity.
- HONESTY METRIC RELEVANCE: this is exactly the self-deception CLAUDE.md
  warns about — the performance dashboard's "realistic P&L net of
  slippage" has been reporting IDENTICAL to paper P&L this whole time
  (slippageGapPct always 0), even though real per-trade slippage data
  existed all along, just recorded into a different file than the one
  the dashboard reads. REASONING STANDARD #6 (costs/frictions first)
  was being silently violated by the measurement layer itself, not the
  strategy.
- FIX (own PR, MEASUREMENT-INTEGRITY isolation — no trading behavior
  touched): added `ml_model_v2.fills_slippage_stats(feedback)`, a pure
  function deriving count/avg_slippage_pct/total_slippage_cost directly
  from trade_feedback's entry-fill records (exit-fill updates and
  orphan-exit records lack expected_price/slippage_pct, so they're
  excluded by construction — verified by test). All four `server/bot.ts`
  call sites now use it instead of reading FILLS_PATH; the dead
  export-trades read (loaded, never used in the CSV) was deleted
  outright.
- BEFORE vs AFTER on identical inputs (PROMOTION RULE + MEASUREMENT
  INTEGRITY requirement — see test_fills_slippage_stats.py, 7 cases):
  BEFORE — any non-empty feedback list still yielded
  `{avgSlippagePct: 0, totalSlippageCost: 0, totalFills: 0}` because the
  code looked at a file nothing writes to. AFTER — real entry-fill
  records (expected_price + slippage_pct present) are correctly counted
  and aggregated; e.g. two synthetic fills with slippage_pct 0.05% and
  0.20% now correctly average to 0.125% instead of reporting 0%. DIRECTION
  OF BIAS: this makes realisticPnlPct/slippageGapPct MORE conservative
  (reveals cost drag previously hidden at zero), never more favorable —
  not the "make the strategy look better" pattern MEASUREMENT INTEGRITY
  treats as suspect by default; it is a named bug (two-file split, one
  side orphaned) fixed by reading from where the data actually lives.
  Could not diff against the ACTUAL 500 live records (no diag probe
  exposes raw trade_feedback content by design — the whitelist
  deliberately excludes raw trade data) — stated honestly rather than
  overclaiming a live before/after; the synthetic-fixture test is the
  verification artifact.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): fixed slippage aggregation
  -> `/api/bot/performance`'s realisticPnlPct/slippageGapPct/totalFills
  now reflect true historical fill quality -> the human's next dashboard
  view of "how much is slippage actually costing us" becomes
  trustworthy for the first time since ml_model_v2 replaced ml_model ->
  no change to what the bot trades, sizes, or when it exits (isolation
  requirement satisfied).
- TESTS: `test_fills_slippage_stats.py` (7 cases, python3 -m pytest —
  full suite: 386 passed, 2 skipped, unchanged from before this change).
  `server/fillsSlippageWiring.test.ts` pins that bot.ts never reads
  FILLS_PATH again and that all three live routes call
  fills_slippage_stats (npm run test:node — full suite: 223/223 passed).
  `npm run check` (tsc): 63 pre-existing errors before AND after this
  change (verified via git stash) — none touch edited lines; tsc is not
  a clean gate in this repo today, unrelated to this PR.
- NOT IN SCOPE (flagged, not fixed, per one-logical-change-per-PR):
  `storage_config.FILLS_PATH` itself and the now-fully-orphaned
  `ml_model.py` module (zero import sites repo-wide) are dead code —
  exactly the kind of finding the in-progress STALENESS AUDIT (register
  above, Python-side sweep still marked UNSCANNED) should sweep up and
  delete outright in a future session; noting it here so that session
  doesn't have to rediscover it.

## 2026-07-05 — [REPAIR] TestFix7 wall-clock-dependent options tests — deterministic time + real contract shape (test-only)

- FOUND BY: the Census-stream ship gate (full pytest at 15:20 ET
  Sunday) — first full run ever to land inside 9:30-16:00 ET.
  test_options_v134_fixes.py::TestFix7 (2 tests) failed with
  KeyError: 'opt_type'.
- ROOT CAUSE (two stacked defects): (1) _setup_earnings_iv_crush
  early-returns None outside 9:30-16:00 ET (_is_regular_hours checks
  time-of-day ONLY — fires on weekends too), and the tests' "if
  result is not None" guards made them pass VACUOUSLY outside that
  window — every prior green run never exercised the deep path.
  (2) Inside the window, the mock fixtures predated the current
  _fetch_options_chain contract shape (keys "type"/"expiry", no
  "delta") so _find_by_delta crashed. Production is unaffected —
  real chains carry opt_type/delta (options_scanner.py contract
  build); the stale fixture was test-only debt.
- FIX (test file only, no runtime change → no version bump, same
  convention as docs PRs): patch _is_regular_hours (deep path runs
  at any wall-clock time) + _get_spy_vs_ma50 (was an unmocked live
  fetch inside the patched-feature call's argument list); fixtures
  rebuilt via a _contract() helper matching the exact
  _fetch_options_chain dict; vacuous guards upgraded to hard
  assertIsNotNone + mock_chain.called (RATCHET: a silent
  None-regression now fails loudly instead of skipping every
  assertion). High-IV test additionally pins the wide-wings label.
- Verified: 47/47 in the file and 382 passed / 1 skipped suite-wide,
  run INSIDE the reproducing ET window.
- LESSON (recurring class): "if result is not None: assert..." is a
  vacuous-pass pattern — the test passes forever if the code path
  dies. Same family as the wall-clock dependence: both make a test's
  meaning depend on state outside the fixture. Grep for the pattern
  during the next staleness audit.

## 2026-07-05 — [PIPELINE] Census port imports — key-gated stream built (BUILD ORDER 3 #4 unblocked) (v1.0.132)

- TERRITORY: T-DATACORE. Human message "CENSUS_API_KEY added key"
  unblocked BLOCKED-FOR-MIKE #6 (the only build-order item that was
  key-blocked). Built server/censusImports.ts + tests + manifest +
  /api/data/imports.
- KEY-LOCATION FINDING: the key is NOT in this session's container
  env (presence-only check; container env is fixed at session start —
  exact FRED precedent). So instead of a session-run backfill, the
  stream is SERVER-SIDE key-gated on the fredMacro pattern:
  censusEnabled() = Boolean(CENSUS_API_KEY), poller no-ops keyless,
  route returns {enabled:false, reason} honesty, activates
  automatically on the next deploy IF the key was set in Railway.
  Wishlist #6 corrected (it originally said "session env; Railway not
  needed" — the built design inverts that) and marked
  DONE-PENDING-VERIFICATION.
- QUERY-SHAPE HONESTY: the intltrade imports/porths parameter set
  could not be live-verified without the key (keyless probes 302 to
  missing_key). Mitigations built in: (a) two documented query
  variants tried in order (full containerized set, then GEN_VAL_MO
  fallback); (b) HEADER-DRIVEN parsing — column order never assumed;
  (c) Census's readable error bodies logged verbatim (key never
  logged, never archived — test-asserted) so a wrong shape is fixed
  from prod logs, not guesswork. LIVE VERIFICATION PENDING: first
  DAILY session after deploy checks /api/data/imports.
- Archive: append-only JSONL day-files under censusimports/, dedup
  key port|month|values so FT920 revisions append as new vintages;
  seedSeen 40d; gz after 2d; daily poll (monthly source, ~45d lag —
  off-days are dedup no-ops); eager boot poll (KNOWN BROKEN #9).
  Missing values null, never zero. Public domain, attribution
  "U.S. Census Bureau (USA Trade Online / FT920)".
- HYPOTHESIS (stays gate-locked; RAW display + archive only): port
  import value/containerized-weight deltas lead retail inventory
  cycles; joins with port-dwell analytics for a two-sided port view
  (demand value × supply friction). Ladder work begins only after
  live data lands and gate 1 (readings vs a second official source)
  is designed.
- Tests: censusImports battery 4/4 (header-driven + shuffled-column
  identity, ''→null, key gating, variant fallback with
  key-never-in-records assertion, dedup/vintage/gz lifecycle);
  manifest envelope battery 3/3 with censusimports.json; tsc at the
  63-error baseline.

## 2026-07-05 — [RULE-REVIEW] Counterfactual-logger check-in + natgas gate-2 design pre-stated (BUILD ORDER 4 #4+#6) (docs)

- [RULE-REVIEW] B4-6 VERDICT: the CLAUDE.md counterfactual mandate is
  SUBSTANTIALLY BUILT — shadow_portfolio.log_candidate() records
  every scanned candidate with features + nightly-backfilled
  +5/+10/+20d outcomes, so threshold rules get prevention-P&L
  post-hoc by re-applying predicates to the archive (strictly better
  than block-event logging for those rules). THE GAP: block-reason
  tags for rules whose predicates need non-logged state (correlation
  blocks, halts, quote-time spreads) — build plan filed for a T-BOT
  session (logging-only at rejection sites, mechanisms untouched);
  first readout unchanged at >=90d shadow history (~2026-10-02).
- [RESEARCH] B4-4: natgas-storage x degree-day gate-2 design
  PRE-STATED before any overlapping live week exists (expanding-
  window degree-day-implied draw; residual sign vs UNG Thu->Wed
  returns vs base rate, regime-split; PASS = >=60% sign hit on
  n>=12 out-of-sample weeks + positive excess; prior ~30% with the
  tank-fill posterior stated). Runs ~2026-09-27.
- Territory note: both filed as docs because the remaining B4 build
  items are T-BOT (#6 gap) and T-CLIENT (#2 UI scalability) — this
  session declared T-DATACORE; the partition holds and nothing
  idles (SESSION BUDGET rule 3).

## 2026-07-05 — [PIPELINE] Operator resolution — gate PASSED 98.4% group-aware, n=2,498 (BUILD ORDER 4 #1) (v1.0.127)

- [T-DATACORE] server/operatorResolution.ts: callsign-prefix
  inference (>=2 prefixed observations, >=60% majority, KNOWN-prefix
  table only — 23 ICAO codes data-driven from our archive survey;
  unknown prefixes stay null), trustee-registrant detection (pattern
  from our own spine: Bank of Utah 490 / UMB 328 / Wilmington 301 /
  TVPX 97 airframes), PARENT_GROUP mapping (Envoy/PSA/Piedmont ->
  American; Endeavor -> Delta; SkyWest/Republic deliberately NOT
  mapped — independent companies under capacity agreements). Fleet
  series now aggregates by resolved OPERATOR (works even for non-US
  hexes the spine can't match), falls back to registrant, labels the
  basis per airframe, counts trustee shadows, and carries the listed
  parent group for ticker-level studies.
- GATE (pre-stated >=90% on >=20 airframes; ran on n=2,498 — 125x
  the minimum): raw registrant-agreement 86.1% -> the mismatch class
  was ENTIRELY parent-registered regional jets flying wholly-owned
  subsidiary callsigns, i.e. the resolution being MORE precise than
  the registrant check (the check was circular for exactly the class
  the module exists for). Group-aware agreement: 98.4% -> PASS.
  Residuals verified correct-by-design (SkyWest/Republic capacity
  flying + one JET BLUE/JETBLUE string artifact). Both numbers
  reported; the check mis-specification is the lesson, not hidden.
- DESIGN LESSON (caught by test fixture): occurrence counting beats
  distinct-callsign dedup — one callsign seen twice is two
  observations of the same operator.
- Unlocks fleet-utilization GATE 2 (utilization x earnings at the
  GROUP level) once archive depth suffices.

## 2026-07-05 — [RESEARCH] BUILD ORDER 4 filed — deepen before new roots (docs)

- [T-DATACORE] Theme: the gate-2 unlocks and queued GIP items that
  tonight's builds created, before any new ingestion. Order (full
  hypotheses + gates in open_questions.md): (1) registrant->operator
  resolution — the B3-1 blocker; callsign-prefix inference from our
  own archive is the free key insight (a hex flying UAL#### IS
  United ops regardless of trustee registrant); (2) UI scalability
  architecture (GIP Part 4, [T-CLIENT]); (3) international aircraft
  registries v1 (the 22% of archived hexes FAA can't match);
  (4) natgas-storage x degree-day gate-2 design, criteria pre-stated
  now, runs at >=12 live-overlap weeks (~2026-09-27); (5) options-
  chain first-week QA (first snapshots 2026-07-06 close);
  (6) [RULE-REVIEW] counterfactual-logger check-in per the CLAUDE.md
  mandate.
- One item per PR; read-before-write rigor unchanged for later
  actions per SESSION BUDGET.

## 2026-07-05 — [RESEARCH] Anomaly-mining pass run — BUILD ORDER 3 COMPLETE (docs)

- [T-DATACORE] B3-6 executed per the angle-hunting mandate; full
  findings + pre-registered mining design in open_questions.md
  (BUILD ORDER 3 item 6 RUN annotation). Summary: (a) the pass found
  and fixed the event-loop scan class (#237/#238) — platform-eating-
  itself anomalies outrank market anomalies; (b) honest verdict that
  2-day-old position archives cannot host market mining (the 32
  dwell "anomalies" are archive-birth artifacts); (c) airline
  utilization baselines established under our coverage; (d) mining
  design PRE-REGISTERED with fixed statistics (z>=3 vs 4-week
  baselines, multiple-hypothesis discounting, out-of-sample-only
  belief) and re-run triggers at 30d/60d archive depth (~2026-08-03).
- BUILD ORDER 3 SCOREBOARD: 6/6 resolved — fleet utilization (gate-1
  PASS 20/20), EIA weekly (44yr backfill), CPC degree days (10.5yr),
  CBP imports (routed to BLOCKED-FOR-MIKE: Census key now required),
  Everything Graph R1 (live end-to-end on the site cards), anomaly
  pass (this entry). Two build orders proposed and executed to
  completion in one session; next session proposes BUILD ORDER 4 or
  takes the GIP queue.

## 2026-07-05 — [REPAIR] portdwell: the SAME event-loop defect, 4th site, heavier window (v1.0.126)

- [T-DATACORE] Minutes after #237's prod verify, BOTH analytics
  endpoints timed out again — portdwell's synchronous 168h scan on
  cache-miss was blocking the loop (shadowstats was collateral: its
  new poller answers instantly, but only when the loop is free).
  This confirms the RECURRENCE reading of the class: fixing
  shadowstats alone could never hold while any sibling still
  scanned synchronously.
- FIX: computePortDwellAsync over readVesselTracksAsync; BOTH
  surfaces (/api/data/portdwell + /api/v1/stats/portdwell) serve an
  eager 10-min poller cache (warming_up / 503+Retry-After during
  the first scan; metered honestly on the v1 side). Ratchet appended
  to the portDwell battery: async deepEqual sync on a fixture
  archive (8/8).
- CLASS CLOSED: grep confirms no remaining route calls a synchronous
  archive scan — every archive-derived surface (shadowstats,
  portdwell, fleet utilization, site timeline, hex enumeration,
  platform stats) is now streaming + eager-cached. The experiments
  pattern note from v1.0.125 stands as the standing rule for new
  surfaces.

## 2026-07-05 — [REPAIR] shadowstats blocked the whole event loop — async streaming + eager poller (v1.0.125)

- [T-DATACORE] FOUND BY the B3-6 anomaly pass surveying prod
  analytics: cold /api/data/shadowstats = 90s timeout (Railway 000)
  then 26s warm. ROOT CAUSE: computeShadowStats ran a SYNCHRONOUS
  72h gz archive scan on the request path (gunzipSync + full-file
  parse) — at current archive size (42k vessels seen this week) the
  scan blocked the ENTIRE Node event loop, starving every other
  route, health checks included. Same defect on the /api/v1 paid
  surface (per-request sync scan behind the API key).
- FIX: readVesselTracksAsync (streaming readline, the proven
  fleetUtilization/aircraftEntities pattern — the loop keeps
  breathing) + computeShadowStatsAsync; both routes now serve a
  10-min EAGER poller cache only (warming_up / 503+Retry-After
  while the first scan runs). v1 payload gains kind/source/zones
  fields (additive superset).
- RATCHET: equivalence test — async streaming reader byte-identical
  to the sync scan on a fixture archive incl. the gz path; full
  stats objects deepEqual.
- PATTERN NOTE (third instance tonight: trains inflight, spine
  disk-read, this): the failing class is always "works at day-0
  archive size, dies as archives grow." Every archive-scanning
  surface now uses the streaming+cached pattern; new ones must
  start there.

## 2026-07-05 — [PRODUCT] Everything Graph R1 PR-B: timeline in the site detail card (v1.0.124)

- [T-CLIENT half, primary territory T-DATACORE per the
  cross-territory rule] The /data site detail card now shows "Past 7
  days within 50 km (own archives)": up to 5 cross-stream events
  (alert ⚠ / fire ▲ / gauge ≈ with severity + date) and the traffic
  line (aircraft + vessel archived points over N days). Fetched
  async on card open from /api/data/site-timeline/:id; any failure
  leaves the section absent — the card never degrades. Site geojson
  features now carry the site id (they did not before — the card had
  no way to reference its site).
- Harness 0 hard failures at 390/768/1440; the section is click- and
  data-gated so page-level visuals are unchanged (screenshots
  reviewed; same precedent as the FAA enrichment in #223).
- This closes BUILD ORDER 3 #5. Remaining: #6 anomaly-mining pass
  ([RESEARCH], terminates in filed entries).

## 2026-07-05 — [PRODUCT] Everything Graph R1 PR-A: site event timeline route (BUILD ORDER 3 #5) (v1.0.123)

- [T-DATACORE] First user-visible cross-stream join, composed
  entirely from archives we already record. server/siteTimeline.ts:
  per strategic site, last-7-day NWS alerts + FIRMS fire detections
  + USGS gauge readings within 50 km (event-stream day-files), plus
  our own aircraft/vessel archived-point density per day. ONE scan
  pass computes all sites together, 6h-cached stale-served. Route
  /api/data/site-timeline/:siteId (events capped 12 newest-first —
  stated; zone-only alerts excluded per their stream's honesty;
  absent days absent, never zero; density fairness note: near-site
  traffic archives at full resolution by design, so day-over-day
  comparisons are fair). Battery 4 tests on writer-shaped fixtures.
- BUILD ORDER 3 #4 (CBP container imports) ROUTED AROUND:
  api.census.gov now requires a key on every request (probed —
  porths redirects to missing_key). Free instant signup =
  BLOCKED-FOR-MIKE #6 with build-first analysis (port-authority
  HTML scraping = 2 ports, fragile, materially worse; recommend the
  key). Nothing blocked — the queue moved on per the directive.
- PR-B next: compact timeline section in the /data site detail card
  + harness (client half, own PR).

## 2026-07-05 — [PIPELINE] NOAA CPC degree days (BUILD ORDER 3 #3) (v1.0.122)

- [T-DATACORE] scripts/cpc_degree_days.py: StatesCONUS
  Heating+Cooling daily files per year (keyless, public domain,
  probed 2026-07-05), pipe-delimited parse with format-change refusal
  (missing Region header or non-YYYYMMDD date columns = hard error,
  never guessed), non-numeric cells null never zero, only 2-char
  codes treated as states. Artifact datacore/cpc/degree_days.json
  (0.96 MB), manifest cpcdegreedays.json, battery 3 tests.
- FIRST BUILD CAPTURED: 96 series (48 CONUS states x H/C), 3,837
  daily points each (2016-01-01 .. 2026-07-03), ZERO fetch failures.
  First DEMAND-side weather series in the archive — pairs with the
  supply-side NWS alerts + OWM fields.
- Selection honesty: census-division and fuel-weighted upstream
  variants deliberately not archived in v1 (stated in script +
  manifest).
- Hypothesis (gate-locked): population-weighted degree-day departures
  lead natgas/power demand and utility earnings surprises; joins the
  EIA natgas storage series (#233) naturally — storage delta vs
  degree-day-implied draw is the obvious first gate-2 study.

## 2026-07-05 — [PIPELINE] EIA weekly petroleum + natgas storage (BUILD ORDER 3 #2) (v1.0.121)

- [T-DATACORE] scripts/eia_weekly.py: five keyless hist_xls series
  (US crude ex-SPR / gasoline / distillate / SPR crude stocks +
  lower-48 natgas working storage; all probed 2026-07-05), parsed
  with the proven xlrd comparator pattern. Series titles are READ
  FROM THE SHEETS and stored (title_as_published) — upstream renames
  surface as git diffs, never silent mislabeling; the titles
  confirmed the key naming live (WCSSTUS1 = SPR, as labeled). Gaps
  skipped, never zero; all-series failure refuses to write. Artifact
  datacore/eia/weekly_series.json (0.23 MB), manifest eiaweekly.json.
- FIRST BUILD CAPTURED: 2,283 weekly crude points back to
  1982-08-20; gasoline 1990-; natgas 2010-; all current through
  2026-06-26. 44 years of the energy-regime axis in one pull.
- Cushing deliberately NOT duplicated — the tank-fill comparator
  owns it (one source of truth per series).
- Hypothesis (gate-locked): storage-vs-seasonal-band deltas condition
  the energy regime; also the standing external-truth source for any
  future inventory root.

## 2026-07-05 — [PIPELINE] Fleet utilization v1 + GATE 1 PASS 20/20 (BUILD ORDER 3 #1) (v1.0.120)

- [T-DATACORE] GATE 1 (spine join accuracy) PASSED under criteria
  pre-stated in open_questions.md BEFORE sampling: 20 stratified
  hexes (10 top-count corporate/llc + 10 seeded-random spine hexes)
  vs INDEPENDENT adsbdb registrations -> 20/20 exact N-number
  matches, 0 unresolvable (criteria floor was 90%). First ladder
  gate to PASS tonight after three honest kills.
- FINDING FOR GATE 2 (logged now, before the study): the top
  corporate hexes are dominated by TRUSTEE/LEASING shells (TVPX x3,
  UMB Bank trustee, leasing LLCs) — FAA registrants hide beneficial
  owners, so the utilization x earnings join will need a
  registrant->operator resolution step (or restrict to
  self-registered operators like the airlines). Payload labels
  owners as REGISTRANTS, never "the company flying it".
- server/fleetUtilization.ts: airborne-point sessionization per hex
  (gap > 45 min = new flight; hours = session span sums, LOWER
  BOUNDS under adaptive thinning; ground points excluded), weekly
  Monday buckets (absent weeks stay absent, never zero), per-owner
  aggregation over spine corporations+LLCs; 6h-cached scan;
  /api/data/fleet-utilization (kind: derived, >=2 airframes).
  Battery 4 tests (week bucketing, sessionization ground truth incl.
  a test-fixture bug where the 'second flight' itself exceeded the
  gap — the code was right, the fixture was wrong; spine join with
  ground/non-corporate exclusions; missing-input grace).
- GATE 2 (not attempted): utilization deltas vs 5/20d returns +
  earnings proximity, base-rate-controlled; needs archive depth +
  the registrant->operator step above.

## 2026-07-05 — [RESEARCH] BUILD ORDER 3 filed — fusion-forward (docs)

- [T-DATACORE] Standing directive: queue emptied (Build Order 2 went
  6/6 same-day), so the next order is self-proposed. Emphasis
  deliberately shifts from new roots to FUSION of the archives now in
  hand — the compounding asset is the accumulation, and five of six
  B2 items were ingestion; the graph earns its keep by joining them.
- Order (hypotheses + ladder paths in open_questions.md): (1)
  corporate-fleet utilization series (aircraft archive x entity
  spine — gate-1 criteria to be pre-stated before scoring), (2) EIA
  weekly petroleum + natgas storage (keyless XLS family proven),
  (3) NOAA CPC degree days (probed keyless 2026-07-05), (4) CBP
  container imports (format at build), (5) Everything Graph R1 site
  event timeline ([PRODUCT], existing archives only), (6) [RESEARCH]
  anomaly-mining pass per the angle-hunting mandate.
- Each item its own PR under the established stream rules.

## 2026-07-05 — [PIPELINE] STB EP724 rail stream — BUILD ORDER 2 COMPLETE (v1.0.119)

- [T-DATACORE] #6, the final Build Order 2 item. scripts/stb_rail.py:
  discovers the newest "EP724 Consolidated Data through" workbook on
  stb.gov (format change = hard exit), parses the wide matrix
  (~484 week columns since 2017-03), and writes the compact keyed
  artifact datacore/rail/ep724_carloads.json (0.83 MB: 255 series x
  484 weeks). SELECTION stated in script + manifest, never silent:
  cat 11 Weekly Carloads By 22 Commodity Categories (176 series —
  the volume spine the hypothesis needs), cat 1 System train speed,
  cat 3 cars-on-line; dwell/grain/Chicago metrics deliberately not
  archived in v1. Non-numeric cells -> null, NEVER zero. The source
  republishes full history weekly, so this is a whole-file rebuild —
  nine years of weekly carloads captured in the first build.
- FORMAT VERIFIED AT BUILD (the build-order flag): wide matrix
  confirmed live; openpyxl gotcha encoded (validates by file
  extension); CPKC-merger key discontinuity documented in the
  manifest confidence model.
- LIVE E2E: BNSF coal 18-22k carloads/week latest month — plausible;
  week axis ends 2026-07-01 (current). Battery 5 tests (selection
  filter, null honesty, short-row padding, changed-axis refusal,
  newest-workbook discovery).
- BUILD ORDER 2 SCOREBOARD: 6/6 resolved same-day — entity spine
  (live in prod), tank-fill v3 (honest gate-1 kill + v3.1
  pre-registered), NWS alerts (live), Treasury auctions (live),
  Drought Monitor (live + label-integrity fix), STB rail (this).
  Next session: propose BUILD ORDER 3 per the standing directive, or
  take the GIP queue's next item.

## 2026-07-05 — [PIPELINE] US Drought Monitor stream (BUILD ORDER 2 #5) (v1.0.118)

- [T-DATACORE] server/droughtMonitor.ts: USDM data services (keyless;
  attribution REQUIRED and carried in every payload: "U.S. Drought
  Monitor (NDMC/USDA/NOAA)"). CONUS + 8 ag/water states — states by
  FIPS (probed: abbreviations return empty). Cumulative D0-D4 + the
  USDM's own published DSCI as the one labeled DERIVED field. 24h
  eager poll over a 70-day window, dedup aoi|map_date (maps final on
  publish), per-AOI failure tolerance. Manifest + RAW route
  /api/data/drought.
- LIVE-CAUGHT INTEGRITY BUG before shipping: the aoi=us endpoint
  returns BOTH "CONUS" and "Total" (incl. AK/HI/PR) rows per week —
  the first parse labeled both CONUS, silently mixing two series.
  Fixed: a row is kept only when its OWN label matches the requested
  AOI, never relabeled; regression test pins it. Verified after fix:
  exactly 9 AOIs per map week (was 11 rows mislabeled into 9 keys).
- LIVE E2E: 99 aoi-weeks; 2026-06-30 map: NE DSCI 251 (D2+ 62%!),
  OK 163, CONUS 157, IL 11 — plausible cross-section. Battery 5
  tests (real-fixture normalization + DSCI math, label honesty,
  malformed-row drops, FIPS table pin, dedup+gz).
- Hypothesis (gate-locked): belt drought DELTAS lead ag commodities
  and food-producer margins by weeks; joins USGS gauges + FIRMS on
  the environmental axis.

## 2026-07-05 — [PIPELINE] Treasury auction results stream (BUILD ORDER 2 #4) (v1.0.117)

- [T-DATACORE] server/treasuryAuctions.ts: TreasuryDirect TA_WS
  (keyless, public domain), 6h eager poll over a 30-day window,
  dedup cusip|auction_date (results immutable; reopenings = new
  dates), numeric normalization ('' -> null, never guessed — bills
  carry discount rates, coupons carry yields, each keeps the other
  null). One DERIVED field, labeled: dealer_take = primary dealer
  accepted / competitive accepted. Manifest + RAW route
  /api/data/treasury-auctions.
- HONESTY: the classic tail-vs-when-issued metric needs a paid 1pm
  WI quote — never faked; free stress metrics are bid_to_cover +
  bidder-class shares (manifest states it).
- LIVE E2E: 34 results-complete auctions in the window; dealer takes
  26-38%, plausible. Battery 3 tests (real-fixture normalization,
  pre-result/malformed drops, dedup+reopening+gz).
- Hypothesis (gate-locked): bid-to-cover deterioration + rising
  dealer take precede rate-regime shifts; archive-first, judged
  after depth accumulates.

## 2026-07-05 — [PIPELINE] NWS severe-weather alerts stream + map layer (BUILD ORDER 2 #3) (v1.0.116)

- [T-DATACORE] server/nwsAlerts.ts: api.weather.gov active alerts
  (keyless, public domain, contact User-Agent per policy), 10-min
  poll eager-boot, append-only archive dedup by alert id (CAP
  messages are immutable — updates arrive as new ids), oldest-half
  Set trim (nasaFirms lesson), gz after 2 days. Manifest
  nwsalerts.json. Route /api/data/alerts serves display-simplified
  polygon rings (<=64 pts).
- GEOMETRY HONESTY: zone-coded alerts (geometry:null upstream) are
  archived with null geo and COUNTED (zone_only) in the payload +
  the layer status note ("N zone-coded alerts not drawn") — visible
  cap; zone-polygon resolution filed as follow-up. Archive stores
  centroid+bbox, never full polygons.
- Map layer (T-CLIENT half, same PR per the cross-territory rule):
  fill+line colored by CAP severity (Extreme red / Severe orange /
  Moderate yellow / Minor blue), detail card with not-for-safety-
  of-life note, legend chips, 5-min hidden-gated poll, off by
  default (initial-load budget).
- LIVE E2E: 228 real alerts parsed (4 Extreme / 48 Severe), 40 with
  polygons rendered, 188 zone-only counted; archive wrote 228
  records. Battery: server 3 tests (parse/split, ring decimation
  closure, dedup+gz); harness 0 hard failures at 390/768/1440 —
  first run FAILED on a stale dist bundle (toggle unclickable),
  rebuilt and green; toggle-consistency + legend-parity batteries
  exercised the new layer.
- Hypothesis (gate-locked, from the build order): severe-alert
  clusters over strategic sites lead sector moves by hours-days.
  This PR is display + archive only.

## 2026-07-05 — [REPAIR] Entity spine unreachable in prod — artifact now BUNDLED, not disk-read (v1.0.115)

- [T-DATACORE] Prod verification caught it (the watcher never fired):
  /api/data/aircraft/entity/:hex served entity:null spine_built:false
  through multiple deploys. ROOT CAUSE: loadEntitySpine read
  datacore/aircraft/entity_spine.json from cwd at runtime — works
  locally and in CI (repo = cwd) but the runtime Docker image copies
  dist/, content/, *.py and NEVER datacore/ (frozen Dockerfile), so
  prod could never see the artifact. Every other datacore JSON works
  because it is IMPORTED and bundled into dist — the spine now does
  the same (static import; dist 1.7MB -> 5.7MB, trivial vs the 1GB
  daemon ceiling). Disk-read path retained for tests only.
- RATCHET: new battery test chdirs away from the repo and asserts the
  spine still serves >10k entities — the exact CI blind spot (cwd =
  repo) that let this ship. Would have failed on the old code.
- LESSON ENCODED (module comment + here): server code must NEVER
  disk-read repo files at runtime; bundled import is the only path
  that survives the image. CI green ≠ prod green when the failure is
  in what the image contains — deploy verification is the gate that
  caught this.

## 2026-07-05 — [RESEARCH] Tank-fill v3 GATE 1: FAIL — S1 double-bounce deltas carry no signal; v3.1 pre-registered (v1.0.114)

- [T-DATACORE] PR-2: scripts/tankfill_s1_estimator.py (per-tank
  log10 p95 VV in disk+halo, per-tank series-median self-ratio,
  fill-direction composite, ascending/34 only) + gate-1 via
  tankfill_gate1.py REUSED UNCHANGED. readings_s1.jsonl (61 scenes,
  234/234 tanks median coverage; whole-file rebuild, manifest
  sentinel1readings.json). Battery test_tankfill_s1.py (6,
  synthetic ground truth). Design note stated BEFORE scoring:
  per-tank median normalization cancels exactly in deltas — the
  binding criteria were immune to the normalization choice.
- VERDICT vs pre-stated criteria: n_matched=60 (ample — the
  INSUFFICIENT-SAMPLE branch never triggered), 57 delta pairs,
  reversals present; delta r = +0.056, sign hit = 0.544 -> FAIL.
  Levels r = +0.41: real but UNBINDING and uncredited — it is the
  exact trend-vs-trend inflation pattern that poisoned v1 optical.
- PRIOR vs OUTCOME: stated P(pass) ~25%; outcome in the 75%. Two
  sensors (S2 optical, S1 radar amplitude) and two designs are now
  honestly dead on week-scale Cushing fill from free 10 m imagery.
- v3.1 PRE-REGISTERED (open_questions.md): multi-week deltas
  (28-42d non-overlapping windows) to average down speckle;
  scored ONLY on scenes acquired after 2026-07-05 (out-of-sample by
  time — testing it today on the data that suggested it would be
  fishing); prior discounted to ~15% (attempt #3). If v3.1 fails,
  the free-imagery tank-fill line TERMINATES and sub-meter paid
  imagery (BLOCKED-FOR-MIKE) is the only path.
- Chip acquisition continues (both stacks, pixel-aligned, ~2.5
  PU/scene combined) — raw material compounds regardless.

## 2026-07-05 — [PIPELINE] Tank-fill v3 PR-1: S1 chip client + 24-mo backfill; gate-1 criteria pre-stated (v1.0.113)

- [T-DATACORE] BUILD ORDER 2 #2, first PR of the S1 successor root.
  scripts/cdse_s1_chips.py: discovery via anonymous earth-search
  (sentinel-1-grd; orbit metadata carried — Cushing is ASCENDING
  relative orbit 34 only, S1A+S1D), chips via the CDSE Process API
  (orthorectified SIGMA0_ELLIPSOID, FLOAT32 VV+VH — probed
  double-bounce sigma0 up to ~2500, integer types would clamp), SAME
  bbox as the v2 optical stack, pinned by test_bbox_matches_v2_stack
  so the stacks stay pixel-aligned for fusion. Manifest
  sentinel1chips.json; battery test_cdse_s1_chips.py (7, recorded
  fixture, no network).
- BACKFILL: 61 scenes 2024-07..2026-07, ZERO failures, ~68 PU
  (0.7% of one month's free tier). Spot-check both ends: ground
  median sigma0 0.07-0.10, double-bounce maxima 2100-2535 — the
  physics is present across the whole era.
- CRITERIA + ESTIMATOR DESIGN PRE-STATED in the workup BEFORE any
  scoring (see open_questions.md TANK-FILL v3 STATUS): p95 VV per
  tank disk+halo, log-domain, per-tank self-ratio, fill-direction
  composite = negative normalized double-bounce; gate 1 = same
  criteria as v2 via tankfill_gate1.py reused as-is; matched pairs
  < 20 -> INSUFFICIENT-SAMPLE, not FAIL (12-day repeat may not
  yield 20 adjacent-week deltas). Discounted prior (attempt #2):
  P(pass) ~25%.
- NEXT: PR-2 = estimator + readings stream + gate-1 attempt.

## 2026-07-05 — [PIPELINE] Aircraft entity spine v1 PR-B: artifact SHIPPED + map enrichment (v1.0.112)

- [T-DATACORE] Prod hex list pulled after PR-A deployed (15,248
  distinct archived airframes as of 2026-07-05T08:27Z); spine built
  against the real FAA files: 11,821 matched (78% — the expected US
  share), composition 6,689 corporations / 3,370 LLCs / 935
  individuals / 180 government. Artifact 3.97 MB compact JSON,
  committed at datacore/aircraft/entity_spine.json; manifest
  aircraftspine.json. Sample verified: a6faee -> N549SC, SPARTAN
  EDUCATION LLC, PIPER PA-28-181 (2020) — a flight school owning the
  archive's most-seen airframe, exactly as base rates predict.
- Map enrichment: aircraft detail card now shows "Registered:
  owner · mfr model · year — N-number, FAA registry" via
  /api/data/aircraft/entity/:hex, async after open; non-US hexes show
  nothing extra (never guessed). Harness 0 hard failures at
  390/768/1440; change is click-gated so page visuals are unchanged
  (screenshots reviewed).
- NEXT for this root: gate 1 = join accuracy vs known corporate
  tails; then the utilization x earnings study (BUILD ORDER 2 #1
  hypothesis). Monthly artifact refresh is a session task (manifest
  cadence).

## 2026-07-05 — [PIPELINE] Aircraft entity spine v1 PR-A: hex enumeration + FAA join tooling (v1.0.111)

- [T-DATACORE] BUILD ORDER 2 #1, first of two PRs.
  server/aircraftEntities.ts: distinct archived icao24s streamed
  gz-aware from the aircraft archive (first/last seen, counts,
  callsigns capped at 5, latest type designator; 6h TTL cache,
  stale-served) + spine artifact serving with per-record evidence
  envelopes. Routes: /api/data/aircraft/hexes (the join-key list) and
  /api/data/aircraft/entity/:hex (degrades to entity:null until the
  artifact ships). scripts/build_entity_spine.py: FAA Releasable
  MASTER.txt x ACFTREF.txt join by exact Mode S hex, header-name
  addressing (refuses changed formats), US-gov public domain; ONLY
  archived hexes emitted — never the 300k-row dump.
- GOTCHAS ENCODED: registry.faa.gov 403s non-browser User-Agents
  (probed; browser UA in the script); FAA header cells carry a BOM +
  stray spaces (caught by the verbatim-fixture test).
- E2E SMOKE on the real FAA files (73 MB, 314k rows): 3-hex list ->
  2 matched (SOUTHWEST AIRLINES CO N1801U/737-8 corporate; an LLC's
  M20C), 1 unmatched (non-US, correctly null). Tests: 5 pytest
  (verbatim FAA fixture rows) + 5 node (fold semantics, gz archive
  scan, TTL contract, artifact degradation).
- PR-B after Railway deploys this: pull prod hex list, build + commit
  the spine artifact, live-verify /entity/:hex, then wire the map
  aircraft detail card to it (owner/model/registrant enrichment).
- HYPOTHESIS (unchanged from the build order): corporate-fleet
  utilization x earnings timing; the spine is the join substrate.

## 2026-07-05 — [RESEARCH] BUILD ORDER 2 filed (self-proposed per the standing directive — build order 1 fully resolved) (docs)

- [T-DATACORE] Standing directive 2026-07-05: "when the wishlist is
  empty, generate the next wishlist yourself." Build order 1 is fully
  resolved (streams #1-#8: 6 built, 1 superseded, 1 gate-1 killed;
  trains repair, perf repair, tank-fill workup through its gate-1
  verdict). BUILD ORDER 2 filed in open_questions.md with per-stream
  trading hypotheses and ladder paths; access LIVE-PROBED keyless
  2026-07-05 for the three new externals (api.weather.gov GeoJSON,
  TreasuryDirect TA_WS JSON, Drought Monitor CSV — all responded).
- Order: (1) aircraft entity spine v1 (GIP queue promotion — queued
  work outranks new streams; fleet-utilization x earnings
  hypothesis), (2) tank-fill v3 S1 gate-1 pipeline (successor root,
  discounted prior ~25%), (3) NWS alerts, (4) Treasury auctions,
  (5) Drought Monitor, (6) STB rail traffic.
- Each ships as its own PR under the stream rules (licensing first,
  archive from day one, envelope manifest, RAW until gate 2).

## 2026-07-05 — [RESEARCH] Tank-fill v2 GATE 1: FAIL (layer of death: DATA) — 24-mo backfill + EIA validation (v1.0.110)

- [T-DATACORE] PR-4 of the build plan. Backfill: 99 master chips
  2024-07..2026-07, ZERO failures, ~139 PU total (1.4% of one month's
  free tier). Estimator swept all chips -> 99 readings_v2 lines
  (89 usable scenes, rest cloud-skipped; registration (0,0)
  everywhere — S2 L2A ortho is that good). Gate scorer:
  scripts/tankfill_gate1.py with matching rules + criteria pinned by
  test_tankfill_gate1.py BEFORE the run (prior logged in the
  v1.0.109 entry: P(pass) ~40%).
- VERDICT vs pre-stated criteria (>=20 matched weeks w/ reversal,
  delta r >= +0.3, delta-sign >= 65%): n_matched=72 (ample, reversals
  present), delta r = -0.06, sign hit = 0.50 (n=64), levels r = +0.11
  -> FAIL. Winter/shoulder split (sun_elev < 50, the workup's
  predicted signal carrier): delta r = -0.284 (WRONG direction),
  sign hit 0.42 -> FAIL harder. This is a real fail on a big sample,
  not a thin-sample fail.
- CONFOUND QUANTIFIED: composite vs sun_elev r = +0.40 (+0.0034
  fill/deg) — the tan-zenith inversion does NOT fully remove the sun
  artifact (the exact v1 poison, now measured per-tank). BUT the
  confound is not the whole story: sun-residualized composites
  (post-hoc DIAGNOSTIC, not a gate re-score) still show delta r
  -0.06, sign hit 0.42 — after removing the sun trend there is NO
  residual inventory signal at all. The 10 m optical sub-pixel
  crescent method does not track Cushing inventories. Period.
- LADDER CONSEQUENCES: root DEAD at gate 1 (DATA — reading fails
  external-truth verification vs EIA). Build-plan PR-5/6/7 (weekly
  cadence surface, gate-2 study, unlock) CANCELLED — they presumed a
  pass. Nothing surfaces on /data (gate rule held: no fill% was ever
  shown). readings_v2 production PAUSED — no further appends from a
  falsified estimator. CHIP ACQUISITION CONTINUES weekly (raw
  material is cheap at 1.4 PU/scene, the archive never refills, and
  the S1 successor needs coincident optical pairs).
- SUCCESSOR ROOT FILED (open_questions.md): Sentinel-1 SAR
  double-bounce tank-fill — different physics (roof-wall corner
  reflection scales with roof depth; cloud-immune; ~2x cadence;
  CDSE creds already proven on S1). Discounted prior: this is
  attempt #2 on the same target (Reasoning Standard #4) — P(gate 1)
  ~25%. Criteria to be pre-stated in its own workup before any run.
- HONESTY NOTE: the negative result is the product here — "10 m free
  optical cannot read Cushing fill" is now a COSTED, evidence-backed
  boundary (99 scenes, 72 matched weeks) that no future session
  needs to re-learn, and it sharpens the BLOCKED-FOR-MIKE sub-meter
  imagery entry (that purchase is now the only optical path).

## 2026-07-05 — [PIPELINE] Tank-fill v2 PR-3: crescent estimator BUILT + run on the first real chip (v1.0.109)

- [T-DATACORE] scripts/tankfill_estimator.py: per-tank up-sun vs
  down-sun coverage-weighted B04 ratio (self-normalizing — paint,
  band, atmosphere cancel), inverted through small-s circle-lens
  geometry (f = 4s/(piR); s -> depth via tan(sun_elev)) to fill %;
  SCL per-pixel cloud masking (tank >10% masked skipped, scene >40%
  skipped whole); integer registration vs a reference chip (+/-3 px
  Pearson search); D^2-weighted site aggregates; readings_v2.jsonl
  one line/scene with assumptions (shadow_k=0.35, API-650 heights)
  and per-tank q flags carried on every record. Manifest
  sentinel2v2.json. Tests: test_tankfill_estimator.py (12, fully
  synthetic supersampled rasters with KNOWN crescent geometry —
  depth recovery + ordering, full-tank reads 1.0, registration
  recovery restores measurements, masking, D^2 math, subpixel flag,
  registry load).
- TWO MEASUREMENT DEFECTS CAUGHT BY THE SYNTHETIC BATTERY before any
  real data was trusted: (a) a 1 px edge margin (the "obvious" mixed-
  pixel hygiene) discards exactly the rim pixels carrying the
  crescent — a 0.5 px-reach crescent read ZERO; replaced with
  supersampled coverage WEIGHTS, no margin. (b) assigning whole
  pixels to halves by center puts the proj==0 boundary line in one
  half -> ~5% false darkening on a synthetic FULL tank (fill read
  0.81); fractional half-membership fixed it (full tank reads 1.00).
  Both are exactly the class of silent bias gate-1 would have eaten.
- FIRST REAL READING (June 17 chip, sun_elev 71.1): 234/234 tanks
  measured, reg (0,0); site fill_d2w: enbridge 0.42, hub 0.59,
  plains 0.65, ring 0.58. ALL 234 flagged q=subpixel (June reach
  0.50 px — the v1-poisoning artifact, now labeled per tank, never
  hidden). Winter scenes are the signal carriers; PR-4 backfill will
  weight accordingly.
- PRIOR (stated before gate-1, Reasoning Standard #10): P(gate-1
  PASS) ~40% as filed in the workup; sub-pixel summer readings are
  expected to contribute ~nothing — if gate-1 passes it will be on
  winter/shoulder deltas.
- NEXT: PR-4 = 24-month chip backfill (~140 PU, split across runs)
  + gate-1 attempt vs EIA weekly Cushing stocks (criteria pre-stated
  in the workup: >=20 matched weeks, >=1 reversal, delta r >= +0.3,
  delta-sign >= 65%).

## 2026-07-05 — [PIPELINE] Tank-fill v2 PR-2: CDSE master-chip client BUILT + live-verified (v1.0.108)

- [T-DATACORE] scripts/cdse_chips.py: one 5-band UINT16 GeoTIFF master
  chip per usable Sentinel-2 scene over the Cushing tank registry.
  Discovery via Element84 earth-search (ANONYMOUS — same source as v1;
  dates, cloud %, sun angles); chips via the Sentinel Hub Process API
  on CDSE (credentialed, free tier). Committed metadata:
  datacore/sentinel2/chips_index.jsonl (dedup by scene, per-scene
  est_pu, monthly PU accounting with a hard 50%-of-free-tier refusal);
  chip binaries gitignored (~1.6 MB/scene, regenerable). Manifest:
  datacore/manifests/sentinel2chips.json. Tests: test_cdse_chips.py
  (7, no network; recorded 2026-07-05 earth-search response).
- WORKUP CORRECTION 1 (bbox): the filed workup chip bbox
  [-96.80,35.90,-96.72,35.98] MISSES 20 of 234 measurable tanks
  (registry extends east to lon -96.7149). Corrected CHIP_BBOX
  [-96.770,35.922,-96.712,35.960] covers all measurable tanks with
  >=250 m margin at ~1/3 the pixel area — 1.40 PU/scene vs the
  workup's ~4.1. RATCHET: test_chip_bbox_covers_every_measurable_tank
  pins coverage against the live registry file.
- WORKUP CORRECTION 2 (decode): pillow CANNOT read 5-sample/pixel
  TIFFs ("pillow decodes it" was wrong); tifffile reads (H,W,5) uint16
  directly — session-local dep, documented in the script for PR-3.
- WORKUP CORRECTION 3 (sun angles): the CDSE SH Catalog does NOT
  expose view:sun_elevation/azimuth at all (probed live; fields.include
  returns nothing) — discovery stays on earth-search, which carries
  both angles per scene.
- LIVE E2E (2026-07-05): token OK; 4 usable scenes since 2026-06-01;
  pulled S2B_14SPE_20260617 (cloud 0.08%): 523x420x5 uint16, plausible
  DN ranges (B08 NIR mean 3124), SCL classes {2,4,5,6,7}; index record
  written; 1.4 of 10,000 monthly PU spent.
- NEXT: PR-3 crescent estimator (sun-sector/anti-sun-sector sub-pixel
  ratio over the registry tanks, readings_v2 schema), then PR-4
  backfill + gate-1 attempt (criteria pre-stated in the workup).

## 2026-07-05 — [REPAIR] Audit defects #2/#9/#10 closed: trains health override, sentinel2 staleness surface, silent-cap edges (v1.0.107)

- #2: the layers registry statically claimed trains "live" through the
  entire outage. /api/data/layers now health-overrides trains to
  status "down" (+note) when the cache is >45 min stale or no source
  is ok (the eager tick refreshes every 10 min, so staleness = real
  outage); the client renders "feed down" red and auto-disables the
  toggle — a dead feed never advertises live again.
- #9: sentinel2 readings are git-side/session-run — a stall was
  invisible. platformStats now exposes sentinel2_last_reading +
  age_days so every DAILY check and dashboard sees staleness.
- #10a: edgar13f with a NULL entryTotal that parses exactly to the
  250 cap can no longer ship 250 rows that look complete — treated as
  over-cap (summary-only, holdingsOmitted=true) per the never-silent
  cap doctrine.
- #10b: nasaFirms bounded its dedup memory with a full clear() — in
  peak fire season that forgets the entire ~3h NRT window at once and
  re-appends it as duplicates. Now trims the OLDEST half (insertion
  order), keeping the recent window intact.
- ALL TEN audit defects from the 2026-07-05 quality audit are now
  closed (#209 #210 #211 #212 #214 #216 + this). Harness green; suites
  green.

## 2026-07-05 — [REPAIR] Manifest accuracy: aircraft field_map corrected + COT manifest created (docs)

- Audit defects #7 + #8 (both manifest-accuracy, one logical change:
  the reader contract must match the writers):
- aircraft.json documented a "gs" field that doesn't exist — the
  writer emits g (on-ground flag) and v (speed m/s, rounded). Fixed.
- The COT stream (Python-side, routine #191) had NO manifest and
  escapes the envelope test (which only scrapes server/*.ts writers).
  datacore/manifests/cot.json now documents it honestly: SINGLE keyed
  JSON with atomic replace (an explicit exception to append-only,
  justified by the keyed weekly-history shape), full per-week field
  map from cftc_cot.py _derive_fields, DERIVED fields labeled, the
  3-days-stale-by-design publication lag stated, and a note that
  Python-side archives are documented by convention rather than
  enforced by the TS envelope test.

## 2026-07-05 — [RESEARCH] Stream #8 gate 1: FAIL — Google Trends via pytrends (layer of death: DATA/access)

- PRIOR (filed before the run): gate-1 stability test may kill the
  stream; PASS = median cross-pull r > 0.95 AND >= 80% of the 20-term
  panel ok every round.
- RUN 1 finding: pytrends (upstream ARCHIVED 2025-04) is
  dependency-rotten — with retries enabled it dies on urllib3 v2
  (Retry method_whitelist TypeError) before reaching Google at all.
  A retries-free control pull SUCCEEDED, so run 1 measured our probe
  config, not the source; the probe was corrected and rerun (rigor:
  the first FAIL would have been a false attribution).
- RUN 2 (corrected, 3 rounds x 20 brand terms, 8-min gaps): rounds
  1-2 = 20/20 ok; STABILITY PASSED emphatically — median cross-pull
  r = 0.998, min 0.952, no unstable term. But round 3 collapsed to
  6/20 on HTTP 429s: Google rate-limits the unofficial path after
  ~45 pulls/35min. VERDICT: FAIL on the availability half of the
  pre-stated criterion.
- HONEST SHAPE OF THE DEATH: the DATA is reproducible; the free
  ACCESS is not production-reliable, and the only client library is
  abandoned. NO archiver, NO manifest, NO daemon route built (as
  planned — no dead code for a dead stream). Residual option filed in
  open_questions: an ultra-low-cadence weekly single-round pull
  (20 pulls/week sits far under the observed limit) could be re-probed
  if a hypothesis ever NEEDS Trends; paid alternatives remain in
  BLOCKED-FOR-MIKE. Probe: scripts/gtrends_probe.py (re-runnable).

## 2026-07-05 — [REPAIR] fredmacro vintage dedup: restart-bloat + revert-drop fixed (v1.0.106) [T-DATACORE]

- Audit defect #6, both halves: (a) seedSeen covered 3 days while each
  poll fetches 120 days — a restart >3d after backfill re-appended
  ~120d x 31 series as duplicate vintage rows with fresh rt; (b) the
  (s,d,v) SET dedup silently dropped a revision that REVERTS to a
  previously seen value — but a revert IS a vintage transition.
- FIX: dedup is now LATEST-VALUE per (s,d) — seed the current value
  from up to 130 days of files (oldest->newest so the last write
  wins), append only when the published value CHANGES. Duplicates die;
  every transition including reverts is recorded. Manifest cadence
  line updated to state the semantics.
- RATCHET: honest restart simulation via a real state-reset hook
  (in-memory map cleared, disk seed must carry it) + a
  revise-then-revert sequence test. 8/8 suite green.

## 2026-07-05 — [PIPELINE] Tank-fill v2 PR-1: Cushing tank registry from OSM (v1.0.105) [T-DATACORE]

- First build step of the filed tank-fill v2 workup: 333 storage-tank
  polygons pulled from OSM Overpass -> datacore/sentinel2/
  cushing_tanks.geojson (git-versioned fixed geometry for the
  crescent-shadow estimator). Per tank: center, equivalent-circle
  diameter from polygon area, site assignment (3 tank_farm sites +
  ring), API-650 48ft default height with an explicit provenance flag
  (OSM has zero height tags here — assumptions labeled, never silent),
  measurable_10m flag (234 tanks >= 40m).
- REGISTRATION VALIDATION (the workup's sanity check): computed ring
  shell capacity = 74.9M bbl vs EIA's ~76M bbl published Cushing
  working capacity — the geometry + default height reproduce the known
  ring within ~1.5%. Strong evidence the registry is sound before any
  imagery sampling happens.
- ODbL attribution in the file + pinned by test (counts, provenance
  flags, bbox bounds, capacity plausibility). Builder script
  re-runnable: scripts/build_tank_registry.py.
- Next per the workup: PR-2 CDSE chip client (fixtures in CI), PR-3
  crescent estimator.

## 2026-07-05 — [REPAIR] earnings8k manifest drift: acceptanceDatetime + ticker now actually stored (v1.0.104) [T-DATACORE]

- Audit defect #5 — the honesty-critical one: the manifest documented
  acceptanceDatetime ("lookahead-free event time") and a ticker entity
  key that the writer NEVER stored; gate-2 work reading the manifest
  would have assumed a timestamp that didn't exist (filedAt is a date
  only). Manifests are the reader contract — drift there poisons
  downstream honesty silently.
- FIX (writer side, not a docs downgrade): acceptanceDatetime = the
  getcurrent entry's <updated> timestamp (when this feed made the
  filing publicly visible — the honest "knowable" time, ISO w/offset);
  ticker = EXACT numeric-CIK match vs SEC company_tickers.json (24h
  cached; failed/empty fetches never cached; unlisted filers stay
  null — never guessed). Manifest wording tightened to say precisely
  what the fields are.
- 2 ratchet tests (feed <updated> capture; CIK map exactness) +
  cache-pollution fix found by the tests themselves (a failed tickers
  fetch was being cached 24h). 11/11 suite green.

## 2026-07-05 — [REPAIR] Optionchains: crash-safe day claim, gzip lifecycle, holiday skip (v1.0.103) [T-DATACORE]

- Audit defect #4, three parts: (a) .last_run_day was claimed BEFORE
  the run — any crash/total failure permanently lost that trading day
  (forward-only archive; a lost day can never be re-bought). Now an
  in-memory guard prevents double-fire and the day is claimed AFTER a
  run that wasn't a total failure (shouldClaimDay: empty universe
  claims, partial failure claims + logs, all-failed retries next
  hourly tick). (b) The manifest promised .jsonl(.gz) but nothing
  gzipped the dir (~3-5MB/day raw) — gzipOldChainDays now runs the
  standard 2-day lifecycle after each run. (c) The archiver ran on
  July 3 (full NYSE closure), burning ~120 API calls to archive stale
  quotes as a fresh day — shouldRunNow now skips holidays PARSED from
  market_calendar.py (frozen source of truth read at boot, never
  duplicated — December's year-add flows through automatically; parse
  failure degrades to weekend-only, never a crash).
- 4 new tests incl. parsing the real market_calendar.py (July 3
  present), the exact holiday-evening skip, claim semantics, and the
  gzip round-trip. 10/10 suite green.

## 2026-07-05 — [REPAIR] archiveStats enumerates from disk — archive-gap rule now covers every kind (v1.0.102) [T-DATACORE]

- Audit defect #3: archiveStats() hardcoded six position kinds, so
  fires, filings, earnings8k, filings13f, fredmacro, optionchains and
  every new stream (usaspending, fda, usgswater, gdelt) were INVISIBLE
  to /api/data/archive/stats — a stalled key or dead archiver would be
  discovered by accident, and the archive-gap rule was unenforceable
  for most of the archive.
- FIX: enumerate directories from disk (new streams appear with ZERO
  code change), keep the position kinds explicitly listed so they
  report {files:0} loudly before first write, skip non-files.
- Side benefit: the landing hero's streams_recording count now grows
  automatically as the new stream dirs land on the volume.
- RATCHET: test creates a never-before-seen stream dir and asserts it
  appears in stats without a code change.

## 2026-07-05 — [REPAIR] Eager archive tick: aircraft + trains no longer visitor-dependent (v1.0.101) [T-DATACORE]

- Audit defect #1 (the top finding): aircraft and trains archiving was
  REQUEST-driven — archiveAircraft/archiveTrains run as fetch side
  effects, so no visitors = no archive. 11 hourly gaps each were
  already permanent in the archive's first 36h, and the record was
  visitor-BIASED (only what someone happened to view). Vessels had the
  eager fix (KNOWN BROKEN #9); these two didn't.
- FIX: a 10-minute tick fires one aircraft snapshot rotating across
  four strategic-site regions (Cushing/south-central, US NE corridor,
  LA/Long Beach, Rotterdam ARA — each region ~every 40 min under the
  point-radius API cap) plus one global trains snapshot (which also
  keeps trainsCache warm for visitors). One extra upstream call per
  10 min per feed. Fires once at boot, not after the first interval.
- RATCHET: source-pin test (regions >= 3, both feeds, 10-min cadence,
  boot-fire) so the tick can't be silently de-scoped.

## 2026-07-05 — [PRODUCT] River-gauge /data layer — stream #6's map surface (v1.0.100) [T-CLIENT + shared registry]

- The USGS stream's geographic surface, per the legend same-PR rule:
  vt-gauge registry icon (staff gauge + water wave, SDF), rivergauges
  registry entry (environmental group, OFF by default — reference
  layer, initial-load budget respected), datamap effect (hourly
  hidden-gated refresh matching the server poll; detail card shows
  reading + provisional/approved label + USGS monitoring link;
  discharge-only gauges labeled ft3/s), legend entry gated on enable.
- Harness: registry + /api/data/rivergauges fixtures; toggle-
  consistency (1440 exercises every live layer incl. gauges) + legend
  parity + perf gates all green 10/10.

## 2026-07-05 — [PIPELINE] Stream #7: GDELT facility-event alerts (v1.0.99) [T-DATACORE]

- Built per the verified brief: server/gdeltEvents.ts — 15-min Events
  export files via the lastupdate.txt pointer (HTTP ONLY — the host's
  HTTPS cert is invalid, verified; acknowledged in the manifest),
  fflate unzip, column indices pinned against a real export, CAMEO
  unrest/strike filter (roots 14/17/18/19/20 + 143x) x ~0.5° boxes
  around the 16 strategic sites -> KB-scale archive, dedup by
  GlobalEventID, republished-file skip, 15-min eager poll,
  /api/data/facility-events (48h rolling window), envelope manifest.
- HONESTY encoded: media event MENTIONS with city/ADM-approximate geo
  (never facility-exact); CAMEO cannot see clean industrial accidents
  (FIRMS/AIS are the physical sensors) — hypothesis is unrest/strike
  bursts as verification prompts, gate-2 = burst->own-sensor
  confirmation rate. GDELT attribution required and carried.
- LIVE E2E: real export downloaded/unzipped/parsed; 0 matched events
  this window (holiday news cycle near 16 sites — the tight filter
  working as designed, not a defect).
- 5 tests; offline pytest green. New dep: fflate (pure-JS zip).
- Stream #8 (pytrends) is a PROBE next, not a build — per the
  downgraded plan after upstream abandonment.

## 2026-07-05 — [PIPELINE] Stream #6: USGS river gauges — barge-corridor water levels (v1.0.98) [T-DATACORE]

- Built per the verified brief: server/usgsWater.ts — 14 live-verified
  gauges (Mississippi St. Louis->Belle Chasse + Missouri/Illinois
  tributaries + 4 Ohio R gauges; dead Metropolis gauge excluded),
  BOTH parameter codes requested (Memphis/Vicksburg are
  discharge-only — verified), one request covers all sites, 1h eager
  poll, /api/data/rivergauges route, envelope manifest.
- Vintage discipline: provisional (P) -> approved (A) revisions append
  as new rows with rt. USGS -999999 sentinels dropped. Low-water
  barge-stress SIGNAL stays gate-2-locked — this is RAW readings only.
- LIVE E2E: 26 readings across the 14 sites (several publish both
  params; Ironton shows a negative discharge — real backwater
  behavior, archived as-is per RAW doctrine).
- QUEUED (own [PRODUCT] PR per the legend same-PR rule): the /data
  registry layer + gauge map icons + legend entry — gauges carry
  lat/lon and are map-plottable.
- 5 tests; offline pytest green.

## 2026-07-05 — [PIPELINE] Stream #5: FDA binary events — approvals + AdCom dates (v1.0.97) [T-DATACORE]

- Built per the verified brief: server/fdaEvents.ts — openFDA drugsfda
  approvals (30-day rolling window) + Federal Register FDA advisory-
  committee meeting notices (the FREE forward-looking catalyst that
  preserves the IV-ramp hypothesis; PDUFA dates are legally
  unpublishable, 21 CFR 314.430 — stated in manifest + route, and we
  do NOT scrape aggregator calendars). 6h poll, 2 req/cycle (far under
  keyless caps), /api/data/fda-events, envelope manifest.
- Meeting dates parsed from official notice text with a CONFIDENCE
  LABEL (parsed/unparsed) — an unparsed date stays null, never
  guessed. pub (FR publication date) = when the public could know.
- LIVE E2E caught an honesty bug BEFORE ship: openFDA returns whole
  applications with full submission history, so the naive parse
  emitted 1,619 "approvals" including years-old supplements; the
  window filter drops out-of-window AP submissions -> 106 in-window
  approvals + 40 adcom notices (13 with parsed dates). Regression test
  pins the window.
- LADDER: gate 1 = adcom date accuracy vs 20 known events (runnable
  from the archive + FR links); gate 2 = IV-ramp reproducibility
  around parsed adcom dates on OUR archived option chains (recording
  since 2026-07-06). 9 tests green; offline pytest green.

## 2026-07-05 — [PIPELINE] Stream #4: USAspending contract-awards archiver (v1.0.96) [T-DATACORE]

- Built per the verified brief: server/usaSpending.ts — transaction-
  level search polled by last_modified_date (the publication axis),
  EXPLICIT $25k floor applied client-side via two sorted passes with
  early stop (the API's own amount filter is award-LIFETIME — trap),
  deobligations kept symmetrically, dedup (aid,mod,amt) with FPDS
  corrections appending as vintage rows, 6h eager poll,
  /api/data/contracts route, envelope manifest.
- TICKER MAPPING precision-first: SEC company_tickers.json exact
  normalized-name match (ambiguous normalized names DROPPED — never
  guess), award-detail parent for large unmatched rows (the
  recipient-profile endpoint is banned: vintage-less, provably wrong
  parents), persistent UEI->ticker cache that compounds forever.
  Unmatched rows archive tkr:null and are skipped by consumers.
- HONESTY ENCODED: rt is the only event date (action_date = signature
  date); DoD/USACE publish ~90 days late — manifest + route note carry
  it; gate 2 must cohort/exclude DoD. DUNS never stored (D&B
  proprietary); UEI only.
- LIVE E2E: 308 real txns >= $25k pulled over the 2-day holiday
  window; 23 name/cache-matched (GM resolved by name, then served
  from cache within the same run — the compounding works); 8 queued
  for parent lookups; all 308 archived and deduped on re-run.
- LADDER: gate 1 (recipient->ticker precision on a 50-award
  hand-check) is now RUNNABLE from the archive alone (mm + mname audit
  fields); not yet attempted. Gate 2 blocked on gate 1 + return
  windows. 8 tests green; offline pytest suite green.

## 2026-07-05 — [RESEARCH] Parallel subagent batch: streams 4-8 verified, tank-fill v2 workup, datacore quality audit (docs)

- Throughput directive executed: five subagents ran while the main
  thread shipped repairs. Filings landed in open_questions.md +
  wishlist BLOCKED-FOR-MIKE; summaries:
- USAspending (#4, next build): endpoints live-verified incl. a $900M
  DOE award to a Centrus (LEU) subsidiary as the hypothesis exemplar;
  DoD publishes ~90 DAYS late (gate-2 cohort/exclude); $25k explicit
  cap = 99.74% of dollars in 20% of rows; DUNS is D&B-proprietary —
  UEI only; recipient-profile parent endpoint is vintage-less and
  WRONG (two proven cases) — parents from award-detail only.
- FDA (#5): PDUFA dates not freely publishable (21 CFR 314.430) —
  free substitute preserves the IV-ramp hypothesis: Federal Register
  AdCom meeting notices (verified) + openFDA approvals.
- USGS (#6): 14 gauges live-verified (Memphis/Vicksburg are
  discharge-only; one dead gauge caught); gets a /data layer.
- GDELT (#7): 15-min export files trivially small after
  CAMEO x facility-bbox filter; host is HTTP-only; CAMEO can't see
  industrial accidents — hypothesis reworded to unrest/strikes.
- pytrends (#8): upstream ARCHIVED 2025-04 — downgraded to a gate-1
  probe; no archiver/manifest unless stability passes.
- Tank-fill v2: full crescent-shadow workup filed (sub-pixel aggregate
  estimator, OSM 333-tank registry verified, 0.4% of CDSE free quota,
  delta-based gate-1 criteria fixing the v1 trend inflation, 7-PR
  build plan). BLOCKED-FOR-MIKE: nothing in the core build.
- Quality audit: 10 prioritized defects filed as the DATACORE DEFECT
  QUEUE in open_questions (top: request-driven aircraft/trains
  archiving = permanent gaps; archiveStats blind to 8 of 11 kinds;
  optionchains loses a day on crashed runs).

## 2026-07-05 — [REPAIR] /data map performance 3/3: aircraft low-zoom render decimation (v1.0.95) [T-CLIENT]

- Two-layer split on ONE source: aircraft-sym (full, minzoom 4.5) +
  aircraft-sym-lo (maxzoom 4.5, stable rank-hash filter keeps 35%).
  At the default continent zoom, 10k overlapping icons were pure
  overdraw; zooming past 4.5 shows every aircraft. Rank hashed from
  icao24 so a given aircraft never flickers in/out across refreshes.
  Click/cursor handlers wired to both layers; teardown removes both.
- NO DATA LOSS by construction and by gate: the source always holds
  the full feed — the 2/3 gate's data-richness guard read 10,000
  unique source features while rendering 3,507. This is the profiled
  fill-rate fix (rendering path), not a data cut.
- Measured (harness, SwiftShader): median frame 117 -> 83ms @1440,
  83 -> 67ms @768, 33 -> 17ms @390; p95 200 -> 117ms @1440; sampled
  frames per pan window 66 -> 116 @390 (smoother). 10/10 green under
  the v1.0.94 calibrated gates.
- Harness aircraft samplers now query both layers (mechanical layer-id
  adaptation; gate thresholds untouched).
- Queued: apply the same split (or clustering) to fires BEFORE the
  FIRMS key lands — fires has no low-zoom mitigation and can't be
  harness-verified until its fixture carries data (noted, not built).

## 2026-07-05 — [RULE-REVIEW] /data map performance 2/3: calibrated perf gate in the harness (v1.0.94) [T-CLIENT tooling]

- Measurement change, own PR per MEASUREMENT INTEGRITY. Performance
  regressions now FAIL the harness like visual ones: map-page TTI
  gate 3000ms (observed ceiling ~1.3s), per-width median-frame gates
  120/200/250ms @390/768/1440 (observed 33-117), p95 gate 350ms
  (observed ceiling 200; warn band from 250), all under SwiftShader
  as regression guards — the S24 remains the on-device acceptance
  ruler per DESIGN.md.
- Metric before vs after on identical inputs: same TTI/median/p95
  numbers measured, only thresholds tightened — the change can only
  make builds FAIL more, never look better (bias direction stated).
- DATA-RICHNESS GUARD: at 1440 the aircraft SOURCE must hold >=9500
  unique features (deduped by icao24) regardless of rendered count —
  the enabler for low-zoom decimation (3/3) and the ban on doing it
  by dropping data.
- Deliberately NOT added: a payload-bytes budget — the harness fixture
  server doesn't run the prod compression middleware, so it would
  measure the wrong thing; compression is pinned by the express
  round-trip test in server/compression.test.ts instead.

## 2026-07-05 — [REPAIR] /data map performance 1/3: eliminate redundant network + render work (v1.0.93) [T-CLIENT+server/index.ts]

- Priority-repair directive: map slow to initially load, sluggish with
  all layers on. Constraint honored: NO layer removed, NO data
  richness reduced — engineering only. Profiled first (subagent
  report); this PR ships the waste-elimination tier (biggest win per
  unit risk); the harness perf GATE ships separately next (measurement
  change = own PR), then low-zoom decimation behind that gate.
- SERVER: response compression (compression middleware in
  server/index.ts — Railway's edge does NOT gzip): aircraft snapshot
  ~0.8MB -> ~120KB, powerplants 800KB -> ~200KB; ~70% of initial-load
  bytes cut. Default filter skips the already-compressed wxtile PNGs.
  Pinned by an end-to-end gzip round-trip test.
- CLIENT (datamap.tsx): (1) moveend fetches debounced 400ms — bare
  moveend fired a full fetch + 10k-feature rebuild on EVERY camera
  settle including each wheel step; (2) hidden-tab gating on
  aircraft/vessels/trains/insider/earnings polls + immediate refresh
  on visibilitychange return; (3) setStatus no-op bail — five default
  polls re-rendered the whole page every 15-60s tick with identical
  payloads; (4) insider/earnings panel-count polls 60s -> 300s (they
  render a count; server caches upstream at 15-min); (5) map event
  handlers de-duplicated — click/mouseenter/mouseleave were stacked on
  every toggle cycle (N clicks -> N detail cards + N trail fetches).
- Harness: 10/10 green at 390/768/1440; fields-on battery's mounted
  wait extended to require BOTH arrows AND temp labels placed (fixed a
  1440 placement-timing flake my re-render reduction exposed — labels
  were visibly rendered in the screenshot while the sampler read 0).
- DEPLOY-VERIFY of v1.0.92 (previous entry): prod /api/data/trains
  responds 200 in 0.8s with 126 live trains (FI 86 ok + NO 40 ok) —
  the permanent-hang repair holds in production.

## 2026-07-05 — [REPAIR] /api/data/trains permanent hang: stuck in-flight promise poisoned the route (v1.0.92) [T-DATACORE]

- Production: /api/data/trains returned NOTHING (HTTP 000 at 90s, zero
  bytes) while every other endpoint was healthy; /data showed the
  trains layer erroring. Archive evidence: trains recorded fine
  through the 05:00 UTC hour, dead after — so the feed worked, then
  ONE fetch wedged.
- ROOT CAUSE (architecture, not upstream): the route shares one
  in-flight promise across requests (`.finally()` clears it) — but
  .finally only fires when the promise SETTLES. One fetchTrains stuck
  past its per-source AbortSignal timeouts (upstream/undici pathology)
  → every subsequent request awaited the same dead promise, forever,
  surviving upstream recovery. /api/data/aircraft carried the
  IDENTICAL latent pattern per bbox key.
- Also found while diagnosing: Digitraffic now REQUIRES
  Accept-Encoding: gzip (406 otherwise — verified live). Undici sends
  it by default, but the header is now explicit so a runtime/bundler
  change can't silently 406 the feed.
- FIX (server/routeGuards.ts, pure + unit-tested): (1) raceDeadline —
  no request waits past 15s; falls to stale-beats-spinner; (2) slot
  expiry — an in-flight older than 45s is abandoned and a fresh fetch
  starts (one stuck fetch can no longer poison the future); (3)
  identity-guarded cleanup — a late-settling orphan can't clobber its
  replacement. Applied to trains AND aircraft.
- RATCHET: 7 tests incl. the exact outage shape (stuck promise →
  deadline rejection), orphan-vs-replacement identity, unhandled-
  rejection absorption, and wiring pins (both routes use the guards;
  gzip header present). Deploy-verify: after merge, /api/data/trains
  must respond <15s with either data or stale/503 — never hang.
- Related audit findings (SEPARATE PRs queued): trains/aircraft
  archives are request-driven (gaps never refill — needs eager tick);
  trains lacks a health-aware registry status override.

## 2026-07-05 — [REPAIR] Temp value-labels ate the wind arrows (v1.0.91) [T-CLIENT]

- Production report: with Temperature value-labels ON, wind arrows
  disappeared leaving orphaned kt text. ROOT CAUSE: both sample the
  same server-side grid points; wx-temp-labels sits higher in the
  style, so it wins MapLibre's symbol placement pass, and the arrows
  layer was only half-shielded (icon-allow-overlap true, but its kt
  text still collided and the icon/text pair got split by the
  collision pass — one half surviving without the other).
- FIX (by construction, not tuning): (1) the arrow+kt pair is now ONE
  unit fully outside the collision pass in BOTH directions —
  allow-overlap + ignore-placement on icon AND text, text-optional
  removed — it can never be hidden, never hide others, never be
  separated; density stays bounded by the sampled grid, so opting out
  of declutter is safe. (2) At shared grid points the two label sets
  dodge by OFFSET: temp label anchored bottom at [0,-1.2] (above the
  point), arrow at the point, kt at [0,1.3] (below).
- RATCHET (repair rule 3): the fields-on battery never exercised this
  state — value-labels defaults OFF, so the bug shipped untested. The
  battery now turns the sub-toggle ON and asserts: temp labels placed,
  arrows still placed, the arrows layer's four collision-exemption
  flags intact, and the temp anchor offset intact. Harness green
  390/768/1440; magnified screenshot review confirms 72°F above /
  arrow / 17 kt below at shared points.
- Harness gotcha recorded: .vt-field-controls is a SIBLING of the
  data-vt-layer row — sub-toggle locators anchor on label text, not
  the row selector.

## 2026-07-05 — [PIPELINE] Stream #3 gate 1: PASSED 10/10 (prod vs fredgraph.csv)

- v1.0.90 deployed; prod /api/data/macro serves 28 public series with
  FRED attribution; restricted series (VIX/BAML/UMCSENT) confirmed
  ABSENT from the live payload.
- Gate 1 spot checks — prod latest value vs the FRED web UI's own
  fredgraph.csv export, exact match required: DGS10 4.48, DGS2 4.17,
  T10Y2Y 0.35, SOFR 3.66, ICSA 215000, UNRATE 4.2, CPIAUCSL 333.979,
  WALCL 6724564, DCOILWTICO 71.87, DTWEXBGS 120.8866 — 10/10 MATCH
  across daily/weekly/monthly cadences. Stream #3 gate 1 PASSED;
  vintage archive recording from this deploy onward.

## 2026-07-05 — [PIPELINE] Stream #3: FRED macro regime feed (v1.0.90) [T-DATACORE]

- Human set FRED_API_KEY in Railway (not in the session env — noted in
  wishlist); built stream #3 same day: server/fredMacro.ts — 31
  regime-relevant series (curve, Fed-produced stress indexes, labor,
  inflation, activity, liquidity, WTI/dollar), key-gated 6h poll,
  /api/data/macro route, envelope manifest.
- POINT-IN-TIME VINTAGE ARCHIVE (the build's real asset): FRED revises
  history silently; every observation is archived with rt = as-seen
  date and a revision appends a NEW row (dedup key is (series, date,
  value), never (series, date)). Recording forward turns the free feed
  into the paid ALFRED-style vintage dataset (BUILD-FIRST #2) — and
  protects future regime backtests from lookahead via revised data
  (Reasoning Standard #7): "known on day X" = filter rt <= X.
- LICENSING SPLIT (checked first): CBOE VIX, ICE BofA HY OAS, UMich
  sentiment are third-party copyrighted → license:"restricted",
  archived for internal regime use only, EXCLUDED from the public
  payload by buildMacroPayload (pinned by test). All other 28 series
  are US-gov/Fed-produced. FRED attribution on the route + manifest.
- Gate 1 (values match FRED web UI on 10 spot checks): runs against
  prod /api/data/macro vs the keyless fredgraph.csv export after this
  deploys (API key lives only on Railway) — result appended below when
  checked. Gate 2 (regime-conditioning improves an existing validated
  signal) not attempted; no validated signal exists yet to condition.
- Tests: 6 new (documented-shape parser incl. string values + "."
  missing, series-table license pins, revision-append vintage
  behavior, mocked end-to-end refresh + restricted-exclusion pin, key
  gating, wiring pins). Offline pytest suite green (120 passed). No
  backtest — data pipeline only.

## 2026-07-05 — [PRODUCT] Hero globe: real registry symbols instead of dots (v1.0.89) [T-CLIENT]

- Directive 2026-07-05 (#data-intel hero only): the globe's colored
  dots became REAL vehicle silhouettes from the SAME shared icon
  registry the /data map uses (lib/mapIcons SDF shapes + classifiers,
  lazy-imported with maplibre) — globe and map cannot diverge, and
  future icon improvements land on the globe automatically.
- Aircraft: classifyAircraft(type, category) → jet/prop/heli/generic
  silhouettes, icon-rotate bound to heading. Vessels:
  classifyVessel(shiptype) → tanker/cargo/boat hulls, rotated to COG.
  Sites: SITE_ICON category markers (anchor/tank-rings/factory),
  upright, amber with SDF halo glow.
- MISSING-CLASSIFICATION DEFAULTS: deterministic hash of the track id
  picks from a believable mix (jets-heavy for aircraft, cargo/tanker
  mix for vessels) — stable across refreshes, and real classification
  always wins when the feed carries it. Honest note: shapes for
  unclassified tracks are DISPLAY defaults, not data claims; headings
  keep the map's ?? 0 convention, never fabricated.
- Perf: symbol layers are the SDF path the /data map profiled (M4,
  fill-rate bound); small fixed icon-size (0.32/0.26 air, 0.30/0.24
  sea desktop/phone) + existing phone caps (500/300) keep the budget —
  cap density, never stutter.
- Harness: new landing-globe battery (scroll #data-intel into view,
  wait for placed features, assert symbol layers + icons varied +
  every icon registered via hasImage + icon-rotate bound to heading;
  landing-globe-{w}.png artifacts). Full run green at 390/768/1440,
  0 hard failures. Self-review: harness fixture (10k aircraft in a US
  box) is too dense to judge legibility, so a production-sparsity
  probe (220 aircraft / 140 vessels global) verified all 10 icon
  shapes render and read at both 1440 and 390 — planes/hulls/anchor/
  tank-rings/factory all legible.
- Legend rule note: the hero is a decorative background, not a map
  surface with a legend — the symbols' legend entries live on /data
  where the same registry shapes are already legend-paired (parity
  battery). No new unpaired symbol class was introduced.

## 2026-07-05 — [PIPELINE] Stream #2: EDGAR 13F-HR institutional-holdings archiver (v1.0.88) [T-DATACORE]

- Built stream #2 of the DATA STREAM EXPANSION build order end-to-end:
  server/edgar13f.ts — getcurrent-feed fetcher (reuses Form 4's Atom
  parser), namespace-tolerant primary-doc + information-table parsers,
  append-only JSONL archive under <archive>/filings13f/ (accession
  dedup, old days gzipped), 15-min eager-boot poll, routes
  /api/data/filings13f + /history, envelope manifest.
- LADDER gate 1 (DATA): PASSED for the parser — fixtures are two real
  live-fetched filings (BURKETT 0001762716-26-000003, ATMOS
  0001905162-26-000005), every asserted field hand-checked against the
  filed XML; live end-to-end pull of 4 real filings verified (periods
  normalized, totals match, archive round-trips). Gate 2 (new-position
  clustering vs forward returns, 45-day lag modeled) NOT attempted —
  RAW as-filed records only, no predictive claim.
- FOCUSED-MANAGER CAP (explicit, never silent): filings with >250
  positions archive summary-only (holdings omitted; info table not
  even fetched). This encodes the hypothesis — capacity-constrained
  managers in small caps (EDGE DOCTRINE #2); mega-manager index
  tables would dominate archive bytes with no cluster signal. Stated
  in the manifest (_cap), the API (focused_cap), and pinned by test.
- Data notes for gate 2: value field is FULL USD (post-2023 rule, not
  thousands — manifest documents it); periodOfReport normalized from
  EDGAR's MM-DD-YYYY; amendments (13F-HR/A) flagged via
  submissionType, never merged into originals.
- Tests: 10 new (parsers on real XML, doc-name picker on both real
  directory shapes, cap behavior incl. no-fetch assertion, archive
  round-trip + dedup, wiring pins). Full offline pytest suite green
  (120 passed); manifest sweep green. No backtest — data pipeline
  only, no trading-logic change.
- Timing: Q2-2026 13F season opened this week (deadline Aug 14) —
  every poll from merge onward lands in the heaviest filing window of
  the quarter; a season's small-manager tail accumulates from day one.

## 2026-07-05 — [RESEARCH] Data-stream expansion: audit + 8-stream build order filed, hypotheses before pulls (docs)

- Stream-expansion directive audited and filed (open_questions DATA
  STREAM EXPANSION): RECORDING NOW = aircraft, vessels, trains, fires,
  Form 4, 8-K language, option chains (Monday), Sentinel-2 readings,
  and COT — stream (2) was merged by a concurrent routine (#191,
  v1.0.86) while this session built the hero; supersession honored, no
  duplicate build.
- Build order 2-8 filed with a PRIOR + LADDER PATH per stream BEFORE
  any first pull (Reasoning Standard #10): 13F clusters (45-day lag
  honestly modeled), FRED (regime input, never traded alone),
  USAspending (award/mcap on small caps), FDA calendars (theta-side
  IV-ramp structure vs our own archived chains), USGS water
  (conditional drought-year signal), GDELT (alert trigger joined to
  own sensors, not a trade), pytrends (gate-1 stability test may kill
  it — a finding, not a failure).
- Credentials: FRED free-key steps filed in wishlist (only key needed
  in the whole batch); everything else keyless; patents stays blocked
  on USPTO ID.me.
- Next session builds #2 (13F clusters) end-to-end per the standard
  doctrine: fetcher/parser/archiver + envelope manifest + registry
  where geographic.

## 2026-07-05 — [PRODUCT] Hero refinements: dominant globe, waitlist right, REAL self-updating stats (v1.0.87)

- Territory: T-CLIENT (+ one datacore server module). Hero-refinements
  directive, scope held to #data-intel.
- REAL STATS root-caused and fixed: prod archiveStats() returns
  files/bytes — the hero was summing a `samples` field that only the
  FIXTURE had, hence the production dash; and it only covers position
  kinds, undercounting streams. New /api/data/platform/stats
  (server/platformStats.ts): layers from the live registry;
  streams_recording = live layers + archive dirs not mapped to a layer
  (DIR_LAYER_MAP for filings→insider, earnings8k→earnings; waitlist/
  apiusage operational dirs and *_tracks excluded); observations =
  REAL line counts across the archive, gz-aware streaming, 10-min TTL
  cache with stale-beats-recount. Nothing hardcoded — every number
  grows as the system grows. 3 new tests incl. the phantom-samples
  regression pin.
- GLOBE prominence: zoom 1.05→1.45 (1.15 phone), brighter land
  (#1b3560/#3a67a6), 20° graticule (mission-control read), points
  2.1px aircraft ≤1200 / 1.9px vessels ≤800 with a GRACEFUL PHONE
  DENSITY CAP (500/300 under 640px — cap density, never stutter),
  glowing 4px site markers, legibility shade lightened (right edge
  0.25→0.04) so the sphere is the centerpiece not a curtain.
- LAYOUT: waitlist moved to a bordered card on the RIGHT of the hero
  row; "Open the live map" button REMOVED per directive (its CSS block
  replaced, not orphaned — dead-code policy). Headline, positioning
  copy, and the imagery strip untouched.
- CONCURRENCY NOTE: routine merged #191 (CFTC COT pipeline, v1.0.86)
  mid-build — stream (2) of the data-expansion directive is theirs;
  read-and-increment took this PR to v1.0.87, no collision.
- Gates: node 130/130; harness 0 hard failures ×3 pages ×3 widths +
  all-off; globe probed under software-GL at 390/1440 (sphere with
  graticule/land/points dominant, text legible); screenshots reviewed.

## 2026-07-04 — [REPAIR] /api/diag shipped: token-gated read-only diagnostics (option d, human-approved) (v1.0.81)

- Territory: T-BOT (server/bot.ts route + pure server/diag.ts). Closes
  the session-self-diagnosis gap: KNOWN BROKEN #3/#4 verification
  (fills firing? feedback accumulating? retrain green?) no longer
  needs the human to paste JSON.
- Shape per the approved option (d): GET /api/diag/:probe, HARD
  WHITELIST {audit tail (time/type/message only), ml status
  (model age + fills/feedback counts), daemon health, positions
  SUMMARY (counts + gross/net exposure — never symbols)}. Closed by
  default: no DIAG_TOKEN or <24 chars ⇒ 404; timing-safe token
  compare; every response passes sanitizeDiag (key-like strings, long
  hex/base64, emails → [redacted]) as defense-in-depth over the
  whitelist shaping. auth.ts (frozen) untouched — pinned by test.
- Token generated and handed to the human (Railway + Claude Code
  session env, header x-diag-token). Test file uses a dummy — the real
  value never enters the repo.
- Gates: node 121/121 (4 new: closed-by-default + token check,
  summary-has-no-symbols, sanitizer A/B incl. survivor timestamps,
  wiring pin incl. auth.ts-untouched); build clean; tsc unchanged (61
  pre-existing). Verification plan: once DIAG_TOKEN is set in Railway,
  next session curls /api/diag/ml and logs the first live reading.

## 2026-07-04 — [PRODUCT] Decisions batch executed: monetization 2+3 delivered, options pilot decided, DIAG approved, CDSE/AIS verified (docs)

- Human decision batch (five items) recorded and executed:
  1. Monetization checklist items 2+3 APPROVED as pre-revenue prep —
     DELIVERED: datacore/LICENSING_AUDIT.md (per-source resell-vs-
     display register the API's LICENSE_MARKS derive from; re-verify
     every row at switch) + datacore/API_TERMS_DRAFT.md (customer ToS,
     explicitly DRAFT/not-in-effect). Items 1+4 wait for the charge
     decision.
  2. Options data: FREE Databento pilot chosen (needs the human's
     account + DATABENTO_KEY in session env — steps delivered); free
     Alpaca daily chain archiver queued as its own [PIPELINE] PR —
     starts regardless of the pilot outcome.
  3. DIAG option (d) APPROVED: token generated + handed to the human
     (Railway + session env); /api/diag route ships as its own code PR
     with the sanitizer test.
  4. CDSE: creds set in Railway; NO code read any CDSE var yet —
     canonical names declared CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (S1
     pipeline will read exactly these); told the human sessions also
     need them in the Claude Code env; OAuth endpoint + STAC catalogue
     verified reachable from our egress. Item stays open until an
     authenticated S1 pull succeeds.
  5. AISSTREAM verified LIVE in prod (enabled:true, 1,838 vessels,
     registry "live") — the reported "awaiting key" was a pre-restart
     tab; wishlist entry closed with evidence.
- Bonus verification while probing: the v1.0.80 liveness field is live
  in production health (checks.bot.liveness {dark:false}, bot active).

## 2026-07-05 — [PRODUCT] Hero globe: live rotating 3D Earth behind #data-intel + real facility imagery strip (v1.0.85)

- Territory: T-CLIENT. Hero-globe directive executed within the stated
  scope (ONLY #data-intel; The Bot/pricing/docs untouched — harness
  data+developers shots unchanged).
- GLOBE: MapLibre globe projection (already the /data dependency —
  lazy chunk, no new library; globe.gl/three.js rejected as +600KB;
  Cesium rejected per directive). Land silhouettes from OUR
  self-hosted NE boundaries — ZERO external tiles, so the globe works
  in the harness and never depends on a CDN. REAL data on the sphere:
  live aircraft (blue, ≤400) + vessels (green, ≤300) from world-bbox
  fetches, strategic sites (amber, Cushing/ports/mills) from
  /api/data/sites. Slow auto-rotation (0.02°/frame) that PAUSES
  off-screen, on hidden tabs, and under prefers-reduced-motion;
  aircraft refresh 30s only while visible.
- PERFORMANCE/DEGRADATION: boot is IntersectionObserver-gated
  (rootMargin 400px — zero cost until scrolled near), WebGL-checked,
  interactive:false + pointer-events:none (can never hijack scroll);
  ~700 points vs the proven 10k budget. Any failure (no WebGL, chunk
  load, fetch) leaves the styled dark-space backdrop with the existing
  "Open the live map" CTA — DEVIATION NOTED HONESTLY: the directive's
  named fallback was a "pre-rendered rotating globe"; a CSS-faked
  rotation is itself the jank risk, so the fallback is clean-static +
  live link, per the never-janky principle that motivated the clause.
- IMAGERY STRIP (in-section): three REAL chips pulled by our own
  pipelines this session — Sentinel-1 SAR (Cushing tank farm,
  radar-bright metal), fresh Sentinel-2 true-color of the Cushing
  tank clusters, Sentinel-2 of the Baytown TX refinery complex; all
  honestly captioned + "contains modified Copernicus Sentinel data
  2026" credit; ~290KB total, lazy-loaded.
- Existing headline/copy/stats/CTA/waitlist untouched (text-shade
  gradient added behind them for legibility; heavier on phones —
  text wins over spectacle at 390).
- Gates: harness 0 hard failures ×3 widths ×3 pages + all-off; node
  127/127; globe probed rendering under software-GL at 390+1440
  (sphere, land, points visible; content legible on top); screenshots
  reviewed.

## 2026-07-05 — [PRODUCT] Landing page: DATA INTELLIGENCE section shipped — the oldest unexecuted directive closes (v1.0.84)

- Territory: T-CLIENT. Task #50 (three-part directive PART 3) finally
  executed. STRICTLY ADDITIVE as directed: git numstat on the three
  raw landing files shows insertions only (the two 1-line "deletions"
  are the no-newline-at-EOF artifact — final lines byte-identical).
  New section #data-intel between The Bot and Pricing, using the
  page's existing design system (section.s, prose-grid, accent).
- COPY PROVENANCE (honesty): the full approved copy block did NOT
  survive context compaction — verbatim survivors are the HEADLINE
  ("The physical economy, observed live.") and the atlas-Part-4
  POSITIONING line ("We are not a basemap competitor — ..."), both
  used verbatim. The one supporting paragraph is drafted new and
  FLAGGED in the PR for human review/replacement.
- GLOBE: the landing already ships a D3 canvas globe as its hero — the
  directive's globe requirement is satisfied by the existing one; a
  second globe would be redundant and janky-risk. The new copy points
  at it ("the globe above isn't decoration — it's the product").
- LIVE STATS (live map layers / data streams recording / archived
  observations) from the public /api/data/layers +
  /api/data/archive/stats endpoints; "Open the live map →" CTA to
  /app#/data; EMAIL-ONLY waitlist reusing POST /api/waitlist
  (source "landing"), explicit "no billing" copy — tripwire untouched.
- GRACEFUL DEGRADATION finding: the landing's script (incl. anything
  appended to it) is D3-CDN-gated — landing.tsx returns early on CDN
  failure. The section's wiring therefore lives in landing.tsx itself
  (React side), so stats degrade to em-dash placeholders and the form
  still posts even with the CDN dark. Probed both states headless.
- LANDING NOW UNDER HARNESS: "/" added to visual_check PAGES (it was
  never tested before) — layout/touch/overflow checks ×3 widths ride
  every future client PR. Gates: harness 0 hard failures across
  data+developers+landing+all-off; node 127/127; screenshots reviewed
  (390 + 1440, section + populated stats).

## 2026-07-05 — [RESEARCH] Databento quality VALIDATED (~$0.30) + CDSE Sentinel-1 verified end-to-end (docs)

- DATABENTO (approved validation stage): 9 stratified days across
  2016–2017 incl. the selloff/Brexit/election event days, 10-name mix,
  ~840k closing-window quote rows. Zero crossed quotes anywhere;
  spreads widen on event days exactly as real markets do (median
  2.8–5.6%, event p90 40–47%); put-call parity implies SPY 216.50 on
  election day = the actual close. VERDICT: quality validated; the
  ~$600 full-history go is now purely a budget call, with ONE
  engineering prerequisite filed — durable storage for the ~5GB slice
  (sessions are ephemeral; deliberately sampled instead of burning
  credits into a disposable container). Details in the wishlist entry.
- CDSE (item 3 of the directive): OAuth token issued with the real
  client credentials; catalog search found a fresh S1D GRDH scene over
  Cushing; a REAL 256×256 Sentinel-1 VV chip of the tank farm pulled
  via the Sentinel Hub Process API (radar-bright tanks resolved) —
  the S1 leg of the fused-sensor engine is UNBLOCKED, and Process-API
  chip windowing is the right primitive (61KB vs 1.7GB products).
  Zipper bulk downloads 401 (audience) — recorded with fixes, not
  needed for the chip design. Wishlist item closed with evidence.
- ALSO VERIFIED this session: /api/diag live with DIAG_TOKEN (401
  bare, data with header; ml probe shows model 17.8h old, 500
  feedback records, fills_count 0 — that zero + a TIER3-DIAG
  "warning — 1 issues" line are follow-ups for a diagnostics session).

## 2026-07-05 — [PIPELINE] Options-chain daily archiver LIVE — forward history starts today (v1.0.83)

- Territory: T-DATACORE. Executes the human's "start now — every day
  not archiving is history permanently lost" mandate (options-data
  decision 2026-07-04). server/optionsChainArchive.ts, FIRMS-poller
  pattern: once per trading day after 16:15 ET (ET-aware, once-per-day
  claim persisted so restarts can't double-fire), ≤120 underlyings
  from the CSP-universe cache (spot prices ride along in the cache
  tuples — zero extra API calls for the strike band) + open-position
  symbols, per underlying the Alpaca v1beta1 snapshots endpoint
  (paginated ≤5 pages, 350ms politeness spacing), filtered to exp ≤60
  days and strikes ±20% of spot.
- FEED HONESTY: paper accounts serve feed=indicative — NOT NBBO. The
  label travels on EVERY record ("feed":"indicative"), in the manifest
  license line, and the URL pins feed=indicative. Databento cbbo-1m is
  the ground-truth complement (pilot verdict GO, 2026-07-04).
- Volume budget stated up front: ~3–5MB/day raw JSONL, gzipped by the
  existing archive compressor; envelope manifest
  datacore/manifests/optionchains.json (enforcement test green).
- Gates: node 127/127 (6 new tests: universe cap/dedup/spot-ridealong,
  OCC parse + DTE/band filters, indicative-label-never-dropped,
  day-file + once-per-ET-day scheduling incl. weekend/pre-close cases,
  pagination + HTTP-error surfacing, manifest + routes wiring pins);
  build clean; server-only.
- First real snapshot: next trading day's close (2026-07-06 Mon) after
  this deploys. Verify via /api/data/archive/stats gaining an
  optionchains kind, or /api/diag audit tail.

## 2026-07-04 — [RESEARCH] Databento pilot EXECUTED: options history pull priced at ~$740, verdict GO (docs; $0 spent)

- Human provided the API key; pilot ran same message via free
  metadata.get_cost calls. PRIOR (stated in the wishlist entry before
  pricing): viable if the full pull quotes under ~$1,500. RESULT:
  cbbo-1m confirmed to 2013-04-01 from the API; measured closing-1-min
  per-day costs SPY $0.0129 / AAPL $0.0035 / F $0.0019; batching
  cost-neutral. Universe estimate ~$0.28/day → 2016→present ≈ $740,
  2013→present ≈ $930. Under budget even at 2x error → GO.
- Staged execution filed: $125 free credits cover a 2016–2017
  validation pull ($0 out of pocket); spend beyond credits waits for
  the human's go after quality validation. Env dependency: session
  environment still lacks DATABENTO_KEY (presence-checked this
  session: DATABENTO_KEY, CDSE_CLIENT_ID, CDSE_CLIENT_SECRET,
  DIAG_TOKEN all missing) — human walked through the setting.
- Two-sided honesty note: statistics schema (OI/settlement) priced
  10-100x the quote slices — sampled, never bulk-pulled.

## 2026-07-04 — [RULE-REVIEW] WORKSTREAM PARTITION amendment SHIPPED (docs)

- Human approved ("WORKSTREAM PARTITION amendment — ship it"). Applied
  to CLAUDE.md as a new section after SESSION BUDGET exactly as
  proposed, dated 2026-07-04: T-DATACORE / T-CLIENT / T-BOT
  territories, SHARED serialize-and-minimize list, and the 6-point
  merge-order protocol (shared edits last, read-and-increment
  versioning, keep-both research merges, identity-gated monitors,
  primary-territory ownership, supersession). Sessions declare their
  territory in their first experiments entry from now on.

## 2026-07-04 — [RULE-REVIEW] ACTIVE ANGLE-HUNTING amendment SHIPPED (docs)

- Human approved ("ANGLE-HUNTING amendment — ship it"). Applied to
  STANDING BEHAVIORS exactly as proposed, dated 2026-07-04: every EDGE
  session not consumed by repair or a higher-priority queued item
  hunts novel angles (cross-connections, anomaly mining, foreign-field
  imports, second-order), with the freedom-plus-rigor discipline —
  every angle logged with its testable form and ladder path, priors
  first, multiplicity discounts, out-of-sample required.

## 2026-07-04 — [REPAIR] Liveness alarm RUNTIME half live: dark loop now degrades /api/health (v1.0.80)

- Completes approved Amendment 2: server/liveness.ts (pure,
  node:test-safe) computes NYSE-session overlap (weekdays 9:30–16:00
  ET, DST-correct via Intl; HOLIDAYS deliberately not excluded — the
  alarm errs loud and market_calendar.py stays the single holiday
  source of truth) and the dark verdict (>2 market hours OR >24h
  wall-clock since last seen active).
- Heartbeat persisted like equityPeak (volume + /tmp fallback) so
  DEPLOYS NEVER RESET THE DARKNESS CLOCK — the equityPeak lesson
  applied preemptively. Fresh installs seed now (no instant false
  alarm). Railway's healthcheck polling drives the assessment; a 60s
  interval touch keeps the stamp fresh regardless; zero disk churn
  while dark (same-object return skips writes).
- /api/health Check 5 now carries checks.bot.liveness {dark,
  marketHours, wallHours, detail} and sets overall status=degraded
  when dark → HTTP 503 → every DAILY routine's health read surfaces
  it top-of-report. This closes the exact gap of the incident: bot
  paused, health "ok", human found it on a dashboard.
- Regression tests (5): heartbeat transitions incl. restart-keeps-
  clock, intraday market-hours math, weekend-spans-no-session, the
  2h/24h thresholds A/B (3h dark, 1.5h not; weekend halt trips 24h
  before Monday, Saturday-night check stays quiet), and a bot.ts
  wiring pin (nextLiveness + loopDark + degraded consumption +
  approved constants 2/24).
- Gates: node 117/117; tsc error count identical to main (61 — all
  pre-existing); build clean; server-only (no harness required).

## 2026-07-04 — [RESEARCH] Satellite multi-sensor roadmap filed + angle-hunting amendment proposed (docs)

- Satellite directive filed as the major roadmap in open_questions.md:
  resolution reality stated first (10m = facility-scale change, never
  object counting; the wall is attacked with MORE SENSORS, not more
  optical); S1 SAR + S2 optical + Landsat thermal fusion with
  per-sensor licenses and revisit cadences; LIDAR = one-time
  calibration only; Phase 1 free fused change detection (Cushing tank
  farms vs EIA — extends the existing gate-1 pipeline; steel yards vs
  AISI; construction vs permits/news); Phase 2 paid counting
  wishlist-gated on Phase-1 validation + revenue; imagery-age
  indicator alongside; validation mandatory with priors, multiplicity
  discounts, out-of-sample.
- Wishlist: CDSE signup steps filed (one credential → S1+S2; S2
  already zero-credential; ASF/Earthdata as the S1 fallback to
  verify); Phase-2 paid imagery entries with build-first analysis and
  honest price notes.
- ACTIVE ANGLE-HUNTING amendment PROPOSED in wishlist (exact text,
  NOT self-applied): recurring EDGE-session mandate to generate novel
  hypotheses (cross-connections, anomaly mining, foreign-field
  imports, second-order) with freedom-plus-rigor discipline — every
  angle logged with its ladder path.

## 2026-07-04 — [REPAIR] Toggle state-desync root-caused two ways: remount delta bug (proven+fixed) + open-tab version skew (guarded) (v1.0.79)

- Directive: prod atlas toggles flip the pill but the label stays "off"
  and nothing renders. State model: pill = enabled[id] (local), label =
  runtime[id].status (set ONLY by each layer's effect), map = the
  effect's add/remove — three views that desync exactly when the
  EFFECT IS MISSING for a rendered row.
- NEW RATCHET first (repro before patch): harness TOGGLE-CONSISTENCY
  battery flips EVERY live registry layer (13) and asserts
  pill+label+map move together, both directions. Result: 13/13 CLEAN
  locally — the atlas desync does NOT reproduce in the current code.
  But the sweep EXPOSED a different real bug:
- BUG A (proven + fixed): live-points layers (aircraft/vessels) toggled
  off→on never re-render — teardown kept sinceRef's delta cursor, the
  refetch sent a stale ?since=, the server answered {unchanged:true},
  and the early return skipped addSource/addLayer entirely. In prod
  this self-heals on the next data tick (~10s aircraft) but is
  indefinite for static feeds and deterministic in the harness (the
  fields-on battery failed with "no aircraft rendered" + a -1 indexOf
  masquerading as an ordering violation). Fix: teardown deletes the
  cursor; the unchanged path also drops it when the source isn't
  mounted. A/B: battery red before, green after.
- BUG B (prod vector, guarded): PROD EVIDENCE — deployed bundle hash
  equals this session's local v1.0.78 build (index--vjwcN8a.js,
  Last-Modified 21:55:30Z), registry + /api/data/boundaries current,
  atlas effects present in every bundle ≥v1.0.74. So no single fresh
  deploy shows the symptom. The coherent vector: an OPEN TAB running an
  old bundle remounts the /data page (hash navigation) → registry
  re-fetches (fresh rows incl. atlas ids) → the old bundle has NO
  effects for those ids → pill flips, label stuck "off", no render.
  Structural guard shipped: /api/data/layers now carries
  server_version; the client compares its baked-in build version and
  (a) shows a "reload to enable the newest layers" notice on mismatch,
  (b) renders rows whose id the bundle has no wiring for
  (id ∉ LAYER_GROUP) as DISABLED with "reload to enable" — a
  functional-looking toggle for an unwired layer is now impossible.
- FALSIFIABILITY: if the desync recurs on a FRESH reload of v1.0.79+,
  the skew hypothesis is wrong — per loop-health rule 4 that recurrence
  becomes a root-cause-only session, no re-patching.
- LEGEND half of the directive: SUPERSEDED before it arrived — v1.0.78
  (#173) replaced the dot legend with registry-drawn symbols + the
  both-directions parity battery; prod had it deployed at 21:55:30Z.
  The "generic dots" observation predates that deploy/reload.
- Gates: node 112/112 (new pins: server_version, skew banner, unwired
  guard, cursor clear); harness 0 hard failures ×3 + developers +
  all-off; toggle battery 13/13; fields diffs 46–65.

## 2026-07-04 — [RULE-REVIEW] Amendment 4 SHIPPED: bloat consolidation (docs) — honest shortfall vs estimate

- Fix 4 applied: DEAD CODE POLICY + CONSTITUTIONAL HYGIENE + the AUDIT
  CYCLE paragraph merged into ONE "AUDITS & DEBT" section (2,171 →
  1,662); SESSION BUDGET 2,025 → 1,447; STANDING BEHAVIORS 4,169 →
  2,952 (history moved to experiments.md, every clause kept); EDGE
  DOCTRINE 4,369 → 3,867 (framing cut; every example name, precedent,
  and the full build-first ladder kept). NO rule lost force — each
  compressed section preserves every normative clause, approval date,
  and exception.
- HONEST SHORTFALL: the filed estimate promised ~27.3K; the file is
  31,062 bytes. Two reasons: (a) Amendments 1–3 ADDED ~1,700 chars of
  approved rule text (platform mission, liveness alarm, sovereignty);
  (b) the filed per-section targets (e.g. EDGE DOCTRINE −1,250)
  assumed cutting example detail that turned out to BE the rule's
  force (standing data-source names, precedents) — cutting further
  trades force for length, which the approval explicitly forbade. Net
  prose cut ~4.0K vs the filed ~4.3K; net file −630 from pre-repair
  because the new rules are net-additive. If a harder target matters
  more than the preserved wording, that is a human call — the next
  30d constitutional audit can propose deeper cuts item-by-item.

## 2026-07-04 — [RULE-REVIEW] Amendment 3 SHIPPED: human-sovereignty clause (docs)

- Fix 3 applied exactly as filed: the verbatim HUMAN SOVEREIGNTY
  paragraph placed FIRST inside AUTONOMY AUTHORIZATION, so the
  delegation is read as subordinate to the human before the delegation
  itself is read. No other rule touched.

## 2026-07-04 — [RULE-REVIEW] Amendment 2 SHIPPED: liveness alarm in Priority 1 (docs; runtime half queued as [REPAIR])

- Fix 2 constitution half applied exactly as filed: LIVENESS ALARM
  appended to GOAL Priority 1 — loop paused/halted/broker-unreadable
  for >2 market hours (or 24h wall-clock) = top-of-report alarm in
  every DAILY session + degraded /api/health.
- Runtime half ships next as its own [REPAIR] PR with a regression
  test: persist state.inactiveSince; /api/health Check 5
  (bot.ts:1049) degrades overall status past the thresholds — the
  hook-confirmed gap that let the loop sit paused unflagged.

## 2026-07-04 — [RULE-REVIEW] Amendment 1 SHIPPED: mission reconciled with the charters (docs)

- Human approved all four constitutional-repair amendments ("ship all
  four in order 1→2→3→4"). This PR applies Fix 1 exactly as filed:
  GOAL section replaced — intelligence-platform mission with the bot
  and API customers as the two first-class consumers; priority ORDER
  preserved; P1 gains "archives recording"; P2 gains ladder-validation
  before trust/trade/sale; P3 = GROW BOTH COMPOUNDING LINES with the
  tend-the-bot-vs-advance-the-platform weighing rule; honesty metric
  two-sided; anti-goals extended with never-sell-unvalidated.
- Wishlist entry annotated APPROVED. Fixes 2 (constitution sentence +
  separate runtime [REPAIR] PR), 3 (sovereignty), 4 (bloat) follow in
  order, each its own PR.

## 2026-07-04 — [PRODUCT] Legend v3: real registry symbols, grouped, collapsible, parity-enforced (v1.0.78)

- Legend directive executed. The old legend hand-duplicated three site
  icons as inline SVGs and showed color dots for everything else —
  exactly the divergence-by-construction the directive kills. New:
  mapIcons.ts exports iconDataURL(name,color) which rasterizes THE SAME
  ImageData registerIcons feeds maplibre (SDF tint emulated with
  source-in compositing, cached per name+color) — legend and map share
  one source of truth and cannot diverge.
- Structure: sections mirror the panel groups (Live Tracking /
  Facilities / Environmental / Fields), Title Case labels, entries
  render ONLY while their layer is on, whole block collapses as one
  unit (open desktop / collapsed phone by default; 44px toggle).
  Color-only chips (altitude tints, raster ramps) stay chips — they
  are color MEANINGS, not symbols.
- DESIGN.md rule added VERBATIM as approved: "Every map symbol ships
  with its legend entry in the same PR, drawn from the shared icon
  registry — a symbol on the map without a matching legend entry (or
  vice versa) is a failed build."
- HARNESS parity assertion, both directions, computed from the LIVE
  style (literal icon-image values + ["get",prop] resolved via
  querySourceFeatures) vs legend [data-vt-icon] DOM: (a) every drawn
  icon has an entry, (b) every entry names a registered icon, (c) no
  empty icon renders. Measured 7–8 icons in use / 16 entries per
  width. A/B-PROVEN: a planted bogus entry failed all three ways
  ("map draws 'vt-train' with NO legend entry", "legend claims
  'vt-bogus' but no such icon is registered", "empty icon render").
  Node pin: DESIGN.md rule text verbatim + iconDataURL usage + no
  hand-drawn SVG duplicates inside the legend.
- Harness note: the taller open legend pushed field-layer rows to the
  scroll edge at 1440 in the fields-on battery (Playwright
  actionability timeout) — battery now collapses the legend and
  center-scrolls rows before clicking. New standing artifact:
  data-legend-{w}.png screenshots (legend beside the live map).
- Gates: node 111/111; harness 0 hard failures ×3 + developers +
  all-off; screenshots reviewed (390px legend fully legible).

## 2026-07-04 — [PRODUCT] Positioning copy on /developers (atlas-parity Part 4) (v1.0.77)

- The directive's honest not-a-basemap framing added to the /developers
  hero: same open geospatial foundation as any Earth viewer, every
  source named, differentiation = live movement + entity fusion +
  market-validated signals + API access; explicit "no claim is made to
  proprietary imagery". Live-vs-coming honesty already per-endpoint on
  the page (meta.coming_gated).
- Pinned by test (waitlist.test.ts): "not a basemap competitor" +
  the no-proprietary-imagery disclaimer must stay on the page; the
  monetization-tripwire string pins are untouched.
- The landing-page additive section (task queue: approved copy from
  the three-part directive) will carry the same positioning when
  built — this PR covers the developer-facing half only.
- Gates: node 110/110; harness developers ×3 clean, 390px screenshot
  reviewed; python untouched.

## 2026-07-04 — [PRODUCT] Atlas parity layer 3: country borders (Natural Earth, self-hosted) live (v1.0.76)

- Third build from the ATLAS PARITY filing: Natural Earth 1:110m
  admin-0 compiled into datacore (254KB slim — properties stripped to
  name+iso3, 177 countries) and served by OUR OWN /api/data/boundaries
  route, day-cached. PUBLIC DOMAIN: zero external dependency, zero
  license constraint on resale (GADM was rejected in the filing for
  its non-commercial clause).
- Base panel group, off by default, fetched ONLY on enable (zero-cost-
  when-off); line layer above rasters / below all data symbols; count
  shown as 177 features; HONESTY note: "1:110m generalized — reference,
  not survey-grade" (+ de-facto boundary policy stated in the registry).
- Gates: node 109/109 (new pin: NE attribution + public-domain wording
  + generalized-resolution honesty); harness 0 hard failures ×3 widths
  + developers + all-off (fixture route added); python untouched.
- ATLAS PARITY buildable trio now COMPLETE (water v1.0.74, forest
  v1.0.75, borders v1.0.76). Remaining are the blocked pair (WorldCover
  WMTS prod-verify; GHSL/WorldPop endpoint research) + positioning
  copy + USDA CDL from the Tier-1 register.

## 2026-07-04 — [PRODUCT] Atlas parity layer 2: forest cover 2020 (JRC GFC2020 via GFW) live (v1.0.75)

- Second build from the ATLAS PARITY filing, same shape as layer 1:
  RAW, Environmental group, off by default, field:true opacity slider,
  legend entry gated on enable, STATIC 2020 vintage stated in registry
  + status note (imagery-date rule). Tiles direct from the GFW public
  tile API (jrc_global_forest_cover/latest/dynamic) — zero server
  cost, zero key; CC BY 4.0 with EC JRC attribution, GFW named as the
  tile service. Pixels verified pre-build in the #167 filing (24,850
  non-transparent px on the z4 probe).
- Gates: node 108/108 (new registry pin: JRC attribution + 2020
  vintage + field flag); harness 0 hard failures ×3 widths +
  developers + all-off; python untouched.
- Remaining from the filed order: NE admin boundaries
  (datacore-compiled vector), then the blocked pair (WorldCover WMTS
  prod-verify; GHSL/WorldPop endpoint research).

## 2026-07-04 — [RULE-REVIEW] Constitutional repair: 4 amendment proposals filed (human-directed; NOTHING self-applied)

- Human directive ran the CONSTITUTIONAL HYGIENE process out of cycle
  (register updated). Four proposals filed in wishlist.md with exact
  text / placement / counts, awaiting item-by-item approval; ship
  order after approval: 1 → 2 → 3 → 4, each its own docs PR.
  1. MISSION RECONCILIATION — GOAL still names the paper account as
     the whole mission while VISION.md/GIP.md define the intelligence
     platform with the bot as one consumer (a live contradiction at
     the top of the constitution). Full replacement GOAL text drafted:
     platform mission, both compounding lines first-class in P3,
     two-sided honesty metric, anti-goals extended with "never sell or
     surface an unvalidated signal". Priority ORDER preserved.
  2. LIVENESS ALARM — proposed N = 2 market hours (+24h wall-clock
     ceiling); HOOK CONFIRMED: /api/health Check 5 (bot.ts:1049)
     already reads killed/active/stopped but never degrades overall
     status — the exact gap that let the loop sit paused unflagged;
     Check 6 (licensing) is the degrade precedent to mirror. Runtime
     half specced as its own [REPAIR] PR with regression test.
  3. SOVEREIGNTY CLAUSE — verbatim human text; placement: first
     paragraph inside AUTONOMY AUTHORIZATION.
  4. BLOAT — measured by section (31,694 bytes total): STANDING
     BEHAVIORS 4,169 (−1,750 of history/narrative), EDGE DOCTRINE
     4,369 (−1,250 of restated precedent), three audit rules in three
     places 2,171 (merge to one AUDITS & DEBT section, −770), SESSION
     BUDGET 2,025 (−520). Net target ~27.3K including Fixes 1–3
     additions; NO rule loses force — only words.
- This session may not self-apply any of it (amendments); the audit's
  only self-applied artifact is the register timestamp + this entry.

## 2026-07-04 — [PRODUCT] Atlas parity layer 1: surface water (JRC GSW v2021) live (v1.0.74)

- First build from the ATLAS PARITY filing: JRC Global Surface Water
  occurrence tiles as a RAW layer — Environmental group, off by
  default, field:true (inherits the v1.0.72 opacity slider at 60%),
  legend ramp (rare→seasonal→permanent) gated on the layer being on.
- Zero server cost: tiles direct from the JRC public bucket (like Esri
  imagery) — no proxy, no key, no Railway budget. Zero-cost-when-off
  by the same lazy effect pattern as terrain (source+layer added on
  enable, removed on disable); all-off harness run stays green.
- HONESTY: status note and registry description state the STATIC
  1984–2021 vintage (imagery-date rule) — this shows where water HAS
  occurred, not live conditions. Attribution EC JRC/Google on-map.
- Tile pixels verified server-side pre-build (open_questions ATLAS
  PARITY: 41,840 non-transparent px on the z4 occurrence probe);
  harness can't render external tiles (aborted for determinism) so the
  in-map screenshot proof is panel/registry/self-see — same evidence
  class as terrain/imagery shipped with.
- Gates: node 107/107 (new registry pin: JRC attribution + vintage
  wording + field flag); harness 0 hard failures ×3 widths + all-off;
  python untouched (green per v1.0.73 repair).
- Next in the filed order: forest cover 2020 (JRC GFC2020 via GFW,
  verified), then NE admin boundaries (datacore-compiled).

## 2026-07-04 — [RESEARCH] Atlas parity filed: free-layer endpoints verified, licenses checked, build order set (docs)

- Geospatial-parity directive Part 1/2 filing (open_questions.md ATLAS
  PARITY section). Endpoint verification done SERVER-SIDE with pixel
  decodes (DESIGN.md tile rule — never HTTP 200s): JRC Global Surface
  Water tiles VERIFIED (65k non-transparent px on the z4 probe), JRC
  Global Forest Cover 2020 10m via the GFW public tile API VERIFIED
  (24.8k px) — the directive's exact dataset; Natural Earth admin-0
  GeoJSON VERIFIED (public domain; GADM REJECTED — non-commercial
  license conflicts with the monetization path). WorldCover: COGs free
  on S3 (verified anonymous listing) but the Terrascope WMTS resets
  from this sandbox — prod-side verify or COG-pyramid fallback filed.
  GHSL/WorldPop: WMS endpoints moved/404 — endpoint research filed,
  licensing already clear (both CC BY 4.0). Elevation: already live
  (Mapterhorn = GLO-30). Dead ends recorded (Hansen GCS tile paths,
  umd_tree_cover_density_2020 "no latest version") so no session
  re-walks them.
- Part 2 BLOCKED-BY-ACCESS boundary filed: Google's Street-View-derived
  professional layers have no free lawful raw material — not a build
  target; free building footprints (Microsoft ODbL / Google Open
  Buildings CC BY) remain the Tier-2 path.
- Parts 3/4: differentiation pointers mapped to existing queue items
  (timeline slider, Everything Graph card, ladder-gated signals,
  /api/v1); positioning copy queued as its own small [T-CLIENT] PR.
- Build order chosen: surface water → forest 2020 → NE boundaries,
  one layer per PR (X7 precedent), all registry-native with field:true
  opacity inheritance from v1.0.72.

## 2026-07-04 — [REPAIR] Local pytest gate repaired: collection breakers + stale pins (KNOWN BROKEN #6 RESOLVED, v1.0.73)

- The constitutional gate (`python3 -m pytest -q`, promotion rule 1) has
  been UNRUNNABLE since the repo import — every session either ran the
  CI 4-file whitelist or scoped around it (KNOWN BROKEN #6 filed it,
  hypothesizing network/keys dependence). Root-caused today; the
  hypothesis was WRONG on both counts:
  1. COLLECTION BREAKERS: two root-level standalone SCRIPTS wear test_
     prefixes. test_auto_discovery.py executes its full discovery
     protocol at import and sys.exit()s → pytest INTERNALERROR kills
     collection for the entire repo. test_full_system.py defines a
     module-level `def test(phase, name, fn)` helper that pytest
     collects and fails ("fixture 'phase' not found") — and its import
     alone costs 62s. Fix: conftest.py collect_ignore for both, with
     the policy documented; both remain runnable directly as scripts.
     No assertion was removed — neither file could execute under pytest
     at all.
  2. STALE PINS (7 failures, none a live bug, none network-dependent):
     (a) test_fixes_pr8 TestTrackFillValidation ×3 — tearDown os.rmdir
     failed because track_fill's atomic write leaves feedback.json.lock
     (fcntl thread-safety, in the code since import); tearDown now
     rmtrees. (b) TestOptionsSlotseparation ×3 — pinned tunable VALUES
     (MAX_POSITIONS==5, MAX_OPTIONS_POSITIONS==3) that dated code
     comments legitimately moved to 8/8 (SIZING-FIX 2026-04-22,
     ALPHA-TUNE 2026-04-21); re-anchored to the MECHANISM (separate
     caps exist structurally; full stock book consumes zero options
     slots) with arithmetic against the live constants — pinning
     tunables in tests contradicts RULE REVIEW's tuning authority.
     (c) TestFix8 ×1 — string pin "max_loss=contract.get" went stale
     when the flow moved through shared_max_loss; the mechanism is
     INTACT and improved (single AND multi-leg paths register the same
     max_loss); re-pinned BOTH hops (contract→shared, shared→register),
     stricter than before.
- RATCHET (loop-health rule 3): test_collection_health.py collects the
  whole repo in a subprocess and demands a clean exit — A/B-proven: with
  conftest.py removed it FAILS carrying the original SystemExit
  diagnostics; with it, green. Any future collection breaker fails the
  gate the day it lands.
- Gate after repair: 311 passed, 1 skipped, ~8s (was: INTERNALERROR; or
  with the breaker excluded, 7 failed + 1 error in 74s — 62s of that was
  test_full_system's import). Count reconciled exactly: −1 full_system
  error entry, +4 test_voltrade_daemon (#164, merged after baseline),
  +1 ratchet.
- CORRECTION to today's v1.0.72 entry (learning-integrity): it
  attributed the gate breakage to "routine commit 2479df0 added
  test_auto_discovery.py" — WRONG. 2479df0 is the repo's INITIAL IMPORT
  commit (74k-line squash, authored 2026-04-23); the breakage is
  pre-existing and was already filed as KNOWN BROKEN #6. No routine
  broke the gate today. The v1.0.72 PR body carries the same error;
  corrected here, append-only.

## 2026-07-04 — [PRODUCT] Weather layer upgrade: opacity sliders, wind arrows, temp labels + scale (v1.0.72)

- Directive: make the now-rendering temp/wind fields usable intelligence
  tools — per-layer opacity control (default ~60% so the base map stays
  visible), aviation-style wind vectors rendered HONESTLY at the data's
  real density, temperature value labels (°F/°C) + color-scale legend.
  All RAW display enhancements — no interpretation, no gating needed.
- HONEST SOURCING (the load-bearing finding): OWM's free tile API is a
  raster COLOR FIELD with no vector data in it. Direction/speed/temp
  numbers exist only in the free current-weather POINT API. So vectors
  and labels come from a sampled point grid: ≤40 points per snapped
  viewport bucket (server/weatherGrid.ts), 10-min shared cache, 45/min
  upstream guard under the 60/min free budget. The UI states the real
  spacing ("one observation per ~N km") and never renders arrows denser
  than the samples — no faked barb density. Barbs proper were rejected:
  at 40 points/viewport the pennant/half-tick grammar would imply
  station-level precision we don't have; arrow + kt text is the honest
  form. Static grid that refetches on pan (debounced 600ms), never an
  animation — phone budget over spectacle.
- Registry-native: layers.json field:true flags opt layers into the
  opacity slider (weather radar included); default 60%, sessionStorage-
  persisted, live setPaintProperty updates. Wind arrows an SDF icon
  (mapIcons registry) with OWM's FROM-direction converted (+180°) to
  pointing direction; temp labels precomputed per °F/°C unit; temp
  color ramp added to the legend labeled "approx — amplified for dark
  basemap" (the proxy amplification from v1.0.69 shifts hues).
- snapBbox bug caught by its own test: quantum derived from each
  viewport's raw span gave nearby viewports different buckets, defeating
  the shared cache. Fix: power-of-two quantum ladder + outward
  (floor/ceil) edge snapping so the bucket always covers the viewport.
- HARNESS fields-on battery (new, all 3 widths): toggles temp+wind as a
  user would, then asserts pixel-level rendering (canvas off/on mean
  diff ≥3; measured 46.8/53.6/56.3), aircraft still rendered with
  fields on, rasters BELOW symbols, 60% default applied, arrows placed
  from the sampled grid, and the v2.4 occlusion hit-test re-run with
  fields on. A/B-PROVEN against a real defect it caught during this
  session: enabling weather grew the attribution to 2 lines at 390px
  and covered the zoom-out button ("OCCLUDED by
  maplibregl-ctrl-attrib-inner"); fixed with an attribution max-width
  cap so it wraps inside the right column. Deterministic wx tile
  fixture stands in for the proxy's amplified OUTPUT (the amplification
  itself stays unit-tested against the real captured prod tile).
- Gates: node 106/106; harness green ×3 + developers + all-off
  (fields diffs above); python vs PRISTINE origin/main baseline —
  identical 7 failed + 1 error on main worktree and on my tree, i.e.
  ZERO new failures from this change. Main's local pytest gate is
  BROKEN independently of U1: routine commit 2479df0 added root-level
  test_auto_discovery.py which pytest collects and which sys.exit()s at
  import (INTERNALERROR, kills the whole run; CI stays green only
  because ci.yml whitelists 4 files), and 7 options/track-fill tests +
  test_full_system.py now fail on pristine main. Filed as the next
  [REPAIR] action — not bundled here per one-change-per-PR.
- Live expectation: default-on look unchanged (fields stay off by
  default); when enabled, base map + live layers remain visible at 60%;
  arrow count per viewport ≤40 with kt labels; °F default matches US
  audience, °C one tap away.

## 2026-07-04 — [REPAIR] Daemon RPC route bug fixed (shadow_stats) + counterfactual-logging dead-config audit finding (v1.0.71)

- Session-start protocol followed in order: CLAUDE.md, experiments.md,
  open_questions.md, wishlist.md all read this session. Loop-health ratio
  over the last 10 entries (API product foundation back through KNOWN
  BROKEN #5): 4 REPAIR / 2 RESEARCH / 2 PRODUCT / 2 PIPELINE — well under
  the 7/10 REPAIR-thrash threshold, no meta-problem to address.
  `/api/health` on prod: all-ok (server/database/alpaca ACTIVE/python/
  licensing all "ok", bot active, equityPeak=108151.39, drawdownPct=0.0%
  — the persisted high-water mark still holding, memory nominal at
  163MB RSS). No live break visible from the public surface; deeper
  audit-log/trade_feedback access remains gated behind requireOwner per
  KNOWN BROKEN #4's unchanged ACCESS LIMITATION, so "fix a bug seen in
  audit logs" (SESSION BUDGET tier 1) was not directly actionable. No
  experiment has matured to a judgeable state this session (Insider
  Form-4, port-dwell, and shadow-fleet gate-2 work are all still
  accumulating history; Sentinel-2 explicitly deferred its next check to
  the June-reversal window, not yet reached).
- Per the KNOWN BROKEN #5 precedent (2026-07-04, same session-budget
  bind: no live audit access, nothing matured to judge), fell through to
  a READ-BEFORE-WRITE static audit as the next best "fix a bug" action —
  this time targeting `shadow_portfolio.py`, since CLAUDE.md's RULE
  REVIEW section names counterfactual logging as the standing evidence
  requirement for every open RULE COST AUDIT question, and no session's
  log had ever mentioned whether that infrastructure exists.
- PRIOR (REASONING STANDARD #10, stated before reading `shadow_portfolio.py`):
  expected counterfactual logging to be wholly unbuilt (open_questions.md's
  RULE COST AUDIT section is headed "after counterfactual logging
  exists," implying it doesn't yet).
- FINDING vs. prior — WRONG, in an interesting way: `shadow_portfolio.py`
  (240+ lines, thorough docstring) already implements almost exactly the
  CLAUDE.md RULE REVIEW spec — `log_candidate()` records
  {ticker, timestamp, score, decision, decision_reason, entry_price,
  regime, 34 ML features} for candidates, and `backfill_outcomes()` is
  wired into `server/bot.ts`'s Tier-1 daily cycle (10pm UTC, confirmed
  live in bot.ts:2717-2733) to fill in forward +5d/+10d/+20d hypothetical
  outcomes via real Alpaca bars, using PATH-DEPENDENT labeling that walks
  the bot's actual take-profit/stop-loss rules rather than close-only
  returns. This has apparently been running daily and accumulating data
  without ever being logged in research/ — a documentation gap, not a
  code gap. HOWEVER: `log_candidate()` is only actually CALLED from one
  place (`bot_engine.py` deep_score(), decision values `taken` /
  `rejected_score`) — the four other decision buckets its own docstring
  names (`rejected_heat`/`rejected_halt`/`rejected_earnings`/
  `rejected_other`) have ZERO call sites anywhere in the repo (grepped).
  So today the shadow archive can only ever answer the MIN_SCORE RULE
  COST AUDIT question, not the spread/correlation/regime/kill-switch
  ones — logged as the natural next PR (open_questions.md KNOWN BROKEN
  #10/RULE COST AUDIT update), not built this session (scope: this
  session's action is the audit + the one confirmed bug, not a new
  wiring project across bot_engine.py's many gate points).
- SECOND FINDING (the confirmed, fixed bug): while checking every
  consumer of `shadow_portfolio.py` per READ BEFORE WRITE, found
  `voltrade_daemon.py`'s RPC route table maps
  `"shadow_stats": ("shadow_portfolio", "get_stats")` — but the real
  function is `get_shadow_stats()`; `get_stats` does not exist. Any RPC
  call to `shadow_stats` would silently return a "Method not found"
  error at runtime. Confirmed via grep that nothing in `server/bot.ts`/
  `server/routes.ts` currently calls this RPC method (latent, not an
  active live break) — `backfill_outcomes` (the piece that actually
  writes data) is unaffected and confirmed working via its own bot.ts
  wiring. This is precisely the "Python signature change with an
  un-updated caller fails silently at runtime, not in CI" class READ
  BEFORE WRITE warns about — except here the caller-side name was wrong
  from the start, not a later rename.
- THIRD FINDING (surfaced investigating why the RULE COST AUDIT
  questions read as unanswerable): `system_config.py`'s `SCORE_BAND_MAX`,
  `MAX_CHANGE_PCT`, `SCORE_BAND_OPTIMAL_LO`, `SCORE_BAND_OPTIMAL_HI` are
  read NOWHERE outside `system_config.py` itself (grepped the entire
  repo) — dead config with comments that claim they gate trades
  ("Skip stocks already up/down 35%+", "Scores above this are often fake
  breakouts") when nothing in `bot_engine.py` enforces either as a hard
  block; `bot_engine.py` only applies a soft score PENALTY for extreme
  `change_pct`, never a skip, and never checks `combined_score` against
  `SCORE_BAND_MAX`/`SCORE_BAND_OPTIMAL_LO/HI` anywhere. Full detail,
  honesty-metric relevance, and the deliberate decision NOT to
  unilaterally wire a hard skip back in (that would be a rule/threshold
  CHANGE requiring RULE REVIEW's evidence-or-ablation gate, which
  neither exists nor can be quickly built — bot_backtest.py/backtest_v2.py
  model ETF rotation, not per-candidate stock selection) are in
  open_questions.md KNOWN BROKEN #10.
- FIX SHIPPED (one logical change): `voltrade_daemon.py`'s `shadow_stats`
  route corrected to `get_shadow_stats`. Regression test added FIRST per
  loop-health rule 3: new `test_voltrade_daemon.py` (no daemon test file
  existed before this PR) walks every route in `RPCDispatcher._routes`
  whose target module exists on disk and asserts the attribute resolves
  to a real callable — confirmed FAILING against the pre-fix code (2/4
  tests failed, pinpointing exactly `shadow_stats` -> `get_stats`), then
  confirmed PASSING (4/4) after the one-line fix. This ratchets against
  the entire class of bug (any future route rename), not just this
  instance — the two genuinely-placeholder routes (`ml_status_impl`/
  `ml_toggle_impl`, which have no corresponding .py file by design and
  fall back to local methods) are explicitly pinned as expected-absent
  so they're never silently miscounted as "checked."
- Verified: full offline CI-gate subset + the new file —
  `python3 -m pytest -q test_risk_controls.py test_audit_critical.py
  test_diagnostic_false_positives.py test_patches_verification.py
  test_voltrade_daemon.py` — 124 passed, 1 skipped (120 pre-existing + 4
  new, identical baseline otherwise; KNOWN BROKEN #6's full-repo
  collection issue is pre-existing and untouched). No `.ts`/`.tsx` files
  touched — Node test suite and the visual harness are out of this PR's
  scope (PROMOTION RULES rule 5, one logical change).
- Downstream chain (REASONING STANDARD #1): fixing the route ->
  `shadow_stats` becomes callable the moment any caller (a future
  dashboard, the still-pending DIAG_TOKEN route, or a CLI probe) wires
  it up -> that caller sees real win-rate-by-decision numbers instead of
  a silent error -> the KNOWN BROKEN #10 dead-config finding gives any
  future session the accurate mental model of which RULE COST AUDIT
  questions are actually answerable today (MIN_SCORE, once ~90 days of
  shadow history accumulate) vs. not (SCORE_BAND_MAX/MAX_CHANGE_PCT,
  which govern nothing yet). Zero live-trading-behavior change from this
  PR — nothing in `bot_engine.py`/`system_config.py`/`strategies/`
  changed, and the daemon route was never called by anything live.
  Version bumped 1.0.70 -> 1.0.71 (read-and-increment) per convention,
  though PROMOTION RULES rule 3's backtest requirement doesn't apply
  (no strategy/parameter change).
- STARVED: no — this session's scope (audit + confirmed-bug fix +
  honest documentation of the dead-config finding) shipped in full.
  High-value work remains queued: KNOWN BROKEN #3/#6, wiring the
  remaining `log_candidate()` decision buckets (spread/correlation/
  regime/kill-switch) so the RULE COST AUDIT's other questions become
  answerable, the SCORE_BAND_MAX/MAX_CHANGE_PCT evidence-or-retire
  decision once shadow history or an ablation harness exists, the
  GEOSPATIAL LICENSING REGISTER items (d)-(g), and the GIP BUILD QUEUE.

## 2026-07-04 — [PRODUCT] /developers page + waitlist + pricing designed-not-enabled (v1.0.71)

- API directive part 2: client/src/pages/developers.tsx renders the
  API reference FROM /api/v1/meta (self-documenting — the page cannot
  drift from the deployed API), a live archive-stats sample fetched by
  the page itself, license marks as they travel with responses, curl
  examples, honest coming_gated list, API pricing tiers marked
  "preview — not for sale yet" (numbers TBA; NO buy buttons, NO
  billing anywhere — tripwire test pins that stripe/checkout/billing
  strings cannot appear on the page), and the waitlist form (email
  only). server/waitlist.ts: validated, deduped (restart-surviving
  seed), day-JSONL PII stream with a manifest that states the
  handling contract (never exposed via API, never in git).
- TRIPWIRE: this session touched pricing DESIGN — the compliance
  re-run was executed and recorded in the MONETIZATION READINESS
  CHECKLIST (wishlist.md) before this shipped; billing remains dark.
- Harness upgraded for the multi-page era: PAGES now carry per-page
  config ({route, map}) — map/perf/self-see batteries run on map
  pages; layout + touch-target checks run everywhere. /developers
  green at 390/768/1440 (screenshots reviewed: theme tokens, live
  sample rendering, 44px targets). 4 new node tests (101/101).

## 2026-07-04 — [PRODUCT] API product foundation — /api/v1 over the archives, key scaffolding, metering, license marks (v1.0.70)

- API directive part 1 built pre-revenue, last mile explicitly gated:
  server/apiProduct.ts (pure module — no express/db imports per the
  runner-hang rule): env-seeded keys ONLY (API_PRODUCT_KEYS; no
  issuance flow by design — issuance binds to billing later),
  per-tier sliding-window rate limits (dev/pro/enterprise), usage
  metering to a NEW manifested stream (<archive>/apiusage/, sha256
  key prefixes — raw keys never logged; forward-enforcement test
  covers the manifest automatically), and LICENSE MARKS on every
  response: aircraft-derived = ODbL share-alike; AIS-derived =
  conditional (aisstream ToS re-read at switch); US-gov streams =
  public domain; OWM excluded from the API entirely (display-only
  product). /api/v1: meta (public docs), tracks/:kind/:id,
  stats/portdwell, stats/shadow, stats/archive — the product IS the
  archive, not a live-proxy duplicate. Gated items (tank-fill, entity
  timelines) appear only under coming_gated — meta honesty pinned by
  test.
- 6 new node tests (97/97): key parsing + closed-by-default, limiter
  window behavior, metering hash discipline, license-mark pins, meta
  honesty, route wiring/guard count. No UI in this PR (/developers is
  the next, T-CLIENT-territory change).

## 2026-07-04 — [RESEARCH] Throughput: workstream-partition amendment proposed + velocity metric + GIP queue territory-tagged (docs)

- Throughput directive executed as filed artifacts: (1) WORKSTREAM
  PARTITION amendment proposal in wishlist.md (exact CLAUDE.md text
  for approval — T-DATACORE / T-CLIENT / T-BOT territories, shared-
  file serialization, 6-point merge-order protocol; rationale: 40 PRs
  merged today across concurrent sessions with 4 recovered collisions
  — territories prevent the class). NOT applied — constitutional
  amendments await approval. (2) VELOCITY metric table in
  usage_log.md (PRs merged/day + queue-depth trend; seeded 32
  yesterday / 40 today). (3) GIP BUILD QUEUE in open_questions.md,
  territory-tagged: aircraft continuity spine [T-DATACORE], UI
  scalability + landing section [T-CLIENT], Sentinel-2 per-tank
  iteration [T-DATACORE], API foundation [SHARED→split]. Parallel-
  subagent use is already standing practice (two research workflows
  this session); codified in the proposal's last clause.
- MONETIZATION READINESS CHECKLIST filed in wishlist.md for approval
  (API-product directive): provider-compliance re-run executed for
  this directive (pricing-design touch trips the tripwire — chain is
  adsb.lol primary/lawful + two non-commercial fallbacks that must
  drop at switch), per-source resell-vs-display licensing audit
  drafted (ODbL share-alike marks on aircraft-derived endpoints; OWM
  excluded from API entirely; aisstream CONDITIONAL pending ToS
  re-read), ToS draft + Stripe wiring plan itemized. Rule restated:
  last mile waits for the human's item-by-item go.

## 2026-07-04 — [PRODUCT] Charters installed: verbatim VISION.md + GIP.md companion (GIP directive Parts 0a + 2)

- VISION.md: the human's verbatim charter text received and installed,
  replacing the labeled reconstruction exactly as its provenance
  banner promised; reconciliation annotations kept; a deltas section
  annotates items the verbatim text adds (tick/futures/crypto data,
  news velocity, web traffic, supply chain, embeddings/RL, five more
  dashboard ideas) with honest statuses.
- GIP.md: full verbatim "Expansion of the Global Intelligence
  Platform" text under its own headings + a session-maintained
  reconciliation annex citing real artifacts
  (SENTINEL2_CHANGE_SPEC.md, EVERYTHING_GRAPH.md, shadowFleet.ts,
  datacoreArchive.ts, the licensing register, the approved inference
  envelope). Honest BLOCKED marks per the directive: object/vehicle
  counting blocked at free 10m (change detection is the lawful free
  version); per-county permit verification is per-target, not global;
  bulk maintenance records and manifest-level cargo data have no free
  lawful source.
- Reading rule extended (human-approved in the directive): PRODUCT and
  EDGE sessions read BOTH charters after CLAUDE.md; usage_log.md
  carries the updated routine-prompt line (HUMAN ACTION to paste).
- Part 0c recorded: options data stays HOLD; the free options-chain
  archiver covers go-forward; revisit only when a gated strategy
  needs deep history.

## 2026-07-04 — [REPAIR] Temp/wind v1.0.69 fix VERIFIED — fields render vividly with real prod tiles at all three widths

- Deploy verified: prod wind tile went 0 → 37,811 strong-alpha pixels
  (the amplification is live). PROOF NOTE (environment honesty): this
  sandbox blocks ALL browser egress (even example.com resets — probed
  with and without the agent proxy), so the "screenshots on
  production" proof ran as a faithful MIRROR: 48 REAL production
  tiles (curl-fetched through the sanctioned proxy, amplification
  confirmed per-tile) rendered by the IDENTICAL client build (same
  merged commit as prod) at 390/768/1440. Result: mean off-vs-on
  pixel diff 156.9 / 151.4 / 157.6 (an invisible layer scores ~0.1;
  floor 3) — screenshots in .visual/prod-weather-{w}-{off,on}.png,
  reviewed: temperature gradients and wind structure clearly legible,
  © OpenWeatherMap attribution rendered, basemap survives through the
  230-alpha cap. Both-layers-stacked is the deliberately-tested worst
  case and is close to saturating — if the human prefers a lighter
  blend, raster-opacity 0.85 → ~0.7 is the single tunable, filed as a
  taste knob, not a defect. scripts/verify_weather_prod.mjs gained
  proxy passthrough for environments where browser egress works.
- Key validity + budget re-checked per the directive: prod status
  endpoint "ok"; the 60-calls/min free budget is shared-cache bounded
  (upstream calls = unique tiles per 10-min TTL; the 48-tile world
  sweep is the practical ceiling per window).

## 2026-07-04 — [PIPELINE] Sentinel-2 tank-fill gate-1 kickoff — PRIOR STATED BEFORE FIRST COMPARISON

- PRIOR (REASONING STANDARD #10 — this entry is written BEFORE running
  the backfill comparison, per the directive's explicit instruction):
  for the v1 facility-scale shadow index (dark-pixel fraction in the
  three verified Cushing tank-farm AOIs, scene-relative threshold,
  tan-zenith normalized) vs EIA weekly Cushing crude stocks over
  ~12-16 backfilled scene-weeks, I expect:
  (a) LEVELS correlation r ≈ +0.2 to +0.5 — weak-to-moderate positive.
      Mechanism: fuller tanks → higher floating roofs → SHALLOWER
      roof-well shadows → per-tank shadow area SHRINKS as stocks rise,
      which argues r NEGATIVE — BUT the v1 index is facility-scale
      dark-fraction, dominated by inter-tank ground shadows cast by
      tank WALLS (constant) and contaminated by ponds/asphalt, so the
      per-tank fill signal is a second-order modulation on a noisy
      base. Sign is genuinely uncertain at facility scale; magnitude
      |r| > 0.5 would surprise me.
  (b) WEEK-OVER-WEEK DELTAS r ≈ 0 (noise-dominated at v1 resolution —
      clouds, sun-angle residuals, tile registration).
  (c) KILL/ITERATE CRITERION: this v1 index is NOT expected to pass
      gate 1. Its job is to prove the pipeline (anonymous scene access
      → windowed reads → archived readings → EIA reconciliation runs
      end-to-end) and establish the noise floor. Gate 1 credit
      requires |r| ≥ 0.5 on levels over ≥12 weeks with a sign
      explainable by mechanism — anything less iterates toward
      per-tank annulus geometry (the spec's real design) rather than
      declaring victory or death on the facility-scale proxy.
- Pipeline facts (probed live before this entry): scene access needs
  ZERO credentials (Element84 earth-search STAC + AWS Open Data
  sentinel-cogs public S3; windowed B04 reads verified over the
  Enbridge AOI); EIA ground truth is keyless (public history XLS,
  current through 2026-06-26: 23.0M → 19.7M bbl June drawdown — a
  real live signal in the comparison window). CDSE credentials are
  NOT required; exact CDSE signup steps filed in wishlist.md as a
  fallback only (per the directive's request).
- RESULT (run AFTER the prior above; scripts/sentinel2_tankfill.py,
  36 readings, 12 scene-weeks matched to EIA weeks, 2026-03-19 →
  2026-06-27, archived in datacore/sentinel2/readings.jsonl):
  Pearson r LEVELS = -0.731; DELTAS = -0.225.
  VS PRIOR: sign matches the physical mechanism I named (fuller tanks
  → higher floating roofs → shallower roof-well shadows → less dark
  area) and the magnitude EXCEEDS my |r|<=0.5 expectation — a genuine
  surprise. Deltas ≈ weak, as predicted.
  HONEST VERDICT — GATE 1 NOT CLAIMED, despite technically meeting
  the |r|>=0.5 + mechanism-sign criterion I wrote above: both series
  are strongly TRENDED over this window (stocks fell near-monotonic
  31.5M→19M bbl; the index rose spring→summer, where imperfect
  sun-angle normalization plus surface seasonality push the same
  direction) — 12 points of trend-vs-trend inflate |r| regardless of
  mechanism; the weak deltas correlation is the tell that detrended
  signal is thin. STANDARD #4 applies: one window, one variant, no
  out-of-sample. WHAT WOULD EARN GATE 1: (a) the June stock REVERSAL
  (18957→19666 kbbl) extending into weeks where the index must turn
  DOWN against the seasonal sun trend — the natural experiment is
  already in motion; (b) per-tank annulus geometry (the spec's real
  design) replacing the facility-scale proxy; (c) >=20 weeks spanning
  at least one full reversal, levels AND deltas both mechanism-signed.
  Weekly readings continue via the archived script — every scene is
  now recorded (collect-everything).

## 2026-07-04 — [REPAIR] Temp/wind recurrence ROOT-CAUSED: OWM 1.0 tiles are intrinsically near-invisible on dark basemaps (v1.0.69)

- RECURRENCE (v2.4 touched this surface once — per loop-health rule 4,
  no re-patch: root-cause analysis). MEASURED on production tiles (six
  real tiles pixel-analyzed): temp_new = uniform 76/255 alpha,
  wind_new = 15-53/255 alpha, ZERO pixels above 120/255 in ANY tile —
  OWM Weather Maps 1.0 palettes are pale low-alpha overlays designed
  for LIGHT basemaps. Attenuation chain: intrinsic alpha (0.3/0.1) ×
  client raster-opacity (0.6) × dark satellite background = 3-18%
  effective visibility. "Not rendering" was rendering — invisibly.
- WHY BOTH PRIOR VERIFICATIONS MISSED IT (the actual generator):
  v1.0.63 verified HTTP 200 + content-type + byte size — never pixels;
  v2.4 fixed STATUS/note display — never pixels. Nothing ever asserted
  the layer's pixel CONTRIBUTION. Ratchet: DESIGN.md gains the
  tile-layer pixel-verification lesson;
  scripts/verify_weather_prod.mjs compiles the check (prod layer-off
  vs layer-on canvas screenshots, mean-pixel-diff floor, all three
  widths).
- FIX AT THE GENERATOR: GL raster paint can only reduce opacity below
  a texture's baked-in alpha — the client cannot fix this. The proxy
  we already own now amplifies each tile once per 10-min TTL (pngjs,
  pure JS): alpha ×3.2 (temp) / ×5.5 (wind, from its measured floor),
  capped at 230 so the basemap survives, +1.6× saturation around luma
  for the pale palette; fully-transparent pixels stay transparent (no
  field invented where none exists); transform fail-open (garbage →
  raw buffer served). Client raster-opacity 0.6 → 0.85 (mild blend
  now, not the visibility mechanism). TEST FIXTURE = a real captured
  production wind tile: must exhibit the defect before amplification
  (zero strong pixels — pinned) and read clearly after; alpha cap
  pinned.
- Verification: post-deploy, scripts/verify_weather_prod.mjs against
  voltradeai.com at 390/768/1440 — results land in this log.

## 2026-07-04 — [RESEARCH] USPTO ODP key path + keyless-bypass verdict (docs)

- Human hit a wall between the submitted ODP form and a key; research
  (primary sources + live HTTP probes) resolved it: the key is gated
  on completing ID.ME FIRST (MyUSPTO → Profile → Verify with ID.me),
  then self-serve at data.uspto.gov/apikey — no approval queue
  documented. Click-by-click filed in wishlist.md.
- Landscape finding worth the entry on its own: the ENTIRE keyless
  USPTO ecosystem died 2025-2026 — bulkdata.uspto.gov retired (host
  dead, probed), Developer Hub decommissioned, PatentsView API offline
  pending ODP relaunch with old keys incompatible, ODP web bulk
  directory account-gated since 2026-06-18, ODP API 401 without a key
  (probed). Only keyless start: Google Patents BigQuery — backfill
  only (repo archived read-only 2026-04-18, freshness unverified).
- Design consequences recorded in NEW DATA ROOTS #4: key-first
  pipeline, single-threaded (ODP burst=1, 429 ⇒ ~7-day lockout risk),
  quotas reset Sun 00:00 UTC, key is a per-person credential tied to
  the human's ID.me.

## 2026-07-04 — [REPAIR] FIRMS activation: env-name mismatch fixed (v1.0.68) + duplicate build superseded

- Human set the FIRMS key in Railway as FIRMS_MAP_KEY and asked "tell
  me if the code expects a different env var name" — it did: the
  routine's merged implementation (v1.0.65, server/nasaFirms.ts) reads
  NASA_FIRMS_MAP_KEY, so the layer would have sat awaiting_key forever
  despite the key existing. Fix: firmsKey() accepts BOTH names (code
  adapts to the action already taken; no Railway rename); the
  awaiting_key reason string names both; regression pinned by a test
  asserting FIRMS_MAP_KEY alone enables the module.
- DOUBLE-BUILD INCIDENT (concurrent-sessions gotcha recurrence,
  CLAIM-before-building rule): this interactive session built a
  parallel FIRMS implementation (own module/route/layer/tests) while
  the routine's version was already merged — discovered at
  cherry-pick time via the wishlist conflict. Per the supersession
  precedent the merged implementation stands; the duplicate was
  abandoned unmerged (PR #155 closed, never double-registered
  anything). Salvaged from the duplicate: the activation fix above
  and the invalid-key probe knowledge (FIRMS returns HTTP 400
  "Invalid MAP_KEY." — designed-status material if their impl ever
  needs it). Root cause of the recurrence: the interactive session
  reacted to the human's key message without re-checking wishlist
  claims first — the routine had marked SCAFFOLDED in the entry the
  session was about to edit. Lesson folded into the gotcha entry:
  CLAIM-check applies to human-triggered work too, not just roadmap
  picks.
- Verification plan: prod /api/data/fires probe post-deploy — expect
  enabled:true with detections (FIRMS keys activate immediately).

## 2026-07-04 — [PIPELINE] SEC 8-K Item 2.02 earnings-language pipeline — gate 1 (DATA) passed (v1.0.67)

- Session start: CLAUDE.md + all of research/ read this session (PRODUCT session mandate). Health check: prod `/api/health` was last verified all-ok earlier the same day (equityPeak persisted, drawdownPct 0.0% — see the entry below); no new critical break surfaced this session and KNOWN BROKEN #3/#4 remain blocked on the DIAG_TOKEN access decision (unchanged, not actionable without human approval) — a PRODUCT session does not preempt the DAILY routines' repair duty per the task mandate, so this did not block proceeding with product work.
- Chose the top-priority queued PRODUCT action per wishlist.md's GEOSPATIAL LICENSING REGISTER "NEXT ACTIONS" note ("...and NEW DATA ROOTS #1 (8-K language pipeline) as the top research build") and open_questions.md's NEW DATA ROOTS build-order rationale: 8-K language ranks #1 because EDGAR history already exists (gate 2 testable immediately, not time-blocked like the jobs/patents roots) with complete small/micro-cap coverage and exact timestamps.
- PRIOR (REASONING STANDARD #10, stated before writing any code): expected the getcurrent Atom feed to require opening every 8-K's own filed document just to learn which Items it covers (mirroring Form 4's feed, which needs a second per-filing fetch for owner data); expected exhibit discovery to need a fixed table-column assumption (Type always in the same position).
- FINDING vs prior: wrong on both counts, in the easier direction. SEC's getcurrent `<summary>` field already lists each filing's Item codes inline ("Item 2.02: Results of Operations and Financial Condition") — no extra fetch needed to filter for Item 2.02 before touching a single filing document. And the index.htm exhibit table's column ORDER varies between filer agents: UniFirst's row has "EX-99" in the Description column; MV Oil Trust's has "EXHIBIT 99.1" in Description with "EX-99.1" in Type — both live-fetched and confirmed 2026-07-04. `pickExhibit99Href` therefore matches on ROW CONTENT (any cell matching `/EX-?\s*99/i`) rather than a fixed column index, so both real formats resolve correctly.
- BUILT: `server/sec8kEarnings.ts` (no API key required, same fair-access terms edgarForm4.ts already relies on). Polls the public "getcurrent" 8-K feed, filters to Item 2.02, resolves each filing's Exhibit 99 press release, converts it to plain text via a dependency-free HTML-to-text pass (full numeric + named entity decode table, verified against real decimal entities `&#8211;`/`&#8220;`/`&#8226;`/`&#64;`/`&#38;`/`&mdash;`/`&rsquo;`/`&ldquo;`/`&rdquo;`). CAUGHT WHILE WRITING THE GATE-1 FIXTURE TEST: numeric `&#160;` decodes to a literal non-breaking-space character, not a regular space — left uncaught, every NBSP in a filed exhibit would have shipped as an invisible non-ASCII character in the archived/served text; fixed by normalizing NBSP to a regular space as part of `htmlToText`, verified by an explicit assertion in the test (`!text.includes(" ")`). Text is truncated at 30,000 chars with an honest `truncated`/`textLength` pair (the fact of truncation is never silently dropped). Filings with no EX-99-class exhibit are skipped, not fabricated with empty text — this is the concrete instance of the HONEST GAP already logged in wishlist.md's NEW DATA ROOTS #1 (Q&A sessions are almost never filed as an exhibit; a filing that announces results only in the 8-K body itself now simply contributes nothing to this feed).
- LADDER — gate 1 (DATA) PASSED: two real, live-fetched filings covering two distinct filer-agent HTML formats (UniFirst Corp Q3 FY2026 results, accession 0001628280-26-046349, Workiva-generated lowercase divs/fonts; MV Oil Trust final-distribution announcement, accession 0001104659-26-080431, classic uppercase P/TABLE/FONT with a data table), every extracted fact hand-verified against the actual rendered exhibit (dollar figures, dates, company names, decoded punctuation) — see `server/sec8kEarnings.test.ts`, 9 tests, all passing, including one end-to-end test through the real fetch layer via an injected fake fetch replaying the same real fixtures (no live network in the suite itself, same principle as edgarForm4.test.ts). Gate 2 (does guidance/results language predict forward returns vs. a size-matched random-entry base rate, per REASONING STANDARD #3) is UNSTARTED and unchanged from open_questions.md's existing hypothesis — ships today as a RAW-DATA overlay only (`/api/data/earnings-language`, `/api/data/earnings-language/history`), no predictive claim, per datacore/README.md's RAW-vs-SIGNAL rule.
- NO UI PAGE YET, DELIBERATELY: mirrors the edgarForm4.ts precedent exactly — the pipeline + API shipped alone in PR #118 (v1.0.47), and the full filings.tsx view followed later in PR #128 (v1.0.55) once the archive had accumulated history. Same sequencing here: a page built today would have nothing to show but a warming-up state. Queued as the natural next PRODUCT action once a few days of archive exist.
- COLLECT-EVERYTHING: archive under `<archive>/earnings8k/`, day-file JSONL deduped by accession, gzip after 2 days — identical shape to edgarForm4.ts's filings archive (discrete dated events, not continuous tracks). Every day not archived is unrecoverable: EDGAR's own full-text search index doesn't expose pre-parsed exhibit text, so this archive is the free BUILD-FIRST substitute (rule #2 — accumulation substitutes for purchase) for what a paid earnings-transcript-history vendor sells.
- Downstream chain traced (REASONING STANDARD #1): filed exhibit text -> archived from today forward -> eventually enough history to attempt gate 2 (guidance-language deltas vs. forward returns, regime-split per STANDARD #2) -> if it passes, a strategy-tournament entrant; if not, the archive still cost nothing extra to build and stands as free ground truth for other language-based hypotheses (job postings, patent filings) that need the same extraction machinery. Zero live-trading-behavior change from this PR — RAW display only, not read by `bot_engine.py` or any strategy, so no interaction with the GOAL priority-3/4 tradeoff to trace beyond that.
- Verified: full server test suite 86/86 passing (77 pre-existing + 9 new, `npx tsx --test server/*.test.ts`); `npx tsc --noEmit` shows zero new errors (61 pre-existing across other files, unrelated to this change, confirmed by name-filtering the output for `sec8kEarnings`/`routes.ts`). Python suite untouched — zero `.py` files modified this PR, one logical change (constitution rule 5).

## 2026-07-04 — [REPAIR] KNOWN BROKEN #5 audit: data modules confirmed wired; closed a real silent-failure blind spot in diagnostics.py (v1.0.66)

- Session start check: CLAUDE.md + all of research/ read this session.
  Loop-health ratio over the last 10 entries (NASA FIRMS back through Port
  Dwell): 1 REPAIR, 1 RESEARCH, 8 PRODUCT — well under the 7/10 thrash
  threshold, no meta-problem to address. `/api/health` on prod: all-ok
  (Alpaca ACTIVE, python bridge ok, bot active, equityPeak=108151.39,
  drawdownPct=0.0% — the persisted high-water mark still holding). No
  critical live break to fix. KNOWN BROKEN #4's ACCESS LIMITATION still
  stands (no DIAG_TOKEN route exists — see wishlist HOLD), so "fix a bug
  seen in audit logs" was not actionable; no experiment has matured to a
  judgeable state this session (Insider Form-4 gate 2, port-dwell gate 2,
  and shadow-fleet gate 1 are all still accumulating history). Per SESSION
  BUDGET, fell through to the next tier: start a new (small, evidence-
  gathering) action. Chose KNOWN BROKEN #5 over a fresh web-research
  fall-through because it is (a) squarely REPAIR MANDATE territory — a
  standing, unresolved constitutional TODO — and (b) fully resolvable via
  READ BEFORE WRITE static analysis alone, unlike #3 (CSP cascade), which
  needs live audit-log access this session doesn't have.
- PRIOR (REASONING STANDARD #10, stated before reading any call sites):
  expected to find at least one of `alt_data.py`/`social_data.py`/
  `institutional_data.py` genuinely orphaned (imported nowhere, or
  imported but never actually invoked) given how many alt-data modules
  this repo accumulated over time, and expected the audit itself to be
  the full session's output with nothing to build.
- FINDING: prior was WRONG on the orphan question — grepped every call
  site for `alt_data`, `macro_data`, `social_data`, `institutional_data`,
  `intelligence`, `finnhub_data`, `alphadesk`, `instrument_selector`,
  `diagnostics`, `tiered_strategy`, `analyze`. All are live-consumed:
  macro/alt/social/finnhub/intel are fetched in parallel inside
  `bot_engine.py:deep_score()` (lines 543-608) with every field read
  downstream into scoring (`macro.get(...)`, `intel.get(...)`, etc. —
  verified past line 609, not just imported-and-discarded);
  `institutional_data.py` feeds `insights.py`, wired to the site's
  `/api/insights/:ticker` route (a user-facing feature, correctly
  separate from the trading loop per GOAL priority 4, not a defect);
  `alphadesk/` wired via routes.ts; `instrument_selector.py` imported at
  bot_engine.py:3026 (note for future sessions: its `intelligence`
  parameter is a DIFFERENT, options-specific dataset from
  `intelligence.py`'s `get_full_intelligence` — same name, unrelated
  data, a naming collision that could mislead a future audit);
  `diagnostics.py` wired into `server/bot.ts`'s Tier-2 cycle (every 5th
  cycle) and its output actually sets `state.positionSizeMultiplier` /
  `state.minScoreThreshold` and can trigger a pause. Nothing in this
  KNOWN BROKEN item was dead code — CLOSED, no wire/retire work needed on
  that front. Prior vs actual, stated per REASONING STANDARD #10: I
  expected an orphan and found none; the correct update is to trust the
  evidence over the prior, not to manufacture a finding.
- REAL GAP FOUND (not what I went looking for, but what the audit
  surfaced): `bot_engine.py`'s five parallel data-source fetchers each
  wrap their call in a bare `except Exception: return {}` with **no
  logging anywhere** — a silent failure by design for graceful
  degradation, which is fine for the SCORE (missing data already degrades
  to neutral), but leaves ZERO trail that a source is down. Cross-checked
  against `diagnostics.py`'s existing API-health monitor
  (`run_diagnostics()` section 4, `api_checks`) and found it already
  tracks polygon/sec_edgar/wikipedia/gdelt/fred cache freshness — but had
  never been extended to cover `social_data.py` (reddit_/gtrends_/
  news_multi_ cache prefixes) or `finnhub_data.py` (fh_ prefix), the two
  sources added later than the original five. This is a live, unmonitored
  blind spot directly adjacent to KNOWN BROKEN #3 (CSP cascade) and the
  HONESTY METRIC: if either source silently died, live scoring would
  quietly run on 3-of-5 signal groups indefinitely with no audit-log
  trace and no session able to tell without re-doing this exact grep.
- FIX (one logical change, `diagnostics.py` only): added `extended_checks`
  (reddit_/fh_ cache-file presence) as a **separate, warnings-only
  bucket** in `run_diagnostics()` #4b — explicitly NOT merged into the
  existing `api_checks`/`failed_apis` list, which drives
  `reduce_position_size` at >=3 failures. Downstream chain (REASONING
  STANDARD #1, traced before writing the diff): merging the two new
  checks into `failed_apis` would silently change the count of monitored
  sources feeding an existing risk-affecting auto-fix -> a position-size
  cut could newly fire in situations that previously wouldn't have
  triggered it -> that is a threshold-behavior change, and RULE REVIEW
  requires evidence + one-at-a-time for exactly this class of change,
  which an audit-driven visibility fix does not carry. Keeping the new
  checks in their own warnings-only bucket means: `problems_summary` (and
  thus the `audit("DIAGNOSTIC", ...)` line bot.ts already logs every 5th
  Tier-2 cycle) will now surface "Extended data sources unavailable:
  [...]" if reddit/finnhub go dark, but `position_size_multiplier` and
  `should_pause` are mathematically untouched by this change — zero
  effect on live trading behavior, pure observability gain. FINNHUB_KEY
  unconfigured (empty or the shipped `YOUR_FINNHUB_KEY_HERE` placeholder)
  is treated as expected-degraded, not a break — mirrors the existing
  ml_model dynamic-criticality false-positive fix
  (test_diagnostic_false_positives.py already exists specifically to
  catch this class of bug).
- Regression tests FIRST (loop-health rule 3), added to
  `test_diagnostic_false_positives.py` (the existing, purpose-built home
  for this exact bug class) rather than a new file: 6 new tests —
  reddit+finnhub both flagged when down; no false-positive warning when
  FINNHUB_KEY is unset (with reddit cached); the shipped placeholder key
  treated as unconfigured; healthy when configured+cached; a
  source-inspection test pinning that `"reddit"`/`"finnhub"` never enter
  the `api_checks` dict literal (the isolation guarantee, verified by
  parsing `run_diagnostics`'s own source — this would have caught a
  future session accidentally merging the buckets); and a
  reduce_position_size isolation test. All 27 tests in the file pass;
  full CI-gate subset (`test_risk_controls.py test_audit_critical.py
  test_diagnostic_false_positives.py test_patches_verification.py`) —
  120 passed, 1 skipped, identical baseline to the pre-existing gate
  (KNOWN BROKEN #6, untouched by this PR).
- Verified: no other file touched (diagnostics.py + its test file only);
  no import cycle introduced (`os.environ` read directly, no new import
  of `finnhub_data`/`social_data` into `diagnostics.py`, avoiding any
  coupling to their heavier dependency surface).
- Version 1.0.65 -> 1.0.66 (read-and-increment). Rollback trigger: if the
  new "Extended data sources unavailable" warning fires persistently in
  production for a source that's actually healthy (a cache-prefix
  mismatch this session's static read missed), revert the `extended_checks`
  block — it is fully additive and isolated, so reverting restores
  exactly the pre-PR observability level with no other side effects.
- MARKET-HOURS NOTE: this session's directive stated the run occurs
  during market hours, so per instruction this PR is left UNMERGED and
  states explicitly that merge should wait until after 4:00 PM ET unless
  the change fixes a critical live break (it does not — pure
  observability addition, isolated from every auto-fix threshold, zero
  live-trading behavior change either way). Not self-merged this session
  regardless of the AUTONOMY AUTHORIZATION default, per the run's own
  instruction.
- STARVED: no — this session's scope (the KNOWN BROKEN #5 audit + the
  gap it surfaced) shipped in full. High-value work remains queued:
  KNOWN BROKEN #3 (CSP cascade, needs live audit-log access) and #6
  (pytest collection), the counterfactual logger, R2 maritime transit
  analytics, and the remaining GEOSPATIAL LICENSING REGISTER items.

## 2026-07-04 — [PRODUCT] NASA FIRMS active-fires layer scaffolded (v1.0.65)

- Session start check: read CLAUDE.md, all of research/, KNOWN BROKEN.
  Nothing there blocks product work: #3 (CSP cascade) and #5 (orphaned
  data modules) both need live-only diagnostics this session can't reach
  (KNOWN BROKEN #4's ACCESS LIMITATION, unchanged since 2026-07-04); #6
  (pytest collection) is a pre-existing, already-scoped-around gap. Per
  the task's own instruction ("product sessions do not preempt the DAILY
  routines' repair duty"), proceeded with product work. Loop-health ratio
  over the last 10 entries: well under the 7/10 REPAIR-thrash threshold
  (mostly PRODUCT/RESEARCH) — no meta-problem to address first.
- Chose the concrete next queued item from the GEOSPATIAL LICENSING
  REGISTER's explicit build-order list (open_questions.md): "(c) FIRMS
  fires — awaiting MAP_KEY human action, may ship scaffolded awaiting_key
  like vessels did, ARCHIVE detections from day one." (a) terrain and (b)
  weather were already shipped; no session had claimed (c) yet (checked
  for a [CLAIMED] tag first per the OPS GOTCHAS double-build rule — none
  found). This is squarely ladder-gate-1-adjacent PRODUCT work (a) from
  the task menu: a RAW-DATA overlay ships ungated per the RAW-vs-SIGNAL
  surface rule (as-is detections + attribution, zero predictive claim),
  and the licensing homework was already done in the 2026-07-04 register
  — no re-research needed, matched the build-order rationale exactly.
- BUILT: `server/nasaFirms.ts` — pure fetch/parse/archive/poll module,
  same shape as `edgarForm4.ts` (discrete dated events, not continuous
  tracks, so it reuses that module's day-file-JSONL-with-dedup archive
  pattern rather than `datacoreArchive.ts`'s adaptive-thinning position-
  track pattern). Key-gated exactly like `vesselStream.ts`
  (`NASA_FIRMS_MAP_KEY`) — `bootFirmsPoll()` no-ops entirely without a
  key, so there is zero upstream traffic or archive writes pre-key.
  `parseFirmsCsv` reads FIRMS' area-CSV by column NAME (not fixed index),
  so it handles both the VIIRS (`bright_ti4`, letter confidence l/n/h)
  and MODIS (`brightness`, numeric 0-100 confidence) column layouts
  without knowing in advance which source served a row — both classify
  through `classifyFirmsConfidence` into the same three-bucket scale.
  Dedup identity (`fireDetectionId`) is satellite+rounded-position+
  acquisition-timestamp, because FIRMS has no stable per-row id of its
  own and re-serves the same detections across overlapping day-range
  polls — verified by the archive test (identical set re-archived writes
  zero the second time).
- Wired in `server/routes.ts`: `/api/data/fires` (enabled:false + reason
  when no key, mirroring `/api/data/vessels`'s shape exactly so the
  client's existing awaiting-key handling needs no new cases) and the
  `/api/data/layers` dynamic-status mapping (fires goes `live` the moment
  the key exists, same as vessels).
- Client (`client/src/pages/datamap.tsx`): new "Environmental" panel
  group (collapsed by default, positioned after Facilities — this group
  now also holds the future R3 roadmap layers: USDA CDL crops, drought/
  soil moisture, USGS groundwater, per open_questions.md); a fires
  useEffect following the vessels awaiting-key pattern; a new `vt-fire`
  SDF icon (mapIcons.ts) tinted by confidence bucket
  (`FIRE_CONFIDENCE_COLOR`); detail card states the LANCE "not for
  safety-of-life use" disclaimer on every detection, not just the layer
  description (the licensing register's stated requirement).
- Downstream chain (REASONING STANDARD #1): key set on Railway -> next
  poll (<=30 min) populates the cache -> `/api/data/fires` flips
  enabled:true -> the layer's `awaiting_key` badge clears to `live` with
  no further code change -> every detection from that point forward is
  archived (no free history exists upstream, so this is the only
  archive-from-day-one window that will ever exist for this root) ->
  a future gate-1/gate-2 signal hypothesis (insurer/utility/timber
  exposure near sustained fire activity) has ground truth to validate
  against once enough history accumulates. Zero effect on the trading
  loop today — this module has no import path into `bot_engine.py`,
  `system_config.py`, `strategies/`, or `server/bot.ts` (SPINOUT-READY
  DATA LAYER boundary), and the layer defaults OFF (opt-in), so
  ZERO-COST-WHEN-OFF holds without any special-casing.
- Regression tests: `server/nasaFirms.test.ts`, 12 cases — VIIRS vs MODIS
  column-name parsing, confidence-scale classification (letter + three
  numeric bands + garbage-input default), dedup-id stability, the
  documented URL shape, key-gating (both `firmsEnabled` and
  `bootFirmsPoll`'s no-op-without-a-key path), a non-ok upstream response
  throwing (no silent empty result), and the archive/gzip/history
  round-trip through real temp-directory I/O (mirroring
  `edgarForm4.test.ts`'s dedup-across-poll-overlap test, with distinct
  synthetic lat/lon per test case — the archive's dedup set is module-
  level and content-keyed, so reusing identical detection content across
  test cases would falsely dedup across unrelated temp dirs, exactly the
  trap edgarForm4.test.ts avoids with unique accession numbers per case).
  `server/layersRegistry.test.ts`'s existing schema-invariant test covers
  the new `layers.json` entry automatically (kind/status/source/
  description all present). All 12 new + 65 pre-existing node tests pass
  (`npm run test:node`, 77/77); Python CI-gate suite untouched by this
  PR (no `.py` files touched) — not re-run, per CLAUDE.md's PROMOTION
  RULE 5 scoping (one logical change; this change has zero Python
  surface).
- Visual verification (PROMOTION RULES rule 6): `npm run build` clean;
  `node scripts/visual_check.mjs --page data` — 0 hard failures at
  390/768/1440 plus the zero-cost-when-off pass. Added the new `fires`
  layer + its `/api/data/fires` fixture to the harness's own FIXTURES
  (it was missing for `shadowstats` too, pre-existing gap, out of this
  PR's one-logical-change scope, not fixed here) specifically so the
  SELF-SEE check exercises the brand-new "Environmental" group's
  collapse/expand/reachability at all three widths — this is exactly the
  defect class the Map v2.4 PR fixed (a panel section existing in code
  but unreachable on screen), so proving it mechanically here rather
  than trusting the pattern by inspection. Screenshots reviewed: new
  "ENVIRONMENTAL 0/1 ON" group renders correctly, collapsed, between
  Facilities and Filings & Flows at 1440px; phone view unaffected
  (panel collapsed by default). Pre-existing warnings only (nav touch
  targets, "Signals — coming soon" clipped-below-fold — the check
  function's own comment says elements that scroll into view below the
  fold are expected to warn, not fail; SELF-SEE's reachability assertion
  for every registered layer, including fires, passed with 0 failures).
- No backtest required (PROMOTION RULES rule 3 scopes that to strategy/
  parameter changes) — this PR touches zero files under `bot_engine.py`,
  `system_config.py`, `strategies/`, or Python at all.
- Version 1.0.64 -> 1.0.65 (read-and-increment, checked against the
  OPS GOTCHAS collision history first).
- Merge timing: 2026-07-04 is a Saturday (confirmed via date computation)
  — markets closed all day, well outside the 9:30-16:00 ET deploy-
  coupling window. Safe to merge immediately.
- STARVED: no — this session's scope (ship the queued FIRMS layer)
  shipped in full: server module, tests, route, client layer, registry
  entry, harness fixture, and doc updates. High-value work remains
  queued: KNOWN BROKEN #3/#5/#6, the counterfactual logger, R2 maritime
  transit analytics, the remaining GEOSPATIAL LICENSING REGISTER items
  (d)-(g), and R5/R6 (Everything Graph, dashboards).

## 2026-07-04 — [RESEARCH] Dual-momentum SPY/QQQ judged out-of-sample — KILLED

- Session start check: `/api/health` all-ok (Alpaca ACTIVE, python bridge ok,
  bot active, equityPeak=108151.39/drawdownPct=0.0% — the 2026-07-03
  persistence fix still holding). Loop-health ratio over the last 10
  experiments.md entries: 1 REPAIR, 7 PRODUCT, 2 RESEARCH — below the 7/10
  thrash-escalation threshold. No DIAG_TOKEN route exists yet (still a
  wishlist HOLD pending human decision), so audit-log/trade_feedback
  inspection remains unavailable to autonomous sessions per KNOWN BROKEN #4's
  access limitation — SESSION BUDGET's "fix a bug seen in audit logs" tier
  was not actionable this session. Chose the next tier: judging a matured
  experiment. The Dual-momentum SPY/QQQ candidate (open_questions.md) had a
  PRIOR and an explicit kill rule recorded 2026-07-03, blocked only on the
  backtest engine (#1), which was rebuilt the same day — the out-of-sample
  test was runnable and simply hadn't been executed yet, buried under a run
  of [PRODUCT] map/geospatial sessions. Judging it outranks starting a new
  experiment or researching new ideas per the SESSION BUDGET order, and it
  is squarely GOAL priority 3 (grow the account) work using REASONING
  STANDARD #2/#4 rigor (regime-split, out-of-sample, discount for variants).
- PRIOR (restated from the 2026-07-03 entry, before running anything this
  session, REASONING STANDARD #10): "edge shrinks but survives ~+1% CAGR
  over SPY ex-2020-21; kill if negative in >=2 sub-periods."
- Built `bot_backtest_subperiods.py`: reuses `bot_backtest.py`'s existing
  pure `fetch()`/`backtest()`/`metrics()` (zero duplication) to split the
  same 2016-2026 SPY/QQQ window into four calendar sub-periods (2016-2019,
  2020-2021 isolated as a known outlier confound, 2022-2023, 2024-2026) and
  compare the dual-momentum (top_n=1, winner-take-all SPY-vs-QQQ, no regime
  filter) config against SPY buy-and-hold in each, applying the
  pre-committed kill rule mechanically (`judge()`) rather than eyeballing.
- RESULT: alpha vs SPY was 2016-2019 -1.09pp, 2020-2021 -13.24pp (excluded
  from the kill count per the prior's own "ex-2020-21" framing), 2022-2023
  +19.44pp, 2024-2026 -11.49pp. **2 of 3 counted sub-periods negative ->
  kill threshold met -> VERDICT: KILL.**
- Prior vs actual (REASONING STANDARD #10): prior expected the edge to
  *shrink but survive*; actual is a clean kill. The pooled 2016-2026 in-
  sample number (+2.2pp CAGR alpha) was almost entirely manufactured by the
  single 2022-2023 sub-period (+19.44pp) — a textbook instance of REASONING
  STANDARD #2 ("works overall often means works in the regime that
  dominated the sample"): 2022 was a rare year where a 2-asset SPY/QQQ
  winner-take-all rotation sidesteps a tech-specific drawdown by holding
  SPY; outside that one regime the strategy underperforms simple SPY
  buy-and-hold. This is exactly the failure mode REASONING STANDARD #4
  warns about in a 1-of-~7-variants-tried search.
- Disposition: NOT promoted to the future strategy tournament. Marked
  KILLED in open_questions.md with the full result and an explicit "do not
  re-propose this exact config" note — prevents a future session from
  re-discovering the same pooled-decade number and shipping it on the
  strength of a single dominant regime.
- Regression tests (new behavior, no existing tests to extend since
  `bot_backtest.py` itself has none — REASONING STANDARD-consistent, pure
  functions only, no network mocking needed): `test_bot_backtest_subperiods.py`,
  9 cases covering `split_dates` inclusive-boundary slicing, `slice_data`
  gap handling (must not synthesize missing dates), `run_subperiod`'s
  insufficient-history guard (must return `{}` rather than crash inside the
  252-day momentum lookback), and `judge()`'s kill-threshold arithmetic
  including the 2020-2021 exclusion and empty-sub-period skipping. All 9
  pass (`python3 -m pytest -q test_bot_backtest_subperiods.py`).
- Verified: full existing CI gate still green after adding the new files —
  `python3 -m pytest -q test_risk_controls.py test_audit_critical.py
  test_diagnostic_false_positives.py test_patches_verification.py` — 114
  passed, 1 skipped (identical to the pre-existing baseline; KNOWN BROKEN
  #6's full-repo-collection issue is pre-existing and untouched by this PR).
- No version bump: this is an offline research/judgment script (network
  fetch of SPY/QQQ closes + pure in-memory backtest), same class as the
  original `bot_backtest.py` — it imports nothing from and is imported by
  nothing in `bot_engine.py`/`system_config.py`/`strategies/`/`server/bot.ts`,
  so it cannot affect live trade attribution (PROMOTION RULES rule 4 exists
  to separate live-code changes' `code_version`, which doesn't apply here).
- Downstream chain (REASONING STANDARD #1): killing this candidate now ->
  it never enters the future strategy tournament on the strength of an
  overfit pooled number -> the tournament's baseline-vs-SPY comparisons stay
  honest (HONESTY METRIC) -> no wasted live-paper capital allocation cycles
  spent proving out a strategy that offline evidence already refutes. Zero
  trading-path impact today (nothing in `bot_engine.py`/`system_config.py`
  changed) — the only effect is closing an open research question with
  evidence instead of leaving it to decay as unexamined backlog.
- STARVED: no — this session's scope (judge the matured dual-momentum
  candidate) fully shipped. High-value work remains queued: KNOWN BROKEN
  #3/#5/#6, the counterfactual logger, Sentinel-2 gate 1, R2 maritime
  transit analytics, and the rest of the geospatial roadmap.

## 2026-07-04 — [REPAIR] Ops: wrong-merge reset emptied PR #148 — recovered, monitor pattern hardened (docs)

- Recurrence of the documented "verify WHICH PR merged" gotcha, now
  with a mechanical fix instead of a re-note (loop-health rule 4:
  recurrence escalates). Sequence: #147 (human-created from a branch
  commit) merged while #148 (v2.4) waited; the hash-only monitor
  fired; the reflexive reset force-push emptied #148's branch and
  GitHub auto-closed it. Recovery: cherry-pick from the local object
  store -> reopened as #149 -> merged clean, zero work lost. Ratchet:
  merge monitors now print the merged commit subject and gate "safe
  to reset" on it matching the expected PR (template proven live on
  #149's watch); OPS GOTCHAS entry strengthened from advice to rule.

## 2026-07-04 — [PRODUCT] Map v2.4 — three production defects fixed, each with A/B-proven enforcement (v1.0.64)

- (1) ETERNAL LOADING root cause: the OWM key WAS fine (activation
  verified on prod same day — see entry below) — the defect was
  client-side: statusFor DROPPED status notes on loading rows, so the
  designed "key activating — auto-retrying" note never rendered and
  the human saw a bare spinner for the whole ~2h activation window.
  Fix: loading rows render their notes; every status change is
  timestamped; a 10s-cadence watchdog upgrades any bare loading >30s
  to an explicit retrying note. DESIGN.md gains the approved
  loading-state rule verbatim. A/B PROOF: probe with a HANGING status
  endpoint — bare loading at t+1.5s, designed retrying note attached
  after crossing 30s (+scan cadence). Harness carries an armed
  assertion (any row loading >30s must have a covnote).
- (2) PERFORMANCE: zero-cost-when-off AUDIT result — all 13 layer
  effects already tear down before any fetch/interval when off; no
  violators found. The real load cost was seven default-ON layers
  fetching at mount. Fix: heavy default-on layers (powerplants,
  insider, shadowstats, portdwell, trains) mount DEFERRED after the
  map's first idle (4s failsafe); base map + aircraft + sites win the
  initial contention. NEW HARNESS STEP: all-layers-off run asserts
  ZERO layer-data API calls (mechanical zero-cost proof) + TTI budget
  2500ms — measured 852ms all-off vs 1016-1579ms with the default
  stack. DESIGN.md gains the zero-cost-when-off rule.
- (3) CONTROL OCCLUSION: zoom controls moved bottom-LEFT (the open
  panel AND the legend both live right-side — bottom-right zoom was
  under the LEGEND, a second occluder the directive didn't know
  about). Self-see now hit-tests zoom/fullscreen controls with the
  panel open. A/B PROOF: reverting to bottom-right fails the harness
  at ALL THREE widths with "map control OCCLUDED by <div
  class='vt-legend'>".
- Gates: node 65/65, python 114/1 skipped, harness green x3 + the new
  zero-cost step, screenshots reviewed (zoom bottom-left at 390).
  Version 1.0.63 -> 1.0.64.

## 2026-07-04 — [PRODUCT] OWM v1.0.63 VERIFIED LIVE on prod (follow-up to the entry below)

- Prod probe sequence: deploy live -> /api/data/weather/global/status
  = "activating" (fresh-key state, retry note surfaced, nothing
  marked broken) -> key ACTIVATED within the watch window -> status
  "ok" and a real temperature tile served through the proxy
  (200 image/png 78KB, /api/data/wxtile/temp_new/2/1/1). Global
  temp/wind fields are live end-to-end; the activation-aware flow
  behaved exactly as designed on a real fresh key.

## 2026-07-04 — [PRODUCT] OWM global temp/wind fields — key-proxied tiles, activation-aware status (v1.0.63)

- Human set OPENWEATHERMAP_KEY in Railway (fresh key, ~2h activation
  on OWM's side). Wired the Tier-1(b) global half: two RAW layers
  (Temperature/Wind, model-derived labeling pinned by test) served
  through OUR tile proxy /api/data/wxtile/... — key never reaches the
  client, and the shared TTL cache bounds upstream calls to
  unique-tiles-per-10min across ALL visitors (free tier is 60
  calls/min; client-direct tiles would blow it on one panning user).
  Zoom capped at 7 (fields are smooth; bounds the cache universe).
- FRESH-KEY RULE implemented exactly as directed: upstream 401/403
  classifies as "activating" -> layer shows LOADING with the note
  "key set — OpenWeatherMap activates fresh keys within ~2h;
  auto-retrying" and re-probes every 10 min; a 5-min negative cache
  stops us hammering OWM meanwhile. Never marked error for a
  fresh-key delay; the note itself says when to re-check the key.
- Tests (5 new): tile validation (allowlist/zoom ceiling/range/
  traversal-shaped input), URL builder key-encoding, the
  401->activating classification with ~2h note, TTL cache expiry +
  bounded eviction, wiring/registry pins (attribution + model-derived
  honesty). OWM's 401-for-inactive-key behavior confirmed live by
  curl (invalid-key probe -> 401), so the activating path is the
  real upstream behavior, not an assumption.
- Verification plan: prod probe of /api/data/weather/global/status
  after the deploy; "activating" expected if the key is under ~2h
  old — recorded as such, not as a failure.

## 2026-07-04 — [PRODUCT] Session close-out: charter + geospatial directive — 10 PRs, queue handed to routines. STARVED.

- Directive execution summary (#136-#145): approved consolidations
  applied (AUDIT CYCLE register + STANDING BEHAVIORS); VISION.md
  installed (labeled reconstruction — verbatim charter still needed
  from the human); wishlist decisions recorded (satellite-AIS
  declined, options HOLD package with verified prices, diagnostics
  EXPLAIN); five new data roots + geospatial licensing register +
  universal-envelope proposal filed (10-agent primary-source research
  pass); Everything Graph v1 spec (flagship, R5) + R6 dashboards;
  Tier-3 Sentinel-2 spec; BUILDS: port dwell analytics v1.0.60
  (directive's highest-value item), terrain hillshade v1.0.61,
  US weather radar v1.0.62.
- STARVED: yes — high-value work remains queued at close (Tier-1
  c-g, Tier-2 buildings, 8-K pipeline, options-chain archiver), all
  with licensing pre-cleared and next-actions filed for the
  [PRODUCT]/[PIPELINE] routines. This is capacity starvation, not
  queue exhaustion.
- Awaiting human: verbatim charter paste; universal-envelope
  approval; options purchase pick (or explicit "stay held"); DIAG
  route approval; four free key/signup actions (OWM, FIRMS, USPTO
  ID.me, Apple EPF); north-star line appended to B3/B4 routine
  prompts (usage_log.md has the exact text).

## 2026-07-04 — [PRODUCT] Tier-1(b): US weather radar layer — NOAA nowCOAST WMS (v1.0.62)

- Built against the licensing register: NOAA nowCOAST base-reflectivity
  mosaic (public domain, no key). COVERAGE HONESTY carried in the
  registry entry, status note, and a registry TEST: US + territories
  only — no free lawful global radar exists (RainViewer
  personal/educational-only + API gutted Jan 2026; Open-Meteo free
  tier non-commercial). Global temp/wind path = OpenWeatherMap free
  key (commercial-OK w/ attribution) — HUMAN ACTION filed; the OWM
  code path is NOT scaffolded (dead-code policy: build when the key
  exists). RAW, default OFF, rendered under all data layers, tiles
  refresh on a 5-min bucket via source.setTiles.
- VERIFICATION (browser egress died mid-session — ops gotcha below):
  GetCapabilities 200; the exact GetMap tile-template URL returns
  200 image/png 5.5KB via curl; CORS verified open
  (access-control-allow-origin: * with an Origin header present);
  in-map source spec printed by probe carries exactly that template;
  layer attaches beneath data layers. In-browser tile-load
  confirmation was NOT possible: headless Chromium lost ALL external
  egress mid-session (even Esri base tiles fail "Failed to fetch
  (0)"; curl fine both direct and via proxy; explicit
  --proxy-server no help; proxy status logs no failures — requests
  never leave the browser). The raster-tiles mechanism used is
  identical to the proven imagery base layer, so residual risk is
  low and confined to runtime tile delivery, which prod will show.
- OPS GOTCHA (avoid re-learning): headless-browser external egress in
  this remote env can die mid-session while curl keeps working; when
  a probe needs tile-load confirmation and the browser is dark,
  curl-verify the exact tile URL + CORS headers instead — that
  covers everything except in-browser compositing.
- Gates: node 60/60 (new registry pin: weather description MUST state
  the US-only limit), python 114/1 skipped, harness green x3 with
  self-see (weather row reachable). Version 1.0.61 -> 1.0.62.

## 2026-07-04 — [PRODUCT] Tier-1(a): terrain hillshade layer — Mapterhorn DEM (v1.0.61)

- First geospatial Tier-1 layer, built against the licensing register
  (Mapterhorn: commercial-OK, terrarium 512px z0-17, attribution via
  TileJSON — verified live this session, TileJSON declares encoding +
  © Mapterhorn). MapTiler free tier rejected per register
  (non-commercial). RAW layer, default OFF (imagery base already
  carries relief; hillshade is the opt-in accent), inserted beneath
  the lowest data layer (symbol/circle/LINE — first probe run caught
  hillshade sitting above the aircraft velocity-vector line layer;
  finder widened, re-verified).
- Also wires the raster-dem source that R4's 3D terrain toggle will
  reuse (setTerrain on the same source) — R4's terrain-source
  prerequisite is now done.
- New test file layersRegistry.test.ts: registry-wide invariants
  (every layer carries kind/status/source/description; SIGNAL layers
  may never be status live — the RAW-vs-SIGNAL rule is now
  machine-enforced) + Mapterhorn attribution pin. This ratchets ALL
  future layers, not just terrain.
- RENDER PROBE (new-layer lesson): toggled via the real panel switch,
  jumped to the Rockies at z9 — source attached, layer ordered under
  data layers, isSourceLoaded=true (DEM webp tiles actually fetched).
  Gates: node 59/59, python 114/1 skipped, harness green x3 with
  self-see. Version 1.0.60 -> 1.0.61.
- Queued follow-up (register note): archive a PMTiles extract of our
  AOIs for provider-disappearance resilience — accumulation
  substitutes for dependency.

## 2026-07-04 — [RESEARCH] Five new data roots filed + geospatial licensing register + universal-envelope proposal (docs PR)

- Charter gap execution items 2-3: a 10-agent research workflow
  (primary-source license verification — every verdict from the
  provider's own ToS/license page, fetched 2026-07-04; ~507k subagent
  tokens, 247 tool uses) produced: (1) five NEW DATA ROOTS entries in
  open_questions.md (8-K earnings language, ATS job postings,
  app-store rankings, USPTO patents, GitHub org activity), each with
  licensing verdicts, honest gaps, priors, and concrete ladder paths;
  build order stated with rationale (8-K first — EDGAR history exists,
  gate 2 not time-blocked). (2) GEOSPATIAL LICENSING REGISTER for
  Tier-1 layers a-g + Tier-2 buildings — notable findings: Open-Meteo
  and RainViewer free tiers unlawful for us; MapTiler free tier
  non-commercial (rejected); NO free current national US pipeline
  vector exists anymore (EIA Atlas dead, HIFLD Open discontinued
  2025) — layer (g) builds from GEM/TX-RRC/OSM with per-source
  coverage honesty. (3) UNIVERSAL ARCHIVE ENVELOPE proposal in
  wishlist (two-tier: dataset manifests retroactively + datum-level
  fields on new pipelines; avoids 3x-ing position-archive volume) —
  awaiting human approval. (4) Two free HUMAN ACTION items filed
  (USPTO ID.me key; Apple EPF enrollment) + Sensor Tower priced
  not-recommended entry with build-first analysis.
- Key licensing kills recorded so no session re-chases them: Motley
  Fool + Seeking Alpha transcripts (ToS bar pipelines), LinkedIn/
  Indeed scraping, Google Play programmatic charts, Appfigures free
  tier, USAJOBS derived use, Apple reviews RSS (verified dead).

## 2026-07-04 — [RESEARCH] Wishlist decisions recorded + options/diagnostics decision packages (docs PR)

- Human decisions 2026-07-04 recorded in wishlist.md: satellite-AIS
  DECLINED (entry retained; revisit trigger = a gated signal that
  specifically needs open-ocean coverage); historical options data
  HOLD (full decision package delivered: unlocks, dependencies,
  vendor prices verified same-day from vendor pages, ranked
  recommendation — Databento $125-credit pilot first, then
  historicaloptiondata.com L2 $1,495 one-off, then ThetaData Pro
  churn ~$160-320 if retention confirmed; free Alpaca chain archiving
  queued regardless); read-only diagnostics EXPLAIN delivered (four
  options risk-assessed; recommended = DIAG_TOKEN-gated whitelist
  route in routes.ts, auth.ts untouched, sanitizer test pinned;
  ships only on explicit approval).
- Sources: options prices from a 10-agent primary-source research
  workflow this session (vendor pricing pages fetched 2026-07-04);
  repo verified private (unauthenticated GitHub 404) for the
  snapshot-option risk assessment.

## 2026-07-04 — [PRODUCT] Tier-3 spec filed: Sentinel-2 facility change detection (docs PR)

- Geospatial directive Tier 3 executed as a spec (build queued for a
  [PIPELINE] session): datacore/SENTINEL2_CHANGE_SPEC.md — weekly
  facility-scale activity indices (yard occupancy, tank shadows, berth
  occupancy) from free Copernicus Sentinel-2 L2A via CDSE windowed COG
  reads; "activity index up/down at facility X," NEVER object counts
  (10m honesty). Generalizes the Cushing tank-shadow idea + the
  imagery-verified facility footprints into one system; port berth
  masks cross-verify the port-dwell AIS counts.
- Ladder paths per facility class stated (tank farms vs EIA Cushing
  weekly; steel yards vs STLD disclosed shipments; ports vs our own
  AIS counts). Nothing surfaces before gate 2 except RAW scene
  metadata (imagery-date honesty rule).
- Sub-meter paid imagery (actual counting) is EXPLICITLY GATED: may
  not enter wishlist with quotes until the free 10m version passes
  gate 2 on any facility class — priced classes noted in the spec
  (archive ~$10-25/km² minimums; monitoring $1000s/mo).
- Docs-only PR, no version bump.

## 2026-07-04 — [PRODUCT] Everything Graph v1 spec filed (flagship) + R5/R6 roadmap slots — docs PR

- Charter directive items 4-5 executed as design artifacts (no build):
  datacore/EVERYTHING_GRAPH.md specs entity types (company/person/
  facility/vessel; aircraft_operator PLANNED pending tail→operator
  gate 1), relationship types (insider_of, operates, located_at,
  calls_at from port-dwell visits), storage (v1 = pure builder +
  cache per the recompute-derivations doctrine; sqlite only past a
  stated size/latency trigger; never auth.ts's db), the entity_map
  resolution table (shared infrastructure with fusion (b) gate 1),
  and a 3-PR build order. Roadmap: R5 (graph, flagship) + R6
  (signal-strength / data-quality / pipeline-health dashboards, all
  sourced from monitoring we already emit) added to MAP V2 ROADMAP.
- Key design call, recorded: the graph is RAW (relationships as filed/
  registered, with provenance + confidence on every edge) — any
  interpretation on top is SIGNAL-class and ladder-gated; and v1 is a
  materialized view, not a store — losing it loses nothing
  (ARCHIVE-ingredients doctrine).
- Docs-only PR, no version bump.

## 2026-07-04 — [PRODUCT] Port dwell analytics from our own AIS archive (v1.0.60) — fusion directive, highest-immediate-value item

- PRIOR (REASONING STANDARD #10, stated before reading any archive
  output): with the archive only ~1 day old, expected near-zero
  completed calls and mostly in-port-now counts; the v1's value is the
  MACHINERY being correct from day one so history accrues into it —
  medians/anomalies become meaningful in 2-4 weeks. Expected design
  risks: (a) harbor craft polluting call counts (mitigated: median SOG
  <=3kts filter), (b) LA/LB fence overlap double-counting (mitigated:
  nearest-port assignment), (c) coverage gaps inflating dwell
  (mitigated: gaps SPLIT visits — lower-bound property, tested).
- Built: server/portDwell.ts (pure, baseDir-injectable, reuses
  shadowFleet.readVesselTracks); /api/data/portdwell (10-min cache);
  geofences = the 9 imagery-verified port terminals from
  datacore/sites (REFERENCE DATA ACCURACY: only verified coordinates
  may become geofences — these are the only port coordinates in the
  repo that qualify). UI: per-port text labels under port sites +
  panel row with per-port note ("X in port · med Yh"), RAW-labeled,
  filings-&-flows group. Anomaly flags 3x-median, suppressed <10
  completed calls/port.
- Tests: 7 new node tests (visit detection, right-censoring,
  nearest-port overlap, speed filter, anomaly threshold + thin-history
  suppression, gap-splits-visit lower-bound, wiring pins). Gates: node
  56/56, python 114 passed / 1 skipped, harness green 390/768/1440
  with self-see. NEW-LAYER RENDER PROBE (MapLibre silent-expression
  lesson): standalone playwright probe jumped to San Pedro Bay at z9
  and asserted queryRenderedFeatures on portdwell-labels — rendered=1
  with correct label text ("Los Angeles / 4 in port · med 22.5h");
  second label collision-hidden as designed (text-allow-overlap off).
- Downstream chain (#1): dwell stats surface -> port calls accumulate
  into distributions -> gate-2 test plan (open_questions.md PORT DWELL)
  becomes runnable in weeks; zero trading-path impact (datacore
  boundary: no bot_engine/system_config imports; worst failure mode is
  an empty stats endpoint).
- Backtest: not applicable (no trading logic changed); version bumped
  1.0.59 -> 1.0.60 for attribution.

## 2026-07-04 — [PRODUCT] VISION.md installed — platform charter (reconstruction), north-star reading rule

- Charter-installation directive executed. HONESTY NOTE: the human's
  verbatim charter text did not arrive (paste placeholder came through
  unfilled; confirmed absent from the session transcript) — VISION.md
  is a clearly-labeled session reconstruction from the directive's own
  enumeration, with a provenance banner asking the human to supply the
  original for verbatim replacement. Installing a labeled
  reconstruction now beats waiting: the north star exists for tonight's
  routines.
- Reconciliation annotations: every charter item marked
  DONE/IN-PROGRESS/QUEUED/NEW/BLOCKED-BY-ACCESS citing the existing
  mechanism (archive doctrine, ladder, DESIGN.md rules, roadmap slots).
  BLOCKED-BY-ACCESS register: card panels, private fleet telemetry,
  sub-meter counting, mid-ocean satellite AIS (declined), filed flight
  plans, US freight-rail positions.
- Reading rule: STANDING BEHAVIORS gains the VISION.md north-star line
  (approved 2026-07-04); KNOWN STATE carries the existence fact.
  Placement reasoning recorded in the rule text itself (directive said
  KNOWN STATE; same message approved the facts-vs-rules split — rule
  goes to STANDING BEHAVIORS, fact to KNOWN STATE).
- Routine prompts: B4/B3 canonical texts exist only in the routine
  platform — usage_log.md now carries the exact north-star line to
  append, flagged HUMAN ACTION NEEDED.
- Docs-only PR, no version bump.

## 2026-07-04 — [RULE-REVIEW] Approved consolidations applied: AUDIT CYCLE + STANDING BEHAVIORS (docs PR)

- Human approvals received 2026-07-04 for: (1) the AUDIT CYCLE
  consolidation proposal (wishlist.md, filed 2026-07-03) and (2) first
  constitutional audit Findings 1 (STANDING BEHAVIORS section) and 2
  (PERIODIC AUDITS register). Applied exactly as proposed: AUDIT CYCLE
  clause appended to SESSION BUDGET; this register created; DEAD CODE
  POLICY and CONSTITUTIONAL HYGIENE trigger sentences trimmed to point
  here (policy bodies untouched); KNOWN STATE's five standing-behavior
  rule paragraphs moved verbatim into a new STANDING BEHAVIORS section.
- F2 note: the AUDIT CYCLE register IS Finding 2's register in the
  concrete form the AUDIT CYCLE proposal specified ("supersedes the
  first audit's Finding-2 sketch") — one register, in experiments.md,
  not a second one in SESSION BUDGET. Shipping both literally would have
  created the exact redundancy the hygiene rule exists to kill.
- Docs-only PR: no behavior change, no version bump (per docs-PR
  precedent #132/#133), no backtest applicable.

## 2026-07-03 — [PIPELINE] Position archive — MAP V2 ROADMAP R1 (PR 2/3 of the Map v2 directive)

- Session start check: read CLAUDE.md, all of research/, per the [PRODUCT]
  session protocol. Loop-health ratio over the last 8 experiments.md
  entries: 4 REPAIR, 1 RESEARCH, 3 PRODUCT/PIPELINE — below the 7/10
  escalation threshold, no concern. KNOWN BROKEN #3/#4/#5 remain
  un-diagnosed (owner-gated diagnostics access, per wishlist) but per this
  session's brief, product work is not preempted by outstanding repairs
  that don't block it; none of them touch datacore/ or the /data surface.
- Chose this action over the other three options in the brief (new
  UI/enrichment work, a new-root proposal, or API/docs hardening) because
  the prior session's Map v2 directive (PR #105, docs-only "PR 1 of 3")
  explicitly named the position archive as the most time-sensitive R1 item
  — "every day not recorded is unrecoverable proprietary data" — and it is
  the load-bearing prerequisite for R2 (maritime transit analytics, "the
  strongest trading-signal candidate here") and three of the four
  ARCHIVE-ENABLED SIGNAL HYPOTHESES in open_questions.md. Every day this
  stays unbuilt is lost history those hypotheses can never recover.
- PRIOR (before writing any code, REASONING STANDARD #10): expected the
  main design risk to be disk growth on the Railway volume, since aircraft
  responses are capped at 800 records and vessels at 1500 and the /data
  page can poll every 10-30s — recording every poll would be tens of MB/day
  from one bbox alone. Expected the fix to be a time-gated sample interval
  decoupled from poll rate, landing in the 15-100MB/mo combined range
  depending on the interval chosen, plus a retention/rollup scheme so the
  raw-file directory doesn't grow forever.
- Design (server/dataArchive.ts, wired into server/routes.ts): aircraft and
  vessel positions are sampled independent of request/poll frequency via a
  30-min-per-kind throttle (module-level last-write timestamp), recorded as
  compact POSITIONAL ARRAYS (not objects — no repeated field names) into
  one append-only JSONL file per UTC day at
  `${DATA_DIR}/archive/{aircraft,vessels}/YYYY-MM-DD.jsonl`, rounded to
  ~11m lat/lon precision. Every sample also updates a tiny permanent
  rollup JSON (`${kind}_rollup.json`: total samples/records, per-day
  counts, first/last day) BEFORE any pruning, so raw-file retention (90
  days) never loses the count history — only the ability to replay exact
  positions that far back. New route `/api/data/archive/stats` exposes
  this for the wishlist.md volume-watch item without a shell.
  Computed at these parameters: aircraft ≈800 recs/sample × ~35B/rec ×
  48 samples/day ≈ 1.3MB/day ≈ 40MB/mo; vessels ≈1500 recs/sample ×
  ~30B/rec × 48/day ≈ 2.1MB/day ≈ 65MB/mo. Combined ≈105MB/mo — close to
  the prior session's aspirational <100MB/mo estimate; wishlist.md updated
  with the real figures and a note to revisit if live stats run hotter
  (record counts near the per-request caps more often than assumed).
- Downstream chain (REASONING STANDARD #1): recording starts today ->
  every subsequent session gains one more day of position history it can
  never otherwise recover -> R2's geofence transit counters and the
  corporate-jet / tanker-routing / destination-prediction hypotheses all
  become buildable once enough days accumulate -> none of this touches
  live trading (datacore/ boundary rule: zero imports from bot_engine.py /
  system_config.py / strategies/ / server/bot.ts) so it cannot affect
  order flow, sizing, or the kill switch even in a bug scenario worse than
  intended (worst case: excess disk usage on the Railway volume, caught by
  the new stats route and the wishlist watch item, not by any trading
  behavior change).
- Boundary discipline verified: `server/dataArchive.ts` has no imports
  from trading modules; it is pure filesystem I/O over positions already
  fetched by the existing `/api/data/aircraft` and `/api/data/vessels`
  handlers (unchanged upstream fetch/cache logic — this only adds a
  recording side-effect after the existing cache-set line in each).
- Regression tests FIRST (loop-health rule 3):
  `server/dataArchive.test.ts` — 7 new tests covering: UTC day-key
  boundaries, interval throttling (3 calls inside one window write only 2
  samples), compact-record rounding/null handling for both aircraft and
  vessel shapes, JSONL round-trip, and an end-to-end
  mkdtemp-write-read-stats-cleanup test proving the throttle, the per-day
  file append, and `/api/data/archive/stats`'s counts all agree. All 7
  pass; `npm run test:node` is 13/13 (6 pre-existing + 7 new).
- Verified: `npx tsc --noEmit` — identical 45 pre-existing errors with and
  without this change (git-stash A/B diff, same method as the prior
  backtest_v2 session), zero new errors, none in the touched/new files.
  `npm run build` succeeds (client 3 chunks + server bundle, same warning
  profile as before — the large maplibre-gl chunk is pre-existing from the
  map slices, not from this change).
- Version: 1.0.41 (from 1.0.40, read-then-incremented per the prior
  session's attribution-note lesson on the 1.0.36 collision).
- Not done (explicitly out of scope for one logical change): geofence
  transit counters (R2 gate 1) — this PR only builds the raw material R2
  will read; WebGL rendering / viewport-fetching / enrichment (rest of
  R1) — untouched, still open per open_questions.md.
- Rollback trigger: if `/api/data/archive/stats.kinds.*.approxBytesOnDisk`
  trends toward the Railway volume's plan limit faster than the ~105MB/mo
  estimate (e.g. because bbox record counts run near the 800/1500 caps
  far more often live than assumed), lengthen `SAMPLE_INTERVAL_MS` in
  `server/dataArchive.ts` first (cheapest lever) before shortening
  `RETENTION_DAYS` — retention loses history depth, the interval only
  loses temporal resolution.
- STARVED: no — this session's scope (position archive) was fully shipped;
  high-value work remains queued (KNOWN BROKEN #3/#4/#5, counterfactual
  logger, Sentinel-2 gate 1, rest of R1/R2/R3/R4) for future sessions.

## 2026-07-03 — [REPAIR] Extended-hours order gating (KNOWN BROKEN #8)

- Session start check: /api/health all-ok (Alpaca ACTIVE, python bridge ok,
  bot active, equityPeak=108151.39/drawdownPct=0.0% — confirms the
  2026-07-03 equityPeak-persistence fix is holding live across deploys).
  Loop-health ratio: 3 of 4 total experiments.md entries are [REPAIR]
  (framework bootstrapped today, <10 entries exist yet — below the 7/10
  escalation threshold, not a concern). Audit log / trade_feedback still
  unreachable from an autonomous session (owner-gated per KNOWN BROKEN #4)
  so this session worked from KNOWN BROKEN #8, the next actionable
  un-diagnosed item, per SESSION BUDGET (no matured experiment to judge —
  everything else logged today is same-day).
- PRIOR (before reading the order-submission code, REASONING STANDARD #10):
  expected either (a) both stock and options extended-hours paths already
  correctly gated (nothing to do), or (b) options orders missing a time
  gate and firing outside 9:30-4:00 relying on Alpaca to reject them
  (wasted scan cycles per the human's framing). Did NOT expect the actual
  finding — that options were fine and the real gap was on the stock side.
- Finding (READ BEFORE WRITE, this session): `executeTrades()` — the only
  function that ever calls `submit_options_order`/`select_contract` — is
  invoked exclusively `if (isMarketOpen)` (bot.ts:3030); outside market
  hours new trades are queued (`morningQueue`) and executed at the next
  open via `executeMorningQueue()`, gated on `clock.is_open`. Options were
  never actually at risk of an off-hours submission attempt. The
  `options_exit` OrderContext variant is declared but never passed by any
  call site — dead but harmless.
  The real bug: `getOrderParams()`'s extended-hours branch (4am-9:30am,
  4pm-8pm ET) computes wider-buffer limit prices for stock/ETF orders but
  never sets Alpaca's `extended_hours: true`. Per Alpaca's API, a
  day-limit order submitted without that flag outside regular hours is
  simply queued for the NEXT REGULAR session — it does not attempt to
  fill during the extended session it was priced for. This branch is hit
  live by the real-time WS position-exit handler (stop_loss/trailing_stop/
  take_profit — fires on any price tick, not gated to market hours) and by
  the Tier-3 SPY/QQQ floor buy. Net effect: a stop-loss or trailing-stop
  computed during a 4am-9:30am or 4pm-8pm price move would never actually
  attempt to execute until 9:30am the next regular session — silently
  defeating the stop during exactly the window (thin liquidity,
  pre-market gaps) it matters most.
- Downstream chain (REASONING STANDARD #1): adding `extended_hours: true`
  → those day-limit orders become eligible to fill during the pre-market/
  after-hours session Alpaca actually runs → a stop-loss priced at 6am can
  fill near 6am instead of silently waiting until 9:30am → smaller
  realized loss on overnight/pre-market adverse moves that would otherwise
  ride uncapped until the regular open → net effect is MORE stops firing
  during extended hours (intended; this is a bug fix restoring intended
  behavior, not a threshold change) with no change to entry cadence (entry
  orders during extended hours were already funneled through the
  market-hours-gated morning queue in the live-fire paths that matter).
- Change (one logical change, options untouched): extracted
  `getETHour`/`getOrderParams`/`OrderContext` out of `server/bot.ts` into a
  new pure module `server/orderParams.ts` (zero behavior change beyond the
  fix — needed because `bot.ts` has import-time side effects and isn't
  safe to import directly in a test). Added `extended_hours: true` to the
  extended-hours branch for `stop_loss`/`trailing_stop`/`take_profit`/
  `new_entry`. Left the options branch (`options_entry`/`options_exit`)
  untouched — Alpaca has no options extended-hours session, so the flag
  must never be set there.
- Regression test FIRST (loop-health rule 3): `server/orderParams.test.ts`
  (Node's built-in `node:test`, zero new dependencies; `getOrderParams` now
  takes an optional `etHourOverride` param for determinism). Verified by
  temporarily stripping the fix and re-running: 4 of 6 assertions FAILED
  on the pre-fix code (stop_loss/trailing_stop/take_profit/new_entry all
  missing `extended_hours`); all 6 pass post-fix. Added `npm run test:node`
  to package.json to run it (`tsx --test server/*.test.ts`). Note: CI's
  node-build job (`.github/workflows/ci.yml`, FROZEN) does not currently
  invoke this script — only `tsc --noEmit` and `npm run build` run in CI.
  Wiring `test:node` into CI is a follow-up worth a human-approved
  wishlist entry since ci.yml can't be self-edited.
- Verified locally: `npm ci && npx tsc --noEmit` shows zero NEW errors
  (diffed against main via `git stash` — identical pre-existing
  vite/client + tsconfig + Buffer.trim() errors, all unrelated to this
  change, all already non-blocking in CI's `|| true`); `npm run build`
  succeeds (client + server bundle); `npm run test:node` — 6/6 pass.
- Version: 1.0.36 (from 1.0.35).
- Frozen-path judgment call, stated explicitly for the human to override:
  this touches order-body fields inside `server/bot.ts`'s stock/ETF order
  construction. Read the FROZEN PATHS order-submission clause as covering
  the HTTP transport/auth/retry mechanics (the `alpaca()` helper) and
  `options_execution.py`'s `submit_options_order`, not the pre-existing,
  already-mutable `getOrderParams()` order-type/pricing logic (which
  already varies type/limit-price/time-in-force by time of day before this
  change) — neither the `alpaca()` transport function, retries, auth, nor
  `options_execution.py` were touched.
- Rollback trigger: if live audit logs (once accessible) show extended-
  hours orders being rejected by Alpaca (e.g. account not enabled for
  extended-hours trading) rather than filling, revert this commit — the
  pre-fix behavior (queue for regular open) is a safe fallback.

## 2026-07-03 — [RESEARCH] Equity-momentum backtest harness (`bot_backtest.py`)
- Change: added a reproducible backtest of the bot's OWN momentum scoring
  (`strategies/momentum.py`) run as a monthly ETF rotation, vs SPY buy&hold.
  Yahoo daily adj-close, stdlib only, cached under `.bt_cache/`.
- Window: 2016-01 → 2026-04 (matches `backtest_10yr_results.json`).
- Result (CAGR / Sharpe / maxDD):
  - SPY buy & hold ............ 14.13% / 0.83 / -33.7%
  - QQQ buy & hold ............ 18.67% / 0.88 / -35.1%
  - Bot momentum top-3 rot. ... 10.09% / 0.63 / -31.4%
  - **Dual-momentum SPY/QQQ ... 16.31% / 0.90 / -28.6%**  (beats SPY; higher Sharpe, lower DD)
  - 200dma regime filter ...... HURT (cash drag in a bull decade)
- Reference: live options-heavy bot (static artifact) = **0.27% CAGR**.
- Hypothesis: the live bot's drag is **options overtrading** (options avg
  -0.75%/trade over 328+ trades), NOT the core signal. A simple rules-based
  equity/ETF momentum core is a large improvement. Dual-momentum SPY/QQQ is a
  candidate but is IN-SAMPLE over a tech-led decade — needs out-of-sample /
  walk-forward validation before any live change (HONESTY METRIC risk).
- Caveat: equity logic only; the options leg can't be backtested without
  historical option prices (see wishlist).

## 2026-07-03 — [REPAIR] Backtest engine rebuild (backtest_v2.py) — PRIOR STATED BEFORE FIRST RUN
- Change: real engine replacing the backtest.py stub (KNOWN BROKEN #1).
  Daily-bar sim of strategies/momentum.py + mean_reversion.py with
  live-identical regime gating (regime_util.classify_regime_5level;
  vxx_ratio = VXX/30d avg per macro_data.py). No lookahead (signal close[i]
  -> fill open[i+1]), 5 bps/side costs, ATR stops (1.5x, 2-6% clamp),
  2R targets, regime time stops. Alpaca-first data, Yahoo fallback.
  Options/squeeze legs NOT simulated (no historical data — wishlist).
- PRIOR (before first real-data run, REASONING STANDARD #10):
  On SPY, 3yr, strategy=all I expect: (a) LOW trade counts (regime gate —
  NEUTRAL/BEAR/PANIC all block entries, so the engine only trades
  BULL/CAUTION days); (b) momentum Sharpe 0.3-0.8, positive but BELOW SPY
  buy-and-hold (base-rate check #3) because gating keeps it in cash much of
  the time; (c) mean-reversion few trades (RSI<40 rarely coincides with
  BULL/CAUTION) with small positive expectancy; (d) runtime well under the
  120s bot.ts budget. The deliverable is infrastructure honesty
  (promotion rule 3 becomes satisfiable), NOT alpha.
- Result (SPY, all, 3yr; runtime 2s — well under the 120s bot.ts budget):
  momentum: 42 trades, Sharpe 0.777, +3.1% total, maxDD 1.48%, win 64.3%,
  alpha -16.5 vs SPY CAGR 17.2 (buy-and-hold base rate). mean_reversion:
  0 trades. VXX data ok (real vxx_ratio on trade records).
- Prior vs actual: (a) low trade count CONFIRMED; (b) momentum Sharpe 0.777
  inside the 0.3-0.8 prior band, below buy-and-hold as predicted; (c)
  mean-reversion "few trades" was directionally right but reality is ZERO —
  oversold days essentially never coincide with BULL/CAUTION + score>=63;
  worth a rule-cost-audit question, not a code change; (d) runtime confirmed.
- Version: 1.0.34. Ships with offline tests (canned bars) in
  test_audit_critical.py; the backtest_v2 regime-consistency gated test now
  RUNS and PASSES; a new test asserts backtest regime blocking stays
  consistent with live system_config.get_adaptive_params.
- Also found (pre-existing, NOT this change): full-repo `pytest -q` dies at
  collection because test_auto_discovery.py calls sys.exit() at module level;
  excluding it, 7 failures + 1 error in network/keys-dependent files
  (identical with and without this change — verified by stash A/B). CI's
  4-file offline subset is green: 110 passed, 1 skipped.

## 2026-07-03 — [REPAIR] Live-system diagnosis (BOOTSTRAP Phase 2, public surface)
- /api/health all-ok (Alpaca ACTIVE, python bridge ok, bot active); calendar
  correct on the July-3 NYSE holiday. Finding: equityPeak in-memory only ->
  drawdown kill-switch high-water mark resets every deploy (6 today);
  strengthening fix touches frozen kill-switch machinery -> wishlist
  proposal, not edited. Deeper verification of KNOWN BROKEN #3/#4 blocked
  by owner-only auth on all diagnostic routes -> access options proposed in
  wishlist. No code change this entry.

## 2026-07-03 — [REPAIR] Kill-switch peak persistence (KNOWN BROKEN #7, human-approved)
- Change: persist state.equityPeak (max-drawdown kill switch high-water mark)
  to /data/voltrade/voltrade_equity_peak.json (/tmp fallback), restored on
  boot — mirroring the 2026-05-05 equity-curve persistence fix for the same
  bug class. All 4 raise-sites save on the same line; halt logic untouched.
  Version 1.0.35.
- Regression test FIRST (loop-health rule 3): TestKillSwitchPeakPersistence
  in test_audit_critical.py — 3 of 4 assertions PROVEN FAILING on the old
  code (no persistence file, no boot restore, raises without save); the 4th
  is a scope guard asserting the halt condition itself is unchanged. All 4
  pass post-fix.
- PRIOR (before live verification, REASONING STANDARD #10): after deploy,
  the peak file is created on the first account poll / Tier 1 cycle (market
  closed until Jul 6, so expect /api/health equityPeak = 0 until then);
  from Jul 6 onward equityPeak survives restarts (nonzero across deploys)
  and drawdownPct measures from the true historical peak.
- Downstream chain (#1): persisted peak -> drawdown measured from true peak
  -> halt CAN fire after a slow multi-deploy bleed -> possibly more halts in
  genuine drawdowns (intended). ROLLBACK TRIGGER: a spurious DRAWDOWN-KILL
  from a stale peak (e.g. after an intentional capital change) -> revert the
  commit AND delete voltrade_equity_peak.json from the volume.

## 2026-07-03 — [PRODUCT] /data map v1 shipped (PROMPTS.md Section A executed)
- A0 status check -> A1-A3 + STARVED metric installed (PR #98, human
  pre-approved inline). A4 map build shipped in 4 vertical slices:
  #99 map+imagery+datacore boundary (v1.0.36), #100 live aircraft/OpenSky
  (v1.0.37), #101 strategic sites x16 (v1.0.38), vessels scaffold this PR
  (v1.0.39, key-gated: goes live automatically when AISSTREAM_KEY is set).
- The /data tab is LIVE on the site: satellite imagery, live ADS-B
  aircraft (30s-cached boundary proxy, stale-over-error), Cushing/STLD/
  ports reference layer with tradable-exposure metadata, RAW/SIGNAL
  labeling per the surface rules, gate-2 disclosure in the layer panel.
- Boundary discipline: frontend calls /api/data/* only; base imagery tiles
  are the documented scoped exception. datacore/ has zero trading imports.
- Ladder position: everything shipped is RAW (ungated by design). First
  SIGNAL candidate remains Sentinel-2 tank-fill at Cushing — gate 1 ground
  truth = EIA weekly storage. The sites layer just gave it its ground.
- STARVED: no — high-value work remains queued (KNOWN BROKEN #3/#4/#5/#6/#8,
  counterfactual logger, Sentinel-2 gate 1) but this session cleared its
  planned scope. Session end.

## 2026-07-03 — [PRODUCT] /data map UX overhaul + two production bug fixes (v1.0.40)
- Human production review failed the v1 map on basic usability -> full
  redesign per DESIGN.md (installed this session, PR #103): full-bleed at
  every width (fixed inset under 56px desktop nav / 48+64px mobile bars),
  collapsible layer panel (FAB top-right; collapsed by default on phone,
  open on desktop), toggle switches with live status dots + count badges,
  RAW/SIGNAL as info-tooltip, aircraft+sites ON by default (alive at first
  paint), site detail card (side card desktop / bottom sheet phone),
  styled popovers, legend, loading skeleton with 8s failsafe, maplibre
  chrome themed (44px touch controls, dark attribution).
- PRODUCTION BUGS found & fixed in scope: (1) /api/data/layers returned []
  in prod — frozen Dockerfile never copies datacore/ into the image; fixed
  by importing the JSON statically so esbuild bakes it into dist/index.cjs.
  (2) aircraft 502 — OpenSky rejects the Railway egress; fixed with UA
  header + adsb.lol fallback (verified live: 679 aircraft over Cushing).
  (3) maplibre CSS dynamically imported -> never applied in prod build ->
  300px phantom canvas + unpositioned controls; fixed with static CSS
  import + resize-on-ready.
- Harness (rule 6): npm run visual GREEN at 390/768/1440, screenshots
  reviewed against DESIGN.md. Harness itself hardened: SwiftShader WebGL
  flags + full network isolation (external requests aborted).
- Remaining known (site shell, NOT this page; follow-up candidate): desktop
  tab nav clips at 768px; several nav buttons < 44px touch floor.
- STARVED: no.

## 2026-07-03 — [PRODUCT] Map v2 directive: constitution + roadmap installed (PR 1 of 3)
- BUILD-FIRST RULE added to EDGE DOCTRINE (human-approved inline):
  4-step free-alternative assessment before any paid wishlist entry;
  honesty clause. DESIGN.md gains PERFORMANCE BUDGET (10k+ features
  smooth on phone, <3s interactive, stale-beats-spinner, server
  proxies-only) + FEATURE COMPLETENESS CHECKLIST (global scale, failure
  modes, first-load/error/empty, client-side heavy work, honest limits).
- Map v2 roadmap filed in open_questions (R1 this session; R2 maritime
  transit analytics ARCHIVE-FIRST; R3 environmental layers; R4 3D globe
  perf-gated) + archive-enabled signal hypotheses with ladder paths.
- Wishlist: OpenSky free account ($0, HUMAN ACTION), FlightAware/SWIM
  priced entry WITH build-first analysis, archive volume watch.
- ATTRIBUTION NOTE: version collision discovered — the concurrent routine
  session's extended-hours fix (PR #97) and map slice 1 (PR #99) both
  carry code_version 1.0.36 (slice 1's hardcoded-string bump silently
  no-op'd). Impact limited: only #97 affects trading behavior. Rule
  forward: bump by READING the current version and incrementing, never by
  replacing a hardcoded prior value.

## 2026-07-03 — [PRODUCT] Map v2 PR 2/3: global feeds + PERMANENT POSITION ARCHIVE (v1.0.41)
- ARCHIVE EVERYTHING is live: server/datacoreArchive.ts records every
  ingested aircraft/vessel position to the Railway volume from this deploy
  forward. Adaptive thinning (30s near strategic sites / 60s low-altitude /
  5min cruise; 2min near ports / 10min open water / 30min anchored),
  hourly JSONL rotate, gzip after 2h, 7-day rollup into per-entity daily
  track summaries (bbox + coarse polyline), /api/data/archive/stats for
  the volume watch, /api/data/track/:kind/:id serves recent trails.
  13 hermetic node:test cases (thinning ordering, cadence enforcement,
  gzip/rollup lifecycle, round-trip, stats).
- Feeds: GLOBAL coverage. Aircraft: OpenSky primary with OAuth2 support
  (activates when OPENSKY_CLIENT_ID/SECRET land — wishlist), adsb.lol
  fallback with HONEST partial-coverage flag when the viewport needs
  >250nm; per-provider exponential backoff (30s..15min); in-flight dedup
  so all visitors share ONE upstream request per bbox; ?since= dedup
  returns {unchanged:true} instead of re-sending payloads; caps raised to
  5000 (WebGL client lands in PR 3/3). Vessels: aisstream subscription
  widened to global, ShipStaticData captured (shiptype/destination) and
  merged into reads, caps 20k in-memory / 5k per response, coverage
  honesty in the source string (terrestrial AIS = mid-ocean gaps).
- Feed-error diagnosis (the "feed error — retrying" symptom): root cause
  was OpenSky rejecting Railway egress + no backoff, so every 15s poll
  re-failed the primary before falling back. Backoff now pins the
  provider out for 30s..15min after failures, and the fallback serves
  immediately.
- STARVED: no.

## 2026-07-03 — [PRODUCT] Map v2 client half: WebGL rendering + enrichment (with PR #106)
- Rendering: MapLibre WebGL symbol layers (per-marker DOM eliminated).
  Runtime-generated SDF silhouettes (jet/turboprop/piston/helicopter;
  tanker/cargo/small-craft) tinted per-feature, rotated to heading,
  velocity-vector line layers zoom-gated >=6 (halves low-zoom draw load —
  found by the perf harness). Verified at 10,000 aircraft.
- Enrichment: class from ICAO type designator + emitter category (free
  feed fields — adsb.lol 't'/'category'); detail cards with archive
  trails (/api/data/track), "route data unavailable — filed plans are a
  paid source" honesty line, AIS destination shown when broadcast,
  per-layer stale/partial-coverage notes; Escape/keyboard rules kept.
- Harness upgraded to the perf budget: 10k-feature fixture, __vtMap
  pan-driving, TTI + median-frame guards. Calibration journey (logged
  honestly): p95 -> median-after-warmup (upload hitches vs steady-state),
  fixture ?since= support (validates the delta path AND stops measuring
  redundant re-uploads), dsf 1 (features not pixels under software GL).
  Final: PASS x3 widths, median 67-167ms @10k in SwiftShader, TTI <1.7s.
- CI note: #106's first docker-build failed on npm ECONNRESET (registry
  flake — gate correctly held). Frozen ci.yml/Dockerfile can't grow
  retries without human approval; recurrence -> wishlist proposal.
- STARVED: no.

## 2026-07-03 — [PRODUCT] Archive supersession + coordination lesson (merge of #106/#107 work)
- A routine session and this interactive session BOTH built MAP V2 R1's
  position archive concurrently (#107 merged first; this branch carried the
  fuller build). Reconciled to ONE system: server/datacoreArchive.ts stays
  (adaptive thinning per the ARCHIVE EVERYTHING amendment, gzip, 7-day
  rollups, /api/data/track trails needed by the map client); PR #107's
  server/dataArchive.ts + test removed — its uniform 30-min sampling and
  no-compression design didn't meet the amendment. Adopted from #107: the
  measured growth-estimate discipline (~100MB/mo documented in datacore/
  README + module header). #107's experiments entry stays (append-only).
- Version: three-way lesson compounding — #97/#99 collided on 1.0.36, and
  #106/#107 BOTH took 1.0.41. This merge lands as 1.0.42.
- COORDINATION GOTCHA (added to ops list): concurrent sessions must CLAIM a
  roadmap item before building — append a one-line [CLAIMED <date> <PR#>]
  marker to the roadmap entry in open_questions.md in their FIRST commit,
  and check for claims before starting.

## 2026-07-03 — [REPAIR] Aircraft feed resilience: third provider + cause capture (v1.0.43)
- Live incident found during v1.0.42 verification: production aircraft
  layer dead with "both providers backing off" — OpenSky blocked from
  Railway (known) + adsb.lol egress flake ("fetch failed", cause hidden)
  exponentially pinned the only fallback -> zero aircraft for fresh
  bboxes. Both fallbacks verified healthy from outside Railway (adsb.lol
  764, airplanes.live 548 aircraft, identical field shape) -> egress-
  specific, transient.
- Fix: fallback chain now OpenSky -> adsb.lol -> airplanes.live (per-
  provider backoff so one flake can't kill the layer) + fetch-failure
  cause codes captured into the error string (bare "fetch failed" was
  undiagnosable). Feature-completeness checklist Q2 gap, closed.
- Also explains v1.0.42's empty aircraft archive (writes happen on
  successful fetch only); vessels archive + ShipStaticData typing were
  boot-warm-up, re-verified post-deploy below.

## 2026-07-03 — [PRODUCT] Map v2 Part 1 CLOSED — live verification (v1.0.43)
- Production evidence: aircraft GLOBAL alive (782 over Europe via adsb.lol
  with airplanes.live behind it); POSITION ARCHIVE RECORDING (aircraft +
  vessels files, 1.7MB day one — and the vessels file survived the deploy,
  volume persistence working); vessels feed in expected post-deploy
  warm-up (lazy WS connect — filed as KNOWN BROKEN #9 with the one-line
  eager-connect fix for a routine). ShipStaticData typing verification
  pending warm-up.
- Day tally: 20 PRs through the needs-gated automerge; 4 live incidents
  found by verification and fixed same-day (datacore-not-in-image,
  OpenSky egress block, maplibre CSS phantom canvas, both-providers-
  backing-off); 2 new permanent ops rules (claim-before-build,
  version-by-increment); the archive is accumulating the proprietary
  dataset R2 builds on. STARVED: no — queue is deep but this directive's
  scope is fully shipped.

## 2026-07-03 — [RESEARCH] OpenSky creds verification (negative) + aircraft-feed licensing audit
- Human set OPENSKY_CLIENT_ID/SECRET in Railway (wishlist item closed).
  Verification NEGATIVE: 6+ fresh-bbox probes over ~30 min (wider than
  the 15-min max backoff window, so at least one live OpenSky attempt
  was forced) all served from community fallbacks; OpenSky never took a
  request. OpenSky reachable anonymously from a non-Railway network
  (HTTP 200) -> the Railway egress rejection persists even with OAuth
  creds available. Disambiguation (IP block vs. auth-endpoint failure
  vs. env-not-loaded) needs Railway deploy logs — handed to the human
  in the wishlist entry.
- Prior expectation (recorded in the wishlist entry BEFORE the test):
  creds "may restore the primary feed" — rejected; an egress-level
  block is not an authentication problem.
- Licensing audit (triggered by the verification request): OpenSky
  license = non-profit research/education ONLY, and operational REST-API
  integration requires written permission even for non-profits — both
  tripwires fire for us (paid site features + automated integration).
  airplanes.live free API = explicitly non-commercial. adsb.lol = ODbL
  1.0, the only terms-compatible provider for commercial display today.
  Flagged constraint + provider-priority recommendation filed in
  wishlist.md for human decision — ANALYSIS ONLY per instruction, no
  code or priority-order change shipped.
- STARVED: no — usage-calibration loop build queued in same session.

## 2026-07-03 — [RESEARCH] Correction to the aircraft-feed licensing audit (same day)
- Human correction: VolTradeAI has NO paid product — billing code exists
  but nothing is charged; the site is a proof of concept. The earlier
  entry's claim that the commercial tripwire "fires" was wrong on the
  facts as of today.
- Revised assessment (wishlist entry updated in place): adsb.lol (ODbL)
  compatible now and after monetization; airplanes.live compatible NOW
  (non-commercial POC) but flips incompatible the day anything is
  charged; OpenSky's OPERATIONAL-use clause (written agreement for any
  live/automated integration, even non-profit) still applies today —
  and OpenSky is non-functional from Railway anyway, costing a 12s
  timeout per fresh viewport for zero data.
- Constraint reframed as a MONETIZATION TRIPWIRE: re-run provider
  compliance before enabling billing/ads. Still analysis-only; provider
  order unchanged pending human decision.
- STARVED: no.

## 2026-07-03 — [REPAIR] Vessel stream eager boot connect (KNOWN BROKEN #9, v1.0.44)
- PRIOR (stated before implementing): the aisstream websocket only ever
  connects when `ensureVesselStream()` is called, and the only caller was
  the `/api/data/vessels` GET handler — so a fresh deploy with zero
  visitors to the map stays disconnected indefinitely. Expected fix:
  calling the same function once at route-registration time (server boot)
  closes the gap with no other behavior change, since `ensureVesselStream`
  already no-ops if a socket is OPEN/CONNECTING and no-ops if the key is
  unset.
- Downstream chain (REASONING STANDARD #1): eager connect at boot -> the
  websocket is live before any request arrives -> the 60s archive-snapshot
  interval (already running unconditionally) has real positions to record
  from minute one instead of only after a map visit -> position-archive
  continuity across deploys improves (this feeds R2's transit-analytics
  signal, whose value depends on unbroken history) -> no change to sizing,
  scoring, or any trading path (this module has zero imports from bot_engine/
  system_config/strategies, so the datacore boundary is intact).
- Fix: extracted `vesselStreamEnabled(env)` and `bootVesselStream(env,
  connect)` into new pure module `server/vesselStream.ts` (no imports,
  fully unit-testable — avoids importing routes.ts's heavy deps, notably
  auth.ts's top-level sqlite `db` open, into a test). Replaced the three
  independent `process.env.AISSTREAM_KEY` truthy checks (layers status,
  vessels route, and the new boot call) with the single shared predicate,
  removing a duplication-drift risk. Added
  `bootVesselStream(process.env, ensureVesselStream)` immediately after
  `ensureVesselStream`'s definition in `server/routes.ts`.
- Test (loop-health rule 3 — regression test that would have caught the
  break): `server/vesselStream.test.ts`, 4 cases — `vesselStreamEnabled`
  true/false on key presence, and `bootVesselStream` invokes its connect
  callback iff a key is present (dependency-injected, so it directly
  proves the "connect on boot when enabled" wiring without needing a real
  WebSocket or Express app). 17/17 `npm run test:node` pass (13 pre-
  existing + 4 new).
- `npx tsc --noEmit`: identical 46 pre-existing errors with/without this
  change (git-stash A/B diff), zero new errors, none in touched/new files.
  `npm run build`: client + server bundle succeed. No client/ files
  touched, so DESIGN.md visual-verification (promotion rule 6) doesn't
  apply.
- Version 1.0.43 -> 1.0.44 (read-and-increment).
- STARVED: no — KNOWN BROKEN #5 (data-module wiring audit) and #6 (pytest
  collection) remain queued for a future session; SESSION BUDGET caps
  this session at one action.

## 2026-07-03 — [PIPELINE] OpenSky dropped from aircraft chain (human decision, v1.0.45)
- Change: fetchAircraft in server/routes.ts no longer attempts OpenSky —
  chain is now adsb.lol (primary) -> airplanes.live (fallback). OAuth
  helper removed (v1.0.43 git history holds the implementation for
  reinstatement). layers.json + harness fixture attribution updated;
  coverage note no longer promises OpenSky credentials.
- Why: verification showed OpenSky never serves from Railway even with
  creds (egress block), so it contributed only a ~12s dead attempt per
  fresh viewport; AND its operational-use license clause requires a
  written agreement we don't have. Human emailed OpenSky requesting a
  research agreement — reinstate + re-verify Railway connectivity if
  granted (wishlist entry updated with the trigger).
- Expected effect: fresh-bbox latency drops by the dead-attempt time
  (~12s worst case); zero functional coverage change (OpenSky served 0%
  of requests). Downstream chain: fewer request timeouts -> fewer
  in-flight dedup pile-ups behind slow fetches; archive feed cadence
  unchanged (it records on successful fetches only).
- Regression tests: server/aircraftChain.test.ts (4 source-level tests:
  OpenSky fully absent, adsb.lol before airplanes.live, no stale
  coverage note, layers.json attribution matches chain).
- Rollback trigger: OpenSky grants the research agreement (reinstate),
  or both community providers show sustained simultaneous failure in
  the audit log (re-add a third provider — see the provider-redundancy
  research item in open_questions.md).

## 2026-07-03 — [RULE-REVIEW] Monetization tripwire hardened: runtime guard (v1.0.46)
- Constitutional half (human-approved verbatim, this message): CLAUDE.md
  KNOWN STATE gains the MONETIZATION TRIPWIRE standing rule — sessions
  touching billing/pricing/subscriptions/ads/paid-gating must re-run the
  wishlist provider-compliance check before merging.
- Runtime half (new server/providerCompliance.ts): billingActive() =
  BILLING_ENABLED=true OR STRIPE_SECRET_KEY present (billing.ts — frozen,
  read-only — activates on that key, so key presence is the earliest
  observable monetization signal). While airplanes.live (non-commercial
  license) remains in the aircraft chain, activation produces a throttled
  COMPLIANCE-WARNING row in the persistent audit log + a licensing check
  on /api/health that degrades overall status — a dashboard-only
  monetization flip becomes visible to the next DAILY routine's health
  check within hours, no code change needed to detect it.
- Wiring: boot check at datacore route registration + tick per aircraft
  request (both throttled to one warning per 6h window); /api/health
  Check 6 in bot.ts. Datacore boundary intact: the guard lives in the
  serving layer (routes/bot), NOT datacore/ — datacore keeps zero
  knowledge of billing.
- Tests: server/providerCompliance.test.ts (6) — inactive-by-default,
  BILLING_ENABLED trip, STRIPE_SECRET_KEY trip, tick throttling with
  injected clock, non-commercial list pinned to the live chain, and
  wiring pins for /api/health + aircraft path.
- Measurement-integrity note: this changes /api/health's payload shape
  (adds checks.licensing; can newly degrade status). Direction of bias:
  none on trading metrics — it can only ADD a warning state.
- OPS LESSON (2nd occurrence — now a pattern): importing auth.ts's
  top-level sqlite db into any node:test hangs the runner (open handle).
  PR #113 dodged it with pure vesselStream.ts; this PR first hit it,
  then adopted the same pattern (pure module + injected audit writer).
  RULE: server modules that need the db AND unit tests take the db
  dependency by injection; never import ./auth from a tested module.

## 2026-07-03 — [RULE-REVIEW] Constitution batch: fall-through, dead-code, hygiene + first audit
- Applied three human-approved amendments (this message, bookkept in
  wishlist): SESSION BUDGET -> productive fall-through ladder; DEAD CODE
  POLICY; CONSTITUTIONAL HYGIENE (audit files proposals, never
  self-applies; live rule conflicts resolved by GOAL order + filed).
- Queued research filed per the directive: aircraft/vessel provider
  redundancy (chain is two-deep post-OpenSky; vessels single-sourced);
  OpenSky likely-returner tracked with REVIEW-BY 2026-08-17, no adapter
  retained (git-history reinstatement documented).
- Ran the FIRST CONSTITUTIONAL AUDIT (details in wishlist): 2 findings
  proposed for consolidation (rules-in-KNOWN-STATE -> STANDING BEHAVIORS
  section; scattered periodic duties -> one PERIODIC AUDITS register);
  factual drift fixed directly (backtest STUB claims stale since the
  rebuild); interaction checks clean (STARVED, BUILD-FIRST, tripwire).
- Docs-only; no version bump (no runtime behavior change).
- STARVED: no — remaining queue continues in this same session (daily
  usage loop PR + scale-now schedule answer).

## 2026-07-03 — [RULE-REVIEW] Daily usage-calibration mode + scale-now schedule (docs)
- Applied the approved daily-aggressive usage rule (CLAUDE.md KNOWN
  STATE updated in place — supersedes the weekly-only rule from earlier
  today; usage_log.md carries both modes with the ~2026-07-24 revisit).
- voltrade-usage-check routine prompt canonicalized in usage_log.md
  (DAILY 21:30 ET; draft-only Gmail caveat + Notifications-tab
  recommendation embedded so the routine never depends on send).
- SAME-DAY SCALE RECOMMENDATION recorded in usage_log.md per the human's
  "scale now" directive: create product-am 9:00, daily-midday 12:30,
  product-pm 14:00, product-eve 20:00, usage-check 21:30, edge-late
  22:30 — completes the A5 8-run table + daily nudge. Evidence: weekly
  15% (resets ~Jul 5) with 3 routines + the heaviest interactive day;
  throttle trigger stated (5-hour peaks >~80% → drop order).
- STARVED: no — directive fully executed this session (PRs #114 #115
  #116 + this).

## 2026-07-04 — [PIPELINE] SEC EDGAR Form 4 (insider transactions) — first datacore pipeline beyond aircraft/vessels

- Session start: read CLAUDE.md, all of research/, per [PRODUCT] session
  protocol. Loop-health ratio over the last 10 entries: 0 [REPAIR], 3
  [PIPELINE]/[PRODUCT], 3 [RULE-REVIEW], 1 [RESEARCH] — well below the 7/10
  escalation threshold. KNOWN BROKEN #3/#4/#5 remain open (owner-gated
  diagnostics, per wishlist) but don't touch datacore/ or block product
  work, per this session's brief. Prior branch (#117, daily usage-calibration
  docs) was already merged to main — reset claude/quirky-hopper-u5pdl1 onto
  origin/main fresh per the merged-PR restart protocol, no stacking.
- Chose this action (option (a): advance a datacore/ pipeline through gate
  1) over the other three (UI-only work with no new signal, a proposal-only
  new-root writeup, or API/docs hardening) because EDGE DOCTRINE #1 ("BUILD
  DATA, DON'T BUY IT") names SEC EDGAR real-time Form 4 as a standing
  example, `datacore/layers.json` had exactly one non-"live" candidate
  (`tank_fill`, gated on Sentinel-2 satellite image processing — infeasible
  to attempt end-to-end in one session, no image-analysis toolchain
  available here), and no datacore pipeline had shipped since aircraft/
  vessels/archive — Form 4 requires no API key, no image processing, and no
  new paid access, purely free-data processing (the labor-not-ingredient
  edge the doctrine calls out).
- PRIOR (before writing any parser code, REASONING STANDARD #10): expected
  the main design risk to be SEC's fair-access rate limiting (10 req/s) and
  the feed's atom format double-listing each filing (once per filer, once
  per issuer) causing double-counted/double-fetched filings; expected the
  fix to be accession-number dedup plus a small per-request delay, and
  expected a hand-rolled tag-scoped regex extractor to be sufficient given
  no XML parsing library exists in package.json today (matches CLAUDE.md's
  "don't add unneeded abstractions" — Form 4's schema is flat and stable).
  Both priors held; see design below.
- Verified network reachability first (Bash curl, not a guess): SEC EDGAR
  is directly reachable from this environment. Live-fetched two REAL,
  current Form 4 filings (accessions 0001104659-26-080497 — a derivative
  RSU grant, code A — and 0000902664-26-003001 — a 3-reporting-owner
  non-derivative open-market SALE, code S, two transactions) and
  hand-verified every field (issuer, owner relationship flags, transaction
  code/shares/price/shares-owned-after) by reading the raw filed XML myself
  before writing a single assertion. This IS ladder gate 1 (DATA — verified
  against an external truth source): for a filings parser, the filed
  document itself, read directly, is the only ground truth there is to
  check against — there's no separate "official" source above the filing.
- Design: `server/edgarForm4.ts` (pure module, zero imports from trading
  logic — datacore boundary rule) — a dependency-free tag-scoped XML
  extractor (`parseForm4Xml`), a transaction-code classifier
  (`classifyTransactionCode`: P=open_market_buy the informative discretionary
  case, S=open_market_sale, A=award_grant, M=option_exercise, G=gift,
  F=tax_withholding, else other), an atom-feed parser that dedupes by
  accession number (`parseFilingFeed`), an index.json XML-document picker
  (`pickOwnershipXmlName`), a sequential fetch-with-delay batch fetcher
  (`fetchLatestForm4Filings`), and an in-memory cache + 15-min poll loop
  booted eagerly at route registration (`bootForm4Poll`) — same eager-boot
  shape as `vesselStream.ts`, deliberately avoiding KNOWN BROKEN #9's lazy-
  first-request gap. Wired at `/api/data/insider` in `server/routes.ts`,
  kind: raw (as-filed display, no predictive claim — the interpreted
  "clustering predicts returns" question is gate 2, unattempted, logged as
  a new hypothesis in open_questions.md). Registered in
  `datacore/layers.json`. `datacore/README.md` updated to correct its
  stale aspirational `pipelines/` Python layout note (every real pipeline,
  including this one, lives in `server/*.ts` — DEAD CODE POLICY spirit:
  don't let docs claim an unbuilt structure is authoritative).
- Client (MUTABLE rule: new user-visible bot function needs UI in the same
  PR): `client/src/pages/datamap.tsx` — the insider feed has no lat/lon, so
  rather than force it into the maplibre marker machinery (`wireLivePoints`,
  built for geospatial layers), it renders as an inline expandable list
  directly under its row in the existing layer panel (new `.vt-filings-*`
  CSS in `index.css`), scoped to avoid the floating-panel collision risk a
  new independently-positioned overlay would add on phone widths (site-card
  and layer-panel already claim opposite corners / the mobile bottom sheet).
  Defaulted on (`DEFAULT_ON.insider = true`) — matches aircraft/sites, the
  other no-key-required RAW layers.
- Downstream chain (REASONING STANDARD #1): pipeline ships gate-1-verified
  -> `/api/data/insider` serves real filings today -> the feed accumulates
  its own history from this point forward (no live trading impact —
  datacore boundary rule holds, verified zero imports from bot_engine.py /
  system_config.py / strategies/ / server/bot.ts) -> once enough history
  accumulates (or the free SEC bulk full-index is pulled in, unexplored,
  logged in open_questions.md), gate 2 (does insider-buy clustering predict
  forward returns vs. a random-entry base rate, REASONING STANDARD #3) can
  be attempted -> only then would this ever become a SIGNAL eligible for
  the tournament in strategies/, per the ladder — no shortcut taken.
- Regression tests FIRST (loop-health rule 3): `server/edgarForm4.test.ts`
  — 6 tests: the two gate-1 field-by-field fixture checks against the real
  filings above, transaction-code classification for all 7 codes, feed
  dedup against a real (trimmed) atom-feed snippet proving the actual
  filer/issuer double-listing collapses to one entry, and two
  `pickOwnershipXmlName` cases (found / not-found) against a real
  index.json shape. All 6 pass; `npm run test:node` is 33/33 (27
  pre-existing + 6 new) — no existing assertion touched or weakened.
- Verified: `npx tsc --noEmit` — identical 45 pre-existing errors with and
  without this change, zero new errors, none in touched/new files.
  `npm run build` succeeds (same chunk/warning profile as documented).
  `npm run visual` (client/ touched — PROMOTION RULES rule 6): added
  `insider` to the layers + a 2-filing fixture in
  `scripts/visual_check.mjs`; PASS at 390/768/1440 with 0 hard failures
  (touch-target warnings present are pre-existing global-nav elements, not
  from this change) — screenshots reviewed, the new layer row and its
  inline filings list render correctly with GRANT/SELL badges and RAW
  labeling, no overflow or overlay-coverage regression.
- No backtest applies (data pipeline, not a strategy/parameter change) —
  PROMOTION RULES item 3 is N/A here, same as the aircraft/vessel PRs.
- Version: 1.0.47 (from 1.0.46, read-then-incremented).
- Hypothesis: gate 1 stands permanently verified (parser correctness is a
  static property, not a market claim). Gate 2 hypothesis and prior stated
  in research/open_questions.md ("Insider Form 4 clustering as a signal") —
  expect small positive edge in officer/director open-market buys on
  small/mid caps, near-zero on mega-caps, kill if no separation from
  random-entry baseline after >=90 days of accumulated history.

## 2026-07-03 — [PRODUCT] Strategic-sites accuracy audit: 16/16 imagery-verified, 11 corrected (v1.0.48)
- Human reported Port of Charleston mispositioned; full audit ordered.
  Method (now compiled as scripts/site_verify.py + site_candidate_verify.py
  per EDGE DOCTRINE #3): render every stored coordinate on Esri World
  Imagery with a crosshair; the facility must be visibly present.
  DESIGN.md gains the human-approved REFERENCE DATA ACCURACY rule.
- VERIFIED UNCHANGED (5): cushing_enbridge (tanks under crosshair),
  cushing_plains (tank rows), port_la (wharf/cranes, San Pedro),
  port_nynj (Elizabeth container yard), port_savannah (Garden City
  stacks).
- CORRECTED (11) — old -> new (offset, what the old pin actually was):
  - port_charleston (32.921,-79.86) -> (32.8325,-79.8800): ~10km — pin
    was residential Mount Pleasant; now on Wando Welch container yards.
  - cushing_hub (35.985,-96.767) -> (35.9487,-96.7587): ~4.1km — pin was
    downtown Cushing street grid; now mid-tank-farm in the main district.
  - stld_butler (41.428,-84.855) -> (41.3705,-84.9170): ~8.2km — pin was
    farmland; now on the SDI Butler mill (scrap yard + melt shop visible).
  - stld_columbus (33.532,-88.415) -> (33.4473,-88.5768): ~17.9km(!) —
    pin was Columbus MS suburbs; mill is actually at the Golden Triangle
    megasite next to GTR airport.
  - stld_sinton (28.041,-97.56) -> (28.0563,-97.4493): ~11.0km — pin was
    ranch land W of town; mill is NE along the rail line.
  - stld_columbia_city (41.157,-85.488) -> (41.1199,-85.3484): ~12.4km —
    pin was downtown Columbia City; mill is E of town (structural mill +
    rail sidings visible).
  - port_houston (29.681,-94.942) -> (29.6770,-95.0060): ~6.2km — pin
    was open water in Trinity Bay; now on Barbours Cut yard.
  - port_oakland (37.796,-122.279) -> (37.7980,-122.3145): ~3.1km — pin
    was Jack London Square; now on OICT container yard.
  - port_norfolk (36.877,-76.328) -> (36.9155,-76.3275): ~4.3km — pin
    was Lambert's Point COAL pier (wrong facility); now on NIT apron.
  - port_lb (33.754,-118.216) -> (33.7515,-118.2130): ~370m — pin was in
    the channel; now on Pier J stacks.
  - port_seattle (47.582,-122.352) -> (47.5820,-122.3474): ~340m — pin
    was Harbor Island's fuel tank farm; now on T18 container yard.
- Lesson (feeds the geofence future): "researched coordinates" from
  memory/public materials produced town centroids, not facilities — the
  archive's site-proximity thinning (nearAnySite) has been using these
  wrong positions, so near-site full-resolution sampling was mistargeted
  for 6 of 16 sites by >4km. Corrected data fixes that silently.
- Downstream chain: sites layer markers move -> archive adaptive
  thinning now samples the RIGHT areas at full resolution -> future R2
  transit counters + tank-shadow work inherit verified ground truth.

## 2026-07-03 — [PRODUCT] Site category icons: anchor/tanks/factory silhouettes + legend (v1.0.49)
- Map v2.1 SYMBOLS directive: strategic-site markers upgraded from
  generic colored dots to category silhouettes in the existing SDF icon
  system (per-feature icon-color on GPU, upright, dark halo for imagery
  contrast): vt-port anchor, vt-tank 3-cylinder cluster, vt-mill
  factory-with-chimney. Legend now leads with the three shapes (inline
  SVG twins of the canvas shapes); aircraft/vessel color entries kept.
- Promotion rule 6: npm run visual green at 390/768/1440 (0 hard
  failures; pre-existing site-shell nav warnings unchanged). Icon
  legibility self-reviewed via isolated render (sites-only fixture,
  US view, 390px + 1440px): three shapes distinct at phone size,
  correct category colors, legend aligned.
- Perf unchanged: same symbol-layer path as aircraft (16 features is
  noise next to the 10k-aircraft budget); harness perf medians 117ms
  unchanged from v1.0.48 baseline.

## 2026-07-03 — [PRODUCT] US power plants layer: 9,833 plants, fuel icons, clustering (v1.0.50)
- Map v2.1 POWER PLANTS directive. Free data root per EDGE DOCTRINE #1:
  WRI Global Power Plant Database v1.3.0 (CC BY 4.0 — commercial-safe,
  attribution shipped in layers.json + detail card + legend source).
  Compiler: scripts/build_powerplants.py (re-runnable when WRI updates)
  -> datacore/powerplants/us_power_plants.json (9,833 US plants, 762KB
  compact rows; solar 3283 / gas 1852 / hydro 1449 / wind 1139 / oil 876
  / other 879 / coal 297 / nuclear 58).
- Serving: /api/data/powerplants — whole-file, day-cached, static import
  (esbuild bakes it; Dockerfile never copies datacore/). RAW layer, no
  ladder gate needed (no predictive claim); signal hypotheses filed in
  open_questions.md (EIA-930 generation-mix utilization; NRC outage
  adjacency) with full ladder paths.
- Client: maplibre native clustering (clusterMaxZoom 7, radius 50) so
  ~9.8k features stay legible + cheap on phones; unclustered points are
  8 new fuel silhouettes in the SDF system (atom/coal pile/flame/derrick/
  drop/turbine/sun/bolt) with per-fuel tint; cluster click zooms in;
  point click -> detail card with MW + operator + attribution; legend
  gains a plants row; panel row with count badge (Zap icon).
- Tests: server/powerplants.test.ts (3) — dataset scale + row validity +
  US bounds, CC BY attribution present, route/layers.json wiring.
  Harness fixture serves the REAL compiled dataset so the perf window
  now measures 10k aircraft + 9.8k clustered plants together.

## 2026-07-03 — [PRODUCT] M4 mobile performance pass: fill-rate fix from per-layer profile (v1.0.51)
- Method (compiled: scripts/perf_profile.mjs): pan-frame medians at 390px
  under SwiftShader (mid-range-phone proxy), per-layer A/B at two views.
  PRIOR (stated before profiling): expected the new 9.8k-plant layer or
  its clustering to be the top cost.
- PROFILE REJECTED THE PRIOR: at global zoom, base=17.7ms, +plants+sites
  =17.5ms (clustering is FREE — supercluster renders only ~40 blobs),
  +aircraft=33.8ms. The 10k-aircraft symbol layer was the only
  meaningful cost, and it is FILL-RATE bound (software rasterizer +
  phone GPUs pay per drawn pixel; 10k icons at constant 0.5 scale).
  Continental zoom (z4.5) was already smooth for all combos (21ms).
- Fix: zoom-interpolated icon-size on the two 10k-class live layers
  (aircraft 0.32@z2 -> 0.55@z7; vessels 0.30 -> 0.50) — ~60% fewer
  drawn pixels where icons are dense, full size where you can tap them.
- Measured effect: all-layers global-zoom median 32.8 -> 27.3ms in the
  A/B; full harness at 390px median 33ms p95 67ms (previous runs were
  ~117ms class); 768px 83ms; 1440px 117ms — all far inside the 300ms
  budget. Icons re-verified legible at the new global-zoom size.
- Tile prefetch not measurable offline (harness aborts CDN) — noted
  honestly; raster layer cost showed as flat base across combos.

## 2026-07-03 — [PRODUCT] Map v2.1 ops items: delivery channel fixed, tripwire verified, audit-cycle proposal
- GMAIL DRAFT-ONLY FIX (human directive): both calibration routines
  (usage-check daily 21:30, weekly-review Sun 10:00) now deliver via
  their FINAL SESSION OUTPUT — read from the Claude Code Notifications
  tab, the one channel verified to reach the human (routine completions
  land there by platform design). The Gmail-draft step was dropped from
  both canonical prompts as a dead letterbox (connector verified
  draft-only, no send tool; drafts sat unread). usage_log.md prompts +
  CLAUDE.md KNOWN STATE updated. Honest limit: no push-to-phone channel
  is verifiable from a session; if the Notifications tab proves too
  passive in practice, the human should say so and we revisit.
- TRIPWIRE FALSE-ALARM CHECK: production /api/health shows
  checks.licensing = ok -> STRIPE_SECRET_KEY is NOT set in Railway and
  the compliance warning is NOT firing. Condition for changing the
  trigger ("if firing falsely") unmet — guard unchanged. Residual
  behavior documented: setting a Stripe TEST key without charging would
  fire it; that is by design (earliest observable signal) and the
  warning text says exactly what to do.
- AUDIT CYCLE consolidation proposal filed in wishlist.md per the
  hygiene process (three scattered periodic triggers -> one SESSION
  BUDGET clause + register; policies untouched; human decides).

## 2026-07-03 — [PIPELINE] Aircraft chain three deep: adsb.fi third leg (v1.0.52)
- Human directive (multi-modal expansion): self-hosted receivers OFF the
  table (declined; logged in open_questions so no session re-proposes
  hardware); software-only third provider instead. LICENSING FIRST per
  the standing rule: adsb.fi = personal/non-commercial + attribution
  (same class as airplanes.live; MONETIZATION TRIPWIRE list updated to
  cover it); adsb.one rejected (Cloudflare blocks server egress); ADS-B
  Exchange rejected for the free chain (community API non-commercial +
  keyed; commercial tier is paid Enterprise).
- Integration: PROVIDERS gains per-provider response-array key (adsb.fi
  returns "aircraft" where the others return "ac"; URL pattern
  /api/v2/lat/{lat}/lon/{lon}/dist/{nm}); shared-upstream pattern,
  backoff, cause-capture all inherited. layers.json attribution updated.
- INTERNATIONAL COVERAGE VERIFIED through the new leg (the directive's
  requirement): Tokyo 130, Sydney 146, São Paulo 69 aircraft from
  adsb.fi; legs 1-2 verified global in prior sessions (all three are
  worldwide community networks — coverage everywhere feeders exist).
- Tests updated: aircraftChain.test.ts pins the three-deep order
  (ODbL leader first); providerCompliance.test.ts pins that BOTH
  non-commercial legs are in the tripwire list (sync test caught the
  edit requirement immediately).

## 2026-07-04 — [PRODUCT] Live trains layer: Finland + Norway launch, archived, iconed (v1.0.53)
- Multi-modal directive, TRAINS part. Free real-time rail positions with
  clean licenses (checked FIRST): Finland Digitraffic (CC BY 4.0, no
  key, plain JSON) + Norway Entur (NLOD-class open, ET-Client-Name
  header only, GraphQL mode:RAIL) — both verified live before build.
  US freight rail positions are PROPRIETARY (Class I railroads sell
  them; no free source) — stated in layers.json where users read it AND
  pinned by test so no session chases it. Amtrak has no clean official
  free JSON — future source evaluation, not launch.
- Server: /api/data/trains — pure mapping module server/trainsFeed.ts
  (unit-tested with real captured payloads; m/s->km/h for Entur;
  Digitraffic has no bearing -> null -> upright icon), shared 30s cache
  + in-flight dedup + per-source backoff, per-source status in the
  response so the panel labels coverage HONESTLY ("FI 47 · NO 12").
  Positions feed the permanent archive: datacoreArchive gains the
  trains kind end-to-end (2-min fixed cadence; hourly JSONL; gzip;
  rollup; recentTrack -> click-through trail like aircraft/vessels).
- Client: vt-train SDF locomotive (teal #2dd4bf; rotates to bearing
  where published), 30s poll, detail card with speed + per-country
  source attribution + archive trail, legend entry, panel row with
  count badge + per-source note.
- BUG CAUGHT BY SELF-REVIEW (rule 6 render check, would have shipped
  invisible): a ["case",["get","rotate"],...] icon-rotate expression
  silently killed symbol rendering (source had features, image
  registered, zero rendered). Fix: always-numeric bearing property +
  plain ["get","bearing"]. Lesson: maplibre expression rejection is
  SILENT — any new expression-driven layer needs a rendered-count
  assertion in review, not just "layer exists".
- Gates: node 41/41 (5 new trains tests incl. archive round-trip via the
  shared machinery), CI python 114, harness green 390/768/1440.

## 2026-07-04 — [RESEARCH] Ships coverage verified + trucks build-first conclusion + freight proxies
- SHIPS (directive): aisstream subscription confirmed GLOBAL in code
  (BoundingBoxes ±90/±180 — routes.ts) — the honest gap is physical:
  terrestrial AIS sees ~40-60nm offshore, mid-ocean is dark. Satellite
  AIS filed in wishlist as PRICED (quote-only, ~$500+/mo entry class)
  with build-first analysis: dead-reckoned predicted tracks + coastal
  reacquisition cover most port-transit needs free; do not buy unless
  a gated signal needs mid-ocean truth.
- TRUCKS (directive): build-first analysis TERMINATES — individual
  truck positions are private fleet telematics with no public feed at
  any price tier relevant to us; conclusion filed in open_questions so
  no session chases it. Four free freight PROXIES filed with ladder
  paths instead (CBP border waits, PeMS truck-lane volumes, FMCSA
  carrier census, port TEU monthlies) — archive-first, gate-2 gated.
- STARVED: no — directive fully executed across PRs #124 (aircraft
  third leg), #125 (trains layer), and this docs bundle.

## 2026-07-04 — [PRODUCT] Power-plant position accuracy: EIA-860 coords + top-100 imagery-verified (v1.0.54)
- Directive (Map v2.2): human confirmed Hardeeville mispositioned.
  KEY FINDING that shaped the protocol: GPPD and EIA-860 AGREE on
  Hardeeville's wrong position — the registries share self-reported
  geocodes (address/office, not the plant), so registry cross-agreement
  is NOT verification; imagery is the only ground truth. Rule text
  ("imagery or an authoritative source") interpreted accordingly:
  authoritative-source checks fix DISAGREEMENTS, imagery establishes
  VERIFIED.
- Data work: joined all 9,833 GPPD-US plants to EIA-860 2024 by plant
  code (9,557 matched; 276 GPPD-only = mostly retired since 2021).
  67 disagreements >300m (median 1.6km, worst 13.4km — wind-farm
  centroid vs substation ambiguity dominates); EIA 2024 coordinates now
  preferred for ALL matched plants.
- Imagery verification of the TOP 100 BY MW (directive): composite
  4x4 verification sheets (7 sheets, z14 crops with crosshairs) —
  98/100 passed on sheet review; 2 borderline resolved PASS at z15/z16
  close-up (Bath County = crosshair on the pumped-storage intake works;
  Ravenswood = on the station). 100/100 verified; audit artifact
  checked in (datacore/powerplants/imagery_verified.json).
- Product honesty: row format gains a verified flag; detail card says
  "Position imagery-verified." or "Position approximate
  (registry-reported — GPPD/EIA-860)."; the layer panel row notes
  "top 100 by MW imagery-verified · rest approximate".
- Tests: 7-element row validation + top-100-all-verified + audit
  artifact + EIA-860 credited. 42/42 node, 114 python, harness green.

## 2026-07-04 — [PRODUCT] Form 4 full view + filings archive (v1.0.55)
- Map v2.2 FORM 4 UI directive. Server: filings now ARCHIVED
  (COLLECT-EVERYTHING) — every 15-min poll appends new accessions to
  daily JSONL under the datacore archive volume (restart-safe dedup via
  day-file seeding), days >2 gzipped; /api/data/insider/history?days=N
  merges archive + live cache (history accumulates from 2026-07-04).
- Client: #/data/filings full view (hash-driven overlay; deep-linkable;
  back-button works): readable table at 768/1440 (ticker+company,
  insider+role, color-coded BUY/SELL/GRANT/EXERCISE, shares, price,
  computed value, date, SEC-filing link per row), stacked labeled cards
  at 390; filters all/open-market/buys/sells; designed empty/loading/
  error states. Panel keeps the compact list + "Open full view" button.
- TWO BUGS CAUGHT BY SELF-REVIEW: (1) home.tsx's tab-hash sync stomped
  #/data/filings back to #/data on mount — sync now rewrites only when
  the hash ROOT differs (subpaths survive); (2) the harness fixture
  matcher's startsWith let /api/data/insider shadow .../insider/history
  — exact-match-first fix in visual_check.mjs.
- Tests: archive dedup round-trip, gzip-day readback, route + poll-loop
  wiring pins (45/45 node). Harness green all widths; both view states
  (empty + populated) screenshot-reviewed.

## 2026-07-04 — [PRODUCT] Detail-card link-outs + vessel flag states (v1.0.56)
- Map v2.2 SHIP DETAIL CARDS directive. Vessels: flag state now shown,
  computed locally from the MMSI MID prefix (ITU table baked into
  client/src/lib/mmsiFlag.ts — data derived from the AIS message itself,
  no external lookup); dimensions honestly OMITTED (our aisstream
  subscription doesn't carry them reliably); destination/type/speed
  already shown. LINK OUT (never embed — photo copyright): MarineTraffic
  + VesselFinder by MMSI. Aircraft: Planespotters (photos/registry) by
  hex + adsb.lol live-track link. Rendered as pill links in the detail
  card, 32px min height, external target with rel=noreferrer.
- Gates: build, node 45/45, harness green all widths.

## 2026-07-04 — [PRODUCT] Layer panel v2: collapsible groups + imagery-date honesty (v1.0.57)
- Map v2.2 LAYER PANEL directive: with 7+ layers the flat list stopped
  scaling. Panel now groups into Base / Live tracking / Facilities /
  Filings & flows / Signals—coming-soon (signal/planned layers auto-
  route to the last group), with collapsible headers ("2/3 on" counts),
  per-layer info toggles (name click -> description + source), and the
  existing status lines/badges/switches unchanged. 44px touch targets
  on headers and name buttons.
- IMAGERY METADATA honesty: the imagery row now states "capture date
  unavailable (Esri base tiles)" — DESIGN.md gains the human-approved
  standing rule: show "imagery as of [date]" where a source exposes
  capture dates (Sentinel-2 when it lands), say "date unavailable"
  otherwise; no imagery surface may imply currency it cannot prove.
- Verified at 390/768/1440 (rule 6): desktop screenshot shows all four
  groups with honest per-layer notes; phone keeps the collapsed FAB.

## 2026-07-04 — [PIPELINE] Shadow-fleet analytics from our own AIS archive (v1.0.58)
- Map v2.2 SHADOW FLEET directive. server/shadowFleet.ts derives from
  the vessel archive (first payoff of ARCHIVE EVERYTHING beyond trails):
  gap events (silent >6h AND reappeared >100km), identity candidates
  (name under two MMSIs; new-MMSI-near-last-position hull-swap
  heuristic), loitering (>=4h, median <2kts) in 7 public STS zones
  (datacore/shadow_zones.json: Laconian, Ceuta, Malta, Fujairah,
  Singapore OPL, Kerch, Port Said).
- RAW/SIGNAL boundary enforced: the surface shows COUNTS ONLY with the
  coverage-loss caveat attached (panel row in Filings & flows, zone
  breakdown when loitering >0); per-vessel claims stay OFF until gate 1.
  Gate-1 plan filed (open_questions): enrichment of detections for
  publicly documented shadow vessels (OFAC SDN annexes, KSE lists) vs a
  size-matched random tanker cohort — identical coverage loss in both
  cohorts controls the terrestrial-AIS ambiguity. Gate-2 hypothesis:
  zone event rates lead tanker-rate names (FRO/STNG/TNK basket) +
  crude spreads; second-order reason the edge survives: maritime-intel
  vendors sell to compliance desks, not rate traders.
- Tests: 4 hermetic synthetic-archive cases (gap yes/no discrimination,
  both identity heuristics, loiter vs fast-transit, aggregation +
  caveat + wiring pins). 49/49 node; harness green.

## 2026-07-04 — [RESEARCH] Fusion hypotheses filed + collect-everything audit (Map v2.2 close-out)
- Three fusion hypotheses filed in open_questions with pairings,
  testable claims, and gate-1 ground truths (insider x facility at the
  verified SDI mills; generation shifts x operator tickers with the
  registry->ticker mapping as gate-1 work; ship-movement anomalies x
  tanker/retail names with TEU reports as truth). Logged, not built —
  per the directive.
- COLLECT-EVERYTHING verified in code for every layer: aircraft/
  vessels/trains stream to JSONL; Form 4 archives per poll; static
  reference data (plants/sites/zones) is git-versioned by doctrine;
  derived stats (shadowstats) intentionally not archived (recomputable
  from archived ingredients); imagery honestly not archived until the
  Sentinel-2 pipeline lands. Doctrine line added to open_questions.
- Map v2.2 directive fully executed across PRs #127-#131 + this docs
  close-out. STARVED: no.

## 2026-07-04 — [PRODUCT] Map UI v2.3: panel overflow root-caused + self-see harness rule + fullscreen (v1.0.59)
- ROOT CAUSE of the human-reported clipping: .vt-map-controls was
  position:absolute with NO bottom constraint, so the panel's
  max-height:100% resolved against an auto-height wrapper and never
  engaged — the panel grew past the viewport, clipped by the page's
  overflow:hidden, lower rows unreachable. Fix: wrapper now
  top+bottom-constrained (flex column, pointer-events pass-through
  under the panel); panel max-height:100% + existing overflow-y:auto
  now actually scroll.
- Panel restructured per directive: groups beyond the first fold
  (Facilities, Filings & flows, Signals) start COLLAPSED — headers
  visible, one tap to expand; the Form 4 FEED is fully removed from the
  panel (a feed doesn't belong in a layer-toggle sidebar) — the panel
  keeps one "Open filings view" button; the feed lives only in
  #/data/filings where columns wrap (word-break added), never clip.
  Dead panel-feed code + CSS removed per the dead-code policy.
- HARNESS SELF-SEE (approved amendment, DESIGN.md + wishlist): the
  harness now opens the panel via its own control, expands every
  collapsed group, and asserts panel-bottom-in-viewport, scroll-when-
  overflowing, every registry layer has a reachable row, every toggle
  scrollable-into-view and hit-testable (nothing covering it).
  PROVEN AGAINST THE BUG (loop-health rule 3 applied to the harness
  itself): A/B with the old CSS makes the harness FAIL with exactly the
  reported defect ("panel bottom 1084 past viewport 900"); fixed CSS
  passes. The harness gap that let this ship is closed by construction.
- FULLSCREEN MAP MODE: top-left 44px toggle hides the site nav (desktop
  top bar; phone top+bottom bars) for a true full-viewport map;
  sessionStorage-persisted; map.resize() on toggle. Verified
  mechanically at 390: nav display:none, map rect 0..innerHeight,
  persistence flag set; screenshot reviewed.
- Gates: harness green all widths WITH self-see active; node 49/49;
  CI python 114.

## 2026-07-05 — [PRODUCT] Earnings-language full view (8-K Item 2.02) — v1.0.82
- Territory: T-CLIENT (client/src/pages, index.css, scripts/visual_check.mjs)
  + minimal SHARED touches (datacore/layers.json registry entry,
  package.json version bump, this file) per WORKSTREAM PARTITION.
- Context: NEW DATA ROOTS #1's pipeline (server/sec8kEarnings.ts, gate 1
  DATA passed 2026-07-04, v1.0.67) shipped API-only with no UI, same
  sequencing edgarForm4.ts used before filings.tsx landed. This was the
  explicitly queued next PRODUCT item in open_questions.md's GEOSPATIAL
  LICENSING REGISTER section. Highest-value action this session: give
  the already-validated (gate 1) pipeline its user-facing surface,
  mirroring the Form 4 pattern exactly (panel row -> full hash-routed
  view) rather than starting a new pipeline or a new gate-1 effort.
- Build: new `earnings` entry in datacore/layers.json (RAW, live,
  honest gate-2-open language carried into the description — no
  predictive claim, mirrors sec8kEarnings.ts's own doc comment).
  client/src/pages/earnings.tsx — #/data/earnings full view: card list
  (not a table — this data is prose, not rows) reading
  /api/data/earnings-language/history, company-name filter, item-code
  tags, excerpt/expand-full-release toggle (480-char clamp), link-outs
  to both the SEC filing index and the actual exhibit (never embedded —
  same link-out-only rule as vessel/aircraft photo links). datamap.tsx
  wired identically to the insider pattern: LAYER_GROUP -> filings,
  DEFAULT_ON true, its own polling effect against
  /api/data/earnings-language (not the /history route — mirrors
  insider's live-cache-first pattern), panel-row "Open earnings
  language view" button, hash listener alongside the existing
  filingsOpen state.
- SELF-REVIEW CAUGHT (rule 6, before opening the PR): my first pass
  reused `.vt-filings-seclink` (a 32px icon-only button in the existing
  Form 4 view) for the new Exhibit/Filing link-outs, which render as
  icon+text pairs here — a scratch harness run (temporary copy of
  visual_check.mjs pointed at #/data/earnings, deleted after use, not
  committed) caught both links under the 44px touch-target minimum at
  390/768. Fixed with a dedicated `.vt-earnings-linkbtn` class (44px
  min-height) instead of overloading the shared class, and bumped the
  read-full-release button to 44px too (untriggered by the fixture's
  short sample text, but real releases run long — fixed proactively
  rather than waiting to be caught live).
- Gates: `npm run visual` green at 390/768/1440 (0 hard failures;
  pre-existing site-shell nav warnings unchanged; the one new soft
  warning, "Filings & flows 3/3 on" clipped-control, is the same
  below-the-fold false-positive class as the pre-existing Planner/Taxes
  nav warnings — the self-see battery, which actually scrolls and
  verifies reachability, shows the new layer with 0 self-see failures
  and "14 layers toggled clean" in toggle-consistency); ZERO-COST-WHEN-
  OFF unaffected (earnings gates on `enabled.earnings` like every other
  layer); node test:node 121/121; tsc --noEmit shows only pre-existing
  unrelated errors (verified none touch datamap.tsx or earnings.tsx).
  Python suite not touched (no .py files in this diff) — pytest is not
  installed in this sandbox to re-verify, noted honestly rather than
  claimed.
- Downstream chain (REASONING STANDARD #1): new layer row -> one more
  default-on poll (60s interval, same cadence as insider) -> the
  ZERO-COST-WHEN-OFF gate proves this is skipped entirely when the
  layer is off, so no new baseline cost for users who don't want it;
  when on, a second small JSON poll alongside insider's — negligible
  next to the 10k-aircraft budget already measured in this harness.
- Not attempted this session (correctly out of scope): gate 2 (does
  guidance language predict forward returns) — this PR is a RAW display
  of the pipeline's existing gate-1-passed output only, per the
  SPINOUT-READY / RAW-vs-SIGNAL rule.
- STARVED: no — this was a fully-specified queued item, executed start
  to finish in one PR.

## 2026-07-05 — [PIPELINE] CFTC Commitments of Traders (COT) — rescued from a stalled dirty PR and merged (v1.0.86)

- Territory: T-BOT (new top-level Python module wired into
  server/bot.ts's tier3Strategic; storage_config.py path additions).
- SESSION-START CHECK (Repair Mandate + system health): full offline
  Python gate re-verified green (328 passed, 1 skipped, after this PR's
  17 new tests — was 311/1 before). No KNOWN BROKEN item required
  [REPAIR] this session; #10 (dead SCORE_BAND config) stays correctly
  deferred pending shadow_portfolio history, per its own entry.
- FINDING (the actual highest-value action this session): PR #134
  ("CFTC Commitments of Traders — free positioning-data pipeline,
  v1.0.58") was opened 2026-07-04 from a since-abandoned branch
  (claude/dazzling-planck-64joy2), fully built (`cftc_cot.py`,
  `test_cftc_cot.py`, 17/17 offline tests, live-verified 156/156-week
  backfill across all 7 symbols, 0 validation rejections), but never
  merged — `mergeable_state: "dirty"`, `total_count: 0` check runs, main
  never advanced past its pre-PR base. This is a live instance of the
  OPS GOTCHA already on file ("a dirty claude/* PR stalls SILENTLY: no
  merge ref -> pull_request workflows never start -> no checks, no
  automerge, no error"). Consequence: an entire validated EDGE DOCTRINE
  #1 pipeline sat invisible for a full day — zero references anywhere
  in this file or open_questions.md on main (`grep -rn cftc` across
  both confirmed zero hits before this entry), meaning any session that
  read research/ before this one would have had no idea the work
  existed and could have duplicated it from scratch.
- WHY REBUILD-FROM-DIFF INSTEAD OF A GIT MERGE/CHERRY-PICK: the stale
  branch's base predates ~50 merged PRs (package.json version 1.0.57 vs
  current 1.0.85; research/experiments.md and open_questions.md have
  been rewritten under it many times over) — a cherry-pick would
  conflict on every touched shared file. Per EDGE DOCTRINE #3 (COMPILE
  KNOWLEDGE INTO CODE — never re-reason what's already been reasoned),
  the code itself (`cftc_cot.py`, `test_cftc_cot.py`, the
  storage_config.py path additions, the tier3Strategic wiring diff) was
  reused byte-for-byte from the stale PR's diff; only the
  version-dependent surroundings (package.json bump, this log, the
  open_questions.md entry, the bot.ts insertion point) were re-applied
  fresh against current main.
- RE-VERIFIED FRESH (did not just trust the year-old PR description):
  `python3 -m pytest -q test_cftc_cot.py` 17/17 pass; CI's 4-file
  offline subset 120 passed/1 skipped (unchanged baseline); full bare
  `pytest -q` 328 passed/1 skipped (was 311/1 — net +17, zero
  regressions); `npx tsc --noEmit` diffed before/after the bot.ts change
  line-for-line: exactly one new error, `cotOut.trim()` on a `Buffer`
  return type, the identical pre-existing pattern already present at
  every other `execPythonSerialized(...).stdout.trim()` call site in
  this same function (12 such errors already existed; now 13) — no new
  error *category*; `npm run build` clean. LIVE-VERIFIED against the
  real CFTC Socrata API from this session's sandbox: 156/156 weeks
  backfilled for all 7 symbols (GLD/SLV/USO/CORN/TLT/SPY/QQQ), 0
  accounting-identity rejections; second immediate call confirmed the
  20h staleness guard returns `{"status": "skipped"}` with zero network
  calls. Local test-run archive/checkpoint files removed after
  verification (not part of the repo; would only ever live on the
  Railway volume).
- WIRED AT GATE 1 ONLY: `server/bot.ts` `tier3Strategic()` step 5 calls
  `run_daily_update()` every hourly cycle; the module's own guard makes
  23 of 24 calls a free file-mtime check. Deliberately NOT wired into
  `deep_score`/`macro_data` — gate 2 (does COT-index positioning predict
  forward returns vs. a random-entry baseline) is unstarted, logged with
  a stated prior and kill criteria in open_questions.md.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): one more hourly subprocess
  call in tier3Strategic -> guarded to a cheap file check on 23/24 calls
  -> zero effect on deep_score/scoring/sizing/position count (nothing
  reads COT data yet) -> zero live-trading behavior change from this PR.
  The only observable effect until gate 2 ships is a growing archive
  file on the Railway volume.
- PR #134 (the stalled original) is being closed as superseded by this
  session's PR, which carries its full delta forward — no unique work
  from #134 is lost. Recorded here so the supersession is traceable from
  the log, not just the PR close comment.
- GATE 2 NOT ATTEMPTED THIS SESSION — same prior and kill criteria as
  the original 2026-07-04 build (real edge expected on the commodity
  contracts, weak-to-none expected on SPY/QQQ given the legacy report's
  weaker financial-futures classification); ready to run next session
  now that the backtest engine already exists and 156 weeks are
  archived from day one.
- STARVED: no — recovering already-validated, already-tested work that
  was about to be silently lost was higher expected value than starting
  a brand-new pipeline from zero this session.
