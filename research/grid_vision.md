# GRID VISION — ML-assisted complete US power-grid mapping

> PROVENANCE: human directive received 2026-07-07 in-session
> ("NEW TERRITORY — GRID VISION"). Multi-session program under
> DATACORE MAXIMUS and ALL standing directives (no mock data, no
> artificial walls, continuation mandate, subagent mandate,
> BLOCKED-FOR-MIKE routing, mobile-first 390px). Quoted lines are
> verbatim from the directive; connective text is session-condensed.
> This file is the program charter; the resume block at the bottom is
> the cross-session handoff state (updatable — everything above it is
> stable charter text).

## Objective (verbatim)

"The most complete, accurate, and visually aligned US power-grid map
achievable from real data — OSM as the base, ML imagery detection as
the gap-filler and verifier, expanding to full national coverage
state by state."

## Program rules (inherited + directive-specific)

- NOTHING BUILDS until Phase A files its research doc and Phase A2
  files the products plan: "the products define the spec, not the
  reverse."
- Provenance is a first-class field everywhere: every feature carries
  `osm-verified` / `ml-extended` / `ml-discovered`, rendered
  distinctly on the map, required by ratchet.
- Honesty about physics: wires are not resolvable from free satellite
  imagery — lines are INFERRED from detected towers + corridor
  evidence, and the product says so. Flow animation only ever at the
  data's actual granularity ("NEVER invented per-line flow"). Ages:
  "never a guessed date presented as fact."
- Coverage honesty: poor imagery ⇒ partial coverage stated + dated
  revisit trigger, never faked complete. Per-state coverage manifest
  is a required artifact.
- PAID RESOURCES ARE IN SCOPE: architect to the paid boundary and
  file BLOCKED-FOR-MIKE purchase orders (exact service, cost, signup
  link, credential name for Railway, capability unlocked). "Mike
  approves and pays fast; the moment the credential lands in Railway,
  detect and activate without being told." Continue on CPU-feasible
  scope meanwhile.
- OSM give-back: where license-compatible, contribute verified
  corrections back upstream in OSM-contribution style.
- Security standing rules apply (keys presence-checked only, never
  echoed/logged/archived).

## Territory

GRID VISION is its own workstream territory (T-GRIDVISION):
research/grid_vision*.md, future datacore grid-detection modules +
scripts (detection pipeline, tile builds), and the /data powergrid
layer surfaces it extends. Overlaps with T-CLIENT (datamap.tsx) and
SHARED files follow the WORKSTREAM PARTITION merge-order protocol —
cross-territory changes belong wholly to the GRID VISION session that
owns the primary change.

## PHASE A — RESEARCH FIRST (parallel subagents; nothing builds until this files)

Deliverable: `research/grid_vision_research.md` — a permanent
research doc assembled from four parallel workups:

1. (GV-A1) The April 2025 Taylor & Francis deep-learning
   power-infrastructure detection pipeline — exact citation, obtain
   code, metrics, adaptability verdict
   (ADAPTABLE-AS-IS / WITH-WORK / NOT-USABLE), license constraints.
2. (GV-A2) Labeled training datasets — Duke Electric T&D
   Infrastructure Imagery Dataset (figshare), PLAD
   (github.com/andreluizbvs/PLAD), TTPLA, others; OSM-as-weak-labels;
   ranked by fitness for towers + substations in 0.3–1m US overhead
   imagery, with evaluation-set fitness emphasized (Phase B starts
   with VERIFY).
3. (GV-A3) Methods survey — RetinaNet tower detection + inter-tower
   routing + corridor extraction; the HOT/OSM human-in-the-loop
   hybrid used to map Pakistan/Nigeria/Zambia HV grids; SAR-based
   tower detection (through-cloud, OSM-prompted); 3D-reconstruction /
   shadow height inference.
4. (GV-A4) Imagery inventory — resolution each free source provides
   per US region (Esri/Vantor basemap tiles, NAIP aerials,
   Sentinel-2) with capture dates and BULK-ACCESS LICENSE reality;
   compute assessment — inference cost per state at each imagery
   level, CPU-feasible vs external GPU (if external/paid →
   BLOCKED-FOR-MIKE purchase order).

Each claim in the filed doc distinguishes VERIFIED (fetched, read)
from REPORTED (secondary source). Hypotheses + acceptance gates filed
per the experiments discipline (priors stated before testing).

## PHASE A2 — GRID DATA PRODUCTS PLAN (files before Phase B)

Deliverable: `research/grid_vision_products.md` — enumerate every
credible use of the mapped grid, each with hypothesis, consumer, and
build order. "The products define the spec, not the reverse."

- TRADING: grid-stress signal (mapped capacity × EIA-930 demand ×
  weather); buildout detection as a datacenter/industrial leading
  indicator; outage-exposure screens for listed utilities/industrials;
  retirement + interconnection-queue pipeline signals; validated
  signals feed the regime/ML loop via the ROOT VALIDATION LADDER.
- DATA PRODUCTS: verified grid dataset (ODbL share-alike legal call →
  BLOCKED-FOR-MIKE before sale); grid-stress index / buildout-alerts
  subscription API; per-state coverage reports.
- PLATFORM: alerting; Everything Graph enrichment (plants ↔ operators
  ↔ corridors ↔ facilities).

## PHASE B — DETECTION (spec set by A/A2)

- B1 VERIFY: run detection over existing OSM corridors; confirm/flag
  geometry, including line-end-vs-visible-continuation misalignments.
- B2 EXTEND: confidence-scored continuation of lines beyond OSM ends
  along detected tower chains.
- B3 DISCOVER: blind detection in unmapped areas.
- Provenance tags (`osm-verified` / `ml-extended` / `ml-discovered`)
  attach at creation and render distinctly. Confidence is a required
  field. Lines are tower-chain inferences, stated honestly.

## PHASE C — NATIONAL ROLLOUT

Texas first, then ERCOT neighbors / PJM / MISO / WECC, all 50 states
+ DC. Per state: detection pass + verification metrics + honest
coverage statement + PR MERGED before the next batch starts. Poor
imagery ⇒ partial coverage + dated revisit trigger, never faked
complete.

## PHASE D — VISUALIZATION (premium standard applies)

- D1 zoom-adaptive rendering — dotted-out/solid-in, no popping.
- D2 full color scheme incl. provenance rendering, colorblind-safe.
- D3 power-flow animation ONLY at the data's actual granularity —
  regional interchange as regional flow, "NEVER invented per-line
  flow."
- D4 clickable popups: type/voltage/operator/capacity/confidence/
  provenance/imagery date + entity-graph links.
- D5 equipment knowledge base (tower/substation types, what the
  detector actually saw).
- D6 age/first-seen dating — exactly three states: bounded window
  from historical imagery; authoritative date (labeled with source);
  unknown. "Never a guessed date presented as fact."
- D7 toggles per-state/type/provenance; mobile-first 390px.

## PHASE E — RATCHETS

- Perf harness gates every state's tiles (the all-off TTI and
  per-layer budgets extend to each rollout batch).
- Geometry-alignment test battery.
- Provenance + confidence as REQUIRED fields (schema-enforced).
- Per-state coverage manifest, machine-checked.
- Detection metrics vs labeled datasets published per model rev —
  "no cherry-picked metrics."

## AMENDMENT — DATA STRATEGY & ACCURACY HONESTY (human directive, 2026-07-07)

> PROVENANCE: human amendment received in-session 2026-07-07, installed
> same day. Key language verbatim. Rationale (Mike's framing): the
> initial labeled data (PLAD, Duke) is modest and mostly non-US/
> non-NAIP — training-data growth and honest accuracy reporting are
> FIRST-CLASS, ONGOING work, not one-time setup.

1. SELF-BOOTSTRAPPING TRAINING DATA: use high-confidence OSM-verified
   detections as new labeled examples to expand the training set on
   each round — "the model's confirmed agreements with OSM become
   tomorrow's training data." GATE THIS CAREFULLY: "only
   high-confidence, OSM-corroborated detections enter the training
   set (never the model's own uncertain guesses — that would teach it
   its own errors)." Log the training-set size and composition each
   iteration. Prioritize closing the known gaps: US NAIP-domain
   examples and substations (both underrepresented at start).
2. HUMAN-IN-THE-LOOP CORRECTIONS FEED BACK: every disagreement
   between model and OSM that a human resolves becomes a labeled
   example, per the HOT/OSM pattern. "Corrections are training data,
   not just fixes."
3. PER-STATE ACCURACY, REPORTED OPENLY: for every state, measure
   detection accuracy against OSM ground truth (precision/recall on
   verified corridors) and publish it in the coverage manifest AND on
   the map's layer info (e.g. "Texas: 91% verified against OSM;
   Montana: 61%, sparse imagery, flagged for revisit"). "Never a
   single national accuracy number that hides weak states." The
   measured, disclosed accuracy is a product feature and a selling
   point — "a data buyer trusts a vendor who states their error
   rate. Cherry-picking or hiding weak coverage violates the honesty
   rules." (Consistent with Item 2.8's evidence asymmetry: OSM
   corridors measure RECALL; precision claims still require the
   human-sampled pass — the published per-state number states which
   it is.)
4. ACCURACY-GATED PROMOTION: a state's detections only get promoted
   from 'ml-discovered' toward higher-trust presentation as its
   measured accuracy clears a PRE-STATED bar; below it, they stay
   visibly low-confidence. Improvement over time is expected and
   logged — each retraining round reports whether per-state accuracy
   moved up.

CONTINUOUS: "the model, the training set, and the reported accuracy
all improve as the system runs, and the honesty about where it's weak
is never traded away for a cleaner-looking map." Phase B/C/E specs
absorb these as requirements: training-set composition log per
iteration; corrections pipeline from the adjudication UI; per-state
accuracy in the coverage manifest (Phase E ratchet) and layer info
(Phase D); promotion bars pre-stated before any state's first
measurement.

---

## RESUME STATE (update every session that works the program)

As of 2026-07-07 later same session (claude/new-session-iu72vf):

- PHASE A COMPLETE — all four subagent reports filed in
  research/grid_vision_research.md (Items 1–4 + cross-cutting
  summary). Headlines: T&F paper = ADAPTABLE-WITH-WORK recipe, no
  weights/annotations released, retraining mandatory (tower AP50
  ~73% @0.3 m is the bar); Esri ML/bulk use FORBIDDEN by quoted
  contract clauses (display + identify-metadata use stays fine) →
  NAIP public domain is the substrate, streamed free via Planetary
  Computer STAC; honest scope = transmission towers + substations
  (poles invisible at NAIP GSD); eval = two-layer benchmark (Duke-US
  CC-BY ground truth + OSM-corridor recall with human-sampled
  precision); B1 method = corridor-restricted detector (PI-Detection
  MIT starter + ETDII CC-BY labels) + S1 SAR verify experiment +
  Sundial shadow-height attribute; TX corridor-verify is
  CPU-feasible (~7 h @0.6 m); RunPod purchase order filed
  (BLOCKED-FOR-MIKE, $50 deposit, RUNPOD_API_KEY) — training needs
  GPU regardless, sweeps later.
- PHASE A2 COMPLETE — research/grid_vision_products.md filed: IP
  boundary (provenance tags = license classes; indices/alerts are
  the primary commercial surface, ODbL dataset is credibility +
  give-back), trading uses A1–A4 with priors + ladder paths, build
  order with NO ML on the critical path to the first product
  (grid-stress index v0 from OSM TX + EIA-930 + weather, all
  already archived), and the Phase B spec requirements.
- Existing assets to reuse: /data powergrid TX-pilot layer (OSM
  PMTiles, v1.0.166, wiring fixed v1.0.177); scripts/
  build_power_tiles.sh (TX tiles <1 min, US-full documented);
  EIA-930 grid demand stream live; GEM GIPT 182k units + GEOT
  ownership graph (entity-join spine); Esri capture-date identify
  contract proven (v1.0.173); CDSE Copernicus pipeline (S1/S2
  chips) live for the SAR experiment.
- A1 GATE-1 PASSED (v1.0.180, same session): TX grid capacity
  registry built from a fresh Geofabrik extract
  (datacore/gridvision/tx_grid_registry.json,
  scripts/grid_capacity_tx.py, criteria pre-stated in the script
  header before the comparison ran). 104,928 circuit-km >=69kV
  (118.5% of the ERCOT lower-bound anchor), unknown-voltage share
  3.8%, 345kV+ 31,559 circuit-km; census matches known ERCOT
  structure (138kV subtransmission dominant, 345kV backbone,
  500/230/115kV only at non-ERCOT edges). Definitional caveat
  (route-miles vs circuit-miles) stated in the manifest — gate-2
  refines vs EIA/HIFLD.
- A1 STEP 2 SHIPPED (v1.0.181): county->BA crosswalk
  (datacore/gridvision/tx_county_ba.json via scripts/
  grid_county_ba.py; source chain = GV-A5 workup, research doc Item
  5). 254/254 TX counties, BA codes join EIA-930 directly (ERCO 218
  counties / SWPP 78 / MISO 36 / EPE 3 / PNM 2; 80 multi-BA, NO
  primary fabricated — polygon arbiter queued: EIA Atlas BA layer
  mid-migration, re-check ~late July). State-preferred BA lookup
  (Rio Grande NM-row smear caught + fixed; Harris/MISO Entergy edge
  verified real). RETAIL-BA honesty labeled everywhere.
- A1 STEP 3 SHIPPED (v1.0.182): per-county + per-BA capacity
  distribution (datacore/gridvision/tx_ba_capacity.json via
  scripts/grid_ba_capacity.py — segment-midpoint point-in-polygon
  over Census counties, 16s runtime). All three pre-stated
  expectations held: conservation EXACT (108,515.0 km registry ==
  distributed, test-recomputed), 345kV+ ambiguous share 31.3%
  (inside the predicted 15-35% CREZ-seam band), out-of-state 2.7%.
  ERCO-exclusive 345kV+ backbone: 19,207 circuit-km. Multi-BA
  county km pooled AMBIGUOUS by BA set — never split by invented
  ratio.
- A1 GATE-2 DESIGN FILED (products plan, appended section
  2026-07-07, BEFORE any computation): stress index v0 = demand
  percentile x forecast strain (EIA-930 type DF) x CPC degree-day
  extremity; NO capacity denominator (circuit-km is not MW — the
  map spatializes exposure, honestly); outcome variable
  operationalized without LMPs; fit 2015-2022 / validate 2023-2025;
  seasonal base-rate control; PASS = >=1.5x out-of-sample lift
  stable across three summers; kill date 2026-08-15.
- GATE-2 PREREQS SHIPPED: DF polling (v1.0.183, #321); historical
  backfill mechanism (v1.0.184) — env-gated GRID_DEMAND_BACKFILL=1
  (opt-in OFF, R8; Mike asked to flip it), 2019->now D+DF,
  done-marker, seed-window heap protection. HISTORY CORRECTION: the
  endpoint serves ~2019+ (not 2015 as the design sketch said) — the
  gate-2 split becomes fit 2019-2022 / validate 2023-2025 (same
  three validation summers; note 2020 COVID anomaly sits in
  training).
- RUNPOD_API_KEY LANDED IN RAILWAY (Mike, 2026-07-07) — GPU
  training/sweeps UNBLOCKED. Purchase order in wishlist marked
  resolved.
- BACKFILL FLAG FLIPPED (Mike, 2026-07-07): backfill verified
  RUNNING on prod (1,466 day-files / 44MB plain mid-walk at check
  time; gz-at-end; volume headroom confirmed ~2.5GB vs ~130MB peak).
- GATE-2 COMPUTED 2026-07-07 (scripts/grid_stress_gate2.py against
  keyless EIA-930 bulk history + committed CPC artifact; full record
  datacore/gridvision/gate2_result.json): NOT PASSED — both
  pre-stated outcome operationalizations VOIDED on their own
  spot-validation rules (v1: ERCOT emergencies are well-forecast +
  shed load caps metered demand; v2: pooled percentiles blind to
  ~5-6%/yr demand growth). Recorded-not-claimed lifts 1.455/1.554,
  both single-summer-carried — the stability clause caught real
  regime-carry twice. CONSEQUENCE: stress index = DESCRIPTIVE-ONLY
  surface, labeled non-predictive; no sellable/tradable claim. V3
  path (fresh design, later session, discount compounds): researched
  full public ERCOT event list 2019-2025 as ground truth +
  growth-aware extremes; criteria re-filed before computation.
- ITEM (2) SHIPPED 2026-07-07 (v1.0.191, see experiments.md): descriptive
  stress dashboard surface — server/gridStress.ts (ERCO-only archive fold
  + CPC join, cache + 6h poll) + /api/data/grid-stress + #/data/grid-stress
  panel (client/src/pages/gridstress.tsx, launcher on the /data panel).
  Equal-weighted composite of the three raw ingredients, deliberately NOT
  the voided gate-2 fitted weights; `predictive: false` on every response;
  percentiles withheld (never guessed) below 5 same-month peer days. 9 new
  node tests; npm run visual clean at 390/768/1440 (gridstress page
  registered in the harness).
- PHASE B DATA-PREP SHIPPED 2026-07-07 (#362, v1.0.206, research/
  grid_vision_phaseb.md; non-GPU, stdlib-only): scripts/gridvision_etdii.py
  (CC-BY label downloader/parser — ETDII CC-BY-4.0 figshare 14935434 +
  Duke CC-BY-4.0 6931088 verified live), scripts/gridvision_chips.py
  (OSM-seeded NAIP chip-INDEX builder, corridor-restricted), scripts/
  gridvision_naip_stac.py (MPC STAC/SAS NAIP client — root verified; STAC
  `license` field returns "proprietary" placeholder, real license is
  USDA/FSA public domain, documented), datacore/gridvision/labels_manifest
  .json. 22 tests. VERIFIED US composition: 74 images → 1408 towers, 6
  substations → substation underrep CONFIRMED severe → **v0 is a TOWER
  detector**. FIRST GPU JOB pre-validated vs scripts/runpod_budget.py:
  grid-detector-v0, RTX 4090, max_hours=4, worst-case $1.36, AUTHORIZED,
  unbounded refused. RUNPOD COST-CAP GATE shipped separately (#360,
  scripts/runpod_budget.py + research/runpod_ledger.md — $50 balance,
  ledger, never-unbounded rule).
- NEXT (each its own PR): (1) V3 gate-2 design prep — subagent
  research of the complete dated public ERCOT conservation/EEA event
  list from primary sources; (2) [SHIPPED]; (3a) **RunPod fine-tune of
  grid-detector-v0** — STALE bullet, superseded by real events:
  RUNPOD_API_KEY was added to the session 2026-07-08 (Option A) and 5 GPU
  fine-tune jobs (gv-div1 through gv-div5) already ran 2026-07-08/10 —
  see this file's own later entries + research/grid_vision_phaseb.md's
  "RESULT 2026-07-10" section (diversity augmentation plateaued at 0.197
  AP50 held-out, below the 0.30 gate-1(a) bar — a real ML-progress
  question, not a launch-access blocker). This bullet was never updated
  after that and stayed "BLOCKED-FOR-MIKE" in error for ~2 weeks
  (corrected 2026-07-25; same stale text also lived in CLAUDE.md's
  KNOWN STATE, corrected there too). (3b) [SHIPPED 2026-07-17, v1.0.377]
  — see experiments.md; (3c) fetch Duke-US zips to make substations
  trainable; (4) Phase B1 VERIFY spec. Polygon arbiter for ambiguous pools: EIA
  Atlas BA layer (re-check ~late July).

As of 2026-07-12 (claude/google-maps-api-railway-9d2wwg):

- STATE ROLLOUT + AUTHORITATIVE TIER COMPLETE on /data (this branch's
  prior sessions): all 50 states + DC individual OSM grid toggles, one
  national "all states" tile (blank-tile bug fixed #407 + PMTiles magic
  CI guard), and the HIFLD authoritative trio — transmission (#405),
  substations, power plants (#408, fuel-colored, EIA-860 fields).
- LIDAR LANE PROBED AND PARKED 2026-07-12 (experiments.md entry): 3DEP
  EPT reachable + decodable free in-session, BUT the 2 collects probed
  (incl. a dedicated TL corridor survey) carry min-spec classification
  (no ASPRS 13-16) and show NO structure-band returns at OSM-confirmed
  tower sites. Tower extraction from 3DEP is UNPROVEN — do not build on
  it. scripts/gridvision_lidar_probe.py (+6 tests) makes the wider
  survey a one-command loop; the survey question is filed in
  open_questions.md. If the survey stays negative, street-view ML for
  distribution becomes the top remaining-gap modality.
- LIDAR LANE CLOSED (same day, 2026-07-12): 24-collect stratified survey
  = 1 hit (CO_UpperColorado_2020, class 14/13, zero class 15), and the
  hit's class-14 points FAIL linearity (blobs, ratio 1.7 vs >>10 for
  wires), match neither HIFLD nor OSM lines, and don't follow roads —
  vendor label unreliable. Street-view ML is the top remaining
  distribution-gap modality (costed; wishlist + GPU budget territory).
