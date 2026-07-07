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

---

## RESUME STATE (update every session that works the program)

As of 2026-07-07 (session claude/new-session-iu72vf):

- PHASE A IN FLIGHT — four research subagents launched in parallel
  (GV-A1 pipeline paper/code, GV-A2 labeled datasets, GV-A3 methods
  survey, GV-A4 imagery inventory + compute assessment). Their
  reports assemble into research/grid_vision_research.md; none have
  filed yet. NOTHING BUILDS until that doc + the A2 products plan
  file.
- Existing assets to reuse: /data powergrid TX-pilot layer (OSM
  PMTiles, v1.0.166, wiring fixed v1.0.177) is the OSM base's first
  render; scripts/build_power_tiles.sh reproduces TX tiles in <1 min
  and documents the US-full variant; EIA-930 grid demand stream is
  live (grid-stress ingredient); GEM GIPT power units (182k) +
  ownership graph are the entity-join spine; Esri capture-date
  identify contract is proven (v1.0.173).
- NEXT: (1) assemble research doc when agents return; (2) draft A2
  products plan; (3) only then spec Phase B1 VERIFY (Texas, over the
  existing TX corridor set).
