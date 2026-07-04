# VISION.md — the platform north star

> **PROVENANCE (read this first):** the human authored the platform
> charter on 2026-07-04, but the verbatim text did not survive into the
> directive (the paste placeholder arrived unfilled). This document is a
> session RECONSTRUCTION from the directive's own enumeration, installed
> so the north star exists on day one rather than waiting. **Human: paste
> the original charter text into a session and it replaces the vision
> statement below verbatim; the reconciliation annotations survive.**
>
> Precedence: CLAUDE.md is the constitution and always wins. VISION.md
> answers WHAT the platform is becoming; CLAUDE.md governs HOW every
> change ships (ladder, promotion rules, honesty standards). PRODUCT and
> EDGE sessions read this file immediately after CLAUDE.md.

## The vision

One system that ingests every free, lawful, honest data stream about the
physical and corporate economy — what moves (aircraft, ships, trains),
what stands (facilities, plants, buildings), what companies do (filings,
hiring, patents, code, earnings language, app traction), and what the
planet does around them (weather, fire, drought, water, crops) — archives
all of it forever, links it into one entity graph, validates every
interpretation through the ladder, trades the validated signals with the
bot, and surfaces the whole thing as a product a customer could buy.

The compounding asset is the accumulation: archives nobody can back-buy,
pipelines nobody bills us for, verified reference data, and the graph
that joins them. Every day not recorded is unrecoverable; every insight
not compiled into code is wasted.

## Honest-scope rule (standing)

Anything on this list with no free lawful source is marked
**BLOCKED-BY-ACCESS** with its build-first analysis, so no session chases
ghosts. Currently blocked: **credit/debit card panels** (consumer spend
ground truth — sold only as paid panels; no free raw material exists),
**private fleet telemetry** (trucks — fleet telematics is private;
build-first analysis filed in open_questions.md FREIGHT-ACTIVITY PROXIES,
proxies queued instead), **sub-meter imagery object counting** (cars,
coils, railcars — raw material is paid; the free 10m version is
facility-scale change detection, never object counts), **mid-ocean
satellite AIS** (declined 2026-07-04; revisit only if a gated signal
specifically requires open-ocean coverage), **filed flight plans**
(AeroAPI class — predicted-destination substitute built instead), **US
freight-rail positions** (proprietary — logged so nobody chases it).

## Pillars, reconciled against reality (2026-07-04)

Status key: **DONE** (mechanism live, cited) · **IN-PROGRESS** (partially
live) · **QUEUED** (filed with owner/roadmap slot) · **NEW** (filed by
the charter-gap execution) · **BLOCKED-BY-ACCESS** (no free lawful
source; build-first analysis attached where it lives).

### 1. Collect everything, forever

- **DONE — archive doctrine**: streamed layers → JSONL position/filing
  archives (aircraft, vessels, trains, Form 4); static reference →
  git-versioned datacore/ (sites, power plants, shadow zones); derived
  stats recomputable, never archived (COLLECT-EVERYTHING AUDIT,
  open_questions.md, verified 2026-07-04).
- **DONE — spinout boundary**: datacore/ isolated from trading logic;
  the bot consumes signals like an external customer (KNOWN STATE /
  STANDING BEHAVIORS).
- **IN-PROGRESS — archive volume watch**: ~105MB/mo design figure,
  standing wishlist watch item.

### 2. Verify everything (honesty is the product)

- **DONE — root validation ladder** (CLAUDE.md): DATA → SIGNAL → LOGIC →
  SIZING → EXECUTION; RAW overlays ship ungated but fact-checked;
  SIGNALS gate at ladder 2.
- **DONE — reference-data accuracy + imagery verification**
  (DESIGN.md): 16 strategic sites + top-100 power plants
  imagery-verified; geofence coordinates verified mandatory.
- **DONE — self-see harness** (DESIGN.md): UI verifies its own
  rendering; imagery metadata honesty rule.

### 3. Movement intelligence (what moves)

- **DONE — live layers**: aircraft (3-provider licensed chain), vessels
  (aisstream, global terrestrial), trains (FI+NO, honest per-country
  coverage) — all RAW-labeled, attributed, archived.
- **IN-PROGRESS — maritime analytics core**: shadow-fleet RAW stats live
  (gap/identity/loiter events, gate-1 plan filed); **port dwell
  analytics** building now (2026-07-04 directive: arrivals/departures,
  dwell distributions, anomaly flags from OUR archive — the
  highest-immediate-value fusion, zero new data); R2 transit counters
  queued for routines.
- **QUEUED — aircraft analytics**: corporate-jet/M&A hypothesis,
  destination prediction self-scoring (open_questions.md).
- **BLOCKED-BY-ACCESS**: trucks (proxies queued instead), mid-ocean AIS
  (declined), US freight-rail positions, filed flight plans.

### 4. Corporate activity (what companies do)

- **DONE — Form 4 insider feed**: live, archived, full filings view
  (gate-1 passed; gate-2 hypothesis filed with prior).
- **NEW — earnings-call transcripts**: language/guidance deltas
  (open_questions.md entry with sources, license check, ladder path).
- **NEW — job postings**: hiring velocity/role mix (ATS public JSON
  endpoints; open_questions.md).
- **NEW — GitHub org activity**: engineering momentum on small-cap
  devtools/infra names (open_questions.md).
- **NEW — USPTO patents**: filing velocity/topic shifts (EDGE DOCTRINE
  named it 2026-07-03; now a filed entry with sources + ladder).
- **NEW — app-store rankings**: consumer-app traction (open_questions.md,
  honest about ordinal ranks + estimated downloads).
- **QUEUED — EDGAR beyond Form 4**: 8-K material events, 13F clusters
  (alphadesk/ is the seed — KNOWN BROKEN #5 audits its wiring);
  USAspending/SAM.gov government contracts (EDGE DOCTRINE standing
  example, unfiled → carried by this charter as QUEUED).
- **BLOCKED-BY-ACCESS**: card-panel consumer spend.

### 5. Geospatial / earth observation (what stands, and what changes)

- **DONE — reference layers**: strategic sites (16, verified), US power
  plants (9,833, EIA-located, top-100 verified).
- **QUEUED (Tier 1, build in order, each own PR)**: (a)
  terrain/elevation DEM (unblocks R4 3D globe); (b) live weather; (c)
  NASA FIRMS fires; (d) USDA Cropland Data Layer; (e) drought/soil
  moisture; (f) USGS groundwater (point data, labeled); (g) oil/gas
  infrastructure. All RAW, licensed-checked first, archived per
  doctrine. (Extends the R3 environmental-layers roadmap slot.)
- **NEW (Tier 2) — building intelligence**: footprint datasets queried
  per viewport (count/density/height), construction-growth hypotheses
  filed. Query, don't detect.
- **NEW (Tier 3) — 10m change detection**: Sentinel-2 weekly
  reflectance/area change at strategic sites — "activity index up/down
  at facility X," facility-scale honesty, never object counts;
  generalizes the Cushing tank-shadow idea into one system (spec:
  datacore/SENTINEL2_CHANGE_SPEC.md when filed). Sub-meter paid imagery
  enters wishlist only after the free version passes gate 2.

### 6. Fusion — THE EVERYTHING GRAPH (flagship)

- **NEW — design doc filed** (datacore/EVERYTHING_GRAPH.md): entity
  types (company, person, facility, vessel, aircraft operator),
  relationship types (insider-of, supplies, operates, located-at),
  storage fitting our stack, v1 linking ONLY what we already collect
  (Form 4 insiders ↔ tickers ↔ strategic sites ↔ facility operators).
  Graph queries become a /data page feature when v1 lands.
- **IN-PROGRESS — fusion hypotheses** (logged, not built): insider ×
  facility activity (STLD), generation shifts × utility tickers, ship
  anomalies × commodity/retail tickers (open_questions.md, each with
  gate-1 ground truth).

### 7. Product surface (/data)

- **DONE**: full-viewport map, layer panel v2 with groups, filings
  view, fullscreen mode, self-see enforced, RAW/SIGNAL labeling,
  per-source status + attribution.
- **QUEUED — dashboards** (PRODUCT roadmap): signal-strength panel,
  data-quality panel (feed freshness, archive growth, verification
  coverage), pipeline-health panel — all sourced from monitoring we
  already emit (/api/health checks, archive stats, provider status).
- **QUEUED — graph queries page** (lands with Everything Graph v1).

### 8. The trading loop stays the customer

Every pillar above terminates in the ladder and the bot: validated
signals trade in paper, live-vs-backtest divergence stays the honesty
metric, and the /data surface shows customers exactly what the bot sees
(RAW) or has validated (SIGNAL). The factory (CLAUDE.md GOAL) is
unchanged — this charter only names what the factory is building toward.
