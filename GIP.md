# GIP.md — Expansion of the Global Intelligence Platform

> **PROVENANCE:** human-authored companion charter to VISION.md,
> verbatim text received 2026-07-04. PRODUCT and EDGE sessions read
> VISION.md and GIP.md after CLAUDE.md. Precedence: CLAUDE.md always
> wins — this file names WHAT to build; the constitution governs HOW
> (ladder gates, RAW-vs-SIGNAL surface rules, licensing-first,
> honesty standards). The reconciliation annex at the bottom is
> session-maintained; the charter text itself is verbatim and edited
> only by the human.

## Core Philosophy

The objective is not simply to collect data; public data already exists
in thousands of places. The objective is to create entirely new
intelligence by combining observations from multiple independent sources
and allowing AI to make inferences that no single dataset contains. The
platform continuously asks: what can be observed, measured, counted;
what has changed; how confident are we; can the conclusion be verified
with independent evidence; does this observation have implications for
another industry, company, or financial market. The AI constantly
generates new hypotheses and tests them automatically.

## Inference Engine

Every piece of information becomes an inference rather than a database
record.

## Aircraft

Using ADS-B data, registration databases, maintenance records, flight
history, ownership records, news, and government registration
information, determine ownership changes, registration changes, ICAO hex
changes, tail number changes, fleet transfers, leasing activity,
retirement, storage, reactivation, maintenance trends, flight frequency,
utilization, route changes. When an aircraft changes ownership,
determine whether the hex, registration, operator, or owner changed,
whether it was exported or re-registered — and preserve ONE continuous
historical identity with a timeline of every significant event; clicking
an aircraft reveals its complete life history.

## Maritime Intelligence

Track every vessel, every ownership/flag/name/MMSI change, AIS gaps,
dark vessels, sanction-avoidance behavior, unusual routing, cargo
movements, fleet relationships. Using satellite imagery, identify
physical characteristics — floating-roof tanks and tankers whose roof
position indicates fullness; computer vision estimates roof height,
liquid volume, capacity, percent utilization; every image timestamped;
compare over time (e.g., Tank A: Jan 1 82%, Jan 6 77%, Jan 11 63%,
Jan 16 48%); automatically calculate volume changes, estimate
inflows/outflows, correlate with shipping activity, refinery operations,
government inventory reports, and commodity prices; continuously improve
estimates from historical observations.

## Satellite Intelligence

Detect new homes, demolished buildings; count vehicles; measure parking
occupancy; count aircraft and ships; measure container-yard utilization;
detect road construction, mining expansion, new factories, warehouse
construction; estimate crop health; detect flooding; measure wildfire
damage; detect airport expansion; measure solar farms, wind farms, oil
storage, rail-yard activity, port congestion — every observation becomes
structured data rather than remaining an image.

## AI Verification

The AI never blindly trusts itself; every inference receives a
confidence score; observations are validated with multiple independent
sources (e.g., a detected subdivision verified via local permits, news,
county GIS, property records, press releases, utility permits) and only
after multiple confirmations becomes high-confidence; earlier
conclusions are continually re-evaluated as new evidence arrives.

## Autonomous Dataset Creation

Automatically create datasets that don't exist — global aircraft
ownership history, aircraft utilization history, oil storage estimates,
parking occupancy, warehouse construction, housing development, airport
expansion, shipping congestion, infrastructure growth, power plant
utilization, refinery activity — proprietary assets sellable through
APIs, dashboards, enterprise subscriptions, and used internally by the
trading platform.

## User Interface Philosophy

The satellite page becomes an interactive intelligence workspace — live
and historical imagery, AI-observation overlays, toggleable intelligence
layers, per-inference confidence scores, multi-date timeline slider,
animated change detection, click-any-object for complete
history/evidence/news/companies/market implications, filtering by
geography/date/type/confidence/industry; new AI modules appear as
collapsible layers in the same interface, modular so hundreds of
intelligence products coexist without overwhelming the user.

## Long-Term Vision

A continuously learning digital representation of the physical world —
every aircraft, ship, building, warehouse, refinery, tank, rail yard,
port, airport, and construction site develops a permanent history; the
AI transforms raw observations into verified intelligence, continuously
asking what changed, why, how confident, and whether it matters to
finance, logistics, aviation, maritime, insurance, energy, governments,
and research; the trading platform is ONE consumer of this intelligence
while the Global Intelligence Platform becomes a standalone product
serving many industries through APIs, enterprise dashboards, and custom
analytics.

---

# Reconciliation annex (session-maintained; 2026-07-04)

Status key: DONE (live, cited) · IN-PROGRESS · SPEC'D (design filed) ·
NEW (filed by this directive) · BLOCKED (no free lawful path — with the
honest reason).

- **Core Philosophy / hypothesis loop** — IN-PROGRESS as constitution:
  the ROOT VALIDATION LADDER + open_questions.md hypothesis registry +
  AUDIT CYCLE are the manual form; autonomous hypothesis generation is
  NEW (far-roadmap).
- **Inference envelope ("every piece of information becomes an
  inference")** — SPEC'D → APPROVED 2026-07-04 (Part 0b): the universal
  archive envelope {timestamp UTC, source, confidence, geo,
  entity/ticker linkage, sentiment} — dataset manifests + datum-level
  fields on new pipelines; Everything Graph edges already carry
  {source, confidence, first_seen, last_seen}
  (datacore/EVERYTHING_GRAPH.md).
- **Aircraft continuity spine** — IN-PROGRESS (directed 2026-07-04
  Part 3b): FAA registry × our ADS-B archive (datacoreArchive.ts,
  recording since 2026-07-03) × hex/tail cross-references → one
  identity per airframe with a life timeline. Maintenance records:
  BLOCKED as bulk data (no free lawful source; FAA SDR safety reports
  are the partial free proxy — NEW, unfiled). International registries:
  research pass running (Part 5a).
- **Maritime identity + dark vessels** — IN-PROGRESS: shadowFleet.ts
  ships gap events, identity candidates, loitering (RAW counts,
  gate-1 plan vs OFAC/KSE lists in open_questions.md); ownership/flag
  history needs the Part 5c reference sources (research running).
  Cargo movements: BLOCKED at manifest level (customs data is paid;
  port-call inference is the free substitute — port dwell live
  v1.0.60).
- **Tank fullness (the charter's Tank A timeline)** — IN-PROGRESS:
  SENTINEL2_CHANGE_SPEC.md + scripts/sentinel2_tankfill.py running
  (36 readings archived; first EIA comparison logged honestly, gate 1
  not yet earned). The charter's per-tank percent-utilization
  timeline is exactly the spec's per-tank annulus iteration — NEXT.
  Computer-vision roof-height estimation at 10m is facility-scale
  only; per-tank precision improves with the annulus geometry, and
  sub-meter imagery remains gated on gate-2 (spec).
- **Satellite object counting (vehicles, aircraft, ships, parking)** —
  BLOCKED at free 10m resolution: counting needs sub-meter (paid;
  priced classes in SENTINEL2_CHANGE_SPEC.md). The lawful free version
  is facility-scale CHANGE DETECTION (activity indices) — SPEC'D and
  running at Cushing. Construction/expansion detection at 10m scale:
  NEW, feasible for large footprints (warehouses, solar/wind farms) —
  ladder path via building-permit verification (Part 5b research).
- **AI verification (multi-source confirmation)** — SPEC'D in
  miniature: the ladder's DATA gate IS multi-source verification
  (registry agreement ≠ verification — the Hardeeville lesson);
  per-county permit verification is PER-TARGET toil, not global
  (Part 5b measures which metros have open portals). Continuous
  re-evaluation: the envelope's confidence field + re-run pipelines.
- **Autonomous dataset creation** — IN-PROGRESS by accumulation:
  position archives (aircraft/vessels/trains since 2026-07-03), Form 4
  + 8-K language archives, fires detections, port-dwell stats,
  Sentinel-2 readings — each is a dataset that didn't exist. Sellable
  APIs: the SPINOUT-READY boundary (datacore/) exists for exactly
  this; monetization stays behind the tripwire until the human flips
  it.
- **UI workspace (hundreds of layers, timeline, confidence)** —
  IN-PROGRESS (directed 2026-07-04 Part 4): registry-native lazy
  loading/virtualization/cost budgets + timeline-slider and
  confidence-display capabilities; multi-date imagery architecture
  research running (Part 5d). Today: 16 layers, groups, self-see
  enforcement, zero-cost-when-off.
- **Long-term (standalone product)** — matches the SPINOUT-READY DATA
  LAYER standing rule verbatim (spinout trigger = human decision on
  gate-2 pass + demand).