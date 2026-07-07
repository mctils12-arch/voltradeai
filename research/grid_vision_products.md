# GRID VISION — Phase A2: Grid Data Products Plan

> Filed 2026-07-07 per the GRID VISION directive: "the products
> define the spec, not the reverse." This plan enumerates every
> credible use of the mapped grid — each with hypothesis, consumer,
> and build order — and derives the Phase B spec requirements from
> them. Charter: research/grid_vision.md. Research inputs:
> research/grid_vision_research.md (Items 1–4 all filed same day).
> Trading uses are GATE-LOCKED: nothing trades until its ladder
> gates pass; priors stated here before any testing (REASONING
> STANDARD #10).

## The license-driven product boundary (governs everything below)

Established in grid_vision_research.md Item 2.8: OSM geometry is
ODbL — a published database that corrects/extends OSM **as a
database** must stay ODbL (attribution + share-alike). Three IP
classes therefore exist, and the provenance tags the charter already
requires (`osm-verified` / `ml-extended` / `ml-discovered`) are ALSO
the IP boundary:

1. **OSM-derived geometry** (osm-verified, and ml-extended features
   that are corrections of OSM ways): publishable only as ODbL.
   Sellable as a service (hosting, tiles, API convenience) but the
   database itself must remain open — pricing is for access
   convenience, not exclusivity. ODbL legal call → BLOCKED-FOR-MIKE
   before any sale.
2. **Independently-derived detections** (ml-discovered from
   public-domain NAIP, not traced from OSM): ours outright.
3. **Derived indices and alerts** (grid-stress index, buildout
   events): produced works computed ACROSS databases — generally
   outside ODbL share-alike; cleanest commercial surface. Attribution
   still carried (honesty standard anyway).

Consequence: the SUBSCRIPTION SIGNALS (index/alerts) are the primary
commercial products; the raw verified dataset is a credibility/
marketing asset and an ODbL community contribution, not the revenue
line. This inverts naive "sell the map" thinking and matches
Amendment 5 (validated signal products rank above raw data products).

## A. Trading uses (consumer: the bot, via the ROOT VALIDATION LADDER)

### A1. Grid-stress signal — THE FIRST BUILD (needs NO ML)

- HYPOTHESIS: localized transmission scarcity — regional demand
  (EIA-930, already archived) approaching the mapped import/transfer
  capacity of a region's corridors (OSM voltage/circuits, already in
  the TX pilot data) under weather extremes (NOAA/NWS + CPC degree
  days, already archived) — predicts (a) spot-price/volatility
  episodes that move regulated-utility and merchant-generator
  tickers, and (b) demand-response/curtailment news cycles that move
  energy-intensive industrials (steel, aluminum, datacenter REITs).
- PRIOR (stated before testing): weak-to-moderate. ERCOT scarcity
  episodes are partially anticipated by markets (weather forecasts
  are public); our edge is the JOIN specificity (which corridors,
  which counties, which listed facilities) not the event itself.
  Expect gate-2 to show modest lead only on the facility-exposure
  screen (A3), not the headline index.
- LADDER PATH: gate 1 = mapped corridor capacity vs authoritative
  reference (EIA/HIFLD/ISO published ratings) on a TX sample; gate
  2 = index vs realized ERCOT LMP spreads / EIA-930 interchange
  stress, out-of-sample; base rate = same-period random-day control.
- BUILD ORDER: FIRST. Every ingredient is already archived (OSM TX
  pilot + EIA-930 + weather). No detection model needed. This is
  also the cheapest end-to-end test of the whole product spine.

### A2. Buildout detection — leading indicator

- HYPOTHESIS: new substations/corridor construction visible in
  imagery precede (by quarters) datacenter/industrial capacity
  announcements and the associated moves in utility capex guidance,
  datacenter REITs, and regional industrial names. Grid work is
  long-lead and hard to hide from overhead imagery.
- PRIOR: moderate for direction, weak for tradeable timing —
  interconnection queues (public) already leak much of this; our
  edge is EARLIER visibility (construction vs filing) and JOIN to
  tickers via the Everything Graph.
- LADDER PATH: gate 1 = detected new-features vs dated ground truth
  (interconnection queue completions, EIA-860 in-service dates);
  gate 2 = event study on the joined tickers.
- BUILD ORDER: after Phase B2/B3 detection + D6 first-seen dating
  (needs imagery-date honesty: bounded windows, never guessed).

### A3. Outage/constraint-exposure screens

- HYPOTHESIS: listed companies whose facilities (Everything Graph +
  GEM GIPT units + EIA-860 plants) hang off single-corridor or
  known-constrained grid segments carry measurable weather-event
  beta the market prices slowly.
- PRIOR: weak; this is a risk-factor screen more than an alpha
  signal — value may be as a SIZING/RISK input, which the ladder
  handles at gate 4 rather than gate 2.
- BUILD ORDER: after A1's region/corridor attribution exists;
  detection not required (OSM topology suffices to start).

### A4. Retirement / interconnection-pipeline signals

- HYPOTHESIS: joining the mapped grid to retirement schedules
  (EIA-860) and ISO interconnection queues yields regional
  scarcity-trajectory features for the regime/ML loop (slow-moving,
  quarterly refresh).
- PRIOR: weak-moderate as standalone; likely value is as FEATURES
  into ml_model_v2 regime conditioning, not a standalone strategy.
- BUILD ORDER: data joins only (no ML, no detection); any session
  after A1.

## B. Data products (consumer: external customers via /data + /api/v1)

### B1. Grid-stress index + buildout-alerts subscription API — PRIMARY COMMERCIAL SURFACE

- WHAT: regional (BA/ISO, then county) stress index computed from
  the A1 join; event stream of detected buildout (A2) with
  confidence + provenance + imagery-date bounds on every record.
- IP CLASS: 3 (produced works) — cleanest to sell.
- HONESTY SPEC (Amendment 5c): every value carries freshness,
  provenance, confidence; index methodology page is public; "would a
  paying data customer screenshot this and trust it?"
- BUILD ORDER: index v0 after A1 gate 1; alerts after A2's
  ingredients exist. API rides the existing /api/v1 scaffolding.

### B2. Verified US grid dataset (ODbL)

- WHAT: the state-by-state verified/extended grid — per-feature
  provenance, confidence, coverage manifests.
- IP CLASS: 1 (ODbL) — requires the BLOCKED-FOR-MIKE legal call
  before any paid offering; default posture is open publication +
  OSM give-back (charter rule), with paid convenience access only.
- VALUE EVEN IF NEVER SOLD: credibility engine for B1, the
  demonstration that the platform's verification machinery is real,
  and the substrate all trading uses stand on.
- BUILD ORDER: accumulates through Phase C; per-state artifacts from
  the first rollout state onward.

### B3. Per-state coverage reports

- WHAT: public, dated, per-state statements — % corridors verified,
  detection metrics vs labeled benchmarks, imagery vintage
  distribution, known gaps with revisit triggers.
- IP CLASS: 3. Free (marketing + honesty proof, Amendment 5:
  "the honesty machinery, surfaced beautifully, is simultaneously
  the brand and the proof of accuracy").
- BUILD ORDER: template lands with the FIRST Phase C state and is a
  required rollout artifact thereafter (charter Phase E manifest).

## C. Platform uses (consumer: the site + the graph)

### C1. Alerting

New-feature detections, stress threshold crossings, coverage-manifest
changes — surfaced on /data and as API webhooks later. Rides B1
events; no separate build.

### C2. Everything Graph enrichment

Substations/corridors become graph nodes/edges joined to: GEM GIPT
units (182k, already ingested), EIA-860 plants (already ingested),
OSM operator tags, GEOT entity crosswalk (SEC-CIK — already
ingested). This is what makes A2/A3 joins one query instead of a
project each. BUILD ORDER: schema lands with A1's region attribution
work (shared spine).

## The build order, consolidated (products → spec, no ML on the critical path to first revenue-grade product)

1. **A1 grid-stress ingredients + region attribution** (OSM TX +
   EIA-930 + weather; gate-1 vs published ratings) → unlocks B1
   index v0, A3, A4, C2 schema.
2. **B1 index v0 on /data + /api/v1** (TX/ERCOT first, honest
   methodology page).
3. **Phase B detection per the charter** (B1 VERIFY on TX corridors
   with the GV-A2 two-layer benchmark) → feeds B2/B3 artifacts.
4. **A2 buildout events + alerts** (needs B2/B3 detection + D6
   dating).
5. **Phase C rollout states** each shipping: verified geometry (B2),
   coverage report (B3), index expansion (B1).

## Spec requirements Phase B MUST satisfy (derived from the products)

- Stable per-feature IDs (alerting diffs; graph joins).
- Required fields: provenance tag, confidence, voltage class,
  operator (where tagged), region (BA/ISO + state + county),
  first-seen/imagery-date bounds (three honesty states per charter
  D6), source imagery id+date.
- Per-state coverage manifest machine-readable (Phase E ratchet).
- IP-clean pipeline: detection training/eval only on the
  license-safe stack (Duke-US CC-BY, TTPLA Apache, NAIP PD, OSM
  ODbL-attributed); Esri tiles never feed model training or sold
  geometry; fine-tuned weights stay internal (CC-BY-SA inheritance).
- Every published record joins back to evidence (which imagery,
  which model rev, which OSM object) — the audit trail IS the
  product differentiation.

## BLOCKED-FOR-MIKE items this plan creates

1. **ODbL legal review** (before ANY paid offering of B2 geometry —
   not needed for B1 indices/alerts): one-time counsel review of the
   ODbL produced-work boundary for our exact product shapes. Not
   urgent; file when B2 approaches sale.
2. RunPod GPU purchase order — FILED in wishlist.md (from GV-A4,
   research doc Item 4.8): $50 initial deposit, RUNPOD_API_KEY in
   Railway. Needed for detector TRAINING regardless (no released
   weights exist); full-state sweeps later. The A1→B1-index path
   above deliberately needs no GPU meanwhile.
