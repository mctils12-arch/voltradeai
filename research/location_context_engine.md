# Location Context Engine — the "Zillow for everything" vision

Human-directed 2026-07-11. This is a product-direction note (roadmap +
honesty rails), NOT a constitutional amendment. It operationalizes the
existing CROSS-SYSTEM INTEGRATION PRINCIPLE + Everything Graph for the
/data world viewer.

## The thesis (human's analogy)

Zillow doesn't sell you a house — it aggregates *context around a location*
(school ratings, crime, flood risk, price history) so you can *reason* about
a decision. Our world viewer should do the same for ANY point on Earth: click
a spot and get everything we know about it, synthesized, so a human (or the
bot, or a paying customer) can reason. The map is the door; the dossier is the
value; the AI is the guide that reads it back to you.

Today the layers coexist but don't *interact*. The value shows up when they
combine: "this factory + upwind of a Superfund site + a spike in its power
draw + a recent EPA water violation" is a story no single layer tells. The
Everything Graph (server/entityGraph.ts) is the connective tissue; this note
is the location-first face of it.

## The build shape

1. **Click anywhere → a location dossier** (extends the click-to-identify
   shipped in #429). Instead of just naming the place, gather every layer
   within a radius and return a structured "what's around here": named place,
   nearby facilities (power plants, ports, mills), hazards (below), live
   activity (aircraft/vessels/trains passing), grid region + its stress
   reading. A server endpoint does the spatial gather; the AI pane summarizes.
2. **Hazard/context layers** — each its own toggle AND a dossier input.
3. **The dossier is the cross-join** — the thing customers can't get elsewhere.

## Candidate datasets — all FREE + OWNABLE (EDGE DOCTRINE: build, don't buy)

Priority order (national point/vector data we can tile + archive ourselves):

- **EPA Superfund / NPL sites** — contaminated-site cleanups (EPA FRS/SEMS,
  public domain). Point layer + status.
- **EPA FRS + ECHO** — Facility Registry (millions of regulated facilities)
  and ECHO enforcement/compliance (violations, inspections, penalties). This
  is the big one: factories + their environmental compliance record.
- **EPA water discharge (DMR / ICIS-NPDES via ECHO)** — permitted dischargers
  + reported pollutant loads. The "factory water quality" case.
- **Water Quality Portal (USGS + EPA)** — ambient water-quality samples.
- **NRC** — nuclear power reactors + status (also in our HIFLD plants layer;
  cross-link, don't duplicate).
- **EPA RadNet** — environmental radiation monitoring stations + readings.
- **Nuclear test sites** — historical (NNSA/DOE + curated); mostly static
  reference points (Nevada Test Site, Bikini, Semipalatinsk, etc.).
- **FEMA flood zones / NFHL** — flood risk (the closest direct Zillow parallel).
- **PFAS** — EPA UCMR 5 + ECHO/SDWIS drinking-water PFAS detections
  ("forever chemicals"). Point/system data, public domain.
- **Cancer rates** — CDC US Cancer Statistics + NIH/NCI SEER, county-level
  incidence/mortality (public domain). NOTE: county cancer rates are an
  AGGREGATE STATISTIC, never a claim about a specific address — display at the
  county polygon, never implied down to a point (ecological-fallacy guard).

We already have: power plants (HIFLD/WRI), power grid (OSM+HIFLD), grid stress
(EIA-930), strategic sites, ports, live aircraft/vessels/trains, GeoNames
places, weather, NASA environmental fields.

## THE "ALL LAYERS" LAYER = the dossier

The human's "a layer of all the layers" is the location dossier itself: not a
51st toggle, but the CROSS-JOIN — click a point, gather every layer near it,
present them together. That aggregate is the product; the individual layers are
its ingredients. It updates as its ingredients update.

## DATA QUALITY GATE (human-directed 2026-07-11) — inaccurate inputs must not skew the join

This is Priority 2 (protect the integrity of learning) + ROOT VALIDATION LADDER
gate 1 (DATA), applied to EVERY layer before it feeds the dossier or the graph.
One bad dataset silently skewing a combined reading is the worst failure mode
here — worse than a missing layer. Rules:

1. VALIDATE AT INGEST. Each layer's loader runs plausibility checks before the
   data is stored/served: geometry in valid lat/lon range; values in physical
   bounds (no negative populations, MW, or concentrations; radiation/PFAS
   within known scales); required fields present; units as declared.
2. CROSS-SOURCE TRUTHING where a second source exists (ladder gate 1's
   "verified against an external truth source"): e.g. HIFLD plant count vs EIA;
   EIA US48 daily sum vs EIA's own Grid Monitor. Divergence beyond tolerance
   flags the layer, it does not silently pass.
3. QUARANTINE, DON'T PROPAGATE. A record that fails validation is dropped or
   flagged `suspect`, never fed into the dossier/graph as if clean. A layer
   failing its checks degrades to "unavailable/stale" in the UI, honestly —
   never shown as fresh-and-fine.
4. STALENESS IS A DATA ERROR. Every layer carries a source date + expected
   cadence; past its freshness budget it is marked stale (not silently served),
   and a monitor surfaces it (mirrors the /api/health degraded-state pattern).
5. OUTLIER FLAGGING. Sudden step-changes vs a layer's own recent history are
   flagged for review before they can move a combined reading — a corrupt feed
   often shows as an impossible jump.
6. PROVENANCE ON EVERY VALUE. Source + date + validation status ride with each
   datum to the UI (PREMIUM EXPERIENCE STANDARD (c)); the dossier shows how
   fresh and how trusted each ingredient is.

## CONSTANT UPDATING (human-directed 2026-07-11)

Every layer has an owning server-side refresh on the Railway scheduler (NOT
GitHub Actions) at a cadence matched to its source (hourly for grid demand,
daily/weekly for EPA/EIA registries, on-release for annual stats). Refreshes
are archive-append (history accrues) + dedup, exactly like griddemand. A failed
refresh flags the layer stale (rule 4) rather than serving nothing or serving
old data as current.

## Honesty rails (non-negotiable — RAW vs SIGNAL rule + the ladder)

- A site's **location/status/violation record is RAW/FACTUAL** — display as-is
  with source + date. "Superfund site here, listed 1987" is a fact.
- Any **risk/impact interpretation is a SIGNAL** and must clear ROOT VALIDATION
  LADDER gate 2 before it's stated as a claim. We NEVER render "this raises
  cancer/property risk" or a composite "safety score" as if validated —
  premium presentation of an unvalidated risk number is fraud with good
  typography (Amendment 5c). The dossier shows the facts and their provenance;
  the reasoning is labeled as reasoning, not a verified score.
- Every dossier element carries freshness + source (PREMIUM EXPERIENCE
  STANDARD (c)).

## Trading tie (real, not forced)

Some of these are genuine signal candidates for the bot too: ECHO violations /
new Superfund listings near a small-cap's facilities, discharge anomalies, etc.
Those enter the ladder as hypotheses in open_questions.md — they are NOT
assumed tradeable here. This note is about the PRODUCT (the dossier); trading
use is separate and gated.

## Next concrete step

Ship the first hazard layer (recommend EPA Superfund or FRS+ECHO — free,
national, ownable) as a map layer AND a dossier input, then build the
click-anywhere dossier endpoint. Each is its own PR under the usual rules.

STATUS 2026-07-12: the click-anywhere dossier endpoint SHIPPED (v1.0.287)
as a `hazards` cross-join on the existing `/api/data/dossier` (W5 ENTITY
DOSSIER v2) route rather than a new endpoint — same anchor lat/lon, same
warming-up honesty, one less surface for the client to call. It reports,
within `radius_km` (default 50, max 200) of the clicked point: nearby EPA
Superfund NPL sites, EPA Clean Water Act chronic violators, historical
M6+ earthquakes, and historical nuclear tests — each capped at 10 nearest
with an honest `total_within` count (never silently truncated) and a
`ready` flag so a cold cache (superfund/water-violators poll async) shows
nothing rather than a false "0 nearby, all clear". Wired into datamap.tsx's
existing dossier card (every click handler already sends lat/lon). NOT yet
built: PFAS, RadNet, FEMA flood, CDC/SEER cancer layers (still queued,
each its own ladder-gate-1 pipeline before it can join this cross-join);
a UI toggle to adjust radius_km client-side (currently server default
only); the harness still can't drive a live map click, so the actual
card rendering was code-reviewed + unit-tested, not screenshotted with
real hazard data in view (same limitation the O7/O3 sessions logged).

CORRECTION 2026-07-13: RadNet is NOT a gap — the 2026-07-12 nuclear-wave
session's "ambient radiation" layer (#456, v1.0.293) already includes EPA
RadNet as one of its four national networks (~76 US stations). This file's
"NOT yet built" list above was stale on that point; RadNet was done a
session before this correction, just not cross-referenced back here.

STATUS 2026-07-13: FEMA FLOOD ZONES SHIPPED (hazard layer #3, this
session, T-CLIENT + T-DATACORE) — two parts:
1. MAP OVERLAY: `floodzones` toggle in the hazards group — a raster tile
   source hitting FEMA's own public NFHL MapServer `export` operation
   live via MapLibre's `{bbox-epsg-3857}` template token (confirmed CORS-
   open, confirmed reflects request Origin). Zero server cost/code, same
   "someone else's public tile service" pattern as surfacewater/forest.
   Only renders at roughly property-level zoom (FEMA's own scale limit).
2. DOSSIER POINT LOOKUP: `server/femaFlood.ts`'s `floodZoneAt(lat, lon)` —
   unlike the radius-list hazard categories, "is THIS point in a flood
   zone" is a point-in-polygon query at the exact anchor, so it's awaited
   server-side (routes.ts) before calling the still-pure/sync
   `buildDossier`, then rendered as its own `flood_zone` dossier section
   (zone code, FEMA's own SFHA_TF field — never inferred from the zone
   code ourselves — base flood elevation with FEMA's -9999 no-data
   sentinel converted to null, and FEMA's own plain-English zone-code
   glossary as the `meaning` text). A point outside NFHL's mapped
   footprint (confirmed live: zero features over interior Alaska) reports
   honestly as unmapped, never as "minimal risk". Per-point cache (30-day
   TTL, ~1km grid) keeps repeat/nearby clicks fast and is a good citizen
   of FEMA's public service; failed lookups are never cached (retry next
   click, don't freeze an outage into "unavailable" forever).
Remaining hazard layers still queued: PFAS (EPA UCMR5/SDWIS), CDC/SEER
cancer rates (needs the county-polygon ecological-fallacy display guard
noted above — do not ship as a point layer). radius_km client toggle
still not built. Live map-click screenshot verification still blocked on
the harness limitation noted above.
