# Static-Layer Position Audit — 2026-07-18

[RESEARCH] · T-DATACORE (data verification; no client code touched).
Re-run of the 2026-07-16 wind-farm position-verification agent whose
output was lost before integration (its ephemeral container was
reclaimed; `research/layer_verification_audit.md` never existed — see
experiments.md 2026-07-16 OPS LESSON). Trigger: human report
(2026-07-17, screenshots) of a wind-farm symbol on /data where the
satellite imagery shows no windmill; directive "we need to verify this
all over our layers". Human reference case: **Waverly Wind Farm (KS)**.

## Verdict in one paragraph

The human was right, and it is a DATA problem, not an imagery-lag
problem. In a 60-plant OSM cross-check (40 wind + 20 other fuels),
**4 wind-plant coordinates are materially wrong (5–22 km off)** —
including Waverly Wind Farm (KS), whose catalogued point sits ~17 km
from the nearest turbine and ~22 km from the farm's OSM centroid. All
four wrong records are wind; all 20 other-fuel samples either matched
OSM (18, median error 80 m) or were absent from OSM in ways that do
not implicate our coordinate (2). The error source is the registry
chain itself (GPPD ships the same wrong values — verified against the
GPPD v1.3.0 CSV directly), confirming the 2026-07-04 "Hardeeville
lesson": registries share self-reported geocodes; cross-agreement is
not verification. The four records are fixed in a separate data
commit with OSM-derived coordinates (two independently corroborated by
Global Energy Monitor), applied through a new
`datacore/powerplants/position_overrides.json` consumed by the build
script so rebuilds keep the fixes.

## Method

- Sample: 60 of 9,833 plants from
  `datacore/powerplants/us_power_plants.json` — 40 wind (2 forced:
  the human's Waverly KS case + Waverly Community Wind IA, which
  shares its exact coordinate with its owner's fossil plant; 8 top
  wind by MW; 30 longitude-stratified across the CONUS, lon −121.8 to
  −71.6, 14 distinct 3° longitude bins) and 20 other fuels (gas 5,
  solar 4, hydro 3, coal 3, nuclear 2, oil 2, other 1; including 5
  imagery-verified top-100 plants as positive controls). Deterministic
  seed 20260718.
- Ground truth: OpenStreetMap via the public Overpass API
  (overpass-api.de, custom User-Agent, batched — one bulk query per
  12 plants, 5-batch total, polite sleeps). Match = any
  `power=plant` / `power=generator` feature within **3 km**
  (haversine; way/relation distances to `out center`). Wind samples
  additionally require a wind-tagged feature (`plant:source=wind`,
  `generator:source=wind`, or `generator:method=wind_turbine`) so a
  nearby substation or fossil plant cannot fake a confirmation.
- Failures got two follow-up probes each: a 25 km fuel-consistent
  scan (finds the real farm if our point is a displaced centroid) and
  a 150 km name search (the class-(a) test: is the plant findable in
  OSM elsewhere?).
- Corrections: centroid of the OSM plant relation's member nodes
  (95/87/55/3 nodes for the four fixes), cross-checked against Global
  Energy Monitor `datacore/gem/power_units.json.gz` where the plant
  exists there.
- OSM data timestamp: 2026-07-18T01:28Z. All distances haversine, km.

## Per-layer coordinate provenance (task 1)

| Layer | File | Served at | Coordinate source | Prior verification |
|---|---|---|---|---|
| US power plants (9,833) | `datacore/powerplants/us_power_plants.json` | `/api/data/powerplants` (routes.ts:1319) | WRI GPPD v1.3.0 (CC BY 4.0), EIA-860 coords preferred where matched (`scripts/build_powerplants.py`) | Top 100 by MW imagery-verified 2026-07-04 (`imagery_verified.json`); **0 of 1,139 wind plants were in that verified set** |
| Nuclear facilities (67) | `datacore/nuclear_facilities.json` | routes.ts import line 18 | Wikidata P625 (CC0), fuel-cycle/production tier of the Q1739545 class tree, compiled 2026-07-12 | None (site-scale entities) |
| Strategic sites (16) | `datacore/sites/strategic_sites.json` | `/api/data/sites` (routes.ts:1309) | Hand-curated; ALL 16 imagery-verified 2026-07-03 via `scripts/site_verify.py` crosshair sheets (that audit found 11/16 mispositioned and fixed them) | Full, 2026-07-03 |
| Military installations (3,024) | `datacore/military_installations.json` | `/api/data/military_installations` (routes.ts:2462) | OpenStreetMap (ODbL) — source field = "OpenStreetMap" for **all 3,024** (`scripts/military_installations_build.py`, retrieved 2026-07-17) | N/A — an OSM cross-check would be circular (coords ARE OSM); independent check would need imagery or DoD lists |

## Power-plant OSM cross-check results (tasks 2–3)

Counts, n=60:

| Class | Count | Definition |
|---|---|---|
| CONFIRMED (≤3 km fuel-consistent OSM match) | 53 | median 0.08 km, p90 0.94 km, max 2.06 km |
| CONFIRMED-borderline | 1 | Cloud County Wind Farm — see below |
| (a) coordinate materially wrong | **4** | >3 km from any matching feature AND plant found in OSM elsewhere |
| (b) absent from OSM | 2 | not proof we're wrong — stated explicitly below |
| (c) pure imagery-lag candidates | 0 among failures | every confirmed plant is case (c) if screenshotted over stale imagery |

All five imagery-verified controls matched at 0.03–0.23 km (Crystal
River, John Day, Gibson, Dresden, Oswego Harbor). Wind: 35/40
confirmed; other fuels: 18/20 confirmed, 0 wrong.

### Class (a) — the four wrong records (all wind)

1. **Waverly Wind Farm LLC** (KS, 199 MW, GPPD `USA0057614`) — the
   human's reference case. Ours: 38.2569, −95.8142. No wind feature
   within 3 km; nearest turbine ("Unit 63", node/3879070054) **16.9 km
   NE**; OSM `power=plant` relation/13566614 "Waverly Wind Farm"
   (plant:source=wind) centroid **22.0 km** away; the farm's own
   substation and the "Waverly Wind Farm – Waverly Switching" HV line
   are all in the same NE cluster. GEM places "Waverly wind farm"
   (199.0 MW) 24.9 km from our point and 3.1 km from the OSM centroid
   — two independent sources agree we are wrong. Corrected to
   **38.3513, −95.5924** (centroid of the relation's 95 member nodes).
   The human's screenshot was over empty Kansas farmland ~17+ km from
   the real turbines: catalogued position WRONG, imagery fine.
2. **Waverly Community Wind Project** (IA, 2.7 MW, `USA0057214`) —
   ours: 42.7317, −92.4711, which is 0.46 km from OSM's "South Power
   Plant" (**plant:source=oil**, operator Waverly Utilities) and
   IDENTICAL to our own record for "Waverly Municipal Electric North
   Plant" (oil) — a textbook owner-address geocode. The actual OSM
   relation/14151897 "Waverly Community Wind Project" is **5.8 km E**;
   its turbines 4.3–8.4 km. A wind symbol here renders over a fossil
   plant. Corrected to **42.7435, −92.3919** (member-node centroid).
3. **Scurry County Wind LP** (TX, 130.5 MW, `USA0056506`) — ours:
   32.7204, −100.9952. No wind feature within 3 km (the unnamed
   turbines 4.6 km SW belong to a different farm); OSM
   relation/6903041 "Scurry County Wind" is **14.7 km ENE**, beside
   the Camp Springs Wind substations. Our separate "Scurry County
   Wind II" record (32.7181, −100.7933) is ~2 km from OSM's phase-II
   relation — phase II is fine, only phase I is displaced. Corrected
   to **32.7708, −100.8360** (centroid of 87 member nodes).
4. **Anacacho Wind Farm LLC** (TX, 99.8 MW, `USA0058000`) — ours:
   29.2347, −100.2092; nearest turbines **4.4 km S**, OSM
   relation/4920626 "Anacacho Wind Farm" 5.5 km SSE. GEM: 5.75 km
   from our point, 1.39 km from the OSM centroid — independent
   corroboration. Corrected to **29.1902, −100.1921** (centroid of 55
   member nodes).

GPPD CSV check (fetched from the WRI repo this session): all four
plants carry these exact wrong coordinates in GPPD v1.3.0 itself, so
EIA-860 either agreed or was unmatched — the registry chain is the
fault, not our build. Consistent with experiments.md 2026-07-04
("registries share self-reported geocodes… agreement is NOT
verification").

### Borderline

- **Cloud County Wind Farm** (KS, 201 MW): first pass found nothing
  ≤3 km, but the 25 km follow-up shows OSM's "Meridian Way Wind Farm"
  plant relation centroid **2.42 km** SE (same farm — Cloud County's
  Meridian Way, capacity matches at 201 MW) with turbines from
  3.1 km. Our point sits at the edge of a farm that spans ~10 km —
  usable, not materially wrong. Not fixed (inside tolerance); noted
  because a screenshot at the exact point could still show empty
  ground (see "third failure mode" below).

### Class (b) — absent from OSM (NOT proof we're wrong)

- **South Forks Hydro** (ID, 8 MW): no power feature ≤3 km, no name
  hit ≤150 km. Small canal-hydro country near Twin Falls; OSM simply
  hasn't mapped it. Our coordinate is UNVERIFIED, not disproven.
- **TalenEnergy Martins Creek LLC West Shore** (PA, 37 MW oil
  peaker): no plant/generator ≤3 km, but OSM's "West Shore
  Substation" is **0.09 km** from our point — the eponymous grid site
  corroborates the position; the peaker itself is unmapped. Plausibly
  correct.

### A third failure mode the fix does not remove

Even a CORRECT wind-farm point is a centroid of turbines spread over
kilometres (Orient Wind Farm, 500.8 MW, matched its own OSM relation
at 2.06 km). A user zooming to the symbol can land on empty ground
between turbine strings with perfectly current imagery. This is a
presentation-honesty gap, distinct from (a) and (c) — flagged for the
already-queued card-surfacing slice (T-CLIENT, not this worktree).

## Imagery-date honesty surface (task 4)

What the client already does (`client/src/pages/datamap.tsx`):

- On-map capture-date chip (lines 1954–2010, 7175–7177): Esri
  World_Imagery identify at the view centre on every settle;
  "imagery at centre: YYYY-MM-DD · SOURCE" when known, honest
  "capture date unknown at this zoom / unknown" fallbacks, never a
  fabricated date.
- Per-plant position provenance on the card (4921–4925): "Position
  imagery-verified." vs "Position approximate (registry-reported —
  GPPD/EIA-860)."; layer status notes "top 100 by MW imagery-verified
  · rest approximate" (4931–4932). The HIFLD tile layer mirrors this
  with per-plant VAL_METHOD (3802–3807).

Assessment: case (c) — "our point is right, the imagery is older than
the installation" — **is already covered**: a user comparing a plant
symbol against imagery can read the imagery's capture date on-screen
and the card already declares non-verified positions approximate. The
gap exposed by this audit is not the honesty surface; it is that
"approximate" understated reality for wind (10% of the wind sample
materially wrong) — plus the centroid caveat above, which no current
text states.

## Other layers (task 1 scope, honestly downscoped)

- **Nuclear facilities**: an 8-site OSM spot-check was attempted;
  Overpass throttled/timed out the regex name queries after the
  power-plant batches (public endpoint, polite-use limits — recorded
  as exactly that, no results fabricated). One internal finding
  without Overpass: **"Savannah River Plant" (Q139796054) and
  "Savannah River Site" (Q2458173) are duplicate entities at the
  identical coordinate (33.25, −81.65)** — same installation under
  historical/current names; candidate dedupe for a T-DATACORE
  session. (Wackersdorf and Sellafield THORP/Magnox pairs also share
  coordinates but are genuinely distinct co-located facilities.)
- **Military installations**: all 3,024 coordinates ARE OpenStreetMap
  — cross-checking against OSM proves nothing. Independent
  verification would need imagery sheets (site_verify.py pattern) or
  DoD installation lists; left as a filed follow-up, not silently
  skipped.
- **Strategic sites**: all 16 imagery-verified 2026-07-03 with
  crosshair sheets after that audit found 11/16 mispositioned; no
  re-check needed this session.

## Failure-rate extrapolation (stated, not oversold)

4/40 wind = 10% materially wrong (exact binomial 95% CI ≈ 3–24%).
Over 1,139 wind plants that projects ~30–270 bad records — too many
to hand-fix, few enough to machine-verify: the same Overpass
turbine-proximity check compiled into a script can classify all 1,139
in ~95 batched queries. 0/20 other-fuel wrong (CI 0–17%) — wind is
the concentrated risk, consistent with the mechanism (farm centroids
and owner-address geocodes vs. a visible single-site facility).

## Recommendation

1. **DONE this worktree (separate data commit)**: fix the 4 records
   via `datacore/powerplants/position_overrides.json` (per-record OSM
   evidence + GEM corroboration inside the file), applied to the
   shipped JSON and wired into `scripts/build_powerplants.py` (with
   an original-coordinate guard so a future upstream fix is never
   clobbered by a stale override); regression test added in
   `server/powerplants.test.ts`.
2. **Queued (own session/PR, T-DATACORE)**: batch OSM verification of
   all 1,139 wind plants (script exists in embryo — this audit's
   method); output = more overrides + a per-plant
   position-confidence field.
3. **Queued (T-CLIENT card-surfacing slice, already in the ROUND 10
   queue)**: surface "position OSM-verified / approximate /
   OSM-corrected" on plant cards, and add the wind-centroid caveat
   ("point marks the farm centroid; turbines spread over km").
4. **Filed above**: nuclear Savannah River dedupe; military-layer
   independent verification method.

## Reproducibility

Sampling/matching scripts ran in the session scratchpad (ephemeral);
the method is fully specified here: seed 20260718, batch size 12,
match predicate and radii as stated, Overpass QL pattern
`nwr["power"~"^(plant|generator)$"](around:3000,LAT,LON)` + follow-ups
`around:25000` and name-regex `around:150000`, corrections =
`rel(ID); (node(r)["power"="generator"]; way(r); node(w);); out skel;`
member-node centroid. OSM base timestamp 2026-07-18T01:28Z. Per-record
evidence (relation IDs, distances, GEM coords) is embedded in
`datacore/powerplants/position_overrides.json`.
