# Wind-Fleet Position Audit (full 1,139) — 2026-07-18

[RESEARCH] · T-DATACORE (data verification; no client code touched).
Execution of recommendation 2 of `research/position_audit_2026-07-18.md`
(the 60-plant audit that found 4/40 wind coordinates materially wrong):
batch-verify EVERY wind plant in
`datacore/powerplants/us_power_plants.json` against OpenStreetMap.
Same classes, same 3 km match predicate, same evidence standard, same
override mechanism (`datacore/powerplants/position_overrides.json`).

## Verdict in one paragraph

All 1,139 wind rows were classified against a bulk OSM snapshot
(1,441 wind plants, ~82k wind generators, fetched in 9 regional
queries). **1,105 (97.0%) confirmed** within 3 km (median error
0.14 km). **23 suspects** (>3 km from any wind feature with a
name-matched OSM plant elsewhere): 20 cleared the override evidence
bar and are fixed (errors 7.0–53.2 km; 15 of 20 additionally
corroborated by GEM), 3 are honestly unresolved. **7 absent from
OSM** — 5 of these are GEM-corroborated at our exact coordinate
(retired/small farms OSM never mapped; position fine), 2 carry real
position doubt but no named OSM match, so no override (the bar
requires a named plant, not just a turbine cluster). Full-fleet
material-error rate lands at **24/1,139 = 2.1%** (incl. the 4 prior
fixes) — below the prior audit's 3–24% CI extrapolation, which was
biased upward by its two forced picks (the human's Waverly case and a
shared-coordinate smell). The dominant failure mode is confirmed as
office/POB geocodes and county-centroid-style displacement in the
GPPD registry chain; wind remains the concentrated risk.

## Method

- Universe: all 1,139 `fuel=wind` rows (row = [name, mw, fuel, owner,
  lat, lon, verified]). Every row mapped 1:1 to its GPPD id by
  (name[:60], capacity) against the GPPD v1.3.0 CSV (fetched this
  session from the WRI repo) — 1,139/1,139 matched, 0 ambiguous.
- OSM ground truth fetched in BULK (no per-plant queries): 9 regional
  bboxes — 6 CONUS longitude strips (24.5–49.5°N; splits at −114,
  −104, −97, −90, −80, east edge −66) + AK (56.5–65, −154..−144) +
  HI + PR. Each box = ONE Overpass query pulling (a) all
  `power=plant` + `plant:source=wind` (nwr, `out center` with tags —
  names needed for suspect matching) and (b) all wind generators
  (`generator:source=wind` or `generator:method=wind_turbine`;
  `out skel center`, ids+coords only).
- All matching local (numpy haversine): per row, distance to nearest
  wind generator and nearest wind-plant center.
- Classes (same as the prior audit): **confirmed** ≤3 km ·
  **suspect** >3 km with a name-matched OSM wind plant ≤200 km ·
  **absent** >3 km with no plausible named match (NOT proof of
  error) · **already-overridden** (the 4 prior fixes).
- Name matching: token-normalized (generic tokens stripped, roman
  numerals normalized) with full containment of the shorter core name
  required AND phase-number agreement (so "Los Vientos III" can never
  match phase IV; "Rattlesnake" (Goldwind) never matches "Rattlesnake
  Den" (Invenergy) 184 km away — distance excludes it).
- Override bar (per directive): a named OSM plant match is REQUIRED —
  nearest-turbine-cluster alone is ambiguous between adjacent farms.
  Additionally enforced: a double-placement guard (no other fleet row
  within ~3–5 km of the proposed target, so two of our rows can never
  be moved onto one farm) and capacity/operator consistency review
  per record.
- Corrections: `to` = centroid of the OSM plant relation's member
  geometry (`out geom`, 6–152 pts per farm), fetched for all 20
  targets in ONE follow-up query — same correction method as the
  prior audit.
- GEM corroboration: `datacore/gem/power_units.json.gz` (2,212 US
  wind units), name-matched per suspect; distances to both `from` and
  `to` recorded in the override evidence.
- Consistency check against the prior 60-plant audit: every wind
  plant both audits covered agrees (the prior audit's 4 bad records
  are the 4 already-overridden; its borderline Cloud County is
  confirmed here at 2.42 km via the Meridian Way plant relation; the
  8 apparent differences were all rows in the two east boxes fetched
  after the dry-run comparison — final results agree there too).

## Query log (politeness accounting — public Overpass, custom UA with contact)

| Box | Result | Bytes | Elements | Time | Attempts |
|---|---|---|---|---|---|
| conus_w1 (−125..−114) | ok | 1,578,128 | 10,239 | 211.8 s | 1 |
| conus_w2 (−114..−104) | ok | 1,080,860 | 6,255 | 142.0 s | 3 (see events) |
| conus_c1 (−104..−97) | ok | 6,186,970 | 36,673 | 122.3 s | 2 (one 504) |
| conus_c2 (−97..−90) | ok | 2,303,333 | 13,445 | 197.6 s | 1 |
| conus_e1 (−90..−80) | ok | 1,754,805 | 10,386 | 105.7 s | 7 (see events) |
| conus_e2 (−80..−66) | ok | 1,030,982 | 6,101 | 65.6 s | 3 |
| alaska | ok | 8,008 | 43 | 6.3 s | 1 |
| hawaii | ok | 25,693 | 145 | 6.7 s | 1 |
| puertorico | ok | 10,729 | 64 | 10.2 s | 2 (one 504) |
| member geometry (20 rels) | ok | (not captured) | 20 | ~30 s | 3 (two 504) |

Totals: ~14.0 MB successful payload, 10 successful queries, ~14
failed/aborted attempts. Throttle/latency events, stated honestly:
(1) one conus_w2 request was ABORTED CLIENT-SIDE (orchestration
switch from a background to a foreground fetch loop) — the orphaned
query kept an IP slot busy and the immediate retry got a
"server too busy" dispatcher error; waited 90 s before the successful
retry. (2) conus_e1/e2 initially timed out repeatedly (504s + "query
timed out after 251/401 s") — root cause was query shape, not
politeness: `node["power"="generator"][...]` drives Overpass off the
huge all-generators index, and the eastern US is dense with
rooftop-solar generator nodes. Rewriting the generator clauses
rare-tag-first (`node["generator:source"="wind"]`, dropping the
`power=generator` guard) cut e1 from >400 s timeout to 105.7 s.
Recorded so the next bulk-OSM session starts with the fast form.
(3) Backoffs of 30–90 s taken between retries throughout; two
requests (one failed e1 attempt, the member-geometry query) went to
the Kumi public mirror to spread load. OSM base timestamps span
2026-07-18T02:16:15Z–03:16:10Z (per box).

## Class counts (n = 1,139)

| Class | Count | Notes |
|---|---|---|
| confirmed (≤3 km wind-tagged feature) | **1,105** | median 0.14 km, p90 0.80 km, max 2.84 km |
| suspect (>3 km, named OSM plant elsewhere) | **23** | 20 overridden (below) · 3 unresolved |
| absent from OSM (NOT proof of error) | **7** | 5 GEM-corroborated at our exact point · 2 with real doubt |
| already-overridden (prior audit) | **4** | Waverly KS · Waverly Community IA · Scurry County TX · Anacacho TX |

Material-error rate: 24/1,139 = **2.1%** (4 prior + 20 new). The
prior audit's 10% sample rate (CI 3–24%) overstated the fleet rate —
its sample deliberately included the human-reported case and a
shared-coordinate smell (2/4 of its hits), which a random sample
would not have contained at that rate.

## Suspect table (23) — evidence per record

Columns: plant · GPPD id · MW · our coord · nearest wind generator
(km) · named OSM plant match · GEM name match (dist from our coord) ·
disposition. Full machine-readable evidence (incl. member-centroid
point counts and GEM coords) is in each `position_overrides.json`
record.

| Plant | GPPD id | MW | Our coord | Turb. km | OSM plant match | GEM | Disposition |
|---|---|---|---|---|---|---|---|
| Ranchero Wind Farm LLC | USA0062259 | 300.0 | 30.5949, −101.453 | 26.48 | relation/14148600 'Ranchero Wind Farm' @ 53.16 km | 300.0 MW (operating) @ 51.87 km | OVERRIDE → 31.0154, −101.7187 (53.2 km) |
| El Cabo Wind | USA0058098 | 298.0 | 34.6511, −105.4617 | 8.48 | relation/13683714 'El Cabo Wind Farm' @ 19.33 km | 298.0 MW (operating) @ 25.32 km | OVERRIDE → 34.7212, −105.6418 (18.2 km) |
| Thunder Ranch Wind Project | USA0061269 | 297.8 | 36.522, −97.145 | 4.4 | relation/14130731 'Thunder Ranch Wind Project' @ 14.74 km | 297.8 MW (operating) @ 15.12 km | OVERRIDE → 36.5724, −97.3063 (15.5 km) |
| Palo Duro Wind | USA0059475 | 249.9 | 36.2439, −101.0014 | 12.12 | relation/6861479 'Palo Duro Wind Energy Center' @ 19.26 km | 282.3 MW (operating) @ 23.66 km | OVERRIDE → 36.4214, −101.0295 (19.9 km) |
| Mariah del Norte | USA0059005 | 230.4 | 34.6675, −102.5778 | 4.24 | relation/7309636 'Mariah del Norte Wind' @ 13.12 km | 230.4 MW Del Norte (operating) @ 9.68 km | OVERRIDE → 34.714, −102.7074 (12.9 km) |
| Crowned Ridge Wind Energy Center | USA0060503 | 200.1 | 45.1549, −96.8368 | 7.31 | relation/12114430 'Crowned Ridge Wind Energy Center' @ 13.61 km | 200.0 MW (operating) @ 15.89 km | OVERRIDE → 45.0515, −96.9314 (13.7 km); capacity pins phase I (phase-II rel is 200.6 MW, separate) |
| Los Vientos Windpower III | USA0059320 | 200.0 | 26.3886, −98.8061 | 12.14 | relation/6706827 'Los Vientos Wind III' @ 23.02 km | — | OVERRIDE → 26.4824, −98.5906 (23.9 km) |
| Los Vientos Windpower IV | USA0059321 | 200.0 | 26.3806, −98.8183 | 13.38 | relation/6706825 'Los Vientos Wind IV' @ 23.74 km | — | OVERRIDE → 26.5697, −98.6868 (24.8 km) |
| Post Oak Wind LLC | USA0056483 | 200.0 | 32.5144, −99.6564 | 8.43 | relation/7495747 'Lone Star Post Oak Wind' @ 14.07 km | — | OVERRIDE → 32.5328, −99.5123 (13.7 km); Post Oak = Lone Star phase II (EDP), owner field is the sibling-phase LLC |
| FPL Energy Vansycle LLC (WA) | USA0055560 | 176.9 | 46.06, −118.917 | 3.59 | relation/14124864 'Vansycle Wind Farm' @ 25.59 km | 25.0 MW (operating) @ 23.49 km | UNRESOLVED — both OSM Vansycle plants are already occupied by our separate 25 / 98.9 MW Vansycle rows (capacities disagree with 176.9); this row is the Stateline WA portion with no distinct OSM plant |
| Rattlesnake Power LLC | USA0060743 | 160.0 | 31.3603, −99.5514 | 10.25 | relation/8760035 'Rattlesnake Wind' @ 12.87 km | 160.0 MW (operating) @ 13.08 km | OVERRIDE → 31.2497, −99.5344 (12.4 km) |
| Briscoe Wind Farm | USA0059734 | 150.0 | 34.4323, −101.2372 | 5.09 | relation/7309565 'Briscoe Wind Farm' @ 9.11 km | 150.0 MW (operating) @ 9.08 km | OVERRIDE → 34.3766, −101.3105 (9.1 km) |
| Tule Wind LLC | USA0057913 | 143.0 | 32.6639, −116.2897 | 5.48 | relation/12626257 'Tule Wind Energy Project' @ 11.21 km | 143.0 MW (operating) @ 8.44 km | OVERRIDE → 32.7586, −116.294 (10.5 km) |
| Gunsight Mountain Wind Energy LLC | USA0056776 | 120.0 | 32.2403, −101.4736 | 3.49 | relation/14148608 'Gunsight Mountain Wind Energy' @ 26.88 km | 120.0 MW (operating) @ 27.11 km | OVERRIDE → 32.4864, −101.4484 (27.5 km); registry point sits by the Big Spring facility (different farm) |
| Hancock County Wind Energy Center | USA0056010 | 98.0 | 43.0528, −93.63 | 3.15 | relation/1184839 (same name) @ 6.28 km | 98.0 MW (operating) @ 12.61 km | BORDERLINE, not fixed — turbine at 3.15 km, plant centroid 6.3 km, GEM 12.6 km the OTHER side; point plausibly at farm edge (Cloud County precedent) |
| Charles City Wind Farm | USA0056677 | 80.0 | 43.0003, −92.62 | 4.96 | relation/1198235 'Charles City Wind Farm' @ 9.24 km | 80.0 MW (operating) @ 8.67 km | OVERRIDE → 43.0317, −92.7264 (9.3 km) |
| Marshall Wind Farm | USA0059084 | 73.8 | 39.7011, −96.3608 | 4.61 | relation/14159682 'Marshall Wind Farm' @ 16.78 km | 73.8 MW (operating) @ 17.82 km | OVERRIDE → 39.8502, −96.3715 (16.6 km); nearby turbines belong to adjacent Irish Creek |
| Broken Bow Wind II LLC | USA0058981 | 73.1 | 41.3239, −99.3231 | 16.64 | relation/12284835 'Broken Bow Wind Farm Phase II' @ 28.67 km | 75.0 MW (operating) @ 28.92 km | OVERRIDE → 41.521, −99.5592 (29.5 km); phase-matched, our phase-I row separately confirmed |
| Saratoga Wind Farm | USA0061070 | 66.0 | 43.4489, −92.2807 | 5.61 | relation/13680945 'Saratoga Wind Farm' @ 9.01 km | 66.0 MW (operating) @ 7.57 km | OVERRIDE → 43.3779, −92.3373 (9.1 km) |
| Passadumkeag Windpark LLC | USA0059222 | 42.9 | 45.07, −68.21 | 11.46 | relation/10064219 (our exact name + operator) @ 13.09 km | 42.9 MW (operating) @ 12.25 km | OVERRIDE → 45.132, −68.351 (13.0 km) |
| McFadden Ridge | USA0057039 | 35.2 | 41.7244, −105.9906 | 3.74 | relation/19438307 'High Plains and McFadden Ridge Wind Farms' @ 6.65 km | 35.0 MW (operating) @ 8.22 km | UNRESOLVED — OSM merges McFadden into a combined plant (157.3 MW = our High Plains 122.1 + McFadden 35.2) whose centroid is 0.9 km from our confirmed High Plains row; moving McFadden there would double-place; no separate OSM plant exists |
| Windom Wind Project | USA0056544 | 15.6 | 43.9578, −95.1306 | 4.91 | relation/11987839 'Windom Wind Project' @ 6.59 km | — | OVERRIDE → 43.9608, −95.0437 (7.0 km); compact 15 MW farm, point cannot be inside it |
| Galactic Wind | USA0062161 | 9.9 | 43.1016, −89.3319 | 19.51 | relation/14160864 (our exact name + operator) @ 20.23 km | 9.9 MW (operating) @ 19.7 km | OVERRIDE → 43.173, −89.5607 (20.2 km) |

## Absent from OSM (7) — explicitly NOT proof our coordinate is wrong

| Plant | MW | Our coord | Turb. km | GEM name match |
|---|---|---|---|---|
| Sherbino I Wind Farm | 150.0 | 30.8073, −102.3556 | 8.63 | 150.0 MW (retired) @ 1.95 km — position corroborated |
| Tuolumne Wind Project | 136.3 | 45.8797, −120.8072 | 14.2 | no GEM name match — REAL DOUBT, unresolvable this method |
| South Dakota Wind Energy Cente[r] | 40.5 | 44.5492, −99.5 | 8.0 | 40.5 MW (operating) @ 20.58 km — GEM DISAGREES with our point; OSM silent; no override without a named OSM plant |
| Difwind Farms Ltd VI | 27.1 | 35.0506, −118.1714 | 3.79 | 15.0 MW (retired) @ 0.04 km — corroborated |
| GE - Tehachapi | 5.3 | 35.1553, −118.3694 | 5.23 | retired units @ 0.0 km — corroborated |
| Suzlon Project VIII LLC | 4.2 | 35.75, −102.28 | 26.18 | 2.1 MW (retired) @ 0.0 km — corroborated |
| Pembina Land Port of Entry Wind Turbine | 1.0 | 48.9969, −97.2416 | 14.28 | 1.0 MW (retired) @ 0.0 km — corroborated |

Caveat on the 5 "corroborated" rows: GEM and GPPD partly share
geocode lineage, so exact agreement is weaker than independent
verification (the Hardeeville lesson) — but 4 of the 5 are
retired/tiny farms where OSM absence is expected, and none shows any
contrary evidence.

## Honest limits

- OSM completeness varies by region and vintage; ABSENT ≠ WRONG. Old
  retired Tehachapi-era farms and single-turbine sites are
  systematically under-mapped.
- The 3 km threshold counts a plant-relation centroid OR any turbine;
  a plant at the edge of a 10 km farm can sit 2–3 km from its nearest
  mapped turbine and still be sound (Cloud County, Hancock County).
  Conversely 4 confirmations at 2–3 km could conceivably be
  adjacent-farm turbines — the confirmed class is "consistent with
  OSM", not imagery-verified.
- The corrected coordinates are FARM CENTROIDS (6–152 member points);
  turbines spread over km — the centroid caveat now surfaced on plant
  cards (v1.0.391) applies to corrected rows too.
- e1/e2 were fetched with the rare-tag-driven query form (no
  `power=generator` guard); this could admit a handful of non-power
  features tagged `generator:source=wind` — harmless for
  nearest-distance classification.
- Turbine points exactly on box boundaries can be double-counted
  (boxes share edges); irrelevant for nearest-distance.
- Name matching is conservative (full containment + phase agreement);
  a farm renamed beyond token overlap in OSM would land in "absent"
  rather than "suspect" — some of the 7 absents could be such cases.
- The Overpass member-geometry response size was processed in-stream
  and its byte count not captured; all other query sizes are exact.
- Thunder Ranch's override `from` is its EIA-860 coordinate (EIA had
  already moved it 8.3 km off GPPD's raw value — both wrong). The
  from-guard therefore only fires on a GPPD+EIA rebuild, which is the
  documented production re-run; a GPPD-only sandbox rebuild applies
  23/24 overrides, exactly as designed (guard verified both ways this
  session).

## Integration (separate data commit)

The 20 records were appended to
`datacore/powerplants/position_overrides.json` (same schema, same
from-guard; per-record OSM relation ids, distances, member-centroid
point counts, GEM corroboration). The shipped
`us_power_plants.json` had the 20 rows' coordinates applied
DIRECTLY (verified: exactly 20 rows changed, count/verified_count
untouched) rather than by full rebuild — the GPPD CSV is fetchable,
but a faithful rebuild also needs the EIA-860 xlsx whose exact
vintage the original build did not record; rebuilding with a
different EIA vintage would churn thousands of unrelated
coordinates. Mechanism verified instead by sandbox rebuild (GPPD-only:
"23 position-overridden", Thunder Ranch correctly guard-skipped).
`server/powerplants.test.ts` pins extended: >=24 override records,
uniqueness of override-name matches, and canary coordinates for the
biggest fix (Ranchero, 53.2 km) and an exact-name eastern fix
(Galactic Wind). Suite 5/5.

## Follow-ups filed (not re-filing items already shipped in v1.0.391)

- Tuolumne Wind Project (136.3 MW) and South Dakota Wind Energy
  Center (40.5 MW): position doubt with no OSM plant to correct
  from — candidates for the imagery-crosshair method
  (`scripts/site_verify.py` pattern) in a future session.
- FPL Energy Vansycle (WA) / Stateline complex: our 4 Vansycle-family
  rows vs OSM's 3 plants need entity-level reconciliation, not a
  coordinate fix.
- The reusable batch method (bulk regional Overpass + local matching)
  is fully specified here; the fast query form for generator scans is
  recorded in the query log — compile into `scripts/` if a third
  audit wave (other fuels) is commissioned.
