// LUNAR SURFACE MISSIONS (human directive 2026-08-13: "i want for the moon to
// be able to see the stuff on the moon... the point to be clickable on the
// mission, even the new one from china, and have it as a layer for past moon
// missions with a label").
//
// Supersedes APOLLO_SITES (which held only the six crewed landings). ONE array
// keyed by id — two arrays on the same ids would drift (staleness rule).
//
// ── HONESTY CONTRACT ────────────────────────────────────────────────────────
//  1. Every coordinate is a PUBLISHED selenographic value, degrees, longitude
//     POSITIVE EAST. Nothing is interpolated, averaged, or inferred. Sources
//     were fetched and cross-checked; each site carries its own source_url.
//  2. coord_confidence is not decoration — it changes how the marker renders:
//       surveyed_lro      hardware located in LRO/LROC NAC imagery (metre-level)
//       published_precise published with a stated 1σ, not NAC-located
//       catalogued        mission-era tracking solution, km-scale, never found
//       estimated         impact region only (Luna 2: ±~1°, i.e. ~30 km)
//     `catalogued`/`estimated` sites draw a dashed halo and their cards say
//     "reported", never "located".
//  3. attribution_certain=false where the identification itself is provisional
//     (Luna 25: NASA says a new crater is *likely* Luna 25's — likely is not
//     proven, and the card says so).
//  4. NOT A COMPLETE CATALOGUE. This is a verified subset of lunar surface
//     arrivals, not every object that has reached the Moon. Ranger impacts,
//     Surveyor 5/6/7, Luna 23, the S-IVB impacts and others are absent because
//     they were not verified in this pass — the layer's status line says so
//     rather than implying completeness.
//  5. Traverses (drawn routes) are NOT part of this module. Where a mission's
//     route exists as citable vector data it ships separately; where it does
//     not (Apollo 15/16 — mapped by LROC but never released as data;
//     Lunokhod 1/2 — digitized by MIIGAiK but the archive host is offline and
//     unlicensed; Yutu/Yutu-2 — figures in paywalled papers), NO LINE IS
//     DRAWN. `hardware` below carries the surveyed points we do have; a line
//     between two surveyed points would be a guess.

export type LunarOutcome = "landed" | "partial" | "crashed" | "impact_intentional";
export type LunarKind = "crewed" | "robotic_lander" | "rover" | "sample_return" | "impactor";
export type LunarCoordConfidence = "surveyed_lro" | "published_precise" | "catalogued" | "estimated";

/** An additional surveyed point belonging to the same mission (a rover's final
 *  parking spot, a laser retroreflector) — real coordinates, never a route. */
export interface LunarHardware {
  label: string;
  lat: number;
  lon: number;
  uncertainty_m?: number;
  note?: string;
}

/** Crew facts, present only for crewed landings. */
export interface LunarCrew {
  surface: string[];   // the two who walked; the CMP stayed in orbit
  cmp: string;
  evas: number;
  eva_hours: number;   // documented total EVA duration
  max_eva_km: number;  // FARTHEST documented point from the LM — never a path length
  rover: boolean;
}

export interface LunarSite {
  id: string;
  mission: string;
  agency: string;
  country: string;
  /** selenographic latitude, degrees */
  lat: number;
  /** selenographic longitude, degrees POSITIVE EAST (−180..180] */
  lon: number;
  /** ISO instant of touchdown/impact. Some are date-only — see date_is_day_only. */
  date_utc: string;
  /** true when the exact time is unpublished and must NOT be rendered as one */
  date_is_day_only?: boolean;
  outcome: LunarOutcome;
  kind: LunarKind;
  rover_name?: string;
  /** Earth-facing hemisphere. Invariant: near_side === (Math.abs(lon) < 90). */
  near_side: boolean;
  coord_confidence: LunarCoordConfidence;
  uncertainty_m?: number;
  uncertainty_deg?: number;
  /** false ⇒ the identification is provisional, not the coordinate precision */
  attribution_certain?: boolean;
  region: string;
  source_url: string;
  second_source_url?: string;
  note: string;
  crew?: LunarCrew;
  hardware?: LunarHardware[];
}

const LROC_WAGNER = "https://lroc.im-ldi.com/images/938";

export const LUNAR_SITES: LunarSite[] = [
  // ── Apollo — crewed. Coordinates are the LRO-surveyed descent-stage
  //    positions (Wagner et al. 2017). The six ids are FROZEN STRINGS: they
  //    are the join key into lroc.ts APOLLO_NAC_SITES (the ~0.5 m/px NAC
  //    strips) and into the site card. Never renumber them.
  {
    id: "apollo11", mission: "Apollo 11", agency: "NASA", country: "USA",
    lat: 0.67416, lon: 23.47314, date_utc: "1969-07-20T20:17:40Z",
    outcome: "landed", kind: "crewed", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 0.3,
    region: "Mare Tranquillitatis (Tranquility Base)", source_url: LROC_WAGNER,
    note: "First crewed landing. The single EVA stayed within ~60 m of the LM Eagle.",
    crew: { surface: ["Neil Armstrong", "Buzz Aldrin"], cmp: "Michael Collins", evas: 1, eva_hours: 2.5, max_eva_km: 0.06, rover: false },
  },
  {
    id: "apollo12", mission: "Apollo 12", agency: "NASA", country: "USA",
    lat: -3.0128, lon: -23.4219, date_utc: "1969-11-19T06:54:35Z",
    outcome: "landed", kind: "crewed", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 2.2,
    region: "Oceanus Procellarum", source_url: LROC_WAGNER,
    note: "Precision landing ~155 m from Surveyor 3 — the crew walked to the robotic probe and returned parts of it to Earth.",
    crew: { surface: ["Pete Conrad", "Alan Bean"], cmp: "Richard Gordon", evas: 2, eva_hours: 7.75, max_eva_km: 0.41, rover: false },
  },
  {
    id: "apollo14", mission: "Apollo 14", agency: "NASA", country: "USA",
    lat: -3.64589, lon: -17.47194, date_utc: "1971-02-05T09:18:11Z",
    outcome: "landed", kind: "crewed", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 0.4,
    region: "Fra Mauro highlands", source_url: LROC_WAGNER,
    note: "Cone Crater trek with the pulled MET tool cart; the crew turned back ~30 m below the crater rim.",
    crew: { surface: ["Alan Shepard", "Edgar Mitchell"], cmp: "Stuart Roosa", evas: 2, eva_hours: 9.35, max_eva_km: 1.45, rover: false },
  },
  {
    id: "apollo15", mission: "Apollo 15", agency: "NASA", country: "USA",
    lat: 26.13239, lon: 3.6333, date_utc: "1971-07-30T22:16:29Z",
    outcome: "landed", kind: "crewed", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 1.0,
    region: "Hadley–Apennine (Hadley Rille)", source_url: LROC_WAGNER,
    note: "First Lunar Roving Vehicle. LROC has mapped this traverse but has not released it as vector data, so no route line is drawn — only surveyed points.",
    crew: { surface: ["David Scott", "James Irwin"], cmp: "Alfred Worden", evas: 3, eva_hours: 18.55, max_eva_km: 5.0, rover: true },
    hardware: [{ label: "LRV final parking", lat: 26.13174, lon: 3.63803, uncertainty_m: 0.5 }],
  },
  {
    id: "apollo16", mission: "Apollo 16", agency: "NASA", country: "USA",
    lat: -8.9734, lon: 15.5011, date_utc: "1972-04-21T02:23:35Z",
    outcome: "landed", kind: "crewed", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 3.0,
    region: "Descartes highlands", source_url: LROC_WAGNER,
    note: "Only highlands landing. As with Apollo 15 the traverse is mapped but unreleased as data — surveyed points only, no drawn route.",
    crew: { surface: ["John Young", "Charles Duke"], cmp: "Ken Mattingly", evas: 3, eva_hours: 20.23, max_eva_km: 4.5, rover: true },
    hardware: [{ label: "LRV final parking", lat: -8.9729, lon: 15.5037, uncertainty_m: 3.7 }],
  },
  {
    id: "apollo17", mission: "Apollo 17", agency: "NASA", country: "USA",
    lat: 20.1911, lon: 30.7723, date_utc: "1972-12-11T19:54:57Z",
    outcome: "landed", kind: "crewed", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 3.5,
    region: "Taurus–Littrow valley", source_url: LROC_WAGNER,
    note: "Last crewed landing; the only geologist on the Moon. Longest traverses — the rover reached 7.6 km from the LM.",
    crew: { surface: ["Eugene Cernan", "Harrison Schmitt"], cmp: "Ronald Evans", evas: 3, eva_hours: 22.06, max_eva_km: 7.6, rover: true },
  },

  // ── Soviet programme ───────────────────────────────────────────────────────
  {
    id: "luna2", mission: "Luna 2", agency: "OKB-1", country: "USSR",
    lat: 29.1, lon: 0.0, date_utc: "1959-09-13T21:02:24Z",
    outcome: "impact_intentional", kind: "impactor", near_side: true,
    coord_confidence: "estimated", uncertainty_deg: 1,
    region: "Palus Putredinis, near Autolycus/Archimedes",
    source_url: "https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1959-014A",
    note: "First human-made object to reach another world. The impact crater has NEVER been located — this is the reported impact region (±~1°, roughly 30 km), not a found site.",
  },
  {
    id: "luna9", mission: "Luna 9", agency: "Lavochkin", country: "USSR",
    lat: 7.08, lon: -64.37, date_utc: "1966-02-03T18:45:30Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "catalogued",
    region: "Oceanus Procellarum, west of Reiner and Marius",
    source_url: "https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1966-006A",
    note: "First soft landing on another world. The lander has not been confirmed in NAC imagery; two recent candidate identifications disagree with each other by kilometres, so the catalogued mission-era position is what we show.",
  },
  {
    id: "luna13", mission: "Luna 13", agency: "Lavochkin", country: "USSR",
    lat: 18.87, lon: -62.05, date_utc: "1966-12-24T18:01:00Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "catalogued",
    region: "Oceanus Procellarum, between Krafft and Seleucus",
    source_url: "https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1966-116A",
    note: "Third soft landing; measured soil density with a penetrometer. Not included in the 2016 LROC coordinate release — position is tracking-derived.",
  },
  {
    id: "luna16", mission: "Luna 16", agency: "Lavochkin", country: "USSR",
    lat: -0.5137, lon: 56.3638, date_utc: "1970-09-20T05:18:00Z",
    outcome: "landed", kind: "sample_return", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 3.7,
    region: "Mare Fecunditatis", source_url: LROC_WAGNER,
    note: "First robotic sample return — 101 g brought back by an uncrewed craft.",
  },
  {
    id: "luna17", mission: "Luna 17", agency: "Lavochkin", country: "USSR",
    lat: 38.23764, lon: -35.00163, date_utc: "1970-11-17T03:46:50Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 0.5,
    region: "Mare Imbrium", source_url: LROC_WAGNER,
    note: "Delivered Lunokhod 1, the first rover on another world.",
  },
  {
    id: "lunokhod1", mission: "Lunokhod 1", agency: "Lavochkin", country: "USSR",
    lat: 38.315, lon: -35.0081, date_utc: "1970-11-17T03:46:50Z",
    outcome: "landed", kind: "rover", rover_name: "Lunokhod 1", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 3.8,
    region: "Mare Imbrium", source_url: LROC_WAGNER,
    note: "Drove 10.54 km over 322 days. This marker is its FINAL PARKING SPOT (~2.9 km north of its lander), not the landing point. Its route was digitized from NAC imagery by MIIGAiK, but that archive is offline and unlicensed — so no route line is drawn.",
    hardware: [{ label: "Laser retroreflector", lat: 38.3151705, lon: -35.0079815, note: "position from laser ranging — uncertainty effectively zero" }],
  },
  {
    id: "luna20", mission: "Luna 20", agency: "Lavochkin", country: "USSR",
    lat: 3.7863, lon: 56.6242, date_utc: "1972-02-21T19:19:00Z",
    outcome: "landed", kind: "sample_return", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 4.3,
    region: "Apollonius highlands", source_url: LROC_WAGNER,
    note: "Returned 55 g of highland material.",
  },
  {
    id: "luna21", mission: "Luna 21", agency: "Lavochkin", country: "USSR",
    lat: 25.9994, lon: 30.4076, date_utc: "1973-01-15T22:35:00Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 13.4,
    region: "Le Monnier crater, Mare Serenitatis", source_url: LROC_WAGNER,
    note: "Delivered Lunokhod 2. Landing time is cited as 22:35 or 23:35 UTC in different sources.",
  },
  {
    id: "lunokhod2", mission: "Lunokhod 2", agency: "Lavochkin", country: "USSR",
    lat: 25.8323, lon: 30.9222, date_utc: "1973-01-15T22:35:00Z",
    outcome: "landed", kind: "rover", rover_name: "Lunokhod 2", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 7.5,
    region: "Le Monnier crater, Mare Serenitatis", source_url: LROC_WAGNER,
    note: "The longest distance driven on another world until Opportunity — re-measured from NAC imagery at 39.16 km, more than the 37 km claimed in 1973. This marker is its FINAL PARKING SPOT (~19 km southeast of its lander). Route data is not obtainable or licensed, so no route line is drawn.",
    hardware: [{ label: "Laser retroreflector", lat: 25.8323185, lon: 30.9221468, note: "position from laser ranging — uncertainty effectively zero" }],
  },
  {
    id: "luna24", mission: "Luna 24", agency: "Lavochkin", country: "USSR",
    lat: 12.7142, lon: 62.2129, date_utc: "1976-08-18T06:36:00Z",
    outcome: "landed", kind: "sample_return", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 3.7,
    region: "Mare Crisium", source_url: LROC_WAGNER,
    note: "Final Soviet lunar mission and the last soft landing by anyone until 2013 — returned 170.1 g.",
  },

  // ── Surveyor (US robotic precursors) ───────────────────────────────────────
  {
    id: "surveyor1", mission: "Surveyor 1", agency: "NASA", country: "USA",
    lat: -2.4745, lon: -43.3398, date_utc: "1966-06-02T06:17:36Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 5.6,
    region: "Oceanus Procellarum", source_url: LROC_WAGNER,
    note: "First US soft landing, four months after Luna 9.",
  },
  {
    id: "surveyor3", mission: "Surveyor 3", agency: "NASA", country: "USA",
    lat: -3.0162, lon: -23.418, date_utc: "1967-04-20T00:04:53Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 2.6,
    region: "Oceanus Procellarum", source_url: LROC_WAGNER,
    note: "Bounced twice before settling (the time given is final touchdown). Apollo 12 landed ~155 m away and brought pieces of it home.",
  },

  // ── China ─────────────────────────────────────────────────────────────────
  {
    id: "change3", mission: "Chang'e 3", agency: "CNSA", country: "China",
    lat: 44.1214, lon: -19.5116, date_utc: "2013-12-14T13:11:00Z",
    outcome: "landed", kind: "rover", rover_name: "Yutu", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Mare Imbrium, near Sinus Iridum",
    source_url: "https://lroc.im-ldi.com/images/637",
    note: "First soft landing since 1976. This is the LANDER's surveyed position; Yutu's final parked position is not precisely published, and its route (mapped in the literature) is not available as licensed data — so no rover line is drawn.",
  },
  {
    id: "change4", mission: "Chang'e 4", agency: "CNSA", country: "China",
    lat: -45.457, lon: 177.589, date_utc: "2019-01-03T02:26:00Z",
    outcome: "landed", kind: "rover", rover_name: "Yutu-2", near_side: false,
    coord_confidence: "surveyed_lro", uncertainty_m: 20,
    region: "Von Kármán crater, South Pole–Aitken basin (FAR SIDE)",
    source_url: "https://lroc.im-ldi.com/images/1087",
    note: "First soft landing on the lunar FAR SIDE — only visible when the far side faces you. Yutu-2 is the longest-lived lunar rover; its 132 mapped waypoints exist only in paywalled figures, so no route line is drawn.",
  },
  {
    id: "change5", mission: "Chang'e 5", agency: "CNSA", country: "China",
    lat: 43.0576, lon: -51.9161, date_utc: "2020-12-01T15:11:00Z",
    outcome: "landed", kind: "sample_return", near_side: true,
    coord_confidence: "surveyed_lro", uncertainty_m: 20,
    region: "Oceanus Procellarum, near Mons Rümker",
    source_url: "https://lroc.im-ldi.com/images/1172",
    note: "Returned 1,731 g — the first lunar samples brought to Earth since 1976.",
  },
  {
    id: "change6", mission: "Chang'e 6", agency: "CNSA", country: "China",
    lat: -41.6385, lon: -153.9852, date_utc: "2024-06-01T22:23:00Z",
    outcome: "landed", kind: "sample_return", near_side: false,
    coord_confidence: "surveyed_lro", uncertainty_m: 30,
    region: "Apollo basin, South Pole–Aitken basin (FAR SIDE)",
    source_url: "https://lroc.im-ldi.com/images/1374",
    note: "The first sample return from the lunar FAR SIDE — 1,935.3 g. Only visible when the far side faces you.",
  },

  // ── India / Russia / Japan ────────────────────────────────────────────────
  {
    id: "chandrayaan3", mission: "Chandrayaan-3 (Vikram)", agency: "ISRO", country: "India",
    lat: -69.3741, lon: 32.32, date_utc: "2023-08-23T12:33:00Z",
    outcome: "landed", kind: "rover", rover_name: "Pragyan", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "High southern latitudes — Statio Shiv Shakti",
    source_url: "https://lroc.im-ldi.com/images/1314",
    note: "First landing in the high-southern-latitude region and India's first soft landing. Pragyan drove ~100 m; the route is not published as data.",
  },
  {
    id: "luna25", mission: "Luna 25", agency: "Roscosmos", country: "Russia",
    lat: -57.865, lon: 61.36, date_utc: "2023-08-19T11:58:00Z",
    outcome: "crashed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro", attribution_certain: false,
    region: "Pontécoulant G crater region",
    source_url: "https://www.nasa.gov/missions/lro/nasas-lro-observes-crater-likely-from-luna-25-impact/",
    note: "Russia's first lunar attempt since 1976, lost on descent ~400 km from target. LRO found a new ~10 m crater that NASA says is LIKELY from this impact — the identification is probable, not proven.",
  },
  {
    id: "slim", mission: "SLIM", agency: "JAXA", country: "Japan",
    lat: -13.316, lon: 25.251, date_utc: "2024-01-19T15:20:00Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Shioli crater, near Mare Nectaris",
    source_url: "https://lroc.im-ldi.com/images/1358",
    note: "\"Moon Sniper\" — landed ~55 m from its target, the most precise lunar landing yet, but came to rest nose-down.",
  },
  {
    id: "hakutor1", mission: "Hakuto-R M1", agency: "ispace", country: "Japan",
    lat: 47.581, lon: 44.094, date_utc: "2023-04-25T16:40:00Z",
    outcome: "crashed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Atlas crater, Mare Frigoris",
    source_url: "https://www.nasa.gov/",
    note: "Ran out of propellant during descent after a misread altitude. Carried the UAE's Rashid rover and JAXA's SORA-Q — both lost. LRO imaged the debris field.",
  },
  {
    id: "hakutor2", mission: "Hakuto-R M2 (Resilience)", agency: "ispace", country: "Japan",
    lat: 60.4445, lon: -4.588, date_utc: "2025-06-05T19:24:00Z",
    outcome: "crashed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Mare Frigoris",
    source_url: "https://lroc.im-ldi.com/images/1456",
    note: "Lost during descent when its laser rangefinder failed to slow the craft in time.",
  },
  {
    id: "beresheet", mission: "Beresheet", agency: "SpaceIL / IAI", country: "Israel",
    lat: 32.5956, lon: 19.3496, date_utc: "2019-04-11T19:23:00Z",
    outcome: "crashed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Mare Serenitatis",
    source_url: "https://www.nasa.gov/",
    note: "The first privately funded lunar landing attempt; an engine shutdown during braking ended it. LRO imaged the impact smudge.",
  },

  // ── US commercial (CLPS) ──────────────────────────────────────────────────
  {
    id: "im1", mission: "IM-1 (Odysseus)", agency: "Intuitive Machines / NASA CLPS", country: "USA",
    lat: -80.13, lon: 1.44, date_utc: "2024-02-22T23:24:00Z",
    outcome: "partial", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Malapert A crater, south polar region",
    source_url: "https://www.nasa.gov/",
    note: "First US soft landing since Apollo 17 and the first by a private company — it touched down intact but caught a footpad and tipped over.",
  },
  {
    id: "im2", mission: "IM-2 (Athena)", agency: "Intuitive Machines / NASA CLPS", country: "USA",
    lat: -84.7906, lon: 29.1957, date_utc: "2025-03-06T17:30:00Z",
    outcome: "partial", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Mons Mouton, south polar region",
    source_url: "https://lroc.im-ldi.com/images/1408",
    note: "Reached the surface but came to rest on its side inside a ~20 m crater, ending the mission early.",
  },
  {
    id: "blueghost1", mission: "Blue Ghost Mission 1", agency: "Firefly Aerospace / NASA CLPS", country: "USA",
    lat: 18.562, lon: 61.81, date_utc: "2025-03-02T08:34:00Z",
    outcome: "landed", kind: "robotic_lander", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Mare Crisium",
    source_url: "https://lroc.im-ldi.com/images/1400",
    note: "The first fully successful commercial soft landing — upright, stable, and operated a full lunar day.",
  },

  // ── Deliberate impacts (science end-of-mission) ───────────────────────────
  {
    id: "lcross", mission: "LCROSS (Centaur impact)", agency: "NASA", country: "USA",
    lat: -84.6796, lon: -48.7093, date_utc: "2009-10-09T11:31:19Z",
    outcome: "impact_intentional", kind: "impactor", near_side: true,
    coord_confidence: "published_precise", uncertainty_m: 115,
    region: "Cabeus crater floor (permanently shadowed)",
    source_url: "https://arxiv.org/abs/1103.1687",
    note: "Deliberately crashed to loft a debris plume — the observation that confirmed water ice in a polar cold trap. It sits on a PERMANENTLY SHADOWED crater floor: sunlight never reaches it, so with realistic lighting on it stays dark even when facing you.",
  },
  {
    id: "grail_a", mission: "GRAIL A (Ebb)", agency: "NASA", country: "USA",
    lat: 75.609, lon: -26.593, date_utc: "2012-12-17T22:28:00Z",
    outcome: "impact_intentional", kind: "impactor", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Unnamed massif near the north pole — Sally K. Ride Impact Site",
    source_url: "https://lroc.im-ldi.com/posts/596",
    note: "End-of-mission impact of the first GRAIL gravity-mapping twin.",
  },
  {
    id: "grail_b", mission: "GRAIL B (Flow)", agency: "NASA", country: "USA",
    lat: 75.651, lon: -26.832, date_utc: "2012-12-17T22:29:00Z",
    outcome: "impact_intentional", kind: "impactor", near_side: true,
    coord_confidence: "surveyed_lro",
    region: "Unnamed massif near the north pole — Sally K. Ride Impact Site",
    source_url: "https://lroc.im-ldi.com/posts/596",
    note: "Impacted seconds after its twin; the exact separation in time is approximate.",
  },
  {
    id: "ladee", mission: "LADEE", agency: "NASA", country: "USA",
    lat: 11.8494, lon: -93.2493, date_utc: "2014-04-18", date_is_day_only: true,
    outcome: "impact_intentional", kind: "impactor", near_side: false,
    coord_confidence: "surveyed_lro",
    region: "Eastern rim of Sundman V crater (FAR SIDE)",
    source_url: "https://lroc.im-ldi.com/images/822",
    note: "Dust/atmosphere orbiter, deorbited into the FAR SIDE. The impact time was never published (only that it fell between 04:30 and 05:22 UTC) — so we show the date alone rather than invent a timestamp.",
  },
];

/** Back-compat: the crewed Apollo landings, for callers that only want those. */
export const APOLLO_SITE_IDS = ["apollo11", "apollo12", "apollo14", "apollo15", "apollo16", "apollo17"] as const;

export function getLunarSite(id: string): LunarSite | undefined {
  return LUNAR_SITES.find((s) => s.id === id);
}

/** Near/far-side split, COMPUTED from the array — the previous hand-written
 *  note ("all 6 sites are on the near side… none are on the far side") became
 *  false the moment Chang'e 4/6 and LADEE landed in the data. A computed
 *  string cannot go stale that way. */
export const LUNAR_FAR_SIDE_SITES = LUNAR_SITES.filter((s) => !s.near_side);

export const LUNAR_SIDE_NOTE =
  `${LUNAR_SITES.length - LUNAR_FAR_SIDE_SITES.length} of ${LUNAR_SITES.length} sites are on the near side (Earth-facing); ` +
  `${LUNAR_FAR_SIDE_SITES.map((s) => s.mission).join(", ")} ` +
  `${LUNAR_FAR_SIDE_SITES.length === 1 ? "is" : "are"} on the FAR side and only appear when the far side faces the camera — ` +
  "use the fly-to chips to reach them";

/** The completeness disclaimer. This layer is a verified subset, and saying so
 *  is the difference between honest and misleading. */
export const LUNAR_COVERAGE_NOTE =
  `${LUNAR_SITES.length} verified surface sites (1959–2025) — a checked subset, NOT a complete catalogue of ` +
  "everything that has reached the Moon (Ranger impacts, Surveyor 5–7, Luna 23 and others are not included). " +
  "Every coordinate is a published value with its source on the card.";

/** Per-mission imagery honesty. Only the six Apollo ids have a NAC strip
 *  (lroc.ts APOLLO_NAC_SITES) — nothing else may imply ~0.5 m/px detail. */
export const APOLLO_IMAGERY_NOTE =
  "Our streamed mosaic is LROC WAC (~76–100 m/px) — Apollo hardware is sub-pixel here. " +
  "The descent stages, experiment packages, rover tracks and foot trails ARE resolved in " +
  "LROC NAC frames (~0.5 m/px, public domain), which stream in at this site when you keep zooming.";

export const NON_APOLLO_IMAGERY_NOTE =
  "Our streamed mosaic is LROC WAC (~76–100 m/px). Spacecraft hardware is far below one pixel at this " +
  "resolution — the marker shows the published coordinate, not something visible in our imagery.";

/** LROC featured-site search for a mission (citable provenance). */
export function lrocFeaturedUrl(id: string): string {
  const site = getLunarSite(id);
  return `https://www.lroc.asu.edu/search?q=${encodeURIComponent(site?.mission ?? id)}`;
}
