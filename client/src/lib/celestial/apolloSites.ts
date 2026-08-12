// APOLLO LANDING SITES (human directive 2026-08-12: "have the moon landing
// have points like we have on the earth with symbols that when you click
// them it gives info on the site and zooms in on it").
//
// HONESTY CONTRACT:
//  - Coordinates are the published NASA/LPI landing-site positions
//    (selenographic degrees, positive east) — documented survey values,
//    not estimates.
//  - EVA facts (crew, dates, durations, distances) are published mission
//    records. max_eva_km is the documented FARTHEST POINT from the LM —
//    the range ring drawn from it is labeled approximate extent, NEVER the
//    walked/driven path. Real traverse polylines ship only when ingested
//    from a citable digitized source (filed follow-up); the actual tracks
//    are visible in LROC NAC imagery (~0.5 m/px), which our streamed WAC
//    mosaic (~76-100 m/px) cannot resolve — every card says so.

export interface ApolloSite {
  id: string;
  mission: string;
  /** selenographic latitude/longitude, degrees, +E (NASA/LPI published) */
  lat: number;
  lon: number;
  landed_utc: string;      // touchdown date (UTC)
  crew_surface: string[];  // the two who walked; CMP stayed in orbit
  cmp: string;
  evas: number;
  eva_hours: number;       // total EVA duration, hours (documented)
  max_eva_km: number;      // farthest documented distance from the LM
  rover: boolean;
  region: string;
  note: string;
}

export const APOLLO_SITES: ApolloSite[] = [
  {
    id: "apollo11", mission: "Apollo 11",
    lat: 0.67408, lon: 23.47297,
    landed_utc: "1969-07-20", crew_surface: ["Neil Armstrong", "Buzz Aldrin"], cmp: "Michael Collins",
    evas: 1, eva_hours: 2.5, max_eva_km: 0.06, rover: false,
    region: "Mare Tranquillitatis (Tranquility Base)",
    note: "First crewed landing. The single EVA stayed within ~60 m of the Lunar Module Eagle.",
  },
  {
    id: "apollo12", mission: "Apollo 12",
    lat: -3.01239, lon: -23.42157,
    landed_utc: "1969-11-19", crew_surface: ["Pete Conrad", "Alan Bean"], cmp: "Richard Gordon",
    evas: 2, eva_hours: 7.75, max_eva_km: 0.41, rover: false,
    region: "Oceanus Procellarum",
    note: "Precision landing ~180 m from Surveyor 3 — the crew walked to the robotic probe and returned parts of it to Earth.",
  },
  {
    id: "apollo14", mission: "Apollo 14",
    lat: -3.6453, lon: -17.47136,
    landed_utc: "1971-02-05", crew_surface: ["Alan Shepard", "Edgar Mitchell"], cmp: "Stuart Roosa",
    evas: 2, eva_hours: 9.35, max_eva_km: 1.45, rover: false,
    region: "Fra Mauro highlands",
    note: "Cone Crater trek with the pulled MET tool cart; the crew turned back ~30 m below the crater rim.",
  },
  {
    id: "apollo15", mission: "Apollo 15",
    lat: 26.13222, lon: 3.63386,
    landed_utc: "1971-07-30", crew_surface: ["David Scott", "James Irwin"], cmp: "Alfred Worden",
    evas: 3, eva_hours: 18.55, max_eva_km: 5.0, rover: true,
    region: "Hadley–Apennine (Hadley Rille)",
    note: "First Lunar Roving Vehicle. Traverses along Hadley Rille and the Apennine front; the rover tracks remain visible in NAC imagery.",
  },
  {
    id: "apollo16", mission: "Apollo 16",
    lat: -8.97301, lon: 15.50019,
    landed_utc: "1972-04-21", crew_surface: ["John Young", "Charles Duke"], cmp: "Ken Mattingly",
    evas: 3, eva_hours: 20.23, max_eva_km: 4.5, rover: true,
    region: "Descartes highlands",
    note: "Only highlands landing. Rover traverses to Stone Mountain and North Ray Crater.",
  },
  {
    id: "apollo17", mission: "Apollo 17",
    lat: 20.1908, lon: 30.77168,
    landed_utc: "1972-12-11", crew_surface: ["Eugene Cernan", "Harrison Schmitt"], cmp: "Ronald Evans",
    evas: 3, eva_hours: 22.06, max_eva_km: 7.6, rover: true,
    region: "Taurus–Littrow valley",
    note: "Last crewed landing; the only geologist on the Moon. Longest traverses — the rover reached 7.6 km from the LM.",
  },
];

/** LROC featured-site page per mission (citable provenance for "the real
 *  tracks are visible" — NAC frames showing hardware, experiment packages,
 *  rover tracks and foot trails). */
export function lrocFeaturedUrl(id: string): string {
  return `https://www.lroc.asu.edu/search?q=${encodeURIComponent(id.replace("apollo", "apollo "))}`;
}

/** The honesty line every Apollo card carries about OUR imagery. */
export const APOLLO_IMAGERY_NOTE =
  "Our streamed mosaic is LROC WAC (~76–100 m/px) — Apollo hardware is sub-pixel here. " +
  "The descent stages, experiment packages, rover tracks and foot trails ARE resolved in " +
  "LROC NAC frames (~0.5 m/px, public domain) — NAC site tiles are a filed follow-up.";
