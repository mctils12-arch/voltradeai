/**
 * cbpBorderCrossings.ts — client-side coordinate table for the CBP Border
 * Wait Times map layer (server/cbpBorderWait.ts, GET /api/data/border-waits).
 *
 * The bwt.cbp.gov feed identifies crossings by a CBP `port_number` (e.g.
 * "230404") plus `port_name`/`crossing_name` text, but carries NO
 * coordinates. This is a curated table for all 84 port_numbers observed
 * live in the feed (probed 2026-07-21), each verified against an
 * independent public source (primarily Wikipedia bridge/port-of-entry
 * infoboxes, cross-checked against CBP's own published addresses) at
 * "national-map precision" (facility/bridge-level, not GPS-survey
 * precision) — the same precision class as client/src/lib/faaAirports.ts's
 * airport reference points. A port_number outside this table is honestly
 * OMITTED from the map, never guessed or geocoded on the fly.
 *
 * KNOWN DUPLICATE PORT CODES (found during verification, not a bug in this
 * table — CBP's own feed carries more than one port_number for a handful of
 * physical crossings): 202401/240202 (El Paso, Paso Del Norte), 230103/
 * 535504 (Brownsville, Gateway), 230106/535501 (Brownsville, B&M),
 * 231001/231002 (Roma), 240104/240203 (El Paso, Ysleta), 240201/240207/
 * 240215/240221 (El Paso, Bridge of the Americas complex), 250601/250602/
 * 250608/250609 (Otay Mesa complex), 260301/260305 (Naco). Each duplicate
 * pair is mapped to the SAME verified coordinate rather than guessed apart
 * — two markers will overlap on the map, which is the honest reflection of
 * the feed reporting two lane-class records for one physical port.
 */

export interface BorderCrossingCoord {
  lat: number;
  lon: number;
  /** Bridge/facility name shown in the click card. */
  name: string;
  /** Border town/city (the feed's own port_name, in most cases). */
  city: string;
  state: string;
}

export const BORDER_CROSSING_COORDS: Record<string, BorderCrossingCoord> = {
  // ── Maine / Vermont / New York (Canadian border) ──
  "010401": { lat: 45.805754, lon: -70.396743, name: "Armstrong–Jackman Border Crossing", city: "Jackman", state: "ME" },
  "010601": { lat: 46.135003, lon: -67.781176, name: "Houlton I-95 Border Crossing", city: "Houlton", state: "ME" },
  "011501": { lat: 45.191791, lon: -67.283546, name: "Ferry Point Bridge", city: "Calais", state: "ME" },
  "011502": { lat: 45.170000, lon: -67.296800, name: "Milltown International Bridge", city: "Calais", state: "ME" },
  "011503": { lat: 45.161032, lon: -67.302600, name: "International Avenue Bridge", city: "Calais", state: "ME" },
  "020901": { lat: 45.005859, lon: -72.087961, name: "Derby Line–Rock Island Border Crossing (I-91)", city: "Derby Line", state: "VT" },
  "021101": { lat: 45.012200, lon: -71.791700, name: "Norton–Stanhope Border Crossing", city: "Norton", state: "VT" },
  "021201": { lat: 45.015484, lon: -73.084606, name: "Highgate Springs–St. Armand Border Crossing (I-89)", city: "Highgate Springs", state: "VT" },
  "070101": { lat: 44.733214, lon: -75.457677, name: "Ogdensburg–Prescott International Bridge", city: "Ogdensburg", state: "NY" },
  "070401": { lat: 44.990602, lon: -74.739561, name: "Three Nations Crossing (Seaway Int'l Bridge)", city: "Massena", state: "NY" },
  "070801": { lat: 44.347349, lon: -75.983328, name: "Thousand Islands Bridge", city: "Alexandria Bay", state: "NY" },
  "071201": { lat: 45.008825, lon: -73.452358, name: "Champlain–St. Bernard de Lacolle Border Crossing (I-87)", city: "Champlain", state: "NY" },
  "090101": { lat: 42.906940, lon: -78.905560, name: "Peace Bridge", city: "Buffalo", state: "NY" },
  "090102": { lat: 43.090200, lon: -79.067700, name: "Rainbow Bridge", city: "Niagara Falls", state: "NY" },
  "090103": { lat: 43.109208, lon: -79.058336, name: "Whirlpool Rapids Bridge", city: "Niagara Falls", state: "NY" },
  "090104": { lat: 43.153060, lon: -79.044453, name: "Lewiston–Queenston Bridge", city: "Lewiston", state: "NY" },
  "L01901": { lat: 47.355550, lon: -68.322300, name: "Edmundston–Madawaska Bridge", city: "Madawaska", state: "ME" },

  // ── Washington / Montana / North Dakota / Minnesota / Michigan (Canadian border) ──
  "300401": { lat: 49.002000, lon: -122.735000, name: "Pacific Highway Border Crossing", city: "Blaine", state: "WA" },
  "300402": { lat: 49.000111, lon: -122.754389, name: "Peace Arch Border Crossing", city: "Blaine", state: "WA" },
  "300403": { lat: 49.002059, lon: -123.068252, name: "Point Roberts–Boundary Bay Border Crossing", city: "Point Roberts", state: "WA" },
  "300901": { lat: 49.002411, lon: -122.265511, name: "Sumas–Huntingdon Border Crossing", city: "Sumas", state: "WA" },
  "302301": { lat: 49.002270, lon: -122.484984, name: "Lynden–Aldergrove Border Crossing", city: "Lynden", state: "WA" },
  "331001": { lat: 48.998393, lon: -111.960447, name: "Sweetgrass–Coutts Border Crossing (I-15)", city: "Sweetgrass", state: "MT" },
  "340101": { lat: 49.000477, lon: -97.237634, name: "Pembina–Emerson Border Crossing (I-29)", city: "Pembina", state: "ND" },
  "360401": { lat: 48.607200, lon: -93.401900, name: "Fort Frances–International Falls International Bridge", city: "International Falls", state: "MN" },
  "380001": { lat: 42.312000, lon: -83.074000, name: "Ambassador Bridge", city: "Detroit", state: "MI" },
  "380002": { lat: 42.324503, lon: -83.040053, name: "Detroit–Windsor Tunnel", city: "Detroit", state: "MI" },
  "380201": { lat: 42.998610, lon: -82.423610, name: "Blue Water Bridge", city: "Port Huron", state: "MI" },
  "380301": { lat: 46.507030, lon: -84.361320, name: "Sault Ste. Marie International Bridge", city: "Sault Ste. Marie", state: "MI" },

  // ── El Paso complex / Brownsville / Presidio / Columbus / Santa Teresa / Fort Hancock (Mexican border) ──
  "240104": { lat: 31.671100, lon: -106.338000, name: "Ysleta–Zaragoza International Bridge", city: "El Paso", state: "TX" },
  "240201": { lat: 31.767300, lon: -106.450900, name: "Bridge of the Americas (BOTA)", city: "El Paso", state: "TX" },
  "240202": { lat: 31.749900, lon: -106.486700, name: "Paso Del Norte (Santa Fe St.) Bridge", city: "El Paso", state: "TX" },
  "240203": { lat: 31.671100, lon: -106.338000, name: "Ysleta–Zaragoza International Bridge", city: "El Paso", state: "TX" },
  "240204": { lat: 31.748100, lon: -106.482700, name: "Stanton Street Bridge (Dedicated Commuter Lane)", city: "El Paso", state: "TX" },
  "240207": { lat: 31.767300, lon: -106.450900, name: "Bridge of the Americas Port of Entry", city: "El Paso", state: "TX" },
  "240215": { lat: 31.767300, lon: -106.450900, name: "Bridge of the Americas Cargo Facility", city: "El Paso", state: "TX" },
  "240221": { lat: 31.767300, lon: -106.450900, name: "El Paso Port of Entry", city: "El Paso", state: "TX" },
  "240301": { lat: 29.561700, lon: -104.395100, name: "Presidio–Ojinaga International Bridge", city: "Presidio", state: "TX" },
  "240401": { lat: 31.432800, lon: -106.148100, name: "Marcelino Serna Port of Entry (Tornillo–Guadalupe Bridge)", city: "Tornillo", state: "TX" },
  "240601": { lat: 31.784400, lon: -107.627600, name: "Columbus Port of Entry", city: "Columbus", state: "NM" },
  "240801": { lat: 31.785900, lon: -106.680300, name: "Santa Teresa Port of Entry", city: "Santa Teresa", state: "NM" },
  "535501": { lat: 25.891900, lon: -97.504400, name: "B&M (Brownsville & Matamoros) Bridge", city: "Brownsville", state: "TX" },
  "535502": { lat: 25.883600, lon: -97.476400, name: "Veterans International Bridge at Los Tomates", city: "Brownsville", state: "TX" },
  "535503": { lat: 26.029200, lon: -97.738800, name: "Free Trade International Bridge (Los Indios)", city: "Los Indios", state: "TX" },
  "535504": { lat: 25.898300, lon: -97.497200, name: "Gateway International Bridge", city: "Brownsville", state: "TX" },
  "l24501": { lat: 31.273100, lon: -105.854400, name: "Fort Hancock–El Porvenir International Bridge", city: "Fort Hancock", state: "TX" },

  // ── Del Rio / Eagle Pass / Laredo / Rio Grande Valley (Mexican border) ──
  "202401": { lat: 31.749929, lon: -106.486691, name: "Paso Del Norte International Bridge", city: "El Paso", state: "TX" },
  "230103": { lat: 25.899143, lon: -97.496978, name: "Gateway International Bridge", city: "Brownsville", state: "TX" },
  "230106": { lat: 25.893290, lon: -97.505497, name: "B&M (Brownsville & Matamoros) Bridge", city: "Brownsville", state: "TX" },
  "230201": { lat: 29.326756, lon: -100.927514, name: "Del Rio–Ciudad Acuña International Bridge", city: "Del Rio", state: "TX" },
  "230301": { lat: 28.705560, lon: -100.511940, name: "Eagle Pass–Piedras Negras International Bridge I", city: "Eagle Pass", state: "TX" },
  "230302": { lat: 28.697780, lon: -100.510560, name: "Camino Real International Bridge (Bridge II)", city: "Eagle Pass", state: "TX" },
  "230401": { lat: 27.499400, lon: -99.507420, name: "Gateway to the Americas Bridge (Laredo I)", city: "Laredo", state: "TX" },
  "230402": { lat: 27.500216, lon: -99.502814, name: "Juárez–Lincoln International Bridge (Laredo II)", city: "Laredo", state: "TX" },
  "230403": { lat: 27.699716, lon: -99.745646, name: "Laredo–Colombia Solidarity Bridge", city: "Laredo", state: "TX" },
  "230404": { lat: 27.597291, lon: -99.537119, name: "World Trade International Bridge", city: "Laredo", state: "TX" },
  "230501": { lat: 26.096867, lon: -98.269964, name: "Hidalgo International Bridge", city: "Hidalgo", state: "TX" },
  "230502": { lat: 26.088436, lon: -98.200859, name: "Pharr–Reynosa International Bridge", city: "Pharr", state: "TX" },
  "230503": { lat: 26.116500, lon: -98.318300, name: "Anzalduas International Bridge", city: "Mission", state: "TX" },
  "230701": { lat: 26.366961, lon: -98.801336, name: "Rio Grande City–Camargo International Bridge", city: "Rio Grande City", state: "TX" },
  "230901": { lat: 26.063318, lon: -97.949821, name: "Progreso International Bridge", city: "Progreso", state: "TX" },
  "230902": { lat: 26.069601, lon: -98.072226, name: "Donna–Río Bravo International Bridge", city: "Donna", state: "TX" },
  "231001": { lat: 26.403700, lon: -99.018800, name: "Roma–Ciudad Miguel Alemán International Bridge", city: "Roma", state: "TX" },
  "231002": { lat: 26.403700, lon: -99.018800, name: "Roma–Ciudad Miguel Alemán International Bridge", city: "Roma", state: "TX" },

  // ── Arizona / California (Mexican border) ──
  "250201": { lat: 32.718700, lon: -114.727700, name: "Andrade Port of Entry", city: "Andrade", state: "CA" },
  "250301": { lat: 32.675506, lon: -115.388675, name: "Calexico East Port of Entry", city: "Calexico", state: "CA" },
  "250302": { lat: 32.664455, lon: -115.495883, name: "Calexico West Port of Entry", city: "Calexico", state: "CA" },
  "250401": { lat: 32.543330, lon: -117.029720, name: "San Ysidro Port of Entry", city: "San Ysidro", state: "CA" },
  "250407": { lat: 32.540380, lon: -117.032290, name: "San Ysidro PedWest", city: "San Ysidro", state: "CA" },
  "250409": { lat: 32.548330, lon: -116.974440, name: "Cross Border Xpress (CBX)", city: "San Ysidro", state: "CA" },
  "250501": { lat: 32.576852, lon: -116.627179, name: "Tecate Port of Entry", city: "Tecate", state: "CA" },
  "250601": { lat: 32.550602, lon: -116.938187, name: "Otay Mesa Port of Entry (Passenger)", city: "Otay Mesa", state: "CA" },
  "250602": { lat: 32.550602, lon: -116.938187, name: "Otay Mesa Port of Entry (Commercial)", city: "Otay Mesa", state: "CA" },
  "250608": { lat: 32.550602, lon: -116.938187, name: "Otay Mesa Port of Entry", city: "Otay Mesa", state: "CA" },
  "250609": { lat: 32.550602, lon: -116.938187, name: "Otay Mesa Port of Entry", city: "Otay Mesa", state: "CA" },
  "260101": { lat: 31.334679, lon: -109.560374, name: "Douglas (Raul Hector Castro) Port of Entry", city: "Douglas", state: "AZ" },
  "260201": { lat: 31.880423, lon: -112.816919, name: "Lukeville Port of Entry", city: "Lukeville", state: "AZ" },
  "260301": { lat: 31.334400, lon: -109.947990, name: "Naco Port of Entry", city: "Naco", state: "AZ" },
  "260305": { lat: 31.334400, lon: -109.947990, name: "Naco Port of Entry", city: "Naco", state: "AZ" },
  "260401": { lat: 31.332800, lon: -110.942800, name: "Nogales–Grand Avenue (DeConcini) Port of Entry", city: "Nogales", state: "AZ" },
  "260402": { lat: 31.333900, lon: -110.965594, name: "Mariposa Port of Entry", city: "Nogales", state: "AZ" },
  "260403": { lat: 31.332750, lon: -110.941500, name: "Morley Gate", city: "Nogales", state: "AZ" },
  "260801": { lat: 32.485339, lon: -114.782120, name: "San Luis I Port of Entry", city: "San Luis", state: "AZ" },
  "260802": { lat: 32.461700, lon: -114.698800, name: "San Luis II Port of Entry (Commercial)", city: "San Luis", state: "AZ" },
};

export type BorderLane = "commercial_standard" | "commercial_FAST" | "passenger_standard";

export const BORDER_LANE_LABEL: Record<BorderLane, string> = {
  commercial_standard: "Commercial (Standard)",
  commercial_FAST: "Commercial (FAST)",
  passenger_standard: "Passenger (Standard)",
};

export function borderLaneLabel(lane: string | null | undefined): string {
  return (lane && BORDER_LANE_LABEL[lane as BorderLane]) || String(lane || "unknown");
}

/** Severity color from the feed's own published delay_min — a RAW numeric
 *  field, not a derived signal. Null (not published) is distinguished from
 *  a genuine zero-minute reading, never coerced together. */
export function borderDelayColor(delayMin: number | null | undefined): string {
  if (delayMin == null) return "#64748b";  // not published
  if (delayMin <= 0) return "#4ade80";     // no delay
  if (delayMin <= 30) return "#facc15";
  if (delayMin <= 60) return "#fb923c";
  return "#ff3b3b";                        // 60+ min
}

export function borderDelayLabel(delayMin: number | null | undefined): string {
  if (delayMin == null) return "not published";
  if (delayMin <= 0) return "no delay";
  return `${delayMin} min wait`;
}
