/**
 * faaAirports.ts — client-side coordinate table for the FAA National
 * Airspace Status map layer (server/faaStatus.ts, GET /api/data/airport-status).
 *
 * The FAA feed identifies airports by their 3-letter FAA/IATA location
 * identifier (e.g. "DCA", "JFK", "MEM") but carries NO coordinates —
 * ground stops/GDPs/delays/closures are reported by code only. This is a
 * CURATED subset of major US commercial + cargo-hub airports (the
 * population that realistically appears in NAS status events — Part 121
 * carriers and the handful of GA fields, like Aspen/ASE, that make the
 * feed during mountain weather). Coordinates are airport reference points
 * (public/published, national-map precision — NOT runway/approach
 * precision). An event whose ARPT code isn't in this table is honestly
 * OMITTED from the map, never guessed or geocoded on the fly.
 */

export interface AirportCoord {
  lat: number;
  lon: number;
  name: string;
  city: string;
  state: string;
}

export const AIRPORT_COORDS: Record<string, AirportCoord> = {
  // ── OEP-45 core + major hubs ──
  ATL: { lat: 33.6407, lon: -84.4277, name: "Hartsfield-Jackson Atlanta Intl", city: "Atlanta", state: "GA" },
  ORD: { lat: 41.9742, lon: -87.9073, name: "Chicago O'Hare Intl", city: "Chicago", state: "IL" },
  MDW: { lat: 41.7868, lon: -87.7522, name: "Chicago Midway Intl", city: "Chicago", state: "IL" },
  DFW: { lat: 32.8998, lon: -97.0403, name: "Dallas/Fort Worth Intl", city: "Dallas", state: "TX" },
  DAL: { lat: 32.8471, lon: -96.8518, name: "Dallas Love Field", city: "Dallas", state: "TX" },
  DEN: { lat: 39.8561, lon: -104.6737, name: "Denver Intl", city: "Denver", state: "CO" },
  LAX: { lat: 33.9416, lon: -118.4085, name: "Los Angeles Intl", city: "Los Angeles", state: "CA" },
  JFK: { lat: 40.6413, lon: -73.7781, name: "John F. Kennedy Intl", city: "New York", state: "NY" },
  LGA: { lat: 40.7769, lon: -73.8740, name: "LaGuardia", city: "New York", state: "NY" },
  EWR: { lat: 40.6895, lon: -74.1745, name: "Newark Liberty Intl", city: "Newark", state: "NJ" },
  SFO: { lat: 37.6213, lon: -122.3790, name: "San Francisco Intl", city: "San Francisco", state: "CA" },
  OAK: { lat: 37.7213, lon: -122.2207, name: "Oakland Intl", city: "Oakland", state: "CA" },
  SJC: { lat: 37.3639, lon: -121.9289, name: "San Jose Intl (Mineta)", city: "San Jose", state: "CA" },
  CLT: { lat: 35.2140, lon: -80.9431, name: "Charlotte Douglas Intl", city: "Charlotte", state: "NC" },
  LAS: { lat: 36.0840, lon: -115.1537, name: "Harry Reid Intl", city: "Las Vegas", state: "NV" },
  PHX: { lat: 33.4352, lon: -112.0101, name: "Phoenix Sky Harbor Intl", city: "Phoenix", state: "AZ" },
  IAH: { lat: 29.9902, lon: -95.3368, name: "George Bush Intercontinental", city: "Houston", state: "TX" },
  HOU: { lat: 29.6454, lon: -95.2789, name: "William P. Hobby", city: "Houston", state: "TX" },
  MIA: { lat: 25.7959, lon: -80.2870, name: "Miami Intl", city: "Miami", state: "FL" },
  FLL: { lat: 26.0726, lon: -80.1527, name: "Fort Lauderdale-Hollywood Intl", city: "Fort Lauderdale", state: "FL" },
  PBI: { lat: 26.6832, lon: -80.0956, name: "Palm Beach Intl", city: "West Palm Beach", state: "FL" },
  BOS: { lat: 42.3656, lon: -71.0096, name: "Logan Intl", city: "Boston", state: "MA" },
  MSP: { lat: 44.8848, lon: -93.2223, name: "Minneapolis-Saint Paul Intl", city: "Minneapolis", state: "MN" },
  DTW: { lat: 42.2124, lon: -83.3534, name: "Detroit Metro Wayne County", city: "Detroit", state: "MI" },
  PHL: { lat: 39.8744, lon: -75.2424, name: "Philadelphia Intl", city: "Philadelphia", state: "PA" },
  SEA: { lat: 47.4502, lon: -122.3088, name: "Seattle-Tacoma Intl", city: "Seattle", state: "WA" },
  MCO: { lat: 28.4312, lon: -81.3081, name: "Orlando Intl", city: "Orlando", state: "FL" },
  BWI: { lat: 39.1774, lon: -76.6684, name: "Baltimore/Washington Intl", city: "Baltimore", state: "MD" },
  DCA: { lat: 38.8512, lon: -77.0402, name: "Ronald Reagan Washington National", city: "Washington", state: "DC" },
  IAD: { lat: 38.9531, lon: -77.4565, name: "Washington Dulles Intl", city: "Washington", state: "DC" },
  SLC: { lat: 40.7899, lon: -111.9791, name: "Salt Lake City Intl", city: "Salt Lake City", state: "UT" },
  PVU: { lat: 40.2192, lon: -111.7234, name: "Provo Municipal", city: "Provo", state: "UT" },
  SAN: { lat: 32.7338, lon: -117.1933, name: "San Diego Intl", city: "San Diego", state: "CA" },
  TPA: { lat: 27.9755, lon: -82.5332, name: "Tampa Intl", city: "Tampa", state: "FL" },
  PDX: { lat: 45.5898, lon: -122.5951, name: "Portland Intl", city: "Portland", state: "OR" },
  STL: { lat: 38.7487, lon: -90.3700, name: "St. Louis Lambert Intl", city: "St. Louis", state: "MO" },
  HNL: { lat: 21.3245, lon: -157.9251, name: "Daniel K. Inouye Intl", city: "Honolulu", state: "HI" },
  AUS: { lat: 30.1975, lon: -97.6664, name: "Austin-Bergstrom Intl", city: "Austin", state: "TX" },
  MCI: { lat: 39.2976, lon: -94.7139, name: "Kansas City Intl", city: "Kansas City", state: "MO" },
  RDU: { lat: 35.8801, lon: -78.7880, name: "Raleigh-Durham Intl", city: "Raleigh", state: "NC" },
  CLE: { lat: 41.4117, lon: -81.8498, name: "Cleveland Hopkins Intl", city: "Cleveland", state: "OH" },
  MSY: { lat: 29.9934, lon: -90.2580, name: "Louis Armstrong New Orleans Intl", city: "New Orleans", state: "LA" },
  SMF: { lat: 38.6954, lon: -121.5908, name: "Sacramento Intl", city: "Sacramento", state: "CA" },
  SNA: { lat: 33.6757, lon: -117.8682, name: "John Wayne / Orange County", city: "Santa Ana", state: "CA" },
  ABQ: { lat: 35.0402, lon: -106.6091, name: "Albuquerque Intl Sunport", city: "Albuquerque", state: "NM" },
  BNA: { lat: 36.1263, lon: -86.6774, name: "Nashville Intl", city: "Nashville", state: "TN" },
  PIT: { lat: 40.4915, lon: -80.2329, name: "Pittsburgh Intl", city: "Pittsburgh", state: "PA" },
  IND: { lat: 39.7173, lon: -86.2944, name: "Indianapolis Intl", city: "Indianapolis", state: "IN" },
  JAX: { lat: 30.4941, lon: -81.6879, name: "Jacksonville Intl", city: "Jacksonville", state: "FL" },
  CMH: { lat: 39.9980, lon: -82.8919, name: "John Glenn Columbus Intl", city: "Columbus", state: "OH" },
  MKE: { lat: 42.9472, lon: -87.8966, name: "Milwaukee Mitchell Intl", city: "Milwaukee", state: "WI" },
  OMA: { lat: 41.3032, lon: -95.8941, name: "Eppley Airfield", city: "Omaha", state: "NE" },
  OKC: { lat: 35.3931, lon: -97.6007, name: "Will Rogers World", city: "Oklahoma City", state: "OK" },
  TUL: { lat: 36.1984, lon: -95.8881, name: "Tulsa Intl", city: "Tulsa", state: "OK" },
  RSW: { lat: 26.5362, lon: -81.7552, name: "Southwest Florida Intl", city: "Fort Myers", state: "FL" },
  JAN: { lat: 32.3112, lon: -90.0759, name: "Jackson-Medgar Wiley Evers Intl", city: "Jackson", state: "MS" },
  BHM: { lat: 33.5629, lon: -86.7535, name: "Birmingham-Shuttlesworth Intl", city: "Birmingham", state: "AL" },
  MEM: { lat: 35.0424, lon: -89.9767, name: "Memphis Intl", city: "Memphis", state: "TN" },
  SDF: { lat: 38.1744, lon: -85.7360, name: "Louisville Muhammad Ali Intl", city: "Louisville", state: "KY" },
  CVG: { lat: 39.0489, lon: -84.6678, name: "Cincinnati/N. Kentucky Intl", city: "Hebron", state: "KY" },
  ANC: { lat: 61.1743, lon: -149.9963, name: "Ted Stevens Anchorage Intl", city: "Anchorage", state: "AK" },
  FAI: { lat: 64.8151, lon: -147.8563, name: "Fairbanks Intl", city: "Fairbanks", state: "AK" },
  RNO: { lat: 39.4991, lon: -119.7681, name: "Reno-Tahoe Intl", city: "Reno", state: "NV" },
  ONT: { lat: 34.0560, lon: -117.6012, name: "Ontario Intl", city: "Ontario", state: "CA" },
  BUR: { lat: 34.2007, lon: -118.3585, name: "Hollywood Burbank", city: "Burbank", state: "CA" },
  LGB: { lat: 33.8177, lon: -118.1516, name: "Long Beach Airport", city: "Long Beach", state: "CA" },
  PSP: { lat: 33.8297, lon: -116.5067, name: "Palm Springs Intl", city: "Palm Springs", state: "CA" },
  ELP: { lat: 31.8072, lon: -106.3781, name: "El Paso Intl", city: "El Paso", state: "TX" },
  SAT: { lat: 29.5312, lon: -98.4711, name: "San Antonio Intl", city: "San Antonio", state: "TX" },
  MAF: { lat: 31.9425, lon: -102.2019, name: "Midland Intl Air & Space", city: "Midland", state: "TX" },
  LBB: { lat: 33.6636, lon: -101.8228, name: "Lubbock Preston Smith Intl", city: "Lubbock", state: "TX" },
  ICT: { lat: 37.6499, lon: -97.4331, name: "Wichita Dwight D. Eisenhower National", city: "Wichita", state: "KS" },
  DSM: { lat: 41.5340, lon: -93.6631, name: "Des Moines Intl", city: "Des Moines", state: "IA" },
  BIL: { lat: 45.8077, lon: -108.5429, name: "Billings Logan Intl", city: "Billings", state: "MT" },
  BOI: { lat: 43.5644, lon: -116.2228, name: "Boise Air Terminal", city: "Boise", state: "ID" },
  GEG: { lat: 47.6199, lon: -117.5338, name: "Spokane Intl", city: "Spokane", state: "WA" },
  PWM: { lat: 43.6462, lon: -70.3093, name: "Portland Intl Jetport", city: "Portland", state: "ME" },
  BDL: { lat: 41.9389, lon: -72.6832, name: "Bradley Intl", city: "Hartford", state: "CT" },
  PVD: { lat: 41.7240, lon: -71.4283, name: "T.F. Green Intl", city: "Providence", state: "RI" },
  MHT: { lat: 42.9326, lon: -71.4357, name: "Manchester-Boston Regional", city: "Manchester", state: "NH" },
  BTV: { lat: 44.4719, lon: -73.1533, name: "Burlington Intl", city: "Burlington", state: "VT" },
  ALB: { lat: 42.7483, lon: -73.8017, name: "Albany Intl", city: "Albany", state: "NY" },
  ROC: { lat: 43.1189, lon: -77.6724, name: "Greater Rochester Intl", city: "Rochester", state: "NY" },
  BUF: { lat: 42.9405, lon: -78.7322, name: "Buffalo Niagara Intl", city: "Buffalo", state: "NY" },
  SYR: { lat: 43.1112, lon: -76.1063, name: "Syracuse Hancock Intl", city: "Syracuse", state: "NY" },
  ISP: { lat: 40.7952, lon: -73.1002, name: "Long Island MacArthur", city: "Islip", state: "NY" },
  TEB: { lat: 40.8501, lon: -74.0608, name: "Teterboro", city: "Teterboro", state: "NJ" },
  HPN: { lat: 41.0670, lon: -73.7076, name: "Westchester County", city: "White Plains", state: "NY" },
  RIC: { lat: 37.5052, lon: -77.3197, name: "Richmond Intl", city: "Richmond", state: "VA" },
  ORF: { lat: 36.8946, lon: -76.2012, name: "Norfolk Intl", city: "Norfolk", state: "VA" },
  CHS: { lat: 32.8986, lon: -80.0405, name: "Charleston Intl", city: "Charleston", state: "SC" },
  CAE: { lat: 33.9388, lon: -81.1195, name: "Columbia Metropolitan", city: "Columbia", state: "SC" },
  GSP: { lat: 34.8957, lon: -82.2189, name: "Greenville-Spartanburg Intl", city: "Greer", state: "SC" },
  SAV: { lat: 32.1276, lon: -81.2021, name: "Savannah/Hilton Head Intl", city: "Savannah", state: "GA" },
  TLH: { lat: 30.3965, lon: -84.3503, name: "Tallahassee Intl", city: "Tallahassee", state: "FL" },
  PNS: { lat: 30.4734, lon: -87.1866, name: "Pensacola Intl", city: "Pensacola", state: "FL" },
  DAB: { lat: 29.1799, lon: -81.0581, name: "Daytona Beach Intl", city: "Daytona Beach", state: "FL" },
  MYR: { lat: 33.6797, lon: -78.9283, name: "Myrtle Beach Intl", city: "Myrtle Beach", state: "SC" },
  ILM: { lat: 34.2706, lon: -77.9026, name: "Wilmington Intl", city: "Wilmington", state: "NC" },
  GSO: { lat: 36.0978, lon: -79.9373, name: "Piedmont Triad Intl", city: "Greensboro", state: "NC" },
  AVL: { lat: 35.4362, lon: -82.5418, name: "Asheville Regional", city: "Asheville", state: "NC" },
  TYS: { lat: 35.8110, lon: -83.9940, name: "McGhee Tyson", city: "Knoxville", state: "TN" },
  CHA: { lat: 35.0353, lon: -85.2038, name: "Chattanooga Metropolitan", city: "Chattanooga", state: "TN" },
  LIT: { lat: 34.7294, lon: -92.2243, name: "Bill and Hillary Clinton National", city: "Little Rock", state: "AR" },
  SHV: { lat: 32.4466, lon: -93.8256, name: "Shreveport Regional", city: "Shreveport", state: "LA" },
  BTR: { lat: 30.5332, lon: -91.1496, name: "Baton Rouge Metropolitan", city: "Baton Rouge", state: "LA" },
  MOB: { lat: 30.6912, lon: -88.2428, name: "Mobile Regional", city: "Mobile", state: "AL" },
  HSV: { lat: 34.6372, lon: -86.7751, name: "Huntsville Intl", city: "Huntsville", state: "AL" },
  MGM: { lat: 32.3006, lon: -86.3939, name: "Montgomery Regional", city: "Montgomery", state: "AL" },
  GRR: { lat: 42.8808, lon: -85.5228, name: "Gerald R. Ford Intl", city: "Grand Rapids", state: "MI" },
  FNT: { lat: 42.9654, lon: -83.7436, name: "Bishop Intl", city: "Flint", state: "MI" },
  LAN: { lat: 42.7787, lon: -84.5874, name: "Capital Region Intl", city: "Lansing", state: "MI" },
  TOL: { lat: 41.5868, lon: -83.8078, name: "Toledo Express", city: "Toledo", state: "OH" },
  DAY: { lat: 39.9024, lon: -84.2194, name: "Dayton Intl", city: "Dayton", state: "OH" },
  AZO: { lat: 42.2350, lon: -85.5522, name: "Kalamazoo/Battle Creek Intl", city: "Kalamazoo", state: "MI" },
  MSN: { lat: 43.1399, lon: -89.3375, name: "Dane County Regional", city: "Madison", state: "WI" },
  GRB: { lat: 44.4851, lon: -88.1296, name: "Green Bay-Austin Straubel Intl", city: "Green Bay", state: "WI" },
  CID: { lat: 41.8847, lon: -91.7108, name: "Eastern Iowa", city: "Cedar Rapids", state: "IA" },
  FSD: { lat: 43.5820, lon: -96.7419, name: "Sioux Falls Regional (Foss Field)", city: "Sioux Falls", state: "SD" },
  RAP: { lat: 44.0453, lon: -103.0574, name: "Rapid City Regional", city: "Rapid City", state: "SD" },
  FAR: { lat: 46.9207, lon: -96.8158, name: "Hector Intl", city: "Fargo", state: "ND" },
  BIS: { lat: 46.7727, lon: -100.7467, name: "Bismarck Municipal", city: "Bismarck", state: "ND" },
  GFK: { lat: 47.9493, lon: -97.1761, name: "Grand Forks Intl", city: "Grand Forks", state: "ND" },
  FWA: { lat: 40.9785, lon: -85.1951, name: "Fort Wayne Intl", city: "Fort Wayne", state: "IN" },
  SBN: { lat: 41.7087, lon: -86.3173, name: "South Bend Intl", city: "South Bend", state: "IN" },
  EVV: { lat: 38.0370, lon: -87.5324, name: "Evansville Regional", city: "Evansville", state: "IN" },
  PIA: { lat: 40.6642, lon: -89.6933, name: "Peoria Intl", city: "Peoria", state: "IL" },
  SPI: { lat: 39.8441, lon: -89.6779, name: "Abraham Lincoln Capital", city: "Springfield", state: "IL" },
  MLI: { lat: 41.4485, lon: -90.5075, name: "Quad City Intl", city: "Moline", state: "IL" },
  RFD: { lat: 42.1954, lon: -89.0972, name: "Chicago Rockford Intl", city: "Rockford", state: "IL" },
  SGF: { lat: 37.2457, lon: -93.3886, name: "Springfield-Branson National", city: "Springfield", state: "MO" },
  COU: { lat: 38.8181, lon: -92.2196, name: "Columbia Regional", city: "Columbia", state: "MO" },
  DUL: { lat: 46.8420, lon: -92.1936, name: "Duluth Intl", city: "Duluth", state: "MN" },
  RST: { lat: 43.9083, lon: -92.4999, name: "Rochester Intl", city: "Rochester", state: "MN" },
  CWA: { lat: 44.7776, lon: -89.6668, name: "Central Wisconsin", city: "Mosinee", state: "WI" },
  ATW: { lat: 44.2581, lon: -88.5190, name: "Appleton Intl", city: "Appleton", state: "WI" },
  // ── Mountain / West ──
  COS: { lat: 38.8058, lon: -104.7008, name: "Colorado Springs", city: "Colorado Springs", state: "CO" },
  ASE: { lat: 39.2232, lon: -106.8688, name: "Aspen-Pitkin County", city: "Aspen", state: "CO" },
  EGE: { lat: 39.6426, lon: -106.9178, name: "Eagle County Regional", city: "Vail/Eagle", state: "CO" },
  HDN: { lat: 40.4812, lon: -107.2178, name: "Yampa Valley", city: "Steamboat Springs", state: "CO" },
  GJT: { lat: 39.1224, lon: -108.5267, name: "Grand Junction Regional", city: "Grand Junction", state: "CO" },
  JAC: { lat: 43.6073, lon: -110.7377, name: "Jackson Hole", city: "Jackson", state: "WY" },
  SUN: { lat: 43.5044, lon: -114.2964, name: "Friedman Memorial (Sun Valley)", city: "Hailey", state: "ID" },
  MSO: { lat: 46.9163, lon: -114.0906, name: "Missoula Montana", city: "Missoula", state: "MT" },
  BZN: { lat: 45.7776, lon: -111.1530, name: "Bozeman Yellowstone Intl", city: "Bozeman", state: "MT" },
  GTF: { lat: 47.4820, lon: -111.3706, name: "Great Falls Intl", city: "Great Falls", state: "MT" },
  FCA: { lat: 48.3105, lon: -114.2560, name: "Glacier Park Intl", city: "Kalispell", state: "MT" },
  IDA: { lat: 43.5146, lon: -112.0708, name: "Idaho Falls Regional", city: "Idaho Falls", state: "ID" },
  TWF: { lat: 42.4818, lon: -114.4877, name: "Magic Valley Regional", city: "Twin Falls", state: "ID" },
  SGU: { lat: 37.0364, lon: -113.5103, name: "St. George Regional", city: "St. George", state: "UT" },
  FLG: { lat: 35.1385, lon: -111.6710, name: "Flagstaff Pulliam", city: "Flagstaff", state: "AZ" },
  TUS: { lat: 32.1161, lon: -110.9410, name: "Tucson Intl", city: "Tucson", state: "AZ" },
  YUM: { lat: 32.6566, lon: -114.6060, name: "Yuma Intl", city: "Yuma", state: "AZ" },
  // ── Pacific ──
  OGG: { lat: 20.8986, lon: -156.4306, name: "Kahului", city: "Kahului", state: "HI" },
  KOA: { lat: 19.7388, lon: -156.0456, name: "Ellison Onizuka Kona Intl", city: "Kailua-Kona", state: "HI" },
  LIH: { lat: 21.9760, lon: -159.3390, name: "Lihue", city: "Lihue", state: "HI" },
  ITO: { lat: 19.7214, lon: -155.0485, name: "Hilo Intl", city: "Hilo", state: "HI" },
  GUM: { lat: 13.4834, lon: 144.7960, name: "Antonio B. Won Pat Intl", city: "Hagatna", state: "GU" },
  // ── PNW / Alaska ──
  PSC: { lat: 46.2647, lon: -119.1190, name: "Tri-Cities", city: "Pasco", state: "WA" },
  YKM: { lat: 46.5682, lon: -120.5442, name: "Yakima Air Terminal", city: "Yakima", state: "WA" },
  EUG: { lat: 44.1246, lon: -123.2119, name: "Eugene Airport (Mahlon Sweet)", city: "Eugene", state: "OR" },
  MFR: { lat: 42.3742, lon: -122.8735, name: "Rogue Valley Intl", city: "Medford", state: "OR" },
  JNU: { lat: 58.3549, lon: -134.5763, name: "Juneau Intl", city: "Juneau", state: "AK" },
  KTN: { lat: 55.3556, lon: -131.7136, name: "Ketchikan Intl", city: "Ketchikan", state: "AK" },
  BET: { lat: 60.7798, lon: -161.8380, name: "Bethel", city: "Bethel", state: "AK" },
  OME: { lat: 64.5122, lon: -165.4453, name: "Nome", city: "Nome", state: "AK" },
  BRW: { lat: 71.2854, lon: -156.7659, name: "Wiley Post-Will Rogers Memorial", city: "Utqiagvik", state: "AK" },
  // ── Texas secondary + more cargo/GA ──
  HRL: { lat: 26.2285, lon: -97.6544, name: "Valley Intl", city: "Harlingen", state: "TX" },
  BRO: { lat: 25.9068, lon: -97.4259, name: "Brownsville/South Padre Island", city: "Brownsville", state: "TX" },
  CRP: { lat: 27.7704, lon: -97.5012, name: "Corpus Christi Intl", city: "Corpus Christi", state: "TX" },
  AMA: { lat: 35.2194, lon: -101.7059, name: "Rick Husband Amarillo Intl", city: "Amarillo", state: "TX" },
  ABI: { lat: 32.4113, lon: -99.6819, name: "Abilene Regional", city: "Abilene", state: "TX" },
  CLL: { lat: 30.5886, lon: -96.3638, name: "Easterwood", city: "College Station", state: "TX" },
  TYR: { lat: 32.3541, lon: -95.4024, name: "Tyler Pounds Regional", city: "Tyler", state: "TX" },
  LRD: { lat: 27.5438, lon: -99.4616, name: "Laredo Intl", city: "Laredo", state: "TX" },
  // ── Extra business-jet / GA fields prone to weather delays ──
  VNY: { lat: 34.2098, lon: -118.4899, name: "Van Nuys", city: "Van Nuys", state: "CA" },
  APF: { lat: 26.1526, lon: -81.7752, name: "Naples Municipal", city: "Naples", state: "FL" },
  MTN: { lat: 33.9151, lon: -84.5188, name: "Dobbins Air Reserve Base", city: "Marietta", state: "GA" },
  PDK: { lat: 33.8756, lon: -84.3020, name: "DeKalb-Peachtree", city: "Atlanta", state: "GA" },
  FTY: { lat: 33.7794, lon: -84.5214, name: "Fulton County (Brown Field)", city: "Atlanta", state: "GA" },
};

/** Event categories the FAA NAS status feed reports (server/faaStatus.ts
 *  FaaEvent["type"]). Color encodes SEVERITY-OF-DISRUPTION as a discrete
 *  category the feed itself reports — never a graded/inferred score, since
 *  "avg"/"max" fields are free-text durations ("2 hours and 30 minutes"),
 *  not reliably parseable numbers. */
export type FaaEventType = "ground_stop" | "closure" | "ground_delay" | "delay";

export const FAA_EVENT_LABEL: Record<FaaEventType, string> = {
  ground_stop: "Ground Stop",
  closure: "Airport Closure",
  ground_delay: "Ground Delay Program",
  delay: "Arrival/Departure Delay",
};

export const FAA_EVENT_COLOR: Record<FaaEventType, string> = {
  ground_stop: "#ff3b3b",   // no traffic accepted — most acute
  closure: "#c026d3",       // physically closed — distinct from a stop
  ground_delay: "#fb923c",  // GDP — metered arrivals
  delay: "#facc15",         // arrival/departure delay — mildest
};

export function faaEventColor(type: string | null | undefined): string {
  return (type && FAA_EVENT_COLOR[type as FaaEventType]) || "#94a3b8";
}

export function faaEventLabel(type: string | null | undefined): string {
  return (type && FAA_EVENT_LABEL[type as FaaEventType]) || String(type || "unknown");
}
