/** Plane identity decode (plane-tracking T1, human directive 2026-08-08:
 *  track planes by tail number, type, manufacturer, country).
 *
 *  Three deterministic decode tables — SYMBOLS-NOT-DOTS honesty rules
 *  apply: every decode comes from a documented public table, and anything
 *  outside the table returns null/"uncatalogued", never a guess.
 *
 *  1. countryFromIcao24 — ICAO 24-bit address allocation blocks (ICAO
 *     Annex 10 Vol III; the same public table every ADS-B site uses).
 *     Major blocks only; smaller states fall through to null.
 *  2. countryFromRegistration — registration-prefix decode (ITU/ICAO
 *     nationality marks), used as a fallback/cross-check.
 *  3. typeInfo — ICAO type designator → manufacturer + model for the
 *     common civil fleet; uncatalogued designators return the code
 *     itself so the UI never fabricates a model name.
 *
 *  mmsiFlag.ts is the structural precedent (vessel MID → flag state). */

type Range = [number, number, string];

// ICAO 24-bit allocation blocks, sorted by start. [startIncl, endIncl, country]
const ICAO_RANGES: Range[] = [
  [0x004000, 0x0043FF, "Zimbabwe"],
  [0x006000, 0x006FFF, "Mozambique"],
  [0x008000, 0x00FFFF, "South Africa"],
  [0x010000, 0x017FFF, "Egypt"],
  [0x018000, 0x01FFFF, "Libya"],
  [0x020000, 0x027FFF, "Morocco"],
  [0x028000, 0x02FFFF, "Tunisia"],
  [0x0A0000, 0x0A7FFF, "Algeria"],
  [0x0D0000, 0x0D7FFF, "Mexico"],
  [0x0D8000, 0x0DFFFF, "Venezuela"],
  [0x100000, 0x1FFFFF, "Russia"],
  [0x300000, 0x33FFFF, "Italy"],
  [0x340000, 0x37FFFF, "Spain"],
  [0x380000, 0x3BFFFF, "France"],
  [0x3C0000, 0x3FFFFF, "Germany"],
  [0x400000, 0x43FFFF, "United Kingdom"],
  [0x440000, 0x447FFF, "Austria"],
  [0x448000, 0x44FFFF, "Belgium"],
  [0x450000, 0x457FFF, "Bulgaria"],
  [0x458000, 0x45FFFF, "Denmark"],
  [0x460000, 0x467FFF, "Finland"],
  [0x468000, 0x46FFFF, "Greece"],
  [0x470000, 0x477FFF, "Hungary"],
  [0x478000, 0x47FFFF, "Norway"],
  [0x480000, 0x487FFF, "Netherlands"],
  [0x488000, 0x48FFFF, "Poland"],
  [0x490000, 0x497FFF, "Portugal"],
  [0x498000, 0x49FFFF, "Czechia"],
  [0x4A0000, 0x4A7FFF, "Romania"],
  [0x4A8000, 0x4AFFFF, "Sweden"],
  [0x4B0000, 0x4B7FFF, "Switzerland"],
  [0x4B8000, 0x4BFFFF, "Türkiye"],
  [0x4C8000, 0x4C87FF, "Ireland"],
  [0x4CA000, 0x4CAFFF, "Ireland"],
  [0x4D0000, 0x4D03FF, "Malta"],
  [0x500000, 0x5003FF, "San Marino"],
  [0x501000, 0x5013FF, "Albania"],
  [0x502C00, 0x502FFF, "Croatia"],
  [0x505C00, 0x505FFF, "Slovenia"],
  [0x506C00, 0x506FFF, "Slovakia"],
  [0x508000, 0x50FFFF, "Ukraine"],
  [0x600000, 0x67FFFF, "Ukraine"],
  [0x680000, 0x6F7FFF, "Kazakhstan"],
  [0x700000, 0x700FFF, "Afghanistan"],
  [0x702000, 0x702FFF, "Bangladesh"],
  [0x708000, 0x70FFFF, "Saudi Arabia"],
  [0x710000, 0x717FFF, "Saudi Arabia"],
  [0x718000, 0x71FFFF, "South Korea"],
  [0x720000, 0x727FFF, "North Korea"],
  [0x728000, 0x72FFFF, "Iraq"],
  [0x730000, 0x737FFF, "Iran"],
  [0x738000, 0x73FFFF, "Israel"],
  [0x740000, 0x747FFF, "Jordan"],
  [0x748000, 0x74FFFF, "Lebanon"],
  [0x750000, 0x757FFF, "Malaysia"],
  [0x758000, 0x75FFFF, "Philippines"],
  [0x760000, 0x767FFF, "Pakistan"],
  [0x768000, 0x76FFFF, "Singapore"],
  [0x770000, 0x777FFF, "Sri Lanka"],
  [0x778000, 0x77FFFF, "Syria"],
  [0x780000, 0x7BFFFF, "China"],
  [0x7C0000, 0x7FFFFF, "Australia"],
  [0x800000, 0x83FFFF, "India"],
  [0x840000, 0x87FFFF, "Japan"],
  [0x880000, 0x887FFF, "Thailand"],
  [0x888000, 0x88FFFF, "Viet Nam"],
  [0x890000, 0x890FFF, "Yemen"],
  [0x894000, 0x894FFF, "United Arab Emirates"],
  [0x895000, 0x8953FF, "Bahrain"],
  [0x896000, 0x896FFF, "Kuwait"],
  [0x897000, 0x8973FF, "Qatar"],
  [0x899000, 0x8993FF, "Taiwan"],
  [0x8A0000, 0x8A7FFF, "Indonesia"],
  [0x900000, 0x9FFFFF, "(reserved)"],
  [0xA00000, 0xAFFFFF, "United States"],
  [0xC00000, 0xC3FFFF, "Canada"],
  [0xC80000, 0xC87FFF, "New Zealand"],
  [0xC88000, 0xC88FFF, "Fiji"],
  [0xE00000, 0xE3FFFF, "Argentina"],
  [0xE40000, 0xE7FFFF, "Brazil"],
  [0xE80000, 0xE80FFF, "Chile"],
  [0xE84000, 0xE84FFF, "Ecuador"],
  [0xE88000, 0xE88FFF, "Paraguay"],
  [0xE8C000, 0xE8CFFF, "Peru"],
  [0xE90000, 0xE90FFF, "Uruguay"],
  [0xE94000, 0xE94FFF, "Bolivia"],
  [0x0AA000, 0x0AAFFF, "Colombia"],
];

/** Country of registry from the ICAO 24-bit address. null = outside the
 *  catalogued blocks (honest unknown, never a guess). */
export function countryFromIcao24(hex: string | null | undefined): string | null {
  if (!hex) return null;
  const v = parseInt(String(hex).trim(), 16);
  if (!Number.isFinite(v) || v < 0 || v > 0xFFFFFF) return null;
  for (const [a, b, c] of ICAO_RANGES) {
    if (v >= a && v <= b) return c === "(reserved)" ? null : c;
  }
  return null;
}

// Registration nationality prefixes (longest-match-first at lookup).
const REG_PREFIXES: Array<[string, string]> = [
  ["N", "United States"], ["C-", "Canada"], ["CF-", "Canada"],
  ["G-", "United Kingdom"], ["D-", "Germany"], ["F-", "France"],
  ["I-", "Italy"], ["EC-", "Spain"], ["PH-", "Netherlands"],
  ["OO-", "Belgium"], ["HB-", "Switzerland"], ["OE-", "Austria"],
  ["SE-", "Sweden"], ["LN-", "Norway"], ["OY-", "Denmark"], ["OH-", "Finland"],
  ["SP-", "Poland"], ["OK-", "Czechia"], ["EI-", "Ireland"], ["CS-", "Portugal"],
  ["SX-", "Greece"], ["TC-", "Türkiye"], ["UR-", "Ukraine"], ["RA-", "Russia"],
  ["JA", "Japan"], ["HL", "South Korea"], ["B-", "China/Taiwan/HK/Macau"],
  ["VT-", "India"], ["VH-", "Australia"], ["ZK-", "New Zealand"],
  ["PK-", "Indonesia"], ["9M-", "Malaysia"], ["9V-", "Singapore"],
  ["HS-", "Thailand"], ["VN-", "Viet Nam"], ["RP-", "Philippines"],
  ["AP-", "Pakistan"], ["4X-", "Israel"], ["A6-", "United Arab Emirates"],
  ["A7-", "Qatar"], ["A9C-", "Bahrain"], ["9K-", "Kuwait"], ["HZ-", "Saudi Arabia"],
  ["SU-", "Egypt"], ["ZS-", "South Africa"], ["5N-", "Nigeria"], ["ET-", "Ethiopia"],
  ["PP-", "Brazil"], ["PR-", "Brazil"], ["PS-", "Brazil"], ["PT-", "Brazil"], ["PU-", "Brazil"],
  ["LV-", "Argentina"], ["CC-", "Chile"], ["HK-", "Colombia"], ["XA-", "Mexico"],
  ["XB-", "Mexico"], ["XC-", "Mexico"], ["YV-", "Venezuela"], ["HC-", "Ecuador"],
];

/** Country from the registration's nationality mark. Longest prefix wins
 *  (A9C- before A-; PP-/PR-/PS-/PT- all Brazil). null when unrecognized. */
export function countryFromRegistration(reg: string | null | undefined): string | null {
  if (!reg) return null;
  const r = String(reg).trim().toUpperCase();
  if (!r) return null;
  let best: [string, string] | null = null;
  for (const e of REG_PREFIXES) {
    if (r.startsWith(e[0]) && (!best || e[0].length > best[0].length)) best = e;
  }
  return best ? best[1] : null;
}

// ICAO type designator → manufacturer + model, common civil fleet.
const TYPES: Record<string, [string, string]> = {
  // Boeing
  B712: ["Boeing", "717-200"], B722: ["Boeing", "727-200"],
  B732: ["Boeing", "737-200"], B733: ["Boeing", "737-300"], B734: ["Boeing", "737-400"],
  B735: ["Boeing", "737-500"], B736: ["Boeing", "737-600"], B737: ["Boeing", "737-700"],
  B738: ["Boeing", "737-800"], B739: ["Boeing", "737-900"],
  B37M: ["Boeing", "737 MAX 7"], B38M: ["Boeing", "737 MAX 8"], B39M: ["Boeing", "737 MAX 9"], B3XM: ["Boeing", "737 MAX 10"],
  B741: ["Boeing", "747-100"], B742: ["Boeing", "747-200"], B743: ["Boeing", "747-300"],
  B744: ["Boeing", "747-400"], B748: ["Boeing", "747-8"],
  B752: ["Boeing", "757-200"], B753: ["Boeing", "757-300"],
  B762: ["Boeing", "767-200"], B763: ["Boeing", "767-300"], B764: ["Boeing", "767-400"],
  B772: ["Boeing", "777-200"], B773: ["Boeing", "777-300"], B77L: ["Boeing", "777-200LR"], B77W: ["Boeing", "777-300ER"],
  B778: ["Boeing", "777-8"], B779: ["Boeing", "777-9"],
  B788: ["Boeing", "787-8"], B789: ["Boeing", "787-9"], B78X: ["Boeing", "787-10"],
  // Airbus
  A19N: ["Airbus", "A319neo"], A20N: ["Airbus", "A320neo"], A21N: ["Airbus", "A321neo"],
  A318: ["Airbus", "A318"], A319: ["Airbus", "A319"], A320: ["Airbus", "A320"], A321: ["Airbus", "A321"],
  A332: ["Airbus", "A330-200"], A333: ["Airbus", "A330-300"], A338: ["Airbus", "A330-800neo"], A339: ["Airbus", "A330-900neo"],
  A342: ["Airbus", "A340-200"], A343: ["Airbus", "A340-300"], A345: ["Airbus", "A340-500"], A346: ["Airbus", "A340-600"],
  A359: ["Airbus", "A350-900"], A35K: ["Airbus", "A350-1000"],
  A388: ["Airbus", "A380-800"],
  A306: ["Airbus", "A300-600"], A310: ["Airbus", "A310"],
  BCS1: ["Airbus", "A220-100"], BCS3: ["Airbus", "A220-300"],
  // Embraer
  E135: ["Embraer", "ERJ-135"], E145: ["Embraer", "ERJ-145"],
  E170: ["Embraer", "E170"], E175: ["Embraer", "E175"], E190: ["Embraer", "E190"], E195: ["Embraer", "E195"],
  E290: ["Embraer", "E190-E2"], E295: ["Embraer", "E195-E2"],
  E50P: ["Embraer", "Phenom 100"], E55P: ["Embraer", "Phenom 300"],
  E545: ["Embraer", "Legacy 450"], E550: ["Embraer", "Legacy 500"],
  // Bombardier / regional
  CRJ2: ["Bombardier", "CRJ200"], CRJ7: ["Bombardier", "CRJ700"], CRJ9: ["Bombardier", "CRJ900"], CRJX: ["Bombardier", "CRJ1000"],
  DH8A: ["De Havilland Canada", "Dash 8-100"], DH8B: ["De Havilland Canada", "Dash 8-200"],
  DH8C: ["De Havilland Canada", "Dash 8-300"], DH8D: ["De Havilland Canada", "Dash 8-400"],
  AT43: ["ATR", "42-300"], AT45: ["ATR", "42-500"], AT46: ["ATR", "42-600"],
  AT72: ["ATR", "72-200"], AT75: ["ATR", "72-500"], AT76: ["ATR", "72-600"],
  // Business jets
  C25A: ["Cessna", "Citation CJ2"], C25B: ["Cessna", "Citation CJ3"], C25C: ["Cessna", "Citation CJ4"],
  C525: ["Cessna", "Citation CJ1"], C550: ["Cessna", "Citation II"], C560: ["Cessna", "Citation V"],
  C56X: ["Cessna", "Citation Excel"], C680: ["Cessna", "Citation Sovereign"], C68A: ["Cessna", "Citation Latitude"],
  C700: ["Cessna", "Citation Longitude"], C750: ["Cessna", "Citation X"],
  CL30: ["Bombardier", "Challenger 300"], CL35: ["Bombardier", "Challenger 350"], CL60: ["Bombardier", "Challenger 600"],
  GL5T: ["Bombardier", "Global 5000"], GLEX: ["Bombardier", "Global Express"], GL7T: ["Bombardier", "Global 7500"],
  GLF4: ["Gulfstream", "G-IV"], GLF5: ["Gulfstream", "G-V"], GLF6: ["Gulfstream", "G650"],
  G280: ["Gulfstream", "G280"], GA5C: ["Gulfstream", "G500"], GA6C: ["Gulfstream", "G600"],
  F900: ["Dassault", "Falcon 900"], F2TH: ["Dassault", "Falcon 2000"], FA7X: ["Dassault", "Falcon 7X"], FA8X: ["Dassault", "Falcon 8X"],
  FA50: ["Dassault", "Falcon 50"],
  LJ35: ["Learjet", "35"], LJ45: ["Learjet", "45"], LJ60: ["Learjet", "60"], LJ75: ["Learjet", "75"],
  H25B: ["Hawker", "800"], HDJT: ["Honda", "HondaJet"],
  PC12: ["Pilatus", "PC-12"], PC24: ["Pilatus", "PC-24"],
  TBM9: ["Daher", "TBM 900"], TBM8: ["Daher", "TBM 850"],
  // GA
  C172: ["Cessna", "172 Skyhawk"], C182: ["Cessna", "182 Skylane"], C206: ["Cessna", "206 Stationair"],
  C208: ["Cessna", "208 Caravan"], C210: ["Cessna", "210 Centurion"],
  P28A: ["Piper", "PA-28 Cherokee"], P28B: ["Piper", "PA-28 Arrow"], PA31: ["Piper", "PA-31 Navajo"],
  PA34: ["Piper", "PA-34 Seneca"], PA46: ["Piper", "PA-46 Malibu"], M20P: ["Mooney", "M20"],
  SR20: ["Cirrus", "SR20"], SR22: ["Cirrus", "SR22"], SF50: ["Cirrus", "Vision Jet"],
  BE20: ["Beechcraft", "King Air 200"], B350: ["Beechcraft", "King Air 350"], BE36: ["Beechcraft", "Bonanza 36"],
  BE58: ["Beechcraft", "Baron 58"], BE9L: ["Beechcraft", "King Air 90"],
  DA40: ["Diamond", "DA40"], DA42: ["Diamond", "DA42"], DA62: ["Diamond", "DA62"],
  // Helicopters
  R22: ["Robinson", "R22"], R44: ["Robinson", "R44"], R66: ["Robinson", "R66"],
  B06: ["Bell", "206 JetRanger"], B407: ["Bell", "407"], B429: ["Bell", "429"], B412: ["Bell", "412"],
  EC30: ["Airbus Helicopters", "EC130"], EC35: ["Airbus Helicopters", "H135"], EC45: ["Airbus Helicopters", "H145"],
  H60: ["Sikorsky", "UH-60 Black Hawk"], S76: ["Sikorsky", "S-76"], S92: ["Sikorsky", "S-92"],
  AS50: ["Airbus Helicopters", "AS350 Écureuil"],
  // Freighters / other
  MD11: ["McDonnell Douglas", "MD-11"], MD82: ["McDonnell Douglas", "MD-82"], MD83: ["McDonnell Douglas", "MD-83"],
  MD88: ["McDonnell Douglas", "MD-88"], MD90: ["McDonnell Douglas", "MD-90"],
  A124: ["Antonov", "An-124"], A225: ["Antonov", "An-225"], AN26: ["Antonov", "An-26"],
  C130: ["Lockheed", "C-130 Hercules"], C17: ["Boeing", "C-17 Globemaster III"],
  K35R: ["Boeing", "KC-135"], B52: ["Boeing", "B-52"],
  SB20: ["Saab", "2000"], SF34: ["Saab", "340"], F100: ["Fokker", "100"], F70: ["Fokker", "70"],
  SU95: ["Sukhoi", "Superjet 100"],
};

/** Manufacturer + model for an ICAO type designator. Uncatalogued codes
 *  return {mfr: null, model: null, label: code} — the UI shows the raw
 *  code rather than a fabricated name. */
export function typeInfo(code: string | null | undefined): { mfr: string | null; model: string | null; label: string } {
  const c = String(code || "").trim().toUpperCase();
  if (!c) return { mfr: null, model: null, label: "" };
  const hit = TYPES[c];
  if (!hit) return { mfr: null, model: null, label: c };
  return { mfr: hit[0], model: hit[1], label: `${hit[0]} ${hit[1]}` };
}
