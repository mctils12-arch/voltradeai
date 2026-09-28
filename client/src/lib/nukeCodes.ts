/**
 * nukeCodes.ts — plain-English decodings for the "Nuclear Explosions
 * 1945-1998" catalog codes, verified against the catalog's own key:
 * Bergkvist & Ferm, FOA-R--00-01572-180--SE (Swedish Defence Research
 * Establishment / SIPRI, 2000), section 3.2 "Key to data", cross-checked
 * with DOE/NV-209 Rev 16 for US shots. Codes the FOA key itself leaves
 * undefined (SB, SPACE, SHIP) are labeled as inferred, never as official.
 *
 * HONESTY: every string is a decode of a documented catalog code or a dated
 * factual statement — no speculation about individual tests. Where the
 * catalog gives no purpose/type, the card says so instead of guessing.
 */

/** Purpose codes -> plain English (FOA key wording, lightly de-jargoned). */
export const NUKE_PURPOSE: Record<string, string> = {
  WR: "Weapons related — part of the weapon development program (also the catalog's default when no purpose was recorded)",
  WE: "Weapons effects — measuring what a detonation does to structures, hardware and materiel (US/UK/French shots)",
  SE: "Safety experiment — verifying a warhead won't produce nuclear yield in an accident (US/French shots)",
  PNE: "Peaceful nuclear explosion — civil/industrial application (Soviet PNEs include industrial explosions and PNE-technology tests)",
  "PNE:PLO": "Peaceful nuclear explosion — US Plowshare program (civil engineering applications)",
  "PNE:V": "Peaceful nuclear explosion — US Vela Uniform shot (research into detecting underground tests seismically)",
  FMS: "Soviet test to study the phenomena of a nuclear explosion",
  SAM: "Soviet test studying accidental modes and emergencies",
  SB: "Code left undefined by the catalog's key; the five SB shots are US Hardtack II safety experiments per DOE records (NV-209)",
  TRANSP: "Transportation-storage safety — the 1963 Roller Coaster shots at Nellis Air Force Range",
  COMBAT: "Combat use — the only two: Hiroshima and Nagasaki, August 1945",
  ME: "Military exercise with a real detonation — the one such test is the Soviet Totskoye exercise, 14 Sep 1954",
  NA: "Purpose not stated in the catalog",
};

/** Emplacement codes -> plain English (FOA key; truncated forms expanded). */
export const NUKE_TYPE: Record<string, string> = {
  SHAFT: "Vertical drilled/mined shaft — the detonation is buried and (usually) contained",
  "SHAFT/GR": "Well sunk into the ground of the atoll (French Polynesia pattern)",
  "SHAFT/LG": "Well drilled into the atoll lagoon floor (French Polynesia pattern)",
  TUNNEL: "Horizontal tunnel driven into a mountain or mesa",
  GALLERY: "Horizontal mined gallery — same class as tunnel shots (French In Ecker pattern)",
  UG: "Underground, emplacement not further specified in the catalog",
  MINE: "In a mine working (the one such shot is the Soviet KLIVAZH PNE, 1979)",
  CRATER: "Detonated in a crater at the surface",
  ATMOSPH: "Atmospheric burst, method of lofting not further specified in the catalog",
  AIRDROP: "Free-fall bomb dropped from an aircraft",
  TOWER: "Device mounted atop a steel or wooden tower",
  SURFACE: "On the ground, at or near the surface",
  BALLOON: "Device suspended from a tethered balloon",
  BARGE: "On a barge — the Pacific proving-grounds lagoon pattern",
  ROCKET: "Rocket-lofted to altitude",
  UW: "Underwater",
  SPACE: "Very high altitude (Soviet K-series rows; the catalog's key leaves this code undefined)",
  SHIP: "Aboard a ship — the one such shot is UK HURRICANE 1952, fired inside HMS Plym",
  WATERSUR: "At the water surface",
  "WATER SU": "At the water surface",
};

/** Decode a possibly slash-combined purpose code ("WR/SE") to prose. */
export function decodePurpose(p?: string | null): string {
  const raw = String(p || "").trim().toUpperCase();
  if (!raw) return NUKE_PURPOSE.NA;
  if (NUKE_PURPOSE[raw]) return NUKE_PURPOSE[raw];
  // combined codes: decode each part, join (width-truncated forms like
  // WR/F/SA = WR/FMS/SAM decode part-by-part with best-effort expansion)
  const EXPAND: Record<string, string> = { F: "FMS", SA: "SAM", P: "PNE", S: "SE", W: "WR" };
  const parts = raw.split("/").map((x) => {
    const k = EXPAND[x.trim()] || x.trim();
    const hit = NUKE_PURPOSE[k];
    return hit ? hit.split(" — ")[0] : k;
  });
  return parts.length > 1 ? `Multiple purposes: ${parts.join(" + ")}` : raw;
}

export function decodeType(t?: string | null): string {
  const raw = String(t || "").trim().toUpperCase();
  return NUKE_TYPE[raw] || (raw ? `Emplacement code "${raw}" (not in the documented code table)` : "Emplacement not stated in the catalog");
}

/** Which organization ran the testing program — country + date resolved
 *  against the documented agency timeline (year granularity; handovers
 *  mid-year are noted in the label). Factual attribution of the PROGRAM,
 *  not a claim about an individual shot's chain of command. */
export function testingAgency(country?: string | null, year?: number | null): string {
  const c = String(country || "").toUpperCase();
  const y = Number(year) || 0;
  switch (c) {
    case "USA":
      if (y <= 1946) return "Manhattan Engineer District (US Army) — the AEC took over 1 Jan 1947";
      if (y <= 1974) return "US Atomic Energy Commission (AEC); weapons-effects shots run with the Department of Defense";
      if (y <= 1977) return "US Energy Research & Development Administration (ERDA; became DOE Oct 1977)";
      return "US Department of Energy (DOE) national-laboratory weapons program";
    case "USSR":
      if (y <= 1952) return "Soviet First Chief Directorate — predecessor of the Ministry of Medium Machine Building";
      return "Soviet Ministry of Medium Machine Building (Minsredmash) — the USSR's nuclear-weapons ministry, with the Ministry of Defense running the test sites";
    case "UK":
      return y >= 1962
        ? "UK AWRE (later AWE), fired jointly with the USA at the Nevada Test Site (all UK tests from 1962)"
        : "UK Atomic Weapons Research Establishment (AWRE), Ministry of Supply — Australian and Pacific sites";
    case "FRANCE":
      return "France CEA Direction des applications militaires (DAM) with DIRCEN — Sahara 1960-66, French Polynesia 1966-96";
    case "CHINA":
      return y >= 1982
        ? "China Academy of Engineering Physics under COSTIND — Lop Nur test base"
        : "China Ninth Academy under the Second Ministry of Machine Building — Lop Nur test base";
    case "INDIA":
      return "India Bhabha Atomic Research Centre (BARC) / Department of Atomic Energy — Pokhran range";
    case "PAKIST":
      return "Pakistan Atomic Energy Commission (PAEC) — Chagai district sites";
    default:
      return "";
  }
}

/** Yield in plain terms: kt of TNT equivalent + a Hiroshima-scale anchor.
 *  Reference: Hiroshima "Little Boy" ~15 kt (LANL LA-8819; the catalog's own
 *  row also carries 15 kt). */
export function yieldContext(kt?: number | null): string {
  const y = Number(kt);
  if (!Number.isFinite(y) || y <= 0) return "Yield not stated in the catalog.";
  const hiro = y / 15;
  const scale =
    hiro >= 2 ? `≈${hiro >= 20 ? Math.round(hiro).toLocaleString() : hiro.toFixed(1)}× the Hiroshima bomb` :
    hiro >= 0.8 ? "≈ the Hiroshima bomb (~15 kt)" :
    `≈${Math.max(1, Math.round(hiro * 100))}% of the Hiroshima bomb`;
  return `${y.toLocaleString()} kt = ${(y * 1000).toLocaleString()} tons of TNT equivalent (${scale}).`;
}

/** Buried emplacements — blast contained, no surface blast ring. CRATER is
 *  NOT here: per the FOA key it means detonated IN a crater, at the surface. */
const BURIED = new Set(["SHAFT", "SHAFT/GR", "SHAFT/LG", "TUNNEL", "GALLERY", "UG", "MINE"]);

/** 5-psi blast-radius ESTIMATE in km — Glasstone & Dolan cube-root scaling,
 *  1 kt surface burst ≈ 0.47 km. An estimate of severe-blast-damage reach,
 *  NOT fallout modeling (fallout depends on weather/burst height the catalog
 *  doesn't record). Buried shots return null: the blast is contained. */
export function blastRadiusKm(kt?: number | null, emplacement?: string | null): number | null {
  const y = Number(kt);
  if (!Number.isFinite(y) || y <= 0) return null;
  if (BURIED.has(String(emplacement || "").toUpperCase().trim())) return null;
  return 0.47 * Math.cbrt(y);
}

/** Site codes -> plain English. The catalog's site field records WHERE a
 *  test was fired; the country field records WHO fired it (UK tests from
 *  1962 were fired at the US Nevada Test Site). Includes the unambiguous
 *  typo variants present in the source mirror (MUEUEOA/MURUHOA/HURUROA for
 *  Mururoa, MELLIS for Nellis, N2 RUSS for NZ RUSS). Codes that are
 *  ambiguous or unknown (e.g. MTR RUSS, HTR RUSS, KZ RUSS) are NOT decoded
 *  — the card shows the raw code rather than a guess. */
const NUKE_SITE: Record<string, string> = {
  "NTS": "Nevada Test Site, USA",
  "NELLIS NV": "Nellis Air Force Range, Nevada, USA",
  "MELLIS NV": "Nellis Air Force Range, Nevada, USA",
  "C. NEVADA": "Central Nevada, USA",
  "FALLON NV": "near Fallon, Nevada, USA",
  "AMCHITKA AK": "Amchitka Island, Alaska, USA",
  "ALAMOGORDO": "Alamogordo (Trinity site), New Mexico, USA",
  "CARLSBAD NM": "near Carlsbad, New Mexico, USA",
  "FARMINGT NM": "near Farmington, New Mexico, USA",
  "HATTIESB MS": "near Hattiesburg, Mississippi, USA",
  "HATTIESE MS": "near Hattiesburg, Mississippi, USA",
  "GRAND V CO": "Grand Valley, Colorado, USA",
  "RIFLE CO": "near Rifle, Colorado, USA",
  "ENEWETAK": "Enewetak Atoll, Marshall Islands (US Pacific Proving Grounds)",
  "BIKINI": "Bikini Atoll, Marshall Islands (US Pacific Proving Grounds)",
  "JOHNSTON IS": "Johnston Island, Pacific Ocean",
  "CHRISTMAS IS": "Christmas Island (Kiritimati), Pacific Ocean",
  "MALDEN IS": "Malden Island, Pacific Ocean",
  "PACIFIC": "Pacific Ocean",
  "OFFUSWCOAST": "Pacific Ocean, off the US west coast",
  "S.ATLANTIC": "South Atlantic Ocean",
  "S. ATLANTIC": "South Atlantic Ocean",
  "HIROSHIMA": "Hiroshima, Japan",
  "NAGASAKI": "Nagasaki, Japan",
  "SEMI KAZAKH": "Semipalatinsk Test Site, Kazakhstan (USSR)",
  "NZ RUSS": "Novaya Zemlya, Russia (USSR)",
  "N2 RUSS": "Novaya Zemlya, Russia (USSR)",
  "AZGIR KAZAKH": "Azgir, Kazakhstan (USSR)",
  "AZGIE KAZAKH": "Azgir, Kazakhstan (USSR)",
  "AZGIR": "Azgir, Kazakhstan (USSR)",
  "MURUROA": "Mururoa Atoll, French Polynesia",
  "MUEUEOA": "Mururoa Atoll, French Polynesia",
  "MURUHOA": "Mururoa Atoll, French Polynesia",
  "MURUEOA": "Mururoa Atoll, French Polynesia",
  "HURUROA": "Mururoa Atoll, French Polynesia",
  "W MURUROA": "west of Mururoa Atoll, French Polynesia",
  "WSW MURUROA": "west-southwest of Mururoa Atoll, French Polynesia",
  "FANGATAUFA": "Fangataufa Atoll, French Polynesia",
  "FANGATAUFAA": "Fangataufa Atoll, French Polynesia",
  "REGGANE ALG": "Reggane, Algeria (Sahara)",
  "IN ECKER ALG": "In Ekker, Algeria (Sahara)",
  "MARALI AUSTR": "Maralinga, South Australia",
  "EMU AUSTR": "Emu Field, South Australia",
  "MONTEB AUSTR": "Montebello Islands, Western Australia",
  "LOP NOR": "Lop Nur, Xinjiang, China",
  "POKHRAN": "Pokhran, Rajasthan, India",
  "CHAGAI": "Chagai Hills, Balochistan, Pakistan",
  "KHARAN": "Kharan Desert, Balochistan, Pakistan",
};

/** Plain-English site for a catalog site code, keeping the code visible so
 *  the decode is checkable. Unknown/ambiguous codes are returned raw. */
export function decodeSite(code?: string | null): string {
  const c = String(code || "").trim();
  if (!c) return "";
  const name = NUKE_SITE[c.toUpperCase()];
  return name ? `${name} (catalog code ${c})` : c;
}

function _hav(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const r = Math.PI / 180;
  const h = Math.sin(((lat2 - lat1) * r) / 2) ** 2
    + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lon2 - lon1) * r) / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(h));
}

const _deg = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(1)}°${v >= 0 ? pos : neg}`;

/** For records the site-consistency gate re-plotted (loc === "site", see
 *  scripts/nuclear_tests_site_check.py): says where the dot is, what the
 *  catalog's own coordinates were, and that the exact point is unknown.
 *  Returns "" for normally-located records. `fmtDist` renders km in the
 *  user's unit system (pass lib/units fmtKm). */
export function siteLocationNote(
  t: { loc?: string | null; r?: string | null; lat?: number; lon?: number; src_lat?: number; src_lon?: number },
  fmtDist: (km: number) => string,
): string {
  if (t.loc !== "site" || t.src_lat == null || t.src_lon == null || t.lat == null || t.lon == null) return "";
  const site = NUKE_SITE[String(t.r || "").toUpperCase()] || t.r || "its recorded site";
  const off = _hav(Number(t.src_lat), Number(t.src_lon), Number(t.lat), Number(t.lon));
  return `Map position: plotted at the center of ${site}. The source catalog's own coordinates for `
    + `this test (${_deg(Number(t.src_lat), "N", "S")}, ${_deg(Number(t.src_lon), "E", "W")}) lie `
    + `~${fmtDist(off)} away and contradict its recorded site, so the exact shot point is unknown.`;
}

// WHERE a test happened, as a present-day host country — distinct from who
// fired it (human, 2026-09-28, on a UK test card: "It's in the usa").
// Explicit per-site entries; Soviet regional labels resolve by their
// catalog republic suffix (RUSS / KAZAKH), which the catalog states — the
// region part may be ambiguous but the republic is not. Unknown -> "".
const NUKE_SITE_HOST: Record<string, string> = {
  "NTS": "USA", "NELLIS NV": "USA", "MELLIS NV": "USA", "FALLON NV": "USA", "C. NEVADA": "USA",
  "ALAMOGORDO": "USA", "CARLSBAD NM": "USA", "FARMINGT NM": "USA", "HATTIESB MS": "USA",
  "HATTIESE MS": "USA", "GRAND V CO": "USA", "RIFLE CO": "USA", "AMCHITKA AK": "USA (Alaska)",
  "JOHNSTON IS": "Johnston Atoll (US territory)",
  "ENEWETAK": "Marshall Islands (then US-administered)", "BIKINI": "Marshall Islands (then US-administered)",
  "CHRISTMAS IS": "Kiribati (then a British colony)", "MALDEN IS": "Kiribati (then a British colony)",
  "PACIFIC": "Pacific Ocean", "OFFUSWCOAST": "Pacific Ocean, off the US west coast",
  "S.ATLANTIC": "South Atlantic Ocean", "S. ATLANTIC": "South Atlantic Ocean",
  "HIROSHIMA": "Japan", "NAGASAKI": "Japan",
  "MURUROA": "French Polynesia", "MUEUEOA": "French Polynesia", "MURUHOA": "French Polynesia",
  "MURUEOA": "French Polynesia", "HURUROA": "French Polynesia", "W MURUROA": "French Polynesia",
  "WSW MURUROA": "French Polynesia", "FANGATAUFA": "French Polynesia", "FANGATAUFAA": "French Polynesia",
  "REGGANE ALG": "Algeria", "IN ECKER ALG": "Algeria",
  "MARALI AUSTR": "Australia", "EMU AUSTR": "Australia", "MONTEB AUSTR": "Australia",
  "LOP NOR": "China", "POKHRAN": "India", "CHAGAI": "Pakistan", "KHARAN": "Pakistan",
  "UZBEK": "Uzbekistan (then USSR)", "MARY TURKMEN": "Turkmenistan (then USSR)",
  "AZGIR": "Kazakhstan (then USSR)", "KAZAKHSTAN": "Kazakhstan (then USSR)",
};

/** Present-day host country of a catalog site code ("" when unknown). */
export function siteHostCountry(code?: string | null): string {
  const c = String(code || "").trim().toUpperCase();
  if (!c) return "";
  if (NUKE_SITE_HOST[c]) return NUKE_SITE_HOST[c];
  if (/\bRUS[SE]$/.test(c)) return "Russia (then USSR)";   // incl. catalog typo "JAKUTS RUSE"
  if (/\bKAZAKH$/.test(c)) return "Kazakhstan (then USSR)";
  return "";
}

/** True when the tester fired on its own soil — the card then doesn't
 *  repeat the country as "in …". */
export function testedAtHome(testerCode?: string | null, host?: string): boolean {
  const h = String(host || "");
  switch (String(testerCode || "").toUpperCase()) {
    case "USA": return h.startsWith("USA");
    case "USSR": return h.includes("then USSR");
    case "CHINA": return h === "China";
    case "INDIA": return h === "India";
    case "PAKIST": return h === "Pakistan";
    default: return false;
  }
}
