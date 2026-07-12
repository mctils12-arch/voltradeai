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
