/**
 * operatorResolution.ts — registrant -> OPERATOR resolution (BUILD ORDER 4
 * #1, 2026-07-05; the blocker the fleet-utilization gate-1 pass surfaced:
 * 1,190+ archived airframes are registered to trustee/leasing shells that
 * hide the operating company).
 *
 * The free key insight: WE ALREADY RECORD CALLSIGNS. A hex flying UAL####
 * callsigns IS United operations regardless of who holds the FAA
 * registration. Resolution basis is labeled on every result (GIP evidence
 * envelope); anything unresolved stays registrant-labeled — never guessed.
 *
 * The prefix table below is DATA-DRIVEN: built from the top 3-letter ICAO
 * prefixes actually observed in our archive (2026-07-05 survey), majors +
 * regionals + fractionals + cargo. It is a deliberate honest SUBSET —
 * unknown prefixes resolve to null, and the table grows as the archive
 * surfaces new ones.
 */

export interface OperatorResolution {
  operator: string;
  icao: string;
  basis: "callsign-prefix";
  /** share of the hex's prefixed callsigns matching the winning prefix */
  agreement: number;
}

/** ICAO 3-letter telephony prefixes observed in OUR archive (top-N of the
 *  2026-07-05 survey), mapped to operator names. */
export const ICAO_OPERATORS: Record<string, string> = {
  AAL: "AMERICAN AIRLINES",
  SWA: "SOUTHWEST AIRLINES",
  DAL: "DELTA AIR LINES",
  UAL: "UNITED AIRLINES",
  JBU: "JETBLUE AIRWAYS",
  SKW: "SKYWEST AIRLINES",
  FFT: "FRONTIER AIRLINES",
  ENY: "ENVOY AIR",
  AAY: "ALLEGIANT AIR",
  RPA: "REPUBLIC AIRWAYS",
  JIA: "PSA AIRLINES",
  EDV: "ENDEAVOR AIR",
  ASA: "ALASKA AIRLINES",
  EJA: "NETJETS",
  LXJ: "FLEXJET",
  MXY: "BREEZE AIRWAYS",
  PDT: "PIEDMONT AIRLINES",
  DLH: "LUFTHANSA",
  RYR: "RYANAIR",
  UPS: "UPS AIRLINES",
  FDX: "FEDEX EXPRESS",
  NKS: "SPIRIT AIRLINES",
  HAL: "HAWAIIAN AIRLINES",
};

/** Wholly-owned regional carriers -> listed parent group (public, factual:
 *  Envoy/PSA/Piedmont are American Airlines Group subsidiaries; Endeavor is
 *  Delta's). The gate-1 self-consistency run surfaced exactly this class:
 *  parent-registered airframes flying subsidiary callsigns — resolution
 *  more precise than registration. SkyWest/Republic are deliberately NOT
 *  mapped: they are independent listed/private companies operating under
 *  capacity agreements, not subsidiaries. */
export const PARENT_GROUP: Record<string, string> = {
  "ENVOY AIR": "AMERICAN AIRLINES",
  "PSA AIRLINES": "AMERICAN AIRLINES",
  "PIEDMONT AIRLINES": "AMERICAN AIRLINES",
  "ENDEAVOR AIR": "DELTA AIR LINES",
};

export function operatorGroup(operator: string): string {
  return PARENT_GROUP[operator] || operator;
}

/** Registrants that are trustee/leasing shells — the REGISTRANT is known
 *  not to be the operator even when no callsign resolution exists. Pattern
 *  built from our spine's own composition (Bank of Utah 490 airframes, UMB
 *  328, Wilmington 301, TVPX 97, ...). */
const TRUSTEE_RE = /\b(TRUSTEE|TRUST CO|TRUST COMPANY|LEASING|LESSOR)\b|TVPX/;

export function isTrusteeRegistrant(owner: string | null | undefined): boolean {
  return !!owner && TRUSTEE_RE.test(owner.toUpperCase());
}

const PREFIX_RE = /^([A-Z]{3})\d/;
const MIN_SAMPLES = 2;
const MIN_AGREEMENT = 0.6;

/** Resolve a hex's operator from its archived callsigns. Requires >= 2
 *  prefixed samples with a >= 60% majority on a KNOWN prefix; otherwise
 *  null (bizjet callsigns are usually the registration itself — those
 *  stay registrant-labeled by design). */
export function resolveOperator(callsigns: Array<string | null | undefined>): OperatorResolution | null {
  const counts = new Map<string, number>();
  let prefixed = 0;
  for (const c of callsigns) {
    const m = PREFIX_RE.exec((c || "").toUpperCase().trim());
    if (!m) continue;
    prefixed++;
    counts.set(m[1], (counts.get(m[1]) || 0) + 1);
  }
  if (prefixed < MIN_SAMPLES) return null;
  let best: string | null = null;
  let bestN = 0;
  counts.forEach((n, p) => { if (n > bestN) { bestN = n; best = p; } });
  if (!best || !(best in ICAO_OPERATORS)) return null;
  const agreement = bestN / prefixed;
  if (agreement < MIN_AGREEMENT) return null;
  return { operator: ICAO_OPERATORS[best], icao: best, basis: "callsign-prefix", agreement: +agreement.toFixed(3) };
}
