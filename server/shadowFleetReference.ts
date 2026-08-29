/**
 * shadowFleetReference.ts — GATE 1 (DATA) reference-list half of the
 * shadow-fleet signal validation plan (research/open_questions.md
 * "SHADOW-FLEET SIGNAL"): "build a reference list of publicly documented
 * shadow-fleet vessels (OFAC SDN vessel annexes + KSE Institute dark-fleet
 * publications provide MMSIs/IMOs)." This module builds the OFAC half —
 * the machine-readable one; KSE Institute's dark-fleet coverage is
 * narrative/PDF publications (already used qualitatively for
 * datacore/shadow_zones.json's STS-zone list) with no equivalent
 * structured MMSI export, so it stays a manual secondary cross-check
 * rather than a second ingest pipeline.
 *
 * Dependency-free XML parsing — same regex-block pattern as
 * server/euLoad.ts's tagBlocks/tagText (SDN.XML has a flat, regular
 * <sdnEntry><idList><id>... structure; a full XML-parser dependency is not
 * justified for one government feed).
 *
 * Source: OFAC (US Treasury, Office of Foreign Assets Control) Specially
 * Designated Nationals (SDN) list. US government work product — public
 * domain, no reuse restriction (same license class as CFTC/FRED/EDGAR
 * already used elsewhere in this codebase).
 */

export interface SdnVesselRef {
  uid: string;
  name: string;
  mmsi: string;
  imo?: string;
  vesselType?: string;
  vesselFlag?: string;
  programs: string[];
}

// The lookahead after `${tag}` matters: SDN.XML has sibling tags where one
// name is a prefix of another (program / programList, id / idList) — a
// bare `<${tag}[^>]*>` would let `<program[^>]*>` swallow the opening
// `<programList>` tag too (`[^>]*` matches "List"), pairing it with the
// wrong closing tag. Requiring the next character to be whitespace, `/`,
// or `>` rules that out.
const tagText = (block: string, tag: string): string | null => {
  const m = block.match(new RegExp(`<${tag}(?=[\\s\\/>])[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return m ? m[1].trim() : null;
};
const tagBlocks = (xml: string, tag: string): string[] => {
  const out: string[] = [];
  const re = new RegExp(`<${tag}(?=[\\s\\/>])[^>]*>([\\s\\S]*?)<\\/${tag}>`, "gi");
  let m;
  while ((m = re.exec(xml))) out.push(m[1]);
  return out;
};

export const OFAC_SDN_XML_URL = "https://www.treasury.gov/ofac/downloads/sdn.xml";

/** Parses the raw SDN.XML into Vessel-type entries that carry an MMSI
 *  idList entry. MMSI is the join key against our own AIS archive
 *  (server/shadowFleet.ts's `Pt`/GapEvent records key on MMSI), so a
 *  vessel entry we cannot join by MMSI is not useful as a gate-1
 *  reference row and is excluded rather than kept with a null field —
 *  callers should never have to re-derive "joinable" from a partial
 *  record. */
export function parseSdnVesselEntries(xml: string): SdnVesselRef[] {
  const out: SdnVesselRef[] = [];
  for (const entry of tagBlocks(xml, "sdnEntry")) {
    if ((tagText(entry, "sdnType") || "").trim() !== "Vessel") continue;
    const uid = tagText(entry, "uid") || "";
    const name = tagText(entry, "lastName") || "";
    let mmsi: string | null = null;
    let imo: string | null = null;
    for (const idBlock of tagBlocks(entry, "id")) {
      const idType = (tagText(idBlock, "idType") || "").trim();
      const idNumber = (tagText(idBlock, "idNumber") || "").trim();
      if (idType === "MMSI" && /^\d{7,9}$/.test(idNumber)) {
        mmsi = idNumber;
      } else if (/Vessel Registration Identification/i.test(idType)) {
        const m = idNumber.match(/IMO\s*(\d{5,8})/i);
        if (m) imo = m[1];
      }
    }
    if (!mmsi) continue;
    const programs = tagBlocks(entry, "program").map((p) => p.trim()).filter(Boolean);
    const vesselInfoBlock = tagBlocks(entry, "vesselInfo")[0] || "";
    const vesselType = tagText(vesselInfoBlock, "vesselType") || undefined;
    const vesselFlag = tagText(vesselInfoBlock, "vesselFlag") || undefined;
    out.push({ uid, name, mmsi, imo: imo || undefined, vesselType, vesselFlag, programs });
  }
  return out;
}

type FetchFn = (url: string, init?: RequestInit) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

/** Live fetch of the current SDN.XML. Treasury serves this via a redirect
 *  to a time-limited S3 URL; the injected fetchImpl must follow redirects
 *  (Node's global fetch does by default). */
export async function fetchOfacSdnXml(fetchImpl: FetchFn = fetch): Promise<string> {
  const r = await fetchImpl(OFAC_SDN_XML_URL, {
    headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
    signal: AbortSignal.timeout(60000),
  });
  if (!r.ok) throw new Error(`OFAC SDN fetch failed: http ${r.status}`);
  return r.text();
}
