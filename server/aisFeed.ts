// SECOND AIS SOURCE — Fintraffic / Digitraffic (Finland), added 2026-08-12
// after aisstream.io went silent provider-side for its entire user base on
// 2026-08-05 (root cause + evidence trail in research/wishlist.md). This is
// deliberately NOT a replacement: aisstream's pitch was worldwide terrestrial
// coverage and nothing free matches that. It is a FLOOR — real, licensed,
// live positions in Finnish waters instead of a fully dark archive
// everywhere, and it removes the single-point-of-failure that cost 6+ days.
//
// LICENCE (MONETIZATION TRIPWIRE — this path is monetizable, so the licence
// decides admissibility): CC 4.0 BY. Fintraffic's terms grant distribution,
// remixing and COMMERCIAL use provided the source is credited. The required
// attribution string is exported below and must ride with every surface that
// shows this data. We already ingest Digitraffic under the same licence for
// the trains layer, so this is a proven pattern, not a new legal posture.
//
// HONESTY: positions are mapped straight through — no smoothing, no
// inference, no gap-filling. Vessels with no position are dropped rather
// than placed at a guessed location. Coverage is Finnish waterways only and
// every consumer is expected to say so rather than imply global coverage.

/** Attribution required by Fintraffic's CC 4.0 BY terms — verbatim. */
export const DIGITRAFFIC_AIS_ATTRIBUTION =
  "Source: Fintraffic / digitraffic.fi, license CC 4.0 BY";

export const DIGITRAFFIC_AIS_LOCATIONS_URL =
  "https://meri.digitraffic.fi/api/ais/v1/locations";
export const DIGITRAFFIC_AIS_VESSELS_URL =
  "https://meri.digitraffic.fi/api/ais/v1/vessels";

/** One normalized vessel fix, shaped to drop straight into the existing
 *  vesselPositions/vesselStatics maps the aisstream socket feeds. */
export interface AisFix {
  mmsi: string;
  lat: number;
  lon: number;
  /** knots, or null when the feed's not-available sentinel is present */
  sog: number | null;
  /** degrees, or null when the not-available sentinel (360) is present */
  cog: number | null;
  name: string | null;
  shiptype: number | null;
  destination: string | null;
  /** epoch SECONDS of the fix (Digitraffic sends ms) */
  at: number;
}

/** AIS not-available sentinels, per ITU-R M.1371: COG 360.0 means "not
 *  available" and SOG 102.3 means "not available" — passing either through
 *  as a real value would be inventing data. */
const COG_NA = 360;
const SOG_NA = 102.3;

const cleanStr = (v: unknown): string | null => {
  if (typeof v !== "string") return null;
  const s = v.trim();
  return s.length ? s : null;
};

/**
 * Map Digitraffic's AIS GeoJSON + vessel-metadata payloads into normalized
 * fixes. Pure: no clock reads, no network, no mutation of the inputs — the
 * caller passes `nowSec` so tests are deterministic.
 *
 * `locations` is the FeatureCollection from /api/ais/v1/locations; `vessels`
 * is the array from /api/ais/v1/vessels (static data: name, type,
 * destination). Metadata is optional — a position with no metadata entry
 * still yields a fix with null name/type, which is honest and renderable.
 */
export function mapDigitrafficAis(
  locations: any,
  vessels: any,
  nowSec: number,
): AisFix[] {
  const meta = new Map<string, { name: string | null; shiptype: number | null; destination: string | null }>();
  if (Array.isArray(vessels)) {
    for (const v of vessels) {
      const mmsi = v?.mmsi;
      if (mmsi == null) continue;
      meta.set(String(mmsi), {
        name: cleanStr(v.name),
        // Digitraffic exposes the raw AIS ship-type code; keep the code so
        // the existing symbol/decode tables keep working unchanged.
        shiptype: typeof v.shipType === "number" ? v.shipType
          : typeof v.shiptype === "number" ? v.shiptype : null,
        destination: cleanStr(v.destination),
      });
    }
  }

  const feats = Array.isArray(locations?.features) ? locations.features
    : Array.isArray(locations) ? locations : [];
  const out: AisFix[] = [];
  for (const f of feats) {
    const p = f?.properties || {};
    const mmsi = f?.mmsi ?? p.mmsi;
    if (mmsi == null) continue;
    const c = f?.geometry?.coordinates;
    if (!Array.isArray(c) || c.length < 2) continue;
    const lon = Number(c[0]);
    const lat = Number(c[1]);
    // a fix without a usable position is dropped, never guessed
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) continue;

    const sogRaw = typeof p.sog === "number" ? p.sog : null;
    const cogRaw = typeof p.cog === "number" ? p.cog : null;
    const m = meta.get(String(mmsi));
    // timestampExternal is epoch ms of the fix; fall back to now when absent
    const tMs = typeof p.timestampExternal === "number" ? p.timestampExternal : null;
    out.push({
      mmsi: String(mmsi),
      lat, lon,
      sog: sogRaw == null || sogRaw >= SOG_NA ? null : sogRaw,
      cog: cogRaw == null || cogRaw >= COG_NA ? null : cogRaw,
      name: m?.name ?? null,
      shiptype: m?.shiptype ?? null,
      destination: m?.destination ?? null,
      at: tMs != null ? Math.floor(tMs / 1000) : nowSec,
    });
  }
  return out;
}

/** Drop fixes older than `maxAgeSec` — a stale position rendered as live is
 *  the same dishonesty as a dead feed claiming live status. */
export function freshAisFixes(fixes: AisFix[], nowSec: number, maxAgeSec = 3600): AisFix[] {
  return fixes.filter((f) => nowSec - f.at <= maxAgeSec && f.at <= nowSec + 300);
}
