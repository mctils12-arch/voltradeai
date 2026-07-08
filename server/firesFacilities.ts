// firesFacilities — the first BACKEND-INFERENCE cross-tie (worldview-globe
// Pillar 6): join live NASA active-fire detections to OUR strategic-facility
// archive to surface assets with active fires nearby. This is the entity-graph
// connective tissue the directive calls for — the globe SHOWS the fires and the
// sites; this computes the JOIN the backend can reason on.
//
// HONESTY / LADDER: this is a RAW cross-tie (gate-0 observation) — "facility X
// has N fire detections within R km right now." It is NOT a validated signal.
// The gate-2 hypothesis (fires near insured/industrial/utility assets predict
// insurer / utility / operator returns) is filed in research/open_questions.md
// and stays gated until an outcome study runs. Detections are VIIRS 375m NRT
// (~3h latency), NOT for safety-of-life use.

export interface Facility {
  id: string; name: string; lat: number; lon: number;
  category?: string; operator?: string;
}
export interface FireLike { lat: number; lon: number; frp?: number | null; confidence?: string }

export interface FacilityFireHit {
  id: string; name: string; lat: number; lon: number;
  category?: string; operator?: string;
  fires_within: number;   // detections inside the radius
  nearest_km: number;     // distance to the closest detection
  max_frp: number | null; // strongest fire radiative power (MW) among them — intensity proxy
}

/** Great-circle distance in km between two lat/lon points. Pure. */
export function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}

/**
 * For each facility, count the fire detections within `radiusKm`, the distance
 * to the nearest, and the strongest FRP among them. Returns ONLY facilities with
 * at least one nearby fire, sorted by proximity (nearest first). Pure.
 */
export function firesNearFacilities(
  facilities: Facility[], fires: FireLike[], radiusKm: number,
): FacilityFireHit[] {
  const out: FacilityFireHit[] = [];
  for (const f of facilities) {
    if (!Number.isFinite(f.lat) || !Number.isFinite(f.lon)) continue;
    let count = 0;
    let nearest = Infinity;
    let maxFrp: number | null = null;
    for (const fire of fires) {
      if (!Number.isFinite(fire.lat) || !Number.isFinite(fire.lon)) continue;
      const d = haversineKm(f.lat, f.lon, fire.lat, fire.lon);
      if (d <= radiusKm) {
        count += 1;
        if (d < nearest) nearest = d;
        if (fire.frp != null && (maxFrp == null || fire.frp > maxFrp)) maxFrp = fire.frp;
      }
    }
    if (count > 0) {
      out.push({
        id: f.id, name: f.name, lat: f.lat, lon: f.lon,
        category: f.category, operator: f.operator,
        fires_within: count, nearest_km: Math.round(nearest * 10) / 10, max_frp: maxFrp,
      });
    }
  }
  return out.sort((a, b) => a.nearest_km - b.nearest_km);
}
