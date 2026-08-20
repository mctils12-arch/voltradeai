// FiresNearFacilitiesView — strategic assets with active NASA FIRMS fire
// detections nearby (#/data/fires-near-facilities). server/routes.ts shipped
// the API-only route (server/firesFacilities.ts, worldview-globe Pillar 6
// backend-inference cross-tie) with no client view — same "shipped-data-
// no-UI" gap class as drought/midas/atsSummary before it (this is the next
// of the 4 named in the 2026-08-20 drought session's own NEXT note). RAW
// cross-tie (gate-0 observation), NOT a validated signal — the fires-near-
// assets -> insurer/utility/operator-returns hypothesis is filed in
// open_questions.md and stays gated. Not a spatial map layer (it's a
// derived join over the existing fires feed + strategic-sites archive, not
// its own registry entry), so it launches from the panel-top "Streams
// inventory" list like drought/crop-conditions. Reuses the generic
// .vt-filings-* table CSS (midas.tsx precedent) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Flame } from "lucide-react";

interface FacilityFireHit {
  id: string;
  name: string;
  lat: number;
  lon: number;
  category?: string;
  operator?: string;
  fires_within: number;
  nearest_km: number;
  max_frp: number | null;
}
interface FiresNearFacilitiesPayload {
  enabled?: boolean;
  awaiting_key?: boolean;
  warming_up?: boolean;
  kind?: string;
  as_of?: number;
  radius_km?: number;
  fires_checked?: number;
  facilities_with_fires?: number;
  facilities?: FacilityFireHit[];
  note?: string;
  source?: string;
}

const frp = (n: number | null) => (n == null ? "—" : `${n.toFixed(1)} MW`);
const km = (n: number) => `${n.toFixed(1)} km`;

export default function FiresNearFacilitiesView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FiresNearFacilitiesPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/fires-near-facilities");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const facilities = data?.facilities ?? [];

  return (
    <div className="vt-filings-page" role="region" aria-label="Strategic facilities near active fires">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Flame size={16} />
        <div>
          <div className="vt-filings-title">Facilities near active fires — NASA FIRMS</div>
          <div className="vt-filings-sub">
            strategic assets with VIIRS fire detections nearby right now — RAW cross-tie, no predictive claim ·{" "}
            {data?.radius_km != null ? `${data.radius_km} km radius` : ""} ·{" "}
            <a href="https://firms.modaps.eosdis.nasa.gov" target="_blank" rel="noreferrer">NASA FIRMS <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data && data.awaiting_key && (
        <div className="vt-filings-state">NASA_FIRMS_MAP_KEY not set — fires feed inactive.</div>
      )}
      {!error && data && !data.awaiting_key && data.warming_up && (
        <div className="vt-filings-state">Fires feed warming up — first detections still loading.</div>
      )}

      {!error && data && data.enabled && !data.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data.note ? data.note + " " : ""}
          </div>
          <div className="vt-filings-sub">
            {data.fires_checked != null ? `${data.fires_checked.toLocaleString()} active detections checked` : ""}
            {data.facilities_with_fires != null ? ` · ${data.facilities_with_fires} of our strategic facilities have a detection within ${data.radius_km} km` : ""}
          </div>

          {facilities.length > 0 ? (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Facility</th><th>Category</th><th>Operator</th>
                    <th className="num">Fires within {data.radius_km} km</th>
                    <th className="num">Nearest</th><th className="num">Max FRP</th>
                  </tr>
                </thead>
                <tbody>
                  {facilities.map((f) => (
                    <tr key={f.id}>
                      <td data-l="Facility">{f.name}</td>
                      <td data-l="Category">{f.category ?? "—"}</td>
                      <td data-l="Operator">{f.operator ?? "—"}</td>
                      <td data-l="Fires within" className="num">{f.fires_within}</td>
                      <td data-l="Nearest" className="num">{km(f.nearest_km)}</td>
                      <td data-l="Max FRP" className="num">{frp(f.max_frp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="vt-filings-state">No strategic facility currently has an active fire detection within {data.radius_km} km.</div>
          )}
        </div>
      )}
    </div>
  );
}
