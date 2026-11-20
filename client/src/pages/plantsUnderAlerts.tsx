// PlantsUnderAlertsView — power plants inside an active NWS warning polygon
// (#/data/plants-under-alerts). server/plantsUnderAlerts.ts shipped the join
// and its route (worldview-globe Pillar 6) with no client view; UI coverage
// batch C (2026-11-20) gives it the same launcher-and-panel form as
// "Facilities near active fires". RAW cross-tie, no predictive claim: the
// warning-over-generation hypothesis is filed and gated, and nothing on this
// page reflects it. Not a map layer of its own (the plants and the warnings
// are each already layers); this is the join between them.
//
// Sentences and figures come from client/src/lib/dataPanelsView.ts (tested).
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Zap } from "lucide-react";
import { mwLabel, fuelLine, endsLabel, ageLabel } from "@/lib/dataPanelsView";

interface PlantHit { name: string; fuel: string; capacity_mw: number; operator?: string | null; }
interface AlertHit {
  id: string;
  event: string;
  severity?: string | null;
  area?: string | null;
  ends?: string | null;
  plants: PlantHit[];
  plant_count: number;
  total_capacity_mw: number;
  capacity_by_fuel: Record<string, number>;
}
interface Payload {
  kind?: string;
  warming_up?: boolean;
  as_of?: number;
  alerts_checked?: number;
  zone_only_excluded?: number;
  plants_checked?: number;
  alerts_with_plants?: number;
  alerts?: AlertHit[];
  note?: string;
  source?: string;
  error?: string;
}

// A hurricane warning can cover hundreds of plants; the list opens on the
// largest and says how many more there are.
const FIRST = 8;

function AlertBlock({ a, now }: { a: AlertHit; now: number }) {
  const [all, setAll] = useState(false);
  const shown = all ? a.plants : a.plants.slice(0, FIRST);
  const rest = a.plants.length - shown.length;
  return (
    <div className="vt-pua-alert" data-vt-pua-alert>
      <div className="vt-pua-alert-head">
        <span className="vt-pua-alert-name">{a.event}{a.area ? ` — ${a.area}` : ""}</span>
        <span className="vt-pua-alert-meta">
          {a.severity ? `${a.severity} · ` : ""}{endsLabel(a.ends, now)}
        </span>
      </div>
      <div className="vt-filings-sub" data-vt-pua-sum>
        {a.plant_count} plant{a.plant_count === 1 ? "" : "s"} inside the polygon · {mwLabel(a.total_capacity_mw)} nameplate · {fuelLine(a.capacity_by_fuel)}
      </div>
      <div className="vt-filings-tablewrap vt-tablewrap-flow">
        <table className="vt-filings-table">
          <thead>
            <tr><th>Plant</th><th>Fuel</th><th>Operator</th><th className="num">Nameplate capacity</th></tr>
          </thead>
          <tbody>
            {shown.map((p, i) => (
              <tr key={`${p.name}-${i}`}>
                <td data-l="Plant">{p.name}</td>
                <td data-l="Fuel">{p.fuel || "—"}</td>
                <td data-l="Operator">{p.operator || "—"}</td>
                <td data-l="Nameplate capacity" className="num">{mwLabel(p.capacity_mw)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(rest > 0 || all) && a.plants.length > FIRST && (
        <button type="button" className="vt-pua-more" data-vt-pua-more aria-expanded={all} onClick={() => setAll((v) => !v)}>
          {all ? `Show the largest ${FIRST} only` : `Show the other ${rest} plant${rest === 1 ? "" : "s"} (smaller)`}
        </button>
      )}
    </div>
  );
}

export default function PlantsUnderAlertsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/plants-under-alerts");
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (!stop) { setData(d); setError(false); setNow(Date.now()); }
      } catch {
        if (!stop) setError(true);
      }
    };
    load();
    // Warnings change by the minute; the server reads its own poll cache, so
    // a 2-minute re-read costs one in-memory join.
    const t = window.setInterval(load, 120000);
    return () => { stop = true; window.clearInterval(t); };
  }, []);

  const alerts = data?.alerts ?? [];
  const ready = !!data && !data.warming_up;

  return (
    <div className="vt-filings-page" role="region" aria-label="Power plants under active weather warnings">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Zap size={16} />
        <div>
          <div className="vt-filings-title">Power plants under weather warnings — NWS</div>
          <div className="vt-filings-sub">
            generating capacity inside an active warning polygon right now — RAW cross-tie, no predictive claim ·{" "}
            <span data-vt-pua-age>{ready ? `warnings read ${ageLabel(data!.as_of, now)}` : data?.warming_up ? "warming up…" : "loading…"}</span> ·{" "}
            <a href="https://www.weather.gov/documentation/services-web-api" target="_blank" rel="noreferrer">NWS api.weather.gov <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && !data && <div className="vt-filings-state">The join did not load. A refresh asks again.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {data?.warming_up && <div className="vt-filings-state">The weather-warning feed is still warming up after a restart.</div>}

      {ready && (
        <div className="vt-shortvol-body">
          {error && <div className="vt-filings-sub">The last refresh failed; this is the previous reading.</div>}
          <div className="vt-filings-sub" data-vt-pua-scope>
            {(data!.alerts_checked ?? 0).toLocaleString("en-US")} warning{data!.alerts_checked === 1 ? "" : "s"} with a polygon checked against{" "}
            {(data!.plants_checked ?? 0).toLocaleString("en-US")} US plants · {alerts.length} cover{alerts.length === 1 ? "s" : ""} at least one plant
            {typeof data!.zone_only_excluded === "number" && data!.zone_only_excluded > 0
              ? ` · ${data!.zone_only_excluded.toLocaleString("en-US")} alerts are issued for a forecast zone with no polygon and could not be checked`
              : ""}
          </div>
          {alerts.length === 0 ? (
            <div className="vt-filings-state">No active warning polygon covers a plant in the database right now.</div>
          ) : (
            alerts.map((a) => <AlertBlock key={a.id} a={a} now={now} />)
          )}
          <div className="vt-filings-sub vt-streams-foot">
            Plant locations and nameplate capacity are from the WRI Global Power Plant Database (2021 vintage, CC BY 4.0): plants built
            since are missing, and a plant inside a warning is not a plant that is offline. Not for safety-of-life use.
            {data!.source ? ` Source: ${data!.source}.` : ""}
          </div>
        </div>
      )}
    </div>
  );
}
