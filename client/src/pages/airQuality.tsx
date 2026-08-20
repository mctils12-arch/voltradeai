// AirQualityView — current air-quality readings at our 16 strategic sites
// (#/data/air-quality). server/airQuality.ts shipped the API-only route with
// no client view — the same "shipped-data-no-UI" gap class as drought/midas/
// fires-near-facilities before it, and the next of the 3 named in the
// 2026-08-20 fires session's own NEXT note. RAW display only (gate-0
// observation): no predictive claim is made or implied here — the
// air-quality-at-industrial-sites → activity hypothesis stays filed and
// ladder-gated in open_questions.md. Not a spatial map layer (a per-site
// reading table over the existing strategic-sites archive, not its own
// registry entry), so it launches from the panel-top "Streams inventory"
// list like drought/fires-near-facilities. Reuses the generic .vt-filings-*
// table CSS — no new styles.
//
// The two AQI columns run in OPPOSITE directions and are never combined or
// ramped by us; see client/src/lib/aqIndex.ts for the evidence and the
// provider-color decode that keeps that honest.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Wind } from "lucide-react";
import { indexColorCss, indexTextColor, unitSymbol, readingAge, type AqIndexColor } from "@/lib/aqIndex";

interface AqIndexRec {
  code?: string;
  displayName?: string;
  aqi?: number | null;
  aqiDisplay?: string;
  color?: AqIndexColor;
  category?: string;
  dominantPollutant?: string;
}
interface AqReading {
  siteId: string;
  dateTime: string;
  uaqi: number | null;
  uaqi_category: string | null;
  uaqi_dominant: string | null;
  epa_aqi: number | null;
  epa_category: string | null;
  epa_dominant: string | null;
  pm25: number | null;
  pm25_units: string | null;
  no2: number | null;
  no2_units: string | null;
  indexes?: AqIndexRec[];
  rt: string;
}
interface AqBudget {
  monthly_free_tier?: number;
  daily_call_budget?: number;
  poll_interval_hours?: number;
  site_count?: number;
  sites_this_cycle?: number;
  rotating_subset?: boolean;
  projected_calls_per_day?: number;
}
interface AqPayload {
  kind?: string;
  enabled?: boolean;
  awaiting_key?: boolean;
  warming_up?: boolean;
  awaiting_enable?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  note?: string;
  count?: number;
  budget?: AqBudget;
  sites?: AqReading[];
  issues?: Record<string, string>;
}

const conc = (v: number | null, units: string | null) =>
  v == null ? "—" : `${v.toFixed(2)} ${unitSymbol(units)}`.trim();

/** The provider's own color for one index code, when it sent one. */
const colorFor = (r: AqReading, code: string): AqIndexColor | null =>
  (r.indexes ?? []).find((x) => x?.code === code)?.color ?? null;

function AqiChip({ value, color }: { value: number | null; color: AqIndexColor | null }) {
  if (value == null) return <span>—</span>;
  const bg = indexColorCss(color);
  return (
    <span
      style={{
        display: "inline-block",
        minWidth: 34,
        padding: "1px 7px",
        borderRadius: 5,
        fontVariantNumeric: "tabular-nums",
        fontWeight: 600,
        background: bg ?? "var(--surface-2, rgba(255,255,255,0.08))",
        color: bg ? indexTextColor(color) : "var(--text-primary)",
      }}
    >
      {value}
    </span>
  );
}

export default function AirQualityView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<AqPayload | null>(null);
  const [error, setError] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/air-quality");
        const d = await r.json();
        if (!stop) { setData(d); setNow(Date.now()); }
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const sites = data?.sites ?? [];
  const issues = Object.entries(data?.issues ?? {});
  const budget = data?.budget;
  const inactive = data ? data.enabled === false || data.awaiting_key === true : false;

  return (
    <div className="vt-filings-page" role="region" aria-label="Air quality at strategic sites">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Wind size={16} />
        <div>
          <div className="vt-filings-title">Air quality at strategic sites</div>
          <div className="vt-filings-sub">
            current conditions at our tracked ports, hubs and mills — RAW readings, no predictive claim ·{" "}
            {data?.attribution ?? ""} ·{" "}
            <a href="https://developers.google.com/maps/documentation/air-quality" target="_blank" rel="noreferrer">
              Air Quality API <ExternalLink size={11} />
            </a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && inactive && (
        <div className="vt-filings-state">GOOGLE_MAPS_API_KEY not set — air-quality feed inactive.</div>
      )}
      {!error && data && !inactive && data.awaiting_enable && (
        <div className="vt-filings-state">
          The Air Quality API is not yet enabled on the Google Cloud project — readings populate on the next poll once it is.
        </div>
      )}
      {!error && data && !inactive && !data.awaiting_enable && data.warming_up && (
        <div className="vt-filings-state">Warming up — the first poll is still in flight.</div>
      )}

      {!error && data && !inactive && !data.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            The two indexes below run in <strong>opposite directions</strong> and are never combined here:
            Universal AQI is 0–100 where <strong>higher is cleaner</strong>; US EPA AQI is 0–500 where{" "}
            <strong>higher is dirtier</strong>. Each chip is painted with the colour the provider sent for
            that reading, not a scale of ours.
          </div>
          {data.note && <div className="vt-filings-sub">{data.note}</div>}
          {budget && (
            <div className="vt-filings-sub">
              Free-tier budget: polling every {budget.poll_interval_hours}h across {budget.site_count} sites
              {budget.rotating_subset ? " (rotating subset — not every site is read every cycle)" : " (every site every cycle)"}
              {budget.projected_calls_per_day != null && budget.daily_call_budget != null
                ? ` · ~${budget.projected_calls_per_day} of ${budget.daily_call_budget} calls/day`
                : ""}
            </div>
          )}
          {issues.length > 0 && (
            <div className="vt-filings-sub">
              Sites with a read issue this cycle: {issues.map(([id, msg]) => `${id} (${msg})`).join(" · ")}
            </div>
          )}

          {sites.length > 0 ? (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Site</th>
                    <th>Observed</th>
                    <th className="num">Universal AQI</th>
                    <th className="num">US EPA AQI</th>
                    <th>Dominant</th>
                    <th className="num">PM2.5</th>
                    <th className="num">NO2</th>
                  </tr>
                </thead>
                <tbody>
                  {sites.map((s) => {
                    const age = readingAge(s.dateTime, now);
                    const dom = s.epa_dominant ?? s.uaqi_dominant;
                    const domAlt =
                      s.uaqi_dominant && s.epa_dominant && s.uaqi_dominant !== s.epa_dominant
                        ? ` / ${s.uaqi_dominant} (UAQI)`
                        : "";
                    return (
                      <tr key={s.siteId}>
                        <td data-l="Site">{s.siteId}</td>
                        <td data-l="Observed" title={s.dateTime}>{age ?? "—"}</td>
                        <td data-l="Universal AQI" className="num">
                          <AqiChip value={s.uaqi} color={colorFor(s, "uaqi")} />
                          <div className="vt-filings-sub">{s.uaqi_category ?? ""}</div>
                        </td>
                        <td data-l="US EPA AQI" className="num">
                          <AqiChip value={s.epa_aqi} color={colorFor(s, "usa_epa")} />
                          <div className="vt-filings-sub">{s.epa_category ?? ""}</div>
                        </td>
                        <td data-l="Dominant">{dom ? dom + domAlt : "—"}</td>
                        <td data-l="PM2.5" className="num">{conc(s.pm25, s.pm25_units)}</td>
                        <td data-l="NO2" className="num">{conc(s.no2, s.no2_units)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="vt-filings-state">No readings in the current cycle.</div>
          )}
        </div>
      )}
    </div>
  );
}
