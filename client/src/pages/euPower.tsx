// EuPowerView — ENTSO-E EU power markets: load, generation mix, day-ahead
// price (#/data/eu-power). Three RAW pipelines (server/euLoad.ts gate1_pass
// since 2026-07-07, server/euGenerationMix.ts since 2026-07-21,
// server/euDayAheadPrices.ts since 2026-07-27 — all wishlist 9c follow-ups,
// same ENTSOE_API_KEY/ENTSOE_TOKEN gate) had zero client view until now —
// this closes that gap, the same shape as the eu-macro/fred-macro/ats-summary
// precedents. RAW display only (kind:"raw"): realised load/generation and
// day-ahead auction clearing prices as ENTSO-E published them, never
// resampled, never zero-filled for absent zones (see `issues`). No ladder
// gate applies (no predictive claim). Reuses vt-filings-*/vt-gridstress-tile*
// CSS only — no new styles needed.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Plug } from "lucide-react";

interface LoadStat {
  zone: string; latest_ts: string; latest_mw: number | null; resolution: string;
  points_in_window: number; window_min_mw: number | null; window_max_mw: number | null; window_mean_mw: number | null;
}
interface GenMixStat {
  zone: string; psr: string; psr_name: string; latest_ts: string; latest_mw: number | null; resolution: string;
  points_in_window: number; window_min_mw: number | null; window_max_mw: number | null; window_mean_mw: number | null;
}
interface PriceStat {
  zone: string; latest_ts: string; latest_price: number | null; currency: string; unit: string; resolution: string;
  points_in_window: number; window_min_price: number | null; window_max_price: number | null;
  window_mean_price: number | null; negative_price_points: number;
}
interface EuPayload<T> {
  kind: string; enabled?: boolean; reason?: string; warming_up?: boolean;
  source?: string; attribution?: string; time?: string; note?: string;
  count?: number; zones: T[]; issues?: Record<string, string>;
}

type TabKey = "load" | "genmix" | "price";

const ZONE_NAMES: Record<string, string> = {
  DE_LU: "Germany–Luxembourg", FR: "France", ES: "Spain", IT: "Italy",
  NL: "Netherlands", PL: "Poland", BE: "Belgium", SE: "Sweden",
};
const zoneName = (z: string) => ZONE_NAMES[z] || z;

const fmtMw = (v: number | null) => v == null ? "—" : `${Math.round(v).toLocaleString()} MW`;
const fmtPrice = (v: number | null, currency: string, unit: string) =>
  v == null ? "—" : `${v.toFixed(2)} ${currency}/${unit}`;
const fmtTs = (ts: string) => ts ? ts.replace("T", " ") + "Z" : "—";

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    return await r.json();
  } catch {
    return null;
  }
}

function IssuesNote({ issues }: { issues?: Record<string, string> }) {
  const entries = Object.entries(issues || {});
  if (!entries.length) return null;
  return (
    <div className="vt-filings-sub">
      absent this cycle: {entries.map(([z, why]) => `${zoneName(z)} (${why})`).join("; ")}
    </div>
  );
}

export default function EuPowerView({ onBack }: { onBack: () => void }) {
  const [load, setLoad] = useState<EuPayload<LoadStat> | null>(null);
  const [genmix, setGenmix] = useState<EuPayload<GenMixStat> | null>(null);
  const [price, setPrice] = useState<EuPayload<PriceStat> | null>(null);
  const [error, setError] = useState(false);
  const [tab, setTab] = useState<TabKey>("load");
  const [genmixZone, setGenmixZone] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    (async () => {
      const [l, g, p] = await Promise.all([
        fetchJson<EuPayload<LoadStat>>("/api/data/eu-load"),
        fetchJson<EuPayload<GenMixStat>>("/api/data/eu-generation-mix"),
        fetchJson<EuPayload<PriceStat>>("/api/data/eu-day-ahead-prices"),
      ]);
      if (stop) return;
      if (!l && !g && !p) { setError(true); return; }
      setLoad(l); setGenmix(g); setPrice(p);
    })();
    return () => { stop = true; };
  }, []);

  const genmixZones = useMemo(() => {
    const seen = new Set<string>();
    (genmix?.zones || []).forEach((s) => seen.add(s.zone));
    return Array.from(seen).sort();
  }, [genmix]);
  const activeGenmixZone = genmixZone && genmixZones.includes(genmixZone) ? genmixZone : genmixZones[0] ?? null;
  const genmixRows = useMemo(
    () => (genmix?.zones || [])
      .filter((s) => s.zone === activeGenmixZone)
      .slice()
      .sort((a, b) => (b.latest_mw ?? -1) - (a.latest_mw ?? -1)),
    [genmix, activeGenmixZone],
  );

  const enabled = load?.enabled !== false;
  const anyWarming = load?.warming_up || genmix?.warming_up || price?.warming_up;
  const anyLive = (load?.zones?.length || 0) > 0 || (genmix?.zones?.length || 0) > 0 || (price?.zones?.length || 0) > 0;

  return (
    <div className="vt-filings-page" role="region" aria-label="EU power markets — ENTSO-E">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Plug size={16} />
        <div>
          <div className="vt-filings-title">EU power markets — ENTSO-E</div>
          <div className="vt-filings-sub">
            realised load, generation by fuel, day-ahead clearing price — 8 bidding zones, RAW, no predictive claim ·{" "}
            {anyLive ? "live" : anyWarming ? "warming up…" : enabled ? "loading…" : "awaiting API key"} ·{" "}
            <a href="https://transparency.entsoe.eu" target="_blank" rel="noreferrer">ENTSO-E Transparency Platform <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !enabled && (
        <div className="vt-filings-state">{load?.reason || "ENTSOE_API_KEY not set — awaiting a free ENTSO-E token."}</div>
      )}
      {!error && enabled && anyWarming && !anyLive && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && enabled && !anyWarming && !anyLive && (
        <div className="vt-filings-state">No data archived yet.</div>
      )}

      {!error && enabled && anyLive && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-filters" role="group" aria-label="EU power views">
            <button type="button" className="vt-filings-filter" aria-pressed={tab === "load"}
                    disabled={!load?.zones?.length}
                    style={tab === "load" ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                    onClick={() => setTab("load")}>
              Actual load
            </button>
            <button type="button" className="vt-filings-filter" aria-pressed={tab === "genmix"}
                    disabled={!genmix?.zones?.length}
                    style={tab === "genmix" ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                    onClick={() => setTab("genmix")}>
              Generation mix
            </button>
            <button type="button" className="vt-filings-filter" aria-pressed={tab === "price"}
                    disabled={!price?.zones?.length}
                    style={tab === "price" ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                    onClick={() => setTab("price")}>
              Day-ahead price
            </button>
          </div>

          {tab === "load" && load && (
            <>
              <div className="vt-filings-sub">{load.note}</div>
              <div className="vt-gridstress-grid">
                {load.zones.map((s) => (
                  <div className="vt-gridstress-tile" key={s.zone}>
                    <div className="vt-gridstress-tile-label">{zoneName(s.zone)}</div>
                    <div className="vt-gridstress-tile-value">{fmtMw(s.latest_mw)}</div>
                    <div className="vt-gridstress-tile-sub">
                      as of {fmtTs(s.latest_ts)} ({s.resolution}) · window {fmtMw(s.window_min_mw)}–{fmtMw(s.window_max_mw)}, mean {fmtMw(s.window_mean_mw)}
                    </div>
                  </div>
                ))}
              </div>
              <IssuesNote issues={load.issues} />
            </>
          )}

          {tab === "genmix" && genmix && (
            <>
              <div className="vt-filings-sub">{genmix.note}</div>
              <div className="vt-filings-filters" role="group" aria-label="Generation mix zone">
                {genmixZones.map((z) => (
                  <button key={z} type="button" className="vt-filings-filter" aria-pressed={activeGenmixZone === z}
                          style={activeGenmixZone === z ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                          onClick={() => setGenmixZone(z)}>
                    {zoneName(z)}
                  </button>
                ))}
              </div>
              {genmixRows.length > 0 ? (
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Fuel / technology</th><th className="num">Latest</th><th className="num">Window mean</th><th>As of</th></tr>
                    </thead>
                    <tbody>
                      {genmixRows.map((r) => (
                        <tr key={r.psr}>
                          <td data-l="Fuel">{r.psr_name}</td>
                          <td data-l="Latest" className="num">{fmtMw(r.latest_mw)}</td>
                          <td data-l="Window mean" className="num">{fmtMw(r.window_mean_mw)}</td>
                          <td data-l="As of">{fmtTs(r.latest_ts)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <div className="vt-filings-state">No generation mix archived yet for this zone.</div>}
              <IssuesNote issues={genmix.issues} />
            </>
          )}

          {tab === "price" && price && (
            <>
              <div className="vt-filings-sub">{price.note}</div>
              <div className="vt-gridstress-grid">
                {price.zones.map((s) => (
                  <div className="vt-gridstress-tile" key={s.zone}>
                    <div className="vt-gridstress-tile-label">{zoneName(s.zone)}</div>
                    <div className="vt-gridstress-tile-value">{fmtPrice(s.latest_price, s.currency, s.unit)}</div>
                    <div className="vt-gridstress-tile-sub">
                      as of {fmtTs(s.latest_ts)} · window {fmtPrice(s.window_min_price, s.currency, s.unit)}–{fmtPrice(s.window_max_price, s.currency, s.unit)}
                      {s.negative_price_points > 0 ? ` · ${s.negative_price_points} negative-price point${s.negative_price_points === 1 ? "" : "s"}` : ""}
                    </div>
                  </div>
                ))}
              </div>
              <IssuesNote issues={price.issues} />
            </>
          )}

          <div className="vt-filings-sub vt-streams-foot">
            {load?.attribution || genmix?.attribution || price?.attribution || "ENTSO-E Transparency Platform"} · per-series attribution required for reuse
          </div>
        </div>
      )}
    </div>
  );
}
