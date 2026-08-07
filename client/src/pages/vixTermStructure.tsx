// VixTermStructureView — Cboe volatility-index term structure
// (#/data/vix-term-structure). server/cboeVix.ts (GATE 1 DATA shipped
// v1.0.609, 2026-08-07) has archived VIX1D/VIX9D/VIX/VIX3M/VIX6M/VVIX
// daily since its build with no client view until now — this closes
// that gap, the same shape as the App Store rankings / GitHub activity
// precedents. RAW display only (kind:"raw"): the six levels are Cboe's
// own published values, and the two ratios are plain arithmetic on top
// of them (not an inference), so this needs no ladder gate — but the
// module's own docstring is explicit that any PREDICTIVE reading of the
// ratios (regime-input replacement candidate for KNOWN BROKEN #20's VXX
// proxy) stays SIGNAL-gate-locked, restated here rather than implied.
// Reuses the vt-filings shell + vt-gridstress-grid/tile/composite CSS
// (grid-stress precedent) — no new styles needed.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Waves } from "lucide-react";

interface VixDay {
  date: string;
  vix1d: number | null;
  vix9d: number | null;
  vix: number | null;
  vix3m: number | null;
  vix6m: number | null;
  vvix: number | null;
  vix_vix3m_ratio: number | null;
  vix9d_vix_ratio: number | null;
}
interface VixPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  latest: VixDay | null;
  recent: VixDay[];
  note?: string;
}

const lvl = (n: number | null) => (n == null ? "—" : n.toFixed(2));
const ratio = (n: number | null) => (n == null ? "—" : n.toFixed(3));

const TENOR_TILES: Array<{ key: keyof VixDay; label: string; sub: string }> = [
  { key: "vix1d", label: "VIX1D", sub: "1-day" },
  { key: "vix9d", label: "VIX9D", sub: "9-day" },
  { key: "vix", label: "VIX", sub: "30-day" },
  { key: "vix3m", label: "VIX3M", sub: "3-month" },
  { key: "vix6m", label: "VIX6M", sub: "6-month" },
  { key: "vvix", label: "VVIX", sub: "vol-of-vol" },
];

export default function VixTermStructureView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<VixPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/vix-term-structure");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const latest = data?.latest ?? null;
  const history = (data?.recent ?? []).slice().reverse().slice(0, 20);
  const contango = latest?.vix_vix3m_ratio != null && latest.vix_vix3m_ratio < 1;

  return (
    <div className="vt-filings-page" role="region" aria-label="Cboe volatility term structure">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Waves size={16} />
        <div>
          <div className="vt-filings-title">Volatility term structure — Cboe VIX complex</div>
          <div className="vt-filings-sub">
            VIX1D/VIX9D/VIX/VIX3M/VIX6M + VVIX daily close — RAW, no predictive claim ·{" "}
            {latest ? `as of ${latest.date}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://www.cboe.com/tradable_products/vix/" target="_blank" rel="noreferrer">Cboe Global Markets <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a moment on a cold start.</div>
      )}
      {!error && !data?.warming_up && !latest && (
        <div className="vt-filings-state">No day archived yet.</div>
      )}

      {!error && latest && (
        <div className="vt-shortvol-body">
          <div className="vt-gridstress-grid">
            {TENOR_TILES.map((t) => (
              <div className="vt-gridstress-tile" key={t.key}>
                <div className="vt-gridstress-tile-label">{t.label}</div>
                <div className="vt-gridstress-tile-value">{lvl(latest[t.key] as number | null)}</div>
                <div className="vt-gridstress-tile-sub">{t.sub}</div>
              </div>
            ))}
          </div>

          <div className="vt-gridstress-composite">
            <span className="vt-gridstress-composite-label">
              VIX / VIX3M — {contango ? "contango (normal)" : "backwardation (near-term stress)"}
            </span>
            <span className="vt-gridstress-composite-value">{ratio(latest.vix_vix3m_ratio)}</span>
          </div>
          <div className="vt-gridstress-composite">
            <span className="vt-gridstress-composite-label">
              VIX9D / VIX — front-end vs 30-day level
            </span>
            <span className="vt-gridstress-composite-value">{ratio(latest.vix9d_vix_ratio)}</span>
          </div>

          <div className="vt-filings-sub">
            ratios are plain arithmetic on Cboe's own published closes, not a fitted or inferred
            reading — any regime/predictive use stays gated pending SIGNAL-layer validation
          </div>

          {history.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th className="num">VIX1D</th>
                    <th className="num">VIX9D</th>
                    <th className="num">VIX</th>
                    <th className="num">VIX3M</th>
                    <th className="num">VIX6M</th>
                    <th className="num">VVIX</th>
                    <th className="num">VIX/VIX3M</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((row) => (
                    <tr key={row.date}>
                      <td data-l="Date">{row.date}</td>
                      <td data-l="VIX1D" className="num">{lvl(row.vix1d)}</td>
                      <td data-l="VIX9D" className="num">{lvl(row.vix9d)}</td>
                      <td data-l="VIX" className="num">{lvl(row.vix)}</td>
                      <td data-l="VIX3M" className="num">{lvl(row.vix3m)}</td>
                      <td data-l="VIX6M" className="num">{lvl(row.vix6m)}</td>
                      <td data-l="VVIX" className="num">{lvl(row.vvix)}</td>
                      <td data-l="VIX/VIX3M" className="num">{ratio(row.vix_vix3m_ratio)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {data?.note && <div className="vt-filings-sub vt-streams-foot">{data.note}</div>}
        </div>
      )}
    </div>
  );
}
