// JodiOilStocksView — JODI World oil closing-stock levels (#/data/jodi-oil-stocks).
// server/jodiOil.ts (RAW view, shipped this session) reads the static
// datacore/jodi/primary_stocks.json archive (GATE 1 DATA passed 2026-08-06,
// GATE 2 SIGNAL killed same day) with no client view until now — same shape
// as the VIX term-structure / App Store rankings precedents. RAW display
// only (kind:"raw", predictive:false): the TOTCRUDE levels are countries'
// own self-reported figures, and the gate-2 kill status is restated in the
// UI verbatim (server's own note) rather than implied. Every row carries
// its OWN reporting period since staleness varies a lot by country — no
// single "as of" date is asserted for the whole table.
// Reuses the vt-filings shell (same as vix-term-structure) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, Droplet } from "lucide-react";

interface JodiRow {
  area: string;
  name: string;
  period: string;
  levelKbbl: number;
  assessmentCode: string;
  priorPeriod: string | null;
  priorLevelKbbl: number | null;
  deltaKbbl: number | null;
  deltaPct: number | null;
}
interface JodiPayload {
  kind: string;
  source?: string;
  attribution?: string;
  license?: string;
  product?: string;
  archiveLatestPeriod?: string;
  seriesCount?: number;
  countriesReporting?: number;
  rows: JodiRow[];
  note?: string;
}

const level = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 0 });
const delta = (n: number | null) => {
  if (n == null) return "—";
  const s = n > 0 ? "+" : "";
  return `${s}${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
};
const deltaPct = (n: number | null) => (n == null ? "—" : `${n > 0 ? "+" : ""}${n.toFixed(1)}%`);

export default function JodiOilStocksView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<JodiPayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/jodi-oil-stocks");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = data?.rows ?? [];

  return (
    <div className="vt-filings-page" role="region" aria-label="JODI world oil closing-stock levels">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Droplet size={16} />
        <div>
          <div className="vt-filings-title">World oil closing-stock levels — JODI</div>
          <div className="vt-filings-sub">
            TOTCRUDE closing stocks, thousand barrels, self-reported by {data?.countriesReporting ?? "…"} countries · RAW ·{" "}
            <a href="https://www.jodidata.org/oil/" target="_blank" rel="noreferrer">JODI Oil World Database <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data && rows.length === 0 && <div className="vt-filings-state">No TOTCRUDE series archived yet.</div>}

      {!error && data && rows.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data.note}
          </div>

          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Country</th>
                  <th>Period</th>
                  <th className="num">Level (kbbl)</th>
                  <th>vs. period</th>
                  <th className="num">Δ (kbbl)</th>
                  <th className="num">Δ%</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.area}>
                    <td data-l="Country">{row.name}</td>
                    <td data-l="Period">{row.period}</td>
                    <td data-l="Level (kbbl)" className="num">{level(row.levelKbbl)}</td>
                    <td data-l="vs. period">{row.priorPeriod ?? "—"}</td>
                    <td data-l="Δ (kbbl)" className="num">{delta(row.deltaKbbl)}</td>
                    <td data-l="Δ%" className="num">{deltaPct(row.deltaPct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="vt-filings-sub vt-streams-foot">
            {data.attribution} · {data.license} · archive latest period {data.archiveLatestPeriod} ·{" "}
            {data.seriesCount} total series archived (TOTCRUDE/CRUDEOIL/NGL/OTHERCRUDE combined) — this
            view shows TOTCRUDE only. Every row's period is its own country's last reported month, not a
            shared "as of" date — several areas stop reporting for years at a time.
          </div>
        </div>
      )}
    </div>
  );
}
