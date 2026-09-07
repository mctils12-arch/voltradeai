// UnComtradeView — USA bilateral goods-trade archive (#/data/un-comtrade).
// server/unComtrade.ts (RAW view, shipped 2026-09-07, GATE 1 DATA passed
// same day — see datacore/signal_ladder.json's un_comtrade_bilateral_trade
// entry) reads the static datacore/un_comtrade/bilateral_trade.json archive
// with no client view until now — same shipped-data-no-client-page-gap
// pattern this repo has closed for JODI/FINRA/plant-operations before.
// RAW display only (kind:"raw", predictive:false): USA vs its 6 largest
// tracked partners, cmdCode=TOTAL (UN's own computed aggregate, not
// commodity-level detail). No spatial coordinates in the source feed, so
// this launches from the panel-top "streams" list like fred-macro/
// jodi-oil-stocks/eu-macro, not from a map-layer toggle.
// Reuses the vt-filings shell (same as jodiOilStocks.tsx) — no new styles.
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, ArrowLeftRight } from "lucide-react";

interface UnComtradeRow {
  partnerCode: number;
  partnerName: string;
  period: string;
  importsCifUsd: number | null;
  exportsFobUsd: number | null;
  tradeBalanceUsd: number | null;
  priorPeriod: string | null;
  priorImportsCifUsd: number | null;
  importsDeltaPct: number | null;
}
interface UnComtradePayload {
  kind: string;
  source?: string;
  attribution?: string;
  license?: string;
  reporter?: string;
  cmdCode?: string;
  archiveLatestPeriod?: string | null;
  partnerCount?: number;
  rows: UnComtradeRow[];
  note?: string;
}

const usd = (n: number | null) => {
  if (n == null) return "—";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  return `${sign}$${abs.toLocaleString()}`;
};
const pct1 = (n: number | null) => (n == null ? "—" : `${n > 0 ? "+" : ""}${n.toFixed(1)}%`);
const fmtPeriod = (p: string) => (p.length === 6 ? `${p.slice(0, 4)}-${p.slice(4)}` : p);

export default function UnComtradeView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<UnComtradePayload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/un-comtrade");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = data?.rows ?? [];
  const maxAbsBalance = Math.max(1, ...rows.map((r) => Math.abs(r.tradeBalanceUsd ?? 0)));

  return (
    <div className="vt-filings-page" role="region" aria-label="USA bilateral goods trade — UN Comtrade">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <ArrowLeftRight size={16} />
        <div>
          <div className="vt-filings-title">USA bilateral goods trade — UN Comtrade</div>
          <div className="vt-filings-sub">
            {data?.reporter ?? "USA"} vs {data?.partnerCount ?? "…"} major partners, monthly · RAW ·{" "}
            <a href="https://comtradeapi.un.org" target="_blank" rel="noreferrer">UN Comtrade Database <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data && rows.length === 0 && <div className="vt-filings-state">No bilateral series archived yet.</div>}

      {!error && data && rows.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {data.note}
          </div>

          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Partner</th>
                  <th>Period</th>
                  <th className="num">Imports (CIF)</th>
                  <th className="num">Imports vs. prior</th>
                  <th className="num">Exports (FOB)</th>
                  <th className="num">Trade balance</th>
                  <th>Balance</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const bal = row.tradeBalanceUsd;
                  const barPct = bal == null ? 0 : Math.round((Math.abs(bal) / maxAbsBalance) * 100);
                  const deficit = bal != null && bal < 0;
                  return (
                    <tr key={row.partnerCode}>
                      <td data-l="Partner">{row.partnerName}</td>
                      <td data-l="Period">{fmtPeriod(row.period)}</td>
                      <td data-l="Imports (CIF)" className="num">{usd(row.importsCifUsd)}</td>
                      <td data-l="Imports vs. prior" className="num">{pct1(row.importsDeltaPct)}</td>
                      <td data-l="Exports (FOB)" className="num">{usd(row.exportsFobUsd)}</td>
                      <td data-l="Trade balance" className="num" style={{ color: bal == null ? undefined : deficit ? "var(--accent-red, #e5484d)" : "var(--accent-green, #30a46c)" }}>
                        {bal == null ? "—" : bal >= 0 ? `+${usd(bal)}` : usd(bal)}
                      </td>
                      <td data-l="Balance">
                        <div style={{ display: "flex", alignItems: "center", height: 14, background: "rgba(128,128,128,0.15)", borderRadius: 3, overflow: "hidden", minWidth: 60 }}>
                          <div style={{
                            width: `${barPct}%`,
                            height: "100%",
                            background: bal == null ? "transparent" : deficit ? "var(--accent-red, #e5484d)" : "var(--accent-green, #30a46c)",
                          }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="vt-filings-sub vt-streams-foot">
            {data.attribution} · {data.license} · archive latest period {data.archiveLatestPeriod ? fmtPeriod(data.archiveLatestPeriod) : "—"} ·{" "}
            reporter total-commodity aggregate (cmdCode {data.cmdCode ?? "TOTAL"}), not HS-6 detail. Each partner's row
            uses that partner's own latest archived month — periods are not forced to align across partners.
          </div>
        </div>
      )}
    </div>
  );
}
