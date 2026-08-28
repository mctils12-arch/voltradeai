// FinraShortInterestView — FINRA consolidated short interest + Reg SHO
// threshold list (#/data/short-interest). server/finraQuery.ts shipped the
// API-only route (/api/data/short-interest, DATACORE MAXIMUS census build
// #4 part 1, v1.0.207, 2026-07-07) with no client view until now — the last
// FINRA Query API cluster root still missing one (weeklySummary/
// monthlySummary/blocksSummary got atsSummary.tsx the same census build;
// consolidatedShortInterest did not). RAW display of FINRA's own
// semi-monthly settlement POSITIONS (days-to-cover, change vs. prior
// settlement) — distinct from the daily EXECUTION-volume flow proxy at
// /data/short-volume (shortvol.tsx); the two are never conflated here.
// No predictive claim; the settlement-stress composite hypothesis this
// feeds (short interest x FTD x threshold persistence) is a separately
// gated [RESEARCH] item. Reuses the generic .vt-filings-*/.vt-shortvol-*
// CSS (atsSummary.tsx/jodiOilStocks.tsx precedent) — no new styles needed.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, TrendingDown } from "lucide-react";

interface DaysToCoverRow { symbol: string; name: string; days_to_cover: number; short_qty: number; adv: number; }
interface ChangePctRow { symbol: string; name: string; change_pct: number; short_qty: number; }
interface ShortInterestSummary {
  settlement_date: string;
  records: number;
  top_days_to_cover: DaysToCoverRow[];
  top_change_pct: ChangePctRow[];
  adv_floor: number;
  position_floor: number;
  top_cap: number;
}
interface ThresholdRow { symbol: string; name: string; market: string; }
interface ThresholdSummary { trade_date: string; count: number; symbols: ThresholdRow[]; }
interface ShortInterestResponse {
  kind: string; source?: string; attribution?: string; time?: string; note?: string;
  warming_up?: boolean;
  settlement_date?: string | null;
  si_records?: number;
  si: ShortInterestSummary | null;
  threshold: ThresholdSummary | null;
}

type TabKey = "days_to_cover" | "change_pct" | "threshold";

const num = (n: number) => n.toLocaleString();
const dtc1 = (n: number) => n.toFixed(1);
const pct1 = (n: number) => `${n > 0 ? "+" : ""}${n.toFixed(1)}%`;

export default function FinraShortInterestView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<ShortInterestResponse | null>(null);
  const [error, setError] = useState(false);
  const [tab, setTab] = useState<TabKey>("days_to_cover");

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/short-interest");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const tabs = useMemo(() => {
    const t: Array<{ key: TabKey; label: string; enabled: boolean }> = [
      { key: "days_to_cover", label: "Top days-to-cover", enabled: !!data?.si?.top_days_to_cover?.length },
      { key: "change_pct", label: "Top change % (settlement)", enabled: !!data?.si?.top_change_pct?.length },
      { key: "threshold", label: "Reg SHO threshold list", enabled: !!data?.threshold?.symbols?.length },
    ];
    return t;
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Consolidated short interest and threshold list (FINRA)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <TrendingDown size={16} />
        <div>
          <div className="vt-filings-title">Short interest &amp; threshold list — FINRA</div>
          <div className="vt-filings-sub">
            semi-monthly settlement short positions (days-to-cover, change vs. prior settlement) + daily Reg SHO
            threshold securities — RAW, no predictive claim ·{" "}
            {data?.si ? `settlement ${data.si.settlement_date}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://api.finra.org" target="_blank" rel="noreferrer">FINRA Query API <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}

      {!error && !data?.warming_up && (
        <div className="vt-shortvol-body">
          {data?.note && <div className="vt-filings-sub">{data.note}</div>}
          <div className="vt-filings-sub">
            SHORT INTEREST (positions, ~T+9 publish lag) is distinct from short VOLUME (daily execution flow, see the
            separate short-volume view) — a large short-interest reading does not mean heavy selling happened today.
          </div>

          <div className="vt-filings-filters" role="group" aria-label="Leaderboard">
            {tabs.map((t) => (
              <button
                key={t.key}
                type="button"
                className="vt-filings-filter"
                aria-pressed={tab === t.key}
                disabled={!t.enabled}
                style={tab === t.key ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                onClick={() => setTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {!data && <div className="vt-filings-state">Loading…</div>}

          {data && tab === "days_to_cover" && (
            data.si && data.si.top_days_to_cover.length > 0 ? (
              <>
                <div className="vt-filings-sub">
                  settlement {data.si.settlement_date} · {data.si.records.toLocaleString()} symbols reported ·
                  {" "}ADV floor {num(data.si.adv_floor)} shares/day (keeps illiquid junk out of the ranking)
                </div>
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Ticker</th><th>Name</th><th className="num">Days to cover</th><th className="num">Short qty</th><th className="num">ADV</th></tr>
                    </thead>
                    <tbody>
                      {data.si.top_days_to_cover.map((r) => (
                        <tr key={r.symbol}>
                          <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol}</span></td>
                          <td data-l="Name">{r.name || "—"}</td>
                          <td data-l="Days to cover" className="num">{dtc1(r.days_to_cover)}</td>
                          <td data-l="Short qty" className="num">{num(r.short_qty)}</td>
                          <td data-l="ADV" className="num">{num(r.adv)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : <div className="vt-filings-state">No short-interest leaderboard archived yet.</div>
          )}

          {data && tab === "change_pct" && (
            data.si && data.si.top_change_pct.length > 0 ? (
              <>
                <div className="vt-filings-sub">
                  settlement {data.si.settlement_date} · position floor {num(data.si.position_floor)} shares on both
                  the current and prior settlement (keeps a near-zero base from exploding the % figure)
                </div>
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Ticker</th><th>Name</th><th className="num">Change vs. prior</th><th className="num">Short qty</th></tr>
                    </thead>
                    <tbody>
                      {data.si.top_change_pct.map((r) => (
                        <tr key={r.symbol}>
                          <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol}</span></td>
                          <td data-l="Name">{r.name || "—"}</td>
                          <td data-l="Change vs. prior" className="num">{pct1(r.change_pct)}</td>
                          <td data-l="Short qty" className="num">{num(r.short_qty)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : <div className="vt-filings-state">No change-leaderboard archived yet.</div>
          )}

          {data && tab === "threshold" && (
            data.threshold && data.threshold.symbols.length > 0 ? (
              <>
                <div className="vt-filings-sub">
                  trade date {data.threshold.trade_date} · {data.threshold.count.toLocaleString()} names on FINRA's
                  OTC-side Reg SHO threshold list (persistent fails-to-deliver, not a short-interest reading)
                </div>
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr><th>Ticker</th><th>Name</th><th>Market</th></tr>
                    </thead>
                    <tbody>
                      {data.threshold.symbols.map((r) => (
                        <tr key={r.symbol}>
                          <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol}</span></td>
                          <td data-l="Name">{r.name || "—"}</td>
                          <td data-l="Market">{r.market || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : <div className="vt-filings-state">No threshold list archived yet.</div>
          )}
        </div>
      )}
    </div>
  );
}
