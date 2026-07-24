// AtsSummaryView — FINRA ATS/OTC venue volume leaderboards (#/data/ats-summary).
// DATACORE MAXIMUS census build #4 part 2 shipped the API-only route
// (/api/data/ats-summary, server/finraQuery.ts, v1.0.208, 2026-07-08) with
// no client view — this is that follow-up, named directly in wishlist.md's
// DATACORE MAXIMUS resume block since that date. RAW display of FINRA's own
// precomputed leaderboards (weekly ATS-vs-OTC per-symbol volume, monthly OTC
// per-symbol volume, monthly per-venue block-trading ranks) — no predictive
// claim; the settlement-stress composite hypothesis these feed is a
// separately gated [RESEARCH] item. Reuses the generic .vt-filings-*/
// .vt-shortvol-* CSS (methaneHotspots.tsx precedent) — no new styles needed.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Landmark } from "lucide-react";

interface SymbolRow { symbol: string; name: string; shares: number; trades: number; notional: number; }
interface AtsSymbolSummary {
  week_start?: string;
  month_start?: string;
  tiers_covered: string[];
  records: number;
  composition: Record<string, number>;
  top_ats_by_symbol?: SymbolRow[];
  top_otc_by_symbol: SymbolRow[];
}
interface BlockVenueRow {
  mpid: string; name: string; block_quantity: number; block_count: number;
  average_block_size: number; ats_share_percent: number; ats_block_share_rank: number;
}
interface AtsBlocksSummary { month_start: string; records: number; top_venues_by_block_volume: BlockVenueRow[]; }
interface AtsResponse {
  kind: string; source?: string; attribution?: string; time?: string; note?: string;
  warming_up?: boolean;
  weekly: AtsSymbolSummary | null;
  monthly: AtsSymbolSummary | null;
  blocks: AtsBlocksSummary | null;
}

type TabKey = "weekly_ats" | "weekly_otc" | "monthly_otc" | "blocks";

const num = (n: number) => n.toLocaleString();
const pct1 = (n: number) => `${n.toFixed(1)}%`;

function SymbolTable({ rows }: { rows: SymbolRow[] }) {
  if (rows.length === 0) return <div className="vt-filings-state">No symbols in this leaderboard yet.</div>;
  return (
    <div className="vt-filings-tablewrap">
      <table className="vt-filings-table">
        <thead>
          <tr><th>Ticker</th><th>Name</th><th className="num">Shares</th><th className="num">Trades</th><th className="num">Notional</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol}>
              <td data-l="Ticker"><span className="vt-filings-ticker">{r.symbol}</span></td>
              <td data-l="Name">{r.name || "—"}</td>
              <td data-l="Shares" className="num">{num(r.shares)}</td>
              <td data-l="Trades" className="num">{num(r.trades)}</td>
              <td data-l="Notional" className="num">${num(r.notional)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AtsSummaryView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<AtsResponse | null>(null);
  const [error, setError] = useState(false);
  const [tab, setTab] = useState<TabKey>("weekly_ats");

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/ats-summary");
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
      { key: "weekly_ats", label: "Weekly — ATS", enabled: !!data?.weekly?.top_ats_by_symbol?.length },
      { key: "weekly_otc", label: "Weekly — OTC", enabled: !!data?.weekly?.top_otc_by_symbol?.length },
      { key: "monthly_otc", label: "Monthly — OTC", enabled: !!data?.monthly?.top_otc_by_symbol?.length },
      { key: "blocks", label: "Blocks — venues", enabled: !!data?.blocks?.top_venues_by_block_volume?.length },
    ];
    return t;
  }, [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="ATS and OTC venue volume (FINRA)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Landmark size={16} />
        <div>
          <div className="vt-filings-title">ATS / OTC venue volume — FINRA</div>
          <div className="vt-filings-sub">
            per-symbol ATS &amp; OTC leaderboards (weekly, monthly) + per-venue block-trading ranks — RAW, no predictive claim ·{" "}
            {data?.weekly ? `week of ${data.weekly.week_start}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
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
            monthly cut has no ATS-vs-OTC venue split (FINRA publishes tier NMS/OTCE, not venue-typed rows there) —
            only weekly distinguishes ATS from OTC by symbol.
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

          {data && tab === "weekly_ats" && (
            data.weekly ? (
              <>
                <div className="vt-filings-sub">
                  week of {data.weekly.week_start} · tiers covered: {data.weekly.tiers_covered.join(", ") || "none yet"} ·{" "}
                  {data.weekly.records.toLocaleString()} raw records
                </div>
                <SymbolTable rows={data.weekly.top_ats_by_symbol || []} />
              </>
            ) : <div className="vt-filings-state">No weekly summary archived yet.</div>
          )}

          {data && tab === "weekly_otc" && (
            data.weekly ? (
              <>
                <div className="vt-filings-sub">
                  week of {data.weekly.week_start} · tiers covered: {data.weekly.tiers_covered.join(", ") || "none yet"} ·{" "}
                  {data.weekly.records.toLocaleString()} raw records
                </div>
                <SymbolTable rows={data.weekly.top_otc_by_symbol} />
              </>
            ) : <div className="vt-filings-state">No weekly summary archived yet.</div>
          )}

          {data && tab === "monthly_otc" && (
            data.monthly ? (
              <>
                <div className="vt-filings-sub">
                  month of {data.monthly.month_start} · tiers covered: {data.monthly.tiers_covered.join(", ") || "none yet"} ·{" "}
                  {data.monthly.records.toLocaleString()} raw records
                </div>
                <SymbolTable rows={data.monthly.top_otc_by_symbol} />
              </>
            ) : <div className="vt-filings-state">No monthly summary archived yet.</div>
          )}

          {data && tab === "blocks" && (
            data.blocks && data.blocks.top_venues_by_block_volume.length > 0 ? (
              <>
                <div className="vt-filings-sub">
                  month of {data.blocks.month_start} · {data.blocks.records.toLocaleString()} venues reporting ·
                  {" "}FINRA-precomputed ranks (not re-derived here)
                </div>
                <div className="vt-filings-tablewrap">
                  <table className="vt-filings-table">
                    <thead>
                      <tr>
                        <th>Rank</th><th>Venue (MPID)</th><th className="num">Block shares</th>
                        <th className="num">Block count</th><th className="num">Avg block size</th><th className="num">ATS share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.blocks.top_venues_by_block_volume.map((v) => (
                        <tr key={v.mpid}>
                          <td data-l="Rank">#{v.ats_block_share_rank || "—"}</td>
                          <td data-l="Venue (MPID)"><span className="vt-filings-ticker">{v.mpid}</span>{v.name ? ` — ${v.name}` : ""}</td>
                          <td data-l="Block shares" className="num">{num(v.block_quantity)}</td>
                          <td data-l="Block count" className="num">{num(v.block_count)}</td>
                          <td data-l="Avg block size" className="num">{num(Math.round(v.average_block_size))}</td>
                          <td data-l="ATS share" className="num">{pct1(v.ats_share_percent)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : <div className="vt-filings-state">No block-trading summary archived yet.</div>
          )}
        </div>
      )}
    </div>
  );
}
