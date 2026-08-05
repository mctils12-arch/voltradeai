// AppStoreRankingsView — Apple App Store rank + review-velocity watchlist
// (#/data/appstore-rankings). server/appStoreRankings.ts (GATE 1 DATA
// shipped v1.0.568) has archived this daily since 2026-08-01 with zero
// client view until now — this closes that gap, matching the
// crop-conditions/attention precedent (RAW display, .vt-filings-* shell,
// no new CSS). SOBER PRIOR from the module's own header: this is the most
// arbitraged alt-data category on the street — the point of showing it is
// the archive itself, not a claimed edge. GATE 1 (DATA) only, no signal.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, Smartphone } from "lucide-react";

interface RankRecord {
  t: "rank";
  ticker: string;
  company: string;
  storefront: "us" | "gb" | "ca";
  chart: "top-free" | "top-grossing";
  rank: number | null;
}
interface RatingRecord {
  t: "rating";
  ticker: string;
  company: string;
  avgRating: number | null;
  ratingCount: number | null;
}
type AppStoreRecord = RankRecord | RatingRecord;

interface Payload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  count: number;
  note?: string;
  records: AppStoreRecord[];
}

interface Row {
  ticker: string;
  company: string;
  usFree: number | null;
  usGrossing: number | null;
  avgRating: number | null;
  ratingCount: number | null;
}

const fmtRank = (n: number | null) => (n == null ? "outside top 100" : `#${n}`);
const fmtCount = (n: number | null) => (n == null ? "—" : n.toLocaleString());
const fmtRating = (n: number | null) => (n == null ? "—" : n.toFixed(1));

function buildRows(records: AppStoreRecord[]): Row[] {
  const byTicker = new Map<string, Row>();
  const get = (ticker: string, company: string): Row => {
    let row = byTicker.get(ticker);
    if (!row) {
      row = { ticker, company, usFree: null, usGrossing: null, avgRating: null, ratingCount: null };
      byTicker.set(ticker, row);
    }
    return row;
  };
  for (const r of records) {
    if (r.t === "rank") {
      if (r.storefront !== "us") continue; // US only in this view — GB/CA archived, not shown here
      const row = get(r.ticker, r.company);
      if (r.chart === "top-free") row.usFree = r.rank;
      else row.usGrossing = r.rank;
    } else {
      const row = get(r.ticker, r.company);
      row.avgRating = r.avgRating;
      row.ratingCount = r.ratingCount;
    }
  }
  return [...byTicker.values()].sort((a, b) => {
    const bestA = Math.min(a.usFree ?? Infinity, a.usGrossing ?? Infinity);
    const bestB = Math.min(b.usFree ?? Infinity, b.usGrossing ?? Infinity);
    if (bestA !== bestB) return bestA - bestB;
    return a.ticker.localeCompare(b.ticker);
  });
}

export default function AppStoreRankingsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/appstore-rankings");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = useMemo(() => buildRows(data?.records ?? []), [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="App Store rank + review watchlist">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Smartphone size={16} />
        <div>
          <div className="vt-filings-title">App Store rankings — consumer-app watchlist</div>
          <div className="vt-filings-sub">
            daily US chart rank &amp; rating for 16 consumer-app tickers — RAW, no predictive claim ·{" "}
            {data?.time ? `updated ${new Date(data.time).toLocaleString()}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://rss.marketingtools.apple.com" target="_blank" rel="noreferrer">Apple marketingtools RSS <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && !data?.warming_up && data && rows.length === 0 && (
        <div className="vt-filings-state">No rows for today yet.</div>
      )}

      {!error && rows.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub vt-shortvol-topheader">
            ranked by best US chart position (top-free or top-grossing) ·
            GB/CA storefronts archived too, not shown in this view · Android excluded (ToS-blocked)
          </div>
          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>App</th>
                  <th className="num">Top Free (US)</th>
                  <th className="num">Top Grossing (US)</th>
                  <th className="num">Rating</th>
                  <th className="num">Ratings</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.ticker}>
                    <td data-l="Ticker"><span className="vt-filings-ticker">{row.ticker}</span></td>
                    <td data-l="App">{row.company}</td>
                    <td data-l="Top Free (US)" className="num">{fmtRank(row.usFree)}</td>
                    <td data-l="Top Grossing (US)" className="num">{fmtRank(row.usGrossing)}</td>
                    <td data-l="Rating" className="num">{fmtRating(row.avgRating)}</td>
                    <td data-l="Ratings" className="num">{fmtCount(row.ratingCount)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data?.note && <div className="vt-filings-sub vt-streams-foot">{data.note}</div>}
        </div>
      )}
    </div>
  );
}
