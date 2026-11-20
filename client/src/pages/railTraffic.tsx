// RailTrafficView — Class I railroad weekly carloads, STB EP 724
// (#/data/rail-traffic). server/railTraffic.ts has served this on the keyed
// API and listed it on /developers since 2026-09; no /data panel read it.
// UI coverage batch C (2026-11-20) places it beside "US port imports".
// RAW, self-reported counts. Gate 1 (data) passed 2026-09-27; gate 2
// (signal) was REJECTED 2026-09-28, and the page says so in the server's own
// words: nothing here is a trading signal.
//
// The week-over-week column is signed and uncoloured (portImports.tsx
// precedent): more carloads is not "good", fewer is not "bad".
import { useEffect, useState } from "react";
import { ArrowLeft, ExternalLink, TrainFront } from "lucide-react";
import { signedPct, weeksBehind } from "@/lib/dataPanelsView";

interface Commodity {
  variable: string;
  latestWeekCarloads: number | null;
  priorWeekCarloads: number | null;
  weekOverWeekDeltaPct: number | null;
  isIntermodal?: boolean;
}
interface Railroad { railroad: string; latestWeekCarloads: number | null; }
interface Payload {
  kind?: string;
  latestWeek?: string | null;
  priorWeek?: string | null;
  weeksArchived?: number;
  railroadsReporting?: number;
  archiveBuiltAt?: string | null;
  systemLatestWeekCarloads?: number | null;
  systemIntermodalLatestWeekCarloads?: number | null;
  systemNonIntermodalLatestWeekCarloads?: number | null;
  commodities?: Commodity[];
  railroads?: Railroad[];
  note?: string;
  attribution?: string;
}

const count = (n: number | null | undefined) => (typeof n === "number" ? n.toLocaleString("en-US") : "—");

export default function RailTrafficView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/rail-traffic");
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const commodities = data?.commodities ?? [];
  const railroads = data?.railroads ?? [];
  const behind = weeksBehind(data?.latestWeek, Date.now());
  const built = data?.archiveBuiltAt ? data.archiveBuiltAt.slice(0, 10) : null;

  return (
    <div className="vt-filings-page" role="region" aria-label="Class I railroad weekly carloads">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <TrainFront size={16} />
        <div>
          <div className="vt-filings-title">Rail carloads — Class I railroads, weekly</div>
          <div className="vt-filings-sub">
            carloads originated, as the railroads report them to the regulator — RAW, no predictive claim ·{" "}
            <span data-vt-rail-week>{data?.latestWeek ? `week of ${data.latestWeek}` : error ? "not loaded" : "loading…"}</span> ·{" "}
            <a href="https://www.stb.gov/reports-data/rail-service-data/" target="_blank" rel="noreferrer">Surface Transportation Board EP 724 <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">The archive did not load. A refresh asks again.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data && !data.latestWeek && <div className="vt-filings-state">The rail archive holds no week yet.</div>}

      {!error && data?.latestWeek && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub" data-vt-rail-age>
            Newest week in our archive: {data.latestWeek}
            {behind != null ? ` (${behind === 0 ? "this week" : `${behind} week${behind === 1 ? "" : "s"} ago`})` : ""}
            {built ? ` · archive rebuilt ${built}` : ""}
            {typeof data.weeksArchived === "number" ? ` · ${data.weeksArchived.toLocaleString("en-US")} weeks held` : ""}
            {typeof data.railroadsReporting === "number" ? ` · ${data.railroadsReporting} railroads reported this week` : ""}
            . The archive is rebuilt from the regulator's weekly file by a session, not polled live.
          </div>

          <div className="vt-rail-tiles">
            <div className="vt-rail-tile"><div className="vt-rail-tile-l">All carloads, latest week</div><div className="vt-rail-tile-v" data-vt-rail-total>{count(data.systemLatestWeekCarloads)}</div></div>
            <div className="vt-rail-tile"><div className="vt-rail-tile-l">Intermodal (containers, trailers)</div><div className="vt-rail-tile-v">{count(data.systemIntermodalLatestWeekCarloads)}</div></div>
            <div className="vt-rail-tile"><div className="vt-rail-tile-l">All other freight</div><div className="vt-rail-tile-v">{count(data.systemNonIntermodalLatestWeekCarloads)}</div></div>
          </div>

          <div className="vt-filings-tablewrap vt-tablewrap-flow">
            <table className="vt-filings-table" data-vt-rail-commodities>
              <thead>
                <tr>
                  <th>Commodity group</th>
                  <th className="num">Week of {data.latestWeek}</th>
                  <th className="num">{data.priorWeek ? `Week of ${data.priorWeek}` : "Prior week"}</th>
                  <th className="num">Change</th>
                </tr>
              </thead>
              <tbody>
                {commodities.map((c) => (
                  <tr key={c.variable}>
                    <td data-l="Commodity group">{c.variable}{c.isIntermodal ? " (intermodal)" : ""}</td>
                    <td data-l="Latest week" className="num">{count(c.latestWeekCarloads)}</td>
                    <td data-l="Prior week" className="num">{count(c.priorWeekCarloads)}</td>
                    <td data-l="Change" className="num">{signedPct(c.weekOverWeekDeltaPct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {railroads.length > 0 && (
            <div className="vt-filings-tablewrap vt-tablewrap-flow">
              <table className="vt-filings-table" data-vt-rail-railroads>
                <thead><tr><th>Railroad</th><th className="num">Carloads, week of {data.latestWeek}</th></tr></thead>
                <tbody>
                  {railroads.map((r) => (
                    <tr key={r.railroad}>
                      <td data-l="Railroad">{r.railroad}</td>
                      <td data-l="Carloads" className="num">{count(r.latestWeekCarloads)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {data.note && <div className="vt-filings-sub vt-streams-foot">{data.note}</div>}
        </div>
      )}
    </div>
  );
}
