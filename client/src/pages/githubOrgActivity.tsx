// GithubOrgActivityView — GitHub org engineering-momentum watchlist
// (#/data/github-activity). server/githubOrgActivity.ts (GATE 1 DATA
// shipped v1.0.??? — see the module header) has archived this weekly
// since 2026-08-02 with zero client view until now — this closes that
// gap, the near-mechanical repeat of the App Store rankings precedent
// (#697, 2026-08-05) queued at the end of that session's own log.
// RAW display only (kind:"raw" per datacore/README.md), no signal
// claim — GATE 2 (velocity deltas vs forward returns) is unstarted per
// the module's own header and datacore/signal_ladder.json.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ExternalLink, GitBranch } from "lucide-react";

interface GithubActivityRecord {
  t: "org_week";
  key: string;
  ticker: string;
  org: string;
  weekStart: string;
  weekEnd: string;
  mergedPRs: number | null;
  commits: number | null;
  uniqueActorsSample: number | null;
  actorSampleCapped: boolean;
  rt: string;
}

interface Payload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  count: number;
  note?: string;
  records: GithubActivityRecord[];
}

const fmtCount = (n: number | null) => (n == null ? "—" : n.toLocaleString());

// One row per ticker: the most recently archived week only — the cache
// holds a rolling ~8-week window per org (see githubOrgActivity.ts's
// refreshGithubActivityCache), but this first view shows the latest
// snapshot rather than a trend, matching the App Store rankings
// precedent's current-snapshot-only scoping (a `/history` follow-up is
// the same queued next step named for that view).
function latestRowsByTicker(records: GithubActivityRecord[]): GithubActivityRecord[] {
  const byTicker = new Map<string, GithubActivityRecord>();
  for (const r of records) {
    const prior = byTicker.get(r.ticker);
    if (!prior || r.weekStart > prior.weekStart) byTicker.set(r.ticker, r);
  }
  return Array.from(byTicker.values()).sort((a, b) => {
    const bm = (b.mergedPRs ?? -1) - (a.mergedPRs ?? -1);
    if (bm !== 0) return bm;
    return a.ticker.localeCompare(b.ticker);
  });
}

export default function GithubOrgActivityView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/github-activity");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const rows = useMemo(() => latestRowsByTicker(data?.records ?? []), [data]);

  return (
    <div className="vt-filings-page" role="region" aria-label="GitHub org engineering-momentum watchlist">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <GitBranch size={16} />
        <div>
          <div className="vt-filings-title">GitHub engineering momentum — develop-in-public watchlist</div>
          <div className="vt-filings-sub">
            weekly merged-PR &amp; commit counts for 15 devtools/SaaS tickers — RAW, no predictive claim ·{" "}
            {data?.time ? `updated ${new Date(data.time).toLocaleString()}` : data?.warming_up ? "warming up…" : "loading…"} ·{" "}
            <a href="https://docs.github.com/en/rest/search" target="_blank" rel="noreferrer">GitHub REST Search API <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a while on a cold start (15 orgs, rate-limited).</div>
      )}
      {!error && !data?.warming_up && data && rows.length === 0 && (
        <div className="vt-filings-state">No completed week archived yet.</div>
      )}

      {!error && rows.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub vt-shortvol-topheader">
            most recently archived completed week per org · public-repo activity is a strategic, biased slice
            — real signal only for companies that genuinely develop in public, noise for the rest
          </div>
          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>GitHub org</th>
                  <th>Week</th>
                  <th className="num">Merged PRs</th>
                  <th className="num">Commits</th>
                  <th className="num">Unique actors</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.ticker}>
                    <td data-l="Ticker"><span className="vt-filings-ticker">{row.ticker}</span></td>
                    <td data-l="GitHub org">{row.org}</td>
                    <td data-l="Week">{row.weekStart}</td>
                    <td data-l="Merged PRs" className="num">{fmtCount(row.mergedPRs)}</td>
                    <td data-l="Commits" className="num">{fmtCount(row.commits)}</td>
                    <td data-l="Unique actors" className="num">
                      {fmtCount(row.uniqueActorsSample)}{row.actorSampleCapped ? " (capped)" : ""}
                    </td>
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
