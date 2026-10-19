/**
 * Live sponsor->ticker join for the ClinicalTrials.gov archive.
 *   tsx scripts/clinical_trials_sponsor_join.ts [out.json]
 * Pulls the current tracked trials, SEC company_tickers.json, and the EDGAR SIC
 * for each name-matched CIK (<= ~8 req/s, SEC fair access), then writes the
 * join table (default datacore/clinical_trials_sponsor_map.json) and prints
 * coverage. The map is a CURRENT-DAY snapshot: gate 2 must key on CIK and
 * re-check ticker validity at event date (ticker reuse hazard).
 */
import fs from "fs";
import { fetchTrials } from "../server/clinicalTrials";
import { joinSponsors, joinStats, type SecTickerRow } from "../server/clinicalTrialsTicker";

const UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function sicOf(cik: number): Promise<number | null> {
  try {
    const r = await fetch(`https://data.sec.gov/submissions/CIK${String(cik).padStart(10, "0")}.json`, { headers: UA, signal: AbortSignal.timeout(20000) });
    await sleep(130);
    if (!r.ok) return null;
    const n = Number(((await r.json()) as { sic?: string | number }).sic);
    return Number.isFinite(n) && n > 0 ? n : null;
  } catch { return null; }
}

(async () => {
  const out = process.argv[2] || "datacore/clinical_trials_sponsor_map.json";
  const trials = await fetchTrials();
  const per = new Map<string, number>();
  for (const t of trials) if (t.sp) per.set(t.sp, (per.get(t.sp) || 0) + 1);
  const r = await fetch("https://www.sec.gov/files/company_tickers.json", { headers: UA });
  if (!r.ok) throw new Error(`company_tickers ${r.status}`);
  const rows = await joinSponsors(per.keys(), (await r.json()) as Record<string, SecTickerRow>, sicOf);
  const stats = joinStats(rows, per);
  const keep = rows.filter((x) => x.status !== "unmatched");
  fs.writeFileSync(out, JSON.stringify({ generated: new Date().toISOString().slice(0, 10), stats, rows: keep }, null, 1) + "\n");
  console.log(JSON.stringify(stats));
  for (const x of rows.filter((y) => y.status === "rejected_sic" || y.status === "sic_unknown")) console.log(x.status, x.sponsor, x.ticker, x.sic);
})();
