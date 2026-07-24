/**
 * gate1_usaspending_ticker_check.ts — ROOT VALIDATION LADDER gate 1 for
 * usaSpending.ts's ticker matcher (recipient name -> public-company ticker).
 *
 * server/usaSpending.ts's header has stated since 2026-07-05 (v1.0.96):
 * "gate 1 (recipient->ticker precision on a 50-award hand-check) is now
 * RUNNABLE from the archive alone ... not yet attempted." This script runs
 * it, against LIVE USAspending.gov data, through the actual production
 * matcher (normalizeCompanyName / buildTickerMap / resolveTickers /
 * applyParentMatch imported unchanged from server/usaSpending.ts — this
 * is NOT a reimplementation of the matching logic, only the pagination
 * loop is script-local so a wider date window than fetchRecentContracts'
 * hardcoded 2 days can be sampled).
 *
 * OUTPUT: prints a JSON summary (matched rows by method, a sample for
 * hand verification) to stdout. The actual gate-1 verdict — comparing
 * each matched (recipient name, ticker) pair against known real company
 * identities — is a human/agent judgment step done AFTER reading this
 * output, then recorded in research/open_questions.md and
 * research/experiments.md. This script produces evidence, not the verdict.
 *
 * Usage: npx tsx scripts/gate1_usaspending_ticker_check.ts [days] [limit]
 */
import {
  parseTxnRow, buildTickerMap, resolveTickers, applyParentMatch,
  fetchAwardParent, CONTRACT_FLOOR_USD, PARENT_LOOKUP_MIN_USD,
  type ContractTxn, type UeiCache,
} from "../server/usaSpending";

const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)", "Content-Type": "application/json" };
const TXN_URL = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/";
const SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json";
const TXN_FIELDS = [
  "Award ID", "Mod", "Recipient Name", "Recipient UEI", "recipient_id",
  "Action Date", "Transaction Amount", "Transaction Description",
  "Award Type", "Awarding Agency", "Awarding Sub Agency", "NAICS", "PSC",
  "generated_internal_id",
];

async function postJson(url: string, body: any): Promise<any> {
  const r = await fetch(url, { method: "POST", headers: UA, body: JSON.stringify(body), signal: AbortSignal.timeout(20000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}: ${await r.text()}`);
  return r.json();
}

/** Wider-window sample, sorted by |amount| descending (positives only, for
 *  a clean top-N-by-dollar-value gate-1 sample) — deliberately NOT the
 *  production fetchRecentContracts (that's locked to a 2-day rolling
 *  window); this script needs more candidate rows to hit a meaningful
 *  count of real public-company recipients. */
async function fetchTopTransactions(days: number, limit: number): Promise<ContractTxn[]> {
  const now = Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const start = new Date(now - days * 86400_000).toISOString().slice(0, 10);
  const out: ContractTxn[] = [];
  for (let page = 1; out.length < limit; page++) {
    const d = await postJson(TXN_URL, {
      filters: {
        time_period: [{ start_date: start, end_date: rt, date_type: "last_modified_date" }],
        award_type_codes: ["A", "B", "C", "D"],
      },
      fields: TXN_FIELDS, limit: 100, page, sort: "Transaction Amount", order: "desc",
    });
    const rows: any[] = d?.results || [];
    if (!rows.length) break;
    for (const row of rows) {
      const amt = Number(row["Transaction Amount"] ?? 0);
      if (amt < CONTRACT_FLOOR_USD) { return out; } // monotone under desc sort
      out.push(parseTxnRow(row, rt));
    }
    if (!d?.page_metadata?.hasNext) break;
    await new Promise((res) => setTimeout(res, 150));
  }
  return out.slice(0, limit);
}

async function main() {
  const days = Number(process.argv[2] || 45);
  const limit = Number(process.argv[3] || 300);

  const [txns, secJson] = await Promise.all([
    fetchTopTransactions(days, limit),
    fetch(SEC_TICKERS_URL, { headers: { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" } }).then((r) => r.json()),
  ]);
  const tickerMap = buildTickerMap(secJson);
  const ueiCache: UeiCache = {}; // fresh — this is a gate-1 probe of the matcher, not the live cache
  const { resolved, needParent } = resolveTickers(txns, tickerMap, ueiCache);

  // Exercise the parent-lookup path too (same production function), capped
  // politely — this is a one-shot probe, not the 6h live poller.
  const parentSample = needParent.slice(0, 25);
  for (const t of parentSample) {
    try {
      const { pn, puei } = await fetchAwardParent(t.aid);
      applyParentMatch(t, pn, puei, tickerMap, ueiCache);
    } catch { /* honest skip, same as production */ }
    await new Promise((res) => setTimeout(res, 150));
  }

  const matchedByMethod: Record<string, number> = {};
  const matchedSample: any[] = [];
  const unmatchedLargeSample: any[] = [];
  for (const t of resolved) {
    if (t.tkr) {
      matchedByMethod[t.mm || "?"] = (matchedByMethod[t.mm || "?"] || 0) + 1;
      matchedSample.push({ recipient: t.r, ticker: t.tkr, method: t.mm, matched_name: t.mname, parent: t.pn, amt: t.amt, uei: t.uei });
    } else if (Math.abs(t.amt) >= PARENT_LOOKUP_MIN_USD) {
      unmatchedLargeSample.push({ recipient: t.r, amt: t.amt, uei: t.uei, parent_checked: parentSample.includes(t) });
    }
  }

  console.log(JSON.stringify({
    window_days: days, sample_size: txns.length,
    matched_total: matchedSample.length, matched_by_method: matchedByMethod,
    unmatched_large_total: unmatchedLargeSample.length,
    matched_sample: matchedSample, // full list — hand-check draws from here
    unmatched_large_sample: unmatchedLargeSample.slice(0, 40),
  }, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
