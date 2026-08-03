/**
 * occ_volume_gate1.ts — GATE 1 (DATA layer) for the occ_options_volume
 * root: does our parser's reading of OCC's per-trade-side volume-query
 * feed reconcile against OCC's own PUBLISHED total?
 *
 * ORIGINAL CRITERION (occVolume.ts's header comment, filed 2026-07-06):
 * "one day's totals vs OCC's published daily volume page." That page —
 * theocc.com/market-data/market-data-reports/volume-and-open-interest/
 * daily-volume — sits behind a Cloudflare JS challenge (verified this
 * session: both WebFetch and a browser-UA curl get "Just a moment..."
 * 403s; only the marketdata.theocc.com API subdomain this pipeline
 * already depends on is unprotected). SUBSTITUTED, honestly: OCC's own
 * monthly press releases (theocc.com/newsroom, indexed and readable via
 * search-engine snippets even though the source page itself is blocked)
 * headline an exact total-contracts figure per calendar month, broken
 * into Equity/ETF/Index. That is still "OCC's own published" volume —
 * a stronger ground truth than a single day, since it forces every
 * trading day's parse in the month to be correct simultaneously, not
 * just one.
 *
 * CRITERIA (pre-stated BEFORE running, Reasoning Standard #10):
 *   - Month: June 2026 (last fully-elapsed month with a published OCC
 *     report at gate time; June 19 2026 Juneteenth is the one 2026
 *     market holiday inside it per market_calendar.py).
 *   - TRUTH: OCC's "June 2026 Monthly Volume Data" release — Equity
 *     778,333,823 + ETF 689,472,511 + Index 135,685,225 = 1,603,491,559
 *     total cleared contracts (retrieved via web search 2026-08-03;
 *     the source press release page itself is Cloudflare-blocked from
 *     automated fetch, so this constant is a hand-recorded citation,
 *     not a live pull — matches how other roots in this repo cite a
 *     non-machine-readable truth source, e.g. finra_gate1.ts's facility
 *     files vs. this: here even the truth number itself had to be
 *     read by a human/session, not polled).
 *   - METHOD: for every trading day in the month, fetch the SAME feed
 *     occVolume.ts's production poller uses, parse with the SAME
 *     `parseOcc` shipped in server/occVolume.ts (zero reimplementation —
 *     a mismatch here is a real parser bug, not a script drift), sum
 *     `qty` across every row, divide by 2 (each cleared contract is
 *     recorded once per clearing side — occVolume.ts's own aggregateOcc
 *     does the identical halving per-underlying; this script does it
 *     at the whole-month level).
 *   - PASS = exact match (0 contracts, not just "close") against the
 *     press-release total. OCC's own number is itself exact to the
 *     contract, so any real parser defect should show up as a nonzero
 *     diff, not a rounding-scale one.
 *   - PRIOR: P(exact match) ~85% — the shipped parser already has unit
 *     tests (occVolume.test.ts) pinning the halving logic and CRLF/
 *     trailing-comma handling; residual risk was an unnoticed body
 *     variant (partial-day republish, a symbol type OCC excludes from
 *     its own headline total) that the existing tests wouldn't catch
 *     because they run on synthetic fixtures, not a full live month.
 *
 * Session-run: `npx tsx scripts/occ_volume_gate1.ts [YYYYMM] [officialTotal]`
 * Result goes in research/experiments.md + datacore/signal_ladder.json,
 * never into any runtime path — this script touches no production state.
 */
import { fetchOccDay, type OccBodyKind } from "../server/occVolume";

const MONTH = process.argv[2] || "202606";
const OFFICIAL_TOTAL = Number(process.argv[3] || 1_603_491_559);

function tradingDaysYmd(yyyymm: string): string[] {
  const year = Number(yyyymm.slice(0, 4));
  const month = Number(yyyymm.slice(4, 6)); // 1-indexed
  const out: string[] = [];
  const d = new Date(Date.UTC(year, month - 1, 1));
  while (d.getUTCMonth() === month - 1) {
    const dow = d.getUTCDay();
    if (dow !== 0 && dow !== 6) {
      out.push(d.toISOString().slice(0, 10).replace(/-/g, ""));
    }
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return out;
}

async function main() {
  const days = tradingDaysYmd(MONTH);
  console.log(`occ_volume_gate1: ${MONTH} — ${days.length} weekday candidates`);

  let grandTotal = 0;
  let daysWithData = 0;
  const byKind = new Map<OccBodyKind, number>();
  const perDay: { day: string; total: number }[] = [];

  for (const ymd of days) {
    const { kind, rows } = await fetchOccDay(ymd);
    byKind.set(kind, (byKind.get(kind) || 0) + 1);
    if (kind === "data" && rows.length) {
      const qtySum = rows.reduce((s, r) => s + r.qty, 0);
      const dayTotal = qtySum / 2;
      grandTotal += dayTotal;
      daysWithData++;
      perDay.push({ day: rows[0].date, total: dayTotal });
      console.log(`  ${rows[0].date}: ${rows.length} rows, total_cleared=${dayTotal.toLocaleString()}`);
    } else {
      console.log(`  ${ymd}: ${kind} (excluded from sum — honest no-op, matches occVolume.ts's own refreshOcc handling)`);
    }
    // politeness spacing, matches occVolume.ts's own deep-backfill pacing
    await new Promise((res) => setTimeout(res, 500));
  }

  const diff = grandTotal - OFFICIAL_TOTAL;
  console.log(`\ndays_with_data=${daysWithData}/${days.length}`);
  console.log(`body kinds:`, Object.fromEntries(byKind));
  console.log(`GRAND TOTAL (our parse, ${MONTH}): ${grandTotal.toLocaleString()}`);
  console.log(`OFFICIAL TOTAL (OCC press release):   ${OFFICIAL_TOTAL.toLocaleString()}`);
  console.log(`DIFF: ${diff.toLocaleString()} (${((diff / OFFICIAL_TOTAL) * 100).toFixed(4)}%)`);
  console.log(`\nVERDICT (pre-stated bar: exact match): ${diff === 0 ? "PASS" : "FAIL"}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
