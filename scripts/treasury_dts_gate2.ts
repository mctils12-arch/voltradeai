/**
 * treasury_dts_gate2.ts — GATE 2 (SIGNAL) for the treasury_daily_statement
 * root: does withheld-tax deposit growth carry statistical information
 * about payroll growth, with NO trading involved (CLAUDE.md's ROOT
 * VALIDATION LADDER, gate 2 definition)?
 *
 * LADDER PATH AS FILED (server/treasuryDts.ts's own file header, BUILD
 * ORDER 6 #2, 2026-07-06): "withheld-tax YoY growth vs payroll-surprise
 * dates." HONEST SUBSTITUTION, stated up front per the BUILD-FIRST /
 * MEASUREMENT INTEGRITY norms rather than silently redefined: "surprise"
 * means actual-vs-consensus-forecast, and no free consensus/forecast
 * source was found (economist survey medians are a paid product — Bloomberg/
 * Econoday). What IS freely available, keyless, and a legitimate ground
 * truth for a NOWCAST hypothesis is BLS's own ACTUAL Nonfarm Payrolls
 * print (FRED series PAYEMS, public domain, U.S. government data) — the
 * number a nowcast would be trying to anticipate in the first place.
 * Ground truth here is PAYEMS YoY growth, not surprise-vs-consensus.
 * Gate 3+ (LOGIC — does an actual trading edge exist around release dates)
 * would be where a consensus/surprise source becomes necessary; gate 2
 * only asks whether the raw statistical relationship exists at all.
 *
 * TWO CORRELATIONS ARE PRE-REGISTERED BEFORE ANY DATA IS FETCHED
 * (REASONING STANDARD #10 — state the prior, then update; REASONING
 * STANDARD #4 — distrust results in proportion to variants tried, so
 * both variants are named now, not chosen after seeing numbers):
 *
 *   TEST A (GATING, matches the filed ladder path literally): same-month
 *   correlation between withheld-tax YoY growth and PAYEMS YoY growth.
 *   PRIOR: medium-high existence (both track the same labor-market
 *   cycle — REASONING STANDARD #3's base-rate concern: high FICA-withheld
 *   correlation may just reflect shared macro trend, not novel
 *   information; that is *expected* and does not by itself invalidate
 *   gate 2, which only asks whether the archive tracks something real).
 *   PASS BAR: Pearson r >= 0.30 AND two-tailed p < 0.05 (a materially
 *   positive, statistically significant relationship — deliberately a
 *   much lower bar than gate 1's 0.85, because this is a genuine
 *   cross-series ECONOMIC signal test, not a reconciliation of two
 *   measures of the same underlying activity).
 *
 *   TEST B (INFORMATIONAL, NOT GATING): withheld-tax YoY growth in month
 *   M-1 vs PAYEMS YoY growth in month M — the actual "nowcast ahead of
 *   the release" shape the hypothesis claims. No pass bar is set for this
 *   one; it is reported alongside Test A so a future LOGIC-gate session
 *   knows whether the relationship is genuinely LEADING or only
 *   contemporaneous (a real difference for whether a lead-time trading
 *   edge could exist at all), without this script quietly picking
 *   whichever of the two looks better after the fact.
 *
 * WINDOW: the category name WITHHELD_TAX_CATEGORY ("Taxes - Withheld
 * Individual/FICA") did not exist before a DTS taxonomy change — bisected
 * live this session: present 2023-02-28, absent 2023-01-31 and every
 * earlier date checked back to 2018. Before that date the closest DTS
 * line was the unsplit "Cash FTD's Received (Table IV)" (Federal Tax
 * Deposits, mixing withheld and non-withheld amounts) — a different,
 * not-directly-comparable series, so this run does NOT attempt to splice
 * the two. This was discovered by running the script, not assumed from
 * documentation — the original plan (WINDOW_START 2017-01, an 8+ year,
 * multi-regime sample) was overwritten by this finding, not the other
 * way around: WINDOW_START is 2022-02 (one year of runway before the
 * earliest possible YoY month) purely to skip API calls this taxonomy
 * change guarantees will return null, and the real usable sample is only
 * 2024-02 through the latest month (~2.5 years, n=31) — a materially
 * smaller and less regime-diverse window than intended. That shortfall
 * is itself part of this gate's result, not a footnote: REASONING
 * STANDARD #2 (regime-condition everything) is compromised here by data
 * availability, not by choice — a future re-run after this window has
 * aged further will have more power without needing new code.
 *
 * Usage: npx tsx scripts/treasury_dts_gate2.ts
 * Prints a JSON verdict to stdout. Result goes in research/experiments.md
 * + datacore/signal_ladder.json — this script touches no runtime state.
 */
import { pathToFileURL } from "node:url";
import { parseDts, sumWithheldTaxDeposits, type DtsRow } from "../server/treasuryDts";

const DTS_URL =
  "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/deposits_withdrawals_operating_cash";
const PAYEMS_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PAYEMS";
const PASS_R = 0.30;
const PASS_P = 0.05;
const WINDOW_START = "2022-02"; // one year of runway before WITHHELD_TAX_CATEGORY's earliest existence (2023-02, bisected live)
const COVID_EXCLUDE_START = "2020-03";
const COVID_EXCLUDE_END = "2021-06";

export function monthRange(startYm: string, endYm: string): string[] {
  const [sy, sm] = startYm.split("-").map(Number);
  const [ey, em] = endYm.split("-").map(Number);
  const out: string[] = [];
  let y = sy, m = sm;
  while (y < ey || (y === ey && m <= em)) {
    out.push(`${y}-${String(m).padStart(2, "0")}`);
    m++; if (m > 12) { m = 1; y++; }
  }
  return out;
}

/** FiscalData intermittently 500s on an otherwise-valid query (confirmed
 *  live this session: an immediate manual retry of the exact same URL
 *  returned 200) — a plain transient hiccup, not a real gap, so retry
 *  a bounded number of times before giving up. */
async function fetchJson(url: string, attempts = 3): Promise<any> {
  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(30000) as any });
      if (!r.ok) throw new Error(`${url} -> ${r.status}`);
      return JSON.parse(await r.text());
    } catch (e) {
      lastErr = e;
      if (i < attempts - 1) await new Promise((res) => setTimeout(res, 1500 * (i + 1)));
    }
  }
  throw lastErr;
}

/** The newest DTS record_date on or before a calendar month's last day,
 *  returned only if it actually falls IN that month (else null). */
async function lastBusinessDayOfMonth(yyyyMm: string): Promise<string | null> {
  const [y, m] = yyyyMm.split("-").map(Number);
  const lastDay = new Date(Date.UTC(y, m, 0)).toISOString().slice(0, 10);
  const url = `${DTS_URL}?filter=record_date:lte:${lastDay}&sort=-record_date&page%5Bsize%5D=1`;
  const j = await fetchJson(url);
  const d = j?.data?.[0]?.record_date;
  return typeof d === "string" && d.startsWith(yyyyMm) ? d : null;
}

async function dtsDepositsForDay(day: string): Promise<DtsRow[]> {
  const url = `${DTS_URL}?filter=record_date:eq:${day}&page%5Bsize%5D=250`;
  return parseDts(await fetchJson(url), day);
}

async function withheldTaxByMonth(months: string[]): Promise<Map<string, number>> {
  const out = new Map<string, number>();
  for (const month of months) {
    const day = await lastBusinessDayOfMonth(month);
    if (!day) continue;
    const rows = await dtsDepositsForDay(day);
    if (!rows.length) continue;
    out.set(month, sumWithheldTaxDeposits(rows));
  }
  return out;
}

/** Keyless FRED CSV export — same "web UI's own download" precedent
 *  fredMacro.ts's own gate 1 established; no FRED_API_KEY needed. */
async function payemsByMonth(): Promise<Map<string, number>> {
  let text = "";
  let lastErr: unknown;
  for (let i = 0; i < 3; i++) {
    try {
      const r = await fetch(PAYEMS_CSV_URL, { signal: AbortSignal.timeout(30000) as any });
      if (!r.ok) throw new Error(`PAYEMS csv -> ${r.status}`);
      text = await r.text();
      lastErr = null;
      break;
    } catch (e) {
      lastErr = e;
      if (i < 2) await new Promise((res) => setTimeout(res, 1500 * (i + 1)));
    }
  }
  if (lastErr) throw lastErr;
  const out = new Map<string, number>();
  for (const line of text.split("\n").slice(1)) {
    const [date, val] = line.split(",");
    if (!date || !val) continue;
    const n = parseFloat(val);
    if (!Number.isFinite(n)) continue;
    out.set(date.slice(0, 7), n);
  }
  return out;
}

export function yoyGrowth(byMonth: Map<string, number>, month: string): number | null {
  const [y, m] = month.split("-").map(Number);
  const priorMonth = `${y - 1}-${String(m).padStart(2, "0")}`;
  const cur = byMonth.get(month);
  const prior = byMonth.get(priorMonth);
  if (cur == null || prior == null || prior === 0) return null;
  return (cur - prior) / prior;
}

export function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  const cov = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / n;
  const sx = Math.sqrt(xs.reduce((a, x) => a + (x - mx) ** 2, 0) / n);
  const sy = Math.sqrt(ys.reduce((a, y) => a + (y - my) ** 2, 0) / n);
  return cov / (sx * sy);
}

/** Two-tailed p-value for a Pearson r via the t-distribution, computed
 *  from the regularized incomplete beta function (no stats library dep). */
export function pValueForR(r: number, n: number): number {
  const df = n - 2;
  if (df <= 0) return 1;
  const t = Math.abs(r) * Math.sqrt(df / (1 - r * r));
  const x = df / (df + t * t);
  return ibeta(x, df / 2, 0.5);
}

/** Numerical-Recipes-style continued fraction for the incomplete beta
 *  function. Only valid/fast-converging for x < (a+1)/(a+b+2) — ibeta
 *  below is responsible for calling it with swapped (a,b,1-x) otherwise,
 *  never by flipping this function's own return value at the same
 *  (a,b,x) triple (that was this script's original bug: A/B-verified
 *  against Python's scipy.stats.t.cdf this session — e.g. r=0.163,n=31
 *  read p=0.846 before the fix and p=0.382 after, matching scipy exactly). */
function betacf(x: number, a: number, b: number): number {
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 200; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return h;
}

function ibeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);
  if (x < (a + 1) / (a + b + 2)) return (front * betacf(x, a, b)) / a;
  return 1 - (front * betacf(1 - x, b, a)) / b;
}

function lgamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

async function main() {
  const today = new Date();
  const endYm = `${today.getUTCFullYear()}-${String(today.getUTCMonth() + 1).padStart(2, "0")}`;
  const months = monthRange(WINDOW_START, endYm);

  const withheld = await withheldTaxByMonth(months);
  const payems = await payemsByMonth();

  const contemporaneous: { month: string; withheldYoy: number; payemsYoy: number }[] = [];
  const leadOneMonth: { month: string; withheldYoyPriorMonth: number; payemsYoy: number }[] = [];

  for (const month of months) {
    const payemsYoy = yoyGrowth(payems, month);
    if (payemsYoy == null) continue;
    const withheldYoy = yoyGrowth(withheld, month);
    if (withheldYoy != null) contemporaneous.push({ month, withheldYoy, payemsYoy });

    const [y, m] = month.split("-").map(Number);
    const prevM = m === 1 ? 12 : m - 1;
    const prevY = m === 1 ? y - 1 : y;
    const priorMonth = `${prevY}-${String(prevM).padStart(2, "0")}`;
    const withheldYoyPrior = yoyGrowth(withheld, priorMonth);
    if (withheldYoyPrior != null) leadOneMonth.push({ month, withheldYoyPriorMonth: withheldYoyPrior, payemsYoy });
  }

  if (contemporaneous.length < 12) {
    console.log(JSON.stringify({
      verdict: "ERROR",
      reason: `only ${contemporaneous.length} reconcilable months, need >= 12`,
      contemporaneous,
    }, null, 2));
    process.exitCode = 1;
    return;
  }

  const isCovid = (month: string) => month >= COVID_EXCLUDE_START && month <= COVID_EXCLUDE_END;
  const exCovid = contemporaneous.filter((p) => !isCovid(p.month));

  const rAll = pearson(contemporaneous.map((p) => p.withheldYoy), contemporaneous.map((p) => p.payemsYoy));
  const pAll = pValueForR(rAll, contemporaneous.length);
  const rExCovid = exCovid.length >= 12
    ? pearson(exCovid.map((p) => p.withheldYoy), exCovid.map((p) => p.payemsYoy))
    : null;
  const pExCovid = rExCovid != null ? pValueForR(rExCovid, exCovid.length) : null;

  const rLead = leadOneMonth.length >= 12
    ? pearson(leadOneMonth.map((p) => p.withheldYoyPriorMonth), leadOneMonth.map((p) => p.payemsYoy))
    : null;
  const pLead = rLead != null ? pValueForR(rLead, leadOneMonth.length) : null;

  // TEST A (gating) verdict is the ALL-SAMPLE contemporaneous correlation,
  // per the pre-registered bar above — the ex-COVID and lead numbers are
  // reported for context, never substituted in if the gating number fails.
  const verdict = rAll >= PASS_R && pAll < PASS_P ? "PASS" : "FAIL";

  console.log(JSON.stringify({
    verdict,
    testA_gating: {
      description: "same-month withheld-tax YoY growth vs PAYEMS YoY growth",
      passBar: { r: PASS_R, p: PASS_P },
      n: contemporaneous.length,
      pearson_r: rAll,
      p_value: pAll,
    },
    context_exCovid: rExCovid != null ? {
      description: `same as Test A, excluding ${COVID_EXCLUDE_START}..${COVID_EXCLUDE_END}`,
      n: exCovid.length, pearson_r: rExCovid, p_value: pExCovid,
    } : null,
    testB_informational_leadOneMonth: rLead != null ? {
      description: "withheld-tax YoY growth in month M-1 vs PAYEMS YoY growth in month M (NOT gating)",
      n: leadOneMonth.length, pearson_r: rLead, p_value: pLead,
    } : null,
    contemporaneous,
  }, null, 2));
  process.exitCode = verdict === "PASS" ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
