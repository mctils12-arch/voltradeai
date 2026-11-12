// [RULE-REVIEW] measurement, 2026-11-12 — server/cspPriorCohort.ts: the
// pre-registered re-measure of the csp prior (research/open_questions.md,
// "2026-11-06 — findings from the refocus build night", first item). The
// Wilson interval against textbook values, the cohort's selection, the ONE
// definition it shares with the all-lots expectancy, the verdict on both
// sides of the rule, and the parity of the mirrored prior with
// position_sizing.py. Run: npx tsx --test server/cspPriorCohort.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  CSP_COHORT_MIN_LOTS, CSP_COHORT_OPENED_FROM, CSP_COHORT_OPENED_FROM_MS, CSP_PRIOR_WIN_RATE, WILSON_Z_95,
  cspPriorCohort, wilsonInterval,
} from "./cspPriorCohort";
import { edgeLedger } from "./edgeLedger";
import { sanitizeDiag } from "./diag";
import type { RealizedEvent } from "./sleeveLedger";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");
const T = (iso: string) => Date.parse(iso);
const r4 = (x: number) => Math.round(x * 10000) / 10000;

/** One closed lot. closeDir +1 = the closing fill bought back a short. */
const lot = (symbol: string, opened: string | null, closed: string, pnl: number, basis: number, closeDir: 1 | -1 = 1): RealizedEvent =>
  ({ symbol, t: T(closed), pnl, openedAt: opened === null ? null : T(opened), qty: 1, closeDir, basis });

/** `wins` winning and `losses` losing short puts, all opened after the cutoff. */
function newLots(wins: number, losses: number): RealizedEvent[] {
  const out: RealizedEvent[] = [];
  for (let i = 0; i < wins + losses; i++) {
    // one contract per lot: AMD 2026-12-18 puts, strikes $100, $101, … ; +50 % of the credit kept or −120 %
    const symbol = `AMD261218P${String((100 + i) * 1000).padStart(8, "0")}`;
    out.push(lot(symbol, "2026-11-09T15:00:00Z", `2026-11-${10 + (i % 3)}T16:00:00Z`, i < wins ? 50 : -120, 100));
  }
  return out;
}

test("Wilson 95 % score interval: the textbook values to 4 d.p.", () => {
  const at = (wins: number, n: number) => { const ci = wilsonInterval(wins, n); assert.ok(ci); return [r4(ci[0]), r4(ci[1])]; };
  assert.deepEqual(at(0, 10), [0, 0.2775]);
  assert.deepEqual(at(5, 10), [0.2366, 0.7634]);
  assert.deepEqual(at(10, 10), [0.7225, 1]);
  assert.deepEqual(at(30, 100), [0.2189, 0.3958]);
  assert.equal(WILSON_Z_95, 1.959964);
  // exactly 0 and exactly 1 at the ends, not −1e-17 or 0.9999999999999999
  for (const n of [1, 7, 10, 29, 30, 316]) {
    assert.equal(wilsonInterval(0, n)![0], 0, `0/${n}: the lower bound is exactly 0`);
    assert.equal(wilsonInterval(n, n)![1], 1, `${n}/${n}: the upper bound is exactly 1`);
  }
});

test("Wilson: each bound solves the score equation (p̂ − p)² = z² p (1 − p) / n, and the interval holds p̂ inside [0, 1]", () => {
  const z2 = WILSON_Z_95 * WILSON_Z_95;
  for (const n of [1, 2, 5, 12, 30, 31, 100, 316, 5000]) {
    for (const wins of new Set([0, 1, Math.floor(n / 3), Math.floor(n / 2), n - 1, n].filter((w) => w >= 0 && w <= n))) {
      const ci = wilsonInterval(wins, n);
      assert.ok(ci, `${wins}/${n}`);
      const p = wins / n;
      for (const b of ci) {
        assert.ok(Math.abs((p - b) ** 2 - (z2 * b * (1 - b)) / n) < 1e-12, `${wins}/${n}: ${b} is a root`);
        assert.ok(b >= 0 && b <= 1);
      }
      assert.ok(ci[0] <= p && p <= ci[1] && ci[0] < ci[1], `${wins}/${n}: lo ≤ p̂ ≤ hi`);
    }
  }
});

test("Wilson: no data or impossible counts give no interval (null), never NaN", () => {
  for (const [wins, n] of [[0, 0], [1, 0], [-1, 10], [11, 10], [NaN, 10], [5, NaN], [5, Infinity], [5, -3]] as const) {
    assert.equal(wilsonInterval(wins, n), null, `${wins}/${n}`);
  }
});

test("the pre-registered numbers: lots opened from 2026-11-06 (UTC), read from 30 closed lots on", () => {
  // research/open_questions.md, 2026-11-06 [HONESTY METRIC]: "only lots OPENED on or after 2026-11-06. At n ≥ 30 closed new lots…"
  assert.equal(CSP_COHORT_OPENED_FROM, "2026-11-06");
  assert.equal(CSP_COHORT_OPENED_FROM_MS, Date.UTC(2026, 10, 6));
  assert.equal(CSP_COHORT_MIN_LOTS, 30);
});

test("the prior mirrors position_sizing.py: the source line and the value the sizer falls back to", () => {
  // the source: the csp_options row of _get_historical_stats' default
  const src = fs.readFileSync(path.join(repo, "position_sizing.py"), "utf8");
  const fn = src.slice(src.indexOf("def _get_historical_stats()"), src.indexOf("def _normalize_vol_to_daily_pct("));
  const rows = [...fn.matchAll(/"csp_options":\s*\{"win_rate":\s*([0-9.]+),/g)];
  assert.equal(rows.length, 1, "one csp_options prior row in _get_historical_stats");
  assert.equal(Number(rows[0][1]), CSP_PRIOR_WIN_RATE, "CSP_PRIOR_WIN_RATE must change with position_sizing.py's csp prior");
  // the value: what the sizers and the tax filter get when no trade feedback overrides the bucket
  const program = [
    "import position_sizing as ps",
    "ps.TRADE_FEEDBACK_PATH = '/nonexistent/voltrade_trade_feedback.json'",
    "print(ps._get_historical_stats()['by_strategy']['csp_options']['win_rate'])",
  ].join("\n");
  const out = execFileSync("python3", ["-c", program], { cwd: repo, stdio: ["ignore", "pipe", "pipe"] }).toString().trim();
  assert.equal(Number(out), CSP_PRIOR_WIN_RATE);
});

// Friday 2026-11-06 is the first day of the cohort.
const MIXED: RealizedEvent[] = [
  lot("AMD261218P00120000", "2026-11-05T20:55:00Z", "2026-11-10T15:00:00Z", 100, 200),        // short put opened the day before: out
  lot("XOM261218P00100000", "2026-11-05T23:59:59.999Z", "2026-11-10T15:00:00Z", 100, 200),    // 1 ms before the cutoff: out
  lot("KO261218P00060000", "2026-11-06T00:00:00.000Z", "2026-11-09T15:00:00Z", 30, 100),      // at the cutoff: in, +30 %
  lot("AMD261218P00110000", "2026-11-06T14:31:00Z", "2026-11-09T15:30:00Z", 100, 200),        // in, +50 % of the credit kept
  lot("XOM261218C00130000", "2026-11-09T15:00:00Z", "2026-11-10T16:00:00Z", -150, 100),       // short call bought back dearer: in, −150 %
  lot("GE261218P00150000", "2026-11-09T15:00:00Z", "2026-11-11T16:00:00Z", 0, 80),            // flat: in, a loss of 0
  lot("GE261218C00200000", "2026-11-09T15:00:00Z", "2026-11-10T16:00:00Z", 500, 300, -1),     // a LONG call sold: not a short lot
  lot("AMD", "2026-11-09T15:00:00Z", "2026-11-10T16:00:00Z", 50, 1000),                       // a covered short STOCK lot: not an option
  lot("T261218P00021000", null, "2026-11-10T16:00:00Z", 20, 50),                              // no known open time: undated, not guessed in
  lot("F261218P00010000", "2026-11-09T15:00:00Z", "2026-11-10T16:00:00Z", 5, 0),              // no basis: cannot be priced
];

test("the cohort: closed SHORT OPTION lots OPENED on or after 2026-11-06, nothing else", () => {
  const c = cspPriorCohort(MIXED, T("2026-11-12T17:00:00Z"));
  assert.equal(c.since, "2026-11-06T00:00:00.000Z");
  assert.equal(c.n, 4);
  assert.equal(c.wins, 2);
  assert.equal(c.winRatePct, 50);
  assert.equal(c.meanReturnPct, -17.5, "(30 + 50 − 150 + 0) / 4 percent of the premium");
  assert.deepEqual(c.wilson95Pct, [15, 85]);
  assert.equal(c.undated, 1);
  assert.equal(c.lastCloseAt, "2026-11-11T16:00:00.000Z");
  assert.equal(c.fetchedAt, "2026-11-12T17:00:00.000Z");
  assert.equal(c.priorWinRatePct, 71.6);
  assert.equal(c.needed, 30);
  assert.equal(c.enough, false);
  assert.equal(c.priorExcluded, false);
  assert.equal(c.verdict, "4 of 30 needed lots closed; too few to judge");
  // the order of the lots does not matter
  assert.deepEqual(cspPriorCohort([...MIXED].reverse(), T("2026-11-12T17:00:00Z")), c);
});

test("one definition: the cohort's row is the all-lots expectancy's options row over the selected lots", () => {
  const FLOOR = new Set(["QQQ", "SPY", "SSO", "BOXX", "AMD"]); // a floor set never moves an option lot
  const selected = MIXED.filter((e) => e.closeDir === 1 && e.openedAt !== null && e.openedAt >= T("2026-11-06T00:00:00Z"));
  const row = edgeLedger(selected, new Map(), FLOOR).options;
  const c = cspPriorCohort(MIXED);
  assert.deepEqual([c.n, c.wins, c.winRatePct, c.meanReturnPct, c.lastCloseAt], [row.n, row.wins, row.winRate, row.expectancyPct, row.lastCloseAt]);
  // /api/bot/tax's liveExpectancy.options counts every option lot, long ones and older ones too
  const all = edgeLedger(MIXED, new Map(), FLOOR).options;
  assert.equal(all.n, 8);
  assert.equal(all.n, c.n + 2 /* opened before the cutoff */ + 1 /* long */ + 1 /* undated */);
});

test("too few lots: the rule is not read, whatever the interval says", () => {
  const twelve = cspPriorCohort(newLots(5, 7));
  assert.equal(twelve.n, 12);
  assert.equal(twelve.enough, false);
  assert.equal(twelve.priorExcluded, false);
  assert.deepEqual(twelve.wilson95Pct, [19.33, 68.05], "the interval is shown, and it excludes 71.6 %");
  assert.equal(twelve.verdict, "12 of 30 needed lots closed; too few to judge");
  const twentyNine = cspPriorCohort(newLots(0, 29));
  assert.deepEqual(twentyNine.wilson95Pct, [0, 11.7]);
  assert.equal(twentyNine.enough, false);
  assert.equal(twentyNine.priorExcluded, false, "29 lots, none won: still not read");
  assert.equal(twentyNine.verdict, "29 of 30 needed lots closed; too few to judge");
});

test("30 lots: the prior outside the interval triggers the rebuild verdict, on either side", () => {
  const below = cspPriorCohort(newLots(15, 15));
  assert.equal(below.n, 30);
  assert.equal(below.enough, true);
  assert.equal(below.priorExcluded, true);
  assert.deepEqual(below.wilson95Pct, [33.15, 66.85]);
  assert.equal(below.winRatePct, 50);
  assert.equal(below.meanReturnPct, -35, "(15 × 50 − 15 × 120) / 30 percent of the premium");
  assert.equal(below.verdict, "prior 71.6% is outside the live 95% interval [33.15%, 66.85%] (above it; 15 of 30 lots won): rebuild the csp prior (RULE-REVIEW)");
  const above = cspPriorCohort(newLots(27, 3));
  assert.equal(above.priorExcluded, true);
  assert.deepEqual(above.wilson95Pct, [74.38, 96.54]);
  assert.equal(above.verdict, "prior 71.6% is outside the live 95% interval [74.38%, 96.54%] (below it; 27 of 30 lots won): rebuild the csp prior (RULE-REVIEW)");
  // the edges of the rule at n = 30: 16 wins still excludes, 17 does not; 26 does not, 27 does
  assert.deepEqual([16, 17, 26, 27].map((w) => cspPriorCohort(newLots(w, 30 - w)).priorExcluded), [true, false, false, true]);
});

test("30 lots: the prior inside the interval does not trigger it", () => {
  const c = cspPriorCohort(newLots(21, 9));
  assert.equal(c.enough, true);
  assert.equal(c.priorExcluded, false);
  assert.deepEqual(c.wilson95Pct, [52.12, 83.34]);
  assert.equal(c.verdict, "prior 71.6% is inside the live 95% interval [52.12%, 83.34%] (21 of 30 lots won): the pre-registered rebuild is not triggered");
  // more lots only narrow it: 316 lots at the old 20.6 % are far outside
  const old = cspPriorCohort(newLots(65, 251));
  assert.equal(old.priorExcluded, true);
  assert.deepEqual(old.wilson95Pct, [16.48, 25.37]);
});

test("an empty cohort: nulls, no interval, no NaN, and a verdict that says so", () => {
  for (const events of [[], MIXED.filter((e) => e.openedAt !== null && e.openedAt < CSP_COHORT_OPENED_FROM_MS)]) {
    const c = cspPriorCohort(events);
    assert.deepEqual([c.n, c.wins, c.winRatePct, c.wilson95Pct, c.meanReturnPct, c.lastCloseAt, c.fetchedAt], [0, 0, null, null, null, null, null]);
    assert.equal(c.enough, false);
    assert.equal(c.priorExcluded, false);
    assert.equal(c.verdict, "0 of 30 needed lots closed; too few to judge");
    assert.ok(!JSON.stringify(c).includes("NaN"));
  }
});

test("the tax diag probe serves the block as computed: the sanitizer changes nothing in it", () => {
  for (const c of [cspPriorCohort([]), cspPriorCohort(MIXED, T("2026-11-12T17:00:00Z")), cspPriorCohort(newLots(15, 15)), cspPriorCohort(newLots(21, 9))]) {
    assert.deepEqual(sanitizeDiag(c), c);
  }
});
