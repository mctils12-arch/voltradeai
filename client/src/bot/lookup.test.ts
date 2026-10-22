import { test } from "node:test";
import assert from "node:assert/strict";
import { parseLookupInput, gateRows, factorBars, verdictWord, chainLine, lookupErrors, pushHistory, LOOKUP_HISTORY_MAX, type LookupResult } from "./lookup";

const base = (over: Partial<LookupResult> = {}): LookupResult => ({ symbol: "NVDA", at: "2026-10-22T16:00:00Z", errors: {}, ...over });

test("the box accepts a symbol or an OCC code (looked up by its underlying) and rejects the rest", () => {
  assert.deepEqual(parseLookupInput(" nvda "), { symbol: "NVDA", underlying: "NVDA", option: false });
  assert.deepEqual(parseLookupInput("TEM261120P00070000"), { symbol: "TEM261120P00070000", underlying: "TEM", option: true });
  assert.equal(parseLookupInput("BRK.B"), null);
  assert.equal(parseLookupInput(""), null);
  assert.equal(parseLookupInput("12345"), null);
});

test("gate rows keep the scan's order; after the first failure the rest read as skipped; an unavailable gate is unknown", () => {
  const rows = gateRows({ stopped_at: "admission", verdict: "stopped at admission", gates: [
    { gate: "universe", pass: true, why: "in the scan's universe" }, { gate: "regime", pass: true, why: "NEUTRAL" }, { gate: "cooldown", pass: true, why: "no recent stop" },
    { gate: "admission", pass: false, why: "score 58.2 vs MIN_SCORE 65" }, { gate: "sector", pass: false, why: "cap" }, { gate: "sizing", pass: null, why: "sizing unavailable" },
  ] });
  assert.deepEqual(rows.map((r) => r.status), ["pass", "pass", "pass", "fail", "skipped", "skipped"]);
  assert.equal(rows[3].label, "admission");
  const unknown = gateRows({ stopped_at: null, verdict: "undecided (a gate could not run)", gates: [{ gate: "universe", pass: true, why: "" }, { gate: "sizing", pass: null, why: "sizing unavailable" }] });
  assert.deepEqual(unknown.map((r) => r.status), ["pass", "unknown"]);
  assert.deepEqual(gateRows(undefined), []);
});

test("factor bars are signed, largest magnitude first, capped", () => {
  const bars = factorBars({ momentum: -18, mean_reversion: 31, vrp: 4, squeeze: 2, volume: 2.6, sentiment: -9, junk: Number.NaN }, 5);
  assert.deepEqual(bars.map((b) => b.key), ["mean_reversion", "momentum", "sentiment", "vrp", "volume"]);
  assert.equal(bars[1].value, -18);
  assert.deepEqual(factorBars(null), []);
});

test("the verdict word: would place it (green), stopped at admission (orange), stopped elsewhere (red), not found, undecided", () => {
  const ok = base({ decision: { stopped_at: null, verdict: "would place it", gates: [] } });
  assert.deepEqual(verdictWord(ok), { word: "would place it", tone: "g" });
  assert.deepEqual(verdictWord(base({ decision: { stopped_at: "admission", verdict: "stopped at admission", gates: [] } })), { word: "stopped at admission", tone: "o" });
  assert.deepEqual(verdictWord(base({ decision: { stopped_at: "regime", verdict: "stopped at regime", gates: [] } })), { word: "stopped at regime", tone: "r" });
  assert.deepEqual(verdictWord(base({ errors: { snapshot: "no snapshot for this symbol" } })), { word: "not found", tone: "t" });
  assert.deepEqual(verdictWord(base({ decision: { stopped_at: null, verdict: "undecided (a gate could not run)", gates: [] } })), { word: "undecided", tone: "t" });
  assert.deepEqual(verdictWord(undefined), { word: "—", tone: "t" });
});

test("the chain line names the ATM IV, the IV rank and the scanner's delta picks, and says when there is nothing", () => {
  const line = chainLine({ contracts: 412, atm_iv: 61.2, iv_rank: 88, expiries: ["2026-11-21"], cc: { occ: "x", strike: 200, exp_date: "2026-11-21", days_out: 30, bid: 1.9, ask: 2, delta: 0.25, iv: 0.58, oi: 900, premium_pct: 1.0 },
    csp: { occ: "y", strike: 170, exp_date: "2026-11-21", days_out: 30, bid: 2.4, ask: 2.6, delta: -0.22, iv: 0.61, oi: 1200, premium_pct: 1.3 } });
  assert.equal(line, "412 contracts · ATM IV 61% · IV rank 88 · CSP 170 11-21 δ0.22 $2.40 (1.3%) · CC 200 11-21 δ0.25 $1.90 (1.0%)");
  assert.equal(chainLine({ contracts: 0, atm_iv: null, csp: null, cc: null, expiries: [] }), "no contracts in the scanner's window");
  assert.equal(chainLine(undefined), "chain not fetched");
});

test("errors read as step: message; the history keeps the newest per symbol, capped", () => {
  assert.deepEqual(lookupErrors(base({ errors: { chain: "HTTP 429", iv_rank: "timeout" } })), ["chain: HTTP 429", "iv_rank: timeout"]);
  let h: LookupResult[] = [];
  for (let i = 0; i < LOOKUP_HISTORY_MAX + 3; i++) h = pushHistory(h, base({ symbol: `S${i}` }));
  assert.equal(h.length, LOOKUP_HISTORY_MAX);
  assert.equal(h[0].symbol, `S${LOOKUP_HISTORY_MAX + 2}`);
  const again = pushHistory(h, base({ symbol: "S5", at: "later" }));
  assert.equal(again[0].at, "later");
  assert.equal(again.filter((x) => x.symbol === "S5").length, 1);
});
