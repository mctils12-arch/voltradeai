import { test } from "node:test";
import assert from "node:assert/strict";
import { annotateWithTicker, loadSponsorIndex, sponsorIndex, _resetSponsorIndexForTests, HEALTH_SICS, joinSponsors, joinStats, tickerToCik, type SecTickerRow } from "./clinicalTrialsTicker";

const TICKERS: Record<string, SecTickerRow> = {
  "0": { cik_str: 1595097, ticker: "CRBP", title: "Corbus Pharmaceuticals Holdings, Inc." },
  "1": { cik_str: 931015, ticker: "PII", title: "Polaris Inc." },
  "2": { cik_str: 21665, ticker: "CL", title: "COLGATE PALMOLIVE CO" },
  "3": { cik_str: 555, ticker: "UNRD", title: "Unreadable Therapeutics Inc" },
};
const SIC: Record<number, number | null> = { 1595097: 2834, 931015: 3790, 21665: 2844, 555: null };
const sicOf = async (cik: number) => SIC[cik] ?? null;

test("tickerToCik maps every ticker and ignores malformed rows", () => {
  const m = tickerToCik({ ...TICKERS, bad: { cik_str: NaN, ticker: "X", title: "x" } as SecTickerRow });
  assert.equal(m.get("CRBP"), 1595097);
  assert.equal(m.has("X"), false);
});

test("exact name match + health SIC -> matched", async () => {
  const [r] = await joinSponsors(["Corbus Pharmaceuticals Inc."], TICKERS, sicOf);
  assert.deepEqual(r, { sponsor: "Corbus Pharmaceuticals Inc.", status: "matched", ticker: "CRBP", cik: 1595097, sic: 2834 });
});

test("name collision with a non-health filer is rejected by the SIC lock (Polaris Group vs Polaris Inc.)", async () => {
  const [r] = await joinSponsors(["Polaris Group"], TICKERS, sicOf);
  assert.equal(r.status, "rejected_sic");
  assert.equal(r.ticker, "PII");
  assert.equal(r.sic, 3790);
});

test("unreadable SIC fails closed, not matched", async () => {
  const [r] = await joinSponsors(["Unreadable Therapeutics"], TICKERS, sicOf);
  assert.equal(r.status, "sic_unknown");
});

test("no fuzzy matching: near-miss name is unmatched and sicOf is never called", async () => {
  let calls = 0;
  const [r] = await joinSponsors(["Corbus Pharma"], TICKERS, async () => { calls++; return 2834; });
  assert.equal(r.status, "unmatched");
  assert.equal(calls, 0);
});

test("duplicate and empty sponsor names are collapsed", async () => {
  const rows = await joinSponsors(["Corbus Pharmaceuticals Inc.", "Corbus Pharmaceuticals Inc.", ""], TICKERS, sicOf);
  assert.equal(rows.length, 1);
});

test("joinStats counts trials by matched sponsors only", async () => {
  const rows = await joinSponsors(["Corbus Pharmaceuticals Inc.", "Polaris Group", "Nobody"], TICKERS, sicOf);
  const s = joinStats(rows, new Map([["Corbus Pharmaceuticals Inc.", 3], ["Polaris Group", 5], ["Nobody", 2]]));
  assert.equal(s.matched, 1); assert.equal(s.rejected_sic, 1); assert.equal(s.unmatched, 1);
  assert.equal(s.trials_total, 10); assert.equal(s.trials_matched, 3);
  assert.equal(s.trial_coverage, 0.3);
});

test("health SIC set is the pre-registered five", () => {
  assert.deepEqual([...HEALTH_SICS].sort(), [2833, 2834, 2835, 2836, 8731]);
});

test("sponsorIndex keeps only matched rows; annotate never guesses", () => {
  const idx = sponsorIndex({ rows: [
    { sponsor: "Corbus Pharmaceuticals Inc.", status: "matched", ticker: "CRBP", cik: 1595097, sic: 2834 },
    { sponsor: "Polaris Group", status: "rejected_sic", ticker: "PII", cik: 931015, sic: 3790 },
    { sponsor: "Unreadable", status: "sic_unknown", ticker: "UNRD", cik: 555, sic: null },
  ] });
  assert.equal(idx.size, 1);
  const out = annotateWithTicker([{ sponsor: "Corbus Pharmaceuticals Inc.", n: 1 }, { sponsor: "Polaris Group", n: 2 }, { sponsor: "Nobody", n: 3 }], idx);
  assert.deepEqual(out.map((e) => e.ticker), ["CRBP", null, null]);
  assert.equal(out[0].cik, 1595097);
  assert.equal(out[1].cik, null);
  assert.equal(out[0].n, 1, "original fields preserved");
  assert.equal(annotateWithTicker([{ sponsor: null }], idx)[0].ticker, null, "null sponsor is tolerated");
});

test("loadSponsorIndex reads the committed map and degrades to empty on a bad path", () => {
  _resetSponsorIndexForTests();
  const real = loadSponsorIndex();
  assert.ok(real.size > 100, "committed snapshot has >100 matched sponsors");
  assert.equal(real.get("Sanofi")?.ticker, "SNY");
  _resetSponsorIndexForTests();
  assert.equal(loadSponsorIndex("/nonexistent/map.json").size, 0);
  _resetSponsorIndexForTests();
});
