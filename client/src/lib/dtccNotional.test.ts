// Run: npx tsx --test client/src/lib/dtccNotional.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { dtccNotionalText, dtccRankingLine } from "./dtccNotional";

test("a cell: a number with its currency, a capped value with the source's +, a blank that is not a zero", () => {
  assert.equal(dtccNotionalText(220_000_000, "USD"), "USD 220,000,000");
  assert.equal(dtccNotionalText(220_000_000, "USD", false), "USD 220,000,000");
  assert.equal(dtccNotionalText(250_000_000, "USD", true), "USD 250,000,000+");
  assert.equal(dtccNotionalText(null, "USD"), "— (not reported)");
  assert.equal(dtccNotionalText(null, "USD", true), "— (not reported)");
});

test("no counts from the server: no line", () => {
  assert.equal(dtccRankingLine(undefined, 250_000_000), null);
  assert.equal(dtccRankingLine(null), null);
});

test("the file of 2026-11-26 as the vendor published it: every row is accounted for", () => {
  // 13,376 US-underlier rows: 10,936 USD at or under the cap, 17 capped,
  // 2,422 in other currencies ranked nowhere, 3 blank. (The 700,000,000 MXN
  // row is one of the 2,422, not a placeholder: the cap is a dollar figure.)
  const line = dtccRankingLine({ usd: 10_934, usd_capped: 17, other_currency: 2_422, over_cap_unmarked: 0, blank: 3 }, 250_000_000)!;
  assert.equal(line,
    "Ranked in US dollars: 10,951 USD events in this file · 17 at or above the source's cap, written 250,000,000+, listed first"
    + " · not in the list: 2,422 in other currencies (no exchange rate is applied) · 3 with no notional reported");
});

test("a placeholder day: the unmarked value above the cap is named and counted", () => {
  const line = dtccRankingLine({ usd: 5, usd_capped: 0, other_currency: 0, over_cap_unmarked: 1, blank: 0 }, 250_000_000)!;
  assert.match(line, /not in the list: 1 unmarked value above the cap \(a reporter's placeholder\)/);
  assert.doesNotMatch(line, /listed first|other currencies|no notional/);
});

test("nothing left out: the line says only what was ranked", () => {
  assert.equal(dtccRankingLine({ usd: 1, usd_capped: 0, other_currency: 0, over_cap_unmarked: 0, blank: 0 }),
    "Ranked in US dollars: 1 USD event in this file");
});
