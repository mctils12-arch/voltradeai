// filingHistory.test.ts — the /data filing pages say which archive days they
// show (2026-11-12 [PRODUCT]; live: the Form 4 page printed "1,800 filings,
// last 30 days" over 7 archive days, the 13F page "532 filings, last 45 days"
// over one). The label comes from the route's own `covered`/`truncated`, the
// older page from its own `next`.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { coverageLabel, inclusiveDays, mergeFilingPages, olderPageUrl } from "./filingHistory";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = (rel: string) => fs.readFileSync(path.join(here, "..", rel), "utf8");

type F = { accession: string; filedAt?: string | null };
const f = (a: string): F => ({ accession: a });

// the Form 4 route's live answer on 2026-11-12 (metadata verbatim, filings elided)
const LIVE_FORM4 = {
  kind: "raw", days: 30,
  window: { from: "2026-10-14", to: "2026-11-12" },
  covered: { from: "2026-11-06", to: "2026-11-12" },
  truncated: true, next: "?from=2026-10-14&to=2026-11-05",
  max_filings: 2000, count: 1800,
  filings: [f("a"), f("b"), f("c")],
};

test("inclusiveDays counts calendar days, both ends in", () => {
  assert.equal(inclusiveDays("2026-11-06", "2026-11-12"), 7);
  assert.equal(inclusiveDays("2026-10-14", "2026-11-12"), 30);
  assert.equal(inclusiveDays("2026-11-12", "2026-11-12"), 1);
  assert.equal(inclusiveDays("2026-11-12", "2026-11-06"), null, "a reversed span is malformed, not negative");
  assert.equal(inclusiveDays("", "2026-11-12"), null);
  assert.equal(inclusiveDays("not-a-day", "2026-11-12"), null);
});

test("a truncated first page is labelled by the days it covers, not the days it asked for", () => {
  const { filings, coverage } = mergeFilingPages([LIVE_FORM4], 30);
  assert.equal(filings.length, 3);
  assert.equal(coverage.legacy, false);
  assert.deepEqual(coverage.covered, { from: "2026-11-06", to: "2026-11-12" });
  assert.equal(coverage.daysCovered, 7);
  assert.equal(coverage.daysRequested, 30);
  assert.equal(coverage.truncated, true);
  assert.equal(coverage.next, "?from=2026-10-14&to=2026-11-05");
  const label = coverageLabel(coverage, "filings");
  assert.equal(label, "3 filings · 2026-11-06 → 2026-11-12 · 7 of the 30 days requested — older days not loaded");
  assert.doesNotMatch(label, /last 30 days/, "the old echoed-days phrasing is gone when the server states coverage");
});

test("the 13F deadline-week case: one archive day of 45 reads as that one day", () => {
  const { coverage } = mergeFilingPages([{
    days: 45, window: { from: "2026-09-29", to: "2026-11-12" }, covered: { from: "2026-11-12", to: "2026-11-12" },
    truncated: true, next: "?from=2026-09-29&to=2026-11-11", count: 532, filings: [f("x")],
  }], 45);
  assert.equal(coverage.daysCovered, 1);
  assert.equal(coverage.daysRequested, 45);
  assert.equal(coverageLabel(coverage, "filings"), "1 filings · 2026-11-12 · 1 of the 45 days requested — older days not loaded");
});

test("loading the older page extends the covered span backwards and clears truncation when the server does", () => {
  const older = {
    window: { from: "2026-10-14", to: "2026-11-05" }, covered: { from: "2026-10-14", to: "2026-11-05" },
    truncated: false, next: null, count: 2, filings: [f("c"), f("d")], // "c" repeats across a day boundary
  };
  const { filings, coverage } = mergeFilingPages([LIVE_FORM4, older], 30);
  assert.deepEqual(filings.map((x) => x.accession), ["a", "b", "c", "d"], "a cross-day repeat is counted once");
  assert.equal(coverage.count, 4);
  assert.deepEqual(coverage.covered, { from: "2026-10-14", to: "2026-11-12" });
  assert.equal(coverage.daysCovered, 30);
  assert.equal(coverage.truncated, false);
  assert.equal(coverage.next, null);
  assert.equal(coverageLabel(coverage, "filings"), "4 filings · 2026-10-14 → 2026-11-12 · all 30 days requested");
});

test("a second truncated page keeps the walk going from ITS next, not the first page's", () => {
  const mid = {
    window: { from: "2026-10-14", to: "2026-11-05" }, covered: { from: "2026-11-01", to: "2026-11-05" },
    truncated: true, next: "?from=2026-10-14&to=2026-10-31", count: 1, filings: [f("d")],
  };
  const { coverage } = mergeFilingPages([LIVE_FORM4, mid], 30);
  assert.equal(coverage.truncated, true);
  assert.equal(coverage.next, "?from=2026-10-14&to=2026-10-31");
  assert.deepEqual(coverage.covered, { from: "2026-11-01", to: "2026-11-12" });
  assert.equal(coverage.daysCovered, 12);
});

test("an older server (no `covered`) keeps the legacy wording and offers nothing to load", () => {
  const { coverage } = mergeFilingPages([{ days: 30, count: 2, filings: [f("a"), f("b")] }], 30);
  assert.equal(coverage.legacy, true);
  assert.equal(coverage.truncated, false);
  assert.equal(coverage.next, null);
  assert.equal(coverageLabel(coverage, "releases"), "2 releases, last 30 days");
  const { coverage: none } = mergeFilingPages([{ filings: [] }], 45);
  assert.equal(coverageLabel(none, "filings"), "0 filings, last 45 days", "no days echoed → the page's own default");
  assert.equal(coverageLabel(null, "filings"), "loading filings…");
});

test("the count is the deduplicated list, never the server's per-page count summed", () => {
  const a = { ...LIVE_FORM4, count: 1800, filings: [f("a"), f("a"), f("b")] };
  const { filings, coverage } = mergeFilingPages([a], 30);
  assert.equal(filings.length, 2);
  assert.equal(coverage.count, 2);
});

test("olderPageUrl keeps the route path and takes the server's whole query string", () => {
  assert.equal(olderPageUrl("/api/data/insider/history?days=30", "?from=2026-10-14&to=2026-11-05"), "/api/data/insider/history?from=2026-10-14&to=2026-11-05");
  assert.equal(olderPageUrl("/api/data/earnings-language/history?days=30", "?from=2026-10-14&to=2026-11-05&text=0"),
    "/api/data/earnings-language/history?from=2026-10-14&to=2026-11-05&text=0", "text=0 travels with next, nothing re-derived");
  assert.equal(olderPageUrl("/api/data/filings13f/history", "from=a&to=b"), "/api/data/filings13f/history?from=a&to=b");
  assert.equal(olderPageUrl("/api/data/insider/history?days=30", null), null);
  assert.equal(olderPageUrl("/api/data/insider/history?days=30", ""), null);
});

test("all three EDGAR pages read their archive through the shared hook and print its label", () => {
  for (const page of ["pages/filings.tsx", "pages/edgar13f.tsx", "pages/earnings.tsx"]) {
    const s = src(page);
    assert.match(s, /useFilingHistory</, `${page} uses the hook`);
    assert.match(s, /\{hist\.label\}/, `${page} prints the coverage label`);
    assert.doesNotMatch(s, /last \{[^}]*days\} days/, `${page} no longer prints the echoed days as the range`);
    assert.match(s, /Load older days/, `${page} offers the older page`);
  }
});
