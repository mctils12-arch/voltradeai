// What a filing-history page holds, in words ([PRODUCT] 2026-11-15).
// Run: npx tsx --test client/src/lib/filingCoverage.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";
import { readCoverage, extendCoverage, appendOlderFilings, coverageWords, loadOlderLabel, spanDays } from "./filingCoverage";

// production, read 2026-11-15 ~18:20Z (rows dropped): the two pages this module exists for
const FORM4_LIVE = { kind: "raw", days: 30, window: { from: "2026-10-17", to: "2026-11-15" }, covered: { from: "2026-11-10", to: "2026-11-15" },
  truncated: true, next: "?from=2026-10-17&to=2026-11-09", max_filings: 2000, days_scanned: 6, archive_start: "2026-07-04", count: 1864 };
const F13_LIVE = { kind: "raw", days: 45, window: { from: "2026-10-02", to: "2026-11-15" }, covered: { from: "2026-11-14", to: "2026-11-15" },
  truncated: true, next: "?from=2026-10-02&to=2026-11-13", max_filings: 500, days_scanned: 2, archive_start: "2026-07-05", count: 160 };

test("spanDays counts inclusive UTC days and refuses anything that is not two ordered days", () => {
  assert.equal(spanDays("2026-10-17", "2026-11-15"), 30);
  assert.equal(spanDays("2026-11-15", "2026-11-15"), 1);
  assert.equal(spanDays("2026-11-15", "2026-11-14"), 0);
  assert.equal(spanDays("nope", "2026-11-14"), 0);
});

test("REGRESSION: the live Form 4 page is six of thirty days, and the words say so (it was labelled 'last 30 days')", () => {
  const c = readCoverage(FORM4_LIVE)!;
  assert.equal(c.windowDays, 30);
  assert.equal(c.coveredDays, 6);
  assert.equal(c.truncated, true);
  assert.equal(coverageWords(c), "2026-11-10 → 2026-11-15 · 6 of the 30 days asked for — older days not loaded");
  assert.equal(loadOlderLabel(c, false), "Load older days (24 days before 2026-11-10 not loaded)");
  assert.equal(loadOlderLabel(c, true), "Loading older days…");
});

test("REGRESSION: the live 13F page is two of forty-five days", () => {
  const c = readCoverage(F13_LIVE)!;
  assert.equal(coverageWords(c), "2026-11-14 → 2026-11-15 · 2 of the 45 days asked for — older days not loaded");
});

test("a complete page says 'all N days' and offers nothing older", () => {
  const c = readCoverage({ window: { from: "2026-10-17", to: "2026-11-15" }, covered: { from: "2026-10-17", to: "2026-11-15" }, truncated: false, next: null })!;
  assert.equal(coverageWords(c), "2026-10-17 → 2026-11-15 · all 30 days");
  assert.equal(loadOlderLabel(c, false), null);
  const one = readCoverage({ window: { from: "2026-11-15", to: "2026-11-15" }, covered: { from: "2026-11-15", to: "2026-11-15" }, truncated: false })!;
  assert.equal(coverageWords(one), "2026-11-15 · 1 day");
});

test("a payload that states no coverage yields none: the page claims no window instead of guessing", () => {
  assert.equal(readCoverage(null), null);
  assert.equal(readCoverage({ days: 30, count: 2 }), null); // a server older than the contract
  assert.equal(readCoverage({ window: { from: "2026-11-15", to: "2026-11-01" }, covered: { from: "2026-11-15", to: "2026-11-01" } }), null);
  assert.equal(readCoverage({ window: { from: "x", to: "y" }, covered: { from: "x", to: "y" } }), null);
  assert.equal(coverageWords(null), "");
  assert.equal(loadOlderLabel(null, false), null);
});

test("truncated with no way to continue still reads as truncated, and renders no button", () => {
  const c = readCoverage({ ...FORM4_LIVE, next: null })!;
  assert.equal(c.truncated, true);
  assert.match(coverageWords(c), /6 of the 30 days asked for/);
  assert.equal(loadOlderLabel(c, false), null);
  // a `next` that is not a query string is not followed
  assert.equal(readCoverage({ ...FORM4_LIVE, next: "https://example.com/x" })!.next, null);
});

test("extendCoverage: an older page that continues the run moves covered.from back and keeps walking", () => {
  const c = readCoverage(FORM4_LIVE)!;
  const older = { window: { from: "2026-10-17", to: "2026-11-09" }, covered: { from: "2026-11-03", to: "2026-11-09" }, truncated: true, next: "?from=2026-10-17&to=2026-11-02" };
  const e = extendCoverage(c, older);
  assert.deepEqual(e.covered, { from: "2026-11-03", to: "2026-11-15" });
  assert.equal(e.coveredDays, 13);
  assert.equal(e.windowDays, 30);
  assert.equal(e.truncated, true);
  assert.equal(e.next, "?from=2026-10-17&to=2026-11-02");
  // the last page reaches the window's first day: complete, nothing older
  const last = extendCoverage(e, { window: { from: "2026-10-17", to: "2026-11-02" }, covered: { from: "2026-10-17", to: "2026-11-02" }, truncated: false, next: null });
  assert.equal(last.truncated, false);
  assert.equal(last.next, null);
  assert.equal(coverageWords(last), "2026-10-17 → 2026-11-15 · all 30 days");
});

test("extendCoverage: a page that does not continue the run is never presented as loaded — the walk stops", () => {
  const c = readCoverage(FORM4_LIVE)!;
  // a gap (ends two days before the loaded run), a page with no coverage, a page reaching before the window
  for (const older of [
    { window: { from: "2026-10-17", to: "2026-11-08" }, covered: { from: "2026-11-01", to: "2026-11-08" }, truncated: true, next: "?from=2026-10-17&to=2026-10-31" },
    { days: 30, count: 0 },
    null,
    { window: { from: "2026-09-01", to: "2026-11-09" }, covered: { from: "2026-09-01", to: "2026-11-09" }, truncated: false, next: null },
  ]) {
    const e = extendCoverage(c, older as any);
    assert.deepEqual(e.covered, c.covered);
    assert.equal(e.truncated, true, "still says older days are not loaded");
    assert.equal(e.next, null);
    assert.equal(loadOlderLabel(e, false), null);
  }
});

test("appendOlderFilings keeps one row per accession, loaded copy first", () => {
  const have = [{ accession: "a", v: 1 }, { accession: "b", v: 1 }];
  const out = appendOlderFilings(have, [{ accession: "b", v: 2 }, { accession: "c", v: 2 }, { accession: "c", v: 3 }, null as any]);
  assert.deepEqual(out, [{ accession: "a", v: 1 }, { accession: "b", v: 1 }, { accession: "c", v: 2 }]);
  assert.deepEqual(appendOlderFilings(have, null), have);
  assert.notEqual(appendOlderFilings(have, []), have, "returns a new array");
});

// RATCHET: a /data page that fetches a `*-history` filing route must read its coverage through this module —
// a page that renders `days`/`count` alone is the defect this file was written for.
test("RATCHET: every client page fetching a filing-history route reads coverage through filingCoverage", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const pages = path.join(here, "..", "pages");
  const ROUTES = /\/api\/data\/(insider|filings13f)\/history/;
  const users = fs.readdirSync(pages).filter((f) => f.endsWith(".tsx") && ROUTES.test(fs.readFileSync(path.join(pages, f), "utf8")));
  assert.deepEqual(users.sort(), ["edgar13f.tsx", "filings.tsx"], "the set of pages reading these routes changed — update this ratchet with it");
  for (const f of users) {
    const src = fs.readFileSync(path.join(pages, f), "utf8");
    assert.match(src, /from "@\/lib\/filingCoverage"|from "\.\.\/lib\/filingCoverage"/, `${f} must import filingCoverage`);
    assert.match(src, /coverageWords\(/, `${f} must render coverageWords()`);
    assert.doesNotMatch(src, /last \{[^}]*days[^}]*\} days/, `${f} still prints "last N days" from the echoed days field`);
  }
});
