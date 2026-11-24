// REGRESSION (2026-11-24, KNOWN BROKEN #61): the Form 4 poll and the nightly
// pass found a filing's document only through its directory listing
// (`index.json`). Measured 2026-11-24 02:35Z (Mon 21:35 ET): the listing
// answered 404 "Directory listing is Not Available Yet" for every sampled
// Form 4 accepted since about 15:45 ET, the feed kept listing them, each poll
// failed 25 documents of 25 and was counted ok, and the archive took no row
// from 16:43 ET on. The filing's submission text (`<accession>.txt`) answered
// 200 for the same filings.
//
// The fixtures are public SEC records fetched that hour:
//   0001493152-26-053616  listing published: its submission text and the
//                         ownership.xml the listing names (the two copies)
//   0001683382-26-000008  accepted 21:11 ET, listing 404: its submission text
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  describeDocumentError,
  fetchForm4Document,
  fetchLatestForm4Filings,
  form4PollStats,
  form4PollUnreadError,
  ownershipXmlFromSubmission,
  parseForm4Xml,
  refreshForm4Cache,
  _resetForm4CacheForTests,
} from "./edgarForm4";
import { runForm4Completeness } from "./edgarForm4Completeness";
import { edgarPollHealthStatus, _resetEdgarPollHealthForTests } from "./edgarPollHealth";
import { sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = (name: string) => fs.readFileSync(path.join(here, "__fixtures__", "form4_submission", name), "utf8");
const SUB_PUBLISHED = fixture("0001493152-26-053616.txt");
const XML_PUBLISHED = fixture("0001493152-26-053616.ownership.xml");
const SUB_UNPUBLISHED = fixture("0001683382-26-000008.txt");
const NOT_YET = "<html><h1>Directory listing is Not Available Yet</h1></html>";

const resp = (status: number, body = "") => ({ ok: status >= 200 && status < 300, status, text: async () => body });
const throwing = (get: (url: string) => ReturnType<typeof resp>) => async (url: string) => {
  const r = get(url);
  if (!r.ok) throw Object.assign(new Error(`${url} -> ${r.status}`), { status: r.status });
  return r.text();
};

test("the submission text holds the same Form 4 the listing's document does (real filing, both copies)", () => {
  const xml = ownershipXmlFromSubmission(SUB_PUBLISHED);
  assert.ok(xml, "an ownership document is found in the submission text");
  const viaSubmission = parseForm4Xml(xml!);
  const viaListing = parseForm4Xml(XML_PUBLISHED);
  assert.deepEqual(viaSubmission, viaListing);
  assert.ok(viaListing.issuerCik, "the fixture is a real Form 4 with an issuer");
  assert.ok(viaListing.owners.length >= 1);
});

test("a submission text with no ownership document reads as null, never as a row", () => {
  assert.equal(ownershipXmlFromSubmission(""), null);
  assert.equal(ownershipXmlFromSubmission("<SEC-DOCUMENT>\n<DOCUMENT>\n<TYPE>424B2\n<TEXT>\n<XML>\n<other/>\n</XML>\n</TEXT>\n</DOCUMENT>"), null);
});

test("fetchForm4Document: listing first and unchanged; a 404 listing falls back to the submission text; nothing else does", async () => {
  const base = "https://www.sec.gov/Archives/edgar/data/1368458/000168338226000008/";
  const acc = "0001683382-26-000008";
  // listing published: two requests, the submission text is never asked for
  let asked: string[] = [];
  const published = await fetchForm4Document(throwing((url) => {
    asked.push(url);
    if (url.endsWith("/index.json")) return resp(200, JSON.stringify({ directory: { item: [{ name: "ownership.xml" }] } }));
    if (url.endsWith("/ownership.xml")) return resp(200, XML_PUBLISHED);
    return resp(404);
  }), base, acc);
  assert.equal(published.via, "listing");
  assert.deepEqual(asked, [`${base}index.json`, `${base}ownership.xml`]);
  // listing not published (the 2026-11-23 evening): the submission text
  asked = [];
  const unpublished = await fetchForm4Document(throwing((url) => {
    asked.push(url);
    return url.endsWith(`/${acc}.txt`) ? resp(200, SUB_UNPUBLISHED) : resp(404, NOT_YET);
  }), base, acc);
  assert.equal(unpublished.via, "submission");
  assert.deepEqual(asked, [`${base}index.json`, `${base}${acc}.txt`]);
  assert.equal(unpublished.parsed?.issuerCik, "0001368458");
  assert.equal(unpublished.parsed?.issuerTradingSymbol, "SBH");
  // a refusal or a server error on the listing is not asked a second way
  for (const status of [403, 429, 500]) {
    asked = [];
    await assert.rejects(fetchForm4Document(throwing((url) => { asked.push(url); return resp(status); }), base, acc), new RegExp(`-> ${status}$`));
    assert.equal(asked.length, 1, `${status}: one request`);
  }
});

function feedOf(accessions: Array<[string, string]>): string {
  return `<feed>${accessions.map(([cik, acc]) => `<entry>
<title>4 - Someone (${cik.padStart(10, "0")}) (Issuer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/${cik}/${acc.replace(/-/g, "")}/${acc}-index.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-11-23 &lt;b&gt;AccNo:&lt;/b&gt; ${acc} &lt;b&gt;Size:&lt;/b&gt; 5 KB</summary>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
</entry>`).join("\n")}</feed>`;
}

test("2026-11-23 evening replayed: every listing answers 404 and the poll still returns the filing, read from its submission text", async () => {
  _resetForm4CacheForTests();
  const impl = async (url: string) => {
    if (url.includes("browse-edgar")) return resp(200, feedOf([["1368458", "0001683382-26-000008"]]));
    if (url.endsWith("/0001683382-26-000008.txt")) return resp(200, SUB_UNPUBLISHED);
    return resp(404, NOT_YET);
  };
  const filings = await fetchLatestForm4Filings(25, impl as any, 0);
  assert.equal(filings.length, 1);
  assert.equal(filings[0].accession, "0001683382-26-000008");
  assert.equal(filings[0].issuerName, "Sally Beauty Holdings, Inc.");
  const stats = form4PollStats()!;
  assert.equal(stats.returned, 1);
  assert.equal(stats.via_submission, 1);
  assert.equal(stats.failed, 0);
  assert.equal(stats.first_error, null);
  assert.equal(form4PollUnreadError(stats), null);
});

test("a poll that reads none of the Form 4s on its page is a failed poll, said without naming a filing", async () => {
  _resetForm4CacheForTests();
  _resetEdgarPollHealthForTests();
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "form4-listing-"));
  // the listing AND the submission text both 404: nothing readable
  const impl = async (url: string) =>
    url.includes("browse-edgar")
      ? resp(200, feedOf([["1368458", "0001683382-26-000008"], ["2075159", "0001493152-26-053616"]]))
      : resp(404, NOT_YET);
  const orig = console.error;
  console.error = () => {};
  try {
    await refreshForm4Cache(25, impl as any);
  } finally {
    console.error = orig;
  }
  const stats = form4PollStats()!;
  assert.equal(stats.returned, 0);
  assert.equal(stats.failed, 2);
  assert.equal(stats.first_error, "<accession>.txt -> 404");
  const status = edgarPollHealthStatus(tmp);
  const row = (status.streams as any[]).find((r) => r.stream === "filings");
  assert.equal(row.polls_ok, 0, "before this repair the poll was counted ok");
  assert.equal(row.polls_failed, 1);
  assert.equal(row.consecutive_failures, 1);
  assert.equal(row.last_error, "0 of 2 Form 4 documents could be read: <accession>.txt -> 404");
  assert.deepEqual(sanitizeDiag(status), status);
  assert.ok(!/\d{10}-\d{2}-\d{6}|Archives/.test(JSON.stringify(status)), "no accession or filing URL in the probe");
});

test("form4PollUnreadError: only a poll with failures and nothing read; an empty page and a partly read page are good polls", () => {
  const base = { at: "", feed_filings: 0, form4: 0, other_form: 0, unknown_form: 0, over_limit: 0, returned: 0, no_issuer: 0, failed: 0, via_submission: 0, first_error: null };
  assert.equal(form4PollUnreadError(null), null);
  assert.equal(form4PollUnreadError(base), null);
  assert.equal(form4PollUnreadError({ ...base, returned: 3, failed: 22, first_error: "index.json -> 404" }), null);
  assert.equal(form4PollUnreadError({ ...base, no_issuer: 2 }), null);
  assert.match(String(form4PollUnreadError({ ...base, failed: 25, first_error: "index.json -> 404" })?.message), /^0 of 25 .* index\.json -> 404$/);
});

test("describeDocumentError names the file and status, never the filing", () => {
  assert.equal(describeDocumentError(new Error("https://www.sec.gov/Archives/edgar/data/1368458/000168338226000008/index.json -> 404")), "index.json -> 404");
  assert.equal(describeDocumentError(new Error("https://www.sec.gov/Archives/edgar/data/1/2/0001683382-26-000008.txt -> 503")), "<accession>.txt -> 503");
  assert.equal(describeDocumentError(new Error("The operation was aborted due to timeout")), "The operation was aborted due to timeout");
  assert.equal(describeDocumentError("fetch https://www.sec.gov/x failed"), "fetch <url> failed");
});

test("the nightly pass reads a missing filing from its submission text when its listing is not published", async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "form4-listing-night-"));
  const idx = [
    "Form Type   Company Name                                                  CIK         Date Filed  File Name",
    "---------------------------------------------------------------------------------------------------------------------------------------------",
    "4           Sally Beauty Holdings, Inc.                                   1368458     20261123    edgar/data/1368458/0001683382-26-000008.txt",
  ].join("\n");
  const asked: string[] = [];
  const fetchImpl = async (url: string) => {
    asked.push(url);
    if (url.includes("/daily-index/")) return url.includes("form.20261123.idx") ? resp(200, idx) : resp(404);
    if (url.endsWith("/0001683382-26-000008.txt")) return resp(200, SUB_UNPUBLISHED);
    return resp(404, NOT_YET);
  };
  const run = await runForm4Completeness({
    fetch: fetchImpl as any,
    now: () => Date.parse("2026-11-24T08:00:00Z"),
    baseDir: tmp,
    delayMs: 0,
  } as any);
  const day = run.days.find((d) => d.day === "2026-11-23")!;
  assert.equal(day.failed, 0, "before this repair the 404 listing was a failed filing");
  assert.equal(day.backfilled, 1);
  assert.ok(asked.some((u) => u.endsWith("/index.json")), "the listing is still asked first");
});
