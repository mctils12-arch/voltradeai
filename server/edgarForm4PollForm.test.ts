// REGRESSION (2026-11-17): EDGAR's "getcurrent" listing treats `type=4` as a
// PREFIX. Measured live 2026-11-17 16:14Z on a 100-entry page: 30 entries
// were Form 4 (15 filings); 70 were 424B2 (59), 497J (4), 485BXT (2),
// 485APOS (2), 40-17G (2) and 497K (1) — 54 filings. The poll took the first
// 25 filings whatever their form, fell back to the first .xml in each
// directory, found no ownership fields, and archived the prospectus as a
// Form 4 row with no issuer: 189 of the 600 rows filed 2026-11-13 (measured
// 2026-11-17 02:38Z). Each such row also used one of the poll's 25 slots.
//
// The entries below are copied from that live page (public SEC records).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseFilingFeed,
  feedEntryForm,
  isForm4Type,
  fetchLatestForm4Filings,
  form4PollStats,
  _resetForm4CacheForTests,
} from "./edgarForm4";
import { form4CompletenessStatus } from "./edgarForm4Completeness";

function entry(form: string | null, name: string, cik: string, accession: string, role = "Filer", title?: string): string {
  const dir = accession.replace(/-/g, "");
  return `<entry>
<title>${title ?? `${form} - ${name} (${cik}) (${role})`}</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/${Number(cik)}/${dir}/${accession}-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-11-17 &lt;b&gt;AccNo:&lt;/b&gt; ${accession} &lt;b&gt;Size:&lt;/b&gt; 6 KB
</summary>
<updated>2026-11-17T11:10:33-05:00</updated>
${form == null ? "" : `<category scheme="https://www.sec.gov/" label="form type" term="${form}"/>`}
<id>urn:tag:sec.gov,2008:accession-number=${accession}</id>
</entry>`;
}
const feed = (...entries: string[]) =>
  `<?xml version="1.0" encoding="ISO-8859-1" ?>\n<feed xmlns="http://www.w3.org/2005/Atom">\n${entries.join("\n")}\n</feed>`;

const PROSPECTUS = entry("424B2", "JPMorgan Chase Financial Co. LLC", "0001665650", "0001918704-26-035831");
const FORM4_REPORTING = entry("4", "Walta Brian Michael", "0002149506", "0001822993-26-000185", "Reporting");
const FORM4_ISSUER = entry("4", "Jackson Financial Inc.", "0001822993", "0001822993-26-000185", "Issuer");
const FUND_485 = entry("485BXT", "Tidal Trust IV", "0002043390", "0001999371-26-025603");
const FUND_4017G = entry("40-17G", "Capital Group Multi-Asset Income Builder ETF", "0002135658", "0000051931-26-001134");
const AMENDMENT = entry("4/A", "Example Amending Owner", "0001999998", "0001999998-26-000007", "Reporting");

const ownership = (cik: string, name: string, symbol: string) =>
  `<ownershipDocument><periodOfReport>2026-11-13</periodOfReport><issuer><issuerCik>${cik}</issuerCik><issuerName>${name}</issuerName><issuerTradingSymbol>${symbol}</issuerTradingSymbol></issuer></ownershipDocument>`;

/** Serves the feed and, per accession directory, an index.json and one XML. */
function edgar(feedXml: string, docs: Record<string, string>) {
  const requested: string[] = [];
  const impl = async (url: string) => {
    requested.push(url);
    if (url.includes("browse-edgar")) return { ok: true, status: 200, text: async () => feedXml };
    for (const [accession, xml] of Object.entries(docs)) {
      const dir = accession.replace(/-/g, "");
      if (!url.includes(`/${dir}/`)) continue;
      if (url.endsWith("/index.json")) return { ok: true, status: 200, text: async () => JSON.stringify({ directory: { item: [{ name: "doc.xml" }] } }) };
      if (url.endsWith("/doc.xml")) return { ok: true, status: 200, text: async () => xml };
    }
    return { ok: false, status: 404, text: async () => "" };
  };
  const filingRequests = () => requested.filter((u) => !u.includes("browse-edgar"));
  return { impl, requested, filingRequests };
}

test("parseFilingFeed: each entry carries the form type it states (category term)", () => {
  const entries = parseFilingFeed(feed(PROSPECTUS, FORM4_REPORTING, FORM4_ISSUER, FUND_485, FUND_4017G, AMENDMENT));
  assert.deepEqual(entries.map((e) => e.form), ["424B2", "4", "485BXT", "40-17G", "4/A"]);
  assert.equal(entries[1].accession, "0001822993-26-000185", "the reporting and issuer entries are one filing");
});

test("feedEntryForm: the title's lead is read when the entry has no category; neither readable is null", () => {
  assert.equal(feedEntryForm("<title>4 - Someone (0001999999) (Reporting)</title>", "4 - Someone (0001999999) (Reporting)"), "4");
  assert.equal(feedEntryForm("", "13F-HR/A - A MANAGER, LLC (0001762716) (Filer)"), "13F-HR/A");
  assert.equal(feedEntryForm("", "1-A POS - An Issuer (0001999999) (Filer)"), "1-A POS");
  assert.equal(feedEntryForm("", "a title with no form lead"), null);
  assert.equal(feedEntryForm(`<category scheme="https://www.sec.gov/" label="form type" term=""/>`, "no lead here"), null);
  // the category wins over a title that disagrees
  assert.equal(feedEntryForm(`<category label="form type" term="424B2"/>`, "4 - Someone (0001999999) (Reporting)"), "424B2");
});

test("isForm4Type: exactly 4 and 4/A; every look-alike the prefix lets in is refused", () => {
  assert.equal(isForm4Type("4"), true);
  assert.equal(isForm4Type("4/A"), true);
  for (const other of ["424B2", "424B3", "424B5", "485BPOS", "485BXT", "485APOS", "487", "497", "497J", "497K", "40-17G", "40-F", "425", "4 ", "", "3", "5", null]) {
    assert.equal(isForm4Type(other), false, String(other));
  }
});

test("REGRESSION: prospectuses listed ahead of the Form 4s are not fetched, not returned, and take no slot", async () => {
  // 46 distinct 424B2 filings ahead of 15 Form 4 filings: the live page's
  // counts. Before the repair, limit 25 fetched 25 prospectuses and 0 Form 4s.
  const acc = (prefix: string, i: number) => `${prefix}-26-${String(i).padStart(6, "0")}`;
  const entries: string[] = [];
  const docs: Record<string, string> = {};
  for (let i = 1; i <= 46; i++) {
    const a = acc("0001918704", i);
    entries.push(entry("424B2", "A Structured Note Issuer LLC", "0001665650", a));
    docs[a] = `<FilingSummary><note>a prospectus supplement's exhibit XML: no ownership fields</note></FilingSummary>`;
  }
  for (let i = 1; i <= 15; i++) {
    const a = acc("0001822993", i);
    entries.push(entry("4", "An Owner", "0002149506", a, "Reporting"), entry("4", "An Issuer Inc.", "0001822993", a, "Issuer"));
    docs[a] = ownership("0001822993", "An Issuer Inc.", "AISS");
  }
  const e = edgar(feed(...entries), docs);
  const filings = await fetchLatestForm4Filings(25, e.impl as any, 0);

  assert.equal(filings.length, 15, "every Form 4 on the page");
  assert.ok(filings.every((f) => f.issuerCik === "0001822993" && f.issuerTradingSymbol === "AISS"));
  assert.ok(filings.every((f) => f.accession.startsWith("0001822993-")));
  assert.deepEqual(e.filingRequests().filter((u) => u.includes("/000191870426")), [], "no prospectus directory is requested");
  assert.equal(e.filingRequests().length, 30, "two requests per Form 4, none for anything else");
  assert.deepEqual(
    { ...form4PollStats(), at: null },
    { at: null, feed_filings: 61, form4: 15, other_form: 46, unknown_form: 0, over_limit: 0, returned: 15, no_issuer: 0, failed: 0, via_submission: 0, first_error: null },
  );
});

test("the limit counts Form 4s: with other forms ahead, `limit` Form 4s are still fetched and the rest reported", async () => {
  const docs = {
    "0001822993-26-000185": ownership("0001822993", "Jackson Financial Inc.", "JXN"),
    "0001999998-26-000007": ownership("0001888888", "EXAMPLE ISSUER INC", "EXMP"),
    "0001999997-26-000001": ownership("0001888887", "THIRD ISSUER INC", "THRD"),
  };
  const third = entry("4", "A Third Owner", "0001999997", "0001999997-26-000001", "Reporting");
  const e = edgar(feed(PROSPECTUS, FUND_485, FUND_4017G, FORM4_REPORTING, FORM4_ISSUER, AMENDMENT, third), docs);
  const filings = await fetchLatestForm4Filings(2, e.impl as any, 0);
  assert.deepEqual(filings.map((f) => f.accession), ["0001822993-26-000185", "0001999998-26-000007"], "a 4 and a 4/A");
  assert.equal(form4PollStats()?.over_limit, 1);
  assert.equal(form4PollStats()?.other_form, 3);
});

test("a document with no issuer is never returned, whatever the entry said its form was", async () => {
  const noForm = entry(null, "", "", "0001999996-26-000001", "Filer", "an entry whose title states no form");
  const noFormEmpty = entry(null, "", "", "0001999995-26-000001", "Filer", "another entry with no form");
  const docs = {
    "0001822993-26-000185": `<ownershipDocument><periodOfReport>2026-11-13</periodOfReport></ownershipDocument>`,
    "0001999996-26-000001": ownership("0001888886", "UNLABELLED ISSUER INC", "UNLB"),
    "0001999995-26-000001": `<FilingSummary/>`,
  };
  const e = edgar(feed(FORM4_REPORTING, FORM4_ISSUER, noForm, noFormEmpty), docs);
  const filings = await fetchLatestForm4Filings(25, e.impl as any, 0);
  // An entry that states no form is still fetched (a feed format change must
  // not silence the archive) and kept only because its document names an issuer.
  assert.deepEqual(filings.map((f) => f.accession), ["0001999996-26-000001"]);
  const s = form4PollStats()!;
  assert.equal(s.unknown_form, 2);
  assert.equal(s.no_issuer, 2);
  assert.equal(s.returned, 1);
});

test("a filing whose directory cannot be read is counted failed and does not stop the batch", async () => {
  const e = edgar(feed(AMENDMENT, FORM4_REPORTING), { "0001822993-26-000185": ownership("0001822993", "Jackson Financial Inc.", "JXN") });
  const filings = await fetchLatestForm4Filings(25, e.impl as any, 0);
  assert.deepEqual(filings.map((f) => f.accession), ["0001822993-26-000185"]);
  assert.equal(form4PollStats()?.failed, 1);
});

test("the form4_completeness probe carries the poll's latest page as counts only", async () => {
  _resetForm4CacheForTests();
  assert.equal((form4CompletenessStatus({}) as any).poll, null, "no poll yet in this process");
  const e = edgar(feed(PROSPECTUS, FORM4_REPORTING, FORM4_ISSUER), { "0001822993-26-000185": ownership("0001822993", "Jackson Financial Inc.", "JXN") });
  await fetchLatestForm4Filings(25, e.impl as any, 0);
  const poll = (form4CompletenessStatus({}) as any).poll;
  assert.equal(poll.form4, 1);
  assert.equal(poll.other_form, 1);
  assert.equal(poll.returned, 1);
  const text = JSON.stringify(poll);
  for (const leak of ["JXN", "Jackson", "0001822993", "sec.gov"]) assert.ok(!text.includes(leak), leak);
});
