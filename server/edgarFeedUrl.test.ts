// REGRESSION (2026-11-16): EDGAR's "getcurrent" listing refuses a `count`
// that is not one of the page sizes its own form offers. Measured
// 2026-11-17 00:15Z: count=75 -> HTTP 400 for type 4 and type 13F-HR; 80 and
// 100 -> the live feed. The Form 4 poll asked for limit * 3 = 75, every poll
// threw, and the Form 4 archive took no row from about 12:40Z on 2026-11-16
// (39 rows on that Monday's day file against 553 the Friday before).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  EDGAR_FEED_PAGE_SIZES,
  edgarFeedCount,
  edgarCurrentFeedUrl,
  fetchLatestForm4Filings,
} from "./edgarForm4";
import { fetchLatest13FFilings } from "./edgar13f";
import { fetchLatestEarnings8Ks } from "./sec8kEarnings";

const here = path.dirname(fileURLToPath(import.meta.url));

const countOf = (url: string): number => Number(new URL(url).searchParams.get("count"));

/** EDGAR as measured: a feed request with a count outside the offered sizes
 *  is answered 400; an offered size gets the feed. */
function edgarLike(feedXml: string, docs: Record<string, string> = {}) {
  const feedRequests: string[] = [];
  const impl = async (url: string) => {
    if (url.includes("browse-edgar")) {
      feedRequests.push(url);
      if (!EDGAR_FEED_PAGE_SIZES.includes(countOf(url))) return { ok: false, status: 400, text: async () => "Bad Request" };
      return { ok: true, status: 200, text: async () => feedXml };
    }
    for (const [suffix, body] of Object.entries(docs)) {
      if (url.endsWith(suffix)) return { ok: true, status: 200, text: async () => body };
    }
    return { ok: false, status: 404, text: async () => "" };
  };
  return { impl, feedRequests };
}

const FORM4_FEED = `<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
<title>4 - Example Reporting Owner (0001999999) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1999999/000199999926000001/0001999999-26-000001-index.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-11-16 &lt;b&gt;AccNo:&lt;/b&gt; 0001999999-26-000001 &lt;b&gt;Size:&lt;/b&gt; 5 KB</summary>
<updated>2026-11-16T19:13:03-05:00</updated>
</entry>
</feed>`;

const FORM4_DOCS = {
  "/index.json": JSON.stringify({ directory: { item: [{ name: "ownership.xml" }] } }),
  "/ownership.xml": `<ownershipDocument><periodOfReport>2026-11-13</periodOfReport><issuer><issuerCik>0001888888</issuerCik><issuerName>EXAMPLE ISSUER INC</issuerName><issuerTradingSymbol>EXMP</issuerTradingSymbol></issuer></ownershipDocument>`,
};

test("edgarFeedCount: only a page size EDGAR offers, the smallest that holds the wish", () => {
  assert.deepEqual([...EDGAR_FEED_PAGE_SIZES], [10, 20, 40, 80, 100]);
  assert.equal(edgarFeedCount(75), 80, "the refused request of 2026-11-16");
  assert.equal(edgarFeedCount(80), 80);
  assert.equal(edgarFeedCount(81), 100);
  assert.equal(edgarFeedCount(1), 10);
  assert.equal(edgarFeedCount(40), 40);
  assert.equal(edgarFeedCount(300), 100, "more than the largest page: the largest page");
  for (const bad of [0, -5, NaN, Infinity]) assert.equal(edgarFeedCount(bad), 100, `${bad}`);
  for (let wanted = 1; wanted <= 400; wanted++) {
    assert.ok(EDGAR_FEED_PAGE_SIZES.includes(edgarFeedCount(wanted)), `wanted ${wanted}`);
  }
});

test("edgarCurrentFeedUrl: the form type and an offered count, Atom output", () => {
  const u = new URL(edgarCurrentFeedUrl("13F-HR", 75));
  assert.equal(u.hostname, "www.sec.gov");
  assert.equal(u.searchParams.get("action"), "getcurrent");
  assert.equal(u.searchParams.get("type"), "13F-HR");
  assert.equal(u.searchParams.get("count"), "80");
  assert.equal(u.searchParams.get("output"), "atom");
  assert.equal(u.searchParams.get("owner"), "include");
});

test("REGRESSION: the Form 4 poll's default limit (25) asks for a count EDGAR answers, and its filing is returned", async () => {
  const edgar = edgarLike(FORM4_FEED, FORM4_DOCS);
  const filings = await fetchLatestForm4Filings(25, edgar.impl as any, 0);
  assert.equal(edgar.feedRequests.length, 1);
  assert.equal(countOf(edgar.feedRequests[0]), 80, "was limit * 3 = 75, which EDGAR refuses");
  assert.equal(filings.length, 1);
  assert.equal(filings[0].accession, "0001999999-26-000001");
  assert.equal(filings[0].filedAt, "2026-11-16");
  assert.equal(filings[0].issuerTradingSymbol, "EXMP");
});

test("every limit a caller can pass asks each of the three feeds for an offered count", async () => {
  const empty = `<feed xmlns="http://www.w3.org/2005/Atom"></feed>`;
  for (const limit of [1, 5, 15, 25, 33, 40, 50, 100]) {
    const edgar = edgarLike(empty);
    assert.deepEqual(await fetchLatestForm4Filings(limit, edgar.impl as any, 0), [], `form 4 limit ${limit}`);
    assert.deepEqual(await fetchLatest13FFilings(limit, edgar.impl as any, 0), [], `13F limit ${limit}`);
    assert.deepEqual(await fetchLatestEarnings8Ks(limit, edgar.impl as any, 0), [], `8-K limit ${limit}`);
    assert.equal(edgar.feedRequests.length, 3);
    for (const url of edgar.feedRequests) assert.ok(EDGAR_FEED_PAGE_SIZES.includes(countOf(url)), url);
  }
});

test("RATCHET: no server module writes a getcurrent feed URL of its own", () => {
  const offenders: string[] = [];
  for (const name of fs.readdirSync(here)) {
    if (!name.endsWith(".ts") || name.endsWith(".test.ts")) continue;
    const src = fs.readFileSync(path.join(here, name), "utf8");
    const hits = src.split("action=getcurrent").length - 1;
    if (hits > (name === "edgarForm4.ts" ? 1 : 0)) offenders.push(`${name} (${hits})`);
  }
  assert.deepEqual(offenders, [], "build the URL with edgarCurrentFeedUrl(), which only asks for a page size EDGAR offers");
});
