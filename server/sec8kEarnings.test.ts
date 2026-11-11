import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parse8KFeed,
  hasItem202,
  pickExhibit99Href,
  htmlToText,
  fetchLatestEarnings8Ks,
  MAX_EXHIBIT_TEXT_LEN,
} from "./sec8kEarnings";

// ROOT VALIDATION LADDER gate 1 (DATA) fixtures — every block below is a
// real snippet fetched live from SEC EDGAR on 2026-07-04 (real accessions,
// not synthetic), and every extracted fact was hand-checked against the
// filed exhibit itself before being asserted here. The filing text IS the
// ground truth for this pipeline (same rationale edgarForm4.test.ts states
// for Form 4 — there is no separate "official" source to check it against).

// ── Fixture A: getcurrent 8-K Atom feed, two real entries ───────────────────
// AppTech Payments Corp 8-K (accession 0001683168-26-005262, filed
// 2026-07-02) tags Items 1.01/2.03/9.01 — NO Item 2.02, must be filtered out.
// MV Oil Trust 8-K (accession 0001104659-26-080431, filed 2026-07-02) tags
// Item 2.02 among others — must pass the filter.
const REAL_8K_FEED = `<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Latest Filings - Sat, 04 Jul 2026 14:10:37 EDT</title>
<entry>
<title>8-K - AppTech Payments Corp. (0001070050) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1070050/000168316826005262/0001683168-26-005262-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-07-02 &lt;b&gt;AccNo:&lt;/b&gt; 0001683168-26-005262 &lt;b&gt;Size:&lt;/b&gt; 270 KB
&lt;br&gt;Item 1.01: Entry into a Material Definitive Agreement
&lt;br&gt;Item 2.03: Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement of a Registrant
&lt;br&gt;Item 9.01: Financial Statements and Exhibits
</summary>
<updated>2026-07-02T17:29:10-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0001683168-26-005262</id>
</entry>
<entry>
<title>8-K - MV Oil Trust (0001371782) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1371782/000110465926080431/0001104659-26-080431-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-07-02 &lt;b&gt;AccNo:&lt;/b&gt; 0001104659-26-080431 &lt;b&gt;Size:&lt;/b&gt; 43 KB
&lt;br&gt;Item 2.02: Results of Operations and Financial Condition
&lt;br&gt;Item 3.01: Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing
&lt;br&gt;Item 9.01: Financial Statements and Exhibits
</summary>
<updated>2026-07-02T17:21:41-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0001104659-26-080431</id>
</entry>
</feed>`;

test("parse8KFeed extracts both real entries with correct fields", () => {
  const entries = parse8KFeed(REAL_8K_FEED);
  assert.equal(entries.length, 2);
  const [appTech, mvOil] = entries;
  assert.equal(appTech.accession, "0001683168-26-005262");
  assert.equal(appTech.cik, "0001070050");
  assert.equal(appTech.companyName, "AppTech Payments Corp.");
  assert.deepEqual(appTech.itemCodes, ["1.01", "2.03", "9.01"]);
  assert.equal(mvOil.accession, "0001104659-26-080431");
  assert.equal(mvOil.cik, "0001371782");
  assert.equal(mvOil.companyName, "MV Oil Trust");
  assert.deepEqual(mvOil.itemCodes, ["2.02", "3.01", "9.01"]);
  assert.equal(
    mvOil.indexUrl,
    "https://www.sec.gov/Archives/edgar/data/1371782/000110465926080431/0001104659-26-080431-index.htm",
  );
});

test("hasItem202 filters correctly — this is the entire earnings-language gate", () => {
  const entries = parse8KFeed(REAL_8K_FEED);
  const filtered = entries.filter(hasItem202);
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].companyName, "MV Oil Trust");
});

// ── Fixture B: real index.htm document tables, two real filer-agent formats
// (row content varies: "EX-99" vs "EXHIBIT 99.1" in the description column,
// both landing in an EX-99* Type column — confirmed by fetching both live).

const REAL_INDEX_UNIFIRST = `<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr class="oddRow">
   <td scope="row">1</td>
   <td scope="row">10-Q</td>
   <td scope="row"><a href="/Archives/edgar/data/717954/000162828026046349/unf-20260701.htm">unf-20260701.htm</a></td>
   <td scope="row">8-K</td>
   <td scope="row">27788</td>
</tr>
<tr class="evenRow">
   <td scope="row">2</td>
   <td scope="row">EX-99</td>
   <td scope="row"><a href="/Archives/edgar/data/717954/000162828026046349/unf-2026xq3xex99earningsre.htm">unf-2026xq3xex99earningsre.htm</a></td>
   <td scope="row">EX-99</td>
   <td scope="row">387455</td>
</tr>
</table>`;

const REAL_INDEX_MVOIL = `<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr class="oddRow">
   <td scope="row">1</td>
   <td scope="row">8-K</td>
   <td scope="row"><a href="/Archives/edgar/data/1371782/000110465926080431/tm2618490d1_8k.htm">tm2618490d1_8k.htm</a></td>
   <td scope="row">8-K</td>
   <td scope="row">12345</td>
</tr>
<tr class="evenRow">
   <td scope="row">2</td>
   <td scope="row">EXHIBIT 99.1</td>
   <td scope="row"><a href="/Archives/edgar/data/1371782/000110465926080431/tm2618490d1_ex99-1.htm">tm2618490d1_ex99-1.htm</a></td>
   <td scope="row">EX-99.1</td>
   <td scope="row">19884</td>
</tr>
</table>`;

test("pickExhibit99Href finds the exhibit regardless of description-column wording", () => {
  assert.equal(
    pickExhibit99Href(REAL_INDEX_UNIFIRST),
    "/Archives/edgar/data/717954/000162828026046349/unf-2026xq3xex99earningsre.htm",
  );
  assert.equal(
    pickExhibit99Href(REAL_INDEX_MVOIL),
    "/Archives/edgar/data/1371782/000110465926080431/tm2618490d1_ex99-1.htm",
  );
});

test("pickExhibit99Href returns null when no EX-99 row exists (HONEST GAP case)", () => {
  const noExhibit = `<table><tr><td>1</td><td>8-K</td><td><a href="/a/8k.htm">8k.htm</a></td><td>8-K</td><td>100</td></tr></table>`;
  assert.equal(pickExhibit99Href(noExhibit), null);
});

// ── Fixture C: real Exhibit 99 documents, byte-for-byte as fetched, verified
// against the actual UniFirst Q3 FY2026 and MV Oil Trust press releases.

const REAL_EXHIBIT_UNIFIRST = `<DOCUMENT>
<TYPE>EX-99
<SEQUENCE>2
<FILENAME>unf-2026xq3xex99earningsre.htm
<DESCRIPTION>EX-99
<TEXT>
<html><head>
<!-- Document created using Wdesk -->
<!-- Copyright 2026 Workiva -->
<title>Document</title></head><body><div id="iea7839edb16e47bc8193c067143768c0_1"></div><div style="min-height:45pt;width:100%"><div style="text-align:justify"><font><br></font></div></div><div style="text-align:right"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:700;line-height:120%"> Exhibit 99</font></div><div style="text-align:justify"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:700;line-height:120%">Investor Relations Contact</font></div><div style="text-align:justify"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:9pt;font-weight:400;line-height:120%">Shane O'Connor, Executive Vice President &#38; CFO</font></div><div style="text-align:justify"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:9pt;font-weight:400;line-height:120%">UniFirst Corporation</font></div><div style="text-align:justify"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:9pt;font-weight:400;line-height:120%">978-658-8888</font></div><div style="margin-bottom:12pt;text-align:justify"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:9pt;font-weight:400;line-height:120%">shane_oconnor&#64;unifirst.com</font></div><div style="margin-bottom:12pt;text-align:center"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:14pt;font-weight:700;line-height:120%">UNIFIRST ANNOUNCES FINANCIAL RESULTS FOR THE THIRD QUARTER OF FISCAL 2026</font></div><div style="margin-bottom:6pt;text-align:justify"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:700;line-height:120%">Wilmington, MA &#8211; July&#160;1, 2026</font><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:400;line-height:120%"> &#8211; UniFirst Corporation (NYSE&#58; UNF) (&#8220;UniFirst&#8221; or the &#8220;Company&#8221;) today reported results for its fiscal 2026 third quarter ended May 30, 2026.</font></div><div style="margin-bottom:6pt;text-align:justify"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:700;line-height:120%">Third Quarter 2026 Consolidated Results</font></div><div style="padding-left:94.5pt;text-align:justify;text-indent:-49.5pt"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:400;line-height:120%">&#8226;</font><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:400;line-height:120%;padding-left:44.52pt">Consolidated revenues increased 3.9% to $634.4 million compared to $610.8 million in the third quarter of fiscal 2025, driven by organic growth in the core Uniform &#38; Facility Service Solutions segment.</font></div><div style="padding-left:94.5pt;text-align:justify;text-indent:-49.5pt"><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:400;line-height:120%">&#8226;</font><font style="color:#000000;font-family:'Calibri',sans-serif;font-size:10pt;font-weight:400;line-height:120%;padding-left:44.52pt">Operating income and Adjusted EBITDA were $23.0 million and $82.6 million, respectively, compared to $48.2 million and $85.8 million, respectively, in the third quarter of fiscal 2025.</font></div>`;

test("htmlToText decodes numeric entities and strips tags — UniFirst exhibit, hand-verified against the filed release", () => {
  const text = htmlToText(REAL_EXHIBIT_UNIFIRST);
  // Verified against the actual UniFirst Q3 FY2026 press release (NYSE: UNF):
  assert.match(text, /UNIFIRST ANNOUNCES FINANCIAL RESULTS FOR THE THIRD QUARTER OF FISCAL 2026/);
  assert.match(text, /Shane O'Connor, Executive Vice President & CFO/); // &#38; -> &
  assert.match(text, /shane_oconnor@unifirst\.com/); // &#64; -> @
  assert.match(text, /Wilmington, MA – July 1, 2026/); // &#8211; -> – ; &#160; -> normal space, not NBSP
  assert.match(text, /“UniFirst” or the “Company”/); // &#8220;/&#8221; -> curly quotes
  assert.match(text, /Consolidated revenues increased 3\.9% to \$634\.4 million compared to \$610\.8 million/);
  assert.match(text, /Operating income and Adjusted EBITDA were \$23\.0 million and \$82\.6 million/);
  assert.ok(!text.includes("<div"), "no HTML tags should survive extraction");
  assert.ok(!text.includes("&#"), "no raw numeric entities should survive decoding");
  assert.ok(!text.includes(" "), "NBSP should be normalized to a regular space");
});

const REAL_EXHIBIT_MVOIL = `<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>tm2618490d1_ex99-1.htm
<DESCRIPTION>EXHIBIT 99.1
<TEXT>
<HTML>
<HEAD>
     <TITLE></TITLE>
</HEAD>
<BODY STYLE="font: 10pt Times New Roman, Times, Serif">

<P STYLE="margin: 0">&nbsp;</P>

<P STYLE="text-align: right; margin: 0"><B>Exhibit 99.1</B></P>

<P STYLE="margin: 0">&nbsp;</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">MV Oil Trust</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">&nbsp;</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0"><B>MV Oil Trust Announces Final Trust Distribution</B></P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">&nbsp;</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0"><B>MV OIL TRUST</B></P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0"><B>The Bank of New York Mellon Trust Company, N.A., Trustee</B></P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0"><B>&nbsp;</B></P>

<P STYLE="margin: 0pt 1in 0pt 0; font: 10pt Times New Roman, Times, Serif; text-align: right"><B><I>NEWS
RELEASE</I></B></P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">&nbsp;</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0"><B>FOR IMMEDIATE RELEASE</B></P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">&nbsp;</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">Houston, Texas, July 2, 2026 &mdash; MV Oil Trust (NYSE: MVO) announced
the Trust distribution of net profits for the quarterly payment period ended June 30, 2026.</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">&nbsp;</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">Unitholders of record on July 15, 2026 will receive a distribution
amounting to $6,829,206 or $0.593844 per unit payable July 24, 2026. This distribution will be the final Trust distribution.</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">&nbsp;</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">Volumes, average price and net profits for the payment period were:</P>

<P STYLE="font: 10pt Times New Roman, Times, Serif; margin: 0pt 0">&nbsp;</P>

<TABLE CELLSPACING="0" CELLPADDING="0" ALIGN="CENTER" STYLE="font: 10pt Times New Roman, Times, Serif; width: 90%; border-collapse: collapse">
  <TR STYLE="background-color: #CCEEFF">
    <TD STYLE="vertical-align: top; padding-left: 10.1pt; text-indent: -10.1pt"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">Volume
    (BOE)</FONT></TD>
    <TD STYLE="vertical-align: bottom">&nbsp;</TD>
    <TD COLSPAN="2" STYLE="vertical-align: bottom; text-align: right"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">140,993</FONT></TD>
    <TD STYLE="vertical-align: bottom">&nbsp;</TD></TR>
  <TR>
    <TD STYLE="vertical-align: top; width: 84%; padding-left: 10.1pt; text-indent: -10.1pt"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">Average
    price (per BOE)</FONT></TD>
    <TD STYLE="vertical-align: bottom; width: 2%">&nbsp;</TD>
    <TD STYLE="vertical-align: bottom; width: 1%"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">$</FONT></TD>
    <TD STYLE="vertical-align: bottom; width: 12%; text-align: right"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">91.58</FONT></TD>
    <TD STYLE="vertical-align: bottom; width: 1%">&nbsp;</TD></TR>
  <TR STYLE="background-color: #CCEEFF">
    <TD STYLE="vertical-align: top; padding-left: 10.1pt; text-indent: -10.1pt"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">Gross
    proceeds</FONT></TD>
    <TD STYLE="vertical-align: bottom">&nbsp;</TD>
    <TD STYLE="vertical-align: bottom"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">$</FONT></TD>
    <TD STYLE="vertical-align: bottom; text-align: right"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">12,911,770</FONT></TD>
    <TD STYLE="vertical-align: bottom">&nbsp;</TD></TR>
  <TR>
    <TD STYLE="vertical-align: top; padding-left: 10.1pt; text-indent: -10.1pt"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">Net
    cash proceeds available for distribution</FONT></TD>
    <TD STYLE="vertical-align: bottom">&nbsp;</TD>
    <TD STYLE="vertical-align: bottom"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">$</FONT></TD>
    <TD STYLE="vertical-align: bottom; text-align: right"><FONT STYLE="font-family: Times New Roman, Times, Serif; font-size: 10pt">6,829,206</FONT></TD>
    <TD STYLE="vertical-align: bottom">&nbsp;</TD></TR>
  </TABLE>`;

test("htmlToText handles the second real filer-agent format (uppercase tags, named entities, tables) — MV Oil Trust exhibit", () => {
  const text = htmlToText(REAL_EXHIBIT_MVOIL);
  // Verified against the actual MV Oil Trust final-distribution press release (NYSE: MVO):
  assert.match(text, /MV Oil Trust Announces Final Trust Distribution/);
  assert.match(text, /Houston, Texas, July 2, 2026 — MV Oil Trust \(NYSE: MVO\) announced/); // &mdash; -> —
  assert.match(text, /distribution\s+amounting to \$6,829,206 or \$0\.593844 per unit/);
  assert.ok(text.includes("140,993"), "table cell values survive extraction (volume, BOE)");
  assert.ok(text.includes("91.58"), "table cell values survive extraction (avg price)");
  assert.ok(text.includes("12,911,770"), "table cell values survive extraction (gross proceeds)");
  assert.ok(!text.includes("<TD"), "no HTML tags should survive extraction");
  assert.ok(!text.includes("&nbsp;") && !text.includes("&mdash;"), "no raw entities should survive decoding");
});

// ── End-to-end fetch layer, using an injected fake fetch over the exact real
// fixtures above (no live network in the test suite itself — the fixtures
// above are what makes this an honest gate-1 check rather than a mock test).

function fakeFetch(routes: Record<string, string>) {
  return async (url: string) => {
    for (const key of Object.keys(routes)) {
      if (url.includes(key)) return { ok: true, status: 200, text: async () => routes[key] };
    }
    return { ok: false, status: 404, text: async () => "" };
  };
}

test("fetchLatestEarnings8Ks end-to-end: filters Item 2.02, finds the exhibit, extracts text, skips non-matching filings", async () => {
  const routes: Record<string, string> = {
    "action=getcurrent": REAL_8K_FEED,
    "1371782/000110465926080431/0001104659-26-080431-index.htm": REAL_INDEX_MVOIL,
    "tm2618490d1_ex99-1.htm": REAL_EXHIBIT_MVOIL,
  };
  const results = await fetchLatestEarnings8Ks(15, fakeFetch(routes) as any, 0);
  // AppTech (no Item 2.02) must never appear; only MV Oil Trust should.
  assert.equal(results.length, 1);
  const r = results[0];
  assert.equal(r.companyName, "MV Oil Trust");
  assert.equal(r.accession, "0001104659-26-080431");
  assert.deepEqual(r.itemCodes, ["2.02", "3.01", "9.01"]);
  assert.equal(r.exhibitUrl, "https://www.sec.gov/Archives/edgar/data/1371782/000110465926080431/tm2618490d1_ex99-1.htm");
  assert.match(r.text, /MV Oil Trust Announces Final Trust Distribution/);
  assert.equal(r.truncated, false);
  assert.equal(r.textLength, r.text.length);
});

test("fetchLatestEarnings8Ks skips a matching filing with no EX-99 exhibit (HONEST GAP: results announced only in the 8-K body)", async () => {
  const noExhibitIndex = `<table><tr><td>1</td><td>8-K</td><td><a href="/a/8k.htm">8k.htm</a></td><td>8-K</td><td>100</td></tr></table>`;
  const routes: Record<string, string> = {
    "action=getcurrent": REAL_8K_FEED,
    "1371782/000110465926080431/0001104659-26-080431-index.htm": noExhibitIndex,
  };
  const results = await fetchLatestEarnings8Ks(15, fakeFetch(routes) as any, 0);
  assert.equal(results.length, 0);
});

test("MAX_EXHIBIT_TEXT_LEN truncation is honest — truncated flag set and textLength reports the real, untruncated length", async () => {
  const longText = "X".repeat(MAX_EXHIBIT_TEXT_LEN + 5000);
  const longExhibit = `<DOCUMENT><TEXT><html><body><p>${longText}</p></body></html>`;
  const routes: Record<string, string> = {
    "action=getcurrent": REAL_8K_FEED,
    "1371782/000110465926080431/0001104659-26-080431-index.htm": REAL_INDEX_MVOIL,
    "tm2618490d1_ex99-1.htm": longExhibit,
  };
  const results = await fetchLatestEarnings8Ks(15, fakeFetch(routes) as any, 0);
  assert.equal(results.length, 1);
  assert.equal(results[0].truncated, true);
  assert.equal(results[0].text.length, MAX_EXHIBIT_TEXT_LEN);
  assert.equal(results[0].textLength, MAX_EXHIBIT_TEXT_LEN + 5000);
});

// ── [REPAIR 2026-07-05, audit defect #5] manifest drift: acceptanceDatetime
// + ticker were DOCUMENTED in datacore/manifests/earnings8k.json but never
// stored — downstream gate-2 work reading the manifest would assume a
// lookahead-free timestamp that didn't exist. These pin the fields for real.
import { getCikTickerMap, resetCikTickerCacheForTests } from "./sec8kEarnings";

test("parse8KFeed captures the entry <updated> timestamp as acceptanceDatetime", () => {
  const atom = `<feed><entry>
<title>8-K - EXAMPLE CORP (0000320193)</title>
<link rel="alternate" href="https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/0000320193-26-000001-index.htm"/>
<summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-07-02 &lt;b&gt;AccNo:&lt;/b&gt; 0000320193-26-000001 &lt;b&gt;Size:&lt;/b&gt; 20 KB Item 2.02: Results of Operations</summary>
<updated>2026-07-02T16:31:22-04:00</updated>
</entry></feed>`;
  const entries = parse8KFeed(atom);
  assert.equal(entries.length, 1);
  assert.equal(entries[0].acceptanceDatetime, "2026-07-02T16:31:22-04:00",
    "the lookahead-free event time the manifest documents must be stored");
});

test("getCikTickerMap: exact CIK match, unlisted filers stay null-able, fetch failure degrades to empty", async () => {
  resetCikTickerCacheForTests();
  const good = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({
    "0": { cik_str: 320193, ticker: "AAPL", title: "Apple Inc." },
  }) });
  const map = await getCikTickerMap(good as any);
  assert.equal(map.get("320193"), "AAPL");
  assert.equal(map.get("999999999"), undefined, "unlisted CIK resolves to nothing — never guessed");
});

// [REPAIR 2026-07-11] multi-security-class CIKs: SEC lists a row per listed
// class (common, warrants, preferred, units, rights). The common/primary
// class must win regardless of feed order — live-verified against CIK
// 797468 (Occidental Petroleum: "OXY" common + "OXY-WT" warrants).
test("getCikTickerMap: primary (unsuffixed) ticker wins over a warrant/preferred suffix, common-then-suffixed order", async () => {
  resetCikTickerCacheForTests();
  const fx = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({
    "0": { cik_str: 797468, ticker: "OXY", title: "OCCIDENTAL PETROLEUM CORP /DE/" },
    "1": { cik_str: 797468, ticker: "OXY-WT", title: "OCCIDENTAL PETROLEUM CORP /DE/" },
  }) });
  const map = await getCikTickerMap(fx as any);
  assert.equal(map.get("797468"), "OXY");
});

test("getCikTickerMap: primary (unsuffixed) ticker wins over a warrant/preferred suffix, suffixed-then-common order (order-independent)", async () => {
  resetCikTickerCacheForTests();
  const fx = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({
    "0": { cik_str: 797468, ticker: "OXY-WT", title: "OCCIDENTAL PETROLEUM CORP /DE/" },
    "1": { cik_str: 797468, ticker: "OXY", title: "OCCIDENTAL PETROLEUM CORP /DE/" },
  }) });
  const map = await getCikTickerMap(fx as any);
  assert.equal(map.get("797468"), "OXY");
});

test("getCikTickerMap: no primary class exists — falls back to the first-seen suffixed ticker (never fabricated)", async () => {
  resetCikTickerCacheForTests();
  const fx = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({
    "0": { cik_str: 1234567, ticker: "ZZZ-WT", title: "SPAC SHELL CORP" },
    "1": { cik_str: 1234567, ticker: "ZZZ-U", title: "SPAC SHELL CORP" },
  }) });
  const map = await getCikTickerMap(fx as any);
  assert.equal(map.get("1234567"), "ZZZ-WT");
});

// ── Archive + cold-cache backfill (2026-09-15 — cold-cache-no-disk-backfill
// audit; mirrors edgarForm4.test.ts's identically-shaped archive/backfill
// tests, same archiveBaseDir()/DATA_DIR resolution convention) ─────────────
import fs2 from "node:fs";
import os2 from "node:os";
import path2 from "node:path";
import {
  archiveEarnings8Ks, readEarnings8kHistory, gzipOldEarnings8kDays,
  refreshEarnings8kCache, latestEarnings8Ks, _resetEarnings8kCacheForTests,
  type Earnings8K,
} from "./sec8kEarnings";

const mkFiling8k = (acc: string, filedAt: string): Earnings8K => ({
  accession: acc, cik: "1", companyName: "TEST CO", filedAt,
  acceptanceDatetime: filedAt, ticker: "TST", itemCodes: ["2.02"],
  indexUrl: `https://www.sec.gov/x/${acc}/`, exhibitUrl: `https://www.sec.gov/x/${acc}/ex99.htm`,
  text: "results text", textLength: 12, truncated: false,
});

test("archiveEarnings8Ks appends once per accession (restart-safe dedup) and history reads back", () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-"));
  const t0 = Date.UTC(2026, 6, 4, 12, 0, 0);
  assert.equal(archiveEarnings8Ks([mkFiling8k("E-1", "2026-07-04"), mkFiling8k("E-2", "2026-07-04")], base, t0), 2);
  assert.equal(archiveEarnings8Ks([mkFiling8k("E-1", "2026-07-04"), mkFiling8k("E-3", "2026-07-04")], base, t0 + 1000), 1,
    "already-archived accession must not duplicate");
  const hist = readEarnings8kHistory(7, base, t0 + 2000);
  assert.equal(hist.length, 3);
  assert.ok(hist.every((f) => f.companyName === "TEST CO"));
});

test("gzipped old earnings8k days remain readable through readEarnings8kHistory", () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-gz-"));
  const t0 = Date.UTC(2026, 6, 1, 12, 0, 0);
  archiveEarnings8Ks([mkFiling8k("G-1", "2026-07-01")], base, t0);
  const gz = gzipOldEarnings8kDays(base, t0 + 3 * 86400_000);
  assert.equal(gz, 1, "old day file should gzip");
  const hist = readEarnings8kHistory(7, base, t0 + 3 * 86400_000);
  assert.equal(hist.length, 1);
  assert.equal(hist[0].accession, "G-1");
});

test("refreshEarnings8kCache: cold cache backfills from the on-disk earnings8k archive when the live poll throws — the same cold-cache-no-disk-backfill class edgarForm4.ts's refreshForm4Cache already closed, found unfixed here by the 2026-09-15 module audit (a redeploy/SEC-EDGAR-outage must not report warming_up over real archived filings, and must not silently cache an empty list either)", async () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-coldcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetEarnings8kCacheForTests();
  try {
    // earnings8kDir(undefined) resolves through archiveBaseDir() ->
    // DATA_DIR/datacore_archive/earnings8k — mirrored here so the archived
    // day lands where refreshEarnings8kCache's own baseDir-less backfill
    // path will actually look for it. Dated "today" (real wall clock)
    // since refreshEarnings8kCache/backfillEarnings8kFromArchive take no
    // injectable nowMs and the default 5-day lookback is measured from
    // Date.now().
    const dir = path2.join(base, "datacore_archive", "earnings8k");
    fs2.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs2.writeFileSync(path2.join(dir, `${today}.jsonl`), JSON.stringify(mkFiling8k("COLD-1", today)) + "\n");

    assert.equal(latestEarnings8Ks(), null, "cache must still be cold going into this cycle");
    const origFetch = global.fetch;
    global.fetch = (async () => { throw new Error("SEC EDGAR unreachable"); }) as any;
    try {
      await refreshEarnings8kCache(15);
    } finally {
      global.fetch = origFetch;
    }
    const cached = latestEarnings8Ks();
    assert.ok(cached, "cache must be populated, not left null, despite the live poll throwing");
    assert.equal(cached!.filings.length, 1);
    assert.equal(cached!.filings[0].accession, "COLD-1", "backfilled from the archived filing, not fabricated");
  } finally {
    _resetEarnings8kCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshEarnings8kCache: an empty-but-non-throwing live poll also backfills from disk when the cache is cold (not just the throw path)", async () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-coldcache-empty-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetEarnings8kCacheForTests();
  try {
    const dir = path2.join(base, "datacore_archive", "earnings8k");
    fs2.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs2.writeFileSync(path2.join(dir, `${today}.jsonl`), JSON.stringify(mkFiling8k("COLD-2", today)) + "\n");

    // An empty (no Item 2.02 filings this cycle), NON-throwing atom feed —
    // fetchLatestEarnings8Ks resolves to [] rather than throwing.
    const origFetch = global.fetch;
    global.fetch = (async () => ({
      ok: true, status: 200,
      text: async () => `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>`,
    })) as any;
    try {
      await refreshEarnings8kCache(15);
    } finally {
      global.fetch = origFetch;
    }
    const cached = latestEarnings8Ks();
    assert.ok(cached, "cache must be populated from disk, not left null, on an empty-but-successful poll");
    assert.equal(cached!.filings.length, 1);
    assert.equal(cached!.filings[0].accession, "COLD-2");
  } finally {
    _resetEarnings8kCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("history route + poll-loop archiving are wired", () => {
  const routes = fs2.readFileSync(path2.join(path2.dirname(new URL(import.meta.url).pathname), "routes.ts"), "utf8");
  assert.ok(routes.includes("/api/data/earnings-language"), "earnings-language route missing");
  const mod = fs2.readFileSync(path2.join(path2.dirname(new URL(import.meta.url).pathname), "sec8kEarnings.ts"), "utf8");
  assert.ok(mod.includes("archiveEarnings8Ks(filings)"), "refresh loop must archive every poll");
});

// ── Windowed history reader (2026-11-11): the /history routes used to
// answer `?days=90` with the newest 500 filings and echo `days: 90` back —
// in the Q3 2026 season that was 4 trading days presented as 90 with no
// marker. These tests pin the honest replacement. ───────────────────────────
import {
  parseEarnings8kHistoryQuery, readEarnings8kHistoryWindow, earnings8kHistoryNextQuery,
  omitEarnings8kText, EARNINGS8K_ARCHIVE_START, EARNINGS8K_HISTORY_MAX_FILINGS,
  EARNINGS8K_HISTORY_MAX_SPAN_DAYS,
} from "./sec8kEarnings";

/** Writes `perDay` filings into each listed UTC archive day. Accessions carry
 *  a per-seed tag because archiveEarnings8Ks dedups through a MODULE-level
 *  accession set that outlives each test's temp dir — the same accession
 *  string reused by a later test would be silently dropped as "already
 *  archived" (the first version of these tests hit exactly that). */
let seedTag = 0;
function seedArchiveDays(base: string, days: string[], perDay: number): void {
  const tag = `S${++seedTag}`;
  for (const day of days) {
    const t = Date.parse(`${day}T12:00:00Z`);
    const rows = Array.from({ length: perDay }, (_, i) => mkFiling8k(`${tag}-${day}-${i}`, day));
    assert.equal(archiveEarnings8Ks(rows, base, t), perDay);
  }
}

const NOW_11_11 = Date.UTC(2026, 10, 11, 13, 0, 0);

test("parseEarnings8kHistoryQuery: legacy ?days keeps working (today-(N-1)..today, capped at 90, default 30) and never throws on junk", () => {
  const w = parseEarnings8kHistoryQuery({ days: "7" }, NOW_11_11);
  assert.deepEqual([w.from, w.to, w.days], ["2026-11-05", "2026-11-11", 7]);
  assert.equal(parseEarnings8kHistoryQuery({ days: "200" }, NOW_11_11).days, 90, "the 90-day legacy cap is unchanged");
  assert.equal(parseEarnings8kHistoryQuery({}, NOW_11_11).days, 30);
  assert.equal(parseEarnings8kHistoryQuery({ days: "banana" }, NOW_11_11).days, 30);
  const junk = parseEarnings8kHistoryQuery({ from: "not-a-day", to: "2026-13-45", limit: "-4", text: "maybe" }, NOW_11_11);
  assert.deepEqual([junk.from, junk.to, junk.maxFilings, junk.includeText, junk.days], ["2026-10-13", "2026-11-11", 1, true, 30],
    "malformed from/to fall back to the legacy default window; limit clamps to >= 1");
  assert.equal(parseEarnings8kHistoryQuery({ days: "7", limit: "9999" }, NOW_11_11).maxFilings, EARNINGS8K_HISTORY_MAX_FILINGS);
});

test("parseEarnings8kHistoryQuery: explicit from/to is an archive-day window — clamped to today, to the archive start, to from<=to, and to the max span; text=0 / fields=meta omit text", () => {
  const w = parseEarnings8kHistoryQuery({ from: "2026-09-01", to: "2026-09-30", text: "0" }, NOW_11_11);
  assert.deepEqual([w.from, w.to, w.includeText, w.days], ["2026-09-01", "2026-09-30", false, null], "from/to wins; `days` is null (not echoed as a fake lookback)");
  assert.equal(parseEarnings8kHistoryQuery({ from: "2026-09-01", to: "2027-01-01" }, NOW_11_11).to, "2026-11-11", "to is clamped to today");
  assert.equal(parseEarnings8kHistoryQuery({ from: "2020-01-01", to: "2026-07-10" }, NOW_11_11).from, EARNINGS8K_ARCHIVE_START, "nothing exists before the archive start");
  const swapped = parseEarnings8kHistoryQuery({ from: "2026-10-20", to: "2026-10-01" }, NOW_11_11);
  assert.equal(swapped.from, swapped.to, "from > to collapses to a one-day window rather than throwing");
  const span = parseEarnings8kHistoryQuery({ from: "2026-07-04", to: "2028-06-01" }, Date.UTC(2028, 5, 1));
  assert.equal(Math.round((Date.parse(span.to) - Date.parse(span.from)) / 86400_000) + 1, EARNINGS8K_HISTORY_MAX_SPAN_DAYS, "span bounded by raising from");
  assert.equal(parseEarnings8kHistoryQuery({ days: "3", fields: "meta" }, NOW_11_11).includeText, false);
  assert.equal(parseEarnings8kHistoryQuery({ to: "2026-11-10" }, NOW_11_11).from, "2026-10-12", "to alone => 30 days ending at to");
});

test("readEarnings8kHistoryWindow: whole archive days are atomic, the page stops BEFORE the day that would exceed the cap, and truncated/covered/next_to say exactly what was read", () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-win-"));
  // 6 archive days x 40 filings = 240; cap 100 => page 1 holds 11-10 + 11-09 (80), stops before 11-06
  seedArchiveDays(base, ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"], 40);
  const win = parseEarnings8kHistoryQuery({ from: "2026-11-01", to: "2026-11-10", limit: "100" }, NOW_11_11);
  const page = readEarnings8kHistoryWindow(win, base);
  assert.equal(page.filings.length, 80, "80 not 100 — a day is never split");
  assert.equal(page.truncated, true);
  assert.deepEqual(page.covered, { from: "2026-11-07", to: "2026-11-10" },
    "covered names the days read in full — including the empty weekend days 11-07/11-08, which WERE read (an empty day is covered, not skipped)");
  assert.equal(page.next_to, "2026-11-06", "the first day NOT read is the cursor");
  assert.deepEqual(page.window, { from: "2026-11-01", to: "2026-11-10" }, "the requested window is echoed, not silently rewritten");
  assert.equal(page.days_scanned, 4);
  assert.equal(page.text_omitted, false);
  assert.equal(earnings8kHistoryNextQuery(page), "?from=2026-11-01&to=2026-11-06&limit=100");
  assert.ok((page.filings[0] as any).filedAt >= (page.filings[79] as any).filedAt, "newest first");
});

test("readEarnings8kHistoryWindow: walking the cursor to the end covers every filing exactly once (no gaps, no duplicates) and the last page is not truncated", () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-walk-"));
  const days = ["2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"];
  seedArchiveDays(base, days, 35); // 350 total
  assert.equal(gzipOldEarnings8kDays(base, NOW_11_11), 9, "every day older than 2 days gzips (11-10 stays plain) — a mixed gz/plain archive must page identically");
  const seen = new Set<string>();
  let win = parseEarnings8kHistoryQuery({ from: "2026-10-25", to: "2026-11-10", limit: "100", text: "0" }, NOW_11_11);
  let pages = 0;
  for (;;) {
    const page = readEarnings8kHistoryWindow(win, base);
    pages++;
    for (const f of page.filings) {
      assert.ok(!seen.has(f.accession), `duplicate across pages: ${f.accession}`);
      seen.add(f.accession);
      assert.ok(!("text" in f), "text=0 strips the exhibit text");
      assert.equal((f as any).textLength, 12, "textLength survives so the consumer knows what it skipped");
    }
    if (!page.truncated) {
      assert.equal(page.next_to, null);
      assert.equal(earnings8kHistoryNextQuery(page), null);
      assert.equal(page.covered.from, "2026-10-25", "the final page reaches window.from even through empty days");
      break;
    }
    const next = earnings8kHistoryNextQuery(page)!;
    assert.match(next, /^\?from=2026-10-25&to=2026-\d\d-\d\d&limit=100&text=0$/);
    const q = Object.fromEntries(new URLSearchParams(next.slice(1)));
    win = parseEarnings8kHistoryQuery(q, NOW_11_11);
    assert.ok(pages < 20, "cursor must terminate");
  }
  assert.equal(seen.size, 350);
  assert.equal(pages, 5, "350 filings at 35/day with a 100 cap => 2 days per page => 5 pages");
});

test("readEarnings8kHistoryWindow: the cap is soft by at most ONE day — a newest day larger than the cap is still returned whole so the page always makes progress; an empty window is honest (0 filings, not truncated, covered = the window)", () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-soft-"));
  seedArchiveDays(base, ["2026-11-10"], 7);
  seedArchiveDays(base, ["2026-11-09"], 2);
  const page = readEarnings8kHistoryWindow(parseEarnings8kHistoryQuery({ from: "2026-11-09", to: "2026-11-10", limit: "5" }, NOW_11_11), base);
  assert.equal(page.filings.length, 7, "the first day exceeds the cap and is still whole");
  assert.equal(page.truncated, true);
  assert.equal(page.next_to, "2026-11-09");
  assert.equal(page.malformed_lines, 0);
  // a torn line (kill mid-append) is counted, not silently dropped
  fs2.appendFileSync(path2.join(base, "earnings8k", "2026-11-10.jsonl"), '{"accession":"T-1","cik":"1","company');
  const torn = readEarnings8kHistoryWindow(parseEarnings8kHistoryQuery({ from: "2026-11-10", to: "2026-11-10" }, NOW_11_11), base);
  assert.deepEqual([torn.filings.length, torn.malformed_lines], [7, 1]);
  const empty = readEarnings8kHistoryWindow(parseEarnings8kHistoryQuery({ from: "2026-08-01", to: "2026-08-05" }, NOW_11_11), base);
  assert.deepEqual([empty.filings.length, empty.truncated, empty.next_to, empty.days_scanned], [0, false, null, 5]);
  assert.deepEqual(empty.covered, { from: "2026-08-01", to: "2026-08-05" });
  const stripped = omitEarnings8kText(mkFiling8k("S-1", "2026-11-10"));
  assert.ok(!("text" in stripped) && stripped.exhibitUrl.endsWith("ex99.htm"));
});

test("readEarnings8kHistory (legacy newest-N reader) is unchanged — the cold-cache backfill still gets the newest `limit` rows across days", () => {
  const base = fs2.mkdtempSync(path2.join(os2.tmpdir(), "vt-8k-legacy-"));
  seedArchiveDays(base, ["2026-11-09", "2026-11-10"], 4);
  const hist = readEarnings8kHistory(5, base, NOW_11_11, 3);
  assert.equal(hist.length, 3);
  assert.ok(hist.every((f) => f.filedAt === "2026-11-10"), "newest-first then sliced, exactly as before");
});
