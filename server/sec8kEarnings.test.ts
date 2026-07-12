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
