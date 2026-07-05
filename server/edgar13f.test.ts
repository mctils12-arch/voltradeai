import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
import {
  parse13FPrimaryDoc,
  parse13FInfoTable,
  pick13FDocNames,
  normalize13FDate,
  fetchLatest13FFilings,
  archive13FFilings,
  read13FHistory,
  trimHoldings,
  FOCUSED_MAX_HOLDINGS,
} from "./edgar13f";

// ROOT VALIDATION LADDER gate 1 (DATA) fixtures — both fetched live from
// SEC EDGAR on 2026-07-04 (real accessions, not synthetic) and every field
// asserted below was hand-verified against the filed XML. The filing text
// itself, read directly, IS the external ground truth for a filings parser.

// Accession 0001762716-26-000003 — BURKETT FINANCIAL SERVICES, LLC,
// 13F-HR for period 06-30-2026: 266 entries, $319,580,122 total.
const BURKETT_PRIMARY_XML = `<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler" xmlns:com="http://www.sec.gov/edgar/common">
  <schemaVersion>X0202</schemaVersion>
  <headerData>
    <submissionType>13F-HR</submissionType>
    <filerInfo>
      <liveTestFlag>LIVE</liveTestFlag>
      <filer>
        <credentials>
          <cik>0001762716</cik>
          <ccc>XXXXXXXX</ccc>
        </credentials>
      </filer>
      <periodOfReport>06-30-2026</periodOfReport>
    </filerInfo>
  </headerData>
  <formData>
    <coverPage>
      <reportCalendarOrQuarter>06-30-2026</reportCalendarOrQuarter>
      <filingManager>
        <name>BURKETT FINANCIAL SERVICES, LLC</name>
        <address>
          <com:street1>128 EAST MAIN STREET</com:street1>
          <com:city>ROCK HILL</com:city>
          <com:stateOrCountry>SC</com:stateOrCountry>
        </address>
      </filingManager>
      <reportType>13F HOLDINGS REPORT</reportType>
      <form13FFileNumber>028-23712</form13FFileNumber>
    </coverPage>
    <summaryPage>
      <otherIncludedManagersCount>0</otherIncludedManagersCount>
      <tableEntryTotal>266</tableEntryTotal>
      <tableValueTotal>319580122</tableValueTotal>
      <isConfidentialOmitted>false</isConfidentialOmitted>
    </summaryPage>
  </formData>
</edgarSubmission>`;

// First two <infoTable> blocks of the same filing's information table
// (2026qtr2submissionapr2026.xml), verbatim including the ns1: prefixes.
const BURKETT_INFOTABLE_XML = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
	<ns1:infoTable>
		<ns1:nameOfIssuer>Schwab US Large Cap Growth ETF</ns1:nameOfIssuer>
		<ns1:titleOfClass>Equity</ns1:titleOfClass>
		<ns1:cusip>808524300</ns1:cusip>
		<ns1:value>54654995</ns1:value>
		<ns1:shrsOrPrnAmt>
			<ns1:sshPrnamt>1615100</ns1:sshPrnamt>
			<ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
		</ns1:shrsOrPrnAmt>
		<ns1:investmentDiscretion>SOLE</ns1:investmentDiscretion>
		<ns1:votingAuthority>
			<ns1:Sole>0</ns1:Sole>
			<ns1:Shared>0</ns1:Shared>
			<ns1:None>1615100</ns1:None>
		</ns1:votingAuthority>
	</ns1:infoTable>
	<ns1:infoTable>
		<ns1:nameOfIssuer>Schwab US Dividend Equity ETF</ns1:nameOfIssuer>
		<ns1:titleOfClass>Equity</ns1:titleOfClass>
		<ns1:cusip>808524797</ns1:cusip>
		<ns1:value>54175667</ns1:value>
		<ns1:shrsOrPrnAmt>
			<ns1:sshPrnamt>1708472</ns1:sshPrnamt>
			<ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
		</ns1:shrsOrPrnAmt>
		<ns1:investmentDiscretion>SOLE</ns1:investmentDiscretion>
		<ns1:votingAuthority>
			<ns1:Sole>0</ns1:Sole>
			<ns1:Shared>0</ns1:Shared>
			<ns1:None>1708472</ns1:None>
		</ns1:votingAuthority>
	</ns1:infoTable>
</ns1:informationTable>`;

// SYNTHETIC (labeled as such — no put/call filing was in the live sample):
// one real-schema block exercising the optional <putCall> path.
const PUTCALL_INFOTABLE_XML = `<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
	<ns1:infoTable>
		<ns1:nameOfIssuer>SPDR S&amp;P 500 ETF TR</ns1:nameOfIssuer>
		<ns1:titleOfClass>TR UNIT</ns1:titleOfClass>
		<ns1:cusip>78462F103</ns1:cusip>
		<ns1:value>1000000</ns1:value>
		<ns1:shrsOrPrnAmt>
			<ns1:sshPrnamt>2000</ns1:sshPrnamt>
			<ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
		</ns1:shrsOrPrnAmt>
		<ns1:putCall>Put</ns1:putCall>
		<ns1:investmentDiscretion>SOLE</ns1:investmentDiscretion>
	</ns1:infoTable>
</ns1:informationTable>`;

test("parse13FPrimaryDoc extracts manager, period, and summary totals from a real filing", () => {
  const p = parse13FPrimaryDoc(BURKETT_PRIMARY_XML);
  assert.equal(p.managerCik, "0001762716");
  assert.equal(p.managerName, "BURKETT FINANCIAL SERVICES, LLC");
  assert.equal(p.periodOfReport, "2026-06-30"); // filed as 06-30-2026
  assert.equal(p.submissionType, "13F-HR");
  assert.equal(p.entryTotal, 266);
  assert.equal(p.valueTotal, 319580122);
  assert.equal(p.otherManagersCount, 0);
});

test("parse13FInfoTable extracts namespace-prefixed holdings, hand-checked values", () => {
  const h = parse13FInfoTable(BURKETT_INFOTABLE_XML);
  assert.equal(h.length, 2);
  assert.equal(h[0].issuer, "Schwab US Large Cap Growth ETF");
  assert.equal(h[0].cusip, "808524300");
  assert.equal(h[0].value, 54654995); // full USD, post-2023 rule
  assert.equal(h[0].shares, 1615100);
  assert.equal(h[0].sharesType, "SH");
  assert.equal(h[0].discretion, "SOLE");
  assert.equal(h[0].putCall, null);
  assert.equal(h[1].cusip, "808524797");
  assert.equal(h[1].shares, 1708472);
});

test("parse13FInfoTable handles optional putCall and XML entity unescaping", () => {
  const h = parse13FInfoTable(PUTCALL_INFOTABLE_XML);
  assert.equal(h.length, 1);
  assert.equal(h[0].issuer, "SPDR S&P 500 ETF TR");
  assert.equal(h[0].putCall, "Put");
});

test("parse13FInfoTable caps entries at maxEntries", () => {
  const one = BURKETT_INFOTABLE_XML;
  assert.equal(parse13FInfoTable(one, 1).length, 1);
});

test("normalize13FDate converts EDGAR MM-DD-YYYY and passes ISO through", () => {
  assert.equal(normalize13FDate("06-30-2026"), "2026-06-30");
  assert.equal(normalize13FDate("2026-06-30"), "2026-06-30");
  assert.equal(normalize13FDate(null), null);
});

test("pick13FDocNames finds primary_doc plus filer-named info tables (both real shapes)", () => {
  // Real directory shapes from the two live filings: custom info-table names.
  const burkett = { directory: { item: [
    { name: "0001762716-26-000003-index.html" },
    { name: "2026qtr2submissionapr2026.xml" },
    { name: "primary_doc.xml" },
  ] } };
  assert.deepEqual(pick13FDocNames(burkett), { primary: "primary_doc.xml", info: "2026qtr2submissionapr2026.xml" });
  const atmos = { directory: { item: [
    { name: "13F_ATMOS_6_30_26.xml" },
    { name: "primary_doc.xml" },
  ] } };
  assert.deepEqual(pick13FDocNames(atmos), { primary: "primary_doc.xml", info: "13F_ATMOS_6_30_26.xml" });
});

// ── Focused-manager cap: filings over FOCUSED_MAX_HOLDINGS archive as
// summary-only records, and the info table is never even fetched. ─────────

const FEED_ATOM = `<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
<title>13F-HR - BURKETT FINANCIAL SERVICES, LLC (0001762716) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1762716/000176271626000003/0001762716-26-000003-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-07-02 &lt;b&gt;AccNo:&lt;/b&gt; 0001762716-26-000003 &lt;b&gt;Size:&lt;/b&gt; 160 KB
</summary>
</entry>
</feed>`;

function mockFetchFor(bodies: Record<string, string>) {
  const requested: string[] = [];
  const impl = async (url: string) => {
    requested.push(url);
    for (const [suffix, body] of Object.entries(bodies)) {
      if (url.includes(suffix)) return { ok: true, status: 200, text: async () => body };
    }
    return { ok: false, status: 404, text: async () => "" };
  };
  return { impl, requested };
}

test("fetchLatest13FFilings: focused filing gets its full table, oversized filing is summary-only", async () => {
  // BURKETT reports 266 entries — over the 250 cap — so holdings must be
  // omitted AND the info-table document must never be requested.
  assert.ok(266 > FOCUSED_MAX_HOLDINGS);
  const over = mockFetchFor({
    "output=atom": FEED_ATOM,
    "index.json": JSON.stringify({ directory: { item: [
      { name: "primary_doc.xml" }, { name: "2026qtr2submissionapr2026.xml" },
    ] } }),
    "primary_doc.xml": BURKETT_PRIMARY_XML,
    "2026qtr2submissionapr2026.xml": BURKETT_INFOTABLE_XML,
  });
  const filings = await fetchLatest13FFilings(5, over.impl as any, 0);
  assert.equal(filings.length, 1);
  assert.equal(filings[0].accession, "0001762716-26-000003");
  assert.equal(filings[0].holdingsOmitted, true);
  assert.equal(filings[0].holdings, null);
  assert.ok(!over.requested.some((u) => u.includes("2026qtr2submissionapr2026.xml")),
    "info table must not be fetched for an over-cap filing");

  // Same filing with a focused-size summary → full table archived.
  const focused = mockFetchFor({
    "output=atom": FEED_ATOM,
    "index.json": JSON.stringify({ directory: { item: [
      { name: "primary_doc.xml" }, { name: "2026qtr2submissionapr2026.xml" },
    ] } }),
    "primary_doc.xml": BURKETT_PRIMARY_XML.replace("<tableEntryTotal>266<", "<tableEntryTotal>2<"),
    "2026qtr2submissionapr2026.xml": BURKETT_INFOTABLE_XML,
  });
  const f2 = await fetchLatest13FFilings(5, focused.impl as any, 0);
  assert.equal(f2[0].holdingsOmitted, false);
  assert.equal(f2[0].holdings?.length, 2);
  assert.equal(f2[0].holdings?.[0].cusip, "808524300");
});

test("archive13FFilings + read13FHistory round-trip with accession dedup", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt13f-"));
  const filing = {
    ...parse13FPrimaryDoc(BURKETT_PRIMARY_XML),
    accession: "0001762716-26-000003",
    filedAt: "2026-07-02",
    cik: "0001762716",
    indexUrl: "https://www.sec.gov/x",
    holdings: parse13FInfoTable(BURKETT_INFOTABLE_XML),
    holdingsOmitted: false,
  };
  assert.equal(archive13FFilings([filing as any], dir), 1);
  assert.equal(archive13FFilings([filing as any], dir), 0, "same accession never archives twice");
  const hist = read13FHistory(3, dir);
  assert.equal(hist.length, 1);
  assert.equal(hist[0].managerName, "BURKETT FINANCIAL SERVICES, LLC");
  assert.equal(hist[0].holdings?.length, 2);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("trimHoldings keeps the top-N rows by value", () => {
  const mk = (v: number) => ({ issuer: `X${v}`, titleOfClass: null, cusip: null, value: v, shares: null, sharesType: null, putCall: null, discretion: null });
  const f: any = { holdings: [mk(1), mk(9), mk(5)], holdingsOmitted: false };
  const t = trimHoldings(f, 2);
  assert.deepEqual(t.holdings.map((h: any) => h.value), [9, 5]);
  assert.equal(f.holdings.length, 3, "original untouched");
});

// ── Wiring pins: route + boot registered; manifest present ─────────────────

test("routes.ts boots the 13F poll and registers the API routes; manifest exists", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("boot13FPoll()"), "13F poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/filings13f"'), "live route registered");
  assert.ok(routes.includes('"/api/data/filings13f/history"'), "history route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "filings13f.json"), "utf8"));
  assert.equal(manifest.stream, "filings13f");
  assert.ok(String(manifest.license).toLowerCase().includes("public"), "license line required");
  assert.ok(JSON.stringify(manifest).includes("250"), "focused-manager cap must be stated in the manifest, never silent");
});
