import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseForm4Xml,
  parseFilingFeed,
  pickOwnershipXmlName,
  classifyTransactionCode,
} from "./edgarForm4";

// ROOT VALIDATION LADDER gate 1 (DATA) fixtures — both fetched live from
// SEC EDGAR on 2026-07-03 (real accessions, not synthetic) and every field
// below was hand-verified against the filed XML before being asserted here.
// This IS the external ground truth for a filings parser: the filing text
// itself, read directly, is authoritative — there is no separate "official"
// source to check a Form 4 against.

// Accession 0001104659-26-080497 — Richard Christian M / CYPHERPUNK
// TECHNOLOGIES INC., a derivative-only RSU grant (transaction code A).
const RSU_GRANT_XML = `<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0609</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-07-01</periodOfReport>
    <notSubjectToSection16>0</notSubjectToSection16>
    <issuer>
        <issuerCik>0001509745</issuerCik>
        <issuerName>CYPHERPUNK TECHNOLOGIES INC.</issuerName>
        <issuerTradingSymbol>CYPH</issuerTradingSymbol>
        <issuerForeignTradingSymbol></issuerForeignTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001645967</rptOwnerCik>
            <rptOwnerName>Richard Christian M</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle></officerTitle>
            <otherText></otherText>
        </reportingOwnerRelationship>
    </reportingOwner>
    <aff10b5One>0</aff10b5One>
    <derivativeTable>
        <derivativeTransaction>
            <securityTitle><value>Restricted Stock Units</value></securityTitle>
            <conversionOrExercisePrice><value>0</value><footnoteId id="F1"/></conversionOrExercisePrice>
            <transactionDate><value>2026-07-01</value></transactionDate>
            <deemedExecutionDate></deemedExecutionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>A</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionTimeliness></transactionTimeliness>
            <transactionAmounts>
                <transactionShares><value>75000</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <exerciseDate><footnoteId id="F2"/></exerciseDate>
            <expirationDate><footnoteId id="F2"/></expirationDate>
            <underlyingSecurity>
                <underlyingSecurityTitle><value>Common Stock</value></underlyingSecurityTitle>
                <underlyingSecurityShares><value>75000</value></underlyingSecurityShares>
            </underlyingSecurity>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>75000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </derivativeTransaction>
    </derivativeTable>
</ownershipDocument>`;

// Accession 0000902664-26-003001 — Oasis Management / STRATUS PROPERTIES INC,
// 3 reporting owners, non-derivative-only, two open-market SALES (code S).
const MULTI_OWNER_SALE_XML = `<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0609</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-06-30</periodOfReport>
    <notSubjectToSection16>0</notSubjectToSection16>
    <issuer>
        <issuerCik>0000885508</issuerCik>
        <issuerName>STRATUS PROPERTIES INC</issuerName>
        <issuerTradingSymbol>STRS</issuerTradingSymbol>
        <issuerForeignTradingSymbol></issuerForeignTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001317904</rptOwnerCik>
            <rptOwnerName>Oasis Management Co Ltd.</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>1</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle></officerTitle>
            <otherText></otherText>
        </reportingOwnerRelationship>
    </reportingOwner>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001530803</rptOwnerCik>
            <rptOwnerName>Oasis Investments II Master Fund Ltd.</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>1</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle></officerTitle>
            <otherText></otherText>
        </reportingOwnerRelationship>
    </reportingOwner>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001456474</rptOwnerCik>
            <rptOwnerName>Fischer Seth</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>1</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle></officerTitle>
            <otherText></otherText>
        </reportingOwnerRelationship>
    </reportingOwner>
    <aff10b5One>0</aff10b5One>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock, par value $0.01 per share</value></securityTitle>
            <transactionDate><value>2026-06-30</value></transactionDate>
            <deemedExecutionDate></deemedExecutionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>S</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionTimeliness></transactionTimeliness>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>28.90</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>961129</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>I</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock, par value $0.01 per share</value></securityTitle>
            <transactionDate><value>2026-07-02</value></transactionDate>
            <deemedExecutionDate></deemedExecutionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>S</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionTimeliness></transactionTimeliness>
            <transactionAmounts>
                <transactionShares><value>117612</value></transactionShares>
                <transactionPricePerShare><value>28.0339</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>843517</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>I</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>`;

test("gate 1: derivative RSU grant filing parses to the hand-verified values", () => {
  const p = parseForm4Xml(RSU_GRANT_XML);
  assert.equal(p.issuerCik, "0001509745");
  assert.equal(p.issuerName, "CYPHERPUNK TECHNOLOGIES INC.");
  assert.equal(p.issuerTradingSymbol, "CYPH");
  assert.equal(p.periodOfReport, "2026-07-01");
  assert.equal(p.owners.length, 1);
  assert.deepEqual(p.owners[0], {
    cik: "0001645967",
    name: "Richard Christian M",
    isDirector: true,
    isOfficer: false,
    isTenPercentOwner: false,
    officerTitle: null,
  });
  assert.equal(p.transactions.length, 1);
  const t = p.transactions[0];
  assert.equal(t.table, "derivative");
  assert.equal(t.securityTitle, "Restricted Stock Units");
  assert.equal(t.transactionDate, "2026-07-01");
  assert.equal(t.transactionCode, "A");
  assert.equal(t.kind, "award_grant");
  assert.equal(t.shares, 75000);
  assert.equal(t.pricePerShare, 0);
  assert.equal(t.acquiredDisposedCode, "D");
  assert.equal(t.sharesOwnedAfter, 75000);
});

test("gate 1: multi-owner non-derivative sale filing parses to the hand-verified values", () => {
  const p = parseForm4Xml(MULTI_OWNER_SALE_XML);
  assert.equal(p.issuerName, "STRATUS PROPERTIES INC");
  assert.equal(p.issuerTradingSymbol, "STRS");
  assert.equal(p.owners.length, 3);
  assert.equal(p.owners[0].name, "Oasis Management Co Ltd.");
  assert.equal(p.owners[1].name, "Oasis Investments II Master Fund Ltd.");
  assert.equal(p.owners[2].name, "Fischer Seth");
  assert.ok(p.owners.every((o) => o.isTenPercentOwner && !o.isDirector && !o.isOfficer));
  assert.equal(p.transactions.length, 2);
  assert.equal(p.transactions[0].table, "nonDerivative");
  assert.equal(p.transactions[0].transactionCode, "S");
  assert.equal(p.transactions[0].kind, "open_market_sale");
  assert.equal(p.transactions[0].shares, 10000);
  assert.equal(p.transactions[0].pricePerShare, 28.90);
  assert.equal(p.transactions[0].sharesOwnedAfter, 961129);
  assert.equal(p.transactions[1].shares, 117612);
  assert.equal(p.transactions[1].pricePerShare, 28.0339);
  assert.equal(p.transactions[1].sharesOwnedAfter, 843517);
});

test("classifyTransactionCode maps SEC codes to kinds (P is the informative insider-buy case)", () => {
  assert.equal(classifyTransactionCode("P"), "open_market_buy");
  assert.equal(classifyTransactionCode("S"), "open_market_sale");
  assert.equal(classifyTransactionCode("A"), "award_grant");
  assert.equal(classifyTransactionCode("M"), "option_exercise");
  assert.equal(classifyTransactionCode("G"), "gift");
  assert.equal(classifyTransactionCode("F"), "tax_withholding");
  assert.equal(classifyTransactionCode("Z"), "other");
  assert.equal(classifyTransactionCode(null), "other");
});

// Real (trimmed) snippet of the EDGAR getcurrent atom feed, fetched
// 2026-07-03 — CYPHERPUNK TECHNOLOGIES appears twice for the SAME accession
// (once as "Issuer", once would be as "Reporting" for the filer) plus one
// distinct accession, proving the dedup-by-accession logic against the real
// feed shape rather than an invented one.
const SAMPLE_FEED_ATOM = `<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Latest Filings - Fri, 03 Jul 2026 20:05:55 EDT</title>
<entry>
<title>4 - Richard Christian M (0001645967) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1645967/000110465926080497/0001104659-26-080497-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-07-02 &lt;b&gt;AccNo:&lt;/b&gt; 0001104659-26-080497 &lt;b&gt;Size:&lt;/b&gt; 6 KB
</summary>
<updated>2026-07-02T21:59:00-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
<id>urn:tag:sec.gov,2008:accession-number=0001104659-26-080497</id>
</entry>
<entry>
<title>4 - CYPHERPUNK TECHNOLOGIES INC. (0001509745) (Issuer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1509745/000110465926080497/0001104659-26-080497-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-07-02 &lt;b&gt;AccNo:&lt;/b&gt; 0001104659-26-080497 &lt;b&gt;Size:&lt;/b&gt; 6 KB
</summary>
<updated>2026-07-02T21:59:00-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
<id>urn:tag:sec.gov,2008:accession-number=0001104659-26-080497</id>
</entry>
<entry>
<title>4 - Oei Khing Djien (0002092424) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/2092424/000110465926080496/0001104659-26-080496-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-07-02 &lt;b&gt;AccNo:&lt;/b&gt; 0001104659-26-080496 &lt;b&gt;Size:&lt;/b&gt; 6 KB
</summary>
<updated>2026-07-02T21:58:28-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
<id>urn:tag:sec.gov,2008:accession-number=0001104659-26-080496</id>
</entry>
</feed>`;

test("parseFilingFeed dedupes the two filer/issuer entries the real feed emits per accession", () => {
  const entries = parseFilingFeed(SAMPLE_FEED_ATOM);
  assert.equal(entries.length, 2, "expected 2 unique accessions from 3 raw entries");
  assert.equal(entries[0].accession, "0001104659-26-080497");
  assert.equal(entries[0].filedAt, "2026-07-02");
  assert.equal(entries[1].accession, "0001104659-26-080496");
});

test("pickOwnershipXmlName finds the ownership XML in a real index.json shape, skipping non-xml items", () => {
  const indexJson = {
    directory: {
      item: [
        { name: "0000902664-26-003001-index-headers.html", type: "text.gif" },
        { name: "0000902664-26-003001-index.html", type: "text.gif" },
        { name: "0000902664-26-003001.txt", type: "text.gif" },
        { name: "ownership.xml", type: "text.gif", size: "8978" },
      ],
      name: "/Archives/edgar/data/1317904/000090266426003001",
    },
  };
  assert.equal(pickOwnershipXmlName(indexJson), "ownership.xml");
});

test("pickOwnershipXmlName returns null when a filing has no XML document", () => {
  const indexJson = { directory: { item: [{ name: "0001-index.html", type: "text.gif" }] } };
  assert.equal(pickOwnershipXmlName(indexJson), null);
});
