import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { unescapeXmlText, unescapeXmlTextOrNull } from "./xmlText";
import { parseForm4Xml, parseFilingFeed, cleanForm4Filing, readFilingHistory, form4HistoryPage, type Form4Filing } from "./edgarForm4";
import { parse8KFeed, cleanEarnings8K, readEarnings8kHistory, readEarnings8kHistoryWindow, type Earnings8K } from "./sec8kEarnings";

// open_questions.md (ph7), 2026-11-26: /data/earnings showed "DEERE &amp; CO".
// The strings below are the ones production served on 2026-11-27 00:25Z
// (30 days of /api/data/earnings-language/history and /api/data/insider/history).

test("the five predefined entities and numeric references are unescaped", () => {
  assert.equal(unescapeXmlText("DEERE &amp; CO"), "DEERE & CO");
  assert.equal(unescapeXmlText("MERRILL LYNCH, PIERCE, FENNER &amp; SMITH INC."), "MERRILL LYNCH, PIERCE, FENNER & SMITH INC.");
  assert.equal(unescapeXmlText("Ordinary shares, no par value (&quot;Ordinary Shares&quot;)"), 'Ordinary shares, no par value ("Ordinary Shares")');
  assert.equal(unescapeXmlText("O&apos;Reilly &lt;A&gt;"), "O'Reilly <A>");
  assert.equal(unescapeXmlText("Moody&#39;s &#x26; Co &#233;"), "Moody's & Co é");
});

test("one pass only: an escaped entity becomes the entity's text, not its character", () => {
  assert.equal(unescapeXmlText("a &amp;lt; b"), "a &lt; b");
  assert.equal(unescapeXmlText("&amp;amp;"), "&amp;");
});

test("what XML does not define is left as written", () => {
  assert.equal(unescapeXmlText("AT&T; R&D"), "AT&T; R&D");
  assert.equal(unescapeXmlText("A&nbsp;B"), "A&nbsp;B");
  assert.equal(unescapeXmlText("bad &#0; &#xD800; &#1114112;"), "bad &#0; &#xD800; &#1114112;");
  assert.equal(unescapeXmlText("no entity here"), "no entity here");
});

test("an absent field stays absent", () => {
  assert.equal(unescapeXmlTextOrNull(null), null);
  assert.equal(unescapeXmlTextOrNull(undefined), undefined);
  assert.equal(unescapeXmlTextOrNull("A &amp; B"), "A & B");
});

const FORM4_XML = `<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2026-11-20</periodOfReport>
  <issuer>
    <issuerCik>0000810958</issuerCik>
    <issuerName>CITIZENS &amp; NORTHERN CORP</issuerName>
    <issuerTradingSymbol>CZNC</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0000000001</rptOwnerCik>
      <rptOwnerName>MERRILL LYNCH, PIERCE, FENNER &amp; SMITH INC.</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>President &amp; CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Ordinary shares, no par value (&quot;Ordinary Shares&quot;)</value></securityTitle>
      <transactionDate><value>2026-11-20</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>20.5</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>`;

test("Form 4: names, the officer title and the security title are parsed unescaped; codes and numbers are as before", () => {
  const p = parseForm4Xml(FORM4_XML);
  assert.equal(p.issuerName, "CITIZENS & NORTHERN CORP");
  assert.equal(p.issuerTradingSymbol, "CZNC");
  assert.equal(p.owners[0].name, "MERRILL LYNCH, PIERCE, FENNER & SMITH INC.");
  assert.equal(p.owners[0].officerTitle, "President & CEO");
  assert.equal(p.transactions[0].securityTitle, 'Ordinary shares, no par value ("Ordinary Shares")');
  assert.equal(p.transactions[0].transactionCode, "P");
  assert.equal(p.transactions[0].shares, 100);
  assert.equal(p.transactions[0].pricePerShare, 20.5);
  assert.equal(p.issuerCik, "0000810958");
});

const entry = (form: string, name: string, cik: string, acc: string, items = "") => `<entry>
<title>${form} - ${name} (${cik}) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/${Number(cik)}/${acc.replace(/-/g, "")}/${acc}-index.htm"/>
<summary type="html">
 &lt;b&gt;Filed:&lt;/b&gt; 2026-11-25 &lt;b&gt;AccNo:&lt;/b&gt; ${acc} &lt;b&gt;Size:&lt;/b&gt; 1 MB${items}
</summary>
<updated>2026-11-25T06:45:10-05:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="${form}"/>
<id>urn:tag:sec.gov,2008:accession-number=${acc}</id>
</entry>`;
const feed = (...entries: string[]) => `<?xml version="1.0" encoding="ISO-8859-1" ?>\n<feed xmlns="http://www.w3.org/2005/Atom">\n<title>Latest Filings</title>\n${entries.join("\n")}\n</feed>`;

test("8-K feed: the company name is the title's text, unescaped; the CIK and accession are as before", () => {
  const rows = parse8KFeed(feed(entry("8-K", "DEERE &amp; CO", "0000315189", "0000315189-26-000070", "\n&lt;br&gt;Item 2.02: Results of Operations and Financial Condition")));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].companyName, "DEERE & CO");
  assert.equal(rows[0].cik, "0000315189");
  assert.equal(rows[0].accession, "0000315189-26-000070");
  assert.deepEqual(rows[0].itemCodes, ["2.02"]);
});

test("Form 4 feed: the title is unescaped; the CIK is still read from it", () => {
  const rows = parseFilingFeed(feed(entry("4", "CITIZENS &amp; NORTHERN CORP", "0000810958", "0000810958-26-000099")));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].title, "4 - CITIZENS & NORTHERN CORP (0000810958) (Filer)");
  assert.equal(rows[0].cik, "0000810958");
  assert.equal(rows[0].form, "4");
});

// ── rows archived before the fix hold the escaped form; every reader returns them clean ──

const NOW = Date.parse("2026-11-26T12:00:00Z");
const OLD_FORM4: Form4Filing = {
  accession: "0000810958-26-000099", filedAt: "2026-11-25", cik: "0000810958",
  indexUrl: "https://www.sec.gov/Archives/edgar/data/810958/000081095826000099/0000810958-26-000099-index.htm",
  issuerCik: "0000810958", issuerName: "CITIZENS &amp; NORTHERN CORP", issuerTradingSymbol: "CZNC", periodOfReport: "2026-11-20",
  owners: [{ cik: "1", name: "MERRILL LYNCH, PIERCE, FENNER &amp; SMITH INC.", isDirector: false, isOfficer: true, isTenPercentOwner: false, officerTitle: "President &amp; CEO" }],
  transactions: [{ table: "nonDerivative", securityTitle: "Common (&quot;Shares&quot;)", transactionDate: "2026-11-20", transactionCode: "P", kind: "open_market_buy", shares: 100, pricePerShare: 20.5, acquiredDisposedCode: "A", sharesOwnedAfter: null }],
};
const CLEAN_FORM4: Form4Filing = { ...OLD_FORM4, accession: "0000000002-26-000001", issuerName: "PLAIN CORP", owners: [{ ...OLD_FORM4.owners[0], name: "DOE JANE", officerTitle: null }], transactions: [] };
const OLD_8K: Earnings8K = {
  accession: "0000315189-26-000070", cik: "0000315189", companyName: "DEERE &amp; CO", filedAt: "2026-11-25", acceptanceDatetime: null,
  ticker: "DE", itemCodes: ["2.02"], indexUrl: "https://www.sec.gov/x-index.htm", exhibitUrl: "https://www.sec.gov/x.htm",
  text: "Deere &amp; Company reported", textLength: 27, truncated: false,
  coRegistrants: [{ cik: "0000027673", companyName: "JOHN DEERE CAPITAL &amp; CREDIT CORP", ticker: null }],
};

function archiveDir(sub: string, files: Record<string, { rows: unknown[]; gz?: boolean }>): string {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "xmltext-"));
  fs.mkdirSync(path.join(base, sub), { recursive: true });
  for (const [day, f] of Object.entries(files)) {
    const body = f.rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
    if (f.gz) fs.writeFileSync(path.join(base, sub, `${day}.jsonl.gz`), zlib.gzipSync(body));
    else fs.writeFileSync(path.join(base, sub, `${day}.jsonl`), body);
  }
  return base;
}

test("cleanForm4Filing: every name field of an old row; a clean row is the same object", () => {
  const c = cleanForm4Filing(OLD_FORM4);
  assert.equal(c.issuerName, "CITIZENS & NORTHERN CORP");
  assert.equal(c.owners[0].name, "MERRILL LYNCH, PIERCE, FENNER & SMITH INC.");
  assert.equal(c.owners[0].officerTitle, "President & CEO");
  assert.equal(c.transactions[0].securityTitle, 'Common ("Shares")');
  assert.equal(c.transactions[0].shares, 100);
  assert.equal(OLD_FORM4.issuerName, "CITIZENS &amp; NORTHERN CORP", "the row handed in is not changed");
  assert.equal(cleanForm4Filing(CLEAN_FORM4), CLEAN_FORM4);
  assert.equal(cleanForm4Filing(c), c, "a second pass changes nothing");
});

test("Form 4 archive readers return old rows unescaped, plain and compressed days alike", () => {
  const base = archiveDir("filings", {
    "2026-11-26": { rows: [CLEAN_FORM4] },
    "2026-11-25": { rows: [OLD_FORM4], gz: true },
  });
  try {
    const hist = readFilingHistory(5, base, NOW);
    assert.equal(hist.length, 2);
    assert.ok(!JSON.stringify(hist).includes("&amp;") && !JSON.stringify(hist).includes("&quot;"));
    const page = form4HistoryPage({ from: "2026-11-24", to: "2026-11-26" }, base, NOW, null);
    assert.equal(page.count, 2);
    const old = page.filings.find((f) => f.accession === OLD_FORM4.accession) as Form4Filing;
    assert.equal(old.issuerName, "CITIZENS & NORTHERN CORP");
    assert.equal(old.owners[0].officerTitle, "President & CEO");
    assert.ok(!JSON.stringify(page).includes("&amp;"));
    // the day file itself is not rewritten
    const onDisk = zlib.gunzipSync(fs.readFileSync(path.join(base, "filings", "2026-11-25.jsonl.gz"))).toString("utf8");
    assert.ok(onDisk.includes("CITIZENS &amp; NORTHERN CORP"));
  } finally { fs.rmSync(base, { recursive: true, force: true }); }
});

test("cleanEarnings8K: the registrant names; the exhibit text is not touched", () => {
  const c = cleanEarnings8K(OLD_8K);
  assert.equal(c.companyName, "DEERE & CO");
  assert.equal(c.coRegistrants?.[0].companyName, "JOHN DEERE CAPITAL & CREDIT CORP");
  assert.equal(c.text, OLD_8K.text);
  assert.equal(cleanEarnings8K(c), c);
});

test("8-K archive readers return old rows unescaped", () => {
  const base = archiveDir("earnings8k", { "2026-11-25": { rows: [OLD_8K], gz: true } });
  try {
    const hist = readEarnings8kHistory(5, base, NOW);
    assert.equal(hist.length, 1);
    assert.equal(hist[0].companyName, "DEERE & CO");
    const page = readEarnings8kHistoryWindow({ from: "2026-11-24", to: "2026-11-26", maxFilings: 500, includeText: false }, base);
    assert.equal(page.filings.length, 1);
    assert.equal(page.filings[0].companyName, "DEERE & CO");
    assert.equal((page.filings[0] as Earnings8K).coRegistrants?.[0].companyName, "JOHN DEERE CAPITAL & CREDIT CORP");
  } finally { fs.rmSync(base, { recursive: true, force: true }); }
});
