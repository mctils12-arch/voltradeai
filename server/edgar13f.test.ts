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
  refresh13FCache,
  latest13FFilings,
  _reset13FCacheForTests,
  type Filing13F,
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

// ── Cold-cache-no-disk-backfill (2026-09-16 — cold-cache-no-disk-backfill
// audit; mirrors sec8kEarnings.test.ts's identically-shaped cold-cache
// tests, same DATA_DIR/archiveBaseDir() resolution convention) ────────────

const mkFiling13F = (acc: string, filedAt: string): Filing13F => ({
  managerCik: "0001762716", managerName: "TEST MANAGER LLC",
  periodOfReport: "2026-06-30", submissionType: "13F-HR",
  entryTotal: 1, valueTotal: 1000, otherManagersCount: 0,
  accession: acc, filedAt, cik: "0001762716",
  indexUrl: `https://www.sec.gov/x/${acc}/`,
  holdings: null, holdingsOmitted: false,
});

test("refresh13FCache: cold cache backfills from the on-disk filings13f archive when the live poll throws — the same cold-cache-no-disk-backfill class edgarForm4.ts/sec8kEarnings.ts already closed, found unfixed here by the 2026-09-15 module audit (a redeploy/SEC-EDGAR-outage must not report empty over real archived filings, and must not silently cache an empty list either)", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-13f-coldcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _reset13FCacheForTests();
  try {
    // filings13fDir(undefined) resolves through archiveBaseDir() ->
    // DATA_DIR/datacore_archive/filings13f — mirrored here so the archived
    // day lands where refresh13FCache's own baseDir-less backfill path will
    // actually look for it. Dated "today" (real wall clock) since
    // refresh13FCache/backfill13FFromArchive take no injectable nowMs and
    // the default 5-day lookback is measured from Date.now().
    const dir = path.join(base, "datacore_archive", "filings13f");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(mkFiling13F("COLD-1", today)) + "\n");

    assert.equal(latest13FFilings(), null, "cache must still be cold going into this cycle");
    const origFetch = global.fetch;
    global.fetch = (async () => { throw new Error("SEC EDGAR unreachable"); }) as any;
    try {
      await refresh13FCache(40);
    } finally {
      global.fetch = origFetch;
    }
    const cached = latest13FFilings();
    assert.ok(cached, "cache must be populated, not left null, despite the live poll throwing");
    assert.equal(cached!.filings.length, 1);
    assert.equal(cached!.filings[0].accession, "COLD-1", "backfilled from the archived filing, not fabricated");
  } finally {
    _reset13FCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refresh13FCache: an empty-but-non-throwing live poll also backfills from disk when the cache is cold (not just the throw path)", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-13f-coldcache-empty-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _reset13FCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "filings13f");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(mkFiling13F("COLD-2", today)) + "\n");

    // An empty (no new filings this cycle), NON-throwing atom feed —
    // fetchLatest13FFilings resolves to [] rather than throwing.
    const origFetch = global.fetch;
    global.fetch = (async () => ({
      ok: true, status: 200,
      text: async () => `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>`,
    })) as any;
    try {
      await refresh13FCache(40);
    } finally {
      global.fetch = origFetch;
    }
    const cached = latest13FFilings();
    assert.ok(cached, "cache must be populated from disk, not left null, on an empty-but-successful poll");
    assert.equal(cached!.filings.length, 1);
    assert.equal(cached!.filings[0].accession, "COLD-2");
  } finally {
    _reset13FCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
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

// ── Windowed + paged history (2026-11-12) — the honest reader behind
// /api/data/filings13f/history and /api/v1/data/13f-holdings-history ────────
import {
  filings13fHistoryPage, filings13fHistoryArchive,
  FILINGS13F_ARCHIVE_START, FILINGS13F_HISTORY_MAX_FILINGS, FILINGS13F_HISTORY_MAX_DAYS,
} from "./edgar13f";

test("filings13fHistoryPage: REGRESSION (live 2026-11-11, Q3 season, both the preview route and the SOLD v1 mirror) — `?days=120` over a season-heavy archive is a WHOLE-days page that says truncated/covered/next, never 500 filings spanning two days labelled 120; holdings are trimmed per route (25 RAW display vs FOCUSED_MAX_HOLDINGS keyed) on archived AND live rows", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt13f-page-"));
  const dir = path.join(base, "filings13f");
  fs.mkdirSync(dir, { recursive: true });
  const t = Date.UTC(2026, 10, 11, 13, 0, 0); // 2026-11-11T13:00Z
  const withHoldings = (f: Filing13F, n: number): Filing13F => ({
    ...f, holdings: Array.from({ length: n }, (_, i) => ({ issuer: `ISSUER ${i}`, titleOfClass: "COM", cusip: `${String(i).padStart(9, "0")}`, value: n - i, shares: 1, shareType: "SH", putCall: null, discretion: "SOLE", votingSole: 1, votingShared: 0, votingNone: 0 } as any)),
  });
  // production's own shape on 2026-11-11: 11-10 400 rows, 11-09 well over the 100 the old reader cut it to
  const counts: Record<string, number> = { "2026-11-10": 400, "2026-11-09": 300, "2026-11-06": 50 };
  for (const [day, n] of Object.entries(counts)) {
    fs.writeFileSync(path.join(dir, `${day}.jsonl`),
      Array.from({ length: n }, (_, i) => JSON.stringify(withHoldings(mkFiling13F(`${day}-${i}`, `${day}T16:00:00`), 40))).join("\n") + "\n");
  }
  const a = filings13fHistoryArchive(25, base);
  assert.deepEqual([a.dir, a.archiveStart, a.maxFilings, a.maxDays], [dir, FILINGS13F_ARCHIVE_START, FILINGS13F_HISTORY_MAX_FILINGS, FILINGS13F_HISTORY_MAX_DAYS]);
  assert.deepEqual([FILINGS13F_HISTORY_MAX_FILINGS, FILINGS13F_HISTORY_MAX_DAYS], [500, 120], "the page cap IS read13FHistory's historical 500 and the routes' 120-day bound — reported, not changed");
  const raw = filings13fHistoryPage({ days: "120" }, 25, base, t, []);
  assert.equal(raw.days, 120);
  assert.deepEqual([raw.count, raw.truncated, raw.covered, raw.next], [400, true, { from: "2026-11-10", to: "2026-11-11" }, "?from=2026-07-15&to=2026-11-09"],
    "11-10's 400 fit; 11-09's 300 would make 700 > 500 -> the page stops at the day boundary and SAYS so");
  assert.ok(raw.filings.every((f) => f.holdings?.length === 25), "RAW route trim: 25 holdings per filing");
  assert.equal(raw.archive_start, "2026-07-05");
  const keyed = filings13fHistoryPage({ days: "120" }, FOCUSED_MAX_HOLDINGS, base, t, []);
  assert.ok(keyed.filings.every((f) => f.holdings?.length === 40), "keyed mirror keeps the full as-filed table (40 < FOCUSED_MAX_HOLDINGS)");
  // the old reader returned 500 rows (400 + the first 100 of 11-09) labelled days=120 with no marker
  assert.equal(read13FHistory(120, base, t).length, 500, "the legacy reader still caps at 500 (cold-cache backfill keeps using it)");
  // the cursor walk reaches the oldest day
  const next = filings13fHistoryPage(Object.fromEntries(new URLSearchParams(raw.next!.slice(1))), 25, base, t, []);
  assert.deepEqual([next.count, next.truncated, next.covered.from, next.days], [350, false, "2026-07-15", null]);
  // live merge only on a page that ends today, deduped by accession, trimmed like the archived rows
  const live = [withHoldings(mkFiling13F("2026-11-10-0", "2026-11-10T16:00:00"), 60), withHoldings(mkFiling13F("LIVE-1", "2026-11-11T12:30:00"), 60)];
  const today = filings13fHistoryPage({ days: "2" }, 25, base, t, live);
  assert.equal(today.count, 401, "one live row added, the already-archived one dropped");
  assert.equal(today.filings.find((f) => f.accession === "LIVE-1")?.holdings?.length, 25, "live rows pass through the same trim");
  assert.equal(filings13fHistoryPage({ from: "2026-11-01", to: "2026-11-10" }, 25, base, t, live).filings.some((f) => f.accession === "LIVE-1"), false);
  // the archive writer and the page agree on the day file (round-trip through the real writer)
  const base2 = fs.mkdtempSync(path.join(os.tmpdir(), "vt13f-page-w-"));
  assert.equal(archive13FFilings([mkFiling13F("W-1", "2026-11-11T09:00:00")], base2, t), 1);
  assert.equal(filings13fHistoryPage({ days: "1" }, 25, base2, t, []).count, 1);
  fs.rmSync(base, { recursive: true, force: true });
  fs.rmSync(base2, { recursive: true, force: true });
});

// ── [PIPELINE 2026-11-24, open_questions.md (ai3)] The poll counts what it
// could not read. Before: every per-filing error was caught and dropped, and a
// poll that read nothing was an ok poll with 0 rows. ─────────────────────────
import { _resetEdgarPollHealthForTests, edgarPollHealthStatus, pollDocuments } from "./edgarPollHealth";
import { stageOf13FUrl } from "./edgar13f";

const stream13f = (status: Record<string, unknown>) =>
  (status.streams as Array<Record<string, any>>).find((s) => s.stream === "filings13f")!;

// [PIPELINE 2026-11-24, (ai2)] Since the submission-text fallback, a 404 listing
// is no longer the end of a filing: the reader asks for the submission text
// next. This test's EDGAR answers 404 for that too, so the filing is still
// failed and the poll still a failed poll; the failure named is the last
// request made, the submission text.
test("13F poll: a filing whose listing AND submission text are unpublished is counted failed with what was asked and the status, and the poll is a failed poll", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-13f-unread-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _reset13FCacheForTests();
  _resetEdgarPollHealthForTests();
  try {
    // the feed lists the filing; its index.json answers 404 (EDGAR's "Directory
    // listing is Not Available Yet", measured for Form 4 on 2026-11-23)
    const edgar = mockFetchFor({ "output=atom": FEED_ATOM });
    await refresh13FCache(40, edgar.impl as any, 0);
    assert.deepEqual({ ...pollDocuments("filings13f"), at: null },
      { at: null, candidates: 1, returned: 0, skipped: 0, failed: 1, first_error: "submission text -> 404", via_submission: 0 });
    assert.deepEqual(edgar.requested.filter((u) => !u.includes("output=atom")).map((u) => u.replace(/^.*\//, "")),
      ["index.json", "0001762716-26-000003.txt"], "the listing, then the submission text, and nothing else");
    const r = stream13f(edgarPollHealthStatus(base));
    assert.equal(r.polls_ok, 0);
    assert.equal(r.polls_failed, 1);
    assert.equal(r.consecutive_failures, 1);
    assert.equal(r.last_error, "0 of 1 13F filings could be read: submission text -> 404");
    assert.equal(r.documents.first_error, "submission text -> 404");
    assert.ok(!JSON.stringify(r).includes("0001762716"), "the probe names no filing");
  } finally {
    _reset13FCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("13F poll: a failed information table is named by its kind, never by the filer's file name; a read filing and an empty page are ok polls", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-13f-docs-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _reset13FCacheForTests();
  _resetEdgarPollHealthForTests();
  try {
    const listing = JSON.stringify({ directory: { item: [{ name: "primary_doc.xml" }, { name: "13F_BURKETT_6_30_26.xml" }] } });
    const focusedPrimary = BURKETT_PRIMARY_XML.replace("<tableEntryTotal>266<", "<tableEntryTotal>2<");
    // the information table is missing: the filing is not returned
    const noTable = mockFetchFor({ "output=atom": FEED_ATOM, "index.json": listing, "primary_doc.xml": focusedPrimary });
    assert.deepEqual(await fetchLatest13FFilings(5, noTable.impl as any, 0), []);
    assert.equal(pollDocuments("filings13f")?.first_error, "information table -> 404");
    assert.ok(!String(pollDocuments("filings13f")?.first_error).includes("BURKETT"));

    // the whole filing reads: an ok poll, nothing failed
    const whole = mockFetchFor({
      "output=atom": FEED_ATOM, "index.json": listing, "primary_doc.xml": focusedPrimary,
      "13F_BURKETT_6_30_26.xml": BURKETT_INFOTABLE_XML,
    });
    await refresh13FCache(40, whole.impl as any, 0);
    assert.deepEqual({ ...pollDocuments("filings13f"), at: null },
      { at: null, candidates: 1, returned: 1, skipped: 0, failed: 0, first_error: null, via_submission: 0 });
    let r = stream13f(edgarPollHealthStatus(base));
    assert.equal(r.polls_ok, 1);
    assert.equal(r.polls_failed, 0);
    assert.equal(r.rows_archived, 1);

    // a directory with no primary document is skipped, not failed
    const noPrimary = mockFetchFor({ "output=atom": FEED_ATOM, "index.json": JSON.stringify({ directory: { item: [] } }) });
    await refresh13FCache(40, noPrimary.impl as any, 0);
    assert.equal(pollDocuments("filings13f")?.skipped, 1);
    // an empty page: 0 candidates, an ok poll
    const empty = mockFetchFor({ "output=atom": `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>` });
    await refresh13FCache(40, empty.impl as any, 0);
    assert.equal(pollDocuments("filings13f")?.candidates, 0);
    r = stream13f(edgarPollHealthStatus(base));
    assert.equal(r.polls_ok, 3);
    assert.equal(r.polls_failed, 0);
  } finally {
    _reset13FCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("stageOf13FUrl names the three requests", () => {
  const b = "https://www.sec.gov/Archives/edgar/data/1762716/000176271626000003/";
  assert.equal(stageOf13FUrl(`${b}index.json`), "index.json");
  assert.equal(stageOf13FUrl(`${b}primary_doc.xml`), "primary_doc.xml");
  assert.equal(stageOf13FUrl(`${b}2026qtr2submissionapr2026.xml`), "information table");
  assert.equal(stageOf13FUrl(`${b}0001762716-26-000003.txt`), "submission text");
});

// ── [PIPELINE 2026-11-24, open_questions.md (ai2)] the submission-text fallback ──
// Measured on production 2026-11-24 13:37Z: 3 of 23 13F filings on the feed
// page answered 404 for index.json and the live poll skipped them.
//
// WHAT THESE FIXTURES ARE: the two XML documents are the real ones above
// (accession 0001762716-26-000003). The submission-text CONTAINER around them
// is assembled here in EDGAR's format, copied line for line from the real
// Form 4 submission text in __fixtures__/form4_submission/ (the session that
// built this could not fetch a 13F submission text: sec.gov refuses a client
// that declares no contact). The check against EDGAR's real 13F texts is the
// running comparison the poll makes on production (`submission_check`).
import {
  accessionOfFilingDir, docs13FFromSubmission, docs13FFromSubmissionText, fetch13FFilingDocs, read13FFiling,
  readTextCapped, SUBMISSION_MAX_BYTES,
} from "./edgar13f";
import { publicEdgarPollView, submissionCheck } from "./edgarPollHealth";

const sgmlDoc = (type: string, seq: number, name: string, xml: string) =>
  `<DOCUMENT>\n<TYPE>${type}\n<SEQUENCE>${seq}\n<FILENAME>${name}\n<TEXT>\n<XML>\n${xml}\n</XML>\n</TEXT>\n</DOCUMENT>\n`;
const submissionOf = (primary: string, info: string | null) =>
  `<SEC-DOCUMENT>0001762716-26-000003.txt : 20260704\n<SEC-HEADER>0001762716-26-000003.hdr.sgml : 20260704\nACCESSION NUMBER:\t\t0001762716-26-000003\nCONFORMED SUBMISSION TYPE:\t13F-HR\n</SEC-HEADER>\n`
  + sgmlDoc("13F-HR", 1, "primary_doc.xml", primary)
  + (info == null ? "" : sgmlDoc("INFORMATION TABLE", 2, "2026qtr2submissionapr2026.xml", info))
  + `</SEC-DOCUMENT>\n`;
const FOCUSED_PRIMARY = BURKETT_PRIMARY_XML.replace("<tableEntryTotal>266<", "<tableEntryTotal>2<");
const DIR = "https://www.sec.gov/Archives/edgar/data/1762716/000176271626000003/";
const ACC = "0001762716-26-000003";
const LISTING = JSON.stringify({ directory: { item: [{ name: "primary_doc.xml" }, { name: "2026qtr2submissionapr2026.xml" }] } });
const throwingGet = (get: (url: string) => { status: number; body?: string }, asked: string[], caps: Array<number | undefined> = []) =>
  async (url: string, opts?: { maxBytes?: number }) => {
    asked.push(url.replace(DIR, ""));
    caps.push(opts?.maxBytes);
    const r = get(url);
    if (r.status !== 200) throw Object.assign(new Error(`${url} -> ${r.status}`), { status: r.status });
    return r.body ?? "";
  };

test("the submission text gives the same 13F record the listing's two documents do, for a kept table and for an over-cap filing", async () => {
  for (const primary of [FOCUSED_PRIMARY, BURKETT_PRIMARY_XML]) {
    const viaListing = await read13FFiling(DIR, throwingGet((url) => {
      if (url.endsWith("/index.json")) return { status: 200, body: LISTING };
      if (url.endsWith("/primary_doc.xml")) return { status: 200, body: primary };
      if (url.endsWith("/2026qtr2submissionapr2026.xml")) return { status: 200, body: BURKETT_INFOTABLE_XML };
      return { status: 404 };
    }, []));
    const asked: string[] = [];
    const caps: Array<number | undefined> = [];
    const viaSubmission = await read13FFiling(DIR, throwingGet((url) =>
      (url.endsWith(`/${ACC}.txt`) ? { status: 200, body: submissionOf(primary, BURKETT_INFOTABLE_XML) } : { status: 404 }), asked, caps));
    assert.equal(viaListing.via, "listing");
    assert.equal(viaSubmission.via, "submission");
    assert.deepEqual(viaSubmission.doc, viaListing.doc);
    assert.deepEqual(asked, ["index.json", `${ACC}.txt`]);
    assert.deepEqual(caps, [undefined, SUBMISSION_MAX_BYTES], "the submission text is asked for with the size cap");
  }
  // the kept-table case really carries the table
  const kept = await docs13FFromSubmissionText(submissionOf(FOCUSED_PRIMARY, BURKETT_INFOTABLE_XML));
  assert.equal(kept?.managerName, "BURKETT FINANCIAL SERVICES, LLC");
  assert.equal(kept?.holdings?.length, 2);
  assert.equal(kept?.holdings?.[0].cusip, "808524300");
  assert.equal(kept?.holdingsOmitted, false);
});

test("docs13FFromSubmission finds the documents by what they are, not by TYPE or file name; no primary document reads as null", async () => {
  const odd = submissionOf(FOCUSED_PRIMARY, BURKETT_INFOTABLE_XML)
    .replace("<TYPE>13F-HR\n", "<TYPE>13F-HR/A\n").replace("<FILENAME>primary_doc.xml", "<FILENAME>renamed.xml")
    .replace("<TYPE>INFORMATION TABLE", "<TYPE>SOMETHING ELSE");
  const got = docs13FFromSubmission(odd);
  assert.ok(got.primary?.includes("<edgarSubmission"));
  assert.ok(got.info?.includes("informationTable"));
  assert.equal(got.whole, true);
  assert.equal(await docs13FFromSubmissionText("<SEC-DOCUMENT>\n<DOCUMENT>\n<TYPE>4\n<TEXT>\n<XML>\n<ownershipDocument/>\n</XML>\n</TEXT>\n</DOCUMENT>\n</SEC-DOCUMENT>\n"), null);
  assert.equal(accessionOfFilingDir(DIR), ACC);
  assert.equal(accessionOfFilingDir("https://www.sec.gov/Archives/edgar/data/1762716/"), null);
});

test("a submission text cut by the size cap: an over-cap filing still reads from its primary document; a table that is to be kept is never returned in part", async () => {
  const whole = submissionOf(FOCUSED_PRIMARY, BURKETT_INFOTABLE_XML);
  const cutInTable = whole.slice(0, whole.indexOf("808524797")); // inside the second holding
  await assert.rejects(docs13FFromSubmissionText(cutInTable), /information table did not end/);
  // the same cut under a primary document reporting 266 rows: summary-only, as from the listing
  const over = submissionOf(BURKETT_PRIMARY_XML, BURKETT_INFOTABLE_XML);
  const overCut = await docs13FFromSubmissionText(over.slice(0, over.indexOf("808524797")));
  assert.equal(overCut?.holdingsOmitted, true);
  assert.equal(overCut?.holdings, null);
  assert.equal(overCut?.entryTotal, 266);
  // no stated row count and a cut table: size unknown, so summary-only, never a part table
  const noTotal = submissionOf(FOCUSED_PRIMARY.replace(/<tableEntryTotal>2<\/tableEntryTotal>/, ""), BURKETT_INFOTABLE_XML);
  const noTotalCut = await docs13FFromSubmissionText(noTotal.slice(0, noTotal.indexOf("808524797")));
  assert.equal(noTotalCut?.holdingsOmitted, true);
  assert.equal(noTotalCut?.holdings, null);
  // cut before the primary document ended: an error, not a skipped filing
  await assert.rejects(docs13FFromSubmissionText(whole.slice(0, whole.indexOf("<summaryPage>"))), /no primary document/);
});

test("readTextCapped stops the download at the cap and cancels it; a response with no stream is read and cut", async () => {
  const enc = new TextEncoder();
  let reads = 0;
  let cancelled = false;
  const body = {
    getReader: () => ({
      read: async () => (++reads <= 5 ? { done: false, value: enc.encode("abcdefghij") } : { done: true }),
      cancel: async () => { cancelled = true; },
    }),
  };
  assert.equal(await readTextCapped({ text: async () => "unused", body }, 25), "abcdefghijabcdefghijabcde");
  assert.equal(reads, 3, "the fourth chunk is never asked for");
  assert.equal(cancelled, true);
  assert.equal(await readTextCapped({ text: async () => "abcdefghij" }, 4), "abcd");
  reads = 0; cancelled = false;
  assert.equal((await readTextCapped({ text: async () => "", body }, 1000)).length, 50, "a short answer is read whole");
  assert.equal(cancelled, false);
});

test("a listing that is refused or errors is not asked a second way", async () => {
  for (const status of [403, 429, 500]) {
    const asked: string[] = [];
    await assert.rejects(read13FFiling(DIR, throwingGet(() => ({ status }), asked)), new RegExp(`-> ${status}$`));
    assert.deepEqual(asked, ["index.json"], `${status}: one request`);
  }
  // a getText whose error carries no status field (the message alone says 404) still falls back
  const doc = await fetch13FFilingDocs(DIR, async (url) => {
    if (url.endsWith(".txt")) return submissionOf(FOCUSED_PRIMARY, BURKETT_INFOTABLE_XML);
    throw new Error(`${url} -> 404`);
  });
  assert.equal(doc?.holdings?.length, 2);
});

test("2026-11-24 morning replayed: the listing answers 404, the poll returns the filing from its submission text, counts it, and it is archived", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-13f-sub-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _reset13FCacheForTests();
  _resetEdgarPollHealthForTests();
  try {
    const edgar = mockFetchFor({ "output=atom": FEED_ATOM, [`${ACC}.txt`]: submissionOf(FOCUSED_PRIMARY, BURKETT_INFOTABLE_XML) });
    await refresh13FCache(40, edgar.impl as any, 0);
    assert.deepEqual({ ...pollDocuments("filings13f"), at: null },
      { at: null, candidates: 1, returned: 1, skipped: 0, failed: 0, first_error: null, via_submission: 1 });
    const r = stream13f(edgarPollHealthStatus(base));
    assert.equal(r.polls_ok, 1);
    assert.equal(r.polls_failed, 0);
    assert.equal(r.rows_archived, 1);
    assert.equal(r.submission_check, null, "a filing read from the submission text is not compared with itself");
    const row = latest13FFilings()!.filings[0];
    assert.equal(row.accession, ACC);
    assert.equal(row.holdings?.length, 2);
    assert.ok(!("via" in row), "the archived row has the same fields whichever copy it was read from");
    assert.equal(publicEdgarPollView(base).streams[1].documents?.via_submission, 1);
  } finally {
    _reset13FCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("the running check: one filing a poll read from its listing is read again from the submission text and compared, once per filing", async () => {
  _reset13FCacheForTests();
  _resetEdgarPollHealthForTests();
  const listed = {
    "output=atom": FEED_ATOM, "index.json": LISTING, "primary_doc.xml": FOCUSED_PRIMARY,
    "2026qtr2submissionapr2026.xml": BURKETT_INFOTABLE_XML,
  };
  // the two copies agree
  const same = mockFetchFor({ ...listed, [`${ACC}.txt`]: submissionOf(FOCUSED_PRIMARY, BURKETT_INFOTABLE_XML) });
  assert.equal((await fetchLatest13FFilings(5, same.impl as any, 0)).length, 1);
  assert.deepEqual(submissionCheck("filings13f"), { compared: 1, agree: 1, disagree: 0, unread: 0 });
  assert.equal(same.requested.filter((u) => u.endsWith(".txt")).length, 1);
  assert.equal(pollDocuments("filings13f")?.via_submission, 0);
  // the next poll sees the same filing: it is not compared again, and no extra request is made
  const again = mockFetchFor({ ...listed, [`${ACC}.txt`]: submissionOf(FOCUSED_PRIMARY, BURKETT_INFOTABLE_XML) });
  await fetchLatest13FFilings(5, again.impl as any, 0);
  assert.equal(again.requested.filter((u) => u.endsWith(".txt")).length, 0);
  assert.equal(submissionCheck("filings13f")?.compared, 1);
  // a submission text that reads differently is counted, and the filing is still returned from its listing
  _reset13FCacheForTests();
  const differs = mockFetchFor({ ...listed, [`${ACC}.txt`]: submissionOf(FOCUSED_PRIMARY.replace("BURKETT FINANCIAL", "SOMEONE ELSE"), BURKETT_INFOTABLE_XML) });
  const got = await fetchLatest13FFilings(5, differs.impl as any, 0);
  assert.equal(got[0].managerName, "BURKETT FINANCIAL SERVICES, LLC");
  assert.deepEqual(submissionCheck("filings13f"), { compared: 2, agree: 1, disagree: 1, unread: 0 });
  // a submission text that cannot be read is counted unread and fails nothing
  _reset13FCacheForTests();
  const absent = mockFetchFor(listed);
  assert.equal((await fetchLatest13FFilings(5, absent.impl as any, 0)).length, 1);
  assert.deepEqual(submissionCheck("filings13f"), { compared: 3, agree: 1, disagree: 1, unread: 1 });
  assert.equal(pollDocuments("filings13f")?.failed, 0);
  _reset13FCacheForTests();
});
