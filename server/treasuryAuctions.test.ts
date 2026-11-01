// Treasury auctions battery (BUILD ORDER 2 #4, 2026-07-05). Fixture values
// copied from the live TA_WS probe of 2026-07-05 (a real 4-week bill).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseAuctions, archiveAuctions, gzipOldAuctionDays,
  backfillAuctionsFromArchive, refreshAuctionCache, latestAuctions, _resetAuctionCacheForTests,
} from "./treasuryAuctions";

const BILL = {
  cusip: "912797US4", securityType: "Bill", securityTerm: "4-Week",
  auctionDate: "2026-07-02T00:00:00", issueDate: "2026-07-07T00:00:00",
  maturityDate: "2026-08-04T00:00:00", reopening: "No",
  bidToCoverRatio: "2.720000",
  highYield: "", highDiscountRate: "3.605000", highInvestmentRate: "3.665000",
  averageMedianYield: "", averageMedianDiscountRate: "3.510000",
  totalAccepted: "84196225000",
  competitiveTendered: "223450205000", competitiveAccepted: "77296225000",
  primaryDealerTendered: "150230000000", primaryDealerAccepted: "22571240000",
  directBidderTendered: "6150000000", directBidderAccepted: "2900000000",
  indirectBidderTendered: "67070205000", indirectBidderAccepted: "51824985000",
};

test("parseAuctions: real bill normalizes; '' stays null, never guessed", () => {
  const recs = parseAuctions([BILL], "2026-07-05");
  assert.equal(recs.length, 1);
  const a = recs[0];
  assert.equal(a.cusip, "912797US4");
  assert.equal(a.auction_date, "2026-07-02");
  assert.equal(a.type, "Bill");
  assert.equal(a.reopening, false);
  assert.equal(a.bid_to_cover, 2.72);
  assert.equal(a.high_yield, null, "bills carry no yield — null, not 0");
  assert.equal(a.high_discount_rate, 3.605);
  assert.equal(a.high_investment_rate, 3.665);
  assert.equal(a.total_accepted, 84196225000);
  // DERIVED dealer take = pd_accepted / competitive_accepted
  assert.equal(a.dealer_take, +(22571240000 / 77296225000).toFixed(4));
});

test("parseAuctions: pre-result and malformed rows dropped", () => {
  const recs = parseAuctions([
    { ...BILL, bidToCoverRatio: "" },              // announced, not yet auctioned
    { ...BILL, cusip: "" },                        // no cusip
    { ...BILL, auctionDate: null },                // no date
    { ...BILL, primaryDealerAccepted: "" },        // partial results: kept, dealer_take null
    "garbage",
  ], "2026-07-05");
  assert.equal(recs.length, 1);
  assert.equal(recs[0].dealer_take, null);
  assert.deepEqual(parseAuctions(null, "2026-07-05"), []);
});

test("archive: dedup by cusip|auction_date; reopening on a new date is fresh", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tsy-"));
  const now = Date.parse("2026-07-05T12:00:00Z");
  const recs = parseAuctions([BILL], "2026-07-05");
  assert.equal(archiveAuctions(recs, base, now), 1);
  assert.equal(archiveAuctions(recs, base, now), 0, "same auction never re-archives");
  const reopen = parseAuctions([{ ...BILL, auctionDate: "2026-07-09T00:00:00", reopening: "Yes" }], "2026-07-09");
  assert.equal(archiveAuctions(reopen, base, now), 1, "same cusip, new auction date = new record");
  // gz lifecycle
  assert.equal(gzipOldAuctionDays(base, now + 3 * 86400_000), 1);
  const day = path.join(base, "treasuryauctions", "2026-07-05.jsonl");
  assert.ok(!fs.existsSync(day) && fs.existsSync(`${day}.gz`));
});

// Cold-cache-no-disk-backfill fix thread (started 2026-09-10; see
// usgsWater.ts/cbpBorderWait.ts for the same shape): a cold boot or a live
// TA_WS outage must not leave /api/data/treasury-auctions warming_up forever
// when a real, immutable archive of past auction results already sits on disk.

test("backfillAuctionsFromArchive: keeps only the latest rt per (cusip, auction_date) identity across the lookback window", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vttsy-backfill-"));
  const dir = path.join(root, "treasuryauctions");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-10T12:00:00Z");
  const day0 = new Date(now).toISOString().slice(0, 10);
  const day1 = new Date(now - 86400_000).toISOString().slice(0, 10);
  const olderView = { ...parseAuctions([BILL], "2026-09-09")[0] };
  const newerView = { ...parseAuctions([BILL], "2026-09-10")[0], bid_to_cover: 2.9 };
  const otherAuction = parseAuctions([{ ...BILL, cusip: "912796XY1", auctionDate: "2026-09-08T00:00:00" }], "2026-09-10")[0];
  fs.writeFileSync(path.join(dir, `${day1}.jsonl`), JSON.stringify(olderView) + "\n");
  fs.writeFileSync(path.join(dir, `${day0}.jsonl`), [newerView, otherAuction].map((o) => JSON.stringify(o)).join("\n") + "\n");
  const out = backfillAuctionsFromArchive(root, now, 30);
  assert.equal(out.length, 2, "two distinct (cusip, auction_date) identities");
  const bill = out.find((o) => o.cusip === "912797US4")!;
  assert.equal(bill.bid_to_cover, 2.9, "the newer-rt record wins over the older one");
  assert.ok(out.find((o) => o.cusip === "912796XY1"), "the second identity is preserved");
});

test("backfillAuctionsFromArchive: respects the days window and an empty lookback reconstructs nothing", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vttsy-backfill-window-"));
  const dir = path.join(root, "treasuryauctions");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-10T12:00:00Z");
  const tooOld = new Date(now - 40 * 86400_000).toISOString().slice(0, 10);
  const row = parseAuctions([BILL], "2026-07-31")[0];
  fs.writeFileSync(path.join(dir, `${tooOld}.jsonl`), JSON.stringify(row) + "\n");
  assert.deepEqual(backfillAuctionsFromArchive(root, now, 30), [], "a day 40 days back is outside the default 30-day window");
  assert.deepEqual(backfillAuctionsFromArchive(root, now, 45), [row], "widening the window picks it up");
});

test("refreshAuctionCache: cold cache backfills from the on-disk archive when the live poll throws", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vttsy-coldcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetAuctionCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "treasuryauctions");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    const row = parseAuctions([BILL], today)[0];
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(row) + "\n");
    assert.equal(latestAuctions(), null, "cache must still be cold going into this cycle");
    const throwingFetch = (async () => { throw new Error("treasurydirect.gov unreachable"); }) as any;
    await refreshAuctionCache(throwingFetch);
    const cached = latestAuctions();
    assert.ok(cached, "cache must be populated, not left null, despite the live poll throwing");
    assert.equal(cached!.auctions.length, 1);
    assert.equal(cached!.auctions[0].cusip, "912797US4", "backfilled from the archived record, not fabricated");
  } finally {
    _resetAuctionCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshAuctionCache: a subsequently-warm cache is untouched by a second transport failure", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vttsy-warmcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetAuctionCacheForTests();
  try {
    const goodFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([BILL]) });
    await refreshAuctionCache(goodFetch as any, Date.parse("2026-09-10T12:00:00Z"));
    const warm = latestAuctions();
    assert.equal(warm!.auctions.length, 1, "warm the cache first");

    const throwingFetch = (async () => { throw new Error("503"); }) as any;
    await refreshAuctionCache(throwingFetch);
    assert.equal(latestAuctions(), warm, "a transport failure must not clobber an already-warm cache with a stale disk read");
  } finally {
    _resetAuctionCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshAuctionCache: a genuinely empty but successful poll on a cold cache is still trusted, not overridden by the archive", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vttsy-coldcache-empty-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetAuctionCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "treasuryauctions");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    const row = parseAuctions([BILL], today)[0];
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(row) + "\n");
    const emptyFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([]) });
    await refreshAuctionCache(emptyFetch as any);
    const cached = latestAuctions();
    assert.ok(cached, "cache must be populated on a successful poll, even an empty one");
    assert.equal(cached!.auctions.length, 0, "an empty-but-successful poll is trusted as-is, never silently swapped for a stale archive read");
  } finally {
    _resetAuctionCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

// ── Schema drift regression (2026-11-01) ─────────────────────────────────
// TreasuryDirect dropped `auctionDate` and renamed the bidder/yield fields
// ~2026-10-16; parseAuctions silently dropped every row and the archive went
// dark for 16 days (stream health "stale", /api/data/treasury-auctions
// count 0). Fixtures below are rows copied from the LIVE 2026-10-29 payload
// (fields trimmed to those the parser reads).
const BILL_V2 = {
  cusip: "912797VZ7", securityType: "Bill", securityTerm: "4-Week",
  competitiveAuctionCloseDate: "2026-10-29T00:00:00", issueDate: "2026-11-03T00:00:00",
  maturityDate: "2026-12-01T00:00:00", reopening: "Yes", bidToCoverRatio: "3.10",
  highYield: "", highRate: "3.910", investmentRate: "3.976", medianYield: "", medianRate: "3.860",
  totalParAwardedInsideOfOfferingAmount: "115000324600",
  subtotalCompetitiveParRecognizedInsideOfOfferingAmount: "348759551000",
  subtotalCompetitiveParAwardedInsideOfOfferingAmount: "107332678000",
  primaryDealersCompetitiveParRecognized: "209755000000", primaryDealersCompetitiveParAwarded: "26232400000",
  directBiddersCompetitiveParRecognized: "15850000000", directBiddersCompetitiveParAwarded: "6151760000",
  indirectBiddersCompetitiveParRecognized: "123154551000", indirectBiddersCompetitiveParAwarded: "74948518000",
};
const NOTE_V2 = {
  cusip: "91282CRV5", securityType: "Note", securityTerm: "7-Year",
  competitiveAuctionCloseDate: "2026-10-29T00:00:00", issueDate: "2026-11-02T00:00:00",
  maturityDate: "2033-10-31T00:00:00", reopening: "No", bidToCoverRatio: "2.46",
  highYield: "4.894", highRate: "", investmentRate: "", medianYield: "4.840", medianRate: "",
  totalParAwardedInsideOfOfferingAmount: "44000055600",
  subtotalCompetitiveParRecognizedInsideOfOfferingAmount: "108083482000",
  subtotalCompetitiveParAwardedInsideOfOfferingAmount: "43640334000",
  primaryDealersCompetitiveParRecognized: "57297000000", primaryDealersCompetitiveParAwarded: "4107456000",
  directBiddersCompetitiveParRecognized: "20055000000", directBiddersCompetitiveParAwarded: "14321000000",
  indirectBiddersCompetitiveParRecognized: "30731482000", indirectBiddersCompetitiveParAwarded: "25211878000",
};

test("parseAuctions: current TA_WS schema (no auctionDate) parses bill + note", () => {
  const recs = parseAuctions([BILL_V2, NOTE_V2], "2026-11-01");
  assert.equal(recs.length, 2, "both live rows must parse — this was 0 before the fix");
  const [b, n] = recs;
  assert.equal(b.auction_date, "2026-10-29");
  assert.equal(b.type, "Bill");
  assert.equal(b.reopening, true);
  assert.equal(b.bid_to_cover, 3.1);
  assert.equal(b.high_discount_rate, 3.91);
  assert.equal(b.high_investment_rate, 3.976);
  assert.equal(b.high_yield, null, "bills carry no yield — null, never guessed");
  assert.equal(b.avg_median_discount_rate, 3.86);
  assert.equal(b.dealer_take, +(26232400000 / 107332678000).toFixed(4));
  assert.equal(n.high_yield, 4.894);
  assert.equal(n.avg_median_yield, 4.84);
  assert.equal(n.high_discount_rate, null);
  assert.equal(n.auction_date, "2026-10-29");
});

test("parseAuctions: v2 bidder splits reconcile (PD+direct+indirect == competitive; B2C == published)", () => {
  for (const [raw, a] of [[BILL_V2, parseAuctions([BILL_V2], "x")[0]], [NOTE_V2, parseAuctions([NOTE_V2], "x")[0]]] as const) {
    assert.equal(a.primary_dealer_accepted! + a.direct_accepted! + a.indirect_accepted!, a.competitive_accepted,
      "bidder-class awards must sum to the competitive subtotal to the dollar");
    assert.equal(a.primary_dealer_tendered! + a.direct_tendered! + a.indirect_tendered!, a.competitive_tendered);
    assert.ok(a.total_accepted! >= a.competitive_accepted!);
    assert.equal(a.bid_to_cover, Number(raw.bidToCoverRatio));
  }
});

test("parseAuctions: v2 rows dedup against v1-era archive keys (same cusip|auction_date identity)", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "tsy-v2-"));
  const now = Date.parse("2026-11-01T12:00:00Z");
  const v1 = parseAuctions([{ ...BILL, cusip: "912797VZ7", auctionDate: "2026-10-29T00:00:00" }], "2026-10-30");
  const v2 = parseAuctions([BILL_V2], "2026-11-01");
  assert.equal(archiveAuctions(v1, base, now), 1);
  assert.equal(archiveAuctions(v2, base, now), 0, "schema change must not double-archive an auction");
});

test("fetchAuctions: rows with results but zero parsed logs SCHEMA DRIFT loudly", async () => {
  const { fetchAuctions } = await import("./treasuryAuctions");
  const errs: string[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errs.push(a.join(" ")); };
  try {
    const drifted = { ...BILL_V2, competitiveAuctionCloseDate: undefined, cusip: undefined };
    const f = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([drifted]) });
    assert.deepEqual(await fetchAuctions(f as any), []);
  } finally { console.error = orig; }
  assert.ok(errs.some((e) => e.includes("SCHEMA DRIFT")), "silent degradation is how this went dark for 16 days");
});
