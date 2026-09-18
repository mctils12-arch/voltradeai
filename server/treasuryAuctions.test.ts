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
