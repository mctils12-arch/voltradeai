// Treasury auctions battery (BUILD ORDER 2 #4, 2026-07-05). Fixture values
// copied from the live TA_WS probe of 2026-07-05 (a real 4-week bill).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseAuctions, archiveAuctions, gzipOldAuctionDays } from "./treasuryAuctions";

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
