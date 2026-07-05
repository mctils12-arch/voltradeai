/**
 * treasuryAuctions.ts — US Treasury auction results (BUILD ORDER 2 #4,
 * 2026-07-05; access live-probed keyless the same day). TreasuryDirect
 * TA_WS: US government work, public domain.
 *
 * HYPOTHESIS (gate-locked in the build order — this stream is archive +
 * RAW display only): bid-to-cover deterioration and a rising dealer take
 * (dealers absorb what investors won't) precede rate-regime shifts the
 * bot's regime classifier consumes. The classic "tail vs when-issued"
 * stress metric needs a WI quote at 1pm ET, which is PAID data — the
 * free stress metrics here are bid-to-cover + bidder-class shares, and
 * the manifest says so.
 *
 * TA_WS serves every numeric as a STRING with '' for absent — bills
 * carry discount rates (highDiscountRate), notes/bonds carry yields
 * (highYield); whichever a record lacks stays null, never guessed.
 * Auction results are immutable once published — dedup by
 * cusip|auctionDate (reopenings share a cusip but auction on new dates).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const TA_WS_URL = "https://www.treasurydirect.gov/TA_WS/securities/auctioned?days=30&format=json";

export interface AuctionRec {
  cusip: string;
  auction_date: string;        // YYYY-MM-DD
  issue_date: string | null;
  maturity_date: string | null;
  type: string;                // Bill | Note | Bond | TIPS | FRN | CMB
  term: string | null;
  reopening: boolean | null;
  bid_to_cover: number | null;
  high_yield: number | null;           // notes/bonds/TIPS
  high_discount_rate: number | null;   // bills
  high_investment_rate: number | null; // bills, coupon-equivalent
  avg_median_yield: number | null;
  avg_median_discount_rate: number | null;
  total_accepted: number | null;
  competitive_tendered: number | null;
  competitive_accepted: number | null;
  primary_dealer_tendered: number | null;
  primary_dealer_accepted: number | null;
  direct_tendered: number | null;
  direct_accepted: number | null;
  indirect_tendered: number | null;
  indirect_accepted: number | null;
  /** DERIVED: primary_dealer_accepted / competitive_accepted — the "dealers
   *  ate it" share; null whenever either input is absent. */
  dealer_take: number | null;
  rt: string;                  // as-seen UTC date
}

const num = (v: any): number | null => {
  if (typeof v !== "string" || v.trim() === "") return null;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
};

const day = (v: any): string | null =>
  typeof v === "string" && v.length >= 10 ? v.slice(0, 10) : null;

/** TA_WS array -> normalized records. Rows without a cusip + auction date
 *  (announced-but-unauctioned entries appear in the window) are dropped;
 *  rows with a date but no RESULTS yet keep null metrics and re-archive
 *  is prevented by the dedup key, so results-day rows append as fresh
 *  vintages only when the numbers differ is NOT needed — TA_WS rows are
 *  results-complete once bidToCoverRatio is set, and pre-result rows are
 *  skipped below. */
export function parseAuctions(json: any, rt: string): AuctionRec[] {
  const out: AuctionRec[] = [];
  for (const r of Array.isArray(json) ? json : []) {
    const cusip = r?.cusip;
    const ad = day(r?.auctionDate);
    if (!cusip || !ad) continue;
    const b2c = num(r?.bidToCoverRatio);
    if (b2c == null) continue; // pre-results row — picked up on a later poll
    const pdAcc = num(r?.primaryDealerAccepted);
    const compAcc = num(r?.competitiveAccepted);
    out.push({
      cusip,
      auction_date: ad,
      issue_date: day(r?.issueDate),
      maturity_date: day(r?.maturityDate),
      type: r?.securityType || "Unknown",
      term: r?.securityTerm || null,
      reopening: r?.reopening === "Yes" ? true : r?.reopening === "No" ? false : null,
      bid_to_cover: b2c,
      high_yield: num(r?.highYield),
      high_discount_rate: num(r?.highDiscountRate),
      high_investment_rate: num(r?.highInvestmentRate),
      avg_median_yield: num(r?.averageMedianYield),
      avg_median_discount_rate: num(r?.averageMedianDiscountRate),
      total_accepted: num(r?.totalAccepted),
      competitive_tendered: num(r?.competitiveTendered),
      competitive_accepted: compAcc,
      primary_dealer_tendered: num(r?.primaryDealerTendered),
      primary_dealer_accepted: pdAcc,
      direct_tendered: num(r?.directBidderTendered),
      direct_accepted: num(r?.directBidderAccepted),
      indirect_tendered: num(r?.indirectBidderTendered),
      indirect_accepted: num(r?.indirectBidderAccepted),
      dealer_take: pdAcc != null && compAcc != null && compAcc > 0
        ? +(pdAcc / compAcc).toFixed(4) : null,
      rt,
    });
  }
  return out;
}

// ── Fetch ───────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchAuctions(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<AuctionRec[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const r = await fetchImpl(TA_WS_URL, {
    headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
    signal: AbortSignal.timeout(20000) as any,
  });
  if (!r.ok) throw new Error(`treasurydirect -> ${r.status}`);
  return parseAuctions(JSON.parse(await r.text()), rt);
}

// ── Archive (dedup cusip|auction_date — results are immutable) ─────────────

const archivedKeys = new Set<string>();
let seeded = false;

function auctionsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "treasuryauctions");
}

const keyOf = (a: AuctionRec) => `${a.cusip}|${a.auction_date}`;

function seedSeen(dir: string, nowMs: number): void {
  // 30-day fetch window -> seed 35 days of files so restarts never re-archive
  for (let i = 0; i < 35; i++) {
    const d = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${d}.jsonl`), path.join(dir, `${d}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { archivedKeys.add(keyOf(JSON.parse(line))); } catch {}
      }
    }
  }
}

export function archiveAuctions(recs: AuctionRec[], baseDir?: string, nowMs?: number): number {
  const dir = auctionsDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = recs.filter((r) => !archivedKeys.has(keyOf(r)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((r) => archivedKeys.add(keyOf(r)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] treasuryauctions archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldAuctionDays(baseDir?: string, nowMs?: number): number {
  const dir = auctionsDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; auctions: AuctionRec[] } | null = null;
let polling = false;

export function latestAuctions() {
  return cache;
}

export async function refreshAuctionCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const auctions = await fetchAuctions(fetchImpl, nowMs);
    if (auctions.length || !cache) cache = { at: Date.now(), auctions };
    try { archiveAuctions(auctions, undefined, nowMs); } catch {}
    try { gzipOldAuctionDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] treasuryauctions refresh:", e?.message || e);
  }
}

/** 6h poll — auctions settle ~1pm ET on auction days; the 30-day fetch
 *  window plus dedup makes cadence forgiving. Eager boot per KNOWN
 *  BROKEN #9. */
export function bootTreasuryPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshAuctionCache();
  setInterval(() => { refreshAuctionCache(); }, intervalMs).unref?.();
}
