/**
 * appStoreRankings.ts — App Store rank + review-velocity archiver.
 * EDGE DOCTRINE axis (a), NEW DATA ROOTS #3 (research/open_questions.md).
 *
 * HYPOTHESIS (gate 2, not attempted here): consumer-app companies' mobile
 * rank/rating trajectory leads or confirms usage trends the market later
 * prices (DUOL/BMBL/MTCH/HOOD/COIN/RBLX class). SOBER PRIOR stated before
 * building, per REASONING STANDARD #10: this is the MOST arbitraged
 * alt-data category on the street — expect near-zero edge on these
 * large/mid-caps; the point of building it anyway is the archive itself
 * (BUILD-FIRST rule #2, accumulation) plus a cheap, zero-token nightly
 * script, not a claim of a found edge.
 *
 * LICENSING (research/open_questions.md NEW DATA ROOTS #3): Apple RSS
 * marketingtools top-chart JSON + iTunes Lookup rating counts are public
 * feeds, CONDITIONAL on low-volume internal use (this stream: ~28
 * calls/day). Google Play is PROHIBITED programmatically (robots.txt +
 * ToS) — Android side is dark here, stated honestly, never backfilled by
 * guessing. Apple customer-reviews RSS is VERIFIED DEAD (not attempted).
 *
 * HONEST GAPS: no downloads/revenue figures are free anywhere — ordinal
 * chart RANK and top-grossing rank are the only volume proxies. Charts
 * are OVERALL top-free/top-grossing (not per-genre): a watchlist app
 * ranked #1 in its genre but outside the overall top 100 records
 * rank:null, not a fabricated worst-case rank — a documented scoping
 * gap future work could close with genre-specific charts. Storefronts
 * cover 3 of ~175 countries (US/GB/CA) — a scoping choice, not global
 * coverage.
 *
 * WATCHLIST: hand-verified live via iTunes Search (2026-08-01) — each
 * entry's sellerName / bundleId was read from a live API response and
 * cross-checked against the public company before being hardcoded here
 * (the job-board resolver's name-guessing failure mode, open_questions.md
 * NEW DATA ROOTS #2, is exactly what this avoids — every id below is a
 * verified match, never inferred from ticker/company-name similarity).
 *
 * Boundary: no imports from trading logic (SPINOUT-READY DATA LAYER).
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface WatchlistApp {
  ticker: string;
  company: string;
  appId: string; // Apple trackId, verified live 2026-08-01
  appName: string;
}

// Verified live 2026-08-01 via `itunes.apple.com/search?term=<name>&entity=software`
// — trackId + sellerName + bundleId read from the actual response, not guessed.
export const WATCHLIST: WatchlistApp[] = [
  { ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", appName: "Duolingo: Language Lessons" },
  { ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", appName: "Bumble Dating App: Meet & Date" },
  { ticker: "MTCH", company: "Match Group, Inc. (Tinder)", appId: "547702041", appName: "Tinder Dating App: Date & Chat" },
  { ticker: "HOOD", company: "Robinhood Markets, Inc.", appId: "938003185", appName: "Robinhood: Trading & Investing" },
  { ticker: "COIN", company: "Coinbase, Inc.", appId: "886427730", appName: "Coinbase: Buy Crypto & Stocks" },
  { ticker: "RBLX", company: "Roblox Corporation", appId: "431946152", appName: "Roblox" },
  { ticker: "PINS", company: "Pinterest, Inc.", appId: "429047995", appName: "Pinterest" },
  { ticker: "SNAP", company: "Snap, Inc.", appId: "447188370", appName: "Snapchat" },
  { ticker: "UBER", company: "Uber Technologies, Inc.", appId: "368677368", appName: "Uber - Request a ride" },
  { ticker: "ABNB", company: "Airbnb, Inc.", appId: "401626263", appName: "Airbnb" },
  { ticker: "DASH", company: "DoorDash, Inc.", appId: "719972451", appName: "DoorDash: Food, Grocery, More" },
  { ticker: "LYFT", company: "Lyft, Inc.", appId: "529379082", appName: "Lyft" },
  { ticker: "SPOT", company: "Spotify Technology S.A.", appId: "324684580", appName: "Spotify: Music and Podcasts" },
  { ticker: "CHGG", company: "Chegg, Inc.", appId: "385758163", appName: "Chegg Study - Homework Help" },
  { ticker: "ETSY", company: "Etsy, Inc.", appId: "477128284", appName: "Etsy: Shop Home, Style & More" },
  { ticker: "YELP", company: "Yelp, Inc.", appId: "284910350", appName: "Yelp: Food, Services & Reviews" },
];

const STOREFRONTS = ["us", "gb", "ca"] as const;
type Storefront = (typeof STOREFRONTS)[number];
const CHARTS = ["top-free", "top-grossing"] as const;
type Chart = (typeof CHARTS)[number];

export interface RankRecord {
  t: "rank";
  key: string; // rt|storefront|chart|ticker
  ticker: string;
  company: string;
  appId: string;
  storefront: Storefront;
  chart: Chart;
  rank: number | null; // null = outside the fetched top 100, never fabricated
  rt: string;
}

export interface RatingRecord {
  t: "rating";
  key: string; // rt|ticker
  ticker: string;
  company: string;
  appId: string;
  avgRating: number | null;
  ratingCount: number | null;
  version: string | null;
  rt: string;
}

export type AppStoreRecord = RankRecord | RatingRecord;

/** Parses one marketingtools RSS chart response into a rank row PER
 *  watchlist app (present or not) — a stable row per ticker per day is
 *  what makes rank:null legible as "checked, not found" rather than
 *  "never fetched". */
export function parseChart(json: any, storefront: Storefront, chart: Chart, watchlist: WatchlistApp[], rt: string): RankRecord[] {
  const results: any[] = json?.feed?.results || [];
  const rankById = new Map<string, number>();
  results.forEach((r, i) => {
    if (r?.id) rankById.set(String(r.id), i + 1);
  });
  return watchlist.map((a) => ({
    t: "rank",
    key: `${rt}|${storefront}|${chart}|${a.ticker}`,
    ticker: a.ticker,
    company: a.company,
    appId: a.appId,
    storefront,
    chart,
    rank: rankById.get(a.appId) ?? null,
    rt,
  }));
}

/** Parses an iTunes Lookup batch response into rating rows. Apps absent
 *  from the response (delisted, bad id) get null fields — never dropped,
 *  so a broken watchlist entry is visible in the archive, not silent. */
export function parseLookup(json: any, watchlist: WatchlistApp[], rt: string): RatingRecord[] {
  const results: any[] = json?.results || [];
  const byId = new Map<string, any>();
  for (const r of results) {
    if (r?.trackId != null) byId.set(String(r.trackId), r);
  }
  return watchlist.map((a) => {
    const r = byId.get(a.appId);
    return {
      t: "rating",
      key: `${rt}|${a.ticker}`,
      ticker: a.ticker,
      company: a.company,
      appId: a.appId,
      avgRating: typeof r?.averageUserRating === "number" ? r.averageUserRating : null,
      ratingCount: typeof r?.userRatingCount === "number" ? r.userRatingCount : null,
      version: typeof r?.version === "string" ? r.version : null,
      rt,
    };
  });
}

// ── Fetch (injectable) ──────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

async function getJson(fetchImpl: FetchFn, url: string): Promise<any> {
  const r = await fetchImpl(url, { headers: UA, signal: AbortSignal.timeout(20000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return JSON.parse(await r.text());
}

/** One poll cycle: 3 storefronts x 2 charts (6 calls) + 1 US ratings batch
 *  lookup = 7 calls/cycle; at the 6h cadence below that is ~28 calls/day,
 *  matching the ladder's "~30 calls/day" budget. Each source failure is
 *  isolated — one dead storefront/chart never drops the rest of the
 *  cycle (mirrors fetchFdaEvents's per-source try/catch). */
export async function fetchAppStoreSnapshot(
  fetchImpl: FetchFn = fetch as any, nowMs?: number,
): Promise<AppStoreRecord[]> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const out: AppStoreRecord[] = [];
  for (const storefront of STOREFRONTS) {
    for (const chart of CHARTS) {
      try {
        const url = `https://rss.marketingtools.apple.com/api/v2/${storefront}/apps/${chart}/100/apps.json`;
        out.push(...parseChart(await getJson(fetchImpl, url), storefront, chart, WATCHLIST, rt));
      } catch (e: any) {
        console.error(`[datacore] appstore chart ${storefront}/${chart}:`, e?.message || e);
      }
    }
  }
  try {
    const ids = WATCHLIST.map((a) => a.appId).join(",");
    const url = `https://itunes.apple.com/lookup?id=${ids}&country=us`;
    out.push(...parseLookup(await getJson(fetchImpl, url), WATCHLIST, rt));
  } catch (e: any) {
    console.error("[datacore] appstore ratings lookup:", e?.message || e);
  }
  return out;
}

// ── Archive ────────────────────────────────────────────────────────────────

const archivedKeys = new Set<string>();
let seeded = false;

function appstoreDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "appstore");
}

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < 3; i++) { // records are daily-keyed; a short seed window suffices
    const day = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { archivedKeys.add(JSON.parse(line).key); } catch {}
      }
    }
  }
}

export function archiveAppStoreRecords(records: AppStoreRecord[], baseDir?: string, nowMs?: number): number {
  const dir = appstoreDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = records.filter((r) => r.key && !archivedKeys.has(r.key));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((r) => archivedKeys.add(r.key));
    if (archivedKeys.size > 50_000) archivedKeys.clear();
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] appstore archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldAppStoreDays(baseDir?: string, nowMs?: number): number {
  const dir = appstoreDir(baseDir);
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

let cache: { at: number; records: AppStoreRecord[] } | null = null;
let polling = false;

export function latestAppStoreRankings(): { at: number; records: AppStoreRecord[] } | null {
  return cache;
}

export async function refreshAppStoreCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const records = await fetchAppStoreSnapshot(fetchImpl, nowMs);
    if (records.length || !cache) cache = { at: Date.now(), records };
    try { archiveAppStoreRecords(records, undefined, nowMs); } catch {}
    try { gzipOldAppStoreDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] appstore refresh:", e?.message || e);
  }
}

/** 6h poll — ranks/ratings move slowly relative to a trading day; 4
 *  cycles/day (~28 calls total) sits far under any Apple public-feed
 *  rate concern. */
export function bootAppStorePoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshAppStoreCache();
  setInterval(() => { refreshAppStoreCache(); }, intervalMs).unref?.();
}
