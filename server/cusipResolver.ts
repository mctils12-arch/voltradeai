/**
 * cusipResolver.ts — CUSIP/ISIN -> ticker resolution via OpenFIGI's free
 * public mapping API (api.openfigi.com/v3/mapping).
 *
 * WHY THIS EXISTS: dtccSwaps.ts's archived rows carry only underlierId (raw
 * CUSIP/ISIN) and underlierName (DTCC's own free-text UPI description) — no
 * ticker. Matching a swap notional against the bot's actual tradable
 * universe (item (3) queued in the 2026-08-22 GATE 1 session's NEXT list,
 * research/experiments.md) needs that join. BUILD-FIRST (CLAUDE.md): the raw
 * material (CUSIP/ISIN reference-data mapping) is NOT something we can
 * derive from data we already hold — a CUSIP is an opaque identifier with no
 * public checksum-only path to a ticker — so this is the one case where an
 * external LOOKUP service is the correct answer, not a build-around. OpenFIGI
 * (Bloomberg, api.openfigi.com) is free and keyless for exactly this use: a
 * reference-data MAPPING lookup for identifiers we have observed, not bulk
 * redistribution of their database — this module resolves only CUSIPs we've
 * actually archived and caches the result forever (never re-queries a CUSIP
 * once resolved OR confirmed unresolvable), which is both the frugal and the
 * compliant way to use it. An optional `OPENFIGI_API_KEY` (env var, free
 * registration raises the request-rate ceiling — see openfigi.com/api) is
 * honored if present but never required.
 *
 * Generic by design (any datacore module with a CUSIP/ISIN needing a ticker
 * can reuse resolveCusips) — dtccSwaps.ts is the first, not the only,
 * consumer. This module never itself hits the network from an import; only
 * resolveCusips()/refreshCusipCache() do, and both take an injectable
 * fetchImpl so no test needs live network.
 */
import fs from "fs";
import path from "path";

const OPENFIGI_URL = "https://api.openfigi.com/v3/mapping";

// Free/keyless OpenFIGI tier: 25 requests/minute, <=10 jobs/request. A
// registered free key raises both ceilings (25 req / 6s, <=100 jobs/req) —
// see openfigi.com/api. Conservative defaults so this never needs tuning
// per environment; a present OPENFIGI_API_KEY widens both automatically.
const KEYLESS_BATCH_SIZE = 10;
const KEYLESS_DELAY_MS = 2600; // >25/min with margin
const KEYED_BATCH_SIZE = 100;
const KEYED_DELAY_MS = 300; // >25/6s with margin

export interface ResolvedIdentifier {
  ticker: string;
  name: string;
  figi: string;
  exchCode: string;
}

// A cache entry is `null` for a CUSIP OpenFIGI genuinely has no mapping for
// (queried once, confirmed empty) — distinct from "never queried" (absent
// key). Storing the negative result is what makes the cache actually save
// requests: without it, every run would re-query every unresolvable CUSIP.
export interface CusipCacheEntry {
  result: ResolvedIdentifier | null;
  resolvedAt: string;
}
export type CusipCache = Record<string, CusipCacheEntry>;

// cachePath is caller-supplied (this module has no archive of its own —
// see the header note above), so durability is the CALLER's obligation, not
// this module's: the one existing caller, scripts/dtcc_resolve_cusips.ts,
// derives its path from archiveBaseDir() (server/datacoreArchive.ts), same
// as every other datacore archive, so a resolved-ticker cache survives a
// Railway redeploy exactly like dtccSwaps.ts's own archive does.
export function loadCusipCache(cachePath: string): CusipCache {
  try {
    return JSON.parse(fs.readFileSync(cachePath, "utf8"));
  } catch {
    return {};
  }
}

export function saveCusipCache(cachePath: string, cache: CusipCache): void {
  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(cache, null, 0));
}

export function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

interface OpenFigiDatum {
  figi: string;
  name: string;
  ticker: string;
  exchCode: string;
  securityType?: string;
  marketSector?: string;
}
type OpenFigiJobResult = { data: OpenFigiDatum[] } | { error: string; warning?: string };

/** OpenFIGI returns every listing venue for a security (US primary tape,
 *  Chicago, unlisted composite, ...) — prefer the primary US composite
 *  listing (exchCode "US"); an equity swap underlier with no US-listed
 *  entry at all is out of scope for this pipeline's US-tradable-universe
 *  join anyway, so falling back to the first entry is a reasonable,
 *  honestly-labeled best-effort rather than a silent misattribution (the
 *  ticker/exchCode are both returned, so a caller can always filter
 *  further). Returns null only when OpenFIGI returned zero listings. */
export function pickBestMatch(data: OpenFigiDatum[]): ResolvedIdentifier | null {
  if (!data.length) return null;
  const usComposite = data.find((d) => d.exchCode === "US") || data[0];
  return {
    ticker: usComposite.ticker,
    name: usComposite.name,
    figi: usComposite.figi,
    exchCode: usComposite.exchCode,
  };
}

/** Pairs a batch's OpenFIGI response array (order-aligned with the request
 *  jobs per the API's documented contract) back to the CUSIPs that were
 *  queried. A per-job `error` (e.g. "No identifier found") resolves to
 *  null, not an exception — an unmapped CUSIP is an expected, cacheable
 *  outcome, not a fetch failure. */
export function parseOpenFigiBatch(cusips: string[], body: OpenFigiJobResult[]): Map<string, ResolvedIdentifier | null> {
  const out = new Map<string, ResolvedIdentifier | null>();
  cusips.forEach((cusip, i) => {
    const job = body[i];
    out.set(cusip, job && "data" in job ? pickBestMatch(job.data) : null);
  });
  return out;
}

type NodeFetchResponse = { ok: boolean; status: number; json: () => Promise<any> };
type FetchFn = (url: string, init: { method: string; headers: Record<string, string>; body: string }) => Promise<NodeFetchResponse>;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Resolves a list of CUSIPs to tickers, skipping anything already in the
 *  on-disk cache (resolved OR confirmed-unresolvable) and persisting new
 *  results as it goes (a crash mid-run loses at most the in-flight batch,
 *  never previously-cached work). Rate-limits itself to stay under
 *  OpenFIGI's published ceiling for whichever tier `apiKey` implies. */
export async function resolveCusips(
  cusips: string[],
  opts: { cachePath: string; fetchImpl?: FetchFn; apiKey?: string; nowIso?: () => string } = { cachePath: "" },
): Promise<Map<string, ResolvedIdentifier | null>> {
  const fetchImpl = opts.fetchImpl ?? (fetch as unknown as FetchFn);
  const nowIso = opts.nowIso ?? (() => new Date().toISOString());
  const cache = loadCusipCache(opts.cachePath);
  const out = new Map<string, ResolvedIdentifier | null>();

  const unique = Array.from(new Set(cusips));
  const toQuery: string[] = [];
  for (const c of unique) {
    if (c in cache) out.set(c, cache[c].result);
    else toQuery.push(c);
  }
  if (!toQuery.length) return out;

  const batchSize = opts.apiKey ? KEYED_BATCH_SIZE : KEYLESS_BATCH_SIZE;
  const delayMs = opts.apiKey ? KEYED_DELAY_MS : KEYLESS_DELAY_MS;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (opts.apiKey) headers["X-OPENFIGI-APIKEY"] = opts.apiKey;

  const batches = chunk(toQuery, batchSize);
  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i];
    const jobs = batch.map((cusip) => ({ idType: "ID_CUSIP", idValue: cusip }));
    const resp = await fetchImpl(OPENFIGI_URL, { method: "POST", headers, body: JSON.stringify(jobs) });
    if (!resp.ok) {
      console.error(`[datacore] cusipResolver: OpenFIGI HTTP ${resp.status} on batch ${i + 1}/${batches.length} — leaving ${batch.length} CUSIP(s) uncached for a future run`);
      continue;
    }
    const body = (await resp.json()) as OpenFigiJobResult[];
    const resolved = parseOpenFigiBatch(batch, body);
    const resolvedAt = nowIso();
    for (const [cusip, result] of resolved) {
      out.set(cusip, result);
      cache[cusip] = { result, resolvedAt };
    }
    saveCusipCache(opts.cachePath, cache);
    if (i < batches.length - 1) await sleep(delayMs);
  }
  return out;
}
