/**
 * dtcc_resolve_cusips.ts — resolves every US-underlier CUSIP/ISIN currently
 * archived by dtccSwaps.ts to a ticker via cusipResolver.ts (OpenFIGI),
 * writing the result to <archiveBaseDir>/dtccswaps/cusip_ticker_cache.json.
 *
 * This is item (3) from the 2026-08-22 GATE 1 session's NEXT queue
 * (research/experiments.md): "CUSIP-to-ticker resolution for the archived
 * US rows (would enable matching against the bot's actual tradable
 * universe)". Deliberately a standalone re-runnable script, not wired into
 * the boot-time poll — resolution only needs to run once per NEW CUSIP ever
 * observed (the cache makes repeat runs near-free), and OpenFIGI is a
 * third-party dependency this Node process's TRADING LOOP (priority 1,
 * KEEP THE SYSTEM ALIVE) must never block on — same reasoning dtccSwaps.ts
 * itself already documents for why its own poll runs on a 6h timer, not
 * inline with any request path.
 *
 * Usage: npx tsx scripts/dtcc_resolve_cusips.ts [--limit N] [--dry-run]
 *   --limit N   only resolve the first N never-seen CUSIPs this run
 *               (keeps a single run inside OpenFIGI's free-tier rate limit;
 *               re-run to continue — already-cached CUSIPs cost nothing)
 *   --dry-run   print what WOULD be queried, make no network calls
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "../server/datacoreArchive";
import { cusipFromUsIsin, type DtccSwapRow } from "../server/dtccSwaps";
import { resolveCusips, loadCusipCache } from "../server/cusipResolver";

function dtccDir(): string {
  return path.join(archiveBaseDir(), "dtccswaps");
}

export function cusipCachePath(): string {
  return path.join(dtccDir(), "cusip_ticker_cache.json");
}

/** Every archived row's underlierId is already known-US (dtccSwaps.ts's
 *  isUsUnderlier filter ran before archiving) — this just normalizes CUSIP
 *  vs ISIN sourcing down to one 9-char CUSIP per row. */
export function cusipForRow(row: Pick<DtccSwapRow, "underlierId" | "underlierIdSource">): string | null {
  if (row.underlierIdSource === "CUSIP") return row.underlierId;
  if (row.underlierIdSource === "ISIN") return cusipFromUsIsin(row.underlierId);
  return null;
}

export function readArchivedCusips(dir: string): string[] {
  const seen = new Set<string>();
  let files: string[];
  try { files = fs.readdirSync(dir); } catch { return []; }
  let malformedLines = 0;
  for (const f of files) {
    if (!f.endsWith(".jsonl.gz") && !f.endsWith(".jsonl")) continue;
    const raw = f.endsWith(".gz")
      ? zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString("utf8")
      : fs.readFileSync(path.join(dir, f), "utf8");
    for (const line of raw.split("\n")) {
      if (!line) continue;
      try {
        const row = JSON.parse(line) as DtccSwapRow;
        const cusip = cusipForRow(row);
        if (cusip) seen.add(cusip);
      } catch {
        malformedLines++; // dtccSwaps.ts's own loader already surfaces these on write; count here too rather than swallow (MASTER PROGRAM D4 — a silent comment-only catch looks considered but is not)
      }
    }
  }
  if (malformedLines) {
    console.error(`[datacore] dtcc_resolve_cusips readArchivedCusips: ${malformedLines} malformed archive line(s) skipped`);
  }
  return Array.from(seen);
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run");
  const limitIdx = args.indexOf("--limit");
  const limit = limitIdx >= 0 ? Number(args[limitIdx + 1]) : undefined;

  const dir = dtccDir();
  const cachePath = cusipCachePath();
  const all = readArchivedCusips(dir);
  const cache = loadCusipCache(cachePath);
  const unresolved = all.filter((c) => !(c in cache));
  const toRun = limit ? unresolved.slice(0, limit) : unresolved;

  console.log(`archived distinct US CUSIPs: ${all.length}`);
  console.log(`already cached: ${all.length - unresolved.length}`);
  console.log(`never-seen this run: ${unresolved.length}${limit ? ` (running ${toRun.length} per --limit)` : ""}`);

  if (dryRun) {
    console.log(toRun.slice(0, 20).join("\n"));
    return;
  }
  if (!toRun.length) { console.log("nothing to resolve."); return; }

  const apiKey = process.env.OPENFIGI_API_KEY;
  const result = await resolveCusips(toRun, { cachePath, apiKey });
  const resolved = Array.from(result.values()).filter(Boolean).length;
  console.log(`resolved this run: ${resolved}/${toRun.length} (${toRun.length - resolved} confirmed no-mapping, now cached negative)`);
}

// Guarded so dtcc_resolve_cusips.test.ts can import the pure functions above
// without triggering a live run (and its network calls) as an import side effect.
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
