/**
 * dtcc_cap_marks_seed.ts — read the source's cap mark ("250,000,000+") out of
 * dated DTCC vendor files from a session machine and write the seed the
 * server folds into its cap-mark record (server/dtccCapMarks.ts,
 * open_questions.md (bc1)).
 *
 * WHY A SESSION RUNS THIS: the vendor stops serving a dated file two years
 * after it is written (server/dtccSwaps.ts THE EDGE). The archive's rows for
 * the days at that limit were read before v1.0.1572 and hold null where the
 * source wrote the mark. A day that leaves the window before a deployed
 * server re-reads it loses its marks for good; a session can read it now.
 *
 * It uses the server's own reader and parser (streamDtccZip, parseHeader,
 * parseDtccLine), so "marked" means exactly what the archive means by it:
 * US-underlier rows only, the mark on Notional amount-Leg 1.
 *
 * Usage:
 *   npx tsx scripts/dtcc_cap_marks_seed.ts --out datacore/dtcc_cap_marks_seed.json \
 *       2024-11-26 2024-11-27            # from the vendor
 *   npx tsx scripts/dtcc_cap_marks_seed.ts --out ... --dir /path/to/zips 2024-11-26
 *       # from files already downloaded, named SEC_YYYY_MM_DD.zip
 *   add --sample <file> to also write one real line of each notional form
 *   (header first) from the first day read: the parser's test fixture.
 * Days already in --out are kept; a day read again replaces its entry.
 */
import fs from "fs";
import path from "path";
import { Readable } from "stream";
import { dtccUrl, streamDtccZip, parseHeader, parseCsvLine, parseDtccLine, dtccNotionalClass, type DtccNotionalClass } from "../server/dtccSwaps";
import type { DtccCapMarksSeed } from "../server/dtccCapMarks";

function arg(name: string): string | null {
  const i = process.argv.indexOf(name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : null;
}

async function main() {
  const out = arg("--out");
  const dir = arg("--dir");
  const samplePath = arg("--sample");
  const flagValues = new Set([out, dir, samplePath].filter(Boolean) as string[]);
  const days = process.argv.slice(2).filter((a) => /^\d{4}-\d{2}-\d{2}$/.test(a) && !flagValues.has(a));
  if (!out || !days.length) {
    console.error("usage: dtcc_cap_marks_seed.ts --out <seed.json> [--dir <zips>] [--sample <file>] YYYY-MM-DD ...");
    process.exit(2);
  }
  let seed: DtccCapMarksSeed & Record<string, unknown> = { days: {} };
  if (fs.existsSync(out)) seed = JSON.parse(fs.readFileSync(out, "utf8"));
  const fileFetch = async (url: string) => {
    const m = /EQUITIES_(\d{4}_\d{2}_\d{2})\.zip/.exec(url);
    const fp = path.join(dir!, `SEC_${m![1]}.zip`);
    if (!fs.existsSync(fp)) return { ok: false, status: 404, body: null };
    return { ok: true, status: 200, body: Readable.toWeb(fs.createReadStream(fp)) as any };
  };
  let sampled = false;
  for (const day of days) {
    let idx: Record<string, number> | null = null;
    let expectedCols = 0;
    let usRows = 0;
    const ids: string[] = [];
    let header = "";
    const sample: Partial<Record<DtccNotionalClass, string>> = {};
    const { ok, status, lines } = await streamDtccZip(dtccUrl(day.replace(/-/g, "_")), (line, isHeader) => {
      if (isHeader) { idx = parseHeader(line); expectedCols = parseCsvLine(line).length; header = line; return; }
      if (!idx) return;
      const row = parseDtccLine(line, idx, expectedCols);
      if (!row) return;
      usRows++;
      if (row.notionalCappedLeg1) ids.push(row.disseminationId);
      const cls = dtccNotionalClass(row);
      if (!sample[cls]) sample[cls] = line;
    }, dir ? (fileFetch as any) : undefined);
    if (!ok) {
      console.error(`${day}: not read (HTTP ${status}) — left out of the seed`);
      continue;
    }
    seed.days[day] = { usRows, ids: Array.from(new Set(ids)).sort(), readAt: new Date().toISOString() };
    console.log(`${day}: ${lines} lines, ${usRows} US-underlier rows, ${seed.days[day].ids.length} with the cap mark`);
    if (samplePath && !sampled) {
      const order: DtccNotionalClass[] = ["usd", "usd_capped", "other_currency", "over_cap_unmarked", "blank"];
      const kept = order.filter((c) => sample[c]);
      fs.writeFileSync(samplePath, [header, ...kept.map((c) => sample[c]!)].join("\n") + "\n");
      console.log(`sample: ${kept.join(", ")} (vendor file dated ${day}) -> ${samplePath}`);
      sampled = true;
    }
  }
  const sorted: DtccCapMarksSeed["days"] = {};
  for (const k of Object.keys(seed.days).sort()) sorted[k] = seed.days[k];
  seed.days = sorted;
  fs.writeFileSync(out, JSON.stringify(seed, null, 1) + "\n");
}

main().catch((e) => { console.error(e); process.exit(1); });
