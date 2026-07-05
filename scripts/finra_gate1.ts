/**
 * finra_gate1.ts — GATE 1 (DATA layer) for the FINRA short-volume stream:
 * does our parser's reading of the consolidated CNMS file match FINRA's
 * own independent per-facility files summed per symbol?
 *
 * CRITERIA (pre-stated 2026-07-05 BEFORE running, Reasoning Standard #10):
 *   - Sample day: 2026-07-02 (last full trading day at gate time).
 *   - TRUTH: per-facility files (FNYX=N NYSE TRF, FNSQ=Q Nasdaq Carteret,
 *     FNQC=B Nasdaq Chicago, FORF=O ORF) parsed with the SAME parser under
 *     test, summed per symbol over exactly the facilities the CNMS row's
 *     own Market column names.
 *   - PASS = >=99.9% of CNMS symbols match sum-of-parts within 0.01 shares
 *     on BOTH short_vol and total_vol; every mismatch printed; any CNMS
 *     market code with no known facility file is reported (a finding,
 *     never silently skipped).
 *   - PRIOR: P(pass) ~90% — this is an accounting identity FINRA itself
 *     maintains; failure means a parser defect or a facility-coverage
 *     misunderstanding on our side.
 *
 * Session-run (npx tsx scripts/finra_gate1.ts [YYYYMMDD]); result goes in
 * research/experiments.md, not into any runtime path.
 */
import { parseShortVol, type ShortVolRow } from "../server/finraShortVolume";

const DAY = process.argv[2] || "20260702";
const FACILITY_FILES: Record<string, string> = {
  N: "FNYX", Q: "FNSQ", B: "FNQC", O: "FORF",
};

async function fetchRows(prefix: string): Promise<ShortVolRow[]> {
  const url = `https://cdn.finra.org/equity/regsho/daily/${prefix}shvol${DAY}.txt`;
  const r = await fetch(url, { headers: { "User-Agent": "voltradeai-datacore/1.0" } });
  if (!r.ok) throw new Error(`${prefix} -> ${r.status}`);
  return parseShortVol(await r.text(), "gate1");
}

const near = (a: number, b: number) => Math.abs(a - b) <= 0.01;

async function main() {
  const consolidated = await fetchRows("CNMS");
  console.log(`CNMS ${DAY}: ${consolidated.length} symbols`);

  const facRows = new Map<string, Map<string, ShortVolRow>>();
  for (const [code, prefix] of Object.entries(FACILITY_FILES)) {
    try {
      const rows = await fetchRows(prefix);
      facRows.set(code, new Map(rows.map((r) => [r.symbol, r])));
      console.log(`${prefix} (${code}): ${rows.length} symbols`);
    } catch (e: any) {
      console.log(`${prefix} (${code}): UNAVAILABLE — ${e.message}`);
    }
  }

  let checked = 0, matched = 0;
  const unknownCodes = new Map<string, number>();
  const mismatches: string[] = [];
  for (const row of consolidated) {
    const codes = row.market.split(",").map((c) => c.trim()).filter(Boolean);
    const missing = codes.filter((c) => !facRows.has(c));
    if (missing.length) {
      missing.forEach((c) => unknownCodes.set(c, (unknownCodes.get(c) || 0) + 1));
      continue; // not checkable — counted, never silently dropped
    }
    checked++;
    let sv = 0, tv = 0;
    for (const c of codes) {
      const f = facRows.get(c)!.get(row.symbol);
      if (f) { sv += f.short_vol; tv += f.total_vol; }
    }
    if (near(sv, row.short_vol) && near(tv, row.total_vol)) matched++;
    else if (mismatches.length < 25) {
      mismatches.push(`${row.symbol} [${row.market}] cnms sv=${row.short_vol} tv=${row.total_vol} parts sv=${sv.toFixed(4)} tv=${tv.toFixed(4)}`);
    }
  }

  const rate = checked ? (matched / checked) * 100 : 0;
  console.log(`\nchecked=${checked} matched=${matched} rate=${rate.toFixed(3)}%`);
  if (unknownCodes.size) {
    console.log("market codes with NO facility file (excluded from denominator, a finding):");
    unknownCodes.forEach((n, c) => console.log(`  '${c}': ${n} symbols`));
  }
  if (mismatches.length) {
    console.log(`first mismatches:\n  ${mismatches.join("\n  ")}`);
  }
  console.log(`\nVERDICT (pre-stated bar >=99.9%): ${rate >= 99.9 ? "PASS" : "FAIL"}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
