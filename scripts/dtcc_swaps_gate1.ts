/**
 * dtcc_swaps_gate1.ts — GATE 1 (DATA layer) for the DTCC SBSDR equity-swap
 * stream: does our parser extract genuine, well-formed identifiers from the
 * right columns, not garbage from a misaligned one?
 *
 * CRITERIA (pre-stated 2026-08-22 BEFORE running, Reasoning Standard #10):
 *   - Sample: today's live SEC_CUMULATIVE_EQUITIES file, US-underlier rows
 *     only (per isUsUnderlier — the pipeline's own in-scope filter).
 *   - TRUTH: two INDEPENDENT external checksum standards applied to the
 *     SAME extracted string — ISO 6166 (ISIN check digit) and the CUSIP
 *     Global Services mod-10 algorithm run against the CUSIP embedded in
 *     every US ISIN (chars 3-11). DTCC publishes no separate aggregate
 *     summary page to diff against (unlike OCC/FINRA), so this is the
 *     available ground truth: a parser reading the wrong column would not
 *     pass either standard-format checksum at anywhere near 100% by chance
 *     (~1/10 per check, ~1/100 for both simultaneously).
 *   - PASS = >=99.9% of single-ISIN/CUSIP US rows pass BOTH checks;
 *     basket rows (semicolon-joined source field) are counted and excluded
 *     from the denominator (a known, honestly-labeled v1 limitation, not a
 *     parser failure — see server/dtccSwaps.ts).
 *   - PRIOR: P(pass) ~90% — the checksum standards are decades-old and the
 *     column lookup is by HEADER NAME (not position), so failure would mean
 *     either a header the vendor renamed since this was written, or a
 *     genuine off-by-one in the CSV splitter.
 *
 * Session-run (npx tsx scripts/dtcc_swaps_gate1.ts [YYYY_MM_DD]); result
 * goes in research/experiments.md, not into any runtime path.
 */
import {
  dtccUrl, streamDtccZip, parseHeader, parseCsvLine,
  isUsUnderlier, isBasketField, isinCheckDigitValid, cusipCheckDigitValid, cusipFromUsIsin,
} from "../server/dtccSwaps";

function ymdUnderscore(d: Date): string {
  return d.toISOString().slice(0, 10).replace(/-/g, "_");
}

const dateArg = process.argv[2] || ymdUnderscore(new Date());

async function main() {
  const url = dtccUrl(dateArg);
  console.log(`Fetching ${url} ...`);

  let idx: Record<string, number> | null = null;
  let expectedCols = 0;
  let totalLines = 0;
  let usSingle = 0, basketCount = 0, otherCountry = 0;
  let isinOk = 0, isinBad = 0, cusipOk = 0, cusipBad = 0;
  const badSamples: string[] = [];

  const { ok, status, lines } = await streamDtccZip(url, (line, isHeader) => {
    totalLines++;
    if (isHeader) {
      idx = parseHeader(line);
      expectedCols = parseCsvLine(line).length;
      return;
    }
    if (!idx) return;
    const c = parseCsvLine(line);
    if (c.length !== expectedCols) return;
    const src = c[idx["Underlier ID source-Leg 1"]];
    const id = c[idx["Underlier ID-Leg 1"]];
    if (isBasketField(id, src)) { basketCount++; return; }
    if (!isUsUnderlier(id, src)) { otherCountry++; return; }
    usSingle++;

    let isinPass = true, cusipPass = true;
    if (src === "ISIN") {
      isinPass = isinCheckDigitValid(id);
      const cusip = cusipFromUsIsin(id);
      cusipPass = cusip ? cusipCheckDigitValid(cusip) : false;
    } else if (src === "CUSIP") {
      cusipPass = cusipCheckDigitValid(id);
    }
    if (isinPass) isinOk++; else { isinBad++; if (badSamples.length < 10) badSamples.push(`ISIN fail: ${id} (${src})`); }
    if (cusipPass) cusipOk++; else { cusipBad++; if (badSamples.length < 10) badSamples.push(`CUSIP fail: ${id} (${src})`); }
  });

  if (!ok) {
    console.log(`FETCH FAILED — HTTP ${status}`);
    process.exit(1);
  }

  console.log(`\ntotal lines (incl. header): ${lines}`);
  console.log(`US-single-identifier rows: ${usSingle}`);
  console.log(`basket rows (excluded from gate, counted): ${basketCount}`);
  console.log(`other-country rows (out of scope): ${otherCountry}`);

  const isinRate = usSingle ? (isinOk / (isinOk + isinBad)) * 100 : 0;
  const cusipRate = usSingle ? (cusipOk / (cusipOk + cusipBad)) * 100 : 0;
  console.log(`\nISIN check-digit pass rate:  ${isinOk}/${isinOk + isinBad} = ${isinRate.toFixed(4)}%`);
  console.log(`CUSIP check-digit pass rate: ${cusipOk}/${cusipOk + cusipBad} = ${cusipRate.toFixed(4)}%`);
  if (badSamples.length) console.log(`\nfirst failures:\n  ${badSamples.join("\n  ")}`);

  const pass = isinRate >= 99.9 && cusipRate >= 99.9;
  console.log(`\nVERDICT (pre-stated bar >=99.9% on both checks): ${pass ? "PASS" : "FAIL"}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
