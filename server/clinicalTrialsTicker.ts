/**
 * ClinicalTrials.gov sponsor -> listed-equity join (gate-2 prerequisite (i)
 * of the 2026-10-19 open_questions.md hypothesis).
 *
 * PRECISION-FIRST, two locks, either one rejecting drops the sponsor:
 *   1. EXACT normalized-name match against SEC company_tickers.json using the
 *      shared usaSpending matcher (same suffix stripping, same ambiguity-drops
 *      rule) — no fuzzy matching, ever.
 *   2. SIC SECTOR LOCK: the matched filer's EDGAR SIC must be a
 *      pharma/biotech/research code. Exists because lock 1 alone mapped the
 *      Taiwanese "Polaris Group" to Polaris Inc. (PII, powersports, SIC 3790)
 *      in the 2026-10-19 live run — a name collision, not an identity.
 *
 * Recall is deliberately modest (private, foreign-unlisted and Chinese A/H-share
 * sponsors never appear in company_tickers.json); an unmatched sponsor is
 * dropped from gate 2, never guessed. No trading imports (datacore boundary).
 */
import fs from "node:fs";
import path from "node:path";
import { buildTickerMap, normalizeCompanyName } from "./usaSpending";
import { repoDataPath } from "./repoFiles";

/** Pre-registered: 2833 medicinal chemicals, 2834 pharma preps, 2835 in-vitro
 *  diagnostics, 2836 biologicals, 8731 commercial physical/biological research. */
export const HEALTH_SICS: ReadonlySet<number> = new Set([2833, 2834, 2835, 2836, 8731]);

export type JoinStatus = "matched" | "rejected_sic" | "sic_unknown" | "unmatched";

export interface SponsorJoin {
  sponsor: string;
  status: JoinStatus;
  ticker: string | null;
  cik: number | null;
  sic: number | null;
}

export interface SecTickerRow { cik_str: number; ticker: string; title: string }

/** ticker -> cik from company_tickers.json (first row per ticker wins; tickers
 *  are unique in that file). */
export function tickerToCik(json: Record<string, SecTickerRow> | null | undefined): Map<string, number> {
  const m = new Map<string, number>();
  for (const row of Object.values(json || {})) {
    if (row?.ticker && Number.isFinite(row.cik_str) && !m.has(row.ticker)) m.set(row.ticker, Number(row.cik_str));
  }
  return m;
}

/** Joins sponsor names. `sicOf` returns the EDGAR SIC for a CIK, or null when
 *  it could not be read — an unreadable SIC is `sic_unknown` and NOT matched
 *  (fail closed). Pure given its inputs. */
export async function joinSponsors(
  sponsors: Iterable<string>,
  tickersJson: Record<string, SecTickerRow>,
  sicOf: (cik: number) => Promise<number | null>,
): Promise<SponsorJoin[]> {
  const nameMap = buildTickerMap(tickersJson);
  const cikOf = tickerToCik(tickersJson);
  const seen = new Set<string>();
  const out: SponsorJoin[] = [];
  for (const sponsor of sponsors) {
    if (!sponsor || seen.has(sponsor)) continue;
    seen.add(sponsor);
    const ticker = nameMap.get(normalizeCompanyName(sponsor)) ?? null;
    const cik = ticker ? cikOf.get(ticker) ?? null : null;
    if (!ticker || cik == null) { out.push({ sponsor, status: "unmatched", ticker: null, cik: null, sic: null }); continue; }
    const sic = await sicOf(cik);
    const status: JoinStatus = sic == null ? "sic_unknown" : HEALTH_SICS.has(sic) ? "matched" : "rejected_sic";
    out.push({ sponsor, status, ticker, cik, sic });
  }
  return out;
}

export function joinStats(rows: SponsorJoin[], trialsBySponsor: Map<string, number>) {
  const by = (s: JoinStatus) => rows.filter((r) => r.status === s);
  const trials = (rs: SponsorJoin[]) => rs.reduce((a, r) => a + (trialsBySponsor.get(r.sponsor) || 0), 0);
  const total = trials(rows);
  return {
    sponsors: rows.length,
    matched: by("matched").length,
    rejected_sic: by("rejected_sic").length,
    sic_unknown: by("sic_unknown").length,
    unmatched: by("unmatched").length,
    trials_total: total,
    trials_matched: trials(by("matched")),
    trial_coverage: total ? trials(by("matched")) / total : 0,
  };
}

/** Display-side use of the join (identity only, NO signal claim): sponsor ->
 *  {ticker,cik} for `matched` rows only. rejected_sic / sic_unknown /
 *  unmatched sponsors get no ticker — the display never guesses. */
export interface SponsorMapFile { rows?: SponsorJoin[] }
export type SponsorIndex = Map<string, { ticker: string; cik: number }>;

export function sponsorIndex(file: SponsorMapFile | null | undefined): SponsorIndex {
  const m: SponsorIndex = new Map();
  for (const r of file?.rows || []) {
    if (r?.status === "matched" && r.ticker && r.cik != null) m.set(r.sponsor, { ticker: r.ticker, cik: r.cik });
  }
  return m;
}

export function annotateWithTicker<T extends { sponsor: string }>(
  events: T[],
  idx: SponsorIndex,
): Array<T & { ticker: string | null; cik: number | null }> {
  return events.map((e) => {
    const hit = idx.get(e.sponsor);
    return { ...e, ticker: hit?.ticker ?? null, cik: hit?.cik ?? null };
  });
}

let cachedIndex: SponsorIndex | null = null;
/** Lazy, cached load of the committed snapshot map. Missing/corrupt file
 *  degrades to an empty index (tickers null) and logs — never throws. */
export function loadSponsorIndex(jsonPath?: string): SponsorIndex {
  if (cachedIndex) return cachedIndex;
  const p = jsonPath || repoDataPath(path.join("datacore", "clinical_trials_sponsor_map.json"));
  try {
    cachedIndex = sponsorIndex(JSON.parse(fs.readFileSync(p, "utf-8")));
  } catch (e) {
    console.error("[clinical-trials] sponsor map load:", (e as Error)?.message || e);
    return new Map();
  }
  return cachedIndex;
}

export function _resetSponsorIndexForTests(): void {
  cachedIndex = null;
}
