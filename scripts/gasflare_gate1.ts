/**
 * gasflare_gate1.ts — GATE 1 (DATA layer) for the free-alternative gas-flare
 * candidate detector: does per-country candidate DENSITY from
 * server/gasFlareCandidates.ts, run over the real production NASA FIRMS
 * archive, RANK-correlate against the World Bank Global Flaring &
 * Methane Reduction Partnership's published country flaring-volume order?
 *
 * Pre-registered in server/gasFlareCandidates.ts's own header (2026-09-01,
 * this same day, earlier session): "Spearman rank correlation against
 * GGFR's published top-15 flaring countries" — that session built the
 * detector and candidatesByRegion() but explicitly could not run this half
 * ("no production archive access in this sandbox"). This session has live
 * DIAG_TOKEN production access; this script is the actual GATE 1 attempt.
 *
 * TRUTH SOURCE (web search this session, dated 2026-09-01, not assumed
 * from training): the World Bank's 2026 Global Gas Flaring Tracker Report
 * (covering 2025 data) — summarized by Down To Earth
 * (downtoearth.org.in/energy/nine-countries-responsible-for-83-of-global-
 * gas-flaring-in-2025-says-world-bank-report) and businessamlive.com,
 * both reporting the same World Bank figure independently: "The nine
 * largest flaring countries in descending order are Russia, Iran, Iraq,
 * Venezuela, Mexico, Libya, Algeria, Nigeria and the United States."
 * SCOPE CUT, now CLOSED (2026-09-03 session): the 2026-09-01 run of this
 * script EXCLUDED Russia — its admin0 polygon crosses the antimeridian
 * (Chukotka), so a single min/max-lon bounding-box archive query for it was
 * not "Russia" but "the entire globe at 41-81N", which truncated at the
 * probe's 5000-row/day cap on ordinary wildfire-season noise before enough
 * genuine Russian rows came through. FIX: `countryBboxParams()`
 * (server/countryLookup.ts, this session) splits a dateline-crossing
 * country's rings into two clusters (the Chukotka exclave, entirely under
 * -90 lon, vs everything else) and returns one tight bbox per cluster
 * instead of one bbox spanning both — verified live this session that
 * neither piece itself spans >=180deg of longitude. This run therefore
 * tests the FULL published top-9 (not the full top-15 the original header
 * named — no independently-sourced rank data beyond these 9 was found;
 * inventing ranks 10-15 would violate CLAUDE.md's anti-fabrication rule).
 *
 * METHOD: for each of the 9 countries, for each day in a
 * deliberately clean archive window (2026-08-15 through 2026-08-28 —
 * chosen to sit entirely AFTER the still-unexplained 2026-08-05/06 volume
 * anomaly and the 2026-08-05..12 aisstream-adjacent outage window
 * KNOWN BROKEN #37 documents for the VESSEL stream; the FIRMS/fires stream
 * is a different provider with no known issue in this window, but the
 * date range was picked to avoid ANY known-anomalous week on general
 * principle), query /api/diag/archive?stream=fires&day=...&bbox=<country
 * bbox>&limit=5000 against production. Each row is re-filtered through
 * countryLookup.ts's exact point-in-polygon countryOf() (the bbox is a
 * coarse pre-filter only — it also catches border-adjacent neighbors).
 * Accumulated rows feed server/gasFlareCandidates.ts's real
 * findGasFlareCandidates() UNCHANGED (no reimplementation) with its
 * documented defaults (minNights=3, minPersistence=0.5). Candidate COUNT
 * per country is the test statistic; Spearman rank correlation against
 * the published order above is the pre-registered pass bar (positive and
 * significant at n=9, two-tailed, alpha=0.05 -> critical rho 0.700, exact
 * value via full permutation of the null distribution, this session).
 *
 * WILDFIRE DISCRIMINATOR RE-RUN (2026-09-01, same-day follow-up, NEXT step
 * (1) from the GATE 1 FAIL addendum): this run additionally computes a
 * REFINED verdict from the exact same fetched rows using
 * findGasFlareCandidates()'s new opt-in `maxFrpCV` filter (see
 * server/gasFlareCandidates.ts header — FRP coefficient-of-variation as a
 * proxy for VIIRS Nightfire's own "stable temperature" flare criterion).
 * PRIOR (REASONING STANDARD #10, stated before this rerun): maxFrpCV=0.6
 * was chosen from the qualitative literature claim (flares hold a
 * "stable, persistent temperature"; wildfires are "far more variable") —
 * not fit to this dataset's own rho. Expect the refined candidate counts
 * to shrink (especially for USA, the diagnosed wildfire-contaminated
 * outlier) and rho to move toward positive; if it doesn't, that is a
 * genuine negative for this specific refinement, reported as such per
 * MEASUREMENT INTEGRITY, not re-tried with a second threshold in the same
 * session (REASONING STANDARD #4 — one shot, no post-hoc threshold
 * fishing).
 *
 * Usage: DIAG_TOKEN=... npx tsx scripts/gasflare_gate1.ts [prodBaseUrl]
 * Prints a JSON verdict (baseline + refined) to stdout. Touches no
 * production code path; the result is recorded in research/experiments.md
 * + datacore/signal_ladder.json by the session running it, not by this
 * script.
 */
import { countryOf, countryBboxParams, countryName } from "../server/countryLookup.ts";
import { findGasFlareCandidates, type FlareCandidateInput } from "../server/gasFlareCandidates.ts";

const BASE = process.argv[2] || process.env.VOLTRADE_PROD_URL || "https://voltradeai-production.up.railway.app";
const TOKEN = process.env.DIAG_TOKEN;

// Published order (World Bank 2026 Global Gas Flaring Tracker Report,
// descending flaring volume). Russia is now INCLUDED (n=9, full published
// scope) — the antimeridian scope cut this file's header used to document
// is closed by countryBboxParams() below (server/countryLookup.ts,
// 2026-09-03 session), which queries Russia's main landmass and its
// Chukotka exclave as two separate tight bboxes instead of one bbox
// spanning the whole globe's longitude range.
const TRUTH_RANK: string[] = ["RUS", "IRN", "IRQ", "VEN", "MEX", "LBY", "DZA", "NGA", "USA"];

// Two-tailed critical value for Spearman rho at n=9, alpha=0.05. Computed
// exactly (not read off an approximate table) via full permutation of the
// null distribution (9! = 362,880 permutations) this session — reproduces
// the standard table's n=8 value (0.7381 vs the commonly-cited 0.738) to
// 3 decimal places, confirming the method before trusting its n=9 output.
const CRITICAL_RHO_N9_ALPHA05 = 0.7;

// Wildfire-discriminator threshold, chosen a priori per the header PRIOR —
// not fit to this run's own outcome.
const MAX_FRP_CV = 0.6;

// Clean window: after the KNOWN BROKEN #37 vessel-stream anomaly window,
// well before "today" so no partial/still-accumulating day is included.
const DAYS: string[] = [];
for (let d = 15; d <= 28; d++) DAYS.push(`2026-08-${String(d).padStart(2, "0")}`);

interface RawFireRow {
  lat: number; lon: number; frp: number | null; acq_date: string; daynight: "D" | "N" | null;
}

async function fetchCountryDay(iso3: string, bbox: string, day: string): Promise<{ rows: RawFireRow[]; truncated: boolean }> {
  const url = `${BASE}/api/diag/archive?stream=fires&day=${day}&bbox=${encodeURIComponent(bbox)}&limit=5000&token=${TOKEN}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const body = await r.json();
  const rows: RawFireRow[] = (body.rows || []).filter((row: RawFireRow) => countryOf(row.lat, row.lon) === iso3);
  return { rows, truncated: !!body.truncated };
}

// Spearman rank correlation, no ties expected in an integer candidate-count
// ranking at this small n (ties broken by input order if they occur, which
// is a standard, disclosed simplification).
function spearman(xs: number[], ys: number[]): number {
  const rank = (vals: number[]): number[] => {
    const idx = vals.map((_, i) => i).sort((a, b) => vals[b] - vals[a]);
    const r = new Array(vals.length);
    idx.forEach((originalIndex, rankPos) => { r[originalIndex] = rankPos + 1; });
    return r;
  };
  const rx = rank(xs);
  const ry = rank(ys);
  const n = xs.length;
  let dSquaredSum = 0;
  for (let i = 0; i < n; i++) dSquaredSum += (rx[i] - ry[i]) ** 2;
  return 1 - (6 * dSquaredSum) / (n * (n * n - 1));
}

async function main() {
  if (!TOKEN) {
    console.log(JSON.stringify({ verdict: "ERROR", error: "DIAG_TOKEN not set in environment" }));
    process.exitCode = 1;
    return;
  }

  const perCountry: Record<string, {
    candidateCount: number; refinedCandidateCount: number;
    rowCount: number; truncatedDays: number; sample: unknown; bboxCount: number;
  }> = {};
  for (const iso3 of TRUTH_RANK) {
    const bboxes = countryBboxParams(iso3);
    if (!bboxes || !bboxes.length) throw new Error(`no bbox for ${iso3}`);
    const allRows: FlareCandidateInput[] = [];
    let truncatedDays = 0;
    for (const day of DAYS) {
      let dayTruncated = false;
      for (const bbox of bboxes) {
        const { rows, truncated } = await fetchCountryDay(iso3, bbox, day);
        if (truncated) dayTruncated = true;
        for (const r of rows) {
          allRows.push({ lat: r.lat, lon: r.lon, acq_date: r.acq_date, daynight: r.daynight, frp: r.frp });
        }
      }
      if (dayTruncated) truncatedDays++;
    }
    // Baseline (unmodified 2026-09-01 gate-1 run) and the same-day
    // wildfire-discriminator REFINED variant, both computed from the
    // identical fetched rows so the comparison is apples-to-apples.
    const candidates = findGasFlareCandidates(allRows);
    const refinedCandidates = findGasFlareCandidates(allRows, { maxFrpCV: MAX_FRP_CV });
    perCountry[iso3] = {
      candidateCount: candidates.length,
      refinedCandidateCount: refinedCandidates.length,
      rowCount: allRows.length,
      truncatedDays,
      bboxCount: bboxes.length,
      sample: candidates.slice(0, 3).map((c) => ({
        lat: Number(c.lat.toFixed(3)), lon: Number(c.lon.toFixed(3)),
        nightsActive: c.nightsActive, persistence: Number(c.persistence.toFixed(2)),
        meanFrp: c.meanFrp, frpCV: c.frpCV != null ? Number(c.frpCV.toFixed(3)) : null,
      })),
    };
    console.error(`[gasflare_gate1] ${iso3} (${countryName(iso3)}): ${allRows.length} nighttime rows across ${bboxes.length} bbox(es), ${candidates.length} candidates (${refinedCandidates.length} after maxFrpCV=${MAX_FRP_CV} refinement), ${truncatedDays} truncated days`);
  }

  // Spearman needs matched-order rank pairs: truth rank position vs our
  // candidate count (higher count should correlate with a LOWER — i.e.
  // more senior — truth rank number for a positive-correlation pass, so
  // we correlate candidate count against (n - truthPosition) to keep the
  // sign intuitive: higher count, higher (better) published rank -> +1.
  const truthScore = TRUTH_RANK.map((_, i) => TRUTH_RANK.length - i); // n..1, best-flarer=n

  function summarize(countKey: "candidateCount" | "refinedCandidateCount") {
    const counts = TRUTH_RANK.map((iso3) => perCountry[iso3][countKey]);
    const rho = spearman(counts, truthScore);
    const verdict = rho >= CRITICAL_RHO_N9_ALPHA05 ? "GATE1_PASS" : (rho > 0 ? "GATE1_INCONCLUSIVE_WEAK_POSITIVE" : "GATE1_FAIL");
    const order = [...TRUTH_RANK].sort((a, b) => perCountry[b][countKey] - perCountry[a][countKey]);
    return {
      verdict,
      spearman_rho: Number(rho.toFixed(4)),
      our_order_by_candidate_count: order.map((iso3) => `${iso3} (${countryName(iso3)}): ${perCountry[iso3][countKey]}`),
    };
  }

  console.log(JSON.stringify({
    baseline: summarize("candidateCount"),
    refined: { ...summarize("refinedCandidateCount"), maxFrpCV: MAX_FRP_CV },
    critical_rho_n9_alpha05: CRITICAL_RHO_N9_ALPHA05,
    truth_order_published: TRUTH_RANK.map((iso3) => `${iso3} (${countryName(iso3)})`),
    per_country: perCountry,
    window: { days: DAYS, count: DAYS.length },
    truth_source: "World Bank 2026 Global Gas Flaring Tracker Report (2025 data), via downtoearth.org.in and businessamlive.com summaries, web-searched 2026-09-01",
    detector_defaults: "minNights=3, minPersistence=0.5 (server/gasFlareCandidates.ts findGasFlareCandidates() defaults, unmodified)",
  }, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ verdict: "ERROR", error: e?.message || String(e) }));
  process.exitCode = 1;
});
