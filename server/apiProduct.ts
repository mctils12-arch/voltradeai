/**
 * apiProduct.ts — the /api/v1 data-product foundation (throughput/API
 * directive 2026-07-04). Everything buildable PRE-REVENUE: versioned read
 * endpoints over the datacore archives, API-key auth scaffolding, per-key
 * rate limits, and usage metering from day one. The last mile (key sales,
 * billing, pricing enablement) waits for the human's go on the
 * MONETIZATION READINESS CHECKLIST (wishlist.md) — nothing here charges,
 * bills, or gates payment.
 *
 * Pre-revenue key issuance: keys are ENV-SEEDED only (API_PRODUCT_KEYS =
 * "key:label:tier,key2:label2:tier2"). No signup flow exists on purpose —
 * issuance binds to billing later, per the checklist.
 *
 * LICENSE MARKS: every response envelope names the license of what it
 * carries (the resell-vs-display audit in wishlist.md) — ODbL share-alike
 * for aircraft-derived data, conditional for AIS-derived, public-domain
 * for US-gov streams. Endpoints over gated SIGNALS (tank-fill, entity
 * timelines) do not exist until their ladder gates pass.
 *
 * Pure module: no express, no db imports (the auth.ts import hangs the
 * test runner — standing rule); storage/metering writers injected or
 * fs-appended under the archive base like every other stream.
 */
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { archiveBaseDir } from "./datacoreArchive";

export type ApiTier = "dev" | "pro" | "enterprise";
export interface ApiKeyInfo { label: string; tier: ApiTier }

/** Per-tier limits — {perMinute, perDay}. Enterprise is contract-shaped
 *  later; scaffolding keeps it finite so a leaked key can't melt the box. */
export const TIER_LIMITS: Record<ApiTier, { perMinute: number; perDay: number }> = {
  dev: { perMinute: 60, perDay: 10_000 },
  pro: { perMinute: 600, perDay: 100_000 },
  enterprise: { perMinute: 3_000, perDay: 1_000_000 },
};

export function parseApiKeys(env: NodeJS.ProcessEnv = process.env): Map<string, ApiKeyInfo> {
  const out = new Map<string, ApiKeyInfo>();
  for (const part of (env.API_PRODUCT_KEYS || "").split(",")) {
    const [key, label, tier] = part.trim().split(":");
    if (!key || key.length < 12) continue; // refuse trivially guessable keys
    out.set(key, { label: label || "unlabeled", tier: (["dev", "pro", "enterprise"].includes(tier) ? tier : "dev") as ApiTier });
  }
  return out;
}

/** Sliding-window limiter (minute + day windows per key). Pure/testable:
 *  caller supplies now. */
export function makeRateLimiter() {
  const hits = new Map<string, number[]>();
  return {
    allow(key: string, tier: ApiTier, now = Date.now()): { ok: boolean; retryAfterSec?: number } {
      const lim = TIER_LIMITS[tier];
      const arr = (hits.get(key) || []).filter((t) => now - t < 86_400_000);
      const lastMin = arr.filter((t) => now - t < 60_000);
      if (lastMin.length >= lim.perMinute) return { ok: false, retryAfterSec: 60 };
      if (arr.length >= lim.perDay) return { ok: false, retryAfterSec: 3600 };
      arr.push(now);
      hits.set(key, arr);
      return { ok: true };
    },
    size() { return hits.size; },
  };
}

/** Keys never land raw in logs/archives — hash for metering identity. */
export function keyId(key: string): string {
  return crypto.createHash("sha256").update(key).digest("hex").slice(0, 12);
}

/** Usage metering: day-JSONL under the archive base (a stream like any
 *  other — manifested in datacore/manifests/apiusage.json). */
export function meterUsage(rec: { key: string; endpoint: string; status: number; tier: ApiTier },
                           baseDir?: string, nowMs?: number): void {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "apiusage");
  const now = nowMs ?? Date.now();
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`),
      JSON.stringify({ t: Math.floor(now / 1000), k: keyId(rec.key), e: rec.endpoint, s: rec.status, ti: rec.tier }) + "\n");
  } catch {}
}

/** The resell-vs-display audit, applied per endpoint (wishlist checklist
 *  item 2). Every v1 response carries its mark — honesty travels with the
 *  data. */
export const LICENSE_MARKS: Record<string, { license: string; attribution: string; resell: "ok" | "share-alike" | "conditional" }> = {
  "tracks/aircraft": {
    license: "ODbL 1.0 share-alike (adsb.lol-derived database) + non-commercial fallback sources until the monetization switch",
    attribution: "adsb.lol + airplanes.live + adsb.fi",
    resell: "share-alike",
  },
  "tracks/vessels": {
    license: "aisstream.io terms (redistribution CONDITIONAL — re-verify at monetization switch)",
    attribution: "aisstream.io",
    resell: "conditional",
  },
  "tracks/trains": {
    license: "CC BY 4.0 (Digitraffic) + NLOD (Entur)",
    attribution: "Digitraffic Finland + Entur Norway",
    resell: "ok",
  },
  "stats/portdwell": {
    license: "derived from AIS positions — inherits aisstream conditionality",
    attribution: "VolTradeAI datacore over aisstream.io",
    resell: "conditional",
  },
  "stats/shadow": {
    license: "derived from AIS positions — inherits aisstream conditionality",
    attribution: "VolTradeAI datacore over aisstream.io",
    resell: "conditional",
  },
  "stats/archive": {
    license: "VolTradeAI operational metadata",
    attribution: "VolTradeAI datacore",
    resell: "ok",
  },
  graph: {
    license: "derived from EDGAR Form 4 filings, entity_map, and our own AIS position archive — inherits aisstream conditionality via calls_at edges",
    attribution: "VolTradeAI datacore (SEC EDGAR, GEM ownership, aisstream.io)",
    resell: "conditional",
  },
  "stats/plant-operations": {
    license: "U.S. EPA Clean Air Markets Division (CAMD) unit-level CEMS reporting — public domain (US federal government work)",
    attribution: "U.S. EPA Clean Air Markets Division (CAMD)",
    resell: "ok",
  },
  "stats/secftd": {
    license: "U.S. SEC CNS fails-to-deliver, half-month files — public domain (US federal government work)",
    attribution: "U.S. Securities and Exchange Commission (CNS fails-to-deliver)",
    resell: "ok",
  },
  "stats/midas": {
    license: "U.S. SEC MIDAS individual-security market-structure metrics, quarterly files — public domain (US federal government work)",
    attribution: "U.S. Securities and Exchange Commission (MIDAS)",
    resell: "ok",
  },
  "stats/occ-volume": {
    license: "The Options Clearing Corporation (OCC) daily cleared-volume report — OCC informational-use terms, NOT US government work product like the CAMD/FTD/MIDAS streams above; redistribution needs OCC permission (bot-internal display + gated derived signals accepted, raw bulk resale needs separate OCC review, unresolved).",
    attribution: "The Options Clearing Corporation (OCC) daily volume",
    resell: "conditional",
  },
  "data/earnings-language": {
    license: "SEC EDGAR 8-K Item 2.02 filing record is public, but each filing's Exhibit 99 press-release TEXT is issuer-authored — NOT U.S. government work product like the CAMD/FTD/MIDAS datasets above. Displayed as-filed for research/transparency use; bulk resale of the extracted text has not been separately rights-cleared.",
    attribution: "Filing company (per record) via SEC EDGAR",
    resell: "conditional",
  },
  "data/appstore-rankings": {
    license: "Apple marketingtools RSS top-chart JSON + iTunes Lookup rating counts — public feeds, CONDITIONAL on low-volume internal use (research/open_questions.md NEW DATA ROOTS #3); a metered, resold-to-external-customers mirror has not been separately confirmed as still within that low-volume terms-of-use envelope, so this stays marked conditional rather than ok like the government-produced CAMD/FTD/MIDAS streams.",
    attribution: "Apple Inc.",
    resell: "conditional",
  },
  "data/github-activity": {
    license: "GitHub REST/Search API — public repository activity is CONDITIONAL under the GitHub API Terms (aggregated, non-personal metrics are an accepted use; server/githubOrgActivity.ts LICENSING note, verified 2026-07-04). A metered, resold-to-external-customers mirror has not been separately confirmed as still within that accepted-use envelope, so this stays marked conditional like the earnings-language/appstore-rankings mirrors, not ok like the government-produced CAMD/FTD/MIDAS streams.",
    attribution: "GitHub, Inc. (public repository activity, aggregated)",
    resell: "conditional",
  },
  "data/crop-conditions": {
    license: "USDA National Agricultural Statistics Service (NASS) QuickStats weekly Crop Progress — public domain (US federal government work), same class as the CAMD/FTD/MIDAS streams above.",
    attribution: "U.S. Department of Agriculture, National Agricultural Statistics Service (QuickStats)",
    resell: "ok",
  },
  "stats/vix-term-structure": {
    license: "Cboe Global Markets informational-use terms (VIX1D/VIX9D/VIX/VIX3M/VIX6M/VVIX daily close) — NOT US government work product like the CAMD/FTD/MIDAS/crop-conditions streams above; redistribution of the raw series needs Cboe permission, same posture as the OCC options-volume stream.",
    attribution: "Cboe Global Markets volatility indices",
    resell: "conditional",
  },
  "stats/nrc-reactor-status": {
    license: "U.S. Nuclear Regulatory Commission Power Reactor Status Reports — public domain (US federal government work), same class as the CAMD/FTD/MIDAS/crop-conditions streams above.",
    attribution: "U.S. Nuclear Regulatory Commission (NRC)",
    resell: "ok",
  },
  "data/13f-holdings": {
    license: "SEC EDGAR 13F-HR institutional-holdings filings — the EDGAR full-text record is public, but each filing (manager identity, holdings table: issuer/CUSIP/shares/value/discretion) is submitted by the reporting institutional manager, NOT authored or computed by the SEC itself like the CAMD/FTD/MIDAS/crop-conditions/NRC streams above; redistribution rights for a resold bulk mirror have not been separately confirmed, same posture as the issuer-authored earnings-language stream.",
    attribution: "SEC EDGAR 13F-HR filings (per reporting institutional manager)",
    resell: "conditional",
  },
  "stats/eu-macro": {
    license: "European macro cluster (ECB Data Portal EUR/USD + €STR + Eurosystem balance sheet; Eurostat EA20 industrial production; Deutsche Bundesbank 10Y Bund yield) — all three source licenses verified verbatim from their own reuse-policy documents at build time (ECB's 'Policy regarding the reuse of ESCB statistics'; Eurostat's copyright notice; Bundesbank's data-license terms): commercial reuse permitted with attribution, same class as the CAMD/FTD/MIDAS/crop-conditions/NRC public-domain US-gov streams above, not conditional like OCC/Cboe/issuer-authored data.",
    attribution: "per-series (each series carries its own required attribution string: ECB statistics / Eurostat sts_inpr_m / Deutsche Bundesbank)",
    resell: "ok",
  },
  "stats/fred-macro": {
    license: "FRED (Federal Reserve Bank of St. Louis) macro regime cluster — 28 of the module's 31 curated series (rates/curve, financial stress, labor, inflation, activity, money & liquidity, commodities/dollar), all Fed- or US-government-produced. The 3 third-party-copyrighted series in the same module (CBOE VIX, ICE BofA HY OAS, UMich Consumer Sentiment) are marked license:'restricted' in fredMacro.ts and are EXCLUDED from this endpoint's payload — internal regime use only, never product-surfaced, same exclusion buildMacroPayload() already applies to the public /api/data/macro route.",
    attribution: "Source: FRED, Federal Reserve Bank of St. Louis",
    resell: "ok",
  },
  "data/bank-failures": {
    license: "FDIC Bank Data API bank-failures event stream (api.fdic.gov/banks/failures) — public domain (US federal government work), same class as the CAMD/FTD/MIDAS/crop-conditions/NRC/eu-macro/fred-macro streams above.",
    attribution: "Federal Deposit Insurance Corporation (FDIC Bank Data API)",
    resell: "ok",
  },
  "data/gnss-integrity-signal": {
    license: "Derived from our own aircraft position archive (broadcast-origin ADS-B nic field) — inherits ODbL 1.0 share-alike from adsb.lol like tracks/aircraft above. A SOLD surface of this signal must derive from adsb.lol data alone (server/gnssIntegritySignal.ts's own LICENSE_NOTE), per the MONETIZATION TRIPWIRE — server/providerCompliance.ts enforces adsb.lol stays primary while billing is inactive.",
    attribution: "VolTradeAI datacore over adsb.lol (ODbL 1.0)",
    resell: "share-alike",
  },
  "data/dtcc-swaps": {
    license: "DTCC Security-Based Swap Data Repository (SBSDR) real-time public dissemination, SEC-mandated under Reg SBSR — informational-use terms (server/dtccSwaps.ts's own LICENSE_NOTE), NOT US government work product like the CAMD/FTD/MIDAS/crop-conditions/NRC/eu-macro/fred-macro/bank-failures streams above; each event is submitted by the reporting swap participant, not authored by DTCC or the SEC. Same conditional posture as OCC/Cboe — redistribution of the raw dissemination stream needs separate review, unresolved.",
    attribution: "DTCC SBSDR real-time public dissemination (SEC Reg SBSR)",
    resell: "conditional",
  },
  "data/fleet-utilization": {
    license: "Derived from our own aircraft position archive (broadcast-origin ADS-B) joined against the FAA aircraft registry entity spine — inherits ODbL 1.0 share-alike from adsb.lol like tracks/aircraft and data/gnss-integrity-signal above. GATE 1 (join accuracy) PASSED 2026-07-05 (20/20 stratified hexes matched an independent adsbdb registration exactly). A SOLD surface of this series must derive from adsb.lol data alone (server/providerCompliance.ts), per the MONETIZATION TRIPWIRE — enforced while billing is inactive.",
    attribution: "VolTradeAI datacore over adsb.lol (ODbL 1.0) + FAA Aircraft Registry (registrant identity)",
    resell: "share-alike",
  },
  "data/insider": {
    license: "SEC EDGAR Form 4 (insider transaction) filings — the EDGAR record is public, but each filing (issuer, reporting owner, transaction table) is submitted by the reporting insider/company, NOT authored or computed by the SEC itself like the CAMD/FTD/MIDAS/crop-conditions/NRC/eu-macro/fred-macro/bank-failures streams above; same conditional posture as the issuer-authored earnings-language and manager-submitted 13F-holdings streams. GATE 1 (DATA) PASSED (server/edgarForm4.test.ts — every extracted field hand-checked against filed XML). The buy-clustering SIGNAL hypothesis this parser feeds was GATE 2 KILLED in both directions (datacore/signal_ladder.json, sec_form4_insider_clustering) — this endpoint is RAW as-filed display only, never a predictive claim.",
    attribution: "SEC EDGAR (Form 4) filings, per reporting insider/issuer",
    resell: "conditional",
  },
  "data/attention": {
    license: "Wikimedia pageviews REST API (per-article daily view counts, en.wikipedia, all-access/agent=user) — the counts are computed by the Wikimedia Foundation itself from its own server logs, not user- or issuer-submitted content like the Form 4/13F/earnings-language streams above; verified verbatim at dumps.wikimedia.org/other/pageviews/readme.html: 'All Analytics datasets are available under the Creative Commons CC0 dedication' — same public-domain-equivalent class as the CAMD/FTD/MIDAS/crop-conditions/NRC/eu-macro/fred-macro/bank-failures streams above, not conditional like the issuer-authored streams.",
    attribution: "Wikimedia pageviews API (Wikimedia Foundation, CC0)",
    resell: "ok",
  },
  "data/cot": {
    license: "CFTC Commitments of Traders (disaggregated futures-only) — publicreporting.cftc.gov Socrata dataset 72hh-3qpy. The weekly report is compiled and published BY the CFTC itself from clearing-member position filings, a US government work product like the CAMD/FTD/MIDAS/crop-conditions/NRC/eu-macro/fred-macro/bank-failures/attention streams above (server/cftcCot.ts's own header: 'US government work, public domain') — not submitted-content conditional like the issuer-authored Form 4/13F/earnings-language/DTCC streams.",
    attribution: "CFTC Commitments of Traders (disaggregated)",
    resell: "ok",
  },
  "data/contracts": {
    license: "USAspending.gov federal contract-award transactions (award types A-D, |Transaction Amount| >= $25,000) — the award/transaction record is compiled and published BY the U.S. Department of the Treasury itself from agency-submitted FPDS actions, a US government work product like the CAMD/FTD/MIDAS/crop-conditions/NRC/eu-macro/fred-macro/bank-failures/attention/cot streams above (server/usaSpending.ts's own header: 'US-government work, free incl. commercial') — not submitted-content conditional like the issuer-authored Form 4/13F/earnings-language/DTCC streams.",
    attribution: "USAspending.gov, U.S. Department of the Treasury",
    resell: "ok",
  },
  "data/short-volume": {
    license: "FINRA Reg SHO daily consolidated short-sale volume (CNMS file) — FINRA itself compiles and publishes this file from exchange/TRF/ADF member reporting, free for use with attribution (server/finraShortVolume.ts's own header), NOT US government work product like the CAMD/FTD/MIDAS/crop-conditions/NRC/eu-macro/fred-macro/bank-failures/attention/cot/contracts streams above — same informational-use-terms posture as the OCC/Cboe streams, not submitted-content conditional like the issuer-authored Form 4/13F/earnings-language/DTCC streams either.",
    attribution: "FINRA Reg SHO daily short sale volume",
    resell: "conditional",
  },
  "data/methane-plumes": {
    license: "Global Energy Monitor — Methane Emitters Tracker (GMET) satellite plume detections (CarbonMapper/GHGSat-class providers, as catalogued by GEM) joined against GEM's own Oil & Gas Extraction Tracker + Global Coal Mine Tracker for nearest-asset proximity — GEM publishes both source datasets under CC BY 4.0, the same open-attribution class as the CFTC COT/USAspending/FRED/crop-conditions/bank-failures/NRC/attention/Digitraffic-trains streams above, NOT conditional like the issuer-authored Form 4/13F/earnings-language/DTCC streams or the informational-use-terms OCC/Cboe/FINRA streams.",
    attribution: "Global Energy Monitor — Methane Emitters Tracker + Oil & Gas Extraction Tracker + Global Coal Mine Tracker (CC BY 4.0)",
    resell: "ok",
  },
};

/** Self-documenting endpoint reference — /developers renders this; gated
 *  items are listed as coming so the docs stay honest about what exists. */
export function apiMeta() {
  return {
    version: "v1",
    auth: "x-api-key header (or ?api_key=). Keys are invite-only during the preview — join the waitlist on /developers.",
    endpoints: [
      { path: "/api/v1/tracks/:kind/:id", params: "kind=aircraft|vessels|trains; id=icao24|MMSI|train id; ?hours<=168", desc: "Recent position track from our own archive (recording since 2026-07-03)." },
      { path: "/api/v1/stats/portdwell", params: "-", desc: "Per-port dwell statistics (completed calls, in-port-now, medians, 3x-median anomaly flags) over the 9 imagery-verified port geofences.", preview: "/api/data/portdwell" },
      { path: "/api/v1/stats/shadow", params: "-", desc: "Dark-ship RAW statistics: AIS gap events, identity candidates, STS-zone loitering — counts with honest coverage caveats.", preview: "/api/data/shadowstats" },
      { path: "/api/v1/stats/archive", params: "-", desc: "Archive growth metadata (streams, samples, days recorded).", preview: "/api/data/archive/stats" },
      { path: "/api/v1/graph", params: "?entity=<ticker|MMSI|CIK|facility id>&hops<=3 (omit entity for counts-only)", desc: "Everything Graph v1 — Form 4 insiders, entity_map operator->ticker, and AIS port-call edges, joined into one node/edge graph. RAW (asserts filed relationships with provenance; no predictive claim).", preview: "/api/data/graph" },
      { path: "/api/v1/stats/plant-operations", params: "-", desc: "Per-facility power-plant utilization ground truth (sum grossLoad MW-days, sum operating hours) from EPA's own unit-level CEMS reporting, TX pilot scope, quarterly cadence. RAW, no predictive claim — public-domain US federal data, resell ok.", preview: "/api/data/plant-operations" },
      { path: "/api/v1/stats/secftd", params: "-", desc: "SEC CNS fails-to-deliver leaderboard: newest settlement date's top fail balances (>=100k share floor, stated). A level, not a daily flow, published on a 2.5-4.5 week SEC lag. RAW, no predictive claim — public-domain US federal data, resell ok.", preview: "/api/data/ftd" },
      { path: "/api/v1/stats/midas", params: "-", desc: "SEC MIDAS individual-security market-structure metrics: cross-sectional lit/hidden/odd-lot/cancel data per (date, ticker), quarterly files with a multi-quarter publish lag. Rank scale differs by kind (Stock deciles 1-10, ETF quartiles 1-4, never comparable). RAW, no predictive claim — public-domain US federal data, resell ok. A candidate HFT-colonization filter; gate-2 signal testing not yet attempted (see research/open_questions.md).", preview: "/api/data/microstructure" },
      { path: "/api/v1/stats/occ-volume", params: "-", desc: "OCC daily cleared options volume, top underlyings by customer/market-maker put-call split (qty counts each clearing side; totals halved). GATE 1 (DATA) PASSED 2026-08-03 (0 diff vs. OCC's own published June-2026 monthly total across 21 trading days). GATE 2 (SIGNAL) on the customer call/put-skew hypothesis was KILLED 2026-08-03/08-05 (the pre-registered direction reversed and a reversed-direction offshoot also failed disjoint out-of-sample replication) — not a validated trading signal. RAW display + archive only, no predictive claim. OCC informational-use terms, not government work product — conditional resell, see license_marks.", preview: "/api/data/occ-volume" },
      { path: "/api/v1/data/earnings-language", params: "-", desc: "Most-recent SEC 8-K Item 2.02 (earnings-results) filings: as-filed Exhibit 99 press-release text, resolved ticker, filing/acceptance timestamps (lookahead-free). RAW as-filed display, no predictive claim — gate-2 (does guidance-language tone predict forward returns) has only an encouraging but INCOMPLETE preliminary pilot (research/open_questions.md). Exhibit text is issuer-authored, not government work product — conditional resell, see license_marks.", preview: "/api/data/earnings-language" },
      { path: "/api/v1/data/appstore-rankings", params: "-", desc: "Daily App Store chart rank + rating counts for a 16-app hand-verified consumer watchlist (US/GB/CA storefronts, top-free/top-grossing). RAW display, no predictive claim — GATE 2 (vs company-reported metrics) needs ~90 days of history, not attempted before ~2026-10-30 (research/open_questions.md NEW DATA ROOTS #3). Android excluded (ToS-blocked); rank:null means outside the top 100 that day, never fabricated. Conditional resell, see license_marks.", preview: "/api/data/appstore-rankings" },
      { path: "/api/v1/data/github-activity", params: "-", desc: "Weekly merged-PR + commit + unique-actor counts for a 15-org hand-verified develop-in-public engineering watchlist (small-cap devtools through large-cap controls). RAW display, no predictive claim — GATE 2 (does public commit/PR velocity lead or confirm market-priced trends) has NOT been attempted; the module's own sober prior expects real structure for at most a third of the panel. mergedPRs excludes bot-app PRs; commits is unfiltered; uniqueActorsSample is bot-filtered but capped at a 100-item page (actorSampleCapped:true undercounts). Conditional resell, see license_marks.", preview: "/api/data/github-activity" },
      { path: "/api/v1/data/crop-conditions", params: "-", desc: "Most-recent week's USDA NASS national weekly condition ratings (5 classes via short_desc) for corn + soybeans, Monday releases in season. GATE 1 (DATA) PASSED 2026-08-04 (0pp difference vs. USDA's own published Crop Progress bulletin) — condition-DELTA signals stay gate-2-locked (research/open_questions.md), this endpoint mirrors the validated raw levels only. Requires the server's NASS_API_KEY to be configured; returns 503 if not. Public-domain US federal data, freely resellable.", preview: "/api/data/crop-conditions" },
      { path: "/api/v1/stats/vix-term-structure", params: "-", desc: "Cboe VIX1D/VIX9D/VIX/VIX3M/VIX6M/VVIX daily close term structure plus two derived ratios (vix/vix3m contango-vs-backwardation, vix9d/vix front-end stress), latest day + a 30-day recent window. GATE 1 (DATA) PASSED 2026-08-07 (exact match vs. FRED's independent VIXCLS series for 3/3 spot-checked dates) — RAW/regime-feature framing only, no predictive claim; gate-2 signal testing not attempted. Cboe informational-use terms, not government work product — conditional resell, see license_marks.", preview: "/api/data/vix-term-structure" },
      { path: "/api/v1/stats/nrc-reactor-status", params: "-", desc: "Daily percent-of-rated-thermal-power per operating NRC reactor unit (unit granularity), plus a per-plant join (units grouped onto the WRI/HIFLD registry's lat/lon, mean power bucketed into full/reduced/outage/unknown) for the newest reporting day. GATE 1 (DATA) PASSED 2026-08-04 (registry-match check, see scripts/nrc_gate1_registry_match.ts). RAW display only — outage-adjacent SIGNAL hypothesis stays gate-2-locked (research/open_questions.md POWER-PLANT SIGNAL HYPOTHESES). Public-domain US federal data, freely resellable.", preview: "/api/data/nrc-reactor-status" },
      { path: "/api/v1/data/13f-holdings", params: "-", desc: "Most-recent SEC EDGAR 13F-HR institutional holdings filings: manager identity, filing period, and the FULL as-filed holdings table (issuer, CUSIP, shares, value, discretion) for focused managers holding <=250 positions — mega-managers over that cap return a summary-only record (holdingsOmitted=true) instead of an index-hugging wall of rows, the same hypothesis-driven FOCUSED_MAX_HOLDINGS cap the archive itself applies. Unlike the /data map's top-25-by-value UI display trim, this endpoint returns every stored position for a focused filing. RAW as-filed display, no predictive claim — GATE 2 (new small-cap position clustering vs 60-90d forward returns; the 45-day filing lag is modeled honestly, holdings are stale when public) NOT attempted. Filings are submitted by the reporting manager, not government-authored — conditional resell, see license_marks.", preview: "/api/data/filings13f" },
      { path: "/api/v1/stats/eu-macro", params: "-", desc: "European macro regime cluster: ECB EUR/USD reference rate + €STR + weekly Eurosystem balance-sheet total, Eurostat EA20 industrial production, and the 10Y Bund yield (Deutsche Bundesbank) — 5 curated series, each with latest/prev values and a recent history window. REGIME INPUT feed (same framing as the FRED macro cluster) — never a direct trading signal, gate-2 signal testing not attempted. Keyless (all three sources free with attribution). Commercial reuse permitted with attribution, verified verbatim from each source's own reuse-policy document — public-domain-equivalent resell, see license_marks.", preview: "/api/data/eu-macro" },
      { path: "/api/v1/stats/fred-macro", params: "-", desc: "FRED macro regime cluster: 28 Fed/US-government-produced rates-curve, financial-stress, labor, inflation, activity, and money/liquidity series (3-month through 30-year Treasury yields, Fed Funds, SOFR, jobless claims, CPI, industrial production, M2, Fed balance sheet, WTI, trade-weighted dollar, and more), each with latest/prev values and a recent history window. REGIME INPUT feed (same framing as the eu-macro cluster) — never a direct trading signal, gate-2 signal testing not attempted. 3 third-party-copyrighted series (CBOE VIX, ICE BofA HY OAS, UMich Consumer Sentiment) are archived for internal regime use only and are EXCLUDED from this payload. Requires the server's FRED_API_KEY to be configured; returns 503 if not. Public-domain US federal/Fed data, freely resellable.", preview: "/api/data/macro" },
      { path: "/api/v1/data/bank-failures", params: "-", desc: "Most-recent US bank failures/assistance events from the FDIC's own failures endpoint (institution, cert, fail date, city/state, charter class, assets/deposits at failure in $ thousands, estimated DIF loss). GATE 1 (DATA) PASSED 2026-08-18 (3/4 sampled failures exact-matched an independent FDIC Call Report AND the FDIC's own press-release figures; the 4th's discrepancy was traced to the FDIC financials index lagging its own failures record for the single most-recent event, not a parsing defect — research/experiments.md 2026-08-18 entry). RAW display only — the deposit-flight-leads-KRE SIGNAL hypothesis stays gate-2-locked (blocked on both live market-return data and the still-unbuilt ticker/entity-graph join for mostly-private regional banks). cost_k is null until the FDIC estimates it, never coerced to zero. Public-domain US federal data, freely resellable.", preview: "/api/data/bank-failures" },
      { path: "/api/v1/data/gnss-integrity-signal", params: "-", desc: "Per-altitude-band GNSS position-integrity degradation over the Baltic Bornholm corridor, from our own broadcast-origin ADS-B archive: one-tailed exact binomial test per band, candidate region's nic==0 (zero containment) rate vs. a control region's own observed rate as the null, at p<0.01. THE FIRST GATE 2 (SIGNAL)-PASSED root exposed on this API (datacore/signal_ladder.json, gnss_integrity_adsb, gate2_pass, re-confirmed and strengthened across two re-runs). GATE 1 is PARTIAL — DTU Space's Bornholm RF station independently corroborates the phenomenon/region, not this exact sample's specific dates. Not tradeable: this is gate 2 (statistical discrimination), not gate 3 (backtested entry/exit) — no position sizing or trading decision is made from it. Aircraft-archive-derived, ODbL share-alike lineage — see license_marks.", preview: "/api/data/gnss-integrity-signal" },
      { path: "/api/v1/data/dtcc-swaps", params: "-", desc: "DTCC SBSDR equity total-return-swap dissemination events on US-CUSIP/ISIN underliers only (volume-budget scope decision, 2026-08-22): file/source date, today's US-underlier row count, new rows archived, total archived, and the largest-notional events from the source file's most recent published day (dissemination id, action type, event/effective timestamps, notional amount + currency where not masked by the source's own Dodd-Frank real-time-reporting cap, underlier id/source/name). GATE 1 (DATA) PASSED 2026-08-22 (two independent checksum standards, ISO 6166 ISIN and CUSIP Global Services mod-10, both >=99.998% on the live file — scripts/dtcc_swaps_gate1.ts). RAW display only — the fresh-large-notional-clustering SIGNAL hypothesis stays gate-2-locked pending archive depth. top_rows is the current poll cycle's ranking, not a running archive-wide one. Not US-government work product — conditional resell, see license_marks.", preview: "/api/data/dtcc-swaps" },
      { path: "/api/v1/data/fleet-utilization", params: "?top<=200 (default 50)", desc: "Corporate/LLC fleet utilization: per-owner weekly flight counts and airborne hours, sessionized from our own aircraft position archive and joined against the FAA registry entity spine (owners with <2 airframes excluded). GATE 1 (join accuracy) PASSED 2026-07-05 (20/20 stratified hexes matched an independent adsbdb registration exactly) — GATE 2 (utilization x earnings surprise) NOT attempted, this is descriptive, not a trading signal. Owners are FAA REGISTRANTS, not necessarily beneficial owners (trustee/leasing shells hide the real operator); airborne hours are LOWER BOUNDS under adaptive archive sampling; weeks without archive coverage are absent, not zero. Aircraft-archive-derived, ODbL share-alike lineage — see license_marks.", preview: "/api/data/fleet-utilization" },
      { path: "/api/v1/data/insider", params: "-", desc: "Most-recent SEC EDGAR Form 4 (insider transaction) filings: issuer, reporting owner (director/officer/10%-owner flags), and the full derivative/non-derivative transaction table (transaction code, shares, price, shares owned after) — RAW as-filed display, no predictive claim. GATE 1 (DATA) PASSED (server/edgarForm4.test.ts). The buy-clustering SIGNAL hypothesis this parser feeds was GATE 2 KILLED in both directions (research/open_questions.md; datacore/signal_ladder.json sec_form4_insider_clustering) — filings are submitted by the reporting insider/issuer, not government-authored, conditional resell, see license_marks.", preview: "/api/data/insider" },
      { path: "/api/v1/data/attention", params: "-", desc: "Daily Wikimedia pageviews for a curated 23-ticker company-article seed (en.wikipedia, all-access/agent=user) — RAW daily view counts, an attention PROXY, no spike/z-score claim until the archive holds trailing history and gate 2 runs. GATE 1 (DATA) PASSED 2026-08-18 (11/11 hand-checked tickers show pageviews peaking above trailing baseline in the [8-K Item 2.02 filing date, +1] window; a redirect-stub undercount affecting 3 seed pairs was found and fixed the same session). GATE 2 (does an attention spike lead volume/volatility 1-5d) NOT attempted. Computed by the Wikimedia Foundation from its own server logs, CC0 — freely resellable, see license_marks.", preview: "/api/data/attention" },
      { path: "/api/v1/data/cot", params: "-", desc: "CFTC Commitments of Traders, disaggregated futures-only: weekly positioning by trader category (producer/merchant, swap, managed-money, other-reportable — long/short/spread) for every reported contract market, Tuesday as-of/Friday-publish. GATE 1 (DATA) PASSED 2026-07-05 (0 rejections across a 156-week backfill, 7 symbols). GATE 2 (managed-money positioning-extreme mean-reversion) has already run a first-pass screen: GLD/CORN/SPY/QQQ/TLT/SLV were KILLED; only USO shows a marginal effect (p=0.0355) that fails the multi-comparison Bonferroni bar and was explicitly NOT promoted to logic gate 3. RAW display + archive only, no predictive claim. US government work product, public domain, freely resellable.", preview: "/api/data/cot" },
      { path: "/api/v1/data/contracts", params: "-", desc: "Most-recent USAspending.gov federal contract-award transactions (award types A-D, |Transaction Amount| >= $25,000; each row carries a precision-first ticker match — persistent UEI cache -> exact SEC company-name match -> award-detail FPDS parent, never fuzzy; unmatched rows return tkr:null and must be skipped, never guessed). GATE 1 (recipient->ticker matcher) PASSED 2026-07-24. GATE 2 (large award/market-cap ratio predicts better forward returns for small caps) was REJECTED 2026-08-15 (adequately powered at 5d, n=50 high_ratio/n=43 low_ratio, no positive separation at any horizon; the one nominally-interesting result was WRONG-SIGNED and fails the multi-comparison Bonferroni bar) — RAW as-seen display only, no predictive claim. action_date is the contract's signature date, not an event date; rt (as-seen date) is the only honest event date, and DoD/USACE awards publish roughly 90 days late. Public-domain US federal data, freely resellable.", preview: "/api/data/contracts" },
      { path: "/api/v1/data/short-volume", params: "-", desc: "FINRA Reg SHO daily consolidated (CNMS) short-sale volume: market-wide aggregate short ratio plus a top-ratio list of symbols clearing a stated total-volume floor. This is short-marked EXECUTION volume (a flow proxy), NOT short interest — the distinction matters and is stated on every response. GATE 1 (DATA) PASSED 2026-07-05. GATE 2 (short-ratio extremes predict reversals) FIRST-PASS RUN 2026-08-06 FAILED the pre-registered composite bar (ordering); a PRE-REGISTERED FOLLOW-UP RETEST 2026-08-15 against an unbiased population baseline also failed to clear significance (t=1.303 vs crit=2.131) — two consecutive fails on the same window, VERDICT FAIL/INCONCLUSIVE, not killed (same window twice, not a disjoint out-of-sample replication or sign reversal). RAW display only, no predictive claim. FINRA informational-use terms, not government work product — conditional resell, see license_marks.", preview: "/api/data/short-volume" },
      { path: "/api/v1/data/methane-plumes", params: "-", desc: "Global Energy Monitor Methane Emitters Tracker (GMET): dated satellite methane-plume detections (CarbonMapper/GHGSat-class providers, as catalogued by GEM), each joined to its nearest catalogued GEM oil/gas-extraction or coal-mine asset within a stated match radius (or null when nothing catalogued is that close). GATE 1 (plume detection itself) is calibrated upstream by GEM/CarbonMapper/GHGSat and effectively trivial to inherit. GATE 2(a) (the proximity join) SHIPPED 2026-07-19 — gates 2(b)-(d) (repeat-detection rate, a same-universe base rate, and matching operators' own disclosed emissions) are explicitly NOT built. nearestAsset is a GEOMETRIC PROXIMITY FACT, not a confirmed or claimed emissions attribution — RAW display only, no predictive claim. GEM publishes both source datasets under CC BY 4.0, freely resellable with attribution.", preview: "/api/data/methane-plumes" },
      { path: "/api/v1/meta", params: "-", desc: "This document.", preview: "/api/v1/meta" },
    ],
    coming_gated: [
      "tank-fill readings (Sentinel-2 — ladder gate 2 not yet passed; experimental readings stay internal)",
    ],
    agent_tools: "/api/v1/agent-tools",
    limits: TIER_LIMITS,
    license_marks: LICENSE_MARKS,
    disclaimer: "Data as-is; not for safety-of-life use; attribution and share-alike marks travel with each response.",
  };
}

/** Agent tool spec — the LIVE API rendered as function-calling tool
 *  definitions so a developer can hand VolTradeAI's verified physical-world
 *  data straight to an AI agent (Anthropic tool use, OpenAI functions, or an
 *  MCP server). Derived from the SAME live endpoint set as apiMeta(), so
 *  gated signals can never leak in; each tool names the license_marks key(s)
 *  of what it returns, so provenance and freshness travel into the agent's
 *  context, not just the raw number. Public — it is documentation, not data;
 *  the calls themselves still require an x-api-key. This is the "ground-truth
 *  layer for AI agents" surface: an agent grounded here answers from observed,
 *  archived measurement instead of model-generated plausibility. */
export function agentToolSpec(baseUrl = "https://voltradeai.com") {
  const tools = [
    {
      name: "voltrade_get_track",
      description: "Recent position track for one aircraft, vessel, or train from VolTradeAI's own continuously-recorded archive. Returns observed, timestamped positions — ground truth, not a prediction.",
      input_schema: {
        type: "object",
        properties: {
          kind: { type: "string", enum: ["aircraft", "vessels", "trains"], description: "Asset class." },
          id: { type: "string", description: "icao24 (aircraft), MMSI (vessel), or train id." },
          hours: { type: "integer", minimum: 1, maximum: 168, default: 24, description: "Lookback window in hours (max 168)." },
        },
        required: ["kind", "id"],
      },
      endpoint: "GET /api/v1/tracks/{kind}/{id}?hours={hours}",
      returns_provenance: ["tracks/aircraft", "tracks/vessels", "tracks/trains"],
    },
    {
      name: "voltrade_port_dwell_stats",
      description: "Per-port dwell statistics over 9 imagery-verified port geofences: completed calls, ships in-port now, median dwell, and 3x-median anomaly flags. RAW overlay — descriptive, not a trading signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/portdwell",
      returns_provenance: ["stats/portdwell"],
    },
    {
      name: "voltrade_shadow_fleet_stats",
      description: "Dark-ship RAW statistics: AIS gap events, identity candidates, and STS-zone loitering counts, with honest coverage caveats. RAW overlay — not a signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/shadow",
      returns_provenance: ["stats/shadow"],
    },
    {
      name: "voltrade_archive_stats",
      description: "Archive growth metadata: streams recorded, sample counts, and days of history — how much verified physical-economy data the platform holds.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/archive",
      returns_provenance: ["stats/archive"],
    },
    {
      name: "voltrade_get_graph",
      description: "Everything Graph v1 — Form 4 insider filings, entity_map operator->ticker joins, and AIS port-call edges, joined into one node/edge graph. Omit entity for counts-only; pass an entity to get its neighborhood. RAW overlay — asserts filed relationships with provenance, no predictive claim.",
      input_schema: {
        type: "object",
        properties: {
          entity: { type: "string", description: "Optional: ticker, MMSI, CIK, or facility id. Omit for graph-wide counts only." },
          hops: { type: "integer", minimum: 0, maximum: 3, default: 1, description: "Neighborhood radius when entity is given." },
        },
        required: [],
      },
      endpoint: "GET /api/v1/graph?entity={entity}&hops={hops}",
      returns_provenance: ["graph"],
    },
    {
      name: "voltrade_plant_operations_stats",
      description: "Per-facility power-plant utilization ground truth (summed gross load MW-days, summed operating hours) from the U.S. EPA's own unit-level Continuous Emissions Monitoring (CEMS) reporting, TX pilot scope, quarterly cadence. Public-domain US federal data, freely resellable. RAW overlay — direct plant-utilization ground truth, not a trading signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/plant-operations",
      returns_provenance: ["stats/plant-operations"],
    },
    {
      name: "voltrade_secftd_stats",
      description: "SEC CNS fails-to-deliver leaderboard for the newest published settlement date: top fail balances by share quantity (>=100k share floor, stated), from the SEC's own half-month CNS files. A fail BALANCE (level), not a daily flow, published on a 2.5-4.5 week SEC lag. Public-domain US federal data, freely resellable. RAW overlay — a crowded/settlement-stress indicator, not a standalone trading signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/secftd",
      returns_provenance: ["stats/secftd"],
    },
    {
      name: "voltrade_midas_stats",
      description: "SEC MIDAS individual-security market-structure metrics: cross-sectional lit/hidden/odd-lot/cancel-to-trade data per (date, ticker) from the SEC's own quarterly files (multi-quarter publish lag). Rank scale differs by kind (Stock deciles 1-10, ETF quartiles 1-4, never comparable). Public-domain US federal data, freely resellable. RAW overlay — a candidate HFT-colonization filter, not a validated trading signal (gate-2 unattempted).",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/midas",
      returns_provenance: ["stats/midas"],
    },
    {
      name: "voltrade_occ_volume_stats",
      description: "OCC daily cleared options volume: top underlyings by customer/market-maker put-call split (quantity counts each clearing side, from the Options Clearing Corporation's own daily volume report). Public-domain-adjacent but OCC-permission-conditional resell — not U.S. government work product like the SEC MIDAS/FTD tools above. GATE 1 (DATA) PASSED — verified 0 difference against OCC's own published monthly total. GATE 2 (SIGNAL): the customer call/put-skew hypothesis was KILLED (pre-registered direction reversed, and the reversed direction also failed independent out-of-sample replication) — RAW overlay only, not a validated trading signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/occ-volume",
      returns_provenance: ["stats/occ-volume"],
    },
    {
      name: "voltrade_earnings_language",
      description: "Most-recent SEC 8-K Item 2.02 filings: as-filed Exhibit 99 earnings press-release text per company, with resolved ticker and exact filing/acceptance timestamps (lookahead-free). RAW as-filed display — gate-2 signal testing (does guidance-language tone predict forward returns) has only a preliminary, INCOMPLETE pilot result, not a validated trading signal. Exhibit text is issuer-authored, not government work product like the SEC MIDAS/FTD tools above — conditional resell.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/earnings-language",
      returns_provenance: ["data/earnings-language"],
    },
    {
      name: "voltrade_appstore_rankings",
      description: "Daily App Store chart rank + rating counts for a 16-app hand-verified consumer watchlist (DUOL/BMBL/MTCH/HOOD/COIN/RBLX-class, US/GB/CA storefronts). RAW display — GATE 2 (rank/rating trends vs company-reported metrics) needs ~90 days of history and has NOT been attempted yet. Android is dark (Google Play ToS-blocked, stated honestly, never backfilled). Public Apple feeds, conditional resell.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/appstore-rankings",
      returns_provenance: ["data/appstore-rankings"],
    },
    {
      name: "voltrade_github_activity",
      description: "Weekly merged-PR + commit + unique-actor counts for a 15-org hand-verified develop-in-public engineering watchlist (MDB/NET/DDOG/PLTR-class). RAW display — GATE 2 (does commit/PR velocity lead or confirm market-priced trends) has NOT been attempted, and the module's own sober prior expects real structure for at most a third of the panel, not the whole watchlist.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/github-activity",
      returns_provenance: ["data/github-activity"],
    },
    {
      name: "voltrade_crop_conditions",
      description: "Most-recent week's USDA NASS national weekly crop condition ratings (5 classes) for corn + soybeans, Monday releases in season. GATE 1 (DATA) PASSED — verified 0pp difference against USDA's own published Crop Progress bulletin. RAW display — condition-DELTA signal testing (GATE 2, vs forward grain futures returns) has NOT been attempted. Public-domain US federal data, freely resellable.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/crop-conditions",
      returns_provenance: ["data/crop-conditions"],
    },
    {
      name: "voltrade_vix_term_structure",
      description: "Cboe VIX1D/VIX9D/VIX/VIX3M/VIX6M/VVIX daily close term structure plus two derived ratios (vix/vix3m contango-vs-backwardation, vix9d/vix front-end stress), from Cboe's own daily index price feed. GATE 1 (DATA) PASSED — Cboe's own VIX close matched FRED's independently-published VIXCLS series exactly on every date spot-checked. RAW / regime-feature display — GATE 2 (does the term-structure shape predict forward returns) has NOT been attempted; not wired into any trading decision today. Cboe informational-use terms, not government work product like the SEC/EPA/USDA tools above — conditional resell.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/vix-term-structure",
      returns_provenance: ["stats/vix-term-structure"],
    },
    {
      name: "voltrade_nrc_reactor_status",
      description: "Daily percent-of-rated-thermal-power for every operating U.S. NRC-licensed reactor unit, from the NRC's own Power Reactor Status Reports, plus a per-plant join (units grouped onto the registry's plant lat/lon, mean reported power bucketed into full/reduced/outage/unknown — a single-unit outage at an otherwise-full multi-unit plant reads as 'reduced,' with the unit-level detail preserved, never collapsed away). GATE 1 (DATA) PASSED — units matched against the WRI/HIFLD plant registry. RAW display — the outage-adjacent SIGNAL hypothesis (research/open_questions.md POWER-PLANT SIGNAL HYPOTHESES) has NOT been gate-2 tested. Public-domain US federal data, freely resellable.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/nrc-reactor-status",
      returns_provenance: ["stats/nrc-reactor-status"],
    },
    {
      name: "voltrade_thirteenf_holdings",
      description: "Most-recent SEC EDGAR 13F-HR institutional holdings filings: manager identity, filing period, and the full as-filed holdings table (issuer, CUSIP, shares, value, investment discretion) for focused managers holding <=250 positions — mega-managers over that cap return a summary-only record (holdingsOmitted=true) rather than an index-hugging wall of rows, the same hypothesis-driven cap the archive itself applies (EDGE DOCTRINE: fish where whales can't). Unlike the /data map's top-25-by-value display trim, this tool returns every stored position for a focused filing. RAW as-filed display, no predictive claim — GATE 2 (does new small-cap position clustering by capacity-constrained managers precede 60-90 day outperformance; the 45-day filing lag is modeled honestly, holdings are stale when public) has NOT been attempted. Filings are submitted by the reporting manager, not government-authored like the SEC MIDAS/FTD tools above — conditional resell.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/13f-holdings",
      returns_provenance: ["data/13f-holdings"],
    },
    {
      name: "voltrade_fred_macro",
      description: "FRED (Federal Reserve Bank of St. Louis) macro regime cluster: 28 Fed/US-government-produced series spanning rates & curve (3-month through 30-year Treasury yields, 10Y-2Y and 10Y-3M spreads, Fed Funds, SOFR, breakeven inflation), financial stress (St. Louis Fed / Chicago Fed indexes), labor (jobless claims, unemployment, payrolls), inflation (CPI, core CPI, core PCE), activity (industrial production, housing starts/permits, retail sales), money & liquidity (M2, Fed balance sheet, reverse repo), and commodities/dollar (WTI, trade-weighted dollar) — each with its latest value, prior value, and a recent history window. REGIME INPUT feed, NOT a direct trading signal on its own, and gate-2 signal testing has NOT been attempted. 3 third-party-copyrighted series in the same underlying module (CBOE VIX, ICE BofA HY OAS, UMich Consumer Sentiment) are archived for internal regime use only and are EXCLUDED from this tool's response. Public-domain US federal/Fed data, freely resellable.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/fred-macro",
      returns_provenance: ["stats/fred-macro"],
    },
    {
      name: "voltrade_eu_macro",
      description: "European macro regime cluster: ECB EUR/USD reference rate, €STR (euro short-term rate), weekly Eurosystem balance-sheet total, Eurostat EA20 industrial production, and the 10Y Bund yield (Deutsche Bundesbank) — 5 curated series, each with its latest value, prior value, and a recent history window. REGIME INPUT feed, the same framing as the FRED macro cluster (voltrade tools above) — NOT a direct trading signal on its own, and gate-2 signal testing has NOT been attempted. All three sources are keyless and free; commercial reuse is permitted with attribution, verified verbatim from each source's own reuse-policy document at build time.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/eu-macro",
      returns_provenance: ["stats/eu-macro"],
    },
    {
      name: "voltrade_bank_failures",
      description: "Most-recent US bank failures/assistance events from the FDIC's own failures endpoint: institution name, FDIC cert, fail date, city/state, charter class, assets/deposits at failure ($ thousands), and estimated DIF loss (null until the FDIC estimates it). GATE 1 (DATA) PASSED — 3 of 4 sampled failures exact-matched an independent FDIC Call Report and the FDIC's own press-release figures; the one discrepancy was traced to the FDIC's financials index lagging its own failures record for the most recent event, not a parsing defect. RAW display — the deposit-flight-leads-KRE SIGNAL hypothesis has NOT been gate-2 tested. Public-domain US federal data, freely resellable.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/bank-failures",
      returns_provenance: ["data/bank-failures"],
    },
    {
      name: "voltrade_gnss_integrity_signal",
      description: "Per-altitude-band GNSS position-integrity degradation over the Baltic Bornholm corridor, from our own broadcast-origin ADS-B archive: a one-tailed exact binomial test per altitude band, testing whether the candidate region's nic==0 (zero position-integrity containment) rate is elevated beyond chance versus a control region's own observed rate as the null, at p<0.01. THE FIRST GATE 2 (SIGNAL)-PASSED root on this API — re-confirmed and strengthened across two re-runs (datacore/signal_ladder.json, gnss_integrity_adsb, gate2_pass). GATE 1 is PARTIAL, not full: DTU Space's Bornholm RF station independently corroborates the phenomenon and region, not this exact sample's specific dates. NOT tradeable — this is gate 2 (statistical discrimination), not gate 3 (backtested entry/exit); no position sizing or trading decision is made from it. Aircraft-archive-derived, ODbL 1.0 share-alike (adsb.lol) lineage, same as the tracks/aircraft tool above.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/gnss-integrity-signal",
      returns_provenance: ["data/gnss-integrity-signal"],
    },
    {
      name: "voltrade_dtcc_swaps",
      description: "DTCC Security-Based Swap Data Repository (SBSDR) equity total-return-swap dissemination events, scoped to US-CUSIP/ISIN underliers only: file/source date, today's US-underlier row count, new rows archived since the last poll, total rows archived, and the largest-notional events from the source file's most recent published day (dissemination id, action type, event/effective timestamps, notional amount + currency where not masked by the source's own Dodd-Frank real-time-reporting cap, underlier id/source/name). GATE 1 (DATA) PASSED — two independent checksum standards (ISO 6166 ISIN, CUSIP Global Services mod-10) both scored >=99.998% against a pre-stated 99.9% bar on the live file. RAW display — the fresh-large-notional-clustering SIGNAL hypothesis has NOT been gate-2 tested (blocked on accumulating archive depth). SEC-mandated public dissemination (Reg SBSR), informational-use terms, not government-authored — conditional resell.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/dtcc-swaps",
      returns_provenance: ["data/dtcc-swaps"],
    },
    {
      name: "voltrade_fleet_utilization",
      description: "Corporate/LLC fleet utilization: per-owner weekly flight counts and airborne hours, sessionized from VolTradeAI's own aircraft position archive and joined against the FAA aircraft registry entity spine (owners with fewer than 2 airframes excluded). GATE 1 (join accuracy) PASSED — 20/20 stratified hexes matched an independent adsbdb registration exactly. NOT a trading signal — GATE 2 (utilization vs earnings surprise) has NOT been attempted. Owners are FAA REGISTRANTS, not necessarily beneficial owners (trustee/leasing shells hide the real operator); airborne hours are LOWER BOUNDS under adaptive archive sampling; weeks without archive coverage are absent, never zero. Aircraft-archive-derived, ODbL 1.0 share-alike (adsb.lol) lineage, same as the tracks/aircraft and GNSS-integrity-signal tools above.",
      input_schema: {
        type: "object",
        properties: {
          top: { type: "integer", minimum: 1, maximum: 200, default: 50, description: "Max owners returned, ranked by airframe count." },
        },
        required: [],
      },
      endpoint: "GET /api/v1/data/fleet-utilization?top={top}",
      returns_provenance: ["data/fleet-utilization"],
    },
    {
      name: "voltrade_insider",
      description: "Most-recent SEC EDGAR Form 4 (insider transaction) filings: issuer identity, reporting owner (director/officer/10%-owner flags), and the full derivative/non-derivative transaction table (transaction code, shares, price per share, shares owned after). RAW as-filed display, no predictive claim. GATE 1 (DATA) PASSED — every extracted field hand-checked against filed XML (server/edgarForm4.test.ts). NOT a trading signal — the buy-clustering hypothesis this same parser feeds was GATE 2 KILLED in both directions (code-S sales mirror test and full-8-quarter code-P re-run both reversed the stated prior significantly at 60d). Filings are submitted by the reporting insider/issuer, not SEC-authored, same conditional-resell posture as the earnings-language and 13F-holdings tools above.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/insider",
      returns_provenance: ["data/insider"],
    },
    {
      name: "voltrade_attention",
      description: "Daily Wikimedia pageviews for a curated 23-ticker company-article seed (en.wikipedia, all-access/agent=user) — RAW daily view counts, an attention PROXY, no spike or z-score claim. GATE 1 (DATA) PASSED — 11/11 hand-checked tickers showed pageviews peaking above their own trailing baseline in the [8-K earnings-filing date, +1] window. NOT a trading signal — GATE 2 (does an attention spike lead volume/volatility 1-5d) has NOT been attempted. Computed by the Wikimedia Foundation itself from its own server logs, released CC0 (public domain) — freely resellable, unlike the issuer-authored earnings-language/13F-holdings/insider tools above.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/attention",
      returns_provenance: ["data/attention"],
    },
    {
      name: "voltrade_cot",
      description: "CFTC Commitments of Traders, disaggregated futures-only: weekly positioning by trader category (producer/merchant, swap, managed-money, other-reportable) across every reported contract market. RAW display, no predictive claim. GATE 1 (DATA) PASSED — 0 rejections across a 156-week backfill (7 symbols). NOT a trading signal — GATE 2's first-pass screen KILLED the positioning-extreme mean-reversion hypothesis on GLD/CORN/SPY/QQQ/TLT/SLV; the one nominal survivor (USO, p=0.0355) fails the Bonferroni multi-comparison bar and was not promoted. US government work product, public domain, freely resellable — unlike the issuer-authored insider/13F-holdings/earnings-language/DTCC tools above.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/cot",
      returns_provenance: ["data/cot"],
    },
    {
      name: "voltrade_usaspending_contracts",
      description: "Most-recent USAspending.gov federal contract-award transactions (award types A-D, |Transaction Amount| >= $25,000), each with a precision-first ticker match (persistent UEI cache -> exact SEC company-name match -> award-detail FPDS parent; unmatched rows return tkr:null and must never be guessed). RAW as-seen display, no predictive claim. GATE 1 (recipient->ticker matcher) PASSED. NOT a trading signal — GATE 2 (large award/market-cap ratio predicts better small-cap forward returns) was REJECTED 2026-08-15 (no positive separation at any horizon across an adequately powered n=50/43 split; the one nominally-interesting result was wrong-signed and fails the multi-comparison Bonferroni bar). action_date is the contract's signature date, not an event date — rt (as-seen date) is the only honest event date, and DoD/USACE awards publish roughly 90 days late. US government work product, public domain, freely resellable — same posture as the CFTC COT/FRED/crop-conditions/bank-failures/NRC/attention tools above.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/contracts",
      returns_provenance: ["data/contracts"],
    },
    {
      name: "voltrade_short_volume",
      description: "FINRA Reg SHO daily consolidated (CNMS) short-sale volume: market-wide aggregate short ratio plus a top-ratio list of symbols above a stated total-volume floor. Short-marked EXECUTION volume (a flow proxy), NOT short interest. RAW display, no predictive claim. GATE 1 (DATA) PASSED — 0 rejections across the file's own sum-of-parts identity check. NOT a trading signal — GATE 2's pre-registered first-pass screen FAILED the composite-bar ordering test, and a pre-registered follow-up retest against an unbiased population baseline also failed to clear significance (t=1.303 < crit=2.131) — two consecutive fails on the same window, VERDICT FAIL/INCONCLUSIVE, not a killed hypothesis (would need a disjoint out-of-sample window or a sign reversal). FINRA informational-use terms, not government work product — conditional resell, unlike the CFTC COT/USAspending/FRED/crop-conditions/bank-failures/NRC/attention tools above.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/short-volume",
      returns_provenance: ["data/short-volume"],
    },
    {
      name: "voltrade_methane_plumes",
      description: "Global Energy Monitor Methane Emitters Tracker (GMET): dated satellite methane-plume detections (CarbonMapper/GHGSat-class providers, as catalogued by GEM), each joined to its nearest catalogued GEM oil/gas-extraction or coal-mine asset within a stated match radius (null when nothing catalogued is that close). RAW display, no predictive claim. GATE 1 (plume detection) is calibrated upstream by GEM/CarbonMapper/GHGSat, effectively trivial to inherit. NOT a trading signal — GATE 2(a) (the proximity join itself) SHIPPED, but gates 2(b)-(d) (repeat-detection rate, a same-universe base rate, and matching operators' own disclosed emissions) are explicitly NOT built. nearestAsset is a GEOMETRIC PROXIMITY FACT, not a confirmed or claimed emissions attribution. GEM publishes both source datasets under CC BY 4.0 — freely resellable with attribution, same posture as the CFTC COT/USAspending/FRED/crop-conditions/bank-failures/NRC/attention tools above.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/data/methane-plumes",
      returns_provenance: ["data/methane-plumes"],
    },
  ];
  return {
    version: "v1",
    format: "JSON-Schema tool definitions — drop-in for Anthropic tool use, OpenAI function calling, or an MCP server.",
    base_url: baseUrl,
    auth: "Send x-api-key on every call (invite-only during the preview — join the waitlist on /developers). This spec is public; the data behind each tool requires a key.",
    ground_truth_note: "Every tool returns observed, archived measurements carrying provenance and a generated_at timestamp — built to ground AI agents in what is physically true rather than model-generated plausibility.",
    tools,
    license_marks: LICENSE_MARKS,
    excluded_gated: apiMeta().coming_gated,
    disclaimer: apiMeta().disclaimer,
  };
}
