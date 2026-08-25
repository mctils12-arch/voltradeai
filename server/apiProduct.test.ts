// API product foundation — pure-module tests (no express, no db, no network).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseApiKeys, makeRateLimiter, keyId, meterUsage, apiMeta, agentToolSpec,
  LICENSE_MARKS, TIER_LIMITS,
} from "./apiProduct";

const here = path.dirname(fileURLToPath(import.meta.url));

test("keys are env-seeded only, tiered, and trivially short keys are refused", () => {
  const m = parseApiKeys({ API_PRODUCT_KEYS: "supersecretkey123:alice:pro, k2:short:dev, anotherlongkey456:bob:nonsense" } as any);
  assert.equal(m.size, 2, "short key refused");
  assert.deepEqual(m.get("supersecretkey123"), { label: "alice", tier: "pro" });
  assert.equal(m.get("anotherlongkey456")!.tier, "dev", "unknown tier falls back to dev");
  assert.equal(parseApiKeys({} as any).size, 0, "no env -> no keys -> the API is fully closed");
});

test("rate limiter: per-minute window enforces tier limits and recovers", () => {
  const rl = makeRateLimiter();
  const t0 = 1_000_000_000_000;
  for (let i = 0; i < TIER_LIMITS.dev.perMinute; i++) {
    assert.ok(rl.allow("k", "dev", t0 + i * 10).ok);
  }
  const denied = rl.allow("k", "dev", t0 + 59_000);
  assert.equal(denied.ok, false);
  assert.equal(denied.retryAfterSec, 60);
  assert.ok(rl.allow("k", "dev", t0 + 61_000 + TIER_LIMITS.dev.perMinute * 10).ok, "window slides");
});

test("metering: raw keys NEVER reach the usage archive — only sha256 prefixes", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-api-"));
  const NOW = Date.UTC(2026, 6, 4, 12, 0, 0);
  meterUsage({ key: "supersecretkey123", endpoint: "/api/v1/stats/archive", status: 200, tier: "dev" }, base, NOW);
  const day = fs.readFileSync(path.join(base, "apiusage", "2026-07-04.jsonl"), "utf8");
  assert.ok(!day.includes("supersecretkey123"), "raw key leaked into the archive");
  const rec = JSON.parse(day.trim());
  assert.equal(rec.k, keyId("supersecretkey123"));
  assert.equal(rec.k.length, 12);
  assert.equal(rec.e, "/api/v1/stats/archive");
});

test("license marks: aircraft-derived endpoints carry ODbL share-alike; OWM is absent from the API entirely", () => {
  assert.equal(LICENSE_MARKS["tracks/aircraft"].resell, "share-alike");
  assert.ok(LICENSE_MARKS["tracks/aircraft"].license.includes("ODbL"));
  assert.equal(LICENSE_MARKS["tracks/vessels"].resell, "conditional");
  assert.equal(LICENSE_MARKS["stats/portdwell"].resell, "conditional", "AIS-derived stats inherit conditionality");
  const meta = apiMeta();
  assert.ok(!JSON.stringify(meta).toLowerCase().includes("openweathermap"),
    "OWM tiles are a display product — they may not appear on the data API");
});

test("meta honesty: gated products listed as coming, never as live endpoints; Graph v1 now IS live", () => {
  const meta = apiMeta();
  const paths = meta.endpoints.map((e: any) => e.path).join(" ");
  assert.ok(!paths.includes("tank"), "tank-fill must not be a live endpoint before gate 2");
  assert.ok(paths.includes("/api/v1/graph"), "Everything Graph v1 shipped — its keyed mirror must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/plant-operations"), "EPA CAMD keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/secftd"), "SEC FTD keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/midas"), "SEC MIDAS keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/occ-volume"), "OCC options volume keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/earnings-language"), "SEC 8-K earnings-language keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/appstore-rankings"), "App Store rankings keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/github-activity"), "GitHub org engineering-momentum keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/crop-conditions"), "USDA NASS crop conditions keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/vix-term-structure"), "Cboe VIX term structure keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/nrc-reactor-status"), "NRC reactor status keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/13f-holdings"), "SEC 13F-HR institutional holdings keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/eu-macro"), "European macro cluster keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/fred-macro"), "FRED macro cluster keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/bank-failures"), "FDIC bank failures keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/gnss-integrity-signal"), "GNSS integrity signal keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/dtcc-swaps"), "DTCC SBSDR equity swaps keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/fleet-utilization"), "corporate-fleet utilization keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/insider"), "SEC Form 4 insider transactions keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/data/attention"), "Wikimedia pageviews attention proxy keyed mirror shipped — must be a live endpoint");
  assert.ok(meta.coming_gated.length >= 1, "tank-fill remains the one still-gated product");
  assert.ok(!meta.coming_gated.join(" ").includes("Everything Graph"), "graph must not be listed as coming once live");
  assert.ok(meta.disclaimer.includes("safety-of-life"));
});

test("wiring pinned: /api/v1 routes registered behind requireApiKey; meta is the only public one", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  for (const p of ["/api/v1/meta", "/api/v1/tracks/:kind/:id", "/api/v1/stats/portdwell", "/api/v1/stats/shadow", "/api/v1/stats/archive", "/api/v1/graph", "/api/v1/stats/plant-operations", "/api/v1/stats/secftd", "/api/v1/stats/midas", "/api/v1/stats/occ-volume", "/api/v1/data/earnings-language", "/api/v1/data/appstore-rankings", "/api/v1/data/github-activity", "/api/v1/data/crop-conditions", "/api/v1/stats/vix-term-structure", "/api/v1/stats/nrc-reactor-status", "/api/v1/data/13f-holdings", "/api/v1/stats/eu-macro", "/api/v1/stats/fred-macro", "/api/v1/data/bank-failures", "/api/v1/data/gnss-integrity-signal", "/api/v1/data/dtcc-swaps", "/api/v1/data/fleet-utilization", "/api/v1/data/insider", "/api/v1/data/attention"]) {
    assert.ok(routes.includes(`"${p}"`), `route ${p} missing`);
  }
  const v1Block = routes.slice(routes.indexOf("/api/v1 — the DATA PRODUCT"));
  const guarded = (v1Block.match(/requireApiKey\(req, res\)/g) || []).length;
  assert.ok(guarded >= 20, `expected >=20 key-guarded endpoints, found ${guarded}`);
  assert.ok(routes.includes("meterUsage"), "metering must be wired");
});

test("graph license mark: conditional resell, inherits AIS conditionality like the port stats", () => {
  assert.equal(LICENSE_MARKS["graph"].resell, "conditional");
  assert.ok(LICENSE_MARKS["graph"].license.includes("aisstream"));
});

test("plant-operations license mark: public-domain US-gov data resells freely, unlike the AIS-derived stats", () => {
  assert.equal(LICENSE_MARKS["stats/plant-operations"].resell, "ok");
  assert.ok(LICENSE_MARKS["stats/plant-operations"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["stats/plant-operations"].license.includes("EPA"));
});

test("secftd license mark: public-domain US-gov data resells freely; agent tool documents it", () => {
  assert.equal(LICENSE_MARKS["stats/secftd"].resell, "ok");
  assert.ok(LICENSE_MARKS["stats/secftd"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["stats/secftd"].license.includes("SEC"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_secftd_stats");
  assert.ok(tool, "voltrade_secftd_stats tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/secftd"]);
});

test("midas license mark: public-domain US-gov data resells freely; agent tool documents it as an unvalidated candidate filter", () => {
  assert.equal(LICENSE_MARKS["stats/midas"].resell, "ok");
  assert.ok(LICENSE_MARKS["stats/midas"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["stats/midas"].license.includes("SEC"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_midas_stats");
  assert.ok(tool, "voltrade_midas_stats tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/midas"]);
  assert.ok(tool.description.includes("not a validated trading signal"), "honesty: gate-2 status must travel with the tool description");
});

test("occ-volume license mark: OCC informational-use terms are CONDITIONAL resell, not ok like the government-produced CAMD/FTD/MIDAS stats; agent tool documents the gate-2 kill honestly", () => {
  assert.equal(LICENSE_MARKS["stats/occ-volume"].resell, "conditional",
    "OCC's own terms require permission for raw bulk resale — must not be mismarked resell:ok like the SEC/EPA public-domain streams");
  assert.ok(LICENSE_MARKS["stats/occ-volume"].license.includes("OCC"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_occ_volume_stats");
  assert.ok(tool, "voltrade_occ_volume_stats tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/occ-volume"]);
  assert.ok(tool.description.includes("KILLED"), "honesty: gate-2's killed status must travel with the tool description, not just the raw archive framing");
  assert.ok(tool.description.includes("GATE 1"), "honesty: gate-1-pass status must also travel with the tool description");
});

test("nrc-reactor-status license mark: public-domain US-gov data resells freely like plant-operations/secftd/midas/crop-conditions, not conditional like OCC/Cboe/issuer-authored streams; agent tool documents gate-2 as unattempted", () => {
  assert.equal(LICENSE_MARKS["stats/nrc-reactor-status"].resell, "ok",
    "NRC Power Reactor Status Reports are US federal government work — must not be mismarked conditional like the OCC/Cboe/issuer-authored streams");
  assert.ok(LICENSE_MARKS["stats/nrc-reactor-status"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["stats/nrc-reactor-status"].attribution.includes("NRC"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_nrc_reactor_status");
  assert.ok(tool, "voltrade_nrc_reactor_status tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/nrc-reactor-status"]);
  assert.ok(tool.description.includes("GATE 1"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("NOT"), "honesty: gate-2-unattempted status must travel with the tool description");
});

test("earnings-language license mark: issuer-authored exhibit text is CONDITIONAL resell, unlike the government-produced CAMD/FTD/MIDAS stats; agent tool documents the incomplete gate-2 status", () => {
  assert.equal(LICENSE_MARKS["data/earnings-language"].resell, "conditional",
    "the Exhibit 99 press-release text is issuer-authored, not U.S. government work product — must not be marked resell:ok like the CAMD/FTD/MIDAS datasets");
  assert.ok(LICENSE_MARKS["data/earnings-language"].license.includes("issuer-authored"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_earnings_language");
  assert.ok(tool, "voltrade_earnings_language tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/earnings-language"]);
  assert.ok(tool.description.includes("INCOMPLETE"), "honesty: the gate-2 pilot's incomplete status must travel with the tool description");
});

test("appstore-rankings license mark: CONDITIONAL resell like earnings-language, not ok like the government-produced CAMD/FTD/MIDAS stats; agent tool documents gate-2 as not attempted", () => {
  assert.equal(LICENSE_MARKS["data/appstore-rankings"].resell, "conditional",
    "the underlying Apple feeds are public but conditional on low-volume internal use — a metered external mirror must not be mismarked resell:ok");
  assert.ok(LICENSE_MARKS["data/appstore-rankings"].license.includes("Apple"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_appstore_rankings");
  assert.ok(tool, "voltrade_appstore_rankings tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/appstore-rankings"]);
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("github-activity license mark: CONDITIONAL resell like earnings-language/appstore-rankings, not ok like the government-produced CAMD/FTD/MIDAS stats; agent tool documents gate-2 as not attempted", () => {
  assert.equal(LICENSE_MARKS["data/github-activity"].resell, "conditional",
    "GitHub REST/Search API's public repo activity is a conditional accepted use, not government work product — must not be mismarked resell:ok");
  assert.ok(LICENSE_MARKS["data/github-activity"].license.includes("GitHub"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_github_activity");
  assert.ok(tool, "voltrade_github_activity tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/github-activity"]);
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("crop-conditions license mark: USDA NASS is public-domain US-gov data like the CAMD/FTD/MIDAS stats, not conditional like earnings-language/appstore-rankings/github-activity; agent tool documents gate-1-pass/gate-2-unattempted status", () => {
  assert.equal(LICENSE_MARKS["data/crop-conditions"].resell, "ok",
    "USDA NASS QuickStats is US federal government work product — must be marked resell:ok like the other government-produced streams");
  assert.ok(LICENSE_MARKS["data/crop-conditions"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["data/crop-conditions"].license.includes("NASS"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_crop_conditions");
  assert.ok(tool, "voltrade_crop_conditions tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/crop-conditions"]);
  assert.ok(tool.description.includes("GATE 1 (DATA) PASSED"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("vix-term-structure license mark: Cboe informational-use terms are CONDITIONAL resell, not ok like the government-produced CAMD/FTD/MIDAS/crop-conditions streams; agent tool documents gate-1-pass/gate-2-unattempted status", () => {
  assert.equal(LICENSE_MARKS["stats/vix-term-structure"].resell, "conditional",
    "Cboe's own terms require permission for raw bulk resale — must not be mismarked resell:ok like the SEC/EPA/USDA public-domain streams");
  assert.ok(LICENSE_MARKS["stats/vix-term-structure"].license.includes("Cboe"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_vix_term_structure");
  assert.ok(tool, "voltrade_vix_term_structure tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/vix-term-structure"]);
  assert.ok(tool.description.includes("GATE 1 (DATA) PASSED"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("13f-holdings license mark: manager-submitted filings are CONDITIONAL resell like earnings-language, not ok like the government-produced CAMD/FTD/MIDAS/crop-conditions/NRC stats; agent tool documents the full-holdings response-shape decision and gate-2 as unattempted", () => {
  assert.equal(LICENSE_MARKS["data/13f-holdings"].resell, "conditional",
    "13F-HR filings are submitted by the reporting institutional manager, not authored or computed by the SEC itself — must not be mismarked resell:ok like the CAMD/FTD/MIDAS/crop-conditions/NRC streams");
  assert.ok(LICENSE_MARKS["data/13f-holdings"].license.includes("13F-HR"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_thirteenf_holdings");
  assert.ok(tool, "voltrade_thirteenf_holdings tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/13f-holdings"]);
  assert.ok(tool.description.includes("full"), "honesty: the deliberate full-holdings (not top-25-trimmed) response shape must travel with the tool description");
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("eu-macro license mark: commercial reuse permitted with attribution resells freely like the public-domain CAMD/FTD/MIDAS/crop-conditions/NRC streams, not conditional like OCC/Cboe/issuer-authored data; agent tool documents it as a REGIME INPUT with gate-2 unattempted", () => {
  assert.equal(LICENSE_MARKS["stats/eu-macro"].resell, "ok",
    "all three source licenses (ECB/Eurostat/Bundesbank) were verified commercial-reuse-permitted-with-attribution at build time — must not be mismarked conditional like OCC/Cboe/issuer-authored streams");
  assert.ok(LICENSE_MARKS["stats/eu-macro"].license.includes("ECB"));
  assert.ok(LICENSE_MARKS["stats/eu-macro"].license.includes("Eurostat"));
  assert.ok(LICENSE_MARKS["stats/eu-macro"].license.includes("Bundesbank"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_eu_macro");
  assert.ok(tool, "voltrade_eu_macro tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/eu-macro"]);
  assert.ok(tool.description.includes("REGIME INPUT"), "honesty: the regime-input-only framing must travel with the tool description, same as fredMacro");
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("fred-macro license mark: 28 public Fed/US-gov series resell freely like the eu-macro/CAMD/FTD/MIDAS/crop-conditions/NRC streams; the 3 restricted (VIXCLS/BAMLH0A0HYM2/UMCSENT) series stay excluded, never product-surfaced; agent tool documents it as a REGIME INPUT with gate-2 unattempted", () => {
  assert.equal(LICENSE_MARKS["stats/fred-macro"].resell, "ok",
    "the exposed series are Fed/US-government-produced (buildMacroPayload already strips the 3 restricted series) — must not be mismarked conditional like OCC/Cboe/issuer-authored streams");
  assert.ok(LICENSE_MARKS["stats/fred-macro"].license.includes("FRED"));
  assert.ok(LICENSE_MARKS["stats/fred-macro"].license.includes("restricted"),
    "the license text must itself document that the 3 third-party-copyrighted series are excluded, not silently omitted");
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_fred_macro");
  assert.ok(tool, "voltrade_fred_macro tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/fred-macro"]);
  assert.ok(tool.description.includes("REGIME INPUT"), "honesty: the regime-input-only framing must travel with the tool description, same as eu-macro");
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
  assert.ok(tool.description.includes("EXCLUDED"), "honesty: the 3 restricted series' exclusion must travel with the tool description, not just the license mark");
});

test("bank-failures license mark: public-domain US-gov data resells freely like crop-conditions/NRC/eu-macro/fred-macro, not conditional like OCC/Cboe/issuer-authored streams; agent tool documents gate-2 as unattempted and the null-cost honesty rule", () => {
  assert.equal(LICENSE_MARKS["data/bank-failures"].resell, "ok",
    "the FDIC's own failures endpoint is US federal government work — must not be mismarked conditional like the OCC/Cboe/issuer-authored streams");
  assert.ok(LICENSE_MARKS["data/bank-failures"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["data/bank-failures"].attribution.includes("FDIC"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_bank_failures");
  assert.ok(tool, "voltrade_bank_failures tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/bank-failures"]);
  assert.ok(tool.description.includes("GATE 1"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("NOT been gate-2 tested"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
  assert.ok(tool.description.includes("null"), "honesty: the never-coerce-cost-to-zero rule must travel with the tool description");
});

test("gnss-integrity-signal license mark: aircraft-archive-derived ODbL share-alike like tracks/aircraft, not conditional like the AIS-derived stats; agent tool documents it as the first gate-2-passed signal on the API", () => {
  assert.equal(LICENSE_MARKS["data/gnss-integrity-signal"].resell, "share-alike",
    "inherits ODbL from adsb.lol via the aircraft archive, same lineage as tracks/aircraft — must not be mismarked conditional like the AIS-derived stats");
  assert.ok(LICENSE_MARKS["data/gnss-integrity-signal"].license.includes("ODbL"));
  assert.ok(LICENSE_MARKS["data/gnss-integrity-signal"].license.includes("MONETIZATION TRIPWIRE"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_gnss_integrity_signal");
  assert.ok(tool, "voltrade_gnss_integrity_signal tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/gnss-integrity-signal"]);
  assert.ok(tool.description.includes("GATE 2"), "honesty: gate-2-pass status must travel with the tool description");
  assert.ok(tool.description.includes("PARTIAL"), "honesty: gate-1's partial (not full) status must travel with the tool description");
  assert.ok(tool.description.includes("NOT tradeable"), "honesty: this is a statistical signal, not a trading decision — must say so");
});

test("dtcc-swaps license mark: SEC-mandated dissemination submitted by reporting swap participants, conditional like OCC/Cboe not public-domain like the US-gov streams; agent tool documents gate-1-pass/gate-2-not-attempted honestly", () => {
  assert.equal(LICENSE_MARKS["data/dtcc-swaps"].resell, "conditional",
    "each event is submitted by the reporting swap participant, not authored by DTCC/SEC — must not be mismarked ok like the US-gov streams");
  assert.ok(LICENSE_MARKS["data/dtcc-swaps"].license.includes("Reg SBSR"));
  assert.ok(LICENSE_MARKS["data/dtcc-swaps"].attribution.includes("DTCC"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_dtcc_swaps");
  assert.ok(tool, "voltrade_dtcc_swaps tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/dtcc-swaps"]);
  assert.ok(tool.description.includes("GATE 1"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("NOT been gate-2 tested"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("fleet-utilization license mark: aircraft-archive-derived ODbL share-alike like tracks/aircraft/gnss-integrity-signal, not conditional like the AIS-derived stats; agent tool documents gate-1-pass/gate-2-not-attempted honestly", () => {
  assert.equal(LICENSE_MARKS["data/fleet-utilization"].resell, "share-alike",
    "inherits ODbL from adsb.lol via the aircraft archive, same lineage as tracks/aircraft — must not be mismarked conditional like the AIS-derived stats");
  assert.ok(LICENSE_MARKS["data/fleet-utilization"].license.includes("ODbL"));
  assert.ok(LICENSE_MARKS["data/fleet-utilization"].license.includes("MONETIZATION TRIPWIRE"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_fleet_utilization");
  assert.ok(tool, "voltrade_fleet_utilization tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/fleet-utilization"]);
  assert.ok(tool.description.includes("GATE 1"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("NOT been gate-2 tested") || tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
  assert.ok(tool.description.includes("LOWER BOUNDS"), "honesty: the adaptive-sampling lower-bound caveat must travel with the tool description");
});

test("insider (SEC Form 4) license mark: issuer/insider-submitted filings are CONDITIONAL resell like earnings-language/13f-holdings, not ok like the government-produced CAMD/FTD/MIDAS streams; agent tool documents the gate-2 KILL honestly, not a silent gate-2-not-attempted", () => {
  assert.equal(LICENSE_MARKS["data/insider"].resell, "conditional",
    "Form 4 filings are submitted by the reporting insider/issuer, not authored or computed by the SEC itself — must not be mismarked ok like the CAMD/FTD/MIDAS/crop-conditions/NRC streams");
  assert.ok(LICENSE_MARKS["data/insider"].license.includes("Form 4"));
  assert.ok(LICENSE_MARKS["data/insider"].license.includes("KILLED"),
    "honesty: the license mark itself must say the buy-clustering signal hypothesis was gate-2 killed, not just gate-1-passed");
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_insider");
  assert.ok(tool, "voltrade_insider tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/insider"]);
  assert.ok(tool.description.includes("GATE 1"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("KILLED"), "honesty: the gate-2 KILL (not just 'not attempted') must travel with the tool description — this hypothesis was actually tested and failed");
});

test("attention (Wikimedia pageviews) license mark: CC0 public-domain-equivalent resells freely like the government-produced CAMD/FTD/MIDAS streams, not conditional like the issuer-authored insider/13f-holdings/earnings-language streams; agent tool documents gate-2 as not attempted", () => {
  assert.equal(LICENSE_MARKS["data/attention"].resell, "ok",
    "Wikimedia computes pageview counts itself from its own server logs and releases them CC0 — must not be mismarked conditional like the issuer-authored streams");
  assert.ok(LICENSE_MARKS["data/attention"].license.includes("CC0"));
  assert.ok(LICENSE_MARKS["data/attention"].license.includes("Wikimedia"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t) => t.name === "voltrade_attention");
  assert.ok(tool, "voltrade_attention tool must exist");
  assert.deepEqual(tool.returns_provenance, ["data/attention"]);
  assert.ok(tool.description.includes("GATE 1"), "honesty: gate-1-pass status must travel with the tool description");
  assert.ok(tool.description.includes("NOT been attempted"), "honesty: gate-2's not-yet-attempted status must travel with the tool description");
});

test("every v1 endpoint documents a preview (or states it needs a live id), so /developers can't silently drift", () => {
  const meta = apiMeta();
  for (const e of meta.endpoints as any[]) {
    if (e.path === "/api/v1/tracks/:kind/:id") continue; // needs a real id, no static preview possible
    assert.ok(typeof e.preview === "string" && e.preview.length > 0, `${e.path} missing a preview route for the docs explorer`);
  }
});

test("honesty: every v1 response and its public preview mirror carry generated_at, so freshness is never silently omitted", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(/const v1Envelope = .*generated_at: new Date\(generatedAt/s.test(routes), "v1Envelope must stamp generated_at");
  for (const call of ['v1Envelope("stats/portdwell", dwellCache.data, dwellCache.at)',
                       'v1Envelope("stats/shadow", shadowCache.data, shadowCache.at)']) {
    assert.ok(routes.includes(call), `expected cache-timestamped envelope call: ${call}`);
  }
  for (const previewSnippet of ['res.json({ ...shadowCache.data, generated_at:',
                                 'res.json({ ...dwellCache.data, generated_at:',
                                 'res.json({ ...archiveStats(), generated_at:']) {
    assert.ok(routes.includes(previewSnippet), `public preview route missing generated_at: ${previewSnippet}`);
  }
});

test("agent tool spec: one tool per LIVE endpoint (meta excluded), gated signals never leak in", () => {
  const spec = agentToolSpec();
  // meta is docs-about-docs and is not a tool; every other live endpoint is.
  const liveDataEndpoints = apiMeta().endpoints.filter((e: any) => e.path !== "/api/v1/meta");
  assert.equal(spec.tools.length, liveDataEndpoints.length, "tool count must track the live data endpoints");
  // Gated signals may be NAMED in excluded_gated (honest roadmap) but must
  // never appear as a callable tool — scan the tools array specifically.
  const toolBlob = JSON.stringify(spec.tools).toLowerCase();
  assert.ok(!toolBlob.includes("tank"), "gated tank-fill signal must never appear as a tool");
  assert.ok(!toolBlob.includes("timeline"), "gated entity timelines must never appear as a tool");
  assert.ok(!toolBlob.includes("openweathermap"), "OWM is a display product, not on the data API");
  // The gated roadmap is surfaced honestly as excluded, not silently dropped.
  assert.deepEqual(spec.excluded_gated, apiMeta().coming_gated);
});

test("agent tool spec: every tool is valid JSON-Schema and its provenance keys resolve to real license marks", () => {
  const spec = agentToolSpec();
  for (const tool of spec.tools as any[]) {
    assert.ok(/^voltrade_[a-z_]+$/.test(tool.name), `tool name not agent-safe: ${tool.name}`);
    assert.equal(tool.input_schema.type, "object", `${tool.name} input_schema must be an object`);
    assert.ok(tool.input_schema.properties && typeof tool.input_schema.properties === "object");
    assert.ok(Array.isArray(tool.returns_provenance) && tool.returns_provenance.length > 0,
      `${tool.name} must declare what it returns`);
    for (const mark of tool.returns_provenance) {
      assert.ok(LICENSE_MARKS[mark], `${tool.name} references unknown license mark ${mark}`);
    }
  }
  // Provenance travels WITH the spec so an agent can cite it, not just the number.
  assert.deepEqual(spec.license_marks, LICENSE_MARKS);
  assert.ok(spec.ground_truth_note.toLowerCase().includes("provenance"));
  assert.ok(apiMeta().agent_tools === "/api/v1/agent-tools", "meta must point agents at the tool spec");
});

test("agent tool spec: /api/v1/agent-tools is wired and public (docs, not data — like /meta)", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('"/api/v1/agent-tools"'), "agent-tools route missing");
  assert.ok(routes.includes("agentToolSpec()"), "agent-tools must serve agentToolSpec()");
  // It sits with the public meta doc, NOT behind requireApiKey (it's a spec, not data).
  const specLine = routes.split("\n").find((l) => l.includes('"/api/v1/agent-tools"')) || "";
  assert.ok(!specLine.includes("requireApiKey"), "the spec itself is public documentation");
});
