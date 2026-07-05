import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  normalizeCompanyName,
  buildTickerMap,
  parseTxnRow,
  fetchRecentContracts,
  resolveTickers,
  applyParentMatch,
  archiveContractTxns,
  loadUeiCache,
  saveUeiCache,
  CONTRACT_FLOOR_USD,
} from "./usaSpending";

const here = path.dirname(fileURLToPath(import.meta.url));

// REAL sample row captured live from POST /api/v2/search/spending_by_transaction
// on 2026-07-05 — the hypothesis exemplar ($900M DOE task order to a Centrus
// Energy (LEU) subsidiary). Field names are the API's display names.
const REAL_ROW = {
  "Award ID": "89243226FNE400212", "Mod": "0",
  "Recipient Name": "AMERICAN CENTRIFUGE OPERATING, LLC",
  "Recipient UEI": "L8VHV5CNBV97", "Action Date": "2026-07-01",
  "Transaction Amount": 900000000.0,
  "Transaction Description": "THE PURPOSE OF THIS TASK ORDER (TO) IS TO ESTABLISH NEW ANNUAL DOMESTIC COMMERCIAL HIGH-ASSAY LOW-ENRICHED URANIUM (HALEU) CAPACITY...",
  "Award Type": "DELIVERY ORDER", "Awarding Agency": "Department of Energy",
  "Awarding Sub Agency": "Department of Energy",
  "NAICS": { code: "325180", description: "OTHER BASIC INORGANIC CHEMICAL MANUFACTURING" },
  "PSC": { code: "6850", description: "MISCELLANEOUS CHEMICAL SPECIALTIES" },
  "recipient_id": "d527144c-7fea-82ff-aff0-e95e5fd6e488-C",
  "generated_internal_id": "CONT_AWD_89243226FNE400212_8900_89243225DNE000022_8900",
};

test("normalizeCompanyName: suffix stripping makes filer and SEC names meet; too-short residue yields no key", () => {
  assert.equal(normalizeCompanyName("LEIDOS, INC."), normalizeCompanyName("Leidos Holdings, Inc."));
  assert.equal(normalizeCompanyName("The Boeing Company"), "BOEING");
  assert.equal(normalizeCompanyName("AMERICAN CENTRIFUGE OPERATING, LLC"), "AMERICAN CENTRIFUGE OPERATING");
  assert.equal(normalizeCompanyName("Co, Inc."), "", "all-suffix names must not produce a match key");
  assert.equal(normalizeCompanyName(null), "");
});

test("buildTickerMap: exact keys only; ambiguous normalized names dropped (never guess)", () => {
  const map = buildTickerMap({
    "0": { cik_str: 1, ticker: "BA", title: "Boeing Company" },
    "1": { cik_str: 2, ticker: "LDOS", title: "Leidos Holdings, Inc." },
    "2": { cik_str: 3, ticker: "AAA", title: "Acme Widgets Inc" },
    "3": { cik_str: 4, ticker: "BBB", title: "ACME WIDGETS CORP" }, // same normalized name, different ticker
  });
  assert.equal(map.get("BOEING"), "BA");
  assert.equal(map.get("LEIDOS"), "LDOS");
  assert.equal(map.get("ACME WIDGETS"), undefined, "collision must drop both");
});

test("parseTxnRow maps the real API row, hand-checked", () => {
  const t = parseTxnRow(REAL_ROW, "2026-07-05");
  assert.equal(t.aid, "CONT_AWD_89243226FNE400212_8900_89243225DNE000022_8900");
  assert.equal(t.mod, "0");
  assert.equal(t.amt, 900000000);
  assert.equal(t.uei, "L8VHV5CNBV97");
  assert.equal(t.naics, "325180");
  assert.equal(t.ag, "Department of Energy");
  assert.ok(t.desc!.length <= 300);
  assert.equal(t.rt, "2026-07-05");
  assert.equal(t.tkr, null, "no mapping at parse time");
});

test("fetchRecentContracts: two sorted passes stop at the floor; deobligations kept; floor rows excluded", async () => {
  const calls: any[] = [];
  const mkRow = (aid: string, amt: number) => ({ ...REAL_ROW, "generated_internal_id": aid, "Transaction Amount": amt });
  const impl = async (_url: string, init: any) => {
    const body = JSON.parse(init.body);
    calls.push(body);
    const rows = body.order === "desc"
      ? [mkRow("A1", 900000000), mkRow("A2", 50000), mkRow("A3", 24999), mkRow("A4", 10)]
      : [mkRow("D1", -5400000), mkRow("D2", -25000), mkRow("D3", -24999), mkRow("D4", 0)];
    return { ok: true, status: 200, text: async () => JSON.stringify({ results: rows, page_metadata: { hasNext: true } }) };
  };
  const txns = await fetchRecentContracts(impl as any, Date.parse("2026-07-05T12:00:00Z"), 0);
  const ids = txns.map((t) => t.aid).sort();
  assert.deepEqual(ids, ["A1", "A2", "D1", "D2"], "kept |amt| >= floor only, both signs");
  assert.equal(calls.length, 2, "early stop: one page per pass despite hasNext");
  assert.ok(calls.every((c) => c.filters.time_period[0].date_type === "last_modified_date"),
    "must poll by last_modified_date (publication axis), never action_date");
  assert.ok(txns.every((t) => Math.abs(t.amt) >= CONTRACT_FLOOR_USD));
});

test("resolveTickers: cache wins, then exact name; unmatched large rows queue for parent; small unmatched stay null", () => {
  const map = buildTickerMap({ "0": { cik_str: 1, ticker: "BA", title: "Boeing Company" } });
  const cache: any = { UEI_CACHED: { tkr: "LDOS", mm: "name", at: "2026-07-01" } };
  const mk = (r: string, uei: string, amt: number) =>
    parseTxnRow({ ...REAL_ROW, "Recipient Name": r, "Recipient UEI": uei, "Transaction Amount": amt, "generated_internal_id": r }, "2026-07-05");
  const txns = [mk("THE BOEING COMPANY", "UEI_B", 30000), mk("Whoever LLC", "UEI_CACHED", 30000),
                mk("Unknown Sub LLC", "UEI_U", 2000000), mk("Tiny Vendor LLC", "UEI_T", 30000)];
  const { needParent } = resolveTickers(txns, map, cache);
  assert.equal(txns[0].tkr, "BA");
  assert.equal(txns[0].mm, "name");
  assert.equal(cache["UEI_B"].tkr, "BA", "resolution compounds into the UEI cache");
  assert.equal(txns[1].tkr, "LDOS");
  assert.equal(txns[1].mm, "cache");
  assert.deepEqual(needParent.map((t) => t.r), ["Unknown Sub LLC"], "only large unmatched rows fetch parents");
  assert.equal(txns[3].tkr, null, "small unmatched rows stay null — never fuzzy");
});

test("applyParentMatch resolves via award-detail parent and records the audit trail", () => {
  const map = buildTickerMap({ "0": { cik_str: 1, ticker: "LEU", title: "Centrus Energy Corp" } });
  const cache: any = {};
  const t = parseTxnRow(REAL_ROW, "2026-07-05");
  applyParentMatch(t, "CENTRUS ENERGY CORP", "PARENT_UEI", map, cache);
  assert.equal(t.tkr, "LEU");
  assert.equal(t.mm, "parent");
  assert.equal(t.mname, "CENTRUS ENERGY");
  assert.equal(t.pn, "CENTRUS ENERGY CORP");
  assert.equal(cache["L8VHV5CNBV97"].tkr, "LEU", "child UEI now cached forever");
});

test("archive round-trip: (aid,mod,amt) dedup; a corrected amount appends as a new vintage row", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtusa-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const t = parseTxnRow(REAL_ROW, "2026-07-05");
  assert.equal(archiveContractTxns([t], dir, t0), 1);
  assert.equal(archiveContractTxns([t], dir, t0), 0, "same (aid,mod,amt) never twice");
  const corrected = { ...t, amt: 899000000, rt: "2026-07-06" };
  assert.equal(archiveContractTxns([corrected], dir, t0 + 86400_000), 1, "correction appends");
  // UEI cache persistence
  saveUeiCache({ X: { tkr: "BA", mm: "name", at: "2026-07-05" } } as any, dir);
  assert.equal(loadUeiCache(dir).X.tkr, "BA");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("routes.ts boots the contracts poll and registers /api/data/contracts; manifest exists with the DUNS + DoD honesty lines", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootContractsPoll()"), "contracts poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/contracts"'), "contracts route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "usaspending.json"), "utf8"));
  assert.equal(manifest.stream, "usaspending");
  assert.ok(String(manifest.license).includes("DUNS"), "D&B DUNS restriction must be stated");
  assert.ok(String(manifest.confidence_model).includes("90 DAYS"), "DoD publication-delay honesty must be stated");
  assert.ok(JSON.stringify(manifest).includes("25,000") || JSON.stringify(manifest).includes("$25"), "explicit cap stated");
});
