import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { tickerForFleetOwner, allFleetOperatorTickers } from "./fleetOperatorTickers";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const jsonPath = path.join(root, "datacore", "fleet_operator_tickers.json");
const raw = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

test("every entry has required fields, a valid confidence/thesis, and a real ticker", () => {
  assert.ok(Array.isArray(raw.entries) && raw.entries.length > 0);
  const seen = new Set<string>();
  for (const e of raw.entries) {
    assert.ok(typeof e.operator === "string" && e.operator.length > 0);
    assert.ok(!seen.has(e.operator), `duplicate operator key: ${e.operator}`);
    seen.add(e.operator);
    assert.ok(["high", "medium"].includes(e.confidence), `bad confidence: ${e.confidence}`);
    assert.ok(["operational_proxy", "control_comparison"].includes(e.thesis), `bad thesis: ${e.thesis}`);
    assert.ok(typeof e.ticker === "string" && /^[A-Z.]+$/.test(e.ticker), `bad ticker for ${e.operator}`);
    assert.ok(typeof e.note === "string" && e.note.length > 20, `missing provenance note for ${e.operator}`);
  }
});

test("tickerForFleetOwner resolves a known owner exactly", () => {
  const tmdx = tickerForFleetOwner("TRANSMEDICS INC");
  assert.ok(tmdx);
  assert.equal(tmdx!.ticker, "TMDX");
  assert.equal(tmdx!.thesis, "operational_proxy");

  const lh = tickerForFleetOwner("LABORATORY CORPORATION OF AMERICA HOLDINGS");
  assert.ok(lh);
  assert.equal(lh!.ticker, "LH");
  assert.equal(lh!.thesis, "control_comparison");
});

test("tickerForFleetOwner never guesses — unknown or unmapped owners return null", () => {
  assert.equal(tickerForFleetOwner("UNITED AIRLINES"), null);
  assert.equal(tickerForFleetOwner("SOME RANDOM LLC"), null);
  assert.equal(tickerForFleetOwner(""), null);
});

test("allFleetOperatorTickers exposes exactly the JSON file's entries", () => {
  assert.equal(allFleetOperatorTickers().length, raw.entries.length);
});
