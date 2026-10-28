// regimeCounterfactual — aggregates of the regime engine's legacy-vs-engine
// counterfactual log (2026-10-28 live flip). Run: npx tsx --test server/regimeCounterfactual.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "node:url";
import {
  REGIME_COUNTERFACTUAL_FILE, readRegimeCounterfactual, summarizeRegimeCounterfactual, regimeCounterfactualBody,
} from "./regimeCounterfactual";
import { DIAG_PROBES } from "./diag";

const regime = { as_of: "2026-10-27", composite: "STRESSED", stale: false };
const rows = [
  { date: "2026-10-28", kind: "sizing", sleeve: "stock", ticker: "AMD", legacy: { multiplier: 0.375, vix_scalar: 0.75, stress_mult: 0.5 }, engine: { multiplier: 0.5, applied: true, stale: false }, differs: true, regime },
  { date: "2026-10-28", kind: "sizing", sleeve: "stock", ticker: "NVDA", legacy: { multiplier: 0.625 }, engine: { multiplier: 0.5, applied: true, stale: false }, differs: true, regime },
  { date: "2026-10-28", kind: "sizing", sleeve: "tier1", legacy: { multiplier: 0.85 }, engine: { multiplier: 1.0, applied: true, stale: true }, differs: true, regime },
  { date: "2026-10-28", kind: "selection", ticker: "XYZ", legacy: { decision: "take" }, engine: { decision: "block", blocked_by: ["knife"], would_block: ["knife", "rs"] }, differs: true, shadow_decision: "rejected_regime_knife", regime },
  { date: "2026-10-28", kind: "selection", ticker: "ABC", legacy: { decision: "take" }, engine: { decision: "take", blocked_by: [], would_block: [], dip_neutralized: true }, differs: true, regime },
  { date: "2026-10-28", kind: "options", ticker: "KO", legacy: { decision: "take", size_mult: 1 }, engine: { decision: "take", blocked_by: [], would_block: [], size_mult: 0.5 }, differs: true, regime },
  { date: "2026-10-28", kind: "label", legacy: { label: "NEUTRAL_BULL" }, engine: { label: "CAUTION", applied: true }, differs: true, regime },
  { date: "2026-10-27", kind: "label", legacy: { label: "NEUTRAL_BULL" }, engine: { label: "BULL", applied: false }, differs: true, regime },
];

test("summary: per day per kind, means per sleeve, blocks by guard, label pairs — and no ticker", () => {
  const out = summarizeRegimeCounterfactual(rows);
  assert.deepEqual(out.map((d) => d.date), ["2026-10-27", "2026-10-28"]);
  const d = out[1];
  assert.deepEqual(d.sizing.stock, { n: 2, applied: 2, stale: 0, mean_legacy_multiplier: 0.5, mean_engine_multiplier: 0.5 });
  assert.deepEqual(d.sizing.tier1, { n: 1, applied: 1, stale: 1, mean_legacy_multiplier: 0.85, mean_engine_multiplier: 1 });
  assert.deepEqual(d.selection.decisions, { block: 1, take: 1 });
  assert.deepEqual(d.selection.blocks_by_guard, { knife: 1 });
  assert.deepEqual(d.selection.would_block_by_guard, { knife: 1, rs: 1 });
  assert.equal(d.selection.dip_neutralized, 1);
  assert.equal(d.selection.linked_shadow, 1);
  assert.equal(d.options.size_cut, 1);
  assert.deepEqual(d.label.pairs, { "NEUTRAL_BULL>CAUTION": 1 });
  assert.equal(out[0].label.applied, 0);
  assert.equal(d.composites.STRESSED, 7);
  const text = JSON.stringify(out);
  for (const t of ["AMD", "NVDA", "XYZ", "ABC", "KO"]) assert.ok(!text.includes(t), `ticker ${t} must not leave the aggregate`);
});

test("reader: current + rotated file, date window, torn lines skipped, missing dir empty", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "regime-cf-"));
  assert.deepEqual(readRegimeCounterfactual(dir, "2026-01-01"), []);
  fs.writeFileSync(path.join(dir, `${REGIME_COUNTERFACTUAL_FILE}.1`), JSON.stringify(rows[7]) + "\n");
  fs.writeFileSync(path.join(dir, REGIME_COUNTERFACTUAL_FILE), rows.slice(0, 7).map((r) => JSON.stringify(r)).join("\n") + "\n{\"date\":\"2026-10-28\",\"ki");
  assert.equal(readRegimeCounterfactual(dir, "2026-10-01").length, 8);
  assert.equal(readRegimeCounterfactual(dir, "2026-10-28").length, 7);
  const body = regimeCounterfactualBody(dir, new Date("2026-10-28T20:00:00Z"), "2");
  assert.equal(body.days, 2);
  assert.equal(body.since, "2026-10-27");
  assert.equal(body.rows, 8);
  assert.equal(regimeCounterfactualBody(dir, new Date("2026-10-28T20:00:00Z"), "1").rows, 7);
  assert.equal(regimeCounterfactualBody(dir, new Date("2026-10-28T20:00:00Z"), "9999").days, 120);
  assert.equal(regimeCounterfactualBody(dir, new Date("2026-10-28T20:00:00Z"), "junk").days, 14);
});

test("the probe is whitelisted and wired in bot.ts", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("regime_counterfactual"));
  const bot = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "bot.ts"), "utf8");
  assert.ok(bot.includes('case "regime_counterfactual"'));
  assert.ok(bot.includes("regimeCounterfactualBody(regimeDataDir(process.env), new Date(), req.query.days)"));
});
