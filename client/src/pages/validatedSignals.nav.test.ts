// Source-text pins for the Signals PRODUCT hub (#/data/validated-signals,
// PLATFORM INTEGRATION PROGRAM). The "Signals — validated signal products"
// nav leaf shipped disabled as `soon: true` on 2026-07-11 and stayed that way
// for four months while two gate-2-passed signals were live on /data and
// /api/v1. Pinned so it cannot quietly go back to "soon", and so the overlay
// stays wired end to end (nav leaf -> hash -> datamap overlay -> component).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const read = (rel: string) => readFileSync(path.resolve(here, rel), "utf8");
const home = read("./home.tsx");
const datamap = read("./datamap.tsx");
const view = read("./validatedSignals.tsx");
const css = read("../index.css");

test("the Signals nav leaf is live (not `soon`) and routes to the product hub hash", () => {
  const leaf = home.match(/\{ key: "signals",[^\n]*\}/)?.[0];
  assert.ok(leaf, "Signals leaf present in the Data menu");
  assert.doesNotMatch(leaf!, /soon: true/, "the leaf is no longer disabled as 'soon'");
  assert.match(leaf!, /action: \{ kind: "hash", hash: "#\/data\/validated-signals" \}/);
});

test("datamap wires the hub as an overlay: state, hashchange listener, render, launcher", () => {
  assert.match(datamap, /import ValidatedSignalsView from "\.\/validatedSignals";/);
  assert.match(datamap, /useState\(\(\) => window\.location\.hash === "#\/data\/validated-signals"\)/);
  assert.match(datamap, /setValidatedSignalsOpen\(window\.location\.hash === "#\/data\/validated-signals"\);/);
  assert.match(datamap, /\{validatedSignalsOpen && \(\s*<ValidatedSignalsView onBack=/);
  assert.match(datamap, /data-vt-validated-signals-launch/);
});

test("the hub reads the catalog route and surfaces the honesty contract, not a bare product list", () => {
  assert.match(view, /fetch\("\/api\/data\/signal-products"\)/);
  assert.match(view, /data\.honesty/, "server-side honesty statement rendered verbatim");
  assert.match(view, /data\.unwired\.length > 0/, "validated-but-unwired roots are shown, never hidden");
  assert.match(view, /not tradeable · gate 3 not passed/);
  assert.match(view, /relativeAge\(p\.live\.generated_at, now\)/, "every live reading carries its age");
  assert.match(view, /licensePosture\(p\.license\.resell\)/, "license posture on every card");
});

test("hub CSS uses theme tokens only (no hex literals) and keeps 44px touch targets", () => {
  const block = css.slice(css.indexOf("Signals PRODUCT hub"));
  assert.ok(block.length > 0, "css block present");
  assert.doesNotMatch(block, /#[0-9a-fA-F]{3,8}\b/, "no hardcoded hex colors in the hub block");
  assert.match(block, /\.vt-sigprod-btn \{[^}]*min-height: 44px/);
});
