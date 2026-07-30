// qualityDashboard.test.ts — MAP V2 ROADMAP R6(b) DATA-QUALITY dashboard.
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildStreamHealthSummary, buildArchiveSummary, verificationCoverage } from "./qualityDashboard";

test("buildStreamHealthSummary counts every health bucket, including zero-count ones", () => {
  const s = buildStreamHealthSummary([
    { health: "live" }, { health: "live" }, { health: "recent" },
    { health: "stale" }, { health: "no-data" },
  ] as any);
  assert.deepEqual(s, { total: 5, live: 2, recent: 1, stale: 1, no_data: 1 });
});

test("buildStreamHealthSummary on an empty inventory is all zeros, never crashes", () => {
  assert.deepEqual(buildStreamHealthSummary([]), { total: 0, live: 0, recent: 0, stale: 0, no_data: 0 });
});

test("buildArchiveSummary sorts kinds largest-first and sums totals", () => {
  const out = buildArchiveSummary({
    totalBytes: 300,
    kinds: {
      small: { files: 1, bytes: 50 },
      big: { files: 3, bytes: 200 },
      mid: { files: 2, bytes: 50 },
    },
  });
  assert.equal(out.total_bytes, 300);
  assert.equal(out.total_files, 6);
  assert.deepEqual(out.kinds.map((k) => k.kind), ["big", "small", "mid"]);
});

test("buildArchiveSummary handles an empty archive without crashing", () => {
  const out = buildArchiveSummary({ totalBytes: 0, kinds: {} });
  assert.deepEqual(out, { total_bytes: 0, total_files: 0, kinds: [] });
});

test("verificationCoverage: sites/ports are honestly 100%-of-total by the registry's own blanket claim, plants is a real partial fraction", () => {
  const v = verificationCoverage();
  // sites: verified === total by construction (strategic_sites.json's own
  // "ALL 16 imagery-verified" doc claim covers every row, not a subset)
  assert.ok(v.sites.total > 0, "expected at least one strategic site");
  assert.equal(v.sites.verified, v.sites.total);
  // ports: a strict subset of sites, also fully verified, and no larger
  // than the sites total
  assert.ok(v.ports.total > 0, "expected at least one port geofence");
  assert.equal(v.ports.verified, v.ports.total);
  assert.ok(v.ports.total <= v.sites.total);
  // plants: a genuine partial fraction (top-100-by-MW out of the full US
  // registry) — this is the one honest "not fully covered" number
  assert.ok(v.plants.total > 9000, "expected ~9.8k US plants");
  assert.equal(v.plants.verified, 100);
  assert.ok(v.plants.verified < v.plants.total);
});
