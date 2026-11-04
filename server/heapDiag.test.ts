/**
 * heapDiag.test.ts — the read-only heap diagnostic (memory audit 2026-11-03,
 * A0). Pure-module tests plus a source ratchet pinning that /api/health
 * publishes the block and that the registered gauges exist where the audit
 * named the structures.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  HEAP_DIAG_FLAG, heapDiagEnabled, heapDiagnostic, heapComposition, readHeapGauges,
  registerHeapGauge, trackSet, trackMap, _resetHeapGaugesForTests,
} from "./heapDiag";
import { registerBuiltinHeapGauges, DTCC_SEEN_ID_BYTES_EST } from "./heapGauges";

const HERE = path.dirname(fileURLToPath(import.meta.url));

test("flag: on unless VOLTRADE_HEAP_DIAG=0 (read-only, so default on)", () => {
  assert.equal(heapDiagEnabled({}), true);
  assert.equal(heapDiagEnabled({ [HEAP_DIAG_FLAG]: "1" }), true);
  assert.equal(heapDiagEnabled({ [HEAP_DIAG_FLAG]: "0" }), false);
  const off = heapDiagnostic({ [HEAP_DIAG_FLAG]: "0" });
  assert.deepEqual(off, { enabled: false, flag: HEAP_DIAG_FLAG });
});

test("composition: every V8 space is reported in MB with the process split and leak counters", () => {
  const c = heapComposition();
  for (const name of ["new_space", "old_space", "code_space", "large_object_space"]) {
    assert.ok(c.spaces[name], `space ${name} present`);
    assert.ok(c.spaces[name].used_mb >= 0 && c.spaces[name].size_mb >= 0);
  }
  assert.ok(c.process.heap_used_mb > 0 && c.process.rss_mb > 0);
  assert.ok(c.v8.heap_size_limit_mb > 0);
  assert.ok(c.v8.detached_contexts >= 0 && c.v8.native_contexts >= 1);
});

test("gauges: registered closures are read sorted, a throwing gauge becomes an error entry, replace-by-name", () => {
  _resetHeapGaugesForTests();
  const s = new Set<string>(["a", "b"]);
  const m = new Map<string, number>([["x", 1]]);
  trackSet("z.set", s, 10);
  trackMap("a.map", m);
  registerHeapGauge("m.boom", () => { throw new Error("nope"); });
  registerHeapGauge("m.boom", () => { throw new Error("replaced"); });
  const r = readHeapGauges();
  assert.deepEqual(Object.keys(r), ["a.map", "m.boom", "z.set"]);
  assert.deepEqual(r["z.set"], { entries: 2, cap: 10 });
  assert.deepEqual(r["a.map"], { entries: 1 });
  assert.deepEqual(r["m.boom"], { error: "replaced" });
  s.add("c");
  assert.equal((readHeapGauges()["z.set"] as { entries: number }).entries, 3, "gauge is live, not a snapshot");
  const d = heapDiagnostic({});
  assert.ok(d.enabled);
  if (d.enabled) {
    assert.equal(d.gauge_count, 3);
    assert.equal(d.caches["z.set"] && (d.caches["z.set"] as { entries: number }).entries, 3);
  }
  _resetHeapGaugesForTests();
});

test("gauge reads are cheap: 200 gauges read in well under a probe interval", () => {
  _resetHeapGaugesForTests();
  for (let i = 0; i < 200; i++) trackSet(`t.${i}`, new Set([i]));
  const t0 = performance.now();
  for (let k = 0; k < 50; k++) readHeapGauges();
  const perRead = (performance.now() - t0) / 50;
  assert.ok(perRead < 20, `one read of 200 gauges took ${perRead.toFixed(2)}ms`);
  _resetHeapGaugesForTests();
});

test("builtin gauges: dtcc/entityGraph/requestLog register and read without the pollers running", () => {
  _resetHeapGaugesForTests();
  registerBuiltinHeapGauges();
  const r = readHeapGauges();
  assert.deepEqual(Object.keys(r), ["dtcc.seenIds", "entityGraph.graphCache", "requestLog.ring"]);
  const dtcc = r["dtcc.seenIds"] as { entries: number; bytes: number; mode: string; memoryCount: number | null; dbCount: number | null; mismatches: number };
  assert.equal(dtcc.entries, 0, "no poll has run in this process");
  assert.equal(dtcc.bytes, 0 * DTCC_SEEN_ID_BYTES_EST);
  // A1: the gauge carries the dedup store's own view (dual-run default)
  assert.ok(["memory", "dual", "sqlite"].includes(dtcc.mode));
  assert.equal(dtcc.memoryCount, null, "the Set is not built until a poll runs");
  assert.equal(dtcc.dbCount, null, "no DB opened in this process");
  assert.equal(dtcc.mismatches, 0);
  assert.deepEqual(r["entityGraph.graphCache"], { entries: 0 });
  assert.ok((r["requestLog.ring"] as { cap: number }).cap > 0);
  _resetHeapGaugesForTests();
});

// ── source ratchets ─────────────────────────────────────────────────────────
test("bot.ts: /api/health publishes the diagnostic under checks.memory.detail and registers the builtin gauges once", () => {
  const bot = fs.readFileSync(path.join(HERE, "bot.ts"), "utf8");
  const start = bot.indexOf('app.get("/api/health"');
  const end = bot.indexOf("app.get(", start + 10);
  const handler = bot.slice(start, end);
  assert.match(handler, /detail:\s*heapDiagnostic\(\)/, "memory block must carry detail: heapDiagnostic()");
  assert.equal((bot.match(/registerBuiltinHeapGauges\(\)/g) || []).length, 1, "builtin gauges register exactly once");
  // the structures the audit named register a gauge next to their declaration
  const expect: Array<[string, string]> = [
    ["routes.ts", 'trackMap("routes.aircraftCache"'],
    ["routes.ts", 'trackMap("routes.vesselSnapCache"'],
    ["routes.ts", 'trackMap("routes.windowCache"'],
    ["globalAircraft.ts", 'trackMap("globalAircraft.bodyCache"'],
    ["aircraftTrips.ts", 'registerHeapGauge("aircraftTrips.bloomCache"'],
    ["airportsIndex.ts", 'registerHeapGauge("airportsIndex"'],
    ["nasaFirms.ts", 'trackSet("nasaFirms.archivedIds"'],
    ["gdeltEvents.ts", 'trackSet("gdeltEvents.archivedIds"'],
    ["nwsAlerts.ts", 'trackSet("nwsAlerts.archivedIds"'],
    ["fdaEvents.ts", 'trackSet("fdaEvents.archivedKeys"'],
    ["edgarForm4.ts", 'trackSet("edgarForm4.archivedAccessions"'],
    ["bot.ts", 'trackSet("bot.sseClients"'],
    ["bot.ts", 'trackSet("bot.longPythonJobsInFlight"'],
  ];
  for (const [file, needle] of expect) {
    const src = file === "bot.ts" ? bot : fs.readFileSync(path.join(HERE, file), "utf8");
    assert.ok(src.includes(needle), `${file} must register ${needle}`);
  }
});

test("every module-level dedup Set / cache Map under server/ is gauged (sweep ratchet)", () => {
  // The sweep registered one gauge per column-0 `const x = new Set<…>()` /
  // `new Map<…>()` dedup/cache declaration (constant lookup tables excluded).
  // Pin the floor so a new poller cannot add an ungauged retained structure
  // without this number moving in the same PR.
  let registrations = 0;
  let gaugedFiles = 0;
  for (const f of fs.readdirSync(HERE)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts") || f === "heapDiag.ts") continue;
    const src = fs.readFileSync(path.join(HERE, f), "utf8");
    const n = (src.match(/\b(?:trackSet|trackMap|registerHeapGauge)\("/g) || []).length;
    if (n > 0) gaugedFiles++;
    registrations += n;
  }
  assert.ok(registrations >= 78, `expected >= 78 gauge registrations across server/, found ${registrations}`);
  assert.ok(gaugedFiles >= 58, `expected >= 58 gauged modules, found ${gaugedFiles}`);
  // frozen modules are never touched by the sweep
  for (const frozen of ["auth.ts", "billing.ts"]) {
    assert.ok(!fs.readFileSync(path.join(HERE, frozen), "utf8").includes("heapDiag"), `${frozen} is FROZEN`);
  }
});

test("the audit this diagnostic baselines is checked in", () => {
  const p = path.join(HERE, "..", "docs", "memory_audit_2026-11-03.md");
  assert.ok(fs.existsSync(p), "docs/memory_audit_2026-11-03.md must exist");
  const txt = fs.readFileSync(p, "utf8");
  assert.match(txt, /DTCC/);
  assert.match(txt, /Proposed fixes/i);
});
