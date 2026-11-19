// 2026-11-19: Node's fatal-error report, kept on the volume and summarised by
// the next boot (server/fatalReport.ts). The fixture below is the shape of a
// REAL report produced on 2026-11-19 by running Node into its heap limit with
// reportOnFatalError set at runtime (event / trigger / heap / native frames).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import {
  createFatalReports, fatalReportDir, summarizeReport, FATAL_REPORT, FATAL_REPORT_SUBDIR,
  FATAL_REPORT_MAX_PER_BOOT, FATAL_REPORT_MAX_BYTES, type ReportApi,
} from "./fatalReport";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OOM = JSON.stringify({
  header: { reportVersion: 3, event: "Allocation failed - JavaScript heap out of memory", trigger: "OOMError",
    dumpEventTime: "2026-11-18T14:51:09Z", nodejsVersion: "v20.19.0", cwd: "/app", commandLine: ["node", "--max-old-space-size=6144", "dist/index.cjs"] },
  javascriptStack: { message: "No stack.", stack: ["Unavailable."] },
  javascriptHeap: { totalMemory: 6_500_000_000, usedMemory: 6_430_000_000, memoryLimit: 6_492_782_592 },
  nativeStack: [{ pc: "0x1", symbol: " [node]" }, { pc: "0x2", symbol: "node::TriggerNodeReport(v8::Isolate*, char const*) [node]" },
    { pc: "0x3", symbol: "node::OOMErrorHandler(char const*, v8::OOMDetails const&) [node]" },
    { pc: "0x4", symbol: "v8::internal::Heap::FatalProcessOutOfMemory(char const*) [node]" }],
  resourceUsage: { rss: 7_100_000_000, maxRss: 7_200_000_000 },
  environmentVariables: { ALPACA_SECRET_KEY: "sk-do-not-leak", DIAG_TOKEN: "tok-do-not-leak" },
});

function tmp(): string { return fs.mkdtempSync(path.join(os.tmpdir(), "fatalrep-")); }

test("the reports live on the volume, so they outlive the container that aborted", () => {
  assert.equal(fatalReportDir({ DATA_DIR: "/data/voltrade" }, () => false), "/data/voltrade/" + FATAL_REPORT_SUBDIR);
  assert.equal(fatalReportDir({}, (p) => p === "/data"), "/data/voltrade/" + FATAL_REPORT_SUBDIR);
  assert.ok(fatalReportDir({}, () => false).endsWith(path.sep + FATAL_REPORT_SUBDIR));
});

test("a heap-limit abort is summarised: event, heap against its limit, memory, the useful native frames", () => {
  const s = summarizeReport("/x/report.20261118.145109.1.0.001.json", OOM);
  assert.equal(s.event, "Allocation failed - JavaScript heap out of memory");
  assert.equal(s.trigger, "OOMError");
  assert.equal(s.at, "2026-11-18T14:51:09Z");
  assert.deepEqual([s.heapUsedMB, s.heapLimitMB, s.rssMB], [6132, 6192, 6771]);
  assert.deepEqual(s.native, ["node::OOMErrorHandler(char const*, v8::OOMDetails const&) [node]", "v8::internal::Heap::FatalProcessOutOfMemory(char const*) [node]"]);
  assert.match(s.line, /^the previous process ended ITSELF with a fatal error at 2026-11-18T14:51:09Z: Allocation failed - JavaScript heap out of memory \(OOMError\)\. JS heap 6132 MB of a 6192 MB limit, process memory 6771 MB\./);
});

test("no secret can reach the summary: environment, command line and cwd are never read", () => {
  const s = summarizeReport("/x/report.a.json", OOM);
  const all = JSON.stringify(s);
  for (const leak of ["sk-do-not-leak", "tok-do-not-leak", "ALPACA_SECRET_KEY", "max-old-space-size", "/app"]) {
    assert.equal(all.includes(leak), false, leak);
  }
});

test("an unreadable or foreign file is reported as such, never thrown", () => {
  for (const raw of ["", "{not json", "[]", '"x"', "{}"]) {
    const s = summarizeReport("/x/report.b.json", raw);
    assert.equal(s.event, "unreadable report");
    assert.deepEqual([s.heapUsedMB, s.heapLimitMB, s.rssMB, s.native.length], [null, null, null, 0]);
    assert.match(s.line, /JS heap unknown of a unknown limit/);
  }
});

test("boot: every report is summarised to the audit log and the raw file is deleted", () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, "report.20261118.145109.1.0.001.json"), OOM);
  fs.writeFileSync(path.join(dir, "notes.txt"), "not a report");
  const rows: Array<[string, string]> = [];
  const fr = createFatalReports({ dir, fs, audit: (a, d) => { rows.push([a, d]); } });
  const out = fr.collect();
  assert.equal(out.length, 1);
  assert.equal(rows.length, 1);
  assert.equal(rows[0][0], FATAL_REPORT);
  assert.match(rows[0][1], /heap out of memory/);
  assert.equal(rows[0][1].includes("do-not-leak"), false);
  assert.deepEqual(fs.readdirSync(dir), ["notes.txt"]);                    // the raw report is gone; other files untouched
  const st = fr.status();
  assert.equal(st.found_at_boot, 1);
  assert.equal(st.last?.event, "Allocation failed - JavaScript heap out of memory");
  assert.equal(st.gates_top_level_status, false);
  assert.equal(fr.collect().length, 0);                                    // nothing left to find
});

test("boot: a burst leaves many reports — the newest five are summarised, all are deleted", () => {
  const dir = tmp();
  for (let i = 0; i < 8; i++) {
    const p = path.join(dir, `report.2026111${i}.000000.1.0.001.json`);
    fs.writeFileSync(p, OOM);
    fs.utimesSync(p, new Date(2026, 10, 10 + i), new Date(2026, 10, 10 + i));
  }
  const rows: string[] = [];
  const fr = createFatalReports({ dir, fs, audit: (_a, d) => { rows.push(d); } });
  assert.equal(fr.collect().length, FATAL_REPORT_MAX_PER_BOOT);
  assert.equal(rows.length, FATAL_REPORT_MAX_PER_BOOT + 1);
  assert.match(rows[rows.length - 1], /^3 older fatal-error report\(s\) deleted unread/);
  assert.match(rows[0], /report\.20261117/);                               // newest first
  assert.deepEqual(fs.readdirSync(dir), []);
});

test("boot: an oversized file is not read into memory, and a missing directory is simply nothing", () => {
  const dir = tmp();
  const big = path.join(dir, "report.big.json");
  fs.writeFileSync(big, "x");
  const fakeFs = { ...fs, statSync: () => ({ size: FATAL_REPORT_MAX_BYTES + 1, mtimeMs: 1 }), readFileSync: () => { throw new Error("must not be read"); } };
  const rows: string[] = [];
  const fr = createFatalReports({ dir, fs: fakeFs as unknown as typeof fs, audit: (_a, d) => { rows.push(d); } });
  assert.equal(fr.collect().length, 1);
  assert.match(rows[0], /was not read \(\d+ bytes\); deleted/);
  assert.deepEqual(fs.readdirSync(dir), []);
  const none = createFatalReports({ dir: path.join(dir, "missing"), fs, audit: () => { throw new Error("no rows expected"); } });
  assert.deepEqual(none.collect(), []);
});

test("enable: the report is pointed at the volume, compact, without environment where Node can leave it out", () => {
  const dir = path.join(tmp(), "fatal_reports");
  const modern: ReportApi = { directory: "", reportOnFatalError: false, compact: false, excludeEnv: false };
  const fr = createFatalReports({ dir, fs, audit: () => {} });
  assert.equal(fr.status().status, "off");
  assert.equal(fr.enable(modern), true);
  assert.deepEqual(modern, { directory: dir, reportOnFatalError: true, compact: true, excludeEnv: true });
  assert.ok(fs.existsSync(dir));
  assert.deepEqual([fr.status().status, fr.status().env_excluded], ["ok", true]);
  const node20: ReportApi = { directory: "", reportOnFatalError: false };
  const fr20 = createFatalReports({ dir, fs, audit: () => {} });
  assert.equal(fr20.enable(node20), true);
  assert.equal("excludeEnv" in node20, false);                              // never invented on a runtime that lacks it
  assert.equal(fr20.status().env_excluded, false);
  assert.equal(createFatalReports({ dir, fs, audit: () => {} }).enable(null), false);
  const broken = { ...fs, mkdirSync: () => { throw new Error("ENOSPC"); } };
  const frBroken = createFatalReports({ dir, fs: broken as unknown as typeof fs, audit: () => {} });
  assert.equal(frBroken.enable({ directory: "", reportOnFatalError: false }), false);
  assert.deepEqual([frBroken.status().status, frBroken.status().faults], ["off", 1]);
});

test("REAL RUN: a Node process driven into its heap limit leaves a report this module reads back", () => {
  const dir = tmp();
  const code = `process.report.directory=${JSON.stringify(dir)};process.report.reportOnFatalError=true;process.report.compact=true;`
    + `const a=[];for(;;)a.push(new Array(1e5).fill(Math.random()));`;
  const r = spawnSync(process.execPath, ["--max-old-space-size=24", "-e", code], { env: { PATH: process.env.PATH ?? "", LEAK_CHECK: "leak-me-not" }, timeout: 60_000 });
  assert.notEqual(r.status, 0);
  const rows: string[] = [];
  const out = createFatalReports({ dir, fs, audit: (_a, d) => { rows.push(d); } }).collect();
  assert.equal(out.length, 1);
  assert.match(out[0].event, /heap out of memory/i);
  assert.ok((out[0].heapLimitMB ?? 0) > 0 && (out[0].rssMB ?? 0) > 0);
  assert.equal(rows[0].includes("leak-me-not"), false);
  assert.deepEqual(fs.readdirSync(dir), []);
});

test("wiring: boot collects before it enables, both before the exit recorder, and health shows the block without gating", () => {
  const bot = fs.readFileSync(path.join(HERE, "bot.ts"), "utf8");
  assert.match(bot, /fatalReports\.collect\(\);\s*fatalReports\.enable\(process\.report as unknown as ReportApi \| undefined\);\s*exitRecorder\.begin\(\);/);
  assert.match(bot, /createFatalReports\(\{ dir: fatalReportDir\(process\.env\), fs, audit:/);
  assert.match(bot, /checks\.checks\.fatal_reports = fatalReports\.status\(\);/);
  assert.doesNotMatch(fs.readFileSync(path.join(HERE, "healthGate.ts"), "utf8"), /fatal_reports/);
});
