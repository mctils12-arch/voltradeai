// earningsQoq — weekly job + read-only status for scripts/earnings_language_qoq_gate2.py
// (gate 2, paired-filing QoQ delta, of the 8-K earnings-language root).
// Run: npx tsx --test server/earningsQoq.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  earningsQoqFlag, earningsQoqCmd, inEarningsQoqWindow, decideEarningsQoq, parseEarningsQoqSummary, runEarningsQoq, earningsQoqStatus,
  EARNINGS_QOQ_TIMEOUT_MS, EARNINGS_QOQ_STALE_DAYS, EARNINGS_QOQ_MIN_FREE_BYTES, EARNINGS_QOQ_WINDOW_START_MIN, EARNINGS_QOQ_WINDOW_END_MIN,
} from "./earningsQoq";
import { inLazyPricesWindow } from "./lazyPrices";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// 2026-11-15 is a Sunday — the ladder's registered not_before date; 23:30Z = 18:30 ET (EST)
const SUN_IN = Date.parse("2026-11-15T23:30:00Z");
const DAY = 86_400_000;
const base = { enabled: true, latest: null, lastAttemptWeek: "", codeVersion: "1.0.1", freeBytes: 5e9 as number | null };

test("flag: env present wins; registry enabledWhen agrees", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "earnings_qoq_weekly");
  assert.ok(job && job.kind === "interval" && job.source === "server/earningsQoq.ts");
  assert.ok(job!.maxRunMs >= EARNINGS_QOQ_TIMEOUT_MS, "registry maxRunMs must cover the job budget");
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_EARNINGS_QOQ: v };
    assert.equal(earningsQoqFlag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window is Sunday 18:00-19:00 ET only, after lazyPrices (16-18) and done before the 20:15 panel build", () => {
  assert.equal(inEarningsQoqWindow(SUN_IN), true);
  assert.equal(inEarningsQoqWindow(Date.parse("2026-11-15T22:30:00Z")), false);   // 17:30 ET (lazyPrices' window)
  assert.equal(inEarningsQoqWindow(Date.parse("2026-11-16T00:30:00Z")), false);   // 19:30 ET
  assert.equal(inEarningsQoqWindow(Date.parse("2026-11-14T23:30:00Z")), false);   // Saturday
  assert.equal(inLazyPricesWindow(SUN_IN), false, "the two Sunday windows are disjoint");
  // the latest start plus the full budget ends before the 20:15 ET panel-build window
  assert.ok(EARNINGS_QOQ_WINDOW_END_MIN * 60_000 + EARNINGS_QOQ_TIMEOUT_MS <= (20 * 60 + 15) * 60_000);
  assert.ok(EARNINGS_QOQ_WINDOW_START_MIN >= 18 * 60);
  // the first run lands on the registered date: a Sunday tick on 2026-11-15 decides "run"
  assert.equal(decideEarningsQoq(SUN_IN, base).action, "run");
});

test("decide: idle outside/disabled/already-attempted; skip on low disk or a fresh same-version result; run when stale or code changed", () => {
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, enabled: false }).action, "idle");
  assert.equal(decideEarningsQoq(Date.parse("2026-11-14T23:30:00Z"), base).action, "idle");
  const run = decideEarningsQoq(SUN_IN, base);
  assert.equal(run.action, "run");
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, lastAttemptWeek: (run as { week: string }).week }).action, "idle");
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, freeBytes: EARNINGS_QOQ_MIN_FREE_BYTES - 1 }).action, "skip");
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, freeBytes: 1_000_000_000 }).action, "run", "typical live free space (~1 GB) must not block");
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, freeBytes: null }).action, "run");
  const fresh = { code_version: "1.0.1", generated_at: new Date(SUN_IN - 2 * DAY).toISOString() };
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, latest: fresh }).action, "skip");
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, latest: { ...fresh, code_version: "1.0.0" } }).action, "run");
  const weekOld = { ...fresh, generated_at: new Date(SUN_IN - (EARNINGS_QOQ_STALE_DAYS + 1) * DAY).toISOString() };
  assert.equal(decideEarningsQoq(SUN_IN, { ...base, latest: weekOld }).action, "run", "a weekly cadence re-runs last Sunday's result");
});

test("run: own long-job lane — budget, SIGKILL group, thread caps, parsed summary, version never interpolated unchecked", async () => {
  const calls: Array<{ cmd: string; opts: Record<string, unknown> }> = [];
  const ok: WalkforwardExec = async (cmd, opts) => {
    calls.push({ cmd, opts: opts as Record<string, unknown> });
    return { stdout: 'noise\n{"status":"ok","verdict":"FAIL","pairs":612,"n_obs":{"1":600,"3":590,"5":570},"elapsed_s":400.1,"peak_rss_mb":310}\n', stderr: "" };
  };
  const r = await runEarningsQoq({ PATH: "/usr/bin" }, ok);
  assert.equal(r.status, "ok");
  assert.match(r.detail, /verdict FAIL: 612 pairs/);
  assert.match(calls[0].cmd, /^python3 scripts\/earnings_language_qoq_gate2\.py --run --nice 19 --max-rss-mb \d+ --code-version \S+$/);
  assert.equal(calls[0].opts.timeout, EARNINGS_QOQ_TIMEOUT_MS);
  assert.equal(calls[0].opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((calls[0].opts.env as Record<string, string>)[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { killed: true }); };
  assert.match((await runEarningsQoq({}, killed)).detail, /budget — process group killed/);
  assert.equal(parseEarningsQoqSummary('{"status":"skipped","reason":"no earnings8k filings found"}').status, "skipped");
  assert.equal(parseEarningsQoqSummary('{"status":"error","reason":"MemoryError: x"}').status, "error");
  assert.equal(parseEarningsQoqSummary("Traceback").status, "error");
  assert.equal(earningsQoqCmd("1.0.9; rm -rf /"), earningsQoqCmd("unknown"));
});

test("status: headline fields only, survives sanitizeDiag, carries the registered date; empty dir => nulls", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "eq-"));
  try {
    const empty = earningsQoqStatus({ DATA_DIR: tmp }, path.join(tmp, "earnings_qoq"));
    assert.equal(empty.latest, null);
    assert.equal(empty.registered_not_before, "2026-11-15");
    fs.mkdirSync(path.join(tmp, "earnings_qoq"));
    fs.writeFileSync(path.join(tmp, "earnings_qoq", "latest.json"), JSON.stringify({
      schema: "earnings_qoq_gate2/1", generated_at: "2026-11-15T23:40:00Z", code_version: "1.0.1", source: "archive-dir",
      archive: { first_day: "2026-07-04", last_day: "2026-11-13" }, pairing: { pairs: 612, follow_on_dropped: 40 },
      features: { dtone: { "1": { n: 600, verdict: "FAIL" } } }, root_verdict: { result: "FAIL", primary: "dtone" }, variants: 2,
      first_sufficient_run: { generated_at: "2026-11-15T23:40:00Z", root_verdict: "FAIL" }, secret_extra: "x", samples: [{ ticker: "LEAK" }],
    }));
    const st = earningsQoqStatus({ DATA_DIR: tmp }, path.join(tmp, "earnings_qoq"));
    const latest = st.latest as Record<string, any>;
    assert.equal(latest.root_verdict.result, "FAIL");
    assert.equal(latest.variants, 2);
    assert.equal(latest.first_sufficient_run.root_verdict, "FAIL");
    assert.ok(!JSON.stringify(st).includes("secret_extra"));
    assert.ok(!JSON.stringify(st).includes("LEAK"), "only whitelisted blocks are served");
    assert.deepEqual(sanitizeDiag(st), st);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("wiring: whitelisted probe, read-only case in bot.ts, booted from routes.ts, no mutex/daemon/order path, never ticks on boot", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("earnings_qoq"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "earnings_qoq": {');
  assert.ok(i > 0);
  const block = bot.slice(i, bot.indexOf("}", bot.indexOf("return res.json", i)) + 1);
  assert.ok(block.includes("sanitizeDiag(earningsQoqStatus())"));
  assert.ok(!/execPython|pythonRpc|beat\(|runEarningsQoq/.test(block));
  assert.ok(fs.readFileSync(path.join(here, "routes.ts"), "utf8").includes("bootEarningsQoq();"));
  const src = fs.readFileSync(path.join(here, "earningsQoq.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootEarningsQoq"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot));
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc/.test(src));
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(src.includes('beat("earnings_qoq_weekly", "start")') && src.includes('beat("earnings_qoq_weekly", "finish"'));
  // the Python the job runs exists and pins the same registered constants the test above relies on
  const py = fs.readFileSync(path.join(here, "..", "scripts", "earnings_language_qoq_gate2.py"), "utf8");
  assert.ok(py.includes("MIN_GAP_DAYS = 50") && py.includes("MAX_GAP_DAYS = 130") && py.includes('PRIMARY = "dtone"'));
  assert.ok(py.includes("first_sufficient_run"), "the first sufficient look must be carried forward as the registered verdict");
});
