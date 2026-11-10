// lazyPrices — weekly job + read-only status for lazy_prices_run.py (gate 2 of
// the Lazy Prices root). Run: npx tsx --test server/lazyPrices.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  lazyPricesFlag, lazyPricesCmd, lazyPricesPairsDir, inLazyPricesWindow, decideLazyPrices, parseLazyPricesSummary, runLazyPrices, lazyPricesStatus,
  LAZY_PRICES_TIMEOUT_MS, LAZY_PRICES_STALE_DAYS, LAZY_PRICES_MIN_FREE_BYTES,
} from "./lazyPrices";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// 2026-11-01 is a Sunday; 21:30Z = 16:30 ET (EST after the Nov 1 fall-back at 06:00Z)
const SUN_IN = Date.parse("2026-11-01T21:30:00Z");
const DAY = 86_400_000;
const base = { enabled: true, panelBuildId: "b20261029", latest: null, lastAttemptWeek: "", codeVersion: "1.0.1", freeBytes: 5e9 as number | null };

test("flag: env present wins; registry enabledWhen agrees", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "lazy_prices_weekly");
  assert.ok(job && job.kind === "interval" && job.source === "server/lazyPrices.ts");
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_LAZY_PRICES: v };
    assert.equal(lazyPricesFlag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window is Sunday 16:00-18:00 ET only, disjoint from edgeP2 (10-16) and the 20:15 panel build", () => {
  assert.equal(inLazyPricesWindow(SUN_IN), true);
  assert.equal(inLazyPricesWindow(Date.parse("2026-11-01T20:30:00Z")), false);   // 15:30 ET (edgeP2's window)
  assert.equal(inLazyPricesWindow(Date.parse("2026-11-01T23:30:00Z")), false);   // 18:30 ET
  assert.equal(inLazyPricesWindow(Date.parse("2026-10-31T21:30:00Z")), false);   // Saturday
});

test("decide: idle outside/disabled/already-attempted; skip without panel or low disk; run when stale or changed", () => {
  assert.equal(decideLazyPrices(SUN_IN, { ...base, enabled: false }).action, "idle");
  assert.equal(decideLazyPrices(Date.parse("2026-10-31T21:30:00Z"), base).action, "idle");
  const run = decideLazyPrices(SUN_IN, base);
  assert.equal(run.action, "run");
  assert.equal(decideLazyPrices(SUN_IN, { ...base, lastAttemptWeek: (run as { week: string }).week }).action, "idle");
  assert.equal(decideLazyPrices(SUN_IN, { ...base, panelBuildId: null }).action, "skip");
  assert.equal(decideLazyPrices(SUN_IN, { ...base, freeBytes: LAZY_PRICES_MIN_FREE_BYTES - 1 }).action, "skip");
  assert.equal(decideLazyPrices(SUN_IN, { ...base, freeBytes: 1_188_000_000 }).action, "run", "typical live free space must not block");
  assert.equal(decideLazyPrices(SUN_IN, { ...base, freeBytes: null }).action, "run");
  const fresh = { build_id: "b20261029", code_version: "1.0.1", generated_at: new Date(SUN_IN - 3 * DAY).toISOString() };
  assert.equal(decideLazyPrices(SUN_IN, { ...base, latest: fresh }).action, "skip");
  assert.equal(decideLazyPrices(SUN_IN, { ...base, latest: { ...fresh, build_id: "b20261001" } }).action, "run");
  assert.equal(decideLazyPrices(SUN_IN, { ...base, latest: { ...fresh, code_version: "1.0.0" } }).action, "run");
  const stale = { ...fresh, generated_at: new Date(SUN_IN - (LAZY_PRICES_STALE_DAYS + 1) * DAY).toISOString() };
  assert.equal(decideLazyPrices(SUN_IN, { ...base, latest: stale }).action, "run");
});

test("run: own long-job lane — budget, SIGKILL group, thread caps, parsed summary, version never interpolated unchecked", async () => {
  const calls: Array<{ cmd: string; opts: Record<string, unknown> }> = [];
  const ok: WalkforwardExec = async (cmd, opts) => {
    calls.push({ cmd, opts: opts as Record<string, unknown> });
    return { stdout: 'noise\n{"status":"ok","verdict":"INSUFFICIENT","build_id":"b20261029","n_obs":812,"elapsed_s":400.1,"peak_rss_mb":900}\n', stderr: "" };
  };
  const r = await runLazyPrices({ PATH: "/usr/bin" }, ok);
  assert.equal(r.status, "ok");
  assert.match(r.detail, /verdict INSUFFICIENT/);
  assert.match(calls[0].cmd, /^python3 lazy_prices_run\.py --run --nice 19 --max-rss-mb \d+ --code-version \S+ --pairs-dir \S+$/);
  assert.equal(calls[0].opts.timeout, LAZY_PRICES_TIMEOUT_MS);
  assert.equal(calls[0].opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((calls[0].opts.env as Record<string, string>)[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { killed: true }); };
  assert.match((await runLazyPrices({}, killed)).detail, /budget — process group killed/);
  assert.equal(parseLazyPricesSummary('{"status":"skipped","reason":"no filings archive yet"}').status, "skipped");
  assert.equal(parseLazyPricesSummary("Traceback").status, "error");
  assert.equal(lazyPricesCmd("1.0.9; rm -rf /"), lazyPricesCmd("unknown"));
});

test("status: headline fields only, survives sanitizeDiag; empty dir => nulls", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lp-"));
  try {
    assert.equal(lazyPricesStatus({ DATA_DIR: tmp }, path.join(tmp, "lazy_prices"), path.join(tmp, "pp")).latest, null);
    fs.mkdirSync(path.join(tmp, "lazy_prices"));
    fs.writeFileSync(path.join(tmp, "lazy_prices", "latest.json"), JSON.stringify({
      schema: "lazy_prices_gate2/1", generated_at: "2026-11-15T21:00:00Z", code_version: "1.0.1", panel: { build_id: "b20261029" },
      verdict: { result: "INSUFFICIENT", why: "few cohorts" }, variants: 1, n_obs: 812, secret_extra: "x",
    }));
    const st = lazyPricesStatus({ DATA_DIR: tmp }, path.join(tmp, "lazy_prices"), path.join(tmp, "pp"));
    const latest = st.latest as Record<string, any>;
    assert.equal(latest.verdict.result, "INSUFFICIENT");
    assert.equal(latest.n_obs, 812);
    assert.ok(!JSON.stringify(st).includes("secret_extra"));
    assert.deepEqual(sanitizeDiag(st), st);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("wiring: whitelisted probe, read-only case in bot.ts, booted from routes.ts, no mutex/daemon/order path, never ticks on boot", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("lazy_prices"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "lazy_prices": {');
  assert.ok(i > 0);
  const block = bot.slice(i, bot.indexOf("}", bot.indexOf("return res.json", i)) + 1);
  assert.ok(block.includes("sanitizeDiag(lazyPricesStatus())"));
  assert.ok(!/execPython|pythonRpc|beat\(|runLazyPrices/.test(block));
  assert.ok(fs.readFileSync(path.join(here, "routes.ts"), "utf8").includes("bootLazyPrices();"));
  const src = fs.readFileSync(path.join(here, "lazyPrices.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootLazyPrices"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot));
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc/.test(src));
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(src.includes('beat("lazy_prices_weekly", "start")') && src.includes('beat("lazy_prices_weekly", "finish"'));
});

test("pairs cache dir: container tmp by default (never the volume), safe override only, passed to the job", () => {
  const d = lazyPricesPairsDir({});
  assert.ok(!d.startsWith("/data"), d);
  assert.match(lazyPricesCmd("1.0.1", {}), new RegExp(`--pairs-dir ${d.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`));
  assert.equal(lazyPricesPairsDir({ VOLTRADE_LAZY_PRICES_PAIRS_DIR: "/scratch/p" }), "/scratch/p");
  assert.equal(lazyPricesPairsDir({ VOLTRADE_LAZY_PRICES_PAIRS_DIR: "/x; rm -rf /" }), d, "unsafe override ignored");
});
