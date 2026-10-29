// walkforward — weekly job + read-only status for walkforward.py (the
// long-term point-in-time test of the stock score's price inputs).
// Run: npx tsx --test server/walkforward.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  walkforwardFlag, walkforwardCmd, isoWeekKey, inWalkforwardWindow, decideWalkforward, parseRunSummary, runWalkforward,
  walkforwardStatus, currentPanelBuild, latestSummary, WALKFORWARD_TIMEOUT_MS, WALKFORWARD_ENV_CAPS, WALKFORWARD_SUBDIR,
  WALKFORWARD_STALE_DAYS, type WalkforwardExec,
} from "./walkforward";
import { execGroupKill } from "./pythonExec";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { sanitizeDiag, DIAG_PROBES } from "./diag";
import { version as pkgVersion } from "../package.json";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");
// 2026-10-31 is a Saturday; 14:00Z = 10:00 ET (EDT)
const SAT_10_ET = Date.parse("2026-10-31T14:00:00Z");
const DAY = 86_400_000;

test("flag: env present wins (0/off/empty disable), absent = ON; registry enabledWhen agrees", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "walkforward_weekly");
  assert.ok(job && job.kind === "interval" && job.source === "server/walkforward.ts");
  assert.equal(job.kind === "interval" && job.cadenceMs, 7 * DAY);
  for (const [v, on] of [[undefined, true], ["1", true], ["on", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_WALKFORWARD: v };
    assert.equal(walkforwardFlag(env).on, on, `value ${v}`);
    assert.equal(job!.enabledWhen!(env), on, `registry disagrees on ${v}`);
  }
});

test("ISO week keys and the Sat/Sun 06:00-16:00 ET window", () => {
  assert.equal(isoWeekKey("2026-10-29"), "2026-W44");
  assert.equal(isoWeekKey("2021-01-03"), "2020-W53");
  assert.equal(isoWeekKey("2024-12-30"), "2025-W01");
  assert.equal(isoWeekKey("2026-10-31"), isoWeekKey("2026-11-01"), "Saturday and Sunday share a week");
  assert.equal(inWalkforwardWindow(SAT_10_ET), true);
  assert.equal(inWalkforwardWindow(Date.parse("2026-11-01T20:59:00Z")), true);        // Sun 15:59 EST (DST ended that morning)
  assert.equal(inWalkforwardWindow(Date.parse("2026-11-01T21:00:00Z")), false);       // Sun 16:00 EST
  assert.equal(inWalkforwardWindow(Date.parse("2026-10-31T09:59:00Z")), false);       // Sat 05:59 EDT
  assert.equal(inWalkforwardWindow(Date.parse("2026-10-31T10:00:00Z")), true);        // Sat 06:00 EDT
  assert.equal(inWalkforwardWindow(SAT_10_ET - DAY), false);                          // Friday
});

test("decide: idle outside the window / when off / once per week; skip without a panel or when fresh; run on new build, new code, stale", () => {
  const base = { enabled: true, panelBuildId: "b20261029", latest: null, lastAttemptWeek: "", codeVersion: "1.0.1" };
  assert.equal(decideWalkforward(SAT_10_ET, { ...base, enabled: false }).action, "idle");
  assert.equal(decideWalkforward(SAT_10_ET - DAY, base).action, "idle");
  assert.equal(decideWalkforward(SAT_10_ET, { ...base, lastAttemptWeek: "2026-W44" }).action, "idle");
  const noPanel = decideWalkforward(SAT_10_ET, { ...base, panelBuildId: null });
  assert.equal(noPanel.action, "skip");
  assert.match(noPanel.reason, /no complete price-panel build/);
  assert.equal(decideWalkforward(SAT_10_ET, base).action, "run");
  const fresh = { build_id: "b20261029", code_version: "1.0.1", generated_at: new Date(SAT_10_ET - 3 * DAY).toISOString() };
  assert.equal(decideWalkforward(SAT_10_ET, { ...base, latest: fresh }).action, "skip");
  assert.equal(decideWalkforward(SAT_10_ET, { ...base, latest: { ...fresh, build_id: "b20261001" } }).action, "run");
  assert.equal(decideWalkforward(SAT_10_ET, { ...base, latest: { ...fresh, code_version: "1.0.0" } }).action, "run");
  const stale = { ...fresh, generated_at: new Date(SAT_10_ET - (WALKFORWARD_STALE_DAYS + 1) * DAY).toISOString() };
  assert.equal(decideWalkforward(SAT_10_ET, { ...base, latest: stale }).action, "run");
});

test("run: own long-job lane — explicit budget, SIGKILL to the process group, thread caps, parsed summary", async () => {
  const calls: Array<{ cmd: string; opts: Record<string, unknown> }> = [];
  const ok: WalkforwardExec = async (cmd, opts) => {
    calls.push({ cmd, opts: opts as Record<string, unknown> });
    return { stdout: 'noise\n{"status":"ok","build_id":"b20261029","as_of":"2026-10-28","rows":10,"symbols":3,"elapsed_s":1.0,"peak_rss_mb":90}\n', stderr: "" };
  };
  const r = await runWalkforward({ PATH: "/usr/bin" }, ok);
  assert.equal(r.status, "ok");
  assert.match(r.detail, /b20261029/);
  assert.equal(calls[0].cmd, walkforwardCmd());
  assert.match(calls[0].cmd, /^python3 walkforward\.py --run --nice 19 --max-rss-mb \d+ --code-version /);
  assert.equal(calls[0].opts.timeout, WALKFORWARD_TIMEOUT_MS);
  assert.equal(calls[0].opts.killSignal, "SIGKILL");
  const env = calls[0].opts.env as Record<string, string>;
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal(env[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("Command failed"), { killed: true, signal: "SIGKILL" }); };
  assert.match((await runWalkforward({}, killed)).detail, /budget — process group killed/);
  const memErr: WalkforwardExec = async () => { throw Object.assign(new Error("Command failed"), { code: 1, stdout: '{"status":"error","reason":"memory budget: RSS 950 MB > budget 900 MB"}\n' }); };
  const m = await runWalkforward({}, memErr);
  assert.equal(m.status, "error");
  assert.match(m.detail, /memory budget/);
  assert.equal(parseRunSummary('{"status":"skipped","reason":"no complete price-panel build"}').status, "skipped");
  assert.equal(parseRunSummary("Traceback ...").status, "error");
  assert.equal(walkforwardCmd("1.0.9; rm -rf /"), walkforwardCmd("unknown"), "the version is never interpolated unchecked");
});

test("end to end through the real Python: synthetic panel -> latest.json -> status probe (no ticker leaks)", async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "wf-e2e-"));
  try {
    const panel = path.join(tmp, "price_panel");
    const out = path.join(tmp, WALKFORWARD_SUBDIR);
    execFileSync("python3", ["-c", `import walkforward_synth as ws; ws.synthetic_panel(${JSON.stringify(panel)}, n_names=14, signal=0.003, seed=3)`], { cwd: repo });
    const viaRepo: WalkforwardExec = (cmd, opts) => execGroupKill(`${cmd} --panel-root ${panel} --out-dir ${out}`, { ...opts, cwd: repo });
    const r = await runWalkforward(process.env, viaRepo);
    assert.equal(r.status, "ok", r.detail);
    assert.equal(currentPanelBuild(panel), "b20261029");
    const ls = latestSummary(out);
    assert.equal(ls?.build_id, "b20261029");
    assert.equal(ls?.code_version, pkgVersion);
    const st = walkforwardStatus({ DATA_DIR: tmp }, out, panel);
    const latest = st.latest as Record<string, unknown>;
    assert.equal(latest.schema, "walkforward/v1");
    const verdicts = latest.verdicts as Record<string, Record<string, unknown>>;
    assert.ok("momentum" in verdicts && "price_composite" in verdicts);
    assert.ok((latest.not_measurable as string[]).includes("vrp"));
    assert.equal((st.last_run as Record<string, unknown>).status, "ok");
    const flat = JSON.stringify(st);
    assert.ok(!/S0\d\d/.test(flat), "no ticker in the probe");
    assert.deepEqual(sanitizeDiag(st), st, "the probe payload survives sanitizeDiag unchanged (depth/size within limits)");
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("diag probe `walkforward` is whitelisted, wired, sanitized and read-only", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("walkforward"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "walkforward": {');
  assert.ok(i > 0);
  const block = bot.slice(i, bot.indexOf("}", bot.indexOf("return res.json", i)) + 1);
  assert.ok(block.includes("sanitizeDiag(walkforwardStatus())"));
  assert.ok(!/execPython|pythonRpc|beat\(|runWalkforward/.test(block), "the probe only reads");
  const st = walkforwardStatus({ DATA_DIR: fs.mkdtempSync(path.join(os.tmpdir(), "wf-empty-")) });
  assert.equal(st.latest, null);
  assert.equal(st.last_run, null);
});

test("wiring: booted from routes.ts, never runs on boot, never on the mutex or an order path; storage_config names the dir", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootWalkforward();"));
  const src = fs.readFileSync(path.join(here, "walkforward.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootWalkforward"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot), "a boot must not run a tick immediately");
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc/.test(src), "never the global Python mutex or the daemon");
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(src.includes('beat("walkforward_weekly", "start")') && src.includes('beat("walkforward_weekly", "finish"'));
  const sc = fs.readFileSync(path.join(repo, "storage_config.py"), "utf8");
  assert.ok(sc.includes(`WALKFORWARD_DIR = os.path.join(DATA_DIR, "${WALKFORWARD_SUBDIR}")`));
  const doc = fs.readFileSync(path.join(repo, "research", "walkforward.md"), "utf8");
  for (const k of ["latest.json", "brain_long_term", "not_measurable", "inputs", "walkforward/v1"]) assert.ok(doc.includes(k), `contract doc lacks ${k}`);
});
