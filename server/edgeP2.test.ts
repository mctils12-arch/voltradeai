// edgeP2 — weekly job + read-only status for edge_p2_insider.py (gate 2 of
// pre-registration P2). Run: npx tsx --test server/edgeP2.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  edgeP2Flag, edgeP2Cmd, inEdgeP2Window, decideEdgeP2, parseEdgeP2Summary, runEdgeP2, edgeP2Status,
  EDGE_P2_TIMEOUT_MS, EDGE_P2_STALE_DAYS, EDGE_P2_MIN_FREE_BYTES,
} from "./edgeP2";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// 2026-11-01 is a Sunday; 15:00Z = 11:00 ET (EST after the Nov 1 fall-back at 06:00Z)
const SUN_11_ET = Date.parse("2026-11-01T16:00:00Z");
const DAY = 86_400_000;
const base = { enabled: true, panelBuildId: "b20261029", latest: null, lastAttemptWeek: "", codeVersion: "1.0.1", freeBytes: 5e9 as number | null };

test("flag: env present wins; registry enabledWhen agrees", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "edge_p2_weekly");
  assert.ok(job && job.kind === "interval" && job.source === "server/edgeP2.ts");
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_EDGE_P2: v };
    assert.equal(edgeP2Flag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window is Sunday 10:00-16:00 ET only (Saturday belongs to the walk-forward)", () => {
  assert.equal(inEdgeP2Window(SUN_11_ET), true);
  assert.equal(inEdgeP2Window(Date.parse("2026-11-01T14:30:00Z")), false);   // 09:30 ET
  assert.equal(inEdgeP2Window(Date.parse("2026-10-31T16:00:00Z")), false);   // Saturday
  assert.equal(inEdgeP2Window(Date.parse("2026-11-01T21:30:00Z")), false);   // 16:30 ET
});

test("decide: idle outside/disabled/already-attempted; skip without panel or on low disk; run when stale or changed", () => {
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, enabled: false }).action, "idle");
  assert.equal(decideEdgeP2(Date.parse("2026-10-31T16:00:00Z"), base).action, "idle");
  const run = decideEdgeP2(SUN_11_ET, base);
  assert.equal(run.action, "run");
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, lastAttemptWeek: (run as { week: string }).week }).action, "idle");
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, panelBuildId: null }).action, "skip");
  const low = decideEdgeP2(SUN_11_ET, { ...base, freeBytes: EDGE_P2_MIN_FREE_BYTES - 1 });
  assert.equal(low.action, "skip");
  assert.match(low.reason, /starve the panel build/);
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, freeBytes: null }).action, "run", "unknown free space does not block");
  const fresh = { build_id: "b20261029", code_version: "1.0.1", generated_at: new Date(SUN_11_ET - 3 * DAY).toISOString() };
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, latest: fresh }).action, "skip");
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, latest: { ...fresh, build_id: "b20261001" } }).action, "run");
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, latest: { ...fresh, code_version: "1.0.0" } }).action, "run");
  const stale = { ...fresh, generated_at: new Date(SUN_11_ET - (EDGE_P2_STALE_DAYS + 1) * DAY).toISOString() };
  assert.equal(decideEdgeP2(SUN_11_ET, { ...base, latest: stale }).action, "run");
});

test("run: own long-job lane — budget, SIGKILL group, thread caps, parsed summary, version never interpolated unchecked", async () => {
  const calls: Array<{ cmd: string; opts: Record<string, unknown> }> = [];
  const ok: WalkforwardExec = async (cmd, opts) => {
    calls.push({ cmd, opts: opts as Record<string, unknown> });
    return { stdout: 'noise\n{"status":"ok","verdict":"INCONCLUSIVE","build_id":"b20261029","events":812,"elapsed_s":400.1,"peak_rss_mb":900}\n', stderr: "" };
  };
  const r = await runEdgeP2({ PATH: "/usr/bin" }, ok);
  assert.equal(r.status, "ok");
  assert.match(r.detail, /verdict INCONCLUSIVE/);
  assert.match(calls[0].cmd, /^python3 edge_p2_insider\.py --run --nice 19 --max-rss-mb \d+ --code-version /);
  assert.equal(calls[0].opts.timeout, EDGE_P2_TIMEOUT_MS);
  assert.equal(calls[0].opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((calls[0].opts.env as Record<string, string>)[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { killed: true }); };
  assert.match((await runEdgeP2({}, killed)).detail, /budget — process group killed/);
  assert.equal(parseEdgeP2Summary('{"status":"insufficient","reason":"Form 4 history incomplete"}').status, "insufficient");
  assert.equal(parseEdgeP2Summary("Traceback").status, "error");
  assert.equal(edgeP2Cmd("1.0.9; rm -rf /"), edgeP2Cmd("unknown"));
});

test("status: headline cells only, no identifiers, survives sanitizeDiag; empty dir => nulls", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "p2-"));
  try {
    assert.equal(edgeP2Status({ DATA_DIR: tmp }, path.join(tmp, "edge_p2"), path.join(tmp, "pp")).latest, null);
    fs.mkdirSync(path.join(tmp, "edge_p2"));
    const cell = { months: 117, mean_bp: 41.2, t_nw3: 2.1, sd_bp: 400, mde_bp: 103, secret_extra: "x" };
    fs.writeFileSync(path.join(tmp, "edge_p2", "latest.json"), JSON.stringify({
      schema: "edge_p2/v1", generated_at: "2026-11-01T18:00:00Z", code_version: "1.0.1", panel: { build_id: "b20261029" }, verdict: "INCONCLUSIVE", why: "underpowered",
      results: { pit: { opportunistic_21d_matched: cell, routine_21d_matched: cell }, survivor_only: { opportunistic_21d_matched: cell } },
    }));
    const st = edgeP2Status({ DATA_DIR: tmp }, path.join(tmp, "edge_p2"), path.join(tmp, "pp"));
    const latest = st.latest as Record<string, any>;
    assert.equal(latest.verdict, "INCONCLUSIVE");
    assert.equal(latest.pit.opportunistic_21d.mean_bp, 41.2);
    assert.ok(!JSON.stringify(st).includes("secret_extra"));
    assert.deepEqual(sanitizeDiag(st), st);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("wiring: whitelisted probe, read-only case in bot.ts, booted from routes.ts, no mutex/daemon/order path, never ticks on boot", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("edge_p2"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "edge_p2": {');
  assert.ok(i > 0);
  const block = bot.slice(i, bot.indexOf("}", bot.indexOf("return res.json", i)) + 1);
  assert.ok(block.includes("sanitizeDiag(edgeP2Status())"));
  assert.ok(!/execPython|pythonRpc|beat\(|runEdgeP2/.test(block));
  assert.ok(fs.readFileSync(path.join(here, "routes.ts"), "utf8").includes("bootEdgeP2();"));
  const src = fs.readFileSync(path.join(here, "edgeP2.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootEdgeP2"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot));
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc/.test(src));
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(src.includes('beat("edge_p2_weekly", "start")') && src.includes('beat("edge_p2_weekly", "finish"'));
});
