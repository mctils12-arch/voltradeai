#!/usr/bin/env node
// deploy_gate_smoke.mjs — Railway's deploy probe, run in CI before the merge.
//
// WHY (KNOWN BROKEN #41, 2026-09-10 -> 2026-09-16): production sat at
// Railway's 502 edge fallback for five days while 30 PRs merged, because
// every fresh container answered /api/health with 503 (the LIVENESS ALARM
// mapped to the HTTP code) and railway.json's healthcheck rejected it. Every
// test was green the whole time: nothing in CI ever booted the built bundle
// and asked it the one question Railway asks. This does.
//
// WHAT IT DOES: `npm run build` (what the Dockerfile ships), then boot
// `node dist/index.cjs` exactly as run_with_daemon.sh does (NODE_ENV=
// production, no broker credentials, no daemon), against the WORST persisted
// state the volume can hand a new container — kill switch latched ON and a
// liveness stamp days old (both restored on boot by design, both set to
// stay that way across deploys). Then poll /api/health the way Railway does
// and require HTTP 200 within railway.json's healthcheckTimeout (60s).
//
// A 503 here means: this commit, deployed, would be rejected by Railway and
// the previous container (or the 502 page) would keep serving. That is a
// merge-blocking finding regardless of what the payload says is wrong.
//
// SAFETY: refuses to run where /data is a MOUNT POINT (a real Railway
// volume) or RAILWAY_ENVIRONMENT is set. The fixtures go wherever bot.ts
// will READ them on this boot: /data/voltrade if that directory already
// exists (a root-run local boot creates it on its first save), otherwise
// the /tmp fallback paths. Pre-existing files are backed up and restored.
//
// Usage: node scripts/deploy_gate_smoke.mjs [--skip-build]
// Exit 0 = the container would pass the gate. Exit 1 = it would be rejected.
// Exit 2 = the smoke could not run (treated as failure by the gate script).
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HEALTHCHECK_TIMEOUT_S = 60; // railway.json (FROZEN) healthcheckTimeout
const BOOT_GRACE_S = 30;          // image pull/start isn't ours; only the app's own boot counts
const POLL_MS = 2000;

// The exact persisted-state shape bot.ts restores on boot (KILL_SWITCH_PATH /
// LIVENESS_PATH fallbacks). 2026-09-10T03:12:54Z is the real DD-HALT stamp.
const fixtureBodies = {
  "voltrade_kill_switch.json": { killSwitch: true, reason: "deploy_gate_smoke: DD-HALT latched", savedAt: "2026-09-10T03:12:54Z" },
  "voltrade_liveness.json": { lastActiveAt: Date.parse("2026-09-10T03:12:54Z"), savedAt: "2026-09-10T03:12:54Z" },
};

function log(msg) { console.log(`[deploy-gate-smoke] ${msg}`); }
function fail(code, msg) { console.error(`[deploy-gate-smoke] ${msg}`); process.exit(code); }

function dataIsMounted() {
  try { return fs.readFileSync("/proc/mounts", "utf8").split("\n").some((l) => l.split(" ")[1] === "/data"); }
  catch { return false; }
}
if (dataIsMounted() || process.env.RAILWAY_ENVIRONMENT) {
  fail(2, "refusing to run: /data is a mounted volume (or RAILWAY_ENVIRONMENT is set) — the smoke writes halt-state fixtures and must never touch real state");
}
// bot.ts reads KILL_SWITCH_PATH/LIVENESS_PATH under /data/voltrade FIRST and
// falls back to /tmp only when they are absent — so the fixtures must land
// where this boot will actually look.
const STATE_DIR = fs.existsSync("/data/voltrade") ? "/data/voltrade" : "/tmp";

const skipBuild = process.argv.includes("--skip-build");
if (!skipBuild) {
  log("npm run build (what the Dockerfile ships)");
  const b = spawnSync("npm", ["run", "build"], { cwd: repoRoot, stdio: "inherit", env: process.env });
  if (b.status !== 0) fail(2, `build failed (exit ${b.status}) — cannot smoke a bundle that does not exist`);
}
const bundle = path.join(repoRoot, "dist", "index.cjs");
if (!fs.existsSync(bundle)) fail(2, `${bundle} missing — run without --skip-build`);
if (!fs.existsSync(path.join(repoRoot, "dist", "public"))) fail(2, "dist/public missing — serveStatic() throws at boot without the client build");

const FIXTURES = Object.fromEntries(Object.entries(fixtureBodies).map(([f, body]) => [path.join(STATE_DIR, f), body]));
log(`halt-state fixtures -> ${STATE_DIR} (${Object.keys(fixtureBodies).join(", ")})`);

// back up + write fixtures
const backups = new Map();
for (const [p, body] of Object.entries(FIXTURES)) {
  if (fs.existsSync(p)) backups.set(p, fs.readFileSync(p));
  fs.writeFileSync(p, JSON.stringify(body));
}
function restoreFixtures() {
  for (const p of Object.keys(FIXTURES)) {
    try {
      if (backups.has(p)) fs.writeFileSync(p, backups.get(p));
      else fs.rmSync(p, { force: true });
    } catch {}
  }
}

const port = await new Promise((resolve, reject) => {
  const srv = net.createServer();
  srv.listen(0, "127.0.0.1", () => { const { port } = srv.address(); srv.close(() => resolve(port)); });
  srv.on("error", reject);
});

log(`booting node dist/index.cjs on :${port} with kill switch ON + stale liveness, no broker creds, no daemon`);
const env = { ...process.env, NODE_ENV: "production", PORT: String(port), VOLTRADE_DAEMON_ENABLED: "false" };
for (const k of ["ALPACA_KEY", "ALPACA_SECRET", "STRIPE_SECRET_KEY", "RESEND_KEY"]) delete env[k];
const child = spawn("node", ["--max-old-space-size=1024", bundle], {
  cwd: repoRoot, env, detached: true, stdio: ["ignore", "pipe", "pipe"],
});
let bootLog = "";
const keep = (chunk) => { bootLog += chunk.toString(); if (bootLog.length > 64_000) bootLog = bootLog.slice(-64_000); };
child.stdout.on("data", keep);
child.stderr.on("data", keep);
let exited = null;
child.on("exit", (code, signal) => { exited = { code, signal }; });

function shutdown() {
  try { process.kill(-child.pid, "SIGTERM"); } catch {}
  setTimeout(() => { try { process.kill(-child.pid, "SIGKILL"); } catch {} }, 3000).unref();
  restoreFixtures();
}
process.on("exit", restoreFixtures);
process.on("SIGINT", () => { shutdown(); process.exit(2); });
process.on("SIGTERM", () => { shutdown(); process.exit(2); });

async function probe() {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 10_000);
  try {
    const res = await fetch(`http://127.0.0.1:${port}/api/health`, { signal: ctrl.signal });
    const text = await res.text();
    let json = null;
    try { json = JSON.parse(text); } catch {}
    return { status: res.status, json, text };
  } catch (e) {
    return { status: 0, json: null, text: String(e?.message || e) };
  } finally {
    clearTimeout(t);
  }
}

const started = Date.now();
const deadline = started + (HEALTHCHECK_TIMEOUT_S + BOOT_GRACE_S) * 1000;
let last = null;
let firstAnswerAt = null;
while (Date.now() < deadline) {
  if (exited) {
    shutdown();
    console.error(bootLog.slice(-4000));
    fail(1, `the process EXITED during boot (code ${exited.code}, signal ${exited.signal}) — Railway would restart-loop this container`);
  }
  last = await probe();
  if (last.status !== 0 && firstAnswerAt === null) firstAnswerAt = Date.now();
  if (last.status === 200) break;
  // a 503 is Railway's rejection; keep polling in case a check is merely slow
  // to warm — but never past the deadline, and the deadline is the finding.
  await new Promise((r) => setTimeout(r, POLL_MS));
}

const elapsed = ((Date.now() - started) / 1000).toFixed(1);
shutdown();

if (last?.status === 200) {
  const gates = last.json?.serving?.gates?.join("+") ?? "?";
  const status = last.json?.status ?? "?";
  const dark = last.json?.checks?.bot?.liveness?.dark;
  log(`PASS: /api/health answered 200 in ${elapsed}s (status=${status}, serving gates=${gates}, liveness.dark=${dark})`);
  if (dark !== true) {
    // the fixture did not take: the test proved nothing about the case that
    // took the site down. Loud, not green.
    fail(2, "fixture did not apply — liveness.dark is not true, so the Sept-10 case was not exercised (did the fallback paths change?)");
  }
  process.exit(0);
}

console.error(bootLog.slice(-4000));
if (last?.status === 503) {
  const failing = last.json?.serving?.failing ?? "(no serving block — pre-healthGate handler?)";
  const degraded = Object.entries(last.json?.checks ?? {}).filter(([, v]) => v && v.status && v.status !== "ok").map(([k, v]) => `${k}=${v.status}`).join(", ");
  fail(1, `REJECTED: /api/health answered 503 after ${elapsed}s — Railway's healthcheck would fail this deploy and the OLD container (or the 502 page) would keep serving.\n  gating checks failing: ${JSON.stringify(failing)}\n  every non-ok check: ${degraded || "(none)"}\n  This is KNOWN BROKEN #41's exact failure mode; see server/healthGate.ts.`);
}
fail(1, `NO 200 within ${HEALTHCHECK_TIMEOUT_S}s (+${BOOT_GRACE_S}s boot grace): last answer status=${last?.status} body=${(last?.text || "").slice(0, 300)}; first answer at ${firstAnswerAt ? ((firstAnswerAt - started) / 1000).toFixed(1) + "s" : "never"}`);
