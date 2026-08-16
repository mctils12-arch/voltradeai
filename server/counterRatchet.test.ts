// THE COUNTER RATCHET (MASTER PROGRAM T1.7, 2026-08-14).
//
// scripts/program_status.sh has measured the §4.2 counters since #822 — but
// only when a human typed the command. Measured is not enforced, and the gap
// was not theoretical: `commented_empty_catch` drifted 112 -> 113 during the
// 2026-08-14 session from a concurrent session's merge, and nothing noticed
// until someone happened to run the script.
//
// scripts/counter_ratchet.sh closes that. These tests pin its decision logic
// against synthetic baselines — the real counters are already asserted by the
// script itself in CI, and what needs proving here is that it BLOCKS.
//
// Three gates in this repo now share one shape (rule in a mutable tested
// script, pin in a data file, one `run:` line in the FROZEN workflow):
// tsc_ratchet.sh, gated_tests.sh, and this.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCRIPT = path.join(repoRoot, "scripts", "counter_ratchet.sh");
const BASELINE = path.join(repoRoot, "ci", "counter_baseline.txt");

function runRatchet(baselineFile?: string): { code: number; out: string } {
  const env = { ...process.env } as Record<string, string>;
  if (baselineFile) env.COUNTER_BASELINE_FILE = baselineFile;
  try {
    const out = execFileSync("bash", [SCRIPT], {
      cwd: repoRoot, encoding: "utf8", env,
      stdio: ["ignore", "pipe", "pipe"], timeout: 300_000,
    });
    return { code: 0, out };
  } catch (err) {
    const e = err as { status?: number; stdout?: string; stderr?: string };
    return { code: e.status ?? -1, out: `${e.stdout ?? ""}${e.stderr ?? ""}` };
  }
}

function withBaseline(body: string, fn: (file: string) => void): void {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "counter-"));
  const file = path.join(dir, "b.txt");
  fs.writeFileSync(file, body);
  try { fn(file); } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

test("the committed baseline is well-formed and every counter has a direction", () => {
  const lines = fs.readFileSync(BASELINE, "utf8")
    .split("\n").map((l) => l.split("#")[0].trim()).filter(Boolean);
  assert.ok(lines.length >= 20, `expected the full counter set, found ${lines.length}`);
  for (const l of lines) {
    const parts = l.split(/\s+/);
    assert.equal(parts.length, 3, `malformed pin line: ${l}`);
    assert.match(parts[1], /^\d+$/, `non-numeric pin: ${l}`);
    assert.ok(
      ["non-increasing", "non-decreasing"].includes(parts[2]),
      `a pin without a direction cannot ratchet: ${l}`,
    );
  }
  // tsc is gated separately by scripts/tsc_ratchet.sh against
  // ci/tsc_baseline.txt. Two pins for one number is how they drift apart.
  assert.ok(!/^tsc_(errors|2304)\b/m.test(lines.join("\n")),
    "tsc counters belong to tsc_ratchet.sh, not here");
});

test("a REGRESSED non-increasing counter fails the build", () => {
  // The load-bearing case: the thing that would have caught the 112 -> 113 drift.
  withBaseline("empty_ts_catch 400 non-increasing\n", (file) => {
    const { code, out } = runRatchet(file);
    assert.equal(code, 1, `expected exit 1, got ${code}\n${out}`);
    assert.match(out, /empty_ts_catch: 400 -> \d+/);
    assert.match(out, /Fix the code, not the pin/);
    assert.match(out, /suppression/, "must name the failure mode, not just fail");
  });
});

test("a REGRESSED non-decreasing counter fails — assertions cannot quietly vanish", () => {
  // CLAUDE.md: "Never delete or weaken an existing assertion to make your
  // change pass." Now that tests block a merge, this is the cheapest wrong turn.
  withBaseline("assertions 99999 non-decreasing\n", (file) => {
    const { code, out } = runRatchet(file);
    assert.equal(code, 1, `expected exit 1, got ${code}`);
    assert.match(out, /assertions: 99999 -> \d+/);
  });
});

test("a pinned counter that vanished from the report fails rather than being skipped", () => {
  // Silently skipping an unreported pin is how a ratchet stops covering half
  // its surface without anyone noticing.
  withBaseline("ghost_counter 1 non-increasing\n", (file) => {
    const { code, out } = runRatchet(file);
    assert.equal(code, 2, `expected exit 2, got ${code}`);
    assert.match(out, /pinned but .* no longer reports it/);
  });
});

test("a missing or malformed baseline fails rather than silently passing", () => {
  const absent = path.join(os.tmpdir(), "definitely-absent-counters.txt");
  assert.equal(runRatchet(absent).code, 2);
  withBaseline("empty_ts_catch not_a_number non-increasing\n", (file) => {
    assert.equal(runRatchet(file).code, 2);
  });
});

test("an IMPROVED counter passes, and says which pin to lower", () => {
  // An unlowered pin lets the next change quietly give the gain back.
  withBaseline("uncapped_surface 9 non-increasing\n", (file) => {
    const { code, out } = runRatchet(file);
    assert.equal(code, 0, `an improvement must not fail the build\n${out}`);
    assert.match(out, /IMPROVED/);
    assert.match(out, /ci\/counter_baseline\.txt/, "must name the file to edit");
  });
});

test("ci.yml runs the counter ratchet in the blocking test job", () => {
  const raw = fs.readFileSync(path.join(repoRoot, ".github", "workflows", "ci.yml"), "utf8");
  // Strip comments FIRST (PROGRAM_STATE.md L15).
  const code = raw.split("\n").filter((l) => !/^\s*#/.test(l)).join("\n");
  assert.match(code, /run: bash scripts\/counter_ratchet\.sh/);
  const testJob = code.slice(code.indexOf("\n  test:"), code.indexOf("\n  docker-build:"));
  assert.ok(
    /counter_ratchet\.sh/.test(testJob),
    "the ratchet must live in the BLOCKING test job — in a continue-on-error " +
    "job it would report and gate nothing, which is what T1.1 shipped on purpose",
  );
});
