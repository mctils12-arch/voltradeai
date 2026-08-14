// TSC-RATCHET (MASTER PROGRAM T1.6 / Q5, 2026-08-13). Tests for the gate that
// replaced `npx tsc --noEmit || true` in ci.yml's node-build job.
//
// The old line printed the typecheck into a log nobody opened for months, and
// two live user-facing bugs sat in that output (research/tsc_baseline.md §1).
//
// These tests exercise scripts/tsc_ratchet.sh's DECISION LOGIC against
// synthetic baselines rather than re-running the real typecheck — the real
// count is already asserted by tsc2304Ratchet.test.ts and by the script itself
// in CI. What needs pinning here is that the gate actually BLOCKS: a gate
// nobody has watched fail is not a gate, it is a decoration.
//
// The `|| true` this replaces is a standing reminder of that: it "ran" on every
// CI build for months and gated nothing.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCRIPT = path.join(repoRoot, "scripts", "tsc_ratchet.sh");
const BASELINE = path.join(repoRoot, "ci", "tsc_baseline.txt");

/** Run the ratchet against a baseline file, returning exit code + output. */
function runRatchet(baselineFile: string): { code: number; out: string } {
  try {
    const out = execFileSync("bash", [SCRIPT], {
      cwd: repoRoot,
      encoding: "utf8",
      env: { ...process.env, TSC_BASELINE_FILE: baselineFile },
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 300_000,
    });
    return { code: 0, out };
  } catch (err) {
    const e = err as { status?: number; stdout?: string; stderr?: string };
    return { code: e.status ?? -1, out: `${e.stdout ?? ""}${e.stderr ?? ""}` };
  }
}

function withTempBaseline(body: string, fn: (file: string) => void): void {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tsc-ratchet-"));
  const file = path.join(dir, "baseline.txt");
  fs.writeFileSync(file, body);
  try {
    fn(file);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

test("the committed baseline is well-formed and pins a TOTAL", () => {
  const raw = fs.readFileSync(BASELINE, "utf8");
  const m = raw.match(/^TOTAL (\d+)$/m);
  assert.ok(m, "ci/tsc_baseline.txt must contain a `TOTAL <n>` line");
  const total = Number(m![1]);

  // Not 0, and the reason is worth stating where someone will read it: three
  // of the twelve remaining errors are in server/billing.ts, a FROZEN PATH an
  // autonomous session may not edit. A zero-pin would either block this gate
  // forever or push a future session into a frozen path to satisfy it.
  assert.ok(total > 0, "pin of 0 is unreachable while 3 errors sit in FROZEN billing.ts");
  assert.ok(
    total <= 12,
    `the pin is downward-only (MASTER PROGRAM D4). Found TOTAL ${total}, ` +
    `above the 12 established by PRs #823/#824/#825. Raising a pin is stop ` +
    `condition 2 and needs a human.`,
  );
});

test("a pin below the real count FAILS the build (exit 1)", () => {
  // The load-bearing test. If this ever passes, the gate has stopped gating.
  withTempBaseline("TOTAL 0\n", (file) => {
    const { code, out } = runRatchet(file);
    assert.equal(code, 1, `expected exit 1 on regression, got ${code}.\n${out}`);
    assert.match(out, /FAIL: typecheck errors rose/, "must say the count rose");
    assert.match(out, /Per-code diff/, "must print the per-code diff");
    assert.match(
      out, /do NOT raise the pin/,
      "must warn against the suppression 'fix' — MASTER PROGRAM §12 names it",
    );
  });
});

test("a missing baseline FAILS rather than silently passing (exit 2)", () => {
  // A ratchet that stops ratcheting is the drift this whole program exists to
  // prevent, so 'cannot check' must never be reported as 'checked and fine'.
  const { code, out } = runRatchet(path.join(os.tmpdir(), "definitely-absent.txt"));
  assert.equal(code, 2, `expected exit 2 when the baseline is missing, got ${code}`);
  assert.match(out, /not found/, out);
});

test("a malformed baseline FAILS rather than defaulting to something permissive", () => {
  withTempBaseline("# no TOTAL line here\nTS2339 6\n", (file) => {
    const { code, out } = runRatchet(file);
    assert.equal(code, 2, `expected exit 2 on a malformed baseline, got ${code}`);
    assert.match(out, /no valid 'TOTAL/, out);
  });
});

test("ci.yml calls the ratchet and no longer swallows the typecheck", () => {
  const ci = fs.readFileSync(path.join(repoRoot, ".github", "workflows", "ci.yml"), "utf8");

  // Comment lines are stripped before the swallow check, because the intent is
  // "no CI STEP swallows the typecheck" and a comment cannot execute. This is
  // narrowing to the assertion's actual meaning, not relaxing it — the very
  // comment documenting the old line contains it verbatim, and matching that
  // is a false positive. (Second time this session prose moved one of my own
  // checks; see PROGRAM_STATE.md L9.)
  const executable = ci
    .split("\n")
    .filter((l) => !/^\s*#/.test(l))
    .join("\n");

  assert.match(
    executable, /run: bash scripts\/tsc_ratchet\.sh/,
    "ci.yml's node-build job must invoke the ratchet script",
  );
  assert.ok(
    !/tsc --noEmit \|\| true/.test(executable),
    "`npx tsc --noEmit || true` must not come back as an executed step — it is " +
    "the line that let 83 errors, including two live bugs, print unread for months.",
  );
});
