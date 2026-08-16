// THE TEST GATE (MASTER PROGRAM T1.2 / T1.3, 2026-08-14).
//
// scripts/gated_tests.sh is what makes ~3,586 tests able to block a merge.
// Before T1.1 CI ran four test files; T1.1 ran all 368 non-blocking to measure
// a baseline; this is the half that gives them teeth.
//
// These tests exercise the gate's DECISION LOGIC against synthetic quarantine
// files rather than running the real suites — the suites are already run by the
// gate itself in CI, and a test that takes three minutes to assert an exit code
// gets skipped by the next person in a hurry.
//
// What needs pinning is that the gate BLOCKS. A gate nobody has watched fail is
// a decoration, and this repo has already shipped one: `npx tsc --noEmit
// || true` ran on every build for months and gated nothing.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCRIPT = path.join(repoRoot, "scripts", "gated_tests.sh");
const QUARANTINE = path.join(repoRoot, "ci", "quarantine.txt");
const MAX_FILE = path.join(repoRoot, "ci", "quarantine_max.txt");

/** Run the gate with a synthetic quarantine/pin, returning exit code + output. */
function runGate(env: Record<string, string>): { code: number; out: string } {
  try {
    const out = execFileSync("bash", [SCRIPT], {
      cwd: repoRoot,
      encoding: "utf8",
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 120_000,
    });
    return { code: 0, out };
  } catch (err) {
    const e = err as { status?: number; stdout?: string; stderr?: string };
    return { code: e.status ?? -1, out: `${e.stdout ?? ""}${e.stderr ?? ""}` };
  }
}

function withFile(body: string, fn: (file: string) => void): void {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gate-"));
  const file = path.join(dir, "q.txt");
  fs.writeFileSync(file, body);
  try {
    fn(file);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const iso = (offsetDays: number): string => {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() + offsetDays);
  return d.toISOString().slice(0, 10);
};

test("the committed quarantine is within its pin and every entry is dated", () => {
  const entries = fs.readFileSync(QUARANTINE, "utf8")
    .split("\n")
    .filter((l) => l.trim() && !l.trimStart().startsWith("#"));
  const pin = Number(
    fs.readFileSync(MAX_FILE, "utf8").split("\n")
      .find((l) => l.trim() && !l.trimStart().startsWith("#"))?.trim(),
  );

  assert.ok(Number.isInteger(pin), "ci/quarantine_max.txt must hold a numeric pin");
  assert.ok(entries.length <= pin, `quarantine ${entries.length} exceeds pin ${pin}`);
  assert.ok(
    pin <= 1,
    `the pin is downward-only (MASTER PROGRAM D4). Found ${pin}; raising it is ` +
    `stop condition 2 and needs a human, because "quarantine the test I just ` +
    `broke" is the easiest way to turn a merge gate into decoration.`,
  );

  for (const line of entries) {
    assert.match(
      line, /review by \d{4}-\d{2}-\d{2}/,
      `every quarantine entry needs an expiry — a quarantine is a promise to ` +
      `come back, and the date is what makes it one:\n  ${line}`,
    );
    const due = new Date(line.match(/review by (\d{4}-\d{2}-\d{2})/)![1]);
    assert.ok(due.getTime() > Date.now(), `overdue quarantine entry:\n  ${line}`);
  }
});

test("a GROWN quarantine fails the build (T1.3) — you cannot park a test you just broke", () => {
  // The load-bearing rule. Without it the gate is trivially defeated: any red
  // test gets appended here and CI goes green having checked nothing.
  withFile(
    `a/one.test.ts  # x (added ${iso(0)}, review by ${iso(20)})\n` +
    `b/two.test.ts  # y (added ${iso(0)}, review by ${iso(20)})\n`,
    (file) => {
      const { code, out } = runGate({ QUARANTINE_FILE: file });
      assert.equal(code, 1, `expected exit 1 on a grown quarantine, got ${code}\n${out}`);
      assert.match(out, /quarantine grew 1 -> 2/);
      assert.match(out, /belongs FIXED, not parked/, "must say why, not just that");
    },
  );
});

test("an OVERDUE entry fails the build (T1.3)", () => {
  withFile(`a/one.test.ts  # x (added ${iso(-60)}, review by ${iso(-1)})\n`, (file) => {
    const { code, out } = runGate({ QUARANTINE_FILE: file });
    assert.equal(code, 1, `expected exit 1 on an overdue entry, got ${code}\n${out}`);
    assert.match(out, /review date .* passed/);
    assert.match(out, /do NOT simply push the date out/i);
  });
});

test("an UNDATED entry fails the build — no open-ended quarantines", () => {
  withFile("a/one.test.ts  # no expiry on this one\n", (file) => {
    const { code, out } = runGate({ QUARANTINE_FILE: file });
    assert.equal(code, 1, `expected exit 1 on an undated entry, got ${code}`);
    assert.match(out, /no 'review by/);
  });
});

test("a review date beyond the 30-day ceiling fails — you cannot date it into next year", () => {
  withFile(`a/one.test.ts  # x (added ${iso(0)}, review by ${iso(400)})\n`, (file) => {
    const { code, out } = runGate({ QUARANTINE_FILE: file });
    assert.equal(code, 1, `expected exit 1 on an over-long window, got ${code}`);
    assert.match(out, /more than 30 days out/);
  });
});

test("a missing quarantine or pin file fails rather than silently passing", () => {
  // "Cannot check" must never be reported as "checked and fine" — that is the
  // drift this whole program exists to prevent.
  const absent = path.join(os.tmpdir(), "definitely-absent-quarantine.txt");
  assert.equal(runGate({ QUARANTINE_FILE: absent }).code, 2);
  assert.equal(runGate({ QUARANTINE_MAX_FILE: absent }).code, 2);
});

test("ci.yml runs the gate, and the test job no longer tolerates failure", () => {
  const raw = fs.readFileSync(path.join(repoRoot, ".github", "workflows", "ci.yml"), "utf8");
  // Strip comments FIRST (PROGRAM_STATE.md L15): the comments explaining this
  // change quote `continue-on-error`, and a source scan cannot tell code from
  // prose about code.
  const code = raw.split("\n").filter((l) => !/^\s*#/.test(l)).join("\n");

  assert.match(code, /run: bash scripts\/gated_tests\.sh/, "ci.yml must invoke the gate");

  const testJob = code.slice(code.indexOf("\n  test:"), code.indexOf("\n  docker-build:"));
  assert.ok(
    !/^\s{4}continue-on-error:\s*true/m.test(testJob),
    "the `test` JOB must not carry continue-on-error — that is what made it " +
    "advisory under T1.1, and T1.2 is the step that gives it teeth",
  );
  assert.ok(
    /needs\.test\.result != 'failure'/.test(code),
    "auto-merge must block on the test job; being in `needs` alone only makes " +
    "it WAIT, which is exactly what T1.1 shipped deliberately",
  );
});
