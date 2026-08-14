// CI ENV HONESTY (Q18, 2026-08-14). `ci.yml` set `VOLTRADE_CI: "1"` in two jobs
// beside the note "Network-dependent tests are excluded in CI; they run in the
// agent's session against live APIs instead."
//
// The CLAIM was true. The MECHANISM named was not: grep found zero readers of
// that variable anywhere in tracked code (D7). The exclusion is actually done
// by conftest.py's `collect_ignore`, which drops two standalone scripts that
// cannot execute under pytest at all.
//
// An env var that looks like a switch and is wired to nothing is worse than no
// switch — it makes the next reader believe a control exists. So the variable
// is gone and the comment now names conftest.py. These tests keep both true.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** Executable lines only. The comments explaining this fix NAME the variable on
 *  purpose, and a source scan cannot tell code from prose about code
 *  (PROGRAM_STATE.md L15 — six occurrences in one session before the rule
 *  became reflex). */
function executable(file: string): string {
  return fs.readFileSync(path.join(repoRoot, file), "utf8")
    .split("\n").filter((l) => !/^\s*#/.test(l)).join("\n");
}

test("no workflow sets an env var that nothing reads", () => {
  const ci = executable(".github/workflows/ci.yml");
  assert.ok(
    !/^\s*VOLTRADE_CI\s*:/m.test(ci),
    "VOLTRADE_CI is back in ci.yml. Nothing reads it — if a test now genuinely " +
    "consults it, wire that up first and this assertion can be revisited; " +
    "setting it alone only implies a control that does not exist.",
  );
});

test("the test-exclusion mechanism named in ci.yml actually exists", () => {
  // The comment is only worth keeping while the thing it points at is real.
  const conftest = fs.readFileSync(path.join(repoRoot, "conftest.py"), "utf8");
  assert.match(conftest, /collect_ignore\s*=/,
    "ci.yml credits conftest.py's collect_ignore with excluding the standalone " +
    "scripts. If that goes away, ci.yml's comment becomes false again.");
  for (const f of ["test_full_system.py", "test_auto_discovery.py"]) {
    assert.ok(conftest.includes(f), `${f} must stay in collect_ignore — it cannot run under pytest`);
    assert.ok(fs.existsSync(path.join(repoRoot, f)), `${f} is ignored but no longer exists`);
  }
});

test("the gate script does not set the dead variable either", () => {
  assert.ok(
    !/VOLTRADE_CI/.test(executable("scripts/gated_tests.sh")),
    "scripts/gated_tests.sh must not set VOLTRADE_CI — it was the assignment " +
    "that made D7 briefly report a false all-clear (it counted a SET as a READ).",
  );
});
