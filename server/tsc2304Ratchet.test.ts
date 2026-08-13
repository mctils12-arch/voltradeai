// D1 — the TS2304 RATCHET (MASTER PROGRAM §0.7 DETECT duty, 2026-08-13).
//
// TS2304 is "Cannot find name": an identifier that does not resolve. It is the
// one TypeScript error code that is NEVER config noise and never a false
// positive — a name is either in scope or it is not — so unlike the rest of the
// typecheck baseline it can be pinned at zero immediately and permanently.
//
// WHY IT EXISTS. Two live bugs of this exact class were found the first time
// anyone ran `npx tsc --noEmit` on this repo (research/tsc_baseline.md):
//
//   1. `altScale` — declared at datamap.tsx:4089 inside the flight-track paint
//      closure, used in the READOUT TICK, a different function. The
//      ReferenceError was swallowed by
//      `catch { /* readouts must never break the tick */ }`, which also skipped
//      every statement after it: GND SPD, VERT SPD, and the entire
//      follow-aircraft recenter block.
//   2. `e` in `focusSat(index)` — left behind when the satellite-focus body was
//      extracted out of `onClick(e)` for the SatFinder entrance. Satellite
//      clicks never stamped `__vtFeatClaim`, so the click-off handler took the
//      wrong branch.
//
// Both came from ordinary refactors — a closure split and a function extraction
// — and both were invisible because three defences composed: `|| true` on the
// CI typecheck hid the error, an empty catch block swallowed the throw, and a
// reassuring comment made the swallow look deliberate. One of them survived a
// full static audit of the codebase.
//
// A ratchet on the TOTAL error count would not have caught either: 86% of that
// count is two mechanical causes (an untyped `execPythonSerialized`, a
// `tsconfig.json` with no `target`), so the total is too noisy to pin. This
// test pins the always-real subset instead.
//
// ─── SCOPE HONESTY ────────────────────────────────────────────────────────
// This test is NOT yet enforced by CI. `ci.yml`'s `node-build` job — the only
// job a `.tsx` change triggers — runs `npm ci`, `npx tsc --noEmit || true`, and
// `npm run build`, and invokes no test suite at all. So this file runs today
// only under `npm run test:node` (or `npx tsx --test server/*.test.ts`).
//
// That is deliberate and it is the point of Q10/T1.1, which wires the suites
// into CI. Putting the ratchet in `test_audit_critical.py` instead would be
// WORSE, not better: CI's path filter routes `.py` changes to `python-tests`
// and `.tsx` changes to `node-build`, so a Python-side guard would never fire
// for the TypeScript files it guards (PROGRAM_STATE.md L7). This file is
// written to be already-correct when the gate turns on.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** Full-project typecheck. Returns tsc's stdout (it reports errors there, and
 *  exits non-zero whenever any exist — which is the normal case while the
 *  non-TS2304 baseline is still being worked down, so a throw is expected and
 *  its stdout is the real result). */
function typecheck(): string {
  try {
    return execFileSync("npx", ["tsc", "--noEmit"], {
      cwd: repoRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 300_000,
    });
  } catch (err) {
    // Distinguish "tsc ran and found errors" from "tsc could not run at all".
    // Without this, a missing node_modules would surface as a PASSING test —
    // the ratchet would silently stop ratcheting, which is the failure mode
    // this whole program exists to prevent.
    //
    // Narrowed off `unknown` rather than typed `any`: this file introduces the
    // `tsc_2304` ratchet, so it has no business pushing the `ts_any` one up.
    const e = err as { stdout?: string; stderr?: string } | null;
    const out = `${e?.stdout ?? ""}${e?.stderr ?? ""}`;
    assert.ok(
      out.includes("error TS") || out.trim() === "",
      `tsc did not run — cannot verify the TS2304 ratchet.\n` +
      `Run \`npm ci\` first. Raw output:\n${out.slice(0, 2000)}`,
    );
    return out;
  }
}

test("D1 ratchet: zero TS2304 'Cannot find name' errors repo-wide", () => {
  const offenders = typecheck()
    .split("\n")
    .filter((l) => l.includes("error TS2304"));

  assert.deepEqual(
    offenders,
    [],
    `${offenders.length} identifier(s) used outside their declaring scope.\n\n` +
    offenders.map((l) => `  ${l.trim()}`).join("\n") +
    `\n\nEach of these is a ReferenceError at runtime, not a type nitpick, and ` +
    `if it sits inside a try block it kills every statement after it in that ` +
    `block — not just its own line.\n` +
    `Fix the scope; do NOT silence this by declaring a placeholder or widening ` +
    `a type. See research/tsc_baseline.md §1.`,
  );
});
