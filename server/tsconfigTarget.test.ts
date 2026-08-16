// TSCONFIG-TARGET (T0.0/Q3, 2026-08-13). `tsconfig.json` declared
// `"lib": ["esnext", "dom", "dom.iterable"]` but no `"target"`, so tsc fell
// back to its ES5 default and rejected `for…of` over a Map, Set or NodeList —
// iteration every runtime and every build path in this repo supports.
//
// That incoherence ("you may use every esnext API, but you may not iterate a
// Map") manufactured 29 errors: 23 TS2802 + 6 TS7006, verified by experiment to
// vanish the moment a target is set, with nothing new appearing. Together with
// Q2's 42 that was 71 of an 83-error baseline — 86% of the list that nobody
// could act on and that had two live bugs buried in it.
//
// WHY THIS NEEDS PINNING RATHER THAN JUST FIXING: the failure mode is an
// ABSENT key. Nothing looks wrong at the callsite, nothing looks wrong in the
// config, and the 29 errors reappear silently the moment someone tidies the
// file. An absent key has no natural place to leave a comment, so the test is
// the comment.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** tsconfig.json, tolerating the // comments TS config files permit. */
function tsconfig(): { compilerOptions?: Record<string, unknown> } {
  const raw = fs.readFileSync(path.join(repoRoot, "tsconfig.json"), "utf8");
  return JSON.parse(raw.replace(/^\s*\/\/.*$/gm, ""));
}

// Everything at or above ES2015 emits real iteration; below it, tsc rejects
// `for…of` over Map/Set/NodeList unless `downlevelIteration` is on.
const ITERATION_SAFE = new Set([
  "ES2015", "ES2016", "ES2017", "ES2018", "ES2019", "ES2020",
  "ES2021", "ES2022", "ES2023", "ES2024", "ESNEXT",
]);

test("tsconfig declares an explicit compilerOptions.target", () => {
  const target = tsconfig().compilerOptions?.target;
  assert.ok(
    typeof target === "string" && target.length > 0,
    "tsconfig.json must declare `compilerOptions.target`.\n" +
    "With it absent, tsc falls back to its ES5 default and rejects iteration " +
    "over Map/Set/NodeList — 29 phantom errors (23 TS2802 + 6 TS7006) against " +
    "a runtime that does not exist. `lib` is already `esnext`, so the ES5 " +
    "default is not conservative, it is incoherent.",
  );
});

test("the target is at or above ES2015, so iteration typechecks", () => {
  const target = String(tsconfig().compilerOptions?.target ?? "").toUpperCase();
  assert.ok(
    ITERATION_SAFE.has(target),
    `compilerOptions.target is "${target}", which puts tsc back below the ` +
    `iteration floor and re-manufactures the 29 TS2802/TS7006 errors.\n` +
    `Use one of: ${[...ITERATION_SAFE].join(", ")}.`,
  );
});

test("the target does not outrun what the real build targets can run", () => {
  // The tsconfig is noEmit, so this target never downlevels anything — it only
  // decides what tsc PERMITS. That makes it a promise about the two real build
  // paths, and the promise has to stay true:
  //   server → esbuild, no explicit target, running on node:20 (Dockerfile)
  //   client → vite 7, default `build.target: 'baseline-widely-available'`
  //            (Chrome/Edge 107, Firefox 104, Safari 16)
  // Safari 16 and node 20 both implement ES2022 in full, so ES2022 is the
  // honest ceiling today. It is NOT open-ended: if a future session sets
  // ESNext here, tsc would wave through syntax vite passes straight to a
  // browser that cannot parse it. Checked against vite's default rather than
  // assumed — under vite 4/5's older 'modules' default the ceiling was ES2020.
  const target = String(tsconfig().compilerOptions?.target ?? "").toUpperCase();
  const TOO_NEW = new Set(["ES2023", "ES2024", "ESNEXT"]);
  assert.ok(
    !TOO_NEW.has(target),
    `compilerOptions.target is "${target}", above what the real build paths ` +
    `guarantee. The tsconfig is noEmit, so nothing downlevels this — tsc would ` +
    `permit syntax vite hands straight to Safari 16.\n` +
    `Raise this only alongside vite's build.target and the runtime node version.`,
  );
});
