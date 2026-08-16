// EXEC-STDOUT-TYPING (T0.0/Q2, 2026-08-13). `execAsync` wraps
// `promisify(exec)` and is the single funnel every Python subprocess call in
// bot.ts goes through — 2 direct callers plus 44 via `execPythonSerialized`.
//
// Its `opts` parameter was `any`, which defeated overload resolution: TypeScript
// could not tell promisify(exec)'s string signature from its Buffer one, picked
// Buffer, and every `stdout.trim()` downstream became a type error. That was 42
// of the 83-error `tsc --noEmit` baseline — 51% of the entire list produced by
// one inference, which is why the list was too noisy for anyone to start on and
// why two genuine bugs (research/tsc_baseline.md §1) sat inside it unread.
//
// This pins the two properties that keep stdout a string. Both are source
// patterns rather than behavioural assertions because the thing being protected
// IS a compile-time inference: by the time code runs, the type is gone. The
// behavioural half is covered by tsc itself — `tsc_2304`'s sibling counter
// `tsc_errors` drops 78 -> 41 with this change and rises again without it.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

/** The `execAsync` definition, from its `const` to the end of that statement. */
function execAsyncDecl(): string {
  const start = bot.indexOf("const execAsync =");
  assert.ok(start > 0, "execAsync declaration not found in bot.ts");
  const end = bot.indexOf(";", bot.indexOf("_execRaw(", start));
  assert.ok(end > start, "could not find the end of the execAsync declaration");
  return bot.slice(start, end + 1);
}

test("execAsync's opts is typed ExecOptions, not any", () => {
  const decl = execAsyncDecl();
  assert.match(
    decl,
    /const execAsync = \(cmd: string, opts\?: ExecOptions\)/,
    "execAsync's `opts` must be typed `ExecOptions`.\n" +
    "Typing it `any` defeats overload resolution on promisify(exec): TypeScript " +
    "falls back to the Buffer signature and all 42 `stdout.trim()` call sites " +
    "become type errors again.\nFound:\n" + decl,
  );
  assert.ok(
    /import \{[^}]*\btype ExecOptions\b[^}]*\} from "child_process"/.test(bot),
    "ExecOptions must be imported from child_process for the annotation above.",
  );
});

test("execAsync pins encoding utf8, and pins it AFTER the opts spread", () => {
  const decl = execAsyncDecl();

  assert.match(
    decl,
    /encoding:\s*"utf8"/,
    "execAsync must pass `encoding: \"utf8\"` explicitly.\n" +
    "This is not a behaviour change — node's `exec` already defaults to utf8 — " +
    "it states the default so the string overload is unambiguous.\nFound:\n" + decl,
  );

  // Order matters and is the whole point: `encoding` after `...opts` means a
  // caller cannot flip the funnel into Buffer mode. Before `...opts` it would
  // be overridable, and one caller passing `encoding: 'buffer'` would silently
  // break 46 call sites' `.trim()` at runtime while typechecking clean.
  const spreadAt = decl.indexOf("...opts");
  const encodingAt = decl.indexOf('encoding: "utf8"');
  assert.ok(spreadAt > 0, `no \`...opts\` spread found in:\n${decl}`);
  assert.ok(
    encodingAt > spreadAt,
    "`encoding: \"utf8\"` must come AFTER the `...opts` spread so a caller " +
    "cannot override the funnel into Buffer mode.\nFound:\n" + decl,
  );
});

test("execPythonSerialized's opts is typed ExecOptions, not any", () => {
  assert.match(
    bot,
    /async function execPythonSerialized\(cmd: string, opts\?: ExecOptions\)/,
    "execPythonSerialized's `opts` must be `ExecOptions`. It forwards straight " +
    "into execAsync, so an `any` here re-introduces the same ambiguity one " +
    "level up — and this is the entry point 44 of the 46 call sites use.",
  );
});
