#!/usr/bin/env python3
"""
param_shadows_outer_binding.py — the param_shadows_outer_binding counter
(D14, MASTER PROGRAM §0.7 DETECT duty).

WHAT THIS CLOSES: PROGRAM_STATE.md's DETECTORS table has carried this seed,
unbuilt, since it was added alongside D11/D12 (2026-08-14): "functions taking
a parameter that shadows an outer binding of the same name (the inverse of
D1 — would catch the `focusSat` extraction *before* the binding is lost)."
D1 (`tsc_2304`) catches an identifier USED but never DECLARED in scope — the
`focusSat` incident (research/tsc_baseline.md, server/tsc2304Ratchet.test.ts)
was TypeScript correctly flagging a free variable `e` left behind when a
handler body was extracted out of its enclosing closure. This detector
catches the shape TypeScript is silent on: a function PARAMETER that shares
a name with an outer binding compiles cleanly and simply shadows it for the
whole function body — so a later edit that assumes the body reads the OUTER
binding silently reads the shadowed inner one instead. This is exactly the
class ESLint's `no-shadow` rule exists for, and this repo has zero ESLint
config (verified: no `.eslintrc*` / `eslint.config.*` anywhere in the tree).

WHAT THIS COUNTS, PRECISELY (module scope, deliberately, like D5
`conflicting_const` and D11 `dup_precise_literal`): a `client/src/**/*.ts(x)`
file (excluding `*.test.*`) has a MODULE-TOP-LEVEL binding — `const NAME`,
`let NAME`, or `function NAME(...)`, all anchored at column 0 so an indented
(nested) declaration never counts as "outer" — and, ELSEWHERE in that SAME
file, a DIFFERENT module-top-level function or `const NAME = (...) => {...}`
arrow assignment declares a PARAMETER literally named `NAME`, where that
parameter is (a) at least MIN_NAME_LEN characters and (b) referenced at
least once inside that function's own body. Only top-level functions are
SCANNED for shadowing parameters (an indented, nested function is not
"module scope" and needs real closure-chain resolution to reason about
correctly, which — same discipline as D12's own docstring: "real scope
analysis would need an AST, not a grep" — this module deliberately does not
attempt). A parameter that merely happens to share a top-level name but is
never read in the body is excluded: an unused shadowing parameter is far
lower risk, and is the kind of thing an unused-parameter lint would already
catch if one were ever enabled here.

WHY MIN_NAME_LEN = 3. Live-checking this repo's actual top-level bindings
before shipping the rule (PROGRAM_STATE.md's own D13 precedent: sanity-check
the count before trusting it) turned up short, purely conventional top-level
physics/math constants — `PI`, `MU`, `J2`/`J3`/`J4`/`J8` (orbital
perturbation coefficients, client/src/lib/orbital/propagate.ts and
client/src/lib/celestial/rotation.ts), `S`, `CH`, `CW`, `km`, `v3` — that a
naive 1-2 character name match would flag against any unrelated same-named
parameter (a `km` or `v3` local in a completely different function), which
would be noise, not signal: exactly the "one-letter conventional names that
are virtually always intentional/harmless" case this session's brief warned
against. Filtering the SHADOWED name to >=3 characters removes all of them
without weakening the check on any name actually descriptive enough to be
worth protecting.

WHY THE BASELINE IS 0, NOT SEEDED DEBT (unlike D3/D4/D13): this is the first
counter in the DETECTORS table verified, by direct measurement of the live
tree with TWO independently-written scan passes (module-top-level-functions
only, and — as a second check — every function/arrow at any nesting depth
in the file), to have zero existing occurrences before MIN_NAME_LEN was even
applied. Same precedent as D12 `orphaned_set_interval` (baseline 0): the
counter's job from day one is to be a tripwire against a NEW shadow being
introduced, not to work down existing debt, and it was A/B-verified live via
an induced synthetic occurrence during this session (reverted; the unit
tests below pin the same synthetic case permanently).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ts_code_only import blank_source, read_text

CLIENT_SRC = "client/src"
MIN_NAME_LEN = 3

_TOP_CONST = re.compile(r"^(?:export\s+)?(?:const|let)\s+([A-Za-z_$][\w$]*)\b", re.M)
_TOP_FUNC_NAME = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)\s*\(",
    re.M,
)
_TOP_ARROW_START = re.compile(
    r"^(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*(?::[^=\n]+)?=\s*(?:async\s*)?\(",
    re.M,
)
_PARAM_NAME = re.compile(r"^\s*(?:\.\.\.)?([A-Za-z_$][\w$]*)\s*(?::|=|,|$)")


def top_level_binding_names(source: str) -> set[str]:
    """Names of module-scope `const`/`let`/`function` declarations, anchored
    at column 0 so an indented (nested) declaration is never counted."""
    return set(_TOP_CONST.findall(source)) | set(_TOP_FUNC_NAME.findall(source))


def _matching(source: str, open_idx: int, open_ch: str, close_ch: str) -> int | None:
    """Index of the bracket matching `source[open_idx]`, by raw depth count
    (same technique as D2 `long_try_empty_catch` above — no string/comment
    blanking, so a literal brace inside a string can misalign the count;
    accepted, existing precedent, and the function returns None rather than
    guessing when the file turns out unbalanced from `open_idx`)."""
    depth = 0
    for j in range(open_idx, len(source)):
        c = source[j]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j
    return None


def _split_top_level_commas(params_raw: str) -> list[str]:
    """Split a parameter list on commas that are not nested inside
    `()`/`[]`/`{}`/`<>` (destructuring defaults, generics, inline object/
    function types)."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for c in params_raw:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth <= 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur)
    return parts


def top_level_functions(source: str) -> list[dict]:
    """Every module-top-level named `function` declaration or
    `const NAME = (...) => { ... }` arrow assignment, each with its raw
    parameter-list text and full body text (braces included).

    Arrow functions with a single-expression body (no `{ ... }` block) are
    skipped: this detector needs a body to scan for parameter USAGE, and a
    bracket-less single expression is both harder to bound reliably with a
    regex and a much lower-risk shape to begin with — a one-line arrow has
    nowhere for a shadowed reference to hide.
    """
    starts: list[tuple[str, int]] = []
    for m in _TOP_FUNC_NAME.finditer(source):
        starts.append((m.group(1), m.end() - 1))
    for m in _TOP_ARROW_START.finditer(source):
        starts.append((m.group(1), m.end() - 1))

    out = []
    for name, popen in starts:
        pclose = _matching(source, popen, "(", ")")
        if pclose is None:
            continue
        params_raw = source[popen + 1:pclose]
        # Skip past an optional return-type annotation / arrow to the `{`
        # that opens the body. A window is enough here (return types are not
        # thousands of characters); no window match means no `{ ... }` body
        # (e.g. a single-expression arrow), which this detector skips.
        rest = source[pclose + 1: pclose + 400]
        bm = re.match(r"\s*(?::[^{=;]+)?(?:=>)?\s*\{", rest, re.S)
        if not bm:
            continue
        brace_open = pclose + 1 + bm.end() - 1
        brace_close = _matching(source, brace_open, "{", "}")
        if brace_close is None:
            continue
        out.append({
            "name": name,
            "params_raw": params_raw,
            "body": source[brace_open:brace_close + 1],
        })
    return out


def find_shadowing_params(source: str) -> list[dict]:
    """Every {"function": name, "param": pname} where a module-top-level
    function's parameter shadows another module-top-level binding of the
    same (>=MIN_NAME_LEN-character) name, and the parameter is referenced
    inside that function's own body."""
    top_names = {n for n in top_level_binding_names(source) if len(n) >= MIN_NAME_LEN}
    if not top_names:
        return []
    hits = []
    for fn in top_level_functions(source):
        body_code = blank_source(fn["body"])
        for frag in _split_top_level_commas(fn["params_raw"]):
            frag = frag.strip()
            if not frag or frag[:1] in "{[" or frag == "this":
                continue  # destructured/rest-object or `this` params: no single shadow name
            pm = _PARAM_NAME.match(frag)
            if not pm:
                continue
            pname = pm.group(1)
            if pname not in top_names:
                continue
            if re.search(r"\b" + re.escape(pname) + r"\b", body_code):
                hits.append({"function": fn["name"], "param": pname})
    return hits


def _tracked_client_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", f"{CLIENT_SRC}/**/*.ts", f"{CLIENT_SRC}/**/*.tsx"],
        capture_output=True, text=True,
    ).stdout.split()
    return [f for f in out if "/node_modules/" not in f and ".test." not in f]


def compute() -> dict:
    total = 0
    by_file: dict[str, list[dict]] = {}
    for f in _tracked_client_files():
        src = read_text(f)
        if src is None:
            continue
        hits = find_shadowing_params(src)
        if hits:
            by_file[f] = hits
            total += len(hits)
    return {"count": total, "by_file": by_file}


if __name__ == "__main__":
    result = compute()
    if "--verbose" in sys.argv:
        for f, hits in sorted(result["by_file"].items()):
            for h in hits:
                print(f"{f}: {h['function']}({h['param']})")
    print(result["count"])
