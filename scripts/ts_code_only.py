"""Tell CODE from PROSE in TypeScript sources, for the §4.2 counters.

WHY THIS EXISTS (MASTER PROGRAM Q13, MEASUREMENT INTEGRITY). `empty_ts_catch`
and `ts_any` were plain `grep` over raw file text, so a comment ABOUT the
pattern counted as the pattern. Fifteen of their matches were documentation:
495 -> 494 and 1251 -> 1237.

This is the sixth time in one session a source-scraping check has been
defeated by prose describing what it looks for (L9, L11, L12, L15, and twice
during #833 — one of which produced a false PASS). The standing rule that came
out of those is "strip comments FIRST and assert on the construct, not the
string", and this module is that rule compiled into code so the next counter
gets it for free instead of rediscovering it.

TWO DESIGN CONSTRAINTS, both learned by getting them wrong first:

1. CALLERS MUST EXCLUDE BY LOCATION, not by re-matching cleaned text — which
   is why `blank_noncode` returns a string of IDENTICAL LENGTH rather than a
   shorter one. Blanking comment text and re-running the regex sends
   `empty_ts_catch` UP, 495 -> 516: `catch { // why\n}` blanks to
   `catch {      \n}`, which now matches. Those 21 are precisely what D4
   `commented_empty_catch` counts, and the two counters are deliberately
   DISJOINT. Matching raw text and discarding matches whose bytes turn out to
   be non-code can only ever REMOVE, so the counters cannot merge.

2. SCAN PER LINE. The first implementation lexed whole files with a real
   TS-ish tokenizer. The regex literal `/'/g` at server/billing.ts:83 opened a
   quote that never closed, and the following 30 lines were declared non-code —
   silently excluding four REAL `catch (err: any)` annotations. Distinguishing
   a regex literal from division needs a parser; bounding the damage does not.
   A mis-parse here can corrupt exactly the line that causes it.

EVERY AMBIGUITY RESOLVES TOWARD COUNTING. Backticks are left alone (template
literals span lines and their `${}` interpolations are code). An unterminated
quote is left alone. A block comment is recognised only where it OPENS or
CONTINUES a line (`/*`, `*`), never mid-line — so `f(x: any) /* note: any */`
still counts twice. All fifteen real exclusions on the 2026-08-14 tree are
line-leading, so closing that gap would buy nothing and cost the guarantee.
These are non-increasing counters, so under-exclusion keeps them honestly
high, while over-exclusion makes the tree look better than it is — and
MEASUREMENT INTEGRITY treats a ruler change in the flattering direction as
suspect by default.
"""
import re
import subprocess
import sys

COMMENT_LINE = re.compile(r"^\s*(//|/\*|\*)")


def blank_noncode(line: str) -> str:
    """Replace comment and string-literal TEXT with spaces, preserving length.

    Length preservation is load-bearing: callers align offsets between the raw
    source and the blanked source to decide whether a match sits in code.
    """
    if COMMENT_LINE.match(line):
        return " " * len(line)
    out = list(line)
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            for k in range(i, n):
                out[k] = " "
            break
        if c in "\"'":
            j = i + 1
            while j < n:
                if line[j] == "\\":
                    j += 2
                    continue
                if line[j] == c:
                    break
                j += 1
            if j >= n:  # unterminated on this line: not a literal, keep it
                i += 1
                continue
            for k in range(i + 1, j):
                out[k] = " "
            i = j + 1
            continue
        i += 1
    return "".join(out)


def blank_source(src: str) -> str:
    """blank_noncode over a whole file. Same length as `src`, always."""
    return "\n".join(blank_noncode(l) for l in src.split("\n"))


def code_matches(pattern: "re.Pattern[str]", src: str) -> list:
    """Matches of `pattern` in `src` that are NOT comment or string text.

    Location exclusion: the pattern runs against the RAW source, and a match
    survives only if its bytes are unchanged by blanking. This can subtract
    matches and never add them (constraint 1 above).
    """
    blanked = blank_source(src)
    return [
        m
        for m in pattern.finditer(src)
        if blanked[m.start() : m.end()] == src[m.start() : m.end()]
    ]


def read_text(path: str):
    """Read a tracked file, or WARN and return None.

    Not `except OSError: continue`. A tracked file that cannot be read means
    the counter is measuring a smaller tree than it claims, and a non-increasing
    counter silently getting smaller is the flattering direction. `silent_py`
    counted this module's first draft and was right to: CLAUDE.md rates a
    pipeline that swallows errors worse than one that stops.
    """
    try:
        return open(path).read()
    except OSError as e:
        print(f"ts_code_only: SKIPPING unreadable tracked file {path}: {e}",
              file=sys.stderr)
        return None


def ts_files() -> list:
    """The tracked TS/TSX files the counters measure."""
    out = subprocess.run(
        ["git", "ls-files", "*.ts", "*.tsx"], capture_output=True, text=True
    ).stdout.split()
    return [f for f in out if "/node_modules/" not in f]


def count_in_tree(pattern: "re.Pattern[str]") -> int:
    total = 0
    for f in ts_files():
        src = read_text(f)
        if src is not None:
            total += len(code_matches(pattern, src))
    return total
