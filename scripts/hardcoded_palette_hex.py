#!/usr/bin/env python3
"""
hardcoded_palette_hex.py — the hardcoded_palette_hex counter (D13, MASTER
PROGRAM §0.7 DETECT duty).

WHAT THIS CLOSES: PROGRAM_STATE.md's DETECTORS table has carried this exact
seed, unbuilt, since D11/D12 were seeded (2026-08-14): "theme-token literals
hardcoded instead of referenced ... the off-palette-hex half is not yet
built." `design_token_drift.py` (D12) only checks that DESIGN.md's table
agrees with `client/src/index.css`'s `:root` block — it says nothing about
whether a COMPONENT actually uses `var(--accent)` or instead restates
`"#4d9fff"` as a string literal. `dup_precise_literal` (D11) doesn't catch
this either: it targets high-precision numeric constants (>=7 significant
digits, e.g. WGS-84 radii), and a 6-hex-digit color has no digit-count
threshold in common with that class.

WHAT THIS COUNTS, PRECISELY: an occurrence, in `client/src/**/*.ts(x)`
(excluding `*.test.*`), of one of DESIGN.md's canonical hex token VALUES
(`--bg-primary #050a13`, `--accent #4d9fff`, etc. — the font tokens have no
hex value and are skipped), matched case-insensitively as a literal string.
Each occurrence is counted (not each file), matching D11's "count the
redundant copies, not the files" convention, so the counter falls exactly
one per fix. `index.css` itself is never scanned — it is the canonical
DEFINITION site DESIGN.md's table already keeps in sync via D12; restating a
value there is not drift, it's the source of truth.

WHY COMMENTS ARE BLANKED BUT STRING LITERALS ARE NOT (the inverse of
`ts_code_only.blank_noncode`, deliberately): a color hex value lives INSIDE a
string literal at its call site (`fill: "#4d9fff"`, `stroke: '#ff5a6e'`) —
blanking strings the way the empty-catch/`ts_any` counters do would blank out
every real match this detector exists to find. So this module blanks only
comment text (same line-leading `//`/`/*`/`*` convention as
`ts_code_only.COMMENT_LINE`, plus a string-aware scan for a `//` that starts
outside any string), leaving string and template-literal contents intact.

BASELINE IS NOT ZERO, same precedent as D3 `boundary_any` (233) and D4
`commented_empty_catch` (112): this counts EXISTING debt, seeded at whatever
the live tree has today, `non-increasing` from here — new hardcoded palette
hex is what the counter exists to block, not a demand that ~400 pre-existing
call sites get refactored in the same PR that adds the detector.
"""
from __future__ import annotations

import re
import subprocess
import sys

MD_PATH = "DESIGN.md"
CLIENT_SRC = "client/src"

_HEX_TOKEN_ROW = re.compile(r"^\|\s*`(--[a-z0-9-]+)`\s*\|\s*`(#[0-9a-fA-F]{3,8})`\s*\|", re.M)

_COMMENT_LEADING = re.compile(r"^\s*(//|/\*|\*)")


def palette_hex_values(md_path: str = MD_PATH) -> list[str]:
    """Hex-valued rows of DESIGN.md's canonical theme-tokens table (the
    two font tokens have no hex value and never match `_HEX_TOKEN_ROW`)."""
    try:
        with open(md_path) as f:
            src = f.read()
    except OSError:
        return []
    return sorted({val.lower() for _name, val in _HEX_TOKEN_ROW.findall(src)})


def _blank_comments_keep_strings(line: str) -> str:
    """Blank comment text only; string/template-literal contents survive
    (that is exactly where the hex literals this detector looks for live)."""
    if _COMMENT_LEADING.match(line):
        return " " * len(line)
    out = list(line)
    i, n = 0, len(line)
    in_string: str | None = None
    while i < n:
        c = line[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == in_string:
                in_string = None
            i += 1
            continue
        if c in "\"'`":
            in_string = c
            i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            for k in range(i, n):
                out[k] = " "
            break
        i += 1
    return "".join(out)


def _tracked_client_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", f"{CLIENT_SRC}/**/*.ts", f"{CLIENT_SRC}/**/*.tsx"],
        capture_output=True, text=True,
    ).stdout.split()
    return [f for f in out if "/node_modules/" not in f and ".test." not in f]


def find_hardcoded_hex(source: str, palette: list[str]) -> list[tuple[int, str]]:
    """Returns [(1-indexed line, matched hex)] for every palette-hex
    occurrence in `source` that survives comment-blanking."""
    if not palette:
        return []
    pattern = re.compile("|".join(re.escape(h) for h in palette), re.I)
    hits: list[tuple[int, str]] = []
    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        blanked = _blank_comments_keep_strings(raw_line)
        for m in pattern.finditer(blanked):
            hits.append((lineno, m.group(0).lower()))
    return hits


def compute(md_path: str = MD_PATH) -> dict:
    palette = palette_hex_values(md_path)
    total = 0
    by_file: dict[str, list[tuple[int, str]]] = {}
    for f in _tracked_client_files():
        try:
            with open(f, encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        hits = find_hardcoded_hex(src, palette)
        if hits:
            by_file[f] = hits
            total += len(hits)
    return {"count": total, "by_file": by_file}


if __name__ == "__main__":
    result = compute()
    if "--verbose" in sys.argv:
        for f, hits in sorted(result["by_file"].items()):
            for lineno, hexval in hits:
                print(f"{f}:{lineno}: {hexval}")
    print(result["count"])
