#!/usr/bin/env python3
"""
design_token_drift.py — the design_token_drift counter's actual logic
(MASTER PROGRAM T8.1 / D12), extracted out of scripts/program_status.sh's
inline heredoc so it can be imported and unit-tested directly (the
gate2_stats.py / test_law_iv_context_modules.py precedent: a detector that
only ever ran embedded in a bash script had zero coverage of its own
behavior — it was exercised, never verified).

WHAT THIS CHECKS, PRECISELY (and what it does not): DESIGN.md's "Canonical
theme tokens" table documents a curated, small, semantic subset of
client/src/index.css's `:root` block (`--bg-primary`, `--text-primary`,
`--accent`, ... — the names DESIGN.md tells page authors to use, per its
own rule 5: "Theme tokens only ... canonical values below"). index.css
additionally defines a much larger set of shadcn/ui-style implementation
tokens (`--background`, `--card`, `--popover`, HSL triplets, radii, ...)
that DESIGN.md's table was never meant to enumerate one-for-one — they are
internal plumbing, not the author-facing palette.

So drift is checked ONE DIRECTION ONLY: every token DESIGN.md documents
must exist in index.css with the exact value DESIGN.md claims for it. A
CSS-only token with no DESIGN.md entry is NOT drift — it is normal, and
flagging it would fire on ~70 healthy shadcn tokens with no actionable fix
(this repo's own real index.css/DESIGN.md pair was checked directly: 87
css `:root` custom properties exist, 15 are in DESIGN.md's table, and the
other 72 are exactly this legitimate internal-plumbing case, none of them
a canonical author-facing token DESIGN.md forgot to list).

CORRECTION (found writing this test, D12 corrected in the same PR): the
inline version's own comment claimed "a mismatch in either direction is
drift", which was never true of the code beneath it — the loop only ever
walked `md.items()`. Fixed here to state the real, intentional semantics
instead of an aspirational one nobody had checked against the actual
token tables. The counter's VALUE is unchanged (0 before this PR, 0
after, on the identical live DESIGN.md/index.css pair) — this is a
comment/coverage fix, not a behavior change, so ci/counter_baseline.txt's
pin does not move.
"""
from __future__ import annotations

import re

CSS_PATH = "client/src/index.css"
MD_PATH = "DESIGN.md"


def css_tokens(path: str) -> dict[str, str]:
    """Custom properties defined in the `:root` block only — later scoped
    overrides (e.g. a `.dark` or component-local block) are legitimately
    different values and not part of this comparison."""
    out: dict[str, str] = {}
    try:
        with open(path) as f:
            src = f.read()
    except OSError:
        return out
    m = re.search(r":root\s*\{(.*?)\}", src, re.S)
    if not m:
        return out
    for name, val in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", m.group(1)):
        out[name] = re.sub(r"\s+", " ", val).strip()
    return out


def md_tokens(path: str) -> dict[str, str]:
    """Rows of DESIGN.md's `| \`--token\` | \`value\` | ... |` table."""
    out: dict[str, str] = {}
    try:
        with open(path) as f:
            src = f.read()
    except OSError:
        return out
    for name, val in re.findall(r"^\|\s*`(--[a-z0-9-]+)`\s*\|\s*`([^`]+)`\s*\|", src, re.M):
        out[name] = re.sub(r"\s+", " ", val).strip()
    return out


def compute_drift(css_path: str = CSS_PATH, md_path: str = MD_PATH) -> dict:
    """Returns {"drift": int, "missing": [...], "mismatched": [(name, doc_val, css_val), ...]}.

    `missing`: documented in DESIGN.md but undefined in index.css's :root.
    `mismatched`: documented with a value that disagrees with index.css.
    A CSS-only token (defined but undocumented) is deliberately absent from
    both lists — see the module docstring.
    """
    css = css_tokens(css_path)
    md = md_tokens(md_path)
    missing, mismatched = [], []
    for name, val in md.items():
        if name not in css:
            missing.append(name)
        elif css[name] != val:
            mismatched.append((name, val, css[name]))
    return {
        "drift": len(missing) + len(mismatched),
        "missing": sorted(missing),
        "mismatched": sorted(mismatched),
    }


if __name__ == "__main__":
    print(compute_drift()["drift"])
