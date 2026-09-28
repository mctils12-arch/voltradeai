#!/usr/bin/env python3
"""
duplicate_route_registration.py — the duplicate_route_registration counter
(D15, MASTER PROGRAM §0.7 DETECT duty).

WHAT THIS CLOSES: PROGRAM_STATE.md's DETECTORS table's "Seeds not yet taken"
list was down to two non-viable entries this session (the `useEffect`
ref-omission seed explicitly SKIPPED as low-value by a prior session's own
brief; the `layers.json` registry-id seed already marked "investigated,
correctly NOT built" — no consistent `/api/data/<id>` naming convention
exists to check against). Per D13/D14's own precedent ("a future session
owing the §0.7 duty needs a fresh ACTIVE-ANGLE-HUNTING pass to find the next
seed rather than pulling from this list"), this is a freshly-hunted seed,
not one carried over.

WHAT THIS COUNTS, PRECISELY: two (or more) `app.<method>("path", ...)`
registrations across every tracked, non-test `server/*.ts` file that share
the same HTTP method and the same PATH once each `:paramName` route-param
segment is normalized to a single canonical `:param` — `/api/foo/:id` and
`/api/foo/:ticker` are the SAME route SHAPE to Express and collide exactly
the way two literally identical path strings would.

WHY THIS IS A REAL BUG CLASS, NOT A STYLE NIT: Express resolves routes in
REGISTRATION ORDER — the first `app.get()`/`app.post()`/etc. call for a
given method+path wins every request, permanently. A second, later
registration for the identical method+path pattern is silently unreachable
dead code: no runtime error, no log line, nothing but a handler that can
never fire. Same "mechanism with no visible off-switch" shape as D7
(`dead_workflow_env`) and D12 (`orphaned_set_interval`).

WHY THIS SCANS EVERY `server/*.ts` FILE, NOT JUST `routes.ts`: every route-
registration module in this repo (`registerAdminStats`, `registerAuthRoutes`,
`registerBillingRoutes`, `registerNewsletterRoutes`, `registerRobots`,
`registerTerms`, plus `routes.ts`/`bot.ts` themselves) takes the SAME shared
top-level `app: Express` instance and calls `app.<method>` directly on it —
verified by reading every `registerX(app: Express)`-shaped export's
signature, not assumed. A duplicate across two of these files is therefore a
REAL cross-file collision, exactly as live as a duplicate inside one file.

WHY NOT `router.`-MOUNTED SUB-PREFIXES: `router.<method>(` does not occur
anywhere in this tree outside `*.test.ts` files (grepped this session). A
codebase using `app.use(prefix, subRouter)` would need the mount prefix
folded into the comparison to avoid false collisions between routers
mounted at different paths — out of scope here because that pattern does
not exist in this tree today. Documented, not silently assumed: a future
session introducing `router.`-based mounting must re-scope this detector
before trusting it.

WHY COMMENTS ARE BLANKED, STRINGS ARE NOT (D13 `hardcoded_palette_hex`'s own
precedent, reimplemented per-module rather than shared): the route path
lives INSIDE the string literal that is this detector's actual signal
(`app.get("/api/health", ...)`), so blanking strings the way
`ts_code_only.blank_source` does would blank out every real match. Only
comment text is blanked. Unlike D13 (per-line matching is enough for a hex
literal), the method call and its path string can span multiple lines here
(found live this session: `server/billing.ts`'s webhook route opens
`app.post(` on one line and states the path string on the next) — so
blanking is done per-line (to keep the line-leading `//`/`/*`/`*` comment
rule simple) but MATCHING runs over the whole blanked file joined back
together, not line-by-line, so a call spanning lines is still found.

WHY TEMPLATE-LITERAL PATHS AREN'T HANDLED: grepped this session — no
`app.<method>(\`...\`` occurrence exists anywhere in this tree today. If one
is ever added, this detector will not see it; a future session adding
template-literal route paths should extend `_ROUTE_CALL` rather than trust
this detector blindly at that point.

BASELINE IS 0, same precedent as D12/D14 (a tripwire against a NEW
collision, not existing debt to pay down): verified live, this session, two
independent widths — `server/routes.ts` alone (210 registrations, 0
collisions, both on raw path strings and after `:param` normalization) and
every tracked non-test `server/*.ts` file combined (265 registrations, 0
collisions, same two checks) — matching D14's own two-independent-passes
discipline before trusting a clean baseline.
"""
from __future__ import annotations

import re
import subprocess
import sys

_COMMENT_LEADING = re.compile(r"^\s*(//|/\*|\*)")
_ROUTE_CALL = re.compile(
    r"\bapp\.(get|post|put|delete|patch)\(\s*[\"']([^\"']+)[\"']"
)
_PARAM_SEGMENT = re.compile(r"/:[A-Za-z0-9_]+")


def _blank_comments_keep_strings(line: str) -> str:
    """Blank comment text only; string literal contents (where the route
    path itself lives) survive unchanged, same length as `line`."""
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


def _blank_comments(source: str) -> str:
    """`_blank_comments_keep_strings` over a whole file, same length as
    `source` (offsets stay aligned so a match's line number can be
    recovered by counting newlines up to its start)."""
    return "\n".join(_blank_comments_keep_strings(l) for l in source.split("\n"))


def normalize_path(path: str) -> str:
    """Collapse every `:paramName` route-param segment to a single
    canonical `:param` — two routes differing only in param NAME are the
    same route SHAPE to Express and collide identically."""
    return _PARAM_SEGMENT.sub("/:param", path)


def extract_route_registrations(source: str) -> list[tuple[str, str, int]]:
    """[(method, normalized_path, 1-indexed line)] for every
    `app.<method>("path", ...)` call that survives comment-blanking. Matches
    over the WHOLE blanked file (not line-by-line) so a call whose method
    and path string span multiple lines is still found."""
    blanked = _blank_comments(source)
    out = []
    for m in _ROUTE_CALL.finditer(blanked):
        method, path = m.group(1), m.group(2)
        lineno = blanked.count("\n", 0, m.start()) + 1
        out.append((method, normalize_path(path), lineno))
    return out


def _tracked_server_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "server/*.ts"], capture_output=True, text=True,
    ).stdout.split()
    return [f for f in out if ".test." not in f]


def find_duplicate_routes(
    routes: list[tuple[str, str, str, int]]
) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """`routes`: [(method, normalized_path, file, line)]. Returns
    {(method, normalized_path): [(file, line), ...]} for every method+path
    registered more than once, across any combination of files."""
    by_key: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for method, path, f, lineno in routes:
        by_key.setdefault((method, path), []).append((f, lineno))
    return {k: v for k, v in by_key.items() if len(v) > 1}


def compute() -> dict:
    all_routes: list[tuple[str, str, str, int]] = []
    for f in _tracked_server_files():
        try:
            with open(f) as fh:
                src = fh.read()
        except OSError as e:
            print(
                f"duplicate_route_registration: SKIPPING unreadable tracked "
                f"file {f}: {e}",
                file=sys.stderr,
            )
            continue
        for method, path, lineno in extract_route_registrations(src):
            all_routes.append((method, path, f, lineno))
    dups = find_duplicate_routes(all_routes)
    # Count REDUNDANT registrations (extra copies beyond the first one that
    # actually wins), same "count the copies, not the groups" convention as
    # D5/D11: a route registered 3 times is 2 redundant, unreachable copies,
    # not 1 "finding".
    total = sum(len(v) - 1 for v in dups.values())
    return {"count": total, "duplicates": dups}


if __name__ == "__main__":
    result = compute()
    if "--verbose" in sys.argv:
        for (method, path), sites in sorted(result["duplicates"].items()):
            print(f"{method.upper()} {path}:")
            for f, lineno in sites:
                print(f"  {f}:{lineno}")
    print(result["count"])
