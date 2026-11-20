#!/usr/bin/env python3
"""UI coverage register: which backend outputs reach the user interface.

WHY (owner directive 2026-11-19, execution-layer Addendum 1 A8/A9; CLAUDE.md
STANDING BEHAVIORS, UI INTEGRATION AND ONE ORDER PATH): every backend feature
is to be integrated into the existing UI where it is relevant to the user, and
UI_COVERAGE.md is the register, kept current by every PR that adds or changes
a backend feature. A register kept by hand goes stale, so this script derives
what it can from the code and takes the rest from one notes file:

  derived   every HTTP route the server registers (server/*.ts), and whether
            client/src references its path; every keyed-API route's presence
            in the catalog the /developers page renders; the map layers in
            datacore/layers.json (the layers panel is drawn from that file);
            the diagnostic probes behind /api/diag/:probe.
  by hand   ci/ui_coverage_notes.json: a status and a note for each item the
            client does not reference (surfaced | partial | missing |
            plumbing | legacy).

Usage:
  python3 scripts/ui_coverage.py            print the summary
  python3 scripts/ui_coverage.py --write    rewrite UI_COVERAGE.md
  python3 scripts/ui_coverage.py --check    exit 1 if UI_COVERAGE.md is stale
                                            or any item is unclassified

A route "referenced by the client" means its path (up to the first parameter)
appears in a non-test file under client/src. That proves a page calls it, not
that the page shows everything it returns.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(ROOT, "ci", "ui_coverage_notes.json")
OUT = os.path.join(ROOT, "UI_COVERAGE.md")
STATUSES = ("surfaced", "partial", "missing", "plumbing", "legacy")
ROUTE_RE = re.compile(r"\b(?:app|router)\.(get|post|put|delete|patch)\(\s*[\"'`]([^\"'`]+)[\"'`]")
CATALOG_RE = re.compile(r"path:\s*[\"'`](/api/v1[^\"'`]*)")
PROBE_RE = re.compile(r"^        case \"([a-z0-9_-]+)\"")


def _read(path):
    with open(path, errors="ignore") as f:
        return f.read()


def server_routes(root=ROOT):
    """[(METHOD, path, file, line)] in file order, first registration wins."""
    seen, out = set(), []
    for f in sorted(glob.glob(os.path.join(root, "server", "*.ts"))):
        if f.endswith(".test.ts"):
            continue
        for i, line in enumerate(_read(f).split("\n"), 1):
            m = ROUTE_RE.search(line)
            if m and (m.group(1), m.group(2)) not in seen:
                seen.add((m.group(1), m.group(2)))
                out.append((m.group(1).upper(), m.group(2), os.path.relpath(f, root), i))
    return out


def client_text(root=ROOT):
    parts = []
    for f in sorted(glob.glob(os.path.join(root, "client", "src", "**", "*.ts*"), recursive=True)):
        if ".test." not in f:
            parts.append(_read(f))
    return "\n".join(parts)


def route_base(path):
    """The part of a route a client string must contain: up to the first parameter."""
    return re.split(r"/:|\*", path)[0].rstrip("/")


def page_calls(path, client):
    """A page requests this route: its path up to the first parameter appears in
    client/src AND ends there. Substring alone is not enough: /api/news is not
    called by a page that reads /api/newsletter/issues, and /api/x/track/:hex is
    not called by one that reads /api/x/tracked (both were counted as called
    until 2026-11-20). After the matched part a route WITH a parameter may go on
    with "/" (its value follows); a fixed route may not go on at all."""
    base = route_base(path)
    if not base:
        return False
    has_param = re.search(r"/:|\*", path) is not None
    tail = r"(?![A-Za-z0-9_\-])" if has_param else r"(?![A-Za-z0-9_\-/])"
    return re.search(re.escape(base) + tail, client) is not None


def api_catalog(root=ROOT):
    out = set()
    for f in glob.glob(os.path.join(root, "server", "*.ts")):
        if not f.endswith(".test.ts"):
            out.update(CATALOG_RE.findall(_read(f)))
    return out


def diag_probes(root=ROOT):
    """Probe names: the `case "x"` labels of the /api/diag/:probe switch in server/bot.ts."""
    lines = _read(os.path.join(root, "server", "bot.ts")).split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if 'app.get("/api/diag/:probe"' in l)
    except StopIteration:
        return []
    out = []
    for l in lines[start:]:
        m = PROBE_RE.match(l)
        if m:
            out.append(m.group(1))
        elif out and l.startswith("  app."):   # the next route registration ends the switch
            break
    return out


def build(root=ROOT):
    notes = json.load(open(NOTES if root == ROOT else os.path.join(root, "ci", "ui_coverage_notes.json")))
    client = client_text(root)
    catalog = api_catalog(root)
    rows, unclassified, redundant = [], [], []
    for method, path, f, line in server_routes(root):
        key = f"{method} {path}"
        # the file, not the line: a line number would make this register stale
        # on every edit above a route, and the register is checked by a test
        where = f
        note = notes["routes"].get(key)
        if page_calls(path, client):
            rows.append(("route", key, where, "surfaced", "a page calls it"))
            if note:
                redundant.append(key)
        elif path in catalog:
            rows.append(("route", key, where, "surfaced", "keyed API: listed on /developers from the live catalog (/api/v1/meta)"))
            if note:
                redundant.append(key)
        elif note:
            rows.append(("route", key, where, note[0], note[1]))
        else:
            rows.append(("route", key, where, "UNCLASSIFIED", ""))
            unclassified.append(key)
    for p in diag_probes(root):
        note = notes["probes"].get(p)
        if note:
            rows.append(("probe", p, "server/bot.ts /api/diag/" + p, note[0], note[1]))
        else:
            rows.append(("probe", p, "server/bot.ts /api/diag/" + p, "UNCLASSIFIED", ""))
            unclassified.append("probe " + p)
    for name, note in notes["strategies"].items():
        rows.append(("strategy", name, name, note[0], note[1]))
    stale = redundant + [k for k in notes["routes"] if k not in {r[1] for r in rows if r[0] == "route"}]
    stale += ["probe " + k for k in notes["probes"] if k not in {r[1] for r in rows if r[0] == "probe"}]
    bad = [r[1] for r in rows if r[3] not in STATUSES and r[3] != "UNCLASSIFIED"]
    layers = json.load(open(os.path.join(root, "datacore", "layers.json")))["layers"]
    return {"rows": rows, "unclassified": unclassified, "stale_notes": stale, "bad_status": bad, "layers": layers}


def counts(rows, kind):
    out = {s: 0 for s in STATUSES + ("UNCLASSIFIED",)}
    for r in rows:
        if r[0] == kind:
            out[r[3]] += 1
    return out


def render(b):
    rows, layers = b["rows"], b["layers"]
    L = []
    w = L.append
    w("# UI coverage — which backend outputs reach the user interface")
    w("")
    w("GENERATED by `python3 scripts/ui_coverage.py --write`. Do not edit by hand: change the code or")
    w("`ci/ui_coverage_notes.json` and regenerate. `test_ui_coverage.py` fails when this file is stale or")
    w("when a backend item has no classification.")
    w("")
    w("Rule (CLAUDE.md STANDING BEHAVIORS, UI INTEGRATION AND ONE ORDER PATH): every backend feature is")
    w("integrated into the existing UI where it is relevant to the user, by extending existing views, and")
    w("any PR that adds or changes a backend feature updates its row here.")
    w("")
    w("Status: **surfaced** a page shows it · **partial** a page shows part of it · **missing** user-relevant,")
    w("no UI yet · **plumbing** not a user feature · **legacy** nothing reads it; a removal candidate, not a UI gap.")
    w("")
    w("\"A page calls it\" is derived from the code: the route's path appears in `client/src`. It proves a")
    w("page requests it, not that the page shows every field.")
    w("")
    w("## Summary")
    w("")
    w("| | surfaced | partial | missing | plumbing | legacy |")
    w("|---|---|---|---|---|---|")
    for kind, label in (("route", "HTTP routes"), ("probe", "Diagnostic probes (token-gated)"), ("strategy", "Strategies and decision stages")):
        c = counts(rows, kind)
        w(f"| {label} | {c['surfaced']} | {c['partial']} | {c['missing']} | {c['plumbing']} | {c['legacy']} |")
    live = sum(1 for l in layers if l.get("status") == "live")
    w("")
    w(f"Map layers: {len(layers)} in `datacore/layers.json` ({live} live). The /data layers panel, the legend and")
    w("the layer cards are drawn from that file, so a layer in the registry is on the map by construction;")
    w("layers are not listed one by one here.")
    w("")
    for status, title in (("missing", "Missing: user-relevant, no UI yet"), ("partial", "Partial"), ("legacy", "Legacy: removal candidates, not UI gaps")):
        sel = [r for r in rows if r[3] == status]
        w(f"## {title} ({len(sel)})")
        w("")
        w("| Kind | Item | Backend location | What it is, and where it belongs in the existing UI |")
        w("|---|---|---|---|")
        for kind, item, where, _s, note in sel:
            w(f"| {kind} | `{item}` | `{where}` | {note} |")
        w("")
    sel = [r for r in rows if r[3] == "plumbing"]
    w(f"## Plumbing: not user features ({len(sel)})")
    w("")
    w("| Kind | Item | Why |")
    w("|---|---|---|")
    for kind, item, where, _s, note in sel:
        w(f"| {kind} | `{item}` | {note} |")
    w("")
    sel = [r for r in rows if r[3] == "surfaced"]
    w(f"## Surfaced ({len(sel)})")
    w("")
    w("| Kind | Item | Backend location | How |")
    w("|---|---|---|---|")
    for kind, item, where, _s, note in sel:
        w(f"| {kind} | `{item}` | `{where}` | {note} |")
    w("")
    return "\n".join(L)


def problems(b):
    out = []
    if b["unclassified"]:
        out.append("UNCLASSIFIED backend items (add a [status, note] to ci/ui_coverage_notes.json, or make a page use them): " + ", ".join(b["unclassified"]))
    if b["stale_notes"]:
        out.append("notes for items that no longer exist, or that a page now calls (delete them from ci/ui_coverage_notes.json): " + ", ".join(b["stale_notes"]))
    if b["bad_status"]:
        out.append("notes with a status outside " + "/".join(STATUSES) + ": " + ", ".join(b["bad_status"]))
    return out


def main(argv):
    b = build()
    text = render(b)
    probs = problems(b)
    if "--write" in argv:
        with open(OUT, "w") as f:
            f.write(text)
        print("wrote", os.path.relpath(OUT, ROOT))
    if "--check" in argv:
        current = _read(OUT) if os.path.exists(OUT) else ""
        if current != text:
            probs.append("UI_COVERAGE.md is stale: run `python3 scripts/ui_coverage.py --write` and commit it")
    for kind in ("route", "probe", "strategy"):
        print(kind, counts(b["rows"], kind))
    for p in probs:
        print("PROBLEM:", p)
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
