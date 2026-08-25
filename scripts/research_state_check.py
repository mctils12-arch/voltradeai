#!/usr/bin/env python3
"""
research_state_check.py — compile the MEMORY PROTOCOL's manual "read
open_questions.md's KNOWN BROKEN section, compute the thrash ratio, check
the AUDITS & DEBT register" step into a script (EDGE DOCTRINE #3: COMPILE
KNOWLEDGE INTO CODE — never reason the same diagnosis twice).

Every scheduled-routine session's own SESSION-START CHECKS narrative
re-derives the same three things by hand, verbatim in shape, from
research/open_questions.md (now ~11,000 lines / ~700KB) and
research/experiments.md (now ~58,000 lines): "walked every numbered KNOWN
BROKEN entry end to end", "loop-health ratio, last 10 tagged entries",
"AUDITS & DEBT register ... nothing overdue". That reasoning has been done
by hand, near-identically, across dozens of sessions (grep either file for
"Loop-health ratio" or "Walked every numbered KNOWN BROKEN entry" to see
the pattern repeat) — exactly EDGE DOCTRINE #3's "the second occurrence
becomes a script" case, and unlike scripts/session_health_check.py (which
checks LIVE production state), nothing before this covered the STATIC
research/ bookkeeping a session currently has to open two huge files and
read by eye to reconstruct.

This is READ-ONLY and LOCAL: it parses research/open_questions.md and
research/experiments.md from disk. No network call, no side effects, no
substitute for READ BEFORE WRITE — anything this script flags as OPEN or
NEEDS-REVIEW is a pointer to go read the source text, not a verdict. The
KNOWN BROKEN classifier in particular is a best-effort heuristic over
organically-evolved prose (~30 items, several different closing phrasings
used over weeks by different sessions) — it is deliberately biased toward
under-classifying (reporting NEEDS-REVIEW rather than guessing CLOSED),
since a false CLOSED risks silently skipping a still-open repair item
while a false NEEDS-REVIEW only costs a future session a few seconds of
re-confirming something that was already fine.

Usage:
  python3 scripts/research_state_check.py [--repo-root PATH] [--json]

Exit code: 0 = all clear, 1 = WARN present (non-blocking, log and move
on), 2 = ALARM present (thrash ratio at/above the Priority-1 trigger).
"""
import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timezone

OK, WARN, ALARM = "OK", "WARN", "ALARM"
_SEVERITY_RANK = {OK: 0, WARN: 1, ALARM: 2}

VALID_TAGS = ("REPAIR", "RESEARCH", "RULE-REVIEW", "PIPELINE", "PRODUCT", "NO-ACTION")
THRASH_WINDOW = 10
THRASH_TRIGGER = 7

# Explicit closing phrasings actually used in research/open_questions.md's
# KNOWN BROKEN section as of 2026-08-20 (grepped, not guessed) — kept as a
# named tuple so a future session extending the convention only has to add
# a string here, not touch the classifier logic. Word-boundary-matched
# (see classify_known_broken) so "RESOLVED" cannot false-positive-match
# inside "UNRESOLVED", the same substring trap "FIXED" would hit inside
# "NOT FIXED" if it were ever added bare to this tuple.
_CLOSED_MARKERS = (
    "RESOLVED", "CLOSED", "ROOT CAUSE FOUND + FIXED", "FOUND + FIXED",
)
_CLOSED_MARKER_RES = [re.compile(r"\b" + re.escape(m) + r"\b") for m in _CLOSED_MARKERS]


def finding(severity, label, detail):
    return {"severity": severity, "label": label, "detail": detail}


# ── pure parsers/classifiers (no I/O) ─────────────────────────────────────

def extract_audits_register_section(experiments_md_text):
    """Return the text strictly between the '## AUDITS & DEBT REGISTER'
    heading and the next top-level '## ' heading. experiments.md is
    58,000+ lines and contains other unrelated 4-column pipe tables further
    down (e.g. GRID VISION GPU-training result tables) — without this
    scope, a generic table scan over the whole file picks those up too."""
    lines = experiments_md_text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and line.strip().startswith("## AUDITS & DEBT REGISTER"):
            start = i + 1
            continue
        if start is not None and line.startswith("## "):
            end = i
            break
    if start is None:
        return ""
    return "\n".join(lines[start:end] if end is not None else lines[start:])


def parse_audits_register(experiments_md_text):
    """Parse the '| Audit | Cadence | Last run | Next due |' markdown table
    CLAUDE.md's AUDITS & DEBT section mandates at the top of
    research/experiments.md. Returns a list of
    {audit, cadence, last_run, next_due} dicts (next_due may be None for
    rows like CALENDAR YEAR-ADD's 'never yet run' last_run — next_due is
    still a real date there and is parsed independently)."""
    section = extract_audits_register_section(experiments_md_text)
    rows = []
    for line in section.splitlines():
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", line)
        if not m:
            continue
        audit, cadence, last_run, next_due = (g.strip() for g in m.groups())
        if audit in ("Audit", "---") or set(audit) == {"-"}:
            continue
        due_match = re.match(r"^(\d{4}-\d{2}-\d{2})$", next_due)
        rows.append({
            "audit": audit,
            "cadence": cadence,
            "last_run": last_run,
            "next_due": due_match.group(1) if due_match else None,
        })
    return rows


def check_audits_overdue(register, today):
    """today: a date object. Any row whose next_due has passed is WARN —
    matches the AUDITS & DEBT rule's own framing (debt, not an emergency);
    LIVENESS/repair items carry their own ALARM path elsewhere."""
    if not register:
        return finding(WARN, "audits_register", "no AUDITS & DEBT register table found in experiments.md")
    overdue = []
    for row in register:
        if not row["next_due"]:
            continue
        due = datetime.strptime(row["next_due"], "%Y-%m-%d").date()
        if due < today:
            overdue.append(f"{row['audit']} (due {row['next_due']})")
    if overdue:
        return finding(WARN, "audits_register", f"overdue: {'; '.join(overdue)}")
    return finding(OK, "audits_register", f"none overdue ({len(register)} audits tracked)")


_SESSION_HEADER_RE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\b")
_TAG_RE = re.compile(r"\[(" + "|".join(re.escape(t) for t in VALID_TAGS) + r")\]")


def parse_session_tags(experiments_md_text, window=THRASH_WINDOW):
    """research/experiments.md is append-only and its header says NEWEST
    AT TOP — but physical position cannot be trusted as a proxy for
    recency: KNOWN BROKEN #36 found a same-day block of a dozen sessions
    landing at the file's tail instead of its head, because the
    WORKSTREAM PARTITION MERGE-ORDER PROTOCOL resolves concurrent
    research/* conflicts by keeping both sides (append-only spirit), which
    can place a session's entry anywhere relative to other concurrent
    sessions' entries, not just at the head. What every header DOES
    reliably carry is an accurate `YYYY-MM-DD` date (sessions get the date
    right; only physical placement drifts under concurrent merges) — so
    this scans the WHOLE file, parses each header's date, and sorts
    newest-date-first regardless of physical position. Original top-down
    file order is used only as a tiebreaker among headers sharing the same
    calendar date (the one residual ambiguity plain-text dates can't
    resolve — same-day session order — versus the multi-day
    misordering this replaces). Returns the `window` most-recent tags,
    newest first; a header with no recognizable [TAG] on its own line is
    recorded as None (counted as "untagged", not silently dropped, so a
    malformed entry cannot hide from the ratio)."""
    entries = []
    for idx, line in enumerate(experiments_md_text.splitlines()):
        m = _SESSION_HEADER_RE.match(line)
        if not m:
            continue
        try:
            header_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            # The regex only checks digit-shape (\d{4}-\d{2}-\d{2}), not a
            # valid calendar date, so a typo like "2026-13-01" matches here
            # and fails here. Don't silently drop it (same "malformed entry
            # cannot hide" principle as the untagged-header case below) —
            # surface it and sort it as the oldest possible entry, so a
            # malformed date can fall out of a small window on its own
            # merits rather than vanishing unreported.
            print(f"[research_state_check] WARNING: unparseable date in header: {line!r}", file=sys.stderr)
            header_date = date.min
        tag_m = _TAG_RE.search(line)
        entries.append((header_date, idx, tag_m.group(1) if tag_m else None))
    entries.sort(key=lambda e: (-e[0].toordinal(), e[1]))
    return [tag for _, _, tag in entries[:window]]


def check_thrash_ratio(tags, trigger=THRASH_TRIGGER, window=THRASH_WINDOW):
    """CLAUDE.md HEALTH OF THE LOOP ITSELF rule 2: 7+ of the last 10
    tagged entries being [REPAIR] means the meta-problem ('system
    generates breaks faster than fixes hold') becomes the Priority-1 item
    — ALARM, not WARN, since the rule calls it out by name as
    Priority-1."""
    if len(tags) < window:
        return finding(
            WARN, "thrash_ratio",
            f"only {len(tags)} tagged session(s) found (need {window}) — "
            "experiments.md may be shorter than expected or the header "
            "pattern didn't match; verify manually",
        )
    repair_count = sum(1 for t in tags if t == "REPAIR")
    untagged = sum(1 for t in tags if t is None)
    detail = f"{repair_count}/{window} REPAIR in the last {window} tagged sessions"
    if untagged:
        detail += f" ({untagged} untagged header(s) counted as non-REPAIR)"
    if repair_count >= trigger:
        return finding(
            ALARM, "thrash_ratio",
            f"{detail} — AT/ABOVE the {trigger}+ Priority-1 trigger "
            "(CLAUDE.md HEALTH OF THE LOOP ITSELF rule 2): stop normal "
            "work, diagnose the break-generator, not the next break",
        )
    return finding(OK, "thrash_ratio", f"{detail} — below the {trigger}+ thrash trigger")


_KNOWN_BROKEN_ITEM_RE = re.compile(r"^(\d+)\.\s+\*\*(.*)$")


def extract_known_broken_section(open_questions_md_text):
    """Return the text strictly between the '## KNOWN BROKEN' heading and
    the next top-level '## ' heading (RULE COST AUDIT today)."""
    lines = open_questions_md_text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and line.strip().startswith("## KNOWN BROKEN"):
            start = i + 1
            continue
        if start is not None and line.startswith("## ") and not line.strip().startswith("## KNOWN BROKEN"):
            end = i
            break
    if start is None:
        return ""
    return "\n".join(lines[start:end] if end is not None else lines[start:])


def parse_known_broken_items(section_text):
    """Split the KNOWN BROKEN section into numbered items. Each item's
    'block' is everything from its own numbered header line up to (not
    including) the next numbered item, so multi-paragraph UPDATE notes
    stay attached to the item they amend. Returns a list of
    {number, header, block} in file order (ascending item number is the
    house convention, not enforced here)."""
    lines = section_text.splitlines()
    starts = [i for i, line in enumerate(lines) if _KNOWN_BROKEN_ITEM_RE.match(line)]
    items = []
    for idx, i in enumerate(starts):
        j = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        block = "\n".join(lines[i:j])
        num = _KNOWN_BROKEN_ITEM_RE.match(lines[i]).group(1)
        header = lines[i]
        items.append({"number": int(num), "header": header.strip(), "block": block})
    return items


def classify_known_broken(item):
    """Advisory-only, see module docstring. CLOSED requires an explicit
    marker string (RESOLVED/CLOSED/FIXED-with-a-root-cause) ANYWHERE in
    the item's block, since several items (e.g. #30) close via a later
    '**ITEM #N CLOSED.**' sentence rather than in their own opening
    bracket. Everything else is NEEDS-REVIEW, INCLUDING items that are
    deliberately, correctly left open pending RULE-REVIEW evidence (e.g.
    #20) — this function does not know the difference between 'still
    genuinely broken' and 'open by design awaiting evidence'; a session
    reading the flagged block is what tells them apart, same as today."""
    block_upper = item["block"].upper()
    if any(pat.search(block_upper) for pat in _CLOSED_MARKER_RES):
        return "CLOSED"
    return "NEEDS-REVIEW"


def check_known_broken(items):
    if not items:
        return finding(WARN, "known_broken", "no numbered KNOWN BROKEN items found — parser or file may have drifted")
    open_items = [it for it in items if classify_known_broken(it) == "NEEDS-REVIEW"]
    if open_items:
        nums = ", ".join(f"#{it['number']}" for it in open_items)
        return finding(
            OK, "known_broken",
            f"{len(items)} items total, {len(open_items)} without an explicit close marker "
            f"({nums}) — advisory only, read each before treating it as a repair blocker",
        )
    return finding(OK, "known_broken", f"{len(items)} items total, all carry an explicit close marker")


def run_all_checks(register, tags, known_broken_items, today):
    return [
        check_audits_overdue(register, today),
        check_thrash_ratio(tags),
        check_known_broken(known_broken_items),
    ]


def overall_exit_code(findings):
    worst = max((_SEVERITY_RANK[f["severity"]] for f in findings), default=0)
    return {0: 0, 1: 1, 2: 2}[worst]


# ── local file I/O (thin, not unit-tested directly) ───────────────────────

def _read(repo_root, relpath):
    path = os.path.join(repo_root, relpath)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        print(f"[research_state_check] could not read {relpath}: {e}", file=sys.stderr)
        return ""


def gather(repo_root):
    experiments = _read(repo_root, "research/experiments.md")
    open_questions = _read(repo_root, "research/open_questions.md")
    register = parse_audits_register(experiments)
    tags = parse_session_tags(experiments)
    kb_section = extract_known_broken_section(open_questions)
    items = parse_known_broken_items(kb_section)
    return register, tags, items


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument(
        "--repo-root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
    )
    ap.add_argument("--json", action="store_true", help="emit findings as JSON instead of text")
    args = ap.parse_args()

    register, tags, items = gather(os.path.abspath(args.repo_root))
    findings = run_all_checks(register, tags, items, date.today())

    if args.json:
        print(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "findings": findings}, indent=2))
    else:
        for f in findings:
            print(f"[{f['severity']:5s}] {f['label']}: {f['detail']}")

    return overall_exit_code(findings)


if __name__ == "__main__":
    sys.exit(main())
