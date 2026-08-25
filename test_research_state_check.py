"""
test_research_state_check.py — pins the pure parsers/classifiers in
scripts/research_state_check.py against synthetic text fixtures shaped
like research/experiments.md and research/open_questions.md, including
the two real-file bugs found and fixed while building this script (the
AUDITS & DEBT register table needing section-scoping so an unrelated pipe
table elsewhere in the 58k-line file isn't picked up, and the KNOWN BROKEN
item pattern needing to match numbered items without a '[STATUS]' bracket,
e.g. item #4).
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import research_state_check as rsc  # noqa: E402


# ── extract_audits_register_section / parse_audits_register ───────────────

EXPERIMENTS_FIXTURE = """# Experiment Log

Append-only. Newest at top.

## AUDITS & DEBT REGISTER (live state)

| Audit | Cadence | Last run | Next due |
|---|---|---|---|
| STALENESS AUDIT | 30d | 2026-08-15 | 2026-09-14 |
| CONSTITUTIONAL AUDIT | 30d | 2026-08-16 | 2026-09-15 |
| CALENDAR YEAR-ADD | annual (December) | never yet run | 2026-12-01 |

## 2026-08-20 (scheduled-routine PRODUCT session) [PRODUCT] — first entry

body text here

## 2026-08-19 (5) (scheduled-routine PRODUCT session) [PRODUCT] — second

body text

## 2026-08-18 — [REPAIR] third entry

body

## 2026-08-17 — [REPAIR] fourth entry

body

## 2026-08-16 — [REPAIR] fifth entry

body

## 2026-08-15 — [REPAIR] sixth entry

body

## 2026-08-14 — [REPAIR] seventh entry

body

## 2026-08-13 — [PIPELINE] eighth entry

body

## 2026-08-12 — [RESEARCH] ninth entry

body

## 2026-08-11 — [RULE-REVIEW] tenth entry

body

## 2026-08-10 — [PRODUCT] eleventh entry (outside the 10-window)

body

## GRID VISION UNRELATED RESULTS TABLE (deliberately shaped like a
register table, further down the file, must NOT be picked up)

| held-out region | baseline (n, default, 1-region) | + strong aug (s) |
|---|---|---|
| AZ (desert) | 0.056 | 0.147 |
| KS (plains) | 0.059 | 0.107 |
"""


def test_extract_audits_register_section_stops_before_next_heading():
    section = rsc.extract_audits_register_section(EXPERIMENTS_FIXTURE)
    assert "STALENESS AUDIT" in section
    assert "GRID VISION" not in section
    assert "held-out region" not in section


def test_parse_audits_register_ignores_unrelated_tables_elsewhere_in_file():
    rows = rsc.parse_audits_register(EXPERIMENTS_FIXTURE)
    assert len(rows) == 3
    assert {r["audit"] for r in rows} == {"STALENESS AUDIT", "CONSTITUTIONAL AUDIT", "CALENDAR YEAR-ADD"}


def test_parse_audits_register_parses_next_due_dates():
    rows = rsc.parse_audits_register(EXPERIMENTS_FIXTURE)
    by_audit = {r["audit"]: r for r in rows}
    assert by_audit["STALENESS AUDIT"]["next_due"] == "2026-09-14"


def test_parse_audits_register_handles_never_yet_run_last_run():
    rows = rsc.parse_audits_register(EXPERIMENTS_FIXTURE)
    by_audit = {r["audit"]: r for r in rows}
    assert by_audit["CALENDAR YEAR-ADD"]["last_run"] == "never yet run"
    assert by_audit["CALENDAR YEAR-ADD"]["next_due"] == "2026-12-01"


def test_parse_audits_register_empty_when_no_section():
    assert rsc.parse_audits_register("no register here") == []


# ── check_audits_overdue ───────────────────────────────────────────────────

def test_check_audits_overdue_ok_when_all_future():
    register = [{"audit": "A", "cadence": "30d", "last_run": "x", "next_due": "2099-01-01"}]
    f = rsc.check_audits_overdue(register, date(2026, 8, 20))
    assert f["severity"] == rsc.OK


def test_check_audits_overdue_warn_when_past_due():
    register = [{"audit": "A", "cadence": "30d", "last_run": "x", "next_due": "2026-01-01"}]
    f = rsc.check_audits_overdue(register, date(2026, 8, 20))
    assert f["severity"] == rsc.WARN
    assert "A (due 2026-01-01)" in f["detail"]


def test_check_audits_overdue_warn_when_register_empty():
    f = rsc.check_audits_overdue([], date(2026, 8, 20))
    assert f["severity"] == rsc.WARN


def test_check_audits_overdue_ignores_rows_with_no_parsed_date():
    register = [{"audit": "A", "cadence": "annual", "last_run": "never", "next_due": None}]
    f = rsc.check_audits_overdue(register, date(2026, 8, 20))
    assert f["severity"] == rsc.OK


# ── parse_session_tags / check_thrash_ratio ────────────────────────────────

def test_parse_session_tags_reads_newest_first_up_to_window():
    tags = rsc.parse_session_tags(EXPERIMENTS_FIXTURE, window=10)
    assert tags == [
        "PRODUCT", "PRODUCT", "REPAIR", "REPAIR", "REPAIR",
        "REPAIR", "REPAIR", "PIPELINE", "RESEARCH", "RULE-REVIEW",
    ]
    # the 11th entry (outside the window) must not appear
    assert len(tags) == 10


def test_parse_session_tags_records_none_for_untagged_header():
    text = "## 2026-08-20 — an entry with no bracket tag at all\n\nbody\n"
    tags = rsc.parse_session_tags(text, window=1)
    assert tags == [None]


def test_parse_session_tags_keeps_header_with_unparseable_date_instead_of_dropping_it():
    # "2026-13-01" matches the header regex's digit-shape but is not a real
    # calendar date (month 13). It must still surface in the tag list (sorted
    # as oldest) rather than silently vanishing, per the same "malformed
    # entry cannot hide from the ratio" principle as the untagged case above.
    text = (
        "## 2026-08-20 — [PRODUCT] real entry\n\nbody\n\n"
        "## 2026-13-01 — [REPAIR] entry with a typo'd month\n\nbody\n"
    )
    tags = rsc.parse_session_tags(text, window=10)
    assert tags == ["PRODUCT", "REPAIR"]


# KNOWN BROKEN #36 regression: a same-or-later-dated block landed at the
# file's TAIL (not its head) — reproduces the real bug where a dozen
# sessions were merged in below older content instead of above it.
MISORDERED_FIXTURE = """# Experiment Log

Append-only. Newest at top.

## 2026-08-20 — [PRODUCT] correctly-placed newest entry at the top

body

## 2026-08-10 — [PIPELINE] an old entry, correctly near the top for its date

body

## 2026-08-19 — [REPAIR] a NEWER entry that landed below an older one

body

## 2026-08-18 — [REPAIR] another newer-than-08-10 entry, also misplaced

body

## 2026-07-03 — [RESEARCH] a genuinely ancient entry, correctly at the tail

body
"""


def test_parse_session_tags_sorts_by_date_not_physical_position():
    # Physical top-down order is: 08-20, 08-10, 08-19, 08-18, 07-03.
    # Chronological order must be: 08-20, 08-19, 08-18, 08-10, 07-03.
    tags = rsc.parse_session_tags(MISORDERED_FIXTURE, window=10)
    assert tags == ["PRODUCT", "REPAIR", "REPAIR", "PIPELINE", "RESEARCH"]


def test_parse_session_tags_same_date_ties_break_by_file_order():
    text = (
        "## 2026-08-20 — [PRODUCT] first physically, same date\n\nbody\n\n"
        "## 2026-08-20 — [REPAIR] second physically, same date\n\nbody\n"
    )
    tags = rsc.parse_session_tags(text, window=10)
    assert tags == ["PRODUCT", "REPAIR"]


def test_check_thrash_ratio_reflects_misordered_dates_not_physical_position():
    # This is the actual #36 failure mode: truncating to a small window.
    # Physical top-down scan of the first 3 headers hits 08-20/08-10/08-19
    # (1 REPAIR — the 08-10 PIPELINE entry displaces a real REPAIR out of
    # the window). Chronological order is 08-20/08-19/08-18, both REPAIR —
    # a materially different, correct ratio.
    tags = rsc.parse_session_tags(MISORDERED_FIXTURE, window=3)
    assert tags == ["PRODUCT", "REPAIR", "REPAIR"]
    f = rsc.check_thrash_ratio(tags, trigger=2, window=3)
    assert f["severity"] == rsc.ALARM
    assert "2/3 REPAIR" in f["detail"]


def test_check_thrash_ratio_ok_below_trigger():
    tags = rsc.parse_session_tags(EXPERIMENTS_FIXTURE, window=10)
    f = rsc.check_thrash_ratio(tags)
    assert f["severity"] == rsc.OK
    assert "5/10 REPAIR" in f["detail"]


def test_check_thrash_ratio_alarm_at_trigger():
    tags = ["REPAIR"] * 7 + ["PRODUCT"] * 3
    f = rsc.check_thrash_ratio(tags)
    assert f["severity"] == rsc.ALARM
    assert "7/10 REPAIR" in f["detail"]
    assert "Priority-1" in f["detail"]


def test_check_thrash_ratio_warn_when_window_incomplete():
    f = rsc.check_thrash_ratio(["REPAIR", "PRODUCT"])
    assert f["severity"] == rsc.WARN


def test_check_thrash_ratio_counts_untagged_as_non_repair():
    tags = ["REPAIR"] * 6 + [None] * 4
    f = rsc.check_thrash_ratio(tags)
    assert f["severity"] == rsc.OK
    assert "6/10 REPAIR" in f["detail"]
    assert "4 untagged" in f["detail"]


# ── extract_known_broken_section / parse_known_broken_items ───────────────

OPEN_QUESTIONS_FIXTURE = """# Open Questions

## KNOWN BROKEN — fix these first (repair mandate)

1. **[RESOLVED 2026-07-03]** ~~First item.~~ closed cleanly.

2. **Human-reported: no bracket at all on this header.**
   Some unresolved prose describing a symptom, never given a status tag.

3. **[FOUND 2026-07-11, not fixed — design/threshold judgment call]**
   Deliberately left open pending RULE-REVIEW evidence, per this repo's
   own convention (this text must NOT match on a bare "FIXED" substring
   inside "not fixed").

   UPDATE 2026-08-01: still gated, no new evidence yet.

4. **[FOUND 2026-08-13, NOT PATCHED]** Third recurrence, structural fix
   proposed in wishlist.md, not shipped this session.

   STRUCTURAL FIX SHIPPED 2026-08-15: closes the gap this item diagnosed.

   LIVE VERIFICATION CONFIRMED 2026-08-16. **ITEM #4 CLOSED.**

## RULE COST AUDIT — after counterfactual logging exists

- some unrelated bullet, not part of KNOWN BROKEN
- 1. **[this looks like a numbered item but is NOT inside KNOWN BROKEN]**
"""


def test_extract_known_broken_section_stops_before_next_heading():
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    assert "First item" in section
    assert "RULE COST AUDIT" not in section
    assert "not part of KNOWN BROKEN" not in section


def test_parse_known_broken_items_finds_all_four_including_bracketless():
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    items = rsc.parse_known_broken_items(section)
    assert [it["number"] for it in items] == [1, 2, 3, 4]


def test_parse_known_broken_items_attaches_update_paragraphs_to_owning_item():
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    items = rsc.parse_known_broken_items(section)
    item3 = next(it for it in items if it["number"] == 3)
    assert "still gated, no new evidence" in item3["block"]
    assert "Third recurrence" not in item3["block"]


# ── classify_known_broken / check_known_broken ─────────────────────────────

def test_classify_known_broken_closed_on_resolved_bracket():
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    items = {it["number"]: it for it in rsc.parse_known_broken_items(section)}
    assert rsc.classify_known_broken(items[1]) == "CLOSED"


def test_classify_known_broken_needs_review_when_no_marker():
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    items = {it["number"]: it for it in rsc.parse_known_broken_items(section)}
    assert rsc.classify_known_broken(items[2]) == "NEEDS-REVIEW"


def test_classify_known_broken_does_not_false_positive_on_not_fixed_substring():
    """The single most important negative case: an item whose header says
    "not fixed" must never read as closed just because "fixed" is a
    substring of "not fixed" — that would silently hide the most
    explicitly-still-open item shape this repo uses (see the real #20)."""
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    items = {it["number"]: it for it in rsc.parse_known_broken_items(section)}
    assert rsc.classify_known_broken(items[3]) == "NEEDS-REVIEW"


def test_classify_known_broken_closed_via_later_item_closed_sentence():
    """Mirrors the real item #30 shape: opens as NOT PATCHED, closes via a
    later '**ITEM #N CLOSED.**' sentence rather than in its own bracket."""
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    items = {it["number"]: it for it in rsc.parse_known_broken_items(section)}
    assert rsc.classify_known_broken(items[4]) == "CLOSED"


def test_check_known_broken_lists_open_item_numbers():
    section = rsc.extract_known_broken_section(OPEN_QUESTIONS_FIXTURE)
    items = rsc.parse_known_broken_items(section)
    f = rsc.check_known_broken(items)
    assert f["severity"] == rsc.OK
    assert "#2, #3" in f["detail"]
    assert "4 items total" in f["detail"]


def test_check_known_broken_warn_when_no_items_found():
    f = rsc.check_known_broken([])
    assert f["severity"] == rsc.WARN


# ── overall_exit_code ───────────────────────────────────────────────────────

def test_overall_exit_code_worst_of_all_findings():
    findings = [rsc.finding(rsc.OK, "a", ""), rsc.finding(rsc.ALARM, "b", ""), rsc.finding(rsc.WARN, "c", "")]
    assert rsc.overall_exit_code(findings) == 2


def test_overall_exit_code_zero_when_all_ok():
    findings = [rsc.finding(rsc.OK, "a", ""), rsc.finding(rsc.OK, "b", "")]
    assert rsc.overall_exit_code(findings) == 0


# ── run against the real repo files (smoke test, not a behavior pin) ──────

def test_run_all_checks_against_real_repo_files_does_not_crash():
    repo_root = os.path.join(os.path.dirname(__file__))
    register, tags, items = rsc.gather(repo_root)
    findings = rsc.run_all_checks(register, tags, items, date.today())
    assert len(findings) == 3
    assert all(f["severity"] in (rsc.OK, rsc.WARN, rsc.ALARM) for f in findings)
    # the real register and KNOWN BROKEN section must both be non-empty —
    # an empty result here would mean the section-boundary parsing drifted
    # from the real files' current headings.
    assert len(register) >= 1
    assert len(items) >= 1
