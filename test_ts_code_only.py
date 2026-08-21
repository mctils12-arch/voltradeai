"""Q13 — the §4.2 counters must count CODE, not prose about code.

WHAT WENT WRONG. `empty_ts_catch` and `ts_any` were `grep` over raw file text,
so a comment describing the pattern counted as the pattern. Fifteen matches on
the 2026-08-14 tree were documentation, not code.

This is the sixth time in one session a source-scraping check has been defeated
by a comment naming what it looks for (L9, L11, L12, L15, and twice during
#833 — one of those produced a false PASS, the dangerous direction). These
tests exist so the seventh time fails a build instead of a session.

THE TWO CONSTRAINTS BELOW ARE THE POINT. Each has its own test because each
was discovered by implementing the naive version first and watching it produce
a wrong number:

  * `test_blanking_never_creates_a_match` — the counters must exclude by
    LOCATION, not by re-matching cleaned text. Cleaning first sends
    `empty_ts_catch` UP (495 -> 516 on this tree) by turning
    `catch { // why }` into `catch {        }`, which is exactly D4
    `commented_empty_catch`'s domain. That would silently merge two counters
    the program keeps disjoint on purpose.

  * `test_a_malformed_line_cannot_poison_the_next_one` — the scan is per line.
    A file-wide lexer desynced on the regex literal `/'/g` at
    server/billing.ts:83 and declared the following 30 lines non-code,
    excluding four REAL `catch (err: any)` annotations without a word.
"""
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from ts_code_only import (  # noqa: E402
    blank_noncode,
    blank_source,
    code_matches,
    read_text,
)

EMPTY_CATCH = re.compile(r"catch\s*(\([^)]*\))?\s*\{\s*\}")
TS_ANY = re.compile(r":\s*any\b")


def test_a_comment_about_the_pattern_is_not_the_pattern():
    """The defect itself. Fails against a raw `grep`, which counts both."""
    src = "// a bare `catch {}` swallows the error\ntry { go(); } catch {}\n"
    assert len(EMPTY_CATCH.findall(src)) == 2, "raw text really does match twice"
    assert len(code_matches(EMPTY_CATCH, src)) == 1


def test_prose_using_the_word_any_is_not_a_type_annotation():
    src = (
        "// Failure-safe: any IndexedDB error is treated as a miss\n"
        "/** default: any port */\n"
        " * deliberately: any CREDIBLE equity read counts\n"
        "function f(x: any) { return x; }\n"
    )
    assert len(TS_ANY.findall(src)) == 4
    assert len(code_matches(TS_ANY, src)) == 1


def test_a_string_asserting_on_source_text_is_not_code():
    """server/tier2DaemonTimeoutVisibility.test.ts:24 and three siblings."""
    src = 'const i = bot.indexOf("} catch (err: any) {", start);\n'
    assert len(TS_ANY.findall(src)) == 1
    assert code_matches(TS_ANY, src) == []


def test_blanking_never_creates_a_match():
    """CONSTRAINT 1 — the counters must not be able to merge.

    A comment-bodied catch belongs to D4 `commented_empty_catch`. If
    `empty_ts_catch` were computed by cleaning the text and re-matching, this
    would count in BOTH.
    """
    src = "try { go(); } catch {\n  // deliberate: readouts must never break\n}\n"
    assert EMPTY_CATCH.findall(src) == [], "not an empty catch to begin with"
    assert EMPTY_CATCH.findall(blank_source(src)), "cleaning DOES create one"
    assert code_matches(EMPTY_CATCH, src) == [], "location exclusion must not"


def test_blanking_preserves_length_exactly():
    """What makes location exclusion possible at all."""
    for line in (
        "const x: any = 1; // any old comment",
        '  bot.indexOf("} catch (err: any) {")',
        "   * a block comment interior",
        "const s = `template ${a.b} text`;",
        "const re = /'/g;",
        "",
    ):
        assert len(blank_noncode(line)) == len(line), repr(line)


def test_a_malformed_line_cannot_poison_the_next_one():
    """CONSTRAINT 2 — the billing.ts:83 regression, as a unit test.

    The apostrophe inside `/'/g` is unbalanced to any quote-counting scan. A
    file-wide lexer swallowed everything after it; a per-line scan may get THIS
    line wrong and must still get the next one right.
    """
    src = (
        "query: `email:'${email.replace(/'/g, \"\\\\'\")}'`,\n"
        "} catch (err: any) {\n"
    )
    assert len(code_matches(TS_ANY, src)) == 1


def test_ambiguity_resolves_toward_counting():
    """Under-exclusion keeps a non-increasing counter honestly high.

    Over-exclusion makes the tree look better than it is, and MEASUREMENT
    INTEGRITY treats a ruler change in the flattering direction as suspect. So
    the unresolvable cases — template literals, unterminated quotes — stay
    counted rather than being guessed away.
    """
    assert len(code_matches(TS_ANY, "const s = `x: any`;\n")) == 1
    assert len(code_matches(TS_ANY, "const s = 'unterminated: any\n")) == 1


def test_the_two_catch_counters_are_disjoint_on_the_real_tree():
    """D4's domain and `empty_ts_catch`'s must not overlap, tree-wide.

    Asserted on the actual repo, not a fixture: the merge risk is a property of
    the two definitions meeting real code, and a fixture cannot show that.
    """
    commented = re.compile(
        r"catch\s*(?:\([^)]*\))?\s*\{\s*(?://[^\n]*|/\*.*?\*/)\s*\}", re.S
    )
    files = [
        f
        for f in subprocess.run(
            ["git", "ls-files", "*.ts", "*.tsx"], capture_output=True, text=True
        ).stdout.split()
        if "/node_modules/" not in f
    ]
    for f in files:
        src = read_text(f)
        assert src is not None, f"{f} is tracked but unreadable — the counter would silently shrink"
        d4 = {(m.start(), m.end()) for m in commented.finditer(src)}
        mine = {(m.start(), m.end()) for m in code_matches(EMPTY_CATCH, src)}
        assert not (d4 & mine), f"{f}: a catch counted by BOTH counters"


@pytest.mark.parametrize(
    "counter,pattern,expected",
    [("empty_ts_catch", EMPTY_CATCH, 493), ("ts_any", TS_ANY, 1239)],
)
def test_the_pinned_values_are_what_the_module_measures(counter, pattern, expected):
    """Ties the module to ci/counter_baseline.txt.

    Not a duplicate of scripts/counter_ratchet.sh: that compares the SCRIPT's
    output to the pin, this compares the MODULE's output to the pin. If
    program_status.sh ever stops calling this module, the ratchet keeps passing
    and only this test notices.
    """
    pins = {}
    for line in open("ci/counter_baseline.txt"):
        parts = line.split("#")[0].split()
        if len(parts) == 3:
            pins[parts[0]] = int(parts[1])
    assert pins[counter] == expected, "pin moved without updating this test"

    files = [
        f
        for f in subprocess.run(
            ["git", "ls-files", "*.ts", "*.tsx"], capture_output=True, text=True
        ).stdout.split()
        if "/node_modules/" not in f
    ]
    total = 0
    for f in files:
        src = read_text(f)
        assert src is not None, f"{f} is tracked but unreadable"
        total += len(code_matches(pattern, src))
    assert total == expected


def test_program_status_actually_uses_this_module():
    """The wiring, so the module cannot become decoration.

    L15's lesson applied: assert on the CONSTRUCT (an import that runs), not on
    a string that a comment could satisfy. Comments are stripped first.
    """
    code = "\n".join(
        l for l in open("scripts/program_status.sh") if not re.match(r"\s*#", l)
    )
    assert "from ts_code_only import" in code
    assert "grep -hoE ':\\s*any" not in code, "the raw grep is back"
