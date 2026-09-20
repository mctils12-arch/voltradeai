"""test_spinout_boundary.py — mechanical enforcement of the SPINOUT-READY
DATA LAYER standing behavior (CLAUDE.md KNOWN STATE, human-approved
2026-07-03; restated as datacore/README.md's "Boundary rules
(non-negotiable)").

That rule has stood on prose alone since 2026-07-03: dozens of sessions
have cited it, none had mechanically checked it. The RENDERING & MOTION
LAW section of CLAUDE.md already states the general principle this file
applies here too — "Prose in CLAUDE.md is not enforcement... assertions
are part of the harness so CI blocks regressions rather than a human
noticing them later." This is that assertion for the data-layer boundary.

TWO DIRECTIONS, BOTH CHECKED (the rule is two-sided even though
datacore/README.md's numbered list reads as one):

1. "Nothing in datacore/ imports from or knows about trading logic
   (bot_engine.py, system_config.py, strategies/, server/bot.ts)."
   Checked here as: no server/*.ts module other than server/bot.ts itself
   imports FROM server/bot.ts. In practice every data/pipeline module in
   this repo lives directly in server/*.ts (datacore/README.md's own
   "Layout" section documents this — the datacore/ directory itself holds
   mostly static JSON/manifests, not code). server/routes.ts is the one
   legitimate exception: it is the HTTP wiring layer that must reference
   bot.ts's exports to mount routes, not a data pipeline. That single
   exception is an explicit, reviewable allowlist below, not a wildcard —
   a second file starting to import bot.ts fails this test loudly, the
   same discipline dup_precise_literal/conflicting_const already use
   elsewhere in this repo's counter-ratchet.

2. "Signals and overlay data are exposed exclusively through the
   /api/data/* routes... the bot consumes signals the same way an
   external API customer would." Checked here as: none of the live
   trading-logic Python files (CLAUDE.md's own CODEBASE MAP / MUTABLE
   list: bot_engine.py, system_config.py, risk_kill_switch.py,
   ml_model_v2.py, tiered_strategy.py, analyze.py, insights.py,
   instrument_selector.py, strategies/*.py) reference `datacore/` at all
   — confirmed by direct grep before writing this test (zero hits), so
   this locks in a real, currently-true invariant rather than asserting
   a strawman. scripts/*_gate1.ts / *_gate2.py ladder-testing scripts are
   deliberately OUT of scope: CLAUDE.md's own ROOT VALIDATION LADDER
   expects those to read datacore/ directly to grade a root before it
   ever reaches the bot — that is gate testing, not the live trading
   path, and gating it here would be testing the wrong thing.

Both checks are static source scans (no network, no execution of the
scanned files) — CI-safe, fast, and immune to the "modules exist" partial
credit ANALYST CONSOLE's own audits have flagged elsewhere: this counts
actual import/reference strings, not directory listings.
"""
import glob
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Direction 1: server/*.ts modules that MAY import server/bot.ts.
# routes.ts is the HTTP wiring layer (the API boundary itself), not a data
# pipeline module — it legitimately needs bot.ts's exports to mount routes.
ALLOWED_BOT_IMPORTERS = {"routes.ts"}

BOT_IMPORT_RE = re.compile(
    r"""(?:from\s+|require\()\s*['"]\./bot(?:\.js)?['"]"""
)

# Direction 2: the live trading-logic files (CLAUDE.md CODEBASE MAP /
# MUTABLE section) that must never reference datacore/ directly.
TRADING_LOGIC_FILES = [
    "bot_engine.py",
    "system_config.py",
    "risk_kill_switch.py",
    "ml_model_v2.py",
    "tiered_strategy.py",
    "analyze.py",
    "insights.py",
    "instrument_selector.py",
]

DATACORE_REF_RE = re.compile(r"datacore")


class TestSpinoutBoundaryDirection1(unittest.TestCase):
    """No server/*.ts data/pipeline module imports server/bot.ts, except
    the one reviewed allowlist entry above."""

    def test_only_allowed_files_import_bot_ts(self):
        server_dir = os.path.join(REPO_ROOT, "server")
        importers = set()
        for path in sorted(glob.glob(os.path.join(server_dir, "*.ts"))):
            name = os.path.basename(path)
            if name == "bot.ts":
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            if BOT_IMPORT_RE.search(content):
                importers.add(name)

        self.assertEqual(
            importers,
            ALLOWED_BOT_IMPORTERS,
            "A server/*.ts file outside the reviewed allowlist now imports "
            "server/bot.ts, violating the SPINOUT-READY DATA LAYER rule "
            "('nothing in datacore/ imports from or knows about trading "
            "logic'). If this is a genuine new HTTP-wiring-layer file, add "
            "it to ALLOWED_BOT_IMPORTERS deliberately in the same PR that "
            "adds the import; if it is a data/pipeline module reaching "
            "into the orchestrator, that is the violation this test exists "
            "to catch.",
        )


class TestSpinoutBoundaryDirection2(unittest.TestCase):
    """No live trading-logic Python file reads datacore/ directly,
    bypassing the /api/data/* boundary and ladder-gate labeling."""

    def test_trading_logic_files_do_not_reference_datacore(self):
        offenders = {}
        for rel_path in TRADING_LOGIC_FILES:
            abs_path = os.path.join(REPO_ROOT, rel_path)
            if not os.path.exists(abs_path):
                continue
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            hits = DATACORE_REF_RE.findall(content)
            if hits:
                offenders[rel_path] = len(hits)

        self.assertEqual(
            offenders,
            {},
            "One or more live trading-logic files reference 'datacore' "
            "directly, violating the SPINOUT-READY DATA LAYER rule "
            "(signals must be exposed only through the /api/data/* API "
            "boundary — 'the bot consumes signals the same way an "
            "external API customer would'). If this is a genuine new "
            "signal integration, it should read the signal via the same "
            "route/RPC boundary every other consumer uses, not the raw "
            "datacore/ file.",
        )

    def test_strategies_directory_does_not_reference_datacore(self):
        strategies_dir = os.path.join(REPO_ROOT, "strategies")
        offenders = {}
        for path in sorted(glob.glob(os.path.join(strategies_dir, "*.py"))):
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            hits = DATACORE_REF_RE.findall(content)
            if hits:
                offenders[os.path.relpath(path, REPO_ROOT)] = len(hits)

        self.assertEqual(
            offenders,
            {},
            "One or more strategies/*.py modules reference 'datacore' "
            "directly — see test_trading_logic_files_do_not_reference_"
            "datacore's docstring for the rule this enforces.",
        )


if __name__ == "__main__":
    unittest.main()
