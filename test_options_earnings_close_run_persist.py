"""REPAIR 2026-11-12 — EARNINGS CLOSE: the halted subprocess fallback no longer
repeats the backfill row.

What was wrong: the backfill row ("earnings_backfill": every sold option's
next report and close session) is written on the first pass after a process
start, remembered in module memory (options_manager._EARNINGS_RUN). When the
daemon does not answer, bot.ts's halted exits-only pass runs the manager in a
fresh `python3 -c` process per call (pythonCall's subprocess fallback). Every
call was then a process start, so the row repeated on every tick
(research/open_questions.md, 2026-11-05 follow-ups, "[EARNINGS CLOSE]
Residuals").

Pinned:
- two real `python3 -c` processes on one ET day: the first writes the row, the
  second does not, and both judge the positions the same way;
- a one-shot process keeps the ET day of its backfill pass in
  EARNINGS_RUN_PATH (storage_config owns the path), written atomically. The
  next ET day runs it again. A first pass with no sold option uses the day up,
  as it uses up a long-lived process's flag;
- a missing, corrupt or malformed file means "not run yet"; an unwritable one
  is logged and nothing raises;
- a long-lived process (the daemon, a test run) behaves exactly as before: it
  backfills on its first pass whatever the file says, and never reads or
  writes it;
- an inert pass (no FINNHUB_KEY, flag off) writes nothing;
- bot.ts's fallback is still a `python3 -c` one-liner, which is what marks a
  one-shot process.
Mocked like test_options_earnings_close.py: no network, the New York clock
fixed at Thursday 2026-11-05 10:30 unless a test moves it.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402
import options_scanner  # noqa: E402
import storage_config  # noqa: E402
import system_config  # noqa: E402
from test_options_earnings_close import _EarningsCase, csp, long_put  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
ET = ZoneInfo("America/New_York")
THU, FRI = "2026-11-05", "2026-11-06"
REPORTS = {"GOOGL": [("2026-11-12", "bmo")]}      # next week: scheduled, nothing is due


def book():
    """Two sold puts no rule fires on: GOOGL reports next week, AAL does not report."""
    return [csp("GOOGL", 40, 250), csp("AAL", 29, 13)]


class _MarkerCase(_EarningsCase):
    """_EarningsCase with the run marker in a temp dir and the process's
    backfill flag as a new process has it (not yet run)."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.marker = os.path.join(tmp.name, "voltrade_options_earnings_run.json")
        p = patch.object(om, "EARNINGS_RUN_PATH", self.marker, create=True)
        p.start()
        self.addCleanup(p.stop)
        om._EARNINGS_RUN["backfill_done"] = False

    def one_shot(self, on=True):
        return patch.object(om, "_one_shot_process", return_value=on, create=True)

    def fresh_process(self, legs=None, **broker_kw):
        """The next process's first pass: module memory starts empty."""
        om._EARNINGS_RUN["backfill_done"] = False
        om._EARNINGS_REPORT_CACHE.clear()
        return self.run_book(book() if legs is None else legs, **{"reports": REPORTS, **broker_kw})

    def backfills(self, result):
        return self.of_type(result, "earnings_backfill")

    def read_marker(self):
        with open(self.marker) as f:
            return json.load(f)

    def tick_report(self, marker_path, now):
        """one_shot_tick's body: this process's first pass, as JSON-able facts."""
        self.now = now
        om._EARNINGS_RUN["backfill_done"] = False     # what a new process starts with
        with patch.object(om, "EARNINGS_RUN_PATH", marker_path, create=True):
            result = self.run_book(book(), reports=REPORTS)
        return {"argv0": sys.argv[0], "submits": self.submits,
                "types": [a.get("type") for a in result["actions"]],
                "backfill": [{r["underlying"]: r["status"] for r in a["legs"]} for a in self.backfills(result)]}


def one_shot_tick(marker_path: str, now_iso: str) -> None:
    """Run by TestTwoRealOneShotProcesses in a real `python3 -c` process: one
    manage tick over book(), printed as one JSON line."""
    case = _MarkerCase("tick_report")
    case.setUp()
    try:
        print(json.dumps(case.tick_report(marker_path, datetime.fromisoformat(now_iso))))
    finally:
        case.doCleanups()


class TestTwoRealOneShotProcesses(unittest.TestCase):
    """The defect itself: bot.ts's fallback is `python3 -c "... manage_options_positions(...)"`."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.marker = os.path.join(tmp.name, "voltrade_options_earnings_run.json")

    def process(self, now_iso):
        code = f"import test_options_earnings_close_run_persist as t; t.one_shot_tick({self.marker!r}, {now_iso!r})"
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_second_process_of_the_et_day_does_not_repeat_the_row(self):
        first = self.process("2026-11-05T10:30:00-05:00")
        second = self.process("2026-11-05T10:33:00-05:00")
        self.assertEqual((first["argv0"], second["argv0"]), ("-c", "-c"), "real one-shot processes")
        self.assertEqual(first["backfill"], [{"GOOGL": "scheduled", "AAL": "none_before_expiry"}])
        self.assertEqual(second["backfill"], [], "the row is not repeated on the next tick")
        self.assertEqual([t for t in first["types"] if t != "earnings_backfill"], second["types"],
                         "everything else the pass does is the same")
        self.assertEqual((first["submits"], second["submits"]), ([], []))
        with open(self.marker) as f:
            self.assertEqual(json.load(f), {"backfill_day": THU})


class TestOneShotMarker(_MarkerCase):
    def test_the_row_once_per_et_day_and_again_the_next_day(self):
        with self.one_shot():
            first = self.fresh_process()
            self.assertEqual(len(self.backfills(first)), 1)
            self.assertEqual(self.read_marker(), {"backfill_day": THU})
            self.assertEqual(os.listdir(os.path.dirname(self.marker)), [os.path.basename(self.marker)],
                             "written through a temp file that is gone")
            for _ in range(3):
                self.assertEqual(self.backfills(self.fresh_process()), [], "every later tick that day is silent")
            self.now = datetime(2026, 11, 6, 10, 30, tzinfo=ET)
            self.assertEqual(len(self.backfills(self.fresh_process())), 1, "a new ET day runs it again")
            self.assertEqual(self.read_marker(), {"backfill_day": FRI})
            self.assertEqual(self.backfills(self.fresh_process()), [])

    def test_a_first_pass_with_no_sold_option_uses_the_day_up(self):
        with self.one_shot():
            self.assertEqual(self.backfills(self.fresh_process([long_put()])), [])
            self.assertEqual(self.read_marker(), {"backfill_day": THU})
            self.assertEqual(self.backfills(self.fresh_process()), [], "as in a long-lived process: no later pass backfills")

    def test_a_missing_corrupt_or_malformed_file_means_not_run_yet(self):
        for label, content in (("missing", None), ("empty", ""), ("not JSON", "{\"backfill_day\": \"2026-1"),
                               ("a list", "[\"2026-11-05\"]"), ("another key", "{\"day\": \"2026-11-05\"}"),
                               ("not a string", "{\"backfill_day\": 20261105}"), ("null", "null"),
                               ("yesterday", "{\"backfill_day\": \"2026-11-04\"}"), ("bytes", b"\xff\xfe\x00")):
            with self.subTest(label), self.one_shot():
                if os.path.exists(self.marker):
                    os.remove(self.marker)
                if content is not None:
                    with open(self.marker, "wb") as f:
                        f.write(content if isinstance(content, bytes) else content.encode())
                self.assertEqual(len(self.backfills(self.fresh_process())), 1)
                self.assertEqual(self.read_marker(), {"backfill_day": THU}, "and the file is whole again")

    def test_an_unwritable_marker_never_raises_and_the_row_is_still_written(self):
        os.mkdir(self.marker)                           # a directory where the file should be: no read, no replace
        with self.one_shot():
            for _ in range(2):
                result = self.fresh_process()
                self.assertEqual(len(self.backfills(result)), 1, "nothing remembered: the row repeats, as before")
                self.assertNotIn("error", result)
        self.assertTrue(os.path.isdir(self.marker))

    def test_an_inert_pass_writes_nothing(self):
        for label, off in (("no FINNHUB_KEY", patch.object(options_scanner, "FINNHUB_KEY", "")),
                           ("flag off", patch.dict(system_config.BASE_CONFIG, {"OPTIONS_EARNINGS_CLOSE_ENABLED": False}))):
            with self.subTest(label), off, self.one_shot():
                self.assertEqual(self.backfills(self.fresh_process()), [])
                self.assertFalse(os.path.exists(self.marker))
        self.assertFalse(om._EARNINGS_RUN["backfill_done"], "an inert pass does not use up the backfill")


class TestLongLivedProcessIsUnchanged(_MarkerCase):
    """The daemon: its in-memory flag is the whole rule, the file is not its business."""

    def test_it_is_not_a_one_shot_process(self):
        for argv, one_shot in ((["-c"], True), (["voltrade_daemon.py"], False), (["bot_engine.py", "full"], False),
                               (["/usr/bin/pytest", "-q"], False), ([""], False), ([], False)):
            with self.subTest(argv=argv), patch.object(sys, "argv", argv):
                self.assertIs(om._one_shot_process(), one_shot)

    def test_a_restart_backfills_whatever_the_file_says_and_leaves_it_alone(self):
        with open(self.marker, "w") as f:
            json.dump({"backfill_day": THU}, f)         # a one-shot process already ran today
        before = os.stat(self.marker).st_mtime_ns
        for _ in range(2):                              # two daemon starts on that day
            result = self.fresh_process()
            self.assertEqual(len(self.backfills(result)), 1, "the first pass after a process start, as before")
            self.assertEqual(self.backfills(self.run_again()), [], "and only the first")
        self.assertEqual((self.read_marker(), os.stat(self.marker).st_mtime_ns), ({"backfill_day": THU}, before))

    def test_it_never_creates_the_file(self):
        self.assertEqual(len(self.backfills(self.fresh_process())), 1)
        self.assertFalse(os.path.exists(self.marker))
        self.assertEqual(os.listdir(os.path.dirname(self.marker)), [])


class TestWiring(unittest.TestCase):
    def test_storage_config_owns_the_path(self):
        self.assertEqual(storage_config.OPTIONS_EARNINGS_RUN_PATH,
                         os.path.join(storage_config.DATA_DIR, "voltrade_options_earnings_run.json"))
        self.assertEqual(om.EARNINGS_RUN_PATH, storage_config.OPTIONS_EARNINGS_RUN_PATH)

    def test_the_halted_fallback_is_still_a_python_c_one_liner(self):
        # _one_shot_process() recognises the fallback by `python3 -c`: if this command changes shape, so must it
        with open(os.path.join(REPO, "server", "bot.ts")) as f:
            bot = f.read()
        self.assertRegex(bot, re.compile(
            r'"manage_options_exits",\s*\{ allow_rolls: false \},\s*'
            r'`python3 -c "import json; from options_manager import manage_options_positions; '
            r'print\(json\.dumps\(manage_options_positions\(allow_rolls=False\), default=str\)\)"`'))


if __name__ == "__main__":
    unittest.main()
