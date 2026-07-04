"""Regression guard for the local gate itself (loop-health rule 3).

The 2026-07-04 repair of KNOWN BROKEN #6 found the constitutional gate
(`python3 -m pytest -q`) had been broken at COLLECTION since the repo
import: a root-level script with a test_ prefix sys.exit()'d during
pytest's import phase, so no session could run the full suite. This test
would have caught that the day it happened: it collects the whole repo
in a subprocess and demands a clean exit. Any future file that breaks
collection (import-time sys.exit, syntax error, import error) fails
here with pytest's own diagnostics in the message.
"""

import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))


class TestCollectionHealth(unittest.TestCase):
    def test_full_repo_pytest_collection_succeeds(self):
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True, text=True, timeout=600, cwd=REPO,
        )
        out = (r.stdout or "") + (r.stderr or "")
        self.assertNotIn(
            "INTERNALERROR", out,
            "pytest collection INTERNALERROR — a test_ file is executing "
            "at import (see conftest.py for the standalone-script policy):\n"
            + out[-3000:],
        )
        self.assertEqual(
            r.returncode, 0,
            "full-repo pytest collection must succeed — the local gate is "
            "unrunnable otherwise:\n" + out[-3000:],
        )


if __name__ == "__main__":
    unittest.main()
