"""KNOWN BROKEN #44 (2026-11-06, fixed without touching the FROZEN Dockerfile):
code the server runs must import from what the production image ships.

The Dockerfile's runtime stage copies the repo-root *.py files, strategies/
and alphadesk/ (read from the Dockerfile itself below) and never scripts/.
So the insider_cusum_gate2 diag probe, which imported scripts/
insider_cusum_probe.py, failed live with ModuleNotFoundError from
2026-09-21 on. The module and its by-path loader chain (hurst_exponent_probe
-> permutation_entropy_probe -> critical_slowing_down_probe) now live at the
repo root; scripts/ keeps thin shims for human CLI runs and for the sibling
research scripts that load them by path.

These tests rebuild the image's Python layout from symlinks and run the
probe's own import prelude (read from server/bot.ts) inside it.

RECURRENCE, 2026-11-15 (root-caused 2026-11-16): the earnings_qoq_weekly job's
first registered run died on "python3: can't open file
'/app/scripts/earnings_language_qoq_gate2.py'". The #44 class guard matched
three IMPORT idioms (sys.path, spec_from_file_location, os.path.join) and a
shell command naming a scripts/ file is none of them. The guard below is now
about the image, not about idioms: every Python file a server command names
must exist in the image layout, and the local-import closure of each must
resolve inside it. The job's own command is run in the layout as written.
"""
import ast
import json
import glob
import importlib.util
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
MOVED = ("insider_cusum_probe", "hurst_exponent_probe",
         "permutation_entropy_probe", "critical_slowing_down_probe")


def _runtime_copy_sources():
    """What the Dockerfile's FINAL stage copies from the build context (not
    `COPY --from=builder`), e.g. '*.py', 'strategies/', 'alphadesk/'."""
    lines = open(os.path.join(REPO, "Dockerfile"), encoding="utf-8").read().splitlines()
    last_from = max(i for i, line in enumerate(lines) if line.strip().upper().startswith("FROM "))
    srcs = []
    for line in lines[last_from + 1:]:
        m = re.match(r"\s*COPY\s+(?!--)(\S+)\s+\S+\s*$", line)
        if m:
            srcs.append(m.group(1))
    return srcs


def _image_layout(dest):
    """Symlink what the runtime stage copies into dest (the image's /app)."""
    for src in _runtime_copy_sources():
        if src.endswith("/"):
            path = os.path.join(REPO, src.rstrip("/"))
            if os.path.isdir(path):
                os.symlink(path, os.path.join(dest, os.path.basename(path)))
        elif any(ch in src for ch in "*?["):
            for path in glob.glob(os.path.join(REPO, src)):
                os.symlink(path, os.path.join(dest, os.path.basename(path)))
        elif os.path.exists(os.path.join(REPO, src)):
            os.symlink(os.path.join(REPO, src), os.path.join(dest, os.path.basename(src)))


def _probe_program():
    """The insider_cusum_gate2 probe's python3 -c program, as bot.ts runs it."""
    bot = open(os.path.join(REPO, "server", "bot.ts"), encoding="utf-8").read()
    start = bot.index('case "insider_cusum_gate2"')
    m = re.search(r'python3 -c "\n(.*?)\n"`', bot[start:start + 4000], re.S)
    assert m, "insider_cusum_gate2's python3 -c program not found in server/bot.ts"
    return m.group(1)


def test_runtime_stage_ships_root_python_files():
    srcs = _runtime_copy_sources()
    assert "*.py" in srcs, srcs
    assert "strategies/" in srcs and "alphadesk/" in srcs, srcs


def test_insider_cusum_gate2_imports_in_the_production_image_layout(tmp_path):
    prog = _probe_program()
    assert "import insider_cusum_probe" in prog
    # run_probe() itself needs the volume's Form 4 archive and live bars; every
    # other line of the server's program runs as written, then each module's
    # own by-path loader is exercised in the same layout.
    prelude = [line for line in prog.splitlines() if "run_probe()" not in line]
    check = prelude + [
        "import importlib.util as _ilu, os as _os",
        "_hep = insider_cusum_probe._load_hurst_probe()",
        "_hep.tertile_welch([(float(i), float(i % 7)) for i in range(60)])",
        "_spec = _ilu.spec_from_file_location('permutation_entropy_probe', "
        "_os.path.join(_os.path.dirname(_hep.__file__), 'permutation_entropy_probe.py'))",
        "_pep = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_pep)",
        "_csd = _pep._load_csd_module()",
        "print('FILES', insider_cusum_probe.__file__, _hep.__file__, _pep.__file__, _csd.__file__)",
    ]
    _image_layout(str(tmp_path))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", "\n".join(check)], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"the probe's imports fail in the image layout:\n{r.stderr[-2000:]}"
    files = r.stdout.split("FILES", 1)[1].split()
    assert len(files) == 4
    for f in files:
        assert os.path.dirname(os.path.abspath(f)) == str(tmp_path), f"{f} did not load from the image layout"


def _code_hits(path, pattern):
    """Matches in code, not prose: comment-only lines (// or * in TypeScript,
    # in Python) are blanked first, keeping line numbers."""
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    marks = ("#",) if path.endswith(".py") else ("//", "*", "/*")
    text = "\n".join("" if line.lstrip().startswith(marks) else line for line in lines)
    return [f"{os.path.relpath(path, REPO)}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)[:100]!r}"
            for m in pattern.finditer(text)]


def test_no_runtime_code_loads_from_scripts():
    """The class guard: no server-side TypeScript (its python3 -c programs) and
    no Python the image ships may put scripts/ on sys.path or load a file
    from it. A new scripts/-backed probe would fail live the same way."""
    pattern = re.compile(
        r"""sys\.path\.(?:insert|append)\([^)]*['"][^'"]*\bscripts\b"""
        r"""|spec_from_file_location\([^)]*\bscripts\b"""
        r"""|os\.path\.join\([^)]*['"]scripts['"]""")
    hits = []
    for top in ("server", "shared"):
        for root, _dirs, files in os.walk(os.path.join(REPO, top)):
            for name in files:
                if name.endswith(".ts") and not name.endswith(".test.ts"):
                    hits += _code_hits(os.path.join(root, name), pattern)
    for path in glob.glob(os.path.join(REPO, "*.py")):
        if not os.path.basename(path).startswith("test_"):
            hits += _code_hits(path, pattern)
    for pkg in ("strategies", "alphadesk"):
        for root, _dirs, files in os.walk(os.path.join(REPO, pkg)):
            for name in files:
                if name.endswith(".py") and not name.startswith("test_"):
                    hits += _code_hits(os.path.join(root, name), pattern)
    assert hits == [], "runtime code loads from scripts/, which the image does not ship:\n" + "\n".join(hits)


@pytest.mark.parametrize("name", MOVED)
def test_scripts_shim_serves_the_root_module(name):
    root_file = os.path.join(REPO, name + ".py")
    shim_file = os.path.join(REPO, "scripts", name + ".py")
    assert os.path.isfile(root_file), f"{name}.py must live at the repo root (the image ships root *.py)"
    spec = importlib.util.spec_from_file_location(name, shim_file)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert shim.run_probe.__code__.co_filename == root_file, "the shim must serve the root module's code"
    assert "def run_probe" not in open(shim_file, encoding="utf-8").read(), "the shim holds no logic"


@pytest.mark.parametrize("name", MOVED)
def test_scripts_cli_entry_point_still_works(name):
    r = subprocess.run([sys.executable, os.path.join("scripts", name + ".py"), "--help"],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage:" in r.stdout


# ── 2026-11-16: the guard is about the image, not about import idioms ────────

EARNINGS_MOVED = (("earnings_language_qoq_gate2", "main"),
                  ("earnings_language_gate2", "tone_score"),
                  ("earnings_language_gate2_robustness", "horizon_report"))
# `python3 x.py`, `python3 -u x.py`, and argv forms like ["python3", "x.py"].
_PY_CMD = re.compile(r"""\bpython3?["'`,\s]+(?:-u["'`,\s]+)?["'`]?([A-Za-z0-9_./-]+\.py)\b""")


def _server_python_commands():
    """{python file named by a server command: [where]} over server/ and shared/
    TypeScript code (comment-only lines and *.test.ts excluded)."""
    found = {}
    for top in ("server", "shared"):
        for root, _dirs, files in os.walk(os.path.join(REPO, top)):
            for name in files:
                if not name.endswith(".ts") or name.endswith(".test.ts"):
                    continue
                path = os.path.join(root, name)
                lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
                for i, line in enumerate(lines, 1):
                    if line.lstrip().startswith(("//", "*", "/*")):
                        continue
                    for m in _PY_CMD.finditer(line):
                        found.setdefault(m.group(1), []).append(f"{os.path.relpath(path, REPO)}:{i}")
    return found


def _repo_python_modules_outside(layout):
    """Top-level module names that exist as <name>.py (or a package dir) in the
    repo but NOT in the image layout — what an import would silently find on a
    developer machine and miss in production."""
    inside, outside = set(), {}
    skip = {"node_modules", ".git", "dist", "__pycache__", ".bt_cache", ".venv", "venv"}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, f), REPO)
            # shipped anywhere in the image (root *.py, strategies/, alphadesk/ …)
            if os.path.exists(os.path.join(layout, rel)):
                inside.add(f[:-3])
            else:
                outside.setdefault(f[:-3], rel)
    return {n: p for n, p in outside.items()
            if n not in inside and not os.path.isdir(os.path.join(layout, n))}


def _local_imports(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_every_python_file_a_server_command_names_is_in_the_image(tmp_path):
    cmds = _server_python_commands()
    # non-vacuous: the scan must see the long-job lane's commands it exists for
    for known in ("walkforward.py", "lazy_prices_run.py", "edge_p2_insider.py",
                  "event_reaction_panel.py", "earnings_language_qoq_gate2.py"):
        assert known in cmds, f"the command scan no longer finds {known}: {sorted(cmds)}"
    _image_layout(str(tmp_path))
    missing = {f: where for f, where in cmds.items() if not os.path.isfile(os.path.join(str(tmp_path), f))}
    assert missing == {}, ("a server command names a Python file the production image does not ship "
                           f"(runtime stage copies {_runtime_copy_sources()}): {missing}")


def test_server_run_python_import_closure_resolves_in_the_image(tmp_path):
    """Walk the imports of every file a server command runs, through the image
    layout only. An import whose module exists in the repo but outside the
    layout (scripts/, datacore/, …) works on a laptop and fails in production."""
    layout = str(tmp_path)
    _image_layout(layout)
    outside = _repo_python_modules_outside(layout)
    todo = [f[:-3] for f in _server_python_commands() if "/" not in f]
    seen, bad = set(), []
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        path = os.path.join(layout, name + ".py")
        if not os.path.isfile(path):
            continue
        for imp in sorted(_local_imports(path)):
            if imp in outside:
                bad.append(f"{name}.py imports {imp} -> only at {outside[imp]}")
            elif os.path.isfile(os.path.join(layout, imp + ".py")):
                todo.append(imp)
    assert len(seen) > 20, f"closure walk too small to mean anything: {sorted(seen)}"
    assert bad == [], "server-run Python imports modules the image does not ship:\n" + "\n".join(bad)


def test_earnings_qoq_job_command_runs_in_the_production_image_layout(tmp_path):
    """REGRESSION 2026-11-15: the job's command, read from server/earningsQoq.ts
    and run as written with cwd = the image layout. On main it exits 2 with
    "can't open file"; here it must reach its own JSON summary line."""
    src = open(os.path.join(REPO, "server", "earningsQoq.ts"), encoding="utf-8").read()
    m = re.search(r"return `(python3 [^`]+)`;", src)
    assert m, "earningsQoqCmd's command template not found"
    cmd = m.group(1).replace("${EARNINGS_QOQ_MAX_RSS_MB}", "900").replace("${v}", "0.0.0-test")
    assert "${" not in cmd, cmd
    layout, data = tmp_path / "app", tmp_path / "data"
    layout.mkdir(); data.mkdir()
    _image_layout(str(layout))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DATA_DIR"] = str(data)
    argv = [sys.executable] + cmd.split()[1:]
    r = subprocess.run(argv, cwd=str(layout), env=env, capture_output=True, text=True, timeout=300)
    assert "can't open file" not in r.stderr and "ModuleNotFoundError" not in r.stderr, r.stderr[-2000:]
    last = (r.stdout.strip().split("\n") or [""])[-1]
    summary = json.loads(last)  # the one-line summary parseEarningsQoqSummary reads
    # an empty volume has no earnings8k archive: the honest outcome is "skipped"
    assert summary.get("status") == "skipped", summary
    assert "earnings8k" in str(summary.get("reason")), summary


def test_filings13f_clusters_job_command_runs_in_the_production_image_layout(tmp_path):
    """The nightly 13F cluster job's command (server/filings13fClusters.ts), run
    as written with cwd = the image layout (the class of the 2026-11-15 break)."""
    src = open(os.path.join(REPO, "server", "filings13fClusters.ts"), encoding="utf-8").read()
    m = re.search(r"return `(python3 [^`]+)`;", src)
    assert m, "filings13fClustersCmd's command template not found"
    cmd = m.group(1).replace("${FILINGS13F_CLUSTERS_MAX_RSS_MB}", "700").replace("${v}", "0.0.0-test")
    assert "${" not in cmd, cmd
    assert cmd.split()[1] in _server_python_commands(), "the command scan must see this job's file"
    layout, data = tmp_path / "app", tmp_path / "data"
    layout.mkdir(); data.mkdir()
    _image_layout(str(layout))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DATA_DIR"] = str(data)
    argv = [sys.executable] + cmd.split()[1:]
    r = subprocess.run(argv, cwd=str(layout), env=env, capture_output=True, text=True, timeout=300)
    assert "can't open file" not in r.stderr and "ModuleNotFoundError" not in r.stderr, r.stderr[-2000:]
    summary = json.loads((r.stdout.strip().split("\n") or [""])[-1])
    # an empty volume has no 13F archive: the honest outcome is "skipped"
    assert summary.get("status") == "skipped", summary
    assert "filings13f" in str(summary.get("reason")), summary


@pytest.mark.parametrize("name,attr", EARNINGS_MOVED)
def test_earnings_language_scripts_shim_serves_the_root_module(name, attr):
    root_file = os.path.join(REPO, name + ".py")
    shim_file = os.path.join(REPO, "scripts", name + ".py")
    assert os.path.isfile(root_file), f"{name}.py must live at the repo root (the image ships root *.py)"
    spec = importlib.util.spec_from_file_location(name, shim_file)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert getattr(shim, attr).__code__.co_filename == root_file, "the shim must serve the root module's code"
    assert "\ndef " not in open(shim_file, encoding="utf-8").read(), "the shim holds no logic"


def test_snapshot_refresh_job_and_its_four_refreshers_import_in_the_production_image_layout(tmp_path):
    """The nightly snapshot refresh (server/snapshotRefresh.ts, 2026-11-25): its
    command names a root file, and the four refreshers it loads BY NAME at run
    time (importlib, which the static import-closure test above cannot see)
    are in the image too. No network: an unknown --only stops before any fetch."""
    src = open(os.path.join(REPO, "server", "snapshotRefresh.ts"), encoding="utf-8").read()
    m = re.search(r"return `(python3 [^`]+)`;", src)
    assert m, "snapshotRefreshCmd's command template not found"
    cmd = m.group(1).replace("${SNAPSHOT_REFRESH_MAX_RSS_MB}", "600").replace("${v}", "0.0.0-test")
    assert "${" not in cmd, cmd
    assert cmd.split()[1] in _server_python_commands(), "the command scan must see this job's file"
    layout, data = tmp_path / "app", tmp_path / "data"
    layout.mkdir(); data.mkdir()
    _image_layout(str(layout))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DATA_DIR"] = str(data)
    r = subprocess.run([sys.executable] + cmd.split()[1:] + ["--only", "not-a-source"], cwd=str(layout), env=env,
                       capture_output=True, text=True, timeout=120)
    assert "can't open file" not in r.stderr and "ModuleNotFoundError" not in r.stderr, r.stderr[-2000:]
    assert json.loads((r.stdout.strip().split("\n") or [""])[-1]).get("status") == "error"
    prog = ("import importlib, snapshot_refresh_job as j\n"
            "mods = [importlib.import_module(s['module']) for s in j.SOURCES]\n"
            "assert all(callable(m.main) and m.OUT for m in mods)\n"
            "print(len(mods))")
    r = subprocess.run([sys.executable, "-c", prog], cwd=str(layout), env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "4", r.stderr[-2000:]


def test_dtcc_trs_panel_job_command_runs_in_the_production_image_layout(tmp_path):
    """The DTCC equity-swap panel job (server/dtccTrsPanel.ts, 2026-11-26): its
    command names a root file and the module it imports (dtcc_trs_panel, which
    reads market_calendar.py beside it) is in the image. No network:
    --max-days 0 stops before the first vendor read."""
    src = open(os.path.join(REPO, "server", "dtccTrsPanel.ts"), encoding="utf-8").read()
    m = re.search(r"return `(python3 [^`]+)`;", src)
    assert m, "dtccTrsPanelCmd's command template not found"
    cmd = (m.group(1).replace("${DTCC_TRS_PANEL_MAX_RSS_MB}", "700")
           .replace("${DTCC_TRS_PANEL_BUDGET_S}", "600").replace("${v}", "0.0.0-test"))
    assert "${" not in cmd, cmd
    assert cmd.split()[1] in _server_python_commands(), "the command scan must see this job's file"
    layout, data = tmp_path / "app", tmp_path / "data"
    layout.mkdir(); data.mkdir()
    _image_layout(str(layout))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DATA_DIR"] = str(data)
    env["TMPDIR"] = str(tmp_path)
    r = subprocess.run([sys.executable] + cmd.split()[1:] + ["--max-days", "0"], cwd=str(layout), env=env,
                       capture_output=True, text=True, timeout=120)
    assert "can't open file" not in r.stderr and "ModuleNotFoundError" not in r.stderr, r.stderr[-2000:]
    summary = json.loads((r.stdout.strip().split("\n") or [""])[-1])
    assert summary.get("status") == "ok" and summary.get("stopped") == "day limit", summary
    assert summary.get("read") == 0 and summary.get("days_ok") == 0 and summary.get("backlog", 0) > 0, summary
    assert os.path.isfile(data / "dtcc_trs_panel" / "last_run.json")
    prog = "import dtcc_trs_panel as p\nassert len(p.nyse_closures()) > 20\nprint('ok')"
    r = subprocess.run([sys.executable, "-c", prog], cwd=str(layout), env=env, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == "ok", r.stderr[-2000:]
