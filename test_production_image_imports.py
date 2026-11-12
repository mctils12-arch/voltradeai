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
"""
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
