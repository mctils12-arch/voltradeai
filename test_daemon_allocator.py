"""
KNOWN BROKEN #55 (2026-11-20) — the daemon runs its objects on glibc's
allocator so that freed scan memory goes back to the OS.

Read on production on 2026-11-20 at 16:12-16:22Z: a scan takes the
interpreter from 2.5 M to 4.7-5.2 M allocated blocks (a mid-scan census names
shadow_portfolio._SHADOW_CACHE, the parsed shadow log, as the largest holder),
the blocks come back two minutes after the scan, and resident anonymous
memory moves only from 409 to 385 MB. The interpreter's own allocator returns
an arena only when every block in it is free. The deep-score guard read 456 MB
against a 400 MB trim line on 30 scans in a row.

These tests pin the fix and the break it answers:
- the daemon starts itself again once, in place, with PYTHONMALLOC=malloc,
  and cannot loop;
- the off switch and a PYTHONMALLOC somebody else set are both respected;
- a failed start leaves the daemon running as before;
- health says which allocator is really in use;
- REGRESSION: a shadow-log-sized parse, freed with survivors scattered
  through it, stays resident on the interpreter's allocator and is returned
  on glibc's (the mechanism measured, in two child interpreters);
- END TO END: `python3 voltrade_daemon.py` comes up on the socket, with the
  same PID it was started with, on the requested allocator.
"""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
import time

import pytest

import voltrade_daemon as vd

_spec = importlib.util.spec_from_file_location(
    "session_health_check_allocator",
    os.path.join(os.path.dirname(__file__), "scripts", "session_health_check.py"))
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)

LINUX = sys.platform.startswith("linux")
MARK = vd._ALLOCATOR_REEXEC_MARK


# ── the decision ─────────────────────────────────────────────────────────────

def test_default_is_to_start_again_on_malloc():
    env = {"PATH": "/usr/bin", "ALPACA_KEY": "k"}
    new = vd._allocator_reexec_env(env, platform="linux")
    assert new["PYTHONMALLOC"] == "malloc"
    assert new[MARK] == "1"
    # everything the daemon was given is carried over, and the caller's
    # mapping is not edited
    assert new["PATH"] == "/usr/bin" and new["ALPACA_KEY"] == "k"
    assert "PYTHONMALLOC" not in env and MARK not in env


def test_the_second_start_never_starts_a_third():
    first = vd._allocator_reexec_env({}, platform="linux")
    assert vd._allocator_reexec_env(first, platform="linux") is None
    # and the mark alone is enough, even after PYTHONMALLOC is taken back out
    # of the environment for the daemon's children
    assert vd._allocator_reexec_env({MARK: "1"}, platform="linux") is None


@pytest.mark.parametrize("word", ["pymalloc", "PYMALLOC", " off ", "0", "false", "no", "default"])
def test_the_off_switch(word):
    assert vd._allocator_requested({vd.ALLOCATOR_ENV: word}) == "pymalloc"
    assert vd._allocator_reexec_env({vd.ALLOCATOR_ENV: word}, platform="linux") is None


@pytest.mark.parametrize("word", ["malloc", "MALLOC", "", "glibc", "1"])
def test_anything_else_reads_as_the_default(word):
    assert vd._allocator_requested({vd.ALLOCATOR_ENV: word}) == "malloc"


@pytest.mark.parametrize("chosen", ["malloc", "pymalloc", "debug", "pymalloc_debug"])
def test_a_pythonmalloc_already_set_is_left_alone(chosen):
    assert vd._allocator_reexec_env({"PYTHONMALLOC": chosen}, platform="linux") is None


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_only_on_linux(platform):
    assert vd._allocator_reexec_env({}, platform=platform) is None


# ── the start itself ─────────────────────────────────────────────────────────

def test_it_starts_the_same_interpreter_with_the_same_arguments():
    calls = []
    assert vd._reexec_on_requested_allocator(
        env={"A": "1"}, execve=lambda *a: calls.append(a), platform="linux") is True
    (exe, argv, env), = calls
    assert exe == sys.executable
    assert argv == list(getattr(sys, "orig_argv", [sys.executable] + sys.argv))
    assert env == {"A": "1", "PYTHONMALLOC": "malloc", MARK: "1"}


def test_nothing_is_started_when_there_is_nothing_to_do():
    calls = []
    for env in ({vd.ALLOCATOR_ENV: "pymalloc"}, {MARK: "1"}, {"PYTHONMALLOC": "malloc"}):
        assert vd._reexec_on_requested_allocator(
            env=env, execve=lambda *a: calls.append(a), platform="linux") is False
    assert calls == []


def test_a_failed_start_leaves_the_daemon_running_as_before():
    def refuse(*_a):
        raise OSError(8, "Exec format error")
    assert vd._reexec_on_requested_allocator(env={}, execve=refuse, platform="linux") is False


def test_importing_the_module_starts_nothing():
    """Only `python3 voltrade_daemon.py` may replace the process: an import
    (this test file, bot tooling, the other daemon tests) must not."""
    src = open(vd.__file__).read()
    call = "_reexec_on_requested_allocator()"
    lines = [i for i, l in enumerate(src.splitlines()) if l.strip() == call]
    assert len(lines) == 1
    before = [l for l in src.splitlines()[:lines[0]] if l.strip() and not l.strip().startswith("#")]
    assert before[-1] == 'if __name__ == "__main__":'


# ── what health says ─────────────────────────────────────────────────────────

def test_health_reports_the_allocator_really_in_use():
    h = vd._dispatcher._health({})
    a = h["allocator"]
    assert set(a) == {"in_use", "requested", "restarted_in_place"}
    # this test process: the truth comes from the interpreter, not the request
    expected = "malloc" if os.environ.get("PYTHONMALLOC") == "malloc" else "pymalloc"
    if os.environ.get("PYTHONMALLOC") in (None, "", "malloc", "pymalloc"):
        assert a["in_use"] == expected
    assert a["requested"] == vd._allocator_requested(os.environ)
    assert a["restarted_in_place"] is bool(os.environ.get(MARK))


def test_the_memory_block_is_untouched():
    h = vd._dispatcher._health({})
    assert "allocator" not in h["memory"]


def test_the_health_check_names_the_allocator():
    memory = {"malloc": {"in_use_mb": 190.0, "free_retained_mb": 20.0},
              "python_blocks": 0, "at_start": {"python_blocks": 0}}
    on = hc._memory_makeup(memory, None, {"in_use": "malloc", "requested": "malloc", "restarted_in_place": True})
    assert "objects allocated by glibc" in on
    assert "python blocks 0" not in on        # a 0 that means "not counted" is not printed as a count
    not_on = hc._memory_makeup(dict(memory, python_blocks=2500000), None,
                               {"in_use": "pymalloc", "requested": "malloc", "restarted_in_place": False})
    assert "requested but NOT in use" in not_on and "python blocks 2500000" in not_on
    # switched off, and a daemon older than the key: the line reads as before
    off = hc._memory_makeup(dict(memory, python_blocks=2500000), None,
                            {"in_use": "pymalloc", "requested": "pymalloc", "restarted_in_place": False})
    old = hc._memory_makeup(dict(memory, python_blocks=2500000), None)
    assert off == old and "glibc" not in old and "python blocks 2500000 (start 0)" in old


# ── REGRESSION: the mechanism, measured ──────────────────────────────────────

_SCAN = textwrap.dedent("""
    import ctypes, gc, json, sys
    def rss_mb():
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024
    libc = ctypes.CDLL(None)
    # shaped like a shadow record: mostly small objects (floats, short
    # strings, small dicts), which is what the interpreter's allocator serves
    blob = json.dumps([{"ticker": "T%04d" % (i % 900), "score": i * 0.5,
                        "features": {"f%03d" % j: i * 0.001 + j for j in range(30)},
                        "breakdown": {"b%02d" % j: {"v": i * 0.01 + j, "w": j * 0.5 + i, "why": "r%d_%d" % (i, j)}
                                      for j in range(12)}}
                       for i in range(9000)])
    gc.collect()
    libc.malloc_trim(0)
    start = rss_mb()
    kept = {}
    for scan in range(3):
        records = json.loads(blob)                 # the shadow log, parsed for a scan
        for i in range(0, len(records), 40):       # what outlives the scan: one small object
            kept[(scan % 2, i)] = records[i]["breakdown"]["b00"]   # of every 40th record
        peak = rss_mb()
        del records
        gc.collect()
        libc.malloc_trim(0)                        # bot_engine._release_freed_memory
    print(json.dumps({"start": start, "peak": peak, "after": rss_mb(),
                      "blocks": sys.getallocatedblocks()}))
""")


def _scan_in_child(allocator):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONMALLOC"}
    if allocator == "malloc":
        env["PYTHONMALLOC"] = "malloc"
    out = subprocess.run([sys.executable, "-c", _SCAN], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(not LINUX or not os.path.exists("/proc/self/status"), reason="glibc/Linux only")
def test_a_scan_sized_parse_is_returned_on_malloc_and_held_on_pymalloc():
    held = _scan_in_child("pymalloc")
    freed = _scan_in_child("malloc")
    grew = held["peak"] - held["start"]
    assert grew > 25, held                      # the parse is big enough to tell
    # the break: with one small object in every 40th record still alive (well
    # under 1% of the parse), the interpreter's allocator keeps most of the
    # scan's memory resident
    assert held["after"] - held["start"] > 0.6 * grew, held
    # the fix: glibc gives most of it back
    assert freed["after"] - freed["start"] < 0.4 * (freed["peak"] - freed["start"]), freed
    assert freed["after"] - freed["start"] < 0.5 * (held["after"] - held["start"]), (held, freed)
    # and the reading health uses to tell them apart
    assert held["blocks"] > 0 and freed["blocks"] == 0


# ── END TO END: the daemon really comes up that way ──────────────────────────

def _health_over_socket(path):
    """One line-delimited JSON call, the way server/bot.ts's pythonRpc makes it."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(20)
    s.connect(path)
    s.sendall(b'{"method": "health", "args": {}}\n')
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    doc = json.loads(buf)
    assert doc["status"] == "ok", doc
    return doc["result"]


def _start_daemon(tmp_path, extra_env):
    sock = str(tmp_path / "d.sock")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONMALLOC", MARK, vd.ALLOCATOR_ENV)}
    env.update({"VOLTRADE_DAEMON_SOCKET": sock, "VOLTRADE_AUTOSEED": "false"}, **extra_env)
    proc = subprocess.Popen([sys.executable, os.path.basename(vd.__file__)],
                            cwd=os.path.dirname(os.path.abspath(vd.__file__)), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 90
    while time.time() < deadline:
        assert proc.poll() is None, "daemon exited during startup"
        # the socket file appears at bind(); connect_ex says when it listens
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listening = os.path.exists(sock) and probe.connect_ex(sock) == 0
        probe.close()
        if listening:
            return proc, _health_over_socket(sock)
        time.sleep(0.3)
    proc.kill()
    raise AssertionError("daemon did not answer on its socket within 90 s")


@pytest.mark.skipif(not LINUX, reason="glibc/Linux only")
@pytest.mark.parametrize("extra,in_use,restarted", [
    ({}, "malloc", True),
    ({vd.ALLOCATOR_ENV: "pymalloc"}, "pymalloc", False),
])
def test_the_daemon_comes_up_on_the_requested_allocator(tmp_path, extra, in_use, restarted):
    proc, h = _start_daemon(tmp_path, extra)
    try:
        assert h["alive"] is True
        assert h["allocator"]["in_use"] == in_use
        assert h["allocator"]["restarted_in_place"] is restarted
        # in place: the process the supervisor started IS the daemon
        assert h["pid"] == proc.pid
        if in_use == "malloc":
            assert h["memory"]["python_blocks"] == 0
            # the children it starts are back on the interpreter's default
            environ = open(f"/proc/{proc.pid}/environ", "rb").read().split(b"\0")
            assert b"PYTHONMALLOC=malloc" in environ      # how the interpreter itself was started
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_children_do_not_inherit_the_allocator():
    """After the second start the variable is taken out of os.environ, so a
    Python the daemon starts (the feedback seeder) runs as it did before."""
    code = ("import os, sys; os.environ['%s']='1'; os.environ['PYTHONMALLOC']='malloc'; "
            "sys.path.insert(0, %r); import voltrade_daemon; "
            "print('PYTHONMALLOC' in os.environ)" % (MARK, os.path.dirname(os.path.abspath(vd.__file__))))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONMALLOC", MARK)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False")
