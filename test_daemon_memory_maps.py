"""
KNOWN BROKEN #55 (2026-11-19) — what the daemon's resident memory is made of.

The census (memory_census.py) was read on production on 2026-11-19 at 20:19Z,
19.4 hours and 115 scans into the daemon's life: every named Python holder
together came to about 30 MB (vol_surface._surface_cache 18.5 MB,
options_scanner._chain_cache 6.3 MB) in a 470 MB process, and glibc counted
62 MB of C heap in use. About 380 MB had no reader, so no cache bound could
bring the daemon under the deep-score guard's 400 MB trim line and nothing
said what could.

These tests pin the reader that covers the rest: the daemon's health RPC
reports resident memory by kind of mapping from the kernel's smaps, the
kinds add up to the process's RSS, and the session health check prints it.
"""
import importlib.util
import os

import pytest

import voltrade_daemon as vd

_spec = importlib.util.spec_from_file_location(
    "session_health_check_maps",
    os.path.join(os.path.dirname(__file__), "scripts", "session_health_check.py"))
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


def _mapping(header, rss_kb):
    return (f"{header}\n"
            f"Size:               {rss_kb * 2} kB\n"
            f"KernelPageSize:        4 kB\n"
            f"Rss:                {rss_kb} kB\n"
            f"Pss:                {rss_kb} kB\n"
            f"Anonymous:             0 kB\n"
            f"VmFlags: rd wr mr mw me ac\n")


SMAPS = "".join([
    _mapping("55d0c0000000-55d0c0400000 r-xp 00000000 08:01 1234    /usr/local/bin/python3.11", 2048),
    _mapping("55d0c1000000-55d0c9000000 rw-p 00000000 00:00 0       [heap]", 66560),
    _mapping("7f0000000000-7f0004000000 rw-p 00000000 00:00 0 ", 204800),
    _mapping("7f0004000000-7f0008000000 ---p 00000000 00:00 0", 0),
    _mapping("7f0010000000-7f0010100000 rw-p 00000000 00:00 0", 1024),
    _mapping("7f0020000000-7f0022000000 r-xp 00000000 08:01 99      "
             "/usr/lib/python3/site-packages/scipy.libs/libscipy_openblas64_-ff651d7f.so", 30720),
    _mapping("7f0022000000-7f0022400000 rw-p 01f00000 08:01 99      "
             "/usr/lib/python3/site-packages/scipy.libs/libscipy_openblas64_-ff651d7f.so", 1024),
    _mapping("7f0030000000-7f0030800000 r-xp 00000000 08:01 77      "
             "/usr/lib/python3/site-packages/numpy/_core/_multiarray_umath.cpython-311-x86_64-linux-gnu.so", 4096),
    _mapping("7f0031000000-7f0031001000 r--p 00000000 08:01 78      /usr/lib/x86_64-linux-gnu/libtiny.so.1", 4),
    _mapping("7f0040000000-7f0041000000 r--s 00000000 08:01 55      /data/voltrade/model.bin", 8192),
    _mapping("7ffc00000000-7ffc00021000 rw-p 00000000 00:00 0       [stack]", 132),
    _mapping("7ffc00100000-7ffc00102000 r-xp 00000000 00:00 0       [vdso]", 8),
])


@pytest.fixture
def smaps_file(tmp_path):
    p = tmp_path / "smaps"
    p.write_text(SMAPS)
    return str(p)


def test_each_mapping_is_counted_under_one_kind(smaps_file):
    m = vd._maps_snapshot(smaps_file)
    assert m["heap_mb"] == 65.0
    assert m["anonymous_mb"] == 201.0          # 204800 + 0 + 1024 kB
    assert m["anonymous_maps"] == 3
    assert m["libraries_mb"] == 35.0           # 30720 + 1024 + 4096 + 4 kB
    assert m["files_mb"] == 10.0               # the interpreter and the model file
    assert m["stacks_mb"] == 0.1
    assert m["other_mb"] == 0.0                # [vdso], 8 kB


def test_the_kinds_add_up_to_the_total(smaps_file):
    m = vd._maps_snapshot(smaps_file)
    kinds = sum(m[f"{k}_mb"] for k in vd._MAPS_KINDS)
    assert abs(kinds - m["total_mb"]) <= 0.3   # each kind is rounded to 0.1
    assert m["total_mb"] == round((2048 + 66560 + 204800 + 1024 + 30720 + 1024 + 4096 + 4 + 8192 + 132 + 8) / 1024, 1)


def test_libraries_are_named_largest_first_and_small_ones_left_out(smaps_file):
    m = vd._maps_snapshot(smaps_file)
    assert m["top_libraries"] == [
        {"name": "libscipy_openblas64_", "mb": 31.0},
        {"name": "_multiarray_umath", "mb": 4.0},
    ]


def test_a_library_is_named_without_its_version_or_hash():
    assert vd._library_name("/x/libscipy_openblas64_-ff651d7f.so") == "libscipy_openblas64_"
    assert vd._library_name("/x/_multiarray_umath.cpython-311-x86_64-linux-gnu.so") == "_multiarray_umath"
    assert vd._library_name("/lib/libc.so.6") == "libc"
    assert vd._library_name("/lib/lib_lightgbm.so") == "lib_lightgbm"


def test_no_path_leaves_the_process(smaps_file):
    text = repr(vd._maps_snapshot(smaps_file))
    assert "/" not in text and "site-packages" not in text and "model.bin" not in text


def test_a_system_without_smaps_reads_none(tmp_path):
    assert vd._maps_snapshot(str(tmp_path / "absent")) is None


def test_a_malformed_rss_line_is_passed_over(tmp_path):
    p = tmp_path / "smaps"
    p.write_text("7f0000000000-7f0004000000 rw-p 00000000 00:00 0 \nRss:\nRss: many kB\nRss: 2048 kB\n")
    assert vd._maps_snapshot(str(p))["anonymous_mb"] == 2.0


@pytest.mark.skipif(not os.path.exists("/proc/self/smaps"), reason="no smaps on this system")
def test_this_process_reads_within_a_tenth_of_its_rss():
    m = vd._maps_snapshot()
    rss = vd._current_rss_mb()
    assert m["total_mb"] > 20
    assert abs(m["total_mb"] - rss) <= 0.1 * rss + 2
    assert m["libraries_mb"] > 0 and m["anonymous_mb"] > 0
    assert m["ms"] < 500


@pytest.mark.skipif(not os.path.exists("/proc/self/smaps"), reason="no smaps on this system")
def test_anonymous_memory_held_shows_as_anonymous():
    """200 MB in one array lands in an unnamed mapping, not in [heap]: the
    reading tells the two apart."""
    before = vd._maps_snapshot()
    held = bytearray(200 * 1024 * 1024)
    for i in range(0, len(held), 4096):
        held[i] = 1
    after = vd._maps_snapshot()
    assert after["anonymous_mb"] - before["anonymous_mb"] >= 150
    assert abs(after["heap_mb"] - before["heap_mb"]) < 50
    del held


def test_health_serves_the_reading_now_and_at_start():
    h = vd._dispatcher._health({})
    assert "memory_maps" in h
    if os.path.exists("/proc/self/smaps"):
        m = h["memory_maps"]
        assert set(f"{k}_mb" for k in vd._MAPS_KINDS) <= set(m)
        assert set(f"{k}_mb" for k in vd._MAPS_KINDS) <= set(m["at_start"])
        assert "at_start" not in m["at_start"]
    else:
        assert h["memory_maps"] is None


def test_the_memory_block_keeps_its_shape():
    """`memory` is pinned key for key by test_daemon_memory_visibility; the
    new reading is its own key and adds nothing there."""
    assert "maps" not in vd._dispatcher._health({})["memory"]


def test_the_reader_changes_nothing():
    """Read-only: it opens one file for reading and calls nothing that
    frees, collects or trims."""
    import inspect
    src = inspect.getsource(vd._maps_snapshot)
    for word in ("gc.", "malloc_trim", "madvise", '"w"', "os.remove", "del "):
        assert word not in src


# ── the session health check prints it ───────────────────────────────────

MEMORY_2026_11_19 = {
    "rss_current_mb": 467.5, "python_blocks": 1715096,
    "malloc": {"in_use_mb": 62.1, "free_retained_mb": 180.1},
    "at_start": {"python_blocks": 294443},
}


def test_the_health_check_prints_the_split():
    maps = dict({
        "total_mb": 467.5, "heap_mb": 70.0, "anonymous_mb": 300.0, "anonymous_maps": 412,
        "libraries_mb": 80.0, "files_mb": 17.0, "stacks_mb": 0.5, "other_mb": 0.0,
        "top_libraries": [{"name": "libscipy_openblas64_", "mb": 30.0}, {"name": "lib_lightgbm", "mb": 9.0}]})
    text = hc._memory_makeup(MEMORY_2026_11_19, maps)
    assert "resident by mapping: anonymous 300.0MB in 412 maps" in text
    assert "[heap] 70.0MB" in text and "libraries 80.0MB (libscipy_openblas64_ 30.0, lib_lightgbm 9.0)" in text
    assert "files 17.0MB" in text
    assert "C heap in use 62.1MB" in text and "python blocks 1715096 (start 294443)" in text


def test_an_older_daemon_without_the_reading_prints_as_before():
    text = hc._memory_makeup(MEMORY_2026_11_19)
    assert "resident by mapping" not in text
    assert "C heap in use 62.1MB, freed but retained 180.1MB" in text
    assert hc._memory_makeup(MEMORY_2026_11_19, None) == text


def test_the_daemon_finding_carries_the_split():
    daemon = {"alive": True, "rss_current_mb": 467.5, "rss_peak_mb": 600.2, "memory": MEMORY_2026_11_19,
              "memory_maps": {"anonymous_mb": 300.0, "anonymous_maps": 412, "heap_mb": 70.0,
                              "libraries_mb": 80.0, "files_mb": 17.0, "top_libraries": []},
              "deep_guard": {"counts": {"normal": 14, "trim": 101, "skip": 0, "survival": 0},
                             "consecutive_not_normal": 101,
                             "last": {"decision": "trim", "rss_mb": 430, "threshold_mb": 400}}}
    f = hc.check_daemon_memory(daemon)
    assert f["severity"] == hc.WARN
    assert "resident by mapping: anonymous 300.0MB in 412 maps" in f["detail"]
    assert "[heap] 70.0MB, libraries 80.0MB, files 17.0MB" in f["detail"]
