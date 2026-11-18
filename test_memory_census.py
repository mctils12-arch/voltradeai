"""
KNOWN BROKEN #55 (2026-11-18) — the census that names WHO holds the daemon's
Python objects.

Live at 16:12Z the daemon held 520 MB after 1.3 h and 20 scans: 3.82 M
interpreter blocks (0.29 M at start) and a C heap of only 170 MB, so live
Python objects. The instrument of 2026-11-17 lists module-level containers
by LENGTH in this repo's own modules; its longest was 1,050 entries. It
cannot see a 50-key cache of 40,000-object values, an instance attribute, a
closure or a library cache.

These tests build each of those holders and require the census to name it,
to stay inside its limits, to emit no value, and to be reachable only the
way the daemon and the diag route say.
"""
import inspect
import os
import re
import sys
import threading
import types

import memory_census as mc
import voltrade_daemon as vd

HERE = os.path.dirname(os.path.abspath(__file__))


def _fake(name="fake_scanner_mod"):
    m = types.ModuleType(name)
    return m, {name: m}


def _chain(n, tag):
    return [{"occ": f"{tag}{j}", "strike": float(j) + 0.25, "bid": j * 0.1 + 0.01} for j in range(n)]


def test_a_short_cache_of_large_values_is_named_first():
    # The shape the length-ranked instrument misses: 50 keys, 100,000 objects.
    m, mods = _fake()
    m._chain_cache = {f"T{i}_7_60": _chain(500, f"T{i}") for i in range(50)}
    m.small = {i: i for i in range(5000)}
    out = mc.census(modules=mods, top=500)
    # Closures and frames of the test runner itself are holders too; the
    # claim is about module-level names.
    top = [r for r in out["holders"] if not r["name"].startswith(("closure ", "frame "))][0]
    assert top["name"] == "fake_scanner_mod._chain_cache"
    assert top["len"] == 50 and top["type"] == "dict"
    assert top["objects"] > 100_000 and top["mb"] > 1
    assert "truncated" not in top
    assert out["stopped_early"] is None


def test_an_instance_attribute_a_class_attribute_and_a_closure_are_named():
    m, mods = _fake()

    class Client:
        registry = {}

        def __init__(self):
            self.rows = [[float(i), float(i) + 0.5] for i in range(4000)]

    Client.__module__ = "fake_scanner_mod"
    Client.registry = {i: [float(i) + 0.5] for i in range(3000)}
    m.Client = Client
    m.client = Client()

    def make():
        hidden = [{"a": float(i) + 0.5} for i in range(5000)]
        return lambda: hidden

    m.fn = make()
    out = mc.census(modules=mods, top=500)
    names = {r["name"]: r for r in out["holders"]}
    assert names["fake_scanner_mod.client"]["objects"] > 12_000
    assert names["fake_scanner_mod.Client.registry"]["objects"] > 6_000
    closure = [r for n, r in names.items() if n.startswith("closure ") and n.endswith(":hidden")]
    assert closure and closure[0]["objects"] > 10_000 and closure[0]["len"] == 5000


def test_a_running_threads_local_is_named_by_its_function():
    ready, release = threading.Event(), threading.Event()

    def leaked_dispatch():
        payload = [{"k": float(i) + 0.5} for i in range(6000)]
        ready.set()
        release.wait(10)
        return payload

    t = threading.Thread(target=leaked_dispatch, daemon=True)
    t.start()
    try:
        assert ready.wait(5)
        out = mc.census(modules={}, top=500)
        rows = [r for r in out["holders"] if r["name"] == "frame leaked_dispatch:payload"]
        assert rows and rows[0]["objects"] > 12_000
    finally:
        release.set()
        t.join(5)


def test_an_object_is_counted_under_one_name_only():
    m, mods = _fake()
    shared = _chain(2000, "S")
    m.a_first = {"x": shared}
    m.b_second = {"y": shared}
    out = mc.census(modules=mods, top=500)
    names = {r["name"]: r["objects"] for r in out["holders"]}
    assert names["fake_scanner_mod.a_first"] > 8000
    assert "fake_scanner_mod.b_second" not in names   # under 1,000 once the shared list is taken


def test_root_descends_one_level_and_ranks_the_keys():
    m, mods = _fake()
    m._cache = {"SPY_7_60": _chain(3000, "SPY"), "AAPL_7_60": _chain(300, "AAPL"), "n": 1}
    out = mc.census(root="fake_scanner_mod._cache", modules=mods)
    assert out["root"]["len"] == 3 and out["children_total"] == 3
    assert [c["name"] for c in out["children"]] == ["[SPY_7_60]", "[AAPL_7_60]", "[n]"]
    assert out["children"][0]["objects"] > 10 * out["children"][1]["objects"] * 0.9
    assert out["root"]["objects"] == sum(c["objects"] for c in out["children"])
    inst = types.SimpleNamespace(big=_chain(1000, "B"), tiny=1)
    m.obj = inst
    out = mc.census(root="fake_scanner_mod.obj", modules=mods)
    assert out["children"][0]["name"] == ".big"
    assert mc.census(root="fake_scanner_mod.nope", modules=mods) == {"error": "no such name: fake_scanner_mod.nope"}
    assert "error" in mc.census(root="nomodule.attr", modules=mods)


def test_limits_cut_the_walk_and_say_so():
    m, mods = _fake()
    m.huge = [[float(i) + 0.5] for i in range(40_000)]
    m.other = [[float(i) + 0.25] for i in range(40_000)]
    out = mc.census(modules=mods, per_root=10_000, samples=0)
    rows = {r["name"]: r for r in out["holders"]}
    assert rows["fake_scanner_mod.huge"]["truncated"] is True
    assert rows["fake_scanner_mod.huge"]["objects"] == 10_000
    out = mc.census(modules=mods, max_objects=8_000, per_root=1_000_000, samples=0)
    assert out["stopped_early"] == "max_objects"
    assert out["objects_walked"] < 20_000
    ticks = iter(range(10_000_000))
    out = mc.census(modules=mods, budget_sec=1, samples=0, clock=lambda: next(ticks))
    assert out["stopped_early"] == "budget_sec"


def test_the_longest_unreached_container_gets_its_owner_chain():
    # Held by an instance that no module-level name, closure or frame reaches
    # from the modules given: the chain climbs to whatever does hold it.
    class Registry:
        pass

    def build():   # no local of the test's own frame may hold the registry
        reg = Registry()
        reg.cache = {"rows": [[float(i) + 0.5] for i in range(20_000)]}
        mod = types.ModuleType("keeper_mod")
        mod.the_registry = reg
        return mod

    keep = build()
    try:
        sys.modules["keeper_mod"] = keep
        out = mc.census(modules={"keeper_mod": types.ModuleType("empty")}, samples=3)
    finally:
        del sys.modules["keeper_mod"]
    owners = [o for o in out["unreached_owners"] if o.get("len") == 20_000]
    assert owners, out["unreached_owners"]
    chain = owners[0]["held_by"]
    assert chain[0] == "dict(len=1) as [rows]"
    assert any("Registry" in step for step in chain)
    assert out["unreached_total"] > 20_000
    assert {"type", "count"} == set(out["types"][0])


def test_no_value_is_emitted_and_url_keys_are_reported_by_length():
    m, mods = _fake()
    secret = "sk-live-DO-NOT-PRINT-0123456789"
    m.session_headers = {"APCA-API-SECRET-KEY": secret,
                         "https://api.example.com/v1?apikey=" + secret: _chain(600, "U"),
                         ("AAPL", 30): _chain(600, "A"), object(): 1}
    out = mc.census(root="fake_scanner_mod.session_headers", modules=mods)
    text = repr(out)
    assert secret not in text and "api.example.com" not in text
    labels = [c["name"] for c in out["children"]]
    assert "[APCA-API-SECRET-KEY]" in labels            # the key's name, never its value
    assert any(re.fullmatch(r"\[<url-like key, \d+ chars>\]", n) for n in labels)
    assert "[(AAPL, 30)]" in labels and "[<object>]" in labels
    assert mc._key_label("x" * 500) == "x" * 48
    full = repr(mc.census(modules=mods))
    assert secret not in full


def test_the_walk_never_runs_the_code_of_what_it_measures():
    # Found while building: a MagicMock standing in for a function answered
    # len(), __sizeof__ and every attribute read, and the walk ran out of
    # memory. Lengths and sizes now come from the builtin type's own slot.
    from unittest import mock
    calls = []

    class Hostile(dict):
        def __len__(self):
            calls.append("len")
            raise RuntimeError("no")

        def __sizeof__(self):
            calls.append("sizeof")
            raise RuntimeError("no")

        def __getattr__(self, name):
            calls.append("getattr " + name)
            raise RuntimeError("no")

        def items(self):
            calls.append("items")
            raise RuntimeError("no")

    m, mods = _fake()
    h = Hostile()
    dict.__setitem__(h, "rows", _chain(1000, "H"))
    mm = mock.MagicMock()
    mm(1, 2, 3)
    m.h = h
    m.patched_fn = mm
    m.plain = {"x": h, "mock": mm}
    out = mc.census(modules=mods, top=500)
    assert calls == []
    assert len(mm.mock_calls) == 1          # the mock was not called or read into new children
    rows = {r["name"]: r for r in out["holders"]}
    assert rows["fake_scanner_mod.h"]["objects"] > 3000 and rows["fake_scanner_mod.h"]["len"] == 1
    out = mc.census(root="fake_scanner_mod.h", modules=mods)
    assert calls == [] and out["children"][0]["name"] == "[rows]"
    import json
    json.dumps(out)   # plain JSON types only


def test_a_namespace_is_never_billed_to_the_name_that_reaches_it():
    # An instance that keeps a reference to a module's globals (compiled
    # functions and some library objects do) must not claim the module.
    m, mods = _fake()
    m.big = _chain(5000, "G")
    holder = types.SimpleNamespace(g=vars(m))
    m.a_holder = holder               # sorts before `big`
    out = mc.census(modules=mods, top=500)
    rows = {r["name"]: r["objects"] for r in out["holders"]}
    assert rows["fake_scanner_mod.big"] > 15_000
    assert "fake_scanner_mod.a_holder" not in rows


# ── the daemon: on demand, one at a time, not during a scan ──────────────

def test_the_daemon_registers_the_census_and_runs_it():
    d = vd.RPCDispatcher()
    assert d._routes["memory_census"] == (None, "_memory_census")
    r = d.dispatch("memory_census", {})
    assert r["status"] == "ok"
    assert "holders" in r["result"] and "rss_current_mb" in r["result"]
    r = d.dispatch("memory_census", {"root": "voltrade_daemon._modules_loaded"})
    assert r["status"] == "ok" and r["result"]["root"]["type"] == "dict"


def test_the_daemon_refuses_a_census_during_a_scan_unless_forced(monkeypatch):
    d = vd.RPCDispatcher()
    monkeypatch.setattr(vd, "_active_dispatch_snapshot",
                        lambda: [{"method": "run_full_scan", "elapsed_sec": 12.0}, {"method": "health", "elapsed_sec": 0}])
    out = d._memory_census({})
    assert out["refused"].startswith("a scan is running") and out["active"] == ["run_full_scan"]
    assert "holders" in d._memory_census({"force": True})
    monkeypatch.setattr(vd, "_active_dispatch_snapshot", lambda: [{"method": "health", "elapsed_sec": 0}])
    assert "holders" in d._memory_census(None)


def test_only_one_census_runs_at_a_time():
    d = vd.RPCDispatcher()
    assert vd._CENSUS_LOCK.acquire(blocking=False)
    try:
        assert d._memory_census({}) == {"refused": "a census is already running"}
    finally:
        vd._CENSUS_LOCK.release()
    assert "holders" in d._memory_census({})   # and the lock is released after a run


def test_the_health_rpc_does_not_run_the_census():
    # health is polled by every session and by bot.ts's timeout branch; the
    # census holds the GIL and must never ride on it.
    src = inspect.getsource(vd.RPCDispatcher._health)
    assert "census" not in src
    assert "bot_engine" not in open(os.path.join(HERE, "memory_census.py")).read().split('"""', 2)[2]


def test_the_diag_route_asks_the_daemon_only_and_the_health_check_points_at_it():
    bot = open(os.path.join(HERE, "server", "bot.ts")).read()
    i = bot.index('case "daemon": {')
    case = bot[i:i + 1800]
    assert 'pythonRpc("memory_census", args, 30000)' in case
    assert "daemonCensusArgs(req.query.root, req.query.force)" in case
    assert "pythonCall(" not in case           # no subprocess fallback: it would measure itself
    hc = open(os.path.join(HERE, "scripts", "session_health_check.py")).read()
    assert hc.count("/api/diag/daemon?census=1") == 2
