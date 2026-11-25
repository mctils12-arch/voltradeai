"""test_snapshot_refresh_job.py — snapshot_refresh_job.py, the nightly refresh the
server runs (server/snapshotRefresh.ts; research/open_questions.md (ah4)).

No test here touches the network: the refreshers are replaced by a writer.
"""
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import snapshot_refresh_job as job  # noqa: E402

NOW = dt.datetime(2026, 11, 25, 7, 0, tzinfo=dt.timezone.utc)


def _eia(built, last="2026-11-13", n=5):
    return {"built": built, "series": {f"s{i}": {"last": last, "points": []} for i in range(n)}}


def _writer(doc):
    def call(module, staging):
        with open(staging, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
    return call


def _bundle(tmp_path, rel, doc):
    fp = tmp_path / "image" / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(doc))
    return [str(tmp_path / "image")]


EIA = job.SOURCES[0]


def test_sources_are_the_four_the_session_command_refreshes_and_each_module_is_at_the_root():
    spec = importlib.util.spec_from_file_location("snapshot_refresh", os.path.join(REPO, "scripts", "snapshot_refresh.py"))
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    assert [(s["key"], s["module"] + ".py") for s in job.SOURCES] == list(sr.STEPS)
    for s in job.SOURCES:
        assert os.path.exists(os.path.join(REPO, s["module"] + ".py")), "the image ships root *.py only"
        assert os.path.exists(os.path.join(REPO, s["rel"])), "the bundled copy is the cold start"
        assert s["min_age_h"] >= 20


def test_summarize_reads_every_real_bundled_snapshot():
    for s in job.SOURCES:
        with open(os.path.join(REPO, s["rel"]), encoding="utf-8") as fh:
            doc = json.load(fh)
        out = job.summarize(s["key"], doc)
        assert out["n_series"] > 0 and re.match(r"^\d{4}-\d{2}", out["latest"]), (s["key"], out)
        assert job.built_of(doc) is not None, s["key"]


def test_validate_accepts_newer_and_refuses_older_smaller_or_earlier():
    served = _eia("2026-11-24T01:30:00+00:00")
    assert job.validate("eia", _eia("2026-11-25T06:00:00+00:00", last="2026-11-20"), served) == (True, "ok")
    ok, why = job.validate("eia", _eia("2026-11-24T01:30:00+00:00"), served)
    assert not ok and "not newer" in why
    ok, why = job.validate("eia", _eia("2026-11-25T06:00:00+00:00", n=4), served)
    assert not ok and "4 series against 5" in why, "one of five series missing is under 90%"
    ok, why = job.validate("eia", _eia("2026-11-25T06:00:00+00:00", last="2026-11-06"), served)
    assert not ok and "earlier" in why
    ok, why = job.validate("eia", {"series": {"a": {"last": "2026-11-20"}}}, served)
    assert not ok and "build time" in why
    ok, why = job.validate("eia", {"built": "2026-11-25T06:00:00+00:00", "series": {}}, served)
    assert not ok and "unexpected shape" in why
    ok, why = job.validate("eia", None, served)
    assert not ok


def test_jodi_one_series_fewer_of_350_is_accepted():
    """Measured 2026-11-25: the source served 349 series where the bundled copy held 350."""
    def jodi(built, n, latest):
        return {"built_at": built, "latest_period": latest, "series": {str(i): [] for i in range(n)}}
    assert job.validate("jodi", jodi("2026-11-25T02:39:00+00:00", 349, "2026-09"), jodi("2026-11-02T13:40:01+00:00", 350, "2026-08"))[0]


def test_served_copy_is_the_newer_of_bundled_and_volume(tmp_path):
    rel = EIA["rel"]
    roots = _bundle(tmp_path, rel, _eia("2026-11-24T01:30:00+00:00"))
    out_root = str(tmp_path / "vol")
    assert job.served_copy(rel, out_root, roots)[1] == "bundled"
    os.makedirs(os.path.dirname(os.path.join(out_root, rel)))
    with open(os.path.join(out_root, rel), "w") as fh:
        json.dump(_eia("2026-11-25T06:00:00+00:00"), fh)
    assert job.served_copy(rel, out_root, roots)[1] == "refreshed"
    # a deploy that bundles a newer file than the volume holds wins back
    roots = _bundle(tmp_path, rel, _eia("2026-11-26T01:00:00+00:00"))
    assert job.served_copy(rel, out_root, roots)[1] == "bundled"
    with open(os.path.join(out_root, rel), "w") as fh:
        fh.write("{not json")
    assert job.served_copy(rel, out_root, roots)[1] == "bundled"


def test_refresh_one_fresh_copy_fetches_nothing(tmp_path):
    roots = _bundle(tmp_path, EIA["rel"], _eia("2026-11-25T01:00:00+00:00"))
    def boom(module, staging):
        raise AssertionError("a 6-hour-old copy must not be fetched again")
    r = job.refresh_one(EIA, str(tmp_path / "vol"), NOW, call=boom, bundled_roots=roots)
    assert r["status"] == "fresh" and r["origin"] == "bundled"


def test_refresh_one_writes_the_volume_copy_and_leaves_no_staging_file(tmp_path):
    roots = _bundle(tmp_path, EIA["rel"], _eia("2026-11-02T01:00:00+00:00", last="2026-10-23"))
    out_root = str(tmp_path / "vol")
    new = _eia("2026-11-25T06:59:00+00:00", last="2026-11-20")
    r = job.refresh_one(EIA, out_root, NOW, call=_writer(new), bundled_roots=roots)
    assert r["status"] == "refreshed" and r["latest"] == "2026-11-20" and r["n_series"] == 5
    final = os.path.join(out_root, EIA["rel"])
    with open(final) as fh:
        assert json.load(fh) == new
    assert os.listdir(os.path.dirname(final)) == [os.path.basename(final)]
    # the bundled file is never written
    with open(os.path.join(roots[0], EIA["rel"])) as fh:
        assert json.load(fh)["built"] == "2026-11-02T01:00:00+00:00"


def test_refresh_one_refused_file_keeps_the_served_copy(tmp_path):
    roots = _bundle(tmp_path, EIA["rel"], _eia("2026-11-02T01:00:00+00:00"))
    out_root = str(tmp_path / "vol")
    r = job.refresh_one(EIA, out_root, NOW, call=_writer(_eia("2026-11-25T06:59:00+00:00", n=2)), bundled_roots=roots)
    assert r["status"] == "rejected" and "2 series against 5" in r["reason"]
    assert os.listdir(os.path.dirname(os.path.join(out_root, EIA["rel"]))) == []


def test_refresh_one_reports_a_failed_fetch_with_its_reason(tmp_path):
    roots = _bundle(tmp_path, EIA["rel"], _eia("2026-11-02T01:00:00+00:00"))
    def refused(module, staging):
        raise OSError("HTTP Error 403: Forbidden")
    r = job.refresh_one(EIA, str(tmp_path / "vol"), NOW, call=refused, bundled_roots=roots)
    assert r == {"status": "error", "reason": "OSError: HTTP Error 403: Forbidden"}


def test_call_refresher_turns_a_refusal_to_write_into_an_error_and_keeps_stdout_clean(tmp_path, capsys):
    class Fake:
        OUT = None
        @staticmethod
        def main():
            print("progress line")
            raise SystemExit("every series failed — refusing to write an empty artifact")
    with pytest.raises(RuntimeError, match="every series failed"):
        job._call_refresher("eia_weekly", str(tmp_path / "x"), load=lambda name: Fake)
    assert Fake.OUT == str(tmp_path / "x")
    assert "progress line" not in capsys.readouterr().out, "stdout carries the one JSON line only"
    class Returns1(Fake):
        @staticmethod
        def main():
            return 1
    with pytest.raises(RuntimeError, match="returned 1"):
        job._call_refresher("jodi_oil", str(tmp_path / "y"), load=lambda name: Returns1)


def test_run_one_source_down_does_not_stop_the_others_and_the_run_reads_error(tmp_path):
    roots = []
    for s in job.SOURCES:
        roots = _bundle(tmp_path, s["rel"], _eia("2026-11-02T01:00:00+00:00"))
    new = _eia("2026-11-25T06:59:00+00:00", last="2026-11-20")
    def call(module, staging):
        if module == "stb_rail":
            raise OSError("HTTP Error 403: Forbidden")
        _writer(new)(module, staging)
    # eia-shaped documents: the cpc/rail/jodi summaries do not read them, so only eia and the failing source are run
    # the budget is the job's own process's; inside a whole pytest run this process is far larger
    out = job.run(str(tmp_path / "vol"), only=["eia", "rail"], code_version="t", max_rss_mb=1e9, now=NOW, call=call, bundled_roots=roots)
    assert out["status"] == "error" and out["refreshed"] == 1
    assert out["sources"]["eia"]["status"] == "refreshed" and out["sources"]["rail"]["status"] == "error"
    with open(os.path.join(str(tmp_path / "vol"), "last_run.json")) as fh:
        last = json.load(fh)
    assert last["schema"] == job.SCHEMA and last["finished_at"] and last["sources"].keys() == {"eia", "rail"}
    assert last["vintage_bytes"] == job.vintage_bytes(str(tmp_path / "vol")) > 0, "the run reports the record's size"
    assert job.overall_status({"a": {"status": "fresh"}, "b": {"status": "refreshed"}}) == "ok"
    assert job.overall_status({"a": {"status": "rejected"}}) == "error"


def test_run_over_its_memory_budget_starts_no_further_source(tmp_path):
    def boom(module, staging):
        raise AssertionError("no source may start over budget")
    out = job.run(str(tmp_path / "vol"), only=["eia"], max_rss_mb=0.0, now=NOW, call=boom, bundled_roots=[str(tmp_path)])
    assert out["status"] == "error" and "over the 0 MB budget" in out["sources"]["eia"]["reason"]


def test_cli_prints_one_json_line_and_refuses_an_unknown_source(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(REPO, "snapshot_refresh_job.py"), "--run", "--only", "nope",
                        "--out-root", str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert json.loads(r.stdout.strip().split("\n")[-1])["status"] == "error"


def test_the_two_spreadsheet_readers_are_in_the_production_requirements():
    """The image installs requirements.txt and nothing else; without these two
    the EIA and rail sources fail every night (measured: ModuleNotFoundError)."""
    with open(os.path.join(REPO, "requirements.txt"), encoding="utf-8") as fh:
        names = {re.split(r"[<>=!~ ]", line.strip(), maxsplit=1)[0].lower() for line in fh
                 if line.strip() and not line.startswith("#")}
    assert {"xlrd", "openpyxl"} <= names


@pytest.mark.parametrize("name", [s["module"] for s in job.SOURCES])
def test_scripts_shim_serves_the_root_refresher(name):
    shim_file = os.path.join(REPO, "scripts", name + ".py")
    assert "\ndef " not in open(shim_file, encoding="utf-8").read(), "the shim holds no logic"
    spec = importlib.util.spec_from_file_location("shim_" + name, shim_file)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert callable(shim.main)
    assert os.path.normpath(shim.OUT).startswith(os.path.join(REPO, "datacore")), "a hand run still writes the repo's datacore/"


# ── the vintage record (open_questions.md (ao3)) ─────────────────────────────

def _eia_pts(built, **series):
    return {"built": built, "series": {k: {"last": pts[-1][0], "points": pts} for k, pts in series.items()}}


def test_flatten_reads_every_real_bundled_snapshot_and_a_copy_differs_from_itself_nowhere():
    for s in job.SOURCES:
        with open(os.path.join(REPO, s["rel"]), encoding="utf-8") as fh:
            doc = json.load(fh)
        flat = job.flatten(s["key"], doc)
        assert len(flat) == job.summarize(s["key"], doc)["n_series"], s["key"]
        assert sum(len(v) for v in flat.values()) > 1000, s["key"]
        assert job.diff_points(flat, flat) == []


def test_flatten_leaves_out_unreported_periods_and_refuses_a_misaligned_array():
    rail = {"weeks": ["2026-11-11", "2026-11-18"], "series": {"A": [1.0, None], "B": [None, 2.0]}}
    assert job.flatten("rail", rail) == {"A": {"2026-11-11": 1.0}, "B": {"2026-11-18": 2.0}}
    cpc = {"axes": {"H": ["2026-11-20", "2026-11-21"], "C": ["2026-11-21"]}, "series": {"AL|H": [4, 5], "AL|C": [0]}}
    assert job.flatten("cpc", cpc) == {"AL|H": {"2026-11-20": 4, "2026-11-21": 5}, "AL|C": {"2026-11-21": 0}}
    with pytest.raises(ValueError, match="1 values against 2 periods"):
        job.flatten("rail", {"weeks": ["a", "b"], "series": {"A": [1.0]}})
    jodi = {"series": {"AU|CRUDEOIL": {"points": [["2026-08", 8165.96, "1"]]}}}
    assert job.flatten("jodi", jodi) == {"AU|CRUDEOIL": {"2026-08": [8165.96, "1"]}}


def test_diff_points_names_first_prints_revisions_and_removals():
    old = {"a": {"w1": 1.0, "w2": 2.0}, "gone": {"w1": 9.0}}
    new = {"a": {"w1": 1.0, "w2": 2.5, "w3": 3.0}, "born": {"w3": 7.0}}
    assert job.diff_points(old, new) == [
        ("a", "w2", "revised", 2.0, 2.5), ("a", "w3", "new", None, 3.0),
        ("born", "w3", "new", None, 7.0), ("gone", "w1", "removed", 9.0, None)]
    # JODI: a changed assessment code with the same value is a revision
    assert job.diff_points({"s": {"m": [5.0, "3"]}}, {"s": {"m": [5.0, "1"]}}) == [("s", "m", "revised", [5.0, "3"], [5.0, "1"])]


def test_refresh_records_what_changed_before_the_served_copy_is_replaced(tmp_path):
    """THE CASE THIS EXISTS FOR: the source revises a week it already published.
    After the refresh the old value is on no disk but in the record."""
    served = _eia_pts("2026-11-24T01:30:00+00:00", crude=[["2026-11-06", 414417.0], ["2026-11-13", 417797.0]])
    new = _eia_pts("2026-11-25T06:59:00+00:00", crude=[["2026-11-06", 414417.0], ["2026-11-13", 417500.0], ["2026-11-20", 419000.0]])
    roots = _bundle(tmp_path, EIA["rel"], served)
    out_root = str(tmp_path / "vol")
    r = job.refresh_one(EIA, out_root, NOW, call=_writer(new), bundled_roots=roots)
    assert r["status"] == "refreshed" and r["vintage"] == {"new": 1, "revised": 1, "removed": 0}
    v = job.read_vintage("eia", out_root)
    assert v["bad_lines"] == 0
    assert v["builds"] == [{"schema": job.VINTAGE_SCHEMA, "key": "eia", "seen": "2026-11-25T06:59:00+00:00",
                            "prev": "2026-11-24T01:30:00+00:00", "new": 1, "revised": 1, "removed": 0,
                            "series_added": [], "series_removed": []}]
    assert v["rows"] == [
        {"seen": "2026-11-25T06:59:00+00:00", "prev": "2026-11-24T01:30:00+00:00", "series": "crude",
         "period": "2026-11-13", "kind": "revised", "old": 417797.0, "new": 417500.0},
        {"seen": "2026-11-25T06:59:00+00:00", "prev": "2026-11-24T01:30:00+00:00", "series": "crude",
         "period": "2026-11-20", "kind": "new", "old": None, "new": 419000.0}]
    assert job.overall_status({"eia": r}) == "ok"
    size = job.vintage_bytes(out_root)
    assert size == sum(os.path.getsize(os.path.join(out_root, "vintages", f)) for f in ("eia.jsonl", "builds.jsonl")) > 0
    assert job.vintage_bytes(str(tmp_path / "nowhere")) == 0
    # the next night diffs against the volume copy, and the record grows by that night only
    later = _eia_pts("2026-11-26T06:59:00+00:00", crude=new["series"]["crude"]["points"])
    r2 = job.refresh_one(EIA, out_root, NOW + dt.timedelta(days=1), call=_writer(later), bundled_roots=roots)
    assert r2["vintage"] == {"new": 0, "revised": 0, "removed": 0}
    v = job.read_vintage("eia", out_root)
    assert [b["seen"] for b in v["builds"]] == ["2026-11-25T06:59:00+00:00", "2026-11-26T06:59:00+00:00"], "a night nothing changed still has its build row"
    assert len(v["rows"]) == 2


def test_a_series_that_leaves_the_source_is_named_on_the_build_row(tmp_path):
    """Measured 2026-11-25: JODI served 349 series where the copy of 2 November held 350 (PE|NGL)."""
    def jodi(built, latest, **series):
        return {"built_at": built, "latest_period": latest, "series": {k.replace("_", "|"): {"points": p} for k, p in series.items()}}
    src = job.SOURCES[3]
    keep = {f"X{i}_TOTCRUDE": [["2026-08", float(i), "1"]] for i in range(12)}
    roots = _bundle(tmp_path, src["rel"], jodi("2026-11-02T13:40:01+00:00", "2026-08", PE_NGL=[["2026-07", 0.0, "3"], ["2026-08", 0.0, "3"]], **keep))
    out_root = str(tmp_path / "vol")
    r = job.refresh_one(src, out_root, NOW, call=_writer(jodi("2026-11-25T06:59:00+00:00", "2026-08", **keep)), bundled_roots=roots)
    assert r["status"] == "refreshed" and r["vintage"] == {"new": 0, "revised": 0, "removed": 2}
    v = job.read_vintage("jodi", out_root)
    assert v["builds"][0]["series_removed"] == ["PE|NGL"]
    assert [(x["period"], x["old"], x["new"]) for x in v["rows"]] == [("2026-07", [0.0, "3"], None), ("2026-08", [0.0, "3"], None)]


def test_a_rebased_history_keeps_the_whole_previous_copy_instead_of_its_rows(tmp_path, monkeypatch):
    import gzip
    monkeypatch.setattr(job, "VINTAGE_MAX_ROWS", 3)
    served = _eia_pts("2026-11-24T01:30:00+00:00", crude=[[f"w{i}", float(i)] for i in range(5)])
    new = _eia_pts("2026-11-25T06:59:00+00:00", crude=[[f"w{i}", float(i) + 0.5] for i in range(5)])
    roots = _bundle(tmp_path, EIA["rel"], served)
    out_root = str(tmp_path / "vol")
    r = job.refresh_one(EIA, out_root, NOW, call=_writer(new), bundled_roots=roots)
    assert r["vintage"] == {"new": 0, "revised": 5, "removed": 0, "full_prev": "eia_20261124T013000Z.json.gz"}
    with gzip.open(os.path.join(out_root, "vintages", "full", "eia_20261124T013000Z.json.gz"), "rt") as fh:
        assert json.load(fh) == served
    v = job.read_vintage("eia", out_root)
    assert v["rows"] == [] and v["builds"][0]["full_prev"] == "eia_20261124T013000Z.json.gz" and v["builds"][0]["revised"] == 5
    assert not os.path.exists(os.path.join(out_root, "vintages", "eia.jsonl"))


def test_a_record_that_cannot_be_written_does_not_stop_the_refresh_and_the_run_reads_error(tmp_path):
    served = _eia_pts("2026-11-24T01:30:00+00:00", crude=[["2026-11-13", 1.0]])
    new = _eia_pts("2026-11-25T06:59:00+00:00", crude=[["2026-11-13", 2.0]])
    roots = _bundle(tmp_path, EIA["rel"], served)
    out_root = str(tmp_path / "vol")
    os.makedirs(out_root)
    with open(os.path.join(out_root, "vintages"), "w") as fh:  # a file where the directory must be
        fh.write("x")
    r = job.refresh_one(EIA, out_root, NOW, call=_writer(new), bundled_roots=roots)
    assert r["status"] == "refreshed" and "error" in r["vintage"]
    with open(os.path.join(out_root, EIA["rel"])) as fh:
        assert json.load(fh) == new, "the served copy is still refreshed"
    assert job.overall_status({"eia": r}) == "error", "and the job row says the record was not written"


def test_point_rows_without_their_build_row_are_not_read_and_a_torn_line_is_counted(tmp_path):
    """A run killed between the two appends leaves point rows with no build row;
    the next night writes them again under its own `seen`."""
    out_root = str(tmp_path / "vol")
    vdir = os.path.join(out_root, "vintages")
    os.makedirs(vdir)
    row = {"seen": "2026-11-25T06:59:00+00:00", "prev": "p", "series": "crude", "period": "w", "kind": "revised", "old": 1.0, "new": 2.0}
    with open(os.path.join(vdir, "eia.jsonl"), "w") as fh:
        fh.write(json.dumps(row) + "\n" + json.dumps({**row, "seen": "2026-11-26T06:59:00+00:00"}) + "\n" + '{"seen": "2026-11-2')
    with open(os.path.join(vdir, "builds.jsonl"), "w") as fh:
        fh.write(json.dumps({"key": "eia", "seen": "2026-11-26T06:59:00+00:00"}) + "\n" + json.dumps({"key": "cpc", "seen": "2026-11-25T06:59:00+00:00"}) + "\n")
    v = job.read_vintage("eia", out_root)
    assert [x["seen"] for x in v["rows"]] == ["2026-11-26T06:59:00+00:00"] and v["bad_lines"] == 1
    assert job.read_vintage("rail", out_root) == {"builds": [], "rows": [], "bad_lines": 0}


def test_the_off_switch_stops_the_record_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VOLTRADE_SNAPSHOT_VINTAGES", "0")
    served = _eia_pts("2026-11-24T01:30:00+00:00", crude=[["2026-11-13", 1.0]])
    new = _eia_pts("2026-11-25T06:59:00+00:00", crude=[["2026-11-13", 2.0]])
    out_root = str(tmp_path / "vol")
    r = job.refresh_one(EIA, out_root, NOW, call=_writer(new), bundled_roots=_bundle(tmp_path, EIA["rel"], served))
    assert r["status"] == "refreshed" and "vintage" not in r
    assert not os.path.exists(os.path.join(out_root, "vintages"))
    for v, on in (("1", True), ("on", True), ("0", False), ("off", False), ("false", False), ("", False)):
        assert job.vintages_enabled({"VOLTRADE_SNAPSHOT_VINTAGES": v}) is on
    assert job.vintages_enabled({}) is True
