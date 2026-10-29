"""price_panel.py — the Python reader of server/pricePanel.ts's panel.

The fixture under server/__fixtures__/price_panel/ was written by the TS
encoder (server/pricePanelPyParity.test.ts regenerates it and re-checks this
decoder against TS decodeSegment on a fuzz panel too); expected_decoded.json
is the TS decoder's output for it.
"""
import gzip
import json
import os
import shutil
import types

import numpy as np
import pytest

import price_panel as pp

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "server", "__fixtures__", "price_panel")
EXPECTED = json.load(open(os.path.join(FIX, "expected_decoded.json")))


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def test_scalar_decoder_matches_ts_decode_exactly():
    b = pp.current_build(FIX)
    got = [{"s": ln["s"], "mode": ln["mode"], "seg": ln["seg"], "rows": pp.decode_rows(ln)} for ln in pp.iter_lines(b["dir"])]
    assert got == EXPECTED


def test_vector_decoder_matches_ts_decode_exactly():
    b = pp.current_build(FIX)
    lines = list(pp.iter_lines(b["dir"]))
    assert len(lines) == len(EXPECTED)
    for ln, exp in zip(lines, EXPECTED):
        a = pp.decode_arrays(ln)
        for i, r in enumerate(exp["rows"]):
            assert pp.day_str(a["days"][i]) == r["d"]
            for k in ("o", "h", "l", "c"):
                assert float(a[k][i]) == r[k], (ln["s"], i, k)   # exact, not approx
            assert int(a["v"][i]) == r["v"]
            assert _num(float(a["rc"][i])) == r["rc"], (ln["s"], i, "rc")


def test_current_build_only_trusts_current_json(tmp_path):
    assert pp.current_build(str(tmp_path)) is None                       # nothing there
    shutil.copytree(FIX, tmp_path / "p")
    root = str(tmp_path / "p")
    b = pp.current_build(root)
    assert b["build_id"] == "b20261029" and b["manifest"]["as_of"] == "2016-04-15"
    (tmp_path / "p" / "current.json").write_text(json.dumps({"build_id": "../etc"}))
    assert pp.current_build(root) is None                                # path-shaped id refused
    (tmp_path / "p" / "current.json").write_text(json.dumps({"build_id": "b20990101"}))
    assert pp.current_build(root) is None                                # pointer to a missing build
    (tmp_path / "p" / "current.json").unlink()
    (tmp_path / "p" / "building.json").write_text(json.dumps({"build_id": "b20261029"}))
    assert pp.current_build(root) is None                                # an in-progress build is never read


def _series():
    return {s.key: s for s in pp.iter_series(pp.current_build(FIX)["dir"])}


def test_series_policy_raw_preferred_split_at_event_and_flags():
    by = _series()
    reus = [s for s in by.values() if s.symbol == "REUS"]
    assert {s.mode for s in reus} == {"raw"}, "a symbol with raw (asof=-) lines uses only those"
    ev = pp.day_num("2016-02-10")
    old = next(s for s in reus if s.part == 0)
    new = next(s for s in reus if s.part == 1)
    assert old.last_day <= ev < new.first_day, "the raw series is split at the corporate-action date"
    assert old.delisted and old.event_ends_series and not new.event_ends_series
    dead = next(s for s in by.values() if s.symbol == "DEAD")
    assert dead.last_day <= pp.day_num("2016-02-15") and dead.delisted and dead.event_ends_series
    assert "clipped_after_event" in dead.flags
    gaps = sorted((s for s in by.values() if s.symbol == "GAPS"), key=lambda s: s.seg)
    assert [s.seg for s in gaps] == [0, 1] and gaps[1].first_day - gaps[0].last_day > 30
    aaa = by["AAA|default|0|0"]
    assert not aaa.delisted and aaa.asset_id == "id-aaa" and aaa.adjusted
    assert aaa.dates[0] == "2016-01-04" and len(aaa.dates) == len(aaa.c) == 70


def test_pit_universe_delisted_name_disappears_and_ipo_absent_before_first_bar():
    ser = list(_series().values())
    dead = next(s for s in ser if s.symbol == "DEAD")
    last = pp.day_str(dead.last_day)
    after = pp.day_str(dead.last_day + 3 if pp.date.fromisoformat(last).weekday() == 4 else dead.last_day + 1)
    keys_on_last = pp.universe_on(ser, last)
    assert dead.key in keys_on_last
    assert dead.key not in pp.universe_on(ser, after), "a delisted series leaves the universe after its last bar"
    newb = next(s for s in ser if s.symbol == "NEWB")
    assert newb.dates[0] == "2016-03-01"
    assert newb.key not in pp.universe_on(ser, "2016-02-29"), "a future IPO is absent before its first bar"
    assert newb.key in pp.universe_on(ser, "2016-03-01")


def test_universe_membership_never_reads_future_bars():
    ser = list(_series().values())
    d = "2016-02-05"
    day = pp.day_num(d)
    base = pp.universe_on(ser, d, min_price=5.0, min_dollar_volume=1e6)
    for s in ser:   # wreck every bar after d
        fut = s.days > day
        s.c[fut] = 1e-9
        s.raw_close[fut] = 1e-9
        s.v[fut] = 0
    assert pp.universe_on(ser, d, min_price=5.0, min_dollar_volume=1e6) == base
    # and truncating the future entirely changes nothing either
    for s in ser:
        keep = s.days <= day
        for k in ("days", "o", "h", "l", "c", "v", "raw_close"):
            setattr(s, k, getattr(s, k)[keep])
    assert pp.universe_on([s for s in ser if len(s.days)], d, min_price=5.0, min_dollar_volume=1e6) == base


def test_price_filter_uses_raw_close_not_back_adjusted():
    aaa = _series()["AAA|default|0|0"]
    i = 10   # pre-split: raw = 2 x 1.0123 x adjusted
    adj, raw = float(aaa.c[i]), float(aaa.raw_close[i])
    assert raw == pytest.approx(adj * 2 * 1.0123, rel=1e-6)
    floor = (adj + raw) / 2
    m = pp.eligible_mask(aaa, min_price=floor, min_dollar_volume=0)
    assert bool(m[i]), "the raw close clears a floor the back-adjusted price does not"
    peny = next(s for s in _series().values() if s.symbol == "PENY")
    assert np.isnan(peny.raw_close).all()
    assert np.array_equal(pp.pit_price(peny), peny.c), "no raw close recorded -> adjusted close (counted by callers)"


def test_trailing_mean_and_dollar_volume_filter_are_trailing():
    x = np.arange(1.0, 31.0)
    tm = pp.trailing_mean(x, 20)
    for i in range(len(x)):
        lo = max(0, i - 19)
        assert tm[i] == pytest.approx(x[lo:i + 1].mean())
    aaa = _series()["AAA|default|0|0"]
    dv = aaa.c * aaa.v.astype(float)
    thr = float(np.median(dv))
    m = pp.eligible_mask(aaa, 0, min_dollar_volume=thr)
    naive = [dv[max(0, i - 19):i + 1].mean() >= thr for i in range(len(dv))]
    assert list(m) == naive
    assert 0 < sum(naive) < len(naive), "the threshold actually splits the bars"


def test_streaming_is_lazy_and_summary_counts():
    b = pp.current_build(FIX)
    assert isinstance(pp.iter_lines(b["dir"]), types.GeneratorType)
    assert isinstance(pp.iter_series(b["dir"]), types.GeneratorType)
    s = pp.summarize(b)
    assert s["symbols"] == 6 and s["delisted_symbols"] >= 3
    assert s["first_bar"] == "2016-01-04" and s["bars_without_raw_close"] == 30


def test_decode_rejects_a_short_line():
    b = pp.current_build(FIX)
    ln = next(pp.iter_lines(b["dir"]))
    bad = dict(ln, n=ln["n"] + 1)
    with pytest.raises((ValueError, IndexError)):
        pp.decode_arrays(bad)


def test_shard_reader_ignores_non_shard_files(tmp_path):
    shutil.copytree(FIX, tmp_path / "p")
    bars = tmp_path / "p" / "b20261029" / "bars"
    (bars / "b00001.jsonl.gz.tmp").write_bytes(b"partial")
    (bars / "notes.txt").write_text("x")
    with gzip.open(bars / "b00002.jsonl.gz", "wt") as fh:
        fh.write("\n")
    b = pp.current_build(str(tmp_path / "p"))
    assert len(list(pp.iter_lines(b["dir"]))) == len(EXPECTED)


def test_cli_summary_and_no_panel(tmp_path, capsys):
    assert pp.main(["--root", FIX]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["build_id"] == "b20261029"
    assert pp.main(["--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "no_panel"
