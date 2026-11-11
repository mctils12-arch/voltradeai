"""test_earnings_language_qoq_gate2.py — pure-function + runner checks for
scripts/earnings_language_qoq_gate2.py (the paired-filing QoQ language-delta
gate 2 of the SEC 8-K earnings-language root). No network: bars are injected
synthetic series, mirroring test_earnings_language_gate2.py."""
import datetime as dt
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "earnings_language_qoq_gate2",
    os.path.join(os.path.dirname(__file__), "scripts", "earnings_language_qoq_gate2.py"))
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

POS = "strong record growth improved momentum robust gains delivered exceeded "
NEG = "decline weakness shortfall impairment restructuring volatile loss decreased "
NEUTRAL = "the company reported results for the quarter ended and discussed operations segments "


def _text(pos_n: int, neg_n: int, neutral_n: int = 40) -> str:
    return (POS * pos_n + NEG * neg_n + NEUTRAL * neutral_n).strip()


def _filing(cik, ticker, filed, text, acc=None, accept_hour=8, truncated=False):
    return {"accession": acc or f"{cik}-{filed}", "cik": cik, "ticker": ticker, "filedAt": filed,
            "acceptanceDatetime": f"{filed}T{accept_hour:02d}:30:00", "text": text, "truncated": truncated}


def _weekdays(start: str, end: str) -> list:
    d, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out = []
    while d <= e:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


DATES = _weekdays("2026-06-01", "2026-12-31")


def _bars_factory(planted: dict):
    """planted: ticker -> (filedAt, step_return). Close is 100 before the first
    reaction day and 100*(1+step) from it on; SPY flat => alpha == step."""
    def bars(symbol: str, days: int):  # noqa: ARG001
        if symbol == "SPY" or symbol not in planted:
            return {"date": DATES, "close": [100.0] * len(DATES)}
        filed, step = planted[symbol]
        idx = next(i for i, d in enumerate(DATES) if d >= filed)  # pre-market filing => filed day reacts
        return {"date": DATES, "close": [100.0 if i < idx else 100.0 * (1 + step) for i in range(len(DATES))]}
    return bars


# ── text features ──────────────────────────────────────────────────────
def test_similarity_identical_is_one_disjoint_is_zero_and_dtone_sign():
    a = _text(6, 1)
    assert abs(q.cosine_similarity(q.tokens(a), q.tokens(a)) - 1.0) < 1e-9
    assert q.cosine_similarity(q.tokens("alpha beta gamma"), q.tokens("delta epsilon")) == 0.0
    assert q.cosine_similarity([], q.tokens("x y")) is None
    f = q.pair_features(_text(8, 0), _text(0, 8))
    assert f["dtone"] > 0, "more positive words now than before => positive delta"
    assert 0.0 < f["similarity"] < 1.0


# ── pairing ────────────────────────────────────────────────────────────
def test_build_pairs_anchors_follow_ons_gaps_and_drops_are_counted():
    body = _text(3, 3)
    filings = [
        _filing("100", "AAA", "2026-07-30", body),                 # anchor (Q2)
        _filing("100", "AAA", "2026-08-02", body),                 # follow-on (<50d) dropped
        _filing("100", "AAA", "2026-10-29", body),                 # anchor -> pair (gap 91)
        _filing("200", "BBB", "2026-07-01", body),
        _filing("200", "BBB", "2026-11-20", body),                 # gap 142 > 130 -> gap_too_long
        _filing("300", "CCC", "2026-10-15", body),                 # single -> unpaired_first only
        _filing("400", None, "2026-10-15", body),                  # no ticker
        _filing("500", "EEE", "2026-07-20", "too short"),          # short text
        _filing("500", "EEE", "2026-10-20", body),
        _filing("600", "FFF", "2026-08-01", body),
        _filing("600", "FFG", "2026-10-30", body),                 # ticker change, same CIK -> still a pair
        {"accession": "nocik", "ticker": "ZZZ", "filedAt": "2026-10-01", "text": body},  # no cik
    ]
    filings.append(dict(filings[0]))  # duplicate accession ignored
    pairs, c = q.build_pairs(filings)
    assert c["filings_in"] == len(filings)
    assert c["no_ticker"] == 1 and c["no_cik"] == 1 and c["short_text"] == 1
    assert c["follow_on_dropped"] == 1
    assert c["gap_too_long"] == 1
    assert c["ticker_changed"] == 1
    assert c["pairs"] == 2 and len(pairs) == 2
    got = {p["cik"]: p for p in pairs}
    assert got["100"]["ticker"] == "AAA" and got["100"]["gap_days"] == 91 and got["100"]["prior_filedAt"] == "2026-07-30"
    assert got["600"]["ticker"] == "FFG", "the current filing's ticker is the one used for returns"
    # the first anchor of every CIK is unpaired by construction: 5 CIKs keep a usable row
    # (100, 200, 300, 500 — its short-text filing dropped, one left — and 600)
    assert c["unpaired_first"] == c["ciks"] == 5
    assert "cik" in pairs[0] and "dtone" in pairs[0] and "similarity" in pairs[0]


def test_min_gap_and_max_gap_are_the_registered_constants():
    assert q.MIN_GAP_DAYS == 50 and q.MAX_GAP_DAYS == 130 and q.MIN_WORDS == 200
    assert q.MIN_SAMPLE == 30 and q.PASS_SPREAD == 0.005 and q.HORIZONS == (1, 3, 5)
    assert q.FEATURES == ("dtone", "similarity") and q.PRIMARY == "dtone" and q.VARIANTS == 2


# ── verdicts ───────────────────────────────────────────────────────────
def test_mechanical_verdict_insufficient_pass_and_fail():
    small = [{"filedAt": "2026-10-01", "tone_per_1000w": float(i), "alpha": 0.01 * i} for i in range(10)]
    assert q.mechanical_verdict(small)["verdict"] == "INSUFFICIENT"
    good = [{"filedAt": "2026-10-01", "tone_per_1000w": float(i), "alpha": (0.03 if i >= 20 else -0.03)} for i in range(40)]
    v = q.mechanical_verdict(good)
    assert v["verdict"] == "PASS" and v["outlier_robust"] is True and v["spread"] > 0.005
    flat = [{"filedAt": "2026-10-01", "tone_per_1000w": float(i), "alpha": 0.001 * ((-1) ** i)} for i in range(40)]
    assert q.mechanical_verdict(flat)["verdict"] == "FAIL"


def test_root_verdict_requires_primary_to_pass_robustly_on_two_horizons():
    ok = {"verdict": "PASS", "outlier_robust": True, "robustness": {"verdict": "REPLICATES"}}
    fragile = {"verdict": "PASS", "outlier_robust": True, "robustness": {"verdict": "FRAGILE"}}
    assert q.root_verdict({"dtone": {"1": ok, "3": ok, "5": fragile}})["result"] == "PASS"
    assert q.root_verdict({"dtone": {"1": ok, "3": fragile, "5": fragile}})["result"] == "FAIL"
    assert q.root_verdict({"dtone": {"1": {"verdict": "INSUFFICIENT"}, "3": {"verdict": "INSUFFICIENT"}, "5": {"verdict": "INSUFFICIENT"}}})["result"] == "INSUFFICIENT"
    # the secondary feature passing alone never promotes the root
    assert q.root_verdict({"dtone": {"1": fragile, "3": fragile, "5": fragile}, "similarity": {"1": ok, "3": ok, "5": ok}})["result"] == "FAIL"


# ── end to end on synthetic data ───────────────────────────────────────
def _synthetic_universe(n_per_month: int = 36):
    """n CIKs reporting in Oct and n in Nov, each with a July/August prior.
    Planted: alpha = +2% when the tone delta is positive, -2% when negative,
    plus a tiny deterministic wobble — a real, month-replicating signal."""
    filings, planted = [], {}
    k = 0
    for month, cur_days in (("10", ["2026-10-%02d" % d for d in (14, 15, 16, 19, 20, 21, 22, 23, 26, 27, 28, 29)]),
                            ("11", ["2026-11-%02d" % d for d in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17)])):
        for i in range(n_per_month):
            k += 1
            cik, tk = str(1000 + k), f"T{k:03d}"
            cur = cur_days[i % len(cur_days)]
            prior = (dt.date.fromisoformat(cur) - dt.timedelta(days=84 + (i % 7))).isoformat()
            up = (i % 2 == 0)
            filings.append(_filing(cik, tk, prior, _text(3, 3)))
            filings.append(_filing(cik, tk, cur, _text(7, 1) if up else _text(1, 7)))
            planted[tk] = (cur, (0.02 if up else -0.02) + 0.0005 * ((i % 5) - 2))
    return filings, planted


def test_compute_recovers_a_planted_month_replicating_signal_and_keeps_identifiers_out():
    filings, planted = _synthetic_universe()
    res = q.compute(filings, _bars_factory(planted), today=dt.date(2026, 12, 31))
    assert res["pairing"]["pairs"] == 72
    for h in ("1", "3", "5"):
        e = res["features"]["dtone"][h]
        assert e["n"] == 72 and e["verdict"] == "PASS" and e["outlier_robust"]
        assert e["robustness"]["verdict"] == "REPLICATES", e["robustness"]
        assert set(e["robustness"]["by_month"]) == {"2026-10", "2026-11"}
    assert res["root"]["result"] == "PASS" and res["root"]["primary"] == "dtone"
    # similarity is identical across up/down pairs here, so it carries no planted signal
    assert res["features"]["similarity"]["1"]["verdict"] in ("FAIL", "PASS")
    for rows in res["samples"].values():
        for r in rows:
            assert "ticker" not in r and "cik" not in r and "accession" not in r


def test_compute_lookahead_reference_price_follows_the_level_rule():
    """After-close filing: the filing day's close is the reference and the
    next day is the first reaction day; a pre-market filing reacts same day."""
    body = _text(3, 3)
    filings = [_filing("1", "AMC", "2026-07-30", body), _filing("1", "AMC", "2026-10-29", _text(7, 1), accept_hour=17),
               _filing("2", "PRE", "2026-07-30", body), _filing("2", "PRE", "2026-10-29", _text(7, 1), accept_hour=8)]
    # the jump happens on 10-29 itself for both tickers
    bars = _bars_factory({"AMC": ("2026-10-29", 0.05), "PRE": ("2026-10-29", 0.05)})
    res = q.compute(filings, bars, today=dt.date(2026, 12, 31))
    by = {}
    for r in res["samples"]["1"]:
        by.setdefault(round(r["alpha"], 4), 0)
        by[round(r["alpha"], 4)] += 1
    # PRE (pre-market): reference = 10-28 close 100, +1d = 10-29 close 105 => +5%.
    # AMC (after close): reference = 10-29 close (already 105), +1d = 10-30 => 0%: the
    # same-day move is correctly NOT credited to a filing that came after it.
    assert by.get(0.05) == 1 and by.get(0.0) == 1, by


def test_run_writes_outputs_summary_line_and_carries_first_sufficient_run(tmp_path):
    filings, planted = _synthetic_universe()
    fixture = tmp_path / "filings.json"
    fixture.write_text(json.dumps({"filings": filings}))
    out_dir = str(tmp_path / "earnings_qoq")
    bars = _bars_factory(planted)
    s1 = q.run(out_dir, None, str(fixture), bars, "1.0.1", 900.0, today=dt.date(2026, 12, 31))
    assert s1["status"] == "ok" and s1["verdict"] == "PASS" and s1["pairs"] == 72
    latest = json.loads((tmp_path / "earnings_qoq" / "latest.json").read_text())
    assert latest["schema"] == q.SCHEMA and latest["variants"] == 2 and latest["root_verdict"]["result"] == "PASS"
    assert latest["first_sufficient_run"]["generated_at"] == latest["generated_at"]
    assert latest["first_sufficient_run"]["code_version"] == "1.0.1"
    blob = json.dumps(latest)
    for ident in ("T001", "1001", "AMC", "accession"):
        assert ident not in blob, f"identifier {ident} leaked into latest.json"
    assert (tmp_path / "earnings_qoq" / "samples_latest.json").exists()
    assert json.loads((tmp_path / "earnings_qoq" / "last_run.json").read_text())["status"] == "ok"
    # second run with a different code version: first_sufficient_run is carried, not overwritten
    s2 = q.run(out_dir, None, str(fixture), bars, "1.0.2", 900.0, today=dt.date(2026, 12, 31))
    assert s2["status"] == "ok"
    latest2 = json.loads((tmp_path / "earnings_qoq" / "latest.json").read_text())
    assert latest2["code_version"] == "1.0.2"
    assert latest2["first_sufficient_run"] == latest["first_sufficient_run"]


def test_run_skips_when_the_archive_is_empty_and_main_prints_one_json_line(tmp_path, capsys):
    out = q.run(str(tmp_path / "o"), str(tmp_path / "missing_archive"), None, lambda s, d: {}, "1.0.1", 900.0)
    assert out["status"] == "skipped" and "no earnings8k filings" in out["reason"]
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"filings": []}))
    rc = q.main(["--run", "--input-json", str(empty), "--out-dir", str(tmp_path / "o2"), "--code-version", "1.0.1"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 0 and json.loads(line)["status"] == "skipped"


def test_main_error_path_writes_last_run_and_exits_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = q.main(["--run", "--input-json", str(bad), "--out-dir", str(tmp_path / "o3")])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 1 and json.loads(line)["status"] == "error"
    assert json.loads((tmp_path / "o3" / "last_run.json").read_text())["status"] == "error"


def test_archive_loader_reads_gz_and_plain_days_through_the_level_module(tmp_path):
    import gzip
    d = tmp_path / "earnings8k"
    d.mkdir()
    today = dt.date.today()
    (d / f"{today.isoformat()}.jsonl").write_text(json.dumps(_filing("1", "A", today.isoformat(), _text(3, 3))) + "\n")
    with gzip.open(d / f"{(today - dt.timedelta(days=3)).isoformat()}.jsonl.gz", "wt") as f:
        f.write(json.dumps(_filing("2", "B", (today - dt.timedelta(days=3)).isoformat(), _text(3, 3))) + "\n")
    rows, source, span = q.load_filings(str(d), None)
    assert len(rows) == 2 and source.startswith("archive-dir:")
    assert span["first_day"] == (today - dt.timedelta(days=3)).isoformat() and span["last_day"] == today.isoformat()
