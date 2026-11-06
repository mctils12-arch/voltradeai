#!/usr/bin/env python3
"""
Gate 2 for the Tier-0 stream flags (scripts/stream_signal_gate2.py): the
pre-registered test on synthetic flags with known answers, the base-rate and
independence rules, and the log-file contract with server/streamSignals.ts.

Run: python3 -m pytest -q test_stream_signal_gate2.py
"""
import json
import os
import random
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import stream_signal_gate2 as g2  # noqa: E402

DATES = [f"2026-{m:02d}-{d:02d}" for m in (3, 4, 5, 6) for d in range(1, 29)]   # 112 sessions


def _book(n_tickers=30, n_flags=8, move=0.0, drift=0.0, noise=0.004, seed=7):
    """n_tickers x n_flags first-of-day flags. After each flag the next close
    moves `move` in the flag's direction (negative = reversal) on top of a
    per-session `drift` and noise."""
    rnd = random.Random(seed)
    bars, rows = {}, []
    for k in range(n_tickers):
        t = f"T{k:02d}"
        closes, px = [], 100.0
        flag_days = sorted(rnd.sample(range(5, len(DATES) - 6), n_flags))
        dirs = {d: rnd.choice((1, -1)) for d in flag_days}
        for i in range(len(DATES)):
            px *= 1 + drift + rnd.gauss(0, noise)
            if i - 1 in dirs:
                px *= 1 + dirs[i - 1] * move
            closes.append(px)
        bars[t] = {"date": list(DATES), "close": closes}
        for i in flag_days:
            rows.append({"ticker": t, "et_day": DATES[i], "et_minute": 600, "price": closes[i],
                         "direction": "bullish" if dirs[i] > 0 else "bearish"})
    return rows, bars


def test_known_reversal_is_found():
    rows, bars = _book(move=-0.01)
    out = g2.score(rows, bars)
    assert out["n_tested"] == 240 and out["verdict"] == "REVERSAL", out["excess_by_horizon"]
    assert out["excess_by_horizon"]["h1"]["t"] < -g2.T_CRIT


def test_known_continuation_is_found():
    out = g2.score(*_book(move=+0.01))
    assert out["verdict"] == "CONTINUATION"


def test_no_effect_is_no_edge():
    out = g2.score(*_book(move=0.0, seed=11))
    assert out["verdict"] == "NO_EDGE", out["excess_by_horizon"]["h1"]


def test_below_200_flags_draws_no_verdict():
    rows, bars = _book(n_tickers=10, move=-0.02)
    out = g2.score(rows, bars)
    assert out["n_tested"] == 80 and out["verdict"] == "INSUFFICIENT_N"
    assert out["excess_by_horizon"]["h1"]["t"] < -g2.T_CRIT, "the numbers are still reported"


def test_the_base_rate_removes_drift():
    """All-bullish flags on uptrending tickers look like continuation in raw
    returns; against each ticker's own drift they are nothing."""
    rows, bars = _book(move=0.0, drift=0.004, noise=0.002, seed=3)
    for r in rows:
        r["direction"] = "bullish"
    out = g2.score(rows, bars)
    raw_h1 = sum(bars[r["ticker"]]["close"][DATES.index(r["et_day"]) + 1] / r["price"] - 1 for r in rows) / len(rows)
    assert raw_h1 > 0.003, "premise: raw next-day returns are positive"
    assert out["verdict"] == "NO_EDGE", out["excess_by_horizon"]["h1"]


def test_only_the_first_flag_of_a_ticker_day_is_tested_and_windows_must_be_complete():
    rows, bars = _book(n_tickers=2, n_flags=3)
    dup = dict(rows[0], et_minute=700)
    late = dict(rows[0], et_day=DATES[-2])
    out = g2.score(rows + [dup, late, {"ticker": "T00"}, {"_bad_line": True}], bars)
    assert out["n_tested"] == 6
    assert out["skipped"] == {"later flag, same ticker and day (overlapping window)": 1,
                              "outcome window not complete yet": 1, "incomplete row": 1, "unparseable line": 1}


def test_load_rows_reads_the_log_and_marks_bad_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"ticker": "AMD"}) + "\nnot json\n\n")
    rows = g2.load_rows(str(p))
    assert rows == [{"ticker": "AMD"}, {"_bad_line": True}]
    assert g2.load_rows(str(tmp_path / "missing.jsonl")) == []


def test_the_script_reads_the_file_bot_ts_writes():
    ts = open(os.path.join(_ROOT, "server", "streamSignals.ts")).read()
    name = re.search(r'STREAM_SIGNAL_LOG_FILE = "([^"]+)"', ts).group(1)
    assert g2._default_log().endswith("/" + name)
    bot = open(os.path.join(_ROOT, "server", "bot.ts")).read()
    block = bot[bot.index("// ── Tier-0 stream signal (2026-11-06)"):bot.index('ws.on("error"')]
    for field in ("et_day: sess.day", "et_minute: sess.minute", "ticker, price: close", "direction: sig.direction"):
        assert field in block, f"bot.ts must log `{field}`: the gate-2 test reads it"
