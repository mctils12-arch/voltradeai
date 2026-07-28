# Regression test for the 2026-07-28 options-flow outage root cause.
#
# Production trail: every Tier2 options candidate died since ~2026-07-23 with
#   T2-FAIL <ticker>: Contract selection failed: '>' not supported between
#   instances of 'NoneType' and 'int'
# The v1.0.526 frame instrumentation named the line live:
#   @position_sizing.py:301:<listcomp>
# trade_feedback had begun carrying OPEN trades with "pnl_pct": null;
# t.get("pnl_pct", 0) does not guard an EXISTING null, so None > 0 raised and
# _dynamic_options_size -> select_contract failed for every candidate.
#
# The repair excludes unclosed records from the stats entirely — never
# counting an open trade as a 0% "loss" (that would deflate win_rate and the
# Kelly base). Offline-safe: feedback file is a tmp fixture.
import json

import position_sizing


def _stats_with(tmp_path, monkeypatch, trades):
    p = tmp_path / "trade_feedback.json"
    p.write_text(json.dumps(trades))
    monkeypatch.setattr(position_sizing, "TRADE_FEEDBACK_PATH", str(p))
    return position_sizing._get_historical_stats()


def _closed(n_win, n_loss):
    return ([{"strategy": "momentum", "pnl_pct": 3.0}] * n_win
            + [{"strategy": "momentum", "pnl_pct": -1.5}] * n_loss)


def test_null_pnl_records_do_not_crash(tmp_path, monkeypatch):
    trades = _closed(8, 4) + [{"strategy": "momentum", "pnl_pct": None}] * 5
    stats = _stats_with(tmp_path, monkeypatch, trades)  # raised TypeError pre-fix
    assert stats["overall"]["total_trades"] == 12       # nulls excluded, not counted


def test_open_trades_do_not_deflate_win_rate(tmp_path, monkeypatch):
    # 8 wins / 4 losses closed -> 66.7%. Pre-fix semantics would have put the
    # 5 open trades in the "losses" bucket (None <= 0 conceptually) and read
    # 8/17 = 47% — a phantom edge-killer.
    stats = _stats_with(tmp_path, monkeypatch, _closed(8, 4) + [{"pnl_pct": None}] * 5)
    assert stats["overall"]["win_rate"] == round(8 / 12, 3)


def test_mostly_open_history_falls_back_to_defaults(tmp_path, monkeypatch):
    # 30 records but only 6 closed: below the 10-closed floor -> defaults,
    # same as a young account. Open trades are not evidence.
    trades = _closed(4, 2) + [{"pnl_pct": None}] * 24
    stats = _stats_with(tmp_path, monkeypatch, trades)
    assert stats["overall"]["total_trades"] == 0        # the defaults block


def test_all_closed_history_unchanged(tmp_path, monkeypatch):
    stats = _stats_with(tmp_path, monkeypatch, _closed(7, 5))
    assert stats["overall"]["total_trades"] == 12
    assert stats["overall"]["win_rate"] == round(7 / 12, 3)
