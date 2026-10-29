# Regression test for audit N13 (docs/audits/trading-engine-audit.md 6.8): the DD
# sizing scalar in position_sizing.calculate_position read key "peak_equity" from
# voltrade_equity_peak.json, but Node (server/bot.ts saveEquityPeak) writes
# "equityPeak" -- so the peak always fell back to current equity and the scalar
# was permanently 1.0. Offline-safe: DATA_DIR is a tmp fixture.
import json

import position_sizing

TRADE = {"ticker": "AAPL", "price": 100.0, "score": 75, "volume": 50_000_000,
         "deep_score": 75, "side": "buy", "trade_type": "etf"}


def _scalar(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(position_sizing, "DATA_DIR", str(tmp_path))
    # Positive-edge feedback fixture so the negative-EV halt does not pre-empt sizing.
    fb = tmp_path / "trade_feedback.json"
    fb.write_text(json.dumps([{"strategy": "momentum", "pnl_pct": 3.0}] * 12
                             + [{"strategy": "momentum", "pnl_pct": -1.5}] * 4))
    monkeypatch.setattr(position_sizing, "TRADE_FEEDBACK_PATH", str(fb))
    if payload is not None:
        (tmp_path / "voltrade_equity_peak.json").write_text(json.dumps(payload))
    return position_sizing.calculate_position(dict(TRADE), 88_000.0, [], {})["scalars"]["drawdown_scalar"]


def test_node_written_key_drives_the_scalar(tmp_path, monkeypatch):
    # 88k vs 100k peak = -12% -> 0.6 tier. Pre-fix this returned 1.0.
    assert _scalar(tmp_path, monkeypatch, {"equityPeak": 100_000.0, "savedAt": "x"}) == 0.6


def test_legacy_key_still_honoured(tmp_path, monkeypatch):
    assert _scalar(tmp_path, monkeypatch, {"peak_equity": 100_000.0}) == 0.6


def test_no_file_is_full_size(tmp_path, monkeypatch):
    assert _scalar(tmp_path, monkeypatch, None) == 1.0
