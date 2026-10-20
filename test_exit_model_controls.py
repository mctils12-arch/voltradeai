"""Research test 1 ([RULE-REVIEW] 2026-10-20): the exit-model reconciliation's
two CONTROL arms and the random-entry BASE RATE in backtest_v2.

research/bot_logic_research_2026-10-20.md Q1 asks for legacy vs live vs two
controls — time-exit only, and a wide (3xATR) stop with the same time exit —
measured against random entries with the same regime gating and sizing
(REASONING STANDARD #3). These tests pin the arms' mechanics on hand-built
bars through the REAL simulate(); the legacy model's byte-identity is pinned
separately in test_backtest_v2_exit_model.py and must keep passing.
"""
import pytest

import backtest_v2 as bt
import test_backtest_v2_exit_model as T   # helpers only (_case, synth_bars)

# flat history at 100 +/- 1 -> atr14 == 2.0 absolute; entry fills at the open of
# path[0] with cost 0, so a 3xATR stop sits at entry - 6.0.
FLAT = (100.0, 101.0, 99.0, 100.0)


def _run(monkeypatch, path, model, **kw):
    out, bars = T._case(monkeypatch, path, exit_model=model, **kw)
    T._only_trade(out)
    return out["trades"][0], out


def test_controls_are_known_models_and_legacy_stays_the_default():
    assert bt.CONTROL_MODELS == ("time_only", "wide_stop")
    assert set(bt.CONTROL_MODELS) <= set(bt.EXIT_MODELS)
    assert bt.DEFAULT_EXIT_MODEL == "legacy"
    assert bt.WIDE_STOP_ATR_MULT == 3.0


def test_time_only_ignores_a_crash_and_a_moonshot_and_exits_at_the_regime_time_stop(monkeypatch):
    # a bar that closes 20% down (every stop fires) then one 60% up (every
    # target fires), then flat: time_only holds through both
    path = [(100.0, 100.5, 79.0, 80.0), (80.0, 161.0, 79.0, 160.0)] + [(160.0, 161.0, 159.0, 160.0)] * 12
    t, out = _run(monkeypatch, path, "time_only")
    assert t["exit_reason"] == "time_stop"
    assert t["holding_days"] == bt.get_adaptive_params("BULL")["time_stop_days"]
    assert out["exit_model"] == "time_only"
    assert t["net_pct"] == pytest.approx(60.0, abs=0.01)


def test_wide_stop_is_three_atr_below_entry_and_has_no_target(monkeypatch):
    # closes at 94.5 (2.75 ATR down) -> no stop; the next close at 93.9 (3.05 ATR) -> stop
    path = [(100.0, 100.5, 94.0, 94.5), (94.5, 95.0, 93.5, 93.9)] + [FLAT] * 3
    t, _ = _run(monkeypatch, path, "wide_stop")
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(93.9)
    assert t["holding_days"] == 1
    # a +60% close is never a take-profit for the control arm
    path = [(100.0, 161.0, 99.0, 160.0)] + [(160.0, 161.0, 159.0, 160.0)] * 12
    t, _ = _run(monkeypatch, path, "wide_stop")
    assert t["exit_reason"] == "time_stop"
    assert t["net_pct"] == pytest.approx(60.0, abs=0.01)


def test_legacy_output_carries_no_exit_model_label(monkeypatch):
    path = [(100.0, 161.0, 99.0, 160.0)] + [(160.0, 161.0, 159.0, 160.0)] * 12
    _, out = T._case(monkeypatch, path, exit_model="legacy")
    assert "exit_model" not in out and "random_entry" not in out


def _random_run(model, rate, seed, bars=None, regimes=None):
    bars = bars or T.synth_bars(400, seed=5)
    spy, vxx = bars, None
    if regimes is not None:
        import backtest_v2 as b
        b_regime = b.regime_series
        b.regime_series = lambda s, v: (list(regimes), "ok")
        try:
            return bt.simulate("TEST", "momentum", bars, spy, vxx, exit_model=model, random_entry=(rate, seed))
        finally:
            b.regime_series = b_regime
    return bt.simulate("TEST", "momentum", bars, spy, vxx, exit_model=model, random_entry=(rate, seed))


def test_random_entry_is_reproducible_by_seed_and_labelled():
    bars = T.synth_bars(400, seed=5)
    n = len(bars["date"])
    import backtest_v2 as b
    keep = b.regime_series
    b.regime_series = lambda s, v: (["BULL"] * n, "ok")
    try:
        a = bt.simulate("TEST", "momentum", bars, bars, None, exit_model="legacy", random_entry=(0.3, 11))
        a2 = bt.simulate("TEST", "momentum", bars, bars, None, exit_model="legacy", random_entry=(0.3, 11))
        c = bt.simulate("TEST", "momentum", bars, bars, None, exit_model="legacy", random_entry=(0.3, 12))
    finally:
        b.regime_series = keep
    assert a == a2, "same seed, same run"
    assert a["trades"] != c["trades"], "a different seed gives different entries"
    assert a["n_trades"] > 0
    assert all(t["score"] is None for t in a["trades"]), "random entries carry no score"
    assert a["exit_model"] == "legacy" and a["random_entry"] == {"rate": 0.3, "seed": 11}


def test_random_entry_respects_the_regime_gate_and_enters_at_rate_one():
    bars = T.synth_bars(300, seed=9)
    n = len(bars["date"])
    import backtest_v2 as b
    keep = b.regime_series
    try:
        b.regime_series = lambda s, v: (["NEUTRAL"] * n, "ok")   # max_pos 0: no entries ever
        blocked = bt.simulate("TEST", "momentum", bars, bars, None, exit_model="time_only", random_entry=(1.0, 1))
        b.regime_series = lambda s, v: (["BULL"] * n, "ok")
        every = bt.simulate("TEST", "momentum", bars, bars, None, exit_model="time_only", random_entry=(1.0, 1))
    finally:
        b.regime_series = keep
    assert blocked["n_trades"] == 0
    # rate 1 enters on the first bar and re-enters the bar after every exit:
    # with a 10-day BULL time stop the count is ~ n / 11
    assert every["n_trades"] >= (n - 2) // 12
    assert every["trades"][0]["date_entry"] == bars["date"][1], "first signal bar 0 fills at bar 1's open"


def test_random_entry_rate_is_validated():
    bars = T.synth_bars(120, seed=2)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            bt.simulate("TEST", "momentum", bars, bars, None, random_entry=(bad, 1))
