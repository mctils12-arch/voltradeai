"""scripts/exit_model_reconciliation.py — the pure parts (research test 1,
[RULE-REVIEW] 2026-10-20): the DSR bar, Lo's SE, and the aggregation that
turns job results into the report's cells. The script's run itself is
evidence (research/exit_model_reconciliation_<date>.md), not a test."""
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("emr", os.path.join(HERE, "scripts", "exit_model_reconciliation.py"))
emr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emr)


def test_expected_max_sharpe_matches_the_research_files_arithmetic():
    # research/bot_logic_research_2026-10-20.md Q4: N=8 -> ~1.4 SD, N=324 -> ~2.9 SD, N=1000 -> ~3.3 SD
    assert abs(emr.expected_max_sharpe(8, 1.0) - 1.4) < 0.1
    assert abs(emr.expected_max_sharpe(324, 1.0) - 2.9) < 0.1
    assert abs(emr.expected_max_sharpe(1000, 1.0) - 3.3) < 0.1
    assert emr.expected_max_sharpe(1, 1.0) == 0.0 and emr.expected_max_sharpe(4, 0.0) == 0.0
    assert emr.expected_max_sharpe(4, 4.0) == 2 * emr.expected_max_sharpe(4, 1.0)


def test_lo_se_scales_with_one_over_root_t():
    se1 = emr.lo_se_annual(1.0, 252)
    se4 = emr.lo_se_annual(1.0, 1008)
    assert abs(se1 / se4 - 2.0) < 1e-9
    assert math.isnan(emr.lo_se_annual(1.0, 1))


def _res(strategy, arm, ticker, trades, seed=None, sharpe=1.0):
    return {"ticker": ticker, "strategy": strategy, "arm": arm, "seed": seed, "rate": 0.1 if seed else None,
            "n_trades": len(trades), "sharpe": sharpe, "cagr": 5.0, "maxdd": 10.0, "t_days": 2520,
            "data_quality": "ok", "trades": trades}


def _t(net, reason="time_stop", regime="BULL", hold=5):
    return {"net_pct": net, "pnl": net, "hold": hold, "reason": reason, "regime": regime}


def test_aggregate_pools_trades_by_regime_and_subtracts_the_random_base_rate():
    results = [
        _res("momentum", "legacy", "AAA", [_t(2.0), _t(-1.0, "stop_loss"), _t(4.0, regime="CAUTION")], sharpe=1.2),
        _res("momentum", "legacy", "BBB", [_t(1.0)], sharpe=0.8),
        _res("momentum", "legacy", "AAA", [_t(0.5), _t(-0.5)], seed=1),
        _res("momentum", "legacy", "BBB", [_t(1.0)], seed=1),
        _res("momentum", "live", "AAA", [_t(3.0)], sharpe=0.4),
        _res("momentum", "live", "BBB", [_t(3.0)], sharpe=0.6),
    ]
    agg = emr.aggregate(results)
    c = agg["cells"]["momentum|legacy"]
    assert c["tickers"] == 2 and c["sharpe_mean"] == 1.0
    all_ = c["by_regime"]["ALL"]
    assert all_["n"] == 4 and all_["mean_net_pct"] == 1.5 and all_["win_rate_pct"] == 75.0
    assert all_["exit_mix"] == {"stop_loss": 0.25, "time_stop": 0.75}
    # pooled SE of (2, -1, 4, 1): pstdev 1.803 / sqrt(4)
    assert all_["se_net_pct"] == round(1.8027756377319946 / 2, 3)
    assert all_["random_n"] == 3 and abs(all_["random_mean_net_pct"] - 0.333) < 1e-9
    assert all_["alpha_pct_per_trade"] == round(1.5 - 1.0 / 3, 3)   # cells are rounded to 3 dp
    assert c["by_regime"]["CAUTION"]["n"] == 1 and c["by_regime"]["CAUTION"]["alpha_pct_per_trade"] is None
    assert "BEAR" not in c["by_regime"]
    # the DSR bar sees the two arms' mean Sharpes (1.0 and 0.5)
    d = agg["dsr"]["momentum"]
    assert d["n_arms"] == 2 and abs(d["var_sharpe"] - 0.0625) < 1e-9
    assert agg["best_arm_count_by_ticker"] == {"momentum": {"legacy": 2}}


def test_render_lists_the_prior_and_every_declared_arm():
    agg = emr.aggregate([_res("momentum", a, "AAA", [_t(1.0)], sharpe=0.5) for a in emr.ARMS])
    md = emr.render(agg, {"date": "2026-10-20", "universe": ["AAA"], "n_universe": 1, "years": 3, "seeds": 0,
                          "workers": 1, "degraded": 0, "data_source": "test", "runtime_s": 0, "generated_at": "t"})
    assert "PRIOR (stated before the run)" in md and emr.PRIOR[:40] in md
    for arm in emr.ARMS:
        assert f"| momentum | {arm} |" in md
    assert "SURVIVORSHIP" in md
