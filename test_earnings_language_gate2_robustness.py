import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import earnings_language_gate2_robustness as r


def _mk(n, signal, month, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        tone = rng.uniform(0, 10)
        a = signal * tone + rng.gauss(0, 1)
        out.append({"filedAt": f"{month}-{(i % 20) + 1:02d}", "tone_per_1000w": tone, "alpha": a})
    return out


def test_real_signal_in_both_months_replicates():
    rep = r.horizon_report(_mk(80, 0.5, "2026-08", 1) + _mk(80, 0.5, "2026-09", 2))
    assert rep["verdict"] == "REPLICATES" and rep["perm_p"] < 0.05


def test_signal_in_one_month_only_is_fragile():
    rep = r.horizon_report(_mk(120, 0.0, "2026-08", 3) + _mk(120, 0.8, "2026-09", 4))
    assert rep["verdict"] == "FRAGILE"
    assert set(rep["by_month"]) == {"2026-08", "2026-09"}


def test_pure_noise_is_fragile():
    assert r.horizon_report(_mk(200, 0.0, "2026-09", 5))["verdict"] == "FRAGILE"


def test_spearman_ignores_outliers():
    x = [1, 2, 3, 4, 5]
    assert abs(r.spearman(x, [1, 2, 3, 4, 1000]) - 1.0) < 1e-9
