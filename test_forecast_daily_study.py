"""Pure-computation tests for scripts/forecast_daily_study.py (Part C range
study, research/forecast_daily_study_2026-11-04.md). Synthetic data only, no
network. The load-bearing ones are the no-lookahead pair: a regressor at
origin t reads closes <= t only, and a walk-forward refit trains only on
labels that end >= h sessions before the first origin it forecasts.
"""
import math
import os
import sys

import numpy as np

# APPENDED, never inserted at 0: scripts/ holds a source_validation.py that
# would shadow the root module of the same name for every later test.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import forecast_daily_study as st  # noqa: E402


def _closes(n, seed=0, sd=0.01):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, sd, n)))


def test_har_rows_regressors_use_only_closes_up_to_t():
    c = _closes(200, seed=1)
    a = st.har_rows(c, 1)
    t = 120
    c2 = c.copy()
    c2[t + 1:] *= 3.0                       # a shock strictly after t
    b = st.har_rows(c2, 1)
    for k in ("v_d", "v_w", "v_m", "rv20"):
        assert a[k][t] == b[k][t], k
    r = np.diff(np.log(c))
    assert abs(a["v_m"][t] - math.sqrt(np.mean(r[t - 22:t] ** 2))) < 1e-12
    assert abs(a["v_w"][t] - math.sqrt(np.mean(r[t - 5:t] ** 2))) < 1e-12
    assert abs(a["v_d"][t] - abs(r[t - 1])) < 1e-12


def test_har_rows_target_is_the_forward_move():
    c = _closes(60, seed=2)
    for h in (1, 5):
        rows = st.har_rows(c, h)
        assert abs(rows["y"][30] - abs(math.log(c[30 + h] / c[30]))) < 1e-12
        assert np.all(np.isnan(rows["y"][len(c) - h:]))
        assert np.all(np.isnan(rows["v_m"][:22]))


def test_training_mask_purges_and_embargoes_by_the_horizon():
    t = np.arange(100)
    for h in (1, 5):
        m = st.training_mask(t, 60, h)
        assert t[m].max() + h == 60 - h      # last label ends h sessions before the refit origin
        assert not np.any(m & (t + h > 60 - h))


def test_walk_forward_refit_never_sees_labels_after_its_cutoff():
    n = 900
    rows = {s: st.har_rows(_closes(n, seed=k + 3), 1) for k, s in enumerate(("A", "B"))}
    first, last = 700, 760
    q, fits = st.walk_forward_q(rows, ["A", "B"], ["v_d", "v_w", "v_m"], 1, first, last)
    assert fits and all(f["n_train"] >= st.MIN_TRAIN for f in fits)
    # poison every label that ends after the first refit's cutoff: the first
    # refit segment's forecasts must not move
    poisoned = {s: {k: v.copy() for k, v in r.items()} for s, r in rows.items()}
    for s in poisoned:
        tt = np.arange(n)
        poisoned[s]["y"][tt + 1 > first - 1] *= 50.0
    q2, _ = st.walk_forward_q(poisoned, ["A", "B"], ["v_d", "v_w", "v_m"], 1, first, last)
    seg = slice(first, first + st.REFIT_EVERY)
    for s in ("A", "B"):
        assert np.allclose(q[s][seg], q2[s][seg], equal_nan=True)


def test_block_bootstrap_ci_brackets_the_estimate_and_is_deterministic():
    rng = np.random.default_rng(5)
    num = rng.binomial(10, 0.68, 500).astype(float)
    den = np.full(500, 10.0)
    lo, hi = st.block_bootstrap_ratio(num, den, n_boot=400)
    est = num.sum() / den.sum()
    assert lo < est < hi and hi - lo < 0.05
    assert (lo, hi) == st.block_bootstrap_ratio(num, den, n_boot=400)
    c_lo, c_hi = st.block_bootstrap_ratio(np.full(100, 7.0), np.full(100, 10.0), n_boot=200)
    assert abs(c_lo - 0.7) < 1e-12 and abs(c_hi - 0.7) < 1e-12


def test_remaining_fraction_is_the_share_of_variance_left():
    # S at every mark = C_{D-1} except the last mark sits halfway (in log) to C_D
    obs = []
    for k in range(60):
        c0, c = 100.0, 100.0 * math.exp(0.01 if k % 2 else -0.01)
        s = [c0] * 6 + [math.sqrt(c0 * c)]
        obs.append({"k": k + 1, "date": f"d{k}", "c0": c0, "c": c, "s": s})
    f = st.remaining_fraction({"X": obs}, ["X"])
    assert all(abs(v - 1.0) < 1e-12 for v in f[:6])
    assert abs(f[6] - 0.25) < 1e-12        # half the log move left -> a quarter of the variance


def test_interval_score_matches_the_definition():
    q, y = np.array([0.01, 0.01]), np.array([0.005, 0.03])
    got = st.interval_scores(q, y)
    a = 1 - st.TAU
    assert np.allclose(got, [0.02, 0.02 + (2 / a) * 0.02])


def test_load_cboe_parses_both_csv_shapes(tmp_path):
    p = tmp_path / "VIX.csv"
    p.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2024,13.0,14.0,12.5,13.2\n01/03/2024,13.2,14.4,13.0,bad\n")
    assert st.load_cboe(str(p)) == {"2024-01-02": 13.2}
    p2 = tmp_path / "GVZ.csv"
    p2.write_text("DATE,GVZ\n01/02/2024,15.5\n")
    assert st.load_cboe(str(p2)) == {"2024-01-02": 15.5}
