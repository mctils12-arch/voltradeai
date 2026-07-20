# Regression test for the 2026-07-20 bad-print IV repair (analyze.py).
#
# Live failure this pins: at 13:41 UTC (11 minutes after the open) the
# /api/analyze/TSLA UI displayed ATM IV 389.7% against rv20 59.3 — a
# yfinance impliedVolatility bad print from wide market-open quotes — and
# derived VRP +330.4% plus a confident "sell vol" recommendation from it.
# plausible_atm_iv() is the gate that keeps such prints out of the headline
# IV and the vol surface. Offline-safe: pure function, no network.
from analyze import plausible_atm_iv


def test_rejects_the_observed_tsla_bad_print():
    # 389.7% IV vs rv20 59.3 → cap is max(150, 4*59.3)=237.2 → rejected.
    assert not plausible_atm_iv(389.7, 59.3)


def test_accepts_normal_readings():
    assert plausible_atm_iv(65.4, 59.3)     # the sane TSLA reading, same day
    assert plausible_atm_iv(22.0, 18.0)     # calm mega-cap
    assert plausible_atm_iv(45.0, 60.0)     # IV below RV (post-event crush)


def test_high_iv_names_survive_via_the_rv_ratio_bound():
    # A genuinely explosive name: rv20 90 → cap 360; IV 280 is real, keep it.
    assert plausible_atm_iv(280.0, 90.0)


def test_low_rv_names_still_get_the_150_floor_cap():
    # rv20 20 → 4x is 80, but the cap floors at 150: pre-earnings IV 120 on a
    # calm name is legitimate and must not be rejected.
    assert plausible_atm_iv(120.0, 20.0)
    assert not plausible_atm_iv(160.0, 20.0)


def test_missing_rv_uses_absolute_bounds():
    assert plausible_atm_iv(200.0, None)
    assert not plausible_atm_iv(300.0, None)
    assert not plausible_atm_iv(300.0, 0)


def test_garbage_inputs_rejected():
    assert not plausible_atm_iv(None, 50.0)
    assert not plausible_atm_iv(0.4, 50.0)      # sub-1% IV is a dead quote
    assert not plausible_atm_iv("nan-ish", 50.0)
