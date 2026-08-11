# Regression test for the 2026-08-06 full-code-review finding "BS put-delta
# branch" (analyze.py's black_scholes near-expiry/zero-vol intrinsic
# fallback). The fallback branch (T <= 0.001 or sigma <= 0) computed delta
# and prob_itm with call-only logic ("1.0 if S > K else 0.0") regardless of
# option_type, so a deep-ITM put near expiry (S < K) reported delta=0.0 and
# prob_itm=0.0 -- exactly backwards -- while a deep-OTM put reported
# delta=1.0 and prob_itm=100.0. This feeds composite_score's prob-of-profit
# term for CSP and iron-condor scoring (analyze.py's bs_sp/bs_down/bs_p call
# sites), so a same-day-expiry or missing-IV put leg could score as if it
# were certain to lose (OTM misread as ITM) or certain to win (ITM misread
# as OTM). Offline-safe: pure function, no network.
from analyze import black_scholes


def test_itm_put_near_expiry_has_negative_delta_and_full_prob_itm():
    # S=90 < K=110: deep ITM put, 8.6-minute-to-expiry fallback (T<=0.001).
    r = black_scholes(90, 110, 0.0005, 0.05, 0.2, option_type='put')
    assert r["delta"] == -1.0
    assert r["prob_itm"] == 100.0
    assert r["price"] == 20.0


def test_otm_put_near_expiry_has_zero_delta_and_zero_prob_itm():
    # S=120 > K=110: deep OTM put, same near-expiry fallback.
    r = black_scholes(120, 110, 0.0005, 0.05, 0.2, option_type='put')
    assert r["delta"] == 0.0
    assert r["prob_itm"] == 0.0
    assert r["price"] == 0.0


def test_itm_put_zero_sigma_uses_the_same_fallback():
    # Missing/zero IV data hits the same branch (sigma<=0), independent of T.
    r = black_scholes(90, 110, 30 / 365.0, 0.05, 0.0, option_type='put')
    assert r["delta"] == -1.0
    assert r["prob_itm"] == 100.0


def test_call_branch_unchanged_itm_and_otm():
    # The call side of this fallback was already correct -- pin it so a
    # future edit to the shared branch can't silently break calls instead.
    itm_call = black_scholes(120, 110, 0.0005, 0.05, 0.2, option_type='call')
    assert itm_call["delta"] == 1.0
    assert itm_call["prob_itm"] == 100.0
    assert itm_call["price"] == 10.0

    otm_call = black_scholes(90, 110, 0.0005, 0.05, 0.2, option_type='call')
    assert otm_call["delta"] == 0.0
    assert otm_call["prob_itm"] == 0.0
    assert otm_call["price"] == 0.0
