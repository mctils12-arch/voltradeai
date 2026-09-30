"""Ratchet: IV-crush score must not run on a made-up ATM IV (hv20/25% fallback)."""
import instrument_selector as isel


def _run(monkeypatch, atm_iv):
    isel._intel_cache.clear()
    calls = []
    monkeypatch.setattr(isel, "compute_iv_crush_score",
                        lambda iv, e: calls.append(iv) or (50, 10.0, "x"))
    monkeypatch.setattr(isel, "_safe_call",
                        lambda fn, *a, default=None, label="": fn(*a) if label == "compute_iv_crush_score" else default)
    td = {"score": 10, "earnings_intel": {"days_to_earnings": 5},
          "vol_metrics": {"hv20": 40}}
    if atm_iv is not None:
        td["atm_iv"] = atm_iv
    out = isel.get_instrument_intelligence("ZZZT", 100.0, td)
    return out, calls


def test_missing_iv_skips_crush(monkeypatch):
    out, calls = _run(monkeypatch, None)
    assert calls == []
    assert out["iv_crush_score"] is None
    assert any("no observed ATM IV" in s for s in out["fns_skipped"])


def test_observed_iv_still_scores(monkeypatch):
    out, calls = _run(monkeypatch, 0.55)
    assert calls == [0.55]
    assert out["iv_crush_score"] == 50
