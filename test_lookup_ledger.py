"""lookup_ledger: recording, dedup, outcome settlement vs SPY, the report."""
import lookup_ledger as L


def _out(sym="NVDA", at="2026-10-20T15:00:00Z", price=100.0, chg=-6.0, score=70, ms=63, stopped=None):
    return {"symbol": sym, "at": at, "errors": {},
            "quick": {"price": price, "change_pct": chg}, "deep": {"score": score},
            "regime": {"label": "NEUTRAL", "min_score": ms}, "held": False,
            "decision": {"verdict": f"stopped at {stopped}" if stopped else "would place it", "stopped_at": stopped}}


def _bars(start_day, closes):
    return [{"t": f"2026-10-{start_day + i:02d}T04:00:00Z", "c": c} for i, c in enumerate(closes)]


def test_entry_rejects_unrecordable(monkeypatch):
    assert L.entry_from_lookup({"symbol": "X", "errors": {"symbol": "bad"}}) is None
    assert L.entry_from_lookup({"symbol": "X", "errors": {}, "quick": {}, "decision": {}}) is None
    e = L.entry_from_lookup(_out())
    assert e["verdict"] == "would place it" and e["outcomes"] == {} and e["change_pct"] == -6.0


def test_record_dedups_within_window(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LEDGER_PATH", str(tmp_path / "l.json"))
    assert L.record(_out(at="2026-10-20T15:00:00Z"))
    assert not L.record(_out(at="2026-10-20T17:00:00Z"))      # same symbol, 2h later
    assert L.record(_out(sym="AMD", at="2026-10-20T17:00:00Z"))  # other symbol
    assert L.record(_out(at="2026-10-21T15:00:00Z"))            # a day later
    assert len(L.load()) == 3


def test_record_never_raises(monkeypatch):
    monkeypatch.setattr(L, "LEDGER_PATH", "/nonexistent-dir/x/l.json")
    assert L.record(_out()) is False


def test_outcomes_settle_only_horizons_with_bars():
    row = L.entry_from_lookup(_out())          # d0 = 2026-10-20 (bar 20 is day 0)
    bars = _bars(19, [99, 100, 110, 111, 112, 113, 114])   # 19,20,21,22,23,24,25
    spy = _bars(19, [400, 400, 404, 404, 404, 404, 408])
    res = L.outcomes_from_bars(row, bars, spy)
    assert set(res) == {"+1d", "+5d"}                        # 20 trading days not there
    assert res["+1d"]["ret_pct"] == 10.0                     # 110 vs lookup price 100
    assert res["+1d"]["cc_pct"] == 10.0 and res["+1d"]["spy_cc_pct"] == 1.0
    assert res["+1d"]["excess_pct"] == 9.0
    assert res["+5d"]["end_date"] == "2026-10-25"


def test_outcomes_need_a_base_bar():
    row = L.entry_from_lookup(_out(at="2026-10-18T15:00:00Z"))
    # first bar is AFTER the look-up date: no base close to measure from
    assert L.outcomes_from_bars(row, _bars(19, [100, 101, 102]), []) == {}


def test_backfill_fills_and_gives_up(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    monkeypatch.setattr(L, "LEDGER_PATH", str(tmp_path / "l.json"))
    L.record(_out(at="2026-10-20T15:00:00Z"))
    L.record(_out(sym="OLD", at="2026-06-01T15:00:00Z"))
    calls = []

    def fetch(sym, start, end):
        calls.append(sym)
        if sym == "OLD":
            return []
        return _bars(19, [99, 100, 110, 111, 112, 113, 114]) if sym == "NVDA" else _bars(19, [400] * 7)

    st = L.backfill(fetch=fetch, now=datetime(2026, 10, 26, tzinfo=timezone.utc))
    rows = {r["symbol"]: r for r in L.load()}
    assert set(rows["NVDA"]["outcomes"]) == {"+1d", "+5d"}
    assert all(v == {"no_price_data": True} for v in rows["OLD"]["outcomes"].values())
    assert st["gave_up"] == 3 and st["updated"] == 2
    # idempotent: a second run fetches nothing new for settled rows
    calls.clear()
    L.backfill(fetch=fetch, now=datetime(2026, 10, 26, tzinfo=timezone.utc))
    assert "OLD" not in calls


def test_report_groups_and_flags_thin():
    rows = []
    for i in range(3):
        r = L.entry_from_lookup(_out(sym=f"S{i}", chg=-7.0, score=70))
        r["outcomes"] = {"+5d": {"ret_pct": -4.0 + i, "excess_pct": -5.0 + i}}
        rows.append(r)
    r2 = L.entry_from_lookup(_out(sym="Z", chg=1.0, score=50, stopped="admission"))
    rows.append(r2)
    rep = L.report(rows)
    assert rep["total"] == 4 and rep["pending"] == 4
    cell = rep["admitted_by_move"]["down >=5%"]["+5d"]
    assert cell["n"] == 3 and cell["thin"] is True
    assert cell["mean_ret_pct"] == -3.0 and cell["win_rate"] == 0.0
    assert "flat/up" not in rep["admitted_by_move"]          # score 50 < 63: not admitted
    assert "stopped at admission" in rep["by_verdict"]
    assert "not a random sample" in rep["caveat"]
