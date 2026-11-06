"""
dealer_gamma.py — Edge P1's short-premium filter (2026-11-06). Pins: the rank is
gate 2's formula bit-for-bit; the server's squeeze_gex cache is read as the
server writes it; stale / thin / missing data makes the filter INACTIVE (never
blocks); only SOLD premium is ever skipped and one-sided short puts are only
logged by default (gate 3); every skip / would-skip is logged once per ET day
(JSON line + WARNING + scan field) with a plain-English reason; the Tier-1 and
scanner wiring; no raw GEX value in the outward state.
"""
import gzip
import json
import logging
import os
import sys
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "scripts"))   # appended: never shadow a root module (source_validation)
import dealer_gamma as dg  # noqa: E402
import edge_p1_dealer_gamma as p1  # noqa: E402

ET = ZoneInfo("America/New_York")
REAL_DATA_DIR = dg._data_dir          # captured at collection, before conftest patches it
CFG = {"enabled": True, "low_rank": 0.2, "max_stale_sessions": 3, "enforce_short_puts": False}


def rows_ending(last_day: str, n: int = 260, last_gex: float = -1.0, seed: int = 0):
    """n archive rows on consecutive weekdays ending at last_day; the last GEX
    value is `last_gex` (the others are N(0,1) + 5, so -1 ranks at the bottom)."""
    rng = np.random.default_rng(seed)
    days, d = [], date.fromisoformat(last_day)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d = date.fromordinal(d.toordinal() - 1)
    days.reverse()
    vals = list(rng.normal(5, 1, n - 1)) + [last_gex]
    return [{"date": x, "gex": float(v), "fetched_at": "T"} for x, v in zip(days, vals)]


@pytest.fixture
def files(tmp_path):
    paths = {"state": str(tmp_path / "state.json"), "decisions": str(tmp_path / "decisions.jsonl")}
    with patch.object(dg, "_paths", return_value=paths):
        yield paths


def _lines(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


# ── rank math ───────────────────────────────────────────────────────────────
def test_last_rank_is_gate2s_trailing_pct_rank_bit_for_bit():
    vals = list(np.random.default_rng(3).normal(size=600))
    full = p1.trailing_pct_rank(vals, window=252)
    for t in (251, 300, 599):
        assert dg.last_rank(vals[:t + 1]) == full[t]
    assert dg.last_rank(vals[:251]) is None                     # 251 rows: still burn-in
    # the server fixture (squeezeGex.test.ts): window 5 over [5,1,2,3,4] -> 0.75; ties -> 0.5
    assert dg.last_rank([5, 1, 2, 3, 4], window=5) == 0.75
    assert dg.last_rank([2, 2, 2, 2, 2], window=5) == 0.5
    assert dg.last_rank([1, 2, 3, 4, 5], window=5) == 1.0 and dg.last_rank([5, 4, 3, 2, 1], window=5) == 0.0


# ── reading the server's cache ──────────────────────────────────────────────
def _write_year(d, year, rows, gz=False):
    os.makedirs(d, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if gz:
        with gzip.open(os.path.join(d, f"{year}.jsonl.gz"), "wt") as f:
            f.write(text)
    else:
        with open(os.path.join(d, f"{year}.jsonl"), "w") as f:
            f.write(text)


def test_read_rows_reads_plain_and_gz_years_sorted_deduped_and_skips_junk(tmp_path):
    d = str(tmp_path / "squeeze_gex")
    _write_year(d, "2025", [{"date": "2025-12-31", "price": 1, "dix": 0.4, "gex": 7.0, "fetched_at": "a"}], gz=True)
    _write_year(d, "2026", [{"date": "2026-01-05", "gex": 9.0}, {"date": "2026-01-02", "gex": 8.0}])
    with open(os.path.join(d, "2026.jsonl"), "a") as f:
        f.write("not json\n" + json.dumps({"date": "bad", "gex": 1}) + "\n" + json.dumps({"date": "2026-01-06"}) + "\n")
    # a gz copy of a year that also has a plain file is ignored (the plain file is current)
    _write_year(d, "2026", [{"date": "2026-01-02", "gex": -999.0}], gz=True)
    with open(os.path.join(d, "notes.txt"), "w") as f:
        f.write("ignored")
    rows = dg.read_rows(base=str(tmp_path))
    assert [(r["date"], r["gex"]) for r in rows] == [("2025-12-31", 7.0), ("2026-01-02", 8.0), ("2026-01-05", 9.0)]
    assert dg.read_rows(base=str(tmp_path / "missing")) == []


def test_archive_dir_mirrors_the_servers_archive_base_dir(monkeypatch):
    monkeypatch.setattr(dg, "_data_dir", REAL_DATA_DIR)
    monkeypatch.setenv("DATA_DIR", "/srv/x")
    assert dg.archive_dir() == "/srv/x/datacore_archive/squeeze_gex"
    monkeypatch.delenv("DATA_DIR")
    import storage_config
    assert dg.archive_dir() == os.path.join(storage_config.DATA_DIR, "datacore_archive", "squeeze_gex")
    assert dg.archive_dir("/b") == "/b/squeeze_gex"


# ── staleness -> inactive ───────────────────────────────────────────────────
def test_sessions_since_counts_nyse_sessions():
    assert dg.sessions_since("2026-11-06", date(2026, 11, 9)) == 1      # Fri -> Mon
    assert dg.sessions_since("2026-11-06", date(2026, 11, 8)) == 0      # weekend
    assert dg.sessions_since("2026-11-25", date(2026, 11, 27)) == 1     # Thanksgiving is not a session
    assert dg.sessions_since("2026-11-02", date(2026, 11, 6)) == 4
    assert dg.sessions_since("garbage", date(2026, 11, 6)) == 30


def test_fresh_low_rank_is_active_low_gamma():
    st = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    assert st["active"] and st["lowGamma"] and not st["stale"] and st["ageSessions"] == 1
    assert st["rank252"] == 0.0 and st["asOf"] == "2026-11-05" and st["threshold"] == 0.2
    assert "SqueezeMetrics" in st["source"] and "not ours" in st["source"]


def test_staleness_over_three_sessions_makes_the_filter_inactive():
    ok = dg.compute_state(rows_ending("2026-11-03"), date(2026, 11, 6), CFG)      # 3 sessions: still active
    assert ok["active"] and ok["ageSessions"] == 3
    st = dg.compute_state(rows_ending("2026-11-02"), date(2026, 11, 6), CFG)      # 4 sessions
    assert st["stale"] and not st["active"] and not st["lowGamma"]
    assert "inactive" in st["reason"] and "4 sessions old" in st["reason"]
    d = dg.decide("iron_condor", "sell", st)
    assert d["action"] == "allow" and "inactive" in d["reason"]


def test_thin_missing_or_disabled_data_is_inactive():
    thin = dg.compute_state(rows_ending("2026-11-05", n=200), date(2026, 11, 6), CFG)
    assert not thin["active"] and thin["rank252"] is None and "252" in thin["reason"]
    none = dg.compute_state([], date(2026, 11, 6), CFG)
    assert not none["active"] and none["asOf"] is None and "inactive" in none["reason"]
    off = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), {**CFG, "enabled": False})
    assert not off["active"] and "switched off" in off["reason"]
    assert dg.decide("iron_condor", "sell", off)["action"] == "allow"


def test_get_state_fails_open_on_a_read_error(files):
    with patch.object(dg, "read_rows", side_effect=OSError("disk")):
        st = dg.get_state(force=True)
    assert not st["active"] and "error" in st["reason"]


def test_get_state_reads_the_cache_and_writes_the_public_state_file(files, tmp_path):
    base = tmp_path / "data"
    d = base / "datacore_archive" / "squeeze_gex"
    rows = rows_ending("2026-11-05")
    _write_year(str(d), "2026", [r for r in rows if r["date"] >= "2026"])
    _write_year(str(d), "2025", [r for r in rows if r["date"] < "2026"], gz=True)
    now = datetime(2026, 11, 6, 10, 0, tzinfo=ET)
    with patch.object(dg, "_data_dir", return_value=str(base)):
        st = dg.get_state(force=True, now=now)
        assert st["active"] and st["lowGamma"] and st["rows"] == 260
        on_disk = json.load(open(files["state"]))
        assert on_disk["asOf"] == "2026-11-05" and on_disk["label"].startswith("dealer gamma: low (rank 0.00)")
        assert "gex" not in on_disk and on_disk["gexNegative"] is True
        # cached: the same dict comes back without re-reading
        with patch.object(dg, "read_rows", side_effect=AssertionError("re-read")):
            assert dg.get_state(now=now)["asOf"] == "2026-11-05"


# ── only short premium is filtered ──────────────────────────────────────────
def test_structure_classification_covers_every_live_strategy_name():
    assert dg.structure_of("iron_condor", "sell") == "short_two_sided"         # Setup 1 / Setup 3 condors
    assert dg.structure_of("qqq_iron_condor", "sell") == "short_two_sided"
    assert dg.structure_of("bull_put_credit_spread", "sell") == "short_put"    # Setup 2 (VXX panic)
    assert dg.structure_of("sell_cash_secured_put", "sell") == "short_put"     # Tier 1 SELL_CSP
    assert dg.structure_of("covered_call", "sell") == "short_call"
    for long_name in ("buy_put", "buy_call", "bull_call_spread", "bear_put_spread"):
        assert dg.structure_of(long_name, "sell") == "long"                    # name wins over a stray side
    assert dg.structure_of("mystery_structure", "sell") == "short_two_sided"   # sold + unknown -> enforced
    assert dg.structure_of("mystery_structure", "buy") == "long"
    assert dg.structure_of(None, None) == "long"


def test_decide_skips_only_sold_premium_and_logs_csps_by_default():
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    assert dg.decide("buy_put", "buy", low)["action"] == "allow"
    assert dg.decide("buy_call", "buy", low)["shortPremium"] is False
    condor = dg.decide("iron_condor", "sell", low)
    assert condor["action"] == "skip" and condor["enforced"]
    assert condor["reason"].startswith("dealer gamma low (rank 0.00): realized vol tends to exceed implied, no new short premium")
    csp = dg.decide("sell_cash_secured_put", "sell", low)
    assert csp["action"] == "would_skip" and not csp["enforced"] and "logged only" in csp["reason"]
    enforced = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), {**CFG, "enforce_short_puts": True})
    assert dg.decide("sell_cash_secured_put", "sell", enforced)["action"] == "skip"
    normal = dg.compute_state(rows_ending("2026-11-05", last_gex=50.0), date(2026, 11, 6), CFG)
    assert not normal["lowGamma"] and dg.decide("iron_condor", "sell", normal)["action"] == "allow"
    # the threshold is configurable and strict: rank == threshold is NOT low
    at = dg.compute_state(rows_ending("2026-11-05", last_gex=50.0), date(2026, 11, 6), {**CFG, "low_rank": 1.0})
    assert at["rank252"] == 1.0 and not at["lowGamma"]


# ── logging ─────────────────────────────────────────────────────────────────
def test_record_logs_json_warning_and_pending_once_per_day(files, caplog):
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    d = dg.decide("iron_condor", "sell", low)
    t1 = datetime(2026, 11, 6, 10, 0, tzinfo=ET)
    with caplog.at_level(logging.WARNING, logger="dealer_gamma"):
        rec = dg.record(d, "nvda", "options_scanner.get_options_trades", setup="earnings_iv_crush",
                        strategy="iron_condor", score=71.0, price=190.0, now=t1)
    assert rec["new"] and rec["ticker"] == "NVDA" and rec["action"] == "skip" and rec["day"] == "2026-11-06"
    assert any("DEALER-GAMMA SKIP NVDA" in m and "no new short premium" in m for m in caplog.messages)
    assert dg.record(d, "NVDA", "x", setup="earnings_iv_crush", now=t1) is None          # same day: once
    assert dg.record(d, "NVDA", "x", setup="high_iv_premium_sale", now=t1)["new"]        # other setup: new
    assert dg.record({"action": "allow"}, "NVDA", "x", now=t1) is None                   # allows never log
    t2 = datetime(2026, 11, 9, 10, 0, tzinfo=ET)
    assert dg.record(d, "NVDA", "x", setup="earnings_iv_crush", now=t2)["new"]           # next ET day: again
    lines = _lines(files["decisions"])
    assert [(x["ticker"], x["setup"], x["day"]) for x in lines] == [
        ("NVDA", "earnings_iv_crush", "2026-11-06"), ("NVDA", "high_iv_premium_sale", "2026-11-06"),
        ("NVDA", "earnings_iv_crush", "2026-11-09")]
    assert lines[0]["rank252"] == 0.0 and lines[0]["asOf"] == "2026-11-05" and lines[0]["enforced"] is True
    assert [r["setup"] for r in dg.drain()] == ["earnings_iv_crush", "high_iv_premium_sale", "earnings_iv_crush"]
    assert dg.drain() == []


def test_check_never_raises_and_allows_on_failure(files):
    with patch.object(dg, "get_state", side_effect=RuntimeError("boom")):
        d = dg.check("iron_condor", "sell", "AAPL", "src")
    assert d["action"] == "allow" and "error" in d["reason"]


def test_shadow_slot_bounds_writes_per_scan_and_drain_resets():
    assert [dg.shadow_slot() for _ in range(5)] == [True, True, True, False, False]
    dg.drain()
    assert dg.shadow_slot() is True


def test_scan_field_carries_state_without_raw_gex_and_drains(files):
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    with patch.object(dg, "get_state", return_value=low):
        dg.check("iron_condor", "sell", "AMD", "src", setup="high_iv_premium_sale")
        f = dg.scan_field()
    assert "gex" not in f["state"] and f["state"]["rank252"] == 0.0 and f["state"]["lowGamma"] is True
    assert f["state"]["label"] == ("dealer gamma: low (rank 0.00) — no new condors / strangles; "
                                   "cash-secured puts and put spreads logged only")
    assert [(r["ticker"], r["action"]) for r in f["decisions"]] == [("AMD", "skip")]
    with patch.object(dg, "get_state", return_value=low):
        assert dg.scan_field()["decisions"] == []
    with patch.object(dg, "public_state", side_effect=ValueError("x")):
        assert "error" in dg.scan_field()


def test_labels_are_plain_english():
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), {**CFG, "enforce_short_puts": True})
    assert dg.label(low) == "dealer gamma: low (rank 0.00) — no new short premium"
    normal = dg.compute_state(rows_ending("2026-11-05", last_gex=50.0), date(2026, 11, 6), CFG)
    assert dg.label(normal) == "dealer gamma: normal (rank 1.00) — no restriction"
    assert dg.label(dg.compute_state([], date(2026, 11, 6), CFG)).startswith("dealer gamma: filter inactive")


def test_config_keys_live_in_system_config_with_the_preregistered_threshold():
    import system_config as sc
    assert sc.BASE_CONFIG["DEALER_GAMMA_FILTER_ENABLED"] is True
    assert sc.BASE_CONFIG["DEALER_GAMMA_LOW_RANK"] == p1.GATE3_LOW_RANK == 0.2
    assert sc.BASE_CONFIG["DEALER_GAMMA_MAX_STALE_SESSIONS"] == 3
    assert sc.BASE_CONFIG["DEALER_GAMMA_ENFORCE_SHORT_PUTS"] is False
    assert dg.config() == {"enabled": True, "low_rank": 0.2, "max_stale_sessions": 3, "enforce_short_puts": False}


# ── wiring: Tier 1 ──────────────────────────────────────────────────────────
UNIVERSE = ["AMD", "AAPL", "MSFT", "NVDA", "AMZN"]


def _run_t1(state):
    import tiered_strategy as ts
    logged = []
    ctx = ts.TierContext(equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0,
                         positions=[], vxx_ratio=1.0, spy_vs_ma50=1.0)
    with patch("tiered_strategy._get_t1_universe", return_value=UNIVERSE), \
         patch("csp_universe._load_layer2_cache", return_value={}), \
         patch("regime_engine.engine_allowed", return_value=False), \
         patch("tiered_strategy.log_csp_gate_shadow", side_effect=lambda rej: logged.extend(rej) or {}), \
         patch.object(dg, "get_state", return_value=state):
        actions = ts.tier1_csp_core(ctx)
    return actions, logged


def test_tier1_inactive_filter_is_the_baseline(files):
    actions, logged = _run_t1(dg.compute_state([], date(2026, 11, 6), CFG))
    assert [a.ticker for a in actions] == UNIVERSE[:3] and logged == []
    assert {a.metadata["dealer_gamma"]["action"] for a in actions} == {"allow"}
    assert _lines(files["decisions"]) == []


def test_tier1_low_gamma_logs_would_skip_for_the_returned_csps_only(files):
    base, _ = _run_t1(dg.compute_state([], date(2026, 11, 6), CFG))
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    actions, logged = _run_t1(low)
    assert [a.ticker for a in actions] == [a.ticker for a in base]                 # nothing blocked
    assert [a.size_pct for a in actions] == [a.size_pct for a in base]
    assert [(r["ticker"], r["decision"]) for r in logged] == [(t, "would_reject_dealer_gamma") for t in UNIVERSE[:3]]
    assert [(x["ticker"], x["action"], x["source"]) for x in _lines(files["decisions"])] == [
        (t, "would_skip", "tiered_strategy.tier1_csp_core") for t in UNIVERSE[:3]]
    assert all(a.metadata["dealer_gamma"]["action"] == "would_skip" for a in actions)


def test_tier1_enforced_puts_propose_nothing_and_log_the_three_it_would_have(files):
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), {**CFG, "enforce_short_puts": True})
    actions, logged = _run_t1(low)
    assert actions == []
    assert [(r["ticker"], r["decision"]) for r in logged] == [(t, "rejected_dealer_gamma") for t in UNIVERSE[:3]]
    assert [x["action"] for x in _lines(files["decisions"])] == ["skip"] * 3
    assert "no new short premium" in logged[0]["reason"]


# ── wiring: the options scanner ─────────────────────────────────────────────
def _opps():
    return [
        {"ticker": "NVDA", "setup": "earnings_iv_crush", "score": 80, "price": 190, "action_label": "SELL IRON CONDOR",
         "options_strategy": "iron_condor", "reasoning": "t", "source": "options_scanner", "side": "sell"},
        {"ticker": "SPY", "setup": "vxx_panic_put_sale", "score": 78, "price": 680, "action_label": "SELL PUT SPREAD",
         "options_strategy": "bull_put_credit_spread", "reasoning": "t", "source": "options_scanner", "side": "sell"},
        {"ticker": "TSLA", "setup": "high_iv_premium_sale", "score": 75, "price": 420, "action_label": "SELL IRON CONDOR",
         "options_strategy": "iron_condor", "reasoning": "t", "source": "options_scanner", "side": "sell"},
    ]


def _run_scanner(state):
    import options_scanner as osc
    shadow = []
    scan = {"opportunities": _opps(), "setup_counts": {}, "vxx_ratio": 1.31, "regime": "PANIC",
            "scanned": 3, "duration_secs": 0.1}
    with patch("options_scanner.scan_options", return_value=scan), \
         patch("market_calendar.should_skip_new_options", return_value=(False, "")), \
         patch("vol_surface.get_surface_score", return_value={}), \
         patch("shadow_portfolio.log_candidate", side_effect=lambda **kw: shadow.append(kw)), \
         patch.object(dg, "get_state", return_value=state):
        trades = osc.get_options_trades(100_000, [], max_new=5, min_score=60)
    return trades, shadow


def test_scanner_low_gamma_skips_condors_and_logs_the_put_spread(files):
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    trades, shadow = _run_scanner(low)
    assert [t["ticker"] for t in trades] == ["SPY"]
    assert trades[0]["dealer_gamma"]["action"] == "would_skip"
    assert sorted((s["ticker"], s["decision"]) for s in shadow) == [
        ("NVDA", "rejected_dealer_gamma"), ("SPY", "would_reject_dealer_gamma"), ("TSLA", "rejected_dealer_gamma")]
    assert all(s["side"] == "sell" and s["trade_type"] == "options" and s["features"] == {} for s in shadow)
    assert {(x["ticker"], x["action"]) for x in _lines(files["decisions"])} == {
        ("NVDA", "skip"), ("SPY", "would_skip"), ("TSLA", "skip")}


def test_scanner_inactive_or_normal_gamma_changes_nothing(files):
    for st in (dg.compute_state([], date(2026, 11, 6), CFG),
               dg.compute_state(rows_ending("2026-11-05", last_gex=50.0), date(2026, 11, 6), CFG)):
        trades, shadow = _run_scanner(st)
        assert [t["ticker"] for t in trades] == ["NVDA", "SPY", "TSLA"] and shadow == []
    assert _lines(files["decisions"]) == []


def test_scanner_gate_never_touches_a_bought_option(files):
    import options_scanner as osc
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    opp = {"ticker": "QQQ", "setup": "x", "options_strategy": "buy_put", "side": "buy", "score": 70, "price": 500}
    with patch.object(dg, "get_state", return_value=low):
        assert osc._dealer_gamma_gate(opp, 1.0, "BULL")["action"] == "allow"
    assert "dealer_gamma" not in opp and _lines(files["decisions"]) == []


def test_scanner_shadow_writes_are_bounded_per_scan(files):
    import options_scanner as osc
    low = dg.compute_state(rows_ending("2026-11-05"), date(2026, 11, 6), CFG)
    shadow = []
    with patch.object(dg, "get_state", return_value=low), \
         patch("shadow_portfolio.log_candidate", side_effect=lambda **kw: shadow.append(kw)):
        for i in range(6):
            osc._dealer_gamma_gate({"ticker": f"T{i}", "setup": "s", "options_strategy": "iron_condor",
                                    "side": "sell", "score": 70, "price": 10}, 1.0, "BULL")
    assert len(shadow) == dg.SHADOW_WRITES_PER_SCAN and len(_lines(files["decisions"])) == 6


# ── wiring: the scan result field ───────────────────────────────────────────
def test_scan_market_returns_the_dealer_gamma_field():
    with open(os.path.join(ROOT, "bot_engine.py")) as f:
        src = f.read()
    i_ret = src.index('"data_source_errors": _source_diag,')
    i_field = src.index('"dealer_gamma": _scan_dealer_gamma(),')
    assert 0 < i_field - i_ret < 400
    import bot_engine
    with patch.object(dg, "scan_field", side_effect=RuntimeError("x")):
        assert bot_engine._scan_dealer_gamma()["decisions"] == []
    with patch.object(dg, "scan_field", return_value={"state": {}, "decisions": [1]}):
        assert bot_engine._scan_dealer_gamma()["decisions"] == [1]
