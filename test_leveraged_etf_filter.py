"""leveraged_etf_filter — ONE leveraged / inverse exclusion for every options
path ([REPAIR] 2026-10-20, coverage repair of an existing rule).

Regression for the live finding: MUU / NVDL / METU (2x single-stock funds)
were traded as short-premium underlyings for −$2,001 because csp_universe's
static list predated them, its ticker heuristic matched nothing, and
options_scanner applied no exclusion at all. These tests pin: the NAME
classifier on real fund names (and the look-alikes it must keep), the seed,
the names cache + writer, the refusal record, and BOTH consumers refusing
MUU while keeping MU.
"""
import json
import time

import pytest

import leveraged_etf_filter as lef
import csp_universe
import options_scanner

EXCLUDE = [
    ("MUU", "Direxion Daily MU Bull 2X Shares"), ("NVDL", "GraniteShares 2x Long NVDA Daily ETF"),
    ("METU", "Direxion Daily META Bull 2X Shares"), ("AAPU", "Direxion Daily AAPL Bull 2X Shares"),
    ("TSLL", "Direxion Daily TSLA Bull 2X Shares"), ("CONL", "GraniteShares 2x Long COIN Daily ETF"),
    ("MSTU", "T-Rex 2X Long MSTR Daily Target ETF"), ("MSTX", "Defiance Daily Target 2X Long MSTR ETF"),
    ("NVDX", "T-Rex 2X Long NVIDIA Daily Target ETF"), ("TSLQ", "Tradr 2X Short TSLA Daily ETF"),
    ("SMCX", "Defiance Daily Target 2X Long SMCI ETF"), ("TQQQ", "ProShares UltraPro QQQ"),
    ("SQQQ", "ProShares UltraPro Short QQQ"), ("UVXY", "ProShares Ultra VIX Short-Term Futures ETF"),
    ("SVXY", "ProShares Short VIX Short-Term Futures ETF"), ("PSQ", "ProShares Short QQQ"),
    ("SOXL", "Direxion Daily Semiconductor Bull 3X Shares"), ("SPDN", "Direxion Daily S&P 500 Bear 1X Shares"),
    ("FNGU", "MicroSectors FANG+ Index 3X Leveraged ETN"), ("BOIL", "ProShares Ultra Bloomberg Natural Gas"),
    ("LABU", "Direxion Daily S&P Biotech Bull 3x Shares"), ("TMV", "Direxion Daily 20+ Year Treasury Bear 3X Shares"),
]
KEEP = [
    ("MU", "Micron Technology Inc"), ("NVDA", "NVIDIA Corporation"), ("META", "Meta Platforms Inc"),
    ("DJCO", "Daily Journal Corp"), ("SHV", "iShares Short Treasury Bond ETF"), ("VGLT", "Vanguard Long-Term Treasury ETF"),
    ("SPY", "SPDR S&P 500 ETF Trust"), ("QQQ", "Invesco QQQ Trust"), ("VIXY", "ProShares VIX Short-Term Futures ETF"),
    ("UCTT", "Ultra Clean Holdings Inc"), ("RARE", "Ultragenyx Pharmaceutical Inc"), ("JPST", "JPMorgan Ultra-Short Income ETF"),
    ("TXN", "Texas Instruments Inc"), ("SVOL", "Simplify Volatility Premium ETF"),
    ("VXX", "iPath Series B S&P 500 VIX Short-Term Futures ETN"), ("XOM", "Exxon Mobil Corp"),
    ("HDGE", "AdvisorShares Ranger Equity Bear ETF"), ("NOBL", "ProShares S&P 500 Dividend Aristocrats ETF"),
    ("MOON", "Direxion Moonshot Innovators ETF"), ("BAR", "GraniteShares Gold Trust"), ("BX", "Blackstone Inc"),
    ("XP", "XP Inc"), ("X", "United States Steel Corp"),
]


def test_names_are_classified_by_their_leverage_vocabulary_not_by_look_alikes():
    for sym, name in EXCLUDE:
        assert lef.is_leveraged_or_inverse("ZZZZ", name), f"{sym}: {name!r} must be refused by NAME"
    for sym, name in KEEP:
        assert not lef.is_leveraged_or_inverse(sym, name), f"{sym}: {name!r} must be kept"


def test_the_seed_covers_the_names_the_ledger_already_paid_for_without_a_name():
    for sym in ("MUU", "nvdl", "METU", "TQQQ", "UVXY"):
        assert lef.is_leveraged_or_inverse(sym)
    assert not lef.is_leveraged_or_inverse("MU")
    assert not lef.is_leveraged_or_inverse("")
    assert {"TQQQ", "SQQQ", "SVXY", "MUU", "NVDL", "METU"} <= lef.KNOWN_LEVERAGED_TICKERS


@pytest.fixture
def names_file(tmp_path, monkeypatch):
    path = tmp_path / "universe_names.json"
    monkeypatch.setattr(lef, "NAMES_CACHE_PATH", str(path))
    lef._cache.update(at=0.0, path=None, names={}, excluded=lef.KNOWN_LEVERAGED_TICKERS)
    return path


def test_the_names_cache_widens_the_seed_and_a_missing_file_leaves_the_seed(names_file):
    assert lef.load_universe_names() == {}
    assert lef.leveraged_symbols() == lef.KNOWN_LEVERAGED_TICKERS
    names_file.write_text(json.dumps({"XYZQ": "Some Issuer 2X Long XYZ Daily ETF", "mu": "Micron Technology Inc", "BULL": "Daily Bull 3X"}))
    lef._cache.update(at=0.0, path=None)
    ex = lef.leveraged_symbols()
    assert {"XYZQ", "BULL"} <= ex and "MU" not in ex and lef.KNOWN_LEVERAGED_TICKERS <= ex
    assert lef.exclude_leveraged(["MU", "XYZQ", "NVDA", "MUU"]) == ["MU", "NVDA"]
    # explicit names never touch the file
    assert "ABCD" in lef.leveraged_symbols({"ABCD": "ProShares UltraPro Something"})


def test_write_universe_names_writes_symbol_to_name_and_invalidates_the_cache(names_file):
    lef.leveraged_symbols()  # warm (seed only)
    n = lef.write_universe_names([
        {"symbol": "MUU", "name": "Direxion Daily MU Bull 2X Shares", "tradable": True},
        {"symbol": "mu", "name": "Micron Technology Inc"},
        {"symbol": "", "name": "nameless"}, "junk", None,
    ])
    assert n == 2
    assert json.loads(names_file.read_text()) == {"MUU": "Direxion Daily MU Bull 2X Shares", "MU": "Micron Technology Inc"}
    assert lef.load_universe_names()["MU"] == "Micron Technology Inc"
    assert "MUU" in lef.leveraged_symbols() and "MU" not in lef.leveraged_symbols()
    assert lef.write_universe_names([{"symbol": "A", "name": "B"}], path="/nonexistent-dir/x.json") == 0


def test_record_refusal_is_one_line_per_symbol_per_day_and_never_raises(tmp_path, monkeypatch):
    path = tmp_path / "refusals.jsonl"
    monkeypatch.setattr(lef, "REFUSALS_PATH", str(path))
    lef._recorded.clear()
    now = time.time()
    assert lef.record_refusal("MUU", 61.25, "options_scanner", now=now) is True
    assert lef.record_refusal("muu", 61.30, "options_scanner", now=now + 60) is False
    assert lef.record_refusal("MUU", 61.30, "csp_universe", now=now) is True, "a different path is its own record"
    assert lef.record_refusal("MUU", 70.0, "options_scanner", now=now + 86_400) is True, "a new UTC day records again"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 3
    assert lines[0]["ticker"] == "MUU" and lines[0]["rule"] == "leveraged_etf" and lines[0]["price"] == 61.25
    assert set(lines[0]) == {"date", "ticker", "rule", "price", "path", "t"}
    assert lef.record_refusal("X", 1.0, "p", refusals_path="/nonexistent-dir/r.jsonl", now=now) is False


def test_csp_universe_layer1_refuses_a_leveraged_name_and_keeps_its_underlying(tmp_path, monkeypatch):
    monkeypatch.setattr(csp_universe, "UNIVERSE_CACHE_PATH", str(tmp_path / "universe_cache.json"))
    monkeypatch.setattr(csp_universe, "_fetch_account_equity", lambda: 100_000.0)
    monkeypatch.setattr(csp_universe, "leveraged_symbols", lambda: frozenset({"MUU", "ZZZL"}))
    snap = {
        "MU": {"dailyBar": {"c": 110.0, "v": 5_000_000}},
        "MUU": {"dailyBar": {"c": 60.0, "v": 5_000_000}},
        "ZZZL": {"dailyBar": {"c": 50.0, "v": 5_000_000}},
    }
    syms = [c[0] for c in csp_universe._layer1_hard_gates(snap_data=snap)]
    assert "MU" in syms and "MUU" not in syms and "ZZZL" not in syms
    assert "TQQQ" in csp_universe.CSP_BLOCKED_TICKERS and "MUU" in csp_universe.CSP_BLOCKED_TICKERS
    assert csp_universe._is_likely_etf_leveraged("MUU") and csp_universe._is_likely_etf_leveraged("zzzl")
    assert not csp_universe._is_likely_etf_leveraged("MU")


def test_options_scanner_refuses_leveraged_movers_records_them_and_keeps_the_underlying(tmp_path, monkeypatch):
    monkeypatch.setattr(options_scanner, "_options_candidates_cache", [])
    monkeypatch.setattr(options_scanner, "_options_candidates_time", 0.0)
    monkeypatch.setattr(lef, "leveraged_symbols", lambda names=None: frozenset({"MUU", "NVDL"}))
    monkeypatch.setattr(lef, "REFUSALS_PATH", str(tmp_path / "refusals.jsonl"))
    lef._recorded.clear()
    snap = {
        "MU": {"dailyBar": {"c": 110.0, "v": 5_000_000}, "prevDailyBar": {"c": 100.0}},     # +10%: high-IV
        "MUU": {"dailyBar": {"c": 60.0, "v": 5_000_000}, "prevDailyBar": {"c": 50.0}},      # +20%: the 2x fund
        "NVDL": {"dailyBar": {"c": 80.0, "v": 5_000_000}, "prevDailyBar": {"c": 80.1}},     # quiet: low-IV tier, also refused
        "CALM": {"dailyBar": {"c": 100.0, "v": 5_000_000}, "prevDailyBar": {"c": 100.1}},   # quiet, kept in the low-IV classification
    }
    result = options_scanner._get_options_candidates(snap_data=snap)
    syms = [r[0] for r in result]
    assert "MU" in syms and "MUU" not in syms and "NVDL" not in syms
    assert options_scanner._LAST_LEVERAGED_REFUSALS == [("MUU", 60.0)], "only a would-be candidate is a refusal"
    lines = [json.loads(l) for l in (tmp_path / "refusals.jsonl").read_text().splitlines()]
    assert [(l["ticker"], l["price"], l["path"]) for l in lines] == [("MUU", 60.0, "options_scanner")]
