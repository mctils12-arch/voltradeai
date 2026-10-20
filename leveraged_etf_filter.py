"""leveraged_etf_filter.py — ONE leveraged / inverse ETF exclusion for every
options path ([REPAIR] 2026-10-20: coverage repair of an EXISTING rule).

The live sleeve-by-underlying read (research/experiments.md 2026-10-20
~21:25Z) found MUU, NVDL and METU — 2x single-stock funds — traded as
short-premium underlyings for −$2,001 on 17 fills (22% of the options
sleeve's realized loss on 3 of 85 names). csp_universe.py blocked
leveraged/inverse ETFs for the CSP path by a static ticker list written
before the 2024–25 single-stock 2x wave, its ticker-pattern heuristic
matched nothing real, and options_scanner.py's multi-leg path applied no
exclusion at all — while its Stage 2 ranks candidates by the day's absolute
move, which PROMOTES 2x funds by construction (they move twice their
underlying).

The rule's rationale is the CSP list's own: daily-rebalance decay, wide
spreads and no IV history make premium selling on these structurally
unprofitable. This module decides from the fund's NAME (Alpaca /v2/assets,
cached by bot_engine._get_full_universe into universe_names.json — no extra
network call), with the static seed when no name is known. HONESTY: a name
that merely contains "Daily", "Long" or "Short" (Daily Journal, Long-Term
Treasury, Short Treasury Bond) is NOT leveraged — the patterns require the
leverage vocabulary (a factor like 2X / -1X, ProShares Ultra / Short,
"Daily … Bull/Bear/Long/Short", "Leveraged", "Inverse"). Every refusal is
recorded (voltrade_leveraged_refusals.jsonl: date, ticker, rule, price,
path) so the rule earns a measurable prevention-P&L like any other.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

try:
    from storage_config import DATA_DIR
except ImportError:  # pragma: no cover
    DATA_DIR = "/tmp"

NAMES_CACHE_PATH = os.path.join(DATA_DIR, "universe_names.json")
REFUSALS_PATH = os.path.join(DATA_DIR, "voltrade_leveraged_refusals.jsonl")
NAMES_TTL_SEC = 3600  # the names file is rewritten daily; re-read hourly
logger = logging.getLogger("leveraged_etf_filter")
RULE = "leveraged_etf"

# Seed: the leveraged / inverse section csp_universe.CSP_BLOCKED_TICKERS
# carried since 2026-05 (now shared from here) plus the single-stock funds the
# live ledger already paid for.
KNOWN_LEVERAGED_TICKERS: FrozenSet[str] = frozenset({
    "TQQQ", "SQQQ", "UPRO", "SPXU", "UVXY", "SVXY", "TMF", "TMV",
    "SOXL", "SOXS", "TNA", "TZA", "LABU", "LABD", "SPXL", "SPXS",
    "FAS", "FAZ", "YINN", "YANG", "JNUG", "JDST", "NUGT", "DUST",
    "BOIL", "KOLD", "UGL", "GLL", "UCO", "SCO", "UVIX", "SVIX",
    "MUU", "NVDL", "METU",
})

_PATTERNS: Tuple[re.Pattern, ...] = (
    # a leverage factor in the name: "2X", "3x", "-1X", "1.5x", "0.5x"
    re.compile(r"(?<![A-Za-z0-9.])-?\d(?:\.\d+)?\s?[xX](?![A-Za-z0-9])"),
    # ProShares Ultra / UltraPro / UltraShort / Short (−1x) — issuer-qualified so
    # "Ultra Clean Holdings" or "JPMorgan Ultra-Short Income" are not caught;
    # "Short-Term" (VIX Short-Term Futures, 1x) is excluded by the lookahead
    re.compile(r"\bproshares\b.*\b(?:ultra\w*|short)\b(?!-term)", re.I),
    # Direxion "Daily … Bull/Bear"; GraniteShares / T-Rex / Defiance / Tradr "2x Long … Daily"
    re.compile(r"\bdaily\b.*\b(?:bull|bear|long|short)\b", re.I),
    re.compile(r"\b(?:bull|bear|long|short)\b.*\bdaily\b", re.I),
    re.compile(r"\b(?:leveraged|inverse)\b", re.I),
)


def is_leveraged_or_inverse(symbol: str, name: Optional[str] = None) -> bool:
    """True for a leveraged or inverse fund by its NAME; the seed list when no
    name is known. Never guesses from the ticker alone."""
    sym = (symbol or "").strip().upper()
    if sym in KNOWN_LEVERAGED_TICKERS:
        return True
    if not name:
        return False
    return any(p.search(name) for p in _PATTERNS)


_lock = threading.Lock()
_cache: Dict[str, object] = {"at": 0.0, "path": None, "names": {}, "excluded": KNOWN_LEVERAGED_TICKERS}


def load_universe_names(path: Optional[str] = None, now: Optional[float] = None) -> Dict[str, str]:
    """{SYMBOL: name} from the names cache (hourly in-process TTL); {} when the
    file is missing or unreadable — the seed still applies."""
    _refresh(path, now)
    return _cache["names"]  # type: ignore[return-value]


def leveraged_symbols(names: Optional[Dict[str, str]] = None) -> FrozenSet[str]:
    """Every symbol to exclude: the seed ∪ the cached names that match. With
    explicit `names` (tests) the file is not consulted."""
    if names is not None:
        return _excluded_from(names)
    _refresh(None, None)
    return _cache["excluded"]  # type: ignore[return-value]


def _excluded_from(names: Dict[str, str]) -> FrozenSet[str]:
    out = set(KNOWN_LEVERAGED_TICKERS)
    for sym, name in names.items():
        if is_leveraged_or_inverse(sym, name):
            out.add(str(sym).upper())
    return frozenset(out)


def _refresh(path: Optional[str], now: Optional[float]) -> None:
    path = path or NAMES_CACHE_PATH
    now = time.time() if now is None else now
    with _lock:
        if _cache["path"] == path and now - float(_cache["at"]) < NAMES_TTL_SEC:  # type: ignore[arg-type]
            return
        names: Dict[str, str] = {}
        try:
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                names = {str(k).upper(): str(v or "") for k, v in raw.items()}
        except Exception as e:  # missing until the first daily universe refresh: the seed applies
            logger.debug("universe names unavailable (%s: %s) — seed list only", type(e).__name__, e)
            names = {}
        _cache.update(at=now, path=path, names=names, excluded=_excluded_from(names))


def write_universe_names(assets: Iterable[dict], path: Optional[str] = None) -> int:
    """Called by bot_engine._get_full_universe with the /v2/assets rows it
    already fetched: writes {SYMBOL: name} atomically and invalidates the
    in-process cache. Returns the count written; never raises (0 on failure)."""
    path = path or NAMES_CACHE_PATH
    try:
        names: Dict[str, str] = {}
        for a in assets:
            if not isinstance(a, dict):
                continue
            sym = str(a.get("symbol") or "").strip().upper()
            if sym:
                names[sym] = str(a.get("name") or "")
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(names, f)
        os.replace(tmp, path)
        with _lock:
            _cache.update(at=0.0, path=None)
        return len(names)
    except Exception as e:
        logger.warning("could not write universe names to %s (%s: %s)", path, type(e).__name__, e)
        return 0


_recorded: set = set()


def record_refusal(symbol: str, price: float, path_name: str, refusals_path: Optional[str] = None,
                   now: Optional[float] = None) -> bool:
    """Counterfactual record for one refused candidate — once per symbol per
    UTC day per process (append-only JSONL; a scorer dedups on date+ticker).
    Returns True when a line was written. Never raises."""
    refusals_path = refusals_path or REFUSALS_PATH
    now = time.time() if now is None else now
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    key = (day, (symbol or "").upper(), path_name)
    with _lock:
        if key in _recorded:
            return False
        _recorded.add(key)
    try:
        rec = {"date": day, "ticker": key[1], "rule": RULE, "price": round(float(price or 0), 4), "path": path_name,
               "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}
        with open(refusals_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return True
    except Exception as e:
        logger.warning("could not record the %s refusal of %s (%s: %s)", RULE, key[1], type(e).__name__, e)
        return False


def exclude_leveraged(symbols: Iterable[str], names: Optional[Dict[str, str]] = None) -> List[str]:
    """`symbols` minus the excluded set, order kept."""
    ex = leveraged_symbols(names)
    return [s for s in symbols if str(s).upper() not in ex]
