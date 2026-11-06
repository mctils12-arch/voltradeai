"""dealer_gamma.py — Edge P1's live input: SqueezeMetrics' published dealer-
gamma estimate (GEX) and its pre-registered trailing-252 percentile rank, plus
the short-premium entry filter that reads it (2026-11-06; refocus spec §3
step 6 "options flow/positioning: dealer gamma, implied vs realized vol").

SOURCE — NOT OURS. SqueezeMetrics' free DIX.csv (date, price, dix, gex), their
model ESTIMATE of S&P 500 dealers' gamma ($ per 1% move; it assumes dealers are
long calls and short puts). The server's existing squeeze_gex stream
(server/squeezeGex.ts, every 6 h) caches it under
<DATA_DIR>/datacore_archive/squeeze_gex/<year>.jsonl[.gz]. This module only
READS that cache: it never fetches, so the bot adds no second retriever of
their file. SqueezeMetrics' Terms of Service (squeezemetrics.com/monitor/terms,
read 2026-11-06) grant no licence to the data and say users may not
"systematically retrieve data or other content from the Platform to create or
compile, directly or indirectly, a collection, compilation, database, or
directory without written permission"; the content is "for general
information only". No CSV-specific terms are published. So: internal use only,
never redistributed, and no raw GEX value leaves this process (the scan field
and the diag probe carry the rank, the date and the verdict only).

RANK — bit-for-bit gate 2's definition (scripts/edge_p1_dealer_gamma.
trailing_pct_rank, server/squeezeGex.trailingPctRank): R_t = (count of the 252
rows ending at t with GEX < G_t + 0.5 x ties excluding itself) / 251, using
rows <= t only; undefined until 252 rows exist.

THE RULE (research/edge_preregistration.md § P1, gate 3): no NEW short premium
when the last published R is below DEALER_GAMMA_LOW_RANK (0.2). Gate 3
(scripts/edge_p1_gate3_rule_sim.py, SPY model replay 2012-05 -> 2026-10,
2026-11-06) found that rule HURT a 30-delta cash-secured-put book and roughly
halved a 1-SD strangle book's max drawdown, so one-sided short puts (CSPs, put
credit spreads) are LOGGED as "would skip" unless DEALER_GAMMA_ENFORCE_SHORT_PUTS,
while two-sided / call-side short premium (iron condors, strangles) is skipped.
Bought options are never affected.

FAILS OPEN. Filter disabled, archive missing, < 252 rows, a latest row older
than DEALER_GAMMA_MAX_STALE_SESSIONS NYSE sessions, or a read error -> the
filter is INACTIVE: it never blocks, and its state says why.

OUR OWN GEX. The open-interest sidecar (server/optionsOiArchive.ts, recording
since 2026-10-29) is gate 1's input. It holds a handful of sessions, far short
of a 252-session rank, so it is not used here and no rank is inferred from it.

Logging: every skip / would-skip is a JSON line in DEALER_GAMMA_DECISIONS_PATH,
a WARNING log line, and a pending record that bot_engine.scan_market returns as
its `dealer_gamma` field (bot.ts writes one GAMMA-SKIP / GAMMA-WOULD-SKIP audit
row per new record). Repeats are bounded: once per ET day per ticker, setup
and action. The latest state is written to DEALER_GAMMA_STATE_PATH for the
`gamma` diag probe.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

logger = logging.getLogger("dealer_gamma")

RANK_WINDOW = 252
SOURCE = "SqueezeMetrics DIX.csv (published estimate, not ours)"
STATE_TTL_S = 60.0
PENDING_MAX = 200
SHADOW_WRITES_PER_SCAN = 3      # shadow_portfolio.log_candidate re-dumps a ~45MB log per call (KNOWN BROKEN #18)
SHADOW_WINDOW_S = 120.0
SKIP_DECISION = "rejected_dealer_gamma"
WOULD_SKIP_DECISION = "would_reject_dealer_gamma"
_ET = ZoneInfo("America/New_York")
_YEAR_FILE = re.compile(r"^(\d{4})\.jsonl(\.gz)?$")

# Strategy names as the scanner, the tier engine and options_execution spell
# them. Anything sold that is not listed falls back on `side`.
SHORT_PUT_STRATEGIES = frozenset({
    "sell_cash_secured_put", "cash_secured_put", "csp", "csp_core", "sell_put", "short_put",
    "bull_put_credit_spread", "put_credit_spread", "bull_put_spread",
})
SHORT_TWO_SIDED_STRATEGIES = frozenset({
    "iron_condor", "sell_iron_condor", "qqq_iron_condor", "short_strangle", "sell_strangle",
    "short_straddle", "sell_straddle", "iron_butterfly",
})
SHORT_CALL_STRATEGIES = frozenset({
    "covered_call", "sell_covered_call", "sell_call", "short_call",
    "bear_call_credit_spread", "call_credit_spread", "bear_call_spread",
})
LONG_STRATEGIES = frozenset({
    "buy_call", "buy_put", "long_call", "long_put", "bull_call_spread", "bear_put_spread",
    "long_straddle", "long_strangle", "protective_put", "tail_hedge",
})


# ── config ──────────────────────────────────────────────────────────────────
def config() -> Dict[str, Any]:
    """The four system_config keys (config_overrides.json honoured via cfg)."""
    try:
        import system_config as _sc
        src = getattr(_sc, "cfg", None) or _sc.BASE_CONFIG
    except ImportError as e:  # never block a scan over a config import
        logger.warning(f"dealer_gamma: system_config unavailable ({e}); defaults used")
        src = {}
    return {
        "enabled": bool(src.get("DEALER_GAMMA_FILTER_ENABLED", True)),
        "low_rank": float(src.get("DEALER_GAMMA_LOW_RANK", 0.20)),
        "max_stale_sessions": int(src.get("DEALER_GAMMA_MAX_STALE_SESSIONS", 3)),
        "enforce_short_puts": bool(src.get("DEALER_GAMMA_ENFORCE_SHORT_PUTS", False)),
    }


def _paths() -> Dict[str, str]:
    from storage_config import DEALER_GAMMA_DECISIONS_PATH, DEALER_GAMMA_STATE_PATH
    return {"state": DEALER_GAMMA_STATE_PATH, "decisions": DEALER_GAMMA_DECISIONS_PATH}


# ── the cache the server writes ─────────────────────────────────────────────
def _data_dir() -> str:
    """Mirror of server/datacoreArchive.ts archiveBaseDir()'s data dir: env
    DATA_DIR wins, else storage_config.DATA_DIR (as sec_filings_text does)."""
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        from storage_config import DATA_DIR as data_dir  # noqa: N811
    return data_dir


def archive_dir(base: Optional[str] = None) -> str:
    """<data dir>/datacore_archive/squeeze_gex, or <base>/squeeze_gex."""
    if base:
        return os.path.join(base, "squeeze_gex")
    return os.path.join(_data_dir(), "datacore_archive", "squeeze_gex")


def _archive_files(d: str) -> List[str]:
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return []
    return sorted(n for n in names if _YEAR_FILE.match(n))


def read_rows(base: Optional[str] = None) -> List[dict]:
    """Every archived row, ascending by date, one per date (a plain year file
    wins over a gz one for the same year, as the server writes them). Malformed
    lines are skipped, never filled."""
    d = archive_dir(base)
    by_date: Dict[str, dict] = {}
    bad = 0
    files = _archive_files(d)
    plain_years = {n[:4] for n in files if not n.endswith(".gz")}
    for n in files:
        if n.endswith(".gz") and n[:4] in plain_years:
            continue
        p = os.path.join(d, n)
        opener = gzip.open if n.endswith(".gz") else open
        try:
            with opener(p, "rt", encoding="utf-8") as fh:
                raw = fh.read()
        except (OSError, EOFError) as e:
            logger.warning(f"dealer_gamma: cannot read {p}: {e}")
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                dt, gex = str(r["date"]), float(r["gex"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", dt) or gex != gex:
                bad += 1
                continue
            by_date[dt] = {"date": dt, "gex": gex, "fetched_at": r.get("fetched_at")}
    if bad:
        logger.debug(f"dealer_gamma: {bad} malformed archive line(s) skipped in {d}")
    return [by_date[k] for k in sorted(by_date)]


def last_rank(values: Sequence[float], window: int = RANK_WINDOW) -> Optional[float]:
    """Trailing percentile rank of values[-1] within the last `window` values
    (inclusive), gate 2's formula. None before a full window."""
    if len(values) < window:
        return None
    w = values[-window:]
    v = w[-1]
    below = sum(1 for x in w if x < v)
    ties = sum(1 for x in w if x == v) - 1
    return (below + 0.5 * ties) / (window - 1)


def sessions_since(as_of: str, today: date, cap: int = 30) -> int:
    """NYSE sessions after `as_of` up to and including `today` (market_calendar's
    holiday table; weekends closed). A row dated the previous session reads 1."""
    from market_calendar import is_market_holiday
    try:
        d = date.fromisoformat(as_of)
    except ValueError:
        return cap
    n = 0
    cur = d + timedelta(days=1)
    while cur <= today and n < cap:
        if not is_market_holiday(cur):
            n += 1
        cur += timedelta(days=1)
    return n


def et_today(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(_ET)).astimezone(_ET).date()


def compute_state(rows: Sequence[dict], today: date, cfg: Optional[Dict[str, Any]] = None) -> dict:
    """The filter's state from archived rows (pure). Keys: asOf, gex, rank252,
    lowGamma, threshold, source, stale, ageSessions, active, enabled,
    enforceShortPuts, rows, reason."""
    cfg = cfg or config()
    st: Dict[str, Any] = {"asOf": None, "gex": None, "rank252": None, "lowGamma": False,
                          "threshold": cfg["low_rank"], "source": SOURCE, "stale": True,
                          "ageSessions": None, "active": False, "enabled": cfg["enabled"],
                          "enforceShortPuts": cfg["enforce_short_puts"],
                          "maxStaleSessions": cfg["max_stale_sessions"], "rows": len(rows),
                          "fetchedAt": None, "reason": ""}
    if not rows:
        st["reason"] = "no SqueezeMetrics GEX rows cached yet (server squeeze_gex stream) - filter inactive"
        return st
    last = rows[-1]
    st["asOf"], st["gex"], st["fetchedAt"] = last["date"], last["gex"], last.get("fetched_at")
    st["rank252"] = last_rank([r["gex"] for r in rows])
    st["ageSessions"] = sessions_since(last["date"], today)
    st["stale"] = st["ageSessions"] > cfg["max_stale_sessions"]
    if st["rank252"] is None:
        st["reason"] = f"only {len(rows)} GEX rows cached, a 252-session rank needs 252 - filter inactive"
    elif st["stale"]:
        st["reason"] = (f"latest SqueezeMetrics GEX is {st['ageSessions']} sessions old (as of {last['date']}; "
                        f"limit {cfg['max_stale_sessions']}) - filter inactive")
    else:
        st["active"] = True
        st["lowGamma"] = st["rank252"] < cfg["low_rank"]
        st["reason"] = (f"dealer gamma {'low' if st['lowGamma'] else 'normal'} (rank {st['rank252']:.2f} "
                        f"{'<' if st['lowGamma'] else '>='} {cfg['low_rank']:.2f}, SqueezeMetrics GEX as of {last['date']})")
    if not cfg["enabled"]:
        st["active"] = False
        st["reason"] = "dealer gamma filter switched off (system_config DEALER_GAMMA_FILTER_ENABLED)"
    return st


def label(state: dict) -> str:
    """One plain-English line for the UI."""
    if not state.get("active"):
        return f"dealer gamma: filter inactive — {state.get('reason') or 'no data'}"
    r = state.get("rank252")
    if state.get("lowGamma"):
        tail = ("no new short premium" if state.get("enforceShortPuts")
                else "no new condors / strangles; cash-secured puts and put spreads logged only")
        return f"dealer gamma: low (rank {r:.2f}) — {tail}"
    return f"dealer gamma: normal (rank {r:.2f}) — no restriction"


def public_state(state: dict) -> dict:
    """The state minus the raw GEX value (licence posture: aggregates only)."""
    out = {k: v for k, v in state.items() if k != "gex"}
    out["gexNegative"] = (state["gex"] < 0) if isinstance(state.get("gex"), (int, float)) else None
    out["label"] = label(state)
    return out


# ── cached state for the live paths ─────────────────────────────────────────
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"at": 0.0, "sig": None, "rows": None, "state": None, "day": None}


def _signature(d: str) -> tuple:
    sig = []
    for n in _archive_files(d):
        try:
            s = os.stat(os.path.join(d, n))
        except OSError as e:
            logger.debug(f"dealer_gamma: stat {n} failed: {e}")
            continue
        sig.append((n, s.st_mtime_ns, s.st_size))
    return tuple(sig)


def get_state(force: bool = False, now: Optional[datetime] = None, base: Optional[str] = None) -> dict:
    """The current state, cached STATE_TTL_S; the archive is re-read only when
    its files change. Never raises: an error -> an inactive state saying so."""
    t = time.time()
    today = et_today(now)
    try:
        with _LOCK:
            if (not force and _CACHE["state"] is not None and t - _CACHE["at"] < STATE_TTL_S
                    and _CACHE["day"] == today and base is None):
                return dict(_CACHE["state"])
            d = archive_dir(base)
            sig = _signature(d)
            rows = _CACHE["rows"] if (sig == _CACHE["sig"] and _CACHE["rows"] is not None and base is None) else read_rows(base)
            state = compute_state(rows, today)
            state["computedAt"] = datetime.now(_ET).isoformat(timespec="seconds")
            if base is None:
                _CACHE.update({"at": t, "sig": sig, "rows": rows, "state": state, "day": today})
                _write_state(state)
            return dict(state)
    except Exception as e:  # noqa: BLE001 — logged; the filter must fail open
        logger.warning(f"dealer_gamma: state unavailable ({type(e).__name__}: {e}) - filter inactive")
        cfg = config()
        return {**compute_state([], today, cfg), "reason": f"dealer gamma state error ({type(e).__name__}) - filter inactive"}


def _write_state(state: dict) -> None:
    """The public state, written atomically on every recompute (at most once per
    STATE_TTL_S), so its computedAt says when the trading path last read it."""
    try:
        p = _paths()["state"]
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(public_state(state), fh, default=str)
        os.replace(tmp, p)
    except OSError as e:
        logger.warning(f"dealer_gamma: state file not written: {e}")


# ── the decision ────────────────────────────────────────────────────────────
def structure_of(strategy: Optional[str], side: Optional[str] = None) -> str:
    """'long' | 'short_put' | 'short_two_sided' | 'short_call' for an option
    entry; an unknown strategy that is SOLD reads as short_two_sided (the
    enforced class: premium is being sold and its shape is unknown)."""
    s = str(strategy or "").strip().lower()
    if s in SHORT_PUT_STRATEGIES:
        return "short_put"
    if s in SHORT_TWO_SIDED_STRATEGIES:
        return "short_two_sided"
    if s in SHORT_CALL_STRATEGIES:
        return "short_call"
    if s in LONG_STRATEGIES:
        return "long"
    return "short_two_sided" if str(side or "").strip().lower() in ("sell", "short") else "long"


def decide(strategy: Optional[str], side: Optional[str], state: dict) -> dict:
    """allow | skip | would_skip for one new option entry under `state` (pure)."""
    struct = structure_of(strategy, side)
    out = {"action": "allow", "structure": struct, "shortPremium": struct != "long",
           "enforced": False, "rank252": state.get("rank252"), "asOf": state.get("asOf"),
           "lowGamma": bool(state.get("lowGamma")), "active": bool(state.get("active")),
           "threshold": state.get("threshold"), "reason": ""}
    if struct == "long":
        out["reason"] = "bought option: dealer gamma filter does not apply"
        return out
    if not state.get("active"):
        out["reason"] = f"dealer gamma filter inactive: {state.get('reason')}"
        return out
    r = state.get("rank252")
    if not state.get("lowGamma"):
        out["reason"] = f"dealer gamma normal (rank {r:.2f}): short premium allowed"
        return out
    enforce = struct != "short_put" or bool(state.get("enforceShortPuts"))
    out["enforced"] = enforce
    if enforce:
        out["action"] = "skip"
        out["reason"] = (f"dealer gamma low (rank {r:.2f}): realized vol tends to exceed implied, "
                         f"no new short premium (SqueezeMetrics GEX estimate as of {state.get('asOf')})")
    else:
        out["action"] = "would_skip"
        out["reason"] = (f"dealer gamma low (rank {r:.2f}): would skip this short put, but it is logged only - "
                         f"the 2012-2026 SPY replay found skipping cash-secured puts in low gamma cost more "
                         f"premium than it saved (SqueezeMetrics GEX estimate as of {state.get('asOf')})")
    return out


# ── recording ───────────────────────────────────────────────────────────────
_PENDING: List[dict] = []
_SEEN: Dict[str, Any] = {"day": None, "keys": set()}
_SHADOW: Dict[str, Any] = {"since": 0.0, "n": 0}


def record(decision: dict, ticker: str, source: str, setup: str = "", strategy: str = "",
           score: Optional[float] = None, price: Optional[float] = None,
           now: Optional[datetime] = None) -> Optional[dict]:
    """Log one skip / would_skip: once per ET day per (ticker, setup, action).
    Returns the record ({..., "new": True}) the first time, else None."""
    if decision.get("action") not in ("skip", "would_skip"):
        return None
    ts = (now or datetime.now(_ET)).astimezone(_ET)
    day = ts.date().isoformat()
    key = f"{str(ticker).upper()}|{setup or strategy}|{decision['action']}"
    with _LOCK:
        if _SEEN["day"] != day:
            _SEEN["day"], _SEEN["keys"] = day, set()
        if key in _SEEN["keys"]:
            return None
        _SEEN["keys"].add(key)
    rec = {"ts": ts.isoformat(timespec="seconds"), "day": day, "ticker": str(ticker).upper(),
           "source": source, "setup": setup, "strategy": strategy, "structure": decision.get("structure"),
           "action": decision["action"], "enforced": bool(decision.get("enforced")),
           "rank252": decision.get("rank252"), "threshold": decision.get("threshold"),
           "asOf": decision.get("asOf"), "score": score, "price": price,
           "reason": decision.get("reason"), "new": True}
    try:
        with open(_paths()["decisions"], "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except OSError as e:
        logger.warning(f"dealer_gamma: decision not written to the JSON log: {e}")
    with _LOCK:
        _PENDING.append(rec)
        del _PENDING[:-PENDING_MAX]
    logger.warning(f"DEALER-GAMMA {decision['action'].upper()} {rec['ticker']} ({setup or strategy}, {source}): "
                   f"{decision.get('reason')}")
    return rec


def check(strategy: Optional[str], side: Optional[str], ticker: str, source: str, setup: str = "",
          score: Optional[float] = None, price: Optional[float] = None) -> dict:
    """decide() on the current state + record() a skip / would_skip. Never raises:
    any failure returns an `allow` decision that says so."""
    try:
        d = decide(strategy, side, get_state())
        if d["action"] != "allow":
            d["record"] = record(d, ticker, source, setup=setup, strategy=str(strategy or ""),
                                 score=score, price=price)
        return d
    except Exception as e:  # noqa: BLE001 — logged; never block on a filter failure
        logger.warning(f"dealer_gamma: check failed for {ticker} ({type(e).__name__}: {e}) - allowed")
        return {"action": "allow", "structure": structure_of(strategy, side), "enforced": False,
                "reason": f"dealer gamma check error ({type(e).__name__}) - allowed"}


def shadow_slot() -> bool:
    """True while this scan may still write a shadow_portfolio record for a
    gamma decision (at most SHADOW_WRITES_PER_SCAN per ~2 minutes)."""
    t = time.time()
    with _LOCK:
        if t - _SHADOW["since"] > SHADOW_WINDOW_S:
            _SHADOW["since"], _SHADOW["n"] = t, 0
        if _SHADOW["n"] >= SHADOW_WRITES_PER_SCAN:
            return False
        _SHADOW["n"] += 1
        return True


def drain() -> List[dict]:
    """The records logged since the last drain (oldest first), cleared."""
    with _LOCK:
        out = list(_PENDING)
        _PENDING.clear()
        _SHADOW["since"], _SHADOW["n"] = 0.0, 0
    return out


def _reset_for_tests() -> None:
    """Clear every per-process cache (state, pending records, dedup, shadow budget)."""
    with _LOCK:
        _CACHE.update({"at": 0.0, "sig": None, "rows": None, "state": None, "day": None})
        _PENDING.clear()
        _SEEN["day"], _SEEN["keys"] = None, set()
        _SHADOW["since"], _SHADOW["n"] = 0.0, 0


def scan_field() -> dict:
    """bot_engine.scan_market's `dealer_gamma` field: the state as the scan read
    it (no raw GEX) and the decisions logged during the scan. Never raises."""
    try:
        return {"state": public_state(get_state()), "decisions": drain()}
    except Exception as e:  # noqa: BLE001 — reported in the field
        logger.warning(f"dealer_gamma: scan field failed: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"[:160], "decisions": []}
