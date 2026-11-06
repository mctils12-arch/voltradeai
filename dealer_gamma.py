"""dealer_gamma.py — Edge P1's live input: SqueezeMetrics' published dealer-
gamma estimate (GEX) and its pre-registered trailing-252 percentile rank, plus
the short-premium entry filter that reads it (2026-11-06; refocus spec §3
step 6 "options flow/positioning: dealer gamma, implied vs realized vol").

SOURCE — NOT OURS. SqueezeMetrics' free DIX.csv (date, price, dix, gex), their
model ESTIMATE of S&P 500 dealers' gamma ($ per 1% move; it assumes dealers are
long calls and short puts). SqueezeMetrics' Terms of Service
(squeezemetrics.com/monitor/terms, read 2026-11-06) grant no licence to the
data and say users may not "systematically retrieve data or other content from
the Platform to create or compile, directly or indirectly, a collection,
compilation, database, or directory without written permission"; the content
is "for general information only". No CSV-specific terms are published. So
NOTHING of it is stored (2026-11-06): the server's squeeze_gex poll
(server/squeezeGex.ts, every 6 h) reads the file — which carries its own daily
history — in memory and writes only a DERIVED summary, SUMMARY_FILE in the data
dir (the latest row's date, the row count, the trailing-252 rank, the sign of
the latest GEX, the fetch time; no SqueezeMetrics value). This module only READS
that summary: it never fetches, so the bot adds no second retriever of their
file, and no raw GEX value is ever in this process (the scan field and the diag
probe carry the rank, the date and the verdict only). Until 2026-11-06 the
server kept a daily archive of the file under datacore_archive/squeeze_gex/ and
this module read that; the server deletes it at boot.

RANK — bit-for-bit gate 2's definition (scripts/edge_p1_dealer_gamma.
trailing_pct_rank; server/squeezeGex.trailingPctRank computes it into the
summary): R_t = (count of the 252 rows ending at t with GEX < G_t + 0.5 x ties
excluding itself) / 251, using rows <= t only; undefined until 252 rows exist.

THE RULE (research/edge_preregistration.md § P1, gate 3): no NEW short premium
when the last published R is below DEALER_GAMMA_LOW_RANK (0.2). Gate 3
(scripts/edge_p1_gate3_rule_sim.py, SPY model replay 2012-05 -> 2026-10,
2026-11-06) found that rule HURT a 30-delta cash-secured-put book and roughly
halved a 1-SD strangle book's max drawdown, so one-sided short puts (CSPs, put
credit spreads) are LOGGED as "would skip" unless DEALER_GAMMA_ENFORCE_SHORT_PUTS,
while two-sided / call-side short premium (iron condors, strangles) is skipped.
Bought options are never affected.

FAILS OPEN. Filter disabled, no summary yet, < 252 rows, a latest row older
than DEALER_GAMMA_MAX_STALE_SESSIONS NYSE sessions, or a read error (an
unreadable or malformed summary included) -> the filter is INACTIVE: it never
blocks, and its state says why.

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

import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
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
SUMMARY_FILE = "voltrade_squeeze_gex_summary.json"   # server/squeezeGex.ts SQUEEZE_SUMMARY_FILE
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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


# ── the summary the server writes ───────────────────────────────────────────
def _data_dir() -> str:
    """Mirror of server/squeezeGex.ts squeezeSummaryPath()'s data dir: env
    DATA_DIR wins, else storage_config.DATA_DIR (as sec_filings_text does)."""
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        from storage_config import DATA_DIR as data_dir  # noqa: N811
    return data_dir


def summary_path(base: Optional[str] = None) -> str:
    """<data dir>/SUMMARY_FILE, or <base>/SUMMARY_FILE."""
    return os.path.join(base or _data_dir(), SUMMARY_FILE)


def _bad_summary(p: str, why: str) -> ValueError:
    return ValueError(f"squeeze_gex summary {p}: {why}")


def read_summary(base: Optional[str] = None) -> Optional[dict]:
    """The server's derived summary ({asOf, rows, rank252, gexNegative,
    fetchedAt}), or None while none exists (no successful read of the file
    yet). A summary that cannot be parsed, has the wrong shape, or contradicts
    itself (a rank without 252 rows, 252+ rows without a rank) RAISES: get_state
    turns that into an inactive state that says so — never a guess."""
    p = summary_path(base)
    try:
        with open(p, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return None
    if not isinstance(raw, dict) or raw.get("v") != 1:
        raise _bad_summary(p, "not a v1 summary")
    as_of, rows, rank = raw.get("asOf"), raw.get("rows"), raw.get("rank252")
    neg, fetched = raw.get("gexNegative"), raw.get("fetchedAt")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise _bad_summary(p, f"rows {rows!r}")
    if as_of is not None and not (isinstance(as_of, str) and _DATE.match(as_of)):
        raise _bad_summary(p, f"asOf {as_of!r}")
    if rows and as_of is None:
        raise _bad_summary(p, "rows without a date")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, (int, float)) or not 0.0 <= rank <= 1.0):
        raise _bad_summary(p, f"rank252 {rank!r}")
    if (rank is None) != (rows < RANK_WINDOW):
        raise _bad_summary(p, f"rank252 {rank!r} with {rows} rows")
    if neg is not None and not isinstance(neg, bool):
        raise _bad_summary(p, f"gexNegative {neg!r}")
    if fetched is not None and not isinstance(fetched, str):
        raise _bad_summary(p, f"fetchedAt {fetched!r}")
    return {"asOf": as_of, "rows": rows, "rank252": None if rank is None else float(rank),
            "gexNegative": neg, "fetchedAt": fetched}


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


def compute_state(summary: Optional[dict], today: date, cfg: Optional[Dict[str, Any]] = None) -> dict:
    """The filter's state from the server's summary (pure; None = no summary
    yet). Keys: asOf, rank252, gexNegative, lowGamma, threshold, source, stale,
    ageSessions, active, enabled, enforceShortPuts, maxStaleSessions, rows,
    fetchedAt, reason. Same states, labels and reasons the archive path gave
    for the same rows (test_dealer_gamma pins them)."""
    cfg = cfg or config()
    n = int(summary.get("rows") or 0) if summary else 0
    st: Dict[str, Any] = {"asOf": None, "rank252": None, "gexNegative": None, "lowGamma": False,
                          "threshold": cfg["low_rank"], "source": SOURCE, "stale": True,
                          "ageSessions": None, "active": False, "enabled": cfg["enabled"],
                          "enforceShortPuts": cfg["enforce_short_puts"],
                          "maxStaleSessions": cfg["max_stale_sessions"], "rows": n,
                          "fetchedAt": None, "reason": ""}
    if not n:
        st["reason"] = "no SqueezeMetrics GEX rows cached yet (server squeeze_gex stream) - filter inactive"
        return st
    as_of = summary["asOf"]
    st["asOf"], st["gexNegative"], st["fetchedAt"] = as_of, summary.get("gexNegative"), summary.get("fetchedAt")
    st["rank252"] = summary.get("rank252")
    st["ageSessions"] = sessions_since(as_of, today)
    st["stale"] = st["ageSessions"] > cfg["max_stale_sessions"]
    if st["rank252"] is None:
        st["reason"] = f"only {n} GEX rows cached, a 252-session rank needs 252 - filter inactive"
    elif st["stale"]:
        st["reason"] = (f"latest SqueezeMetrics GEX is {st['ageSessions']} sessions old (as of {as_of}; "
                        f"limit {cfg['max_stale_sessions']}) - filter inactive")
    else:
        st["active"] = True
        st["lowGamma"] = st["rank252"] < cfg["low_rank"]
        st["reason"] = (f"dealer gamma {'low' if st['lowGamma'] else 'normal'} (rank {st['rank252']:.2f} "
                        f"{'<' if st['lowGamma'] else '>='} {cfg['low_rank']:.2f}, SqueezeMetrics GEX as of {as_of})")
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
    """The state for the state file, the scan field and the probe, plus its
    label. It never holds a raw GEX value (the summary has none; the filter
    drops a stray "gex" key all the same)."""
    out = {k: v for k, v in state.items() if k != "gex"}
    out["gexNegative"] = state.get("gexNegative") if isinstance(state.get("gexNegative"), bool) else None
    out["label"] = label(state)
    return out


# ── cached state for the live paths ─────────────────────────────────────────
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"at": 0.0, "sig": None, "loaded": False, "summary": None, "state": None, "day": None}


def _signature(p: str) -> Optional[tuple]:
    try:
        s = os.stat(p)
    except OSError as e:
        logger.debug(f"dealer_gamma: stat {p} failed: {e}")
        return None
    return (s.st_mtime_ns, s.st_size)


def get_state(force: bool = False, now: Optional[datetime] = None, base: Optional[str] = None) -> dict:
    """The current state, cached STATE_TTL_S; the summary is re-read only when
    its file changes. Never raises: an error -> an inactive state saying so."""
    t = time.time()
    today = et_today(now)
    try:
        with _LOCK:
            if (not force and _CACHE["state"] is not None and t - _CACHE["at"] < STATE_TTL_S
                    and _CACHE["day"] == today and base is None):
                return dict(_CACHE["state"])
            sig = _signature(summary_path(base))
            cached = base is None and _CACHE["loaded"] and sig == _CACHE["sig"]
            summary = _CACHE["summary"] if cached else read_summary(base)
            state = compute_state(summary, today)
            state["computedAt"] = datetime.now(_ET).isoformat(timespec="seconds")
            if base is None:
                _CACHE.update({"at": t, "sig": sig, "loaded": True, "summary": summary, "state": state, "day": today})
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
        _CACHE.update({"at": 0.0, "sig": None, "loaded": False, "summary": None, "state": None, "day": None})
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
