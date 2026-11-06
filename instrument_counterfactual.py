#!/usr/bin/env python3
"""
instrument_counterfactual.py — "log why the rejected instrument lost"
(2026-11-06; human refocus spec §3 step 7: "instrument choice: stock vs options
vs both").

THE LEDGER — DATA_DIR/voltrade_instrument_counterfactual.jsonl (+ .1, the
previous generation), append-only, three row kinds:
  decision  written by Node (server/instrumentCounterfactual.ts) when bot.ts
            executeTrades ENTERS a trade the instrument selector decided (a
            live order sent, or a shadow-mode entry). `chosen` = what was
            ordered; `rejected` = the instrument passed over across the
            stock/options line: the option alternative of a stock / ETF buy
            (instrument_selector.option_alternative: the selector's strategy,
            score, why options lost, and the ATM proxy to price) or the stock
            alternative of an options entry (the scan's price, size, stop and
            target). The same leg rides the decision journal row's `rejected`
            entry as `would_be`; `id` is that journal row's id.
  proxy     written here, once per decision whose option leg has no contract:
            the listed contract the proxy resolved to, or null and why.
  outcome   written here, one per decision and horizon (+1d / +5d / +20d
            TRADING days): both legs' returns, their difference and a
            plain-English line ("chose stock: +2.1%; the ATM call proxy it
            passed over would have made +38.0% ...").

THE MATH — both legs on ONE clock: close-to-close from the entry day's close to
the close N trading days later (trading days = the underlying's daily bars).
  shares (stock / ETF): (close_N / close_0 - 1) x 100 (negated for a short).
  options: per-share position value V = sum(+close of a bought leg, -close of a
           sold leg); return = (V_N - V_0) / basis x 100 where basis = V_0 for
           a net debit (the premium paid), else the capital at risk per share
           (the recorded cash_required, else max_loss, / (100 x qty); a lone
           sold put's strike; a lone sold call's underlying close). A leg that
           expires inside the horizon is worth its intrinsic value at the
           underlying's close on the expiry date. A leg with no trade ON the
           entry day is UNPRICED — a later price never stands in for the entry;
           a stale exit mark (no trade on day N) is the last close before it,
           counted in stale_days.
  diff_pp = chosen - rejected, in percentage points, each leg on its own
           capital. A decision's "edge dollars" (the aggregate, Node) assume
           the same capital in the alternative.

HONESTY: the entry day's intraday move and any fill slippage are in neither leg
(the same clock for both); a limit that never filled still counts (the decision
was made); an option the selector never built a contract for — the scan fetches
no chain — is a PROXY: a BOUGHT ATM call (put for a bearish idea) expiring
proxy_dte_min..proxy_dte_max days after the entry, the strike nearest the entry
price that traded on the entry day. It is labelled "ATM call proxy" in every
line and never passed off as the selector's own strategy, whose name and score
are kept beside it. Multi-leg structures are priced only from recorded legs.

COST: bot.ts runs backfill() nightly (checkShadowBackfill's 22:00 UTC slot, a
python3 -c subprocess): per pending decision one stock-bars call per symbol and
one option-bars call, plus one contracts call per proxy — a few dozen calls.

OFF SWITCH: system_config BASE_CONFIG["INSTRUMENT_COUNTERFACTUAL"]["enabled"];
the env var VOLTRADE_INSTRUMENT_COUNTERFACTUAL wins whenever it is PRESENT
("off"/"0"/"false" = nothing written or priced) — the decision journal's rule,
mirrored by server/instrumentCounterfactual.ts.

LOGGING ONLY: nothing here changes an order, a size, a gate or a decision;
nothing prints to stdout (bot.ts parses the caller's print).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

try:
    from storage_config import DATA_DIR
except ImportError:  # pragma: no cover
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

log = logging.getLogger("voltrade.instrument_counterfactual")

FILENAME = "voltrade_instrument_counterfactual.jsonl"
CONFIG_KEY = "INSTRUMENT_COUNTERFACTUAL"
ENV_VAR = "VOLTRADE_INSTRUMENT_COUNTERFACTUAL"
OFF_VALUES = ("", "0", "off", "false", "no", "none")
DEFAULT_MAX_BYTES = 5_000_000      # mirrors system_config INSTRUMENT_COUNTERFACTUAL.max_bytes
MIN_MAX_BYTES = 100_000
ROW_MAX_BYTES = 4_000              # mirrors server/instrumentCounterfactual.ts
DEFAULT_PROXY_DTE = (30, 45)       # mirrors system_config proxy_dte_min / proxy_dte_max
SCHEMA_VERSION = 1
HORIZONS = {"+1d": 1, "+5d": 5, "+20d": 20}
GIVE_UP_DAYS = 70                  # past this age an unsettled horizon gets a terminal row
WINDOW_PAD_DAYS = 6                # bars are fetched from the entry day minus this ...
WINDOW_SPAN_DAYS = 45              # ... to the entry day plus this (20 trading days + holidays)
PROXY_STRIKE_BAND = 0.15           # the proxy's contracts are searched within +-15% of the entry price
PROXY_CANDIDATES = 4               # nearest strikes priced to find one that traded on the entry day
ALPACA_DATA_URL = "https://data.alpaca.markets"
OPTION_BARS_MAX_PAGES = 5

_write_lock = threading.Lock()
_OCC_RE = re.compile(r"^([A-Z.]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


# ── switch + paths ────────────────────────────────────────────────────────────
def _config() -> dict:
    try:
        from system_config import BASE_CONFIG
    except ImportError as e:
        log.warning(f"[INSTRUMENT-CF] system_config unavailable ({e}); defaults used")
        return {}
    c = BASE_CONFIG.get(CONFIG_KEY) or {}
    return c if isinstance(c, dict) else {}


def enabled(env=None, config=None) -> bool:
    """The env var when PRESENT (any value wins), else the config flag (default on)."""
    env = os.environ if env is None else env
    v = env.get(ENV_VAR)
    if v is not None:
        return str(v).strip().lower() not in OFF_VALUES
    cfg = _config() if config is None else config
    return bool(cfg.get("enabled", True))


def max_bytes(config=None) -> int:
    cfg = _config() if config is None else config
    try:
        v = int(cfg.get("max_bytes", DEFAULT_MAX_BYTES))
    except (TypeError, ValueError):
        v = DEFAULT_MAX_BYTES
    return max(MIN_MAX_BYTES, v)


def ledger_path() -> str:
    return os.path.join(DATA_DIR, FILENAME)


# ── small helpers ─────────────────────────────────────────────────────────────
def _num(x) -> Optional[float]:
    """A finite float, else None (bools are not numbers here)."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _text(x, n=300) -> str:
    return " ".join(("" if x is None else str(x)).split())[:n]


def now_iso(now=None) -> str:
    dt = now or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _day(s) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def parse_occ(symbol) -> Optional[dict]:
    """'AMD261218C00150000' -> {underlying, expiry 'YYYY-MM-DD', right 'C'|'P', strike}; else None."""
    m = _OCC_RE.match(str(symbol or "").strip().upper())
    if not m:
        return None
    return {"symbol": m.group(0), "underlying": m.group(1), "expiry": f"20{m.group(2)}-{m.group(3)}-{m.group(4)}",
            "right": m.group(5), "strike": int(m.group(6)) / 1000.0}


def describe_occ(symbol) -> str:
    o = parse_occ(symbol)
    if not o:
        return str(symbol or "")
    return f"{o['underlying']} ${o['strike']:g} {'call' if o['right'] == 'C' else 'put'} expiring {o['expiry']}"


def _bar_date(bar: dict) -> str:
    return str(bar.get("t", ""))[:10]


def _sorted_bars(bars) -> List[dict]:
    return sorted((b for b in (bars or []) if isinstance(b, dict) and _bar_date(b)), key=_bar_date)


def _base_index(bars: List[dict], day: str) -> Optional[int]:
    """Index of the last bar dated on or before `day` (bars ascending)."""
    idx = None
    for i, b in enumerate(bars):
        if _bar_date(b) <= day:
            idx = i
        else:
            break
    return idx


def _close_on(bars: List[dict], day: str) -> Optional[float]:
    """The close of the bar dated exactly `day`, else None."""
    for b in bars:
        if _bar_date(b) == day:
            c = _num(b.get("c"))
            return c if c is not None and c >= 0 else None
    return None


def _close_at_or_before(bars: List[dict], day: str) -> Tuple[Optional[float], Optional[str]]:
    i = _base_index(bars, day)
    if i is None:
        return None, None
    c = _num(bars[i].get("c"))
    return (c, _bar_date(bars[i])) if c is not None and c >= 0 else (None, None)


def _unpriced(reason: str) -> dict:
    return {"priced": False, "reason": _text(reason, 220)}


# ── the legs ──────────────────────────────────────────────────────────────────
def share_leg(bars, entry_day: str, end_day: str, side: str = "buy") -> dict:
    """A stock / ETF leg, close-to-close: the last close on or before the entry
    day to the last close on or before the horizon day."""
    bars = _sorted_bars(bars)
    c0, d0 = _close_at_or_before(bars, entry_day)
    cn, dn = _close_at_or_before(bars, end_day)
    if not c0 or c0 <= 0:
        return _unpriced(f"no daily close on or before the entry day ({entry_day})")
    if cn is None or dn is None or (dn <= entry_day < end_day):
        return _unpriced(f"no daily close after the entry day up to {end_day}")
    pct = (cn / c0 - 1.0) * 100.0
    if str(side).lower() in ("sell", "short"):
        pct = -pct
    return {"priced": True, "v0": round(c0, 4), "vN": round(cn, 4), "pct": round(pct, 3), "basis": "price",
            "entry_close_date": d0, "end_close_date": dn}


def _leg_sign(side) -> int:
    return -1 if str(side or "").lower() in ("sell", "short", "sell_to_open") else 1


def option_position(legs, option_bars: Dict[str, list], under_bars, entry_day: str, end_day: str,
                    capital_per_share: Optional[float] = None) -> dict:
    """An option position of one or more legs ([{symbol, side, ratio?}]), per
    share, close-to-close from the entry day. See the module doc for the basis
    and expiry rules."""
    if not legs:
        return _unpriced("no option contract to price")
    under = _sorted_bars(under_bars)
    parsed = []
    for leg in legs:
        occ = parse_occ((leg or {}).get("symbol"))
        if occ is None:
            return _unpriced(f"{_text((leg or {}).get('symbol'), 30)} is not an OCC contract symbol")
        parsed.append((leg, occ))
    v0 = vn = 0.0
    stale, expired = 0, False
    for leg, occ in parsed:
        sign = _leg_sign(leg.get("side")) * (_num(leg.get("ratio")) or 1.0)
        bars = _sorted_bars((option_bars or {}).get(occ["symbol"]))
        c0 = _close_on(bars, entry_day)
        if c0 is None:
            return _unpriced(f"the {describe_occ(occ['symbol'])} had no trade on the entry day ({entry_day})")
        if end_day > occ["expiry"]:
            s_exp, _ = _close_at_or_before(under, occ["expiry"])
            if s_exp is None:
                return _unpriced(f"no underlying close on or before the {occ['expiry']} expiry")
            cn = max(0.0, s_exp - occ["strike"]) if occ["right"] == "C" else max(0.0, occ["strike"] - s_exp)
            expired = True
        else:
            cn, dn = _close_at_or_before(bars, end_day)
            if cn is None or dn is None:
                return _unpriced(f"the {describe_occ(occ['symbol'])} has no close up to {end_day}")
            stale = max(stale, sum(1 for b in under if dn < _bar_date(b) <= end_day))
        v0 += sign * c0
        vn += sign * cn
    if v0 > 0:
        basis, kind = v0, "premium"
    else:
        basis, kind = _num(capital_per_share), "capital"
        if basis is None and len(parsed) == 1 and _leg_sign(parsed[0][0].get("side")) < 0:
            occ = parsed[0][1]
            if occ["right"] == "P":
                basis = occ["strike"]                                   # cash-secured put
            else:
                basis, _ = _close_at_or_before(under, entry_day)        # covered call: the shares
        if not basis or basis <= 0:
            return _unpriced("a credit position with no recorded capital at risk")
    pct = (vn - v0) / basis * 100.0
    return {"priced": True, "v0": round(v0, 4), "vN": round(vn, 4), "pct": round(pct, 3), "basis": kind,
            "basis_per_share": round(basis, 4), "stale_days": stale, "expired": expired}


def _capital_per_share(chosen: dict) -> Optional[float]:
    cap = _num(chosen.get("cash_required")) or _num(chosen.get("max_loss"))
    qty = _num(chosen.get("qty")) or 1.0
    return cap / (100.0 * qty) if cap and cap > 0 and qty > 0 else None


# ── words ─────────────────────────────────────────────────────────────────────
def _chosen_words(d: dict) -> str:
    c = d.get("chosen") or {}
    inst = c.get("instrument")
    if inst == "etf":
        return f"the leveraged ETF {_text(c.get('symbol'), 12)}"
    if inst == "option":
        strat = _text(c.get("strategy"), 60).replace("_", " ").strip()
        return f"options ({strat})" if strat else "options"
    return "stock"


def _rejected_words(d: dict, proxy: bool) -> str:
    r = d.get("rejected") or {}
    if r.get("instrument") == "stock":
        return "stock"
    right = str(((r.get("proxy") or {}).get("right")) or ((r.get("contract") or {}).get("right")) or "C").upper()
    word = "put" if right == "P" else "call"
    return f"ATM {word} proxy" if proxy else f"ATM {word}"


def _basis_words(leg: dict) -> str:
    if leg.get("basis") == "premium":
        return " on the premium paid"
    if leg.get("basis") == "capital":
        return " on the capital at risk"
    return ""


def outcome_line(d: dict, horizon: str, ch: dict, rj: dict, proxy: bool) -> str:
    """'chose stock: +2.1%; the ATM call proxy it passed over would have made +38.0% ...'"""
    n = HORIZONS.get(horizon, 0)
    cw, rw = _chosen_words(d), _rejected_words(d, proxy)
    if ch.get("priced"):
        head = f"chose {cw}: {ch['pct']:+.1f}%{_basis_words(ch)}"
    else:
        head = f"chose {cw} (its return could not be priced: {ch.get('reason')})"
    if rj.get("priced"):
        p = rj["pct"]
        tail = f"; the {rw} it passed over would have {'made' if p >= 0 else 'lost'} " \
               f"{('+' if p >= 0 else '') + format(p if p >= 0 else -p, '.1f')}%{_basis_words(rj)}"
    else:
        tail = f"; the {rw} it passed over could not be priced ({rj.get('reason')})"
    verdict = ""
    if ch.get("priced") and rj.get("priced"):
        diff = ch["pct"] - rj["pct"]
        verdict = (f" — the choice was right by {diff:.1f} points" if diff > 0
                   else f" — the choice cost {-diff:.1f} points" if diff < 0 else " — a tie")
    when = f" ({n} trading day{'s' if n != 1 else ''}, both legs close-to-close from the entry day)"
    return _text(head + tail + verdict + when, 600)


def outcome_row(d: dict, horizon: str, end_day: Optional[str], ch: dict, rj: dict, proxy: bool,
                now=None) -> dict:
    both = ch.get("priced") and rj.get("priced")
    diff = round(ch["pct"] - rj["pct"], 3) if both else None
    return {
        "v": SCHEMA_VERSION, "kind": "outcome", "cf_id": d.get("id"), "horizon": horizon, "ts": now_iso(now),
        "entry_date": d.get("entry_date"), "end_date": end_day, "clock": "close_to_close_from_entry_day",
        "chosen": ch, "rejected": rj, "diff_pp": diff,
        "winner": None if diff is None else ("chosen" if diff > 0 else "rejected" if diff < 0 else "tie"),
        "proxy": bool(proxy), "line": outcome_line(d, horizon, ch, rj, proxy),
    }


# ── read / fold / append ──────────────────────────────────────────────────────
def read_rows(include_rotated: bool = True) -> list:
    """Every parseable row, oldest generation first."""
    out = []
    p = ledger_path()
    for path in ([p + ".1"] if include_rotated else []) + [p]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            log.warning(f"[INSTRUMENT-CF] {path} unreadable: {e}")
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as e:
                log.debug(f"[INSTRUMENT-CF] malformed row skipped: {e}")
    return out


def fold(rows) -> Tuple[Dict[str, dict], Dict[str, Dict[str, dict]], Dict[str, dict]]:
    """(decisions by id, outcomes by id and horizon, proxies by id) — order-agnostic."""
    decisions, outcomes, proxies = {}, {}, {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        kind = r.get("kind")
        if kind == "decision" and r.get("id"):
            decisions[str(r["id"])] = r
        elif kind == "outcome" and r.get("cf_id") and r.get("horizon") in HORIZONS:
            outcomes.setdefault(str(r["cf_id"]), {}).setdefault(r["horizon"], r)
        elif kind == "proxy" and r.get("cf_id"):
            proxies.setdefault(str(r["cf_id"]), r)
    return decisions, outcomes, proxies


def _fit(row: dict) -> str:
    text = json.dumps(row, separators=(",", ":"), default=str)
    if len(text.encode("utf-8")) <= ROW_MAX_BYTES:
        return text
    r = dict(row)
    r["line"] = _text(r.get("line"), 300)
    for k in ("chosen", "rejected"):
        if isinstance(r.get(k), dict):
            r[k] = {kk: (_text(v, 120) if isinstance(v, str) else v) for kk, v in r[k].items()}
    r["truncated"] = True
    return json.dumps(r, separators=(",", ":"), default=str)


def append_rows(rows) -> int:
    """Append rows (ONE write on an O_APPEND fd, so a Node append never lands
    inside a line), rotating to .1 above max_bytes. Returns rows written; never raises."""
    try:
        lines = [_fit(r) + "\n" for r in rows or [] if isinstance(r, dict)]
        if not lines:
            return 0
        data = "".join(lines).encode("utf-8")
        p = ledger_path()
        with _write_lock:
            if os.path.exists(p) and os.path.getsize(p) > max_bytes():
                os.replace(p, p + ".1")
            fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
            finally:
                os.close(fd)
        return len(lines)
    except (OSError, TypeError, ValueError) as e:
        log.warning(f"[INSTRUMENT-CF] rows not written (no decision changed): {e}")
        return 0


# ── network (the nightly job only — never the scan path) ──────────────────────
def _alpaca_headers() -> dict:
    return {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", "")}


def _throttle() -> None:
    try:
        from alpaca_rate_limiter import alpaca_throttle
    except ImportError as e:
        log.debug(f"[INSTRUMENT-CF] rate limiter unavailable: {e}")
        return
    alpaca_throttle.acquire()


def fetch_stock_bars(symbol: str, start: str, end: str) -> list:
    """Daily bars for one stock / ETF (shadow_portfolio's batch fetcher, split-adjusted)."""
    from shadow_portfolio import _fetch_historical_bars_batch
    return (_fetch_historical_bars_batch([symbol], start, end) or {}).get(symbol) or []


def fetch_option_bars(symbols, start: str, end: str) -> Dict[str, list]:
    """Alpaca's historical daily OPTION bars for OCC contracts (the endpoint the
    `option_bars` diag probe passes through). Raises on an HTTP error."""
    import requests
    syms = sorted({str(s).upper() for s in symbols or [] if parse_occ(s)})
    out: Dict[str, list] = {}
    token = None
    for _ in range(OPTION_BARS_MAX_PAGES):
        if not syms:
            break
        _throttle()
        params = {"symbols": ",".join(syms), "timeframe": "1Day", "start": start, "end": end,
                  "limit": 10000, "sort": "asc"}
        if token:
            params["page_token"] = token
        r = requests.get(f"{ALPACA_DATA_URL}/v1beta1/options/bars", params=params,
                         headers=_alpaca_headers(), timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca option bars HTTP {r.status_code}: {r.text[:120]}")
        data = r.json() or {}
        for sym, bars in (data.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars or [])
        token = data.get("next_page_token")
        if not token:
            break
    return out


def fetch_contracts(underlying: str, kind: str, exp_gte: str, exp_lte: str,
                    strike_gte: float, strike_lte: float) -> list:
    """Listed option contracts (Alpaca trading API): active first, then inactive
    (an expired proxy). Raises on an HTTP error."""
    import requests
    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    for status in ("active", "inactive"):
        _throttle()
        r = requests.get(f"{base}/v2/options/contracts", params={
            "underlying_symbols": underlying, "type": kind, "status": status,
            "expiration_date_gte": exp_gte, "expiration_date_lte": exp_lte,
            "strike_price_gte": f"{strike_gte:.2f}", "strike_price_lte": f"{strike_lte:.2f}", "limit": 1000,
        }, headers=_alpaca_headers(), timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca option contracts HTTP {r.status_code}: {r.text[:120]}")
        data = r.json()
        rows = data if isinstance(data, list) else (data or {}).get("option_contracts") or []
        if rows:
            return rows
    return []


# ── the proxy ─────────────────────────────────────────────────────────────────
def resolve_proxy(d: dict, fetch_contracts_fn: Callable, fetch_option_bars_fn: Callable,
                  start: str, end: str) -> Tuple[Optional[dict], Optional[str]]:
    """The listed contract an ATM proxy resolves to: the expiry nearest the
    middle of the DTE window, then the strike nearest the entry price (a tie
    goes to the lower) among the PROXY_CANDIDATES nearest that traded ON the
    entry day — entry-day facts only. (contract, None) or (None, why)."""
    rej = d.get("rejected") or {}
    p = rej.get("proxy") if isinstance(rej.get("proxy"), dict) else {}
    spot = _num(p.get("underlying_price"))
    entry = _day(d.get("entry_date"))
    right = "P" if str(p.get("right") or "C").upper() == "P" else "C"
    lo = int(_num(p.get("dte_min")) or DEFAULT_PROXY_DTE[0])
    hi = int(_num(p.get("dte_max")) or DEFAULT_PROXY_DTE[1])
    under = _text(p.get("underlying") or d.get("ticker"), 12).upper()
    if not spot or spot <= 0 or entry is None or not under:
        return None, "the entry price or day was not recorded"
    kind = "call" if right == "C" else "put"
    rows = fetch_contracts_fn(under, kind, (entry + timedelta(days=lo)).isoformat(),
                              (entry + timedelta(days=hi)).isoformat(),
                              spot * (1 - PROXY_STRIKE_BAND), spot * (1 + PROXY_STRIKE_BAND))
    listed = []
    for c in rows or []:
        occ = parse_occ((c or {}).get("symbol")) if isinstance(c, dict) else None
        if occ and occ["right"] == right:
            listed.append(occ)
    if not listed:
        return None, (f"no listed {kind} expiring {lo}-{hi} days after {entry.isoformat()} "
                      f"within {PROXY_STRIKE_BAND:.0%} of ${spot:.2f}")
    mid = (lo + hi) / 2.0
    expiry = min({o["expiry"] for o in listed},
                 key=lambda e: (abs((date.fromisoformat(e) - entry).days - mid), e))
    near = sorted((o for o in listed if o["expiry"] == expiry),
                  key=lambda o: (abs(o["strike"] - spot), o["strike"]))[:PROXY_CANDIDATES]
    bars = fetch_option_bars_fn([o["symbol"] for o in near], start, end) or {}
    for o in near:
        if _close_on(_sorted_bars(bars.get(o["symbol"])), entry.isoformat()) is not None:
            return {"symbol": o["symbol"], "strike": o["strike"], "right": right, "expiry": expiry,
                    "dte": (date.fromisoformat(expiry) - entry).days, "kind": p.get("kind") or f"atm_{kind}",
                    "description": describe_occ(o["symbol"])}, None
    return None, (f"none of the {len(near)} strikes nearest ${spot:.2f} expiring {expiry} "
                  f"traded on the entry day ({entry.isoformat()})")


# ── settling one decision ─────────────────────────────────────────────────────
def settle(d: dict, missing: List[str], proxy_row: Optional[dict], today: date,
           fetch_bars: Callable, fetch_option_bars_fn: Callable, fetch_contracts_fn: Callable,
           now=None) -> List[dict]:
    """The new rows (a proxy row at most once, then one outcome per horizon whose
    N-th trading day after the entry has closed). Raises on a fetch error."""
    entry = _day(d.get("entry_date"))
    ticker = _text(d.get("ticker"), 12).upper()
    if entry is None or not ticker:
        return []
    e = entry.isoformat()
    start = (entry - timedelta(days=WINDOW_PAD_DAYS)).isoformat()
    end = min(today, entry + timedelta(days=WINDOW_SPAN_DAYS)).isoformat()
    chosen = d.get("chosen") if isinstance(d.get("chosen"), dict) else {}
    rej = d.get("rejected") if isinstance(d.get("rejected"), dict) else {}
    share_syms = [ticker]
    if chosen.get("instrument") in ("stock", "etf"):
        share_syms.append(_text(chosen.get("symbol") or ticker, 12).upper())
    if rej.get("instrument") == "stock":
        share_syms.append(_text(rej.get("symbol") or ticker, 12).upper())
    bars = {s: _sorted_bars(fetch_bars(s, start, end)) for s in dict.fromkeys(share_syms)}
    cal = bars.get(ticker) or []
    i0 = _base_index(cal, e)
    due = [h for h in missing if i0 is not None and i0 + HORIZONS[h] < len(cal)]
    if not due:
        return []
    new: List[dict] = []
    # the rejected option leg: the selector's own contract, else the proxy
    rej_legs, rej_reason, proxy = None, None, False
    if rej.get("instrument") == "option":
        c = rej.get("contract") if isinstance(rej.get("contract"), dict) else None
        if c and parse_occ(c.get("symbol")):
            rej_legs = [{"symbol": parse_occ(c.get("symbol"))["symbol"], "side": "buy"}]
        else:
            proxy = True
            if proxy_row is None:
                contract, why = resolve_proxy(d, fetch_contracts_fn, fetch_option_bars_fn, start, end)
                proxy_row = {"v": SCHEMA_VERSION, "kind": "proxy", "cf_id": d.get("id"), "ts": now_iso(now),
                             "contract": contract, "reason": why}
                new.append(proxy_row)
            pc = proxy_row.get("contract") if isinstance(proxy_row.get("contract"), dict) else None
            if pc and parse_occ(pc.get("symbol")):
                rej_legs = [{"symbol": pc["symbol"], "side": "buy"}]
            else:
                rej_reason = proxy_row.get("reason") or "the proxy did not resolve to a listed contract"
    chosen_legs = [x for x in (chosen.get("legs") or []) if isinstance(x, dict)] \
        if chosen.get("instrument") == "option" else []
    opt_syms = [x["symbol"] for x in chosen_legs + (rej_legs or []) if parse_occ(x.get("symbol"))]
    obars = fetch_option_bars_fn(opt_syms, start, end) if opt_syms else {}
    for h in due:
        end_day = _bar_date(cal[i0 + HORIZONS[h]])
        if chosen.get("instrument") == "option":
            ch = option_position(chosen_legs, obars, cal, e, end_day, _capital_per_share(chosen))
        else:
            ch = share_leg(bars.get(_text(chosen.get("symbol") or ticker, 12).upper()), e, end_day,
                           chosen.get("side") or "buy")
        if rej.get("instrument") == "stock":
            rj = share_leg(bars.get(_text(rej.get("symbol") or ticker, 12).upper()), e, end_day,
                           rej.get("side") or "buy")
        elif rej_legs:
            rj = option_position(rej_legs, obars, cal, e, end_day)
        else:
            rj = _unpriced(rej_reason or "no contract to price")
        new.append(outcome_row(d, h, end_day, ch, rj, proxy, now))
    return new


def give_up_row(d: dict, horizon: str, now=None) -> dict:
    why = f"no price data within {GIVE_UP_DAYS} days of the entry"
    row = outcome_row(d, horizon, None, _unpriced(why), _unpriced(why), False, now)
    row["unpriced"] = True
    return row


# ── the nightly job ───────────────────────────────────────────────────────────
def backfill(max_decisions: int = 40, fetch_bars: Optional[Callable] = None,
             fetch_option_bars_fn: Optional[Callable] = None, fetch_contracts_fn: Optional[Callable] = None,
             now: Optional[datetime] = None, time_budget_s: float = 60.0) -> dict:
    """Settle every horizon the market has produced since each recorded decision.
    The fetchers are injectable for tests. Returns counts; never raises."""
    stats = {"enabled": True, "decisions": 0, "outcomes_written": 0, "proxies_written": 0, "pending": 0,
             "gave_up": 0, "fetch_errors": 0, "written": 0}
    if not enabled():
        stats["enabled"] = False
        return stats
    fetch_bars = fetch_bars or fetch_stock_bars
    fetch_option_bars_fn = fetch_option_bars_fn or fetch_option_bars
    fetch_contracts_fn = fetch_contracts_fn or fetch_contracts
    now = now or datetime.now(timezone.utc)
    today = now.date()
    decisions, outcomes, proxies = fold(read_rows())
    stats["decisions"] = len(decisions)
    started = time.monotonic()
    done = 0
    new_rows: List[dict] = []
    for d in sorted(decisions.values(), key=lambda r: str(r.get("ts") or "")):
        did = str(d.get("id"))
        missing = [h for h in HORIZONS if h not in outcomes.get(did, {})]
        entry = _day(d.get("entry_date"))
        if not missing or entry is None:
            continue
        age = (today - entry).days
        if age < 1 or done >= max_decisions or time.monotonic() - started > time_budget_s:
            stats["pending"] += 1
            continue
        done += 1
        got: List[dict] = []
        try:
            got = settle(d, missing, proxies.get(did), today, fetch_bars, fetch_option_bars_fn,
                         fetch_contracts_fn, now)
        except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            stats["fetch_errors"] += 1
            log.warning(f"[INSTRUMENT-CF] {d.get('ticker')} {did}: not settled tonight ({e})")
        new_rows.extend(got)
        stats["proxies_written"] += sum(1 for r in got if r.get("kind") == "proxy")
        settled = {r.get("horizon") for r in got if r.get("kind") == "outcome"}
        stats["outcomes_written"] += len(settled)
        left = [h for h in missing if h not in settled]
        if left and age > GIVE_UP_DAYS:
            new_rows.extend(give_up_row(d, h, now) for h in left)
            stats["gave_up"] += len(left)
        elif left:
            stats["pending"] += 1
    stats["written"] = append_rows(new_rows)
    return stats


if __name__ == "__main__":  # python3 instrument_counterfactual.py -> tonight's backfill stats
    logging.basicConfig(level=logging.INFO)
    log.info(json.dumps(backfill(), default=str))
