"""
entry_shadow.py — the ENTRY-SHADOW switch (human directive 2026-10-28, paper
account): "move the options sleeve and stock entries to shadow mode. Keep the
floor basket and all exits live."

Shadow mode = the bot computes the would-be entry exactly as before, LOGS it
(shadow_portfolio decision "shadow_mode_entry", so the +5/+10/+20d and
exit-aligned backfills measure its outcome) and places NO order.

SLEEVES (gating is by call site, not by symbol):
  "stocks"  — every new stock/ETF position the bot opens: executeTrades stock
              and ETF-substitution buys, the morning queue, the Tier-3 BUY,
              the third leg's SVXY buy (bot.ts + bot_engine._run_third_leg).
  "options" — every options OPEN: tier SELL_CSP / BUY_PUT, scanner-direct
              trades, evaluate_and_execute (CSPs, covered calls, spreads),
              the QQQ-put convexity overlay, and a roll's opening leg (a roll
              becomes close-only through options_manager's allow_rolls=False,
              the N15 halt path).
  The FLOOR (bot_engine._manage_spy_floor / _manage_defensive_floor: the
  FLOOR_BASKET legs + FLOOR_TICKER + DEFENSIVE_FLOOR_TICKER) is never gated,
  nor is any exit. `ledger_sleeve()` tags each record with the P&L ledger's
  own symbol classification (server/sleeveLedger.ts sleeveOf) for joins —
  note the ledger books SVXY and SPY under "floor" (bot.ts
  FLOOR_AND_LEG_TICKERS) although their entries here are "stocks".

FLAG: system_config BASE_CONFIG["SHADOW_ENTRY_SLEEVES"]; the env var
VOLTRADE_SHADOW_ENTRIES wins whenever it is PRESENT (same rule as
regime_engine.flag_raw) — "off"/"0"/""/"none"/"false" = nothing shadowed, so
a Railway env var rolls it back without a deploy. Node mirror:
server/entryShadow.ts (test_entry_shadow.py pins the default parity).

Never raises into a trading path: every public recorder swallows its own
errors (logging must not break the loop), and the flag readers fall back to
the config default.
"""
import logging
import os
import time
from datetime import datetime, timezone

log = logging.getLogger("voltrade.entry_shadow")

ENV_VAR = "VOLTRADE_SHADOW_ENTRIES"
CONFIG_KEY = "SHADOW_ENTRY_SLEEVES"
SLEEVES = ("stocks", "options")
OFF_VALUES = ("", "0", "off", "none", "false")
_ALIASES = {"stock": "stocks", "stocks": "stocks", "option": "options", "options": "options"}

SHADOW_ORDER_STATUS = "shadowed"


def _base_config() -> dict:
    try:
        from system_config import BASE_CONFIG
        return BASE_CONFIG
    except Exception as e:
        log.warning(f"[ENTRY-SHADOW] system_config unavailable ({e}); entries without an env var are LIVE")
        return {}


def flag_raw(env=None, config=None):
    """(raw value, source): the env var when PRESENT (any value wins), else the
    BASE_CONFIG default with source 'config'."""
    env = os.environ if env is None else env
    v = env.get(ENV_VAR)
    if v is not None:
        return str(v), "env"
    cfg = _base_config() if config is None else config
    d = cfg.get(CONFIG_KEY)
    if d is None or d is False:
        return "", "config"
    return str(d), "config"


def parse_sleeves(raw) -> set:
    """'stocks,options' -> {'stocks','options'}; an off value -> set(); 'all' -> both.
    Unknown tokens are ignored (status() reports them)."""
    v = str(raw if raw is not None else "").strip().lower()
    if v in OFF_VALUES:
        return set()
    out = set()
    for tok in v.replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok == "all":
            out |= set(SLEEVES)
        elif tok in _ALIASES:
            out.add(_ALIASES[tok])
    return out


def shadowed_sleeves(env=None, config=None) -> set:
    raw, _src = flag_raw(env, config)
    return parse_sleeves(raw)


def is_shadowed(sleeve: str, env=None, config=None) -> bool:
    try:
        return _ALIASES.get(str(sleeve).lower(), str(sleeve)) in shadowed_sleeves(env, config)
    except Exception as e:  # never break a trading path on the flag read
        log.warning(f"[ENTRY-SHADOW] flag read failed ({e}); treating {sleeve} as live")
        return False


def status(env=None, config=None) -> dict:
    """{stocks, options, source, value} — what the gates act on NOW."""
    raw, src = flag_raw(env, config)
    on = parse_sleeves(raw)
    return {"stocks": "stocks" in on, "options": "options" in on, "source": src, "value": raw}


# ── classification (the P&L ledger's, for joins) ─────────────────────────────
def _is_occ(symbol: str) -> bool:
    s = str(symbol or "")
    return len(s) > 15 and s[-15:-9].isdigit() and s[-9] in "CP" and s[-8:].isdigit()


def floor_ledger_tickers(config=None) -> set:
    """bot_engine.FLOOR_AND_LEG_TICKERS / bot.ts FLOOR_AND_LEG_TICKERS — the set
    server/sleeveLedger.ts books as the "floor" sleeve."""
    cfg = _base_config() if config is None else config
    return ({"SVXY", "SPY"}
            | {t for t in (cfg.get("FLOOR_BASKET") or {}) if t != "CASH"}
            | {cfg.get("FLOOR_TICKER", "QQQ")}
            | {cfg.get("DEFENSIVE_FLOOR_TICKER", "GLD")})


def ledger_sleeve(symbol: str, option=None, config=None) -> str:
    """sleeveLedger.sleeveOf: OCC -> options; floor/leg ETF -> floor; else stocks."""
    if option if option is not None else _is_occ(symbol):
        return "options"
    return "floor" if str(symbol or "").upper() in floor_ledger_tickers(config) else "stocks"


# ── recording ────────────────────────────────────────────────────────────────
def et_day(now=None) -> str:
    """The ET trading date (YYYY-MM-DD) a shadow entry is deduplicated on."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _num(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default  # NaN -> default
    except (TypeError, ValueError):
        return default


def build_entry(*, sleeve: str, ticker: str, source: str, side: str = "buy", qty=0, price=0.0,
                notional=None, score=0.0, instrument=None, instrument_price=None, contract=None,
                relabel: bool = False, reason: str = "", trade_type=None, action_label=None,
                now=None, env=None) -> dict:
    """One would-be entry in the shape shadow_portfolio.log_shadow_entries takes.
    `ticker`/`price` are the UNDERLYING the bot scored (what the backfills
    measure); `instrument` is what would have been ordered (ETF, OCC, ...).
    `relabel`: the candidate already has a fresh deep_score "taken" record —
    relabel it instead of appending a second record for the same decision."""
    t = str(ticker or "").upper()
    q = _num(qty)
    px = _num(price)
    inst = str(instrument or t)
    inst_px = _num(instrument_price, px) if instrument_price is not None else px
    value = _num(notional, q * inst_px) if notional is not None else round(q * inst_px, 2)
    _raw, flag_src = flag_raw(env)
    details = {
        "sleeve": sleeve,
        "ledger_sleeve": ledger_sleeve(inst),
        "source": str(source)[:80],
        "day": et_day(now),
        "side": str(side or ""),
        "qty": q,
        "notional": round(value, 2),
        "instrument": inst,
        "instrument_price": round(inst_px, 4),
        "flag_source": flag_src,
    }
    if contract:
        details["contract"] = contract
    return {
        "ticker": t,
        "sleeve": sleeve,
        "day": details["day"],
        "score": _num(score),
        "price": px,
        "side": str(side or ""),
        "trade_type": trade_type or ("options" if sleeve == "options" else "stock"),
        "action_label": action_label or ("SHADOW " + str(side or "buy").upper()),
        "reason": (str(reason) or f"entry shadow mode: {sleeve} via {source}")[:200],
        "relabel": bool(relabel),
        "details": details,
    }


# In-process once-per-day memory: lets a daemon-resident caller (third leg,
# convexity overlay, rolls — they re-propose the same entry every scan) skip
# the ~45MB shadow-log load for an entry it already recorded today. The log
# itself is the persistent dedupe (shadow_portfolio.log_shadow_entries).
_SEEN = {"day": None, "keys": set()}


def _seen_key(e: dict) -> str:
    return f"{e.get('sleeve')}|{e.get('ticker')}"


def record_entries(entries: list) -> dict:
    """Record would-be entries (one shadow-log load/save). Returns
    {"received", "recorded", "repeats", "results": [{ticker, sleeve, source, new}], ...}.
    Never raises."""
    out = {"received": 0, "recorded": 0, "repeats": 0, "results": []}
    try:
        if not isinstance(entries, list) or not entries:
            return out
        out["received"] = len(entries)
        todo = []
        for e in entries:
            if not isinstance(e, dict) or not e.get("ticker") or e.get("sleeve") not in SLEEVES:
                continue
            day = e.get("day") or et_day()
            if _SEEN["day"] != day:
                _SEEN["day"], _SEEN["keys"] = day, set()
            if _seen_key(e) in _SEEN["keys"] and not e.get("relabel"):
                out["repeats"] += 1
                out["results"].append({"ticker": e["ticker"], "sleeve": e["sleeve"],
                                       "source": (e.get("details") or {}).get("source"), "new": False})
                continue
            todo.append(e)
        if todo:
            import shadow_portfolio
            res = shadow_portfolio.log_shadow_entries(todo)
            for r in res.get("results", []):
                out["results"].append(r)
                if r.get("new"):
                    out["recorded"] += 1
                else:
                    out["repeats"] += 1
                _SEEN["keys"].add(f"{r.get('sleeve')}|{r.get('ticker')}")
            out["saved"] = res.get("saved", False)
            if res.get("error"):
                out["error"] = res["error"]
        for r in out["results"]:
            if r.get("new"):
                log.info(f"[ENTRY-SHADOW] {r.get('sleeve')} {r.get('ticker')} via {r.get('source')}: "
                         f"would-be entry logged, no order placed")
        return out
    except Exception as ex:
        log.warning(f"[ENTRY-SHADOW] record failed (no order was placed either way): {ex}")
        out["error"] = str(ex)[:200]
        return out


def record_entry(**kw) -> dict:
    """build_entry + record_entries for one entry; returns its result row."""
    try:
        e = build_entry(**kw)
        res = record_entries([e])
        rows = res.get("results") or []
        row = dict(rows[0]) if rows else {"ticker": e["ticker"], "sleeve": e["sleeve"], "new": False}
        if res.get("error"):
            row["error"] = res["error"]
        return row
    except Exception as ex:
        log.warning(f"[ENTRY-SHADOW] record_entry failed: {ex}")
        return {"ticker": str(kw.get("ticker", "")).upper(), "sleeve": kw.get("sleeve"), "new": False,
                "error": str(ex)[:200]}


def record_node_entries(entries: list) -> dict:
    """bot.ts's batch recorder (daemon route "entry_shadow_record"): server/
    entryShadow.ts NodeShadowEntry dicts -> build_entry -> record_entries."""
    built = []
    for e in entries if isinstance(entries, list) else []:
        if not isinstance(e, dict) or e.get("sleeve") not in SLEEVES or not e.get("ticker"):
            continue
        try:
            built.append(build_entry(
                sleeve=e["sleeve"], ticker=e["ticker"], source=str(e.get("source") or "bot.ts"),
                side=str(e.get("side") or "buy"), qty=e.get("qty", 0), price=e.get("price", 0.0),
                notional=e.get("notional"), score=e.get("score", 0.0), instrument=e.get("instrument"),
                instrument_price=e.get("instrument_price"), relabel=bool(e.get("relabel")),
                reason=str(e.get("reason") or "")))
        except Exception as ex:
            log.warning(f"[ENTRY-SHADOW] bad node entry {e.get('ticker')}: {ex}")
    return record_entries(built)


# ── options helpers ──────────────────────────────────────────────────────────
_CONTRACT_KEYS = ("occ_symbol", "short_call", "long_call", "short_put", "long_put", "call_leg", "put_leg",
                  "strike", "expiry", "option_type", "side", "qty", "limit_price", "net_credit", "credit",
                  "max_cost", "max_loss", "cash_required", "delta", "strategy")


def contract_details(contract: dict, strategy: str = "") -> dict:
    """The JSON-safe subset of a select_contract() result a shadow record keeps."""
    out = {}
    if not isinstance(contract, dict):
        return out
    for k in _CONTRACT_KEYS:
        v = contract.get(k)
        if v is None or v == "":
            continue
        if isinstance(v, dict):
            v = v.get("occ_symbol") or v.get("symbol") or ""
            if not v:
                continue
        out[k] = v if isinstance(v, (int, float, str, bool)) else str(v)
    if strategy and "strategy" not in out:
        out["strategy"] = strategy
    return out


def _underlying_price(ticker: str) -> float:
    """Latest trade for the underlying (market-data GET only) — the shadow
    record's entry_price when the caller had none (tier CSPs pass price 0)."""
    try:
        import requests
        from options_execution import ALPACA_DATA, _alpaca_headers
        r = requests.get(f"{ALPACA_DATA}/v2/stocks/{ticker}/snapshot", headers=_alpaca_headers(), timeout=5)
        if r.status_code != 200:
            return 0.0
        d = r.json() or {}
        return _num((d.get("latestTrade") or {}).get("p") or (d.get("dailyBar") or {}).get("c"))
    except Exception as e:
        log.debug(f"[ENTRY-SHADOW] underlying price for {ticker} unavailable: {e}")
        return 0.0


def shadow_option_open(contract: dict, ticker: str, price: float, source: str, score=0.0,
                       strategy: str = "", relabel: bool = False, reason: str = "") -> dict:
    """The options gate's replacement for submit_options_order(contract) while
    the options sleeve is shadowed: record the would-be open, return an
    order-shaped dict with status "shadowed" (no order_id — nothing rests)."""
    c = contract_details(contract, strategy)
    px = _num(price) or _underlying_price(str(ticker).upper())
    qty = _num((contract or {}).get("qty"), 1.0) or 1.0
    limit_px = _num((contract or {}).get("limit_price"))
    notional = _num((contract or {}).get("cash_required")) or _num((contract or {}).get("max_cost")) \
        or round(limit_px * 100 * qty, 2)
    row = record_entry(sleeve="options", ticker=ticker, source=source, side=str((contract or {}).get("side") or "sell"),
                       qty=qty, price=px, notional=notional, score=score,
                       instrument=c.get("occ_symbol") or str(ticker).upper(), instrument_price=limit_px,
                       contract=c, relabel=relabel, reason=reason, trade_type="options",
                       action_label=("SHADOW " + str(strategy or c.get("strategy") or "OPTIONS")).upper()[:40])
    return {"status": SHADOW_ORDER_STATUS, "shadow": True, "new": bool(row.get("new")),
            "ticker": str(ticker).upper(), "occ_symbol": c.get("occ_symbol", ""), "qty": qty,
            "limit_price": limit_px, "detail": "entry shadow mode (options): would-be open logged, no order placed",
            **({"error": row["error"]} if row.get("error") else {})}


def roll_skip_entries(actions: list, source: str = "options_manager.roll") -> list:
    """Would-be ROLL opens skipped by options_manager while options are shadowed
    (their close half stays live): one entry per close whose reason carries
    options_manager.ROLL_SKIPPED_SHADOW_DETAIL."""
    out = []
    try:
        from options_manager import ROLL_SKIPPED_SHADOW_DETAIL
    except Exception as e:
        log.warning(f"[ENTRY-SHADOW] options_manager unavailable, roll opens not logged: {e}")
        return out
    for a in actions or []:
        if not isinstance(a, dict) or ROLL_SKIPPED_SHADOW_DETAIL not in str(a.get("reason", "")):
            continue
        t = str(a.get("ticker") or "").upper()
        if not t:
            continue
        out.append(build_entry(sleeve="options", ticker=t, source=source, side="sell", qty=1,
                               price=_underlying_price(t), notional=0.0, instrument=t,
                               contract={"rolled_from": str(a.get("occ_symbol") or ""),
                                         "note": "roll open skipped; replacement contract not selected"},
                               trade_type="options", action_label="SHADOW ROLL",
                               reason=f"roll open shadowed ({a.get('type', 'close')}); close stayed live"))
    return out
