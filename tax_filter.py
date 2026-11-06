"""
tax_filter.py — the tax-aware entry filter and its decision log (refocus spec
2026-11-05 §6 and pipeline step 8; [RULE-REVIEW] 2026-11-06).

THE QUESTION. Before every NEW entry: after costs and short-term tax, does
this trade's expected return over its expected holding period beat simply
holding SPY for the same days (whose gain is taxed only when it is eventually
sold, at the long-term rate)? A trade whose after-tax edge is not above a small
margin is skipped. Every decision, allowed or skipped, is logged
(DATA_DIR/voltrade_tax_decisions.jsonl, served by GET /api/bot/tax).

THE ESTIMATE. Every input is a number the system already measures:
  expected  p·W − (1−p)·L per trade: the bucket's win rate p, average win W and
            average loss L from position_sizing._get_historical_stats(), the
            numbers the Kelly sizer sizes the same trade with (trade feedback
            when the bucket has it, else the documented backtest priors). The
            bucket is position_sizing._infer_strategy's.
  holding   the mean calendar days held in the sample those stats were measured
            on (TAX_FILTER.HOLD_DAYS); an option is never held past expiry.
  costs     the repo's round-trip cost model for the trade: stocks/ETFs
            position_sizing._estimate_costs (as sizing computed it); options the
            fee per contract (position_sizing.OPTIONS_FEE_PER_CONTRACT) plus
            the entry walk (options_execution._optimized_limit_price, 30 % of
            the half-spread) at the widest spread the liquidity filter admits
            (options_execution.MAX_SPREAD_PCT), both sides.
  tax       the short-term rate on the result (the long-term rate past a year);
            a negative expectation is reduced the same way — losses are assumed
            to offset other short-term gains (the linear model).
  SPY       TAX_FILTER.SPY_EXPECTED_RETURN_ANNUAL_PCT × days / 365, taxed at the
            long-term rate (buy-and-hold defers its tax to an eventual long-term
            sale; the time value of that deferral is NOT credited, which
            flatters the trade very slightly).
  allowed   afterTax − spyAfterTax > MIN_EDGE_ANNUAL_PCT × days / 365.

NOT USED, ON PURPOSE. The composite score: the 20,000-record shadow study found
it ANTI-PREDICTIVE (research/open_questions.md [CRITICAL 2026-10-26]), so no
score -> return mapping is validated and a high score earns no extra expected
return here. The stop/target geometry: the live exits are scale-outs in thirds,
trailing and time stops, so a trade's TP/SL distances are not its payoff; the
bucket's measured average win/loss already reflect the exit mix the bot runs.

BIASES, STATED (MEASUREMENT INTEGRITY):
  - the bucket stats are already net of the costs their own trades paid
    (backtest net_pct; live fill prices), so subtracting this trade's modeled
    cost counts costs about twice — stricter, by the spec's "costs first";
    TAX_FILTER.SUBTRACT_COSTS turns it off.
  - the Kelly priors come from backtest_10yr_results.json (an engine not in the
    repo; research/bot_cockpit_audit.md calls it a known-weak bar). The filter
    answers the TAX question on the sizer's own expectancy; whether that
    expectancy is right is the sizer's question (GET /api/bot/tax shows the
    live broker expectancy next to it).
  - SPY's return is an assumed long-run average, not a forecast.
  - every number is an estimate at assumed rates: the paper account pays no tax.

Never raises into a trading path: a failure inside a gate_* helper returns an
ALLOWED decision carrying the error (the filter must not silently stop trading,
and the logged row means it cannot hide either).
"""
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

log = logging.getLogger("voltrade.tax_filter")

SCHEMA = "tax_decision/v1"
DECISIONS_FILE = "voltrade_tax_decisions.jsonl"
LONG_TERM_DAYS = 365          # "more than one year" — server/taxLedger.ts LONG_TERM_MS
# options_execution._optimized_limit_price walks 30 % of the half-spread from
# the mid toward the natural side (its local WALK_PCT); test_tax_filter.py pins it.
OPTION_ENTRY_WALK = 0.30
# a dedupe read looks this far back into the log (bytes): one row per
# (UTC day, ticker, kind, instrument, source, verdict) — a scan repeats its
# candidates every few minutes and the log is a decision record, not a tick log
_DEDUPE_TAIL_BYTES = 256 * 1024

# Mirrors system_config BASE_CONFIG["TAX_FILTER"] (the source of truth) for a
# process where system_config cannot be imported; test_tax_filter.py pins parity.
DEFAULTS = {
    "ENABLED": True,
    "SHORT_TERM_RATE": 0.24,
    "LONG_TERM_RATE": 0.15,
    "SPY_EXPECTED_RETURN_ANNUAL_PCT": 10.0,
    "MIN_EDGE_ANNUAL_PCT": 2.0,
    "SUBTRACT_COSTS": True,
    "HOLD_DAYS": {"stocks": 6.36, "etf": 6.76, "csp_options": 6.39, "vrp": 6.45},
    "LOG_MAX_BYTES": 4 * 1024 * 1024,
    "LT_DEFER_ENABLED": True,
    "LT_DEFER_WINDOW_DAYS": 45,
    "HARVEST_ENABLED": True,
    "HARVEST_MIN_NET_USD": 20.0,
    "HARVEST_MAX_PER_SCAN": 2,
    "REALIZED_MAX_AGE_H": 48,
}

# Option strategies that SELL premium (position_sizing._infer_strategy's
# csp_options list and seed_feedback_from_backtest's mapping, debit structures
# excluded). Long premium has no measured win rate/payoff anywhere in the repo:
# those opens are allowed and logged as "not estimated", never guessed.
PREMIUM_SELLING = ("sell_cash_secured_put", "covered_call", "bull_put_credit_spread",
                   "iron_condor", "qqq_iron_condor", "short_straddle")
_LEG_KEYS = ("short_call", "long_call", "short_put", "long_put", "call_leg", "put_leg",
             "long_leg", "short_leg")


# ── configuration ──────────────────────────────────────────────────────────

def config(overrides: dict = None) -> dict:
    """DEFAULTS overlaid with system_config BASE_CONFIG["TAX_FILTER"], then `overrides`."""
    cfg = dict(DEFAULTS)
    try:
        from system_config import BASE_CONFIG
        user = BASE_CONFIG.get("TAX_FILTER")
        if isinstance(user, dict):
            cfg.update(user)
    except ImportError as e:
        log.warning(f"[TAX] system_config unavailable ({e}); using tax_filter.DEFAULTS")
    if overrides:
        cfg.update(overrides)
    return cfg


def rates(cfg: dict = None) -> dict:
    c = cfg or config()
    return {"shortTerm": float(c["SHORT_TERM_RATE"]), "longTerm": float(c["LONG_TERM_RATE"])}


def _data_dir() -> str:
    try:
        from storage_config import DATA_DIR
        return DATA_DIR
    except ImportError:
        return "/data/voltrade" if os.path.isdir("/data") else "/tmp"


def decisions_path() -> str:
    return os.path.join(_data_dir(), DECISIONS_FILE)


# ── the math (pure) ────────────────────────────────────────────────────────

def expectancy_pct(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """p·W − (1−p)·L, percent per trade. avg_loss is a magnitude (the sizer's convention)."""
    p = float(win_rate)
    return p * float(avg_win) - (1.0 - p) * abs(float(avg_loss))


def holding_term(days: float) -> str:
    """"long" only for MORE than one year held (server/taxLedger.ts termOf)."""
    return "long" if float(days) > LONG_TERM_DAYS else "short"


def spy_leg(holding_days: float, cfg: dict) -> tuple:
    """(SPY expected return %, after long-term tax %) over `holding_days` calendar days."""
    pct = float(cfg["SPY_EXPECTED_RETURN_ANNUAL_PCT"]) * float(holding_days) / 365.0
    return pct, pct * (1.0 - float(cfg["LONG_TERM_RATE"]))


def assess(expected_pct: float, holding_days: float, cost_pct: float = 0.0, cfg: dict = None,
           max_gain_pct: float = None) -> dict:
    """The numbers behind one decision. `expected_pct` is the per-trade expectancy
    (percent of the money committed); `max_gain_pct` caps it (a sold option cannot
    earn more than its premium); `cost_pct` is the modeled round trip."""
    c = cfg or config()
    days = max(float(holding_days), 0.0)
    gross = float(expected_pct)
    capped = max_gain_pct is not None and gross > float(max_gain_pct)
    if capped:
        gross = float(max_gain_pct)
    cost = float(cost_pct or 0.0) if c.get("SUBTRACT_COSTS", True) else 0.0
    net = gross - cost
    term = holding_term(days)
    rate = float(c["LONG_TERM_RATE"] if term == "long" else c["SHORT_TERM_RATE"])
    after = net * (1.0 - rate)
    spy_pct, spy_after = spy_leg(days, c)
    margin = float(c["MIN_EDGE_ANNUAL_PCT"]) * days / 365.0
    edge = after - spy_after
    return {
        "expectedPct": round(float(expected_pct), 4), "capped": capped, "costPct": round(cost, 4),
        "netPct": round(net, 4), "afterTaxPct": round(after, 4), "taxRate": rate, "term": term,
        "spyPct": round(spy_pct, 4), "spyAfterTaxPct": round(spy_after, 4),
        "edgePp": round(edge, 4), "marginPp": round(margin, 4), "holdingDays": round(days, 2),
        "wouldAllow": edge > margin,
    }


def _f(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}"


# plain-English names of the sizing buckets for the reasons
BUCKET_LABELS = {"stocks": "Stock", "etf": "Leveraged-ETF", "csp_options": "Option-selling",
                 "vrp": "Volatility-premium", "overall": "All"}


def reason_for(a: dict, label: str, stats: dict) -> str:
    """Plain English for one assessed decision."""
    d = a["holdingDays"]
    days = f"{d:.0f} day{'s' if round(d) != 1 else ''}"
    src = stats.get("source") or "the sizer's stats"
    wr = stats.get("winRate")
    basis = (f"{label} trades average {_f(a['expectedPct'])}% per trade over about {days} ({src}"
             + (f", win rate {wr * 100:.0f}%, avg win +{stats.get('avgWin', 0):.2f}%, avg loss -{abs(stats.get('avgLoss', 0)):.2f}%" if wr is not None else "")
             + ")")
    cap = " (capped at the premium this contract can earn)" if a.get("capped") else ""
    spy = f"holding SPY for the same {days} is expected to return {_f(a['spyAfterTaxPct'])}% after {a.get('ltRatePct', 15):.0f}% long-term tax"
    tax = f"{a['taxRate'] * 100:.0f}% {'long' if a['term'] == 'long' else 'short'}-term tax"
    if a["netPct"] <= 0:
        verdict = "Skipped" if not a["wouldAllow"] else "Allowed"
        return (f"{verdict}: {basis}{cap}; after {a['costPct']:.2f}% costs that is {_f(a['netPct'])}%, "
                f"a loss before tax, while {spy}.")
    body = (f"{basis}{cap}; after {a['costPct']:.2f}% costs and {tax} that is {_f(a['afterTaxPct'])}%, "
            f"vs {_f(a['spyAfterTaxPct'])}% for holding SPY the same {days} after "
            f"{a.get('ltRatePct', 15):.0f}% long-term tax: an edge of {_f(a['edgePp'])} pp")
    if a["wouldAllow"]:
        return f"Allowed: {body} (needs more than {_f(a['marginPp'], 3)} pp)."
    return f"Skipped: {body}, not more than the {_f(a['marginPp'], 3)} pp margin."


# ── the sizer's stats (lazy, cached per feedback-file version) ─────────────

_STATS = {"key": None, "at": 0.0, "stats": None}
_STATS_TTL_S = 60.0


def sizing_stats() -> dict:
    """position_sizing._get_historical_stats(), re-read at most every 60 s or when
    the trade-feedback file changes (a scan asks once per candidate)."""
    try:
        from storage_config import TRADE_FEEDBACK_PATH
        key = os.path.getmtime(TRADE_FEEDBACK_PATH) if os.path.exists(TRADE_FEEDBACK_PATH) else None
    except (ImportError, OSError):
        key = None
    now = time.time()
    if _STATS["stats"] is not None and _STATS["key"] == key and now - _STATS["at"] < _STATS_TTL_S:
        return _STATS["stats"]
    from position_sizing import _get_historical_stats
    stats = _get_historical_stats()
    _STATS.update(key=key, at=now, stats=stats)
    return stats


def bucket_stats(bucket: str, stats: dict = None) -> dict:
    """{winRate, avgWin, avgLoss, n, source, bucket} for a sizing bucket (the
    overall row when the bucket is unknown to the stats)."""
    s = stats if stats is not None else sizing_stats()
    by = (s or {}).get("by_strategy") or {}
    row = by.get(bucket)
    used = bucket
    if not isinstance(row, dict):
        row, used = (s or {}).get("overall") or {}, "overall"
    n = int(row.get("trades", row.get("total_trades", 0)) or 0)
    return {
        "bucket": used, "winRate": float(row.get("win_rate", 0.0) or 0.0),
        "avgWin": float(row.get("avg_win", 0.0) or 0.0), "avgLoss": abs(float(row.get("avg_loss", 0.0) or 0.0)),
        "n": n,
        "source": (f"trade feedback, {n} closed trades" if n > 0
                   else "Kelly prior from the 2016-2026 backtest"),
    }


def bucket_for(trade: dict) -> str:
    """The sizing bucket (position_sizing._infer_strategy) — the one the sizer used."""
    from position_sizing import _infer_strategy
    return _infer_strategy(trade or {})


# ── decisions ──────────────────────────────────────────────────────────────

def _now_iso(now: datetime = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def _base_row(ticker, side, kind, instrument, source, cfg, now=None) -> dict:
    return {
        "schema": SCHEMA, "ts": _now_iso(now), "ticker": str(ticker or "").upper(), "side": str(side or "buy"),
        "kind": kind, "instrument": instrument, "source": source,
        "enforced": bool(cfg.get("ENABLED", True)), "rates": rates(cfg),
        "spyAssumptionAnnualPct": float(cfg["SPY_EXPECTED_RETURN_ANNUAL_PCT"]),
    }


def decide(*, ticker: str, side: str, instrument: str, bucket: str, stats: dict, cost_pct: float,
           holding_days: float = None, max_gain_pct: float = None, source: str, label: str = None,
           cfg: dict = None, now: datetime = None, extra: dict = None) -> dict:
    """One priced entry decision: the assessed numbers + a plain-English reason.
    `allowed` is the verdict when the filter is enforced, else True (and
    `wouldAllow` keeps the verdict for the counterfactual)."""
    c = cfg or config()
    hold = float(holding_days) if holding_days is not None else float((c.get("HOLD_DAYS") or {}).get(bucket, DEFAULTS["HOLD_DAYS"]["vrp"]))
    expected = expectancy_pct(stats["winRate"], stats["avgWin"], stats["avgLoss"])
    a = assess(expected, hold, cost_pct, c, max_gain_pct=max_gain_pct)
    a["ltRatePct"] = float(c["LONG_TERM_RATE"]) * 100.0
    row = _base_row(ticker, side, "entry", instrument, source, c, now)
    row.update({k: v for k, v in a.items() if k != "ltRatePct"})
    row.update({"bucket": bucket, "stats": stats, "priced": True})
    row["allowed"] = bool(a["wouldAllow"]) or not row["enforced"]
    reason = reason_for(a, label or BUCKET_LABELS.get(bucket, bucket), stats)
    if not row["enforced"] and not a["wouldAllow"]:
        reason = "Filter off (TAX_FILTER.ENABLED=False), so not skipped; it would have been — " + reason
    row["reason"] = reason
    if extra:
        row.update(extra)
    return row


def decide_unpriced(*, ticker: str, side: str, instrument: str, why: str, source: str,
                    cfg: dict = None, now: datetime = None, error: str = None) -> dict:
    """An entry the filter cannot price (no measured stats, a hedge, an error):
    allowed, with the reason why — never a guessed number."""
    c = cfg or config()
    row = _base_row(ticker, side, "entry", instrument, source, c, now)
    row.update({"allowed": True, "wouldAllow": True, "priced": False, "expectedPct": None, "afterTaxPct": None,
                "spyAfterTaxPct": None, "holdingDays": None, "reason": why})
    if error:
        row["error"] = str(error)[:200]
    return row


def decide_blocked(*, ticker: str, side: str, instrument: str, why: str, source: str,
                   cfg: dict = None, now: datetime = None) -> dict:
    """An entry refused outright (a harvested symbol inside its wash-sale
    window). Independent of TAX_FILTER.ENABLED: the harvested loss depends on it."""
    c = cfg or config()
    row = _base_row(ticker, side, "entry", instrument, source, c, now)
    row.update({"allowed": False, "wouldAllow": False, "priced": False, "expectedPct": None, "afterTaxPct": None,
                "spyAfterTaxPct": None, "holdingDays": None, "reason": why, "washSale": True})
    return row


# ── the decision log ───────────────────────────────────────────────────────

def _dedupe_key(row: dict) -> tuple:
    return (str(row.get("ts", ""))[:10], row.get("ticker"), row.get("kind"), row.get("instrument"),
            row.get("strategy"), row.get("source"), bool(row.get("allowed")), bool(row.get("wouldAllow")))


def _seen_today(path: str, key: tuple) -> bool:
    """True when the log's tail already holds a row with this dedupe key."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(max(0, size - _DEDUPE_TAIL_BYTES))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError as e:
        log.warning(f"[TAX] decision log unreadable for dedupe ({e}); logging the row")
        return False
    for line in reversed(tail):
        prev = _json_line(line)
        if isinstance(prev, dict) and _dedupe_key(prev) == key:
            return True
    return False


def _json_line(line: str):
    """One JSONL line parsed, or None for a torn line (a partial write, or the
    cut first line of a tail window) — skipped by every reader, never fatal."""
    try:
        return json.loads(line)
    except ValueError:
        return None


def log_decision(row: dict, path: str = None, dedupe: bool = True, max_bytes: int = None) -> bool:
    """Append one decision row (JSONL). Once per (UTC day, ticker, kind, instrument,
    source, verdict) when `dedupe`; rotated to <path>.1 above max_bytes (one
    generation kept). Returns whether a row was written. Never raises."""
    try:
        p = path or decisions_path()
        if dedupe and _seen_today(p, _dedupe_key(row)):
            return False
        cap = int(max_bytes or config().get("LOG_MAX_BYTES") or DEFAULTS["LOG_MAX_BYTES"])
        if os.path.exists(p) and os.path.getsize(p) > cap:
            os.replace(p, p + ".1")
        with open(p, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return True
    except Exception as e:   # logging must never break a trading path — but it is reported
        log.warning(f"[TAX] decision not logged for {row.get('ticker')}: {type(e).__name__}: {e}")
        return False


SUMMARY_KEYS = ("allowed", "wouldAllow", "enforced", "priced", "expectedPct", "costPct", "afterTaxPct",
                "spyAfterTaxPct", "edgePp", "marginPp", "holdingDays", "bucket", "reason")


def summary(row: dict) -> dict:
    """The compact form a trade or contract carries downstream (bot.ts, the journal)."""
    return {k: row.get(k) for k in SUMMARY_KEYS if k in (row or {})}


def filter_tier_buys(actions: list, stats: dict = None, cfg: dict = None, path: str = None) -> tuple:
    """(kept actions, skipped decision rows): every tier-engine BUY through
    gate_tier_buy; every other action (SELL_CSP, BUY_PUT — option opens, judged
    at contract selection) passes untouched."""
    kept, skipped = [], []
    for a in actions or []:
        if str(getattr(a, "action", "")).upper() != "BUY":
            kept.append(a)
            continue
        row = gate_tier_buy(str(getattr(a, "ticker", "")), str(getattr(a, "strategy", "")),
                            source=f"tier{getattr(a, 'tier', '')}.BUY", stats=stats, cfg=cfg, path=path)
        if row.get("allowed", True):
            kept.append(a)
        else:
            skipped.append(row)
    return kept, skipped


def recent_decisions(limit: int = 100, path: str = None) -> list:
    """The newest `limit` rows, newest first (torn lines skipped)."""
    p = path or decisions_path()
    try:
        with open(p) as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        row = _json_line(line)
        if isinstance(row, dict):
            out.append(row)
            if len(out) >= limit:
                break
    return out


# ── gates (the call sites' one-liners) ─────────────────────────────────────

def gate_scan_trade(trade: dict, sizing: dict, sizing_input: dict = None, source: str = "scan_market",
                    stats: dict = None, cfg: dict = None, path: str = None) -> dict:
    """Layer A — a stock-loop trade scan_market is about to return, after sizing.

    Judges the order the trade will place in shares: the ETF for an ETF
    substitution, else the stock. A trade whose chosen instrument is OPTIONS is
    judged here on its STOCK FALLBACK (bot.ts buys the stock when the option
    cannot be placed); the option itself is judged at contract selection
    (gate_option_contract). Returns the logged decision; fails open."""
    t = trade or {}
    ticker = t.get("ticker", "")
    try:
        c = cfg or config()
        instrument = str(t.get("instrument") or "stock")
        etf = instrument == "etf" and t.get("etf_shares")
        if etf:
            bucket, order_ticker, inst = "etf", t.get("instrument_ticker") or ticker, "etf"
        else:
            bucket = bucket_for(sizing_input if sizing_input is not None else t)
            order_ticker, inst = ticker, ("stock_fallback" if instrument == "options" else "stock")
        cost = float(((sizing or {}).get("costs") or {}).get("total_cost_pct") or 0.0)
        blk = wash_block(order_ticker)
        if blk:
            row = decide_blocked(ticker=order_ticker, side=t.get("side", "buy"), instrument=inst, source=source,
                                 why=wash_reason(order_ticker, blk), cfg=c)
            row["bucket"] = bucket
            log_decision(row, path=path)
            return row
        row = decide(ticker=order_ticker, side=t.get("side", "buy"), instrument=inst, bucket=bucket,
                     stats=bucket_stats(bucket, stats), cost_pct=cost, source=source, cfg=c,
                     label=(BUCKET_LABELS["etf"] if etf else None),
                     extra={"underlying": ticker, "score": t.get("score")})
    except Exception as e:   # fail open, logged (never a silent skip, never a crash)
        log.warning(f"[TAX] scan gate failed for {ticker}: {type(e).__name__}: {e}")
        row = decide_unpriced(ticker=ticker, side=t.get("side", "buy"), instrument="stock", source=source,
                              why="Allowed (not estimated): the tax estimate failed — see error.",
                              error=f"{type(e).__name__}: {e}", cfg=cfg)
    log_decision(row, path=path)
    return row


def gate_tier_buy(ticker: str, strategy: str = "", source: str = "tier_engine.BUY", stats: dict = None,
                  cfg: dict = None, path: str = None) -> dict:
    """Layer A — a tier-engine BUY (T3 trend capture: leveraged index ETFs). No
    sizing has run yet, so the cost is the repo model with volume unknown."""
    try:
        from position_sizing import _estimate_costs
        bucket = bucket_for({"ticker": ticker, "instrument": "etf" if strategy == "trend_long" else ""})
        cost = float(_estimate_costs(1, 1.0, 0, "stock")["total_cost_pct"])
        blk = wash_block(ticker)
        if blk:
            row = decide_blocked(ticker=ticker, side="buy", instrument="etf", source=source,
                                 why=wash_reason(ticker, blk), cfg=cfg)
        else:
            row = decide(ticker=ticker, side="buy", instrument="etf" if bucket == "etf" else "stock", bucket=bucket,
                         stats=bucket_stats(bucket, stats), cost_pct=cost, source=source, cfg=cfg,
                         extra={"strategy": strategy, "costModel": "position_sizing._estimate_costs, volume unknown"})
    except Exception as e:
        log.warning(f"[TAX] tier gate failed for {ticker}: {type(e).__name__}: {e}")
        row = decide_unpriced(ticker=ticker, side="buy", instrument="etf", source=source,
                              why="Allowed (not estimated): the tax estimate failed — see error.",
                              error=f"{type(e).__name__}: {e}", cfg=cfg)
    log_decision(row, path=path)
    return row


def _days_to_expiry(contract: dict, today: date = None) -> float:
    for k in ("days_to_expiry", "days_short"):
        v = contract.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    exp = contract.get("expiry") or contract.get("short_expiry")
    if exp:
        try:
            return float((date.fromisoformat(str(exp)[:10]) - (today or date.today())).days)
        except ValueError:
            return 0.0
    return 0.0


def option_notional(contract: dict, strategy: str) -> float:
    """The base a sold-premium open's return is measured on: its STRIKE NOTIONAL
    (strike × 100 × contracts; the cash securing a put). That is the basis the
    csp_options stats were measured on (the backtest's net_pct is P&L over the
    stock-equivalent notional), so the bucket's per-trade percent applies to it
    unchanged. A defined-risk structure ties up less money than its notional;
    that capital efficiency is NOT credited here (stated, conservative)."""
    qty = abs(float(contract.get("qty") or 1))
    if strategy == "sell_cash_secured_put" and contract.get("cash_required"):
        return float(contract["cash_required"])
    for k in ("strike", "short_put_strike", "short_strike", "short_call_strike"):
        v = contract.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v) * 100.0 * qty
    return 0.0


def option_premium(contract: dict) -> float:
    """Dollars of premium the open collects (the most a sold structure can earn)."""
    qty = abs(float(contract.get("qty") or 1))
    for k in ("premium_received", "total_credit"):
        v = contract.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    for k in ("net_credit", "limit_price"):
        v = contract.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v) * 100.0 * qty
    return 0.0


def option_round_trip_cost(contract: dict) -> float:
    """Dollars: the fee per contract on every leg, both sides, plus the entry walk
    at the widest spread the liquidity filter admits on the order's own (net)
    price, both sides — the repo's own constants. A multi-leg order is priced
    at one net limit, so the walk applies once to its net premium."""
    from position_sizing import OPTIONS_FEE_PER_CONTRACT
    from options_execution import MAX_SPREAD_PCT
    qty = abs(float(contract.get("qty") or 1))
    legs = sum(1 for k in _LEG_KEYS if contract.get(k)) or 1
    fees = float(OPTIONS_FEE_PER_CONTRACT) * qty * legs * 2
    spread = 2 * OPTION_ENTRY_WALK * (float(MAX_SPREAD_PCT) / 2.0) * option_premium(contract)
    return fees + spread


def gate_option_contract(contract: dict, ticker: str, strategy: str, metadata: dict = None,
                         source: str = "options_execution.select_contract", underlying_price: float = 0.0,
                         stats: dict = None, cfg: dict = None, path: str = None, shadow: bool = True) -> dict:
    """Layer B — an option OPEN at contract selection, before any order is built
    from it. Premium sellers are priced (csp_options stats, the premium as the
    cap, the contract's own costs and expiry); a hedge (metadata tax_role
    "hedge") and long premium (no measured stats) are allowed and logged as not
    estimated. A new skip also writes a shadow_portfolio "rejected_tax" record
    so the rule's prevention P&L is measured. Fails open."""
    meta = metadata if isinstance(metadata, dict) else {}
    side = str(contract.get("side") or ("sell" if strategy in PREMIUM_SELLING else "buy"))
    try:
        c = cfg or config()
        blk = wash_block(ticker)
        if blk:
            row = decide_blocked(ticker=ticker, side=side, instrument="option", source=source, cfg=c,
                                 why=wash_reason(ticker, blk))
        elif meta.get("tax_role") == "hedge":
            row = decide_unpriced(ticker=ticker, side=side, instrument="option", source=source, cfg=c,
                                  why=f"Allowed (hedge): {strategy} is insurance, judged by its risk role, not by expected return.")
        elif strategy not in PREMIUM_SELLING:
            row = decide_unpriced(ticker=ticker, side=side, instrument="option", source=source, cfg=c,
                                  why=(f"Allowed (not estimated): the bot has no measured win rate or payoff for {strategy} "
                                       f"(long premium), so the tax filter cannot price it and does not guess."))
        else:
            money = option_notional(contract, strategy)
            if not money > 0:
                raise ValueError(f"no strike notional on the {strategy} contract")
            premium = option_premium(contract)
            cost_pct = option_round_trip_cost(contract) / money * 100.0
            hold = float((c.get("HOLD_DAYS") or {}).get("csp_options", DEFAULTS["HOLD_DAYS"]["csp_options"]))
            dte = _days_to_expiry(contract)
            if dte > 0:
                hold = min(hold, dte)
            row = decide(ticker=ticker, side=side, instrument="option", bucket="csp_options",
                         stats=bucket_stats("csp_options", stats), cost_pct=cost_pct, holding_days=hold,
                         max_gain_pct=premium / money * 100.0, source=source, cfg=c,
                         label=f"{BUCKET_LABELS['csp_options']} ({strategy})",
                         extra={"strategy": strategy, "contract": contract.get("occ_symbol") or None,
                                "notional": round(money, 2), "premium": round(premium, 2), "dte": dte or None})
    except Exception as e:
        log.warning(f"[TAX] option gate failed for {ticker} {strategy}: {type(e).__name__}: {e}")
        row = decide_unpriced(ticker=ticker, side=side, instrument="option", source=source,
                              why="Allowed (not estimated): the tax estimate failed — see error.",
                              error=f"{type(e).__name__}: {e}", cfg=cfg)
    written = log_decision(row, path=path)
    if written and shadow and not row.get("allowed"):
        _shadow_reject(ticker, row, underlying_price, meta)
    return row


def _shadow_reject(ticker: str, row: dict, underlying_price: float, meta: dict) -> None:
    """A shadow_portfolio "rejected_tax" record for a skipped option open, once per
    logged decision (the log's dedupe makes it once per ticker per day)."""
    try:
        import shadow_portfolio
        shadow_portfolio.log_candidate(
            ticker=str(ticker).upper(), features={}, score=0.0, decision="rejected_tax",
            decision_reason=str(row.get("reason", ""))[:200], entry_price=float(underlying_price or 0.0),
            vxx_ratio=float(meta.get("vxx_ratio", 1.0) or 1.0), regime_label=str(meta.get("regime_label", "unknown")),
            side=str(row.get("side", "sell")), action_label="OPTIONS", trade_type="options")
    except Exception as e:   # measurement only: the skip stands either way
        log.warning(f"[TAX] rejected_tax shadow record failed for {ticker}: {type(e).__name__}: {e}")


def skip_error(row: dict) -> str:
    """The error text a skipped contract returns to select_contract's callers.
    The tier dispatcher's one-liner files a "rejected_capital" shadow record
    for any error containing "budget" or "capital", so neither word may appear."""
    text = str(row.get("reason", "skipped"))[:300]
    return "tax filter: " + re.sub(r"(?i)budget|capital", "money", text)


# ── holding winners past one year (exit deferral) ──────────────────────────
#
# A winning stock/ETF lot within LT_DEFER_WINDOW_DAYS of its long-term date
# keeps its discretionary profit-taking exits (scale-out, take-profit, time
# stop) on hold while the tax saving of waiting exceeds the expected give-back
# of waiting. Stops, trailing stops, breakeven, the hard floor, POS-KILL, kill
# switches and every risk exit are NEVER deferred — they keep running through
# the wait, which is what bounds the give-back. bot.ts's 14-day TIME-EXIT is
# not held either: its band (P&L at most +2 %) caps the saving at ~0.18 % of
# the position, below the give-back of any wait.
#
#   saving    = gain / value × (short − long) — percent of the position's value
#               (selling now pays the short-term rate on the gain, waiting the
#               long-term rate)
#   give-back = E[max(0, −X)] × (1 − long), X ~ N(0, σ·√T): the expected
#               adverse move over the T trading days left, driftless, after the
#               long-term tax it would no longer pay. σ = ATR% / 1.596 (for a
#               driftless random walk E[high − low] = 2·√(2/π)·σ, Feller 1951);
#               T = calendar days × 252/365.
#   defer     = saving > give-back
#
# BIASES, STATED: the upside of waiting is ignored and the stop that caps the
# downside is not credited, so the rule defers LESS than an expected-value rule
# would; the holding period is the stop state's entry_date (when the bot first
# managed the position — the same date bot.ts reads), not a broker lot date.

LT_DEFER_EXITS = ("scale_out", "take_profit", "time_stop")   # what a hold may delay; nothing else
ATR_PER_SIGMA = 1.596          # E[daily high − low] / σ for a driftless random walk (Feller 1951)
_DOWNSIDE_SEMI_MEAN = 0.3989   # E[max(0, −Z)] for Z ~ N(0, 1) = 1/√(2π)


def days_to_long_term(days_held: int) -> int:
    """Calendar days until a sale is long-term: a lot held MORE than one year,
    i.e. sold on day 366 or later after its purchase date."""
    return LONG_TERM_DAYS + 1 - int(days_held)


def lt_deferral(*, ticker: str, side: str, gain_pct: float, entry_date: str, atr_pct: float,
                today: date = None, cfg: dict = None, source: str = "manage_positions",
                path: str = None, log_row: bool = True) -> dict:
    """The deferral verdict for a position's discretionary exits.

    Returns {defer, eligible, daysToLongTerm, longTermDate, taxSavingPct,
    giveBackPct, reason}; `defer` is True only when the position is a long
    winner inside the window and the saving beats the give-back (and the
    switch is on). An eligible verdict is logged (kind "exit_deferral", once
    per ticker and verdict per day). Never raises: any failure is no deferral."""
    out = {"defer": False, "eligible": False, "daysToLongTerm": None, "longTermDate": None,
           "taxSavingPct": None, "giveBackPct": None, "reason": ""}
    try:
        c = cfg or config()
        d0 = date.fromisoformat(str(entry_date)[:10])
        t = today or date.today()
        held = (t - d0).days
        left = days_to_long_term(held)
        out["daysToLongTerm"] = left
        out["longTermDate"] = (d0 + timedelta(days=LONG_TERM_DAYS + 1)).isoformat()
        g = float(gain_pct) / 100.0
        window = int(c.get("LT_DEFER_WINDOW_DAYS", DEFAULTS["LT_DEFER_WINDOW_DAYS"]))
        if str(side).lower() not in ("long", "buy") or g <= 0 or left <= 0 or left > window:
            return out
        out["eligible"] = True
        st, lt = float(c["SHORT_TERM_RATE"]), float(c["LONG_TERM_RATE"])
        saving = g / (1.0 + g) * (st - lt) * 100.0
        sigma = max(float(atr_pct or 0.0), 0.0) / ATR_PER_SIGMA
        giveback = _DOWNSIDE_SEMI_MEAN * sigma * ((left * 252.0 / 365.0) ** 0.5) * (1.0 - lt)
        out["taxSavingPct"], out["giveBackPct"] = round(saving, 3), round(giveback, 3)
        beneficial = saving > giveback
        enabled = bool(c.get("LT_DEFER_ENABLED", True))
        out["defer"] = beneficial and enabled
        what = (f"{str(ticker).upper()} is {left} day{'s' if left != 1 else ''} from long-term ({out['longTermDate']}); "
                f"selling now would pay {st * 100:.0f}% instead of {lt * 100:.0f}% on a +{gain_pct:.1f}% gain")
        cmp = (f"saves about {saving:.2f}% of the position, "
               f"{'more' if beneficial else 'less'} than the {giveback:.2f}% expected give-back over those days "
               f"(ATR {float(atr_pct or 0):.1f}%; stops keep running)")
        if out["defer"]:
            out["reason"] = f"Deferred: {what}; waiting {cmp} — profit-taking exits wait until {out['longTermDate']}."
        elif beneficial:
            out["reason"] = f"Not deferred (TAX_FILTER.LT_DEFER_ENABLED off): {what}; waiting {cmp}."
        else:
            out["reason"] = f"Not deferred: {what}; waiting {cmp}, so the exits run as usual."
        if log_row:
            row = _base_row(ticker, "sell", "exit_deferral", "stock", source, c)
            row.update({"allowed": not out["defer"], "wouldAllow": not beneficial, "priced": True,
                        "expectedPct": None, "afterTaxPct": None, "spyAfterTaxPct": None,
                        "holdingDays": float(held), "gainPct": round(float(gain_pct), 3),
                        "daysToLongTerm": left, "longTermDate": out["longTermDate"],
                        "taxSavingPct": out["taxSavingPct"], "giveBackPct": out["giveBackPct"],
                        "atrPct": round(float(atr_pct or 0), 3), "reason": out["reason"]})
            log_decision(row, path=path)
        return out
    except Exception as e:   # no deferral on a broken input — and the reason says why
        log.warning(f"[TAX] LT deferral failed for {ticker}: {type(e).__name__}: {e}")
        out["reason"] = f"no deferral: {type(e).__name__}: {e}"
        return out


# ── tax-loss harvesting ────────────────────────────────────────────────────
#
# A LOSING stock position whose loss would offset gains already realized this
# calendar year is sold and the proceeds go into its sector's SPDR ETF — a
# correlated replacement that is not "substantially identical" — so market
# exposure stays while the loss is banked. The sold symbol, and any option on
# it, cannot be bought back for 30 days after the sale (the wash-sale window;
# the entry gates above enforce it). Acted on only when every condition holds
# (switch on, a replacement, no option on the underlying, no working order,
# no recent loss sale of the replacement, a fresh realized-gains snapshot, the
# market open, no kill switch, stock entries live, and a tax saving above the
# swap's modeled cost by HARVEST_MIN_NET_USD); otherwise reported with the
# reason. Selling the WHOLE position means no lot of it can be the
# replacement for another (shares acquired in the 30 days before are sold in
# the same sale).
#
# Realized gains come from bot.ts: the sleeve ledger's tax view of this
# calendar year (server/taxLedger.ts netting, wash-sale disallowances added
# back) written to DATA_DIR/voltrade_tax_realized.json on each refresh.
#
# BIASES, STATED: the saving is this year's tax at the assumed rates; the
# replacement's later gain is taxed in its turn (harvesting moves tax later and
# can turn a short-term loss into a long-term gain rate — neither credited);
# the $3,000 ordinary-income offset and carry-forward value are not counted;
# "not substantially identical" for a single stock vs its sector ETF is the
# standard reading, not a ruling.

REALIZED_FILE = "voltrade_tax_realized.json"            # bot.ts writes it (server/taxAware.ts)
HARVEST_REVIEW_FILE = "voltrade_tax_harvest_review.json"  # the latest review, read by GET /api/bot/tax
HARVEST_STATE_FILE = "voltrade_tax_harvested.json"       # harvested symbols and their wash-sale windows
WASH_DAYS = 30


def _sector_etfs() -> dict:
    """Sector name -> SPDR ETF (macro_data.SECTOR_ETFS inverted, the map
    bot_engine's sector names come from)."""
    from macro_data import SECTOR_ETFS
    return {name: etf for etf, name in SECTOR_ETFS.items()}


def realized_snapshot(path: str = None, now: datetime = None, cfg: dict = None) -> tuple:
    """(snapshot dict | None, why-not). None when missing, unreadable, another
    year's, or older than REALIZED_MAX_AGE_H."""
    c = cfg or config()
    p = path or os.path.join(_data_dir(), REALIZED_FILE)
    try:
        with open(p) as f:
            snap = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"this year's realized gains are unknown (no readable sleeve-ledger snapshot: {type(e).__name__})"
    t = now or datetime.now(timezone.utc)
    try:
        as_of = datetime.fromisoformat(str(snap.get("asOf", "")).replace("Z", "+00:00"))
    except ValueError:
        return None, "this year's realized gains are unknown (the snapshot has no valid asOf)"
    age_h = (t - as_of).total_seconds() / 3600.0
    if age_h > float(c.get("REALIZED_MAX_AGE_H", DEFAULTS["REALIZED_MAX_AGE_H"])) or int(snap.get("year", 0)) != t.year:
        return None, f"this year's realized gains are unknown (the sleeve-ledger snapshot is {age_h:.0f} h old)"
    return snap, ""


def harvested_symbols(path: str = None, today: date = None) -> dict:
    """{SYMBOL: {soldOn, washSaleUntil, replacement}} still inside the window."""
    p = path or os.path.join(_data_dir(), HARVEST_STATE_FILE)
    try:
        with open(p) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return {}
    d = (today or date.today()).isoformat()
    return {k: v for k, v in (state or {}).items() if isinstance(v, dict) and str(v.get("washSaleUntil", "")) >= d}


def wash_block(ticker: str, path: str = None, today: date = None) -> dict:
    """The harvest that keeps `ticker` (or options on it) from being bought
    back today, or None."""
    return harvested_symbols(path, today).get(str(ticker or "").upper())


def wash_reason(ticker: str, block: dict) -> str:
    return (f"Skipped: {str(ticker).upper()} was sold for a tax-loss harvest on {block.get('soldOn')}; buying it "
            f"(or an option on it) back before {block.get('washSaleUntil')} has passed would be a wash sale and "
            f"cancel the loss.")


def _record_harvest(ticker: str, replacement: str, today: date, path: str = None) -> None:
    p = path or os.path.join(_data_dir(), HARVEST_STATE_FILE)
    state = harvested_symbols(p, today)
    state[ticker] = {"soldOn": today.isoformat(), "washSaleUntil": (today + timedelta(days=WASH_DAYS)).isoformat(),
                     "replacement": replacement}
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, p)


def _num(v):
    """float(v), or None for a missing or malformed broker field."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _occ_root(symbol: str) -> str:
    return symbol[:-15] if len(symbol) > 15 and re.search(r"\d{6}[CP]\d{8}$", symbol) else symbol


def harvest_candidates(positions: list, realized: dict, sector_of, *, floor_tickers=(), open_orders=(),
                       today: date = None, cfg: dict = None, realized_why: str = "") -> list:
    """Pure: one row per LOSING stock position — {ticker, qty, unrealized,
    replacement, washSaleUntil, taxSaving, swapCost, action, reason}; action
    "harvest" when every rule holds, else "none" with the reason."""
    c = cfg or config()
    d = today or date.today()
    until = (d + timedelta(days=WASH_DAYS)).isoformat()
    st, lt = float(c["SHORT_TERM_RATE"]), float(c["LONG_TERM_RATE"])
    held_options = {_occ_root(str(p.get("symbol", ""))).upper() for p in positions or []
                    if str(p.get("asset_class", "")) == "us_option" or len(str(p.get("symbol", ""))) > 8}
    work = {str(s).upper() for s in open_orders or ()}
    floor = {str(s).upper() for s in floor_tickers or ()}
    etfs = _sector_etfs()
    loss_sales = {str(k).upper(): v for k, v in ((realized or {}).get("recentLossSales") or {}).items()}
    st_gain = max(0.0, float((realized or {}).get("shortAfterNetting") or 0.0))
    lt_gain = max(0.0, float((realized or {}).get("longAfterNetting") or 0.0))
    out = []
    for p in positions or []:
        sym = str(p.get("symbol", "")).upper()
        if str(p.get("asset_class", "us_equity")) == "us_option" or len(sym) > 8:
            continue
        qty, upl, price = _num(p.get("qty")), _num(p.get("unrealized_pl")), _num(p.get("current_price"))
        if qty is None or upl is None or price is None or qty <= 0 or upl >= 0 or str(p.get("side", "long")) != "long":
            continue
        loss = -upl
        row = {"ticker": sym, "qty": qty, "unrealized": round(upl, 2), "replacement": None, "washSaleUntil": until,
               "taxSaving": None, "swapCost": None, "action": "none", "reason": ""}
        sector = sector_of(sym)
        repl = etfs.get(sector)
        if repl and repl != sym and sym not in etfs.values():
            row["replacement"] = repl
        why = None
        if sym in floor:
            why = "floor-managed: the floor would buy it back within 30 days (a wash sale), so it is never harvested"
        elif realized is None:
            why = realized_why or "this year's realized gains are unknown"
        elif not row["replacement"]:
            why = f"no replacement that is correlated but not substantially identical (sector: {sector or 'unknown'})"
        else:
            off_st = min(loss, st_gain)
            off_lt = min(loss - off_st, lt_gain)
            saving = off_st * st + off_lt * lt
            from position_sizing import _estimate_costs
            cost = float(_estimate_costs(int(qty), price, 0, "stock")["total_cost_dollars"])
            row["taxSaving"], row["swapCost"] = round(saving, 2), round(cost, 2)
            if saving <= 0:
                why = "no gains realized this year for the loss to offset (it would only carry forward)"
            elif saving - cost < float(c.get("HARVEST_MIN_NET_USD", DEFAULTS["HARVEST_MIN_NET_USD"])):
                why = (f"saves about ${saving:,.0f} of tax but the swap costs about ${cost:,.0f} — under the "
                       f"${float(c.get('HARVEST_MIN_NET_USD', 20)):,.0f} net bar")
            elif sym in held_options:
                why = f"an open option on {sym} could make the sale a wash sale"
            elif row["replacement"] in loss_sales:
                why = f"{row['replacement']} was sold at a loss on {str(loss_sales[row['replacement']])[:10]}; buying it now would wash that loss"
            elif sym in work or row["replacement"] in work:
                why = "an order is already working on one of the two symbols"
            else:
                row["action"] = "harvest"
                row["reason"] = (f"Harvest: sell {qty:g} {sym} at a ${loss:,.0f} loss — it offsets this year's realized "
                                 f"gains, saving about ${saving:,.0f} of tax at {st * 100:.0f}% (swap cost ~${cost:,.0f}) — "
                                 f"and buy {row['replacement']} ({sector}) with the proceeds; {sym} can be bought back "
                                 f"after {until}.")
        if why:
            row["reason"] = f"Not harvested: {why}."
        out.append(row)
    return out


def harvest_step(deps: dict, cfg: dict = None, today: date = None, path: str = None) -> dict:
    """One review per scan: candidates (always written to the review file and
    logged), and the swaps that pass every gate placed through deps["order"]
    (a marketable DAY limit at the NBBO — bot_engine's floor order helper). The
    replacement is bought only after the sale was accepted. Never raises."""
    out = {"candidates": [], "actions": [], "error": None}
    try:
        c = cfg or config()
        d = today or date.today()
        realized, why = realized_snapshot(cfg=c)
        positions = deps["positions"]() or []
        try:
            work = deps["open_orders"]()
        except Exception as e:   # unknown working orders = treat every symbol as busy this pass
            log.warning(f"[TAX] harvest: open orders unreadable ({type(e).__name__}: {e}); nothing is swapped")
            work = {str(p.get("symbol", "")).upper() for p in positions}
        cands = harvest_candidates(positions, realized, deps["sector_of"], floor_tickers=deps.get("floor_tickers", ()),
                                   open_orders=work, today=d, cfg=c, realized_why=why)
        gate = None
        if not c.get("HARVEST_ENABLED", True):
            gate = "the switch TAX_FILTER.HARVEST_ENABLED is off"
        elif deps.get("killed"):
            gate = "the kill switch is on, or its state is unknown this scan (no new entries)"
        else:
            import entry_shadow
            if entry_shadow.is_shadowed("stocks"):
                gate = "stock entries are in shadow mode, so the replacement could not be bought"
            else:
                ok, clock_why = deps["clock"]()
                if not ok:
                    gate = clock_why
        done = 0
        for row in cands:
            if row["action"] != "harvest":
                continue
            if gate or done >= int(c.get("HARVEST_MAX_PER_SCAN", DEFAULTS["HARVEST_MAX_PER_SCAN"])):
                row["action"] = "recommend"
                row["reason"] = row["reason"].replace("Harvest:", f"Recommended (not acted on: {gate or 'the per-scan limit was reached'}):", 1)
                continue
            act = {"ticker": row["ticker"], "replacement": row["replacement"], "sold": None, "bought": None, "error": None}
            bid, _ = deps["quote"](row["ticker"])
            _, ask = deps["quote"](row["replacement"])
            if not (bid > 0 and ask > 0):
                row["action"] = "recommend"
                row["reason"] = row["reason"].replace("Harvest:", "Recommended (not acted on: no two-sided quote):", 1)
                continue
            sold = deps["order"](row["ticker"], int(row["qty"]), "sell", bid)
            act["sold"] = sold
            if not sold.get("ok"):
                act["error"] = f"sale refused: {sold.get('error')}"
                row["action"] = "failed"
                row["reason"] = f"Harvest failed: the sale of {row['ticker']} was refused ({sold.get('error')}); nothing was bought."
            else:
                _record_harvest(row["ticker"], row["replacement"], d, path)
                n = int(row["qty"] * bid // ask)
                bought = deps["order"](row["replacement"], n, "buy", ask) if n > 0 else {"ok": False, "error": "0 shares"}
                act["bought"] = bought
                if not bought.get("ok"):
                    act["error"] = f"replacement not bought: {bought.get('error')}"
                    row["reason"] += f" The replacement buy failed ({bought.get('error')}); the proceeds stay in cash."
                row["action"] = "harvested"
                done += 1
            out["actions"].append(act)
        out["candidates"] = cands
        _write_review(cands, realized, why, c)
        for row in cands:
            lr = _base_row(row["ticker"], "sell", "harvest", "stock", "harvest_step", c)
            lr.update({"allowed": row["action"] in ("harvested", "harvest"), "wouldAllow": row["action"] != "none",
                       "priced": row["taxSaving"] is not None, "expectedPct": None, "afterTaxPct": None,
                       "spyAfterTaxPct": None, "holdingDays": None, "strategy": row["action"],
                       **{k: row[k] for k in ("unrealized", "replacement", "washSaleUntil", "taxSaving", "swapCost", "reason")}})
            log_decision(lr)
        return out
    except Exception as e:   # the review must never break the scan — reported in the result
        log.warning(f"[TAX] harvest step failed: {type(e).__name__}: {e}")
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out


def _write_review(cands: list, realized, why: str, cfg: dict) -> None:
    p = os.path.join(_data_dir(), HARVEST_REVIEW_FILE)
    body = {"asOf": _now_iso(), "enabled": bool(cfg.get("HARVEST_ENABLED", True)),
            "realized": {k: (realized or {}).get(k) for k in ("asOf", "shortAfterNetting", "longAfterNetting")} if realized else None,
            "realizedWhy": why or None,
            "harvest": [{k: r[k] for k in ("ticker", "unrealized", "replacement", "washSaleUntil", "reason", "action", "taxSaving")}
                        for r in cands]}
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(body, f)
    os.replace(tmp, p)
