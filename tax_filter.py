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
from datetime import date, datetime, timezone

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
        if meta.get("tax_role") == "hedge":
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
