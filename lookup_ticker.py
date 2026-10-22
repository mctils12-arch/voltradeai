"""
lookup_ticker.py — what the bot thinks of ONE symbol right now, through the
scan's own path ([PRODUCT] 2026-10-22, human-asked: "a way to look up a
stock / option chain and see what the bot thinks of it, so I can see if
it's making the right decisions").

NOT a second opinion. The lookup walks the same steps bot_engine's Tier-2
scan walks for a candidate, in the same order, with the same functions:

  1. snapshot         Alpaca /v2/stocks/snapshots → (close, open, volume,
                      prev_close), parsed by bot_engine._parse_snapshot_batch
  2. quick pass       the scan's quick score (formula kept in step by a
                      test that reads the scan's source)
  3. deep_score       bot_engine.deep_score(ticker, quick) — the strategy
                      modules, VRP, sentiment, earnings, vol, the ML column
  4. gates            the scan's per-candidate gates in the scan's order:
                      universe floors · regime · stop cooldown · admission
                      (MIN_SCORE from get_adaptive_params) · sector /
                      correlation · sizing (position_sizing.calculate_position)
  5. chain            options_scanner._fetch_options_chain + the setup the
                      scanner itself would pick (its own function)

Every step that touches the network is guarded: a failed step is reported
under `errors[step]` and the card still renders what it has. Nothing is
inferred past what the functions return. Pure pieces (quick pass, gates,
chain summary) are tested without the network (test_lookup_ticker.py).
"""
from __future__ import annotations

import logging
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger("lookup_ticker")

# The scan's cooldown after a stop, per regime (bot_engine._scan_market_inner).
COOLDOWN_SECS = {"PANIC": 14400, "BEAR": 10800, "CAUTION": 9000, "BULL": 5400, "NEUTRAL": 7200}
NO_NEW_LONGS = ("BEAR", "PANIC")
GATE_ORDER = ("universe", "regime", "cooldown", "admission", "sector", "sizing")

# Deep-score fields the card shows (everything else stays server-side)
DEEP_FIELDS = (
    "deep_score", "quick_score", "side", "action_label", "reasons", "sector", "rsi", "vrp",
    "ewma_rv", "garch_rv", "spread_pct", "factors", "ml_score", "ml_confidence", "ml_source",
    "ml_win_prob", "sentiment", "earnings_days", "news_sentiment", "strategy",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── 2. the scan's quick pass, for one symbol ────────────────────────────────
def quick_result_from_snapshot(sym: str, snap: tuple, *, opening_half_hour: bool = False,
                               min_price: float = 5, min_volume: int = 500_000) -> dict:
    """The scan's quick score for one symbol (bot_engine._scan_market_inner, kept
    in step by test_lookup_ticker). `snap` is (close, open, volume, prev_close).
    The scan SKIPS symbols under its price/volume floors; the lookup never
    skips — the floors become `floors` flags so the card can say the scan
    would not have looked at this name."""
    c, o, v, pc = snap
    c = float(c); o = float(o); v = int(v); pc = float(pc)
    min_vol = 100_000 if opening_half_hour else min_volume
    change_pct = ((c - pc) / pc * 100) if pc > 0 else 0.0
    capped = min(abs(change_pct), 15.0)
    extreme_penalty = -30 if abs(change_pct) > 50 else (-15 if abs(change_pct) > 30 else 0)
    vol_score = min(v / 1_000_000, 5.0) * 2
    direction_bonus = 5 if abs(change_pct) > 10 else (2 if abs(change_pct) > 5 else 0)
    return {
        "ticker": sym,
        "price": round(c, 2),
        "close": c,
        "open": o,
        "prev_close": pc,
        "volume": v,
        "dollar_volume": round(c * v, 2),
        "change_pct": round(change_pct, 2),
        "quick_score": capped * 3 + vol_score + direction_bonus + extreme_penalty,
        "above_vwap": c > o,
        "range_pct": 0,
        "vwap_dist": 0,
        "reasons": [],
        "floors": {
            "price_ok": c >= min_price,
            "volume_ok": v >= min_vol,
            "symbol_ok": "." not in sym and len(sym) <= 5,
            "min_price": min_price,
            "min_volume": min_vol,
        },
    }


# ── 4. the scan's gates, in the scan's order ────────────────────────────────
def evaluate_gates(*, score: Optional[float], floors: dict, regime: str, min_score: float,
                   cooldown_left_s: float, sector_blocked: Optional[bool], max_sector: Optional[int],
                   sizing: Optional[dict]) -> dict:
    """Each gate carries pass/fail and the why; `stopped_at` is the first gate
    the scan would have stopped at (None = the scan would have placed it)."""
    gates: list[dict] = []
    floors_ok = bool(floors.get("price_ok")) and bool(floors.get("volume_ok")) and bool(floors.get("symbol_ok"))
    why = []
    if not floors.get("price_ok"): why.append(f"price under ${floors.get('min_price')}")
    if not floors.get("volume_ok"): why.append(f"volume under {int(floors.get('min_volume') or 0):,}")
    if not floors.get("symbol_ok"): why.append("symbol outside the scan's universe pattern")
    gates.append({"gate": "universe", "pass": floors_ok, "why": " · ".join(why) if why else "in the scan's universe"})
    reg_ok = regime not in NO_NEW_LONGS
    gates.append({"gate": "regime", "pass": reg_ok, "why": f"{regime}: no new stock longs" if not reg_ok else f"{regime}"})
    cd_ok = cooldown_left_s <= 0
    gates.append({"gate": "cooldown", "pass": cd_ok, "why": f"stopped out recently · {int(cooldown_left_s // 60)} min of the {regime} cooldown left" if not cd_ok else "no recent stop"})
    adm_ok = score is not None and score >= min_score
    gates.append({"gate": "admission", "pass": adm_ok, "why": (f"score {score:.1f} vs MIN_SCORE {min_score:g}" if score is not None else f"no score · MIN_SCORE {min_score:g}")})
    if sector_blocked is None:
        gates.append({"gate": "sector", "pass": None, "why": "sector check unavailable"})
    else:
        gates.append({"gate": "sector", "pass": not sector_blocked, "why": (f"sector / correlation cap reached (max {max_sector}/sector)" if sector_blocked else "room in its sector")})
    if sizing is None:
        gates.append({"gate": "sizing", "pass": None, "why": "sizing unavailable"})
    else:
        blocked = bool(sizing.get("blocked"))
        shares = sizing.get("shares")
        value = sizing.get("position_value")
        gates.append({"gate": "sizing", "pass": not blocked, "why": (sizing.get("block_reason") or sizing.get("reason") or "refused") if blocked else (f"{shares} sh · ${value:,.0f}" if isinstance(value, (int, float)) and shares else "sized")})
    stopped_at = next((g["gate"] for g in gates if g["pass"] is False), None)
    verdict = "would place it" if stopped_at is None and all(g["pass"] for g in gates) else (f"stopped at {stopped_at}" if stopped_at else "undecided (a gate could not run)")
    return {"gates": gates, "stopped_at": stopped_at, "verdict": verdict}


def summarize_deep(deep: dict) -> dict:
    out = {k: deep.get(k) for k in DEEP_FIELDS if k in deep}
    out["score"] = deep.get("deep_score", deep.get("quick_score"))
    return out


# ── 5. the chain, summarized from the scanner's own contracts ───────────────
def chain_summary_from_contracts(contracts: list, price: float) -> dict:
    """Pure: ATM IV (the nearest-the-money contracts within 21–45 days), the
    put and call the scanner's delta finder would choose at its standard
    deltas, and the counts. Contracts are the dicts _fetch_options_chain
    returns (opt_type, strike, bid, ask, iv, delta, oi, days_out, exp_date…)."""
    if not contracts:
        return {"contracts": 0, "atm_iv": None, "csp": None, "cc": None, "expiries": []}
    window = [c for c in contracts if 21 <= int(c.get("days_out") or 0) <= 45] or contracts
    near = sorted(window, key=lambda c: abs(float(c.get("moneyness") or 1.0) - 1.0))[:6]
    ivs = [float(c["iv"]) for c in near if isinstance(c.get("iv"), (int, float)) and float(c["iv"]) > 0]
    atm_iv = round(sum(ivs) / len(ivs) * 100, 1) if ivs else None

    def pick(opt_type: str, target: float) -> Optional[dict]:
        try:
            from options_scanner import _find_by_delta  # the scanner's own chooser
            c = _find_by_delta(window, opt_type, target)
        except Exception:
            c = None
        if not c:
            return None
        return {
            "occ": c.get("occ_symbol"), "strike": c.get("strike"), "exp_date": c.get("exp_date"), "days_out": c.get("days_out"),
            "bid": c.get("bid"), "ask": c.get("ask"), "delta": c.get("delta"), "delta_real": c.get("delta_real"), "iv": c.get("iv"), "oi": c.get("oi"),
            "premium_pct": round(float(c.get("bid") or 0) / price * 100, 2) if price > 0 else None,
        }

    expiries = sorted({str(c.get("exp_date")) for c in contracts})
    return {"contracts": len(contracts), "atm_iv": atm_iv, "window_days": "21–45" if window is not contracts else "all", "csp": pick("put", 0.25), "cc": pick("call", 0.25), "expiries": expiries[:8]}


# ── I/O: the orchestrator ───────────────────────────────────────────────────
def _fetch_snapshot(sym: str) -> tuple[Optional[tuple], Optional[str]]:
    import requests  # noqa: WPS433 — the scan's own client
    import bot_engine as be
    from alpaca_feed import data_feed
    try:
        from alpaca_rate_limiter import alpaca_throttle
        alpaca_throttle.acquire()
    except Exception as e:  # the throttle is a courtesy for one request; a lookup without it still answers
        log.debug("lookup: Alpaca throttle unavailable (%s)", e)
    r = requests.get(f"{be.ALPACA_DATA_URL}/v2/stocks/snapshots?symbols={sym}&feed={data_feed()}", headers=be._alpaca_headers(), timeout=15)
    out, detail = be._parse_snapshot_batch(r.json(), r.status_code)
    return out.get(sym), detail


def _opening_half_hour() -> bool:
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
        return et.hour == 9 and 30 <= et.minute < 60
    except Exception as e:  # no ET clock → the regular volume floor, as the scan would apply
        log.debug("lookup: ET clock unavailable (%s)", e)
        return False


def _cooldown_left(sym: str, regime: str) -> float:
    try:
        from storage_config import DATA_DIR
    except Exception:
        DATA_DIR = os.environ.get("VOLTRADE_DATA_DIR", "/tmp")
    try:
        with open(os.path.join(DATA_DIR, "voltrade_stop_cooldown.json")) as f:
            last = float(json.load(f).get(sym, 0) or 0)
    except Exception:
        last = 0.0
    if not last:
        return 0.0
    return max(0.0, COOLDOWN_SECS.get(regime, 7200) - (time.time() - last))


def lookup(symbol: str, with_chain: bool = True) -> dict:
    """What the bot thinks of `symbol` right now. Dict is JSON-safe."""
    sym = str(symbol or "").strip().upper()
    t0 = time.time()
    out: dict[str, Any] = {"symbol": sym, "at": _now_iso(), "path": "snapshot → quick pass → deep_score → gates (universe · regime · cooldown · admission · sector · sizing) → chain", "errors": {}, "timings": {}}
    if not sym or len(sym) > 6 or not sym.isalpha():
        out["errors"]["symbol"] = "a stock symbol is 1–6 letters (an option is looked up by its underlying)"
        return out
    # 1. snapshot
    try:
        snap, detail = _fetch_snapshot(sym)
    except Exception as e:  # network / auth
        snap, detail = None, f"{type(e).__name__}: {str(e)[:120]}"
    out["timings"]["snapshot_s"] = round(time.time() - t0, 2)
    if not snap:
        out["errors"]["snapshot"] = detail or "no snapshot for this symbol"
        return out
    # 2. quick pass
    import bot_engine as be
    quick = quick_result_from_snapshot(sym, snap, opening_half_hour=_opening_half_hour(), min_price=be.MIN_PRICE, min_volume=be.MIN_VOLUME)
    floors = quick.pop("floors")
    out["quick"] = {"price": quick["price"], "change_pct": quick["change_pct"], "volume": quick["volume"], "dollar_volume": quick["dollar_volume"], "quick_score": round(quick["quick_score"], 2), "above_open": quick["above_vwap"]}
    out["floors"] = floors
    # 3. deep score (the scan's own, with the per-source diagnostics)
    t1 = time.time()
    diag: dict = {}
    try:
        deep = be.deep_score(sym, dict(quick), _diag=diag)
    except Exception as e:
        deep = dict(quick)
        out["errors"]["deep_score"] = f"{type(e).__name__}: {str(e)[:160]}"
    out["timings"]["deep_s"] = round(time.time() - t1, 2)
    out["deep"] = summarize_deep(deep)
    out["sources"] = diag
    score = out["deep"].get("score")
    # 4. the scan's inputs for its gates
    macro: dict = {}
    try:
        from macro_data import get_macro_snapshot
        macro = get_macro_snapshot() or {}
    except Exception as e:
        out["errors"]["macro"] = f"{type(e).__name__}: {str(e)[:120]}"
    vxx = float(macro.get("vxx_ratio", 1.0) or 1.0)
    spy_ma = float(macro.get("spy_vs_ma50", 1.0) or 1.0)
    b200 = int(macro.get("spy_below_200_days", 0) or 0)
    regime = "NEUTRAL"
    try:
        from system_config import get_market_regime, get_adaptive_params
        regime = str(get_market_regime(vxx, spy_ma, spy_below_200_days=b200, spy_above_200d=bool(macro.get("spy_above_200d", True))))
        params = get_adaptive_params(vxx_ratio=vxx, spy_vs_ma50=spy_ma, spy_below_200_days=b200) or {}
    except Exception as e:
        params = {}
        out["errors"]["params"] = f"{type(e).__name__}: {str(e)[:120]}"
    min_score = float(params.get("MIN_SCORE", 63))
    max_sector = params.get("MAX_SECTOR_POSITIONS")
    out["regime"] = {"label": regime, "vxx_ratio": round(vxx, 3), "spy_vs_ma50": round(spy_ma, 4), "spy_below_200_days": b200, "min_score": min_score}
    positions: list = []
    try:
        positions = be.get_alpaca_positions() or []
        if not isinstance(positions, list):
            positions = []
    except Exception as e:
        out["errors"]["positions"] = f"{type(e).__name__}: {str(e)[:120]}"
    tickers = [p.get("symbol") for p in positions if isinstance(p, dict)]
    out["held"] = sym in tickers
    sector_blocked: Optional[bool] = None
    try:
        sector_blocked = bool(be.check_sector_correlation(sym, tickers, sector=deep.get("sector"), existing_positions=positions, max_sector_positions=max_sector))
    except Exception as e:
        out["errors"]["sector"] = f"{type(e).__name__}: {str(e)[:120]}"
    sizing: Optional[dict] = None
    try:
        from position_sizing import calculate_position
        acct = be.get_alpaca_account() or {}
        equity = float(acct.get("equity") or 0)
        trade = {"ticker": sym, "price": quick["price"], "deep_score": score, "score": score, "side": deep.get("side", "buy"), "action_label": deep.get("action_label", "BUY"),
                 "rsi": deep.get("rsi", 50), "ewma_rv": deep.get("ewma_rv"), "garch_rv": deep.get("garch_rv"), "volume": quick["volume"], "vrp": deep.get("vrp", 0), "strategy": deep.get("strategy")}
        sizing = calculate_position(trade, equity, positions, macro=macro) if equity > 0 else None
        if sizing is not None:
            sizing = {k: sizing.get(k) for k in ("blocked", "block_reason", "reason", "shares", "position_value", "kelly_fraction", "risk_pct", "scalars", "strategy") if k in sizing}
    except Exception as e:
        out["errors"]["sizing"] = f"{type(e).__name__}: {str(e)[:120]}"
    out["sizing"] = sizing
    out["decision"] = evaluate_gates(score=score, floors=floors, regime=regime, min_score=min_score, cooldown_left_s=_cooldown_left(sym, regime), sector_blocked=sector_blocked, max_sector=max_sector, sizing=sizing)
    # 5. the chain, through the scanner's own fetch and its own setup pick
    if with_chain:
        t2 = time.time()
        try:
            from options_scanner import _fetch_options_chain, _fetch_iv_rank, _setup_csp_normal_market
            contracts = _fetch_options_chain(sym, quick["price"], 7, 60) or []
            chain = chain_summary_from_contracts(contracts, quick["price"])
            try:
                chain["iv_rank"] = _fetch_iv_rank(sym)
            except Exception as e:
                out["errors"]["iv_rank"] = f"{type(e).__name__}: {str(e)[:120]}"
            try:
                setup = _setup_csp_normal_market(sym, quick["price"], vxx)
                chain["scanner_setup"] = {k: setup.get(k) for k in ("setup", "action_label", "options_strategy", "strike", "premium", "otm_pct", "premium_yield", "annual_yield", "iv_rank", "avg_iv", "best_put_occ", "reasoning", "score")} if setup else None
            except Exception as e:
                out["errors"]["scanner_setup"] = f"{type(e).__name__}: {str(e)[:120]}"
            out["chain"] = chain
        except Exception as e:
            out["errors"]["chain"] = f"{type(e).__name__}: {str(e)[:160]}"
        out["timings"]["chain_s"] = round(time.time() - t2, 2)
    out["timings"]["total_s"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":  # python3 lookup_ticker.py NVDA
    import sys
    print(json.dumps(lookup(sys.argv[1] if len(sys.argv) > 1 else "SPY"), indent=1, default=str))
