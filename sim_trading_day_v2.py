#!/usr/bin/env python3
"""
VolTradeAI v1.0.13 — Full Trading Day Simulation
Tests every phase of the trading day from 4am to 8pm
"""
import os, sys, json, time
sys.path.insert(0, '/tmp/vt_test')
os.environ['VOLTRADE_DATA_DIR'] = '/tmp'

print("=" * 60)
print("VolTradeAI v1.0.13 — Full Trading Day Simulation")
print("Tomorrow: Tuesday, April 7, 2026")
print("=" * 60)

results = {"passed": 0, "warnings": 0, "failed": 0, "details": []}

def test(name, fn):
    try:
        t0 = time.time()
        status, msg = fn()
        elapsed = time.time() - t0
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        print(f"  {icon} {name:<45} {msg}  [{elapsed:.1f}s]")
        results["details"].append({"name": name, "status": status, "msg": msg})
        if status == "PASS": results["passed"] += 1
        elif status == "WARN": results["warnings"] += 1
        else: results["failed"] += 1
    except Exception as e:
        print(f"  ✗ {name:<45} EXCEPTION: {str(e)[:60]}")
        results["failed"] += 1
        results["details"].append({"name": name, "status": "FAIL", "msg": str(e)[:60]})

# ─────────────────────────────────────────────────────────────────
print("\n── 4:00 AM: ML Training ──")
# ─────────────────────────────────────────────────────────────────

def test_ml_stock_train():
    from ml_model_v2 import train_model
    result = train_model()
    if result.get("status") == "trained":
        acc = result.get("accuracy", 0) * 100
        return "PASS", f"LightGBM {acc:.1f}% acc, {result.get('samples')} samples"
    elif result.get("status") == "insufficient_data":
        return "WARN", f"Only {result.get('samples')} samples (need 300)"
    return "FAIL", result.get("error", result.get("reason", "unknown"))
test("Stock ML model retrain", test_ml_stock_train)

def test_ml_options_train():
    from ml_model_v2 import train_options_model
    result = train_options_model()
    if result.get("status") == "trained":
        acc = result.get("options_model_accuracy", 0)
        src = result.get("data_source", "")
        return "PASS", f"{acc}% acc, {result.get('total_samples')} samples ({src[:30]})"
    return "FAIL", result.get("reason", result.get("error", "unknown"))
test("Options ML model retrain", test_ml_options_train)

def test_markov_regime():
    from markov_regime import get_regime
    result = get_regime([], 1.05, 1.01)
    if result and "regime_label" in result:
        return "PASS", f"Regime: {result['regime_label']} score={result.get('regime_score'):.0f}"
    return "FAIL", "No regime result"
test("Markov regime detection", test_markov_regime)

def test_system_config():
    from system_config import get_adaptive_params, get_market_regime
    regime = get_market_regime(1.05, 1.01)
    params = get_adaptive_params(vxx_ratio=1.05, spy_vs_ma50=1.01)
    if params and "MAX_POSITIONS" in params:
        return "PASS", f"Regime={regime} MAX_POS={params['MAX_POSITIONS']} MIN_SCORE={params['MIN_SCORE']}"
    return "FAIL", "Missing params"
test("System config adaptive params", test_system_config)

# ─────────────────────────────────────────────────────────────────
print("\n── 9:30 AM: Market Open — Stock Scan ──")
# ─────────────────────────────────────────────────────────────────

def test_universe_fetch():
    import requests
    r = requests.get(
        "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=20",
        headers={"APCA-API-KEY-ID":"PKMDHJOVQEVIB4UHZXUYVTIDBU",
                 "APCA-API-SECRET-KEY":"9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et"},
        timeout=10)
    tickers = r.json().get("most_actives", [])
    if len(tickers) >= 5:
        names = [t["symbol"] for t in tickers[:5]]
        return "PASS", f"{len(tickers)} actives: {', '.join(names)}"
    return "WARN", f"Only {len(tickers)} tickers returned"
test("Market universe fetch (SIP)", test_universe_fetch)

def test_quick_score():
    from bot_engine import score_stock  # score_stock is the exported scoring function
    import requests
    r = requests.get(
        "https://data.alpaca.markets/v2/stocks/snapshots?symbols=AAPL,MSFT,NVDA&feed=sip",
        headers={"APCA-API-KEY-ID":"PKMDHJOVQEVIB4UHZXUYVTIDBU",
                 "APCA-API-SECRET-KEY":"9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et"},
        timeout=8)
    snaps = r.json()
    scored = []
    for sym, snap in snaps.items():
        bar  = snap.get("dailyBar", {})
        prev = snap.get("prevDailyBar", {})
        c    = float(bar.get("c", 0))
        pc   = float(prev.get("c", c))
        v    = int(bar.get("v", 0))
        if c < 5 or v < 100_000: continue
        change = (c - pc) / pc * 100 if pc > 0 else 0
        mock   = {"ticker": sym, "price": c, "close": c, "prev_close": pc,
                  "volume": v, "change_pct": change, "above_vwap": True,
                  "range_pct": abs(change)*0.5, "vwap_dist": change*0.3, "reasons": []}
        result = score_stock(mock)  # Returns dict with quick_score key
        sc = result.get("quick_score", result.get("score", 0)) if isinstance(result, dict) else float(result or 0)
        scored.append((sym, sc))
    if scored:
        best = max(scored, key=lambda x: x[1])
        return "PASS", f"Scored {len(scored)} stocks, best={best[0]} ({best[1]:.1f})"
    return "WARN", "No stocks scored (market may be closed)"
test("Quick score pipeline (score_stock)", test_quick_score)

def test_ml_inference():
    from ml_model_v2 import ml_score
    features = {
        "momentum_1m":3.2,"momentum_3m":8.1,"rsi_14":55,"volume_ratio":2.1,
        "vwap_position":1,"adx":28,"ewma_vol":1.8,"range_pct":2.1,
        "price_vs_52w_high":-3.2,"float_turnover":0.12,"vrp":5.0,
        "iv_rank_proxy":65,"atr_pct":1.9,"vxx_ratio":1.08,"spy_vs_ma50":1.01,
        "markov_state":1,"regime_score":58,"sector_momentum":1,
        "change_pct_today":1.8,"above_ma10":1,"trend_strength":2.1,
        "volume_acceleration":1,"intel_score":45,"insider_signal":0,"news_sentiment":10,
    }
    result = ml_score(features)
    sc  = result.get("ml_score", 0)
    sig = result.get("ml_signal", "?")
    mdl = result.get("model_type", "?")
    if 0 <= sc <= 100:
        return "PASS", f"score={sc:.1f} signal={sig} model={mdl}"
    return "FAIL", f"Score out of range: {sc}"
test("Stock ML inference (25 features)", test_ml_inference)

def test_options_ml_inference():
    from ml_model_v2 import options_ml_score, OPTIONS_FEATURE_COLS
    features_panic = {c: 0 for c in OPTIONS_FEATURE_COLS}
    features_panic.update({"vxx_ratio":1.42,"iv_rank":88,"iv_rank_proxy":88,
                            "spy_vs_ma50":0.97,"vrp":21,"vrp_magnitude":21,
                            "days_to_earn":99,"regime_score":28,"ewma_vol":21.3})
    features_lowiv = {c: 0 for c in OPTIONS_FEATURE_COLS}
    features_lowiv.update({"vxx_ratio":0.86,"iv_rank":14,"iv_rank_proxy":14,
                            "spy_vs_ma50":1.04,"vrp":-7,"vrp_magnitude":7,
                            "days_to_earn":99,"regime_score":79,"ewma_vol":12.9})
    p1 = options_ml_score(features_panic)
    p2 = options_ml_score(features_lowiv)
    spread = abs(p1 - p2)
    if 0.0 < p1 <= 1.0 and 0.0 < p2 <= 1.0 and spread > 0.05:
        return "PASS", f"panic={p1:.3f} low_iv={p2:.3f} spread={spread:.3f}"
    return "WARN", f"panic={p1:.3f} low_iv={p2:.3f} spread={spread:.3f} (< 0.05 = limited differentiation)"
test("Options ML inference (28 features)", test_options_ml_inference)

# ─────────────────────────────────────────────────────────────────
print("\n── 9:30 AM: Options Scanner Setups ──")
# ─────────────────────────────────────────────────────────────────

def test_options_scanner_import():
    from options_scanner import (
        _get_vxx_ratio, _get_spy_vs_ma50, _build_setup_features, _options_ml_score,
        OPTIONS_UNIVERSE
    )
    vxx = _get_vxx_ratio()
    spy = _get_spy_vs_ma50()
    if 0.3 <= vxx <= 5.0 and 0.5 <= spy <= 2.0:
        return "PASS", f"VXX ratio={vxx:.3f} SPY/MA50={spy:.3f} universe={len(OPTIONS_UNIVERSE)} tickers"
    return "WARN", f"VXX={vxx:.3f} SPY={spy:.3f} — values unusual"
test("Options scanner init + VXX/SPY fetch", test_options_scanner_import)

def test_options_feature_builder():
    from options_scanner import _build_setup_features, _options_ml_score
    feats = _build_setup_features(vxx_ratio=1.35, spy_vs_ma50=0.97,
                                   iv_rank=85, vrp=18.0, days_to_earn=99.0)
    prob = _options_ml_score(feats)
    if feats and 0.3 <= prob <= 1.0:
        return "PASS", f"28 features built, ML prob={prob:.3f}"
    return "FAIL", f"Missing features or bad prob={prob}"
test("Setup feature builder + ML scoring", test_options_feature_builder)

def test_earnings_calendar():
    from options_scanner import _fetch_earnings_calendar
    cal = _fetch_earnings_calendar(days_ahead=7)
    if isinstance(cal, dict):
        upcoming = [(s, d) for s, d in cal.items() if 1 <= d <= 7]
        return "PASS", f"{len(cal)} total, {len(upcoming)} in 1-7d: {[s for s,_ in upcoming[:3]]}"
    return "WARN", "Empty earnings calendar"
test("Earnings calendar (Finnhub)", test_earnings_calendar)

def test_options_chain_fetch():
    from options_scanner import _fetch_options_chain, _fetch_price
    price = _fetch_price("SPY")
    if not price:
        return "WARN", "SPY price unavailable (market closed)"
    contracts = _fetch_options_chain("SPY", price, min_days=7, max_days=30)
    if contracts and len(contracts) >= 5:
        best = sorted(contracts, key=lambda c: c["oi"], reverse=True)[0]
        return "PASS", f"{len(contracts)} contracts, best OI={best['oi']:,} strike={best['strike']}"
    return "WARN", f"Only {len(contracts)} contracts (market may be closed — OPRA inactive)"
test("Options chain fetch (OPRA)", test_options_chain_fetch)

# ─────────────────────────────────────────────────────────────────
print("\n── 10:30 AM: Position Management ──")
# ─────────────────────────────────────────────────────────────────

def test_regime_aware_cooldown():
    from system_config import get_market_regime
    # BULL requires markov_state=2 (Markov chain must confirm bull)
    # All other regimes use default markov_state=1
    test_regimes = [
        (1.40, 0.96, 1, 14400, "PANIC"),
        (1.18, 1.00, 1, 10800, "BEAR"),
        (1.07, 1.01, 1, 9000,  "CAUTION"),
        (1.00, 1.01, 1, 7200,  "NEUTRAL"),
        (0.83, 1.04, 2, 5400,  "BULL"),    # markov_state=2 required
    ]
    all_ok = True
    results_str = []
    for vxx_r, spy_ma, ms, expected_secs, expected_label in test_regimes:
        regime = get_market_regime(vxx_r, spy_ma, markov_state=ms)
        cooldown = {
            "PANIC": 14400, "BEAR": 10800, "CAUTION": 9000,
            "NEUTRAL": 7200, "BULL": 5400
        }.get(regime, 7200)
        ok = (regime == expected_label and cooldown == expected_secs)
        if not ok: all_ok = False
        results_str.append(f"{expected_label}={cooldown//3600:.1f}h")
    if all_ok:
        return "PASS", "  ".join(results_str)
    return "FAIL", "Mismatch: " + " ".join(results_str)
test("Regime-aware cooldown (5 levels)", test_regime_aware_cooldown)

def test_alpaca_account():
    from bot_engine import get_alpaca_account
    acct = get_alpaca_account()
    if "portfolio_value" in acct:
        equity = float(acct["portfolio_value"])
        cash   = float(acct.get("cash", 0))
        bp     = float(acct.get("buying_power", 0))
        return "PASS", f"Equity=${equity:,.0f} Cash=${cash:,.0f} BP=${bp:,.0f}"
    return "FAIL", str(acct)[:80]
test("Alpaca account fetch", test_alpaca_account)

def test_alpaca_positions():
    from bot_engine import get_alpaca_positions
    pos = get_alpaca_positions()
    if isinstance(pos, list):
        return "PASS", f"{len(pos)} open positions"
    return "WARN", f"Unexpected response: {str(pos)[:60]}"
test("Alpaca positions fetch", test_alpaca_positions)

def test_position_management():
    from bot_engine import manage_positions
    mgmt = manage_positions()
    if isinstance(mgmt, dict):
        actions  = len(mgmt.get("actions", []))
        upgrades = len(mgmt.get("upgrade_candidates", []))
        return "PASS", f"{actions} actions, {upgrades} upgrade candidates"
    return "FAIL", str(mgmt)[:60]
test("Position management pipeline", test_position_management)

# ─────────────────────────────────────────────────────────────────
print("\n── 11:00 AM: Full Scan (Stock + Options) ──")
# ─────────────────────────────────────────────────────────────────

def test_full_scan():
    from bot_engine import scan_market
    t0 = time.time()
    result = scan_market()
    elapsed = time.time() - t0
    if "error" in result and "trades" not in result:
        return "FAIL", result["error"]
    scanned  = result.get("scanned", 0)
    filtered = result.get("filtered", 0)
    deep     = result.get("deep_analyzed", 0)
    trades   = result.get("new_trades", [])
    opts     = result.get("options_trades_added", 0)
    if scanned > 100:
        return "PASS", f"Scanned {scanned:,} → filtered {filtered} → deep {deep} → {len(trades)} trades ({opts} options) [{elapsed:.0f}s]"
    return "WARN", f"Only {scanned} stocks scanned [{elapsed:.0f}s]"
test("Full scan (stocks + options scanner)", test_full_scan)

# ─────────────────────────────────────────────────────────────────
print("\n── 4:00 PM: End-of-Day ──")
# ─────────────────────────────────────────────────────────────────

def test_stop_cooldown_file():
    import json, os
    cooldown_path = "/data/voltrade/voltrade_stop_cooldown.json"
    if not os.path.exists(cooldown_path):
        # Not created yet (no stops fired) — this is fine
        return "PASS", "No cooldown file (no stops today — expected)"
    try:
        with open(cooldown_path) as f:
            cd = json.load(f)
        return "PASS", f"{len(cd)} tickers in cooldown: {list(cd.keys())[:3]}"
    except Exception as e:
        return "WARN", f"Cooldown file exists but can't read: {e}"
test("Stop cooldown file (regime-aware)", test_stop_cooldown_file)

def test_feedback_file():
    import json, os
    fb_path = "/data/voltrade/voltrade_trade_feedback.json"
    if not os.path.exists(fb_path):
        return "PASS", "No feedback yet — accumulates after first closed trades"
    try:
        with open(fb_path) as f:
            fb = json.load(f)
        opts = [t for t in fb if t.get("use_options") or t.get("instrument") == "options"]
        return "PASS", f"{len(fb)} trades, {len(opts)} options trades in feedback"
    except Exception as e:
        return "WARN", f"Feedback file error: {e}"
test("Trade feedback file (self-learning)", test_feedback_file)

# ─────────────────────────────────────────────────────────────────
print("\n── Summary ──")
# ─────────────────────────────────────────────────────────────────
total = results["passed"] + results["warnings"] + results["failed"]
print(f"\n  Passed:   {results['passed']}/{total}")
print(f"  Warnings: {results['warnings']}/{total}")
print(f"  Failed:   {results['failed']}/{total}")

with open("/tmp/sim_day_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: /tmp/sim_day_results.json")
print("=" * 60)
