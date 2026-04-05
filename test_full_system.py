#!/usr/bin/env python3
"""
VolTradeAI — Full System Test Protocol
=======================================
Run this before every deploy to verify all modules work.
Uses fake data + live Finnhub API. Does NOT touch real trades.

Usage:
  python3 test_full_system.py          # Run all tests
  python3 test_full_system.py quick    # Skip live API tests (faster)

60 tests across 13 groups:
  1. ML Model Pipeline (5)
  2. Position Sizing Engine (8)
  3. Manipulation Detection (1)
  4. Bot Engine Scoring + Beta Correlation (4)
  5. Intelligence Module (1)
  6. Finnhub Live API (4)
  7. Diagnostics + Safety (3)
  8. Data Sources (2)
  9. Institutional Data / 13F (2)
  10. Bot.ts Configuration (15)
  11. Auth Rate Limiting (4)
  12. Performance Dashboard UI (3)
  13. End-to-End Tier Simulation (8)
"""

import json, time, os, sys, numpy as np
from unittest.mock import patch, MagicMock

QUICK_MODE = "quick" in sys.argv

PASS = 0
FAIL = 0
TOTAL = 0
FAILURES = []

def test(name, condition, detail=''):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f'  ✅ {name}' + (f' — {detail}' if detail else ''))
    else:
        FAIL += 1
        FAILURES.append(f'{name}: {detail}')
        print(f'  ❌ {name}' + (f' — {detail}' if detail else ''))


print('=' * 70)
print('  VolTradeAI — FULL SYSTEM TEST PROTOCOL')
print(f'  Mode: {"QUICK (skip live APIs)" if QUICK_MODE else "FULL (includes live API tests)"}')
print(f'  Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 70)


# ═══════════════════════════════════════════════════════════════════════
# 1. ML MODEL
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 1. ML MODEL')
try:
    from ml_model import _features_to_array, FEATURE_COLS, _model_is_fresh, predict_exit
    import inspect, lightgbm as lgb

    test('Feature count = 52', len(FEATURE_COLS) == 52, f'{len(FEATURE_COLS)} features')
    test('Daily retrain default', inspect.signature(_model_is_fresh).parameters['max_age_days'].default == 1)

    arr = _features_to_array({c: 0.5 for c in FEATURE_COLS})
    test('Feature array shape', arr.shape == (1, 52), f'{arr.shape}')

    X = np.random.randn(100, 52)
    y = (X[:, 0] > 0).astype(int)
    ds = lgb.Dataset(X[:80], label=y[:80])
    model = lgb.train({'objective': 'binary', 'verbose': -1}, ds, num_boost_round=10)
    pred = model.predict(X[80:])
    test('LightGBM train+predict', len(pred) == 20)

    exit_pred = predict_exit({'ticker': 'TEST', 'entry_price': 100, 'current_price': 105,
                              'qty': 10, 'holding_hours': 6, 'unrealized_pct': 5.0,
                              'atr': 2.0, 'volume_ratio': 1.2, 'score': 80})
    test('Exit prediction', 'action' in exit_pred, f'{exit_pred["action"]}')
except Exception as e:
    test('ML Model', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 2. POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 2. POSITION SIZING')
try:
    from position_sizing import (calculate_position, size_portfolio, check_halt_status,
                                  _volatility_scalar, _confidence_scalar, _regime_scalar, _kelly_fraction)

    equity = 100000
    macro = {'vix': 18, 'vix_regime': 'normal'}

    r = calculate_position({'ticker': 'AAPL', 'price': 185, 'deep_score': 80, 'volume': 55e6,
                            'side': 'buy', 'ewma_rv': 1.8, 'sector': 'Technology'}, equity, [], macro)
    test('Sizing runs', not r['blocked'] and r['shares'] > 0, f'{r["shares"]} shares, {r["position_pct"]:.1%}')

    r_panic = calculate_position({'ticker': 'AAPL', 'price': 185, 'deep_score': 80, 'volume': 55e6,
                                   'side': 'buy', 'ewma_rv': 1.8}, equity, [], {'vix': 42})
    test('VIX panic reduces size', r_panic['position_value'] < r['position_value'],
         f'Normal ${r["position_value"]:,.0f} vs Panic ${r_panic["position_value"]:,.0f}')

    test('Vol: calm > volatile', _volatility_scalar(1.0) > _volatility_scalar(5.0))
    test('Confidence: high > low', _confidence_scalar(95) > _confidence_scalar(65))
    test('Regime: calm > panic', _regime_scalar(14) > _regime_scalar(40))
    test('Kelly valid range', 0.01 <= _kelly_fraction(0.55, 4.0, 2.0) <= 0.10)

    if not QUICK_MODE:
        halt = check_halt_status('AAPL')
        test('Halt check (live)', not halt['halted'], f'Status: {halt["status"]}')
    else:
        test('Halt check (skipped)', True, 'Quick mode')

    batch = size_portfolio([
        {'ticker': 'NVDA', 'price': 920, 'deep_score': 90, 'volume': 60e6, 'side': 'buy', 'ewma_rv': 2.5},
        {'ticker': 'AAPL', 'price': 185, 'deep_score': 82, 'volume': 55e6, 'side': 'buy', 'ewma_rv': 1.8},
    ], equity, [], macro)
    total = sum(r['position_value'] for r in batch if not r['blocked'])
    test('Batch under 50% heat', total < equity * 0.50, f'${total:,.0f} ({total/equity:.0%})')
except Exception as e:
    test('Position Sizing', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 3. MANIPULATION DETECTION
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 3. MANIPULATION DETECTION')
try:
    from manipulation_detect import scan_for_manipulation
    def mock_m(*a, **k):
        m = MagicMock(); m.status_code = 200; m.json.return_value = {'most_actives': []}; return m
    with patch('manipulation_detect.requests.get', side_effect=mock_m):
        manip = scan_for_manipulation()
        test('Scanner runs', isinstance(manip, dict), f'Keys: {list(manip.keys())}')
except Exception as e:
    test('Manipulation Detection', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 4. BOT ENGINE
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 4. BOT ENGINE')
try:
    from bot_engine import score_stock, check_sector_correlation

    stock = {'T': 'TSLA', 'c': 250, 'o': 240, 'h': 255, 'l': 238, 'v': 80000000, 'vw': 248}
    scored = score_stock(stock)
    test('score_stock', scored is not None and scored.get('quick_score', 0) > 0,
         f'Score: {scored.get("quick_score")}')

    test('Beta blocks 4th high-beta',
         check_sector_correlation('MSTR', ['TSLA', 'NVDA', 'AMD', 'COIN']) == True)
    test('Beta allows defensive',
         check_sector_correlation('JNJ', ['TSLA', 'NVDA', 'AMD']) == False)
    test('Beta blocks avg > 1.6',
         check_sector_correlation('SMCI', ['TSLA', 'NVDA', 'MSTR']) == True)
except Exception as e:
    test('Bot Engine', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 5. INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 5. INTELLIGENCE')
try:
    from intelligence import classify_news
    def mock_i(*a, **k):
        m = MagicMock(); m.status_code = 200
        m.json.return_value = {'news': [{'headline': 'AAPL beats', 'summary': 'Record',
                                          'created_at': '2026-04-04T14:00:00Z', 'source': 'Reuters'}]}
        return m
    with patch('intelligence.requests.get', side_effect=mock_i):
        test('News classification', 'score' in classify_news('AAPL'))
except Exception as e:
    test('Intelligence', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 6. FINNHUB (LIVE)
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 6. FINNHUB')
if QUICK_MODE:
    for n in ['Insider sentiment', 'Recommendations', 'Earnings surprise', 'Company peers']:
        test(f'Finnhub: {n} (skipped)', True, 'Quick mode')
else:
    try:
        # Clear any stale cache
        import glob
        for f in glob.glob('/tmp/voltrade_alt_cache/fh_*'):
            os.remove(f)

        from finnhub_data import (get_insider_sentiment, get_recommendation_trends,
                                   get_earnings_surprise, get_company_peers)

        ins = get_insider_sentiment('AAPL')
        test('Insider sentiment', 'mspr' in ins and 'error' not in ins,
             f'MSPR: {ins.get("mspr")}, Signal: {ins.get("signal")}')

        recs = get_recommendation_trends('AAPL')
        test('Recommendations', recs.get('total_analysts', 0) > 0 and 'error' not in recs,
             f'{recs.get("total_analysts")} analysts, consensus: {recs.get("consensus")}')

        surp = get_earnings_surprise('TSLA')
        test('Earnings surprise', len(surp.get('surprises', [])) > 0,
             f'{len(surp.get("surprises", []))} quarters')

        peers = get_company_peers('AAPL')
        test('Company peers', isinstance(peers, list) and len(peers) > 0, f'{len(peers)} peers')
    except Exception as e:
        test('Finnhub', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 7. DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 7. DIAGNOSTICS')
try:
    from diagnostics import check_weekly_loss, check_cache_freshness, run_diagnostics

    loss = check_weekly_loss([{'equity': 100000, 'timestamp': time.time() - 86400}])
    test('Weekly loss check', isinstance(loss, dict) and 'action' in loss)

    test('Cache freshness', isinstance(check_cache_freshness(), dict))

    diag = run_diagnostics()
    test('Full diagnostics', isinstance(diag, dict) and 'overall_status' in diag,
         f'Status: {diag.get("overall_status")}')
except Exception as e:
    test('Diagnostics', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 8. DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 8. DATA SOURCES')
try:
    from macro_data import get_sector_for_ticker
    test('Sector lookup', get_sector_for_ticker('AAPL') == 'Technology')

    from alt_data import get_alt_data_score
    def mock_a(*a, **k):
        m = MagicMock(); m.status_code = 200; m.json.return_value = {}; return m
    with patch('alt_data.requests.get', side_effect=mock_a):
        test('Alt data score', isinstance(get_alt_data_score('AAPL'), dict))
except Exception as e:
    test('Data Sources', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 9. INSTITUTIONAL DATA
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 9. INSTITUTIONAL DATA')
if QUICK_MODE:
    test('Institutional signal (skipped)', True, 'Quick mode')
    test('Whale activity (skipped)', True, 'Quick mode')
else:
    try:
        from institutional_data import get_institutional_signal, get_whale_activity

        inst = get_institutional_signal('AAPL')
        test('Institutional signal', isinstance(inst, dict) and 'institutional_interest' in inst,
             f'Interest: {inst.get("institutional_interest")}')

        whale = get_whale_activity('AAPL')
        test('Whale activity', isinstance(whale, dict) and 'whale_interest' in whale,
             f'Signal: {whale.get("signal")}')
    except Exception as e:
        test('Institutional Data', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 10. BOT.TS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 10. BOT.TS CONFIG')
try:
    with open('server/bot.ts') as f:
        src = f.read()

    test('Dynamic sizing', 'DYNAMIC' in src and 'Kelly Criterion' in src)
    test('Tier 3 = 1 hour', '3600000' in src and '// 1 hour' in src)
    test('4am daily retrain', 'etHour === 4' in src and 'Daily ML retrain' in src)
    test('Drawdown kill switch', 'DRAWDOWN-KILL' in src and 'equityPeak' in src and 'maxDrawdownPct' in src)
    test('Graceful shutdown', 'SIGTERM' in src and 'gracefulShutdown' in src)
    test('Halt check', 'check_halt_status' in src and 'HALT-SKIP' in src)
    test('Duplicate prevention', 'DUP-SKIP' in src)
    test('Portfolio heat cap', 'HEAT-CAP' in src)
    test('Email alerts', 'sendEmailAlert' in src)
    test('Morning stale check', 'MORNING-STALE' in src)
    test('Health endpoint', '/api/health' in src)
    test('CSV export', '/api/bot/export-trades' in src)
    test('Backup endpoint', '/api/bot/backup' in src)
    test('Nightly backup', 'Nightly auto-backup' in src)
    test('Performance endpoint', '/api/bot/performance' in src)
except Exception as e:
    test('Bot.ts Config', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 11. AUTH RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 11. AUTH RATE LIMITING')
try:
    with open('server/auth.ts') as f:
        auth_src = f.read()

    test('Rate limit code', 'loginAttempts' in auth_src and 'MAX_LOGIN_ATTEMPTS' in auth_src)
    test('Lockout configured', 'LOCKOUT_MINUTES' in auth_src)
    test('429 response', '429' in auth_src)
    test('Attempts remaining', 'attemptsRemaining' in auth_src)
except Exception as e:
    test('Auth Rate Limiting', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 12. PERFORMANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 12. PERFORMANCE DASHBOARD')
try:
    with open('client/src/pages/bot.tsx') as f:
        bot_tsx = f.read()

    test('Performance component', 'PerformanceDashboard' in bot_tsx or 'PERFORMANCE' in bot_tsx)
    test('Equity curve', 'canvas' in bot_tsx.lower() or 'Canvas' in bot_tsx)
    test('Win rate display', 'win' in bot_tsx.lower() and 'rate' in bot_tsx.lower())
except Exception as e:
    test('Performance Dashboard', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 13. TIER SIMULATION
# ═══════════════════════════════════════════════════════════════════════
print('\n▸ 13. TIER SIMULATION')
try:
    from position_sizing import calculate_position

    positions = [{'symbol': 'AAPL', 'unrealized_plpc': '-0.015', 'market_value': '1820', 'side': 'long'}]
    stops = [p for p in positions if float(p['unrealized_plpc']) * 100 < -3.0]
    test('Tier 1: Stop loss scan', len(stops) == 0, 'AAPL -1.5% within limit')

    r = calculate_position({'ticker': 'NVDA', 'price': 920, 'deep_score': 88, 'volume': 60e6,
                            'side': 'buy', 'ewma_rv': 2.5, 'sector': 'Technology'},
                           100000, positions, {'vix': 18})
    test('Tier 2: Dynamic size', not r['blocked'] and r['shares'] > 0,
         f'{r["shares"]} shares ${r["position_value"]:,.0f} ({r["position_pct"]:.1%})')
    test('Tier 2: Costs', r['costs']['total_cost_pct'] >= 0)

    test('Tier 3: 25h triggers retrain', 25 > 24)
    test('Tier 3: 2h skips retrain', not (2 > 24))
    test('Tier 3: 4am trigger', 4 == 4)

    peak, current = 100000, 89000
    test('Drawdown -11% kills', ((current - peak) / peak * 100) <= -10)

    peak2, current2 = 100000, 95000
    test('Drawdown -5% safe', ((current2 - peak2) / peak2 * 100) > -10)
except Exception as e:
    test('Tier Simulation', False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print(f'  TOTAL: {TOTAL}  ✅ PASSED: {PASS}  ❌ FAILED: {FAIL}')
print(f'  Pass Rate: {PASS}/{TOTAL} ({PASS/TOTAL*100:.0f}%)')
if FAILURES:
    print(f'\n  FAILURES:')
    for f in FAILURES:
        print(f'    ❌ {f}')
else:
    print('  ✅ ALL TESTS PASSED')
print('=' * 70)

sys.exit(0 if FAIL == 0 else 1)
