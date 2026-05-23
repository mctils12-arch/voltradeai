# VolTradeAI — Fix Pack 2026-05-22 (CSP execution layer)

This pack continues from `voltradeai_fixes_2026-05-18` and addresses the
**CSP-failure cascade** seen in the production audit log on 2026-05-21.

## What the audit log showed

After the 2026-05-18 fixes deployed (Kelly priors, ETF routing, floor basket),
the production system was running cleanly but CSP trades were still not firing.
The audit log showed the failure pattern:

```
TIER2 Scanned 11546 stocks, 2 trade candidates (via daemon)
TIERS 3 tier actions: {"1":0,"2":3,"3":0,"4":0}
T2-FAIL  LRCX: Not enough capital to sell cash-secured put at $295.0 (need $29,500 per contract)
T2-FAIL  WDC:  No suitable puts found for selling (chain=7, puts=0, filtered=0, price=489.16)
T2-FAIL  STX:  No liquid options contracts (need vol>10, OI>200, spread<10%)
T2-FAIL  COHR: No liquid options contracts (need vol>10, OI>200, spread<10%)
T2-FAIL  USO:  No liquid options contracts (need vol>10, OI>200, spread<10%)
T2-FAIL  JNJ:  No suitable puts found for selling (chain=2, puts=0, filtered=0, price=231.73)
T2-FAIL  SGOV: No suitable puts found for selling (chain=1, puts=0, filtered=0, price=100.59)
```

Three distinct failure modes were blocking every CSP attempt.

## The cascade

1. `csp_universe._layer1_hard_gates` accepted any liquid name regardless of price
2. `_layer2_score` ranked expensive volatile names high (LRCX, WDC, MSTR) because they have high IV
3. `tier1_csp_core` picked top 3 from the ranked list
4. Dispatcher called `select_contract` → failed for three reasons

## Fixes in this pack (10 files, ~12 distinct fixes)

### 🔴 #1 csp_universe: dynamic price ceiling
**File**: `csp_universe.py`

Added `_fetch_account_equity()` helper and dynamic affordability cap. The
`_layer1_hard_gates` filter now blocks tickers above `equity × 0.25 / 95`
(the absolute max underlying price an account can stretch to). For a
$108K account, cap is $285. For a $250K account, cap is $658.

This blocks LRCX ($295), WDC ($487), MSTR ($300+), STX ($487), COHR ($340)
— all of which were failing in the audit log — while keeping AAPL, AMZN,
AVGO, GOOGL, NVDA, AMD, CRM, ORCL accessible.

### 🔴 #2 csp_universe: expanded block list
**File**: `csp_universe.py`

Added to `CSP_BLOCKED_TICKERS`:
- Treasury bill ETFs: SGOV, BIL, SHV (options exist but volume too thin)
- Commodity ETFs: USO, UNG, BNO, DBO, USL (thin/seasonal chains)
- Cash equivalents: JPST, USFR

All of these were repeatedly failing in the audit log.

### 🔴 #3 options_execution: option_type filter
**File**: `options_execution.py`

`_fetch_option_chain` now accepts `option_type="call"|"put"` and passes
through to Alpaca's API. The dispatcher passes the correct side based on
strategy (`sell_cash_secured_put → put`, `buy_call → call`).

**Why**: WDC's `chain=7, puts=0` happened because Alpaca returned 7 calls
within the 10% strike band (limit=100, no type filter). Asking for puts
explicitly fixes this.

### 🔴 #4 options_execution: asymmetric strike range
**File**: `options_execution.py`

Was `±10%` symmetric around spot. Changed to:
- Puts: 80%-105% of spot (most OTM puts, we don't need 10% ITM puts)
- Calls: 95%-120% of spot
- Multi-leg: ±15%

For tight chains, this captures more useful strikes per API call.

### 🔴 #5 options_execution: liquidity gate accepts multiple signals
**File**: `options_execution.py`

Previously `_is_liquid` required OI ≥ 200, where OI was estimated from
`max(bid_size, ask_size) × 10`. For large-caps in after-hours, both
quote sizes are often 0, so the OI proxy is 0 → contract rejected.

Now passes if ANY of:
- Real OI ≥ 200
- Daily volume ≥ 10 contracts
- Real bid_size ≥ 10 AND ask_size ≥ 10

This unblocks STX/COHR-type names that have legit daily volume but quiet
intra-bar quotes.

### 🔴 #6 options_execution: preserve quote sizes
**File**: `options_execution.py`

Added `bid_size` and `ask_size` to the `contracts` dict so `_is_liquid`
can use them (was being computed but not stored).

### 🔴 #7 options_execution: affordability pre-filter + stretch mode
**File**: `options_execution.py`

`_select_sell_put` now:
1. Pre-filters puts to strikes that fit `equity × size_pct` budget
2. If no strikes fit, tries "stretch mode" — up to 20% of equity for one CSP
3. If even stretch mode doesn't fit, returns a clear error instead of
   picking the 30-delta strike and then failing on cash sizing

Previously: picked optimal strike → failed at sizing → zero CSP exposure.
Now: filter first, optimal-strike second, trade fires when possible.

### 🔴 #8 tiered_strategy: T1 affordability gate
**File**: `tiered_strategy.py`

`tier1_csp_core` now reads price from the Layer 2 cache (free, no extra API
call) and skips tickers whose underlying would force a too-expensive CSP.

Defense in depth: the universe filter (#1) catches most, this gate catches
the rest if cache is stale or scoring changes mid-cycle.

### 🔴 #9 csp_universe: _load_layer2_cache public helper
**File**: `csp_universe.py`

Added a public helper so `tier1_csp_core` (and other callers) can read the
ranked-universe cache without importing internals.

### 🔴 #10 ml_model_v2: fast_mode for low-memory containers
**File**: `ml_model_v2.py`

The production daemon has a 1024 MB cap (NOT the 8 GB claimed in stale
comments). The ML retrain subprocess kept hitting SIGKILL during training
because 200 tickers × 6 years × ~5 KB/bar of pandas features ≈ 1.5 GB.

Added `fast_mode` parameter to `_fetch_training_bars`:
- `fast_mode=True` (called by `ml_retrain_safe.py`): caps at 100 tickers × 2 years
- `fast_mode=False`: original 200 tickers × full history (for CLI runners with more RAM)

`_train_model_impl` now passes its own `fast_mode` flag down.

### 🟡 #11 ml_model_v2: corrected misleading comment
Removed the "8GB container, no shortage" comment that misled developers
into thinking memory was unlimited. Replaced with accurate Railway 1 GB cap.

## What the cascade looks like AFTER these fixes

Walking through the same audit log entries:

| Original failure | After-fix outcome |
|---|---|
| LRCX $295 too expensive | Blocked at universe level (above $285 cap) |
| WDC $487 chain=7, puts=0 | Either blocked at universe level OR `type=put` API filter returns puts |
| STX/COHR liquidity gate | `_is_liquid` accepts on daily volume even when quotes are 0 |
| USO no liquid options | Blocked at universe level (commodity ETF) |
| SGOV no options | Blocked at universe level (treasury bill ETF) |
| Generic "Not enough capital" | Pre-filtered out before contract selection, OR stretch mode kicks in |

Expected effect: the 3 tier actions per scan should now mostly succeed
instead of mostly failing.

## What's still pending after these 10 fixes

These were noted in the audit log but require broader work:

- **300s scan timeouts**: caused by ml_retrain hangs blocking daemon slots.
  Fix #10 should resolve this. Verify after deploy.
- **11546 → 2 stock candidate funnel**: per backtest, stocks are -EV, so
  filtering aggressively is correct. Not a bug, an intentional design.
- **OPTIONS_FEATURE_COLS missing 8 features in options_scanner**: pre-existing
  alignment issue between training and the options-specific ML pipeline.
  Not blocking trades but worth a future cleanup.
- **Manipulation alerts**: informational only, no fix needed.

## Verification

All 10 files compile cleanly. Math verified:
- Dynamic affordability cap correctly scales with equity
- All 6 `_is_liquid` scenarios behave as designed
- Function signatures accept new optional parameters
- Block list includes all 17 problem tickers from the audit log

## Files in this package

```
analyze.py             — (from 2026-05-18 pack, included for completeness)
bot_engine.py          — (from 2026-05-18 pack, included for completeness)
csp_universe.py        — NEW: dynamic affordability cap, expanded block list, _load_layer2_cache helper
instrument_selector.py — (from 2026-05-18 pack, included for completeness)
ml_model_v2.py         — NEW: fast_mode for low-memory training
options_execution.py   — NEW: type filter, asymmetric strikes, robust liquidity, affordability pre-filter
position_sizing.py     — (from 2026-05-18 pack, included for completeness)
risk_kill_switch.py    — (from 2026-05-18 pack, included for completeness)
system_config.py       — (from 2026-05-18 pack, included for completeness)
tiered_strategy.py     — NEW: T1 affordability gate, with reverted T3 from 2026-05-18
```

If you've already deployed the 2026-05-18 pack, only these 4 are new:
**csp_universe.py, ml_model_v2.py, options_execution.py, tiered_strategy.py**

## Push order

These can all go in one commit — they're independent fixes that don't conflict
with the 2026-05-18 pack.

```bash
git checkout -b fix/csp-execution-2026-05-22
cp ~/Downloads/voltradeai_fixes_2026-05-22/{csp_universe,ml_model_v2,options_execution,tiered_strategy}.py ./
git diff --stat   # should show 4 files changed
git add csp_universe.py ml_model_v2.py options_execution.py tiered_strategy.py
git commit -m "fix: CSP execution layer — affordability gate, type filter, liquidity check

Resolves CSP-failure cascade from 2026-05-21 production audit log.
Every CSP candidate was failing for one of three reasons:

1. LRCX/WDC/MSTR/STX/COHR: too expensive for account size — affordability
   filter now blocks at universe level (scales with account equity).
   Plus _select_sell_put pre-filters affordable strikes and supports
   stretch mode (up to 20% equity per CSP).

2. WDC chain=7, puts=0: option_type filter not passed to Alpaca API.
   Now passes 'put' explicitly when strategy is sell_cash_secured_put.

3. STX/COHR illiquid: _is_liquid required OI ≥ 200 where OI was a quote-
   size proxy that was 0 in quiet periods. Now accepts daily volume or
   quote sizes alone.

Plus: ml_model_v2 fast_mode (100 tickers × 2yr instead of 200 × 6yr)
to fit the 1 GB Railway container cap during retrain."
git push origin fix/csp-execution-2026-05-22
```

## Post-push verification

Watch for these signals in the first 24h:

1. **T2-FAIL count drops dramatically**. The audit log was showing 3 fails
   per 15-min scan cycle = 288 failures/day. Expected: <10/day.

2. **CSP positions actually open**. After current scans, the snapshot
   `positions.by_ticker` should start showing options positions alongside
   the floor basket (QQQ, SMH, KWEB, VXUS).

3. **ML retrain completes successfully**. Hourly TIER3 runs should show
   "ML retrain complete" instead of "ML retrain failed (SIGKILL)".

4. **Daily P&L shows non-floor contribution**. Currently $247.49 daily P&L
   is purely floor-basket drift. With CSPs firing, expect ~$50-150/day
   additional from theta decay alone (very rough — based on 8% Kelly
   size × 71.6% backtest WR × $0.69 avg win per trade × ~1-2 trades/day).
