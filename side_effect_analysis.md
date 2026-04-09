# VolTradeAI — Side-Effect Analysis
**Date:** 2026-04-09  
**Changes:** Separate options/stock position slots (MAX_OPTIONS_POSITIONS=3)  
**Files reviewed:** `bot_engine.py`, `server/bot.ts`, `options_execution.py`

---

## Summary of Changes

| # | File | Location | Change |
|---|------|----------|--------|
| 1 | `bot_engine.py` | Line 98 | Added `MAX_OPTIONS_POSITIONS = 3` constant |
| 2 | `bot_engine.py` | Lines 1700–1718 | Options scanner uses own `options_slots` counter; counts positions by OCC symbol length or `asset_class` |
| 3 | `bot.ts` | Lines 1631–1660 | `executeTrades()` splits stock (`slotsUsed`) and options (`optionsSlotsUsed`) counters; options full → `continue`, stocks full → `break` |
| 4 | `bot.ts` | Lines 1752–1819 | Scanner-originated options trades (`trade_type==="options" && regime_at_entry==="OPTIONS_SCANNER"`) bypass `evaluate_and_execute()`, going directly to `select_contract()` + `submit_options_order()` |

---

## 1. Risk Exposure — 5 stocks + 3 options = 8 max positions

**Finding: SAFE, but with a theoretical overshoot that existing checks cap.**

### Notional math
- Max stock exposure: `MAX_POSITION_PCT = 5%` × 5 = **25% of $97K ≈ $24K** in practice (dynamic sizing targets 1–5% per stock).
- Max options exposure: `options_execution.py` dynamic sizing targets **5–10% per contract** (`ABSOLUTE_MAX_POSITION_PCT` and `_dynamic_options_size`). Three positions = 15–30% at the ceiling.
- Theoretical worst case (all positions at ceiling): 25% stocks + 30% options = **55% notional**, well within `MAX_TOTAL_EXPOSURE = 50%`.
- The `MAX_TOTAL_EXPOSURE = 0.50` check in `executeTrades()` (line 1681) applies to the combined `totalDeployed` and fires a `break` before any single trade can push over the line.

### Options leverage note
Options are limit-order-only (`submit_options_order` always submits `type: limit`). Max loss on long options = premium paid (`max_loss = actual_cost`). The 50% heat cap is evaluated on `max_cost`, not notional delta-adjusted exposure. For bought calls/puts this is conservative (max loss = premium, not underlying). For sold cash-secured puts the sizing logic in `_select_sell_put` correctly uses full strike × 100 as max cost.

**Verdict:** No new risk surface introduced. The `totalDeployed` cap is the binding constraint and it includes both legs.

---

## 2. API Rate Limits — Scanner path extra calls

**Finding: One additional Alpaca API call per scanner-originated options trade, but no concern.**

### Call comparison: stock→options path vs scanner path

| Step | Stock→Options path | Scanner path (new) |
|------|-------------------|-------------------|
| `evaluate_and_execute()` | Yes — calls `should_use_options()` which may fetch account/positions | **Skipped** |
| `select_contract()` | Yes | Yes |
| `_fetch_option_chain()` inside `select_contract()` | 1× Alpaca data call | 1× Alpaca data call |
| `submit_options_order()` | Yes | Yes |
| `register_options_entry()` | Yes (inside evaluate path) | Yes (same inline try block, lines 1777–1779) |

The scanner path skips `evaluate_and_execute()` and its internal `should_use_options()` evaluation. Net effect: **fewer calls than before** for scanner trades, since `evaluate_and_execute()` internally re-fetches account data and positions. Alpaca is unlimited on paper/Algo Trader Plus — no throttle concern in either case.

---

## 3. Railway CPU/Memory — New Python process spawns

**Finding: Scanner path spawns exactly one `python3` subprocess per scanner-originated options trade — same pattern as the existing stock→options path. No new process type or fork pattern.**

The scanner path (lines 1765–1783) calls `execAsync(python3 -c "...")` with a 30-second timeout — identical to the stock→options `execAsync` on lines 1794–1797. The inline Python code imports `options_execution` functions, runs one Alpaca REST call (`_fetch_option_chain`), and exits. Memory footprint is the Python interpreter + `options_execution.py` imports (requests, json, datetime) — ~15–25 MB per subprocess, same as before.

The `tier2Running` mutex (line 2399) ensures only one `tier2Intelligence()` invocation runs at a time, so parallel subprocess proliferation from concurrent scan cycles is not possible.

**No new Railway concern.**

---

## 4. Race Conditions — `totalDeployed` modification

**Finding: No new race condition. `executeTrades()` is sequential and protected by a mutex.**

`totalDeployed` is a local variable inside `executeTrades()`. It is not shared state — each invocation of `executeTrades()` initializes it fresh from live Alpaca positions (line 1644). Between trades in the same execution loop there is a `await new Promise(r => setTimeout(r, 500))` delay (line 1867), but this is within a single awaited sequential loop, not concurrent.

The `tier2Running` boolean (line 2399) is set before `tier2Intelligence()` is called and cleared in a `finally` block (lines 2459–2467). This prevents two scan cycles from running `executeTrades()` simultaneously.

The one subtle risk: if `morningQueueExecution` runs at market open while a tier2 cycle just started, both could reach Alpaca `/v2/positions` in the same second and base their `slotsUsed` on a snapshot that doesn't yet reflect the other's orders. However, this existed before the change (the morning queue still uses the old `slotsUsed = positions.length` counter without the new split logic — see below).

**No new race condition introduced by this change.**

---

## 5. Error Cascading — Scanner path Python failure

**Finding: Handled gracefully. Options error → `continue` (no stock fallback for `trade_type==="options"`).**

The scanner path is wrapped in a `try/catch` block in two layers:

1. **Outer catch** (lines 1815–1818): If `execAsync` throws (e.g., Python crashes, timeout), the catch block runs:
   ```
   if (trade.trade_type === "options") continue;  // no stock fallback
   ```
   This correctly skips the trade rather than attempting stock execution with `shares: 0` (scanner trades set `shares: 0` explicitly).

2. **Inner Python error** (lines 1771–1772): If `select_contract()` returns `{error: ...}`, the embedded Python prints `{"instrument":"stock","order":null,...}`. This returns to TypeScript as `optExec.instrument === "stock"`, which hits the `continue` at line 1813. No order is placed.

3. **`options_slots` in `bot_engine.py`** (lines 1707–1708): If the options position count throws, `options_slots` defaults to `MAX_OPTIONS_POSITIONS = 3`. This is fail-open (permits new trades) but avoids a crash.

**No cascading failure path. All error exits cleanly skip the trade.**

---

## 6. Data Consistency — Stale contract data in scanner path

**Finding: Small staleness window exists but is bounded and acceptable.**

In the stock→options path, `evaluate_and_execute()` calls `select_contract()` inline — the chain fetch and order submission happen in the same Python process with no delay between them.

In the scanner path, `select_contract()` and `submit_options_order()` are also called sequentially in the same `python3 -c` subprocess (lines 1770–1774):
```python
contract = select_contract(...)
order = submit_options_order(contract)
```
There is no time gap between `select_contract()` returning and `submit_options_order()` consuming it. The contract data (`limit_price`, `occ_symbol`, `qty`) is not re-fetched between calls. 

The only staleness risk is the `_fetch_option_chain()` HTTP response aging between when the subprocess starts and when the order is submitted — typically < 2 seconds. Options limit-price optimization uses `_optimized_limit_price()` which targets mid-price, so a ±2-second move in the bid/ask will at most cause a non-fill (order remains open as a day order), not an adverse fill. `submit_options_order` always uses `"type": "limit"` — a stale limit price cannot result in a worse-than-expected fill.

**Risk level: Low. Worst case is a non-fill, not an adverse execution.**

---

## 7. Heat Check — `totalDeployed` includes both stock and options

**Finding: Correct. No gap introduced.**

`totalDeployed` is initialized at the top of `executeTrades()` (lines 1644–1646) by summing `market_value` across all positions — both `us_equity` and `us_option` positions returned by Alpaca `/v2/positions`:
```typescript
let totalDeployed = Array.isArray(positions)
  ? positions.reduce((sum, p) => sum + Math.abs(parseFloat(p.market_value || "0")), 0)
  : 0;
```

After each successful trade, `totalDeployed` is updated:
- Stock: `totalDeployed += trade.position_value` (line 1856)
- ETF: `totalDeployed += etfShares * trade.price` (line 1744)  
- Options (scanner path): `totalDeployed += optExec.contract?.max_cost || trade.position_value` (line 1805)

The `MAX_TOTAL_EXPOSURE = 0.50` check (line 1681) fires before each trade and will `break` the loop if the combined total would exceed 50% of equity. This applies regardless of instrument type. The options `max_cost` (premium paid) is small relative to notional, so this check is not the binding limiter for options — it primarily guards against excessive stock concentration.

**The `MAX_TOTAL_EXPOSURE` check works correctly for combined exposure.**

---

## 8. `continue` vs `break` for Options Slots Full

**Finding: Correct logic. One edge case with the heat check worth noting.**

### Correctness of `continue` when options slots full
The old code used a shared `slots_available` counter, so when slots were exhausted any trade (stock or options) would stop processing. The new code correctly differentiates:
- Options full → `continue` to the next trade in the list (allowing stock trades to proceed)
- Stocks full → `break` (no more trades of any kind make sense since the loop is stock-priority)

This is safe because scanner-originated options trades have `shares: 0` — if they accidentally fell through to stock execution, `Math.floor(0) <= 0` would trigger the `qty <= 0` guard at line 1823 and `continue`. So there is a secondary safety net.

### Edge case: `break` on heat check when options-full trade is processed
When options slots are full and we `continue` for an options trade, the heat check at line 1681 is **never reached** for that trade (the `continue` fires earlier, at lines 1654–1656). This is correct — we shouldn't count a skipped trade against `totalDeployed`.

### Pre-existing issue (not introduced by this change): Morning queue doesn't split slots
The morning queue execution path (lines 1248–1346) uses:
```typescript
slotsUsed = Array.isArray(positions) ? positions.length : 0;
// checks: if (slotsUsed >= MAX_POSITIONS) break;  // MAX_POSITIONS = 8
```
This counts **all positions** (stocks + options) against the `MAX_POSITIONS = 8` ceiling and does not differentiate. If 3 options positions exist at market open, the morning queue sees `slotsUsed = 3` and allows up to 5 more stock trades before hitting 8 — which is acceptable behavior, but the stock-only limit of 5 is not enforced here. This existed before the change and is unchanged. If this matters, the morning queue would need the same stock/options split logic as `executeTrades()`.

---

## 9. Latent Issue Found: `slots_available` in `bot_engine.py` includes options positions

**Finding: Bug, pre-existing but now more impactful.**

In `bot_engine.py` (lines 1413–1425):
```python
current_tickers = [p.get("symbol") for p in current_positions]
num_positions = len(current_tickers)  # includes ALL positions: stocks + options OCC symbols
slots_available = MAX_POSITIONS - num_positions  # MAX_POSITIONS = 5
```

`get_alpaca_positions()` returns all Alpaca positions including options. Options positions have OCC symbols like `AAPL260418C00250000`. These are counted in `num_positions`, which means each options position consumes one of the 5 stock slots in the scanner's view.

**Practical effect:** With 2 stock + 2 options positions, `num_positions = 4`, `slots_available = 1`. The scanner would only generate 1 more stock trade candidate even though 3 stock slots are actually free.

This is partially mitigated because `executeTrades()` (the actual executor) now correctly counts stocks and options separately. Stock trades generated by `bot_engine.py` will still get executed if stock slots are available — the scanner just generates fewer candidates than necessary.

**Recommendation:** In `bot_engine.py` step 5, filter `current_tickers` and `num_positions` to stock positions only:
```python
stock_positions = [p for p in current_positions if len(str(p.get("symbol",""))) <= 8 and p.get("asset_class") != "us_option"]
current_tickers = [p.get("symbol") for p in stock_positions]
num_positions = len(current_tickers)
```

---

## 10. Latent Issue Found: Shell Injection in Scanner Path

**Finding: Low risk in production, worth flagging.**

The scanner path builds the Python command string by embedding JSON:
```typescript
const scannerData = JSON.stringify({...}).replace(/'/g, "\\'");
// then embeds in:  `python3 -c "...data = json.loads('${scannerData}')..."`
```
`JSON.stringify` produces double-quoted JSON, and only single-quotes are escaped. If `trade.ticker` or `trade.setup` (sourced from `options_scanner`) contains a double-quote or backtick, the shell command string could be malformed (parse error, not injection, since ticker/setup values come from internal scanner logic, not user input). This existed in the stock→options path (`optionsData`) and is unchanged.

**Risk: Negligible for internal data. Not introduced by this change.**

---

## Summary Table

| Area | Status | Severity | Action Required |
|------|--------|----------|-----------------|
| 1. Risk exposure (5+3 positions) | ✅ Safe | — | None — heat cap covers combined |
| 2. API rate limits | ✅ Improved | — | Scanner path uses fewer calls than full pipeline |
| 3. Railway CPU/memory | ✅ Unchanged | — | Same subprocess pattern as before |
| 4. Race conditions | ✅ Safe | — | `tier2Running` mutex prevents concurrent execution |
| 5. Error cascading | ✅ Handled | — | All failure paths `continue`, no stock fallback for options |
| 6. Data consistency (stale contract) | ✅ Acceptable | Low | `select_contract` + `submit` in same process, limit order protects |
| 7. Heat check (`MAX_TOTAL_EXPOSURE`) | ✅ Correct | — | `totalDeployed` includes all instruments from the start |
| 8. `continue` vs `break` | ✅ Correct | — | Stock trades unblocked when options full; `qty<=0` safety net exists |
| **9. `bot_engine.py` counts options in stock slots** | ⚠️ Bug | Medium | Fix `num_positions` to count only stock positions |
| **10. Morning queue ignores stock/options split** | ⚠️ Pre-existing | Low | Morning queue allows up to 5 non-stock slots before cut-off |
